"""Refresh the four scripted demo traces offline after the Phase 1 model changes.

Each trace is replayed through the real ``run_retention_agent`` loop with a
``ScriptedMockClient`` that repeats the recorded tool-call sequence. The tools,
policy gate and formatter run again, so the churn probability, SHAP drivers,
cohort figures and gate results come from the current model and data. The
reasoning text stays a labelled scripted fixture. Numbers quoted inside that
text are rewritten from the new tool results. No provider request is made.

Usage:
    python scripts/regenerate_scripted_traces.py --check
    python scripts/regenerate_scripted_traces.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import MODEL_NAME, ScriptedMockClient, run_retention_agent  # noqa: E402
from agent.tools import load_phase1_runtime  # noqa: E402


DEMO_DIR = PROJECT_ROOT / "demo_traces"
PROBABILITY_TEXT = re.compile(r"Churn probability: [0-9.]+%")
COHORT_TEXT = re.compile(r"Cohort churn rate: [0-9.]+%")


def _turns(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group recorded events back into the assistant messages that produced them."""

    turns: list[dict[str, Any]] = []
    previous = None
    for event in trace:
        kind = event["type"]
        if kind == "model_thought":
            turns.append({"text": event["content"]["text"], "calls": []})
        elif kind == "tool_call":
            if previous not in {"model_thought", "tool_call"}:
                turns.append({"text": None, "calls": []})
            turns[-1]["calls"].append(event["content"])
        previous = kind
    if turns and turns[-1]["calls"]:
        # The recorded run ended with a stop message that carried no text.
        turns.append({"text": None, "calls": []})
    return turns


def _response(text: str | None, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {
                    "content": text,
                    "tool_calls": [
                        {
                            "id": call["tool_use_id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["input"]),
                            },
                        }
                        for call in calls
                    ],
                },
            }
        ]
    }


def _drivers(runtime: Any, customer: dict[str, Any]) -> list[dict[str, Any]]:
    """Top five SHAP drivers, computed the same way as app.py's case review."""

    frame = runtime.prepare_feature_vector(customer)
    values = np.asarray(shap.TreeExplainer(runtime.model)(frame).values[0], dtype=float)
    names = list(runtime.feature_names)
    order = np.argsort(np.abs(values))[::-1][:5]
    return [
        {
            "feature": names[index],
            "value": customer["profile"][names[index]],
            "shap_value": float(values[index]),
            "direction": "increases_churn" if values[index] >= 0 else "decreases_churn",
        }
        for index in order
    ]


def _rewrite_numbers(text: str | None, probability: float, cohort_rate: float | None) -> str | None:
    if text is None:
        return None
    text = PROBABILITY_TEXT.sub(f"Churn probability: {probability:.2%}", text)
    if cohort_rate is not None:
        text = COHORT_TEXT.sub(f"Cohort churn rate: {cohort_rate:.2%}", text)
    return text


def regenerate(demo: dict[str, Any], runtime: Any) -> dict[str, Any]:
    customer = {
        key: value
        for key, value in demo["customer"].items()
        if key not in {"churn_probability", "churn_drivers", "phase1_prediction"}
    }
    probability = runtime.predict_churn_risk(customer)["churn_probability"]
    turns = _turns(demo["trace"])

    def replay(rate: float | None) -> dict[str, Any]:
        responses = [
            _response(_rewrite_numbers(turn["text"], probability, rate), turn["calls"])
            for turn in turns
        ]
        clock_values = iter(event["timestamp"] for event in demo["trace"])
        return run_retention_agent(
            {**customer, "churn_probability": probability},
            client=ScriptedMockClient(responses),
            phase1_runtime=runtime,
            clock=lambda: next(clock_values),
        )

    # The first pass finds the new cohort rate; the second quotes it in the text.
    cohort_rate = next(
        event["content"]["result"]["churn_rate"]
        for event in replay(None)["trace"]
        if event["type"] == "tool_result"
        and event["content"].get("name") == "segment_comparison"
    )
    result = replay(cohort_rate)

    refreshed_customer = dict(result["customer"])
    refreshed_customer.pop("phase1_prediction", None)
    ordered = {}
    for key in demo["customer"]:
        if key == "churn_drivers":
            ordered[key] = _drivers(runtime, refreshed_customer)
        elif key == "churn_probability":
            ordered[key] = probability
        else:
            ordered[key] = refreshed_customer[key]
    return {
        **demo,
        # The recorded runtime model id follows the configured Groq model.
        "recording": {**demo["recording"], "model": MODEL_NAME},
        "customer": ordered,
        "trace": result["trace"],
        "recommendation": result["recommendation"],
    }


def _summary(demo: dict[str, Any]) -> str:
    rec = demo["recommendation"]
    return (
        f"p={demo['customer']['churn_probability']:.4f} "
        f"verdict={rec['checker_verdict']} action={rec['action']} "
        f"flags={rec['regulatory_flags']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Show outcome changes only.")
    mode.add_argument("--write", action="store_true", help="Rewrite demo_traces/*.json.")
    args = parser.parse_args()

    runtime = load_phase1_runtime()
    changed_outcomes = 0
    for path in sorted(DEMO_DIR.glob("*.json")):
        old = json.loads(path.read_text(encoding="utf-8"))
        new = regenerate(old, runtime)
        before, after = _summary(old), _summary(new)
        outcome_changed = (
            old["recommendation"]["action"] != new["recommendation"]["action"]
            or old["recommendation"]["checker_verdict"]
            != new["recommendation"]["checker_verdict"]
            or old["recommendation"]["regulatory_flags"]
            != new["recommendation"]["regulatory_flags"]
        )
        changed_outcomes += outcome_changed
        print(f"{old['demo_id']}")
        print(f"  before: {before}")
        print(f"  after:  {after}{'  OUTCOME CHANGED' if outcome_changed else ''}")
        if args.write:
            path.write_text(
                json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    print(f"\n{changed_outcomes} scenario outcome(s) changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
