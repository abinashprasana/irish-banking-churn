"""Evaluate the governed retention agent without spending paid API credit.

Dry run (default or ``--dry-run``):
    Replays the four recorded traces and validates tool IDs, output schema, and
    policy-verdict consistency. It makes zero Groq requests.

Live (``--live``):
    Runs selected recorded customer scenarios through Groq's free-tier
    ``llama-3.3-70b-versatile`` endpoint. It requires ``GROQ_API_KEY`` and is
    protected by the same in-memory daily/request limits as the application.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import (  # noqa: E402
    PHASE1_FEATURE_SCHEMA,
    Recommendation,
    predict_customer_churn_risk,
)


DEMO_DIR = PROJECT_ROOT / "demo_traces"
REQUIRED_TOOLS = frozenset(
    [
        "product_lookup",
        "segment_comparison",
        "regulatory_constraint_checker",
        "recommendation_formatter",
    ]
)
REQUIRED_KEYS = frozenset(
    [
        "schema_version",
        "demo_id",
        "title",
        "recording",
        "customer",
        "trace",
        "recommendation",
    ]
)


def _tool_calls(trace: list[dict]) -> list[str]:
    return [
        event["content"]["name"]
        for event in trace
        if event["type"] == "tool_call"
    ]


def _check_scenario(demo: dict, scenario_id: str) -> list[str]:
    failures: list[str] = []
    missing = REQUIRED_KEYS - set(demo)
    if missing:
        failures.append(f"{scenario_id}: missing keys {sorted(missing)}")
        return failures

    trace = demo["trace"]
    steps = [event["step"] for event in trace]
    if steps != list(range(1, len(steps) + 1)):
        failures.append(f"{scenario_id}: trace steps not sequential: {steps}")

    calls = _tool_calls(trace)
    if set(calls) != REQUIRED_TOOLS:
        failures.append(
            f"{scenario_id}: wrong tool set - got {sorted(set(calls))}, "
            f"expected {sorted(REQUIRED_TOOLS)}"
        )
    if (
        "regulatory_constraint_checker" in calls
        and "recommendation_formatter" in calls
        and calls.index("regulatory_constraint_checker")
        >= calls.index("recommendation_formatter")
    ):
        failures.append(f"{scenario_id}: formatter called before checker")

    call_ids = {
        event["content"]["tool_use_id"]: event["content"]["name"]
        for event in trace
        if event["type"] == "tool_call"
    }
    result_ids = {
        event["content"]["tool_use_id"]: event["content"]["name"]
        for event in trace
        if event["type"] == "tool_result"
    }
    if call_ids != result_ids:
        failures.append(f"{scenario_id}: tool_call / tool_result ID mismatch")

    recording = demo["recording"]
    if recording.get("phase1_runtime_capture") is not True:
        failures.append(f"{scenario_id}: missing real Phase 1 runtime capture marker")
    try:
        phase1_prediction = predict_customer_churn_risk(demo["customer"])
    except Exception as exc:
        failures.append(
            f"{scenario_id}: Phase 1 runtime prediction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return failures
    recorded_probability = demo["customer"].get("churn_probability")
    if not isinstance(recorded_probability, (int, float)) or not math.isclose(
        recorded_probability,
        phase1_prediction["churn_probability"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        failures.append(
            f"{scenario_id}: recorded churn probability does not match the "
            "trained Phase 1 model"
        )
    if phase1_prediction["feature_columns"] != list(PHASE1_FEATURE_SCHEMA):
        failures.append(f"{scenario_id}: Phase 1 feature order is not exact")
    trace_predictions = [
        event["content"]["result"]["target_phase1_prediction"]
        for event in trace
        if event["type"] == "tool_result"
        and event["content"].get("name") == "segment_comparison"
        and "target_phase1_prediction" in event["content"].get("result", {})
    ]
    if len(trace_predictions) != 1 or not math.isclose(
        trace_predictions[0].get("churn_probability", -1.0)
        if trace_predictions
        else -1.0,
        phase1_prediction["churn_probability"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        failures.append(
            f"{scenario_id}: trace lacks the matching real Phase 1 prediction"
        )

    recommendation = demo["recommendation"]
    try:
        Recommendation.model_validate(recommendation)
    except ValidationError as exc:
        failures.append(f"{scenario_id}: recommendation schema failed: {exc}")
        return failures
    if (
        recommendation["checker_verdict"] == "blocked"
        and recommendation["action"] != "no_recommendation"
    ):
        failures.append(
            f"{scenario_id}: blocked outcome must have action='no_recommendation'"
        )
    return failures


def _print_result(scenario_id: str, title: str, failures: list[str]) -> None:
    status = "PASS" if not failures else "FAIL"
    print(f"  [{status}] {scenario_id}: {title}")
    for failure in failures:
        print(f"       - {failure}")


def _selected_paths(scenario: str | None = None) -> list[Path]:
    paths = sorted(DEMO_DIR.glob("*.json"))
    if scenario is None:
        return paths
    selected = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("demo_id") == scenario:
            selected.append(path)
    return selected


def dry_run(scenario: str | None = None) -> int:
    paths = _selected_paths(scenario)
    if not paths:
        print("ERROR: no matching recorded traces found in demo_traces/.")
        return 1

    print(f"\nDry-run eval - {len(paths)} scenario(s) - zero Groq requests\n")
    all_failures: list[str] = []
    blocked_count = 0
    for path in paths:
        demo = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = demo.get("demo_id", path.name)
        failures = _check_scenario(demo, scenario_id)
        all_failures.extend(failures)
        _print_result(scenario_id, demo.get("title", ""), failures)
        if demo.get("recommendation", {}).get("checker_verdict") == "blocked":
            blocked_count += 1

    if scenario is None:
        print(f"\n  Blocked outcomes: {blocked_count}/{len(paths)} (must be >= 2)")
        if blocked_count < 2:
            all_failures.append(
                f"too few blocked outcomes: {blocked_count} (need >= 2)"
            )

    if all_failures:
        print(f"\nResult: FAIL - {len(all_failures)} issue(s)\n")
        return 1
    print(f"\nResult: PASS - all {len(paths)} scenarios passed dry-run\n")
    return 0


def live_run(scenario: str | None = None) -> int:
    from agent.loop import (
        MAX_LIVE_API_CALLS,
        MAX_TOKENS,
        MODEL_NAME,
        create_live_client,
        resolve_groq_api_key,
        run_retention_agent,
    )

    api_key = resolve_groq_api_key()
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set. No Groq request was made.")
        return 1

    paths = _selected_paths(scenario)
    if not paths:
        print("ERROR: no matching recorded traces found in demo_traces/.")
        return 1

    print(f"\nLive Groq eval - {len(paths)} scenario(s)")
    print(f"Model: {MODEL_NAME}")
    print(f"Max completion tokens per request: {MAX_TOKENS}")
    print(f"Model-call cap per run: {MAX_LIVE_API_CALLS}\n")

    all_failures: list[str] = []
    for path in paths:
        demo = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = demo.get("demo_id", path.name)
        title = demo.get("title", "")
        print(f"\n  Running {scenario_id}: {title} ...")
        try:
            client = create_live_client(api_key=api_key)
            result = run_retention_agent(demo["customer"], client=client)
            live_demo = {
                "schema_version": demo.get("schema_version", "1.0"),
                "demo_id": scenario_id,
                "title": title,
                "recording": {
                    "mode": "eval_live_groq",
                    "real_api_calls": client.chat.completions.request_count,
                    "model": MODEL_NAME,
                    "max_tokens_per_call": MAX_TOKENS,
                    "call_cap": MAX_LIVE_API_CALLS,
                    "phase1_runtime_capture": True,
                    "phase1_model_artifact": result["phase1_prediction"][
                        "model_artifact"
                    ],
                    "phase1_prediction_method": result["phase1_prediction"][
                        "prediction_method"
                    ],
                },
                "customer": result["customer"],
                "trace": result["trace"],
                "recommendation": result["recommendation"],
            }
            failures = _check_scenario(live_demo, scenario_id)
            all_failures.extend(failures)
            _print_result(scenario_id, title, failures)
        except Exception as exc:
            message = (
                f"{scenario_id}: live run raised {type(exc).__name__}: {exc}"
            )
            all_failures.append(message)
            print(f"  [FAIL] {message}")

    if all_failures:
        print(f"\nResult: FAIL - {len(all_failures)} issue(s)\n")
        return 1
    print(f"\nResult: PASS - all {len(paths)} scenarios passed live eval\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate recorded traces without making a Groq request (default).",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Run through the Groq free tier and consume shared daily quota.",
    )
    parser.add_argument(
        "--scenario",
        help="Optional demo_id; omit to evaluate all four recorded scenarios.",
    )
    args = parser.parse_args()
    sys.exit(live_run(args.scenario) if args.live else dry_run(args.scenario))
