"""Build the sanitized, deterministic evidence bundle used by the web case study.

The exporter reads repository-owned sources only. It deliberately excludes model
thought text, raw tool payloads, encoded feature vectors, timestamps, and policy
fingerprints from the public bundle.

Usage:
    python scripts/export_case_study.py --write
    python scripts/export_case_study.py --check
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_rules import (  # noqa: E402
    HIGH_RISK_HUMAN_REVIEW_THRESHOLD,
    RULES,
)


DATA_PATH = PROJECT_ROOT / "data" / "irish_banking_churn.csv"
MODEL_CARD_PATH = PROJECT_ROOT / "model_card.md"
LOOP_PATH = PROJECT_ROOT / "agent" / "loop.py"
TRACE_DIR = PROJECT_ROOT / "demo_traces"
TEST_DIR = PROJECT_ROOT / "tests"
OUTPUT_PATH = PROJECT_ROOT / "web" / "src" / "data" / "evidence.generated.json"
BENCHMARK_PATH = PROJECT_ROOT / "results" / "benchmark_uci_bank_marketing.json"
RED_TEAM_DIR = PROJECT_ROOT / "redteam" / "results"

HARM_LABELS = {
    "H06": "Disclosure about another customer",
    "H07": "Unsupported regulatory or guarantee claim",
    "H08": "Action and justification disagree",
    "H11": "Thought text pushes a blocked action",
    "H12": "Offer does not fit the profile",
}
EVIDENCE_DATE = "2026-08-15"

PROJECT = {
    "name": "Atlantic Ledger",
    "descriptor": "Irish banking churn and governed retention intelligence",
    "repositoryUrl": "https://github.com/abinashprasana/irish-banking-churn",
    "labUrl": "https://abinashprasana-irish-banking-churn-app-aidovf.streamlit.app/",
}

DATA_SOURCES = [
    {
        "name": "Central Bank of Ireland account migration statistics",
        "url": "https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/account-migration-statistics",
        "usage": "Historical context for KBC Bank Ireland and Ulster Bank account closures.",
    },
    {
        "name": "CCPC switching research (phase 2)",
        "url": "https://www.ccpc.ie/about-us/advocacy-and-research/research/publication-details/ccpc-switching-research-%28phase-2%29",
        "usage": "A published switching-difficulty figure used as one synthetic-data assumption.",
    },
]

METRIC_IDS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "rocAuc",
    "averagePrecision",
)
METRIC_LABELS = (
    "Accuracy",
    "Precision",
    "Recall",
    "F1 score",
    "ROC-AUC",
    "Average precision",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.relative_to(PROJECT_ROOT)} must contain an object")
    return payload


def _runtime_model_id() -> str:
    tree = ast.parse(LOOP_PATH.read_text(encoding="utf-8"), filename=str(LOOP_PATH))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MODEL_NAME"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise ValueError("agent/loop.py does not define a literal MODEL_NAME")


def _dataset_evidence() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "churn" not in reader.fieldnames:
            raise ValueError("dataset must have a header and a churn target")
        rows = 0
        churned = 0
        for row in reader:
            rows += 1
            churned += int(row["churn"])

    features = [
        name for name in reader.fieldnames if name not in {"customer_id", "churn"}
    ]
    test_count = round(rows * 0.20)
    return {
        "recordCount": rows,
        "featureCount": len(features),
        "featureNames": features,
        "churnRate": round(churned / rows, 4),
        "trainCount": rows - test_count,
        "testCount": test_count,
        "synthetic": True,
        "sources": DATA_SOURCES,
        "disclaimer": (
            "All customer records and labels are synthetic. Published Irish sources "
            "provide context and one modelling assumption; they do not validate the "
            "generated population or model."
        ),
    }


def _parse_markdown_row(line: str) -> list[str]:
    return [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]


def _model_evidence(recordings: list[dict[str, Any]]) -> dict[str, Any]:
    text = MODEL_CARD_PATH.read_text(encoding="utf-8")
    benchmark_rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        cells = _parse_markdown_row(line) if line.startswith("|") else []
        if len(cells) != 7 or cells[0] not in {
            "XGBoost (Selected)",
            "Random Forest",
            "Logistic Regression",
        }:
            continue
        values = [float(cell) for cell in cells[1:]]
        benchmark_rows.append(
            {
                "model": cells[0],
                "selected": cells[0] == "XGBoost (Selected)",
                "metrics": [
                    {"id": metric_id, "label": label, "value": value}
                    for metric_id, label, value in zip(
                        METRIC_IDS, METRIC_LABELS, values, strict=True
                    )
                ],
            }
        )
    if len(benchmark_rows) != 3:
        raise ValueError("model_card.md must contain all three benchmark rows")

    feature_pattern = re.compile(
        r"^\d+\. `(?P<name>[^`]+)` \(Mean Absolute SHAP: (?P<value>\d+\.\d+)\)$",
        re.MULTILINE,
    )
    top_features = [
        {"rank": rank, "name": match.group("name"), "meanAbsoluteShap": float(match.group("value"))}
        for rank, match in enumerate(feature_pattern.finditer(text), start=1)
    ]
    if len(top_features) != 5:
        raise ValueError("model_card.md must contain exactly five ranked SHAP features")

    artifact_names = {recording.get("phase1_model_artifact") for recording in recordings}
    prediction_methods = {
        recording.get("phase1_prediction_method") for recording in recordings
    }
    if len(artifact_names) != 1 or None in artifact_names:
        raise ValueError("recorded traces disagree on the Phase 1 model artifact")
    if len(prediction_methods) != 1 or None in prediction_methods:
        raise ValueError("recorded traces disagree on the prediction method")

    selected = next(row for row in benchmark_rows if row["selected"])
    return {
        "selectedModel": "XGBoost binary classifier",
        "artifact": next(iter(artifact_names)),
        "predictionMethod": next(iter(prediction_methods)),
        "threshold": 0.50,
        "metrics": selected["metrics"],
        "benchmarks": benchmark_rows,
        "topFeatures": top_features,
        "explanationCaveat": (
            "SHAP values explain the fitted model's raw output on synthetic data. "
            "They are neither causal effects nor probability-point changes."
        ),
    }


def _benchmark_evidence() -> dict[str, Any]:
    source = _read_json(BENCHMARK_PATH)
    return {
        "dataset": source["dataset"],
        "citation": source["citation"],
        "sourceUrl": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "licence": source["licence"],
        "target": source["target"],
        "records": source["records"],
        "positiveRate": source["positive_rate"],
        "droppedColumns": source["dropped_columns"],
        "models": [
            {
                "model": row["model"],
                "rocAuc": row["roc_auc"],
                "averagePrecision": row["average_precision"],
                "f1": row["f1"],
            }
            for row in source["metrics"]
        ],
    }


def _proportion(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "successes": entry["successes"],
        "total": entry["total"],
        "interval": entry["wilson_95"],
    }


def _red_team_evidence() -> dict[str, Any]:
    runs = {gate: _read_json(RED_TEAM_DIR / f"offline_gate_{gate}_all.json") for gate in ("on", "off")}
    for gate, run in runs.items():
        if run["mode"] != "offline" or run["provider_requests"] != 0:
            raise ValueError(f"red team gate {gate} summary must be an offline, zero request run")
    attacks = [line for line in (PROJECT_ROOT / "redteam" / "attacks" / "attacks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    benign = [line for line in (PROJECT_ROOT / "redteam" / "attacks" / "benign.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in attacks]

    def gate_view(run: dict[str, Any]) -> dict[str, Any]:
        summary = run["summary"]
        return {
            "headline": _proportion(summary["headline"]["overall"]),
            "headlineByFamily": {
                family: _proportion(entry)
                for family, entry in summary["headline"]["by_family"].items()
            },
            "headlineOutcomes": [
                {"id": outcome["id"], "family": outcome["family"], "unsafe": outcome["success"]}
                for outcome in run["outcomes"]
                if outcome.get("headline") and not outcome.get("skipped")
            ],
            "operatorSurface": _proportion(summary["operator_surface"]["overall"]),
            "knownGaps": _proportion(summary["known_gap"]["overall"]),
            "benignBlocked": _proportion(summary["benign"]["false_block"]),
            "benignFlagged": _proportion(summary["benign"]["oracle_flagged"]),
        }

    gap_rows: dict[str, list[str]] = {}
    for row in runs["on"]["summary"]["known_gap"]["records"]:
        for harm in row["violates"]:
            gap_rows.setdefault(harm, []).append(row["id"])
    return {
        "attackCount": len(records),
        "familyCount": len({record["family"] for record in records}),
        "offlineAttackCount": sum("offline" in record["modes"] for record in records),
        "benignCount": len(benign),
        "includesDrafts": runs["on"]["includes_drafts"] or runs["off"]["includes_drafts"],
        "gateOn": gate_view(runs["on"]),
        "gateOff": gate_view(runs["off"]),
        "knownGapRows": [
            {"harmId": harm, "label": HARM_LABELS.get(harm, harm), "attackIds": ids}
            for harm, ids in sorted(gap_rows.items())
        ],
        "closedBypassIds": sorted(record["id"] for record in records if record["status"] == "closed_bypass"),
        "live": None,
    }


def _test_function_count() -> int:
    count = 0
    for path in sorted(TEST_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
    return count


def _result_by_call_id(trace: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        event["content"]["tool_use_id"]: event["content"]
        for event in trace
        if event.get("type") == "tool_result"
    }


def _tool_summary(name: str, result: dict[str, Any]) -> str:
    if name == "product_lookup":
        offers = result.get("offers", [])
        return f"Found {len(offers)} synthetic catalogue option{'s' if len(offers) != 1 else ''}."
    if name == "segment_comparison":
        return (
            f"Compared with a synthetic cohort of {result.get('cohort_size', 0):,} "
            f"customers at a {float(result.get('churn_rate', 0)):.1%} churn rate."
        )
    if name == "regulatory_constraint_checker":
        failed = result.get("failed_rule_ids", [])
        return (
            f"Blocked by {', '.join(failed)}."
            if failed
            else "All four prototype policy rules passed."
        )
    if name == "recommendation_formatter":
        return f"Structured the final {result.get('checker_verdict', 'unknown')} decision."
    return "Completed a deterministic project tool step."


def _scenario(demo: dict[str, Any], runtime_model_id: str) -> dict[str, Any]:
    recording = demo["recording"]
    if recording.get("real_api_calls") != 0:
        raise ValueError(f"{demo.get('demo_id')}: public scenarios must make zero API requests")
    if recording.get("reasoning_source") != "scripted_fixture":
        raise ValueError(f"{demo.get('demo_id')}: reasoning source must be explicit")
    if recording.get("model_output_captured") is not False:
        raise ValueError(f"{demo.get('demo_id')}: scripted text cannot claim model capture")
    if recording.get("model") != runtime_model_id:
        raise ValueError(f"{demo.get('demo_id')}: recorded runtime model metadata drifted")

    trace = demo["trace"]
    results = _result_by_call_id(trace)
    tool_steps = []
    for event in trace:
        if event.get("type") != "tool_call":
            continue
        content = event["content"]
        result_event = results.get(content["tool_use_id"])
        if result_event is None:
            raise ValueError(f"{demo.get('demo_id')}: tool call lacks a matching result")
        result = result_event.get("result", {})
        tool_steps.append(
            {
                "kind": "tool",
                "name": content["name"],
                "status": "error" if result_event.get("is_error") else "complete",
                "summary": _tool_summary(content["name"], result),
            }
        )

    gates = [event["content"] for event in trace if event.get("type") == "gate_check"]
    if not gates:
        raise ValueError(f"{demo.get('demo_id')}: trace has no policy gate evidence")
    gate = gates[-1]
    recommendation = demo["recommendation"]
    flags = recommendation.get("regulatory_flags", [])
    if recommendation["checker_verdict"] == "blocked":
        status = "blocked"
    elif any("human_review_required" in flag for flag in flags):
        status = "review_required"
    else:
        status = "approved"

    customer = demo["customer"]
    return {
        "id": demo["demo_id"],
        "title": demo["title"],
        "mode": "recorded",
        "recording": {
            "apiRequests": 0,
            "runtimeModelId": runtime_model_id,
            "modelOutputCaptured": False,
            "reasoningSource": "scripted_fixture",
            "label": "Recorded and verified replay · 0 API requests",
        },
        "customer": {
            "id": customer["customer_id"],
            "profile": customer["profile"],
            "churnProbability": customer["churn_probability"],
            "drivers": customer.get("churn_drivers", []),
            "heldProducts": customer.get("held_products", []),
            "governance": customer.get("governance", {}),
        },
        "toolSteps": tool_steps,
        "policy": {
            "actionId": gate["action_id"],
            "verdict": gate["checker_verdict"],
            "failedRuleIds": gate["failed_rule_ids"],
            "rules": [
                {
                    "id": rule["rule_id"],
                    "description": rule["description"],
                    "passed": rule["passed"],
                    "reason": rule["reason"],
                }
                for rule in gate["rule_results"]
            ],
        },
        "finalDecision": {
            "status": status,
            "action": recommendation["action"],
            "justification": recommendation["justification"],
            "confidence": recommendation["confidence"],
            "flags": flags,
            "verdict": recommendation["checker_verdict"],
        },
    }


def build_bundle() -> dict[str, Any]:
    demos = [_read_json(path) for path in sorted(TRACE_DIR.glob("*.json"))]
    if len(demos) != 4:
        raise ValueError(f"expected four recorded scenarios, found {len(demos)}")
    recordings = [demo["recording"] for demo in demos]
    runtime_model_id = _runtime_model_id()
    scenarios = [_scenario(demo, runtime_model_id) for demo in demos]
    test_count = _test_function_count()
    blocked_count = sum(
        scenario["finalDecision"]["status"] == "blocked" for scenario in scenarios
    )
    tools = sorted(
        {step["name"] for scenario in scenarios for step in scenario["toolSteps"]}
    )
    return {
        "schemaVersion": "1.0",
        "generatedAt": EVIDENCE_DATE,
        "generatedFrom": [
            "data/irish_banking_churn.csv",
            "models/xgboost_churn_model.pkl",
            "model_card.md",
            "agent/loop.py",
            "agent/policy_rules.py",
            "demo_traces/*.json",
            "tests/test_*.py",
            "results/benchmark_uci_bank_marketing.json",
            "redteam/results/offline_gate_on_all.json",
            "redteam/results/offline_gate_off_all.json",
        ],
        "evidence": {
            "project": PROJECT,
            "dataset": _dataset_evidence(),
            "model": _model_evidence(recordings),
            "agent": {
                "modelId": runtime_model_id,
                "tools": tools,
                "recordedMode": "zero_request_scripted_replay",
                "recordedReasoningSource": "scripted_fixture",
                "liveSmokeTest": "required_before_enabling_live_mode",
            },
            "governance": {
                "humanReviewThreshold": HIGH_RISK_HUMAN_REVIEW_THRESHOLD,
                "rules": [
                    {"id": rule.rule_id, "description": rule.description}
                    for rule in RULES
                ],
                "claimScope": (
                    "These are deterministic prototype controls over synthetic cases, "
                    "not legal determinations or evidence of bank-policy compliance."
                ),
            },
            "realDataBenchmark": _benchmark_evidence(),
            "redTeam": _red_team_evidence(),
            "verification": {
                "testsPassed": test_count,
                "testsTotal": test_count,
                "scenariosPassed": len(scenarios),
                "scenariosTotal": len(scenarios),
                "blockedOutcomes": blocked_count,
                "apiRequests": 0,
                "method": "Deterministic local tests and zero-request recorded-trace evaluation.",
            },
        },
        "scenarios": scenarios,
    }


def render_bundle() -> str:
    return json.dumps(build_bundle(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Regenerate the bundle.")
    mode.add_argument("--check", action="store_true", help="Fail if the bundle drifted.")
    args = parser.parse_args()

    rendered = render_bundle()
    if args.write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(rendered, encoding="utf-8")
        print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
        return 0

    if not OUTPUT_PATH.is_file():
        print(f"ERROR: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} does not exist")
        return 1
    if OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
        print(
            "ERROR: case-study evidence drifted; run "
            "`python scripts/export_case_study.py --write` and review the diff"
        )
        return 1
    print(f"PASS: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
