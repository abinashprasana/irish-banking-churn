"""Evaluate the governed retention agent without spending paid API credit.

Dry run (default or ``--dry-run``):
    Replays the four recorded traces and validates tool IDs, output schema, and
    policy-verdict consistency. It makes zero Groq requests.

Live (``--live``):
    Runs selected recorded customer scenarios through Groq's free-tier
    ``qwen/qwen3.8-27b`` endpoint (or Gemini with ``--provider gemini``). It requires ``GROQ_API_KEY`` and is
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


def live_run(
    scenario: str | None = None,
    provider: str = "groq",
    model: str | None = None,
    max_requests: int = 40,
) -> int:
    import time

    from agent.loop import (
        MAX_LIVE_API_CALLS,
        MAX_TOKENS,
        create_live_client,
        resolve_groq_api_key,
        run_retention_agent,
    )
    from agent.providers import GeminiAPIError, resolve_gemini_api_key
    from agent.rate_limits import RateLimitSafetyError
    from redteam.live import BudgetExhausted, PacedQuota, UsageLedger, attach_usage_recorder
    from redteam.oracle import ScenarioFacts, judge

    key_name = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
    api_key = resolve_gemini_api_key() if provider == "gemini" else resolve_groq_api_key()
    if not api_key:
        print(f"ERROR: {key_name} is not set. No {provider} request was made.")
        return 1

    paths = _selected_paths(scenario)
    if not paths:
        print("ERROR: no matching recorded traces found in demo_traces/.")
        return 1

    model_name = create_live_client(api_key=api_key, provider=provider, model=model).model_name
    ledger = UsageLedger()
    # One paced guard for the whole run, so scenarios wait for the minute window.
    quota = PacedQuota(max_requests, ledger, provider=provider, scope=model_name)
    print(f"\nLive {provider} eval - {len(paths)} scenario(s)")
    print(f"Model: {model_name}")
    print(f"Max completion tokens per request: {MAX_TOKENS}")
    print(f"Model-call cap per run: {MAX_LIVE_API_CALLS}\n")

    all_failures: list[str] = []
    rows: list[dict] = []
    stopped = None
    for path in paths:
        demo = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = demo.get("demo_id", path.name)
        title = demo.get("title", "")
        print(f"\n  Running {scenario_id}: {title} ...")
        client = attach_usage_recorder(
            create_live_client(api_key=api_key, provider=provider, quota_guard=quota, model=model),
            ledger,
        )
        start = len(ledger.requests)
        started = time.monotonic()
        try:
            result = run_retention_agent(demo["customer"], client=client)
        except (BudgetExhausted, RateLimitSafetyError) as exc:
            stopped = f"{scenario_id}: {exc}"
            break
        except GeminiAPIError as exc:
            if exc.status_code == 429:
                stopped = f"{scenario_id}: provider quota reached (HTTP 429)"
                break
            message = f"{scenario_id}: live run raised {type(exc).__name__}: {exc}"
            all_failures.append(message)
            rows.append({"scenario": scenario_id, "passed": False, "error": message})
            print(f"  [FAIL] {message}")
            continue
        except Exception as exc:
            message = f"{scenario_id}: live run raised {type(exc).__name__}: {exc}"
            all_failures.append(message)
            rows.append({"scenario": scenario_id, "passed": False, "error": message})
            print(f"  [FAIL] {message}")
            continue

        requests = ledger.requests[start:]
        tokens = {
            key: sum(item[key] or 0 for item in requests)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        recommendation = result["recommendation"]
        facts = ScenarioFacts.from_customer(demo["customer"], result["customer"]["churn_probability"])
        findings = [finding.as_dict() for finding in judge(recommendation, result["trace"], facts)]
        live_demo = {
            "schema_version": demo.get("schema_version", "1.0"),
            "demo_id": scenario_id,
            "title": title,
            "recording": {
                "mode": f"eval_live_{provider}",
                "real_api_calls": client.chat.completions.request_count,
                "model": client.model_name,
                "model_output_captured": True,
                "reasoning_source": f"live_{provider}",
                "max_tokens_per_call": MAX_TOKENS,
                "call_cap": MAX_LIVE_API_CALLS,
                "phase1_runtime_capture": True,
                "phase1_model_artifact": result["phase1_prediction"]["model_artifact"],
                "phase1_prediction_method": result["phase1_prediction"]["prediction_method"],
            },
            "customer": result["customer"],
            "trace": result["trace"],
            "recommendation": recommendation,
        }
        failures = _check_scenario(live_demo, scenario_id)
        all_failures.extend(failures)
        print(
            f"       requests {client.chat.completions.request_count} · "
            f"prompt tokens {tokens['prompt_tokens']} · completion tokens {tokens['completion_tokens']} · "
            f"total tokens {tokens['total_tokens']} · {time.monotonic() - started:.1f}s"
        )
        print(
            f"       outcome {recommendation['action']} · {recommendation['checker_verdict']} · "
            f"flags {recommendation['regulatory_flags']} · turns {result['turns']} · "
            f"oracle findings {[item['harm_id'] for item in findings] or 'none'}"
        )
        _print_result(scenario_id, title, failures)
        rows.append(
            {
                "scenario": scenario_id,
                "passed": not failures,
                "failures": failures,
                "scripted_outcome": [demo["recommendation"]["action"], demo["recommendation"]["checker_verdict"]],
                "live_outcome": [recommendation["action"], recommendation["checker_verdict"]],
                "flags": recommendation["regulatory_flags"],
                "oracle_findings": findings,
                "turns": result["turns"],
                "requests": client.chat.completions.request_count,
                "tokens": tokens,
                "seconds": round(time.monotonic() - started, 1),
            }
        )

    summary_dir = PROJECT_ROOT / "redteam" / "runs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
    summary_path = summary_dir / f"live_eval_{provider}_{safe_model}.json"
    summary_path.write_text(
        json.dumps({"provider": provider, "model": model_name, "stopped": stopped, "scenarios": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nRequests used: {quota.used} · summary {summary_path.relative_to(PROJECT_ROOT)} (gitignored)")
    if stopped:
        print(f"Stopped early: {stopped}")
        return 1
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
        "--provider",
        choices=["groq", "gemini"],
        default="groq",
        help="Live model provider. Keys come from the environment or the gitignored .env.",
    )
    parser.add_argument(
        "--model",
        help="Live only. Override the provider's default model, for example gemini-3.5-flash-lite.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=40,
        help="Live only. Request budget for the whole run; stops cleanly when reached.",
    )
    parser.add_argument(
        "--scenario",
        help="Optional demo_id; omit to evaluate all four recorded scenarios.",
    )
    args = parser.parse_args()
    if args.live:
        from agent.providers import load_env_file

        load_env_file()
    sys.exit(
        live_run(args.scenario, args.provider, args.model, args.max_requests)
        if args.live
        else dry_run(args.scenario)
    )
