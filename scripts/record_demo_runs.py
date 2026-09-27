"""Record the four demo scenarios live and save them beside the scripted traces.

Live captures go to demo_traces/live/ with provider, model, timestamp and token
usage in their recording block. The public scripted traces in demo_traces/ and the
case study exporter are not touched. Keys come from GROQ_API_KEY or GEMINI_API_KEY
in the environment or the gitignored .env, and are never printed or saved.

Usage:
    python scripts/record_demo_runs.py --provider gemini
    python scripts/record_demo_runs.py --provider groq --max-requests 40
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import (  # noqa: E402
    MAX_LIVE_API_CALLS,
    MAX_TOKENS,
    create_live_client,
    resolve_groq_api_key,
    run_retention_agent,
)
from agent.providers import GeminiAPIError, load_env_file, resolve_gemini_api_key  # noqa: E402
from agent.rate_limits import RateLimitSafetyError  # noqa: E402
from redteam.live import BudgetExhausted, PacedQuota, UsageLedger, attach_usage_recorder  # noqa: E402


DEMO_DIR = PROJECT_ROOT / "demo_traces"
LIVE_DIR = DEMO_DIR / "live"


def _usage(requests: list[dict]) -> dict[str, int]:
    return {
        key: sum(item[key] or 0 for item in requests)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--max-requests", type=int, default=40, help="Stops cleanly when reached.")
    parser.add_argument("--scenario", nargs="*", help="Record only these demo ids.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenarios already captured in demo_traces/live/.")
    args = parser.parse_args()

    load_env_file()
    key_name = "GEMINI_API_KEY" if args.provider == "gemini" else "GROQ_API_KEY"
    api_key = resolve_gemini_api_key() if args.provider == "gemini" else resolve_groq_api_key()
    if not api_key:
        raise SystemExit(f"{key_name} is not present. No live client was created and no request was made.")

    paths = sorted(DEMO_DIR.glob("*.json"))
    if len(paths) != 4:
        raise SystemExit(f"Expected exactly four demo traces, found {len(paths)}.")
    if args.scenario:
        paths = [path for path in paths if path.stem in set(args.scenario)]
    if args.skip_existing:
        paths = [path for path in paths if not (LIVE_DIR / path.name).is_file()]
    if not paths:
        print("Nothing to record.")
        return 0

    ledger = UsageLedger()
    quota = PacedQuota(args.max_requests, ledger, provider=args.provider)
    model_name = create_live_client(api_key=api_key, provider=args.provider).model_name
    print(f"Recording four live traces with {args.provider}, model {model_name}.")
    print(f"Maximum completion tokens per request: {MAX_TOKENS}. Model-call cap per run: {MAX_LIVE_API_CALLS}.")

    LIVE_DIR.mkdir(exist_ok=True)
    stopped = None
    for path in paths:
        scripted = json.loads(path.read_text(encoding="utf-8"))
        client = attach_usage_recorder(
            create_live_client(api_key=api_key, quota_guard=quota, provider=args.provider), ledger
        )
        start = len(ledger.requests)
        try:
            result = run_retention_agent(scripted["customer"], client=client)
        except (BudgetExhausted, RateLimitSafetyError) as exc:
            stopped = f"{scripted['demo_id']}: {exc}"
            break
        except GeminiAPIError as exc:
            if exc.status_code != 429:
                raise
            stopped = f"{scripted['demo_id']}: provider quota reached (HTTP 429)"
            break
        usage = _usage(ledger.requests[start:])
        customer = {key: value for key, value in result["customer"].items() if key != "phase1_prediction"}
        artifact = {
            "schema_version": scripted["schema_version"],
            "demo_id": scripted["demo_id"],
            "title": scripted["title"],
            "recording": {
                "mode": "owner_recorded_live",
                "live_capture": True,
                "provider": args.provider,
                "model": client.model_name,
                "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "real_api_calls": client.chat.completions.request_count,
                "token_usage": usage,
                "model_output_captured": True,
                "reasoning_source": f"live_{args.provider}",
                "max_tokens_per_call": MAX_TOKENS,
                "call_cap": MAX_LIVE_API_CALLS,
                "phase1_runtime_capture": True,
                "phase1_model_artifact": result["phase1_prediction"]["model_artifact"],
                "phase1_prediction_method": result["phase1_prediction"]["prediction_method"],
                "synthetic_data": True,
                "synthetic_governance": True,
                "note": (
                    "Captured from a live model run on synthetic data. The model is not "
                    "deterministic, so a new capture can differ. The trace was not edited."
                ),
            },
            "customer": customer,
            "trace": result["trace"],
            "recommendation": result["recommendation"],
        }
        serialised = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
        secret = os.environ.get(key_name, "")
        if "gsk_" in serialised or (secret and secret in serialised):
            raise SystemExit("Refusing to write a trace that contains key material.")
        before = scripted["recommendation"]
        after = result["recommendation"]
        changed = (before["action"], before["checker_verdict"]) != (after["action"], after["checker_verdict"])
        print(
            f"  {scripted['demo_id']}: {after['action']} · {after['checker_verdict']} · "
            f"{client.chat.completions.request_count} requests · {usage['total_tokens']} tokens"
            + (f"  (scripted outcome was {before['action']} · {before['checker_verdict']})" if changed else "")
        )
        # Each capture is complete on its own, so save it before the next run.
        target = LIVE_DIR / path.name
        target.write_text(serialised, encoding="utf-8")
        print(f"    wrote {target.relative_to(PROJECT_ROOT)}")

    if stopped:
        print(f"Stopped early, nothing partial was saved: {stopped}")
    totals = _usage(ledger.requests)
    print(f"Total: {quota.used} requests · {totals['prompt_tokens']} prompt tokens · "
          f"{totals['completion_tokens']} completion tokens · {totals['total_tokens']} total tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
