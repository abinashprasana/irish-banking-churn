"""Refresh all recorded traces with the shared Groq free-tier backend."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import (  # noqa: E402
    MAX_LIVE_API_CALLS,
    MAX_TOKENS,
    MODEL_NAME,
    create_live_client,
    resolve_groq_api_key,
    run_retention_agent,
)


DEMO_DIR = PROJECT_ROOT / "demo_traces"


def main() -> None:
    api_key = resolve_groq_api_key()
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not present. No live client was created and no request was made."
        )

    paths = sorted(DEMO_DIR.glob("*.json"))
    if len(paths) != 4:
        raise SystemExit(f"Expected exactly four demo traces, found {len(paths)}.")

    print("Refreshing four traces through the Groq free tier.")
    print(f"Model: {MODEL_NAME}")
    print(f"Maximum completion tokens per request: {MAX_TOKENS}")
    print(f"Model-call cap per customer run: {MAX_LIVE_API_CALLS}")

    completed = []
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        client = create_live_client(api_key=api_key)
        result = run_retention_agent(artifact["customer"], client=client)
        artifact["recording"] = {
            "mode": "owner_recorded_live_groq",
            "real_api_calls": client.chat.completions.request_count,
            "model": MODEL_NAME,
            "model_output_captured": True,
            "reasoning_source": "live_groq",
            "max_tokens_per_call": MAX_TOKENS,
            "call_cap": MAX_LIVE_API_CALLS,
            "phase1_runtime_capture": True,
            "phase1_model_artifact": result["phase1_prediction"]["model_artifact"],
            "phase1_prediction_method": result["phase1_prediction"][
                "prediction_method"
            ],
        }
        artifact["customer"] = result["customer"]
        artifact["trace"] = result["trace"]
        artifact["recommendation"] = result["recommendation"]
        completed.append((path, artifact))

    # Preserve all existing artifacts unless every live run completes.
    for path, artifact in completed:
        path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Updated {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
