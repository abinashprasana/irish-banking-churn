import copy
import json
from pathlib import Path

from scripts.eval_agent import _check_live_capture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_live(demo):
    live = copy.deepcopy(demo)
    live["recording"] = {
        **{k: v for k, v in demo["recording"].items() if k not in {"note"}},
        "live_capture": True,
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "captured_at": "2026-09-30T10:00:00Z",
        "reasoning_source": "live_gemini",
        "model_output_captured": True,
        "real_api_calls": 4,
        "token_usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
    }
    return live


def test_live_capture_contract_accepts_marked_traces_and_rejects_scripted_ones():
    demo = json.loads((PROJECT_ROOT / "demo_traces" / "01_allowed_fee_waiver.json").read_text(encoding="utf-8"))
    assert _check_live_capture(_as_live(demo), "live/01") == []

    unmarked = _as_live(demo)
    unmarked["recording"]["live_capture"] = False
    unmarked["recording"]["reasoning_source"] = "scripted_fixture"
    failures = _check_live_capture(unmarked, "live/01")
    assert any("live capture marker" in f for f in failures)
    assert any("scripted_fixture" in f for f in failures)


def test_every_committed_live_capture_passes_the_contract():
    for path in sorted((PROJECT_ROOT / "demo_traces" / "live").glob("*.json")):
        demo = json.loads(path.read_text(encoding="utf-8"))
        assert _check_live_capture(demo, path.name) == [], path.name
