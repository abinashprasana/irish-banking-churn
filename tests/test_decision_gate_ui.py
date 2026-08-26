import json
from pathlib import Path

from decision_gate_ui import build_decision_stages, recommendation_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _recorded_runs():
    for path in sorted((PROJECT_ROOT / "demo_traces").glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def test_recommendation_state_distinguishes_governed_outcomes():
    assert recommendation_state(None) == "agent_ready"
    assert recommendation_state({"checker_verdict": "blocked"}) == "blocked"
    assert recommendation_state(
        {
            "checker_verdict": "approved",
            "regulatory_flags": ["HUM-003:human_review_required"],
        }
    ) == "review_required"
    assert recommendation_state(
        {"checker_verdict": "approved", "regulatory_flags": []}
    ) == "approved"


def test_recorded_trace_stages_retain_every_event_and_all_four_tools():
    for run in _recorded_runs():
        stages = build_decision_stages(run["trace"])

        assert [stage["key"] for stage in stages] == [
            "product",
            "segment",
            "policy",
            "format",
        ]
        assert all(stage["events"] for stage in stages)
        assert [
            event
            for stage in stages
            for event in stage["events"]
        ] == run["trace"]


def test_blocked_recordings_read_as_governed_states_not_runtime_errors():
    blocked_runs = [
        run
        for run in _recorded_runs()
        if run["recommendation"]["checker_verdict"] == "blocked"
    ]
    assert blocked_runs
    for run in blocked_runs:
        stages = build_decision_stages(run["trace"])
        assert recommendation_state(run["recommendation"]) == "blocked"
        assert stages[2]["status"] == "blocked"
        assert stages[3]["status"] == "blocked"
        assert "stopped" in stages[2]["summary"].lower()
        assert "no recommendation" in stages[3]["summary"].lower()
