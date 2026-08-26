from __future__ import annotations

from pathlib import Path

import pytest

import lab_ui.decision_instrument as instrument
from lab_ui.decision_instrument import _normalize_steps


ROOT = Path(__file__).resolve().parents[1]


def test_default_instrument_steps_form_the_governed_pathway() -> None:
    assert [step["id"] for step in _normalize_steps(None)] == [
        "profile",
        "model",
        "explanation",
        "policy",
    ]


def test_instrument_steps_are_bounded_and_have_unique_ids() -> None:
    steps = [
        {"id": "duplicate", "label": "First", "summary": "A"},
        {"id": "duplicate", "label": "Second", "summary": "B"},
    ] + [
        {"id": f"step-{index}", "label": str(index), "summary": "C"}
        for index in range(10)
    ]

    normalized = _normalize_steps(steps)  # type: ignore[arg-type]

    assert len(normalized) == 8
    assert len({step["id"] for step in normalized}) == 8
    assert normalized[0]["id"] == "duplicate"
    assert normalized[1]["id"] == "duplicate-2"


@pytest.mark.parametrize(
    "state",
    [
        "draft",
        "evaluating",
        "scored",
        "stale",
        "agent_ready",
        "approved",
        "blocked",
        "review_required",
        "unavailable",
    ],
)
def test_component_accepts_every_decision_state(monkeypatch, state: str) -> None:
    captured: dict = {}

    class Result:
        selected_step = "model"

    def fake_component(**kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(instrument, "_COMPONENT", fake_component)
    selected = instrument.render_decision_instrument(
        variant="assessment",
        stage=state,  # type: ignore[arg-type]
        key=f"instrument-{state}",
    )

    assert selected == "model"
    assert captured["data"]["stage"] == state
    assert set(captured["default"]) == {"selected_step"}


def test_component_returns_only_a_valid_selected_step(monkeypatch) -> None:
    class Result:
        selected_step = "not-a-step"

    monkeypatch.setattr(instrument, "_COMPONENT", lambda **_: Result())
    selected = instrument.render_decision_instrument(
        variant="governance",
        stage="approved",
        steps=[
            {"id": "lookup", "label": "Product lookup", "summary": "Complete"},
            {"id": "policy", "label": "Policy check", "summary": "Approved"},
        ],
        selected_step="policy",
        key="governance-instrument",
    )

    assert selected == "policy"


def test_component_reuses_its_persisted_selected_step(monkeypatch) -> None:
    captured: dict = {}

    class Result:
        selected_step = "policy"

    def fake_component(**kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setitem(
        instrument.st.session_state,
        "persisted-instrument",
        {"selected_step": "policy"},
    )
    monkeypatch.setattr(instrument, "_COMPONENT", fake_component)
    selected = instrument.render_decision_instrument(
        variant="governance",
        stage="approved",
        steps=[
            {"id": "lookup", "label": "Product lookup", "summary": "Complete"},
            {"id": "policy", "label": "Policy check", "summary": "Approved"},
        ],
        key="persisted-instrument",
    )

    assert selected == "policy"
    assert captured["data"]["selected_step"] == "policy"


def test_component_source_encodes_semantics_and_reduced_motion() -> None:
    source = (ROOT / "lab_ui/component/src/decision-instrument.tsx").read_text(
        encoding="utf-8"
    )
    css = (ROOT / "lab_ui/assets/decision-instrument.css").read_text(encoding="utf-8")

    for semantic in [
        'role="progressbar"',
        'aria-current=',
        'aria-live="polite"',
        'aria-label="Decision provenance"',
    ]:
        assert semantic in source
    assert "prefers-reduced-motion: reduce" in css
    assert "--instrument-teal-text: #0f6b55" in css
    assert "--instrument-amber-text: #7a4d12" in css
    assert "@keyframes" not in css
    assert "animation:" not in css
