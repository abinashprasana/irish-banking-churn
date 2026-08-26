"""Deployment-safe wrapper for the Atlantic Ledger decision instrument.

The frontend is authored in TypeScript/React and committed as a production bundle.
Community Cloud only reads the built asset; Node is not needed at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import streamlit as st

InstrumentVariant = Literal["assessment", "governance"]
DecisionState = Literal[
    "draft",
    "evaluating",
    "scored",
    "stale",
    "agent_ready",
    "approved",
    "blocked",
    "review_required",
    "unavailable",
]
StepStatus = Literal["pending", "active", "complete", "blocked", "review"]
VerdictStatus = Literal["approved", "blocked", "review_required", "unavailable"]


class InstrumentStep(TypedDict):
    id: str
    label: str
    summary: str
    status: NotRequired[StepStatus]
    detail: NotRequired[str]


class ProvenanceItem(TypedDict):
    label: str
    value: str


class Thresholds(TypedDict):
    lower: float
    higher: float


class Verdict(TypedDict):
    status: VerdictStatus
    label: str
    detail: NotRequired[str]


_ASSET_DIR = Path(__file__).with_name("assets")
_COMPONENT = st.components.v2.component(
    "atlantic_ledger_decision_instrument",
    html='<div data-decision-instrument-root></div>',
    css=(_ASSET_DIR / "decision-instrument.css").read_text(encoding="utf-8"),
    js=(_ASSET_DIR / "decision-instrument.bundle.js").read_text(encoding="utf-8"),
    isolate_styles=True,
)


def _noop() -> None:
    """Provide the callback required for a persistent component state value."""


def _clean_text(value: object, *, limit: int = 240) -> str:
    """Return bounded plain text; React performs the final HTML escaping."""

    return str(value or "").strip()[:limit]


def _normalize_steps(steps: list[InstrumentStep] | None) -> list[InstrumentStep]:
    source = steps or [
        {
            "id": "profile",
            "label": "Profile",
            "summary": "A synthetic customer profile is assembled from ordered inputs.",
        },
        {
            "id": "model",
            "label": "Model",
            "summary": "The held-out churn model produces a probability estimate.",
        },
        {
            "id": "explanation",
            "label": "Explanation",
            "summary": "Signed drivers describe the evidence behind this score.",
        },
        {
            "id": "policy",
            "label": "Policy",
            "summary": "Deterministic rules govern the proposed response.",
        },
    ]
    normalized: list[InstrumentStep] = []
    seen: set[str] = set()
    for index, item in enumerate(source[:8]):
        step_id = _clean_text(item.get("id"), limit=48) or f"step-{index + 1}"
        if step_id in seen:
            step_id = f"{step_id}-{index + 1}"
        seen.add(step_id)
        status = item.get("status", "pending")
        if status not in {"pending", "active", "complete", "blocked", "review"}:
            status = "pending"
        normalized.append(
            {
                "id": step_id,
                "label": _clean_text(item.get("label"), limit=64) or f"Step {index + 1}",
                "summary": _clean_text(item.get("summary")),
                "status": cast(StepStatus, status),
                "detail": _clean_text(item.get("detail"), limit=600),
            }
        )
    return normalized


def render_decision_instrument(
    *,
    variant: InstrumentVariant,
    stage: DecisionState,
    steps: list[InstrumentStep] | None = None,
    selected_step: str | None = None,
    score: float | None = None,
    thresholds: Thresholds | None = None,
    provenance: list[ProvenanceItem] | None = None,
    verdict: Verdict | None = None,
    rule_ids: list[str] | None = None,
    key: str,
) -> str:
    """Render the instrument and return only its selected step identifier.

    All content crosses the component boundary as bounded JSON primitives. The
    frontend renders strings as React text nodes and never uses raw HTML.
    """

    normalized_steps = _normalize_steps(steps)
    valid_ids = {item["id"] for item in normalized_steps}
    component_state = st.session_state.get(key, {})
    persisted_step = (
        component_state.get("selected_step")
        if isinstance(component_state, Mapping)
        else None
    )
    initial_step = (
        selected_step
        if selected_step in valid_ids
        else persisted_step
        if persisted_step in valid_ids
        else normalized_steps[0]["id"]
    )

    normalized_score = None
    if score is not None:
        normalized_score = min(1.0, max(0.0, float(score)))

    normalized_thresholds = thresholds or {"lower": 0.3, "higher": 0.6}
    lower = min(1.0, max(0.0, float(normalized_thresholds["lower"])))
    higher = min(1.0, max(lower, float(normalized_thresholds["higher"])))

    safe_provenance = [
        {
            "label": _clean_text(item.get("label"), limit=64),
            "value": _clean_text(item.get("value"), limit=120),
        }
        for item in (provenance or [])[:6]
    ]
    safe_rules = [_clean_text(rule_id, limit=64) for rule_id in (rule_ids or [])[:8]]
    safe_verdict = None
    if verdict:
        verdict_status = verdict.get("status", "unavailable")
        if verdict_status not in {"approved", "blocked", "review_required", "unavailable"}:
            verdict_status = "unavailable"
        safe_verdict = {
            "status": verdict_status,
            "label": _clean_text(verdict.get("label"), limit=100),
            "detail": _clean_text(verdict.get("detail"), limit=400),
        }

    result = _COMPONENT(
        key=key,
        data={
            "variant": variant,
            "stage": stage,
            "steps": normalized_steps,
            "selected_step": initial_step,
            "score": normalized_score,
            "thresholds": {"lower": lower, "higher": higher},
            "provenance": safe_provenance,
            "verdict": safe_verdict,
            "rule_ids": safe_rules,
        },
        default={"selected_step": initial_step},
        on_selected_step_change=_noop,
        width="stretch",
        height="content",
    )
    candidate = getattr(result, "selected_step", None)
    return candidate if candidate in valid_ids else initial_step


__all__ = [
    "DecisionState",
    "InstrumentStep",
    "InstrumentVariant",
    "ProvenanceItem",
    "Thresholds",
    "Verdict",
    "render_decision_instrument",
]
