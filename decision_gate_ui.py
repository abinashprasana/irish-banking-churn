"""Pure presentation helpers for the Atlantic Ledger decision gate.

The Streamlit view deliberately keeps the agent and policy interfaces untouched.
These helpers only translate an existing recommendation and trace into stable UI
states so recorded and live runs are presented in the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


TRACE_STAGE_DEFINITIONS = (
    ("product", "Product lookup", "product_lookup"),
    ("segment", "Segment comparison", "segment_comparison"),
    ("policy", "Policy constraint check", "regulatory_constraint_checker"),
    ("format", "Recommendation formatting", "recommendation_formatter"),
)


def recommendation_state(recommendation: Mapping[str, Any] | None) -> str:
    """Return the governed product state for an existing recommendation."""
    if not recommendation:
        return "agent_ready"
    if recommendation.get("checker_verdict") != "approved":
        return "blocked"
    flags = recommendation.get("regulatory_flags") or ()
    if any("human_review_required" in str(flag).lower() for flag in flags):
        return "review_required"
    return "approved"


def _tool_name(event: Mapping[str, Any]) -> str | None:
    content = event.get("content")
    if not isinstance(content, Mapping):
        return None
    name = content.get("name")
    return str(name) if name else None


def _stage_summary(stage_key: str, events: list[Mapping[str, Any]]) -> str:
    results = [
        event.get("content", {}).get("result", {})
        for event in events
        if event.get("type") == "tool_result"
        and not event.get("content", {}).get("is_error", False)
    ]
    if stage_key == "product" and results:
        offer_count = len(results[-1].get("offers", ()))
        noun = "option" if offer_count == 1 else "options"
        return f"{offer_count} eligible {noun} returned"
    if stage_key == "segment" and results:
        result = results[-1]
        cohort_size = result.get("cohort_size", "—")
        churn_rate = result.get("churn_rate")
        rate_text = f"{float(churn_rate):.1%}" if churn_rate is not None else "—"
        return f"Cohort of {cohort_size} · {rate_text} churn"
    if stage_key == "policy":
        gate_events = [event for event in events if event.get("type") == "gate_check"]
        if gate_events:
            content = gate_events[-1].get("content", {})
            rules = content.get("rule_results", ())
            failed = content.get("failed_rule_ids", ())
            if content.get("passed"):
                return f"All {len(rules)} local rules passed"
            noun = "rule" if len(failed) == 1 else "rules"
            return f"{len(failed)} local {noun} stopped the action"
    if stage_key == "format":
        final_events = [event for event in events if event.get("type") == "final_output"]
        if final_events:
            content = final_events[-1].get("content", {})
            if content.get("checker_verdict") != "approved":
                return "No recommendation issued"
            flags = content.get("regulatory_flags", ())
            if any("human_review_required" in str(flag).lower() for flag in flags):
                return "Advisor review retained in the output"
            return "Governed recommendation formatted"
    if any(event.get("content", {}).get("is_error") for event in events):
        return "Stage stopped safely"
    return "No result recorded for this stage"


def _stage_status(stage_key: str, events: list[Mapping[str, Any]]) -> str:
    if not events:
        return "pending"
    if any(event.get("content", {}).get("is_error") for event in events):
        return "unavailable"
    if stage_key == "policy":
        for event in events:
            if event.get("type") == "gate_check" and not event.get("content", {}).get(
                "passed", False
            ):
                return "blocked"
    if stage_key == "format":
        finals = [event for event in events if event.get("type") == "final_output"]
        if finals:
            final = finals[-1].get("content", {})
            if final.get("checker_verdict") != "approved":
                return "blocked"
            if any(
                "human_review_required" in str(flag).lower()
                for flag in final.get("regulatory_flags", ())
            ):
                return "review_required"
    return "complete"


def build_decision_stages(
    trace: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group a full agent trace into four task-focused stages.

    Every source event is retained exactly once. Model thoughts are attached to the
    most recently active stage; the initial thought belongs to product lookup.
    """
    stages = [
        {
            "key": key,
            "label": label,
            "tool_name": tool_name,
            "events": [],
        }
        for key, label, tool_name in TRACE_STAGE_DEFINITIONS
    ]
    tool_indexes = {
        stage["tool_name"]: index for index, stage in enumerate(stages)
    }
    active_index = 0

    for event in trace:
        event_type = event.get("type")
        event_tool = _tool_name(event)
        if event_tool in tool_indexes:
            active_index = tool_indexes[event_tool]
        elif event_type == "gate_check":
            active_index = tool_indexes["regulatory_constraint_checker"]
        elif event_type == "final_output":
            active_index = tool_indexes["recommendation_formatter"]
        stages[active_index]["events"].append(event)

    for stage in stages:
        stage["status"] = _stage_status(stage["key"], stage["events"])
        stage["summary"] = _stage_summary(stage["key"], stage["events"])
    return stages
