"""Pure state and normalization helpers for the Atlantic Ledger case review.

The Streamlit view deliberately keeps its widget code in ``app.py``.  This module
holds the product state contract and the rules that must remain deterministic so
they can be tested without importing (and executing) the Streamlit application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agent.tools import PHASE1_FEATURE_SCHEMA, Phase1SchemaError


class LabState(str, Enum):
    """Shared lifecycle states used by the interactive lab."""

    DRAFT = "draft"
    EVALUATING = "evaluating"
    SCORED = "scored"
    STALE = "stale"
    AGENT_READY = "agent_ready"
    APPROVED = "approved"
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RiskBand:
    """Display metadata for the locally configured probability bands."""

    key: str
    label: str
    message: str


@dataclass(frozen=True)
class CaseAssessment:
    """Serializable result contract persisted across Streamlit reruns."""

    customer_id: str
    ordered_profile: dict[str, Any]
    probability: float
    band: str
    drivers: list[dict[str, Any]]
    counterfactuals: list[dict[str, Any]]
    input_fingerprint: str
    model_provenance: dict[str, Any]
    created_at: str
    customer_payload: dict[str, Any]
    shap_values: list[float] = field(default_factory=list)
    shap_base_value: float | None = None
    shap_input_values: list[float] = field(default_factory=list)
    state: LabState = LabState.SCORED

    def to_session(self) -> dict[str, Any]:
        """Return primitives that work with serializable Streamlit sessions."""

        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_session(cls, payload: Mapping[str, Any] | None) -> CaseAssessment | None:
        """Safely restore an assessment, returning ``None`` for absent state."""

        if not payload:
            return None
        values = dict(payload)
        values["state"] = LabState(values.get("state", LabState.SCORED.value))
        return cls(**values)

    def with_state(self, state: LabState) -> CaseAssessment:
        return replace(self, state=state)

    def with_counterfactuals(
        self, counterfactuals: list[dict[str, Any]]
    ) -> CaseAssessment:
        return replace(self, counterfactuals=counterfactuals)


def normalize_case_profile(raw_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize dependencies and return the exact trained feature order.

    The UI synchronizes the relevant controls live.  These rules are repeated at
    the model boundary so hidden/stale widget values can never create an invalid
    customer object.
    """

    supplied = set(raw_profile)
    expected = set(PHASE1_FEATURE_SCHEMA)
    missing = expected - supplied
    unexpected = supplied - expected
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        raise Phase1SchemaError(
            "Case review profile schema mismatch: " + "; ".join(details)
        )

    normalized = dict(raw_profile)
    normalized["has_direct_debits"] = bool(normalized["has_direct_debits"])
    if not normalized["has_direct_debits"]:
        normalized["direct_debit_count"] = 0

    normalized["was_kbc_ulster_customer"] = bool(
        normalized["was_kbc_ulster_customer"]
    )
    if not normalized["was_kbc_ulster_customer"]:
        normalized["months_since_switching"] = 0
        normalized["experienced_switching_difficulty"] = False

    mortgage_linked = (
        normalized["account_type"] == "Current + Mortgage"
        or bool(normalized["has_mortgage"])
    )
    normalized["has_mortgage"] = mortgage_linked
    if mortgage_linked:
        normalized["account_type"] = "Current + Mortgage"

    return {name: normalized[name] for name in PHASE1_FEATURE_SCHEMA}


def case_input_fingerprint(
    ordered_profile: Mapping[str, Any], customer_id: str
) -> str:
    """Create a stable fingerprint of every value carried into a decision."""

    canonical = {
        "customer_id": customer_id.strip(),
        "profile": [[name, ordered_profile[name]] for name in PHASE1_FEATURE_SCHEMA],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def risk_band_for_probability(probability: float) -> RiskBand:
    """Return the documented local display band for a probability."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if probability < 0.30:
        return RiskBand(
            "low",
            "Lower score band",
            "The fitted model assigns a lower churn probability to this synthetic profile.",
        )
    if probability < 0.60:
        return RiskBand(
            "middle",
            "Middle score band",
            "The fitted model assigns a middle-range churn probability to this synthetic profile.",
        )
    return RiskBand(
        "high",
        "Higher score band",
        "The fitted model assigns a higher churn probability to this synthetic profile. Treat the score as decision support, not a customer decision.",
    )


def assessment_state_for_draft(
    assessment: CaseAssessment, current_fingerprint: str
) -> LabState:
    """Derive freshness without mutating the stored evidence."""

    if assessment.input_fingerprint != current_fingerprint:
        return LabState.STALE
    return LabState.SCORED


def can_handoff_assessment(
    assessment: CaseAssessment | None, current_fingerprint: str
) -> bool:
    """Only a scored result for the exact visible draft can enter the agent."""

    return bool(
        assessment
        and assessment.input_fingerprint == current_fingerprint
        and assessment.state in {LabState.SCORED, LabState.AGENT_READY}
    )


def build_phase1_customer(
    customer_id: str, ordered_profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the unchanged Phase 1/agent customer envelope."""

    account_type = str(ordered_profile["account_type"])
    has_mortgage = bool(ordered_profile["has_mortgage"])
    held_products: list[str] = []
    if "Current" in account_type:
        held_products.append("current_account")
    if "Savings" in account_type:
        held_products.append("savings_account")
    if has_mortgage:
        held_products.append("mortgage")

    return {
        "customer_id": customer_id.strip() or "ATL-DEMO-001",
        "profile": dict(ordered_profile),
        "held_products": sorted(set(held_products)),
        "governance": {
            "in_arrears": False,
            "vulnerable_customer": False,
        },
        "governance_note": (
            "Case review supplies model features only. The synthetic governance "
            "overlay defaults to no arrears and not vulnerable."
        ),
        "counterfactuals": [],
        "churn_drivers": [],
    }
