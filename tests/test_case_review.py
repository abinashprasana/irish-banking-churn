from copy import deepcopy

import pytest

from agent.tools import PHASE1_FEATURE_SCHEMA, Phase1SchemaError
from case_review import (
    CaseAssessment,
    LabState,
    assessment_state_for_draft,
    build_phase1_customer,
    can_handoff_assessment,
    case_input_fingerprint,
    normalize_case_profile,
    risk_band_for_probability,
)


@pytest.fixture
def raw_profile():
    return {
        "age": 42,
        "tenure_months": 24,
        "account_type": "Current Account",
        "monthly_balance_eur": 2500.0,
        "num_products": 2,
        "monthly_transaction_count": 45,
        "monthly_transaction_amount_eur": 1200.0,
        "has_direct_debits": True,
        "direct_debit_count": 4,
        "uses_digital_bank_secondary": False,
        "was_kbc_ulster_customer": False,
        "months_since_switching": 12,
        "experienced_switching_difficulty": True,
        "branch_visits_monthly": 1,
        "customer_service_calls_6months": 1,
        "has_complaint_history": False,
        "credit_score_band": "Medium",
        "has_mortgage": False,
        "has_savings_goal": True,
    }


def _assessment(profile, fingerprint):
    customer = build_phase1_customer("ATL-DEMO-001", profile)
    return CaseAssessment(
        customer_id=customer["customer_id"],
        ordered_profile=profile,
        probability=0.42,
        band="middle",
        drivers=[],
        counterfactuals=[],
        input_fingerprint=fingerprint,
        model_provenance={"model_artifact": "model.pkl"},
        created_at="2026-08-15T00:00:00+00:00",
        customer_payload=customer,
        state=LabState.SCORED,
    )


def test_normalization_preserves_exact_model_field_order(raw_profile):
    normalized = normalize_case_profile(raw_profile)

    assert tuple(normalized) == PHASE1_FEATURE_SCHEMA
    assert normalized["months_since_switching"] == 0
    assert normalized["experienced_switching_difficulty"] is False


def test_normalization_enforces_control_dependencies(raw_profile):
    raw_profile.update(
        {
            "has_direct_debits": False,
            "direct_debit_count": 9,
            "was_kbc_ulster_customer": False,
            "months_since_switching": 30,
            "experienced_switching_difficulty": True,
            "has_mortgage": True,
            "account_type": "Savings Account",
        }
    )

    normalized = normalize_case_profile(raw_profile)

    assert normalized["direct_debit_count"] == 0
    assert normalized["months_since_switching"] == 0
    assert normalized["experienced_switching_difficulty"] is False
    assert normalized["has_mortgage"] is True
    assert normalized["account_type"] == "Current + Mortgage"


def test_normalization_rejects_schema_drift(raw_profile):
    raw_profile["unexpected"] = 1

    with pytest.raises(Phase1SchemaError, match="unexpected"):
        normalize_case_profile(raw_profile)


def test_fingerprint_covers_profile_and_customer_reference(raw_profile):
    normalized = normalize_case_profile(raw_profile)
    first = case_input_fingerprint(normalized, "ATL-DEMO-001")
    reordered = dict(reversed(list(normalized.items())))

    assert case_input_fingerprint(reordered, "ATL-DEMO-001") == first
    assert case_input_fingerprint(normalized, "ATL-DEMO-002") != first

    changed = deepcopy(normalized)
    changed["age"] += 1
    assert case_input_fingerprint(changed, "ATL-DEMO-001") != first


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.2999, "low"),
        (0.30, "middle"),
        (0.5999, "middle"),
        (0.60, "high"),
        (1.0, "high"),
    ],
)
def test_risk_band_boundaries(probability, expected):
    assert risk_band_for_probability(probability).key == expected


def test_stale_assessment_is_persisted_but_cannot_handoff(raw_profile):
    normalized = normalize_case_profile(raw_profile)
    scored_fingerprint = case_input_fingerprint(normalized, "ATL-DEMO-001")
    assessment = _assessment(normalized, scored_fingerprint)

    changed = deepcopy(normalized)
    changed["monthly_transaction_count"] += 1
    draft_fingerprint = case_input_fingerprint(changed, "ATL-DEMO-001")

    assert assessment_state_for_draft(assessment, draft_fingerprint) is LabState.STALE
    assert not can_handoff_assessment(assessment, draft_fingerprint)
    assert assessment.probability == 0.42


def test_fresh_assessment_round_trips_and_can_handoff(raw_profile):
    normalized = normalize_case_profile(raw_profile)
    fingerprint = case_input_fingerprint(normalized, "ATL-DEMO-001")
    assessment = _assessment(normalized, fingerprint)

    restored = CaseAssessment.from_session(assessment.to_session())

    assert restored == assessment
    assert assessment_state_for_draft(restored, fingerprint) is LabState.SCORED
    assert can_handoff_assessment(restored, fingerprint)


def test_customer_envelope_preserves_phase1_contract(raw_profile):
    raw_profile.update(
        {"account_type": "Current + Mortgage", "has_mortgage": True}
    )
    normalized = normalize_case_profile(raw_profile)

    customer = build_phase1_customer("ATL-DEMO-001", normalized)

    assert tuple(customer["profile"]) == PHASE1_FEATURE_SCHEMA
    assert customer["held_products"] == ["current_account", "mortgage"]
    assert customer["governance"] == {
        "in_arrears": False,
        "vulnerable_customer": False,
    }
