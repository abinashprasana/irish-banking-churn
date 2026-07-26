from copy import deepcopy

import pytest

from agent.tools import (
    PHASE1_FEATURE_SCHEMA,
    Phase1SchemaError,
    load_phase1_runtime,
    predict_customer_churn_risk,
    segment_comparison,
)


CUSTOMER_A = {
    "customer_id": "IRLBANK_FEATURE_A",
    "profile": {
        "age": 43,
        "tenure_months": 114,
        "account_type": "Current Account",
        "monthly_balance_eur": 2584.72,
        "num_products": 1,
        "monthly_transaction_count": 16,
        "monthly_transaction_amount_eur": 671.10,
        "has_direct_debits": False,
        "direct_debit_count": 0,
        "uses_digital_bank_secondary": True,
        "was_kbc_ulster_customer": True,
        "months_since_switching": 3,
        "experienced_switching_difficulty": True,
        "branch_visits_monthly": 0,
        "customer_service_calls_6months": 1,
        "has_complaint_history": False,
        "credit_score_band": "Low",
        "has_mortgage": False,
        "has_savings_goal": True,
    },
}

CUSTOMER_B = {
    "customer_id": "IRLBANK_FEATURE_B",
    "profile": {
        "age": 57,
        "tenure_months": 166,
        "account_type": "Current + Mortgage",
        "monthly_balance_eur": 8650.00,
        "num_products": 3,
        "monthly_transaction_count": 54,
        "monthly_transaction_amount_eur": 3290.00,
        "has_direct_debits": True,
        "direct_debit_count": 6,
        "uses_digital_bank_secondary": False,
        "was_kbc_ulster_customer": False,
        "months_since_switching": 48,
        "experienced_switching_difficulty": False,
        "branch_visits_monthly": 1,
        "customer_service_calls_6months": 0,
        "has_complaint_history": False,
        "credit_score_band": "High",
        "has_mortgage": True,
        "has_savings_goal": True,
    },
}


def test_phase1_risk_changes_with_the_real_customer_feature_vector():
    runtime = load_phase1_runtime()

    tool_output_a = segment_comparison(
        CUSTOMER_A,
        phase1_runtime=runtime,
    )
    tool_output_b = segment_comparison(
        CUSTOMER_B,
        phase1_runtime=runtime,
    )
    prediction_a = tool_output_a["target_phase1_prediction"]
    prediction_b = tool_output_b["target_phase1_prediction"]

    assert prediction_a["churn_probability"] != pytest.approx(
        prediction_b["churn_probability"]
    )
    assert prediction_a["feature_columns"] == list(PHASE1_FEATURE_SCHEMA)
    assert prediction_b["feature_columns"] == list(PHASE1_FEATURE_SCHEMA)
    assert prediction_a["prediction_method"] == (
        "model.predict_proba(feature_vector)[0, 1]"
    )


def test_phase1_runtime_is_cached_and_schema_mismatches_fail_loudly():
    assert load_phase1_runtime() is load_phase1_runtime()

    missing_feature = deepcopy(CUSTOMER_A)
    del missing_feature["profile"]["tenure_months"]
    with pytest.raises(Phase1SchemaError, match="missing=.*tenure_months"):
        predict_customer_churn_risk(missing_feature)

    unknown_encoding = deepcopy(CUSTOMER_A)
    unknown_encoding["profile"]["account_type"] = "Invented Account"
    with pytest.raises(Phase1SchemaError, match="unknown label"):
        predict_customer_churn_risk(unknown_encoding)
