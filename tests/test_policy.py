from dataclasses import FrozenInstanceError

import pytest

from agent.policy_rules import (
    ActionPolicyContext,
    CustomerPolicyContext,
    HIGH_RISK_HUMAN_REVIEW_THRESHOLD,
    RULES,
)
from agent.tools import regulatory_constraint_checker
from agent.tools import (
    PolicyGateError,
    TOOL_DEFINITIONS,
    canonical_action_context,
    product_lookup,
    recommendation_formatter,
    segment_comparison,
)


def customer(**overrides):
    values = {
        "customer_id": "IRLBANK_TEST",
        "churn_probability": 0.60,
        "held_products": ("current_account",),
        "in_arrears": False,
        "vulnerable_customer": False,
    }
    values.update(overrides)
    return CustomerPolicyContext(**values)


def action(**overrides):
    values = {
        "action_id": "fee_waiver_6m",
        "category": "fee_relief",
        "is_credit": False,
        "is_upsell": False,
        "target_product": None,
        "requires_human_review": False,
    }
    values.update(overrides)
    return ActionPolicyContext(**values)


def result_for(rule_id, checked_customer, checked_action):
    decision = regulatory_constraint_checker(checked_customer, checked_action)
    return next(result for result in decision.rule_results if result.rule_id == rule_id)


def test_every_rule_has_a_passing_and_failing_case():
    cases = {
        "ARR-001": (
            (customer(in_arrears=True), action()),
            (
                customer(in_arrears=True),
                action(action_id="credit_limit_review", category="credit", is_credit=True),
            ),
        ),
        "HOLD-002": (
            (
                customer(held_products=("current_account",)),
                action(
                    action_id="savings_bonus",
                    category="savings",
                    target_product="savings_account",
                    is_upsell=True,
                ),
            ),
            (
                customer(held_products=("current_account", "savings_account")),
                action(
                    action_id="savings_bonus",
                    category="savings",
                    target_product="savings_account",
                    is_upsell=True,
                ),
            ),
        ),
        "HUM-003": (
            (
                customer(churn_probability=0.90),
                action(requires_human_review=True),
            ),
            (
                customer(churn_probability=0.90),
                action(requires_human_review=False),
            ),
        ),
        "VUL-004": (
            (customer(vulnerable_customer=True), action()),
            (
                customer(vulnerable_customer=True),
                action(
                    action_id="savings_bonus",
                    category="savings",
                    target_product="savings_account",
                    is_upsell=True,
                ),
            ),
        ),
    }

    assert {rule.rule_id for rule in RULES} == set(cases)
    for rule_id, (passing_case, failing_case) in cases.items():
        passing = result_for(rule_id, *passing_case)
        failing = result_for(rule_id, *failing_case)
        assert passing.passed is True, rule_id
        assert failing.passed is False, rule_id
        assert passing.reason and failing.reason


def test_threshold_is_exclusive_and_checker_returns_every_rule():
    checked_customer = customer(churn_probability=HIGH_RISK_HUMAN_REVIEW_THRESHOLD)
    decision = regulatory_constraint_checker(checked_customer, action())
    payload = decision.as_dict()

    assert decision.passed is True
    assert [item["rule_id"] for item in payload["rule_results"]] == [
        rule.rule_id for rule in RULES
    ]
    assert all(item["reason"] for item in payload["rule_results"])


def test_arrears_and_vulnerable_safeguards_fail_closed_without_short_circuiting():
    restricted_customer = customer(
        churn_probability=0.90,
        in_arrears=True,
        vulnerable_customer=True,
    )
    credit_upsell = action(
        action_id="mortgage_switch_offer",
        category="mortgage",
        is_credit=True,
        is_upsell=True,
        target_product="mortgage",
        requires_human_review=True,
    )
    decision = regulatory_constraint_checker(restricted_customer, credit_upsell)

    assert decision.passed is False
    assert {"ARR-001", "VUL-004"}.issubset(decision.failed_rule_ids)
    assert len(decision.rule_results) == len(RULES)
    with pytest.raises(TypeError):
        regulatory_constraint_checker(restricted_customer, credit_upsell, rules=())

    disguised_upsell = action(
        action_id="fee_bundle",
        category="fee_relief",
        is_upsell=True,
    )
    assert result_for("VUL-004", restricted_customer, disguised_upsell).passed is False


def test_policy_decision_is_immutable_and_bound_to_exact_customer_and_action():
    checked_customer = customer(churn_probability=0.90)
    checked_action = action(requires_human_review=True)
    decision = regulatory_constraint_checker(checked_customer, checked_action)

    assert decision.authorizes(checked_customer, checked_action) is True
    assert decision.authorizes(
        checked_customer,
        action(action_id="different_action", requires_human_review=True),
    ) is False
    assert decision.authorizes(
        customer(customer_id="IRLBANK_OTHER", churn_probability=0.90),
        checked_action,
    ) is False
    with pytest.raises(FrozenInstanceError):
        decision.passed = False


def test_product_lookup_is_synthetic_and_canonical_metadata_is_fail_closed():
    catalogue = product_lookup()
    names = [schema["name"] for schema in TOOL_DEFINITIONS]

    assert catalogue["synthetic"] is True
    assert len(catalogue["offers"]) >= 4
    assert names == [
        "product_lookup",
        "segment_comparison",
        "regulatory_constraint_checker",
        "recommendation_formatter",
    ]

    mortgage = canonical_action_context("mortgage_fixed_rate_review", True)
    assert mortgage.is_credit is True
    assert mortgage.is_upsell is True
    with pytest.raises(KeyError):
        canonical_action_context("model_invented_offer", True)
    with pytest.raises(TypeError):
        canonical_action_context("fee_waiver_6m", "false")


def test_segment_comparison_is_deterministic_and_uses_local_synthetic_data():
    target = {
        "customer_id": "IRLBANK_00001",
        "profile": {
            "age": 47,
            "tenure_months": 62,
            "account_type": "Current + Savings",
            "monthly_balance_eur": 4320.50,
            "num_products": 2,
            "monthly_transaction_count": 31,
            "monthly_transaction_amount_eur": 1780.25,
            "has_direct_debits": True,
            "direct_debit_count": 4,
            "uses_digital_bank_secondary": False,
            "was_kbc_ulster_customer": False,
            "months_since_switching": 24,
            "experienced_switching_difficulty": False,
            "branch_visits_monthly": 1,
            "customer_service_calls_6months": 0,
            "has_complaint_history": False,
            "credit_score_band": "High",
            "has_mortgage": False,
            "has_savings_goal": True,
        },
    }
    first = segment_comparison(target)
    second = segment_comparison(target)

    assert first == second
    assert first["cohort_size"] > 0
    assert 0.0 <= first["churn_rate"] <= 1.0
    assert first["cohort_definition"]["age_band"] == "45-59"
    assert first["target_phase1_prediction"]["customer_id"] == "IRLBANK_00001"
    assert first["target_phase1_prediction"]["prediction_method"].startswith(
        "model.predict_proba"
    )
    assert "synthetic" in first["data_note"].lower()


def test_formatter_validates_schema_and_requires_exact_approval():
    checked_customer = customer(churn_probability=0.90)
    checked_action = canonical_action_context("fee_waiver_6m", True)
    decision = regulatory_constraint_checker(checked_customer, checked_action)
    valid = {
        "action": "fee_waiver_6m",
        "justification": "Fee relief addresses service friction without an upsell.",
        "confidence": 0.82,
        "regulatory_flags": [],
        "checker_verdict": "approved",
    }

    output = recommendation_formatter(
        valid,
        customer=checked_customer,
        proposed_action=checked_action,
        policy_decision=decision,
    )
    assert output["regulatory_flags"] == ["HUM-003:human_review_required"]

    with pytest.raises(PolicyGateError):
        recommendation_formatter(
            valid,
            customer=checked_customer,
            proposed_action=checked_action,
            policy_decision=None,
        )
    with pytest.raises(PolicyGateError):
        recommendation_formatter(
            {**valid, "action": "dedicated_service_review"},
            customer=checked_customer,
            proposed_action=checked_action,
            policy_decision=decision,
        )
    for malformed in (
        {key: value for key, value in valid.items() if key != "justification"},
        {**valid, "confidence": "high"},
        {**valid, "confidence": 1.1},
    ):
        with pytest.raises(ValueError):
            recommendation_formatter(
                malformed,
                customer=checked_customer,
                proposed_action=checked_action,
                policy_decision=decision,
            )


def test_blocked_decision_emits_refusal_and_cannot_authorize_an_alternative():
    checked_customer = customer(in_arrears=True, churn_probability=0.90)
    blocked_action = canonical_action_context("mortgage_fixed_rate_review", True)
    decision = regulatory_constraint_checker(checked_customer, blocked_action)
    refusal = {
        "action": "no_recommendation",
        "justification": "The proposed credit action is blocked for this customer.",
        "confidence": 1.0,
        "regulatory_flags": [],
        "checker_verdict": "blocked",
    }
    output = recommendation_formatter(
        refusal,
        customer=checked_customer,
        proposed_action=blocked_action,
        policy_decision=decision,
    )

    assert output["action"] == "no_recommendation"
    assert output["checker_verdict"] == "blocked"
    assert "ARR-001" in output["regulatory_flags"]

    with pytest.raises(PolicyGateError):
        recommendation_formatter(
            {
                **refusal,
                "action": "fee_waiver_6m",
                "checker_verdict": "approved",
            },
            customer=checked_customer,
            proposed_action=blocked_action,
            policy_decision=decision,
        )
