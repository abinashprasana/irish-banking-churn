from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_a_runtime_contracts_capture_the_integration_boundary():
    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    loop_source = (PROJECT_ROOT / "agent" / "loop.py").read_text(encoding="utf-8")
    policy_source = (PROJECT_ROOT / "agent" / "policy_rules.py").read_text(
        encoding="utf-8"
    )
    tools_source = (PROJECT_ROOT / "agent" / "tools.py").read_text(
        encoding="utf-8"
    )
    limits_source = (PROJECT_ROOT / "agent" / "rate_limits.py").read_text(
        encoding="utf-8"
    )
    model_card = (PROJECT_ROOT / "model_card.md").read_text(encoding="utf-8")

    assert "predict_proba(X_cast)" in app_source
    assert '"prediction_method": "model.predict_proba(feature_vector)[0, 1]"' in tools_source
    assert "in_arrears" in policy_source
    assert "vulnerable_customer" in policy_source
    assert "structured refusal" in tools_source
    assert "failed_rule_ids" in policy_source
    assert 'MODEL_NAME = "qwen/qwen3.6-27b"' in loop_source
    assert "DAILY_SAFETY_MARGIN" in limits_source
    assert "SESSION_RUN_CAP" in limits_source
    assert "EU AI Act Article 86" in model_card
    assert "not evidence of legal compliance" in model_card
    assert "EBA" + "-compliant" not in model_card
    assert "AI Act" + " compliant" not in model_card
