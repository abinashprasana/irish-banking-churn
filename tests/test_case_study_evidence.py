import json
from pathlib import Path

from scripts import export_case_study


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = PROJECT_ROOT / "web" / "src" / "data" / "evidence.generated.json"


def test_case_study_evidence_schema_is_sanitized_and_complete():
    bundle = export_case_study.build_bundle()

    assert set(bundle) == {
        "schemaVersion",
        "generatedAt",
        "generatedFrom",
        "evidence",
        "scenarios",
    }
    assert bundle["schemaVersion"] == "1.0"
    assert bundle["evidence"]["project"]["name"] == "Atlantic Ledger"
    assert bundle["evidence"]["dataset"]["recordCount"] == 10_000
    assert bundle["evidence"]["dataset"]["featureCount"] == 19
    assert bundle["evidence"]["model"]["artifact"] == "xgboost_churn_model.pkl"
    assert bundle["evidence"]["agent"]["modelId"] == "qwen/qwen3.6-27b"
    assert len(bundle["evidence"]["governance"]["rules"]) == 4
    assert bundle["evidence"]["verification"]["scenariosPassed"] == 4
    assert bundle["evidence"]["verification"]["blockedOutcomes"] == 2
    assert len(bundle["scenarios"]) == 4
    assert {scenario["mode"] for scenario in bundle["scenarios"]} == {"recorded"}
    assert {
        scenario["recording"]["reasoningSource"] for scenario in bundle["scenarios"]
    } == {"scripted_fixture"}
    assert {
        scenario["recording"]["modelOutputCaptured"]
        for scenario in bundle["scenarios"]
    } == {False}

    serialized = json.dumps(bundle)
    for excluded in (
        "model_thought",
        "encoded_feature_vector",
        "customer_fingerprint",
        "action_fingerprint",
        "tool_use_id",
        "timestamp",
    ):
        assert excluded not in serialized


def test_generated_case_study_evidence_has_not_drifted():
    assert EVIDENCE_PATH.read_text(encoding="utf-8") == export_case_study.render_bundle()
