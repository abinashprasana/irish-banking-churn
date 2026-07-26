from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_a_documents_capture_the_integration_boundary():
    understanding = (PROJECT_ROOT / "UNDERSTANDING.md").read_text(encoding="utf-8")
    progress = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Exact Phase 1 input and output contract" in understanding
    assert "predict_proba(X)" in understanding
    assert "in_arrears" in understanding
    assert "vulnerable_customer" in understanding
    assert "Never call a paid API" in progress
    assert "The Retention Agent Can Refuse" in readme
    assert readme.index("failed_rule_ids") < readme.index("The four tools")
    assert "local tool-calling guide" in readme
    assert "llama-3.3-70b-versatile" in readme
    assert "rate-limit reference" in readme
    assert "AI Act: implications for the EU banking and payments sector" in readme
    assert "EBA" + "-compliant" not in readme
    assert "AI Act" + " compliant" not in readme
