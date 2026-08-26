"""Production-level AppTest coverage for the premium Streamlit lab shell.

The React component is replaced with a synchronous test double here because
Streamlit Components v2 registrations are scoped to one AppTest script runner
and are not re-registered when Python reuses the imported wrapper on a rerun.
The component contract itself has dedicated unit tests; these tests exercise
the real ``app.py`` routing, state, widgets, model bootstrap, and error paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import agent.loop as agent_loop
import agent.tools as agent_tools
import lab_ui.decision_instrument as decision_instrument


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
BLOCKED_RECORDING = "Local arrears rule blocks a credit-related mortgage action"


@pytest.fixture(autouse=True)
def stable_component_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep AppTest reruns focused on the Python integration contract."""

    def component_double(**kwargs):
        return SimpleNamespace(selected_step=kwargs["data"]["selected_step"])

    monkeypatch.setattr(decision_instrument, "_COMPONENT", component_double)


def _app_for(workspace: str) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["lab_workspace"] = workspace
    return app


def _demo_customer() -> dict:
    demo_path = sorted((PROJECT_ROOT / "demo_traces").glob("*.json"))[0]
    return json.loads(demo_path.read_text(encoding="utf-8"))["customer"]


def _html(app: AppTest) -> str:
    return "\n".join(markdown.value for markdown in app.markdown)


@pytest.mark.parametrize(
    "workspace",
    ["Case review", "Decision gate", "Model evidence", "Data & limits"],
)
def test_keyed_tabs_render_only_the_active_workspace(workspace: str) -> None:
    app = _app_for(workspace).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Case review",
        "Decision gate",
        "Model evidence",
        "Data & limits",
    ]
    assert app.session_state["lab_workspace"] == workspace
    assert [tab.label for tab in app.tabs if tab.children] == [workspace]


def test_missing_secret_keeps_current_case_safe_and_offline() -> None:
    app = _app_for("Decision gate")
    customer = _demo_customer()
    app.session_state["phase1_selected_customer"] = customer

    app.run()

    source = next(
        control
        for control in app.segmented_control
        if control.label == "Evidence source"
    )
    run_button = app.button(key="run_governed_recommendation")
    rendered = _html(app)

    assert not app.exception
    assert not app.error
    assert source.value == "Current case"
    assert source.options == ["Current case", "Recorded replay"]
    assert run_button.disabled is True
    assert "Live run unavailable" in rendered
    assert "No valid deployment key is configured" in rendered
    assert app.session_state["phase1_selected_customer"] == customer


def test_recorded_block_is_a_governed_fallback_not_an_app_error() -> None:
    app = _app_for("Decision gate")
    app.session_state["decision_gate_recording"] = BLOCKED_RECORDING

    app.run()

    source = next(
        control
        for control in app.segmented_control
        if control.label == "Evidence source"
    )
    scenario = next(
        control
        for control in app.selectbox
        if control.label == "Recorded governed scenario"
    )
    rendered = _html(app)

    assert not app.exception
    assert not app.error
    assert source.value == "Recorded replay"
    assert source.options == ["Recorded replay"]
    assert scenario.value == BLOCKED_RECORDING
    assert "Recorded" in rendered and "zero requests" in rendered
    assert "Governed block" in rendered
    assert "no action may proceed" in rendered
    assert [heading.value for heading in app.subheader] == [
        "Decision timeline",
        "Policy ledger",
    ]


def test_schema_failure_is_reported_without_losing_the_ready_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_schema(*_args, **_kwargs):
        raise agent_tools.Phase1SchemaError("forced schema mismatch")

    monkeypatch.setattr(agent_tools, "predict_customer_churn_risk", fail_schema)
    app = _app_for("Case review").run()

    app.button(key="case_predict_churn_risk").click().run()

    assert not app.exception
    assert [error.value for error in app.error] == [
        "Phase 1 feature schema validation failed: forced schema mismatch"
    ]
    assert "case_assessment" not in app.session_state
    assert app.button(key="case_predict_churn_risk").disabled is False


def test_live_runtime_failure_stops_safely_and_retains_the_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_local_env_key")

    def fail_client(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(agent_loop, "create_live_client", fail_client)
    app = _app_for("Decision gate")
    customer = _demo_customer()
    app.session_state["phase1_selected_customer"] = customer
    app.run()

    assert app.button(key="run_governed_recommendation").disabled is False
    app.button(key="run_governed_recommendation").click().run()

    assert not app.exception
    assert [error.value for error in app.error] == [
        "Live run stopped safely: provider unavailable"
    ]
    assert app.session_state["phase1_selected_customer"] == customer
    assert "retention_live_result" not in app.session_state
