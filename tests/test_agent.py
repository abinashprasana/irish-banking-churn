import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
import pytest

from agent.loop import (
    AgentLoopError,
    AgentLoopLimitError,
    LiveModeError,
    MAX_LIVE_API_CALLS,
    MAX_TOKENS,
    MODEL_NAME,
    ScriptedMockClient,
    UnsafeClientError,
    create_live_client,
    resolve_groq_api_key,
    run_retention_agent,
)
from agent.rate_limits import (
    DAILY_LIMIT_MESSAGE,
    DAILY_REQUEST_CAP,
    InMemoryRequestQuota,
    MINUTE_LIMIT_MESSAGE,
    RateLimitSafetyError,
    SESSION_RUN_CAP,
    SessionRunLimitError,
    reserve_session_run,
)
from agent.tools import GROQ_TOOL_DEFINITIONS


FIXED_TIME = lambda: "2026-07-13T12:00:00Z"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_customer(**governance):
    return {
        "customer_id": "IRLBANK_DEMO",
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
        "churn_probability": 0.88,
        "churn_drivers": [
            {
                "feature": "num_products",
                "value": 1,
                "shap_value": 2.1,
                "direction": "increases_churn",
            }
        ],
        "governance": {
            "in_arrears": governance.get("in_arrears", False),
            "vulnerable_customer": governance.get("vulnerable_customer", False),
        },
    }


def response(finish_reason, content=None, *tool_calls):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": list(tool_calls),
                },
            }
        ]
    }


def tool_call(tool_id, name, tool_input):
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(tool_input, separators=(",", ":")),
        },
    }


def test_trajectory_executes_multiple_tool_calls_and_preserves_matching_ids():
    client = ScriptedMockClient(
        [
            response(
                "tool_calls",
                "Inspect offers and the comparable cohort.",
                tool_call("call_products", "product_lookup", {"category": "fee_relief"}),
                tool_call("call_segment", "segment_comparison", {}),
            ),
            response(
                "tool_calls",
                None,
                tool_call(
                    "call_gate",
                    "regulatory_constraint_checker",
                    {"action_id": "fee_waiver_6m", "requires_human_review": True},
                ),
            ),
            response(
                "tool_calls",
                None,
                tool_call(
                    "call_format",
                    "recommendation_formatter",
                    {
                        "action": "fee_waiver_6m",
                        "justification": "Fee relief addresses switching friction without an upsell.",
                        "confidence": 0.83,
                        "regulatory_flags": [],
                        "checker_verdict": "approved",
                    },
                ),
            ),
            response("stop", "Recommendation complete."),
        ]
    )

    result = run_retention_agent(synthetic_customer(), client=client, clock=FIXED_TIME)

    assert result["recommendation"]["action"] == "fee_waiver_6m"
    assert result["model"] == MODEL_NAME
    assert len(client.calls) == 4
    assert client.calls[0]["max_completion_tokens"] == MAX_TOKENS
    assert client.calls[0]["tools"] == list(GROQ_TOOL_DEFINITIONS)
    assert all(tool["type"] == "function" for tool in client.calls[0]["tools"])
    second_request_messages = client.calls[1]["messages"]
    tool_results = [
        message for message in second_request_messages if message["role"] == "tool"
    ]
    assert [item["tool_call_id"] for item in tool_results] == [
        "call_products",
        "call_segment",
    ]
    trace = result["trace"]
    assert [event["step"] for event in trace] == list(range(1, len(trace) + 1))
    assert all(set(event) == {"step", "type", "content", "timestamp"} for event in trace)
    assert [
        event["content"]["name"]
        for event in trace
        if event["type"] == "tool_call"
    ] == [
        "product_lookup",
        "segment_comparison",
        "regulatory_constraint_checker",
        "recommendation_formatter",
    ]


def test_actual_groq_sdk_contract_runs_full_tool_trajectory_without_network(
    monkeypatch,
):
    from groq import Groq as GroqSDK

    scripted = [
        response(
            "tool_calls",
            "Inspect the local inputs.",
            tool_call("sdk_products", "product_lookup", {"category": "fee_relief"}),
            tool_call("sdk_segment", "segment_comparison", {}),
        ),
        response(
            "tool_calls",
            None,
            tool_call(
                "sdk_gate",
                "regulatory_constraint_checker",
                {"action_id": "fee_waiver_6m", "requires_human_review": True},
            ),
        ),
        response(
            "tool_calls",
            None,
            tool_call(
                "sdk_formatter",
                "recommendation_formatter",
                {
                    "action": "fee_waiver_6m",
                    "justification": "A human-reviewed fee waiver addresses friction.",
                    "confidence": 0.82,
                    "regulatory_flags": [],
                    "checker_verdict": "approved",
                },
            ),
        ),
        response("stop", "Complete."),
    ]
    captured_requests = []

    def handler(request):
        captured_requests.append(json.loads(request.content.decode("utf-8")))
        payload = scripted.pop(0)
        payload.update(
            {
                "id": f"chatcmpl-{len(captured_requests)}",
                "object": "chat.completion",
                "created": 1785090000,
                "model": MODEL_NAME,
            }
        )
        payload["choices"][0]["index"] = 0
        return httpx.Response(200, json=payload, request=request)

    sdk_client = GroqSDK(
        api_key="gsk_test_only",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr("groq.Groq", lambda *, api_key: sdk_client)
    quota = InMemoryRequestQuota(requests_per_minute=30, daily_request_cap=20)
    live_client = create_live_client(api_key="gsk_test_only", quota_guard=quota)

    result = run_retention_agent(synthetic_customer(), client=live_client)

    assert result["recommendation"]["action"] == "fee_waiver_6m"
    assert len(captured_requests) == 4
    assert captured_requests[0]["model"] == MODEL_NAME
    assert captured_requests[0]["tools"] == list(GROQ_TOOL_DEFINITIONS)
    assert [
        message["tool_call_id"]
        for message in captured_requests[1]["messages"]
        if message["role"] == "tool"
    ] == ["sdk_products", "sdk_segment"]
    assert live_client.chat.completions.request_count == 4


def test_formatter_before_checker_returns_tool_error_and_no_final_output():
    client = ScriptedMockClient(
        [
            response(
                "tool_calls",
                None,
                tool_call(
                    "call_early",
                    "recommendation_formatter",
                    {
                        "action": "fee_waiver_6m",
                        "justification": "Unchecked output.",
                        "confidence": 0.9,
                        "regulatory_flags": [],
                        "checker_verdict": "approved",
                    },
                ),
            ),
            response("stop", "Done."),
        ]
    )

    with pytest.raises(AgentLoopError, match="ended without"):
        run_retention_agent(synthetic_customer(), client=client, clock=FIXED_TIME)
    tool_result = client.calls[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_early"
    assert "PolicyGateError" in json.loads(tool_result["content"])["error"]


def test_loop_terminates_at_configured_turn_cap():
    endless = [
        response(
            "tool_calls",
            None,
            tool_call(f"call_{index}", "product_lookup", {}),
        )
        for index in range(2)
    ]
    client = ScriptedMockClient(endless)

    with pytest.raises(AgentLoopLimitError, match="2-turn cap"):
        run_retention_agent(
            synthetic_customer(), client=client, max_turns=2, clock=FIXED_TIME
        )
    assert len(client.calls) == 2


def test_fully_blocked_customer_yields_structured_refusal_not_exception():
    client = ScriptedMockClient(
        [
            response(
                "tool_calls",
                None,
                tool_call(
                    "call_block",
                    "regulatory_constraint_checker",
                    {
                        "action_id": "mortgage_fixed_rate_review",
                        "requires_human_review": True,
                    },
                ),
            ),
            response(
                "tool_calls",
                None,
                tool_call(
                    "call_refusal",
                    "recommendation_formatter",
                    {
                        "action": "no_recommendation",
                        "justification": "Credit upsell is prohibited for this customer.",
                        "confidence": 1.0,
                        "regulatory_flags": [],
                        "checker_verdict": "blocked",
                    },
                ),
            ),
            response("stop", "Refusal complete."),
        ]
    )

    result = run_retention_agent(
        synthetic_customer(in_arrears=True, vulnerable_customer=True),
        client=client,
        clock=FIXED_TIME,
    )

    final = result["recommendation"]
    assert final["action"] == "no_recommendation"
    assert final["checker_verdict"] == "blocked"
    assert {"ARR-001", "VUL-004"}.issubset(final["regulatory_flags"])


def test_live_client_is_groq_guarded_and_mock_remains_default(monkeypatch):
    default_client = ScriptedMockClient()
    assert default_client._retention_agent_client_kind == "mock"
    with pytest.raises(AgentLoopError, match="scripted mock"):
        run_retention_agent(synthetic_customer())

    with pytest.raises(LiveModeError, match="GROQ_API_KEY"):
        create_live_client(api_key="")

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return response("stop", "")

    class FakeGroq:
        def __init__(self, *, api_key):
            assert api_key == "gsk_never_used_test_value"
            self.chat = type("FakeChat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("groq.Groq", FakeGroq)
    quota = InMemoryRequestQuota(requests_per_minute=100, daily_request_cap=100)
    guarded = create_live_client(
        api_key="gsk_never_used_test_value",
        quota_guard=quota,
    )
    with pytest.raises(LiveModeError, match="max_completion_tokens"):
        guarded.chat.completions.create(max_completion_tokens=MAX_TOKENS + 1)
    for _ in range(MAX_LIVE_API_CALLS):
        guarded.chat.completions.create(max_completion_tokens=MAX_TOKENS)
    with pytest.raises(LiveModeError, match="call cap"):
        guarded.chat.completions.create(max_completion_tokens=MAX_TOKENS)
    assert quota.snapshot()["daily_requests"] == MAX_LIVE_API_CALLS

    class ArbitraryClient:
        _retention_agent_client_kind = "mock"
        chat = object()

    with pytest.raises(UnsafeClientError):
        run_retention_agent(synthetic_customer(), client=ArbitraryClient())


def test_groq_invalid_generated_tool_call_is_retried_and_counted(monkeypatch):
    class InvalidToolGeneration(Exception):
        status_code = 400
        body = {
            "error": {
                "message": "Invalid tool call generated",
                "failed_generation": {"reason": "arguments are not valid JSON"},
            }
        }

    class RetryCompletions:
        def __init__(self):
            self.attempts = 0

        def create(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise InvalidToolGeneration("failed_generation")
            return response("stop", "")

    completions = RetryCompletions()

    class RetryGroq:
        def __init__(self, *, api_key):
            self.chat = type("RetryChat", (), {"completions": completions})()

    monkeypatch.setattr("groq.Groq", RetryGroq)
    monkeypatch.setattr("agent.loop.time.sleep", lambda _: None)
    quota = InMemoryRequestQuota(requests_per_minute=30, daily_request_cap=10)
    client = create_live_client(api_key="gsk_test_only", quota_guard=quota)
    client.chat.completions.create(max_completion_tokens=MAX_TOKENS)

    assert completions.attempts == 2
    assert client.chat.completions.call_count == 1
    assert client.chat.completions.request_count == 2
    assert quota.snapshot()["daily_requests"] == 2


def test_key_resolution_and_free_tier_quota_guards(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_local_env_key")
    assert resolve_groq_api_key({"GROQ_API_KEY": "gsk_production_secret"}) == (
        "gsk_production_secret"
    )
    assert resolve_groq_api_key({}) == "gsk_local_env_key"
    assert resolve_groq_api_key({"GROQ_API_KEY": "your-groq-key-here"}) == (
        "gsk_local_env_key"
    )
    assert resolve_groq_api_key({"GROQ_API_KEY": "not-a-groq-key"}) == (
        "gsk_local_env_key"
    )
    monkeypatch.delenv("GROQ_API_KEY")
    assert resolve_groq_api_key({}) is None
    assert resolve_groq_api_key({"GROQ_API_KEY": "your-groq-key-here"}) is None
    assert resolve_groq_api_key({"GROQ_API_KEY": "not-a-groq-key"}) is None

    assert DAILY_REQUEST_CAP == 950
    now = [datetime(2026, 7, 26, tzinfo=timezone.utc)]
    ticks = [0.0]
    quota = InMemoryRequestQuota(
        requests_per_minute=30,
        daily_request_cap=2,
        now=lambda: now[0],
        monotonic=lambda: ticks[0],
    )
    quota.reserve_request()
    ticks[0] += 1
    quota.reserve_request()
    with pytest.raises(RateLimitSafetyError, match=DAILY_LIMIT_MESSAGE):
        quota.reserve_request()
    with pytest.raises(RateLimitSafetyError, match=DAILY_LIMIT_MESSAGE):
        quota.ensure_run_available()

    ticks[0] = 0.0
    minute_quota = InMemoryRequestQuota(
        requests_per_minute=2,
        daily_request_cap=10,
        now=lambda: now[0],
        monotonic=lambda: ticks[0],
    )
    minute_quota.reserve_request()
    ticks[0] += 1
    minute_quota.reserve_request()
    with pytest.raises(RateLimitSafetyError, match=MINUTE_LIMIT_MESSAGE):
        minute_quota.reserve_request()
    ticks[0] = 61.0
    minute_quota.reserve_request()

    session = {}
    for expected_remaining in range(SESSION_RUN_CAP - 1, -1, -1):
        assert reserve_session_run(session) == expected_remaining
    with pytest.raises(SessionRunLimitError, match="session demo limit"):
        reserve_session_run(session)


def test_recorded_demos_and_phase2_wiring_remain_valid():
    demo_paths = sorted((PROJECT_ROOT / "demo_traces").glob("*.json"))
    assert len(demo_paths) == 4
    blocked_count = 0
    expected_types = {
        "model_thought",
        "tool_call",
        "tool_result",
        "gate_check",
        "final_output",
    }
    expected_tools = {
        "product_lookup",
        "segment_comparison",
        "regulatory_constraint_checker",
        "recommendation_formatter",
    }
    for demo_path in demo_paths:
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        assert set(demo) == {
            "schema_version",
            "demo_id",
            "title",
            "recording",
            "customer",
            "trace",
            "recommendation",
        }
        assert isinstance(demo["recording"]["real_api_calls"], int)
        assert demo["recording"]["real_api_calls"] >= 0
        assert demo["recording"]["model"] == MODEL_NAME
        trace = demo["trace"]
        assert [event["step"] for event in trace] == list(range(1, len(trace) + 1))
        assert all(
            set(event) == {"step", "type", "content", "timestamp"}
            for event in trace
        )
        assert {event["type"] for event in trace} == expected_types
        calls = {
            event["content"]["tool_use_id"]: event["content"]["name"]
            for event in trace
            if event["type"] == "tool_call"
        }
        results = {
            event["content"]["tool_use_id"]: event["content"]["name"]
            for event in trace
            if event["type"] == "tool_result"
        }
        assert calls == results
        assert set(calls.values()) == expected_tools
        if demo["recommendation"]["checker_verdict"] == "blocked":
            blocked_count += 1
            assert demo["recommendation"]["action"] == "no_recommendation"
    assert blocked_count >= 2

    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(app_source)
    assert '"Retention agent"' in app_source
    assert "Bring your own" not in app_source
    assert "GROQ_API_KEY" in app_source
    assert "SESSION_RUN_CAP" in app_source

    recorder_source = (PROJECT_ROOT / "scripts" / "record_demo_runs.py").read_text(
        encoding="utf-8"
    )
    ast.parse(recorder_source)
    assert "GROQ_API_KEY" in recorder_source

    loop_source = (PROJECT_ROOT / "agent" / "loop.py").read_text(encoding="utf-8")
    assert "while turn < max_turns" in loop_source
