import io
import json
import urllib.error

import pytest

from agent import providers
from agent.loop import (
    GeminiLiveClient,
    LiveModeError,
    MODEL_NAME,
    create_live_client,
    run_retention_agent,
)
from agent.providers import (
    GEMINI_CHAT_URL,
    GEMINI_MODEL_NAME,
    GeminiAPIError,
    load_env_file,
    resolve_gemini_api_key,
)
from agent.rate_limits import InMemoryRequestQuota
from redteam.scenarios import build_customer


def _response(message, finish_reason):
    return {"choices": [{"finish_reason": finish_reason, "message": message}], "model": GEMINI_MODEL_NAME}


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
        "extra_content": {"google": {"thought_signature": f"sig-{call_id}"}},
    }


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_gemini_runs_the_same_governed_loop_through_the_compatible_endpoint(monkeypatch):
    scripted = [
        _response({"content": None, "tool_calls": [_call("c1", "product_lookup", {"category": "fee_relief"})]}, "tool_calls"),
        _response(
            {"content": None, "tool_calls": [_call("c2", "regulatory_constraint_checker", {"action_id": "fee_waiver_6m", "requires_human_review": True})]},
            "tool_calls",
        ),
        _response(
            {
                "content": None,
                "tool_calls": [
                    _call(
                        "c3",
                        "recommendation_formatter",
                        {
                            "action": "fee_waiver_6m",
                            "justification": "Fee relief for a current account holder, with advisor review.",
                            "confidence": 0.8,
                            "regulatory_flags": [],
                            "checker_verdict": "approved",
                        },
                    )
                ],
            },
            "tool_calls",
        ),
        _response({"content": "Done.", "tool_calls": []}, "stop"),
    ]
    sent = []

    def fake_urlopen(request, timeout):
        sent.append((request.full_url, dict(request.header_items()), json.loads(request.data)))
        return FakeResponse(json.dumps(scripted.pop(0)).encode())

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    quota = InMemoryRequestQuota(requests_per_minute=10, daily_request_cap=10)
    client = create_live_client(api_key="test-gemini-key", provider="gemini", quota_guard=quota)
    assert type(client) is GeminiLiveClient

    result = run_retention_agent(build_customer("01_allowed_fee_waiver"), client=client)

    assert result["model"] == GEMINI_MODEL_NAME != MODEL_NAME
    assert result["recommendation"]["action"] == "fee_waiver_6m"
    assert result["recommendation"]["regulatory_flags"] == ["HUM-003:human_review_required"]
    assert client.chat.completions.request_count == 4
    # The thought signature from turn one goes back unchanged on turn two.
    echoed = [m for m in sent[1][2]["messages"] if m["role"] == "assistant"][0]["tool_calls"][0]
    assert echoed["extra_content"] == {"google": {"thought_signature": "sig-c1"}}
    url, headers, body = sent[0]
    assert url == GEMINI_CHAT_URL
    assert headers["Authorization"] == "Bearer test-gemini-key"
    assert body["model"] == GEMINI_MODEL_NAME and body["temperature"] == 0


def test_gemini_errors_are_retryable_shaped_and_never_echo_the_key(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(GEMINI_CHAT_URL, 400, "bad", {}, io.BytesIO(b"bad key test-gemini-key"))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    client = providers.GeminiCompatClient("test-gemini-key")
    with pytest.raises(GeminiAPIError) as caught:
        client.chat.completions.create(model=GEMINI_MODEL_NAME)
    assert caught.value.status_code == 400
    assert "test-gemini-key" not in str(caught.value)


def test_gemini_key_resolution_and_env_file_loading(tmp_path, monkeypatch):
    # Register the variable so monkeypatch removes whatever load_env_file sets.
    monkeypatch.setenv("GEMINI_API_KEY", "registered")
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(LiveModeError, match="GEMINI_API_KEY"):
        create_live_client(api_key="  ", provider="gemini")
    with pytest.raises(LiveModeError, match="unknown live provider"):
        create_live_client(api_key="x", provider="other")
    assert resolve_gemini_api_key() is None
    assert resolve_gemini_api_key({"GEMINI_API_KEY": "secret-from-streamlit"}) == "secret-from-streamlit"

    env = tmp_path / ".env"
    env.write_text('# comment\nGEMINI_API_KEY="from-file"\nGROQ_API_KEY=\n', encoding="utf-8")
    assert load_env_file(env) == ["GEMINI_API_KEY"]
    assert resolve_gemini_api_key() == "from-file"
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    assert load_env_file(env) == []
    assert resolve_gemini_api_key() == "from-shell"


def test_gemini_model_override_is_validated_and_groq_stays_fixed():
    client = create_live_client(api_key="k", provider="gemini", model="gemini-3.5-flash-lite")
    assert client.model_name == "gemini-3.5-flash-lite"
    assert create_live_client(api_key="k", provider="gemini").model_name == GEMINI_MODEL_NAME
    with pytest.raises(LiveModeError, match="unsupported Gemini model"):
        create_live_client(api_key="k", provider="gemini", model="gpt-4o")
    with pytest.raises(LiveModeError, match="only the configured model"):
        create_live_client(api_key="gsk_test_only", provider="groq", model="other")


def test_paced_quota_keeps_a_daily_count_per_model(tmp_path, monkeypatch):
    from redteam import live

    monkeypatch.setattr(live, "QUOTA_STATE_PATH", tmp_path / ".redteam_quota.json")
    first = live.PacedQuota(5, live.UsageLedger(), provider="gemini", scope="gemini-3.5-flash")
    second = live.PacedQuota(5, live.UsageLedger(), provider="gemini", scope="gemini-3.1-flash-lite")
    first.reserve_request()
    first.reserve_request()
    second.reserve_request()
    assert '"requests": 2' in (tmp_path / ".redteam_quota_gemini_gemini-3.5-flash.json").read_text(encoding="utf-8")
    assert '"requests": 1' in (tmp_path / ".redteam_quota_gemini_gemini-3.1-flash-lite.json").read_text(encoding="utf-8")


def test_quota_file_name_is_safe_for_model_ids_with_slashes(tmp_path, monkeypatch):
    from redteam import live

    monkeypatch.setattr(live, "QUOTA_STATE_PATH", tmp_path / ".redteam_quota.json")
    quota = live.PacedQuota(1, live.UsageLedger(), provider="groq", scope="qwen/qwen3.6-27b")
    quota.reserve_request()
    assert (tmp_path / ".redteam_quota_groq_qwen_qwen3.6-27b.json").is_file()
