"""Hand-written Groq Chat Completions tool loop with a structural policy gate."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Mapping

from agent.policy_rules import ActionPolicyContext, CustomerPolicyContext, PolicyDecision
from agent.rate_limits import GLOBAL_REQUEST_QUOTA, InMemoryRequestQuota
from agent.tools import (
    GROQ_TOOL_DEFINITIONS,
    Phase1ModelRuntime,
    PolicyGateError,
    canonical_action_context,
    load_phase1_runtime,
    predict_customer_churn_risk,
    product_lookup,
    recommendation_formatter,
    regulatory_constraint_checker,
    segment_comparison,
)
from agent.trace import TraceRecorder


MODEL_NAME = "llama-3.3-70b-versatile"
MAX_LOOP_TURNS = 6
MAX_TOKENS = 1024
MAX_LIVE_API_CALLS = 6
_GROQ_KEY_PREFIX = "gsk_"
_GROQ_KEY_PLACEHOLDERS = frozenset(
    {
        "...",
        "your-free-tier-key",
        "your-groq-api-key",
        "your-groq-key-here",
    }
)

SYSTEM_PROMPT = """You are a retention recommendation agent for a synthetic Irish banking demonstration.
Use product_lookup and segment_comparison before choosing an action. Before any final
action, call regulatory_constraint_checker for that exact action. That tool applies
local synthetic project rules, not law or a bank policy. Then call
recommendation_formatter. If an action is blocked, check a safer alternative or
format a no_recommendation refusal. Never claim regulatory compliance, guaranteed
savings, or product eligibility. Customer facts and catalogue policy metadata are
injected by the runtime and are authoritative within this demonstration. The churn
probability is recomputed from the trained Phase 1 model for the current feature vector
and must not be replaced with a model-authored value. Use only the supplied tools and
pass strict JSON arguments matching their schemas.
"""


class AgentLoopError(RuntimeError):
    """Raised when the scripted/model trajectory cannot yield a governed result."""


class AgentLoopLimitError(AgentLoopError):
    """Raised when the hand-written loop reaches its configured turn limit."""


class LiveModeError(RuntimeError):
    """Raised when a live client safety invariant is absent or invalid."""


class UnsafeClientError(RuntimeError):
    """Raised when a client did not come from an approved mock or live factory."""


class _MockChatCompletions:
    def __init__(self, owner: "ScriptedMockClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        self._owner.calls.append(copy.deepcopy(kwargs))
        if not self._owner.responses:
            raise AgentLoopError("scripted mock has no response remaining")
        return self._owner.responses.pop(0)


class _MockChat:
    def __init__(self, owner: "ScriptedMockClient") -> None:
        self.completions = _MockChatCompletions(owner)


class ScriptedMockClient:
    """Small deterministic OpenAI-compatible fake; this is the default client."""

    _retention_agent_client_kind = "mock"

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.chat = _MockChat(self)


def _validated_groq_api_key(value: Any) -> str | None:
    """Return a plausible Groq key while rejecting examples and malformed values."""

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or candidate.casefold() in _GROQ_KEY_PLACEHOLDERS:
        return None
    if not candidate.startswith(_GROQ_KEY_PREFIX):
        return None
    return candidate


def resolve_groq_api_key(secrets: Any | None = None) -> str | None:
    """Prefer a valid Streamlit secret in production, then the local environment."""

    secret_value = None
    if secrets is not None:
        try:
            secret_value = secrets.get("GROQ_API_KEY")
        except Exception:
            # Streamlit raises when no secrets file exists in local development.
            secret_value = None
    resolved_secret = _validated_groq_api_key(secret_value)
    if resolved_secret is not None:
        return resolved_secret
    return _validated_groq_api_key(os.environ.get("GROQ_API_KEY"))


def _is_retryable(exc: Exception) -> bool:
    """True for transient failures and Groq-generated invalid tool-call JSON."""

    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504, 529):
        return True
    if status == 400:
        body = getattr(exc, "body", None)
        if isinstance(body, Mapping):
            error = body.get("error", body)
            if isinstance(error, Mapping) and "failed_generation" in error:
                return True
        message = str(exc).lower()
        if "failed_generation" in message or "invalid tool call generated" in message:
            return True
    return "timeout" in type(exc).__name__.lower()


_LIVE_GATE_TOKEN = object()


class _GuardedChatCompletions:
    def __init__(self, completions: Any, quota_guard: InMemoryRequestQuota) -> None:
        self._completions = completions
        self._quota_guard = quota_guard
        self.call_count = 0
        self.request_count = 0

    def create(self, **kwargs: Any) -> Any:
        if self.call_count >= MAX_LIVE_API_CALLS:
            raise LiveModeError(
                f"live API call cap reached ({MAX_LIVE_API_CALLS} calls per run)"
            )
        max_tokens = kwargs.get("max_completion_tokens")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_TOKENS:
            raise LiveModeError(
                f"max_completion_tokens must be capped at {MAX_TOKENS}"
            )
        self.call_count += 1
        last_exc = None
        for attempt in range(3):
            self._quota_guard.reserve_request()
            self.request_count += 1
            try:
                return self._completions.create(**kwargs)
            except Exception as exc:
                if _is_retryable(exc) and attempt < 2:
                    last_exc = exc
                    time.sleep(2**attempt)
                    continue
                raise
        raise last_exc  # type: ignore[misc]


class _GuardedChat:
    def __init__(self, chat: Any, quota_guard: InMemoryRequestQuota) -> None:
        self.completions = _GuardedChatCompletions(chat.completions, quota_guard)


class GroqLiveClient:
    """A Groq SDK client reachable only through create_live_client."""

    _retention_agent_client_kind = "groq_live"

    def __init__(
        self,
        client: Any,
        token: object,
        quota_guard: InMemoryRequestQuota,
    ) -> None:
        if token is not _LIVE_GATE_TOKEN:
            raise LiveModeError("live client must be created by create_live_client")
        self.chat = _GuardedChat(client.chat, quota_guard)


def create_live_client(
    *,
    api_key: str,
    quota_guard: InMemoryRequestQuota = GLOBAL_REQUEST_QUOTA,
) -> GroqLiveClient:
    """Construct the only permitted live Groq client.

    The app resolves its server-side key from ``st.secrets`` first and the
    environment second. This factory accepts the resolved value and never logs it.
    """

    validated_key = _validated_groq_api_key(api_key)
    if validated_key is None:
        raise LiveModeError("GROQ_API_KEY is missing, malformed, or still a placeholder")
    from groq import Groq

    return GroqLiveClient(
        Groq(api_key=validated_key),
        _LIVE_GATE_TOKEN,
        quota_guard,
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices", [])
    if not choices:
        raise AgentLoopError("Groq response contained no choices")
    return choices[0]


def _serialise_tool_call(tool_call: Any) -> dict[str, Any]:
    function = _get(tool_call, "function", {})
    call_id = _get(tool_call, "id")
    name = _get(function, "name")
    arguments = _get(function, "arguments", "{}")
    if not isinstance(call_id, str) or not call_id:
        raise AgentLoopError("tool call is missing a non-empty id")
    if not isinstance(name, str) or not name:
        raise AgentLoopError("tool call is missing a function name")
    if isinstance(arguments, Mapping):
        arguments = json.dumps(dict(arguments), separators=(",", ":"))
    if not isinstance(arguments, str):
        raise AgentLoopError("tool call arguments must be a JSON string")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_tool_arguments(arguments: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("tool arguments must decode to an object")
    return payload


def _derive_held_products(profile: Mapping[str, Any]) -> tuple[str, ...]:
    products: set[str] = set()
    account_type = str(profile.get("account_type", ""))
    if "Current" in account_type:
        products.add("current_account")
    if "Savings" in account_type:
        products.add("savings_account")
    if profile.get("has_mortgage") is True:
        products.add("mortgage")
    return tuple(sorted(products))


def customer_policy_context(customer: Mapping[str, Any]) -> CustomerPolicyContext:
    """Build trusted policy facts from the runtime-selected synthetic customer."""

    profile = customer.get("profile", customer)
    governance = customer.get("governance", {})
    if not isinstance(profile, Mapping) or not isinstance(governance, Mapping):
        raise TypeError("customer profile and governance must be mappings")
    held_products = customer.get("held_products")
    if held_products is None:
        held_products = _derive_held_products(profile)
    return CustomerPolicyContext(
        customer_id=str(customer.get("customer_id", profile.get("customer_id", ""))),
        churn_probability=customer["churn_probability"],
        held_products=tuple(held_products),
        in_arrears=governance.get("in_arrears", False),
        vulnerable_customer=governance.get("vulnerable_customer", False),
    )


@dataclass(slots=True)
class _LoopState:
    customer_record: Mapping[str, Any]
    customer_policy: CustomerPolicyContext
    phase1_runtime: Phase1ModelRuntime
    phase1_prediction: Mapping[str, Any]
    latest_action: ActionPolicyContext | None = None
    latest_decision: PolicyDecision | None = None
    final_output: dict[str, Any] | None = None
    executed_tools: list[str] = field(default_factory=list)


def _expect_keys(payload: Mapping[str, Any], allowed: set[str]) -> None:
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(f"unexpected tool input fields: {sorted(unexpected)}")


def _execute_tool(
    name: str,
    tool_input: Mapping[str, Any],
    state: _LoopState,
    trace: TraceRecorder,
) -> Any:
    if not isinstance(tool_input, Mapping):
        raise TypeError("tool input must be an object")
    state.executed_tools.append(name)

    if name == "product_lookup":
        _expect_keys(tool_input, {"category"})
        return product_lookup(tool_input.get("category"))

    if name == "segment_comparison":
        _expect_keys(tool_input, set())
        return segment_comparison(
            state.customer_record,
            phase1_runtime=state.phase1_runtime,
        )

    if name == "regulatory_constraint_checker":
        _expect_keys(tool_input, {"action_id", "requires_human_review"})
        action = canonical_action_context(
            tool_input["action_id"], tool_input["requires_human_review"]
        )
        decision = regulatory_constraint_checker(state.customer_policy, action)
        state.latest_action = action
        state.latest_decision = decision
        payload = decision.as_dict()
        trace.log("gate_check", payload)
        return payload

    if name == "recommendation_formatter":
        if state.latest_action is None or state.latest_decision is None:
            raise PolicyGateError(
                "regulatory_constraint_checker must run before recommendation_formatter"
            )
        output = recommendation_formatter(
            tool_input,
            customer=state.customer_policy,
            proposed_action=state.latest_action,
            policy_decision=state.latest_decision,
        )
        state.final_output = output
        trace.log("final_output", output)
        return output

    raise KeyError(f"unknown tool: {name}")


def _blocked_refusal(state: _LoopState, trace: TraceRecorder) -> dict[str, Any] | None:
    decision = state.latest_decision
    action = state.latest_action
    if decision is None or action is None or decision.passed:
        return None
    reasons = [result.reason for result in decision.rule_results if not result.passed]
    candidate = {
        "action": "no_recommendation",
        "justification": " ".join(reasons),
        "confidence": 1.0,
        "regulatory_flags": [],
        "checker_verdict": "blocked",
    }
    output = recommendation_formatter(
        candidate,
        customer=state.customer_policy,
        proposed_action=action,
        policy_decision=decision,
    )
    trace.log("final_output", output)
    return output


def _safe_client(client: Any | None) -> Any:
    selected = ScriptedMockClient() if client is None else client
    if type(selected) not in {ScriptedMockClient, GroqLiveClient}:
        raise UnsafeClientError(
            "client must be ScriptedMockClient or created by create_live_client"
        )
    return selected


def _completed_result(
    state: _LoopState,
    trace: TraceRecorder,
    turns: int,
) -> dict[str, Any] | None:
    final_output = state.final_output or _blocked_refusal(state, trace)
    if final_output is None:
        return None
    return {
        "recommendation": final_output,
        "trace": trace.as_list(),
        "turns": turns,
        "model": MODEL_NAME,
        "customer": copy.deepcopy(dict(state.customer_record)),
        "phase1_prediction": copy.deepcopy(dict(state.phase1_prediction)),
    }


def run_retention_agent(
    customer: Mapping[str, Any],
    *,
    client: Any | None = None,
    phase1_runtime: Phase1ModelRuntime | None = None,
    max_turns: int = MAX_LOOP_TURNS,
    clock: Any | None = None,
) -> dict[str, Any]:
    """Run a bounded Groq-compatible tool loop and return a governed output."""

    if not isinstance(max_turns, int) or not 1 <= max_turns <= MAX_LOOP_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_LOOP_TURNS}")
    safe_client = _safe_client(client)
    runtime = phase1_runtime or load_phase1_runtime()
    phase1_prediction = predict_customer_churn_risk(
        customer, phase1_runtime=runtime
    )
    authoritative_customer = copy.deepcopy(dict(customer))
    authoritative_customer["churn_probability"] = phase1_prediction[
        "churn_probability"
    ]
    authoritative_customer["phase1_prediction"] = copy.deepcopy(
        phase1_prediction
    )
    trace = TraceRecorder() if clock is None else TraceRecorder(clock)
    state = _LoopState(
        authoritative_customer,
        customer_policy_context(authoritative_customer),
        runtime,
        phase1_prediction,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Produce a governed retention recommendation for this synthetic "
                "customer. The Phase 1 prediction metadata proves the runtime model "
                "call and is authoritative:\n"
                f"{json.dumps(authoritative_customer, sort_keys=True, default=str)}"
            ),
        },
    ]

    turn = 0
    while turn < max_turns:
        turn += 1
        response = safe_client.chat.completions.create(
            model=MODEL_NAME,
            max_completion_tokens=MAX_TOKENS,
            temperature=0,
            messages=messages,
            tools=list(GROQ_TOOL_DEFINITIONS),
            tool_choice="auto",
        )
        choice = _first_choice(response)
        message = _get(choice, "message", {})
        finish_reason = _get(choice, "finish_reason")
        text_content = _get(message, "content")
        if text_content is not None and not isinstance(text_content, str):
            raise AgentLoopError("assistant message content must be text or null")

        raw_tool_calls = _get(message, "tool_calls", []) or []
        tool_calls = [_serialise_tool_call(call) for call in raw_tool_calls]
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": text_content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if text_content:
            trace.log("model_thought", {"text": text_content})

        if finish_reason == "tool_calls":
            if not tool_calls:
                raise AgentLoopError(
                    "tool_calls finish reason contained no function calls"
                )
            for tool_call in tool_calls:
                tool_id = tool_call["id"]
                function = tool_call["function"]
                name = function["name"]
                arguments = function["arguments"]
                try:
                    tool_input = _parse_tool_arguments(arguments)
                    displayed_input = dict(tool_input)
                    argument_error = None
                except Exception as exc:
                    tool_input = {}
                    displayed_input = {"raw_arguments": arguments}
                    argument_error = exc

                trace.log(
                    "tool_call",
                    {
                        "tool_use_id": tool_id,
                        "name": name,
                        "input": displayed_input,
                    },
                )
                try:
                    if argument_error is not None:
                        raise argument_error
                    result = _execute_tool(name, tool_input, state, trace)
                    is_error = False
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    is_error = True
                trace.log(
                    "tool_result",
                    {
                        "tool_use_id": tool_id,
                        "name": name,
                        "is_error": is_error,
                        "result": result,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": name,
                        "content": json.dumps(result, sort_keys=True, default=str),
                    }
                )
            continue

        if finish_reason == "stop":
            completed = _completed_result(state, trace, turn)
            if completed is None:
                raise AgentLoopError(
                    "model ended without a formatter result or a checked blocked action"
                )
            return completed

        raise AgentLoopError(f"unsupported finish reason: {finish_reason!r}")

    completed = _completed_result(state, trace, max_turns)
    if completed is not None:
        return completed
    raise AgentLoopLimitError(f"agent loop reached the {max_turns}-turn cap")
