"""Hand-written raw Anthropic Messages API tool loop with a structural policy gate."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import os
from typing import Any, Mapping

from agent.policy_rules import ActionPolicyContext, CustomerPolicyContext, PolicyDecision
from agent.tools import (
    PolicyGateError,
    TOOL_DEFINITIONS,
    canonical_action_context,
    product_lookup,
    recommendation_formatter,
    regulatory_constraint_checker,
    segment_comparison,
)
from agent.trace import TraceRecorder


MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_LOOP_TURNS = 10
MAX_TOKENS = 256
MAX_LIVE_API_CALLS = 6
LIVE_CONFIRMATION_PHRASE = "RUN LIVE RETENTION AGENT"
ESTIMATED_FOUR_RUN_COST_USD = 0.049

SYSTEM_PROMPT = """You are a retention recommendation agent for a synthetic Irish banking demonstration.
Use product_lookup and segment_comparison as needed. Before any final action, call
regulatory_constraint_checker for that exact action. Then call recommendation_formatter.
If an action is blocked, check a safer alternative or format a no_recommendation refusal.
Never claim regulatory compliance, guaranteed savings, or product eligibility.
Customer facts and catalogue policy metadata are injected by the runtime and are authoritative.
"""


class AgentLoopError(RuntimeError):
    """Raised when the scripted/model trajectory cannot yield a governed result."""


class AgentLoopLimitError(AgentLoopError):
    """Raised when the hand-written loop reaches its configured turn limit."""


class LiveModeError(RuntimeError):
    """Raised when any live-mode guard is absent or invalid."""


class UnsafeClientError(RuntimeError):
    """Raised when a client did not come from an approved mock or live factory."""


class _MockMessages:
    def __init__(self, owner: "ScriptedMockClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        self._owner.calls.append(copy.deepcopy(kwargs))
        if not self._owner.responses:
            raise AgentLoopError("scripted mock has no response remaining")
        return self._owner.responses.pop(0)


class ScriptedMockClient:
    """Small deterministic Messages API fake; this is the default client."""

    _retention_agent_client_kind = "mock"

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.messages = _MockMessages(self)


_LIVE_GATE_TOKEN = object()


class _GuardedLiveMessages:
    def __init__(self, messages: Any) -> None:
        self._messages = messages
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        if self.call_count >= MAX_LIVE_API_CALLS:
            raise LiveModeError(
                f"live API call cap reached ({MAX_LIVE_API_CALLS} calls per run)"
            )
        max_tokens = kwargs.get("max_tokens")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_TOKENS:
            raise LiveModeError(f"max_tokens must be capped at {MAX_TOKENS}")
        self.call_count += 1
        return self._messages.create(**kwargs)


class GuardedLiveClient:
    """A raw SDK client reachable only through create_live_client."""

    _retention_agent_client_kind = "live_confirmed"

    def __init__(self, client: Any, token: object) -> None:
        if token is not _LIVE_GATE_TOKEN:
            raise LiveModeError("live client must be created by create_live_client")
        self.messages = _GuardedLiveMessages(client.messages)


def create_live_client(*, confirmation: str) -> GuardedLiveClient:
    """Construct the only permitted live client after both independent guards."""

    print(f"Live model: {MODEL_NAME}")
    print(
        "Estimated total for four short recorded runs: "
        f"{ESTIMATED_FOUR_RUN_COST_USD:.3f} USD (under 0.05 USD)."
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LiveModeError("ANTHROPIC_API_KEY is not present in the environment")
    if confirmation != LIVE_CONFIRMATION_PHRASE:
        raise LiveModeError(
            f"runtime confirmation must exactly equal: {LIVE_CONFIRMATION_PHRASE}"
        )

    from anthropic import Anthropic

    return GuardedLiveClient(Anthropic(api_key=api_key), _LIVE_GATE_TOKEN)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _serialise_block(block: Any) -> dict[str, Any]:
    if isinstance(block, Mapping):
        return dict(block)
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json")
    block_type = _get(block, "type")
    if block_type == "text":
        return {"type": "text", "text": _get(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": _get(block, "id"),
            "name": _get(block, "name"),
            "input": _get(block, "input", {}),
        }
    raise AgentLoopError(f"unsupported response content block: {block_type!r}")


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
        return segment_comparison(state.customer_record)

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
    if type(selected) not in {ScriptedMockClient, GuardedLiveClient}:
        raise UnsafeClientError(
            "client must be ScriptedMockClient or created by create_live_client"
        )
    return selected


def run_retention_agent(
    customer: Mapping[str, Any],
    *,
    client: Any | None = None,
    max_turns: int = MAX_LOOP_TURNS,
    clock: Any | None = None,
) -> dict[str, Any]:
    """Run a bounded Messages API loop and return a governed output plus trace."""

    if not isinstance(max_turns, int) or not 1 <= max_turns <= MAX_LOOP_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_LOOP_TURNS}")
    safe_client = _safe_client(client)
    trace = TraceRecorder() if clock is None else TraceRecorder(clock)
    state = _LoopState(customer, customer_policy_context(customer))
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Produce a governed retention recommendation for this synthetic "
                f"customer:\n{json.dumps(customer, sort_keys=True, default=str)}"
            ),
        }
    ]

    turn = 0
    while turn < max_turns:
        turn += 1
        response = safe_client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=list(TOOL_DEFINITIONS),
        )
        content = [_serialise_block(block) for block in _get(response, "content", [])]
        stop_reason = _get(response, "stop_reason")
        messages.append({"role": "assistant", "content": content})

        for block in content:
            if block.get("type") == "text" and block.get("text"):
                trace.log("model_thought", {"text": block["text"]})

        if stop_reason == "tool_use":
            tool_uses = [block for block in content if block.get("type") == "tool_use"]
            if not tool_uses:
                raise AgentLoopError("tool_use stop reason contained no tool_use blocks")
            tool_results: list[dict[str, Any]] = []
            for block in tool_uses:
                tool_id = block.get("id")
                name = block.get("name")
                tool_input = block.get("input", {})
                trace.log(
                    "tool_call",
                    {"tool_use_id": tool_id, "name": name, "input": tool_input},
                )
                try:
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
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result, sort_keys=True, default=str),
                }
                if is_error:
                    result_block["is_error"] = True
                tool_results.append(result_block)
            messages.append({"role": "user", "content": tool_results})
            continue

        if stop_reason == "end_turn":
            final_output = state.final_output or _blocked_refusal(state, trace)
            if final_output is None:
                raise AgentLoopError(
                    "model ended without a formatter result or a checked blocked action"
                )
            return {
                "recommendation": final_output,
                "trace": trace.as_list(),
                "turns": turn,
                "model": MODEL_NAME,
            }

        raise AgentLoopError(f"unsupported stop reason: {stop_reason!r}")

    if state.final_output is not None:
        final_output = state.final_output
    else:
        final_output = _blocked_refusal(state, trace)
    if final_output is not None:
        return {
            "recommendation": final_output,
            "trace": trace.as_list(),
            "turns": max_turns,
            "model": MODEL_NAME,
        }
    raise AgentLoopLimitError(f"agent loop reached the {max_turns}-turn cap")

