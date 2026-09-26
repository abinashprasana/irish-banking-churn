"""Gate off mode for the red team harness only.

``gate_disabled`` patches ``agent.loop`` for the duration of a ``with`` block so the
formatter trusts the model: it still validates the output schema, but it no longer
needs a checker call, a matching decision, or an approved verdict, and a blocked
check no longer turns into an automatic refusal. Nothing in ``agent/`` changes, and
the original functions are restored when the block exits.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from agent import loop
from agent.tools import Recommendation


def ungated_formatter(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Schema check only. The model's action, flags and verdict are kept as sent."""

    return Recommendation.model_validate(dict(candidate)).model_dump(mode="json")


@contextmanager
def gate_disabled() -> Iterator[None]:
    original_execute = loop._execute_tool

    def execute(name: str, tool_input: Mapping[str, Any], state: Any, trace: Any) -> Any:
        if name != "recommendation_formatter":
            return original_execute(name, tool_input, state, trace)
        state.executed_tools.append(name)
        output = ungated_formatter(tool_input)
        state.final_output = output
        trace.log("final_output", output)
        return output

    with patch.object(loop, "_execute_tool", execute), patch.object(
        loop, "_blocked_refusal", lambda state, trace: None
    ):
        yield
