"""Offline stand in for a manipulated model.

A script is a list of turns. Each turn has optional ``thought`` text and a list of
``calls``. A call names a tool and gives either ``arguments`` (an object, sent as
JSON) or ``raw_arguments`` (a string sent unchanged, for malformed input). A final
turn with no calls ends the run. The responses feed ``ScriptedMockClient``, so the
real loop, tools, gate and formatter all run and no provider request is made.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agent.loop import ScriptedMockClient


def build_responses(script: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    responses = []
    call_number = 0
    for turn in script:
        tool_calls = []
        for call in turn.get("calls", []):
            call_number += 1
            arguments = call.get("raw_arguments")
            if arguments is None:
                arguments = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            tool_calls.append(
                {
                    "id": f"adv_{call_number:02d}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": arguments},
                }
            )
        responses.append(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                        "message": {"content": turn.get("thought"), "tool_calls": tool_calls},
                    }
                ]
            }
        )
    if not responses or responses[-1]["choices"][0]["finish_reason"] != "stop":
        responses.append(
            {"choices": [{"finish_reason": "stop", "message": {"content": None, "tool_calls": []}}]}
        )
    return responses


def scripted_client(script: Sequence[Mapping[str, Any]]) -> ScriptedMockClient:
    return ScriptedMockClient(build_responses(script))
