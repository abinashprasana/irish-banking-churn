"""Record schema for redteam/attacks/*.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from redteam.oracle import ALL_HARMS
from redteam.scenarios import scenario_names


ATTACKS_PATH = Path(__file__).resolve().parent / "attacks" / "attacks.jsonl"
BENIGN_PATH = Path(__file__).resolve().parent / "attacks" / "benign.jsonl"

Family = Literal[
    "instruction_injection",
    "persuasion",
    "tool_argument_tampering",
    "relabelling",
    "coverage_gap",
    "reasoning_manipulation",
    "benign",
]
Surface = Literal[
    "customer_reference",
    "scenario_metadata",
    "tool_error_reflection",
    "tool_result",
    "checker_input",
    "formatter_input",
    "policy_decision",
    "model_reasoning",
    "none",
]
Defence = Literal[
    "gate_blocks",
    "formatter_rejects",
    "catalogue_rejects",
    "schema_rejects",
    "decision_mismatch",
    "decision_unforgeable",
    "ordering_check",
    "none",
]
Status = Literal["covered_by_rule", "known_gap", "closed_bypass", "benign"]

# Surface S2 in redteam/README.md: operator controlled, reported in its own table.
OPERATOR_SURFACES = frozenset({"scenario_metadata"})


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[ab]\d{3}$")
    family: Family
    surface: Surface
    description: str = Field(min_length=1)
    scenario: str
    payload: dict[str, Any]
    violates: list[str]
    expected_defence: Defence
    status: Status
    modes: list[Literal["offline", "live"]] = Field(min_length=1)
    reviewed: bool
    observe: list[Literal["system_prompt_leak"]] = []

    @field_validator("scenario")
    @classmethod
    def _known_scenario(cls, value: str) -> str:
        if value not in scenario_names():
            raise ValueError(f"unknown scenario {value!r}")
        return value

    @field_validator("violates")
    @classmethod
    def _known_harms(cls, value: list[str]) -> list[str]:
        unknown = set(value) - ALL_HARMS
        if unknown:
            raise ValueError(f"unknown harm ids {sorted(unknown)}")
        return value

    @property
    def headline(self) -> bool:
        return self.status in {"covered_by_rule", "closed_bypass"} and self.surface not in OPERATOR_SURFACES


def load_records(path: Path) -> list[Record]:
    records = [
        Record.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate record ids in {path.name}")
    return records
