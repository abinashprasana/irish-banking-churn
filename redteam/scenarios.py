"""Synthetic base customers that attack and benign records build on.

The four public demo customers come from demo_traces/. The others are rows of the
synthetic dataset chosen for their recomputed churn risk or cohort size. Governance
flags are always a synthetic overlay set by the record, never read from the data.
"""

from __future__ import annotations

import copy
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo_traces"
DATA_PATH = PROJECT_ROOT / "data" / "irish_banking_churn.csv"

# Dataset rows picked on the retrained model (commit c4f3cff). Risk in brackets.
DATASET_BASES = {
    "near_threshold_above": "IRLBANK_00064",  # 0.766, just above the 0.75 review line
    "near_threshold_below": "IRLBANK_00078",  # 0.729, just below it
    "low_risk": "IRLBANK_00004",  # 0.026
    "small_cohort": "IRLBANK_02690",  # 0.896, only one other customer in its cohort
    "savings_only": "IRLBANK_00013",  # 0.998, savings account, no direct debits, not migrated
    "mortgage_holder": "IRLBANK_01219",  # 0.822, holds a mortgage
}

_INTEGER_FIELDS = {
    "age",
    "tenure_months",
    "num_products",
    "monthly_transaction_count",
    "direct_debit_count",
    "months_since_switching",
    "branch_visits_monthly",
    "customer_service_calls_6months",
}
_FLOAT_FIELDS = {"monthly_balance_eur", "monthly_transaction_amount_eur"}
_DROP_FROM_DEMO = {"churn_probability", "churn_drivers", "phase1_prediction"}


def _typed(field: str, value: str) -> Any:
    if field in _INTEGER_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    if value in {"True", "False"}:
        return value == "True"
    return value


@lru_cache(maxsize=1)
def _dataset_rows() -> dict[str, dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {row["customer_id"]: row for row in csv.DictReader(handle)}


@lru_cache(maxsize=1)
def _base_customers() -> dict[str, dict[str, Any]]:
    bases: dict[str, dict[str, Any]] = {}
    for path in sorted(DEMO_DIR.glob("*.json")):
        demo = json.loads(path.read_text(encoding="utf-8"))
        bases[demo["demo_id"]] = {
            key: value
            for key, value in demo["customer"].items()
            if key not in _DROP_FROM_DEMO
        }
    rows = _dataset_rows()
    for name, customer_id in DATASET_BASES.items():
        row = rows[customer_id]
        profile = {
            field: _typed(field, value)
            for field, value in row.items()
            if field not in {"customer_id", "churn"}
        }
        bases[name] = {
            "customer_id": customer_id,
            "profile": profile,
            "governance": {"in_arrears": False, "vulnerable_customer": False},
        }
    return bases


def scenario_names() -> list[str]:
    return sorted(_base_customers())


def build_customer(scenario: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a fresh customer record with the record's overrides applied."""

    bases = _base_customers()
    if scenario not in bases:
        raise KeyError(f"unknown base scenario {scenario!r}; expected one of {sorted(bases)}")
    customer = copy.deepcopy(bases[scenario])
    overrides = overrides or {}
    if "customer_id" in overrides:
        customer["customer_id"] = overrides["customer_id"]
    if "governance" in overrides:
        customer.setdefault("governance", {}).update(overrides["governance"])
    if "held_products" in overrides:
        customer["held_products"] = list(overrides["held_products"])
    if "profile" in overrides:
        customer["profile"].update(overrides["profile"])
    for key, value in overrides.get("extra", {}).items():
        customer[key] = copy.deepcopy(value)
    return customer
