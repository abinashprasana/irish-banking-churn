"""The four plain functions exposed to the governed retention agent as tools."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr

from agent.policy_rules import (
    ActionPolicyContext,
    CustomerPolicyContext,
    HIGH_RISK_HUMAN_REVIEW_THRESHOLD,
    PolicyDecision,
    issue_policy_decision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = PROJECT_ROOT / "data" / "retention_products.json"
DATA_PATH = PROJECT_ROOT / "data" / "irish_banking_churn.csv"


class PolicyGateError(RuntimeError):
    """Raised when output formatting lacks exact policy authority."""


class Recommendation(BaseModel):
    """Strict fixed schema for every recommendation and refusal."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: StrictStr = Field(min_length=1)
    justification: StrictStr = Field(min_length=1)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    regulatory_flags: list[StrictStr]
    checker_verdict: Literal["approved", "blocked"]


PRODUCT_LOOKUP_SCHEMA = {
    "name": "product_lookup",
    "description": (
        "Read the local synthetic retention-offer catalogue. Product policy "
        "metadata is authoritative and cannot be overridden by model input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "credit",
                    "current_account",
                    "fee_relief",
                    "mortgage",
                    "savings",
                    "service",
                ],
                "description": "Optional catalogue category filter.",
            }
        },
        "additionalProperties": False,
    },
}

SEGMENT_COMPARISON_SCHEMA = {
    "name": "segment_comparison",
    "description": (
        "Compare the target with synthetic Phase 1 customers in the same "
        "migration segment, age band, account type, and product-count band."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

REGULATORY_CONSTRAINT_CHECKER_SCHEMA = {
    "name": "regulatory_constraint_checker",
    "description": (
        "Run every deterministic retention policy rule for a proposed catalogue "
        "action. Customer facts and canonical product metadata are injected by "
        "the runtime and cannot be supplied or overridden by the model."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action_id": {
                "type": "string",
                "description": "The exact identifier returned by product_lookup.",
            },
            "requires_human_review": {
                "type": "boolean",
                "description": "Whether the recommendation includes human review.",
            },
        },
        "required": ["action_id", "requires_human_review"],
        "additionalProperties": False,
    },
}

RECOMMENDATION_FORMATTER_SCHEMA = {
    "name": "recommendation_formatter",
    "description": (
        "Validate the final structured recommendation or refusal. The runtime "
        "injects the matching policy decision; checker_verdict text is not authority."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "minLength": 1},
            "justification": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "regulatory_flags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "checker_verdict": {
                "type": "string",
                "enum": ["approved", "blocked"],
            },
        },
        "required": [
            "action",
            "justification",
            "confidence",
            "regulatory_flags",
            "checker_verdict",
        ],
        "additionalProperties": False,
    },
}

TOOL_DEFINITIONS = (
    PRODUCT_LOOKUP_SCHEMA,
    SEGMENT_COMPARISON_SCHEMA,
    REGULATORY_CONSTRAINT_CHECKER_SCHEMA,
    RECOMMENDATION_FORMATTER_SCHEMA,
)


def _load_catalogue(path: Path = CATALOGUE_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("synthetic") is not True or not isinstance(payload.get("offers"), list):
        raise ValueError("retention catalogue must be explicitly synthetic and contain offers")
    return payload


def _offer_by_id(action_id: str, path: Path = CATALOGUE_PATH) -> dict[str, Any]:
    if not isinstance(action_id, str) or not action_id.strip():
        raise ValueError("action_id must be a non-empty string")
    wanted = action_id.strip().lower()
    for offer in _load_catalogue(path)["offers"]:
        if offer.get("action_id") == wanted:
            return offer
    raise KeyError(f"unknown retention action: {wanted}")


def canonical_action_context(
    action_id: str,
    requires_human_review: bool,
    *,
    catalogue_path: Path = CATALOGUE_PATH,
) -> ActionPolicyContext:
    """Resolve model-selected ID against trusted metadata and bind review intent."""

    if type(requires_human_review) is not bool:
        raise TypeError("requires_human_review must be a bool")
    offer = _offer_by_id(action_id, catalogue_path)
    return ActionPolicyContext(
        action_id=offer["action_id"],
        category=offer["category"],
        is_credit=offer["is_credit"],
        is_upsell=offer["is_upsell"],
        target_product=offer["target_product"],
        requires_human_review=requires_human_review,
    )


def product_lookup(
    category: str | None = None, *, catalogue_path: Path = CATALOGUE_PATH
) -> dict[str, Any]:
    """Return all or one category of clearly synthetic local retention offers."""

    catalogue = _load_catalogue(catalogue_path)
    if category is not None:
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a non-empty string when supplied")
        normalised = category.strip().lower()
        allowed = set(PRODUCT_LOOKUP_SCHEMA["input_schema"]["properties"]["category"]["enum"])
        if normalised not in allowed:
            raise ValueError(f"unsupported product category: {normalised}")
        offers = [offer for offer in catalogue["offers"] if offer["category"] == normalised]
    else:
        offers = list(catalogue["offers"])
    return {
        "catalogue_name": catalogue["catalogue_name"],
        "synthetic": True,
        "currency": catalogue["currency"],
        "offers": offers,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"expected a Boolean value, got {value!r}")


def _age_band(age: int) -> str:
    if age < 18:
        raise ValueError("age must be at least 18")
    if age <= 29:
        return "18-29"
    if age <= 44:
        return "30-44"
    if age <= 59:
        return "45-59"
    return "60+"


def _customer_profile(customer: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    profile = customer.get("profile", customer)
    if not isinstance(profile, Mapping):
        raise TypeError("customer profile must be a mapping")
    customer_id = customer.get("customer_id", profile.get("customer_id", ""))
    if not isinstance(customer_id, str):
        raise TypeError("customer_id must be a string")
    return customer_id, profile


def _row_product_signals(row: Mapping[str, object]) -> tuple[str, ...]:
    account_type = str(row["account_type"])
    products: set[str] = set()
    if "Current" in account_type:
        products.add("current_account")
    if "Savings" in account_type:
        products.add("savings_account")
    if _as_bool(row["has_mortgage"]):
        products.add("mortgage")
    return tuple(sorted(products))


def segment_comparison(
    customer: Mapping[str, Any], *, data_path: Path = DATA_PATH
) -> dict[str, Any]:
    """Return deterministic retention aggregates for a strict similar cohort."""

    customer_id, profile = _customer_profile(customer)
    try:
        target_age_band = _age_band(int(profile["age"]))
        target_account_type = str(profile["account_type"])
        target_num_products = int(profile["num_products"])
        migrated = _as_bool(profile["was_kbc_ulster_customer"])
    except KeyError as exc:
        raise ValueError(f"missing segment field: {exc.args[0]}") from exc

    retained_products: Counter[str] = Counter()
    churned_count = 0
    retained_count = 0
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["customer_id"] == customer_id:
                continue
            if _age_band(int(row["age"])) != target_age_band:
                continue
            if row["account_type"] != target_account_type:
                continue
            if int(row["num_products"]) != target_num_products:
                continue
            if _as_bool(row["was_kbc_ulster_customer"]) != migrated:
                continue
            if int(row["churn"]) == 1:
                churned_count += 1
            else:
                retained_count += 1
                retained_products.update(_row_product_signals(row))

    cohort_size = retained_count + churned_count
    churn_rate = round(churned_count / cohort_size, 4) if cohort_size else None
    common_products = [
        {"product": name, "retained_customers": count}
        for name, count in sorted(
            retained_products.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "cohort_definition": {
            "migration_segment": "former_kbc_ulster" if migrated else "other",
            "age_band": target_age_band,
            "account_type": target_account_type,
            "num_products": target_num_products,
        },
        "cohort_size": cohort_size,
        "retained_count": retained_count,
        "churned_count": churned_count,
        "churn_rate": churn_rate,
        "most_common_products_among_retained": common_products,
        "data_note": "Aggregates use the local synthetic Phase 1 dataset only.",
    }


def regulatory_constraint_checker(
    customer: CustomerPolicyContext,
    proposed_action: ActionPolicyContext,
) -> PolicyDecision:
    """Return all deterministic rule results and exact-action approval evidence."""

    return issue_policy_decision(customer, proposed_action)


def _derived_regulatory_flags(
    customer: CustomerPolicyContext,
    action: ActionPolicyContext,
    decision: PolicyDecision,
) -> list[str]:
    if not decision.passed:
        return list(decision.failed_rule_ids)
    if (
        customer.churn_probability > HIGH_RISK_HUMAN_REVIEW_THRESHOLD
        and action.requires_human_review
    ):
        return ["HUM-003:human_review_required"]
    return []


def recommendation_formatter(
    candidate: Mapping[str, Any],
    *,
    customer: CustomerPolicyContext,
    proposed_action: ActionPolicyContext,
    policy_decision: PolicyDecision | None,
) -> dict[str, Any]:
    """Validate and emit output only with matching runtime-issued policy evidence."""

    recommendation = Recommendation.model_validate(dict(candidate))
    if policy_decision is None:
        raise PolicyGateError("recommendation_formatter requires a policy decision")
    if not policy_decision.matches(customer, proposed_action):
        raise PolicyGateError("policy decision does not match this customer and action")

    if policy_decision.passed:
        if recommendation.action != proposed_action.action_id:
            raise PolicyGateError("approved policy decision is for a different action")
        if recommendation.checker_verdict != "approved":
            raise PolicyGateError("approved action must carry the approved verdict")
    else:
        if recommendation.action != "no_recommendation":
            raise PolicyGateError(
                "a blocked action may only produce a structured no_recommendation refusal"
            )
        if recommendation.checker_verdict != "blocked":
            raise PolicyGateError("structured refusal must carry the blocked verdict")

    output = recommendation.model_dump(mode="json")
    output["regulatory_flags"] = _derived_regulatory_flags(
        customer, proposed_action, policy_decision
    )
    output["checker_verdict"] = policy_decision.checker_verdict
    return output

