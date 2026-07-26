"""Deterministic policy contexts, rules, and immutable checker decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Real
from typing import Callable


HIGH_RISK_HUMAN_REVIEW_THRESHOLD = 0.75
ALLOWED_ACTION_CATEGORIES = frozenset(
    {"credit", "fee_relief", "mortgage", "savings", "service", "current_account"}
)
VULNERABLE_ALLOWED_CATEGORIES = frozenset({"fee_relief", "service"})


def _normalise_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip().lower()


@dataclass(frozen=True, slots=True)
class CustomerPolicyContext:
    """Trusted customer facts injected by the runtime, never by the model."""

    customer_id: str
    churn_probability: float
    held_products: tuple[str, ...] = ()
    in_arrears: bool = False
    vulnerable_customer: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "customer_id",
            _normalise_identifier(self.customer_id, "customer_id"),
        )
        if isinstance(self.churn_probability, bool) or not isinstance(
            self.churn_probability, Real
        ):
            raise TypeError("churn_probability must be a number")
        probability = float(self.churn_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("churn_probability must be between 0 and 1")
        object.__setattr__(self, "churn_probability", probability)
        if type(self.in_arrears) is not bool:
            raise TypeError("in_arrears must be a bool")
        if type(self.vulnerable_customer) is not bool:
            raise TypeError("vulnerable_customer must be a bool")
        products = tuple(
            sorted(
                {
                    _normalise_identifier(product, "held_products item")
                    for product in self.held_products
                }
            )
        )
        object.__setattr__(self, "held_products", products)


@dataclass(frozen=True, slots=True)
class ActionPolicyContext:
    """Canonical action metadata resolved from the trusted product catalogue."""

    action_id: str
    category: str
    is_credit: bool = False
    is_upsell: bool = False
    target_product: str | None = None
    requires_human_review: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action_id", _normalise_identifier(self.action_id, "action_id")
        )
        category = _normalise_identifier(self.category, "category")
        if category not in ALLOWED_ACTION_CATEGORIES:
            raise ValueError(f"unsupported action category: {category}")
        object.__setattr__(self, "category", category)
        for field_name in ("is_credit", "is_upsell", "requires_human_review"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if self.target_product is not None:
            object.__setattr__(
                self,
                "target_product",
                _normalise_identifier(self.target_product, "target_product"),
            )


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule_id: str
    description: str
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "passed": self.passed,
            "reason": self.reason,
        }


RuleCheck = Callable[[CustomerPolicyContext, ActionPolicyContext], RuleResult]


@dataclass(frozen=True, slots=True)
class PolicyRule:
    rule_id: str
    description: str
    check: RuleCheck = field(repr=False, compare=False)


def _result(rule_id: str, description: str, passed: bool, reason: str) -> RuleResult:
    return RuleResult(rule_id, description, passed, reason)


def check_arrears_credit(
    customer: CustomerPolicyContext, action: ActionPolicyContext
) -> RuleResult:
    rule_id = "ARR-001"
    description = "Customers in arrears cannot receive credit or credit-limit offers."
    blocked = customer.in_arrears and (
        action.is_credit or action.category in {"credit", "mortgage"}
    )
    reason = (
        "Blocked: the customer is in arrears and the proposed action is credit-related."
        if blocked
        else "Passed: no prohibited arrears-related credit offer is proposed."
    )
    return _result(rule_id, description, not blocked, reason)


def check_existing_product(
    customer: CustomerPolicyContext, action: ActionPolicyContext
) -> RuleResult:
    rule_id = "HOLD-002"
    description = "Do not offer a product the customer already holds."
    blocked = (
        action.target_product is not None
        and action.target_product in customer.held_products
    )
    reason = (
        f"Blocked: the customer already holds {action.target_product}."
        if blocked
        else "Passed: the action does not duplicate an existing product holding."
    )
    return _result(rule_id, description, not blocked, reason)


def check_high_risk_human_review(
    customer: CustomerPolicyContext, action: ActionPolicyContext
) -> RuleResult:
    rule_id = "HUM-003"
    description = (
        "Recommendations above the churn-risk threshold require human review."
    )
    blocked = (
        customer.churn_probability > HIGH_RISK_HUMAN_REVIEW_THRESHOLD
        and not action.requires_human_review
    )
    reason = (
        "Blocked: churn probability exceeds "
        f"{HIGH_RISK_HUMAN_REVIEW_THRESHOLD:.0%} but human review was not included."
        if blocked
        else "Passed: the human-review requirement is satisfied or not triggered."
    )
    return _result(rule_id, description, not blocked, reason)


def check_vulnerable_customer(
    customer: CustomerPolicyContext, action: ActionPolicyContext
) -> RuleResult:
    rule_id = "VUL-004"
    description = (
        "Vulnerable customers may receive only non-upsell service or fee-relief actions."
    )
    blocked = customer.vulnerable_customer and (
        action.is_upsell or action.category not in VULNERABLE_ALLOWED_CATEGORIES
    )
    reason = (
        "Blocked: vulnerable-customer safeguards permit only non-upsell service "
        "or fee-relief actions."
        if blocked
        else "Passed: vulnerable-customer restrictions are satisfied or not triggered."
    )
    return _result(rule_id, description, not blocked, reason)


RULES: tuple[PolicyRule, ...] = (
    PolicyRule(
        "ARR-001",
        "Customers in arrears cannot receive credit or credit-limit offers.",
        check_arrears_credit,
    ),
    PolicyRule(
        "HOLD-002",
        "Do not offer a product the customer already holds.",
        check_existing_product,
    ),
    PolicyRule(
        "HUM-003",
        "Recommendations above the churn-risk threshold require human review.",
        check_high_risk_human_review,
    ),
    PolicyRule(
        "VUL-004",
        "Vulnerable customers may receive only non-upsell service or fee-relief actions.",
        check_vulnerable_customer,
    ),
)


def _fingerprint(payload: dict[str, object]) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def customer_fingerprint(customer: CustomerPolicyContext) -> str:
    return _fingerprint(
        {
            "customer_id": customer.customer_id,
            "churn_probability": customer.churn_probability,
            "held_products": list(customer.held_products),
            "in_arrears": customer.in_arrears,
            "vulnerable_customer": customer.vulnerable_customer,
        }
    )


def action_fingerprint(action: ActionPolicyContext) -> str:
    return _fingerprint(
        {
            "action_id": action.action_id,
            "category": action.category,
            "is_credit": action.is_credit,
            "is_upsell": action.is_upsell,
            "target_product": action.target_product,
            "requires_human_review": action.requires_human_review,
        }
    )


_DECISION_ISSUER = object()


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Runtime-issued evidence bound to one exact customer/action pair."""

    _issuer: object = field(repr=False, compare=False)
    customer_hash: str
    action_hash: str
    action_id: str
    passed: bool
    rule_results: tuple[RuleResult, ...]

    def __post_init__(self) -> None:
        if self._issuer is not _DECISION_ISSUER:
            raise TypeError("PolicyDecision objects can only be issued by the checker")

    @property
    def checker_verdict(self) -> str:
        return "approved" if self.passed else "blocked"

    @property
    def failed_rule_ids(self) -> tuple[str, ...]:
        return tuple(result.rule_id for result in self.rule_results if not result.passed)

    def authorizes(
        self, customer: CustomerPolicyContext, action: ActionPolicyContext
    ) -> bool:
        return self.passed and self.matches(customer, action)

    def matches(
        self, customer: CustomerPolicyContext, action: ActionPolicyContext
    ) -> bool:
        """Return whether this decision belongs to the exact checked context."""

        return (
            self.customer_hash == customer_fingerprint(customer)
            and self.action_hash == action_fingerprint(action)
            and self.action_id == action.action_id
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "checker_verdict": self.checker_verdict,
            "passed": self.passed,
            "failed_rule_ids": list(self.failed_rule_ids),
            "customer_fingerprint": self.customer_hash,
            "action_fingerprint": self.action_hash,
            "rule_results": [result.as_dict() for result in self.rule_results],
        }


def issue_policy_decision(
    customer: CustomerPolicyContext,
    action: ActionPolicyContext,
) -> PolicyDecision:
    """Evaluate every rule without short-circuiting and issue immutable evidence."""

    results = tuple(rule.check(customer, action) for rule in RULES)
    return PolicyDecision(
        _DECISION_ISSUER,
        customer_fingerprint(customer),
        action_fingerprint(action),
        action.action_id,
        all(result.passed for result in results),
        results,
    )
