import manifest from "@/data/evidence.generated.json";
import type { Direction, UiScenario } from "@/lib/content-types";

const actionLabels: Record<string, string> = {
  fee_waiver_6m: "Six-month maintenance fee waiver",
  dedicated_service_review: "Dedicated service and switching-friction review",
  no_recommendation: "No recommendation issued",
};

function displayValue(value: unknown): string {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return new Intl.NumberFormat("en-IE").format(value);
  return String(value);
}

function scenarioProfileFacts(scenario: (typeof manifest.scenarios)[number]) {
  const profile = scenario.customer.profile;
  const governance = scenario.customer.governance;
  return [
    { label: "Age", value: String(profile.age) },
    { label: "Account", value: profile.account_type },
    { label: "Products", value: String(profile.num_products) },
    {
      label: "Migration segment",
      value: profile.was_kbc_ulster_customer ? "Former KBC / Ulster" : "Other",
    },
    {
      label: "Switching difficulty",
      value: displayValue(profile.experienced_switching_difficulty),
    },
    { label: "Arrears overlay", value: displayValue(governance.in_arrears) },
    {
      label: "Vulnerability overlay",
      value: displayValue(governance.vulnerable_customer),
    },
    {
      label: "Held products",
      value: scenario.customer.heldProducts
        .map((product) => product.replaceAll("_", " "))
        .join(", "),
    },
  ];
}

export const evidenceManifest = manifest;

export const uiScenarios: UiScenario[] = manifest.scenarios.map((scenario) => ({
  id: scenario.id,
  title: scenario.title,
  customerId: scenario.customer.id,
  probability: scenario.customer.churnProbability,
  profileFacts: scenarioProfileFacts(scenario),
  drivers: scenario.customer.drivers.map((driver) => ({
    feature: driver.feature,
    value: displayValue(driver.value),
    direction: driver.direction as Direction,
    shapValue: driver.shap_value,
  })),
  tools: scenario.toolSteps.map((tool) => ({
    name: tool.name,
    summary: tool.summary,
  })),
  policyRules: scenario.policy.rules.map((rule) => ({
    id: rule.id,
    description: rule.description,
    passed: rule.passed,
    reason: rule.reason,
  })),
  verdict: scenario.policy.verdict as "approved" | "blocked",
  failedRuleIds: scenario.policy.failedRuleIds,
  action: scenario.finalDecision.action,
  actionLabel:
    actionLabels[scenario.finalDecision.action] ??
    scenario.finalDecision.action.replaceAll("_", " "),
  justification: scenario.finalDecision.justification,
  confidence: scenario.finalDecision.confidence,
  flags: scenario.finalDecision.flags,
}));
