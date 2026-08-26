export type Direction = "increases_churn" | "decreases_churn";

export type UiDriver = {
  feature: string;
  value: string;
  direction: Direction;
  shapValue?: number;
};

export type UiPolicyRule = {
  id: string;
  description: string;
  passed: boolean;
  reason: string;
};

export type UiToolStep = {
  name: string;
  summary: string;
};

export type UiScenario = {
  id: string;
  title: string;
  customerId: string;
  probability: number;
  profileFacts: Array<{ label: string; value: string }>;
  drivers: UiDriver[];
  tools: UiToolStep[];
  policyRules: UiPolicyRule[];
  verdict: "approved" | "blocked";
  failedRuleIds: string[];
  action: string;
  actionLabel: string;
  justification: string;
  confidence: number;
  flags: string[];
};
