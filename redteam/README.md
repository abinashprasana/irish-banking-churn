# Red team evaluation: attack surface

Scope: this suite tests the Atlantic Ledger retention agent in this repository only, using synthetic customers, a synthetic offer catalogue and synthetic governance flags. It sends no request to any third party system other than the Groq or Gemini model endpoints the application already uses, and only in opt in live mode. No real customer data is used.

This file lists every place where untrusted or model influenced content enters or leaves the agent, as found in the code on branch `redteam-eval`. The unsafe outcomes are defined in [harm_spec.md](harm_spec.md).

## How content flows

1. The runtime builds a customer record, recomputes the churn probability with `predict_proba`, and serialises the whole record into the first user message (`agent/loop.py`, `run_retention_agent`).
2. The model calls four tools. Results return to the model as JSON tool messages and are written to the trace.
3. The checker issues a `PolicyDecision` bound to the exact customer and action fingerprints. The formatter accepts output only when it matches the latest decision.
4. The lab shows the relationship manager (RM) the final recommendation, the justification, the model thought text, and a per stage summary of the trace (`app.py`, `decision_gate_ui.py`).

## Surfaces

| # | Surface | Enters or leaves | Free text | Controlled by | Checked by |
|---|---|---|---|---|---|
| S1 | Customer reference (`app.py` text input `case_customer_reference`) | Enters the prompt as `customer_id`, and is used for policy fingerprints and cohort exclusion | Yes, no length limit | Any lab visitor | Stripped only |
| S2 | Top level customer keys other than `profile` (`governance_note`, `churn_drivers`, `counterfactuals`, any extra key) | Enters the prompt; the whole record is serialised | Yes | Scenario files and runtime code (operator) | Not checked |
| S3 | `profile` fields | Enters the prompt and the model | No | Lab visitor | `prepare_feature_vector`: exact keys, numeric, bool or known label |
| S4 | `governance` flags and `held_products` | Enters the prompt and the policy context | No | Operator; case review fixes both flags to false | Type checks in `CustomerPolicyContext` |
| S5 | `product_lookup` result | Tool result read by the model | Yes (names, descriptions, notes) | Repository file `data/retention_products.json` | Trusted file, `synthetic: true` check |
| S6 | `segment_comparison` result | Tool result read by the model and shown to the RM as "Cohort of N" | No, but numbers can describe very few people | Derived from the CSV | None; cohort cells go as small as 1 customer (9 cells hold 5 or fewer) |
| S7 | Tool error strings | Tool result read by the model | Yes, and they repeat model input (for example `unknown retention action: <input>`) | Model, reflected | None |
| S8 | Checker arguments `action_id`, `requires_human_review` | Model to tool | `action_id` is a string | Model | `canonical_action_context`: strip, lower case, exact catalogue match; strict bool |
| S9 | Formatter arguments `action`, `justification`, `confidence`, `regulatory_flags`, `checker_verdict` | Model to tool, then to the RM | `justification` is free text | Model | Pydantic `Recommendation` (strict, extra fields forbidden); decision match; flags and verdict replaced by the runtime; justification not checked |
| S10 | Assistant message text (`model_thought`) | Model to trace, shown to the RM | Yes | Model | Not checked |
| S11 | Final output to the RM | Leaves the system | `justification` and thought text | Model, bounded by S9 | As S9 |

## Where the code differs from the README

1. The request limits live in process memory (`GLOBAL_REQUEST_QUOTA`). Each command line run starts again at zero, so the 950 per day cap is not enforced across separate runs of `eval_agent.py --live` or `record_demo_runs.py`. At the per minute limit the guard raises an error instead of waiting.
2. Tokens per minute are not guarded, and `run_retention_agent` does not keep `response.usage`, so token counts are not recorded anywhere today.
3. The formatter checks against the latest checker decision only. An approved output formatted earlier stays in `final_output` if the model later checks a blocked action and stops.
4. The system prompt tells the model not to claim compliance, guaranteed savings or eligibility. Nothing in code checks the justification or the thought text for those claims.
5. The catalogue has no offer in the `credit` category. ARR-001 can only fire through `mortgage_fixed_rate_review`.
6. The lab labels the product lookup stage "N eligible options returned". The tool does not assess eligibility. This is a UI wording point and is out of scope for this backend work.
