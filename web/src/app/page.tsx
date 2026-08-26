import type { CSSProperties } from "react";

import { BrandLockup } from "@/components/brand-lockup";
import { BrandMark } from "@/components/brand-mark";
import { DecisionJourney } from "@/components/decision-journey";
import { DecisionStack } from "@/components/decision-stack";
import { ScenarioExplorer } from "@/components/scenario-explorer";
import { evidenceManifest, uiScenarios } from "@/lib/evidence";
import { site } from "@/lib/site";

function asPercent(value: number, digits = 1) {
  return new Intl.NumberFormat("en-IE", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function metricValue(
  metrics: Array<{ id: string; value: number }>,
  id: string,
) {
  return metrics.find((metric) => metric.id === id)?.value ?? 0;
}

function titleCase(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function Home() {
  const { evidence } = evidenceManifest;
  const averagePrecision = metricValue(evidence.model.metrics, "averagePrecision");
  const selectedScenario = uiScenarios[0];

  return (
    <>
      <main id="main-content">
        <section className="hero section-shell" id="top" aria-labelledby="hero-heading">
          <div className="hero-copy">
            <p className="eyebrow hero-eyebrow">
              <span>Independent banking AI case study</span>
              <span>Synthetic data</span>
              <span>Ireland</span>
            </p>
            <h1 id="hero-heading" aria-label="Know who may leave — decide with care">
              <span className="hero-heading-line">Know who may leave</span>
              <em className="hero-heading-line">Decide with care</em>
            </h1>
            <p className="hero-deck">
              A synthetic Irish banking study connecting churn prediction, model evidence, counterfactual exploration, and a deterministic retention-policy gate.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href="#system">
                Explore the system
              </a>
              <a
                className="button button-secondary"
                href={site.labUrl}
                target="_blank"
                rel="noreferrer"
              >
                Open interactive lab <span aria-hidden="true">↗</span>
              </a>
            </div>
            <p className="hero-disclosure">
              Research prototype. No real customer data, automated outreach, or compliance claim.
            </p>
          </div>
          <DecisionStack />
        </section>

        <section className="evidence-strip" aria-label="Verified project evidence">
          <dl className="section-shell">
            <div>
              <dt>Customer profiles</dt>
              <dd>{evidence.dataset.recordCount.toLocaleString("en-IE")}</dd>
              <small>Fully synthetic</small>
            </div>
            <div>
              <dt>Model inputs</dt>
              <dd>{evidence.dataset.featureCount}</dd>
              <small>Irish migration context</small>
            </div>
            <div>
              <dt>Average precision</dt>
              <dd>{averagePrecision.toFixed(3)}</dd>
              <small>Original holdout</small>
            </div>
            <div>
              <dt>Deterministic tests</dt>
              <dd>
                {evidence.verification.testsPassed}/{evidence.verification.testsTotal}
              </dd>
              <small>Zero skipped</small>
            </div>
          </dl>
        </section>

        <section className="context section-shell section-space" aria-labelledby="context-heading">
          <div className="section-intro">
            <p className="eyebrow">The context</p>
            <h2 id="context-heading">A banking disruption, treated as context rather than a claim</h2>
            <p>
              KBC Bank Ireland and Ulster Bank announced their intentions to leave the Irish market in 2021. That account migration gives this study a meaningful setting, but it does not prove that the same customers remain at unusual churn risk today.
            </p>
            <p>
              Published evidence informs the historical story and one generation assumption. The customer population, labels, migration share, and behaviour remain constructed.
            </p>
          </div>

          <div className="context-ledger">
            <article>
              <span className="context-figure">1,167,219</span>
              <p>current and deposit accounts closed at the two exiting banks from the start of 2022 to the end of June 2023.</p>
              <a href={evidence.dataset.sources[0].url} target="_blank" rel="noreferrer">
                Central Bank of Ireland source <span aria-hidden="true">↗</span>
              </a>
            </article>
            <article>
              <span className="context-figure">60%</span>
              <p>of respondents in CCPC research reported challenges while switching or closing an affected account.</p>
              <a href={evidence.dataset.sources[1].url} target="_blank" rel="noreferrer">
                CCPC research source <span aria-hidden="true">↗</span>
              </a>
            </article>
            <p className="context-caveat">The 60% figure is used only as a probability for one synthetic subgroup. It is not presented as a subgroup estimate from the CCPC.</p>
          </div>
        </section>

        <section className="system-section section-space" id="system" aria-labelledby="system-heading">
          <div className="section-shell">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">The system</p>
                <h2 id="system-heading">Prediction is the opening signal, not the final decision</h2>
              </div>
              <p>
                The project closes the gap between identifying a high-risk synthetic case and deciding whether any response is suitable enough to reach advisor review.
              </p>
            </div>

            <ol className="system-flow">
              <li>
                <span>01</span>
                <div>
                  <strong>Synthetic profile</strong>
                  <p>19 constructed customer and migration-related inputs.</p>
                </div>
                <small>Local dataset</small>
              </li>
              <li>
                <span>02</span>
                <div>
                  <strong>Churn estimate</strong>
                  <p>XGBoost returns a review signal through <code>predict_proba</code>.</p>
                </div>
                <small>Phase 1</small>
              </li>
              <li>
                <span>03</span>
                <div>
                  <strong>Inspectable evidence</strong>
                  <p>SHAP explains the fitted model; DiCE explores candidate scenarios.</p>
                </div>
                <small>Explanation</small>
              </li>
              <li>
                <span>04</span>
                <div>
                  <strong>Bounded tool path</strong>
                  <p>Four deterministic tools inspect products, cohorts, policy, and output shape.</p>
                </div>
                <small>Phase 2</small>
              </li>
              <li>
                <span>05</span>
                <div>
                  <strong>Governed outcome</strong>
                  <p>The gate returns either a reviewable action or a structured refusal.</p>
                </div>
                <small>Human remains accountable</small>
              </li>
            </ol>
          </div>
        </section>

        <section className="model-section section-shell section-space" id="evidence" aria-labelledby="model-heading">
          <div className="section-heading-row">
            <div>
              <p className="eyebrow">Model evidence</p>
              <h2 id="model-heading">Selected for the imbalanced problem, not the largest accuracy badge</h2>
            </div>
            <p>
              Average precision is the primary comparison because a classifier predicting every customer as retained would still reach 79% accuracy while identifying no churners.
            </p>
          </div>

          <div className="benchmark-layout">
            <div className="benchmark-plot" aria-label="Average precision by model">
              <div className="benchmark-axis">
                <span>0.0</span>
                <span>Average precision</span>
                <span>0.9</span>
              </div>
              {evidence.model.benchmarks.map((benchmark) => {
                const value = metricValue(benchmark.metrics, "averagePrecision");
                const style = {
                  "--benchmark-width": `${(value / 0.9) * 100}%`,
                } as CSSProperties;
                return (
                  <div className="benchmark-row" data-selected={benchmark.selected} key={benchmark.model}>
                    <div>
                      <strong>{benchmark.model}</strong>
                      <span>{value.toFixed(4)}</span>
                    </div>
                    <div className="benchmark-track">
                      <span style={style} />
                    </div>
                  </div>
                );
              })}
              <p>XGBoost is 0.102 average-precision points above the logistic-regression baseline on this holdout.</p>
            </div>

            <div className="metric-focus">
              <span className="mono-label">SELECTED MODEL / HOLDOUT</span>
              <strong>{asPercent(averagePrecision)}</strong>
              <h3>Average precision</h3>
              <p>
                {asPercent(metricValue(evidence.model.metrics, "recall"))} recall and {asPercent(metricValue(evidence.model.metrics, "precision"))} precision at the evaluated threshold.
              </p>
              <dl>
                <div>
                  <dt>ROC-AUC</dt>
                  <dd>{metricValue(evidence.model.metrics, "rocAuc").toFixed(4)}</dd>
                </div>
                <div>
                  <dt>F1 score</dt>
                  <dd>{metricValue(evidence.model.metrics, "f1").toFixed(4)}</dd>
                </div>
              </dl>
            </div>
          </div>

          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <caption>Complete model comparison on the original 2,000-profile test set</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Accuracy</th>
                  <th scope="col">Precision</th>
                  <th scope="col">Recall</th>
                  <th scope="col">F1</th>
                  <th scope="col">ROC-AUC</th>
                  <th scope="col">Avg. precision</th>
                </tr>
              </thead>
              <tbody>
                {evidence.model.benchmarks.map((benchmark) => (
                  <tr key={benchmark.model} data-selected={benchmark.selected}>
                    <th scope="row">{benchmark.model}</th>
                    {["accuracy", "precision", "recall", "f1", "rocAuc", "averagePrecision"].map((id) => (
                      <td key={id}>{metricValue(benchmark.metrics, id).toFixed(4)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="shap-layout">
            <div className="shap-copy">
              <p className="eyebrow">Global explanation</p>
              <h3>What most shaped this fitted model?</h3>
              <p>
                Across the 2,000-profile holdout, product count and months since switching have the largest mean absolute SHAP effects. That pattern comes from the generated data and label rule.
              </p>
              <p className="evidence-caveat">{evidence.model.explanationCaveat}</p>
            </div>
            <ol className="shap-ranking">
              {evidence.model.topFeatures.map((feature) => {
                const style = {
                  "--shap-width": `${(feature.meanAbsoluteShap / evidence.model.topFeatures[0].meanAbsoluteShap) * 100}%`,
                } as CSSProperties;
                return (
                  <li key={feature.name}>
                    <span>0{feature.rank}</span>
                    <div>
                      <strong>{titleCase(feature.name)}</strong>
                      <span className="shap-track">
                        <i style={style} />
                      </span>
                    </div>
                    <em>{feature.meanAbsoluteShap.toFixed(3)}</em>
                  </li>
                );
              })}
            </ol>
          </div>
        </section>

        <section className="journey-section section-space" aria-labelledby="journey-heading">
          <div className="section-shell">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">One customer, five decisions</p>
                <h2 id="journey-heading">Follow the evidence without losing the human boundary</h2>
              </div>
              <p>One verified recorded case shows where modelling ends, where local prototype controls begin, and why a passed gate is still not an instruction to contact a customer.</p>
            </div>
            <DecisionJourney scenario={selectedScenario} />
          </div>
        </section>

        <section className="replay-section section-shell section-space" id="decision" aria-labelledby="replay-heading">
          <div className="section-heading-row">
            <div>
              <p className="eyebrow">Recorded decision replay</p>
              <h2 id="replay-heading">Passing and blocked outcomes deserve equal visibility</h2>
            </div>
            <p>
              All four scenarios use verified Phase 1 probabilities. Two pass the local gate with advisor review; two are blocked and return no recommendation.
            </p>
          </div>
          <ScenarioExplorer scenarios={uiScenarios} />
        </section>

        <section className="governance-section section-space" id="governance" aria-labelledby="governance-heading">
          <div className="section-shell">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">Governance by construction</p>
                <h2 id="governance-heading">The agent can propose — the gate decides what may proceed</h2>
              </div>
              <p>
                Each rule runs in deterministic Python against the exact synthetic customer-action pair. A blocked action cannot be formatted as approved, and all rule results remain visible.
              </p>
            </div>

            <div className="governance-rules">
              {evidence.governance.rules.map((rule, index) => (
                <article key={rule.id}>
                  <span>0{index + 1}</span>
                  <strong>{rule.id}</strong>
                  <p>{rule.description}</p>
                  <small>Deterministic</small>
                </article>
              ))}
            </div>

            <div className="governance-boundary">
              <div>
                <BrandMark className="boundary-mark" tone="duotone" decorative />
                <h3>A safeguard is not a compliance claim</h3>
              </div>
              <p>{evidence.governance.claimScope}</p>
              <dl>
                <div>
                  <dt>Human-review threshold</dt>
                  <dd>{asPercent(evidence.governance.humanReviewThreshold, 0)}</dd>
                </div>
                <div>
                  <dt>Recorded agent requests</dt>
                  <dd>{evidence.verification.apiRequests}</dd>
                </div>
                <div>
                  <dt>Blocked eval outcomes</dt>
                  <dd>{evidence.verification.blockedOutcomes}/{evidence.verification.scenariosTotal}</dd>
                </div>
              </dl>
            </div>
          </div>
        </section>

        <section className="limits section-shell section-space" aria-labelledby="limits-heading">
          <div className="limits-heading">
            <p className="eyebrow">Boundaries</p>
            <h2 id="limits-heading">What the system predicts, and what it deliberately does not decide</h2>
          </div>
          <div className="limits-columns">
            <article>
              <span>Predicts</span>
              <h3>A synthetic churn probability</h3>
              <p>It estimates how the fitted XGBoost model scores a constructed customer profile at one point in time.</p>
            </article>
            <article>
              <span>Explains</span>
              <h3>Model behaviour, not causality</h3>
              <p>SHAP and DiCE make the fitted system inspectable. They do not prove why a real person leaves or prescribe how to retain them.</p>
            </article>
            <article>
              <span>Does not decide</span>
              <h3>Suitability, eligibility, or contact</h3>
              <p>Local rules demonstrate control flow. A trained advisor and production governance process would still own any real decision.</p>
            </article>
          </div>
          <div className="production-needs">
            <h3>Production adoption would additionally require</h3>
            <ul>
              <li>Representative, consented, and quality-controlled bank data</li>
              <li>Legal, compliance, fairness, privacy, and model-risk review</li>
              <li>Authentication, audit storage, monitoring, incident response, and change control</li>
              <li>Outcome measurement and human-factors testing before any customer workflow</li>
            </ul>
          </div>
        </section>

        <section className="artifacts-section section-space" aria-labelledby="artifacts-heading">
          <div className="section-shell artifacts-layout">
            <div>
              <p className="eyebrow">Project record</p>
              <h2 id="artifacts-heading">Built to be inspected, not merely presented</h2>
              <p>
                Designed and developed by {site.author}: synthetic-data generation, model comparison, explanation, counterfactual exploration, a bounded tool loop, deterministic controls, recorded evaluations, and this product case study.
              </p>
              <div className="verification-stamp">
                <span className="mono-label">LAST VERIFIED / {evidenceManifest.generatedAt}</span>
                <strong>{evidence.verification.testsPassed}/{evidence.verification.testsTotal} tests · {evidence.verification.scenariosPassed}/{evidence.verification.scenariosTotal} scenarios</strong>
              </div>
            </div>
            <nav className="artifact-links" aria-label="Project artifacts">
              <a href={site.repositoryUrl} target="_blank" rel="noreferrer">
                <span>Source repository</span>
                <small>Architecture, code, and setup</small>
                <em aria-hidden="true">↗</em>
              </a>
              <a href={`${site.repositoryUrl}/blob/main/model_card.md`} target="_blank" rel="noreferrer">
                <span>Model card</span>
                <small>Intended use, evidence, and limits</small>
                <em aria-hidden="true">↗</em>
              </a>
              <a href={`${site.repositoryUrl}/tree/main/demo_traces`} target="_blank" rel="noreferrer">
                <span>Recorded traces</span>
                <small>Four sanitized decision outcomes</small>
                <em aria-hidden="true">↗</em>
              </a>
              <a href={`${site.repositoryUrl}/blob/main/scripts/eval_agent.py`} target="_blank" rel="noreferrer">
                <span>Evaluation harness</span>
                <small>Zero-request verification path</small>
                <em aria-hidden="true">↗</em>
              </a>
            </nav>
          </div>
        </section>

        <section className="lab-cta section-shell section-space" aria-labelledby="lab-heading">
          <div>
            <p className="eyebrow">Interactive lab</p>
            <h2 id="lab-heading">Inspect the working system</h2>
            <p>Run synthetic profiles through the predictor, inspect local evidence, explore counterfactuals, and review governed recorded outcomes in the operational lab.</p>
          </div>
          <a className="button button-primary" href={site.labUrl} target="_blank" rel="noreferrer">
            Open interactive lab <span aria-hidden="true">↗</span>
          </a>
        </section>
      </main>

      <footer className="site-footer">
        <div className="section-shell">
          <BrandLockup
            className="footer-brand"
            descriptor={site.descriptor}
            tone="ink"
          />
          <p>Synthetic research prototype · No real customer data · Ireland · 2026</p>
          <a href="#top">Back to top ↑</a>
        </div>
      </footer>
    </>
  );
}
