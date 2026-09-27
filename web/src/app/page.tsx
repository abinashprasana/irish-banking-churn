import type { CSSProperties } from "react";

import { BrandLockup } from "@/components/brand-lockup";
import { BrandMark } from "@/components/brand-mark";
import { ConfusionMatrix } from "@/components/confusion-matrix";
import { CountUp } from "@/components/count-up";
import { DecisionJourney } from "@/components/decision-journey";
import { DecisionStack } from "@/components/decision-stack";
import { FlowRule } from "@/components/flow-rule";
import { InView } from "@/components/in-view";
import { RedTeamPanel } from "@/components/red-team-panel";
import { ScenarioExplorer } from "@/components/scenario-explorer";
import { evidenceManifest, modelDisplayName, uiScenarios } from "@/lib/evidence";
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

const featurePhrases: Record<string, string> = {
  num_products: "product count",
  account_type: "account type",
  has_direct_debits: "direct debits",
  months_since_switching: "months since switching",
  tenure_months: "tenure",
};

function featurePhrase(name: string) {
  return featurePhrases[name] ?? name.replaceAll("_", " ");
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
  const selectedBenchmark = evidence.model.benchmarks.find((benchmark) => benchmark.selected);
  const baselineBenchmark = evidence.model.benchmarks.find((benchmark) => benchmark.model.startsWith("Logistic"));
  const precisionGap =
    selectedBenchmark && baselineBenchmark
      ? metricValue(selectedBenchmark.metrics, "averagePrecision") -
        metricValue(baselineBenchmark.metrics, "averagePrecision")
      : 0;
  const [firstFeature, secondFeature] = evidence.model.topFeatures;
  const realData = evidence.realDataBenchmark;

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
            <h1 id="hero-heading" aria-label="Know who may leave, decide with care">
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
          <DecisionStack averagePrecision={averagePrecision} />
        </section>

        <section className="evidence-strip" aria-label="Verified project evidence">
          <dl className="section-shell">
            <div>
              <dt>Customer profiles</dt>
              <dd><CountUp value={evidence.dataset.recordCount} format="integer" /></dd>
              <small>Fully synthetic</small>
            </div>
            <div>
              <dt>Model inputs</dt>
              <dd><CountUp value={evidence.dataset.featureCount} format="integer" /></dd>
              <small>Irish migration context</small>
            </div>
            <div>
              <dt>Average precision</dt>
              <dd><CountUp value={averagePrecision} format="fixed3" /></dd>
              <small>Original holdout</small>
            </div>
            <div>
              <dt>Deterministic tests</dt>
              <dd>
                <CountUp value={evidence.verification.testsPassed} format="integer" />/{evidence.verification.testsTotal}
              </dd>
              <small>Zero skipped</small>
            </div>
          </dl>
        </section>

        <section className="context section-shell section-space" aria-labelledby="context-heading">
          <div className="section-intro">
            <p className="eyebrow">The context</p>
            <h2 id="context-heading">Two bank exits set the scene</h2>
            <p>
              KBC Bank Ireland and Ulster Bank announced their intentions to leave the Irish market in 2021. That account migration gives this study a meaningful setting, but it does not prove that the same customers remain at unusual churn risk today.
            </p>
            <p>
              Published evidence informs the historical story and one generation assumption. The customer population, labels, migration share, and behaviour remain constructed.
            </p>
            <p>
              The practical problem starts after the score. A retention team can see that a customer looks likely to leave, but the number says nothing about which offer is suitable, or whether one should be made at all. This study is written for the people in that gap: the analyst acting on the score and the reviewer who later has to explain the call.
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
                <h2 id="system-heading">The score opens a case, the rules decide it</h2>
              </div>
              <p>
                The model&apos;s job ends at a probability. Anything proposed after that has to clear four deterministic rules, and the language model can&apos;t overrule them. That split is the core idea of the project.
              </p>
            </div>

            <FlowRule>
            <InView as="ol" className="system-flow reveal-group">
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
                  <p>
                    Four deterministic tools inspect products, cohorts, policy, and output shape.
                    <span className="system-flow__providers">
                      {`The live model is ${evidence.agent.providers
                        .map((provider) => {
                          const name = modelDisplayName(provider.modelId);
                          return name.startsWith(provider.name) ? name : `${name} on ${provider.name}`;
                        })
                        .join(" or ")}.`}
                    </span>
                  </p>
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
            </InView>
            </FlowRule>
          </div>
        </section>

        <section className="model-section section-shell section-space" id="evidence" aria-labelledby="model-heading">
          <div className="section-heading-row">
            <div>
              <p className="eyebrow">Model evidence</p>
              <h2 id="model-heading">Judged on average precision, since churners are rare</h2>
            </div>
            <p>
              Average precision is the primary comparison because a classifier predicting every customer as retained would still reach 79% accuracy while identifying no churners.
            </p>
          </div>

          <div className="benchmark-layout">
            <InView className="benchmark-plot grow-bars" aria-label="Average precision by model">
              <div className="benchmark-axis">
                <span>0.0</span>
                <span>Average precision</span>
                <span>0.9</span>
              </div>
              {evidence.model.benchmarks.map((benchmark, index) => {
                const value = metricValue(benchmark.metrics, "averagePrecision");
                const style = {
                  "--benchmark-width": `${(value / 0.9) * 100}%`,
                  "--i": index,
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
              <p>XGBoost is {precisionGap.toFixed(3)} average-precision points above the logistic-regression baseline on this holdout.</p>
            </InView>

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

          <ConfusionMatrix confusion={evidence.model.confusion} />

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
                Across the 2,000-profile holdout, {featurePhrase(firstFeature.name)} and {featurePhrase(secondFeature.name)} have the largest mean absolute SHAP effects. That pattern comes from the generated data and label rule.
              </p>
              <p className="evidence-caveat">{evidence.model.explanationCaveat}</p>
            </div>
            <InView as="ol" className="shap-ranking grow-bars">
              {evidence.model.topFeatures.map((feature, index) => {
                const style = {
                  "--shap-width": `${(feature.meanAbsoluteShap / evidence.model.topFeatures[0].meanAbsoluteShap) * 100}%`,
                  "--i": index,
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
            </InView>
          </div>

          <div className="real-data">
            <div className="real-data__copy">
              <p className="eyebrow">Real data check</p>
              <h3>The same recipe on real bank customers</h3>
              <p>
                UCI Bank Marketing holds {realData.records.toLocaleString("en-IE")} customers of a Portuguese bank (Moro, Cortez and Rita, 2014). Its target is whether a customer took a term deposit, so this checks the training method on real, imbalanced data. It says nothing about churn.
              </p>
              <p className="evidence-caveat">
                {asPercent(realData.positiveRate)} positive rate. The <code>{realData.droppedColumns.join(", ")}</code> column was dropped because it is only known after the outcome. <a href={realData.sourceUrl} target="_blank" rel="noreferrer">UCI dataset page <span aria-hidden="true">↗</span></a>
              </p>
            </div>
            <InView className="real-data__plot grow-bars" aria-label="ROC-AUC on UCI Bank Marketing by model">
              <div className="benchmark-axis">
                <span>0.0</span>
                <span>ROC-AUC</span>
                <span>1.0</span>
              </div>
              {[...realData.models].reverse().map((row, index) => {
                const style = {
                  "--benchmark-width": `${row.rocAuc * 100}%`,
                  "--i": index,
                } as CSSProperties;
                return (
                  <div className="benchmark-row" data-selected={row.model === "XGBoost"} key={row.model}>
                    <div>
                      <strong>{row.model}</strong>
                      <span>{row.rocAuc.toFixed(4)} · AP {row.averagePrecision.toFixed(4)}</span>
                    </div>
                    <div className="benchmark-track">
                      <span style={style} />
                    </div>
                  </div>
                );
              })}
            </InView>
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
                <h2 id="governance-heading">The agent can propose, the gate decides what may proceed</h2>
              </div>
              <p>
                Each rule runs in deterministic Python against the exact synthetic customer-action pair. A blocked action cannot be formatted as approved, and all rule results remain visible.
              </p>
            </div>

            <InView className="governance-rules reveal-group">
              {evidence.governance.rules.map((rule, index) => (
                <article key={rule.id} style={{ "--i": index } as CSSProperties}>
                  <span>0{index + 1}</span>
                  <strong>{rule.id}</strong>
                  <p>{rule.description}</p>
                  <small>
                    <svg className="rule-check" viewBox="0 0 16 16" aria-hidden="true">
                      <path d="M3 8.5L6.5 12L13 4.5" pathLength={1} />
                    </svg>
                    Deterministic
                  </small>
                </article>
              ))}
            </InView>

            <div className="governance-boundary">
              <div>
                <BrandMark className="boundary-mark" tone="duotone" decorative />
                <h3>Safeguards make no compliance claim</h3>
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

        <section className="red-team-section section-space" id="red-team" aria-labelledby="red-team-heading">
          <div className="section-shell">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">Red team</p>
                <h2 id="red-team-heading">Attacking the gate on purpose</h2>
              </div>
              <p>
                {`${evidence.redTeam.attackCount} attacks try to push a blocked offer past the gate: injected instructions, persuasion, tampered tool calls, relabelled offers and outright gaps. Each outcome is judged by an oracle written from a separate harm specification, which never imports the gate's code.`}
              </p>
            </div>
            <RedTeamPanel redTeam={evidence.redTeam} />
          </div>
        </section>

        <section className="limits section-shell section-space" aria-labelledby="limits-heading">
          <div className="limits-heading">
            <p className="eyebrow">Boundaries</p>
            <h2 id="limits-heading">What the system predicts, and what it leaves to people</h2>
          </div>
          <InView className="limits-columns reveal-group">
            <article>
              <span>Predicts</span>
              <h3>A synthetic churn probability</h3>
              <p>It estimates how the fitted XGBoost model scores a constructed customer profile at one point in time.</p>
            </article>
            <article>
              <span>Explains</span>
              <h3>How the model behaves</h3>
              <p>SHAP and DiCE make the fitted system inspectable. They do not prove why a real person leaves or prescribe how to retain them.</p>
            </article>
            <article>
              <span>Does not decide</span>
              <h3>Suitability, eligibility, or contact</h3>
              <p>Local rules demonstrate control flow. A trained advisor and production governance process would still own any real decision.</p>
            </article>
          </InView>
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
              <h2 id="artifacts-heading">Every decision here can be checked</h2>
              <p>
                Designed and built by {site.author}. The reasoning behind it is written down: average precision over accuracy for an imbalanced problem, policy rules kept outside the language model, and blocked outcomes published next to passing ones.
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
              <a href={`${site.repositoryUrl}/tree/main/redteam`} target="_blank" rel="noreferrer">
                <span>Red team suite</span>
                <small>Harm spec, attacks, oracle and results</small>
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
