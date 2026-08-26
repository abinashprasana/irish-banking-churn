"use client";

import { type KeyboardEvent, useId, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";

import type { UiScenario } from "@/lib/content-types";

function formatProbability(value: number) {
  return new Intl.NumberFormat("en-IE", {
    style: "percent",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function verdictLabel(verdict: UiScenario["verdict"]) {
  return verdict === "approved" ? "Passed local gate" : "Blocked by local gate";
}

export function ScenarioExplorer({ scenarios }: { scenarios: UiScenario[] }) {
  const [selectedId, setSelectedId] = useState(scenarios[0]?.id ?? "");
  const baseId = useId();
  const reduceMotion = useReducedMotion();
  const selected = scenarios.find((scenario) => scenario.id === selectedId) ?? scenarios[0];

  if (!selected) return null;

  function handleTabKeys(event: KeyboardEvent<HTMLDivElement>) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const current = scenarios.findIndex((scenario) => scenario.id === selected.id);
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? scenarios.length - 1
          : event.key === "ArrowRight"
            ? (current + 1) % scenarios.length
            : (current - 1 + scenarios.length) % scenarios.length;
    setSelectedId(scenarios[next].id);
    document.getElementById(`${baseId}-tab-${scenarios[next].id}`)?.focus();
  }

  return (
    <div className="scenario-explorer">
      <div className="scenario-tabs" role="tablist" aria-label="Recorded decision scenarios" onKeyDown={handleTabKeys}>
        {scenarios.map((scenario, index) => (
          <button
            id={`${baseId}-tab-${scenario.id}`}
            key={scenario.id}
            type="button"
            role="tab"
            tabIndex={scenario.id === selected.id ? 0 : -1}
            aria-selected={scenario.id === selected.id}
            aria-controls={`${baseId}-panel-${scenario.id}`}
            onClick={() => setSelectedId(scenario.id)}
          >
            <span className="scenario-number">0{index + 1}</span>
            <span>
              <strong>{scenario.customerId}</strong>
              <small data-verdict={scenario.verdict}>{verdictLabel(scenario.verdict)}</small>
            </span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          id={`${baseId}-panel-${selected.id}`}
          key={selected.id}
          role="tabpanel"
          aria-labelledby={`${baseId}-tab-${selected.id}`}
          className="scenario-panel"
          initial={reduceMotion ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduceMotion ? { opacity: 1 } : { opacity: 0, y: -4 }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="scenario-panel-header">
            <div>
              <span className="mono-label">RECORDED / ZERO API REQUESTS</span>
              <h3>{selected.title}</h3>
            </div>
            <div className="scenario-score">
              <strong>{formatProbability(selected.probability)}</strong>
              <span>churn probability</span>
            </div>
          </div>

          <div className="scenario-grid">
            <section aria-labelledby={`${baseId}-profile-heading`}>
              <div className="scenario-section-heading">
                <span>01</span>
                <h4 id={`${baseId}-profile-heading`}>Profile and evidence</h4>
              </div>
              <dl className="scenario-facts">
                {selected.profileFacts.map((fact) => (
                  <div key={fact.label}>
                    <dt>{fact.label}</dt>
                    <dd>{fact.value}</dd>
                  </div>
                ))}
              </dl>
              <ol className="scenario-drivers">
                {selected.drivers.slice(0, 5).map((driver) => (
                  <li key={driver.feature}>
                    <span className="driver-name">{driver.feature.replaceAll("_", " ")}</span>
                    <span className="driver-value">{driver.value}</span>
                    <span className="driver-direction" data-direction={driver.direction}>
                      {driver.direction === "increases_churn" ? "Raises score" : "Lowers score"}
                    </span>
                  </li>
                ))}
              </ol>
            </section>

            <section aria-labelledby={`${baseId}-tools-heading`}>
              <div className="scenario-section-heading">
                <span>02</span>
                <h4 id={`${baseId}-tools-heading`}>Tool path</h4>
              </div>
              <ol className="tool-path">
                {selected.tools.map((tool, index) => (
                  <li key={`${tool.name}-${index}`}>
                    <span className="tool-path-index">{index + 1}</span>
                    <div>
                      <strong>{tool.name.replaceAll("_", " ")}</strong>
                      <p>{tool.summary}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>

            <section className="policy-result" aria-labelledby={`${baseId}-gate-heading`}>
              <div className="scenario-section-heading">
                <span>03</span>
                <h4 id={`${baseId}-gate-heading`}>Policy gate</h4>
              </div>
              <div className="policy-rule-grid">
                {selected.policyRules.map((rule) => (
                  <article key={rule.id} data-passed={rule.passed}>
                    <div>
                      <strong>{rule.id}</strong>
                      <span>{rule.passed ? "Pass" : "Blocked"}</span>
                    </div>
                    <p>{rule.reason}</p>
                  </article>
                ))}
              </div>
            </section>

            <section className="final-decision" data-verdict={selected.verdict} aria-labelledby={`${baseId}-outcome-heading`}>
              <div className="scenario-section-heading">
                <span>04</span>
                <h4 id={`${baseId}-outcome-heading`}>Decision ledger</h4>
              </div>
              <span className="verdict-label">{verdictLabel(selected.verdict)}</span>
              <h4>{selected.actionLabel}</h4>
              <p>{selected.justification}</p>
              <dl>
                <div>
                  <dt>Agent confidence</dt>
                  <dd>{formatProbability(selected.confidence)}</dd>
                </div>
                <div>
                  <dt>Rule flags</dt>
                  <dd>{selected.flags.length ? selected.flags.join(" · ") : "None"}</dd>
                </div>
              </dl>
            </section>
          </div>

          <p className="recorded-note">
            Recorded and verified replay. The probability was captured from the local model and is rechecked by the dry-run evaluation. Viewing this panel does not rerun the agent, tools, or policy gate.
          </p>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
