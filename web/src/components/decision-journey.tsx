"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";

import type { UiScenario } from "@/lib/content-types";

const steps = [
  {
    label: "Customer signals",
    title: "Begin with observable profile inputs",
    copy: "A synthetic profile supplies the same 19 fields used by the fitted model. Governance flags stay separate from prediction inputs.",
  },
  {
    label: "Churn score",
    title: "Estimate risk, without turning a score into a decision",
    copy: "XGBoost returns a probability. The score identifies a case for review; it does not authorize an offer or customer contact.",
  },
  {
    label: "Model evidence",
    title: "Expose what shaped this prediction",
    copy: "Local SHAP evidence ranks the model inputs that moved its raw output. It is an explanation of the model, not proof of causality.",
  },
  {
    label: "Policy gate",
    title: "Test the proposed action against deterministic rules",
    copy: "All four local rules run for the exact customer-action pair. The language model cannot override a failed result.",
  },
  {
    label: "Advisor decision",
    title: "Return a governed outcome, not an automated instruction",
    copy: "Passing actions still require human judgement. Blocked actions become structured refusals with the failed rule IDs preserved.",
  },
];

function formatProbability(value: number) {
  return new Intl.NumberFormat("en-IE", {
    style: "percent",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function DecisionJourney({ scenario }: { scenario: UiScenario }) {
  const [activeStep, setActiveStep] = useState(0);
  const stepRefs = useRef<Array<HTMLElement | null>>([]);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    const elements = stepRefs.current.filter(Boolean) as HTMLElement[];
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const index = Number((visible.target as HTMLElement).dataset.step);
        setActiveStep(index);
      },
      { rootMargin: "-34% 0px -42% 0px", threshold: [0.2, 0.5, 0.8] },
    );

    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="journey-layout">
      <div className="journey-steps">
        {steps.map((step, index) => (
          <article
            className="journey-step"
            data-step={index}
            data-active={activeStep === index}
            key={step.label}
            ref={(element) => {
              stepRefs.current[index] = element;
            }}
          >
            <span className="journey-index mono-label">0{index + 1}</span>
            <div>
              <p className="journey-label">{step.label}</p>
              <h3>{step.title}</h3>
              <p>{step.copy}</p>
            </div>
          </article>
        ))}
      </div>

      <aside className="journey-record">
        <div className="journey-record-topline">
          <span className="mono-label">CASE / {scenario.customerId}</span>
          <span>Recorded</span>
        </div>

        <div className="record-progress" aria-hidden="true">
          {steps.map((step, index) => (
            <span key={step.label} data-complete={index <= activeStep} />
          ))}
        </div>

        <motion.div
          className="record-content"
          key={activeStep}
          initial={reduceMotion ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="record-stage">{steps[activeStep].label}</span>

          {activeStep === 0 && (
            <dl className="record-facts">
              {scenario.profileFacts.slice(0, 4).map((fact) => (
                <div key={fact.label}>
                  <dt>{fact.label}</dt>
                  <dd>{fact.value}</dd>
                </div>
              ))}
            </dl>
          )}

          {activeStep === 1 && (
            <div className="record-score">
              <strong>{formatProbability(scenario.probability)}</strong>
              <span>Phase 1 churn probability</span>
              <small>Captured from model.predict_proba</small>
            </div>
          )}

          {activeStep === 2 && (
            <ol className="record-drivers">
              {scenario.drivers.slice(0, 3).map((driver, index) => (
                <li key={driver.feature}>
                  <span>{index + 1}</span>
                  <div>
                    <strong>{driver.feature.replaceAll("_", " ")}</strong>
                    <small>{driver.value}</small>
                  </div>
                  <em data-direction={driver.direction}>
                    {driver.direction === "increases_churn" ? "↑" : "↓"}
                  </em>
                </li>
              ))}
            </ol>
          )}

          {activeStep === 3 && (
            <ul className="record-rules">
              {scenario.policyRules.map((rule) => (
                <li key={rule.id} data-passed={rule.passed}>
                  <span>{rule.passed ? "Pass" : "Block"}</span>
                  <strong>{rule.id}</strong>
                </li>
              ))}
            </ul>
          )}

          {activeStep === 4 && (
            <div className="record-outcome" data-verdict={scenario.verdict}>
              <span>{scenario.verdict === "approved" ? "Passed local gate" : "Blocked"}</span>
              <strong>{scenario.actionLabel}</strong>
              <p>{scenario.justification}</p>
            </div>
          )}
        </motion.div>

        <p className="record-disclaimer">
          Synthetic customer and policy scenario. An advisor remains accountable.
        </p>
      </aside>
    </div>
  );
}
