"use client";

import { motion, useReducedMotion } from "motion/react";

const reveal = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0 },
};

export function DecisionStack() {
  const reduceMotion = useReducedMotion();
  const initial = reduceMotion ? "visible" : "hidden";

  return (
    <div className="decision-stack" aria-labelledby="decision-stack-title">
      <div className="decision-stack-heading">
        <span className="mono-label">SYSTEM / GOVERNED RETENTION</span>
        <span className="live-evidence">Recorded evidence</span>
      </div>

      <svg
        viewBox="0 0 680 520"
        role="img"
        aria-labelledby="decision-stack-title decision-stack-description"
      >
        <title id="decision-stack-title">
          Evidence flows through a model and deterministic policy gate
        </title>
        <desc id="decision-stack-description">
          Synthetic customer signals enter an XGBoost churn model, model evidence
          is inspected, a retention action is checked against four local rules, and
          only a governed outcome reaches advisor review.
        </desc>

        <defs>
          <pattern
            id="ledger-grid"
            width="28"
            height="28"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 28 0 L 0 0 0 28"
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.08"
              strokeWidth="1"
            />
          </pattern>
        </defs>

        <rect className="stack-grid" x="0" y="0" width="680" height="520" />

        <motion.g
          variants={reveal}
          initial={initial}
          animate="visible"
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
        >
          <text className="stack-kicker" x="42" y="70">
            SYNTHETIC CUSTOMER SIGNALS
          </text>
          <g className="signal-rows">
            <line x1="42" y1="95" x2="168" y2="95" />
            <line x1="42" y1="119" x2="137" y2="119" />
            <line x1="42" y1="143" x2="186" y2="143" />
            <line x1="42" y1="167" x2="116" y2="167" />
          </g>
          <text className="stack-note" x="42" y="200">
            19 model inputs
          </text>
        </motion.g>

        <motion.path
          className="flow-line flow-line-primary"
          d="M198 132H248"
          initial={{ pathLength: reduceMotion ? 1 : 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.08 }}
        />

        <motion.g
          variants={reveal}
          initial={initial}
          animate="visible"
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
        >
          <rect className="stack-panel" x="248" y="60" width="186" height="144" />
          <text className="stack-kicker" x="270" y="91">
            CHURN MODEL
          </text>
          <text className="stack-value" x="270" y="139">
            84.2%
          </text>
          <text className="stack-note" x="270" y="169">
            average precision
          </text>
          <path className="spark-line" d="M270 188L301 176L328 181L357 153L389 158L412 133" />
        </motion.g>

        <motion.path
          className="flow-line flow-line-secondary"
          d="M340 204V250"
          initial={{ pathLength: reduceMotion ? 1 : 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        />

        <motion.g
          variants={reveal}
          initial={initial}
          animate="visible"
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        >
          <rect className="stack-panel stack-panel-offset" x="218" y="250" width="244" height="112" />
          <text className="stack-kicker" x="240" y="282">
            MODEL EVIDENCE
          </text>
          <g className="evidence-bars">
            <rect x="240" y="303" width="158" height="8" />
            <rect x="240" y="321" width="93" height="8" />
            <rect x="240" y="339" width="72" height="8" />
          </g>
          <text className="stack-note" x="414" y="310">
            SHAP
          </text>
        </motion.g>

        <motion.path
          className="flow-line flow-line-primary"
          d="M462 306H508"
          initial={{ pathLength: reduceMotion ? 1 : 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
        />

        <motion.g
          variants={reveal}
          initial={initial}
          animate="visible"
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
        >
          <circle className="gate-ring" cx="561" cy="306" r="53" />
          <path className="gate-check" d="M538 307L553 322L584 285" />
          <text className="stack-kicker gate-label" x="561" y="384">
            POLICY GATE
          </text>
          <text className="stack-note gate-label" x="561" y="405">
            4 deterministic rules
          </text>
        </motion.g>

        <motion.path
          className="flow-line flow-line-secondary"
          d="M561 417V450H421"
          initial={{ pathLength: reduceMotion ? 1 : 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
        />

        <motion.g
          variants={reveal}
          initial={initial}
          animate="visible"
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
        >
          <rect className="outcome-panel" x="222" y="424" width="199" height="62" />
          <text className="stack-kicker" x="245" y="450">
            GOVERNED OUTCOME
          </text>
          <text className="stack-note" x="245" y="473">
            Advisor remains accountable
          </text>
        </motion.g>
      </svg>

      <div className="decision-stack-footer">
        <span>Synthetic inputs</span>
        <span>Deterministic gate</span>
        <span>Human decision</span>
      </div>
    </div>
  );
}
