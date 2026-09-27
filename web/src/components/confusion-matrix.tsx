"use client";

import { type CSSProperties, useRef } from "react";

import { CountUp } from "@/components/count-up";
import { useRevealState } from "@/components/in-view";

type Confusion = {
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  truePositive: number;
};

export function ConfusionMatrix({ confusion }: { confusion: Confusion }) {
  const ref = useRef<HTMLDivElement>(null);
  const state = useRevealState(ref);
  const { trueNegative, falsePositive, falseNegative, truePositive } = confusion;
  const churners = truePositive + falseNegative;

  const cells = [
    { key: "tp", label: "Churned, flagged", value: truePositive, tone: "found" },
    { key: "fn", label: "Churned, missed", value: falseNegative, tone: "missed" },
    { key: "fp", label: "Stayed, flagged", value: falsePositive, tone: "extra" },
    { key: "tn", label: "Stayed, not flagged", value: trueNegative, tone: "quiet" },
  ];

  return (
    <div className="confusion" ref={ref} data-inview={state}>
      <div className="confusion__copy">
        <p className="eyebrow">Holdout outcomes</p>
        <h3>What the flags look like on real cases</h3>
        <p>
          {`Of ${churners} churners in the holdout, the model flagged ${truePositive} and missed ${falseNegative}. It also flagged ${falsePositive} customers who stayed. That is why a flag goes to an advisor first.`}
        </p>
      </div>
      <figure className="confusion__grid" aria-label="XGBoost holdout confusion matrix">
        <div className="confusion__axis confusion__axis--top" aria-hidden="true">
          <span>Flagged</span>
          <span>Not flagged</span>
        </div>
        <div className="confusion__axis confusion__axis--side" aria-hidden="true">
          <span>Churned</span>
          <span>Stayed</span>
        </div>
        <div className="confusion__cells">
          {cells.map((cell, index) => (
            <div
              key={cell.key}
              className="confusion__cell"
              data-tone={cell.tone}
              style={{ "--i": index } as CSSProperties}
            >
              <strong>
                <CountUp value={cell.value} format="integer" />
              </strong>
              <span>{cell.label}</span>
            </div>
          ))}
        </div>
        <figcaption>2,000 held-out synthetic profiles · decision threshold 0.5</figcaption>
      </figure>
    </div>
  );
}
