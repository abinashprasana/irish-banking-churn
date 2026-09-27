"use client";

import { type CSSProperties, useRef } from "react";

import { useRevealState } from "@/components/in-view";
import type { evidenceManifest } from "@/lib/evidence";

type Live = NonNullable<(typeof evidenceManifest)["evidence"]["redTeam"]["live"]>;

function actionLabel(action: string | null) {
  if (!action) return "No output (turn cap)";
  return action.replaceAll("_", " ");
}

export function LiveCheck({ live, modelLabel }: { live: Live; modelLabel: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const state = useRevealState(ref);
  const outputs = live.attacks.filter((attack) => attack.finalAction);
  const actions = new Set(outputs.map((attack) => attack.finalAction));
  const wordingFlags = live.attacks.filter((attack) => attack.harmIds.length > 0).length;
  const asked = live.attacks.filter((attack) => attack.unsafe && attack.harmIds.length > 0).length;
  const choice =
    actions.size === 1
      ? `In every run with an output, the model chose the ${actionLabel(outputs[0].finalAction)}.`
      : `The model produced an output in ${outputs.length} of ${live.attacksRun} runs.`;
  const gate = outputs.every((attack) => attack.verdict === "approved")
    ? " It never proposed a blocked offer, so the gate had nothing to stop."
    : "";
  const wording =
    wordingFlags > 0
      ? ` Its weak spot is wording: it called an offer compliant in ${wordingFlags} of ${live.attacksRun} runs${
          asked === 1 ? ", once because the attack asked it to" : asked > 1 ? `, ${asked} times because the attack asked` : ""
        }.`
      : "";

  return (
    <div className="live-check" ref={ref} data-inview={state}>
      <div className="live-check__head">
        <span className="mono-label">{`Live check · ${live.attacksRun} attacks · ${modelLabel}`}</span>
        <h3>What a live model did under attack</h3>
        <p>{choice + gate + wording}</p>
      </div>
      <ol className="live-check__cards">
        {live.attacks.map((attack, index) => (
          <li
            key={attack.id}
            data-unsafe={attack.unsafe}
            data-flagged={attack.harmIds.length > 0}
            style={{ "--i": index } as CSSProperties}
          >
            <div className="live-check__meta">
              <span>{attack.id}</span>
              <span>{attack.family.replaceAll("_", " ")}</span>
            </div>
            <p className="live-check__tried">&ldquo;{attack.input}&rdquo;</p>
            <p className="live-check__did">
              <span>Model</span> {actionLabel(attack.finalAction)}
            </p>
            {attack.harmIds.length > 0 ? (
              <span className="live-check__flag">
                {attack.unsafe ? "Unsafe" : "Wording flagged"} · {attack.harmIds.join(", ")}
              </span>
            ) : (
              <span className="live-check__flag" data-clean="true">
                Held
              </span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
