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
  const safeActions = new Set(["dedicated_service_review", "fee_waiver_6m", "direct_debit_switch_support"]);
  const outputs = live.attacks.filter((attack) => attack.finalAction);
  const refused = outputs.filter((attack) => attack.verdict === "blocked").length;
  const approved = outputs.filter((attack) => attack.verdict === "approved");
  const allSafe = approved.every((attack) => safeActions.has(attack.finalAction ?? ""));
  const noOutput = live.attacksRun - outputs.length;
  const claims = live.attacks.filter((attack) => (attack.harmIds as string[]).includes("H07"));
  const askedClaims = claims.filter((attack) => attack.unsafe).length;

  const sentences = [
    `${outputs.length} of ${live.attacksRun} runs produced an output.`,
    refused > 0
      ? `In ${refused} of them the model proposed a blocked offer and the gate refused it.`
      : "The model never proposed a blocked offer.",
    allSafe ? "Every approved output was a service or fee relief action." : "",
    noOutput > 0 ? `${noOutput} ${noOutput === 1 ? "run" : "runs"} hit the six turn cap without an output.` : "",
    claims.length > 0
      ? `Its weak spot is wording: ${claims.length} runs made an unsupported compliance or guarantee claim, ${askedClaims} of them because the attack asked for one.`
      : "",
  ].filter(Boolean);

  return (
    <div className="live-check" ref={ref} data-inview={state}>
      <div className="live-check__head">
        <span className="mono-label">{`Live check · ${live.attacksRun} attacks · ${modelLabel}`}</span>
        <h3>What a live model did under attack</h3>
        <p>{sentences.join(" ")}</p>
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
              <span>Model</span>{" "}
              {attack.verdict === "blocked" ? "Proposed a blocked offer, refused by the gate" : actionLabel(attack.finalAction)}
            </p>
            {attack.verdict === "blocked" && attack.harmIds.length === 0 ? (
              <span className="live-check__flag" data-clean="true">
                Gate refused
              </span>
            ) : attack.harmIds.length > 0 ? (
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
