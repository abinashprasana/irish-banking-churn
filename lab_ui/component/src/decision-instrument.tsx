import React, { useEffect, useId, useLayoutEffect, useMemo, useState } from "react";
import { createRoot, Root } from "react-dom/client";

type DecisionState =
  | "draft"
  | "evaluating"
  | "scored"
  | "stale"
  | "agent_ready"
  | "approved"
  | "blocked"
  | "review_required"
  | "unavailable";

type StepStatus = "pending" | "active" | "complete" | "blocked" | "review";

type Step = {
  id: string;
  label: string;
  summary: string;
  detail?: string;
  status?: StepStatus;
};

type ProvenanceItem = { label: string; value: string };
type Verdict = {
  status: "approved" | "blocked" | "review_required" | "unavailable";
  label: string;
  detail?: string;
};

type InstrumentData = {
  variant: "assessment" | "governance";
  stage: DecisionState;
  steps: Step[];
  selected_step: string;
  score: number | null;
  thresholds: { lower: number; higher: number };
  provenance: ProvenanceItem[];
  verdict: Verdict | null;
  rule_ids: string[];
};

type StreamlitComponent = {
  data: unknown;
  parentElement: ParentNode;
  setStateValue: (name: string, value: unknown) => void;
};

const roots = new WeakMap<object, Root>();
const stateLabels: Record<DecisionState, string> = {
  draft: "Draft",
  evaluating: "Evaluating",
  scored: "Scored",
  stale: "Recalculation needed",
  agent_ready: "Agent ready",
  approved: "Approved",
  blocked: "Policy blocked",
  review_required: "Advisor review",
  unavailable: "Unavailable",
};

function safeText(value: unknown, limit = 600): string {
  return String(value ?? "").trim().slice(0, limit);
}

function safeData(value: unknown): InstrumentData {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const permittedStates = new Set(Object.keys(stateLabels));
  const stage = permittedStates.has(String(raw.stage))
    ? (String(raw.stage) as DecisionState)
    : "draft";
  const variant = raw.variant === "governance" ? "governance" : "assessment";
  const statuses = new Set(["pending", "active", "complete", "blocked", "review"]);
  const rawSteps = Array.isArray(raw.steps) ? raw.steps : [];
  const steps = rawSteps.slice(0, 8).map((entry, index) => {
    const item = entry && typeof entry === "object" ? (entry as Record<string, unknown>) : {};
    return {
      id: safeText(item.id, 48) || `step-${index + 1}`,
      label: safeText(item.label, 64) || `Step ${index + 1}`,
      summary: safeText(item.summary, 240),
      detail: safeText(item.detail, 600),
      status: statuses.has(String(item.status)) ? (String(item.status) as StepStatus) : "pending",
    };
  });
  const hasScore = raw.score !== null && raw.score !== undefined && raw.score !== "";
  const scoreValue = hasScore ? Number(raw.score) : Number.NaN;
  const thresholdSource =
    raw.thresholds && typeof raw.thresholds === "object"
      ? (raw.thresholds as Record<string, unknown>)
      : {};
  const lower = Math.min(1, Math.max(0, Number(thresholdSource.lower) || 0.3));
  const higher = Math.min(1, Math.max(lower, Number(thresholdSource.higher) || 0.6));
  const provenanceSource = Array.isArray(raw.provenance) ? raw.provenance : [];
  const provenance = provenanceSource.slice(0, 6).map((entry) => {
    const item = entry && typeof entry === "object" ? (entry as Record<string, unknown>) : {};
    return { label: safeText(item.label, 64), value: safeText(item.value, 120) };
  });
  const verdictSource =
    raw.verdict && typeof raw.verdict === "object"
      ? (raw.verdict as Record<string, unknown>)
      : null;
  const verdictStatuses = new Set(["approved", "blocked", "review_required", "unavailable"]);
  const verdict = verdictSource
    ? {
        status: verdictStatuses.has(String(verdictSource.status))
          ? (String(verdictSource.status) as Verdict["status"])
          : "unavailable",
        label: safeText(verdictSource.label, 100),
        detail: safeText(verdictSource.detail, 400),
      }
    : null;
  return {
    variant,
    stage,
    steps,
    selected_step: safeText(raw.selected_step, 48),
    score: Number.isFinite(scoreValue) ? Math.min(1, Math.max(0, scoreValue)) : null,
    thresholds: { lower, higher },
    provenance,
    verdict,
    rule_ids: (Array.isArray(raw.rule_ids) ? raw.rule_ids : [])
      .slice(0, 8)
      .map((item) => safeText(item, 64)),
  };
}

function ShieldIcon({ status }: { status: Verdict["status"] }): React.JSX.Element {
  return (
    <svg className="verdict__shield" viewBox="0 0 32 32" aria-hidden="true">
      <path
        d="M16 3.5 26 7v7.1c0 6.6-3.8 11.6-10 14.4-6.2-2.8-10-7.8-10-14.4V7l10-3.5Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      {status === "approved" ? (
        <path d="m11.3 16.1 3.1 3 6.5-7" fill="none" stroke="currentColor" strokeWidth="1.5" />
      ) : status === "blocked" ? (
        <path d="m12 12 8 8m0-8-8 8" fill="none" stroke="currentColor" strokeWidth="1.5" />
      ) : status === "review_required" ? (
        <path d="M16 10.5v7m0 3.5v.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
      ) : (
        <path d="M11.5 16h9" fill="none" stroke="currentColor" strokeWidth="1.5" />
      )}
    </svg>
  );
}

function ScorePanel({ score, thresholds }: Pick<InstrumentData, "score" | "thresholds">) {
  if (score === null) return null;
  const percent = Math.round(score * 1000) / 10;
  const label = score < thresholds.lower ? "Lower band" : score < thresholds.higher ? "Middle band" : "Higher band";
  return (
    <section className="score-panel" aria-label="Churn score">
      <div>
        <div className="score-panel__value">{percent.toFixed(1)}%</div>
        <div className="score-panel__label">Churn probability · {label}</div>
      </div>
      <div className="score-panel__track">
        <div
          className="score-track"
          role="progressbar"
          aria-label="Churn probability"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          aria-valuetext={`${percent.toFixed(1)} percent, ${label}`}
        >
          <div
            className="score-track__fill"
            style={{ "--score-ratio": String(score) } as React.CSSProperties}
          />
          <span
            className="score-track__marker"
            data-label={`${Math.round(thresholds.lower * 100)}%`}
            style={{ left: `${thresholds.lower * 100}%` }}
          />
          <span
            className="score-track__marker"
            data-label={`${Math.round(thresholds.higher * 100)}%`}
            style={{ left: `${thresholds.higher * 100}%` }}
          />
        </div>
      </div>
    </section>
  );
}

function DecisionInstrument({
  data,
  onSelect,
}: {
  data: InstrumentData;
  onSelect: (step: string) => void;
}) {
  const firstId = data.steps[0]?.id ?? "";
  const requestedId = data.steps.some((step) => step.id === data.selected_step)
    ? data.selected_step
    : firstId;
  const [selectedId, setSelectedId] = useState(requestedId);
  const [detailVisible, setDetailVisible] = useState(false);
  const detailId = useId();
  useEffect(() => setSelectedId(requestedId), [requestedId]);
  useLayoutEffect(() => {
    setDetailVisible(false);
    const frame = window.requestAnimationFrame(() => setDetailVisible(true));
    return () => window.cancelAnimationFrame(frame);
  }, [selectedId]);
  const selected = useMemo(
    () => data.steps.find((step) => step.id === selectedId) ?? data.steps[0],
    [data.steps, selectedId],
  );
  const title = data.variant === "governance" ? "Governed decision trace" : "Evidence pathway";

  return (
    <section className="instrument" aria-label={title}>
      <header className="instrument__header">
        <div>
          <p className="instrument__eyebrow">
            {data.variant === "governance" ? "Decision instrument" : "Assessment instrument"}
          </p>
          <h3 className="instrument__title">{title}</h3>
        </div>
        <span className="state-badge" data-state={data.stage} role="status">
          {stateLabels[data.stage]}
        </span>
      </header>

      <ScorePanel score={data.score} thresholds={data.thresholds} />

      <div className="instrument__body">
        <nav aria-label={data.variant === "governance" ? "Governance timeline" : "Evidence stages"}>
          <ol className="timeline">
            {data.steps.map((step, index) => (
              <li className="timeline__item" key={step.id}>
                <button
                  className="timeline__button"
                  type="button"
                  aria-current={step.id === selectedId ? "step" : undefined}
                  aria-controls={detailId}
                  onClick={() => {
                    setSelectedId(step.id);
                    onSelect(step.id);
                  }}
                >
                  <span className="timeline__ordinal">{String(index + 1).padStart(2, "0")}</span>
                  <span className="timeline__copy">
                    <span className="timeline__label">{step.label}</span>
                    <span className="timeline__summary">{step.summary}</span>
                  </span>
                  <span
                    className="timeline__status"
                    data-status={step.status}
                    aria-label={`${step.status} status`}
                    role="img"
                  />
                </button>
              </li>
            ))}
          </ol>
        </nav>

        <section className="detail" id={detailId} aria-live="polite">
          {selected ? (
            <div
              className="detail__content"
              data-visible={detailVisible ? "true" : "false"}
              key={selected.id}
            >
              <p className="instrument__micro-label">Selected stage</p>
              <h4 className="detail__title">{selected.label}</h4>
              <p className="detail__summary">{selected.summary}</p>
              {selected.detail ? <p className="detail__description">{selected.detail}</p> : null}

              {data.provenance.length ? (
                <dl className="provenance" aria-label="Decision provenance">
                  {data.provenance.map((item, index) => (
                    <div className="provenance__item" key={`${item.label}-${index}`}>
                      <dt>{item.label}</dt>
                      <dd>{item.value}</dd>
                    </div>
                  ))}
                </dl>
              ) : null}

              {data.verdict ? (
                <div className="verdict" data-status={data.verdict.status}>
                  <ShieldIcon status={data.verdict.status} />
                  <div>
                    <p className="verdict__label">{data.verdict.label}</p>
                    {data.verdict.detail ? (
                      <p className="verdict__detail">{data.verdict.detail}</p>
                    ) : null}
                    {data.rule_ids.length ? (
                      <div className="rules" aria-label="Applied policy rules">
                        {data.rule_ids.map((rule) => (
                          <span className="rule" key={rule}>{rule}</span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="detail__summary">No decision stages are available.</p>
          )}
        </section>
      </div>
    </section>
  );
}

export default function render(component: StreamlitComponent): void {
  const container = component.parentElement.querySelector<HTMLElement>(
    "[data-decision-instrument-root]",
  );
  if (!container) return;
  const parentKey = component.parentElement as object;
  let root = roots.get(parentKey);
  if (!root) {
    root = createRoot(container);
    roots.set(parentKey, root);
  }
  root.render(
    <DecisionInstrument
      data={safeData(component.data)}
      onSelect={(step) => component.setStateValue("selected_step", step)}
    />,
  );
}
