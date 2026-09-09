"use client";

export interface PredictionResult {
  probs: number[];
  // a lightweight gate model rejects climbs it has no real basis to judge
  // (too few holds, or a gap far beyond anything in the training data) --
  // see generate_gate_dataset.py. probs is still populated when this is
  // false, but it's an unsupported extrapolation, not a real prediction.
  valid: boolean;
  validConfidence: number;
  exampleName: string | null;
  trueGrade: number | null;
}

interface ResultPanelProps {
  result: PredictionResult | null;
}

export default function ResultPanel({ result }: ResultPanelProps) {
  return (
    <div className="mt-3 flex w-[460px] flex-none flex-col gap-3.5">
      {!result ? (
        <p className="text-sm italic" style={{ color: "var(--ink-faint)" }}>
          place some holds, then click &quot;predict grade&quot;
        </p>
      ) : !result.valid ? (
        <InvalidClimb result={result} />
      ) : (
        <div className="flex flex-col gap-3.5">
          <Verdict result={result} />
          {result.exampleName && (
            <div className="flex gap-4 font-mono text-xs" style={{ color: "var(--ink-muted)" }}>
              <span>
                example: <b style={{ fontSize: 13, color: "var(--ink)" }}>{result.exampleName}</b>
              </span>
              <span>
                true grade: <b style={{ fontSize: 13, color: "var(--ink)" }}>V{result.trueGrade}</b>
              </span>
            </div>
          )}
          <Bars probs={result.probs} />
        </div>
      )}
    </div>
  );
}

function InvalidClimb({ result }: { result: PredictionResult }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span
        className="font-sans text-2xl font-bold leading-none"
        style={{ fontFamily: "var(--font-display)", color: "var(--finish)" }}
      >
        not climbable
      </span>
      <span className="max-w-[70ch] text-sm leading-snug" style={{ color: "var(--ink-muted)" }}>
        this hold selection doesn&apos;t look like anything in the training data -- too few holds, or a gap between
        holds far beyond any real climb. the grade model has no real basis to judge it, so no grade is shown.
      </span>
      <span className="font-mono text-xs" style={{ color: "var(--ink-faint)" }}>
        {(result.validConfidence * 100).toFixed(1)}% confidence in that call
      </span>
    </div>
  );
}

function Verdict({ result }: { result: PredictionResult }) {
  const top = result.probs.indexOf(Math.max(...result.probs));
  return (
    <div className="flex flex-col gap-0.5">
      <span
        className="font-sans text-[40px] font-bold leading-none"
        style={{ fontFamily: "var(--font-display)", color: "var(--ink)" }}
      >
        V{top}
      </span>
      <span className="font-mono text-xs" style={{ color: "var(--ink-muted)" }}>
        {(result.probs[top] * 100).toFixed(1)}% confidence
      </span>
    </div>
  );
}

function Bars({ probs }: { probs: number[] }) {
  const top = probs.indexOf(Math.max(...probs));
  return (
    <div className="flex flex-col gap-[3px]">
      {probs.map((p, g) => {
        const pct = p * 100;
        return (
          <div key={g} className="grid grid-cols-[28px_1fr_40px] items-center gap-1.5">
            <span className="text-right font-mono text-[11px]" style={{ color: "var(--ink-faint)" }}>
              V{g}
            </span>
            <div className="h-2.5 overflow-hidden rounded" style={{ background: "var(--edge)" }}>
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.max(pct, 0.5)}%`,
                  background: g === top ? "var(--ink)" : "var(--ink-faint)",
                }}
              />
            </div>
            <span className="text-right font-mono text-[10.5px]" style={{ color: "var(--ink-muted)" }}>
              {pct.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
