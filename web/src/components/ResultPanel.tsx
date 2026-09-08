"use client";

export interface PredictionResult {
  probs: number[];
  exampleName: string | null;
  trueGrade: number | null;
}

interface ResultPanelProps {
  result: PredictionResult | null;
}

export default function ResultPanel({ result }: ResultPanelProps) {
  return (
    <div
      className="flex w-[460px] flex-none flex-col gap-3.5 rounded-[10px] border p-4"
      style={{ background: "var(--surface)", borderColor: "var(--edge)" }}
    >
      {!result ? (
        <p className="text-sm italic" style={{ color: "var(--ink-faint)" }}>
          place some holds, then click &quot;predict grade&quot;
        </p>
      ) : (
        <div className="flex gap-4">
          <div className="flex w-[150px] flex-none flex-col gap-3.5">
            <Verdict result={result} />
            {result.exampleName && (
              <div className="flex flex-col gap-1 font-mono text-xs" style={{ color: "var(--ink-muted)" }}>
                <span>
                  example: <b style={{ fontSize: 13, color: "var(--ink)" }}>{result.exampleName}</b>
                </span>
                <span>
                  true grade: <b style={{ fontSize: 13, color: "var(--ink)" }}>V{result.trueGrade}</b>
                </span>
              </div>
            )}
          </div>
          {/* GRADE_SCALE_TOP_OFFSET: bumps the bar chart down so it starts
              lower than the verdict beside it, instead of top-aligned --
              tweak this value directly to move it further up/down */}
          <div className="min-w-0 flex-1 pt-8">
            <Bars probs={result.probs} />
          </div>
        </div>
      )}
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
