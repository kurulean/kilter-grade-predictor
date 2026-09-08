"use client";

import type { GradeEstimate } from "@/lib/estimate-grade";
import type { HoldType } from "@/lib/climb-types";

const LEGEND: { type: HoldType; label: string; color: string }[] = [
  { type: "start", label: "Start", color: "var(--start)" },
  { type: "normal", label: "Middle", color: "var(--middle)" },
  { type: "finish", label: "Finish", color: "var(--finish)" },
  { type: "foot", label: "Foot", color: "var(--foot)" },
];

interface GradePanelProps {
  estimate: GradeEstimate | null;
  holdCount: number;
  onClear: () => void;
}

export default function GradePanel({ estimate, holdCount, onClear }: GradePanelProps) {
  return (
    <div className="flex w-full flex-none flex-col gap-4 sm:w-[240px]">
      <div>
        {estimate ? (
          <>
            <div className="flex items-baseline gap-2">
              <span
                className="text-[40px] leading-none"
                style={{ fontFamily: "var(--font-mono)", fontWeight: 500, color: "var(--accent-ink)" }}
              >
                {estimate.label}
              </span>
              <span className="font-mono text-[11px]" style={{ color: "var(--ink-faint)" }}>
                estimated
              </span>
            </div>
            <p className="mt-1.5 text-xs leading-snug" style={{ color: "var(--ink-muted)" }}>
              {estimate.reason}
            </p>
          </>
        ) : (
          <p className="text-sm italic" style={{ color: "var(--ink-faint)" }}>
            place at least 2 holds to estimate a grade
          </p>
        )}
      </div>

      <div className="font-mono text-xs" style={{ color: "var(--ink-muted)" }}>
        {holdCount} hold{holdCount === 1 ? "" : "s"} placed
      </div>

      <div className="flex flex-col gap-1.5">
        {LEGEND.map((item) => (
          <div
            key={item.type}
            className="flex items-center gap-2 font-mono text-[11px]"
            style={{ color: "var(--ink-muted)" }}
          >
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: item.color }} />
            {item.label}
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={onClear}
        className="self-start font-mono text-xs font-semibold underline underline-offset-2"
        style={{ color: "var(--ink-muted)" }}
      >
        clear board
      </button>
    </div>
  );
}
