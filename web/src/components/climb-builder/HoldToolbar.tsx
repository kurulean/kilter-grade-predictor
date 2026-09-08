"use client";

import type { HoldType } from "@/lib/climb-types";

const OPTIONS: { type: HoldType; label: string; color: string }[] = [
  { type: "start", label: "Start", color: "var(--start)" },
  { type: "finish", label: "Finish", color: "var(--finish)" },
  { type: "foot", label: "Foot", color: "var(--foot)" },
  { type: "normal", label: "Middle", color: "var(--middle)" },
];

interface HoldToolbarProps {
  x: number;
  y: number;
  current: HoldType;
  onChoose: (type: HoldType) => void;
  onClose: () => void;
}

// positioned by the caller using screen coordinates already converted
// from the hold's svg-space cx/cy -- see ClimbingBoard's screenPointFor
export default function HoldToolbar({ x, y, current, onChoose, onClose }: HoldToolbarProps) {
  return (
    <div
      className="absolute z-10 flex -translate-x-1/2 -translate-y-[calc(100%+10px)] gap-1 rounded-md border p-1"
      style={{ left: x, top: y, borderColor: "var(--edge-strong)", background: "var(--bg)" }}
    >
      {OPTIONS.map((opt) => (
        <button
          key={opt.type}
          type="button"
          onClick={() => onChoose(opt.type)}
          title={opt.label}
          aria-pressed={current === opt.type}
          className="flex h-7 w-7 items-center justify-center rounded-md border-2 text-[10px] font-semibold text-white"
          style={{
            borderColor: current === opt.type ? "var(--ink)" : "transparent",
            background: opt.color,
          }}
        >
          {opt.label[0]}
        </button>
      ))}
      <button
        type="button"
        onClick={onClose}
        title="Done"
        className="ml-1 flex h-7 w-7 items-center justify-center rounded-md"
        style={{ color: "var(--ink-muted)" }}
      >
        ✓
      </button>
    </div>
  );
}
