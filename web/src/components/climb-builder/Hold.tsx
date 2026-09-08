"use client";

import { memo } from "react";
import type { Hold as HoldData, HoldType } from "@/lib/climb-types";

// fill color per assigned role, pulled from the app's shared design
// tokens (globals.css) so this matches everywhere else and gets
// dark-mode support for free
const TYPE_FILL: Record<HoldType, string> = {
  normal: "var(--middle)",
  start: "var(--start)",
  finish: "var(--finish)",
  foot: "var(--foot)",
};

interface HoldProps {
  hold: HoldData;
  type: HoldType | null; // null means not selected
  onSelect: (id: number) => void;
}

function Hold({ hold, type, onSelect }: HoldProps) {
  const isSelected = type !== null;

  return (
    <g
      className="group cursor-pointer"
      onClick={() => onSelect(hold.id)}
      role="button"
      aria-label={`Hold ${hold.id}${isSelected ? `, ${type}` : ""}`}
    >
      {/* enlarged invisible hit area so small holds stay easy to tap */}
      <path d={hold.path} fill="transparent" stroke="transparent" strokeWidth={10} />

      {/* selected-state fill, fully transparent until a role is assigned
          so an unselected hold shows the original photo untouched */}
      <path
        d={hold.path}
        fill={isSelected ? TYPE_FILL[type] : "transparent"}
        fillOpacity={isSelected ? 0.7 : 0}
        stroke={isSelected ? "#ffffff" : "transparent"}
        strokeWidth={isSelected ? 1.2 : 0}
        className="transition-[fill-opacity] duration-100"
      />

      {/* hover tint, gives feedback on both selected and unselected holds
          without touching the layer above -- pure css, no per-hold state */}
      <path
        d={hold.path}
        fill="#ffffff"
        className="opacity-0 transition-opacity duration-100 group-hover:opacity-25"
      />
    </g>
  );
}

// HOLD_COUNT of these render every time the board's selection state changes
// (see data/holds.ts). memo skips the ones that didn't actually change
// since props compare by value (id, type are primitives) rather than by
// object identity.
export default memo(Hold);
