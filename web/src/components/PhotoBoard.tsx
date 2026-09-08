"use client";

import { useEffect, useState, type CSSProperties } from "react";
import { holds as photoHolds } from "@/data/holds";
import { loadPhotoHoldMap, type GridCell } from "@/lib/photo-hold-map";
import { loadUntrainableHoldIds } from "@/lib/untrainable-holds";
import { holdKey, type Role } from "@/lib/grid";

interface PhotoBoardProps {
  holdState: Record<string, Role>;
  onToggleHold: (col: number, row: number) => void;
  /** the role a click would assign right now -- unselected holds preview a
   * glow in this color on hover, so you can see what you're about to place
   * before you commit to it. */
  currentRole: Role;
}

const ROLE_COLOR: Record<Role, string> = {
  start: "var(--start)",
  middle: "var(--middle)",
  finish: "var(--finish)",
  foot: "var(--foot)",
};

// this is the actual physical Kilter board -- these hold outlines (see
// HOLD_COUNT in data/holds.ts) were detected from a clean board render
// (scripts/detect_holds.py) and calibrated against the same database
// coordinates hold_mapping.py uses for the trained model (see
// scripts/calibrate_holds.py). the source photo is expected to already be
// cropped to the trainable region (real y <= 152 -- no climb in the
// training data has ever used a hold above that), so every detected hold
// should get a confident match and be clickable; untrainable_hold_ids.json
// (fetched below) is normally empty and only drops something from the
// render if a future source image brings dead-zone rows back into frame.
export default function PhotoBoard({ holdState, onToggleHold, currentRole }: PhotoBoardProps) {
  const [map, setMap] = useState<Record<string, GridCell> | null>(null);
  const [untrainable, setUntrainable] = useState<Set<number> | null>(null);

  useEffect(() => {
    loadPhotoHoldMap().then(setMap);
    loadUntrainableHoldIds().then(setUntrainable);
  }, []);

  const visibleHolds = untrainable ? photoHolds.filter((hold) => !untrainable.has(hold.id)) : photoHolds;

  return (
    <svg
      viewBox="0 0 458 458"
      role="img"
      aria-label="The real Kilter board. Click a hold to add it to your climb."
      className="w-full h-auto rounded-[10px]"
    >
      <image href="/board.png" x={0} y={0} width={458} height={458} />

      {visibleHolds.map((hold) => {
        const cell = map?.[String(hold.id)];
        const role = cell ? holdState[holdKey(cell.col, cell.row)] : undefined;
        const clickable = Boolean(cell);

        const glowColor = role ? ROLE_COLOR[role] : ROLE_COLOR[currentRole];

        return (
          <g
            key={hold.id}
            className={clickable ? "cursor-pointer group" : undefined}
            onClick={clickable ? () => onToggleHold(cell!.col, cell!.row) : undefined}
            style={clickable ? ({ "--glow-color": glowColor } as CSSProperties) : undefined}
          >
            {/* enlarged invisible hit area for small holds */}
            {clickable && <path d={hold.path} fill="transparent" stroke="transparent" strokeWidth={6} />}

            <path
              d={hold.path}
              fill={role ? ROLE_COLOR[role] : "transparent"}
              fillOpacity={role ? 0.75 : clickable ? 0 : 0.15}
              stroke={clickable ? "transparent" : "var(--ink-faint)"}
              strokeWidth={clickable ? 0 : 0.5}
              className={
                clickable
                  ? `hold-glow transition-[fill-opacity] duration-100 ${role ? "hold-glow-selected" : "hold-glow-hover"}`
                  : undefined
              }
            />

            {clickable && (
              <path
                d={hold.path}
                fill="#ffffff"
                className="opacity-0 transition-opacity duration-100 group-hover:opacity-25"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
