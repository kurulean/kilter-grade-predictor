"use client";

import type { CSSProperties } from "react";
import { ROLES, type Role } from "@/lib/grid";

interface ControlsProps {
  currentRole: Role;
  onRoleChange: (role: Role) => void;
  angle: number;
  onAngleChange: (angle: number) => void;
  onClear: () => void;
  onExample: () => void;
  onPredict: () => void;
}

const ROLE_VAR: Record<Role, string> = {
  start: "var(--start)",
  middle: "var(--middle)",
  finish: "var(--finish)",
  foot: "var(--foot)",
};

export default function Controls({
  currentRole,
  onRoleChange,
  angle,
  onAngleChange,
  onClear,
  onExample,
  onPredict,
}: ControlsProps) {
  return (
    <div
      className="flex flex-wrap items-center gap-3.5 rounded-[10px] border p-4"
      style={{ background: "var(--surface)", borderColor: "var(--edge)" }}
    >
      <div className="flex gap-1.5">
        {ROLES.map((role) => (
          <button
            key={role}
            onClick={() => onRoleChange(role)}
            className="glow-btn flex items-center gap-1.5 rounded-md border-2 px-3 py-2 font-mono text-xs font-semibold"
            style={{
              borderColor: currentRole === role ? ROLE_VAR[role] : "var(--edge-strong)",
              color: currentRole === role ? ROLE_VAR[role] : "var(--ink-muted)",
              background: "var(--surface)",
              "--glow-color": ROLE_VAR[role],
            } as CSSProperties}
          >
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: ROLE_VAR[role] }} />
            {role}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2 font-mono text-xs" style={{ color: "var(--ink-muted)" }}>
        angle
        <input
          type="range"
          min={0}
          max={70}
          step={5}
          value={angle}
          onChange={(e) => onAngleChange(Number(e.target.value))}
          className="accent-[var(--accent)]"
        />
        <span className="inline-block min-w-[34px] font-semibold" style={{ color: "var(--ink)" }}>
          {angle}&deg;
        </span>
      </div>

      <button
        onClick={onClear}
        className="glow-btn rounded-md border px-3 py-2 font-mono text-xs font-semibold"
        style={{
          borderColor: "var(--edge-strong)",
          color: "var(--ink-muted)",
          background: "var(--surface)",
          "--glow-color": "var(--ink-muted)",
        } as CSSProperties}
      >
        clear board
      </button>
      <button
        onClick={onExample}
        className="glow-btn rounded-md border px-3 py-2 font-mono text-xs font-semibold"
        style={{
          borderColor: "var(--edge-strong)",
          color: "var(--ink-muted)",
          background: "var(--surface)",
          "--glow-color": "var(--ink-muted)",
        } as CSSProperties}
      >
        load example
      </button>
      <button
        onClick={onPredict}
        className="glow-btn rounded-md border px-3 py-2 font-mono text-xs font-semibold text-white"
        style={{ borderColor: "var(--accent)", background: "var(--accent)", "--glow-color": "var(--accent)" } as CSSProperties}
      >
        predict grade
      </button>
    </div>
  );
}
