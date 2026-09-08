"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { holds, BOARD_WIDTH, BOARD_HEIGHT } from "@/data/holds";
import type { HoldType, SelectedHold } from "@/lib/climb-types";
import Hold from "./Hold";
import HoldToolbar from "./HoldToolbar";

export interface ClimbingBoardProps {
  boardImageSrc?: string;
  onChange?: (holds: SelectedHold[]) => void;
}

export default function ClimbingBoard({
  boardImageSrc = "/board.png",
  onChange,
}: ClimbingBoardProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const [selected, setSelected] = useState<Record<number, HoldType>>({});
  const [activeHoldId, setActiveHoldId] = useState<number | null>(null);
  const [toolbarPos, setToolbarPos] = useState<{ x: number; y: number } | null>(null);

  const holdById = useMemo(() => new Map(holds.map((h) => [h.id, h])), []);

  const emitChange = useCallback(
    (next: Record<number, HoldType>) => {
      onChange?.(Object.entries(next).map(([id, type]) => ({ id: Number(id), type })));
    },
    [onChange]
  );

  // converts one hold's svg-space (cx, cy) into a pixel position relative
  // to the wrapper div, using the ratio between the svg's rendered width
  // and its viewBox width -- valid because the svg is a perfect square
  // scaled uniformly (width:100%, height:auto), so x and y share one scale
  const screenPointFor = useCallback(
    (id: number) => {
      const svg = svgRef.current;
      const wrapper = wrapperRef.current;
      const hold = holdById.get(id);
      if (!svg || !wrapper || !hold) return null;
      const svgRect = svg.getBoundingClientRect();
      const wrapperRect = wrapper.getBoundingClientRect();
      const scale = svgRect.width / BOARD_WIDTH;
      return {
        x: svgRect.left - wrapperRect.left + hold.cx * scale,
        y: svgRect.top - wrapperRect.top + hold.cy * scale,
      };
    },
    [holdById]
  );

  const handleSelect = useCallback(
    (id: number) => {
      setSelected((prev) => {
        const next = { ...prev };
        if (id in next) {
          // already selected: clicking again deselects it outright
          delete next[id];
          setActiveHoldId((current) => {
            if (current === id) setToolbarPos(null);
            return current === id ? null : current;
          });
        } else {
          // newly selected: default to "normal" and open the toolbar
          // so the user can immediately assign its real role
          next[id] = "normal";
          setActiveHoldId(id);
          setToolbarPos(screenPointFor(id));
        }
        emitChange(next);
        return next;
      });
    },
    [emitChange, screenPointFor]
  );

  const handleAssignType = useCallback(
    (type: HoldType) => {
      setActiveHoldId((current) => {
        if (current === null) return current;
        setSelected((prev) => {
          const next = { ...prev, [current]: type };
          emitChange(next);
          return next;
        });
        return current;
      });
    },
    [emitChange]
  );

  function closeToolbar() {
    setActiveHoldId(null);
    setToolbarPos(null);
  }

  return (
    <div ref={wrapperRef} className="relative w-full max-w-[780px]">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`}
        className="block h-auto w-full select-none"
        role="img"
        aria-label="Interactive climbing wall. Click a hold to add it to your climb."
      >
        {/* the real board photo, sharing this svg's own coordinate space
            so it can never drift out of alignment with the hold paths --
            one element to scale, instead of an <img> and an <svg> that
            have to be kept in sync separately */}
        <image href={boardImageSrc} x={0} y={0} width={BOARD_WIDTH} height={BOARD_HEIGHT} />

        {holds.map((hold) => (
          <Hold key={hold.id} hold={hold} type={selected[hold.id] ?? null} onSelect={handleSelect} />
        ))}
      </svg>

      {activeHoldId !== null && toolbarPos && (
        <HoldToolbar
          x={toolbarPos.x}
          y={toolbarPos.y}
          current={selected[activeHoldId] ?? "normal"}
          onChoose={handleAssignType}
          onClose={closeToolbar}
        />
      )}
    </div>
  );
}
