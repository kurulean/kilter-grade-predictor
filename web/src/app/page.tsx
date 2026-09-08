"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import PhotoBoard from "@/components/PhotoBoard";
import Controls from "@/components/Controls";
import ResultPanel, { type PredictionResult } from "@/components/ResultPanel";
import { predictGrade } from "@/lib/api";
import { holdKey, CODE_TO_ROLE, ROLE_TO_CODE, type Role } from "@/lib/grid";

const EXAMPLE = {
  name: "Floats Your Boat",
  trueGrade: 0,
  angle: 30,
  cells: [
    [19, 5, 15], [21, 7, 15], [13, 9, 15], [21, 11, 15], [13, 13, 15],
    [19, 15, 12], [23, 15, 12], [23, 19, 13], [15, 21, 13], [25, 21, 15],
    [19, 27, 13], [19, 29, 13], [23, 33, 13], [23, 37, 14], [27, 37, 14],
  ] as [number, number, number][],
};

export default function Home() {
  const [holdState, setHoldState] = useState<Record<string, Role>>({});
  const [currentRole, setCurrentRole] = useState<Role>("start");
  const [angle, setAngle] = useState(30);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const exampleRef = useRef<{ name: string; trueGrade: number } | null>(null);

  const predict = useCallback(async (state: Record<string, Role>, angleVal: number) => {
    const cells: [number, number, number][] = Object.entries(state).map(([key, role]) => {
      const [col, row] = key.split(",").map(Number);
      return [col, row, ROLE_TO_CODE[role]];
    });

    setPredicting(true);
    setError(null);
    try {
      const probs = await predictGrade(cells, angleVal);
      setResult({
        probs,
        exampleName: exampleRef.current?.name ?? null,
        trueGrade: exampleRef.current?.trueGrade ?? null,
      });
    } catch {
      setError("couldn't reach the model server -- is `python serve.py` running?");
    } finally {
      setPredicting(false);
    }
  }, []);

  const loadExample = useCallback(() => {
    const newState: Record<string, Role> = {};
    EXAMPLE.cells.forEach(([col, row, code]) => {
      newState[holdKey(col, row)] = CODE_TO_ROLE[code];
    });
    setHoldState(newState);
    setAngle(EXAMPLE.angle);
    exampleRef.current = { name: EXAMPLE.name, trueGrade: EXAMPLE.trueGrade };
    predict(newState, EXAMPLE.angle);
  }, [predict]);

  // load the example once on mount, so the page opens in a working
  // state rather than an empty board
  useEffect(() => {
    loadExample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggleHold(col: number, row: number) {
    const key = holdKey(col, row);
    setHoldState((prev) => {
      const next = { ...prev };
      if (next[key] === currentRole) {
        delete next[key];
      } else {
        next[key] = currentRole;
      }
      return next;
    });
    exampleRef.current = null;
  }

  function clearBoard() {
    setHoldState({});
    exampleRef.current = null;
    setResult(null);
    setError(null);
  }

  const holdCount = Object.keys(holdState).length;

  return (
    <div className="mx-auto flex max-w-[1200px] flex-col gap-5 px-5 py-8 pb-16">
      <div>
        <h1
          className="text-[clamp(28px,5vw,40px)] leading-[1.05]"
          style={{ fontFamily: "var(--font-mono)", fontWeight: 500 }}
        >
          Kilter grade predictor
        </h1>
        <p className="mt-2 max-w-[64ch] text-[15px] leading-[1.55]" style={{ color: "var(--ink-muted)" }}>
          Place holds on the real board, pick an angle, and get a V-grade prediction from the
          actual trained model, running on a real backend.
        </p>
      </div>

      <Controls
        currentRole={currentRole}
        onRoleChange={setCurrentRole}
        angle={angle}
        onAngleChange={setAngle}
        onClear={clearBoard}
        onExample={loadExample}
        onPredict={() => predict(holdState, angle)}
      />

      <div className="flex flex-wrap items-start gap-4.5">
        <figure className="-mt-4 mb-0 min-w-[280px] flex-[1_1_500px]">
          <PhotoBoard holdState={holdState} onToggleHold={toggleHold} currentRole={currentRole} />
          <figcaption className="pt-2 text-center font-mono text-xs" style={{ color: "var(--ink-faint)" }}>
            {holdCount} hold{holdCount === 1 ? "" : "s"} placed
            {predicting && " · predicting…"}
          </figcaption>
          {error && (
            <p className="mt-1 text-center font-mono text-xs" style={{ color: "var(--finish)" }}>
              {error}
            </p>
          )}
        </figure>

        <ResultPanel result={result} />
      </div>
    </div>
  );
}
