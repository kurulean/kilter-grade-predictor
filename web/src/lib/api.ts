// talks to the real PyTorch model, served by serve.py (run
// `python serve.py` or `uvicorn serve:app --port 8000` from the project root)
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface PredictApiResult {
  probs: number[];
  // a lightweight gate model runs ahead of the grade model server-side and
  // rejects climbs it has no real basis to judge (e.g. too few holds, or a
  // gap far beyond anything in the training data) -- see
  // generate_gate_dataset.py for why. probs is still returned even when
  // valid is false so the UI *could* show it, but should treat it as
  // unsupported rather than a real prediction.
  valid: boolean;
  validConfidence: number;
}

export async function predictGrade(cells: [number, number, number][], angle: number): Promise<PredictApiResult> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cells, angle }),
  });
  if (!res.ok) {
    throw new Error(`prediction request failed: ${res.status}`);
  }
  const data = await res.json();
  return {
    probs: data.probs as number[],
    valid: data.valid as boolean,
    validConfidence: data.valid_confidence as number,
  };
}
