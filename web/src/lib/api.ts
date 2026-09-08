// talks to the real PyTorch model, served by serve.py (run
// `python serve.py` or `uvicorn serve:app --port 8000` from the project root)
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function predictGrade(
  cells: [number, number, number][],
  angle: number
): Promise<number[]> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cells, angle }),
  });
  if (!res.ok) {
    throw new Error(`prediction request failed: ${res.status}`);
  }
  const data = await res.json();
  return data.probs as number[];
}
