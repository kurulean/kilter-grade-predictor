import type { Hold, HoldType } from "./climb-types";

export interface GradeEstimate {
  grade: number; // 0-13, styled as a V-grade
  label: string; // "V4"
  reason: string; // one-line explanation of what drove the number
}

// NOTE: this is a hand-tuned heuristic, not a trained model. The real
// Kilter CNN elsewhere in this project was trained on the actual Kilter
// board's own hold coordinates -- it has no meaningful output for holds
// detected on this photo, so reusing it here would just dress up a
// meaningless number as a real prediction.
//
// This instead scores a climb using the same ideas as the project's own
// classical-feature baseline: how far apart consecutive holds are and
// how much vertical ground the route covers, offset by how many holds
// are available to use along the way.
export function estimateGrade(
  selected: Record<number, HoldType>,
  holdById: Map<number, Hold>
): GradeEstimate | null {
  const chosen = Object.keys(selected)
    .map((id) => holdById.get(Number(id)))
    .filter((h): h is Hold => h !== undefined);

  if (chosen.length < 2) return null;

  // approximate climbing order: bottom of the photo (larger y) to top
  const ordered = [...chosen].sort((a, b) => b.cy - a.cy);

  let maxGap = 0;
  let totalGap = 0;
  for (let i = 1; i < ordered.length; i++) {
    const dx = ordered[i].cx - ordered[i - 1].cx;
    const dy = ordered[i].cy - ordered[i - 1].cy;
    const gap = Math.hypot(dx, dy);
    maxGap = Math.max(maxGap, gap);
    totalGap += gap;
  }
  const avgGap = totalGap / (ordered.length - 1);
  const verticalSpan = ordered[0].cy - ordered[ordered.length - 1].cy;

  const raw = maxGap / 35 + avgGap / 55 + verticalSpan / 260 - chosen.length / 10;
  const grade = Math.max(0, Math.min(13, Math.round(raw)));

  return {
    grade,
    label: `V${grade}`,
    reason: `${chosen.length} holds · longest reach ${Math.round(maxGap)}px · ${Math.round(
      verticalSpan
    )}px of height`,
  };
}
