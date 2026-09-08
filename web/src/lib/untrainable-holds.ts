// photo hold ids that confidently match a real hold ABOVE the model's grid
// (real y > MAX_Y=152 in hold_mapping.py) -- i.e. holds no climb in the
// training data (train+val+test, 237,902 climbs) has ever used, since the
// real board's edge_top never exceeds 152. computed by scripts/calibrate_holds.py;
// re-run that whenever data/holds.ts changes.
export async function loadUntrainableHoldIds(): Promise<Set<number>> {
  const res = await fetch("/data/untrainable_hold_ids.json");
  const ids: number[] = await res.json();
  return new Set(ids);
}
