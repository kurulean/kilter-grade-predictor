// maps a detected photo-hold id (from data/holds.ts) to its real position
// on the model's 47x38 grid, computed by calibrating the photo against the
// database's real hold coordinates (see hold_mapping.py / scripts/calibrate_holds.py)
export interface GridCell {
  col: number;
  row: number;
}

export async function loadPhotoHoldMap(): Promise<Record<string, GridCell>> {
  const res = await fetch("/data/photo_hold_map.json");
  return res.json();
}
