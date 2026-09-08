"""Calibrate the photo-detected hold outlines (web/src/data/holds.ts) against
the board's real physical hold positions (data/kilter_data.sqlite, via
hold_mapping.py) and write web/public/data/photo_hold_map.json:
{ "<photo_hold_id>": {"col": int, "row": int} } for photo holds matched to a
real hold inside the 47x38 grid the trained model actually uses
(hold_mapping's MIN_Y..MAX_Y range). Also writes
web/public/data/untrainable_hold_ids.json, which PhotoBoard.tsx removes from
the render entirely -- normally empty, since the source photo is expected to
already be cropped to the trainable region (no dead-zone rows above
MAX_Y=152 in frame); it exists as a safety net in case a future source image
includes that dead strip again.

The affine fit is calibrated using ONLY in-grid real holds (not all 692 in
the DB) -- fitting against the full board including the ~51 dead-zone holes
above MAX_Y=152 systematically distorts the transform when those rows
aren't actually in the photo, which shows up as exactly the kind of
top-of-image mismatches this was built to avoid.

Re-run this whenever web/src/data/holds.ts (the photo hold set) changes.
"""

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from hold_mapping import load_placement_lookup, xy_to_cell  # noqa: E402

DB_PATH = REPO_ROOT / "data" / "kilter_data.sqlite"
HOLDS_TS_PATH = REPO_ROOT / "web" / "src" / "data" / "holds.ts"
OUT_PATH = REPO_ROOT / "web" / "public" / "data" / "photo_hold_map.json"
UNTRAINABLE_OUT_PATH = REPO_ROOT / "web" / "public" / "data" / "untrainable_hold_ids.json"

GOOD_PX = 40.0  # with the in-grid-only affine fit, coverage climbs smoothly
# with this cap (509 -> 538 -> 548 -> 554 -> 559 -> 560 photo holds matched
# as the cap widens from 12 to 40px) instead of the sharp plateau-then-huge-
# jump pattern a mis-scoped fit produces -- a smooth curve like this means
# the wider net is still catching genuine nearby matches, not forcing
# distant wrong ones. Re-check this sweep (see calibrate_holds sweep in
# conversation history / re-derive it) if this stops reaching full coverage
# after a holds.ts regeneration.


def load_photo_holds(path):
    text = path.read_text(encoding="utf-8")
    entries = []
    for m in re.finditer(
        r'"cx":\s*([\d.]+),\s*"cy":\s*([\d.]+),\s*"area":\s*[\d.]+,\s*"path":\s*"[^"]*",\s*"id":\s*(\d+)',
        text,
    ):
        cx, cy, hold_id = float(m.group(1)), float(m.group(2)), int(m.group(3))
        entries.append((hold_id, cx, cy))
    if not entries:
        raise RuntimeError(f"no holds parsed from {path} -- check the regex against its format")
    return entries


def apply_affine(pts, params):
    a, b, c, d, tx, ty = params
    x = a * pts[:, 0] + b * pts[:, 1] + tx
    y = c * pts[:, 0] + d * pts[:, 1] + ty
    return np.stack([x, y], axis=1)


def fit_affine(src, dst):
    n = src.shape[0]
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    A[0::2, 0] = src[:, 0]
    A[0::2, 1] = src[:, 1]
    A[0::2, 4] = 1
    A[1::2, 2] = src[:, 0]
    A[1::2, 3] = src[:, 1]
    A[1::2, 5] = 1
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    return params


def main():
    conn = sqlite3.connect(str(DB_PATH))
    lookup = load_placement_lookup(conn, layout_id=1)
    real_ids = list(lookup.keys())
    real_pts = np.array([[lookup[i][0], lookup[i][1]] for i in real_ids], dtype=float)

    photo_entries = load_photo_holds(HOLDS_TS_PATH)
    photo_ids = [e[0] for e in photo_entries]
    photo_pts = np.array([[e[1], e[2]] for e in photo_entries], dtype=float)

    print(f"real holds: {len(real_ids)}  photo holds: {len(photo_ids)}")

    # split real holds into the ones the model's grid covers and the ones
    # above it (edge_top never exceeds MAX_Y across all 237,902 climbs, so
    # nothing has ever trained on those) -- do this BEFORE fitting the
    # affine transform, since the fit itself must only ever be told about
    # the in-grid holds (see module docstring)
    in_grid = {}  # real_idx -> (col, row)
    dead_zone = []  # real_idx
    for real_idx, real_id in enumerate(real_ids):
        x, y, _role = lookup[real_id]
        try:
            in_grid[real_idx] = xy_to_cell(x, y)
        except ValueError:
            dead_zone.append(real_idx)

    fit_pts = real_pts[list(in_grid.keys())]

    rx_min, ry_min = fit_pts.min(axis=0)
    rx_max, ry_max = fit_pts.max(axis=0)
    px_min, py_min = photo_pts.min(axis=0)
    px_max, py_max = photo_pts.max(axis=0)
    sx = (px_max - px_min) / (rx_max - rx_min)
    sy = (py_max - py_min) / (ry_max - ry_min)
    # initial guess: bbox scale with y-flip (photo y grows downward, board y grows upward)
    params = np.array([sx, 0, 0, -sy, px_min - rx_min * sx, py_max + sy * ry_min])

    tree = cKDTree(photo_pts)

    for iteration in range(15):
        mapped = apply_affine(fit_pts, params)
        dist, idx = tree.query(mapped)
        threshold = max(15, np.percentile(dist, 80))
        keep = dist < threshold
        params = fit_affine(fit_pts[keep], photo_pts[idx[keep]])

    mapped = apply_affine(fit_pts, params)
    dist, idx = tree.query(mapped)
    print(f"final median residual: {np.median(dist):.2f}px  p95: {np.percentile(dist, 95):.2f}px")

    mapped_transformed = apply_affine(real_pts, params)

    def optimal_match(real_indices, photo_indices, max_dist):
        """Bipartite assignment (Hungarian algorithm) minimizing total
        distance, with a reject option -- avoids the greedy pitfall where a
        closer real hold grabs a photo blob first, leaving a farther real
        hold with nothing nearby even though a better global pairing exists,
        *without* forcing every real hold to take something.

        A plain complete assignment (real_indices x photo_indices, nothing
        else) has to use up almost every photo hold whenever the two sets
        are close in size, however few of those pairs are actually correct
        -- one real hold with no good candidate nearby forces it (and the
        chain of reassignments that ripples from it) onto a wrong distant
        blob, which can perturb many otherwise-correct nearby matches just
        to keep the assignment complete. Padding the cost matrix with one
        dummy "no match" column per real hold, each costing exactly
        max_dist, fixes that: a real hold only takes a genuine photo hold
        when that's cheaper than rejecting, so hopeless cases opt out
        instead of dragging good matches down with them.
        """
        real_indices = list(real_indices)
        photo_indices = list(photo_indices)
        if not real_indices or not photo_indices:
            return []
        real_pos = mapped_transformed[real_indices]
        photo_pos = photo_pts[photo_indices]
        real_cost = np.linalg.norm(real_pos[:, None, :] - photo_pos[None, :, :], axis=2)
        reject_cost = np.full((len(real_indices), len(real_indices)), max_dist)
        cost = np.concatenate([real_cost, reject_cost], axis=1)
        row_idx, col_idx = linear_sum_assignment(cost)
        triples = []
        for r, c in zip(row_idx, col_idx):
            if c >= len(photo_indices):
                continue  # matched to a dummy -- rejected, no confident candidate
            d = real_cost[r, c]
            if d <= max_dist:
                triples.append((real_indices[r], photo_indices[c], d))
        return triples

    all_photo_idx = range(len(photo_ids))
    grid_matches = optimal_match(in_grid.keys(), all_photo_idx, max_dist=GOOD_PX)
    best_for_photo = {
        photo_ids[photo_idx]: {"col": int(in_grid[real_idx][0]), "row": int(in_grid[real_idx][1])}
        for real_idx, photo_idx, _d in grid_matches
    }
    print(f"mapped {len(best_for_photo)} / {len(in_grid)} in-grid real holds to distinct photo holds "
          f"(of {len(photo_ids)} available, within {GOOD_PX}px)")
    if grid_matches:
        residuals = [d for _, _, d in grid_matches]
        print(f"  match residuals: median {np.median(residuals):.2f}px, "
              f"p95 {np.percentile(residuals, 95):.2f}px, max {max(residuals):.2f}px")

    # dead_zone real holds intentionally aren't matched at all: the source
    # photo is expected to already be cropped to exclude them, so giving
    # them a shot at unclaimed photo holds would just risk misattributing
    # an in-grid hold's photo blob to a real hold that was never in frame.
    # untrainable_hold_ids.json stays empty unless a future source image
    # brings that dead strip back into frame.
    untrainable_ids = []
    print(f"dead-zone real holds: {len(dead_zone)} (not matched -- source photo is cropped to exclude them)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(best_for_photo, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_PATH}")

    UNTRAINABLE_OUT_PATH.write_text(json.dumps(untrainable_ids, indent=2), encoding="utf-8")
    print(f"wrote {UNTRAINABLE_OUT_PATH}")


if __name__ == "__main__":
    main()
