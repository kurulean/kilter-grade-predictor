"""Build data/gate_dataset.npz: a binary "is this climb physically
plausible" dataset for training a lightweight validity gate that runs
before the grade model.

Why this exists: the grade model has zero training examples below 3 holds
(see the investigation this came out of), and nothing sparse is ever rated
below V5 in the real data -- so a 2-hold, opposite-corner "climb" is a true
blind spot, not just a rare case. Softmax always returns a confident-looking
distribution regardless, so the grade model alone can't signal "I don't
know". This script generates synthetic implausible climbs to teach a
separate small classifier to recognize that blind spot directly, rather
than papering over it with a hardcoded serve.py rule.

Positives (label=1, "valid"): real climbs, reused as-is from
data/kilter_images.npz, one column per role. This is a diverse, realistic
population of the kind of climb the grade model was actually trained on.

Negatives (label=0, "invalid"): synthetic, generated two ways, calibrated
against real hold-spacing statistics rather than arbitrary numbers --
- too_sparse: 0-2 total holds. The real training data's floor is 3 holds
  (11 climbs at exactly 3); nothing below that exists anywhere in it.
- impossible_gap: a realistic hold COUNT (3-15, matching the real
  distribution) but with at least one consecutive-by-height gap far beyond
  anything observed in real climbs. Measured directly from a 20k-climb
  sample of the real training set: the largest single consecutive gap ever
  seen is ~34.2 grid cells (p99.9 ~33.0). This uses a 40-cell floor for the
  forced gap, a clear margin above that observed max, so there's no overlap
  between "legitimately hard reach" and "synthetic impossible".

Both negative categories only ever place holds at real physical hold
positions (from the DB, same as build_dataset.py) -- never arbitrary grid
cells -- so the gate learns about spacing/count, not about "occupies a
position no real hold exists at" (a trivial, uninteresting shortcut that
wouldn't transfer to how real climbs are encoded).
"""

import sqlite3

import numpy as np

import hold_mapping as hm
from hold_mapping import load_placement_lookup

DB = "data/kilter_data.sqlite"
LAYOUT_ID = 1
OUT_PATH = "data/gate_dataset.npz"
SOURCE_NPZ = "data/kilter_images.npz"

ROLE_TO_CHANNEL = {12: 0, 13: 1, 14: 2, 15: 3}  # start, middle, finish, foot
START, MIDDLE, FINISH, FOOT = 12, 13, 14, 15
NUM_CHANNELS = 4
MAX_ANGLE = 70.0
ANGLE_CHOICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# calibrated against real training data (see module docstring) -- gaps at
# or beyond this are never seen in any real climb (observed max ~34.2)
IMPOSSIBLE_GAP_CELLS = 40.0

RNG = np.random.default_rng(0)


def build_image(cells):
    img = np.zeros((NUM_CHANNELS, hm.GRID_ROWS, hm.GRID_COLS), dtype=np.uint8)
    for col, row, role in cells:
        channel = ROLE_TO_CHANNEL.get(role)
        if channel is not None:
            img[channel, row, col] = 1
    return img


def load_real_positions():
    """(col, row) for every physical hold inside the model's grid -- the
    only positions synthetic negatives are ever allowed to use.
    """
    conn = sqlite3.connect(DB)
    lookup = load_placement_lookup(conn, layout_id=LAYOUT_ID)
    conn.close()
    positions = []
    for x, y, _role in lookup.values():
        try:
            positions.append(hm.xy_to_cell(x, y))
        except ValueError:
            continue  # outside the kept grid range (the dead zone above y=152)
    return positions


def gen_too_sparse(positions):
    """0-2 holds total -- below the real data's floor of 3."""
    n = RNG.choice([0, 1, 2], p=[0.15, 0.35, 0.5])
    if n == 0:
        return []
    chosen = [positions[i] for i in RNG.choice(len(positions), size=n, replace=False)]
    if n == 1:
        col, row = chosen[0]
        return [(col, row, START)]
    # 2 holds: lower one is start, higher one is finish (matches how a real
    # climb's roles work, so the gate learns about spacing, not "which
    # channels are lit")
    chosen.sort(key=lambda cr: cr[1])
    (c0, r0), (c1, r1) = chosen
    return [(c0, r0, START), (c1, r1, FINISH)]


def gen_impossible_gap(positions, by_row):
    """A plausible hold COUNT (3-15) but with one gap far beyond anything
    real, by anchoring two clusters far apart and only adding extra holds
    near one anchor or the other -- never in the open gap between them,
    which would legitimately bridge it and defeat the point.
    """
    rows_sorted = sorted(by_row.keys())
    # low anchor from the bottom fifth of rows in use, high anchor from the
    # top fifth, so the forced span is large before even checking distance
    lo_band = rows_sorted[: max(1, len(rows_sorted) // 5)]
    hi_band = rows_sorted[-max(1, len(rows_sorted) // 5) :]

    for _ in range(50):  # a handful of retries in case a draw falls short
        lo_row = RNG.choice(lo_band)
        hi_row = RNG.choice(hi_band)
        if hi_row - lo_row < IMPOSSIBLE_GAP_CELLS - 5:
            continue
        lo_col = RNG.choice(by_row[lo_row])
        hi_col = RNG.choice(by_row[hi_row])
        gap = np.hypot(hi_row - lo_row, hi_col - lo_col)
        if gap < IMPOSSIBLE_GAP_CELLS:
            continue

        cells = [(lo_col, lo_row, START), (hi_col, hi_row, FINISH)]

        # extra holds only within 4 rows of either anchor, so they thicken
        # the start/finish clusters without ever bridging the gap
        n_extra = RNG.integers(1, 12)
        near_rows = [r for r in rows_sorted if abs(r - lo_row) <= 4 or abs(r - hi_row) <= 4]
        candidates = [(c, r) for r in near_rows for c in by_row[r]]
        if candidates:
            n_extra = min(n_extra, len(candidates))
            idx = RNG.choice(len(candidates), size=n_extra, replace=False)
            for i in idx:
                col, row = candidates[i]
                if (col, row) in {(lo_col, lo_row), (hi_col, hi_row)}:
                    continue
                role = MIDDLE if RNG.random() < 0.6 else FOOT
                cells.append((col, row, role))

        # confirm the forced gap really does survive as the max gap (it
        # should, by construction, but this is cheap insurance against a
        # coincidental near-bridge from the extra holds)
        pts = sorted([(r, c) for c, r, _role in cells])
        gaps = [np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
        if max(gaps) >= IMPOSSIBLE_GAP_CELLS:
            return cells
    return None  # gave up this attempt; caller just tries again


def generate_negatives(positions, n):
    by_row = {}
    for col, row in positions:
        by_row.setdefault(row, []).append(col)

    images, angles = [], []
    while len(images) < n:
        if RNG.random() < 0.4:
            cells = gen_too_sparse(positions)
        else:
            cells = gen_impossible_gap(positions, by_row)
            if cells is None:
                continue
        images.append(build_image(cells))
        angles.append(RNG.choice(ANGLE_CHOICES) / MAX_ANGLE)
    return np.stack(images), np.array(angles, dtype=np.float32)


def main():
    positions = load_real_positions()
    print(f"real in-grid hold positions available: {len(positions)}")

    source = np.load(SOURCE_NPZ)
    out_images, out_angles, out_labels, out_split = [], [], [], []

    for split in ["train", "val", "test"]:
        mask = source["split"] == split
        pos_images = source["images"][mask]
        pos_angles = source["angles"][mask]
        n_pos = len(pos_images)

        neg_images, neg_angles = generate_negatives(positions, n_pos)

        out_images.append(pos_images)
        out_angles.append(pos_angles)
        out_labels.append(np.ones(n_pos, dtype=np.int64))
        out_split.append(np.full(n_pos, split))

        out_images.append(neg_images)
        out_angles.append(neg_angles)
        out_labels.append(np.zeros(len(neg_images), dtype=np.int64))
        out_split.append(np.full(len(neg_images), split))

        print(f"{split}: {n_pos} valid + {len(neg_images)} synthetic invalid")

    np.savez_compressed(
        OUT_PATH,
        images=np.concatenate(out_images).astype(np.uint8),
        angles=np.concatenate(out_angles).astype(np.float32),
        labels=np.concatenate(out_labels).astype(np.int64),
        split=np.concatenate(out_split),
    )
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
