"""Detect individual hold outlines from the board photo and regenerate
web/src/data/holds.ts + web/public/board.png from scratch.

Source: a square render of the board, holds on a solid background (light or
dark -- background_level() figures out which, so this doesn't need to be
told). If the source carries alpha, it's flattened onto white first (a flat
image with alpha=255 everywhere flattens to itself, so this is a no-op for
already-opaque sources).

1. flatten onto white using the source's own alpha (standard "over"
   compositing), so an opaque source and an alpha-matted one are handled
   the same way from here on
2. threshold to a binary hold/background mask (background_level() finds the
   background's gray value from the histogram mode, then anything more than
   MASK_MARGIN away from it counts as hold), connected-component label it
3. size-aware targeted watershed split: a *global* watershed (splitting
   every blob by its distance-transform peaks) badly over-segments here --
   these holds are irregular/organic enough that many single holds have two
   natural "lobes" that both register as local maxima. Instead, leave
   normal-sized blobs untouched entirely, and only split blobs whose area
   is a clear multiple of a typical single hold's area (i.e. actually-
   merged touching holds), splitting each into exactly round(area/typical)
   pieces via a locally-constrained watershed (peak_local_max with
   num_peaks capped to that estimate).
4. for each resulting region: bounding box, centroid, pixel-count area, and
   an SVG path from its simplified convex hull
5. scale from the 1254x1254 source into the app's 458x458 coordinate system,
   and save a resized copy (with our own clean alpha, not the source's) as
   the new board.png

Run scripts/calibrate_holds.py afterward to regenerate the DB calibration
against this new hold set.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_IMAGE = Path(r"C:\Users\Matthew\Downloads\Untitled (1650 x 1650 px)(2).png")
BOARD_OUT = REPO_ROOT / "web" / "public" / "board.png"
HOLDS_TS_OUT = REPO_ROOT / "web" / "src" / "data" / "holds.ts"

BOARD_SIZE = 458  # the app's fixed SVG/image coordinate system
MIN_BLOB_SIZE = 20  # px, drops stray specks
SPLIT_MIN_DISTANCE = 10  # px, minimum peak separation within a merged blob
MASK_MARGIN = 60  # gray-value distance from the background level to count as "hold"


def background_level(gray):
    """The background's gray value -- the mode of the histogram, since
    background pixels vastly outnumber hold pixels regardless of whether
    it's light-on-dark or dark-on-light.
    """
    hist, edges = np.histogram(gray, bins=256, range=(0, 256))
    return float(edges[np.argmax(hist)])


def flatten_onto_white(rgba):
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgb * alpha + 255.0 * (1 - alpha)


def simplify_hull(points, max_vertices=10):
    """Convex hull of a blob's pixels, thinned to at most max_vertices by
    dropping the hull vertices with the least angular contribution.
    """
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    while len(verts) > max_vertices:
        prev = np.roll(verts, 1, axis=0)
        nxt = np.roll(verts, -1, axis=0)
        tri_area = 0.5 * np.abs(
            (verts[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
            - (nxt[:, 0] - prev[:, 0]) * (verts[:, 1] - prev[:, 1])
        )
        drop = np.argmin(tri_area)
        verts = np.delete(verts, drop, axis=0)
    return verts


NECK_RATIO = 0.55  # a split is only trusted if the boundary between the two
# pieces is this much narrower than the smaller piece's own "radius" --
# otherwise it's almost certainly one whole hold cut through its wide
# middle, not two touching holds pinched together at a real neck


def split_blob(blob_mask, x0, y0, expected_count):
    """Split one blob (local boolean mask, offset by x0,y0 in the full
    image) into up to `expected_count` regions via a locally-constrained
    watershed -- but only where shape actually supports it.

    Area alone can't tell "one big hold" from "two touching holds": a blob
    being oversized is necessary but not sufficient evidence of a merge. So
    after watershed proposes a split, each resulting boundary is checked
    for a genuine neck -- the distance-transform value right at the seam
    between two pieces, compared to how "thick" (peak distance-transform)
    each piece is on its own. A real merge pinches down near that seam; a
    single organically-shaped hold usually doesn't, even where its own
    shape has two internal bumps. Pieces whose shared boundary isn't
    pinched get merged back together rather than kept as a false split.
    """
    dist = ndimage.distance_transform_edt(blob_mask)
    coords = peak_local_max(
        dist, min_distance=SPLIT_MIN_DISTANCE, labels=blob_mask, num_peaks=expected_count
    )
    if len(coords) < 2:
        ys, xs = np.where(blob_mask)
        return [(ys + y0, xs + x0)]
    markers = np.zeros(dist.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels = watershed(-dist, markers, mask=blob_mask)
    n_pieces = len(coords)
    peak_dist = {i + 1: dist[tuple(coords[i])] for i in range(n_pieces)}

    parent = list(range(n_pieces + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # a pixel labeled i that's 4-adjacent to a differently-labeled pixel j
    # sits right on that seam -- its own distance-transform value is the
    # neck width at that point
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(labels, shift, axis=(0, 1))
        seam = (labels > 0) & (shifted > 0) & (labels != shifted)
        if not seam.any():
            continue
        seam_labels = labels[seam]
        seam_dist = dist[seam]
        for i, j in set(zip(seam_labels.tolist(), shifted[seam].tolist())):
            if i >= j:
                continue
            neck = seam_dist[(seam_labels == i) | (seam_labels == j)].min()
            core_radius = min(peak_dist[i], peak_dist[j])
            if neck >= NECK_RATIO * core_radius:
                union(i, j)

    groups = {}
    for i in range(1, n_pieces + 1):
        groups.setdefault(find(i), []).append(i)

    regions = []
    for group in groups.values():
        group_mask = np.isin(labels, group)
        ys, xs = np.where(group_mask)
        if len(xs) == 0:
            continue
        regions.append((ys + y0, xs + x0))
    return regions


def main():
    source = Image.open(SOURCE_IMAGE).convert("RGBA")
    src_size = source.size[0]
    assert source.size[0] == source.size[1], f"expected a square source image, got {source.size}"
    rgba = np.array(source)
    flattened = flatten_onto_white(rgba)
    gray = flattened.mean(axis=2)

    bg_level = background_level(gray)
    dark_background = bg_level < 128
    mask = (gray > bg_level + MASK_MARGIN) if dark_background else (gray < bg_level - MASK_MARGIN)
    print(f"background level: {bg_level:.0f} ({'dark' if dark_background else 'light'} background)")

    # every hold has a small dark bolt-hole dot near its center, dark enough
    # to fall on the background side of the threshold -- left alone, that
    # punches a tiny hole in the middle of every blob, and the distance
    # transform then measures distance to *that* too, not just the true
    # outer edge. That fakes a "pinch" right next to every hold's own
    # center, which watershed reads as a seam between two touching holds
    # every single time -- filling interior holes first removes the
    # artifact at its source rather than trying to filter it out downstream.
    mask = ndimage.binary_fill_holes(mask)

    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.array(ndimage.sum(mask, labeled, range(1, n + 1)))
    kept = [i + 1 for i, s in enumerate(sizes) if s >= MIN_BLOB_SIZE]
    print(f"raw components: {n}, kept after noise filter: {len(kept)}")

    core = sizes[sizes >= MIN_BLOB_SIZE]
    core = core[(core > np.percentile(core, 15)) & (core < np.percentile(core, 70))]
    typical_area = np.median(core)
    print(f"typical single-hold area: {typical_area:.0f}px")

    regions = []  # list of (ys, xs) in full-image coords
    n_split_blobs = 0
    for label in kept:
        size = sizes[label - 1]
        expected = max(1, round(size / typical_area))
        if expected <= 1:
            ys, xs = np.where(labeled == label)
            regions.append((ys, xs))
            continue
        n_split_blobs += 1
        y0, y1 = np.where(labeled == label)[0].min(), np.where(labeled == label)[0].max()
        x0, x1 = np.where(labeled == label)[1].min(), np.where(labeled == label)[1].max()
        local_mask = labeled[y0:y1 + 1, x0:x1 + 1] == label
        regions.extend(split_blob(local_mask, x0, y0, expected))

    print(f"blobs split: {n_split_blobs}, total regions after splitting: {len(regions)}")

    scale = BOARD_SIZE / src_size
    holds = []
    for ys, xs in regions:
        if len(xs) < MIN_BLOB_SIZE:
            continue
        area_px = len(xs)
        cx, cy = xs.mean() * scale, ys.mean() * scale
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        width, height = (x1 - x0) * scale, (y1 - y0) * scale

        points = np.stack([xs, ys], axis=1).astype(float)
        verts = simplify_hull(points) * scale
        path = "M " + " L ".join(f"{round(vx)} {round(vy)}" for vx, vy in verts) + " Z"

        holds.append({
            "x": round(x0 * scale),
            "y": round(y0 * scale),
            "width": round(width),
            "height": round(height),
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "area": round(area_px * scale * scale),
            "path": path,
        })

    holds.sort(key=lambda h: (round(h["cy"] / 12), h["cx"]))
    for i, h in enumerate(holds, start=1):
        h["id"] = i

    print(f"final hold count: {len(holds)}")

    lines = [
        "export type Hold = {",
        "  id: number;",
        "  x: number;",
        "  y: number;",
        "  width: number;",
        "  height: number;",
        "  cx: number;",
        "  cy: number;",
        "  area: number;",
        "  path: string;",
        "};",
        "",
        f"export const BOARD_WIDTH = {BOARD_SIZE};",
        f"export const BOARD_HEIGHT = {BOARD_SIZE};",
        f"export const HOLD_COUNT = {len(holds)};",
        "",
        "export const holds: Hold[] = [",
    ]
    for h in holds:
        lines.append(
            "  { "
            f'"x": {h["x"]}, "y": {h["y"]}, "width": {h["width"]}, "height": {h["height"]}, '
            f'"cx": {h["cx"]}, "cy": {h["cy"]}, "area": {h["area"]}, "path": "{h["path"]}", "id": {h["id"]} '
            "},"
        )
    lines[-1] = lines[-1].rstrip(",")
    lines.append("];")
    lines.append("")

    HOLDS_TS_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {HOLDS_TS_OUT}")

    # soft ramp centered on the same background level used for the mask:
    # transparent right at the background color, fully opaque MASK_MARGIN
    # past it (in whichever direction "away from background" is), so the
    # final board.png's alpha edges land exactly where the mask's do
    if dark_background:
        alpha = np.clip((gray - bg_level) / MASK_MARGIN, 0.0, 1.0)
    else:
        alpha = np.clip((bg_level - gray) / MASK_MARGIN, 0.0, 1.0)
    alpha_u8 = (alpha * 255).round().astype(np.uint8)
    out_rgba = np.dstack([flattened.astype(np.uint8), alpha_u8])
    out_img = Image.fromarray(out_rgba, mode="RGBA").resize((BOARD_SIZE, BOARD_SIZE), Image.LANCZOS)
    out_img.save(BOARD_OUT)
    print(f"wrote {BOARD_OUT} ({BOARD_SIZE}x{BOARD_SIZE}, RGBA)")


if __name__ == "__main__":
    main()
