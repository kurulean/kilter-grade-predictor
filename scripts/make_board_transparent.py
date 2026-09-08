"""Regenerate web/public/board.png with real alpha transparency instead of a
solid white background, so the SVG page background shows through around the
holds instead of a white rectangle.

Reuses the same grayscale threshold as detect_holds.py to tell hold pixels
from background, but feathers a soft ramp across it (rather than a hard
cutoff) so anti-aliased hold edges don't get a jagged, cut-out look.
"""

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_IMAGE = Path(r"C:\Users\Matthew\Downloads\kilter wall.png")
BOARD_OUT = REPO_ROOT / "web" / "public" / "board.png"
BOARD_SIZE = 458

# fully opaque at/below OPAQUE_AT, fully transparent at/above TRANSPARENT_AT,
# linear ramp between -- keeps the same ~245 midpoint detect_holds.py uses
# for the hard hold/background split, just softened for a clean edge
OPAQUE_AT = 235
TRANSPARENT_AT = 253


def main():
    color = Image.open(SOURCE_IMAGE).convert("RGB")
    gray = np.array(color.convert("L"), dtype=np.float32)

    alpha = np.clip((TRANSPARENT_AT - gray) / (TRANSPARENT_AT - OPAQUE_AT), 0.0, 1.0)
    alpha_u8 = (alpha * 255).round().astype(np.uint8)

    rgba = np.dstack([np.array(color), alpha_u8])
    out = Image.fromarray(rgba, mode="RGBA").resize((BOARD_SIZE, BOARD_SIZE), Image.LANCZOS)
    out.save(BOARD_OUT)
    print(f"wrote {BOARD_OUT} ({BOARD_SIZE}x{BOARD_SIZE}, RGBA)")


if __name__ == "__main__":
    main()
