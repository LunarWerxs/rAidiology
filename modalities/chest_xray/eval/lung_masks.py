"""CheXmask's lung masks for NIH's pictures, as 24 by 24 soft masks, built once on the processor for
reader.py's `lung_aux` setting (docs/efficiency, trial 11).

Each picture's right lung (CheXmask's first 44 landmarks) and left lung (the next 50) are filled as polygons
on a 96 by 96 grid and averaged down to 24 by 24, so an edge cell is a fraction. Stored as uint8 0..255,
shape (pictures, 2, 24, 24), in a .npy that training memory-maps; a second file holds the picture names
and CheXmask's quality (Dice RCA mean) in the same row order. A picture whose landmarks did not parse has
all-zero masks and a NaN quality. CheXmask's CSV is never read inside a training loop.

Build (about a minute, no card):
    python modalities/chest_xray/eval/lung_masks.py build
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402

SIDE = 24                   # the mask's side, the stride-16 map of a 384 px picture
SUPER = 4                   # polygons are filled at SIDE * SUPER and averaged down
PATH = os.path.join(paths.WORK, "eval", "chest", "lung_masks_24.npy")
INDEX = os.path.join(paths.WORK, "eval", "chest", "lung_masks_24_index.npz")
N_RIGHT, N_LEFT = 44, 50    # CheXmask's landmark order: right lung, left lung, heart (tools/chexmask_ctr.py)


def build_command():
    return "python modalities/chest_xray/eval/lung_masks.py build"


def fill(points, side=SIDE, sup=SUPER):
    """(side, side) float32 in [0, 1]: the share of each cell inside the polygon `points` (n x 2, fractions of the picture)."""
    from PIL import Image, ImageDraw
    n = side * sup
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).polygon([(float(x) * n, float(y) * n) for x, y in points], fill=1)
    return np.asarray(img, np.float32).reshape(side, sup, side, sup).mean(axis=(1, 3))


def masks_of(landmarks):
    """(2, SIDE, SIDE) uint8 0..255 (right, left lung) from one picture's 240 landmark numbers (fractions); None when not finite."""
    pts = np.asarray(landmarks, np.float64).reshape(-1, 2)
    if not np.isfinite(pts).all():
        return None
    both = np.stack([fill(pts[:N_RIGHT]), fill(pts[N_RIGHT:N_RIGHT + N_LEFT])])
    return np.rint(both * 255).astype(np.uint8)


def build():
    tools = os.path.join(ROOT, "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    import chexmask_ctr as cc
    image, lm, dice = cc.load_landmarks()
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    out = np.lib.format.open_memmap(PATH + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(len(image), 2, SIDE, SIDE))
    quality = dice.astype(np.float32).copy()
    bad = 0
    for i in range(len(image)):
        m = masks_of(lm[i])
        if m is None:
            out[i] = 0
            quality[i] = np.nan
            bad += 1
        else:
            out[i] = m
    out.flush()
    del out
    np.savez(INDEX + ".tmp.npz", image=image, dice=quality)
    os.replace(PATH + ".tmp.npy", PATH)
    os.replace(INDEX + ".tmp.npz", INDEX)
    print("lung masks: %d pictures (%d without parsed landmarks) -> %s" % (len(image), bad, PATH), flush=True)


def load():
    """(masks memmap (N, 2, SIDE, SIDE) uint8, picture names, quality) from the build."""
    z = np.load(INDEX, allow_pickle=False)
    return np.load(PATH, mmap_mode="r"), z["image"].astype(str), z["dice"]


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        raise SystemExit("usage: python lung_masks.py build")
    build()
