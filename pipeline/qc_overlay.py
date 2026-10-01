"""Draw TotalSpineSeg label contours + names on sagittal slices for visual QC.
Usage: python qc_overlay.py <image.nii.gz> <labels.nii.gz> <out.png> <slice> [<slice> ...]"""
import sys

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from labels import NAMES

img_p, lab_p, out = sys.argv[1:4]
slices = [int(s) for s in sys.argv[4:]]
img = np.asarray(nib.load(img_p).dataobj).astype(np.float32)
lab = np.asarray(nib.load(lab_p).dataobj).astype(np.int32)
lo, hi = np.percentile(img, [0.5, 99.7])
font = _font(11)
rng = np.random.default_rng(3)
colors = {k: tuple(int(c) for c in rng.integers(80, 255, 3)) for k in NAMES}
colors[1], colors[2] = (255, 215, 0), (0, 255, 255)
tiles = []

def _font(size):
    """Arial where it is installed (Windows), else Pillow's own font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

for s in slices:
    # nibabel array is (i=col, j=row, k=slice) for these sagittal files; show as row x col
    a = img[:, :, s].T
    L = lab[:, :, s].T
    g = (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1)
    for k in np.unique(L):
        if k == 0:
            continue
        m = L == k
        edge = m & ~ndimage.binary_erosion(m)
        rgb[edge] = colors.get(int(k), (255, 0, 255))
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for k in np.unique(L):
        if k in (0, 1, 2):
            continue
        yy, xx = np.nonzero(L == k)
        d.text((int(xx.max()) + 3 if NAMES.get(int(k), "").startswith("disc") else int(xx.min()) - 28, int(yy.mean()) - 6),
               NAMES.get(int(k), str(k)).replace("disc_", ""), fill=colors.get(int(k)), font=font)
    d.text((4, 4), f"slice {s}", fill=(255, 255, 0), font=font)
    tiles.append(im)
W = sum(t.width for t in tiles)
canvas = Image.new("RGB", (W, tiles[0].height))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0))
    x += t.width
canvas.save(out)
print(out, canvas.size)
