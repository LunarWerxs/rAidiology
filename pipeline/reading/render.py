"""Render every MR series on a patient disc to PNGs for the AI readers: per-slice images, montages,
and axial-slice localizer lines drawn on each region's mid-sagittal T2. Only MR image series are
read (series.scan), so scanned paperwork and key-object notes never reach a reader.
Each slice is written in three windows from the 16-bit data (windows.py): <NN>.png (standard),
<NN>_fluid.png (fluid and disc) and <NN>_marrow.png (bone and marrow).
Usage: python render.py <cd_folder> <reader_dir>   (writes <reader_dir>/png)"""
import json
import os
import sys

import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from series import pick, scan  # noqa: E402
from windows import render_all, to8  # noqa: E402


def font(sz):
    for p in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def write_series(out, key, vol):
    """Write one series' slices in every window plus its montage under <out>; returns the windows."""
    sdir = os.path.join(out, key)
    os.makedirs(sdir, exist_ok=True)
    imgs, used = render_all(vol)
    tiles = []
    for i in range(len(vol)):
        im = Image.fromarray(imgs["standard"][i])
        im.save(os.path.join(sdir, f"{i:02d}.png"))
        for n in ("fluid", "marrow"):
            Image.fromarray(imgs[n][i]).save(os.path.join(sdir, f"{i:02d}_{n}.png"))
        t = im.resize((256, 256), Image.LANCZOS).convert("RGB")
        ImageDraw.Draw(t).text((4, 2), f"{i}", fill=(255, 255, 0), font=font(18))
        tiles.append(t)
    cols = 6 if len(tiles) > 16 else 5 if len(tiles) > 9 else 4
    rows = (len(tiles) + cols - 1) // cols
    mont = Image.new("RGB", (cols * 256, rows * 256))
    for i, t in enumerate(tiles):
        mont.paste(t, ((i % cols) * 256, (i // cols) * 256))
    mont.save(os.path.join(out, f"{key}__montage.png"))
    return used


def main(cd, reader_dir):
    out = os.path.join(reader_dir, "png")
    os.makedirs(out, exist_ok=True)
    found = scan(cd)
    loaded = {s["name"]: [pydicom.dcmread(p) for p in s["files"]] for s in found}
    meta = {}
    for key, ds in loaded.items():
        vol = np.stack([d.pixel_array.astype(np.float32) for d in ds])
        used = write_series(out, key, vol)
        meta[key] = {
            "n": len(ds),
            "pixel_spacing": [float(x) for x in ds[0].PixelSpacing],
            "thickness": float(ds[0].SliceThickness),
            "orientation": [float(x) for x in ds[0].ImageOrientationPatient],
            "positions": [[float(x) for x in d.ImagePositionPatient] for d in ds],
            "window": used["standard"],
            "windows": used,
        }
        print(key, len(ds), vol.shape)
    json.dump(meta, open(os.path.join(out, "series_meta.json"), "w"), indent=1)

    # Axial localizer lines on the mid-sagittal T2 of each region.
    for region in sorted({s["region"] for s in found}):
        sag_s = pick(found, region, "sagittal", "T2")
        axials = [s["name"] for s in found if s["region"] == region and s["plane"] == "axial"]
        if not sag_s or not axials:
            continue
        sag = loaded[sag_s["name"]]
        mid = sag[len(sag) // 2]
        a = mid.pixel_array.astype(np.float32)
        img = Image.fromarray(to8(a, *np.percentile(a, [0.5, 99.7]))).convert("RGB")
        dr = ImageDraw.Draw(img)
        o = np.array(mid.ImageOrientationPatient, float)
        row_dir, col_dir = o[:3], o[3:]
        origin = np.array(mid.ImagePositionPatient, float)
        ps = [float(x) for x in mid.PixelSpacing]
        colors = [(255, 80, 80), (80, 200, 255), (120, 255, 120)]
        for ci, ax_key in enumerate(axials):
            for i, d in enumerate(loaded[ax_key]):
                ao = np.array(d.ImageOrientationPatient, float)
                ap = np.array(d.ImagePositionPatient, float)
                aps = [float(x) for x in d.PixelSpacing]
                w = d.Columns * aps[1]
                pts = []
                # center line of the axial slab across its column extent, midline row
                for c in (0.0, w):
                    p = ap + ao[:3] * c + ao[3:] * (d.Rows * aps[0] / 2)
                    v = p - origin
                    pts.append((float(np.dot(v, row_dir) / ps[1]), float(np.dot(v, col_dir) / ps[0])))
                # project the axial row direction instead if the column direction is ~normal to the sagittal
                if abs(pts[0][0] - pts[1][0]) < 5 and abs(pts[0][1] - pts[1][1]) < 5:
                    pts = []
                    for r in (0.0, d.Rows * aps[0]):
                        p = ap + ao[:3] * (w / 2) + ao[3:] * r
                        v = p - origin
                        pts.append((float(np.dot(v, row_dir) / ps[1]), float(np.dot(v, col_dir) / ps[0])))
                dr.line(pts, fill=colors[ci % 3], width=1)
                dr.text((pts[1][0] + 2, pts[1][1] - 6), f"{ax_key.split('_')[1]}:{i}", fill=colors[ci % 3], font=font(11))
        img.save(os.path.join(out, f"{region}_axial_localizer.png"))
        print("localizer", region, sag_s["name"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
