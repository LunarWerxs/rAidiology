"""Export de-identified JPEG slice stacks (pixels only) for the web viewer, plus an
"AI outlines" stack of the TotalSpineSeg labels over each sagittal T2. Series are chosen per
region from nifti_dir/series.json (written by to_nifti.py), so any spine disc works.
Usage: python build_stacks.py <nifti_dir> <tss_step2_dir> <site_dir>"""
import json
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402
paths.use("spine_mri")
from labels import NAMES, kind  # noqa: E402
from series import REGION_NAME, pick  # noqa: E402

PLAIN = {"C": "Neck", "T": "Upper back", "L": "Lower back", "X": "Spine"}
WANT = [("sagittal", "T2", "side view (T2)"), ("sagittal", "T1", "side view (T1)"),
        ("sagittal", "STIR", "side view (STIR)"), ("axial", "T2", "cross-sections (T2)")]
PALETTE = {"vertebra": (217, 207, 185), "disc": (159, 176, 191), "cord": (214, 178, 94),
           "canal": (156, 196, 184), "sacrum": (217, 207, 185)}


def sharpness(vol):
    g = np.hypot(ndimage.sobel(vol, 0), ndimage.sobel(vol, 1))
    return float(g.mean() / (vol.mean() + 1e-6))


def slices(vol):
    lo, hi = np.percentile(vol, [0.5, 99.7])
    for k in range(vol.shape[2]):
        a = vol[:, :, k].T
        yield (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def main(nii, tss, site):
    series = json.load(open(os.path.join(nii, "series.json")))
    out = os.path.join(site, "data", "stacks")
    index = []
    for region in sorted({s["region"] for s in series}):
        for plane, seq, what in WANT:
            cands = [s for s in series if s["region"] == region and s["plane"] == plane and s["sequence"] == seq
                     and os.path.exists(os.path.join(nii, f"{s['name']}.nii.gz"))]
            if not cands:
                continue
            vols = {s["name"]: np.asarray(nib.load(os.path.join(nii, f"{s['name']}.nii.gz")).dataobj).astype(np.float32) for s in cands}
            src = max(cands, key=lambda s: (s["count"], sharpness(vols[s["name"]])))["name"]  # repeats: keep the sharper
            n = nib.load(os.path.join(nii, f"{src}.nii.gz"))
            vol = vols[src]
            z = n.header.get_zooms()
            sid = f"{region}_{'SAG' if plane == 'sagittal' else 'AX'}_{seq}"
            d = os.path.join(out, sid)
            os.makedirs(d, exist_ok=True)
            for k, g in enumerate(slices(vol)):
                Image.fromarray(g).save(os.path.join(d, f"{k:02d}.jpg"), quality=88)
            orient = ({"left": "Front", "right": "Back", "top": "Head", "bottom": "Feet"} if plane == "sagittal"
                      else {"left": "Right", "right": "Left", "top": "Front", "bottom": "Back"})
            entry = {"id": sid, "label": f"{PLAIN[region]}, {what}", "region": REGION_NAME[region], "plane": plane,
                     "sequence": seq, "count": int(vol.shape[2]), "width": int(vol.shape[0]), "height": int(vol.shape[1]),
                     "pixel_mm": [round(float(z[1]), 3), round(float(z[0]), 3)],
                     "path": f"data/stacks/{sid}/{{i}}.jpg", "default_slice": int(vol.shape[2] // 2),
                     "orientation": orient, "source_series": src}
            index.append(entry)
            lab_p = os.path.join(tss, f"{src}.nii.gz")
            if plane == "sagittal" and seq == "T2" and os.path.exists(lab_p):
                lab = np.asarray(nib.load(lab_p).dataobj).astype(np.int32)
                sd = os.path.join(out, sid + "_AI")
                os.makedirs(sd, exist_ok=True)
                for k, g in enumerate(slices(vol)):
                    L = lab[:, :, k].T
                    rgb = np.stack([g] * 3, -1).astype(np.float32)
                    for v in np.unique(L):
                        if v == 0 or int(v) not in NAMES:
                            continue
                        m = L == v
                        kd = kind(NAMES[int(v)])
                        col = np.array(PALETTE[kd], np.float32)
                        alpha = 0.18 if kd in ("vertebra", "sacrum", "canal") else 0.38
                        rgb[m] = rgb[m] * (1 - alpha) + col * alpha
                        rgb[m & ~ndimage.binary_erosion(m)] = col
                    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(os.path.join(sd, f"{k:02d}.jpg"), quality=88)
                index.append(dict(entry, id=sid + "_AI", label=entry["label"] + " with AI outlines", path=f"data/stacks/{sid}_AI/{{i}}.jpg"))
            print(sid, "<-", src, vol.shape)
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
