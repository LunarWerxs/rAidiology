"""Per-disc 2.5D crops for the disc grader (Phase 2), the same function for SPIDER and a patient disc.

A disc's crop is sampled on a fixed physical grid: 5 sagittal planes 3.3 mm apart around the disc's
centre, each H x W pixels of 0.6 mm, turned so the disc's long axis (front-to-back) is horizontal.
The centre and angle come from the disc's TotalSpineSeg label on T2 (or SPIDER's mask, for the
`spider_masks` comparison). T1 is sampled at the same physical points through its own affine
(when there is no T1 the channel is zeros and a flag says so). Each series is scaled by its own
1st and 99th percentiles, so 0 ~ dark and 1 ~ bright on every scanner.
Output: float16 array (2 sequences, 5 planes, H, W).
Usage: python crops.py <dev|test> [work_dir]  -> <work>/<split>/crops.npz"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from labels import NAMES  # noqa: E402
from prepare import WORK, disc_id  # noqa: E402
from spider import guard_test  # noqa: E402

H, W, PLANES = 80, 112, 5
PIX_MM, PLANE_MM = 0.6, 3.3
LEVEL_ID = {v[5:].replace("_", "-"): k for k, v in NAMES.items() if v.startswith("disc_")}


class Series:
    def __init__(self, path):
        img = nib.load(path)
        self.a = np.asarray(img.dataobj).astype(np.float32)
        self.aff = img.affine
        self.inv = np.linalg.inv(img.affine)
        body = self.a[self.a > 0.1 * np.percentile(self.a, 99)]
        self.lo, self.hi = np.percentile(body if body.size > 1000 else self.a, [1, 99])

    def sample(self, pts):
        """Trilinear samples at world points (N, 3), scaled to the series' 1st..99th percentiles."""
        v = (self.inv @ np.c_[pts, np.ones(len(pts))].T)[:3]
        s = ndimage.map_coordinates(self.a, v, order=1, mode="constant", cval=0.0)
        return (s - self.lo) / max(self.hi - self.lo, 1e-6)


def disc_frame(lab, aff, label):
    """World centre, unit axes (ap, si, lr) of a disc label in a PIR volume, the ap axis along the
    disc's long axis in the sagittal plane; None if the label is absent."""
    vox = np.argwhere(lab == label)
    if len(vox) < 20:
        return None
    world = (aff @ np.c_[vox, np.ones(len(vox))].T)[:3].T
    c = world.mean(0)
    lr = aff[:3, 2] / np.linalg.norm(aff[:3, 2])           # PIR axis 2 runs to the patient's right
    ap0 = aff[:3, 0] / np.linalg.norm(aff[:3, 0])          # axis 0 runs posterior
    si0 = aff[:3, 1] / np.linalg.norm(aff[:3, 1])          # axis 1 runs inferior
    q = world - c
    uv = np.c_[q @ ap0, q @ si0]
    _, _, vt = np.linalg.svd(uv - uv.mean(0), full_matrices=False)
    d = vt[0] if vt[0][0] >= 0 else -vt[0]                 # long axis, pointing backward
    ap = d[0] * ap0 + d[1] * si0
    ap /= np.linalg.norm(ap)
    si = np.cross(lr, ap)
    if si @ si0 < 0:
        si = -si
    return c, ap, si, lr


def grid(c, ap, si, lr):
    u = (np.arange(W) - (W - 1) / 2) * PIX_MM
    v = (np.arange(H) - (H - 1) / 2) * PIX_MM
    w = (np.arange(PLANES) - (PLANES - 1) / 2) * PLANE_MM
    ww, vv, uu = np.meshgrid(w, v, u, indexing="ij")
    return c + uu[..., None] * ap + vv[..., None] * si + ww[..., None] * lr   # (PLANES, H, W, 3)


def crop(t2, t1, lab, label):
    """(2, PLANES, H, W) float16 crop and has_t1, or None when the label is missing."""
    f = disc_frame(lab, t2.aff, label)
    if f is None:
        return None, False
    pts = grid(*f).reshape(-1, 3)
    x = np.zeros((2, PLANES, H, W), np.float32)
    x[0] = t2.sample(pts).reshape(PLANES, H, W)
    if t1 is not None:
        x[1] = t1.sample(pts).reshape(PLANES, H, W)
    return np.clip(x, -0.5, 2.0).astype(np.float16), t1 is not None


def main(split, work=WORK):
    guard_test(split)
    base = os.path.join(work, split)
    m = pd.read_csv(os.path.join(base, "matches.csv"), keep_default_na=False)
    X, rows = [], []
    for p in sorted(m[(m.in_scope == 1)].patient.unique()):
        mp = m[(m.patient == p) & (m.in_scope == 1)]
        t2 = Series(os.path.join(base, "t2", f"{p}.nii.gz"))
        t1p = os.path.join(base, "t1", f"{p}.nii.gz")
        t1 = Series(t1p) if os.path.exists(t1p) else None
        labs = {"tss": np.asarray(nib.load(os.path.join(base, "tss", "step2_output", f"{p}.nii.gz")).dataobj).astype(np.int32),
                "spider_masks": np.asarray(nib.load(os.path.join(base, "spiderlab", f"{p}.nii.gz")).dataobj).astype(np.int32)}
        for _, r in mp.iterrows():
            for loc in ("tss", "spider_masks"):
                if loc == "tss" and r.matched != 1:
                    continue
                label = int(r.tss_label) if loc == "tss" else disc_id(int(r.ivd))
                x, has_t1 = crop(t2, t1, labs[loc], label)
                if x is None:
                    continue
                X.append(x)
                rows.append({"patient": int(p), "ivd": int(r.ivd), "localisation": loc, "fold": r.fold,
                             "vendor": r.vendor, "has_t1": int(has_t1)})
    idx = pd.DataFrame(rows)
    np.savez_compressed(os.path.join(base, "crops.npz"), x=np.stack(X))   # the index is crops_index.csv
    idx.to_csv(os.path.join(base, "crops_index.csv"), index=False)
    print(split, "crops", len(X), idx.groupby("localisation").size().to_dict(), "no T1:", int((idx.has_t1 == 0).sum()))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or [WORK]))
