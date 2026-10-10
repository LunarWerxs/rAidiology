"""Quantitative per-level measurements from the TotalSpineSeg labels on the sagittal T2.
Measured on the mid-sagittal slice (the one with the most canal voxels), so these are
approximations: +-1 pixel is +-0.4 to 0.55 mm.
Usage: python metrics.py <image.nii.gz> <labels.nii.gz> <out.json> [mid_slice]
measure() is the same step as a function (the SPIDER evaluation calls it)."""
import json
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

from labels import NAMES, ORDER


def _canal_cos_by_row(canal, col_mm, row_mm):
    # canal centerline angle for obliquity correction
    cr = [r for r in np.arange(canal.shape[0]) if canal[r].any()]
    cx = np.array([np.nonzero(canal[r])[0].mean() for r in cr])
    slope = np.gradient(ndimage.uniform_filter1d(cx, 15)) if len(cx) > 3 else np.zeros(len(cr))
    return {r: 1 / np.sqrt(1 + (s * col_mm / row_mm) ** 2) for r, s in zip(cr, slope)}


def _width(mask, r, cos_by_row, col_mm):
    if r not in cos_by_row or not mask[r].any():
        return None
    return float(mask[r].sum() * col_mm * cos_by_row[r])


def _band(mask, r0, r1, fn, cos_by_row, col_mm):
    vals = [_width(mask, r, cos_by_row, col_mm) for r in range(r0, r1 + 1)]
    vals = [v for v in vals if v is not None]
    return float(fn(vals)) if vals else None


def _csf_median(img, lab, slabs):
    csf_vals = []
    for k in slabs:
        m = ndimage.binary_erosion((lab[:, :, k].T == 2), iterations=1)
        csf_vals.append(img[:, :, k].T[m])
    return float(np.median(np.concatenate(csf_vals))) if csf_vals else None


def _disc_signal(img, lab, slabs, v):
    disc_sig = []
    for k in slabs:
        mk = ndimage.binary_erosion(lab[:, :, k].T == v, iterations=1)
        if mk.sum():
            disc_sig.append(img[:, :, k].T[mk])
    return float(np.median(np.concatenate(disc_sig))) if disc_sig else None


def _vert_rows(L, present):
    # body center: centroid of the anterior half of the vertebra mask
    vert_rows = {}
    for name, v in present.items():
        if not name.startswith("disc_") and name not in ("cord", "canal") and np.any(L == v):
            yy, xx = np.nonzero(L == v)
            ant = xx <= np.median(xx)
            vert_rows[name] = int(np.round(yy[ant].mean()))
    return vert_rows


def _disc_record(name, v, L, img, lab, slabs, geo, csf_med, vert_rows, canal, cord, cos_by_row):
    col_mm, row_mm = geo
    up, lo = name.split("_")[1], name.split("_")[2]
    m = L == v
    if m.sum() < 10:
        return None
    yy, xx = np.nonzero(m)
    pts = np.stack([xx * col_mm, yy * row_mm], 1)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    major = vt[0]
    proj = (pts - c) @ major
    length = float(proj.max() - proj.min())
    height = float(m.sum() * col_mm * row_mm / max(length, 1e-6))
    # posterior margin row of the disc (most posterior 15% of pixels)
    post = xx >= np.percentile(xx, 85)
    r_disc = int(np.round(yy[post].mean()))
    sig = _disc_signal(img, lab, slabs, v)
    canal_disc = _band(canal, r_disc - 2, r_disc + 2, min, cos_by_row, col_mm)
    cord_disc = _band(cord, r_disc - 2, r_disc + 2, min, cos_by_row, col_mm)
    ref = [vert_rows.get(up), vert_rows.get(lo if lo != "S1" else "sacrum")]
    ref = [r for r in ref if r is not None]
    canal_ref = [band_ for band_ in (_band(canal, r - 2, r + 2, np.mean, cos_by_row, col_mm) for r in ref) if band_]
    cord_ref = [band_ for band_ in (_band(cord, r - 2, r + 2, np.mean, cos_by_row, col_mm) for r in ref) if band_]
    rec = {
        "level": f"{up}-{lo}", "label": v,
        "disc_height_mm": round(height, 1), "disc_ap_length_mm": round(length, 1),
        "disc_to_csf_signal": round(sig / csf_med, 2) if sig and csf_med else None,
        "canal_ap_at_disc_mm": round(canal_disc, 1) if canal_disc else None,
        "canal_ap_mid_body_mm": round(float(np.mean(canal_ref)), 1) if canal_ref else None,
    }
    if rec["canal_ap_at_disc_mm"] and rec["canal_ap_mid_body_mm"]:
        rec["canal_narrowing_pct"] = round(100 * (1 - rec["canal_ap_at_disc_mm"] / rec["canal_ap_mid_body_mm"]), 0)
    if cord_disc:
        rec["cord_ap_at_disc_mm"] = round(cord_disc, 1)
        if cord_ref:
            rec["cord_ap_ref_mm"] = round(float(np.mean(cord_ref)), 1)
            rec["mscc_pct"] = round(100 * (1 - cord_disc / float(np.mean(cord_ref))), 0)
    return rec


def _conus_tip(lab, present):
    if not ((lab == 1).any() and "L1" in present):
        return None
    ks = [k for k in range(lab.shape[2]) if (lab[:, :, k] == 1).any()]
    best = max(ks, key=lambda k: (lab[:, :, k] == 1).sum())
    L = lab[:, :, best].T
    yy, _ = np.nonzero(L == 1)
    tip = int(yy.max())
    where = None
    for name, v in present.items():
        if name in ("cord", "canal"):
            continue
        rr = np.nonzero(L == v)[0]
        if len(rr) and rr.min() <= tip <= rr.max():
            where = name
    return {"row": tip, "at": where}


def measure(img_p, lab_p, mid=None):
    nimg = nib.load(img_p)
    img = np.asarray(nimg.dataobj).astype(np.float32)
    lab = np.asarray(nib.load(lab_p).dataobj).astype(np.int32)
    zooms = nimg.header.get_zooms()
    col_mm, row_mm = float(zooms[0]), float(zooms[1])
    canal_counts = [(lab[:, :, k] == 2).sum() for k in range(lab.shape[2])]
    cord_counts = [(lab[:, :, k] == 1).sum() for k in range(lab.shape[2])]
    # midline = most cord where there is a real cord (cervical), else most canal (lumbar)
    auto = int(np.argmax(cord_counts)) if sum(cord_counts) > 5000 else int(np.argmax(canal_counts))
    mid = auto if mid is None else int(mid)
    slabs = [k for k in (mid - 1, mid, mid + 1) if 0 <= k < lab.shape[2]]
    L = lab[:, :, mid].T  # rows x cols; +col = posterior, +row = inferior

    canal = (L == 2) | (L == 1)
    cord = L == 1
    cos_by_row = _canal_cos_by_row(canal, col_mm, row_mm)
    csf_med = _csf_median(img, lab, slabs)

    present = {NAMES[int(v)]: int(v) for v in np.unique(lab) if int(v) in NAMES}
    vert_rows = _vert_rows(L, present)
    out_levels = []
    for name, v in sorted(present.items(), key=lambda kv: ORDER.index(kv[0].split("_")[2]) if kv[0].startswith("disc_") else 99):
        if not name.startswith("disc_"):
            continue
        rec = _disc_record(name, v, L, img, lab, slabs, (col_mm, row_mm), csf_med, vert_rows, canal, cord, cos_by_row)
        if rec is not None:
            out_levels.append(rec)

    res = {"mid_slice": mid, "csf_median": csf_med, "levels": out_levels}
    tip = _conus_tip(lab, present)
    if tip is not None:
        res["conus_tip"] = tip
    return res


def main(img_p, lab_p, out, mid=None):
    res = measure(img_p, lab_p, mid)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4], mid=int(sys.argv[4]) if len(sys.argv) > 4 else None)
