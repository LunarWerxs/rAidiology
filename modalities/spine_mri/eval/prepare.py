"""Convert one split's SPIDER series into what the pipeline reads, and check SPIDER's label
convention on every mask before anything relies on it.

For each patient of the split that has a T2 (spider.t2_choice):
  <work>/<split>/t2/<p>.nii.gz         the T2, PIR (TotalSpineSeg input; outputs keep the name)
  <work>/<split>/t1/<p>.nii.gz         the T1 when present, PIR (grader input, Phase 2)
  <work>/<split>/mask/<p>.nii.gz       SPIDER's mask for that T2, PIR, same grid (asserted)
  <work>/<split>/spiderlab/<p>.nii.gz  SPIDER's mask relabelled with TotalSpineSeg ids at the
                                       nominal (bottom-up) levels, so metrics.py can measure discs
                                       located by SPIDER's own masks
  <work>/<split>/prepare.json          the convention checks and counts
SPIDER masks are used only for matching and for that comparison, never as model input.
When the T2's mask lacks a graded disc that SPIDER annotated on the same study's T2 SPACE or T1,
that disc's mask is resampled from the other series onto the T2 through world coordinates (nearest
neighbour) and recorded in prepare.json. A graded disc still absent after that lies outside the
T2's field of view (patient 35's T2 stops below its fourth disc): match.py drops it as not in view.
The order check is fatal only for discs 1-6 (the scored levels); a structure within 5 rows of the
top edge is cut by the field of view and is not checked.
Usage: python prepare.py <dev|test> [work_dir]"""
import json
import os
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402
paths.use("spine_mri")
from spider import guard_test, grades, mha_to_nifti, path, patient_series  # noqa: E402
from split import load as load_split  # noqa: E402

WORK = os.path.join(paths.WORK, "eval", "spider")
CANAL = 100
EDGE = 5   # rows: a structure this close to the top edge is cut by the field of view


def vert_id(i):
    """TotalSpineSeg vertebra id of SPIDER vertebra i at its nominal level (1 = L5 ... 5 = L1, 6 = T12 ...)."""
    return 46 - i if i <= 5 else 38 - i if i <= 17 else None


def disc_id(i):
    """TotalSpineSeg disc id of SPIDER disc 200+i at its nominal level (1 = L5-S1, 6 = T12-L1, 7 = T11-T12 ...)."""
    return 100 if i == 1 else 97 - i if 2 <= i <= 6 else 89 - i if 7 <= i <= 17 else None


def spider_to_tss(m):
    out = np.zeros(m.shape, np.uint16)
    out[m == CANAL] = 2
    for v in np.unique(m):
        v = int(v)
        if 1 <= v < CANAL and vert_id(v):
            out[m == v] = vert_id(v)
        elif v > 200 and disc_id(v - 200):
            out[m == v] = disc_id(v - 200)
    return out


def fill_missing_discs(p, info, msk_path, t2_path, graded):
    """Add graded discs missing from the T2 mask from the study's T2 SPACE or T1 mask; returns
    [(ivd, source)] added. Rewrites the mask file when anything was added."""
    import SimpleITK as sitk
    m_img = sitk.ReadImage(msk_path)
    m = sitk.GetArrayFromImage(m_img)
    missing = [i for i in graded if i >= 1 and not (m == 200 + i).any()]
    added = []
    for seq in ("t2_SPACE", "t1"):
        if not missing:
            break
        src = path("masks", f"{p}_{seq}.mha")
        if not os.path.exists(src) or info["t2"].endswith(seq):
            continue
        other = sitk.Resample(sitk.ReadImage(src), sitk.ReadImage(t2_path), sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        o = sitk.GetArrayFromImage(other)
        for i in list(missing):
            sel = (o == 200 + i) & (m == 0)
            if sel.sum() >= 20:
                m[sel] = 200 + i
                added.append((i, seq))
                missing.remove(i)
    if added:
        out = sitk.GetImageFromArray(m.astype(np.uint16))
        out.CopyInformation(m_img)
        sitk.WriteImage(out, msk_path)
    return added


def rows_centroid(m, v):
    yy = np.nonzero(m == v)[1]  # PIR: axis 1 runs toward inferior
    return float(yy.mean()) if len(yy) else None


def main(split, work=WORK):
    guard_test(split)
    sp = load_split()
    ps = patient_series()
    g = grades()
    base = os.path.join(work, split)
    report = {"split": split, "patients": 0, "no_t2": [], "t2_space": [], "no_t1": [], "grid_mismatch": [],
              "order_violations": [], "graded_without_mask": [], "mask_filled_from_other_series": [],
              "ivd0_mask": [], "num_discs_mismatch": []}
    for key, s in sorted(sp["patients"].items(), key=lambda kv: int(kv[0])):
        if s["split"] != split:
            continue
        p = int(key)
        info = ps[p]
        if not info["t2"]:
            report["no_t2"].append(p)
            continue
        report["patients"] += 1
        if info["t2"].endswith("SPACE"):
            report["t2_space"].append(p)
        t2 = os.path.join(base, "t2", f"{p}.nii.gz")
        msk = os.path.join(base, "mask", f"{p}.nii.gz")
        if not os.path.exists(t2):
            mha_to_nifti(path("images", info["t2"] + ".mha"), t2)
        graded = sorted(int(i) for i in g[g.patient == p].ivd)
        orig = set(int(v) - 200 for v in np.unique(sitk.GetArrayFromImage(sitk.ReadImage(path("masks", info["t2"] + ".mha")))) if v >= 200)
        lacking = [i for i in graded if i >= 1 and i not in orig]
        if not os.path.exists(msk) or lacking:
            mha_to_nifti(path("masks", info["t2"] + ".mha"), msk, label=True)   # start from SPIDER's own mask
        added = fill_missing_discs(p, info, msk, t2, graded) if lacking else []
        lab_p = os.path.join(base, "spiderlab", f"{p}.nii.gz")
        if lacking:
            report["mask_filled_from_other_series"].append([p, added])
            if os.path.exists(lab_p):
                os.remove(lab_p)
        a, b = nib.load(t2), nib.load(msk)
        if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-3):
            report["grid_mismatch"].append(p)
        if info["t1"]:
            t1 = os.path.join(base, "t1", f"{p}.nii.gz")
            if not os.path.exists(t1):
                mha_to_nifti(path("images", info["t1"] + ".mha"), t1)
        else:
            report["no_t1"].append(p)
        m = np.asarray(b.dataobj).astype(np.int32)
        labs = set(int(v) for v in np.unique(m))
        discs = sorted(v - 200 for v in labs if v >= 200)
        if 0 in discs:
            report["ivd0_mask"].append(p)
        # convention: disc 200+i lies below vertebra i and above vertebra i-1; a structure cut by
        # the top edge of the image has a misleading centroid, so it is not checked
        def cut(v):
            rows = np.nonzero(m == v)[1]
            return len(rows) == 0 or rows.min() < EDGE

        for i in discs:
            if i < 1 or cut(i) or cut(200 + i):
                continue
            d, up, lo = rows_centroid(m, 200 + i), rows_centroid(m, i), rows_centroid(m, i - 1) if i > 1 else None
            if up is not None and not d > up:
                report["order_violations"].append([p, i, "disc not below vertebra i"])
            if lo is not None and not d < lo:
                report["order_violations"].append([p, i, "disc not above vertebra i-1"])
        missing = [i for i in graded if i not in discs]
        if missing:
            report["graded_without_mask"].append([p, missing])
        if int(info["row"].num_discs) != len([i for i in discs if i >= 1]):
            report["num_discs_mismatch"].append([p, int(info["row"].num_discs), len(discs)])
        if not os.path.exists(lab_p):
            os.makedirs(os.path.dirname(lab_p), exist_ok=True)
            nib.save(nib.Nifti1Image(spider_to_tss(m), b.affine, b.header), lab_p)
    for k in list(report):
        if isinstance(report[k], list):
            report["n_" + k] = len(report[k])
    json.dump(report, open(os.path.join(base, "prepare.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=1))
    scored_bad = [v for v in report["order_violations"] if v[1] <= 6]   # IVD 1-6: the levels the scorecard can score
    if report["grid_mismatch"] or scored_bad:
        sys.exit("SPIDER convention check failed; see prepare.json")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or [WORK]))
