"""SPIDER (Zenodo 10159290, CC BY 4.0) access: which series a patient has, which T2 to use, the
grades, and conversion of its .mha volumes to NIfTI in the orientation to_nifti.py produces for a
patient disc's sagittal T2 (PIR), so metrics.py and TotalSpineSeg see the same axis order.

SPIDER numbers structures bottom-up (paper, Segmentation data): vertebra 1 is the most caudal
lumbar vertebra, disc 200+i lies under vertebra i, the canal is 100. That count is not an
anatomical level (transitional vertebrae shift it), so it is used only as a "nominal" level for
reporting and for unmatched discs; the scored level comes from TotalSpineSeg."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from core import paths  # noqa: E402
ROOT = os.environ.get("SPIDER_ROOT", os.path.join(paths.DATASETS, "public", "spider"))
ORIENT = "PIR"  # what to_nifti.py writes for a patient disc's sagittal T2 (checked on real series)
SEED = 20260930

ITEMS = ["herniation", "narrowing", "bulging", "spondylolisthesis", "up_endplate", "low_endplate", "modic"]
GRADE_COLS = {"modic": "Modic", "up_endplate": "UP endplate", "low_endplate": "LOW endplate",
              "spondylolisthesis": "Spondylolisthesis", "herniation": "Disc herniation",
              "narrowing": "Disc narrowing", "bulging": "Disc bulging", "pfirrmann": "Pfirrman grade"}
# Nominal level of SPIDER disc i, counted up from the most caudal lumbar vertebra.
NOMINAL = {1: "L5-S1", 2: "L4-L5", 3: "L3-L4", 4: "L2-L3", 5: "L1-L2", 6: "T12-L1", 7: "T11-T12",
           8: "T10-T11", 9: "T9-T10"}
SCORED_LEVELS = ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def guard_test(split):
    """The test split is opened once, by final_eval.py, which sets RAIDIOLOGY_FINAL_EVAL=1 for its steps."""
    if split == "test" and os.environ.get("RAIDIOLOGY_FINAL_EVAL") != "1":
        raise SystemExit("The test split is opened only by final_eval.py (once).")


def path(*p):
    return os.path.join(ROOT, *p)


def overview():
    o = pd.read_csv(path("overview.csv"))
    o["patient"] = o.new_file_name.str.split("_").str[0].astype(int)
    o["seq"] = o.new_file_name.str.split("_", n=1).str[1]
    o["Manufacturer"] = o.Manufacturer.str.strip()
    o["sex"] = o.sex.str.strip()
    return o


def vendor(manufacturer):
    m = manufacturer.lower()
    return "Philips" if "philips" in m else "Siemens" if "siemens" in m else manufacturer


def grades():
    """One row per graded disc, with rAidiology's item names; modic keeps its type (0-3)."""
    g = pd.read_csv(path("radiological_gradings.csv"))
    out = pd.DataFrame({"patient": g["Patient"].astype(int), "ivd": g["IVD label"].astype(int)})
    for k, c in GRADE_COLS.items():
        out[k] = g[c].astype(int)
    return out


def t2_choice(seqs):
    """The T2 rule: `_t2`, or `_t2_SPACE` only when `_t2` is missing, else None."""
    return "t2" if "t2" in seqs else "t2_SPACE" if "t2_SPACE" in seqs else None


def patient_series():
    """{patient: {"t2": "<name>"|None, "t1": "<name>"|None, "row": overview row of the T2 (or first)}}"""
    o = overview()
    out = {}
    for p, d in o.groupby("patient"):
        seqs = set(d.seq)
        t2 = t2_choice(seqs)
        row = d[d.seq == t2].iloc[0] if t2 else d.iloc[0]
        out[int(p)] = {"t2": f"{p}_{t2}" if t2 else None, "t1": f"{p}_t1" if "t1" in seqs else None,
                       "row": row}
    return out


def reorient(img):
    import SimpleITK as sitk
    return sitk.DICOMOrient(img, ORIENT)


def mha_to_nifti(src, dst, label=False):
    """Read a SPIDER .mha, reorient to PIR, write NIfTI; returns the reoriented image."""
    import nibabel as nib
    import SimpleITK as sitk
    img = reorient(sitk.ReadImage(src))
    if label:
        img = sitk.Cast(img, sitk.sitkUInt16)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sitk.WriteImage(img, dst)
    codes = nib.aff2axcodes(nib.load(dst).affine)
    assert "".join(codes) == ORIENT, f"{dst}: axis codes {codes}, metrics.py needs {ORIENT}"
    return img


def assert_axes(nifti_path):
    """metrics.py reads lab[:, :, k].T as rows = inferior, cols = posterior, k = left->right."""
    import nibabel as nib
    codes = "".join(nib.aff2axcodes(nib.load(nifti_path).affine))
    assert codes == ORIENT, f"{nifti_path}: axis codes {codes}, metrics.py needs {ORIENT}"


def rng():
    return np.random.default_rng(SEED)
