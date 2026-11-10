"""Cervical cord compression compared with healthy adults (Phase 3): rAidiology's own healthy-cord reference.

1. sct_deepseg spinalcord (Spinal Cord Toolbox 7.3) segments the cord on the cervical axial T2 (when several axial T2
   series share one geometry, the sharpest: highest variance of the Laplacian).
2. Vertebral levels come from the outliner's labels on the cervical sagittal T2 (labels.outline_path: outliner v2 by
   default, TotalSpineSeg step-2 ids), taken in world coordinates, so they apply to the axial series' cord.
3. At each cervical level where the reading (reading.json) reports cord or canal narrowing, the compression site is the
   cord section nearest that disc; the levels above and below are the mid-bodies of the vertebrae either side.
4. healthy_norms.py measures the patient's cord (area, AP and transverse diameters perpendicular to the centreline)
   there and computes SCT's normalised MSCC formula against healthy_norms.json (203 healthy adults of Spine Generic
   data-multi-subject, CC BY 4.0), matched by sex and age +-10 years (both sexes when fewer than 10 match).
   No -normalize-hc: the PAM50 template it needed (no licence) is no longer used anywhere here.
--check labels C5-C6 whatever the reading says, only to prove the tools run; its output goes to
work/sct_compression.check.json and never into the report.
Usage: python cord_compression.py <reading.json> <work_dir> --age <years> --sex <M|F> [--check]"""
import argparse
import json
import os
import re
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from labels import NAMES, outline_path  # noqa: E402
import healthy_norms as hn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402
SCT = os.path.join(paths.TOOLS, "sct_7.3", "python", "envs", "venv_sct", "Scripts")
NARROW = re.compile(r"(cord|canal)[^.]*(narrow|compress|flatten|stenosis|indent)", re.I)


def sct(tool, *args):
    exe = os.path.join(SCT, tool + ".exe")
    print("+", tool, " ".join(args), flush=True)
    subprocess.run([exe, *args], check=True)


def sharpest(paths):
    def score(p):
        a = np.asarray(nib.load(p).dataobj).astype(np.float32)
        return float(np.mean([ndimage.laplace(a[..., k]).var() / max(a[..., k].var(), 1e-6) for k in range(a.shape[2])]))
    return max(paths, key=score)


def world(aff, ijk):
    return (aff @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T


def outline_points(outline_p):
    """{name: world centroid (mm)} of every vertebra and disc in an outliner label map (labels.NAMES)."""
    img = nib.load(outline_p)
    lab = np.asarray(img.dataobj).astype(np.int32)
    out = {}
    for v in np.unique(lab):
        name = NAMES.get(int(v))
        if name and name not in ("cord", "canal", "sacrum"):
            out[name] = world(img.affine, np.argwhere(lab == v)).mean(0)
    return out


def on_cord(secs, point):
    """The cord section level with a world point (its nearest section, if the point is within 3 mm of it along the
    cord); None when the point lies beyond the measured cord."""
    i = hn.arc_of(secs, point)
    along = abs(float((np.asarray(point) - secs["points"][i]) @ secs["tangents"][i]))
    return i if along <= 3.0 and 0 < i < len(secs["area"]) - 1 else None


def compression(secs, pts, level, norms, sex, age):
    """The patient's cord at a reported disc and the vertebrae either side, each against healthy adults at that level,
    and SCT's normalised MSCC for each metric (diameter_AP is the headline, as sct_compute_compression's default)."""
    upper, lower = level.split("-")
    i = on_cord(secs, pts["disc_" + level.replace("-", "_")]) if "disc_" + level.replace("-", "_") in pts else None
    if i is None:
        return {"cord_compression_ratio_hc": None, "note": f"the axial cord does not reach the {level} disc"}
    at = hn.value_at(secs, i, hn.DISC_HALF_MM)
    side = {}
    for name in (upper, lower):
        j = on_cord(secs, pts[name]) if name in pts else None
        if j is not None:
            side[name] = hn.value_at(secs, j, hn.MID_HALF_MM)
    if not side:
        return {"cord_compression_ratio_hc": None, "patient": {level: at},
                "note": f"the axial cord reaches neither {upper} nor {lower}, so there is nothing to compare with"}
    above, below = side.get(upper, side.get(lower)), side.get(lower, side.get(upper))
    ref = {lv: hn.match(norms, lv, sex, age) for lv in (upper, lower, level)}
    h_above, h_below = (ref[upper] if upper in side else ref[lower]), (ref[lower] if lower in side else ref[upper])
    mscc = {}
    for m in ("diameter_AP", "area", "diameter_RL"):
        hs = (h_above["mean"][m], h_below["mean"][m], ref[level]["mean"][m])
        mscc[m] = round(hn.mscc_norm((above[m], below[m], at[m]), hs), 3) if all(hs) else None
    out = {"cord_compression_ratio_hc": mscc["diameter_AP"], "mscc_normalized": mscc,
           "patient": {level: at, **side},
           "healthy": {lv: {k: r[k] for k in ("sex", "age", "n", "fallback", "mean", "sd")} for lv, r in ref.items()}}
    if len(side) == 1:
        out["note"] = f"only {next(iter(side))} is on the axial cord, so it stands for both sides (as SCT does)"
    return out


def reported_levels(reading):
    out = []
    for lv in reading.get("levels", []):
        if lv.get("region") != "cervical":
            continue
        text = " ".join(str(lv.get(k, "")) for k in ("disc", "note"))
        if lv.get("canal", "none") != "none" or NARROW.search(text):
            out.append(lv["level"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reading")
    ap.add_argument("work")
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--sex", choices=["M", "F"], required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    reading = json.load(open(a.reading, encoding="utf-8"))
    levels = ["C5-C6"] if a.check else reported_levels(reading)
    series = json.load(open(os.path.join(a.work, "nifti", "series.json")))
    axials = [os.path.join(a.work, "nifti", s["name"] + ".nii.gz") for s in series
              if s["region"] == "C" and s["plane"] == "axial" and s["sequence"] == "T2"]
    sag = [s for s in series if s["region"] == "C" and s["plane"] == "sagittal" and s["sequence"] == "T2"]
    outline_p = outline_path(a.work, max(sag, key=lambda s: s["count"])["name"])
    out_json = os.path.join(a.work, "sct_compression.check.json" if a.check else "sct_compression.json")
    res = {"tool": "rAidiology healthy_norms.py: cord area and diameters perpendicular to the centreline, SCT's normalised "
                   "MSCC formula; cord segmentation by Spinal Cord Toolbox 7.3 sct_deepseg spinalcord",
           "reference": "rAidiology's healthy-cord reference (sct/healthy_norms.json): 203 healthy adults of Spine Generic "
                        "data-multi-subject (CC BY 4.0), measured from its cord masks and hand-placed disc points",
           "matched": {"sex": a.sex, "age": [a.age - 10, a.age + 10]},
           "licences": "Spine Generic data-multi-subject CC BY 4.0 (credit: doi:10.5281/zenodo.4299140); "
                       "Spinal Cord Toolbox LGPL-3.0 (cord segmentation only)",
           "levels_from_reading": levels, "levels": {}}
    if not levels:
        res["note"] = "The reading reports no cord or canal narrowing at any cervical level, so no compression site was labelled."
        json.dump(res, open(out_json, "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    d = os.path.join(a.work, "sct")
    os.makedirs(d, exist_ok=True)
    ax = sharpest(axials)
    res["axial_series"] = os.path.basename(ax)
    seg = os.path.join(d, "cord_seg.nii.gz")
    sct("sct_deepseg", "spinalcord", "-i", ax, "-o", seg)
    secs = hn.sections(nib.load(seg))
    if secs is None:
        res["note"] = "sct_deepseg found no cord on the axial series."
    else:
        pts = outline_points(outline_p)
        norms = hn.load()
        for lv in levels:
            res["levels"][lv] = compression(secs, pts, lv, norms, a.sex, a.age)
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
