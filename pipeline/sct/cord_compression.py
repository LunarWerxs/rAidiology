"""Cervical cord compression compared with healthy adults (Phase 3), with Spinal Cord Toolbox 7.3.

1. sct_deepseg spinalcord segments the cord on the cervical axial T2 (when several axial T2
   series share one geometry, the sharpest: highest variance of the Laplacian).
2. Vertebral levels come from TotalSpineSeg's labels on the cervical sagittal T2, mapped to each
   axial slice through world coordinates (thick, widely spaced axial slices are too
   sparse for sct_label_vertebrae). SCT numbering: C1 = 1 ... C7 = 7, T1 = 8.
3. One compression label (sct_label_utils -create) on the cord at each cervical level where the
   reading (reading.json) reports cord or canal narrowing, on the axial slice nearest that disc.
4. sct_compute_compression -normalize-hc 1 against the healthy-control database (203 adults from
   spine-generic, PAM50-normalized-metrics, MIT), matched by sex and age +-10 years.
PAM50 template: permission pending; TSS weights: permission pending (LICENSING.md).
--check labels C5-C6 whatever the reading says, only to prove the tools run; its output goes to
work/sct_compression.check.json and never into the report.
Usage: python cord_compression.py <reading.json> <work_dir> --age <years> --sex <M|F> [--check]"""
import argparse
import csv
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
from labels import NAMES  # noqa: E402

SCT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "_tools", "sct_7.3", "python", "envs", "venv_sct", "Scripts"))
SCT_LEVEL = {**{f"C{i}": i for i in range(1, 8)}, **{f"T{i}": 7 + i for i in range(1, 13)}}
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


def vertebra_ranges(tss_p):
    """{name: (z_min, z_max)} in world mm (RAS z = superior) for each vertebra in a TotalSpineSeg label map."""
    img = nib.load(tss_p)
    lab = np.asarray(img.dataobj).astype(np.int32)
    out = {}
    for v in np.unique(lab):
        name = NAMES.get(int(v))
        if name in SCT_LEVEL:
            z = world(img.affine, np.argwhere(lab == v))[:, 2]
            out[name] = (float(z.min()), float(z.max()))
    return out


def disc_z(tss_p, level):
    img = nib.load(tss_p)
    lab = np.asarray(img.dataobj).astype(np.int32)
    name = "disc_" + level.replace("-", "_")
    ids = [k for k, v in NAMES.items() if v == name]
    vox = np.argwhere(lab == ids[0]) if ids else np.empty((0, 3))
    return float(world(img.affine, vox)[:, 2].mean()) if len(vox) else None


def vertfile(seg_p, ranges, out_p):
    img = nib.load(seg_p)
    seg = np.asarray(img.dataobj) > 0.5
    out = np.zeros(seg.shape, np.uint8)
    for k in range(seg.shape[2]):
        if not seg[..., k].any():
            continue
        c = np.argwhere(seg[..., k]).mean(0)
        z = world(img.affine, np.array([[c[0], c[1], k]]))[0, 2]
        name = min(ranges, key=lambda n: 0 if ranges[n][0] <= z <= ranges[n][1] else min(abs(z - ranges[n][0]), abs(z - ranges[n][1])))
        out[..., k][seg[..., k]] = SCT_LEVEL[name]
    nib.save(nib.Nifti1Image(out, img.affine, img.header), out_p)


def label_points(seg_p, zs):
    """x,y,z voxel strings on the cord at the axial slice nearest each world z."""
    img = nib.load(seg_p)
    seg = np.asarray(img.dataobj) > 0.5
    pts = []
    for z in zs:
        ks = [k for k in range(seg.shape[2]) if seg[..., k].any()]
        kz = {}
        for k in ks:
            c = np.argwhere(seg[..., k]).mean(0)
            kz[k] = world(img.affine, np.array([[c[0], c[1], k]]))[0, 2]   # z at the cord, not the slice corner
        k = min(ks, key=lambda k: abs(kz[k] - z))
        c = np.round(np.argwhere(seg[..., k]).mean(0)).astype(int)
        pts.append(f"{c[0]},{c[1]},{k},1")
    return pts


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
    tss_p = os.path.join(a.work, "tss_out", "step2_output", max(sag, key=lambda s: s["count"])["name"] + ".nii.gz")
    out_json = os.path.join(a.work, "sct_compression.check.json" if a.check else "sct_compression.json")
    res = {"tool": "Spinal Cord Toolbox 7.3, sct_compute_compression -normalize-hc 1",
           "reference": "203 healthy adults (spine-generic, PAM50-normalized-metrics, MIT)",
           "matched": {"sex": a.sex, "age": [a.age - 10, a.age + 10]},
           "licences": "PAM50 template: permission pending; TSS weights: permission pending",
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
    vert = os.path.join(d, "cord_seg_labeled.nii.gz")
    vertfile(seg, vertebra_ranges(tss_p), vert)
    zs = [disc_z(tss_p, lv) for lv in levels]
    keep = [(lv, z) for lv, z in zip(levels, zs) if z is not None]
    lab = os.path.join(d, "compression_labels.nii.gz")
    sct("sct_label_utils", "-i", seg, "-create", ":".join(label_points(seg, [z for _, z in keep])), "-o", lab)
    csv_p = os.path.join(d, "compression_metrics.csv")
    sct("sct_compute_compression", "-i", seg, "-vertfile", vert, "-l", lab, "-normalize-hc", "1",
        "-sex", a.sex, "-age", str(a.age - 10), str(a.age + 10), "-o", csv_p)
    rows = list(csv.DictReader(open(csv_p)))
    for (lv, _), r in zip(keep, rows):
        num = {k: float(v) for k, v in r.items() if re.fullmatch(r"-?[\d.]+(e-?\d+)?", str(v).strip() or "x")}
        hc = [v for k, v in num.items() if "normalized" in k.lower() and "ratio" in k.lower()]
        res["levels"][lv] = {"cord_compression_ratio_hc": round(hc[0], 3) if hc else None, "sct_row": r}
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
