"""Extra training scans from NLST: LIDC-annot-NLST501, for the nodule detector and the cancer-or-not model.

What. LIDC-annot-NLST501 (TCIA, CC BY 4.0, Renoust et al. 2026, DOI 10.7937/Y2DX-E334) is 501 low-dose chest
CTs of the National Lung Screening Trial, each read by four radiologists the way LIDC-IDRI was: every lesion
marked as a target lesion (TL, 4 to 30 mm), a mass (30 to 60 mm) or a non-target lesion (NTL, anything else
that looks like a nodule), the most suspicious rated for malignancy from 1 to 10. The marks are voxel
positions; this module turns them into the scanner's millimetres with each scan's DICOM geometry and gives
them the two answer keys the project already uses. fetch_tcia.py downloads the scans; whatever is complete
(a `.unpacked` marker) is used, so it works on a partly downloaded set.

Why. More scans to learn from, never to be scored. The detector (`nodule.py train --extra-nlst`) gets a
second family of scans from another trial and scanners; the cancer-or-not model
(`nodule_malignancy.py train --extra-nlst`) gets labels of a kind LIDC does not have: NLST follows every
patient for years, so a nodule in a patient who never had lung cancer is benign by follow-up, and the one
nodule that grew into the patient's cancer is malignant. These labels are follow-up and pathology at the
patient level, stronger than LIDC's reader opinions. Their weak point: the data says which PATIENT had
cancer, not which nodule. Picking the most suspicious nodule of a cancer patient is an ASSUMPTION (one
nodule per cancer patient, and it is the one the readers rated highest, at 5 or more); every other nodule of
a cancer patient is left out rather than guessed.

The scans (`scans()`), (uid, patient, folder) of each complete download except:
  - the 24 corrected series: the annotations and series.csv name a corrected copy (a new series uid, a slice
    dropped, merged or duplicated; correction.csv) while the download is NLST's original, so a mark's slice
    number does not refer to the slices we hold. They are left out (the original's uid appears in no mark);
  - slices thicker than 2.5 mm (LUNA16's rule); 28 of the 501 are 3.2 mm.

Coordinates. The documentation (page 12, annotations table) says only "location_x/y/z: declared position in
x/y/z in voxel": it does not say whether counting starts at 0 or 1, whether z runs with the slice's
position or against it, or whether x and y are mirrored. The page-4 text says the NIfTI masks match the
DICOM files. So the convention is a setting, `CONVENTION`, read as: x is the column and y the row of the
slice, z the index of the slice; `base` the first index (0 or 1), `z` "asc" when index 0 is the lowest z
(ImagePositionPatient) and "desc" when it is the highest, `flip_x` / `flip_y` when the column or row counts
from the far side. `check` tries all 16 on the marks and prints how many centres land on tissue under each;
the one in CONVENTION must reach at least 90%, else `check` says which to set.

The rule (LUNA16's, as lidc_extra.py applies it to LIDC; marks of different readers are matched into one
lesion greedily nearest first, a reader's mark joining a lesion when it lies within 1.5 times the larger of
the two radii (2 mm at least) of one of its marks, never two marks of one reader in a lesion).
  - nodules: a TL or Mass lesion marked by at least 3 readers whose mean diameter is 3 mm or more; centre =
    mean of the marks, diameter = their mean (extracted_lad when the mark has it, else reported_measure).
  - excluded findings (counted neither way): every other TL or Mass lesion, and every NTL mark; diameter
    the measure, 0 when missing.
Both frames use nodule.py's columns: seriesuid, coordX, coordY, coordZ, diameter_mm.

Labels (`labels()`), one row per nodule above: seriesuid, patient, coordX/Y/Z, diameter_mm, n_readers,
suspicion_mean (mean tl_malignancy_suspicion over its marks), nlst_diagnosis and label: 0 for every nodule
of a patient with no cancer in NLST's follow-up; 1 for the one nodule with the highest suspicion_mean in a
cancer patient, when that mean is 5 or more; NaN (left out, the trainer drops it) for every other nodule of
a cancer patient.

The label fix (idea 23; docs/efficiency/README.md, "Cancer or not v5, idea 23"; HSwarm job 20261005-044211-4fe2).
The cancers' own NLST-Sybil boxes, mapped in 3D by a skeptic, contradict 12 of the 112 positives (another nodule,
or none) and confirm 26 cancers `labels()` left unknown. `labels(fixed=True)` applies nlst501_fix.csv: the 38
changed rows (seriesuid, coordinates to 0.1 mm, label_before, label_after, evidence), each matched to exactly one
nodule (within 0.06 mm) whose label must equal label_before, else an assertion fails. `fixcsv` writes that file
from the skeptic's work/audit/ct-lung/nlst-skeptic/nlst_fix.json (git-ignored) and exits 2 if any of its 472
rows disagrees with `labels()`. The default `labels()` is unchanged; nodule_malignancy.py uses the fixed one
only under `--extra-nlst --set nlst_fix=true`.

    python modalities/ct/eval/nlst501.py prepare [--jobs N]
    python modalities/ct/eval/nlst501.py check
    python modalities/ct/eval/nlst501.py fixcsv

`prepare` builds the 1 mm lung boxes into the same folder LIDC's scans use, so nodule.Scan finds them. `check`
asserts and prints: the scans used and left out by reason; the nodules and excluded findings; how many nodule
centres lie above -600 HU in the prepared box; the coordinates' test, at least 90% of the solid marks' centres
above -600 HU (ground-glass nodules are rightly fainter); how many readers marked each nodule; the labels'
counts; and that no NLST series is a LUNA16 or LIDC one. Numbers the check printed (2026-10-03): 450 scans
used (51 left out: 24 corrected, 27 thicker than 2.5 mm); 1,255 nodules in 375 scans and 15,269 excluded
findings; 88.9% of nodule centres and 90.8% of 3,718 solid marks above -600 HU under CONVENTION; readers per
nodule 3: 584, 4: 671; labels 0: 732, 1: 112, left out: 411; no overlap with LUNA16 or LIDC.
"""
import argparse
import collections
import functools
import glob
import itertools
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import paths  # noqa: E402
import lung_prepare as lp  # noqa: E402

ROOT = os.path.join(paths.DATASETS, "public", "lidc-annot-nlst501")
GEO = os.path.join(paths.WORK, "eval", "nlst501", "geometry")
PREFIX = "LIDC-annot-NLST501-"
COLS = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
MAX_THICKNESS_MM = 2.5      # LUNA16's rule
MIN_READERS = 3             # LUNA16: at least 3 of the 4 radiologists
MIN_DIAMETER_MM = 3.0
MATCH_FACTOR = 1.5          # a mark joins a lesion within this many radii (the larger of the two)
MATCH_MIN_RADIUS_MM = 2.0
MIN_SUSPICION = 5.0         # the cancer patient's most suspicious nodule is malignant from this mean
MIN_DENSE = 0.90
DENSE_HU = -600.0
# How the voxel marks count (see the docstring; the documentation does not say, `check` tests all 16).
# Measured 2026-10-03 (the documentation states none): x the column, y the row, both counted from 1, z the slice
# counted from 1 down from the highest ImagePositionPatient z. On 740 solid marks of 80 scans 90.9% of centres
# lie above -600 HU this way; 72 to 83% with x or y moved 2 pixels, 47 to 52% with z moved one slice.
CONVENTION = {"base": 1, "z": "desc", "flip_x": False, "flip_y": False}


def _csv(name):
    return pd.read_csv(os.path.join(ROOT, PREFIX + name + ".csv"), encoding="utf-8-sig")


# ---- the scans -------------------------------------------------------------------------------

def _select():
    """(scans, left_out): the usable complete downloads and {reason: count} of the complete ones left out."""
    series = _csv("series").set_index("series_uid")
    corrected = set(_csv("correction").series_uid_original)
    with open(os.path.join(ROOT, "series.json"), encoding="utf-8") as f:
        records = json.load(f)
    keep, out = [], collections.Counter()
    for x in sorted(records, key=lambda r: r["SeriesInstanceUID"]):
        uid = x["SeriesInstanceUID"]
        folder = os.path.join(ROOT, x["PatientID"], uid)
        if not os.path.exists(os.path.join(folder, ".unpacked")):
            out["still downloading"] += 1
            continue
        if uid in corrected:
            out["corrected series (marks refer to a corrected copy)"] += 1
        elif uid not in series.index:
            out["not in series.csv"] += 1
        elif not series.at[uid, "reconthickness"] <= MAX_THICKNESS_MM:      # also when missing
            out["thicker than %.1f mm" % MAX_THICKNESS_MM] += 1
        else:
            keep.append((uid, x["PatientID"], folder))
    return keep, out


def scans():
    """(uid, patient, folder) of every complete download that is usable (see the docstring)."""
    return _select()[0]


def prepare(jobs=None):
    """Build the 1 mm lung boxes of scans() into the folder LIDC's boxes are in."""
    lp.build(jobs, odir=lp.out_dir(), scans=scans())


def boxed(uid):
    return all(os.path.exists(p) for p in lp.paths_for(lp.out_dir(), uid))


# ---- voxel marks to millimetres --------------------------------------------------------------

def _head(path):
    import pydicom
    d = pydicom.dcmread(path, stop_before_pixels=True,
                        specific_tags=["ImagePositionPatient", "PixelSpacing", "ImageOrientationPatient", "Columns", "Rows"])
    iop = [float(v) for v in getattr(d, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    return [float(v) for v in d.ImagePositionPatient], [float(v) for v in d.PixelSpacing], iop, [int(d.Columns), int(d.Rows)]


def geometry(uid, folder):
    """{'zs': slice z (mm) rising, repeated z read once as lung_prepare.read_scan does, 'x0', 'y0' (mm, of the
    lowest slice), 'spacing': [row, col] mm, 'dir': [+1 or -1 for x with the column, for y with the row], 'shape': [columns, rows]}.
    Cached as JSON under <Work>/eval/nlst501/geometry/."""
    cache = os.path.join(GEO, uid + ".json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    with ThreadPoolExecutor(8) as pool:
        heads = list(pool.map(_head, files))
    heads.sort(key=lambda h: h[0][2])
    zs = np.array([h[0][2] for h in heads])
    keep = np.r_[True, np.diff(zs) > 1e-3]
    first = [h for h, k in zip(heads, keep) if k][0]
    g = {"zs": zs[keep].tolist(), "x0": first[0][0], "y0": first[0][1], "spacing": first[1], "shape": first[3],
         "dir": [1 if first[2][0] >= 0 else -1, 1 if first[2][4] >= 0 else -1]}
    os.makedirs(GEO, exist_ok=True)
    with open(cache + ".part", "w", encoding="utf-8") as f:
        json.dump(g, f)
    os.replace(cache + ".part", cache)
    return g


def to_world(g, x, y, z, conv=None):
    """World (x, y, z) mm of voxel marks (arrays: column, row, slice index) in a scan of geometry g."""
    c = CONVENTION if conv is None else conv
    zs = np.asarray(g["zs"], float)
    col, row, k = np.asarray(x, float) - c["base"], np.asarray(y, float) - c["base"], np.asarray(z, float) - c["base"]
    n_col, n_row = g["shape"]
    if c["flip_x"]:
        col = n_col - 1 - col
    if c["flip_y"]:
        row = n_row - 1 - row
    k = k if c["z"] == "asc" else len(zs) - 1 - k
    return (g["x0"] + g["dir"][0] * col * g["spacing"][1], g["y0"] + g["dir"][1] * row * g["spacing"][0],
            np.interp(k, np.arange(len(zs)), zs))


# ---- the marks and the lesions ---------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def marks():
    """One row per mark of every reader in a usable scan: seriesuid, patient (pid), reader, finding_type,
    x_mm, y_mm, z_mm, diameter_mm (extracted_lad, else reported_measure; NaN when neither), suspicion, plus
    the voxel column, row and slice (vx, vy, vz) the marks were given in."""
    keep = {u: (p, f) for u, p, f in scans()}
    a = _csv("annotations")
    a = a[a.series_uid.isin(keep) & a.location_x.notna() & a.location_y.notna() & a.location_z.notna()]
    out = []
    for uid, d in a.groupby("series_uid", sort=True):
        g = geometry(uid, keep[uid][1])
        x, y, z = to_world(g, d.location_x, d.location_y, d.location_z)
        out.append(pd.DataFrame({
            "seriesuid": uid, "patient": d.pid.to_numpy(), "reader": (d.db + "/" + d.reader_id).to_numpy(),
            "finding_type": d.finding_type.to_numpy(), "x_mm": x, "y_mm": y, "z_mm": z,
            "diameter_mm": d.extracted_lad.fillna(d.reported_measure).to_numpy(),
            "suspicion": d.tl_malignancy_suspicion.to_numpy(float),
            "vx": d.location_x.to_numpy(), "vy": d.location_y.to_numpy(), "vz": d.location_z.to_numpy()}))
    cols = ["seriesuid", "patient", "reader", "finding_type", "x_mm", "y_mm", "z_mm", "diameter_mm", "suspicion",
            "vx", "vy", "vz"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def _match(d):
    """Lists of row indices of `d` (TL and Mass marks of one scan), one list per lesion: greedy, nearest pair
    first, never two marks of one reader."""
    xyz = d[["x_mm", "y_mm", "z_mm"]].to_numpy()
    rad = np.maximum(d.diameter_mm.fillna(0).to_numpy() / 2, MATCH_MIN_RADIUS_MM)
    rd = d.reader.to_numpy()
    pairs = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if rd[i] == rd[j]:
                continue
            dist = float(np.linalg.norm(xyz[i] - xyz[j]))
            if dist <= MATCH_FACTOR * max(rad[i], rad[j]):
                pairs.append((dist, i, j))
    group = list(range(len(d)))
    members = {i: [i] for i in range(len(d))}
    for _, i, j in sorted(pairs):
        a, b = group[i], group[j]
        if a == b or {rd[k] for k in members[a]} & {rd[k] for k in members[b]}:
            continue
        for k in members[b]:
            group[k] = a
        members[a] += members.pop(b)
    return sorted(members.values(), key=min)


@functools.lru_cache(maxsize=1)
def lesions():
    """One row per TL or Mass lesion: seriesuid, patient, coordX/Y/Z, diameter_mm (mean over its marks, 0 when
    none has one), n_readers, suspicion_mean (NaN when no mark has a rating), nodule (3+ readers, 3 mm+)."""
    m = marks()
    m = m[m.finding_type.isin(["TL", "Mass"])]
    rows = []
    for uid, d in m.groupby("seriesuid", sort=True):
        d = d.reset_index(drop=True)
        for idx in _match(d):
            g = d.iloc[idx]
            diam = float(g.diameter_mm.mean()) if g.diameter_mm.notna().any() else 0.0
            rows.append({"seriesuid": uid, "patient": g.patient.iloc[0], "coordX": g.x_mm.mean(),
                         "coordY": g.y_mm.mean(), "coordZ": g.z_mm.mean(), "diameter_mm": diam,
                         "n_readers": len(g),
                         "suspicion_mean": float(g.suspicion.mean()) if g.suspicion.notna().any() else np.nan})
    df = pd.DataFrame(rows, columns=["seriesuid", "patient", "coordX", "coordY", "coordZ", "diameter_mm",
                                     "n_readers", "suspicion_mean"])
    df["nodule"] = (df.n_readers >= MIN_READERS) & (df.diameter_mm >= MIN_DIAMETER_MM)
    return df


def _frame(df):
    return pd.DataFrame({"seriesuid": df.seriesuid.to_numpy(), "coordX": df.coordX.to_numpy(),
                         "coordY": df.coordY.to_numpy(), "coordZ": df.coordZ.to_numpy(),
                         "diameter_mm": df.diameter_mm.fillna(0).to_numpy()}, columns=COLS)


def build(uids):
    """(ann, exc) of the given scans by the rule in the docstring."""
    keep = set(uids)
    les = lesions()
    les = les[les.seriesuid.isin(keep)]
    m = marks()
    ntl = m[(m.finding_type == "NTL") & m.seriesuid.isin(keep)]
    ntl = ntl.rename(columns={"x_mm": "coordX", "y_mm": "coordY", "z_mm": "coordZ"})
    ann = _frame(les[les.nodule])
    exc = pd.concat([_frame(les[~les.nodule]), _frame(ntl)], ignore_index=True)
    return ann, exc


def reference():
    """(uids, ann, exc): the usable scans that have a prepared box, with nodule.py's answer-key columns."""
    uids = [u for u, _, _ in scans() if boxed(u)]
    ann, exc = build(uids)
    return uids, ann, exc


def labels(fixed=False):
    """One row per nodule: seriesuid, patient, coordX/Y/Z, diameter_mm, n_readers, suspicion_mean,
    nlst_diagnosis, label (see the docstring: 0 non-cancer patient, 1 the cancer patient's most suspicious
    nodule when its mean is 5 or more, NaN for every other nodule of a cancer patient).
    fixed=True applies nlst501_fix.csv (see the docstring); the frame is built fresh on every call, so the
    default rows are never touched by a fixed call."""
    n = _labels()
    return _apply_fix(n) if fixed else n


def _labels():
    les = lesions()
    n = les[les.nodule].drop(columns="nodule").reset_index(drop=True)
    dx = _csv("patients").set_index("pid").nlst_diagnosis
    n["nlst_diagnosis"] = n.patient.map(dx)
    n["label"] = np.nan
    n.loc[n.nlst_diagnosis == "non-cancer", "label"] = 0.0
    cancer = n[(n.nlst_diagnosis == "cancer") & n.suspicion_mean.notna()]
    for _, g in cancer.groupby("patient"):
        best = g.suspicion_mean.idxmax()
        if n.at[best, "suspicion_mean"] >= MIN_SUSPICION:
            n.at[best, "label"] = 1.0
    return n


# ---- the label fix (idea 23) -------------------------------------------------------------------

FIX_CSV = os.path.join(HERE, "nlst501_fix.csv")
FIX_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                        "work", "audit", "ct-lung", "nlst-skeptic", "nlst_fix.json")
FIX_TOL_MM = 0.06
FIX_COLS = ["seriesuid", "patient", "coordX", "coordY", "coordZ", "label_before", "label_after", "evidence"]


def _same_label(a, b):
    return (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and float(a) == float(b))


def _apply_fix(n):
    """n with the label of each nlst501_fix.csv row set to its label_after (NaN when empty). Each row must match
    exactly one nodule of its series within FIX_TOL_MM on every axis and its current label must be label_before."""
    fix = pd.read_csv(FIX_CSV, dtype={"seriesuid": str, "patient": str})
    n = n.copy()
    for r in fix.itertuples():
        d = n[(n.seriesuid == r.seriesuid)
              & ((n.coordX - r.coordX).abs() <= FIX_TOL_MM) & ((n.coordY - r.coordY).abs() <= FIX_TOL_MM)
              & ((n.coordZ - r.coordZ).abs() <= FIX_TOL_MM)]
        assert len(d) == 1, "nlst501_fix.csv: %s (%s, %s, %s) matches %d nodules, not 1" \
            % (r.seriesuid, r.coordX, r.coordY, r.coordZ, len(d))
        i = d.index[0]
        assert _same_label(n.at[i, "label"], r.label_before), \
            "nlst501_fix.csv: %s row %d has label %s, the file says label_before %s" \
            % (r.seriesuid, i, n.at[i, "label"], r.label_before)
        n.at[i, "label"] = r.label_after if not pd.isna(r.label_after) else np.nan
    return n


def fixcsv(src=FIX_JSON, dst=FIX_CSV):
    """Writes nlst501_fix.csv, the rows of the skeptic's nlst_fix.json whose label changes, after checking every
    one of the file's rows against labels(). Returns 2 (and writes nothing) when any row disagrees."""
    with open(src, encoding="utf-8") as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = rows.get("rows", rows)
    lab = labels()
    bad, out = [], []
    for r in rows:
        try:
            i = int(r["labels_index"])
            m = re.match(r"labels\(\) row (\d+) \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", str(r["nodule_key"]))
            assert m, "nodule_key not understood: %r" % r["nodule_key"]
            assert int(m.group(1)) == i, "nodule_key row %s is not labels_index %d" % (m.group(1), i)
            assert 0 <= i < len(lab), "labels_index %d outside 0..%d" % (i, len(lab) - 1)
            row = lab.iloc[i]
            assert row.seriesuid == r["series"], "series %s is not %s" % (row.seriesuid, r["series"])
            assert str(row.patient) == str(r["patient"]), "patient %s is not %s" % (row.patient, r["patient"])
            for col, g in zip(("coordX", "coordY", "coordZ"), m.groups()[1:]):
                assert abs(float(row[col]) - float(g)) <= FIX_TOL_MM, "%s %.3f is not %s" % (col, row[col], g)
            assert _same_label(row.label, r["current_label"]), \
                "current_label %s is not labels() label %s" % (r["current_label"], row.label)
            if not _same_label(r["current_label"], r["proposed_label"]):
                out.append({"seriesuid": row.seriesuid, "patient": row.patient,
                            "coordX": round(float(row.coordX), 1), "coordY": round(float(row.coordY), 1),
                            "coordZ": round(float(row.coordZ), 1),
                            "label_before": "" if pd.isna(row.label) else int(row.label),
                            "label_after": "" if pd.isna(r["proposed_label"]) else int(r["proposed_label"]),
                            "evidence": " ".join(str(r.get("evidence", "")).split())[:200]})
        except AssertionError as e:
            bad.append("row %s: %s" % (r.get("labels_index"), e))
    if bad:
        print("fixcsv refused: %d of %d rows disagree with labels()" % (len(bad), len(rows)), file=sys.stderr)
        for b in bad[:20]:
            print("  " + b, file=sys.stderr)
        return 2
    df = pd.DataFrame(out, columns=FIX_COLS)
    df.to_csv(dst, index=False, lineterminator="\n")
    gone = int(((df.label_before == 1) & (df.label_after == "")).sum())
    new = int(((df.label_before == "") & (df.label_after == 1)).sum())
    print("fixcsv: %d rows read, all agree with labels(); wrote %d changed rows to %s (%d from 1 to unknown, %d from"
          " unknown to 1)" % (len(rows), len(df), os.path.relpath(dst), gone, new))
    return 0


# ---- check -----------------------------------------------------------------------------------

def _centre_hu(box, meta, xyz):
    """Mean HU of the 3x3x3 voxels at a world point in a prepared box (None when outside it)."""
    c = np.rint(lp.world_to_voxel(meta, xyz)).astype(int)
    if not ((c >= 1).all() and (c <= np.array(box.shape) - 2).all()):
        return None
    return float(lp.to_hu(np.asarray(box[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, c[2] - 1:c[2] + 2])).mean())


def _dense_share(frame, uids):
    """Share of the rows (seriesuid, coordX/Y/Z) in the given scans whose centre is above -600 HU."""
    hits, total = 0, 0
    for u in uids:
        d = frame[frame.seriesuid == u]
        if d.empty:
            continue
        meta = lp.load_meta(u)
        box = np.load(lp.paths_for(lp.out_dir(), u)[0], mmap_mode="r")
        for r in d.itertuples():
            total += 1
            hu = _centre_hu(box, meta, [r.coordX, r.coordY, r.coordZ])
            hits += hu is not None and hu > DENSE_HU
    return hits, total


def _conventions():
    return [{"base": b, "z": z, "flip_x": fx, "flip_y": fy}
            for b, z, fx, fy in itertools.product((0, 1), ("asc", "desc"), (False, True), (False, True))]


def _solid_share(conv=None, sample=None):
    """(hits, total): the solid TL and Mass marks (a reader's texture 'Solid') of the prepared scans, or of the
    first `sample` of them, whose centre lies above DENSE_HU under `conv`. Solid marks only: NLST's screening
    scans hold many ground-glass and part-solid nodules whose centre is rightly fainter than -600 HU, so all
    nodules together reach only about 73% even with the right coordinates."""
    keep = {u: f for u, _, f in scans() if boxed(u)}
    a = _csv("annotations")
    a = a[a.series_uid.isin(keep) & a.finding_type.isin(["TL", "Mass"]) & a.location_z.notna()
          & (a.tl_texture == "Solid")]
    uids = sorted(a.series_uid.unique())
    hits = total = 0
    for u in uids[:sample] if sample else uids:
        d = a[a.series_uid == u]
        x, y, z = to_world(geometry(u, keep[u]), d.location_x, d.location_y, d.location_z, conv)
        h, t = _dense_share(pd.DataFrame({"seriesuid": u, "coordX": x, "coordY": y, "coordZ": z}), [u])
        hits, total = hits + h, total + t
    return hits, total


def _which_convention(sample=40):
    """[(share, convention)] best first, from the solid marks of up to `sample` prepared scans."""
    res = []
    for conv in _conventions():
        h, t = _solid_share(conv, sample)
        res.append((h / max(t, 1), conv))
    return sorted(res, key=lambda r: -r[0])


def check():
    import lidc
    keep, left_out = _select()
    assert keep, "no usable NLST scan is downloaded yet: run fetch_tcia.py lidc-annot-nlst501"
    print("scans used: %d (complete downloads, 2.5 mm slices or thinner, not corrected)" % len(keep))
    for reason, n in sorted(left_out.items()):
        print("  left out, %s: %d" % (reason, n))
    uids = [u for u, _, _ in keep]
    both = set(uids) & (set(lp.luna16_subsets()) | set(lidc.series_records()))
    assert not both, "an NLST series is a LUNA16 or LIDC one: %s" % sorted(both)[:3]
    print("no NLST series is a LUNA16 or LIDC series")

    les = lesions()
    ann, exc = build(uids)
    print("nodules (TL or Mass, at least %d readers, %.0f mm or more): %d in %d scans; excluded findings: %d in %d scans"
          % (MIN_READERS, MIN_DIAMETER_MM, len(ann), ann.seriesuid.nunique(), len(exc), exc.seriesuid.nunique()))
    nod = les[les.nodule]
    print("readers per nodule: " + ", ".join("%d: %d" % (k, v) for k, v in sorted(nod.n_readers.value_counts().items())))
    assert len(ann) and nod.n_readers.min() >= MIN_READERS

    have = [u for u in uids if boxed(u)]
    assert have, "no NLST box is prepared: run `nlst501.py prepare`"
    hits, total = _dense_share(ann, have)
    print("nodule centres above %d HU in the prepared box: %d of %d (%.1f%%; ground-glass ones are rightly fainter),"
          " %d scans boxed of %d" % (DENSE_HU, hits, total, 100 * hits / max(total, 1), len(have), len(uids)))
    hits, total = _solid_share()
    share = hits / max(total, 1)
    print("solid marks' centres above %d HU: %d of %d (%.1f%%; the coordinates' test, at least %.0f%%)"
          % (DENSE_HU, hits, total, 100 * share, 100 * MIN_DENSE))
    if share < MIN_DENSE:
        best = _which_convention()
        print("the coordinates are wrong under CONVENTION %s; centres on tissue per convention (best first):" % CONVENTION)
        for s, conv in best[:5]:
            print("  %.1f%%  %s" % (100 * s, conv))
        raise AssertionError("only %.1f%% of solid marks' centres above %d HU: set nlst501.CONVENTION to %s"
                             % (100 * share, DENSE_HU, best[0][1]))

    lab = labels()
    print("labels: %d nodules: label 0 (non-cancer patient) %d, label 1 (most suspicious nodule of a cancer patient,"
          " mean %g or more) %d, left out %d"
          % (len(lab), int((lab.label == 0).sum()), MIN_SUSPICION, int((lab.label == 1).sum()), int(lab.label.isna().sum())))
    assert len(lab) == len(nod)
    assert lab[lab.nlst_diagnosis == "cancer"].groupby("patient").label.sum().max() <= 1, "two label-1 nodules in a patient"
    print("check ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("prepare")
    b.add_argument("--jobs", type=int)
    sub.add_parser("check")
    sub.add_parser("fixcsv")
    a = ap.parse_args()
    if a.cmd == "fixcsv":
        return fixcsv()
    if a.cmd == "prepare":
        prepare(a.jobs)
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
