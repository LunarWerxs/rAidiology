"""The mammography reader, v1, on CMMD (docs/efficiency/README.md, "Mammography: the first reader, on CMMD", idea 27).

CMMD (CC BY 4.0, LICENSING.md): 5,202 DICOM mammograms of 1,775 patients; CMMD_clinicaldata_revision.xlsx has one
row per breast with a lesion: ID1 (patient), LeftRight (L or R) and classification (Benign or Malignant).

Where each fact comes from (read 2026-10-05): the patient is the DICOM PatientID (the folder D1-xxxx agrees; a mismatch
is counted by `gate`); the side is the DICOM ImageLaterality; the view is read from PatientOrientation, because these
files carry no ViewPosition: ['P', 'L'] is a CC (the second axis runs lateral), ['P', 'FL'] or ['P', 'FR'] is an MLO
(foot, oblique). The label is joined by (patient, side).

    python mammo.py gate                                  the join, the counts, five folds, cmmd_index.csv
    python mammo.py prepare                               every picture once into a uint8 memmap, 1024 x 640
    python mammo.py train --version N [--folds 0,1] [--epochs E] [--set key=<json> ...]
    python mammo.py eval --version N                      the verdicts (against 0.848 and 0.885)
    python mammo.py outlines                              TOMPEI-CMMD outlines -> 64 x 40 masks beside the pixels (processor)
    python mammo.py prepare-extra [--only cbis,cesm]      extra training mammograms, prepared as CMMD's are (processor)
    python mammo.py compare --base 1,11 --new 2,12 --step N   idea 33's paired verdict of two-seed versions
    python mammo.py read --versions 1,2,3 [--against 5,15]   idea 65: nets read together (mean probability)
    python mammo.py prepare-lesions [--limit N]   idea 66: 512 x 512 full-detail crops around TOMPEI's outlined lesions (processor)
    python mammo.py locate --version 6 [--limit N] [--places K] [--apart C]   idea 67: v6's outline head finds the lesion;
                                                a full-detail crop is cut there (K > 1: its K best cells, C cells apart)
    python mammo.py score-crops --version 21 --as 22 [--places K]   idea 67: v21's fold nets read the found crops, as
                                                version 22 (K > 1: a picture's logit is its largest over the K crops)
    python mammo.py found-masks --places K                 idea 95: masks for the found crops, for aux training
    python mammo.py selftest                              CPU, seconds, synthetic data

Idea 33 settings (--set key=<json>; each is absent from the settings, so from v1's fingerprint, when off):
    seed_offset=k       the net's start, the augmentation draws and the batch order use the seeds shifted by 7919 * k;
                        the folds and the early-stop patients do not change
    aux="tompei"        a training-only head on the stride-16 features draws the lesion outline (64 x 40) on training
                        pictures that have one; aux_weight (default 0.5) is its loss weight; the read never uses it
    extra=["cbis","cesm"]   extra mammograms (prepare-extra) join every fold's training rows only

Idea 42 step 2 setting:
    init="chest-v103"   the trunk starts from NIH chest reader v103 fold 0 (ours, never trained on any mammogram); the heads
                        stay fresh; the fingerprint names the file, its size and its mtime

    lesions=true   idea 66: train on the lesion crops (prepare-lesions) in place of the whole pictures; not with aux or extra

Idea 81 settings (each absent from the settings when off):
    found=K        train on the found crops (`locate --places K`) of ALL pictures in place of the whole pictures, with the picture's
                   own label, multiple-instance style: a sample is a picture, its K crops go through the net, the picture's logit is
                   the MAX over its K crop logits, BCE on that; pictures per batch = batch // K (at least 1); crops are augmented one
                   by one as the lesions path's are; folds, patients and labels come from the per-picture table, joined to the found
                   csv's place-0 rows (exact); fold%d.csv has one row per picture; not with lesions, aux or extra
    init_version=V fold k's net starts from version V's fold%d.pt (all weights, strict) in place of scratch or init; refused unless
                   V's fold k csv holds exactly the patients with fold == k here (so the held-out patients were never trained on);
                   the fingerprint names V's checkpoint files, sizes and mtimes

Net: chest_xray/eval/reader.py's ChestNet trunk from scratch unless init is set (no pretrained weights of any kind, LICENSING.md), one
logit (malignant), BCE, bfloat16. The pictures are mirrored so the chest wall is on the left, so augmentation never
flips horizontally; a vertical flip is drawn. Outputs: <WORK>/eval/mammo/.
"""
import argparse
import glob
import json
import math
import os
import queue
import re
import sys
import threading
import time
import zipfile
import xml.etree.ElementTree as ET

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402
from core.rows import FileRows  # noqa: E402

CMMD = os.path.join(paths.BESIDE, "Datasets", "public", "cmmd")
XLSX = os.path.join(CMMD, "files", "CMMD_clinicaldata_revision.xlsx")
OUT = os.path.join(paths.WORK, "eval", "mammo")
INDEX = os.path.join(OUT, "cmmd_index.csv")
PIX = os.path.join(OUT, "cmmd_1024x640.u8")
DONE = os.path.join(OUT, "cmmd_1024x640.done")
H, W = 1024, 640
FOLDS = 5
SEED = 20261005
VAL_SHARE = 0.10
MIN_LABELLED = 3700   # 1,872 labelled breasts x 2 views; the other 1,458 pictures show unlabelled breasts (bar amended 2026-10-05)
BARS = (0.848, 0.885)
BOOT, BOOT_SEED = 2000, 0
# the card share: v1's five folds peaked at 1.31 to 1.39 GB allocated (batch 12 at 1024 x 640) under 3.0 GB, so 2.0
# (2026-10-05 08:30; docs/efficiency, idea 31's development, "The lease was 7 times the need")
LEASE_GB = 2.0
PROCS = 6
SETTINGS = {
    "epochs": 25, "batch": 12, "lr": 3e-4, "weight_decay": 1e-4,
    "rotate_deg": 8.0, "scale": 0.10, "translate": 0.05, "brightness": 0.10, "contrast": 0.10,
    "widths": None, "dropout": None,    # None: reader.CFG's
}
IDEA33 = ("seed_offset", "aux", "aux_weight", "extra")   # settings left out of o when off (v1's fingerprint stays)
IDEA42 = ("init",)                                       # likewise: out of o when off
IDEA56 = ("pool",)                                       # likewise: out of o when off
IDEA66 = ("lesions",)                                    # likewise: out of o when off
IDEA81 = ("found", "init_version")                       # likewise: out of o when off
SEED_STEP = 7919
EXTRAS = ("cbis", "cesm")
AUX_WEIGHT = 0.5
GH, GW = H // 16, W // 16     # the stride-16 grid: 64 x 40
TOMPEI = os.path.join(paths.BESIDE, "Datasets", "public", "tompei-cmmd", "files", "TOMPEI-CMMD_v01_20250123")
OUTL = os.path.join(OUT, "cmmd_outlines_64x40.u8")
OUTH = os.path.join(OUT, "cmmd_outlines.has")
LS = 512
LES = os.path.join(OUT, "cmmd_lesions_512.u8")
LESCSV = os.path.join(OUT, "cmmd_lesions.csv")
CBIS_DIR = os.path.join(paths.BESIDE, "Datasets", "public", "cbis-ddsm")
CESM_DIR = os.path.join(paths.BESIDE, "Datasets", "public", "cdd-cesm")
CESM_JPG = os.path.join(CESM_DIR, "aspera", "Low Energy Images _0.64GB_ and Subtracted Images _0.82 GB_", "PKG - CDD-CESM",
                        "CDD-CESM", "Low energy images of CDD-CESM")
CESM_XLSX = os.path.join(CESM_DIR, "files", "Radiology-manual-annotations.xlsx")
CESM_COL = "Pathology Classification/ Follow up"
STEP_GAIN = 0.01


# ---------------------------------------------------------------- labels, views, folds

def read_xlsx(path):
    """The first sheet as a list of {header: value} dicts, with the standard library only (the venv has no openpyxl)."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    strings = [("".join(t.text or "" for t in si.iter(ns + "t"))) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        d = {}
        for c in r:
            v = c.find(ns + "v")
            if v is not None:
                d[re.match("[A-Z]+", c.get("r")).group()] = strings[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(d)
    head = rows[0]
    return [{head[k]: v for k, v in r.items() if k in head} for r in rows[1:]]


def breast_labels(rows):
    """{(patient, side): 1 malignant / 0 benign} from the clinical rows; a row with another classification is left out."""
    out = {}
    for r in rows:
        c = str(r.get("classification", "")).strip().lower()
        side = str(r.get("LeftRight", "")).strip().upper()[:1]
        if c in ("malignant", "benign") and side in ("L", "R"):
            out[(str(r["ID1"]).strip(), side)] = int(c == "malignant")
    return out


def view_of(orientation):
    """'MLO' or 'CC' from PatientOrientation (a list like ['P', 'FL']): the second axis names foot or head for an oblique view."""
    if orientation is None or len(orientation) < 2:
        return ""
    return "MLO" if any(ch in str(orientation[1]).upper() for ch in "FH") else "CC"


def join_labels(pics, labels):
    """pics: dicts with patient, side, view. Returns (label per picture or -1, reason per picture or '')."""
    lab, why = [], []
    patients = {p for p, _ in labels}
    for p in pics:
        if not p["patient"]:
            lab.append(-1), why.append("no PatientID")
        elif p["side"] not in ("L", "R"):
            lab.append(-1), why.append("no ImageLaterality")
        elif p["patient"] not in patients:
            lab.append(-1), why.append("patient not in the clinical table")
        elif (p["patient"], p["side"]) not in labels:
            lab.append(-1), why.append("that side has no row in the clinical table")
        else:
            lab.append(labels[(p["patient"], p["side"])]), why.append("")
    return lab, why


def assign_folds(patient_label, k=FOLDS, seed=SEED):
    """{patient: fold}: by patient, seeded, stratified by the patient's label (1 when any breast is malignant)."""
    rng = np.random.default_rng(seed)
    out, n = {}, 0
    for cls in (1, 0):
        ps = sorted(p for p, y in patient_label.items() if y == cls)
        for i in rng.permutation(len(ps)):
            out[ps[i]] = n % k
            n += 1
    return out


def patient_labels(patients, labels):
    """{patient: 1 when any labelled breast is malignant}, over the patients given."""
    out = {}
    for p in set(patients):
        out[p] = int(any(labels.get((p, s)) == 1 for s in ("L", "R")))
    return out


# ---------------------------------------------------------------- gate

def scan_dicoms():
    import pydicom
    files = sorted(glob.glob(os.path.join(CMMD, "**", "*.dcm"), recursive=True))
    pics = []
    for f in files:
        d = pydicom.dcmread(f, stop_before_pixels=True)
        rel = os.path.relpath(f, CMMD).replace(os.sep, "/")
        side = str(d.get("ImageLaterality", "") or d.get("Laterality", "")).strip().upper()[:1]
        pics.append({"path": rel, "patient": str(d.get("PatientID", "")).strip(), "side": side,
                     "view": view_of(d.get("PatientOrientation")), "folder": rel.split("/")[0]})
    return pics


def gate():
    import pandas as pd
    labels = breast_labels(read_xlsx(XLSX))
    pics = scan_dicoms()
    lab, why = join_labels(pics, labels)
    df = pd.DataFrame(pics)
    df["label"], df["why"] = lab, why
    ok = df[df.label >= 0].copy()
    pl = patient_labels(ok.patient, labels)
    fold = assign_folds(pl)
    ok["fold"] = ok.patient.map(fold)
    os.makedirs(OUT, exist_ok=True)
    ok[["path", "patient", "side", "view", "label", "fold"]].to_csv(INDEX, index=False)
    print("pictures %d, labelled %d, breasts %d, patients %d" % (len(df), len(ok), len(ok.groupby(["patient", "side"])),
                                                                ok.patient.nunique()))
    print("malignant share: %.1f%% of pictures, %.1f%% of breasts, %.1f%% of patients"
          % (100 * ok.label.mean(), 100 * ok.drop_duplicates(["patient", "side"]).label.mean(), 100 * np.mean(list(pl.values()))))
    print("views: CC %d, MLO %d, unknown %d" % tuple((ok.view == v).sum() for v in ("CC", "MLO", "")))
    print("folds (pictures): %s" % ", ".join("%d: %d" % (k, (ok.fold == k).sum()) for k in range(FOLDS)))
    bad = df[df.why != ""]
    print("unlabelled pictures: %d" % len(bad))
    for r, n in bad.why.value_counts().items():
        print("  %5d  %s" % (n, r))
    print("PatientID differs from its folder: %d pictures" % int((df.patient != df.folder).sum()))
    twice = int((ok.groupby("patient").fold.nunique() > 1).sum())
    print("patients in two folds: %d" % twice)
    views = ok.groupby(["patient", "side"]).view.apply(lambda v: tuple(sorted(v)))
    odd = int((views != ("CC", "MLO")).sum())
    print("labelled breasts without exactly one CC and one MLO: %d" % odd)
    if len(ok) >= MIN_LABELLED and twice == 0 and odd == 0:
        print("GATE PASSED")
        return 0
    print("GATE FAILED")
    return 1


# ---------------------------------------------------------------- the picture

def window_to_u8(ds):
    """The DICOM's pixels as uint8: its VOI LUT when it has a window, else min-max; MONOCHROME1 inverted."""
    arr = ds.pixel_array
    if "WindowCenter" in ds:
        try:
            from pydicom.pixels import apply_voi_lut
            arr = apply_voi_lut(arr, ds)
        except Exception:  # noqa: BLE001  a window pydicom cannot apply falls back to min-max
            pass
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = 255.0 - arr
    return np.clip(arr + 0.5, 0, 255).astype(np.uint8)


def breast_crop(a, out_h=H, out_w=W, transform=False):
    """uint8 (h, w) -> uint8 (out_h, out_w): the largest bright connected component (a threshold, Otsu's on a quarter-size copy,
    never below 10), everything outside it set black, cropped to its box, mirrored so the chest wall is on the left (the half of
    the box holding more of the breast is the chest-wall half), resized with the aspect kept and padded black at the right
    and bottom. With transform=True it returns (picture, t): t maps an original pixel (x, y) to the picture, see map_points."""
    import cv2
    h, w = a.shape
    y0 = x0 = 0
    small = cv2.resize(a, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (small > max(thr, 10)).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        crop = a
        keep = np.ones(a.shape, bool)
    else:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        full = cv2.resize((comp == big).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        full = cv2.dilate(full, np.ones((9, 9), np.uint8))
        ys, xs = np.nonzero(full)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = (a * full)[y0:y1, x0:x1]
        keep = full[y0:y1, x0:x1] > 0
    mid = crop.shape[1] // 2
    mirror = bool(keep[:, mid:].sum() > keep[:, :mid].sum())
    if mirror:
        crop = crop[:, ::-1]
    ch, cw = crop.shape
    s = min(out_h / ch, out_w / cw)
    nh, nw = max(1, min(out_h, int(round(ch * s)))), max(1, min(out_w, int(round(cw * s))))
    res = cv2.resize(np.ascontiguousarray(crop), (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((out_h, out_w), np.uint8)
    out[:nh, :nw] = res
    if transform:
        return out, {"y0": int(y0), "x0": int(x0), "mirror": mirror, "cw": int(cw), "s": float(s)}
    return out


def map_points(pts, t):
    """(k, 2) original (x, y) -> where those pixels land in breast_crop's output: shift by the box, mirror within its width,
    scale (pixel centres)."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    cx, cy = p[:, 0] - t["x0"], p[:, 1] - t["y0"]
    if t["mirror"]:
        cx = (t["cw"] - 1) - cx
    return np.stack([(cx + 0.5) * t["s"] - 0.5, (cy + 0.5) * t["s"] - 0.5], axis=1)


def unmap_point(px, py, t):
    """The exact inverse of map_points for one point: a place in breast_crop's output -> the original (x, y), floats."""
    xr = (px + 0.5) / t["s"] - 0.5
    yr = (py + 0.5) / t["s"] - 0.5
    if t["mirror"]:
        xr = (t["cw"] - 1) - xr
    return float(xr + t["x0"]), float(yr + t["y0"])


def rasterise(polys, h=H, w=W):
    """uint8 (h, w) of 0/1: every polygon (a (k, 2) array of x, y in the output frame) filled."""
    import cv2
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(p)).astype(np.int32)], 1)
    return m


def to_grid(m, gh=GH, gw=GW):
    """A 0/1 mask at 1024 x 640 -> uint8 gh x gw of its area share x 255 (area averaging)."""
    import cv2
    g = cv2.resize(m.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)


def prep_one(job):
    import pydicom
    i, path = job
    return i, breast_crop(window_to_u8(pydicom.dcmread(os.path.join(CMMD, path))))


def load_index():
    import pandas as pd
    if not os.path.exists(INDEX):
        sys.exit("no %s: run `gate` first" % INDEX)
    return pd.read_csv(INDEX, keep_default_na=False)


def pixels(mode="r"):
    n = len(load_index())
    if mode == "r":                       # read by file reads, not a memory map (idea 53); a writer still gets a memmap
        return FileRows(PIX, (n, H, W))
    return np.memmap(PIX, dtype=np.uint8, mode=mode, shape=(n, H, W))


def prepare():
    import multiprocessing as mp
    df = load_index()
    n = len(df)
    fresh = not (os.path.exists(PIX) and os.path.getsize(PIX) == n * H * W and os.path.exists(DONE)
                 and os.path.getsize(DONE) == n)
    pix = np.memmap(PIX, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n, H, W))
    done = np.memmap(DONE, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n,))
    todo = [(i, p) for i, p in enumerate(df.path) if not done[i]]
    print("prepare: %d pictures, %d to do" % (n, len(todo)), flush=True)
    t0 = time.time()
    with mp.Pool(min(PROCS, mp.cpu_count())) as pool:
        for c, (i, a) in enumerate(pool.imap_unordered(prep_one, todo, chunksize=4), 1):
            pix[i] = a
            done[i] = 1
            if c % 200 == 0:
                pix.flush(), done.flush()
                print("  %d / %d, %.0f s" % (c, len(todo), time.time() - t0), flush=True)
    pix.flush(), done.flush()
    print("prepare: done, %d of %d pictures" % (int(done.sum()), n))
    return 0 if int(done.sum()) == n else 1


# ---------------------------------------------------------------- outlines (TOMPEI-CMMD)

def outline_files(folder=TOMPEI):
    """{(patient, side): path} of the MLO annotation files ('._' files and anything else are skipped)."""
    out = {}
    for f in sorted(os.listdir(folder)):
        m = re.match(r"^(.+)_MLO_([LR])_AnnotationFile\.json$", f)
        if m and not f.startswith("._"):
            out[(m.group(1), m.group(2))] = os.path.join(folder, f)
    return out


def read_outlines(path):
    """([(k, 2) float arrays of x, y], [labels]) from one annotation file."""
    items = json.load(open(path, encoding="utf-8"))
    polys, labs = [], []
    for it in items:
        pts = [(float(p["x"]), float(p["y"])) for p in it.get("cgPoints", [])]
        if len(pts) >= 3:
            polys.append(np.array(pts)), labs.append(str(it.get("label", "")).strip().lower())
    return polys, labs


def outline_one(job):
    """(row, 64 x 40 mask, outline pixels, of them on picture pixels, identical to the prepared picture) for one matched picture."""
    import pydicom
    i, path, polys = job
    out, t = breast_crop(window_to_u8(pydicom.dcmread(os.path.join(CMMD, path))), transform=True)
    m = rasterise([map_points(p, t) for p in polys])
    on = int(((m > 0) & (out > 0)).sum())
    same = os.path.exists(PIX) and bool(np.array_equal(out, np.fromfile(PIX, np.uint8, count=H * W, offset=i * H * W).reshape(H, W)))
    return i, to_grid(m), int((m > 0).sum()), on, same


def outlines():
    import multiprocessing as mp
    df = load_index()
    n = len(df)
    at = {(p, s): i for i, (p, s, v) in enumerate(zip(df.patient, df.side, df.view)) if v == "MLO"}
    files = outline_files()
    jobs, miss, labels = [], 0, {}
    for key, f in files.items():
        if key not in at:
            miss += 1
            continue
        polys, labs = read_outlines(f)
        for lb in labs:
            labels[lb] = labels.get(lb, 0) + 1
        if polys:
            jobs.append((at[key], df.path[at[key]], polys))
    print("outlines: %d annotation files, %d matched to a CMMD MLO picture, %d without a picture in the index (misses)"
          % (len(files), len(files) - miss, miss))
    print("outline labels: %s" % ", ".join("%s %d" % kv for kv in sorted(labels.items())))
    os.makedirs(OUT, exist_ok=True)
    grid = np.memmap(OUTL, dtype=np.uint8, mode="w+", shape=(n, GH, GW))
    has = np.memmap(OUTH, dtype=np.uint8, mode="w+", shape=(n,))
    t0, tot, on, same = time.time(), 0, 0, 0
    with mp.Pool(min(PROCS, mp.cpu_count())) as pool:
        for c, (i, g, a, b, e) in enumerate(pool.imap_unordered(outline_one, jobs, chunksize=4), 1):
            grid[i], has[i] = g, 1
            tot, on, same = tot + a, on + b, same + int(e)
            if c % 200 == 0:
                print("  %d / %d, %.0f s" % (c, len(jobs), time.time() - t0), flush=True)
    grid.flush(), has.flush()
    print("outlines: %d pictures with an outline -> %s" % (int(has.sum()), OUTL))
    print("outline pixels on non-black picture pixels: %.4f (%d of %d)" % (on / max(tot, 1), on, tot))
    print("re-cropped picture identical to the prepared one: %d of %d" % (same, len(jobs)))
    return 0 if len(jobs) else 1


def outline_table(n):
    """(masks FileRows (n, 64, 40) uint8, has (n,) bool) built by `outlines`."""
    if not (os.path.exists(OUTL) and os.path.exists(OUTH)):
        sys.exit("no outline masks: run `outlines` first")
    has = np.array(np.memmap(OUTH, dtype=np.uint8, mode="r", shape=(n,))) > 0
    return FileRows(OUTL, (n, GH, GW)), has


# ---------------------------------------------------------------- lesion crops (idea 66)

def lesion_crop(a, polys, mirror):
    """(512 x 512 uint8 crop of a around the largest outline's box centre, cx, cy, ow, oh); black past a's edge."""
    def area(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    p = max(polys, key=area)
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    cx, cy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
    return crop_at(a, cx, cy, mirror), cx, cy, int(round(x1 - x0)), int(round(y1 - y0))


def crop_at(a, cx, cy, mirror):
    """512 x 512 uint8 window of a, rows cy-256..cy+255 and columns cx-256..cx+255, black past a's edge, mirrored when `mirror`."""
    crop = np.zeros((LS, LS), np.uint8)
    r0, c0 = cy - LS // 2, cx - LS // 2
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r0 + LS, a.shape[0]), min(c0 + LS, a.shape[1])
    if sr1 > sr0 and sc1 > sc0:
        crop[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = a[sr0:sr1, sc0:sc1]
    if mirror:
        crop = crop[:, ::-1]
    return np.ascontiguousarray(crop)


def lesion_one(job):
    import pydicom
    i, path, polys = job
    a = window_to_u8(pydicom.dcmread(os.path.join(CMMD, path)))
    mirror = bool(breast_crop(a, transform=True)[1]["mirror"])
    return (i,) + lesion_crop(a, polys, mirror)


def prepare_lesions(limit=None):
    import multiprocessing as mp
    import pandas as pd
    df = load_index()
    at = {(p, s): i for i, (p, s, v) in enumerate(zip(df.patient, df.side, df.view)) if v == "MLO"}
    jobs = []
    for key, f in outline_files().items():
        if key not in at:
            continue
        polys, _ = read_outlines(f)
        if polys:
            jobs.append((at[key], df.path[at[key]], polys))
    jobs.sort(key=lambda j: j[0])
    if limit is not None:
        jobs = jobs[:limit]
    m = len(jobs)
    print("prepare-lesions: %d matched" % m, flush=True)
    os.makedirs(OUT, exist_ok=True)
    grid = np.memmap(LES, dtype=np.uint8, mode="w+", shape=(max(m, 1), LS, LS))
    rows = []
    t0 = time.time()
    with mp.Pool(min(PROCS, mp.cpu_count())) as pool:
        for c, (i, crop, cx, cy, ow, oh) in enumerate(pool.imap(lesion_one, jobs, chunksize=4)):
            grid[c] = crop
            r = df.iloc[i]
            rows.append({"row": i, "path": r["path"], "patient": r["patient"], "side": r["side"], "view": r["view"],
                         "label": r["label"], "fold": r["fold"], "cx": cx, "cy": cy, "ow": ow, "oh": oh,
                         "outlines": len(jobs[c][2])})
            if (c + 1) % 200 == 0:
                print("  %d / %d, %.0f s" % (c + 1, m, time.time() - t0), flush=True)
    grid.flush()
    out = pd.DataFrame(rows, columns=["row", "path", "patient", "side", "view", "label", "fold", "cx", "cy", "ow", "oh", "outlines"])
    out.to_csv(LESCSV, index=False)
    print("prepare-lesions: matched %d, written %d, cancers %d, benign %d, %.0f s -> %s"
          % (m, len(out), int((out.label == 1).sum()), int((out.label == 0).sum()), time.time() - t0, LES))
    return 0 if m else 1


# ---------------------------------------------------------------- the reader finds the lesion, then looks again (idea 67)

FOUND = os.path.join(OUT, "cmmd_found_512.u8")
FOUNDCSV = os.path.join(OUT, "cmmd_found.csv")


def found_masks_paths(places):
    """(masks file, has file) for `places` places per picture."""
    return (os.path.join(OUT, "cmmd_found_masks_k%d.u8" % places), os.path.join(OUT, "cmmd_found_masks_k%d.has" % places))


def load_fold_net(version, k, dev):
    """(net on dev in eval mode, mean, sd) of version's fold k, built from the settings its checkpoint carries."""
    import torch
    ck = torch.load(os.path.join(out_dir(version), "fold%d.pt" % k), map_location="cpu", weights_only=False)
    o = ck["settings"]
    reader = reader_module()
    widths = o.get("widths") or reader.CFG["widths"]
    dropout = reader.CFG["dropout"] if o.get("dropout") is None else o["dropout"]
    net = make_net(reader, widths, dropout, bool(o.get("aux")), pool=o.get("pool"))
    net.load_state_dict(ck["net"])
    return net.to(dev).eval(), ck["mean"], ck["sd"]


def found_one(job):
    """(crop, cx, cy, dist_mm) for one picture: the crop is cut where the outline head's peak cell lies in the DICOM."""
    import pydicom
    i, path, gy, gx, polys = job
    ds = pydicom.dcmread(os.path.join(CMMD, path))
    a = window_to_u8(ds)
    _, t = breast_crop(a, transform=True)
    x, y = unmap_point(16 * gx + 7.5, 16 * gy + 7.5, t)
    cx, cy = int(round(x)), int(round(y))
    dist = float("nan")
    if polys:
        def area(p):
            u, v = p[:, 0], p[:, 1]
            return 0.5 * abs(float(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))
        p = max(polys, key=area)
        ox, oy = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
        sp = ds.get("ImagerPixelSpacing")
        dist = math.hypot(cx - ox, cy - oy) * (float(sp[0]) if sp else 0.094090909)
    return i, crop_at(a, cx, cy, t["mirror"]), cx, cy, dist


def found_paths(places):
    """(crops file, csv file) of locate: the single-place pair, or the pair for `places` places per picture."""
    if places == 1:
        return FOUND, FOUNDCSV
    return (os.path.join(OUT, "cmmd_found_512_k%d.u8" % places), os.path.join(OUT, "cmmd_found_k%d.csv" % places))


def pick_cells(m, places, apart):
    """[(gy, gx)] of a map's `places` cells, greedily: the highest, then the highest at least `apart` cells from all taken."""
    taken = [np.unravel_index(int(np.argmax(m)), m.shape)]
    gy, gx = np.indices(m.shape)
    while len(taken) < places:
        ok = np.ones(m.shape, bool)
        for ty, tx in taken:
            ok &= np.hypot(gy - ty, gx - tx) >= apart
        if not ok.any():
            taken.append(taken[-1])
            continue
        taken.append(np.unravel_index(int(np.argmax(np.where(ok, m, -np.inf))), m.shape))
    return taken


def locate(version, limit=None, places=1, apart=6):
    import multiprocessing as mp
    import pandas as pd
    import torch
    from core import ledger
    limit_card_memory(torch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_index()
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    n = len(df)
    pix = pixels()
    found, foundcsv = found_paths(places)
    gys, gxs, peaks = np.zeros((n, places), np.int64), np.zeros((n, places), np.int64), np.zeros((n, places), np.float64)
    t0 = time.time()
    for k in range(FOLDS):
        rows = np.nonzero(df.fold.astype(int).to_numpy() == k)[0]
        if not len(rows):
            continue
        if not json_aux(version, k):
            sys.exit("v%d fold %d has no outline head (aux is off): idea 67 needs a net trained with aux" % (version, k))
        net, mean, sd = load_fold_net(version, k, dev)
        with torch.no_grad():       # full precision: the peak cell and logit are then the same at any batch size
            for s in range(0, len(rows), 8):
                r = rows[s:s + 8]
                x = ((torch.from_numpy(np.ascontiguousarray(pix[r]))[:, None].to(dev).float() / 255.0) - mean) / sd
                m = net.forward_both(x)[1].float().cpu().numpy()[:, 0]
                for j, ri in enumerate(r):
                    for p, (gy, gx) in enumerate(pick_cells(m[j], places, apart)):
                        gys[ri, p], gxs[ri, p], peaks[ri, p] = gy, gx, m[j][gy, gx]
        del net
        print("locate: fold %d, %d pictures, %.0f s" % (k, len(rows), time.time() - t0), flush=True)
    files = outline_files()
    jobs = []
    for i, (p, s, v, path) in enumerate(zip(df.patient, df.side, df.view, df.path)):
        polys = None
        if v == "MLO" and (p, s) in files:
            polys = read_outlines(files[(p, s)])[0] or None
        for q in range(places):       # job index i * places + q is picture i's place q
            jobs.append((i * places + q, path, int(gys[i, q]), int(gxs[i, q]), polys))
    os.makedirs(OUT, exist_ok=True)
    grid = np.memmap(found, dtype=np.uint8, mode="w+", shape=(max(n * places, 1), LS, LS))
    cxs, cys, dist = np.zeros(n * places, np.int64), np.zeros(n * places, np.int64), np.full(n * places, np.nan)
    with mp.Pool(min(PROCS, mp.cpu_count())) as pool:
        for c, (i, crop, cx, cy, d) in enumerate(pool.imap(found_one, jobs, chunksize=4), 1):
            grid[i], cxs[i], cys[i], dist[i] = crop, cx, cy, d
            if c % 200 == 0:
                print("  %d / %d, %.0f s" % (c, len(jobs), time.time() - t0), flush=True)
    grid.flush()
    if places == 1:
        out = pd.DataFrame({"row": np.arange(n), "path": df.path, "patient": df.patient, "side": df.side, "view": df.view,
                            "label": df.label, "fold": df.fold, "gy": gys[:, 0], "gx": gxs[:, 0], "peak": peaks[:, 0],
                            "cx": cxs, "cy": cys, "dist_mm": dist})
        nearest = dist
    else:
        rep = np.repeat(np.arange(n), places)
        out = pd.DataFrame({"row": rep, "place": np.tile(np.arange(places), n), "path": df.path.to_numpy()[rep],
                            "patient": df.patient.to_numpy()[rep], "side": df.side.to_numpy()[rep],
                            "view": df.view.to_numpy()[rep], "label": df.label.to_numpy()[rep],
                            "fold": df.fold.to_numpy()[rep], "gy": gys.reshape(-1), "gx": gxs.reshape(-1),
                            "peak": peaks.reshape(-1), "cx": cxs, "cy": cys, "dist_mm": dist})
        out = out[["row", "path", "patient", "side", "view", "label", "fold", "place", "gy", "gx", "peak", "cx", "cy",
                   "dist_mm"]]
        far = np.where(np.isnan(dist), np.inf, dist).reshape(n, places).min(axis=1)
        nearest = np.where(np.isinf(far), np.nan, far)      # each picture's nearest place
    out.to_csv(foundcsv, index=False)
    ok = ~np.isnan(nearest)
    med = float(np.median(nearest[ok])) if ok.any() else float("nan")
    within = float((nearest[ok] <= 10.0).mean()) if ok.any() else float("nan")
    with ledger.run("experiment", "mammo locate v%d (idea 67)" % version) as led:
        led.settings_from({"version": version, "limit": limit} if places == 1
                          else {"version": version, "limit": limit, "places": places, "apart": apart})
        print("locate: %d pictures, %d with an outline (MLO); median distance to the outline's centre %.1f mm; within 10 mm: %.3f; %.0f s -> %s"
              % (n, int(ok.sum()), med, within, time.time() - t0, found))
        led.number("pictures", n, "count").number("outlined", int(ok.sum()), "count")
        led.number("median_dist_mm", med).number("share_within_10mm", within)
    return 0


def crop_mask(polys, cx, cy, mirror):
    """uint8 (32, 32) of the outlines (DICOM x, y polygons) inside the 512 x 512 crop centred at cx, cy, as crop_at cuts it."""
    shifted = [np.asarray(p, np.float64) - np.array([cx - LS // 2, cy - LS // 2]) for p in polys]
    m = rasterise(shifted, LS, LS)
    if mirror:
        m = m[:, ::-1]
    return to_grid(np.ascontiguousarray(m), LS // 16, LS // 16)


def found_masks_one(job):
    """(first crop row, (K, 32, 32) masks) for one outlined picture; its DICOM is read once, only for the mirror flag."""
    import pydicom
    row0, path, centres, polys = job
    mirror = bool(breast_crop(window_to_u8(pydicom.dcmread(os.path.join(CMMD, path))), transform=True)[1]["mirror"])
    return row0, np.stack([crop_mask(polys, cx, cy, mirror) for cx, cy in centres])


def found_masks(places):
    import multiprocessing as mp
    import pandas as pd
    from core import ledger
    df = load_index()
    n = len(df)
    found, foundcsv = found_paths(places)
    masks_path, has_path = found_masks_paths(places)
    if not (os.path.exists(foundcsv) and os.path.exists(found)):
        sys.exit("no found crops for K=%d: run `locate --places %d` first" % (places, places))
    fc = pd.read_csv(foundcsv, keep_default_na=False)
    first = fc[fc.place.astype(int) == 0] if places > 1 else fc
    if len(first) != len(df) or list(first.path) != list(df.path) or len(fc) != n * places:
        sys.exit("the found csv (%d rows) does not match the picture table (%d x %d places): rerun locate --places %d"
                 % (len(fc), len(df), places, places))
    cxs, cys = fc.cx.astype(int).to_numpy(), fc.cy.astype(int).to_numpy()      # row i * places + q is picture i's place q
    files = outline_files()
    jobs = []
    has_outline = np.zeros(n, np.uint8)
    for i, (path, v, p, s) in enumerate(zip(df.path, df.view, df.patient, df.side)):
        if v == "MLO" and (p, s) in files:
            polys = read_outlines(files[(p, s)])[0]
            if polys:
                has_outline[i] = 1
                jobs.append((i * places, path, [(int(cxs[i * places + q]), int(cys[i * places + q])) for q in range(places)], polys))
    print("found-masks: %d outlined pictures, %d crops to draw" % (int(has_outline.sum()), len(jobs) * places), flush=True)
    os.makedirs(OUT, exist_ok=True)
    masks = np.memmap(masks_path, dtype=np.uint8, mode="w+", shape=(n * places, LS // 16, LS // 16))
    t0 = time.time()
    with mp.Pool(min(PROCS, mp.cpu_count())) as pool:
        for c, (row0, mk) in enumerate(pool.imap(found_masks_one, jobs, chunksize=4), 1):
            masks[row0:row0 + places] = mk
            if c % 200 == 0:
                print("  %d / %d, %.0f s" % (c, len(jobs), time.time() - t0), flush=True)
    masks.flush()
    has = np.memmap(has_path, dtype=np.uint8, mode="w+", shape=(n,))
    has[:] = has_outline
    has.flush()
    cell = (np.asarray(masks) >= 128).reshape(n, places, -1).any(axis=2)      # (pictures, places): the crop holds a cell >= 128
    pics_cell, crops_cell = int(cell.any(axis=1).sum()), int(cell.sum())
    secs = time.time() - t0
    with ledger.run("experiment", "mammo found-masks (idea 95)") as led:
        led.settings_from({"places": places})
        print("found-masks: %d pictures, %d outlined, %d outlined with a crop holding a cell >= 128, %d such crops, %.0f s -> %s"
              % (n, int(has_outline.sum()), pics_cell, crops_cell, secs, masks_path))
        led.number("pictures", n, "count").number("outlined", int(has_outline.sum()), "count")
        led.number("outlined_with_cell_crop", pics_cell, "count").number("cell_crops", crops_cell, "count")
        led.number("seconds", secs)
    return 0


def json_aux(version, k):
    """Whether version's fold k was trained with the outline head."""
    import torch
    ck = torch.load(os.path.join(out_dir(version), "fold%d.pt" % k), map_location="cpu", weights_only=False)
    return bool(ck["settings"].get("aux"))


def score_crops(version, as_version, places=1):
    import pandas as pd
    import torch
    from core import ledger
    limit_card_memory(torch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    found, foundcsv = found_paths(places)
    if not (os.path.exists(foundcsv) and os.path.exists(found)):
        sys.exit("no found crops: run `locate` first")
    df = pd.read_csv(foundcsv, keep_default_na=False)
    if places > 1:
        df = df[df.place == 0].reset_index(drop=True)       # one row per picture; each has `places` crops, row i * places + p
    n = len(df)
    crops = np.memmap(found, dtype=np.uint8, mode="r", shape=(n * places, LS, LS))
    logit = np.zeros(n * places, np.float64)
    for k in range(FOLDS):
        rows = np.nonzero(df.fold.astype(int).to_numpy() == k)[0]
        if not len(rows):
            continue
        rows = (rows[:, None] * places + np.arange(places)[None, :]).reshape(-1)
        net, mean, sd = load_fold_net(version, k, dev)
        with torch.no_grad():       # full precision, as in locate
            for s in range(0, len(rows), 24):
                r = rows[s:s + 24]
                x = ((torch.from_numpy(np.ascontiguousarray(crops[r]))[:, None].to(dev).float() / 255.0) - mean) / sd
                logit[r] = net(x).float().squeeze(1).cpu().numpy()
        del net
    res = df[["path", "patient", "side", "view", "label", "fold"]].copy()
    if places > 1:
        each = logit.reshape(n, places)
        logit = each.max(axis=1)
    res["logit"] = logit
    res["p"] = 1.0 / (1.0 + np.exp(-logit))
    if places > 1:
        for p in range(places):
            res["logit_%d" % p] = each[:, p]
    d = out_dir(as_version)
    os.makedirs(d, exist_ok=True)
    res.to_csv(os.path.join(d, "oof.csv"), index=False)
    with ledger.run("experiment", "mammo score-crops v%d as v%d (idea 67)" % (version, as_version)) as led:
        led.settings_from({"version": version, "as": as_version, "places": places})
        a = auroc(res.label, res.logit)
        print("score-crops: %d pictures, per-picture AUROC %.4f -> %s" % (n, a, os.path.join(d, "oof.csv")))
        for k in sorted(res.fold.unique()):
            f = res[res.fold == k]
            print("  fold %s: %d pictures, AUROC %.4f" % (k, len(f), auroc(f.label, f.logit)))
        led.number("picture_auroc", a).number("pictures", n, "count")
    return 0


# ---------------------------------------------------------------- extra mammograms

def minmax_u8(arr):
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def cbis_pictures(ix):
    """One picture per (patient, side, view) of cbis.py's index: train rows of patients with no test row; the Mass- file when the
    film is under both; the label is 1 when any lesion of that view has y == 1. [{path (under CBIS), patient, side, view, label}]."""
    test_p = set(ix[ix.split == "test"].patient)
    tr = ix[(ix.split == "train") & ~ix.patient.isin(test_p)]
    out = []
    for (p, s, v), g in tr.groupby(["patient", "side", "view"]):
        have = g[g.full != ""]
        if not (g.y >= 0).any() or not len(have):
            continue
        mass = have[have.type == "mass"]
        out.append({"path": (mass if len(mass) else have).iloc[0].full, "patient": p, "side": s, "view": v,
                    "label": int((g.y == 1).any())})
    return out


def cesm_pictures(rows):
    """[{path (under CESM_JPG), patient, side, view, label}] from the annotation rows: DM, Malignant 1 / Benign 0, Normal dropped."""
    out, seen = [], set()
    for r in rows:
        cls = str(r.get(CESM_COL, "")).strip().lower()
        name = str(r.get("Image_name", "")).strip()
        if str(r.get("Type", "")).strip() != "DM" or cls not in ("malignant", "benign") or not name or name in seen:
            continue
        seen.add(name)
        out.append({"path": name + ".jpg", "patient": "cesm-" + str(r.get("Patient_ID", "")).strip(),
                    "side": str(r.get("Side", "")).strip().upper()[:1], "view": str(r.get("View", "")).strip().upper(),
                    "label": int(cls == "malignant")})
    return out


def extra_paths(name):
    return (os.path.join(OUT, "extra_%s_index.csv" % name), os.path.join(OUT, "extra_%s_1024x640.u8" % name),
            os.path.join(OUT, "extra_%s_1024x640.done" % name))


def extra_one(job):
    """(row, uint8 1024 x 640 or None): read as cbis.read_pixels / cdd-cesm's JPG, min-max to uint8, breast_crop."""
    name, i, path = job
    try:
        if name == "cbis":
            import SimpleITK as sitk
            a = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(CBIS_DIR, path)))
            a = minmax_u8(a[0] if a.ndim == 3 else a)
        else:
            import cv2
            a = cv2.imread(os.path.join(CESM_JPG, path), cv2.IMREAD_GRAYSCALE)
            if a is None:
                return i, None
        return i, breast_crop(a)
    except Exception as e:  # noqa: BLE001  a bad file is counted, not fatal
        print("  %s row %d: %s" % (name, i, e), flush=True)
        return i, None


def cbis_module():
    """cbis.py does `import mammo` at its top, so it is imported here, inside functions only; its INITS and load_init are the warm start's."""
    sys.path.insert(0, HERE)
    import cbis
    return cbis


def extra_list(name):
    import pandas as pd
    if name == "cbis":
        cbis = cbis_module()
        ix = cbis.load_index() if os.path.exists(cbis.INDEX) else cbis.build_index()
        return pd.DataFrame(cbis_pictures(ix))
    return pd.DataFrame(cesm_pictures(read_xlsx(CESM_XLSX)))


def prepare_extra(only):
    from multiprocessing.pool import ThreadPool
    bad = 0
    for name in only:
        idx, pix_p, done_p = extra_paths(name)
        pics = extra_list(name)
        n = len(pics)
        os.makedirs(OUT, exist_ok=True)
        pics.to_csv(idx, index=False)
        fresh = not (os.path.exists(pix_p) and os.path.getsize(pix_p) == n * H * W and os.path.exists(done_p)
                     and os.path.getsize(done_p) == n)
        pix = np.memmap(pix_p, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n, H, W))
        done = np.memmap(done_p, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n,))
        todo = [(name, i, p) for i, p in enumerate(pics.path) if not done[i]]
        print("prepare-extra %s: %d pictures, %d to do" % (name, n, len(todo)), flush=True)
        t0 = time.time()
        with ThreadPool(4) as pool:
            for c, (i, a) in enumerate(pool.imap_unordered(extra_one, todo), 1):
                if a is not None:
                    pix[i] = a
                    done[i] = 1
                if c % 200 == 0:
                    pix.flush(), done.flush()
                    print("  %d / %d, %.0f s" % (c, len(todo), time.time() - t0), flush=True)
        pix.flush(), done.flush()
        fail = n - int(done.sum())
        bad += fail
        print("%s: pictures %d, malignant %d, benign %d, patients %d, failures %d"
              % (name, n, int((pics.label == 1).sum()), int((pics.label == 0).sum()), pics.patient.nunique(), fail))
    return 0 if bad == 0 else 1


class Stack:
    """Rows 0.. from the first array, then each later array from its start row (CMMD's pictures, then the extras')."""

    def __init__(self, parts):
        self.parts = parts       # [(start row, array (m, h, w))]

    def __getitem__(self, r):
        r = np.asarray(r)
        out = np.empty((len(r),) + self.parts[0][1].shape[1:], np.uint8)
        for start, arr in self.parts:
            m = (r >= start) & (r < start + len(arr))
            if m.any():
                out[m] = arr[r[m] - start]
        return out


def load_extras(names, pix, n):
    """(pictures, labels of the extra rows, the extra rows that are prepared, {name: count}); rows of extras follow CMMD's n."""
    import pandas as pd
    parts, ys, rows, counts, start = [(0, pix)], [], [], {}, n
    for name in names:
        idx, pix_p, done_p = extra_paths(name)
        if not (os.path.exists(idx) and os.path.exists(done_p)):
            sys.exit("no extra pictures for %s: run `prepare-extra` first" % name)
        df = pd.read_csv(idx, keep_default_na=False)
        m = len(df)
        done = np.array(np.memmap(done_p, dtype=np.uint8, mode="r", shape=(m,))) > 0
        parts.append((start, FileRows(pix_p, (m, H, W))))
        ys.append(df.label.to_numpy().astype(np.float32))
        rows.append(start + np.nonzero(done)[0])
        counts[name] = int(done.sum())
        start += m
    return Stack(parts), np.concatenate(ys), np.concatenate(rows), counts


# ---------------------------------------------------------------- scores

def auroc(y, s):
    """Rank AUROC with tied scores shared (NaN when one class is missing)."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    pos = int((y == 1).sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def patient_bootstrap(patient, y, s, n=BOOT, seed=BOOT_SEED):
    """(auroc, low, high): the 95% percentile interval over n resamples of the patients, with replacement."""
    patient, y, s = np.asarray(patient), np.asarray(y), np.asarray(s)
    names, inv = np.unique(patient, return_inverse=True)
    groups = [np.nonzero(inv == i)[0] for i in range(len(names))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = auroc(y[idx], s[idx])
        if not math.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))   # one class only: no interval
    return auroc(y, s), float(lo), float(hi)


def verdict(lo, hi, bar):
    """BEATEN when the interval's low end is above the bar, BEHIND when its top is below, else LEVEL."""
    return "BEATEN" if lo > bar else ("BEHIND" if hi < bar else "LEVEL")


def out_dir(version):
    return os.path.join(OUT, "mammo-v%d" % version)


def evaluate(version):
    import pandas as pd
    from core import ledger
    d = out_dir(version)
    df = pd.read_csv(os.path.join(d, "oof.csv"), keep_default_na=False)
    with ledger.run("experiment", "mammo v%d eval (idea 27)" % version) as led:
        led.settings_from({"version": version, "bars": list(BARS), "bootstrap": BOOT})
        a, lo, hi = patient_bootstrap(df.patient, df.label, df.logit)
        print("per picture: %d pictures, %d patients, AUROC %.4f (95%% CI %.4f to %.4f)"
              % (len(df), df.patient.nunique(), a, lo, hi))
        br = df.groupby(["patient", "side"], as_index=False).agg(label=("label", "first"), logit=("logit", "mean"))
        ba, blo, bhi = patient_bootstrap(br.patient, br.label, br.logit)
        print("per breast (mean of its pictures' logits): %d breasts, AUROC %.4f (95%% CI %.4f to %.4f)" % (len(br), ba, blo, bhi))
        for k in sorted(df.fold.unique()):
            f = df[df.fold == k]
            print("  fold %s: %d pictures, AUROC %.4f" % (k, len(f), auroc(f.label, f.logit)))
        led.number("picture_auroc", a).number("picture_auroc_low", lo).number("picture_auroc_high", hi)
        led.number("breast_auroc", ba).number("breast_auroc_low", blo).number("breast_auroc_high", bhi)
        led.number("pictures", len(df), "count")
        first = "Verdict (idea 27, per picture against %.3f): %s" % (BARS[0], verdict(lo, hi, BARS[0]))
        print(first)
        print("against %.3f: %s (per picture); per breast %s, against %.3f: %s"
              % (BARS[1], verdict(lo, hi, BARS[1]), verdict(blo, bhi, BARS[0]), BARS[0], verdict(blo, bhi, BARS[1])))
        led.note_is(first)
    return 0


def step_verdict(step, gain, d):
    """idea 33's line: KEPT when the two-seed gain is above the seed spread d and at least 0.01."""
    kept = gain > d and gain >= STEP_GAIN
    return "Verdict (idea 33 step %d, two-seed gain over the base above d and at least 0.01): %s" % (step, "KEPT" if kept else "NOT KEPT")


def paired_bootstrap(patient, y, s_new, s_base, n=BOOT, seed=BOOT_SEED):
    """(diff, low, high): AUROC(new) - AUROC(base), the same patient resamples for both."""
    patient, y, s_new, s_base = np.asarray(patient), np.asarray(y), np.asarray(s_new), np.asarray(s_base)
    names, inv = np.unique(patient, return_inverse=True)
    groups = [np.nonzero(inv == i)[0] for i in range(len(names))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = auroc(y[idx], s_new[idx]) - auroc(y[idx], s_base[idx])
        if not math.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))   # one class only: no interval
    return auroc(y, s_new) - auroc(y, s_base), float(lo), float(hi)


def aligned_oof(versions, index, subset=False):
    """One frame in the index's picture order with a logit column per version; refuses a version that does not cover every picture
    (subset: it may hold more, whose extra rows are ignored)."""
    import pandas as pd
    out = index[["path", "patient", "side", "label"]].copy()
    for v in versions:
        f = os.path.join(out_dir(v), "oof.csv")
        if not os.path.exists(f):
            sys.exit("compare: no %s" % f)
        df = pd.read_csv(f, keep_default_na=False).drop_duplicates("path").set_index("path")
        if subset:
            if not set(index.path) <= set(df.index):
                sys.exit("read: v%d does not cover all %d pictures asked for" % (v, len(index)))
        elif len(df) != len(index) or set(df.index) != set(index.path):
            sys.exit("compare: v%d covers %d of the %d CMMD pictures" % (v, len(df), len(index)))
        out["v%d" % v] = df.loc[out.path, "logit"].to_numpy(float)
    return out


def compare(base, new, step):
    from core import ledger
    index = load_index()
    df = aligned_oof(base + new, index)
    br = df.groupby(["patient", "side"], as_index=False).agg(label=("label", "first"), **{"v%d" % v: ("v%d" % v, "mean") for v in base + new})
    au = {v: auroc(df.label, df["v%d" % v]) for v in base + new}
    d = abs(au[base[0]] - au[base[1]]) if len(base) > 1 else 0.0
    gain = float(np.mean([au[v] for v in new]) - np.mean([au[v] for v in base]))
    with ledger.run("experiment", "mammo compare v%s against v%s (idea 33 step %d)" % (",".join(map(str, new)), ",".join(map(str, base)), step)) as led:
        led.settings_from({"base": base, "new": new, "step": step, "bootstrap": BOOT})
        for v in base + new:
            print("v%d per picture AUROC %.4f" % (v, au[v]))
        print("d = |v%d - v%d| = %.4f; two-seed mean base %.4f, new %.4f, gain %.4f"
              % (base[0], base[-1], d, np.mean([au[v] for v in base]), np.mean([au[v] for v in new]), gain))
        rec = {"base": base, "new": new, "step": step, "auroc": {str(v): au[v] for v in base + new}, "d": d, "gain": gain}
        for what, t in (("picture", df), ("breast", br)):
            diff, lo, hi = paired_bootstrap(t.patient, t.label, t["v%d" % new[0]], t["v%d" % base[0]])
            print("paired patient bootstrap, v%d - v%d per %s: %+.4f (95%% CI %+.4f to %+.4f)" % (new[0], base[0], what, diff, lo, hi))
            rec[what] = {"diff": diff, "low": lo, "high": hi}
            led.number("%s_diff" % what, diff).number("%s_diff_low" % what, lo).number("%s_diff_high" % what, hi)
        a, lo, hi = patient_bootstrap(df.patient, df.label, df["v%d" % new[0]])
        mean_new = float(np.mean([au[v] for v in new]))
        for bar in BARS:
            print("new versions' mean AUROC %.4f against %.3f: %s (v%d's interval %.4f to %.4f)"
                  % (mean_new, bar, verdict(lo, hi, bar), new[0], lo, hi))
        rec["new_mean"], rec["seed0_ci"] = mean_new, [lo, hi]
        line = step_verdict(step, gain, d)
        print(line)
        rec["verdict"] = line
        led.number("d", d).number("gain", gain).number("new_mean_auroc", mean_new)
        led.note_is(line)
    json.dump(rec, open(os.path.join(out_dir(new[0]), "compare-%d.json" % new[0]), "w"), indent=1)
    return 0


def read_scores(frame, versions):
    """Per row, the mean over the versions of sigmoid(logit)."""
    return np.mean([1.0 / (1.0 + np.exp(-np.asarray(frame["v%d" % v], float))) for v in versions], axis=0)


def read(versions, against, only_in=None):
    from core import ledger
    import pandas as pd
    index = load_index()
    if only_in is not None:
        f = os.path.join(out_dir(only_in), "oof.csv")
        if not os.path.exists(f):
            sys.exit("read: no %s" % f)
        keep = set(pd.read_csv(f, keep_default_na=False).path)
        index = index[index.path.isin(keep)].reset_index(drop=True)
    df = aligned_oof(versions + against, index, subset=only_in is not None)
    df["read"] = read_scores(df, versions)
    if against:
        df["against"] = read_scores(df, against)
    cols = {"read": ("read", "mean")}
    if against:
        cols["against"] = ("against", "mean")
    br = df.groupby(["patient", "side"], as_index=False).agg(label=("label", "first"), **cols)
    au = {v: auroc(df.label, df["v%d" % v]) for v in versions + against}
    key = ",".join(map(str, versions))
    with ledger.run("experiment", "mammo read v%s against v%s (idea 65)" % (key, ",".join(map(str, against)))) as led:
        sets = {"versions": versions, "against": against, "bootstrap": BOOT}
        if only_in is not None:
            sets["only_in"] = only_in
        led.settings_from(sets)
        for v in versions + against:
            print("v%d per picture AUROC %.4f" % (v, au[v]))
        rec = {"versions": versions, "against": against, "auroc": {str(v): au[v] for v in versions + against}}
        if only_in is not None:
            rec["only_in"] = only_in
        for what, t in (("picture", df), ("breast", br)):
            a, lo, hi = patient_bootstrap(t.patient, t.label, t["read"])
            print("read per %s: AUROC %.4f (95%% CI %.4f to %.4f)" % (what, a, lo, hi))
            for bar in BARS:
                print("  against %.3f: %s" % (bar, verdict(lo, hi, bar)))
            rec[what] = {"auroc": a, "low": lo, "high": hi}
            led.number("%s_auroc" % what, a).number("%s_auroc_low" % what, lo).number("%s_auroc_high" % what, hi)
            if against:
                diff, dlo, dhi = paired_bootstrap(t.patient, t.label, t["read"], t["against"])
                print("paired patient bootstrap, read - against read per %s: %+.4f (95%% CI %+.4f to %+.4f)" % (what, diff, dlo, dhi))
                rec[what].update({"diff": diff, "diff_low": dlo, "diff_high": dhi})
                led.number("%s_diff" % what, diff).number("%s_diff_low" % what, dlo).number("%s_diff_high" % what, dhi)
        line = None
        if against:
            line = ("Verdict (idea 65, the read against v%s paired per picture, kept when the low end is above 0): %s"
                    % (",".join(map(str, against)), "KEPT" if rec["picture"]["diff_low"] > 0 else "NOT KEPT"))
            print(line)
            led.note_is(line)
        rec["verdict"] = line
    name = "read-%s%s.json" % ("-".join(map(str, versions)), "" if only_in is None else "-in%d" % only_in)
    json.dump(rec, open(os.path.join(OUT, name), "w"), indent=1)
    return 0


# ---------------------------------------------------------------- training

def reader_module():
    """chest_xray/eval/reader.py, imported as it is (it sets up its own paths)."""
    sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
    import reader
    return reader


def limit_card_memory(torch):
    """Lease the card before touching CUDA, then cap the process at the lease (RAIDIOLOGY_CARD_GB overrides)."""
    if not torch.cuda.is_available():
        return
    from core import card
    gb = float(os.environ.get("RAIDIOLOGY_CARD_GB", LEASE_GB))
    card.lease(gb, "mammo train")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, gb * 2 ** 30 / total), torch.cuda.current_device())


def augment(x, o, gen=None, m=None):
    """x (B, 1, h, w) float in [0, 1], on the card: a small rotation, scale and shift as one affine warp (aspect-correct), a vertical
    flip drawn for half the pictures (never horizontal: the pictures are all mirrored chest-wall left), then brightness and contrast.
    With masks m (B, 1, gh, gw) the same warp is applied to them and (x, m) is returned; the draws are the same."""
    import torch
    import torch.nn.functional as F
    n, _, h, w = x.shape
    r = lambda: torch.rand(n, device=x.device, generator=gen) * 2 - 1  # noqa: E731
    ang = r() * o["rotate_deg"] * math.pi / 180.0
    sc = 1.0 + r() * o["scale"]
    tx, ty = r() * o["translate"] * 2, r() * o["translate"] * 2
    flip = torch.where(torch.rand(n, device=x.device, generator=gen) < 0.5, -1.0, 1.0)
    c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = c, -s * h / w, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = s * w / h, c, ty
    theta[:, 1, :] *= flip[:, None]
    xm = None if m is None else F.grid_sample(m, F.affine_grid(theta, list(m.shape), align_corners=False), mode="bilinear",
                                              padding_mode="zeros", align_corners=False)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    x = x * (1.0 + r() * o["contrast"]).view(n, 1, 1, 1) + (r() * o["brightness"]).view(n, 1, 1, 1)
    x = x.clamp_(0, 1)
    return x if m is None else (x, xm)


def batches(pix, rows, y, batch, shuffle, seed):
    """A background thread reading (index-sorted) batches from the memmap: yields (uint8 tensor (B, 1, H, W), labels, rows)."""
    import torch
    q = queue.Queue(maxsize=4)
    order = np.random.default_rng(seed).permutation(len(rows)) if shuffle else np.arange(len(rows))

    def work():
        for a in range(0, len(order), batch):
            ix = np.sort(order[a:a + batch])
            r = rows[ix]
            q.put((torch.from_numpy(np.ascontiguousarray(pix[r]))[:, None], torch.from_numpy(y[ix].astype(np.float32)), r))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


class PlacePix:
    """Idea 81: picture rows over n * K found crops (row i * K + q is picture i's place q); pix[r] is (len(r) * K, h, w), picture-major."""

    def __init__(self, pix, places):
        self.pix, self.places = pix, places

    def __getitem__(self, r):
        r = np.asarray(r)
        return self.pix[(r[:, None] * self.places + np.arange(self.places)[None, :]).reshape(-1)]


def mil_max(z, places):
    """Idea 81: z (pictures * places,) crop logits, picture-major -> (pictures,) the largest per picture (max pooling)."""
    return z.reshape(-1, places).max(dim=1).values


AUX_KEYS = {"aux.weight", "aux.bias"}


def load_init_state(net, state, version):
    """Strict load of `state`, except that an aux net may lack exactly the head's tensors (they then start fresh)."""
    have = set(net.state_dict())
    if AUX_KEYS <= have and not (AUX_KEYS & set(state)):
        missing, unexpected = net.load_state_dict(state, strict=False)
        if set(missing) != AUX_KEYS or unexpected:
            sys.exit("init_version=%d: the net's tensors differ beyond the outline head (missing %s, unexpected %s)"
                     % (version, sorted(set(missing) - AUX_KEYS), list(unexpected)))
        print("init_version=%d: the outline head starts fresh" % version, flush=True)
    else:
        net.load_state_dict(state)      # strict


def load_init_version_net(net, version, fold):
    import torch
    ck = torch.load(os.path.join(out_dir(version), "fold%d.pt" % fold), map_location="cpu", weights_only=False)
    load_init_state(net, ck["net"], version)


def check_init_folds(df, version, k):
    """Exit unless version's fold k csv exists and every patient in any of version's fold j csvs has fold == j in df
    (so version's fold k net, trained on its patients of the other folds, never saw a fold k patient of df)."""
    import pandas as pd
    d = out_dir(version)
    if not os.path.exists(os.path.join(d, "fold%d.csv" % k)):
        sys.exit("init_version=%d: no %s" % (version, os.path.join(d, "fold%d.csv" % k)))
    ours = dict(zip(df.patient.astype(str), df.fold.astype(int)))
    bad = set()
    for j in range(FOLDS):
        p = os.path.join(d, "fold%d.csv" % j)
        if os.path.exists(p):
            bad |= {q for q in set(pd.read_csv(p, keep_default_na=False).patient.astype(str)) if ours.get(q) != j}
    if bad:
        sys.exit("init_version=%d: %d of its patients sit in a different fold (or are absent) in this table: not the same folds"
                 % (version, len(bad)))


def predict(net, pix, rows, mean, sd, dev, batch=24, places=1):
    import torch
    net.eval()
    out = np.zeros(len(rows), np.float32)
    pos = {int(r): i for i, r in enumerate(rows)}
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for xb, _, r in batches(pix, rows, np.zeros(len(rows)), batch, False, 0):
            z = net(((xb.to(dev).float() / 255.0) - mean) / sd).float().squeeze(1)
            z = (mil_max(z, places) if places > 1 else z).cpu().numpy()
            for j, rr in enumerate(r):
                out[pos[int(rr)]] = z[j]
    return out


def run_seeds(o, fold, ep=0):
    """(net seed, batch-order seed of epoch ep): v1's, shifted by 7919 * seed_offset (the folds and the early-stop split never shift)."""
    k = SEED_STEP * int(o.get("seed_offset", 0))
    return SEED + fold + k, SEED * 7 + fold * 101 + ep + k


def aux_loss(a, m, use):
    """BCE of the outline head's logits a (B, 1, gh, gw) against the masks m (same shape, 0..1), over the pictures `use` (B,) marks; 0 when none."""
    import torch.nn.functional as F
    per = F.binary_cross_entropy_with_logits(a.float(), m, reduction="none").mean(dim=(1, 2, 3))
    u = use.to(per.dtype)
    return (per * u).sum() / u.sum().clamp(min=1.0)


def aux_dice(net, pix, masks, rows, mean, sd, dev, batch=24):
    """(mean Dice, pictures scored) of the outline head on `rows` that have an outline (cut at 0.5 both sides; a picture whose mask
    has no cell at 0.5 is not scored)."""
    import torch
    net.eval()
    rows = np.sort(np.asarray(rows))
    dice = []
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for a in range(0, len(rows), batch):
            r = rows[a:a + batch]
            x = ((torch.from_numpy(np.ascontiguousarray(pix[r]))[:, None].to(dev).float() / 255.0) - mean) / sd
            p = (net.forward_both(x)[1].float() > 0).cpu().numpy()[:, 0]
            t = np.asarray(masks[r]) >= 128
            for pi, ti in zip(p, t):
                if ti.sum():
                    dice.append(2.0 * (pi & ti).sum() / (pi.sum() + ti.sum()))
    return (float(np.mean(dice)) if dice else float("nan")), len(dice)


def aux_dice_crops(net, pix, masks, mask_has, rows, K, mean, sd, dev, batch=24):
    """(mean Dice, crops scored) for crops of pictures in `rows` that have an outline (mask >= 128)."""
    import torch
    net.eval()
    rows = np.sort(np.asarray(rows))
    dice = []
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for a in range(0, len(rows), batch):
            r = rows[a:a + batch]
            crop_rows = (r[:, None] * K + np.arange(K)[None, :]).reshape(-1)
            x = ((torch.from_numpy(np.ascontiguousarray(pix[r]))[:, None].to(dev).float() / 255.0) - mean) / sd   # pix: PlacePix, picture rows
            p = (net.forward_both(x)[1].float() > 0).cpu().numpy()[:, 0]
            t = np.asarray(masks[crop_rows]) >= 128
            for pi, ti in zip(p, t):
                if ti.sum():
                    dice.append(2.0 * (pi & ti).sum() / (pi.sum() + ti.sum()))
    return (float(np.mean(dice)) if dice else float("nan")), len(dice)


def _build_mammo_net_class(reader):
    """Returns MammoNet class with gated attention pooling (idea 56)."""
    import torch
    import torch.nn as nn
    class MammoNet(reader.ChestNet):
        def __init__(self, widths, dropout, stem_stride, n_labels, lung_aux=False, renderings=1):
            super().__init__(widths, dropout, stem_stride, n_labels, lung_aux=lung_aux, renderings=renderings)
            c = widths[-1]
            self.att_v = nn.Linear(c, 128)
            self.att_u = nn.Linear(c, 128)
            self.att_w = nn.Linear(128, 1)
            nn.init.zeros_(self.att_w.weight)
            nn.init.zeros_(self.att_w.bias)
        def pool(self, f, weights=False):
            """Gated attention pooling of stride-64 feature map f (B, C, h, w)."""
            with torch.autocast(device_type=f.device.type, enabled=False):
                f = f.float()
                b, c, h, w = f.shape
                f_flat = f.view(b, c, -1).permute(0, 2, 1)
                s = self.att_w(torch.tanh(self.att_v(f_flat)) * torch.sigmoid(self.att_u(f_flat)))
                s = s.squeeze(-1)
                w = torch.softmax(s, dim=1)
                pooled = (f_flat * w.unsqueeze(-1)).sum(dim=1)
            if weights:
                return pooled, w.float()
            return pooled.float()
        def forward(self, x):
            if self.renderings == 3:
                x = reader.add_renderings(x)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            return self.head(self.drop(self.pool(self.trunk(x))))
        def forward_both(self, x):
            if self.renderings == 3:
                x = reader.add_renderings(x)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            f = self.trunk[:reader.LUNG_STAGES](x)
            return self.head(self.drop(self.pool(self.trunk[reader.LUNG_STAGES:](f)))), self.aux(f)
    return MammoNet


def make_net(reader, widths, dropout, aux, pool=None):
    """The v1 trunk; with aux a 1x1 head on the stride-16 map predicts one lesion channel (reader's lung_aux head, one channel).
    With pool="gated", use MammoNet with gated attention pooling."""
    import torch.nn as nn
    if pool == "gated":
        MammoNetCls = _build_mammo_net_class(reader)
        net = MammoNetCls(widths, dropout, reader.CFG["stem_stride"], 1, lung_aux=aux)
    else:
        net = reader.ChestNet(widths, dropout, reader.CFG["stem_stride"], 1, lung_aux=aux)
    if aux:
        net.aux = nn.Conv2d(widths[reader.LUNG_STAGES - 1], 1, 1)
    return net


def att_entropy(net, pix, rows, mean, sd, dev, batch=24):
    """(mean entropy in nats of the attention weights over pictures in rows, cells): log(cells) when uniform."""
    import torch
    net.eval()
    rows = np.sort(np.asarray(rows))
    if not len(rows):
        return 0.0, 0
    entropies, cells = [], 0
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for a in range(0, len(rows), batch):
            r = rows[a:a + batch]
            x = ((torch.from_numpy(np.ascontiguousarray(pix[r]))[:, None].to(dev).float() / 255.0) - mean) / sd
            if net.renderings == 3:
                x = reader_module().add_renderings(x)
            if net.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            _, w = net.pool(net.trunk(x), weights=True)
            cells = int(w.shape[1])
            ent = -(w * torch.log(torch.clamp(w, min=1e-9))).sum(dim=1)
            entropies.append(ent.cpu().numpy())
    return float(np.mean(np.concatenate(entropies))), cells


def split_fold(df, fold, seed=SEED):
    """(train rows, early-stop rows, held-out rows): the held-out fold; 10% of the other patients (seeded) for early stopping."""
    te = np.nonzero(df.fold.to_numpy() == fold)[0]
    rest = df[df.fold != fold]
    ps = np.array(sorted(rest.patient.unique()))
    rng = np.random.default_rng(seed + 1000 + fold)
    val_p = set(ps[rng.permutation(len(ps))[:max(1, int(round(len(ps) * VAL_SHARE)))]])
    is_val = df.patient.isin(val_p).to_numpy() & (df.fold.to_numpy() != fold)
    tr = np.nonzero((df.fold.to_numpy() != fold) & ~is_val)[0]
    return tr, np.nonzero(is_val)[0], te


def train_fold(df, pix, fold, o, dev, partial, fingerprint, extra=None, aux=None):
    """extra: (rows, labels) of the extra pictures (their rows follow CMMD's, in `pix`); aux: (masks, has) of CMMD's outlines.
    Returns (held-out rows, their logits, best early-stop AUROC, history, info dict, best_state, mean, sd)."""
    import torch
    import torch.nn as nn
    from core import ckpt
    reader = reader_module()
    widths = o["widths"] or reader.CFG["widths"]
    dropout = reader.CFG["dropout"] if o["dropout"] is None else o["dropout"]
    tr, va, te = split_fold(df, fold)
    K = o.get("found") or 1
    pb = max(1, o["batch"] // K)     # K = 1 without found: the batch is o["batch"]
    y = df.label.to_numpy().astype(np.float32)
    tr_all = tr
    if extra is not None:
        y, tr_all = np.concatenate([y, extra[1]]), np.concatenate([tr, extra[0]])
    torch.manual_seed(run_seeds(o, fold)[0])
    net = make_net(reader, widths, dropout, aux is not None, pool=o.get("pool")).to(dev)
    if o.get("init"):     # a starting point only: a resumed partial below overwrites it
        cbis_module().load_init(net, o["init"])
    if o.get("init_version"):
        check_init_folds(df, o["init_version"], fold)
        load_init_version_net(net, o["init_version"], fold)
    n_cmmd = len(df)
    opt = torch.optim.AdamW(net.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o["epochs"])
    samp = np.sort(np.random.default_rng(SEED).choice(tr, min(200, len(tr)), replace=False))
    sx = torch.from_numpy(np.ascontiguousarray(pix[samp])).float() / 255.0
    mean, sd = float(sx.mean()), float(sx.std().clamp_min(1e-3))
    epoch, best, best_state, hist = 0, -1.0, None, []
    ck = ckpt.load_partial(partial, fingerprint)
    if ck is not None:
        net.load_state_dict(ck["net"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        epoch, best, best_state, hist = ck["epoch"], ck["best"], ck["best_state"], ck["hist"]
        ckpt.set_rng_state(ck["rng"])
        print("fold %d: resumed after epoch %d" % (fold, epoch), flush=True)
    loss_fn = nn.BCEWithLogitsLoss()
    every = ckpt.Every()
    cuda = dev.type == "cuda"
    for ep in range(epoch, o["epochs"]):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb, rb in batches(pix, tr_all, y[tr_all], pb, True, run_seeds(o, fold, ep)[1]):
            if len(yb) < 2:
                continue
            if aux is None:
                x = augment(xb.to(dev).float() / 255.0, o)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=cuda):
                    z = net((x - mean) / sd).float().squeeze(1)
                if o.get("found"):
                    z = mil_max(z, K)
                loss = loss_fn(z, yb.to(dev))
            else:
                if o.get("found"):
                    crop_rows = (rb[:, None] * K + np.arange(K)[None, :]).reshape(-1)
                    use = aux[3][rb]
                    use_crops = np.repeat(use, K)
                    m = torch.from_numpy(np.asarray(aux[2][crop_rows])).to(dev).float()[:, None] / 255.0
                else:
                    use = (rb < n_cmmd) & aux[1][np.minimum(rb, n_cmmd - 1)]
                    m = torch.from_numpy(np.asarray(aux[0][np.where(use, rb, 0)])).to(dev).float()[:, None] / 255.0
                    use_crops = use
                x, m = augment(xb.to(dev).float() / 255.0, o, m=m)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=cuda):
                    z, a = net.forward_both((x - mean) / sd)
                if o.get("found"):
                    z_pic = mil_max(z.float().squeeze(1), K)
                    loss = loss_fn(z_pic, yb.to(dev)) + o["aux_weight"] * aux_loss(a, m, torch.from_numpy(use_crops).to(dev))
                else:
                    loss = loss_fn(z.float().squeeze(1), yb.to(dev)) + o["aux_weight"] * aux_loss(a, m, torch.from_numpy(use).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + float(loss) * len(yb), n + len(yb)
        sched.step()
        a = auroc(y[va], predict(net, pix, va, mean, sd, dev, batch=max(1, 24 // K), places=K))
        hist.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val_auroc": a})
        if a > best:
            best, best_state = a, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print("fold %d epoch %d/%d: loss %.4f, early-stop AUROC %.4f (best %.4f), %.0f s"
              % (fold, ep + 1, o["epochs"], tot / max(n, 1), a, best, time.time() - t0), flush=True)
        if every.due() or ep + 1 == o["epochs"]:
            ckpt.save_partial(partial, {"fingerprint": fingerprint, "net": net.state_dict(), "opt": opt.state_dict(),
                                        "sched": sched.state_dict(), "epoch": ep + 1, "best": best,
                                        "best_state": best_state, "hist": hist, "rng": ckpt.rng_state()})
            every.mark()
    net.load_state_dict(best_state)
    info = {}
    if aux is not None:
        if o.get("found"):
            dice, scored = aux_dice_crops(net, pix, aux[2], aux[3], va[aux[3][va]], K, mean, sd, dev,
                                          batch=max(1, 24 // K))
            print("fold %d: the outline head's mean Dice on %d early-stop crops with an outline: %.4f" % (fold, scored, dice), flush=True)
            info = {"aux_dice": dice, "aux_dice_crops": scored}
        else:
            dice, scored = aux_dice(net, pix, aux[0], va[aux[1][va]], mean, sd, dev)
            print("fold %d: the outline head's mean Dice on %d early-stop pictures with an outline: %.4f" % (fold, scored, dice), flush=True)
            info = {"aux_dice": dice, "aux_dice_pictures": scored}
    if o.get("pool"):
        ent, cells = att_entropy(net, pix, va, mean, sd, dev)
        info["att_entropy"] = ent
        info["att_cells"] = cells
        print("fold %d: attention entropy %.4f, log(%d) = %.4f" % (fold, ent, cells, math.log(cells)), flush=True)
    return te, predict(net, pix, te, mean, sd, dev, batch=max(1, 24 // K), places=K), best, hist, info, best_state, mean, sd


def parse_sets(items):
    o = dict(SETTINGS)
    for it in items or []:
        k, _, v = it.partition("=")
        if k not in SETTINGS and k not in IDEA33 and k not in IDEA42 and k not in IDEA56 and k not in IDEA66 and k not in IDEA81:
            sys.exit("unknown setting %r; known: %s" % (k, ", ".join(list(SETTINGS) + list(IDEA33) + list(IDEA42) + list(IDEA56) + list(IDEA66) + list(IDEA81))))
        o[k] = json.loads(v)
    return drop_off(o)


def drop_off(o):
    """Idea 33's settings leave o when off, so v1's fingerprint is the same string as before; aux_weight is set when aux is on."""
    if not o.get("seed_offset"):
        o.pop("seed_offset", None)
    elif not isinstance(o["seed_offset"], int) or isinstance(o["seed_offset"], bool):
        sys.exit("seed_offset must be an integer, got %r" % (o["seed_offset"],))
    if o.get("aux"):
        if o["aux"] != "tompei":
            sys.exit("aux must be \"tompei\", got %r" % (o["aux"],))
        o["aux_weight"] = float(o.get("aux_weight", AUX_WEIGHT))
    else:
        if "aux_weight" in o:
            sys.exit("aux_weight needs aux=\"tompei\"")
        o.pop("aux", None)
    if o.get("extra"):
        bad = [e for e in o["extra"] if e not in EXTRAS]
        if bad or not isinstance(o["extra"], list):
            sys.exit("extra must be a list of %s, got %r" % (", ".join(EXTRAS), o["extra"]))
        o["extra"] = sorted(set(o["extra"]))
    else:
        o.pop("extra", None)
    if o.get("init"):
        inits = cbis_module().INITS
        if not isinstance(o["init"], str) or o["init"] not in inits:
            sys.exit("init must be one of %s, got %r" % (", ".join(sorted(inits)), o["init"]))
    else:
        o.pop("init", None)
    if o.get("pool"):
        if o["pool"] != "gated":
            sys.exit("pool must be \"gated\", got %r" % (o["pool"],))
    else:
        o.pop("pool", None)
    if o.get("lesions"):
        if o["lesions"] is not True:
            sys.exit("lesions must be true, got %r" % (o["lesions"],))
        if o.get("aux") or o.get("extra"):
            sys.exit("lesions cannot be used with aux or extra")
    else:
        o.pop("lesions", None)
    if o.get("found"):
        if not isinstance(o["found"], int) or isinstance(o["found"], bool) or o["found"] < 1:
            sys.exit("found must be an integer >= 1, got %r" % (o["found"],))
        if o.get("lesions") or o.get("extra"):
            sys.exit("found cannot be used with lesions or extra")
    else:
        o.pop("found", None)
    if o.get("init_version"):
        if not isinstance(o["init_version"], int) or isinstance(o["init_version"], bool):
            sys.exit("init_version must be an integer, got %r" % (o["init_version"],))
    else:
        o.pop("init_version", None)
    return o


def train(version, folds, epochs, sets):
    import pandas as pd
    import torch
    from core import ckpt, ledger, polite  # noqa: F401
    o = parse_sets(sets)
    if epochs:
        o["epochs"] = epochs
    limit_card_memory(torch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if o.get("lesions"):
        if not (os.path.exists(LESCSV) and os.path.exists(LES)):
            sys.exit("no lesion crops: run prepare-lesions first")
        df = pd.read_csv(LESCSV, keep_default_na=False)
        pix = FileRows(LES, (len(df), LS, LS))
    elif o.get("found"):
        K = o["found"]
        found, foundcsv = found_paths(K)
        if not (os.path.exists(foundcsv) and os.path.exists(found)):
            sys.exit("no found crops for K=%d: run `locate --version 6 --places %d` first" % (K, K))
        df = load_index()
        fc = pd.read_csv(foundcsv, keep_default_na=False)
        if K > 1:
            fc = fc[fc.place.astype(int) == 0]
        if len(fc) != len(df) or list(fc.path) != list(df.path):
            sys.exit("the found csv (%d pictures) does not match the picture table (%d): rerun locate --places %d" % (len(fc), len(df), K))
        if os.path.getsize(found) != len(df) * K * LS * LS:
            sys.exit("%s is not %d crops of %d x %d: rerun locate --places %d" % (found, len(df) * K, LS, LS, K))
        pix = PlacePix(FileRows(found, (len(df) * K, LS, LS)), K)
    else:
        df = load_index()
        pix = pixels()
    d = out_dir(version)
    os.makedirs(d, exist_ok=True)
    extra, aux, fpd = None, None, {"version": version, "settings": o, "seed": SEED, "pictures": len(df)}
    if o.get("lesions"):
        fpd["lesions"] = len(df)
    if o.get("extra"):
        pix, ey, er, counts = load_extras(o["extra"], pix, len(df))
        extra, fpd["extra_pictures"] = (er, ey), counts
        print("extra pictures joining the training rows: %s" % counts, flush=True)
    if o.get("aux"):
        aux = outline_table(len(df))
        print("outlines: %d CMMD pictures have one" % int(aux[1].sum()), flush=True)
        if o.get("found"):
            K = o["found"]
            masks_path, has_path = found_masks_paths(K)
            if not (os.path.exists(masks_path) and os.path.exists(has_path)):
                sys.exit("no found masks for K=%d: run `found-masks --places %d` first" % (K, K))
            expected_size = len(df) * K * 32 * 32
            if os.path.getsize(masks_path) != expected_size:
                sys.exit("%s is not %d crops of 32 x 32: rerun found-masks --places %d" % (masks_path, len(df) * K, K))
            found_masks_obj = FileRows(masks_path, (len(df) * K, 32, 32))
            found_has = np.array(np.memmap(has_path, dtype=np.uint8, mode="r", shape=(len(df),))) > 0
            aux = (aux[0], aux[1], found_masks_obj, found_has)
            fpd["found_masks"] = {"path": masks_path, "size": os.path.getsize(masks_path), "mtime": int(os.path.getmtime(masks_path))}
    if o.get("init"):
        p = cbis_module().INITS[o["init"]]
        fpd["init_file"] = {"path": p, "size": os.path.getsize(p), "mtime": int(os.path.getmtime(p))}
    if o.get("init_version"):
        files = {}
        for k in range(FOLDS):
            p = os.path.join(out_dir(o["init_version"]), "fold%d.pt" % k)
            if os.path.exists(p):
                files[k] = {"path": p, "size": os.path.getsize(p), "mtime": int(os.path.getmtime(p))}
        missing = [k for k in folds if k not in files]
        if missing:
            sys.exit("init_version=%d has no fold.pt for folds %s" % (o["init_version"], missing))
        fpd["init_version_files"] = files
    fp = json.dumps(fpd, sort_keys=True)
    for k in folds:
        out_csv = os.path.join(d, "fold%d.csv" % k)
        meta = out_csv + ".json"
        if os.path.exists(out_csv) and os.path.exists(meta) and json.load(open(meta))["fingerprint"] == fp:
            print("fold %d: already done" % k)
            continue
        with ledger.run("experiment", "mammo v%d train fold %d" % (version, k)) as led:
            led.settings_from(o)
            te, z, best, hist, info, best_state, mean, sd = train_fold(df, pix, k, o, dev, os.path.join(d, "fold%d.partial.pt" % k), fp + "|fold%d" % k,
                                                                        extra, aux)
            f = df.iloc[te][["path", "patient", "side", "view", "label", "fold"]].copy()
            f["logit"], f["p"] = z, 1.0 / (1.0 + np.exp(-z))
            torch.save({"net": best_state, "mean": mean, "sd": sd, "settings": o}, os.path.join(d, "fold%d.pt" % k))
            f.to_csv(out_csv, index=False)
            json.dump({"fingerprint": fp, "best_early_stop_auroc": best, "history": hist, **info}, open(meta, "w"))
            ckpt.clear_partial(os.path.join(d, "fold%d.partial.pt" % k))
            led.number("early_stop_auroc", best).number("held_out_auroc", auroc(f.label, f.logit))
            led.number("epochs", o["epochs"], "count")
            if "aux_dice" in info:
                led.number("aux_dice", info["aux_dice"])
            if "att_entropy" in info:
                led.number("att_entropy", info["att_entropy"])
            if dev.type == "cuda":
                led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
                led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)  # what the cap counts
                torch.cuda.reset_peak_memory_stats()
            print("fold %d: held-out AUROC %.4f -> %s" % (k, auroc(f.label, f.logit), out_csv), flush=True)
    parts = [pd.read_csv(p, keep_default_na=False) for p in sorted(glob.glob(os.path.join(d, "fold*.csv")))]
    if parts:
        pd.concat(parts).to_csv(os.path.join(d, "oof.csv"), index=False)
        print("oof.csv: %d pictures from %d folds" % (sum(len(p) for p in parts), len(parts)))
    return 0


# ---------------------------------------------------------------- selftest

def selftest():
    import cv2
    import torch
    from sklearn.metrics import roc_auc_score
    # the join: side and patient, with the reasons
    rows = [{"ID1": "P1", "LeftRight": "L", "classification": "Malignant"}, {"ID1": "P1", "LeftRight": "R", "classification": "Benign"},
            {"ID1": "P2", "LeftRight": "R", "classification": "Benign"}, {"ID1": "P3", "LeftRight": "L", "classification": "Other"}]
    labels = breast_labels(rows)
    assert labels == {("P1", "L"): 1, ("P1", "R"): 0, ("P2", "R"): 0}, labels
    pics = [{"patient": "P1", "side": "L"}, {"patient": "P1", "side": "R"}, {"patient": "P2", "side": "L"},
            {"patient": "P9", "side": "R"}, {"patient": "P2", "side": ""}, {"patient": "P3", "side": "L"}]
    lab, why = join_labels(pics, labels)
    assert lab == [1, 0, -1, -1, -1, -1], lab
    assert why[2].startswith("that side") and why[3].startswith("patient not") and why[4] == "no ImageLaterality"
    assert why[5].startswith("patient not"), why[5]      # a row with another classification is not a label
    assert view_of(["P", "FL"]) == "MLO" and view_of(["P", "L"]) == "CC" and view_of(None) == ""
    # folds by patient, stratified
    rng = np.random.default_rng(1)
    pl = {"p%03d" % i: int(i % 4 == 0) for i in range(200)}
    fold = assign_folds(pl)
    assert set(fold) == set(pl) and set(fold.values()) == set(range(FOLDS))
    for k in range(FOLDS):
        mem = [p for p in pl if fold[p] == k]
        assert abs(len(mem) - 40) <= 1 and abs(sum(pl[p] for p in mem) - 10) <= 1, (k, len(mem))
    assert fold == assign_folds(pl) and fold != assign_folds(pl, seed=SEED + 1)
    # patient_labels: a malignant breast makes the patient malignant
    assert patient_labels(["P1", "P1", "P2"], labels) == {"P1": 1, "P2": 0}
    # the crop and mirror: a breast on either side, a far speck of text that must go
    for side in ("right", "left"):
        a = np.zeros((900, 700), np.uint8)
        yy, xx = np.mgrid[:900, :700]
        cx = 700 if side == "right" else 0
        a[((xx - cx) / 450.0) ** 2 + ((yy - 450) / 380.0) ** 2 < 1] = 160
        a[20:40, 330:350] = 255 if side == "right" else 0       # a bright speck, away from the breast
        a[20:40, 5:25] = 255
        o = breast_crop(a)
        assert o.shape == (H, W) and o.dtype == np.uint8
        assert o[:, :W // 2].astype(int).sum() > o[:, W // 2:].astype(int).sum() * 1.3, side     # chest wall on the left
        assert o[:, :8].max() > 0 and o[-1].max() == 0                                          # filled to the left edge, padded below
        assert (o == 255).sum() == 0, "the speck outside the breast was kept"
    # the AUROC, the bootstrap and the verdict
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    s[:20] = np.round(s[:20])
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-9
    pat = np.repeat(np.arange(100), 3)
    a, lo, hi = patient_bootstrap(pat, y, s, n=200)
    assert lo < a < hi and 0 < lo and hi < 1
    a, lo, hi = patient_bootstrap(pat, y, 2.0 * y - 1, n=50)
    assert a == lo == hi == 1.0
    assert verdict(0.86, 0.90, 0.848) == "BEATEN" and verdict(0.80, 0.86, 0.848) == "LEVEL" and verdict(0.78, 0.84, 0.848) == "BEHIND"
    assert verdict(0.848, 0.9, 0.848) == "LEVEL" and verdict(0.7, 0.848, 0.848) == "LEVEL"
    # the augmentation: shape kept, range kept, identity when every draw is zero, a vertical flip only
    x = torch.rand(4, 1, 64, 40)
    o0 = {"rotate_deg": 0.0, "scale": 0.0, "translate": 0.0, "brightness": 0.0, "contrast": 0.0}
    g = torch.Generator().manual_seed(3)
    z = augment(x.clone(), o0, g)
    same = [torch.allclose(z[i], x[i], atol=1e-4) for i in range(4)]
    flipped = [torch.allclose(z[i], x[i].flip(1), atol=1e-4) for i in range(4)]
    assert all(a or b for a, b in zip(same, flipped)) and not any(torch.allclose(z[i], x[i].flip(2), atol=1e-4) for i in range(4))
    z = augment(x.clone(), dict(SETTINGS), g)
    assert z.shape == x.shape and float(z.min()) >= 0 and float(z.max()) <= 1
    # the early-stop split: no patient in two of train, early stop and held out
    import pandas as pd
    df = pd.DataFrame({"patient": ["p%d" % (i // 3) for i in range(150)], "label": [i % 2 for i in range(150)]})
    df["fold"] = df.patient.map(assign_folds(patient_labels(df.patient, {(p, "L"): 1 for p in df.patient})))
    tr, va, te = split_fold(df, 2)
    sets = [set(df.patient[i]) for i in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]) and len(va) > 0
    assert len(tr) + len(va) + len(te) == 150
    selftest_idea33(df)
    selftest_idea81()
    # idea 65: the read is the mean of the nets' probabilities
    tiny = pd.DataFrame({"v1": [0.0, 2.0], "v2": [0.0, -2.0]})
    assert np.allclose(read_scores(tiny, [1, 2]), [0.5, 0.5])
    assert np.allclose(read_scores(tiny, [1]), [0.5, 1 / (1 + math.exp(-2.0))])
    # idea 66: the lesion crop is centred on the outline's box, black past the edge
    a = (np.arange(600 * 700) % 251 + 1).astype(np.uint8).reshape(600, 700)
    sq = np.array([[90.0, 40.0], [110.0, 40.0], [110.0, 60.0], [90.0, 60.0]])
    big = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
    crop, cx, cy, ow, oh = lesion_crop(a, [big, sq], False)
    assert crop.shape == (LS, LS) and (cx, cy, ow, oh) == (100, 50, 20, 20)
    assert crop[256, 256] == a[50, 100] and not crop[:206].any() and not crop[:, :156].any()
    assert crop[206, 156] == a[0, 0] and np.array_equal(lesion_crop(a, [sq], True)[0], crop[:, ::-1])
    # idea 67: unmap_point inverts map_points, mirrored or not; crop_at is lesion_crop's cut
    for mir in (False, True):
        t = {"y0": 30, "x0": 17, "mirror": mir, "cw": 301, "s": 0.8125}
        for ox, oy in ((17.0, 30.0), (100.0, 250.0), (300.5, 77.25)):
            mx, my = map_points([(ox, oy)], t)[0]
            ux, uy = unmap_point(mx, my, t)
            assert abs(ux - ox) < 1e-9 and abs(uy - oy) < 1e-9, (mir, ox, oy, ux, uy)
    assert np.array_equal(crop_at(a, 100, 50, False), crop) and np.array_equal(crop_at(a, 100, 50, True), crop[:, ::-1])
    print("selftest ok")
    return 0


def selftest_idea81():
    import tempfile
    import torch
    import pandas as pd
    global OUT
    # (a) found is refused with lesions or extra, but allowed with aux; (c) no new keys when off
    for bad in ("lesions=true", 'extra=["cbis"]'):
        try:
            parse_sets(["found=3", bad])
        except SystemExit:
            continue
        raise AssertionError("found with %s was not refused" % bad)
    assert parse_sets([]) == dict(SETTINGS) and parse_sets(["found=null", "init_version=0"]) == dict(SETTINGS)
    assert parse_sets(["found=3", "init_version=21"])["found"] == 3 and parse_sets(["init_version=21"])["init_version"] == 21
    assert parse_sets(["found=3", 'aux="tompei"'])["found"] == 3 and parse_sets(["found=3", 'aux="tompei"'])["aux"] == "tompei"
    # (b) the pooling: per-picture max, gradient only on each picture's max crop
    z = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.5], requires_grad=True)
    m = mil_max(z, 3)
    assert torch.equal(m, torch.tensor([0.9, 0.7]))
    m.sum().backward()
    assert z.grad.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0], z.grad
    assert PlacePix(np.arange(12).reshape(6, 2), 3)[np.array([1])].reshape(-1).tolist() == [6, 7, 8, 9, 10, 11]
    # found+aux: one training step with K=3 crops, B=2 pictures, one with outline one without
    reader = reader_module()
    K = 3
    net = make_net(reader, [4, 4, 4, 4, 4], 0.1, True)
    masks = np.zeros((2 * K, 32, 32), np.uint8)
    masks[0, 10:14, 10:14] = 255      # picture 0 has an outline in its first crop; picture 1 has none
    has = np.array([True, False])
    rb = np.array([0, 1])
    yb = torch.tensor([1.0, 0.0])
    crop_rows = (rb[:, None] * K + np.arange(K)[None, :]).reshape(-1)
    for use_pics in (has, np.array([False, False])):
        x = torch.rand(2 * K, 1, LS, LS)
        m = torch.from_numpy(masks[crop_rows]).float()[:, None] / 255.0
        x, m = augment(x, SETTINGS, m=m)
        z, a = net.forward_both(x)
        assert z.shape == (2 * K, 1) and a.shape == (2 * K, 1, 32, 32) and m.shape == a.shape, (z.shape, a.shape, m.shape)
        aux_term = aux_loss(a, m, torch.from_numpy(np.repeat(use_pics[rb], K)))
        loss = torch.nn.BCEWithLogitsLoss()(mil_max(z.float().squeeze(1), K), yb) + 0.5 * aux_term
        assert math.isfinite(float(loss)) and math.isfinite(float(aux_term))
        loss.backward()
        assert float(aux_term) == 0.0 or use_pics.any()
        if not use_pics.any():
            assert float(aux_term) == 0.0
    # the crop mask: a 64 x 64 square at DICOM x 300..363, y 400..463 in the crop centred at (300, 400) fills crop columns 256..319 and
    # rows 256..319, so cells 16..19 of 32 on both axes; mirrored, the columns become 512-320..512-257 = 192..255, cells 12..15
    sq = np.array([[300.0, 400.0], [364.0, 400.0], [364.0, 464.0], [300.0, 464.0]])
    for mir, cols in ((False, range(16, 20)), (True, range(12, 16))):
        g = crop_mask([sq], 300, 400, mir)
        assert g.shape == (32, 32) and g.dtype == np.uint8
        got = np.argwhere(g >= 128)
        assert set(got[:, 0]) == set(range(16, 20)) and set(got[:, 1]) == set(cols) and len(got) == 16, (mir, got.tolist())
    assert not crop_mask([sq], 5000, 5000, False).any()
    # aux_dice_crops reads through PlacePix (picture rows): K = 3, pictures 0 and 1, one outlined crop; it must run
    tiny = make_net(reader, [4, 4, 4, 4, 4], 0.0, True)
    crops = np.random.default_rng(0).integers(0, 256, (2 * K, LS, LS), dtype=np.uint8)
    dm, dn = aux_dice_crops(tiny, PlacePix(crops, K), masks, has, np.array([0, 1]), K, 0.5, 0.25, torch.device("cpu"), batch=1)
    assert dn <= 1 and (math.isnan(dm) or 0.0 <= dm <= 1.0), (dm, dn)
    # (d) init_version: V's patients must keep their folds (a subset is fine); a moved patient or a missing fold csv is refused
    old = OUT
    with tempfile.TemporaryDirectory() as tmp:
        OUT = tmp
        try:
            os.makedirs(out_dir(99))
            ours = pd.DataFrame({"patient": ["a", "b", "c", "d"], "fold": [0, 0, 1, 1]})
            w = lambda j, ps: pd.DataFrame({"patient": ps}).to_csv(os.path.join(out_dir(99), "fold%d.csv" % j), index=False)  # noqa: E731
            w(0, ["a"])
            w(1, ["c", "c"])
            check_init_folds(ours, 99, 0)
            check_init_folds(ours, 99, 1)
            for k in (2,):      # no fold2.csv
                try:
                    check_init_folds(ours, 99, k)
                except SystemExit:
                    continue
                raise AssertionError("missing fold csv not refused")
            w(1, ["c", "b"])    # b is fold 0 here
            try:
                check_init_folds(ours, 99, 1)
            except SystemExit:
                pass
            else:
                raise AssertionError("a patient in a different fold was not refused")
        finally:
            OUT = old
    # init_version into an aux net: a state without aux.* leaves exactly aux.weight and aux.bias missing; any other gap is refused
    src = make_net(reader, [4, 4, 4, 4, 4], 0.0, True).state_dict()
    state = {k: v for k, v in src.items() if k not in AUX_KEYS}
    dst = make_net(reader, [4, 4, 4, 4, 4], 0.0, True)
    missing, unexpected = dst.load_state_dict(state, strict=False)
    assert set(missing) == AUX_KEYS and not unexpected
    load_init_state(dst, state, 0)
    assert all(torch.equal(dst.state_dict()[k], state[k]) for k in state)
    other = next(k for k in state if k.startswith("trunk."))
    try:
        load_init_state(make_net(reader, [4, 4, 4, 4, 4], 0.0, True), {k: v for k, v in state.items() if k != other}, 0)
    except SystemExit:
        pass
    else:
        raise AssertionError("a state missing a trunk tensor was accepted")
    try:
        load_init_state(make_net(reader, [4, 4, 4, 4, 4], 0.0, True), {k: v for k, v in src.items() if k != "aux.bias"}, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a state with half the head was accepted")


def selftest_idea33(df):
    import torch
    import pandas as pd
    # settings: none given is v1's dict and fingerprint string; seed_offset=0 is no setting; a refused combination
    fpv = lambda o: json.dumps({"version": 1, "settings": o, "seed": SEED, "pictures": 3744}, sort_keys=True)  # noqa: E731
    assert parse_sets([]) == dict(SETTINGS) and fpv(parse_sets([])) == fpv(dict(SETTINGS))
    assert parse_sets(["seed_offset=0", "aux=null", "extra=[]"]) == dict(SETTINGS)
    assert parse_sets(["seed_offset=1"])["seed_offset"] == 1 and "aux" not in parse_sets(["seed_offset=1"])
    o = parse_sets(["aux=\"tompei\"", "extra=[\"cesm\",\"cbis\"]"])
    assert o["aux"] == "tompei" and o["aux_weight"] == 0.5 and o["extra"] == ["cbis", "cesm"]
    for bad in (["aux_weight=0.3"], ["aux=\"x\""], ["extra=[\"nope\"]"], ["seed_offset=1.5"], ["init=\"nope\""]):
        try:
            parse_sets(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    # idea 42 step 2: init is absent when off, names a cbis.INITS key when on, and load_init copies every trunk tensor
    assert "init" not in parse_sets([]) and parse_sets([]) == dict(SETTINGS) and "init" not in parse_sets(["init=null"])
    assert parse_sets(["init=\"chest-v103\""])["init"] == "chest-v103"
    import tempfile
    rd = reader_module()
    inet = make_net(rd, rd.CFG["widths"], 0.0, True)
    trunk = {k: v for k, v in inet.state_dict().items() if k.startswith("trunk.")}
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save({"state": trunk}, tmp)
        got = cbis_module().load_init(make_net(rd, rd.CFG["widths"], 0.0, True), "chest-v103", say=lambda *a: None, path=tmp)
    finally:
        os.remove(tmp)
    assert got == len(trunk) and got > 0, (got, len(trunk))
    # seeds: v1's at k = 0, shifted by 7919 k otherwise; the folds and the early-stop split do not take k
    assert run_seeds({}, 2, 3) == (SEED + 2, SEED * 7 + 2 * 101 + 3) and run_seeds({"seed_offset": 0}, 2, 3) == run_seeds({}, 2, 3)
    a, b = run_seeds({}, 2, 3), run_seeds({"seed_offset": 1}, 2, 3)
    assert b[0] - a[0] == 7919 and b[1] - a[1] == 7919
    assert "seed_offset" not in assign_folds.__code__.co_varnames and "seed_offset" not in split_fold.__code__.co_varnames
    assert all(np.array_equal(x, y) for x, y in zip(split_fold(df, 1), split_fold(df, 1)))
    # breast_crop's transform: a marked 5 x 5 block lands where map_points says, mirrored or not; the polygon mask sits under it
    for side, cx, mark in (("right", 700, (560, 300)), ("left", 0, (140, 300))):
        a = np.zeros((900, 700), np.uint8)
        yy, xx = np.mgrid[:900, :700]
        a[((xx - cx) / 450.0) ** 2 + ((yy - 450) / 380.0) ** 2 < 1] = 160
        a[mark[1] - 2:mark[1] + 3, mark[0] - 2:mark[0] + 3] = 250
        out, t = breast_crop(a, transform=True)
        assert np.array_equal(out, breast_crop(a)) and t["mirror"] == (side == "right"), (side, t)
        px, py = map_points([mark], t)[0]
        w = np.clip(out.astype(np.float64) - 160, 0, None)
        gy, gx = (w * np.mgrid[:H, :W][0]).sum() / w.sum(), (w * np.mgrid[:H, :W][1]).sum() / w.sum()
        assert abs(gx - px) < 2 and abs(gy - py) < 2, (side, (gx, gy), (px, py))
        sq = np.array([[mark[0] - 20, mark[1] - 20], [mark[0] + 20, mark[1] - 20], [mark[0] + 20, mark[1] + 20], [mark[0] - 20, mark[1] + 20]])
        mm = rasterise([map_points(sq, t)])
        g = to_grid(mm)
        assert g.shape == (GH, GW) and g.dtype == np.uint8 and g.max() > 0
        gyc, gxc = [(g.astype(np.float64) * c).sum() / g.sum() for c in np.mgrid[:GH, :GW]]
        assert abs(gxc * 16 + 8 - px) < 12 and abs(gyc * 16 + 8 - py) < 12, (side, (gxc, gyc), (px, py))
        assert ((mm > 0) & (out > 0)).sum() / (mm > 0).sum() > 0.99
    # the extra-row reader: CMMD's rows, then the extras'
    c0, c1 = np.full((3, 4, 5), 1, np.uint8), np.full((2, 4, 5), 9, np.uint8)
    st = Stack([(0, c0), (3, c1)])
    got = st[np.array([1, 3, 4])]
    assert got.shape == (3, 4, 5) and [int(got[i, 0, 0]) for i in range(3)] == [1, 9, 9]
    # CBIS: one picture per view, any malignant lesion -> 1, the Mass- file preferred, test patients out
    cols = ["split", "type", "patient", "side", "view", "y", "full"]
    ix = pd.DataFrame([["train", "calc", "A", "L", "CC", 1, "a_calc.dcm"], ["train", "mass", "A", "L", "CC", 0, "a_mass.dcm"],
                       ["train", "mass", "A", "L", "CC", 0, "a_mass.dcm"], ["train", "mass", "A", "R", "CC", 0, "ar.dcm"],
                       ["train", "mass", "B", "L", "MLO", 1, "b.dcm"], ["test", "mass", "B", "R", "MLO", 0, "bt.dcm"],
                       ["train", "calc", "C", "L", "MLO", -1, ""], ["train", "calc", "D", "R", "MLO", 0, "d.dcm"]], columns=cols)
    pics = {(p["patient"], p["side"], p["view"]): p for p in cbis_pictures(ix)}
    assert set(pics) == {("A", "L", "CC"), ("A", "R", "CC"), ("D", "R", "MLO")}, set(pics)    # B has a test row, C has no usable lesion
    assert pics[("A", "L", "CC")]["label"] == 1 and pics[("A", "L", "CC")]["path"] == "a_mass.dcm" and pics[("A", "R", "CC")]["label"] == 0
    cr = [{"Image_name": "P1_L_DM_MLO", "Patient_ID": "1", "Side": "L", "Type": "DM", "View": "MLO", CESM_COL: "Malignant"},
          {"Image_name": "P2_R_DM_CC", "Patient_ID": "2", "Side": "R", "Type": "DM", "View": "CC", CESM_COL: "Benign "},
          {"Image_name": "P3_R_DM_CC", "Patient_ID": "3", "Side": "R", "Type": "DM", "View": "CC", CESM_COL: "Normal"},
          {"Image_name": "P1_L_CESM_MLO", "Patient_ID": "1", "Side": "L", "Type": "CESM", "View": "MLO", CESM_COL: "Malignant"}]
    cp = cesm_pictures(cr)
    assert [(p["path"], p["label"]) for p in cp] == [("P1_L_DM_MLO.jpg", 1), ("P2_R_DM_CC.jpg", 0)]
    # the compare verdict at its edges
    assert step_verdict(2, 0.02, 0.02).endswith(": NOT KEPT") and step_verdict(2, 0.0099, 0.0).endswith(": NOT KEPT")
    assert step_verdict(2, 0.02, 0.01).endswith(": KEPT") and not step_verdict(2, 0.02, 0.01).endswith("NOT KEPT")
    assert step_verdict(7, 0.02, 0.01) == "Verdict (idea 33 step 7, two-seed gain over the base above d and at least 0.01): KEPT"
    # the aux net: the logit and a stride-16 map; the loss counts only the pictures marked
    reader = reader_module()
    net = make_net(reader, [4, 4, 4, 4, 4], 0.1, True)
    z, a = net.forward_both(torch.rand(2, 1, 128, 80))      # five stages need at least 128 x 80; the stride-16 map is 8 x 5
    assert z.shape == (2, 1) and a.shape == (2, 1, 8, 5), (z.shape, a.shape)
    assert net(torch.rand(2, 1, 128, 80)).shape == (2, 1)
    mk = torch.rand(2, 1, 8, 5)
    assert float(aux_loss(a, mk, torch.tensor([False, False]))) == 0.0 and math.isfinite(float(aux_loss(a, mk, torch.tensor([True, False]))))
    zm, mm = augment(torch.rand(2, 1, 64, 40), SETTINGS, m=torch.rand(2, 1, GH, GW))
    assert zm.shape == (2, 1, 64, 40) and mm.shape == (2, 1, GH, GW)
    # idea 56: pool setting; a zero-initialized MammoNet equals ChestNet's mean pool
    assert "pool" not in parse_sets([]) and parse_sets([]) == dict(SETTINGS) and "pool" not in parse_sets(["pool=null"])
    assert parse_sets(["pool=\"gated\""])["pool"] == "gated"
    for bad in (["pool=\"x\""], ["pool=true"], ["pool=1"]):
        try:
            parse_sets(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    reader = reader_module()
    widths = [4, 4, 4, 4, 4]
    net_plain = make_net(reader, widths, 0.0, True, pool=None)
    net_gated = make_net(reader, widths, 0.0, True, pool="gated")
    plain_state = net_plain.state_dict()
    gated_state = net_gated.state_dict()
    for k in plain_state:
        if k.startswith("trunk.") or k.startswith("head") or k.startswith("aux") or k == "drop":
            if k in gated_state:
                gated_state[k] = plain_state[k]
    net_gated.load_state_dict(gated_state)
    net_plain.eval()
    net_gated.eval()
    x = torch.rand(2, 1, 256, 192)
    z_plain = net_plain(x)
    z_gated = net_gated(x)
    assert z_plain.shape == (2, 1) and z_gated.shape == (2, 1)
    assert torch.allclose(z_plain, z_gated, atol=1e-5), ("zero-init gated pool should match mean pool", float((z_plain - z_gated).abs().max()))
    zp, ap = net_plain.forward_both(x)
    zg, ag = net_gated.forward_both(x)
    assert zp.shape == zg.shape and ap.shape == ag.shape, (zp.shape, zg.shape, ap.shape, ag.shape)
    assert torch.allclose(zp, zg, atol=1e-4), ("forward_both logits mismatch", float((zp - zg).abs().max()))
    assert torch.allclose(ap, ag, atol=1e-4), ("forward_both aux mismatch", float((ap - ag).abs().max()))
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        trunk_plain = {k: v for k, v in net_plain.state_dict().items() if k.startswith("trunk.")}
        torch.save({"state": trunk_plain}, tmp)
        net_gated2 = make_net(reader, widths, 0.0, True, pool="gated")
        att_before = {k: v.clone() for k, v in net_gated2.state_dict().items() if k.startswith("att_")}
        cbis_module().load_init(net_gated2, "chest-v103", say=lambda *a: None, path=tmp)
        trunk_gated2 = {k: v for k, v in net_gated2.state_dict().items() if k.startswith("trunk.")}
        assert len(trunk_gated2) == len(trunk_plain)
        for k in trunk_plain:
            assert torch.allclose(trunk_gated2[k], trunk_plain[k]), f"trunk mismatch at {k}"
        for k in att_before:
            assert torch.allclose(net_gated2.state_dict()[k], att_before[k]), f"att weights changed at {k}"
    finally:
        os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("gate", "prepare", "selftest", "outlines"):
        sub.add_parser(n)
    pe = sub.add_parser("prepare-extra")
    pe.add_argument("--only", default=",".join(EXTRAS))
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--step", type=int, required=True)
    r = sub.add_parser("read")
    r.add_argument("--versions", required=True)
    r.add_argument("--against", default="")
    r.add_argument("--only-in", type=int, default=None)
    pl = sub.add_parser("prepare-lesions")
    pl.add_argument("--limit", type=int, default=None)
    lo = sub.add_parser("locate")
    lo.add_argument("--version", type=int, required=True)
    lo.add_argument("--limit", type=int, default=None)
    lo.add_argument("--places", type=int, default=1)
    lo.add_argument("--apart", type=float, default=6)
    sc = sub.add_parser("score-crops")
    sc.add_argument("--version", type=int, required=True)
    sc.add_argument("--as", type=int, required=True, dest="as_version")
    sc.add_argument("--places", type=int, default=1)
    fm = sub.add_parser("found-masks")
    fm.add_argument("--places", type=int, required=True)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--folds", default="0,1,2,3,4")
    t.add_argument("--epochs", type=int)
    t.add_argument("--set", action="append", default=[], metavar="key=<json>")
    e = sub.add_parser("eval")
    e.add_argument("--version", type=int, required=True)
    x = ap.parse_args()
    if x.cmd == "gate":
        return gate()
    if x.cmd == "prepare":
        return prepare()
    if x.cmd == "train":
        return train(x.version, [int(f) for f in x.folds.split(",")], x.epochs, x.set)
    if x.cmd == "eval":
        return evaluate(x.version)
    if x.cmd == "outlines":
        return outlines()
    if x.cmd == "prepare-extra":
        return prepare_extra([e for e in x.only.split(",") if e])
    if x.cmd == "compare":
        return compare([int(v) for v in x.base.split(",")], [int(v) for v in x.new.split(",")], x.step)
    if x.cmd == "read":
        return read([int(v) for v in x.versions.split(",") if v], [int(v) for v in x.against.split(",") if v], x.only_in)
    if x.cmd == "prepare-lesions":
        return prepare_lesions(x.limit)
    if x.cmd == "locate":
        return locate(x.version, x.limit, x.places, x.apart)
    if x.cmd == "score-crops":
        return score_crops(x.version, x.as_version, x.places)
    if x.cmd == "found-masks":
        return found_masks(x.places)
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
