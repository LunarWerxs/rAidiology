"""Canal, recesses and foramina measured on axial slices (idea 30, docs/efficiency/README.md).

Data: Firat University Hospital's Medical Lumbar Spine 3D Axial MRI (Mendeley wjxkrkxhb8 v1, CC BY 4.0), read in
place under Datasets/public/lumbar-axial-stenosis-firat, never written to. 500 patients, 12,445 structures with a box,
a level and an anteroposterior diameter in mm. No published score exists; the first test number is the floor.

    python axial_stenosis.py index                       boxes, DICOM mapping, grade check, test list and five folds
    python axial_stenosis.py prepare                     the 80 x 80 3-channel crops (CPU, resumable)
    python axial_stenosis.py train --version N [--folds 0,1] [--epochs E] [--set key=<json>]
                                                         (idea 35: --set mirror=true, see below)
    python axial_stenosis.py test  --version N           opened once, only after an OPENS TEST verdict
    python axial_stenosis.py selftest                    CPU, seconds

The net regresses the diameter in mm (Huber, beta 1 mm, clipped at 0); the grade is the dataset's own cutoff on it.
The PNG is the DICOM's pixel grid at 8 bits, but the XML <size> and some PNGs differ from the DICOM rows x columns:
boxes are mapped from the PNG's own header size to the DICOM's rows x columns (foramen.map_box). An Enhanced MR file
(one file, many frames) gives its slice as the frame its PNG was made from (foramen.frame_of).

Optional setting `mirror` (idea 35; not in CFG, written to cfg.json only when true, read with cfg.get("mirror", False)):
in training each batch item is flipped left-right with probability 0.5 and its structure swapped (LLS<->RLS,
LFS<->RFS, CCS stays); every reading (stop set, held-out, test) is the mean of the plain prediction and the
prediction on the flipped crops with the structures swapped. Off, nothing changes and no random draw is spent.
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
sys.path.insert(0, HERE)
from core import ledger, paths  # noqa: E402
from foramen import (assign_split, confusion, dicom_info, frame_of, map_box, neighbours,  # noqa: E402
                     png_size, qwk_conf)

ROOT = os.environ.get("RAIDIOLOGY_AXIAL_ROOT",
                      "D:/NEWProjects/active/rAidiology/Datasets/public/lumbar-axial-stenosis-firat/"
                      "Medical_Lumbar_Spine_3D_Axial_MRI_Dataset V0.1")
OUT = os.path.join(paths.WORK, "eval", "spine_mri", "axial_stenosis")
LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
STRUCTS = ["CCS", "LLS", "RLS", "LFS", "RFS"]
GRADE_WORDS = {"normal": 0, "stenosis": 1, "severe_stenosis": 2}
NAME = re.compile(r"^(CCS|LLS|RLS|LFS|RFS)_(normal|stenosis|severe_stenosis)$")
FOLDS = 5
N_TEST = 100
CROP_MM, CROP_PX = 40.0, 80           # 0.5 mm per pixel
GATE_KAPPA, GATE_CCS_MAE = 0.75, 1.5
CFG = {"width": 32, "epochs": 40, "batch": 64, "lr": 2e-3, "weight_decay": 1e-4, "early_stop_frac": 0.15,
       "drop": 0.3, "shift_px": 4.0, "jitter": 0.05, "huber_beta": 1.0, "seed": 0}
OPTIONAL = {"mirror": False}          # settings kept out of CFG (v1's cfg.json is compared exactly); absent = default
SWAP = [0, 2, 1, 4, 3]                # STRUCTS index under a left-right flip: LLS<->RLS, LFS<->RFS, CCS stays
LEASE_GB = 1.5
MARKER = "test_opened.json"


def out(*p):
    return os.path.join(OUT, *p)


# ---- the grade rule ---------------------------------------------------------------------------------------

def grade_of(structure, d):
    """0 normal, 1 stenosis, 2 severe, by the dataset's own cutoffs on the diameter in mm."""
    if structure == "CCS":
        return 2 if d < 10.0 else (1 if d < 12.0 else 0)
    if structure in ("LFS", "RFS"):
        return 2 if d < 2.0 else (1 if d < 4.0 else 0)
    return 2 if d <= 3.0 else (1 if d < 5.0 else 0)       # LLS, RLS


def grades_of(structures, d):
    return np.array([grade_of(s, float(v)) for s, v in zip(structures, d)], int)


# ---- reading the data -------------------------------------------------------------------------------------

def level_of(text):
    t = (text or "").strip().upper().replace("\u2013", "-").replace("\u2014", "-").replace(" ", "")
    return LEVELS.index(t) if t in LEVELS else -1


def parse_voc(text, strays=None):
    """(frame width, height, [(structure, xml grade 0-2, diameter mm, level text, (x0, y0, x1, y1))]) from one XML.
    An object whose name is not <structure>_<grade> or that has no diameter is not returned; if `strays` (a
    Counter) is given its name is counted there."""
    root = ET.fromstring(text.strip())
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        m = NAME.match(name)
        dm = re.search(r"-?\d+(?:\.\d+)?", o.findtext("ap_diameter") or "")
        b = o.find("bndbox")
        if not m or not dm or b is None:
            if strays is not None:
                strays[(name or "(no name)") if not m else name + " (no diameter or box)"] += 1
            continue
        box = tuple(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((m.group(1), GRADE_WORDS[m.group(2)], float(dm.group(0)), (o.findtext("level") or "").strip(), box))
    return w, h, objs


def header_spacing(ds):
    if "PixelSpacing" in ds:
        return [float(v) for v in ds.PixelSpacing]
    try:
        return [float(v) for v in ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing]
    except Exception:
        return None


def sibling_spacing(files, rows, cols):
    import pydicom
    for f in files:
        ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
        sp = header_spacing(ds)
        if sp is not None and (int(ds.Rows), int(ds.Columns)) == (rows, cols):
            return sp
    return None


def build_index(root, say=print):
    import pydicom
    from collections import Counter
    loc = os.path.join(root, "Localization_and_Severity_Classification")
    strays, bad_level, rows, nopng = Counter(), Counter(), [], 0
    for pdir in sorted(d for d in os.listdir(loc) if os.path.isdir(os.path.join(loc, d))):
        patient = pdir
        dfiles = sorted(glob.glob(os.path.join(root, "DICOM", patient, "IM*.dcm")))
        for xp in sorted(glob.glob(os.path.join(loc, pdir, "IM*.xml"))):
            stem = os.path.basename(xp)[:-4]
            w, h, objs = parse_voc(open(xp, encoding="utf-8", errors="replace").read(), strays)
            if not objs:
                continue
            dn = os.path.join(root, "DICOM", patient, stem + ".dcm")
            if not os.path.exists(dn):
                say("no DICOM for %s/%s" % (patient, stem))
                continue
            ds = pydicom.dcmread(dn, force=True, stop_before_pixels=True)
            nr, nc = int(ds.Rows), int(ds.Columns)
            sp = header_spacing(ds) or sibling_spacing(dfiles, nr, nc)
            if sp is None:
                say("no pixel spacing for %s/%s: its structures are dropped" % (patient, stem))
                continue
            pn = os.path.join(loc, pdir, stem + ".png")
            frame = None
            if os.path.exists(pn):
                with open(pn, "rb") as fh:
                    frame = png_size(fh.read(24))
            if frame is None:
                nopng += 1
                frame = (w, h) if w >= 16 and h >= 16 else (nc, nr)
            for st, xg, d, lv, box in objs:
                li = level_of(lv)
                if li < 0:
                    bad_level[lv] += 1
                    continue
                m = map_box(box, frame, nr, nc)
                rows.append(dict(patient=patient, slice=stem, dicom=stem + ".dcm", level=li, structure=st,
                                 diameter_mm=d, grade=grade_of(st, d), xml_grade=xg, x0=m[0], y0=m[1], x1=m[2],
                                 y1=m[3], frame_w=frame[0], frame_h=frame[1], rows=nr, cols=nc,
                                 sp_row=sp[0], sp_col=sp[1]))
    return pd.DataFrame(rows), strays, bad_level, nopng


# ---- the split --------------------------------------------------------------------------------------------

def split_table(df, n_test=N_TEST, seed=0):
    """patient -> 'test' or '0'..'4'; stratified by each patient's worst grade over all its structures."""
    worst = df.groupby("patient")["grade"].max().to_dict()
    f = assign_split(worst, None, n_test, seed, FOLDS)
    return {p: ("test" if v == -1 else str(v)) for p, v in f.items()}


def check_split(df):
    per = df.groupby("patient")["fold"].nunique()
    if (per > 1).any():
        raise SystemExit("a patient is in two folds: %s" % list(per[per > 1].index[:5]))


def fold_int(v):
    return -1 if str(v) == "test" else int(v)


def cmd_index(a):
    os.makedirs(OUT, exist_ok=True)
    df, strays, bad_level, nopng = build_index(ROOT)
    fold = split_table(df)
    df["fold"] = df["patient"].map(fold)
    check_split(df)
    df.to_csv(out("index.csv"), index=False)
    n = len(df)
    ok = int((df.grade == df.xml_grade).sum())
    print("objects %d (expected 12445), patients %d" % (n, df.patient.nunique()))
    print("other object names (ignored): %s" % (dict(strays) or "none"))
    print("unreadable levels (dropped): %s" % (dict(bad_level) or "none"))
    print("grades reproduced from the diameter by the cutoff rule: %d of %d" % (ok, n))
    for _, r in df[df.grade != df.xml_grade].head(30).iterrows():
        print("  MISMATCH %s/%s %s level %d: %.2f mm, xml grade %d, rule %d"
              % (r.patient, r.slice, r.structure, r.level, r.diameter_mm, r.xml_grade, r.grade))
    print("PNG to DICOM: %d boxes needed a scale (frame != DICOM), %d slices had no readable PNG header"
          % (int(((df.frame_w != df.cols) | (df.frame_h != df.rows)).sum()), nopng))
    print("per structure (normal / stenosis / severe):")
    for s in STRUCTS:
        c = df[df.structure == s].grade.value_counts()
        print("  %s: %d / %d / %d" % (s, c.get(0, 0), c.get(1, 0), c.get(2, 0)))
    print("per grade:  " + "  ".join("%d: %d" % (g, k) for g, k in df.grade.value_counts().sort_index().items()))
    print("per fold (patients / objects):  " + "  ".join(
        "%s: %d / %d" % (f, df[df.fold == f].patient.nunique(), (df.fold == f).sum())
        for f in ["0", "1", "2", "3", "4", "test"]))
    worst = df.groupby("patient")["grade"].max()
    print("patients by worst grade, per fold:")
    for f in ["0", "1", "2", "3", "4", "test"]:
        c = worst[[p for p in worst.index if fold[p] == f]].value_counts()
        print("  %s: %s" % (f, " ".join("%d:%d" % (g, c.get(g, 0)) for g in (0, 1, 2))))
    return 0


# ---- crops ------------------------------------------------------------------------------------------------

def crop_from(slices, cx, cy, sp_row, sp_col):
    """(3, 80, 80) uint8 from three slices (3, H, W): 40 mm around (cx, cy) in pixels, resampled to 0.5 mm
    (bilinear, edge pixels repeated), 1st to 99th percentile of the whole crop to 0..255."""
    from scipy.ndimage import map_coordinates
    k = (np.arange(CROP_PX) - (CROP_PX - 1) / 2.0) * (CROP_MM / CROP_PX)
    yy, xx = np.meshgrid(cy + k / sp_row, cx + k / sp_col, indexing="ij")
    c = np.stack([map_coordinates(s.astype(np.float32), [yy, xx], order=1, mode="nearest") for s in slices])
    lo, hi = np.percentile(c, [1, 99])
    if hi <= lo:
        return np.zeros(c.shape, np.uint8)
    return np.clip((c - lo) / (hi - lo) * 255.0, 0, 255).round().astype(np.uint8)


KEY_COLS = ["patient", "slice", "level", "structure", "diameter_mm"]


def cmd_prepare(a):
    import pydicom
    idx = pd.read_csv(out("index.csv"), dtype={"patient": str, "fold": str})
    n = len(idx)
    table = idx[["patient", "slice", "level", "structure", "diameter_mm", "grade", "fold"]].copy()
    table.insert(0, "row", np.arange(n))
    mm_path, done_path, rows_path = out("crops.u8"), out("prepare_done.txt"), out("crops_rows.csv")
    fresh = True
    if os.path.exists(rows_path) and os.path.exists(mm_path) and os.path.getsize(mm_path) == n * 3 * CROP_PX * CROP_PX:
        old = pd.read_csv(rows_path, dtype={"patient": str})
        fresh = not (len(old) == n and (old[KEY_COLS].astype(str).values == table[KEY_COLS].astype(str).values).all())
    if fresh:
        if os.path.exists(done_path):
            os.remove(done_path)
            print("the index changed: crops restart")
        np.memmap(mm_path, np.uint8, "w+", shape=(n, 3, CROP_PX, CROP_PX)).flush()
    table.to_csv(rows_path, index=False)
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    mm = np.memmap(mm_path, np.uint8, "r+", shape=(n, 3, CROP_PX, CROP_PX))
    todo = [p for p in sorted(idx.patient.unique()) if p not in done]
    print("crops: %d structures, %d patients left of %d" % (n, len(todo), idx.patient.nunique()))
    for k, p in enumerate(todo):
        infos, multi, cache = {}, set(), {}
        for f in sorted(glob.glob(os.path.join(ROOT, "DICOM", p, "IM*.dcm"))):
            ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            base = os.path.basename(f)
            if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
                multi.add(base)
            else:
                infos[base] = dicom_info(ds)

        def pixels(base):
            if base not in cache:
                cache[base] = pydicom.dcmread(os.path.join(ROOT, "DICOM", p, base), force=True).pixel_array
            return cache[base]

        for i in idx.index[idx.patient == p]:
            r = idx.loc[i]
            if r.dicom in multi:
                # an Enhanced MR file holds the whole stack: the labelled slice is the frame its PNG was made from,
                # its neighbours the frames beside it (as foramen.prepare)
                st = pixels(r.dicom)
                png = os.path.join(ROOT, "Localization_and_Severity_Classification", p, r.slice + ".png")
                fr = frame_of(st, open(png, "rb").read())
                three = [st[max(fr - 1, 0)], st[fr], st[min(fr + 1, len(st) - 1)]]
            else:
                prev, cur, nxt = neighbours(infos, r.dicom)
                three = [pixels(prev), pixels(cur), pixels(nxt)]
            mm[i] = crop_from(three, (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0, r.sp_row, r.sp_col)
        mm.flush()
        with open(done_path, "a") as f:
            f.write(p + "\n")
        if (k + 1) % 50 == 0:
            print("  %d / %d patients" % (k + 1, len(todo)), flush=True)
    print("crops done: %s" % mm_path)
    return 0


# ---- scores -----------------------------------------------------------------------------------------------

def stats_from(C, E):
    """Per-structure numbers from C (5, 3, 3) grade confusions and E (5, 2) = (sum of absolute errors, count)."""
    res = {}
    for i, s in enumerate(STRUCTS):
        n = C[i].sum()
        res[s] = dict(n=int(n),
                      mae=float(E[i, 0] / E[i, 1]) if E[i, 1] > 0 else float("nan"),
                      kappa=float(qwk_conf(C[i])) if n > 0 else float("nan"),
                      acc=float(100.0 * np.trace(C[i]) / n) if n > 0 else float("nan"),
                      within1=float(100.0 * (n - C[i][0, 2] - C[i][2, 0]) / n) if n > 0 else float("nan"))
    ks = [res[s]["kappa"] for s in STRUCTS if res[s]["n"] > 0]
    res["mean_kappa"] = float(np.mean(ks)) if ks else float("nan")
    return res


def tables(df, patients):
    """Per-patient C (P, 5, 3, 3) and E (P, 5, 2) from a frame with structure, y_mm, p_mm, y_grade, p_grade."""
    pos = {p: i for i, p in enumerate(patients)}
    C, E = np.zeros((len(patients), 5, 3, 3)), np.zeros((len(patients), 5, 2))
    pi = df.patient.map(pos).to_numpy()
    si = df.structure.map({s: i for i, s in enumerate(STRUCTS)}).to_numpy()
    np.add.at(C, (pi, si, df.y_grade.to_numpy(int), df.p_grade.to_numpy(int)), 1)
    np.add.at(E, (pi, si, 0), np.abs(df.y_mm.to_numpy() - df.p_mm.to_numpy()))
    np.add.at(E, (pi, si, 1), 1)
    return C, E


def metrics(df):
    C, E = tables(df.assign(patient="all"), ["all"])
    return stats_from(C[0], E[0])


def boot_metrics(df, n=2000, seed=0):
    """metrics() with a patient-clustered bootstrap 95% interval: {structure: {metric: (est, lo, hi)}}, plus mean_kappa."""
    pats = sorted(df.patient.unique())
    C, E = tables(df, pats)
    est = stats_from(C.sum(0), E.sum(0))
    rng = np.random.RandomState(seed)
    runs = [stats_from(C[ix].sum(0), E[ix].sum(0)) for ix in (rng.randint(0, len(pats), len(pats)) for _ in range(n))]

    def ci(get):
        v = np.array([get(r) for r in runs], float)
        return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))

    res = {}
    for s in STRUCTS:
        res[s] = {k: (est[s][k],) + ci(lambda r, s=s, k=k: r[s][k]) for k in ("mae", "kappa", "acc", "within1")}
        res[s]["n"] = est[s]["n"]
    res["mean_kappa"] = (est["mean_kappa"],) + ci(lambda r: r["mean_kappa"])
    res["patients"] = len(pats)
    return res


def verdict_of(mean_kappa, ccs_mae):
    ok = mean_kappa >= GATE_KAPPA and ccs_mae < GATE_CCS_MAE
    return "OPENS TEST" if ok else "STAYS ON DEVELOPMENT"


def verdict_line(mean_kappa, ccs_mae):
    return "Verdict (idea 30 step 1, mean kappa at least 0.75 and canal error under 1.5 mm): %s" % verdict_of(mean_kappa, ccs_mae)


def table_lines(m):
    lines = ["%-4s %6s %9s %8s %7s %11s" % ("", "n", "MAE mm", "kappa", "acc %", "within 1 %")]
    for s in STRUCTS:
        r = m[s]
        lines.append("%-4s %6d %9.2f %8.3f %7.1f %11.1f" % (s, r["n"], r["mae"], r["kappa"], r["acc"], r["within1"]))
    lines.append("mean of the five kappas: %.3f" % m["mean_kappa"])
    return lines


def boot_lines(b):
    lines = []
    for s in STRUCTS:
        r = b[s]
        lines.append("%-4s n %5d  MAE %.2f (%.2f to %.2f)  kappa %.3f (%.3f to %.3f)  acc %.1f (%.1f to %.1f)  "
                     "within 1 %.1f (%.1f to %.1f)" % ((s, r["n"]) + r["mae"] + r["kappa"] + r["acc"] + r["within1"]))
    lines.append("mean kappa %.3f (95%% %.3f to %.3f), %d patients" % (b["mean_kappa"] + (b["patients"],)))
    return lines


def may_open_test(version, root=None):
    """(ok, reason): the version's verdict must be OPENS TEST, complete over all folds, and no marker may exist."""
    root = root or OUT
    m = os.path.join(root, MARKER)
    if os.path.exists(m):
        return False, "the test list was already opened (%s)" % m
    v = os.path.join(root, "v%d" % version, "verdict.json")
    if not os.path.exists(v):
        return False, "v%d has no verdict (run train over all five folds first)" % version
    d = json.load(open(v))
    if not d.get("complete") or d.get("verdict") != "OPENS TEST":
        return False, "v%d did not earn it: %s" % (version, d.get("verdict"))
    return True, ""


# ---- the network ------------------------------------------------------------------------------------------

def make_net(cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self, width, drop):
            super().__init__()
            ch = [3, width, width * 2, width * 4, width * 8]
            layers = []
            for i in range(4):
                layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True),
                           nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)]
            self.body = nn.Sequential(*layers)
            self.drop = nn.Dropout(drop)
            self.head = nn.Linear(ch[-1] + len(STRUCTS) + len(LEVELS), 1)

        def forward(self, x, structure, level):
            f = F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)
            oh = torch.cat([F.one_hot(structure, len(STRUCTS)), F.one_hot(level, len(LEVELS))], 1).to(f.dtype)
            return self.head(self.drop(torch.cat([f, oh], 1))).squeeze(1)

    return Net(cfg["width"], cfg["drop"])


def augment(x, rng, shift_px, jitter):
    """A random shift of up to shift_px pixels in each direction and brightness/contrast jitter; no flip here (a
    left-right flip swaps left and right structures: that is the `mirror` setting, see mirror_batch)."""
    import torch
    import torch.nn.functional as F
    n, dev = x.shape[0], x.device
    u = lambda lo, hi, *s: torch.tensor(rng.uniform(lo, hi, s), device=dev, dtype=torch.float32)  # noqa: E731
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    lim = 2.0 * shift_px / x.shape[-1]
    theta[:, :, 2] = u(-lim, lim, n, 2)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    # x is in (pixel / 255 - 0.5) / 0.25 units: a 5% brightness step is 0.05 / 0.25 of them
    return x * u(1 - jitter, 1 + jitter, n, 1, 1, 1) + u(-jitter, jitter, n, 1, 1, 1) / 0.25


def prep(u8, dev):
    return (u8.to(dev).float() / 255.0 - 0.5) / 0.25


def swap_structs(st):
    """The structure indices as seen in a left-right flipped crop."""
    import torch
    return torch.as_tensor(SWAP, device=st.device)[st]


def mirror_batch(xb, st, rng, on):
    """With `on`, flip each item left-right with probability 0.5 (one draw per item from rng) and swap its structure;
    off, the inputs come back as they are and rng is not touched."""
    import torch
    if not on:
        return xb, st
    f = torch.as_tensor(rng.rand(len(xb)) < 0.5, device=xb.device)
    return torch.where(f[:, None, None, None], torch.flip(xb, [-1]), xb), torch.where(f, swap_structs(st), st)


def predict(net, X, st, lv, dev, batch=256, mirror=False):
    """Diameters in mm, clipped at 0. With `mirror`: the mean of the plain reading and the reading of the left-right
    flipped crops with the structures swapped."""
    import torch
    if mirror:
        return (predict(net, X, st, lv, dev, batch) + predict(net, torch.flip(X, [-1]), swap_structs(st), lv, dev, batch)) / 2.0
    net.eval()
    res = []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            xb = prep(X[a:a + batch], dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z = net(xb, st[a:a + batch], lv[a:a + batch])
            res.append(z.float().cpu().numpy())
    return np.clip(np.concatenate(res), 0.0, None)


def setup_device():
    import torch
    if os.environ.get("RAIDIOLOGY_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        from core import card
        gb = float(os.environ.get("RAIDIOLOGY_CARD_GB", LEASE_GB))
        # a short job (core/card.py's short lane): v1's five folds took 791 s beside two other jobs (2026-10-05);
        # RAIDIOLOGY_CARD_SHORT_MIN, when set, decides instead
        card.lease(gb, "axial_stenosis", short_min=None if os.environ.get("RAIDIOLOGY_CARD_SHORT_MIN") else 20)
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, gb * 2 ** 30 / total), torch.cuda.current_device())
        return torch.device("cuda")
    return torch.device("cpu")


def load_data():
    import torch
    rows = pd.read_csv(out("crops_rows.csv"), dtype={"patient": str, "fold": str})
    rows["fold"] = rows["fold"].map(fold_int)
    done = set(open(out("prepare_done.txt")).read().split()) if os.path.exists(out("prepare_done.txt")) else set()
    if set(rows.patient) - done:
        raise SystemExit("prepare has not finished (%d patients left)" % len(set(rows.patient) - done))
    mm = np.memmap(out("crops.u8"), np.uint8, "r", shape=(len(rows), 3, CROP_PX, CROP_PX))
    return rows, torch.from_numpy(np.ascontiguousarray(mm))


def tensors(rows, X, dev, ix):
    import torch
    st = torch.tensor(rows.structure.map({s: i for i, s in enumerate(STRUCTS)}).to_numpy()[ix], dtype=torch.long, device=dev)
    lv = torch.tensor(rows.level.to_numpy()[ix], dtype=torch.long, device=dev)
    y = torch.tensor(rows.diameter_mm.to_numpy()[ix], dtype=torch.float32, device=dev)
    return X[torch.as_tensor(ix)], st, lv, y


def train_fold(rows, X, k, cfg, dev, say=print):
    import copy
    import torch
    import torch.nn.functional as F
    seed = cfg["seed"] * 100 + k
    mirror = bool(cfg.get("mirror", False))
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    pats = sorted(rows[(rows.fold >= 0) & (rows.fold != k)].patient.unique())
    rng.shuffle(pats)
    n_stop = max(1, int(round(len(pats) * cfg["early_stop_frac"])))
    stop = set(pats[:n_stop])
    dev_rows = (rows.fold >= 0) & (rows.fold != k)
    tr = np.where(dev_rows & ~rows.patient.isin(stop))[0]
    sv = np.where(dev_rows & rows.patient.isin(stop))[0]
    ho = np.where(rows.fold == k)[0]
    Xt, stt, lt, yt = tensors(rows, X, dev, tr)
    Xs, sts, ls, _ = tensors(rows, X, dev, sv)
    ys_np = rows.diameter_mm.to_numpy()[sv]
    net = make_net(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * int(np.ceil(len(tr) / cfg["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.15)
    best, best_mae, best_ep = None, float("inf"), -1
    for ep in range(cfg["epochs"]):
        net.train()
        perm = torch.as_tensor(rng.permutation(len(tr)), device=dev)
        tot = 0.0
        for a in range(0, len(tr), cfg["batch"]):
            b = perm[a:a + cfg["batch"]]
            if len(b) < 2:
                continue
            xb = augment(prep(Xt[b.cpu()], dev), rng, cfg["shift_px"], cfg["jitter"])
            xb, sb = mirror_batch(xb, stt[b], rng, mirror)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z = net(xb, sb, lt[b])
            loss = F.smooth_l1_loss(z.float(), yt[b], beta=cfg["huber_beta"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        mae = float(np.abs(predict(net, Xs.cpu(), sts, ls, dev, mirror=mirror) - ys_np).mean())
        if mae < best_mae:
            best, best_mae, best_ep = copy.deepcopy(net.state_dict()), mae, ep
        say("  fold %d epoch %2d loss %.4f stop-set MAE %.3f mm%s" % (k, ep, tot / max(len(tr), 1), mae,
                                                                     "  *" if ep == best_ep else ""))
    net.load_state_dict(best)
    Xh, sth, lh, _ = tensors(rows, X, dev, ho)
    return net, predict(net, Xh, sth, lh, dev, mirror=mirror), ho, best_mae, best_ep


def pred_frame(rows, ix, p_mm, fold):
    d = rows.iloc[ix][["row", "patient", "structure", "level", "fold"]].copy()
    d["fold"] = fold
    d["y_mm"] = rows.diameter_mm.to_numpy()[ix]
    d["p_mm"] = p_mm
    d["y_grade"] = grades_of(d.structure, d.y_mm)
    d["p_grade"] = grades_of(d.structure, d.p_mm)
    return d


def load_oof(vdir):
    parts = [pd.read_csv(os.path.join(vdir, f), dtype={"patient": str}) for f in sorted(os.listdir(vdir))
             if re.match(r"oof_fold\d\.csv$", f)]
    return pd.concat(parts, ignore_index=True) if parts else None


def jnum(o):
    if isinstance(o, dict):
        return {k: jnum(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jnum(v) for v in o]
    return float(o) if isinstance(o, (np.floating,)) else o


def parse_settings(epochs, sets):
    """CFG with --epochs and --set key=<json> applied. An OPTIONAL key is written only when it differs from its
    default, so a run without it keeps CFG exactly."""
    cfg = dict(CFG)
    if epochs:
        cfg["epochs"] = epochs
    for s in sets or []:
        key, val = s.split("=", 1)
        if key not in cfg and key not in OPTIONAL:
            raise SystemExit("unknown setting %s (known: %s)" % (key, ", ".join(list(cfg) + list(OPTIONAL))))
        v = json.loads(val)
        if key in OPTIONAL:
            if v != OPTIONAL[key]:
                cfg[key] = v
            else:
                cfg.pop(key, None)
        else:
            cfg[key] = v
    return cfg


def cmd_train(a):
    import torch
    cfg = parse_settings(a.epochs, a.set)
    vdir = out("v%d" % a.version)
    os.makedirs(vdir, exist_ok=True)
    cfg_path = os.path.join(vdir, "cfg.json")
    if os.path.exists(cfg_path) and json.load(open(cfg_path)) != cfg:
        raise SystemExit("v%d was trained with other settings (%s); take a new version" % (a.version, cfg_path))
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    folds = [int(f) for f in a.folds.split(",")] if a.folds else list(range(FOLDS))
    dev = setup_device()
    rows, X = load_data()
    with ledger.run("experiment", "axial_stenosis v%d" % a.version) as led:
        led.settings_from(dict(cfg, folds=folds, version=a.version))
        for k in folds:
            oof = os.path.join(vdir, "oof_fold%d.csv" % k)
            if os.path.exists(oof):
                print("fold %d: already done" % k)
                continue
            net, p, ho, bm, be = train_fold(rows, X, k, cfg, dev)
            torch.save(net.state_dict(), os.path.join(vdir, "fold%d.pt" % k))
            d = pred_frame(rows, ho, p, k)
            d.to_csv(oof, index=False)
            print("fold %d: best stop-set MAE %.3f mm at epoch %d; held-out MAE %.3f mm"
                  % (k, bm, be, float(np.abs(d.y_mm - d.p_mm).mean())))
        df = load_oof(vdir)
        if df is None:
            return 1
        df.to_csv(os.path.join(vdir, "oof.csv"), index=False)
        m = metrics(df)
        have = sorted(int(f) for f in df.fold.unique())
        complete = have == list(range(FOLDS))
        print("out-of-fold (%d structures, %d patients, folds %s):" % (len(df), df.patient.nunique(), ",".join(map(str, have))))
        for line in table_lines(m):
            print(line)
        if not complete:
            print("only %d of %d folds are done: this is not final" % (len(have), FOLDS))
        v = verdict_of(m["mean_kappa"], m["CCS"]["mae"])
        json.dump(jnum(dict(verdict=v, complete=complete, folds=have, ccs_mae=m["CCS"]["mae"], **m)),
                  open(os.path.join(vdir, "verdict.json"), "w"), indent=1)
        for s in STRUCTS:
            led.number("oof_mae_%s" % s.lower(), m[s]["mae"], "mm").number("oof_kappa_%s" % s.lower(), m[s]["kappa"])
        led.number("oof_mean_kappa", m["mean_kappa"]).number("oof_n", len(df))
        print(verdict_line(m["mean_kappa"], m["CCS"]["mae"]))
    return 0


def cmd_test(a):
    import torch
    ok, why = may_open_test(a.version)
    if not ok:
        raise SystemExit("refused: " + why)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(version=a.version, at=pd.Timestamp.now().isoformat()),
              open(out(MARKER), "w"))                         # first: a crash after this still counts as opened
    vdir = out("v%d" % a.version)
    cfg = json.load(open(os.path.join(vdir, "cfg.json")))
    dev = setup_device()
    rows, X = load_data()
    ix = np.where(rows.fold == -1)[0]
    Xe, se, le, _ = tensors(rows, X, dev, ix)
    ps = []
    with ledger.run("experiment", "axial_stenosis v%d test" % a.version) as led:
        led.settings_from(dict(cfg, version=a.version))
        for k in range(FOLDS):
            net = make_net(cfg).to(dev)
            net.load_state_dict(torch.load(os.path.join(vdir, "fold%d.pt" % k), map_location=dev))
            ps.append(predict(net, Xe.cpu(), se, le, dev, mirror=bool(cfg.get("mirror", False))))
        d = pred_frame(rows, ix, np.mean(ps, 0), "test")
        d.to_csv(os.path.join(vdir, "test.csv"), index=False)
        b = boot_metrics(d)
        json.dump(jnum(b), open(os.path.join(vdir, "test.json"), "w"), indent=1)
        for s in STRUCTS:
            led.number("test_mae_%s" % s.lower(), b[s]["mae"][0], "mm").number("test_kappa_%s" % s.lower(), b[s]["kappa"][0])
        led.number("test_mean_kappa", b["mean_kappa"][0]).number("test_n", len(d))
        print("test (%d structures, %d patients), the five fold nets averaged:" % (len(d), b["patients"]))
        for line in boot_lines(b):
            print(line)
        print("no published score: this is the line's floor")
    return 0


# ---- selftest ---------------------------------------------------------------------------------------------

VOC = """<annotation><size><width>512</width><height>512</height></size>
<object><ap_diameter>14.06mm</ap_diameter><name>CCS_normal</name><bndbox><xmin>249</xmin><ymin>244</ymin><xmax>263</xmax><ymax>308</ymax></bndbox><level>L1-L2</level></object>
<object><ap_diameter>1.5mm</ap_diameter><name>RFS_severe_stenosis</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox><level>L5-S1</level></object>
<object><name>classes</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
</annotation>"""


def selftest():
    from sklearn.metrics import cohen_kappa_score
    bad = []

    def check(name, ok):
        print("%-56s %s" % (name, "ok" if ok else "FAIL"))
        if not ok:
            bad.append(name)

    _selftest_parsing(check)
    _selftest_split(check)
    _selftest_scores(check, cohen_kappa_score)
    _selftest_verdicts(check)
    _selftest_crops(check)
    x = _selftest_net(check)
    _selftest_mirror(check)
    _selftest_settings(check)
    _selftest_mirror_batch(check, x)
    print("selftest: %s" % ("FAILED: " + ", ".join(bad) if bad else "all passed"))
    return 1 if bad else 0


def _selftest_parsing(check):
    from collections import Counter
    stray = Counter()
    w, h, objs = parse_voc(VOC, stray)
    check("voc: structure, grade, diameter, level, box", (w, h) == (512, 512) and len(objs) == 2 and
          objs[0] == ("CCS", 0, 14.06, "L1-L2", (249.0, 244.0, 263.0, 308.0)) and
          objs[1] == ("RFS", 2, 1.5, "L5-S1", (10.0, 20.0, 30.0, 60.0)))
    check("voc: a stray name is counted, not returned", sum(stray.values()) == 1 and len(objs) == 2)
    check("levels", level_of("L5-S1") == 4 and level_of(" l1\u2013l2 ") == 0 and level_of("L6-L7") == -1)
    g = grade_of
    check("grade: canal 9.99 / 10 / 11.99 / 12", [g("CCS", v) for v in (9.99, 10.0, 11.99, 12.0)] == [2, 1, 1, 0])
    check("grade: foramen 1.99 / 2 / 3.99 / 4", [g("LFS", v) for v in (1.99, 2.0, 3.99, 4.0)] == [2, 1, 1, 0]
          and [g("RFS", v) for v in (1.99, 2.0, 3.99, 4.0)] == [2, 1, 1, 0])
    check("grade: recess 3 / 3.01 / 4.99 / 5", [g("LLS", v) for v in (3.0, 3.01, 4.99, 5.0)] == [2, 1, 1, 0]
          and [g("RLS", v) for v in (3.0, 3.01, 4.99, 5.0)] == [2, 1, 1, 0])
    check("grade: closed (0 mm) is severe", all(g(s, 0.0) == 2 for s in STRUCTS))
    check("box 512 -> 288 DICOM", map_box((256, 128, 512, 256), (512, 512), 288, 288) == (144.0, 72.0, 288.0, 144.0))
    check("box 512 PNG -> 224 rows x 288 cols", map_box((512, 512, 512, 512), (512, 512), 224, 288) == (288.0, 224.0, 288.0, 224.0))


def _selftest_split(check):
    rng = np.random.RandomState(3)
    pats =["%04d" % i for i in range(1, 501)]
    rows = []
    for p in pats:
        wg = int(rng.choice(3, p=[.5, .3, .2]))
        for i, s in enumerate(STRUCTS):
            rows.append(dict(patient=p, structure=s, grade=wg if i == 0 else 0))
    sdf = pd.DataFrame(rows)
    f = split_table(sdf)
    sdf["fold"] = sdf.patient.map(f)
    test = [p for p, v in f.items() if v == "test"]
    check("split: 100 test, 400 in five folds, every patient once", len(test) == 100 and len(f) == 500 and
          sorted(set(f.values())) == ["0", "1", "2", "3", "4", "test"] and
          all(sum(v == str(k) for v in f.values()) == 80 for k in range(5)))
    check("split: no patient in two folds, none in test and a fold", sdf.groupby("patient").fold.nunique().max() == 1)
    worst = sdf.groupby("patient").grade.max()
    check("split: test stratified by worst grade", abs((worst[test] == 2).mean() - (worst == 2).mean()) < 0.05)
    check("split: seeded", split_table(sdf) == f and split_table(sdf, seed=1) != f)
    try:
        check_split(pd.DataFrame({"patient": ["a", "a"], "fold": ["0", "1"]}))
        check("split: a patient in two folds is refused", False)
    except SystemExit:
        check("split: a patient in two folds is refused", True)


def _selftest_scores(check, cohen_kappa_score):
    # scores
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    p = np.array([0, 1, 1, 2, 2, 2, 0, 0, 0, 0])
    check("kappa equals sklearn's", abs(qwk_conf(confusion(y, p, 3)) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-9)
    yy = np.array([3.0, 8.0, 11.0, 13.0])
    pp = np.array([4.0, 8.0, 9.0, 13.5])
    d = pd.DataFrame(dict(patient=["a", "a", "b", "b"], structure="CCS", y_mm=yy, p_mm=pp,
                          y_grade=grades_of(["CCS"] * 4, yy), p_grade=grades_of(["CCS"] * 4, pp)))
    m = metrics(d)
    check("MAE on hand-made arrays", abs(m["CCS"]["mae"] - (1 + 0 + 2 + 0.5) / 4) < 1e-12 and m["LFS"]["n"] == 0)
    check("accuracy and within one on hand-made arrays", m["CCS"]["acc"] == 75.0 and m["CCS"]["within1"] == 100.0)
    d2 = d.assign(patient=["a", "b", "c", "d"])
    b = boot_metrics(d2, n=100)
    check("bootstrap holds the estimate", b["CCS"]["mae"][1] <= b["CCS"]["mae"][0] <= b["CCS"]["mae"][2] and
          abs(b["CCS"]["mae"][0] - m["CCS"]["mae"]) < 1e-12 and b["patients"] == 4)


def _selftest_verdicts(check):
    # verdict and the test marker
    check("verdict at the gate edges", verdict_of(0.75, 1.49) == "OPENS TEST" and verdict_of(0.7499, 1.0) == "STAYS ON DEVELOPMENT"
          and verdict_of(0.9, 1.5) == "STAYS ON DEVELOPMENT" and verdict_of(0.9, 1.4999) == "OPENS TEST")
    check("verdict line wording", verdict_line(0.9, 1.0) ==
          "Verdict (idea 30 step 1, mean kappa at least 0.75 and canal error under 1.5 mm): OPENS TEST"
          and verdict_line(0.5, 1.0).endswith(": STAYS ON DEVELOPMENT"))
    with tempfile.TemporaryDirectory() as t:
        check("test refused: no verdict", not may_open_test(1, t)[0])
        os.makedirs(os.path.join(t, "v1"))
        vp = os.path.join(t, "v1", "verdict.json")
        json.dump(dict(verdict="STAYS ON DEVELOPMENT", complete=True), open(vp, "w"))
        check("test refused: stays on development", not may_open_test(1, t)[0])
        json.dump(dict(verdict="OPENS TEST", complete=False), open(vp, "w"))
        check("test refused: not all folds", not may_open_test(1, t)[0])
        json.dump(dict(verdict="OPENS TEST", complete=True), open(vp, "w"))
        check("test allowed: opens, complete, no marker", may_open_test(1, t)[0])
        json.dump({}, open(os.path.join(t, MARKER), "w"))
        check("test refused: a marker exists", not may_open_test(1, t)[0])


def _selftest_crops(check):
    # crops and the net
    img = np.tile(np.arange(200, dtype=np.float32), (200, 1))
    c = crop_from([img, img, img], 100.0, 100.0, 0.5, 0.5)
    check("crop: 3 x 80 x 80 uint8 over 0..255", c.shape == (3, 80, 80) and c.dtype == np.uint8 and c.min() == 0 and c.max() == 255)
    infos = {"a": (9, 9, (1.0,), 3.0, 1), "b": (9, 9, (1.0,), 1.0, 2), "c": (9, 9, (1.0,), 2.0, 3)}
    check("neighbours by position, ends repeat", neighbours(infos, "c") == ("b", "c", "a") and neighbours(infos, "b") == ("b", "b", "c"))


def _selftest_net(check):
    import torch
    net = make_net(dict(width=8, drop=0.3))
    x = torch.randn(4, 3, CROP_PX, CROP_PX)
    z = net(x, torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 4]))
    check("net: one output per row", tuple(z.shape) == (4,))
    pr = predict(net, torch.randint(0, 255, (5, 3, CROP_PX, CROP_PX), dtype=torch.uint8), torch.zeros(5, dtype=torch.long),
                 torch.zeros(5, dtype=torch.long), torch.device("cpu"))
    check("predict: mm, clipped at 0", pr.shape == (5,) and (pr >= 0).all())
    xa = augment(x, np.random.RandomState(0), 4.0, 0.05)
    check("augment keeps the shape", tuple(xa.shape) == tuple(x.shape))
    return x


def _selftest_mirror(check):
    import torch
    # idea 35: mirror
    sw = swap_structs(torch.arange(5)).tolist()
    check("mirror: swap is an involution, CCS fixed, sides exchanged",
          swap_structs(torch.as_tensor(sw)).tolist() == [0, 1, 2, 3, 4] and sw[0] == 0 and
          [STRUCTS[i] for i in sw] == ["CCS", "RLS", "LLS", "RFS", "LFS"])
    tiny = make_net(dict(width=4, drop=0.3))
    Xm = torch.randint(0, 255, (4, 3, CROP_PX, CROP_PX), dtype=torch.uint8)
    sm, lm = torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 4])
    both = (predict(tiny, Xm, sm, lm, torch.device("cpu")) +
            predict(tiny, torch.flip(Xm, [-1]), swap_structs(sm), lm, torch.device("cpu"))) / 2.0
    check("mirror: predict is the mean of plain and flipped-swapped",
          np.allclose(predict(tiny, Xm, sm, lm, torch.device("cpu"), mirror=True), both, atol=1e-5))


def _selftest_settings(check):
    c0 = parse_settings(None, None)
    check("settings: no --set is exactly CFG, no mirror key", c0 == CFG and "mirror" not in c0)
    c1 = parse_settings(None, ["mirror=true"])
    check("settings: --set mirror=true is written", c1.get("mirror") is True and {k: v for k, v in c1.items() if k != "mirror"} == CFG)
    check("settings: --set mirror=false leaves the key out", parse_settings(None, ["mirror=false"]) == CFG)


def _selftest_mirror_batch(check, x):
    import torch
    r0 = np.random.RandomState(5)
    s0 = r0.get_state()[1].copy(), r0.get_state()[2]
    xo, so = mirror_batch(x, torch.tensor([0, 1, 2, 3]), r0, False)
    check("mirror off: no rng draw, batch unchanged", (r0.get_state()[1] == s0[0]).all() and r0.get_state()[2] == s0[1]
          and xo is x and so.tolist() == [0, 1, 2, 3])
    xm, smb = mirror_batch(x, torch.tensor([0, 1, 2, 3]), np.random.RandomState(5), True)
    fl = [not torch.equal(xm[i], x[i]) for i in range(4)]
    check("mirror on: flipped items have swapped structures", all(
        (smb[i].item() == SWAP[i] and torch.equal(xm[i], torch.flip(x[i], [-1]))) if fl[i] else
        (smb[i].item() == i and torch.equal(xm[i], x[i])) for i in range(4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("prepare")
    s = sub.add_parser("train")
    s.add_argument("--version", type=int, required=True)
    s.add_argument("--folds", help="e.g. 0,1 (default all five)")
    s.add_argument("--epochs", type=int)
    s.add_argument("--set", action="append", help="key=<json>, e.g. --set drop=0.2")
    s = sub.add_parser("test")
    s.add_argument("--version", type=int, required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"index": cmd_index, "prepare": cmd_prepare, "train": cmd_train, "test": cmd_test,
            "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
