"""Foraminal stenosis, graded 0-3, on the expert boxes (idea 28, docs/efficiency/README.md).

Data: LSS-MRI-AISSLab, sagittal T2 (LICENSING.md). Read in place from its unpacked folder (ROOT: DICOM/, Foramina_Detection/, Segmentation/).
To beat: Phadke et al., J Imaging 2026, 12:388 (QWK 0.813, 0.769 to 0.847; accuracy 75.3; within one 98.3;
a locked test set of 94 patients). A from-scratch CNN, no pretrained weights of any kind, on 2.5D crops of
the expert boxes with an ordinal head (3 cumulative logits, P(grade > k)).

    python foramen.py index   [--test-list FILE]     boxes, DICOM mapping, test list and five folds
    python foramen.py prepare                        the 64 x 64 3-channel crops (CPU, resumable)
    python foramen.py train --version N [--folds 0,1] [--epochs E] [--set key=<json>]
                                     idea 34: --set 'init="disc-v133"' starts the body from our own disc grader
                                     idea 78: --set ema=0.995 keeps an exponential moving average of the weights; the stop set picks among the averaged weights
    python foramen.py compare --base 1,11 --new 2,12  idea 34: two-seed gain over the base, KEPT or NOT KEPT, the gate
    python foramen.py test  --version N              opened once, only after an OPENS TEST verdict
    python foramen.py test  --versions 2,12          idea 34: the ten nets (five folds x two versions), after compare
    python foramen.py selftest                       CPU, seconds

How the PNGs relate to the DICOMs (read 2026-10-05 over all 1,474 annotated slices): the PNG is the same
pixel grid as its DICOM (same rows x columns: 320, 512, 576 ... 1008 square), scaled to 8 bits, not resized
to 512. The XML <size> says 512 on some files whatever the real size. The box is therefore mapped from the
PNG's own size (read from its header) to the DICOM's rows x columns; where they are equal (all but two files)
the map is the identity. The crop's neighbours are found by slice position along the slice normal
(ImagePositionPatient . normal of ImageOrientationPatient); a file without a position falls back to
InstanceNumber.
"""
import argparse
import io
import json
import os
import re
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import ledger, paths  # noqa: E402

ROOT = os.environ.get("RAIDIOLOGY_LSS_DIR", os.path.join(paths.DATASETS, "public", "lss-sagittal-aisslab"))
OUT = os.path.join(paths.WORK, "eval", "spine_mri", "foramen")
LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
SIDES = ["R", "L"]                    # side input: 0 right, 1 left
NAME = re.compile(r"^([LR])FS([0-3])$")
FOLDS = 5
N_TEST = 94
CROP_MM, CROP_PX = 32.0, 64           # 0.5 mm per pixel
PAPER = {"qwk": 0.813, "lo": 0.769, "hi": 0.847, "acc": 75.3, "within1": 98.3, "patients": 469, "foramina": 2979}
GATE = 0.80
CFG = {"width": 32, "epochs": 40, "batch": 64, "lr": 2e-3, "weight_decay": 1e-4, "early_stop_frac": 0.15,
       "drop": 0.3, "flip": 0.0, "seed": 0}
LEASE_GB = 1.5
MARKER = "test_opened.json"
INITS = {"disc-v133": os.path.join("disc-grader-v133", "fold0.pt")}   # idea 34; under paths.MODELS, never in CFG
# Which of the disc grader's 10 input channels are T2 planes (crops.py + grader.py, read 2026-10-05):
# crops.crop() returns x of shape (2, PLANES, H, W) with x[0] = T2 and x[1] = T1 (crops.py lines 81-91), five planes
# (PLANES = 5) along the left-right axis, plane 2 the centre (grid(): w = (arange(5) - 2) * 3.3 mm, line 76).
# grader.Net.forward does self.body(x.flatten(1, 2)) (grader.py line 104): flatten(1, 2) on (2, 5, H, W) gives
# channel = sequence * 5 + plane, so channels 0-4 are T2 planes 0-4 and 5-9 are T1. The centre T2 plane is channel 2,
# its neighbours 1 and 3. foramen's crops are ordered (previous, own, next) slice (cmd_prepare: three = [prev, cur,
# nxt]; crop_from stacks them in that order), so the new body.0.weight is the grader's [:, [1, 2, 3]]: previous <- 1,
# own <- 2, next <- 3. The grader's planes run along one axis in one direction and the slice order in the other
# sense is unknown; the 3 x 3 kernels do not care much and training adapts them.
GRADER_T2_PLANES = [1, 2, 3]


def out(*p):
    return os.path.join(OUT, *p)


# ---- reading the dataset folder ----------------------------------------------------------------------------------

def list_files(root):
    """Every file under `root` as a forward-slash name relative to it, sorted (the names the zip held, e.g.
    "DICOM/0001/IM000002.dcm")."""
    names = []
    for d, _, files in os.walk(root):
        rel = os.path.relpath(d, root)
        base = "" if rel == "." else rel.replace(os.sep, "/") + "/"
        names += [base + f for f in files]
    return sorted(names)


def read_file(root, name, n=-1):
    """The bytes of a listed name (the first n when n >= 0)."""
    with open(os.path.join(root, *name.split("/")), "rb") as f:
        return f.read(n)


def require_root(root):
    if not os.path.isdir(root):
        raise SystemExit("the LSS-MRI-AISSLab folder %s does not exist (set RAIDIOLOGY_LSS_DIR to the unpacked set: "
                         "DICOM/, Foramina_Detection/, Segmentation/)" % root)


def level_of(text):
    t = (text or "").strip().upper().replace("\u2013", "-").replace("\u2014", "-").replace(" ", "")
    return LEVELS.index(t) if t in LEVELS else -1


def parse_voc(text, strays=None):
    """(frame width, height, [(side, grade, level text, (x0, y0, x1, y1))]) from one Pascal VOC XML. An object whose
    name is not [LR]FS[0-3] is not returned; if `strays` (a Counter) is given its name is counted there."""
    root = ET.fromstring(text.strip())
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        m = NAME.match(name)
        if not m:
            if strays is not None:
                strays[name] += 1
            continue
        b = o.find("bndbox")
        box = tuple(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((m.group(1), int(m.group(2)), (o.findtext("level") or "").strip(), box))
    return w, h, objs


def map_box(box, frame_wh, rows, cols):
    """A box in the PNG's frame to the DICOM's own pixels (a plain scale per axis; identity when the sizes match)."""
    sx, sy = cols / float(frame_wh[0]), rows / float(frame_wh[1])
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def png_size(head):
    """(width, height) from the first 24 bytes of a PNG, or None when they are not a sane PNG header."""
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return (w, h) if 16 <= w <= 8192 and 16 <= h <= 8192 else None


def sibling_spacing(root, names, patient, rows, cols):
    """Pixel spacing from another slice of the same patient with the same matrix (one DICOM, 0245/IM000001, has none)."""
    import io
    import pydicom
    for n in sorted(x for x in names if x.startswith("DICOM/%s/" % patient) and x.endswith(".dcm")):
        ds = pydicom.dcmread(io.BytesIO(read_file(root, n)), force=True, stop_before_pixels=True)
        if "PixelSpacing" in ds and (int(ds.Rows), int(ds.Columns)) == (rows, cols):
            return [float(v) for v in ds.PixelSpacing]
    return None


def build_index(root, say=print):
    import pydicom
    from collections import Counter
    import io
    names = set(list_files(root))
    xmls = sorted(n for n in names if n.startswith("Foramina_Detection/") and n.endswith(".xml"))
    strays, bad_level, rows, nopng = Counter(), Counter(), [], 0
    for n in xmls:
        patient, stem = n.split("/")[1], os.path.basename(n)[:-4]
        w, h, objs = parse_voc(read_file(root, n).decode("utf-8", "replace"), strays)
        if not objs:
            continue
        dn = "DICOM/%s/%s.dcm" % (patient, stem)
        if dn not in names:
            say("no DICOM for %s" % n)
            continue
        ds = pydicom.dcmread(io.BytesIO(read_file(root, dn)), force=True, stop_before_pixels=True)
        nr, nc = int(ds.Rows), int(ds.Columns)
        sp = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else sibling_spacing(root, names, patient, nr, nc)
        if sp is None:
            say("no pixel spacing for %s: its foramina are dropped" % dn)
            continue
        pn = "Foramina_Detection/%s/%s.png" % (patient, stem)
        frame = png_size(read_file(root, pn, 24)) if pn in names else None
        if frame is None:
            nopng += 1
            frame = (w, h) if w >= 16 and h >= 16 else (nc, nr)
        for side, grade, lv, box in objs:
            li = level_of(lv)
            if li < 0:
                bad_level[lv] += 1
                continue
            d = map_box(box, frame, nr, nc)
            rows.append(dict(patient=patient, slice=stem, png=stem + ".png", dicom=stem + ".dcm", level=li, side=side,
                             grade=grade, x0=box[0], y0=box[1], x1=box[2], y1=box[3], frame_w=frame[0],
                             frame_h=frame[1], dx0=d[0], dy0=d[1], dx1=d[2], dy1=d[3], rows=nr, cols=nc,
                             sp_row=sp[0], sp_col=sp[1]))
    df = pd.DataFrame(rows)
    return df, strays, bad_level, nopng


# ---- the split --------------------------------------------------------------------------------------------

def stratified_pick(patients, worst, n, seed):
    """n patients drawn at random (seeded) in proportion to each worst-grade group's size."""
    rng = np.random.RandomState(seed)
    groups = {g: sorted(p for p in patients if worst[p] == g) for g in sorted(set(worst[p] for p in patients))}
    total = float(len(patients))
    quota = {g: n * len(v) / total for g, v in groups.items()}
    take = {g: int(q) for g, q in quota.items()}
    for g in sorted(groups, key=lambda g: (-(quota[g] - take[g]), g)):
        if sum(take.values()) >= n:
            break
        take[g] += 1
    pick = []
    for g, v in groups.items():
        v = list(v)
        rng.shuffle(v)
        pick += v[:min(take[g], len(v))]
    return sorted(pick)


def assign_split(worst, test_patients=None, n_test=N_TEST, seed=0, folds=FOLDS):
    """patient -> fold 0..folds-1, or -1 for the test list. `worst` maps patient -> worst grade."""
    pats = sorted(worst)
    if test_patients is not None:
        test = sorted(set(test_patients) & set(pats))
    else:
        test = stratified_pick(pats, worst, n_test, seed)
    ts = set(test)
    rest = [p for p in pats if p not in ts]
    rng = np.random.RandomState(seed + 1)
    order = []
    for g in sorted(set(worst[p] for p in rest)):
        v = [p for p in rest if worst[p] == g]
        rng.shuffle(v)
        order += v
    fold = {p: i % folds for i, p in enumerate(order)}
    fold.update({p: -1 for p in test})
    return fold


def read_test_list(path):
    d = pd.read_csv(path, dtype=str)
    d.columns = [c.strip().lower() for c in d.columns]
    keep = d[d["split"].str.strip().str.lower().isin(["test", "testing"])]
    return sorted(str(p).strip().zfill(4) for p in keep["patient"])


def cmd_index(a):
    os.makedirs(OUT, exist_ok=True)
    require_root(ROOT)
    df, strays, bad_level, nopng = build_index(ROOT)
    paper_list = read_test_list(a.test_list) if a.test_list else None
    worst = df.groupby("patient")["grade"].max().to_dict()
    fold = assign_split(worst, paper_list)
    df["fold"] = df["patient"].map(fold).astype(int)
    df.to_csv(out("index.csv"), index=False)
    t = sorted(p for p, f in fold.items() if f == -1)
    pd.DataFrame({"patient": t, "split": "test"}).to_csv(out("test_list.csv"), index=False)
    source = "paper" if paper_list is not None else "own"
    json.dump({"source": source, "file": a.test_list, "patients": len(t)}, open(out("split_info.json"), "w"))
    if paper_list is not None:
        gone = sorted(set(paper_list) - set(worst))
        print("the paper's test list: %d patients, %d of them hold a graded foramen here%s"
              % (len(paper_list), len(paper_list) - len(gone), (" (none for: %s)" % ", ".join(gone)) if gone else ""))
    print("PNG to DICOM: the same pixel grid (8-bit rendering, no resize); %d boxes needed a scale (frame != DICOM), "
          "%d slices had no readable PNG header" % (int(((df.frame_w != df.cols) | (df.frame_h != df.rows)).sum()), nopng))
    print("other object names (ignored): %s" % (dict(strays) or "none"))
    print("unreadable levels (dropped): %s" % (dict(bad_level) or "none"))
    print("patients %d (paper %d), foramina %d (paper %d)" % (df.patient.nunique(), PAPER["patients"], len(df),
                                                              PAPER["foramina"]))
    print("per grade:  " + "  ".join("%d: %d" % (g, n) for g, n in df.grade.value_counts().sort_index().items()))
    print("per level:  " + "  ".join("%s: %d" % (LEVELS[l], n) for l, n in df.level.value_counts().sort_index().items()))
    print("per fold (patients / foramina):  " + "  ".join(
        "%s: %d / %d" % ("test" if f == -1 else f, df[df.fold == f].patient.nunique(), (df.fold == f).sum())
        for f in sorted(df.fold.unique(), key=lambda f: (f == -1, f))))
    print("test list: %s (%s)" % (len(t), "the paper's" if source == "paper" else "our own seeded draw"))
    check_split(df)
    return 0


def check_split(df):
    per = df.groupby("patient")["fold"].nunique()
    if (per > 1).any():
        raise SystemExit("a patient is in two folds: %s" % list(per[per > 1].index[:5]))


# ---- crops ------------------------------------------------------------------------------------------------

def crop_files(crop_mm=None):
    """(crops, rows, done) paths of the crop cache of this size; None or 32 mm keeps today's names."""
    if crop_mm is None or float(crop_mm) == CROP_MM:
        return out("crops.u8"), out("crops_rows.csv"), out("prepare_done.txt")
    t = "%gmm" % crop_mm
    return out("crops_%s.u8" % t), out("crops_%s_rows.csv" % t), out("prepare_done_%s.txt" % t)


def crop_px_of(crop_mm=None):
    return CROP_PX if crop_mm is None else int(round(crop_mm / 0.5))


def check_crop_mm(v):
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 and abs(v / 0.5 - round(v / 0.5)) < 1e-9):
        raise SystemExit("crop_mm must be a number above 0 that is a whole number of 0.5 mm pixels, got %r" % (v,))


def crop_from(slices, cx, cy, sp_row, sp_col, crop_mm=CROP_MM, crop_px=CROP_PX):
    """(3, crop_px, crop_px) uint8 from three slices (3, H, W): crop_mm (32 by default) around (cx, cy) in pixels,
    resampled to 0.5 mm, 1st to 99th percentile of the whole crop to 0..255."""
    from scipy.ndimage import map_coordinates
    k = (np.arange(crop_px) - (crop_px - 1) / 2.0) * (crop_mm / crop_px)
    yy, xx = np.meshgrid(cy + k / sp_row, cx + k / sp_col, indexing="ij")
    c = np.stack([map_coordinates(s.astype(np.float32), [yy, xx], order=1, mode="nearest") for s in slices])
    lo, hi = np.percentile(c, [1, 99])
    if hi <= lo:
        return np.zeros(c.shape, np.uint8)
    return np.clip((c - lo) / (hi - lo) * 255.0, 0, 255).round().astype(np.uint8)


def frame_of(stack, png_bytes):
    """Index of the frame of `stack` (frames, H, W) that the PNG's pixels come from (correlation above 0.99)."""
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("L"), np.float64).ravel()
    r = [np.corrcoef(s.astype(np.float64).ravel(), im)[0, 1] if s.size == im.size else -1.0 for s in stack]
    f = int(np.argmax(r))
    if r[f] < 0.99:
        raise RuntimeError("no frame matches the PNG (best correlation %.3f)" % r[f])
    return f


def neighbours(infos, name):
    """Names of the slice before, itself and the slice after, among files of the same shape and orientation.
    infos: name -> (rows, cols, orientation tuple or None, position or None, instance number)."""
    r, c, iop, _, _ = infos[name]
    same = [n for n, v in infos.items() if (v[0], v[1]) == (r, c) and v[2] == iop]
    use_pos = all(infos[n][3] is not None for n in same)
    same.sort(key=lambda n: (infos[n][3] if use_pos else infos[n][4], n))
    i = same.index(name)
    return same[max(i - 1, 0)], name, same[min(i + 1, len(same) - 1)]


def dicom_info(ds):
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    pos = None
    if iop is not None and ipp is not None:
        o = [float(v) for v in iop]
        nrm = np.cross(o[:3], o[3:])
        pos = float(np.dot(nrm, [float(v) for v in ipp]))
    return (int(ds.Rows), int(ds.Columns), tuple(round(float(v), 2) for v in iop) if iop is not None else None, pos,
            int(getattr(ds, "InstanceNumber", 0) or 0))


def cmd_prepare(a):
    import pydicom
    cmm = float(getattr(a, "crop_mm", CROP_MM))
    check_crop_mm(cmm)
    px = crop_px_of(cmm)
    import io
    idx = pd.read_csv(out("index.csv"), dtype={"patient": str})
    n = len(idx)
    table = idx[["patient", "slice", "level", "side", "grade", "fold"]].copy()
    table.insert(0, "row", np.arange(n))
    mm_path, rows_path, done_path = crop_files(cmm)
    fresh = True
    if os.path.exists(rows_path) and os.path.exists(mm_path) and os.path.getsize(mm_path) == n * 3 * px * px:
        old = pd.read_csv(rows_path, dtype={"patient": str})
        fresh = not (len(old) == n and (old[["patient", "slice", "level", "side", "grade"]].astype(str).values
                                         == table[["patient", "slice", "level", "side", "grade"]].astype(str).values).all())
    if fresh:
        if os.path.exists(done_path):
            os.remove(done_path)
            print("the index changed: crops restart")
        np.memmap(mm_path, np.uint8, "w+", shape=(n, 3, px, px)).flush()
    table.to_csv(rows_path, index=False)
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    mm = np.memmap(mm_path, np.uint8, "r+", shape=(n, 3, px, px))
    require_root(ROOT)
    members = {}
    for nm in list_files(ROOT):
        if nm.startswith("DICOM/") and nm.endswith(".dcm"):
            members.setdefault(nm.split("/")[1], []).append(nm)
    todo = [p for p in sorted(idx.patient.unique()) if p not in done]
    print("crops: %d foramina, %d patients left of %d" % (n, len(todo), idx.patient.nunique()))
    for k, p in enumerate(todo):
        pix, infos = {}, {}
        for nm in members[p]:
            ds = pydicom.dcmread(io.BytesIO(read_file(ROOT, nm)), force=True)
            base = os.path.basename(nm)
            infos[base], pix[base] = dicom_info(ds), ds.pixel_array
        for i in idx.index[idx.patient == p]:
            r = idx.loc[i]
            if pix[r.dicom].ndim == 3:
                # an Enhanced MR file holds the whole stack (0245/IM000001, 13 frames): the labelled slice is the
                # frame its PNG was made from, its neighbours the frames beside it (queue 95, 2026-10-05 05:07)
                st = pix[r.dicom]
                f = frame_of(st, read_file(ROOT, "Foramina_Detection/%s/%s.png" % (p, r.slice)))
                three = [st[max(f - 1, 0)], st[f], st[min(f + 1, len(st) - 1)]]
            else:
                prev, cur, nxt = neighbours(infos, r.dicom)
                three = [pix[prev], pix[cur], pix[nxt]]
            mm[i] = crop_from(three, (r.dx0 + r.dx1) / 2.0, (r.dy0 + r.dy1) / 2.0, r.sp_row, r.sp_col, cmm, px)
        mm.flush()
        with open(done_path, "a") as f:
            f.write(p + "\n")
        if (k + 1) % 50 == 0:
            print("  %d / %d patients" % (k + 1, len(todo)), flush=True)
    print("crops done: %s" % mm_path)
    return 0


# ---- scores -----------------------------------------------------------------------------------------------

def decode(z):
    """Grade = how many of the 3 cumulative probabilities P(grade > k) are above 0.5."""
    z = np.asarray(z, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-z)) > 0.5).sum(1).astype(int)


def confusion(y, p, k=4):
    c = np.zeros((k, k))
    np.add.at(c, (np.asarray(y, int), np.asarray(p, int)), 1)
    return c


def qwk_conf(c):
    k = len(c)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / float((k - 1) ** 2)
    e = np.outer(c.sum(1), c.sum(0)) / max(c.sum(), 1e-12)
    den = (w * e).sum()
    return 1.0 - (w * c).sum() / den if den > 0 else 0.0


def qwk(y, p):
    return qwk_conf(confusion(y, p))


def boot_qwk(patients, y, p, n=2000, seed=0):
    """QWK with a patient-clustered bootstrap 95% interval: (qwk, low, high)."""
    pats = np.asarray(patients)
    uniq, inv = np.unique(pats, return_inverse=True)
    C = np.zeros((len(uniq), 4, 4))
    np.add.at(C, (inv, np.asarray(y, int), np.asarray(p, int)), 1)
    rng = np.random.RandomState(seed)
    s = np.array([qwk_conf(C[rng.randint(0, len(uniq), len(uniq))].sum(0)) for _ in range(n)])
    return qwk_conf(C.sum(0)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summary(df):
    y, p = df.grade.to_numpy(int), df.pred.to_numpy(int)
    q, lo, hi = boot_qwk(df.patient.to_numpy(), y, p)
    return dict(qwk=q, lo=lo, hi=hi, acc=100.0 * float((y == p).mean()), within1=100.0 * float((abs(y - p) <= 1).mean()),
                n=len(df), patients=int(df.patient.nunique()))


def verdict_of(q):
    return "OPENS TEST" if q >= GATE else "STAYS ON DEVELOPMENT"


def verdict_line(q):
    return "Verdict (idea 28 step 1, out-of-fold QWK at least %.2f): %s" % (GATE, verdict_of(q))


def paper_line(lo, hi):
    word = "BEATEN" if lo > PAPER["qwk"] else ("BEHIND" if hi < PAPER["qwk"] else "LEVEL")
    return "Against Phadke et al. 2026 (QWK %.3f): %s" % (PAPER["qwk"], word)


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


def kept_lines(gain, d, mean_new):
    """Idea 34's two verdict lines (the second only when KEPT): gain over d and at least 0.01; the new mean at the gate."""
    kept = bool(gain > d and gain >= 0.01)      # a plain bool: numpy's would stop compare's json.dump (q104, 11:27)
    lines = ["Verdict (idea 34, two-seed gain over the base above d and at least 0.01): %s" % ("KEPT" if kept else "NOT KEPT")]
    if kept:
        lines.append("Verdict (idea 34, the kept version's two-seed mean out-of-fold QWK at least %.2f): %s"
                     % (GATE, verdict_of(mean_new)))
    return kept, lines


def paired_boot_diff(new, base, n=2000, seed=0):
    """Patient bootstrap of QWK(new) - QWK(base) over the same foramina, the same patient draw for both: (diff, lo, hi)."""
    new, base = new.sort_values("row").reset_index(drop=True), base.sort_values("row").reset_index(drop=True)
    uniq, inv = np.unique(new.patient.to_numpy(), return_inverse=True)
    Cn, Cb = np.zeros((len(uniq), 4, 4)), np.zeros((len(uniq), 4, 4))
    np.add.at(Cn, (inv, new.grade.to_numpy(int), new.pred.to_numpy(int)), 1)
    np.add.at(Cb, (inv, base.grade.to_numpy(int), base.pred.to_numpy(int)), 1)
    rng = np.random.RandomState(seed)
    s = []
    for _ in range(n):
        i = rng.randint(0, len(uniq), len(uniq))
        s.append(qwk_conf(Cn[i].sum(0)) - qwk_conf(Cb[i].sum(0)))
    return qwk_conf(Cn.sum(0)) - qwk_conf(Cb.sum(0)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def mean_prob_decode(zs):
    """The ten-net read: the mean of the nets' cumulative probabilities P(grade > k) and the grade they give.
    zs: list of (n, 3) logits. Returns (probabilities (n, 3), grades (n,))."""
    p = np.mean([1.0 / (1.0 + np.exp(-np.asarray(z, np.float64))) for z in zs], 0)
    return p, (p > 0.5).sum(1).astype(int)


def may_open_test_many(versions, root=None):
    """(ok, reason): compare-<first>.json must name these versions as the new ones, say KEPT and OPENS TEST, every
    version must be complete, and no marker may exist."""
    root = root or OUT
    m = os.path.join(root, MARKER)
    if os.path.exists(m):
        return False, "the test list was already opened (%s)" % m
    c = os.path.join(root, "v%d" % versions[0], "compare-%d.json" % versions[0])
    if not os.path.exists(c):
        return False, "no %s (run compare first)" % c
    d = json.load(open(c))
    if d.get("new") != list(versions) or not d.get("kept") or d.get("gate") != "OPENS TEST":
        return False, "compare-%d.json does not say KEPT and OPENS TEST for versions %s" % (versions[0], list(versions))
    for v in versions:
        f = os.path.join(root, "v%d" % v, "verdict.json")
        if not os.path.exists(f) or not json.load(open(f)).get("complete"):
            return False, "v%d is not complete over all five folds" % v
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
            self.head = nn.Linear(ch[-1] + len(LEVELS) + 1, 3)

        def forward(self, x, level, side):
            f = F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)
            oh = F.one_hot(level, len(LEVELS)).to(f.dtype)
            return self.head(self.drop(torch.cat([f, oh, side[:, None].to(f.dtype)], 1)))

    return Net(cfg["width"], cfg["drop"])


def load_init(net, name, say=print, models=None):
    """Idea 34: the body of a freshly built Net from the disc grader's fold-0 state_dict. Every body.* tensor is
    copied as is except body.0.weight (32, 10, 3, 3) -> (32, 3, 3, 3), taken at the T2 planes (GRADER_T2_PLANES, the
    crop's previous, own, next channels). The head stays as built."""
    import torch
    path = os.path.join(models or paths.MODELS, INITS[name])
    g = torch.load(path, map_location="cpu")
    own = net.state_dict()
    new, n = {}, 0
    for key, v in g.items():
        if not key.startswith("body."):
            continue
        if key == "body.0.weight":
            assert tuple(v.shape) == (own[key].shape[0], 10, 3, 3), "grader first conv is %s" % (tuple(v.shape),)
            v = v[:, GRADER_T2_PLANES]
        assert key in own and tuple(v.shape) == tuple(own[key].shape), "%s: %s vs %s" % (
            key, tuple(v.shape), tuple(own[key].shape) if key in own else None)
        new[key] = v.clone()
        n += 1
    missing, unexpected = net.load_state_dict(new, strict=False)
    assert not unexpected and all(m.startswith("head.") for m in missing), (missing, unexpected)
    assert n == sum(1 for k in own if k.startswith("body.")), "the grader lacks body tensors"
    say("init %s: %s, %d body tensors copied (body.0.weight at T2 channels %s of 10), head fresh"
        % (name, path, n, GRADER_T2_PLANES))


def augment(x, side, rng, flip):
    import torch
    import torch.nn.functional as F
    n, dev = x.shape[0], x.device
    u = lambda lo, hi, *s: torch.tensor(rng.uniform(lo, hi, s), device=dev, dtype=torch.float32)  # noqa: E731
    ang, sc = u(-0.12, 0.12, n), u(0.9, 1.1, n)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang) * sc, -torch.sin(ang) * sc
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang) * sc, torch.cos(ang) * sc
    theta[:, :, 2] = u(-0.08, 0.08, n, 2)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    x = x * u(0.85, 1.15, n, 1, 1, 1) + u(-0.15, 0.15, n, 1, 1, 1)
    if flip > 0:                                   # a left-right swap of the crop swaps the side input with it
        f = torch.tensor(rng.random(n) < flip, device=dev)
        x[f] = x[f].flip(3)
        side = torch.where(f, 1 - side, side)
    return x, side


def prep(u8, dev):
    return (u8.to(dev).float() / 255.0 - 0.5) / 0.25


def predict(net, X, lv, sd, dev, batch=256):
    import torch
    net.eval()
    res = []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            xb = prep(X[a:a + batch], dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z = net(xb, lv[a:a + batch], sd[a:a + batch])
            res.append(z.float().cpu().numpy())
    return np.concatenate(res)


def setup_device():
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        from core import card
        # a short job (core/card.py's short lane): five folds took 153 s once v11 held the card, after an 85 min
        # wait behind larger shares (2026-10-05); RAIDIOLOGY_CARD_SHORT_MIN, when set, decides instead
        card.lease(float(os.environ.get("RAIDIOLOGY_CARD_GB", LEASE_GB)), "foramen",
                   short_min=None if os.environ.get("RAIDIOLOGY_CARD_SHORT_MIN") else 10)
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, LEASE_GB * 2 ** 30 / total), torch.cuda.current_device())
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(crop_mm=None):
    import torch
    mm_path, rows_path, done_path = crop_files(crop_mm)
    if not os.path.exists(rows_path):
        raise SystemExit("prepare has not run for crop %s mm (%s is missing)" % (crop_mm or CROP_MM, rows_path))
    rows = pd.read_csv(rows_path, dtype={"patient": str})
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    if set(rows.patient) - done:
        raise SystemExit("prepare has not finished (%d patients left)" % len(set(rows.patient) - done))
    px = crop_px_of(crop_mm)
    mm = np.memmap(mm_path, np.uint8, "r", shape=(len(rows), 3, px, px))
    return rows, torch.from_numpy(np.ascontiguousarray(mm))


def tensors(rows, X, dev, ix):
    import torch
    lv = torch.tensor(rows.level.to_numpy()[ix], dtype=torch.long, device=dev)
    sd = torch.tensor((rows.side.to_numpy()[ix] == "L").astype(np.int64), device=dev)
    y = torch.tensor(rows.grade.to_numpy()[ix], dtype=torch.long, device=dev)
    return X[torch.as_tensor(ix)], lv, sd, y


def ema_step(ema, net, d):
    """ema = d * ema + (1 - d) * net over all parameters and floating buffers; integer buffers are copied. No random numbers."""
    import torch
    with torch.no_grad():
        e, n = ema.state_dict(), net.state_dict()
        for key, v in n.items():
            if v.is_floating_point():
                e[key].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                e[key].copy_(v)


def train_fold(rows, X, k, cfg, dev, say=print):
    import copy
    import torch
    import torch.nn.functional as F
    seed = cfg["seed"] * 100 + k
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    dev_rows = rows[rows.fold >= 0]
    pats = sorted(dev_rows[dev_rows.fold != k].patient.unique())
    rng.shuffle(pats)
    n_stop = max(1, int(round(len(pats) * cfg["early_stop_frac"])))
    stop = set(pats[:n_stop])
    tr = np.where((rows.fold >= 0) & (rows.fold != k) & ~rows.patient.isin(stop))[0]
    st = np.where((rows.fold >= 0) & (rows.fold != k) & rows.patient.isin(stop))[0]
    ho = np.where(rows.fold == k)[0]
    Xt, lt, st_, yt = tensors(rows, X, dev, tr)
    Xs, ls, ss, ys = tensors(rows, X, dev, st)
    net = make_net(cfg)
    if cfg.get("init"):
        load_init(net, cfg["init"], say)
    net = net.to(dev)
    ema = copy.deepcopy(net) if cfg.get("ema") else None
    opt =torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * int(np.ceil(len(tr) / cfg["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.15)
    ys_np = rows.grade.to_numpy()[st]
    best, best_q, best_ep = None, -2.0, -1
    for ep in range(cfg["epochs"]):
        net.train()
        perm = torch.as_tensor(rng.permutation(len(tr)), device=dev)
        tot = 0.0
        for a in range(0, len(tr), cfg["batch"]):
            b = perm[a:a + cfg["batch"]]
            if len(b) < 2:
                continue
            xb, sb = augment(prep(Xt[b.cpu()], dev), st_[b], rng, cfg["flip"])
            cum = torch.stack([(yt[b] > j).float() for j in range(3)], 1)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z = net(xb, lt[b], sb)
            loss = F.binary_cross_entropy_with_logits(z.float(), cum)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                ema_step(ema, net, cfg["ema"])
            tot += float(loss) * len(b)
        ev = net if ema is None else ema
        q = qwk(ys_np, decode(predict(ev, Xs.cpu(), ls, ss, dev)))
        if q > best_q:
            best, best_q, best_ep = copy.deepcopy(ev.state_dict()), q, ep
        say("  fold %d epoch %2d loss %.4f stop-set QWK %.4f%s%s%s" % (k, ep, tot / max(len(tr), 1), q,
                                                                    " (EMA)" if ema is not None else "",
                                                                    " (crop %g mm)" % cfg["crop_mm"] if "crop_mm" in cfg else "",
                                                                    "  *" if ep == best_ep else ""))
    net.load_state_dict(best)
    Xh, lh, sh, _ = tensors(rows, X, dev, ho)
    return net, predict(net, Xh, lh, sh, dev), ho, best_q, best_ep


def load_oof(vdir):
    parts = [pd.read_csv(os.path.join(vdir, f), dtype={"patient": str}) for f in sorted(os.listdir(vdir))
             if re.match(r"oof_fold\d\.csv$", f)]
    return pd.concat(parts, ignore_index=True) if parts else None


def build_cfg(epochs, sets):
    """CFG plus --epochs and --set. `init` is known but absent from CFG: it enters the cfg only when given, so a
    version trained without it keeps exactly today's cfg.json and ledger settings."""
    cfg = dict(CFG)
    if epochs:
        cfg["epochs"] = epochs
    for s in sets or []:
        key, val = s.split("=", 1)
        if key not in cfg and key not in ("init", "ema", "crop_mm"):
            raise SystemExit("unknown setting %s (known: %s, init, ema, crop_mm)" % (key, ", ".join(CFG)))
        cfg[key] = json.loads(val)
    if "ema" in cfg and not (isinstance(cfg["ema"], (int, float)) and not isinstance(cfg["ema"], bool) and 0 < cfg["ema"] < 1):
        raise SystemExit("ema must be a decay strictly between 0 and 1, got %r" % (cfg["ema"],))
    if "crop_mm" in cfg:
        check_crop_mm(cfg["crop_mm"])
    if "init" in cfg and cfg["init"] not in INITS:
        raise SystemExit("unknown init %r (known: %s)" % (cfg["init"], ", ".join(INITS)))
    return cfg


def cmd_train(a):
    import torch
    cfg = build_cfg(a.epochs, a.set)
    vdir = out("v%d" % a.version)
    os.makedirs(vdir, exist_ok=True)
    cfg_path = os.path.join(vdir, "cfg.json")
    if os.path.exists(cfg_path) and json.load(open(cfg_path)) != cfg:
        raise SystemExit("v%d was trained with other settings (%s); take a new version" % (a.version, cfg_path))
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    folds = [int(f) for f in a.folds.split(",")] if a.folds else list(range(FOLDS))
    dev = setup_device()
    rows, X = load_data(cfg.get("crop_mm"))
    with ledger.run("experiment", "foramen v%d" % a.version) as led:
        led.settings_from(dict(cfg, folds=folds, version=a.version))
        for k in folds:
            oof = os.path.join(vdir, "oof_fold%d.csv" % k)
            if os.path.exists(oof):
                print("fold %d: already done" % k)
                continue
            net, z, ho, bq, be = train_fold(rows, X, k, cfg, dev)
            torch.save(net.state_dict(), os.path.join(vdir, "fold%d.pt" % k))
            d = rows.iloc[ho][["row", "patient", "slice", "level", "side", "grade"]].copy()
            d["fold"] = k
            d["pred"] = decode(z)
            for j in range(3):
                d["z%d" % j] = z[:, j]
            d.to_csv(oof, index=False)
            ep_path = os.path.join(vdir, "stop_epochs.json")      # idea 34; v1 has none, its files stay as they are
            eps = json.load(open(ep_path)) if os.path.exists(ep_path) else {}
            eps[str(k)] = int(be)
            json.dump(eps, open(ep_path, "w"))
            print("fold %d: best stop-set QWK %.4f at epoch %d; held-out QWK %.4f" % (k, bq, be, qwk(d.grade, d.pred)))
        df = load_oof(vdir)
        if df is None:
            return 1
        df.to_csv(os.path.join(vdir, "oof.csv"), index=False)
        s = summary(df)
        have = sorted(df.fold.unique())
        complete = have == list(range(FOLDS))
        print("out-of-fold (%d foramina, %d patients, folds %s): QWK %.4f (95%% %.4f to %.4f), accuracy %.1f, within one %.1f"
              % (s["n"], s["patients"], ",".join(map(str, have)), s["qwk"], s["lo"], s["hi"], s["acc"], s["within1"]))
        if not complete:
            print("only %d of %d folds are done: this is not final" % (len(have), FOLDS))
        v = verdict_of(s["qwk"])
        json.dump(dict(verdict=v, complete=complete, folds=[int(f) for f in have], **s),
                  open(os.path.join(vdir, "verdict.json"), "w"), indent=1)
        led.number("oof_qwk", s["qwk"]).number("oof_qwk_lo", s["lo"]).number("oof_qwk_hi", s["hi"])
        led.number("oof_accuracy", s["acc"], "%").number("oof_within_one", s["within1"], "%").number("oof_n", s["n"])
        print(verdict_line(s["qwk"]))
    return 0


def version_list(text):
    return [int(v) for v in text.split(",")]


def cmd_compare(a):
    base, new = version_list(a.base), version_list(a.new)
    if len(base) != 2 or len(new) != 2:
        raise SystemExit("compare needs two versions each: --base 1,11 --new 2,12")
    oofs = {}
    for v in base + new:
        f = out("v%d" % v, "oof.csv")
        if not os.path.exists(f):
            raise SystemExit("refused: v%d has no oof.csv" % v)
        oofs[v] = pd.read_csv(f, dtype={"patient": str})
    rows0 = sorted(oofs[base[0]]["row"])
    for v in base + new:
        if sorted(oofs[v]["row"]) != rows0:
            raise SystemExit("refused: v%d's out-of-fold file is not over the same foramina as v%d" % (v, base[0]))
    q, stops = {}, {}
    for v in base + new:
        s = summary(oofs[v])
        q[v] = s["qwk"]
        sp = out("v%d" % v, "stop_epochs.json")
        stops[v] = json.load(open(sp)) if os.path.exists(sp) else None
        print("v%d: out-of-fold QWK %.4f (95%% %.4f to %.4f), accuracy %.1f, within one %.1f%s"
              % (v, s["qwk"], s["lo"], s["hi"], s["acc"], s["within1"],
                 ("; stop epochs " + ", ".join("fold %s: %d" % (k, e) for k, e in sorted(stops[v].items())))
                 if stops[v] else "; stop epochs not recorded"))
    d = abs(q[base[0]] - q[base[1]])
    mean_base, mean_new = (q[base[0]] + q[base[1]]) / 2.0, (q[new[0]] + q[new[1]]) / 2.0
    gain = mean_new - mean_base
    diff, lo, hi = paired_boot_diff(oofs[new[0]], oofs[base[0]])
    print("d = |QWK(v%d) - QWK(v%d)| = %.4f; base mean %.4f, new mean %.4f, gain %.4f" % (base[0], base[1], d, mean_base,
                                                                                         mean_new, gain))
    print("paired patient bootstrap (2000, seed 0) of QWK(v%d) - QWK(v%d): %.4f (95%% %.4f to %.4f)"
          % (new[0], base[0], diff, lo, hi))
    kept, lines = kept_lines(gain, d, mean_new)
    for ln in lines:
        print(ln)
    res = dict(base=base, new=new, qwk={str(v): q[v] for v in q}, stop_epochs={str(v): stops[v] for v in stops}, d=d,
               mean_base=mean_base, mean_new=mean_new, gain=gain, boot_diff=diff, boot_lo=lo, boot_hi=hi, kept=kept,
               gate=verdict_of(mean_new) if kept else None, verdict_lines=lines)
    json.dump(res, open(out("v%d" % new[0], "compare-%d.json" % new[0]), "w"), indent=1)
    with ledger.run("experiment", "foramen compare v%d" % new[0]) as led:
        led.settings_from(dict(base=a.base, new=a.new))
        led.number("d", d).number("gain", gain).number("mean_new_qwk", mean_new).number("boot_diff", diff)
        led.number("boot_diff_lo", lo).number("boot_diff_hi", hi)
    return 0


def print_test(s, info):
    print("test (%d foramina, %d patients): QWK %.4f (95%% %.4f to %.4f); accuracy %.1f (paper %.1f); "
          "within one %.1f (paper %.1f)" % (s["n"], s["patients"], s["qwk"], s["lo"], s["hi"], s["acc"], PAPER["acc"],
                                            s["within1"], PAPER["within1"]))
    print(paper_line(s["lo"], s["hi"]))
    print("test list: %s" % ("the paper's own (same patients)" if info.get("source") == "paper"
                             else "our own seeded draw (same dataset, other split)"))


def cmd_test_many(a):
    """Idea 34's test read: the mean cumulative probabilities of five folds x the given versions' nets."""
    import torch
    versions = version_list(a.versions)
    ok, why = may_open_test_many(versions)
    if not ok:
        raise SystemExit("refused: " + why)
    info = json.load(open(out("split_info.json"))) if os.path.exists(out("split_info.json")) else {"source": "own"}
    json.dump(dict(version=versions[0], versions=versions, source=info.get("source"), at=pd.Timestamp.now().isoformat()),
              open(out(MARKER), "w"))                         # first: a crash after this still counts as opened
    dev = setup_device()
    rows, _ = load_data()
    ix = np.where(rows.fold == -1)[0]
    test_x = {}                                                   # crop size -> test crops, loaded once each
    zs = []
    with ledger.run("experiment", "foramen v%s test" % "+".join(map(str, versions))) as led:
        led.settings_from(dict(versions=versions, test_list=info.get("source")))
        for v in versions:
            cfg = json.load(open(out("v%d" % v, "cfg.json")))
            cm = cfg.get("crop_mm")
            if cm not in test_x:
                rows_c, Xc = load_data(cm)
                test_x[cm] = tensors(rows_c, Xc, dev, ix)
            Xe, le, se, _ = test_x[cm]
            for k in range(FOLDS):
                net = make_net(cfg).to(dev)
                net.load_state_dict(torch.load(out("v%d" % v, "fold%d.pt" % k), map_location=dev))
                zs.append(predict(net, Xe.cpu(), le, se, dev))
        p, pred = mean_prob_decode(zs)
        d = rows.iloc[ix][["row", "patient", "slice", "level", "side", "grade"]].copy()
        d["pred"] = pred
        for j in range(3):
            d["p%d" % j] = p[:, j]
        d.to_csv(out("v%d" % versions[0], "test_preds.csv"), index=False)
        s = summary(d)
        json.dump(dict(versions=versions, nets=len(zs), **s), open(out("v%d" % versions[0], "test_result.json"), "w"),
                  indent=1)
        led.number("test_qwk", s["qwk"]).number("test_qwk_lo", s["lo"]).number("test_qwk_hi", s["hi"])
        led.number("test_accuracy", s["acc"], "%").number("test_within_one", s["within1"], "%").number("test_n", s["n"])
        print("versions %s: the mean of %d nets' cumulative probabilities" % (",".join("v%d" % v for v in versions), len(zs)))
        print_test(s, info)
    return 0


def cmd_test(a):
    import torch
    if a.versions:
        if a.version:
            raise SystemExit("give --version or --versions, not both")
        return cmd_test_many(a)
    if not a.version:
        raise SystemExit("give --version N or --versions A,B")
    ok, why = may_open_test(a.version)
    if not ok:
        raise SystemExit("refused: " + why)
    info = json.load(open(out("split_info.json"))) if os.path.exists(out("split_info.json")) else {"source": "own"}
    json.dump(dict(version=a.version, source=info.get("source"), at=pd.Timestamp.now().isoformat()),
              open(out(MARKER), "w"))                         # first: a crash after this still counts as opened
    cfg = json.load(open(out("v%d" % a.version, "cfg.json")))
    dev = setup_device()
    rows, X = load_data(cfg.get("crop_mm"))
    ix = np.where(rows.fold == -1)[0]
    Xe, le, se, _ = tensors(rows, X, dev, ix)
    zs = []
    with ledger.run("experiment", "foramen v%d test" % a.version) as led:
        led.settings_from(dict(cfg, version=a.version, test_list=info.get("source")))
        for k in range(FOLDS):
            net = make_net(cfg).to(dev)
            net.load_state_dict(torch.load(out("v%d" % a.version, "fold%d.pt" % k), map_location=dev))
            zs.append(predict(net, Xe.cpu(), le, se, dev))
        z = np.mean(zs, 0)
        d = rows.iloc[ix][["row", "patient", "slice", "level", "side", "grade"]].copy()
        d["pred"] = decode(z)
        for j in range(3):
            d["z%d" % j] = z[:, j]
        d.to_csv(out("v%d" % a.version, "test_preds.csv"), index=False)
        s = summary(d)
        json.dump(s, open(out("v%d" % a.version, "test_result.json"), "w"), indent=1)
        led.number("test_qwk", s["qwk"]).number("test_qwk_lo", s["lo"]).number("test_qwk_hi", s["hi"])
        led.number("test_accuracy", s["acc"], "%").number("test_within_one", s["within1"], "%").number("test_n", s["n"])
        print_test(s, info)
    return 0


# ---- selftest ---------------------------------------------------------------------------------------------

VOC = """<annotation><size><width>512</width><height>512</height></size>
<object><name>LFS2</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox><level>L4-L5</level></object>
<object><name>classes</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
</annotation>"""


def selftest():
    from sklearn.metrics import cohen_kappa_score
    bad = []

    def check(name, ok):
        print("%-52s %s" % (name, "ok" if ok else "FAIL"))
        if not ok:
            bad.append(name)

    rng = np.random.RandomState(3)
    _selftest_parsing(check)
    _selftest_folders(check)
    _selftest_split(check, rng)
    _selftest_scores(check, rng, cohen_kappa_score)
    _selftest_crops(check)
    _selftest_crop_mm(check)
    _selftest_neighbours(check)
    _selftest_verdicts(check)
    _selftest_init(check)
    _selftest_ema(check)
    _selftest_kept_lines(check)
    _selftest_paired(check)
    _selftest_ten_net(check)
    _selftest_versions(check)
    print("selftest: %s" % ("FAILED: " + ", ".join(bad) if bad else "all passed"))
    return 1 if bad else 0


def _selftest_parsing(check):
    from collections import Counter
    stray = Counter()
    w, h, objs = parse_voc(VOC, stray)
    check("voc: name, level, box", (w, h) == (512, 512) and objs == [("L", 2, "L4-L5", (10.0, 20.0, 30.0, 60.0))])
    check("voc: a stray name is counted, not graded", stray == {"classes": 1} and len(objs) == 1)
    check("levels", level_of("L5-S1") == 4 and level_of(" l1\u2013l2 ") == 0 and level_of("L6-L7") == -1 and level_of("L1-L2") == 0)
    check("box 512 -> 320 DICOM", map_box((256, 128, 512, 256), (512, 512), 320, 320) == (160.0, 80.0, 320.0, 160.0))
    check("box same size is the identity", map_box((1, 2, 3, 4), (576, 576), 576, 576) == (1.0, 2.0, 3.0, 4.0))
    head = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 320, 320)
    check("png header size", png_size(head) == (320, 320) and png_size(b"x" * 24) is None)


def _selftest_folders(check):
    with tempfile.TemporaryDirectory() as t:
        for rel in ("DICOM/0001/IM000001.dcm", "DICOM/0001/IM000002.dcm", "Foramina_Detection/0001/IM000001.xml"):
            os.makedirs(os.path.dirname(os.path.join(t, *rel.split("/"))), exist_ok=True)
            open(os.path.join(t, *rel.split("/")), "wb").write(b"HEAD" + rel.encode())
        check("folder listing: forward-slash relative names, sorted",
              list_files(t) == ["DICOM/0001/IM000001.dcm", "DICOM/0001/IM000002.dcm", "Foramina_Detection/0001/IM000001.xml"])
        check("folder file read: whole and first n bytes", read_file(t, "DICOM/0001/IM000002.dcm", 4) == b"HEAD"
              and read_file(t, "DICOM/0001/IM000002.dcm").endswith(b"IM000002.dcm"))
        try:
            require_root(os.path.join(t, "absent"))
            check("a missing folder is refused", False)
        except SystemExit as e:
            check("a missing folder is refused", "RAIDIOLOGY_LSS_DIR" in str(e) and "absent" in str(e))


def _selftest_split(check, rng):
    from collections import Counter
    worst ={"%04d" % i: int(rng.choice(4, p=[.4, .25, .2, .15])) for i in range(1, 470)}
    f = assign_split(worst, None, 94, 0)
    test = [p for p, v in f.items() if v == -1]
    check("split: 94 test, all patients placed once", len(test) == 94 and len(f) == len(worst))
    check("split: five folds, balanced", sorted(set(f.values())) == [-1, 0, 1, 2, 3, 4] and
          max(Counter(v for v in f.values() if v >= 0).values()) - min(Counter(v for v in f.values() if v >= 0).values()) <= 1)
    share = lambda ps: np.mean([worst[p] == 3 for p in ps])  # noqa: E731
    check("split: test stratified by worst grade", abs(share(test) - np.mean([v == 3 for v in worst.values()])) < 0.06)
    check("split: seeded", assign_split(worst, None, 94, 0) == f and assign_split(worst, None, 94, 1) != f)
    given = ["0001", "0002", "0003", "9999"]
    g = assign_split(worst, given)
    check("split: a given test list is used, absent patients ignored",
          sorted(p for p, v in g.items() if v == -1) == given[:3] and all(v in range(5) for p, v in g.items() if p not in given))
    d = pd.DataFrame({"patient": ["a", "a", "b"], "fold": [0, 1, 0]})
    try:
        check_split(d)
        check("split: a patient in two folds is refused", False)
    except SystemExit:
        check("split: a patient in two folds is refused", True)
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "s.csv")
        pd.DataFrame({"patient": [1, 22, 333], "split": ["test", "train", "Test"]}).to_csv(p, index=False)
        check("test list file: patient padded, split=test", read_test_list(p) == ["0001", "0333"])


def _selftest_scores(check, rng, cohen_kappa_score):
    # ordinal decoding and scores
    z = np.array([[5, 5, 5], [5, 5, -5], [5, -5, -5], [-5, -5, -5], [-5, 5, 5], [0.1, -0.1, -9]])
    check("ordinal decoding", list(decode(z)) == [3, 2, 1, 0, 2, 1])
    y, p = rng.randint(0, 4, 400), None
    p = np.clip(y + rng.randint(-1, 2, 400), 0, 3)
    check("qwk equals sklearn's", abs(qwk(y, p) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-9)
    pat = rng.randint(0, 60, 400)
    q, lo, hi = boot_qwk(pat, y, p, n=200)
    check("bootstrap interval holds the estimate", lo <= q <= hi and abs(q - qwk(y, p)) < 1e-12)


def _selftest_crops(check):
    # crops
    img = np.tile(np.arange(200, dtype=np.float32), (200, 1))
    c = crop_from([img, img, img], 100.0, 100.0, 0.5, 0.5)
    check("crop: 3 x 64 x 64 uint8 over 0..255", c.shape == (3, 64, 64) and c.dtype == np.uint8 and c.min() == 0 and c.max() == 255)
    ramp = np.tile(np.clip((np.arange(200, dtype=np.float32) - 90) * 12.75, 0, 255), (200, 1))
    c64, c96 = crop_from([ramp] * 3, 100.0, 100.0, 0.5, 0.5), crop_from([ramp] * 3, 100.0, 100.0, 0.5, 0.5, 48.0, 96)
    check("crop 48 mm: 3 x 96 x 96 uint8", c96.shape == (3, 96, 96) and c96.dtype == np.uint8)
    check("crop 48 mm: centre 64 x 64 equals the 32 mm crop within 1 grey level",
          np.abs(c96[:, 16:80, 16:80].astype(int) - c64.astype(int)).max() <= 1)
    check("crop_files: None and 32 are today's names",
          crop_files(None) == crop_files(32) and [os.path.basename(f) for f in crop_files(32)]
          == ["crops.u8", "crops_rows.csv", "prepare_done.txt"])
    check("crop_files: 48 mm names", [os.path.basename(f) for f in crop_files(48)]
          == ["crops_48mm.u8", "crops_48mm_rows.csv", "prepare_done_48mm.txt"])


def _selftest_crop_mm(check):
    check("cfg: crop_mm absent by default, set by --set", "crop_mm" not in build_cfg(None, [])
          and build_cfg(None, ["crop_mm=48"])["crop_mm"] == 48)
    for bv in ("0", "-4", "48.3"):
        try:
            build_cfg(None, ["crop_mm=" + bv])
            refused = False
        except SystemExit:
            refused = True
        check("cfg: crop_mm=%s is refused" % bv, refused)


def _selftest_neighbours(check):
    infos = {"a": (9, 9, (1.0,), 3.0, 1), "b": (9, 9, (1.0,), 1.0, 2), "c": (9, 9, (1.0,), 2.0, 3), "d": (5, 5, (1.0,), 0.0, 4)}
    check("neighbours by position, ends repeat", neighbours(infos, "c") == ("b", "c", "a") and
          neighbours(infos, "b") == ("b", "b", "c"))


def _selftest_verdicts(check):
    # verdict and the test marker
    check("verdict at the gate", verdict_of(0.80) == "OPENS TEST" and verdict_of(0.7999) == "STAYS ON DEVELOPMENT")
    check("verdict line wording", verdict_line(0.9) == "Verdict (idea 28 step 1, out-of-fold QWK at least 0.80): OPENS TEST")
    check("paper line", paper_line(0.82, 0.9).endswith("BEATEN") and paper_line(0.77, 0.85).endswith("LEVEL")
          and paper_line(0.6, 0.8).endswith("BEHIND"))
    with tempfile.TemporaryDirectory() as t:
        check("test refused: no verdict", not may_open_test(1, t)[0])
        os.makedirs(os.path.join(t, "v1"))
        json.dump(dict(verdict="STAYS ON DEVELOPMENT", complete=True), open(os.path.join(t, "v1", "verdict.json"), "w"))
        check("test refused: stays on development", not may_open_test(1, t)[0])
        json.dump(dict(verdict="OPENS TEST", complete=False), open(os.path.join(t, "v1", "verdict.json"), "w"))
        check("test refused: not all folds", not may_open_test(1, t)[0])
        json.dump(dict(verdict="OPENS TEST", complete=True), open(os.path.join(t, "v1", "verdict.json"), "w"))
        check("test allowed: opens, complete, no marker", may_open_test(1, t)[0])
        os.makedirs(os.path.join(t, "v2"))
        json.dump(dict(verdict="OPENS TEST", complete=True), open(os.path.join(t, "v2", "verdict.json"), "w"))
        json.dump({}, open(os.path.join(t, MARKER), "w"))
        check("test refused: a marker exists (any version)", not may_open_test(1, t)[0] and not may_open_test(2, t)[0])


def _selftest_init(check):
    # idea 34: init from the disc grader, compare, the ten-net read
    import torch
    check("cfg: init absent by default and unset", "init" not in CFG and build_cfg(None, None) == CFG
          and build_cfg(None, ["flip=0.5"]).get("init") is None and build_cfg(None, ['init="disc-v133"'])["init"] == "disc-v133")
    net = make_net(CFG)
    g = {}
    for key, v in make_net(dict(CFG, seed=9)).state_dict().items():
        if key.startswith("body."):
            g[key] = torch.randn(v.shape) if v.dtype.is_floating_point else v.clone()
    g["body.0.weight"] = torch.randn(CFG["width"], 10, 3, 3)
    g["head.weight"], g["head.bias"] = torch.randn(3, 32 * 8 + len(LEVELS) + 1), torch.randn(3)
    head0 = {k: v.clone() for k, v in net.state_dict().items() if k.startswith("head.")}
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "disc-grader-v133"))
        torch.save(g, os.path.join(t, INITS["disc-v133"]))
        lines = []
        load_init(net, "disc-v133", lines.append, models=t)
    sd = net.state_dict()
    check("init: body tensors equal (running stats too)", all(torch.equal(sd[k], v) for k, v in g.items()
                                                              if k.startswith("body.") and k != "body.0.weight"))
    check("init: body.0.weight is the grader's T2 planes 1, 2, 3 in order",
          tuple(sd["body.0.weight"].shape) == (CFG["width"], 3, 3, 3) and GRADER_T2_PLANES == [1, 2, 3]
          and all(torch.equal(sd["body.0.weight"][:, i], g["body.0.weight"][:, 1 + i]) for i in range(3)))
    check("init: the head stays fresh", all(torch.equal(sd[k], head0[k]) and not torch.equal(sd[k], g[k]) for k in head0))
    check("init: one line printed", len(lines) == 1 and "disc-v133" in lines[0] and "[1, 2, 3]" in lines[0])


def _selftest_ema(check):
    import copy
    import torch
    check("cfg: ema absent by default", build_cfg(None, []) == CFG and "ema" not in build_cfg(None, []))
    check("cfg: ema=0.995 is set", build_cfg(None, ["ema=0.995"])["ema"] == 0.995)
    try:
        build_cfg(None, ["ema=1.5"])
        refused = False
    except SystemExit:
        refused = True
    check("cfg: ema=1.5 is refused", refused)
    tm = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    em = copy.deepcopy(tm)
    w0, m0 = em[0].weight.detach().clone(), em[1].running_mean.clone()
    with torch.no_grad():
        tm[0].weight.add_(1.0)
        tm[1].running_mean.add_(2.0)
    ema_step(em, tm, 0.9)
    check("ema: weight is 0.9 * w0 + 0.1 * w1", torch.allclose(em[0].weight, 0.9 * w0 + 0.1 * tm[0].weight, atol=1e-6))
    check("ema: running_mean likewise", torch.allclose(em[1].running_mean, 0.9 * m0 + 0.1 * tm[1].running_mean, atol=1e-6))
    eo = make_net(dict(CFG, ema=0.995))
    ema_step(eo, make_net(dict(CFG, ema=0.995)), 0.995)
    check("ema: the averaged state dict loads strict", not make_net(dict(CFG, ema=0.995)).load_state_dict(eo.state_dict(), strict=True).missing_keys)


def _selftest_kept_lines(check):
    ks = lambda gain, d, m: kept_lines(gain, d, m)  # noqa: E731
    check("compare: gain equal to d is NOT KEPT", not ks(0.01, 0.01, 0.9)[0] and ks(0.01, 0.01, 0.9)[1][0].endswith(": NOT KEPT"))
    check("compare: gain 0.0099 with d 0 is NOT KEPT", not ks(0.0099, 0.0, 0.9)[0])
    check("compare: gain 0.02 with d 0.01 is KEPT", ks(0.02, 0.01, 0.9)[0] and len(ks(0.02, 0.01, 0.9)[1]) == 2)
    check("compare: NOT KEPT prints no gate line", len(ks(0.0, 0.0, 0.9)[1]) == 1)
    check("compare: the gate at 0.80", ks(0.02, 0.01, 0.80)[1][1].endswith(": OPENS TEST")
          and ks(0.02, 0.01, 0.7999)[1][1].endswith(": STAYS ON DEVELOPMENT"))
    check("compare: verdict wording", ks(0.02, 0.01, 0.9)[1][0] ==
          "Verdict (idea 34, two-seed gain over the base above d and at least 0.01): KEPT"
          and ks(0.02, 0.01, 0.9)[1][1] == "Verdict (idea 34, the kept version's two-seed mean out-of-fold QWK at least 0.80): OPENS TEST")


def _selftest_paired(check):
    rr = np.random.RandomState(5)
    yy, pp = rr.randint(0, 4, 300), None
    pat3 = rr.randint(0, 40, 300)
    mk = lambda pr: pd.DataFrame({"row": np.arange(300), "patient": pat3.astype(str), "grade": yy, "pred": pr})  # noqa: E731
    pb, pn = np.clip(yy + rr.randint(-2, 3, 300), 0, 3), np.clip(yy + rr.randint(-1, 2, 300), 0, 3)
    dd, lo, hi = paired_boot_diff(mk(pn), mk(pb), n=200)
    check("compare: paired bootstrap holds its estimate", abs(dd - (qwk(yy, pn) - qwk(yy, pb))) < 1e-12 and lo <= dd <= hi
          and paired_boot_diff(mk(pn), mk(pn), n=50)[1:] == (0.0, 0.0))


def _selftest_ten_net(check):
    za, zb = np.full((2, 3), 2.0), np.full((2, 3), -2.0)
    za[1], zb[1] = [-9.0, -9.0, -9.0], [9.0, 9.0, 9.0]
    p, gr = mean_prob_decode([za, zb])
    check("ten-net read: mean of probabilities, not of logits", abs(p[0, 0] - 0.5) < 1e-12 and abs(p[1, 0] - 0.5) < 1e-3
          and gr.shape == (2,))
    p3, gr3 = mean_prob_decode([za, za, zb])
    check("ten-net read: three nets lean to the majority", list(gr3) == [3, 0] and abs(p3[0, 0] - (2 * 0.8808 + 0.1192) / 3) < 1e-3)


def _selftest_versions(check):
    with tempfile.TemporaryDirectory() as t:
        for v in (2, 12):
            os.makedirs(os.path.join(t, "v%d" % v))
            json.dump(dict(verdict="OPENS TEST", complete=True), open(os.path.join(t, "v%d" % v, "verdict.json"), "w"))
        check("test --versions refused: no compare file", not may_open_test_many([2, 12], t)[0])
        cj = os.path.join(t, "v2", "compare-2.json")
        json.dump(dict(new=[2, 12], kept=True, gate="STAYS ON DEVELOPMENT"), open(cj, "w"))
        check("test --versions refused: stays on development", not may_open_test_many([2, 12], t)[0])
        json.dump(dict(new=[2, 12], kept=True, gate="OPENS TEST"), open(cj, "w"))
        check("test --versions allowed: KEPT and OPENS TEST", may_open_test_many([2, 12], t)[0])
        check("test --versions refused: other versions", not may_open_test_many([2, 13], t)[0])
        json.dump({}, open(os.path.join(t, MARKER), "w"))
        check("test --versions refused: a marker exists", not may_open_test_many([2, 12], t)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index")
    s.add_argument("--test-list", help="CSV with patient and split columns (the paper's data/splits.csv)")
    s = sub.add_parser("prepare")
    s.add_argument("--crop-mm", type=float, default=CROP_MM, help="crop side in mm at 0.5 mm a pixel (idea 87), default 32")
    s = sub.add_parser("train")
    s.add_argument("--version", type=int, required=True)
    s.add_argument("--folds", help="e.g. 0,1 (default all five)")
    s.add_argument("--epochs", type=int)
    s.add_argument("--set", action="append", help="key=<json>, e.g. --set flip=0.5")
    s = sub.add_parser("compare")
    s.add_argument("--base", required=True, help="two version dirs, e.g. 1,11")
    s.add_argument("--new", required=True, help="two version dirs, e.g. 2,12")
    s = sub.add_parser("test")
    s.add_argument("--version", type=int)
    s.add_argument("--versions", help="idea 34: e.g. 2,12 (the ten-net read)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"index": cmd_index, "prepare": cmd_prepare, "train": cmd_train, "compare": cmd_compare, "test": cmd_test,
            "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
