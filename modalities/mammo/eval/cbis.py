"""CBIS-DDSM, benign or malignant, on the official split (docs/efficiency/README.md, "Mammography: CBIS-DDSM masses", idea 31).

TCIA's case CSVs (mass_case_description_{train,test}_set.csv, calc_case_description_*) give one row per lesion per view. The
UIDs and file names inside their paths do not match the disk, so only the FIRST path component is used: the full mammogram is
the one DICOM under <Mass-Test_P_00016_LEFT_CC>/, the ROI folder is that name + "_<abnormality id>" and holds two DICOMs, the
mask (the one with the full image's Rows x Columns) and a small crop. No PixelSpacing (scanned film): windows are in pixels.
80 rows' masks are not the full image's size (shrunk about 1/1.149, and off the lesion by up to 349 px once scaled back): those
rows are centred on CBIS's own crop of the lesion instead, found in the full image by exact match (correlation at least 0.99;
on rows with a full-size mask the crop's centre is the mask box's within 10 px, 2026-10-05).

    python cbis.py index                          the rows, the files, the folds -> <WORK>/eval/mammo/cbis/index.csv
    python cbis.py prepare [--type mass|calc|both] [--px 256|512]   a 1024 px window round each ROI box, area-resampled to 256
                                                  (default, cbis_256.u8) or 512 (cbis_512.u8, beside it), into a memmap
    python cbis.py train --version N [--type mass|calc] [--folds 0,1] [--epochs E] [--set key=<json> ...]
    python cbis.py test --version N               once, only when development OPENED TEST (or compare's KEPT + OPENS TEST); the five nets' mean on the test rows
    python cbis.py compare --base A,B --new C,D --step N   idea 36: two-seed gain of C,D over A,B out of fold (kept / opens test)
    python cbis.py selftest                       CPU, synthetic data

Net: chest_xray/eval/reader.py's trunk from scratch, as mammo.py builds it (no pretrained weights, LICENSING.md), one logit,
BCE. Development is five folds by patient over the training patients that are not also in a test CSV (those get fold
'excluded'); test is opened once per type (test_opened_<type>.json is written before any model loads) and a version's
settings are locked.

Optional settings (idea 36; --set seed_offset=1 --set px=512): seed_offset (k, non-negative int, default 0) shifts the net's seed and
the batch-order seeds by 7919 * k, never the folds, the early-stop rows or the normalisation sample; px (256 or 512, default 256)
is the picture size the window is resampled to. Each is left out of a version's locked settings when it is at its default, so
versions trained before they existed keep their settings exactly.
Optional setting (idea 42; --set 'init="chest-v103"'): init starts the trunk from our own NIH chest reader v103 (fold 0); the head stays fresh.
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mammo  # noqa: E402  (no side effects beyond KMP_BLOCKTIME and the repo path)
from mammo import auroc, patient_bootstrap, verdict  # noqa: E402,F401
from core import paths  # noqa: E402
from core.rows import FileRows  # noqa: E402

CBIS = os.path.join(paths.BESIDE, "Datasets", "public", "cbis-ddsm")
CSVS = os.path.join(CBIS, "files")
OUT = os.path.join(paths.WORK, "eval", "mammo", "cbis")
INDEX = os.path.join(OUT, "index.csv")
PIX = os.path.join(OUT, "cbis_256.u8")
DONE = os.path.join(OUT, "cbis_256.done")
WIN, PX = 1024, 256
SEED_STEP = mammo.SEED_STEP
STEP_GAIN = 0.01
OPTIONAL = {"seed_offset": 0, "px": 256, "init": None}    # settings that enter o only when they differ from these
INITS = {"chest-v103": os.path.join(paths.WORK, "eval", "chest", "reader-v103", "fold0.pt"),
         # CBIS-DDSM mass reader v5's fold 0 net (itself started from chest v103), which never saw CBIS's test
         "cbis-v5": os.path.join(paths.WORK, "eval", "mammo", "cbis", "v5", "fold0.pt"),
         # OMAMA-DB pretraining reader v1's fold 0 net (omama.py; trained on OMAMA-DB mammograms only, never on CMMD)
         "omama-v1": os.path.join(paths.WORK, "eval", "mammo", "omama", "v1", "fold0.pt"),
         # OMAMA-DB crop reader v1's fold 0 net (omama.py train-crops, idea 128: native 512 px crops of OMAMA's cancers
         # from the originals, started from omama-v1; never saw CMMD)
         "omama-crops-v1": os.path.join(paths.WORK, "eval", "mammo", "omama", "crops-v1", "fold0.pt")}
FOLDS = 5
SEED = mammo.SEED
BOOT, BOOT_SEED = 2000, 0
# the card share by picture size: at 256 px the five folds of v1 (masses) peaked at 0.39 to 0.45 GB allocated
# (2026-10-05 08:15) under a 3.0 GB lease that then headed the card's queue for an hour; 512 px has four times
# the activations
LEASE_GB = {256: 1.0, 512: 2.5}
GATE = 0.75
BAR_SCRATCH, BAR_IMAGENET = 0.716, 0.804
VERSION_COLUMNS = ["row", "split", "type", "patient", "side", "view", "abn", "pathology", "y", "full", "mask", "locate", "rows",
                   "cols", "usable", "reason", "fold"]
MATCH = 0.99         # the crop's least correlation with the full image where it is found
SETTINGS = {
    "epochs": 40, "batch": 32, "lr": 3e-4, "weight_decay": 1e-4,
    "scale": 0.10, "brightness": 0.10, "contrast": 0.10,
    "widths": None, "dropout": None,    # None: reader.CFG's
}


# ---------------------------------------------------------------- the CSVs and the disk

def first_component(path):
    return str(path).replace("\\", "/").strip().split("/")[0].strip()


def folders_of(image_path, abn):
    """(full folder, ROI folder) on disk from a CSV row: the first component of 'image file path', and that name + _<abnormality id>."""
    full = first_component(image_path)
    return full, "%s_%d" % (full, int(abn))


def label_of(pathology):
    """1 MALIGNANT; 0 BENIGN or BENIGN_WITHOUT_CALLBACK; None for anything else."""
    p = str(pathology).strip().upper()
    if p == "MALIGNANT":
        return 1
    return 0 if p in ("BENIGN", "BENIGN_WITHOUT_CALLBACK") else None


def pick_mask(full_rows_cols, candidates):
    """candidates: [(path, rows, cols)]. (path, '') for the one file with the full image's Rows x Columns, else (None, reason)."""
    same = [c for c in candidates if (c[1], c[2]) == tuple(full_rows_cols)]
    if len(same) == 1:
        return same[0][0], ""
    if not same:
        return None, "no ROI file of the full image's size (the mask and the crop both differ)"
    return None, "%d ROI files of the full image's size" % len(same)


def pick_crop(full_rows_cols, candidates):
    """When no ROI file has the full image's size: (path, '') for the crop, the smaller of exactly two files and smaller than the
    full image both ways; else (None, reason)."""
    if len(candidates) != 2:
        return None, "no ROI file of the full image's size, and %d ROI files (not a mask and a crop)" % len(candidates)
    c = min(candidates, key=lambda c: c[1] * c[2])
    if not (c[1] < full_rows_cols[0] and c[2] < full_rows_cols[1]):
        return None, "no ROI file of the full image's size, and no crop smaller than it"
    return c[0], ""


def pix_path(px=PX):
    return PIX if px == PX else os.path.join(OUT, "cbis_%d.u8" % px)


def done_path(px=PX):
    return DONE if px == PX else os.path.join(OUT, "cbis_%d.done" % px)


def marker(typ):
    return os.path.join(OUT, "test_opened_%s.json" % typ)


def dicoms_in(folder):
    out = []
    for dp, _, fs in os.walk(folder):     # one case folder: <folder>/<series uid>/<file>.dcm
        out += [os.path.join(dp, f) for f in fs if f.lower().endswith(".dcm")]
    return sorted(out)


def read_csvs():
    import pandas as pd
    frames = []
    for split in ("train", "test"):
        for typ in ("mass", "calc"):
            d = pd.read_csv(os.path.join(CSVS, "%s_case_description_%s_set.csv" % (typ, split)), keep_default_na=False)
            frames.append(pd.DataFrame({
                "split": split, "type": typ, "patient": d["patient_id"].astype(str).str.strip(),
                "side": d["left or right breast"], "view": d["image view"], "abn": d["abnormality id"].astype(int),
                "pathology": d["pathology"].astype(str).str.strip(), "image_path": d["image file path"]}))
    return pd.concat(frames, ignore_index=True)


def assign_patient_folds(train_labels, test_patients, k=FOLDS, seed=SEED):
    """{patient: fold or 'excluded'}: train_labels is {patient: 1 when any lesion is malignant}; a patient who is also in a
    test CSV is 'excluded'; the others get a seeded fold stratified by that label."""
    shared = set(train_labels) & set(test_patients)
    fold = mammo.assign_folds({p: y for p, y in train_labels.items() if p not in shared}, k, seed)
    out = dict(fold)
    out.update({p: "excluded" for p in shared})
    return out


def build_index():
    import pydicom
    df = read_csvs()
    heads = {}

    def head(path):
        if path not in heads:
            h = pydicom.dcmread(path, stop_before_pixels=True)
            heads[path] = (int(h.Rows), int(h.Columns))
        return heads[path]

    full_cache, rows = {}, []
    for r in df.itertuples():
        full, roi = folders_of(r.image_path, r.abn)
        y = label_of(r.pathology)
        rec = {"full": "", "mask": "", "locate": "", "rows": 0, "cols": 0, "usable": 0, "reason": ""}
        if y is None:
            rec["reason"] = "pathology %r" % r.pathology
        else:
            if full not in full_cache:
                full_cache[full] = dicoms_in(os.path.join(CBIS, full)) if os.path.isdir(os.path.join(CBIS, full)) else None
            fl = full_cache[full]
            if fl is None:
                rec["reason"] = "full folder missing"
            elif len(fl) != 1:
                rec["reason"] = "full folder holds %d DICOMs" % len(fl)
            elif not os.path.isdir(os.path.join(CBIS, roi)):
                rec["reason"] = "ROI folder missing"
            else:
                rec["full"] = os.path.relpath(fl[0], CBIS).replace(os.sep, "/")
                rec["rows"], rec["cols"] = head(fl[0])
                cands = [(p,) + head(p) for p in dicoms_in(os.path.join(CBIS, roi))]
                m, why = pick_mask((rec["rows"], rec["cols"]), cands)
                rec["locate"] = "mask"
                if m is None and why.startswith("no ROI file"):
                    m, why = pick_crop((rec["rows"], rec["cols"]), cands)
                    rec["locate"] = "crop"
                if m is None:
                    rec["reason"], rec["locate"] = why, ""
                else:
                    rec["mask"], rec["usable"] = os.path.relpath(m, CBIS).replace(os.sep, "/"), 1
        rows.append(dict(row=len(rows), split=r.split, type=r.type, patient=r.patient, side=r.side, view=r.view, abn=r.abn,
                         pathology=r.pathology, y=-1 if y is None else y, **rec))
    import pandas as pd
    ix = pd.DataFrame(rows)
    tr = ix[ix.split == "train"]
    lab = {p: int((g.y == 1).any()) for p, g in tr.groupby("patient")}
    fold = assign_patient_folds(lab, set(ix[ix.split == "test"].patient))
    ix["fold"] = [fold.get(p, "") if s == "train" else "test" for p, s in zip(ix.patient, ix.split)]
    return ix[VERSION_COLUMNS]


def index():
    ix = build_index()
    os.makedirs(OUT, exist_ok=True)
    ix.to_csv(INDEX, index=False)
    print("index: %d rows -> %s" % (len(ix), INDEX))
    print("full folders used: %d, ROI folders used: %d" % (ix[ix.usable == 1].full.nunique(), int((ix.usable == 1).sum())))
    for (s, t), g in ix[ix.locate == "crop"].groupby(["split", "type"]):
        print("  centred on the crop (mask not the image's size): %-5s %-4s %d rows" % (s, t, len(g)))
    print("rows per split / type / pathology:")
    for (s, t, p), g in ix.groupby(["split", "type", "pathology"]):
        print("  %-5s %-4s %-24s %5d" % (s, t, p, len(g)))
    ex = ix[ix.fold == "excluded"]
    print("excluded patients (train patients also in a test CSV): %d (%d train rows)" % (ex.patient.nunique(), len(ex)))
    bad = ix[ix.usable == 0]
    print("unusable rows: %d" % len(bad))
    for (s, t, r), g in bad.groupby(["split", "type", "reason"]):
        print("  %5d  %-5s %-4s %s" % (len(g), s, t, r))
    for t in ("mass", "calc"):
        print("per fold (%s train rows / patients / malignant rows):" % t)
        for k in [str(i) for i in range(FOLDS)] + ["excluded"]:
            g = ix[(ix.split == "train") & (ix.type == t) & (ix.fold.astype(str) == k)]
            print("  fold %-8s %5d rows %5d patients %5d malignant" % (k, len(g), g.patient.nunique(), int((g.y == 1).sum())))
    twice = int((ix[ix.split == "train"].groupby("patient").fold.nunique() > 1).sum())
    print("patients in two folds: %d" % twice)
    return 0


def load_index():
    import pandas as pd
    if not os.path.exists(INDEX):
        sys.exit("no %s: run `index` first" % INDEX)
    return pd.read_csv(INDEX, keep_default_na=False)


# ---------------------------------------------------------------- the picture

def to_u8(arr, mono1=False):
    """The whole image scaled to 0..255 by its own min and max (MONOCHROME1 inverted, as mammo.window_to_u8 does)."""
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / max(hi - lo, 1e-6) * 255.0
    if mono1:
        a = 255.0 - a
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def box_centre(mask):
    """(y, x) of the centre of the nonzero pixels' bounding box, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (int(ys.min()) + int(ys.max()) + 1) // 2, (int(xs.min()) + int(xs.max()) + 1) // 2


def find_crop(full, crop, q=4):
    """(y, x, correlation): the centre (box_centre's convention) of where `crop` sits in `full`, both raw pixels. A search at
    1/q size over the windows that are not flat (flat background makes normalised correlation meaningless), then the exact
    place at full size within q * 3 px of it."""
    import cv2
    f, c = full.astype(np.float32), crop.astype(np.float32)
    fq = cv2.resize(f, (f.shape[1] // q, f.shape[0] // q), interpolation=cv2.INTER_AREA)
    cq = cv2.resize(c, (max(c.shape[1] // q, 8), max(c.shape[0] // q, 8)), interpolation=cv2.INTER_AREA)
    s = cv2.matchTemplate(fq, cq, cv2.TM_CCOEFF_NORMED)
    th, tw = cq.shape
    mu = cv2.boxFilter(fq, -1, (tw, th), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    mu2 = cv2.boxFilter(fq * fq, -1, (tw, th), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0))[:s.shape[0], :s.shape[1]]
    s[(sd < 0.02 * max(float(f.max() - f.min()), 1.0)) | ~np.isfinite(s)] = -1
    _, _, _, (lx, ly) = cv2.minMaxLoc(s)
    pad = 3 * q
    ya, xa = max(ly * q - pad, 0), max(lx * q - pad, 0)
    reg = f[ya:ly * q + c.shape[0] + pad, xa:lx * q + c.shape[1] + pad]
    if reg.shape[0] < c.shape[0] or reg.shape[1] < c.shape[1]:
        return None
    s2 = cv2.matchTemplate(reg, c, cv2.TM_CCOEFF_NORMED)
    s2[~np.isfinite(s2)] = -1
    _, best, _, (bx, by) = cv2.minMaxLoc(s2)
    return ya + by + c.shape[0] // 2, xa + bx + c.shape[1] // 2, float(best)


def window(a, cy, cx, size=WIN):
    """a (h, w) -> (size, size) of the same dtype: the image's pixels round (cy, cx), zeros beyond its edge; pixel (cy, cx) lands
    on (size // 2, size // 2)."""
    out = np.zeros((size, size), a.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    ya, xa = max(y0, 0), max(x0, 0)
    yb, xb = min(y0 + size, a.shape[0]), min(x0 + size, a.shape[1])
    if yb > ya and xb > xa:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
    return out


def shrink(w, px=PX):
    import cv2
    return cv2.resize(np.ascontiguousarray(w), (px, px), interpolation=cv2.INTER_AREA)


def read_pixels(path):
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a


def prep_group(job, px=PX):
    """job: (full path, [(row, ROI file path, 'mask' or 'crop')]) -> [(row, uint8 (px, px) or None, why)]; the full image is
    decoded once."""
    import pydicom
    full, items = job
    try:
        mono1 = str(pydicom.dcmread(os.path.join(CBIS, full), stop_before_pixels=True).get("PhotometricInterpretation", "")) == "MONOCHROME1"
        raw = read_pixels(os.path.join(CBIS, full))
        img = to_u8(raw, mono1)
    except Exception as e:  # noqa: BLE001
        return [(r, None, "full image: %s" % e) for r, _, _ in items]
    out = []
    for r, m, how in items:
        try:
            roi = read_pixels(os.path.join(CBIS, m))
            if how == "crop":
                hit = find_crop(raw, roi)
                if hit is None or hit[2] < MATCH:
                    out.append((r, None, "crop not found in the full image (correlation %s)" % (None if hit is None else "%.3f" % hit[2])))
                    continue
                c = hit[:2]
            elif roi.shape != img.shape:
                out.append((r, None, "mask shape %s, image %s" % (roi.shape, img.shape)))
                continue
            else:
                c = box_centre(roi)
            out.append((r, None, "empty mask") if c is None else (r, shrink(window(img, *c), px), ""))
        except Exception as e:  # noqa: BLE001
            out.append((r, None, "%s: %s" % (how, e)))
    return out


def pixels(n, mode="r", px=PX):
    if mode == "r":                       # read by file reads, not a memory map (idea 53); a writer still gets a memmap
        return FileRows(pix_path(px), (n, px, px))
    return np.memmap(pix_path(px), dtype=np.uint8, mode=mode, shape=(n, px, px))


def prepare(typ="both", px=PX):
    ix = load_index()
    n = len(ix)
    pp, dp = pix_path(px), done_path(px)
    fresh = not (os.path.exists(pp) and os.path.getsize(pp) == n * px * px and os.path.exists(dp) and os.path.getsize(dp) == n)
    pix = np.memmap(pp, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n, px, px))
    done = np.memmap(dp, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n,))     # 1 done, 2 failed (see cbis_<px>.bad.csv)
    want = ix[(ix.usable == 1) & (ix.type.isin(("mass", "calc") if typ == "both" else (typ,)))]
    want = want[done[want.row.to_numpy()] == 0]
    groups = {}
    for r in want.itertuples():
        groups.setdefault(r.full, []).append((int(r.row), r.mask, r.locate))
    print("prepare %s at %d px: %d rows to do from %d full images" % (typ, px, len(want), len(groups)), flush=True)
    t0, c, bad = time.time(), 0, []
    with ThreadPoolExecutor(4) as ex:
        for res in ex.map(prep_group, list(groups.items()), itertools.repeat(px)):
            for r, a, why in res:
                if a is None:
                    done[r] = 2
                    bad.append((r, why))
                else:
                    pix[r], done[r] = a, 1
                c += 1
            if c % 200 < len(res):
                pix.flush(), done.flush()
                print("  %d / %d, %.0f s" % (c, len(want), time.time() - t0), flush=True)
    pix.flush(), done.flush()
    if bad:
        with open(os.path.join(OUT, "cbis_%d.bad.csv" % px), "a") as f:
            f.writelines("%d,%s\n" % (r, w.replace(",", ";")) for r, w in bad)
    print("prepare: %d done, %d failed" % (c - len(bad), len(bad)))
    return 1 if bad else 0


# ---------------------------------------------------------------- training

def device(torch):
    if os.environ.get("RAIDIOLOGY_DEVICE", "").strip() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def est_card_min(px, epochs):
    """Minutes of card for five folds: about 2 s an epoch at 256 px, measured 2026-10-05 (v1 masses 42 to 68 s a fold
    of 40 epochs, v2 calcifications 80 to 86 s), four times that at 512, with a margin of 1.5; a test reads in about
    5 minutes (epochs None)."""
    if epochs is None:
        return 5.0
    return round(1.5 * FOLDS * epochs * 2.0 * (px / 256) ** 2 / 60.0, 1)


def limit_card_memory(torch, dev, px=None, epochs=None):
    """Lease the card before touching CUDA, then cap the process at the lease (RAIDIOLOGY_CARD_GB overrides, else
    LEASE_GB for the picture size). The card is told how long it is held (est_card_min), so a short run takes the
    short lane (RAIDIOLOGY_CARD_SHORT_MIN, when set, decides instead)."""
    if dev.type != "cuda":
        return
    from core import card
    gb = float(os.environ.get("RAIDIOLOGY_CARD_GB", LEASE_GB[px or PX]))
    card.lease(gb, "cbis train", short_min=None if os.environ.get("RAIDIOLOGY_CARD_SHORT_MIN") else est_card_min(px or PX, epochs))
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, gb * 2 ** 30 / total), torch.cuda.current_device())


def augment(x, o, gen=None):
    """x (B, 1, s, s) float in [0, 1]: a random 90-degree turn, horizontal and vertical flips (each drawn per picture), a scale
    in 1 +- o['scale'] (zeros at the edge), then brightness and contrast."""
    import torch
    import torch.nn.functional as F
    n = x.shape[0]
    r = lambda: torch.rand(n, device=x.device, generator=gen) * 2 - 1  # noqa: E731
    turns = torch.randint(0, 4, (n,), device=x.device, generator=gen).tolist()
    fh = (torch.rand(n, device=x.device, generator=gen) < 0.5).tolist()
    fv = (torch.rand(n, device=x.device, generator=gen) < 0.5).tolist()
    out = []
    for i in range(n):
        a = torch.rot90(x[i], turns[i], (1, 2))
        a = a.flip(2) if fh[i] else a
        out.append(a.flip(1) if fv[i] else a)
    x = torch.stack(out)
    sc = 1.0 + r() * o["scale"]
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 1, 1] = 1.0 / sc, 1.0 / sc
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    x = x * (1.0 + r() * o["contrast"]).view(n, 1, 1, 1) + (r() * o["brightness"]).view(n, 1, 1, 1)
    return x.clamp_(0, 1)


def parse_sets(items, epochs=None):
    o = dict(SETTINGS)
    opt = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if k in OPTIONAL:
            opt[k] = json.loads(v)
        elif k in SETTINGS:
            o[k] = json.loads(v)
        else:
            sys.exit("unknown setting %r; known: %s" % (k, ", ".join(list(SETTINGS) + list(OPTIONAL))))
    if "seed_offset" in opt and (not isinstance(opt["seed_offset"], int) or isinstance(opt["seed_offset"], bool) or opt["seed_offset"] < 0):
        sys.exit("seed_offset must be a non-negative integer, got %r" % (opt["seed_offset"],))
    if "px" in opt and (not isinstance(opt["px"], int) or isinstance(opt["px"], bool) or opt["px"] not in (256, 512)):
        sys.exit("px must be 256 or 512, got %r" % (opt["px"],))
    if opt.get("init") is not None and not (isinstance(opt["init"], str) and opt["init"] in INITS):
        sys.exit("init must be one of %s, got %r" % (", ".join(sorted(INITS)), opt["init"]))
    o.update({k: v for k, v in opt.items() if v != OPTIONAL[k]})
    if epochs:
        o["epochs"] = epochs
    return o


def run_seeds(o, fold, ep=0):
    """(net seed, batch-order seed of epoch ep): today's, shifted by 7919 * seed_offset (folds, early-stop rows and the
    normalisation sample never shift)."""
    k = SEED_STEP * int(o.get("seed_offset", 0))
    return SEED + fold + k, SEED * 7 + fold * 101 + ep + k


def lock_conflict(old, new):
    """None when the saved lock equals this run's {'type', 'settings'}; else the sentence that refuses the run."""
    if json.loads(json.dumps(old)) == json.loads(json.dumps(new)):
        return None
    return "this version was started with other settings (%s); a version's settings are locked, use a new --version" % json.dumps(old, sort_keys=True)


def out_dir(version):
    return os.path.join(OUT, "v%d" % version)


def make_net(reader, o, dev):
    widths = o["widths"] or reader.CFG["widths"]
    dropout = reader.CFG["dropout"] if o["dropout"] is None else o["dropout"]
    return reader.ChestNet(widths, dropout, reader.CFG["stem_stride"], 1).to(dev)


def load_init(net, name, say=print, path=None):
    """Copy the trunk tensors of one of our own readers' checkpoints into `net` (name and shape must both match); the head stays as built.
    Chest reader checkpoints keep their weights under `state`, CBIS ones under `net`. Returns the count copied."""
    import torch
    path = path or INITS[name]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state" in ck:
        src = ck["state"]
    elif "net" in ck:
        src = ck["net"]
    else:
        sys.exit("init %s: %s has neither a `state` nor a `net` key; keys found: %s" % (name, path, ", ".join(sorted(map(str, ck)))))
    have = net.state_dict()
    new = {k: v for k, v in src.items() if k.startswith("trunk.") and k in have and tuple(have[k].shape) == tuple(v.shape)}
    n_trunk = sum(1 for k in have if k.startswith("trunk."))
    skipped = sum(1 for k in src if k.startswith("trunk.")) - len(new)
    say("init %s: %s, %d of %d trunk tensors copied, %d skipped, head fresh" % (name, path, len(new), n_trunk, skipped))
    if len(new) < 0.9 * n_trunk:
        sys.exit("init %s copied only %d of %d trunk tensors; the checkpoint does not fit this net" % (name, len(new), n_trunk))
    net.load_state_dict(new, strict=False)
    return len(new)


def train_fold(d, pix, fold, o, dev, partial, fingerprint, net_path):
    """d: the training rows of the type (columns row, patient, y, fold as int). Returns (held-out positions, probabilities, best early-stop AUROC)."""
    import torch
    import torch.nn as nn
    from core import ckpt
    reader = mammo.reader_module()
    tr, va, te = mammo.split_fold(d, fold)
    grow, y = d.row.to_numpy(), d.y.to_numpy().astype(np.float32)
    torch.manual_seed(run_seeds(o, fold)[0])
    net = make_net(reader, o, dev)
    if o.get("init"):
        load_init(net, o["init"])
    opt = torch.optim.AdamW(net.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o["epochs"])
    samp = np.sort(np.random.default_rng(SEED).choice(grow[tr], min(200, len(tr)), replace=False))
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
        for xb, yb, _ in mammo.batches(pix, grow[tr], y[tr], o["batch"], True, run_seeds(o, fold, ep)[1]):
            if len(yb) < 2:
                continue
            x = augment(xb.to(dev).float() / 255.0, o)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=cuda):
                z = net((x - mean) / sd).float().squeeze(1)
            loss = loss_fn(z, yb.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + float(loss) * len(yb), n + len(yb)
        sched.step()
        a = auroc(y[va], mammo.predict(net, pix, grow[va], mean, sd, dev))
        hist.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val_auroc": a})
        if best_state is None or (a == a and a > best):
            best, best_state = (a if a == a else best), {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print("fold %d epoch %d/%d: loss %.4f, early-stop AUROC %.4f (best %.4f), %.0f s"
              % (fold, ep + 1, o["epochs"], tot / max(n, 1), a, best, time.time() - t0), flush=True)
        if every.due() or ep + 1 == o["epochs"]:
            ckpt.save_partial(partial, {"fingerprint": fingerprint, "net": net.state_dict(), "opt": opt.state_dict(),
                                        "sched": sched.state_dict(), "epoch": ep + 1, "best": best,
                                        "best_state": best_state, "hist": hist, "rng": ckpt.rng_state()})
            every.mark()
    net.load_state_dict(best_state)
    torch.save({"net": best_state, "mean": mean, "sd": sd}, net_path)
    return te, mammo.predict(net, pix, grow[te], mean, sd, dev), best, hist


def type_rows(ix, typ, split):
    return ix[(ix.type == typ) & (ix.split == split) & (ix.usable == 1)]


def need_pixels(rows, px=PX):
    """Refuse (returns the sentence) when a usable row has no prepared picture at px."""
    dp = done_path(px)
    done = np.fromfile(dp, dtype=np.uint8) if os.path.exists(dp) else None
    cmd = "prepare" if px == PX else "prepare --px %d" % px
    if done is None or len(done) <= int(rows.row.max()) or not os.path.exists(pix_path(px)):
        return "no prepared pictures at %d px: run `%s` first" % (px, cmd)
    miss = int((done[rows.row.to_numpy()] != 1).sum())
    return "%d of %d rows have no prepared picture at %d px: run `%s`" % (miss, len(rows), px, cmd) if miss else None


def dev_verdict(a):
    return "OPENS TEST" if a >= GATE else "STAYS ON DEVELOPMENT"


def dev_line(typ, a):
    s = "Verdict (idea 31 development, out-of-fold %s AUROC at least %.2f): %s" % (typ, GATE, dev_verdict(a))
    return s + (" (reported; calcifications have no comparator)" if typ == "calc" else "")


def develop(version, typ, ix):
    """Pools the folds' out-of-fold rows; prints and saves the development verdict when all five folds are in."""
    import pandas as pd
    from core import ledger
    d = out_dir(version)
    parts = [pd.read_csv(p, keep_default_na=False) for p in sorted(glob.glob(os.path.join(d, "fold[0-9].csv")))]
    if not parts:
        return 0
    oof = pd.concat(parts)
    oof.to_csv(os.path.join(d, "oof.csv"), index=False)
    have = sorted(int(k) for k in oof.fold.unique())
    print("oof.csv: %d rows from folds %s" % (len(oof), have))
    complete = have == list(range(FOLDS))
    dev = {"version": version, "type": typ, "folds": have, "complete": complete, "rows": len(oof), "verdict": None}
    if complete:
        a, lo, hi = patient_bootstrap(oof.patient, oof.y, oof.p, BOOT, BOOT_SEED)
        for k in have:
            f = oof[oof.fold == k]
            print("  fold %d: %d rows, AUROC %.4f" % (k, len(f), auroc(f.y, f.p)))
        print("out-of-fold %s AUROC %.4f (95%% CI %.4f to %.4f), %d rows, %d patients" % (typ, a, lo, hi, len(oof), oof.patient.nunique()))
        line = dev_line(typ, a)
        print(line)
        dev.update(auroc=a, low=lo, high=hi, verdict=dev_verdict(a), line=line, patients=int(oof.patient.nunique()))
        with ledger.run("experiment", "cbis v%d development (idea 31)" % version) as led:
            led.settings_from({"version": version, "type": typ, "gate": GATE, "bootstrap": BOOT})
            led.number("oof_auroc", a).number("oof_auroc_low", lo).number("oof_auroc_high", hi).number("rows", len(oof), "count")
            led.note_is(line)
    json.dump(dev, open(os.path.join(d, "dev.json"), "w"), indent=1)
    return 0


def train(version, typ, folds, epochs, sets):
    import torch
    from core import card, ckpt, ledger
    o = parse_sets(sets, epochs)
    d = out_dir(version)
    os.makedirs(d, exist_ok=True)
    lock = {"type": typ, "settings": o}
    lp = os.path.join(d, "lock.json")
    if os.path.exists(lp):
        why = lock_conflict(json.load(open(lp)), lock)
        if why:
            sys.exit("refused: " + why)
    else:
        json.dump(lock, open(lp, "w"), indent=1, sort_keys=True)
    ix = load_index()
    tr = type_rows(ix, typ, "train")
    tr = tr[tr.fold.astype(str).isin([str(i) for i in range(FOLDS)])].copy()
    why = need_pixels(tr, o.get("px", PX))
    if why:
        sys.exit("refused: " + why)
    tr["fold"] = tr.fold.astype(int)
    tr = tr.reset_index(drop=True)
    dev = device(torch)
    card.require_card(dev, torch)
    limit_card_memory(torch, dev, o.get("px", PX), o["epochs"])
    pix = pixels(len(ix), px=o.get("px", PX))
    fpd = {"version": version, "type": typ, "settings": o, "seed": SEED, "rows": len(tr)}
    if o.get("init"):
        ipath = INITS[o["init"]]
        fpd["init_file"] = {"path": ipath, "size": os.path.getsize(ipath), "mtime": int(os.path.getmtime(ipath))}
    fp = json.dumps(fpd, sort_keys=True)
    for k in folds:
        out_csv, net_path = os.path.join(d, "fold%d.csv" % k), os.path.join(d, "fold%d.pt" % k)
        meta, partial = out_csv + ".json", os.path.join(d, "fold%d.partial.pt" % k)
        if os.path.exists(out_csv) and os.path.exists(net_path) and os.path.exists(meta) and json.load(open(meta))["fingerprint"] == fp:
            print("fold %d: already done" % k)
            continue
        with ledger.run("experiment", "cbis v%d %s train fold %d (idea 31)" % (version, typ, k)) as led:
            led.settings_from(o)
            te, z, best, hist = train_fold(tr, pix, k, o, dev, partial, fp + "|fold%d" % k, net_path)
            f = tr.iloc[te][["row", "patient", "fold", "y"]].copy()
            f["p"] = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
            f.to_csv(out_csv, index=False)
            json.dump({"fingerprint": fp, "best_early_stop_auroc": best, "history": hist}, open(meta, "w"))
            ckpt.clear_partial(partial)
            led.number("early_stop_auroc", best).number("held_out_auroc", auroc(f.y, f.p))
            led.number("epochs", o["epochs"], "count")
            if dev.type == "cuda":
                led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
                led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)  # what the cap counts
                torch.cuda.reset_peak_memory_stats()
            print("fold %d: held-out AUROC %.4f -> %s" % (k, auroc(f.y, f.p), out_csv), flush=True)
    return develop(version, typ, ix)


# ---------------------------------------------------------------- test

def bar_line(name, bar, lo, hi, note=""):
    return "Against %s (%.3f)%s: %s" % (name, bar, note, verdict(lo, hi, bar))


def comparator_lines(lo, hi):
    return [bar_line("Tsochatzidis et al. 2019 from scratch", BAR_SCRATCH, lo, hi),
            bar_line("Tsochatzidis et al. 2019 fine-tuned from ImageNet", BAR_IMAGENET, lo, hi, ", not like for like")]


def step_kept(gain, d):
    return bool(gain > d and gain >= STEP_GAIN)


def step_line(step, gain, d):
    return "Verdict (idea 36 step %d, two-seed gain over the base above d and at least 0.01): %s" % (step, "KEPT" if step_kept(gain, d) else "NOT KEPT")


def step_gate(step, new_mean):
    return "Verdict (idea 36 step %d, the new two-seed mean out-of-fold AUROC at least 0.75): %s" % (step, dev_verdict(new_mean))


def compare_opens(rec, version):
    """True when a compare record (or None) names `version` as its first new version, KEPT, with the gate OPENS TEST."""
    return bool(rec and rec.get("new") and rec["new"][0] == version and rec.get("kept") is True and rec.get("gate") == "OPENS TEST")


def test_refusal(dev, nets_present, marker_path, cmp_rec=None, version=None):
    """None when test may open, else why not. dev: the saved development record (or None); marker_path: this type's test
    marker when it exists, else None; cmp_rec: out_dir(version)/compare-<version>.json's content when it exists (idea 36)."""
    if marker_path:
        return "test was already opened (%s exists); it is opened once" % marker_path
    if not dev or not dev.get("complete"):
        return "development is not complete over all five folds"
    if dev.get("verdict") != "OPENS TEST" and not compare_opens(cmp_rec, version):
        return "development's verdict is %s, not OPENS TEST" % (dev.get("verdict") or "missing")
    if not nets_present:
        return "a fold's net file is missing"
    return None


def oof_frame(versions):
    """(type, frame in row order with row, patient, y and a column p<v> per version); refuses locks of different types and
    versions that do not cover the same rows."""
    import pandas as pd
    out, typ = None, None
    for v in versions:
        d = out_dir(v)
        lp, fp = os.path.join(d, "lock.json"), os.path.join(d, "oof.csv")
        if not os.path.exists(lp) or not os.path.exists(fp):
            sys.exit("compare: v%d has no lock.json or oof.csv" % v)
        t = json.load(open(lp))["type"]
        if typ not in (None, t):
            sys.exit("compare: v%d is type %s, the others %s" % (v, t, typ))
        typ = t
        f = pd.read_csv(fp, keep_default_na=False).drop_duplicates("row").sort_values("row").reset_index(drop=True)
        if out is None:
            out = f[["row", "patient", "y"]].copy()
        elif len(f) != len(out) or not (f.row.to_numpy() == out.row.to_numpy()).all() or not (f.y.to_numpy() == out.y.to_numpy()).all():
            sys.exit("compare: v%d does not cover the same rows as v%d (%d against %d)" % (v, versions[0], len(f), len(out)))
        out["p%d" % v] = f.p.to_numpy(float)
    return typ, out


def compare(base, new, step):
    from core import ledger
    typ, df = oof_frame(base + new)
    au = {v: auroc(df.y, df["p%d" % v]) for v in base + new}
    d = abs(au[base[0]] - au[base[1]]) if len(base) > 1 else 0.0
    mean_base, mean_new = float(np.mean([au[v] for v in base])), float(np.mean([au[v] for v in new]))
    gain = mean_new - mean_base
    kept = step_kept(gain, d)
    gate = dev_verdict(mean_new) if kept else None
    with ledger.run("experiment", "cbis compare v%s against v%s (idea 36 step %d)" % (",".join(map(str, new)), ",".join(map(str, base)), step)) as led:
        led.settings_from({"base": base, "new": new, "step": step, "bootstrap": BOOT})
        for v in base + new:
            print("v%d out-of-fold AUROC %.4f" % (v, au[v]))
        print("d = |v%d - v%d| = %.4f; two-seed mean base %.4f, new %.4f, gain %.4f" % (base[0], base[-1], d, mean_base, mean_new, gain))
        diff, lo, hi = mammo.paired_bootstrap(df.patient, df.y, df["p%d" % new[0]], df["p%d" % base[0]], BOOT, BOOT_SEED)
        print("paired patient bootstrap, v%d - v%d: %+.4f (95%% CI %+.4f to %+.4f)" % (new[0], base[0], diff, lo, hi))
        line = step_line(step, gain, d)
        print(line)
        if kept:
            print(step_gate(step, mean_new))
        rec = {"base": base, "new": new, "step": step, "type": typ, "auroc": {str(v): au[v] for v in base + new}, "d": d, "gain": gain,
               "kept": kept, "gate": gate, "bootstrap": {"diff": diff, "low": lo, "high": hi}}
        led.number("d", d).number("gain", gain).number("new_mean_auroc", mean_new)
        led.number("diff", diff).number("diff_low", lo).number("diff_high", hi)
        led.note_is(line)
    json.dump(rec, open(os.path.join(out_dir(new[0]), "compare-%d.json" % new[0]), "w"), indent=1)
    return 0


def test(version):
    import pandas as pd
    import torch
    from core import card, ledger
    d = out_dir(version)
    lp, dp = os.path.join(d, "lock.json"), os.path.join(d, "dev.json")
    if not os.path.exists(lp):
        sys.exit("refused: no development run for version %d" % version)
    lock = json.load(open(lp))
    typ, o = lock["type"], lock["settings"]
    dev_rec = json.load(open(dp)) if os.path.exists(dp) else None
    nets = [os.path.join(d, "fold%d.pt" % k) for k in range(FOLDS)]
    mk = marker(typ)
    cp = os.path.join(d, "compare-%d.json" % version)
    cmp_rec = json.load(open(cp)) if os.path.exists(cp) else None
    why = test_refusal(dev_rec, all(os.path.exists(p) for p in nets), mk if os.path.exists(mk) else None, cmp_rec, version)
    ix = load_index()
    te = type_rows(ix, typ, "test").reset_index(drop=True)
    px = o.get("px", PX)
    why = why or need_pixels(te, px)
    if why:
        sys.exit("refused: " + why)
    excluded = set(ix[ix.fold.astype(str) == "excluded"].patient)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"version": version, "type": typ, "opened_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(te)}, open(mk, "w"))
    dv = device(torch)
    card.require_card(dv, torch)
    limit_card_memory(torch, dv, o.get("px", PX))
    reader = mammo.reader_module()
    pix = pixels(len(ix), px=px)
    rows = te.row.to_numpy()
    ps = []
    for p in nets:
        ck = torch.load(p, map_location="cpu")
        net = make_net(reader, o, dv)
        net.load_state_dict(ck["net"])
        z = mammo.predict(net, pix, rows, ck["mean"], ck["sd"], dv)
        ps.append(1.0 / (1.0 + np.exp(-z.astype(np.float64))))
    te["p"] = np.mean(ps, axis=0)
    for k, pk in enumerate(ps):
        te["p%d" % k] = pk
    te["excluded"] = te.patient.isin(excluded).astype(int)
    te[["row", "patient", "y", "p", "excluded"] + ["p%d" % k for k in range(FOLDS)]].to_csv(os.path.join(d, "test.csv"), index=False)
    a, lo, hi = patient_bootstrap(te.patient, te.y, te.p, BOOT, BOOT_SEED)
    keep = te[te.excluded == 0]
    ka, klo, khi = patient_bootstrap(keep.patient, keep.y, keep.p, BOOT, BOOT_SEED)
    print("test %s rows: %d (of %d in the CSVs; the rest unusable), %d patients" % (typ, len(te), int((ix[(ix.split == "test") & (ix.type == typ)].shape[0])), te.patient.nunique()))
    print("test AUROC %.4f (95%% CI %.4f to %.4f)" % (a, lo, hi))
    print("test AUROC without the %d excluded patients: %.4f (95%% CI %.4f to %.4f), %d rows" % (len(excluded & set(te.patient)), ka, klo, khi, len(keep)))
    lines = comparator_lines(lo, hi) if typ == "mass" else ["Test calc AUROC reported; calcifications have no comparator"]
    res = {"version": version, "type": typ, "rows": len(te), "auroc": a, "low": lo, "high": hi,
           "auroc_without_excluded": ka, "low_without_excluded": klo, "high_without_excluded": khi, "lines": lines}
    json.dump(res, open(os.path.join(d, "test.json"), "w"), indent=1)
    with ledger.run("experiment", "cbis v%d test (idea 31)" % version) as led:
        led.settings_from({"version": version, "type": typ, "bootstrap": BOOT})
        led.number("test_auroc", a).number("test_auroc_low", lo).number("test_auroc_high", hi)
        led.number("test_auroc_without_excluded", ka).number("rows", len(te), "count")
        led.note_is(lines[0])
    for ln in lines:
        print(ln)
    return 0


# ---------------------------------------------------------------- selftest

def selftest():
    import cv2
    import pandas as pd
    import torch
    # the folder mapping: the first path component only, the ROI folder by the abnormality id
    full, roi = folders_of("Mass-Test_P_00016_LEFT_CC/1.2.3/4.5.6/000000.dcm", 1)
    assert (full, roi) == ("Mass-Test_P_00016_LEFT_CC", "Mass-Test_P_00016_LEFT_CC_1"), (full, roi)
    assert folders_of("Calc-Training_P_00008_RIGHT_MLO\\a\\b\\c.dcm", 12)[1] == "Calc-Training_P_00008_RIGHT_MLO_12"
    # the mask is the file of the full image's size; no match or two matches is unusable, never a guess
    assert pick_mask((4000, 3000), [("crop.dcm", 300, 200), ("mask.dcm", 4000, 3000)]) == ("mask.dcm", "")
    assert pick_mask((4000, 3000), [("a.dcm", 4000, 3000), ("b.dcm", 300, 200)])[0] == "a.dcm"
    p, why = pick_mask((4000, 3000), [("a.dcm", 300, 200), ("b.dcm", 4001, 3000)])
    assert p is None and "no ROI file" in why, why
    p, why = pick_mask((10, 10), [("a.dcm", 10, 10), ("b.dcm", 10, 10)])
    assert p is None and "2 ROI files" in why, why
    # no full-size mask: the crop (the smaller of exactly two, smaller than the image both ways) locates the lesion instead
    assert pick_crop((4000, 3000), [("mask.dcm", 3480, 2610), ("crop.dcm", 300, 200)]) == ("crop.dcm", "")
    assert pick_crop((4000, 3000), [("a.dcm", 3480, 2610), ("b.dcm", 300, 200), ("c.dcm", 9, 9)])[0] is None
    assert pick_crop((400, 300), [("a.dcm", 500, 400), ("b.dcm", 300, 301)])[0] is None
    # find_crop: the crop's exact centre, in box_centre's convention, with a flat background beside the textured part
    g = np.random.default_rng(1)
    full = np.zeros((600, 450), np.uint16)
    tissue = cv2.GaussianBlur(g.normal(0, 1, (600, 330)).astype(np.float32), (0, 0), 12)
    tissue = 20000 + 40000 * (tissue - tissue.min()) / (tissue.max() - tissue.min()) + g.normal(0, 600, tissue.shape)
    full[:, 120:] = np.clip(tissue, 0, 65535).astype(np.uint16)
    for (y, x, h, w) in ((150, 180, 61, 50), (500, 380, 90, 70), (0, 122, 40, 33)):
        hit = find_crop(full, full[y:y + h, x:x + w])
        box = np.zeros(full.shape, np.uint8)
        box[y:y + h, x:x + w] = 1
        assert hit[:2] == box_centre(box) and hit[2] > 0.999, (y, x, hit)
    # one test marker per type
    assert marker("mass") != marker("calc") and marker("mass").endswith("test_opened_mass.json")
    # the label rule
    assert [label_of(x) for x in ("MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK", "odd")] == [1, 0, 0, None]
    # the window: centre right, zeros beyond the edge (a corner and a middle)
    a = np.arange(1, 50 * 60 + 1, dtype=np.uint16).reshape(50, 60)
    w = window(a, 3, 5, 16)                                    # near the top-left corner
    assert w.shape == (16, 16) and w[8, 8] == a[3, 5] and w[0, 0] == 0 and w[4, 4] == 0 and w[5, 3] == a[0, 0], w
    assert w[8:, 8:].min() > 0 and (w[:5] == 0).all() and (w[:, :3] == 0).all()
    w = window(a, 25, 30, 16)
    assert (w == a[17:33, 22:38]).all() and w[8, 8] == a[25, 30]
    assert window(a, 49, 59, 16)[8, 8] == a[49, 59] and (window(a, 49, 59, 16)[9:] == 0).all()
    m = np.zeros((40, 40), np.uint8)
    m[10:15, 20:30] = 255
    assert box_centre(m) == (12, 25) and box_centre(np.zeros((4, 4))) is None
    # the 1024 -> 256 area resample is the 4x4 block mean; the scale by the image's own range, MONOCHROME1 inverted
    r = np.random.default_rng(0).integers(0, 256, (WIN, WIN)).astype(np.uint8)
    s = shrink(r)
    ref = r.reshape(PX, 4, PX, 4).astype(np.float64).mean((1, 3))
    assert s.shape == (PX, PX) and np.abs(s.astype(np.float64) - ref).max() <= 1.0
    u = to_u8(np.array([[100, 200], [300, 500]], np.uint16))
    assert u.min() == 0 and u.max() == 255 and u[0, 0] == 0 and u[1, 1] == 255
    assert (to_u8(np.array([[100, 500]], np.uint16), True) == np.array([[255, 0]])).all()
    # folds: no patient in two folds, the shared patients excluded, stratified
    lab = {"p%03d" % i: int(i % 3 == 0) for i in range(300)}
    tests = {"p000", "p001", "p002", "q1", "q2"}
    f = assign_patient_folds(lab, tests)
    assert {p for p, k in f.items() if k == "excluded"} == {"p000", "p001", "p002"} and len(f) == 300
    assert set(v for v in f.values() if v != "excluded") == set(range(FOLDS))
    sizes = [sum(1 for v in f.values() if v == k) for k in range(FOLDS)]
    assert max(sizes) - min(sizes) <= 1 and f == assign_patient_folds(lab, tests)
    # the verdict and the comparator lines at their edges
    assert dev_verdict(0.75) == "OPENS TEST" and dev_verdict(0.7499) == "STAYS ON DEVELOPMENT"
    assert dev_line("mass", 0.8) == "Verdict (idea 31 development, out-of-fold mass AUROC at least 0.75): OPENS TEST"
    assert dev_line("calc", 0.6) == ("Verdict (idea 31 development, out-of-fold calc AUROC at least 0.75): STAYS ON DEVELOPMENT"
                                     " (reported; calcifications have no comparator)")
    c = lambda lo, hi: comparator_lines(lo, hi)  # noqa: E731
    assert c(0.72, 0.80)[0] == "Against Tsochatzidis et al. 2019 from scratch (0.716): BEATEN"
    assert c(0.716, 0.80)[0].endswith("LEVEL") and c(0.65, 0.716)[0].endswith("LEVEL") and c(0.60, 0.70)[0].endswith("BEHIND")
    assert c(0.65, 0.78)[1] == "Against Tsochatzidis et al. 2019 fine-tuned from ImageNet (0.804), not like for like: BEHIND"
    assert c(0.81, 0.85)[1].endswith("BEATEN") and c(0.70, 0.804)[1].endswith("LEVEL")
    # the test refusals and the allowed case
    ok = {"complete": True, "verdict": "OPENS TEST"}
    assert test_refusal(None, True, None) and test_refusal({"complete": False, "verdict": None}, True, None)
    assert "STAYS" in test_refusal({"complete": True, "verdict": "STAYS ON DEVELOPMENT"}, True, None)
    assert "already opened" in test_refusal(ok, True, marker("mass")) and test_refusal(ok, False, None)
    assert test_refusal(ok, True, None) is None
    # idea 36: optional settings stay out of o at their defaults; seeds; pixel paths; the step verdict; the compare gate
    assert parse_sets([]) == dict(SETTINGS) and parse_sets(["seed_offset=0", "px=256"]) == dict(SETTINGS)
    assert parse_sets(["seed_offset=1"])["seed_offset"] == 1 and "px" not in parse_sets(["seed_offset=1"])
    assert parse_sets(["px=512"])["px"] == 512 and "seed_offset" not in parse_sets(["px=512"])
    for bad in (["px=300"], ["seed_offset=-1"], ["seed_offset=true"], ["seed_offset=1.5"], ["px=true"]):
        try:
            parse_sets(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    assert run_seeds({}, 2, 3) == (SEED + 2, SEED * 7 + 2 * 101 + 3) and run_seeds({"seed_offset": 0}, 2, 3) == run_seeds({}, 2, 3)
    assert run_seeds({"seed_offset": 1}, 2, 3) == (SEED + 2 + 7919, SEED * 7 + 2 * 101 + 3 + 7919)
    assert pix_path(256) == PIX and DONE == done_path(256) and pix_path(512).endswith("cbis_512.u8") and done_path(512).endswith("cbis_512.done")
    assert step_line(1, 0.0099, 0.0).endswith("NOT KEPT") and step_line(1, 0.02, 0.01).endswith(": KEPT") and step_line(1, 0.02, 0.02).endswith("NOT KEPT")
    assert step_line(2, 0.02, 0.01) == "Verdict (idea 36 step 2, two-seed gain over the base above d and at least 0.01): KEPT"
    assert step_gate(2, 0.76).endswith("at least 0.75): OPENS TEST") and step_gate(2, 0.7).endswith("STAYS ON DEVELOPMENT")
    good = {"new": [3, 13], "kept": True, "gate": "OPENS TEST"}
    stay = {"complete": True, "verdict": "STAYS ON DEVELOPMENT"}
    assert test_refusal(stay, True, None, good, 3) is None
    assert test_refusal(stay, True, None, dict(good, kept=False), 3) and test_refusal(stay, True, None, dict(good, gate="STAYS ON DEVELOPMENT"), 3)
    assert test_refusal(stay, True, None, good, 13) and test_refusal(stay, True, None, None, 3)
    assert "already opened" in test_refusal(stay, True, marker("mass"), good, 3)
    # the lock
    assert lock_conflict({"type": "mass", "settings": SETTINGS}, {"type": "mass", "settings": dict(SETTINGS)}) is None
    assert lock_conflict({"type": "mass", "settings": SETTINGS}, {"type": "mass", "settings": dict(SETTINGS, lr=1e-3)})
    assert parse_sets(["lr=0.001"], 3)["lr"] == 0.001 and parse_sets([], 3)["epochs"] == 3
    # the augmentation: shape and range kept; with no scale, brightness or contrast each picture is one of the eight turns and flips
    x = torch.rand(6, 1, 32, 32)
    z = augment(x.clone(), {"scale": 0.0, "brightness": 0.0, "contrast": 0.0}, torch.Generator().manual_seed(3))
    for i in range(6):
        eight = [torch.rot90(t, k, (1, 2)) for k in range(4) for t in (x[i], x[i].flip(2))]
        assert any(torch.allclose(z[i], e, atol=1e-4) for e in eight), i
    z = augment(x.clone(), SETTINGS, torch.Generator().manual_seed(4))
    assert z.shape == x.shape and float(z.min()) >= 0 and float(z.max()) <= 1
    # the early-stop split used here: no patient in two of train, early stop and held out
    df = pd.DataFrame({"patient": ["p%d" % (i // 2) for i in range(160)], "y": [i % 2 for i in range(160)], "row": np.arange(160) + 7})
    df["fold"] = df.patient.map(lambda p: int(p[1:]) % FOLDS)
    tr, va, te = mammo.split_fold(df, 1)
    sets = [set(df.patient[i]) for i in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]) and len(va) > 0
    # the net builds as mammo builds it and returns one logit per input
    reader = mammo.reader_module()
    net = make_net(reader, dict(SETTINGS), torch.device("cpu")).eval()
    with torch.no_grad():
        out = net(torch.rand(2, 1, PX, PX))
    assert out.numel() == 2, out.shape
    assert cv2.__version__
    # idea 42: init stays out of o unless set; an unknown name is refused; every trunk tensor of a fake checkpoint is copied
    assert "init" not in parse_sets([]) and parse_sets(['init="chest-v103"'])["init"] == "chest-v103" and "init" not in parse_sets(["init=null"])
    for bad in (['init="nope"'], ["init=3"], ["init=[1]"]):
        try:
            parse_sets(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    import tempfile
    torch.manual_seed(1)
    donor = make_net(reader, dict(SETTINGS), torch.device("cpu"))
    torch.manual_seed(2)
    fresh = make_net(reader, dict(SETTINGS), torch.device("cpu"))
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "fake.pt")
        torch.save({"state": {k: v for k, v in donor.state_dict().items() if k.startswith("trunk.")}}, fake)
        n_copied = load_init(fresh, "chest-v103", path=fake)
    n_trunk = sum(1 for k in fresh.state_dict() if k.startswith("trunk."))
    assert n_copied == n_trunk > 0 and all(torch.equal(v, donor.state_dict()[k]) for k, v in fresh.state_dict().items() if k.startswith("trunk."))
    # idea 42 step 3: a CBIS checkpoint keeps its weights under `net` (loads all of them); one with neither key stops
    assert "cbis-v5" in INITS and parse_sets(['init="cbis-v5"'])["init"] == "cbis-v5"
    torch.manual_seed(3)
    fresh2 = make_net(reader, dict(SETTINGS), torch.device("cpu"))
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "fake_net.pt")
        torch.save({"mean": 0.0, "net": {k: v for k, v in donor.state_dict().items() if k.startswith("trunk.")}, "sd": 1.0}, fake)
        assert load_init(fresh2, "cbis-v5", say=lambda *a: None, path=fake) == n_trunk
        none = os.path.join(td, "fake_none.pt")
        torch.save({"mean": 0.0, "sd": 1.0}, none)
        try:
            load_init(fresh2, "cbis-v5", say=lambda *a: None, path=none)
        except SystemExit as e:
            assert "mean" in str(e) and "sd" in str(e), e
        else:
            raise AssertionError("a checkpoint with neither key was accepted")
    assert all(torch.equal(v, donor.state_dict()[k]) for k, v in fresh2.state_dict().items() if k.startswith("trunk."))
    print("selftest: all passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("index", "selftest"):
        sub.add_parser(n)
    pr = sub.add_parser("prepare")
    pr.add_argument("--type", choices=("mass", "calc", "both"), default="both")
    pr.add_argument("--px", type=int, choices=(256, 512), default=PX)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--type", choices=("mass", "calc"), default="mass")
    t.add_argument("--folds", default="0,1,2,3,4")
    t.add_argument("--epochs", type=int)
    t.add_argument("--set", action="append", default=[], metavar="key=<json>")
    e = sub.add_parser("test")
    e.add_argument("--version", type=int, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True, metavar="A,B")
    c.add_argument("--new", required=True, metavar="C,D")
    c.add_argument("--step", type=int, required=True)
    x = ap.parse_args()
    if x.cmd == "index":
        return index()
    if x.cmd == "prepare":
        return prepare(x.type, x.px)
    if x.cmd == "train":
        return train(x.version, x.type, [int(f) for f in x.folds.split(",")], x.epochs, x.set)
    if x.cmd == "test":
        return test(x.version)
    if x.cmd == "compare":
        return compare([int(v) for v in x.base.split(",")], [int(v) for v in x.new.split(",")], x.step)
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
