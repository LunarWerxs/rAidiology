"""OMAMA-DB as a pretraining stage for the CMMD readers (idea 103, docs/efficiency/README.md).

OMAMA-DB (Harvard Dataverse, MIT licence): 163,568 2D screening mammograms, one JSON of labels each. A reader is trained
on the used pictures (cancer or not, fold 0 only) and its trunk is the init="omama-v1" of the CMMD readers (cbis.INITS).

    python omama.py index [--rebuild] [--neg-ratio 4] [--seed 20261007]   the pictures used, five folds -> OUT/index.csv
    python omama.py prepare                               every used picture once into a uint8 memmap, 1024 x 640
    python omama.py pool [--limit N]                      every picture in META_INDEX, 2 x 2 means: 512 x 320 -> OUT/pool
    python omama.py train --version N [--epochs E] [--set key=<json> ...]  fold 0 only (mammo.train_fold)
    python omama.py cam-places                            omama-v1's top 3 cells per prepared picture -> OUT/omama_cam_places.csv
    python omama.py crops                                 512 px crops of the originals on G: at those cells -> OUT/omama_crops_3x512.u8
    python omama.py train-crops [--epochs E] [--set key=<json> ...]   found=3 MIL crop reader, fold 0 -> OUT/crops-v1/fold0.pt
    python omama.py check-geometry                        reads a few files on G:: a left and a right copy against their originals
    python omama.py selftest                            CPU, seconds, synthetic data in a temp folder
"""
import argparse
import functools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import mammo  # noqa: E402  (its import sets the repo path and KMP_BLOCKTIME)
from core import paths  # noqa: E402
from core.fetch import Turn  # noqa: E402
from core.rows import FileRows  # noqa: E402

PUBLIC = os.path.join(paths.BESIDE, "Datasets", "public", "omama-db-2d-1024", "2d_resized_1024")
IMG_DIR = os.path.join(PUBLIC, "images")
META_DIR = os.path.join(PUBLIC, "metadata")
META_INDEX = os.path.join(paths.WORK, "eval", "mammo", "omama_index.csv")
OUT = os.path.join(paths.WORK, "eval", "mammo", "omama")
INDEX = os.path.join(OUT, "index.csv")
PIX = os.path.join(OUT, "omama_1024x640.u8")
DONE = os.path.join(OUT, "omama_1024x640.done")
POOL = os.path.join(OUT, "pool")
POOL_PIX = os.path.join(POOL, "omama_pool_512x320.u8")
POOL_DONE = os.path.join(POOL, "omama_pool_512x320.done")
POOL_CSV = os.path.join(POOL, "omama_pool.csv")
POOL_JSON = os.path.join(POOL, "omama_pool.json")
H, W = mammo.H, mammo.W
PH, PW = H // 2, W // 2
FOLDS = mammo.FOLDS
SEED = 20261007
NEG_RATIO = 4
THREADS = 16
CHUNK = 2000
DEFAULT_INIT = "cbis-v5"
META_COLS = ["uid", "patient", "side", "view", "label", "spacing"]
COLS = ["path", "patient", "side", "view", "label", "fold"]
REFUSED = ("aux", "extra", "found", "lesions", "init_version")
ORIG = os.environ.get("RAIDIOLOGY_OMAMA_ORIG", os.path.join(paths.BESIDE, "Datasets", "public", "omama-db-2d", "2d"))
ORIG_IMG = os.path.join(ORIG, "images")
ORIG_META = os.path.join(ORIG, "metadata")
V1 = os.path.join(OUT, "v1", "fold0.pt")
CAM = os.path.join(OUT, "omama_cam_places.csv")
BOXES = os.path.join(OUT, "omama_cam_boxes.csv")
CROPS_PIX = os.path.join(OUT, "omama_crops_3x512.u8")
CROPS_DONE = os.path.join(OUT, "omama_crops_3x512.done")
CROPS_CAM = os.path.join(OUT, "omama_crops_3x512.cam.json")
CROPS_INDEX = os.path.join(OUT, "omama_crops_index.csv")
CROPS_OUT = os.path.join(OUT, "crops-v1")
LS = mammo.LS
PLACES = 3
TARGET_MM = 0.0941              # CMMD's pitch
RESAMPLE_MM = (0.07, 0.1)       # the spacings resampled to TARGET_MM; any other keeps its native pixels
CAM_CHUNK = 64
G_THREADS = 4                   # loads at a time from G:, a 5,400 rpm disk
BOX_SAVE = 2000                 # pictures between saves of the boxes
APART = 1.2                     # cells between places: see top_places
BOX_COLS = ["uid", "side", "box_y0", "box_x0", "mirror", "cw", "fit_s", "small"]
CROP_CHUNK = 200                # a chunk's crops wait in memory until written: 200 x 0.8 MB


# ---------------------------------------------------------------- the index

def meta_row(uid):
    try:
        with open(os.path.join(META_DIR, uid + ".json")) as f:
            m = json.load(f)
        return {"uid": uid, "patient": m.get("PatientID", ""), "side": m.get("ImageLaterality", ""),
                "view": m.get("View", ""), "label": m.get("label", ""),
                "spacing": (m.get("ImagerPixelSpacing") or [""])[0]}
    except Exception as e:  # noqa: BLE001  an unreadable JSON is counted, not fatal
        print("  %s: %s" % (uid, e), flush=True)
        return None


def scan_meta():
    rows, bad = [], 0
    with Turn([META_DIR], "omama index", rank=5) as t:
        names = sorted(f[:-len(".json")] for f in os.listdir(META_DIR) if f.endswith(".json"))
        for a in range(0, len(names), CHUNK):
            with ThreadPoolExecutor(THREADS) as ex:
                for r in ex.map(meta_row, names[a:a + CHUNK]):
                    if r is None:
                        bad += 1
                    else:
                        rows.append(r)
            t.share()
    print("metadata: %d pictures read, %d unreadable" % (len(rows), bad), flush=True)
    return pd.DataFrame(rows, columns=META_COLS)


def meta_table(rebuild):
    if rebuild or not os.path.exists(META_INDEX):
        meta = scan_meta()
        os.makedirs(os.path.dirname(META_INDEX), exist_ok=True)
        meta.to_csv(META_INDEX, index=False)
        return meta
    return pd.read_csv(META_INDEX, keep_default_na=False, dtype={"uid": str, "patient": str})


def select(meta, neg_ratio, seed):
    """The pictures used: every IndexCancer picture and neg_ratio times as many NonCancer ones (one per patient, seeded);
    PreIndexCancer and Unknown are left out. Folds are by patient, stratified by label (mammo.assign_folds)."""
    pos = meta[meta.label == "IndexCancer"]
    neg = meta[meta.label == "NonCancer"].drop_duplicates("patient")
    k = min(len(neg), neg_ratio * len(pos))
    neg = neg.iloc[np.sort(np.random.default_rng(seed).choice(len(neg), k, replace=False))]
    use = pd.concat([pos, neg]).sort_values("uid").reset_index(drop=True)
    cancer = (use.label == "IndexCancer").astype(int)
    folds = mammo.assign_folds(cancer.groupby(use.patient).max().to_dict(), seed=seed)
    return pd.DataFrame({"path": use.uid, "patient": use.patient, "side": use.side, "view": use.view,
                         "label": cancer, "fold": use.patient.map(folds).astype(int)})


def report(ix):
    print("pictures %d, patients %d" % (len(ix), ix.patient.nunique()))
    print("by label: IndexCancer %d, NonCancer %d" % ((ix.label == 1).sum(), (ix.label == 0).sum()))
    print("folds (pictures, patients): %s" % ", ".join(
        "%d: %d, %d" % (k, (ix.fold == k).sum(), ix[ix.fold == k].patient.nunique()) for k in range(FOLDS)))


def build_index(rebuild, neg_ratio, seed):
    ix = select(meta_table(rebuild), neg_ratio, seed)
    os.makedirs(OUT, exist_ok=True)
    ix.to_csv(INDEX, index=False)
    report(ix)
    return 0


# ---------------------------------------------------------------- the picture

def first_value(v):
    """The first number of a DICOM value: a list, a single number, or a backslash-separated string; None when empty."""
    if isinstance(v, str):
        v = v.split("\\")[0]
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else None
    return None if v in (None, "") else float(v)


def window_u8(data, meta):
    """uint8 at the picture's own size: the first window of the metadata (DICOM linear VOI, clipped to [c - w/2, c + w/2])
    when it has one, then min-max. Not yet cropped to the breast."""
    a = np.asarray(data, dtype=np.float32)
    c, w = first_value(meta.get("WindowCenter")), first_value(meta.get("WindowWidth"))
    if c is not None and w is not None:
        a = np.clip(a, c - w / 2, c + w / 2)
    return mammo.minmax_u8(a)


def picture(data, meta):
    """uint8 (H, W): window_u8, then breast_crop."""
    return mammo.breast_crop(window_u8(data, meta))


def load_one(job):
    """(row, uint8 (H, W) or None): a bad file is printed and left undone, not fatal."""
    i, uid, img_dir, meta_dir = job
    try:
        with np.load(os.path.join(img_dir, uid + ".npz")) as z:
            data = z["data"]
        with open(os.path.join(meta_dir, uid + ".json")) as f:
            meta = json.load(f)
        return i, picture(data, meta)
    except Exception as e:  # noqa: BLE001
        print("  %s: %s" % (uid, e), flush=True)
        return i, None


def fill_rows(uids, img_dir, meta_dir, pix_path, done_path, h, w, name, load, rank=5, order=None, chunk=CHUNK,
              threads=THREADS):
    """The loop shared by prepare, pool and crops: load(job) -> (row, uint8 (h, w) or None) for each uid, written into its row of
    the memmap and flagged in the done file; a done row is never read again. Returns the number of pictures left undone.
    order: the rows to do, in this order (default every row, in index order), or a function called inside the turn that
    returns them; chunk: rows loaded at a time; threads: loads at a time."""
    n = len(uids)
    fresh = not (os.path.exists(pix_path) and os.path.getsize(pix_path) == n * h * w and os.path.exists(done_path)
                 and os.path.getsize(done_path) == n)
    pix = np.memmap(pix_path, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n, h, w))
    done = np.memmap(done_path, dtype=np.uint8, mode="w+" if fresh else "r+", shape=(n,))
    t0 = time.time()
    with Turn([img_dir, meta_dir], "omama %s" % name, rank=rank) as t:
        rows = order() if callable(order) else (range(n) if order is None else order)
        todo = [(i, uids[i], img_dir, meta_dir) for i in rows if not done[i]]
        print("%s: %d pictures, %d to do" % (name, n, len(todo)), flush=True)
        for a in range(0, len(todo), chunk):
            with ThreadPoolExecutor(threads) as ex:
                for i, arr in ex.map(load, todo[a:a + chunk]):
                    if arr is not None:
                        pix[i] = arr
                        done[i] = 1
            pix.flush(), done.flush()
            print("  %d / %d, %.0f s" % (min(a + chunk, len(todo)), len(todo), time.time() - t0), flush=True)
            t.share()
    pix.flush(), done.flush()
    undone = n - int(done.sum())
    print("%s: done, %d of %d pictures, %d counted as failures" % (name, n - undone, n, undone))
    return undone


def prepare_rows(uids, img_dir, meta_dir, pix_path, done_path):
    """Every picture once, 1024 x 640, into its row of the memmap. Returns the number of pictures left undone."""
    return fill_rows(uids, img_dir, meta_dir, pix_path, done_path, H, W, "prepare", load_one)


def prepare():
    if not os.path.exists(INDEX):
        sys.exit("no %s: run `index` first" % INDEX)
    df = pd.read_csv(INDEX, keep_default_na=False, dtype={"path": str, "patient": str})
    os.makedirs(OUT, exist_ok=True)
    prepare_rows(list(df.path), IMG_DIR, META_DIR, PIX, DONE)
    return 0


# ---------------------------------------------------------------- the pool

def pool_one(job):
    """(row, uint8 (PH, PW) or None): load_one's picture with each 2 x 2 block replaced by its mean (float32, rounded)."""
    i, arr = load_one(job)
    if arr is None:
        return i, None
    return i, np.rint(arr.reshape(PH, 2, PW, 2).astype(np.float32).mean(axis=(1, 3))).astype(np.uint8)


def pool(limit):
    """Every picture in META_INDEX (cancer or not, no labels used), in its order, at 512 x 320 -> OUT/pool."""
    if not os.path.exists(META_INDEX):
        sys.exit("no %s: run `index` first" % META_INDEX)
    if limit is not None and limit < 1:
        sys.exit("--limit takes a count of at least 1")
    meta = pd.read_csv(META_INDEX, keep_default_na=False, dtype={"uid": str, "patient": str})
    if limit is not None:
        meta = meta.iloc[:limit]
    n = len(meta)
    os.makedirs(POOL, exist_ok=True)  # may be a link to another disk: left as it is
    pd.DataFrame({"row": list(range(n)), "uid": list(meta.uid), "patient": list(meta.patient)}).to_csv(POOL_CSV, index=False)
    t0 = time.time()
    # 2026-10-08: the pool reads G: at rank 2, ahead of the rank-3 multiview unpack and the rank-5 downloads (the data thread's ask)
    undone = fill_rows(list(meta.uid), IMG_DIR, META_DIR, POOL_PIX, POOL_DONE, PH, PW, "pool", pool_one, rank=2)
    info = {"rows": n, "done": n - undone, "h": PH, "w": PW, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - t0, 1), "source": "omama-db-2d-1024"}
    with open(POOL_JSON, "w") as f:
        json.dump(info, f)
    return 0


def write_csv(df, path):
    """The table written whole, then swapped in: a run cut short leaves the old file, never half of a new one."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cam_net(dev):
    """(net on dev in eval mode, mean, sd) of omama-v1's fold 0, built from the settings its checkpoint carries (as mammo.load_fold_net)."""
    import torch
    ck = torch.load(V1, map_location="cpu", weights_only=False)
    o = ck["settings"]
    reader = mammo.reader_module()
    widths = o.get("widths") or reader.CFG["widths"]
    dropout = reader.CFG["dropout"] if o.get("dropout") is None else o["dropout"]
    net = mammo.make_net(reader, widths, dropout, False)
    net.load_state_dict(ck["net"])
    return net.to(dev).eval(), ck["mean"], ck["sd"]


def cell_logits(net, x):
    """(B, fh, fw) logits of x (B, 1, H, W), already normalised: the head applied to each cell of the trunk's last map. ChestNet
    mean-pools before its linear head, so the cells' mean is the picture's logit: an exact class-activation map, no Grad-CAM."""
    import torch
    if net.renderings == 3:
        x = mammo.reader_module().add_renderings(x)
    if net.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return net.head(net.trunk(x).permute(0, 2, 3, 1))[..., 0]


def top_places(grid, places=PLACES, apart=APART):
    """[(gy, gx, logit)] of a (fh, fw) grid: the largest cell, then (mammo.pick_cells) the largest at least `apart` cells from
    all taken. omama-v1's grid has a 64 px stride, a cell about 13 mm at its 0.21 mm per pixel, so apart=1.2 drops the four
    neighbours (13 mm) and keeps the diagonals (19 mm): places 19 mm or more apart (v33's found=3 places were 6 cells of 16 px,
    about 26 mm); the three largest cells alone are mostly neighbours."""
    return [(int(gy), int(gx), float(grid[gy, gx])) for gy, gx in mammo.pick_cells(grid, places, apart)]


def box_one(uid, side):
    """breast_crop's transform and the 1024 copy's side as a table row, from the copy with prepare's own window and crop;
    None when the copy cannot be read (printed, left out)."""
    try:
        with np.load(os.path.join(IMG_DIR, uid + ".npz")) as z:
            data = z["data"]
        with open(os.path.join(META_DIR, uid + ".json")) as f:
            meta = json.load(f)
        t = mammo.breast_crop(window_u8(data, meta), transform=True)[1]
        return {"uid": uid, "side": side, "box_y0": t["y0"], "box_x0": t["x0"], "mirror": int(t["mirror"]), "cw": t["cw"],
                "fit_s": t["s"], "small": max(data.shape)}
    except Exception as e:  # noqa: BLE001
        print("  %s: %s" % (uid, e), flush=True)
        return None


def read_boxes(df):
    """The boxes of the prepared pictures: the saved ones, then the rest read from the 1024 copies inside one G: turn (the
    omama-db-2d-1024 folder is a junction onto G:), saved every BOX_SAVE pictures; a rerun reads only what is missing."""
    have = {}
    if os.path.exists(BOXES):
        have = {r["uid"]: r for r in pd.read_csv(BOXES, keep_default_na=False, dtype={"uid": str}).to_dict("records")}
    todo = [(u, s) for u, s in zip(df.path, df.side) if u not in have]
    print("boxes: %d pictures, %d to read" % (len(df), len(todo)), flush=True)
    if not todo:  # every box is saved: no G: turn needed
        return have
    t0 = time.time()
    with Turn([IMG_DIR, META_DIR], "omama cam-places", rank=2) as t:
        for a in range(0, len(todo), BOX_SAVE):
            with ThreadPoolExecutor(G_THREADS) as ex:
                for r in ex.map(lambda us: box_one(*us), todo[a:a + BOX_SAVE]):
                    if r is not None:
                        have[r["uid"]] = r
            write_csv(pd.DataFrame(list(have.values()), columns=BOX_COLS), BOXES)
            print("  %d / %d, %.0f s" % (min(a + BOX_SAVE, len(todo)), len(todo), time.time() - t0), flush=True)
            t.share()
    return have


def cam_places():
    """omama-v1's three places of every prepared picture (top_places of its class-activation map) with the box that places them
    on the original: the boxes first, read from the 1024 copies under the G: turn (rank 2, as pool) and saved to
    OUT/omama_cam_boxes.csv, then the net -> OUT/omama_cam_places.csv (written whole at the end: cell rows and columns on the
    stride grid, their logits, the picture's logit)."""
    import torch
    df, pix = prepared_rows()
    boxes = read_boxes(df)
    mammo.limit_card_memory(torch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, mean, sd = cam_net(dev)
    recs, t0 = [], time.time()
    with torch.no_grad():       # full precision, as locate: the cells' logits are the same at any batch size
        for a in range(0, len(df), CAM_CHUNK):
            r = np.arange(a, min(a + CAM_CHUNK, len(df)))
            x = torch.from_numpy(np.ascontiguousarray(pix[r]))[:, None].to(dev).float() / 255.0
            g = cell_logits(net, (x - mean) / sd).float().cpu().numpy()
            stride = H // g.shape[1]
            if stride * g.shape[1] != H or stride * g.shape[2] != W:
                sys.exit("the cell grid %s does not tile %d x %d" % (g.shape[1:], H, W))
            for j, uid in enumerate(df.path.iloc[r]):
                if uid not in boxes:
                    continue
                rec = dict(boxes[uid], picture_logit=float(g[j].mean()), stride=stride)
                for q, (gy, gx, z) in enumerate(top_places(g[j]), 1):
                    rec.update({"gy%d" % q: gy, "gx%d" % q: gx, "logit%d" % q: z})
                recs.append(rec)
            print("  %d / %d, %.0f s" % (min(a + CAM_CHUNK, len(df)), len(df), time.time() - t0), flush=True)
    cols = BOX_COLS[:1] + ["gy1", "gx1", "logit1", "gy2", "gx2", "logit2", "gy3", "gx3", "logit3", "picture_logit", "stride"] \
        + BOX_COLS[1:]
    write_csv(pd.DataFrame(recs, columns=cols), CAM)
    print("cam-places: %d of %d prepared pictures, %.0f s -> %s" % (len(recs), len(df), time.time() - t0, CAM), flush=True)
    return 0


def native_crop(full, cx, cy, ratio, mirror):
    """LS x LS uint8 of the original `full` around (cx, cy): a window of LS * ratio pixels (made even), black past its edge,
    resized to LS (the window's centre pixel becomes the crop's), mirrored when `mirror` as mammo.crop_at does."""
    import cv2
    R = 2 * int(round(LS * ratio / 2))
    win = np.zeros((R, R), np.uint8)
    x0, y0 = cx - R // 2, cy - R // 2
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + R, full.shape[0]), min(x0 + R, full.shape[1])
    if sy1 > sy0 and sx1 > sx0:
        win[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = full[sy0:sy1, sx0:sx1]
    if R != LS:
        win = cv2.resize(win, (LS, LS), interpolation=cv2.INTER_AREA if R > LS else cv2.INTER_LINEAR)
    if mirror:
        win = win[:, ::-1]
    return np.ascontiguousarray(win)


def copy_to_original(xs, ys, full_shape, small, side):
    """A point (x, y) of the 1024 copy -> the original's pixel (x, y), floats. The copy is the original scaled by
    small / max(shape) and padded to small x small; a right breast's picture is pushed right by the padding, a left one's
    starts at x = 0."""
    scale = small / max(full_shape)
    ox = small - int(round(full_shape[1] * scale)) if side == "R" else 0
    return (xs - ox + 0.5) / scale - 0.5, (ys + 0.5) / scale - 0.5


def places_crops(full, rec, spacing):
    """(PLACES * LS, LS) uint8, the picture's places in order: each cell's centre in the 1024 x 640 picture is mapped back through
    breast_crop's transform to the 1024 copy, then (copy_to_original) to the original's pixels."""
    t = {"y0": int(rec["box_y0"]), "x0": int(rec["box_x0"]), "mirror": bool(int(rec["mirror"])), "cw": int(rec["cw"]),
         "s": float(rec["fit_s"])}
    ratio = TARGET_MM / spacing if spacing is not None and any(abs(spacing - m) < 1e-4 for m in RESAMPLE_MM) else 1.0
    stride = int(rec["stride"])
    out = np.zeros((PLACES * LS, LS), np.uint8)
    for q in range(PLACES):
        y = (int(rec["gy%d" % (q + 1)]) + 0.5) * stride - 0.5
        x = (int(rec["gx%d" % (q + 1)]) + 0.5) * stride - 0.5
        xs, ys = mammo.unmap_point(x, y, t)
        xf, yf = copy_to_original(xs, ys, full.shape, int(rec["small"]), rec["side"])
        out[q * LS:(q + 1) * LS] = native_crop(full, int(round(xf)), int(round(yf)), ratio, t["mirror"])
    return out


def spacing_of(text):
    """A picture's ImagerPixelSpacing in mm from the metadata table; None when it has none."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def crop_one(job, recs, spacings):
    """(row, uint8 (PLACES * LS, LS) or None): one picture's places cut from its original on G:; a bad file is printed and left undone."""
    i, uid, img_dir, meta_dir = job
    try:
        with np.load(os.path.join(img_dir, uid + ".npz")) as z:
            data = z["data"]
        with open(os.path.join(meta_dir, uid + ".json")) as f:
            meta = json.load(f)
        return i, places_crops(window_u8(data, meta), recs[i], spacings[i])
    except Exception as e:  # noqa: BLE001
        print("  %s: %s" % (uid, e), flush=True)
        return i, None


def crops():
    """The 512 px crops at each picture's cam places, from the originals on G: (one G: turn at rank 2, as pool; the files in
    creation order from one listing taken inside the turn, G_THREADS at a time)
    -> OUT/omama_crops_3x512.u8 (row i holds picture i's three crops) and OUT/omama_crops_index.csv. The size and time of the
    cam csv are kept beside the done flags: a changed cam csv puts the crops back to none done."""
    if not (os.path.exists(INDEX) and os.path.exists(CAM)):
        sys.exit("run `index` and `cam-places` first")
    for d in (ORIG_IMG, ORIG_META):
        if not os.path.isdir(d):
            sys.exit("no %s: the originals are at Datasets/public/omama-db-2d/2d (or set RAIDIOLOGY_OMAMA_ORIG)" % d)
    stamp = {"size": os.path.getsize(CAM), "mtime": int(os.path.getmtime(CAM))}
    if os.path.exists(CROPS_DONE):
        old = json.load(open(CROPS_CAM)) if os.path.exists(CROPS_CAM) else None
        if old != stamp:
            print("crops: the cam csv changed since the crops were cut; starting over", flush=True)
            os.remove(CROPS_DONE)
    with open(CROPS_CAM, "w") as f:
        json.dump(stamp, f)
    df = pd.read_csv(INDEX, keep_default_na=False, dtype={"path": str, "patient": str})
    meta = pd.read_csv(META_INDEX, keep_default_na=False, dtype={"uid": str, "patient": str})
    spacing = {u: spacing_of(s) for u, s in zip(meta.uid, meta.spacing)}
    cam = pd.read_csv(CAM, keep_default_na=False, dtype={"uid": str})
    recs = {r["uid"]: r for r in cam.to_dict("records")}
    n = len(df)
    have = [i for i in range(n) if df.path.iloc[i] in recs]
    spc = [spacing.get(u) for u in df.path]
    index = df[["path", "patient", "side", "view", "label", "fold"]].copy()
    index["spacing"] = ["" if s is None else s for s in spc]
    index["places"] = df.path.isin(list(recs)).astype(int)
    os.makedirs(OUT, exist_ok=True)
    write_csv(index, CROPS_INDEX)
    print("crops: %d pictures with places, %d without (not prepared, or the 1024 copy unreadable)" % (len(have), n - len(have)),
          flush=True)
    def disk_order():
        made = {e.name[:-len(".npz")]: e.stat().st_ctime for e in os.scandir(ORIG_IMG) if e.name.endswith(".npz")}
        return sorted((i for i in have if df.path.iloc[i] in made), key=lambda i: made[df.path.iloc[i]])

    t0 = time.time()
    load = functools.partial(crop_one, recs=[recs.get(u) for u in df.path], spacings=spc)
    fill_rows(list(df.path), ORIG_IMG, ORIG_META, CROPS_PIX, CROPS_DONE, PLACES * LS, LS, "crops", load, rank=2,
              order=disk_order, chunk=CROP_CHUNK, threads=G_THREADS)
    print("crops: %.0f s -> %s" % (time.time() - t0, CROPS_PIX), flush=True)
    return 0


# ---------------------------------------------------------------- training

class Picked:
    """The prepared rows of the index, by position: pix[r] is the memmap row rows[r]."""

    def __init__(self, mem, rows):
        self.mem, self.rows = mem, rows

    def __getitem__(self, r):
        return self.mem[self.rows[np.asarray(r)]]


class PickedPlaces:
    """The crops of the done pictures by position: pix[r] is the `places` crops of picture rows[r], picture-major."""

    def __init__(self, mem, rows, places):
        self.mem, self.rows, self.places = mem, rows, places

    def __getitem__(self, r):
        r = self.rows[np.asarray(r)]
        return self.mem[(r[:, None] * self.places + np.arange(self.places)[None, :]).reshape(-1)]


def prepared_rows():
    if not (os.path.exists(INDEX) and os.path.exists(DONE) and os.path.exists(PIX)):
        sys.exit("run `index` and `prepare` first")
    df = pd.read_csv(INDEX, keep_default_na=False, dtype={"path": str, "patient": str})
    n = len(df)
    done = np.array(np.memmap(DONE, dtype=np.uint8, mode="r", shape=(n,)))
    rows = np.nonzero(done)[0]
    if not len(rows):
        sys.exit("no prepared pictures: run `prepare` first")
    print("prepared %d of %d pictures" % (len(rows), n), flush=True)
    return df.iloc[rows].reset_index(drop=True), Picked(FileRows(PIX, (n, H, W)), rows)


def fit(df, pix, o, dev, d, fp):
    """Fold 0's trunk: mammo.train_fold, then fold0.pt, fold0.csv and fold0.csv.json in d. Returns (best early-stop AUROC, held-out AUROC)."""
    import torch
    from core import ckpt
    partial = os.path.join(d, "fold0.partial.pt")
    te, z, best, hist, info, best_state, mean, sd = mammo.train_fold(df, pix, 0, o, dev, partial, fp + "|fold0", None, None)
    f = df.iloc[te][COLS].copy()
    f["logit"], f["p"] = z, 1.0 / (1.0 + np.exp(-z))
    torch.save({"net": best_state, "mean": mean, "sd": sd, "settings": o}, os.path.join(d, "fold0.pt"))
    f.to_csv(os.path.join(d, "fold0.csv"), index=False)
    with open(os.path.join(d, "fold0.csv.json"), "w") as fh:
        json.dump({"fingerprint": fp, "best_early_stop_auroc": best, "history": hist, **info}, fh)
    ckpt.clear_partial(partial)
    return best, mammo.auroc(f.label, f.logit)


def train(version, epochs, sets):
    import torch
    from core import ledger, polite  # noqa: F401
    items = list(sets)
    if not any(s.partition("=")[0] == "init" for s in items):
        items.insert(0, 'init="%s"' % DEFAULT_INIT)
    o = mammo.parse_sets(items)
    for k in REFUSED:
        if o.get(k):
            sys.exit("omama training takes no %s setting" % k)
    if epochs:
        o["epochs"] = epochs
    mammo.limit_card_memory(torch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, pix = prepared_rows()
    d = os.path.join(OUT, "v%d" % version)
    os.makedirs(d, exist_ok=True)
    fpd = {"version": version, "dataset": "omama", "settings": o, "seed": SEED, "pictures": len(df),
           "index": {"size": os.path.getsize(INDEX), "mtime": int(os.path.getmtime(INDEX))}}
    if o.get("init"):
        p = mammo.cbis_module().INITS[o["init"]]
        fpd["init_file"] = {"path": p, "size": os.path.getsize(p), "mtime": int(os.path.getmtime(p))}
    fp = json.dumps(fpd, sort_keys=True)
    out_csv = os.path.join(d, "fold0.csv")
    meta_p = out_csv + ".json"
    if os.path.exists(out_csv) and os.path.exists(meta_p):
        with open(meta_p) as f:
            if json.load(f)["fingerprint"] == fp:
                print("fold 0: already done")
                return 0
    with ledger.run("experiment", "omama v%d train fold 0" % version) as led:
        led.settings_from(o)
        best, held = fit(df, pix, o, dev, d, fp)
        led.number("early_stop_auroc", best).number("held_out_auroc", held)
        led.number("epochs", o["epochs"], "count")
        if dev.type == "cuda":
            led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
            led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)
            torch.cuda.reset_peak_memory_stats()
    print("fold 0: held-out AUROC %.4f -> %s" % (held, out_csv), flush=True)
    return 0


def crop_rows():
    if not (os.path.exists(CROPS_INDEX) and os.path.exists(CROPS_DONE) and os.path.exists(CROPS_PIX)):
        sys.exit("run `crops` first")
    df = pd.read_csv(CROPS_INDEX, keep_default_na=False, dtype={"path": str, "patient": str})
    n = len(df)
    rows = np.nonzero(np.array(np.memmap(CROPS_DONE, dtype=np.uint8, mode="r", shape=(n,))))[0]
    if not len(rows):
        sys.exit("no crops yet: run `crops` first")
    print("crops of %d of %d pictures" % (len(rows), n), flush=True)
    return df.iloc[rows].reset_index(drop=True), PickedPlaces(FileRows(CROPS_PIX, (n * PLACES, LS, LS)), rows, PLACES)


def train_crops(epochs, sets):
    """A found=3 MIL crop reader on the OMAMA crops, fold 0 (mammo.train_fold, as v1): the largest of a picture's three crop logits
    against its label. Starts from cbis-v5 unless init is given. -> OUT/crops-v1/fold0.pt, the start cbis.INITS names."""
    import torch
    from core import ledger, polite  # noqa: F401
    items = list(sets)
    if any(s.partition("=")[0] in ("found", "init_version") for s in items):
        sys.exit("train-crops sets found=%d itself: it takes no found or init_version setting" % PLACES)
    if not any(s.partition("=")[0] == "init" for s in items):
        items.insert(0, 'init="%s"' % DEFAULT_INIT)
    o = mammo.parse_sets(["found=%d" % PLACES] + items)
    for k in ("aux", "extra", "lesions", "init_version"):
        if o.get(k):
            sys.exit("omama crops training takes no %s setting" % k)
    if epochs:
        o["epochs"] = epochs
    mammo.limit_card_memory(torch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, pix = crop_rows()
    d = CROPS_OUT
    os.makedirs(d, exist_ok=True)
    fpd = {"version": "crops-v1", "dataset": "omama-crops", "settings": o, "seed": SEED, "pictures": len(df),
           "index": {"size": os.path.getsize(CROPS_INDEX), "mtime": int(os.path.getmtime(CROPS_INDEX))},
           "crops": {"size": os.path.getsize(CROPS_PIX), "mtime": int(os.path.getmtime(CROPS_PIX))}}
    if o.get("init"):
        p = mammo.cbis_module().INITS[o["init"]]
        fpd["init_file"] = {"path": p, "size": os.path.getsize(p), "mtime": int(os.path.getmtime(p))}
    fp = json.dumps(fpd, sort_keys=True)
    out_csv = os.path.join(d, "fold0.csv")
    meta_p = out_csv + ".json"
    if os.path.exists(out_csv) and os.path.exists(meta_p):
        with open(meta_p) as f:
            if json.load(f)["fingerprint"] == fp:
                print("fold 0: already done")
                return 0
    with ledger.run("experiment", "omama crops-v1 train fold 0") as led:
        led.settings_from(o)
        best, held = fit(df, pix, o, dev, d, fp)
        led.number("early_stop_auroc", best).number("held_out_auroc", held)
        led.number("epochs", o["epochs"], "count")
        share, n_pos = top_crop_share(pd.read_csv(out_csv, keep_default_na=False, dtype={"path": str}), CAM)
        if n_pos:
            led.number("top_crop_above_picture_share", share)
        if dev.type == "cuda":
            led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
            led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)
            torch.cuda.reset_peak_memory_stats()
    print("fold 0: held-out AUROC %.4f -> %s" % (held, out_csv), flush=True)
    print("fold 0: %d held-out positives, the top crop logit exceeds omama-v1's picture logit for %.3f of them" % (n_pos, share),
          flush=True)
    return 0


def top_crop_share(held, cam_path):
    """(share, positives): of the held-out positives of fold0.csv (its logit is the largest of the picture's crop logits), the
    share whose logit exceeds omama-v1's picture logit in the cam csv."""
    cam = pd.read_csv(cam_path, keep_default_na=False, dtype={"uid": str}).set_index("uid").picture_logit
    pos = held[(held.label == 1) & held.path.isin(cam.index)]
    if not len(pos):
        return float("nan"), 0
    return float((pos.logit.to_numpy() > cam.loc[pos.path].to_numpy()).mean()), len(pos)


def check_geometry(count=1):
    """Reads real files on G: (a few): for `count` left and `count` right pictures of the index, a patch of the 1024 copy and the same
    place cut from the original through copy_to_original must correlate (>= 0.9). The only check that touches G:, run by hand."""
    import cv2
    df = pd.read_csv(INDEX, keep_default_na=False, dtype={"path": str, "patient": str})
    bad = 0
    for side in ("L", "R"):
        for uid in list(df[df.side == side].path[:count]):
            with np.load(os.path.join(ORIG_IMG, uid + ".npz")) as z:
                full = z["data"]
            with np.load(os.path.join(IMG_DIR, uid + ".npz")) as z:
                small = z["data"]
            with open(os.path.join(ORIG_META, uid + ".json")) as f:
                meta = json.load(f)
            a, b = window_u8(small, meta).astype(np.float32), window_u8(full, meta)
            ys, xs = np.nonzero(cv2.GaussianBlur(a, (0, 0), 20) > 0.5 * cv2.GaussianBlur(a, (0, 0), 20).max())
            x, y, hw = int(xs.mean()), int(ys.mean()), 48
            patch = a[y - hw:y + hw, x - hw:x + hw]
            xf, yf = copy_to_original(x, y, full.shape, max(small.shape), side)
            r = int(round(hw * max(full.shape) / max(small.shape)))
            cut = cv2.resize(native_crop(b, int(round(xf)), int(round(yf)), 2 * r / LS, False), (2 * hw, 2 * hw),
                             interpolation=cv2.INTER_AREA)
            c = float(np.corrcoef(patch.ravel(), cut.astype(np.float32).ravel())[0, 1])
            print("%s %s: copy %s, original %s, correlation %.3f" % (side, uid, small.shape, full.shape, c), flush=True)
            bad += c < 0.9
    return 1 if bad else 0


# ---------------------------------------------------------------- selftest

def check_select():
    rows = []
    for p in range(8):
        rows.append(("c%dR" % p, "C%d" % p, "R", "cranio-caudal", "IndexCancer", "0.09"))
        rows.append(("c%dL" % p, "C%d" % p, "L", "cranio-caudal", "Unknown", "0.09"))
    for p in range(4):
        rows.append(("q%d" % p, "Q%d" % p, "L", "cranio-caudal", "PreIndexCancer", "0.09"))
    for p in range(40):
        rows.append(("n%d" % p, "N%d" % p, "R", "cranio-caudal", "NonCancer", "0.09"))
    meta = pd.DataFrame(rows, columns=META_COLS)
    ix = select(meta, NEG_RATIO, 1)
    assert set(ix.path) <= {"c%dR" % p for p in range(8)} | {"n%d" % p for p in range(40)}, sorted(ix.path)
    assert (ix.label == 1).sum() == 8 and (ix.label == 0).sum() == 32, ix.label.value_counts()
    assert ix.path.is_unique and ix.patient.is_unique
    assert ix.groupby("patient").fold.nunique().max() == 1
    assert set(ix.fold) <= set(range(FOLDS))
    assert select(meta, NEG_RATIO, 1).equals(ix)
    few = meta[(meta.label == "IndexCancer") | meta.patient.isin(["N0", "N1"])]
    assert (select(few, NEG_RATIO, 1).label == 0).sum() == 2


def fake_breast(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:1024, :1024]
    inside = ((xx - 900) / 700.0) ** 2 + ((yy - 512) / 450.0) ** 2 < 1
    vals = 500 + 3000 * xx / 1023.0 + rng.integers(0, 50, (1024, 1024))
    return np.where(inside, vals, 0).astype(np.uint16)


def check_prepare(tmp):
    img, meta_dir = os.path.join(tmp, "images"), os.path.join(tmp, "metadata")
    os.makedirs(img)
    os.makedirs(meta_dir)
    np.savez(os.path.join(img, "win.npz"), data=fake_breast(0))
    np.savez(os.path.join(img, "plain.npz"), data=fake_breast(1))
    np.savez(os.path.join(img, "single.npz"), data=fake_breast(2))
    with open(os.path.join(img, "corrupt.npz"), "wb") as f:
        f.write(b"not a zip file")
    win = {"WindowCenter": [2000.0, 1900.0], "WindowWidth": [1500.0, 900.0], "label": "IndexCancer"}
    single = {"WindowCenter": 2000.0, "WindowWidth": 1500.0, "label": "NonCancer"}
    for uid, m in (("win", win), ("plain", {"label": "NonCancer"}), ("single", single),
                   ("corrupt", {"label": "NonCancer"})):
        with open(os.path.join(meta_dir, uid + ".json"), "w") as f:
            json.dump(m, f)
    pix_p, done_p = os.path.join(tmp, "p.u8"), os.path.join(tmp, "p.done")
    uids = ["win", "plain", "single", "corrupt"]
    assert prepare_rows(uids, img, meta_dir, pix_p, done_p) == 1
    assert np.fromfile(done_p, np.uint8).tolist() == [1, 1, 1, 0]
    pix = np.fromfile(pix_p, np.uint8).reshape(4, H, W)
    assert np.array_equal(pix[0], picture(fake_breast(0), win))
    assert not np.array_equal(pix[0], picture(fake_breast(0), {})), "the window changed nothing"
    assert np.array_equal(pix[1], picture(fake_breast(1), {}))
    assert np.array_equal(pix[2], picture(fake_breast(2), single)) and pix[2].max() > 0
    assert pix[0].max() > 0 and pix[3].max() == 0
    pool_p, pool_done_p = os.path.join(tmp, "q.u8"), os.path.join(tmp, "q.done")
    assert fill_rows(uids, img, meta_dir, pool_p, pool_done_p, PH, PW, "pool", pool_one) == 1
    assert np.fromfile(pool_done_p, np.uint8).tolist() == [1, 1, 1, 0]
    got = np.fromfile(pool_p, np.uint8).reshape(4, PH, PW)
    for r in range(3):
        p = pix[r].astype(np.float32)
        mean = (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]) / 4
        assert np.array_equal(got[r], np.rint(mean).astype(np.uint8)), "pool row %d is not the 2 x 2 mean" % r
    np.savez(os.path.join(img, "plain.npz"), data=fake_breast(5))
    assert prepare_rows(uids, img, meta_dir, pix_p, done_p) == 1
    assert np.array_equal(np.fromfile(pix_p, np.uint8).reshape(4, H, W)[1], pix[1]), "a done row was read again"


def check_train(tmp):
    import torch
    n, extra = 50, 2
    uids = ["t%02d" % i for i in range(n)] + ["u0", "u1"]
    df = pd.DataFrame({"path": uids, "patient": ["P%02d" % i for i in range(n)] + ["U0", "U1"], "side": "L",
                       "view": "cranio-caudal", "label": [i % 2 for i in range(n)] + [1, 1],
                       "fold": [(i // 2) % FOLDS for i in range(n)] + [0, 0]})
    full = np.random.default_rng(7).integers(0, 256, (n + extra, H, W), dtype=np.uint8)
    done = np.array([1] * n + [0] * extra, np.uint8)
    pix_p, done_p = os.path.join(tmp, "t.u8"), os.path.join(tmp, "t.done")
    full.tofile(pix_p)
    done.tofile(done_p)
    rows = np.nonzero(np.fromfile(done_p, np.uint8))[0]
    d2 = df.iloc[rows].reset_index(drop=True)
    pix = Picked(FileRows(pix_p, (n + extra, H, W)), rows)
    assert len(d2) == n and not d2.path.str.startswith("u").any()
    assert np.array_equal(pix[[3, 4]], full[[3, 4]])
    o = mammo.parse_sets(["widths=[4,4,4,4,4]", "batch=4", "epochs=1"])
    d = os.path.join(tmp, "v1")
    os.makedirs(d)
    fit(d2, pix, o, torch.device("cpu"), d, json.dumps({"selftest": True}, sort_keys=True))
    for name in ("fold0.pt", "fold0.csv", "fold0.csv.json"):
        assert os.path.exists(os.path.join(d, name)), name
    assert not os.path.exists(os.path.join(d, "fold0.partial.pt"))
    got = pd.read_csv(os.path.join(d, "fold0.csv"), keep_default_na=False)
    assert len(got) == int((d2.fold == 0).sum()) and {"logit", "p"} <= set(got.columns)
    assert 0.0 <= got.p.min() and got.p.max() <= 1.0
    net = mammo.make_net(mammo.reader_module(), [4, 4, 4, 4, 4], 0.0, False)
    copied = mammo.cbis_module().load_init(net, "omama-v1", say=lambda *a: None, path=os.path.join(d, "fold0.pt"))
    assert copied > 0, copied


def check_init():
    cbis = mammo.cbis_module()
    assert "omama-v1" in cbis.INITS and cbis.INITS["omama-v1"].endswith(os.path.join("omama", "v1", "fold0.pt"))
    assert mammo.parse_sets(['init="omama-v1"'])["init"] == "omama-v1"


def check_cam():
    import torch
    torch.manual_seed(0)
    net = mammo.make_net(mammo.reader_module(), [4, 4, 4, 4, 4], 0.0, False).eval()
    x = torch.rand(2, 1, 256, 160)
    with torch.no_grad():
        g = cell_logits(net, x)
        feat = net.trunk(x)
        pooled = net(x)[:, 0]
        assert g.shape[0] == 2 and g.shape[1:] == feat.shape[2:], (g.shape, feat.shape)
        assert torch.allclose(g.mean(dim=(1, 2)), pooled, atol=1e-5), "the cells' mean is not the picture's logit"
        for i, j in ((0, 0), (feat.shape[2] // 2, 1), (feat.shape[2] - 1, feat.shape[3] - 1)):
            assert torch.allclose(g[:, i, j], net.head(feat[:, :, i, j])[:, 0], atol=1e-6), (i, j)
    grid = np.array([[0.1, 0.9], [0.5, 0.7]])
    g = np.zeros((4, 6))
    g[1, 1], g[1, 2], g[2, 2], g[3, 5] = 5.0, 4.0, 3.0, 2.0     # (1, 2) is a neighbour of the largest, (2, 2) a diagonal
    assert top_places(g) == [(1, 1, 5.0), (2, 2, 3.0), (3, 5, 2.0)], top_places(g)


def check_crop_geometry():
    full = np.zeros((900, 700), np.uint8)
    full[396:405, 296:305] = 255                    # a 9 x 9 mark centred on column 300, row 400
    c = native_crop(full, 300, 400, 1.0, False)
    assert c[252:261, 252:261].min() == 255 and c[256, 256] == 255 and int((c > 127).sum()) == 81
    r = native_crop(full, 300, 400, TARGET_MM / 0.1, False)     # spacing 0.1 mm: a 482 px window, resized to LS
    ys, xs = np.nonzero(r > 127)
    assert abs(ys.mean() - 256) < 1 and abs(xs.mean() - 256) < 1, (ys.mean(), xs.mean())
    assert native_crop(full, 300, 400, TARGET_MM / 0.07, False).shape == (LS, LS)
    _, t = mammo.breast_crop(mammo.minmax_u8(fake_breast(3).astype(np.float32)), transform=True)
    for x, y in ((300.0, 400.0), (450.5, 80.0)):
        px, py = mammo.map_points([(x, y)], t)[0]
        bx, by = mammo.unmap_point(px, py, t)
        assert abs(bx - x) < 1e-6 and abs(by - y) < 1e-6, (x, y, bx, by)


def check_copy_map():
    import cv2
    full = np.zeros((900, 600), np.uint8)
    full[496:505, 196:205] = 255                    # a mark centred on column 200, row 500
    small_w = int(round(600 * 1024 / 900))
    body = cv2.resize(full, (small_w, 1024), interpolation=cv2.INTER_AREA)
    for side, ox in (("L", 0), ("R", 1024 - small_w)):
        copy = np.zeros((1024, 1024), np.uint8)
        copy[:, ox:ox + small_w] = body
        ys, xs = np.nonzero(copy > 40)
        x, y = copy_to_original(xs.mean(), ys.mean(), full.shape, 1024, side)
        assert abs(x - 200) < 1.5 and abs(y - 500) < 1.5, (side, x, y)
    x, y = copy_to_original(xs.mean(), ys.mean(), full.shape, 1024, "L")
    assert abs(x - 200) > 100, "a right picture mapped as a left one must miss by the padding"


def check_mil():
    import torch
    z = torch.tensor([0.1, -2.0, 0.5, 3.0, 1.0, -1.0])     # two pictures of three crops each
    assert torch.equal(mammo.mil_max(z, 3), torch.tensor([0.5, 3.0])), mammo.mil_max(z, 3)
    y = torch.tensor([1.0, 0.0])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    assert torch.allclose(loss_fn(mammo.mil_max(z, 3), y), loss_fn(torch.tensor([0.5, 3.0]), y))


def check_share():
    import tempfile
    held = pd.DataFrame({"path": ["a", "b", "c", "d"], "label": [1, 1, 1, 0], "logit": [2.0, -1.0, 0.5, 9.0]})
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "cam.csv")
        pd.DataFrame({"uid": ["a", "b", "c", "d"], "picture_logit": [1.0, 0.0, 0.5, 0.0]}).to_csv(p, index=False)
        share, n = top_crop_share(held, p)
    assert n == 3 and abs(share - 1 / 3) < 1e-9, (share, n)


def check_train_crops(tmp):
    import torch
    n = 50
    uids = ["k%02d" % i for i in range(n)]
    df = pd.DataFrame({"path": uids, "patient": ["P%02d" % i for i in range(n)], "side": "L",
                       "view": "cranio-caudal", "label": [i % 2 for i in range(n)], "fold": [(i // 2) % FOLDS for i in range(n)]})
    arr = np.random.default_rng(3).integers(0, 256, (n * PLACES, LS, LS), dtype=np.uint8)
    p = os.path.join(tmp, "k.u8")
    arr.tofile(p)
    pix = PickedPlaces(FileRows(p, (n * PLACES, LS, LS)), np.arange(n), PLACES)
    assert np.array_equal(pix[np.array([3])], arr[9:12]), "picture 3's crops are not rows 9 to 11"
    sub = PickedPlaces(FileRows(p, (n * PLACES, LS, LS)), np.array([1, 3]), PLACES)
    assert np.array_equal(sub[np.array([1])], arr[9:12]), "the done subset is not indexed by position"
    o = mammo.parse_sets(["found=3", "widths=[4,4,4,4,4]", "batch=6", "epochs=1"])
    d = os.path.join(tmp, "c1")
    os.makedirs(d)
    fit(df, pix, o, torch.device("cpu"), d, json.dumps({"selftest": True}, sort_keys=True))
    for name in ("fold0.pt", "fold0.csv", "fold0.csv.json"):
        assert os.path.exists(os.path.join(d, name)), name
    got = pd.read_csv(os.path.join(d, "fold0.csv"), keep_default_na=False)
    assert len(got) == int((df.fold == 0).sum()) and {"logit", "p"} <= set(got.columns)
    net = mammo.make_net(mammo.reader_module(), [4, 4, 4, 4, 4], 0.0, False)
    copied = mammo.cbis_module().load_init(net, "omama-v1", say=lambda *a: None, path=os.path.join(d, "fold0.pt"))
    assert copied > 0, copied


def selftest():
    import shutil
    import tempfile
    check_select()
    check_init()
    check_cam()
    check_crop_geometry()
    check_copy_map()
    check_mil()
    check_share()
    tmp = tempfile.mkdtemp(prefix="omama-selftest-")
    try:
        check_prepare(tmp)
        check_train(tmp)
        check_train_crops(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("omama selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index")
    ix.add_argument("--rebuild", action="store_true")
    ix.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    ix.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("prepare")
    pl = sub.add_parser("pool")
    pl.add_argument("--limit", type=int)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--epochs", type=int)
    t.add_argument("--set", action="append", default=[], metavar="key=<json>")
    sub.add_parser("cam-places")
    sub.add_parser("crops")
    tc = sub.add_parser("train-crops")
    tc.add_argument("--epochs", type=int)
    tc.add_argument("--set", action="append", default=[], metavar="key=<json>")
    sub.add_parser("check-geometry")
    sub.add_parser("selftest")
    x = ap.parse_args()
    if x.cmd == "index":
        return build_index(x.rebuild, x.neg_ratio, x.seed)
    if x.cmd == "prepare":
        return prepare()
    if x.cmd == "pool":
        return pool(x.limit)
    if x.cmd == "train":
        return train(x.version, x.epochs, x.set)
    if x.cmd == "cam-places":
        return cam_places()
    if x.cmd == "crops":
        return crops()
    if x.cmd == "train-crops":
        return train_crops(x.epochs, x.set)
    if x.cmd == "check-geometry":
        return check_geometry()
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
