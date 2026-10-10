"""The baselines a trained segmenter has to beat, scored before it exists.

Declared in METRICS_SPEC.md before anything was trained. The point of them is not that any is good;
it is that "the segmenter got 0.7" means nothing until somebody has said what 0.7 is better than,
and a reference that is too weak makes the ship rule a formality.

- **always-empty** predicts nothing anywhere. Dice 0, invention rate 0. It is the floor that proves
  the metric is not quietly rewarding silence, which on a dataset where most structures are absent
  from most scans is a real risk.
- **where-it-usually-is** is a shape prior with no picture in it: for each structure, the cells of a
  normalised box where it turns up most often across development scans, as many cells as it usually
  occupies. It knows anatomy is not placed at random and nothing else.
- **nearest-scan** is classical single-atlas segmentation, and it is the one that makes this an
  honest contest: find the development scan whose field of view and brightness profile most
  resemble this one, and copy its expert masks across. No model, no training, decades old, and it
  is a genuinely hard thing for a mediocre network to beat.
- **bone-window** takes the better of those two regions and keeps only the voxels whose Hounsfield
  brightness fits what the structure is made of. CT brightness means the same thing on every
  scanner, so that is physics rather than a guess, and it is the reason a threshold rule is real
  engineering here where it would be a straw man on an X-ray.

Everything is **out of fold**: a development scan in fold 2 is predicted from folds 0, 1, 3 and 4,
never from itself. A prior or an atlas that has seen the scan it predicts is a model that has seen
the answer, however simple it looks.

    python modalities/ct/eval/baselines.py boxes            one pass over every scan
    python modalities/ct/eval/baselines.py predict dev      writes the prediction rows
    python modalities/ct/eval/baselines.py check            proves the fast counting form is exact
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import datasets  # noqa: E402
from core import ledger, paths  # noqa: E402

OUT = os.path.join(paths.WORK, "eval", "ct")
BOXES = os.path.join(OUT, "boxes")
SIGNATURES = os.path.join(OUT, "signatures.csv")
BOX = 64                      # the normalised cube every scan is squashed into
SIG = 8                       # the brightness signature is this cube of mean Hounsfield units
METHODS = ("always-empty", "where-it-usually-is", "nearest-scan", "bone-window")
ALL_FOLDS = "all"

# Hounsfield windows, from the physics rather than from the data: these are the standard ranges for
# what each kind of tissue is made of, they are the same on every scanner, and that is the whole
# reason a rule like this is worth having on CT and worthless on an X-ray.
BONE_HU = (200, 3000)
SOFT_HU = (-50, 250)
AIR_HU = (-1100, -400)
BONE_WORDS = ("vertebrae", "rib", "femur", "humerus", "hip", "sacrum", "skull", "clavicula",
              "scapula", "sternum", "costal_cartilages", "patella", "tibia", "fibula", "tarsal",
              "metatarsal", "phalanges", "carpal", "metacarpal", "radius", "ulna")
AIR_WORDS = ("lung", "trachea")


def window_for(structure):
    if any(w in structure for w in BONE_WORDS):
        return BONE_HU
    if any(w in structure for w in AIR_WORDS):
        return AIR_HU
    return SOFT_HU


def axis_index(n_out, n_in):
    """`n_out` nearest-neighbour indices into an axis of length `n_in`.

    Both directions of the resampling are this one function, which is why the argument order is
    worth saying out loud: the first number is how many indices come back, the second is the axis
    they point into. Getting it the wrong way round returns the array untouched at full size, which
    is what happened on the first run of this file."""
    return np.minimum(((np.arange(n_out) + 0.5) * n_in / n_out).astype(np.int64), n_in - 1)


def to_box(mask, box=BOX):
    """Squash a volume into the normalised cube by nearest neighbour.

    Nearest neighbour and not an interpolation: what is being carried is which voxels are inside a
    structure, and an interpolated 0.4 is not half inside anything."""
    return mask[np.ix_(*[axis_index(box, s) for s in mask.shape])]


def from_box(box_volume, shape):
    """And back out to a scan's own grid, the same way."""
    return box_volume[np.ix_(*[axis_index(s, BOX) for s in shape])]


def box_file(scan):
    return os.path.join(BOXES, "%s.npy" % scan)


def _one_box(args):
    """One scan: its 117 masks squashed into the cube and packed, plus a brightness signature.

    Written to disk rather than returned, because 117 structures at 64 cubed is 122 MB as booleans
    and this crosses a process boundary once per scan. What comes back is the signature, which is a
    few hundred numbers."""
    import nibabel as nib  # noqa: PLC0415

    scan, names = args
    folder = os.path.join(datasets.root(), scan, datasets.SET["masks"])
    out = np.zeros((len(names), BOX, BOX, BOX), dtype=bool)
    sizes = np.zeros(len(names), dtype=np.int64)
    for i, n in enumerate(names):
        m = np.asanyarray(nib.load(os.path.join(folder, "%s.nii.gz" % n)).dataobj)
        if m.any():
            out[i] = to_box(m > 0)
            sizes[i] = int(out[i].sum())
    os.makedirs(BOXES, exist_ok=True)
    np.save(box_file(scan), np.packbits(out))
    ct = np.asanyarray(nib.load(datasets.scan_path(scan)).dataobj).astype(np.float32)
    # Mean brightness over a coarse cube: enough to tell a chest from a pelvis and a contrast study
    # from a plain one, and small enough that comparing every scan with every other is instant.
    ix, iy, iz = (axis_index(s, SIG) for s in ct.shape)
    flat = ((ix[:, None, None] * SIG + iy[None, :, None]) * SIG + iz[None, None, :]).ravel()
    total = np.bincount(flat, weights=ct.ravel(), minlength=SIG ** 3)
    count = np.bincount(flat, minlength=SIG ** 3)
    sig = total / np.maximum(count, 1)
    row = {"scan": scan, "shape_x": ct.shape[0], "shape_y": ct.shape[1], "shape_z": ct.shape[2]}
    row.update({"sig%d" % i: float(v) for i, v in enumerate(sig)})
    row.update({"size%d" % i: int(v) for i, v in enumerate(sizes)})
    return row


def build_boxes(workers=None, limit=None):
    table = datasets.index()
    names = datasets.structures()
    scans = [str(v) for v in table[datasets.SET["key"]]]
    if limit:
        scans = scans[:limit]
    workers = workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(_one_box, [(s, names) for s in scans], chunksize=2), 1):
            rows.append(row)
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                print("  %d of %d, %.1f scans/s, %.0f s left"
                      % (i, len(scans), rate, (len(scans) - i) / rate), flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(SIGNATURES, index=False)
    return len(rows), time.time() - t0


def load_box(scan, n_structures):
    packed = np.load(box_file(scan))
    shape = (n_structures, BOX, BOX, BOX)
    return np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)


PRIORS = os.path.join(OUT, "priors.npy")


def load_priors(split, names, rebuild=False):
    """The priors, built once and kept on disk.

    Building them reads every development scan's box, which is four gigabytes; doing that again for
    each caller, and then pickling 180 MB of booleans into every worker process, is the kind of
    waste that only shows up as a run that feels slow for no reason. On disk they are memory-mapped
    instead, so eight workers share one copy."""
    if rebuild or not os.path.exists(PRIORS):
        built = build_priors(split, names)
        order = [str(f) for f in range(split["folds"])] + [ALL_FOLDS]
        stack = np.stack([np.packbits(built[k].reshape(-1)) for k in order])
        os.makedirs(OUT, exist_ok=True)
        np.save(PRIORS, stack)
    stack = np.load(PRIORS, mmap_mode="r")
    order = [str(f) for f in range(split["folds"])] + [ALL_FOLDS]
    n = len(names) * BOX ** 3
    return {k: np.unpackbits(np.asarray(stack[i]), count=n).reshape(len(names), -1).astype(bool)
            for i, k in enumerate(order)}


def build_priors(split, names):
    """One prior per fold, each built from every other fold, plus one from all of development.

    The threshold is the part worth explaining. "The cells where it appears in more than half the
    scans" sounds right and produces nothing at all: every scan covers a different part of a body,
    so after squashing each one into the same cube, no cell holds the same organ in half of them.
    Measured 2026-10-01: that rule left all 117 structures empty.

    So instead each structure gets the cells it turns up in most often, as many cells as it usually
    occupies - its median size across the scans that contain it. A region of about the right size
    in the most likely place, which is what the name of the baseline always claimed."""
    folds = split["dev_folds"]
    n_folds = split["folds"]
    sums = np.zeros((n_folds, len(names), BOX ** 3), dtype=np.uint16)
    counts = np.zeros(n_folds, dtype=np.int64)
    sizes = {f: [[] for _ in names] for f in range(n_folds)}
    for scan, f in sorted(folds.items()):
        box = load_box(scan, len(names)).reshape(len(names), -1)
        sums[f] += box
        counts[f] += 1
        for i in np.flatnonzero(box.any(axis=1)):
            sizes[f][i].append(int(box[i].sum()))

    out = {}
    for fold in list(range(n_folds)) + [ALL_FOLDS]:
        keep = [f for f in range(n_folds) if f != fold]
        total = sums[keep].sum(axis=0)
        prior = np.zeros((len(names), BOX ** 3), dtype=bool)
        for i in range(len(names)):
            seen = [v for f in keep for v in sizes[f][i]]
            if not seen:
                continue
            k = int(np.median(seen))
            if k <= 0:
                continue
            row = total[i]
            k = min(k, int((row > 0).sum()))
            if k <= 0:
                continue
            cut = np.partition(row, -k)[-k]
            chosen = np.flatnonzero(row >= max(cut, 1))
            prior[i, chosen[:k] if len(chosen) > k else chosen] = True
        out[str(fold)] = prior.reshape((len(names), BOX, BOX, BOX))
    return out


def build_atlas_choice(split, names):
    """For every scan, the development scan from another fold that looks most like it.

    Classical single-atlas segmentation: the nearest neighbour by brightness profile and by the
    shape of the field of view, which between them say roughly what part of a body is in the
    picture. Out of fold, so nothing is ever its own atlas."""
    sig = pd.read_csv(SIGNATURES)
    cols = [c for c in sig.columns if c.startswith("sig")] + ["shape_x", "shape_y", "shape_z"]
    sig = sig.set_index("scan")
    x = sig[cols].to_numpy(dtype=np.float64)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    at = {s: i for i, s in enumerate(sig.index.astype(str))}
    folds = split["dev_folds"]
    dev = sorted(folds)
    choice = {}
    for scan in dev + list(split["test"]):
        if scan not in at:
            continue
        mine = folds.get(scan)
        pool = [s for s in dev if s != scan and (mine is None or folds[s] != mine)]
        idx = np.array([at[s] for s in pool])
        d = ((x[idx] - x[at[scan]]) ** 2).sum(axis=1)
        choice[scan] = pool[int(np.argmin(d))]
    return choice


def cell_index(shape):
    """For every voxel of a scan, which cell of the normalised box it falls in, flattened.

    The whole prediction pass hangs off this. Written out naively, a baseline means building a
    full-size boolean volume for each of 117 structures and intersecting it with the scan three
    times over: about six thousand million operations per scan, which turns a ten-minute pass into
    an afternoon. With the cell each voxel belongs to known once, every count below becomes a sum
    over 262,144 cells or over the structure's own voxels, which are a tiny fraction of the scan.
    The answers are identical, not approximate; `check` proves it against the naive form."""
    ix, iy, iz = (axis_index(s, BOX) for s in shape)
    return ((ix[:, None, None] * BOX + iy[None, :, None]) * BOX + iz[None, None, :]).ravel()


def _counts_for_scan(scan, names, regions, ct):
    """Every baseline's three counts for one scan, from cell sums rather than volume passes.

    `regions` maps a method name to a (structures x BOX^3) boolean array of box cells it predicts."""
    import nibabel as nib  # noqa: PLC0415

    folder = os.path.join(datasets.root(), scan, datasets.SET["masks"])
    cells = cell_index(ct.shape)
    flat = ct.ravel()
    cells_per = np.bincount(cells, minlength=BOX ** 3)
    in_window, window_cells = {}, {}
    for w in (BONE_HU, SOFT_HU, AIR_HU):
        mask = (flat >= w[0]) & (flat <= w[1])
        in_window[w] = mask
        window_cells[w] = np.bincount(cells[mask], minlength=BOX ** 3)

    rows = []
    for i, n in enumerate(names):
        truth = np.asanyarray(nib.load(os.path.join(folder, "%s.nii.gz" % n)).dataobj)
        if truth.shape != ct.shape:
            rows += [{"method": m, "scan": scan, "structure": n, "in_scope": 0,
                      "drop_reason": "geometry_mismatch", "truth_voxels": 0, "pred_voxels": 0,
                      "both_voxels": 0} for m in METHODS]
            continue
        at = np.flatnonzero(truth.ravel())
        here = cells[at]
        w = window_for(n)
        counts = {"always-empty": (0, 0)}
        for m in ("where-it-usually-is", "nearest-scan"):
            where = regions[m][i]
            counts[m] = (int(cells_per[where].sum()), int(where[here].sum()))
        best = max(("where-it-usually-is", "nearest-scan"), key=lambda m: counts[m][1])
        where = regions[best][i]
        counts["bone-window"] = (int(window_cells[w][where].sum()),
                                 int((where[here] & in_window[w][at]).sum()))
        for m in METHODS:
            p, b = counts[m]
            rows.append({"method": m, "scan": scan, "structure": n, "in_scope": 1,
                         "drop_reason": "", "truth_voxels": int(at.size), "pred_voxels": p,
                         "both_voxels": b})
    return rows


_PRIORS = {}


def _regions_for(scan, fold, names, atlas):
    prior = _PRIORS[str(fold)]
    other = load_box(atlas[scan], len(names)).reshape(len(names), -1)
    return {"where-it-usually-is": prior, "nearest-scan": other}


def _predict_one(args):
    import nibabel as nib  # noqa: PLC0415

    scan, names, fold, atlas = args
    try:
        ct = np.asanyarray(nib.load(datasets.scan_path(scan)).dataobj)
    except Exception:  # noqa: BLE001  a scan that will not open is a row, not a crash
        return [{"method": m, "scan": scan, "structure": n, "in_scope": 0,
                 "drop_reason": "unreadable", "truth_voxels": 0, "pred_voxels": 0, "both_voxels": 0}
                for n in names for m in METHODS]
    return _counts_for_scan(scan, names, _regions_for(scan, fold, names, atlas), ct)


def _init_worker(split, names):
    _PRIORS.update(load_priors(split, names))


def naive_counts(scan, names, regions, ct):
    """The same counts the obvious way, used only to prove the fast form is not approximate."""
    import nibabel as nib  # noqa: PLC0415

    folder = os.path.join(datasets.root(), scan, datasets.SET["masks"])
    rows = []
    for i, n in enumerate(names):
        truth = np.asanyarray(nib.load(os.path.join(folder, "%s.nii.gz" % n)).dataobj) > 0
        got = {"always-empty": (0, 0)}
        full = {}
        for m in ("where-it-usually-is", "nearest-scan"):
            full[m] = from_box(regions[m][i].reshape(BOX, BOX, BOX), ct.shape)
            got[m] = (int(full[m].sum()), int(np.count_nonzero(full[m] & truth)))
        best = max(("where-it-usually-is", "nearest-scan"), key=lambda m: got[m][1])
        lo, hi = window_for(n)
        windowed = full[best] & (ct >= lo) & (ct <= hi)
        got["bone-window"] = (int(windowed.sum()), int(np.count_nonzero(windowed & truth)))
        for m in METHODS:
            rows.append({"method": m, "scan": scan, "structure": n, "truth_voxels": int(truth.sum()),
                         "pred_voxels": got[m][0], "both_voxels": got[m][1]})
    return rows


def check(n_scans=2):
    """Run both forms over a couple of scans and refuse to pass unless every count matches."""
    import nibabel as nib  # noqa: PLC0415

    split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    names = datasets.structures()
    _PRIORS.update(load_priors(split, names))
    atlas = build_atlas_choice(split, names)
    bad = 0
    for scan in sorted(split["dev_folds"])[:n_scans]:
        fold = split["dev_folds"][scan]
        ct = np.asanyarray(nib.load(datasets.scan_path(scan)).dataobj)
        regions = _regions_for(scan, fold, names, atlas)
        t0 = time.time()
        fast = _counts_for_scan(scan, names, regions, ct)
        fast_s = time.time() - t0
        t0 = time.time()
        slow = naive_counts(scan, names, regions, ct)
        slow_s = time.time() - t0
        for a, b in zip(fast, slow):
            for k in ("truth_voxels", "pred_voxels", "both_voxels"):
                if a[k] != b[k]:
                    bad += 1
                    print("MISMATCH %s %s %s %s: %d against %d"
                          % (scan, a["structure"], a["method"], k, a[k], b[k]))
        print("%s: %d count(s) agree, fast %.2f s against naive %.2f s, %.1fx"
              % (scan, 3 * len(fast), fast_s, slow_s, slow_s / max(fast_s, 1e-9)))
    if bad:
        raise SystemExit("%d count(s) disagree: the fast form is wrong, do not use it." % bad)
    print("the two forms agree on every count.")


def predict(split_name, workers=None, limit=None):
    datasets.guard_test(split_name)
    if not os.path.isdir(BOXES):
        raise SystemExit("No boxes yet. Run: python modalities/ct/eval/baselines.py boxes")
    split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    names = datasets.structures()
    print("building the out-of-fold priors and choosing an atlas for each scan...", flush=True)
    priors = load_priors(split, names)
    atlas = build_atlas_choice(split, names)
    empty = int(sum(1 for i in range(len(names)) if not priors[ALL_FOLDS][i].any()))
    print("  %d of %d structures have an empty prior" % (empty, len(names)), flush=True)

    work = ([(s, None) for s in split["test"]] if split_name == "test"
            else sorted(split["dev_folds"].items()))
    if limit:
        work = work[:limit]
    folds = dict(work)
    workers = workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    rows = []
    t0 = time.time()
    jobs = [(s, names, ALL_FOLDS if f is None else f, atlas) for s, f in work]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(split, names)) as pool:
        for i, got in enumerate(pool.map(_predict_one, jobs, chunksize=1), 1):
            rows += got
            if i % 50 == 0:
                rate = i / (time.time() - t0)
                print("  %d of %d, %.1f scans/s, %.0f s left"
                      % (i, len(work), rate, (len(work) - i) / rate), flush=True)
    d = pd.DataFrame(rows)
    d["split"] = split_name
    d["fold"] = ["" if folds.get(s) is None else folds[s] for s in d["scan"]]
    d = d[["method", "split", "fold", "scan", "structure", "in_scope", "drop_reason",
           "truth_voxels", "pred_voxels", "both_voxels"]]
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "pred_baselines_%s.csv" % split_name)
    d.to_csv(f, index=False)
    return d, f, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("boxes"); b.add_argument("--workers", type=int); b.add_argument("--limit", type=int)
    q = sub.add_parser("predict"); q.add_argument("split", choices=["dev", "test"])
    q.add_argument("--workers", type=int); q.add_argument("--limit", type=int)
    c = sub.add_parser("check", help="prove the fast counting form equals the obvious one")
    c.add_argument("--scans", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "check":
        return check(a.scans)
    if a.cmd == "boxes":
        with ledger.run("baselines", "ct boxes") as r:
            r.settings_from({"workers": a.workers or "default", "box": BOX})
            n, seconds = build_boxes(a.workers, a.limit)
            r.number("scans", n, "count").number("box", BOX, "count")
            r.number("scans_per_second", n / max(seconds, 1e-9), "count")
        print("%d scan(s) squashed into the box -> %s" % (n, BOXES))
    else:
        with ledger.run("baselines", "ct %s" % a.split) as r:
            r.settings_from({"workers": a.workers or "default", "methods": ",".join(METHODS)})
            d, f, seconds = predict(a.split, a.workers, a.limit)
            r.number("rows", len(d), "count").number("scans", d["scan"].nunique(), "count")
            r.number("seconds_per_scan", seconds / max(d["scan"].nunique(), 1), "s")
        print("%d row(s) over %d scan(s) -> %s" % (len(d), d["scan"].nunique(), f))
        here = d[(d["in_scope"] == 1) & (d["truth_voxels"] > 0)]
        for m in METHODS:
            g = here[here["method"] == m]
            dice = 2 * g["both_voxels"] / (g["truth_voxels"] + g["pred_voxels"]).clip(lower=1)
            print("  %-22s mean Dice over present structures %.4f" % (m, dice.mean()))


if __name__ == "__main__":
    main()
