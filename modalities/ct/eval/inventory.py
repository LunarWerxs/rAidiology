"""Count what is actually in the masks, once, so everything after it is arithmetic.

Almost every question the split, the baselines and the scorecard ask turns out to be a question
about counts: which structures are in which scans, how big each one is, how many scans never show
it at all. Reading 143,676 mask files to answer each of those separately would be the slow way to
be wrong, so they are read once and the counts are written to `work/eval/ct/inventory.csv`.

What it produces, one row per scan:

- `voxels`, `shape_x/y/z`, `spacing_x/y/z` - the scan's own geometry.
- one column per structure, holding how many voxels that structure occupies in that scan. **Zero is
  the common case and it is information, not a gap**: most CT scans cover part of a body, so a scan
  of a head has no femur in it. A model that is never scored on a structure the scan does not
  contain is being scored honestly; one that is scored against an all-zero mask is being given
  marks for saying nothing.

It also checks that every scan carries the same 117 structure names, because a quietly missing mask
would otherwise read as an absent organ.

    python modalities/ct/eval/inventory.py            [--limit N] [--workers N]
"""
import argparse
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
FILE = os.path.join(OUT, "inventory.csv")


def one(args):
    """Every count for one scan. Runs in its own process: the work is decompressing, not waiting."""
    import nibabel as nib  # noqa: PLC0415  imported in the worker, not the parent

    image_id, names = args
    row = {"image_id": image_id}
    folder = os.path.join(datasets.root(), image_id, datasets.SET["masks"])
    present = sorted(n[:-7] for n in os.listdir(folder) if n.endswith(".nii.gz"))
    if present != names:
        missing = sorted(set(names) - set(present))
        extra = sorted(set(present) - set(names))
        return {"image_id": image_id, "error": "mask set differs: missing %s, extra %s"
                % (missing[:4] or "none", extra[:4] or "none")}
    first = True
    for n in names:
        img = nib.load(os.path.join(folder, "%s.nii.gz" % n))
        if first:
            shape = img.shape
            zooms = img.header.get_zooms()
            row.update({"shape_x": shape[0], "shape_y": shape[1], "shape_z": shape[2],
                        "spacing_x": round(float(zooms[0]), 4),
                        "spacing_y": round(float(zooms[1]), 4),
                        "spacing_z": round(float(zooms[2]), 4),
                        "voxels": int(np.prod(shape))})
            first = False
        row[n] = int(np.count_nonzero(np.asanyarray(img.dataobj)))
    return row


def build(limit=None, workers=None):
    table = datasets.index()
    names = datasets.structures()
    ids = [str(v) for v in table[datasets.SET["key"]]]
    if limit:
        ids = ids[:limit]
    workers = workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    rows, bad = [], []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(one, [(s, names) for s in ids], chunksize=4), 1):
            (bad if "error" in row else rows).append(row)
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                print("  %d of %d, %.1f scans/s, %.0f s left" % (i, len(ids), rate, (len(ids) - i) / rate),
                      flush=True)
    for b in bad:
        print("PROBLEM %s: %s" % (b["image_id"], b["error"]))
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(FILE, index=False)
    return d, names, len(bad), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    with ledger.run("prepare", "ct inventory") as r:
        r.setting("workers", a.workers or "default")
        d, names, bad, seconds = build(a.limit, a.workers)
        counts = d[names]
        per_scan = (counts > 0).sum(axis=1)
        share = (counts > 0).mean(axis=0).sort_values()
        r.number("scans", len(d), "count").number("structures", len(names), "count")
        r.number("unreadable", bad, "count")
        r.number("scans_per_second", len(d) / max(seconds, 1e-9), "count")
        r.number("median_structures_per_scan", float(per_scan.median()), "count")
        r.note_is("%d scan(s), %d structure(s), %d problem(s)" % (len(d), len(names), bad))
    print("\n%d scan(s) x %d structure(s) -> %s" % (len(d), len(names), FILE))
    print("a scan contains %d structures at the median (%d to %d)"
          % (per_scan.median(), per_scan.min(), per_scan.max()))
    print("\nrarest structures, by the share of scans that contain them:")
    for n, v in share.head(8).items():
        print("  %-34s %5.1f%% of scans" % (n, 100 * v))
    print("commonest:")
    for n, v in share.tail(4).items():
        print("  %-34s %5.1f%% of scans" % (n, 100 * v))


if __name__ == "__main__":
    main()
