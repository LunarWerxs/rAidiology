"""A structure predicted smaller than any real one is not there: drop it, and see what that does.

    python modalities/ct/eval/min_size.py --version 2 [--q 0 0.01 0.05] [--tag raw]

Version 2 outlines a structure the scan does not contain more often than version 1 did (15.1% of
absent pairs against 14.1%, on all 1,139 development scans), most likely because its crops now seek
out rare structures. An invented structure is usually a small island (the median is 42 voxels), and a
real one seldom is. So: for each fold, the q-th percentile of every structure's true size on the scans
of the other folds is its smallest believable size, and a prediction below it on fold k is dropped,
its voxels left unlabelled. The threshold never sees the fold it is applied to.

It is computed on the counts the scorer already has (a whole structure dropped is its predicted and
shared voxels set to nothing), so no scan is read again. Each q writes out-of-fold files beside the
model's (`oof_fold{K}_min{q}.csv`) for compare.py, against the same model read without the filter.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import segmenter  # noqa: E402  arkitect-allow: test-floor-loosened - a sibling module after the sys.path line, as in every eval script here

FOLDS = 5


def smallest_believable(rows, q):
    """Per structure, the q-th percentile of its true size where it is present."""
    here = rows[(rows["in_scope"] == 1) & (rows["truth_voxels"] > 0)]
    return here.groupby("structure")["truth_voxels"].quantile(q)


def filtered(rows, floor):
    out = rows.copy()
    small = (out["pred_voxels"] > 0) & (out["pred_voxels"] < out["structure"].map(floor).fillna(0))
    out.loc[small, ["pred_voxels", "both_voxels"]] = 0
    return out, int(small.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--q", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    ap.add_argument("--tag", default="", help="the out-of-fold files' tag, if not the model's own reading")
    a = ap.parse_args()
    d = segmenter.work_dir(a.version)
    tag = "_" + a.tag if a.tag else ""
    folds = {k: pd.read_csv(os.path.join(d, "oof_fold%d%s.csv" % (k, tag)), dtype={"scan": str}) for k in range(FOLDS)}
    for q in a.q:
        dropped = 0
        for k, rows in folds.items():
            others = pd.concat([folds[j] for j in folds if j != k], ignore_index=True)
            got, n = filtered(rows, smallest_believable(others, q))
            dropped += n
            got.to_csv(os.path.join(d, "oof_fold%d%s_min%g.csv" % (k, tag, q)), index=False)
        absent = sum(int(((r["in_scope"] == 1) & (r["truth_voxels"] == 0) & (r["pred_voxels"] > 0)).sum())
                     for r in folds.values())
        print("q %g: %d predicted structures dropped over five folds (%d were inventions before)"
              % (q, dropped, absent), flush=True)


if __name__ == "__main__":
    main()
