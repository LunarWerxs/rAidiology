"""How deep stage 2 should read stage 1's list: for each depth K, each scan's top K candidates by stage
1's score (p1), how many nodules they reach and the CPM of each score on them.

    python tools/nodule_depth.py <stage 2 candidates.csv (p1, p2)> [--depths 100 200 300 500 1000]
        [--write <folder>]

A nodule is reached when a candidate lies within its radius (froc.py's rule). With --write, each
depth's list goes to <folder>/depth<K>.csv (p1 and p2 kept) for tools/froc_compare.py.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "modalities", "ct", "eval"), ROOT]
import froc  # noqa: E402

COLS = ["seriesuid", "coordX", "coordY", "coordZ"]


def reached(cand, ann):
    """Per nodule of the scans in `cand`, True when any candidate lies within its radius."""
    by_scan = {u: g[["coordX", "coordY", "coordZ"]].to_numpy(np.float64) for u, g in cand.groupby("seriesuid")}
    out = []
    for r in ann.itertuples():
        pts = by_scan.get(r.seriesuid)
        if pts is None:
            out.append(False)
            continue
        d2 = ((pts - (r.coordX, r.coordY, r.coordZ)) ** 2).sum(1)
        out.append(bool((d2 <= (r.diameter_mm / 2.0) ** 2).any()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--depths", type=int, nargs="+", default=[100, 200, 300, 500, 1000])
    ap.add_argument("--write")
    x = ap.parse_args()

    ann, exc, _ = froc.load_reference()
    c = pd.read_csv(x.candidates)
    uids = sorted(set(c.seriesuid))
    a = ann[ann.seriesuid.isin(uids)]
    c["rank"] = c.groupby("seriesuid").p1.rank(ascending=False, method="first")
    small = (a.diameter_mm <= 5).to_numpy()
    print("%d scans, %d nodules (%d of them 5 mm or less), up to %.0f candidates a scan"
          % (len(uids), len(a), small.sum(), c.groupby("seriesuid").size().max()))
    print("depth  a scan  reached (of 100)  5 mm or less  CPM p1   CPM p2   CPM p1*p2")
    if x.write:
        os.makedirs(x.write, exist_ok=True)
    for k in x.depths:
        top = c[c["rank"] <= k]
        hit = reached(top, a)
        cpm = {}
        for key, p in (("p1", top.p1), ("p2", top.p2), ("p12", top.p1 * top.p2)):
            cpm[key] = froc.score(top[COLS].assign(probability=p.astype(float)), ann, exc, uids)["cpm"]
        print("%5d  %6.1f  %4d (%5.1f)      %3d of %3d    %.4f   %.4f   %.4f"
              % (k, len(top) / len(uids), hit.sum(), 100.0 * hit.mean(), hit[small].sum(), small.sum(),
                 cpm["p1"], cpm["p2"], cpm["p12"]))
        if x.write:
            top.drop(columns="rank").to_csv(os.path.join(x.write, "depth%d.csv" % k), index=False)


if __name__ == "__main__":
    main()
