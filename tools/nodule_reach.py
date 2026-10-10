"""How many nodules a candidate list reaches in each scan's top K by its own score (`probability`), for
any list: a stage 1 version's, or a join with the blobs.

    python tools/nodule_reach.py <candidates.csv> [<candidates.csv> ...]

A nodule is reached when a candidate lies within its radius (froc.py's rule, as tools/nodule_depth.py).
2026-10-04: v3's top 100 reached 209 of fold 0's 240 nodules against v2's 176, at the same CPM.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "modalities", "ct", "eval"), os.path.join(ROOT, "tools"), ROOT]
import froc  # noqa: E402
from nodule_depth import reached  # noqa: E402

DEPTHS = (5, 10, 25, 50, 100, None)


def main():
    ann, _, _ = froc.load_reference()
    for f in sys.argv[1:]:
        c = pd.read_csv(f)
        uids = sorted(set(c.seriesuid))
        a = ann[ann.seriesuid.isin(uids)]
        small = (a.diameter_mm <= 5).to_numpy()
        c["rank"] = c.groupby("seriesuid").probability.rank(ascending=False, method="first")
        out = []
        for k in DEPTHS:
            hit = reached(c if k is None else c[c["rank"] <= k], a)
            out.append("top %s: %d (%d of %d small)" % (k or "all", hit.sum(), hit[small].sum(), small.sum()))
        print("%s: %d scans, %d nodules, %.0f a scan\n  %s" % (f, len(uids), len(a), len(c) / len(uids), "; ".join(out)))


if __name__ == "__main__":
    main()
