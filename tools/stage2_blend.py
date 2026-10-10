"""The mean of several stage 2 nets' scores on one candidate list.

    python tools/stage2_blend.py A.csv B.csv OUT.csv [--more C.csv D.csv ...] [--column p2]

A, B (and each --more file) are `nodule_fpr.py` outputs scored on the same list (seriesuid, coordX,
coordY, coordZ, p1, p2, fold). OUT keeps A's rows with `p2` replaced by the mean of every file's.
Every place must be in every file, or nothing is written: a place one net never scored would be read
as a miss. 2026-10-04: more than two files, for the five-net blend (docs/efficiency, "every stage 2
net together").
"""
import argparse
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import ledger  # noqa: E402

KEY = ["seriesuid", "coordX", "coordY", "coordZ"]


def blend(frames, column="p2", stats=None):
    """frames: two or more data frames scored on the same places. Returns the first with `column` the mean.
    A dict given as `stats` is filled with the rank agreement it prints (average, lowest, highest)."""
    frames = [f.copy() for f in frames]
    for df in frames:
        for c in KEY[1:]:
            df[c] = df[c].round(3)
        if df.duplicated(KEY).any():
            raise SystemExit("a place is listed twice in one file")
    first = frames[0]
    m = first
    cols = []
    for n, df in enumerate(frames):
        name = "%s_%d" % (column, n)
        cols.append(name)
        m = m.merge(df[KEY + [column]].rename(columns={column: name}), on=KEY, how="outer", indicator=True)
        only = m["_merge"].value_counts()
        if only.get("left_only", 0) or only.get("right_only", 0):
            raise SystemExit("the lists differ: file %d has %d places the first lacks and lacks %d of its places"
                             % (n + 1, only.get("right_only", 0), only.get("left_only", 0)))
        m = m.drop(columns="_merge")
    m[column] = m[cols].mean(axis=1)
    rank = m[cols].corr(method="spearman")
    pairs = [rank.iloc[i, j] for i, j in itertools.combinations(range(len(cols)), 2)]
    print("blended %d places in %d scans from %d nets; rank agreement %.3f on average (%.3f to %.3f)"
          % (len(m), m.seriesuid.nunique(), len(cols), sum(pairs) / len(pairs), min(pairs), max(pairs)))
    if stats is not None:
        stats.update(rank_mean=sum(pairs) / len(pairs), rank_min=min(pairs), rank_max=max(pairs))
    return m[list(first.columns)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("out")
    p.add_argument("--more", nargs="+", default=[], help="further files scored on the same list")
    p.add_argument("--column", default="p2")
    args = p.parse_args(argv)
    files = [args.a, args.b] + args.more
    with ledger.run("experiment", "stage 2 blend of %d nets into %s" % (len(files), os.path.basename(args.out))) as led:
        led.settings_from({"files": files, "out": args.out, "column": args.column})
        stats = {}
        out = blend([pd.read_csv(f) for f in files], args.column, stats)
        out.to_csv(args.out, index=False)
        print("wrote", args.out)
        led.number("places", len(out), "count").number("scans", out.seriesuid.nunique(), "count")
        led.number("nets", len(files), "count")
        led.number("rank_agreement_mean", stats.get("rank_mean")).number("rank_agreement_min", stats.get("rank_min"))
        led.number("rank_agreement_max", stats.get("rank_max"))


if __name__ == "__main__":
    sys.exit(main())
