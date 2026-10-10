"""Two lung nodule candidate lists on the same scans, compared the way the other readers are: the
difference in CPM with a 95% interval from resampling scans, the same draw for both.

    python tools/froc_compare.py <A candidates.csv> <B candidates.csv> [--name "<what B tries>"]
        [--boot 1000] [--column probability] [--a-column ...] [--b-column ...] [--margin 0.01]

The scans are every scan either file has a candidate on: a scan with none in one file is still
scored there (its nodules are misses), so both are judged on the same scans. The verdict is
compare.py's: BETTER when the interval's low end is above 0, NOT WORSE when above minus the margin,
WORSE when the high end is below 0, else INCONCLUSIVE. The result goes into the ledger as an
experiment unless --dry.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "modalities", "ct", "eval"), ROOT]
import froc  # noqa: E402
from core import ledger  # noqa: E402


def curve_parts(cand, ann, exc, uids):
    m = froc._match(cand, ann, exc, uids)
    prob, gt, scan, last = froc._curve_inputs(m)
    return m, gt, scan, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--name", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--column", default="probability")
    ap.add_argument("--a-column")
    ap.add_argument("--b-column")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    x = ap.parse_args()

    ann, exc, _ = froc.load_reference()
    frames = []
    for path, col in ((x.a, x.a_column or x.column), (x.b, x.b_column or x.column)):
        f = pd.read_csv(path)
        frames.append(f[["seriesuid", "coordX", "coordY", "coordZ"]].assign(probability=f[col].astype(float)))
    a_uids, b_uids = set(frames[0].seriesuid), set(frames[1].seriesuid)
    uids = sorted(a_uids | b_uids)
    if a_uids ^ b_uids:
        print("%d scans have candidates in only one file; both are scored on all %d" % (len(a_uids ^ b_uids), len(uids)))
    parts = [curve_parts(f, ann, exc, uids) for f in frames]
    (ma, ga, sa_, la), (mb, gb, sb, lb) = parts
    assert list(ma["uids"]) == list(mb["uids"]), "the two matchings ordered the scans differently"
    n, lesions = len(ma["uids"]), ma["lesions"].astype(np.float64)

    def cpm(g, s, last, w):
        f, sens = froc._curve(g, w[s], last, float(w @ lesions), n)
        return float(froc._sens_at(f, sens).mean())

    one = np.ones(n)
    ca, cb = cpm(ga, sa_, la, one), cpm(gb, sb, lb, one)
    rng = np.random.default_rng(x.seed)
    diffs = np.empty(x.boot)
    for i in range(x.boot):
        w = np.bincount(rng.integers(n, size=n), minlength=n).astype(np.float64)
        diffs[i] = cpm(gb, sb, lb, w) - cpm(ga, sa_, la, w)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = ("BETTER" if lo > 0 else "NOT WORSE" if lo > -x.margin else "WORSE" if hi < 0 else "INCONCLUSIVE")
    print("%d scans, %d nodules; %d paired resamples of scans" % (n, int(lesions.sum()), x.boot))
    print("CPM  A %.4f  B %.4f  B - A %+.4f (%+.4f to %+.4f)" % (ca, cb, cb - ca, lo, hi))
    print("Verdict (CPM, margin %.3f): %s" % (x.margin, verdict))
    if not x.dry:
        name = x.name or "%s vs %s" % (os.path.basename(os.path.dirname(x.a)), os.path.basename(os.path.dirname(x.b)))
        with ledger.run("experiment", name) as led:
            led.settings_from({"a": x.a, "b": x.b, "boot": x.boot, "margin": x.margin})
            led.number("cpm_a", ca, "cpm").number("cpm_b", cb, "cpm").number("cpm_diff", cb - ca, "cpm")
            led.number("cpm_diff_low", float(lo), "cpm").number("cpm_diff_high", float(hi), "cpm")


if __name__ == "__main__":
    main()
