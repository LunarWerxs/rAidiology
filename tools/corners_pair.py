"""Two corner versions on the same held-out patients: per-patient mean miss, B minus A, with a paired patient
bootstrap (corners.py's BOOT resamples and seed). The step-1 bar of idea 24 (docs/efficiency, "Corners v9, idea 24")
reads this: B goes on when it is at least --need px lower and the top of the interval is below 0.

Usage: python tools/corners_pair.py A B [--view LA] [--folds 0] [--need 0.5] [--name "..."]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modalities", "spine_xray", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import corners  # noqa: E402
from core import ledger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=int)
    ap.add_argument("b", type=int)
    ap.add_argument("--view", default="LA", choices=["AP", "LA"])
    ap.add_argument("--folds", default="0")
    ap.add_argument("--need", type=float, default=0.5, help="px B must be lower by")
    ap.add_argument("--name", default=None)
    o = ap.parse_args()
    name = o.name or "corners v%d against v%d, %s view, folds %s" % (o.b, o.a, o.view, o.folds)
    with ledger.run("experiment", name) as led:
        led.settings_from({"a": o.a, "b": o.b, "view": o.view, "folds": o.folds, "need": o.need})
        compare(o, name, led)


def compare(o, name, led):
    folds = [int(k) for k in o.folds.split(",")]
    ra, rb = corners.load_oof(o.view, o.a, folds), corners.load_oof(o.view, o.b, folds)
    common = np.intersect1d(ra["ids"], rb["ids"])
    if len(common) < len(ra["ids"]) or len(common) < len(rb["ids"]):
        print("only %d patients in both (A %d, B %d)" % (len(common), len(ra["ids"]), len(rb["ids"])))
    ia = {p: i for i, p in enumerate(ra["ids"])}
    ib = {p: i for i, p in enumerate(rb["ids"])}
    sa, sb = [ia[p] for p in common], [ib[p] for p in common]
    if not np.allclose(ra["truth"][sa], rb["truth"][sb], equal_nan=True):
        raise SystemExit("the drawn corners differ between A and B: not the same answer key")
    ma = corners.miss_per_patient(ra["pred"][sa], ra["truth"][sa])
    mb = corners.miss_per_patient(rb["pred"][sb], rb["truth"][sb])
    vals = np.stack([ma, mb, mb - ma], axis=1)
    bm = corners.boot_means(vals)
    lo, hi = corners.ci(bm)
    mean = vals.mean(0)
    print("%s: %d patients, mean corner miss at full resolution (paired patient bootstrap, %d resamples, seed %d)"
          % (name, len(common), corners.BOOT, corners.BOOT_SEED))
    print("  A v%-4d %7.2f px (%.2f to %.2f); median %.2f" % (o.a, mean[0], lo[0], hi[0], float(np.median(ma))))
    print("  B v%-4d %7.2f px (%.2f to %.2f); median %.2f" % (o.b, mean[1], lo[1], hi[1], float(np.median(mb))))
    print("  B minus A %+.2f px (95%% CI %+.2f to %+.2f); B better on %d of %d patients"
          % (mean[2], lo[2], hi[2], int((mb < ma).sum()), len(common)))
    on = bool(mean[2] <= -o.need and hi[2] < 0)
    print("Verdict (at least %.2f px lower, interval's top below 0): %s" % (o.need, "GOES ON" if on else "A FINDING"))
    led.number("miss_a", float(mean[0]), "px").number("miss_b", float(mean[1]), "px")
    led.number("b_minus_a", float(mean[2]), "px").number("b_minus_a_low", float(lo[2]), "px")
    led.number("b_minus_a_high", float(hi[2]), "px").number("patients", int(len(common)), "count")
    led.number("goes_on", 1 if on else 0, "flag")


if __name__ == "__main__":
    main()
