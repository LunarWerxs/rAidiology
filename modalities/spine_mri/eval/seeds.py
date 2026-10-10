"""Two seeds' graders averaged against each alone, on SPIDER's development discs (out-of-fold).

The rule (docs/efficiency, "Two seeds averaged"): the shipped grader becomes the two seeds' models
averaged when the average is at least each seed alone on exact grade, within one grade and kappa.
Averaging is of the logits, as `grader.logits` averages its nets, so a grade is still 1 plus the
number of cut-point logits above zero.
Usage: python seeds.py <S0/oof_logits.npz> <S1/oof_logits.npz> [--which nested]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import METRICS  # noqa: E402


def spider(path, which):
    d = np.load(path, allow_pickle=False)
    sp = d["collection"] == "spider" if "collection" in d.files else np.ones(len(d["y"]), bool)
    key = list(zip(d["patient"][sp].astype(int), d["ivd"][sp].astype(int)))
    return dict(zip(key, d[which][sp])), dict(zip(key, 1 + (d["y"][sp][:, :4] > 0.5).sum(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s0")
    ap.add_argument("s1")
    ap.add_argument("--which", default="nested", choices=["nested", "single"])
    a = ap.parse_args()
    (z0, t0), (z1, _) = spider(a.s0, a.which), spider(a.s1, a.which)
    keys = sorted(set(z0) & set(z1))
    t = np.array([t0[k] for k in keys])
    l0, l1 = np.stack([z0[k] for k in keys]), np.stack([z1[k] for k in keys])
    grade = {name: 1 + (z[:, :4] > 0).sum(1) for name, z in (("seed 0", l0), ("seed 1", l1), ("averaged", (l0 + l1) / 2))}
    ok = True
    print("SPIDER development, %d discs, %s ensembles" % (len(keys), a.which))
    for m, fn in METRICS.items():
        v = {name: fn(t, g) for name, g in grade.items()}
        passes = v["averaged"] >= max(v["seed 0"], v["seed 1"])
        ok &= passes
        print("  %-10s seed 0 %.4f  seed 1 %.4f  averaged %.4f  %s" % (m, v["seed 0"], v["seed 1"], v["averaged"],
                                                                     "at least each" if passes else "below one"))
    print("AVERAGE SHIPS" if ok else "AVERAGE DOES NOT SHIP")


if __name__ == "__main__":
    main()
