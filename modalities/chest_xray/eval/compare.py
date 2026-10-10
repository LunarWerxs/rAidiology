"""Judge one chest X-ray reader run (B) against another (A), on the pictures both predicted out of fold.

Each side is a reader version number (its `oof_logits.npz`, reader.py) or `baseline:<method>` (a
no-picture baseline's scores in `baselines_oof.npz`, baselines.py). The two are joined on the image
name; only pictures both predicted are compared, and it stops when fewer than 1,000 are shared.

For each of the 14 findings and for Abnormal: the AUROC of A, of B, and B minus A with a 95%
percentile interval from paired resamples of PATIENTS (the same resample for A and B, so a patient's
pictures stay together). The mean over the 14 findings gets its interval from the same resamples.
The verdict is on that mean: BETTER when the interval's low end is above 0; NOT WORSE when it is
above minus the margin; WORSE when the high end is below 0; else INCONCLUSIVE. When both are reader
runs, the mean seconds per epoch is compared too. Recorded in the ledger (unless --no-ledger).

Usage: python compare.py <a> <b> [--name "what B is"] [--boot 1000] [--margin 0.01] [--no-ledger]
       (a, b: a reader version number, or baseline:<method>)"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import ledger, paths  # noqa: E402
import datasets  # noqa: E402

LABELS = datasets.LABELS
N_FINDINGS = len(datasets.FINDINGS)
SEED = 20261003
MIN_SHARED = 1000


def run_dir(version):
    return os.path.join(paths.WORK, "eval", "chest", "reader-v%d" % version)


def parse(side):
    """'7' -> ('reader', 7); 'baseline:<method>' -> ('baseline', method)."""
    s = str(side)
    if s.startswith("baseline:"):
        return "baseline", s.split(":", 1)[1]
    return "reader", int(s)


def load(side):
    """(image, patient, y [n,15], score [n,15]) of one side."""
    kind, key = parse(side)
    if kind == "reader":
        path = os.path.join(run_dir(key), "oof_logits.npz")
        arr = "logit"
    else:
        path = os.path.join(paths.WORK, "eval", "chest", "baselines_oof.npz")
        arr = key
    if not os.path.exists(path):
        raise SystemExit("%s: no such file (%s)" % (side, path))
    d = np.load(path, allow_pickle=False)
    if arr not in d.files:
        raise SystemExit("%s: %s has no array '%s' (it has %s)" % (side, path, arr, ", ".join(d.files)))
    return d["image"].astype(str), d["patient"], d["y"], d[arr].astype(np.float64)


def epoch_seconds(version):
    with open(os.path.join(run_dir(version), "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return float(np.mean([s for fold in meta["folds"] for s in fold["epoch_seconds"]]))


class Auroc:
    """AUROC of one score column that can be re-weighted by integer row counts (a bootstrap draw).
    The rows are sorted once; ties count half."""

    def __init__(self, y, score):
        self.order = np.argsort(score, kind="stable")
        s = score[self.order]
        self.y = (y[self.order] > 0).astype(np.float64)
        self.group = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])]).astype(np.int64)
        self.n_groups = int(self.group[-1]) + 1 if len(s) else 0

    def __call__(self, w):
        w = w[self.order]
        wp, wn = w * self.y, w * (1 - self.y)
        P = np.bincount(self.group, wp, self.n_groups)
        N = np.bincount(self.group, wn, self.n_groups)
        tp, tn = P.sum(), N.sum()
        if tp == 0 or tn == 0:
            return np.nan
        below = np.cumsum(N) - N
        return float((P * (below + 0.5 * N)).sum() / (tp * tn))


def aurocs(y, score):
    return [Auroc(y[:, j], score[:, j]) for j in range(len(LABELS))]


def summary(fns, w):
    """The 15 AUROCs under row weights w, then the mean of the 14 findings."""
    per = np.array([f(w) for f in fns])
    return np.append(per, np.nanmean(per[:N_FINDINGS]))


def compare(a, b, boot=1000, margin=0.01):
    """Returns a dict: shared, patients, rows (label, A, B, diff, low, high), verdict, mean14 and abnormal."""
    ia, pa, ya, sa = load(a)
    ib, pb, yb, sb = load(b)
    shared = np.intersect1d(ia, ib)
    print("%s vs %s: %d pictures shared (A has %d, B has %d)" % (a, b, len(shared), len(ia), len(ib)))
    if len(shared) < MIN_SHARED:
        raise SystemExit("only %d pictures are shared (under %d): stopping" % (len(shared), MIN_SHARED))
    pos_a = pd.Series(np.arange(len(ia)), index=ia).loc[shared].to_numpy()
    pos_b = pd.Series(np.arange(len(ib)), index=ib).loc[shared].to_numpy()
    if (ya[pos_a] != yb[pos_b]).any():
        raise SystemExit("the two sides disagree on a picture's labels; they are not the same data")
    patient = pa[pos_a]
    y = ya[pos_a]
    fa, fb = aurocs(y, sa[pos_a]), aurocs(y, sb[pos_b])

    codes, uniq = pd.factorize(patient)
    n_pat = len(uniq)
    ones = np.ones(len(codes))
    va, vb = summary(fa, ones), summary(fb, ones)
    rng = np.random.default_rng(SEED)
    diffs = np.empty((boot, len(LABELS) + 1))
    for i in range(boot):
        counts = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)  # the same draw for A and B
        w = counts[codes].astype(np.float64)  # a patient drawn k times weighs each of its rows k
        diffs[i] = summary(fb, w) - summary(fa, w)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5], axis=0)
    names = LABELS[:N_FINDINGS] + ["mean of 14 findings", "Abnormal"]
    idx = list(range(N_FINDINGS)) + [len(LABELS), N_FINDINGS]  # summary(): 14 findings, Abnormal (14), then the mean (15)
    rows = [(n, va[j], vb[j], vb[j] - va[j], lo[j], hi[j]) for n, j in zip(names, idx)]
    m = rows[N_FINDINGS]
    verdict = ("BETTER" if m[4] > 0 else "NOT WORSE" if m[4] > -margin else "WORSE" if m[5] < 0 else "INCONCLUSIVE")
    return {"shared": len(shared), "patients": n_pat, "rows": rows, "verdict": verdict,
            "mean14": m, "abnormal": rows[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="a reader version number, or baseline:<method>")
    ap.add_argument("b", help="a reader version number, or baseline:<method>")
    ap.add_argument("--name", default="", help="what B is, for the ledger row")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    r = compare(args.a, args.b, args.boot, args.margin)
    print("%d patients; %d paired resamples of patients\n" % (r["patients"], args.boot))
    print("| label | A | B | B - A (95% CI) |\n|---|---|---|---|")
    for n, va, vb, d, lo, hi in r["rows"]:
        print("| %s | %.4f | %.4f | %+.4f (%+.4f to %+.4f) |" % (n, va, vb, d, lo, hi))
    print("\nVerdict (mean of 14 findings, margin %.3f): %s" % (args.margin, r["verdict"]))
    ka, kb = parse(args.a), parse(args.b)
    if ka[0] == "reader" and kb[0] == "reader":
        ea, eb = epoch_seconds(ka[1]), epoch_seconds(kb[1])
        print("epoch seconds: A %.1f, B %.1f; B is %.2fx the speed of A" % (ea, eb, ea / eb))
    if not args.no_ledger:
        _, _, _, d, lo, hi = r["mean14"]
        _, _, _, da, loa, hia = r["abnormal"]
        numbers = {"auroc.mean14_b_minus_a": (d, ""), "auroc.mean14_b_minus_a_low": (lo, ""),
                   "auroc.mean14_b_minus_a_high": (hi, ""), "auroc.Abnormal_b_minus_a": (da, ""),
                   "auroc.Abnormal_b_minus_a_low": (loa, ""), "auroc.Abnormal_b_minus_a_high": (hia, ""),
                   "shared_images": (r["shared"], "count")}
        ledger.write("experiment", args.name or "chest X-ray %s vs %s" % (args.a, args.b), time.time() - t0, numbers,
                     settings={"a": args.a, "b": args.b, "boot": args.boot, "margin": args.margin, "seed": SEED},
                     note=r["verdict"])


if __name__ == "__main__":
    main()
