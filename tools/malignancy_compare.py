"""Paired comparison of two cancer-or-not versions on the same LIDC lesions, out of fold.

Each version's <WORK>/eval/lung/malignancy-v<N>/oof.csv holds one logit per held-out lesion. The lesions
both scored (same scan and centre) are compared: B - A of the AUROC, with a 95% interval from 1000
bootstrap resamples of patients (a patient's lesions move together; work/eval/lung/index.csv names each
scan's patient). The verdict reads as tools/froc_compare.py's: BETTER when the interval's low end is above
0, NOT WORSE when above minus the margin, WORSE when its high end is below 0, else INCONCLUSIVE (until
2026-10-10 WORSE needed the high end below minus the margin; v4 against v2, -0.0165 to -0.0029, read
INCONCLUSIVE then and reads WORSE now; it was dropped either way). With --by-size the compared number is the within-size AUROC (pairs of one cancer-looking and
one benign lesion in the same diameter band, bands weighted by their pairs, recomputed in each resample),
since the diameter alone scores 0.939 on these labels.

Usage: python tools/malignancy_compare.py <A version> <B version> [--name text] [--margin 0.01] [--boot 1000]
       [--by-size]
       python tools/malignancy_compare.py --selftest"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "ct", "eval"))
from core import ledger, paths  # noqa: E402
from nodule_malignancy import auroc  # noqa: E402

BANDS = [(6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20)]  # mm of diameter_mm, for --by-size
KEY = ["seriesuid", "coordX", "coordY", "coordZ"]


def load(version):
    df = pd.read_csv(os.path.join(paths.WORK, "eval", "lung", "malignancy-v%d" % version, "oof.csv"))
    df[["coordX", "coordY", "coordZ"]] = df[["coordX", "coordY", "coordZ"]].round(2)
    return df


def band_pairs(y, score, diam):
    """Per band: (n1, n0, U) where U counts the (label 1, label 0) pairs in that band with the 1 scored
    higher, ties half. Empty bands give zeros."""
    out = []
    for lo, hi in BANDS:
        k = (diam >= lo) & (diam < hi)
        yy, ss = np.asarray(y)[k], np.asarray(score, float)[k]
        n1, n0 = int((yy == 1).sum()), int((yy == 0).sum())
        if n1 == 0 or n0 == 0:
            out.append((n1, n0, 0.0))
            continue
        r = pd.Series(ss).rank().to_numpy()
        out.append((n1, n0, float(r[yy == 1].sum() - n1 * (n1 + 1) / 2)))
    return out


def within_band_auroc(y, score, diam):
    """AUROC over pairs of one label-1 and one label-0 lesion in the same diameter band (ties half); the
    mean of the bands' AUROCs weighted by n1 * n0. NaN when no band has a pair. Lesions outside the bands
    are left out."""
    parts = band_pairs(y, score, diam)
    pairs = sum(n1 * n0 for n1, n0, _ in parts)
    return sum(u for _, _, u in parts) / pairs if pairs else float("nan")


def selftest():
    # two bands by hand: [6,8): labels 1,1,0 scores 3,1,2 -> pairs (3>2) (1<2) = 1 of 2;
    # [8,10): labels 1,0,0 scores 5,5,4 -> tie half + win = 1.5 of 2; a lesion at 25 mm is ignored
    y = np.array([1, 1, 0, 1, 0, 0, 1])
    s = np.array([3, 1, 2, 5, 5, 4, 0.0])
    d = np.array([6, 7, 7.9, 8, 9, 9.9, 25.0])
    assert abs(within_band_auroc(y, s, d) - (1 + 1.5) / 4) < 1e-12
    assert np.isnan(within_band_auroc(np.array([1, 1]), np.array([1.0, 2.0]), np.array([7.0, 7.0])))  # no pair
    rng = np.random.default_rng(1)
    n = 6000
    diam = rng.uniform(6, 20, n)
    # the label depends on the band only (its middle), not on the diameter inside it
    mid = np.array([(lo + hi) / 2 for lo, hi in BANDS])[np.digitize(diam, [hi for _, hi in BANDS[:-1]])]
    lab = (rng.random(n) < 1 / (1 + np.exp(-(mid - 13) * 1.2))).astype(int)
    size_only = diam + rng.normal(0, 0.3, n)
    plain = auroc(lab, size_only)
    assert plain > 0.85, plain
    w0 = within_band_auroc(lab, diam, diam)
    assert abs(w0 - 0.5) < 0.1, w0
    w1 = within_band_auroc(lab, diam + 0.6 * lab, diam)
    assert w1 > 0.65, w1
    print("selftest ok")


def record(led, lesions, patients, positives, resamples, pa, pb, lo, hi, line):
    """The numbers every mode prints, and the verdict line, into the ledger."""
    led.number("lesions", lesions, "count").number("patients", patients, "count")
    led.number("cancer_looking", positives, "count").number("resamples", resamples, "count")
    led.number("auroc_a", pa).number("auroc_b", pb).number("auroc_diff", pb - pa)
    led.number("auroc_diff_low", lo).number("auroc_diff_high", hi)
    led.note_is(line)


def main():
    if "--selftest" in sys.argv[1:]:
        selftest()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=int)
    ap.add_argument("b", type=int)
    ap.add_argument("--name", default="")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--by-size", action="store_true")
    a = ap.parse_args()
    with ledger.run("experiment", "malignancy v%d vs v%d%s" % (a.b, a.a, " by size" if a.by_size else "")) as led:
        led.settings_from({"a": a.a, "b": a.b, "margin": a.margin, "boot": a.boot, "by_size": a.by_size,
                           "label": a.name})
        compare(a, led)


def compare(a, led):
    m = load(a.a).merge(load(a.b)[KEY + ["label", "logit"]], on=KEY, suffixes=("_a", "_b"))
    if (m.label_a != m.label_b).any():
        raise SystemExit("the two versions label %d shared lesions differently" % int((m.label_a != m.label_b).sum()))
    who = pd.read_csv(os.path.join(paths.WORK, "eval", "lung", "index.csv")).set_index("seriesuid").patient
    m["patient"] = m.seriesuid.map(who).fillna(m.seriesuid)
    y, sa, sb = m.label_a.to_numpy(), m.logit_a.to_numpy(float), m.logit_b.to_numpy(float)
    diam = m.diameter_mm.to_numpy(float)
    measure = (lambda yy, ss, dd: within_band_auroc(yy, ss, dd)) if a.by_size else (lambda yy, ss, dd: auroc(yy, ss))
    groups = [g.to_numpy() for _, g in pd.Series(np.arange(len(m))).groupby(m.patient.to_numpy())]
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(a.boot):
        pick = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        d = measure(y[pick], sb[pick], diam[pick]) - measure(y[pick], sa[pick], diam[pick])
        if d == d:
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pa, pb = auroc(y, sa), auroc(y, sb)
    wa, wb = measure(y, sa, diam), measure(y, sb, diam)
    verdict = ("BETTER" if lo > 0 else "NOT WORSE" if lo > -a.margin else
               "WORSE" if hi < 0 else "INCONCLUSIVE")
    if a.name:
        print(a.name)
    print("%d shared lesions of %d patients (%d cancer-looking); %d bootstrap resamples of patients"
          % (len(m), len(groups), int(y.sum()), len(diffs)))
    if not a.by_size:
        print("AUROC  A %.4f  B %.4f  B - A %+.4f (%+.4f to %+.4f)" % (pa, pb, pb - pa, lo, hi))
        line = "Verdict (AUROC, margin %.3f): %s" % (a.margin, verdict)
        print(line)
        record(led, len(m), len(groups), int(y.sum()), len(diffs), pa, pb, lo, hi, line)
        return
    span = "%d to %d mm" % (BANDS[0][0], BANDS[-1][1])
    print("Measure: within-size AUROC (pairs of one cancer-looking and one benign lesion in the same band, "
          "%s; bands weighted by their pairs)" % span)
    print("Plain AUROC, all lesions (reported)  A %.4f  B %.4f  B - A %+.4f" % (pa, pb, pb - pa))
    print("Diameter alone: plain AUROC over all lesions %.4f; within-size AUROC %.4f"
          % (auroc(y, diam), within_band_auroc(y, diam, diam)))
    for (blo, bhi), A, B, D in zip(BANDS, band_pairs(y, sa, diam), band_pairs(y, sb, diam),
                                   band_pairs(y, diam, diam)):
        n1, n0 = A[0], A[1]
        f = lambda t: "%.4f" % (t[2] / (t[0] * t[1])) if t[0] * t[1] else "   -  "
        print("  band %2d to %2d mm: %4d cancer-looking, %4d benign  AUROC A %s  B %s  (diameter alone %s)"
              % (blo, bhi, n1, n0, f(A), f(B), f(D)))
    print("Within-size AUROC (%s)  A %.4f  B %.4f  B - A %+.4f (%+.4f to %+.4f)" % (span, wa, wb, wb - wa, lo, hi))
    line = "Verdict (within-size AUROC, margin %.3f): %s" % (a.margin, verdict)
    print(line)
    record(led, len(m), len(groups), int(y.sum()), len(diffs), pa, pb, lo, hi, line)
    led.number("within_size_auroc_a", wa).number("within_size_auroc_b", wb)
    led.number("within_size_auroc_diff", wb - wa).number("within_size_auroc_diff_low", lo)
    led.number("within_size_auroc_diff_high", hi)


if __name__ == "__main__":
    main()
