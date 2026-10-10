"""A/B of two disc-grader trainings on SPIDER's development discs, out of fold, for a trial's verdict.

Each side is a run's `oof_logits.npz` (grader.py train). The grade is 1 + the number of cumulative
logits above 0 (a temperature never moves that line), read from the nested ensembles' predictions
(`--which nested`, what ships) or the single outer models'. Only SPIDER's discs are compared, and
only discs both runs predicted; a collection one run trained on beside SPIDER is reported on its
own when that run has out-of-fold predictions for it.

For exact grade, within one grade and quadratic weighted kappa: B minus A with a 95% percentile
interval from score.py's paired bootstrap by patient (2,000 draws, its seed). A difference is "up"
or "down" when its interval clears zero and its size clears the given seed noise (`--noise
exact=0.03,qwk=0.02`: what the same recipe moves at another seed), or measured from two runs of
one recipe at two seeds (`--noise-from seed0/oof_logits.npz,seed1/oof_logits.npz`: the size of each
metric's difference on the same discs). The verdict, as the X-ray compare: dropped when any is down,
kept when all are up, else inconclusive. Recorded in the ledger.

Usage: python compare.py "<what>" --a A/oof_logits.npz --b B/oof_logits.npz
                         [--noise k=v,... | --noise-from S0/oof_logits.npz,S1/oof_logits.npz] [--which nested]"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import ledger  # noqa: E402
from score import m_auroc, m_exact, m_qwk, m_within_one, paired_bootstrap, bootstrap  # noqa: E402
from spider import ITEMS  # noqa: E402

METRICS = {"exact": m_exact, "within_one": m_within_one, "qwk": m_qwk}


def frame(path, which):
    d = np.load(path, allow_pickle=False)
    z, y = d[which], d["y"]
    n = len(y)
    coll = d["collection"] if "collection" in d.files else np.array(["spider"] * n)
    return pd.DataFrame({"patient": d["patient"].astype(int), "ivd": d["ivd"].astype(int),
                         "collection": coll, "truth": 1 + (y[:, :4] > 0.5).sum(1),
                         "grade": 1 + (z[:, :4] > 0).sum(1)})


def spider_pair(pa, pb, which):
    """SPIDER discs both runs predicted: truth, A's grade, B's grade, patient."""
    fa, fb = frame(pa, which), frame(pb, which)
    both = fa[fa.collection == "spider"].merge(fb[fb.collection == "spider"], on=["patient", "ivd"], suffixes=("_a", "_b"))
    if (both.truth_a != both.truth_b).any():
        raise SystemExit("the two runs disagree on the truth of a disc; they were not trained on the same grades")
    return both.truth_a.values, both.grade_a.values, both.grade_b.values, both.patient.values


def item_pair(pa, pb, which, item):
    """SPIDER discs both runs predicted: an item's truth, A's logit, B's logit, patient."""
    j = 4 + ITEMS.index(item)
    out = []
    for p in (pa, pb):
        d = np.load(p, allow_pickle=False)
        sp = d["collection"] == "spider" if "collection" in d.files else np.ones(len(d["y"]), bool)
        out.append(pd.DataFrame({"patient": d["patient"][sp].astype(int), "ivd": d["ivd"][sp].astype(int),
                                 "truth": d["y"][sp, j], "score": d[which][sp, j]}))
    both = out[0].merge(out[1], on=["patient", "ivd"], suffixes=("_a", "_b"))
    return both.truth_a.values.astype(int), both.score_a.values, both.score_b.values, both.patient.values


def seed_noise(pa, pb, which, items=()):
    """{metric: |B - A|} for one recipe at two seeds (and `items`' AUROC as <item>_auroc)."""
    t, ga, gb, _ = spider_pair(pa, pb, which)
    out = {name: round(abs(fn(t, gb) - fn(t, ga)), 4) for name, fn in METRICS.items()}
    for item in items:
        t, sa, sb, _ = item_pair(pa, pb, which, item)
        out[item + "_auroc"] = round(abs(m_auroc(t, sb) - m_auroc(t, sa)), 4)
    return out


def _resolve_noise(a, items):
    noise = {k: float(v) for k, v in (i.split("=", 1) for i in a.noise.split(",") if i)}
    if a.noise_from:
        noise = seed_noise(*a.noise_from.split(","), a.which, items)
        a.noise = ",".join("%s=%.4f" % kv for kv in noise.items())
        print("seed noise, from %s: %s" % (a.noise_from, a.noise))
    return noise


def _record(name, va, vb, lo, hi, noise, numbers, verdicts):
    beyond = abs(vb - va) > noise.get(name, 0.0)
    verdicts.append("up" if lo > 0 and beyond else "down" if hi < 0 and beyond else "level")
    print("  %-10s A %.4f  B %.4f  B minus A %+.4f (95%% CI %+.4f to %+.4f)%s"
          % (name, va, vb, vb - va, lo, hi, ", against a seed noise of %.4f" % noise[name] if name in noise else ""))
    numbers.update({"%s_a" % name: (va, ""), "%s_b" % name: (vb, ""), "%s_b_minus_a" % name: (vb - va, ""),
                    "%s_b_minus_a_low" % name: (lo, ""), "%s_b_minus_a_high" % name: (hi, "")})


def _own_out_of_fold(fb, numbers):
    for c in sorted(set(fb.collection) - {"spider"}):
        m = fb[fb.collection == c]
        tt, gg, pp = m.truth.values, m.grade.values, m.patient.values
        parts = []
        for name, fn in METRICS.items():
            lo, hi, _ = bootstrap(pp, fn, tt, gg)
            parts.append("%s %.4f (%.4f to %.4f)" % (name, fn(tt, gg), lo, hi))
            numbers["%s.%s_b" % (c, name)] = (fn(tt, gg), "")
        print("  B's own out-of-fold on %s, %d discs of %d patients: %s" % (c, len(m), len(set(pp)), ", ".join(parts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise", default="")
    ap.add_argument("--noise-from", default="")
    ap.add_argument("--which", default="nested", choices=["nested", "single"])
    ap.add_argument("--items", default="", help="items whose AUROC is compared too, e.g. bulging,spondylolisthesis")
    a = ap.parse_args()
    items = [i for i in a.items.split(",") if i]
    noise = _resolve_noise(a, items)
    t0 = time.time()
    t, ga, gb, pts = spider_pair(a.a, a.b, a.which)
    fb = frame(a.b, a.which)
    print("%s: SPIDER development, %d discs of %d patients, %s ensembles" % (a.what, len(t), len(set(pts)), a.which))
    numbers, verdicts = {}, []
    for name, fn in METRICS.items():
        va, vb = fn(t, ga), fn(t, gb)
        lo, hi, _ = paired_bootstrap(pts, fn, (t, gb), (t, ga))
        _record(name, va, vb, lo, hi, noise, numbers, verdicts)
    for item in items:
        name = item + "_auroc"
        ti, sa, sb, pi = item_pair(a.a, a.b, a.which, item)
        va, vb = m_auroc(ti, sa), m_auroc(ti, sb)
        lo, hi, _ = paired_bootstrap(pi, m_auroc, (ti, sb), (ti, sa))
        _record(name, va, vb, lo, hi, noise, numbers, verdicts)
    _own_out_of_fold(fb, numbers)
    verdict = "dropped" if "down" in verdicts else "kept" if all(v == "up" for v in verdicts) else "inconclusive"
    print(verdict.upper())
    ledger.write("experiment", a.what, time.time() - t0, numbers,
                 settings={"verdict": verdict, "a": ledger.shown(a.a), "b": ledger.shown(a.b),
                           "noise": a.noise, "which": a.which},
                 note="modalities/spine_mri/eval/compare.py")


if __name__ == "__main__":
    main()
