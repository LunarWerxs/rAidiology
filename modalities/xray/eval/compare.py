"""Is detector B more accurate than detector A? Paired, on the same development pictures.

    python modalities/xray/eval/compare.py "<what changed>" --a <oof_dev.csv> --b <oof_dev.csv> [--noise D]

Both files are a detector's out-of-fold development scores (detector.py writes `oof_dev.csv` beside
its fold weights: dataset, image, patient, fold, truth, score). Per collection, the AUROC of each on
the pictures both scored, and the difference with score.py's patient bootstrap. **kept** when the
whole 95% interval is above zero on every collection and beyond the seed-to-seed noise (`--noise`,
the difference two seeds of one recipe landed apart: one number, or one a collection as
`fracatlas=0.0106,grazpedwri=0.0030`, since a seed moves the two by different amounts), **dropped**
when any is below, else
**inconclusive**. Written to the run ledger as an `experiment`. The test split is never an argument.
"""
import argparse
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(os.path.dirname(HERE)))]
import datasets  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every eval script here
import mirror  # noqa: E402  arkitect-allow: test-floor-loosened - the patient bootstrap of AUROC differences lives there
from core import ledger  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every eval script here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise", default="0")
    a = ap.parse_args()
    if "=" in a.noise:
        noise = {k: float(v) for k, v in (i.split("=", 1) for i in a.noise.split(","))}
    else:
        noise = {d: float(a.noise) for d in datasets.NAMES}
    t0 = time.time()
    ra = pd.read_csv(a.a, dtype={"patient": str, "image": str})
    rb = pd.read_csv(a.b, dtype={"patient": str, "image": str})
    both = ra.merge(rb[["dataset", "image", "score"]], on=["dataset", "image"], suffixes=("_a", "_b"))
    numbers, verdicts = {}, []
    print("%s:" % a.what)
    for d in datasets.NAMES:
        m = both[both.dataset == d]
        t = m.truth.values.astype(int)
        auc_a, auc_b, diff, lo, hi, n_p = mirror.paired_auroc(t, m.score_b.values, m.score_a.values, m.patient.values)
        beyond = abs(diff) > noise.get(d, 0.0)
        verdicts.append("up" if lo > 0 and beyond else "down" if hi < 0 and beyond else "level")
        print("  %s, %d pictures of %d patients: AUROC A %.4f, B %.4f, B minus A %+.4f (95%% CI %+.4f to %+.4f)%s"
              % (d, len(m), n_p, auc_a, auc_b, diff, lo, hi, ", against a seed noise of %.4f" % noise[d] if noise.get(d) else ""))
        numbers.update({"%s.auroc_a" % d: (auc_a, ""), "%s.auroc_b" % d: (auc_b, ""),
                        "%s.auroc_minus_reference" % d: (diff, ""), "%s.auroc_minus_reference_low" % d: (lo, ""),
                        "%s.auroc_minus_reference_high" % d: (hi, "")})
    verdict = "dropped" if "down" in verdicts else "kept" if all(v == "up" for v in verdicts) else "inconclusive"
    print(verdict.upper())
    ledger.write("experiment", a.what, time.time() - t0, numbers,
                 settings={"verdict": verdict, "a": ledger.shown(a.a), "b": ledger.shown(a.b), "noise": str(a.noise)},
                 note="modalities/xray/eval/compare.py")


if __name__ == "__main__":
    main()
