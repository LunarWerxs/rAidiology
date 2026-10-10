"""Is recipe B more accurate than recipe A? A paired answer on the same held-out development scans.

    python compare.py "<what changed>" --a <prediction csv>... --b <prediction csv>... [--resamples 2000]
        [--noise D] [--dry]

Both files are prediction rows in the scorer's format (an out-of-fold file of one fold, say, from
two versions of the segmenter). Only the scans both cover are compared. The mean Dice of each is
taken exactly as the scorecard takes it (`score.per_method`), on the same resampled sets of scans,
and the difference is resampled inside each draw, so a hard draw counts against both alike.

The verdict, written to the run ledger as an `experiment` (`python core/ledger.py tried`):
**kept** when the whole 95% interval of B minus A is above zero, **dropped** when it is below,
**inconclusive** otherwise.

**Two trainings are two draws.** The interval is over scans: it says these two models differ on these
scans, not that the recipe made the difference. When B is a retrained model, give `--noise` the
seed-to-seed difference measured for that fold (the same recipe trained with two seeds and compared
here), and B has to beat A by more than that as well. Reading the same model two ways has no such
noise and leaves it at 0. The held-out test split is never an argument here: an accuracy A/B is
decided on development folds, and the test is opened once, by `final_eval.py`, for what was kept.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import datasets  # noqa: E402
import score  # noqa: E402
from core import ledger  # noqa: E402


def rows(files):
    """One recipe's prediction rows, from one file or several (a fold each, say)."""
    d = pd.concat([pd.read_csv(f, dtype={"scan": str}) for f in files], ignore_index=True)
    if (d["split"] == "test").any():
        raise SystemExit("%s holds test rows: accuracy A/Bs are decided on development data only" % ", ".join(files))
    if d.duplicated(["scan", "structure"]).any():
        raise SystemExit("%s: a scan appears twice; give each fold's file once" % ", ".join(files))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--a", required=True, nargs="+")
    ap.add_argument("--b", required=True, nargs="+")
    ap.add_argument("--resamples", type=int, default=score.RESAMPLES)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="seed-to-seed mean Dice difference for this fold; B must beat A by more")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    ra, rb = rows(a.a), rows(a.b)
    scans = sorted(set(ra["scan"]) & set(rb["scan"]))
    if len(scans) < score.MIN_SCANS:
        raise SystemExit("only %d scans in common: not a comparison" % len(scans))
    ra, rb = ra[ra["scan"].isin(scans)], rb[rb["scan"].isin(scans)]
    draws = score.Draws(scans, a.resamples, datasets.SEED)
    ma, mb = score.per_method(ra, scans, draws), score.per_method(rb, scans, draws)
    diff, lo, hi = score.difference(mb, ma, draws)
    inv, inv_lo, inv_hi = draws.ci(mb["invention_draws"] - ma["invention_draws"])
    beyond = abs(diff) > a.noise
    verdict = "kept" if lo > 0 and beyond else "dropped" if hi < 0 and beyond else "inconclusive"
    moved = sorted(((mb["per_structure"][n][0] - ma["per_structure"][n][0], n)
                    for n in ma["per_structure"] if n in mb["per_structure"]
                    and np.isfinite(ma["per_structure"][n][0]) and np.isfinite(mb["per_structure"][n][0])))
    print("%s, on %d development scans:" % (a.what, len(scans)))
    print("  mean Dice A %.4f (%.4f to %.4f)" % ma["mean_dice"])
    print("  mean Dice B %.4f (%.4f to %.4f)" % mb["mean_dice"])
    print("  B minus A   %+.4f (95%% CI %+.4f to %+.4f)%s" % (diff, lo, hi,
          ", against a seed-to-seed noise of %.4f" % a.noise if a.noise else ""))
    print("  invention rate A %.4f, B %.4f, B minus A %+.4f (95%% CI %+.4f to %+.4f)%s"
          % (ma["invention_rate"][0], mb["invention_rate"][0], inv, inv_lo, inv_hi,
             ": B outlines absent structures more often" if inv_lo > 0 else ""))
    print("  lost most:   " + ", ".join("%s %+.3f" % (n, d) for d, n in moved[:4]))
    print("  gained most: " + ", ".join("%s %+.3f" % (n, d) for d, n in moved[::-1][:4]))
    print("%s" % verdict.upper())
    if a.dry:
        return
    ledger.write("experiment", a.what, time.time() - t0,
                 {"dice_a": (ma["mean_dice"][0], ""), "dice_b": (mb["mean_dice"][0], ""),
                  "dice_minus_reference": (diff, ""), "dice_minus_reference_low": (lo, ""),
                  "dice_minus_reference_high": (hi, ""),
                  "invention_a": (ma["invention_rate"][0], "share"), "invention_b": (mb["invention_rate"][0], "share"),
                  "invention_minus_reference": (inv, "share"), "invention_minus_reference_low": (inv_lo, "share"),
                  "invention_minus_reference_high": (inv_hi, "share"),
                  "scans": (len(scans), "count")},
                 settings={"verdict": verdict, "a": " ".join(ledger.shown(f) for f in a.a),
                           "b": " ".join(ledger.shown(f) for f in a.b),
                           "resamples": str(a.resamples), "noise": str(a.noise)},
                 note="modalities/ct/eval/compare.py")


if __name__ == "__main__":
    main()
