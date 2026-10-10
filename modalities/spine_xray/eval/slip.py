"""Slipped vertebrae from side-view corners, measured the way a radiologist grades them (Meyerding).

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646 (LICENSING.md, BUU-LSPINE row). Plan and pass bars:
docs/efficiency/README.md, "Spine X-ray: vertebra corners and slipped vertebrae from BUU-LSPINE".

The slip at a level. Level i (0..4) is L(i+1) over L(i+2), the last being L5 over S1. Take the lower
vertebra's upper endplate (the line from its front corner A to its back corner P) and the upper
vertebra's back-lower corner Q. The signed slip is how far Q sits forward of P along that endplate, as a
share of the endplate's length: dot(Q - P, A - P) / |A - P|^2. Positive is forward (anterolisthesis, the
dataset's class 3), negative behind (retrolisthesis, class 4). Which end is front, and which rows are
which, is in buu.py (the front end of a side-view line is the one with the smaller x; on the drawn
lines it gives +0.12 on class 3 and -0.16 on class 4). The written score (largest absolute raw slip
over the five levels) is printed for context only: a normal spine already sits about 0.05 behind at most
levels, so it gives AUROC 0.726 even on the drawn corners.

The patient's score is the AMENDED one: for held-out fold k, each level's slip minus that level's median
slip over the patients of the other folds (same source: drawn corners with --truth, the model's out-of-fold
corners otherwise; never the held-out fold), then the largest absolute value over the five levels.

The reading is out of fold: the corners come from corners.py's held-out predictions of side view (LA)
fold k, made by a net that never saw fold k's patients. `--truth` scores the drawn corners instead, an
upper bound for what any corner reader could do with this rule.

Bar (amended, docs/efficiency/README.md): per-patient AUROC of the amended score for "any side-view line
of class 3 or 4" over all five folds, KEPT when it is not worse than the drawn corners' on the same patients
at margin 0.03: the low end of the paired patient-bootstrap 95% interval (1000 resamples, seed 0, one
resample for both) of model AUROC minus drawn-corner AUROC is -0.03 or above. The 0.85 target on the AUROC's
low end is printed as context. Accuracy, sensitivity and specificity come from a threshold on the amended
score chosen inside each training fold (Youden's J on the other folds' scores) and applied to the held-out
fold; the paper's 92.3% is printed beside accuracy, and since about 95% of patients have no slip a reader
that never calls one is already about 95% accurate, so sensitivity and specificity say more.
With --truth there is nothing to judge: it prints the drawn corners' amended AUROC and interval, and
writes no ledger row.

Usage: python slip.py --version N [--folds all]       corners from corners.py train --view LA --version N
       python slip.py --truth
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import ledger  # noqa: E402
import buu  # noqa: E402

BOOT = 1000
BOOT_SEED = 0
BAR = 0.85                   # the original target, printed as context
MARGIN = 0.03                # amended bar: model AUROC minus drawn-corner AUROC, interval low end >= -MARGIN
PAPER_ACCURACY = 0.923       # Klinwichit et al. 2023, side view, slips called right
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
VIEW = "LA"


def slips(pts):
    """(N, 22, 2) corners -> (N, 5) signed Meyerding slips, one per level.
    Line r's points are 2r (front) and 2r + 1 (back); the upper vertebra's lower edge is line 2i + 1 and
    the lower vertebra's upper edge line 2i + 2."""
    pts = np.asarray(pts, np.float64)
    out = np.zeros((len(pts), 5))
    for i in range(5):
        a, p = pts[:, 2 * (2 * i + 2)], pts[:, 2 * (2 * i + 2) + 1]      # lower vertebra's upper edge
        q = pts[:, 2 * (2 * i + 1) + 1]                                  # upper vertebra's back-lower corner
        ap = a - p
        out[:, i] = ((q - p) * ap).sum(-1) / np.maximum((ap * ap).sum(-1), 1e-9)
    return out


def amended(sl, fold):
    """The amended score: each level's slip minus that level's median over the patients of the OTHER folds
    (from the same source as sl, never the held-out fold), then the largest absolute value over the levels."""
    out = np.zeros(len(sl))
    for k in sorted(set(fold.tolist())):
        te = fold == k
        out[te] = np.abs(sl[te] - np.median(sl[~te], axis=0)).max(axis=1)
    return out


def auroc(score, truth):
    from sklearn.metrics import roc_auc_score
    t = np.asarray(truth, int)
    if t.min() == t.max():
        return float("nan")
    return float(roc_auc_score(t, score))


def youden(score, truth):
    """The cut (score >= cut calls a slip) with the best sensitivity + specificity - 1."""
    s, t = np.asarray(score, float), np.asarray(truth, int)
    pos, neg = max(int(t.sum()), 1), max(int((1 - t).sum()), 1)
    best, at = -2.0, float(np.median(s))
    for c in np.unique(s):
        y = s >= c
        j = (y & (t == 1)).sum() / pos + ((~y) & (t == 0)).sum() / neg - 1.0
        if j > best:
            best, at = j, float(c)
    return at


def rates(call, truth):
    call, truth = np.asarray(call, bool), np.asarray(truth, bool)
    acc = float((call == truth).mean())
    sens = float((call & truth).sum() / max(truth.sum(), 1))
    spec = float((~call & ~truth).sum() / max((~truth).sum(), 1))
    return acc, sens, spec


def verified_report(pid, label, level_label, score, call, level_score):
    """Print and return (ledger numbers) the out-of-fold slip results on the 400 verified patients only (buu.verified_ids):
    AUROC, accuracy, sensitivity, specificity at the calls already made, and each level's AUROC.
    level_score (n, 5) is the score the caller judged each level's marks by."""
    m = np.isin(pid, sorted(buu.verified_ids()))
    y, c = np.asarray(label)[m], np.asarray(call)[m]
    auc = auroc(np.asarray(score)[m], y)
    acc, sens, spec = rates(c, y)
    print("on the 400 verified patients: %d patients, %d with a slip; AUROC %.4f, accuracy %.3f, sensitivity %.3f, "
          "specificity %.3f (thresholds as chosen on all patients)" % (int(m.sum()), int(y.sum()), auc, acc, sens, spec))
    numbers = {"auroc_verified400": (auc, ""), "accuracy_verified400": (acc, ""), "sensitivity_verified400": (sens, ""),
               "specificity_verified400": (spec, ""), "patients_verified400": (int(m.sum()), "count"),
               "patients_with_slip_verified400": (int(y.sum()), "count")}
    print("per level, on the 400 verified patients:")
    for i, name in enumerate(LEVELS):
        ll = np.asarray(level_label)[m, i]
        pos = int(ll.sum())
        lv = auroc(np.asarray(level_score)[m, i], ll) if pos >= 5 else float("nan")
        print("  %-6s marks %3d  AUROC for the level's marks %s" % (name, pos, "%.3f" % lv if lv == lv else "n/a"))
        numbers["auroc_level%d_verified400" % (i + 1)] = (lv, "")
        numbers["marks_level%d_verified400" % (i + 1)] = (pos, "count")
    return numbers


def main():
    t0 = time.time()
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, default=None, help="corners.py version whose LA out-of-fold corners to read")
    ap.add_argument("--folds", default="all", help="'all' or comma-separated fold numbers (default all)")
    ap.add_argument("--truth", action="store_true", help="score the drawn corners (an upper bound)")
    a = ap.parse_args()
    if not a.truth and a.version is None:
        raise SystemExit("give --version N (corners.py's), or --truth")
    ids, truth_pts, classes = buu.annotations(VIEW)
    fold_all = buu.folds_of(ids)
    label_all = buu.slip_label(classes)
    level_label_all = np.isin(classes[:, 1:10:2], buu.SLIP_CLASSES).astype(int)       # lines 1, 3, 5, 7, 9
    if a.truth:
        folds = list(range(buu.FOLDS)) if a.folds == "all" else sorted({int(x) for x in a.folds.split(",")})
        keep = np.isin(fold_all, folds)
        pts = truth_pts[keep]
        source = "drawn corners (upper bound)"
    else:
        import corners
        folds = corners.parse_folds(a.folds)
        r = corners.load_oof(VIEW, a.version, folds)
        where = {int(i): n for n, i in enumerate(ids)}
        keep = np.array(sorted(where[int(i)] for i in r["ids"]))
        order = {int(i): n for n, i in enumerate(r["ids"])}
        pts = r["pred"][[order[int(i)] for i in ids[keep]]]
        source = "corners.py v%d out-of-fold corners" % a.version
    pid, fold, label, level_label = ids[keep], fold_all[keep], label_all[keep], level_label_all[keep]
    if len(set(fold.tolist())) < 2:
        raise SystemExit("a threshold chosen inside the training folds needs at least two folds scored; got %s" % folds)
    sl = slips(pts)
    true_sl = slips(truth_pts[keep])
    written = np.abs(sl).max(axis=1)
    score = amended(sl, fold)
    n_pos = int(label.sum())
    print("%s: %d patients in folds %s, %d with a side-view slip (class 3 or 4); score = largest |slip - level's usual slip| over 5 levels"
          % (source, len(pid), ",".join(map(str, sorted(set(fold.tolist())))), n_pos))
    # per level: how well the slip tracks the drawn one, and how well it flags that level's marks
    print("per level (signed slip as a share of the endplate's length):")
    for i, name in enumerate(LEVELS):
        err = np.abs(sl[:, i] - true_sl[:, i])
        pos = int(level_label[:, i].sum())
        lv = auroc(np.abs(sl[:, i]), level_label[:, i]) if pos >= 5 else float("nan")
        print("  %-6s marks %3d  slip from drawn corners on marked lines %+.3f, on others %+.3f; mean |error| vs drawn %.3f; AUROC for the level's marks %s"
              % (name, pos, float(true_sl[level_label[:, i] == 1, i].mean()) if pos else float("nan"),
                 float(true_sl[level_label[:, i] == 0, i].mean()), float(err.mean()),
                 "%.3f" % lv if lv == lv else "n/a"))
    # threshold inside each training fold, applied to the held-out fold
    call = np.zeros(len(pid), bool)
    cuts = {}
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        cuts[k] = youden(score[tr], label[tr])
        call[te] = score[te] >= cuts[k]
    acc, sens, spec = rates(call, label)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(pid), (BOOT, len(pid)))
    accs, senss, specs = [], [], []
    for r in idx:
        ac, se, sp = rates(call[r], label[r])
        accs.append(ac), senss.append(se), specs.append(sp)

    def boot_auc(sc):
        return np.array([auroc(sc[r], label[r]) for r in idx])

    auc, aucs = auroc(score, label), boot_auc(score)
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    print("%s, written score: AUROC %.4f (context only: largest |raw slip| over 5 levels, before the amendment)"
          % ("Drawn corners" if a.truth else "Model corners", auroc(written, label)))
    if a.truth:
        print("Drawn corners, amended score: AUROC %.4f (patient-bootstrap 95%% CI %.4f to %.4f, %d resamples, seed %d)"
              % (auc, lo, hi, BOOT, BOOT_SEED))
        return                  # a measurement of the ground truth, nothing to judge: no ledger row
    dscore = amended(true_sl, fold)
    dauc, daucs = auroc(dscore, label), boot_auc(dscore)
    dlo, dhi = np.nanpercentile(daucs, [2.5, 97.5])
    diff = aucs - daucs                      # same resample for both
    dflo, dfhi = np.nanpercentile(diff, [2.5, 97.5])
    d = auc - dauc
    print("Model corners, amended score: AUROC %.4f (patient-bootstrap 95%% CI %.4f to %.4f, %d resamples, seed %d)"
          % (auc, lo, hi, BOOT, BOOT_SEED))
    print("Drawn corners, amended score: AUROC %.4f (%.4f to %.4f)" % (dauc, dlo, dhi))
    print("Model minus drawn corners: %+.4f (paired patient-bootstrap 95%% CI %+.4f to %+.4f)" % (d, dflo, dfhi))
    print("target, context only: AUROC low end %.4f against %.2f" % (lo, BAR))
    print("thresholds chosen inside each training fold (amended score): %s"
          % ", ".join("fold %d %.3f" % (k, c) for k, c in cuts.items()))
    print("at those thresholds: accuracy %.3f (%.3f to %.3f; the paper's %.3f), sensitivity %.3f (%.3f to %.3f), "
          "specificity %.3f (%.3f to %.3f); a reader that never calls a slip scores %.3f"
          % (acc, *np.percentile(accs, [2.5, 97.5]), PAPER_ACCURACY, sens, *np.percentile(senss, [2.5, 97.5]),
             spec, *np.percentile(specs, [2.5, 97.5]), 1 - n_pos / len(pid)))
    kept = bool(dflo >= -MARGIN)
    print("Verdict (slips, not worse than drawn corners, margin %.2f): %s" % (MARGIN, "KEPT" if kept else "NOT KEPT"))
    ver = verified_report(pid, label, level_label, score, call, np.abs(sl))
    write_row(a, t0, fold, pid, n_pos,
              {**ver, "auroc": (auc, ""), "auroc_low": (lo, ""), "auroc_high": (hi, ""), "auroc_drawn": (dauc, ""),
               "auroc_diff": (d, ""), "auroc_diff_low": (dflo, ""), "auroc_diff_high": (dfhi, ""),
               "auroc_written": (auroc(written, label), ""), "accuracy": (acc, ""), "sensitivity": (sens, ""),
               "specificity": (spec, "")}, kept)


def write_row(a, t0, fold, pid, n_pos, numbers, kept):
    numbers.update({"patients": (len(pid), "count"), "patients_with_slip": (n_pos, "count")})
    settings = {"source": "corners v%d" % a.version,
                "folds": ",".join(map(str, sorted(set(fold.tolist())))), "score": "amended"}
    numbers["kept"] = (float(kept), "")
    settings.update({"verdict": "KEPT" if kept else "NOT KEPT",
                     "bar": "AUROC not worse than drawn corners, margin %.2f" % MARGIN})
    name = "spine xray slips: corners v%d out of fold, any class 3 or 4 per patient" % a.version
    ledger.write("experiment", name, time.time() - t0, numbers, settings=settings,
                 note="modalities/spine_xray/eval/slip.py; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


if __name__ == "__main__":
    main()
