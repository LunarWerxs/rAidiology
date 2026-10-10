"""Slipped vertebrae, v3: the call learned from the corners' geometry instead of one hand-made number.

Dataset: Klinwichit P, et al. BUU-LSPINE (LICENSING.md, BUU-LSPINE row). Plan and bar: docs/efficiency/README.md,
"Slips v3: the call learned from the corners".

Why: slip.py's amended score (largest |slip - level's usual slip|) reads each level well on the refined corners
(0.84 to 0.88 at the three lowest) but the patient's largest level collects every level's false alarms, so the
patient AUROC is 0.7946 against the drawn corners' 0.8706. Here one logistic regression sees more of the
geometry at each level and is trained on our own refined out-of-fold corners, so it learns their noise.

What: one row per (patient, level); the label is BUU's own mark on that level (class 3 or 4). Features per level
(Dp, Da the lower vertebra's upper endplate back and front corner, Up, Ua the upper vertebra's lower endplate's;
L = |Da - Dp|, u = (Da - Dp) / L; buu.py says which end is front): back slip dot(Up - Dp, Da - Dp) / L^2 (slip.py's),
front slip dot(Ua - Da, Da - Dp) / L^2, endplate angle in degrees (signed, lower endplate's direction to the upper's),
front disc height |Ua - Da| / L, back disc height |Up - Dp| / L, endplate length ratio |Ua - Up| / L. Added per
fold: the amended back slip (minus that level's median over the TRAINING folds' patients) and a one-hot of the
level. StandardScaler + LogisticRegression(C=1, class_weight='balanced'). Patient score = largest level probability.
For held-out fold k the model is fit on the other four folds' refined corners; the threshold is Youden's J on
inner out-of-fold scores (fit on three of those four, score the fourth).

Bar (as slips v1 and v2): KEPT when learned AUROC minus the drawn corners' amended AUROC (0.8706, same 2,000
patients) has a paired patient-bootstrap 95% interval (1000 resamples, seed 0) with its low end >= -0.03. Printed
beside it: AUROC against corners v2's hand rule and v1's 0.8093, accuracy, sensitivity and specificity against the
paper's 0.923, per-level AUROC. Reported, not a gate: the same procedure fit and judged on the drawn corners.

Slips v4 (--features v4, the default; --features v3 is the run above, unchanged): v3's signed amended back slip put
the marked level on the wrong side at L2/L3, where BUU's marks are mostly backward slips, so the model ranked the
largest forward slips and not the largest slips. v4 adds the amended front slip (minus its level's training median)
and the absolute values of both amended slips, so size counts whichever way the vertebra moved. Bar unchanged.

Usage: python slip_learned.py --corners-version 2 [--features v3|v4] [--folds all]
       python slip_learned.py --selftest        no data files
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
import slip  # noqa: E402

V1_AUROC = 0.8093            # slip.py on corners v1, out of fold, all 2,000 patients (corners_refine.V1_AUROC)
BASE = ["back slip", "front slip", "angle deg", "front height", "back height", "length ratio"]
N_LEVELS = 5


def features(pts):
    """(N, 22, 2) corners -> (N, 5, 6) base features, columns as BASE."""
    pts = np.asarray(pts, np.float64)
    out = np.zeros((len(pts), N_LEVELS, len(BASE)))
    out[:, :, 0] = slip.slips(pts)
    for i in range(N_LEVELS):
        ua, up = pts[:, 2 * (2 * i + 1)], pts[:, 2 * (2 * i + 1) + 1]
        da, dp = pts[:, 2 * (2 * i + 2)], pts[:, 2 * (2 * i + 2) + 1]
        d, u = da - dp, ua - up
        length = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
        out[:, i, 1] = ((ua - da) * d).sum(-1) / length ** 2
        out[:, i, 2] = np.degrees(np.arctan2(d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0], (d * u).sum(-1)))
        out[:, i, 3] = np.linalg.norm(ua - da, axis=-1) / length
        out[:, i, 4] = np.linalg.norm(up - dp, axis=-1) / length
        out[:, i, 5] = np.linalg.norm(u, axis=-1) / length
    return out


def level_medians(base):
    """(2, 5): each level's median back slip (row 0) and front slip (row 1) over these (training) patients only."""
    return np.median(base[:, :, :2], axis=0).T


def design(base, med, features="v4"):
    """(n, 5, 6) base + (2, 5) training medians -> (n * 5, 12 or 15) rows.

    v3: base, amended back slip, level one-hot. v4: base, amended back slip, amended front slip, |amended back slip|,
    |amended front slip|, level one-hot."""
    n = len(base)
    back = base[:, :, 0] - med[0][None, :]
    onehot = np.broadcast_to(np.eye(N_LEVELS)[None], (n, N_LEVELS, N_LEVELS))
    if features == "v3":
        cols = [base, back[:, :, None], onehot]
    elif features == "v4":
        front = base[:, :, 1] - med[1][None, :]
        cols = [base, back[:, :, None], front[:, :, None], np.abs(back)[:, :, None], np.abs(front)[:, :, None], onehot]
    else:
        raise ValueError("features must be v3 or v4, got %r" % (features,))
    return np.concatenate(cols, axis=2).reshape(n * N_LEVELS, -1)


def fit(base, level_label, features="v4"):
    """Fit on these patients; returns a function base -> (n, 5) level probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    med = level_medians(base)
    model = Pipeline([("scale", StandardScaler()),
                      ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))])
    model.fit(design(base, med, features), np.asarray(level_label).reshape(-1))
    return lambda b: model.predict_proba(design(b, med, features))[:, 1].reshape(len(b), N_LEVELS)


def learned(base, level_label, label, fold, features="v4"):
    """Out-of-fold level probabilities (n, 5) and the call at each fold's inner Youden cut."""
    prob = np.zeros((len(base), N_LEVELS))
    call = np.zeros(len(base), bool)
    cuts = {}
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        prob[te] = fit(base[tr], level_label[tr], features)(base[te])
        inner = np.zeros(int(tr.sum()))                       # inner out-of-fold patient scores
        ftr, btr, ltr = fold[tr], base[tr], level_label[tr]
        for j in sorted(set(ftr.tolist())):
            inner[ftr == j] = fit(btr[ftr != j], ltr[ftr != j], features)(btr[ftr == j]).max(axis=1)
        cuts[k] = slip.youden(inner, label[tr])
        call[te] = prob[te].max(axis=1) >= cuts[k]
    return prob, call, cuts


def selftest():
    n = 11
    pts = np.zeros((n, 22, 2))
    for r in range(11):                                       # line r at y = 100 r, front x = 0, back x = 100
        pts[:, 2 * r] = (0.0, 100.0 * r)
        pts[:, 2 * r + 1] = (100.0, 100.0 * r)
    pts[0, 3] = (75.0, 100.0)                                 # level 0: Up 25 forward of Dp
    pts[0, 2] = (-10.0, 100.0)                                # level 0: Ua 10 in front of Da
    pts[1, 7] = (120.0, 300.0)                                # level 1: Up 20 behind Dp
    th = np.radians(10.0)                                     # level 2: upper endplate turned 10 degrees
    pts[2, 10] = pts[2, 11] + 100.0 * np.array([-np.cos(th), -np.sin(th)])
    f = features(pts)
    assert abs(f[0, 0, 0] - 0.25) < 1e-9 and abs(f[1, 1, 0] + 0.2) < 1e-9, "back slip sign/value"
    assert abs(f[0, 0, 1] - 0.10) < 1e-9, "front slip"
    assert abs(f[2, 2, 2] - 10.0) < 1e-9 and abs(f[3, 2, 2]) < 1e-9, "angle"
    assert abs(f[3, 0, 3] - 1.0) < 1e-9 and abs(f[3, 0, 4] - 1.0) < 1e-9 and abs(f[3, 0, 5] - 1.0) < 1e-9
    assert np.allclose(f[:, :, 0], slip.slips(pts)), "back slip is slip.slips"
    med = level_medians(f[3:])                                # training rows: patients 3.. only
    assert np.allclose(med, 0.0) and level_medians(f)[0, 0] == 0.0
    base = f.copy()
    base[:3, 0, 0] = 5.0                                      # test rows move; training medians must not
    x = design(base, med, "v3")
    assert x.shape[1] == 12
    assert np.allclose(x[:, 6].reshape(n, 5)[3:], base[3:, :, 0] - med[0]), "amended from training medians"
    assert np.allclose(x[:3 * 5, 6].reshape(3, 5)[:, 0], 5.0), "test rows unshifted by their own values"
    assert np.allclose(x[:, 7:], np.tile(np.eye(5), (n, 1))), "one-hot"
    base[:, :, 1] = np.arange(n * 5).reshape(n, 5) / 7.0      # v4: front slip amended and made absolute, training medians only
    med4 = level_medians(base[3:])
    assert med4.shape == (2, 5) and np.allclose(med4[1], np.median(base[3:, :, 1], axis=0))
    x4 = design(base, med4, "v4")
    assert x4.shape == (n * 5, 15), "v4 has 15 columns"
    front = base[:, :, 1] - med4[1][None, :]
    back = base[:, :, 0] - med4[0][None, :]
    assert np.allclose(x4[:, :6], base.reshape(n * 5, 6)) and np.allclose(x4[:, 10:], np.tile(np.eye(5), (n, 1)))
    assert np.allclose(x4[:, 6].reshape(n, 5), back)
    assert np.allclose(x4[:, 7].reshape(n, 5), front), "amended front slip from training medians"
    assert np.allclose(x4[:, 8].reshape(n, 5), np.abs(back)) and np.allclose(x4[:, 9].reshape(n, 5), np.abs(front))
    rng = np.random.default_rng(0)
    tb = rng.normal(size=(40, 5, 6))
    tl = np.zeros((40, 5), int)
    tl[:10, 2] = 1
    tb[:10, 2, 0] += 4.0
    p = fit(tb, tl)(tb)
    assert p.shape == (40, 5) and p[:10, 2].mean() > p[10:, 2].mean(), "model learned the marked level"
    assert (p.argmax(axis=1)[:10] == 2).mean() > 0.5, "the patient score is the largest level probability"
    nb = rng.normal(size=(40, 5, 6))                          # v4: a large NEGATIVE amended back slip marks the level
    nl = np.zeros((40, 5), int)
    nl[:10, 2] = 1
    nb[:10, 2, 0] -= 4.0
    q = fit(nb, nl, "v4")(nb)
    assert q[:10, 2].mean() > q[10:, 2].mean(), "v4 ranks a backward slip above unmarked rows"
    assert slip.auroc(q[:, 2], nl[:, 2]) > 0.9, "v4 backward slips rank high"
    print("selftest ok")


def boot_stats(score, label, idx):
    aucs = np.array([slip.auroc(score[r], label[r]) for r in idx])
    return slip.auroc(score, label), aucs


def main():
    t0 = time.time()
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corners-version", type=int, default=2, help="corners.py version whose LA out-of-fold corners to read")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--features", choices=["v3", "v4"], default="v4", help="v4 adds the front slip and both absolute slips")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    import buu
    import corners
    folds = corners.parse_folds(a.folds)
    if len(folds) < 3:
        raise SystemExit("the inner out-of-fold threshold needs at least three folds; got %s" % folds)
    r = corners.load_oof(slip.VIEW, a.corners_version, folds)
    ids, truth_all, classes = buu.annotations(slip.VIEW)
    fold_all = buu.folds_of(ids)
    label_all = buu.slip_label(classes)
    level_all = np.isin(classes[:, 1:10:2], buu.SLIP_CLASSES).astype(int)
    where = {int(i): n for n, i in enumerate(ids)}
    keep = np.array(sorted(where[int(i)] for i in r["ids"]))
    order = {int(i): n for n, i in enumerate(r["ids"])}
    ref = r["pred"][[order[int(i)] for i in ids[keep]]].astype(np.float64)
    truth = truth_all[keep].astype(np.float64)
    pid, fold, label, level_label = ids[keep], fold_all[keep], label_all[keep], level_all[keep]
    n, n_pos = len(pid), int(label.sum())
    base_ref = features(ref)
    print("slips %s (learned): %d patients in folds %s, %d with a side-view slip (class 3 or 4); corners v%d out of fold"
          % (a.features, n, ",".join(map(str, folds)), n_pos, a.corners_version))
    prob, call, cuts = learned(base_ref, level_label, label, fold, a.features)
    score = prob.max(axis=1)
    dscore = slip.amended(slip.slips(truth), fold)            # drawn corners, hand rule
    hand = slip.amended(slip.slips(ref), fold)                # corners v2, hand rule
    d_prob, _, _ = learned(features(truth), level_label, label, fold, a.features)
    ceiling = d_prob.max(axis=1)
    idx = np.random.default_rng(slip.BOOT_SEED).integers(0, n, (slip.BOOT, n))
    auc, aucs = boot_stats(score, label, idx)
    dauc, daucs = boot_stats(dscore, label, idx)
    hauc, haucs = boot_stats(hand, label, idx)
    cauc, caucs = boot_stats(ceiling, label, idx)
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    dlo, dhi = np.nanpercentile(daucs, [2.5, 97.5])
    clo, chi = np.nanpercentile(caucs, [2.5, 97.5])
    dflo, dfhi = np.nanpercentile(aucs - daucs, [2.5, 97.5])   # same resample for both
    hflo, hfhi = np.nanpercentile(aucs - haucs, [2.5, 97.5])
    print("Learned, corners v%d: AUROC %.4f (patient-bootstrap 95%% CI %.4f to %.4f, %d resamples, seed %d)"
          % (a.corners_version, auc, lo, hi, slip.BOOT, slip.BOOT_SEED))
    print("Drawn corners, amended score: AUROC %.4f (%.4f to %.4f)" % (dauc, dlo, dhi))
    print("Learned minus drawn corners: %+.4f (paired patient-bootstrap 95%% CI %+.4f to %+.4f)" % (auc - dauc, dflo, dfhi))
    kept = bool(dflo >= -slip.MARGIN)
    print("Verdict (slips %s, learned, not worse than drawn corners, margin %.2f): %s" % (a.features, slip.MARGIN, "KEPT" if kept else "NOT KEPT"))
    print("against corners v%d's hand rule (%.4f, computed here): %+.4f (paired CI %+.4f to %+.4f); against slips v1's %.4f: %+.4f"
          % (a.corners_version, hauc, auc - hauc, hflo, hfhi, V1_AUROC, auc - V1_AUROC))
    accs, senss, specs = [], [], []
    for rr in idx:
        ac, se, sp = slip.rates(call[rr], label[rr])
        accs.append(ac), senss.append(se), specs.append(sp)
    acc, sens, spec = slip.rates(call, label)
    print("thresholds chosen on inner out-of-fold scores: %s" % ", ".join("fold %d %.3f" % (k, c) for k, c in cuts.items()))
    print("at those thresholds: accuracy %.3f (%.3f to %.3f; the paper's %.3f), sensitivity %.3f (%.3f to %.3f), "
          "specificity %.3f (%.3f to %.3f); a reader that never calls a slip scores %.3f"
          % (acc, *np.percentile(accs, [2.5, 97.5]), slip.PAPER_ACCURACY, sens, *np.percentile(senss, [2.5, 97.5]),
             spec, *np.percentile(specs, [2.5, 97.5]), 1 - n_pos / n))
    print("per level (level probability against the level's own marks):")
    for i, name in enumerate(slip.LEVELS):
        pos = int(level_label[:, i].sum())
        lv = slip.auroc(prob[:, i], level_label[:, i]) if pos >= 5 else float("nan")
        print("  %-6s marks %3d  AUROC %s" % (name, pos, "%.3f" % lv if lv == lv else "n/a"))
    print("Ceiling, reported, not a gate (same procedure fit and judged on the drawn corners): AUROC %.4f (%.4f to %.4f)"
          % (cauc, clo, chi))
    ver = slip.verified_report(pid, label, level_label, score, call, prob)
    numbers = {**ver, "auroc": (auc, ""), "auroc_low": (lo, ""), "auroc_high": (hi, ""), "auroc_drawn": (dauc, ""),
               "auroc_diff": (auc - dauc, ""), "auroc_diff_low": (dflo, ""), "auroc_diff_high": (dfhi, ""),
               "auroc_hand_rule": (hauc, ""), "auroc_ceiling_drawn": (cauc, ""), "accuracy": (acc, ""),
               "sensitivity": (sens, ""), "specificity": (spec, ""), "patients": (n, "count"),
               "patients_with_slip": (n_pos, "count"), "kept": (float(kept), "")}
    settings = {"source": "corners v%d" % a.corners_version, "folds": ",".join(map(str, folds)),
                "features": a.features, "model": "logistic regression C=1 balanced, one row per patient and level",
                "verdict": "KEPT" if kept else "NOT KEPT",
                "bar": "AUROC not worse than drawn corners, margin %.2f" % slip.MARGIN}
    name = ("spine xray slips v4: learned from corners v%d, absolute slips" if a.features == "v4"
            else "spine xray slips v3: learned from corners v%d") % a.corners_version
    ledger.write("experiment", name, time.time() - t0,
                 numbers, settings=settings,
                 note="modalities/spine_xray/eval/slip_learned.py; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


if __name__ == "__main__":
    main()
