"""The no-picture baselines a chest X-ray reader has to beat, scored out of fold on development
patients only.

A reader that cannot beat "how old is the patient, which sex, was it a bedside film, and how many
visits has the patient had" has learned nothing from the picture. This matters on NIH ChestX-ray14:
its bedside 'AP' films are taken on patients too ill to stand for a 'PA' film, so they are far
likelier to be abnormal, and the follow-up number says the same of a patient who keeps coming back.
A network can score well on those facts alone, so they are the floor.

Two methods, each fitted separately for each of the 15 labels (LABELS: the 14 findings and Abnormal):
- `prevalence`, no features. Every picture gets the training rows' rate of the label, so its AUROC
  is 0.5 by construction. It is what guessing scores, and it shows the scorer is honest.
- `age_sex_view`, a logistic regression (max_iter=2000) in a StandardScaler pipeline on four facts a
  person knows without looking at the picture: age clipped to 0..100 (NIH has a few impossible
  ages), sex (1 if 'M'), view (1 if 'AP', else 0 for 'PA') and log1p(follow_up).

Protocol, and why:
- Development rows only (split.frame('dev')); the test list is never opened. Fold k is predicted by
  a model fitted on the other four folds, so no picture is scored by a model that saw it or saw its
  patient (the split is by patient).
- AUROC per label is over all out-of-fold rows pooled. The 95% interval is a percentile bootstrap
  over PATIENTS (--boot resamples, default 1000, numpy default_rng(20261003)): patient ids are drawn
  with replacement and every row of a drawn patient comes along, because a patient's pictures are
  not independent. Each patient's row numbers are worked out once. The mean over the 14 findings
  (not Abnormal) has its interval from the same resamples.
- The out-of-fold scores are saved to <WORK>/eval/chest/baselines_oof.npz (image, patient, fold,
  y int8 [n, 15] in LABELS order, and one float32 [n, 15] array per method) so the reader's scores
  can be set beside them row for row. One ledger row per method, kind 'chest_baseline'.

Usage: python modalities/chest_xray/eval/baselines.py [--boot 1000]"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import ledger, paths  # noqa: E402
import datasets  # noqa: E402
import split as splitmod  # noqa: E402

METHODS = ["prevalence", "age_sex_view"]
FOLDS = 5
SEED = 20261003
OUT = os.path.join(paths.WORK, "eval", "chest", "baselines_oof.npz")


def features(df):
    """(n, 4) float64: age clipped to 0..100, sex 1 if M, view 1 if AP, log1p(follow_up)."""
    return np.column_stack([
        df.age.clip(0, 100).to_numpy(float),
        (df.sex == "M").to_numpy(float),
        (df.view == "AP").to_numpy(float),
        np.log1p(df.follow_up.to_numpy(float)),
    ])


def out_of_fold(method, X, y, fold):
    """(n, len(LABELS)) float32 scores; every row is scored by a model fitted on the other folds."""
    score = np.zeros(y.shape, np.float32)
    for k in range(FOLDS):
        fit, tgt = fold != k, fold == k
        for j in range(y.shape[1]):
            if method == "prevalence" or y[fit, j].min() == y[fit, j].max():
                score[tgt, j] = y[fit, j].mean()          # a label with one class to learn from is its rate
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[fit], y[fit, j])
            score[tgt, j] = clf.predict_proba(X[tgt])[:, 1]
    return score


def auroc(y, s):
    """AUROC, or nan when the rows hold only one class."""
    return float("nan") if y.min() == y.max() else float(roc_auc_score(y, s))


def bootstrap(y, score, patient, n_boot):
    """(n_boot, len(LABELS) + 1) AUROCs of patient resamples; the last column is the mean over the
    14 findings. A label with one class in a resample is nan there and left out of that mean."""
    ids, inverse = np.unique(patient, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    cuts = np.cumsum(np.bincount(inverse))[:-1]
    rows_of = np.split(order, cuts)                       # each patient's row numbers, once
    g = np.random.default_rng(SEED)
    n_labels = y.shape[1]
    out = np.full((n_boot, n_labels + 1), np.nan)
    for b in range(n_boot):
        pick = g.integers(0, len(ids), len(ids))
        rows = np.concatenate([rows_of[i] for i in pick])
        for j in range(n_labels):
            out[b, j] = auroc(y[rows, j], score[rows, j])
        out[b, n_labels] = np.nanmean(out[b, :len(datasets.FINDINGS)])
    return out


def summarise(y, score, patient, n_boot):
    """{'auroc': [15 + mean14], 'low': [...], 'high': [...]} over all out-of-fold rows."""
    point = [auroc(y[:, j], score[:, j]) for j in range(y.shape[1])]
    point.append(float(np.mean(point[:len(datasets.FINDINGS)])))
    boot = bootstrap(y, score, patient, n_boot)
    return {"auroc": np.array(point), "low": np.nanpercentile(boot, 2.5, axis=0),
            "high": np.nanpercentile(boot, 97.5, axis=0)}


def print_table(results):
    names = datasets.LABELS
    print("%-20s" % "label" + "".join("  %-26s" % m for m in METHODS))

    def cell(r, j):
        return "%.3f (%.3f to %.3f)" % (r["auroc"][j], r["low"][j], r["high"][j])

    for j, name in enumerate(names[:len(datasets.FINDINGS)]):
        print("%-20s" % name + "".join("  %-26s" % cell(results[m], j) for m in METHODS))
    print("%-20s" % "mean over findings" + "".join("  %-26s" % cell(results[m], len(names)) for m in METHODS))
    print("%-20s" % "Abnormal" + "".join("  %-26s" % cell(results[m], len(names) - 1) for m in METHODS))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prevalence and age/sex/view baselines for chest X-ray, out of fold.")
    ap.add_argument("--boot", type=int, default=1000, help="patient bootstrap resamples")
    a = ap.parse_args(argv)
    fr = splitmod.frame("dev")
    labels = datasets.LABELS
    y = fr[labels].to_numpy(np.int8)
    fold = fr.fold.to_numpy(int)
    patient = fr.patient.to_numpy(int)
    X = features(fr)
    print("%d development pictures, %d patients, %d resamples" % (len(fr), len(np.unique(patient)), a.boot),
          flush=True)
    scores, results, seconds = {}, {}, {}
    for m in METHODS:
        t0 = time.time()
        scores[m] = out_of_fold(m, X, y, fold)
        results[m] = summarise(y, scores[m], patient, a.boot)
        seconds[m] = time.time() - t0
        print("%s: wall %.1f s" % (m, seconds[m]), flush=True)
    print_table(results)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, image=fr.image.to_numpy(str), patient=patient, fold=fold, y=y, **scores)
    print(OUT)
    settings = {"boot": a.boot, "folds": FOLDS, "seed": SEED}
    for m in METHODS:
        numbers = {"auroc.%s" % n: (float(results[m]["auroc"][j]), "auroc") for j, n in enumerate(labels)}
        numbers["auroc.mean14"] = (float(results[m]["auroc"][len(labels)]), "auroc")
        ledger.write("chest_baseline", "chest baseline %s" % m, seconds[m], numbers,
                     note="out of fold, development patients only", settings=settings)


if __name__ == "__main__":
    main()
