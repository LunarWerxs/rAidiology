"""The baselines a fracture detector has to beat, fitted on development folds only, out of fold.

Three methods, the names fixed by METRICS_SPEC.md, each fitted separately for each dataset (a wrist
model has no business predicting a hip):

- `prevalence`, the floor. The score is the fitting folds' fracture rate, the same number on every
  row; the prediction is 1 when that rate is at least 0.5, else 0. It is what guessing scores.
- `meta`, the NO-IMAGE floor: a logistic regression on facts a person knows without looking at the
  picture. GRAZPEDWRI-DX: age as a number, plus one-hot sex, laterality, projection and device
  manufacturer. FracAtlas publishes no age or sex, so there it is one-hot body part and view only.
  Deliberately NOT used: cast, metal, osteopenia, hardware, multiscan, fracture count and AO
  classification. Those are read off the image, or are the answer itself (a cast almost always means
  a fracture somebody already found), and using them would be cheating: a detector that merely beats
  a baseline holding the answer proves nothing. A detector that cannot beat `meta` has learned
  nothing from the pixels.
- `features`, the simple-measurement baseline, the X-ray analogue of the spine's measurement rules.
  Classical image statistics only, no deep learning and no pretrained weights, from the cached
  384 px array: over a fixed 6x6 grid of cells, each cell's mean and standard deviation and the mean
  and standard deviation of its Sobel gradient magnitude; plus a 16-bin global intensity histogram,
  an 8-bin gradient-orientation histogram weighted by magnitude, the share of non-zero pixels, and
  the source image's aspect ratio and (log) pixel count from prepare's json. The grid is fixed so a
  cell means the same part of the frame on every image; the histograms are shares, so they do not
  depend on image size. The features are standardised and fed to sklearn LogisticRegression
  (max_iter=2000, class_weight='balanced', random_state=datasets.SEED). The feature matrix holds
  pixels only (no labels, no split), so it is cached for every image of a dataset at once under
  preds.WORK/baselines/ and a re-run takes seconds; --rebuild-features recomputes it.

Protocol, and why:
- Development rows are OUT OF FOLD: fold k is predicted by a model fitted on the other four folds, so
  no image is scored by a model that saw it, or saw its patient (the split is by patient).
- The threshold is Youden's J (sensitivity + specificity - 1) maximised on the fitting folds' own
  rows, never on the fold being predicted, so the reported sensitivity and specificity are honest.
- Test rows use one model and one threshold fitted on all five development folds. The dev run saves
  them under preds.WORK/baselines/; the test run loads them and refuses to start if they are absent,
  so the test split can never be used to tune anything. datasets.guard_test(split) is called first.
- Every image of the split gets a row, in scope unless prepare marks it unreadable, in which case
  in_scope is 0, drop_reason is preds.UNREADABLE and the prediction and score are empty. Unreadable
  images are never used to fit.
- Deterministic: nothing here is random except through datasets.SEED; the thread pool only decides
  the order work finishes in, and each thread writes its own row.
- Weights: none pretrained, nothing downloaded. Commercial-clean by construction.

Usage: python baselines.py <dev|test> [--rebuild-features]   (test needs dev run first)"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402,F401
from core import ledger  # noqa: E402
import datasets  # noqa: E402
import preds  # noqa: E402
import prepare  # noqa: E402
import split as splitmod  # noqa: E402

METHODS = ["prevalence", "meta", "features"]
FOLDS = 5
GRID = 6
INT_BINS = 16
ORI_BINS = 8
JOBS = min(8, os.cpu_count() or 1)
PROGRESS = 2000
BASE = os.path.join(preds.WORK, "baselines")

# The facts a person knows without looking at the picture. Nothing else may be added here.
META_CATEGORICAL = {"grazpedwri": ["sex", "laterality", "projection", "device_manufacturer"],
                    "fracatlas": ["body_part", "view"]}
META_NUMERIC = {"grazpedwri": ["age"], "fracatlas": []}


# ---------------------------------------------------------------- the classical image features

def image_features(img, w, h):
    """One image (size x size uint8, letterboxed) to a fixed-length vector of classical statistics."""
    size = img.shape[0]
    f = img.astype(np.float32) / 255.0
    gx = ndimage.sobel(f, axis=1)
    gy = ndimage.sobel(f, axis=0)
    mag = np.hypot(gx, gy)
    c = size // GRID
    cut = c * GRID

    def cells(a):
        r = a[:cut, :cut].reshape(GRID, c, GRID, c)
        return r.mean(axis=(1, 3)).ravel(), r.std(axis=(1, 3)).ravel()

    fm, fs = cells(f)
    gm, gs = cells(mag)
    hist = np.bincount((img >> 4).ravel(), minlength=INT_BINS)[:INT_BINS].astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.minimum((ang / np.pi * ORI_BINS).astype(np.int64), ORI_BINS - 1)
    ori = np.bincount(bins.ravel(), weights=mag.ravel(), minlength=ORI_BINS)[:ORI_BINS]
    ori = ori / max(ori.sum(), 1e-9)
    nonzero = float(np.count_nonzero(img)) / img.size
    shape = [w / float(h), np.log(float(w) * float(h))]
    return np.concatenate([fm, fs, gm, gs, hist, ori, [nonzero], shape]).astype(np.float32)


def feature_paths(dataset):
    base = os.path.join(BASE, "features_%s_%d" % (dataset, prepare.SIZE))
    return base + ".npy", base + ".json"


def feature_matrix(dataset, rebuild=False):
    """(N, n_features) float32 in datasets.index order; an unreadable image's row is zeros."""
    t0 = time.time()
    arr, rowof, size = prepare.load(dataset)
    keys = list(rowof)
    npy, js = feature_paths(dataset)
    if not rebuild and os.path.exists(npy) and os.path.exists(js):
        try:
            meta = json.load(open(js, encoding="utf-8"))
            if meta.get("images") == keys and int(meta.get("size", -1)) == size:
                X = np.load(npy)
                if X.shape[0] == len(keys):
                    print("%s: feature cache reused (%d x %d), %.1f s" % (dataset, X.shape[0], X.shape[1],
                                                                          time.time() - t0), flush=True)
                    return X
        except (OSError, ValueError):
            pass
    os.makedirs(BASE, exist_ok=True)
    shapes = json.load(open(prepare.cache_paths(dataset, size)[1], encoding="utf-8"))["source_shape"]
    bad = prepare.unreadable(dataset, size)
    todo = [k for k in keys if k not in bad]
    first = image_features(np.asarray(arr[rowof[todo[0]]]), *shapes[todo[0]])
    X = np.zeros((len(keys), len(first)), np.float32)
    print("%s: measuring %d images with %d threads" % (dataset, len(todo), JOBS), flush=True)

    def one(k):
        i = rowof[k]
        X[i] = image_features(np.asarray(arr[i]), *shapes[k])

    done = 0
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        for _ in pool.map(one, todo):
            done += 1
            if done % PROGRESS == 0 or done == len(todo):
                print("  %d/%d  %.0f s" % (done, len(todo), time.time() - t0), flush=True)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    with open(npy + ".part", "wb") as fh:
        np.save(fh, X)
    os.replace(npy + ".part", npy)
    with open(js + ".part", "w", encoding="utf-8") as fh:        # the json last: a cut-short run is not "built"
        json.dump({"dataset": dataset, "size": size, "images": keys, "n_features": int(X.shape[1])}, fh)
    os.replace(js + ".part", js)
    print("%s: features %d x %d, wall %.1f s" % (dataset, X.shape[0], X.shape[1], time.time() - t0), flush=True)
    return X


# ---------------------------------------------------------------- the no-image facts

def meta_matrix(dataset):
    """(N, k) float design matrix in datasets.index order, from the pre-image facts only.

    Categories are read from the whole index, which is fine: they are values of a fact, not labels, so
    nothing about any outcome leaks. A column the index lacks is an error, never silently dropped."""
    df = datasets.index(dataset)
    need = META_CATEGORICAL[dataset] + META_NUMERIC[dataset]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit("%s: the index has no column %s, so `meta` cannot be built" % (dataset, missing))
    parts = []
    for c in META_NUMERIC[dataset]:
        v = pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")
        parts.append(v.fillna(v.median() if v.notna().any() else 0.0).astype(float).to_frame(c))
    for c in META_CATEGORICAL[dataset]:
        parts.append(pd.get_dummies(df[c].str.strip(), prefix=c, dtype=float))
    return pd.concat(parts, axis=1).to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- fitting

def youden_threshold(score, truth):
    """The score at which sensitivity + specificity - 1 is largest; predict 1 when score >= it."""
    truth = np.asarray(truth, int)
    if truth.min() == truth.max():
        return 0.5
    fpr, tpr, thr = roc_curve(truth, score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def fit_one(method, X, y):
    """A fitted model dict {kind, model, thr} on rows X (ignored by `prevalence`) and truth y."""
    y = np.asarray(y, int)
    if method == "prevalence":
        rate = float(y.mean())
        return {"kind": "prevalence", "rate": rate, "thr": 0.5}
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=datasets.SEED))
    clf.fit(X, y)
    thr = youden_threshold(clf.predict_proba(X)[:, 1], y)
    return {"kind": "logistic", "model": clf, "thr": thr}


def score_with(m, X):
    if m["kind"] == "prevalence":
        return np.full(len(X), m["rate"])
    return m["model"].predict_proba(X)[:, 1]


def predict_with(m, s):
    if m["kind"] == "prevalence":
        return (s >= 0.5).astype(int)
    return (s >= m["thr"]).astype(int)


def model_path(method, dataset):
    return os.path.join(BASE, "model_%s_%s.pkl" % (method, dataset))


def rows_for(fr, keep, score, pred, with_fold):
    """Prediction rows for every image of the frame; `keep` marks readable images."""
    out = []
    for i, r in enumerate(fr.itertuples(index=False)):
        d = {"patient": r.patient, "image": r.image, "fold": r.fold if with_fold else "",
             "in_scope": 1 if keep[i] else 0, "drop_reason": "" if keep[i] else preds.UNREADABLE,
             "fracture_pred": "", "fracture_score": ""}
        if keep[i]:
            d["fracture_pred"] = int(pred[i])
            d["fracture_score"] = repr(float(score[i]))
        out.append(d)
    return out


def run(split, rebuild_features):
    """Every method on every dataset for one split. The ledger entry is written even on a crash,
    so a run that died half way still leaves its time and what it had got through."""
    with ledger.run("baselines", split) as led:
        _run(split, rebuild_features, led)


def _run(split, rebuild_features, led):
    datasets.guard_test(split)
    t_all = time.time()
    if split == "test":
        for d in datasets.NAMES:
            for m in METHODS:
                if not os.path.exists(model_path(m, d)):
                    raise SystemExit("no fitted baseline at %s; run `python baselines.py dev` first"
                                     % model_path(m, d))
    for d in datasets.NAMES:
        t_d = time.time()
        fr = splitmod.frame(d, split)
        _, rowof, _ = prepare.load(d)
        bad = prepare.unreadable(d)
        idx = np.array([rowof[str(k)] for k in fr.image])
        keep = np.array([str(k) not in bad for k in fr.image])
        truth = fr.truth.to_numpy(int)
        mats = {"meta": meta_matrix(d)[idx]}
        if "features" in METHODS:
            mats["features"] = feature_matrix(d, rebuild_features)[idx]
        mats["prevalence"] = np.zeros((len(fr), 1))
        for m in METHODS:
            t0 = time.time()
            X = mats[m]
            score = np.zeros(len(fr))
            pred = np.zeros(len(fr), int)
            if split == "dev":
                fold = fr.fold.to_numpy(int)
                thrs = []
                for k in range(FOLDS):
                    fit = keep & (fold != k)
                    tgt = keep & (fold == k)
                    if not fit.any() or not tgt.any():
                        continue
                    model = fit_one(m, X[fit], truth[fit])
                    s = score_with(model, X[tgt])
                    score[tgt], pred[tgt] = s, predict_with(model, s)
                    thrs.append(model["thr"])
                final = fit_one(m, X[keep], truth[keep])      # the one the test run will load
                os.makedirs(BASE, exist_ok=True)
                with open(model_path(m, d), "wb") as fh:
                    pickle.dump(final, fh)
                thr_text = "fold thresholds %s, all-dev %.4f" % ([round(t, 4) for t in thrs], final["thr"])
            else:
                with open(model_path(m, d), "rb") as fh:
                    model = pickle.load(fh)
                s = score_with(model, X[keep])
                score[keep], pred[keep] = s, predict_with(model, s)
                thr_text = "threshold %.4f (fitted on all dev folds)" % model["thr"]
            preds.write(m, d, split, rows_for(fr, keep, score, pred, split == "dev"))
            led.number("%s.%s.rows" % (d, m), len(fr), "count")
            led.number("%s.%s.seconds" % (d, m), time.time() - t0, "seconds")
            print("%s / %s: %d rows, %s, wall %.1f s" % (m, d, len(fr), thr_text, time.time() - t0), flush=True)
        print("%s: stage wall %.1f s" % (d, time.time() - t_d), flush=True)
    preds.collect()
    print("baselines %s: total wall %.1f s" % (split, time.time() - t_all), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prevalence, no-image and classical-feature baselines.")
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--rebuild-features", action="store_true", help="measure every image again")
    a = ap.parse_args(argv)
    run(a.split, a.rebuild_features)


if __name__ == "__main__":
    main()
