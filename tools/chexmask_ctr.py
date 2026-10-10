"""Does the cardiothoracic ratio from CheXmask's outlines add to the chest reader's cardiomegaly score?

Cardiomegaly is the reader's weak finding against radiologists, and the cardiothoracic ratio (CTR, heart
width over chest width) is what a radiologist measures to call it. CheXmask has outlined the lungs and
heart of every NIH ChestX-ray8 picture with HybridGNet (a model, not NIH's labels and not the reader's
own), so the CTR it gives carries no label from NIH: nothing leaks into the test.

Per picture: heart width = max x - min x of the heart's landmarks; chest width = max x - min x over both
lungs' landmarks; CTR = heart / chest. A picture is usable when CheXmask's `Dice RCA (Mean)` is at least
--min-dice (its README advises 0.7); otherwise its CTR is NaN. The per-picture table (image, ctr,
dice_mean, usable, plus the --min-dice it was made with) is cached in
<work>/eval/chest/chexmask_ctr.csv and reused while it is newer than the CheXmask file and was made with
the same --min-dice. The landmarks are 120 points: right lung 44, left lung 50, heart 26 (HybridGNet's
graph, in CheXmask's order right lung, left lung, heart).

On the pictures of the chosen folds that reader version scored out of fold, cardiomegaly AUROC of
(a) the CTR alone, (b) the reader's cardiomegaly logit alone, (c) a logistic regression on [reader logit,
CTR, missing-CTR flag] (standardized inputs; every prediction out of fold, 5 splits grouped by patient).
A picture without a usable CTR gets the median usable CTR and a flag of 1. The differences (c) - (b) and
(a) - (b) come with 95% intervals from paired resamples of patients, the same way compare.py does them
(its Auroc class, its seed, one draw for all three scores).

Data: CheXmask (Gaggion et al.), CC BY 4.0. Cite: Gaggion N, Mosquera C, Mansilla L, et al. CheXmask: a
large-scale dataset of anatomical segmentation masks for multi-center chest x-ray images. Scientific Data
11, 511 (2024). https://doi.org/10.1038/s41597-024-03358-1

--ctr-file <csv> reads the CTR table (columns image, ctr, ...; e.g. tools/chest_outline.py's
ctr_fold0.csv, our own model's outlines) instead of CheXmask's, for every picture the file has; its ctr is
used as it stands (--min-dice does not apply), and the printed rows and the ledger line say "our CTR
(<file name>)" instead of CheXmask.

Usage: python tools/chexmask_ctr.py --version 102 [--min-dice 0.7] [--folds 0] [--resamples 1000] [--no-ledger]
                                    [--ctr-file work/eval/chest/outline-v1/ctr_fold0.csv]
       (--folds takes one fold or a comma list, e.g. 0,1)"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
from core import ledger, paths  # noqa: E402
import compare  # noqa: E402
import datasets  # noqa: E402

CSV = os.path.join(paths.DATASETS, "public", "chexmask", "ChestX-Ray8.csv")
CACHE = os.path.join(paths.WORK, "eval", "chest", "chexmask_ctr.csv")
# the NIH file names its picture column as NIH does, "Image Index" (the data dictionary's "Image ID" is the
# other files'; read from the file's header, 2026-10-03)
ID = "Image Index"
COLS = [ID, "Dice RCA (Mean)", "Landmarks"]
N_RIGHT, N_LEFT, N_HEART = 44, 50, 26
NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
N_POINTS = N_RIGHT + N_LEFT + N_HEART
FRAME = 1024.0              # NIH's pictures, and CheXmask's landmarks, are 1024 pixels square
LANDMARKS = os.path.join(paths.WORK, "eval", "chest", "chexmask_landmarks.npz")


def ctr_from_points(points):
    """Heart width over chest width from one picture's 120 (x, y) points (any scale), in CheXmask's order."""
    x = np.asarray(points, dtype=np.float64).reshape(-1, 2)[:, 0]
    heart = x[N_RIGHT + N_LEFT:]
    lungs = x[:N_RIGHT + N_LEFT]
    chest = lungs.max() - lungs.min()
    return float((heart.max() - heart.min()) / chest) if chest > 0 else np.nan


def ctr_of(landmarks):
    """Heart width over chest width from one picture's landmark string; NaN when it does not parse."""
    v = np.array(NUM.findall(landmarks), dtype=np.float64)
    if v.size != 2 * (N_RIGHT + N_LEFT + N_HEART):
        return np.nan
    return ctr_from_points(v)


# 2026-10-04: moved here from chest_outline.py so reader.py (grow_heart) can load them; chest_outline imports reader.
def build_landmarks():
    """image, landmarks / 1024 (N x 240, NaN where the string did not parse), dice for every picture in
    CheXmask's NIH file, read in chunks."""
    images, lms, dice = [], [], []
    for chunk in pd.read_csv(CSV, usecols=COLS, chunksize=20000):
        images.append(chunk[ID].astype(str).map(os.path.basename).to_numpy(str))
        dice.append(chunk["Dice RCA (Mean)"].to_numpy(np.float32))
        part = np.full((len(chunk), 2 * N_POINTS), np.nan, np.float32)
        for i, s in enumerate(chunk["Landmarks"]):
            v = NUM.findall(s)
            if len(v) == 2 * N_POINTS:
                part[i] = np.array(v, dtype=np.float64) / FRAME
        lms.append(part)
    return np.concatenate(images), np.concatenate(lms), np.concatenate(dice)


def load_landmarks():
    """(image names, N x 240 landmarks / 1024, dice mean): the cache when newer than CheXmask's file."""
    if os.path.exists(LANDMARKS) and os.path.getmtime(LANDMARKS) > os.path.getmtime(CSV):
        z = np.load(LANDMARKS, allow_pickle=False)
        print("landmarks: cached %s (%d pictures)" % (LANDMARKS, len(z["image"])), flush=True)
        return z["image"], z["landmarks"], z["dice"]
    if not os.path.exists(CSV):
        raise SystemExit("no CheXmask file at %s" % CSV)
    image, lm, dice = build_landmarks()
    os.makedirs(os.path.dirname(LANDMARKS), exist_ok=True)
    np.savez(LANDMARKS, image=image, landmarks=lm, dice=dice)
    print("landmarks: built from %s (%d pictures), cached in %s" % (CSV, len(image), LANDMARKS), flush=True)
    return image, lm, dice


def build_table(min_dice):
    """image, ctr, dice_mean, usable for every picture in CheXmask's NIH file, read in chunks."""
    parts = []
    for chunk in pd.read_csv(CSV, usecols=COLS, chunksize=20000):
        d = chunk["Dice RCA (Mean)"].to_numpy(float)
        ctr = np.array([ctr_of(s) for s in chunk["Landmarks"]])
        usable = (d >= min_dice) & np.isfinite(ctr)
        parts.append(pd.DataFrame({"image": chunk[ID].astype(str).map(os.path.basename),
                                   "ctr": np.where(usable, ctr, np.nan), "dice_mean": d,
                                   "usable": usable.astype(int), "min_dice": min_dice}))
    return pd.concat(parts, ignore_index=True)


def load_table(min_dice):
    """The cached table when it is newer than CheXmask's file and was made with this --min-dice."""
    if os.path.exists(CACHE) and os.path.getmtime(CACHE) > os.path.getmtime(CSV):
        t = pd.read_csv(CACHE)
        if (t.min_dice == min_dice).all():
            print("CTR table: cached %s (%d pictures)" % (CACHE, len(t)))
            return t
    if not os.path.exists(CSV):
        raise SystemExit("no CheXmask file at %s" % CSV)
    t = build_table(min_dice)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    t.to_csv(CACHE, index=False)
    print("CTR table: built from %s (%d pictures), cached in %s" % (CSV, len(t), CACHE))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True, help="the chest reader's version number")
    ap.add_argument("--min-dice", type=float, default=0.7)
    ap.add_argument("--folds", default="0", help="fold or comma list of folds to score")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--ctr-file", default=None, help="read the CTR table from this csv instead of CheXmask's")
    args = ap.parse_args()
    # what the CTR is called in every printed row and the ledger line, so the two sources cannot be confused
    ctr_name = "CTR" if args.ctr_file is None else "our CTR (%s)" % os.path.basename(args.ctr_file)
    t0 = time.time()
    folds = [int(f) for f in args.folds.split(",")]

    path = os.path.join(compare.run_dir(args.version), "oof_logits.npz")
    if not os.path.exists(path):
        raise SystemExit("no such file: %s" % path)
    z = np.load(path, allow_pickle=False)
    keep = np.isin(z["fold"], folds)
    image = z["image"].astype(str)[keep]
    patient = z["patient"][keep]
    col = datasets.LABELS.index("Cardiomegaly")
    y = (z["y"][keep][:, col] > 0).astype(int)
    logit = z["logit"][keep][:, col].astype(np.float64)

    if args.ctr_file is None:
        table = load_table(args.min_dice)
        usable_by = "CheXmask Dice RCA mean >= %.2f" % args.min_dice
    else:
        if not os.path.exists(args.ctr_file):
            raise SystemExit("no such file: %s" % args.ctr_file)
        table = pd.read_csv(args.ctr_file)
        usable_by = "a finite value in %s" % os.path.basename(args.ctr_file)
        print("CTR table: %s (%d pictures)" % (args.ctr_file, len(table)))
    ctr = pd.Series(table.set_index("image").ctr).reindex(image).to_numpy(float)
    have = np.isfinite(ctr)
    print("reader v%d, fold(s) %s: %d pictures, %d patients, %d with cardiomegaly; %d pictures (%.1f%%) have a "
          "usable CTR (%s)"
          % (args.version, args.folds, len(image), len(np.unique(patient)), y.sum(), have.sum(),
             100.0 * have.mean(), usable_by))
    if have.sum() < 100 or y[have].min() == y[have].max():
        raise SystemExit("too few pictures with a usable CTR to test anything")
    ctr = np.where(have, ctr, np.median(ctr[have]))
    flag = (~have).astype(np.float64)

    x = np.column_stack([logit, ctr, flag])
    oof = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups=patient):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        oof[te] = model.fit(x[tr], y[tr]).decision_function(x[te])

    scores = {"CTR alone": ctr, "reader's logit alone": logit, "reader + CTR (out of fold)": oof}
    fns = {k: compare.Auroc(y.astype(np.float64), v) for k, v in scores.items()}
    codes, uniq = pd.factorize(patient)
    n_pat = len(uniq)
    ones = np.ones(len(codes))
    full = {k: f(ones) for k, f in fns.items()}
    rng = np.random.default_rng(compare.SEED)
    d_c = np.empty(args.resamples)
    d_a = np.empty(args.resamples)
    for i in range(args.resamples):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)[codes].astype(np.float64)  # one draw for all
        b = fns["reader's logit alone"](w)
        d_c[i] = fns["reader + CTR (out of fold)"](w) - b
        d_a[i] = fns["CTR alone"](w) - b
    ci_c = np.nanpercentile(d_c, [2.5, 97.5])
    ci_a = np.nanpercentile(d_a, [2.5, 97.5])

    a, b, c = full["CTR alone"], full["reader's logit alone"], full["reader + CTR (out of fold)"]
    print("%d patients; %d paired resamples of patients\n" % (n_pat, args.resamples))
    print("| label | A | B | B - A (95% CI) |\n|---|---|---|---|")
    print("| Cardiomegaly: reader's logit (A) vs reader + %s (B) | %.4f | %.4f | %+.4f (%+.4f to %+.4f) |"
          % (ctr_name, b, c, c - b, ci_c[0], ci_c[1]))
    print("| Cardiomegaly: reader's logit (A) vs %s alone (B) | %.4f | %.4f | %+.4f (%+.4f to %+.4f) |"
          % (ctr_name, b, a, a - b, ci_a[0], ci_a[1]))
    verdict = "BETTER" if ci_c[0] > 0 else ("WORSE" if ci_c[1] < 0 else "INCONCLUSIVE")
    print("\nVerdict (cardiomegaly AUROC, %s added): %s" % (ctr_name, verdict))
    print("\n%s among the %d pictures with one (cardiomegaly / not): mean %.3f / %.3f; share above 0.5 %.1f%% / %.1f%%"
          % (ctr_name, have.sum(), ctr[have & (y == 1)].mean(), ctr[have & (y == 0)].mean(),
             100.0 * (ctr[have & (y == 1)] > 0.5).mean(), 100.0 * (ctr[have & (y == 0)] > 0.5).mean()))

    if not args.no_ledger:
        numbers = {"auroc.cardiomegaly_ctr": (a, ""), "auroc.cardiomegaly_reader": (b, ""),
                   "auroc.cardiomegaly_reader_plus_ctr": (c, ""),
                   "auroc.cardiomegaly_plus_ctr_minus_reader_low": (ci_c[0], ""),
                   "auroc.cardiomegaly_plus_ctr_minus_reader_high": (ci_c[1], ""),
                   "pictures": (len(image), "count"), "pictures_with_ctr": (int(have.sum()), "count")}
        settings = {"version": args.version, "min_dice": args.min_dice, "folds": args.folds,
                    "resamples": args.resamples, "seed": compare.SEED}
        if args.ctr_file is None:
            title = "chest: CTR from CheXmask added to reader v%d's cardiomegaly (fold %s)" % (args.version, args.folds)
        else:
            title = "chest: %s added to reader v%d's cardiomegaly (fold %s)" % (ctr_name, args.version, args.folds)
            settings["ctr_file"] = args.ctr_file
        ledger.write("experiment", title, time.time() - t0, numbers, settings=settings, note=verdict)


if __name__ == "__main__":
    main()
