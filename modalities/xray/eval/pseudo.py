"""Self-training for the fracture detector (idea 133, docs/efficiency/README.md): the current best ensemble
labels an unlabelled pool, and only the pictures it is sure of teach a student.

The teacher is v140's five fold nets, each of which never trained on a pool picture (a pool is in no
collection the detector trains on), so every teacher probability is a read the net did not learn from.
The five are averaged as the mean of their sigmoids. A pool's own labels are never read: BoneFract is
trained on only through these teacher labels. FracAtlas and GRAZPEDWRI-DX are never a pool; they are the
collections the detector is evaluated on.

How to run (from the repository root; both need the graphics card, so queue them, do not run them beside a training):

  1. The teacher labels the pool. v140's recipe is `--epochs 25 --set val_share=0 --set 'init="chest-v103"'`,
     seed 0 (docs/efficiency/README.md:1368-1369).
       python modalities/xray/eval/pseudo.py score --versions 140 --pool bonefract --out work/eval/xray/pseudo-v140.csv
  2. The student is v140's recipe, plus the confident pool pictures and the stronger augmentation on them.
       python modalities/xray/eval/detector.py train --version 147 --epochs 25 --set val_share=0 --set 'init="chest-v103"' --set 'pseudo_csv="work/eval/xray/pseudo-v140.csv"' --set strong_aug=true
     Then compare.py: A = v140's oof_dev.csv, B = v147's, with the bar's noise and pass line.

  python modalities/xray/eval/pseudo.py selftest    the confident-pick rule and the csv round trip, on made-up numbers (seconds, CPU)
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402
import prepare  # noqa: E402

FOLDS = 5
FOLD_COLUMNS = ["p_fold%d" % k for k in range(FOLDS)]    # fold k's net alone; it never trained on fold k
COLUMNS = ["picture", "dataset", "patient", "p_teacher"] + FOLD_COLUMNS   # p_teacher: their mean, for reporting only
LOW, HIGH = 0.1, 0.9          # detector.py's pseudo_low and pseudo_high defaults
# The one-hot a pool picture is read with: the adults' profile (score_images' own choice for a hip, hand or shoulder).
TEACHER_DATASET = "fracatlas"


def label_of(p, low, high):
    """The pseudo label of each teacher probability: 0 at or below `low`, 1 at or above `high`, NaN between (dropped)."""
    p = np.asarray(p, float)
    return np.where(p <= low, 0.0, np.where(p >= high, 1.0, np.nan))


def picked(frame, low, high, fold):
    """The rows of a pseudo-label frame fold `fold`'s net alone is sure of (p_fold{fold}), with their 0.0 or 1.0 label in `label`."""
    lab = label_of(frame[FOLD_COLUMNS[fold]].values, low, high)
    keep = ~np.isnan(lab)
    return frame[keep].assign(label=lab[keep])


def write(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame[COLUMNS].to_csv(path, index=False, float_format="%.17g")     # 17 digits: a float64 reads back exactly


def read(path):
    """The csv's rows; picture and patient stay text (a picture key may have leading zeros)."""
    df = pd.read_csv(path, dtype={"picture": str, "dataset": str, "patient": str}, float_precision="round_trip")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit("%s: not a pseudo-label csv, no column %s" % (path, ", ".join(missing)))
    return df


def check_pools(pools):
    """A pool is a collection from datasets.ALL that the detector is not scored on: never FracAtlas or GRAZPEDWRI-DX."""
    bad = [p for p in pools if p in datasets.NAMES or p not in datasets.ALL]
    if bad:
        raise SystemExit("pool %s: a pool is one of %s, and never one the detector is scored on (%s)"
                         % (", ".join(bad), ", ".join(sorted(set(datasets.ALL) - set(datasets.NAMES))),
                            ", ".join(datasets.NAMES)))
    return list(pools)


def pool_names(path, names=None):
    """The pools a pseudo-label csv names, sorted; refused when one is a collection the detector trains or is scored on."""
    names = datasets.NAMES if names is None else names
    pools = check_pools(sorted(set(read(path).dataset)))
    clash = [p for p in pools if p in names]
    if clash:
        raise SystemExit("pseudo_csv names %s, which the run already trains on; a pool must be another collection" % ", ".join(clash))
    return pools


def score(versions, pools, out):
    import torch
    import detector

    check_pools(pools)
    n = len(datasets.NAMES)
    teachers = []
    for v in versions:
        _meta, folds = detector.load_folds(v, n)
        teachers += folds
    print("teacher: %s, %d fold nets" % (", ".join("v%d" % v for v in versions), len(teachers)), flush=True)
    parts = []
    for pool in pools:
        idx = datasets.index(pool)
        arr, order, size = prepare.load(pool)
        bad = prepare.unreadable(pool)
        df = pd.DataFrame({"dataset": pool, "dataset_index": datasets.NAMES.index(TEACHER_DATASET),
                           "image": idx.image.values, "patient": idx.patient.values,
                           "truth": 0})            # feed() reads the column; the teacher never uses it
        df["cache_row"] = [order[i] for i in df.image]
        df["readable"] = [i not in bad for i in df.image]
        detector.number_rows(df)
        pix = detector.stack(df, {pool: (arr, order, size)})
        rows = df.index.values[df.readable.values]
        pf = np.zeros((FOLDS, len(rows)), np.float64)
        for i, (net, mean, sd) in enumerate(teachers):      # net i is fold i % 5 of its version
            pf[i % FOLDS] += detector.predict_rows(net, pix, df, rows, torch.tensor(mean, device=detector.DEV),
                                                   torch.tensor(sd, device=detector.DEV), n, detector.CFG["batch"])
        pf /= len(teachers) // FOLDS
        p = pf.mean(axis=0)
        del pix
        part = pd.DataFrame({"picture": df.image.values[rows], "dataset": pool,
                             "patient": df.patient.values[rows], "p_teacher": p,
                             **{c: pf[k] for k, c in enumerate(FOLD_COLUMNS)}})
        sure = (p <= LOW).sum() + (p >= HIGH).sum()
        print("%s: %d scored, %d unreadable left out; %d at or below %.2f, %d at or above %.2f (%.1f%% kept)"
              % (pool, len(part), int((~df.readable).sum()), (p <= LOW).sum(), LOW, (p >= HIGH).sum(), HIGH,
                 100.0 * sure / max(len(part), 1)), flush=True)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    write(frame, out)
    print("wrote %s: %d pictures" % (out, len(frame)), flush=True)


def _expect(ok, what):
    if not ok:
        raise SystemExit("selftest FAILED: %s" % what)
    print("ok  %s" % what, flush=True)


def selftest():
    """The confident-pick rule and the csv round trip, on made-up numbers; seconds, no data, no card."""
    lab = label_of([0.0, 0.1, 0.1000001, 0.5, 0.8999999, 0.9, 1.0], LOW, HIGH)
    _expect(np.array_equal(lab[[0, 1]], [0.0, 0.0]) and np.array_equal(lab[[5, 6]], [1.0, 1.0]),
            "at or below 0.1 is 0, at or above 0.9 is 1 (the edges are inclusive)")
    _expect(np.isnan(lab[[2, 3, 4]]).all(), "between 0.1 and 0.9 is dropped")

    frame = pd.DataFrame({"picture": ["Arm_patient0004_Negative_001", "0123", "x9"],
                          "dataset": ["bonefract", "bonefract", "bonefract"],
                          "patient": ["0004", "0123", "9"],
                          "p_teacher": [0.1 / 3, 0.97, 0.5],
                          **{c: [0.1 / 3, 0.97, 0.5] for c in FOLD_COLUMNS}})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pseudo.csv")
        write(frame, path)
        back = read(path)
    _expect(list(back.picture) == list(frame.picture) and list(back.patient) == ["0004", "0123", "9"],
            "csv round trip keeps picture and patient as text, leading zeros included")
    _expect(np.array_equal(back.p_teacher.values, frame.p_teacher.values),
            "csv round trip keeps every probability exactly")
    kept = picked(back, LOW, HIGH, 0)
    _expect(list(kept.picture) == ["Arm_patient0004_Negative_001", "0123"] and list(kept.label) == [0.0, 1.0],
            "the picks: 0.0333 a negative, 0.97 a positive, 0.5 dropped")
    # the folds disagree: fold k's picks read p_fold{k} only, never another fold's or the mean
    dis = pd.DataFrame({"picture": ["a", "b", "c"], "dataset": "bonefract", "patient": ["1", "2", "3"],
                        "p_teacher": [0.5, 0.5, 0.5]})
    for k, c in enumerate(FOLD_COLUMNS):
        dis[c] = 0.5
    dis.loc[0, "p_fold1"] = 0.99     # a is positive for fold 1 alone
    dis.loc[1, "p_fold2"] = 0.01     # b is negative for fold 2 alone
    dis.loc[2, "p_fold3"] = 0.95     # c is positive for fold 3 alone
    got = {k: (list(picked(dis, LOW, HIGH, k).picture), list(picked(dis, LOW, HIGH, k).label)) for k in range(FOLDS)}
    _expect(got == {0: ([], []), 1: (["a"], [1.0]), 2: (["b"], [0.0]), 3: (["c"], [1.0]), 4: ([], [])},
            "fold k's picks read p_fold{k} only (folds made to disagree)")
    refused = False
    try:
        check_pools(["fracatlas"])
    except SystemExit:
        refused = True
    _expect(refused, "FracAtlas is refused as a pool")
    print("selftest passed", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Self-training's teacher labels for the fracture detector (idea 133).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="the teacher's probability for every picture of a pool")
    s.add_argument("--versions", type=int, nargs="+", required=True,
                   help="the teacher: each version's five fold nets, read and averaged (v140)")
    s.add_argument("--pool", action="append", required=True,
                   help="a collection to label, not FracAtlas or GRAZPEDWRI-DX; repeat for more (bonefract)")
    s.add_argument("--out", required=True, help="the csv: picture, dataset, patient, p_teacher")
    sub.add_parser("selftest", help="the confident-pick rule and the csv round trip, on made-up numbers")
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.versions, a.pool, a.out)
    else:
        selftest()


if __name__ == "__main__":
    main()
