"""The locked split for chest X-ray: NIH's own test list is the test, and the rest is development in
5 folds. Seed 20261003. Written once, committed, and never regenerated: `split.json` is the record.

The unit is the patient, so a person's pictures never straddle a fold or the test.

Why each choice:
- Test = the patients of NIH's `test_list.txt`, not a draw of our own. NIH made that split by patient,
  and every published result on this set uses it, so our scores can be set beside theirs.
- Development = the patients of `train_val_list.txt`, dealt to 5 folds. The strata are whether any of
  the patient's pictures is abnormal, times the patient's sex (read off their first picture in name
  order), so each fold has about the same share of abnormal patients and of women.
- Procedure (deterministic, one numpy Generator): for each stratum in sorted name order, its patients
  sorted by number are permuted; the strata, concatenated in that order, are dealt to folds 0..4 in
  turn with one running counter, so a small stratum still spreads over the folds.
- Refuses to overwrite. A split that is written again with a different result would quietly move
  patients between development and the folds a model was already scored on; --force is for the
  rare case that is meant.

Usage: python split.py [--force]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402

SEED = 20261003
FOLDS = 5
PATH = os.path.join(HERE, "split.json")


def _patients():
    """The index, after checking no patient is on both sides of NIH's split."""
    df = datasets.index()
    sides = df.groupby("patient").nih_split.nunique()
    if (sides > 1).any():
        raise SystemExit("nih-chestxray14: %d patients are in both of NIH's lists" % int((sides > 1).sum()))
    return df


def make(path=PATH, force=False):
    if os.path.exists(path) and not force:
        raise SystemExit("%s exists: a locked split is written once, committed, never regenerated "
                         "(--force only if that is meant)" % path)
    df = _patients()
    dev = df[df.nih_split == "train_val"]
    first = dev.groupby("patient").first()                 # df is sorted by image: the first picture
    anyab = dev.groupby("patient").Abnormal.max()
    stratum = {int(p): "%s %s" % ("abnormal" if anyab[p] else "normal", first.sex[p]) for p in first.index}
    g = np.random.default_rng(SEED)
    order = []
    for name in sorted(set(stratum.values())):
        members = sorted(p for p in stratum if stratum[p] == name)
        order += [members[i] for i in g.permutation(len(members))]
    folds = {str(p): i % FOLDS for i, p in enumerate(order)}
    test_patients = sorted(int(p) for p in df[df.nih_split == "test"].patient.unique())
    counts = {"images": int(len(df)), "patients": int(df.patient.nunique()),
              "dev_patients": len(order), "dev_images": int(len(dev)),
              "test_patients": len(test_patients), "test_images": int((df.nih_split == "test").sum()),
              "strata": {k: sum(1 for v in stratum.values() if v == k) for k in sorted(set(stratum.values()))},
              "fold_patients": {str(f): sum(1 for v in folds.values() if v == f) for f in range(FOLDS)}}
    s = {"seed": SEED, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "folds": folds,
         "test_patients": test_patients, "counts": counts}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    return s


def load(path=PATH):
    if not os.path.exists(path):
        raise SystemExit("no split at %s; write it once with: python modalities/chest_xray/eval/split.py" % path)
    return json.load(open(path, encoding="utf-8"))


def frame(split, s=None):
    """index() rows of 'dev' or 'test', with `split` and `fold` (-1 for test) added."""
    if split not in ("dev", "test"):
        raise SystemExit("split is 'dev' or 'test', not %r" % (split,))
    datasets.guard_test(split)
    s = s or load()
    df = datasets.index().copy()
    is_test = df.patient.isin(set(s["test_patients"]))
    fold = df.patient.map(lambda p: s["folds"].get(str(p), -1))
    stray = ~is_test & (fold == -1)
    if stray.any():
        raise SystemExit("%d pictures are of patients that are not in split.json" % int(stray.sum()))
    df["split"] = np.where(is_test, "test", "dev")
    df["fold"] = np.where(is_test, -1, fold).astype(int)
    return df[df.split == split].reset_index(drop=True)


def _summary(df):
    return "%6d patients %7d images %5.1f%% abnormal" % (df.patient.nunique(), len(df), 100 * df.Abnormal.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write the locked chest X-ray split, once.")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split.json")
    a = ap.parse_args()
    s = make(force=a.force)
    df = datasets.index()
    test = set(s["test_patients"])
    dev = df[~df.patient.isin(test)]
    for f in range(FOLDS):
        print("fold %d  %s" % (f, _summary(dev[dev.patient.map(lambda p: s["folds"][str(p)]) == f])))
    print("test    %s" % _summary(df[df.patient.isin(test)]))
    print(PATH)
