"""The locked CT split. Written once, committed as `split.json`, never regenerated.

**The test set is the authors' own.** The TotalSegmentator dataset publishes a train/validation/test
split, and this keeps the authors' 89 test scans as the held-out test rather than cutting a new one.
Two reasons, and both matter more than tidiness: it is a split nobody here chose after seeing a
single result, and it is the split the published numbers for this dataset were produced on, so a
comparison with them means something. Cutting our own would have bought nothing except the chance
to pick a favourable one by accident.

Their train and validation scans become our development set, 1,139 scans, dealt into 5 folds here.

**The unit is the scan, not the patient**, because the dataset publishes no patient identifier.
Two scans of one person could therefore fall on both sides of the split, exactly the limitation
FracAtlas has on the X-ray side. It goes on the scorecard rather than being quietly hoped away.

Development folds are stratified by facts from the published metadata: study type x institute x
whether the scan is marked as showing pathology. Strata with fewer than 10 scans are pooled into one
"pooled small strata" stratum, so a rare combination does not land entirely in one fold.

Procedure, deterministic, one numpy Generator seeded with `datasets.SEED`: for each stratum in
sorted name order, its scans sorted by image id are permuted, then dealt to folds 0..4 in turn with
one running counter carried across strata, so small strata spread over folds instead of each
starting again at fold 0.

Usage: python split.py [out.json]        (refuses to overwrite a committed split; see --force)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402

MIN_STRATUM = 10
FOLDS = 5
PATH = os.path.join(HERE, "split.json")


def stratum_of(row):
    return "%s|%s|%s" % (row["study_type"] or "unknown", row["institute"] or "unknown",
                         "pathology" if str(row["pathology"]) not in ("no_pathology", "", "nan")
                         else "no_pathology")


def build():
    table = datasets.index()
    key = datasets.SET["key"]
    table["stratum"] = table.apply(stratum_of, axis=1)

    held = table[table["given_split"] == "test"]
    dev = table[table["given_split"] != "test"].copy()

    counts = dev["stratum"].value_counts()
    small = set(counts[counts < MIN_STRATUM].index)
    dev["stratum"] = dev["stratum"].where(~dev["stratum"].isin(small), "pooled small strata")

    rng = np.random.default_rng(datasets.SEED)
    folds = {}
    turn = 0
    for name in sorted(dev["stratum"].unique()):
        ids = sorted(str(v) for v in dev.loc[dev["stratum"] == name, key])
        for i in rng.permutation(len(ids)):
            folds[ids[int(i)]] = turn % FOLDS
            turn += 1

    return {
        "dataset": datasets.NAME,
        "seed": datasets.SEED,
        "folds": FOLDS,
        "unit": "scan",
        "unit_note": ("The dataset publishes no patient identifier, so the scan is the unit and two "
                      "scans of one person could fall on both sides. This is on the scorecard."),
        "test_from": ("the authors' own published test split, kept rather than re-cut, so the held-out "
                      "set was not chosen by anybody here and matches the split the published "
                      "numbers for this dataset use"),
        "test": sorted(str(v) for v in held[key]),
        "dev_folds": {s: int(f) for s, f in sorted(folds.items())},
        "strata_used": sorted(dev["stratum"].unique().tolist()),
        "min_stratum": MIN_STRATUM,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=PATH)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split. The split is the record of what was held "
                         "out; regenerating it after a model has seen the development set is how a "
                         "test set stops being a test set.")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        raise SystemExit("%s already exists. That file is the locked split and is committed. Pass "
                         "--force only if nothing has been trained against it yet." % a.out)
    out = build()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    per_fold = {}
    for f_ in out["dev_folds"].values():
        per_fold[f_] = per_fold.get(f_, 0) + 1
    print("test %d scan(s) (the authors' own), development %d scan(s) in %d folds: %s"
          % (len(out["test"]), len(out["dev_folds"]), FOLDS,
             ", ".join("%d=%d" % (k, per_fold[k]) for k in sorted(per_fold))))
    print("%d stratum/strata used" % len(out["strata_used"]))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
