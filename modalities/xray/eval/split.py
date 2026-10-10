"""The locked split for both X-ray datasets: 20% of the units held out as test, 80% development in
5 folds. Seed 20261001. Written once, committed, and never regenerated: `split.json` is the record.

The unit is the thing resampled and the thing that may not appear on both sides:
- `grazpedwri`: the **patient** (`patient_id`), so a person's images never straddle the split.
- `fracatlas`: the **image**, because FracAtlas publishes no patient identifier. Two radiographs of
  the same person could therefore fall on both sides there. That limitation is on the scorecard and
  is why the ship decision rests on GRAZPEDWRI-DX.

Units are stratified by facts taken from the dataset's own index (`datasets.index`), read off the
unit's first image in key order:
- `grazpedwri`: age band x sex x whether any of the patient's images shows a fracture;
- `fracatlas`: body part x view x whether the image shows a fracture.
Strata with fewer than 10 units are pooled into one "pooled small strata" stratum.

Procedure (deterministic, one numpy Generator per dataset): for each stratum in sorted name order,
the stratum's units sorted by key (numerically when every key is an integer, else as strings) are
permuted; the first round(0.2 * n) go to test; the rest of every stratum, concatenated in that same
order, are dealt to folds 0..4 in turn with one running counter, so small strata spread over folds.

Usage: python split.py [out.json]
       python split.py --add <dataset>   (a collection added later; the locked ones stay as they are)"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402

MIN_STRATUM = 10
FOLDS = 5
TEST_SHARE = 0.2
PATH = os.path.join(HERE, "split.json")


def sort_key(keys):
    """Numeric order when every key is an integer, else string order. Deterministic either way."""
    try:
        [int(k) for k in keys]
        return lambda k: (0, int(k), "")
    except ValueError:
        return lambda k: (1, 0, k)


def unit_facts(dataset):
    """{unit: stratum name before pooling}, from the unit's first image in key order."""
    df = datasets.index(dataset)
    cols = datasets.SETS[dataset]["stratify_split"]
    first = df.sort_values("image").groupby("patient", sort=False).first()
    return {u: " ".join(str(r[c]) for c in cols) for u, r in first.iterrows()}


def make_one(dataset):
    raw = unit_facts(dataset)
    counts = {}
    for v in raw.values():
        counts[v] = counts.get(v, 0) + 1
    strat = {u: (v if counts[v] >= MIN_STRATUM else "pooled small strata") for u, v in raw.items()}
    g = np.random.default_rng(datasets.SEED)
    key = sort_key(list(raw))
    test, dev_order = [], []
    for name in sorted(set(strat.values())):
        members = sorted((u for u in strat if strat[u] == name), key=key)
        perm = [members[i] for i in g.permutation(len(members))]
        k = int(round(TEST_SHARE * len(members)))
        test += perm[:k]
        dev_order += perm[k:]
    assign = {u: "test" for u in test}
    for i, u in enumerate(dev_order):
        assign[u] = i % FOLDS
    df = datasets.index(dataset)
    images = df.groupby("patient").size().to_dict()
    return {
        "source": datasets.SETS[dataset]["cite"],
        "unit": "patient" if datasets.SETS[dataset]["patient"] else "image (no patient identifier is published)",
        "n": {"units": len(assign), "images": int(len(df)),
              "test_units": len(test), "dev_units": len(dev_order),
              "test_images": int(sum(images[u] for u in test)),
              "dev_images": int(sum(images[u] for u in dev_order))},
        "strata": dict(sorted(counts.items())),
        "dev_fold_units": {str(f): sum(1 for v in assign.values() if v == f) for f in range(FOLDS)},
        "assignment": {u: assign[u] for u in sorted(assign, key=key)},
    }


def make():
    return {"seed": datasets.SEED, "folds": FOLDS, "test_share": TEST_SHARE,
            "min_stratum": MIN_STRATUM, "rule": __doc__.split("Usage:")[0].strip(),
            "datasets": {d: make_one(d) for d in datasets.NAMES}}


def load(p=None):
    return json.load(open(p or PATH, encoding="utf-8"))


def assignment(dataset, s=None):
    """{unit: 0..4 for development, "test"}."""
    return (s or load())["datasets"][dataset]["assignment"]


def frame(dataset, split, s=None):
    """index(dataset) rows of that split, with `split` and `fold` columns added."""
    a = assignment(dataset, s)
    df = datasets.index(dataset).copy()
    where = df.patient.map(a)
    if where.isna().any():
        raise SystemExit("%s: %d unit(s) are not in split.json" % (dataset, int(where.isna().sum())))
    df["split"] = np.where(where == "test", "test", "dev")
    df["fold"] = [("" if w == "test" else int(w)) for w in where]
    return df[df.split == split].reset_index(drop=True)


def add(dataset, path=PATH):
    """Lock a collection added later (`datasets.ALL`) into the split record, by the same rule and
    seed, before anything is fitted to it. Each collection draws from its own generator, so the
    ones already locked are rewritten byte for byte or not at all: this refuses if one would change."""
    s = load(path)
    if dataset in s["datasets"]:
        raise SystemExit("%s is already in %s; a locked split is never regenerated" % (dataset, path))
    before = {d: json.dumps(v, sort_keys=True) for d, v in s["datasets"].items()}
    s["datasets"][dataset] = make_one(dataset)
    if any(json.dumps(s["datasets"][d], sort_keys=True) != before[d] for d in before):
        raise SystemExit("adding %s would change a locked split; nothing written" % dataset)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, sort_keys=False)
    v = s["datasets"][dataset]
    print("%-11s %s  folds %s  strata %d" % (dataset, v["n"], v["dev_fold_units"], len(v["strata"])))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--add":
        add(sys.argv[2])
        sys.exit(0)
    out = sys.argv[1] if len(sys.argv) > 1 else PATH
    s = make()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, sort_keys=False)
    for d, v in s["datasets"].items():
        print("%-11s %s  folds %s  strata %d" % (d, v["n"], v["dev_fold_units"], len(v["strata"])))
    print(out)
