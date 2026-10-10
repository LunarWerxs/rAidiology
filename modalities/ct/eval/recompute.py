"""An independent recomputation of the CT scorecard, written from METRICS_SPEC.md alone.

This is a deliberate second implementation. It was written without reading the scorer, so where the
two agree the agreement means something, and where they differ one of them has misread the
specification. It must never be merged with the scorer, share helpers with it, or be "tidied" into
calling it: two scripts that share code are one script checked against itself.

    python modalities/ct/eval/recompute.py [split] [--reference METHOD] [--compare scorecard.tsv] [--dir DIR]

Reads every work/eval/ct/pred_*_<split>.csv (or DIR/pred_*_<split>.csv with --dir, a second look's folder) and prints one line per number:
stratum, method, metric, value, low, high, tab-separated. Only headline_dice, invention_rate and
(with --reference) headline_dice_minus_reference carry an interval; invented_size_median and
n_eligible_structures print `nan` for both bounds."""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK = os.path.join(ROOT, "work", "eval", "ct")
RESAMPLES = 2000
MIN_PRESENT = 10
TOLERANCE = 1e-9
COLUMNS = ["method", "split", "fold", "scan", "structure", "in_scope", "drop_reason",
           "truth_voxels", "pred_voxels", "both_voxels"]
DROP_REASONS = {"unreadable", "geometry_mismatch"}


def load(split, folder=None):
    folder = folder or WORK
    files = sorted(glob.glob(os.path.join(folder, "pred_*_%s.csv" % split)))
    if not files:
        raise SystemExit("No predictions at %s" % os.path.join(folder, "pred_*_%s.csv" % split))
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype={"scan": str, "structure": str, "method": str, "split": str,
                                  "drop_reason": str, "fold": str}, keep_default_na=False)
        missing = [c for c in COLUMNS if c not in d.columns]
        if missing:
            raise SystemExit("%s lacks columns %s" % (f, missing))
        frames.append(d[COLUMNS])
    d = pd.concat(frames, ignore_index=True)
    if (d["split"] != split).any():
        raise SystemExit("Rows from another split in the %s files" % split)
    for c in ("in_scope", "truth_voxels", "pred_voxels", "both_voxels"):
        d[c] = pd.to_numeric(d[c], errors="raise").astype(np.int64)
    if not d["in_scope"].isin([0, 1]).all():
        raise SystemExit("in_scope must be 0 or 1")
    dropped = d[d["in_scope"] == 0]
    if not dropped["drop_reason"].isin(DROP_REASONS).all():
        raise SystemExit("A row is out of scope for a reason the specification does not allow")
    kept = d[d["in_scope"] == 1]
    if (kept["drop_reason"] != "").any():
        raise SystemExit("An in-scope row carries a drop_reason")
    if (d[["truth_voxels", "pred_voxels", "both_voxels"]] < 0).any().any():
        raise SystemExit("Negative voxel count")
    if (d["both_voxels"] > d[["truth_voxels", "pred_voxels"]].min(axis=1)).any():
        raise SystemExit("both_voxels exceeds truth_voxels or pred_voxels")
    if d.duplicated(["method", "scan", "structure"]).any():
        raise SystemExit("Duplicate (method, scan, structure) rows")
    return d


def grid(d, scans, structures):
    """Per method, (scans x structures) arrays. A missing pair is an error, never a zero, because
    zero-filling is how a method that skipped the hard scans scores well."""
    out = {}
    for method, m in d.groupby("method", sort=True):
        m = m.set_index(["scan", "structure"])
        want = pd.MultiIndex.from_product([scans, structures], names=["scan", "structure"])
        absent = want.difference(m.index)
        if len(absent):
            raise SystemExit("%s is missing %d (scan, structure) rows, e.g. %s"
                             % (method, len(absent), list(absent[:3])))
        extra = m.index.difference(want)
        if len(extra):
            raise SystemExit("%s has rows outside the split, e.g. %s" % (method, list(extra[:3])))
        m = m.reindex(want)
        shape = (len(scans), len(structures))
        out[method] = {c: m[c].to_numpy().reshape(shape)
                       for c in ("in_scope", "truth_voxels", "pred_voxels", "both_voxels")}
    return out


def stratum_masks(meta):
    """The declared strata, `all` first. Rebuilt here from SET rather than through
    datasets.strata(), and lower-casing both sides: index() lower-cases the columns, so a declared
    value such as institute "I" would otherwise match nothing and main_institute would be empty."""
    yield "all", pd.Series(True, index=meta.index)
    for name, (column, how, value) in datasets.SET["strata"].items():
        col = meta[column].astype(str).str.strip().str.lower()
        value = str(value).strip().lower()
        if how == "is":
            yield name, col == value
        elif how == "is not":
            yield name, col != value
        elif how == "contains":
            yield name, col.str.contains(value, regex=False)
        else:
            raise SystemExit("Unknown stratum rule %r for %s" % (how, name))


def score(g, counts):
    """Every number for one method on one stratum, for each row of `counts` (resamples x scans).
    Row 0 of the result is the observed sample, the rest are resamples."""
    inscope = g["in_scope"] == 1
    truth, pred, both = g["truth_voxels"], g["pred_voxels"], g["both_voxels"]
    present = inscope & (truth > 0)
    denom = truth + pred
    if (present & (denom == 0)).any():
        raise SystemExit("Present-but-empty pair: the specification makes this an error")
    dice = np.where(present, 2.0 * both / np.where(denom == 0, 1, denom), 0.0)

    # Eligibility is fixed on the observed scans, not re-decided per resample: the headline should
    # be the same set of structures in every draw, or the interval mixes two different averages.
    n_present = present.sum(axis=0)
    eligible = n_present >= MIN_PRESENT

    full = np.vstack([np.ones((1, counts.shape[1])), counts])
    with np.errstate(invalid="ignore", divide="ignore"):
        per_structure = (full @ dice[:, eligible]) / (full @ present[:, eligible])
        # ponytail: a resample can draw no scan containing an eligible structure; that structure
        # is skipped in that draw rather than counted as 0. Rare at >= 10 present scans.
        headline = np.nanmean(per_structure, axis=1) if eligible.any() else \
            np.full(full.shape[0], math.nan)

        absent = inscope & (truth == 0)
        invented = absent & (pred > 0)
        rate = (full @ invented.sum(axis=1)) / (full @ absent.sum(axis=1))

    # Observed sample only, no interval: a method that never invented has no median, so nan, not 0.
    sizes = pred[invented]
    size = float(np.median(sizes)) if sizes.size else math.nan

    return {"headline_dice": headline, "invention_rate": rate, "invented_size_median": size,
            "n_eligible_structures": int(eligible.sum())}


def interval(x):
    x = x[~np.isnan(x)]
    if not len(x):
        return math.nan, math.nan
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def compute(split, reference=None, folder=None):
    datasets.guard_test(split)
    d = load(split, folder)
    scans = sorted(d["scan"].unique())
    structures = datasets.structures()
    if len(structures) != 117:
        print("warning: %d structures on disk, the specification says 117" % len(structures),
              file=sys.stderr)
    grids = grid(d, scans, structures)
    if reference is not None and reference not in grids:
        raise SystemExit("--reference %r has no rows in the %s predictions (methods: %s)"
                         % (reference, split, ", ".join(sorted(grids))))

    meta = datasets.index()
    meta["image_id"] = meta["image_id"].astype(str)
    meta = meta.set_index("image_id")
    unknown = sorted(set(scans) - set(meta.index))
    if unknown:
        raise SystemExit("Scans not in the metadata, e.g. %s" % unknown[:3])
    meta = meta.loc[scans]

    cells = []
    for stratum, mask in stratum_masks(meta.reset_index()):
        idx = np.flatnonzero(mask.to_numpy())
        if not len(idx):
            print("warning: stratum %s is empty" % stratum, file=sys.stderr)
            continue
        # One generator per stratum, seeded afresh, and the same count matrix for every method:
        # that is what makes a difference between two methods' intervals meaningful.
        rng = np.random.default_rng(datasets.SEED)
        draws = rng.integers(0, len(idx), size=(RESAMPLES, len(idx)))
        counts = np.zeros((RESAMPLES, len(idx)))
        np.add.at(counts, (np.repeat(np.arange(RESAMPLES), len(idx)), draws.ravel()), 1)
        scores = {method: score({k: v[idx] for k, v in grids[method].items()}, counts)
                  for method in sorted(grids)}
        for method, s in scores.items():
            for metric in ("headline_dice", "invention_rate"):
                low, high = interval(s[metric][1:])
                cells.append((stratum, method, metric, float(s[metric][0]), low, high))
            for metric in ("invented_size_median", "n_eligible_structures"):
                cells.append((stratum, method, metric, float(s[metric]), math.nan, math.nan))
            if reference is not None and method != reference:
                # Differenced inside each resample (same drawn scans, same count matrix), then
                # percentiled; a nan headline on either side gives a nan difference, which
                # interval() leaves out.
                diff = s["headline_dice"] - scores[reference]["headline_dice"]
                low, high = interval(diff[1:])
                cells.append((stratum, method, "headline_dice_minus_reference",
                              float(diff[0]), low, high))
    return cells


def read_tsv(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            out[tuple(parts[:3])] = tuple(float(p) for p in parts[3:])
    return out


def same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= TOLERANCE


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("split", nargs="?", default="dev")
    p.add_argument("--compare")
    p.add_argument("--reference")
    p.add_argument("--dir", help="read pred_*_<split>.csv from this folder instead of work/eval/ct")
    a = p.parse_args()

    cells = compute(a.split, a.reference, os.path.abspath(a.dir) if a.dir else None)
    for c in cells:
        print("%s\t%s\t%s\t%r\t%r\t%r" % c)

    if a.compare:
        theirs = read_tsv(a.compare)
        ours = {c[:3]: c[3:] for c in cells}
        bad = []
        for key in sorted(set(ours) | set(theirs)):
            if key not in theirs or key not in ours:
                bad.append("%s only in %s" % ("/".join(key), "ours" if key in ours else "theirs"))
                continue
            for name, x, y in zip(("value", "low", "high"), ours[key], theirs[key]):
                if not same(x, y):
                    bad.append("%s %s: ours %r, theirs %r" % ("/".join(key), name, x, y))
        for b in bad:
            print("DIFFERS\t" + b, file=sys.stderr)
        if bad:
            sys.exit(1)
        print("agrees on all %d cells" % len(ours), file=sys.stderr)


if __name__ == "__main__":
    main()
