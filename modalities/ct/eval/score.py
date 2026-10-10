"""Turn prediction rows into the scorecard, exactly as METRICS_SPEC.md describes it.

This script reads one file, `predictions.csv`, and opens no scan. That is deliberate: everything
scored here is arithmetic on three voxel counts, so the second, independently written script that
recomputes these numbers cannot agree with this one by sharing a bug about how a volume was read.
If the two disagree on any cell, nothing is published.

What it writes, per method and per pre-declared stratum:

- **mean Dice**, the mean over structures of each structure's mean Dice across the scans that
  contain it. Each structure counts once. Never weighted by volume, which would make the number a
  report on the liver with the other 116 structures rounded off.
- **the invention rate**, how often a method outlines a structure in a scan that does not contain
  one, because most pairs are absent ones and no Dice column would ever show it.
- **95% confidence intervals** by resampling scans, and the interval of the difference between each
  method and the best of the others, resampled on the same scans.

    python modalities/ct/eval/score.py dev            [--resamples 2000] [--out SCORECARD.md]
    python modalities/ct/eval/score.py test --dir DIR   read pred_*_<split>.csv from DIR and write the card
                                                        and tsv there (a second look's own folder)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import datasets  # noqa: E402
from core import ledger, paths  # noqa: E402

OUT = os.path.join(paths.WORK, "eval", "ct")
RESAMPLES = 2000
MIN_SCANS = 10                 # a structure in fewer scans than this is reported, not averaged in
CANDIDATE = "segmenter"        # the trained method; every other method in the rows is a baseline


def scan_dice(rows):
    """One mean Dice per scan, as a Series indexed by scan, for a single method's rows.

    The per-scan form of the headline: the Dice of every structure present in the scan (in scope,
    truth_voxels > 0), averaged within the scan, over the structures the headline scores (present in
    at least MIN_SCANS scans). Used by the second look's development gate."""
    r = rows[(rows["in_scope"] == 1) & (rows["truth_voxels"] > 0)].copy()
    r["dice"] = 2.0 * r["both_voxels"] / np.maximum(r["truth_voxels"] + r["pred_voxels"], 1.0)
    counts = r.groupby("structure")["scan"].nunique()
    r = r[r["structure"].isin(counts[counts >= MIN_SCANS].index)]
    return r.groupby("scan")["dice"].mean()


def load(split, folder=None):
    """Every prediction row for a split, with the scan metadata joined on for the strata."""
    folder = folder or OUT
    frames = []
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if f.startswith("pred_") and f.endswith("_%s.csv" % split):
            frames.append(pd.read_csv(os.path.join(folder, f)))
    if not frames:
        raise SystemExit("No pred_*_%s.csv under %s. Run baselines.py predict %s first."
                         % (split, folder, split))
    d = pd.concat(frames, ignore_index=True)
    meta = datasets.index().rename(columns={datasets.SET["key"]: "scan"})
    return d.merge(meta, on="scan", how="left")


class Draws:
    """How many times each scan was drawn, for every resample, computed once and shared.

    Shared on purpose: two methods compared on different draws are not a comparison. One matrix
    here means every number on the scorecard, and every difference between numbers, comes from the
    same 2,000 imagined repetitions of the study."""

    def __init__(self, scans, resamples, seed):
        self.scans = list(scans)
        n = len(self.scans)
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, n, size=(resamples, n))
        self.m = np.zeros((resamples, n), dtype=np.int32)
        for b in range(resamples):
            self.m[b] = np.bincount(picks[b], minlength=n)
        self.m = np.vstack([np.ones((1, n), dtype=np.int32), self.m])   # row 0 is the real study

    def ci(self, values):
        """values[0] is the measured number; the rest are the resamples."""
        rest = values[1:]
        ok = np.isfinite(rest)
        if not ok.any():
            return float(values[0]), float("nan"), float("nan")
        return (float(values[0]), float(np.percentile(rest[ok], 2.5)),
                float(np.percentile(rest[ok], 97.5)))


def per_method(rows, scans, draws):
    """Mean Dice and invention rate for one method over one stratum, resampled.

    The matrix form from the spec: per-scan sums are a vector, so a resample is a matrix product
    rather than another pass over the rows."""
    index = {s: i for i, s in enumerate(scans)}
    names = sorted(rows["structure"].unique())
    s_index = {n: i for i, n in enumerate(names)}
    n_s, n_c = len(names), len(scans)

    dice_sum = np.zeros((n_s, n_c))          # Dice of each present (structure, scan)
    present = np.zeros((n_s, n_c))
    invented = np.zeros(n_c)                 # absent pairs where something was predicted
    absent = np.zeros(n_c)
    invented_sizes = []

    r = rows[rows["in_scope"] == 1]
    si = r["structure"].map(s_index).to_numpy()
    ci = r["scan"].map(index).to_numpy()
    t = r["truth_voxels"].to_numpy(dtype=np.float64)
    p = r["pred_voxels"].to_numpy(dtype=np.float64)
    b = r["both_voxels"].to_numpy(dtype=np.float64)

    here = t > 0
    np.add.at(dice_sum, (si[here], ci[here]), 2 * b[here] / np.maximum(t[here] + p[here], 1.0))
    np.add.at(present, (si[here], ci[here]), 1.0)
    gone = ~here
    np.add.at(absent, ci[gone], 1.0)
    np.add.at(invented, ci[gone], (p[gone] > 0).astype(float))
    invented_sizes = p[gone][p[gone] > 0]

    m = draws.m.T                                           # scans x draws
    num = dice_sum @ m                                      # structures x draws
    den = present @ m
    eligible = present.sum(axis=1) >= MIN_SCANS             # fixed by the study, not by the draw
    with np.errstate(invalid="ignore", divide="ignore"):
        per_structure = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    headline = np.nanmean(per_structure[eligible], axis=0) if eligible.any() else np.full(m.shape[1], np.nan)
    rate = (invented @ m) / np.maximum(absent @ m, 1e-9)

    return {
        "mean_dice": draws.ci(headline),
        "headline_draws": headline,      # row 0 measured, the rest resamples: for paired differences
        "invention_rate": draws.ci(rate),
        "invention_draws": rate,         # the same, for a paired difference of invention rates
        # Not a zero when a method never invents anything: there is no median of nothing, and a 0
        # there reads as "it invented things of no size", which is a different and untrue claim.
        "invented_median_voxels": float(np.median(invented_sizes)) if len(invented_sizes) else float("nan"),
        "structures_scored": int(eligible.sum()),
        "structures_seen": n_s,
        "per_structure": {n: (float(per_structure[i, 0]), int(present[i].sum()))
                          for n, i in s_index.items()},
    }


def difference(a, b, draws):
    """Mean Dice of `a` minus mean Dice of `b`, with its interval.

    Both were resampled on the same drawn scans, so the subtraction happens inside each resample
    and the percentiles are taken of the differences. Subtracting two separately computed intervals
    would throw away exactly what makes a paired comparison sharp: on a hard draw both methods do
    badly together."""
    return draws.ci(a["headline_draws"] - b["headline_draws"])


def scorecard(split, resamples, out_path, reference=None, folder=None):
    d = load(split, folder)
    methods = sorted(d["method"].unique())
    if reference and reference not in methods:
        raise SystemExit("The reference %r has no rows in the %s predictions." % (reference, split))
    dropped = d[d["in_scope"] == 0]
    lines = ["# CT scorecard (%s split)" % split, "",
             "Computed by `modalities/ct/eval/score.py` from prediction rows only, exactly as "
             "[METRICS_SPEC.md](METRICS_SPEC.md) defines it. %d resamples, seed %d, the unit is the "
             "scan." % (resamples, datasets.SEED), "",
             "**The dataset publishes no patient identifier**, so two scans of one person could be "
             "treated as independent here. Read every interval with that in mind.", ""]

    results = {}
    empty = []
    for stratum, mask in datasets.strata(d):
        part = d[mask]
        if not len(part):
            # Said out loud rather than skipped. A declared stratum that matches nothing is either a
            # dataset without those scans or a mistake in how it was declared, and the difference
            # between those two is exactly what a silent skip hides.
            empty.append(stratum)
            continue
        scans = sorted(part["scan"].unique())
        draws = Draws(scans, resamples, datasets.SEED)
        results[stratum] = {m: per_method(part[part["method"] == m], scans, draws) for m in methods}
        results[stratum]["_scans"] = len(scans)
        if reference:
            results[stratum]["_minus_reference"] = {
                m: difference(results[stratum][m], results[stratum][reference], draws)
                for m in methods if m != reference}

    lines += ["## Mean Dice, by stratum", "",
              "| stratum | scans | " + " | ".join(methods) + " |",
              "|---|---:|" + "---:|" * len(methods)]
    for stratum, got in results.items():
        cells = []
        for m in methods:
            v, lo, hi = got[m]["mean_dice"]
            cells.append("%.4f (%.4f to %.4f)" % (v, lo, hi))
        lines.append("| %s | %d | %s |" % (stratum, got["_scans"], " | ".join(cells)))

    lines += ["", "## How often each method outlines a structure that is not there", "",
              "The majority of (scan, structure) pairs are absent ones, and no Dice column would "
              "ever show this.", "",
              "| method | invention rate | median invented size | structures scored |",
              "|---|---:|---:|---:|"]
    top = results.get("all", {})
    for m in methods:
        if m not in top:
            continue
        v, lo, hi = top[m]["invention_rate"]
        lines.append("| %s | %.4f (%.4f to %.4f) | %.0f voxels | %d of %d |"
                     % (m, v, lo, hi, top[m]["invented_median_voxels"],
                        top[m]["structures_scored"], top[m]["structures_seen"]))

    baselines = [m for m in methods if m != CANDIDATE]
    if top and baselines:
        best = reference or max(baselines, key=lambda m: top[m]["mean_dice"][0])
        if reference:
            lines += ["", "## The reference", "",
                      "The reference is **%s**, the strongest baseline on the development split, "
                      "chosen and written down before the test split was opened. On this split it "
                      "scores %.4f mean Dice." % (best, top[best]["mean_dice"][0])]
        else:
            lines += ["", "## The reference", "",
                      "On this split the strongest baseline is **%s** at %.4f mean Dice. That is the "
                      "number a trained segmenter has to beat, with the confidence interval of the "
                      "difference above zero, before it goes anywhere near the app."
                      % (best, top[best]["mean_dice"][0])]

        if reference and CANDIDATE in top:
            lines += ["", "## The segmenter against the reference", "",
                      "Mean Dice of the segmenter minus mean Dice of %s, resampled on the same scans "
                      "for both. The ship rule reads the first row." % reference, "",
                      "| stratum | scans | segmenter | %s | difference (95%% CI) |" % reference,
                      "|---|---:|---:|---:|---:|"]
            for stratum, got in results.items():
                v, lo, hi = got["_minus_reference"][CANDIDATE]
                lines.append("| %s | %d | %.4f | %.4f | %+.4f (%+.4f to %+.4f) |"
                             % (stratum, got["_scans"], got[CANDIDATE]["mean_dice"][0],
                                got[reference]["mean_dice"][0], v, lo, hi))

        if CANDIDATE in top:
            # Every structure, not a sample of them: the question is scored per structure because
            # an average hides the one the model never finds, and so would a table of highlights.
            rows = sorted(top[CANDIDATE]["per_structure"].items(), key=lambda kv: kv[1][0])
            lines += ["", "## Every structure", "",
                      "Mean Dice over the scans that contain the structure, weakest first. No "
                      "intervals here: the contract puts them on the headline.", "",
                      "| structure | segmenter | %s | scans containing it |" % best,
                      "|---|---:|---:|---:|"]
            for n, (v, c) in rows:
                lines.append("| %s | %.4f | %.4f | %d |" % (n, v, top[best]["per_structure"][n][0], c))
        else:
            rows = sorted(top[best]["per_structure"].items(), key=lambda kv: kv[1][0])
            lines += ["", "### Where %s is weakest and strongest" % best, "",
                      "| structure | Dice | scans containing it |", "|---|---:|---:|"]
            for n, (v, c) in rows[:8] + rows[-4:]:
                lines.append("| %s | %.4f | %d |" % (n, v, c))

    if empty:
        lines += ["", "## Declared strata that matched no scan", "",
                  "Reported rather than skipped: a stratum that matches nothing is either a dataset "
                  "without those scans or a mistake in how it was declared.", "",
                  "- " + ", ".join(empty)]

    if len(dropped):
        lines += ["", "## Rows not scored", ""]
        for why, g in dropped.groupby("drop_reason"):
            lines.append("- %s: %d row(s) over %d scan(s)" % (why, len(g), g["scan"].nunique()))

    lines += ["", "---", "", "Not a medical device. These are measurements on a public research "
              "dataset, not a claim about anybody's scan."]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return results, methods, out_path


def write_tsv(results, methods, path):
    """The same numbers in the machine-comparable form the spec fixes, for the recomputation.

    Metric names come from METRICS_SPEC.md rather than from whatever this file happens to call its
    variables: the whole value of a second implementation is lost if the two agree on every number
    and disagree on what to call three of them."""
    lines = []
    for stratum, got in results.items():
        for m in methods:
            if m not in got:
                continue
            for metric, key in (("headline_dice", "mean_dice"),
                                ("invention_rate", "invention_rate")):
                v, lo, hi = got[m][key]
                lines.append("%s\t%s\t%s\t%r\t%r\t%r" % (stratum, m, metric, v, lo, hi))
            nan = float("nan")
            lines.append("%s\t%s\t%s\t%r\t%r\t%r"
                         % (stratum, m, "invented_size_median",
                            got[m]["invented_median_voxels"], nan, nan))
            lines.append("%s\t%s\t%s\t%r\t%r\t%r"
                         % (stratum, m, "n_eligible_structures",
                            float(got[m]["structures_scored"]), nan, nan))
            if m in got.get("_minus_reference", {}):
                v, lo, hi = got["_minus_reference"][m]
                lines.append("%s\t%s\t%s\t%r\t%r\t%r"
                             % (stratum, m, "headline_dice_minus_reference", v, lo, hi))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--out", help="default: SCORECARD.md for dev until the test is opened, then the work folder")
    ap.add_argument("--tsv", help="the same numbers in the form the independent recomputation "
                                  "compares; default work/eval/ct/scorecard_<split>.tsv")
    ap.add_argument("--reference", help="a baseline's name: adds each other method's mean Dice minus "
                                        "its, resampled on the same scans (the ship rule's number)")
    ap.add_argument("--dir", help="read the pred files from this folder and default --out/--tsv into it "
                                  "(a second look's own folder); default is work/eval/ct")
    a = ap.parse_args()
    datasets.guard_test(a.split)
    folder = os.path.abspath(a.dir) if a.dir else OUT
    # Once the test has been opened, SCORECARD.md is the published card (decision, test, development)
    # and only the final evaluation writes it. A bare re-score then lands in the work folder instead
    # of quietly replacing the published card with half of itself.
    published = os.path.exists(os.path.join(OUT, "final_eval.json"))
    a.out = a.out or (os.path.join(folder, "card_%s.md" % a.split) if published or a.split == "test" or a.dir
                      else os.path.join(HERE, "SCORECARD.md"))
    a.tsv = a.tsv or os.path.join(folder, "scorecard_%s.tsv" % a.split)
    t0 = time.time()
    with ledger.run("score", "ct scorecard %s" % a.split) as r:
        results, methods, out = scorecard(a.split, a.resamples, a.out, a.reference, folder)
        top = results.get("all", {})
        for m in methods:
            if m in top:
                r.number("%s.mean_dice" % m, top[m]["mean_dice"][0])
                r.number("%s.invention_rate" % m, top[m]["invention_rate"][0])
        for m, (v, lo, hi) in top.get("_minus_reference", {}).items():
            r.number("%s.minus_reference" % m, v).number("%s.minus_reference_low" % m, lo)
        r.settings_from({"methods": ",".join(methods), "reference": a.reference or "none",
                         "resamples": a.resamples})
        if a.split == "dev" and top:
            # Every structure's Dice, so the ledger can name the weakest. Development only: picking
            # what to work on next from the held-out test would be training on the test.
            who = CANDIDATE if CANDIDATE in top else max(
                (m for m in methods if m in top), key=lambda m: top[m]["mean_dice"][0])
            r.setting("each_is", who)
            for n, (v, _count) in top[who]["per_structure"].items():
                r.number("each.%s" % n, v)
        r.note_is("%d method(s), %d stratum/strata" % (len(methods), len(results)))
    cells = write_tsv(results, methods, a.tsv)
    print("%s (%.1f s), %d cell(s) for the recomputation in %s" % (out, time.time() - t0, cells, a.tsv))
    for m in methods:
        if m in top:
            v, lo, hi = top[m]["mean_dice"]
            print("  %-22s mean Dice %.4f (%.4f to %.4f)" % (m, v, lo, hi))


if __name__ == "__main__":
    main()
