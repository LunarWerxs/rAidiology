"""Score the chest X-ray reader against ChestX-Det's radiologist labels, beside NIH's report-mined ones.

NIH's labels were mined from reports by NLP; ChestX-Det had three board-certified radiologists box 13
findings on 3,578 NIH pictures (3,025 of them downloadable and matched back by tools/chestxdet_match.py).
For the pictures a reader version scored out of fold, each finding the two share is scored both ways:
the AUROC against the radiologists' boxes (a picture is positive when it holds one) and against NIH's
label, on the same pictures, plus how often NIH's label agrees with the radiologists (caught and cleared,
out of 100). ChestX-Det chose mostly abnormal pictures (2,478 of the 3,025 hold a finding), so these
AUROCs say how well a finding is told apart among sick chests, not across all comers.

With two versions (A B), also the paired difference B - A of the mean AUROC over the shared findings
against the radiologists, on the pictures both scored, from 1000 bootstrap resamples of patients:
BETTER when its 95% interval is above zero, WORSE when below, else INCONCLUSIVE.

Usage: python tools/chestxdet_eval.py <reader version, e.g. 101> [<another version>]"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
from core import ledger, paths  # noqa: E402
import datasets  # noqa: E402

CXD = os.path.join(paths.DATASETS, "public", "chestx-det")
SHARED = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
          "Effusion": "Effusion", "Emphysema": "Emphysema", "Fibrosis": "Fibrosis", "Mass": "Mass",
          "Nodule": "Nodule", "Pleural Thickening": "Pleural_Thickening", "Pneumothorax": "Pneumothorax"}


def auroc(score, y):
    y = np.asarray(y, int)
    if y.min() == y.max():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def boot(score, y, n=1000, seed=0):
    g = np.random.default_rng(seed)
    v = [auroc(score[i], y[i]) for i in (g.integers(0, len(y), len(y)) for _ in range(n))]
    return np.nanpercentile(v, [2.5, 97.5])


def paired(a, b, led=None):
    """B - A of the mean AUROC over the shared findings against the radiologists, by patient bootstrap."""
    za, zb = (np.load(os.path.join(paths.WORK, "eval", "chest", "reader-v%d" % v, "oof_logits.npz")) for v in (a, b))
    la = pd.DataFrame(za["logit"], columns=datasets.LABELS, index=za["image"])
    lb = pd.DataFrame(zb["logit"], columns=datasets.LABELS, index=zb["image"])
    m = pd.read_csv(os.path.join(CXD, "nih_match.csv"))
    m = m[m.image.isin(la.index) & m.image.isin(lb.index)].reset_index(drop=True)
    syms = {}
    for part in ("train", "test"):
        for r in json.load(open(os.path.join(CXD, "ChestX_Det_%s.json" % part), encoding="utf-8")):
            syms[r["file_name"]] = set(r["syms"])
    ys = {c: np.array([c in syms[f] for f in m.file_name], int) for c in SHARED}

    def mean_diff(idx):
        d = [auroc(lb.loc[m.image[idx], lab].to_numpy(float), ys[c][idx])
             - auroc(la.loc[m.image[idx], lab].to_numpy(float), ys[c][idx]) for c, lab in SHARED.items()]
        return float(np.nanmean(d))

    pats = m.patient.to_numpy()
    groups = {p: np.where(pats == p)[0] for p in np.unique(pats)}
    keys = list(groups)
    g = np.random.default_rng(0)
    full = mean_diff(np.arange(len(m)))
    boots = [mean_diff(np.concatenate([groups[keys[i]] for i in g.integers(0, len(keys), len(keys))]))
             for _ in range(1000)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])  # a resample with no finding scorable is NaN, not a vote
    verdict = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "INCONCLUSIVE")
    print("v%d against v%d on %d pictures of %d patients, mean AUROC over the %d shared findings against the "
          "radiologists: B - A %+.4f (%+.4f to %+.4f)" % (b, a, len(m), len(keys), len(SHARED), full, lo, hi))
    line = "Verdict (radiologists' labels): %s" % verdict
    print(line)
    if led is not None:
        led.number("paired.pictures", len(m), "count").number("paired.patients", len(keys), "count")
        led.number("paired.auroc_diff", full).number("paired.auroc_diff_low", lo).number("paired.auroc_diff_high", hi)
        led.note_is(line)


def report(version, led=None):
    z = np.load(os.path.join(paths.WORK, "eval", "chest", "reader-v%d" % version, "oof_logits.npz"))
    logit = pd.DataFrame(z["logit"], columns=datasets.LABELS, index=z["image"])
    m = pd.read_csv(os.path.join(CXD, "nih_match.csv"))
    syms = {}
    for part in ("train", "test"):
        for r in json.load(open(os.path.join(CXD, "ChestX_Det_%s.json" % part), encoding="utf-8")):
            syms[r["file_name"]] = set(r["syms"])
    m = m[m.image.isin(logit.index)]
    nih = datasets.index().set_index("image").loc[m.image]
    print("reader v%d: %d of ChestX-Det's %d matched pictures were scored out of fold (%d with a finding)"
          % (version, len(m), len(syms), sum(bool(syms[f]) for f in m.file_name)))
    print("%-19s %4s %4s  %-26s %-26s %s" % ("finding", "rad+", "nih+", "AUROC, radiologists' boxes",
                                              "AUROC, NIH's labels", "NIH label vs radiologists: caught / cleared"))
    rows = []
    for cxd, lab in SHARED.items():
        y_rad = np.array([cxd in syms[f] for f in m.file_name], int)
        y_nih = nih[lab].to_numpy(int)
        s = logit.loc[m.image, lab].to_numpy(float)
        a_rad, a_nih = auroc(s, y_rad), auroc(s, y_nih)
        lo, hi = boot(s, y_rad)
        caught = 100.0 * (y_nih[y_rad == 1] == 1).mean() if y_rad.sum() else float("nan")
        cleared = 100.0 * (y_nih[y_rad == 0] == 0).mean()
        rows.append((cxd, a_rad, a_nih))
        if led is not None:
            k = "v%d.%s" % (version, cxd.replace(" ", "_"))
            led.number(k + ".auroc_radiologists", a_rad).number(k + ".auroc_radiologists_low", lo)
            led.number(k + ".auroc_radiologists_high", hi).number(k + ".auroc_nih", a_nih)
            led.number(k + ".nih_caught", caught).number(k + ".nih_cleared", cleared)
            led.number(k + ".radiologist_positives", y_rad.sum(), "count").number(k + ".nih_positives", y_nih.sum(), "count")
        print("%-19s %4d %4d  %.3f (%.3f to %.3f)       %.3f                      %5.1f / %5.1f"
              % (cxd, y_rad.sum(), y_nih.sum(), a_rad, lo, hi, a_nih, caught, cleared))
    r = np.array([[a, b] for _, a, b in rows], float)
    print("mean over the %d findings: %.3f against the radiologists, %.3f against NIH's labels"
          % (len(rows), np.nanmean(r[:, 0]), np.nanmean(r[:, 1])))
    if led is not None:
        led.number("v%d.pictures" % version, len(m), "count")
        led.number("v%d.mean_auroc_radiologists" % version, np.nanmean(r[:, 0]))
        led.number("v%d.mean_auroc_nih" % version, np.nanmean(r[:, 1]))


def main():
    versions = [int(v) for v in sys.argv[1:3]]
    with ledger.run("chest_eval", "chestxdet " + " vs ".join("v%d" % v for v in versions)) as led:
        led.settings_from({"versions": versions, "boot": 1000})
        for v in versions:
            report(v, led)
            print()
        if len(versions) == 2:
            paired(*versions, led=led)


if __name__ == "__main__":
    main()
