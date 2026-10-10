"""The cancer-or-not model and its training labels against LIDC's own pathology table (a check, nothing trains).

LIDC-IDRI ships `tcia-diagnosis-data-2012-04-20.xls`: for 157 of its patients, a diagnosis at the patient level
(1 benign or non-malignant, 2 primary lung cancer, 3 metastatic; 0 unknown) and how it was made (1 two years of
stable images, 2 biopsy, 3 surgical resection, 4 progression or response). Our cancer-or-not labels come from the
radiologists' malignancy ratings instead (median above 3 malignant, below 3 benign), so this is the one place in
LIDC where a label made by pathology can be set against them.

Per patient with a known diagnosis and at least one scored lesion in a version's out-of-fold file:
- the model reads the patient as its most suspicious lesion (the largest out-of-fold probability), and its
  patient-level AUROC against malignant (2 or 3) versus benign (1) is reported with a 95% interval from 2000
  patient resamples;
- the rating labels read the patient as malignant when any of its lesions is labelled 1, and their sensitivity
  and specificity against the same diagnosis are reported;
- both again on the patients whose diagnosis came from tissue (biopsy or resection) or progression, leaving out
  the two-years-stable ones.
Patients whose lesions all had a median rating of exactly 3 are not in the out-of-fold file and so not counted.

--all reads oof_all.csv (tools/malignancy_all.py: every lesion) instead of oof.csv. --learned adds step B of idea 25: a
per-patient logistic regression on pathology (largest p, mean of the two largest, lesion count, largest diameter, most
readers) in five folds by patient repeated 20 times, against the largest-p read, with the keep rule and Wang et al.'s line.

--pair A B [--all] compares two versions on the patients with a known diagnosis scored in BOTH (each read as its most
suspicious lesion): each one's AUROC on that same set, and the interval (2.5th to 97.5th percentile) of B minus A from
2000 patient resamples (seed 60, the same resample for both). BETTER when the low end is above 0, WORSE when the high
end is below 0, else INCONCLUSIVE.

Usage: python tools/lidc_pathology.py [version ...] [--all] [--learned]   (default 2)
       python tools/lidc_pathology.py --pair A B [--all]
       python tools/lidc_pathology.py selftest
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import ledger, paths  # noqa: E402

XLS = os.path.join(paths.BESIDE, "Datasets", "public", "lidc-idri", "tcia-diagnosis-data-2012-04-20.xls")


def diagnoses():
    d = pd.read_excel(XLS)
    d = d.iloc[:, :3]
    d.columns = ["patient", "diagnosis", "method"]
    d = d.dropna(subset=["patient"])
    d["patient"] = d.patient.astype(str).str.strip()
    d["diagnosis"] = pd.to_numeric(d.diagnosis, errors="coerce")
    d["method"] = pd.to_numeric(d.method, errors="coerce")
    return d


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def top2_mean(s):
    return float(np.sort(np.asarray(s, float))[-2:].mean())


def patients_of(version, use_all=False):
    """One row per patient: p (largest), top2 (mean of the two largest), lesions, diameter (largest, mm), rated
    (1 when any lesion is labelled 1), and readers (most radiologists on any lesion) when the file has the column.
    use_all reads oof_all.csv (every lesion, idea 25) instead of oof.csv."""
    oof = pd.read_csv(os.path.join(paths.WORK, "eval", "lung", "malignancy-v%d" % version,
                                   "oof_all.csv" if use_all else "oof.csv"))
    idx = pd.read_csv(os.path.join(paths.WORK, "eval", "lung", "index.csv"), dtype={"patient": str})
    oof = oof.merge(idx[["seriesuid", "patient"]], on="seriesuid", how="left")
    return patient_table(oof)


def patient_table(oof):
    oof = oof.assign(is_malignant=(oof.label == 1).astype(int))
    agg = {"p": ("p", "max"), "top2": ("p", top2_mean), "rated": ("is_malignant", "max"), "lesions": ("p", "size"),
           "diameter": ("diameter_mm", "max")}
    if "readers" in oof:
        agg["readers"] = ("readers", "max")
    return oof.groupby("patient").agg(**agg).reset_index()


# ---- step B: a patient read learned on pathology (idea 25) ----------------------------------------------

WANG_READ, WANG_RATINGS = 0.877, 0.762
REPEATS, BOOT, MIN_GAIN = 20, 2000, 0.02


def feature_cols(pt):
    cols = ["p", "top2", "lesions", "diameter"] + (["readers"] if "readers" in pt else [])
    return cols


def learned_scores(X, y, seed):
    """Pooled out-of-fold probability of a standardised L2 logistic regression (C=1), five folds by patient
    stratified by diagnosis; fewer folds when a class is that small."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        return None
    out = np.zeros(len(y))
    for trn, tst in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0))   # L2 is its default
        m.fit(X[trn], y[trn])
        out[tst] = m.predict_proba(X[tst])[:, 1]
    return out


def boot_aurocs(y, scores, idx):
    return np.array([auroc(y[i], scores[i]) for i in idx])


def learned_read(q, repeats=REPEATS, boot=BOOT):
    """The learned read against the largest-p read on the patients of `q` (columns y, p, top2, lesions, diameter
    [, readers]). Returns a dict: per-repeat AUROCs, repeat 0's interval, largest p's AUROC and interval, and the paired
    difference (learned minus largest p, on repeat 0's scores, the same resamples) with its interval."""
    y = q.y.to_numpy().astype(int)
    X = q[feature_cols(q)].to_numpy(float)
    runs = [learned_scores(X, y, r) for r in range(repeats)]
    if runs[0] is None:
        return None
    aucs = np.array([auroc(y, s) for s in runs])
    idx = np.random.default_rng(0).integers(0, len(y), (boot, len(y)))
    bl, bp = boot_aurocs(y, runs[0], idx), boot_aurocs(y, q.p.to_numpy(float), idx)
    pa = auroc(y, q.p)
    return {"aucs": aucs, "mean": float(aucs.mean()), "min": float(aucs.min()), "max": float(aucs.max()),
            "r0": float(aucs[0]), "r0_ci": tuple(float(v) for v in np.nanpercentile(bl, [2.5, 97.5])),
            "p_auc": pa, "p_ci": tuple(float(v) for v in np.nanpercentile(bp, [2.5, 97.5])),
            "gain": float(aucs.mean()) - pa, "paired": float(aucs[0] - pa),
            "paired_ci": tuple(float(v) for v in np.nanpercentile(bl - bp, [2.5, 97.5]))}


def kept(res):
    return res["gain"] >= MIN_GAIN and res["paired_ci"][0] > 0


def wang_word(lo, hi):
    return "BEATEN" if lo > WANG_READ else "BEHIND" if hi < WANG_READ else "LEVEL"


def verdict_lines(res):
    """The two closing lines: the keep rule, then the kept read against Wang et al. 2022 (0.877)."""
    k = kept(res)
    lo, hi = res["r0_ci"] if k else res["p_ci"]
    return ["Verdict (idea 25, learned read at least 0.02 over largest p, paired low end above 0): "
            + ("KEPT" if k else "NOT KEPT"),
            "Against Wang et al. 2022 (%.3f): %s" % (WANG_READ, wang_word(lo, hi))]


def learned_report(q, version, subset, k, led, final):
    cols = feature_cols(q)
    if "readers" not in q:
        print("  (the file has no readers column: the readers feature is left out)")
    res = learned_read(q)
    if res is None:
        print("v%d, %s, learned read: too few patients of one class for folds" % (version, subset))
        return
    print("v%d, %s, learned read (%s; logistic L2 C=1, 5 folds x %d repeats): AUROC mean %.3f, range %.3f to %.3f; "
          "repeat 0 %.3f (%.3f to %.3f)" % (version, subset, ", ".join(cols), REPEATS, res["mean"], res["min"],
                                              res["max"], res["r0"], *res["r0_ci"]))
    print("  largest p on the same patients %.3f (%.3f to %.3f); learned minus largest p: mean of repeats %+.3f, "
          "repeat 0 paired %+.3f (%.3f to %.3f)" % (res["p_auc"], *res["p_ci"], res["gain"], res["paired"], *res["paired_ci"]))
    y = q.y.to_numpy().astype(int)
    idx = np.random.default_rng(0).integers(0, len(y), (BOOT, len(y)))
    ra = auroc(y, q.rated)
    rlo, rhi = np.nanpercentile(boot_aurocs(y, q.rated.to_numpy(float), idx), [2.5, 97.5])
    print("  the radiologists' ratings read (patient malignant when any lesion's median is above 3): AUROC %.3f "
          "(%.3f to %.3f); Wang et al. 2022 %.3f for the ratings" % (ra, rlo, rhi, WANG_RATINGS))
    led.number(k + ".learned_auroc_mean", res["mean"]).number(k + ".learned_auroc_min", res["min"])
    led.number(k + ".learned_auroc_max", res["max"]).number(k + ".learned_auroc_r0", res["r0"])
    led.number(k + ".learned_auroc_r0_low", res["r0_ci"][0]).number(k + ".learned_auroc_r0_high", res["r0_ci"][1])
    led.number(k + ".largest_p_auroc", res["p_auc"]).number(k + ".learned_gain_mean", res["gain"])
    led.number(k + ".learned_paired_r0", res["paired"]).number(k + ".learned_paired_low", res["paired_ci"][0])
    led.number(k + ".learned_paired_high", res["paired_ci"][1])
    led.number(k + ".ratings_auroc", ra).number(k + ".ratings_auroc_low", float(rlo)).number(k + ".ratings_auroc_high", float(rhi))
    if final:
        for line in verdict_lines(res):
            print(line)
        led.number(k + ".learned_kept", int(kept(res)), "flag")


def report(version, led, use_all=False, learned=False):
    d = diagnoses()
    scored = patients_of(version, use_all)
    pt = scored.merge(d, on="patient", how="inner")
    pt = pt[pt.diagnosis.isin([1, 2, 3])].copy()
    pt["y"] = (pt.diagnosis >= 2).astype(int)
    rng = np.random.default_rng(0)
    for subset, m in (("all known", np.ones(len(pt), bool)), ("tissue or progression", pt.method.isin([2, 3, 4]).values)):
        q = pt[m]
        a = auroc(q.y, q.p)
        boots = []
        for _ in range(2000):
            r = q.iloc[rng.integers(0, len(q), len(q))]
            boots.append(auroc(r.y, r.p))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        sens = float((q.rated[q.y == 1] == 1).mean()) if (q.y == 1).any() else float("nan")
        spec = float((q.rated[q.y == 0] == 0).mean()) if (q.y == 0).any() else float("nan")
        print("v%d, %s: %d patients (%d malignant, %d benign); model AUROC %.3f (%.3f to %.3f); "
              "the rating labels: sensitivity %.2f, specificity %.2f"
              % (version, subset, len(q), int(q.y.sum()), int((q.y == 0).sum()), a, lo, hi, sens, spec))
        k = "v%d%s.%s" % (version, "_all" if use_all else "", subset.replace(" ", "_"))
        led.number(k + ".patients", len(q), "count").number(k + ".malignant", int(q.y.sum()), "count")
        led.number(k + ".auroc", a).number(k + ".auroc_low", lo).number(k + ".auroc_high", hi)
        led.number(k + ".labels_sensitivity", sens).number(k + ".labels_specificity", spec)
        if learned:
            learned_report(q, version, subset, k, led, final=subset == "all known")
    missing = sorted(set(d.patient[d.diagnosis.isin([1, 2, 3])]) - set(scored.patient))
    print("  %d patients with a known diagnosis have no scored lesion (all rated 3, or none found): %s"
          % (len(missing), ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")))


PAIR_SEED = 60


def pair_verdict(lo, hi):
    return "BETTER" if lo > 0 else "WORSE" if hi < 0 else "INCONCLUSIVE"


def pair_read(q_a, q_b, boot=BOOT, seed=PAIR_SEED):
    """q_a, q_b: patient, y, p (one row per patient) for the two versions. Keeps the patients in both and returns a
    dict: n, malignant, benign, aucs a and b, and the interval of b minus a over `boot` patient resamples (the same
    resample for both versions) with its verdict."""
    q = q_a[["patient", "y", "p"]].merge(q_b[["patient", "p"]], on="patient", suffixes=("_a", "_b"))
    y, pa, pb = q.y.to_numpy().astype(int), q.p_a.to_numpy(float), q.p_b.to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(y), (boot, len(y)))
    delta = boot_aurocs(y, pb, idx) - boot_aurocs(y, pa, idx)
    lo, hi = (float(v) for v in np.nanpercentile(delta, [2.5, 97.5]))
    a, b = auroc(y, pa), auroc(y, pb)
    return {"n": len(y), "malignant": int(y.sum()), "benign": int((y == 0).sum()), "a": a, "b": b, "diff": b - a,
            "low": lo, "high": hi, "verdict": pair_verdict(lo, hi)}


def scored_known(version, d, use_all=False):
    """patient, y, p for the patients with a known diagnosis (1, 2 or 3) scored in `version`'s file."""
    pt = patients_of(version, use_all).merge(d, on="patient", how="inner")
    pt = pt[pt.diagnosis.isin([1, 2, 3])].copy()
    pt["y"] = (pt.diagnosis >= 2).astype(int)
    return pt[["patient", "y", "p"]]


def pair(va, vb, led, use_all=False):
    d = diagnoses()
    r = pair_read(scored_known(va, d, use_all), scored_known(vb, d, use_all))
    print("pair v%d v%d%s: %d patients (%d malignant, %d benign); A %.3f, B %.3f, B - A %+.3f (%+.3f to %+.3f), "
          "%d patient resamples: %s"
          % (va, vb, ", all lesions" if use_all else "", r["n"], r["malignant"], r["benign"], r["a"], r["b"],
             r["diff"], r["low"], r["high"], BOOT, r["verdict"]))
    k = "pair_v%d_v%d%s" % (va, vb, "_all" if use_all else "")
    led.number(k + ".patients", r["n"], "count").number(k + ".malignant", r["malignant"], "count")
    led.number(k + ".auroc_a", r["a"]).number(k + ".auroc_b", r["b"]).number(k + ".diff", r["diff"])
    led.number(k + ".diff_low", r["low"]).number(k + ".diff_high", r["high"])
    led.number(k + ".better", int(r["verdict"] == "BETTER"), "flag")


def selftest():
    """CPU, seconds, no data: the learned read on synthetic patient tables, the verdict and Wang lines' logic."""
    rng = np.random.default_rng(0)
    n = 120
    y = (np.arange(n) % 2).astype(int)
    # size carries the label, the model's p is noise: the learned read should beat largest p and be kept
    q = pd.DataFrame({"y": y, "p": rng.uniform(size=n), "top2": rng.uniform(size=n), "lesions": rng.integers(1, 4, n),
                      "diameter": 8 + 6 * y + rng.normal(0, 2, n), "readers": rng.integers(1, 5, n)})
    res = learned_read(q, repeats=3, boot=300)
    assert len(res["aucs"]) == 3 and res["mean"] > 0.8 and res["p_auc"] < 0.65, res
    assert kept(res), res
    assert verdict_lines(res)[0].endswith(": KEPT") and "Against Wang et al. 2022 (0.877): " in verdict_lines(res)[1]
    # largest p already carries the label and the other features are noise: nothing to gain, not kept
    p = 0.5 * y + rng.uniform(0, 0.45, n)
    q2 = pd.DataFrame({"y": y, "p": p, "top2": rng.uniform(size=n), "lesions": rng.integers(1, 4, n),
                       "diameter": rng.normal(10, 2, n), "readers": rng.integers(1, 5, n)})
    r2 = learned_read(q2, repeats=3, boot=300)
    assert not kept(r2) and verdict_lines(r2)[0].endswith(": NOT KEPT"), r2
    assert verdict_lines(r2)[1].endswith(wang_word(*r2["p_ci"])), "NOT KEPT must score largest p's interval"
    # no readers column: the feature is left out
    assert feature_cols(q.drop(columns="readers")) == ["p", "top2", "lesions", "diameter"]
    assert learned_read(q.drop(columns="readers"), repeats=2, boot=100) is not None
    # the Wang line
    assert wang_word(0.88, 0.95) == "BEATEN" and wang_word(0.80, 0.90) == "LEVEL" and wang_word(0.5, 0.87) == "BEHIND"
    # the keep rule: needs both the 0.02 gain and a paired low end above 0
    base = {"gain": 0.03, "paired_ci": (0.001, 0.1), "r0_ci": (0.9, 0.95), "p_ci": (0.5, 0.6)}
    assert kept(base) and not kept({**base, "gain": 0.019}) and not kept({**base, "paired_ci": (0.0, 0.1)})
    assert verdict_lines(base)[1].endswith("BEATEN") and verdict_lines({**base, "gain": 0.0})[1].endswith("BEHIND")
    # the patient table: top2 is the mean of the two largest, readers the most on any lesion, rated any label 1
    oof = pd.DataFrame({"patient": ["a", "a", "a", "b"], "p": [0.1, 0.9, 0.5, 0.3], "label": [0, np.nan, 1, np.nan],
                        "diameter_mm": [5, 6, 7, 8], "readers": [1, 4, 2, 3]})
    t = patient_table(oof).set_index("patient")
    assert abs(t.loc["a", "top2"] - 0.7) < 1e-12 and t.loc["a", "p"] == 0.9 and t.loc["a", "lesions"] == 3
    assert t.loc["a", "readers"] == 4 and t.loc["a", "diameter"] == 7 and t.loc["a", "rated"] == 1 and t.loc["b", "rated"] == 0
    assert "readers" not in patient_table(oof.drop(columns="readers"))
    # the pair read: the patients in both only, one shared resample, the verdict words
    pq = pd.DataFrame({"patient": list("abcdefgh"), "y": [0, 1] * 4, "p": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]})
    r_same = pair_read(pq, pq, boot=200)
    assert r_same["n"] == 8 and r_same["diff"] == 0 and r_same["low"] == r_same["high"] == 0 and r_same["verdict"] == "INCONCLUSIVE"
    r_gone = pair_read(pq, pq.iloc[:6].assign(p=1 - pq.p[:6]), boot=200)
    assert r_gone["n"] == 6 and r_gone["a"] == 1.0 and r_gone["b"] == 0.0 and r_gone["verdict"] == "WORSE", r_gone
    assert pair_verdict(0.001, 0.2) == "BETTER" and pair_verdict(-0.2, -0.001) == "WORSE" and pair_verdict(0.0, 0.2) == "INCONCLUSIVE"
    print("selftest ok")
    return 0


def main():
    if sys.argv[1:2] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("versions", nargs="*", type=int, default=[2])
    ap.add_argument("--all", action="store_true", help="read oof_all.csv (every lesion) instead of oof.csv")
    ap.add_argument("--learned", action="store_true", help="step B: the patient read learned on pathology")
    ap.add_argument("--pair", type=int, nargs=2, metavar=("A", "B"),
                    help="B minus A on the patients scored in both versions, 2000 shared patient resamples")
    a = ap.parse_args()
    if a.pair:
        with ledger.run("measurement", "LIDC pathology pair v%d v%d%s" % (*a.pair, " all" if a.all else "")) as led:
            led.settings_from({"pair": a.pair, "boot": BOOT, "seed": PAIR_SEED, "table": os.path.basename(XLS),
                               **({"oof_all": True} if a.all else {})})
            pair(*a.pair, led, a.all)
        return
    versions = a.versions or [2]
    name = "LIDC pathology " + " ".join("v%d" % v for v in versions) + (" all" if a.all else "") + (" learned" if a.learned else "")
    with ledger.run("measurement", name) as led:
        led.settings_from({"versions": versions, "boot": 2000, "table": os.path.basename(XLS),
                           **({"oof_all": True} if a.all else {}),
                           **({"learned": {"repeats": REPEATS, "C": 1.0, "folds": 5}} if a.learned else {})})
        for v in versions:
            report(v, led, a.all, a.learned)


if __name__ == "__main__":
    main()
