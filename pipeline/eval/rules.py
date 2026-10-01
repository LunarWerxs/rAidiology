"""Baseline (a), the rules: today's measurements (metrics.py) with thresholds fitted on development
folds only.

- Pfirrmann from the disc-to-CSF signal ratio: four cut-points c1 > c2 > c3 > c4, grade =
  1 + (number of cut-points above the ratio). Fitted by matching the training grade proportions,
  then coordinate ascent on quadratic weighted kappa over the training ratios' 1st..99th
  percentiles.
- Narrowing from the disc height index (height / front-to-back length): score = -index, threshold
  at the training maximum of Youden's J (sensitivity + specificity - 1).
No other item has a rule today, so the rules predict nothing else.

Development rows are out-of-fold: fold k is predicted with thresholds fitted on the other four.
Test rows (final evaluation) use thresholds fitted on all development folds.
Both localisations are measured: discs located by TotalSpineSeg (`tss`) and by SPIDER's own masks
(`spider_masks`, relabelled at SPIDER's bottom-up levels by prepare.py).
Usage: python rules.py <dev|test> [work_dir]   (test needs dev measured first)"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from metrics import measure  # noqa: E402
from prepare import WORK  # noqa: E402
from score import m_qwk  # noqa: E402
from spider import guard_test, ITEMS  # noqa: E402

LOCS = {"tss": ("tss/step2_output", "tss_level"), "spider_masks": ("spiderlab", "nominal_level")}


def measured(split, work):
    """Per-patient metrics.py output for both localisations, cached in <split>/metrics/."""
    base = os.path.join(work, split)
    m = pd.read_csv(os.path.join(base, "matches.csv"), dtype=str, keep_default_na=False)
    out = {}
    for p in sorted(set(m[m.drop_reason != "no T2"].patient), key=int):
        for loc, (sub, _) in LOCS.items():
            cache = os.path.join(base, "metrics", f"{p}_{loc}.json")
            if not os.path.exists(cache):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                try:
                    res = measure(os.path.join(base, "t2", f"{p}.nii.gz"), os.path.join(base, sub, f"{p}.nii.gz"))
                except Exception as e:  # a volume metrics.py cannot measure: every disc is a miss
                    res = {"error": f"{type(e).__name__}: {e}", "levels": []}
                json.dump(res, open(cache, "w"), indent=1)
            out[(int(p), loc)] = json.load(open(cache))
    return out


def features(split, work):
    base = os.path.join(work, split)
    m = pd.read_csv(os.path.join(base, "matches.csv"), keep_default_na=False)
    meas = measured(split, work)
    rows = []
    for _, r in m.iterrows():
        for loc, (_, lev_col) in LOCS.items():
            ratio = hidx = np.nan
            res = meas.get((int(r.patient), loc))
            ok = r.in_scope == 1 and res and (loc != "tss" or r.matched == 1)
            if ok:
                lv = {x["level"]: x for x in res["levels"]}.get(r[lev_col])
                if lv:
                    ratio = lv.get("disc_to_csf_signal") if lv.get("disc_to_csf_signal") is not None else np.nan
                    if lv.get("disc_height_mm") and lv.get("disc_ap_length_mm"):
                        hidx = lv["disc_height_mm"] / lv["disc_ap_length_mm"]
            d = r.to_dict()
            d.update(localisation=loc, ratio=ratio, hidx=hidx)
            rows.append(d)
    return pd.DataFrame(rows)


def predict_grade(r, cuts):
    return 1 + (np.asarray(r)[:, None] < np.asarray(cuts)[None, :]).sum(1)


def fit_cuts(r, g):
    r, g = np.asarray(r, float), np.asarray(g, int)
    cum = np.cumsum([np.mean(g == k) for k in (1, 2, 3, 4)])
    cuts = [float(np.quantile(r, 1 - c)) for c in cum]
    cand = np.unique(np.percentile(r, np.arange(1, 100)))
    best = m_qwk(g, predict_grade(r, cuts))
    for _ in range(10):
        changed = False
        for j in range(4):
            hi = cuts[j - 1] if j > 0 else np.inf
            lo = cuts[j + 1] if j < 3 else -np.inf
            for c in cand[(cand < hi) & (cand > lo)]:
                trial = cuts[:j] + [float(c)] + cuts[j + 1:]
                q = m_qwk(g, predict_grade(r, trial))
                if q > best + 1e-12:
                    best, cuts, changed = q, trial, True
        if not changed:
            break
    return cuts


def fit_youden(s, t):
    s, t = np.asarray(s, float), np.asarray(t, int)
    best, thr = -np.inf, None
    for c in np.unique(s):
        y = s >= c
        j = np.mean(y[t == 1]) + np.mean(~y[t == 0]) - 1
        if j > best + 1e-12:
            best, thr = j, float(c)
    return thr


def fit(train):
    a = train.dropna(subset=["ratio"])
    b = train.dropna(subset=["hidx"])
    return {"cuts": fit_cuts(a.ratio, a.t_pfirrmann), "narrow_thr": fit_youden(-b.hidx, b.t_narrowing)}


def apply(df, f):
    out = df.copy()
    out["pfirrmann_pred"] = np.where(df.ratio.notna(), predict_grade(df.ratio.fillna(0), f["cuts"]), np.nan)
    out["narrowing_score"] = -df.hidx
    out["narrowing_pred"] = np.where(df.hidx.notna(), (-df.hidx >= f["narrow_thr"]).astype(float), np.nan)
    return out


def with_truth(df):
    from spider import grades
    g = grades().rename(columns={k: "t_" + k for k in ["pfirrmann", *ITEMS]})
    return df.merge(g, on=["patient", "ivd"], how="left")


def main(split, work=WORK):
    guard_test(split)
    feats = with_truth(features(split, work))
    feats["split"] = split
    preds, fits = [], {}
    for loc in LOCS:
        F = feats[feats.localisation == loc]
        if split == "dev":
            for k in range(5):
                tr = F[(F.in_scope == 1) & (F.fold.astype(str) != str(k)) & (F.fold.astype(str) != "")]
                f = fit(tr)
                fits[f"{loc} fold {k}"] = f
                preds.append(apply(F[F.fold.astype(str) == str(k)], f))
        else:
            dev = with_truth(features("dev", work))
            f = fit(dev[(dev.localisation == loc) & (dev.in_scope == 1)])
            fits[f"{loc} all dev"] = f
            preds.append(apply(F, f))
    P = pd.concat(preds, ignore_index=True)
    P["method"] = "rules"
    cols = ["method", "localisation", "split", "fold", "patient", "ivd", "vendor", "nominal_level", "tss_level",
            "dice", "matched", "in_scope", "drop_reason", "pfirrmann_pred"]
    for b in ITEMS:
        if f"{b}_pred" not in P:
            P[f"{b}_pred"] = np.nan
        if f"{b}_score" not in P:
            P[f"{b}_score"] = np.nan
        cols += [f"{b}_pred", f"{b}_score"]
    P = P[cols].sort_values(["localisation", "patient", "ivd"])
    out = os.path.join(work, "..", f"pred_rules_{split}.csv")
    P.to_csv(out, index=False)
    json.dump(fits, open(os.path.join(work, "..", f"rules_fits_{split}.json"), "w"), indent=1)
    print("wrote", os.path.normpath(out), len(P), "rows;", {k: [round(c, 3) for c in v["cuts"]] for k, v in fits.items()})


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or [WORK]))
