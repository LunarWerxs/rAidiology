"""Write SCORECARD.md from work/eval/predictions.csv and SPIDER's radiological_gradings.csv,
exactly as METRICS_SPEC.md defines (that document wins over this code).
Usage: python score.py [--predictions P] [--gradings G] [--out SCORECARD.md] [--reference METHOD]"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402
paths.use("spine_mri")
from spider import GRADE_COLS, ITEMS, SEED, path  # noqa: E402

PRED = os.path.join(paths.WORK, "eval", "predictions.csv")
B = 2000
PF_METRICS = ["exact", "within_one", "qwk"]
BIN_METRICS = ["sensitivity", "specificity", "auroc"]
METHOD_ORDER = ["rules", "panel", "grader", "medgemma", "combo"]
LOC_ORDER = ["tss", "spider_masks"]


def m_exact(t, y):
    return float(np.mean(t == y)) if len(t) else np.nan


def m_within_one(t, y):
    return float(np.mean(np.abs(t - y) <= 1)) if len(t) else np.nan


def m_qwk(t, y):
    n = len(t)
    if not n:
        return np.nan
    O = np.zeros((5, 5))
    np.add.at(O, (t.astype(int) - 1, y.astype(int) - 1), 1)
    E = np.outer(O.sum(1), O.sum(0)) / n
    i = np.arange(5)
    W = (i[:, None] - i[None, :]) ** 2 / 16.0
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def m_sens(t, y):
    pos = t == 1
    return float(np.mean(y[pos] == 1)) if pos.any() else np.nan


def m_spec(t, y):
    neg = t == 0
    return float(np.mean(y[neg] == 0)) if neg.any() else np.nan


def m_auroc(t, s):
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    if not n1 or not n0:
        return np.nan
    r = rankdata(s, method="average")
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


FN = {"exact": m_exact, "within_one": m_within_one, "qwk": m_qwk,
      "sensitivity": m_sens, "specificity": m_spec, "auroc": m_auroc}


def bootstrap(patients, fn, *arrays):
    """Percentile CI by patient, per METRICS_SPEC.md; returns (lo, hi, n_kept)."""
    P = np.array(sorted(set(int(p) for p in patients)))
    rows = {p: np.nonzero(patients == p)[0] for p in P}
    draws = np.random.default_rng(SEED).integers(0, len(P), size=(B, len(P)))
    vals = []
    for d in draws:
        idx = np.concatenate([rows[P[k]] for k in d])
        v = fn(*(a[idx] for a in arrays))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def paired_bootstrap(patients, fn, a_arrays, b_arrays):
    P = np.array(sorted(set(int(p) for p in patients)))
    rows = {p: np.nonzero(patients == p)[0] for p in P}
    draws = np.random.default_rng(SEED).integers(0, len(P), size=(B, len(P)))
    vals = []
    for d in draws:
        idx = np.concatenate([rows[P[k]] for k in d])
        v = fn(*(x[idx] for x in a_arrays)) - fn(*(x[idx] for x in b_arrays))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def load(pred_p, grad_p):
    pr = pd.read_csv(pred_p, dtype={"tss_level": str, "nominal_level": str, "drop_reason": str})
    g = pd.read_csv(grad_p)
    truth = pd.DataFrame({"patient": g["Patient"].astype(int), "ivd": g["IVD label"].astype(int)})
    for k, c in GRADE_COLS.items():
        truth["t_" + k] = g[c].astype(int)
    truth["modic_type"] = truth["t_modic"]
    truth["t_modic"] = (truth["t_modic"] > 0).astype(int)
    df = pr.merge(truth, on=["patient", "ivd"], how="left", validate="many_to_one")
    assert df["t_pfirrmann"].notna().all(), "prediction rows without a grading"
    for c in ("drop_reason", "tss_level", "nominal_level"):
        df[c] = df[c].fillna("")
    return df


def scored_items(gdf):
    out = []
    if gdf["pfirrmann_pred"].notna().any():
        out.append("pfirrmann")
    out += [b for b in ITEMS if gdf[f"{b}_pred"].notna().any()]
    return out


def filled(gdf, item, full_gdf=None):
    """(t, y, s|None) for the in-scope rows of gdf, with misses filled per the spec. Score fill
    bounds come from full_gdf (the whole group) so vendor subsets use the same values."""
    full = gdf if full_gdf is None else full_gdf
    t = gdf[f"t_{item}"].to_numpy(int)
    if item == "pfirrmann":
        y = gdf["pfirrmann_pred"].to_numpy(float, copy=True)
        miss = np.isnan(y)
        y[miss] = np.where(t[miss] >= 3, 1, 5)
        return t, y.astype(int), None
    y = gdf[f"{item}_pred"].to_numpy(float, copy=True)
    miss = np.isnan(y)
    y[miss] = 1 - t[miss]
    s = None
    if full[f"{item}_score"].notna().any():
        fs = full[f"{item}_score"].dropna().to_numpy(float)
        s = gdf[f"{item}_score"].to_numpy(float, copy=True)
        sm = np.isnan(s)
        s[sm] = np.where(t[sm] == 1, fs.min() - 1, fs.max() + 1)
    return t, y.astype(int), s


def metric_rows(gdf, full_gdf=None):
    """[(item, metric, value, lo, hi, n_boot)] for one group (or one vendor slice of it)."""
    out = []
    pts = gdf["patient"].to_numpy(int)
    for item in scored_items(full_gdf if full_gdf is not None else gdf):
        t, y, s = filled(gdf, item, full_gdf)
        if item == "pfirrmann":
            pairs = [(m, (t, y)) for m in PF_METRICS]
        else:
            pairs = [("sensitivity", (t, y)), ("specificity", (t, y))]
            if s is not None:
                pairs.append(("auroc", (t, s)))
        for m, arrs in pairs:
            v = FN[m](*arrs)
            lo, hi, nb = bootstrap(pts, FN[m], *arrs)
            out.append((item, m, v, lo, hi, nb))
    return out


def fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def accounting(df, split):
    a = df[(df.method == "rules") & (df.localisation == "tss") & (df.split == split)]
    if a.empty:
        return []
    elig = a[(a.ivd >= 1) & ~a.drop_reason.isin(["no T2", "not in the T2's field of view"])]
    mt = elig[elig.matched == 1]
    sc = a[a.in_scope == 1]
    q = [("graded_rows", len(a)), ("patients", a.patient.nunique()),
         ("excluded_no_t2_rows", int((a.drop_reason == "no T2").sum())),
         ("excluded_no_t2_patients", a[a.drop_reason == "no T2"].patient.nunique()),
         ("ivd0_rows", int((a.drop_reason == "IVD label 0").sum())),
         ("outside_t2_view_rows", int((a.drop_reason == "not in the T2's field of view").sum())),
         ("eligible", len(elig)), ("matched", len(mt)),
         ("match_rate", len(mt) / len(elig) if len(elig) else np.nan),
         ("level_differs", int((mt.tss_level != mt.nominal_level).sum())),
         ("level_mislabel_rate", float((mt.tss_level != mt.nominal_level).mean()) if len(mt) else np.nan),
         ("dropped_matched_outside", int((a.drop_reason == "matched outside T12-L1..L5-S1").sum())),
         ("dropped_unmatched_outside", int((a.drop_reason == "unmatched, nominal level outside T12-L1..L5-S1").sum())),
         ("scored", len(sc)), ("scored_unmatched", int((sc.matched == 0).sum()))]
    for k in range(4):
        q.append((f"modic_type_{k}", int((sc.modic_type == k).sum())))
    return q


def order_key(method, loc):
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else 99, LOC_ORDER.index(loc) if loc in LOC_ORDER else 9)


def tables(df, split, reference=None):
    """Markdown lines for one split's accounting, overall, per-vendor and paired tables."""
    L = []
    acc = accounting(df, split)
    if acc:
        L += ["| split | quantity | value |", "|---|---|---|"]
        for k, v in acc:
            L.append(f"| {split} | {k} | {v if isinstance(v, (int, np.integer)) else fmt(v)} |")
        L.append("")
    sdf = df[(df.split == split) & (df.in_scope == 1)]
    groups = sorted(sdf.groupby(["method", "localisation"]).groups, key=lambda k: order_key(*k))
    L += ["| split | item | metric | method | localisation | n_discs | n_patients | value | ci_low | ci_high | n_boot |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    per = {}
    for method, loc in groups:
        g = sdf[(sdf.method == method) & (sdf.localisation == loc)]
        for item, m, v, lo, hi, nb in metric_rows(g):
            per[(method, loc, item, m)] = v
            L.append(f"| {split} | {item} | {m} | {method} | {loc} | {len(g)} | {g.patient.nunique()} | "
                     f"{fmt(v)} | {fmt(lo)} | {fmt(hi)} | {nb} |")
    L += ["", "| split | vendor | item | metric | method | localisation | n_discs | n_patients | value | ci_low | ci_high | n_boot |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for method, loc in groups:
        if loc != "tss":
            continue
        g = sdf[(sdf.method == method) & (sdf.localisation == loc)]
        for vendor in sorted(g.vendor.unique()):
            gv = g[g.vendor == vendor]
            for item, m, v, lo, hi, nb in metric_rows(gv, full_gdf=g):
                L.append(f"| {split} | {vendor} | {item} | {m} | {method} | {loc} | {len(gv)} | {gv.patient.nunique()} | "
                         f"{fmt(v)} | {fmt(lo)} | {fmt(hi)} | {nb} |")
    if reference:
        L += ["", "| split | item | metric | method | reference | n_discs | n_patients | difference | ci_low | ci_high | n_boot |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        ref = sdf[(sdf.method == reference) & (sdf.localisation == "tss")]
        for method, loc in groups:
            if loc != "tss" or method == reference:
                continue
            g = sdf[(sdf.method == method) & (sdf.localisation == "tss")]
            keys = set(zip(g.patient, g.ivd)) & set(zip(ref.patient, ref.ivd))
            ga = g[[k in keys for k in zip(g.patient, g.ivd)]].sort_values(["patient", "ivd"])
            gb = ref[[k in keys for k in zip(ref.patient, ref.ivd)]].sort_values(["patient", "ivd"])
            ta, ya, _ = filled(ga, "pfirrmann", g)
            tb, yb, _ = filled(gb, "pfirrmann", ref)
            pts = ga.patient.to_numpy(int)
            d = m_qwk(ta, ya) - m_qwk(tb, yb)
            lo, hi, nb = paired_bootstrap(pts, m_qwk, (ta, ya), (tb, yb))
            L.append(f"| {split} | pfirrmann | qwk | {method} | {reference} | {len(ga)} | {len(set(pts))} | "
                     f"{fmt(d)} | {fmt(lo)} | {fmt(hi)} | {nb} |")
    return L, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=PRED)
    ap.add_argument("--gradings", default=path("radiological_gradings.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "SCORECARD.md"))
    ap.add_argument("--reference", default=None, help="reference baseline for the paired test differences")
    ap.add_argument("--header", default=os.path.join(HERE, "scorecard_header.md"))
    a = ap.parse_args()
    df = load(a.predictions, a.gradings)
    out = [open(a.header, encoding="utf-8").read().rstrip(), ""]
    out += ["## Development (5-fold, out-of-fold)", ""]
    lines, _ = tables(df, "dev")
    out += lines + [""]
    out += ["## Held-out test", ""]
    if (df.split == "test").any():
        lines, _ = tables(df, "test", a.reference)
        out += lines
    else:
        out.append("Not yet run. The test split is opened once, in the final evaluation, for every method together.")
    open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
