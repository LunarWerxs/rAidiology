"""The X-ray scorecard: every number in SCORECARD.md, computed from work/eval/xray/predictions.csv
and the two datasets' own index files, exactly as METRICS_SPEC.md defines them.

Nothing here decides anything. It reports: each method, each dataset, each pre-declared stratum,
five metrics, with a 95% patient-level bootstrap interval on every one. `final_eval.py` adds the
paired differences and the ship decision, and `recompute.py`, written by an agent that has never
seen this file, recomputes the whole card from METRICS_SPEC.md alone.

Why it is written the way it is: a patient bootstrap resampling 6,091 patients, 2,000 times, for
each of a couple of hundred numbers is far too slow row by row, so every metric is computed from
per-patient aggregates and one matrix product, which is the same arithmetic in a different order
(METRICS_SPEC.md, "An equivalent fast form"). The draws depend only on the seed and the number of
patients, so they are cached and reused, which the specification's recipe allows because a fresh
Generator with the same seed and the same size always produces the same draws.

Usage: python score.py [--reference <method>] [--out SCORECARD.md]
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import ledger  # noqa: E402
import datasets  # noqa: E402
import preds  # noqa: E402

METRICS = ["auroc", "sensitivity", "specificity", "balanced_accuracy", "kappa"]
BOOT = 2000
SEED = datasets.SEED
CHUNK = 250                      # resamples per chunk when AUROC is bootstrapped, to bound memory
HEADER = os.path.join(HERE, "scorecard_header.md")
OUT = os.path.join(HERE, "SCORECARD.md")



def truth_table(dataset):
    """image -> truth and one boolean column per stratum, read from the dataset's own index."""
    df = datasets.index(dataset)
    st = datasets.strata(dataset)
    out = pd.DataFrame({"image": df.image.values, "truth": df.truth.values})
    for name, mask in st.items():
        out["st_" + name] = mask.values
    return out.set_index("image"), list(st)


def load(predictions=None):
    p = predictions or os.path.join(preds.WORK, "predictions.csv")
    if not os.path.exists(p):
        raise SystemExit("%s is missing: run the methods, then preds.collect()." % p)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    missing = [c for c in preds.COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit("%s is missing column(s) %s" % (p, missing))
    bad = set(df.dataset) - set(datasets.NAMES)
    if bad:
        raise SystemExit("%s names unknown dataset(s) %s" % (p, sorted(bad)))
    return df


class Group:
    """One (method, dataset, split): its in-scope rows, with the misses filled as the spec says."""

    def __init__(self, rows, truth, strata_names):
        self.rows = rows
        self.strata_names = strata_names
        t = truth.truth.reindex(rows.image).values
        if np.isnan(t.astype(float)).any():
            raise SystemExit("predictions.csv names images that are not in the dataset's index")
        self.truth = t.astype(int)
        self.patient = rows.patient.str.strip().values
        self.in_stratum = {n: truth["st_" + n].reindex(rows.image).values.astype(bool)
                           for n in strata_names}
        pred_empty = rows.fracture_pred.values == ""
        self.scored = bool((~pred_empty).any())
        y = np.where(pred_empty, 1 - self.truth,
                     pd.to_numeric(rows.fracture_pred.replace("", "0")).values.astype(int))
        self.y = y.astype(int)
        s_empty = rows.fracture_score.values == ""
        self.has_score = bool((~s_empty).any())
        if self.has_score:
            raw = pd.to_numeric(rows.fracture_score.replace("", "nan")).values.astype(float)
            lo, hi = np.nanmin(raw), np.nanmax(raw)
            self.s = np.where(s_empty, np.where(self.truth == 1, lo - 1.0, hi + 1.0), raw)
        else:
            self.s = None

    def select(self, stratum):
        return np.ones(len(self.rows), bool) if stratum == "all" else self.in_stratum[stratum]


def groups(df):
    """{(method, dataset, split): Group}, and the stratum names per dataset."""
    tt, names = {}, {}
    for d in datasets.NAMES:
        tt[d], names[d] = truth_table(d)
    out = {}
    for (m, d, sp), rows in df.groupby(["method", "dataset", "split"], sort=True):
        inside = rows[rows.in_scope.str.strip() == "1"]
        if inside.empty:
            continue
        out[(m, d, sp)] = Group(inside.reset_index(drop=True), tt[d], names[d])
    return out, names



def confusion(t, y, w=None):
    """TP, FP, FN, TN, weighting each row by w (its bootstrap multiplicity) when given."""
    w = np.ones(len(t)) if w is None else w
    return (w[(t == 1) & (y == 1)].sum(), w[(t == 0) & (y == 1)].sum(),
            w[(t == 1) & (y == 0)].sum(), w[(t == 0) & (y == 0)].sum())


def from_counts(metric, c):
    """metric from a (..., 4) array of TP, FP, FN, TN; nan where it is undefined."""
    tp, fp, fn, tn = (c[..., i].astype(float) for i in range(4))
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        if metric == "sensitivity":
            return sens
        if metric == "specificity":
            return spec
        if metric == "balanced_accuracy":
            return (sens + spec) / 2.0
        if metric == "kappa":
            po = np.where(n > 0, (tp + tn) / n, np.nan)
            pe = np.where(n > 0, ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n), np.nan)
            return np.where(np.isclose(pe, 1.0), np.nan, (po - pe) / (1.0 - pe))
    raise SystemExit("from_counts does not know %r" % metric)


def auroc_from_bins(pos, neg):
    """AUROC from per-score-value counts, the value bins ascending. pos, neg: (..., V)."""
    n_pos, n_neg = pos.sum(-1), neg.sum(-1)
    below = np.cumsum(neg, axis=-1) - neg                 # negatives strictly below this value
    u = (pos * (below + 0.5 * neg)).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((n_pos > 0) & (n_neg > 0), u / (n_pos * n_neg), np.nan)



_draws = collections.OrderedDict()


def multiplicity(n):
    """(BOOT, n) float32: how often each of n patients appears in each resample. Cached by n: a
    fresh Generator(SEED) with the same size always draws the same thing."""
    if n in _draws:
        _draws.move_to_end(n)
        return _draws[n]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n, size=(BOOT, n))
    m = np.zeros((BOOT, n), np.float32)
    flat = (draws + np.arange(BOOT)[:, None] * n).ravel()
    np.add.at(m.reshape(-1), flat, 1.0)
    _draws[n] = m
    while len(_draws) > 4:
        _draws.popitem(last=False)
    return m


def patient_index(patients):
    """P (sorted distinct, as stripped strings) and each row's position in it."""
    P = sorted(set(patients))
    pos = {p: i for i, p in enumerate(P)}
    return P, np.array([pos[p] for p in patients], np.int64)


def interval(values):
    v = values[~np.isnan(values)]
    if v.size == 0:
        return None, None, 0
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), int(v.size)


def number(group, stratum, metric):
    """(value, ci_low, ci_high, n_boot, n_images, n_patients), or None when not reported."""
    keep = group.select(stratum)
    if not keep.any():
        return None
    t, y = group.truth[keep], group.y[keep]
    pats = group.patient[keep]
    P, row_unit = patient_index(pats)
    n_images, n_patients = int(keep.sum()), len(P)
    M = multiplicity(n_patients)
    if metric == "auroc":
        if not group.has_score:
            return None
        s = group.s[keep]
        vals, bins = np.unique(s, return_inverse=True)
        V = len(vals)
        Spos = sparse.csr_matrix((np.ones((t == 1).sum(), np.float32),
                                  (row_unit[t == 1], bins[t == 1])), shape=(n_patients, V))
        Sneg = sparse.csr_matrix((np.ones((t == 0).sum(), np.float32),
                                  (row_unit[t == 0], bins[t == 0])), shape=(n_patients, V))
        value = auroc_from_bins(np.asarray(Spos.sum(0)).ravel(), np.asarray(Sneg.sum(0)).ravel())
        boots = []
        for a in range(0, BOOT, CHUNK):
            m = M[a:a + CHUNK]
            boots.append(auroc_from_bins(np.asarray(Spos.T.dot(m.T).T),
                                         np.asarray(Sneg.T.dot(m.T).T)))
        b = np.concatenate(boots)
    else:
        if not group.scored:
            return None
        C = np.zeros((n_patients, 4), np.float32)
        cell = (t == 1) * 2 + (y == 1)                    # 3 = TP, 1 = FP, 2 = FN, 0 = TN
        to_col = {3: 0, 1: 1, 2: 2, 0: 3}
        for k, col in to_col.items():
            np.add.at(C[:, col], row_unit[cell == k], 1.0)
        value = from_counts(metric, C.sum(0))
        b = from_counts(metric, M @ C)
    lo, hi, n_boot = interval(np.atleast_1d(b))
    return (None if np.isnan(value) else float(value)), lo, hi, n_boot, n_images, n_patients


def paired(a, b, stratum, metric):
    """a's metric minus b's, on the rows in scope for both. Each keeps its own fill values."""
    shared = sorted(set(a.rows.image) & set(b.rows.image))
    if not shared:
        return None
    want = set(shared)
    sides = []
    for g in (a, b):
        keep = g.select(stratum) & np.array([i in want for i in g.rows.image])
        if not keep.any():
            return None
        order = np.argsort(g.rows.image.values[keep], kind="stable")
        sides.append({"t": g.truth[keep][order], "y": g.y[keep][order],
                      "p": g.patient[keep][order],
                      "s": None if g.s is None else g.s[keep][order],
                      "img": g.rows.image.values[keep][order]})
    if not np.array_equal(sides[0]["img"], sides[1]["img"]):
        return None
    P, row_unit = patient_index(sides[0]["p"])
    n_images, n_patients = len(sides[0]["img"]), len(P)
    M = multiplicity(n_patients)
    if metric == "auroc":
        if sides[0]["s"] is None or sides[1]["s"] is None:
            return None
        per = []
        for side in sides:
            t, s = side["t"], side["s"]
            vals, bins = np.unique(s, return_inverse=True)
            V = len(vals)
            Sp = sparse.csr_matrix((np.ones((t == 1).sum(), np.float32),
                                    (row_unit[t == 1], bins[t == 1])), shape=(n_patients, V))
            Sn = sparse.csr_matrix((np.ones((t == 0).sum(), np.float32),
                                    (row_unit[t == 0], bins[t == 0])), shape=(n_patients, V))
            full = auroc_from_bins(np.asarray(Sp.sum(0)).ravel(), np.asarray(Sn.sum(0)).ravel())
            chunks = [auroc_from_bins(np.asarray(Sp.T.dot(M[i:i + CHUNK].T).T),
                                      np.asarray(Sn.T.dot(M[i:i + CHUNK].T).T))
                      for i in range(0, BOOT, CHUNK)]
            per.append((full, np.concatenate(chunks)))
        value = per[0][0] - per[1][0]
        b = per[0][1] - per[1][1]
    else:
        per = []
        for side in sides:
            C = np.zeros((n_patients, 4), np.float32)
            cell = (side["t"] == 1) * 2 + (side["y"] == 1)
            for k, col in {3: 0, 1: 1, 2: 2, 0: 3}.items():
                np.add.at(C[:, col], row_unit[cell == k], 1.0)
            per.append((from_counts(metric, C.sum(0)), from_counts(metric, M @ C)))
        value = per[0][0] - per[1][0]
        b = per[0][1] - per[1][1]
    lo, hi, n_boot = interval(np.atleast_1d(b))
    return (None if np.isnan(value) else float(value)), lo, hi, n_boot, n_images, n_patients



def accounting(df, dataset, split, strata_names):
    g = df[(df.method == "prevalence") & (df.dataset == dataset) & (df.split == split)]
    if g.empty:
        return []
    truth, _ = truth_table(dataset)
    inside = g[g.in_scope.str.strip() == "1"]
    t = truth.truth.reindex(inside.image).values.astype(int)
    rows = [("rows", len(g)), ("patients", g.patient.str.strip().nunique()),
            ("in_scope_rows", len(inside)),
            ("unreadable_rows", int((g.drop_reason.str.strip() == preds.UNREADABLE).sum())),
            ("fracture_rows", int(t.sum())),
            ("prevalence", (t.mean() if len(t) else float("nan"), 3))]
    for name in [n for n in strata_names if n != "all"]:
        mask = truth["st_" + name].reindex(inside.image).values.astype(bool)
        rows.append(("%s_rows" % name, int(mask.sum())))
        rows.append(("%s_prevalence" % name,
                     (t[mask].mean() if mask.any() else float("nan"), 3)))
    if split == "dev":
        for f in range(5):
            rows.append(("fold_%d_patients" % f,
                         g[g.fold.str.strip() == str(f)].patient.str.strip().nunique()))
    return [(dataset, split, k, v) for k, v in rows]



def fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.3f" % v


def cell(v):
    if isinstance(v, tuple):
        return fmt(v[0])
    return str(v)


def table1(lines, rows):
    lines += ["| dataset | split | quantity | value |", "|---|---|---|---|"]
    for d, sp, k, v in rows:
        lines.append("| %s | %s | %s | %s |" % (d, sp, k, cell(v)))
    lines.append("")


def table2(lines, rows):
    lines += ["| dataset | split | stratum | metric | method | n_images | n_patients | value | ci_low | ci_high | n_boot |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %d | %d | %s | %s | %s | %d |" % r)
    lines.append("")


def table3(lines, rows):
    lines += ["| dataset | split | stratum | metric | method | reference | n_images | n_patients | difference | ci_low | ci_high | n_boot |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %d | %d | %s | %s | %s | %d |" % r)
    lines.append("")


def stratum_order(names):
    """`all` first, then the rest in the dataset's own order. datasets.strata already yields `all`
    first, so the filter is what keeps it from being scored twice."""
    return ["all"] + [n for n in names if n != "all"]


def metric_rows(gs, names, split):
    out = []
    for (m, d, sp), g in sorted(gs.items()):
        if sp != split:
            continue
        for st in stratum_order(names[d]):
            for metric in METRICS:
                r = number(g, st, metric)
                if r is None:
                    continue
                v, lo, hi, nb, ni, npat = r
                out.append((d, sp, st, metric, m, ni, npat, fmt(v), fmt(lo), fmt(hi), nb))
    out.sort(key=lambda r: (r[0], r[1], stratum_order(names[r[0]]).index(r[2]),
                            METRICS.index(r[3]), r[4]))
    return out


def paired_rows(gs, names, reference):
    out = []
    for d in datasets.NAMES:
        ref = gs.get((reference, d, "test"))
        if ref is None:
            continue
        for st in stratum_order(names[d]):
            for (m, dd, sp), g in sorted(gs.items()):
                if dd != d or sp != "test" or m == reference:
                    continue
                r = paired(g, ref, st, "auroc")
                if r is None:
                    continue
                v, lo, hi, nb, ni, npat = r
                out.append((d, "test", st, "auroc", m, reference, ni, npat, fmt(v), fmt(lo), fmt(hi), nb))
    out.sort(key=lambda r: (r[0], stratum_order(names[r[0]]).index(r[2]), r[4]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None, help="the baseline the paired differences are against")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--predictions", default=None)
    a = ap.parse_args()
    t0 = time.time()
    with ledger.run("score", "scorecard", note="reference=%s" % (a.reference or "none")) as led:
        df = load(a.predictions)
        gs, names = groups(df)
        splits = sorted(set(df.split))
        lines = [open(HEADER, encoding="utf-8").read().rstrip(), ""]
        for sp in ("dev", "test"):
            if sp not in splits:
                if sp == "test":
                    lines += ["## Held-out test", "", "Not yet run: `final_eval.py` opens it once.", ""]
                continue
            lines += ["## %s" % ("Development folds (out of fold)" if sp == "dev" else "Held-out test"),
                      "", "### Data", ""]
            acc = []
            for d in datasets.NAMES:
                acc += accounting(df, d, sp, names[d])
            table1(lines, acc)
            lines += ["### Scores", ""]
            rows = metric_rows(gs, names, sp)
            table2(lines, rows)
            for d, s, st, metric, m, _ni, _np, v, _lo, _hi, _nb in rows:
                led.number("%s.%s.%s.%s.%s" % (d, s, st, metric, m), v)
            print("%s: %d accounting rows, %d scored numbers (%.0fs)"
                  % (sp, len(acc), len(rows), time.time() - t0), flush=True)
        if a.reference and "test" in splits:
            lines += ["### Difference from the reference baseline", "",
                      "Each method's AUROC minus **%s**'s, on the images both scored, patients resampled."
                      % a.reference, ""]
            pr = paired_rows(gs, names, a.reference)
            table3(lines, pr)
            for d, s, st, metric, m, ref, _ni, _np, diff, lo, _hi, _nb in pr:
                led.number("diff.%s.%s.%s.%s.%s-%s" % (d, s, st, metric, m, ref), diff)
                led.number("diff_ci_low.%s.%s.%s.%s.%s-%s" % (d, s, st, metric, m, ref), lo)
            print("paired: %d rows (%.0fs)" % (len(pr), time.time() - t0), flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        print("%s written in %.0fs" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
