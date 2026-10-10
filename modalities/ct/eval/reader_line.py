"""Score a lung candidates file against the doctors' line (each radiologist judged by the other three).

The candidates (seriesuid, coordX, coordY, coordZ in world mm, a score column) are classified once per
(scan, left-out reader, key) by lidc_readers.classify, the code the doctors were scored by, with
LUNA16's hit rule. A lesion's best score is the highest score among candidates that hit it; the false
alarms' scores are those of candidates classified as false alarms.

The gate (2-of-3 key): the system may raise as many false alarms, summed over the same 3,544 pairs, as
the doctors did (k). Its threshold is just above the (k+1)-th highest false-alarm score, so a lesion
is caught when its best score is strictly above it. A paired patient bootstrap resamples patients and
recomputes the doctors' numbers, k and the system's caught count in each resample.

    python modalities/ct/eval/reader_line.py --candidates <csv> [--column p2] [--name text] [--boot 1000]
    python modalities/ct/eval/reader_line.py --selftest
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ledger, paths  # noqa: E402
import lidc_readers as lr  # noqa: E402

RULE = "luna16"
ALLOWANCES = (0.125, 0.25, 0.5, 1, 2, 4, 8)
KEY_TEXT = {"2of3": "2-of-3 key", "3of3": "3-of-3 key"}
ANNOTATIONS = os.path.join(paths.DATASETS, "public", "luna16", "annotations.csv")
EMPTY_KEYS = pd.DataFrame(columns=lr.KEY_COLS)


def pair_arrays(xyz, score, keys_k, key):
    """(best score per reference lesion (-inf = never caught), false-alarm scores, ignored count) of one pair."""
    cls = lr.classify(np.column_stack([xyz, np.full(len(xyz), np.nan)]) if len(xyz) else np.zeros((0, 4)),
                      keys_k, key, RULE)
    ref = np.flatnonzero(keys_k.role.isin(lr.KEYS[key]).to_numpy())
    pos = np.full(len(keys_k), -1)
    pos[ref] = np.arange(len(ref))
    best = np.full(len(ref), -np.inf)
    hit = cls >= 0
    np.maximum.at(best, pos[cls[hit]], np.asarray(score, dtype=float)[hit])
    return best, np.asarray(score, dtype=float)[cls == -2], int((cls == -1).sum())


class Gate:
    """System lesions (best score, patient) and false alarms (score, patient) of one key."""

    def __init__(self, best, bpat, fa, fpat):
        order = np.argsort(-fa, kind="stable")
        self.fa, self.fpat = fa[order], fpat[order]
        self.best, self.bpat = best, bpat

    def threshold(self, w, k):
        """Just above the (k+1)-th highest false-alarm score under patient weights w (-inf if fewer)."""
        cum = np.cumsum(w[self.fpat])
        i = int(np.searchsorted(cum, k + 1))
        return -np.inf if i >= len(cum) else float(self.fa[i])

    def caught(self, w, k):
        """(lesions with best score strictly above the threshold, weighted; the threshold)."""
        t = self.threshold(w, k)
        return int(w[self.bpat][self.best > t].sum()), t


def collect(cand, column, pairs, keys, pat_code):
    """Per key: Gate inputs over every pair. pairs: list of (uid, reader)."""
    by = {u: (g[["coordX", "coordY", "coordZ"]].to_numpy(float), g[column].to_numpy(float))
          for u, g in cand.groupby("seriesuid")}
    kk = {ix: g.reset_index(drop=True) for ix, g in keys.groupby(["series_uid", "left_out_reader"])}
    none = (np.zeros((0, 3)), np.zeros(0))
    out = {key: dict(best=[], bpat=[], bread=[], fa=[], fpat=[], fread=[]) for key in lr.KEYS}
    for uid, r in pairs:
        xyz, sc = by.get(uid, none)
        k = kk.get((uid, r), EMPTY_KEYS)
        for key in lr.KEYS:
            b, f, _ = pair_arrays(xyz, sc, k, key)
            o = out[key]
            o["best"].append(b)
            o["bpat"].append(np.full(len(b), pat_code[uid]))
            o["bread"].append(np.full(len(b), r))
            o["fa"].append(f)
            o["fpat"].append(np.full(len(f), pat_code[uid]))
            o["fread"].append(np.full(len(f), r))
    return {key: {n: np.concatenate(v) for n, v in o.items()} for key, o in out.items()}


def make_gate(c, reader=None):
    mb = np.ones(len(c["best"]), bool) if reader is None else c["bread"] == reader
    mf = np.ones(len(c["fa"]), bool) if reader is None else c["fread"] == reader
    return Gate(c["best"][mb], c["bpat"][mb], c["fa"][mf], c["fpat"][mf])


def bootstrap(gate, doc, n_pat, boot, seed):
    """System minus doctors' sensitivity (points) over `boot` patient resamples. doc: per-patient ref, hits, fa."""
    ix = np.random.default_rng(seed).integers(0, n_pat, (boot, n_pat))
    diff = np.full(boot, np.nan)
    for i in range(boot):
        w = np.bincount(ix[i], minlength=n_pat)
        ref = w @ doc["ref"]
        if ref > 0:
            diff[i] = 100 * (gate.caught(w, int(w @ doc["fa"]))[0] - w @ doc["hits"]) / ref
    return diff[~np.isnan(diff)]


def verdict(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    word = "BEATEN" if lo > 0 else "BEHIND" if hi < 0 else "LEVEL"
    return word, float(np.mean(d)), float(lo), float(hi)


def doctor_pairs(scores, key, pairs, pat_code):
    """Doctors' per-pair frame (aligned with `pairs`) and per-patient sums for one key."""
    s = scores[(scores.key == key) & (scores.rule == RULE)].set_index(["series_uid", "reader"])
    s = s.loc[pairs]
    pat = np.array([pat_code[u] for u, _ in pairs])
    n = max(pat_code.values()) + 1
    per = {c: np.bincount(pat, weights=s[c].to_numpy(float), minlength=n) for c in ("ref", "hits", "fa")}
    return s, per


def scope(scores, luna):
    s = scores[scores.series_uid.isin(luna) & (scores.rule == RULE)]
    pairs = sorted(set(zip(s.series_uid, s.reader)))
    pat = dict(zip(s.series_uid, s.patient))
    codes = {p: i for i, p in enumerate(sorted(set(pat.values())))}
    return pairs, {u: codes[p] for u, p in pat.items()}


def report(cand, column, name, boot, seed, tag, led=None):
    mk, sc = lr.load_marks()
    keys, scores = lr.run(mk, sc)
    luna = set(pd.read_csv(lr.LUNA_SUBSETS).seriesuid)
    pairs, pat_code = scope(scores, luna)
    n_pairs, n_pat = len(pairs), len(set(pat_code.values()))
    cand = cand[cand.seriesuid.isin({u for u, _ in pairs})]
    print("%s: %d candidates on %d scans; %d pairs, %d patients" % (
        name, len(cand), cand.seriesuid.nunique(), n_pairs, n_pat))
    if led is not None:
        led.number("candidates", len(cand), "count").number("scans", cand.seriesuid.nunique(), "count")
        led.number("pairs", n_pairs, "count").number("patients", n_pat, "count")
    c = collect(cand, column, pairs, keys, pat_code)
    rows = []

    def add(section, key, item, system, doctors=np.nan, lo=np.nan, hi=np.nan):
        rows.append(dict(name=name, section=section, key=key, item=item, system=system, doctors=doctors,
                         lo=lo, hi=hi))
        if led is not None and not item.startswith("verdict_"):
            k = "%s.%s.%s" % (key, section, item)
            led.number(k, system).number(k + ".doctors", doctors).number(k + ".low", lo).number(k + ".high", hi)

    for key in lr.KEYS:
        ds, per = doctor_pairs(scores, key, pairs, pat_code)
        gate = make_gate(c[key])
        ones = np.ones(n_pat, dtype=int)
        n_ref, doc_hits, doc_fa = int(ds.ref.sum()), int(ds.hits.sum()), int(ds.fa.sum())
        assert n_ref == len(c[key]["best"]), (n_ref, len(c[key]["best"]))
        caught, thr = gate.caught(ones, doc_fa)
        dsens, ssens = 100 * doc_hits / n_ref, 100 * caught / n_ref
        d = bootstrap(gate, per, n_pat, boot, seed)
        word, mean, lo, hi = verdict(d)
        point = ssens - dsens
        text = "Verdict (doctors' line, %s): %s, system minus doctors %+.2f (%.2f to %.2f) points" % (
            KEY_TEXT[key], word, point, lo, hi)
        if led is not None and key == "2of3":
            led.note_is(text)
        if key == "2of3":
            print("\n" + text)
        else:
            print("\n" + text.replace("Verdict", "Reported"))
        print("  doctors: %d false alarms on %d pairs, %d of %d reference lesions (%.2f); system: threshold %s, "
              "%d caught (%.2f)" % (doc_fa, n_pairs, doc_hits, n_ref, dsens, thr, caught, ssens))
        add("gate", key, "k_false_alarms", doc_fa, doc_fa)
        add("gate", key, "reference_lesions", n_ref, n_ref)
        add("gate", key, "caught", caught, doc_hits)
        add("gate", key, "sensitivity", ssens, dsens, lo, hi)
        add("gate", key, "system_minus_doctors", point, 0.0, lo, hi)
        add("gate", key, "verdict_" + word, point, 0.0, lo, hi)
        # the system's false alarms at the doctors' sensitivity
        sb = np.sort(c[key]["best"][np.isfinite(c[key]["best"])])[::-1]
        if doc_hits == 0:
            need = 0.0
        elif doc_hits > len(sb):
            need = float("nan")
            print("  system never reaches the doctors' %d caught lesions" % doc_hits)
        else:
            need = float((c[key]["fa"] >= sb[doc_hits - 1]).sum()) / n_pairs
        print("  system false alarms per pair at the doctors' sensitivity: %.4f (doctors %.4f)" % (
            need, doc_fa / n_pairs))
        add("fa_at_doctors_sensitivity", key, "fa_per_pair", need, doc_fa / n_pairs)
        sens = []
        for a in ALLOWANCES:
            sens.append(100 * gate.caught(ones, int(round(a * n_pairs)))[0] / n_ref)
            add("sensitivity_at_fa_per_pair", key, "%g" % a, sens[-1])
        add("sensitivity_at_fa_per_pair", key, "mean", float(np.mean(sens)))
        print("  system sensitivity at %s false alarms per pair: %s; mean %.2f" % (
            "/".join("%g" % a for a in ALLOWANCES), " ".join("%.2f" % s for s in sens), np.mean(sens)))
        for pos in range(4):
            m = (ds.index.get_level_values("reader") == pos)
            pk, ph, pn = int(ds.fa[m].sum()), int(ds.hits[m].sum()), int(ds.ref[m].sum())
            g = make_gate(c[key], pos)
            pc, _ = g.caught(ones, pk)
            print("  reader position %d: doctors %d false alarms, sens %.2f; system sens %.2f" % (
                pos, pk, 100 * ph / pn, 100 * pc / pn))
            add("position", key, "reader_%d_k_false_alarms" % pos, pk, pk)
            add("position", key, "reader_%d_sensitivity" % pos, 100 * pc / pn, 100 * ph / pn)
    out = pd.DataFrame(rows)
    os.makedirs(lr.OUT, exist_ok=True)
    path = os.path.join(lr.OUT, "reader_line_%s.csv" % tag)
    out.to_csv(path, index=False)
    print("\n" + out.to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\nwrote %s" % path)


def selftest():
    # (a) synthetic scan: three reference lesions, one irrelevant mark, a tie at the threshold score
    keys = pd.DataFrame(dict(series_uid="syn", left_out_reader=0, lesion=range(4),
                             x_mm=[0.0, 100, 200, 300], y_mm=0.0, z_mm=0.0, diameter_mm=10.0, n_marking=[2, 2, 3, 1],
                             role=["reference_2of3"] * 3 + ["irrelevant"]))
    xyz = np.array([[0, 0, 0], [2, 0, 0], [100, 0, 0], [500, 0, 0], [600, 0, 0], [700, 0, 0], [300, 0, 0]], float)
    sc = np.array([0.9, 0.5, 0.7, 0.8, 0.7, 0.6, 0.95])
    cls = lr.classify(np.column_stack([xyz, np.full(len(xyz), np.nan)]), keys, "2of3", RULE)
    assert cls.tolist() == [0, 0, 1, -2, -2, -2, -1], cls
    best, fa, ign = pair_arrays(xyz, sc, keys, "2of3")
    assert best.tolist() == [0.9, 0.7, -np.inf] and sorted(fa.tolist()) == [0.6, 0.7, 0.8] and ign == 1
    g = Gate(best, np.zeros(3, int), fa, np.zeros(3, int))
    w = np.ones(1, dtype=int)
    for k, want, thr in ((0, 1, 0.8), (1, 1, 0.7), (2, 2, 0.6), (3, 2, -np.inf), (9, 2, -np.inf)):
        assert g.caught(w, k) == (want, thr), (k, g.caught(w, k))
    assert g.caught(np.array([2]), 1) == (2, 0.8)       # doubled patient: 0.8 twice, so the 2nd highest is 0.8
    print("synthetic gate: caught, k and threshold (tie at 0.7) as defined")

    mk, sc_ = lr.load_marks()
    rkeys, scores = lr.run(mk, sc_)
    luna = set(pd.read_csv(lr.LUNA_SUBSETS).seriesuid)
    pairs, pat_code = scope(scores, luna)
    # (b) every doctor's own marks as findings, score 1.0 at threshold 1.0
    pts = pd.read_csv(os.path.join(lr.OUT, "reader_points.csv"))
    pts = pts[(pts.scope == "luna16") & (pts.rule == RULE)]
    nod = mk[mk.kind == "nodule"]
    own = {ix: g[["x_mm", "y_mm", "z_mm"]].to_numpy(float) for ix, g in nod.groupby(["series_uid", "reader"])}
    kk = {ix: g.reset_index(drop=True) for ix, g in rkeys.groupby(["series_uid", "left_out_reader"])}
    for key in lr.KEYS:
        tot = {}
        for uid, r in pairs:
            f = own.get((uid, r), np.zeros((0, 3)))
            best, fas, ign = pair_arrays(f, np.ones(len(f)), kk.get((uid, r), EMPTY_KEYS), key)
            t = tot.setdefault(r, np.zeros(4, int))
            t += [len(best), int((best >= 1.0).sum()), int((fas >= 1.0).sum()), ign]
        tot["pooled"] = sum(tot.values())
        for pos, t in tot.items():
            row = pts[(pts.key == key) & (pts.reader_position.astype(str) == str(pos))]
            assert len(row) == 1, (key, pos)
            want = row[["ref_lesions", "hits", "false_alarms", "ignored"]].iloc[0].astype(int).tolist()
            assert t.tolist() == want, (key, pos, t.tolist(), want)
    print("doctors' own marks through this path reproduce reader_points.csv (both keys, 4 positions + pooled)")
    # (c) keys equal reader_keys.csv
    ref = pd.read_csv(os.path.join(lr.OUT, "reader_keys.csv"))
    srt = ["series_uid", "left_out_reader", "lesion"]
    a, b = rkeys.sort_values(srt).reset_index(drop=True), ref.sort_values(srt).reset_index(drop=True)
    assert len(a) == len(b), (len(a), len(b))
    for col in ("series_uid", "left_out_reader", "lesion", "n_marking", "role"):
        assert (a[col].to_numpy() == b[col].to_numpy()).all(), col
    for col in ("x_mm", "y_mm", "z_mm", "diameter_mm"):
        assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), atol=1e-6, rtol=0, equal_nan=True), col
    print("keys from run() equal reader_keys.csv (%d rows)" % len(a))
    # (d) coordinate frame: LUNA16 annotations inside a reference lesion for every left-out reader
    ann = pd.read_csv(ANNOTATIONS)
    four = set(rkeys.series_uid)
    ann = ann[ann.seriesuid.isin(four)]
    ok = 0
    for r in ann.itertuples():
        good = True
        for p in range(4):
            k = kk.get((r.seriesuid, p), EMPTY_KEYS)
            k = k[k.role.isin(lr.KEYS["2of3"])]
            d = np.linalg.norm(k[["x_mm", "y_mm", "z_mm"]].to_numpy(float) - [r.coordX, r.coordY, r.coordZ], axis=1)
            good = good and bool((d <= r.diameter_mm / 2).any())
        ok += good
    share = ok / len(ann)
    print("LUNA16 annotations on four-reader scans inside a 2-of-3 reference lesion for every left-out reader: "
          "%d of %d = %.4f" % (ok, len(ann), share))
    assert share >= 0.99, share
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates")
    ap.add_argument("--column", default="p2")
    ap.add_argument("--name", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.candidates:
        ap.error("--candidates is required unless --selftest")
    p = os.path.abspath(a.candidates)
    tag = os.path.basename(os.path.dirname(p)) + "_" + os.path.splitext(os.path.basename(p))[0]
    cand = pd.read_csv(p, usecols=["seriesuid", "coordX", "coordY", "coordZ", a.column])
    name = a.name or tag
    with ledger.run("measurement", "reader line of " + name) as led:
        led.settings_from({"candidates": p, "column": a.column, "boot": a.boot, "seed": a.seed})
        report(cand, a.column, name, a.boot, a.seed, tag, led)


if __name__ == "__main__":
    main()
