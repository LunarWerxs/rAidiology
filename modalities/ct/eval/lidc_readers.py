"""The doctors' line for lung nodule detection: how well one radiologist does when judged by the other three.

In LIDC-IDRI each chest CT was read by four radiologists. To say how good a computer model is, it
helps to know how good a doctor is on the very same scans. So for each scan and each radiologist
r we build an answer key from the OTHER three only, then score r's own findings against it. A
model is later scored against exactly the same keys (once per left-out reader, so 4 times per scan),
and its sensitivity at the doctors' false-alarm rate is compared with theirs.

The key: the other three's marks of 3 mm or more are matched into lesions (the rule in
lidc.match_lesions). A lesion marked by at least 2 of the 3 is "reference" (the lenient key); marked
by all 3 is the strict key (then the 2-of-3 lesions count neither way). A lesion marked by only 1
of the 3, plus their under-3 mm points and their "non-nodule" marks, are "irrelevant": a finding
there is neither a hit nor a false alarm.

The score, the LUNA16 way: a finding of r inside a reference lesion's radius (mean diameter / 2,
centre = mean of its marks) is a hit (several findings on one lesion count once); a finding inside
an irrelevant mark's radius (its diameter / 2, at least 3 mm) is ignored; any other finding is a
false alarm. We also report the looser rule of lidc.py (distance within the larger of the two radii
plus 2 mm) so the effect of the hit rule is visible.

Caution: these are the UNBLINDED reads. In the second round every radiologist saw the other three's
marks before finalising their own, so the four are not independent, and a doctor scored against
the other three is flattered: part of the key was shaped by that doctor's own answer. The doctors'
line is therefore somewhat higher than a truly independent one would be. Reader numbers (0 to 3)
are only the position of a reading session in a scan's file, not the same person across scans.

Outputs under <Work>/eval/lung/: reader_keys.csv (one row per series_uid, left_out_reader, lesion:
x/y/z_mm, diameter_mm, n_marking, role reference_2of3 / reference_3of3 / irrelevant) and
reader_points.csv (the printed tables: per reader position and pooled, for all scans and for
LUNA16's 888, for both keys and both hit rules).

    python modalities/ct/eval/lidc_readers.py [--selfcheck]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import paths  # noqa: E402
import lidc  # noqa: E402

OUT = os.path.join(paths.WORK, "eval", "lung")
LUNA_SUBSETS = os.path.join(paths.DATASETS, "public", "luna16", "subsets.csv")
NOTE = ("(unblinded reads: each doctor saw the other three's marks before finalising, "
        "which flatters a doctor scored against the others)")
KEYS = {"2of3": ("reference_2of3", "reference_3of3"), "3of3": ("reference_3of3",)}
RULES = ("luna16", "lidc")
KEY_COLS = ["series_uid", "left_out_reader", "lesion", "x_mm", "y_mm", "z_mm", "diameter_mm", "n_marking", "role"]


def scan_keys(uid, scan_marks, left_out):
    """Answer key of one scan with reader `left_out` left out: DataFrame with KEY_COLS."""
    others = scan_marks[scan_marks.reader != left_out].reset_index(drop=True)
    nod = others[others.kind == "nodule"].reset_index(drop=True)
    rows = []
    for ix in lidc.match_lesions(nod):
        g = nod.iloc[ix]
        role = {1: "irrelevant", 2: "reference_2of3"}.get(len(g), "reference_3of3")
        rows.append((g.x_mm.mean(), g.y_mm.mean(), g.z_mm.mean(), g.diameter_mm.mean(), len(g), role))
    for r in others[others.kind != "nodule"].itertuples():
        rows.append((r.x_mm, r.y_mm, r.z_mm, float("nan"), 1, "irrelevant"))
    df = pd.DataFrame(rows, columns=["x_mm", "y_mm", "z_mm", "diameter_mm", "n_marking", "role"])
    df.insert(0, "lesion", range(len(df)))
    df.insert(0, "left_out_reader", left_out)
    df.insert(0, "series_uid", uid)
    return df[KEY_COLS]


def classify(findings, keys, key, rule):
    """Per finding: row position in `keys` of the reference lesion it hits (nearest such), -1 if ignored,
    -2 if a false alarm. findings: Nx4 array x, y, z, diameter (mm); keys: scan_keys() frame."""
    f = np.asarray(findings, dtype=float).reshape(-1, 4)
    if not len(keys) or not len(f):
        return np.full(len(f), -2, dtype=int)
    isref = keys.role.isin(KEYS[key]).to_numpy()[None]
    xyz = keys[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    krad = np.nan_to_num(keys.diameter_mm.to_numpy(dtype=float) / 2)[None]
    d = np.linalg.norm(f[:, None, :3] - xyz[None], axis=2)
    lim = krad if rule == "luna16" else np.maximum(krad, np.nan_to_num(f[:, 3] / 2)[:, None]) + lidc.TOL_MM
    inside = isref & (d <= lim)
    ignored = (~isref & (d <= np.maximum(krad, 3.0))).any(axis=1)
    near = np.argmin(np.where(inside, d, np.inf), axis=1)
    return np.where(inside.any(axis=1), near, np.where(ignored, -1, -2)).astype(int)


def score_scan(findings, keys, key, rule):
    """(reference lesions, hits, false alarms, ignored findings) of one reader's findings.
    findings: Nx4 array x, y, z, diameter (mm); keys: scan_keys() frame; key '2of3' or '3of3'."""
    cls = classify(findings, keys, key, rule)
    n_ref = int(keys.role.isin(KEYS[key]).sum())
    return n_ref, len(set(cls[cls >= 0].tolist())), int((cls == -2).sum()), int((cls == -1).sum())


def load_marks():
    for attempt in range(5):
        try:
            return lidc.marks(), lidc.scans()
        except PermissionError:
            time.sleep(3 + attempt * 3)
    return lidc.marks(), lidc.scans()


def run(mk, sc):
    """(keys frame, per-(scan, left-out reader, key, rule) score frame) over every 4-reader scan."""
    pat = dict(zip(sc.series_uid, sc.patient))
    four = sc.series_uid[sc.n_readers == 4]
    by = {u: d.reset_index(drop=True) for u, d in mk[mk.series_uid.isin(set(four))].groupby("series_uid")}
    keys, rows = [], []
    for uid in sorted(four):
        d = by.get(uid)
        if d is None:
            d = mk.iloc[0:0]
        for r in range(4):
            k = scan_keys(uid, d, r)
            keys.append(k)
            f = d[(d.reader == r) & (d.kind == "nodule")][["x_mm", "y_mm", "z_mm", "diameter_mm"]].to_numpy()
            for key in KEYS:
                for rule in RULES:
                    n_ref, h, fa, ig = score_scan(f, k, key, rule)
                    rows.append((uid, pat[uid], r, key, rule, n_ref, h, fa, ig))
    keys = pd.concat(keys, ignore_index=True) if keys else pd.DataFrame(columns=KEY_COLS)
    cols = ["series_uid", "patient", "reader", "key", "rule", "ref", "hits", "fa", "ignored"]
    return keys, pd.DataFrame(rows, columns=cols)


def summarize(d, n_boot=1000, seed=0):
    """Row of results for a frame of per-scan scores: sensitivity /100 with 95% interval over patients."""
    ref, hits, fa = int(d.ref.sum()), int(d.hits.sum()), int(d.fa.sum())
    out = {"scans": len(d), "ref_lesions": ref, "hits": hits, "false_alarms": fa, "ignored": int(d.ignored.sum()),
           "sensitivity": 100 * hits / ref if ref else float("nan"),
           "fa_per_scan": fa / len(d) if len(d) else float("nan")}
    p = d.groupby("patient")[["ref", "hits", "fa"]].sum().to_numpy()
    n = d.groupby("patient").size().to_numpy()
    lo = hi = float("nan")
    if len(p) > 1 and ref:
        ix = np.random.default_rng(seed).integers(0, len(p), (n_boot, len(p)))
        r, h = p[ix, 0].sum(1), p[ix, 1].sum(1)
        s = 100 * h[r > 0] / r[r > 0]
        lo, hi = np.percentile(s, [2.5, 97.5])
    out.update(sens_lo=lo, sens_hi=hi)
    return out


def tables(scores, luna):
    rows = []
    for scope, sub in (("all", scores), ("luna16", scores[scores.series_uid.isin(luna)] if luna is not None else None)):
        if sub is None or not len(sub):
            continue
        for key in KEYS:
            for rule in RULES:
                s = sub[(sub.key == key) & (sub.rule == rule)]
                for pos in [0, 1, 2, 3, "pooled"]:
                    t = s if pos == "pooled" else s[s.reader == pos]
                    rows.append(dict(scope=scope, key=key, rule=rule, reader_position=pos, **summarize(t)))
    return pd.DataFrame(rows)


def show(points):
    for (scope, key, rule), t in points.groupby(["scope", "key", "rule"], sort=False):
        print("\n%s scans; reference = %s; hit rule = %s %s" % (
            scope, "at least 2 of the other 3" if key == "2of3" else "all 3 of the other 3",
            "LUNA16 (lesion radius)" if rule == "luna16" else "lidc.py (larger radius + 2 mm)", NOTE))
        v = t[["reader_position", "scans", "ref_lesions", "hits", "sensitivity", "sens_lo", "sens_hi",
               "false_alarms", "fa_per_scan", "ignored"]]
        print(v.to_string(index=False, float_format=lambda x: "%.2f" % x))


def selfcheck():
    def m(reader, kind, x, y, z, diam=5.0):
        return dict(series_uid="syn", patient="p", reader=reader, kind=kind, x_mm=x, y_mm=y, z_mm=z,
                    diameter_mm=diam if kind == "nodule" else float("nan"))
    rows = [m(r, "nodule", 10, 10, 10) for r in range(4)]          # all four mark it
    rows += [m(0, "nodule", 50, 50, 50),                          # only reader 0
             m(1, "non_nodule", 52, 50, 50),                      # reader 1's non-nodule beside it
             m(0, "nodule", 90, 0, 0),                            # a plain false alarm
             m(1, "nodule", 0, 80, 0), m(2, "nodule", 0, 80, 0),  # two of the others, not reader 0
             m(3, "nodule", -50, -50, -50)]                       # one only
    d = pd.DataFrame(rows)
    k = scan_keys("syn", d, 0)
    roles = k.groupby("role").size().to_dict()
    assert roles == {"irrelevant": 2, "reference_2of3": 1, "reference_3of3": 1}, roles
    f = d[(d.reader == 0) & (d.kind == "nodule")][["x_mm", "y_mm", "z_mm", "diameter_mm"]].to_numpy()
    assert score_scan(f, k, "2of3", "luna16") == (2, 1, 1, 1), score_scan(f, k, "2of3", "luna16")
    assert score_scan(f, k, "3of3", "luna16") == (1, 1, 1, 1), score_scan(f, k, "3of3", "luna16")
    near = np.array([[10.0, 10.0, 14.0, 5.0]])    # 4 mm off a 5 mm lesion: LUNA16 miss, lidc.py hit
    assert score_scan(near, k, "3of3", "luna16")[1] == 0 and score_scan(near, k, "3of3", "lidc")[1] == 1
    print("synthetic scan: hits, false alarms, ignores and roles as defined")
    ok, _ = lidc.unpacked()
    if not ok:
        print("no LIDC scans unpacked yet; synthetic check only")
    else:
        mk, sc = load_marks()
        print("real run: %d scans complete, %d with 4 readers" % (len(sc), int((sc.n_readers == 4).sum())))
        main(quiet_keys=False)
    print("selfcheck ok")


def main(quiet_keys=False):
    mk, sc = load_marks()
    keys, scores = run(mk, sc)
    luna = set(pd.read_csv(LUNA_SUBSETS).seriesuid) if os.path.exists(LUNA_SUBSETS) else None
    points = tables(scores, luna)
    os.makedirs(OUT, exist_ok=True)
    keys.to_csv(os.path.join(OUT, "reader_keys.csv"), index=False)
    points.to_csv(os.path.join(OUT, "reader_points.csv"), index=False)
    n4 = int((sc.n_readers == 4).sum())
    print("scans with exactly 4 readers scored: %d of %d unpacked%s" % (
        n4, len(sc), "; %d of them in LUNA16" % scores[scores.series_uid.isin(luna)].series_uid.nunique()
        if luna is not None else ""))
    r = keys[keys.role.isin(KEYS["2of3"])]
    print("reference lesions per left-out reader (2 of 3, all scans): " + ", ".join(
        "%d: %d" % (p, n) for p, n in r.groupby("left_out_reader").size().items()))
    show(points)
    print("\nwrote reader_keys.csv (%d rows) and reader_points.csv in %s" % (len(keys), OUT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    selfcheck() if a.selfcheck else main()
