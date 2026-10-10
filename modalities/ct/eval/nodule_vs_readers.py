"""Does the lung nodule detector beat the radiologists on the radiologists' own answer keys?

lidc_readers.py scores each of the four radiologists of a scan against an answer key made from the
OTHER three. Here the detector is scored against the very same keys, once per left-out reader
position and key ('2of3': nodules at least two of the other three marked; '3of3': all three), with
froc.py's matching (a candidate inside the lesion's radius is a hit, one inside an excluded finding
is ignored, any other is a false alarm; the excluded findings are the keys' "irrelevant" rows, and for
the 3of3 key also the 2-of-3 lesions). The radiologist's own marks are scored with lidc_readers'
LUNA16 rule on exactly the same scans.

The detector's point for a radiologist: the lowest probability threshold whose false alarms per scan are
at or below that radiologist's, and the detector's sensitivity there. Pooled over the four positions the
difference (detector minus radiologists, out of 100) gets a 95% interval from resampling PATIENTS
(paired: one resample scores both, thresholds re-chosen inside each resample). The detector beats the
radiologists when that interval is above zero (docs/efficiency/README.md, 'The first lung nodule
detector', bar 3).

Caution: these are the UNBLINDED reads, which flatter the radiologists (see lidc_readers.py).

    python modalities/ct/eval/nodule_vs_readers.py <N>
    python modalities/ct/eval/nodule_vs_readers.py --selfcheck
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ledger, paths  # noqa: E402
import froc  # noqa: E402
import lidc_readers as lr  # noqa: E402

LUNG = os.path.join(paths.WORK, "eval", "lung")
RATES = (0.125, 0.25, 0.5, 1.0)
XYZ = ["x_mm", "y_mm", "z_mm"]
NOTE = ("These are the UNBLINDED reads: each radiologist saw the other three's marks before finalising, "
        "which flatters the radiologists.")


def _frame(rows, scan_id):
    """A froc annotation frame from key rows (unsized rows get a 6 mm diameter: radius at least 3 mm, as in lidc_readers)."""
    d = np.nan_to_num(rows.diameter_mm.to_numpy(dtype=float), nan=0.0)
    return pd.DataFrame({"seriesuid": rows.series_uid.to_numpy(), "coordX": rows.x_mm.to_numpy(),
                         "coordY": rows.y_mm.to_numpy(), "coordZ": rows.z_mm.to_numpy(),
                         "diameter_mm": np.maximum(d, 6.0)})


def prepare(cands, keys, marks, scans_frame):
    """Everything per (reader position, key) that a resample needs: the detector's hits and false alarms
    with their scores and scan, and the radiologist's own hits and false alarms per scan."""
    four = set(scans_frame.series_uid[scans_frame.n_readers == 4])
    cands = cands.assign(seriesuid=cands.seriesuid.astype(str))
    uids = sorted(set(cands.seriesuid) & {str(u) for u in four})
    cands = cands[cands.seriesuid.isin(uids)]
    pat = dict(zip(scans_frame.series_uid.astype(str), scans_frame.patient))
    patient = np.array([pat[u] for u in uids])
    keys = keys.assign(series_uid=keys.series_uid.astype(str))
    marks = marks.assign(series_uid=marks.series_uid.astype(str))
    kby = {k: g for k, g in keys.groupby(["series_uid", "left_out_reader"])}
    mby = {k: g for k, g in marks[marks.series_uid.isin(uids)].groupby(["series_uid", "reader"])}
    empty_k = keys.iloc[0:0]
    data = {}
    for r in range(4):
        for key, roles in lr.KEYS.items():
            k = keys[keys.left_out_reader == r]
            ref = k[k.role.isin(roles)]
            exc = k[~k.role.isin(roles)]
            m = froc._match(cands, _frame(ref, None), _frame(exc, None), uids)
            ref_n, hits, fa = np.zeros(len(uids)), np.zeros(len(uids)), np.zeros(len(uids))
            for i, u in enumerate(uids):
                g = mby.get((u, r))
                f = (g[g.kind == "nodule"][XYZ + ["diameter_mm"]].to_numpy() if g is not None
                     else np.zeros((0, 4)))
                n_ref, h, a, _ = lr.score_scan(f, kby.get((u, r), empty_k), key, "luna16")
                ref_n[i], hits[i], fa[i] = n_ref, h, a
            assert (ref_n == m["lesions"]).all(), "reference counts differ between the two scorers"
            to = np.argsort(-m["tp_prob"], kind="stable")
            fo = np.argsort(-m["fp_prob"], kind="stable")
            data[(r, key)] = dict(
                ref=ref_n, hits=hits, fa=fa,
                tp_p=m["tp_prob"][to], tp_s=m["tp_scan"][to], fp_p=m["fp_prob"][fo], fp_s=m["fp_scan"][fo])
    return uids, patient, data


def detector_at(d, counts, budget):
    """(true positives, false alarms) of the detector at the lowest threshold whose false alarms per scan
    (counts-weighted) are at or below `budget`. Scans are weighted by `counts` (a resample)."""
    n = counts.sum()
    c = np.cumsum(counts[d["fp_s"]])
    over = np.flatnonzero(c > budget * n + 1e-9)
    if len(over) == 0:
        return counts[d["tp_s"]].sum(), c[-1] if len(c) else 0.0
    cut = d["fp_p"][over[0]]                     # the first false alarm that does not fit: stay above its score
    fp_in = counts[d["fp_s"]][d["fp_p"] > cut].sum()
    return counts[d["tp_s"]][d["tp_p"] > cut].sum(), fp_in


def evaluate(data, counts):
    """Per key: the four readers' rows and the pooled row, for the scan weights `counts`."""
    n = counts.sum()
    out = {}
    for key in lr.KEYS:
        rows = []
        for r in range(4):
            d = data[(r, key)]
            ref, hits, fa = (counts * d["ref"]).sum(), (counts * d["hits"]).sum(), (counts * d["fa"]).sum()
            tp, fp = detector_at(d, counts, fa / n)
            at = [100 * detector_at(d, counts, b)[0] / ref if ref else np.nan for b in RATES]
            rows.append(dict(reader_position=r, ref=ref, reader_hits=hits, reader_fa=fa, det_hits=tp, det_fa=fp, at=at))
        out[key] = rows
    return out


def line(row, n):
    s_r = 100 * row["reader_hits"] / row["ref"] if row["ref"] else np.nan
    s_d = 100 * row["det_hits"] / row["ref"] if row["ref"] else np.nan
    return s_r, s_d


def pooled(rows):
    ref = sum(r["ref"] for r in rows)
    return dict(reader_position="pooled", ref=ref, reader_hits=sum(r["reader_hits"] for r in rows),
                reader_fa=sum(r["reader_fa"] for r in rows) / 4, det_hits=sum(r["det_hits"] for r in rows),
                det_fa=sum(r["det_fa"] for r in rows) / 4,
                at=[100 * sum(r["at"][i] * r["ref"] for r in rows) / ref if ref else np.nan for i in range(len(RATES))])


def compare(cands, keys, marks, scans_frame, n_boot=1000, seed=0):
    """The whole comparison: (rows frame, {key: (difference, low, high)}, number of scans)."""
    uids, patient, data = prepare(cands, keys, marks, scans_frame)
    n = len(uids)
    base = evaluate(data, np.ones(n))
    _, pid = np.unique(patient, return_inverse=True)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in lr.KEYS}
    for _ in range(n_boot if len(set(patient)) > 1 else 0):
        cp = np.bincount(rng.integers(0, pid.max() + 1, pid.max() + 1), minlength=pid.max() + 1).astype(float)
        ev = evaluate(data, cp[pid])
        for key in lr.KEYS:
            p = pooled(ev[key])
            if p["ref"]:
                diffs[key].append(100 * (p["det_hits"] - p["reader_hits"]) / p["ref"])
    rows, ci = [], {}
    for key in lr.KEYS:
        for row in base[key] + [pooled(base[key])]:
            s_r, s_d = line(row, n)
            rows.append(dict(key=key, reader_position=row["reader_position"], scans=n,
                             ref_lesions=int(row["ref"]), reader_sensitivity=s_r,
                             reader_fa_per_scan=row["reader_fa"] / n,
                             detector_sensitivity=s_d, detector_fa_per_scan=row["det_fa"] / n,
                             difference=s_d - s_r,
                             **{"detector_at_%g" % b: v for b, v in zip(RATES, row["at"])}))
        d = np.array(diffs[key])
        lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
        ci[key] = (rows[-1]["difference"], lo, hi)
        rows[-1]["diff_lo"], rows[-1]["diff_hi"] = lo, hi
    return pd.DataFrame(rows), ci, n


def verdict(lo, hi):
    if lo > 0:
        return "the detector beats the radiologists"
    if hi < 0:
        return "the detector is worse than the radiologists"
    return "the detector does not beat the radiologists (the interval includes zero)"


def show(rows, ci, version, n):
    print("\nNodule detector v%d against the radiologists, %d scans read by four radiologists" % (version, n))
    print(NOTE)
    for key, what in (("2of3", "at least 2 of the other 3 radiologists agree"), ("3of3", "all 3 of the other radiologists agree")):
        print("\nNodules that %s:" % what)
        for r in rows[rows.key == key].itertuples():
            who = "Pooled over the 4 radiologist positions" if r.reader_position == "pooled" else "Radiologist position %s" % r.reader_position
            print("  %s: out of 100 such nodules the radiologist catches %.1f with %.2f false alarms a scan; "
                  "the detector, allowed the same false alarms, catches %.1f (difference %+.1f)"
                  % (who, r.reader_sensitivity, r.reader_fa_per_scan, r.detector_sensitivity, r.difference))
        t = rows[(rows.key == key) & (rows.reader_position == "pooled")].iloc[0]
        print("  Detector alone, out of 100 caught at 1/8, 1/4, 1/2, 1 false alarms a scan: %s"
              % ", ".join("%.1f" % t["detector_at_%g" % b] for b in RATES))
        d, lo, hi = ci[key]
        print("  Pooled difference, detector minus radiologists: %+.1f out of 100 (95%% interval %+.1f to %+.1f): %s"
              % (d, lo, hi, verdict(lo, hi)))


def run(version, n_boot=1000):
    out = os.path.join(LUNG, "nodule-v%d" % version)
    cands = pd.read_csv(os.path.join(out, "candidates_oof.csv"))
    mk, sc = lr.load_marks()
    kp = os.path.join(LUNG, "reader_keys.csv")
    keys = pd.read_csv(kp) if os.path.exists(kp) else lr.run(mk, sc)[0]
    with ledger.run("nodule", "nodule v%d against the radiologists" % version) as led:
        led.settings_from({"version": version, "boot": n_boot})
        rows, ci, n = compare(cands, keys, mk, sc, n_boot)
        rows.to_csv(os.path.join(out, "vs_readers.csv"), index=False)
        led.number("scans", n, "count")
        for key in lr.KEYS:
            p = rows[(rows.key == key) & (rows.reader_position == "pooled")].iloc[0]
            led.number("reader_sensitivity_%s" % key, p.reader_sensitivity)
            led.number("detector_sensitivity_%s" % key, p.detector_sensitivity)
            led.number("sensitivity_difference_%s" % key, ci[key][0])
            led.number("sensitivity_difference_low_%s" % key, ci[key][1])
            led.number("sensitivity_difference_high_%s" % key, ci[key][2])
            led.number("reader_fa_per_scan_%s" % key, p.reader_fa_per_scan, "count")
    show(rows, ci, version, n)
    print("\nwrote %s" % os.path.join(out, "vs_readers.csv"))
    return rows, ci


def selfcheck():
    def m(uid, reader, kind, x, y, z, diam=6.0):
        return dict(series_uid=uid, patient="p" + uid, reader=reader, kind=kind, x_mm=x, y_mm=y, z_mm=z,
                    diameter_mm=diam if kind == "nodule" else float("nan"))
    rows = []
    for u in ("a", "b"):                       # every reader marks one lesion at (10,10,10); only reader 0 adds a false alarm
        rows += [m(u, r, "nodule", 10, 10, 10) for r in range(4)]
    rows += [m("a", 0, "nodule", 90, 0, 0), m("b", 0, "nodule", 90, 0, 0)]
    mk = pd.DataFrame(rows)
    sc = pd.DataFrame({"series_uid": ["a", "b"], "patient": ["pa", "pb"], "n_readers": [4, 4]})
    keys = pd.concat([lr.scan_keys(u, mk[mk.series_uid == u], r) for u in ("a", "b") for r in range(4)])
    perfect = pd.DataFrame([dict(seriesuid=u, coordX=10, coordY=10, coordZ=10, probability=0.9) for u in "ab"]
                           + [dict(seriesuid=u, coordX=0, coordY=0, coordZ=60, probability=0.5) for u in "ab"])
    rows_p, ci_p, n = compare(perfect, keys, mk, sc, n_boot=50)
    p = rows_p[(rows_p.key == "2of3")]
    r0 = p[p.reader_position == 0].iloc[0]
    # reader 0 has 1 false alarm a scan, so the detector's own false alarm (at 0.5) is allowed
    assert (r0.reader_sensitivity, r0.reader_fa_per_scan) == (100.0, 1.0), r0
    assert (r0.detector_sensitivity, r0.detector_fa_per_scan) == (100.0, 1.0), r0
    r1 = p[p.reader_position == 1].iloc[0]                  # a clean reader: 0 false alarms allowed, the 0.9 hits still found
    assert (r1.reader_fa_per_scan, r1.detector_sensitivity, r1.detector_fa_per_scan) == (0.0, 100.0, 0.0), r1
    assert p.iloc[0]["detector_at_0.125"] == 100.0
    assert p[p.reader_position == "pooled"].iloc[0].difference == 0.0 and ci_p["2of3"][0] == 0.0
    # a detector that finds nothing: candidates only far from everything
    miss = pd.DataFrame([dict(seriesuid=u, coordX=-80, coordY=-80, coordZ=-80, probability=0.7) for u in "ab"])
    rows_m, ci_m, _ = compare(miss, keys, mk, sc, n_boot=50)
    q = rows_m[(rows_m.key == "2of3") & (rows_m.reader_position == "pooled")].iloc[0]
    assert q.detector_sensitivity == 0.0 and q.reader_sensitivity == 100.0 and q.difference == -100.0, q
    assert ci_m["2of3"][0] < 0 and ci_m["2of3"][2] < 0, ci_m
    print("synthetic: detector sensitivity and false alarms as defined; difference %+.0f for a perfect-hit detector, %+.0f for one that finds nothing"
          % (ci_p["2of3"][0], ci_m["2of3"][0]))
    real = os.path.join(LUNG, "nodule-v1", "candidates_oof.csv")
    if os.path.exists(real):
        run(1, n_boot=200)
    print("selfcheck ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("version", nargs="?", type=int)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.version is None:
        ap.error("give the detector version N, or --selfcheck")
    run(a.version, a.boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
