"""Independent recomputation of the "reader line" headline number.

Written apart from the main scorer (no lidc_readers, no reader_line): it
re-implements the matching from the written definition and prints, per key
('2of3', '3of3'): pairs, reference lesions, k (the doctors' pooled false
alarms), the (k+1)-th highest false-alarm score, the system's false alarms
above that cut, lesions caught, and sensitivity.

    python tools/reader_line_check.py [--candidates CSV] [--column p2]
    python tools/reader_line_check.py --selftest
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REF_ROLES = {"2of3": ("reference_2of3", "reference_3of3"), "3of3": ("reference_3of3",)}
IGNORE_FLOOR_MM = 3.0


def score_pair(xyz, sc, ref, nonref):
    """One (scan, reader) pair.

    xyz (n,3) candidate centres, sc (n,) scores; ref / nonref are (m,4) arrays
    of x, y, z, diameter (NaN diameter = 0). Returns (best, fa_scores):
    best[j] = highest score of a candidate that hits reference row j (-inf if
    none), fa_scores = scores of the candidates that are false alarms.
    """
    best = np.full(len(ref), -np.inf)
    n = len(xyz)
    if n == 0:
        return best, np.empty(0)
    hit = np.zeros(n, dtype=bool)
    if len(ref):
        d = np.linalg.norm(xyz[:, None, :] - ref[None, :, :3], axis=2)
        rad = np.nan_to_num(ref[:, 3], nan=0.0) / 2.0
        inside = np.where(d <= rad[None, :], d, np.inf)
        nearest = inside.argmin(axis=1)
        hit = np.isfinite(inside[np.arange(n), nearest])
        if hit.any():
            np.maximum.at(best, nearest[hit], sc[hit])
    ignored = np.zeros(n, dtype=bool)
    if len(nonref):
        d = np.linalg.norm(xyz[:, None, :] - nonref[None, :, :3], axis=2)
        rad = np.maximum(np.nan_to_num(nonref[:, 3], nan=0.0) / 2.0, IGNORE_FLOOR_MM)
        ignored = (d <= rad[None, :]).any(axis=1)
    return best, sc[~hit & ~ignored]


def run_key(pairs, cands, keys, key, k):
    """pairs: list of (series_uid, reader); cands: {uid: (xyz, score)};
    keys: {(uid, reader): (xyz+diam array, role array)}."""
    refroles = REF_ROLES[key]
    empty = np.empty((0, 4))
    bests, fas = [], []
    for uid, reader in pairs:
        arr, roles = keys.get((uid, reader), (empty, np.empty(0, dtype=object)))
        is_ref = np.isin(roles, refroles)
        # non-reference rows: everything that is not reference (3of3: reference_2of3 too)
        ref, non = arr[is_ref], arr[~is_ref]
        xyz, sc = cands.get(uid, (np.empty((0, 3)), np.empty(0)))
        b, fa = score_pair(xyz, sc, ref, non)
        bests.append(b)
        fas.append(fa)
    best = np.concatenate(bests) if bests else np.empty(0)
    fa = np.concatenate(fas) if fas else np.empty(0)
    fa_sorted = np.sort(fa)[::-1]
    cut = float(fa_sorted[k]) if len(fa_sorted) > k else float("-inf")
    n_fa = int((fa > cut).sum())
    assert n_fa <= k, (n_fa, k)
    caught = int((best > cut).sum())
    ref_n = len(best)
    sens = 100.0 * caught / ref_n if ref_n else float("nan")
    return dict(pairs=len(pairs), ref=ref_n, k=k, cut=cut, fa=n_fa, caught=caught, sens=sens)


def fmt(key, r):
    return (f"{key}: pairs={r['pairs']} ref={r['ref']} k={r['k']} cut={r['cut']:.6f} "
            f"fa={r['fa']} caught={r['caught']} sensitivity={r['sens']:.4f}")


def group_keys(df):
    out = {}
    for (uid, reader), g in df.groupby(["series_uid", "left_out_reader"], sort=False):
        arr = g[["x_mm", "y_mm", "z_mm", "diameter_mm"]].to_numpy(dtype=float)
        out[(uid, int(reader))] = (arr, g["role"].to_numpy(dtype=object))
    return out


def group_cands(df, column):
    out = {}
    for uid, g in df.groupby("seriesuid", sort=False):
        out[uid] = (g[["coordX", "coordY", "coordZ"]].to_numpy(dtype=float),
                    g[column].to_numpy(dtype=float))
    return out


def selftest():
    rows = [
        # scan A, reader 0 (scan B has no key rows: an empty key)
        ("A", 0, 0, 0, 0, 0, 10.0, "reference_2of3"),
        ("A", 0, 1, 100, 0, 0, 10.0, "reference_3of3"),
        ("A", 0, 2, 0, 50, 0, 4.0, "irrelevant"),
        ("A", 0, 3, 400, 0, 0, np.nan, "irrelevant"),
    ]
    keys_df = pd.DataFrame(rows, columns=["series_uid", "left_out_reader", "lesion", "x_mm", "y_mm",
                                          "z_mm", "diameter_mm", "role"])
    keys_df["n_marking"] = 1
    cand_rows = [
        ("A", 1, 0, 0, 0.9),     # hits the 2of3-only lesion
        ("A", 100, 1, 0, 0.5),   # hits the 3of3 lesion
        ("A", 0, 51, 0, 0.8),    # ignored by the small irrelevant row
        ("A", 200, 0, 0, 0.7),   # false alarm
        ("A", 300, 0, 0, 0.6),   # false alarm, ties with B's
        ("A", 402, 0, 0, 0.95),  # ignored: NaN diameter counts as 0, floor 3 mm
        ("B", 0, 0, 0, 0.6),     # false alarm (empty key), tie at the cut
        ("B", 0, 0, 5, 0.3),     # false alarm
    ]
    cdf = pd.DataFrame(cand_rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "p2"])
    keys = group_keys(keys_df)
    cands = group_cands(cdf, "p2")
    pairs = [("A", 0), ("B", 0)]
    # false-alarm scores 0.7 0.6 0.6 0.3; k=2 -> cut is the 3rd highest, 0.6
    r = run_key(pairs, cands, keys, "2of3", 2)
    assert (r["pairs"], r["ref"], r["fa"], r["caught"]) == (2, 2, 1, 1), r
    assert r["cut"] == 0.6 and r["sens"] == 50.0, r
    # 3of3: lesion 0 turns into a non-reference row (its hit candidate is ignored)
    r = run_key(pairs, cands, keys, "3of3", 2)
    assert (r["ref"], r["fa"], r["caught"]) == (1, 1, 0), r
    assert r["cut"] == 0.6 and r["sens"] == 0.0, r
    # k above the number of false alarms: no cut, every lesion with a hit counts
    r = run_key(pairs, cands, keys, "2of3", 10)
    assert r["cut"] == float("-inf") and r["fa"] == 4 and r["caught"] == 2 and r["sens"] == 100.0, r
    # k=0: cut is the highest false-alarm score, 0.7
    r = run_key(pairs, cands, keys, "2of3", 0)
    assert r["cut"] == 0.7 and r["fa"] == 0 and r["caught"] == 1, r
    print("selftest ok")


def load_scans():
    path = os.path.join(ROOT, "modalities", "ct", "eval", "lidc.py")
    spec = importlib.util.spec_from_file_location("lidc_scans_only", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.scans()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--column", default="p2")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    from core import paths
    work, datasets = str(paths.WORK), str(paths.DATASETS)
    lung = os.path.join(work, "eval", "lung")
    cand_path = a.candidates or os.path.join(lung, "fpr-v2", "candidates_oof_sys5_tta8.csv")

    sc = load_scans()
    luna = set(pd.read_csv(os.path.join(datasets, "public", "luna16", "subsets.csv"))["seriesuid"])
    uids = [u for u in sc.loc[sc["n_readers"] == 4, "series_uid"] if u in luna]
    pairs = [(u, r) for u in uids for r in range(4)]

    rp = pd.read_csv(os.path.join(lung, "reader_points.csv"))
    row = rp[(rp["scope"] == "luna16") & (rp["key"] == "2of3") & (rp["rule"] == "luna16")
             & (rp["reader_position"] == "pooled")]
    assert len(row) == 1, f"expected one pooled luna16 2of3 row, got {len(row)}"
    k = int(row["false_alarms"].iloc[0])

    keys = group_keys(pd.read_csv(os.path.join(lung, "reader_keys.csv")))
    cands = group_cands(pd.read_csv(cand_path), a.column)
    for key in ("2of3", "3of3"):
        print(fmt(key, run_key(pairs, cands, keys, key, k)))


if __name__ == "__main__":
    main()
