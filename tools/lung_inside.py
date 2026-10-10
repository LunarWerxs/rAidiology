"""Only places inside the lungs (idea 18): drop candidates whose centre is outside the scan's lung mask.

`out_mm` of a point is its Euclidean distance in mm to the nearest voxel of its scan's lung mask (0
inside). The mask is lung_prepare's, kept by nodule_blobs.lung_of in <WORK>/eval/lung/mask/<uid>.npz;
the volume is <WORK>/eval/lung/vol/<uid>.npy with its sidecar .json (origin_mm). Per scan: one
distance transform of the mask's complement, then a lookup at each point's rounded voxel. A point
outside the volume's crop gets the distance at the nearest in-crop voxel plus the distance from the
point to that voxel. A scan with no volume or an empty mask gives NaN, which is counted and printed,
and the filter keeps such rows.

    python tools/lung_inside.py calibrate [--jobs 4]
        out_mm of the 207 nodule centres of lidc_extra.extra_reference() (LIDC's 130 scans LUNA16 left
        out, never scored); D = their max + 2 mm; writes <WORK>/eval/lung/inlung_D.json
    python tools/lung_inside.py filter --candidates <csv> [--d-file <json> | --d <mm>] [--jobs 4]
        writes <stem>_outmm.csv (every row plus out_mm) and <stem>_inlung.csv (rows with out_mm <= D or
        NaN, original columns) next to the candidates; prints what LUNA16's nodules lose
    python tools/lung_inside.py selftest
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "modalities", "ct", "eval")
sys.path[:0] = [EVAL, ROOT]
from core import ledger, paths  # noqa: E402

LUNG = os.path.join(paths.WORK, "eval", "lung")
D_FILE = os.path.join(LUNG, "inlung_D.json")
XYZ = ["coordX", "coordY", "coordZ"]
MARGIN_MM = 2.0


def _blobs_mask(uid, hu_path):
    """The scan's lung mask by nodule_blobs.lung_of (read from its cache, made from the volume when missing)."""
    import nodule_blobs as nb
    if os.path.exists(os.path.join(nb.MASKS, uid + ".npz")):
        return nb.lung_of(uid, None)
    import lung_prepare as lp
    return nb.lung_of(uid, lp.to_hu(np.load(hu_path)))


def scan_out(uid, xyz, vol_dir=None, mask_fn=None):
    """out_mm (n,) of world points `xyz` (n x 3) in scan `uid`; NaN for all when it has no volume or an empty mask."""
    from scipy import ndimage as ndi
    vol_dir = vol_dir or os.path.join(LUNG, "vol")
    npy, js = os.path.join(vol_dir, uid + ".npy"), os.path.join(vol_dir, uid + ".json")
    nan = np.full(len(xyz), np.nan)
    if not (os.path.exists(npy) and os.path.exists(js)):
        return nan
    with open(js, encoding="utf-8") as f:
        meta = json.load(f)
    mask = mask_fn(uid) if mask_fn else _blobs_mask(uid, npy)
    if not mask.any():
        return nan
    spacing = np.asarray(meta["spacing_mm"], dtype=float)[::-1]
    dist = ndi.distance_transform_edt(~mask, sampling=spacing)
    del mask
    p = (np.asarray(xyz, dtype=float)[:, ::-1] - np.asarray(meta["origin_mm"], dtype=float)[::-1]) / spacing
    c = np.clip(np.rint(p), 0, np.asarray(dist.shape) - 1).astype(int)
    return dist[c[:, 0], c[:, 1], c[:, 2]] + np.linalg.norm((p - c) * spacing, axis=1)


def _job(args):
    uid, idx, xyz = args
    return idx, scan_out(uid, xyz)


CACHE_DIR = os.path.join(LUNG, "outmm_cache")


def _file_stamp(path):
    """(size, st_mtime_ns) of a file; (-1, -1) when it is missing."""
    try:
        st = os.stat(path)
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return [-1, -1]


def _stamp(uid):
    """The scan's cache stamp: its mask file (nodule_blobs) and its volume's sidecar json."""
    import nodule_blobs as nb
    return np.array(_file_stamp(os.path.join(nb.MASKS, uid + ".npz"))
                    + _file_stamp(os.path.join(LUNG, "vol", uid + ".json")), dtype=np.int64)


def _cache_read(path):
    """(stamp, xyz, out) of a cache file, or None when missing or unreadable."""
    try:
        with np.load(path) as z:
            return z["stamp"], z["xyz"], z["out"]
    except Exception:
        return None


def _cache_write(path, stamp, xyz, out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".%d.part" % os.getpid()
    with open(tmp, "wb") as f:
        np.savez(f, xyz=np.asarray(xyz, np.float64), out=np.asarray(out, np.float64), stamp=np.asarray(stamp, np.int64))
    os.replace(tmp, path)


def out_mm(df, jobs=4, run=None, cache=True, cache_dir=None, stamp_fn=None):
    """out_mm of every row of `df` (seriesuid, coordX, coordY, coordZ), one scan at a time; prints the NaN scans.

    With `cache` each scan's points and values are kept in <cache_dir>/<uid>.npz (default <LUNG>/outmm_cache),
    stamped by its mask and volume json; a scan whose every row is in a current cache file is not measured.
    With a `run` hook the cache is used only when `cache_dir` is given."""
    out = np.full(len(df), np.nan)
    xyz = df[XYZ].to_numpy(float)
    work = [(uid, np.asarray(ix), xyz[ix]) for uid, ix in df.groupby("seriesuid").indices.items()]
    use = cache and (run is None or cache_dir is not None)
    cache_dir = cache_dir or CACHE_DIR
    stamp_fn = stamp_fn or _stamp
    old = {}      # uid -> (stamp, xyz, out) of a current cache file
    stamps = {}
    todo = []
    for w in work:
        uid, idx, pts = w
        if use:
            stamps[uid] = stamp_fn(uid)
            c = _cache_read(os.path.join(cache_dir, uid + ".npz"))
            if c is not None and np.array_equal(c[0], stamps[uid]):
                old[uid] = c
                look = {tuple(p): v for p, v in zip(c[1].tolist(), c[2])}
                got = [look.get(tuple(p)) for p in pts.tolist()]
                if all(g is not None for g in got):
                    out[idx] = np.array(got, dtype=np.float64)
                    continue
        todo.append(w)
    res = []
    if todo:
        if run is None and jobs > 1:
            with ProcessPoolExecutor(jobs) as ex:
                res = list(ex.map(_job, todo))
        else:
            res = [(idx, (run or scan_out)(uid, pts)) for uid, idx, pts in todo]
    for (uid, _, pts), (idx, o) in zip(todo, res):
        out[idx] = o
        if use:
            path = os.path.join(cache_dir, uid + ".npz")
            keep_x, keep_o = np.empty((0, 3)), np.empty(0)
            if uid in old and np.array_equal(old[uid][0], stamp_fn(uid)):
                keep_x, keep_o = old[uid][1], old[uid][2]
            merged = {tuple(p): v for p, v in zip(keep_x.tolist(), keep_o)}
            merged.update({tuple(p): v for p, v in zip(pts.tolist(), np.asarray(o, np.float64))})
            _cache_write(path, stamps[uid], np.array(list(merged), dtype=np.float64).reshape(-1, 3),
                         np.array(list(merged.values()), dtype=np.float64))
    nan_scans = [uid for uid, idx, _ in work if np.isnan(out[idx]).all()]
    if use:
        print("out_mm cache: %d scans read from it, %d measured" % (len(work) - len(todo), len(todo)), flush=True)
    print("%d scans, %d rows; %d scans with out_mm NaN (no volume or empty mask)%s"
          % (len(work), len(df), len(nan_scans), ": " + " ".join(u[-12:] for u in nan_scans[:20]) if nan_scans else ""),
          flush=True)
    return out


def keep_mask(out, d):
    """True where a row stays: out_mm <= d, or NaN (a missing mask never drops anything)."""
    return ~(out > d)


def calibrate(a, led=None):
    import lidc_extra
    _, ann, _ = lidc_extra.extra_reference()
    o = out_mm(ann, a.jobs, cache=not a.no_cache)
    v = o[~np.isnan(o)]
    if not len(v):
        raise SystemExit("no nodule centre has a lung mask")
    res = {"D": float(v.max() + MARGIN_MM), "n": int(len(v)), "max": float(v.max()), "median": float(np.median(v)),
           "p95": float(np.percentile(v, 95)), "outside": int((v > 0).sum()), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("n %d (of %d, %d NaN), outside %d, median %.2f, p95 %.2f, max %.2f mm -> D = %.2f mm"
          % (res["n"], len(ann), len(ann) - len(v), res["outside"], res["median"], res["p95"], res["max"], res["D"]))
    if led is not None:
        led.number("nodule_centres", res["n"], "count").number("nodule_centres_no_mask", len(ann) - len(v), "count")
        led.number("outside_mask", res["outside"], "count").number("out_mm_median", res["median"], "mm")
        led.number("out_mm_p95", res["p95"], "mm").number("out_mm_max", res["max"], "mm").number("d_mm", res["D"], "mm")
    os.makedirs(os.path.dirname(D_FILE), exist_ok=True)
    with open(D_FILE + ".part", "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(D_FILE + ".part", D_FILE)
    return 0


def _reached(cand, ann):
    """(n,) bool: some candidate of the nodule's scan within diameter_mm / 2 of its centre."""
    from scipy.spatial import cKDTree
    hit = np.zeros(len(ann), bool)
    byuid = {u: g[XYZ].to_numpy(float) for u, g in cand.groupby("seriesuid")}
    for i, (uid, pt, dia) in enumerate(zip(ann.seriesuid, ann[XYZ].to_numpy(float), ann.diameter_mm)):
        if uid in byuid and len(byuid[uid]):
            hit[i] = cKDTree(byuid[uid]).query(pt)[0] <= dia / 2.0
    return hit


def filter_cmd(a, led=None):
    if a.d is not None:
        d = a.d
    else:
        with open(a.d_file or D_FILE, encoding="utf-8") as f:
            d = float(json.load(f)["D"])
    df = pd.read_csv(a.candidates)
    o = out_mm(df, a.jobs, cache=not a.no_cache)
    stem = os.path.splitext(a.candidates)[0]
    keep = keep_mask(o, d)
    df.assign(out_mm=o).to_csv(stem + "_outmm.csv", index=False)
    df[keep].to_csv(stem + "_inlung.csv", index=False)
    nan_rows = int(np.isnan(o).sum())
    print("D %.2f mm: %d rows kept, %d dropped (%.2f%%); %d rows with NaN kept, in %d scans"
          % (d, keep.sum(), (~keep).sum(), 100.0 * (~keep).mean(), nan_rows, df.seriesuid[np.isnan(o)].nunique()))
    if led is not None:
        led.settings_from({"d_mm": d, "candidates": a.candidates})
        led.number("rows", len(df), "count").number("rows_kept", keep.sum(), "count")
        led.number("rows_dropped", (~keep).sum(), "count").number("rows_nan_kept", nan_rows, "count")
    ann = pd.read_csv(os.path.join(paths.DATASETS, "public", "luna16", "annotations.csv"))
    ann = ann[ann.seriesuid.isin(set(df.seriesuid))].reset_index(drop=True)
    before = _reached(df, ann)
    after = _reached(df[keep], ann)
    lost = np.flatnonzero(before & ~after)
    print("LUNA16 nodules in these scans: %d; reached before %d, after %d; %d lose every reaching candidate"
          % (len(ann), before.sum(), after.sum(), len(lost)))
    if led is not None:
        led.number("luna16_nodules", len(ann), "count").number("luna16_reached_before", before.sum(), "count")
        led.number("luna16_reached_after", after.sum(), "count").number("luna16_lost", len(lost), "count")
    if len(lost):
        byuid = {u: g for u, g in df.assign(out_mm=o).groupby("seriesuid")}
        for i in lost:
            r = ann.iloc[i]
            g = byuid[r.seriesuid]
            near = np.linalg.norm(g[XYZ].to_numpy(float) - r[XYZ].to_numpy(float).astype(float), axis=1) <= r.diameter_mm / 2.0
            print("  ...%s  %.1f mm  out_mm of its reaching candidates: %s"
                  % (r.seriesuid[-12:], r.diameter_mm, " ".join("%.2f" % v for v in g.out_mm[near])))
    return 0


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        shape = (20, 20, 20)
        mask = np.zeros(shape, bool)
        mask[5:15, 5:15, 5:15] = True
        origin = [10.0, 20.0, 30.0]          # world x, y, z of voxel [0, 0, 0]
        for uid in ("a", "empty"):
            np.save(os.path.join(tmp, uid + ".npy"), np.zeros(shape, np.uint8))
            with open(os.path.join(tmp, uid + ".json"), "w", encoding="utf-8") as f:
                json.dump({"origin_mm": origin, "spacing_mm": [1.0, 1.0, 1.0], "shape": list(shape)}, f)
        masks = {"a": mask, "empty": np.zeros(shape, bool)}

        def world(z, y, x):
            return [origin[0] + x, origin[1] + y, origin[2] + z]
        pts = np.array([world(10, 10, 10),       # inside
                        world(10, 10, 19),       # 5 voxels beyond the mask's last x (14), still in the crop
                        world(10, 10, 24)])      # 5 voxels past the crop's last x (19)
        o = scan_out("a", pts, tmp, masks.__getitem__)
        assert o[0] == 0.0, o
        assert abs(o[1] - 5.0) < 1e-6, o
        assert abs(o[2] - 10.0) < 1e-6, o        # 5 to the crop's edge + 5 to the mask
        for uid in ("empty", "nothere"):
            n = scan_out(uid, pts, tmp, masks.__getitem__)
            assert len(n) == 3 and np.isnan(n).all(), (uid, n)
        df = pd.DataFrame({"seriesuid": ["a", "a", "a", "nothere"], "coordX": [p[0] for p in pts] + [0.0],
                           "coordY": [p[1] for p in pts] + [0.0], "coordZ": [p[2] for p in pts] + [0.0]})
        full = out_mm(df, 1, run=lambda uid, p: scan_out(uid, p, tmp, masks.__getitem__))
        assert np.isnan(full[3]) and np.allclose(full[:3], [0, 5, 10]), full
        assert keep_mask(full, 5.0).tolist() == [True, True, False, True], "D boundary or NaN"
        assert keep_mask(np.array([5.0, 5.0 + 1e-9]), 5.0).tolist() == [True, False]
        # the per-scan cache
        for uid in ("b",):
            np.save(os.path.join(tmp, uid + ".npy"), np.zeros(shape, np.uint8))
            with open(os.path.join(tmp, uid + ".json"), "w", encoding="utf-8") as f:
                json.dump({"origin_mm": origin, "spacing_mm": [1.0, 1.0, 1.0], "shape": list(shape)}, f)
        masks["b"] = mask
        cdir = os.path.join(tmp, "cache")
        calls = []

        def counted(uid, p):
            calls.append(uid)
            return scan_out(uid, p, tmp, masks.__getitem__)
        mstamp = {"a": 1, "b": 1}
        sfn = lambda uid: np.array([mstamp[uid], 7, -1, -1], dtype=np.int64)  # noqa: E731
        df2 = pd.DataFrame({"seriesuid": ["a", "a", "a", "b", "b"], "coordX": [p[0] for p in pts] + [origin[0] + 3.3] * 2,
                            "coordY": [p[1] for p in pts] + [origin[1] + 4.1] * 2, "coordZ": [p[2] for p in pts] + [origin[2] + 9.7, origin[2] + 2.2]})
        first = out_mm(df2, 1, run=counted, cache_dir=cdir, stamp_fn=sfn)
        assert sorted(calls) == ["a", "b"] and len(os.listdir(cdir)) == 2, (calls, os.listdir(cdir))
        del calls[:]
        second = out_mm(df2, 1, run=counted, cache_dir=cdir, stamp_fn=sfn)
        assert not calls and np.array_equal(first, second, equal_nan=True), "second call must read the cache only"
        df3 = pd.concat([df2, pd.DataFrame({"seriesuid": ["a"], "coordX": [origin[0] + 6.5], "coordY": [origin[1] + 6.5],
                                            "coordZ": [origin[2] + 6.5]})], ignore_index=True)
        third = out_mm(df3, 1, run=counted, cache_dir=cdir, stamp_fn=sfn)
        assert calls == ["a"], calls
        assert np.array_equal(third, out_mm(df3, 1, run=counted, cache=False, cache_dir=cdir, stamp_fn=sfn), equal_nan=True)
        del calls[:]
        assert np.array_equal(third, out_mm(df3, 1, run=counted, cache_dir=cdir, stamp_fn=sfn), equal_nan=True) and not calls
        mstamp["b"] = 2
        out_mm(df3, 1, run=counted, cache_dir=cdir, stamp_fn=sfn)
        assert calls == ["b"], "a changed mask stamp must measure that scan again: %s" % calls
        before = sorted(os.listdir(cdir))
        os.remove(os.path.join(cdir, "a.npz"))
        out_mm(df3, 1, run=counted, cache=False, cache_dir=cdir, stamp_fn=sfn)
        assert sorted(os.listdir(cdir)) == [f for f in before if f != "a.npz"], "cache=False must not write"
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    f = sub.add_parser("filter")
    f.add_argument("--candidates", required=True)
    g = f.add_mutually_exclusive_group()
    g.add_argument("--d-file", help="default %s" % D_FILE)
    g.add_argument("--d", type=float, help="D in mm, instead of the file")
    for q in (c, f):
        q.add_argument("--jobs", type=int, default=4)
        q.add_argument("--no-cache", action="store_true", help="measure every scan; neither read nor write outmm_cache")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    name = ("lung mask distance calibrate" if a.cmd == "calibrate"
            else "lung mask filter of " + os.path.basename(a.candidates))
    with ledger.run("measurement", name) as led:
        led.settings_from({"command": a.cmd, "jobs": a.jobs})
        return {"calibrate": calibrate, "filter": filter_cmd}[a.cmd](a, led)


if __name__ == "__main__":
    sys.exit(main())
