"""A pool of 48 mm cubes from every NLST chest CT we hold, for self-supervised pretraining of stage 2.

What. Each NLST series is turned into the same 1 mm lung picture lung_prepare makes, and 48 cubes of 48 mm
are cut from it: 24 centred on random lung voxels, and 24 on random lung voxels above -500 HU (vessels,
nodules, scars). Where a series has fewer such dense voxels, the rest are drawn uniformly and counted.
Two sources: the prepared LIDC-annot-NLST501 volumes (nlst501.py), and NLST's own labelled copy under
Datasets/public/nlst-labelled, read in memory with lung_prepare's own steps. A series met in both is used
once, from the prepared volume.

Why. The stage 2 false-alarm reducer (nodule_fpr.py) has only the labelled nodules to learn from. Its trunk
can first learn what lung tissue looks like from these cubes by restoration (idea 107, docs/efficiency/
README.md), with no answer key. The dense half keeps the pool from being mostly empty parenchyma. The cubes
stay under work/; nothing is written under Datasets/.

Files, in <WORK>/eval/lung: nlst_pool48.npy (uint8, capacity x 48 x 48 x 48; rows [0, rows) are valid),
nlst_pool48.csv (one line per cube), nlst_pool48.json (progress; a rerun resumes where it stopped).

    python modalities/ct/eval/nlst_pool.py build [--jobs 4] [--limit N]
    python modalities/ct/eval/nlst_pool.py check
    python modalities/ct/eval/nlst_pool.py selftest
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)
from core import ledger, paths  # noqa: E402
import lung_prepare as lp  # noqa: E402

CUBE = 48
PER_SERIES = 48
PER_KIND = 24
DENSE_HU = -500.0
MAX_GAP_MM = 2.5
LABELLED = os.path.join(paths.DATASETS, "public", "nlst-labelled")
DEFAULT_FOLDER = os.path.join(paths.WORK, "eval", "lung")
CSV_COLS = ["row", "series_uid", "patient", "source", "kind", "z", "y", "x"]


class Skip(Exception):
    pass


def seed_of(uid):
    return int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big")


def cut(vol, centre, size=CUBE):
    """size^3 cube centred on voxel `centre` (index size // 2 is the centre), 0 outside the volume, as nodule_fpr's
    crop_of cuts it."""
    img = np.zeros((size,) * 3, np.uint8)
    src, dst = [], []
    for c, n in zip(centre, vol.shape):
        s = int(c) - size // 2
        a, b = max(s, 0), min(s + size, n)
        if b <= a:
            return img
        src.append(slice(a, b))
        dst.append(slice(a - s, b - s))
    img[tuple(dst)] = vol[tuple(src)]
    return img


def sample(u8, seed):
    """(cubes, centres, dense_short) of one volume: PER_SERIES cubes centred on lung-mask voxels."""
    hu = lp.to_hu(u8)
    mask = lp.lung_mask(hu)
    lung = np.flatnonzero(mask)
    if lung.size == 0:
        raise Skip("no lung in the volume")
    dense = np.flatnonzero(mask & (hu > DENSE_HU))
    rng = np.random.default_rng(seed)
    n_dense = min(PER_KIND, dense.size)
    dense_pick = dense[rng.choice(dense.size, n_dense, replace=False, shuffle=False)] if n_dense else dense
    n_uniform = PER_SERIES - n_dense
    uniform_pick = lung[rng.choice(lung.size, n_uniform, replace=lung.size < n_uniform, shuffle=False)]
    zyx = [tuple(int(v) for v in p)
           for p in np.stack(np.unravel_index(np.concatenate([uniform_pick, dense_pick]), mask.shape), axis=1)]
    kinds = ["uniform"] * n_uniform + ["dense"] * n_dense
    cubes = np.stack([cut(u8, c) for c in zyx])
    return cubes, [(k, *c) for k, c in zip(kinds, zyx)], PER_KIND - n_dense


def _labelled(folder):
    try:
        vol, zs, sp, _, _ = lp.read_scan(folder)
    except Exception as e:  # noqa: BLE001  one unreadable series must not stop the pool
        raise Skip("read failed: %s: %s" % (type(e).__name__, e))
    gap = float(np.median(np.diff(zs)))
    if gap > MAX_GAP_MM:
        raise Skip("median slice gap %.2f mm, over %.1f mm" % (gap, MAX_GAP_MM))
    got, litres = lp.crop_to_lungs(lp.resample_1mm(vol, zs, sp))
    if got is None:
        raise Skip("%.2f litre of lung, under %.1f" % (litres, lp.MIN_LITRES))
    return lp.to_uint8(got[0])


def _work(task):
    import torch
    torch.set_num_threads(1)
    uid, patient, source, path = task
    try:
        u8 = np.load(path) if source == "nlst501" else _labelled(path)
        cubes, centres, short = sample(u8, seed_of(uid))
    except Skip as e:
        return uid, patient, source, str(e), None, None, 0
    except Exception as e:  # noqa: BLE001
        return uid, patient, source, "failed: %s: %s" % (type(e).__name__, e), None, None, 0
    return uid, patient, source, None, cubes, centres, short


def _results(todo, jobs):
    if jobs <= 1:
        yield from map(_work, todo)
        return
    with ProcessPoolExecutor(jobs) as pool:
        yield from pool.map(_work, todo, chunksize=1)


def candidates(limit=None):
    """(uid, patient, source, path) of every series to pool: prepared NLST501 volumes, then labelled series."""
    import nlst501
    out = []
    for uid, patient, _ in nlst501.scans():
        if nlst501.boxed(uid):
            out.append((uid, patient, "nlst501", lp.paths_for(lp.out_dir(), uid)[0]))
    used = {c[0] for c in out}
    for patient in sorted(os.listdir(LABELLED)):
        pdir = os.path.join(LABELLED, patient)
        if not os.path.isdir(pdir):
            continue
        for uid in sorted(os.listdir(pdir)):
            folder = os.path.join(pdir, uid)
            if uid not in used and os.path.isdir(folder):
                out.append((uid, patient, "labelled", folder))
    return out[:limit] if limit else out


# ---- the pool files --------------------------------------------------------------------------

def _files(folder):
    base = os.path.join(folder, "nlst_pool48")
    return base + ".npy", base + ".csv", base + ".json"


def _atomic(path, write):
    write(path + ".tmp")
    os.replace(path + ".tmp", path)


def _save_json(path, state):
    def write(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f)
    _atomic(path, write)


def _read_state(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _trim_csv(path, rows):
    """Drops csv lines of cubes past `rows`: a run that died between a csv append and its json write."""
    keep = [r for r in _csv_rows(path) if int(r["row"]) < rows]

    def write(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, CSV_COLS)
            w.writeheader()
            w.writerows(keep)
    _atomic(path, write)


def build(candidates, folder=None, jobs=4):
    folder = folder or DEFAULT_FOLDER
    npy, csv_path, json_path = _files(folder)
    total = len(candidates)
    if not total:
        raise SystemExit("no candidate series: nothing to pool")
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(json_path):
        state = _read_state(json_path)
        if state["capacity"] != PER_SERIES * total or not os.path.exists(npy):
            raise SystemExit("%s was made for %d series; delete the nlst_pool48 files to rebuild"
                             % (json_path, state["capacity"] // PER_SERIES))
        _trim_csv(csv_path, state["rows"])
        arr = np.lib.format.open_memmap(npy, mode="r+")
    else:
        state = {"rows": 0, "capacity": PER_SERIES * total, "cube": CUBE, "series_done": [], "skipped": {},
                 "dense_short": 0, "complete": False}
        arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(state["capacity"], CUBE, CUBE, CUBE))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLS)
        _save_json(json_path, state)
    done = set(state["series_done"]) | set(state["skipped"])
    todo = [c for c in candidates if c[0] not in done]
    print("%d candidate series, %d already done, %d to read" % (total, len(done), len(todo)), flush=True)
    t0 = time.time()
    for uid, patient, source, reason, cubes, centres, short in _results(todo, jobs):
        if reason is None:
            r0 = state["rows"]
            arr[r0:r0 + len(cubes)] = cubes
            arr.flush()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([r0 + i, uid, patient, source, kind, z, y, x]
                                        for i, (kind, z, y, x) in enumerate(centres))
            state["rows"] = r0 + len(cubes)
            state["dense_short"] += short
            state["series_done"].append(uid)
        else:
            state["skipped"][uid] = reason
        seen = len(state["series_done"]) + len(state["skipped"])
        state["complete"] = seen == total
        _save_json(json_path, state)
        if seen % 50 == 0:
            print("%d of %d series: %d done, %d skipped (%.0f s this run)"
                  % (seen, total, len(state["series_done"]), len(state["skipped"]), time.time() - t0), flush=True)
    arr.flush()
    del arr
    rows = _csv_rows(csv_path)
    dense = sum(r["kind"] == "dense" for r in rows)
    result = {"series_done": len(state["series_done"]), "series_skipped": len(state["skipped"]),
              "cubes": state["rows"], "dense_share": dense / max(len(rows), 1),
              "dense_short": state["dense_short"], "complete": state["complete"]}
    print("pool: %d series done, %d skipped, %d cubes, dense share %.3f, %d dense draws short, complete %s; %.0f s"
          % (result["series_done"], result["series_skipped"], result["cubes"], result["dense_share"],
             result["dense_short"], result["complete"], time.time() - t0), flush=True)
    return result


def check(folder=None):
    npy, csv_path, json_path = _files(folder or DEFAULT_FOLDER)
    state = _read_state(json_path)
    arr = np.load(npy, mmap_mode="r")
    rows = state["rows"]
    assert arr.shape[0] == state["capacity"] and rows <= state["capacity"], "npy and json disagree on capacity"
    csv_rows = _csv_rows(csv_path)
    assert [int(r["row"]) for r in csv_rows] == list(range(rows)), "csv rows are not 0..%d" % (rows - 1)
    dense = sum(r["kind"] == "dense" for r in csv_rows)
    print("series done %d, skipped %d, rows %d, complete %s"
          % (len(state["series_done"]), len(state["skipped"]), rows, state["complete"]))
    reasons = collections.Counter(re.sub(r"\d+(\.\d+)?", "#", r) for r in state["skipped"].values())
    for reason, n in reasons.most_common():
        print("  skipped %d: %s" % (n, reason))
    print("dense centres %d of %d (%.1f%%), dense draws short %d"
          % (dense, rows, 100 * dense / max(rows, 1), state["dense_short"]))
    s = s2 = 0.0
    for a in range(0, rows, 32):
        block = np.asarray(arr[a:min(a + 32, rows)], dtype=np.float64)
        s += float(block.sum())
        s2 += float((block * block).sum())
    n = rows * CUBE ** 3
    mean = s / max(n, 1)
    sd = float(np.sqrt(max(s2 / max(n, 1) - mean * mean, 0.0)))
    print("cube values: mean %.1f, sd %.1f (uint8: 0 = -1200 HU, 255 = 600 HU)" % (mean, sd))
    print("check ok")


def selftest():
    import tempfile

    def ref_cut(vol, c, size=CUBE):
        pv = np.pad(vol, size)
        return pv[tuple(slice(v + size - size // 2, v + size - size // 2 + size) for v in c)]

    def phantom(dense):
        hu = np.full((80, 80, 80), 40, np.float32)
        hu[20:60, 25:55, 10:30] = -800
        hu[20:60, 25:55, 50:70] = -800
        if dense:
            hu[30:40, 35:45, 14:24] = -300
        return lp.to_uint8(hu)

    rng = np.random.default_rng(1)
    vol = rng.integers(0, 256, (60, 50, 40), dtype=np.uint8)
    for c in [(0, 0, 0), (59, 49, 39), (30, 25, 20), (0, 25, 39), (59, 0, 20)]:
        assert np.array_equal(cut(vol, c), ref_cut(vol, c)), "cut at %s is not the padded cube" % (c,)
    assert cut(vol, (0, 0, 0)).shape == (CUBE,) * 3

    u8 = phantom(dense=True)
    hu = lp.to_hu(u8)
    mask = lp.lung_mask(hu)
    assert mask.sum() > 0, "phantom has no lung mask"
    cubes, centres, short = sample(u8, seed_of("selftest.1"))
    assert cubes.shape == (PER_SERIES, CUBE, CUBE, CUBE) and cubes.dtype == np.uint8
    assert len(centres) == PER_SERIES and short == 0
    assert sum(k == "dense" for k, *_ in centres) == PER_KIND
    for (k, z, y, x), cube in zip(centres, cubes):
        assert mask[z, y, x], "centre %s is outside the lung mask" % ((z, y, x),)
        if k == "dense":
            assert hu[z, y, x] > DENSE_HU, "dense centre %s is %.0f HU" % ((z, y, x), hu[z, y, x])
        assert np.array_equal(cube, ref_cut(u8, (z, y, x))), "cube at %s is wrong" % ((z, y, x),)
    _, centres2, short2 = sample(phantom(dense=False), seed_of("selftest.2"))
    assert short2 == PER_KIND and len(centres2) == PER_SERIES
    assert sum(k == "dense" for k, *_ in centres2) == 0

    with tempfile.TemporaryDirectory(prefix="nlst_pool_") as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        vols = {"1.2.3.1": phantom(True), "1.2.3.2": phantom(False)}
        cands = []
        for i, (uid, v) in enumerate(vols.items()):
            p = os.path.join(src, uid + ".npy")
            np.save(p, v)
            cands.append((uid, "p%d" % i, "nlst501", p))
        out = os.path.join(tmp, "pool")
        r1 = build(cands, folder=out, jobs=1)
        assert r1["complete"] and r1["cubes"] == 2 * PER_SERIES, r1
        assert r1["series_done"] == 2 and r1["series_skipped"] == 0 and r1["dense_short"] == PER_KIND, r1
        npy, csv_path, json_path = _files(out)
        state = _read_state(json_path)
        assert state["series_done"] == ["1.2.3.1", "1.2.3.2"] and state["skipped"] == {}
        arr = np.load(npy, mmap_mode="r")
        assert arr.shape == (2 * PER_SERIES, CUBE, CUBE, CUBE)
        rows = _csv_rows(csv_path)
        assert [int(r["row"]) for r in rows] == list(range(2 * PER_SERIES))
        for r in rows:
            c = (int(r["z"]), int(r["y"]), int(r["x"]))
            assert np.array_equal(arr[int(r["row"])], ref_cut(vols[r["series_uid"]], c)), \
                "row %s is not the cube at its centre" % r["row"]
        del arr
        r2 = build(cands, folder=out, jobs=1)
        assert r2 == r1, "a second build changed the result"
        assert _csv_rows(csv_path) == rows, "a second build changed the csv"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("--limit", type=int)
    sub.add_parser("check")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "build":
        todo = candidates(a.limit)
        with ledger.run("experiment", "nlst pool") as led:
            led.settings_from({"cube": CUBE, "per_series": PER_SERIES, "per_kind": PER_KIND,
                               "dense_hu": DENSE_HU, "limit": a.limit or 0})
            r = build(todo, jobs=a.jobs)
            led.numbers_from({"series_done": r["series_done"], "series_skipped": r["series_skipped"],
                              "cubes": r["cubes"], "dense_share": r["dense_share"]})
    elif a.cmd == "check":
        check()
    else:
        selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
