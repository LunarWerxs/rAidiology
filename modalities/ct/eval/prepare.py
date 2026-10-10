"""The training cache for the CT segmenter: every scan read once, and never again.

A scan in this dataset is one CT volume and 117 separate mask files, and opening those 118
compressed files is the slow part of everything: two seconds a scan, most of an hour for a pass
over development. A trainer that did that per sample would spend its life decompressing. So each
scan is read once here and left as three things:

- **the picture at 3 mm** (`lo/<scan>_ct.npy`, int16 Hounsfield units). The dataset is 1.5 mm in
  every direction, so 3 mm is exactly every 2 x 2 x 2 block averaged, with no interpolation to argue
  about. It is what the segmenter sees: a 128-voxel cube then spans 38 cm, enough to tell the
  eleventh rib from the twelfth, which a network looking through a keyhole cannot do.
- **the labels at 3 mm** (`lo/<scan>_lab.npy`, uint8, 0 is background): the commonest label in each
  block, with a tie going to a structure rather than to background, so a rib two voxels thick is
  not voted out of existence.
- **the truth on the scan's own grid** (`hi/<scan>.npz`), which is what the scorer's three counts
  are taken against. It is one label per voxel plus a short list of the voxels that sit in more
  than one expert mask. That list is not a nicety: the masks do overlap (measured 2026-10-01, 21 to
  1,584 voxels a scan), so a label map alone undercounts the truth, and the contract's
  `truth_voxels` is the count in each mask, not the count in a flattened copy of them.

Both stores are plain arrays a trainer can memory-map, so a training sample costs the bytes of its
own crop and nothing else.

The test scans are not prepared here unless the test has been deliberately unlocked: preparing them
means reading their expert masks, and the only thing that should do that is the final evaluation.

    python modalities/ct/eval/prepare.py dev             the 1,139 development scans
    python modalities/ct/eval/prepare.py check           every truth count against the baselines' rows
    python modalities/ct/eval/prepare.py ceiling         the best Dice 3 mm labels could ever score
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import datasets  # noqa: E402
from core import ledger, paths  # noqa: E402

OUT = os.path.join(paths.WORK, "eval", "ct")
LO = os.path.join(OUT, "cache", "lo")
HI = os.path.join(OUT, "cache", "hi")
HI15 = os.path.join(OUT, "cache", "hi15")
FINAL_STATE = os.path.join(OUT, "final_eval.json")       # final_eval.py's STATE
POINTS = os.path.join(OUT, "cache", "points_%s.npz")
SPACING = 1.5                 # what the dataset is published at, in millimetres, on every axis
FACTOR = 2                    # so the working grid is 3 mm
AIR = -1024                   # what the picture is padded with where a scan has an odd edge
POINTS_PER_CLASS = 48         # places to centre a training crop on, per structure per scan


def lo_files(scan):
    return os.path.join(LO, "%s_ct.npy" % scan), os.path.join(LO, "%s_lab.npy" % scan)


def hi_file(scan):
    return os.path.join(HI, "%s.npz" % scan)


def hi15_files(scan):
    return os.path.join(HI15, "%s_ct.npy" % scan), os.path.join(HI15, "%s_lab.npy" % scan)


def pad_even(volume, value):
    """Pad the far edge of each odd axis by one voxel, so every 2 x 2 x 2 block is whole."""
    pad = [(0, s % FACTOR) for s in volume.shape]
    if not any(p[1] for p in pad):
        return volume
    return np.pad(volume, pad, mode="constant", constant_values=value)


def blocks(volume):
    """The volume as (x/2, y/2, z/2, 8): each row of 8 is one 2 x 2 x 2 block."""
    x, y, z = (s // FACTOR for s in volume.shape)
    return (volume.reshape(x, FACTOR, y, FACTOR, z, FACTOR)
            .transpose(0, 2, 4, 1, 3, 5).reshape(x, y, z, FACTOR ** 3))


def average_down(ct):
    """The picture at 3 mm: the mean of each block. An average, never a pick of one voxel in eight,
    which would alias a thin bright rib into a dotted line."""
    b = blocks(pad_even(ct, AIR).astype(np.float32))
    return np.rint(b.mean(axis=3)).astype(np.int16)


def vote_down(labels):
    """The labels at 3 mm: the commonest label in each block, a tie going to the higher label.

    Background is label 0, so a tie between a structure and nothing goes to the structure. Most
    blocks hold one label throughout, so only the mixed ones are counted."""
    b = blocks(pad_even(labels, 0))
    flat = b.reshape(-1, FACTOR ** 3)
    out = flat[:, 0].copy()
    mixed = np.flatnonzero((flat != flat[:, :1]).any(axis=1))
    if mixed.size:
        m = np.sort(flat[mixed], axis=1)
        counts = np.stack([(m == m[:, j:j + 1]).sum(axis=1) for j in range(FACTOR ** 3)], axis=1)
        # The last position holding the highest count: sorted ascending, that is the higher label.
        best = FACTOR ** 3 - 1 - np.argmax(counts[:, ::-1], axis=1)
        out[mixed] = m[np.arange(mixed.size), best]
    return out.reshape(b.shape[:3])


def back_up(lo, shape):
    """A 3 mm volume laid back over a scan's own grid, each voxel repeated into its block."""
    up = lo.repeat(FACTOR, axis=0).repeat(FACTOR, axis=1).repeat(FACTOR, axis=2)
    return up[:shape[0], :shape[1], :shape[2]]


def _one(args):
    import nibabel as nib  # noqa: PLC0415

    scan, names, seed = args
    try:
        img = nib.load(datasets.scan_path(scan))
        zooms = tuple(round(float(v), 3) for v in img.header.get_zooms()[:3])
        codes = "".join(nib.aff2axcodes(img.affine))
        if zooms != (SPACING,) * 3 or codes != "RAS":
            # Said rather than resampled quietly: everything downstream assumes one grid, and a scan
            # on another one would be scored against masks it no longer lines up with.
            return {"scan": scan, "ok": 0, "why": "grid %s %s" % (zooms, codes)}
        ct = np.asanyarray(img.dataobj)
        # A NIfTI file is stored first-axis-fastest, and so is the array that comes out of it.
        # Walking 117 of those in the other order copies every one of them: measured, that was two
        # thirds of a scan's preparation. So the masks are read in the order they are stored, and
        # the label map is turned round once at the end.
        labels = np.zeros(ct.shape, dtype=np.uint8, order="F")
        flat = labels.reshape(-1, order="F")
        truth = np.zeros(len(names), dtype=np.int64)
        extra_at, extra_is = [], []
        for i, n in enumerate(names):
            m = np.asanyarray(nib.load(datasets.mask_path(scan, n)).dataobj)
            if m.shape != ct.shape:
                return {"scan": scan, "ok": 0, "why": "mask %s is %s, scan is %s" % (n, m.shape, ct.shape)}
            at = np.flatnonzero(m.reshape(-1, order="F"))
            if not at.size:
                continue
            truth[i] = at.size
            taken = flat[at] != 0
            flat[at[~taken]] = i + 1
            if taken.any():
                extra_at.append(at[taken])
                extra_is.append(np.full(int(taken.sum()), i + 1, dtype=np.uint8))
    except Exception as e:  # noqa: BLE001  a scan that will not open is a row, not a crash
        return {"scan": scan, "ok": 0, "why": "unreadable: %s" % str(e)[:120]}

    labels = np.ascontiguousarray(labels)
    ct = np.ascontiguousarray(ct)
    extra_is = np.concatenate(extra_is) if extra_is else np.zeros(0, dtype=np.uint8)
    if extra_at:
        # The overlap voxels were found in stored order; everything downstream indexes row-major.
        where = np.unravel_index(np.concatenate(extra_at), ct.shape, order="F")
        extra_at = np.ravel_multi_index(where, ct.shape).astype(np.int64)
    else:
        extra_at = np.zeros(0, dtype=np.int64)
    ct_lo = average_down(ct)
    lab_lo = vote_down(labels)

    # What 3 mm labels would score if a model reproduced them perfectly: the ceiling of the working
    # grid, and a check that going down and coming back up lands on the same voxels.
    up = back_up(lab_lo, ct.shape)
    n_all = len(names) + 1
    pred = np.bincount(up.ravel(), minlength=n_all)[1:].astype(np.int64)
    same = labels == up
    both = np.bincount(labels[same].ravel(), minlength=n_all)[1:].astype(np.int64)
    if extra_at.size:
        hit = up.reshape(-1)[extra_at] == extra_is
        both += np.bincount(extra_is[hit], minlength=n_all)[1:]

    rng = np.random.default_rng([seed, int(scan[1:])])
    flat_lo = lab_lo.ravel()
    order = np.argsort(flat_lo, kind="stable")
    present, start = np.unique(flat_lo[order], return_index=True)
    stop = list(start[1:]) + [flat_lo.size]
    pt_is, pt_at = [], []
    for c, a, b in zip(present, start, stop):
        if c == 0:
            continue
        pick = order[a:b]
        if pick.size > POINTS_PER_CLASS:
            pick = rng.choice(pick, POINTS_PER_CLASS, replace=False)
        pt_is.append(np.full(pick.size, c, dtype=np.uint8))
        pt_at.append(pick.astype(np.int64))

    os.makedirs(LO, exist_ok=True)
    os.makedirs(HI, exist_ok=True)
    ct_file, lab_file = lo_files(scan)
    np.save(ct_file, ct_lo)
    np.save(lab_file, lab_lo)
    np.savez_compressed(hi_file(scan), labels=labels, extra_at=extra_at, extra_is=extra_is,
                        truth=truth, ceiling_pred=pred, ceiling_both=both)
    return {"scan": scan, "ok": 1, "why": "", "shape": ct_lo.shape, "voxels": int(ct.size),
            "overlap": int(extra_at.size),
            "pt_is": np.concatenate(pt_is) if pt_is else np.zeros(0, dtype=np.uint8),
            "pt_at": np.concatenate(pt_at) if pt_at else np.zeros(0, dtype=np.int64)}


def scans_of(split_name):
    split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    return sorted(split["dev_folds"]) if split_name == "dev" else sorted(split["test"])


def build(split_name, workers=None, limit=None):
    datasets.guard_test(split_name)
    names = datasets.structures()
    scans = scans_of(split_name)
    if limit:
        scans = scans[:limit]
    # A quarter of the machine, not most of it: this box is also somebody's desk.
    workers = workers or max(2, (os.cpu_count() or 8) // 4)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as pool:
        jobs = [(s, names, datasets.SEED) for s in scans]
        for i, row in enumerate(pool.map(_one, jobs, chunksize=2), 1):
            rows.append(row)
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                print("  %d of %d, %.1f scans/s, %.0f s left"
                      % (i, len(scans), rate, (len(scans) - i) / rate), flush=True)
    good = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    # One file of crop centres for the whole split: a trainer's worker loads it once.
    np.savez(POINTS % split_name,
             scan=np.array([r["scan"] for r in good]),
             shape=np.array([r["shape"] for r in good], dtype=np.int64).reshape(-1, 3),
             count=np.array([r["pt_is"].size for r in good], dtype=np.int64),
             pt_is=np.concatenate([r["pt_is"] for r in good]) if good else np.zeros(0, np.uint8),
             pt_at=np.concatenate([r["pt_at"] for r in good]) if good else np.zeros(0, np.int64))
    pd.DataFrame([{"scan": r["scan"], "why": r["why"]} for r in bad],
                 columns=["scan", "why"]).to_csv(os.path.join(OUT, "cache", "unprepared_%s.csv" % split_name),
                                                 index=False)
    return good, bad, time.time() - t0


def unprepared(split_name):
    """{scan: why} for the scans the cache could not hold. Empty when everything went in."""
    f = os.path.join(OUT, "cache", "unprepared_%s.csv" % split_name)
    if not os.path.exists(f):
        raise SystemExit("No cache for %s yet. Run: python modalities/ct/eval/prepare.py %s"
                         % (split_name, split_name))
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    return dict(zip(d["scan"], d["why"]))


def points(split_name):
    """{scan: (3 mm shape, class of each crop centre, flat index of each crop centre)}."""
    z = np.load(POINTS % split_name)
    # Each array out of the archive once. Indexing the archive inside the loop reads the whole
    # array again every time: measured, 16 seconds for a table that takes a few milliseconds.
    pt_is, pt_at = z["pt_is"], z["pt_at"]
    out, at = {}, 0
    for scan, shape, n in zip(z["scan"], z["shape"], z["count"]):
        out[str(scan)] = (tuple(int(v) for v in shape), pt_is[at:at + n], pt_at[at:at + n])
        at += int(n)
    return out


def check():
    """Every truth count in the cache against the count the baselines' rows carry.

    Two readers of the same masks, written a day apart for different jobs. If they disagree on a
    single (scan, structure) pair, one of them misread a file, and a segmenter scored against the
    wrong one of them is scored against nothing."""
    names = datasets.structures()
    rows = pd.read_csv(os.path.join(OUT, "pred_baselines_dev.csv"),
                       usecols=["method", "scan", "structure", "truth_voxels", "in_scope"])
    rows = rows[(rows["method"] == "always-empty") & (rows["in_scope"] == 1)]
    want = rows.pivot(index="scan", columns="structure", values="truth_voxels")[names]
    skipped = unprepared("dev")
    bad = pairs = 0
    for scan in want.index:
        if scan in skipped:
            continue
        z = np.load(hi_file(scan))
        got = z["truth"]
        flat = np.bincount(z["labels"].ravel(), minlength=len(names) + 1)[1:]
        flat = flat + np.bincount(z["extra_is"], minlength=len(names) + 1)[1:]
        theirs = want.loc[scan].to_numpy()
        pairs += len(names)
        for i in np.flatnonzero((got != theirs) | (flat != theirs)):
            bad += 1
            if bad <= 10:
                print("DIFFERS %s %s: cache %d, label map plus overlaps %d, baselines %d"
                      % (scan, names[i], got[i], flat[i], theirs[i]))
    print("%d (scan, structure) pair(s) over %d scan(s), %d skipped, %d differing"
          % (pairs, pairs // len(names), len(skipped), bad))
    if bad or not pairs:
        raise SystemExit("The cache's truth is not the baselines' truth. Do not train against it.")
    return pairs


def ceiling():
    """Mean Dice, each structure once, if a model reproduced the 3 mm labels exactly."""
    names = datasets.structures()
    skipped = unprepared("dev")
    dice = [[] for _ in names]
    overlap = truth_all = 0
    for scan in scans_of("dev"):
        if scan in skipped:
            continue
        z = np.load(hi_file(scan))
        t, p, b = z["truth"], z["ceiling_pred"], z["ceiling_both"]
        overlap += int(z["extra_at"].size)
        truth_all += int(t.sum())
        for i in np.flatnonzero(t):
            dice[i].append(2.0 * b[i] / (t[i] + p[i]))
    per = np.array([np.mean(d) if len(d) >= 10 else np.nan for d in dice])
    worst = np.argsort(np.where(np.isnan(per), 9, per))[:6]
    print("ceiling of the 3 mm grid: %.4f mean Dice over %d structure(s)"
          % (np.nanmean(per), int(np.isfinite(per).sum())))
    print("  lowest: " + ", ".join("%s %.3f" % (names[i], per[i]) for i in worst))
    print("  voxels in more than one expert mask: %d of %d (%.4f%%)"
          % (overlap, truth_all, 100.0 * overlap / max(truth_all, 1)))
    return float(np.nanmean(per)), overlap, truth_all


def _one_hi15(args):
    import nibabel as nib  # noqa: PLC0415

    scan, seed = args
    try:
        # Read the scan at 1.5 mm (the dataset's own grid)
        img = nib.load(datasets.scan_path(scan))
        zooms = tuple(round(float(v), 3) for v in img.header.get_zooms()[:3])
        codes = "".join(nib.aff2axcodes(img.affine))
        if zooms != (SPACING,) * 3 or codes != "RAS":
            return {"scan": scan, "ok": 0, "why": "grid %s %s" % (zooms, codes)}
        ct = np.asanyarray(img.dataobj)
        ct_padded = pad_even(np.ascontiguousarray(ct), AIR)
        ct_padded = ct_padded.astype(np.int16)
        ct_file, _ = hi15_files(scan)
        np.save(ct_file, ct_padded)

        # Read the labels at 1.5 mm from the hi npz file
        z = np.load(hi_file(scan))
        labels = np.ascontiguousarray(z["labels"])
        lab_padded = pad_even(labels, 0).astype(np.uint8)
        _, lab_file = hi15_files(scan)
        np.save(lab_file, lab_padded)

        voxels_unpadded = int(np.prod(ct.shape))
        voxels_padded = int(np.prod(ct_padded.shape))
        gb_padded = voxels_padded * 3 / 2 ** 30  # int16 picture plus uint8 labels
        return {"scan": scan, "ok": 1, "why": "", "voxels": voxels_unpadded, "voxels_padded": voxels_padded,
                "gb": gb_padded}
    except Exception as e:  # noqa: BLE001
        return {"scan": scan, "ok": 0, "why": "unreadable: %s" % str(e)[:120]}


def open_look(look, state_file=None, latched=None):
    """The second look `look` final_eval recorded, if it is open: not finished, and its latch in the ledger.
    Returns its entry, or exits. The test split is read at 1.5 mm only inside one. `latched` is for the selftest."""
    if look is None:
        raise SystemExit("A test read at 1.5 mm is made inside a second look: pass --look N, the look final_eval opened.")
    state_file = state_file or FINAL_STATE
    state = json.load(open(state_file, encoding="utf-8")) if os.path.exists(state_file) else {}
    found = [l for l in state.get("looks", []) if l.get("look") == look]
    if not found:
        raise SystemExit("No second look %s in %s: the test split is read only inside one." % (look, state_file))
    entry = found[0]
    if entry.get("finished"):
        raise SystemExit("Look %s finished on %s: a look's test read happens once." % (look, entry["finished"]))
    # final_eval.py writes this latch into the ledger before it opens the test (LEDGER_NAME, then the look's name)
    name = "ct test opened second look v%d" % entry["version"]
    if entry.get("with_version") is not None:
        name += "+%d" % entry["with_version"]
    if latched is None:
        latched = any(r.get("name") == name for r in ledger.records("final_eval"))
    if not latched:
        raise SystemExit("The ledger has no latch named '%s': the test split is not opened for this look." % name)
    return entry


def build_hi15(workers=None, limit=None, scans_filter=None, split_name="dev", look=None):
    """Build the 1.5 mm cache (hi15): the scan and labels at the dataset's own grid, pad_even'd. The test
    split's is built only inside the second look `look` (open_look)."""
    datasets.guard_test(split_name)
    if split_name == "test":
        open_look(look)
    scans = scans_of(split_name)
    if scans_filter:
        scans = [s for s in scans if s in scans_filter]
    if limit:
        scans = scans[:limit]
    workers = workers or max(2, (os.cpu_count() or 8) // 4)
    os.makedirs(HI15, exist_ok=True)
    rows, t0, skipped = [], time.time(), unprepared(split_name)
    with ProcessPoolExecutor(workers) as pool:
        jobs = [(s, datasets.SEED) for s in scans if s not in skipped]
        # Check which files already exist to skip them
        jobs_to_do = []
        for scan, seed in jobs:
            ct_file, lab_file = hi15_files(scan)
            if not (os.path.exists(ct_file) and os.path.exists(lab_file)):
                jobs_to_do.append((scan, seed))
        for i, row in enumerate(pool.map(_one_hi15, jobs_to_do, chunksize=2), 1):
            rows.append(row)
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                print("  %d of %d, %.1f scans/s, %.0f s left"
                      % (i, len(jobs_to_do), rate, (len(jobs_to_do) - i) / rate), flush=True)
    good = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    total_gb = sum(g["gb"] for g in good)
    seconds = time.time() - t0
    return good, bad, total_gb, seconds


def selftest():
    """Test hi15 preparation on a tiny synthetic volume with no dataset access."""
    import tempfile

    # Create synthetic CT and labels volumes
    ct = np.ones((3, 3, 3), dtype=np.int16) * 100
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[0:2, 0:2, 0:2] = 1

    # Test pad_even: odd shape should be padded to even
    ct_padded = pad_even(ct, AIR)
    assert ct_padded.shape == (4, 4, 4), f"Expected (4, 4, 4), got {ct_padded.shape}"
    assert ct_padded[3, 3, 3] == AIR, "Padding should be AIR"
    assert np.array_equal(ct_padded[0:3, 0:3, 0:3], ct), "Original data should be unchanged"

    lab_padded = pad_even(labels, 0)
    assert lab_padded.shape == (4, 4, 4), f"Expected (4, 4, 4), got {lab_padded.shape}"
    assert lab_padded[3, 3, 3] == 0, "Padding should be 0"

    # Test that average_down of hi15 CT gives same result as lo cache would
    # The hi15 CT (at 1.5 mm) should when downsampled by 2x equal the 3mm cache
    ct_lo_from_hi = average_down(ct)
    # This is a basic check that average_down works on the hi15 data
    assert ct_lo_from_hi.shape == (2, 2, 2), f"Expected (2, 2, 2), got {ct_lo_from_hi.shape}"

    # Test vote_down on labels
    lab_lo = vote_down(labels)
    assert lab_lo.shape == (2, 2, 2), f"Expected (2, 2, 2), got {lab_lo.shape}"

    print("prepare.py selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["dev", "test", "check", "ceiling", "hi15", "selftest"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--scans", help="comma-separated list of scans to process")
    ap.add_argument("--split", choices=["dev", "test"], default="dev", help="hi15: which split's scans")
    ap.add_argument("--look", type=int, help="hi15 with --split test: the second look final_eval opened")
    a = ap.parse_args()
    if a.what == "selftest":
        selftest()
        return
    if a.what == "check":
        with ledger.run("prepare", "ct cache check") as r:
            r.number("pairs", check(), "count")
        return
    if a.what == "ceiling":
        with ledger.run("measurement", "ct ceiling of the 3 mm grid") as r:
            value, overlap, truth_all = ceiling()
            r.number("mean_dice", value).number("overlap_voxels", overlap, "count")
            r.number("overlap_share", overlap / max(truth_all, 1))
        return
    if a.what == "hi15":
        scans_filter = set(a.scans.split(",")) if a.scans else None
        with ledger.run("prepare", "ct cache hi15") as r:
            good, bad, total_gb, seconds = build_hi15(a.workers, a.limit, scans_filter, a.split, a.look)
            r.number("scans", len(good), "count").number("unprepared", len(bad), "count")
            r.number("total_gb", total_gb, "GB")
            r.number("scans_per_second", len(good) / max(seconds, 1e-9), "count")
        print("%d scan(s) cached under %s, %d could not be" % (len(good), HI15, len(bad)))
        print("  total: %.1f GB in %.0f seconds" % (total_gb, seconds))
        for b in bad[:10]:
            print("  %s: %s" % (b["scan"], b["why"]))
        return
    with ledger.run("prepare", "ct cache %s" % a.what) as r:
        good, bad, seconds = build(a.what, a.workers, a.limit)
        r.number("scans", len(good), "count").number("unprepared", len(bad), "count")
        r.number("scans_per_second", len(good) / max(seconds, 1e-9), "count")
        r.number("overlap_voxels", sum(g["overlap"] for g in good), "count")
    print("%d scan(s) cached under %s, %d could not be" % (len(good), os.path.join(OUT, "cache"), len(bad)))
    for b in bad[:10]:
        print("  %s: %s" % (b["scan"], b["why"]))


if __name__ == "__main__":
    main()
