"""The data session's 1.5 mm CT training copy, round-tripped, for the lung nets to read (idea 139).

What. Each LIDC scan the lung nets read out of fold goes through tools/train_copy.py's own ct_copy() on its DICOM files
(every axis finer than 1.5 mm brought to 1.5 mm, linear, -1024 to 3071 HU, int16 NIfTI), is read back, and is prepared
by lung_prepare's own steps (1 mm, the lung crop, 8 bits) into <OUT>/vol/. A net pointed there (RAIDIOLOGY_LUNG_DIR=<OUT>)
reads what it would read had the scan come from a training copy. The original volume's meta is kept (its patient and
"original_*" fields, so every list and split is the same); the copy's grid and bytes are added under "roundtrip". A scan
the copy cannot hold (no evenly spaced stack, stacks on different grids, under 0.5 litre of lung) keeps its original
volume, and <OUT>/roundtrip.csv says so and why. Anything else (out of memory, a bug) writes nothing for that scan and
fails the build, so a rerun redoes it: an original quietly standing in for a copy would hide the very difference measured.

Why. The owner's storage question (docs/efficiency/README.md, idea 139): is the 1.5 mm copy enough for the 1 mm lung
nets, or must the CT training copies be stored exactly?

    python tools/roundtrip15.py build --out <dir> [--jobs 3] [--limit N]
    python tools/roundtrip15.py malig --version 5 --out <csv>     (scores from the volumes RAIDIOLOGY_LUNG_DIR names)
    python tools/roundtrip15.py malig-compare <a.csv> <b.csv> [--margin 0.01] [--name ...]
"""
import argparse
import csv
import gc
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "modalities", "ct", "eval")
sys.path.insert(0, ROOT)
sys.path.insert(0, EVAL)
sys.path.insert(0, HERE)
from core import ledger, paths  # noqa: E402
import lung_prepare as lp  # noqa: E402

ORIG = os.path.join(paths.WORK, "eval", "lung")      # the originals, whatever RAIDIOLOGY_LUNG_DIR says
COLS = ["seriesuid", "status", "reason", "stacks", "dicom_bytes", "copy_bytes", "copy_spacing_mm", "seconds"]
MALIG_VERSION = 5


class CopyLimit(ValueError):
    """What the copy itself cannot hold: the scan keeps its original volume."""


def read_copy(parts):
    """ct_copy's stacks read back as lung_prepare.read_scan's (HU volume (z, y, x), z of each slice, (row, col)
    spacing, x0, y0): flipped so x and y rise with world x and y, the stacks of one series put back together by z."""
    import SimpleITK as sitk
    slabs = []
    for path, _ in parts:
        img = sitk.ReadImage(path)
        a = sitk.GetArrayFromImage(img).astype(np.float32)
        sx, sy, sz = img.GetSpacing()
        ox, oy, oz = img.GetOrigin()
        d = np.array(img.GetDirection()).reshape(3, 3)
        if np.abs(np.abs(d) - np.eye(3)).max() > 1e-3:
            raise CopyLimit("an oblique stack")
        nz, ny, nx = a.shape
        x0, y0 = ox, oy
        if d[0, 0] < 0:
            a = a[:, :, ::-1]
            x0 = ox - (nx - 1) * sx
        if d[1, 1] < 0:
            a = a[:, ::-1, :]
            y0 = oy - (ny - 1) * sy
        zs = oz + d[2, 2] * sz * np.arange(nz)
        if d[2, 2] < 0:
            a, zs = a[::-1], zs[::-1]
        slabs.append((np.ascontiguousarray(a), zs, (sy, sx), x0, y0))
    first = slabs[0]
    for s in slabs[1:]:
        if (s[0].shape[1:] != first[0].shape[1:] or not np.allclose(s[2], first[2], atol=1e-4)
                or abs(s[3] - first[3]) > 0.01 or abs(s[4] - first[4]) > 0.01):
            raise CopyLimit("stacks on different grids")
    vol = np.concatenate([s[0] for s in slabs])
    zs = np.concatenate([s[1] for s in slabs])
    order = np.argsort(zs, kind="stable")
    vol, zs = vol[order], zs[order]
    keep = np.r_[True, np.diff(zs) > 1e-3]
    return np.ascontiguousarray(vol[keep]), zs[keep], [first[2][0], first[2][1]], first[3], first[4]


def _write(npy, js, u8, meta):
    def w_npy(p):
        with open(p, "wb") as f:
            np.save(f, u8)

    def w_js(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    lp._atomic(npy, w_npy)
    lp._atomic(js, w_js)


def one(job):
    """One scan -> its manifest row; the volume written to <out>/vol/."""
    uid, folder, out, tmp = job
    import torch
    torch.set_num_threads(1)
    import train_copy as tc
    t0 = time.time()
    onpy, ojs = lp.paths_for(ORIG, uid)
    npy, js = lp.paths_for(out, uid)
    with open(ojs, encoding="utf-8") as f:
        meta0 = json.load(f)
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    row = {"seriesuid": uid, "status": "", "reason": "", "stacks": 0, "dicom_bytes": sum(os.path.getsize(p) for p in files),
           "copy_bytes": 0, "copy_spacing_mm": "", "seconds": 0.0}
    base = os.path.join(tmp, uid)
    try:
        try:
            parts = tc.ct_copy(files, base)
        except tc.Unusable as e:
            raise CopyLimit("the copy refuses it: %s" % e) from e
        row["stacks"] = len(parts)
        row["copy_bytes"] = sum(os.path.getsize(p) for p, _ in parts)
        vol, zs, sp, x0, y0 = read_copy(parts)
        row["copy_spacing_mm"] = "%.4f %.4f %.4f" % (sp[1], sp[0], float(np.median(np.diff(zs))))
        hu = lp.resample_1mm(vol, zs, sp)
        del vol
        got, litres = lp.crop_to_lungs(hu)
        if got is None:
            raise CopyLimit("%.2f litre of lung from the copy" % litres)
        crop, lo = got
        u8 = lp.to_uint8(crop)
        meta = dict(meta0)
        meta.update({"origin_mm": [x0 + lo[2], y0 + lo[1], float(zs[0]) + lo[0]], "shape": list(u8.shape),
                     "lung_litres": round(litres, 3),
                     "roundtrip": {"stacks": row["stacks"], "copy_bytes": row["copy_bytes"], "dicom_bytes": row["dicom_bytes"],
                                   "copy_spacing_mm": [round(sp[1], 4), round(sp[0], 4), round(float(np.median(np.diff(zs))), 4)]}})
        _write(npy, js, u8, meta)
        row["status"] = "copy"
    except CopyLimit as e:
        _write(npy, js, np.load(onpy), meta0)
        row["status"], row["reason"] = "original", str(e)
    except Exception as e:  # noqa: BLE001  nothing written: the build fails and a rerun redoes this scan
        row["status"], row["reason"] = "error", "%s: %s" % (type(e).__name__, e)
    finally:
        for p in glob.glob(base + "*.nii.gz*"):
            try:
                os.remove(p)
            except OSError as e:                  # a copy left on disk costs space, not the measurement
                print("  could not remove %s: %s" % (p, e), flush=True)
        gc.collect()
    row["seconds"] = round(time.time() - t0, 1)
    return row


def scans_wanted():
    """[(uid, DICOM folder)]: LUNA16's scans with an original volume, then the other scans malignancy v5 scores."""
    luna = lp.luna16_subsets()
    want = {u for u in luna if os.path.exists(lp.paths_for(ORIG, u)[0])}
    oof = os.path.join(ORIG, "malignancy-v%d" % MALIG_VERSION, "oof.csv")
    with open(oof, newline="") as f:
        want |= {r["seriesuid"] for r in csv.DictReader(f) if not r["seriesuid"].startswith("mo-")}
    no_orig = sorted(u for u in want if not all(os.path.exists(p) for p in lp.paths_for(ORIG, u)))
    if no_orig:                                   # one() reads the original's meta and, for a copy it cannot hold, its volume
        raise SystemExit("%d wanted scans have no original volume: %s" % (len(no_orig), ", ".join(m[-12:] for m in no_orig[:5])))
    folder = {u: d for u, _, d in lp.complete_scans()}
    missing = sorted(want - set(folder))
    if missing:
        raise SystemExit("%d wanted scans have no DICOM folder: %s" % (len(missing), ", ".join(m[-12:] for m in missing[:5])))
    return [(u, folder[u]) for u in sorted(want)]


def build(a):
    out = os.path.abspath(a.out)
    tmp = os.path.join(out, "nii")
    os.makedirs(tmp, exist_ok=True)
    man = os.path.join(out, "roundtrip.csv")
    done = {}
    if os.path.exists(man):
        with open(man, newline="") as f:
            done = {r["seriesuid"]: r for r in csv.DictReader(f) if r["status"] in ("copy", "original")}
    scans = scans_wanted()
    if a.limit:
        scans = scans[:a.limit]
    todo = []
    for uid, folder in scans:
        npy, js = lp.paths_for(out, uid)
        if uid in done and os.path.exists(npy) and os.path.exists(js):
            continue
        for p in (npy, js):                       # a volume with no manifest row is redone
            if os.path.exists(p):
                os.remove(p)
        todo.append((uid, folder, out, tmp))
    print("round trip: %d scans wanted, %d done before, %d to do, %d at a time" % (len(scans), len(scans) - len(todo),
                                                                                 len(todo), a.jobs), flush=True)
    redo = {job[0] for job in todo}               # a redone scan's old row goes: its files were just removed
    rows = [r for u, r in done.items() if u not in redo]
    t0, errors = time.time(), 0
    with ProcessPoolExecutor(a.jobs) as pool:
        for i, r in enumerate(pool.map(one, todo, chunksize=1), 1):
            if r["status"] == "error":
                errors += 1
                print("ERROR %s: %s" % (r["seriesuid"], r["reason"]), flush=True)
            else:
                if r["status"] == "original":
                    print("ORIGINAL KEPT %s: %s" % (r["seriesuid"], r["reason"]), flush=True)
                rows.append(r)
                with open(man + ".part", "w", newline="") as f:
                    w = csv.DictWriter(f, COLS)
                    w.writeheader()
                    w.writerows(rows)
                os.replace(man + ".part", man)
            if i % 25 == 0:
                print("%d/%d scans (%.0f s)" % (i, len(todo), time.time() - t0), flush=True)
    lp.write_index(out)
    copies = [r for r in rows if r["status"] == "copy"]
    kept = len(rows) - len(copies)
    db = sum(float(r["dicom_bytes"]) for r in copies)
    cb = sum(float(r["copy_bytes"]) for r in copies)
    print("round trip: %d copies, %d kept their originals, %d errors; %.0f s" % (len(copies), kept, errors, time.time() - t0))
    if copies:
        print("bytes over the %d copies: DICOM %.2f GB, the 1.5 mm copy %.2f GB (%.1f times smaller); JPEG-LS lossless at "
              "the data session's 3.2 times would be %.2f GB" % (len(copies), db / 1e9, cb / 1e9, db / max(cb, 1), db / 3.2e9))
    if errors:
        raise SystemExit("%d scans failed for a reason the copy does not explain: rerun the same command" % errors)
    return 0


def malig(a):
    """Malignancy v5's out-of-fold lesions scored by the fold nets that never trained on them, from the volumes
    RAIDIOLOGY_LUNG_DIR names (the originals when unset)."""
    import pandas as pd
    import torch
    import nodule_fpr as nf
    import nodule_malignancy as nm
    from core import card
    card.require_card(nf.DEV, torch)
    nm.limit_malignancy_card(nm.SCORING_GB)
    root = nm.out_root(a.version)
    oof = pd.read_csv(os.path.join(root, "oof.csv"), dtype={"seriesuid": str})
    oof = oof[oof.fold >= 0].reset_index(drop=True)
    boxes = {u: nm.lidc_box(u) for u in oof.seriesuid.unique()}
    parts = []
    for k in sorted(int(f) for f in oof.fold.unique()):
        ck = torch.load(os.path.join(root, "fold%d.pt" % k), map_location=nf.DEV, weights_only=False)
        net = nf.build(ck["cfg"])
        net.load_state_dict(ck["state"])
        del ck
        held = oof[oof.fold == k].reset_index(drop=True)
        rows = held[["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "label", "fold"]].copy()
        rows["logit"] = nm.logits_of(net, boxes, held, batch=nm.SCORING_BATCH)
        rows["saved_logit"] = held.logit
        rows["patient"] = [lp.load_meta(u)["patient"] for u in rows.seriesuid]
        parts.append(rows)
        print("fold %d: AUROC %.4f on %d lesions" % (k, nm.auroc(rows.label, rows.logit), len(rows)), flush=True)
        del net
        if nf.DEV.type == "cuda":
            torch.cuda.empty_cache()
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(a.out, index=False)
    print("volumes from %s: AUROC %.4f on %d lesions of %d patients; the saved scores differ by at most %.4f (median %.4f)"
          % (lp.out_dir(), nm.auroc(res.label, res.logit), len(res), res.patient.nunique(),
             float((res.logit - res.saved_logit).abs().max()), float((res.logit - res.saved_logit).abs().median())))
    return 0


def malig_compare(a):
    """B against A on AUROC: paired patient bootstrap and froc_compare's verdict rule."""
    import pandas as pd
    import nodule_malignancy as nm
    key = ["seriesuid", "coordX", "coordY", "coordZ"]
    ta, tb = pd.read_csv(a.a, dtype={"seriesuid": str}), pd.read_csv(a.b, dtype={"seriesuid": str})
    for t in (ta, tb):
        for c in key[1:]:
            t[c] = t[c].round(3)
    m = ta.merge(tb[key + ["logit", "label"]], on=key, suffixes=("_a", "_b"), validate="one_to_one")
    if len(m) != len(ta) or len(m) != len(tb):
        raise SystemExit("A has %d lesions, B %d, %d in both" % (len(ta), len(tb), len(m)))
    if (m.label_a != m.label_b).any():
        raise SystemExit("%d lesions carry different labels in A and B" % int((m.label_a != m.label_b).sum()))
    y = m.label_a.to_numpy()
    sa, sb = m.logit_a.to_numpy(float), m.logit_b.to_numpy(float)
    groups = pd.Series(np.arange(len(m))).groupby(m.patient.to_numpy()).apply(lambda x: x.to_numpy()).tolist()
    rng = np.random.default_rng(a.seed)
    diffs = []
    for _ in range(a.boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        diffs.append(nm.auroc(y[idx], sb[idx]) - nm.auroc(y[idx], sa[idx]))
    lo, hi = np.nanpercentile(np.array(diffs), [2.5, 97.5])
    va, vb = nm.auroc(y, sa), nm.auroc(y, sb)
    verdict = ("BETTER" if lo > 0 else "NOT WORSE" if lo > -a.margin else "WORSE" if hi < 0 else "INCONCLUSIVE")
    print("%d lesions of %d patients; %d paired resamples of patients" % (len(m), len(groups), a.boot))
    print("AUROC  A %.4f  B %.4f  B - A %+.4f (%+.4f to %+.4f)" % (va, vb, vb - va, lo, hi))
    print("Verdict (AUROC, margin %.3f): %s" % (a.margin, verdict))
    with ledger.run("experiment", a.name or "malignancy %s vs %s" % (os.path.basename(a.a), os.path.basename(a.b))) as led:
        led.settings_from({"a": a.a, "b": a.b, "boot": a.boot, "margin": a.margin, "seed": a.seed})
        led.number("auroc_a", va).number("auroc_b", vb).number("auroc_diff", vb - va)
        led.number("auroc_diff_low", float(lo)).number("auroc_diff_high", float(hi))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--limit", type=int, default=0)
    m = sub.add_parser("malig")
    m.add_argument("--version", type=int, default=MALIG_VERSION)
    m.add_argument("--out", required=True)
    c = sub.add_parser("malig-compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--margin", type=float, default=0.01)
    c.add_argument("--boot", type=int, default=1000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--name")
    a = ap.parse_args()
    return {"build": build, "malig": malig, "malig-compare": malig_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
