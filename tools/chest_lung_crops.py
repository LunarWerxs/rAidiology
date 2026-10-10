"""Zoom to the lungs: one cached 384 px picture per chest X-ray, cropped to a box round the lungs.

Trial 6 (docs/efficiency/README.md, "zoom to the lungs, chest v110"): a full picture carries shoulders,
neck and labels; does the reader gain from seeing only the chest at higher effective resolution? This
builds the pictures on the processor so the card stays free.

Writes, under <work>/eval/chest/ (or --out):
  nih_lungs0_384.npy   uint8 (N, 384, 384), in EXACTLY the row order of prepare.py's nih_384 cache
  nih_lungs0_384.json  {size, name, folds, built, images, unreadable, not_built, boxes, counts, ...};
                       boxes is {image: [x0, y0, side, source]} in the 1024 px frame

The box, from the lungs' 94 landmarks (right lung 44, then left lung 50, of CheXmask's 120): their
outermost x and y, widened by 10% of the box's own width each side and 10% of its own height above and
below, squared to the larger side around its centre, then moved inside the picture. A square larger than
the picture is the whole picture.

A box is refused, and the whole picture used, when: any landmark is NaN; the picture is not 1024 x 1024
(the landmarks are in that frame); the squared side is under 40% of the width (a lung outline that small
is wrong); the box centre is outside the middle 60% (0.2 to 0.8, each way); or CheXmask's Dice is under
0.7 (its README's advice).

Whose landmarks place a box (the slip reader's split: reference outlines teach, our own are judged):
- fold 0's held-out pictures: OUR outline model's fold-0 landmarks (outline-v1/landmarks_fold0.npz).
  CheXmask's never place a judged crop. The script checks this set equals split.py's fold-0 rows.
- every other development picture: CheXmask's landmarks, which the outline model learned from.
- NIH's test pictures are not decoded. Their rows stay zero and are listed under not_built, so the
  test list stays closed; only the split's image lists are used.

Pixels: the original PNG, grey, the X-ray module's window over the WHOLE picture, then the crop, then
that module's letterbox to 384. A whole-picture fallback runs the same steps with the crop being the
whole picture, so its row equals nih_384's (read_any) byte for byte. --limit checks that.

An unreadable file stays zeros and is listed; the build goes on. Resumable like prepare.build: .part
renamed when complete, the json last.

Usage: python tools/chest_lung_crops.py [--jobs 8] [--rebuild]
       python tools/chest_lung_crops.py --limit 200 --out <dir>   (smoke run on the first 200 development rows)
       python tools/chest_lung_crops.py --selftest"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "modalities", "chest_xray", "eval")
sys.path.insert(0, EVAL)
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402
import datasets  # noqa: E402
import prepare  # noqa: E402
import split as splitmod  # noqa: E402

NAME = "lungs0"
SIZE = prepare.SIZE
FRAME = 1024                # NIH's pictures, and both sets of landmarks, are 1024 pixels square
N_LUNG = 94                 # right lung 44 + left lung 50, the first points of CheXmask's 120
N_POINTS = 120
PAD = 0.10                  # of the box's own width (each side) and height (above and below)
MIN_SIDE = 0.40             # of the width: a smaller square is refused
MIDDLE = (0.2, 0.8)         # the box centre must lie inside this share of the width and of the height
MIN_DICE = 0.7              # CheXmask's README advice, as chexmask_ctr.py
OUTLINE = os.path.join(paths.WORK, "eval", "chest", "outline-v1", "landmarks_fold0.npz")


def box_of(points):
    """((x0, y0, side), None) for 120 (x, y) points in the 1024 frame, or (None, reason) when the whole
    picture is used instead. Reasons: nan, small, off_centre, oversize."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != N_POINTS or not np.isfinite(pts).all():
        return None, "nan"
    lung = pts[:N_LUNG]
    (x0, y0), (x1, y1) = lung.min(axis=0), lung.max(axis=0)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(w * (1 + 2 * PAD), h * (1 + 2 * PAD))         # widened on both sides, then squared
    if side < MIN_SIDE * FRAME:
        return None, "small"
    lo, hi = MIDDLE[0] * FRAME, MIDDLE[1] * FRAME
    if not (lo <= cx <= hi and lo <= cy <= hi):
        return None, "off_centre"
    s = int(round(side))
    if s >= FRAME:
        return None, "oversize"
    left = min(max(int(round(cx - s / 2)), 0), FRAME - s)     # moved inside the picture
    top = min(max(int(round(cy - s / 2)), 0), FRAME - s)
    return (left, top, s), None


def selftest():
    def pts(x0, y0, x1, y1):
        p = np.zeros((N_POINTS, 2))
        p[:N_LUNG, 0] = np.linspace(x0, x1, N_LUNG)
        p[:N_LUNG, 1] = np.linspace(y0, y1, N_LUNG)
        p[N_LUNG:] = [(x0 + x1) / 2, (y0 + y1) / 2]            # the heart is not used for the box
        return p
    cases = [
        ("widen", pts(300, 300, 700, 700), ((260, 260, 480), None)),             # 400 -> 480, centred on 500
        ("square", pts(200, 400, 800, 600), ((140, 140, 720), None)),            # wide and short: side 1.2 x 600
        ("tall", pts(300, 200, 700, 800), ((140, 140, 720), None)),              # tall and thin: side 1.2 x 600
        ("move inside", pts(20, 300, 420, 700), ((0, 260, 480), None)),          # x0 would be -20
        ("oversize", pts(210, 80, 810, 940), (None, "oversize")),                # 1.2 x 860 = 1032 > 1024
        ("small", pts(300, 300, 500, 500), (None, "small")),                      # 1.2 x 200 = 240 < 409.6
        ("off centre", pts(650, 300, 1010, 700), (None, "off_centre")),          # centre x 830 > 819.2
    ]
    for name, p, want in cases:
        got = box_of(p)
        assert got == want, "%s: got %s, wanted %s" % (name, got, want)
    p = pts(300, 300, 700, 700)
    p[100, 1] = np.nan                                                            # a NaN in the heart still refuses
    assert box_of(p) == (None, "nan")
    assert box_of(pts(300, 300, 700, 700)[:50]) == (None, "nan")                  # too few points
    print("selftest ok")


def landmark_sources():
    """(fold-0 held-out images, {image: points} from our outline model, {image: (points, dice)} from CheXmask)."""
    import chexmask_ctr as cc
    assert cc.N_RIGHT + cc.N_LEFT == N_LUNG and cc.N_POINTS == N_POINTS and cc.FRAME == FRAME
    z = np.load(OUTLINE, allow_pickle=False)
    # 2026-10-04: z["landmarks"] read once. Indexing the NpzFile in the loop re-read the 16 MB array per
    # picture, and each row kept its copy alive: 17,275 copies, out of memory at once.
    lm_ours = z["landmarks"]
    ours = {str(i): lm_ours[k] for k, i in enumerate(z["image"])}
    names, lm, dice = cc.load_landmarks()
    theirs = {str(i): (lm[k], dice[k]) for k, i in enumerate(names)}
    return ours, theirs


def plan(dev, shapes, ours, theirs):
    """{image: [x0, y0, side, source]} for every development picture."""
    fold0 = set(dev.image[dev.fold == 0].astype(str))
    assert fold0 == set(ours), "outline-v1 fold 0 has %d pictures, the split's fold 0 has %d, %d differ" % (
        len(ours), len(fold0), len(fold0 ^ set(ours)))
    boxes = {}
    for img in dev.image.astype(str):
        whole, source, pts = None, None, None
        if shapes.get(img) != [FRAME, FRAME]:
            whole = "not_1024"
        elif img in fold0:
            source, pts = "outline", np.asarray(ours[img], dtype=np.float64) * 1.0     # already in the 1024 frame
        elif img not in theirs:
            whole = "no_landmarks"
        elif not theirs[img][1] >= MIN_DICE:
            whole = "low_dice"
        else:
            source, pts = "chexmask", np.asarray(theirs[img][0], dtype=np.float64) * FRAME   # stored / 1024
        if whole is None:
            box, whole = box_of(pts)
            if box is not None:
                boxes[img] = [box[0], box[1], box[2], source]
                continue
        boxes[img] = [0, 0, FRAME, "whole:" + whole]
    return boxes


_WORKER = {}


def _open_worker(part, size, keys, boxes):
    _WORKER.update(arr=np.lib.format.open_memmap(part, mode="r+"), size=size, keys=keys, boxes=boxes)


def _crop_into(job):
    slot, row = job
    key = _WORKER["keys"][row]
    box = _WORKER["boxes"][key]
    p = datasets.image_path(key)
    if not os.path.exists(p):
        return slot, key, "file missing"
    try:
        from PIL import Image
        xp = prepare._xray_prepare()
        with Image.open(p) as im:
            im.load()
            arr = np.asarray(im.convert("L"))
        pic = xp.scaled(arr)                                  # the window sees the whole picture
        if not box[3].startswith("whole"):
            if pic.shape != (FRAME, FRAME):
                raise ValueError("picture is %s, not %d square" % (pic.shape, FRAME))
            x0, y0, s = box[:3]
            pic = np.ascontiguousarray(pic[y0:y0 + s, x0:x0 + s])
        _WORKER["arr"][slot] = xp.letterbox(pic, _WORKER["size"])
    except Exception as e:  # noqa: BLE001  any decode failure makes the picture unreadable, not the build
        return slot, key, "%s: %s" % (type(e).__name__, e)
    return slot, key, None


def summarize(dev, boxes):
    fold0 = set(dev.image[dev.fold == 0].astype(str))
    counts = {"fold0": Counter(), "others": Counter()}
    sides = {"fold0": [], "others": [], "fold0_cropped": [], "others_cropped": []}
    for img, b in boxes.items():
        side = "fold0" if img in fold0 else "others"
        counts[side][b[3]] += 1
        sides[side].append(b[2])
        if not b[3].startswith("whole"):
            sides[side + "_cropped"].append(b[2])
    med = {k: (float(np.median(v)) if v else None) for k, v in sides.items()}
    return {k: dict(sorted(v.items())) for k, v in counts.items()}, med


def build(out=None, jobs=8, rebuild=False, limit=None):
    t0 = time.time()
    keys = [str(k) for k in datasets.index().image]
    nih_npy, nih_js = prepare.cache_paths(SIZE)
    nih = json.load(open(nih_js, encoding="utf-8"))
    if nih.get("images") != keys:
        raise SystemExit("nih_%d.json does not match the index; rebuild it with: %s --rebuild"
                         % (SIZE, prepare.build_command(SIZE)))
    shapes = nih["source_shape"]
    dev = splitmod.frame("dev")
    dev_rows = {k: i for i, k in enumerate(keys)}
    ours, theirs = landmark_sources()
    boxes = plan(dev, shapes, ours, theirs)
    dev_images = [str(i) for i in dev.image]
    dev_images.sort(key=dev_rows.__getitem__)
    if limit:
        dev_images = dev_images[:limit]
    rows = [dev_rows[i] for i in dev_images]
    built_set = set(dev_images)
    base = os.path.join(out or prepare.CACHE, "nih_%s_%d" % (NAME, SIZE))
    npy, js = base + ".npy", base + ".json"
    n_out = len(rows) if limit else len(keys)
    if not rebuild and not limit and os.path.exists(npy) and os.path.exists(js):
        meta = json.load(open(js, encoding="utf-8"))
        if meta.get("images") == keys and meta.get("name") == NAME:
            print("crops already built (%s); --rebuild to redo" % npy, flush=True)
            return meta
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    part = npy + ".part"
    arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(n_out, SIZE, SIZE))
    arr[:] = 0
    arr.flush()
    del arr
    slots = list(range(n_out)) if limit else rows
    print("cropping %d development pictures (of %d) to %dx%d in %d processes" % (len(rows), len(keys), SIZE, SIZE, jobs),
          flush=True)
    bad, errors, done = set(), {}, 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=_open_worker,
                             initargs=(part, SIZE, keys, boxes)) as pool:
        for slot, key, err in pool.map(_crop_into, list(zip(slots, rows)), chunksize=16):
            if err is not None:
                bad.add(key)
                errors[key] = err
            done += 1
            if done % prepare.PROGRESS == 0 or done == len(rows):
                el = time.time() - t0
                print("  %d/%d  %.0f s  %.1f images/s  %d unreadable" % (done, len(rows), el, done / max(el, 1e-9), len(bad)),
                      flush=True)
    arr = np.lib.format.open_memmap(part, mode="r+")
    arr.flush()
    del arr
    os.replace(part, npy)
    shown = {k: v for k, v in boxes.items() if k in built_set}
    counts, med = summarize(dev[dev.image.astype(str).isin(built_set)], shown)
    meta = {"size": SIZE, "name": NAME, "folds": [0], "built": time.strftime("%Y-%m-%d %H:%M:%S"), "images": keys,
            "unreadable": [k for k in keys if k in bad],
            "not_built": [k for k in keys if k not in built_set],
            "boxes": shown, "counts": counts, "median_side": {"fold0": med["fold0_cropped"], "others": med["others_cropped"]},
            "median_side_all": {"fold0": med["fold0"], "others": med["others"]}}
    if limit:
        meta["rows"] = dev_images                              # the smoke cache holds only these, in this order
        whole = [i for i in dev_images if boxes[i][3].startswith("whole") and i not in bad]
        small = np.load(npy, mmap_mode="r")
        full = np.load(nih_npy, mmap_mode="r")
        slot_of = {k: s for s, k in enumerate(dev_images)}
        wrong = [i for i in whole if not np.array_equal(small[slot_of[i]], full[dev_rows[i]])]
        assert not wrong, "whole-picture rows differ from nih_384: %s" % wrong[:5]
        print("whole-picture rows equal to nih_384: %d of %d checked" % (len(whole) - len(wrong), len(whole)), flush=True)
        del small, full
    tmp = js + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, js)
    for k in meta["unreadable"][:10]:
        print("  unreadable %s: %s" % (k, errors[k]), flush=True)
    print("%s  built=%d  unreadable=%d  wall %.1f s" % (npy, len(rows), len(bad), time.time() - t0), flush=True)
    print("  counts %s" % json.dumps(counts), flush=True)
    print("  median side (px of 1024, cropped only) %s" % json.dumps(meta["median_side"]), flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Crop each chest X-ray to a box round the lungs (trial 6).")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true", help="build again even when the cache is current")
    ap.add_argument("--limit", type=int, help="build only the first N development rows (a smoke run; needs --out)")
    ap.add_argument("--out", help="directory to write into instead of <work>/eval/chest")
    ap.add_argument("--selftest", action="store_true", help="check the box rules on synthetic landmarks and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return
    if a.limit and not a.out:
        raise SystemExit("--limit needs --out, so a smoke run never overwrites the real cache")
    build(out=a.out, jobs=max(1, a.jobs), rebuild=a.rebuild, limit=a.limit)


if __name__ == "__main__":
    main()
