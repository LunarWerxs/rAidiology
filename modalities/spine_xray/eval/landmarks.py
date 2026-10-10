"""More side-view spines to learn from: the vertebral landmark set, read into the corner reader's point order.

Dataset: Mendeley spondylolisthesis vertebral landmark dataset (Keypoint R-CNN labels of lateral lumbar films,
Honduran patients plus BUU-LSPINE patients), CC BY 4.0 (LICENSING.md, spondylolisthesis-landmarks row). Plan and
pass bars: docs/efficiency/README.md, "Corners v9, idea 24: more side views to learn from".

What is in it (read from the files, 2026-10-05). Pictures under Dataset/Train/Keypointrcnn_data/images/{train,val}
and a label file of the same name under labels/. A label is one of two kinds (the same pictures are never in both):
a JSON with "boxes" and "keypoints" (4 points a vertebra), or a YOLO-pose .txt (class, box, then 4 x (x, y,
visibility) as shares of the picture). 484 pictures are labelled in JSON and 214 in .txt; Dataset/Test holds 16
full-size BUU films with no label at all, which are not used. Every labelled picture is a Roboflow export squeezed
to 640 x 640 WITHOUT keeping its aspect, so a film is stretched: BUU's side films are 1.22 times as tall as wide,
here they are square. `load` undoes that with BUU's own median film aspect (ASPECT, printed by `gate`), which is
right for BUU's commonest film shape and off for the others, which the gate's numbers show. Points live in that
un-stretched 640-wide frame, x as drawn, y times ASPECT.

Distinct pictures. Roboflow adds augmented copies (the file name `<base>_jpg.rf.<hash>.jpg`; every name carries a
hash, so the name cannot say which copy is the original). One copy a base picture: a copy in the `val` folder
when there is one (Roboflow augments only `train`), else of several `train` copies the one with the fewest pure
black border pixels (a rotated or shifted copy has black corners). The number of bases whose copies were all
`train` ones, so possibly all augmented, is logged. Roboflow may also have mirrored a copy: the orientation test
below turns every picture so the spine faces left as BUU's do (a picture mirrored by `flip`), and its points with it.

Which vertebra is which. Boxes are ordered top to bottom by the mean height of their four points. The lowest box
is S1, whose upper edge is the one point pair we want from it, unless it looks like a vertebra: S1's box is not
a vertebra's (its 4 points are the sacral plateau and a short stub below it), so the lowest box is called a
vertebra (L5; no S1 in that picture) only when ALL of these sit inside the 1st to 99th percentiles of BUU's drawn
vertebrae: lower edge / upper edge length, height / width, the angle between the two edges, and the upper
edge's steepness (against L5's, the steepest). S1's upper edge is then taken by its place in the label file
(JSON points 2 and 3, YOLO points 0 and 3), because the S1 box is a tilted slab where no position rule holds.
Then L5, L4, ... L1 are the boxes above, and the boxes above L1 are dropped.

A vertebra's four points become (upper-left, upper-right, lower-left, lower-right) by position, never by their
index (the index order differs between files): the two with the smaller x are the left side, and in each side the
smaller y is the upper one; if that is not a plausible quadrilateral (both ends of the upper edge above the lower
edge's, a convex outline) the split by height is tried instead, else the vertebra is left out (NaN). S1's upper
edge is its two points with the smaller y (a steep sacral plateau defeats the left-right split).

Orientation. BUU's lines slope in a fixed pattern when the film faces left (L1 to L3 edges fall to the right, L5
and S1 fall to the left, S1 by 36 degrees on average). Each picture's lines are scored against BUU's mean slope
line by line; a negative score means mirrored, and then the picture and its points are mirrored.

The gate (idea 24): the share of mapped L1..L5 vertebrae whose height/width and endplate angle both sit inside BUU's
pooled 1st-99th percentiles must be 95% or more. The film's aspect is fitted per picture so its median height/width
matches BUU's, which makes the height/width half partly circular; `gate` also prints the share with BUU's fixed
median aspect, and per-level numbers for information. `load` has a drift guard (EXPECT: 590 bases, 501 BUU, 89 other).

`scale` brings each picture's median vertebra width (mean of its two edge lengths, L1 to L5) to BUU's median,
so the nets see vertebrae of the usual size: `render` resizes a picture to a BUU film's size (times a factor), and
`canvas_set` puts it on the corner reader's canvas with the factor BUU's own films get on average.

Leak control: any BUU patient id found in buu's split.json is dropped (none expected; the count is printed).

Folds (idea 119, 2026-10-08): the distinct pictures in five folds, grouped by patient (patient_of: the BUU id, or the
base name for a Honduran picture), seed 0. `python landmarks.py folds` writes landmark_folds.csv beside this file ONCE
(columns base, patient, fold); fold_table() reads it and refuses if its pictures are not load('train')'s. corners.py
and corners_refine.py `landmark_fold=K` keep fold K's pictures out of training; landmarks_test.py --fold K scores them.

Parts. `load(part='train')` reads only Dataset/Train (the labelled pictures; this is all training ever sees, and
corners.py / corners_refine.py `extra_landmarks` ask for part='train' explicitly). `load(part='test')` reads only
Dataset/Test: 16 full-size BUU films with NO label file of any kind (checked 2026-10-05), so its records have all
points NaN and `labelled` False; they can never join training. Note 7 of those 16 patients (3753, 3808, 3870, 4698,
4921, 4924, 4951) also appear in Train as labelled 640 x 640 pictures, so the set's own Train holds their labels.

Scoring against the set's published benchmark (Bansal et al., PLoS One 2026, e0347290: share of keypoints within 5 px
of the label, at 640 x 640): `python landmarks.py score <file> [--part train|test]`. Which keypoints: ours only,
levels L3, L4, L5 and S1's upper edge (points 8..21 of our 22). The truth is each picture's label mapped back into
the picture file's own 640 x 640 frame (un-mirrored, un-stretched back, each line's two ends in the order a reader of
that file would give them: point 0 the end on the left). The share's denominator is every labelled keypoint of those
14; a point with no prediction counts as a miss. Prints the share within 5, 10 and 15 px and the mean distance (px,
over predicted points). With --part test there is nothing to score against (no labels): it says so and exits 1.
--part train scores the labelled pictures, honest only for nets that did not train on them (extra_landmarks off; the
BUU ids 2662+ are not in split.json).
  Prediction file: .npz with `names` (n strings: a picture's file stem, or its base name, as in load()'s records) and
  `points` (n, 22, 2) float, x then y, in the pixels of that picture FILE as stored (a 640 x 640 picture: its own
  pixels; a full film: original film pixels, as the corner nets write theirs), NaN where there is no prediction; or a
  .csv with header `name,point,x,y` (point 0..21), one row a predicted point.

Usage: python landmarks.py gate        the geometry report against BUU's drawn vertebrae
       python landmarks.py selftest    CPU, reads only label files and BUU's labels; the idea-24 gate; exit 1 on a miss
       python landmarks.py score FILE [--part train|test]    share of keypoints within 5 / 10 / 15 px (see above)
       python landmarks.py folds       writes landmark_folds.csv once (idea 119; refuses to overwrite it)
"""
import argparse
import collections
import csv
import functools
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import paths  # noqa: E402
import buu  # noqa: E402

N_POINTS = 22
FRAME = 640.0                      # the exported pictures' side
BUU_NAME = re.compile(r"^(\d{4})-([FM])-(\d{3})Y1$")
GATE_SHARE = 0.95
GATE_PCT = (1.0, 99.0)
S1_UPPER = {"json": (2, 3), "txt": (0, 3)}   # which two of a box's 4 points are its upper edge (see read_label's kinds)
ASPECT_RANGE = (1.0, 2.4)          # a film's height / width is held between BUU's 1st percentile (1.08) and above its 99th (2.27)
CITATION = ("Mendeley spondylolisthesis vertebral landmark dataset, CC BY 4.0 (LICENSING.md, "
            "spondylolisthesis-landmarks row)")
EXPECT = {"bases": 590, "buu": 501, "other": 89}   # measured 2026-10-05 (591 labelled bases, one with no vertebra)


def root():
    return os.path.join(paths.DATASETS, "public", "spondylolisthesis-landmarks", "Dataset")


# --- BUU's own numbers -------------------------------------------------------------------------------------

def _lines(pts):
    """(..., 4, 2) corners [UL, UR, LL, LR] -> (upper-left, upper-right, lower-left, lower-right) edge pieces."""
    return pts[..., 0, :], pts[..., 1, :], pts[..., 2, :], pts[..., 3, :]


def geometry(q):
    """Shape numbers of quadrilaterals q (..., 4, 2) = [UL, UR, LL, LR]: width (mean of the two edge lengths),
    hw (height between the edges' midpoints over width), ll_lu (lower over upper edge), wedge (upper edge's angle
    minus the lower's, degrees, angles read left to right), ang_up (the upper edge's angle, degrees)."""
    ul, ur, ll, lr = _lines(q)
    lu = np.linalg.norm(ur - ul, axis=-1)
    lo = np.linalg.norm(lr - ll, axis=-1)
    width = (lu + lo) / 2
    h = np.linalg.norm((ll + lr) / 2 - (ul + ur) / 2, axis=-1)
    au = np.degrees(np.arctan2(ur[..., 1] - ul[..., 1], ur[..., 0] - ul[..., 0]))
    al = np.degrees(np.arctan2(lr[..., 1] - ll[..., 1], lr[..., 0] - ll[..., 0]))
    return {"width": width, "hw": h / np.maximum(width, 1e-9), "ll_lu": lo / np.maximum(lu, 1e-9),
            "wedge": au - al, "ang_up": au}


@functools.lru_cache(maxsize=1)
def buu_reference():
    """BUU's drawn side-view vertebrae L1..L5 (all 2,000 patients, original pixels) as the numbers this module
    compares against: per-level geometry arrays, the pooled 1st / 99th percentiles, the median width, the mean
    angle of each of the 11 lines (the orientation filter), and the film aspect and canvas factor of the films."""
    _ids, p, _c = buu.annotations("LA")
    p = p.astype(np.float64).reshape(len(p), 11, 2, 2)
    q = np.stack([np.stack([p[:, 2 * i, 0], p[:, 2 * i, 1], p[:, 2 * i + 1, 0], p[:, 2 * i + 1, 1]], axis=1)
                  for i in range(5)], axis=1)                                  # (N, 5, 4, 2)
    g = geometry(q)
    lo, hi = GATE_PCT
    pooled = {k: tuple(np.percentile(g[k], [lo, hi])) for k in ("hw", "ll_lu", "wedge")}
    pooled["abs_wedge"] = (0.0, float(np.percentile(np.abs(g["wedge"]), hi)))
    pooled["abs_ang_up_L5"] = (0.0, float(np.percentile(np.abs(g["ang_up"][:, 4]), hi)))
    per_level = {k: np.percentile(g[k], [lo, hi], axis=0).T for k in ("hw", "wedge")}       # (5, 2)
    ang = np.degrees(np.arctan2(p[:, :, 1, 1] - p[:, :, 0, 1], p[:, :, 1, 0] - p[:, :, 0, 0]))   # (N, 11)
    sizes = _buu_sizes()
    aspect = float(np.median(sizes[:, 1] / sizes[:, 0]))
    return {"hw_median": float(np.median(g["hw"])), "geom": g, "pooled": pooled, "per_level": per_level, "width": float(np.median(g["width"])),
            "line_angle": ang.mean(0), "aspect": aspect, "sizes": sizes, "n": len(p)}


def _buu_sizes():
    """(2000, 2) original (width, height) of BUU's side films: from buu's cache when there is one, else the headers."""
    cp = buu.cache_path("LA", 512, 384)
    ids = buu.annotations("LA")[0]
    if os.path.exists(cp):
        z = np.load(cp)
        if np.array_equal(z["ids"], ids):
            return z["size"].astype(np.float64)
    from PIL import Image
    jpgs = buu.files("LA", ext="jpg")
    out = []
    for i in ids:
        with Image.open(jpgs[int(i)]) as im:
            out.append(im.size)
    return np.array(out, np.float64)


def _median_hw(points):
    """Median height / width of the mapped L1..L5 vertebrae in a (22, 2) point array, or None when there is none."""
    q = points[:20].astype(np.float64).reshape(5, 4, 2)
    have = ~np.isnan(q).any(axis=(1, 2))
    return float(np.median(geometry(q[have])["hw"])) if have.any() else None


# --- reading one picture's label ----------------------------------------------------------------------------

def read_label(path):
    """The picture's vertebrae as a list of (4, 2) arrays in the exported picture's pixels (the 4 points in file order)."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return [np.asarray(k, np.float64)[:, :2] for k in d["keypoints"]]
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if len(t) != 17:
                continue
            v = np.asarray(t[5:], np.float64).reshape(4, 3)
            out.append(v[:, :2] * FRAME)                    # shares of the (640 x 640) picture
    return out


def split_ring(pts):
    """4 points -> [UL, UR, LL, LR] by walking round the outline: the points in order of angle round their centre make
    four edges; the longer pair of opposite edges are the endplates, the one with the smaller mean y is the upper, and
    each endplate's ends are told apart by x. Needs no index order and survives a tilted vertebra (a left-right or
    top-bottom split by position fails on a 30 degree tilt), which the sacral box always is."""
    c = pts.mean(axis=0)
    ring = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]), kind="stable")]
    edge = [(ring[k], ring[(k + 1) % 4]) for k in range(4)]
    length = [float(np.linalg.norm(b - a)) for a, b in edge]
    k = 0 if length[0] + length[2] >= length[1] + length[3] else 1
    pair = [edge[k], edge[k + 2]]
    pair.sort(key=lambda e: float(e[0][1] + e[1][1]))
    up = sorted(pair[0], key=lambda p: p[0])
    lo = sorted(pair[1], key=lambda p: p[0])
    return np.stack([up[0], up[1], lo[0], lo[1]])


def plausible(q):
    """A usable vertebra: each edge runs left to right, the upper edge's ends are above the lower edge's, the outline
    UL -> UR -> LR -> LL is convex with positive area, and its shape numbers are within loose bounds."""
    ul, ur, ll, lr = q
    if not (ul[0] < ur[0] and ll[0] < lr[0] and ul[1] < ll[1] and ur[1] < lr[1]):
        return False
    ring = np.stack([ul, ur, lr, ll])
    e = np.roll(ring, -1, axis=0) - ring
    cross = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
    if not (cross > 0).all():                                  # image y points down: clockwise on screen is positive
        return False
    g = geometry(q)
    return bool(g["width"] > 0 and 0.15 < g["hw"] < 2.0 and 0.5 < g["ll_lu"] < 2.0)


def to_quad(pts):
    """[UL, UR, LL, LR] of 4 points when plausible, else None."""
    q = split_ring(pts)
    return q if plausible(q) else None


def looks_like_vertebra(pts):
    """True when the lowest box is a vertebra rather than S1 (see the docstring): every number inside BUU's pooled
    1st to 99th percentiles."""
    q = to_quad(pts)
    if q is None:
        return False
    ref = buu_reference()["pooled"]
    g = geometry(q)
    ok = lambda v, lohi: lohi[0] <= v <= lohi[1]  # noqa: E731
    return bool(ok(g["ll_lu"], ref["ll_lu"]) and ok(g["hw"], ref["hw"]) and ok(abs(g["wedge"]), ref["abs_wedge"])
                and ok(abs(g["ang_up"]), ref["abs_ang_up_L5"]))


def map_picture(vertebrae, asp, kind="json"):
    """One picture's vertebrae (list of (4, 2), exported pixels) -> dict: points (22, 2) float64 in the un-stretched frame
    (NaN where the set has none), flip (bool, mirrored to face left), has_s1, boxes, dropped (vertebrae with no plausible
    quadrilateral), orientation (the score)."""
    v = [np.asarray(k, np.float64) * np.array([1.0, asp]) for k in vertebrae]
    v.sort(key=lambda k: float(k[:, 1].mean()))
    out = np.full((N_POINTS, 2), np.nan)
    ref = buu_reference()
    has_s1, dropped = False, 0
    below = list(v)
    s1_pts = None
    if below and not looks_like_vertebra(below[-1]):
        has_s1, s1_pts = True, below.pop()
    levels = {}
    for lev, pts in zip(range(4, -1, -1), reversed(below)):                     # L5 first; boxes above L1 fall away
        q = to_quad(pts)
        if q is None:
            dropped += 1
            continue
        levels[lev] = q
    s1_edge = None
    if s1_pts is not None:
        up = s1_pts[list(S1_UPPER[kind])]                     # the sacral box is a tilted slab: by the file's index order
        up = up[np.argsort(up[:, 0], kind="stable")]
        w = np.linalg.norm(up[1] - up[0])
        ws = [geometry(q)["width"] for q in levels.values()]
        if up[0, 0] < up[1, 0] and (not ws or 0.5 < w / float(np.median(ws)) < 2.2):
            s1_edge = up
        else:
            dropped += 1
    for lev, q in levels.items():
        out[4 * lev:4 * lev + 4] = q
    if s1_edge is not None:
        out[20:22] = s1_edge
    lines = out.reshape(11, 2, 2)
    present = ~np.isnan(lines).any(axis=(1, 2))
    ang = np.degrees(np.arctan2(lines[:, 1, 1] - lines[:, 0, 1], lines[:, 1, 0] - lines[:, 0, 0]))
    score = float(np.nansum(np.where(present, ang * ref["line_angle"], 0.0)))
    flip = score < 0
    if flip:
        m = out.copy()
        m[:, 0] = FRAME - m[:, 0]
        pairs = m.reshape(11, 2, 2)[:, ::-1].reshape(N_POINTS, 2)             # each line's two ends trade places
        out = pairs
    return {"points": out, "flip": flip, "has_s1": has_s1 and s1_edge is not None, "boxes": len(v), "dropped": dropped,
            "orientation": score}


# --- the pictures --------------------------------------------------------------------------------------------

def _pictures():
    """[(split, name, label path)] of every labelled picture, and how many pictures have no label."""
    r = os.path.join(root(), "Train", "Keypointrcnn_data")
    out, bare = [], 0
    for sp in ("train", "val"):
        for p in sorted(glob.glob(os.path.join(r, "images", sp, "*.jpg"))):
            n = os.path.basename(p)[:-4]
            lab = [os.path.join(r, "labels", sp, n + e) for e in (".json", ".txt")]
            lab = [x for x in lab if os.path.exists(x)]
            if lab:
                out.append((sp, n, p, lab[0]))
            else:
                bare += 1
    return out, bare + len(glob.glob(os.path.join(root(), "Test", "*.jpg")))


def _black_border(path):
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"))
    e = np.concatenate([a[:4].ravel(), a[-4:].ravel(), a[:, :4].ravel(), a[:, -4:].ravel()])
    return float((e == 0).mean())


def _split_ids():
    with open(buu.SPLIT_PATH, encoding="utf-8") as f:
        return {int(k) for k in json.load(f)["fold_of"]}


def load(part="train"):
    """part 'train' (default): every labelled picture under Dataset/Train only, see _load_train. part 'test': the
    Dataset/Test films only, unlabelled (points all NaN, labelled False), never for training."""
    if part == "train":
        return _load_train()
    if part == "test":
        return _load_test()
    raise ValueError("part is 'train' or 'test', got %r" % (part,))


@functools.lru_cache(maxsize=1)
def _load_test():
    from PIL import Image
    in_split = _split_ids()
    recs = []
    for p in sorted(glob.glob(os.path.join(root(), "Test", "*.jpg"))):
        n = os.path.basename(p)[:-4]
        m = BUU_NAME.match(n)
        if m and int(m.group(1)) in in_split:
            continue
        with Image.open(p) as im:
            size = im.size
        recs.append({"path": p, "name": n, "base": n, "split": "test", "source": "buu" if m else "honduras",
                     "id": int(m.group(1)) if m else None, "points": np.full((N_POINTS, 2), np.nan, np.float32),
                     "size": size, "labelled": False})
    return recs


@functools.lru_cache(maxsize=1)
def _load_train():
    """One record per distinct base picture, in base-name order. Keys: path, name, base, split, source ('buu' or
    'honduras'), id (BUU's 4-digit id as int, None for honduras), points (22, 2) float32 in the un-stretched 640-wide frame
    (NaN where the set has no point), scale (see the docstring), flip, has_s1, n_levels (vertebrae mapped).
    Prints what it did: bases, copies, how many bases had only train (possibly augmented) copies, ids dropped for
    being in split.json."""
    ref = buu_reference()
    asp = ref["aspect"]
    pics, unlabelled = _pictures()
    groups = collections.defaultdict(list)
    for sp, n, p, lab in pics:
        groups[n.split("_jpg.rf.")[0]].append((sp, n, p, lab))
    in_split = _split_ids()
    recs, stats = [], collections.Counter()
    for base in sorted(groups):
        cp = groups[base]
        vals = [c for c in cp if c[0] == "val"]
        if vals:
            pick = sorted(vals, key=lambda c: c[1])[0]
        elif len(cp) > 1:
            stats["only_train_copies"] += 1
            pick = min(cp, key=lambda c: (_black_border(c[2]), c[1]))
        else:
            pick = cp[0]
        stats["copies_dropped"] += len(cp) - 1
        m = BUU_NAME.match(base)
        if m and int(m.group(1)) in in_split:
            stats["in_split_dropped"] += 1
            continue
        vert, kind = read_label(pick[3]), pick[3][-4:].lstrip(".")
        fixed = map_picture(vert, asp, kind)
        res = fixed
        hw = _median_hw(fixed["points"])
        if hw is not None:
            fit = float(np.clip(asp * ref["hw_median"] / hw, ASPECT_RANGE[0], ASPECT_RANGE[1]))
            res = map_picture(vert, fit, kind)
        else:
            fit = asp
        pts = res["points"]
        have = ~np.isnan(pts[:20]).reshape(5, 4, 2).any(axis=(1, 2))
        if not have.any():
            stats["no_vertebra"] += 1
            continue
        g = geometry(pts[:20].reshape(5, 4, 2)[have])
        recs.append({"path": pick[2], "name": pick[1], "base": base, "split": pick[0],
                     "source": "buu" if m else "honduras", "id": int(m.group(1)) if m else None,
                     "points": pts.astype(np.float32), "aspect": fit, "points_fixed": fixed["points"].astype(np.float32),
                     "scale": float(ref["width"] / np.median(g["width"])),
                     "flip": res["flip"], "has_s1": res["has_s1"],
                     "n_levels": int(have.sum())})
        stats["flipped"] += int(res["flip"])
        stats["dropped_vertebrae"] += res["dropped"]
    print("landmarks: %d labelled pictures in %d bases (%d copies set aside; %d bases had only train copies, so the kept "
          "copy may be augmented; %d unlabelled pictures not used); %d bases kept: %d BUU, %d other; mirrored to face "
          "left: %d; vertebrae with no plausible quadrilateral: %d; BUU ids found in split.json and dropped: %d; "
          "bases with no vertebra: %d"
          % (len(pics), len(groups), stats["copies_dropped"], stats["only_train_copies"], unlabelled, len(recs),
             sum(r["source"] == "buu" for r in recs), sum(r["source"] != "buu" for r in recs), stats["flipped"],
             stats["dropped_vertebrae"], stats["in_split_dropped"], stats["no_vertebra"]), flush=True)
    load.stats = dict(stats)
    for r in recs:
        r["labelled"] = True
    return recs


# --- folds for the four-fifths arrangement (idea 119) -------------------------------------------------------

N_FOLDS = 5
FOLD_SEED = 0
FOLD_CSV = os.path.join(HERE, "landmark_folds.csv")


def patient_of(base):
    """The BUU id (as text) for a BUU-named picture, else the base name: the key the folds are grouped by (also
    landmarks_test's patient key, so a patient's bootstrap resamples and its fold agree)."""
    m = BUU_NAME.match(base)
    return m.group(1) if m else base


def fold_rows(recs):
    """[(base, patient, fold)] for the records, sorted by base. Patients are sorted and dealt in a seeded random order
    round the five folds, so no patient splits and the folds' patient counts differ by at most one."""
    units = sorted({patient_of(r["base"]) for r in recs})
    order = np.random.default_rng(FOLD_SEED).permutation(len(units))
    fold = {units[i]: n % N_FOLDS for n, i in enumerate(order)}
    return [(b, patient_of(b), fold[patient_of(b)]) for b in sorted(r["base"] for r in recs)]


def write_folds():
    """Writes landmark_folds.csv once; a csv already there is kept (the folds are fixed once they have been used)."""
    if os.path.exists(FOLD_CSV):
        raise SystemExit("%s exists; the folds are written once and kept" % FOLD_CSV)
    rows = fold_rows(load("train"))
    with open(FOLD_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["base", "patient", "fold"])
        w.writerows(rows)
    print("wrote %s: %d pictures" % (FOLD_CSV, len(rows)), flush=True)


def split_fold(recs, table, fold):
    """(training records, held-out records) for landmark fold `fold`: the held-out pictures never train (idea 119)."""
    return [r for r in recs if table[r["base"]] != fold], [r for r in recs if table[r["base"]] == fold]


def fold_table():
    """{base: fold} from landmark_folds.csv; refuses when its pictures are not load('train')'s (a drifted set)."""
    if not os.path.exists(FOLD_CSV):
        raise SystemExit("%s is missing: python landmarks.py folds writes it once" % FOLD_CSV)
    with open(FOLD_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    table = {r["base"]: int(r["fold"]) for r in rows}
    if len(table) != len(rows) or set(table) != {r["base"] for r in load("train")}:
        raise SystemExit("%s does not match load('train'): %d rows; the set changed, so the folds must be rebuilt by hand"
                         % (FOLD_CSV, len(rows)))
    if not set(table.values()) <= set(range(N_FOLDS)):
        raise SystemExit("%s: a fold outside 0..%d" % (FOLD_CSV, N_FOLDS - 1))
    return table


# --- putting a picture on a film or on the canvas ------------------------------------------------------------

def render(rec, factor=1.0):
    """The picture as a BUU-sized film times `factor`: (uint8 grey image, points (22, 2) float32 with NaN where missing),
    un-stretched, mirrored when the record says so, and resized so the median vertebra width is BUU's times factor."""
    from PIL import Image, ImageOps
    k = rec["scale"] * factor
    with Image.open(rec["path"]) as im:
        im = im.convert("L")
        w, h = im.size
        if rec["flip"]:
            im = ImageOps.mirror(im)
        nw, nh = max(1, int(round(FRAME * k))), max(1, int(round(FRAME * rec["aspect"] * k)))
        img = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    pts = rec["points"].astype(np.float64) * np.array([nw / FRAME, nh / (FRAME * rec["aspect"])])
    return img, pts.astype(np.float32)


def canvas_factor(height, width):
    """The factor buu._fit gives a BUU side film on average (median over the 2,000), so a landmark picture's vertebrae
    land on the canvas at the size BUU's do."""
    s = buu_reference()["sizes"]
    return float(np.median(np.minimum(height / s[:, 1], width / s[:, 0])))


def canvas_set(height=512, width=384):
    """Every record on the corner reader's canvas as BUU pictures are (centred, black padding): pix (M, height, width)
    uint8, pts (M, 22, 2) float32 in canvas pixels, NaN where the set has no point or the point falls off the canvas,
    and the records."""
    recs = load("train")
    f = canvas_factor(height, width)
    pix = np.zeros((len(recs), height, width), np.uint8)
    pts = np.full((len(recs), N_POINTS, 2), np.nan, np.float32)
    for n, r in enumerate(recs):
        img, p = render(r, f)
        nh, nw = img.shape
        ox, oy = (width - nw) // 2, (height - nh) // 2
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + nw, width), min(oy + nh, height)
        pix[n, y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        q = p + np.array([ox, oy], np.float32)
        off = (q[:, 0] < 0) | (q[:, 0] >= width) | (q[:, 1] < 0) | (q[:, 1] >= height)
        q[off] = np.nan
        pts[n] = q
    return pix, pts, recs


# --- the gate ----------------------------------------------------------------------------------------------------

def report(recs, verbose=True, key="points"):
    """The geometry report of the mapped vertebrae against BUU's drawn ones; returns (share inside, number of vertebrae)."""
    ref = buu_reference()
    pts = np.stack([r[key].astype(np.float64) for r in recs])
    q = pts[:, :20].reshape(len(recs), 5, 4, 2)
    have = ~np.isnan(q).any(axis=(2, 3))
    g = geometry(np.where(have[..., None, None], q, 1.0))
    pl = ref["pooled"]
    a = (g["hw"] >= pl["hw"][0]) & (g["hw"] <= pl["hw"][1])
    b = (g["wedge"] >= pl["wedge"][0]) & (g["wedge"] <= pl["wedge"][1])
    ok = a & b & have                                   # the gate: BUU's pooled L1..L5 1st-99th percentiles
    per = {}
    for lev in range(5):                                # information only: against each level's own percentiles
        la = (g["hw"][:, lev] >= ref["per_level"]["hw"][lev, 0]) & (g["hw"][:, lev] <= ref["per_level"]["hw"][lev, 1])
        lb = (g["wedge"][:, lev] >= ref["per_level"]["wedge"][lev, 0]) & (g["wedge"][:, lev] <= ref["per_level"]["wedge"][lev, 1])
        hv = have[:, lev]
        per[lev] = (int(hv.sum()), int((la & lb & hv).sum()), float(la[hv].mean()), float(lb[hv].mean()))
    share = float(ok.sum() / have.sum())
    if verbose:
        print("gate (each picture's film aspect fitted, see the docstring): %d pictures, %d mapped vertebrae (L1..L5), S1 upper edge in %d pictures; BUU median "
              "vertebra width %.1f px, height / width %.2f" % (len(recs), int(have.sum()),
                                                               sum(r["has_s1"] for r in recs), ref["width"], ref["hw_median"]))
        print("  BUU pooled 1st-99th percentiles (the gate): height/width %.2f-%.2f; endplate angle %.1f..%.1f"
              % (pl["hw"] + pl["wedge"]))
        print("  BUU 1st-99th percentiles, per level (L1..L5): height/width %s; endplate angle (upper minus lower edge, degrees) %s"
              % (", ".join("%.2f-%.2f" % tuple(v) for v in ref["per_level"]["hw"]),
                 ", ".join("%.1f..%.1f" % tuple(v) for v in ref["per_level"]["wedge"])))
        lo, hi = GATE_PCT
        for name, gk in (("height/width", "hw"), ("endplate angle", "wedge")):
            v = g[gk][have]
            print("  landmark %-14s percentiles 1/50/99: %s   BUU: %s" % (
                name, np.round(np.percentile(v, [lo, 50, hi]), 2),
                np.round(np.percentile(ref["geom"][gk], [lo, 50, hi]), 2)))
        for lev, (n, k, sa, sb) in per.items():
            print("  L%d: %d vertebrae, %d inside both of its own level's ranges (information) (%.1f%%; height/width %.1f%%, angle %.1f%%)"
                  % (lev + 1, n, k, 100.0 * k / max(n, 1), 100 * sa, 100 * sb))
        print("  share of mapped vertebrae with height/width and endplate angle inside BUU's 1st-99th percentiles: "
              "%.1f%% (%d of %d); the gate wants %.0f%%" % (100 * share, int(ok.sum()), int(have.sum()), 100 * GATE_SHARE))
        if key == "points":
            fixed, _n = report(recs, verbose=False, key="points_fixed")
            print("  for reference, with every film's aspect fixed at BUU's median %.3f instead of fitted: %.1f%%"
                  % (ref["aspect"], 100 * fixed))
        print("  corner order: point 0 of a line is the end on the image's left in every mapped line (checked in selftest)")
    return share, int(have.sum())


# --- scoring against the published benchmark --------------------------------------------------------------------

SCORED = slice(8, 22)               # our levels L3, L4, L5 and S1's upper edge: points 8..21
WITHIN = (5.0, 10.0, 15.0)


def file_frame(rec):
    """The record's 22 points in its picture FILE's 640 x 640 frame: mirrored back when load() mirrored it, y
    un-stretched back, then each line's ends ordered by x (point 0 on the left) as a reader of that file would."""
    p = rec["points"].astype(np.float64).copy()
    if rec["flip"]:
        p[:, 0] = FRAME - p[:, 0]
    p[:, 1] = p[:, 1] / rec["aspect"]
    lines = p.reshape(11, 2, 2)
    swap = lines[:, 0, 0] > lines[:, 1, 0]
    lines[swap] = lines[swap][:, ::-1]
    return lines.reshape(N_POINTS, 2)


def to_frame(points, size):
    """Points in the pixels of a (width, height) picture -> its 640 x 640 export frame."""
    return np.asarray(points, np.float64) * np.array([FRAME / size[0], FRAME / size[1]])


def read_predictions(path):
    """{name: (22, 2) float64} from a .npz (names, points) or .csv (name,point,x,y); see the docstring."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        names, pts = [str(n) for n in z["names"]], np.asarray(z["points"], np.float64)
        assert pts.shape[1:] == (N_POINTS, 2) and len(names) == len(pts), "points must be (n, 22, 2) for n names"
        return dict(zip(names, pts))
    out = {}
    with open(path, encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        assert head == ["name", "point", "x", "y"], "csv header must be name,point,x,y"
        for line in f:
            if not line.strip():
                continue
            n, k, x, y = line.strip().split(",")
            out.setdefault(n, np.full((N_POINTS, 2), np.nan))[int(k)] = (float(x), float(y))
    return out


def score_points(truth, pred):
    """truth, pred: (n, 22, 2) in the 640 frame (NaN = none). Over the scored points the truth has: the share within each
    of WITHIN px (a missing prediction is a miss), the mean distance over the predicted ones, and the counts."""
    t, q = truth[:, SCORED], pred[:, SCORED]
    has = ~np.isnan(t).any(-1)
    got = has & ~np.isnan(q).any(-1)
    d = np.linalg.norm(q - t, axis=-1)
    return {"n": int(has.sum()), "predicted": int(got.sum()),
            "share": {r: float(((d <= r) & got).sum() / max(has.sum(), 1)) for r in WITHIN},
            "mean_px": float(d[got].mean()) if got.any() else float("nan")}


def score(path, part="train", verbose=True):
    """Score a prediction file against the labels of `part`; returns score_points' dict. Exits when part has no labels."""
    recs = load(part)
    if not recs or not recs[0]["labelled"]:
        raise SystemExit("landmarks score: Dataset/%s has %d pictures and no label of any kind, so there is nothing to "
                         "score against (use --part train for the labelled pictures)" % (part.capitalize(), len(recs)))
    from PIL import Image
    pred = read_predictions(path)
    truth, got = [], []
    for r in recs:
        key = r["name"] if r["name"] in pred else r["base"] if r["base"] in pred else None
        if key is None:
            continue
        with Image.open(r["path"]) as im:
            size = im.size
        truth.append(file_frame(r))
        got.append(to_frame(pred[key], size))
    if not truth:
        raise SystemExit("landmarks score: none of the %d predicted names matches a %s picture" % (len(pred), part))
    res = score_points(np.stack(truth), np.stack(got))
    res["pictures"] = len(truth)
    if verbose:
        print("landmarks score (%s, %d pictures, L3..S1 upper edge, %d labelled keypoints, %d predicted; 640 x 640 frame):"
              % (part, len(truth), res["n"], res["predicted"]))
        print("  within 5 px %.2f%%, 10 px %.2f%%, 15 px %.2f%%; mean distance %.2f px (published best: 76.75%% within 5 px)"
              % (100 * res["share"][5.0], 100 * res["share"][10.0], 100 * res["share"][15.0], res["mean_px"]))
    return res


def _score_fixture():
    """The scoring on known answers: the truth shifted by 3, 8, 12 and 20 px and one point missing."""
    import shutil
    import tempfile
    recs = load("train")[:3]
    truth = np.stack([file_frame(r) for r in recs])
    for t in truth:                                                        # point 0 of every mapped line is its left end
        ln = t.reshape(11, 2, 2)
        ok = ~np.isnan(ln).any(axis=(1, 2))
        assert (ln[ok, 0, 0] <= ln[ok, 1, 0]).all()
    pred = truth.copy()
    have = np.argwhere(~np.isnan(truth[:, SCORED]).any(-1))               # (picture, scored point) the truth has
    assert len(have) >= 8
    for (i, k), dx in zip(have[:4], (3.0, 8.0, 12.0, 20.0)):
        pred[i, 8 + k, 0] += dx
    pred[have[4][0], 8 + have[4][1]] = np.nan
    res = score_points(truth, pred)
    n = len(have)
    assert res["n"] == n and res["predicted"] == n - 1
    exact = n - 5                                                          # unmoved and still predicted
    assert abs(res["share"][5.0] - (exact + 1) / n) < 1e-9, res
    assert abs(res["share"][10.0] - (exact + 2) / n) < 1e-9
    assert abs(res["share"][15.0] - (exact + 3) / n) < 1e-9
    assert abs(res["mean_px"] - 43.0 / (n - 1)) < 1e-9
    assert np.allclose(to_frame([[100.0, 200.0]], (1280, 1600)), [[50.0, 80.0]])
    d = tempfile.mkdtemp(prefix="lmscore-")
    try:
        names = np.array([r["name"] for r in recs])
        z, c = os.path.join(d, "p.npz"), os.path.join(d, "p.csv")
        np.savez(z, names=names, points=pred)
        with open(c, "w", encoding="utf-8") as f:
            f.write("name,point,x,y\n")
            for nm, p in zip(names, pred):
                for k in range(N_POINTS):
                    if not np.isnan(p[k]).any():
                        f.write("%s,%d,%r,%r\n" % (nm, k, float(p[k, 0]), float(p[k, 1])))
        for fl in (z, c):
            got = read_predictions(fl)
            assert set(got) == set(names) and np.allclose(np.stack([got[nm] for nm in names]), pred, equal_nan=True)
        assert score(z, "train", verbose=False)["share"] == res["share"], "score() on a file gives the same shares"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return res


def selftest():
    test = load("test")
    train = load("train")
    _selftest_parts(test, train)
    res = _score_fixture()
    print("score fixture ok: %d keypoints, within 5/10/15 px %.1f/%.1f/%.1f%%" % (
        res["n"], 100 * res["share"][5.0], 100 * res["share"][10.0], 100 * res["share"][15.0]))
    recs = load()
    _selftest_records(recs)
    _selftest_points(recs)
    _selftest_gate(recs)
    _selftest_fold_rows()
    table = _selftest_fold_table(recs)
    _selftest_fold_report(recs, table)
    print("selftest ok")


def _selftest_parts(test, train):
    tr_paths = {r["path"] for r in train}
    assert len(test) == 16 and not any(r["labelled"] for r in test), "Test folder: 16 unlabelled films"
    assert all(os.sep + "Test" + os.sep in r["path"] and r["path"] not in tr_paths for r in test), "Test never joins training"
    assert all(os.sep + "Train" + os.sep in r["path"] for r in train), "training reads Dataset/Train only"
    print("parts: Train %d bases (labelled), Test %d films (no labels, never trained on); %d of the Test patients also "
          "have a labelled picture in Train" % (len(train), len(test), len({r["base"] for r in test} & {r["base"] for r in train})))


def _selftest_records(recs):
    n_buu = sum(r["source"] == "buu" for r in recs)
    n_oth = len(recs) - n_buu
    ids = sorted(r["id"] for r in recs if r["id"] is not None)
    print("distinct bases %d (BUU %d, other %d); BUU patient ids %d..%d, %d distinct"
          % (len(recs), n_buu, n_oth, ids[0], ids[-1], len(set(ids))))
    assert (len(recs), n_buu, n_oth) == (EXPECT["bases"], EXPECT["buu"], EXPECT["other"]),         "bases %d (BUU %d, other %d), measured %s" % (len(recs), n_buu, n_oth, EXPECT)
    assert 2662 <= ids[0] and ids[-1] <= 4951, "BUU ids outside 2662..4951"
    assert not set(ids) & _split_ids(), "a BUU id of split.json is among the landmark pictures"


def _selftest_points(recs):
    for r in recs:
        p = r["points"].astype(np.float64)
        assert p.shape == (N_POINTS, 2) and r["scale"] > 0
        for lev in range(5):
            q = p[4 * lev:4 * lev + 4]
            if not np.isnan(q).any():
                assert plausible(q), "%s L%d is not a plausible quadrilateral" % (r["name"], lev + 1)
        lines = p.reshape(11, 2, 2)
        ok = ~np.isnan(lines).any(axis=(1, 2))
        assert (lines[ok, 0, 0] < lines[ok, 1, 0]).all(), "%s: point 0 of a line is not its left end" % r["name"]
        assert r["n_levels"] >= 1


def _selftest_gate(recs):
    share, n = report(recs)
    assert share >= GATE_SHARE, "gate missed: %.1f%% of %d vertebrae inside, wants %.0f%%" % (100 * share, n, 100 * GATE_SHARE)


def _selftest_fold_rows():
    # the folds (idea 119): grouped by patient, the same however the records come, all five folds used; the csv matches
    synth = [{"base": b} for b in ["2700-F-040Y1", "2700-M-040Y1", "2701-F-050Y1", "2702-M-060Y1", "2703-F-040Y1"]
             + ["Case_%d" % i for i in range(1, 9)]]
    rows = fold_rows(synth)
    assert rows == fold_rows(list(reversed(synth))) and rows == fold_rows(synth), "the fold split is not deterministic"
    by_patient = collections.defaultdict(set)
    for _b, p, k in rows:
        by_patient[p].add(k)
    assert all(len(v) == 1 for v in by_patient.values()), "a patient's pictures split across folds"
    assert by_patient["2700"] and len(by_patient) == 12, "the two BUU pictures of one id are one patient"
    assert {k for _b, _p, k in rows} == set(range(N_FOLDS)), "a fold is empty"


def _selftest_fold_table(recs):
    table = fold_table()
    for k in range(N_FOLDS):
        train_k, held = split_fold(recs, table, k)
        assert len(train_k) + len(held) == len(recs) and not {r["base"] for r in train_k} & {r["base"] for r in held}
    tr = collections.Counter(table[r["base"]] for r in recs)
    pat = collections.defaultdict(set)
    for r in recs:
        pat[patient_of(r["base"])].add(table[r["base"]])
    assert all(len(v) == 1 for v in pat.values()), "a patient's pictures split across folds in landmark_folds.csv"
    assert sum(tr.values()) == len(recs) and set(tr) == set(range(N_FOLDS)), "landmark_folds.csv does not cover the set"
    return table


def _selftest_fold_report(recs, table):
    parts = []
    for k in range(N_FOLDS):
        mine = [r for r in recs if table[r["base"]] == k]
        parts.append("fold %d: %d pictures, %d patients (%d BUU, %d other)" % (
            k, len(mine), len({patient_of(r["base"]) for r in mine}), sum(r["source"] == "buu" for r in mine),
            sum(r["source"] != "buu" for r in mine)))
    print("folds (landmark_folds.csv): " + "; ".join(parts))


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["selftest", "gate", "score", "folds"])
    ap.add_argument("file", nargs="?", help="score: the prediction .npz or .csv (see the docstring)")
    ap.add_argument("--part", choices=["train", "test"], default="test", help="score: which part's labels (default test)")
    a = ap.parse_args()
    if a.what == "score":
        if not a.file:
            raise SystemExit("score wants a prediction file")
        score(a.file, a.part)
        return 0
    if a.what == "folds":
        write_folds()
        return 0
    if a.what == "gate":
        report(load())
        return 0
    try:
        selftest()
    except AssertionError as e:
        print("selftest FAILED: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
