"""BUU-LSPINE as the spine X-ray reader reads it: the files, the checks, the patient folds, the loader.

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646. Used under the owners' End-User License Agreement
(LICENSING.md, BUU-LSPINE row); cite it in every report that uses it.

Layout. <DATASETS>/public/buu_lspine/dataset2024/<a folder named in Thai>/{AP,LA}/ holds, for each of
2,000 patients, '<4-digit id>-<F|M>-<3-digit age>Y<view>.jpg' and a '.csv' of the same name; view digit
0 is the front film (AP), 1 the side film (LA). The ids are not contiguous (0001, 0003, ...), so a patient
is its id, never its position. The 400-patient sample (BUU-LSPINE_400/BUU-LSPINE_400/{AP,LA}) is the same
first 400 patients, and it is the release built for slip diagnosis: its workbook (BUU-LSPINE_400_report.xlsx,
sheet diag_all) lists 66 LA diagnoses on 59 patients, all of them the sample CSVs' class 3 or 4 patients.
Slip labels. In the first 400 patients the 2,000 set's posture marks (9 flexion, 10 extension) and some 0s
overwrote the sample's slip marks (found 2026-10-04): the set writes most of the sample's class 3 lines and all
its class 4 lines as 0, 9 or 10, so on its own it calls 49 of the sample's 59 slipped patients normal. So
`annotations('LA')` merges `sample_slip_marks()`: for every sample patient, on each lower-edge row where the
sample's class is 3 or 4, the class becomes the sample's. Nothing else of the sample is used (coordinates stay
the set's), and the sample is never read as extra patients. For ids 401 to 2000 there is no second source, and
the set's negatives there are probably not all negatives (the audit estimates about 70 to 85 more slipped
patients carry a class 9 line), so a slip score is reported on the 400 verified patients (`verified_ids()`)
first. `check` prints the merge.

Annotation rows (read from the data, 2026-10-03). A CSV has no header; one row per vertebral edge line:
x1,y1,x2,y2,class. AP has 10 rows, LA 11, and no file in the 2,000 set breaks that.
  * Row order is top to bottom by vertebra, and within a vertebra upper edge then lower edge: row 2i
    is the upper edge of L(i+1) and row 2i+1 its lower edge (i = 0..4 for L1..L5); on the side view row
    10 is S1's upper edge. In 28 side files two neighbouring rows' mean heights are swapped by a few
    pixels (the lower edge of one vertebra drawn lower on the page than the next upper edge), so the
    order is the identity of the row, never its height rank; nothing here sorts rows.
  * The slip classes sit on the LOWER-edge row of the vertebra that slipped (rows 1, 3, 5, 7, 9), so a
    slip belongs to the level between that lower edge and the next row's upper edge: level i (0..4) is
    L(i+1) over L(i+2), and level 4 is L5 over S1.
  * Which end is which. Every side film faces left (the patient looks to the left of the image), and
    the data agrees: reading x1 as the front (anterior) end, the lower vertebra's upper edge puts the
    upper vertebra's back-lower corner forward of the lower's back-upper corner by +0.12 of the
    endplate's length on class 3 (anterolisthesis) lines and by -0.16 on class 4 (retrolisthesis);
    reading x2 as the front end flips both signs. So on LA, the end with the smaller x is anterior. In
    3 side files one or more rows are written with x1 greater than x2 (0169, 1110, 1997), so `load`
    puts the end with the smaller x first on every line, in both views. On AP, point 0 of a line is
    therefore the end on the image's left, point 1 the end on its right.
  * Classes: 0 normal; AP 1/2 left/right laterolisthesis, 5/6 side bending left/right, 7/8 rotation
    left/right; LA 3 anterolisthesis, 4 retrolisthesis, 9 flexion, 10 extension, 20/21 rare
    transitional-vertebra marks.

Points are ordered line by line, two points a line: point 2r the end with the smaller x, 2r + 1 the
other. 20 points on AP, 22 on LA.

Usage: python buu.py check
       python buu.py split
"""
import argparse
import collections
import functools
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import paths  # noqa: E402

VIEWS = {"AP": {"digit": 0, "rows": 10}, "LA": {"digit": 1, "rows": 11}}
FOLDS = 5
SEED = 0
SLIP_CLASSES = (3, 4)
SLIP_ROWS = (1, 3, 5, 7, 9)          # the lower-edge rows that carry a slip class
SPLIT_PATH = os.path.join(HERE, "split.json")
NAME = re.compile(r"^(\d{4})-([FM])-(\d{3})Y([01])\.(csv|jpg)$")
CITATION = ("Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis "
            "Detection. Applied Sciences 2023;13(15):8646")


def root():
    """The folder holding AP and LA, found by glob because its parent has Thai characters in its name."""
    hits = sorted(glob.glob(os.path.join(paths.DATASETS, "public", "buu_lspine", "dataset2024", "*", "AP")))
    if not hits:
        raise SystemExit("BUU-LSPINE's dataset2024/*/AP not found under %s" % paths.DATASETS)
    return os.path.dirname(hits[0])


def sample_root():
    return os.path.join(paths.DATASETS, "public", "buu_lspine", "BUU-LSPINE_400", "BUU-LSPINE_400")


def files(view, base=None, ext="csv"):
    """{patient id (int): path} for one view's files of one extension."""
    out = {}
    for p in sorted(glob.glob(os.path.join(base or root(), view, "*." + ext))):
        m = NAME.match(os.path.basename(p))
        if not m or int(m.group(4)) != VIEWS[view]["digit"]:
            continue
        out[int(m.group(1))] = p
    return out


def read_csv(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a


def lines_to_points(a):
    """A CSV's rows as points, the end with the smaller x first on every line: (2 * rows, 2) x, y."""
    a = np.asarray(a, np.float64)
    swap = a[:, 0] > a[:, 2]
    p0 = np.where(swap[:, None], a[:, 2:4], a[:, 0:2])
    p1 = np.where(swap[:, None], a[:, 0:2], a[:, 2:4])
    return np.stack([p0, p1], axis=1).reshape(-1, 2)


@functools.lru_cache(maxsize=1)
def sample_slip_marks():
    """The 400-patient sample's slip marks: {patient id: {row: class 3 or 4}}, one entry for every patient of the
    sample's LA CSVs (an empty dict when it has no slip mark), only on the lower-edge rows SLIP_ROWS."""
    if not os.path.isdir(os.path.join(sample_root(), "LA")):
        raise SystemExit("the 400-patient sample's LA folder is not under %s" % sample_root())
    out = {}
    for i, p in files("LA", sample_root()).items():
        a = read_csv(p)
        out[i] = {r: int(a[r, 4]) for r in SLIP_ROWS if r < len(a) and int(a[r, 4]) in SLIP_CLASSES}
    return out


@functools.lru_cache(maxsize=1)
def verified_ids():
    """The ids of the 400 sample patients, whose slip labels the sample's diagnoses verify."""
    return frozenset(sample_slip_marks())


def annotations(view, merged=True):
    """Everything but the pixels, in original pixel coordinates, patients in ascending id.
    Returns ids (N,), points (N, 2 * rows, 2) float32, classes (N, rows) int. On LA the sample's slip marks are
    merged into the classes (see the docstring); merged=False gives the set's own."""
    csvs = files(view)
    ids = np.array(sorted(csvs), dtype=np.int64)
    rows = VIEWS[view]["rows"]
    pts = np.zeros((len(ids), 2 * rows, 2), np.float32)
    cls = np.zeros((len(ids), rows), np.int64)
    marks = sample_slip_marks() if merged and view == "LA" else {}
    for n, i in enumerate(ids):
        a = read_csv(csvs[int(i)])
        if a.shape != (rows, 5):
            raise SystemExit("%s: %s has %s rows, wants %d" % (view, csvs[int(i)], a.shape, rows))
        pts[n] = lines_to_points(a)
        cls[n] = a[:, 4].astype(np.int64)
        for r, c in marks.get(int(i), {}).items():
            cls[n, r] = c
    return ids, pts, cls


def slip_label(la_classes):
    """Per patient: 1 when any side-view line is class 3 or 4."""
    return np.isin(la_classes, SLIP_CLASSES).any(axis=1).astype(np.int64)


def merge_report():
    """Print what annotations('LA') merges from the sample: lines by the set's own class, patients with a slip
    before and after, overall and among the 400 verified patients."""
    marks = sample_slip_marks()
    ids, _p, own = annotations("LA", merged=False)
    _i, _q, now = annotations("LA")
    where = {int(i): n for n, i in enumerate(ids)}
    by_set, by_sample = collections.Counter(), collections.Counter()
    for i, m in marks.items():
        for r, c in m.items():
            if i not in where:
                continue
            by_sample[c] += 1
            if own[where[i], r] != c:
                by_set[(c, int(own[where[i], r]))] += 1
    print("slip merge from the 400-sample (%d patients, %d class 3 or 4 lines): lines changed by (sample class, set class) %s; "
          "sample lines already equal to the set's %d"
          % (len(marks), sum(by_sample.values()), dict(sorted(by_set.items())),
             sum(by_sample.values()) - sum(by_set.values())))
    ver = np.isin(ids, sorted(verified_ids()))
    for name, c in (("set's own", own), ("merged", now)):
        y = slip_label(c)
        print("  patients with a slip, %s labels: %d of %d overall, %d of %d among the 400 verified"
              % (name, int(y.sum()), len(ids), int(y[ver].sum()), int(ver.sum())))


def check():
    bad = []
    csv_ap, csv_la = files("AP"), files("LA")
    jpg_ap, jpg_la = files("AP", ext="jpg"), files("LA", ext="jpg")
    patients = sorted(set(csv_ap) | set(csv_la) | set(jpg_ap) | set(jpg_la))
    images = len(jpg_ap) + len(jpg_la)
    print("root: %s" % root())
    print("patients %d, images %d (AP %d, LA %d), csv files AP %d LA %d"
          % (len(patients), images, len(jpg_ap), len(jpg_la), len(csv_ap), len(csv_la)))
    rowcount, classes = {}, {}
    for view, csvs in (("AP", csv_ap), ("LA", csv_la)):
        want = VIEWS[view]["rows"]
        rc, cc = collections.Counter(), collections.Counter()
        for i, p in csvs.items():
            a = read_csv(p)
            rc[a.shape[0]] += 1
            cc.update(a[:, 4].astype(int).tolist())
            if a.shape != (want, 5):
                bad.append("%s %04d has %s, wants (%d, 5)" % (view, i, a.shape, want))
        rowcount[view], classes[view] = rc, cc
        print("%s: rows per file %s (want %d); lines by class %s"
              % (view, dict(sorted(rc.items())), want, dict(sorted(cc.items()))))
    if set(jpg_ap) != set(csv_ap) or set(jpg_la) != set(csv_la) or set(csv_ap) != set(csv_la):
        bad.append("patients differ between AP csv %d, LA csv %d, AP jpg %d, LA jpg %d"
                   % (len(csv_ap), len(csv_la), len(jpg_ap), len(jpg_la)))
    # the 400 sample against the set, for the same ids
    sr = sample_root()
    for view in VIEWS:
        mine = files(view)
        sample = files(view, sr) if os.path.isdir(os.path.join(sr, view)) else {}
        same = coord = cls_only = missing = 0
        pairs = collections.Counter()
        shown = 0
        for i, p in sample.items():
            if i not in mine:
                missing += 1
                continue
            a, b = read_csv(p), read_csv(mine[i])
            if a.shape != b.shape:
                coord += 1
                continue
            same_xy = np.allclose(a[:, :4], b[:, :4], atol=1e-6)
            diff_cls = a[:, 4] != b[:, 4]
            if same_xy and not diff_cls.any():
                same += 1
                continue
            if same_xy:
                cls_only += 1
            else:
                coord += 1
            pairs.update(zip(a[diff_cls, 4].astype(int).tolist(), b[diff_cls, 4].astype(int).tolist()))
            if shown < 3:
                shown += 1
                print("  sample %s %04d differs from the set's CSV (max coordinate difference %.3f px, %d class marks)"
                      % (view, i, float(np.abs(a[:, :4] - b[:, :4]).max()) if a.shape == b.shape else float("nan"),
                         int(diff_cls.sum())))
        print("%s 400-sample: %d CSVs, %d equal the set's, %d differ only in class marks, %d differ in coordinates, "
              "%d not in the set" % (view, len(sample), same, cls_only, coord, missing))
        if pairs:
            print("  class marks (sample -> set) that differ: %s" % dict(sorted(pairs.items())))
    merge_report()
    if len(patients) != 2000:
        bad.append("%d patients, wants 2000" % len(patients))
    if images != 4000:
        bad.append("%d images, wants 4000" % images)
    print("citation: %s" % CITATION)
    if bad:
        for b in bad[:20]:
            print("FAIL: " + b)
        return 1
    print("OK: 2000 patients, 4000 images, every AP file 10 rows, every LA file 11")
    return 0


def split():
    """Five folds by patient, seed 0, stratified on 'any LA class 3 or 4' so slipped patients spread evenly.
    split.json is locked: it was stratified on the set's own labels, before the sample's slip marks were merged,
    so running this now would stratify on the merged label and change the folds. It is not rewritten."""
    from sklearn.model_selection import StratifiedKFold
    ids, _p, cls = annotations("LA")
    y = slip_label(cls)
    fold = np.zeros(len(ids), np.int64)
    for k, (_tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(ids, y)):
        fold[te] = k
    out = {"seed": SEED, "folds": FOLDS, "stratified_on": "any LA line of class 3 or 4",
           "fold_of": {"%04d" % i: int(f) for i, f in zip(ids, fold)}}
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    for k in range(FOLDS):
        m = fold == k
        print("fold %d: %d patients, %d with a slip" % (k, int(m.sum()), int(y[m].sum())))
    print("wrote %s" % SPLIT_PATH)


def folds_of(ids):
    """Fold number per patient id, from split.json."""
    if not os.path.exists(SPLIT_PATH):
        raise SystemExit("no %s; run: python buu.py split" % SPLIT_PATH)
    with open(SPLIT_PATH, encoding="utf-8") as f:
        fo = json.load(f)["fold_of"]
    return np.array([fo["%04d" % int(i)] for i in ids], dtype=np.int64)


def cache_path(view, height, width):
    return os.path.join(paths.WORK, "eval", "spine_xray", "cache", "%s-%dx%d.npz" % (view, height, width))


def _fit(im, height, width):
    """One image as a (height, width) uint8 canvas, aspect kept, centred, padded with black.
    Returns pixels, scale, (x offset, y offset): resized = original * scale + offset, in continuous
    coordinates where pixel i spans [i, i + 1)."""
    from PIL import Image
    im = im.convert("L")
    w, h = im.size
    s = min(height / h, width / w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    canvas = np.zeros((height, width), np.uint8)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas, nw / w, np.array([ox, oy], np.float64), (nw / w, nh / h)


def load(view, height=512, width=384, use_cache=True):
    """The whole view as arrays, patients in ascending id.

    Returns a dict:
      pix     (N, height, width) uint8, grayscale, aspect kept and padded black
      pts     (N, K, 2) float32, the drawn corners in resized coordinates (x, y), K = 20 (AP) or 22 (LA)
      scale   (N, 2) float32, per axis (x, y) factor: resized = original * scale + offset
      offset  (N, 2) float32, the padding's (x, y)
      size    (N, 2) int32, the original image's (width, height)
      classes (N, rows) int64, the class of each line
      ids     (N,) int64, patient ids
    Original coordinates from resized: (p - offset) / scale. Decoding 2,000 JPGs takes minutes, so the
    result is cached under paths.WORK/eval/spine_xray/cache."""
    from PIL import Image
    cp = cache_path(view, height, width)
    jpgs = files(view, ext="jpg")
    ids, pts_o, cls = annotations(view)
    if use_cache and os.path.exists(cp):
        z = np.load(cp)
        if np.array_equal(z["ids"], ids):
            out = {k: z[k] for k in z.files}
            out["classes"] = cls                  # the cache may hold classes from before the merge
            return out
    n = len(ids)
    pix = np.zeros((n, height, width), np.uint8)
    scale = np.zeros((n, 2), np.float32)
    offset = np.zeros((n, 2), np.float32)
    size = np.zeros((n, 2), np.int32)
    for k, i in enumerate(ids):
        with Image.open(jpgs[int(i)]) as im:
            size[k] = im.size
            pix[k], _s, offset[k], sc = _fit(im, height, width)
        scale[k] = sc
        if (k + 1) % 250 == 0:
            print("  %s: %d / %d images read" % (view, k + 1, n), flush=True)
    pts = pts_o * scale[:, None, :] + offset[:, None, :]
    out = {"pix": pix, "pts": pts.astype(np.float32), "scale": scale, "offset": offset, "size": size,
           "classes": cls, "ids": ids}
    if use_cache:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        np.savez(cp, **out)
    return out


def main():
    sys.stdout.reconfigure(errors="replace")          # the data folder's name is Thai; a cp1252 console cannot print it
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["check", "split"])
    a = ap.parse_args()
    if a.what == "check":
        sys.exit(check())
    split()


if __name__ == "__main__":
    main()
