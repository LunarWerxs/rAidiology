"""Score the spine X-ray corner reader as a vertebra detector: the box around each vertebra's four PREDICTED
corners against the box around its four DRAWN corners.

Dataset and paper: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646 (LICENSING.md, BUU-LSPINE row). Its Table 8 scores lumbar vertebra
detection: each vertebra's box is made from its drawn corners, five classes a view (L1 to L5), one box kept per
class. YOLOv5 got precision 81.93 and recall 95.82 on the front view (AP) and 83.45 and 95.72 on the side view (LA).

Point order (buu.py): points 4i, 4i+1, 4i+2, 4i+3 are vertebra L(i+1)'s upper edge (two points) then its lower edge
(two points), i = 0..4. Only the first 20 points are used; the side view's points 20 and 21 are S1's edge, no box.
A vertebra's box is the axis-aligned box over its four points. Vertebra i's predicted box is compared with vertebra
i's drawn box only (same class), by intersection over union (IoU). With one box a vertebra and five a picture, our
precision equals our recall, so both are scored from the same readings, only the paper's bars differ.

Two readings, because the paper says two things: its text says its scores average over IoU 0.5 to 0.95, while
YOLOv5's precision and recall columns are at IoU 0.5. So both are given (each per patient, then the mean over
patients, with a patient-bootstrap 95% interval from one shared draw of patients):
  a  the share of the five vertebrae found at IoU >= 0.5, in percent.
  b  the share found at IoU >= t, averaged over the ten thresholds 0.50, 0.55 ... 0.95, in percent.
Status against a bar: "ahead" when b's lower end is above the bar; else "ahead at IoU 0.5 only" when a's lower end is
above it; else "level" when a's upper end reaches it; else "behind". `base` is the mean-shape guess that looks at
nothing in the picture (corners.py), read the same way for context.

Usage: python vertebra_boxes.py score --view AP|LA --version N [--folds 0 1 2 3 4] [--out DIR]
       python vertebra_boxes.py selftest
Outputs: <WORK>/eval/spine_xray/vertebra-boxes/<view>-v<N>.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import paths  # noqa: E402
import corners  # noqa: E402

TS = np.round(np.arange(0.50, 0.951, 0.05), 2)
BOOT = 2000
BOOT_SEED = 62
BARS = {"AP": {"precision": 81.93, "recall": 95.82}, "LA": {"precision": 83.45, "recall": 95.72}}


def default_out():
    return os.path.join(paths.WORK, "eval", "spine_xray", "vertebra-boxes")


def boxes(pts):
    """(N, >=20, 2) corners -> (N, 5, 4) boxes [x0, y0, x1, y1], one a vertebra, from points 4i..4i+3."""
    p = np.asarray(pts, np.float64)[:, :20, :].reshape(len(pts), 5, 4, 2)
    return np.concatenate([p.min(axis=2), p.max(axis=2)], axis=-1)


def iou(a, b):
    """IoU of boxes [x0, y0, x1, y1] over the last axis, elementwise."""
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0.0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0.0, None)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def readings(u):
    """u (N, 5) IoUs -> (a, b) per patient, in percent."""
    a = 100.0 * (u >= 0.5).mean(axis=1)
    b = 100.0 * np.mean([(u >= t).mean(axis=1) for t in TS], axis=0)
    return a, b


def status(lo_a, hi_a, lo_b, bar):
    if lo_b > bar:
        return "ahead"
    if lo_a > bar:
        return "ahead at IoU 0.5 only"
    if hi_a >= bar:
        return "level"
    return "behind"


def score(view, version, folds):
    r = corners.load_oof(view, version, folds)
    truth = boxes(r["truth"])
    u = iou(boxes(r["pred"]), truth)
    n = len(u)
    a, b = readings(u)
    vals = np.stack([a, b], axis=1)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(BOOT, n))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    ba, bb = readings(iou(boxes(r["base"]), truth))
    bars = BARS[view]
    st = {k: status(float(lo[0]), float(hi[0]), float(lo[1]), bars[k]) for k in ("precision", "recall")}
    return {"view": view, "version": int(version), "folds": [int(k) for k in folds], "n_patients": int(n),
            "a": {"value": float(a.mean()), "lo": float(lo[0]), "hi": float(hi[0])},
            "b": {"value": float(b.mean()), "lo": float(lo[1]), "hi": float(hi[1])},
            "per_vertebra_a": [float(100.0 * (u[:, i] >= 0.5).mean()) for i in range(5)],
            "median_iou": float(np.median(u)),
            "base": {"a": float(ba.mean()), "b": float(bb.mean())},
            "bars": dict(bars), "status": st}


def report(s):
    print("%s view, version %d, folds %s: %d held-out patients, five vertebrae each (L1 to L5)"
          % (s["view"], s["version"], ",".join(map(str, s["folds"])), s["n_patients"]))
    print("  reading a, found at IoU 0.5:               %6.2f  (%.2f to %.2f)" % (s["a"]["value"], s["a"]["lo"], s["a"]["hi"]))
    print("  reading b, mean over IoU 0.50 to 0.95:     %6.2f  (%.2f to %.2f)" % (s["b"]["value"], s["b"]["lo"], s["b"]["hi"]))
    print("  per vertebra at IoU 0.5 (L1 to L5):        " + "  ".join("%.2f" % v for v in s["per_vertebra_a"]))
    print("  median IoU over all vertebrae:             %.4f" % s["median_iou"])
    print("  mean-shape guess that looks at nothing:    a %.2f, b %.2f" % (s["base"]["a"], s["base"]["b"]))
    for k in ("precision", "recall"):
        print("  %s against the paper's YOLOv5 bar %.2f: %s" % (k, s["bars"][k], s["status"][k]))


def selftest():
    one = np.array([0.0, 0.0, 2.0, 1.0])
    assert abs(float(iou(one, one)) - 1.0) < 1e-12, "identical boxes"
    assert float(iou(one, np.array([5.0, 5.0, 6.0, 6.0]))) == 0.0, "disjoint boxes"
    assert abs(float(iou(one, np.array([1.0, 0.0, 3.0, 1.0]))) - 1.0 / 3.0) < 1e-12, "half-width shift"
    pts = np.zeros((1, 22, 2))
    pts[0, 4:8] = [[1, 5], [3, 5], [0, 9], [4, 8]]
    assert boxes(pts)[0, 1].tolist() == [0.0, 5.0, 4.0, 9.0], "box of L2"
    # two patients, ten IoUs: patient 0 has 1.0, 0.9, 0.7, 0.55, 0.3; patient 1 has 0.95, 0.5, 0.45, 0.0, 0.8
    u = np.array([[1.0, 0.9, 0.7, 0.55, 0.3], [0.95, 0.5, 0.45, 0.0, 0.8]])
    a, b = readings(u)
    assert np.allclose(a, [80.0, 60.0]), a
    want0 = np.mean([sum(v >= t for v in u[0]) / 5 for t in TS]) * 100
    want1 = np.mean([sum(v >= t for v in u[1]) / 5 for t in TS]) * 100
    assert np.allclose(b, [want0, want1]) and abs(b[0] - 52.0) < 1e-9 and abs(b[1] - 36.0) < 1e-9, b
    assert status(90, 95, 85, 80) == "ahead"
    assert status(90, 95, 70, 80) == "ahead at IoU 0.5 only"
    assert status(78, 85, 70, 80) == "level"
    assert status(70, 75, 60, 80) == "behind"
    print("selftest: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="synthetic boxes, thresholds and the status rule (no data files)")
    s = sub.add_parser("score", help="box IoU of the corner reader's out-of-fold corners against the drawn ones")
    s.add_argument("--view", choices=["AP", "LA"], required=True)
    s.add_argument("--version", type=int, required=True)
    s.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    s.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return
    res = score(a.view, a.version, a.folds)
    d = a.out or default_out()
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%s-v%d.json" % (a.view, a.version))
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    report(res)
    print("written: %s" % p)


if __name__ == "__main__":
    main()
