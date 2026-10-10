"""Where the drawn fractures are, on the pictures the detector reads (idea 125): one 0/1 map per
picture, on the net's stride-16 grid (24 x 24 at 384 px), with a mask flag that says whether the
picture had a box source at all.

Box sources, read only, from the YOLO label files beside each dataset:
  grazpedwri  yolov5/labels/<filestem>.txt, the class index 3 (meta.yaml's 'fracture')
  fracatlas   Annotations/YOLO/<image stem>.txt, class 0 (labels.txt's 'fractured'); a healthy
              picture has an empty file and gets an all-zero map with its mask flag set
  other collections have no box source: their mask flags stay 0 and no loss is taken on them
A picture whose EXIF orientation is not 1 keeps mask 0: prepare.read_grey reads the stored array,
and a label drawn in the rotated frame would sit on the wrong place (FracAtlas IMG0002628.jpg, orientation 8).

A box is drawn on the prepared picture exactly where prepare.py's letterbox puts it: the source
pixel edges go through prepare.placement, so the scale and the centring are the picture cache's own.
A cell is set when a box overlaps it, and a box always sets the cell its centre falls in.

The maps are written keyed like the picture cache (same rows, same image list), beside it under
work/: <dataset>_<size>_boxes.npy, <dataset>_<size>_boxes_mask.npy, <dataset>_<size>_boxes.json. Only
the rows detector.py's table gives are built (the development rows), so no test label is opened.

Usage: python detector.py boxes [--size 384]"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402
import preds  # noqa: E402
import prepare  # noqa: E402

STRIDE = 16        # the net's stride-16 map: a stride-2 stem and five halvings, of which the first three
                   # reach 16 (384 px gives 24 x 24; the 6 x 6 grid of the picture's last stage is not used)
ORIENTATION_TAG = 0x0112
SOURCES = {
    "grazpedwri": (os.path.join("yolov5", "labels"), 3),
    "fracatlas": (os.path.join("Annotations", "YOLO"), 0),
}


def cache_paths(dataset, size=prepare.SIZE):
    base = os.path.join(preds.CACHE, "%s_%s_boxes" % (dataset, size))
    return base + ".npy", base + "_mask.npy", base + ".json"


def has_cache(dataset, size=prepare.SIZE):
    """Whether the dataset's box cache is on disk (all three files)."""
    return all(os.path.exists(p) for p in cache_paths(dataset, size))


def orientation(path):
    """The picture's EXIF orientation tag, 1 when it has none."""
    with Image.open(path) as im:
        return int(im.getexif().get(ORIENTATION_TAG, 1))


def grid(size=prepare.SIZE):
    """(rows, columns) of the cell map for a picture cache of `size`."""
    ch, cw = prepare.canvas(size)
    if ch % STRIDE or cw % STRIDE:
        raise SystemExit("the box map needs a canvas that is a multiple of %d; %s is not" % (STRIDE, size))
    return ch // STRIDE, cw // STRIDE


def source_boxes(dataset, key):
    """The fracture boxes of picture `key` as (cx, cy, w, h), normalised to the picture's own size;
    [] for a healthy picture; None when the dataset has no box source or the picture has no label file."""
    if dataset not in SOURCES:
        return None
    sub, cls = SOURCES[dataset]
    stem = key if dataset == "grazpedwri" else os.path.splitext(key)[0]
    p = os.path.join(datasets.root(dataset), sub, stem + ".txt")
    if not os.path.exists(p):
        return None
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5 and int(float(parts[0])) == cls:
                out.append(tuple(float(v) for v in parts[1:]))
    return out


def canvas_rects(boxes, w, h, size=prepare.SIZE):
    """The boxes as (x0, y0, x1, y1) in canvas pixels, through the picture's letterbox placement."""
    nw, nh, top, left = prepare.placement(w, h, size)
    rects = []
    for cx, cy, bw, bh in boxes:
        x0, x1 = (cx - bw / 2) * w, (cx + bw / 2) * w
        y0, y1 = (cy - bh / 2) * h, (cy + bh / 2) * h
        rects.append((left + x0 * nw / w, top + y0 * nh / h, left + x1 * nw / w, top + y1 * nh / h))
    return rects


def _span(a, b, n):
    """The first and last of n cells (STRIDE pixels each) that the span a..b overlaps, clipped to the grid."""
    lo = int(np.floor(a / STRIDE))
    hi = max(int(np.ceil(b / STRIDE)) - 1, lo)
    return min(max(lo, 0), n - 1), min(max(hi, 0), n - 1)


def cell_map(rects, size=prepare.SIZE):
    """The 0/1 cell map of canvas rectangles: every cell a rectangle overlaps is set."""
    rows, cols = grid(size)
    out = np.zeros((rows, cols), np.uint8)
    for x0, y0, x1, y1 in rects:
        c0, c1 = _span(x0, x1, cols)
        r0, r1 = _span(y0, y1, rows)
        out[r0:r1 + 1, c0:c1 + 1] = 1
    return out


def _write(path, arr):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def build(dataset, size, rows):
    """Write the box maps of the picture-cache rows `rows` (the caller's development rows). A row
    whose picture is unreadable, or whose dataset has no box source, keeps a zero map and mask 0."""
    npy, mask_p, js = cache_paths(dataset, size)
    keys = [str(k) for k in datasets.index(dataset).image]
    meta_p = prepare.cache_paths(dataset, size)[1]
    if not os.path.exists(meta_p):
        raise SystemExit("%s: no image cache; build it first with: %s" % (dataset, prepare.build_command(dataset, size)))
    shapes = json.load(open(meta_p, encoding="utf-8"))["source_shape"]
    maps = np.zeros((len(keys), *grid(size)), np.uint8)
    mask = np.zeros(len(keys), np.uint8)
    turned = 0
    for row in rows:
        key = keys[int(row)]
        if key not in shapes:
            continue
        boxes = source_boxes(dataset, key)
        if boxes is None:
            continue
        path = datasets.image_path(dataset, key)
        if path is not None and orientation(path) != 1:
            turned += 1
            continue
        w, h = shapes[key]
        maps[int(row)] = cell_map(canvas_rects(boxes, w, h, size), size)
        mask[int(row)] = 1
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    _write(npy, maps)
    _write(mask_p, mask)
    meta = {"dataset": dataset, "size": size, "images": keys, "rows_built": int(len(rows)),
            "masked": int(mask.sum()), "fracture_cells": int(maps.sum()),
            "boxed_pictures": int((maps.reshape(len(keys), -1).sum(axis=1) > 0).sum()),
            "orientation_masked": turned}
    tmp = js + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, js)
    print("%s: %d rows built, %d with a box source, %d masked for an EXIF orientation other than 1, "
          "%d pictures with a fracture cell, %d cells"
          % (dataset, len(rows), meta["masked"], turned, meta["boxed_pictures"], meta["fracture_cells"]), flush=True)
    return meta


def load(dataset, size=prepare.SIZE):
    """(maps, mask) of a dataset's box cache, memory-mapped and checked against its index."""
    npy, mask_p, js = cache_paths(dataset, size)
    if not has_cache(dataset, size):
        raise SystemExit("%s: no box cache at %s; build it with: python detector.py boxes --size %s"
                         % (dataset, npy, size))
    keys = [str(k) for k in datasets.index(dataset).image]
    if json.load(open(js, encoding="utf-8")).get("images") != keys:
        raise SystemExit("%s: the box cache %s does not match the dataset's index; rebuild it with: "
                         "python detector.py boxes --size %s" % (dataset, npy, size))
    return np.load(npy, mmap_mode="r"), np.load(mask_p, mmap_mode="r")
