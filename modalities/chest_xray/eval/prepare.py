"""Decode every chest X-ray once, into one cached array every later step reads.

This writes, under <work>/eval/chest/:
  nih_<size>.npy    a uint8 array of shape (N, size, size), opened later as a memory map, so a
                    network reads pixels without decoding a 1024-pixel PNG again
  nih_<size>.json   {"size", "built", "images", "unreadable", "source_shape"}: the image names in row
                    order, the names that could not be decoded, and each picture's [width, height]

Row order is datasets.index().image order, so a row number is stable and load() hands back the
image -> row mapping.

Why each choice:
- The X-ray fracture module's own reader (modalities/xray/eval/prepare.py read_any), not a copy: the
  same window (0.5th to 99.5th percentile above the 1st) and the same letterbox, so chest and bone
  pictures meet one intensity rule. It is loaded by file path because both modules are called
  prepare.py and datasets.py, and a plain import would pick up the wrong one. Its picture reader
  converts RGBA and RGB to grey itself, which NIH needs: a few of its PNGs are RGBA.
- An unreadable picture (missing file, decode error) stays zeros and is listed; the build goes on.
- Pixels only: no labels, no split, so the test guard is not involved.
- Resumable and atomic like the X-ray build: a cache that matches the index is kept unless
  --rebuild; the array is written as .part and renamed when complete, the json last, so a run cut
  short never leaves a cache that looks finished.
- Worker processes, not threads (the X-ray build measured threads at no gain, the interpreter lock
  being held for most of the work). Each opens the memory map itself, so pixels never travel
  between processes.

Usage: python prepare.py [--size 384] [--jobs N] [--rebuild]"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402
from core.rows import FileRows  # noqa: E402
import datasets  # noqa: E402

SIZE = 384
PROGRESS = 2000              # images between progress lines
CACHE = os.path.join(paths.WORK, "eval", "chest")
XRAY = os.path.join(ROOT, "modalities", "xray", "eval")


def _xray_prepare():
    """The X-ray module's prepare.py, loaded by path under its own name. It imports its sibling
    `datasets` and `preds`, so while it loads, those names are the X-ray ones, and ours are put back."""
    if "xray_prepare" in sys.modules:
        return sys.modules["xray_prepare"]
    saved = {n: sys.modules.pop(n, None) for n in ("datasets", "preds", "prepare")}
    sys.path.insert(0, XRAY)
    try:
        spec = importlib.util.spec_from_file_location("xray_prepare", os.path.join(XRAY, "prepare.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["xray_prepare"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(XRAY)
        for n in ("datasets", "preds"):
            sys.modules.pop(n, None)
        for n, m in saved.items():
            if m is not None:
                sys.modules[n] = m
    return mod


def cache_paths(size=SIZE):
    base = os.path.join(CACHE, "nih_%d" % size)
    return base + ".npy", base + ".json"


def build_command(size=SIZE):
    return "python modalities/chest_xray/eval/prepare.py" + ("" if size == SIZE else " --size %d" % size)


def decode(image, size):
    """(pixels or None, [w, h] or None, error or None) for one image; never raises."""
    p = datasets.image_path(image)
    if not os.path.exists(p):
        return None, None, "file missing"
    try:
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
        return _xray_prepare().read_any(p, size)[0], [int(w), int(h)], None
    except Exception as e:  # noqa: BLE001  any decode failure makes the image unreadable, not the build
        return None, None, "%s: %s" % (type(e).__name__, e)


_WORKER = {}


def _open_worker(part, size, keys):
    """Each worker process opens the half-built array once and keeps it."""
    _WORKER.update(arr=np.lib.format.open_memmap(part, mode="r+"), size=size, keys=keys)


def _decode_into(row):
    key = _WORKER["keys"][row]
    pix, shape, err = decode(key, _WORKER["size"])
    if pix is not None:
        _WORKER["arr"][row] = pix
    return row, key, shape, err


def _current(size, keys):
    """The cached json when the cache is complete and matches this image list, else None."""
    npy, js = cache_paths(size)
    if not (os.path.exists(npy) and os.path.exists(js)):
        return None
    try:
        meta = json.load(open(js, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if meta.get("images") == keys and meta.get("size") == size else None


def build(size=SIZE, rebuild=False, jobs=None):
    """Decode every image of index() once into the cache; returns the json dict."""
    t0 = time.time()
    keys = [str(k) for k in datasets.index().image]
    npy, js = cache_paths(size)
    if not rebuild:
        meta = _current(size, keys)
        if meta is not None:
            print("cache already built (%s), %d images, %d unreadable; --rebuild to redo"
                  % (npy, len(keys), len(meta["unreadable"])), flush=True)
            return meta
    os.makedirs(CACHE, exist_ok=True)
    jobs = jobs or min(12, os.cpu_count() or 1)
    n = len(keys)
    part = npy + ".part"
    arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(n, size, size))
    arr[:] = 0
    arr.flush()
    del arr                                 # the workers open it themselves, one handle each
    print("decoding %d images to %dx%d in %d processes" % (n, size, size, jobs), flush=True)
    shapes, bad, errors = {}, set(), {}
    done = 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=_open_worker, initargs=(part, size, keys)) as pool:
        for row, key, shape, err in pool.map(_decode_into, range(n), chunksize=16):
            if err is not None:
                bad.add(key)
                errors[key] = err
            else:
                shapes[key] = shape
            done += 1
            if done % PROGRESS == 0 or done == n:
                el = time.time() - t0
                print("  %d/%d  %.0f s  %.1f images/s  %d unreadable"
                      % (done, n, el, done / max(el, 1e-9), len(bad)), flush=True)
    arr = np.lib.format.open_memmap(part, mode="r+")
    arr.flush()
    del arr
    os.replace(part, npy)
    meta = {"size": size, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "images": keys,
            "unreadable": [k for k in keys if k in bad],
            "source_shape": {k: shapes[k] for k in keys if k in shapes}}
    tmp = js + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, js)
    for k in meta["unreadable"][:10]:
        print("  unreadable %s: %s" % (k, errors[k]), flush=True)
    print("%s  N=%d  unreadable=%d  wall %.1f s" % (npy, n, len(bad), time.time() - t0), flush=True)
    return meta


def load(size=SIZE):
    """(read-only FileRows of shape (N, size, size), {image: row}, size).

    Stops with the build command when the cache is missing or no longer matches the index."""
    npy, js = cache_paths(size)
    if not (os.path.exists(npy) and os.path.exists(js)):
        raise SystemExit("no image cache at %s; build it with: %s" % (npy, build_command(size)))
    keys = [str(k) for k in datasets.index().image]
    if json.load(open(js, encoding="utf-8")).get("images") != keys:
        raise SystemExit("the image cache %s does not match the index; rebuild it with: %s --rebuild"
                         % (npy, build_command(size)))
    arr = FileRows.npy(npy)
    if arr.shape != (len(keys), size, size):
        raise SystemExit("the image cache %s has shape %s, expected %s; rebuild it with: %s --rebuild"
                         % (npy, arr.shape, (len(keys), size, size), build_command(size)))
    return arr, {k: i for i, k in enumerate(keys)}, size


def crop_paths(name, size=SIZE):
    """(npy, json) of a crop cache built by tools/chest_lung_crops.py: nih_<name>_<size>."""
    base = os.path.join(CACHE, "nih_%s_%d" % (name, size))
    return base + ".npy", base + ".json"


def load_crop(name, size=SIZE):
    """(read-only FileRows of shape (N, size, size), {image: row}, size, json dict) of a crop cache.

    Its rows are nih_384's rows, so a row number from load() finds the same picture here. Stops with the
    build command when the cache is missing or no longer matches the index."""
    npy, js = crop_paths(name, size)
    cmd = "python tools/chest_lung_crops.py"
    if not (os.path.exists(npy) and os.path.exists(js)):
        raise SystemExit("no crop cache at %s; build it with: %s" % (npy, cmd))
    keys = [str(k) for k in datasets.index().image]
    meta = json.load(open(js, encoding="utf-8"))
    if meta.get("images") != keys:
        raise SystemExit("the crop cache %s does not match the index; rebuild it with: %s --rebuild" % (npy, cmd))
    arr = FileRows.npy(npy)
    if arr.shape != (len(keys), size, size):
        raise SystemExit("the crop cache %s has shape %s, expected %s; rebuild it with: %s --rebuild"
                         % (npy, arr.shape, (len(keys), size, size), cmd))
    return arr, {k: i for i, k in enumerate(keys)}, size, meta


def unreadable(size=SIZE):
    """The image names the build could not decode (missing file or failed decode)."""
    npy, js = cache_paths(size)
    if not os.path.exists(js):
        raise SystemExit("no image cache at %s; build it with: %s" % (npy, build_command(size)))
    return set(json.load(open(js, encoding="utf-8"))["unreadable"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode each chest X-ray once into a memory-mapped cache.")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--rebuild", action="store_true", help="decode again even when the cache matches")
    a = ap.parse_args(argv)
    if a.size < 8:
        raise SystemExit("--size must be at least 8")
    build(size=a.size, rebuild=a.rebuild, jobs=max(1, a.jobs))


if __name__ == "__main__":
    main()
