"""Decode every radiograph of a dataset once, into one cached form every later step reads.

For each dataset this writes, under preds.CACHE:
  <dataset>_<size>.npy    a uint8 array of shape (N, size, size), opened later as a memory map, so
                          a baseline or a network reads pixels without decoding a PNG or JPEG again
  <dataset>_<size>.json   {"dataset", "size", "built", "images", "unreadable", "source_shape"}:
                          the image keys in row order, the keys that could not be decoded, and each
                          image's original [width, height]

Row order is datasets.index(dataset).image order, so a row index is stable and every later step
can find an image's pixels from its key alone (load() hands back that mapping).

Why each choice:
- One intensity rule for both datasets. GRAZPEDWRI-DX is 16-bit greyscale PNG and FracAtlas is
  8-bit greyscale JPEG; both are read as float and windowed to the 0.5th and 99.5th percentiles of
  the pixels above the image's own 1st percentile. Ignoring the darkest pixels drops the black
  border a radiograph carries, which would otherwise drag the window down and wash the bone out.
  The same rule for both means one detector can be trained on one set and shown the other.
- Letterbox, never stretch. The longer side is scaled to `size` with Pillow's LANCZOS filter and
  the result is centred on a zero-filled square canvas. A fracture is a shape, so the aspect ratio
  must survive. A trial may ask for a tall canvas instead, `--size 640x384` (height x width): the
  picture is scaled as large as fits and centred the same way. Children's wrist pictures are 99%
  portrait (height 1.8 times the width), so on a square they fill 56% of it and on 640 x 384 86%.
- An image whose file is missing, or whose decode raises, is listed under "unreadable" and its row
  stays zeros. Nothing crashes the build; the method writers mark those rows with
  preds.UNREADABLE.
- Pixels only. This step reads no labels and no split, so it opens nothing the test guard
  (datasets.guard_test) protects; the guard is applied where a split is chosen.
- Resumable. A cache whose image list matches the index is kept unless --rebuild is given. The
  array is written to a temporary name and renamed only when complete, and the json is written
  last, so a run cut short never leaves a cache that looks finished.
- Parallel, in processes rather than threads. Measured on this machine (2026-10-01): one image
  costs about 16 ms of pure work (6.4 ms to read, 6.0 ms to window, 4.0 ms to letterbox), yet eight
  threads managed 57 images a second, which is what ONE thread does. The interpreter lock is held
  for most of it, so the thread pool bought nothing. Worker processes open the same memory-mapped
  array and write their own rows, so the pixels never travel between processes.
- No sorting. An 8- or 16-bit picture's percentiles are read off its histogram, with numpy's own
  interpolation, and the window is a table of its values (`scaled`): decoding 2.00x faster in five
  of five A/B pairs (2026-10-02), every one of the 24,351 cached pictures byte for byte the same.

Usage: python prepare.py all|grazpedwri|fracatlas|orthofrac [--size 384] [--rebuild] [--jobs 8]"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402,F401
from core import ledger  # noqa: E402
from core.rows import FileRows  # noqa: E402
import datasets  # noqa: E402
import preds  # noqa: E402

SIZE = 384
LOW_CUT = 1.0                # percentile below which pixels count as border and are ignored
WINDOW = (0.5, 99.5)         # percentiles of the remaining pixels that set black and white
PROGRESS = 500               # images between progress lines
WIDE_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I", "F")
TABLE_SPAN = 1 << 20         # an integer picture spanning more values than this is windowed by sorting
FLOAT32_EXACT = 1 << 24      # past this, float32 rounds integers, and the old route read them as float32


def canvas(size):
    """(height, width) of a cache: `size` is a side (a square, the default) or "HxW"."""
    if isinstance(size, str) and "x" in size:
        h, w = size.split("x")
        return int(h), int(w)
    return int(size), int(size)


def size_arg(text):
    """--size: a side, or height x width."""
    if "x" not in text:
        return int(text)
    h, w = canvas(text)
    return "%dx%d" % (h, w)


def cache_paths(dataset, size=SIZE):
    base = os.path.join(preds.CACHE, "%s_%s" % (dataset, size))
    return base + ".npy", base + ".json"


def build_command(dataset, size=SIZE):
    extra = "" if size == SIZE else " --size %s" % size
    return "python modalities/xray/eval/prepare.py %s%s" % (dataset, extra)


def read_grey(p):
    """The image as a 2-D array of its stored values, whatever its bit depth: integers as stored
    (uint8, uint16, int32), or float32 for a floating-point image."""
    with Image.open(p) as im:
        im.load()
        if im.mode in WIDE_MODES:
            arr = np.asarray(im)
        else:
            arr = np.asarray(im.convert("L"))
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("not a 2-D greyscale image: shape %s" % (arr.shape,))
    return arr


def window(arr):
    """Window to the 0.5th..99.5th percentiles of the pixels above the 1st percentile, to 0..1."""
    floor = np.percentile(arr, LOW_CUT)
    body = arr[arr > floor]
    if body.size == 0:          # a flat image: nothing above the border level
        body = arr.ravel()
    lo, hi = np.percentile(body, WINDOW)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _counts(arr):
    """(how many pixels hold each value, the value the counts start at) for an integer picture, or
    None when its values span too much for a table or go past 2^24, where float32 (the old route)
    can no longer hold every integer and the table would be more exact than `window`."""
    if arr.dtype == np.uint8:
        return np.asarray(Image.fromarray(np.ascontiguousarray(arr)).histogram()), 0   # 3x np.bincount
    low, high = int(arr.min()), int(arr.max())
    if high - low >= TABLE_SPAN or high > FLOAT32_EXACT or low < -FLOAT32_EXACT:
        return None
    low = low if arr.dtype.kind == "i" else 0
    return np.bincount((arr.astype(np.int64) - low).ravel() if low else arr.ravel()), low


def _percentiles(counts, low, n, q):
    """np.percentile(x, q) for the n values counted in `counts` (from `low`), read off the counts
    instead of sorted. The same arithmetic as numpy 2's default 'linear' method on a float32 array,
    step for step: the index (n - 1) * q / 100 in double, the two order statistics either side as
    float32, and numpy's two-sided lerp. A scalar q gives float32, as numpy does; a tuple gives double."""
    at = np.asanyarray((n - 1) * np.true_divide(q, 100, out=...))
    below = np.clip(np.floor(at), 0, n - 1).astype(np.intp)
    above = np.clip(np.floor(at) + 1, 0, n - 1).astype(np.intp)
    upto = np.cumsum(counts)
    a = (np.searchsorted(upto, below, side="right") + low).astype(np.float32)
    b = (np.searchsorted(upto, above, side="right") + low).astype(np.float32)
    t = np.asanyarray(at - below, dtype=at.dtype)
    if np.ndim(q) == 0:
        a, b, t = a[()], b[()], float(t)
    gap = b - a
    out = np.add(a, gap * t, out=...)
    np.subtract(b, gap * (1 - t), out=out, where=t >= 0.5, casting="unsafe", dtype=type(out.dtype))
    return out[()] if out.ndim == 0 else out


def scaled(arr):
    """`window(arr) * 255` as float32, what letterbox resamples. An integer picture is windowed from
    its histogram and mapped through a table of its few thousand values, which gives the same bits as
    `window` (checked on every training image of both sets, 2026-10-02) without sorting the pixels
    twice or doing arithmetic per pixel: 2.3x on a 3.8-megapixel JPEG, 1.9x on a 16-bit PNG."""
    table = _counts(arr) if arr.dtype.kind in "ui" else None
    if table is None:
        return (window(np.asarray(arr, dtype=np.float32)) * 255.0).astype(np.float32)
    counts, low = table
    floor = _percentiles(counts, low, arr.size, LOW_CUT)
    first = max(int(np.floor(floor)) + 1 - low, 0)   # the first value above the border level
    body = int(counts[first:].sum())
    lo, hi = (_percentiles(counts[first:], low + first, body, WINDOW) if body
              else _percentiles(counts, low, arr.size, WINDOW))
    if hi <= lo:
        hi = lo + 1.0
    values = np.arange(low, low + len(counts))
    lut = (np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.float32)
    return lut[arr.astype(np.int64) - low] if low else lut[arr]


def placement(w, h, size):
    """(width, height, top, left) of a w x h picture once `letterbox` has scaled and centred it on
    the canvas `size`. The fracture boxes (boxes.py) go through this same function, so a box lands
    where the picture shows it."""
    ch, cw = canvas(size)
    scale = ch / float(max(w, h)) if ch == cw else min(ch / float(h), cw / float(w))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return nw, nh, (ch - nh) // 2, (cw - nw) // 2


def letterbox(pic, size):
    """Scale the longer side of a `scaled` picture to `size` (LANCZOS) and centre it on a
    zero-filled size x size canvas; on an "HxW" canvas, as large as fits, centred."""
    h, w = pic.shape
    ch, cw = canvas(size)
    nw, nh, top, left = placement(w, h, size)
    img = Image.fromarray(pic, mode="F")
    small = np.asarray(img.resize((nw, nh), Image.Resampling.LANCZOS), dtype=np.float32)
    out = np.zeros((ch, cw), np.uint8)
    out[top:top + nh, left:left + nw] = np.clip(np.rint(small), 0, 255).astype(np.uint8)
    return out



def read_any(path, size):
    """One radiograph from a DICOM file or an ordinary image, windowed and letterboxed exactly as
    prepare.py did for every image the detector was trained and scored on. A visitor's scan has to
    go through the same door as the scorecard's, or the number on the scorecard is not the number
    they are getting.

    A file that does not open the way DICOM does (a 128-byte preamble, then "DICM") is tried as a
    picture first. Forcing pydicom through a JPEG to find that out took 135 ms of a 136 ms reading
    on a FracAtlas picture and 22 of 30 on a GRAZPEDWRI-DX one (2026-10-02), and it can only fail. A
    DICOM file written without the preamble, as old equipment does, is not a picture either, so it
    still gets the forced read."""
    from pydicom.misc import is_dicom
    if not is_dicom(path):
        try:
            return _read_picture(path, size)
        except OSError:                 # Pillow's "cannot identify image file" is an OSError
            pass
    return _read_dicom(path, size) or _read_picture(path, size)


def _read_dicom(path, size):
    """(pixels, "DICOM"), or None when pydicom cannot make a radiograph of the file."""
    try:
        import pydicom
        d = pydicom.dcmread(path, force=True)
        if not getattr(d, "Rows", None):
            return None
        arr = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1) or 1)
        intercept = float(getattr(d, "RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept
        if str(getattr(d, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() + arr.min() - arr
        if arr.ndim == 3:
            arr = arr.mean(axis=-1) if arr.shape[-1] in (3, 4) else arr[arr.shape[0] // 2]
        return letterbox(scaled(arr), size), "DICOM"
    except Exception:  # noqa: BLE001  arkitect-allow: test-floor-loosened - as before: a file pydicom cannot read is tried as a picture
        return None


def _read_picture(path, size):
    """(pixels, "image file"): the same door as the cache (`decode`), so a 16-bit picture keeps its
    stored values instead of being clipped to 8 bits by convert("L") (all black, 2026-10-10)."""
    arr = read_grey(path)
    return letterbox(scaled(arr), size), "image file"


def read_many(paths):
    """Decode a handful of radiographs at once.

    Measured 2026-10-01 over twelve held-out images: 0.664 s one at a time against 0.348 s on four
    threads, 1.91x. Pillow lets go of the interpreter lock while it decodes, so threads are enough
    and there is no process pool to pay for. It is worth having because turning a hospital's
    thousand-pixel PNG into an array is the same order of work as the five-model ensemble itself,
    and on a graphics card it is the larger half."""
    if len(paths) < 2:
        return [read_any(p, SIZE) for p in paths]
    with ThreadPoolExecutor(min(4, len(paths))) as pool:
        return list(pool.map(lambda p: read_any(p, SIZE), paths))

def decode(dataset, key, size):
    """(pixels or None, [w, h] or None, error or None) for one image; never raises."""
    p = datasets.image_path(dataset, key)
    if p is None:
        return None, None, "file missing"
    try:
        arr = read_grey(p)
        h, w = arr.shape
        return letterbox(scaled(arr), size), [int(w), int(h)], None
    except Exception as e:  # noqa: BLE001  any decode failure makes the image unreadable, not the build
        return None, None, "%s: %s" % (type(e).__name__, e)


_WORKER = {}


def _open_worker(part, dataset, size, n):
    """Each worker process opens the half-built array once and keeps it."""
    _WORKER["arr"] = np.lib.format.open_memmap(part, mode="r+")
    _WORKER["dataset"] = dataset
    _WORKER["size"] = size


def _decode_into(job):
    """(row, key) -> (row, key, [w, h] or None, error or None), with the pixels written in place."""
    row, key = job
    pix, shape, err = decode(_WORKER["dataset"], key, _WORKER["size"])
    if pix is not None:
        _WORKER["arr"][row] = pix
    return row, key, shape, err


def _current(dataset, size, keys):
    """The cached json when the cache is complete and matches this image list, else None."""
    npy, js = cache_paths(dataset, size)
    if not (os.path.exists(npy) and os.path.exists(js)):
        return None
    try:
        meta = json.load(open(js, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("images") != keys or str(meta.get("size", -1)) != str(size):
        return None
    return meta


def build(dataset, size=SIZE, rebuild=False, jobs=None):
    """Decode every image of `dataset` once into the cache; returns the json dict."""
    if dataset not in datasets.SETS:
        raise SystemExit("unknown dataset %r; one of %s" % (dataset, datasets.ALL))
    t0 = time.time()
    keys = [str(k) for k in datasets.index(dataset).image]
    npy, js = cache_paths(dataset, size)
    if not rebuild:
        meta = _current(dataset, size, keys)
        if meta is not None:
            print("%s: cache already built (%s), %d images, %d unreadable; --rebuild to redo"
                  % (dataset, npy, len(keys), len(meta["unreadable"])), flush=True)
            return meta
    os.makedirs(preds.CACHE, exist_ok=True)
    jobs = jobs or min(8, os.cpu_count() or 1)
    n = len(keys)
    part = npy + ".part"
    arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(n, *canvas(size)))
    arr[:] = 0
    shapes, bad, errors = {}, set(), {}
    arr.flush()
    del arr                                 # the workers open it themselves, one handle each
    print("%s: decoding %d images to %dx%d in %d processes" % (dataset, n, *canvas(size), jobs), flush=True)
    done = 0
    jobs_list = list(enumerate(keys))
    with ProcessPoolExecutor(max_workers=jobs, initializer=_open_worker,
                             initargs=(part, dataset, size, n)) as pool:
        for row, key, shape, err in pool.map(_decode_into, jobs_list, chunksize=16):
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
    meta = {"dataset": dataset, "size": size, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images": keys, "unreadable": [k for k in keys if k in bad],
            "source_shape": {k: shapes[k] for k in keys if k in shapes}}
    tmp = js + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, js)
    for k in meta["unreadable"][:10]:
        print("  unreadable %s: %s" % (k, errors[k]), flush=True)
    print("%s: %s  N=%d  unreadable=%d  wall %.1f s"
          % (dataset, npy, n, len(bad), time.time() - t0), flush=True)
    return meta


def load(dataset, size=SIZE):
    """(read-only FileRows of shape (N, height, width), {image key: row index}, size).

    Stops with the build command when the cache is missing or no longer matches the index."""
    npy, js = cache_paths(dataset, size)
    if not (os.path.exists(npy) and os.path.exists(js)):
        raise SystemExit("%s: no image cache at %s; build it with: %s"
                         % (dataset, npy, build_command(dataset, size)))
    meta = json.load(open(js, encoding="utf-8"))
    keys = [str(k) for k in datasets.index(dataset).image]
    if meta.get("images") != keys:
        raise SystemExit("%s: the image cache %s does not match the dataset's index; rebuild it with: %s --rebuild"
                         % (dataset, npy, build_command(dataset, size)))
    arr = FileRows.npy(npy)
    if arr.shape != (len(keys), *canvas(size)):
        raise SystemExit("%s: the image cache %s has shape %s, expected %s; rebuild it with: %s --rebuild"
                         % (dataset, npy, arr.shape, (len(keys), *canvas(size)), build_command(dataset, size)))
    return arr, {k: i for i, k in enumerate(keys)}, size


def unreadable(dataset, size=SIZE):
    """The image keys the build could not decode (missing file or failed decode)."""
    npy, js = cache_paths(dataset, size)
    if not os.path.exists(js):
        raise SystemExit("%s: no image cache at %s; build it with: %s"
                         % (dataset, npy, build_command(dataset, size)))
    return set(json.load(open(js, encoding="utf-8"))["unreadable"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode each X-ray dataset once into a memory-mapped cache.")
    ap.add_argument("dataset", choices=["all"] + datasets.ALL)
    ap.add_argument("--size", type=size_arg, default=SIZE, help="a side (square), or HEIGHTxWIDTH")
    ap.add_argument("--rebuild", action="store_true", help="decode again even when the cache matches")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    if min(canvas(a.size)) < 8:
        raise SystemExit("--size must be at least 8")
    t0 = time.time()
    names = datasets.ALL if a.dataset == "all" else [a.dataset]
    for d in names:
        t = time.time()
        with ledger.run("prepare", "%s %spx" % (d, a.size), note="jobs=%d" % a.jobs) as led:
            meta = build(d, size=a.size, rebuild=a.rebuild, jobs=max(1, a.jobs))
            el = max(time.time() - t, 1e-9)
            led.number("images", len(meta["images"]), "count")
            led.number("unreadable", len(meta["unreadable"]), "count")
            led.number("images_per_second", len(meta["images"]) / el, "per second")
        print("%s: cache %s  N=%d  unreadable=%d  stage wall %.1f s"
              % (d, cache_paths(d, a.size)[0], len(meta["images"]), len(meta["unreadable"]), time.time() - t),
              flush=True)
    print("prepare: total wall %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
