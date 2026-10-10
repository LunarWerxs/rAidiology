"""What one reading actually costs, measured, so a price can be worked out instead of guessed.

The route rAidiology ships is the visitor's own machine, which costs us nothing. The question behind
this script is the other one: *if* a paid server were ever offered for people with no computer to
run it on, what would a single reading cost to serve? That number has to come from a stopwatch, not
from a feeling, or the price is either a loss on every reading or so padded nobody buys it.

So this times the parts separately, because they scale differently and are billed differently:

- **cold start** - importing torch and loading the five fold weights. Paid once per worker process,
  not per reading, on anything that keeps a process warm. It is the whole bill on a
  scale-to-zero host, which is why it is measured on its own.
- **per image** - the ensemble forward pass, at several thread counts and on the GPU, because a
  host is sold by core and the answer decides how many cores are worth paying for.
- **reading the file** - pulling the pixels out of a DICOM and resizing. Pure CPU, and on a small
  machine it is not the rounding error people assume.
- **memory** - peak resident size, because hosting is sold by the gigabyte as well as by the core.

    python tools/bench_reading.py                 [--images 8] [--threads 1,2,4,8] [--no-gpu]

Every run goes into the ledger, so "did the reading get cheaper" is a query like everything else.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "modalities", "xray", "eval"))
from core import ledger  # noqa: E402


def peak_memory_mb():
    """Peak resident size. -1 when this machine will not say, because a 0 would read as free."""
    try:
        import psutil  # noqa: PLC0415
        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 1e6
    except Exception:  # noqa: BLE001
        return -1.0


def real_images(name, n):
    """Real radiographs rather than the made-up series, because decoding is part of the bill.

    A hospital's PNG is a thousand pixels across and the made-up one is 192, so timing the decode on
    the made-up one would quietly under-price the thing a paying server would actually do."""
    sys.path.insert(0, os.path.join(ROOT, "modalities", "xray", "eval"))
    import datasets  # noqa: PLC0415
    table = datasets.index(name)
    column = "image" if "image" in table.columns else table.columns[0]
    return [datasets.image_path(name, v) for v in table[column].head(n)]


def make_scan(n):
    import demo_scan  # noqa: PLC0415
    folder = tempfile.mkdtemp(prefix="raidiology-bench-")
    paths = demo_scan.write(folder)
    while len(paths) < n:                  # a longer series, by reusing the same three pictures
        src = paths[len(paths) % 3]
        dst = os.path.join(folder, "copy%d.dcm" % len(paths))
        shutil.copy2(src, dst)
        paths.append(dst)
    return folder, paths[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=8, help="images per timed reading")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--device", default="cpu", help="cpu or cuda; one device per run, because the "
                                                    "weights are loaded onto whichever is set when "
                                                    "the process starts")
    ap.add_argument("--real", help="use this many real radiographs from a dataset, e.g. grazpedwri")
    a = ap.parse_args()
    folder, paths = (None, real_images(a.real, a.images)) if a.real else make_scan(a.images)
    thread_counts = [int(t) for t in a.threads.split(",") if t.strip()]

    t0 = time.time()
    os.environ["RAIDIOLOGY_DEVICE"] = a.device
    import torch  # noqa: PLC0415
    import detector  # noqa: PLC0415
    import prepare  # noqa: PLC0415
    import_s = time.time() - t0

    if a.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("This Python has %s, which cannot use a graphics card, so a %s measurement "
                         "here would be a guess. Run it where torch was built with CUDA, or measure "
                         "the CPU, which is the only thing the X-ray route needs."
                         % (torch.__version__, a.device))

    t0 = time.time()
    detector.load_folds(1, 2)
    load_s = time.time() - t0

    t0 = time.time()
    for p in paths:
        prepare.read_any(p, prepare.SIZE)
    decode_s = (time.time() - t0) / len(paths)

    rows = []
    for n in (thread_counts if a.device == "cpu" else [0]):
        if a.device == "cpu":
            torch.set_num_threads(n)
        detector.score_images(paths[:2], 1)                      # warm the allocator and the caches
        if a.device != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        detector.score_images(paths, 1)
        if a.device != "cpu":
            torch.cuda.synchronize()
        label = ("cpu x%d" % n) if a.device == "cpu" else a.device
        rows.append((label, (time.time() - t0) / len(paths)))

    peak = peak_memory_mb()
    if folder:
        shutil.rmtree(folder, ignore_errors=True)

    print("one reading, %d %s image(s)" % (a.images, a.real or "made-up"))
    print("  %-10s %9.2f s   paid once per worker, not per reading" % ("import", import_s))
    print("  %-10s %9.2f s   loading five fold weights" % ("model load", load_s))
    print("  %-10s %9.3f s   per image, pulling pixels out of DICOM" % ("decode", decode_s))
    for name, per in rows:
        print("  %-10s %9.3f s   per image, the five-model ensemble" % (name, per))
    print("  %-10s %9.0f MB  peak resident" % ("memory", peak))
    best = min(rows, key=lambda r: r[1]) if rows else ("none", 0.0)
    whole = (decode_s + best[1]) * a.images
    print("\n  a warm %d-image reading on %s: %.2f s of compute" % (a.images, best[0], whole))
    print("  a cold one (scale-to-zero): %.2f s" % (import_s + load_s + whole))

    with ledger.run("cost", "one x-ray reading on %s" % a.device) as r:
        r.settings_from({"images": a.images, "threads": a.threads, "images_from": a.real or "made-up"})
        r.number("import_s", import_s, "s").number("model_load_s", load_s, "s")
        r.number("decode_s_per_image", decode_s, "s").number("peak_memory_mb", peak, "MB")
        for name, per in rows:
            r.number("per_image_%s" % name.replace(" ", "_"), per, "s")
        r.number("warm_reading_s", whole, "s").number("cold_reading_s", import_s + load_s + whole, "s")
        r.note_is("%d images; fastest %s" % (a.images, best[0]))


if __name__ == "__main__":
    main()
