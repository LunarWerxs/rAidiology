"""Find pictures one X-ray collection shares with another, from their 384 px caches.

A collection added for training must not carry pictures of another collection's test split: two
collections from one country's hospitals (OrthoFrac-XR and FracAtlas are both from Bangladesh) can
hold the same radiograph, and training on it would score the reader on pictures it had seen. Each
cached picture is shrunk to 24 x 24 (the mean of each 16 x 16 block), centred and scaled, and every
pair across the two collections is compared by correlation, as is each picture mirrored left to
right. A pair at or above --at is the same radiograph for this purpose. Prints the count, the pairs
and which split each side is in; with --out, writes the pairs as CSV for the trainer to drop.

A collection may be named among the others too, to find pictures it holds twice (each pair is then
listed from both sides). A collection whose split is not locked yet shows "unsplit".

Usage: python tools/xray_duplicates.py orthofrac fracatlas [grazpedwri ...] [--at 0.97] [--out pairs.csv]"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "modalities", "xray", "eval"))
import datasets  # noqa: E402
import prepare  # noqa: E402
import split as splitmod  # noqa: E402

BLOCK = 16


def thumbs(dataset):
    """(keys, unit vectors of every readable picture and of its mirror)."""
    arr, order, size = prepare.load(dataset)
    bad = prepare.unreadable(dataset)
    keys = [k for k in order if k not in bad]
    rows = np.array([order[k] for k in keys])
    n = size // BLOCK
    out = np.empty((len(rows), n * n), np.float32)
    for i in range(0, len(rows), 512):
        part = np.asarray(arr[rows[i:i + 512]], dtype=np.float32)
        out[i:i + 512] = part.reshape(len(part), n, BLOCK, n, BLOCK).mean(axis=(2, 4)).reshape(len(part), -1)

    def unit(v):
        v = v - v.mean(axis=1, keepdims=True)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)

    mirror = out.reshape(len(out), n, n)[:, :, ::-1].reshape(len(out), -1)
    return keys, unit(out), unit(mirror)


def where(dataset, s):
    if dataset not in s["datasets"]:        # checked before its split is locked: say so
        return lambda k: "unsplit"
    a = splitmod.assignment(dataset, s)
    df = datasets.index(dataset)
    unit = dict(zip(df.image, df.patient))
    return lambda k: "test" if a.get(unit.get(k)) == "test" else "dev %s" % a.get(unit.get(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--at", type=float, default=0.97)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = splitmod.load()
    kn, vn, _ = thumbs(a.new)
    side_new = where(a.new, s)
    pairs = []
    for o in a.others:
        ko, vo, mo = thumbs(o)
        side_o = where(o, s)
        best = np.full(len(kn), -1.0, np.float32)
        for i in range(0, len(ko), 4096):
            c = np.maximum(vn @ vo[i:i + 4096].T, vn @ mo[i:i + 4096].T)
            if o == a.new:                  # a collection against itself: not each picture with itself
                d = np.arange(i, min(i + 4096, len(ko)))
                c[d, d - i] = -1.0
            best = np.maximum(best, c.max(axis=1))
            for r, j in zip(*np.nonzero(c >= a.at)):
                pairs.append((a.new, kn[r], side_new(kn[r]), o, ko[i + j], side_o(ko[i + j]), float(c[r, j])))
        q = np.quantile(best, [0.5, 0.9, 0.99, 1.0])
        print("%s against %s: %d pictures against %d; best match per picture: median %.3f, 90%% %.3f, "
              "99%% %.3f, max %.3f; pairs at %.2f or above: %d"
              % (a.new, o, len(kn), len(ko), *q, a.at, sum(1 for p in pairs if p[3] == o)), flush=True)
    for p in sorted(pairs, key=lambda p: -p[-1])[:40]:
        print("  %s %s (%s)  =  %s %s (%s)  r %.4f" % p)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("dataset,image,split,other_dataset,other_image,other_split,r\n")
            for p in pairs:
                f.write("%s,%s,%s,%s,%s,%s,%.5f\n" % p)
        print(a.out)


if __name__ == "__main__":
    main()
