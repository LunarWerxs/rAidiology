"""Find which NIH ChestX-ray14 picture each ChestX-Det picture is.

ChestX-Det (LICENSING.md; Apache-2.0 labels on NIH pictures) gives boxes by three board-certified
radiologists for 13 findings on 3,578 NIH pictures, but renames them (`36199.png`) and lists no NIH
name. To use its labels on the pictures we hold, and to keep NIH's test pictures out of anything
trained on them, each one is matched back by picture: shrunk to 384 px by the same decoder as the
chest cache, then to 24 x 24 (the mean of each 16 x 16 block, as tools/xray_duplicates.py does),
centred and scaled, and compared by correlation with every NIH picture in the 384 px chest cache.
The best match and the second best are kept: the same radiograph correlates at about 0.99 or more,
and a clear gap to the second says it is not a lookalike.

NIH's 24 x 24 thumbnails are cached at <WORK>/eval/chest/nih_thumbs24.npy (computed once from the
16.5 GB cache, then reused). The pictures are unpacked from `train_data.zip` into
<Datasets>/public/chestx-det/train/ on the first run.

Writes <Datasets>/public/chestx-det/nih_match.csv: file_name, part (train|test, ChestX-Det's own
split), image (the NIH name), r, r_second, patient, nih_split.

Usage: python tools/chestxdet_match.py [--jobs 6]"""
import argparse
import concurrent.futures
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
sys.path.insert(0, ROOT)
import datasets  # noqa: E402
import prepare  # noqa: E402
from core import paths  # noqa: E402

SIZE, BLOCK = 384, 16
CXD = os.path.join(paths.DATASETS, "public", "chestx-det")
NIH_THUMBS = os.path.join(paths.WORK, "eval", "chest", "nih_thumbs24.npy")


def unit(v):
    v = v - v.mean(axis=1, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def shrink(px):
    n = SIZE // BLOCK
    return px.reshape(-1, n, BLOCK, n, BLOCK).mean(axis=(2, 4)).reshape(len(px), -1)


def nih_thumbs():
    arr, order, size = prepare.load(SIZE)
    keys = sorted(order, key=order.get)
    if os.path.exists(NIH_THUMBS):
        t = np.load(NIH_THUMBS)
        if len(t) == len(keys):
            return keys, t
    t = np.empty((len(keys), (SIZE // BLOCK) ** 2), np.float32)
    for i in range(0, len(keys), 2048):
        t[i:i + 2048] = shrink(np.asarray(arr[i:i + 2048], dtype=np.float32))
    os.makedirs(os.path.dirname(NIH_THUMBS), exist_ok=True)
    np.save(NIH_THUMBS + ".part.npy", t)
    os.replace(NIH_THUMBS + ".part.npy", NIH_THUMBS)
    return keys, t


def _one(path):
    return shrink(np.asarray(prepare._xray_prepare().read_any(path, SIZE)[0], dtype=np.float32)[None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    folder = os.path.join(CXD, "train")
    if not os.path.isdir(folder):
        with zipfile.ZipFile(os.path.join(CXD, "train_data.zip")) as z:
            z.extractall(folder + ".part")
        os.replace(folder + ".part", folder)
    files = {}
    for dirpath, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(".png"):
                files[n] = os.path.join(dirpath, n)
    parts = {}
    for part in ("train", "test"):
        for r in json.load(open(os.path.join(CXD, "ChestX_Det_%s.json" % part), encoding="utf-8")):
            parts[r["file_name"]] = part
    names = sorted(n for n in files if n in parts)
    print("ChestX-Det: %d pictures unpacked, %d of them in its label files (%d labelled pictures in all)"
          % (len(files), len(names), len(parts)), flush=True)

    with concurrent.futures.ProcessPoolExecutor(a.jobs) as ex:
        mine = unit(np.stack(list(ex.map(_one, [files[n] for n in names], chunksize=16))))
    keys, nih = nih_thumbs()
    nih = unit(nih)
    best = np.full(len(names), -1, int)
    r1 = np.full(len(names), -2.0)
    r2 = np.full(len(names), -2.0)
    for i in range(0, len(nih), 16384):
        s = mine @ nih[i:i + 16384].T                        # (pictures, chunk)
        top2 = np.argpartition(-s, 1, axis=1)[:, :2]
        for j in range(len(names)):
            for c in top2[j]:
                v = s[j, c]
                if v > r1[j]:
                    r2[j], r1[j], best[j] = r1[j], v, i + c
                elif v > r2[j]:
                    r2[j] = v
    idx = datasets.index().set_index("image")
    out = pd.DataFrame({"file_name": names, "part": [parts[n] for n in names],
                        "image": [keys[b] for b in best], "r": r1.round(5), "r_second": r2.round(5)})
    out["patient"] = out.image.map(idx.patient)
    out["nih_split"] = out.image.map(idx.nih_split)
    dst = os.path.join(CXD, "nih_match.csv")
    out.to_csv(dst + ".part", index=False)
    os.replace(dst + ".part", dst)
    q = out.r.quantile([0.0, 0.01, 0.5]).round(4).tolist()
    print("matched %d: r min %.4f, 1st percentile %.4f, median %.4f; %d at r 0.97 or more; second best's "
          "median %.4f; %d NIH pictures matched twice"
          % (len(out), q[0], q[1], q[2], int((out.r >= 0.97).sum()), out.r_second.median(),
             int(out.image.duplicated().sum())), flush=True)
    print(out.groupby(["part", "nih_split"]).size().to_string(), flush=True)
    print("wrote %s" % dst, flush=True)


if __name__ == "__main__":
    main()
