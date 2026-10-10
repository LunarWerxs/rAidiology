"""Read the training pictures of a chest fold with trained readers and save the mean of their logits.

  python tools/chest_teacher.py --versions 103 106 --fold 0 --out work/eval/chest/teacher-v192-fold0.npz

Every development picture OUTSIDE the fold (the training and early-stop pictures; the held-out fold is never
read) goes through each version's fold-<k> net with that net's own mean, sd and settings; the logits are
averaged over the versions. The file feeds `reader.py train --set distill=<file>` (idea 11, docs/efficiency).
Saved: image (names as reader.table() gives them), logits (N x 15 float32), versions, fold.
RAIDIOLOGY_SOUP_CARD_GB (default 1.5) is the card share leased for the reading.
"""
import argparse
import json
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200; read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
import reader  # noqa: E402  (also binds core, datasets, prepare, split as reader.py does)
from core import card, ledger  # noqa: E402

datasets = reader.datasets
xd = reader.xd


def load_fold(v, fold):
    """(checkpoint, that fold's info) of a trained version; refuses a version without the fold."""
    d = reader.out_dir(v)
    pt, mp = os.path.join(d, "fold%d.pt" % fold), os.path.join(d, "meta.json")
    for p in (pt, mp):
        if not os.path.exists(p):
            raise SystemExit("v%d: %s is missing" % (v, p))
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    infos = [i for i in meta.get("folds", []) if i.get("fold") == fold]
    if not infos:
        raise SystemExit("v%d: meta.json has no fold %d entry" % (v, fold))
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    if "crop" in (ck.get("settings") or {}):
        raise SystemExit("v%d was trained on a crop cache; reader.table() would show it the wrong pictures" % v)
    return ck, infos[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", type=int, nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=reader.CFG["batch"])
    a = ap.parse_args()
    if len(set(a.versions)) != len(a.versions):
        raise SystemExit("--versions needs different versions")
    if os.path.exists(a.out):
        raise SystemExit("%s already exists; pick another --out" % a.out)
    # leased before the checkpoints and data are loaded, as reader.train does
    card.require_card(reader.DEV, torch)
    xd.limit_card_memory()
    loaded = [load_fold(v, a.fold) for v in a.versions]
    # 2026-10-04: a version trained with cache_size reads that cache (idea 17); one table serves one size
    cache_sizes = {v: reader.cache_size_of(ck.get("settings")) for v, (ck, _info) in zip(a.versions, loaded)}
    if len(set(cache_sizes.values())) != 1:
        raise SystemExit("the versions read different picture caches: %s" % cache_sizes)
    df, pix, cache_row, labels = reader.table(cache_sizes[a.versions[0]])
    rows = np.flatnonzero(df.fold.to_numpy() != a.fold)
    y = labels[rows].astype(int)
    n_lab = len(datasets.LABELS)
    sets = []
    with ledger.run("chest_teacher", "teacher of %s fold %d" % ("+".join(str(v) for v in a.versions), a.fold)) as led:
        led.settings_from({"versions": a.versions, "fold": a.fold, "batch": a.batch})
        for v, (ck, info) in zip(a.versions, loaded):
            settings = ck.get("settings") or {}
            cfg = ck["cfg"]
            net = reader.ChestNet(reader.widths_of(settings, cfg), cfg["dropout"], cfg["stem_stride"], n_lab,
                                  renderings=int(settings.get("renderings", 1))).to(reader.DEV)
            net.load_state_dict(ck["state"])
            mean = torch.tensor(info["mean"], dtype=torch.float32, device=reader.DEV)
            sd = torch.tensor(info["sd"], dtype=torch.float32, device=reader.DEV)
            t0 = time.time()
            z = reader.held_out_logits(net, pix, cache_row, labels, rows, mean, sd, a.batch, settings)
            took = time.time() - t0
            per = [xd.auroc(z[:, i], y[:, i]) for i in range(n_lab)]
            m14 = float(np.nanmean(per[:14]))
            print("v%d: read %d pictures in %.1f s; mean AUROC of 14 %.4f on its own training pictures, not a score"
                  % (v, len(rows), took, m14), flush=True)
            led.number("read.seconds.v%d" % v, took, "s").number("auroc.mean14.train.v%d" % v, m14)
            sets.append(z.astype(np.float64))
            del net
    logit = np.mean(sets, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, image=df.image.to_numpy(str)[rows], logits=logit, versions=np.array(a.versions),
             fold=np.array(a.fold))
    print("teacher: %d pictures x %d logits, mean of v%s -> %s"
          % (len(rows), n_lab, "+v".join(str(v) for v in a.versions), a.out))


if __name__ == "__main__":
    main()
