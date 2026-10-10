"""Our own heart and lung outlines for NIH ChestX-ray14 pictures, learned from CheXmask's landmarks.

CheXmask has no outline for a picture a person uploads, so a model has to draw them. This one is the chest
reader's trunk (reader.ChestNet's conv layers, its widths, dropout and stem stride, the same 384-pixel
picture memmap and normalisation) with a head that reads the last feature grid flattened (position is
what an outline is; ChestNet's channel average discards it) into a 240-number output: CheXmask's 120 landmarks (x, y) divided by 1024, NIH's pictures
being 1024 square, in CheXmask's order (right lung 44, left lung 50, heart 26). Spec and bar: "Chest: our
own heart and lung outlines" in docs/efficiency/README.md.

Fold k trains on the development pictures outside fold k that CheXmask outlined with Dice RCA mean >= 0.7,
minus an early-stop set of 10% of those patients (split as reader.train_fold splits its own), keeps the
epoch with the lowest mean point error on that set, and draws every picture of fold k, CheXmask-outlined
or not. Loss is smooth L1 on the normalised coordinates (beta 0.01, about 10 pixels of 1024, so a miss
beyond that costs like an absolute error). No augmentation of any kind: a flip swaps the lungs, and
reader's rotate/translate/jitter would move the picture but not the points. The head's bias starts at the
mean outline. bfloat16 autocast as reader does. The held-out test list is never opened.

CheXmask's landmarks are parsed once into <work>/eval/chest/chexmask_landmarks.npz (image names, a float32
N x 240 array divided by 1024, Dice RCA mean) and reused while newer than the CheXmask file.

Writes <work>/eval/chest/outline-v<version>/: fold<k>.pt (state, info, fingerprint; a finished fold is
resumed from it when the fingerprint matches, RAIDIOLOGY_NO_RESUME=1 retrains), meta.json,
landmarks_fold<k>.npz (image names, predicted N x 240 in the 1024 pixel frame) and ctr_fold<k>.csv in the
columns chexmask_ctr.py caches (image, ctr, dice_mean, usable, min_dice) with ctr from
chexmask_ctr.ctr_from_points, dice_mean and min_dice NaN and usable 1. chexmask_ctr.py --ctr-file reads it.

  python tools/chest_outline.py train  --version 1 [--epochs 12] [--batch 64] [--folds 0]
      RAIDIOLOGY_OUTLINE_CARD_GB (default 2.5) is the card share leased for the training.

  python tools/chest_outline.py report --version 1 [--folds 0]
      On the fold's pictures CheXmask outlined at Dice RCA mean >= 0.7: mean point error against CheXmask in
      pixels of 1024 (overall, right lung, left lung, heart) and the median absolute difference between our
      CTR and CheXmask's. Reported, not a gate.

Data: CheXmask (Gaggion et al.), CC BY 4.0. Cite: Gaggion N, Mosquera C, Mansilla L, et al. CheXmask: a
large-scale dataset of anatomical segmentation masks for multi-center chest x-ray images. Scientific Data
11, 511 (2024). https://doi.org/10.1038/s41597-024-03358-1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
import reader  # noqa: E402  (also binds core, datasets, prepare, split as reader.py does)
from core import card, ledger, paths  # noqa: E402
import chexmask_ctr as cc  # noqa: E402  (imported after reader so `datasets` is the chest one)

datasets = reader.datasets
xd = reader.xd
DEV = reader.DEV
SEED = reader.SEED
CFG = reader.CFG
N_POINTS = cc.N_POINTS
FRAME = cc.FRAME
MIN_DICE = 0.7              # CheXmask's README advice, as chexmask_ctr.py
BETA = 0.01                 # smooth L1's switch from squared to absolute error, in normalised units
STRUCTURES = [("right lung", 0, cc.N_RIGHT), ("left lung", cc.N_RIGHT, cc.N_RIGHT + cc.N_LEFT),
              ("heart", cc.N_RIGHT + cc.N_LEFT, N_POINTS)]


def out_dir(version):
    return os.path.join(paths.WORK, "eval", "chest", "outline-v%d" % version)


def targets_for(names):
    """(N x 240 landmarks / 1024 in `names` order, NaN for a picture CheXmask lacks; dice, NaN likewise)."""
    image, lm, dice = cc.load_landmarks()
    at = pd.Series(np.arange(len(image)), index=image)
    at = at[~at.index.duplicated()].reindex(names).to_numpy()
    have = ~np.isnan(at)
    row = np.where(have, at, 0).astype(np.int64)
    t = np.where(have[:, None], lm[row], np.nan).astype(np.float32)
    d = np.where(have, dice[row], np.nan).astype(np.float32)
    return t, d


def point_error(pred, truth):
    """Per-picture, per-point distance in pixels of 1024 between two N x 240 arrays in normalised units."""
    d = (pred.astype(np.float64) - truth.astype(np.float64)).reshape(len(pred), N_POINTS, 2)
    return np.sqrt((d ** 2).sum(axis=2)) * FRAME


def points(net, pix, cache_row, target, rows, mean, sd, batch):
    """Predicted landmarks / 1024 for `rows`, in `rows` order."""
    net.eval()
    out = np.zeros((len(rows), 2 * N_POINTS), np.float32)
    parts = [rows[a:a + batch] for a in range(0, len(rows), batch)]
    at = 0
    with torch.no_grad():
        for sl, x, _y in reader.prefetch(pix, cache_row, target, parts):
            x = (reader.to_float(x, pix.shape[-1]) - mean) / sd
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                z = net(x)
            out[at:at + len(sl)] = z.float().cpu().numpy()
            at += len(sl)
    return out


class OutlineNet(nn.Module):
    """reader.ChestNet's trunk (the same conv layers), then a head that reads the last feature grid flattened:
    ChestNet averages the grid into one value per channel, which throws away position, and an outline is
    position. `grid` is the grid's side (6 at 384 px)."""

    def __init__(self, grid):
        super().__init__()
        self.trunk = reader.ChestNet(CFG["widths"], CFG["dropout"], CFG["stem_stride"], 2 * N_POINTS).trunk
        self.drop = nn.Dropout(CFG["dropout"])
        self.head = nn.Linear(CFG["widths"][-1] * grid * grid, 2 * N_POINTS)

    def forward(self, x):
        return self.head(self.drop(self.trunk(x).flatten(1)))


def grid_side(px):
    """The trunk's last feature grid side at a px-pixel square picture, from one dummy forward on the processor."""
    trunk = reader.ChestNet(CFG["widths"], CFG["dropout"], CFG["stem_stride"], 1).trunk.eval()
    with torch.no_grad():
        return trunk(torch.zeros(1, 1, px, px)).shape[-1]


def new_net(px):
    return OutlineNet(grid_side(px)).to(DEV)


def train_fold(df, pix, cache_row, target, usable, fold, epochs, batch):
    px = pix.shape[-1]
    fold_of = df.fold.to_numpy()
    tr_all = np.flatnonzero(fold_of != fold)
    te = np.flatnonzero(fold_of == fold)
    # the early-stop patients are drawn exactly as reader.train_fold draws them (from every patient outside
    # the fold), then only the pictures CheXmask outlined well are kept on either side
    patients = np.array(sorted(set(df.patient.iloc[tr_all])))
    rng = np.random.default_rng(SEED + fold)
    held = set(patients[rng.permutation(len(patients))[:int(round(reader.VAL_SHARE * len(patients)))]])
    is_held = df.patient.iloc[tr_all].isin(held).to_numpy()
    good = usable[tr_all]
    val, tr = tr_all[is_held & good], tr_all[~is_held & good]
    mean, sd = reader.mean_sd(pix, cache_row[tr], px, rng)
    net = new_net(px)
    with torch.no_grad():
        net.head.bias.copy_(torch.from_numpy(target[tr].mean(axis=0)))
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g_cpu = np.random.default_rng(SEED + 100 + fold)
    print("fold %d: train %d, early-stop %d (CheXmask Dice RCA mean >= %.1f), predict %d images"
          % (fold, len(tr), len(val), MIN_DICE, len(te)), flush=True)
    best, best_state, since, took = np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        order = tr[g_cpu.permutation(len(tr))]
        parts = [order[a:a + batch] for a in range(0, len(order), batch)]
        total = torch.zeros((), dtype=torch.float64, device=DEV)
        for rows, x, y in reader.prefetch(pix, cache_row, target, parts):
            x = (reader.to_float(x, px) - mean) / sd
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                z = net(x)
            loss = F.smooth_l1_loss(z.float(), y, beta=BETA)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach().double() * len(rows)
        sched.step()
        err = float(point_error(points(net, pix, cache_row, target, val, mean, sd, batch), target[val]).mean())
        took.append(time.time() - t0)
        print("  epoch %2d  loss %.5f  early-stop point error %.2f px  %.0f s" % (ep, float(total) / len(tr), err, took[-1]),
              flush=True)
        if err < best - 1e-3:
            best, since = err, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            since += 1
            if since >= CFG["patience"]:
                print("  stopping: no lower early-stop error for %d epochs" % CFG["patience"], flush=True)
                break
    net.load_state_dict(best_state)
    info = {"fold": fold, "epochs": len(took), "early_stop_point_error_px": best, "mean": float(mean), "sd": float(sd),
            "train_images": len(tr), "early_stop_images": len(val), "predicted_images": len(te),
            "seconds": round(sum(took), 1), "epoch_seconds": [round(t, 2) for t in took]}
    return net, info, te


def write_fold(out, fold, image, pred):
    """landmarks_fold<k>.npz (1024 frame) and ctr_fold<k>.csv for the predicted pictures."""
    full = (pred.astype(np.float64) * FRAME).astype(np.float32)
    np.savez(os.path.join(out, "landmarks_fold%d.npz" % fold), image=image, landmarks=full)
    ctr = np.array([cc.ctr_from_points(p) for p in full])
    pd.DataFrame({"image": image, "ctr": ctr, "dice_mean": np.nan, "usable": 1, "min_dice": np.nan}).to_csv(
        os.path.join(out, "ctr_fold%d.csv" % fold), index=False)


def train(version, epochs, batch, folds):
    datasets.guard_test("dev")
    torch.manual_seed(SEED)
    card.require_card(DEV, torch)
    # 2026-10-03: leased before the data is loaded and before CUDA starts: waiting jobs held 2.1-6.9 GB doing
    # nothing, and at 23:21 the machine ran out of committed memory (a training job crashed)
    # 2026-10-04: one lease, taken inside limit_card_memory: a separate lease here made the second one wait
    # on this process's own 2.5 GB, and the job held that share for 70 minutes doing nothing
    xd.limit_card_memory(gb=float(os.environ.get("RAIDIOLOGY_OUTLINE_CARD_GB", "2.5")), name="chest outline")
    df, pix, cache_row, labels = reader.table()
    target, dice = targets_for(df.image.astype(str).to_numpy())
    usable = np.isfinite(target).all(axis=1) & (dice >= MIN_DICE)
    print("pictures: %d development images, %d with CheXmask landmarks at Dice RCA mean >= %.1f"
          % (len(df), int(usable.sum()), MIN_DICE), flush=True)
    # reader.prefetch indexes its labels by row; NaN rows are never asked for (not usable, or only predicted)
    out = out_dir(version)
    os.makedirs(out, exist_ok=True)
    infos = []
    with ledger.run("chest_outline", "train v%d folds %s" % (version, "".join(str(k) for k in folds))) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "cfg": CFG, "min_dice": MIN_DICE, "beta": BETA})
        fingerprint = {"task": "chest outline", "head": "flattened grid", "version": version, "epochs": epochs, "batch": batch, "cfg": CFG,
                       "seed": SEED, "min_dice": MIN_DICE, "beta": BETA, "split_sha256": xd.sha256(reader.splitmod.PATH),
                       "cache": reader.cache_identity(), "dev_rows": int(len(df)),
                       "landmarks": [os.stat(cc.LANDMARKS).st_size, os.stat(cc.LANDMARKS).st_mtime_ns]}
        fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True, default=str))
        for k in folds:
            path = os.path.join(out, "fold%d.pt" % k)
            ck = reader.load_resumable(path, {**fingerprint, "fold": k})
            if ck is not None:
                net = new_net(pix.shape[-1])
                net.load_state_dict(ck["state"])
                info = ck["info"]
                rows = np.flatnonzero(df.fold.to_numpy() == k)
                mean = torch.tensor(info["mean"], dtype=torch.float32, device=DEV)
                sd = torch.tensor(info["sd"], dtype=torch.float32, device=DEV)
                pred = points(net, pix, cache_row, target, rows, mean, sd, batch)
                print("fold %d: resumed from %s (no training)" % (k, path), flush=True)
            else:
                net, info, rows = train_fold(df, pix, cache_row, target, usable, k, epochs, batch)
                mean = torch.tensor(info["mean"], dtype=torch.float32, device=DEV)
                sd = torch.tensor(info["sd"], dtype=torch.float32, device=DEV)
                pred = points(net, pix, cache_row, target, rows, mean, sd, batch)
                torch.save({"state": net.state_dict(), "cfg": CFG, "info": info,
                            "fingerprint": {**fingerprint, "fold": k}}, path)
                led.number("fold%d.seconds" % k, info["seconds"], "s").number("fold%d.epochs" % k, info["epochs"], "count")
            led.number("fold%d.early_stop_point_error_px" % k, info["early_stop_point_error_px"], "px")
            led.number("fold%d.train_images" % k, info["train_images"], "count")
            # 2026-10-04: to_numpy(str), a fixed-width string array: astype(str).to_numpy() is an object array,
            # which savez pickles and report's np.load(allow_pickle=False) refuses (queue fifty-three's report)
            write_fold(out, k, df.image.to_numpy(str)[rows], pred)
            infos.append(info)
            print("fold %d: drew %d pictures -> %s" % (k, len(rows), out), flush=True)
    # a fold trained in an earlier run keeps its entry
    meta_path = os.path.join(out, "meta.json")
    kept = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            kept = [i for i in json.load(f).get("folds", []) if i["fold"] not in folds]
    meta = {"version": version, "cfg": CFG, "min_dice": MIN_DICE, "beta": BETA, "seed": SEED,
            "folds": sorted(kept + infos, key=lambda i: i["fold"]), "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def report(version, folds):
    out = out_dir(version)
    image, lm, dice = cc.load_landmarks()
    at = pd.Series(np.arange(len(image)), index=image)
    at = at[~at.index.duplicated()]
    for k in folds:
        path = os.path.join(out, "landmarks_fold%d.npz" % k)
        if not os.path.exists(path):
            raise SystemExit("no such file: %s (train fold %d first)" % (path, k))
        z = np.load(path, allow_pickle=False)
        names = z["image"].astype(str)
        row = at.reindex(names).to_numpy()
        have = ~np.isnan(row)
        row = np.where(have, row, 0).astype(np.int64)
        ok = have & (dice[row] >= MIN_DICE) & np.isfinite(lm[row]).all(axis=1)
        pred = z["landmarks"][ok].astype(np.float64) / FRAME
        truth = lm[row[ok]]
        err = point_error(pred, truth)
        ours = np.array([cc.ctr_from_points(p * FRAME) for p in pred])
        theirs = np.array([cc.ctr_from_points(p * FRAME) for p in truth])
        diff = np.abs(ours - theirs)
        print("outline v%d, fold %d: %d of %d pictures have CheXmask landmarks at Dice RCA mean >= %.1f"
              % (version, k, int(ok.sum()), len(names), MIN_DICE))
        print("mean point error against CheXmask, pixels of 1024: all %.2f" % err.mean()
              + "".join(", %s %.2f" % (n, err[:, a:b].mean()) for n, a, b in STRUCTURES))
        d = diff[np.isfinite(diff)]
        print("median absolute CTR difference, ours vs CheXmask's: %.4f (%d pictures)" % (np.median(d), len(d)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=CFG["epochs"])
    t.add_argument("--batch", type=int, default=CFG["batch"])
    r = sub.add_parser("report")
    for p in (t, r):
        p.add_argument("--version", type=int, required=True)
        p.add_argument("--folds", type=int, nargs="+", default=[0], help="folds to train or report, 0 for the trial")
    a = ap.parse_args()
    if a.cmd == "train":
        if DEV.type == "cuda":
            torch.backends.cudnn.benchmark = True
        train(a.version, a.epochs, a.batch, tuple(a.folds))
    else:
        report(a.version, tuple(a.folds))


if __name__ == "__main__":
    main()
