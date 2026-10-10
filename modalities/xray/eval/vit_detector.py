"""The fracture reader from masked-autoencoder pretraining (idea 129, step A): the chest ViT-S of mae-v1, fine-tuned as a
fracture detector on detector.py's own pictures, folds, collections, pos_weight, augmentation, card lease, partials and
out-of-fold writer.

mae-v1's encoder (work/eval/chest/mae-v1/encoder.pt) was pretrained on chest X-rays at 256 px with one grey channel. Here
it reads detector.py's 384 px pictures: its position embedding is resampled to the 24 by 24 grid (mae.load_encoder), and
the channel count stays one, so nothing is replicated. Each picture's patch tokens are averaged (the chest fine-tune's
pooling), the dataset one-hot is appended, and one linear layer gives the logit, as detector.py's head does.

A training picture keeps a random half of its patch tokens (mae.encode takes the kept indices, the way the autoencoder
drops masked patches), so a step attends over 288 patches instead of 576; the read uses all of them. The recipe is the
chest fine-tune's (mae.py's finetune_fold): layer-wise learning-rate decay 0.55, drop path 0.2, weight decay 0.05, five
warm-up epochs then cosine, 5e-4 per 256 pictures, and the epoch with the best early-stop AUROC is kept.

Its folds and meta.json carry "arch": "vit-s16-mae". detector.py's load_folds does not check it: it reads ck["cfg"], which
these folds do not carry, so detector.py's image command stops on a ViT version with a KeyError rather than a clean refusal
(its predict refuses trial versions first). Read a ViT trial only with this file's own predict: `predict test --look` reads
the test split once the recipe is final, and dev reads score each development picture with the fold that held it out.

Usage: python vit_detector.py train --version N [--epochs 30] [--batch 32] [--folds 0,1,2,3,4] [--set key=<json> ...]
       python vit_detector.py predict dev|test --version N [--look] [--batch 32]
       python vit_detector.py selftest
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "chest_xray", "eval"))
import detector as xd  # noqa: E402  (pictures, folds, collections, augmentation, card lease, partials, out-of-fold writer)
from core import card, ckpt, ledger, polite, sets  # noqa: E402
import mae  # noqa: E402  (mae-v1's encoder: make_vit, encode, load_encoder, the layer-wise groups and the learning-rate schedule)

START_MAE = 1                 # mae-v1: the chest encoder (work/eval/chest/mae-v1)
KEEP = 0.5                    # the share of patch tokens a training picture keeps
BATCH = 32                    # 4 to 5 GB of the 6.5 GB card share at 384 px (estimate, not measured)
FT_LR_PER_256 = 5e-4          # mae.py's FT_LR_PER_256
FT_WD = 0.05
LAYER_DECAY = 0.55
DROP_PATH = 0.2
WARM_EPOCHS = 5
OOF_COLUMNS = ["dataset", "image", "patient", "fold", "truth"]
ARCH = "vit-s16-mae"          # written into each fold file and meta.json: detector.py's loaders do not read these

# What a run may pass with `train --set key=<json>`; each is off by default, so the recipe above is what runs without one.
SETTINGS = {
    "keep": "the share of patch tokens a training picture keeps, above 0 and at most 1 (1 is every token); the read uses all",
    "mae": "the masked-autoencoder run whose encoder starts each fold (mae-v<N>; 1 is the chest encoder)",
    "val_share": "the early-stop slice of the training patients, 0 for none (the last epoch is kept)",
    "seed": "shifts every draw: one recipe at two seeds is the noise a change has to clear",
    "datasets": "the collections to train on, a list from datasets.ALL holding the shipped two (as detector.py)",
    "lr_per_256": "the peak learning rate per 256 pictures, scaled by the batch",
    "layer_decay": "the learning rate's factor per block, counted down from the head",
    "drop_path": "the stochastic depth rate of the encoder's blocks",
}


def recipe(settings):
    """The fine-tune's hyperparameters as this run has them: the defaults, with the settings that change them."""
    return {"keep": float(settings.get("keep", KEEP)), "mae": int(settings.get("mae", START_MAE)),
            "lr_per_256": float(settings.get("lr_per_256", FT_LR_PER_256)), "weight_decay": FT_WD,
            "layer_decay": float(settings.get("layer_decay", LAYER_DECAY)),
            "drop_path": float(settings.get("drop_path", DROP_PATH)), "warm_epochs": WARM_EPOCHS,
            "val_share": float(settings.get("val_share", xd.VAL_SHARE))}


def check_settings(settings):
    unknown = sorted(set(settings) - set(SETTINGS))
    if unknown:
        raise SystemExit("unknown setting(s) %s: train takes %s" % (", ".join(unknown), ", ".join(sorted(SETTINGS))))
    r = recipe(settings)
    if not 0.0 < r["keep"] <= 1.0:
        raise SystemExit("keep is the share of patch tokens kept, above 0 and at most 1: %r" % r["keep"])
    if not 0.0 <= r["val_share"] < 1.0:
        raise SystemExit("val_share is a share below 1: %r" % r["val_share"])
    xd.training_names(settings)


class FractureViT(nn.Module):
    """The encoder, the mean of its patch tokens, the dataset one-hot appended, and one linear layer to the logit."""

    def __init__(self, vit, n_datasets):
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.embed_dim + n_datasets, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, which, keep=None):
        """The logit of each picture; with `keep` (B, k), only those patch tokens take part (mae.encode)."""
        pooled = mae.encode(self.vit, x, keep)[:, 1:].mean(dim=1)
        return self.head(torch.cat([pooled, which.to(pooled.dtype)], dim=1)).squeeze(1)


def build(enc_path, px, n_datasets, drop_path):
    """The net: the encoder at enc_path resampled to px (h, w), with its pooled head, on the processor the pictures use."""
    vit, _ = mae.load_encoder(enc_path, px, drop_path)
    return FractureViT(vit, n_datasets).to(xd.DEV)


def keep_index(n_rows, n_patches, share, g, device):
    """(n_rows, k) random patch indices a training picture keeps, k = share of n_patches; None for every token (share 1)."""
    if share >= 1.0:
        return None
    k = int(round(n_patches * share))
    return torch.argsort(torch.rand(n_rows, n_patches, device=device, generator=g), dim=1)[:, :k]


def train_step(net, opt, loss_fn, x, which, y, mean, sd, g, keep):
    """One step from a batch of pictures in [0, 1]: augmented, standardised, the kept tokens' logits, the weights updated.
    Returns the loss, on the processor."""
    x = (xd.augment(x, g) - mean) / sd
    n = (x.shape[-2] // mae.PATCH) * (x.shape[-1] // mae.PATCH)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(x.device.type == "cuda")):
        out = net(x, which, keep_index(len(x), n, keep, g, x.device))
    loss = loss_fn(out.float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach()


def train_fold(pix, df, dev_rows, fold, epochs, batch, seed, n_datasets, enc_path, settings, partial=None, fingerprint=None):
    """Fine-tunes one fold's net from the encoder at enc_path on the development units outside `fold`, keeps the epoch with
    the best early-stop AUROC (the last epoch without an early-stop slice), and scores the fold's pictures with every token.
    Returns (net, info, rows, scores).

    With `partial` and `fingerprint` the fold is checkpointed after every epoch and, when core.ckpt.Every is due, after every
    batch; a partial with the same fingerprint is continued from (detector.train_fold's way, the epoch's batches drawn again
    from its start state and the trained ones skipped)."""
    dev, r = xd.DEV, recipe(settings)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    tr_all = dev_rows[df.fold.values[dev_rows] != fold]
    te = dev_rows[df.fold.values[dev_rows] == fold]
    units = sorted(set(df.patient.values[tr_all]))
    rng = np.random.default_rng(seed + fold)
    held = set(np.asarray(units)[rng.permutation(len(units))[:int(round(r["val_share"] * len(units)))]])
    val = tr_all[np.isin(df.patient.values[tr_all], list(held))]
    tr = tr_all[~np.isin(df.patient.values[tr_all], list(held))]
    sample = pix[torch.as_tensor(df.stack_row.values[tr[:4096]], dtype=torch.long, device=pix.device)]
    mean, sd = xd.pixel_mean_sd(sample)
    del sample
    px = tuple(int(v) for v in pix.shape[-2:])
    net = build(enc_path, px, n_datasets, r["drop_path"])
    steps = -(-len(tr) // batch)
    n_steps, warm = epochs * steps, min(r["warm_epochs"], epochs) * steps
    base = r["lr_per_256"] * batch / 256
    opt = torch.optim.AdamW(mae.param_groups(net, r["weight_decay"], r["layer_decay"], len(net.vit.blocks)),
                            lr=base, betas=(0.9, 0.999))
    pos = float(df.truth.values[tr].sum())
    pos_weight = torch.tensor([max(len(tr) - pos, 1.0) / max(pos, 1.0)], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    g_cpu = torch.Generator().manual_seed(seed + fold)
    g_dev = torch.Generator(device=dev).manual_seed(seed + fold)
    best, best_epoch, best_state, took = -1.0, 0, None, []
    pace = polite.Pace()
    print("fold %d: train %d, early-stop %d, predict %d images (pos_weight %.2f, %d of %d tokens kept in training)"
          % (fold, len(tr), len(val), len(te), float(pos_weight), round(r["keep"] * (px[0] // mae.PATCH) * (px[1] // mae.PATCH)),
             (px[0] // mae.PATCH) * (px[1] // mae.PATCH)), flush=True)
    every = ckpt.Every() if partial else None
    stop = xd.stop_spec()
    start_ep, skip, total0, ep = 1, 0, 0.0, 0

    def checkpoint(done, mid):
        """The partial checkpoint: `done` epochs finished, `mid` the batch state of the epoch after them (or None)."""
        ckpt.save_partial(partial, {
            "net": net.state_dict(), "opt": opt.state_dict(), "epoch": done, "mid": mid, "best": best,
            "best_epoch": best_epoch, "best_state": best_state, "took": took, "g_cpu": g_cpu.get_state(),
            "g_dev": g_dev.get_state(), "rng": ckpt.rng_state(), "fingerprint": fingerprint})
        every.mark()

    ck = ckpt.load_partial(partial, fingerprint) if partial else None
    if ck is not None and 0 <= ck["epoch"] <= epochs and not (ck["mid"] and ck["epoch"] >= epochs):
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        best, best_epoch, took = ck["best"], ck["best_epoch"], list(ck["took"])
        best_state = {k: t.to(dev) for k, t in ck["best_state"].items()} if ck["best_state"] is not None else None
        ckpt.set_rng_state(ck["rng"])
        g_dev.set_state(ck["g_dev"].cpu())
        ep = ck["epoch"]
        if ck["mid"]:
            g_cpu.set_state(ck["mid"]["g_cpu0"])
            start_ep, skip, total0 = ep + 1, ck["mid"]["batch"], ck["mid"]["total"]
        else:
            g_cpu.set_state(ck["g_cpu"])
            start_ep = ep + 1
        print("fold %d: resumed mid-fold at epoch %d of %d (partial checkpoint)%s"
              % (fold, start_ep, epochs, ", batch %d" % skip if skip else ""), flush=True)
    del ck
    for ep in range(start_ep, epochs + 1):
        net.train()
        g0 = g_cpu.get_state()
        t0 = time.time()
        total = total0 if ep == start_ep else 0.0
        skipped = skip if ep == start_ep else 0
        for n_done, rows in enumerate(xd.batches(tr, batch, g_cpu), 1):
            if n_done <= skipped:
                continue
            began = time.time()
            mae.set_lr(opt, base * mae.lr_factor((ep - 1) * steps + n_done - 1, n_steps, warm))
            x, which, y, _m, _s = xd.feed(pix, df, rows, mean, sd, n_datasets)
            loss = train_step(net, opt, loss_fn, x, which, y, mean, sd, g_dev, r["keep"])
            total += float(loss) * len(rows)
            pace.rest(time.time() - began)       # core/polite.py: half the card while somebody is at it
            if every:
                want_stop = stop is not None and stop == (ep, n_done)
                if every.due() or want_stop:
                    checkpoint(ep - 1, {"batch": n_done, "total": total, "g_cpu0": g0})
                    if want_stop:
                        raise xd.SelfTestStop("stopped after batch %d of epoch %d" % (n_done, ep))
        took.append(time.time() - t0)
        if len(val):
            v = xd.auroc(xd.predict_rows(net, pix, df, val, mean, sd, n_datasets, batch), df.truth.values[val])
            print("  epoch %2d  loss %.4f  val auroc %.4f  %.0f s" % (ep, total / max(len(tr), 1), v, took[-1]), flush=True)
            if not np.isnan(v) and (best_state is None or v > best + 1e-4):
                best, best_epoch = v, ep
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            print("  epoch %2d  loss %.4f  %.0f s" % (ep, total / max(len(tr), 1), took[-1]), flush=True)
        if every:
            checkpoint(ep, None)
            if stop == (ep, None):
                raise xd.SelfTestStop("stopped after epoch %d" % ep)
    if len(val) and best_state is None:
        raise SystemExit("fold %d: the early-stop slice's AUROC was undefined in every epoch (one class in %d images); "
                         "use a larger val_share, or 0 for none" % (fold, len(val)))
    if best_state is not None:
        net.load_state_dict(best_state)
    oof = xd.predict_rows(net, pix, df, te, mean, sd, n_datasets, batch)
    info = {"fold": fold, "epochs": epochs, "val_auroc": best if len(val) else float("nan"), "best_epoch": best_epoch, "mean": float(mean),
            "sd": float(sd), "train_images": len(tr), "val_images": len(val), "predicted_images": len(te),
            "epoch_seconds": [round(v, 2) for v in took]}
    return net, info, te, oof


def write_oof(out, df, oof, names, meta):
    """oof_dev.csv and meta.json in detector.py's format: one row per readable development image with its out-of-fold
    score, each collection's Youden threshold and AUROC in meta.json. Returns the thresholds."""
    thresholds = {}
    for d in names:
        m = (df.dataset == d) & df.readable
        thresholds[d] = xd.youden(oof[m].values, df.truth[m].values)
    frame = df.loc[df.readable, OOF_COLUMNS].copy()
    frame["score"] = oof[df.readable].values
    frame.to_csv(os.path.join(out, "oof_dev.csv"), index=False)
    meta.update({"thresholds": thresholds,
                 "dev_auroc": {d: xd.auroc(frame[frame.dataset == d].score.values, frame[frame.dataset == d].truth.values)
                               for d in names}})
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return thresholds


def train(version, epochs, batch, folds, settings=None):
    """Trains the wanted folds of trial `version` (version 100 and up, as detector.py's trials). A finished fold is kept and
    resumed, so the same command run again trains only what is missing; oof_dev.csv is written once all five folds exist."""
    settings = settings or {}
    check_settings(settings)
    if version < xd.TRIAL_FROM:
        raise SystemExit("vit_detector writes trials, version %d and up; version %d is a shipped number" % (xd.TRIAL_FROM, version))
    r = recipe(settings)
    seed = xd.datasets.SEED + 1000 * int(settings.get("seed", 0))
    torch.manual_seed(seed)
    card.require_card(xd.DEV, torch)
    xd.limit_card_memory()      # before the pictures are loaded: a waiting job holds no memory
    names = xd.training_names(settings)
    enc_path = os.path.join(mae.mae_dir(r["mae"]), "encoder.pt")
    if not os.path.exists(enc_path):
        raise SystemExit("no encoder at %s; pretrain it first" % enc_path)
    df, arrays, _ = xd.table(("dev",), names)
    pix = xd.stack(df, arrays)
    dev_rows = df.index.values[df.readable.values]
    n_datasets, px = len(names), tuple(int(v) for v in pix.shape[-2:])
    out = xd.model_dir(version)
    os.makedirs(out, exist_ok=True)
    st, size = os.stat(enc_path), xd.prepare.SIZE
    fingerprint = json.loads(json.dumps({
        "version": version, "epochs": epochs, "batch": batch, "settings": settings, "recipe": r, "seed": seed, "cfg": xd.CFG,
        "encoder": [st.st_size, st.st_mtime_ns], "split_sha256": xd.sha256(xd.splitmod.PATH),
        "cache_sha256": {d: xd.sha256(xd.prepare.cache_paths(d, size)[1]) for d in names},
        "dev_rows": int(len(dev_rows))}, sort_keys=True, default=str))
    oof = pd.Series(np.nan, index=df.index)
    infos = {}
    for k in range(xd.FOLDS):      # every fold: a finished one from an earlier call is loaded even when not listed now
        path = os.path.join(out, "fold%d.pt" % k)
        partial = os.path.join(out, "fold%d.partial.pt" % k)
        fp = {**fingerprint, "fold": k}
        ck = xd.load_resumable(path, fp)
        if ck is not None:
            net = build(enc_path, px, n_datasets, 0.0)
            net.load_state_dict(ck["state"])
            te = dev_rows[df.fold.values[dev_rows] == k]
            mean = torch.tensor(ck["info"]["mean"], dtype=torch.float32, device=xd.DEV)
            sd = torch.tensor(ck["info"]["sd"], dtype=torch.float32, device=xd.DEV)
            oof.iloc[te] = xd.predict_rows(net, pix, df, te, mean, sd, n_datasets, batch)
            infos[k] = ck["info"]
            print("fold %d: resumed from %s (no training)" % (k, path), flush=True)
            continue
        if k not in folds:         # not listed and not finished: this call leaves it to a later one
            continue
        with ledger.run("detector", "vit train fold %d v%d" % (k, version)) as led:
            led.settings_from({"epochs": epochs, "batch": batch, "recipe": r, **({"settings": settings} if settings else {})})
            net, info, rows, scores = train_fold(pix, df, dev_rows, k, epochs, batch, seed, n_datasets, enc_path,
                                                 settings, partial, fp)
            oof.iloc[rows] = scores
            xd.save_fold(path, {"state": net.state_dict(), "recipe": r, "info": info, "fingerprint": fp, "arch": ARCH}, partial)
            led.number("val_auroc", info["val_auroc"]).number("epochs", info["epochs"], "count")
            led.number("train_images", info["train_images"], "count")
            if xd.DEV.type == "cuda":
                led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
                led.number("peak_reserved_gb", torch.cuda.max_memory_reserved() / 2 ** 30)
                torch.cuda.reset_peak_memory_stats()
        infos[k] = info
    missing = [k for k in range(xd.FOLDS) if k not in infos]
    if missing:
        print("folds %s are not trained yet; oof_dev.csv waits for all %d folds" % (missing, xd.FOLDS), flush=True)
        return
    meta = {"version": version, "arch": ARCH, "recipe": r, "settings": settings, "mae": r["mae"], "folds": [infos[k] for k in range(xd.FOLDS)],
            "datasets": names, "seed": seed, "split_sha256": fingerprint["split_sha256"],
            "cache_sha256": fingerprint["cache_sha256"], "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0) if xd.DEV.type == "cuda" else "cpu"}
    thresholds = write_oof(out, df, oof, names, meta)
    print("trained: %s  out-of-fold AUROC %s  thresholds %s"
          % (out, {d: round(v, 4) for d, v in meta["dev_auroc"].items()}, {d: round(v, 4) for d, v in thresholds.items()}),
          flush=True)


def folds_arg(s):
    ks = sorted({int(v) for v in s.split(",") if v.strip()})
    if not ks or any(k < 0 or k >= xd.FOLDS for k in ks):
        raise argparse.ArgumentTypeError("folds are numbers from 0 to %d, comma separated" % (xd.FOLDS - 1))
    return ks


def predict(split, version, batch, look=False):
    """Scores a split's rows with the saved ViT folds and writes pred_<split>.csv in the trial's folder. A development row
    is scored by the fold that held it out, which is its out-of-fold score; a test row by the mean of the five folds.
    The test split is read only with `look`, once the recipe is final (detector.py's predict guards it the same way)."""
    if version < xd.TRIAL_FROM:
        raise SystemExit("vit_detector reads trials, version %d and up" % xd.TRIAL_FROM)
    if split == "test" and not look:
        raise SystemExit("the test split is read only with --look, once the recipe is final; dev is the working split")
    out = xd.model_dir(version)
    meta_path = os.path.join(out, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("no %s: train all %d folds first" % (meta_path, xd.FOLDS))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("arch") != ARCH:
        raise SystemExit("version %d is not a %s trial (arch %r)" % (version, ARCH, meta.get("arch")))
    names = meta["datasets"]
    df, arrays, _ = xd.table((split,), names)
    pix = xd.stack(df, arrays)
    rows = df.index.values[df.readable.values]
    px = tuple(int(v) for v in pix.shape[-2:])
    total, count = np.zeros(len(df)), np.zeros(len(df))
    for k in range(xd.FOLDS):
        ck = torch.load(os.path.join(out, "fold%d.pt" % k), map_location=xd.DEV, weights_only=False)
        if ck.get("arch") != ARCH:
            raise SystemExit("fold %d is not a %s fold" % (k, ARCH))
        net = build(os.path.join(mae.mae_dir(ck["recipe"]["mae"]), "encoder.pt"), px, len(names), 0.0)
        net.load_state_dict(ck["state"])
        mean = torch.tensor(ck["info"]["mean"], dtype=torch.float32, device=xd.DEV)
        sd = torch.tensor(ck["info"]["sd"], dtype=torch.float32, device=xd.DEV)
        sel = rows[df.fold.values[rows] == k] if split == "dev" else rows
        total[sel] += xd.predict_rows(net, pix, df, sel, mean, sd, len(names), batch)
        count[sel] += 1
        del net, ck
    frame = df.loc[df.readable, OOF_COLUMNS].copy()
    frame["score"] = (total / np.maximum(count, 1))[df.readable.values]
    frame.to_csv(os.path.join(out, "pred_%s.csv" % split), index=False)
    for d in names:
        m = frame.dataset == d
        print("%s %s: %d images, AUROC %.4f" % (split, d, int(m.sum()), xd.auroc(frame[m].score.values, frame[m].truth.values)),
              flush=True)


def selftest():
    """Keep and resume on a tiny ViT and made-up pictures, on the processor (about a minute, no data, no card).

    Checks: keeping every token is the plain forward; a fold stopped after epoch 2 (a partial each epoch) and after batch 3 of
    epoch 2 (a partial each batch), then resumed, ends with the weights, held-out scores and early-stop AUROC of the
    uninterrupted fold; the out-of-fold writer's columns are detector.py's."""
    import contextlib
    import io
    import re
    import tempfile
    saved_env = {k: os.environ.get(k) for k in ("RAIDIOLOGY_SHARE", "RAIDIOLOGY_SAVE_EVERY_S",
                                                "RAIDIOLOGY_CKPT_SELFTEST_STOP", "RAIDIOLOGY_NO_RESUME")}
    saved_dev = xd.DEV
    xd.DEV = torch.device("cpu")
    os.environ["RAIDIOLOGY_SHARE"] = "1"
    os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
    try:
        cpu = torch.device("cpu")
        g = torch.Generator().manual_seed(3)
        px, n, epochs, batch = 64, 120, 4, 16
        patient = np.repeat(np.arange(n // 2), 2)
        truth = (patient % 2).astype(np.float32)
        pix = torch.randint(0, 120, (n, px, px), dtype=torch.uint8, generator=g)
        pix[torch.as_tensor(truth > 0)] += 100
        df = pd.DataFrame({"dataset": "grazpedwri", "image": ["img%d" % i for i in range(n)], "dataset_index": 0,
                           "patient": patient, "truth": truth, "fold": patient % 5, "stack_row": np.arange(n),
                           "readable": True})
        rows = np.arange(n)
        settings = {"val_share": 0.3}
        with tempfile.TemporaryDirectory() as tmp:
            enc = os.path.join(tmp, "encoder.pt")
            mae.save_encoder(mae.make_vit(px, 32, 2, 2), enc, {"dim": 32, "depth": 2, "heads": 2})
            net = build(enc, (px, px), 1, 0.0).eval()
            x, which = torch.rand(2, 1, px, px, generator=g), torch.ones(2, 1)
            tokens = (px // mae.PATCH) ** 2
            assert keep_index(2, tokens, 1.0, g, cpu) is None, "keep 1.0 should take no index"
            everything = torch.arange(tokens).expand(2, -1)
            assert torch.allclose(net(x, which), net(x, which, everything), atol=1e-6), "keeping every token is not the plain forward"
            half = keep_index(2, tokens, 0.5, g, cpu)
            assert half.shape == (2, tokens // 2) and all(len(set(row.tolist())) == half.shape[1] for row in half), "half: repeats"
            print("selftest: keep 1.0 is the plain forward; keep 0.5 keeps %d of %d tokens" % (half.shape[1], tokens), flush=True)

            def fold_run(partial, fp):
                buf = io.StringIO()
                torch.manual_seed(1)                # train() seeds once per run; here every fold starts alike
                try:
                    with contextlib.redirect_stdout(buf):
                        out = train_fold(pix, df, rows, 0, epochs, batch, 7, 1, enc, settings, partial, fp)
                except xd.SelfTestStop:
                    out = None
                return out, buf.getvalue()

            def epochs_run(text):
                return [int(m) for m in re.findall(r"^  epoch +(\d+) ", text, re.M)]

            ref, _ = fold_run(None, None)
            for i, (label, stop, resume_at, batch_done) in enumerate((
                    ("stopped after epoch 2", "2", 3, 0), ("stopped after batch 3 of epoch 2", "2:3", 2, 3))):
                path = os.path.join(tmp, "fold0_%d.partial.pt" % i)
                fp = {"selftest": label}
                os.environ["RAIDIOLOGY_SAVE_EVERY_S"] = "0" if batch_done else "100000"
                os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = stop
                out, _ = fold_run(path, fp)
                os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP")
                assert out is None and os.path.exists(path), "%s: no partial after the stop" % label
                before = ckpt.load_partial(path, fp)
                assert before["epoch"] == resume_at - 1 and (before["mid"] or {}).get("batch", 0) == batch_done, label
                out, text = fold_run(path, fp)
                assert "fold 0: resumed mid-fold at epoch %d of %d" % (resume_at, epochs) in text, "%s: no resume line" % label
                assert epochs_run(text) == list(range(resume_at, epochs + 1)), "%s: ran epochs %s" % (label, epochs_run(text))
                (net_r, info_r, _te_r, oof_r), (net_f, info_f, _te_f, oof_f) = ref, out
                for (ka, va), (_kb, vb) in zip(net_r.state_dict().items(), net_f.state_dict().items()):
                    assert torch.allclose(va.float(), vb.float(), rtol=1e-4, atol=1e-5), "%s: weight %s differs" % (label, ka)
                assert abs(info_r["val_auroc"] - info_f["val_auroc"]) < 1e-6 and info_r["epochs"] == info_f["epochs"], label
                assert np.allclose(oof_r, oof_f, atol=1e-5), "%s: held-out scores differ" % label
                xd.save_fold(os.path.join(tmp, "fold0_%d.pt" % i), {"state": net_f.state_dict()}, path)
                assert not os.path.exists(path), "%s: partial left after the fold's save" % label
                print("selftest: %s: resumed at epoch %d, weights and scores equal the uninterrupted fold"
                      % (label, resume_at), flush=True)
            frame = pd.DataFrame({"dataset": ["grazpedwri"] * 6 + ["fracatlas"] * 6, "image": ["i%d" % j for j in range(12)],
                                  "patient": np.arange(12), "fold": np.arange(12) % 5, "truth": [0, 1] * 6, "readable": True})
            write_oof(tmp, frame, pd.Series(np.linspace(0.05, 0.95, 12), index=frame.index), ["grazpedwri", "fracatlas"],
                      {"version": 0})
            cols = list(pd.read_csv(os.path.join(tmp, "oof_dev.csv")).columns)
            assert cols == ["dataset", "image", "patient", "fold", "truth", "score"], "oof columns: %s" % cols
            print("selftest: oof_dev.csv columns are detector.py's: %s" % ", ".join(cols), flush=True)
        print("selftest: ok", flush=True)
    finally:
        xd.DEV = saved_dev
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--folds", type=folds_arg, default=list(range(xd.FOLDS)))
    t.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(SETTINGS)))
    p = sub.add_parser("predict", help="scores a split with the saved ViT folds")
    p.add_argument("split", choices=["dev", "test"])
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--look", action="store_true", help="the test split is read only with this flag")
    sub.add_parser("selftest", help="keep and resume on a tiny ViT, processor only")
    a = ap.parse_args()
    if xd.DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(xd.datasets.SEED)
    if a.cmd == "train":
        train(a.version, a.epochs, a.batch, a.folds, sets.parse(a.set, SETTINGS))
    elif a.cmd == "predict":
        predict(a.split, a.version, a.batch, a.look)
    else:
        selftest()


if __name__ == "__main__":
    main()
