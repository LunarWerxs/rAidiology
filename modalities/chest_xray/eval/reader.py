"""Chest X-ray reader: a small convolutional network trained from scratch to read fifteen labels
(NIH ChestX-ray14's fourteen findings plus Abnormal) from one frontal picture, scored out-of-fold.

From scratch, no pretrained weights of any kind: ImageNet's terms are non-commercial and rAidiology
has to stay commercially clean (LICENSING.md). The trunk is the fracture detector's (five two-conv
stages, BatchNorm, ReLU, MaxPool, stride-2 stem), then global average pooling, dropout and one linear
layer to 15 logits. There is no dataset one-hot: this is one collection. Loss is BCE-with-logits over
the 15 labels.

Fold k trains on the development patients outside fold k, minus an early-stop set of 10% of those
PATIENTS (seeded, so one patient's pictures never sit on both sides), keeps the epoch with the best
mean AUROC on that set, and predicts fold k, which it never saw. The held-out test list is never
opened here.

Pixels: 86k pictures at 384 px are 12.8 GB, more than the card's ceiling or the free RAM, so they stay
in prepare.load's memmap on disk. One background thread reads each batch's rows in a single sorted
fancy-index (disk locality), restores the batch order, pins and copies them to the card, and keeps a
queue two batches deep, so the disk reads overlap the GPU step. Augmentation (detector.augment) and
the normalising mean and sd (measured on 2,000 training pictures of the fold) run on the card.
torch.compile is not used: Windows has no reliable Triton.

Trial switches (--set key=<json>), each off by default: cache_size (read nih_<n> instead of nih_384), size (train and read at a smaller square,
area-resampled on the card), amp (bfloat16 autocast in the training step), pos_weight (BCE
pos_weight per label, negatives over positives of the training rows, clipped to [1, 20]),
radiologist_weight (needs radiologist_labels=1: the radiologists' pictures count k times in the loss).
clean_first (with clean_epochs k): NIH's labels were mined from reports and many are wrong. An earlier
five-fold run's out-of-fold logits say which cells the net that never saw them disputes; for the first k
epochs those cells are left out of the loss, then every cell counts. Early stopping is never masked.
renderings (3): the network also sees a rank-equalised copy and a high-pass copy of each picture, made inside
ChestNet.forward from the one stored grey channel, so every reader built from its saved settings reads them.
lung_aux (w): a head on the stride-16 features draws CheXmask's lungs (24 by 24), loss weight w, on training
pictures with CheXmask quality >= 0.7; the masks come from lung_masks.py's cache. Training only: the head is not
saved and never used to read.
extra (["drr-luna16"]): idea 43 step 2, LUNA16's DRRs (drr_probe.py render) join each fold's training rows as Nodule-only
rows, four copies each an epoch; see SETTINGS.
widths ([64,128,256,512,512]): the five channel counts of ChestNet's levels, in place of CFG's [32,64,128,256,256]; every reader
built from a saved version takes them from its settings (widths_of).
patience (24): whole epochs without a better early-stop AUROC before a fold stops, in place of CFG's 3; see SETTINGS.

Usage: python reader.py train --version 100 [--epochs 12] [--batch 64] [--folds 0] [--set key=<json> ...]
"""
import argparse
import json
import math
import os
import queue
import sys
import threading
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
XRAY_EVAL = os.path.join(os.path.dirname(HERE), "..", "xray", "eval")
XRAY_EVAL = os.path.normpath(XRAY_EVAL)
sys.path.insert(0, XRAY_EVAL)
sys.path.insert(0, ROOT)
import detector as xd  # noqa: E402

# detector.py bound the fracture scan type's datasets, prepare, split and preds under those names.
# Clear them so this scan type's own modules of the same names import cleanly; detector keeps its own.
for _m in ("datasets", "prepare", "split", "preds"):
    sys.modules.pop(_m, None)
while XRAY_EVAL in sys.path:
    sys.path.remove(XRAY_EVAL)
sys.path.insert(0, HERE)
from core import card, ckpt, ledger, paths, polite, sets  # noqa: E402
import datasets  # noqa: E402
import lung_masks  # noqa: E402
import prepare  # noqa: E402
import split as splitmod  # noqa: E402

DEV = xd.DEV
VAL_SHARE = 0.10
SEED = 20261003
CFG = {"epochs": 12, "batch": 64, "lr": 3e-4, "weight_decay": 1e-4, "patience": 3,
       "widths": [32, 64, 128, 256, 256], "dropout": 0.3, "stem_stride": 2,
       "rotate_deg": 10.0, "translate": 0.05, "jitter": 0.10}
SETTINGS = {
    "size": "train and read at a smaller square (e.g. 256), area-resampled on the card after the batch is moved",
    "amp": "bfloat16 autocast in the training step",
    "pos_weight": "BCE pos_weight per label = negatives/positives of the training rows, clipped to [1, 20]",
    "invert": "the share of training pictures shown inverted (detector.augment's); 0 is the recipe's",
    "gamma": "a random gamma change on every training picture, log e in [-gamma, gamma] (detector.augment's)",
    "radiologist_labels": "1: train on ChestX-Det's radiologist answers for the 10 shared findings on the pictures it "
                          "labelled (Abnormal also 1 when they marked any finding); saved out-of-fold y stays NIH's; 0 is the recipe's",
    "radiologist_weight": "k > 0, needs radiologist_labels=1: each training picture the radiologists answered counts k times "
                          "in the loss, every other 1; unset is the recipe's plain mean",
    "distill": "path of a chest_teacher npz (\"{fold}\" in it is the fold trained): the loss is (1 - distill_weight) x the usual BCE on the training labels plus "
               "distill_weight x BCE-with-logits against sigmoid(the teacher's logits), no pos_weight on that term; "
               "early stopping stays on the labels; unset is the recipe's",
    "distill_weight": "w in (0, 1], needs distill: the share of the loss taught by the teacher",
    "grow_heart": "p in (0, 1]: each training picture labelled No Finding with CheXmask quality >= 0.7 grows its heart "
                  "with probability p (docs/efficiency trial 5); unset is the recipe's",
    "crop": "name of a crop cache from tools/chest_lung_crops.py (\"lungs0\": fold 0 lung boxes, same rows as nih_384), "
            "read in place of nih_384; unset is the recipe's",
    "cache_size": "side n of the picture cache to read (nih_<n>, built by prepare.py --size n; e.g. 512); not with crop; "
                  "size may not exceed it; unset is the recipe's nih_384",
    "ema": "decay d in (0, 1): an exponential moving average of the weights and BatchNorm buffers "
           "(torch.optim.swa_utils.AveragedModel, get_ema_multi_avg_fn(d), use_buffers=True) is updated after every "
           "optimiser step; early stopping scores it, it is saved and it reads the held-out fold; unset is the recipe's",
    "clean_first": "path of an out-of-fold logits npz from an earlier five-fold run (image, y, logit): the cells it disputes "
                   "(confident learning's rule) are left out of the training loss for the first clean_epochs epochs; "
                   "early stopping and the held-out fold are never masked; unset is the recipe's",
    "clean_epochs": "k > 0, needs clean_first: how many epochs, from the first, ignore the disputed cells; later epochs count every cell",
    "renderings": "3: ChestNet's first convolution takes three channels, the stored grey picture, its per-picture rank-equalised "
                  "copy and its high-pass copy (the picture minus its Gaussian blur, sigma 8 px at 384, scaled with the side), "
                  "each extra channel scaled per picture to zero mean and unit spread; built in the forward pass, after "
                  "augmentation and resizing, for training and reading alike; 1 is the recipe's and is left out of the fingerprint",
    "lung_aux": "w > 0: a 1x1 head on the stride-16 features predicts CheXmask's left and right lung (24 by 24, from "
                "lung_masks.py's cache); w x its BCE is added to the loss on training pictures whose CheXmask Dice >= 0.7; "
                "the head is dropped from the saved reader and never used to read; prints its Dice on the early-stop pictures; "
                "not with crop; 0 is the recipe's and is left out of the fingerprint",
    "exclude_folds": "a list of folds whose pictures no fold of this run trains or early-stops on; the held-out fold may not be one; "
                     "absent or empty is the recipe's and is left out of the fingerprint",
    "extra": "a list of extra training sets; the only name is \"drr-luna16\" (idea 43 step 2): LUNA16's DRRs from drr_probe.py render, "
             "the 224 scans with a nodule of 10 mm or more as Nodule 1 and the 287 with none as Nodule 0 (the 377 small ones are left out), "
             "four copies each an epoch in every fold's training rows, each augmented as an NIH picture; only the Nodule cell is in the "
             "loss on such a row; never in early stopping, the held-out fold, mean and sd or pos_weight; needs the 384 cache, and not with "
             "crop, lung_aux, grow_heart, distill, clean_first or radiologist_labels; absent or empty is the recipe's and is left out of the fingerprint",
    "widths": "a list of exactly 5 positive whole numbers: the channels of ChestNet's five levels (e.g. [64, 128, 256, 512, 512]); "
              "absent is the recipe's CFG widths and is left out of the fingerprint",
    "patience": "a positive whole number of epochs without a better early-stop AUROC before a fold stops (e.g. 24); "
                "absent is the recipe's CFG 3 and is left out of the fingerprint",
}
EXTRA_NAMES = ("drr-luna16",)
DRR_COPIES = 4              # copies of each DRR row an epoch
GROW_MIN_DICE = 0.7         # CheXmask's README advice, as chexmask_ctr.py
GROW_RANGE = (1.2, 1.4)     # the sideways stretch s is drawn uniformly from this
GROW_MIN_CTR = 0.55         # a grown heart must read at least this CTR (else s is raised to GROW_CTR_TARGET / CTR)
GROW_CTR_TARGET = 0.56
GROW_MAX_S = 1.6            # past this the picture is skipped
GROW_FADE = 0.15            # the stretch fades to nothing over this share of the heart's height above and below
RADIOLOGIST_FINDINGS = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
                        "Effusion": "Effusion", "Emphysema": "Emphysema", "Fibrosis": "Fibrosis", "Mass": "Mass",
                        "Nodule": "Nodule", "Pleural Thickening": "Pleural_Thickening", "Pneumothorax": "Pneumothorax"}


def out_dir(version):
    return os.path.join(paths.WORK, "eval", "chest", "reader-v%d" % version)


HIGHPASS_SIGMA = 8.0        # px at 384; scaled with the picture's side
LUNG_STAGES = 3             # trunk stages up to the stride-16 map (stem stride 2, then three halvings)


def standardise(c):
    """Each picture of (B, 1, H, W) to zero mean and unit spread."""
    return (c - c.mean(dim=(1, 2, 3), keepdim=True)) / c.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)


def gaussian_blur(x, sigma):
    """(B, 1, H, W) blurred by a separable Gaussian (radius 3 sigma), reflect padding."""
    r = min(max(1, int(math.ceil(3 * sigma))), min(x.shape[-2:]) - 1)
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t * t / (2 * sigma * sigma))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def add_renderings(x):
    """(B, 1, H, W) grey to (B, 3, H, W): the grey as it is, then each picture's rank equalisation (every pixel's rank among its
    picture's pixels over their count) and its high-pass (the picture minus its Gaussian blur, sigma HIGHPASS_SIGMA px at 384),
    each scaled per picture to zero mean and unit spread. Float32, no autocast."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        rank = x.flatten(1).argsort(dim=1).argsort(dim=1).float().div_(x[0].numel()).view_as(x)
        high = x - gaussian_blur(x, HIGHPASS_SIGMA * x.shape[-1] / 384.0)
        return torch.cat([x, standardise(rank), standardise(high)], dim=1)


def chest_layout():
    """The memory order ChestNet runs in: "channels_last" (the default since 2026-10-06) or "plain", chosen by
    RAIDIOLOGY_CHEST_LAYOUT; RAIDIOLOGY_CHEST_LAYOUT=plain restores the old order. An A/B
    (tools/ab.py, 5 of 5 pairs) found a training step 1.35x faster in channels-last (0.164 s against 0.221 s at 384 px, batch 64,
    bf16). It is outside every fingerprint and saved setting: the same arithmetic in another memory order."""
    v = os.environ.get("RAIDIOLOGY_CHEST_LAYOUT")
    if v in (None, "", "channels_last"):
        return "channels_last"
    if v == "plain":
        return "plain"
    raise SystemExit("RAIDIOLOGY_CHEST_LAYOUT is %r; the two allowed values are 'plain' and 'channels_last'" % v)


class ChestNet(nn.Module):
    """The trunk, global average pooling, dropout and a linear layer. `renderings` 3 makes the first convolution take
    the three channels add_renderings builds from the one grey input (forward still takes (B, 1, H, W)). `lung_aux`
    adds a 1x1 convolution on the stride-16 map to two lung channels, used by forward_both in training only."""

    def __init__(self, widths, dropout, stem_stride, n_labels, renderings=1, lung_aux=False):
        super().__init__()
        self.renderings = int(renderings)
        stages, c_in = [], self.renderings
        for i, c in enumerate(widths):
            stride = stem_stride if i == 0 else 1
            stages.append(nn.Sequential(
                nn.Conv2d(c_in, c, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2)))
            c_in = c
        self.trunk = nn.Sequential(*stages)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, n_labels)
        if lung_aux:
            self.aux = nn.Conv2d(widths[LUNG_STAGES - 1], 2, 1)
        self.channels_last = chest_layout() == "channels_last"
        if self.channels_last:
            self.to(memory_format=torch.channels_last)

    def forward(self, x):
        if self.renderings == 3:
            x = add_renderings(x)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.head(self.drop(self.trunk(x).mean(dim=(2, 3))))

    def forward_both(self, x):
        """(15 logits, lung logits (B, 2, h, w) at the stride-16 map's size)."""
        if self.renderings == 3:
            x = add_renderings(x)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        f = self.trunk[:LUNG_STAGES](x)
        return self.head(self.drop(self.trunk[LUNG_STAGES:](f).mean(dim=(2, 3)))), self.aux(f)


def widths_of(settings, cfg=None):
    """The level widths a version was trained with: its settings' `widths` when present, else the saved cfg's (CFG's when none)."""
    w = (settings or {}).get("widths")
    return list(w) if w is not None else list((cfg or CFG)["widths"])


def build_net(settings, n_labels=None, lung_aux=False):
    """The ChestNet a version's settings describe: what training builds (lung_aux True when it trains the lung head) and what every
    reader built from a saved version's settings reads with."""
    n_labels = len(datasets.LABELS) if n_labels is None else n_labels
    return ChestNet(widths_of(settings), CFG["dropout"], CFG["stem_stride"], n_labels,
                    renderings=int((settings or {}).get("renderings", 1)), lung_aux=lung_aux)


def lung_table(df, size):
    """(masks memmap, eligible, src) over df's rows for lung_aux: `src` is the row in lung_masks' cache (-1 where CheXmask has none)
    and a row is eligible when its quality is at least GROW_MIN_DICE (grow_heart's rule), its landmarks parsed and its cached picture is square."""
    masks, image, dice = lung_masks.load()
    at = {n: i for i, n in enumerate(image)}
    src = np.array([at.get(n, -1) for n in df.image.to_numpy(str)], dtype=np.int64)
    d = np.full(len(df), np.nan)
    d[src >= 0] = dice[src[src >= 0]]
    with open(prepare.cache_paths(size)[1], encoding="utf-8") as fh:
        shape = json.load(fh)["source_shape"]
    square = np.array([str(n) in shape and shape[str(n)][0] == shape[str(n)][1] for n in df.image])
    return masks, (d >= GROW_MIN_DICE) & square, src


def augment_masks(m, state, device):
    """The picture's flip and affine (detector.augment's first four draws, replayed from the generator `state` taken before it)
    applied to the lung masks m (B, 2, h, w), bilinear. Colour changes do not touch masks. A flip also swaps the two channels,
    so channel 0 is always the lung on the picture's left side and channel 1 the one on its right."""
    g = torch.Generator(device=device)
    g.set_state(state)
    n = m.shape[0]
    flip = torch.rand(n, device=device, generator=g) < 0.5
    m = torch.where(flip[:, None, None, None], m.flip(-1).flip(1), m)     # a mirrored picture swaps which lung is on which side
    ang = (torch.rand(n, device=device, generator=g) * 2 - 1) * (xd.CFG["rotate_deg"] * np.pi / 180.0)
    tx = (torch.rand(n, device=device, generator=g) * 2 - 1) * xd.CFG["translate"] * 2
    ty = (torch.rand(n, device=device, generator=g) * 2 - 1) * xd.CFG["translate"] * 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=device, dtype=m.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, list(m.shape), align_corners=False)
    return F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def lung_logits(aux_out, size):
    """The lung head's logits (B, 2, h, w) at the mask's size (bilinear when the map is another size)."""
    aux_out = aux_out.float()
    if aux_out.shape[-2:] != (size, size):
        aux_out = F.interpolate(aux_out, size=(size, size), mode="bilinear", align_corners=False)
    return aux_out


def lung_aux_loss(aux_out, masks, use):
    """BCE of the lung head against the masks, averaged over the pictures `use` (B,) bool marks; 0 when none."""
    per = F.binary_cross_entropy_with_logits(lung_logits(aux_out, masks.shape[-1]), masks, reduction="none").mean(dim=(1, 2, 3))
    u = use.to(per.dtype)
    return (per * u).sum() / u.sum().clamp(min=1.0)


def lung_dice(net, pix, cache_row, labels, rows, lung, mean, sd, batch, px):
    """Mean Dice (over pictures and both lungs, masks and predictions cut at 0.5) of the lung head on `rows` that have a mask;
    (nan, 0) when none do."""
    masks, eligible, src = lung
    rows = rows[eligible[rows]]
    if not len(rows):
        return float("nan"), 0
    net.eval()
    scores = []
    with torch.no_grad():
        for sl, x, _y in prefetch(pix, cache_row, labels, [rows[a:a + batch] for a in range(0, len(rows), batch)]):
            x = (to_float(x, px) - mean) / sd
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                _z, a = net.forward_both(x)
            truth = torch.from_numpy(np.asarray(masks[src[sl]])).to(DEV).float() / 255.0 > 0.5
            pred = lung_logits(a, truth.shape[-1]) > 0
            inter = (pred & truth).sum(dim=(2, 3)).float()
            both = pred.sum(dim=(2, 3)).float() + truth.sum(dim=(2, 3)).float()
            scores.append(torch.where(both > 0, 2 * inter / both.clamp(min=1.0), torch.ones_like(both)).cpu().numpy().ravel())
    return float(np.concatenate(scores).mean()), int(len(rows))


def cache_size_of(settings):
    """The side of the picture cache a version reads: its `cache_size` setting, else the recipe's 384."""
    return int((settings or {}).get("cache_size", prepare.SIZE))


def table(size=384):
    """Development rows (positions 0..n-1), their labels and fold, and where each one's pixels are in
    the memmap. Unreadable pictures are left out."""
    df = splitmod.frame("dev").reset_index(drop=True)
    pix, order, _ = prepare.load(size)
    bad = prepare.unreadable(size)
    keep = np.array([str(i) not in bad and str(i) in order for i in df.image])
    df = df[keep].reset_index(drop=True)
    cache_row = np.array([order[str(i)] for i in df.image], dtype=np.int64)
    labels = df[datasets.LABELS].to_numpy(np.float32)
    return df, pix, cache_row, labels


def radiologist_training_labels(df, labels):
    """A copy of `labels` where, for every picture ChestX-Det's three board-certified radiologists
    labelled (matched back to NIH's names by tools/chestxdet_match.py), the ten findings both use take
    the radiologists' answer in place of NIH's: Atelectasis, Cardiomegaly, Consolidation, Effusion,
    Emphysema, Fibrosis, Mass, Nodule, Pleural Thickening and Pneumothorax. NIH's other four findings
    stay. Abnormal becomes 1 when NIH's was 1 or the radiologists marked any of their 13 findings.

    Why: NIH's labels were mined from reports by NLP. On the 3,025 matched pictures NIH's label catches
    only 16 of 100 radiologist-marked consolidations and 22 of 100 enlarged hearts, and the reader
    already agrees with the radiologists (mean AUROC 0.798) better than with NIH's labels (0.719).

    Only what the network trains on changes. `labels` itself is left alone, so the saved out-of-fold y
    keeps NIH's labels and every comparison with earlier versions stays like for like. NIH's test
    split is never touched: the development table holds none of it, and this stops if one of the
    overridden pictures is in it. Prints, per finding, how many label cells went 0 to 1 and 1 to 0.
    Returns (the labels, a boolean mask over df's rows: True for the pictures the radiologists answered)."""
    import pandas as pd
    cxd_dir = os.path.join(paths.DATASETS, "public", "chestx-det")
    match = pd.read_csv(os.path.join(cxd_dir, "nih_match.csv"))
    syms = {}
    for part in ("train", "test"):
        with open(os.path.join(cxd_dir, "ChestX_Det_%s.json" % part), encoding="utf-8") as f:
            for r in json.load(f):
                syms[r["file_name"]] = set(r["syms"])
    match = match[match.file_name.isin(syms) & match.image.isin(set(df.image))]
    row_of = {str(i): n for n, i in enumerate(df.image)}
    rows = np.array([row_of[str(i)] for i in match.image], dtype=np.int64)
    if len(rows) and (df.nih_split.to_numpy()[rows] == "test").any():
        raise SystemExit("radiologist_labels: an overridden picture is in NIH's test split")
    out = labels.copy()
    col = {n: j for j, n in enumerate(datasets.LABELS)}
    changed = []
    for cxd, lab in RADIOLOGIST_FINDINGS.items():
        rad = np.array([cxd in syms[f] for f in match.file_name], np.float32)
        before = out[rows, col[lab]].copy()
        out[rows, col[lab]] = rad
        changed.append("%s +%d -%d" % (lab, int(((before == 0) & (rad == 1)).sum()), int(((before == 1) & (rad == 0)).sum())))
    any_rad = np.array([bool(syms[f]) for f in match.file_name])
    ab = col["Abnormal"]
    before = out[rows, ab].copy()
    out[rows, ab] = np.maximum(before, any_rad.astype(np.float32))
    changed.append("Abnormal +%d" % int(((before == 0) & any_rad).sum()))
    print("radiologist_labels: %d pictures, %d label cells changed (0 to 1 / 1 to 0): %s"
          % (len(rows), int((out != labels).sum()), ", ".join(changed)), flush=True)
    answered = np.zeros(len(df), dtype=bool)
    answered[rows] = True
    return out, answered


def prefetch(pix, cache_row, labels, parts):
    """Yield (rows, uint8 pictures on the card, labels on the card) for each batch in `parts`, read by
    one background thread that stays two batches ahead of the consumer."""
    q = queue.Queue(maxsize=2)
    stop = threading.Event()
    cuda = DEV.type == "cuda"

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def work():
        try:
            stream = torch.cuda.Stream() if cuda else None
            for rows in parts:
                if stop.is_set():
                    return
                want = cache_row[rows]
                order = np.argsort(want, kind="stable")
                got = np.asarray(pix[want[order]])
                back = np.empty_like(order)
                back[order] = np.arange(len(order))
                x = torch.from_numpy(got[back])
                y = torch.from_numpy(labels[rows])
                if cuda:
                    with torch.cuda.stream(stream):
                        x = x.pin_memory().to(DEV, non_blocking=True)
                        y = y.to(DEV, non_blocking=True)
                    stream.synchronize()
                put((rows, x, y))
        except BaseException as e:  # noqa: BLE001  handed to the consumer, which raises it
            put(e)
        finally:
            put(None)

    th = threading.Thread(target=work, name="reader-prefetch", daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            rows, x, y = item
            if cuda:
                x.record_stream(torch.cuda.current_stream())
                y.record_stream(torch.cuda.current_stream())
            yield rows, x, y
    finally:
        stop.set()
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        th.join(timeout=10)


def to_float(x, px):
    """uint8 (B, H, W) on the card to float (B, 1, px, px) in [0, 1]."""
    x = x.float().div_(255.0).unsqueeze(1)
    if px != x.shape[-1]:
        x = F.interpolate(x, size=(px, px), mode="area")
    return x


def mean_sd(pix, cache_rows, px, g):
    """Mean and sd (0-1 units) of 2,000 training pictures, summed in float64 a chunk at a time."""
    pick = np.sort(g.choice(cache_rows, size=min(2000, len(cache_rows)), replace=False))
    s = s2 = 0.0
    n = 0
    for a in range(0, len(pick), 250):
        x = to_float(torch.from_numpy(np.asarray(pix[pick[a:a + 250]])), px).double()
        s += float(x.sum())
        s2 += float(x.square().sum())
        n += x.numel()
    m = s / n
    sd = max(((s2 - n * m * m) / (n - 1)) ** 0.5, 1e-3)
    return torch.tensor(m, dtype=torch.float32, device=DEV), torch.tensor(sd, dtype=torch.float32, device=DEV)


def predict(net, pix, cache_row, labels, rows, mean, sd, batch, px):
    """Logits for `rows`, in `rows` order."""
    net.eval()
    out = np.zeros((len(rows), len(datasets.LABELS)), np.float32)
    parts = [rows[a:a + batch] for a in range(0, len(rows), batch)]
    at = 0
    with torch.no_grad():
        for sl, x, _y in prefetch(pix, cache_row, labels, parts):
            x = (to_float(x, px) - mean) / sd
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                z = net(x)
            out[at:at + len(sl)] = z.float().cpu().numpy()
            at += len(sl)
    return out


def mean_auroc(logit, truth):
    v = [xd.auroc(logit[:, i], truth[:, i].astype(int)) for i in range(truth.shape[1])]
    return float(np.nanmean(v)), v


def held_out_logits(net, pix, cache_row, labels, te, mean, sd, batch, settings):
    """A fold's held-out logits: the one code path both a trained fold and a resumed one use, so
    their numbers are identical."""
    return predict(net, pix, cache_row, labels, te, mean, sd, batch, int(settings.get("size", pix.shape[-1])))


def distill_loss(out, y, teacher_logit, w, loss_fn):
    """(1 - w) x the usual loss on the labels + w x BCE against the teacher's probabilities (no pos_weight)."""
    return (1 - w) * loss_fn(out, y) + w * F.binary_cross_entropy_with_logits(out, torch.sigmoid(teacher_logit))


def disputed_cells(y, logit):
    """Confident learning's dispute rule over out-of-fold logits, per output column, over all rows given.
    With p = sigmoid(logit), t1 = mean p over the label-1 rows, t0 = mean (1 - p) over the label-0 rows:
    a label-1 cell is disputed when p <= 1 - t0 and p < t1; a label-0 cell when p >= t1. (n, k) bool."""
    y = np.asarray(y).astype(int)
    p = 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=np.float64)))
    out = np.zeros(y.shape, bool)
    for j in range(y.shape[1]):
        pos, neg = y[:, j] == 1, y[:, j] == 0
        t1 = p[pos, j].mean() if pos.any() else 1.0
        t0 = (1 - p[neg, j]).mean() if neg.any() else 1.0
        out[:, j] = np.where(pos, (p[:, j] <= 1 - t0) & (p[:, j] < t1), p[:, j] >= t1) & (pos | neg)
    return out


def masked_loss(per_cell, disputed):
    """Mean of the per-cell losses over the cells that are not disputed (zero kept cells: a zero loss)."""
    keep = (~disputed).to(per_cell.dtype)
    return (per_cell * keep).sum() / keep.sum().clamp(min=1.0)


def clean_first_mask(path, df, tr, train_labels):
    """(len(df), 15) bool, True where a cell is disputed, for the rows `tr` only (others False). Refuses when a
    training picture is missing from the npz or its labels there differ from `train_labels`'s."""
    with np.load(path) as z:
        image, y, logit = z["image"].astype(str), z["y"], z["logit"]
    disp = disputed_cells(y, logit)
    at = {n: i for i, n in enumerate(image)}
    rows = np.array([at.get(n, -1) for n in df.image.to_numpy(str)[tr]], dtype=np.int64)
    if (rows < 0).any():
        raise SystemExit("clean_first: %d training pictures are not in %s" % (int((rows < 0).sum()), path))
    if (y[rows].astype(int) != np.asarray(train_labels)[tr].astype(int)).any():
        raise SystemExit("clean_first: %s holds labels that differ from the ones this run trains on" % path)
    full = np.zeros((len(df), disp.shape[1]), bool)
    full[tr] = disp[rows]
    return full


def grow_heart_table(df, size):
    """(eligible, box, ctr) over df's rows for grow_heart. `box` is N x 4 (cx, hw, top, bottom) of CheXmask's heart
    in fractions of the picture's side; `ctr` is CheXmask's CTR (chexmask_ctr's rule). A row is eligible when NIH
    labels it No Finding, CheXmask's Dice is at least GROW_MIN_DICE, its landmarks parsed, and the cached picture
    is square (a letterboxed one would put the landmarks off the cache's frame).

    The landmark loader lives in tools/chexmask_ctr.py: tools/chest_outline.py imports this file, so it cannot be
    imported from here (2026-10-04)."""
    tools = os.path.join(ROOT, "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    import chexmask_ctr as cc
    image, lm, dice = cc.load_landmarks()
    at = {n: i for i, n in enumerate(image)}
    src = np.array([at.get(n, -1) for n in df.image.to_numpy(str)], dtype=np.int64)
    have = src >= 0
    pts = np.full((len(df), cc.N_POINTS, 2), np.nan, np.float64)
    pts[have] = lm[src[have]].reshape(-1, cc.N_POINTS, 2)
    d = np.full(len(df), np.nan)
    d[have] = dice[src[have]]
    with open(prepare.cache_paths(size)[1], encoding="utf-8") as fh:
        shape = json.load(fh)["source_shape"]
    square = np.array([str(n) in shape and shape[str(n)][0] == shape[str(n)][1] for n in df.image])
    heart = pts[:, cc.N_RIGHT + cc.N_LEFT:]
    box = np.stack([(heart[..., 0].max(1) + heart[..., 0].min(1)) / 2, (heart[..., 0].max(1) - heart[..., 0].min(1)) / 2,
                    heart[..., 1].min(1), heart[..., 1].max(1)], axis=1)
    ctr = np.array([cc.ctr_from_points(p) if np.isfinite(p).all() else np.nan for p in pts])
    no_finding = ~(df[datasets.FINDINGS].to_numpy() > 0).any(axis=1)
    eligible = no_finding & (d >= GROW_MIN_DICE) & np.isfinite(box).all(axis=1) & np.isfinite(ctr) & square
    return eligible, box, ctr


def grow_draw(g, eligible, ctr, p):
    """Which pictures of a batch grow, and by how much: (positions in the batch, s for each). `eligible` and `ctr`
    are the batch's rows'. Always draws the same numbers from `g` for a batch of this size, whatever is eligible."""
    n = len(eligible)
    pick = g.random(n) < p
    s = g.uniform(GROW_RANGE[0], GROW_RANGE[1], n)
    c = np.where(np.isfinite(ctr) & (ctr > 0), ctr, 1.0)
    s = np.where(c * s < GROW_MIN_CTR, GROW_CTR_TARGET / c, s)
    return np.flatnonzero(eligible & pick & (s <= GROW_MAX_S)), s


def grow_heart_grid(h, w, cx, hw, top, bottom, s, device):
    """The sampling grid (B, h, w, 2) for `F.grid_sample` that stretches a heart sideways by s. cx, hw, top, bottom
    (pixels) and s are numbers or (B,) tensors. Output pixel x samples the input at cx + (x - cx) / s inside the
    grown heart (|x - cx| <= s hw); over the next hw of output the sampling slides linearly back to the identity,
    so the lung beside it is squeezed and everything beyond stays put. The stretch is full between top and
    bottom and fades linearly to nothing over GROW_FADE of the heart's height above and below. Rows keep their
    own y. Per row the map has slope 1/s, then s, then 1, so it never folds."""
    def col(v):
        return torch.as_tensor(v, dtype=torch.float32, device=device).reshape(-1, 1, 1)
    cx, hw, top, bottom, s = col(cx), col(hw), col(top), col(bottom), col(s)
    xs = torch.arange(w, dtype=torch.float32, device=device).add(0.5).reshape(1, 1, w)
    ys = torch.arange(h, dtype=torch.float32, device=device).add(0.5).reshape(1, h, 1)
    fade = (bottom - top) * GROW_FADE
    a = torch.minimum((ys - (top - fade)) / fade, ((bottom + fade) - ys) / fade).clamp(0, 1)
    se = 1 + (s - 1) * a                                    # (B, h, 1): this row's stretch
    d = xs - cx
    ad = d.abs()
    src = torch.where(ad <= se * hw, ad / se,
                      torch.where(ad <= se * hw + hw, hw + (ad - se * hw) * se, ad))
    sx = cx + torch.sign(d) * src
    gy = (ys / h * 2 - 1).expand_as(sx)
    return torch.stack([sx / w * 2 - 1, gy], dim=-1)


def grow_hearts(x, box, s):
    """x (B, 1, H, W) with each picture's heart stretched by s (B,); box (B, 4) is (cx, hw, top, bottom) in pixels.
    One bilinear grid_sample, border padding."""
    _, _, h, w = x.shape
    grid = grow_heart_grid(h, w, box[:, 0], box[:, 1], box[:, 2], box[:, 3], s, x.device)
    return F.grid_sample(x, grid.to(x.dtype), mode="bilinear", padding_mode="border", align_corners=False)


def grown_targets(y, idx):
    """A copy of y (B, labels) where the pictures at `idx` read Cardiomegaly 1 and Abnormal 1, the rest as they were."""
    y = y.clone()
    y[idx, datasets.LABELS.index("Cardiomegaly")] = 1
    y[idx, datasets.LABELS.index("Abnormal")] = 1
    return y


class SelftestStop(Exception):
    """Raised by selftest_stop to cut a fold short on purpose, right after a partial checkpoint is saved."""


def selftest_stop(epoch, batch):
    """RAIDIOLOGY_CKPT_SELFTEST_STOP=<epoch> stops after that epoch's partial save, =<epoch>:<batch> after a
    mid-epoch one (batches done). Unset, this does nothing."""
    want = os.environ.get("RAIDIOLOGY_CKPT_SELFTEST_STOP")
    if want and want == ("%d:%d" % (epoch, batch) if batch else str(epoch)):
        raise SelftestStop(want)


def partial_path(out, fold):
    return os.path.join(out, "fold%d.partial.pt" % fold)


def finish_fold(path, net, info, fingerprint, settings, partial=None):
    """Write the fold's final checkpoint, then drop its partial one."""
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("aux.")}     # the lung head is never kept
    torch.save({"state": state, "cfg": CFG, "info": info, "settings": settings,
                "fingerprint": fingerprint}, path)
    if partial:
        ckpt.clear_partial(partial)


def fold_split(df, fold, settings):
    """(tr, val, te, rng, n_out): row indices of the training pictures, the early-stop pictures and the held-out `fold`, the
    generator that picked the early-stop patients (training goes on drawing from it), and how many pictures the
    `exclude_folds` setting left out of training and early stopping (0 without it)."""
    ex = list(settings.get("exclude_folds") or [])
    if fold in ex:
        raise SystemExit("exclude_folds %s holds the held-out fold %d" % (ex, fold))
    fold_of = df.fold.to_numpy()
    tr_all = np.flatnonzero((fold_of != fold) & ~np.isin(fold_of, ex))
    n_out = int(np.isin(fold_of, ex).sum())
    te = np.flatnonzero(fold_of == fold)
    patients = np.array(sorted(set(df.patient.iloc[tr_all])))
    rng = np.random.default_rng(SEED + fold)
    held = set(patients[rng.permutation(len(patients))[:int(round(VAL_SHARE * len(patients)))]])
    is_held = df.patient.iloc[tr_all].isin(held).to_numpy()
    return tr_all[~is_held], tr_all[is_held], te, rng, n_out


def train_fold(df, pix, cache_row, labels, fold, epochs, batch, settings, train_labels=None, rad_rows=None,
               partial=None, fingerprint=None, lung=None, drr=None):
    """`train_labels` (default `labels`) is what the network trains and early-stops on; the held-out
    fold is always read out against `labels` by the caller.

    With extra=["drr-luna16"], `drr` is load_drr's (pictures, Nodule labels) (loaded here when None): each DRR row becomes
    a training row after df's (row len(df) + j), DRR_COPIES copies each an epoch, with only its Nodule cell in the loss
    (masked_loss, so that cell weighs what one NIH cell does). Early stopping, mean and sd, pos_weight and the held-out
    logits read df's rows only.

    With the `lung_aux` setting, `lung` is lung_table's (masks, eligible, src): the lung head's BCE x lung_aux joins the
    loss on the training pictures that are eligible, and its Dice on the eligible early-stop pictures is printed and kept
    in the returned info as lung_aux_dice. Reading never uses the head.

    With `partial` (a path) and `fingerprint`, fold{k}.partial.pt is saved after every epoch and, when
    core.ckpt.Every is due, mid-epoch; a partial with the same fingerprint is continued from its next
    epoch (or batch), with the same best weights and early-stop counter."""
    if train_labels is None:
        train_labels = labels
    px = int(settings.get("size", pix.shape[-1]))
    patience = int(settings.get("patience", CFG["patience"]))
    amp = bool(settings.get("amp")) and DEV.type == "cuda"
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    tr, val, te, rng, n_out = fold_split(df, fold, settings)
    if n_out:
        print("exclude_folds: %s: %d pictures of those folds left out of training"
              % (list(settings["exclude_folds"]), n_out), flush=True)
    mean, sd = mean_sd(pix, cache_row[tr], px, rng)
    aux_w = float(settings.get("lung_aux") or 0)
    net = build_net(settings, lung_aux=aux_w > 0).to(DEV)
    if aux_w > 0:
        print("lung_aux: %d of %d training pictures have a CheXmask lung mask (Dice >= %g), weight %g"
              % (int(lung[1][tr].sum()), len(tr), GROW_MIN_DICE, aux_w), flush=True)
    ema = None
    if settings.get("ema"):
        # 2026-10-04: idea 5 (docs/efficiency, trial 7): a running average of the weights and BatchNorm buffers,
        # updated after every optimiser step; early stopping scores it and it is what is kept. It touches no
        # generator and not the optimiser, so a run without ema draws and steps exactly as before.
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(float(settings["ema"])), use_buffers=True)
        print("ema: decay %g" % settings["ema"], flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = None
    if settings.get("pos_weight"):
        pos = train_labels[tr].sum(axis=0)
        pw = torch.tensor(np.clip((len(tr) - pos) / np.maximum(pos, 1.0), 1.0, 20.0),
                          dtype=torch.float32, device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    k_rad = settings.get("radiologist_weight")
    if k_rad:
        loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        w = np.where(rad_rows, float(k_rad), 1.0).astype(np.float32)
        print("radiologist_weight: %d of %d training pictures weighted %gx" % (int(rad_rows[tr].sum()), len(tr), k_rad),
              flush=True)
    clean_k, dispute, dispute_info = 0, None, None
    if settings.get("clean_first"):
        # 2026-10-04: idea 15 (docs/efficiency, trial 8): the first clean_epochs epochs ignore the cells an earlier
        # five-fold run's out-of-fold logits dispute. Draws nothing, so the random streams are untouched.
        loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        clean_k = int(settings["clean_epochs"])
        full = clean_first_mask(settings["clean_first"], df, tr, train_labels)
        dispute = torch.from_numpy(full).to(DEV)
        yt = np.asarray(train_labels)[tr].astype(int)
        d_tr = full[tr]
        dispute_info = {"label1": [int((d_tr[:, j] & (yt[:, j] == 1)).sum()) for j in range(yt.shape[1])],
                        "label0": [int((d_tr[:, j] & (yt[:, j] == 0)).sum()) for j in range(yt.shape[1])]}
        print("clean_first: %d of %d training cells disputed (%s), ignored for the first %d epochs"
              % (int(d_tr.sum()), d_tr.size, settings["clean_first"], clean_k), flush=True)
        for j, name in enumerate(datasets.LABELS):
            print("  %-20s label 1: %6d   label 0: %6d" % (name, dispute_info["label1"][j], dispute_info["label0"][j]),
                  flush=True)
    teacher, w_distill = None, None
    if settings.get("distill"):
        # 2026-10-04: idea 11 (docs/efficiency, trial 4): the shipped reading averages two nets, twice the reading
        # cost; a student taught by their average may keep the gain at one net's price.
        # "{fold}" in the path is this fold's number, so one setting serves a five-fold run: each fold has its
        # own teacher, which never read that fold's held-out pictures (2026-10-04, the chest five-fold run)
        t_path = settings["distill"].format(fold=fold)
        with np.load(t_path) as zt:
            t_image, t_logit = zt["image"].astype(str), zt["logits"].astype(np.float32)
            if "fold" in zt.files and int(zt["fold"]) != fold:
                raise SystemExit("distill: %s is fold %d's teacher, not fold %d's" % (t_path, int(zt["fold"]), fold))
        at = {n: i for i, n in enumerate(t_image)}
        names = df.image.to_numpy(str)
        t_rows = np.array([at.get(n, -1) for n in names], dtype=np.int64)
        for what, idx in (("training", tr), ("early-stop", val)):
            if (t_rows[idx] < 0).any():
                raise SystemExit("distill: %d %s pictures have no teacher logit in %s"
                                 % (int((t_rows[idx] < 0).sum()), what, t_path))
        full = np.zeros((len(df), t_logit.shape[1]), np.float32)
        have = t_rows >= 0
        full[have] = t_logit[t_rows[have]]
        teacher = torch.from_numpy(full).to(DEV)
        w_distill = float(settings["distill_weight"])
        print("distill: %d teacher logits from %s, weight %g" % (int(have.sum()), t_path, w_distill),
              flush=True)
    p_grow = settings.get("grow_heart")
    if p_grow:
        # 2026-10-04: idea 12 (docs/efficiency, trial 5). Its own generator, so a run without grow_heart draws
        # exactly what it drew before. Eligibility is NIH's labels (`labels`), only over the training rows.
        # `box` is in fractions of the side and scales to whatever size the network trains at.
        eligible, box, ctr = grow_heart_table(df, pix.shape[-1])
        eligible[~np.isin(np.arange(len(df)), tr)] = False
        g_grow = np.random.default_rng(SEED + 200 + fold)
        print("grow_heart: %d of %d training pictures eligible (No Finding, CheXmask Dice >= %g), p %g"
              % (int(eligible.sum()), len(tr), GROW_MIN_DICE, p_grow), flush=True)
    tr_ep, pix_tr, row_tr, y_tr, drr_off = tr, pix, cache_row, train_labels, None
    if settings.get("extra"):
        # 2026-10-06: idea 43 step 2 (docs/efficiency). DRR rows are virtual rows after df's; only the training prefetch and
        # loss know them. They draw nothing of their own: the epoch's one permutation just has more rows.
        d_pix, d_y = load_drr() if drr is None else drr
        k_drr, j_nod = len(d_pix), datasets.LABELS.index("Nodule")
        d_rows = len(df) + np.arange(k_drr, dtype=np.int64)
        d_lab = np.zeros((k_drr, train_labels.shape[1]), np.float32)
        d_lab[:, j_nod] = d_y
        pix_tr = ExtraPixels(pix, d_pix)
        row_tr = np.concatenate([cache_row, pix.shape[0] + np.arange(k_drr, dtype=np.int64)])
        y_tr = np.concatenate([train_labels, d_lab])
        tr_ep = np.concatenate([tr, np.tile(d_rows, DRR_COPIES)])
        drr_off = torch.zeros(y_tr.shape, dtype=torch.bool, device=DEV)         # True: the cell is left out of the loss
        drr_off[len(df):] = True
        drr_off[len(df):, j_nod] = False
        loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        print("extra drr-luna16: %d DRR rows (%d Nodule 1, %d Nodule 0), %d copies each, %d rows an epoch beside %d NIH rows"
              % (k_drr, int(d_y.sum()), int(k_drr - d_y.sum()), DRR_COPIES, k_drr * DRR_COPIES, len(tr)), flush=True)
    g_cpu = np.random.default_rng(SEED + 100 + fold)
    g_dev = torch.Generator(device=DEV).manual_seed(SEED + fold)
    pace = polite.Pace()
    print("fold %d: train %d, early-stop %d, predict %d images" % (fold, len(tr), len(val), len(te)), flush=True)
    best, best_state, since, took = -1.0, None, 0, []
    start_ep, skip, order_saved, total0, grew0, ep_prior = 1, 0, None, 0.0, 0, 0.0
    every = ckpt.Every()

    def save_part(epoch, batch_done, stopped=False, order=None, total=0.0, grew=0, ep_secs=0.0):
        """Everything the fold needs to continue after `epoch` (batch_done == 0) or `batch_done` batches into it."""
        ckpt.save_partial(partial, {
            "state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "epoch": epoch, "batch": batch_done, "stopped": stopped, "order": order, "total": total, "grew": grew,
            "epoch_seconds_so_far": ep_secs, "best": best, "best_state": best_state, "since": since, "took": took,
            "seconds": sum(took) + ep_secs,
            "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if DEV.type == "cuda" else 0.0,
            "g_cpu": g_cpu.bit_generator.state, "g_dev": g_dev.get_state(),
            "g_grow": g_grow.bit_generator.state if p_grow else None, "rng": ckpt.rng_state(),
            "fingerprint": fingerprint})
        every.mark()
        selftest_stop(epoch, batch_done)

    ck = ckpt.load_partial(partial, fingerprint) if partial else None
    if ck is not None and 0 < ck["epoch"] <= epochs:
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        best, since, took = ck["best"], ck["since"], list(ck["took"])
        best_state = None if ck["best_state"] is None else {k: t.to(DEV) for k, t in ck["best_state"].items()}
        g_cpu.bit_generator.state = ck["g_cpu"]
        g_dev.set_state(ck["g_dev"].cpu())
        if p_grow:
            g_grow.bit_generator.state = ck["g_grow"]
        ckpt.set_rng_state(ck["rng"])
        if ck["batch"]:                                     # cut mid-epoch: carry on with the same order and batch
            start_ep, skip, order_saved = ck["epoch"], ck["batch"], ck["order"]
            total0, grew0, ep_prior = ck["total"], ck["grew"], ck["epoch_seconds_so_far"]
        else:
            start_ep = epochs + 1 if ck["stopped"] else ck["epoch"] + 1
        print("fold %d: resumed mid-fold at epoch %d of %d%s (partial checkpoint)"
              % (fold, min(start_ep, epochs), epochs, ", batch %d" % skip if skip else ""), flush=True)
    del ck
    for ep in range(start_ep, epochs + 1):
        t0 = time.time() - ep_prior
        net.train()
        order = order_saved if order_saved is not None else tr_ep[g_cpu.permutation(len(tr_ep))]
        all_parts = [order[a:a + batch] for a in range(0, len(order), batch)]
        parts = all_parts[skip:]
        total = torch.full((), total0, dtype=torch.float64, device=DEV)
        began = time.time()
        grew = grew0
        done_b = skip
        skip, order_saved, total0, grew0, ep_prior = 0, None, 0.0, 0, 0.0
        for rows, x, y in prefetch(pix_tr, row_tr, y_tr, parts):
            x = to_float(x, px)
            if p_grow:
                pick, s = grow_draw(g_grow, eligible[rows], ctr[rows], p_grow)
                if len(pick):
                    sel = torch.from_numpy(pick).to(DEV)
                    x[sel] = grow_hearts(x[sel], torch.from_numpy(box[rows[pick]] * px), torch.from_numpy(s[pick]))
                    y = grown_targets(y, sel)
                    grew += len(pick)
            if aux_w > 0:
                g_state = g_dev.get_state()                         # augment_masks replays its draws from here
            x = (xd.augment(x, g_dev, settings) - mean) / sd
            if aux_w > 0:
                lm, lung_ok, lung_src = lung
                use = torch.from_numpy(lung_ok[rows]).to(DEV)
                m = torch.from_numpy(np.asarray(lm[np.where(lung_ok[rows], lung_src[rows], 0)])).to(DEV).float() / 255.0
                m = augment_masks(m, g_state, DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                if aux_w > 0:
                    out_all, aux_out = net.forward_both(x)
                    out_all = out_all.float()
                else:
                    out_all = net(x).float()
                if drr_off is not None:
                    loss = masked_loss(loss_none(out_all, y), drr_off[torch.from_numpy(rows).to(DEV)])
                elif dispute is not None and ep <= clean_k:
                    loss = masked_loss(loss_none(out_all, y), dispute[torch.from_numpy(rows).to(DEV)])
                elif k_rad:
                    per = loss_none(out_all, y).mean(dim=1)
                    wb = torch.from_numpy(w[rows]).to(DEV)
                    loss = (per * wb).sum() / wb.sum()
                elif teacher is not None:
                    loss = distill_loss(out_all, y, teacher[torch.from_numpy(rows).to(DEV)], w_distill, loss_fn)
                else:
                    loss = loss_fn(out_all, y)
                if aux_w > 0:
                    loss = loss + aux_w * lung_aux_loss(aux_out, m, use)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update_parameters(net)
            total += loss.detach().double() * len(rows)
            pace.rest(time.time() - began)
            began = time.time()
            done_b += 1
            if partial and done_b < len(all_parts) and every.due():
                save_part(ep, done_b, order=order, total=float(total), grew=grew, ep_secs=time.time() - t0)
        sched.step()
        scored = ema.module if ema is not None else net
        v, _ = mean_auroc(predict(scored, pix, cache_row, labels, val, mean, sd, batch, px), train_labels[val])
        took.append(time.time() - t0)
        print("  epoch %2d  loss %.4f  early-stop auroc %.4f  %.0f s%s"
              % (ep, float(total) / len(tr_ep), v, took[-1], "  grew %d hearts" % grew if p_grow else ""), flush=True)
        stop_now = False
        if v > best + 1e-4:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in scored.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                print("  stopping: no better early-stop AUROC for %d epochs" % patience, flush=True)
                stop_now = True
        if partial:
            save_part(ep, 0, stopped=stop_now)
        if stop_now:
            break
    if best_state is None:
        raise SystemExit("fold %d: no epoch had a defined early-stop AUROC" % fold)
    net.load_state_dict(best_state)
    dice_lung = None
    if aux_w > 0:
        dice_lung, n_dice = lung_dice(net, pix, cache_row, labels, val, lung, mean, sd, batch, px)
        print("lung_aux: the head's mean Dice on the %d early-stop pictures with a mask: %.4f" % (n_dice, dice_lung), flush=True)
    logit = held_out_logits(net, pix, cache_row, labels, te, mean, sd, batch, settings)
    info = {"fold": fold, "epochs": len(took), "early_stop_auroc": best, "mean": float(mean), "sd": float(sd),
            "train_images": len(tr), "early_stop_images": len(val), "predicted_images": len(te),
            "seconds": round(sum(took), 1), "epoch_seconds": [round(t, 2) for t in took]}
    if dice_lung is not None:
        info["lung_aux_dice"] = dice_lung
    if drr_off is not None:
        info["extra"] = {"drr_rows": k_drr, "drr_positives": int(d_y.sum()), "copies": DRR_COPIES}
    if dispute_info is not None:
        info.update({"clean_first": settings["clean_first"], "clean_epochs": clean_k, "disputed": dispute_info})
    return net, info, te, logit


def check_settings(settings):
    unknown = sorted(set(settings) - set(SETTINGS))
    if unknown:
        raise SystemExit("unknown setting(s) %s: train takes %s" % (", ".join(unknown), ", ".join(sorted(SETTINGS))))
    if "radiologist_weight" in settings:
        k = settings["radiologist_weight"]
        if not settings.get("radiologist_labels"):
            raise SystemExit("radiologist_weight needs radiologist_labels=1")
        if isinstance(k, bool) or not isinstance(k, (int, float)) or not k > 0:
            raise SystemExit("radiologist_weight must be a number above 0, got %r" % (k,))
    if "grow_heart" in settings:
        p = settings["grow_heart"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p <= 1:
            raise SystemExit("grow_heart must be a number in (0, 1], got %r" % (p,))
    if "ema" in settings:
        d = settings["ema"]
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 < d < 1:
            raise SystemExit("ema must be a number in (0, 1), got %r" % (d,))
    if "crop" in settings and (not isinstance(settings["crop"], str) or not settings["crop"]):
        raise SystemExit("crop must be the name of a crop cache (e.g. \"lungs0\"), got %r" % (settings["crop"],))
    if "cache_size" in settings:
        n = settings["cache_size"]
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise SystemExit("cache_size must be a whole number above 0 (e.g. 512), got %r" % (n,))
        if "crop" in settings:
            raise SystemExit("cache_size cannot be combined with crop (the crop cache is in nih_384's rows)")
        missing = [p for p in prepare.cache_paths(n) if not os.path.exists(p)]
        if missing:
            raise SystemExit("cache_size %d: %s does not exist; build it with: %s"
                             % (n, ", ".join(missing), prepare.build_command(n)))
        s = settings.get("size")
        if s is not None and (isinstance(s, bool) or not isinstance(s, (int, float)) or s > n):
            raise SystemExit("size %r is larger than cache_size %d (size only shrinks a picture)" % (s, n))
    if "widths" in settings:
        w = settings["widths"]
        if not isinstance(w, list) or len(w) != 5 or any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in w):
            raise SystemExit("widths must be a list of exactly 5 positive whole numbers (e.g. [64, 128, 256, 512, 512]), got %r" % (w,))
    if "patience" in settings and (isinstance(settings["patience"], bool) or not isinstance(settings["patience"], int)
                                   or settings["patience"] < 1):
        raise SystemExit("patience must be a positive whole number of epochs, got %r" % (settings["patience"],))
    if "renderings" in settings and (isinstance(settings["renderings"], bool) or settings["renderings"] not in (1, 3)):
        raise SystemExit("renderings must be 1 or 3, got %r" % (settings["renderings"],))
    if "lung_aux" in settings:
        w = settings["lung_aux"]
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not w >= 0:
            raise SystemExit("lung_aux must be a number, 0 or above (e.g. 0.1), got %r" % (w,))
        if w > 0 and "crop" in settings:
            raise SystemExit("lung_aux cannot be combined with crop (the masks are in the whole picture's frame)")
        if w > 0 and not (os.path.exists(lung_masks.PATH) and os.path.exists(lung_masks.INDEX)):
            raise SystemExit("lung_aux: %s does not exist; build it with: %s" % (lung_masks.PATH, lung_masks.build_command()))
    if "exclude_folds" in settings:
        ex = settings["exclude_folds"]
        if not isinstance(ex, list) or any(isinstance(f, bool) or not isinstance(f, int) or not 0 <= f <= 4 for f in ex):
            raise SystemExit("exclude_folds must be a list of fold numbers 0-4, got %r" % (ex,))
    if "extra" in settings:
        ex = settings["extra"]
        if not isinstance(ex, list) or any(not isinstance(n, str) for n in ex):
            raise SystemExit("extra must be a list of training-set names (%s), got %r" % (", ".join(EXTRA_NAMES), ex))
        bad = sorted(set(n for n in ex if n not in EXTRA_NAMES))
        if bad:
            raise SystemExit("extra: unknown name(s) %s; the only one is %s" % (", ".join(map(repr, bad)), ", ".join(EXTRA_NAMES)))
        if len(set(ex)) != len(ex):
            raise SystemExit("extra names a training set twice: %r" % (ex,))
        if ex:
            clash = [k for k in ("crop", "lung_aux", "grow_heart", "distill", "clean_first", "radiologist_labels") if settings.get(k)]
            if clash:
                raise SystemExit("extra cannot be combined with %s (not needed, and refusing keeps the change small)" % ", ".join(clash))
            if settings.get("cache_size", prepare.SIZE) != prepare.SIZE:
                raise SystemExit("extra needs the %d px cache (the DRRs are %d px), got cache_size %r"
                                 % (prepare.SIZE, prepare.SIZE, settings["cache_size"]))
            missing = [p for p in drr_paths() if not os.path.exists(p)]
            if missing:
                raise SystemExit("extra drr-luna16: %s does not exist; build it with: python modalities/chest_xray/eval/drr_probe.py render"
                                 % ", ".join(missing))
    if "distill_weight" in settings and "distill" not in settings:
        raise SystemExit("distill_weight needs distill (the path of a chest_teacher npz)")
    if "distill" in settings:
        w = settings.get("distill_weight")
        if not isinstance(settings["distill"], str) or not settings["distill"]:
            raise SystemExit("distill must be the path of a chest_teacher npz, got %r" % (settings["distill"],))
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0 < w <= 1:
            raise SystemExit("distill needs distill_weight, a number in (0, 1], got %r" % (w,))
        if "radiologist_weight" in settings:
            raise SystemExit("distill cannot be combined with radiologist_weight")
    if "clean_epochs" in settings and "clean_first" not in settings:
        raise SystemExit("clean_epochs needs clean_first (the path of an out-of-fold logits npz)")
    if "clean_first" in settings:
        c, k = settings["clean_first"], settings.get("clean_epochs")
        if not isinstance(c, str) or not c:
            raise SystemExit("clean_first must be the path of an out-of-fold logits npz, got %r" % (c,))
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise SystemExit("clean_first needs clean_epochs, a whole number above 0, got %r" % (k,))
        if settings.get("radiologist_labels"):
            raise SystemExit("clean_first cannot be combined with radiologist_labels")
        if "distill" in settings:
            raise SystemExit("clean_first cannot be combined with distill")
        if not os.path.isfile(c):
            raise SystemExit("clean_first: %s does not exist" % c)


def selftest():
    """Processor only: no data files, no card. Prints 'selftest ok'; a failure raises."""
    # (a) the dispute rule on a hand-made table. Column 0: p = .9 .8 .3 (y 1 1 1) and .1 .6 (y 0 0)
    p0 = np.array([.9, .8, .3, .1, .6])
    y0 = np.array([1, 1, 1, 0, 0])
    # column 1: p = .7 .2 (y 1 1), .4 .05 .3 (y 0 0 0)
    p1 = np.array([.7, .2, .4, .05, .3])
    y1 = np.array([1, 1, 0, 0, 0])
    y = np.stack([y0, y1], axis=1)
    p = np.stack([p0, p1], axis=1)
    got = disputed_cells(y, np.log(p / (1 - p)))
    # col 0: t1 = (.9+.8+.3)/3 = .6667, t0 = (.9+.4)/2 = .65; label 1 disputed if p <= .35 and < .6667 -> only .3;
    #        label 0 disputed if p >= .6667 -> none (.6 is below)
    # col 1: t1 = .45, t0 = (.6+.95+.7)/3 = .75; label 1 disputed if p <= .25 and < .45 -> .2;
    #        label 0 disputed if p >= .45 -> none (.4 is below)
    want = np.zeros((5, 2), bool)
    want[2, 0] = True
    want[1, 1] = True
    assert (got == want).all(), got
    # a label-0 cell above t1 is disputed: col 0 with y = 1 1 0 0 and p = .5 .5 .9 .1 -> t1 = .5, the .9 is disputed
    g2 = disputed_cells(np.array([[1], [1], [0], [0]]), np.log(np.array([[.5], [.5], [.9], [.1]]) / np.array([[.5], [.5], [.1], [.9]])))
    assert g2[:, 0].tolist() == [False, False, True, False], g2
    # (b) the masked loss
    torch.manual_seed(0)
    out, tgt = torch.randn(6, 3), torch.randint(0, 2, (6, 3)).float()
    per = F.binary_cross_entropy_with_logits(out, tgt, reduction="none")
    dis = torch.zeros(6, 3, dtype=torch.bool)
    dis[0, 1] = dis[3, 0] = dis[3, 2] = dis[5, 1] = True
    assert torch.allclose(masked_loss(per, dis), per[~dis].mean())
    assert torch.allclose(masked_loss(per, torch.zeros_like(dis)), F.binary_cross_entropy_with_logits(out, tgt))
    # (b2) the default channels-last layout: same logits, weights in the other memory order, a bad value refused
    was = os.environ.get("RAIDIOLOGY_CHEST_LAYOUT")
    try:
        os.environ.pop("RAIDIOLOGY_CHEST_LAYOUT", None)
        assert chest_layout() == "channels_last"
        os.environ["RAIDIOLOGY_CHEST_LAYOUT"] = "plain"
        assert chest_layout() == "plain"
        os.environ["RAIDIOLOGY_CHEST_LAYOUT"] = "nope"
        try:
            chest_layout()
            raise AssertionError("chest_layout accepted 'nope'")
        except SystemExit:
            pass
        os.environ["RAIDIOLOGY_CHEST_LAYOUT"] = "plain"
        torch.manual_seed(0)
        plain_net = ChestNet([4, 4, 4, 4, 4], 0.3, 2, 15).eval()
        os.environ["RAIDIOLOGY_CHEST_LAYOUT"] = "channels_last"
        torch.manual_seed(0)
        cl_net = ChestNet([4, 4, 4, 4, 4], 0.3, 2, 15)
        cl_net.load_state_dict(plain_net.state_dict())
        cl_net.eval()
        xin = torch.randn(3, 1, 64, 64)
        with torch.no_grad():
            assert torch.allclose(plain_net(xin), cl_net(xin), atol=1e-4)
        assert cl_net.trunk[0][0].weight.is_contiguous(memory_format=torch.channels_last)
    finally:
        if was is None:
            os.environ.pop("RAIDIOLOGY_CHEST_LAYOUT", None)
        else:
            os.environ["RAIDIOLOGY_CHEST_LAYOUT"] = was
    assert float(masked_loss(per, torch.ones_like(dis))) == 0.0
    # (c) validation
    here = os.path.abspath(__file__)

    def refused(s):
        try:
            check_settings(s)
        except SystemExit:
            return True
        return False
    assert refused({"clean_epochs": 3})
    assert refused({"clean_first": here, "clean_epochs": 3, "radiologist_labels": 1})
    assert refused({"clean_first": here, "clean_epochs": 3, "distill": "x.npz", "distill_weight": 0.5})
    assert refused({"clean_first": here}) and refused({"clean_first": here, "clean_epochs": 0})
    assert refused({"clean_first": here, "clean_epochs": True}) and refused({"clean_first": "", "clean_epochs": 3})
    assert refused({"clean_first": here + ".missing", "clean_epochs": 3})
    check_settings({"clean_first": here, "clean_epochs": 3})
    # (c2) exclude_folds: validation, and the split on a 3-fold table
    for bad in ("2", [5], [-1], [True], [1.5]):
        assert refused({"exclude_folds": bad}), bad
    check_settings({"exclude_folds": [2]})
    check_settings({"exclude_folds": []})
    assert "exclude_folds" in SETTINGS
    import pandas as pd
    t3 = pd.DataFrame({"fold": np.arange(12) % 3, "patient": np.arange(12) % 6})
    tr3, va3, te3, _, n3 = fold_split(t3, 0, {"exclude_folds": [2]})
    assert n3 == 4 and set(te3) == {0, 3, 6, 9}
    assert set(tr3) | set(va3) <= {1, 4, 7, 10} and len(tr3) + len(va3) == 4 and not set(tr3) & set(va3)
    tr3, va3, _, _, n3 = fold_split(t3, 0, {})
    assert n3 == 0 and set(tr3) | set(va3) == {1, 2, 4, 5, 7, 8, 10, 11}
    try:
        fold_split(t3, 2, {"exclude_folds": [2]})
        raise AssertionError("the held-out fold in exclude_folds was not refused")
    except SystemExit:
        pass
    # (d) a default cfg is unchanged: no new key, nothing new in the fingerprint's settings
    check_settings({})
    assert json.dumps({}, sort_keys=True) == "{}"
    assert {"clean_first", "clean_epochs"} <= set(SETTINGS) and "size" in SETTINGS and "ema" in SETTINGS
    # (e) cache_size: refusals, and a cfg without it reads nih_384 with the fingerprint's settings as before
    import shutil
    import tempfile
    assert "cache_size" in SETTINGS and cache_size_of({}) == 384 and cache_size_of({"size": 256}) == 384
    assert cache_size_of({"cache_size": 512}) == 512
    tmp = tempfile.mkdtemp()
    real_paths = prepare.cache_paths
    try:
        def fake_paths(size=prepare.SIZE):
            return os.path.join(tmp, "nih_%d.npy" % size), os.path.join(tmp, "nih_%d.json" % size)
        prepare.cache_paths = fake_paths
        for bad in ("512", 512.0, True, 0, -512, None):
            assert refused({"cache_size": bad}), bad
        assert refused({"cache_size": 512})                         # nih_512 does not exist in the folder
        for p in fake_paths(512):
            open(p, "w").close()
        check_settings({"cache_size": 512})
        check_settings({"cache_size": 512, "size": 256})
        check_settings({"cache_size": 512, "size": 512})
        assert refused({"cache_size": 512, "size": 600})
        assert refused({"cache_size": 512, "crop": "lungs0"})
        assert cache_identity(512) == {"nih_512.npy": [0, os.stat(fake_paths(512)[0]).st_mtime_ns],
                                       "nih_512.json": [0, os.stat(fake_paths(512)[1]).st_mtime_ns]}
    finally:
        prepare.cache_paths = real_paths
        shutil.rmtree(tmp, ignore_errors=True)
    check_settings({"size": 256})                                   # no cache_size: accepted, nothing added to settings
    # (f) renderings and lung_aux: validation, and a default network that is the recipe's
    assert "renderings" in SETTINGS and "lung_aux" in SETTINGS
    for bad in (2, 0, True, "3", 3.5):
        assert refused({"renderings": bad}), bad
    for bad in (-0.1, True, "0.1", None):
        assert refused({"lung_aux": bad}), bad
    check_settings({"renderings": 3})
    check_settings({"renderings": 1, "lung_aux": 0})
    assert refused({"lung_aux": 0.1, "crop": "lungs0"})
    real_lung = (lung_masks.PATH, lung_masks.INDEX)
    lung_masks.PATH = lung_masks.INDEX = os.path.join(tempfile.gettempdir(), "no_such_lung_masks.npy")
    try:
        assert refused({"lung_aux": 0.1})
    finally:
        lung_masks.PATH, lung_masks.INDEX = real_lung
    plain = build_net({})
    assert plain.trunk[0][0].weight.shape[1] == 1 and not any(k.startswith("aux.") for k in plain.state_dict())
    assert "widths" in SETTINGS and widths_of({}) == CFG["widths"] and widths_of({}, {"widths": [4] * 5}) == [4] * 5
    wide = {"widths": [8, 8, 16, 16, 16]}
    wnet = build_net(wide, 3)
    convs = [m.out_channels for m in wnet.trunk.modules() if isinstance(m, nn.Conv2d)]
    assert convs[-1] == 16 and 8 in convs and widths_of(wide, {"widths": [4] * 5}) == wide["widths"]
    wck = {"settings": wide, "cfg": dict(CFG), "state": wnet.state_dict()}
    build_net(wck["settings"], 3).load_state_dict(wck["state"])                  # the rebuild path of a saved version
    assert not any(isinstance(m, nn.Conv2d) and m.out_channels == 8 for m in build_net({}, 3).modules())
    for bad in ([8, 8, 16, 16], [8, 8, 16, 16, 0], [8, 8, 16, 16, 1.5], [8, 8, 16, 16, True], "x"):
        assert refused({"widths": bad})
    assert "patience" in SETTINGS
    for bad in (0, -1, 1.5, True, "3"):
        assert refused({"patience": bad})
    for good in (1, 24):
        assert not refused({"patience": good})
    variants_selftest()
    resume_selftest()
    print("selftest ok")


def _same(a, b):
    """Equal nested dicts, lists, tensors and numbers (tensors bit for bit)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    return a == b


def variants_selftest():
    """renderings=3 and lung_aux=0.1 on a tiny CPU fold each (about 15 s): finite losses, the 3-channel first layer, the lung
    head's finite Dice, a saved renderings=3 reader that loads from its settings and reads back the same logits, a saved
    lung_aux reader without the head that loads into the plain network, and the lung masks following the picture's flip
    and shift."""
    import contextlib
    import io
    import re
    import shutil
    import tempfile
    import pandas as pd
    n, epochs, bsz = 200, 2, 32
    g = np.random.default_rng(3)
    df = pd.DataFrame({"fold": np.arange(n) % 2, "patient": np.arange(n), "image": ["p%d.png" % i for i in range(n)]})
    pix = g.integers(0, 256, (n, 64, 64), dtype=np.uint8)
    labels = (g.random((n, len(datasets.LABELS))) < 0.4).astype(np.float32)
    cache_row = np.arange(n)
    # the three renderings of a picture: its own grey, ranks, and high-pass, the last two at zero mean and unit spread
    x = torch.rand(4, 1, 64, 64)
    r = add_renderings(x)
    assert r.shape == (4, 3, 64, 64) and torch.equal(r[:, :1], x)
    assert torch.allclose(r[:, 1:].mean(dim=(2, 3)), torch.zeros(4, 2), atol=1e-4)
    assert torch.allclose(r[:, 1:].std(dim=(2, 3)), torch.ones(4, 2), atol=1e-3)
    assert torch.equal(r[0, 1].flatten().argsort(), x[0, 0].flatten().argsort())      # the ranks keep the grey order
    flat = torch.full((1, 1, 64, 64), 0.5)
    assert float(gaussian_blur(flat, 3.0).sub(0.5).abs().max()) < 1e-6              # blurring keeps a flat picture flat
    # lung masks follow the picture's flip and affine (no colour jitter, so augment returns the geometry alone)
    saved_xd = dict(xd.CFG)
    xd.CFG.update({"rotate_deg": CFG["rotate_deg"], "translate": CFG["translate"], "jitter": 0.0})
    try:
        ax = torch.linspace(-1, 1, 48)
        blob = torch.exp(-((ax[None, :] - 0.3) ** 2 + (ax[:, None] + 0.2) ** 2) / 0.1)[None, None].repeat(6, 1, 1, 1)
        gen = torch.Generator().manual_seed(5)
        state = gen.get_state()
        want = F.interpolate(xd.augment(blob, gen, {}), size=24, mode="area")
        have = augment_masks(F.interpolate(blob, size=24, mode="area").repeat(1, 2, 1, 1), state, torch.device("cpu"))
        assert float((have[:, 0:1] - want).abs().max()) < 0.06 and float((have[:, 1:] - want).abs().max()) < 0.06
    finally:
        xd.CFG.clear()
        xd.CFG.update(saved_xd)
    tmp = tempfile.mkdtemp()
    saved_widths = CFG["widths"]
    CFG["widths"] = [4, 4, 4, 4, 4]

    def run(settings, **kw):
        buf = io.StringIO()
        torch.manual_seed(0)
        with contextlib.redirect_stdout(buf):
            res = train_fold(df, pix, cache_row, labels, 0, epochs, bsz, settings, **kw)
        losses = [float(v) for v in re.findall(r"loss (\S+)\s+early-stop", buf.getvalue())]
        assert len(losses) == epochs and all(math.isfinite(v) for v in losses), buf.getvalue()
        return res, buf.getvalue()

    try:
        # renderings=3: the first layer takes three channels; a saved reader loads from its own settings and reads the same logits
        s3 = {"renderings": 3}
        (net, info, te, logit), _out = run(s3)
        assert net.renderings == 3 and net.trunk[0][0].weight.shape[1] == 3
        path = os.path.join(tmp, "fold0.pt")
        finish_fold(path, net, info, {"selftest": 3}, s3)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        back = build_net(ck["settings"]).to(DEV)
        back.load_state_dict(ck["state"])
        mean = torch.tensor(ck["info"]["mean"], dtype=torch.float32, device=DEV)
        sd = torch.tensor(ck["info"]["sd"], dtype=torch.float32, device=DEV)
        again = held_out_logits(back, pix, cache_row, labels, te, mean, sd, bsz, ck["settings"])
        assert np.allclose(again, logit, atol=1e-5) and np.isfinite(again).all(), "a saved renderings=3 reader reads other logits"
        try:
            build_net({}).load_state_dict(ck["state"])
            raise AssertionError("a one-channel network took a renderings=3 reader's weights")
        except RuntimeError:
            pass
        # lung_aux=0.1 on synthetic masks (a lung-shaped block per side), about half the pictures eligible
        masks = np.zeros((n, 2, 24, 24), np.uint8)
        masks[:, 0, 4:20, 3:11] = 255
        masks[:, 1, 4:20, 13:21] = 255
        eligible = g.random(n) < 0.6
        lung = (masks, eligible, np.arange(n))
        sl = {"lung_aux": 0.1}
        (net, info, te, logit), out = run(sl, lung=lung)
        assert net.aux.weight.shape == (2, 4, 1, 1) and net.trunk[0][0].weight.shape[1] == 1
        assert math.isfinite(info["lung_aux_dice"]) and 0.0 <= info["lung_aux_dice"] <= 1.0 and "mean Dice" in out, out
        assert np.isfinite(logit).all()
        path = os.path.join(tmp, "fold1.pt")
        finish_fold(path, net, info, {"selftest": 4}, sl)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert not any(k.startswith("aux.") for k in ck["state"])
        build_net(ck["settings"]).load_state_dict(ck["state"])                      # the head is not kept: the plain network loads it
        out_h = torch.randn(4, 2, 4, 4)
        tgt = torch.ones(4, 2, 24, 24)
        assert math.isfinite(float(lung_aux_loss(out_h, tgt, torch.tensor([True, False, True, False]))))
        assert float(lung_aux_loss(out_h, tgt, torch.zeros(4, dtype=torch.bool))) == 0.0
    finally:
        CFG["widths"] = saved_widths
        shutil.rmtree(tmp, ignore_errors=True)


def resume_selftest():
    """Mid-fold resume on a tiny CPU fold (about 20 s): a fold cut after epoch 2, and another cut a batch into
    epoch 2, each continue from the partial checkpoint and end exactly where an uncut fold ends; only the remaining
    epochs run; the optimiser's state is the saved one; the partial is gone once the fold is written; a corrupt
    partial is renamed .bad and the fold starts fresh. Env RAIDIOLOGY_CKPT_SELFTEST_STOP cuts the fold."""
    import contextlib
    import copy
    import io
    import shutil
    import tempfile
    import pandas as pd
    n, epochs, bsz = 200, 5, 32                                     # 90 training pictures: 3 batches an epoch
    g = np.random.default_rng(1)
    df = pd.DataFrame({"fold": np.arange(n) % 2, "patient": np.arange(n), "image": ["p%d.png" % i for i in range(n)]})
    pix = g.integers(0, 256, (n, 64, 64), dtype=np.uint8)
    labels = (g.random((n, len(datasets.LABELS))) < 0.4).astype(np.float32)
    cache_row = np.arange(n)
    fp = {"selftest": 1, "fold": 0}
    tmp = tempfile.mkdtemp()
    part, final = os.path.join(tmp, "fold0.partial.pt"), os.path.join(tmp, "fold0.pt")
    saved_env = {k: os.environ.get(k) for k in ("RAIDIOLOGY_CKPT_SELFTEST_STOP", "RAIDIOLOGY_SAVE_EVERY_S",
                                                "RAIDIOLOGY_NO_RESUME")}
    saved_widths = CFG["widths"]
    spied = []
    real_load = torch.optim.AdamW.load_state_dict

    def spy(self, sd):
        real_load(self, sd)
        spied.append(copy.deepcopy(self.state_dict()))              # what the optimiser holds after the load (a copy: training goes on)

    def run(stop=None, every=None, use_partial=True):
        for k, v in (("RAIDIOLOGY_CKPT_SELFTEST_STOP", stop), ("RAIDIOLOGY_SAVE_EVERY_S", every)):
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                r = train_fold(df, pix, cache_row, labels, 0, epochs, bsz, {}, partial=part if use_partial else None,
                               fingerprint=fp if use_partial else None)
        except SelftestStop:
            r = None
        out = buf.getvalue()
        return r, out, sum(1 for ln in out.splitlines() if ln.startswith("  epoch "))

    torch.optim.AdamW.load_state_dict = spy
    try:
        os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
        CFG["widths"] = [4, 4, 4, 4, 4]
        torch.manual_seed(0)
        (net_full, info_full, _te, logit_full), _out, epochs_full = run(use_partial=False)
        assert epochs_full == info_full["epochs"] >= 4 and not os.path.exists(part)
        for stop, at_epoch, at_batch in (("2", 2, 0), ("2:1", 2, 1)):
            ckpt.clear_partial(part)
            del spied[:]
            torch.manual_seed(0)
            cut, _o, _n = run(stop=stop, every="0" if at_batch else None)
            assert cut is None and os.path.exists(part), "the fold was not cut after a partial save"
            saved = torch.load(part, map_location="cpu", weights_only=False)
            assert saved["epoch"] == at_epoch and saved["batch"] == at_batch and saved["fingerprint"] == fp
            assert len(saved["took"]) == (at_epoch - 1 if at_batch else at_epoch)   # epochs finished before the save
            assert (saved["best_state"] is not None) == (saved["best"] > -1.0)
            (net_b, info_b, _te, logit_b), out_b, epochs_b = run()
            want = "fold 0: resumed mid-fold at epoch %d of %d%s (partial checkpoint)" % (
                at_epoch + (0 if at_batch else 1), epochs, ", batch %d" % at_batch if at_batch else "")
            assert want in out_b, (want, out_b)
            assert spied and _same(spied[0], saved["opt"]), "the optimiser did not get the saved state"
            assert epochs_b == info_full["epochs"] - at_epoch + (1 if at_batch else 0), (epochs_b, info_full["epochs"])
            assert info_b["epochs"] == info_full["epochs"] and len(info_b["epoch_seconds"]) == info_full["epochs"]
            assert abs(info_b["early_stop_auroc"] - info_full["early_stop_auroc"]) < 1e-6, "best score differs"
            sd_full, sd_b = net_full.state_dict(), net_b.state_dict()
            assert all(torch.allclose(sd_full[k].float(), sd_b[k].float(), atol=1e-5, rtol=0) for k in sd_full), \
                "the resumed fold ended with other weights than the uncut one"
            assert np.allclose(logit_full, logit_b, atol=1e-4, equal_nan=True)
            finish_fold(final, net_b, info_b, fp, {}, part)
            assert os.path.exists(final) and not os.path.exists(part) and not os.path.exists(part + ".tmp")
            os.remove(final)
        # a corrupt partial: renamed .bad, the fold starts again from epoch 1
        with open(part, "wb") as f:
            f.write(b"not a checkpoint")
        torch.manual_seed(0)
        _r, out_c, epochs_c = run()
        assert os.path.exists(part + ".bad") and "resumed mid-fold" not in out_c and "WARNING" in out_c
        assert epochs_c == info_full["epochs"], "a fresh start trains every epoch"
        assert ckpt.load_partial(part, {**fp, "fold": 9}) is None   # another fingerprint: not used
    finally:
        del torch.optim.AdamW.load_state_dict                       # back to the inherited one
        CFG["widths"] = saved_widths
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def cache_identity(size=384):
    """The picture cache's size and mtime, not its hash: hashing the 12 GB memmap takes too long."""
    ident = {}
    for p in prepare.cache_paths(size):
        st = os.stat(p)
        ident[os.path.basename(p)] = [st.st_size, st.st_mtime_ns]
    return ident


def drr_paths():
    """(drr_384.npy, drr_index.csv) of drr_probe.py render, under <WORK>/eval/chest/drr-v1 (drr_probe.out_dir's folder; not imported
    from there, it puts the CT scan type's folders on the path)."""
    d = os.path.join(paths.WORK, "eval", "chest", "drr-v1")
    return os.path.join(d, "drr_384.npy"), os.path.join(d, "drr_index.csv")


def drr_identity():
    """The two DRR files' size and mtime, for the fingerprint of a run with extra=["drr-luna16"]."""
    return {os.path.basename(p): [os.stat(p).st_size, os.stat(p).st_mtime_ns] for p in drr_paths()}


def load_drr():
    """(pictures, labels) of the DRR training rows, in drr_index.csv's order: pictures uint8 (k, 384, 384) held in memory (about
    75 MB for 511), labels float32 (k,) = Nodule: 1 for the pos10 scans, 0 for the none ones. The small scans are left out."""
    import pandas as pd
    npy, csv = drr_paths()
    idx = pd.read_csv(csv)
    arr = np.load(npy, mmap_mode="r")
    if arr.dtype != np.uint8 or arr.shape != (len(idx), prepare.SIZE, prepare.SIZE):
        raise SystemExit("extra drr-luna16: %s is %s %s, not uint8 (%d, %d, %d) like %s"
                         % (npy, arr.dtype, arr.shape, len(idx), prepare.SIZE, prepare.SIZE, csv))
    grp = idx["group"].to_numpy(str)
    take = np.flatnonzero((grp == "pos10") | (grp == "none"))
    return np.ascontiguousarray(arr[take]), (grp[take] == "pos10").astype(np.float32)


class ExtraPixels:
    """The training rows' pictures: rows below len(base) come from `base` (the memmap), the ones after from `extra` (in memory).
    Only what prefetch asks of a memmap: pix[array of rows] -> uint8 (n, H, W)."""

    def __init__(self, base, extra):
        self.base, self.extra, self.n = base, extra, len(base)
        self.shape = (self.n + len(extra),) + tuple(base.shape[1:])

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        out = np.empty((len(rows),) + self.shape[1:], np.uint8)
        nih = rows < self.n
        if nih.any():
            out[nih] = np.asarray(self.base[rows[nih]])
        if (~nih).any():
            out[~nih] = self.extra[rows[~nih] - self.n]
        return out


def crop_pixels(name, df, folds):
    """The crop cache's memmap, in nih_384's row order so cache_row stays valid. Refuses a fold the cache
    did not box for and any development row it could not build: those rows are zeros there."""
    pix, _, _, meta = prepare.load_crop(name)
    missing = sorted(set(folds) - set(meta["folds"]))
    if missing:
        raise SystemExit("crop %s holds boxes for folds %s, not %s" % (name, meta["folds"], missing))
    blocked = set(meta["unreadable"]) | set(meta["not_built"])
    used = blocked & set(str(i) for i in df.image)
    if used:
        raise SystemExit("crop %s has no pixels for %d development pictures (e.g. %s)" % (name, len(used), sorted(used)[0]))
    return pix


def load_resumable(path, fingerprint):
    """The checkpoint at `path` if it holds a fingerprint equal to `fingerprint`, else None (also
    when RAIDIOLOGY_NO_RESUME is 1, or the file is missing or unreadable)."""
    if os.environ.get("RAIDIOLOGY_NO_RESUME") == "1" or not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location=DEV, weights_only=False)
    except Exception:
        return None
    return ck if isinstance(ck, dict) and ck.get("fingerprint") == fingerprint else None


def train(version, epochs, batch, folds=(0, 1, 2, 3, 4), settings=None):
    """Train the reader on the given folds. A restarted run resumes from the folds whose saved
    fingerprint matches this run's (RAIDIOLOGY_NO_RESUME=1 to retrain them all)."""
    settings = settings or {}
    check_settings(settings)
    # the recipe's own values are left out, so a run that names them is the run that does not
    settings = {k: v for k, v in settings.items() if not (k == "renderings" and v == 1) and not (k == "lung_aux" and v == 0)
                and not (k == "exclude_folds" and v == []) and not (k == "extra" and v == [])}
    for f in folds:
        if f in (settings.get("exclude_folds") or []):
            raise SystemExit("exclude_folds %s holds fold %d, which this run reads" % (settings["exclude_folds"], f))
    datasets.guard_test("dev")
    torch.manual_seed(SEED)
    card.require_card(DEV, torch)
    # 2026-10-03: leased before the data is loaded and before CUDA starts: waiting jobs held 2.1-6.9 GB doing
    # nothing, and at 23:21 the machine ran out of committed memory (a training job crashed)
    xd.limit_card_memory()
    xd.CFG.update({k: CFG[k] for k in ("rotate_deg", "translate", "jitter")})
    df, pix, cache_row, labels = table(cache_size_of(settings))
    if "crop" in settings:
        pix = crop_pixels(settings["crop"], df, folds)
    rad_rows = None
    if settings.get("radiologist_labels"):
        train_labels, rad_rows = radiologist_training_labels(df, labels)
    else:
        train_labels = labels
    lung = lung_table(df, cache_size_of(settings)) if settings.get("lung_aux") else None
    print("pictures: %d development images, kept on disk (%s)" % (len(df), pix.shape), flush=True)
    out = out_dir(version)
    os.makedirs(out, exist_ok=True)
    n_lab = len(datasets.LABELS)
    logit = np.full((len(df), n_lab), np.nan, np.float32)
    got_fold = np.full(len(df), -1, np.int8)
    infos = []
    with ledger.run("reader", "train v%d folds %s" % (version, "".join(str(k) for k in folds))) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "cfg": CFG, **({"settings": settings} if settings else {})})
        fingerprint = {"version": version, "epochs": epochs, "batch": batch, "settings": settings, "cfg": CFG,
                       "seed": SEED, "split_sha256": xd.sha256(splitmod.PATH), "cache": cache_identity(cache_size_of(settings)),
                       "dev_rows": int(len(df))}
        if "crop" in settings:      # only then, so a run without it keeps the fingerprint saved folds were resumed by
            st = os.stat(prepare.crop_paths(settings["crop"])[0])
            fingerprint["crop_cache"] = [st.st_size, st.st_mtime_ns]
        if settings.get("lung_aux"):    # the mask cache the lung head was taught from, named in the stamp
            fingerprint["lung_masks"] = {os.path.basename(p): [os.stat(p).st_size, os.stat(p).st_mtime_ns]
                                         for p in (lung_masks.PATH, lung_masks.INDEX)}
        if settings.get("extra"):       # only then: the DRR files the extra rows were read from
            fingerprint["drr_files"] = drr_identity()
        fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True, default=str))
        drr = load_drr() if settings.get("extra") else None     # in memory once for every fold
        for k in folds:
            path = os.path.join(out, "fold%d.pt" % k)
            ck = load_resumable(path, {**fingerprint, "fold": k})
            if ck is not None:
                net = build_net(settings, n_lab).to(DEV)
                net.load_state_dict(ck["state"])
                info = ck["info"]
                rows = np.flatnonzero(df.fold.to_numpy() == k)
                mean = torch.tensor(info["mean"], dtype=torch.float32, device=DEV)
                sd = torch.tensor(info["sd"], dtype=torch.float32, device=DEV)
                logit[rows], got_fold[rows] = held_out_logits(net, pix, cache_row, labels, rows, mean, sd, batch,
                                                              settings), k
                print("fold %d: resumed from %s (no training)" % (k, path), flush=True)
                infos.append(info)
                continue
            part = partial_path(out, k)
            net, info, rows, scores = train_fold(df, pix, cache_row, labels, k, epochs, batch, settings, train_labels,
                                                    rad_rows, partial=part, fingerprint={**fingerprint, "fold": k}, lung=lung, drr=drr)
            logit[rows], got_fold[rows] = scores, k
            finish_fold(path, net, info, {**fingerprint, "fold": k}, settings, part)
            infos.append(info)
            led.number("fold%d.seconds" % k, info["seconds"], "s").number("fold%d.epochs" % k, info["epochs"], "count")
            if DEV.type == "cuda":
                led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
                led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)  # what the cap counts
        done = got_fold >= 0
        per = [xd.auroc(logit[done, i], labels[done, i].astype(int)) for i in range(n_lab)]
        findings = float(np.nanmean(per[:14]))
        for name, a in zip(datasets.LABELS, per):
            led.number("auroc.%s" % name, a)
        led.number("auroc.mean14", findings).number("train_images", sum(i["train_images"] for i in infos), "count")
    # plain strings: an object array cannot be read back without pickle (compare.py refuses it)
    np.savez(os.path.join(out, "oof_logits.npz"), image=df.image.to_numpy(str)[done], patient=df.patient.to_numpy()[done],
             fold=got_fold[done], y=labels[done].astype(np.int8), logit=logit[done])
    meta = {"version": version, "cfg": CFG, "settings": settings, "seed": SEED, "folds": infos,
            "auroc": dict(zip(datasets.LABELS, per)), "auroc_mean14": findings,
            "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("\nout-of-fold AUROC (%d images, folds %s) -> %s" % (int(done.sum()), list(folds), out))
    for name, a in zip(datasets.LABELS, per):
        print("  %-20s %.4f" % (name, a))
    print("  %-20s %.4f" % ("mean of 14 findings", findings))
    print("  %-20s %.4f" % ("Abnormal", per[14]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, default=100)
    t.add_argument("--epochs", type=int, default=CFG["epochs"])
    t.add_argument("--batch", type=int, default=CFG["batch"])
    t.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="folds to train, 0 for the cheap trial")
    t.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(SETTINGS)))
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return
    if DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train(a.version, a.epochs, a.batch, tuple(a.folds),
          sets.parse(a.set))


if __name__ == "__main__":
    main()
