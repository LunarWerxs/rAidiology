"""The fracture detector: a small convolutional network trained from scratch on development folds
only, the candidate that has to beat the best baseline before it may go anywhere near the app.

One detector for both datasets, because a visitor's X-ray can be any body part: fold k trains on
both GRAZPEDWRI-DX's and FracAtlas's development units outside fold k and predicts both datasets'
fold k. Each dataset keeps its own decision threshold, cross-fitted, so the threshold applied to
fold k never saw fold k.

From scratch, no pretrained weights of any kind: ImageNet's terms are non-commercial and rAidiology
has to stay commercially clean (LICENSING.md). Five convolutional stages with a stride-2 stem,
global average pooling, then the pooled features with a two-value dataset one-hot appended and one
linear logit. The one-hot cannot leak the answer: it says which collection the image came from,
which the model would learn from the pictures anyway (a paediatric wrist does not look like a hip),
and both collections contain fractured and unfractured images, so knowing the collection tells you
nothing about this image's label beyond its prevalence. It is there so the shared trunk does not have
to waste capacity separating two very different image distributions.

Speed is a requirement, not a nicety: the owner's rule is that a run of this kind finishes inside an
hour, and ten minutes is the target. So the pixels are loaded once into a single uint8 tensor, on the
GPU when there is room for it, and every batch is an index into that tensor: no file reads, no worker
processes, no host-to-device copy per batch. Augmentation runs on the GPU as one batched affine warp.
On an RTX 4070 Ti that is about 25 to 60 seconds an epoch on roughly 13,000 development images, so a
fold stops after 10 to 20 epochs in two to eight minutes, and five folds fit the budget. Every fold's
wall time goes into the ledger (core/ledger.py), so the next pass can see what to attack.
torch.compile is deliberately not used: it needs Triton, which is not reliably present on Windows,
and a build that only works on one machine is worse than a slower one that always works.

Usage: python detector.py train [--version 1] [--epochs 25] [--batch 64] [--set key=<json> ...]
       python detector.py bench [--batch 64] [--set key=<json> ...]
       python detector.py predict <dev|test> [--version 1]
       python detector.py boxes [--size 384]
       python detector.py hotcell [--version 1]
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import card, ckpt, ledger, polite  # noqa: E402
from core import paths, sets  # noqa: E402
import boxes  # noqa: E402
import datasets  # noqa: E402
import preds  # noqa: E402
import prepare  # noqa: E402
import pseudo  # noqa: E402
import split as splitmod  # noqa: E402

def _device():
    """The GPU when there really is one. `is_available()` can still say yes on a machine whose
    devices have been hidden, and the saved weights hold CUDA tensors, so a wrong answer here is a
    crash on somebody else's laptop rather than a slow run. RAIDIOLOGY_DEVICE overrides it."""
    want = os.environ.get("RAIDIOLOGY_DEVICE", "").strip()
    if want:
        return torch.device(want)
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda")
    except Exception:  # noqa: BLE001  a broken driver must not stop a CPU run
        pass
    return torch.device("cpu")


DEV = _device()
FOLDS = 5
VAL_SHARE = 0.15
CFG = {"epochs": 25, "batch": 64, "lr": 3e-4, "weight_decay": 1e-4, "patience": 4,
       "widths": [32, 64, 128, 256, 256], "dropout": 0.3, "stem_stride": 2,
       "rotate_deg": 10.0, "translate": 0.05, "jitter": 0.10}


TRIAL_FROM = 100
PIN_LIMIT = 4 * 2 ** 30       # bytes of pictures pinned in the processor's memory at most (stack)


def model_dir(version):
    """The shipped models live beside the repository; an experiment (version 100 and up) lives under
    work/, so trying a recipe never leaves a model where shipped ones are kept."""
    if version >= TRIAL_FROM:
        return os.path.join(paths.WORK, "eval", "xray", "detector-v%d" % version, "model")
    return os.path.join(paths.MODELS, "xray-fracture-detector-v%d" % version)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Net(nn.Module):
    def __init__(self, widths, dropout, stem_stride, n_datasets, loc=False):
        super().__init__()
        stages, c_in = [], 1
        for i, c in enumerate(widths):
            stride = stem_stride if i == 0 else 1
            stages.append(nn.Sequential(
                nn.Conv2d(c_in, c, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2)))
            c_in = c
        self.trunk = nn.Sequential(*stages)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c_in + n_datasets, 1)
        # the box head (idea 125) exists only in a run that asks for it, so a run without one has the same weights as before
        self.loc = nn.Conv2d(widths[2], 1, 1) if loc else None

    def forward(self, x, which, loc=False):
        """The picture's logit; with `loc` also the box head's cell logits on the stride-16 map (the
        trunk's third stage, 24 x 24 at 384 px)."""
        if not loc:
            return self.logit_from(self.trunk(x), which)
        mid = self.trunk[:3](x)
        return self.logit_from(self.trunk[3:](mid), which), self.loc(mid).squeeze(1)

    def logit_from(self, feat, which):
        """The answer from the trunk's feature map, for a caller that already has the map and must
        not pay for the trunk twice (the findings file draws the map and reads the answer)."""
        return self.head(self.drop(torch.cat([feat.mean(dim=(2, 3)), which], dim=1))).squeeze(1)


def table(splits=("dev",), names=None, size=prepare.SIZE):
    """One row per image of the wanted splits, with where its pixels are in the stacked tensor.
    `names`: the collections, in order (their one-hot index); the shipped two without it. `size`:
    the picture cache's canvas (prepare.py), the 384 square without it."""
    s = splitmod.load()
    parts, arrays, base = [], {}, 0
    for di, d in enumerate(names or datasets.NAMES):
        arr, order, size = prepare.load(d, size)
        bad = prepare.unreadable(d, size)
        arrays[d] = (arr, order, size)
        for sp in splits:
            fr = splitmod.frame(d, sp, s)
            fr = fr.assign(dataset=d, dataset_index=di,
                           cache_row=[order[i] for i in fr.image],
                           readable=[i not in bad for i in fr.image])
            parts.append(fr[["dataset", "dataset_index", "image", "patient", "truth", "split",
                             "fold", "cache_row", "readable"]])
    df = pd.concat(parts, ignore_index=True)
    df["pseudo"] = False
    return number_rows(df), arrays, base


def number_rows(df):
    """Each readable row's place in the stacked pixel tensor (stack), in row order; -1 for an unreadable one."""
    df["stack_row"] = -1
    good = df[df.readable].index
    df.loc[good, "stack_row"] = np.arange(len(good))
    return df


def pseudo_rows(settings, size, names, pool, arrays):
    """Every readable pool picture of the pseudo csv, as table rows: pixels added to `arrays`, `pseudo` True, fold -1
    (no fold's held-out set or scoring holds one) and the teacher's p_fold0..4. Which ones teach, and their label,
    is picked per fold in train_fold from that fold's own net (pick_pseudo), so no fold learns from a net that
    trained on its held-out pictures. A pool picture takes the one-hot slot of pseudo.TEACHER_DATASET, so the net stays as wide as every reader expects."""
    frame = pseudo.read(settings["pseudo_csv"])
    missing = [c for c in pseudo.FOLD_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit("pseudo_csv has no %s: rescore with pseudo.py score" % ", ".join(missing))
    parts = []
    for j, d in enumerate(pool):
        arr, order, sz = prepare.load(d, size)
        arrays[d] = (arr, order, sz)
        bad = prepare.unreadable(d, size)
        sub = frame[frame.dataset == d]
        gone = [i for i in sub.picture if i not in order]
        if gone:
            raise SystemExit("pseudo_csv: %d %s picture(s) are not in its index or cache, e.g. %s; rebuild the csv with pseudo.py"
                             % (len(gone), d, gone[0]))
        sub = sub[np.array([i not in bad for i in sub.picture], dtype=bool)]
        parts.append(pd.DataFrame({
            "dataset": d, "dataset_index": names.index(pseudo.TEACHER_DATASET), "image": sub.picture.values, "patient": sub.patient.values,
            "truth": 0.0, "split": "pseudo", "fold": -1, "cache_row": [order[i] for i in sub.picture],
            "readable": True, "pseudo": True, **{c: sub[c].values for c in pseudo.FOLD_COLUMNS}}))
    return pd.concat(parts, ignore_index=True)


def pick_pseudo(df, extra_rows, fold, opts):
    """The pool rows fold `fold` trains on, labelled by p_fold{fold} alone (the net that never trained on that fold):
    at or below pseudo_low a negative, at or above pseudo_high a positive, between dropped. Their truth is set in `df`
    (0 or 1, or the probability with pseudo_soft); the fold overwrites it again next time."""
    p = df[pseudo.FOLD_COLUMNS[fold]].values[extra_rows]
    lab = pseudo.label_of(p, float(opts.get("pseudo_low", pseudo.LOW)), float(opts.get("pseudo_high", pseudo.HIGH)))
    keep = ~np.isnan(lab)
    rows = extra_rows[keep]
    if not len(rows):
        raise SystemExit("fold %d: no pool picture is confident under p_fold%d; nothing to train on" % (fold, fold))
    df.loc[rows, "truth"] = p[keep] if opts.get("pseudo_soft") else lab[keep]
    print("fold %d: %d of %d pool pictures taught (p_fold%d)" % (fold, len(rows), len(extra_rows), fold), flush=True)
    return rows


HELD_GB = None     # the share core/card.py lease gave this process, once limit_card_memory has run


def limit_card_memory(gb=None, name=None, short_min=None):
    """A hard ceiling on what this process may take from the graphics card: RAIDIOLOGY_CARD_GB, 6.5
    by default here, or 95% of a smaller card.

    Without one the allocator's cache only grows. It reached 10.2 of the card's 12 GB by a third
    fold (queue eleven, 2026-10-02), and on Windows a full card does not fail: when the desktop
    wanted room, epochs went from 13 s to 144. At the ceiling the allocator hands its cached blocks
    back and tries again before it gives up. The measured peak of a fold, the pictures' 2.9 GB
    included, is 4.2 to 4.9 GB allocated and 5.3 reserved at 384 px and batch 64 (queue fifteen,
    2026-10-03, after `pixel_mean_sd`; the 7.7 GB measured before was that one line), so 6.5 holds a
    fold with 0.15 GB more pictures and leaves the desktop 5.5 of the card's 12 (it was 9.0).

    The process also waits here for its share of the card (core/card.py lease), so several
    trainers' ceilings cannot add up past it.

    2026-10-04: a caller that wants a smaller share passes gb (and name, short_min) here instead of
    leasing first: the chest outline job leased 2.5 GB, then this asked again for 6.5, waited on its
    own share and held 2.5 GB for 70 minutes. The ceiling is now the share actually held, not 6.5."""
    global HELD_GB
    if DEV.type != "cuda":
        return
    from core import card as card_lease  # noqa: E402
    # Lease first, sized from nvidia-smi: reading the card through torch.cuda starts CUDA, and a job
    # waiting for its share must hold none (2026-10-03: waiting jobs held 2.1-6.9 GB; the machine ran out of
    # committed memory at 23:21 and a training job crashed)
    smi = card_lease.card_total_gb()
    want = float(os.environ.get("RAIDIOLOGY_CARD_GB", "6.5")) if gb is None else gb
    HELD_GB = card_lease.lease(want if smi is None else min(0.95 * smi, want),
                               name or os.path.basename(sys.argv[0]) or "xray detector", short_min=short_min)
    card = torch.cuda.current_device()  # DEV is plain "cuda", which this call refuses without an index
    total = torch.cuda.get_device_properties(card).total_memory
    torch.cuda.set_per_process_memory_fraction(card_ceiling_bytes() / total, card)


def card_ceiling_bytes():
    total = torch.cuda.get_device_properties(DEV).total_memory
    gb = HELD_GB if HELD_GB is not None else float(os.environ.get("RAIDIOLOGY_CARD_GB", "6.5"))
    return min(0.95 * total, gb * 2 ** 30)


def stack(df, arrays):
    """Every readable row's pixels in one uint8 tensor, on the GPU when it fits, else pinned."""
    good = df[df.readable]
    n = len(good)
    h, w = prepare.canvas(next(iter(arrays.values()))[2])
    want = n * h * w
    where = "cpu"
    if DEV.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        free = min(free, card_ceiling_bytes() - torch.cuda.memory_reserved())
        # Room for the step: measured 1.6 to 2.2 GB beside the pictures at 384 px and batch 64
        # (queue fifteen, 2026-10-03), and some. It was 5.5, sized from a peak that was one line.
        if want + int(2.5 * 2 ** 30) < free:
            where = "cuda"
    # Pinned memory cannot be paged out, so on a shared box only a modest stack is pinned: a batch
    # is 9 MB at 384 px, a few milliseconds to copy unpinned against a step of about 80.
    t = torch.empty((n, h, w), dtype=torch.uint8,
                    device=DEV if where == "cuda" else "cpu",
                    pin_memory=(where == "cpu" and DEV.type == "cuda" and want <= PIN_LIMIT))
    for d in arrays:
        arr = arrays[d][0]
        rows = good[good.dataset == d]
        if rows.empty:
            continue
        src = np.asarray(arr[rows.cache_row.values])       # one read of the memmap per dataset
        t[torch.as_tensor(rows.stack_row.values, dtype=torch.long)] = torch.from_numpy(src).to(t.device)
    print("pixels: %d images, %.2f GB on %s%s" % (n, want / 1e9, where, ", pinned" if t.is_pinned() else ""), flush=True)
    return t


# What a trial may pass with `train --set key=<json>`. Each is off by default, so version 1's recipe
# is what runs without one; docs/efficiency/README.md has what each was measured to do.
SETTINGS = {
    "val_share": "the early-stop slice of the training patients, 0 for none (the last epoch is kept)",
    "seed": "shifts every draw: one recipe at two seeds is the noise a change has to clear",
    "schedule": "cosine (version 1, decayed once an epoch) or onecycle (up to max_lr, then down, every batch)",
    "max_lr": "the one-cycle's peak learning rate", "pct_start": "the share of steps climbing to it",
    "datasets": "the collections to train on, a list from datasets.ALL holding the shipped two "
                "(datasets.NAMES, the default); a collection on trial adds its pictures to every fold",
    "fixed_batch": "every batch one size (the last part-batch dropped in training, padded in predicting), "
                   "so cuDNN tunes once a process, not 16 to 25 s again for each fold's odd remainder",
    "sizes": "progressive resolution, [[share of epochs, pixels], ...], shrunk on the card",
    "layout": "channels_last (version 1) or plain",
    "compile": "torch.compile, which needs triton-windows (../_tools/venv-next)",
    "fused_adam": "AdamW's fused kernel",
    "sync_every": "steps queued before the loop waits for the card (1, version 1's loop, waits every step); "
                  "above 1 the epoch's batch indices and labels sit on the card, so a step copies nothing "
                  "from the host. The same batches and arithmetic, and the same logged loss",
    "canvas": 'the picture cache\'s canvas, "HxW" (prepare.py --size), instead of the 384 square: children\'s '
              "wrist pictures are 99% portrait. Not with sizes",
    "init": "the name of one of our own readers whose `trunk.` tensors start each fold's net (INITS); "
            "chest-v103 is chest reader v103's fold 0",
    "invert": "the share of training pictures shown inverted (white bone on black becomes black on white), "
              "after the brightness nudge; 0 (none) is version 1's",
    "gamma": "a gamma change on every training picture, x ** e with log e uniform in [-gamma, gamma] "
             "(0.3 darkens or lightens the mid-greys by up to about a third); 0 is version 1's",
    "loc_weight": "the weight of a 1x1 head on the stride-16 map trained per cell on the drawn fracture boxes "
                  "(boxes.py; all zeros on a healthy picture, no loss where a picture has no box source), "
                  "added to the picture's own loss; 0 (none) is version 1's, with no head at all. Not with sizes",
    "pseudo_csv": "a pseudo.py score csv (picture, dataset, patient, p_teacher): the pool's pictures the teacher "
                  "is sure of join training as labels, and no fold's held-out set or scoring takes one",
    "pseudo_low": "with pseudo_csv: a pool picture at or below this teacher probability is a negative (0.1)",
    "pseudo_high": "with pseudo_csv: a pool picture at or above this teacher probability is a positive (0.9); "
                   "those between are dropped",
    "pseudo_soft": "with pseudo_csv: the teacher's probability is the BCE target, not 0 or 1 (false)",
    "strong_aug": "with pseudo_csv: a stronger augmentation (STRONG: a wider crop, scale and turn, twice the "
                  "brightness and contrast jitter, a gamma change) on the pseudo-labelled pictures only",
}
# The stronger augmentation of the pseudo-labelled pictures (strong_aug): augment's own plan, with a wider
# crop and turn and a gamma change. Read only by augment_mixed, so no other picture is drawn from it.
STRONG = {**CFG, "rotate_deg": 20.0, "translate": 0.10, "jitter": 0.20, "scale": 0.20, "gamma": 0.30}
PSEUDO_KEYS = ("pseudo_low", "pseudo_high", "pseudo_soft", "strong_aug")

# Our own readers a fold's net may start from (the `init` setting).
INITS = {"chest-v103": os.path.join(paths.WORK, "eval", "chest", "reader-v103", "fold0.pt")}


def load_init(net, name, say=print, path=None):
    """Copy the trunk tensors of one of our own readers' checkpoints into `net` (name and shape must both match); the head stays as built.
    Chest reader checkpoints keep their weights under `state`, CBIS ones under `net`. Returns the count copied."""
    path = path or INITS[name]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state" in ck:
        src = ck["state"]
    elif "net" in ck:
        src = ck["net"]
    else:
        sys.exit("init %s: %s has neither a `state` nor a `net` key; keys found: %s" % (name, path, ", ".join(sorted(map(str, ck)))))
    have = net.state_dict()
    new = {k: v for k, v in src.items() if k.startswith("trunk.") and k in have and tuple(have[k].shape) == tuple(v.shape)}
    n_trunk = sum(1 for k in have if k.startswith("trunk."))
    skipped = sum(1 for k in src if k.startswith("trunk.")) - len(new)
    say("init %s: %s, %d of %d trunk tensors copied, %d skipped, head fresh" % (name, path, len(new), n_trunk, skipped))
    if len(new) < 0.9 * n_trunk:
        sys.exit("init %s copied only %d of %d trunk tensors; the checkpoint does not fit this net" % (name, len(new), n_trunk))
    net.load_state_dict(new, strict=False)
    return len(new)


def init_file(settings):
    """The start file a run's `init` names, as meta.json and the resume fingerprint record it; None without the setting."""
    if not settings.get("init"):
        return None
    p = INITS[settings["init"]]
    return {"path": p, "size": os.path.getsize(p), "mtime": int(os.path.getmtime(p))}


def batches(idx, size, generator=None, drop_last=False):
    order = idx if generator is None else idx[torch.randperm(len(idx), generator=generator).numpy()]
    end = len(order) - (len(order) % size if drop_last else 0)
    for a in range(0, end, size):
        yield order[a:a + size]


def size_at(sizes, epoch, epochs, full):
    """The picture size for `epoch` (1-based) under a progressive schedule, else the cache's own."""
    if not sizes:
        return full
    done = 0.0
    for share, px in sizes:
        done += share
        if epoch <= round(done * epochs):
            return int(px)
    return full


def affine_draw(n, g, device, dtype, hw, wh, plan=None):
    """The flip and the rotate-and-shift of a batch of `n` pictures, drawn from `g` in the order
    augment has always drawn them. `theta` maps output coordinates to input ones (affine_grid).
    A `plan` with a `scale` (STRONG) draws one zoom more, after those, and zooms in below 1."""
    plan = plan or CFG
    flip = torch.rand(n, device=device, generator=g) < 0.5
    ang = (torch.rand(n, device=device, generator=g) * 2 - 1) * (plan["rotate_deg"] * np.pi / 180.0)
    tx = (torch.rand(n, device=device, generator=g) * 2 - 1) * plan["translate"] * 2
    ty = (torch.rand(n, device=device, generator=g) * 2 - 1) * plan["translate"] * 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin * hw, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin * wh, cos, ty
    if plan.get("scale"):
        zoom = 1 + (torch.rand(n, device=device, generator=g) * 2 - 1) * plan["scale"]
        theta[:, :, :2] *= zoom[:, None, None]
    return flip, theta


def augment(x, g, opts=None, cells=None, plan=None):
    """Flip, small rotate and shift, and a brightness and contrast nudge. All on the GPU, all seeded.
    `opts` may add a gamma change and inversion (SETTINGS); they draw from the generator only when set,
    so a run without them sees exactly the batches it saw before they existed. `plan` is CFG's augmentation
    values (STRONG, for the pseudo-labelled pictures, has more); its gamma, when it has one, wins over opts'.
    The rotation is of the picture, so on a canvas that is not square its grid factors carry the
    aspect (height over width and back); on a square they are 1, and x * 1.0 is x exactly.
    `cells` (B x rows x columns, the box map, idea 125) gets the same flip and the same affine, on its
    own grid, by nearest-cell sampling: its values stay 0 or 1, so a sub-cell shift never blurs a
    single cell away (bilinear and a 0.5 threshold did); it is returned beside the picture."""
    plan = plan or CFG
    n = x.shape[0]
    hw, wh = x.shape[-2] / x.shape[-1], x.shape[-1] / x.shape[-2]
    flip, theta = affine_draw(n, g, x.device, x.dtype, hw, wh, plan)
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if cells is not None:
        c = cells[:, None]
        c = torch.where(flip[:, None, None, None], c.flip(-1), c)
        grid_c = F.affine_grid(theta, list(c.shape), align_corners=False)
        c = F.grid_sample(c, grid_c, mode="nearest", padding_mode="zeros", align_corners=False)
        cells = c[:, 0].to(cells.dtype)
    j = plan["jitter"]
    contrast = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device, generator=g) * 2 - 1) * j
    bright = (torch.rand(n, 1, 1, 1, device=x.device, generator=g) * 2 - 1) * j
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * contrast + m + bright).clamp(0.0, 1.0)
    opts = opts or {}
    gam = float(plan.get("gamma") or opts.get("gamma") or 0.0)
    if gam:
        e = torch.exp((torch.rand(n, 1, 1, 1, device=x.device, generator=g) * 2 - 1) * gam)
        x = x.clamp_min(1e-6) ** e.to(x.dtype)
    inv = float(opts.get("invert") or 0.0)
    if inv:
        turn = torch.rand(n, device=x.device, generator=g) < inv
        x = torch.where(turn[:, None, None, None], 1.0 - x, x)
    return x if cells is None else (x, cells)


def augment_mixed(x, g, opts, strong):
    """`augment` for a batch in which the pictures `strong` marks (the pseudo-labelled ones) get STRONG's
    augmentation. Both are drawn for every picture, so the draws never depend on which pictures a batch holds."""
    return torch.where(strong[:, None, None, None], augment(x, g, opts, plan=STRONG), augment(x, g, opts))


def strong_of(strong_all, rows):
    """The batch's mask for the strong augmentation, by dataframe row; None without strong_aug."""
    if strong_all is None:
        return None
    return strong_all[torch.as_tensor(rows, dtype=torch.long, device=strong_all.device)]


def feed(pix, df, rows, mean, sd, n_datasets):
    """One batch: pixels as float, standardised, plus the dataset one-hot and the truth."""
    r = torch.as_tensor(df.stack_row.values[rows], dtype=torch.long, device=pix.device)
    x = pix[r].to(DEV, non_blocking=True).float().div_(255.0).unsqueeze(1)
    which = F.one_hot(torch.as_tensor(df.dataset_index.values[rows], dtype=torch.long, device=DEV),
                      n_datasets).float()
    y = torch.as_tensor(df.truth.values[rows], dtype=torch.float32, device=DEV)
    return x, which, y, mean, sd


def auroc(score, truth):
    s, t = np.asarray(score, float), np.asarray(truth, int)
    if t.min() == t.max():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(t.sum())
    n_neg = len(t) - n_pos
    return float((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def youden(score, truth):
    s, t = np.asarray(score, float), np.asarray(truth, int)
    if t.min() == t.max():
        return 0.5
    cuts = np.unique(s)
    if len(cuts) > 512:
        cuts = np.quantile(s, np.linspace(0, 1, 512))
    best, at = -2.0, 0.5
    pos, neg = t.sum(), len(t) - t.sum()
    for c in cuts:
        y = (s >= c)
        j = (y & (t == 1)).sum() / max(pos, 1) + ((~y) & (t == 0)).sum() / max(neg, 1) - 1.0
        if j > best:
            best, at = j, float(c)
    return at


def predict_rows(net, pix, df, rows, mean, sd, n_datasets, batch, pad=False):
    """Scores for `rows`. With `pad` the last part-batch is filled up with copies of its own last
    picture and the copies' answers thrown away: in eval mode each picture's answer is its own, and
    every call then has the one shape cuDNN has already tuned for."""
    net.eval()
    out = np.zeros(len(rows), np.float32)
    with torch.no_grad():
        for a in range(0, len(rows), batch):
            sl = rows[a:a + batch]
            n = len(sl)
            if pad and n < batch:
                sl = np.concatenate([sl, np.repeat(sl[-1:], batch - n)])
            x, which, _y, _m, _s = feed(pix, df, sl, mean, sd, n_datasets)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                z = net((x - mean) / sd, which)
            out[a:a + n] = torch.sigmoid(z.float())[:n].cpu().numpy()
    return out


def loc_weight_of(opts):
    """The box head's weight in a run's settings (SETTINGS); 0 is no head."""
    return float((opts or {}).get("loc_weight") or 0.0)


def make_net(n_datasets, opts):
    net = Net(CFG["widths"], CFG["dropout"], CFG["stem_stride"], n_datasets, loc=loc_weight_of(opts) > 0).to(DEV)
    if opts.get("layout", "channels_last") == "channels_last":
        net = net.to(memory_format=torch.channels_last)
    return net


def make_opt(net, lr, opts):
    # None, not False, without the setting: what version 1 was built with, and so the same kernels.
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=CFG["weight_decay"],
                             fused=True if DEV.type == "cuda" and opts.get("fused_adam") else None)


def forward_of(net, opts):
    """The network as a step calls it: compiled when asked, which on Windows needs triton-windows."""
    if not opts.get("compile"):
        return net
    return torch.compile(net, mode=opts["compile"] if isinstance(opts["compile"], str) else None)


def cell_loss(logits, cells, ok):
    """Per-cell BCE of the box head against the box map, counted on the pictures that have a box
    source (`ok`, 1 or 0) and averaged over their cells."""
    per = F.binary_cross_entropy_with_logits(logits.float(), cells, reduction="none")
    w = ok[:, None, None]
    return (per * w).sum() / (w.sum() * per.shape[1] * per.shape[2]).clamp_min(1.0)


def cells_of(loc, rows):
    """The box targets of the pictures in dataframe rows `rows`, as float cells and mask flags on the
    card; None when the run has no box head."""
    if loc is None:
        return None
    r = torch.as_tensor(rows, dtype=torch.long, device=DEV)
    return loc[0][r].float(), loc[1][r].float()


def train_step(forward, opt, loss_fn, x, which, y, mean, sd, g, opts, px=None, loc=None, strong=None):
    """One step of training exactly as a fold runs it, from a batch of pixels in [0, 1] to updated
    weights; `bench` times this same function, so a measured setting is the setting trained. `px`
    shrinks the batch on the card first (progressive resolution). `loc` is the batch's box target
    (cells, mask flags) when the run has a box head: the loss is the picture's plus loc_weight times
    the cell loss. `strong` marks the pseudo-labelled pictures of the batch (strong_aug). Returns the loss, on the card."""
    if px and px != x.shape[-1]:
        x = F.interpolate(x, size=(px, px), mode="bilinear", antialias=True, align_corners=False)
    if loc is None:
        x = augment(x, g, opts) if strong is None else augment_mixed(x, g, opts, strong)
    else:
        x, cells = augment(x, g, opts, loc[0])
    x = (x - mean) / sd
    if opts.get("layout", "channels_last") == "channels_last":
        x = x.to(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
        if loc is None:
            loss = loss_fn(forward(x, which), y)
        else:
            logit, cell_logits = forward(x, which, loc=True)
            loss = loss_fn(logit, y) + loc_weight_of(opts) * cell_loss(cell_logits, cells, loc[1])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach()


def run_epoch_queued(forward, opt, sched, loss_fn, pix, stack_rows, which_all, truth_all, parts, mean, sd,
                     g_dev, opts, px, sync_every, pace, skip=0, total0=0.0, after=None, loc=None, strong_all=None):
    """One epoch of `train_fold`'s loop without a copy from the host per step: the epoch's batches
    (`parts`, drawn by `batches` exactly as the plain loop draws them) go to the card once, and the
    loop waits for the card every `sync_every` steps, resting (core/polite.py) per block. The loss
    total is summed on the card in float64 from the same float32 losses in the same order, so it is
    the plain loop's `total` to the bit. Returns that total.

    A mid-epoch resume passes `skip` (batches already trained, dropped from `parts`) and `total0` (their
    loss total); `after(batches done in the epoch, total)` is called after every batch (partial checkpoints).
    `loc` is the box head's (cells, mask flags) by dataframe row, or None."""
    parts = parts[skip:]
    if not parts:                       # resumed after the epoch's last batch
        return float(total0)
    order = torch.as_tensor(np.concatenate(parts), dtype=torch.long)
    on_pix, on_dev = order.to(pix.device), order.to(DEV)
    total = torch.full((), total0, dtype=torch.float64, device=DEV)
    began, at = time.time(), 0
    for i, rows in enumerate(parts, 1):
        n = len(rows)
        x = pix[stack_rows[on_pix[at:at + n]]].to(DEV, non_blocking=True).float().div_(255.0).unsqueeze(1)
        which, y = which_all[on_dev[at:at + n]], truth_all[on_dev[at:at + n]]
        loss = train_step(forward, opt, loss_fn, x, which, y, mean, sd, g_dev, opts, px,
                          cells_of(loc, on_dev[at:at + n]), strong_of(strong_all, on_dev[at:at + n]))
        if sched is not None:
            sched.step()
        total += loss.double() * n
        at += n
        if i % sync_every == 0 or i == len(parts):
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            pace.rest(time.time() - began)
            began = time.time()
        if after is not None:
            after(skip + i, total)
    return float(total)


def pixel_mean_sd(sample, chunk=256):
    """Mean and standard deviation (n - 1) of the uint8 pictures `sample`, in 0-1 units, as float32
    scalars on DEV. Summed as exact integers a chunk at a time: the float copy of 4,096 pictures it
    replaces took 2.25 GB twice over, which was the trainer's whole 7.7 GB peak beside the pixels,
    and at fold 1 it crossed the 9.0 GB ceiling (queue fifteen, 2026-10-03). The numbers differ from
    that float32 copy's only in its rounding."""
    s = s2 = 0
    for part in sample.split(chunk):
        s += int(part.sum(dtype=torch.int64))
        s2 += int(part.to(torch.int32).square().sum(dtype=torch.int64))
    n = sample.numel()
    mean = s / n
    var = (s2 - s * mean) / (n - 1)
    sd = max(var ** 0.5 / 255.0, 1e-3)
    return (torch.tensor(mean / 255.0, dtype=torch.float32, device=DEV),
            torch.tensor(sd, dtype=torch.float32, device=DEV))


def held_out_scores(net, pix, df, te, mean, sd, n_datasets, batch, opts):
    """A fold's held-out scores: the one code path both a trained fold and a resumed one use, so
    their numbers are identical."""
    return predict_rows(net, pix, df, te, mean, sd, n_datasets, batch, bool(opts.get("fixed_batch")))


class SelfTestStop(Exception):
    """Raised by train_fold right after a partial checkpoint when RAIDIOLOGY_CKPT_SELFTEST_STOP asks for it."""


PRIOR_PEAK_GB = 0.0      # the card's peak before a mid-fold resume, for train() to carry into the ledger
RESUME_PROBE = None      # the selftest's hook: called with the state a mid-fold resume has just put back


def stop_spec():
    """RAIDIOLOGY_CKPT_SELFTEST_STOP: "E" stops after epoch E's partial save, "E:B" after batch B of epoch E's."""
    s = os.environ.get("RAIDIOLOGY_CKPT_SELFTEST_STOP", "").strip()
    if not s:
        return None
    e, _, b = s.partition(":")
    return int(e), (int(b) if b else None)


def train_fold(pix, df, dev_rows, fold, epochs, batch, seed, n_datasets, val_share=VAL_SHARE, opts=None,
               partial=None, fingerprint=None, loc=None, extra_rows=None):
    """Train on the development units outside `fold`, early-stopping on a slice of them. With
    `val_share` 0 there is no slice: every unit trains, and the last epoch is the model. `opts` are
    the trial settings (SETTINGS) beyond those two. `loc` is the box targets (cells, mask flags) by
    dataframe row, needed when opts ask for a box head (loc_weight). `extra_rows` are the pseudo-labelled
    pool pictures (pseudo_csv): they join the training in every fold and are never in its slice or its held-out set.

    With `partial` (a path) and `fingerprint`, a partial checkpoint (net, optimizer, scheduler, best state,
    early-stop counter, history, generators) is saved there after every epoch and, mid-epoch, whenever
    core.ckpt.Every is due (300 s); one with the same fingerprint is continued from. The caller clears it
    once the fold's final checkpoint is saved."""
    global PRIOR_PEAK_GB
    PRIOR_PEAK_GB = 0.0
    opts = opts or {}
    if loc_weight_of(opts) > 0 and loc is None:
        raise SystemExit("loc_weight needs the box targets (python detector.py boxes first)")
    if DEV.type == "cuda":
        # The last fold's cached blocks go back to the card first, and again when the picture size
        # changes. Without this the process held 10.2 of the card's 12 GB by its third fold (queue
        # eleven, 2026-10-02) and those epochs went from 13 s to 39, 116 and 144: on Windows a full
        # card does not fail, it spills into ordinary memory. No arithmetic changes.
        torch.cuda.empty_cache()
    fixed = bool(opts.get("fixed_batch"))
    tr_dev = dev_rows[df.fold.values[dev_rows] != fold]
    te = dev_rows[df.fold.values[dev_rows] == fold]
    units = sorted(set(df.patient.values[tr_dev]))
    rng = np.random.default_rng(seed + fold)
    held = set(np.asarray(units)[rng.permutation(len(units))[:int(round(val_share * len(units)))]])
    val = tr_dev[np.isin(df.patient.values[tr_dev], list(held))]
    tr = tr_dev[~np.isin(df.patient.values[tr_dev], list(held))]
    if extra_rows is not None:
        tr = np.concatenate([tr, pick_pseudo(df, extra_rows, fold, opts)])
    sample = pix[torch.as_tensor(df.stack_row.values[tr[:4096]], dtype=torch.long, device=pix.device)]
    mean, sd = pixel_mean_sd(sample)
    del sample
    net = make_net(n_datasets, opts)
    if opts.get("init"):
        load_init(net, opts["init"])
    forward = forward_of(net, opts)
    onecycle = opts.get("schedule", "cosine") == "onecycle"
    per_epoch = len(tr) // batch if fixed else -(-len(tr) // batch)
    opt = make_opt(net, CFG["lr"], opts)
    if onecycle:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(opts.get("max_lr", 1e-3)),
                                                    total_steps=epochs * per_epoch,
                                                    pct_start=float(opts.get("pct_start", 0.25)),
                                                    anneal_strategy="cos")
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos = float(df.truth.values[tr].sum())
    pos_weight = torch.tensor([max(len(tr) - pos, 1.0) / max(pos, 1.0)], device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    g_cpu = torch.Generator().manual_seed(seed + fold)
    g_dev = torch.Generator(device=DEV).manual_seed(seed + fold)
    best, best_state, since = -1.0, None, 0
    pace = polite.Pace()
    full = pix.shape[-1]
    print("fold %d: train %d, early-stop %d, predict %d images (pos_weight %.2f)"
          % (fold, len(tr), len(val), len(te), float(pos_weight)), flush=True)
    sync_every = int(opts.get("sync_every", 1))
    strong_all = torch.as_tensor(df.pseudo.values, dtype=torch.bool, device=DEV) if opts.get("strong_aug") else None
    took = []
    if sync_every > 1:
        # Every row's place in the pixel tensor, collection and truth, on the card once a fold.
        stack_rows = torch.as_tensor(df.stack_row.values, dtype=torch.long, device=pix.device)
        which_all = F.one_hot(torch.as_tensor(df.dataset_index.values, dtype=torch.long, device=DEV),
                              n_datasets).float()
        truth_all = torch.as_tensor(df.truth.values, dtype=torch.float32, device=DEV)
    px = full
    stop = stop_spec()
    every = ckpt.Every() if partial else None
    began_fold = time.time()
    prior_s, g0, t0 = 0.0, g_cpu.get_state(), time.time()

    def checkpoint(done, mid):
        """The partial checkpoint: `done` epochs finished, `mid` the batch state of the epoch after them (or None)."""
        peak = max(PRIOR_PEAK_GB, torch.cuda.max_memory_allocated() / 2 ** 30) if DEV.type == "cuda" else 0.0
        ckpt.save_partial(partial, {
            "net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": done,
            "mid": mid, "best": best, "best_state": best_state, "since": since, "took": took, "px": px,
            "seconds": prior_s + time.time() - began_fold, "peak_gb": peak,
            "g_cpu": g_cpu.get_state(), "g_dev": g_dev.get_state(), "rng": ckpt.rng_state(),
            "fingerprint": fingerprint})
        every.mark()

    def after_batch(n_done, total):
        """Called after each batch of epoch `ep`: a mid-epoch partial when one is due."""
        want_stop = stop is not None and stop == (ep, n_done)
        if every.due() or want_stop:
            checkpoint(ep - 1, {"batch": n_done, "total": float(total), "g_cpu0": g0,
                                "epoch_s": time.time() - t0})
            if want_stop:
                raise SelfTestStop("stopped after batch %d of epoch %d" % (n_done, ep))

    start_ep, skip, total0, ep = 1, 0, 0.0, 0
    ck = ckpt.load_partial(partial, fingerprint) if partial else None
    if ck is not None and 0 <= ck["epoch"] <= epochs and not (ck["mid"] and ck["epoch"] >= epochs):
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best, since, took = ck["best"], ck["since"], list(ck["took"])
        best_state = {k: t.to(DEV) for k, t in ck["best_state"].items()} if ck["best_state"] is not None else None
        px, prior_s = ck["px"], ck["seconds"]
        PRIOR_PEAK_GB = ck["peak_gb"]
        ckpt.set_rng_state(ck["rng"])
        g_dev.set_state(ck["g_dev"].cpu())
        ep = ck["epoch"]
        if ck["mid"]:
            # the epoch's batches are drawn from its start state again, and the trained ones skipped
            g_cpu.set_state(ck["mid"]["g_cpu0"])
            start_ep, skip, total0 = ep + 1, ck["mid"]["batch"], ck["mid"]["total"]
            t0_back = ck["mid"]["epoch_s"]
        else:
            g_cpu.set_state(ck["g_cpu"])
            start_ep = ep + 1
        print("fold %d: resumed mid-fold at epoch %d of %d (partial checkpoint)%s"
              % (fold, start_ep, epochs, ", batch %d" % skip if skip else ""), flush=True)
        if RESUME_PROBE is not None:
            RESUME_PROBE(ck, net, opt, sched, best, since)
    del ck
    for ep in range(start_ep, epochs + 1):
        t0 = time.time() - (t0_back if skip and ep == start_ep else 0.0)
        g0 = g_cpu.get_state()
        net.train()
        total = total0 if ep == start_ep else 0.0
        skipped = skip if ep == start_ep else 0
        was, px = px, size_at(opts.get("sizes"), ep, epochs, full)
        if px != was and DEV.type == "cuda":
            torch.cuda.empty_cache()
        if sync_every > 1:
            total = run_epoch_queued(forward, opt, sched if onecycle else None, loss_fn, pix, stack_rows,
                                     which_all, truth_all, list(batches(tr, batch, g_cpu, drop_last=fixed)),
                                     mean, sd, g_dev, opts, px, sync_every, pace, skipped, total,
                                     after_batch if every else None, loc, strong_all)
        for n_done, rows in (enumerate(batches(tr, batch, g_cpu, drop_last=fixed), 1) if sync_every <= 1 else ()):
            if n_done <= skipped:
                continue
            began = time.time()
            x, which, y, _m, _s = feed(pix, df, rows, mean, sd, n_datasets)
            loss = train_step(forward, opt, loss_fn, x, which, y, mean, sd, g_dev, opts, px, cells_of(loc, rows),
                              strong_of(strong_all, rows))
            if onecycle:
                sched.step()
            total += float(loss) * len(rows)
            pace.rest(time.time() - began)       # core/polite.py: half the card while somebody is at it
            if every:
                after_batch(n_done, total)
        if not onecycle:
            sched.step()
        took.append(time.time() - t0)
        if not len(val):
            best, best_state = float("nan"), None
            print("  epoch %2d  loss %.4f  %.0f s%s" % (ep, total / max(len(tr), 1), time.time() - t0,
                                                       "  at %d px" % px if px != full else ""), flush=True)
        else:
            v = auroc(predict_rows(net, pix, df, val, mean, sd, n_datasets, batch, fixed), df.truth.values[val])
            print("  epoch %2d  loss %.4f  val auroc %.4f  %.0f s" % (ep, total / max(len(tr), 1), v,
                                                                      time.time() - t0), flush=True)
            if v > best + 1e-4:
                best, since = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                since += 1
                if since >= CFG["patience"]:
                    print("  stopping: no better validation AUROC for %d epochs" % CFG["patience"], flush=True)
                    break
        if every:
            checkpoint(ep, None)
            if stop == (ep, None):
                raise SelfTestStop("stopped after epoch %d" % ep)
    if len(val) and best_state is None:
        # Every epoch's validation AUROC was undefined (one class in the slice): no epoch was ever
        # chosen, and keeping whatever the net is after `patience` epochs would be a silent guess.
        raise SystemExit("fold %d: the early-stop slice holds one class (%d images), so no epoch could be "
                         "chosen; use a larger val_share, or 0 for none" % (fold, len(val)))
    if best_state is not None:
        net.load_state_dict(best_state)
    oof = held_out_scores(net, pix, df, te, mean, sd, n_datasets, batch, opts)
    return net, {"fold": fold, "epochs": ep, "val_auroc": best, "mean": float(mean), "sd": float(sd),
                 "train_images": len(tr), "val_images": len(val), "predicted_images": len(te),
                 "epoch_seconds": [round(v, 2) for v in took]}, te, oof


def check_settings(settings):
    unknown = sorted(set(settings) - set(SETTINGS))
    if unknown:
        raise SystemExit("unknown setting(s) %s: train takes %s" % (", ".join(unknown), ", ".join(sorted(SETTINGS))))
    training_names(settings)
    if "init" in settings and settings["init"] not in INITS:
        raise SystemExit("init is one of %s, not %r" % (", ".join(sorted(INITS)), settings["init"]))
    if "canvas" in settings:
        if not (isinstance(settings["canvas"], str) and settings["canvas"] == prepare.size_arg(settings["canvas"])):
            raise SystemExit('canvas is "HEIGHTxWIDTH", for example "640x384"')
        if settings.get("sizes"):
            raise SystemExit("canvas and sizes (progressive resolution, square) do not go together")
    if "loc_weight" in settings:
        w = settings["loc_weight"]
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise SystemExit("loc_weight is a number, 0 (no box head) or more, not %r" % (w,))
        if w > 0 and settings.get("sizes"):
            raise SystemExit("loc_weight and sizes do not go together: the box map is one size")
        if w > 0:
            size = settings.get("canvas", prepare.SIZE)
            boxes.grid(size)     # exits unless the canvas is a multiple of 16
            # the box caches are read before the card is leased, not after 6.5 GB and the stacking (idea 125)
            gone = [d for d in training_names(settings) if d in boxes.SOURCES and not boxes.has_cache(d, size)]
            if gone:
                raise SystemExit("loc_weight needs the box caches of %s: run detector.py boxes first (--size %s)"
                                 % (", ".join(gone), size))
    if "pseudo_csv" in settings:
        if not isinstance(settings["pseudo_csv"], str):
            raise SystemExit("pseudo_csv is the path of a pseudo.py score csv, as text")
        low, high = settings.get("pseudo_low", pseudo.LOW), settings.get("pseudo_high", pseudo.HIGH)
        if not (0 <= low < high <= 1):
            raise SystemExit("pseudo_low and pseudo_high are probabilities with pseudo_low below pseudo_high, not %r and %r"
                             % (low, high))
        if loc_weight_of(settings) > 0:
            raise SystemExit("pseudo_csv and loc_weight do not go together: a pool picture has no box map")
        pseudo.pool_names(settings["pseudo_csv"], training_names(settings))
    else:
        unset = [k for k in PSEUDO_KEYS if k in settings and settings[k] not in (None, False)]
        if unset:
            raise SystemExit("%s need pseudo_csv (pseudo.py score writes it)" % ", ".join(unset))


def training_names(settings):
    """The collections a run trains on: the shipped two, plus any on trial its settings name. The
    shipped two first, in their order, then the trial ones sorted, so the shipped two keep one-hot
    places 0 and 1 and their out-of-fold rows compare with a run trained on the two alone (sorting
    them all would put bonefract first)."""
    asked = settings.get("datasets", datasets.NAMES)
    if not set(datasets.NAMES) <= set(asked) or not set(asked) <= set(datasets.ALL):
        raise SystemExit("datasets must hold %s and come from %s" % (datasets.NAMES, datasets.ALL))
    return list(datasets.NAMES) + sorted(set(asked) - set(datasets.NAMES))


def box_targets(df, size, names):
    """Each development row's box map and mask flag, on the card by dataframe row (boxes.py): the
    box head's targets. A row without a box source has a zero map and mask 0, and so does every row of
    a collection boxes.py has no source for (orthofrac, bonefract): no box cache is read for it."""
    rows_n, cols_n = boxes.grid(size)
    cells = np.zeros((len(df), rows_n, cols_n), np.uint8)
    ok = np.zeros(len(df), np.uint8)
    for d in names:
        if d not in boxes.SOURCES:
            continue
        maps, mask = boxes.load(d, size)
        rows = np.flatnonzero(((df.dataset == d) & df.readable).values)
        src = df.cache_row.values[rows]
        cells[rows] = maps[src]
        ok[rows] = mask[src]
    print("box targets: %d pictures with a box source, %d fracture cells" % (int(ok.sum()), int(cells.sum())), flush=True)
    return torch.as_tensor(cells, device=DEV), torch.as_tensor(ok, device=DEV)


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


def save_fold(path, payload, partial):
    """A finished fold's checkpoint, then its partial checkpoint is no longer needed."""
    torch.save(payload, path)
    ckpt.clear_partial(partial)


def train(version, epochs, batch, settings=None):
    """`settings`: the trial settings described at SETTINGS; none of them is version 1's recipe.

    A restarted run resumes from the folds whose saved fingerprint matches this run's (RAIDIOLOGY_NO_RESUME=1
    to retrain them all)."""
    settings = settings or {}
    check_settings(settings)
    seed = datasets.SEED + 1000 * int(settings.get("seed", 0))
    val_share = float(settings.get("val_share", VAL_SHARE))
    torch.manual_seed(seed)
    card.require_card(DEV, torch)
    limit_card_memory()     # 2026-10-03: before the data is loaded; a waiting job must hold no memory (the 23:21 crash)
    names = training_names(settings)
    size = settings.get("canvas", prepare.SIZE)
    df, arrays, _ = table(("dev",), names, size)
    pool = pseudo.pool_names(settings["pseudo_csv"], names) if settings.get("pseudo_csv") else []
    if pool:
        # the pool's pictures join the table as training rows only (pseudo_rows), after the development rows
        df = number_rows(pd.concat([df, pseudo_rows(settings, size, names, pool, arrays)], ignore_index=True))
    pix = stack(df, arrays)
    loc = box_targets(df, size, names) if loc_weight_of(settings) > 0 else None
    dev_rows = df.index.values[df.readable.values & (df.split.values == "dev")]
    extra = df.index.values[df.pseudo.values] if pool else None
    n_datasets = len(names)     # a pool picture takes the slot the teacher read it with (pseudo.TEACHER_DATASET): no widening
    out = model_dir(version)
    os.makedirs(out, exist_ok=True)
    oof = pd.Series(np.nan, index=df.index)
    folds = []
    split_hash = sha256(splitmod.PATH)
    cache_hashes = {d: sha256(prepare.cache_paths(d, size)[1]) for d in names + pool}
    box_hashes = {d: sha256(boxes.cache_paths(d, size)[0]) for d in names if d in boxes.SOURCES} if loc is not None else {}
    fingerprint = {"version": version, "epochs": epochs, "batch": batch, "settings": settings, "cfg": CFG,
                   "seed": seed, "split_sha256": split_hash, "cache_sha256": cache_hashes,
                   "dev_rows": int(len(dev_rows)), **({"init_file": init_file(settings)} if settings.get("init") else {}),
                   **({"box_sha256": box_hashes} if box_hashes else {}),
                   **({"pseudo_sha256": sha256(settings["pseudo_csv"])} if pool else {})}
    fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True, default=str))
    for k in range(FOLDS):
        path = os.path.join(out, "fold%d.pt" % k)
        ck = load_resumable(path, {**fingerprint, "fold": k})
        if ck is not None:
            net = make_net(n_datasets, settings)
            net.load_state_dict(ck["state"])
            info = ck["info"]
            te = dev_rows[df.fold.values[dev_rows] == k]
            mean = torch.tensor(info["mean"], dtype=torch.float32, device=DEV)
            sd = torch.tensor(info["sd"], dtype=torch.float32, device=DEV)
            oof.iloc[te] = held_out_scores(net, pix, df, te, mean, sd, n_datasets, batch, settings)
            print("fold %d: resumed from %s (no training)" % (k, path), flush=True)
            folds.append(info)
            continue
        with ledger.run("detector", "train fold %d v%d" % (k, version)) as led:
            # Settings only when given, so a default run compares with the ledger's earlier ones.
            led.settings_from({"epochs": epochs, "batch": batch, "cfg": CFG, **({"settings": settings} if settings else {})})
            net, info, rows, scores = train_fold(pix, df, dev_rows, k, epochs, batch,
                                                 seed, n_datasets, val_share, settings,
                                                 partial=os.path.join(out, "fold%d.partial.pt" % k),
                                                 fingerprint={**fingerprint, "fold": k}, loc=loc, extra_rows=extra)
            oof.iloc[rows] = scores
            save_fold(path, {"state": net.state_dict(), "cfg": CFG, "info": info,
                             "fingerprint": {**fingerprint, "fold": k}}, os.path.join(out, "fold%d.partial.pt" % k))
            led.number("val_auroc", info["val_auroc"]).number("epochs", info["epochs"], "count")
            led.number("train_images", info["train_images"], "count")
            if DEV.type == "cuda":
                # What a card ceiling for this trainer would have to leave room for (pixels included).
                led.number("peak_memory_gb", max(PRIOR_PEAK_GB, torch.cuda.max_memory_allocated() / 2 ** 30))
                led.number("peak_reserved_gb", torch.cuda.max_memory_reserved() / 2 ** 30)
                torch.cuda.reset_peak_memory_stats()
        folds.append(info)
    thresholds = {}
    for d in names:
        m = (df.dataset == d) & df.readable
        thresholds[d] = youden(oof[m].values, df.truth[m].values)
    scored = df.readable & (df.split == "dev")       # a pool picture is never in the development scores
    oof_frame = df.loc[scored, ["dataset", "image", "patient", "fold", "truth"]].copy()
    oof_frame["score"] = oof[scored].values
    oof_frame.to_csv(os.path.join(out, "oof_dev.csv"), index=False)
    meta = {"version": version, "cfg": CFG, "settings": settings, "folds": folds, "thresholds": thresholds,
            "datasets": names + pool, "seed": seed,
            "split_sha256": split_hash,
            "cache_sha256": cache_hashes,
            "dev_auroc": {d: auroc(oof_frame[oof_frame.dataset == d].score.values,
                                   oof_frame[oof_frame.dataset == d].truth.values)
                          for d in names},
            "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "device": torch.cuda.get_device_name(0)
            if DEV.type == "cuda" else "cpu"}
    if settings.get("init"):
        meta["init_file"] = init_file(settings)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("trained: %s  out-of-fold AUROC %s  thresholds %s"
          % (out, {k: round(v, 4) for k, v in meta["dev_auroc"].items()},
             {k: round(v, 4) for k, v in thresholds.items()}), flush=True)


def build_boxes(size):
    """The box maps (boxes.py) of every development picture the trainer reads. table() gives only the
    development rows, so no test row's label is opened."""
    df, _arrays, _ = table(("dev",), names=datasets.NAMES, size=size)
    for d in datasets.NAMES:
        rows = df.cache_row.values[((df.dataset == d) & df.readable).values]
        boxes.build(d, size, rows)


def hottest_cells(net, pix, df, rows, mean, sd, n_datasets, batch):
    """The box head's hottest cell of each picture in `rows`, as a flat index on the stride-16 map."""
    net.eval()
    out = np.zeros(len(rows), np.int64)
    with torch.no_grad():
        for a in range(0, len(rows), batch):
            sl = rows[a:a + batch]
            x, which, _y, _m, _s = feed(pix, df, sl, mean, sd, n_datasets)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                _logit, cells = net((x - mean) / sd, which, loc=True)
            out[a:a + len(sl)] = cells.float().flatten(1).argmax(dim=1).cpu().numpy()
    return out


def hotcell(version):
    """How often the box head's hottest cell lies inside a drawn box (idea 125): each fold's network
    reads its own held-out development pictures that are fractured and have a box map, and the share
    whose hottest cell is a box cell is printed and ledgered per collection. Development rows only."""
    out = model_dir(version)
    meta_p = os.path.join(out, "meta.json")
    if not os.path.exists(meta_p):
        raise SystemExit("no detector at %s; train it with: python detector.py train --version %d" % (out, version))
    meta = json.load(open(meta_p, encoding="utf-8"))
    settings = meta.get("settings") or {}
    if loc_weight_of(settings) <= 0:
        raise SystemExit("version %d has no box head (loc_weight 0 or unset): nothing to read" % version)
    names = meta["datasets"]
    size = settings.get("canvas", prepare.SIZE)
    card.require_card(DEV, torch)
    limit_card_memory()
    df, arrays, _ = table(("dev",), names, size)
    pix = stack(df, arrays)
    loc = box_targets(df, size, names)
    cells_all, ok_all = loc[0].cpu().numpy().reshape(len(df), -1), loc[1].cpu().numpy()
    _meta, nets = load_folds(version, len(names))
    dev_rows = df.index.values[df.readable.values]
    hits = {d: [0, 0] for d in names}     # fractured pictures with a box map, and those whose hottest cell is a box cell
    with ledger.run("detector", "hotcell v%d" % version) as led:
        for k, (net, mean, sd) in enumerate(nets):
            te = dev_rows[df.fold.values[dev_rows] == k]
            te = te[(df.truth.values[te] == 1) & (ok_all[te] == 1)]
            if not len(te):
                continue
            hot = hottest_cells(net, pix, df, te, torch.tensor(mean, device=DEV), torch.tensor(sd, device=DEV),
                                len(names), READ_BATCH)
            inside = cells_all[te, hot] == 1
            for d in names:
                s = df.dataset.values[te] == d
                hits[d][0] += int(s.sum())
                hits[d][1] += int((inside & s).sum())
        for d in names:
            n, h = hits[d]
            if not n:
                print("hotcell v%d %s: no fractured development picture with a box map" % (version, d), flush=True)
                continue
            print("hotcell v%d %s: %d of %d fractured pictures with a box map have their hottest cell in a box (%.4f)"
                  % (version, d, h, n, h / n), flush=True)
            led.number("%s.hot_in_box" % d, h / n).number("%s.hot_pictures" % d, n, "count")


def loop(epochs, batch, settings, fold=0):
    """Seconds an epoch of the real training loop, on fold `fold`'s real pictures, nothing saved.

    `bench` times one step on random pixels already on the card; this times the loop around it too
    (fetching each batch, the logged loss, the rest per core/polite.py), which is where a step that
    is quick on the card loses its time. The first epoch carries cuDNN's tuning and is left out of
    the number `tools/ab.py` reads, `seconds_an_epoch`. Every unit trains (val_share 0)."""
    settings = dict(settings or {})
    check_settings(settings)
    card.require_card(DEV, torch)
    limit_card_memory()
    seed = datasets.SEED + 1000 * int(settings.get("seed", 0))
    torch.manual_seed(seed)
    df, arrays, _ = table(("dev",))
    pix = stack(df, arrays)
    dev_rows = df.index.values[df.readable.values]
    with ledger.run("detector", "loop fold %d" % fold) as led:
        led.settings_from({"epochs": epochs, "batch": batch, **({"settings": settings} if settings else {})})
        _net, info, _rows, _scores = train_fold(pix, df, dev_rows, fold, epochs, batch, seed,
                                                len(datasets.NAMES), 0.0, settings)
        later = info["epoch_seconds"][1:] or info["epoch_seconds"]
        led.number("seconds_an_epoch", float(np.median(later)), "s")
        led.number("first_epoch_seconds", info["epoch_seconds"][0], "s")
    print("seconds_an_epoch %.2f (median of epochs 2 to %d; the first took %.1f)"
          % (float(np.median(later)), epochs, info["epoch_seconds"][0]), flush=True)


def bench(batch, settings, steps=60):
    """Milliseconds a training step and pictures a second, of `train_step` itself on random pixels
    already on the card: 15 steps to warm up (cuDNN's tuning, a compile), then `steps` timed. What
    `tools/ab.py` compares two settings with; the pixels are random because a step's arithmetic does
    not depend on what the picture shows."""
    check_settings(settings)
    card.require_card(DEV, torch)
    limit_card_memory()
    n, full = 2048, 384
    pix = torch.randint(0, 256, (n, full, full), dtype=torch.uint8, device=DEV)
    net = make_net(2, settings).train()
    forward = forward_of(net, settings)
    opt = make_opt(net, CFG["lr"], settings)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.72], device=DEV))
    g = torch.Generator(device=DEV).manual_seed(0)
    mean, sd = torch.tensor(0.4, device=DEV), torch.tensor(0.25, device=DEV)
    sizes = settings.get("sizes")
    px = int(sizes[0][1]) if sizes else full

    def one():
        r = torch.randint(0, n, (batch,), device=DEV, generator=g)
        x = pix[r].float().div_(255.0).unsqueeze(1)
        which = F.one_hot(torch.randint(0, 2, (batch,), device=DEV, generator=g), 2).float()
        y = torch.randint(0, 2, (batch,), device=DEV, generator=g).float()
        return train_step(forward, opt, loss_fn, x, which, y, mean, sd, g, settings, px)

    t0 = time.time()
    for _ in range(15):
        float(one())
    warm = time.time() - t0
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        loss = one()
    float(loss)
    dt = (time.time() - t0) / steps
    print("batch %d at %d px: %.1f ms a step, %.0f images/s, warm-up %.1f s, peak %.2f GB"
          % (batch, px, dt * 1000, batch / dt, warm, torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)


_LOADED = {}


def load_folds(version, n_datasets):
    """The five fold weights, loaded once per process.

    Measured 2026-10-01: reading the five checkpoints off disk and rebuilding the networks costs
    about as much as scoring a whole series, and the MCP server calls this on every question
    somebody asks about the same scan. Keyed on the device as well as the version, because the
    device can change between calls and a net on the wrong one is a crash, not a slow answer."""
    key = (version, n_datasets, str(DEV))
    if key in _LOADED:
        return _LOADED[key]
    out = model_dir(version)
    meta_p = os.path.join(out, "meta.json")
    if not os.path.exists(meta_p):
        raise SystemExit("no detector at %s; train it with: python detector.py train --version %d"
                         % (out, version))
    meta = json.load(open(meta_p, encoding="utf-8"))
    nets = []
    for k in range(FOLDS):
        ck = torch.load(os.path.join(out, "fold%d.pt" % k), map_location=DEV, weights_only=False)
        net = Net(ck["cfg"]["widths"], ck["cfg"]["dropout"], ck["cfg"]["stem_stride"], n_datasets,
                  loc="loc.weight" in ck["state"]).to(DEV)
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append((net, ck["info"]["mean"], ck["info"]["sd"]))
    _LOADED[key] = (meta, nets)
    return meta, nets


READ_BATCH = 8
_READING = {}


def reading_folds(version, n_datasets):
    """The fold networks as a processor reads with them: each batch norm folded into the convolution
    before it, and the activations laid out channels-last. The same function to within float
    rounding; measured 2026-10-02 on the real weights at 384 px, the ensemble went from 0.106 to
    0.057 s an image on four threads with images batched eight at a time. Built once a process from
    `load_folds`, whose networks stay as they are for training and for the scored predictions."""
    meta, nets = load_folds(version, n_datasets)
    key = (version, n_datasets)
    if key not in _READING:
        fast = []
        for net, mean, sd in nets:
            net = copy.deepcopy(net).eval()
            for stage in net.trunk:
                for i in range(1, len(stage)):
                    if isinstance(stage[i], nn.BatchNorm2d) and isinstance(stage[i - 1], nn.Conv2d):
                        stage[i - 1] = torch.nn.utils.fusion.fuse_conv_bn_eval(stage[i - 1], stage[i])
                        stage[i] = nn.Identity()
            fast.append((net.to(memory_format=torch.channels_last), mean, sd))
        _READING[key] = fast
    return meta, _READING[key]


def batches_of(pairs, size=READ_BATCH):
    """(path, pixels, how it was read) in runs of `size` whose pictures share one shape."""
    run = []
    for item in pairs:
        if run and (len(run) == size or run[0][1].shape != item[1].shape):
            yield run
            run = []
        run.append(item)
    if run:
        yield run


def write_split(df, split, scores, thresholds, version, folder=None):
    for d in datasets.NAMES:
        rows = []
        for i, r in df[df.dataset == d].iterrows():
            if not r.readable:
                rows.append({"patient": r.patient, "image": r.image, "fold": r.fold, "in_scope": 0,
                             "drop_reason": preds.UNREADABLE})
                continue
            s = float(scores[i])
            rows.append({"patient": r.patient, "image": r.image, "fold": r.fold, "in_scope": 1,
                         "drop_reason": "", "fracture_score": "%.6f" % s,
                         "fracture_pred": int(s >= thresholds[d])})
        preds.write("cnn", d, split, rows, folder=folder)
    preds.collect(folder)


def predict(split, version, batch):
    if version >= TRIAL_FROM:
        # predict writes the scorer's files (pred_cnn_*), which belong to the shipped detector.
        raise SystemExit("version %d is a trial: compare its oof_dev.csv with modalities/xray/eval/compare.py "
                         "instead of predicting into the scorer's files" % version)
    datasets.guard_test(split)
    n_datasets = len(datasets.NAMES)
    meta, nets = load_folds(version, n_datasets)
    with ledger.run("detector", "predict %s v%d" % (split, version)) as led:
        if split == "dev":
            df, arrays, _ = table(("dev",))
            out = model_dir(version)
            oof = pd.read_csv(os.path.join(out, "oof_dev.csv"), dtype={"patient": str})
            key = {(r.dataset, str(r.image)): r.score for r in oof.itertuples()}
            scores = pd.Series([key.get((r.dataset, str(r.image)), np.nan) for _, r in df.iterrows()],
                               index=df.index)
            if scores[df.readable].isna().any():
                raise SystemExit("oof_dev.csv does not cover every development image; retrain.")
        else:
            df, arrays, _ = table(("test",))
            pix = stack(df, arrays)
            rows = df.index.values[df.readable.values]
            acc = np.zeros(len(rows), np.float64)
            for net, mean, sd in nets:
                acc += predict_rows(net, pix, df, rows,
                                    torch.tensor(mean, device=DEV), torch.tensor(sd, device=DEV),
                                    n_datasets, batch)
            scores = pd.Series(np.nan, index=df.index)
            scores.iloc[rows] = acc / len(nets)
        write_split(df, split, scores, meta["thresholds"], version)
        for d in datasets.NAMES:
            m = (df.dataset == d) & df.readable
            led.number("%s.auroc" % d, auroc(scores[m].values, df.truth[m].values))
            led.number("%s.images" % d, int(m.sum()), "count")


def steady_inference():
    """Make a reading repeatable on the machine it runs on.

    Measured 2026-10-01 on twenty held-out radiographs: with cuDNN free to pick an algorithm per
    call, two runs of the same image on the same card disagreed; pinned, they agree exactly. It
    costs nothing here (0.071 s an image against 0.082 with TF32 on), and somebody asking twice
    about their own wrist should not get two answers.

    What this cannot fix is the gap between a processor and a graphics card, measured at 0.0090 on a
    score from 0 to 1: that is different convolution kernels adding up in a different order, not a
    precision setting. 0.14% of the held-out wrist test sits that close to the line.
    `tools/device_agreement.py` is the measurement and `docs/COST_MODEL.md` says what follows from
    it, the first thing being that the shipped route reads on the processor."""
    if DEV.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def score_images(paths, version, dataset="grazpedwri"):
    """The shipped ensemble on one or more radiographs, with the threshold it was scored at.

    `dataset` picks which threshold and which collection one-hot to use: `grazpedwri` for a wrist,
    `fracatlas` for a hand, leg, hip or shoulder."""
    steady_inference()
    n_datasets = len(datasets.NAMES)
    if DEV.type == "cpu":
        return score_images_on_processor(paths, version, dataset, n_datasets)
    meta, nets = load_folds(version, n_datasets)
    which = F.one_hot(torch.tensor([datasets.NAMES.index(dataset)], device=DEV), n_datasets).float()
    out = []
    for p, (pix, kind) in zip(paths, prepare.read_many(paths)):
        x = torch.from_numpy(pix).to(DEV).float().div_(255.0)[None, None]
        acc = 0.0
        with torch.no_grad():
            for net, mean, sd in nets:
                xx = (x - mean) / sd
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                    acc += float(torch.sigmoid(net(xx, which).float())[0])
        score = acc / len(nets)
        out.append({"path": p, "read_as": kind, "score": round(score, 6),
                    "fracture": bool(score >= meta["thresholds"][dataset]),
                    "threshold": meta["thresholds"][dataset], "dataset_profile": dataset,
                    "version": version})
    return out, meta


def score_images_on_processor(paths, version, dataset, n_datasets):
    """`score_images` on the processor, the shipped route: folded networks, eight pictures a call."""
    meta, nets = reading_folds(version, n_datasets)
    one = F.one_hot(torch.tensor([datasets.NAMES.index(dataset)]), n_datasets).float()
    out = []
    for run in batches_of([(p, pix, kind) for p, (pix, kind) in zip(paths, prepare.read_many(paths))]):
        x = torch.from_numpy(np.stack([pix for _, pix, _ in run])).float().div_(255.0)[:, None]
        which = one.expand(len(run), -1)
        acc = torch.zeros(len(run), dtype=torch.float64)
        with torch.inference_mode():
            for net, mean, sd in nets:
                xx = ((x - mean) / sd).contiguous(memory_format=torch.channels_last)
                acc += torch.sigmoid(net(xx, which).float()).double()
        for (p, _, kind), total in zip(run, acc.tolist()):
            score = total / len(nets)
            out.append({"path": p, "read_as": kind, "score": round(score, 6),
                        "fracture": bool(score >= meta["thresholds"][dataset]),
                        "threshold": meta["thresholds"][dataset], "dataset_profile": dataset,
                        "version": version})
    return out, meta


def score_images_with_maps(paths, version, dataset="grazpedwri"):
    """`score_images`, plus the map of where in each picture the score came from.

    This is not a saliency method bolted on afterwards and it is not an approximation. The network
    ends in a global average over the feature map followed by one linear layer, so

        logit = w . mean(F) + (the collection term) = mean(w . F) + (the collection term)

    and `w . F(h, w)` is literally the per-cell decomposition of the number the model produced.
    Averaging it back over the grid returns the logit exactly. What it is NOT is a measured
    location: the detector was scored on whether a fracture is present, never on where, so the map
    is a place to look and the file that carries it says so.

    The grid is 6x6, because that is what the architecture leaves at 384 pixels: a stride-2 stem and
    five halvings. Coarse, and honestly coarse, which is better than an upsampled picture that
    implies a precision nobody measured."""
    steady_inference()
    n_datasets = len(datasets.NAMES)
    meta, nets = load_folds(version, n_datasets)
    which = F.one_hot(torch.tensor([datasets.NAMES.index(dataset)], device=DEV), n_datasets).float()
    out = []
    for p, (pix, kind) in zip(paths, prepare.read_many(paths)):
        x = torch.from_numpy(pix).to(DEV).float().div_(255.0)[None, None]
        acc, maps = 0.0, None
        with torch.no_grad():
            for net, mean, sd in nets:
                xx = (x - mean) / sd
                # float32 and no autocast here: one image is cheap, and a bfloat16 feature map
                # quantises a 36-cell grid enough to move which cell reads as the strongest.
                feat = net.trunk(xx).float()                       # (1, C, 6, 6)
                w = net.head.weight[0, :feat.shape[1]].float()     # the image half of the head
                cam = torch.einsum("c,bchw->bhw", w, feat)[0]
                maps = cam if maps is None else maps + cam
                acc += float(torch.sigmoid(net.logit_from(feat, which).float())[0])
        score = acc / len(nets)
        m = (maps / len(nets)).cpu()
        out.append({"path": p, "read_as": kind, "score": round(score, 6),
                    "fracture": bool(score >= meta["thresholds"][dataset]),
                    "threshold": meta["thresholds"][dataset], "dataset_profile": dataset,
                    "version": version,
                    "map": [float(v) for v in m.flatten().tolist()],
                    "map_size": [int(m.shape[1]), int(m.shape[0])]})
    return out, meta


def box_selftests():
    """The box head (idea 125), on the processor and the tiny widths `selftest` sets. With loc_weight 0 a
    run's steps are the steps without a head, bit for bit; five development FracAtlas pictures' boxes
    land where the picture draws them after letterboxing, and their cell map sets the cells they cover;
    the warped box map is the map of the warped picture to within one cell, for a few random affines."""
    # loc_weight 0 trains exactly as before: the same weights and losses after three steps. The guarantee
    # is draw-order identity, not pinned numbers: a run with loc_weight 0 builds no head and takes the
    # augment path without a cell map, so it draws the same generator values in the same order as a run
    # built without the setting, and the two are compared here rather than against recorded losses.
    runs = []
    for opts in ({"layout": "plain"}, {"layout": "plain", "loc_weight": 0.0}):
        torch.manual_seed(5)
        net = make_net(2, opts)
        assert not any(k.startswith("loc.") for k in net.state_dict()), "loc_weight 0 built a head"
        opt, loss_fn, g = make_opt(net, CFG["lr"], opts), nn.BCEWithLogitsLoss(), torch.Generator().manual_seed(9)
        ls = []
        for step in range(3):
            gx = torch.Generator().manual_seed(100 + step)
            x = torch.rand(8, 1, 64, 64, generator=gx)
            which = F.one_hot(torch.arange(8) % 2, 2).float()
            y = (torch.arange(8) % 2).float()
            ls.append(train_step(net, opt, loss_fn, x, which, y, torch.tensor(0.4), torch.tensor(0.25), g, opts))
        runs.append((torch.stack(ls), net.state_dict()))
    assert torch.equal(runs[0][0], runs[1][0]), "loc_weight 0: the losses differ"
    assert all(torch.equal(v, runs[1][1][k]) for k, v in runs[0][1].items()), "loc_weight 0: a weight differs"
    print("selftest: loc_weight 0 is the same three steps as no head, losses and weights equal", flush=True)

    # with a head, one step moves it; a picture without a box source adds no cell loss
    opts = {"layout": "plain", "loc_weight": 1.0}
    torch.manual_seed(5)
    net = make_net(2, opts)
    opt, loss_fn, g = make_opt(net, CFG["lr"], opts), nn.BCEWithLogitsLoss(), torch.Generator().manual_seed(9)
    cells, ok = torch.zeros(8, 4, 4), torch.tensor([1.0] * 4 + [0.0] * 4)
    cells[:4, 1:3, 1:3] = 1
    before = net.loc.weight.detach().clone()
    x = torch.rand(8, 1, 64, 64, generator=torch.Generator().manual_seed(100))
    train_step(net, opt, loss_fn, x, F.one_hot(torch.arange(8) % 2, 2).float(), (torch.arange(8) % 2).float(),
               torch.tensor(0.4), torch.tensor(0.25), g, opts, None, (cells, ok))
    assert not torch.equal(net.loc.weight, before), "the box head did not train"
    assert float(cell_loss(torch.zeros(8, 4, 4), cells, torch.zeros(8))) == 0.0, "no box source still counts"
    print("selftest: the box head trains, and a picture with no box source is masked", flush=True)

    def dilate(m):
        p = np.pad(m, 1)
        return np.logical_or.reduce([p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)])

    # the boxes of five development FracAtlas pictures, drawn one at a time on the letterboxed picture
    fr = splitmod.frame("fracatlas", "dev", splitmod.load())
    picked = []
    for key in fr.image:
        key = str(key)
        drawn = boxes.source_boxes("fracatlas", key)
        path = datasets.image_path("fracatlas", key)
        if drawn and path is not None:
            picked.append((key, drawn, path))
        if len(picked) == 5:
            break
    assert len(picked) == 5, "fewer than five development FracAtlas pictures with boxes"
    checked, size = 0, prepare.SIZE
    for key, drawn, path in picked:
        arr = prepare.read_grey(path)
        h, w = arr.shape
        for cx, cy, bw, bh in drawn:
            src = np.zeros((h, w), np.float32)
            x0, x1 = int(round((cx - bw / 2) * w)), int(round((cx + bw / 2) * w))
            y0, y1 = int(round((cy - bh / 2) * h)), int(round((cy + bh / 2) * h))
            src[max(y0, 0):max(min(y1, h), 0), max(x0, 0):max(min(x1, w), 0)] = 255
            lit = prepare.letterbox(src, size) >= 128
            r_on, c_on = np.flatnonzero(lit.any(axis=1)), np.flatnonzero(lit.any(axis=0))
            if not r_on.size:
                continue                    # a box under a pixel after the resize: nothing drawn to match
            r0, r1, c0, c1 = r_on[0], r_on[-1] + 1, c_on[0], c_on[-1] + 1
            rect = boxes.canvas_rects([(cx, cy, bw, bh)], w, h, size)[0]
            assert max(abs(c0 - rect[0]), abs(c1 - rect[2]), abs(r0 - rect[1]), abs(r1 - rect[3])) <= 2, \
                "%s: the drawn box is not where the geometry puts it: %s against %s" % (key, (c0, r0, c1, r1), rect)
            m = boxes.cell_map([rect], size)
            tol = 3                                  # the drawn region's edges, within a pixel or two
            for i, j in zip(*np.nonzero(m)):
                assert 16 * i < r1 + tol and 16 * i + 16 > r0 - tol and 16 * j < c1 + tol and 16 * j + 16 > c0 - tol, \
                    "%s: cell (%d, %d) is set away from its box" % (key, i, j)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if 16 * i >= r0 + tol and 16 * i + 16 <= r1 - tol and 16 * j >= c0 + tol and 16 * j + 16 <= c1 - tol:
                        assert m[i, j] == 1, "%s: cell (%d, %d) inside its box is not set" % (key, i, j)
            assert m[(r0 + r1) // 2 // 16, (c0 + c1) // 2 // 16] == 1, "%s: the box centre's cell is not set" % key
            checked += 1
    assert checked >= 5, "only %d boxes could be drawn and checked" % checked
    print("selftest: %d boxes of 5 FracAtlas pictures lie where the prepared picture draws them, cells set" % checked,
          flush=True)

    # the warped box map against the warped picture's own boxes, for four random affines; nearest-cell
    # sampling puts each warped cell at its centre's source cell, so the two agree to within one cell
    key, drawn, path = picked[0]
    arr = prepare.read_grey(path)
    h, w = arr.shape
    rects = boxes.canvas_rects(drawn, w, h, size)
    target = torch.as_tensor(boxes.cell_map(rects, size), dtype=torch.float32)[None]
    rows_n = target.shape[1]
    for seed in range(4):
        _x, warped = augment(torch.zeros(1, 1, size, size), torch.Generator().manual_seed(seed), {}, target)
        flip, theta = affine_draw(1, torch.Generator().manual_seed(seed), "cpu", torch.float32, 1.0, 1.0)
        inv = torch.linalg.inv(torch.cat([theta[0], torch.tensor([[0.0, 0.0, 1.0]])])).numpy()
        expect = np.zeros((rows_n, rows_n), bool)
        for x0, y0, x1, y1 in rects:
            us, vs = np.linspace(x0, x1, 65) / size * 2 - 1, np.linspace(y0, y1, 65) / size * 2 - 1
            u, v = np.meshgrid(us, vs)
            u = -u if bool(flip[0]) else u
            pu, pv = (inv @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))[:2]
            ci, ri = np.floor((pu + 1) / 2 * rows_n).astype(int), np.floor((pv + 1) / 2 * rows_n).astype(int)
            on = (ci >= 0) & (ci < rows_n) & (ri >= 0) & (ri < rows_n)
            expect[ri[on], ci[on]] = True
        got = warped[0].numpy() > 0.5
        assert expect.any() and got.any(), "seed %d: the box left the picture" % seed
        assert not (got & ~dilate(expect)).any() and not (expect & ~dilate(got)).any(), \
            "seed %d: the warped box map is more than one cell from the warped picture's boxes" % seed
    print("selftest: the warped box map is the warped picture's boxes to within one cell, 4 affines", flush=True)

    # a single-cell box survives the warp: bilinear sampling with a 0.5 threshold erased it under a
    # sub-cell shift, nearest-cell sampling keeps it (regression, idea 125 review)
    one = torch.zeros(1, rows_n, rows_n)
    one[0, rows_n // 2, rows_n // 2] = 1
    for seed in range(8):
        _x, warped = augment(torch.zeros(1, 1, size, size), torch.Generator().manual_seed(seed), {}, one)
        assert int(warped.sum()) >= 1, "seed %d: the single-cell box map was erased by the warp" % seed
    print("selftest: a single-cell box map survives the warp, 8 affines", flush=True)


def selftest():
    """Mid-fold resume on the processor with a tiny synthetic fold (about a minute, no data, no card).

    Trains one fold uninterrupted, then trains it again stopped on purpose after epoch 2 (plain loop) and after
    batch 3 of epoch 2 (queued loop, a partial every batch), resumes each, and asserts: the resume starts where
    the partial stopped, its optimizer state, scheduler, best score and early-stop counter equal the saved ones,
    only the remaining epochs run, the finished fold equals the uninterrupted one, the partial is gone after the
    fold's save, and a corrupt partial is renamed .bad with a fresh start."""
    import contextlib
    import io
    import re
    import tempfile
    global DEV, RESUME_PROBE
    saved_env = {k: os.environ.get(k) for k in ("RAIDIOLOGY_SHARE", "RAIDIOLOGY_SAVE_EVERY_S",
                                                "RAIDIOLOGY_CKPT_SELFTEST_STOP", "RAIDIOLOGY_NO_RESUME")}
    saved_cfg, saved_dev = dict(CFG), DEV
    DEV = torch.device("cpu")
    CFG["widths"] = [4, 8, 8, 8, 8]
    os.environ["RAIDIOLOGY_SHARE"] = "1"
    os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
    try:
        g = torch.Generator().manual_seed(3)
        n, epochs, batch = 120, 4, 16
        patient = np.repeat(np.arange(n // 2), 2)
        truth = (patient % 2).astype(np.float32)
        pix = torch.randint(0, 120, (n, 64, 64), dtype=torch.uint8, generator=g)
        pix[torch.as_tensor(truth > 0)] += 100
        df = pd.DataFrame({"dataset_index": 0, "patient": patient, "truth": truth, "fold": patient % 5,
                           "stack_row": np.arange(n), "readable": True})
        rows = np.arange(n)

        def same(a, b):
            if torch.is_tensor(a):
                return torch.is_tensor(b) and a.shape == b.shape and bool(torch.equal(a.cpu(), b.cpu()))
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            return a == b

        def fold_run(partial, fp, opts):
            buf = io.StringIO()
            torch.manual_seed(1)                # train() seeds once per run; here every fold starts alike
            try:
                with contextlib.redirect_stdout(buf):
                    out = train_fold(pix, df, rows, 0, epochs, batch, 7, 1, 0.3, opts, partial, fp)
            except SelfTestStop:
                out = None
            return out, buf.getvalue()

        def epochs_run(text):
            return [int(m) for m in re.findall(r"^  epoch +(\d+) ", text, re.M)]

        def final_same(a, b, what):
            (na, ia, _ta, oa), (nb, ib, _tb, ob) = a, b
            for (ka, va), (kb, vb) in zip(na.state_dict().items(), nb.state_dict().items()):
                assert torch.allclose(va.float(), vb.float(), rtol=1e-4, atol=1e-5), \
                    "%s: weight %s differs" % (what, ka)
            assert abs(ia["val_auroc"] - ib["val_auroc"]) < 1e-6 and ia["epochs"] == ib["epochs"], what
            assert len(ia["epoch_seconds"]) == len(ib["epoch_seconds"]), what
            assert np.allclose(oa, ob, atol=1e-5), "%s: held-out scores differ" % what

        with tempfile.TemporaryDirectory() as tmp:
            for label, opts, stop, resume_at, batch_done in (
                    ("plain loop, stopped after epoch 2", {"layout": "plain"}, "2", 3, 0),
                    ("queued loop, stopped after batch 3 of epoch 2",
                     {"layout": "plain", "sync_every": 2, "schedule": "onecycle"}, "2:3", 2, 3)):
                fp = {"selftest": label}
                path = os.path.join(tmp, "fold0.partial.pt")
                os.environ["RAIDIOLOGY_SAVE_EVERY_S"] = "0" if batch_done else "100000"
                os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
                ref, _ = fold_run(None, None, opts)
                os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = stop
                out, text = fold_run(path, fp, opts)
                assert out is None and os.path.exists(path), "%s: no partial after the stop" % label
                os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP")
                before = ckpt.load_partial(path, fp)
                assert before["epoch"] == resume_at - 1 and (before["mid"] or {}).get("batch", 0) == batch_done, label
                seen = []

                def probe(ck, net, opt, sched, best, since, before=before, seen=seen):
                    seen.append((same(opt.state_dict(), before["opt"]), same(sched.state_dict(), before["sched"]),
                                 best == before["best"] or (best != best and before["best"] != before["best"]),
                                 since == before["since"]))
                RESUME_PROBE = probe
                out, text = fold_run(path, fp, opts)
                RESUME_PROBE = None
                assert "fold 0: resumed mid-fold at epoch %d of %d (partial checkpoint)" % (resume_at, epochs) in text, \
                    "%s: no resume line" % label
                assert seen == [(True, True, True, True)], "%s: restored state differs from the saved: %s" % (label, seen)
                assert epochs_run(text) == list(range(resume_at, epochs + 1)), \
                    "%s: ran epochs %s" % (label, epochs_run(text))
                final_same(ref, out, label)
                save_fold(os.path.join(tmp, "fold0.pt"), {"state": out[0].state_dict()}, path)
                assert not os.path.exists(path) and not os.path.exists(path + ".tmp"), "%s: partial left" % label
                print("selftest: %s: resumed at epoch %d, state equal, finished fold equals the uninterrupted one"
                      % (label, resume_at), flush=True)
            # a corrupt partial: renamed .bad, a fresh start from epoch 1
            os.environ["RAIDIOLOGY_SAVE_EVERY_S"] = "100000"
            path = os.path.join(tmp, "fold1.partial.pt")
            with open(path, "wb") as f:
                f.write(b"not a checkpoint")
            out, text = fold_run(path, {"selftest": "corrupt"}, {"layout": "plain"})
            assert os.path.exists(path + ".bad") and "unreadable" in text, "corrupt partial not renamed .bad"
            assert epochs_run(text) == list(range(1, epochs + 1)) and "resumed mid-fold" not in text, "no fresh start"
            print("selftest: corrupt partial renamed .bad, fresh start", flush=True)
            # the warm start: the donor's trunk lands in a fresh net, a file without weights and an unknown name are refused
            donor, fresh = (Net(CFG["widths"], CFG["dropout"], CFG["stem_stride"], 1) for _ in range(2))
            good, bad = os.path.join(tmp, "donor.pt"), os.path.join(tmp, "nokeys.pt")
            torch.save({"state": donor.state_dict()}, good)
            torch.save({"cfg": {}}, bad)
            n_trunk = sum(1 for k in donor.state_dict() if k.startswith("trunk."))
            assert load_init(fresh, "chest-v103", say=lambda *a: None, path=good) == n_trunk, "init: trunk count"
            assert all(torch.equal(v, fresh.state_dict()[k]) for k, v in donor.state_dict().items()
                       if k.startswith("trunk.")), "init: trunk differs from the donor's"
            for refuse in (lambda: load_init(fresh, "chest-v103", say=lambda *a: None, path=bad),
                           lambda: check_settings({"init": "no-such-reader"})):
                try:
                    refuse()
                except SystemExit:
                    continue
                raise AssertionError("init: a bad file or name was not refused")
            print("selftest: init copies the trunk, refuses a file without weights and an unknown name", flush=True)
        box_selftests()
        print("selftest: ok", flush=True)
    finally:
        DEV, RESUME_PROBE = saved_dev, None
        CFG.clear()
        CFG.update(saved_cfg)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, default=1)
    t.add_argument("--epochs", type=int, default=CFG["epochs"])
    t.add_argument("--batch", type=int, default=CFG["batch"])
    t.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(SETTINGS)))
    b = sub.add_parser("bench", help="milliseconds a training step, for tools/ab.py")
    b.add_argument("--batch", type=int, default=CFG["batch"])
    b.add_argument("--steps", type=int, default=60)
    b.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(SETTINGS)))
    lp = sub.add_parser("loop", help="seconds an epoch of the real training loop on fold 0, for tools/ab.py")
    lp.add_argument("--epochs", type=int, default=4)
    lp.add_argument("--batch", type=int, default=CFG["batch"])
    lp.add_argument("--fold", type=int, default=0)
    lp.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(SETTINGS)))
    sub.add_parser("selftest", help="mid-fold resume, init and box head checks on the processor")
    bx = sub.add_parser("boxes", help="the fracture box maps for the development pictures (loc_weight)")
    bx.add_argument("--size", type=prepare.size_arg, default=prepare.SIZE, help="the picture cache's size")
    hc = sub.add_parser("hotcell", help="share of fractured held-out pictures whose hottest box-head cell is a drawn box")
    hc.add_argument("--version", type=int, default=1)
    p = sub.add_parser("predict")
    p.add_argument("split", choices=["dev", "test"])
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--batch", type=int, default=CFG["batch"])
    i = sub.add_parser("image", help="score one or more radiographs with the shipped ensemble")
    i.add_argument("paths", nargs="+")
    i.add_argument("--version", type=int, default=1)
    i.add_argument("--profile", choices=datasets.NAMES, default="grazpedwri",
                   help="grazpedwri for a wrist, fracatlas for a hand, leg, hip or shoulder")
    a = ap.parse_args()
    if DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(datasets.SEED)
    if a.cmd == "train":
        train(a.version, a.epochs, a.batch, sets.parse(a.set))
    elif a.cmd == "bench":
        bench(a.batch, sets.parse(a.set), a.steps)
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "boxes":
        build_boxes(a.size)
    elif a.cmd == "hotcell":
        hotcell(a.version)
    elif a.cmd == "loop":
        loop(a.epochs, a.batch, sets.parse(a.set), a.fold)
    elif a.cmd == "image":
        rows, meta = score_images(a.paths, a.version, a.profile)
        for r in rows:
            print(json.dumps(r))
        print("# not a medical device. Held-out accuracy of this ensemble is in "
              "modalities/xray/models/fracture-detector.md; it is paediatric-wrist-heavy, so an "
              "adult hip is outside what it was scored on.", flush=True)
    else:
        predict(a.split, a.version, a.batch)


if __name__ == "__main__":
    main()
