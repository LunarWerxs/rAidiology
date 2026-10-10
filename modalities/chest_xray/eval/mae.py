"""Chest reader from masked-autoencoder pretraining (idea 106).

pretrain: a ViT-S/16 learns to rebuild the 90% of each picture that is masked (He et al. 2021, masked autoencoders) on the
unlabelled chest X-ray pool, simsiam's ssl_pool_288.npy, which holds no picture of NIH's test list.
finetune: that encoder with one new linear layer reads NIH ChestX-ray14's fifteen labels, with reader.py's table, folds,
early-stop pictures and labels, and writes reader-v<V>/ in reader.py's format, so compare.py and chest_soup.py logits read it.
finetune-taix: the same encoder on TAIX-Ray's eight findings, with taix.py's rows, early stopping and val report, writing
taix-v<V>/ (val_logits.npz, model.pt) as taix.py train does.
pretrain-mammo: the same MAE on the mammogram pool (omama.py's finished rows, 512 x 320, vertical flips), writing
work/eval/mammo/mae-v<N>/.
finetune-cmmd: that encoder on CMMD with mammo.py's table, folds, pictures and augmentation, writing mammo-v<V>/ as mammo.py
train does, so mammo.py read takes the version.

No pretrained weights: timm's model code builds the ViT from random initialisation, and nothing is downloaded (HF_HUB_OFFLINE=1).

Usage:
  python modalities/chest_xray/eval/mae.py pretrain --version N [--epochs 200] [--until N] [--batch 256] [--card-gb 4.5] [--pool-on-card] [--compile]
  python modalities/chest_xray/eval/mae.py finetune --version V --mae N [--folds 0] [--epochs 75] [--batch 128] [--card-gb 4.5]
  python modalities/chest_xray/eval/mae.py finetune-taix --version V --mae N [--epochs 30] [--batch 128] [--card-gb 4.5]
  python modalities/chest_xray/eval/mae.py pretrain-mammo --version N [--epochs 200] [--until N] [--batch 256] [--card-gb 4.5] [--pool-on-card] [--compile]
  python modalities/chest_xray/eval/mae.py finetune-cmmd --version V --mae-mammo N [--folds 0,1,2,3,4] [--epochs 50] [--batch 32] [--card-gb 4.5]
  python modalities/chest_xray/eval/mae.py selftest
"""
import argparse
import glob
import inspect
import json
import math
import os
import re
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from timm.layers import resample_abs_pos_embed  # noqa: E402
from timm.models.vision_transformer import Block, VisionTransformer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reader  # noqa: E402  (binds xd, core, datasets, prepare and split as reader.py does)
import simsiam  # noqa: E402
import taix  # noqa: E402  (TAIX-Ray's splits, augmentation, predict and val report)
from core import card, ckpt, ledger, paths, polite  # noqa: E402
from core.rows import FileRows  # noqa: E402
# the mammograms: mammo.py's table and pictures, omama.py's pool (appended, so the chest modules above keep their names)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(HERE)), "mammo", "eval"))
import mammo  # noqa: E402  (CMMD's table, folds, pictures and augmentation; its import sets the repo path)
import omama  # noqa: E402  (the pool's files and picture size)

SEED = 20261008
PATCH = 16
MASK_RATIO = 0.9
ENC = {"dim": 384, "depth": 12, "heads": 6}
DEC = {"dim": 256, "depth": 2, "heads": 8}
PRE_PX = 256
FT_PX = 224
MAE_LR_PER_256 = 1.5e-4
FT_LR_PER_256 = 5e-4
FT_WD = 0.05
LAYER_DECAY = 0.55
DROP_PATH = 0.2
FT_WARM_EPOCHS = 5
NO_DECAY = ("cls_token", "pos_embed", "mask_token", "dec_pos")
LOG_EVERY = 200
READER_KEYS = ("image", "patient", "fold", "y", "logit")
POOL_CHUNK_BYTES = 1 << 30    # --pool-on-card reads the memory map this many bytes at a time
CARD_SPARE_GB = 6             # --pool-on-card needs this much free card memory beyond the pool
COMPILE_MODE = "max-autotune-no-cudagraphs"
MAMMO_PX = (omama.PH, omama.PW)   # the mammogram pool's picture, 512 x 320: 32 x 20 patches


def mae_dir(n):
    return os.path.join(paths.WORK, "eval", "chest", "mae-v%d" % n)


def save_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def save_torch(path, obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(reader.DEV.type == "cuda"))


def lease(card_gb, name):
    card.require_card(reader.DEV, torch)
    reader.xd.limit_card_memory(card_gb, name)
    if reader.DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True


def keep_count(n):
    return int(round(n * (1 - MASK_RATIO)))


def as_hw(px):
    """(h, w) of a size: a pair as it is, a square side px as (px, px)."""
    return (px, px) if isinstance(px, (int, np.integer)) else tuple(px)


def shape_settings(px, flip):
    """What a non-square or vertically flipped pretrain adds to its fingerprint and encoder settings: h and w, and the flip
    axis when it is not the chest's horizontal. Nothing for the chest, so its files and partials stay as they were."""
    h, w = as_hw(px)
    out = {"h": h, "w": w} if h != w else {}
    if flip != "h":
        out["flip"] = flip
    return out


def to_float(x, size):
    """uint8 (B, H, W) on the card to float (B, 1, h, w) in [0, 1], area-resized to size when it differs."""
    x = x.float().div_(255.0).unsqueeze(1)
    if tuple(size) != tuple(x.shape[-2:]):
        x = F.interpolate(x, size=tuple(size), mode="area")
    return x


def pool_mean_sd(pix, rows, size, g):
    """reader.mean_sd at any (h, w): mean and sd (0-1 units) of 2,000 pictures of rows, summed in float64 a chunk at a time."""
    pick = np.sort(g.choice(rows, size=min(2000, len(rows)), replace=False))
    s = s2 = 0.0
    n = 0
    for a in range(0, len(pick), 250):
        x = to_float(torch.from_numpy(np.asarray(pix[pick[a:a + 250]])), size).double()
        s += float(x.sum())
        s2 += float(x.square().sum())
        n += x.numel()
    m = s / n
    sd = max(((s2 - n * m * m) / (n - 1)) ** 0.5, 1e-3)
    return torch.tensor(m, dtype=torch.float32, device=reader.DEV), torch.tensor(sd, dtype=torch.float32, device=reader.DEV)


class Rows:
    """The pool's finished rows, by position: pool[r] is the memmap's row rows[r] (r a slice or an index array)."""

    def __init__(self, mem, rows):
        self.mem, self.rows = mem, rows
        self.shape = (len(rows),) + tuple(mem.shape[1:])
        self.nbytes = len(rows) * int(np.prod(mem.shape[1:]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, r):
        return self.mem[self.rows[r]]


def make_vit(px, dim, depth, heads, drop_path=0.0):
    """timm's ViT code: one grey channel, no classifier head, random initialisation. px is a square side or an (h, w)."""
    return VisionTransformer(img_size=px, patch_size=PATCH, in_chans=1, embed_dim=dim, depth=depth, num_heads=heads,
                             num_classes=0, drop_path_rate=drop_path)


def encode(vit, x, keep=None):
    """Class token and patch tokens after the blocks and the norm. With keep (B, k), only those patches take part."""
    t = vit.patch_embed(x)
    if t.dim() == 4:
        t = t.flatten(1, 2)
    t = t + vit.pos_embed[:, 1:]
    if keep is not None:
        t = torch.gather(t, 1, keep[..., None].expand(-1, -1, t.shape[-1]))
    cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(t.shape[0], -1, -1)
    return vit.norm(vit.blocks(torch.cat([cls, t], 1)))


def random_mask(B, n, g, device):
    """keep (B, k): the visible patches' indices; restore (B, n): the inverse shuffle; mask (B, n): 1 where masked."""
    shuffle = torch.argsort(torch.rand(B, n, device=device, generator=g), dim=1)
    restore = torch.argsort(shuffle, dim=1)
    keep = shuffle[:, :keep_count(n)]
    mask = torch.ones(B, n, device=device)
    mask.scatter_(1, keep, 0.0)
    return keep, restore, mask


def patchify(x):
    """(B, 1, H, W) to (B, n, PATCH * PATCH), patches in row-major order (the order patch_embed uses)."""
    B, _, H, W = x.shape
    h, w = H // PATCH, W // PATCH
    return x[:, 0].reshape(B, h, PATCH, w, PATCH).permute(0, 1, 3, 2, 4).reshape(B, h * w, PATCH * PATCH)


class MAE(nn.Module):
    def __init__(self, px, enc=ENC, dec=DEC):
        super().__init__()
        h, w = as_hw(px)
        self.n = (h // PATCH) * (w // PATCH)
        self.vit = make_vit(px, enc["dim"], enc["depth"], enc["heads"])
        self.dec_embed = nn.Linear(enc["dim"], dec["dim"])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec["dim"]))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.n + 1, dec["dim"]))
        self.dec_blocks = nn.Sequential(*[Block(dec["dim"], dec["heads"], mlp_ratio=4.0, qkv_bias=True)
                                          for _ in range(dec["depth"])])
        self.dec_norm = nn.LayerNorm(dec["dim"])
        self.dec_pred = nn.Linear(dec["dim"], PATCH * PATCH)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

    def forward(self, x, g):
        """Mean squared error over the masked patches, each patch's pixels normalised to its own mean and spread (norm_pix_loss)."""
        keep, restore, mask = random_mask(x.shape[0], self.n, g, x.device)
        lat = self.dec_embed(encode(self.vit, x, keep))
        B, D = lat.shape[0], lat.shape[-1]
        masked = self.mask_token.expand(B, self.n - keep.shape[1], -1).to(lat.dtype)
        full = torch.gather(torch.cat([lat[:, 1:], masked], 1), 1, restore[..., None].expand(-1, -1, D))
        d = torch.cat([lat[:, :1], full], 1) + self.dec_pos
        pred = self.dec_pred(self.dec_norm(self.dec_blocks(d)))[:, 1:].float()
        target = patchify(x.float())
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True, unbiased=False) + 1e-6).sqrt()
        return ((pred - target).pow(2).mean(-1) * mask).sum() / mask.sum()


class ChestViT(nn.Module):
    """The encoder, global average pooling of its patch tokens, and one linear layer to the labels."""

    def __init__(self, vit, n_labels):
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.embed_dim, n_labels)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(encode(self.vit, x)[:, 1:].mean(dim=1))


def random_resized(x, g, out_px, flip="h"):
    """(B, 1, S, T) on the card to (B, 1, out_px) (a side, or an (h, w)): per picture a crop of area (0.5, 1) and aspect
    (3/4, 4/3) at a random place, mirrored with probability 0.5 (an affine grid_sample). The mirror is along flip: "h"
    (left-right, the chest's) or "v" (top-bottom, the mammograms': their chest wall is already on the left)."""
    B = x.shape[0]
    oh, ow = as_hw(out_px)

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(B, device=x.device, generator=g)
    area = u(0.5, 1.0)
    ratio = torch.exp(u(math.log(3 / 4), math.log(4 / 3)))
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (u(0.0, 1.0) * 2 - 1) * (1 - w)
    cy = (u(0.0, 1.0) * 2 - 1) * (1 - h)
    sign = torch.where(torch.rand(B, device=x.device, generator=g) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 2] = w * (sign if flip == "h" else 1.0), cx
    theta[:, 1, 1], theta[:, 1, 2] = h * (sign if flip == "v" else 1.0), cy
    grid = F.affine_grid(theta, (B, 1, oh, ow), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def layer_id(name, depth):
    if name.startswith("head.") or name.startswith("vit.norm."):
        return depth + 1
    if name.startswith("vit.blocks."):
        return int(name.split(".")[2]) + 1
    return 0


def param_groups(model, wd, layer_decay=None, depth=None):
    """AdamW groups: no weight decay on 1-D tensors or on the class, position and mask embeddings. With layer_decay each
    group's lr is scaled by layer_decay ** (depth + 1 - layer): the head and final norm are layer depth + 1, block i is i + 1,
    the embeddings 0."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = 0 if layer_decay is None else layer_id(name, depth)
        decay = p.ndim > 1 and not name.endswith(NO_DECAY)
        scale = 1.0 if layer_decay is None else layer_decay ** (depth + 1 - lid)
        groups.setdefault((lid, decay), {"params": [], "lr_scale": scale, "weight_decay": wd if decay else 0.0})["params"].append(p)
    return list(groups.values())


def set_lr(opt, base):
    for gr in opt.param_groups:
        gr["lr"] = base * gr["lr_scale"]


def lr_factor(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def epoch_parts(n, steps, batch, ep):
    """The sorted row indices of each batch of epoch ep: the same order for every run, with or without --pool-on-card."""
    order = np.random.default_rng(SEED + ep).permutation(n)
    return [np.sort(order[a:a + batch]) for a in range(0, steps * batch, batch)]


def load_pool_to_card(pool, dev, say=print):
    """The whole pool as one uint8 tensor on dev, read from the memory map about 1 GB at a time (host memory stays low).
    Returns (tensor, seconds). On a card, refuses when the pool plus CARD_SPARE_GB is more than the card has free."""
    picture = int(np.prod(pool.shape[1:]))
    need = len(pool) * picture
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev)
        if free < need + CARD_SPARE_GB * 2 ** 30:
            raise SystemExit("--pool-on-card needs the pool (%.1f GB) and %d GB more free on the card; %.1f GB is free"
                             % (need / 2 ** 30, CARD_SPARE_GB, free / 2 ** 30))
    t0 = time.time()
    out = torch.empty(tuple(pool.shape), dtype=torch.uint8, device=dev)
    per = max(1, POOL_CHUNK_BYTES // picture)
    for a in range(0, len(pool), per):
        out[a:a + per].copy_(torch.from_numpy(np.ascontiguousarray(pool[a:a + per])))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    secs = time.time() - t0
    say("pool on the card: %d pictures, %.1f GB, read in %.1f s" % (len(pool), need / 2 ** 30, secs))
    return out, secs


def card_batches(pool_t, parts, dev):
    """(rows, uint8 pictures, None) for each batch in parts, gathered on the card: the rows and values reader.prefetch gives."""
    for rows in parts:
        yield rows, pool_t[torch.from_numpy(rows).to(dev)], None


class Compiled:
    """The MAE model's forward through torch.compile, for training only. The plain module keeps the state_dict, so no
    '_orig_mod.' prefix reaches a checkpoint. COMPILE_MODE is tried first; if the compile fails on a call, the default mode
    is used from then on."""

    def __init__(self, model, say=print):
        self.model, self.say = model, say
        self.mode = COMPILE_MODE
        self.fn = torch.compile(model, mode=self.mode)

    def __call__(self, x, g):
        try:
            return self.fn(x, g)
        except Exception as e:  # noqa: BLE001  a compiler that cannot build this mode: the default one is tried once
            if self.mode == "default":
                raise
            self.say("torch.compile mode %s failed (%s: %s); using the default mode"
                     % (self.mode, type(e).__name__, str(e)[:200]))
            self.mode = "default"
            self.fn = torch.compile(self.model)
            return self.fn(x, g)


def save_encoder(vit, path, settings):
    save_torch(path, {"state": {k: v.detach().cpu() for k, v in vit.state_dict().items()}, "settings": settings})


def load_encoder(path, px, drop_path=0.0):
    """(ViT at px with the saved encoder's weights, its settings). The position embeddings are resampled from the saved grid to
    px's grid, the class token kept."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    s = ck["settings"]
    vit = make_vit(px, s["dim"], s["depth"], s["heads"], drop_path)
    state = dict(ck["state"])
    h, w = as_hw(px)
    g = [h // PATCH, w // PATCH]
    if state["pos_embed"].shape[1] != 1 + g[0] * g[1]:
        state["pos_embed"] = resample_abs_pos_embed(state["pos_embed"], new_size=g, num_prefix_tokens=1)
    vit.load_state_dict(state)
    return vit, s


def fit_mae(pool, out, epochs, batch, px, pool_info, enc=ENC, dec=DEC, until=None, say=print, pool_on_card=False,
            compile_model=False, flip="h"):
    """Pretrains to epoch min(until, epochs), continuing from out/partial.pt when it matches. Returns the history list.
    Writes encoder.pt once epoch `epochs` is done. pool_on_card and compile_model are speed options only: neither changes
    the result, so neither is in fp or the encoder's settings; meta.json records them under 'speed'. px is a side or an
    (h, w); flip is the mirror's axis (shape_settings records both when they are not the chest's)."""
    dev = reader.DEV
    os.makedirs(out, exist_ok=True)
    part = os.path.join(out, "partial.pt")
    n = len(pool)
    steps = n // batch
    assert steps >= 1, "the pool (%d) is smaller than one batch (%d)" % (n, batch)
    until = epochs if until is None else min(until, epochs)
    base = MAE_LR_PER_256 * batch / 256
    total, warm = epochs * steps, max(1, round(0.1 * epochs * steps))
    fp = json.loads(json.dumps({"epochs": epochs, "batch": batch, "px": px, "enc": enc, "dec": dec, "mask_ratio": MASK_RATIO,
                                "seed": SEED, "pool": pool_info, **shape_settings(px, flip)}, sort_keys=True))
    torch.manual_seed(SEED)
    mean, sd = pool_mean_sd(pool, np.arange(n), pool.shape[1:], np.random.default_rng(SEED))
    model = MAE(px, enc, dec).to(dev)
    step = Compiled(model, say) if compile_model else model
    opt = torch.optim.AdamW(param_groups(model, 0.05), lr=base, betas=(0.9, 0.95))
    zeros = np.zeros((n, 1), np.float32)
    history, start = [], 1
    ck = ckpt.load_partial(part, fp)
    if ck is not None and 0 < ck["epoch"] <= epochs:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        history, start = ck["history"], ck["epoch"] + 1
        ckpt.set_rng_state(ck["rng"])
        say("resumed from %s after epoch %d" % (part, ck["epoch"]))
    del ck
    pool_t, load_s = load_pool_to_card(pool, dev, say) if pool_on_card else (None, None)
    speed = {"pool_on_card": pool_on_card, "pool_load_seconds": None if load_s is None else round(load_s, 1),
             "compile": compile_model, "compile_mode": None, "pictures_per_second": {}}
    for ep in range(start, until + 1):
        model.train()
        g = torch.Generator(device=dev).manual_seed(SEED + ep)
        parts = epoch_parts(n, steps, batch, ep)
        tot = torch.zeros((), dtype=torch.float64, device=dev)
        t0 = time.time()
        batches = card_batches(pool_t, parts, dev) if pool_t is not None else reader.prefetch(pool, np.arange(n), zeros, parts)
        for i, (_rows, x, _y) in enumerate(batches):
            set_lr(opt, base * lr_factor((ep - 1) * steps + i, total, warm))
            x = (random_resized(to_float(x, pool.shape[1:]), g, px, flip) - mean) / sd
            with amp():
                loss = step(x, g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach().double()
            if (i + 1) % LOG_EVERY == 0:
                say("  epoch %d step %d/%d  loss %.4f  %.0f pictures/s"
                    % (ep, i + 1, steps, float(tot) / (i + 1), (i + 1) * batch / (time.time() - t0)))
        secs = time.time() - t0
        history.append({"epoch": ep, "loss": float(tot) / steps, "seconds": round(secs, 1),
                        "pictures_per_second": round(steps * batch / secs, 1)})
        say("epoch %3d  loss %.4f  %.0f s  %.0f pictures/s" % (ep, history[-1]["loss"], secs, history[-1]["pictures_per_second"]))
        speed["pictures_per_second"][str(ep)] = history[-1]["pictures_per_second"]
        speed["compile_mode"] = step.mode if compile_model else None
        save_json(os.path.join(out, "meta.json"), {"px": px, "epochs": epochs, "batch": batch, "lr": base, "mask_ratio": MASK_RATIO,
                                                   "enc": enc, "dec": dec, "pool": pool_info, "history": history, "speed": speed,
                                                   **shape_settings(px, flip)})
        ckpt.save_partial(part, {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "history": history,
                                 "rng": ckpt.rng_state(), "fingerprint": fp})
    if history and history[-1]["epoch"] == epochs:
        save_encoder(model.vit, os.path.join(out, "encoder.pt"),
                     {"px": px, "dim": enc["dim"], "depth": enc["depth"], "heads": enc["heads"], "patch": PATCH,
                      "mask_ratio": MASK_RATIO, "epochs": epochs, "batch": batch, "seed": SEED, "pool": pool_info,
                      **shape_settings(px, flip)})
    return history


def pretrain(version, epochs, batch, card_gb, until=None, pool_on_card=False, compile_model=False):
    out = mae_dir(version)
    if os.path.exists(os.path.join(out, "encoder.pt")):
        raise SystemExit("%s holds a finished encoder; move it away to pretrain again" % out)
    if not (os.path.exists(simsiam.POOL) and os.path.exists(simsiam.POOL_JSON)):
        raise SystemExit("no pool at %s; build it with: python modalities/chest_xray/eval/simsiam.py pool" % simsiam.POOL)
    with open(simsiam.POOL_JSON, encoding="utf-8") as f:
        counts = json.load(f)
    st = os.stat(simsiam.POOL)
    pool = simsiam.open_pool()
    print("pool: %d pictures of %s, counts %s" % (len(pool), pool.shape[1:], json.dumps(counts)[:300]), flush=True)
    # the card's ceiling for this process must hold the pool as well: the share is the pool's size on top of card_gb
    lease(card_gb + (pool.nbytes / 2 ** 30 if pool_on_card else 0.0), "mae v%d pretrain" % version)
    torch.manual_seed(SEED)
    pool_info = {"file": os.path.basename(simsiam.POOL), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "counts": counts}
    with ledger.run("experiment", "mae pretrain v%d" % version) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "px": PRE_PX, "mask_ratio": MASK_RATIO, "enc": ENC, "dec": DEC,
                           "lr_per_256": MAE_LR_PER_256})
        hist = fit_mae(pool, out, epochs, batch, PRE_PX, pool_info, until=until, pool_on_card=pool_on_card,
                       compile_model=compile_model)
        if os.path.exists(os.path.join(out, "encoder.pt")):
            print("encoder written: %s" % os.path.join(out, "encoder.pt"), flush=True)
        else:
            print("stopped after epoch %d of %d; run again to continue" % (hist[-1]["epoch"], epochs), flush=True)
        led.number("loss", hist[-1]["loss"]).number("seconds", sum(h["seconds"] for h in hist), "s")
        led.number("pictures_per_second", hist[-1]["pictures_per_second"], "pictures/s")
        if reader.DEV.type == "cuda":
            led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)


def mammo_mae_dir(n):
    return os.path.join(mammo.OUT, "mae-v%d" % n)


def pretrain_mammo(version, epochs, batch, card_gb, until=None, pool_on_card=False, compile_model=False):
    """fit_mae on OMAMA-DB's pool: the finished rows only, 512 x 320, mirrored top-bottom (the chest wall is on the left)."""
    out = mammo_mae_dir(version)
    if os.path.exists(os.path.join(out, "encoder.pt")):
        raise SystemExit("%s holds a finished encoder; move it away to pretrain again" % out)
    if not (os.path.exists(omama.POOL_PIX) and os.path.exists(omama.POOL_DONE)):
        raise SystemExit("no pool at %s; build it with: python modalities/mammo/eval/omama.py pool" % omama.POOL_PIX)
    n = os.path.getsize(omama.POOL_DONE)
    if os.path.getsize(omama.POOL_PIX) != n * omama.PH * omama.PW:
        raise SystemExit("%s is not %d pictures of %d x %d; rebuild the pool" % (omama.POOL_PIX, n, omama.PH, omama.PW))
    done = np.array(np.memmap(omama.POOL_DONE, dtype=np.uint8, mode="r", shape=(n,)))
    rows = np.nonzero(done == 1)[0]
    if not len(rows):
        raise SystemExit("the pool has no finished pictures; run omama.py pool first")
    st = os.stat(omama.POOL_PIX)
    pool = Rows(FileRows(omama.POOL_PIX, (n, omama.PH, omama.PW)), rows)
    print("pool: %d of %d pictures, %d x %d" % (len(rows), n, omama.PH, omama.PW), flush=True)
    # the card's ceiling for this process must hold the pool as well: the share is the pool's size on top of card_gb
    lease(card_gb + (pool.nbytes / 2 ** 30 if pool_on_card else 0.0), "mae mammo v%d pretrain" % version)
    torch.manual_seed(SEED)
    pool_info = {"file": os.path.basename(omama.POOL_PIX), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                 "rows_used": int(len(rows))}
    with ledger.run("experiment", "mae mammo pretrain v%d" % version) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "size": [omama.PH, omama.PW], "flip": "v", "mask_ratio": MASK_RATIO,
                           "enc": ENC, "dec": DEC, "lr_per_256": MAE_LR_PER_256})
        hist = fit_mae(pool, out, epochs, batch, MAMMO_PX, pool_info, until=until, pool_on_card=pool_on_card,
                       compile_model=compile_model, flip="v")
        if os.path.exists(os.path.join(out, "encoder.pt")):
            print("encoder written: %s" % os.path.join(out, "encoder.pt"), flush=True)
        else:
            print("stopped after epoch %d of %d; run again to continue" % (hist[-1]["epoch"], epochs), flush=True)
        led.number("loss", hist[-1]["loss"]).number("seconds", sum(h["seconds"] for h in hist), "s")
        led.number("pictures_per_second", hist[-1]["pictures_per_second"], "pictures/s")
        if reader.DEV.type == "cuda":
            led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)


def finetune_fold(df, pix, cache_row, labels, fold, epochs, batch, enc_path, px, part, fingerprint, say=print):
    """Fine-tunes one fold from the saved encoder. Returns (model, info, held-out rows, their logits)."""
    dev = reader.DEV
    tr, val, te, rng, _n_out = reader.fold_split(df, fold, {})
    mean, sd = reader.mean_sd(pix, cache_row[tr], px, rng)
    vit, _ = load_encoder(enc_path, px, DROP_PATH)
    model = ChestViT(vit, labels.shape[1]).to(dev)
    steps = -(-len(tr) // batch)
    total, warm = epochs * steps, min(FT_WARM_EPOCHS, epochs) * steps
    base = FT_LR_PER_256 * batch / 256
    opt = torch.optim.AdamW(param_groups(model, FT_WD, LAYER_DECAY, len(vit.blocks)), lr=base, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_epoch, best_state, took, start = -1.0, 0, None, [], 1
    ck = ckpt.load_partial(part, fingerprint)
    if ck is not None and 0 < ck["epoch"] <= epochs:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best, best_epoch, best_state, took = ck["best"], ck["best_epoch"], ck["best_state"], ck["took"]
        start = ck["epoch"] + 1
        ckpt.set_rng_state(ck["rng"])
        say("fold %d: resumed from %s after epoch %d" % (fold, part, ck["epoch"]))
    del ck
    for ep in range(start, epochs + 1):
        model.train()
        g = torch.Generator(device=dev).manual_seed(SEED + 1000 * fold + ep)
        order = tr[np.random.default_rng(SEED + 1000 * fold + ep).permutation(len(tr))]
        parts = [order[a:a + batch] for a in range(0, len(order), batch)]
        tot = torch.zeros((), dtype=torch.float64, device=dev)
        t0 = time.time()
        for i, (rows, x, y) in enumerate(reader.prefetch(pix, cache_row, labels, parts)):
            set_lr(opt, base * lr_factor((ep - 1) * steps + i, total, warm))
            x = (reader.xd.augment(reader.to_float(x, px), g, {}) - mean) / sd
            with amp():
                out = model(x)
            loss = loss_fn(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach().double() * len(rows)
        took.append(time.time() - t0)
        v, _ = reader.mean_auroc(reader.predict(model, pix, cache_row, labels, val, mean, sd, batch, px), labels[val])
        say("  epoch %2d  loss %.4f  early-stop auroc %.4f  %.0f s" % (ep, float(tot) / len(tr), v, took[-1]))
        if best_state is None or v > best + 1e-4:
            best, best_epoch = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        ckpt.save_partial(part, {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best,
                                 "best_epoch": best_epoch, "best_state": best_state, "took": took, "rng": ckpt.rng_state(),
                                 "fingerprint": fingerprint})
    model.load_state_dict(best_state)
    logit = reader.predict(model, pix, cache_row, labels, te, mean, sd, batch, px)
    info = {"fold": fold, "epochs": len(took), "best_epoch": best_epoch, "early_stop_auroc": best, "mean": float(mean),
            "sd": float(sd), "train_images": len(tr), "early_stop_images": len(val), "predicted_images": len(te),
            "seconds": round(sum(took), 1), "epoch_seconds": [round(t, 2) for t in took]}
    return model, info, te, logit


def run_finetune(out, version, mae, df, pix, cache_row, labels, folds, epochs, batch, px, enc_path, fingerprint, say=print):
    """Fine-tunes each fold from enc_path and writes out/ in reader.py's format (fold<k>.pt, meta.json, oof_logits.npz).
    Returns (meta, the fold infos)."""
    n_lab = labels.shape[1]
    logit = np.full((len(df), n_lab), np.nan, np.float32)
    got_fold = np.full(len(df), -1, np.int8)
    infos = []
    settings = {"arch": "vit-s16", "mae": mae, "px": px, "layer_decay": LAYER_DECAY, "drop_path": DROP_PATH,
                "epochs": epochs, "batch": batch, "lr_per_256": FT_LR_PER_256, "weight_decay": FT_WD}
    for k in folds:
        path = os.path.join(out, "fold%d.pt" % k)
        part = reader.partial_path(out, k)
        net, info, rows, scores = finetune_fold(df, pix, cache_row, labels, k, epochs, batch, enc_path, px, part,
                                                {**fingerprint, "fold": k}, say)
        logit[rows], got_fold[rows] = scores, k
        reader.finish_fold(path, net, info, {**fingerprint, "fold": k}, settings, part)
        infos.append(info)
        say("fold %d: best epoch %d, early-stop auroc %.4f, %.0f s" % (k, info["best_epoch"], info["early_stop_auroc"], info["seconds"]))
    done = got_fold >= 0
    per = [reader.xd.auroc(logit[done, i], labels[done, i].astype(int)) for i in range(n_lab)]
    findings = float(np.nanmean(per[:14]))
    np.savez(os.path.join(out, "oof_logits.npz"), image=df.image.to_numpy(str)[done], patient=df.patient.to_numpy()[done],
             fold=got_fold[done], y=labels[done].astype(np.int8), logit=logit[done])
    meta = {"version": version, "mae": mae, "cfg": {"lr_per_256": FT_LR_PER_256, "weight_decay": FT_WD,
                                                  "layer_decay": LAYER_DECAY, "drop_path": DROP_PATH, "warm_epochs": FT_WARM_EPOCHS},
            "settings": settings, "seed": SEED, "folds": infos, "auroc": dict(zip(reader.datasets.LABELS, per)),
            "auroc_mean14": findings, "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0) if reader.DEV.type == "cuda" else "cpu"}
    save_json(os.path.join(out, "meta.json"), meta)
    return meta, infos


def finetune(version, mae, folds, epochs, batch, card_gb):
    out = reader.out_dir(version)
    if os.path.exists(os.path.join(out, "meta.json")) or glob.glob(os.path.join(out, "fold*.pt")):
        raise SystemExit("%s already holds a finished version; refusing to overwrite it" % out)
    enc_path = os.path.join(mae_dir(mae), "encoder.pt")
    if not os.path.exists(enc_path):
        raise SystemExit("no encoder at %s; pretrain it first" % enc_path)
    reader.datasets.guard_test("dev")
    lease(card_gb, "mae v%d finetune v%d" % (mae, version))
    reader.xd.CFG.update({k: reader.CFG[k] for k in ("rotate_deg", "translate", "jitter")})
    df, pix, cache_row, labels = reader.table(384)
    os.makedirs(out, exist_ok=True)
    print("pictures: %d development images, kept on disk (%s)" % (len(df), pix.shape), flush=True)
    st = os.stat(enc_path)
    fingerprint = json.loads(json.dumps({"version": version, "mae": mae, "epochs": epochs, "batch": batch, "px": FT_PX,
                                         "seed": SEED, "encoder": [st.st_size, st.st_mtime_ns],
                                         "split_sha256": reader.xd.sha256(reader.splitmod.PATH),
                                         "cache": reader.cache_identity(384), "dev_rows": int(len(df))}, sort_keys=True))
    with ledger.run("reader", "mae v%d finetune v%d folds %s" % (mae, version, "".join(map(str, folds)))) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "px": FT_PX, "mae": mae, "layer_decay": LAYER_DECAY,
                           "drop_path": DROP_PATH, "lr_per_256": FT_LR_PER_256})
        meta, infos = run_finetune(out, version, mae, df, pix, cache_row, labels, folds, epochs, batch, FT_PX, enc_path,
                                   fingerprint)
        for info in infos:
            led.number("fold%d.seconds" % info["fold"], info["seconds"], "s").number("fold%d.epochs" % info["fold"], info["epochs"], "count")
        for name, a in meta["auroc"].items():
            led.number("auroc.%s" % name, a)
        led.number("auroc.mean14", meta["auroc_mean14"])
    print("\nout-of-fold AUROC (folds %s) -> %s" % (list(folds), out))
    for name, a in meta["auroc"].items():
        print("  %-20s %.4f" % (name, a))
    print("  %-20s %.4f" % ("mean of 14 findings", meta["auroc_mean14"]), flush=True)


def taix_fit(out, version, mae, t, pix, enc_path, px, epochs, batch, cache_ident, until=None, say=print):
    """Fine-tunes the encoder on TAIX's train rows, early-stopped on the held-out patients (taix.early_stop_split), and
    writes out/model.pt, out/val_logits.npz and out/val_verdict.json as taix.train does. Returns the report, or None when it
    stopped after epoch `until` (out/partial.pt keeps the rest for the next run)."""
    dev = reader.DEV
    os.makedirs(out, exist_ok=True)
    split = t["split"]
    tr, es = taix.early_stop_split(t["patient"], np.flatnonzero(split == "train"))
    val = np.flatnonzero(split == "val")
    assert not (set(t["patient"][tr]) & set(t["patient"][es])), "a patient is on both sides of the early-stop split"
    say("train split %d pictures: %d to train (%d patients), %d to early-stop (%d patients); val %d pictures"
        % ((split == "train").sum(), len(tr), len(set(t["patient"][tr])), len(es), len(set(t["patient"][es])), len(val)))
    st = os.stat(enc_path)
    fp = json.loads(json.dumps({**cache_ident, "px": px, "seed": SEED, "encoder": [st.st_size, st.st_mtime_ns],
                                "train_rows": len(tr), "early_stop_rows": len(es), "val_rows": len(val)}, sort_keys=True))
    torch.manual_seed(SEED)
    # normalisation as taix.fit does it: reader.mean_sd over 2,000 training pictures with taix's seed, at the net's own side (224)
    mean, sd = reader.mean_sd(pix, tr, px, np.random.default_rng(taix.SEED))
    vit, _ = load_encoder(enc_path, px, DROP_PATH)
    model = ChestViT(vit, len(taix.NAMES)).to(dev)
    steps = -(-len(tr) // batch)
    last = epochs if until is None else min(until, epochs)
    total, warm = epochs * steps, min(FT_WARM_EPOCHS, epochs) * steps
    base = FT_LR_PER_256 * batch / 256
    opt = torch.optim.AdamW(param_groups(model, FT_WD, LAYER_DECAY, len(vit.blocks)), lr=base, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    truth = t["truth"]
    y = truth.astype(np.float32)
    cache_row = np.arange(len(pix))
    pace = polite.Pace()
    part = os.path.join(out, "partial.pt")
    best, best_epoch, best_state, took, start = -1.0, 0, None, [], 1
    ck = ckpt.load_partial(part, fp)
    if ck is not None and 0 < ck["epoch"] <= epochs:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best, best_epoch, best_state, took = ck["best"], ck["best_epoch"], ck["best_state"], ck["took"]
        start = ck["epoch"] + 1
        ckpt.set_rng_state(ck["rng"])
        say("resumed from %s after epoch %d" % (part, ck["epoch"]))
    del ck
    for ep in range(start, last + 1):
        model.train()
        g = torch.Generator(device=dev).manual_seed(SEED + ep)
        order = tr[np.random.default_rng(SEED + ep).permutation(len(tr))]
        parts = [order[a:a + batch] for a in range(0, len(order), batch)]
        tot = torch.zeros((), dtype=torch.float64, device=dev)
        t0 = time.time()
        began = t0
        for i, (rows, x, yb) in enumerate(reader.prefetch(pix, cache_row, y, parts)):
            set_lr(opt, base * lr_factor((ep - 1) * steps + i, total, warm))
            x = (taix.augment(reader.to_float(x, px), g, reader.CFG) - mean) / sd     # no flip: it would swap left and right
            with amp():
                z = model(x)
            loss = loss_fn(z.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach().double() * len(rows)
            pace.rest(time.time() - began)
            began = time.time()
        took.append(time.time() - t0)
        z_es = taix.predict(reader, model, pix, truth, es, mean, sd, batch, px, len(taix.NAMES))
        v, _ = reader.mean_auroc(taix.present_logits(z_es, False), truth[es])
        say("  epoch %2d  loss %.4f  early-stop auroc %.4f  %.0f s" % (ep, float(tot) / len(tr), v, took[-1]))
        if best_state is None or v > best + 1e-4:
            best, best_epoch = v, ep
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        ckpt.save_partial(part, {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best,
                                 "best_epoch": best_epoch, "best_state": best_state, "took": took, "rng": ckpt.rng_state(),
                                 "fingerprint": fp})
    if last < epochs:
        say("stopped after epoch %d of %d; run again to continue" % (last, epochs))
        return None
    model.load_state_dict(best_state)
    peak = None
    if dev.type == "cuda":
        peak = (torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30)
    info = {"epochs": len(took), "best_epoch": best_epoch, "early_stop_auroc": best, "mean": float(mean), "sd": float(sd),
            "train_images": len(tr), "early_stop_images": len(es), "val_images": len(val), "seconds": round(sum(took), 1),
            "epoch_seconds": [round(s, 2) for s in took]}
    settings = {"arch": "vit-s16", "mae": mae, "px": px, "layer_decay": LAYER_DECAY, "drop_path": DROP_PATH, "epochs": epochs,
                "batch": batch, "lr_per_256": FT_LR_PER_256, "weight_decay": FT_WD}
    save_torch(os.path.join(out, "model.pt"), {"state": model.state_dict(), "cfg": reader.CFG, "info": info,
                                               "settings": settings, "fingerprint": fp})
    ckpt.clear_partial(part)
    full = taix.predict(reader, model, pix, truth, val, mean, sd, batch, px, len(taix.NAMES))
    logits = taix.present_logits(full, False)
    np.savez(os.path.join(out, "val_logits.npz"), UID=t["uid"][val], PatientID=t["patient"][val], logits=logits,
             truth=truth[val])
    res = taix.bootstrap_auroc(logits, truth[val], t["patient"][val])
    m = taix.print_table("val AUROC (%d pictures, %d patients; patient bootstrap, %d resamples, seed %d)"
                         % (len(val), len(set(t["patient"][val])), taix.BOOT, taix.BOOT_SEED), res)
    line = taix.write_val_verdict(out, m, version)
    say(line)
    return {"info": info, "res": res, "m": m, "line": line, "peak": peak}


def finetune_taix(version, mae, epochs, batch, card_gb):
    out = taix.out_dir(version)
    if any(os.path.exists(os.path.join(out, f)) for f in ("val_logits.npz", "model.pt")):
        raise SystemExit("%s already holds a finished version; refusing to overwrite it" % out)
    enc_path = os.path.join(mae_dir(mae), "encoder.pt")
    if not os.path.exists(enc_path):
        raise SystemExit("no encoder at %s; pretrain it first" % enc_path)
    lease(card_gb, "mae v%d finetune-taix v%d" % (mae, version))
    t, pix = taix.load_table({})
    print("pictures: %d, kept on disk (%s)" % (len(t["uid"]), pix.shape), flush=True)
    st = os.stat(taix.cache_paths(taix.SIZE)[0])
    cache_ident = {"version": version, "mae": mae, "epochs": epochs, "batch": batch, "cache": [st.st_size, st.st_mtime_ns]}
    with ledger.run("experiment", "taix train v%d" % version) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "px": FT_PX, "mae": mae, "layer_decay": LAYER_DECAY,
                           "drop_path": DROP_PATH, "lr_per_256": FT_LR_PER_256, "weight_decay": FT_WD,
                           "cfg": {k: reader.CFG[k] for k in ("rotate_deg", "translate", "jitter")}})
        r = taix_fit(out, version, mae, t, pix, enc_path, FT_PX, epochs, batch, cache_ident)
        info = r["info"]
        led.number("early_stop_auroc", info["early_stop_auroc"]).number("epochs", info["epochs"], "count")
        led.number("seconds", info["seconds"], "s")
        if r["peak"] is not None:
            led.number("peak_memory_gb", r["peak"][0]).number("peak_reserved_memory_gb", r["peak"][1])
            print("peak card memory %.2f GB allocated, %.2f reserved (share %.1f)" % (r["peak"][0], r["peak"][1], card_gb), flush=True)
        for name, (p, lo, hi) in zip(taix.NAMES, r["res"]):
            led.number("val.auroc.%s" % name, p).number("val.lo.%s" % name, lo).number("val.hi.%s" % name, hi)
        led.number("val.auroc.mean8", r["m"])
        led.note_is(r["line"])


def cmmd_predict(net, pix, rows, mean, sd, dev, batch, size):
    """Logits for rows, in rows order; the pictures are area-resized to size as cmmd_fold trains on them."""
    net.eval()
    out = np.zeros(len(rows), np.float32)
    pos = {int(r): i for i, r in enumerate(rows)}
    with torch.no_grad(), amp():
        for xb, _y, r in mammo.batches(pix, rows, np.zeros(len(rows)), batch, False, 0):
            z = net((to_float(xb[:, 0].to(dev), size) - mean) / sd).float().view(-1).cpu().numpy()
            for j, rr in enumerate(r):
                out[pos[int(rr)]] = z[j]
    return out


def cmmd_fold(df, pix, fold, size, epochs, batch, enc_path, part, fingerprint, until=None, say=print):
    """Fine-tunes one CMMD fold from the saved encoder with finetune_fold's recipe, on mammo.py's rows, folds, pictures and
    augmentation (mammo.augment with mammo's default settings), early-stopped on the early-stop patients' AUROC. Returns None
    when it stopped after epoch `until` (part keeps the rest), else a dict: the best epoch's state, the held-out rows and
    logits, the normalisation, the history and the fold's info."""
    dev = reader.DEV
    o = mammo.parse_sets([])
    tr, va, te = mammo.split_fold(df, fold)
    y = df.label.to_numpy().astype(np.float32)
    # the normalisation: 200 seeded training pictures at the net's size, mammo.train_fold's sample
    samp = np.sort(np.random.default_rng(mammo.SEED).choice(tr, min(200, len(tr)), replace=False))
    sx = torch.cat([to_float(torch.from_numpy(np.ascontiguousarray(pix[samp[a:a + 50]])), size) for a in range(0, len(samp), 50)])
    mean, sd = float(sx.mean()), float(sx.std().clamp_min(1e-3))
    del sx
    vit, _ = load_encoder(enc_path, size, DROP_PATH)
    model = ChestViT(vit, 1).to(dev)
    steps = -(-len(tr) // batch)
    last = epochs if until is None else min(until, epochs)
    total, warm = epochs * steps, min(FT_WARM_EPOCHS, epochs) * steps
    base = FT_LR_PER_256 * batch / 256
    opt = torch.optim.AdamW(param_groups(model, FT_WD, LAYER_DECAY, len(vit.blocks)), lr=base, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_epoch, best_state, hist, start = -1.0, 0, None, [], 1
    ck = ckpt.load_partial(part, fingerprint)
    if ck is not None and 0 < ck["epoch"] <= epochs:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best, best_epoch, best_state, hist = ck["best"], ck["best_epoch"], ck["best_state"], ck["hist"]
        start = ck["epoch"] + 1
        ckpt.set_rng_state(ck["rng"])
        say("fold %d: resumed from %s after epoch %d" % (fold, part, ck["epoch"]))
    del ck
    for ep in range(start, last + 1):
        model.train()
        g = torch.Generator(device=dev).manual_seed(SEED + 1000 * fold + ep)
        tot, t0 = torch.zeros((), dtype=torch.float64, device=dev), time.time()
        for i, (xb, yb, rows) in enumerate(mammo.batches(pix, tr, y[tr], batch, True, SEED + 1000 * fold + ep)):
            set_lr(opt, base * lr_factor((ep - 1) * steps + i, total, warm))
            x = (mammo.augment(to_float(xb[:, 0].to(dev), size), o, g) - mean) / sd
            with amp():
                z = model(x)
            loss = loss_fn(z.float(), yb.to(dev).view(-1, 1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach().double() * len(rows)
        secs = time.time() - t0
        v = mammo.auroc(df.label.to_numpy()[va], cmmd_predict(model, pix, va, mean, sd, dev, batch, size))
        hist.append({"epoch": ep, "loss": float(tot) / len(tr), "val_auroc": v, "seconds": round(secs, 1)})
        say("  epoch %2d  loss %.4f  early-stop auroc %.4f  %.0f s" % (ep, hist[-1]["loss"], v, secs))
        if best_state is None or v > best:
            best, best_epoch = v, ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        ckpt.save_partial(part, {"fingerprint": fingerprint, "model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                                 "best": best, "best_epoch": best_epoch, "best_state": best_state, "hist": hist,
                                 "rng": ckpt.rng_state()})
    if last < epochs:
        say("fold %d: stopped after epoch %d of %d; run again to continue" % (fold, last, epochs))
        return None
    model.load_state_dict(best_state)
    logit = cmmd_predict(model, pix, te, mean, sd, dev, batch, size)
    info = {"fold": fold, "epochs": len(hist), "best_epoch": best_epoch, "early_stop_auroc": best, "mean": mean, "sd": sd,
            "train_images": len(tr), "early_stop_images": len(va), "predicted_images": len(te),
            "seconds": round(sum(h["seconds"] for h in hist), 1)}
    return {"info": info, "te": te, "logit": logit, "state": best_state, "best": best, "hist": hist, "mean": mean, "sd": sd}


def cmmd_run(out, version, df, pix, folds, size, enc_path, settings, fp, until=None, say=print):
    """Fine-tunes each fold and writes out/ as mammo.train does: fold<k>.pt, fold<k>.csv, fold<k>.csv.json, then oof.csv from
    every fold csv. A fold whose csv and json carry fp is skipped. Returns False when it stopped after epoch `until`."""
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    for k in folds:
        out_csv = os.path.join(out, "fold%d.csv" % k)
        meta = out_csv + ".json"
        if os.path.exists(out_csv) and os.path.exists(meta) and json.load(open(meta, encoding="utf-8"))["fingerprint"] == fp:
            say("fold %d: already done" % k)
            continue
        part = os.path.join(out, "fold%d.partial.pt" % k)
        with ledger.run("experiment", "mammo v%d train fold %d" % (version, k)) as led:
            led.settings_from(settings)
            if reader.DEV.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            r = cmmd_fold(df, pix, k, size, settings["epochs"], settings["batch"], enc_path, part, fp + "|fold%d" % k, until, say)
            if r is not None:
                f = df.iloc[r["te"]][["path", "patient", "side", "view", "label", "fold"]].copy()
                f["logit"], f["p"] = r["logit"], 1.0 / (1.0 + np.exp(-r["logit"]))
                torch.save({"net": r["state"], "mean": r["mean"], "sd": r["sd"], "settings": settings},
                           os.path.join(out, "fold%d.pt" % k))
                f.to_csv(out_csv, index=False)
                save_json(meta, {"fingerprint": fp, "best_early_stop_auroc": r["best"], "history": r["hist"], **r["info"]})
                ckpt.clear_partial(part)
                led.number("early_stop_auroc", r["best"]).number("held_out_auroc", mammo.auroc(f.label, f.logit))
                led.number("epochs", settings["epochs"], "count")
                if reader.DEV.type == "cuda":
                    led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
                    led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)
        if r is None:
            return False
        say("fold %d: held-out AUROC %.4f -> %s" % (k, mammo.auroc(f.label, f.logit), out_csv))
    parts = [pd.read_csv(p, keep_default_na=False) for p in sorted(glob.glob(os.path.join(out, "fold*.csv")))]
    if parts:
        pd.concat(parts).to_csv(os.path.join(out, "oof.csv"), index=False)
        say("oof.csv: %d pictures from %d folds" % (sum(len(p) for p in parts), len(parts)))
    return True


def finetune_cmmd(version, mae, folds, epochs, batch, card_gb, size=MAMMO_PX, until=None):
    """The CMMD reader from the mammogram encoder, out in mammo.out_dir(version) as mammo.py train writes it. Returns False
    when it stopped early (until)."""
    import pandas as pd
    enc_path = os.path.join(mammo_mae_dir(mae), "encoder.pt")
    if not os.path.exists(enc_path):
        raise SystemExit("no encoder at %s; pretrain it first" % enc_path)
    lease(card_gb, "mae mammo v%d finetune-cmmd v%d" % (mae, version))
    df = mammo.load_index()
    pix = mammo.pixels()
    out = mammo.out_dir(version)
    print("pictures: %d CMMD pictures, kept on disk (%s)" % (len(df), pix.shape), flush=True)
    st = os.stat(enc_path)
    settings = {"arch": "vit-s16", "mae_mammo": mae, "h": size[0], "w": size[1], "epochs": epochs, "batch": batch,
                "layer_decay": LAYER_DECAY, "drop_path": DROP_PATH, "lr_per_256": FT_LR_PER_256, "weight_decay": FT_WD}
    fp = json.dumps({"version": version, "settings": settings, "encoder": [st.st_size, st.st_mtime_ns], "seed": mammo.SEED,
                     "pictures": len(df)}, sort_keys=True)
    if not cmmd_run(out, version, df, pix, folds, size, enc_path, settings, fp, until, print):
        return False
    oof = pd.read_csv(os.path.join(out, "oof.csv"), keep_default_na=False)
    print("\nout-of-fold AUROC (folds %s, %d pictures) -> %s" % (list(folds), len(oof), out))
    print("  %-20s %.4f" % ("malignant", mammo.auroc(oof.label, oof.logit)), flush=True)
    return True


def selftest():
    global lease
    import tempfile
    import types
    import pandas as pd
    import importlib.util
    tiny_enc = {"dim": 64, "depth": 2, "heads": 2}
    tiny_dec = {"dim": 32, "depth": 1, "heads": 2}
    tmp = tempfile.mkdtemp(prefix="mae_selftest_")
    real_dev, real_out_dir = reader.DEV, reader.out_dir
    reader.DEV = torch.device("cpu")
    try:
        # (i) masking keeps round(0.1 x patches) visible per picture, and the restore order is a permutation
        assert keep_count(16) == 2 and keep_count(256) == 26
        g = torch.Generator().manual_seed(1)
        keep, restore, mask = random_mask(5, 16, g, "cpu")
        assert keep.shape == (5, 2) and (mask.sum(1) == 14).all() and (mask.gather(1, keep) == 0).all()
        assert (torch.sort(restore, dim=1).values == torch.arange(16)).all()
        # (ii) one pretrain epoch on a synthetic pool: finite loss, partial.pt holds epoch 1, the next run resumes at epoch 2
        pool = np.random.default_rng(0).integers(0, 256, (40, 64, 64), dtype=np.uint8)
        out = os.path.join(tmp, "mae")
        log = []
        h1 = fit_mae(pool, out, 2, 8, 64, {"synthetic": True}, enc=tiny_enc, dec=tiny_dec, until=1, say=log.append)
        assert len(h1) == 1 and math.isfinite(h1[0]["loss"]), h1
        assert ckpt.load_partial(os.path.join(out, "partial.pt"), json.loads(json.dumps(
            {"epochs": 2, "batch": 8, "px": 64, "enc": tiny_enc, "dec": tiny_dec, "mask_ratio": MASK_RATIO, "seed": SEED,
             "pool": {"synthetic": True}}, sort_keys=True)))["epoch"] == 1
        assert not os.path.exists(os.path.join(out, "encoder.pt"))
        h2 = fit_mae(pool, out, 2, 8, 64, {"synthetic": True}, enc=tiny_enc, dec=tiny_dec, say=log.append)
        assert [h["epoch"] for h in h2] == [1, 2] and math.isfinite(h2[-1]["loss"])
        assert any("resumed" in s for s in log) and os.path.exists(os.path.join(out, "encoder.pt"))
        # (iii) the encoder loads into a fine-tune ViT at 128 px with its position embeddings resampled from 4x4 to 8x8
        enc_path = os.path.join(out, "encoder.pt")
        vit, s = load_encoder(enc_path, 128, DROP_PATH)
        saved = torch.load(enc_path, map_location="cpu", weights_only=False)["state"]
        assert vit.pos_embed.shape == (1, 1 + 64, 64) and torch.equal(vit.patch_embed.proj.weight, saved["patch_embed.proj.weight"])
        assert torch.allclose(vit.pos_embed[:, :1], saved["pos_embed"][:, :1])
        src = inspect.getsource(sys.modules[__name__])
        assert not any(w in src for w in ("pretrained" + "=True", "hf_" + "hub", "load_state_dict_" + "from_url"))
        # (iv) a one-epoch fine-tune on a synthetic two-fold table writes reader's files, and chest_soup's logits read them
        n = 200
        df = pd.DataFrame({"fold": np.arange(n) % 2, "patient": np.arange(n), "image": ["p%d.png" % i for i in range(n)]})
        pix128 = np.random.default_rng(2).integers(0, 256, (n, 128, 128), dtype=np.uint8)
        labels = (np.random.default_rng(3).random((n, len(reader.datasets.LABELS))) < 0.5).astype(np.float32)
        cache_row = np.arange(n)
        fp = {"selftest": True}
        outs = {v: os.path.join(tmp, "reader-v%d" % v) for v in (1, 2)}
        for v in (1, 2):
            os.makedirs(outs[v], exist_ok=True)
            run_finetune(outs[v], v, 1, df, pix128, cache_row, labels, [0], 1, 16, 128, enc_path, fp, say=log.append)
        keys = set(np.load(os.path.join(outs[1], "oof_logits.npz")).files)
        assert keys == set(READER_KEYS), keys
        src_reader = inspect.getsource(reader.train)
        block = re.search(r'np\.savez\(os\.path\.join\(out, "oof_logits\.npz"\),(.*?)\)\n', src_reader, re.S).group(1)
        assert keys == set(re.findall(r"(\w+)=", block)), block
        meta = json.load(open(os.path.join(outs[1], "meta.json"), encoding="utf-8"))
        meta_block = re.search(r"meta = \{(.*?)\}\n", src_reader, re.S).group(1)
        assert set(re.findall(r'"(\w+)":', meta_block)) <= set(meta) | {"mae"}, (meta_block, set(meta))
        assert meta["mae"] == 1 and len(meta["folds"]) == 1 and len(meta["auroc"]) == len(reader.datasets.LABELS)
        info_block = re.search(r'info = \{"fold": fold, (.*?)\}\n', inspect.getsource(reader.train_fold), re.S).group(1)
        assert set(re.findall(r'"(\w+)":', info_block)) <= set(meta["folds"][0]), info_block
        fold = torch.load(os.path.join(outs[1], "fold0.pt"), map_location="cpu", weights_only=False)
        assert set(fold) == {"state", "cfg", "info", "settings", "fingerprint"}
        assert not os.path.exists(reader.partial_path(outs[1], 0)) and not glob.glob(os.path.join(outs[1], "*.partial.pt"))
        spec = importlib.util.spec_from_file_location("chest_soup", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "tools", "chest_soup.py"))
        chest_soup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chest_soup)
        out3 = os.path.join(tmp, "reader-v3")
        reader.out_dir = lambda v: {1: outs[1], 2: outs[2], 3: out3}[v]
        chest_soup.logits(types.SimpleNamespace(versions=[1, 2], out=3))
        assert os.path.exists(os.path.join(out3, "oof_logits.npz")) and os.path.exists(os.path.join(out3, "meta.json"))
        reader.out_dir = real_out_dir
        # (v) layer-wise decay: the head has the full lr, the embeddings the lowest, 0.55 ** (depth + 1)
        model = ChestViT(make_vit(64, 64, 2, 2), 15)
        groups = param_groups(model, FT_WD, LAYER_DECAY, 2)
        head_scale = [gr["lr_scale"] for gr in groups if any(p is model.head.weight for p in gr["params"])]
        assert head_scale == [1.0], head_scale
        assert abs(min(gr["lr_scale"] for gr in groups) - LAYER_DECAY ** 3) < 1e-12
        assert sum(len(gr["params"]) for gr in groups) == len(list(model.parameters()))
        # (vi) --pool-on-card: for two epochs the card's batches are the prefetch path's, row for row and float for float,
        # and two epochs of training reach the same loss with and without it (to 1e-6); the saved encoder has plain keys
        dev = reader.DEV
        steps6 = len(pool) // 8
        pool_t, _secs = load_pool_to_card(pool, dev, say=log.append)
        zeros6 = np.zeros((len(pool), 1), np.float32)
        for ep in (1, 2):
            parts6 = epoch_parts(len(pool), steps6, 8, ep)
            ga = torch.Generator(device=dev).manual_seed(SEED + ep)
            gb = torch.Generator(device=dev).manual_seed(SEED + ep)
            pre = list(reader.prefetch(pool, np.arange(len(pool)), zeros6, parts6))
            card = list(card_batches(pool_t, parts6, dev))
            assert len(pre) == len(card) == steps6, (len(pre), len(card))
            for (ra, xa, _), (rb, xb, _) in zip(pre, card):
                assert np.array_equal(ra, rb) and torch.equal(xa, xb), "epoch %d: the card's pictures differ" % ep
                fa = random_resized(reader.to_float(xa, 64), ga, 32)
                fb = random_resized(reader.to_float(xb, 64), gb, 32)
                assert torch.equal(fa, fb), "epoch %d: the floats after to_float and random_resized differ" % ep
        runs = {}
        for name, on_card in (("prefetch", False), ("card", True)):
            runs[name] = fit_mae(pool, os.path.join(tmp, "speed-" + name), 2, 8, 64, {"synthetic": True}, enc=tiny_enc,
                                 dec=tiny_dec, say=log.append, pool_on_card=on_card)
        assert [h["epoch"] for h in runs["card"]] == [1, 2]
        assert abs(runs["card"][-1]["loss"] - runs["prefetch"][-1]["loss"]) <= 1e-6, (runs["card"][-1], runs["prefetch"][-1])
        for name, on_card in (("prefetch", False), ("card", True)):
            meta_s = json.load(open(os.path.join(tmp, "speed-" + name, "meta.json"), encoding="utf-8"))
            assert meta_s["speed"]["pool_on_card"] is on_card and set(meta_s["speed"]["pictures_per_second"]) == {"1", "2"}
        enc_keys = torch.load(os.path.join(tmp, "speed-card", "encoder.pt"), map_location="cpu", weights_only=False)["state"]
        assert not any("_orig_mod." in k for k in enc_keys)
        # the options parse (--compile itself is not run here: it needs a compiler)
        a1 = build_parser().parse_args(["pretrain", "--version", "1", "--pool-on-card", "--compile"])
        assert a1.pool_on_card and a1.compile_model
        a0 = build_parser().parse_args(["pretrain", "--version", "1"])
        assert not a0.pool_on_card and not a0.compile_model
        # (vii) finetune-taix on a synthetic TAIX table: stopped after epoch 1, resumed, finished; the CLI refuses a finished folder
        n7 = 150
        t7 = {"uid": np.array(["u%d" % i for i in range(n7)]), "patient": np.arange(n7) // 3,
              "split": np.array(["train"] * 120 + ["val"] * 30),
              "truth": (np.random.default_rng(4).random((n7, len(taix.NAMES))) < 0.5).astype(np.int8)}
        pix7 = np.random.default_rng(5).integers(0, 256, (n7, 64, 64), dtype=np.uint8)
        out7 = os.path.join(tmp, "taix-v7")
        ident7 = {"version": 7, "mae": 1, "epochs": 2, "batch": 16, "cache": [0, 0]}
        log7 = []
        assert taix_fit(out7, 7, 1, t7, pix7, enc_path, 64, 2, 16, ident7, until=1, say=log7.append) is None
        assert os.path.exists(os.path.join(out7, "partial.pt")) and not os.path.exists(os.path.join(out7, "model.pt"))
        r7 = taix_fit(out7, 7, 1, t7, pix7, enc_path, 64, 2, 16, ident7, say=log7.append)
        assert any("resumed" in s for s in log7) and len(r7["info"]["epoch_seconds"]) == 2, log7
        z7 = np.load(os.path.join(out7, "val_logits.npz"))
        assert set(z7.files) == {"UID", "PatientID", "logits", "truth"} and z7["logits"].shape == (30, len(taix.NAMES))
        m7 = torch.load(os.path.join(out7, "model.pt"), map_location="cpu", weights_only=False)
        assert m7["settings"]["arch"] == "vit-s16" and set(m7) == {"state", "cfg", "info", "settings", "fingerprint"}
        assert not os.path.exists(os.path.join(out7, "partial.pt")) and os.path.exists(os.path.join(out7, "val_verdict.json"))
        real_taix_out = taix.out_dir
        taix.out_dir = lambda v: out7
        try:
            finetune_taix(7, 1, 2, 16, 4.5)
        except SystemExit as e:
            assert "already holds a finished version" in str(e), e
        else:
            raise AssertionError("finetune-taix did not refuse a folder that holds model.pt")
        finally:
            taix.out_dir = real_taix_out
        a7 = build_parser().parse_args(["finetune-taix", "--version", "7", "--mae", "1"])
        assert a7.cmd == "finetune-taix" and a7.epochs == 30 and a7.batch == 128
        # (viii) the mammogram encoder: a non-square MAE on an (n, 80, 48) pool (sides divide by PATCH), stopped after epoch 1 and
        # resumed; encoder.pt records h, w and the flip; the vertical flip never mirrors left-right, the horizontal one does
        pool8 = np.random.default_rng(7).integers(0, 256, (40, 80, 48), dtype=np.uint8)
        out8 = os.path.join(tmp, "mae-v1")
        log8 = []
        h8 = fit_mae(pool8, out8, 2, 8, (80, 48), {"synthetic": True}, enc=tiny_enc, dec=tiny_dec, until=1, say=log8.append,
                     flip="v")
        assert len(h8) == 1 and math.isfinite(h8[0]["loss"]) and not os.path.exists(os.path.join(out8, "encoder.pt"))
        h8 = fit_mae(pool8, out8, 2, 8, (80, 48), {"synthetic": True}, enc=tiny_enc, dec=tiny_dec, say=log8.append, flip="v")
        assert [h["epoch"] for h in h8] == [1, 2] and any("resumed" in s for s in log8)
        enc8 = os.path.join(out8, "encoder.pt")
        s8 = torch.load(enc8, map_location="cpu", weights_only=False)["settings"]
        assert s8["h"] == 80 and s8["w"] == 48 and s8["flip"] == "v", s8
        vit8, _ = load_encoder(enc8, (80, 48), DROP_PATH)
        assert vit8.pos_embed.shape == (1, 1 + 5 * 3, 64), vit8.pos_embed.shape
        assert patchify(torch.zeros(2, 1, 80, 48)).shape == (2, 15, PATCH * PATCH)
        # a crop of area 0.5 or more always holds the centre column, so a picture bright on its left half stays so under "v";
        # top-bottom mirrors happen; "h" mirrors some left-right
        left8 = torch.zeros(16, 1, 80, 48)
        left8[..., :24] = 1.0
        top8 = torch.zeros(16, 1, 80, 48)
        top8[:, :, :40] = 1.0
        o8 = random_resized(left8, torch.Generator().manual_seed(3), (80, 48), flip="v")
        assert (o8[..., :24].mean((1, 2, 3)) > o8[..., 24:].mean((1, 2, 3))).all()
        o8 = random_resized(top8, torch.Generator().manual_seed(3), (80, 48), flip="v")
        assert (o8[:, :, :40].mean((1, 2, 3)) < o8[:, :, 40:].mean((1, 2, 3))).any()
        o8 = random_resized(left8, torch.Generator().manual_seed(3), (80, 48), flip="h")
        assert (o8[..., :24].mean((1, 2, 3)) < o8[..., 24:].mean((1, 2, 3))).any()
        rw8 = Rows(np.arange(10 * 2 * 3, dtype=np.uint8).reshape(10, 2, 3), np.array([1, 4, 7]))
        assert len(rw8) == 3 and rw8.shape == (3, 2, 3) and rw8.nbytes == 18
        assert np.array_equal(rw8[1:3], rw8.mem[[4, 7]]) and np.array_equal(rw8[np.array([0, 2])], rw8.mem[[1, 7]])
        # (ix) finetune-cmmd on a synthetic CMMD table: mammo's load_index, pixels and out_dir point at the temp folder, the card
        # lease and the ledger are stubbed; five folds, stopped after epoch 1 and resumed; mammo.py's files; aligned_oof reads it
        import contextlib
        import io

        class _Unrecorded:
            """Stands in for a ledger run: the selftest writes no ledger row."""

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __getattr__(self, name):
                return lambda *a, **k: self

        n9 = 60
        df9 = pd.DataFrame({"path": ["c%d.png" % i for i in range(n9)], "patient": [str(i // 2) for i in range(n9)],
                            "side": ["L" if i % 2 == 0 else "R" for i in range(n9)], "view": ["CC"] * n9,
                            "label": ((np.arange(n9) // 2) % 3 == 0).astype(int), "fold": (np.arange(n9) // 2) % 5})
        pix9 = np.random.default_rng(8).integers(0, 256, (n9, 128, 80), dtype=np.uint8)
        real_cmmd = (mammo.load_index, mammo.pixels, mammo.out_dir, mammo.OUT, lease, ledger.run)
        mammo.load_index, mammo.pixels = (lambda: df9), (lambda mode="r": pix9)
        mammo.out_dir, mammo.OUT = (lambda v: os.path.join(tmp, "mammo-v%d" % v)), tmp
        lease, ledger.run = (lambda *a, **k: None), (lambda *a, **k: _Unrecorded())
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert finetune_cmmd(1, 1, (0, 1, 2, 3, 4), 2, 16, 4.5, size=(80, 48), until=1) is False
            out9 = os.path.join(tmp, "mammo-v1")
            assert os.path.exists(os.path.join(out9, "fold0.partial.pt")) and not os.path.exists(os.path.join(out9, "fold0.csv"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert finetune_cmmd(1, 1, (0, 1, 2, 3, 4), 2, 16, 4.5, size=(80, 48)) is True
            assert "fold 0: resumed" in buf.getvalue(), buf.getvalue()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                finetune_cmmd(1, 1, (0, 1, 2, 3, 4), 2, 16, 4.5, size=(80, 48))
            assert buf.getvalue().count("already done") == 5, buf.getvalue()
            c9 = pd.read_csv(os.path.join(out9, "fold0.csv"))
            assert list(c9.columns) == ["path", "patient", "side", "view", "label", "fold", "logit", "p"], list(c9.columns)
            assert len(c9) == int((df9.fold == 0).sum()) and c9.logit.notna().all()
            f9 = torch.load(os.path.join(out9, "fold0.pt"), map_location="cpu", weights_only=False)
            assert set(f9) == {"net", "mean", "sd", "settings"} and f9["settings"]["h"] == 80 and f9["settings"]["w"] == 48
            assert f9["settings"]["arch"] == "vit-s16" and f9["settings"]["mae_mammo"] == 1
            j9 = json.load(open(os.path.join(out9, "fold0.csv.json"), encoding="utf-8"))
            assert {"fingerprint", "best_early_stop_auroc", "history"} <= set(j9) and len(j9["history"]) == 2
            assert not os.path.exists(os.path.join(out9, "fold0.partial.pt"))
            oof9 = pd.read_csv(os.path.join(out9, "oof.csv"))
            assert len(oof9) == n9 and set(oof9.fold) == set(range(5))
            assert mammo.aligned_oof([1], df9)["v1"].notna().all()
        finally:
            mammo.load_index, mammo.pixels, mammo.out_dir, mammo.OUT, lease, ledger.run = real_cmmd
        a9 = build_parser().parse_args(["finetune-cmmd", "--version", "1", "--mae-mammo", "1"])
        assert a9.cmd == "finetune-cmmd" and a9.folds == [0, 1, 2, 3, 4] and a9.epochs == 50 and a9.batch == 32 and a9.mae == 1
        p9 = build_parser().parse_args(["pretrain-mammo", "--version", "1", "--pool-on-card", "--compile"])
        assert p9.pool_on_card and p9.compile_model and p9.epochs == 200 and p9.batch == 256
    finally:
        reader.DEV, reader.out_dir = real_dev, real_out_dir
    print("selftest ok")


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--until", type=int, default=None, help="stop after this epoch (the run resumes from partial.pt next time); default: --epochs")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--card-gb", type=float, default=4.5)
    p.add_argument("--pool-on-card", action="store_true", default=False,
                   help="load the whole pool onto the card once (a big card only); the same batches, values and loss")
    p.add_argument("--compile", dest="compile_model", action="store_true", default=False,
                   help="torch.compile the MAE forward for training (speed only; checkpoints keep plain keys)")
    f = sub.add_parser("finetune")
    f.add_argument("--version", type=int, required=True)
    f.add_argument("--mae", type=int, required=True)
    f.add_argument("--folds", type=int, nargs="+", default=[0])
    f.add_argument("--epochs", type=int, default=75)
    f.add_argument("--batch", type=int, default=128)
    f.add_argument("--card-gb", type=float, default=4.5)
    t = sub.add_parser("finetune-taix")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--mae", type=int, required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--card-gb", type=float, default=4.5)
    m = sub.add_parser("pretrain-mammo")
    m.add_argument("--version", type=int, required=True)
    m.add_argument("--epochs", type=int, default=200)
    m.add_argument("--until", type=int, default=None, help="stop after this epoch (the run resumes from partial.pt next time); default: --epochs")
    m.add_argument("--batch", type=int, default=256)
    m.add_argument("--card-gb", type=float, default=4.5)
    m.add_argument("--pool-on-card", action="store_true", default=False,
                   help="load the finished pool onto the card once (a big card only); the same batches, values and loss")
    m.add_argument("--compile", dest="compile_model", action="store_true", default=False,
                   help="torch.compile the MAE forward for training (speed only; checkpoints keep plain keys)")
    c = sub.add_parser("finetune-cmmd")
    c.add_argument("--version", type=int, required=True)
    c.add_argument("--mae-mammo", dest="mae", type=int, required=True)
    c.add_argument("--folds", type=folds_arg, default="0,1,2,3,4", help="comma-separated fold numbers, as mammo.py train takes")
    c.add_argument("--epochs", type=int, default=50)
    c.add_argument("--batch", type=int, default=32)
    c.add_argument("--card-gb", type=float, default=4.5)
    sub.add_parser("selftest")
    return ap


def folds_arg(s):
    """--folds 0,1,2: the fold numbers, each one of CMMD's folds."""
    ks = [int(v) for v in s.split(",") if v.strip() != ""]
    if not ks or any(k not in range(mammo.FOLDS) for k in ks):
        raise argparse.ArgumentTypeError("folds are numbers from 0 to %d, comma separated: %r" % (mammo.FOLDS - 1, s))
    return ks


def main():
    a = build_parser().parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "pretrain":
        pretrain(a.version, a.epochs, a.batch, a.card_gb, a.until, a.pool_on_card, a.compile_model)
    elif a.cmd == "pretrain-mammo":
        pretrain_mammo(a.version, a.epochs, a.batch, a.card_gb, a.until, a.pool_on_card, a.compile_model)
    elif a.cmd == "finetune-cmmd":
        finetune_cmmd(a.version, a.mae, a.folds, a.epochs, a.batch, a.card_gb)
    elif a.cmd == "finetune-taix":
        finetune_taix(a.version, a.mae, a.epochs, a.batch, a.card_gb)
    else:
        finetune(a.version, a.mae, tuple(a.folds), a.epochs, a.batch, a.card_gb)


if __name__ == "__main__":
    main()
