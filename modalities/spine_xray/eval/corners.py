"""The spine X-ray corner reader: a small heatmap network, trained from scratch, that places the
vertebra corners drawn in BUU-LSPINE (20 on a front view, L1 to L5 upper and lower edges, 22 on a side
view, adding S1's upper edge).

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646 (LICENSING.md, BUU-LSPINE row; the point order is in buu.py).
Plan and pass bars: docs/efficiency/README.md, "Spine X-ray: vertebra corners and slipped vertebrae".

From scratch, no pretrained weights of any kind: ImageNet's terms are non-commercial and rAidiology has
to stay commercially clean (LICENSING.md). A U-Net (stride-2 stem, five levels, bilinear upsampling) on a
512 x 384 canvas (aspect kept, padded black) with one output map per corner at half resolution. The
position is a spatial soft-argmax of each map (sub-pixel), trained with an L1 loss on the coordinates
plus a cross-entropy of each map's softmax against a Gaussian at the drawn corner, which keeps the maps
peaked instead of letting the soft-argmax average over several blobs.

As in detector.py the pixels go once into one uint8 tensor on the card and every batch is an index into
it; augmentation (a small rotation, scale and shift as ONE batched affine warp that moves the corner
points with the same matrix, then a brightness, contrast and gamma nudge) runs on the card. On a front
view a horizontal flip is also drawn, and it swaps the two points of every line (point 0 of a line is the
end on the image's left, so after a flip it is the other end). A side view is never flipped: all side
films face left, and a mirrored film would be a different task.

Folds are five, by patient (split.json from `buu.py split`); fold 0 judges development and no test set is
cut. No early stopping slice either: the last epoch is the model. Each fold writes
<view>-fold<k>.npz holding the held-out patients' predicted corners in ORIGINAL pixel coordinates, the
drawn corners, and two mean-shape baselines. The baseline of the bar (`base`) is each training fold's mean
shape (corners as offsets from their centroid, in units of their RMS spread), scaled by the training
images' mean spread and placed at the training images' mean centroid, in original pixels. It looks at
nothing in the picture, not even its size. `base_size` is a stronger one for context: the same shape, but
spread and centroid taken as shares of the image's own width and height.

Usage: python corners.py train --view AP|LA --version N [--folds 0] [--epochs 80] [--set key=<json> ...]
       python corners.py eval  --view AP|LA --version N [--folds 0]
       python corners.py compare --view AP|LA --a N --b M [--folds 0]     (paired, B minus A)
Setting init="chest-v103" (--set init=\"chest-v103\" with --set widths=[32,64,128,256,256]) starts the encoder from the chest
reader's trunk (the one place this stops being "from scratch"; our own weights, trained on NIH ChestX-ray).
Setting landmark_fold=K (idea 119, 2026-10-08; needs extra_landmarks=true and --view LA): the landmark set's fold K never
trains; the other four folds' pictures join every BUU fold's training, prepared without their labels (landmarks_test.py
training_set: read and oriented as the test reads them, points through the same frames).
Outputs: <WORK>/eval/spine_xray/corners-v<N>/
"""
import argparse
import copy
import json
import math
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import card, ledger, polite  # noqa: E402
from core import ckpt, foldck  # noqa: E402
from core import paths, sets  # noqa: E402
import buu  # noqa: E402


def _device():
    """The GPU when there really is one (RAIDIOLOGY_DEVICE overrides), as detector.py picks it."""
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
STRIDE = 2            # the output maps are at half the canvas' resolution
BOOT = 1000
BOOT_SEED = 0
LEASE_GB = 2.0        # the card share this job asks for (docs/efficiency: about 2 GB)

SETTINGS = {
    "height": 512, "width": 384,
    "epochs": 80, "batch": 12, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 3,
    "widths": [32, 48, 96, 160, 256],
    "rotate_deg": 8.0, "scale_jitter": 0.10, "shift": 0.05,
    "jitter": 0.15, "gamma": 0.3,
    "flip": True,
    "sigma": 6.0,         # the Gaussian target's spread, canvas pixels
    "heat_weight": 0.5,   # the cross-entropy's weight against the L1 on coordinates
    "l1_unit": 8.0,       # canvas pixels the L1 is divided by
    "seed": 0,
    "extra_landmarks": False,   # side view only: add the spondylolisthesis-landmarks pictures to every fold's training
    "init": None,               # "chest-v103": start the encoder from our NIH chest reader's trunk (needs widths [32, 64, 128, 256, 256])
    "landmark_fold": None,      # idea 119: 0 to 4, the landmark set's fold kept out of training (needs extra_landmarks, LA)
}
INITS = {"chest-v103": os.path.join(paths.WORK, "eval", "chest", "reader-v103", "fold0.pt")}
CHEST_WIDTHS = [32, 64, 128, 256, 256]
HELP = {
    "height": "canvas height, pixels", "width": "canvas width, pixels", "epochs": "passes over the training images",
    "batch": "images a step", "lr": "AdamW peak learning rate (linear warm-up, then cosine)",
    "weight_decay": "AdamW weight decay", "warmup": "warm-up epochs", "widths": "channels of the five levels",
    "rotate_deg": "rotation drawn uniformly in +-this", "scale_jitter": "scale drawn in 1 +- this",
    "shift": "shift drawn in +- this share of the canvas", "jitter": "brightness and contrast nudge",
    "gamma": "log of the gamma exponent drawn in +-this", "flip": "front view only: draw a horizontal flip",
    "sigma": "target Gaussian spread in canvas pixels", "heat_weight": "weight of the heat-map cross-entropy",
    "l1_unit": "canvas pixels the coordinate L1 is divided by", "seed": "shifts every draw",
    "extra_landmarks": "side view only: train on the spondylolisthesis-landmarks pictures too (landmarks.py; scaled to BUU's vertebra "
                       "size, on the same canvas, in every fold's training and never held out; points the set lacks are masked "
                       "out of both losses); off = nothing changes, fingerprint included",
    "init": "start the encoder from our own NIH chest reader's trunk: \"chest-v103\" (widths must be [32, 64, 128, 256, 256]; "
            "decoder and head fresh); absent = from scratch, fingerprint unchanged",
    "landmark_fold": "idea 119: landmark set fold K (0-4) never trains, the other folds do, prepared without labels; needs "
                     "extra_landmarks=true and --view LA; off (null) = nothing changes, fingerprint included",
}


def out_dir(version):
    return os.path.join(paths.WORK, "eval", "spine_xray", "corners-v%d" % version)


def parse_sets(items):
    return sets.parse(items, known=SETTINGS)


def parse_folds(text):
    if text == "all":
        return list(range(buu.FOLDS))
    f = sorted({int(t) for t in str(text).split(",") if t.strip() != ""})
    if not f or min(f) < 0 or max(f) >= buu.FOLDS:
        raise SystemExit("--folds: 'all' or numbers from 0 to %d" % (buu.FOLDS - 1))
    return f


# --- the network -----------------------------------------------------------------------------------

def conv_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Stride-2 stem, then levels at 1/2, 1/4, 1/8, 1/16 and 1/32 of the canvas; the decoder climbs back
    to 1/2 and a 1 x 1 convolution gives one map per corner."""

    def __init__(self, widths, n_points):
        super().__init__()
        w = list(widths)
        self.stem = nn.Sequential(nn.Conv2d(1, w[0], 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        self.enc = nn.ModuleList([conv_block(w[0], w[0])] + [conv_block(w[i - 1], w[i]) for i in range(1, len(w))])
        self.dec = nn.ModuleList([conv_block(w[i] + w[i - 1], w[i - 1]) for i in range(len(w) - 1, 0, -1)])
        self.head = nn.Conv2d(w[0], n_points, 1)

    def forward(self, x):
        h = self.enc[0](self.stem(x))
        skips = [h]
        for blk in self.enc[1:]:
            h = blk(F.max_pool2d(h, 2))
            skips.append(h)
        for blk, skip in zip(self.dec, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], dim=1))
        return self.head(h)


def soft_argmax(logits):
    """(B, K, h, w) logits -> coordinates (B, K, 2) in canvas pixels (cell centres at (j + 0.5) * STRIDE)
    and the log-softmax over each map's cells, flat (B, K, h * w). Float32 throughout."""
    b, k, h, w = logits.shape
    logp = F.log_softmax(logits.float().flatten(2), dim=-1)
    p = logp.exp().view(b, k, h, w)
    xs = (torch.arange(w, device=logits.device, dtype=torch.float32) + 0.5) * STRIDE
    ys = (torch.arange(h, device=logits.device, dtype=torch.float32) + 0.5) * STRIDE
    x = (p.sum(2) * xs).sum(-1)
    y = (p.sum(3) * ys).sum(-1)
    return torch.stack([x, y], dim=-1), logp


def gaussian_target(pts, h, w, sigma):
    """(B, K, h * w), each map a normalised Gaussian at the corner (canvas pixels in, cells out)."""
    xs = (torch.arange(w, device=pts.device, dtype=torch.float32) + 0.5) * STRIDE
    ys = (torch.arange(h, device=pts.device, dtype=torch.float32) + 0.5) * STRIDE
    gx = torch.exp(-((xs[None, None, :] - pts[..., 0:1]) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ys[None, None, :] - pts[..., 1:2]) ** 2) / (2 * sigma ** 2))
    t = (gy[..., :, None] * gx[..., None, :]).flatten(2)
    return t / t.sum(-1, keepdim=True).clamp_min(1e-8)


def loss_of(logits, pts, o, mask=None):
    """`mask` (B, K) bool: the points that exist (landmark pictures lack some); None = all, the original expressions."""
    xy, logp = soft_argmax(logits)
    h, w = logits.shape[-2:]
    if mask is not None:
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        l1 = ((xy - pts).abs().sum(-1) * m).sum() / (2 * n) / o["l1_unit"]
        ce = (-(gaussian_target(pts, h, w, o["sigma"]) * logp).sum(-1) * m).sum() / n
        return l1 + o["heat_weight"] * ce, ((xy - pts).norm(dim=-1) * m).sum() / n
    l1 = (xy - pts).abs().mean() / o["l1_unit"]
    ce = -(gaussian_target(pts, h, w, o["sigma"]) * logp).sum(-1).mean()
    return l1 + o["heat_weight"] * ce, (xy - pts).norm(dim=-1).mean()


# --- augmentation on the card -----------------------------------------------------------------------

def augment(x, pts, g, view, o):
    """x (B, 1, H, W) in [0, 1]; pts (B, K, 2) in canvas pixels. Returns both, warped alike."""
    b, _, hh, ww = x.shape
    dev = x.device

    def u(*shape):
        return torch.rand(*shape, device=dev, generator=g) * 2 - 1

    if view == "AP" and o["flip"]:
        f = torch.rand(b, device=dev, generator=g) < 0.5
        x = torch.where(f[:, None, None, None], x.flip(-1), x)
        pf = pts.clone()
        pf[..., 0] = ww - pf[..., 0]
        pf = pf.view(b, -1, 2, 2).flip(2).reshape(b, -1, 2)   # the two ends of every line trade places
        pts = torch.where(f[:, None, None], pf, pts)
    ang = u(b) * (o["rotate_deg"] * math.pi / 180.0)
    sc = 1.0 + u(b) * o["scale_jitter"]
    t = torch.stack([u(b) * o["shift"] * ww, u(b) * o["shift"] * hh], dim=-1)
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    A = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)       # (B, 2, 2)
    c = torch.tensor([ww / 2.0, hh / 2.0], device=dev)
    bvec = c[None] + t - torch.einsum("bij,j->bi", A, c)                                    # p_out = A p_in + bvec
    Fm = torch.zeros(b, 3, 3, device=dev)
    Fm[:, :2, :2], Fm[:, :2, 2], Fm[:, 2, 2] = A, bvec, 1.0
    S = torch.tensor([[2.0 / ww, 0, -1.0], [0, 2.0 / hh, -1.0], [0, 0, 1.0]], device=dev)
    theta = (S[None] @ torch.linalg.inv(Fm) @ torch.linalg.inv(S)[None])[:, :2]           # out -> in, normalised
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    pts = torch.einsum("bij,bkj->bki", A, pts) + bvec[:, None, :]
    j = o["jitter"]
    contrast = 1.0 + u(b)[:, None, None, None] * j
    bright = u(b)[:, None, None, None] * j * 0.5
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * contrast + m + bright).clamp(0.0, 1.0)
    if o["gamma"]:
        e = torch.exp(u(b)[:, None, None, None] * o["gamma"])
        x = x.clamp_min(1e-6) ** e
    return x, pts


# --- baseline ---------------------------------------------------------------------------------------

def _spread(p):
    c = p.mean(axis=-2)
    return c, np.sqrt(((p - c[..., None, :]) ** 2).sum(-1).mean(-1))


def baselines(train_pts, train_size, test_size):
    """The mean-shape baselines, in original pixels, for every test image.
    train_pts (n, K, 2) original pixels, sizes (n, 2) as (width, height).
    `base`: the bar's baseline: the mean shape, the mean spread and the mean centroid, the same for every
    image. `base_size`: spread and centroid as shares of each image's own width and height."""
    c, s = _spread(train_pts)
    shape = ((train_pts - c[:, None, :]) / s[:, None, None]).mean(0)
    base = np.repeat((c.mean(0) + s.mean() * shape)[None], len(test_size), axis=0)
    rel_c = (c / train_size).mean(0)
    rel_s = (s / train_size[:, 1]).mean()
    ts = test_size.astype(np.float64)
    base_size = (rel_c[None, None, :] * ts[:, None, :] + (rel_s * ts[:, 1])[:, None, None] * shape[None])
    return base.astype(np.float32), base_size.astype(np.float32)


# --- training -----------------------------------------------------------------------------------------

def limit_card_memory():
    """Take a card lease sized for this job before touching CUDA, then cap the process at that size
    (RAIDIOLOGY_CARD_GB overrides it), as detector.py does."""
    if DEV.type != "cuda":
        return
    gb = float(os.environ.get("RAIDIOLOGY_CARD_GB", LEASE_GB))
    card.lease(gb, os.path.basename(sys.argv[0]) or "spine corners")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, gb * 2 ** 30 / total), torch.cuda.current_device())


def mean_sd(pix, rows, chunk=64):
    s = s2 = n = 0.0
    for a in range(0, len(rows), chunk):
        x = pix[torch.as_tensor(rows[a:a + chunk], device=pix.device)].double().div_(255.0)
        s += float(x.sum())
        s2 += float((x * x).sum())
        n += x.numel()
    m = s / n
    return m, max(math.sqrt(max(s2 / n - m * m, 0.0)), 1e-3)


def predict(net, pix, rows, mean, sd, batch):
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(rows), batch):
            r = torch.as_tensor(rows[a:a + batch], device=pix.device)
            x = (pix[r].float().div_(255.0).unsqueeze(1) - mean) / sd
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                logits = net(x)
            out.append(soft_argmax(logits)[0].cpu())
    return torch.cat(out).numpy()


PROBE = {}   # the selftest reads what the last train_fold did: steps run, where it resumed, what it loaded


def save_progress(partial, fp, net, opt, sched, gens, **state):
    """The fold's partial checkpoint (core/ckpt.py: atomic, fingerprinted): net, optimiser, scheduler, the
    generators' states, python/numpy/torch RNG, and `state` (epoch, step, order, seconds ...)."""
    ckpt.save_partial(partial, {"fingerprint": fp, "net": net.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "gens": [g.get_state() for g in gens],
                                "rng": ckpt.rng_state(), **state})


def restore_progress(ck, net, opt, sched, gens):
    net.load_state_dict(ck["net"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    for g, s in zip(gens, ck["gens"]):
        g.set_state(s)
    ckpt.set_rng_state(ck["rng"])


def load_chest_init(net, name, say=print, path=None, widths=None):
    """Copy the chest reader's trunk (checkpoint key `state`, `trunk.<i>`) into the U-Net's stem and encoder; the decoder and head
    stay as built. Exits when the widths or a shape do not fit. Returns the number of tensors copied."""
    widths = widths or CHEST_WIDTHS
    have = net.state_dict()
    if [have["stem.0.weight"].shape[0]] + [have["enc.%d.0.weight" % i].shape[0] for i in range(1, 5)] != widths[:1] + widths[1:]:
        sys.exit("init %s: the encoder widths must be %s, this net's are %s"
                 % (name, widths, [int(have["stem.0.weight"].shape[0])] + [int(have["enc.%d.0.weight" % i].shape[0]) for i in range(1, 5)]))
    path = path or INITS[name]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state" not in ck:
        sys.exit("init %s: %s has no `state` key; keys found: %s" % (name, path, ", ".join(sorted(map(str, ck)))))
    src = ck["state"]
    bn = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    pairs = [("stem.0", "trunk.0.0", ("weight",)), ("stem.1", "trunk.0.1", bn),
             ("enc.0.0", "trunk.0.3", ("weight",)), ("enc.0.1", "trunk.0.4", bn)]
    for i in range(1, 5):
        pairs += [("enc.%d.0" % i, "trunk.%d.0" % i, ("weight",)), ("enc.%d.1" % i, "trunk.%d.1" % i, bn),
                  ("enc.%d.3" % i, "trunk.%d.3" % i, ("weight",)), ("enc.%d.4" % i, "trunk.%d.4" % i, bn)]
    new = {}
    for dst, s, names in pairs:
        for n in names:
            k, sk = "%s.%s" % (dst, n), "%s.%s" % (s, n)
            if sk not in src or k not in have:
                sys.exit("init %s: %s lacks %s (or the net lacks %s)" % (name, path, sk, k))
            if tuple(have[k].shape) != tuple(src[sk].shape):
                sys.exit("init %s: %s is %s in %s but %s in the net" % (name, sk, tuple(src[sk].shape), path, tuple(have[k].shape)))
            new[k] = src[sk]
    with torch.no_grad():
        net.load_state_dict(new, strict=False)
    say("init %s: %d tensors copied from %s, decoder and head fresh" % (name, len(new), path))
    return len(new)


def _split(data, fold):
    """The training rows (BUU-LSPINE's other folds, plus any landmark pictures past n_real) and the held-out rows."""
    n_real = data.get("n_real", len(data["ids"]))      # rows past n_real are landmark pictures: training only
    folds = buu.folds_of(data["ids"][:n_real])
    tr = np.where(folds != fold)[0]
    te = np.where(folds == fold)[0]
    n_extra = len(data["pts"]) - n_real
    if n_extra:
        tr = np.concatenate([tr, np.arange(n_real, n_real + n_extra)])
    return n_real, n_extra, tr, te


def _device_data(data):
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    pix = torch.from_numpy(data["pix"]).to(DEV)                                  # the one resident uint8 tensor
    pts = torch.from_numpy(data["pts"]).to(DEV)
    mask = torch.from_numpy(data["mask"]).to(DEV) if "mask" in data else None
    return pix, pts, mask


def _start_net(o, fold, pix, pts, tr):
    """A fresh net and AdamW with the warm-up then cosine schedule; the seed and the order of the draws are what a resume relies on."""
    mean, sd = mean_sd(pix, tr[:256])
    torch.manual_seed(o["seed"] + fold)
    net = UNet(o["widths"], pts.shape[1]).to(DEV)
    if o.get("init"):
        load_chest_init(net, o["init"])
    opt = torch.optim.AdamW(net.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    per_epoch = len(tr) // o["batch"]
    total = o["epochs"] * per_epoch
    warm = o["warmup"] * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    return mean, sd, net, opt, sched, per_epoch


def _resume(o, fold, partial, fp, net, opt, sched, gens):
    """Continues a partial checkpoint whose fingerprint matches, if there is one. Returns where to start: the epoch, the
    step, the epoch's saved order, the epoch times, the loss sums so far and the seconds already spent."""
    PROBE.clear()
    PROBE.update(steps_run=0, resumed_epoch=None)
    ck = ckpt.load_partial(partial, fp) if partial else None
    if ck is None or not 1 <= ck["epoch"] <= o["epochs"] + 1:
        return 1, 0, None, [], (0.0, 0.0), 0.0
    restore_progress(ck, net, opt, sched, gens)
    start_ep, start_s, order_saved, took, sums = ck["epoch"], ck["step"], ck["order"], ck["took"], ck["sums"]
    prior = ck["seconds"]
    PROBE.update(resumed_epoch=start_ep, opt=copy.deepcopy(opt.state_dict()), sched=copy.deepcopy(sched.state_dict()))
    print("fold %d: resumed mid-fold at epoch %d of %d (partial checkpoint)" % (fold, start_ep, o["epochs"]), flush=True)
    return start_ep, start_s, order_saved, took, sums, prior


class _FoldTrainer:
    """train_fold's training loop: the state it used to keep in locals, with each epoch and step as a method. Same calls
    in the same order, so the same draws, weights, checkpoints, prints and PROBE values."""

    def __init__(self, view, data, fold, o, pace, partial, fp):
        self.view, self.o, self.fold, self.pace, self.partial, self.fp = view, o, fold, pace, partial, fp
        self.n_real, self.n_extra, self.tr, self.te = _split(data, fold)
        self.pix, self.pts, self.mask = _device_data(data)
        self.mean, self.sd, self.net, self.opt, self.sched, self.per_epoch = _start_net(o, fold, self.pix, self.pts, self.tr)
        self.g_cpu = torch.Generator().manual_seed(o["seed"] + fold)
        self.g_dev = torch.Generator(device=DEV).manual_seed(o["seed"] + fold)
        print("fold %d: train %d, held out %d, %d steps an epoch" % (fold, len(self.tr), len(self.te), self.per_epoch), flush=True)
        if self.n_extra:
            print("fold %d: %d extra landmark pictures trained on" % (fold, self.n_extra), flush=True)
        self.start_ep, self.start_s, self.order_saved, self.took, self.sums, self.prior = _resume(
            o, fold, partial, fp, self.net, self.opt, self.sched, [self.g_cpu, self.g_dev])
        self.every, self.t_fold = ckpt.Every(), time.time()

    def run(self):
        for ep in range(self.start_ep, self.o["epochs"] + 1):
            self._epoch(ep)
        del self.opt, self.sched
        if DEV.type == "cuda":
            torch.cuda.empty_cache()                 # the training batches' blocks go back before scoring

    def _epoch(self, ep):
        t0 = time.time()
        self.net.train()
        order, s_from, tot_loss, tot_px = self._epoch_start(ep)
        for s in range(s_from, self.per_epoch):
            tot_loss, tot_px = self._step(s, order, tot_loss, tot_px)
            if self.partial and s + 1 < self.per_epoch and self.every.due():
                self._save_mid(ep, s + 1, order, tot_loss, tot_px)
        self.took.append(time.time() - t0)
        self._report(ep, tot_loss, tot_px)
        if self.partial and ep < self.o["epochs"]:
            self._save_end(ep)

    def _epoch_start(self, ep):
        if ep == self.start_ep and self.order_saved is not None:           # the epoch's draw was made before the save
            tot_loss, tot_px = (torch.tensor(v, device=DEV) for v in self.sums)
            return self.order_saved, self.start_s, tot_loss, tot_px
        order = self.tr[torch.randperm(len(self.tr), generator=self.g_cpu).numpy()]
        tot_loss = tot_px = torch.zeros((), device=DEV)
        return order, 0, tot_loss, tot_px

    def _step(self, s, order, tot_loss, tot_px):
        o = self.o
        PROBE["steps_run"] += 1
        began = time.time()
        rows = torch.as_tensor(order[s * o["batch"]:(s + 1) * o["batch"]], device=DEV)
        x = self.pix[rows].float().div_(255.0).unsqueeze(1)
        x, p = augment(x, self.pts[rows], self.g_dev, self.view, o)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
            logits = self.net((x - self.mean) / self.sd)
        loss, px = loss_of(logits, p, o) if self.mask is None else loss_of(logits, p, o, self.mask[rows])
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.sched.step()
        tot_loss, tot_px = tot_loss + loss.detach(), tot_px + px.detach()
        if DEV.type == "cuda" and (s + 1) % 8 == 0:
            torch.cuda.synchronize()
        self.pace.rest(time.time() - began)       # core/polite.py: half the card while somebody is at it
        return tot_loss, tot_px

    def _report(self, ep, tot_loss, tot_px):
        if ep == 1 or ep % 5 == 0 or ep == self.o["epochs"]:
            print("  epoch %3d  loss %.4f  corner miss on training batches %.1f px (canvas)  %.0f s"
                  % (ep, float(tot_loss) / self.per_epoch, float(tot_px) / self.per_epoch, self.took[-1]), flush=True)

    def _save_mid(self, ep, step, order, tot_loss, tot_px):
        save_progress(self.partial, self.fp, self.net, self.opt, self.sched, [self.g_cpu, self.g_dev], epoch=ep, step=step,
                      order=order, took=self.took, sums=(float(tot_loss), float(tot_px)),
                      seconds=self.prior + time.time() - self.t_fold)
        self.every.mark()
        foldck.stop_after("%d.%d" % (ep, step))

    def _save_end(self, ep):
        save_progress(self.partial, self.fp, self.net, self.opt, self.sched, [self.g_cpu, self.g_dev], epoch=ep + 1, step=0,
                      order=None, took=self.took, sums=(0.0, 0.0), seconds=self.prior + time.time() - self.t_fold)
        self.every.mark()
        foldck.stop_after(ep)


def train_fold(view, data, fold, o, pace, partial=None, fp=None):
    """With `partial` (a path) and `fp`, a partial checkpoint is saved after every epoch and, inside a long epoch,
    whenever core.ckpt.Every is due (300 s); a partial with the same fingerprint is continued from."""
    t = _FoldTrainer(view, data, fold, o, pace, partial, fp)
    t.run()
    n_real, n_extra, tr, te = t.n_real, t.n_extra, t.tr, t.te
    pred = predict(t.net, t.pix, te, t.mean, t.sd, max(o["batch"], 8))
    scale, offset = data["scale"][te], data["offset"][te]
    pred_o = ((pred - offset[:, None, :]) / scale[:, None, :]).astype(np.float32)
    truth_o = ((data["pts"][te] - offset[:, None, :]) / scale[:, None, :]).astype(np.float32)
    size = data["size"].astype(np.float64)
    tr_real = tr[tr < n_real]                    # the baselines see BUU's training patients only
    orig_tr = (data["pts"][tr_real] - data["offset"][tr_real][:, None, :]) / data["scale"][tr_real][:, None, :]
    base, base_size = baselines(orig_tr, size[tr_real], size[te])
    info = {"fold": fold, "mean": t.mean, "sd": t.sd, "train_images": int(len(tr)), "predicted_images": int(len(te)),
            "epoch_seconds": [round(v, 2) for v in t.took]}
    if n_extra:
        info["extra_images"] = int(n_extra)
    return t.net, info, te, pred_o, truth_o, base, base_size


def write_fold(d, view, k, o, fingerprint, ids, net, info, pred, truth, base, base_size, partial):
    """The fold's final files (the npz that marks it done, then the model), then its partial checkpoint goes."""
    np.savez(os.path.join(d, "%s-fold%d.npz" % (view, k)), ids=ids, pred=pred, truth=truth, base=base,
             base_size=base_size, fold=np.int64(k), fingerprint=fingerprint)
    torch.save({"state": net.state_dict(), "settings": o, "info": info, "view": view, "points": int(pred.shape[1]),
                "fingerprint": fingerprint}, os.path.join(d, "%s-fold%d.pt" % (view, k)))
    ckpt.clear_partial(partial)


def check_landmark_fold(view, o):
    """idea 119: landmark_fold (None, or 0 to 4) needs extra_landmarks on the side view."""
    k = o.get("landmark_fold")
    if k is None:
        return
    if not (isinstance(k, int) and 0 <= k < 5):
        raise SystemExit("landmark_fold: 0 to 4 (the landmark set's five folds), got %r" % (k,))
    if not o.get("extra_landmarks") or view != "LA":
        raise SystemExit("landmark_fold needs extra_landmarks=true and --view LA")


def fingerprint_settings(o):
    """The settings the fingerprint carries. A setting that is off is left out, so versions trained before it existed
    still match; landmark_fold 0 is a fold (on), only None is off (idea 119)."""
    return {k: v for k, v in o.items() if not (k in ("extra_landmarks", "init") and not v) and not (k == "landmark_fold" and v is None)}


def with_landmarks(data, view, o):
    """A copy of `data` with the landmark pictures appended after the BUU rows (the cached arrays are not touched):
    data["n_real"] BUU rows, then the landmark rows; data["mask"] (N + M, 22) is True where a point exists (a missing
    one is 0 in pts, which the loss never reads). With landmark_fold (idea 119) the rows are landmarks_test.training_set's:
    the set without fold K, prepared without labels; otherwise landmarks.canvas_set (label-fitted, idea 24)."""
    if view != "LA":
        raise SystemExit("extra_landmarks: the landmark set is side views, so --view LA only")
    if o.get("landmark_fold") is None:
        import landmarks
        pix_e, pts_e, _recs = landmarks.canvas_set(o["height"], o["width"])
    else:
        import landmarks_test
        if (o["height"], o["width"]) != landmarks_test.CANVAS:
            raise SystemExit("landmark_fold: the canvas must be %dx%d" % landmarks_test.CANVAS)
        _recs, e = landmarks_test.training_set(o["landmark_fold"])
        pix_e, pts_e = e["canvas"], e["canvas_pts"]
    n = len(data["pts"])
    have = ~np.isnan(pts_e).any(axis=-1)
    out = dict(data)
    out["pix"] = np.concatenate([data["pix"], pix_e])
    out["pts"] = np.concatenate([data["pts"], np.where(have[..., None], pts_e, 0.0).astype(data["pts"].dtype)])
    out["mask"] = np.concatenate([np.ones(data["pts"].shape[:2], bool), have])
    out["n_real"] = n
    print("extra_landmarks: %d landmark pictures on the %dx%d canvas, %d of %d points present"
          % (len(pix_e), o["height"], o["width"], int(have.sum()), have.size), flush=True)
    return out


def train(view, version, folds, settings):
    o = {**SETTINGS, **settings}
    check_landmark_fold(view, o)
    card.require_card(DEV, torch)
    # 2026-10-03: leased before the data is loaded (a job waiting with its data held up to 6.9 GB; the machine
    # ran out of committed memory at 23:21 and a training job crashed)
    limit_card_memory()
    data = buu.load(view, o["height"], o["width"])
    if o["extra_landmarks"]:
        data = with_landmarks(data, view, o)
    d = out_dir(version)
    os.makedirs(d, exist_ok=True)
    # a setting that is off stays out of the fingerprint, so versions trained before it existed still match
    fingerprint = json.dumps({"view": view, "version": version, "settings": fingerprint_settings(o)}, sort_keys=True)
    for k in folds:
        npz = os.path.join(d, "%s-fold%d.npz" % (view, k))
        if os.path.exists(npz) and os.environ.get("RAIDIOLOGY_NO_RESUME") != "1":
            try:
                if str(np.load(npz)["fingerprint"]) == fingerprint:
                    print("fold %d: %s already written for these settings, skipped" % (k, npz), flush=True)
                    continue
            except Exception:  # noqa: BLE001  an unreadable file is simply redone
                pass
        with ledger.run("corners", "train %s fold %d v%d" % (view, k, version)) as led:
            led.settings_from({"view": view, "settings": o})
            partial = os.path.join(d, "%s-fold%d.partial.pt" % (view, k))
            net, info, te, pred, truth, base, base_size = train_fold(view, data, k, o, polite.Pace(), partial,
                                                                     fingerprint + "|fold%d" % k)
            write_fold(d, view, k, o, fingerprint, data["ids"][te], net, info, pred, truth, base, base_size, partial)
            miss = np.linalg.norm(pred - truth, axis=-1).mean()
            miss_b = np.linalg.norm(base - truth, axis=-1).mean()
            led.number("corner_miss_px", miss, "px").number("baseline_miss_px", miss_b, "px")
            led.number("epochs", o["epochs"], "count").number("train_images", info["train_images"], "count")
            if DEV.type == "cuda":
                led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
                torch.cuda.reset_peak_memory_stats()
            print("fold %d: held-out corner miss %.2f px at full resolution, baseline %.2f px -> %s"
                  % (k, miss, miss_b, npz), flush=True)
        del net
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# --- scoring --------------------------------------------------------------------------------------------

def load_oof(view, version, folds):
    d = out_dir(version)
    parts = []
    for k in folds:
        p = os.path.join(d, "%s-fold%d.npz" % (view, k))
        if not os.path.exists(p):
            raise SystemExit("no %s; train it with: python corners.py train --view %s --version %d --folds %d"
                             % (p, view, version, k))
        parts.append(np.load(p))
    return {key: np.concatenate([z[key] for z in parts]) for key in ("ids", "pred", "truth", "base", "base_size")}


def miss_per_patient(a, truth):
    """Mean corner miss in pixels per patient (one image a patient a view)."""
    return np.linalg.norm(a - truth, axis=-1).mean(axis=1)


def boot_means(values, n=BOOT, seed=BOOT_SEED):
    """Means of `n` patient resamples, one row index draw shared by every column of `values` (N, c)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    return values[idx].mean(axis=1)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5], axis=0)
    return lo, hi


def compare(view, a_version, b_version, folds):
    """Paired comparison of two versions' out-of-fold corners: B minus A, mean miss per patient, patient bootstrap."""
    t0 = time.time()
    ra, rb = load_oof(view, a_version, folds), load_oof(view, b_version, folds)
    if not (np.array_equal(ra["ids"], rb["ids"]) and np.array_equal(ra["truth"], rb["truth"])):
        raise SystemExit("versions %d and %d do not hold the same patients and drawn corners; they cannot be paired"
                         % (a_version, b_version))
    vals = np.stack([miss_per_patient(ra["pred"], ra["truth"]), miss_per_patient(rb["pred"], rb["truth"])], axis=1)
    mean = vals.mean(0)
    bm = boot_means(vals)           # one draw of patients for both versions, so the difference is paired
    dlo, dhi = ci(bm[:, 1] - bm[:, 0])
    diff = float(mean[1] - mean[0])
    med = np.median(vals, axis=0)
    print("%s view, folds %s: %d held-out patients, mean corner miss in pixels at full resolution (patient-bootstrap, %d resamples, seed %d)"
          % (view, ",".join(map(str, folds)), len(vals), BOOT, BOOT_SEED))
    print("  A (version %d): %.2f px   B (version %d): %.2f px" % (a_version, mean[0], b_version, mean[1]))
    print("  B minus A %.2f px (95%% CI %.2f to %.2f)" % (diff, dlo, dhi))
    print("  median miss: A %.2f px, B %.2f px" % (med[0], med[1]))
    verdict = "BETTER" if dhi < 0 else "WORSE" if dlo > 0 else "INCONCLUSIVE"
    print("Verdict (B minus A, paired): %s" % verdict)
    ledger.write("experiment", "spine xray corners: %s v%d against v%d folds %s, paired mean corner miss"
                 % (view, b_version, a_version, ",".join(map(str, folds))), time.time() - t0,
                 {"a_miss_px": (float(mean[0]), "px"), "b_miss_px": (float(mean[1]), "px"), "b_minus_a_px": (diff, "px"),
                  "b_minus_a_px_low": (float(dlo), "px"), "b_minus_a_px_high": (float(dhi), "px"),
                  "a_median_px": (float(med[0]), "px"), "b_median_px": (float(med[1]), "px"), "patients": (len(vals), "count")},
                 settings={"view": view, "a": a_version, "b": b_version, "folds": ",".join(map(str, folds)), "verdict": verdict},
                 note="modalities/spine_xray/eval/corners.py compare; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


def evaluate(view, version, folds):
    t0 = time.time()
    r = load_oof(view, version, folds)
    m_model = miss_per_patient(r["pred"], r["truth"])
    m_base = miss_per_patient(r["base"], r["truth"])
    m_size = miss_per_patient(r["base_size"], r["truth"])
    vals = np.stack([m_model, m_base, m_size], axis=1)
    mean = vals.mean(0)
    bm = boot_means(vals)           # one draw of patients for model and baselines, so their ratio is paired
    lo, hi = ci(bm)
    rlo, rhi = np.percentile(bm[:, 0] / bm[:, 1], [2.5, 97.5])
    print("%s view, version %d, folds %s: %d held-out patients, corner miss in pixels at full resolution "
          "(mean over the %d corners, then over patients; patient-bootstrap 95%% CI, %d resamples, seed %d)"
          % (view, version, ",".join(map(str, folds)), len(vals), r["pred"].shape[1], BOOT, BOOT_SEED))
    for name, i in (("model", 0), ("baseline (mean shape, mean spread, mean position)", 1),
                    ("baseline, size-aware (context only)", 2)):
        print("  %-52s %7.2f px  (%.2f to %.2f)" % (name, mean[i], lo[i], hi[i]))
    print("  model / baseline: %.3f (%.3f to %.3f); median miss: model %.2f px, baseline %.2f px"
          % (mean[0] / mean[1], rlo, rhi, float(np.median(m_model)), float(np.median(m_base))))
    kept = bool(mean[0] <= 0.5 * mean[1])
    print("Verdict (corners, half the baseline's miss): %s" % ("KEPT" if kept else "NOT KEPT"))
    mean, lo, hi = ([float(v) for v in a] for a in (mean, lo, hi))   # numpy float32 is not JSON
    numbers = {"corner_miss_px": (mean[0], "px"), "corner_miss_px_low": (lo[0], "px"), "corner_miss_px_high": (hi[0], "px"),
               "baseline_miss_px": (mean[1], "px"), "baseline_miss_px_low": (lo[1], "px"), "baseline_miss_px_high": (hi[1], "px"),
               "baseline_size_miss_px": (mean[2], "px"), "miss_over_baseline": (mean[0] / mean[1], ""),
               "patients": (len(vals), "count"), "kept": (float(kept), "")}
    ledger.write("experiment", "spine xray corners: %s v%d folds %s, mean corner miss vs the mean-shape baseline"
                 % (view, version, ",".join(map(str, folds))), time.time() - t0, numbers,
                 settings={"view": view, "version": version, "folds": ",".join(map(str, folds)),
                           "verdict": "KEPT" if kept else "NOT KEPT", "bar": "model miss <= 0.5 x baseline miss"},
                 note="modalities/spine_xray/eval/corners.py; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


class _NoPace:
    def rest(self, seconds):
        pass


def resume_checks(label, run, root, epochs, probe):
    """The resume proof both corner trainers share. `run(name, stop)` trains one tiny fold whose partial is
    <root>/<name>.partial.pt, finishes it as train() does, and returns its net; `probe` is the trainer's PROBE.
    Epoch-end resume, mid-epoch resume (Every(0): a save every step), a corrupt partial; each must end on the
    weights an uninterrupted run reaches, with the saved optimiser and scheduler, and run only the rest."""
    os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
    os.environ["RAIDIOLOGY_SAVE_EVERY_S"] = "1000000000"
    ref = run("ref", None)
    full = probe["steps_run"]
    per = full // epochs
    assert full == per * epochs and not os.path.exists(os.path.join(root, "ref.partial.pt")), "partial gone after the fold"

    def stopped(name, stop):
        try:
            run(name, stop)
        except foldck.SelftestStop:
            return torch.load(os.path.join(root, name + ".partial.pt"), map_location="cpu", weights_only=False)
        raise AssertionError("the stop did not fire")

    def resumed(name, saved, epoch, steps_left):
        got = run(name, None)
        assert probe["resumed_epoch"] == epoch, "%s: resumed at the saved epoch" % label
        assert foldck.same(probe["opt"], saved["opt"]) and foldck.same(probe["sched"], saved["sched"]), \
            "%s: optimiser and scheduler are the saved ones" % label
        assert probe["steps_run"] == steps_left, "%s: only the rest ran (%d of %d)" % (label, probe["steps_run"], full)
        assert not os.path.exists(os.path.join(root, name + ".partial.pt")), "%s: partial gone after the fold" % label
        assert foldck.same(got.state_dict(), ref.state_dict()), "%s: ends where an uninterrupted run ends" % label

    saved = stopped("a", "3")                                      # after epoch 3's end-of-epoch save
    assert saved["epoch"] == 4 and saved["step"] == 0
    resumed("a", saved, 4, full - 3 * per)
    os.environ["RAIDIOLOGY_SAVE_EVERY_S"] = "0"                    # a save after every step
    saved = stopped("m", "2.1")                                    # inside epoch 2, one step done
    assert saved["epoch"] == 2 and saved["step"] == 1 and saved["order"] is not None
    resumed("m", saved, 2, full - per - 1)
    os.environ["RAIDIOLOGY_SAVE_EVERY_S"] = "1000000000"
    stopped("b", "2")
    with open(os.path.join(root, "b.partial.pt"), "wb") as f:
        f.write(b"not a checkpoint")
    got = run("b", None)
    assert os.path.exists(os.path.join(root, "b.partial.pt.bad")), "%s: corrupt partial renamed .bad" % label
    assert probe["resumed_epoch"] is None and probe["steps_run"] == full, "%s: fresh start" % label
    assert foldck.same(got.state_dict(), ref.state_dict())
    print("%s selftest OK" % label)


def init_checks():
    """The chest-trunk start: parse_sets, the tensors copied, the width refusal (tiny nets, CPU)."""
    import tempfile
    assert parse_sets(["init=\"chest-v103\""])["init"] == "chest-v103" and "init" not in parse_sets([]), "init parses, absent stays out"
    w = [4, 8, 16, 32, 32]

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            ci, st = 1, []
            for i, c in enumerate(w):
                st.append(nn.Sequential(nn.Conv2d(ci, c, 3, stride=2 if i == 0 else 1, padding=1, bias=False), nn.BatchNorm2d(c),
                                        nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                                        nn.MaxPool2d(2)))
                ci = c
            self.trunk = nn.ModuleList(st)

    torch.manual_seed(0)
    fake = Fake()
    for m in fake.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_()
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    try:
        torch.save({"state": fake.state_dict()}, tmp.name)
        net = UNet(w, 4)
        before = net.enc[0][3].weight.clone()
        lines = []
        n = load_chest_init(net, "chest-v103", say=lines.append, path=tmp.name, widths=w)
        assert n == 6 + 6 + 4 * 12, "copied %d tensors" % n
        assert torch.equal(net.stem[0].weight, fake.trunk[0][0].weight) and torch.equal(net.enc[0][0].weight, fake.trunk[0][3].weight)
        assert torch.equal(net.enc[3][4].running_mean, fake.trunk[3][4].running_mean)
        assert torch.equal(net.enc[0][3].weight, before) and not torch.equal(net.enc[0][3].weight, fake.trunk[0][3].weight), \
            "enc[0]'s second conv stays as built"
        assert len(lines) == 1
        try:
            load_chest_init(UNet([4, 8, 16, 32, 64], 4), "chest-v103", say=lambda s: None, path=tmp.name, widths=w)
        except SystemExit:
            pass
        else:
            raise AssertionError("a mismatched width must exit")
        try:
            load_chest_init(UNet([4, 8, 16, 32, 32], 4), "chest-v103", say=lambda s: None, path=tmp.name)
        except SystemExit:
            pass
        else:
            raise AssertionError("widths other than the chest reader's must exit")
    finally:
        os.unlink(tmp.name)
    print("corners init selftest OK")


def landmark_fold_checks():
    """idea 119, no files, no net: landmark_fold K keeps fold K's landmark pictures out of training. A stand-in for
    landmarks_test.training_set returns the set without fold K (landmarks.split_fold on made-up records); with_landmarks
    must ask for K and append exactly those rows. The setting is checked and the fingerprint keeps 0 but drops None."""
    import landmarks
    import landmarks_test
    recs = [{"base": "%d-F-040Y1" % (2700 + i)} for i in range(6)] + [{"base": "Case_%d" % i} for i in range(4)]
    table = {r["base"]: i % 5 for i, r in enumerate(recs)}
    seen = []

    def stand_in(exclude):
        seen.append(exclude)
        train, _held = landmarks.split_fold(recs, table, exclude)
        return train, {"canvas": np.zeros((len(train), 512, 384), np.uint8), "canvas_pts": np.zeros((len(train), 22, 2))}

    n_real = 4
    data = {"pix": np.zeros((n_real, 512, 384), np.uint8), "pts": np.zeros((n_real, 22, 2), np.float32)}
    real = landmarks_test.training_set
    landmarks_test.training_set = stand_in
    try:
        out = with_landmarks(data, "LA", {**SETTINGS, "extra_landmarks": True, "landmark_fold": 2})
    finally:
        landmarks_test.training_set = real
    held = [r for r in recs if table[r["base"]] == 2]
    assert seen == [2], "the trainer asks for the held-out fold"
    assert out["n_real"] == n_real and len(out["pts"]) == n_real + len(recs) - len(held), "fold 2 never trains"
    assert out["mask"].shape == (len(out["pts"]), 22)
    for bad in ({**SETTINGS, "extra_landmarks": False, "landmark_fold": 2}, {**SETTINGS, "extra_landmarks": True, "landmark_fold": 5}):
        try:
            check_landmark_fold("LA", bad)
        except SystemExit:
            pass
        else:
            raise AssertionError("landmark_fold %r with extra_landmarks %r was allowed" % (bad["landmark_fold"], bad["extra_landmarks"]))
    check_landmark_fold("LA", {**SETTINGS, "extra_landmarks": True, "landmark_fold": 0})
    assert "landmark_fold" not in fingerprint_settings(SETTINGS), "off: out of the fingerprint"
    assert fingerprint_settings({**SETTINGS, "landmark_fold": 0})["landmark_fold"] == 0, "fold 0 is on"
    print("corners landmark_fold checks ok")


def selftest():
    """CPU, synthetic tiny images, seconds: python corners.py selftest (with CUDA_VISIBLE_DEVICES empty)."""
    import shutil
    import tempfile
    init_checks()
    landmark_fold_checks()
    g = np.random.default_rng(0)
    n = 10
    data = {"ids": np.arange(n), "pix": g.integers(0, 256, (n, 64, 48), dtype=np.uint8),
            "pts": (g.random((n, 4, 2)) * [40, 56] + 4).astype(np.float32), "scale": np.ones((n, 2)),
            "offset": np.zeros((n, 2)), "size": np.tile([48.0, 64.0], (n, 1))}
    o = {**SETTINGS, "height": 64, "width": 48, "epochs": 5, "batch": 4, "widths": [4, 4, 4, 4, 4], "warmup": 1, "seed": 1}
    root = tempfile.mkdtemp(prefix="corners-selftest-")
    was = buu.folds_of
    buu.folds_of = lambda ids: np.asarray(ids) % 5

    def run(name, stop):
        if stop:
            os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = stop
        else:
            os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
        partial = os.path.join(root, name + ".partial.pt")
        net, info, te, pred, truth, base, base_size = train_fold("LA", data, 0, o, _NoPace(), partial, "fp|fold0")
        os.makedirs(os.path.join(root, name), exist_ok=True)
        write_fold(os.path.join(root, name), "LA", 0, o, "fp", data["ids"][te], net, info, pred, truth, base, base_size,
                   partial)
        return net

    try:
        resume_checks("corners", run, root, o["epochs"], PROBE)
    finally:
        buu.folds_of = was
        os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
        os.environ.pop("RAIDIOLOGY_SAVE_EVERY_S", None)
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="CPU resume proof on synthetic data (seconds)")
    t = sub.add_parser("train", help="train the network on the other folds and write each fold's out-of-fold corners")
    t.add_argument("--view", choices=["AP", "LA"], required=True)
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--folds", default="0", help="'all' or comma-separated fold numbers (default 0)")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--set", action="append", default=[], metavar="key=<json>",
                   help="override: " + "; ".join("%s (%s)" % (k, v) for k, v in HELP.items()))
    e = sub.add_parser("eval", help="mean corner miss against the baseline, with the verdict")
    e.add_argument("--view", choices=["AP", "LA"], required=True)
    e.add_argument("--version", type=int, required=True)
    e.add_argument("--folds", default="0")
    c = sub.add_parser("compare", help="paired comparison of two versions' out-of-fold corners (B minus A)")
    c.add_argument("--view", choices=["AP", "LA"], required=True)
    c.add_argument("--a", type=int, required=True)
    c.add_argument("--b", type=int, required=True)
    c.add_argument("--folds", default="0")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "train":
        s = parse_sets(a.set)
        if a.epochs is not None:
            s["epochs"] = a.epochs
        train(a.view, a.version, parse_folds(a.folds), s)
    elif a.cmd == "compare":
        compare(a.view, a.a, a.b, parse_folds(a.folds))
    else:
        evaluate(a.view, a.version, parse_folds(a.folds))


if __name__ == "__main__":
    main()
