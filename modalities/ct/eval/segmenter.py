"""The CT segmenter: a 3D U-Net trained from scratch on development folds only, the candidate that
has to beat the best baseline (METRICS_SPEC.md) before it goes anywhere near the app.

**From scratch, no pretrained weights of any kind.** ImageNet's terms are non-commercial and so are
most medical model zoos; rAidiology has to stay commercially clean (LICENSING.md), so every weight
here was learned from the TotalSegmentator dataset's CC BY 4.0 scans and nothing else.

**It works at 3 mm and is scored at 1.5 mm.** The network sees the scan at half resolution
(`prepare.py`), where a 128-voxel cube spans 38 cm: enough of a body to count ribs and vertebrae,
which is most of what makes 117 structures hard. Its probabilities are then interpolated back onto
the scan's own grid before a label is chosen, and the three counts the scorer wants are taken there,
against the expert masks as published. Nothing is scored on the coarse grid: a number measured at
3 mm against 3 mm labels is a number about a blurrier question than the one the contract asks.

**One label per voxel**, which is the question as written. The expert masks overlap in a few
voxels per scan, so a perfect answer to this question still scores a hair under 1; `prepare.py
ceiling` says how much.

**Five fold models.** Fold k trains on the other four and predicts fold k, so every development
score comes from a model that never saw that scan. The test split gets all five, their
probabilities averaged. The schedule is a fixed number of steps: nothing is stopped early or picked
by looking at the held-out fold, so the out-of-fold numbers are not flattered by selection. The
handful of held-out scans scored during training are there to draw the learning curve, and nothing
reads them back.

Speed is a requirement here, not a nicety: a fold has to fit inside an hour. So a training sample
is a memory-mapped crop (the bytes of one cube, no decompression), augmentation runs on the
graphics card, and the scoring pass never opens a NIfTI file.

    python modalities/ct/eval/segmenter.py bench                     steps per second and memory
    python modalities/ct/eval/segmenter.py bench --real-crops        the same, on the real 1.5 mm crops of fold 0
    python modalities/ct/eval/segmenter.py train [--fold K] [--version V] [--steps N]
    python modalities/ct/eval/segmenter.py coarse --version V [--folds 0 1 2 3 4]
                                                                     a 3 mm version's out-of-fold label maps,
                                                                     the prior of a cascade (`prior` setting)
    python modalities/ct/eval/segmenter.py predict dev               out-of-fold rows for the scorer
    python modalities/ct/eval/segmenter.py predict test              only inside the final evaluation
    python modalities/ct/eval/segmenter.py predict test --version N --out-dir DIR
                                                                     the pred file goes to DIR (second look)
    python modalities/ct/eval/segmenter.py predict dev --fold K --tag T --with-version N
                                                                     read version V's fold K net together with
                                                                     version N's (two seeds; never the test)
    python modalities/ct/eval/segmenter.py predict test --version V --with-version N --look L --out-dir DIR
                                                                     a pair's test read: both versions' ten nets,
                                                                     equal weight; only inside second look L, which
                                                                     final_eval opened (it passes these, never by hand)
    python modalities/ct/eval/segmenter.py scan <ct.nii.gz> [--out labels.nii.gz]
    python modalities/ct/eval/segmenter.py floors --version V --q 0.01   a trained model's size floor
    python modalities/ct/eval/segmenter.py pieces --version V   a trained model's one-piece lists
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import os
import re
import sys
import threading
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import datasets  # noqa: E402
import pieces  # noqa: E402  arkitect-allow: test-floor-loosened - a sibling module after the sys.path lines, as datasets and prepare are
import prepare  # noqa: E402
from core import card, ckpt, foldck, ledger, paths, polite, sets  # noqa: E402

METHOD = "segmenter"
FOLDS = 5
LOW_HU, HIGH_HU, HU_SCALE = -1000.0, 1500.0, 500.0
CFG = {
    # Version 2's recipe. Against version 1's on fold 0 (2026-10-02, 228 development scans), at two
    # seeds: +0.0127 mean Dice (95% CI +0.0106 to +0.0148) and +0.0055 (+0.0033 to +0.0076), where
    # version 1's recipe at those two seeds landed 0.0001 apart. A sixth level, at 4 cubes across,
    # sees the whole 38 cm cube at once; `rare_sampling` is described at Crops.
    "patch": 128, "widths": [32, 64, 128, 256, 320, 320], "rare_sampling": True,
    # Speed only, the step's arithmetic unchanged: together 8.41 steps a second at 3 mm against 3.07
    # (docs/efficiency/README.md). Compiling needs the triton-windows package: train in venv-next.
    "compile": True, "compile_loss": True, "cudnn_limit": 0, "fused_adam": True,
    # How it reads (READING): drop the stray islands of a structure that is one piece by nature, and
    # drop a structure predicted smaller than 1% of the real ones (pieces.floors).
    "keep_largest": "one_piece", "floor": 0.01,
    # How every activation sits in memory. Measured 2026-10-02 on the RTX 4070 Ti: a 3 mm step runs
    # at 4.44 a second laid out plainly and 3.07 laid out channels-last, where converting layouts
    # around each per-scan normalisation took 147 of the step's 321 ms. Version 1 was trained and
    # scored channels-last and its meta.json has no entry, so it is read that way still.
    "layout": "plain",
    # Coarse first, then fine. At 6 mm a 64-voxel cube shows the same 38 cm of body as a 128-voxel
    # cube at 3 mm for an eighth of the arithmetic, so the network learns where everything is, and
    # which rib is which, many times faster; the 3 mm phase then only has to sharpen the edges.
    # One cube a step at 3 mm: measured on the RTX 4070 Ti, 3.0 steps a second at 6.2 GB, where two
    # cubes spill past the card's 12 GB and run nine times slower.
    "phases": [{"mm": 6, "steps": 6000, "lr": 1e-3, "batch": 2},
               {"mm": 3, "steps": 3500, "lr": 5e-4, "batch": 1}],
    "weight_decay": 3e-5, "warmup": 0.04, "clip": 12.0, "ema": 0.999,
    "structure_share": 0.5,          # crops centred on a structure, each structure equally likely
    "deep": [1.0, 0.5, 0.25],        # loss weight at full, half and quarter resolution
    "warp_share": 0.3, "warp_scale": [0.85, 1.2], "warp_rotate": 0.15,
    "gain": 0.1, "offset": 0.1,
    "loaders": 3, "curve_every": 1500, "curve_scans": 16,
}


def _device():
    """The graphics card when there really is one; RAIDIOLOGY_DEVICE overrides it."""
    want = os.environ.get("RAIDIOLOGY_DEVICE", "").strip()
    if want:
        return torch.device(want)
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda")
    except Exception:  # noqa: BLE001  a broken driver must not stop a processor run
        pass
    return torch.device("cpu")


DEV = _device()


TRIAL_FROM = 100
# Steps queued before the loop waits for the card. Waiting after every step left the card idle while
# the next one was launched: measured 2026-10-02 on version 2's settings, 160 ms a step with the loss
# read back every step, 148 ms with it read every 20, against 146 ms of the card's own work. Pacing
# (core/polite.py) rests the card per block instead, so its share of the card is what it was.
SYNC_EVERY = 4


def model_dir(version):
    """The shipped models live beside the repository. An experiment (version 100 and up) lives
    under work/, so trying a recipe never leaves a model where shipped ones are kept."""
    if version >= TRIAL_FROM:
        return os.path.join(work_dir(version), "model")
    return os.path.join(paths.MODELS, "ct-segmenter-v%d" % version)


def work_dir(version):
    return os.path.join(prepare.OUT, "segmenter-v%d" % version)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_split():
    return json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))



class Stage(nn.Sequential):
    """Two 3 x 3 x 3 convolutions, each normalised per scan and followed by a leaky ReLU.

    Normalised per scan and not per batch: a batch here is one or two cubes, and batch statistics
    over two cubes of different body parts are noise."""

    def __init__(self, c_in, c_out, stride):
        super().__init__(
            nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, inplace=True))


class UNet(nn.Module):
    def __init__(self, widths, n_out, heads=3, float32_heads=False, prior_dim=0):
        """float32_heads: the 1x1 output layers run in float32 under autocast, so a logit keeps more than
        bfloat16's three significant digits (the nodule detector ranks candidates by it).
        prior_dim: the width of the embedding that reads a coarse label cube (`prior`), 0 for none."""
        super().__init__()
        self.float32_heads = float32_heads
        self.prior = nn.Embedding(n_out, prior_dim) if prior_dim else None
        self.down = nn.ModuleList(
            [Stage(1 + prior_dim if i == 0 else widths[i - 1], w, 1 if i == 0 else 2) for i, w in enumerate(widths)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2) for i in range(len(widths) - 1)])
        self.merge = nn.ModuleList(
            [Stage(2 * widths[i], widths[i], 1) for i in range(len(widths) - 1)])
        self.heads = nn.ModuleList([nn.Conv3d(widths[i], n_out, 1) for i in range(heads)])

    def forward(self, x, prior=None):
        """Logits at full resolution, and while training at half and quarter resolution too. `prior`, the
        (B, D, H, W) label ids of the coarse read, is embedded and joins the CT channel before the first conv."""
        if self.prior is not None:
            x = torch.cat([x, self.prior(prior).permute(0, 4, 1, 2, 3)], dim=1)
        skips = []
        for stage in self.down:
            x = stage(x)
            skips.append(x)
        outs = {}
        for i in range(len(self.up) - 1, -1, -1):
            x = self.merge[i](torch.cat([self.up[i](x), skips[i]], dim=1))
            if i < len(self.heads) and (self.training or i == 0):
                if self.float32_heads:
                    with torch.autocast(x.device.type, enabled=False):
                        outs[i] = self.heads[i](x.float())
                else:
                    outs[i] = self.heads[i](x)
        return [outs[i] for i in sorted(outs)]


def fold_blocks(x):
    """Each 2 x 2 x 2 block of a (B, C, D, H, W) tensor as 8 channels at half the size: (B, 8C, D/2, H/2, W/2).
    A block's voxel (dz, dy, dx) becomes channel 8c + 4dz + 2dy + dx, which `unfold_blocks` undoes."""
    b, c, d, h, w = x.shape
    x = x.reshape(b, c, d // 2, 2, h // 2, 2, w // 2, 2)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, 8 * c, d // 2, h // 2, w // 2)


def unfold_blocks(y):
    """The inverse of `fold_blocks`: (B, 8C, d, h, w) back to (B, C, 2d, 2h, 2w)."""
    b, c8, d, h, w = y.shape
    y = y.reshape(b, c8 // 8, 2, 2, 2, d, h, w)
    return y.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, c8 // 8, 2 * d, 2 * h, 2 * w)


class S2DNet(nn.Module):
    """The space-to-depth fine net (idea 127), read at 1.5 mm. The 1.5 mm cube is folded into 8 channels at 3 mm, and the
    3 mm body (UNet's widths) reads it with the coarse prior's embedding joined on. Head A is the body's 3 mm head repeated
    for the 8 sub-voxels and unfolded to 1.5 mm; head B is a thin full-resolution path (the 1.5 mm CT and the body's first
    level, unfolded) whose logits are added to head A's. While training, the body's 3 mm and 6 mm heads come back too, for
    deep supervision, as UNet's do."""

    def __init__(self, widths, n_out, heads=3, prior_dim=16):
        super().__init__()
        self.prior = nn.Embedding(n_out, prior_dim)
        self.down = nn.ModuleList(
            [Stage(8 + prior_dim if i == 0 else widths[i - 1], w, 1 if i == 0 else 2) for i, w in enumerate(widths)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2) for i in range(len(widths) - 1)])
        self.merge = nn.ModuleList(
            [Stage(2 * widths[i], widths[i], 1) for i in range(len(widths) - 1)])
        self.heads = nn.ModuleList([nn.Conv3d(widths[i], n_out, 1) for i in range(heads - 1)])
        self.head_a = nn.Conv3d(widths[0], 8 * n_out, 1)
        self.edge = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.LeakyReLU(0.01, inplace=True))
        self.edge_proj = nn.Conv3d(widths[0], 8 * 16, 1)
        self.edge_mix = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.LeakyReLU(0.01, inplace=True))
        self.edge_out = nn.Conv3d(16, n_out, 1)
        nn.init.zeros_(self.edge_out.weight)
        nn.init.zeros_(self.edge_out.bias)

    def forward(self, x, prior=None):
        """Logits at 1.5 mm, and while training the body's 3 mm and 6 mm logits too. `prior`, the (B, D/2, H/2, W/2) label
        ids of the 3 mm read, is embedded and joins the folded CT before the first conv; without one it is background."""
        if prior is None:
            prior = torch.zeros((x.shape[0],) + tuple(s // 2 for s in x.shape[2:]), dtype=torch.long, device=x.device)
        z = torch.cat([fold_blocks(x), self.prior(prior).permute(0, 4, 1, 2, 3)], dim=1)
        skips = []
        for stage in self.down:
            z = stage(z)
            skips.append(z)
        deep = {}
        for i in range(len(self.up) - 1, -1, -1):
            z = self.merge[i](torch.cat([self.up[i](z), skips[i]], dim=1))
            if i < len(self.heads) and self.training:
                deep[i] = self.heads[i](z)
        edges = self.edge_mix(torch.cat([self.edge(x), unfold_blocks(self.edge_proj(z))], dim=1))
        full = unfold_blocks(self.head_a(z)) + self.edge_out(edges)
        return [full] + [deep[i] for i in sorted(deep)]


def s2d_from_3mm(source, net):
    """A 3 mm net's weights as the start of a space-to-depth net, so at step 0 the S2D net is that net run on the 2 x 2 x 2
    mean of its 1.5 mm cube. The first conv's CT weights, divided by 8, go onto each of the 8 sub-voxel channels; its prior
    channels are the source's own, or the net's own start when the source had no prior. The prior embedding starts at zero,
    head A is the source's 3 mm head copied 8 times, and head B's last conv is zero. The source's 6 mm head onward and its
    prior embedding are not used."""
    own = net.state_dict()
    state = {k: v for k, v in source.items() if k.startswith(("down.", "up.", "merge.", "heads.0.", "heads.1."))}
    stem = source["down.0.0.weight"]
    prior_part = stem[:, 1:] if stem.shape[1] > 1 else own["down.0.0.weight"][:, 8:]
    state["down.0.0.weight"] = torch.cat([stem[:, :1].repeat(1, 8, 1, 1, 1) / 8, prior_part], dim=1)
    state["head_a.weight"] = source["heads.0.weight"].repeat_interleave(8, dim=0)
    state["head_a.bias"] = source["heads.0.bias"].repeat_interleave(8, dim=0)
    state["prior.weight"] = torch.zeros_like(own["prior.weight"])
    state["edge_out.weight"] = torch.zeros_like(own["edge_out.weight"])
    state["edge_out.bias"] = torch.zeros_like(own["edge_out.bias"])
    return {**own, **state}


def layout_of(cfg):
    return torch.contiguous_format if cfg.get("layout") == "plain" else torch.channels_last_3d


def prior_dim_of(cfg):
    return int(cfg["prior"]["dim"]) if cfg.get("prior") else 0


def prior_version_of(cfg):
    return cfg["prior"]["version"] if cfg.get("prior") else None


def net_of(cfg):
    """The network a version is: "unet" (every version so far) or "s2d", the space-to-depth fine net (idea 127)."""
    return cfg.get("net", "unet")


def build(cfg, n_out):
    if net_of(cfg) == "s2d":
        net = S2DNet(cfg["widths"], n_out, heads=len(cfg["deep"]), prior_dim=prior_dim_of(cfg))
    else:
        net = UNet(cfg["widths"], n_out, heads=len(cfg["deep"]), prior_dim=prior_dim_of(cfg))
    net.layout = layout_of(cfg)
    return net.to(DEV, memory_format=net.layout)


def with_prior_input(state, net):
    """A 3 mm net's weights as the start of a version with a prior: the first conv's new input channels are
    zero, so at step 0 the output is the source net's whatever the prior holds; the embedding keeps its own start."""
    state = dict(state)
    w = state["down.0.0.weight"]
    state["down.0.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], net.prior.embedding_dim, *w.shape[2:])], dim=1)
    state["prior.weight"] = net.prior.weight.detach().clone()
    return state


def to_input(hu):
    """Hounsfield units to what the network sees. The same line for training and for a reading."""
    return hu.float().clamp_(LOW_HU, HIGH_HU).div_(HU_SCALE)



def phases_of(cfg):
    """The training phases. Version 1 had one, written as three settings instead of a list."""
    if "phases" in cfg:
        return cfg["phases"]
    return [{"mm": prepare.SPACING * prepare.FACTOR, "steps": cfg["steps"], "lr": cfg["lr"],
             "batch": cfg["batch"]}]


def prior_cube(coarse, src, dst, q):
    """The coarse 3 mm label map cut for a 1.5 mm window: each 3 mm voxel covers 1.5 mm voxels 2i and 2i+1, so
    the window's voxel j takes the coarse voxel j // 2, and anything past the map's edge is background.
    `src` are the window's 1.5 mm slices, `dst` where each lands in the q-wide cube."""
    out = np.zeros((q, q, q), dtype=np.uint8)
    take, put = [], []
    for s, d, n in zip(src, dst, coarse.shape):
        at = np.arange(s.start, s.stop)
        keep = at // prepare.FACTOR < n
        take.append(at[keep] // prepare.FACTOR)
        put.append(d.start + at[keep] - s.start)
    out[np.ix_(*put)] = coarse[np.ix_(*take)]
    return out


def prior_cube_3mm(coarse, src, dst, q):
    """The space-to-depth net's prior: the coarse 3 mm label map cut for a 1.5 mm window whose starts are even, so each
    3 mm voxel is one 2 x 2 x 2 block of the window and the cube is plain slicing, (q/2)^3, with no upsampling.
    `src` and `dst` are as `prior_cube` takes them; anything past the map's edge is background."""
    f = prepare.FACTOR
    out = np.zeros((q // f,) * 3, dtype=np.uint8)
    take, put = [], []
    for s, d, n in zip(src, dst, coarse.shape):
        lo = s.start // f
        hi = min(-(-s.stop // f), n)
        take.append(slice(lo, hi))
        put.append(slice(d.start // f, d.start // f + hi - lo))
    out[tuple(put)] = coarse[tuple(take)]
    return out


def perturb_prior(cube, rng, reach=2):
    """Training only: the coarse cube moved by up to `reach` voxels on each axis (two 1.5 mm voxels, or one 3 mm voxel),
    and each structure in it dropped with chance 0.1, as a coarse read misses and shifts what the fine read will be asked
    to place."""
    q = cube.shape[0]
    shift = rng.integers(-reach, reach + 1, size=3)
    out = np.zeros_like(cube)
    src = tuple(slice(max(-d, 0), q - max(d, 0)) for d in shift)
    dst = tuple(slice(max(d, 0), q - max(-d, 0)) for d in shift)
    out[dst] = cube[src]
    for c in np.unique(out):
        if c and rng.random() < 0.1:
            out[out == c] = 0
    return out


OPEN_FILES = 12      # memmaps a Crops keeps open: four scans' cubes, labels and priors, the ones a crop is drawn from


class Crops(torch.utils.data.Dataset):
    """Random cubes from the 3 mm cache or 1.5 mm cache. Item i is fixed by (seed, i), whichever worker draws it.

    `stride` 2 takes every second voxel of the same stretch of body: the 6 mm phase. The cube that
    comes back is half as wide and shows exactly as much anatomy. `stride` 0 reads from 1.5 mm cache.
    With a `prior`, a 1.5 mm cube comes back with its coarse label cube, perturbed (`perturb_prior`)."""

    def __init__(self, scans, cfg, seed, count, stride=1):
        self.scans, self.cfg, self.seed, self.count, self.stride = list(scans), cfg, seed, count, stride
        # stride 0 is the 1.5 mm phase: the centre is drawn exactly as at 3 mm, the cube cut from the hi15 files
        self.fine = stride == 0
        self.prior = cfg.get("prior")
        self.s2d = net_of(cfg) == "s2d"
        self.opened, self.opened_lock = collections.OrderedDict(), threading.Lock()
        self.points = prepare.points("dev")
        # With `rare_sampling`, a structure-centred crop picks the structure first among all those
        # in the training scans, then a scan that holds it: a kidney cyst in 2% of the scans is then
        # as likely as the liver. Without it the scan comes first and its own structures compete.
        self.holders = {}
        if cfg.get("rare_sampling"):
            for s in self.scans:
                for c in np.unique(self.points[s][1]):
                    self.holders.setdefault(int(c), []).append(s)
        self.kinds = sorted(self.holders)

    def __len__(self):
        return self.count

    def _opened(self, path):
        """np.load(path, mmap_mode="r"), kept for the last OPEN_FILES files read: a crop reads three files, and the same
        scans come up again and again, so the file is opened once and not for every crop."""
        with self.opened_lock:
            if path in self.opened:
                self.opened.move_to_end(path)
                return self.opened[path]
        arr = np.load(path, mmap_mode="r")
        with self.opened_lock:
            self.opened[path] = arr
            while len(self.opened) > OPEN_FILES:
                self.opened.popitem(last=False)
        return arr

    def _structure_first(self, rng, p):
        """`rare_sampling`: the structure is drawn first, from every structure in the training scans,
        then a scan holding it, then a place inside it."""
        if rng.random() < self.cfg["structure_share"]:
            which = self.kinds[int(rng.integers(len(self.kinds)))]
            scan = self.holders[which][int(rng.integers(len(self.holders[which])))]
            shape, pt_is, pt_at = self.points[scan]
            at = rng.choice(pt_at[pt_is == which])
            centre = np.array(np.unravel_index(int(at), shape)) + rng.integers(-p // 4, p // 4 + 1, size=3)
        else:
            scan = self.scans[int(rng.integers(len(self.scans)))]
            shape = self.points[scan][0]
            centre = np.array([rng.integers(s) for s in shape])
        return scan, shape, centre

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        k = max(self.stride, 1)
        p = self.cfg["patch"]              # the stretch of the 3 mm volume a cube covers
        q = p if self.fine else p // k     # the cube's own width in voxels
        # The jitter keeps a structure in the cube's central half. At 1.5 mm the same cube covers half the 3 mm
        # stretch, so the jitter is drawn over half of it (doubled afterwards it is +-patch/4 voxels again).
        jit = p // 2 if self.fine else p
        if self.kinds:
            scan, shape, centre = self._structure_first(rng, jit)
        else:
            scan = self.scans[int(rng.integers(len(self.scans)))]
            shape, pt_is, pt_at = self.points[scan]
            if pt_is.size and rng.random() < self.cfg["structure_share"]:
                # A structure first, then a place inside it: each structure in the scan is as likely
                # as any other, so the adrenal gland is seen as often as the liver.
                which = rng.choice(np.unique(pt_is))
                at = rng.choice(pt_at[pt_is == which])
                centre = np.array(np.unravel_index(int(at), shape))
                centre = centre + rng.integers(-jit // 4, jit // 4 + 1, size=3)
            else:
                centre = np.array([rng.integers(s) for s in shape])
        if self.fine:
            # 3 mm voxel j covers 1.5 mm voxels 2j and 2j+1: the centre goes to one of the two
            centre = centre * prepare.FACTOR
            if not self.s2d:
                # The space-to-depth net reads even starts only, so its centre stays on the first voxel of each pair
                centre = centre + rng.integers(0, prepare.FACTOR, 3)
            shape = tuple(s * prepare.FACTOR for s in shape)
            ct_file, lab_file = prepare.hi15_files(scan)
            p = q                          # the cube is cut at 1.5 mm, patch voxels across
        else:
            ct_file, lab_file = prepare.lo_files(scan)
        ct = self._opened(ct_file)
        lab = self._opened(lab_file)
        out_ct = np.full((q, q, q), prepare.AIR, dtype=np.int16)
        out_lab = np.zeros((q, q, q), dtype=np.uint8)
        src, dst = [], []
        for c, s in zip(centre, shape):
            # Which of the `k` interleaved grids to take: a different one each time, so the coarse
            # phase is not always shown the same voxels of a scan. (Not drawn at 1.5 mm.)
            first = 0 if self.fine else int(rng.integers(k))
            if s >= p:
                a = int(min(max(c - p // 2, 0), s - p))
                src.append(slice(a + first, a + p, k))
                dst.append(slice(0, q))
            else:
                # A scan narrower than the cube sits somewhere inside it, not always in one corner.
                n = len(range(first, s, k))
                a = 2 * int(rng.integers((q - n) // 2 + 1)) if self.s2d else int(rng.integers(q - n + 1))
                src.append(slice(first, s, k))
                dst.append(slice(a, a + n))
        out_ct[tuple(dst)] = ct[tuple(src)]
        out_lab[tuple(dst)] = lab[tuple(src)]
        if self.prior is None:
            return torch.from_numpy(out_ct), torch.from_numpy(out_lab)
        coarse = self._opened(coarse_file(self.prior["version"], scan))
        if self.s2d:
            out_pr = perturb_prior(prior_cube_3mm(coarse, src, dst, q), rng, reach=1)
        else:
            out_pr = perturb_prior(prior_cube(coarse, src, dst, q), rng)
        return torch.from_numpy(out_ct), torch.from_numpy(out_lab), torch.from_numpy(out_pr)


def stream(crops, batch, threads):
    """Batches of crops, read a few ahead by threads in this process.

    Threads and not worker processes. A crop is a memory-mapped slice, so the work is a copy that
    needs no second interpreter, and on Windows every worker process that imports torch commits a
    few gigabytes of its own: six of them ran this machine out of memory before the first step."""
    from collections import deque  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    def one(i):
        got = [crops[j] for j in range(i * batch, (i + 1) * batch)]
        out = tuple(torch.stack(part) for part in zip(*got))
        # Pinned, so the step's non_blocking copy to the card really runs without the loop waiting for it
        return tuple(t.pin_memory() for t in out) if DEV.type == "cuda" else out

    ahead = deque()
    with ThreadPoolExecutor(threads) as pool:
        for i in range(len(crops) // batch):
            ahead.append(pool.submit(one, i))
            if len(ahead) > 2 * threads:
                yield ahead.popleft().result()
        while ahead:
            yield ahead.popleft().result()


def warp(x, y, cfg, gen, prior=None):
    """A small random zoom and turn for some of the batch, on the graphics card. The prior, when there is one,
    takes the labels' turn exactly (nearest), so the two still overlap afterwards.

    No mirroring, ever: left kidney and right kidney are different answers, and a flipped scan
    teaches the network they are interchangeable."""
    n = x.shape[0]
    chosen = torch.rand(n, generator=gen, device=x.device) < cfg["warp_share"]
    if not bool(chosen.any()):
        return x, y, prior
    lo, hi = cfg["warp_scale"]
    zoom = torch.where(chosen, torch.empty(n, device=x.device).uniform_(lo, hi, generator=gen),
                       torch.ones(n, device=x.device))
    ang = torch.empty(n, 3, device=x.device).uniform_(-cfg["warp_rotate"], cfg["warp_rotate"], generator=gen)
    ang = ang * chosen[:, None]
    ca, sa = torch.cos(ang), torch.sin(ang)
    one, zero = torch.ones(n, device=x.device), torch.zeros(n, device=x.device)
    rx = torch.stack([one, zero, zero, zero, ca[:, 0], -sa[:, 0], zero, sa[:, 0], ca[:, 0]], 1).view(n, 3, 3)
    ry = torch.stack([ca[:, 1], zero, sa[:, 1], zero, one, zero, -sa[:, 1], zero, ca[:, 1]], 1).view(n, 3, 3)
    rz = torch.stack([ca[:, 2], -sa[:, 2], zero, sa[:, 2], ca[:, 2], zero, zero, zero, one], 1).view(n, 3, 3)
    theta = torch.cat([(rz @ ry @ rx) * zoom[:, None, None], torch.zeros(n, 3, 1, device=x.device)], dim=2)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                      align_corners=False)[:, 0].long()
    if prior is not None:
        # A space-to-depth net's prior is on the 3 mm grid: the same turn, sampled there (normalised coordinates agree)
        pgrid = grid if prior.shape[1:] == x.shape[2:] else F.affine_grid(theta, [n, 1, *prior.shape[1:]], align_corners=False)
        prior = F.grid_sample(prior[:, None].float(), pgrid, mode="nearest", padding_mode="zeros",
                              align_corners=False)[:, 0].long()
    return x, y, prior


def loss_of(outs, y, cfg):
    """Cross-entropy plus Dice at full resolution, cross-entropy alone on the coarser outputs.

    The Dice term is what keeps small structures alive: cross-entropy over two million voxels is
    almost entirely about the liver and the background, and a network can drive it down a long way
    while never outlining an adrenal gland. It is computed without building a one-hot target, which
    at 118 classes would be a gigabyte per cube for no reason."""
    # Cast inside the softmax rather than copying the logits to fp32 first: the same loss to the
    # last bit (measured), and a 3 mm step 9% faster.
    logp = F.log_softmax(outs[0], dim=1, dtype=torch.float32)
    total = cfg["deep"][0] * (F.nll_loss(logp, y) + dice_miss(logp, y))
    for k in range(1, len(outs)):
        step = 2 ** k
        yk = y[:, ::step, ::step, ::step]
        lk = F.log_softmax(outs[k], dim=1, dtype=torch.float32)
        term = F.nll_loss(lk, yk)
        if k < cfg.get("deep_dice", 1):         # Dice at the coarser heads too, as nnU-Net does
            term = term + dice_miss(lk, yk)
        total = total + cfg["deep"][k] * term
    return total / sum(cfg["deep"][:len(outs)])


def dice_miss(logp, y):
    """One minus the mean soft Dice over the structures present in `y` (none present: 0), from
    log-probabilities, without a one-hot target and without a branch, so it compiles to one graph."""
    p = logp.exp()
    n = p.shape[1]
    flat = y.reshape(-1)
    truth = torch.zeros(n, device=p.device).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    inside = torch.zeros(n, device=p.device).scatter_add_(0, flat, p.gather(1, y[:, None]).reshape(-1))
    said = p.sum(dim=(0, 2, 3, 4))
    here = (truth > 0) & (torch.arange(n, device=p.device) > 0)
    dice = (2 * inside + 1.0) / (said + truth + 1.0)
    return torch.where(here, 1 - dice, torch.zeros_like(dice)).sum() / here.sum().clamp_min(1)



def is_fine_version(meta):
    """A version is fine if its last training phase is at 1.5 mm: it is read at 1.5 mm, by `read_fine`."""
    phases = meta["cfg"].get("phases")          # version 1 had no list: one 3 mm phase
    return bool(phases) and phases[-1]["mm"] == prepare.SPACING


def coarse_dir(version):
    return os.path.join(prepare.OUT, "coarse-v%d" % version)


def coarse_file(version, scan):
    return os.path.join(coarse_dir(version), "%s.npy" % scan)


def coarse_of(version, scan):
    if not os.path.exists(coarse_file(version, scan)):
        raise SystemExit("%s has no coarse map from version %d: python segmenter.py coarse --version %d"
                         % (scan, version, version))
    return np.load(coarse_file(version, scan), mmap_mode="r")


def require_coarse(version, scans):
    missing = [s for s in scans if not os.path.exists(coarse_file(version, s))]
    if missing:
        raise SystemExit("%d scan(s) have no coarse map from version %d (%s first): python segmenter.py coarse "
                         "--version %d" % (len(missing), version, missing[0], version))


def meta_if_any(version):
    """A version's meta.json, or None when it has none (never trained, or not here)."""
    path = os.path.join(model_dir(version), "meta.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def prior_version_for(meta, prior_from=None, named=None):
    """The version whose coarse maps a model reads as its prior: the one its meta names, or, with the development
    setting `prior_from` (idea 130), the version named in its place, which must be a 3 mm version with a meta.json.
    `named` is that version's meta.json (None if it has none)."""
    own = prior_version_of(meta["cfg"])
    if prior_from is None:
        return own
    if own is None:
        raise SystemExit("prior_from needs a model that reads a coarse prior: version %d has none." % meta["version"])
    if named is None:
        raise SystemExit("prior_from names version %d, which has no meta.json: it must be a trained 3 mm version."
                         % prior_from)
    if is_fine_version(named):
        raise SystemExit("prior_from names version %d, a 1.5 mm version: a prior is read from its 3 mm coarse maps."
                         % prior_from)
    return prior_from


def reading_prior(meta, named=None):
    """The version whose coarse maps a model reads when it is read: the standing `prior_from` in its meta.json (set by
    `segmenter.py prior`, so a 1.5 mm model keeps reading a 3 mm version's maps), else the version its own cfg names.
    `named` is the standing version's meta.json; it is looked up when not given."""
    pf = meta.get("reading", {}).get("prior_from")
    if pf is None:
        return prior_version_of(meta["cfg"])
    return prior_version_for(meta, pf, named if named is not None else meta_if_any(pf))


def write_coarse(version, folds, dev_folds, skipped):
    """Each development scan's 3 mm label map, read by the fold net that never trained on it, so a prior is
    never a label its own network was taught. Resumable: a scan is read again only when its file is missing or
    its fold net has changed (the sha256 in manifest.json). Each file is written atomically. Returns the count."""
    d = coarse_dir(version)
    os.makedirs(d, exist_ok=True)
    manifest_path = os.path.join(d, "manifest.json")
    manifest = (json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path)
                else {"version": version, "grid": "3 mm", "folds": {}, "scans": {}})
    written, pace = 0, polite.Pace()
    for fold in folds:
        net_sha = sha256(os.path.join(model_dir(version), "fold%d.pt" % fold))
        todo = [s for s, f in sorted(dev_folds.items()) if f == fold and s not in skipped
                and not (manifest["scans"].get(s, {}).get("net_sha256") == net_sha
                         and os.path.exists(coarse_file(version, s)))]
        if manifest["folds"].get(str(fold)) != net_sha:
            manifest["folds"][str(fold)] = net_sha
            with open(manifest_path + ".part", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=1, sort_keys=True)
            os.replace(manifest_path + ".part", manifest_path)
        if todo:
            models, meta = load_models(version, [fold])
            raw = dict(meta["cfg"], **meta.get("reading", {}))
            overlap, trim = raw.get("overlap", READING["overlap"]), raw.get("trim", READING["trim"])
            for scan in todo:
                began = time.time()
                ct_lo = np.ascontiguousarray(np.load(prepare.lo_files(scan)[0]))
                acc, _ = probabilities(models, ct_lo, meta["cfg"]["patch"], overlap, trim)
                lab = coarse_labels(acc).to(torch.uint8).cpu().numpy()
                del acc
                tmp = os.path.join(d, "%s.part.npy" % scan)
                np.save(tmp, lab)
                os.replace(tmp, coarse_file(version, scan))
                manifest["scans"][scan] = {"fold": fold, "shape": list(lab.shape), "net_sha256": net_sha}
                with open(manifest_path + ".part", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=1, sort_keys=True)
                os.replace(manifest_path + ".part", manifest_path)
                written += 1
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                pace.rest(time.time() - began)
            del models
    return written


def write_test_coarse(version, look, scans, skipped):
    """The test scans' 3 mm label maps, each read by all five of version's nets together (every net's probabilities
    averaged with equal weight). A test read, so it is written only inside the second look `look` whose version takes
    `version` as its prior. Resumable as write_coarse is: a map is read again only when its nets have changed."""
    entry = prepare.open_look(look)
    prior = reading_prior(json.load(open(os.path.join(model_dir(entry["version"]), "meta.json"), encoding="utf-8")))
    if prior != version:
        raise SystemExit("Look %d's version %d takes prior %s, not version %d." % (look, entry["version"], prior, version))
    d = coarse_dir(version)
    os.makedirs(d, exist_ok=True)
    manifest_path = os.path.join(d, "manifest.json")
    manifest = (json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path)
                else {"version": version, "grid": "3 mm", "folds": {}, "scans": {}})
    nets = [sha256(os.path.join(model_dir(version), "fold%d.pt" % k)) for k in range(FOLDS)]
    todo = [s for s in scans if s not in skipped
            and not (manifest["scans"].get(s, {}).get("nets_sha256") == nets and os.path.exists(coarse_file(version, s)))]
    if not todo:
        return 0
    models, meta = load_models(version, range(FOLDS))
    raw = dict(meta["cfg"], **meta.get("reading", {}))
    overlap, trim = raw.get("overlap", READING["overlap"]), raw.get("trim", READING["trim"])
    written, pace = 0, polite.Pace()
    for scan in todo:
        began = time.time()
        ct_lo = np.ascontiguousarray(np.load(prepare.lo_files(scan)[0]))
        acc, _ = probabilities(models, ct_lo, meta["cfg"]["patch"], overlap, trim)
        lab = coarse_labels(acc).to(torch.uint8).cpu().numpy()
        del acc
        tmp = os.path.join(d, "%s.part.npy" % scan)
        np.save(tmp, lab)
        os.replace(tmp, coarse_file(version, scan))
        manifest["scans"][scan] = {"split": "test", "shape": list(lab.shape), "nets_sha256": nets}
        with open(manifest_path + ".part", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        os.replace(manifest_path + ".part", manifest_path)
        written += 1
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        pace.rest(time.time() - began)
    del models
    return written


def coarse_maps(a):
    """The coarse label maps a prior version reads. From a 3 mm version: each development scan read by its held-out
    fold. On the test split (with --look, a second look): each test scan read by all five nets together."""
    meta = json.load(open(os.path.join(model_dir(a.version), "meta.json"), encoding="utf-8"))
    if is_fine_version(meta):
        raise SystemExit("version %d is read at 1.5 mm: a coarse map comes from a 3 mm version" % a.version)
    if (a.split == "test") != (a.look is not None):
        raise SystemExit("--split test goes with --look: the test split is read only inside a second look.")
    steady_inference()
    limit_card_memory()
    split = load_split()
    if a.split == "test":
        datasets.guard_test("test")
        with ledger.run("segmenter", "coarse v%d test look %d" % (a.version, a.look)) as led:
            led.settings_from({"split": "test", "look": a.look})
            written = write_test_coarse(a.version, a.look, sorted(split["test"]), prepare.unprepared("test"))
            led.number("coarse_maps_written", written, "count")
        print("%d coarse map(s) of the test split written under %s" % (written, coarse_dir(a.version)))
        return
    skipped = prepare.unprepared("dev")
    with ledger.run("segmenter", "coarse v%d" % a.version) as led:
        led.settings_from({"folds": a.folds})
        written = write_coarse(a.version, a.folds, split["dev_folds"], skipped)
        led.number("coarse_maps_written", written, "count")
    print("%d coarse map(s) written under %s" % (written, coarse_dir(a.version)))


def owned_intervals(starts, patch, size):
    """The half-open interval [lo, hi) of voxels each window owns along one axis: the boundary between two
    windows sits at the midpoint of their centres, a voxel exactly on it going to the earlier window, the
    first interval starts at 0 and the last ends at `size`. Every voxel is owned once, in order."""
    centres = [s + patch // 2 for s in starts]
    cuts = [0] + [(a + b) // 2 + 1 for a, b in zip(centres, centres[1:])] + [size]
    return [(min(cuts[i], size), min(cuts[i + 1], size)) for i in range(len(starts))]


@torch.no_grad()
def prior_window(coarse, at, patch):
    """The coarse label cube of one 1.5 mm window, on the card, as the embedding's ids: (1, patch, patch, patch)."""
    cube = prior_cube(coarse, [slice(s, s + patch) for s in at], [slice(0, patch)] * 3, patch)
    return torch.as_tensor(cube, device=DEV).long()[None]


@torch.no_grad()
def prior_window_3mm(coarse, at, patch):
    """The space-to-depth net's prior for one 1.5 mm window starting at the even voxels `at`: the 3 mm cube, (1, patch/2, ...)."""
    cube = prior_cube_3mm(coarse, [slice(s, s + patch) for s in at], [slice(0, patch)] * 3, patch)
    return torch.as_tensor(cube, device=DEV).long()[None]


# 2026-10-08: no autograd graphs: the nets' parameters require grad, and the test read sums five nets per window
@torch.no_grad()
def read_fine(models, ct15, patch, reading, names, coarse=None):
    """The labels of a 1.5 mm volume, window by window: `patch` voxels at the same starts as `probabilities`,
    the models' probabilities summed in the window, the label taken on the device, and written only where the
    window owns the voxel (`owned_intervals`). Nothing of 118 classes is held for more than one window.
    `coarse` is the scan's 3 mm label map, the prior of a version that has one."""
    ct = torch.as_tensor(np.ascontiguousarray(ct15), device=DEV)
    shape = tuple(ct.shape)
    x = to_input(ct)
    s2d = isinstance(models[0], S2DNet)
    pad = [max(patch - s, 0) for s in shape]
    if s2d:
        pad = [p + (s + p) % 2 for p, s in zip(pad, shape)]       # every axis even, so the last window starts even too
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=LOW_HU / HU_SCALE)
    full = tuple(x.shape)
    starts = [_starts(full[i], patch, reading["overlap"]) for i in range(3)]
    if s2d:
        # A space-to-depth window starts on an even voxel, so its 2 x 2 x 2 blocks are the prior's 3 mm voxels
        starts = [sorted({s - s % 2 for s in st}) for st in starts]
        shape = full
    own = [owned_intervals(starts[i], patch, shape[i]) for i in range(3)]
    out = torch.zeros(shape, device=DEV, dtype=torch.uint8)
    for (a, (lo0, hi0)) in zip(starts[0], own[0]):
        for (b, (lo1, hi1)) in zip(starts[1], own[1]):
            for (c, (lo2, hi2)) in zip(starts[2], own[2]):
                if hi0 <= lo0 or hi1 <= lo1 or hi2 <= lo2:
                    continue
                window = x[a:a + patch, b:b + patch, c:c + patch][None, None]
                window = window.contiguous(memory_format=models[0].layout)
                pw = None if coarse is None else (prior_window_3mm if s2d else prior_window)(coarse, (a, b, c), patch)
                prob = None
                for m in models:
                    p = window_probabilities(m, window, pw)
                    prob = p if prob is None else prob.add_(p)
                    p = None
                pred = prob.argmax(dim=0).to(torch.uint8)
                out[lo0:hi0, lo1:hi1, lo2:hi2] = pred[lo0 - a:hi0 - a, lo1 - b:hi1 - b, lo2 - c:hi2 - c]
    return out[:ct.shape[0], :ct.shape[1], :ct.shape[2]] if s2d else out


def _starts(size, patch, overlap=0.5):
    """Where the windows along one axis start: `overlap` of each window shared with the next."""
    if size <= patch:
        return [0]
    n = int(math.ceil((size - patch) / (patch * (1.0 - overlap)))) + 1
    return sorted(set(int(round(v)) for v in np.linspace(0, size - patch, n)))


# What a reading takes on the card besides its running total, measured 2026-10-02 on the largest
# development scan (a 2.3 GB total): 3.0 GB with one model, a 5.3 GB peak, and 4.2 GB with all five,
# a 6.5 GB peak. The room kept is a little more than that, because a card's free memory comes in
# pieces, and it still puts every one of the 1,140 prepared scans on the card.
READING_ROOM_BYTES = 4 << 30
SEVERAL_MODELS_BYTES = 1 << 30


HELD_GB = None     # the share core/card.py lease gave this process, once limit_card_memory has run


def card_ceiling_bytes():
    """The share this process holds, else RAIDIOLOGY_CARD_GB (7.5 by default); 95% of a smaller card."""
    want = (HELD_GB if HELD_GB is not None else float(os.environ.get("RAIDIOLOGY_CARD_GB", "7.5"))) * 2 ** 30
    return min(0.95 * torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory, want)


def limit_card_memory(gb=None, name=None, short_min=None):
    """A hard ceiling on what this process may take from the graphics card.

    Without one, the allocator's cache grows until the card is full, and on Windows a full card
    does not fail: it spills into ordinary memory, the run slows to a crawl and the desktop stutters
    for as long as it lasts. That happened to the first training run here (11.7 of 12 GB, no step
    for four minutes). With a ceiling the same situation is an error at once, which is the better
    failure. RAIDIOLOGY_CARD_GB moves it. The process also waits here for its share of the card
    (core/card.py lease), so several processes' ceilings cannot add up past it.

    2026-10-04: the ceiling is the share actually held (a caller may pass a smaller gb), not always
    RAIDIOLOGY_CARD_GB; a process holds one lease, so asking twice no longer makes it wait on itself."""
    global HELD_GB
    if DEV.type != "cuda":
        return
    from core import card as card_lease
    # Lease first, sized from nvidia-smi: reading the card through torch.cuda starts CUDA, and a job
    # waiting for its share must hold none (2026-10-03: waiting jobs held 2.1-6.9 GB; the machine ran out of
    # committed memory at 23:21 and a training job crashed)
    smi = card_lease.card_total_gb()
    want = float(os.environ.get("RAIDIOLOGY_CARD_GB", "7.5")) if gb is None else gb
    HELD_GB = card_lease.lease(want if smi is None else min(0.95 * smi, want), name or "ct segmenter",
                               short_min=short_min)
    card = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(card).total_memory
    torch.cuda.set_per_process_memory_fraction(card_ceiling_bytes() / total, card)


def coarse_labels(acc, slab=32):
    """The likeliest class at every voxel of the working grid, a slab at a time on the card."""
    out = torch.zeros(tuple(acc.shape[1:]), device=DEV, dtype=torch.long)
    for a in range(0, acc.shape[3], slab):
        out[:, :, a:a + slab] = acc[:, :, :, a:a + slab].to(DEV).argmax(dim=0)
    return out


_BELL = {}


def _bell(edges):
    """The weight of each voxel of a window: a window is trusted most in its middle. `edges` is the
    window's three edges; along an edge trimmed to the scan there is one window, so it cancels."""
    if edges not in _BELL:
        g = []
        for n in edges:
            d = torch.arange(n, device=DEV, dtype=torch.float32) - (n - 1) / 2.0
            g.append(torch.exp(-(d ** 2) / (2 * (n / 8.0) ** 2)))
        _BELL[edges] = (g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]).clamp_min(0.01)
    return _BELL[edges]


def window_multiple(m):
    """What a window's edge must be a multiple of: the network halves it once per level below its top."""
    if isinstance(m, GraphReader):
        raise SystemExit("trim needs a PyTorch model: an OpenVINO graph is compiled for one window shape")
    return 2 ** (len(m.down) - 1) * (2 if isinstance(m, S2DNet) else 1)


def _read_half():
    """The half precision a reading runs in: fp16 on the card, as every score was taken; bf16 on a
    processor that does it natively, else full precision. Version 1's fold 0 on 60 of its held-out
    scans, on a Ryzen 9 7950X (2026-10-02): 0.7301 mean Dice in bf16, in fp32 and on the card alike,
    with bf16 1.8x faster than fp32 (21.1 s a scan against 37.6)."""
    if DEV.type == "cuda":
        return torch.float16
    if DEV.type == "cpu" and torch.ops.mkldnn._is_mkldnn_bf16_supported():
        return torch.bfloat16
    return None


READ_HALF = _read_half()


class Softmaxed(nn.Module):
    """The full-resolution head's class probabilities in fp32: what `probabilities` takes from each
    model for each window, as one module, so a graph compiler can fuse the softmax into the network."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return F.softmax(self.net(x)[0].float(), dim=1)


class GraphReader:
    """One fold's network and softmax as a compiled OpenVINO graph, for a reading on a processor.

    Called with a window, (1, 1, p, p, p) fp32, it answers the window's class probabilities,
    (classes, p, p, p) fp32, as `window_probabilities` does for a PyTorch model. The answer is copied
    out of the request, whose buffer the next call overwrites."""

    def __init__(self, net, patch, precision, saved=None, weights_sha256=None):
        import openvino as ov  # noqa: PLC0415  only a processor reading with this engine needs it

        core = ov.Core()
        if saved:
            # A graph saved by tools/ct_openvino_int8.py, refused if its weights are not this fold's.
            graph = core.read_model(saved)
            stamped = graph.get_rt_info(["raidiology", "weights_sha256"]).astype(str)
            if stamped != weights_sha256:
                raise SystemExit("%s was made from other weights than this fold's; run "
                                 "tools/ct_openvino_int8.py export (and quantize) again" % saved)
        else:
            graph = ov.convert_model(Softmaxed(net).eval(), example_input=torch.zeros(1, 1, patch, patch, patch))
        self.compiled = core.compile_model(graph, "CPU", {"INFERENCE_PRECISION_HINT": precision,
                                                          "PERFORMANCE_HINT": "LATENCY"})
        self.request = self.compiled.create_infer_request()
        self.n_out = net.heads[0].out_channels
        self.layout = torch.contiguous_format

    def __call__(self, window):
        self.request.infer({0: window.numpy()})
        return torch.from_numpy(self.request.get_output_tensor(0).data[0].copy())


def n_classes(m):
    return m.n_out if isinstance(m, GraphReader) else m.heads[0].out_channels


def window_probabilities(m, window, prior=None):
    """One model's class probabilities over one window, (classes, p, p, p) fp32."""
    if isinstance(m, GraphReader):
        return m(window)
    with torch.autocast(DEV.type, dtype=READ_HALF, enabled=READ_HALF is not None):
        logits = m(window, prior)[0]
    return F.softmax(logits, dim=1, dtype=torch.float32)[0]


@torch.no_grad()
def probabilities(models, ct_lo, patch, overlap=0.5, trim=False):
    """Class probabilities over a whole 3 mm volume, from overlapping windows, averaged over the
    models given. Returns (classes, x, y, z), half precision, on the device.

    With `trim`, an axis shorter than the cube is padded only up to the next edge the network can
    take (`window_multiple`), not to the whole cube: a scan 76 voxels deep is read 80 or 96 deep,
    not 128. Less air goes through the network, and the per-scan normalisation sees less of it,
    which is why it is a reading to score and not a free speed-up."""
    ct = torch.as_tensor(np.ascontiguousarray(ct_lo), device=DEV)
    shape = tuple(ct.shape)
    edges = (patch,) * 3
    if trim:
        step = window_multiple(models[0])
        edges = tuple(min(patch, -(-s // step) * step) for s in shape)
    pad = [max(e - s, 0) for e, s in zip(edges, shape)]
    x = to_input(ct)
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=LOW_HU / HU_SCALE)
    full = tuple(x.shape)
    n_out = n_classes(models[0])
    # 118 classes over a long scan is gigabytes. The running total stays on the card while it fits
    # under the ceiling beside the reading's own room. Past that it lives in ordinary memory, where
    # it cannot push the card over its edge and is slower: on the largest scan 10.7 s against 4.0 s
    # with one model and 22.5 s against 16.7 s with five, labels identical either way.
    room = READING_ROOM_BYTES + (SEVERAL_MODELS_BYTES if len(models) > 1 else 0)
    big = DEV.type == "cuda" and n_out * int(np.prod(full)) * 2 + room > card_ceiling_bytes()
    where = torch.device("cpu") if big else DEV
    acc = torch.zeros((n_out,) + full, device=where, dtype=torch.float16)
    weight = torch.zeros(full, device=where, dtype=torch.float32)
    bell = _bell(edges)
    e0, e1, e2 = edges
    for a in _starts(full[0], e0, overlap):
        for b in _starts(full[1], e1, overlap):
            for c in _starts(full[2], e2, overlap):
                window = x[a:a + e0, b:b + e1, c:c + e2][None, None]
                window = window.contiguous(memory_format=models[0].layout)
                # Summed and weighted in place: the same additions in the same order as a fresh
                # tensor for each, and a gigabyte less on the card at the peak.
                prob = None
                for m in models:
                    p = window_probabilities(m, window)
                    prob = p if prob is None else prob.add_(p)
                    p = None
                acc[:, a:a + e0, b:b + e1, c:c + e2] += prob.mul_(bell).to(where, torch.float16)
                weight[a:a + e0, b:b + e1, c:c + e2] += (bell * len(models)).to(where)
    acc = acc[:, :shape[0], :shape[1], :shape[2]]
    weight = weight[:shape[0], :shape[1], :shape[2]]
    return acc, weight


@torch.no_grad()
def labels_on_grid(acc, weight, shape, slab=8, tile=None):
    """The label of every voxel of the scan's own grid: the probabilities interpolated up, then the
    likeliest class taken. Interpolating the probabilities and not the labels is what keeps a thin
    structure's edge where it is instead of on a 3 mm staircase.

    Done in slabs along the last axis, because 118 classes at 1.5 mm is several gigabytes at once.
    Each slab carries one extra coarse slice either side, so the answer is identical to doing the
    whole volume in one go. `tile` (LABEL_TILE) cuts each slab into tiles across too, with the same
    one-voxel margin, and interpolates in each only the classes that can win there (`_contenders`)."""
    if tile is None and DEV.type == "cuda":
        return _labels_or_tiles(acc, weight, shape, slab)
    if tile is None:
        tile = LABEL_TILE if DEV.type == "cpu" else 0
    f = prepare.FACTOR
    out = torch.zeros(shape, device=DEV, dtype=torch.uint8)
    nx, ny, nz = acc.shape[1:]
    across = [(0, nx, 0, ny)] if not tile else [(x, min(x + tile, nx), y, min(y + tile, ny))
                                                 for x in range(0, nx, tile) for y in range(0, ny, tile)]
    for a in range(0, nz, slab):
        b = min(a + slab, nz)
        lo, hi = max(a - 1, 0), min(b + 1, nz)
        for x0, x1, y0, y1 in across:
            xlo, xhi, ylo, yhi = max(x0 - 1, 0), min(x1 + 1, nx), max(y0 - 1, 0), min(y1 + 1, ny)
            part = (acc[:, xlo:xhi, ylo:yhi, lo:hi].to(DEV).float() / weight[None, xlo:xhi, ylo:yhi, lo:hi].to(DEV))
            keep = _contenders(part) if tile else None
            up = F.interpolate((part if keep is None else part[keep])[None], scale_factor=f, mode="trilinear",
                               align_corners=False)[0]
            best = up.argmax(dim=0)
            best = (best if keep is None else keep[best]).to(torch.uint8)
            fx0, fx1 = x0 * f, min(x1 * f, shape[0])
            fy0, fy1 = y0 * f, min(y1 * f, shape[1])
            z0, z1 = a * f, min(b * f, shape[2])
            if fx1 > fx0 and fy1 > fy0 and z1 > z0:
                ox, oy, oz = (x0 - xlo) * f, (y0 - ylo) * f, (a - lo) * f
                out[fx0:fx1, fy0:fy1, z0:z1] = best[ox:ox + fx1 - fx0, oy:oy + fy1 - fy0, oz:oz + z1 - z0]
    return out


def _labels_or_tiles(acc, weight, shape, slab=8, tile=None):
    """labels_on_grid whole-slab, as a card always ran it; only a scan that runs the card out of memory
    is labelled again, in tiles (`tile`, else LABEL_TILE, else 32)."""
    try:
        return labels_on_grid(acc, weight, shape, slab, tile=0)
    except torch.OutOfMemoryError:  # 2026-10-08: a large test scan; the tiles give the same labels (_contenders)
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return labels_on_grid(acc, weight, shape, slab, tile=tile or (LABEL_TILE if LABEL_TILE > 0 else 32))


# Tiles of 32 coarse voxels (9.6 cm) across on a processor; 0 is whole slabs. Measured 2026-10-02: 1.46x
# a scan in 3 of 3 pairs, every scan's labels byte-identical (docs/efficiency/README.md).
LABEL_TILE = int(os.environ.get("RAIDIOLOGY_LABEL_TILE", "32"))


def _contenders(part):
    """The classes that can be the likeliest anywhere a tile's interpolation reaches, in order.

    A voxel's probabilities sum to one, so its likeliest class has at least 1/n of it, and a trilinear
    value is never above the largest of the values it is drawn from (its weights are positive and sum
    to one, and rounding cannot break either). A class under 1/n everywhere in the tile, margin
    included, therefore never wins in it, and leaving it out changes no label. 0.99/n allows for the
    half-precision running total summing to a hair under one."""
    n = part.shape[0]
    return (part.amax(dim=(1, 2, 3)) >= 0.99 / n).nonzero().flatten()


def counts_against_truth(pred, scan, n_names):
    """The scorer's three counts for every structure of one scan: truth, predicted, both.

    Truth is the count in each expert mask as published, overlaps included; `both` counts a voxel
    for a structure when the prediction names it and that structure's mask contains it."""
    z = np.load(prepare.hi_file(scan))
    n = n_names + 1
    truth_map = torch.as_tensor(z["labels"], device=DEV)
    if tuple(truth_map.shape) != tuple(pred.shape):
        raise SystemExit("%s: prediction %s is not on the truth's grid %s"
                         % (scan, tuple(pred.shape), tuple(truth_map.shape)))
    table = torch.bincount(truth_map.reshape(-1).long() * n + pred.reshape(-1).long(),
                           minlength=n * n).view(n, n)
    said = table.sum(dim=0)[1:].cpu().numpy()
    both = table.diagonal()[1:].cpu().numpy().copy()
    seen = table.sum(dim=1)[1:].cpu().numpy()
    if z["extra_at"].size:
        at = torch.as_tensor(z["extra_at"], device=DEV)
        want = torch.as_tensor(z["extra_is"], device=DEV)
        hit = pred.reshape(-1)[at] == want
        both += torch.bincount(want[hit].long(), minlength=n)[1:].cpu().numpy()
        seen = seen + np.bincount(z["extra_is"], minlength=n)[1:]
    truth = z["truth"].astype(np.int64)
    if not np.array_equal(seen, truth):
        # The truth counted here, on the card, against the truth counted when the cache was built.
        raise SystemExit("%s: the truth read back for scoring is not the truth that was cached" % scan)
    return truth, said.astype(np.int64), both.astype(np.int64)


def load_models(version, folds):
    d = model_dir(version)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    if prior_version_of(meta["cfg"]) is not None and CPU_ENGINE == "openvino":
        raise SystemExit("version %d takes a coarse prior, which the OpenVINO reader does not feed yet: "
                         "read it with RAIDIOLOGY_CPU_ENGINE=torch" % version)
    models = []
    for k in folds:
        net = build(meta["cfg"], len(meta["structures"]) + 1)
        state = torch.load(os.path.join(d, "fold%d.pt" % k), map_location=DEV, weights_only=True)
        net.load_state_dict(state)
        if DEV.type == "cpu" and (CPU_LAYOUT == "plain" or CPU_ENGINE == "openvino"):
            net.layout = torch.contiguous_format
            net = net.to(memory_format=net.layout)
        net.eval()
        if DEV.type == "cpu" and CPU_ENGINE == "openvino":
            saved = (os.path.join(paths.WORK, "ov", "v%d" % version,
                                  "fold%d%s.xml" % (k, "" if OV_GRAPH == "fp32" else "." + OV_GRAPH))
                     if OV_GRAPH else None)
            net = GraphReader(net, meta["cfg"]["patch"], OV_PRECISION, saved,
                              sha256(os.path.join(d, "fold%d.pt" % k)) if saved else None)
        models.append(net)
    return models, meta


# How a processor lays out a model's activations when it reads. Measured 2026-10-02: version 1 read
# channels-last spent 46% of its network and softmax time converting layouts around each per-scan
# normalisation, as training on the card had. Plain is 1.44x a scan (3 of 3 pairs) and gives the
# scored answer: 0.7301 mean Dice against the card's 0.7301 on 60 held-out development scans of fold
# 0, difference +0.0000 (95% CI -0.0001 to +0.0001) (docs/efficiency/README.md).
CPU_LAYOUT = os.environ.get("RAIDIOLOGY_CPU_LAYOUT", "plain")
# What runs the network when a processor reads: PyTorch, or each fold as an OpenVINO graph with the
# softmax inside (GraphReader) in RAIDIOLOGY_OV_PRECISION, f32 or bf16. Measured 2026-10-02 on
# version 1 (docs/efficiency/README.md): OpenVINO in bf16 and fp32 no faster than PyTorch (0.98x,
# 0.91x a window); int8 1.30x a window but -0.0011 mean Dice (95% CI -0.0018 to -0.0003) on 60
# scans. So PyTorch reads; the arm stays for a later model or release to be measured with one switch.
CPU_ENGINE = os.environ.get("RAIDIOLOGY_CPU_ENGINE", "torch")
OV_PRECISION = os.environ.get("RAIDIOLOGY_OV_PRECISION", "f32")
# Empty: convert the fold's weights in this process. fp32 or int8: the graph tools/ct_openvino_int8.py
# saved under work/ov/v<V>/.
OV_GRAPH = os.environ.get("RAIDIOLOGY_OV_GRAPH", "")


def steady_inference():
    """Pin the convolution algorithm, so the same scan gets the same labels twice on one machine."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



@torch.no_grad()
def curve_point(net, scans, cfg, n_names, stride=1, prior_version=None):
    """Mean Dice over structures on a few held-out scans, at the grid the phase trains on (every
    `stride`-th voxel of the 3 mm cache; stride 0 is the 1.5 mm phase, read whole by `read_fine` and scored
    against the hi15 labels). For the learning curve only: a quick look, coarser than the scored number,
    and nothing is chosen by it. Version 1 read its 6 mm phase at 3 mm, where the anatomy looked twice its
    size to that network, so its 6 mm points (0.25 to 0.31) say little."""
    was = net.training
    net.eval()
    inside = [[] for _ in range(n_names)]
    k = max(stride, 1)
    for scan in scans:
        if stride == 0:
            ct_file, lab_file = prepare.hi15_files(scan)
            coarse = None if prior_version is None else coarse_of(prior_version, scan)
            pred = read_fine([net], np.load(ct_file), cfg["patch"], {"overlap": READING["overlap"]}, None,
                             coarse=coarse).long()
            truth = torch.as_tensor(np.ascontiguousarray(np.load(lab_file)), device=DEV).long()
        else:
            ct_file, lab_file = prepare.lo_files(scan)
            acc, _ = probabilities([net], np.ascontiguousarray(np.load(ct_file)[::k, ::k, ::k]), cfg["patch"] // k)
            pred = coarse_labels(acc)
            del acc
            truth = torch.as_tensor(np.ascontiguousarray(np.load(lab_file)[::k, ::k, ::k]), device=DEV).long()
        n = n_names + 1
        table = torch.bincount(truth.reshape(-1) * n + pred.reshape(-1), minlength=n * n).view(n, n)
        t, p, b = table.sum(1)[1:], table.sum(0)[1:], table.diagonal()[1:]
        for i in torch.nonzero(t).reshape(-1).tolist():
            inside[i].append(float(2 * b[i]) / float(t[i] + p[i]))
    net.train(was)
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    per = [np.mean(v) for v in inside if v]
    return float(np.mean(per)) if per else float("nan")


def amp_of(cfg):
    """The half precision a step runs in. bf16 keeps fp32's range, so it needs no loss scaler."""
    return torch.bfloat16 if cfg.get("amp") == "bf16" else torch.float16


def make_scaler(cfg):
    return torch.amp.GradScaler(DEV.type, enabled=DEV.type == "cuda" and amp_of(cfg) == torch.float16)


def make_opt(net, lr, cfg):
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg["weight_decay"],
                             fused=DEV.type == "cuda" and bool(cfg.get("fused_adam")))


def forward_of(net, cfg):
    """The network as a step calls it: compiled when the setting asks for it, which on Windows
    needs the triton-windows package. A mode name ("max-autotune-no-cudagraphs") passes through."""
    if not cfg.get("compile"):
        return net
    return torch.compile(net, mode=cfg["compile"] if isinstance(cfg["compile"], str) else None)


def loss_fn_of(cfg):
    """The loss as a step calls it: compiled into one fused graph when the setting asks."""
    return torch.compile(loss_of) if cfg.get("compile_loss") else loss_of


def train_step(net, ema, opt, sched, scaler, ct, lab, cfg, gen, step, forward=None, loss_fn=None, prior=None):
    """One step of training exactly as a fold runs it, from a batch of crops to the averaged
    weights; `bench` times this same function, so a measured setting is the setting trained.
    Returns the loss, still on the card, and how many times the step had to be retried."""
    x = to_input(ct.to(DEV, non_blocking=True))[:, None]
    y = lab.to(DEV, non_blocking=True).long()
    if prior is not None:
        prior = prior.to(DEV, non_blocking=True).long()
    x, y, prior = warp(x, y, cfg, gen, prior)
    extra = () if prior is None else (prior,)
    gain = 1 + (torch.rand(x.shape[0], 1, 1, 1, 1, device=DEV, generator=gen) * 2 - 1) * cfg["gain"]
    shift = (torch.rand(x.shape[0], 1, 1, 1, 1, device=DEV, generator=gen) * 2 - 1) * cfg["offset"]
    x = (x * gain + shift).contiguous(memory_format=net.layout)
    retried = 0
    for attempt in (1, 2):
        try:
            with torch.autocast(DEV.type, dtype=amp_of(cfg), enabled=DEV.type == "cuda"):
                outs = (net if forward is None else forward)(x, *extra)
            loss = (loss_of if loss_fn is None else loss_fn)(outs, y, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            break
        except torch.OutOfMemoryError:
            if attempt == 2 or DEV.type != "cuda":
                raise
        # Fold 2 of version 1 died here at step 7,600: a 944 MiB block refused under the ceiling
        # while the card had 3.5 GiB free, the ceiling's room left in pieces. The same cubes go
        # through once more with the cache handed back, outside the except block so its traceback
        # no longer holds the step's tensors; a second refusal is a real shortage and stops the fold.
        outs = loss = None
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        retried += 1
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
    scaler.step(opt)
    scaler.update()
    sched.step()
    with torch.no_grad():
        keep = min(cfg["ema"], (1 + step) / (10 + step))
        if cfg.get("ema_foreach"):
            torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1 - keep)
        else:
            for a, b in zip(ema.parameters(), net.parameters()):
                a.lerp_(b, 1 - keep)
    return loss.detach(), retried


PROBE = {}   # the selftest reads what the last train_fold did: steps run, where it resumed, what it loaded


# The one-hour rule (owner, 2026-10-11: "Why do we not have scans that only run one hour? And then check the results
# and see if they suck before wasting time"). `train --kill-ref V --kill-step S` judges the fold at its first learning-
# curve point at step S or later, about an hour in, against version V's curve of the same fold on the same held-out
# scans, and stops the fold there when it is behind by more than --kill-margin. A fold that starts from another
# version's weights (init_from) has to reach V's finished curve by then; one trained from scratch, V's curve at the same
# step. Measured before the rule: v130, which continued v121, was 0.011 to 0.025 ahead of v121's finished curve inside
# the first hour of every fold, and then trained 2 to 6 hours more. The fold's last partial (300 s at most) is kept.
def kill_reference(kill, fold, probe, cfg):
    """The point fold `fold` has to reach by step kill["step"] ({step, mm, dice, end}), from version kill["ref"]'s
    meta.json; None with no kill line. Refused when the two curves are not like for like."""
    if not kill:
        return None
    meta = json.load(open(os.path.join(model_dir(kill["ref"]), "meta.json"), encoding="utf-8"))
    got = {f["fold"]: f for f in meta.get("folds", [])}.get(fold)
    if got is None or not got.get("curve"):
        raise SystemExit("--kill-ref %d has no learning curve for fold %d" % (kill["ref"], fold))
    if got.get("curve_scans") != probe:
        raise SystemExit("--kill-ref %d drew fold %d's curve on other held-out scans: not like for like"
                         % (kill["ref"], fold))
    points = [p for p in got["curve"] if p["dice"] == p["dice"]]     # a lost look (NaN) is passed over
    end = bool(cfg.get("init_from"))
    at = points if end else [p for p in points if p["step"] <= kill["step"]]
    if not at:
        raise SystemExit("--kill-ref %d has no curve point at step %d or before for fold %d"
                         % (kill["ref"], kill["step"], fold))
    done, mm = 0, None
    for ph in phases_of(cfg):
        done += ph["steps"]
        if kill["step"] <= done:
            mm = ph["mm"]
            break
    if mm != at[-1]["mm"]:
        raise SystemExit("--kill-step %d falls in a %s mm phase, but version %d's point is at %g mm: not like for like"
                         % (kill["step"], mm, kill["ref"], at[-1]["mm"]))
    return dict(at[-1], end=end)


def train_fold(fold, cfg, version, force=False, kill=None):
    """Train fold `fold`. The model is fold{k}.pt, the EMA state dict as ever, and the fingerprint of its training
    settings sits in fold{k}.pt.fp: a fold whose file is there with a matching fingerprint (or with none, an old
    one) is not trained again; a different fingerprint, --force or RAIDIOLOGY_NO_RESUME=1 trains it. A partial
    checkpoint, fold{k}.partial.pt, is saved whenever core.ckpt.Every is due (300 s) and continued from, in the
    phase and step it stopped at; it is removed once the model is saved."""
    if net_of(cfg) == "s2d" and any(ph["mm"] != prepare.SPACING for ph in phases_of(cfg)):
        raise SystemExit("version %d is a space-to-depth net, which is read at 1.5 mm: every phase must be at %g mm"
                         % (version, prepare.SPACING))
    if prior_version_of(cfg) is not None and any(ph["mm"] != prepare.SPACING for ph in phases_of(cfg)):
        raise SystemExit("version %d takes a coarse prior, which is read at 1.5 mm: every phase must be at %g mm"
                         % (version, prepare.SPACING))
    d = model_dir(version)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "fold%d.pt" % fold)
    partial = os.path.join(d, "fold%d.partial.pt" % fold)
    split = load_split()
    names = datasets.structures()
    skipped = prepare.unprepared("dev")
    mine = sorted(s for s, f in split["dev_folds"].items() if f != fold and s not in skipped)
    held = sorted(s for s, f in split["dev_folds"].items() if f == fold and s not in skipped)
    # `seed` shifts every random draw of a fold: the same recipe trained twice with two seeds is how
    # far apart two runs land by chance, the bar a recipe change has to clear (2026-10-02).
    seed = datasets.SEED + fold + 1000 * int(cfg.get("seed", 0))
    fp = foldck.digest({"cfg": cfg, "fold": fold, "seed": seed, "train": mine, "structures": names})
    PROBE.clear()
    PROBE.update(steps_run=0, resumed_step=None, skipped=False, killed=None)
    if os.path.exists(out) and not force and os.environ.get("RAIDIOLOGY_NO_RESUME") != "1" \
            and (foldck.fp_ok(out, fp) or not os.path.exists(out + ".fp")):
        print("fold %d is already trained (%s); --force trains it again" % (fold, out), flush=True)
        PROBE["skipped"] = True
        return None
    if prior_version_of(cfg) is not None:
        require_coarse(prior_version_of(cfg), mine + held)
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    probe = [held[int(i)] for i in np.random.default_rng(seed).permutation(len(held))[:cfg["curve_scans"]]]
    line = kill_reference(kill, fold, probe, cfg)

    net = build(cfg, len(names) + 1).train()
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    scaler = make_scaler(cfg)
    forward = forward_of(net, cfg)
    loss_fn = loss_fn_of(cfg)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = int(cfg.get("cudnn_limit", 10))
    torch.set_num_threads(2)
    pace = polite.Pace()
    phases = phases_of(cfg)
    total = sum(ph["steps"] for ph in phases)
    grid = prepare.SPACING * prepare.FACTOR

    ck = ckpt.load_partial(partial, fp)
    # Load init_from weights if specified (not on resume from partial checkpoint)
    if ck is None or ck["step"] == 0:
        init_from = cfg.get("init_from")
        if init_from:
            init_dir = model_dir(init_from)
            init_meta = json.load(open(os.path.join(init_dir, "meta.json"), encoding="utf-8"))
            if init_meta["cfg"]["widths"] != cfg["widths"] or len(init_meta["structures"]) != len(names):
                raise SystemExit("init_from version %d has widths %s or structures %d, but this version has %s and %d"
                                 % (init_from, init_meta["cfg"]["widths"], len(init_meta["structures"]),
                                    cfg["widths"], len(names)))
            init_weights = torch.load(os.path.join(init_dir, "fold%d.pt" % fold), map_location=DEV, weights_only=True)
            if net_of(cfg) == "s2d" and net_of(init_meta["cfg"]) != "s2d":
                init_weights = s2d_from_3mm(init_weights, net)
            elif prior_version_of(cfg) is not None and prior_version_of(init_meta["cfg"]) is None:
                init_weights = with_prior_input(init_weights, net)
            net.load_state_dict(init_weights)
            ema.load_state_dict(init_weights)
            print("fold %d: loaded init_from version %d" % (fold, init_from), flush=True)
    skip = 0
    if ck is not None and 0 < ck["step"] < total:
        skip = ck["step"]
        net.load_state_dict(ck["net"])
        ema.load_state_dict(ck["ema"])
        scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["gen"])
        ckpt.set_rng_state(ck["rng"])
        print("fold %d: resumed mid-fold at step %d of %d (partial checkpoint)" % (fold, skip, total), flush=True)
        PROBE["resumed_step"] = skip
        ck = {k: ck[k] for k in ("opt", "sched", "curve", "retried", "curve_seconds", "seconds")}   # the weights are loaded

    def batches():
        """Every phase's cubes in turn, each phase with its own optimiser and its own warm-up. After a resume the
        phases already done are passed over, and the phase it stopped in goes on from its saved optimiser and
        schedule with cubes drawn from a new seed (the stream's own position cannot be saved)."""
        done = 0
        for i, ph in enumerate(phases):
            into = max(skip - done, 0)                        # steps of this phase the partial already holds
            done += ph["steps"]
            if into >= ph["steps"]:
                continue
            opt = make_opt(net, ph["lr"], cfg)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ph["lr"], total_steps=ph["steps"],
                                                        pct_start=cfg["warmup"], anneal_strategy="cos")
            if into:
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                PROBE.update(opt=copy.deepcopy(opt.state_dict()), sched=copy.deepcopy(sched.state_dict()))
            # 1.5 mm phase (mm=1.5) uses stride=0 to read from hi15 cache
            stride = 0 if ph["mm"] == prepare.SPACING else int(round(ph["mm"] / grid))
            crops = Crops(mine, cfg, seed + 7919 * i + 104729 * into, (ph["steps"] - into) * ph["batch"],
                          stride=stride)
            for n, batch in enumerate(stream(crops, ph["batch"], cfg["loaders"]), into + 1):
                yield batch, opt, sched, ph, n == ph["steps"]

    curve, t0, seen, running, retried, curve_seconds = [], time.time(), 0, 0.0, 0, 0.0
    if skip:
        curve, retried, curve_seconds = ck["curve"], ck["retried"], ck["curve_seconds"]
        t0 -= ck["seconds"]                                    # seconds trained add up across a resume
    every = ckpt.Every()
    waited, asked = 0.0, time.time()
    block, block_waited = time.time(), 0.0
    with ledger.run("segmenter", "train fold %d v%d" % (fold, version)) as led:
        led.settings_from({"cfg": cfg, "train_scans": len(mine)})
        for step, (batch, opt, sched, ph, phase_ends) in enumerate(batches(), skip + 1):
            PROBE["steps_run"] += 1
            began = time.time()
            waited += began - asked
            block_waited += began - asked
            ct, lab, *prior = batch
            loss, again = train_step(net, ema, opt, sched, scaler, ct, lab, cfg, gen, step, forward, loss_fn,
                                     *prior)
            retried += again
            seen += 1
            running = running + loss                 # on the card, read when the loop next waits
            # Every step is paced before a look at held-out scans too, whatever curve_every is.
            if step % SYNC_EVERY == 0 or phase_ends or step % cfg["curve_every"] == 0:
                running = float(running)
                pace.rest(time.time() - block - block_waited)
                block, block_waited = time.time(), 0.0
            if step % 100 == 0:
                print("  fold %d step %d of %d (%g mm), loss %.4f, %.2f steps/s, using %.0f%% of the "
                      "card, %.0f%% of the time waiting for crops"
                      % (fold, step, total, ph["mm"], running / seen, step / (time.time() - t0),
                         100 * pace.used(), 100 * waited / (time.time() - t0)), flush=True)
                led.point("loss", step, running / seen)
                seen, running = 0, 0.0
            if step % cfg["curve_every"] == 0 or phase_ends:
                # The step's own tensors go back to the allocator first: the look at held-out scans
                # has to fit under the same ceiling as the training it interrupts.
                del loss
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                try:
                    stride = 0 if ph["mm"] == prepare.SPACING else int(round(ph["mm"] / grid))
                    began_curve = time.time()
                    dice = curve_point(ema, probe, cfg, len(names), stride, prior_version_of(cfg))
                    curve_seconds += time.time() - began_curve
                except torch.OutOfMemoryError:
                    # The curve is a look, not a result. Running out of room for it under the
                    # ceiling costs one point of a drawing, never the fold. Fold 0's last point
                    # was lost this way before the line above existed.
                    torch.cuda.empty_cache()
                    dice = float("nan")
                curve.append({"step": step, "seconds": round(time.time() - t0, 1), "mm": ph["mm"],
                              "dice": dice})
                led.point("dice_%gmm" % ph["mm"], step, dice)
                print("  fold %d step %d (%g mm phase): %.4f mean Dice at %g mm on %d held-out scan(s)"
                      % (fold, step, ph["mm"], dice, ph["mm"], len(probe)), flush=True)
                if line is not None and step >= kill["step"] and dice == dice:
                    bar = line["dice"] - kill["margin"]
                    said = ("fold %d at step %d (%.0f s): %.4f on the curve's scans, against %.4f (version %d's %s %.4f, "
                            "less %g)" % (fold, step, time.time() - t0, dice, bar, kill["ref"],
                                          "finished curve" if line["end"] else "curve at step %d" % line["step"],
                                          line["dice"], kill["margin"]))
                    led.number("kill_line_margin", dice - bar)
                    if dice < bar:
                        PROBE["killed"] = step
                        raise SystemExit("KILLED by its kill line: " + said)
                    print("  kill line passed: " + said, flush=True)
                    line = None
                block, block_waited = time.time(), 0.0   # the look is not training: it is not paced
            if step < total and every.due():
                ckpt.save_partial(partial, {
                    "fingerprint": fp, "step": step, "net": net.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "gen": gen.get_state(), "rng": ckpt.rng_state(), "curve": curve, "retried": retried,
                    "curve_seconds": curve_seconds, "seconds": time.time() - t0})
                every.mark()
                block, block_waited = time.time(), 0.0   # writing is not training either
                foldck.stop_after(step)
            asked = time.time()
        seconds = time.time() - t0
        foldck.finish(out, lambda p: torch.save(ema.state_dict(), p), fp)
        ckpt.clear_partial(partial)
        led.number("steps", total, "count").number("train_scans", len(mine), "count")
        led.number("steps_per_second", total / max(seconds, 1e-9))
        led.number("share_of_the_card_used", pace.used())
        led.number("curve_seconds", curve_seconds, "s")
        # What the run took from the rest of the machine: core/polite.py paces only the card.
        me = psutil.Process()
        used = me.cpu_times()
        led.number("cpu_seconds", used.user + used.system, "s")
        peak = getattr(me.memory_info(), "peak_wset", None)  # Windows only; Linux has ru_maxrss (kilobytes)
        if peak is None:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        led.number("host_memory_peak_gb", peak / 2 ** 30, "GB")
        if curve:
            led.number("curve_dice_%gmm" % curve[-1]["mm"], curve[-1]["dice"])
        led.number("steps_retried", retried, "count")
        if DEV.type == "cuda":
            led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
    return {"fold": fold, "file": os.path.basename(out), "sha256": sha256(out),
            "train_scans": len(mine), "held_scans": len(held), "seconds": round(seconds, 1),
            "curve": curve, "curve_scans": probe}


def write_meta(version, cfg, trained, one_piece=None, floors=None):
    d = model_dir(version)
    f = os.path.join(d, "meta.json")
    meta = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    folds = {str(t["fold"]): t for t in meta.get("folds", [])}
    for t in trained:
        folds[str(t["fold"])] = t
    commit = ledger._commit()
    meta = {"name": "ct-segmenter", "version": version, "method": METHOD, "cfg": cfg,
            "structures": datasets.structures(), "spacing_mm": prepare.SPACING * prepare.FACTOR,
            "input": {"low_hu": LOW_HU, "high_hu": HIGH_HU, "divide_by": HU_SCALE},
            "pretrained": "none: trained from scratch", "dataset": datasets.SET["cite"],
            "torch": torch.__version__, "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "commit": commit, "folds": [folds[k] for k in sorted(folds, key=int)]}
    if one_piece:
        meta["one_piece"] = one_piece
    if floors:
        meta["floors"] = floors
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return f


def with_settings(cfg, items):
    """CFG with `key=<json>` settings applied: how an experiment changes the recipe without an edit."""
    cfg = dict(cfg)
    cfg.update(sets.parse(items))
    return cfg


def train(a):
    card.require_card(DEV, torch)
    cfg = with_settings(CFG, a.set)
    if a.patch:
        cfg["patch"] = a.patch
    if a.steps:
        # One plain phase at the working grid: what version 1 was, and what a smoke run wants.
        cfg["phases"] = [{"mm": prepare.SPACING * prepare.FACTOR, "steps": a.steps, "lr": 1e-3,
                          "batch": a.batch or 1}]
    if os.path.exists(os.path.join(model_dir(a.version), "meta.json")) and not a.force:
        was = json.load(open(os.path.join(model_dir(a.version), "meta.json"), encoding="utf-8"))["cfg"]
        if was != cfg:
            raise SystemExit("Version %d was started with other settings (%s). Five folds trained "
                             "differently are not one model: use --version for a new one, or --force."
                             % (a.version, json.dumps(was, sort_keys=True)))
    limit_card_memory()     # before the truth is read: a waiting job must hold no memory (the 2026-10-03 23:21 crash)
    # The one-piece lists the reading needs, counted on the truth before any fold trains (35 s).
    one_piece = pieces.lists(load_split()) if cfg.get("keep_largest") == "one_piece" else None
    floors = pieces.floors(load_split(), cfg["floor"]) if cfg.get("floor", False) is not False else None
    if (a.kill_ref is None) != (a.kill_step is None):
        raise SystemExit("--kill-ref and --kill-step go together")
    kill = None if a.kill_ref is None else {"ref": a.kill_ref, "step": a.kill_step, "margin": a.kill_margin}
    for fold in ([a.fold] if a.fold is not None else range(FOLDS)):
        got = train_fold(fold, cfg, a.version, a.force, kill)
        if got:
            print("fold %d: %.0f s, wrote %s" % (fold, got["seconds"], write_meta(a.version, cfg, [got], one_piece, floors)),
                  flush=True)


def real_cubes(cfg, batch, steps):
    """The batches the 1.5 mm phase trains on, from the real loader: fold 0's training scans (every other fold's), with
    the setting's coarse prior when it has one. The same Crops and stream a fold draws from."""
    split = load_split()
    skipped = prepare.unprepared("dev")
    mine = sorted(s for s, f in split["dev_folds"].items() if f != 0 and s not in skipped)
    if prior_version_of(cfg) is not None:
        require_coarse(prior_version_of(cfg), mine)
    return stream(Crops(mine, cfg, datasets.SEED, batch * steps, stride=0), batch, cfg["loaders"])


def bench(a):
    """Steps per second and peak memory of the real training step (`train_step`) on made-up cubes:
    ten steps to warm up, thirty timed. `--set key=<json>` tries a setting without editing CFG,
    which is how `tools/ab.py` compares two. `--real-crops` takes the cubes from the real 1.5 mm loader instead
    (`real_cubes`), so the rate includes the crops' own wait."""
    cfg = with_settings(CFG, a.set)
    limit_card_memory()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = int(cfg.get("cudnn_limit", 10))
    net = build(cfg, 118).train()
    ema = copy.deepcopy(net)
    for w in ema.parameters():
        w.requires_grad_(False)
    opt = make_opt(net, 1e-4, cfg)
    steps = 40 + (a.profile or 0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=steps, pct_start=cfg["warmup"],
                                                anneal_strategy="cos")
    scaler = make_scaler(cfg)
    forward = forward_of(net, cfg)
    loss_fn = loss_fn_of(cfg)
    gen = torch.Generator(device=DEV).manual_seed(0)
    p, n = a.patch or cfg["patch"], a.batch or 1
    ct = torch.randint(-1000, 1500, (n, p, p, p), dtype=torch.int16)
    lab = torch.randint(0, 118, (n, p, p, p), dtype=torch.uint8)
    feed = real_cubes(dict(cfg, patch=p), n, 40) if a.real_crops else None
    prior = []
    if feed is None and cfg.get("prior"):
        q = p // 2 if net_of(cfg) == "s2d" else p        # the prior sits on the net's own grid
        prior = [torch.randint(0, 118, (n, q, q, q), dtype=torch.uint8)]
    for step in range(1, 41):
        if step == 11:
            torch.cuda.synchronize() if DEV.type == "cuda" else None
            t0 = time.time()
        if feed is not None:
            ct, lab, *prior = next(feed)
        loss, _ = train_step(net, ema, opt, sched, scaler, ct, lab, cfg, gen, step, forward, loss_fn, *prior)
    torch.cuda.synchronize() if DEV.type == "cuda" else None
    rate = 30 / (time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEV.type == "cuda" else float("nan")
    params = sum(v.numel() for v in net.parameters())
    print("patch %d batch %d: %.2f steps/s, %.2f cubes/s, peak %.1f GB, %.1f M weights, loss %.4f"
          % (p, n, rate, rate * n, peak, params / 1e6, float(loss)))
    if a.profile:
        steps_done = iter(range(41, steps + 1))
        profile_steps(lambda: train_step(net, ema, opt, sched, scaler, ct, lab, cfg, gen, next(steps_done),
                                         forward, loss_fn, *prior),
                      a.profile, 1 / rate, "profile a step at patch %d batch %d" % (p, n), cfg)


# What a kernel's name says it is doing. The first group that matches wins, so the optimiser's
# foreach kernels are not counted as normalisation for the "norm" in clip_grad_norm.
KERNEL_KINDS = (
    ("optimiser, clipping, EMA", ("adam", "lerp", "foreach", "multi_tensor", "clip")),
    ("convolution", ("conv", "xmma", "implicit_gemm", "cutlass", "cudnn", "wgrad", "dgrad", "fprop",
                     "gemm", "winograd")),
    ("loss: softmax, nll, Dice", ("softmax", "nll", "scatter", "gather")),
    ("normalisation", ("norm", "welford", "var_mean")),
    ("warp", ("grid_sampler", "affine_grid")),
    ("copies", ("memcpy", "memset", "copy")),
)


def kernel_kind(name):
    low = name.lower()
    for kind, marks in KERNEL_KINDS:
        if any(m in low for m in marks):
            return kind
    return "other elementwise"


def profile_steps(run_step, steps, step_seconds, what, cfg):
    """What the card does in a training step: the PyTorch profiler over `steps` more steps, its
    kernels by kind, and how much of an unprofiled step (`step_seconds`, from `bench`) they fill.
    What the kernels leave empty is all that CUDA graphs or a tighter loop could ever win; a kind's
    share is the most that making it free could."""
    from torch.autograd import DeviceType  # noqa: PLC0415
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            run_step()
        torch.cuda.synchronize()
    spans, by_kind, by_name = [], {}, {}
    for e in prof.events():
        # A user annotation (the optimiser's "Optimizer.step#AdamW.step") spans kernels on the card
        # and would count them twice.
        if e.device_type != DeviceType.CUDA or getattr(e, "is_user_annotation", False):
            continue
        us = e.time_range.elapsed_us()
        spans.append((e.time_range.start, e.time_range.end))
        kind = kernel_kind(e.name)
        by_kind[kind] = by_kind.get(kind, 0.0) + us
        was = by_name.get(e.name, (kind, 0.0, 0))
        by_name[e.name] = (kind, was[1] + us, was[2] + 1)
    if not spans:
        raise SystemExit("the profiler recorded no kernels on the card: nothing to report")
    busy, end = 0.0, None
    for a_, b_ in sorted(spans):            # the union, should two streams ever overlap
        if end is None or a_ > end:
            busy += b_ - a_
            end = b_
        elif b_ > end:
            busy += b_ - end
            end = b_
    kernel_ms = sum(by_kind.values()) / 1000 / steps
    busy_ms, step_ms = busy / 1000 / steps, step_seconds * 1000
    print("%s: a step %.1f ms unprofiled; the card's kernels %.1f ms of it (%.0f%%), so at most "
          "%.1f ms (%.0f%%) for CUDA graphs or a tighter loop to win" % (
              what, step_ms, busy_ms, 100 * busy_ms / step_ms, max(step_ms - busy_ms, 0.0),
              100 * max(step_ms - busy_ms, 0.0) / step_ms))
    print("  by kind, ms a step and share of the kernels:")
    for kind, us in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print("    %-28s %7.1f  %4.0f%%" % (kind, us / 1000 / steps, 100 * us / 1000 / steps / kernel_ms))
    print("  the 25 longest kernels, ms a step, calls a step, kind:")
    for name, (kind, us, count) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:25]:
        print("    %7.2f  %4d  %-26s %s" % (us / 1000 / steps, count // steps, kind, name[:110]))
    with ledger.run("segmenter", what) as led:
        led.settings_from({"cfg": cfg, "profiled_steps": steps})
        led.number("step_ms", step_ms, "ms").number("kernel_ms", busy_ms, "ms")
        led.number("card_busy_share", busy_ms / step_ms)
        for kind, us in by_kind.items():
            led.number("ms_" + re.sub(r"[^a-z]+", "_", kind.lower()).strip("_"), us / 1000 / steps, "ms")



def set_floors(a):
    """Give a trained model a floor: count the sizes for q into its meta.json and read with them from
    now on (`reading` in meta.json, beside the settings it was trained with, which stay as they were)."""
    f = os.path.join(model_dir(a.version), "meta.json")
    meta = json.load(open(f, encoding="utf-8"))
    meta["floors"] = pieces.floors(load_split(), a.q)
    meta["reading"] = dict(meta.get("reading", {}), floor=a.q)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("version %d reads with a floor at q=%g (%d structures)" % (a.version, a.q, len(meta["floors"]["all"])))


def set_prior(a):
    """Make a trained model read version P's coarse maps from now on, in place of its own prior (`reading` in
    meta.json). P is checked the way the development setting `prior_from` is: a 3 mm version with a meta.json."""
    f = os.path.join(model_dir(a.version), "meta.json")
    meta = json.load(open(f, encoding="utf-8"))
    prior_version_for(meta, a.prior_from, meta_if_any(a.prior_from))
    meta["reading"] = dict(meta.get("reading", {}), prior_from=a.prior_from)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("version %d reads version %d's coarse maps from now on" % (a.version, a.prior_from))


def set_pieces(a):
    """Give a trained model the one-piece lists, counted from the truth only (as `train` does when keep_largest
    is "one_piece"), and read keeping the largest piece from now on (`reading` in meta.json)."""
    f = os.path.join(model_dir(a.version), "meta.json")
    if not os.path.exists(f):
        raise SystemExit("No meta.json for version %d: %s" % (a.version, f))
    meta = json.load(open(f, encoding="utf-8"))
    meta["one_piece"] = pieces.lists(load_split())
    meta["reading"] = dict(meta.get("reading", {}), keep_largest="one_piece")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("version %d reads keeping the largest piece of %d structures" % (a.version, len(meta["one_piece"]["all"])))


def rows_for(scan, split_name, fold, names, truth, said, both):
    return [{"method": METHOD, "split": split_name, "fold": "" if fold is None else fold,
             "scan": scan, "structure": n, "in_scope": 1, "drop_reason": "",
             "truth_voxels": int(truth[i]), "pred_voxels": int(said[i]), "both_voxels": int(both[i])}
            for i, n in enumerate(names)]


def dropped_rows(scan, split_name, fold, names, why):
    # The contract allows two reasons and no others. A scan whose masks are a different shape, or
    # on another grid, is a geometry mismatch; anything else that kept it out of the cache is a
    # scan that could not be read.
    reason = "geometry_mismatch" if why.startswith(("grid", "mask")) else "unreadable"
    return [{"method": METHOD, "split": split_name, "fold": "" if fold is None else fold,
             "scan": scan, "structure": n, "in_scope": 0, "drop_reason": reason,
             "truth_voxels": 0, "pred_voxels": 0, "both_voxels": 0} for n in names]


COLUMNS = ["method", "split", "fold", "scan", "structure", "in_scope", "drop_reason",
           "truth_voxels", "pred_voxels", "both_voxels"]


READING = {"overlap": 0.5, "keep_largest": False, "floor": False, "trim": False}


def reading_of(meta, items=None, part="all"):
    """How a model reads: its own recorded settings, then what was decided after it was trained
    (meta.json's `reading`, written by `segmenter.py floors`), then any tried with `predict --set`.
    `keep_largest` "one_piece" becomes the list meta.json records for the scans that part of the
    model trained on, and a `floor` the sizes counted there: a fold's own for its held-out scans, so
    the reading knows nothing of the scans it is scored on, and those of every development scan for
    the five folds together."""
    cfg = dict(meta["cfg"], **meta.get("reading", {}))
    out = {k: cfg.get(k, v) for k, v in READING.items()}
    out.update({k: v for k, v in with_settings({}, items).items() if k in READING})
    if out["keep_largest"] == "one_piece":
        out["keep_largest"] = meta["one_piece"][str(part)]
    if out["floor"] is not False:              # q = 0 is a floor too: the smallest real size seen
        counted = meta.get("floors", {})
        if counted.get("q") != out["floor"]:
            raise SystemExit("This model's floors were counted at q=%s, not %s: python segmenter.py floors "
                             "--version %s --q %s" % (counted.get("q"), out["floor"], meta["version"], out["floor"]))
        out["floor"] = counted[str(part)]
    return out


@torch.no_grad()
def under_the_floor(pred, floor, names):
    """Drop every structure predicted smaller than its floor (voxels at 1.5 mm): its voxels are left
    unlabelled. A structure a scan does not contain, outlined anyway, is usually a small island."""
    if not floor:
        return pred
    counts = torch.bincount(pred.reshape(-1).long(), minlength=len(names) + 1)[1:].cpu().numpy()
    small = [i + 1 for i, n in enumerate(names) if 0 < counts[i] < floor.get(n, 0)]
    if small:
        pred[torch.isin(pred.long(), torch.as_tensor(small, device=pred.device))] = 0
    return pred


@torch.no_grad()
def keep_largest_pieces(acc, only=None):
    """Per structure, keep only its largest connected piece on the working grid: everywhere else the
    structure's probability is set to zero before anything is interpolated, so a stray island of
    rib in the liver goes to whatever was the next likeliest there. `only` (class numbers) limits it
    to structures that are one piece in the first place: done to every structure, it cut apart the
    costal cartilages and pulmonary veins on version 1's fold 0 and lost 0.005 mean Dice overall."""
    from scipy import ndimage  # noqa: PLC0415
    lab = coarse_labels(acc).to(torch.int16).cpu().numpy()
    for c, box in enumerate(ndimage.find_objects(lab), start=1):
        if box is None or (only is not None and c not in only):
            continue
        mine = lab[box] == c
        parts, n = ndimage.label(mine)
        if n < 2:
            continue
        sizes = np.bincount(parts.reshape(-1))
        sizes[0] = 0
        stray = mine & (parts != int(sizes.argmax()))
        region = acc[c][box]
        region[torch.as_tensor(stray, device=region.device)] = 0
    return acc


def read_volume(models, ct_lo, patch, reading, names):
    """The models' probabilities for a 3 mm volume, read the way `reading` says: what a scored scan
    and a scan from outside both go through."""
    acc, weight = probabilities(models, ct_lo, patch, reading["overlap"], reading.get("trim", False))
    # keep_largest is true (every structure) or a list of structure names (only those).
    picked = reading["keep_largest"]
    if picked:
        acc = keep_largest_pieces(acc, {names.index(n) + 1 for n in picked} if isinstance(picked, list) else None)
    return acc, weight


@torch.no_grad()
def predict_scans(models, scans, split_name, fold_of, names, patch, skipped, reading=None, fine=False,
                  prior_version=None):
    reading = reading or dict(READING)
    if fine and reading["keep_largest"]:
        raise SystemExit("A 1.5 mm version reads raw or with a floor only: keep_largest needs the whole scan's "
                         "probabilities, which a 1.5 mm reading never holds.")
    if prior_version is not None:
        require_coarse(prior_version, [s for s in scans if s not in skipped])
    rows, t0, pace = [], time.time(), polite.Pace()
    for i, scan in enumerate(scans, 1):
        fold = fold_of.get(scan)
        if scan in skipped:
            rows += dropped_rows(scan, split_name, fold, names, skipped[scan])
            continue
        began = time.time()
        if fine:
            # The hi15 arrays are pad_even'ed: cropped back to the scan's own grid before anything is counted.
            ct15 = np.load(prepare.hi15_files(scan)[0], mmap_mode="r")
            nx, ny, nz = _native_shape(scan)
            coarse = None if prior_version is None else coarse_of(prior_version, scan)
            pred = read_fine(models, ct15[:nx, :ny, :nz], patch, reading, names, coarse=coarse)
            pred = under_the_floor(pred, reading["floor"], names)
        else:
            ct_lo = np.load(prepare.lo_files(scan)[0])
            acc, weight = read_volume(models, ct_lo, patch, reading, names)
            pred = under_the_floor(labels_on_grid(acc, weight, _native_shape(scan)), reading["floor"], names)
            del acc, weight
        truth, said, both = counts_against_truth(pred, scan, len(names))
        rows += rows_for(scan, split_name, fold, names, truth, said, both)
        pred = None
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        pace.rest(time.time() - began)
        if i % 25 == 0:
            rate = i / (time.time() - t0)
            print("  %d of %d, %.2f scans/s, %.0f s left" % (i, len(scans), rate, (len(scans) - i) / rate),
                  flush=True)
    return rows


def _native_shape(scan):
    """The scan's own grid, read off the truth file's header without decompressing the labels."""
    import zipfile  # noqa: PLC0415

    with zipfile.ZipFile(prepare.hi_file(scan)) as z, z.open("labels.npy") as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                       else np.lib.format.read_array_header_2_0(f))
    return tuple(int(v) for v in shape)


def check_with_version(split, tag, with_version, look=None):
    """A second version's net is a development experiment: it needs --tag. On the test split it is read only
    inside a second look that final_eval opened (it passes --look), never by hand."""
    if with_version is None:
        return
    if split == "test":
        if look is None or tag:
            raise SystemExit("--with-version on the test split is read only by final_eval's second look, which "
                             "passes --look: a hand-typed pair read is refused.")
        return
    if not tag:
        raise SystemExit("--with-version is a development experiment: it needs --tag.")


def check_look_open(version, with_version, look, state_file=None, latched=None):
    """A test read of a pair, or of a version read at 1.5 mm or with a coarse prior, is allowed only inside the second
    look final_eval recorded for that very read (prepare.open_look: the number, the latch, not finished)."""
    entry = prepare.open_look(look, state_file, latched)
    if (entry["version"], entry.get("with_version")) != (version, with_version):
        raise SystemExit("Look %s is version %s with %s, not version %d with %s."
                         % (look, entry["version"], entry.get("with_version"), version, with_version))


def check_pair(meta, meta2, version, other):
    """Two nets read together must share the patch and the structures list, and both be fine or neither."""
    if is_fine_version(meta) != is_fine_version(meta2):
        raise SystemExit("Versions %d and %d cannot be read together: one reads at 1.5 mm and the other at 3 mm."
                         % (version, other))
    if meta["cfg"]["patch"] != meta2["cfg"]["patch"]:
        raise SystemExit("Versions %d and %d have different patches (%s and %s)."
                         % (version, other, meta["cfg"]["patch"], meta2["cfg"]["patch"]))
    if net_of(meta["cfg"]) != net_of(meta2["cfg"]):
        raise SystemExit("Versions %d and %d are different networks (%s and %s), so they cannot be read together."
                         % (version, other, net_of(meta["cfg"]), net_of(meta2["cfg"])))
    if reading_prior(meta) != reading_prior(meta2):
        raise SystemExit("Versions %d and %d take different coarse priors (%s and %s), so they cannot be read together."
                         % (version, other, reading_prior(meta), reading_prior(meta2)))
    if list(meta["structures"]) != list(meta2["structures"]):
        raise SystemExit("Versions %d and %d have different structures lists." % (version, other))


def predict(a):
    datasets.guard_test(a.split)
    with_version = getattr(a, "with_version", None)
    look = getattr(a, "look", None)
    check_with_version(a.split, a.tag, with_version, look)
    if look is not None and a.split != "test":
        raise SystemExit("--look goes with the test split: final_eval's second look.")
    if a.split == "test":
        head = json.load(open(os.path.join(model_dir(a.version), "meta.json"), encoding="utf-8"))
        # A pair, a version read at 1.5 mm and a version with a coarse prior read the test split only inside a look.
        if look is not None or with_version is not None or is_fine_version(head) or prior_version_of(head["cfg"]) is not None:
            check_look_open(a.version, with_version, look)
    if a.set and (a.split == "test" or not a.tag):
        raise SystemExit("A reading setting is a development experiment: it needs --tag, and never the test.")
    prior_from = sets.parse(a.set).get("prior_from") if a.set else None
    named = None
    if prior_from is not None:
        own_meta = json.load(open(os.path.join(model_dir(a.version), "meta.json"), encoding="utf-8"))
        named = meta_if_any(prior_from)
        prior_version_for(own_meta, prior_from, named)
        print("prior_from: version %d's coarse maps are read in place of version %d's own prior, version %d"
              % (prior_from, a.version, prior_version_of(own_meta["cfg"])))
    tag = "_" + a.tag if a.tag else ""
    steady_inference()
    limit_card_memory()
    split = load_split()
    names = datasets.structures()
    skipped = prepare.unprepared(a.split)
    os.makedirs(work_dir(a.version), exist_ok=True)
    out_dir = getattr(a, "out_dir", None) or prepare.OUT      # only the final pred file moves
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, "pred_%s_%s.csv" % (METHOD, a.split))
    with ledger.run("segmenter", "predict %s v%d" % (a.split, a.version)) as led:
        settings = {"fold": "all" if a.fold is None else a.fold, "limit": a.limit or "all",
                    "reading": a.set or [], "tag": a.tag or "",
                    "with_version": "" if with_version is None else with_version,
                    "look": "" if look is None else look}
        if prior_from is not None:
            settings["prior_from"] = prior_from
        led.settings_from(settings)
        t0, predicted = time.time(), 0
        if a.split == "test":
            models, meta = load_models(a.version, range(FOLDS))
            if with_version is not None:
                # A pair: version's five nets and with_version's five, every net averaged with equal weight.
                models2, meta2 = load_models(with_version, range(FOLDS))
                check_pair(meta, meta2, a.version, with_version)
                models = list(models) + list(models2)
            scans = sorted(split["test"])
            rows = predict_scans(models, scans, "test", {}, names, meta["cfg"]["patch"], skipped,
                                 reading_of(meta), is_fine_version(meta), reading_prior(meta))
            predicted = len(scans)
        else:
            rows = []
            for fold in ([a.fold] if a.fold is not None else range(FOLDS)):
                part = os.path.join(work_dir(a.version), "oof_fold%d%s.csv" % (fold, tag))
                # A file older than meta.json was read before the model's reading last changed
                # (a floor set by `segmenter.py floors`, say), so it is read again.
                fresh = os.path.exists(part) and os.path.getmtime(part) > os.path.getmtime(
                    os.path.join(model_dir(a.version), "meta.json"))
                # A pair is always read again, so a tag reused with another partner is never stale.
                if fresh and not a.force and with_version is None:
                    rows += pd.read_csv(part, keep_default_na=False).to_dict("records")
                    continue
                models, meta = load_models(a.version, [fold])
                if with_version is not None:
                    if not os.path.exists(os.path.join(model_dir(with_version), "fold%d.pt" % fold)):
                        raise SystemExit("Version %d has no net for fold %d." % (with_version, fold))
                    models2, meta2 = load_models(with_version, [fold])
                    check_pair(meta, meta2, a.version, with_version)
                    models = list(models) + list(models2)
                scans = sorted(s for s, f in split["dev_folds"].items() if f == fold)
                if a.limit:
                    scans = scans[:a.limit]
                got = predict_scans(models, scans, "dev", split["dev_folds"], names,
                                    meta["cfg"]["patch"], skipped, reading_of(meta, a.set, fold), is_fine_version(meta),
                                    reading_prior(meta) if prior_from is None else prior_version_for(meta, prior_from, named))
                predicted += len(scans)
                if not a.limit:
                    pd.DataFrame(got)[COLUMNS].to_csv(part, index=False)
                rows += got
                del models
        d = pd.DataFrame(rows)[COLUMNS]
        scans_done = d["scan"].nunique()
        here = d[(d["in_scope"] == 1) & (d["truth_voxels"] > 0)]
        dice = 2 * here["both_voxels"] / (here["truth_voxels"] + here["pred_voxels"])
        headline = float(dice.groupby(here["structure"]).mean().mean())
        absent = d[(d["in_scope"] == 1) & (d["truth_voxels"] == 0)]
        invention = float((absent["pred_voxels"] > 0).mean()) if len(absent) else float("nan")
        led.number("scans", scans_done, "count").number("mean_dice_all_structures", headline)
        led.number("invention_rate", invention).number("scans_predicted", predicted, "count")
        # Per scan actually predicted: folds read back from an earlier run took no time here.
        led.number("seconds_per_scan", (time.time() - t0) / max(predicted, 1), "s")
    if a.limit or a.tag or (a.split == "dev" and a.fold is not None):
        # The scorer reads every pred_*_dev.csv it finds and treats a missing row as an error, so a
        # file holding one fold, or a few scans of one, must never be written where it would look.
        print("a partial run: %d scan(s), mean Dice %.4f, invention rate %.4f. The scorer's file is "
              "only written by a full run." % (scans_done, headline, invention))
        return
    d.to_csv(final, index=False)
    print("%d row(s) over %d scan(s) -> %s" % (len(d), scans_done, final))
    print("  mean Dice over structures %.4f, invention rate %.4f (score.py has the intervals)"
          % (headline, invention))



def read_scan(a):
    """Label one CT volume from a NIfTI file, and write the labels on its own grid.

    Measured only on scans at 1.5 mm, which is what the dataset is. A scan at another spacing is
    resampled to 3 mm by interpolation, and that path has no scorecard behind it: the labels come
    with a warning saying so."""
    import nibabel as nib  # noqa: PLC0415

    steady_inference()
    limit_card_memory()
    models, meta = load_models(a.version, range(FOLDS))
    if is_fine_version(meta):
        raise SystemExit("a 1.5 mm version reads prepared scans only for now")
    names = meta["structures"]
    img = nib.as_closest_canonical(nib.load(a.file))
    ct = np.asanyarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    t0 = time.time()
    measured = all(abs(v - prepare.SPACING) < 1e-3 for v in zooms)
    fine = ct
    if not measured:
        # Onto the grid the model was measured on first, so everything after this line is the
        # scored path and the only unscored step is this one resampling.
        size = [max(2, int(round(s * v / prepare.SPACING))) for s, v in zip(ct.shape, zooms)]
        fine = F.interpolate(torch.as_tensor(ct)[None, None], size=size, mode="trilinear",
                             align_corners=False)[0, 0].numpy()
        print("warning: this scan is at %s mm, not 1.5 mm. The scorecard was measured at 1.5 mm; "
              "these labels come through a resampling nobody has scored." % (zooms,), file=sys.stderr)
    lo = prepare.average_down(np.rint(fine).astype(np.int16))
    reading = reading_of(meta)
    acc, weight = read_volume(models, lo, meta["cfg"]["patch"], reading, names)
    pred = under_the_floor(labels_on_grid(acc, weight, tuple(fine.shape)), reading["floor"], names)
    if not measured:
        pred = F.interpolate(pred[None, None].float(), size=ct.shape, mode="nearest")[0, 0].to(torch.uint8)
    seconds = time.time() - t0
    found = torch.bincount(pred.reshape(-1).long(), minlength=len(names) + 1)[1:].cpu().numpy()
    voxel_ml = float(np.prod(zooms)) / 1000.0
    listing = [{"label": i + 1, "structure": n, "voxels": int(found[i]), "millilitres": round(found[i] * voxel_ml, 1)}
               for i, n in enumerate(names) if found[i]]
    if a.out:
        out = nib.Nifti1Image(pred.cpu().numpy(), img.affine)
        out.header.set_zooms(zooms)
        nib.save(out, a.out)
        with open(os.path.splitext(os.path.splitext(a.out)[0])[0] + ".labels.json", "w", encoding="utf-8") as f:
            json.dump({"not_a_medical_device": True, "measured_spacing": measured,
                       "labels": {str(i + 1): n for i, n in enumerate(names)}, "found": listing}, f, indent=1)
    print(json.dumps({"file": a.file, "seconds": round(seconds, 2), "device": DEV.type,
                      "measured_spacing": measured, "structures_found": len(listing),
                      "not_a_medical_device": True}, indent=1))
    for row in listing:
        print("  %-28s %8.1f ml" % (row["structure"], row["millilitres"]))


def selftest(a=None):
    """Resume proof on the CPU in seconds, through the real train_fold with made-up cubes (no scans, no card, no
    ledger row, nothing under models/): 10 steps in two phases, stopped after step 4 (inside phase 1) and after
    step 8 (inside phase 2), each continued from its partial; then the whole-fold skip by fingerprint and a corrupt
    partial. Usage: python segmenter.py selftest (with CUDA_VISIBLE_DEVICES empty)."""
    import shutil
    import tempfile
    import types

    class FakeCrops:
        def __init__(self, scans, cfg, seed, count, stride=1):
            self.seed, self.count, self.stride = seed, count, stride
            if stride == 0:
                # 1.5 mm phase: patch size is unchanged
                self.q = cfg["patch"]
            else:
                self.q = cfg["patch"] // stride

        def __len__(self):
            return self.count

        def __getitem__(self, i):
            r = np.random.default_rng([self.seed, i])
            q = self.q
            return (torch.from_numpy(r.integers(-1000, 1000, (q, q, q)).astype(np.int16)),
                    torch.from_numpy(r.integers(0, 3, (q, q, q)).astype(np.uint8)))

    class FakeLedger:
        def run(self, *args):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __getattr__(self, name):
            return lambda *args, **kw: self

    root = tempfile.mkdtemp(prefix="segmenter-selftest-")
    cfg = {**CFG, "patch": 16, "widths": [4, 4, 4], "rare_sampling": False, "compile": False, "compile_loss": False,
           "cudnn_limit": 0, "fused_adam": False, "curve_every": 1000, "curve_scans": 1, "loaders": 1, "layout": "plain",
           "phases": [{"mm": 6, "steps": 6, "lr": 1e-3, "batch": 2}, {"mm": 3, "steps": 4, "lr": 5e-4, "batch": 1}]}
    mine = {"load_split": lambda: {"dev_folds": {"s%d" % i: i % 5 for i in range(10)}},
            "Crops": FakeCrops, "curve_point": lambda *args, **kw: 0.5, "ledger": FakeLedger(),
            "model_dir": lambda version: os.path.join(root, "v%d" % version)}
    was = {k: globals()[k] for k in mine}
    was_ds, was_prep = datasets.structures, prepare.unprepared
    saved_env = {k: os.environ.get(k) for k in ("RAIDIOLOGY_NO_RESUME", "RAIDIOLOGY_SAVE_EVERY_S",
                                                 "RAIDIOLOGY_CKPT_SELFTEST_STOP", "RAIDIOLOGY_SHARE")}
    os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
    os.environ["RAIDIOLOGY_SAVE_EVERY_S"] = "0"                    # a save after every step
    os.environ["RAIDIOLOGY_SHARE"] = "1"                           # no pacing pauses
    globals().update(mine)
    datasets.structures, prepare.unprepared = (lambda: ["a", "b"]), (lambda split: set())

    def run(version, stop=None, c=cfg):
        if stop:
            os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = str(stop)
        else:
            os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
        return train_fold(0, c, version)

    def stopped(version, stop, c=cfg):
        try:
            run(version, stop, c)
        except foldck.SelftestStop:
            return torch.load(os.path.join(root, "v%d" % version, "fold0.partial.pt"), map_location="cpu",
                              weights_only=False)
        raise AssertionError("the stop did not fire")

    def resume_proof(c, base):
        """The resume proof for the recipe `c`, in versions base+1 to base+4: a fresh start, two stops (inside each phase),
        a finished fold skipped, an old fold kept, a corrupt partial renamed. The 3 mm recipe and the space-to-depth net
        both pass it."""
        v1, v2, v3, v4 = base + 1, base + 2, base + 3, base + 4
        run(v1, c=c)
        assert PROBE["steps_run"] == 10 and PROBE["resumed_step"] is None
        d = os.path.join(root, "v%d" % v1)
        assert os.path.exists(os.path.join(d, "fold0.pt")) and os.path.exists(os.path.join(d, "fold0.pt.fp"))
        assert not os.path.exists(os.path.join(d, "fold0.partial.pt")), "partial gone after the model"
        assert isinstance(torch.load(os.path.join(d, "fold0.pt"), map_location="cpu"), dict), "still a bare state dict"
        for version, stop, steps_left in ((v2, 4, 6), (v3, 8, 2)):
            saved = stopped(version, stop, c)
            assert saved["step"] == stop and not os.path.exists(os.path.join(root, "v%d" % version, "fold0.pt"))
            got = run(version, c=c)
            assert PROBE["resumed_step"] == stop, "resumed at the saved step"
            assert foldck.same(PROBE["opt"], saved["opt"]) and foldck.same(PROBE["sched"], saved["sched"]), \
                "optimiser and schedule are the saved ones"
            assert PROBE["steps_run"] == steps_left, "only the rest ran (%d)" % PROBE["steps_run"]
            dv = os.path.join(root, "v%d" % version)
            assert got["curve"] and not os.path.exists(os.path.join(dv, "fold0.partial.pt")), "partial gone"
            assert os.path.exists(os.path.join(dv, "fold0.pt.fp"))
        # a finished fold with a matching fingerprint is not trained again; another recipe is; an old one (no .fp) is kept
        assert run(v1, c=c) is None and PROBE["skipped"] and PROBE["steps_run"] == 0
        run(v1, c={**c, "ema": 0.99})
        assert not PROBE["skipped"] and PROBE["steps_run"] == 10
        os.remove(os.path.join(d, "fold0.pt.fp"))
        assert run(v1, c=c) is None and PROBE["skipped"], "an old model without a fingerprint stays as it was"
        # a corrupt partial is renamed .bad and the fold starts fresh
        stopped(v4, 3, c)
        with open(os.path.join(root, "v%d" % v4, "fold0.partial.pt"), "wb") as f:
            f.write(b"not a checkpoint")
        run(v4, c=c)
        assert os.path.exists(os.path.join(root, "v%d" % v4, "fold0.partial.pt.bad")), "corrupt partial renamed .bad"
        assert PROBE["resumed_step"] is None and PROBE["steps_run"] == 10, "fresh start"

    try:
        resume_proof(cfg, 0)
        # the space-to-depth net at patch 32: its folded body keeps every level at 2 voxels or more (patch 16 would not)
        resume_proof({**cfg, "net": "s2d", "patch": 32,
                      "phases": [{"mm": 1.5, "steps": 6, "lr": 1e-3, "batch": 2},
                                 {"mm": 1.5, "steps": 4, "lr": 5e-4, "batch": 1}]}, 100)
        # the one-hour rule: behind the reference's curve at the judged step (0.5 against 0.9 less 0.01), the fold
        # stops there; level within the margin (0.5 against 0.505), it trains on; a curve on other scans is refused
        scans = run(60)["curve_scans"]
        for ref, dice, curve_scans, killed, steps in ((61, 0.9, scans, 6, 6), (62, 0.505, scans, None, 10),
                                                      (63, 0.1, ["elsewhere"], None, 0)):
            os.makedirs(os.path.join(root, "v%d" % ref), exist_ok=True)
            with open(os.path.join(root, "v%d" % ref, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"folds": [{"fold": 0, "curve": [{"step": 6, "mm": 6, "dice": dice}],
                                      "curve_scans": curve_scans}]}, fh)
            try:
                train_fold(0, cfg, ref + 10, kill={"ref": ref, "step": 6, "margin": 0.01})
                stop = ""
            except SystemExit as e:
                stop = str(e)
            assert PROBE["killed"] == killed and PROBE["steps_run"] == steps, (ref, PROBE, stop)
            assert ("KILLED" in stop) == bool(killed) and ("not like for like" in stop) == (ref == 63), stop
        m = {"cfg": {"patch": 96}, "structures": ["a", "b"]}
        check_pair(m, {"cfg": {"patch": 96}, "structures": ["a", "b"]}, 1, 2)
        for other in ({"cfg": {"patch": 64}, "structures": ["a", "b"]},
                      {"cfg": {"patch": 96}, "structures": ["a", "c"]}):
            try:
                check_pair(m, other, 1, 2)
                raise AssertionError("an unlike pair must be refused")
            except SystemExit:
                pass
        check_with_version("dev", "t", 2)
        check_with_version("dev", None, None)
        for split, tag in (("test", "t"), ("dev", None)):
            try:
                check_with_version(split, tag, 2)
                raise AssertionError("--with-version outside a tagged development run must be refused")
            except SystemExit:
                pass
        # the pair's test read: refused by hand, and allowed only inside the opened look final_eval recorded
        check_with_version("test", None, 2, 2)
        try:
            check_with_version("test", None, 2)
            raise AssertionError("a hand-typed pair read on the test split must be refused")
        except SystemExit:
            pass
        with tempfile.TemporaryDirectory() as tmpd:
            st = os.path.join(tmpd, "final_eval.json")

            def write_looks(looks):
                with open(st, "w", encoding="utf-8") as f:
                    json.dump({"looks": looks}, f)

            def allowed(version, with_version, look, latched):
                try:
                    check_look_open(version, with_version, look, state_file=st, latched=latched)
                except SystemExit:
                    return False
                return True

            open_pair = {"look": 2, "version": 116, "with_version": 115, "finished": None}
            assert not allowed(116, 115, 2, True), "no state file: no look is open"
            write_looks([open_pair])
            assert allowed(116, 115, 2, True), "the opened pair look reads the test"
            assert not allowed(116, 115, 2, False), "without the ledger latch the test is not read"
            assert not allowed(116, 115, 3, True), "another look number is refused"
            assert not allowed(116, 114, 2, True), "another partner is refused"
            write_looks([{**open_pair, "finished": "t"}])
            assert not allowed(116, 115, 2, True), "a finished look is refused"
            write_looks([{"look": 2, "version": 116, "finished": None}])
            assert not allowed(116, 115, 2, True), "a single version's look is not a pair's"
        # Test 1.5 mm phase in a two-phase train (6 mm, 3 mm, 1.5 mm = 3 phases)
        cfg_1p5 = {**cfg, "phases": [{"mm": 6, "steps": 2, "lr": 1e-3, "batch": 2},
                                     {"mm": 3, "steps": 2, "lr": 5e-4, "batch": 1},
                                     {"mm": 1.5, "steps": 2, "lr": 2.5e-4, "batch": 1}]}
        run(5, c=cfg_1p5)
        assert PROBE["steps_run"] == 6, "three phases = 2+2+2 steps"
        # is_fine_version check
        m_fine = {"cfg": {**cfg_1p5}, "structures": ["a", "b"]}
        assert is_fine_version(m_fine), "version with 1.5 mm as last phase is fine"
        m_not_fine = {"cfg": {**cfg}, "structures": ["a", "b"]}
        assert not is_fine_version(m_not_fine), "version without 1.5 mm phase is not fine"
        # Test init_from: train version 6 with init_from version 1
        # First create a meta.json for version 1 so init_from can load it
        v1_dir = os.path.join(root, "v1")
        os.makedirs(v1_dir, exist_ok=True)
        with open(os.path.join(v1_dir, "meta.json"), "w") as f:
            json.dump({"cfg": cfg, "structures": ["a", "b"], "widths": cfg["widths"]}, f)
        run(6, c={**cfg, "init_from": 1})
        # Check that init_from refuses mismatched widths
        cfg_different_widths = {**cfg, "widths": [8, 8, 8], "init_from": 1}
        try:
            run(7, c=cfg_different_widths)
            raise AssertionError("init_from should refuse different widths")
        except SystemExit:
            pass
        # (a) the ownership intervals on an axis of odd length: every voxel owned once, in order
        for size, patch_ in ((37, 16), (16, 16), (9, 16), (100, 16)):
            starts = _starts(size, patch_, 0.5)
            got = owned_intervals(starts, patch_, size)
            assert got[0][0] == 0 and got[-1][1] == size, "the intervals cover the axis"
            assert all(h0 == l1 for (_, h0), (l1, _) in zip(got, got[1:])), "contiguous and in order"
            assert all(lo <= hi for lo, hi in got), "no interval runs backwards"
            owner = np.full(size, -1)
            for w, (lo, hi) in enumerate(got):
                assert (owner[lo:hi] == -1).all(), "no voxel owned twice"
                owner[lo:hi] = w
            assert (owner >= 0).all(), "every voxel owned"
            centres = np.array([s + patch_ // 2 for s in starts])
            assert (owner == np.abs(np.arange(size)[:, None] - centres[None]).argmin(axis=1)).all(),                 "the nearest centre owns the voxel, a tie going to the earlier window"
        # (b), (c) read_fine against probabilities() in one window, and against a plain numpy reference in several
        net = build({"widths": [4, 4, 4], "deep": [1.0], "layout": "plain"}, 3).eval()
        with torch.no_grad():
            net.heads[0].weight.mul_(50.0)           # decisive probabilities: no near-ties for half precision to flip
        rd = {"overlap": 0.5, "keep_largest": False, "floor": False}
        r = np.random.default_rng(3)
        one = r.integers(-1000, 1000, (16, 16, 16)).astype(np.int16)
        acc, weight = probabilities([net], one, 16, 0.5)
        want = (acc.float() / weight).argmax(dim=0).to(torch.uint8)
        assert torch.equal(read_fine([net], one, 16, rd, ["a", "b"]), want), "one window: argmax of probabilities()"
        many = r.integers(-1000, 1000, (20, 37, 19)).astype(np.int16)
        got = read_fine([net], many, 16, rd, ["a", "b"]).cpu().numpy()
        assert got.shape == many.shape and got.dtype == np.uint8
        x = to_input(torch.as_tensor(many, device=DEV))
        st = [_starts(s, 16, 0.5) for s in many.shape]
        preds = {}
        for a in st[0]:
            for b in st[1]:
                for c in st[2]:
                    w = x[a:a + 16, b:b + 16, c:c + 16][None, None].contiguous()
                    preds[a, b, c] = window_probabilities(net, w).argmax(dim=0).cpu().numpy()
        near = [np.abs(np.arange(many.shape[i])[:, None] - np.array([s + 8 for s in st[i]])[None]).argmin(axis=1)
                for i in range(3)]
        ref = np.zeros(many.shape, dtype=np.uint8)
        for i in range(many.shape[0]):
            for j in range(many.shape[1]):
                for k in range(many.shape[2]):
                    a, b, c = st[0][near[0][i]], st[1][near[1][j]], st[2][near[2][k]]
                    ref[i, j, k] = preds[a, b, c][i - a, j - b, k - c]
        assert np.array_equal(got, ref), "several windows: the owner's label at every voxel"
        # 2026-10-08: the card's test read sums five nets per window; read_fine must keep no autograd graph
        gnets = [build({"widths": [4, 4, 4], "deep": [1.0], "layout": "plain"}, 3).eval() for _ in range(2)]
        assert all(q.requires_grad for n in gnets for q in n.parameters()), "the nets' parameters require grad"
        vol = r.integers(-1000, 1000, (20, 18, 12)).astype(np.int16)
        seen, real_window = [], window_probabilities

        def window_recorder(m, window, prior=None):
            pr = real_window(m, window, prior)
            seen.append((torch.is_grad_enabled(), pr.requires_grad))
            return pr

        globals()["window_probabilities"] = window_recorder
        try:
            decorated = read_fine(gnets, vol, 16, rd, ["a", "b"])
            dec_seen = list(seen)
            seen.clear()
            with torch.enable_grad():
                plain = read_fine.__wrapped__(gnets, vol, 16, rd, ["a", "b"])
        finally:
            globals()["window_probabilities"] = real_window
        assert dec_seen and not any(g or q for g, q in dec_seen), "decorated: every window ran with grad disabled, no graph kept"
        assert seen and all(g and q for g, q in seen), "undecorated under enable_grad: every window keeps a graph"
        assert decorated.requires_grad is False and torch.equal(decorated, plain), "the labels are the same with and without the decorator"
        print("fine read without gradients ok")
        # a fine version read with keep_largest, a fine version paired with a coarse one, and an outside scan
        try:
            predict_scans([net], [], "dev", {}, ["a", "b"], 16, {}, {**rd, "keep_largest": True}, fine=True)
            raise AssertionError("a fine version with keep_largest must be refused")
        except SystemExit:
            pass
        try:
            check_pair({**m_fine, "version": 1}, {**m_not_fine, "version": 2}, 5, 1)
            raise AssertionError("a fine and a coarse version must not be paired")
        except SystemExit:
            pass
        # the 1.5 mm Crops on a tiny cache whose first axis (12) is shorter than the patch (16)
        shape3 = (6, 20, 8)
        hi_dir = os.path.join(root, "hi15")
        os.makedirs(hi_dir, exist_ok=True)
        np.save(os.path.join(hi_dir, "t0_ct.npy"), np.full(tuple(2 * s for s in shape3), 5, dtype=np.int16))
        np.save(os.path.join(hi_dir, "t0_lab.npy"), np.full(tuple(2 * s for s in shape3), 1, dtype=np.uint8))
        was_pts, was_hi15 = prepare.points, prepare.hi15_files
        prepare.points = lambda split: {"t0": (shape3, np.array([1], dtype=np.uint8), np.array([5], dtype=np.int64))}
        prepare.hi15_files = lambda scan: (os.path.join(hi_dir, scan + "_ct.npy"), os.path.join(hi_dir, scan + "_lab.npy"))
        try:
            crops = was["Crops"](["t0"], {**cfg, "rare_sampling": True}, 7, 12, stride=0)
            for i in range(12):
                c_, l_ = crops[i]
                assert tuple(c_.shape) == (16, 16, 16) and c_.dtype == torch.int16 and l_.dtype == torch.uint8
                assert int((c_ == 5).any(dim=2).any(dim=1).sum()) == 12, "the 12-voxel axis sits whole inside the cube"
                assert int((c_ == 5).any(dim=0).any(dim=1).sum()) == 16 and (l_ == 1).sum() == (c_ == 5).sum()
        finally:
            prepare.points, prepare.hi15_files = was_pts, was_hi15
        # the 1.5 mm jitter stays in the cube's central half: +-patch/4 voxels at 1.5 mm, half of that at 3 mm
        prepare.points = lambda split: {"t0": (shape3, np.array([1], dtype=np.uint8), np.array([5], dtype=np.int64))}
        try:
            fc = was["Crops"](["t0"], {**cfg, "rare_sampling": True, "structure_share": 1.0}, 7, 1, stride=0)
            point = np.array(np.unravel_index(5, shape3))
            spread = max(int(np.abs(fc._structure_first(np.random.default_rng(i), cfg["patch"] // 2)[2] - point).max())
                         for i in range(300))
            assert 0 < spread <= cfg["patch"] // 8, "fine jitter is within the central half of the cube (%d)" % spread
        finally:
            prepare.points = was_pts
        # (i) the prior's cut of a coarse map: each 3 mm voxel is two 1.5 mm voxels, and past the map's edge is background
        lab15 = np.random.default_rng(9).integers(0, 4, (20, 18, 12)).astype(np.uint8)
        coarse = prepare.vote_down(lab15)
        rep = np.repeat(np.repeat(np.repeat(coarse, 2, 0), 2, 1), 2, 2)
        want = np.zeros((16, 16, 16), dtype=np.uint8)
        want[:16, :16, :12] = rep[:16, :16, :12]
        got = prior_cube(coarse, [slice(0, 16)] * 3, [slice(0, 16)] * 3, 16)
        assert np.array_equal(got, want), "a window at the origin: each voxel is its coarse voxel, zero past the edge"
        got = prior_cube(coarse, [slice(0, 8)] * 3, [slice(4, 12)] * 3, 16)
        want = np.zeros((16, 16, 16), dtype=np.uint8)
        want[4:12, 4:12, 4:12] = rep[:8, :8, :8]
        assert np.array_equal(got, want), "a window placed at an offset in the cube"
        got = prior_cube(coarse, [slice(10, 26)] * 3, [slice(0, 16)] * 3, 16)
        want = np.zeros((16, 16, 16), dtype=np.uint8)
        for j in range(16):
            for k in range(16):
                for m_ in range(16):
                    idx = ((10 + j) // 2, (10 + k) // 2, (10 + m_) // 2)
                    if all(i < n for i, n in zip(idx, coarse.shape)):
                        want[j, k, m_] = coarse[idx]
        assert np.array_equal(got, want), "a window past the coarse map's near edge is background beyond it"
        # (ii) the prior turns with the labels, exactly: nearest on the same grid
        wcfg = {"warp_share": 1.0, "warp_scale": (0.9, 1.1), "warp_rotate": 0.3}
        xw = torch.rand(2, 1, 8, 8, 8, device=DEV)
        yw = torch.as_tensor(np.random.default_rng(2).integers(0, 4, (2, 8, 8, 8)), device=DEV)
        zw = (yw > 0).long() * 7
        _, y2, z2 = warp(xw, yw, wcfg, torch.Generator(device=DEV).manual_seed(3), zw)
        assert torch.equal(z2, (y2 > 0).long() * 7), "the prior follows the labels through the same warp"
        # (iii) a net with a prior, its first conv's new channels zero, is the 3 mm net at step 0
        cfg_p = {**cfg, "prior": {"version": 1, "dim": 4}}
        net_plain = build(cfg, 3).eval()
        net_prior = build(cfg_p, 3).eval()
        net_prior.load_state_dict(with_prior_input(net_plain.state_dict(), net_prior))
        xp = torch.randn(1, 1, 16, 16, 16, device=DEV)
        pp = torch.randint(0, 3, (1, 16, 16, 16), device=DEV)
        with torch.no_grad():
            diff = (net_plain(xp)[0] - net_prior(xp, pp)[0]).abs().max().item()
        assert diff < 1e-5, "the prior net starts as the plain net (max diff %g)" % diff
        # (v) training's perturbation: the same shape, and no label the cube did not hold
        cube = np.zeros((16, 16, 16), dtype=np.uint8)
        cube[4:12, 4:12, 4:12] = 2
        cube[0:3, 0:3, 0:3] = 1
        moved = perturb_prior(cube, np.random.default_rng(5))
        assert moved.shape == cube.shape and moved.dtype == cube.dtype
        assert set(np.unique(moved)) <= set(np.unique(cube)), "no label appears that the cube did not hold"
        # (iv) coarse maps: written under the fold's own net, not written again, a missing one written again
        was_out, was_lo = prepare.OUT, prepare.lo_files
        v8 = os.path.join(root, "v8")
        os.makedirs(v8, exist_ok=True)
        torch.save(build(cfg, 3).state_dict(), os.path.join(v8, "fold0.pt"))
        with open(os.path.join(v8, "meta.json"), "w") as f:
            json.dump({"cfg": cfg, "structures": ["a", "b"]}, f)
        lo_dir = os.path.join(root, "lo")
        os.makedirs(lo_dir, exist_ok=True)
        for scan in ("s0", "s1"):
            np.save(os.path.join(lo_dir, scan + "_ct.npy"),
                    np.random.default_rng(4).integers(-1000, 1000, (20, 18, 12)).astype(np.int16))
        prepare.OUT = os.path.join(root, "out")
        prepare.lo_files = lambda scan: (os.path.join(lo_dir, scan + "_ct.npy"), None)
        try:
            dev_folds = {"s0": 0, "s1": 0, "s2": 1}
            assert write_coarse(8, [0], dev_folds, set()) == 2, "the two held-out scans of fold 0 are written"
            got = np.load(coarse_file(8, "s0"), mmap_mode="r")
            assert got.shape == (20, 18, 12) and got.dtype == np.uint8 and int(got.max()) < 3
            man = json.load(open(os.path.join(coarse_dir(8), "manifest.json"), encoding="utf-8"))
            assert man["scans"]["s1"]["fold"] == 0 and man["scans"]["s1"]["shape"] == [20, 18, 12]
            assert man["folds"]["0"] == sha256(os.path.join(v8, "fold0.pt")), "the manifest names the net"
            assert write_coarse(8, [0], dev_folds, set()) == 0, "a rerun with the same net writes nothing"
            os.remove(coarse_file(8, "s1"))
            assert write_coarse(8, [0], dev_folds, set()) == 1, "a missing map is written again"
            assert np.array_equal(np.load(coarse_file(8, "s0")), np.asarray(got)), "a map read back is the same"
            require_coarse(8, ["s0", "s1"])
            try:
                require_coarse(8, ["s0", "s9"])
                raise AssertionError("a scan without a coarse map must be refused")
            except SystemExit:
                pass
        finally:
            prepare.OUT, prepare.lo_files = was_out, was_lo
        # idea 130's prior_from: refused on a model with no prior and on a 1.5 mm or missing version; otherwise the
        # named 3 mm version's maps are the prior, and without the setting the model's own
        prior_none = {"version": 5, "cfg": {"patch": 16}}
        prior_own = {"version": 122, "cfg": {"patch": 16, "prior": {"version": 116, "dim": 16}}}
        prior_3mm = {"version": 124,
                     "cfg": {"patch": 16, "phases": [{"mm": prepare.SPACING * prepare.FACTOR, "steps": 10}]}}
        prior_1_5 = {"version": 126, "cfg": {"patch": 16, "phases": [{"mm": prepare.SPACING, "steps": 10}]}}
        for own, from_v, named in ((prior_none, 124, prior_3mm), (prior_own, 126, prior_1_5), (prior_own, 124, None)):
            try:
                prior_version_for(own, from_v, named)
                raise AssertionError("prior_from %d on version %d must be refused" % (from_v, own["version"]))
            except SystemExit:
                pass
        assert prior_version_for(prior_own, None, None) == 116, "without prior_from the model's own prior is read"
        assert prior_version_for(prior_none, None, None) is None, "a model with no prior reads none without prior_from"
        assert prior_version_for(prior_own, 124, prior_3mm) == 124, "prior_from names the version whose maps are read"
        # a standing prior_from (set by `segmenter.py prior`) is read in place of the model's own prior, under the same checks
        assert reading_prior(prior_own) == 116, "without a standing prior_from the model's own prior is read"
        assert reading_prior(dict(prior_own, reading={"prior_from": 124}), prior_3mm) == 124, "a standing prior_from is read"
        for own, named in ((dict(prior_own, reading={"prior_from": 126}), prior_1_5),
                           (dict(prior_none, reading={"prior_from": 124}), prior_3mm)):
            try:
                reading_prior(own, named)
                raise AssertionError("a standing prior_from on version %d must be refused" % own["version"])
            except SystemExit:
                pass
        print("prior_from ok")
        # the test split's coarse maps: all five nets read together, written only inside an open second look whose
        # version takes this one as its prior, and never without one
        was_test = prepare.OUT, prepare.lo_files, prepare.FINAL_STATE, prepare.ledger.records
        tlo = os.path.join(root, "test-lo")
        os.makedirs(tlo, exist_ok=True)
        for scan in ("t0", "t1"):
            np.save(os.path.join(tlo, scan + "_ct.npy"),
                    np.random.default_rng(6).integers(-1000, 1000, (20, 18, 12)).astype(np.int16))
        os.makedirs(os.path.join(root, "v10"), exist_ok=True)
        os.makedirs(os.path.join(root, "v11"), exist_ok=True)
        for k in range(FOLDS):
            torch.save(build(cfg, 3).state_dict(), os.path.join(root, "v10", "fold%d.pt" % k))
        with open(os.path.join(root, "v10", "meta.json"), "w") as f:
            json.dump({"cfg": cfg, "structures": ["a", "b"]}, f)
        with open(os.path.join(root, "v11", "meta.json"), "w") as f:
            json.dump({"cfg": {**cfg, "prior": {"version": 10, "dim": 4}}, "structures": ["a", "b"]}, f)
        latches, state_file = [], os.path.join(root, "final_eval.json")

        def state_with(entry):
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({"looks": [entry]}, f)

        open_look = {"look": 2, "version": 11, "with_version": None, "finished": None}
        prepare.OUT = os.path.join(root, "out-test")
        prepare.lo_files = lambda scan: (os.path.join(tlo, scan + "_ct.npy"), None)
        prepare.FINAL_STATE = state_file
        prepare.ledger.records = lambda kind=None: [{"name": n} for n in latches]
        try:
            state_with(open_look)
            for look, latch in ((None, []), (2, [])):      # no look at all, and a look the ledger has not latched
                latches[:] = latch
                try:
                    write_test_coarse(10, look, ["t0", "t1"], set())
                    raise AssertionError("the test's coarse maps need an open, latched look")
                except SystemExit:
                    pass
            assert not os.path.exists(coarse_file(10, "t0")), "a refused test read writes nothing"
            state_with({**open_look, "finished": "t"})
            latches[:] = ["ct test opened second look v11"]
            try:
                write_test_coarse(10, 2, ["t0", "t1"], set())
                raise AssertionError("a finished look's test coarse maps are refused")
            except SystemExit:
                pass
            state_with(open_look)
            assert write_test_coarse(10, 2, ["t0", "t1"], set()) == 2, "both test scans are written"
            assert write_test_coarse(10, 2, ["t0", "t1"], set()) == 0, "a rerun with the same five nets writes nothing"
            models10, _ = load_models(10, range(FOLDS))
            acc, _ = probabilities(models10, np.ascontiguousarray(np.load(os.path.join(tlo, "t0_ct.npy"))),
                                   cfg["patch"], cfg.get("overlap", READING["overlap"]), cfg.get("trim", READING["trim"]))
            want = coarse_labels(acc).to(torch.uint8).cpu().numpy()
            assert np.array_equal(np.load(coarse_file(10, "t0")), want), "the map is the five nets' average, read as one"
            man = json.load(open(os.path.join(coarse_dir(10), "manifest.json"), encoding="utf-8"))
            nets = [sha256(os.path.join(root, "v10", "fold%d.pt" % k)) for k in range(FOLDS)]
            assert man["scans"]["t0"]["split"] == "test" and man["scans"]["t0"]["nets_sha256"] == nets
        finally:
            prepare.OUT, prepare.lo_files, prepare.FINAL_STATE, prepare.ledger.records = was_test
        print("test coarse ok")
        # the card's fallback: out of memory on the first slab reruns the scan in tiles, with the same labels
        fg = torch.Generator().manual_seed(11)
        fw = torch.rand(20, 18, 11, generator=fg) + 0.5
        fa = (torch.softmax(torch.randn(12, 20, 18, 11, generator=fg), dim=0) * fw).half()
        fshape = tuple(s * prepare.FACTOR for s in (20, 18, 11))
        want_labels = labels_on_grid(fa, fw, fshape, tile=0)
        real_interpolate, calls = F.interpolate, []

        def out_of_memory_once(*args, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("CUDA out of memory (selftest)")
            return real_interpolate(*args, **kw)

        F.interpolate = out_of_memory_once
        try:
            got_labels = _labels_or_tiles(fa, fw, fshape, tile=8)
        finally:
            F.interpolate = real_interpolate
        assert len(calls) > 1, "the first slab ran out of memory and the tiles ran"
        assert torch.equal(got_labels, want_labels), "the tiled fallback gives the untiled labels, element for element"
        print("label fallback identical")
        # (vi) the space-to-depth net (idea 127): from a 3 mm net it starts as that net on the 2 x 2 x 2 mean of its 1.5 mm
        # cube; its deep heads come back as the 3 mm net's do; and a 1.5 mm read through it is full size
        small = {"widths": [4, 4, 4], "deep": [1.0, 0.5, 0.25], "layout": "plain"}
        src3 = build(small, 3).eval()
        s2d_net = build({**small, "net": "s2d", "prior": {"version": 1, "dim": 4}}, 3).eval()
        s2d_net.load_state_dict(s2d_from_3mm(src3.state_dict(), s2d_net))
        hu = torch.as_tensor(np.random.default_rng(12).integers(-900, 1400, (1, 1, 16, 16, 16)), dtype=torch.float32,
                             device=DEV)
        mean = hu.reshape(1, 1, 8, 2, 8, 2, 8, 2).mean(dim=(3, 5, 7))
        with torch.no_grad():
            fine_lab = s2d_net(to_input(hu.clone()))[0].argmax(dim=1)[0]
            coarse_lab = src3(to_input(mean))[0].argmax(dim=1)[0]
        want_lab = coarse_lab.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
        agree = (fine_lab == want_lab).float().mean().item()
        assert agree >= 0.99, "the space-to-depth net matches its 3 mm net on the averaged cube on %.4f of voxels" % agree
        coarse_map = np.random.default_rng(13).integers(0, 3, (10, 9, 6)).astype(np.uint8)
        vol = np.random.default_rng(14).integers(-1000, 1000, (20, 18, 12)).astype(np.int16)
        full_read = read_fine([s2d_net], vol, 16, rd, ["a", "b"], coarse=coarse_map)
        assert tuple(full_read.shape) == vol.shape, "a 1.5 mm read through the space-to-depth net is full size"
        odd = np.random.default_rng(15).integers(-1000, 1000, (20, 37, 19)).astype(np.int16)
        odd_read = read_fine([s2d_net], odd, 16, rd, ["a", "b"], coarse=np.zeros((10, 19, 10), dtype=np.uint8))
        assert tuple(odd_read.shape) == odd.shape, "an odd-shaped volume reads back at its own shape"
        s2d_net.train()
        outs = s2d_net(to_input(hu.clone()), torch.zeros(1, 8, 8, 8, dtype=torch.long, device=DEV))
        assert [tuple(o.shape[2:]) for o in outs] == [(16, 16, 16), (8, 8, 8), (4, 4, 4)], "full, 3 mm and 6 mm outputs"
        deep_loss = loss_of(outs, torch.zeros(1, 16, 16, 16, dtype=torch.long, device=DEV), small)
        assert torch.isfinite(deep_loss).item(), "the deep-supervision loss is finite"
        print("space-to-depth net ok")
        print("segmenter selftest OK")
    finally:
        globals().update(was)
        datasets.structures, prepare.unprepared = was_ds, was_prep
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    b = sub.add_parser("bench")
    b.add_argument("--profile", type=int, help="then profile this many more steps: the card's kernels by kind")
    b.add_argument("--real-crops", action="store_true", dest="real_crops",
                   help="time the step on the real 1.5 mm crops of fold 0's training scans, not random cubes")
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int)
    t.add_argument("--steps", type=int)
    t.add_argument("--force", action="store_true")
    t.add_argument("--kill-ref", dest="kill_ref", type=int,
                   help="the one-hour rule: stop each fold at --kill-step when its learning curve is behind this "
                        "version's (its finished curve when the recipe has init_from, else at the same step)")
    t.add_argument("--kill-step", dest="kill_step", type=int, help="the step to judge at, about an hour in")
    t.add_argument("--kill-margin", dest="kill_margin", type=float, default=0.0,
                   help="how far under the reference the curve may be and still go on (Dice, default 0)")
    for p in (b, t):
        p.add_argument("--patch", type=int)
        p.add_argument("--batch", type=int)
        p.add_argument("--set", action="append", help="key=<json>, a setting to try")
    q = sub.add_parser("predict")
    q.add_argument("split", choices=["dev", "test"])
    q.add_argument("--limit", type=int, help="score only this many scans of each fold; writes nothing")
    q.add_argument("--fold", type=int, help="development only: just this fold's held-out scans")
    q.add_argument("--force", action="store_true")
    q.add_argument("--set", action="append", help="key=<json>, a reading setting to try (overlap, keep_largest)")
    q.add_argument("--out-dir", dest="out_dir", help="write pred_segmenter_<split>.csv here instead of the "
                                                     "eval folder (the second look uses it); nothing else moves")
    q.add_argument("--tag", help="names the trial's out-of-fold files, so the scored ones are never overwritten")
    q.add_argument("--with-version", dest="with_version", type=int,
                   help="needs --tag on development: also read this version's net of the same fold, averaged "
                        "with the version's own. On the test split, only with --look (final_eval's second look)")
    q.add_argument("--look", type=int, help="test split with --with-version only: the second look final_eval opened")
    s = sub.add_parser("scan")
    s.add_argument("file")
    s.add_argument("--out")
    fl = sub.add_parser("floors")
    fl.add_argument("--q", type=float, required=True)
    pr = sub.add_parser("prior")
    pr.add_argument("--from", dest="prior_from", type=int, required=True, help="a 3 mm version whose coarse maps to read")
    pc = sub.add_parser("pieces")
    co = sub.add_parser("coarse")
    co.add_argument("--version", type=int, required=True, help="a 3 mm version whose maps the prior reads")
    co.add_argument("--folds", type=int, nargs="+", default=list(range(FOLDS)),
                    help="the folds whose held-out scans are read (default all five)")
    co.add_argument("--split", choices=["dev", "test"], default="dev")
    co.add_argument("--look", type=int, help="with --split test: the second look final_eval opened")
    for p in (t, q, s, fl, pr, pc):
        p.add_argument("--version", type=int, default=1)
    a = ap.parse_args()
    {"selftest": selftest, "bench": bench, "train": train, "predict": predict, "scan": read_scan, "floors": set_floors,
     "prior": set_prior, "pieces": set_pieces, "coarse": coarse_maps}[a.cmd](a)


if __name__ == "__main__":
    main()
