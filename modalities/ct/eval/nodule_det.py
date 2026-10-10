"""The lung nodule finder as one pass of an anchor-free 3D detector (idea 126): a centre heatmap, a diameter and a
sub-voxel offset at every cell, all from one forward pass.

What it finds. As nodule.py: a candidate is a point in a scan (x, y, z in millimetres) with a probability that a
nodule is there, up to 100 a scan, and froc.py calls a nodule found when a candidate lies closer to its centre than
its radius.

How it differs from nodule.py. nodule.py colours in a ball and reads the height of the colouring's peak. Here the
network is taught a centre heatmap: a Gaussian bump on the cell that holds each nodule's centre, a cell being 2 x 2 x 2
voxels (2 mm), trained with CenterNet's penalty-reduced focal loss, so the score itself learns to tell a nodule's
centre from a vessel. The same pass gives the diameter (L1) and where the centre sits inside its cell (L1), so there
are no anchors and no second stage. Findings LUNA16 excludes get weight 0 in the heatmap loss, as in nodule.py.

The network. A 2x2x2 stride-2 stem (1 to 32 channels, at 2 mm), encoder levels at 2, 4, 8 and 16 mm (widths 32, 64,
128, 256), and a light feature pyramid (FPN) carried back down to the 2 mm grid, where the three heads sit.

Augmentation runs on the card: one affine_grid and one grid_sample per batch carry the scale (0.8 to 1.25), rotations
of up to 15 degrees about each axis, a random rot90 in the axial plane and flips; gamma follows. The nodule centres and
radii go through the same warp, so the targets follow the picture.

Reading. A sliding window of 96 mm at half overlap (corners kept even, so the 2 mm cells line up across windows), the
heatmap logits blended with a bell as nodule.py blends its logits, a 3x3x3 maximum on the cell grid for the peaks, and
the top 100 a scan. Candidates are written in nodule.py's candidate format (candidates_oof.csv in the version's
folder), probability = the sigmoid of the blended logit. That is a heat score, not yet a fitted (Platt) probability:
it ranks the same as the logit, which is all froc.py and the re-scorer's logit(p1) need, and the raw logit is in
fold{k}_diameters.csv beside each candidate's diameter. Each place the blob finder found (nodule_blobs.py's
candidates file, when it exists for the scan) gets the heatmap value at its nearest cell, in blob_heat_oof.csv.

    python modalities/ct/eval/nodule_det.py train --version N [--folds 0 1] [--steps S] [--set key=value]
    python modalities/ct/eval/nodule_det.py read --version N [--folds 0 1] [--set key=value]
    python modalities/ct/eval/nodule_det.py heat --version N --places CSV --out CSV [--folds K ...] [--set key=value]
    python modalities/ct/eval/nodule_det.py bench [--set key=value] [--nodule-v2]
    python modalities/ct/eval/nodule_det.py selftest

Folds are nodule.py's five (LUNA16's ten subsets paired). `train` writes fold{k}.pt in the version's folder, with a
partial checkpoint fold{k}.partial.pt every save_every steps, resumed when its fingerprint matches. Every batch is drawn
from the seed and its batch index alone, and the augmentation from the seed and the step, so a resumed run takes the
same cubes and the same warps an uninterrupted one does. `read` reads the held-out scans of the folds it is given and
scores them with froc.py. Env RAIDIOLOGY_DET_CARD_GB (default 5.0) is the share of the card it leases.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import froc  # noqa: E402
import lung_prepare as lp  # noqa: E402
import nodule  # noqa: E402
from core import card, ledger, paths, polite  # noqa: E402

DEV = nodule.DEV
CFG = {
    "widths": [32, 64, 128, 256], "fpn": 32, "heat_bias": -2.19,
    "steps": 16000, "batch": 8, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 300, "clip": 10.0,
    "nodule_share": 0.5, "shift_mm": 40.0, "max_balls": 8,
    "scale": [0.8, 1.25], "rotate_deg": 15.0, "gamma": [0.8, 1.25],
    "sigma_min": 0.5, "size_weight": 0.1, "offset_weight": 1.0,
    "seed": 0, "log_every": 200, "save_every": 2000, "workers": 8, "prefetch": 3,   # the last three: not in the fingerprint
    # reading
    "overlap": 0.5, "top": 100, "read_batch": 2, "pool": 3,
}
READ_KEYS = ("overlap", "top", "read_batch", "pool")
NOT_TRAINING = (*READ_KEYS, "save_every", "workers", "prefetch", "log_every")   # settings that are not training's
SYNC_EVERY = 4
NODULE_COLUMNS = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "fold"]   # nodule.py's candidates_oof.csv


def out_root(version):
    return os.path.join(paths.WORK, "eval", "lung", "det-v%d" % version)


def card_gb():
    return float(os.environ.get("RAIDIOLOGY_DET_CARD_GB", "5.0"))


_LIMITED = []


def limit_card():
    """Lease this process's share of the card, then a hard ceiling at it (see core/card.py, as nodule.py does)."""
    if DEV.type != "cuda" or _LIMITED:
        return
    gb = card_gb()
    card.lease(gb, "nodule-det")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())
    _LIMITED.append(gb)


class Box:
    """A stand-in for nodule.Scan for the selftest: one array on disk and its balls in voxel units (z, y, x, radius)."""

    def __init__(self, uid, path, shape, nodules):
        self.uid, self.path, self.shape = uid, path, tuple(shape)
        self.nodules = np.asarray(nodules, float).reshape(-1, 4)
        self.whole, self.excluded = self.nodules, np.zeros((0, 4))
        self.meta = {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [1.0, 1.0, 1.0]}


def ball_list(scans):
    return [(s, i) for s in scans for i in range(len(s.nodules))]


def draw_cube(scans, balls, cfg, rng):
    """(scan, corner) of one cube: round a nodule with probability nodule_share, else any cube of a random scan."""
    n = nodule.CUBE
    if balls and rng.random() < cfg["nodule_share"]:
        scan, i = balls[int(rng.integers(len(balls)))]
        centre = scan.nodules[i, :3] + rng.uniform(-cfg["shift_mm"], cfg["shift_mm"], 3)
        start = np.rint(centre).astype(int) - n // 2
    else:
        scan = scans[int(rng.integers(len(scans)))]
        start = np.array([rng.integers(0, max(m - n, 0) + 1) for m in scan.shape])
    start = [int(np.clip(s, 0, m - n)) if m >= n else 0 for s, m in zip(start, scan.shape)]
    return scan, start


def make_batch(scans, balls, cfg, rng, batch):
    """A batch of cubes as (image uint8 (B, N, N, N), ignore uint8 (1 where LUNA16 excludes the voxel's finding),
    balls float32 (B, max_balls, 4) with (z, y, x, radius) of each nodule whose centre is inside its cube or within
    twice its radius (at least 6 voxels) outside it, radius 0 where unused). Each cube is nodule.cube_of's, so the boxes, the paint and the excluded balls are nodule.py's."""
    n = nodule.CUBE
    img = np.zeros((batch, n, n, n), np.uint8)
    ign = np.zeros_like(img)
    out = np.zeros((batch, cfg["max_balls"], 4), np.float32)
    for b in range(batch):
        scan, start = draw_cube(scans, balls, cfg, rng)
        im, _, w = nodule.cube_of(scan, start)
        img[b], ign[b] = im, w == 0
        centres = scan.nodules[:, :3] - np.asarray(start, float)
        reach = 2 * np.maximum(scan.nodules[:, 3], 3.0)[:, None]      # a centre just outside still shapes the bump
        inside = np.where(np.all((centres >= -reach) & (centres < n + reach), axis=1))[0][:cfg["max_balls"]]
        for j, i in enumerate(inside):
            out[b, j] = [*centres[i], scan.nodules[i, 3]]
    return img, ign, out


class CubeStream(IterableDataset):
    """Endless batches for one worker process. Batch j is drawn from [seed, 0, j] alone, so the cubes do not depend on
    the number of workers or on where a resumed run starts: worker k builds batches start + k, start + k + N, ..."""

    def __init__(self, scans, cfg, seed, start, cube):
        self.scans, self.cfg, self.seed, self.start, self.cube = scans, cfg, seed, start, cube
        self.balls = ball_list(scans)

    def __iter__(self):
        info = get_worker_info()
        k, n = (info.id, info.num_workers) if info else (0, 1)
        nodule.CUBE = self.cube                       # a worker is a fresh process: it takes the cube size from here
        j = self.start + k
        while True:
            rng = np.random.default_rng([self.seed, 0, j])
            yield make_batch(self.scans, self.balls, self.cfg, rng, self.cfg["batch"])
            j += n


def _as_tensors(batch):
    """A batch's arrays as torch tensors: DataLoader hands tensors between processes through shared memory."""
    return tuple(torch.from_numpy(a) for a in batch)


class Batches:
    """The batches of CubeStream, built by cfg["workers"] worker processes (0 builds them in this one)."""

    def __init__(self, scans, cfg, seed, start=0):
        kw = {"prefetch_factor": cfg["prefetch"]} if cfg["workers"] else {}
        self.loader = DataLoader(CubeStream(scans, cfg, seed, start, nodule.CUBE), batch_size=None,
                                 num_workers=cfg["workers"], collate_fn=_as_tensors,
                                 persistent_workers=cfg["workers"] > 0, pin_memory=DEV.type == "cuda", **kw)
        self.it = iter(self.loader)

    def get(self):
        return next(self.it)

    def close(self):
        if self.it is not None and hasattr(self.it, "_shutdown_workers"):
            self.it._shutdown_workers()
        self.it = None


class Feed:
    """Batches already on the card: the copy of the next one runs on a side stream while this one trains."""

    def __init__(self, batches):
        self.batches = batches
        self.side = torch.cuda.Stream() if DEV.type == "cuda" else None
        self.ahead = self._fetch()

    def _fetch(self):
        cpu = self.batches.get()
        if self.side is None:
            return tuple(t.to(DEV) for t in cpu), None
        with torch.cuda.stream(self.side):
            return tuple(t.to(DEV, non_blocking=True) for t in cpu), cpu      # `cpu` is kept alive until the next call

    def next(self):
        gpu, _ = self.ahead
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)
            for t in gpu:
                t.record_stream(torch.cuda.current_stream())
        self.ahead = self._fetch()
        return gpu


def rotation(axis, theta):
    """The rotation by theta about one of the three axes of (z, y, x)."""
    i, j = [k for k in range(3) if k != axis]
    m = np.eye(3)
    c, s = math.cos(theta), math.sin(theta)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def draw_warp(rng, cfg, n):
    """Per cube: Q (an orthogonal matrix: a small rotation about each axis, then a rot90 in the axial plane, then
    flips), S (the scale) and G (the gamma). A cube's point c (voxels from its centre) goes to S * Q c."""
    Q, S, G = np.zeros((n, 3, 3)), np.zeros(n), np.zeros(n)
    quarter = np.rint(rotation(0, math.pi / 2))
    for b in range(n):
        S[b] = rng.uniform(*cfg["scale"])
        r = np.eye(3)
        for axis in range(3):
            r = r @ rotation(axis, math.radians(rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"])))
        p = np.linalg.matrix_power(quarter, int(rng.integers(4)))
        flips = np.diag(np.where(rng.random(3) < 0.5, -1.0, 1.0))
        Q[b] = flips @ p @ r
        G[b] = rng.uniform(*cfg["gamma"])
    return Q, S, G


def warp_batch(img, ign, Q, S, G):
    """One affine_grid and grid_sample per batch: the image (trilinear, zeros outside, as the padding of a cube) and the
    ignore map (nearest; zeros outside, so the padding is counted, as nodule.py's padding is). The sampling grid is
    A = Q^T / S in (z, y, x), turned to (x, y, z) for affine_grid; the gamma then clips to [0, 1] and raises."""
    B, n = img.shape[0], img.shape[-1]
    A = Q.transpose(1, 2) / S.view(-1, 1, 1)
    A = A.flip(1).flip(2)
    theta = torch.cat([A, torch.zeros(B, 3, 1, device=DEV)], 2)
    grid = F.affine_grid(theta, (B, 1, n, n, n), align_corners=False)
    x = F.grid_sample(img.float().unsqueeze(1) / 255, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    ig = F.grid_sample(ign.float().unsqueeze(1), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return x.clamp(0, 1) ** G.view(B, 1, 1, 1, 1), ig


def warp_balls(balls, Q, S):
    """Where the balls go under the same warp: centres in cube voxels (B, M, 3), radii (B, M) and which are valid (a
    radius above 0 and a centre inside the cube after the warp or within its radius, at least 3 voxels, outside it: the
    tail of its bump, about 3 sigma, still reaches in). The cube's centre is the origin of the warp."""
    n = nodule.CUBE
    c = balls[..., :3]
    u = 2 * (c + 0.5) / n - 1
    uo = torch.einsum("bij,bmj->bmi", S.view(-1, 1, 1) * Q, u)
    co = (uo + 1) * n / 2 - 0.5
    ro = balls[..., 3] * S.view(-1, 1)
    margin = ro.clamp_min(3.0)[..., None]
    inside = ((co >= -margin) & (co < n + margin)).all(-1)
    return co, ro, (balls[..., 3] > 0) & inside


def conv(c_in, c_out, k=3, s=1):
    return nn.Sequential(nn.Conv3d(c_in, c_out, k, s, padding=(k - 1) // 2, bias=False),
                         nn.BatchNorm3d(c_out), nn.ReLU(inplace=True))


class Det(nn.Module):
    """The one-pass detector: a stride-2 stem, encoder levels at 2, 4, 8 and 16 mm, a light FPN back to the 2 mm grid,
    and the heatmap, diameter and offset heads there (the heatmap bias starts at a prior of about 0.1)."""

    def __init__(self, widths, fpn, heat_bias):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.stem = conv(1, w1, 2, 2)
        self.e1 = conv(w1, w1)
        self.d2 = conv(w1, w2, 2, 2)
        self.e2 = conv(w2, w2)
        self.d3 = conv(w2, w3, 2, 2)
        self.e3 = conv(w3, w3)
        self.d4 = conv(w3, w4, 2, 2)
        self.e4 = conv(w4, w4)
        self.lat = nn.ModuleList([nn.Conv3d(w, fpn, 1) for w in widths])
        self.smooth = conv(fpn, fpn)
        self.heat = nn.Conv3d(fpn, 1, 1)
        self.size = nn.Conv3d(fpn, 1, 1)
        self.offset = nn.Conv3d(fpn, 3, 1)
        nn.init.constant_(self.heat.bias, heat_bias)

    def forward(self, x):
        c1 = self.e1(self.stem(x))
        c2 = self.e2(self.d2(c1))
        c3 = self.e3(self.d3(c2))
        c4 = self.e4(self.d4(c3))
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = lat(c) + F.interpolate(p, scale_factor=2, mode="nearest")
        p = self.smooth(p)
        with torch.autocast(p.device.type, enabled=False):      # the heads in float32, as nodule.py's float32_heads
            p = p.float()
            return self.heat(p), self.size(p), self.offset(p)


def build(cfg):
    return Det(cfg["widths"], cfg["fpn"], cfg["heat_bias"]).to(DEV)


def targets(co, ro, valid, cfg, g):
    """The Gaussian heatmap (the highest of the balls' bumps), the positive cells (each valid ball's centre cell) and
    each ball's cell. A ball's cell is the nearest 2 mm cell to its centre; its bump has sigma radius/3 cells (at least
    sigma_min), so a small nodule is a narrow bump and a large one a wide one."""
    cell = torch.round((co - 0.5) / 2)                      # may lie outside the grid: only the bump's tail is in
    sigma = (ro / 2 / 3).clamp_min(cfg["sigma_min"])[..., None, None, None]
    grid = torch.arange(g, device=co.device, dtype=torch.float32)
    cz, cy, cx = [cell[..., a][..., None, None, None] for a in range(3)]
    d2 = (grid.view(1, 1, g, 1, 1) - cz) ** 2 + (grid.view(1, 1, 1, g, 1) - cy) ** 2 + (grid.view(1, 1, 1, 1, g) - cx) ** 2
    v = valid[..., None, None, None]
    heat = (torch.exp(-d2 / (2 * sigma ** 2)) * v).amax(1)
    pos = ((d2 == 0) & v).any(1)
    return heat, pos, cell


def heat_loss(logit, heat, pos, wh):
    """CenterNet's penalty-reduced focal loss, over the cells whose weight wh is 1, by the number of positive cells
    that have weight (a positive on an excluded finding's ball is not counted)."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos_f = (pos & (wh > 0)).float()
    pos_term = (1 - p) ** 2 * torch.log(p) * pos_f
    neg_term = (1 - heat) ** 4 * p ** 2 * torch.log(1 - p) * (1 - pos_f)
    return -((pos_term + neg_term) * wh).sum() / pos_f.sum().clamp_min(1.0)


def regression(size, off, co, ro, valid, cell, cfg):
    """L1 on the diameter (in units of 10 mm) and on the offset of the centre from its cell's centre (in voxels), at
    each valid ball's cell (balls whose cell lies outside the grid have no cell to read)."""
    g = size.shape[-1]
    bi, mi = (valid & ((cell >= 0) & (cell < g)).all(-1)).nonzero(as_tuple=True)
    nv = max(len(bi), 1)
    c = cell[bi, mi].long()
    sz = size[bi, c[:, 0], c[:, 1], c[:, 2]]
    of = off[bi, :, c[:, 0], c[:, 1], c[:, 2]]
    l_size = (sz - 2 * ro[bi, mi] / 10).abs().sum() / nv
    l_off = (of - (co[bi, mi] - (2 * c.float() + 0.5))).abs().sum() / nv
    return cfg["size_weight"] * l_size + cfg["offset_weight"] * l_off


def loss_of_batch(net, batch, cfg, rng):
    """One batch's loss: draw the warp, warp the cubes and the balls, run the net, take the targets off the warped balls."""
    img, ign, balls = batch
    Q, S, G = [torch.as_tensor(v, dtype=torch.float32, device=DEV) for v in draw_warp(rng, cfg, img.shape[0])]
    x, ig = warp_batch(img, ign, Q, S, G)
    co, ro, valid = warp_balls(balls, Q, S)
    with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
        heat, size, off = net(x)
    heat, size, off = heat.float()[:, 0], size.float()[:, 0], off.float()
    g = heat.shape[-1]
    wh = 1 - F.max_pool3d(ig, 2)[:, 0]
    heat_t, pos, cell = targets(co, ro, valid, cfg, g)
    return heat_loss(heat, heat_t, pos, wh) + regression(size, off, co, ro, valid, cell, cfg)


def train_net(scans, cfg, seed, tag, partial=None, fingerprint=None, stop_at=None):
    """Train a fresh net on `scans`; returns (net, seconds, peak card GB, seconds per step).

    With `partial` (a path) and `fingerprint`, a partial checkpoint is saved there every save_every steps and at
    `stop_at`, and a partial with the same fingerprint is continued from its next step. `stop_at` ends the run early
    (the selftest's stop-and-resume)."""
    torch.manual_seed(seed)
    net = build(cfg).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            fused=DEV.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: nodule.lr_factor(s, cfg))
    start, prior_secs, prior_peak = 0, 0.0, 0.0
    ck = nodule.load_resumable(partial, fingerprint) if partial else None
    if ck is not None and 0 < ck.get("step", 0) < cfg["steps"]:
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, prior_secs, prior_peak = ck["step"], ck["seconds"], ck.get("peak_gb", 0.0)
        print("%s: resumed mid-fold at step %d of %d (partial checkpoint)" % (tag, start, cfg["steps"]), flush=True)
    del ck
    batches = Batches(scans, cfg, seed, start)
    feed = Feed(batches)
    pace = polite.Pace()
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    t0 = block = log_t = time.time()
    t0 -= prior_secs                                       # seconds trained add up across a resume
    running, seen, step = 0.0, 0, start
    try:
        for step in range(start + 1, cfg["steps"] + 1):
            loss = loss_of_batch(net, feed.next(), cfg, np.random.default_rng([seed, 1, step]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
            opt.step()
            sched.step()
            running = running + loss.detach()
            seen += 1
            if step % SYNC_EVERY == 0 or step == cfg["steps"]:
                running = float(running)                   # the wait for the card
                pace.rest(time.time() - block)
                block = time.time()
            if step % cfg["log_every"] == 0 or step == cfg["steps"]:
                running = float(running)
                now = time.time()
                print("  %s step %d of %d, loss %.4f, %.1f s per 100 steps, using %.0f%% of the card"
                      % (tag, step, cfg["steps"], running / seen, 100 * (now - log_t) / seen, 100 * pace.used()),
                      flush=True)
                log_t, running, seen = now, 0.0, 0
            if partial and (step % cfg["save_every"] == 0 or step == stop_at) and step < cfg["steps"]:
                peak_now = max(prior_peak, torch.cuda.max_memory_allocated() / 2 ** 30) if DEV.type == "cuda" else float("nan")
                torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "seconds": time.time() - t0, "peak_gb": peak_now,
                            "fingerprint": fingerprint}, partial + ".tmp")
                os.replace(partial + ".tmp", partial)
            if step == stop_at:
                break
    finally:
        batches.close()
    sync()
    seconds = time.time() - t0
    peak = max(prior_peak, torch.cuda.max_memory_allocated() / 2 ** 30) if DEV.type == "cuda" else float("nan")
    done = max(step - start, 1)
    print("  %s: %d steps in %.0f s, peak card %.2f GB" % (tag, step - start, seconds, peak), flush=True)
    return net, seconds, peak, seconds / done


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def even_starts(full, n, overlap):
    """Window corners along one axis: half overlap, the last one flush with the end, every corner even (so the
    2 mm cells of overlapping windows line up)."""
    step = max(2, int(n * (1 - overlap)) // 2 * 2)
    last = full - n
    return sorted({s - s % 2 for s in [*range(0, last, step), last]})


def cell_bell(g, device):
    """A separable tent over a window's cells: a window is trusted most in its middle."""
    w = 1 - (2 * (torch.arange(g, device=device, dtype=torch.float32) + 0.5) / g - 1).abs() + 0.05
    return w.view(g, 1, 1) * w.view(1, g, 1) * w.view(1, 1, g)


def cells_of(vox, shape):
    """The cell (z, y, x) each voxel position falls in: cell i holds voxels 2i and 2i+1 (centre 2i + 0.5), and a position
    off the grid clips to the nearest edge cell."""
    hi = np.array(shape) - 1
    return np.clip(np.rint((vox - 0.5) / 2), 0, hi).astype(np.int64)


@torch.inference_mode()
def heat_values(logit, vox):
    """The raw logit at each voxel position's cell, and the largest raw logit in the 3x3x3 cells around it (padding is
    -inf, so only cells in the grid count). Both as float64 arrays."""
    idx = torch.as_tensor(cells_of(vox, logit.shape), device=logit.device)
    at = logit[idx[:, 0], idx[:, 1], idx[:, 2]].double()
    pooled = F.max_pool3d(logit[None, None], 3, 1, 1)[0, 0]
    return at.cpu().numpy(), pooled[idx[:, 0], idx[:, 1], idx[:, 2]].double().cpu().numpy()


def blob_heat(logit, scan, rows):
    """The heatmap's probability at each blob finder place (the nearest cell), as a frame with the blob's coordinates."""
    vox = lp.world_to_voxel(scan.meta, rows[["coordX", "coordY", "coordZ"]].to_numpy(float))
    idx = torch.as_tensor(cells_of(vox, logit.shape), device=logit.device)
    val = torch.sigmoid(logit[idx[:, 0], idx[:, 1], idx[:, 2]].double()).cpu().numpy()
    return pd.DataFrame({"seriesuid": scan.uid, "coordX": rows.coordX.to_numpy(), "coordY": rows.coordY.to_numpy(),
                         "coordZ": rows.coordZ.to_numpy(), "heat": val})


def place_heat(logit, scan, rows):
    """The raw heatmap logits at given places (rows of a places frame with its `row`): heat_at at the place's own cell
    and heat_max3, the largest logit in the 3x3x3 cells around it."""
    vox = lp.world_to_voxel(scan.meta, rows[["coordX", "coordY", "coordZ"]].to_numpy(float))
    at, mx = heat_values(logit, vox)
    return pd.DataFrame({"row": rows.row.to_numpy(), "seriesuid": rows.seriesuid.to_numpy(),
                         "coordX": rows.coordX.to_numpy(), "coordY": rows.coordY.to_numpy(),
                         "coordZ": rows.coordZ.to_numpy(), "heat_at": at, "heat_max3": mx})


@torch.inference_mode()
def heat_logit(net, scan, cfg):
    """One scan's blended heatmap on the cell grid: the 5 channels (heatmap logit, diameter, offset z, y, x), each
    cell the bell-weighted mean of the windows that cover it, cropped to the cells inside the scan's box. The logit is
    mean[0]. The windows are run as one batch of read_batch each."""
    net.eval()
    n = nodule.CUBE
    g = n // 2
    vol = np.load(scan.path)
    shape = tuple(vol.shape)
    full = [max(s + s % 2, n) for s in shape]
    x = torch.zeros(full, dtype=torch.uint8, device=DEV)
    x[:shape[0], :shape[1], :shape[2]] = torch.from_numpy(vol).to(DEV)
    corners = [(a, b, c) for a in even_starts(full[0], n, cfg["overlap"])
               for b in even_starts(full[1], n, cfg["overlap"]) for c in even_starts(full[2], n, cfg["overlap"])]
    cells = [f // 2 for f in full]
    acc = torch.zeros((5, *cells), device=DEV)              # heatmap logit, diameter, offset z, y, x
    wsum = torch.zeros(cells, device=DEV)
    bell = cell_bell(g, DEV)
    for i in range(0, len(corners), cfg["read_batch"]):
        group = corners[i:i + cfg["read_batch"]]
        win = torch.stack([x[a:a + n, b:b + n, c:c + n] for a, b, c in group])[:, None].float() / 255.0
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            h, s, o = net(win)
        out = torch.cat([h.float(), s.float(), o.float()], 1)
        for (a, b, c), zw in zip(group, out):
            sl = (slice(a // 2, a // 2 + g), slice(b // 2, b // 2 + g), slice(c // 2, c // 2 + g))
            acc[(slice(None), *sl)] += zw * bell
            wsum[sl] += bell
    cz, cy, cx = [math.ceil(s / 2) for s in shape]          # the cells whose centre is inside the scan's box
    return (acc / wsum.clamp_min(1e-6))[:, :cz, :cy, :cx]


@torch.inference_mode()
def read_scan(net, scan, cfg, blob_rows=None):
    """One scan's candidates (seriesuid, coordX/Y/Z, probability), its diameters (the same rows, diameter_mm), its
    blob-place heat (or None) and the seconds taken. The windows are run as one batch of read_batch each."""
    t0 = time.time()
    mean = heat_logit(net, scan, cfg)
    logit = mean[0]
    k = cfg["pool"]
    peak = F.max_pool3d(logit[None, None], k, 1, k // 2)[0, 0]
    hit = (logit == peak).nonzero()
    score = logit[hit[:, 0], hit[:, 1], hit[:, 2]]
    keep = score.argsort(descending=True)[:cfg["top"]]
    idx = hit[keep]
    prob = torch.sigmoid(score[keep].double()).clamp(1e-12, 1 - 1e-12).cpu().numpy()
    off = mean[2:][:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    centre = (2 * idx.double() + 0.5 + off.double()).cpu().numpy()
    dia = (mean[1][idx[:, 0], idx[:, 1], idx[:, 2]] * 10).double().cpu().numpy()
    xyz = lp.voxel_to_world(scan.meta, centre)
    frame = pd.DataFrame({"seriesuid": scan.uid, "coordX": xyz[:, 0], "coordY": xyz[:, 1], "coordZ": xyz[:, 2],
                          "probability": prob})
    diam = frame.assign(diameter_mm=dia, logit=score[keep].double().cpu().numpy()).drop(columns="probability")
    blobs = blob_heat(logit, scan, blob_rows) if blob_rows is not None and len(blob_rows) else None
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return frame, diam, blobs, time.time() - t0


def load_blobs():
    """{seriesuid: the blob finder's rows} from nodule_blobs.py's default candidates file, or {} when it is not there."""
    path = os.path.join(paths.WORK, "eval", "lung", "blobs", "candidates_oof.csv")
    if not os.path.exists(path):
        print("no blob candidates file at %s: no blob heat written" % path, flush=True)
        return {}
    b = pd.read_csv(path, dtype={"seriesuid": str})
    print("blob candidates: %d rows over %d scans" % (len(b), b.seriesuid.nunique()), flush=True)
    return {uid: g for uid, g in b.groupby("seriesuid")}


def fingerprint_of(version, cfg, folds_of):
    fp = {"version": version, "design": "one-pass centre heatmap",
          "cfg": {k: v for k, v in cfg.items() if k not in NOT_TRAINING},
          "scan_cfg": {k: nodule.CFG[k] for k in ("min_radius_mm", "max_radius_mm", "excluded_min_radius_mm",
                                                  "excluded_margin_mm")},
          "cube": nodule.CUBE,
          "scans": hashlib.sha256(json.dumps(sorted(folds_of.items())).encode()).hexdigest(),
          "annotations_sha256": nodule.sha(os.path.join(nodule.LUNA_DIR, "annotations.csv")),
          "excluded_sha256": nodule.sha(os.path.join(nodule.LUNA_DIR, "evaluationScript", "annotations",
                                                     "annotations_excluded.csv"))}
    return json.loads(json.dumps(fp, sort_keys=True, default=str))


def train(a):
    card.require_card(DEV, torch)
    limit_card()
    cfg = nodule.with_settings(CFG, a.set)
    if a.steps:
        cfg["steps"] = a.steps
    folds_of, missing = nodule.scan_folds(False)
    if missing:
        print("WARNING: %d of LUNA16's scans have no prepared box and are left out" % len(missing), flush=True)
    ann, exc = nodule.reference()
    run_folds = a.folds if a.folds else list(range(5))
    scans = {u: nodule.Scan(u, ann, exc, nodule.CFG) for u in folds_of}
    out = out_root(a.version)
    os.makedirs(out, exist_ok=True)
    fingerprint = fingerprint_of(a.version, cfg, folds_of)
    secs = {}
    with ledger.run("nodule-det", "det v%d 5-fold train" % a.version) as led:
        led.settings_from({"cfg": cfg, "folds": run_folds})
        for k in run_folds:
            mine = [scans[u] for u, f in folds_of.items() if f != k]
            print("fold %d: %d training scans" % (k, len(mine)), flush=True)
            path = os.path.join(out, "fold%d.pt" % k)
            partial = os.path.join(out, "fold%d.partial.pt" % k)
            fp = {**fingerprint, "fold": k}
            ck = nodule.load_resumable(path, fp)
            if ck is not None:
                secs[k] = ck["seconds"]
                print("fold %d: checkpoint found at %s (no training)" % (k, path), flush=True)
                del ck
                continue
            net, secs[k], peak, _ = train_net(mine, cfg, 1000 * int(cfg["seed"]) + k + 1, "fold %d" % k, partial, fp)
            torch.save({"state": net.state_dict(), "cfg": cfg, "fingerprint": fp, "seconds": secs[k],
                        "peak_gb": peak}, path)
            if os.path.exists(partial):
                os.remove(partial)
            del net
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        led.number("train_seconds_per_fold", float(np.mean(list(secs.values()))), "s")
    return 0


def read(a):
    card.require_card(DEV, torch)
    limit_card()
    cfg = nodule.with_settings(CFG, a.set)
    folds_of, _ = nodule.scan_folds(False)
    ann, exc = nodule.reference()
    out = out_root(a.version)
    blobs = load_blobs()
    run_folds = a.folds if a.folds else list(range(5))
    scans_hash = hashlib.sha256(json.dumps(sorted(folds_of.items())).encode()).hexdigest()
    with ledger.run("nodule-det", "det v%d read" % a.version) as led:
        led.settings_from({"cfg": cfg, "folds": run_folds})
        for k in run_folds:
            ck = torch.load(os.path.join(out, "fold%d.pt" % k), map_location=DEV, weights_only=False)
            assert ck["fingerprint"]["fold"] == k and ck["fingerprint"]["scans"] == scans_hash, \
                "fold%d.pt was trained for another fold or another scan set" % k
            net = build(ck["cfg"])
            net.load_state_dict(ck["state"])
            del ck
            held = [nodule.Scan(u, ann, exc, nodule.CFG) for u, f in folds_of.items() if f == k]
            cands, diams, blob_frames, secs = [], [], [], []
            for scan in held:
                frame, diam, bh, s = read_scan(net, scan, cfg, blobs.get(scan.uid))
                cands.append(frame)
                diams.append(diam)
                if bh is not None:
                    blob_frames.append(bh)
                secs.append(s)
            pd.concat(cands, ignore_index=True).to_csv(os.path.join(out, "fold%d_candidates.csv" % k), index=False)
            pd.concat(diams, ignore_index=True).to_csv(os.path.join(out, "fold%d_diameters.csv" % k), index=False)
            if blob_frames:
                pd.concat(blob_frames, ignore_index=True).to_csv(os.path.join(out, "fold%d_blob_heat.csv" % k),
                                                                 index=False)
            print("  fold %d: read %d scans, %.1f s a scan" % (k, len(held), float(np.mean(secs)) if secs else 0.0),
                  flush=True)
            del net
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        present = list(run_folds)                           # only the folds read in this call, never an older fold's file
        cand = pd.concat([pd.read_csv(os.path.join(out, "fold%d_candidates.csv" % k), dtype={"seriesuid": str})
                          .assign(fold=k) for k in present], ignore_index=True)
        cand.to_csv(os.path.join(out, "candidates_oof.csv"), index=False)
        blob_files = [os.path.join(out, "fold%d_blob_heat.csv" % k) for k in present]
        blob_files = [p for p in blob_files if os.path.exists(p)]
        if blob_files:
            pd.concat([pd.read_csv(p, dtype={"seriesuid": str}) for p in blob_files], ignore_index=True) \
                .to_csv(os.path.join(out, "blob_heat_oof.csv"), index=False)
        uids = [u for u, f in folds_of.items() if f in present]
        res = froc.score(cand, ann, exc, uids, boot=1000)
        cpm, lo, hi = res["cpm"], res["boot"]["cpm"][1], res["boot"]["cpm"][2]
        sens = [100 * v for v in res["sens_at"]]
        print("CPM %.4f (95%% interval %.4f to %.4f) on %d scans, %d nodules" % (
            cpm, lo, hi, res["n_scans"], res["n_nodules"]))
        print("sensitivity out of 100 at %s false alarms per scan: %s" % (
            ", ".join("%g" % r for r in froc.FP_RATES), ", ".join("%.1f" % v for v in sens)))
        led.number("cpm", cpm).number("cpm_low", lo).number("cpm_high", hi)
        for r, v in zip(froc.FP_RATES, sens):
            led.number("sensitivity_at_%g" % r, v)
        reach = 100.0 * res["tp"] / max(res["n_nodules"], 1)
        print("reach %d of %d (%.1f of 100) in the top %d" % (res["tp"], res["n_nodules"], reach, cfg["top"]))
        led.number("reach_top100", reach)
        meta = {"version": a.version, "folds": run_folds, "cfg": cfg, "cpm": cpm, "cpm_low": lo, "cpm_high": hi,
                "sensitivity_out_of_100": sens, "fp_rates": list(froc.FP_RATES), "tp": res["tp"], "fp": res["fp"],
                "fn": res["fn"], "n_scans": res["n_scans"], "n_nodules": res["n_nodules"], "reach_top100": reach,
                "commit": ledger._commit(), "torch": torch.__version__, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    return 0


HEAT_COLUMNS = ["row", "seriesuid", "coordX", "coordY", "coordZ", "heat_at", "heat_max3", "fold"]


def heat(a):
    """Scores given places (any CSV with seriesuid, coordX/Y/Z in world mm) with the heatmap, each scan read by the
    net of the fold that never trained on it. No froc score: the raw logits at each place, to out."""
    card.require_card(DEV, torch)
    limit_card()
    cfg = nodule.with_settings(CFG, a.set)
    folds_of, _ = nodule.scan_folds(False)
    ann, exc = nodule.reference()
    out = out_root(a.version)
    run_folds = a.folds if a.folds else list(range(5))
    places = pd.read_csv(a.places, dtype={"seriesuid": str})
    places["row"] = np.arange(len(places))
    by_uid = {u: g for u, g in places.groupby("seriesuid")}
    held = {k: [u for u, f in folds_of.items() if f == k and u in by_uid] for k in run_folds}
    n_read = sum(len(by_uid[u]) for k in run_folds for u in held[k])
    print("places in %s: %d, on scans of the folds read: %d, skipped (scan not in the set): %d"
          % (a.places, len(places), n_read, len(places) - n_read), flush=True)
    scans_hash = hashlib.sha256(json.dumps(sorted(folds_of.items())).encode()).hexdigest()
    frames = []
    with ledger.run("nodule-det", "det v%d heat" % a.version) as led:
        led.settings_from({"cfg": cfg, "folds": run_folds, "places": a.places})
        for k in run_folds:
            ck = torch.load(os.path.join(out, "fold%d.pt" % k), map_location=DEV, weights_only=False)
            assert ck["fingerprint"]["fold"] == k and ck["fingerprint"]["scans"] == scans_hash, \
                "fold%d.pt was trained for another fold or another scan set" % k
            net = build(ck["cfg"])
            net.load_state_dict(ck["state"])
            del ck
            secs = []
            for u in held[k]:
                scan = nodule.Scan(u, ann, exc, nodule.CFG)
                t0 = time.time()
                frames.append(place_heat(heat_logit(net, scan, cfg)[0], scan, by_uid[u]).assign(fold=k))
                secs.append(time.time() - t0)
            print("  fold %d: read %d scans, %.1f s a scan" % (k, len(held[k]), float(np.mean(secs)) if secs else 0.0),
                  flush=True)
            del net
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HEAT_COLUMNS)
        res = res.sort_values("row", kind="stable").reset_index(drop=True)[HEAT_COLUMNS]
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        res.to_csv(a.out, index=False)
        print("folds in the file: %s; rows written to %s: %d" % (
            ", ".join(str(k) for k in sorted(res.fold.unique())) or "none", a.out, len(res)), flush=True)
        led.number("rows_written", float(len(res)))
    return 0


def report(label, rate, batch, net, loss):
    params = sum(p.numel() for p in net.parameters()) / 1e6
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEV.type == "cuda" else float("nan")
    print("%s: %.2f steps/s, %.2f cubes/s, peak %.1f GB, %.1f M weights, loss %.4f"
          % (label, rate, rate * batch, peak, params, loss), flush=True)


def time_steps(net, opt, cfg, step_fn, n=210, warm=10):
    """Steps a second over 200 timed steps (n less the `warm` warm-up ones, which are not timed); each step is a loss,
    a backward and an optimiser step."""
    for i in range(n):
        if i == warm:
            sync()
            t0 = time.time()
        loss = step_fn(i)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
        opt.step()
    sync()
    return (n - warm) / (time.time() - t0), float(loss)


def bench(a):
    """The det step at 96^3, net only (cubes pre-staged on the card) and with the multi-process sampler; with
    --nodule-v2, nodule.py's own v2 step (its net, loss and one-thread sampler) the same two ways. tools/ab.py compares
    the cubes-a-second lines."""
    card.require_card(DEV, torch)
    cfg = nodule.with_settings(CFG, a.set)
    limit_card()
    torch.backends.cudnn.benchmark = True
    folds_of, _ = nodule.scan_folds(False)
    ann, exc = nodule.reference()
    scans = [nodule.Scan(u, ann, exc, nodule.CFG) for u, f in folds_of.items() if f != 0]
    balls = ball_list(scans)
    if a.nodule_v2:
        with open(os.path.join(nodule.out_root(2), "meta.json"), encoding="utf-8") as f:
            v2 = dict(nodule.CFG, **json.load(f)["cfg"])           # v2's own settings: batch 2, not CFG's default
        scans = [nodule.Scan(u, ann, exc, v2) for u, f in folds_of.items() if f != 0]
        img, ign, _ = make_batch(scans, ball_list(scans), cfg, np.random.default_rng(0), v2["batch"])
        img = torch.as_tensor(img, device=DEV)
        cube = torch.stack([img, (torch.rand(img.shape, device=DEV) < 0.01).to(torch.uint8), torch.ones_like(img)], 1)
        net = nodule.build(v2).train()
        opt = torch.optim.AdamW(net.parameters(), lr=v2["lr"], weight_decay=v2["weight_decay"], fused=DEV.type == "cuda")
        if DEV.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rate, loss = time_steps(net, opt, v2, lambda i: v2_loss_on(net, cube, v2))
        report("nodule.py v2 net only (batch %d)" % v2["batch"], rate, v2["batch"], net, loss)
        feed = nodule.Feed(nodule.Sampler(scans, v2, 0, v2["batch"]))
        if DEV.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rate, loss = time_steps(net, opt, v2, lambda i: v2_loss_on(net, feed.next(), v2))
        report("nodule.py v2 with the 1-thread sampler (batch %d)" % v2["batch"], rate, v2["batch"], net, loss)
        return 0
    staged = [torch.as_tensor(v, device=DEV) for v in make_batch(scans, balls, cfg, np.random.default_rng(0), cfg["batch"])]
    net = build(cfg).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=DEV.type == "cuda")
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rate, loss = time_steps(net, opt, cfg, lambda i: loss_of_batch(net, tuple(staged), cfg,
                                                                    np.random.default_rng([0, 1, i])))
    report("det net only", rate, cfg["batch"], net, loss)
    batches = Batches(scans, cfg, 0, 0)
    feed = Feed(batches)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        rate, loss = time_steps(net, opt, cfg, lambda i: loss_of_batch(net, feed.next(), cfg,
                                                                        np.random.default_rng([0, 1, i])))
    finally:
        batches.close()
    report("det with the %d-worker sampler" % cfg["workers"], rate, cfg["batch"], net, loss)
    return 0


def v2_loss_on(net, batch, v2):
    """nodule.py's loss of a batch the way its train_net takes it (without the brightness jitter)."""
    x, y, w = batch[:, 0:1].float() / 255.0, batch[:, 1:2].float(), batch[:, 2:3].float()
    with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
        outs = net(x)
    return nodule.loss_of(outs, y, w, v2)


def check_warp():
    """The warp puts a point where S * Q says: a single bright voxel moves to its predicted place, to within a voxel."""
    n = nodule.CUBE
    img = torch.zeros((1, n, n, n), dtype=torch.uint8, device=DEV)
    img[0, 9, 20, 12] = 255
    Q = np.rint(rotation(0, math.pi / 2)) @ np.diag([-1.0, 1.0, 1.0])
    Qt = torch.tensor(Q[None], dtype=torch.float32, device=DEV)
    St = torch.tensor([1.25], dtype=torch.float32, device=DEV)
    x, _ = warp_batch(img, torch.zeros_like(img), Qt, St, torch.ones(1, device=DEV))
    got = np.array(np.unravel_index(int(x[0, 0].argmax()), (n, n, n)))
    co, _, _ = warp_balls(torch.tensor([[[9.0, 20.0, 12.0, 3.0]]], device=DEV), Qt, St)
    want = np.rint(co[0, 0].cpu().numpy())
    assert np.all(np.abs(got - want) <= 1), "warp: bright voxel at %s, predicted %s" % (got, want)


def check_heat_index():
    """The cell lookup of heat (CPU only): a voxel at a cell's centre reads that cell's logit, heat_max3 is the largest
    logit of its 3x3x3 neighbourhood, and a voxel off the grid clips to the edge cell."""
    logit = torch.tensor(np.random.default_rng(1).normal(0, 1, (6, 5, 4)), dtype=torch.float32)
    cell = np.array([[2, 3, 1]])                           # z, y, x
    at, mx = heat_values(logit, 2 * cell + 0.5)
    assert at[0] == float(logit[2, 3, 1]), "heat_at %g, the cell's logit %g" % (at[0], float(logit[2, 3, 1]))
    assert mx[0] == float(logit[1:4, 2:5, 0:3].max()), "heat_max3 %g" % mx[0]
    at, mx = heat_values(logit, np.array([[100.0, -50.0, 3.0]]))      # clips to cell (5, 0, 1)
    assert at[0] == float(logit[5, 0, 1]), "off-grid heat_at %g" % at[0]
    assert mx[0] == float(logit[4:6, 0:2, 0:3].max()), "off-grid heat_max3 %g" % mx[0]
    print("heat index: a cell's centre, its 3x3x3 maximum and the edge clip agree")


def selftest(a):
    """On the processor, in a few minutes: check the warp's geometry; train a tiny net on a synthetic box with three
    painted balls for 500 steps (two worker processes), check that its read finds all three centres within their
    radius; check that a run stopped at step 250 and resumed equals an uninterrupted run; check the candidate columns
    match nodule.py's and that froc.py scores them."""
    tmp = tempfile.mkdtemp(prefix="nodule-det-")
    keep = nodule.CUBE
    nodule.CUBE = 32                                       # cube_of reads the cube size from nodule.py's module
    try:
        check_warp()
        check_heat_index()
        rng = np.random.default_rng(0)
        vol = np.clip(rng.normal(40, 6, (64, 64, 64)), 0, 255).astype(np.uint8)
        centres = [(16, 20, 22), (44, 30, 40), (30, 50, 16)]
        for c in centres:
            nodule.paint(vol, np.array(c, float), 4.0, 180)
        path = os.path.join(tmp, "box.npy")
        np.save(path, vol)
        box = Box("selftest-box", path, (64, 64, 64), [[*c, 4.0] for c in centres])
        cfg = dict(CFG, widths=[8, 16, 16, 32], fpn=16, steps=500, batch=4, warmup=20, lr=2e-3, nodule_share=0.8,
                   shift_mm=6.0, workers=2, prefetch=2, log_every=100, save_every=100)
        net_a, *_ = train_net([box], cfg, 0, "selftest uninterrupted")
        partial, fp = os.path.join(tmp, "partial.pt"), {"selftest": True}
        train_net([box], cfg, 0, "selftest stopped", partial, fp, stop_at=250)
        assert os.path.exists(partial), "no partial checkpoint at the stop"
        net_b, *_ = train_net([box], cfg, 0, "selftest resumed", partial, fp)
        diff = max(float((net_a.state_dict()[k].float() - net_b.state_dict()[k].float()).abs().max())
                   for k in net_a.state_dict())
        assert diff < 1e-5, "resumed run differs from the uninterrupted one by %g" % diff
        print("resume: stopped at 250 and resumed equals uninterrupted (largest weight difference %g)" % diff)
        frame, _, _, secs = read_scan(net_a, box, cfg)
        assert list(frame.columns) == NODULE_COLUMNS[:5], list(frame.columns)
        cand = frame.assign(fold=0)
        assert list(cand.columns) == NODULE_COLUMNS
        for c in centres:
            d = np.linalg.norm(frame[["coordX", "coordY", "coordZ"]].to_numpy() - np.array([c[2], c[1], c[0]], float), axis=1)
            print("centre %s: nearest candidate %.2f voxels away (radius 4)" % (c, d.min()))
            assert d.min() < 4.0, "centre %s has no candidate within its radius" % (c,)
        ann = pd.DataFrame({"seriesuid": box.uid, "coordX": [c[2] for c in centres], "coordY": [c[1] for c in centres],
                            "coordZ": [c[0] for c in centres], "diameter_mm": 8.0})
        res = froc.score(cand, ann, ann.iloc[:0], [box.uid])
        assert res["tp"] == 3, res
        print("froc: %d of 3 found, CPM %.3f, %d candidates, read in %.1f s" % (res["tp"], res["cpm"], len(frame), secs))
    finally:
        nodule.CUBE = keep
        shutil.rmtree(tmp, ignore_errors=True)
    print("selftest passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "read"):
        p = sub.add_parser(name)
        p.add_argument("--version", type=int, required=True)
        p.add_argument("--folds", type=int, nargs="+")
        p.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(CFG)))
        if name == "train":
            p.add_argument("--steps", type=int)
    b = sub.add_parser("bench")
    b.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(CFG)))
    b.add_argument("--nodule-v2", action="store_true", help="time nodule.py's v2 step instead")
    h = sub.add_parser("heat")
    h.add_argument("--version", type=int, required=True)
    h.add_argument("--places", required=True, help="a CSV with seriesuid, coordX, coordY, coordZ (world mm)")
    h.add_argument("--out", required=True, help="the CSV to write: row, seriesuid, coordX/Y/Z, heat_at, heat_max3, fold")
    h.add_argument("--folds", type=int, nargs="+")
    h.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(CFG)))
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"train": train, "read": read, "heat": heat, "bench": bench, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
