"""The first lung nodule detector: a 3D U-Net that colours in nodules on chest CT, scored the LUNA16 way.

What it finds. A *candidate* is a point in a scan (x, y, z in the scanner's millimetres) with a
probability that a nodule is there. The detector hands over up to 100 per scan; the scorer
(froc.py, a port of LUNA16's own script) calls a nodule found when a candidate lies closer to its
centre than its radius.

Why a ball. LUNA16 gives each nodule as a centre and a diameter, not an outline. So the network is
taught to colour in a ball of the nodule's radius (at least 2 mm, so the smallest are still a few
voxels): a ball is what the scorer's "close enough" means, and the height of the colouring's peak
is the candidate's probability. The scans are the 1 mm lung boxes of lung_prepare.py, so a voxel is
a millimetre and a ball's radius in voxels is its radius in mm.

Why some findings get no weight. LUNA16 keeps a second list of findings it counts neither way
(under 3 mm, or marked by too few radiologists). Calling one of them a nodule is not an error and
calling it normal is not either, so the loss ignores a ball around each (3 mm radius at least, plus
2 mm for the uncertainty in where it is): the network is neither punished nor rewarded there. Where
such a ball overlaps a real nodule's ball the real nodule wins, voxel by voxel.

What CPM is. Sweeping a threshold over the candidates' probabilities gives sensitivity (share of
nodules found) against false alarms per scan; CPM is the mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4
and 8 false alarms per scan, with a 95% interval from resampling scans. Published: nnDetection 93.0.

Folds. LUNA16 has no held-out test set; its ten fixed groups of scans (subsets 0-9) are the
protocol. Development here is five folds, a fold being two groups (subset // 2); `--ten` is the
ten-fold run (fold = subset). Only LUNA16's 888 scans take part. Each fold trains on the other folds
and reads its own scans, so every score is from a model that never saw the scan.

    python modalities/ct/eval/nodule.py train --version N [--ten] [--folds 0 1] [--steps S] [--set key=value] [--extra-lidc] [--extra-nlst]
    python modalities/ct/eval/nodule.py smoke

`--extra-lidc` adds the LIDC-IDRI scans LUNA16 left out (lidc_extra.py, answer key built by LUNA16's
rule) to every fold's training scans: never held out, never read, never scored. Its fingerprint gains
"extra_lidc": True, so runs without the flag keep their saved checkpoints.
`--extra-nlst` does the same with the LIDC-annot-NLST501 scans (nlst501.py, answer key built by the same
rule from four radiologists' marks on NLST scans): its fingerprint gains "extra_nlst": True, only when set.

Env RAIDIOLOGY_NODULE_CARD_GB (default 5.0) is the share of the card it leases and caps itself to.
A restarted run resumes folds whose saved fingerprint matches, their held-out candidates too when the
checkpoint carries them and they were read the way this run reads (READOUT and the reading settings,
READ_KEYS, which are not part of the training fingerprint: a new floor or pool rereads, never retrains;
RAIDIOLOGY_NO_RESUME=1 retrains and rereads). Folds also resume mid-fold from fold{k}.partial.pt, saved
every cfg["save_every"] steps (not in the fingerprint) and deleted when the fold's fold{k}.pt is saved. A new training setting changes every saved fingerprint:
give the saved checkpoints the value they trained with in the same change, or they train again (v1
lacked max_radius_mm and started retraining 8,000 steps to read once).
"""
import argparse
import hashlib
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
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import froc  # noqa: E402
import lung_prepare as lp  # noqa: E402
import segmenter  # noqa: E402
from core import card, ledger, paths, polite, sets  # noqa: E402

DEV = segmenter.DEV
CUBE = 96
LUNA_DIR = os.path.join(paths.DATASETS, "public", "luna16")
CFG = {
    "widths": [16, 32, 64, 128, 256], "deep": [1.0, 0.5],
    "steps": 8000, "batch": 4, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 300, "clip": 12.0,
    "nodule_share": 0.5, "shift_mm": 40.0, "min_radius_mm": 2.0, "max_radius_mm": 0.0,
    "excluded_min_radius_mm": 3.0, "excluded_margin_mm": 2.0, "brightness": 0.05,
    "seed": 0, "log_every": 200, "queue": 3, "save_every": 2000,   # save_every: not in the fingerprint
    # reading
    "overlap": 0.5, "pool": 7, "min_probability": 0.0, "top": 100, "read_batch": 2, "flips": 1,
}
SYNC_EVERY = 4
READOUT = "blended float32 logits"   # how read_scan scores; saved candidates read another way are read again
READ_KEYS = ("overlap", "pool", "min_probability", "top", "read_batch", "flips")
NOT_TRAINING = (*READ_KEYS, "save_every")   # settings that are not part of the training fingerprint
# mirror images each window is also read in, by count (dims of a (batch, 1, z, y, x) window): 1 none,
# 2 left-right, 4 also front-back, 8 also head-foot; their logits are averaged
FLIPS = {1: [()], 2: [(), (4,)], 4: [(), (4,), (3,), (3, 4)],
         8: [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]}   # reading only: not in the training fingerprint


def out_root(version):
    return os.path.join(paths.WORK, "eval", "lung", "nodule-v%d" % version)


def card_gb():
    return float(os.environ.get("RAIDIOLOGY_NODULE_CARD_GB", "5.0"))


_LIMITED = []


def limit_card():
    """Lease this process's share of the card, then a hard ceiling at it (see core/card.py)."""
    if DEV.type != "cuda" or _LIMITED:
        return
    gb = card_gb()
    card.lease(gb, "nodule")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())
    _LIMITED.append(gb)


# ---- the data --------------------------------------------------------------------------------

def reference():
    """(nodules, excluded) frames with seriesuid, coordX/Y/Z, diameter_mm."""
    ann = pd.read_csv(os.path.join(LUNA_DIR, "annotations.csv"))
    exc = pd.read_csv(os.path.join(LUNA_DIR, "evaluationScript", "annotations", "annotations_excluded.csv"))
    return ann, exc


def scan_folds(ten):
    """{seriesuid: fold} for LUNA16's scans that have a prepared box, and the ones that have not."""
    have, missing = {}, []
    for uid, subset in sorted(lp.luna16_subsets().items()):
        npy, js = lp.paths_for(lp.out_dir(), uid)
        if os.path.exists(npy) and os.path.exists(js):
            have[uid] = subset if ten else subset // 2
        else:
            missing.append(uid)
    return have, missing


def training_uids(folds_of, k, extra_uids):
    """The scans fold k trains on: every LUNA16 scan outside fold k, then the extra LIDC scans (never held out)."""
    return [u for u, f in folds_of.items() if f != k] + list(extra_uids)


class Scan:
    """One prepared box with its nodules and excluded findings in voxel units (z, y, x, radius)."""

    def __init__(self, uid, ann, exc, cfg):
        self.uid = uid
        self.meta = lp.load_meta(uid)
        self.path = lp.paths_for(lp.out_dir(), uid)[0]
        self.shape = tuple(self.meta["shape"])
        assert list(self.meta["spacing_mm"]) == [1.0, 1.0, 1.0], "the box is not at 1 mm"
        # The target ball is the nodule's radius, at least min_radius_mm and, when max_radius_mm is set,
        # at most that: then a large nodule weighs about what a small one does in the Dice term, and the
        # rest of it (`whole`) is left out of the loss rather than called normal.
        floor, cap = cfg["min_radius_mm"], cfg["max_radius_mm"]
        self.whole = self._balls(ann, lambda d: max(d / 2.0, floor))
        self.nodules = self._balls(ann, lambda d: min(max(d / 2.0, floor), cap) if cap else max(d / 2.0, floor))
        self.excluded = self._balls(exc, lambda d: max(max(d, 0.0) / 2.0, cfg["excluded_min_radius_mm"])
                                    + cfg["excluded_margin_mm"])

    def _balls(self, frame, radius):
        rows = frame[frame.seriesuid == self.uid]
        if rows.empty:
            return np.zeros((0, 4))
        c = lp.world_to_voxel(self.meta, rows[["coordX", "coordY", "coordZ"]].to_numpy(float))
        r = np.array([radius(d) for d in rows.diameter_mm.to_numpy(float)])
        return np.column_stack([c, r])


def paint(dst, centre, radius, value):
    """Set the voxels of `dst` (a cube) within `radius` of `centre` (cube voxel units) to `value`."""
    lo = [max(int(math.floor(c - radius)), 0) for c in centre]
    hi = [min(int(math.ceil(c + radius)) + 1, s) for c, s in zip(centre, dst.shape)]
    if any(h <= l for l, h in zip(lo, hi)):
        return
    grid = np.ogrid[tuple(slice(l, h) for l, h in zip(lo, hi))]
    d2 = sum((g - c) ** 2 for g, c in zip(grid, centre))
    dst[tuple(slice(l, h) for l, h in zip(lo, hi))][d2 <= radius ** 2] = value


def cube_of(scan, start):
    """(image uint8, target uint8, weight uint8) of the CUBE-wide cube whose corner is voxel `start`.
    Outside the box the image is 0 (padding), the target 0 and the weight 1."""
    img = np.zeros((CUBE,) * 3, np.uint8)
    src, dst = [], []
    for s, n in zip(start, scan.shape):
        a, b = max(s, 0), min(s + CUBE, n)
        src.append(slice(a, b))
        dst.append(slice(a - s, b - s))
    vol = np.load(scan.path, mmap_mode="r")
    img[tuple(dst)] = vol[tuple(src)]
    target = np.zeros_like(img)
    weight = np.ones_like(img)
    off = np.asarray(start, float)
    for z, y, x, r in scan.excluded:
        paint(weight, np.array([z, y, x]) - off, r, 0)
    for z, y, x, r in scan.whole:               # without a cap these are the target balls themselves
        paint(weight, np.array([z, y, x]) - off, r, 0)
    for z, y, x, r in scan.nodules:             # last, so a real nodule overrides an excluded ball
        paint(target, np.array([z, y, x]) - off, r, 1)
        paint(weight, np.array([z, y, x]) - off, r, 1)
    return img, target, weight


class Sampler:
    """Endless batches of augmented cubes, built in one background thread (the boxes do not all fit in
    memory, so cubes are read from the .npy files memory-mapped)."""

    def __init__(self, scans, cfg, seed, batch):
        self.scans, self.cfg, self.batch = scans, cfg, batch
        self.rng = np.random.default_rng(seed)
        self.balls = [(s, i) for s in scans for i in range(len(s.nodules))]
        self.q = queue.Queue(maxsize=int(cfg["queue"]))
        self.error = None
        threading.Thread(target=self._run, daemon=True, name="nodule-cubes").start()

    def start_of(self):
        rng, cfg = self.rng, self.cfg
        if self.balls and rng.random() < cfg["nodule_share"]:
            scan, i = self.balls[int(rng.integers(len(self.balls)))]
            centre = scan.nodules[i, :3] + rng.uniform(-cfg["shift_mm"], cfg["shift_mm"], 3)
            start = np.rint(centre).astype(int) - CUBE // 2
        else:
            scan = self.scans[int(rng.integers(len(self.scans)))]
            start = np.array([rng.integers(0, max(n - CUBE, 0) + 1) for n in scan.shape])
        # A box smaller than the cube sits at the corner (padded with 0); else the cube stays inside it.
        start = [int(np.clip(s, 0, n - CUBE)) if n >= CUBE else 0 for s, n in zip(start, scan.shape)]
        return scan, start

    def augment(self, stack):
        rng = self.rng
        for ax in range(3):
            if rng.random() < 0.5:
                stack = np.flip(stack, axis=ax + 1)
        return np.rot90(stack, int(rng.integers(4)), axes=(2, 3))     # the axial plane is (y, x)

    def _run(self):
        try:
            while True:
                out = torch.empty((self.batch, 3, CUBE, CUBE, CUBE), dtype=torch.uint8,
                                  pin_memory=DEV.type == "cuda")
                view = out.numpy()
                for b in range(self.batch):
                    scan, start = self.start_of()
                    view[b] = self.augment(np.stack(cube_of(scan, start)))
                self.q.put(out)
        except Exception as e:  # noqa: BLE001  handed to the training loop, which stops with it
            self.error = e
            self.q.put(None)

    def get(self):
        got = self.q.get()
        if got is None:
            raise self.error
        return got


class Feed:
    """Batches already on the card: the copy of the next one runs on a side stream while this one trains."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.side = torch.cuda.Stream() if DEV.type == "cuda" else None
        self.ahead = self._fetch()

    def _fetch(self):
        cpu = self.sampler.get()
        if self.side is None:
            return cpu, None
        with torch.cuda.stream(self.side):
            return cpu.to(DEV, non_blocking=True), cpu       # `cpu` is kept alive until the next call

    def next(self):
        gpu, _ = self.ahead
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)
            gpu.record_stream(torch.cuda.current_stream())
        self.ahead = self._fetch()
        return gpu


# ---- the network and its loss ----------------------------------------------------------------

def build(cfg):
    return segmenter.UNet(cfg["widths"], 1, heads=len(cfg["deep"]), float32_heads=True).to(DEV)


def loss_of(outs, y, w, cfg):
    """Weighted cross-entropy plus soft Dice (only where the weight is 1), at full and half resolution.
    The half-resolution target is the average of each 2x2x2 block, its weight 0 if any voxel's was."""
    total = 0.0
    for k, o in enumerate(outs):
        if k:
            yk, wk = F.avg_pool3d(y, 2 ** k), -F.max_pool3d(-w, 2 ** k)
        else:
            yk, wk = y, w
        o = o.float()
        bce = (F.binary_cross_entropy_with_logits(o, yk, reduction="none") * wk).sum() / wk.sum().clamp_min(1.0)
        p, t = torch.sigmoid(o) * wk, yk * wk
        dice = 1 - (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
        total = total + cfg["deep"][k] * (bce + dice)
    return total / sum(cfg["deep"][:len(outs)])


def lr_factor(step, cfg):
    if step < cfg["warmup"]:
        return (step + 1) / cfg["warmup"]
    done = (step - cfg["warmup"]) / max(cfg["steps"] - cfg["warmup"], 1)
    return 0.5 * (1 + math.cos(math.pi * min(done, 1.0)))


def train_net(scans, cfg, seed, tag, partial=None, fingerprint=None):
    """Train a fresh net on `scans`; returns (net, seconds, peak card GB, seconds per step).

    With `partial` (a path) and `fingerprint`, a partial checkpoint is saved there every cfg["save_every"]
    steps, and a partial checkpoint with the same fingerprint is continued from its next step."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    net = build(cfg).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            fused=DEV.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg))
    start, prior_secs, prior_peak, sampler_seed = 0, 0.0, 0.0, seed
    ck = load_resumable(partial, fingerprint) if partial else None
    if ck is not None and 0 < ck.get("step", 0) < cfg["steps"]:
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, prior_secs, prior_peak = ck["step"], ck["seconds"], ck.get("peak_gb", 0.0)
        # The background sampler's exact state cannot be saved: both generators restart from (seed, step).
        torch.manual_seed(seed * 1_000_003 + start)
        gen = torch.Generator(device=DEV).manual_seed(seed * 1_000_003 + start)
        sampler_seed = [seed, start]
        print("fold %s: resumed mid-fold at step %d of %d (partial checkpoint)"
              % (tag.split()[-1], start, cfg["steps"]), flush=True)
    del ck
    feed = Feed(Sampler(scans, cfg, sampler_seed, cfg["batch"]))
    pace = polite.Pace()
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    t0 = block = log_t = time.time()
    t0 -= prior_secs                                       # seconds trained add up across a resume
    running, seen = 0.0, 0
    for step in range(start + 1, cfg["steps"] + 1):
        batch = feed.next()
        x = batch[:, 0:1].float() / 255.0
        y, w = batch[:, 1:2].float(), batch[:, 2:3].float()
        n = x.shape[0]
        gain = 1 + (torch.rand(n, 1, 1, 1, 1, device=DEV, generator=gen) * 2 - 1) * cfg["brightness"]
        shift = (torch.rand(n, 1, 1, 1, 1, device=DEV, generator=gen) * 2 - 1) * cfg["brightness"]
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            outs = net(x * gain + shift)
        loss = loss_of(outs, y, w, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        running = running + loss.detach()
        seen += 1
        if step % SYNC_EVERY == 0 or step == cfg["steps"]:
            running = float(running)                       # the wait for the card
            pace.rest(time.time() - block)
            block = time.time()
        if step % cfg["log_every"] == 0 or step == cfg["steps"]:
            running = float(running)
            now = time.time()
            print("  %s step %d of %d, loss %.4f, %.1f s per 100 steps, using %.0f%% of the card"
                  % (tag, step, cfg["steps"], running / seen, 100 * (now - log_t) / seen, 100 * pace.used()),
                  flush=True)
            log_t, running, seen = now, 0.0, 0
        if partial and step % cfg["save_every"] == 0 and step < cfg["steps"]:
            peak_now = max(prior_peak, torch.cuda.max_memory_allocated() / 2 ** 30) if DEV.type == "cuda" else float("nan")
            torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "seconds": time.time() - t0, "peak_gb": peak_now,
                        "fingerprint": fingerprint}, partial + ".tmp")
            os.replace(partial + ".tmp", partial)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    peak = max(prior_peak, torch.cuda.max_memory_allocated() / 2 ** 30) if DEV.type == "cuda" else float("nan")
    print("  %s: %d steps in %.0f s, peak card %.2f GB" % (tag, cfg["steps"], seconds, peak), flush=True)
    return net, seconds, peak, seconds / cfg["steps"]


# ---- reading a held-out scan -----------------------------------------------------------------

@torch.inference_mode()
def read_scan(net, scan, cfg):
    """Candidates of one scan as a frame (seriesuid, coordX/Y/Z, probability) and the seconds taken.

    96 mm windows with half overlap slide over the box (padded with 0 up to one window where it is
    smaller); the full-resolution logits are blended with a Gaussian bell so a window is trusted most
    in its middle. A candidate is a voxel equal to the max of its 7x7x7 neighbourhood (with probability
    at least min_probability, by default no floor), the top 100 per scan: with float32 logits v2 put
    only 6.3 peaks a scan above 0.001, the list ran out before 8 false alarms a scan, and a candidate
    below every other one can only add nodules at the end of the curve. With "flips" above 1 each window is also read mirrored and
    the logits averaged (FLIPS), that many times the reading. Logits, not sigmoids: a confident net's sigmoids are exactly
    1.0 in float32 above a logit of about 17, so blended sigmoids made flat tops, many peaks on one
    blob and ties the FROC cannot rank (71% of v2's fold-0 candidates scored 1.0). The probability is
    the sigmoid of the blended logit in float64, distinct up to a logit of about 36."""
    t0 = time.time()
    net.eval()
    vol = np.load(scan.path)
    shape = tuple(vol.shape)
    full = tuple(max(s, CUBE) for s in shape)
    x = torch.zeros(full, dtype=torch.uint8, device=DEV)
    x[:shape[0], :shape[1], :shape[2]] = torch.from_numpy(vol).to(DEV)
    acc = torch.zeros(full, device=DEV)
    weight = torch.zeros(full, device=DEV)
    bell = segmenter._bell((CUBE,) * 3)
    corners = [(a, b, c) for a in segmenter._starts(full[0], CUBE, cfg["overlap"])
               for b in segmenter._starts(full[1], CUBE, cfg["overlap"])
               for c in segmenter._starts(full[2], CUBE, cfg["overlap"])]
    for i in range(0, len(corners), cfg["read_batch"]):
        group = corners[i:i + cfg["read_batch"]]
        win = torch.stack([x[a:a + CUBE, b:b + CUBE, c:c + CUBE] for a, b, c in group])[:, None].float() / 255.0
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            z = sum(net(win.flip(f))[0].float().flip(f) if f else net(win)[0].float()
                    for f in FLIPS[cfg["flips"]])[:, 0] / cfg["flips"]
        for (a, b, c), zw in zip(group, z):
            acc[a:a + CUBE, b:b + CUBE, c:c + CUBE] += zw * bell
            weight[a:a + CUBE, b:b + CUBE, c:c + CUBE] += bell
    logit = (acc / weight)[:shape[0], :shape[1], :shape[2]]
    pmin = cfg["min_probability"]
    k, floor = cfg["pool"], (math.log(pmin / (1.0 - pmin)) if pmin > 0 else -math.inf)
    peak = F.max_pool3d(logit[None, None], k, 1, k // 2)[0, 0]
    hit = ((logit == peak) & (logit >= floor)).nonzero()
    score = logit[hit[:, 0], hit[:, 1], hit[:, 2]]
    keep = score.argsort(descending=True)[:cfg["top"]]
    zyx = hit[keep].cpu().numpy()
    score = 1.0 / (1.0 + np.exp(-score[keep].cpu().numpy().astype(np.float64)))
    xyz = lp.voxel_to_world(scan.meta, zyx) if len(zyx) else np.zeros((0, 3))
    frame = pd.DataFrame({"seriesuid": scan.uid, "coordX": xyz[:, 0], "coordY": xyz[:, 1],
                          "coordZ": xyz[:, 2], "probability": score.astype(float)})
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return frame, time.time() - t0


def read_all(net, scans, cfg, tag):
    frames, secs = [], []
    for scan in scans:
        frame, s = read_scan(net, scan, cfg)
        frames.append(frame)
        secs.append(s)
    print("  %s: read %d scans, %.1f s a scan" % (tag, len(scans), float(np.mean(secs)) if secs else 0.0), flush=True)
    return frames, secs


# ---- a run -----------------------------------------------------------------------------------

def with_settings(cfg, items):
    cfg = dict(cfg)
    cfg.update(sets.parse(items, known=cfg))
    return cfg


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_resumable(path, fingerprint):
    """The checkpoint at `path` if its fingerprint equals `fingerprint`, else None (also when
    RAIDIOLOGY_NO_RESUME is 1 or the file is missing or unreadable)."""
    if os.environ.get("RAIDIOLOGY_NO_RESUME") == "1" or not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location=DEV, weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    return ck if isinstance(ck, dict) and ck.get("fingerprint") == fingerprint else None


def train(a):
    card.require_card(DEV, torch)
    # 2026-10-03: leased before the scans are read (a job waiting with its data held up to 6.9 GB; the machine
    # ran out of committed memory at 23:21 and a training job crashed)
    limit_card()
    cfg = with_settings(CFG, a.set)
    if a.steps:
        cfg["steps"] = a.steps
    folds_of, missing = scan_folds(a.ten)
    if missing:
        print("WARNING: %d of LUNA16's scans have no prepared box and are left out: %s"
              % (len(missing), ", ".join(m[-12:] for m in missing[:10])), flush=True)
    n_folds = 10 if a.ten else 5
    run_folds = a.folds if a.folds else list(range(n_folds))
    ann, exc = reference()
    extra = []
    n_nlst = 0
    if a.extra_lidc:
        import lidc_extra
        extra, e_ann, e_exc = lidc_extra.extra_reference()
        assert not set(extra) & set(folds_of), "an extra scan is a LUNA16 scan"
        ann, exc = pd.concat([ann, e_ann], ignore_index=True), pd.concat([exc, e_exc], ignore_index=True)
    if a.extra_nlst:
        import nlst501
        n_uids, n_ann, n_exc = nlst501.reference()
        assert not set(n_uids) & (set(folds_of) | set(extra)), "an NLST scan is a LUNA16 or LIDC scan"
        extra, n_nlst = [*extra, *n_uids], len(n_uids)
        ann, exc = pd.concat([ann, n_ann], ignore_index=True), pd.concat([exc, n_exc], ignore_index=True)
    scans = {u: Scan(u, ann, exc, cfg) for u in [*folds_of, *extra]}
    out = out_root(a.version)
    os.makedirs(out, exist_ok=True)
    fingerprint = {"version": a.version, "ten": bool(a.ten), "cfg": {k: v for k, v in cfg.items() if k not in NOT_TRAINING},
                   "scans": hashlib.sha256(json.dumps(sorted(folds_of.items())).encode()).hexdigest(),
                   "annotations_sha256": sha(os.path.join(LUNA_DIR, "annotations.csv")),
                   "excluded_sha256": sha(os.path.join(LUNA_DIR, "evaluationScript", "annotations",
                                                       "annotations_excluded.csv"))}
    fingerprint = {**fingerprint, **({"extra_lidc": True} if a.extra_lidc else {}),
                   **({"extra_nlst": True} if a.extra_nlst else {})}
    fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True, default=str))
    reading = {"readout": READOUT, **{k: cfg[k] for k in READ_KEYS if k != "read_batch"}}
    name = "nodule v%d %s" % (a.version, "10-fold" if a.ten else "5-fold")
    frames, train_secs, read_secs, peaks, info = [], {}, [], [], {}
    with ledger.run("nodule", name) as led:
        led.settings_from({"cfg": cfg, "ten": bool(a.ten), "folds": run_folds, "extra_lidc": bool(a.extra_lidc),
                            "extra_nlst": bool(a.extra_nlst)})
        for k in run_folds:
            mine = [scans[u] for u in training_uids(folds_of, k, extra)]
            print("fold %d: %d training scans (%d of them extras, %d of those NLST)" % (k, len(mine), len(extra), n_nlst),
                  flush=True)
            held = [scans[u] for u, f in folds_of.items() if f == k]
            path = os.path.join(out, "fold%d.pt" % k)
            partial = os.path.join(out, "fold%d.partial.pt" % k)
            fp = {**fingerprint, "fold": k}
            ck = load_resumable(path, fp)
            net = build(cfg)
            if ck is not None:
                net.load_state_dict(ck["state"])
                train_secs[k], peak = ck["seconds"], ck.get("peak_gb", float("nan"))
                print("fold %d: resumed from %s (no training)" % (k, path), flush=True)
            else:
                net, train_secs[k], peak, _ = train_net(mine, cfg, 1000 * int(cfg["seed"]) + k + 1,
                                                        "fold %d" % k, partial, fp)
                torch.save({"state": net.state_dict(), "cfg": cfg, "fingerprint": fp,
                            "seconds": train_secs[k], "peak_gb": peak}, path)
                if os.path.exists(partial):
                    os.remove(partial)
            peaks.append(peak)
            if ck is not None and "candidates" in ck and ck.get("reading") == reading:
                # The fold's held-out reading travels with its checkpoint: a resumed fold skips it.
                got, secs = [pd.DataFrame(ck["candidates"])], list(ck.get("read_seconds", []))
                print("fold %d: candidates resumed too (no reading)" % k, flush=True)
            else:
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()    # training's freed blocks stay reserved and split; reading needs whole ones
                got, secs = read_all(net, held, cfg, "fold %d" % k)
                torch.save({"state": net.state_dict(), "cfg": cfg, "fingerprint": fp,
                            "seconds": train_secs[k], "peak_gb": peak, "read_seconds": secs, "reading": reading,
                            "candidates": pd.concat(got, ignore_index=True).to_dict("list")}, path)
            read_secs += secs
            frames += [f.assign(fold=k) for f in got]
            info[k] = {"train_scans": len(mine), "held_scans": len(held), "train_seconds": train_secs[k]}
            pd.concat(frames).to_csv(os.path.join(out, "candidates_oof.csv"), index=False)
            del net
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        cand = pd.concat(frames, ignore_index=True)
        uids = [u for u, f in folds_of.items() if f in run_folds]
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
        led.number("train_seconds_per_fold", float(np.mean(list(train_secs.values()))), "s")
        led.number("read_seconds_per_scan", float(np.mean(read_secs)), "s")
        led.number("peak_memory_gb", float(np.nanmax(peaks)) if peaks else float("nan"))
        led.number("scans", len(uids), "count")
        meta = {"name": name, "version": a.version, "ten": bool(a.ten), "folds": run_folds, "cfg": cfg,
                "cpm": cpm, "cpm_low": lo, "cpm_high": hi, "sensitivity_out_of_100": sens,
                "fp_rates": list(froc.FP_RATES), "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
                "n_scans": res["n_scans"], "n_nodules": res["n_nodules"], "per_fold": info,
                "read_seconds_per_scan": float(np.mean(read_secs)), "left_out_unprepared": missing,
                "commit": ledger._commit(), "torch": torch.__version__,
                "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    return 0


def smoke(a):
    """6 scans (4 train, 2 read), 40 steps at batch 2: the shapes, the coordinates and the speed."""
    card.require_card(DEV, torch)
    limit_card()
    cfg = dict(CFG, steps=40, batch=2, warmup=4, log_every=10)
    ann, exc = reference()
    have = []
    for uid in sorted(lp.luna16_subsets()):
        npy, js = lp.paths_for(lp.out_dir(), uid)
        if os.path.exists(npy) and os.path.exists(js) and (ann.seriesuid == uid).any():
            have.append(uid)
        if len(have) == 6:
            break
    assert len(have) == 6, "fewer than 6 prepared LUNA16 scans with nodules"
    scans = [Scan(u, ann, exc, cfg) for u in have]
    mine, held = scans[:4], scans[4:]
    # The targets sit where the nodules are: the cube centred on a nodule has its ball at the middle.
    s = next(s for s in mine if len(s.nodules))
    c = np.rint(s.nodules[0, :3]).astype(int)
    img, tgt, wgt = cube_of(s, [int(v) - CUBE // 2 for v in c])
    mid = (CUBE // 2,) * 3
    assert tgt[mid] == 1 and wgt[mid] == 1, "target is not on the nodule's centre"
    print("cube %s image %s, target voxels %d (nodule radius %.1f mm: ball of about %d), weight-0 voxels %d, "
          "centre HU %.0f" % (s.uid[-12:], img.shape, int(tgt.sum()), s.nodules[0, 3],
                              round(4 / 3 * math.pi * s.nodules[0, 3] ** 3), int((wgt == 0).sum()),
                              float(lp.to_hu(img[mid]))))
    net, seconds, peak, per_step = train_net(mine, cfg, 0, "smoke")
    got, secs = read_all(net, held, cfg, "smoke")
    cand = pd.concat(got, ignore_index=True)
    print("shapes: window %s -> logits %s; %d candidates over %d scans"
          % ((1, 1, CUBE, CUBE, CUBE), tuple(net.eval()(torch.zeros((1, 1, CUBE, CUBE, CUBE), device=DEV))[0].shape),
             len(cand), len(held)))
    for sc, s_read in zip(held, secs):
        mine_c = cand[cand.seriesuid == sc.uid]
        print("scan %s: box %s, read in %.1f s, %d candidates" % (sc.uid[-12:], sc.shape, s_read, len(mine_c)))
        for r in ann[ann.seriesuid == sc.uid].itertuples():
            p = np.array([r.coordX, r.coordY, r.coordZ])
            d = (np.linalg.norm(mine_c[["coordX", "coordY", "coordZ"]].to_numpy() - p, axis=1).min()
                 if len(mine_c) else float("nan"))
            print("  nodule %.1f mm at (%.0f, %.0f, %.0f): nearest candidate %.1f mm away" % (
                r.diameter_mm, *p, d))
    res = froc.score(cand, ann, exc, [s.uid for s in held])
    print("scored with froc: CPM %.4f, TP %d, FP %d, nodules %d" % (res["cpm"], res["tp"], res["fp"], res["n_nodules"]))
    print("speed: %.3f seconds per step at batch %d, peak card %.2f GB; reading %s seconds a scan"
          % (per_step, cfg["batch"], peak, ", ".join("%.1f" % v for v in secs)))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--ten", action="store_true", help="the ten-fold run (fold = LUNA16 subset)")
    t.add_argument("--folds", type=int, nargs="+")
    t.add_argument("--steps", type=int)
    t.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(CFG)))
    t.add_argument("--extra-lidc", action="store_true",
                   help="also train on the LIDC-IDRI scans LUNA16 left out (never held out or scored)")
    t.add_argument("--extra-nlst", action="store_true",
                   help="also train on the LIDC-annot-NLST501 scans (nlst501.py; never held out or scored)")
    sub.add_parser("smoke")
    a = ap.parse_args()
    return train(a) if a.cmd == "train" else smoke(a)


if __name__ == "__main__":
    sys.exit(main())
