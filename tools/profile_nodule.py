"""Which operations take the card step of the lung nodule detector's stage 1 trainer.

  python tools/profile_nodule.py [--variants base channels_last_3d no_autocast batchnorm_swap]
                                 [--set key=<json> ...] [--warmup 20] [--steps 100] [--rounds 3]
                                 [--profile-steps 20] [--scans 4]

Builds the net and one real fold 0 training batch the way nodule.py's train_net does (its CFG and --set,
build, Sampler, loss_of, AdamW, gradient clipping, lr_factor schedule; bfloat16 autocast), then for each
variant: warms up, then times --steps training steps on that one batch with torch.cuda.synchronize() around
them (seconds per 100 steps, peak card memory), then runs torch.profiler over --profile-steps steps, printing
the top 15 ops by self CUDA time. Variants:
  base              the trainer's step as it is
  channels_last_3d  model and input in torch.channels_last_3d
  no_autocast       float32 throughout
  batchnorm_swap    every InstanceNorm3d(c, affine=True) replaced by BatchNorm3d(c): a timing probe only,
                    not a training change (asked for by name, not in the default)
Rounds (2026-10-04): the card is shared, and timing the variants one after another let other jobs ramping up
land on the later variants (base 5.8 s per 100 steps, then 6.2, 37.6 and 22.0: unreadable). So every variant
is built and warmed up once and kept alive (net, optimizer, batch) across rounds, and then --rounds (default 3)
times every variant in turn: in the order given in round 1, reversed in round 2, as given in round 3, and so
on (ABBA), so a slow stretch of the card falls on every variant alike. One line per round and variant. The
summary gives each variant's median over rounds with its ratio to base's median, and each round's ratio to
base in that round, which is the fair one on a shared card. torch.profiler runs once per variant, after the
rounds. With --rounds 1 the seconds are the old run's, plus the ratio. Peak card memory is the largest over
the rounds; all the variants' nets stay on the card together, so it is a little higher than one at a time.
One batch is reused for every step, so the loader costs nothing here: this is the card step alone. The batch
is cut from the first --scans (default 4) fold 0 training scans that have a nodule, so only those few boxes
are opened (their boxes are read memory-mapped, one batch of cubes). Today's v3 run is
  python tools/profile_nodule.py --set batch=2 --set steps=64000 --set min_radius_mm=3 --set max_radius_mm=5
Prints the GPU, torch, cudnn.benchmark and the card's free memory, since the card is shared: read the seconds
with that in mind. RAIDIOLOGY_PROFILE_CARD_GB (default 4.0) is the share leased, as 'nodule profile'.

It changes no training code and writes no file.
"""
import argparse
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200; read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "ct", "eval"))
import nodule as nd  # noqa: E402
from core import card  # noqa: E402

FOLD = 0
VARIANTS = ["base", "channels_last_3d", "no_autocast", "batchnorm_swap"]


def card_gb():
    return float(os.environ.get("RAIDIOLOGY_PROFILE_CARD_GB", "4.0"))


def lease_card():
    """Lease the share, then a hard ceiling at it (nodule.py's limit_card, own name and variable)."""
    gb = card_gb()
    # 2026-10-03: leased before the data is loaded and before CUDA starts: waiting jobs held 2.1-6.9 GB doing
    # nothing, and at 23:21 the machine ran out of committed memory (a training job crashed)
    card.lease(gb, "nodule profile", short_min=10)
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())


def real_batch(cfg, n_scans):
    """One fold 0 training batch (uint8: image, target, weight) from nd.Sampler on a few training scans."""
    folds_of, _ = nd.scan_folds(False)
    ann, exc = nd.reference()
    have = set(ann.seriesuid)
    uids = [u for u in nd.training_uids(folds_of, FOLD, []) if u in have][:n_scans]
    scans = [nd.Scan(u, ann, exc, cfg) for u in uids]
    sampler = nd.Sampler(scans, {**cfg, "queue": 1}, 1000 * int(cfg["seed"]) + FOLD + 1, cfg["batch"])
    return sampler.get().to(nd.DEV), len(scans)       # its thread blocks on the full queue; it is a daemon


def swap_instancenorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.InstanceNorm3d):
            setattr(module, name, nn.BatchNorm3d(child.num_features).to(nd.DEV))
        else:
            swap_instancenorm(child)


class Step:
    """The trainer's training step (train_net's loop body) on one fixed batch, for one variant."""

    def __init__(self, cfg, variant, batch):
        seed = 1000 * int(cfg["seed"]) + FOLD + 1
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=nd.DEV).manual_seed(seed)
        self.cfg = cfg
        self.net = nd.build(cfg).train()
        if variant == "batchnorm_swap":
            swap_instancenorm(self.net)
        self.amp = variant != "no_autocast"
        self.last = variant == "channels_last_3d"
        if self.last:
            self.net = self.net.to(memory_format=torch.channels_last_3d)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                     fused=nd.DEV.type == "cuda")
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: nd.lr_factor(s, cfg))
        self.x = batch[:, 0:1].float() / 255.0
        self.y, self.w = batch[:, 1:2].float(), batch[:, 2:3].float()
        if self.last:
            self.x = self.x.contiguous(memory_format=torch.channels_last_3d)

    def __call__(self):
        cfg = self.cfg
        n = self.x.shape[0]
        gain = 1 + (torch.rand(n, 1, 1, 1, 1, device=nd.DEV, generator=self.gen) * 2 - 1) * cfg["brightness"]
        shift = (torch.rand(n, 1, 1, 1, 1, device=nd.DEV, generator=self.gen) * 2 - 1) * cfg["brightness"]
        with torch.autocast(nd.DEV.type, dtype=torch.bfloat16, enabled=self.amp):
            outs = self.net(self.x * gain + shift)
        loss = nd.loss_of(outs, self.y, self.w, cfg)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg["clip"])
        self.opt.step()
        self.sched.step()


def warm_up(cfg, variant, batch, a):
    print("=== %s: build and warm up ===" % variant, flush=True)
    step = Step(cfg, variant, batch)
    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    return step


def time_round(step, a):
    """Seconds per 100 steps over --steps steps, and the peak card memory in GB, for one variant in one round."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) * 100 / a.steps, torch.cuda.max_memory_allocated() / 2 ** 30


def profile_variant(variant, step, a):
    print("\n=== %s: profile ===" % variant, flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(a.profile_steps):
            step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15), flush=True)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS[:3])
    ap.add_argument("--set", action="append", default=[], help="key=<json>, as nodule.py train")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3, help="timing rounds over all variants (ABBA order)")
    ap.add_argument("--profile-steps", type=int, default=20)
    ap.add_argument("--scans", type=int, default=4, help="fold 0 training scans the batch is cut from")
    a = ap.parse_args()
    cfg = nd.with_settings(nd.CFG, a.set)
    card.require_card(nd.DEV, torch)
    if nd.DEV.type != "cuda":
        raise SystemExit("this profile times the card; RAIDIOLOGY_DEVICE=cpu has nothing to measure")
    lease_card()
    torch.backends.cudnn.benchmark = True       # as train_net sets it before its loop

    free, total = torch.cuda.mem_get_info()
    print("GPU: %s; torch %s (CUDA %s); cudnn.benchmark %s" % (
        torch.cuda.get_device_name(), torch.__version__, torch.version.cuda, torch.backends.cudnn.benchmark))
    print("card: %.2f GB free of %.2f GB; used by others and this process's lease: %.2f GB; own lease %.1f GB" % (
        free / 2 ** 30, total / 2 ** 30, (total - free) / 2 ** 30, card_gb()), flush=True)
    batch, n = real_batch(cfg, a.scans)
    print("batch: %s from %d scans, %d target voxels" % (tuple(batch.shape), n, int(batch[:, 1].sum())), flush=True)

    steps = {v: warm_up(cfg, v, batch, a) for v in a.variants}
    times = {v: [] for v in a.variants}
    peaks = {v: 0.0 for v in a.variants}
    for r in range(1, a.rounds + 1):
        order = a.variants if r % 2 else a.variants[::-1]
        for v in order:
            per100, peak = time_round(steps[v], a)
            times[v].append(per100)
            peaks[v] = max(peaks[v], peak)
            print("round %d  %s: %.1f s per 100 steps" % (r, v, per100), flush=True)
    for v in a.variants:
        profile_variant(v, steps[v], a)

    print("\nsummary (median over %d rounds, s per 100 steps; ratio to base)" % a.rounds)
    base = median(times["base"]) if "base" in times else None
    for v in a.variants:
        m = median(times[v])
        print("%s: %.1f%s, peak card %.2f GB" % (
            v, m, "  ratio %.2f" % (m / base) if base else "", peaks[v]))
    if "base" in times:
        print("ratio to base in each round (fair on a shared card)")
        for v in a.variants:
            print("%s: %s" % (v, "  ".join("%.2f" % (t / b) for t, b in zip(times[v], times["base"]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
