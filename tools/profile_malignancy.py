"""Which operations take the card step of the lung "cancer or not" trainer.

  python tools/profile_malignancy.py [--variants base channels_last_3d no_autocast batchnorm_swap]
                                     [--set key=<json> ...] [--warmup 20] [--steps 100] [--profile-steps 20]

Builds the net and one real fold 0 training batch the way nodule_malignancy.py's train does (its CFG and
--set, nodule_fpr's build, net_input, sampler, loss, AdamW, clip and schedule; bfloat16 autocast), then for
each variant: warms up, times --steps training steps on that one batch with torch.cuda.synchronize() around
them (seconds per 100 steps), and runs torch.profiler over --profile-steps steps, printing the top 15 ops by
self CUDA time. Variants:
  base              the trainer's step as it is
  channels_last_3d  model and input in torch.channels_last_3d
  no_autocast       float32 throughout
  batchnorm_swap    every GroupNorm replaced by BatchNorm3d of the same channels: a timing probe only,
                    not a training change (the net has GroupNorm; asked for by name, not in the default)
One batch is reused for every step, so the loader costs nothing here: this is the card step alone. The batch is
drawn from 64 malignant and 64 benign fold 0 training lesions (not all of them) to keep the memory fill short.
Prints the GPU, torch, cudnn.benchmark and the card's free memory, since the card is shared: read the seconds
with that in mind. RAIDIOLOGY_PROFILE_CARD_GB (default 2.5) is the share leased, as 'malignancy profile'.

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
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "ct", "eval"))
import nodule_fpr as nf  # noqa: E402
import nodule_malignancy as nm  # noqa: E402
from core import card  # noqa: E402
from nodule import with_settings  # noqa: E402

FOLD = 0
LESIONS = 64        # per class
VARIANTS = ["base", "channels_last_3d", "no_autocast", "batchnorm_swap"]


def card_gb():
    return float(os.environ.get("RAIDIOLOGY_PROFILE_CARD_GB", "2.5"))


def lease_card():
    """Lease the share, then a hard ceiling at it (nodule_malignancy's way, own name and variable)."""
    gb = card_gb()
    # 2026-10-03: leased before the data is loaded and before CUDA starts: waiting jobs held 2.1-6.9 GB doing
    # nothing, and at 23:21 the machine ran out of committed memory (a training job crashed)
    card.lease(gb, "malignancy profile", short_min=10)
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())


def real_batch(cfg):
    """One fold 0 training batch (uint8 crops, labels) from nodule_malignancy's table and RAM sampler."""
    df, _ = nm.lesion_table(cfg)
    trn = df[df.fold != FOLD]
    cols = ["seriesuid", "coordX", "coordY", "coordZ"]
    pos = [tuple(r) for r in trn[trn.label == 1][cols].head(LESIONS).itertuples(index=False)]
    neg = [tuple(r) for r in trn[trn.label == 0][cols].head(LESIONS).itertuples(index=False)]
    scans = {u: nm.lidc_box(u) for u in {k[0] for k in pos + neg}}
    nm.fill_ram(scans, pos + neg)
    sampler = nm.RamSampler(scans, pos, (neg, []), {**cfg, "threads": 1, "queue": 1}, cfg["seed"], cfg["batch"], 0.0)
    try:
        imgs, labels = sampler.get()
    finally:
        sampler.stop.set()
    return imgs.to(nf.DEV), labels.to(nf.DEV)


def swap_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, nn.BatchNorm3d(child.num_channels).to(nf.DEV))
        else:
            swap_groupnorm(child)


class Step:
    """The trainer's training step (train_net's loop body) on one fixed batch, for one variant."""

    def __init__(self, cfg, variant, imgs, labels):
        torch.manual_seed(1000 * int(cfg["seed"]) + FOLD + 1)
        self.gen = torch.Generator(device=nf.DEV).manual_seed(1)
        self.cfg, self.labels = cfg, labels
        self.net = nf.build(cfg).train()
        if variant == "batchnorm_swap":
            swap_groupnorm(self.net)
        self.amp = variant != "no_autocast"
        self.last = variant == "channels_last_3d"
        if self.last:
            self.net = self.net.to(memory_format=torch.channels_last_3d)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=True)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: nf.lr_factor(s, cfg))
        self.x = nf.net_input(imgs, self.net.c_in)
        if self.last:
            self.x = self.x.contiguous(memory_format=torch.channels_last_3d)

    def __call__(self):
        n = self.x.shape[0]
        brightness = self.cfg["brightness"]
        gain = 1 + (torch.rand(n, 1, 1, 1, 1, device=nf.DEV, generator=self.gen) * 2 - 1) * brightness
        shift = (torch.rand(n, 1, 1, 1, 1, device=nf.DEV, generator=self.gen) * 2 - 1) * brightness
        with torch.autocast(nf.DEV.type, dtype=torch.bfloat16, enabled=self.amp):
            logits = self.net(self.x * gain + shift)
        loss = F.binary_cross_entropy_with_logits(logits.float(), self.labels)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 12.0)
        self.opt.step()
        self.sched.step()


def run_variant(cfg, variant, imgs, labels, a):
    print("\n=== %s ===" % variant, flush=True)
    step = Step(cfg, variant, imgs, labels)
    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    per100 = (time.time() - t) * 100 / a.steps
    print("%s: %.1f s per 100 steps" % (variant, per100), flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(a.profile_steps):
            step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15), flush=True)
    del step
    torch.cuda.empty_cache()
    return per100


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS[:3])
    ap.add_argument("--set", action="append", default=[], help="key=<json>, as nodule_malignancy train")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--profile-steps", type=int, default=20)
    a = ap.parse_args()
    cfg = with_settings(nm.CFG, a.set)
    card.require_card(nf.DEV, torch)
    if nf.DEV.type != "cuda":
        raise SystemExit("this profile times the card; RAIDIOLOGY_DEVICE=cpu has nothing to measure")
    lease_card()
    torch.backends.cudnn.benchmark = True       # as train_net sets it before its loop

    free, total = torch.cuda.mem_get_info()
    print("GPU: %s; torch %s (CUDA %s); cudnn.benchmark %s" % (
        torch.cuda.get_device_name(), torch.__version__, torch.version.cuda, torch.backends.cudnn.benchmark))
    print("card: %.2f GB free of %.2f GB; used by others and this process's lease: %.2f GB; own lease %.1f GB" % (
        free / 2 ** 30, total / 2 ** 30, (total - free) / 2 ** 30, card_gb()), flush=True)
    imgs, labels = real_batch(cfg)
    print("batch: %s crops, %d malignant" % (tuple(imgs.shape), int(labels.sum())), flush=True)

    times = {v: run_variant(cfg, v, imgs, labels, a) for v in a.variants}
    print("\nsummary (s per 100 steps)")
    for v, s in times.items():
        print("%s: %.1f" % (v, s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
