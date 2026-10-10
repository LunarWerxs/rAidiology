"""Stage 2 of the nodule detector, timed: how fast crops reach its network, in training and in scoring.

    python tools/bench_fpr_crops.py feed  [--rev REV] [--seconds 20] [--output FILE]
    python tools/bench_fpr_crops.py score [--rev REV] [--points 20000] [--output FILE]
    python tools/bench_fpr_crops.py step  [--version 2] [--layout plain|channels_last_3d] [--warmup 10] [--steps 50]
                                          [--device cuda|cpu] [--rev REV]

feed   the training Sampler on the processor (six threads, batches of 32): no card needed
score  score_points on fold 0's first --points candidates_V2 points with fpr-v1's fold-0 network,
       under a 1.5 GB card lease
step   fpr-v<version>'s training step (forward, loss, backward, clip, AdamW) on fixed random crops of the real
       shape and batch, in the plain memory order or channels_last_3d; prints seconds_a_step, steps_per_second
       and first_loss. cuda runs under stage 2 training's 3.5 GB share (RAIDIOLOGY_FPR_CARD_GB overrides); cpu takes none (a smoke run)

--rev runs modalities/ct/eval/nodule_fpr.py as it was at a git revision (copied into work/, which git
ignores); without it, the working tree. Each prints `crops_per_second N`. --output writes what the
network was given or gave (feed: the bytes of the first three batches one thread builds from seed 7;
score: every probability to four places), so tools/ab.py can check two versions give the same thing:

    python tools/ab.py "stage 2: ..." --a "python tools/bench_fpr_crops.py feed --rev <old> --output work/fpr_feed.bin"
        --b "python tools/bench_fpr_crops.py feed --output work/fpr_feed.bin" --output work/fpr_feed.bin
        --metric crops_per_second="crops_per_second ([0-9.]+)" --ignore-wall --repeat 5

step's A/B (first_loss is compared by hand, within 1%: the two layouts are not bit-identical in bfloat16):

    python tools/ab.py "stage 2 step: channels-last 3D" --a "python tools/bench_fpr_crops.py step --layout plain"
        --b "python tools/bench_fpr_crops.py step --layout channels_last_3d"
        --metric steps_per_second="steps_per_second ([0-9.]+)" --ignore-wall --repeat 5
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "modalities", "ct", "eval")
sys.path[:0] = [EVAL, ROOT]


def module(rev):
    """nodule_fpr at `rev` (or the working tree)."""
    path = os.path.join(EVAL, "nodule_fpr.py")
    if rev:
        os.makedirs(os.path.join(ROOT, "work"), exist_ok=True)
        path = os.path.join(ROOT, "work", "_nodule_fpr_%s.py" % rev.replace("/", "_"))
        src = subprocess.run(["git", "show", rev + ":modalities/ct/eval/nodule_fpr.py"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        with open(path, "wb") as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location("nodule_fpr_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step(a):
    """Time nodule_fpr.train_net's step (its loop body, lines for lines) on fixed random crops."""
    import json

    import torch
    import torch.nn.functional as F

    fpr = module(a.rev)
    fpr.DEV = torch.device(a.device)
    dev = fpr.DEV
    with open(os.path.join(fpr.out_root(a.version), "meta.json")) as f:
        cfg = dict(fpr.CFG, **json.load(f)["cfg"])        # a version's saved settings over the defaults
    if dev.type == "cuda":
        os.environ.setdefault("RAIDIOLOGY_FPR_CARD_GB", "3.5")
        fpr.card.require_card(dev, torch)
        fpr.limit_fpr_card()
    last = a.layout == "channels_last_3d"
    fmt = torch.channels_last_3d if last else torch.contiguous_format

    torch.manual_seed(7)
    net = fpr.build(cfg).train()
    if last:
        net = net.to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=dev.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: fpr.lr_factor(s, cfg))
    torch.backends.cudnn.benchmark = True

    n, side = cfg["batch"], fpr.CUBE
    g = torch.Generator().manual_seed(7)
    imgs = torch.randint(0, 256, (n, cfg.get("context", 1) * (1 + int(cfg.get("mirror", 0))), side, side, side),
                         generator=g, dtype=torch.uint8)
    labels = (torch.arange(n) % 2).float()                # half positive
    imgs = imgs.to(dev).contiguous(memory_format=fmt)
    labels = labels.to(dev)
    gen = torch.Generator(device=dev).manual_seed(7)

    def one():
        x = fpr.net_input(imgs, net.c_in)
        gain = 1 + (torch.rand(n, 1, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * cfg["brightness"]
        shift = (torch.rand(n, 1, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * cfg["brightness"]
        x = (x * gain + shift).contiguous(memory_format=fmt)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = net(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
        opt.step()
        sched.step()
        return loss.detach()

    first = float(one())
    for _ in range(a.warmup - 1):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = (time.time() - t0) / a.steps
    print("seconds_a_step %.5f" % secs)
    print("steps_per_second %.4f" % (1 / secs))
    print("first_loss %.6g" % first, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["feed", "score", "step"])
    ap.add_argument("--rev")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--output")
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--layout", choices=["plain", "channels_last_3d"], default="plain")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args()
    if a.mode == "step":
        return step(a)
    if a.mode == "score":
        os.environ.setdefault("RAIDIOLOGY_FPR_CARD_GB", "1.5")
    fpr = module(a.rev)
    import torch

    cfg = dict(fpr.CFG, batch=32)
    folds_of, _ = fpr.scan_folds(False)
    ann, exc = fpr.reference()
    scans = {u: fpr.Scan(u, ann, exc, cfg) for u in folds_of}
    cand = fpr.candidates_frame(os.path.join(fpr.LUNA_DIR, "candidates_V2.csv"), folds_of)

    if a.mode == "feed":
        fpr.DEV = torch.device("cpu")
        train = {u: scans[u] for u, f in folds_of.items() if f != 0}
        pos, neg, hard = fpr.build_training_data(ann, exc, cand, list(train), cfg)
        if a.output:
            s = fpr.Sampler(train, pos, (neg, hard), dict(cfg, threads=1), 7, 32)
            with open(a.output, "wb") as f:
                for _ in range(3):
                    imgs, labels = s.get()
                    f.write(imgs.numpy().tobytes() + labels.numpy().tobytes())
            s.stop.set()
        s = fpr.Sampler(train, pos, (neg, hard), cfg, 0, 32)
        s.get()                                   # threads started, first batch built
        n, t0 = 0, time.time()
        while time.time() - t0 < a.seconds:
            s.get()
            n += 32
        s.stop.set()
        rate = n / (time.time() - t0)
    else:
        fpr.card.require_card(fpr.DEV, torch)
        fpr.limit_fpr_card()
        ck = torch.load(os.path.join(fpr.out_root(1), "fold0.pt"), map_location=fpr.DEV, weights_only=False)
        net = fpr.build(ck["cfg"])
        net.load_state_dict(ck["state"])
        frame = cand[cand.fold == 0].head(a.points).reset_index(drop=True)
        held = {u: scans[u] for u in frame.seriesuid.unique()}
        fpr.score_points(net, held, frame.head(640))      # card warm, kernels chosen
        p2, rate = fpr.score_points(net, held, frame)
        if a.output:
            with open(a.output, "w") as f:
                f.write("\n".join("%.4f" % v for v in p2))
    print("crops_per_second %.1f" % rate, flush=True)


if __name__ == "__main__":
    main()
