"""Time the chest reader's training step: ChestNet at the recipe's widths, forward, BCE, backward and AdamW, bfloat16
autocast, on random pixels, after a warm-up. The A/B bench of docs/efficiency, "The chest trunk's step".

    python tools/chest_bench.py [--steps 50] [--warmup 10] [--batch 64] [--px 384] [--outputs 15]

The arm comes from the environment: RAIDIOLOGY_CHEST_LAYOUT (plain, the default, or channels_last) and
RAIDIOLOGY_CHEST_CUDNN_BENCH (0, the default, or 1). Prints `seconds_a_step <s>` and the card's peak. It leases the
card unless it runs under `core/card.py hold` (RAIDIOLOGY_CARD_HELD).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
sys.path.insert(0, ROOT)

LEASE_GB = 2.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--px", type=int, default=384)
    ap.add_argument("--outputs", type=int, default=15)
    a = ap.parse_args()
    layout = os.environ.get("RAIDIOLOGY_CHEST_LAYOUT", "plain")
    if layout not in ("plain", "channels_last"):
        raise SystemExit("RAIDIOLOGY_CHEST_LAYOUT is plain or channels_last, not %r" % layout)
    search = os.environ.get("RAIDIOLOGY_CHEST_CUDNN_BENCH", "0") == "1"
    import torch
    import reader
    from core import card
    if not torch.cuda.is_available():
        raise SystemExit("the bench needs the card")
    card.lease(LEASE_GB, "chest bench", short_min=5)
    torch.backends.cudnn.benchmark = search
    dev = torch.device("cuda")
    mf = torch.channels_last if layout == "channels_last" else torch.contiguous_format
    torch.manual_seed(0)
    net = reader.ChestNet(reader.CFG["widths"], reader.CFG["dropout"], reader.CFG["stem_stride"], a.outputs)
    net = net.to(dev).to(memory_format=mf)
    opt = torch.optim.AdamW(net.parameters(), lr=reader.CFG["lr"], weight_decay=reader.CFG["weight_decay"])
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(a.batch, 1, a.px, a.px, device=dev, generator=g).contiguous(memory_format=mf)
    y = (torch.rand(a.batch, a.outputs, device=dev, generator=g) < 0.3).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    net.train()
    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    s = (time.perf_counter() - t0) / a.steps
    # GiB, the unit the card's leases count (until 2026-10-05 15:37 this printed allocated / 10^9 alone); a ceiling
    # counts what is reserved
    print("layout %s, cudnn search %s, %d px, batch %d, %d steps: seconds_a_step %.5f; peak card memory %.2f GiB "
          "allocated, reserved_gib %.2f"
          % (layout, "on" if search else "off", a.px, a.batch, a.steps, s, torch.cuda.max_memory_allocated() / 2 ** 30,
             torch.cuda.max_memory_reserved() / 2 ** 30), flush=True)


if __name__ == "__main__":
    main()
