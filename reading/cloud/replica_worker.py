"""A stand-in for reading/cloud/handler.py that needs no network volume, no weights and no scan: it builds the CT
segmenter's five networks with random weights (the same layers and sizes as version 2: 31M parameters each, 118
outputs) and reads a made-up volume of the size asked for, through the same windows, half precision and labelling
step as segmenter.probabilities and labels_on_grid. The arithmetic a reading does depends on the volume's size, not on
the weights or the pictures, so its seconds are a real reading's seconds; what it cannot show is the time to read
the weight files from storage (1.8 s from a pod's disk, measured 2026-10-10), which is reported as not measured.

It exists to measure a RunPod Serverless cold start and the cents per scan on an account that cannot hold a network
volume. Nothing leaves this machine but this file's text. The template runs it from an environment variable
(runpod_serverless.py replica-endpoint), after installing the runpod package and noting how long that took.

Input:  {"shape": [222, 222, 205]}     the scan's grid at 1.5 mm (at most 512 a side)
Output: {"seconds": {...}, "worker": {...}, "windows": n, "gpu": ...}
"""
import math
import os
import time

BOOT = time.time()
import numpy as np  # noqa: E402
import runpod  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

IMPORTED = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HALF = torch.float16 if DEV.type == "cuda" else None
WIDTHS, HEADS, N_OUT, PATCH, FOLDS = [32, 64, 128, 256, 320, 320], 3, 118, 128, 5    # version 2's meta.json
LOW_HU, HIGH_HU, HU_SCALE, AIR = -1000.0, 1500.0, 500.0, -1024                         # segmenter.py, prepare.py


class Stage(nn.Sequential):
    def __init__(self, c_in, c_out, stride):
        super().__init__(
            nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, inplace=True))


class UNet(nn.Module):
    """segmenter.UNet without the prior and the float32 heads, which version 2 does not use."""

    def __init__(self, widths, n_out, heads):
        super().__init__()
        self.down = nn.ModuleList([Stage(1 if i == 0 else widths[i - 1], w, 1 if i == 0 else 2) for i, w in enumerate(widths)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2) for i in range(len(widths) - 1)])
        self.merge = nn.ModuleList([Stage(2 * widths[i], widths[i], 1) for i in range(len(widths) - 1)])
        self.heads = nn.ModuleList([nn.Conv3d(widths[i], n_out, 1) for i in range(heads)])

    def forward(self, x):
        skips = []
        for stage in self.down:
            x = stage(x)
            skips.append(x)
        for i in range(len(self.up) - 1, -1, -1):
            x = self.merge[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.heads[0](x)


def starts(size, patch, overlap=0.5):
    if size <= patch:
        return [0]
    n = int(math.ceil((size - patch) / (patch * (1.0 - overlap)))) + 1
    return sorted(set(int(round(v)) for v in np.linspace(0, size - patch, n)))


def bell(n):
    d = torch.arange(n, device=DEV, dtype=torch.float32) - (n - 1) / 2.0
    g = torch.exp(-(d ** 2) / (2 * (n / 8.0) ** 2))
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).clamp_min(0.01)


def average_down(ct):
    """prepare.average_down: odd edges padded with air, then the mean of each 2 x 2 x 2 block."""
    ct = np.pad(ct, [(0, s % 2) for s in ct.shape], constant_values=AIR).astype(np.float32)
    x, y, z = (s // 2 for s in ct.shape)
    b = ct.reshape(x, 2, y, 2, z, 2).transpose(0, 2, 4, 1, 3, 5).reshape(x, y, z, 8)
    return np.rint(b.mean(axis=3)).astype(np.int16)


@torch.no_grad()
def probabilities(models, ct_lo):
    """segmenter.probabilities with overlap 0.5 and no trim, the running total on the card."""
    ct = torch.as_tensor(np.ascontiguousarray(ct_lo), device=DEV)
    shape = tuple(ct.shape)
    x = ct.float().clamp_(LOW_HU, HIGH_HU).div_(HU_SCALE)
    pad = [max(PATCH - s, 0) for s in shape]
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=LOW_HU / HU_SCALE)
    full = tuple(x.shape)
    acc = torch.zeros((N_OUT,) + full, device=DEV, dtype=torch.float16)
    weight = torch.zeros(full, device=DEV, dtype=torch.float32)
    w = bell(PATCH)
    n = 0
    for a in starts(full[0], PATCH):
        for b in starts(full[1], PATCH):
            for c in starts(full[2], PATCH):
                window = x[a:a + PATCH, b:b + PATCH, c:c + PATCH][None, None].contiguous()
                prob = None
                for m in models:
                    with torch.autocast(DEV.type, dtype=HALF, enabled=HALF is not None):
                        logits = m(window)
                    p = F.softmax(logits, dim=1, dtype=torch.float32)[0]
                    prob = p if prob is None else prob.add_(p)
                acc[:, a:a + PATCH, b:b + PATCH, c:c + PATCH] += prob.mul_(w).to(torch.float16)
                weight[a:a + PATCH, b:b + PATCH, c:c + PATCH] += w * len(models)
                n += 1
    return acc[:, :shape[0], :shape[1], :shape[2]], weight[:shape[0], :shape[1], :shape[2]], n


@torch.no_grad()
def labels_on_grid(acc, weight, shape, slab=8):
    """segmenter.labels_on_grid on a card (whole slabs): probabilities interpolated up to 1.5 mm, then the likeliest."""
    out = torch.zeros(shape, device=DEV, dtype=torch.uint8)
    nz = acc.shape[3]
    for a in range(0, nz, slab):
        b = min(a + slab, nz)
        lo, hi = max(a - 1, 0), min(b + 1, nz)
        part = acc[:, :, :, lo:hi].float() / weight[None, :, :, lo:hi]
        best = F.interpolate(part[None], scale_factor=2, mode="trilinear", align_corners=False)[0].argmax(dim=0)
        z0, z1 = a * 2, min(b * 2, shape[2])
        oz = (a - lo) * 2
        out[:, :, z0:z1] = best[:shape[0], :shape[1], oz:oz + z1 - z0].to(torch.uint8)
    return out


torch.manual_seed(0)
MODELS = [UNet(WIDTHS, N_OUT, HEADS).to(DEV).eval() for _ in range(FOLDS)]
PARAMS = sum(p.numel() for p in MODELS[0].parameters())
if DEV.type == "cuda":
    torch.cuda.synchronize()
READY = time.time()
SERVED = 0


def started():
    """Seconds the start command spent before Python: installing runpod (RAID_T0 to RAID_T1, set by the template)."""
    try:
        return round(float(os.environ["RAID_T1"]) - float(os.environ["RAID_T0"]), 3)
    except (KeyError, ValueError):
        return None


def handler(job):
    global SERVED
    shape = [int(v) for v in (job.get("input") or {}).get("shape", [222, 222, 205])]
    if len(shape) != 3 or not all(16 <= v <= 512 for v in shape):
        return {"error": "shape must be three sizes from 16 to 512"}
    first = SERVED == 0
    SERVED += 1
    t0 = time.time()
    rng = np.random.default_rng(len(shape) + sum(shape))
    ct = rng.normal(0.0, 300.0, size=shape).astype(np.float32)    # made-up Hounsfield units; only the size matters
    t1 = time.time()
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    acc, weight, windows = probabilities(MODELS, average_down(ct))
    pred = labels_on_grid(acc, weight, tuple(shape))
    found = int((torch.bincount(pred.reshape(-1).long(), minlength=N_OUT)[1:] > 0).sum())
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    return {"seconds": {"make_volume": round(t1 - t0, 3), "read": round(t2 - t1, 3)},
            "worker": {"pip_install_runpod": started(), "imports": round(IMPORTED - BOOT, 3),
                       "build_models": round(READY - IMPORTED, 3), "first_request_on_this_worker": first,
                       "alive_s": round(time.time() - BOOT, 1)},
            "shape": shape, "windows": windows, "structures_drawn": found, "params_per_model": PARAMS,
            "peak_card_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if DEV.type == "cuda" else None,
            "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
            "weights_are_random": True}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
