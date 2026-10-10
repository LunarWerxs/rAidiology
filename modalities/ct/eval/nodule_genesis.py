"""Models Genesis for stage 2: the false-alarm reducer's four stages learn to restore damaged lung CT first.

nodule_fpr.py's classifier starts from random weights and learns from LUNA16's training folds alone. Here its four
stages first learn a task that needs no label. Each 48 mm cube from the NLST pool (nlst_pool48.npy, written by
nlst_pool.py) is distorted (flipped, pixels shuffled in small windows, its intensity curve bent, regions painted
over with noise), and a light U-Net decoder with skips must rebuild the undistorted cube. The stages that learn
this are saved as encoder.pt, which nodule_fpr's Classifier loads as its starting point.

Models Genesis (Zhou et al., arXiv 2004.07882) finds that restoring such damage teaches a 3D net the anatomy a
later task needs. The rates are the official code's defaults.

2% of the NLST series, chosen by a stable hash of their SeriesInstanceUID, are never trained on. Their restoration
loss on a fixed 512 cubes is reported every 2,000 steps. A partial checkpoint is saved every 1,000 steps and a
rerun resumes from it.

    python modalities/ct/eval/nodule_genesis.py pretrain --version N [--steps 40000] [--batch 32]
        [--card-gb 3.0] [--seed 0] [--workers N]
    python modalities/ct/eval/nodule_genesis.py selftest   (processor only, a tiny pool in a temp folder)

Output: work/eval/lung/genesis-v<N>/ with encoder.pt and meta.json, and partial.pt while a run is unfinished.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import nodule_fpr as nf  # noqa: E402
from core import card, ckpt, ledger, paths, polite  # noqa: E402

DEV = nf.DEV
CUBE = 48
HELD_SHARE = 0.02
HELD_CUBES = 512
HELD_SEED = 1234
LOG_EVERY = 100
SYNC_EVERY = 4
PREFETCH = 3
CFG = {
    "widths": [24, 48, 96, 192], "groups": 8,
    "steps": 40000, "batch": 32, "lr": 1e-3, "weight_decay": 1e-5, "warmup": 1000,
    "held_every": 2000, "partial_every": 1000,
    "threads": 4, "queue": 6, "workers": 0,
}


def pool_dir():
    return os.path.join(paths.WORK, "eval", "lung")


def held_series(uid, share=HELD_SHARE):
    return int(hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8], 16) % 10000 < share * 10000


def load_pool(base):
    """(pool, train_rows, held_rows): the NLST pool's memory map and its row indices split by series."""
    with open(os.path.join(base, "nlst_pool48.json"), encoding="utf-8") as f:
        rows = int(json.load(f)["rows"])
    pool = np.load(os.path.join(base, "nlst_pool48.npy"), mmap_mode="r")
    if pool.dtype != np.uint8 or pool.ndim != 4 or tuple(pool.shape[1:]) != (CUBE,) * 3 or not 1 <= rows <= len(pool):
        raise SystemExit("the NLST pool in %s does not match its index" % base)
    index = pd.read_csv(os.path.join(base, "nlst_pool48.csv"), usecols=["row", "series_uid"])
    uid_of = dict(zip(index.row.astype(int).tolist(), index.series_uid.astype(str).tolist()))
    if any(r not in uid_of for r in range(rows)):
        raise SystemExit("nlst_pool48.csv lacks some of the pool's %d rows" % rows)
    held = np.array([held_series(uid_of[r]) for r in range(rows)], bool)
    return pool, np.flatnonzero(~held), np.flatnonzero(held)


# ---- the distortions (Models Genesis, official rates) -------------------------------------------

def flip_both(x, y, rng, p=0.4):
    """Each axis flipped with chance p, the target the same way."""
    for ax in range(3):
        if rng.random() < p:
            x, y = np.flip(x, ax), np.flip(y, ax)
    return x, y


def shuffle_pixels(x, rng, p=0.5, windows=1000):
    """With chance p, the pixels of 1,000 random small windows (each side up to a tenth of the cube) are shuffled."""
    if rng.random() >= p:
        return x
    x = x.copy()
    top = max(1, x.shape[0] // 10)
    sizes = rng.integers(1, top + 1, size=(windows, 3))
    starts = rng.integers(0, x.shape[0] - sizes + 1)
    for size, start in zip(sizes.tolist(), starts.tolist()):
        sl = tuple(slice(a, a + s) for a, s in zip(start, size))
        block = x[sl]
        x[sl] = rng.permutation(block.ravel()).reshape(block.shape)
    return x


def _bezier(c, t):
    n = len(c) - 1
    return sum(math.comb(n, i) * (1 - t) ** (n - i) * t ** i * c[i] for i in range(n + 1))


def intensity_curve(x, rng, p=0.9):
    """With chance p, the intensities go through a monotonic Bezier curve; half the time it is inverted."""
    if rng.random() >= p:
        return x
    inner = rng.random((2, 2))
    px = np.array([0.0, inner[0, 0], inner[1, 0], 1.0])
    py = np.array([0.0, inner[0, 1], inner[1, 1], 1.0])
    t = np.linspace(0.0, 1.0, 1000)
    bx, by = np.sort(_bezier(px, t)), np.sort(_bezier(py, t))
    if rng.random() < 0.5:
        by = by[::-1]
    return np.interp(x, bx, by).astype(np.float32)


def _box(rng, n, low, high):
    size = rng.integers(low, high, size=3)
    start = rng.integers(0, n - size + 1)
    return tuple(slice(a, a + s) for a, s in zip(start.tolist(), size.tolist()))


def paint(x, rng, p=0.9, outpaint=0.8):
    """With chance p: out-painting (keep up to five large boxes, noise elsewhere) or in-painting (noise in up to five
    small boxes), out-painting with chance `outpaint`."""
    if rng.random() >= p:
        return x
    n = x.shape[0]
    if rng.random() < outpaint:
        out = rng.random(x.shape, dtype=np.float32)
        for _ in range(int(rng.integers(1, 6))):
            sl = _box(rng, n, n // 3, n // 2 + 1)
            out[sl] = x[sl]
        return out
    out = x.copy()
    for _ in range(int(rng.integers(1, 6))):
        sl = _box(rng, n, 1, n // 4)
        out[sl] = rng.random(out[sl].shape, dtype=np.float32)
    return out


def distort(cube, rng):
    """(distorted, undistorted) float32 cubes in [0, 1]."""
    x, y = flip_both(cube, cube, rng)
    x = shuffle_pixels(x, rng)
    x = intensity_curve(x, rng)
    x = paint(x, rng)
    return np.clip(x, 0.0, 1.0).astype(np.float32), np.ascontiguousarray(y, np.float32)


# ---- the net -------------------------------------------------------------------------------------

def _block(c_in, c_out, groups):
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, padding=1), nn.GroupNorm(groups, c_out), nn.LeakyReLU(),
        nn.Conv3d(c_out, c_out, kernel_size=3, padding=1), nn.GroupNorm(groups, c_out), nn.LeakyReLU())


def _up(x):
    return F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)


class Genesis(nn.Module):
    """nodule_fpr's Classifier stages as the encoder, and a light decoder with skips back to 48 cubed."""

    def __init__(self, widths, groups):
        super().__init__()
        self.classifier = nf.Classifier(widths, groups, c_in=1, norm="group")
        c0, c1, c2, c3 = widths
        self.up3 = _block(c3 + c2, c2, groups)
        self.up2 = _block(c2 + c1, c1, groups)
        self.up1 = _block(c1 + c0, c0, groups)
        self.up0 = _block(c0, c0, groups)
        self.out = nn.Conv3d(c0, 1, kernel_size=1)

    def forward(self, x):
        feats = []
        for stage in self.classifier.stages:
            x = stage(x)
            feats.append(x)
        f1, f2, f3, x = feats
        x = self.up3(torch.cat([_up(x), f3], 1))
        x = self.up2(torch.cat([_up(x), f2], 1))
        x = self.up1(torch.cat([_up(x), f1], 1))
        x = self.up0(_up(x))
        return self.out(x)


# ---- the data and the training -------------------------------------------------------------------

def cube_batches(pool, rows, rng, batch):
    """Endless (xs, ys) training batches: each cube is drawn from `rows` of `pool` and distorted with `rng`. The same
    draws in the same order whichever process builds them."""
    while True:
        pick = rows[rng.integers(len(rows), size=batch)]
        xs = np.empty((batch, 1, CUBE, CUBE, CUBE), np.float32)
        ys = np.empty_like(xs)
        for b, r in enumerate(pick):
            xs[b, 0], ys[b, 0] = distort(np.asarray(pool[r], np.float32) / 255.0, rng)
        yield xs, ys


class Sampler:
    """Endless batches of (distorted, undistorted) training cubes, built by background threads."""

    def __init__(self, pool, rows, seed, cfg):
        self.pool, self.rows, self.batch = pool, rows, cfg["batch"]
        self.q = queue.Queue(maxsize=cfg["queue"])
        self.error, self.stop = None, threading.Event()
        for i in range(cfg["threads"]):
            threading.Thread(target=self._run, args=(np.random.default_rng([seed, i]),),
                             daemon=True, name="genesis-cubes").start()

    def _run(self, rng):
        try:
            batches = cube_batches(self.pool, self.rows, rng, self.batch)
            while not self.stop.is_set():
                xs, ys = next(batches)
                while not self.stop.is_set():
                    try:
                        self.q.put((xs, ys), timeout=0.5)
                        break
                    except queue.Full:
                        pass
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.q.put(None)

    def get(self):
        got = self.q.get()
        if got is None:
            raise self.error
        return got


class CubeStream(IterableDataset):
    """Endless batches for one worker process. It holds the pool's file path, never the memory map, so it pickles under
    Windows spawn as well as Linux fork. Each worker opens the pool itself and draws from [seed, its id], as thread k."""

    def __init__(self, path, rows, seed, batch):
        self.path, self.rows, self.seed, self.batch = path, rows, seed, batch

    def __iter__(self):
        k = get_worker_info().id
        pool = np.load(self.path, mmap_mode="r")
        yield from cube_batches(pool, self.rows, np.random.default_rng([self.seed, k]), self.batch)


def _as_tensors(batch):
    """A worker's (xs, ys) as torch tensors: DataLoader hands tensors between processes through shared memory."""
    xs, ys = batch
    return torch.from_numpy(xs), torch.from_numpy(ys)


class _Stop:
    """sampler.stop.set() shuts a ProcSampler's worker processes down."""

    def __init__(self, close):
        self.close = close

    def set(self):
        self.close()


class ProcSampler:
    """The same batches as Sampler, built by cfg["workers"] worker processes. Worker k draws the stream thread k would;
    DataLoader takes batch j from worker j % N, at that worker's position j // N."""

    def __init__(self, path, rows, seed, cfg):
        self.loader = DataLoader(CubeStream(path, rows, seed, cfg["batch"]), batch_size=None,
                                 num_workers=cfg["workers"], collate_fn=_as_tensors, persistent_workers=True,
                                 pin_memory=DEV.type == "cuda", prefetch_factor=PREFETCH)
        self.it = iter(self.loader)
        self.stop = _Stop(self.close)

    def get(self):
        xs, ys = next(self.it)
        return xs.numpy(), ys.numpy()

    def close(self):
        if self.it is not None:
            self.it._shutdown_workers()
            self.it = None


@torch.inference_mode()
def held_loss(net, held_arr):
    """Mean squared error on the held-out cubes, distorted the same way every time (a fixed seed)."""
    rng = np.random.default_rng(HELD_SEED)
    total, count = 0.0, 0
    for i in range(0, len(held_arr), 32):
        pairs = [distort(np.asarray(c, np.float32) / 255.0, rng) for c in held_arr[i:i + 32]]
        x = torch.from_numpy(np.stack([p[0] for p in pairs])[:, None]).to(DEV)
        y = torch.from_numpy(np.stack([p[1] for p in pairs])[:, None]).to(DEV)
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            logits = net(x)
        total += float(((torch.sigmoid(logits.float()) - y) ** 2).sum())
        count += y.numel()
    return total / count


def settings_of(cfg, seed, rows):
    return {"widths": cfg["widths"], "groups": cfg["groups"], "c_in": 1, "norm": "group",
            "steps": cfg["steps"], "batch": cfg["batch"], "pool_rows": rows, "seed": seed}


def peak_gb(prior):
    if DEV.type != "cuda":
        return float("nan")
    return max(prior, torch.cuda.max_memory_allocated() / 2 ** 30)


def run_pretrain(out, pool, train_rows, held_rows, cfg, seed, stop_at=None):
    """Train (or resume) the restoration net, then write encoder.pt and meta.json to `out`. Returns the run's
    numbers; None when stop_at stops it early (selftest only)."""
    if not len(train_rows):
        raise SystemExit("the NLST pool has no training series")
    rows = len(train_rows) + len(held_rows)
    settings = settings_of(cfg, seed, rows)
    fp = {**settings, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "warmup": cfg["warmup"],
          "held_share": HELD_SHARE}
    steps, batch = cfg["steps"], cfg["batch"]
    os.makedirs(out, exist_ok=True)
    partial = os.path.join(out, "partial.pt")
    held_idx = np.sort(np.random.default_rng(seed).choice(held_rows, min(HELD_CUBES, len(held_rows)), replace=False)) \
        if len(held_rows) else np.array([], np.int64)
    held_arr = np.asarray(pool[held_idx])

    torch.manual_seed(seed)
    net = Genesis(cfg["widths"], cfg["groups"]).to(DEV).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            fused=DEV.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: nf.lr_factor(s, cfg))
    start, curve, held_curve, prior_secs, prior_peak, sampler_seed = 0, [], [], 0.0, 0.0, seed
    ck = ckpt.load_partial(partial, fp)
    if ck is not None and 0 < ck.get("step", 0) < steps:
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, curve, held_curve = ck["step"], ck["curve"], ck["held_curve"]
        prior_secs, prior_peak = ck["seconds"], ck.get("peak_gb", 0.0)
        sampler_seed = seed * 1_000_003 + start
        print("genesis: resumed at step %d of %d (partial checkpoint)" % (start, steps), flush=True)
    del ck

    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True
    if cfg["workers"]:
        sampler = ProcSampler(pool.filename, train_rows, sampler_seed, cfg)
    else:
        sampler = Sampler(pool, train_rows, sampler_seed, cfg)
    pace = polite.Pace()
    t0 = time.time() - prior_secs
    block = log_t = time.time()
    acc, seen = 0.0, 0
    try:
        for step in range(start + 1, steps + 1):
            xs, ys = sampler.get()
            x = torch.from_numpy(xs).to(DEV)
            y = torch.from_numpy(ys).to(DEV)
            with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                logits = net(x)
            loss = F.mse_loss(torch.sigmoid(logits.float()), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc = acc + loss.detach()
            seen += 1

            if step % SYNC_EVERY == 0 or step == steps:
                float(acc)                      # waits for the card, so the pause below sits between real blocks
                pace.rest(time.time() - block)
                block = time.time()
            if step % LOG_EVERY == 0 or step == steps:
                now = time.time()
                mean = float(acc) / seen
                curve.append([step, mean])
                print("  genesis step %d of %d, restoration loss %.5f, %.0f cubes a second, using %.0f%% of the card"
                      % (step, steps, mean, seen * batch / max(now - log_t, 1e-9), 100 * pace.used()), flush=True)
                log_t, acc, seen = now, 0.0, 0
            if len(held_arr) and (step % cfg["held_every"] == 0 or step == steps):
                held_curve.append([step, held_loss(net, held_arr)])
                print("  genesis step %d: held-out loss %.5f on %d cubes"
                      % (step, held_curve[-1][1], len(held_arr)), flush=True)
            if step < steps and (step % cfg["partial_every"] == 0 or step == stop_at):
                ckpt.save_partial(partial, {"state": net.state_dict(), "opt": opt.state_dict(),
                                            "sched": sched.state_dict(), "step": step,
                                            "seconds": time.time() - t0, "peak_gb": peak_gb(prior_peak),
                                            "curve": curve, "held_curve": held_curve, "fingerprint": fp})
            if step == stop_at:
                return None
    finally:
        sampler.stop.set()

    if DEV.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    peak = peak_gb(prior_peak)
    os.makedirs(out, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.classifier.state_dict().items() if k.startswith("stages.")}
    torch.save({"state": state, "settings": settings}, os.path.join(out, "encoder.pt"))
    ckpt.clear_partial(partial)
    res = {"steps": steps, "loss_curve": curve, "held_curve": held_curve, "held_cubes": len(held_arr),
           "seconds": seconds, "peak_gb": peak, "cubes_a_second": steps * batch / max(seconds, 1e-9)}
    meta = {**res, "settings": settings, "train_rows": len(train_rows), "held_rows": len(held_rows),
            "speed": {"workers": cfg["workers"], "threads": cfg["threads"]},
            "commit": ledger._commit(), "torch": torch.__version__,
            "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return res


# ---- the card and the command line ---------------------------------------------------------------

def limit_card(gb, version):
    if DEV.type != "cuda":
        return
    card.lease(gb, "genesis v%d" % version)
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())


def pretrain(a):
    out = os.path.join(pool_dir(), "genesis-v%d" % a.version)
    if os.path.exists(os.path.join(out, "encoder.pt")):
        print("genesis-v%d is already trained (%s); delete that folder to train it again" % (a.version, out))
        return 0
    cfg = dict(CFG, steps=a.steps, batch=a.batch, workers=a.workers)
    card.require_card(DEV, torch)
    with ledger.run("experiment", "genesis pretrain v%d" % a.version) as led:
        limit_card(a.card_gb, a.version)
        pool, train_rows, held_rows = load_pool(pool_dir())
        rows = len(train_rows) + len(held_rows)
        led.settings_from({**settings_of(cfg, a.seed, rows), "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
                           "warmup": cfg["warmup"], "held_share": HELD_SHARE})
        res = run_pretrain(out, pool, train_rows, held_rows, cfg, a.seed)
        for s, v in res["loss_curve"]:
            led.point("loss", s, v)
        for s, v in res["held_curve"]:
            led.point("held_loss", s, v)
        led.number("loss_final", res["loss_curve"][-1][1])
        if res["held_curve"]:
            led.number("held_loss_final", res["held_curve"][-1][1])
        led.number("cubes_a_second", res["cubes_a_second"], "per s")
        led.number("peak_memory_gb", res["peak_gb"])
        led.number("steps", res["steps"], "count")
        led.number("rows", rows, "count")
    print("genesis-v%d: %d steps in %.0f s, peak card %.2f GB, written to %s"
          % (a.version, res["steps"], res["seconds"], res["peak_gb"], out), flush=True)
    return 0


def selftest(a=None):
    """Processor only, a tiny synthetic pool in a temp folder: every distortion keeps shape and the [0, 1] range, the
    decoder's output shape equals the input's, a run stopped after step 3 resumes there, and encoder.pt's stage keys
    load into a fresh nodule_fpr Classifier. Two worker processes draw the threads' batches, and a thread partial
    resumes under them."""
    import contextlib
    import gc
    import io
    import itertools
    import tempfile
    global DEV
    DEV = torch.device("cpu")
    rng = np.random.default_rng(0)
    cube = rng.random((CUBE,) * 3).astype(np.float32)

    def fits(name, out):
        assert out.shape == (CUBE,) * 3, (name, out.shape)
        assert out.min() >= 0.0 and out.max() <= 1.0, (name, out.min(), out.max())

    x, y = flip_both(cube, cube, rng, p=1.0)
    fits("flip x", x)
    fits("flip y", y)
    fits("shuffle", shuffle_pixels(cube, rng, p=1.0))
    fits("curve", intensity_curve(cube, rng, p=1.0))
    fits("out-paint", paint(cube, rng, p=1.0, outpaint=1.0))
    fits("in-paint", paint(cube, rng, p=1.0, outpaint=0.0))
    for _ in range(5):
        x, y = distort(cube, rng)
        fits("distort x", x)
        fits("distort y", y)

    net = Genesis([4, 4, 4, 4], 2)
    batch = torch.from_numpy(rng.random((2, 1, CUBE, CUBE, CUBE), dtype=np.float32))
    assert net(batch).shape == batch.shape, net(batch).shape

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        uids = {True: [], False: []}
        i = 0
        while min(len(v) for v in uids.values()) < 4:
            uid = "1.2.840.%d" % i
            i += 1
            if len(uids[held_series(uid)]) < 4:
                uids[held_series(uid)].append(uid)
        series_of_row = [u for u in uids[True] + uids[False] for _ in range(2)]
        rows = len(series_of_row)
        np.save(os.path.join(tmp, "nlst_pool48.npy"), rng.integers(0, 256, (rows + 4,) + (CUBE,) * 3, dtype=np.uint8))
        with open(os.path.join(tmp, "nlst_pool48.json"), "w", encoding="utf-8") as f:
            json.dump({"rows": rows}, f)
        pd.DataFrame({"row": range(rows + 4), "series_uid": series_of_row + ["unused"] * 4}).to_csv(
            os.path.join(tmp, "nlst_pool48.csv"), index=False)
        pool, train_rows, held_rows = load_pool(tmp)
        assert len(held_rows) == 8 and len(train_rows) == 8, (len(held_rows), len(train_rows))

        cfg = dict(CFG, widths=[4, 4, 4, 4], groups=2, steps=6, batch=2, warmup=2, queue=2, threads=2,
                   held_every=3, partial_every=3)
        ps = ProcSampler(os.path.join(tmp, "nlst_pool48.npy"), train_rows, 3, dict(cfg, workers=2))
        try:
            got = [ps.get() for _ in range(6)]
        finally:
            ps.stop.set()
        for k in (0, 1):
            batches = cube_batches(pool, train_rows, np.random.default_rng([3, k]), cfg["batch"])
            want = list(itertools.islice(batches, 3))
            for j, (wx, wy) in enumerate(want):
                xs, ys = got[2 * j + k]
                assert np.array_equal(xs, wx) and np.array_equal(ys, wy), ("worker", k, "batch", 2 * j + k)

        out = os.path.join(tmp, "genesis-v0")
        assert run_pretrain(out, pool, train_rows, held_rows, cfg, 0, stop_at=3) is None
        assert os.path.exists(os.path.join(out, "partial.pt")) and not os.path.exists(os.path.join(out, "encoder.pt"))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            res = run_pretrain(out, pool, train_rows, held_rows, cfg, 0)
        assert "resumed at step 3 of 6" in printed.getvalue(), printed.getvalue()
        assert res["steps"] == 6 and [s for s, _ in res["held_curve"]] == [3, 6], res["held_curve"]
        assert os.path.exists(os.path.join(out, "meta.json")) and not os.path.exists(os.path.join(out, "partial.pt"))

        enc = torch.load(os.path.join(out, "encoder.pt"), map_location="cpu", weights_only=False)
        s = enc["settings"]
        fresh = nf.Classifier(s["widths"], s["groups"], s["c_in"], s["norm"])
        missing, unexpected = fresh.load_state_dict(enc["state"], strict=False)
        assert not [k for k in missing if k.startswith("stages.")] and not unexpected, (missing, unexpected)
        assert any(k.startswith("stages.") for k in enc["state"]), "encoder.pt holds no stage weights"

        out = os.path.join(tmp, "genesis-v1")
        assert run_pretrain(out, pool, train_rows, held_rows, cfg, 0, stop_at=3) is None
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            res = run_pretrain(out, pool, train_rows, held_rows, dict(cfg, workers=2), 0)
        assert "resumed at step 3 of 6" in printed.getvalue(), printed.getvalue()
        assert res["steps"] == 6 and [s for s, _ in res["held_curve"]] == [3, 6], res["held_curve"]
        with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
            assert json.load(f)["speed"] == {"workers": 2, "threads": cfg["threads"]}
        del pool
        gc.collect()
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--steps", type=int, default=CFG["steps"])
    p.add_argument("--batch", type=int, default=CFG["batch"])
    p.add_argument("--card-gb", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"pretrain": pretrain, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
