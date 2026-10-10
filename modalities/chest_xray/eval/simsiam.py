"""A self-taught chest trunk (docs/efficiency idea 40): SimSiam (Chen and He, 2021) on every frontal chest X-ray we may
use, no labels, then taix.py starts from its trunk (INITS chest-ssl-v1, chest-ssl-v1-e15).

  pool [--jobs N]                 processor only: one uint8 288 px array of the pool, ssl_pool_288.npy, with its csv
                                  (row, source, id, patient) and json (counts, build time, unreadable pictures)
  bench [--batch B] [--steps 30]  B-sized training steps on random pool rows: one line with seconds_a_step, reserved_gib,
                                  steps_an_epoch (RAIDIOLOGY_SSL_CARD_GB share of the card, default 7.0)
  train --version V --epochs E [--batch 128] [--until K]
                                  trunk_e15.pt after epoch 15, trunk.pt after epoch E, state.pt (resumable) every epoch
                                  and mid-epoch; --until K stops after epoch K (the cosine still spans E); exit 3 on collapse
  selftest                        tiny fake data, seconds

Not called ssl.py: a module of that name beside taix.py and reader.py would shadow the standard ssl for every script
run from this folder (renamed 2026-10-05 from the builder's ssl.py).

Usage: python simsiam.py pool --jobs 8
       python simsiam.py train --version 1 --epochs 50
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from core import paths  # noqa: E402
from core.rows import FileRows  # noqa: E402

CACHE = os.path.join(paths.WORK, "eval", "chest")
POOL = os.path.join(CACHE, "ssl_pool_288.npy")
POOL_CSV = os.path.join(CACHE, "ssl_pool_288.csv")
POOL_JSON = os.path.join(CACHE, "ssl_pool_288.json")
PUBLIC = os.path.join(paths.DATASETS, "public")
NIH_DIR = os.path.join(PUBLIC, "nih-chestxray14")
SIDE = 288                       # the pool's side
VIEW = 256                       # the views' side
TAIX_SIDE = 384
TAIX_U8 = os.path.join(CACHE, "taix_384.u8")
TAIX_CSV = os.path.join(CACHE, "taix_384.csv")
SEED = 20261005
KEEP_EPOCH = 15                  # trunk_e15.pt
CARD_GB = float(os.environ.get("RAIDIOLOGY_SSL_CARD_GB", "7.0"))
PROJ_DIM, PRED_DIM = 2048, 512
RECIPE = {"momentum": 0.9, "weight_decay": 1e-4, "lr_per_256": 0.05, "proj": PROJ_DIM, "pred": PRED_DIM,
          "crop": [0.35, 1.0], "ratio": [0.75, 4 / 3], "rot_deg": 10.0, "jitter_p": 0.8, "brightness": 0.4,
          "contrast": [0.6, 1.4], "view": VIEW, "seed": SEED}
COLLAPSE_BELOW = 0.2 / math.sqrt(PROJ_DIM)
CHUNK_CACHED, CHUNK_DECODED = 256, 32


def out_dir(version):
    return os.path.join(CACHE, "ssl-v%d" % version)


def _chest_prepare():
    import prepare                                   # chest_xray/eval/prepare.py (it loads the X-ray module by path)
    return prepare


def _xray():
    return _chest_prepare()._xray_prepare()


# ---------------------------------------------------------------- the pool

def guard_nih(names, test_names):
    """Stops when any chosen NIH picture is on the test list."""
    hit = sorted(set(names) & set(test_names))
    if hit:
        raise SystemExit("%d chosen NIH pictures are on test_list.txt (first: %s); the pool is not built" % (len(hit), hit[0]))


def guard_taix(splits, rows):
    """Stops when any chosen TAIX-Ray row's Split is not 'train'."""
    bad = [int(r) for r in rows if str(splits[r]) != "train"]
    if bad:
        raise SystemExit("%d chosen TAIX-Ray rows are not in the train split (first: row %d); the pool is not built" % (len(bad), bad[0]))


def read_png16(path, mono1, size):
    """A 16-bit chest PNG (12 bits stored) to a size x size uint8. Not xray's _read_picture: its convert('L') clips values
    above 255. The values stay as float32, MONOCHROME1 is inverted, then the module's own window and letterbox."""
    from PIL import Image
    xp = _xray()
    with Image.open(path) as im:
        im.load()
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if mono1:
        arr = arr.max() + arr.min() - arr
    return xp.letterbox(xp.scaled(arr), size)


def resample_rows(rows_u8, size=SIDE):
    """(n, S, S) uint8 to (n, size, size) uint8 by area averaging."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(rows_u8)).float().unsqueeze(1)
    return F.interpolate(x, size=(size, size), mode="area").round().clamp_(0, 255).squeeze(1).to(torch.uint8).numpy()


def _is_front(view):
    return str(view or "").upper() != "LATERAL"


def nih_items():
    """[(source, id, patient, kind, arg)] for NIH train_val_list.txt's pictures, in nih_384.npy's row order."""
    prep = _chest_prepare()
    meta = json.load(open(prep.cache_paths(384)[1], encoding="utf-8"))
    row_of = {k: i for i, k in enumerate(meta["images"])}
    want = [l.strip() for l in open(os.path.join(NIH_DIR, "train_val_list.txt"), encoding="utf-8") if l.strip()]
    test = {l.strip() for l in open(os.path.join(NIH_DIR, "test_list.txt"), encoding="utf-8") if l.strip()}
    guard_nih(want, test)
    missing = [n for n in want if n not in row_of]
    if missing:
        print("  NIH: %d names of train_val_list.txt are not in nih_384 (first %s)" % (len(missing), missing[0]), flush=True)
    return [("nih", n, n[:8], "nih", row_of[n]) for n in sorted((n for n in want if n in row_of), key=row_of.get)]


def taix_items():
    import pandas as pd
    df = pd.read_csv(TAIX_CSV)
    rows = np.flatnonzero(df.Split.to_numpy(str) == "train")
    guard_taix(df.Split.to_numpy(str), rows)
    uid, pid = df.UID.to_numpy(str), df.PatientID.to_numpy(str)
    return [("taix", uid[r], pid[r], "taix", int(r)) for r in rows]


def covid_ny_items():
    out = []
    for js in sorted(glob.glob(os.path.join(PUBLIC, "covid-19-ny-sbu-train", "*", "*", "*.json"))):
        try:
            d = json.load(open(js, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        h = d.get("header") or {}
        if str(h.get("Modality")) not in ("CR", "DX") or str(h.get("BodyPartExamined")).upper() not in ("PORT CHEST", "CHEST"):
            continue
        if not _is_front(h.get("ViewPosition")):
            continue
        mono1 = str(h.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1"
        for png in sorted(d.get("images") or {}):
            out.append(("covid-19-ny-sbu", png, str(h.get("PatientID") or ""), "png16", (os.path.join(os.path.dirname(js), png), mono1)))
    return out


def jpg_items(source, folder):
    base = os.path.join(PUBLIC, folder)
    files = sorted(os.path.join(r, f) for r, _d, fs in os.walk(base) for f in fs if f.lower().endswith(".jpg"))
    return [(source, os.path.relpath(p, base).replace("\\", "/"), "", "img", p) for p in files]


def _scan_dicom(files):
    """[(path, SOPInstanceUID, PatientID)] of the chest X-ray (CR, DX, not lateral) files among `files`."""
    import pydicom
    out = []
    for p in files:
        try:
            d = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if str(getattr(d, "Modality", "")) in ("CR", "DX") and _is_front(getattr(d, "ViewPosition", "")):
                out.append((p, str(d.SOPInstanceUID), str(getattr(d, "PatientID", ""))))
        except Exception:  # noqa: BLE001  not a DICOM file: skipped
            pass
    return out


def covid_ar_items(jobs):
    from concurrent.futures import ProcessPoolExecutor
    base = os.path.join(PUBLIC, "covid-19-ar")
    skip = (".json", ".csv", ".txt", ".png", ".jpg", ".xlsx", ".md", ".pdf")
    files = sorted(os.path.join(r, f) for r, _d, fs in os.walk(base) for f in fs if not f.lower().endswith(skip))
    parts = [files[a:a + 200] for a in range(0, len(files), 200)]
    out = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for got in pool.map(_scan_dicom, parts):
            out += got
    return [("covid-19-ar", sop, pid, "dcm", p) for p, sop, pid in sorted(out)]


_W = {}


def _open_worker(part, nih_npy):
    _W["out"] = np.lib.format.open_memmap(part, mode="r+")
    _W["nih"] = FileRows.npy(nih_npy)       # opened here, in the worker: a FileRows keeps per-thread file handles
    _W["taix"] = None


def _taix_map():
    if _W["taix"] is None:
        n = os.path.getsize(TAIX_U8) // (TAIX_SIDE * TAIX_SIDE)
        _W["taix"] = FileRows(TAIX_U8, (n, TAIX_SIDE, TAIX_SIDE))
    return _W["taix"]


def _decode_chunk(job):
    start, kind, args = job
    out, bad = _W["out"], []
    if kind in ("nih", "taix"):
        import torch
        torch.set_num_threads(1)
        src = _W["nih"] if kind == "nih" else _taix_map()
        out[start:start + len(args)] = resample_rows(np.stack([np.asarray(src[r]) for r in args]))
    else:
        xp = _xray()
        for i, a in enumerate(args):
            try:
                out[start + i] = read_png16(a[0], a[1], SIDE) if kind == "png16" else xp.read_any(a, SIDE)[0]
            except Exception as e:  # noqa: BLE001  left out and listed
                bad.append((start + i, "%s: %s" % (type(e).__name__, e)))
    out.flush()
    return start, len(args), bad


def pool(jobs):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    jobs = max(1, jobs)
    prep = _chest_prepare()
    items = nih_items() + taix_items() + covid_ny_items() + jpg_items("tb-cxr-kiran", "tb-cxr-kiran") \
        + jpg_items("bdcxr-3257", "bdcxr-3257") + covid_ar_items(jobs)
    n = len(items)
    print("pool: %d pictures to place" % n, flush=True)
    os.makedirs(CACHE, exist_ok=True)
    part = POOL + ".part"
    arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(n, SIDE, SIDE))
    arr[:] = 0
    arr.flush()
    del arr
    jobs_list, i = [], 0
    while i < n:
        kind, cap = items[i][3], CHUNK_CACHED if items[i][3] in ("nih", "taix") else CHUNK_DECODED
        j = i
        while j < n and j - i < cap and items[j][3] == kind:
            j += 1
        jobs_list.append((i, kind, [it[4] for it in items[i:j]]))
        i = j
    bad, done = [], 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=_open_worker, initargs=(part, prep.cache_paths(384)[0])) as ex:
        for _s, k, b in ex.map(_decode_chunk, jobs_list):
            bad += b
            done += k
            if done // 5000 != (done - k) // 5000 or done == n:
                print("  %d/%d  %.0f s  %d unreadable" % (done, n, time.time() - t0, len(bad)), flush=True)
    drop = {r for r, _e in bad}
    keep = [r for r in range(n) if r not in drop]
    if drop:                                         # leave the unreadable out: copy the kept rows into a new array
        src = FileRows.npy(part)
        dst = np.lib.format.open_memmap(POOL + ".part2", mode="w+", dtype=np.uint8, shape=(len(keep), SIDE, SIDE))
        for a in range(0, len(keep), 2000):
            dst[a:a + 2000] = src[keep[a:a + 2000]]
        dst.flush()
        src.close()                                  # the file is replaced next, and Windows refuses while it is open
        del dst, src
        os.replace(POOL + ".part2", part)
    os.replace(part, POOL)
    counts = {}
    with open(POOL_CSV + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "source", "id", "patient"])
        for new, r in enumerate(keep):
            it = items[r]
            counts[it[0]] = counts.get(it[0], 0) + 1
            w.writerow([new, it[0], it[1], it[2]])
    meta = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1), "rows": len(keep),
            "counts": counts, "unreadable": [[items[r][0], items[r][1], e] for r, e in bad]}
    with open(POOL_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(POOL_CSV + ".part", POOL_CSV)         # the csv last
    for s, c in counts.items():
        print("  %-18s %7d" % (s, c))
    print("total %d pictures, %d unreadable left out, %.0f s  (%s)" % (len(keep), len(bad), time.time() - t0, POOL), flush=True)


# ---------------------------------------------------------------- views, heads, loss (all on the card)

def make_views(x, g, crop=(0.35, 1.0), ratio=(0.75, 4 / 3), rot_deg=10.0, jitter=True, size=VIEW,
               jitter_p=0.8, brightness=0.4, contrast=(0.6, 1.4)):
    """x: (B, 1, H, W) floats in 0..1 on any device. One affine per picture, a random resized crop (area share `crop`,
    aspect `ratio`) with a rotation of +-rot_deg, sampled to size x size (zeros outside); then with probability jitter_p a
    brightness shift and a contrast change about the picture's own mean. No flip, no blur."""
    import torch
    import torch.nn.functional as F
    n, dev = x.shape[0], x.device

    def uni(lo, hi):
        return torch.rand(n, device=dev, generator=g) * (hi - lo) + lo

    s = uni(*crop)
    r = torch.exp(uni(math.log(ratio[0]), math.log(ratio[1])))
    w, h = (s * r).sqrt().clamp(max=1.0), (s / r).sqrt().clamp(max=1.0)
    cx, cy = (torch.rand(n, device=dev, generator=g) * 2 - 1) * (1 - w), (torch.rand(n, device=dev, generator=g) * 2 - 1) * (1 - h)
    ang = uni(-rot_deg, rot_deg) * math.pi / 180.0
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = w * cos, -h * sin, cx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = w * sin, h * cos, cy
    x = F.grid_sample(x, F.affine_grid(theta, [n, 1, size, size], align_corners=False), mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    if jitter:
        on = (torch.rand(n, device=dev, generator=g) < jitter_p).to(x.dtype).view(n, 1, 1, 1)
        b = uni(-brightness, brightness).view(n, 1, 1, 1) * on
        c = 1.0 + (uni(*contrast).view(n, 1, 1, 1) - 1.0) * on
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * c + m + b
    return x


def build_heads(feat_dim):
    import torch.nn as nn
    proj = nn.Sequential(nn.Linear(feat_dim, PROJ_DIM, bias=False), nn.BatchNorm1d(PROJ_DIM), nn.ReLU(inplace=True),
                         nn.Linear(PROJ_DIM, PROJ_DIM, bias=False), nn.BatchNorm1d(PROJ_DIM), nn.ReLU(inplace=True),
                         nn.Linear(PROJ_DIM, PROJ_DIM, bias=False), nn.BatchNorm1d(PROJ_DIM))
    pred = nn.Sequential(nn.Linear(PROJ_DIM, PRED_DIM, bias=False), nn.BatchNorm1d(PRED_DIM), nn.ReLU(inplace=True),
                         nn.Linear(PRED_DIM, PROJ_DIM))
    return proj, pred


def make_encoder(reader, cfg):
    return reader.ChestNet(cfg["widths"], cfg["dropout"], cfg["stem_stride"], 1, renderings=1)


def features(net, x):
    """The trunk's 256-d vector, in the memory order ChestNet.forward uses."""
    import torch
    if net.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return net.trunk(x).mean(dim=(2, 3))


def simsiam_loss(p1, p2, z1, z2):
    """0.5 D(p1, stopgrad z2) + 0.5 D(p2, stopgrad z1), D = negative mean cosine similarity."""
    import torch.nn.functional as F
    d = lambda p, z: -F.cosine_similarity(p.float(), z.detach().float(), dim=1).mean()  # noqa: E731
    return 0.5 * d(p1, z2) + 0.5 * d(p2, z1)


def z_spread(z):
    """Mean over channels of the std over the batch of the l2-normalised z (healthy: about 1/sqrt(2048))."""
    import torch.nn.functional as F
    return F.normalize(z.float(), dim=1).std(dim=0).mean()


class Model:
    def __init__(self, reader, cfg, dev):
        self.net = make_encoder(reader, cfg).to(dev)
        self.proj, self.pred = build_heads(cfg["widths"][-1])
        self.proj, self.pred = self.proj.to(dev), self.pred.to(dev)

    def train(self):
        self.net.train(), self.proj.train(), self.pred.train()

    def optimiser(self, batch):
        import torch
        base = RECIPE["lr_per_256"] * batch / 256
        main = [p for n, p in self.net.named_parameters() if n.startswith("trunk.")] + list(self.proj.parameters())
        opt = torch.optim.SGD([{"params": main, "lr": base, "cosine": True}, {"params": list(self.pred.parameters()), "lr": base, "cosine": False}],
                              lr=base, momentum=RECIPE["momentum"], weight_decay=RECIPE["weight_decay"])
        return opt, base


def train_step(m, opt, base, step, total_steps, x_u8, g, mean, sd, amp, side_in, view_px):
    """One SimSiam step on a uint8 batch already on the card; returns (loss, z_std) as card tensors."""
    import torch
    cos_lr = base * 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
    for grp in opt.param_groups:
        grp["lr"] = cos_lr if grp["cosine"] else base
    x = x_u8.float().div_(255.0).unsqueeze(1)
    v1 = (make_views(x, g, size=view_px) - mean) / sd
    v2 = (make_views(x, g, size=view_px) - mean) / sd
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        z1, z2 = m.proj(features(m.net, v1)), m.proj(features(m.net, v2))
        p1, p2 = m.pred(z1), m.pred(z2)
    loss = simsiam_loss(p1, p2, z1, z2)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach(), z_spread(z1.detach())


# ---------------------------------------------------------------- training

class _Crash(Exception):
    """Selftest only: stands for a cut mid-epoch."""


def fingerprint_of(version, epochs, batch, pool_path):
    st = os.stat(pool_path)
    return json.loads(json.dumps({"version": version, "epochs": epochs, "batch": batch, "pool": [st.st_size, st.st_mtime_ns],
                                  "recipe": RECIPE}, sort_keys=True))


def save_trunk(net, path):
    import torch
    torch.save({"state": net.state_dict()}, path + ".tmp")
    os.replace(path + ".tmp", path)


def fit_ssl(reader, pix, out, fp, epochs, batch, cfg, until=None, keep_epoch=KEEP_EPOCH, view_px=VIEW, amp=False,
            collapse_check=True, every_s=None, crash_at=None, say=print):
    """Trains (or continues from out/state.pt) up to epoch min(until, epochs); returns {"epoch", "loss", "z_std", "seconds",
    "collapsed", "finished"}. The cosine spans `epochs` whatever `until` says."""
    import torch
    from core import ckpt
    dev = reader.DEV
    os.makedirs(out, exist_ok=True)
    state_path = os.path.join(out, "state.pt")
    until = epochs if until is None else min(until, epochs)
    n = len(pix)
    # whole batches only, as SimSiam's loader (drop_last): a last batch of one picture would stop batch norm, and a
    # small one gives it a noisy batch
    steps_ep = n // batch
    assert steps_ep >= 1, "the pool (%d) is smaller than one batch (%d)" % (n, batch)
    total_steps = epochs * steps_ep
    torch.manual_seed(SEED)
    np.random.seed(SEED % 2 ** 32)
    mean, sd = reader.mean_sd(pix, np.arange(n), pix.shape[-1], np.random.default_rng(SEED))
    m = Model(reader, cfg, dev)
    opt, base = m.optimiser(batch)
    g_dev = torch.Generator(device=dev).manual_seed(SEED)
    g_cpu = np.random.default_rng(SEED + 1)
    every = ckpt.Every(every_s)
    labels = np.zeros(n, np.uint8)
    cache_row = np.arange(n)
    done_ep, skip, order_saved, step, sums = 0, 0, None, 0, [0.0, 0.0]
    info = {"epoch": 0, "loss": float("nan"), "z_std": float("nan"), "seconds": 0.0, "collapsed": False, "finished": False}
    if os.path.exists(state_path):
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        if ck.get("fingerprint") != fp:
            raise SystemExit("%s was written by a different run (version, epochs, batch, pool file or recipe differ); "
                             "move it away to start again, or run with the settings it was made with" % state_path)
        m.net.load_state_dict(ck["net"]), m.proj.load_state_dict(ck["proj"]), m.pred.load_state_dict(ck["pred"])
        opt.load_state_dict(ck["opt"])
        done_ep, skip, order_saved, step, sums = ck["epoch"], ck["batch"], ck["order"], ck["step"], list(ck["sums"])
        g_cpu.bit_generator.state = ck["g_cpu"]
        g_dev.set_state(ck["g_dev"].cpu())
        ckpt.set_rng_state(ck["rng"])
        info.update(epoch=done_ep, seconds=ck.get("seconds", 0.0))
        say("resumed from %s: %d epochs done%s" % (state_path, done_ep, ", batch %d of the next" % skip if skip else ""))

    def save_state(epoch, done_b, order, seconds):
        ckpt.save_partial(state_path, {"net": m.net.state_dict(), "proj": m.proj.state_dict(), "pred": m.pred.state_dict(),
                                       "opt": opt.state_dict(), "epoch": epoch, "batch": done_b, "order": order, "step": step,
                                       "sums": sums, "g_cpu": g_cpu.bit_generator.state, "g_dev": g_dev.get_state(),
                                       "rng": ckpt.rng_state(), "seconds": seconds, "fingerprint": fp})
        every.mark()

    for ep in range(done_ep + 1, until + 1):
        m.train()
        t0 = time.time()
        order = order_saved if order_saved is not None else g_cpu.permutation(n)
        all_parts = [np.sort(order[a:a + batch]) for a in range(0, steps_ep * batch, batch)]
        parts = all_parts[skip:]
        done_b = skip
        tot = torch.tensor(sums, dtype=torch.float64, device=dev)
        skip, order_saved, sums = 0, None, [0.0, 0.0]
        for rows, x, _y in reader.prefetch(pix, cache_row, labels, parts):
            loss, zs = train_step(m, opt, base, step, total_steps, x, g_dev, mean, sd, amp, pix.shape[-1], view_px)
            tot += torch.stack([loss.double(), zs.double()])
            step += 1
            done_b += 1
            if done_b < steps_ep and (every.due() or crash_at == step):
                sums = tot.tolist()
                save_state(ep - 1, done_b, order, info["seconds"] + time.time() - t0)
                sums = [0.0, 0.0]
            if crash_at == step:
                raise _Crash()
        loss_e, z_e = (tot / steps_ep).tolist()
        info.update(epoch=ep, loss=loss_e, z_std=z_e, seconds=info["seconds"] + time.time() - t0)
        say("  epoch %2d  loss %.4f  z_std %.5f (healthy %.5f)  %.0f s" % (ep, loss_e, z_e, 1 / math.sqrt(PROJ_DIM), time.time() - t0))
        if ep == keep_epoch:
            save_trunk(m.net, os.path.join(out, "trunk_e15.pt"))
        if ep == epochs:
            save_trunk(m.net, os.path.join(out, "trunk.pt"))
            info["finished"] = True
        save_state(ep, 0, None, info["seconds"])
        if collapse_check and ep >= 2 and z_e < COLLAPSE_BELOW:
            say("COLLAPSED: z_std %.5f is below %.5f" % (z_e, COLLAPSE_BELOW))
            info["collapsed"] = True
            return info
    return info


def setup_card(reader, name):
    import torch
    from core import card
    card.require_card(reader.DEV, torch)
    reader.xd.limit_card_memory(CARD_GB, name)
    if reader.DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True


def open_pool():
    if not (os.path.exists(POOL) and os.path.exists(POOL_CSV)):
        raise SystemExit("no pool at %s; build it with: python modalities/chest_xray/eval/simsiam.py pool" % POOL)
    return FileRows.npy(POOL)


def train(version, epochs, batch, until):
    import torch
    from core import ledger
    import reader
    out = out_dir(version)
    pix = open_pool()
    setup_card(reader, "ssl train v%d" % version)
    cfg = dict(reader.CFG)
    fp = fingerprint_of(version, epochs, batch, POOL)
    amp = reader.DEV.type == "cuda"
    print("ssl train v%d: %d pictures, %d epochs at batch %d%s, amp %s" % (version, len(pix), epochs, batch,
          "" if until is None else ", stopping after epoch %d" % until, amp), flush=True)
    with ledger.run("experiment", "ssl train v%d" % version) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "until": until, "recipe": RECIPE})
        info = fit_ssl(reader, pix, out, fp, epochs, batch, cfg, until=until, amp=amp)
        led.number("loss", info["loss"]).number("z_std", info["z_std"]).number("seconds", info["seconds"], "s")
        if reader.DEV.type == "cuda":
            peak, held = torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30
            led.number("peak_reserved_memory_gb", held)
            print("peak card memory %.2f GiB allocated, %.2f reserved" % (peak, held), flush=True)
    if info["collapsed"]:
        sys.exit(3)


def bench(batch, steps):
    import torch
    import reader
    pix = open_pool()
    setup_card(reader, "ssl bench")
    dev = reader.DEV
    cfg = dict(reader.CFG)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    mean, sd = reader.mean_sd(pix, np.arange(len(pix)), SIDE, rng)
    m = Model(reader, cfg, dev)
    m.train()
    opt, base = m.optimiser(batch)
    g = torch.Generator(device=dev).manual_seed(SEED)
    parts = [np.sort(rng.choice(len(pix), batch, replace=False)) for _ in range(steps + 3)]
    labels, t0, i = np.zeros(len(pix), np.uint8), None, 0
    for rows, x, _y in reader.prefetch(pix, np.arange(len(pix)), labels, parts):
        if i == 3:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        train_step(m, opt, base, i, steps + 3, x, g, mean, sd, dev.type == "cuda", SIDE, VIEW)
        i += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    held = torch.cuda.max_memory_reserved() / 2 ** 30 if dev.type == "cuda" else 0.0
    print("bench batch %d: seconds_a_step %.5f reserved_gib %.2f steps_an_epoch %d"
          % (batch, (time.time() - t0) / steps, held, len(pix) // batch), flush=True)


# ---------------------------------------------------------------- selftest

def selftest():
    import tempfile
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import reader
    import taix
    # views: shape, and a full crop with no rotation or jitter is the picture resized bilinearly (the same sample points;
    # not the area resize, whose uneven windows at 288 to 256 differ by up to half a pixel's slope)
    ramp = (torch.linspace(0, 1, SIDE).view(1, 1, 1, SIDE) * 0.6 + torch.linspace(0, 1, SIDE).view(1, 1, SIDE, 1) * 0.3).repeat(3, 1, 1, 1)
    g = torch.Generator().manual_seed(1)
    assert make_views(ramp, g).shape == (3, 1, VIEW, VIEW)
    v = make_views(ramp, g, crop=(1, 1), ratio=(1, 1), rot_deg=0.0, jitter=False)
    want = F.interpolate(ramp, size=(VIEW, VIEW), mode="bilinear", align_corners=False)
    assert (v - want).abs().max() < 1e-4, "full-crop view is not the bilinear resize"
    assert not torch.equal(make_views(ramp, g)[0], make_views(ramp, g)[0]), "views are random"
    # the loss: identical views and an identity predictor give -1; the spread of isotropic noise is 1/sqrt(2048)
    net = reader.ChestNet([4] * 5, 0.3, 2, 1)
    proj, _pred = build_heads(4)
    net.train(), proj.train()
    xx = torch.randn(8, 1, 64, 64)
    z = proj(features(net, xx))
    assert abs(float(simsiam_loss(z, z, z, z)) + 1.0) < 1e-5
    zr = torch.randn(2000, PROJ_DIM, generator=torch.Generator().manual_seed(2))
    assert abs(float(z_spread(zr)) * math.sqrt(PROJ_DIM) - 1) < 0.1
    # the pool's guards
    for fn, a in ((guard_nih, (["a.png", "b.png"], {"b.png"})), (guard_taix, (np.array(["train", "val"]), [0, 1]))):
        try:
            fn(*a)
        except SystemExit:
            pass
        else:
            raise AssertionError("%s let a test or non-train picture through" % fn.__name__)
    guard_nih(["a.png"], {"b.png"})
    guard_taix(np.array(["train", "val"]), [0])
    # a 16-bit PNG with values above 255 is windowed, not clipped
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.png")
        pic = (np.linspace(0, 1, 90 * 60).reshape(60, 90) * 4000).astype(np.uint16)
        Image.fromarray(pic).save(p)
        px = read_png16(p, False, 32)
        assert px.shape == (32, 32) and len(np.unique(px)) > 50 and (px == 255).mean() < 0.3, "the 16-bit picture is clipped"
        # MONOCHROME1 inverts: on the picture rows (not the letterbox bands) it runs against the plain read (an even ramp's
        # inverse is its mirror, so the sums are equal and say nothing)
        inv = read_png16(p, True, 32)
        on = (px.max(axis=1) > 0) | (inv.max(axis=1) > 0)
        assert np.corrcoef(px[on].ravel().astype(float), inv[on].ravel().astype(float))[0, 1] < -0.9, "MONOCHROME1 is not inverted"
    # a tiny training: checkpoints, load into a taix net, an interrupted-and-resumed run equals an uninterrupted one
    keep_dev = reader.DEV
    reader.DEV = torch.device("cpu")
    try:
        pix = np.random.default_rng(3).integers(0, 255, (64, 72, 72), dtype=np.uint8)
        cfg = dict(reader.CFG, widths=[4] * 5)
        fp = {"selftest": 1}
        kw = dict(epochs=2, batch=16, cfg=cfg, keep_epoch=1, view_px=64, collapse_check=False, say=lambda *_: None)
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (os.path.join(d, k) for k in "abc")
            ia = fit_ssl(reader, pix, a, fp, **kw)
            assert ia["finished"] and ia["epoch"] == 2 and np.isfinite(ia["loss"]), ia
            assert all(os.path.exists(os.path.join(a, f)) for f in ("trunk_e15.pt", "trunk.pt", "state.pt"))
            tnet = reader.ChestNet(cfg["widths"], cfg["dropout"], cfg["stem_stride"], 31)
            lines = []
            taix.load_init(tnet, "x", say=lines.append, path=os.path.join(a, "trunk.pt"))
            ref = torch.load(os.path.join(a, "trunk.pt"), weights_only=False)["state"]
            assert all(torch.equal(tnet.state_dict()[k], v) for k, v in ref.items() if k.startswith("trunk."))
            ib = fit_ssl(reader, pix, b, fp, until=1, **kw)
            assert ib["epoch"] == 1 and not ib["finished"] and not os.path.exists(os.path.join(b, "trunk.pt"))
            fit_ssl(reader, pix, b, fp, **kw)
            sa, sb = ref, torch.load(os.path.join(b, "trunk.pt"), weights_only=False)["state"]
            assert all((sa[k].float() - sb[k].float()).abs().max() < 1e-6 for k in sa), "--until then resume differs"
            try:
                fit_ssl(reader, pix, b, {"selftest": 2}, **kw)
            except SystemExit as e:
                assert "different run" in str(e)
            else:
                raise AssertionError("a state.pt of another run was continued")
            try:                                          # a cut mid-epoch 2 (step 5 of 4 an epoch), then continued
                fit_ssl(reader, pix, c, fp, every_s=1e9, crash_at=5, **kw)
            except _Crash:
                pass
            fit_ssl(reader, pix, c, fp, **kw)
            sc = torch.load(os.path.join(c, "trunk.pt"), weights_only=False)["state"]
            assert all((sa[k].float() - sc[k].float()).abs().max() < 1e-6 for k in sa), "a mid-epoch resume differs"
    finally:
        reader.DEV = keep_dev
    print("selftest: all passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    b = sub.add_parser("bench")
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--steps", type=int, default=30)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--epochs", type=int, required=True)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--until", type=int, default=None)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "pool":
        pool(a.jobs)
    elif a.cmd == "bench":
        bench(a.batch, a.steps)
    else:
        train(a.version, a.epochs, a.batch, a.until)


if __name__ == "__main__":
    main()
