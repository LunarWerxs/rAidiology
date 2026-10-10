"""Fractures from one X-ray foundation ViT of our own (idea 129, step B): mae-v1's chest encoder continued on our own bone
and spine X-rays, as mae.py continues it on chest pictures.

pool: work/eval/xray/cache/ssl_pool_xr_288.npy, the bone and spine pictures of the development split at the chest pool's
      288 px (simsiam.py's treatment), with its csv (row, dataset, image, patient, split) and json (counts, what was left
      out and why). Bone pictures come from the 384 px caches (no disk read beyond them); BUU-LSPINE's from its own files.
      No test picture is taken. The chest rows stay in simsiam's ssl_pool_288.npy and are read beside this pool.
pretrain-xr: mae-v1's encoder continued on the chest pool and the X-ray pool read together, with a fresh decoder. Each epoch
      draws every chest row once and every bone or spine row XR_REPEAT times, shuffled from a fixed seed. The first
      WARM_EPOCHS epochs keep the encoder frozen (only the decoder and the mask token learn); then everything trains.
      Writes mae-v2's encoder.pt in mae-v1's format, which vit_detector.py reads with --set mae=2.
selftest: the draw list, the test guard, the frozen warm-up, resume, and the encoder's format, on synthetic pools.

Left out: PediaSHF-DX and MFXR, which LICENSING.md does not mark clean for training. Their reasons go into the pool's json.

Usage:
  python modalities/xray/eval/mae_xr.py pool [--jobs 4]
  python modalities/xray/eval/mae_xr.py pretrain-xr --version 2 [--epochs 60] [--batch 128] [--until N] [--card-gb 4.5] [--pool-on-card] [--compile]
  python modalities/xray/eval/mae_xr.py selftest
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SPINE = os.path.join(ROOT, "modalities", "spine_xray", "eval")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "chest_xray", "eval"))  # chest first, as vit_detector.py

from core import ckpt, ledger, paths  # noqa: E402
from core.rows import FileRows  # noqa: E402
import mae  # noqa: E402  (the MAE model, masking, loss, learning-rate schedule, pool reading and the card lease)
import simsiam  # noqa: E402  (the chest pool's files, its picture treatment and its guards)
if HERE not in sys.path:    # reader.py (imported by mae) takes the X-ray folder off the path once it has its modules
    sys.path.append(HERE)
import vit_detector  # noqa: E402  (the fine-tune that reads mae-v2 unchanged; its detector module holds the X-ray sets)

xd = vit_detector.xd
WARM_EPOCHS = 2               # the encoder is frozen for these epochs (only the decoder and the mask token learn)
XR_REPEAT = 3                 # each bone or spine row is drawn this many times an epoch; each chest row once
EPOCHS = 60
BATCH = 128
CACHE_SIZE = 384              # the X-ray caches the bone pictures come from (prepare.py)
SIDE = simsiam.SIDE           # 288: the chest pool's side
PX = mae.PRE_PX               # 256: mae-v1's pretraining side
CHUNK_CACHE = 64              # pool pictures per job read from a 384 px cache
CHUNK_FILE = 16               # pool pictures per job decoded from their files
JOBS = 4
PUBLIC = os.path.join(paths.DATASETS, "public")
CACHE = os.path.join(paths.WORK, "eval", "xray", "cache")
XR_POOL = os.path.join(CACHE, "ssl_pool_xr_288.npy")
XR_CSV = os.path.join(CACHE, "ssl_pool_xr_288.csv")
XR_JSON = os.path.join(CACHE, "ssl_pool_xr_288.json")
SPINE_BUU = "buu-lspine"
TREATMENT = ("bone pictures: the 384 px letterboxed caches (prepare.py) resampled to 288 px by simsiam.resample_rows, area "
             "averaging, as simsiam's NIH and TAIX rows; BUU-LSPINE pictures: xray read_any at 288 px (windowed, letterboxed), "
             "as simsiam's JPEG rows")
LEFT_OUT = {
    "pediashf-dx": "LICENSING.md marks it 'Not used': its 2,015 labelled pictures are all fractures with boxes (four label "
                   "files empty), and its 8,310 validation pictures carry no labels, so there are no healthy elbows for a "
                   "yes/no reader. Not marked clean for training, so it is left out; tools/xray_duplicates.py was not run for it.",
    "mfxr": "LICENSING.md marks it 'Not used yet': its good class holds normal forearms and other fractures together, so only "
            "its 2,795 Monteggia pictures are usable as fractured ones. Not marked clean for training, so it is left out; "
            "tools/xray_duplicates.py was not run for it.",
}


def save_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def xray_prepare():
    return simsiam._xray()


def dev_rows(df, name):
    """df (one set's index rows, with a split column) when every row is a development row. A test row is refused, and the
    set's split is opened only through datasets.guard_test, as everywhere else."""
    marked = df[df.split != "dev"]
    if len(marked):
        raise SystemExit("%s: %d rows are marked %r, not development; the X-ray pool is not built"
                         % (name, len(marked), marked.split.iloc[0]))
    for sp in df.split.unique():
        xd.datasets.guard_test(sp)
    return df


def chest_guard():
    """Refuses the chest pool when a row is on NIH's test list or is a TAIX-Ray row outside its train split. simsiam's pool
    already leaves both out; this reads the pool's csv again before any X-ray row is added to it."""
    import pandas as pd
    with open(simsiam.POOL_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    nih = [r["id"] for r in rows if r["source"] == "nih"]
    with open(os.path.join(simsiam.NIH_DIR, "test_list.txt"), encoding="utf-8") as f:
        test = {line.strip() for line in f if line.strip()}
    simsiam.guard_nih(nih, test)
    t = pd.read_csv(simsiam.TAIX_CSV, dtype=str, keep_default_na=False)
    split = dict(zip(t.UID, t.Split))
    taix = [r["id"] for r in rows if r["source"] == "taix"]
    bad = [u for u in taix if split.get(u) != "train"]
    if bad:
        raise SystemExit("%d chest pool rows from TAIX-Ray are not in its train split (first: %s); the X-ray pool is not built"
                         % (len(bad), bad[0]))
    return {"chest_rows": len(rows), "nih_rows_checked": len(nih), "taix_rows_checked": len(taix)}


class Joined:
    """The chest pool's rows, then the X-ray pool's, read as one pool: row r is chest[r] below n_chest, else xr[r - n_chest]."""

    def __init__(self, chest, xr):
        self.chest, self.xr = chest, xr
        self.n_chest = int(chest.shape[0])
        self.shape = (self.n_chest + int(xr.shape[0]),) + tuple(chest.shape[1:])
        self.nbytes = int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, r):
        r = np.arange(len(self))[r] if isinstance(r, slice) else np.asarray(r, dtype=np.int64)
        out = np.empty((len(r),) + self.shape[1:], np.uint8)
        c = r < self.n_chest
        if c.any():
            out[c] = self.chest[r[c]]
        if (~c).any():
            out[~c] = self.xr[r[~c] - self.n_chest]
        return out


def draws(n_chest, n_xr, ep):
    """One epoch's rows, shuffled: each chest row once, each bone or spine row XR_REPEAT times, the order from SEED + ep."""
    rows = np.concatenate([np.arange(n_chest), np.repeat(np.arange(n_chest, n_chest + n_xr), XR_REPEAT)])
    return np.random.default_rng(mae.SEED + ep).permutation(rows)


def epoch_parts(n_chest, n_xr, batch, steps, ep):
    """The sorted row indices of each batch of epoch ep (the last part-batch is dropped, as mae.py does)."""
    order = draws(n_chest, n_xr, ep)
    return [np.sort(order[a:a + batch]) for a in range(0, steps * batch, batch)]


def set_trainable(vit, on):
    for p in vit.parameters():
        p.requires_grad_(on)


def landmark_buu_ids(landmarks):
    """The BUU ids of the spine landmark set's pictures (landmarks.load for train and test): the pictures the landmark test
    scores (idea 32, idea 119), which no pool may hold."""
    recs = landmarks.load("train") + landmarks.load("test")
    return {int(r["id"]) for r in recs if r.get("source") == "buu" and r.get("id") is not None}


def gather():
    """Every development bone and spine picture in creation order, as (kind, set, image key, patient, where): kind "cache"
    (where is a row of the set's 384 px cache) or "file" (where is the picture's path). Returns (items, caches, report)."""
    s = xd.splitmod.load()
    items, caches = [], {}
    report = {"dev_rows": {}, "left_out": {}, "unreadable": [], "landmark_test": None}
    xp = xray_prepare()
    for d in xd.datasets.ALL:
        arr, order, _size = xp.load(d, CACHE_SIZE)
        bad = xp.unreadable(d, CACHE_SIZE)
        df = dev_rows(xd.splitmod.frame(d, "dev", s), d)
        caches[d] = arr
        report["dev_rows"][d] = len(df)
        for r in df.itertuples(index=False):
            key = str(r.image)
            if key in bad:
                report["unreadable"].append([d, key, "unreadable in the %d px cache" % CACHE_SIZE])
                continue
            items.append(("cache", d, key, str(r.patient), int(order[key])))
    for name, why in LEFT_OUT.items():
        report["left_out"][name] = {"on_disk": os.path.isdir(os.path.join(PUBLIC, name)), "reason": why}
    sys.path.append(SPINE)     # buu.py and landmarks.py, appended so the modules above keep their names
    import buu
    import landmarks
    try:
        buu.root()
    except SystemExit as e:
        report["left_out"][SPINE_BUU] = {"on_disk": False, "reason": str(e)}
        return items, caches, report
    landmark = landmark_buu_ids(landmarks)
    dropped_ids, dropped_pics = set(), 0
    for view in ("AP", "LA"):
        for pid, path in sorted(buu.files(view, ext="jpg").items()):
            if pid in landmark:
                dropped_ids.add(pid)
                dropped_pics += 1
                continue
            items.append(("file", SPINE_BUU, os.path.basename(path)[:-4], "%04d" % pid, path))
    report["landmark_test"] = {"rule": "a BUU-LSPINE patient whose picture is in the spine landmark set (landmarks.load train "
                                       "and test) is left out, both views", "patients_left_out": len(dropped_ids),
                               "pictures_left_out": dropped_pics}
    return items, caches, report


def make_chunk(a, b, items, caches):
    """(a, pictures at SIDE, [(row, error)]) for pool rows a to b: a cache's rows area-resized as simsiam's NIH rows, or files
    read as simsiam's JPEG rows (xray read_any at SIDE). A picture that cannot be read is listed and left out."""
    if items[a][0] == "cache":
        src = caches[items[a][1]]
        x = np.stack([np.asarray(src[items[k][4]]) for k in range(a, b)])
        return a, simsiam.resample_rows(x, SIDE), []
    xp = xray_prepare()
    pics = np.zeros((b - a, SIDE, SIDE), np.uint8)
    bad = []
    for k in range(a, b):
        try:
            pics[k - a] = xp.read_any(items[k][4], SIDE)[0]
        except Exception as e:  # noqa: BLE001  left out and listed, as simsiam.py's pool does
            bad.append((k, "%s: %s" % (type(e).__name__, e)))
    return a, pics, bad


def plan(items):
    """(start, end) of each chunk of pool rows: consecutive rows of one kind and one set, in creation order."""
    out, i, n = [], 0, len(items)
    while i < n:
        kind, d = items[i][0], items[i][1]
        cap = CHUNK_CACHE if kind == "cache" else CHUNK_FILE
        j = i
        while j < n and j - i < cap and items[j][0] == kind and items[j][1] == d:
            j += 1
        out.append((i, j))
        i = j
    return out


def build_pool(jobs):
    """Writes the X-ray pool. Resumable: the pictures go to ssl_pool_xr_288.npy.part as their chunks finish, each chunk's
    flags are written after its pictures, and a run that stops continues from the first chunk not marked done."""
    t0 = time.time()
    chest = chest_guard()
    items, caches, report = gather()
    n = len(items)
    if n == 0:
        raise SystemExit("no bone or spine picture to pool")
    identity = hashlib.sha256("\n".join("%s|%s" % (it[1], it[2]) for it in items).encode("utf-8")).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    part, flags_path, state_path = XR_POOL + ".part", XR_POOL + ".flags", XR_POOL + ".state.json"
    state = {"items": n, "identity": identity, "errors": {}}
    resume = all(os.path.exists(p) for p in (part, flags_path, state_path))
    if resume:
        with open(state_path, encoding="utf-8") as f:
            old = json.load(f)
        resume = old.get("items") == n and old.get("identity") == identity
    if resume:
        arr = np.lib.format.open_memmap(part, mode="r+")
        flags = np.lib.format.open_memmap(flags_path, mode="r+")
        state = old
        print("resumed: %d of %d pictures done" % (int((flags != 0).sum()), n), flush=True)
    else:
        arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(n, SIDE, SIDE))
        flags = np.lib.format.open_memmap(flags_path, mode="w+", dtype=np.uint8, shape=(n,))
        save_json(state_path, state)
    todo = [c for c in plan(items) if (flags[c[0]:c[1]] == 0).any()]
    print("X-ray pool: %d bone and spine pictures in %d chunks, %d to do" % (n, len(plan(items)), len(todo)), flush=True)
    window = 4 * jobs
    last = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for w in range(0, len(todo), window):
            for a, pics, bad in ex.map(lambda c: make_chunk(c[0], c[1], items, caches), todo[w:w + window]):
                b = a + len(pics)
                arr[a:b] = pics
                flags[a:b] = 1
                for k, err in bad:
                    flags[k] = 2
                    state["errors"][str(k)] = err
            arr.flush()
            flags.flush()
            save_json(state_path, state)
            if time.time() - last >= 60 or w + window >= len(todo):
                last = time.time()
                print("  %d/%d pictures  %.0f s  %d unreadable"
                      % (int((flags != 0).sum()), n, time.time() - t0, int((flags == 2).sum())), flush=True)
    if (flags == 0).any():
        raise SystemExit("%d pictures are not done; run the pool again to finish them" % int((flags == 0).sum()))
    arr.flush()
    kept = np.flatnonzero(flags == 1)
    unreadable = report["unreadable"] + [[items[k][1], items[k][2], state["errors"].get(str(k), "unreadable")]
                                         for k in np.flatnonzero(flags == 2)]
    del arr, flags
    if len(kept) < n:
        src = np.load(part, mmap_mode="r")
        dst = np.lib.format.open_memmap(XR_POOL + ".part2", mode="w+", dtype=np.uint8, shape=(len(kept), SIDE, SIDE))
        for a in range(0, len(kept), 2000):
            dst[a:a + 2000] = src[kept[a:a + 2000]]
        dst.flush()
        del dst, src
        os.remove(part)
        os.replace(XR_POOL + ".part2", XR_POOL)
    else:
        os.replace(part, XR_POOL)
    os.remove(flags_path)
    os.remove(state_path)
    counts = dict(Counter(items[k][1] for k in kept))
    meta = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1), "rows": int(len(kept)),
            "size": [SIDE, SIDE], "counts": counts, "dev_rows_by_set": report["dev_rows"], "left_out": report["left_out"],
            "landmark_test": report["landmark_test"], "unreadable": unreadable, "chest_guard": chest,
            "chest_pool": {"file": os.path.basename(simsiam.POOL), "rows": chest["chest_rows"]},
            "split": "development rows of each set's own split (splitmod.frame); BUU-LSPINE has five folds and no held-out "
                     "list, so all its pictures are development",
            "treatment": TREATMENT}
    save_json(XR_JSON, meta)
    with open(XR_CSV + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "dataset", "image", "patient", "split"])
        for new, old in enumerate(kept):
            it = items[old]
            w.writerow([new, it[1], it[2], it[3], "dev"])
    os.replace(XR_CSV + ".part", XR_CSV)
    print("ssl_pool_xr_288: %d pictures %s, %d unreadable left out, %.0f s  (%s)"
          % (len(kept), json.dumps(counts), len(unreadable), time.time() - t0, XR_POOL), flush=True)


def epoch_loop_speed(pool_on_card, load_s, compile_model):
    return {"pool_on_card": pool_on_card, "pool_load_seconds": None if load_s is None else round(load_s, 1),
            "compile": compile_model, "compile_mode": None, "pictures_per_second": {}}


def fit_xr(pool, n_chest, out, start, start_sha, epochs, batch, pool_info, warm=WARM_EPOCHS, until=None, say=print,
           pool_on_card=False, compile_model=False, px=PX, enc=mae.ENC, dec=mae.DEC):
    """Continues the encoder at `start` on the pool to epoch min(until, warm + epochs), continuing from out/partial.pt when it
    matches. Epochs 1 to warm keep the encoder frozen; the rest train everything. Writes out/encoder.pt after the last epoch.
    Returns the history list. pool_on_card and compile_model are speed options only, as in mae.fit_mae."""
    dev = mae.reader.DEV
    os.makedirs(out, exist_ok=True)
    part = os.path.join(out, "partial.pt")
    n = len(pool)
    n_xr = n - n_chest
    steps = (n_chest + XR_REPEAT * n_xr) // batch
    assert steps >= 1, "the draw list (%d) is smaller than one batch (%d)" % (n_chest + XR_REPEAT * n_xr, batch)
    total_ep = warm + epochs
    until = total_ep if until is None else min(until, total_ep)
    base = mae.MAE_LR_PER_256 * batch / 256
    total, lr_warm = total_ep * steps, max(1, round(0.1 * total_ep * steps))
    fp = json.loads(json.dumps({"epochs": epochs, "warm_epochs": warm, "batch": batch, "px": px, "enc": enc, "dec": dec,
                                "mask_ratio": mae.MASK_RATIO, "seed": mae.SEED, "xr_repeat": XR_REPEAT, "chest_rows": n_chest,
                                "xr_rows": n_xr, "start": start_sha, "pool": pool_info}, sort_keys=True))
    torch.manual_seed(mae.SEED)
    mean, sd = mae.pool_mean_sd(pool, np.arange(n), pool.shape[1:], np.random.default_rng(mae.SEED))
    model = mae.MAE(px, enc, dec).to(dev)
    vit, s = mae.load_encoder(start, px, 0.0)
    if (s["dim"], s["depth"], s["heads"]) != (enc["dim"], enc["depth"], enc["heads"]):
        raise SystemExit("%s is a %s encoder, not %s" % (start, s, enc))
    model.vit.load_state_dict(vit.state_dict())
    step = mae.Compiled(model, say) if compile_model else model
    opt = torch.optim.AdamW(mae.param_groups(model, 0.05), lr=base, betas=(0.9, 0.95))
    zeros = np.zeros((n, 1), np.float32)
    history, start_ep = [], 1
    ck = ckpt.load_partial(part, fp)
    if ck is not None and 0 < ck["epoch"] <= total_ep:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        history, start_ep = ck["history"], ck["epoch"] + 1
        ckpt.set_rng_state(ck["rng"])
        say("resumed from %s after epoch %d" % (part, ck["epoch"]))
    del ck
    pool_t, load_s = mae.load_pool_to_card(pool, dev, say) if pool_on_card else (None, None)
    speed = epoch_loop_speed(pool_on_card, load_s, compile_model)
    say("draws per epoch: %d (%.1f%% bone and spine), %d steps of %d; %d warm-up epochs with the encoder frozen"
        % (steps * batch, 100.0 * XR_REPEAT * n_xr / (n_chest + XR_REPEAT * n_xr), steps, batch, warm))
    for ep in range(start_ep, until + 1):
        frozen = ep <= warm
        set_trainable(model.vit, not frozen)
        model.train()
        g = torch.Generator(device=dev).manual_seed(mae.SEED + ep)
        parts = epoch_parts(n_chest, n_xr, batch, steps, ep)
        tot = torch.zeros((), dtype=torch.float64, device=dev)
        t0 = time.time()
        batches = (mae.card_batches(pool_t, parts, dev) if pool_t is not None
                   else mae.reader.prefetch(pool, np.arange(n), zeros, parts))
        for i, (_rows, x, _y) in enumerate(batches):
            mae.set_lr(opt, base * mae.lr_factor((ep - 1) * steps + i, total, lr_warm))
            x = (mae.random_resized(mae.to_float(x, pool.shape[1:]), g, px) - mean) / sd
            with mae.amp():
                loss = step(x, g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach().double()
            if (i + 1) % mae.LOG_EVERY == 0:
                say("  epoch %d step %d/%d  loss %.4f  %.0f pictures/s"
                    % (ep, i + 1, steps, float(tot) / (i + 1), (i + 1) * batch / (time.time() - t0)))
        secs = time.time() - t0
        history.append({"epoch": ep, "frozen_encoder": frozen, "loss": float(tot) / steps, "seconds": round(secs, 1),
                        "pictures_per_second": round(steps * batch / secs, 1)})
        say("epoch %3d%s  loss %.4f  %.0f s  %.0f pictures/s"
            % (ep, " (encoder frozen)" if frozen else "", history[-1]["loss"], secs, history[-1]["pictures_per_second"]))
        speed["pictures_per_second"][str(ep)] = history[-1]["pictures_per_second"]
        speed["compile_mode"] = step.mode if compile_model else None
        save_json(os.path.join(out, "meta.json"), {"px": px, "epochs": epochs, "warm_epochs": warm, "batch": batch,
                                                   "lr": base, "lr_warm_steps": lr_warm, "steps_per_epoch": steps,
                                                   "mask_ratio": mae.MASK_RATIO, "enc": enc, "dec": dec,
                                                   "xr_repeat": XR_REPEAT, "start": {"file": start, "sha256": start_sha},
                                                   "pool": pool_info, "history": history, "speed": speed})
        ckpt.save_partial(part, {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "history": history,
                                 "rng": ckpt.rng_state(), "fingerprint": fp})
    if history and history[-1]["epoch"] == total_ep:
        mae.save_encoder(model.vit, os.path.join(out, "encoder.pt"),
                         {"px": px, "dim": enc["dim"], "depth": enc["depth"], "heads": enc["heads"], "patch": mae.PATCH,
                          "mask_ratio": mae.MASK_RATIO, "epochs": epochs, "batch": batch, "seed": mae.SEED,
                          "pool": pool_info})
    return history


def open_pool():
    """(the pool, the chest rows, the pool info): simsiam's chest pool and the X-ray pool read together, each checked
    against its own json."""
    for p in (simsiam.POOL, simsiam.POOL_JSON, XR_POOL, XR_JSON):
        if not os.path.exists(p):
            raise SystemExit("no pool at %s; build the X-ray pool with: python modalities/xray/eval/mae_xr.py pool" % p)
    with open(XR_JSON, encoding="utf-8") as f:
        xj = json.load(f)
    with open(simsiam.POOL_JSON, encoding="utf-8") as f:
        chest_counts = json.load(f)
    chest = simsiam.open_pool()
    xr = FileRows.npy(XR_POOL)
    if xr.shape[0] != xj["rows"] or tuple(xr.shape[1:]) != (SIDE, SIDE):
        raise SystemExit("%s does not match its json (%d rows of %d px); rebuild the pool" % (XR_POOL, xj["rows"], SIDE))
    cst, xst = os.stat(simsiam.POOL), os.stat(XR_POOL)
    info = {"file": os.path.basename(simsiam.POOL), "bytes": cst.st_size, "mtime_ns": cst.st_mtime_ns, "counts": chest_counts,
            "xr": {"file": os.path.basename(XR_POOL), "bytes": xst.st_size, "mtime_ns": xst.st_mtime_ns, "rows": xj["rows"],
                   "counts": xj["counts"]}}
    return Joined(chest, xr), int(chest.shape[0]), info


def pretrain_xr(version, epochs, batch, card_gb, until=None, pool_on_card=False, compile_model=False):
    out = mae.mae_dir(version)
    if os.path.exists(os.path.join(out, "encoder.pt")):
        raise SystemExit("%s holds a finished encoder; move it away to pretrain again" % out)
    start = os.path.join(mae.mae_dir(1), "encoder.pt")
    if not os.path.exists(start):
        raise SystemExit("no mae-v1 encoder at %s" % start)
    pool, n_chest, pool_info = open_pool()
    start_sha = sha256(start)
    print("pool: %d chest rows and %d bone or spine rows" % (n_chest, len(pool) - n_chest), flush=True)
    mae.lease(card_gb + (pool.nbytes / 2 ** 30 if pool_on_card else 0.0), "mae v%d pretrain-xr" % version)
    torch.manual_seed(mae.SEED)
    with ledger.run("experiment", "mae pretrain-xr v%d" % version) as led:
        led.settings_from({"epochs": epochs, "warm_epochs": WARM_EPOCHS, "batch": batch, "px": PX,
                           "mask_ratio": mae.MASK_RATIO, "enc": mae.ENC, "dec": mae.DEC, "lr_per_256": mae.MAE_LR_PER_256,
                           "xr_repeat": XR_REPEAT, "chest_rows": n_chest, "xr_rows": len(pool) - n_chest,
                           "start_sha256": start_sha})
        hist = fit_xr(pool, n_chest, out, start, start_sha, epochs, batch, pool_info, until=until, pool_on_card=pool_on_card,
                      compile_model=compile_model)
        if not hist:
            print("nothing to do: the run is already past epoch %s" % (until,), flush=True)
            return
        if os.path.exists(os.path.join(out, "encoder.pt")):
            print("encoder written: %s" % os.path.join(out, "encoder.pt"), flush=True)
        else:
            print("stopped after epoch %d of %d; run again to continue" % (hist[-1]["epoch"], WARM_EPOCHS + epochs), flush=True)
        led.number("loss", hist[-1]["loss"]).number("seconds", sum(h["seconds"] for h in hist), "s")
        led.number("pictures_per_second", hist[-1]["pictures_per_second"], "pictures/s")
        if mae.reader.DEV.type == "cuda":
            led.number("peak_reserved_memory_gb", torch.cuda.max_memory_reserved() / 2 ** 30)


def selftest():
    import tempfile
    import pandas as pd
    real_dev = mae.reader.DEV
    mae.reader.DEV = torch.device("cpu")
    px, enc, dec = 64, {"dim": 32, "depth": 2, "heads": 2}, {"dim": 16, "depth": 1, "heads": 2}
    n_chest, n_xr, batch = 20, 6, 8
    rng = np.random.default_rng(1)
    chest = rng.integers(0, 256, (n_chest, 80, 80), dtype=np.uint8)
    xr = rng.integers(0, 256, (n_xr, 80, 80), dtype=np.uint8)
    pool = Joined(chest, xr)
    fake = {"synthetic": True}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            both = pool[np.array([0, n_chest, n_chest + n_xr - 1])]
            assert np.array_equal(both[0], chest[0]) and np.array_equal(both[1], xr[0]) and np.array_equal(both[2], xr[-1])
            assert np.array_equal(pool[3:5], np.stack([chest[3], chest[4]])), "a slice of the joined pool is not its rows"
            for ep in (1, 2):
                cnt = np.bincount(draws(n_chest, n_xr, ep), minlength=n_chest + n_xr)
                assert (cnt[:n_chest] == 1).all() and (cnt[n_chest:] == XR_REPEAT).all(), cnt
            assert not np.array_equal(draws(n_chest, n_xr, 1), draws(n_chest, n_xr, 2)), "the order does not change per epoch"
            assert np.array_equal(draws(n_chest, n_xr, 1), draws(n_chest, n_xr, 1)), "the order is not fixed by the seed"
            marked = pd.DataFrame({"image": ["a", "b"], "split": ["dev", "test"]})
            refused = False
            try:
                dev_rows(marked, "selftest")
            except SystemExit:
                refused = True
            assert refused, "a test row was accepted"
            assert len(dev_rows(marked[marked.split == "dev"], "selftest")) == 1
            start = os.path.join(tmp, "mae-v1", "encoder.pt")
            os.makedirs(os.path.dirname(start))
            mae.save_encoder(mae.make_vit(px, enc["dim"], enc["depth"], enc["heads"]), start,
                             {"px": px, "dim": enc["dim"], "depth": enc["depth"], "heads": enc["heads"], "patch": mae.PATCH,
                              "mask_ratio": mae.MASK_RATIO, "epochs": 1, "batch": batch, "seed": mae.SEED, "pool": fake})
            start_sha = sha256(start)
            start_vit = mae.load_encoder(start, px, 0.0)[0].state_dict()
            log = []
            kw = dict(say=log.append, px=px, enc=enc, dec=dec)
            warm_dir = os.path.join(tmp, "warm")
            fit_xr(pool, n_chest, warm_dir, start, start_sha, 1, batch, fake, warm=2, until=2, **kw)
            warm_state = torch.load(os.path.join(warm_dir, "partial.pt"), map_location="cpu", weights_only=False)["model"]
            for k, v in start_vit.items():
                assert torch.equal(warm_state["vit." + k], v), "the warm-up epochs moved the encoder's %s" % k
            full = os.path.join(tmp, "full")
            full_hist = fit_xr(pool, n_chest, full, start, start_sha, 1, batch, fake, warm=2, **kw)
            assert [h["epoch"] for h in full_hist] == [1, 2, 3] and [h["frozen_encoder"] for h in full_hist] == [True, True, False]
            cut = os.path.join(tmp, "cut")
            fit_xr(pool, n_chest, cut, start, start_sha, 1, batch, fake, warm=2, until=1, **kw)
            assert not os.path.exists(os.path.join(cut, "encoder.pt")), "a stopped run wrote its encoder"
            resumed = fit_xr(pool, n_chest, cut, start, start_sha, 1, batch, fake, warm=2, **kw)
            assert any("resumed from" in s for s in log) and [h["epoch"] for h in resumed] == [1, 2, 3], log[-3:]
            a = torch.load(os.path.join(full, "encoder.pt"), map_location="cpu", weights_only=False)["state"]
            b = torch.load(os.path.join(cut, "encoder.pt"), map_location="cpu", weights_only=False)["state"]
            diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
            assert diff < 1e-6, "resumed weights differ from the uninterrupted run by %g" % diff
            assert any(not torch.equal(a[k], start_vit[k]) for k in start_vit), "the epochs that train everything left the encoder as it was"
            enc_path = os.path.join(full, "encoder.pt")
            saved = torch.load(enc_path, map_location="cpu", weights_only=False)
            assert set(saved) == {"state", "settings"}, set(saved)
            assert set(saved["settings"]) == set(torch.load(start, map_location="cpu", weights_only=False)["settings"]), \
                "the settings keys differ from mae-v1's encoder.pt"
            want = set(mae.make_vit(px, enc["dim"], enc["depth"], enc["heads"]).state_dict())
            assert set(saved["state"]) == want, sorted(set(saved["state"]) ^ want)
            assert vit_detector.build(enc_path, (px, px), 2, 0.0) is not None
    finally:
        mae.reader.DEV = real_dev
    print("mae_xr selftest ok", flush=True)


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--jobs", type=int, default=JOBS, help="threads for the bone pictures (the caches' rows and the BUU files)")
    r = sub.add_parser("pretrain-xr")
    r.add_argument("--version", type=int, required=True)
    r.add_argument("--epochs", type=int, default=EPOCHS, help="epochs after the warm-up ones")
    r.add_argument("--until", type=int, default=None, help="stop after this epoch (the run resumes from partial.pt next time); default: all")
    r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--card-gb", type=float, default=4.5)
    r.add_argument("--pool-on-card", action="store_true", default=False,
                   help="load both pools onto the card once (a big card only); the same batches, values and loss")
    r.add_argument("--compile", dest="compile_model", action="store_true", default=False,
                   help="torch.compile the MAE forward for training (speed only; checkpoints keep plain keys)")
    sub.add_parser("selftest")
    return ap


def main():
    a = build_parser().parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "pool":
        build_pool(max(1, a.jobs))
    else:
        pretrain_xr(a.version, a.epochs, a.batch, a.card_gb, a.until, a.pool_on_card, a.compile_model)


if __name__ == "__main__":
    main()
