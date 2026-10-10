"""The spine X-ray corner reader's second stage: a full-resolution close look at each side-view corner.

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646 (LICENSING.md, BUU-LSPINE row). Plan and pass bars:
docs/efficiency/README.md, "Corners v2: a close look at each corner".

Why. corners.py v1 places a side view's 22 corners on a 512 x 384 canvas with maps at half that size, so a map
pixel is several pixels of the original film; its misses are 11.6 px on average (8.4 median), a whole slip
grade. Here each corner gets a second look at the ORIGINAL resolution (side view LA only).

The crop. A 160 x 160 window of the original film (grey, not resized, black outside the film) centred on the
corner v1 placed for that patient, out of fold (work/eval/spine_xray/corners-v1/LA-fold<k>.npz, all five folds
are needed). `crop` returns the patch and its top-left offset (ox, oy) in whole pixels, so a position in the
patch is a position in the film plus (ox, oy), exactly (continuous coordinates, pixel i spans [i, i + 1), as
buu.py and corners.py use them).

The net. corners.py's U-Net (stride-2 stem, five levels) with 23 input planes: the crop, and the corner's index
(0..21) as a 22-plane one-hot (a real one-hot beside the picture, not an embedding). ONE map comes out, at HALF
the crop's resolution (80 x 80, STRIDE 2 as in v1): the position is corners.py's spatial soft-argmax of that
map, trained with corners.py's loss (L1 on the coordinates plus the map's cross-entropy against a Gaussian at
the drawn corner). From scratch, no pretrained weights (LICENSING.md).

Training, fold k: the patients of the other four folds only. A sample is (patient, corner); its crop is
centred on THAT patient's v1 out-of-fold corner, so the net trains on exactly the misses it will meet, plus a
fresh shift drawn every epoch, uniform over whole pixels in +-4 in x and in y, and a brightness, contrast and
gamma nudge. No flips (a side film is never flipped). The last epoch is the model. Seeded.

How the crops reach the card. The films are decoded once (2,000 jpgs at original size) and every corner's
window is cut then, 168 x 168 around the rounded v1 corner (the 160 px crop plus the 4 px of shift each side),
and kept as one uint8 array (2,000 x 22 x 168 x 168, 1.2 GB) cached under work/eval/spine_xray/cache/ (keyed
by a fingerprint of the v1 corners), so a second run does not read the jpgs again. That array lives in host
memory, NOT on the card (it would not fit in the 1.5 GB share beside the net); each batch is an index into it,
copied to the card as uint8, and the shift and the nudge happen on the card.

Judging fold k: every held-out patient's 22 corners are refined from the unshifted crop and mapped back to
original pixels; <WORK>/eval/spine_xray/corners-v<N>/LA-fold<k>.npz holds the same keys v1's file has (ids,
pred, truth, base, base_size, fold, fingerprint), plus `v1`, v1's corners for the same patients.
After the folds: (a) corners, per-patient mean miss refined, v1 and drawn, refined minus v1 with a paired
patient bootstrap, KEPT when its 95% interval is wholly below 0; (b) slips, slip.py's amended score on the
refined corners against the drawn corners', slip.py's MARGIN rule, AUROC beside v1's 0.8093, accuracy,
sensitivity and specificity at thresholds chosen inside each training fold, per-level AUROC. Both verdicts and
their numbers go to one ledger row.

Setting extra_landmarks (default false; --set extra_landmarks=true): idea 24, the spondylolisthesis-landmarks pictures
(landmarks.py) join every fold's training and are never held out. Each one is rendered at BUU's vertebra size (factor 1)
and its windows are centred on the drawn corner plus a miss drawn from corners v1's own out-of-fold errors (one
training patient's whole 22-corner error, so the misses a net will meet are realistic); a corner the set lacks makes
no sample. Held-out files are unchanged. Off, nothing changes, the fingerprint included.

Setting landmark_fold=K (idea 119, 2026-10-08; needs extra_landmarks=true): the landmark set's fold K never trains, and the
other folds' pictures come from landmarks_test.training_set, prepared without their labels as the test prepares its pictures
(read and oriented as the test reads them; film pixels at BUU's film size; points through the same frames). Their windows are
centred as above. Off (null), nothing changes, the fingerprint included.

Usage: python corners_refine.py --version N [--folds 0,1,2,3,4] [--epochs 40] [--set extra_landmarks=true]
The card share is RAIDIOLOGY_CORNERS_CARD_GB (default 1.5), leased once as a short job of 15 minutes.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import card, ledger, polite  # noqa: E402
from core import ckpt, foldck  # noqa: E402
from core import paths, sets  # noqa: E402
import buu  # noqa: E402
import corners  # noqa: E402
import slip  # noqa: E402

DEV = corners.DEV
VIEW = slip.VIEW
V1_VERSION = 1                  # the corner reader whose out-of-fold corners place every crop
V1_AUROC = 0.8093               # slip.py on corners v1, out of fold, all 2,000 patients
CROP = 160
SHIFT = 4                       # whole pixels of shift each side, drawn per sample per epoch
WIN = CROP + 2 * SHIFT          # the window cut once around each v1 corner
N_CORNERS = 22
CARD_GB = 1.5
SHORT_MIN = 15
SETTINGS = {
    "epochs": 40, "batch": 64, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 2,
    "widths": [24, 48, 96, 128, 192],
    "jitter": 0.15, "gamma": 0.3,
    "sigma": 4.0,         # the Gaussian target's spread, crop pixels
    "heat_weight": 0.5,
    "l1_unit": 4.0,       # crop pixels the coordinate L1 is divided by
    "seed": 0,
    "extra_landmarks": False,   # add the spondylolisthesis-landmarks pictures to every fold's training (see the docstring)
    "landmark_fold": None,      # idea 119: 0 to 4, the landmark set's fold kept out of training (needs extra_landmarks)
}


def out_dir(version):
    return corners.out_dir(version)


def cache_dir():
    return os.path.join(paths.WORK, "eval", "spine_xray", "cache")


def limit_card_memory():
    """Lease the card as a short job before touching CUDA, then cap the process at that size."""
    if DEV.type != "cuda":
        return
    gb = float(os.environ.get("RAIDIOLOGY_CORNERS_CARD_GB", CARD_GB))
    card.lease(gb, "spine corners refine", short_min=SHORT_MIN)
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, gb * 2 ** 30 / total), torch.cuda.current_device())


# --- the crop ----------------------------------------------------------------------------------------

def crop(img, cx, cy, size=CROP):
    """A size x size window of img (H, W) uint8 centred on (cx, cy) (continuous x, y), black outside the film.
    Returns the patch and its top-left offset (ox, oy) in whole pixels: film position = patch position + (ox, oy)."""
    ox = int(math.floor(cx + 0.5)) - size // 2
    oy = int(math.floor(cy + 0.5)) - size // 2
    h, w = img.shape[:2]
    patch = np.zeros((size, size), np.uint8)
    x0, x1 = max(ox, 0), min(ox + size, w)
    y0, y1 = max(oy, 0), min(oy + size, h)
    if x1 > x0 and y1 > y0:
        patch[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = img[y0:y1, x0:x1]
    return patch, (ox, oy)


def window_offsets(v1_pts):
    """(N, 22, 2) v1 corners -> (N, 22, 2) int top-left offsets of the WIN windows."""
    return np.floor(np.asarray(v1_pts, np.float64) + 0.5).astype(np.int64) - WIN // 2


def fingerprint_of(v1_pts):
    h = hashlib.sha1(np.ascontiguousarray(window_offsets(v1_pts)).tobytes())
    h.update(b"win%d" % WIN)
    return h.hexdigest()


def load_windows(ids, v1_pts):
    """(N, 22, WIN, WIN) uint8: each patient's 22 windows cut from the original film around the v1 corners.
    Cached as one .npy (a memory copy of 1.2 GB) keyed by the corners' fingerprint."""
    fp = fingerprint_of(v1_pts)
    d = cache_dir()
    npy, meta = os.path.join(d, "LA-refine-windows.npy"), os.path.join(d, "LA-refine-windows.json")
    if os.path.exists(npy) and os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("fingerprint") == fp and m.get("ids") == [int(i) for i in ids]:
                print("windows: read %s" % npy, flush=True)
                return np.load(npy)
        except Exception:  # noqa: BLE001  an unreadable cache is simply rebuilt
            pass
    from PIL import Image
    jpgs = buu.files(VIEW, ext="jpg")
    out = np.zeros((len(ids), N_CORNERS, WIN, WIN), np.uint8)
    for n, i in enumerate(ids):
        with Image.open(jpgs[int(i)]) as im:
            img = np.asarray(im.convert("L"), dtype=np.uint8)
        for c in range(N_CORNERS):
            out[n, c] = crop(img, v1_pts[n, c, 0], v1_pts[n, c, 1], WIN)[0]
        if (n + 1) % 250 == 0:
            print("  %s: %d / %d films cut" % (VIEW, n + 1, len(ids)), flush=True)
    os.makedirs(d, exist_ok=True)
    tmp = npy + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, npy)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "ids": [int(i) for i in ids]}, f)
    print("windows: wrote %s" % npy, flush=True)
    return out


# --- the net -----------------------------------------------------------------------------------------

def make_net(widths):
    """corners.py's U-Net on 1 + 22 input planes with one output map."""
    net = corners.UNet(widths, 1)
    net.stem[0] = nn.Conv2d(1 + N_CORNERS, widths[0], 3, stride=2, padding=1, bias=False)
    return net


def net_input(x, corner):
    """x (B, 1, CROP, CROP) standardised, corner (B,) long -> (B, 23, CROP, CROP): picture, then the one-hot planes."""
    hot = F.one_hot(corner, N_CORNERS).to(x.dtype)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
    return torch.cat([x, hot], dim=1)


def nudge(x, g, o):
    """x (B, 1, H, W) in [0, 1]: brightness, contrast and gamma, as corners.py's augment (no warp, no flip)."""
    b = x.shape[0]

    def u():
        return torch.rand(b, 1, 1, 1, device=x.device, generator=g) * 2 - 1

    j = o["jitter"]
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * (1.0 + u() * j) + m + u() * j * 0.5).clamp(0.0, 1.0)
    if o["gamma"]:
        x = x.clamp_min(1e-6) ** torch.exp(u() * o["gamma"])
    return x


def shifted(win, dx, dy):
    """win (B, WIN, WIN) on the device, whole-pixel shifts dx, dy (B,) in [-SHIFT, SHIFT] -> (B, CROP, CROP): the
    CROP window whose top-left sits at (SHIFT + dx, SHIFT + dy) inside win."""
    ar = torch.arange(CROP, device=win.device)
    rows = (SHIFT + dy)[:, None] + ar[None]
    cols = (SHIFT + dx)[:, None] + ar[None]
    b = torch.arange(win.shape[0], device=win.device)
    return win[b[:, None, None], rows[:, :, None], cols[:, None, :]]


def predict(net, windows, pi, mean, sd, batch=128):
    """Refined corners (len(pi), 22, 2) in crop coordinates of each corner's unshifted CROP window."""
    net.eval()
    pairs = [(p, c) for p in pi for c in range(N_CORNERS)]
    out = []
    with torch.no_grad():
        for a in range(0, len(pairs), batch):
            chunk = pairs[a:a + batch]
            w = np.stack([windows[p, c, SHIFT:SHIFT + CROP, SHIFT:SHIFT + CROP] for p, c in chunk])
            x = (torch.from_numpy(w).to(DEV).float().div_(255.0).unsqueeze(1) - mean) / sd
            cc = torch.tensor([c for _, c in chunk], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                logits = net(net_input(x, cc))
            out.append(corners.soft_argmax(logits)[0][:, 0].cpu())
    return torch.cat(out).numpy().reshape(len(pi), N_CORNERS, 2)


PROBE = {}   # the selftest reads what the last train_fold did: steps run, where it resumed, what it loaded


def train_fold(fold, data, o, pace, partial=None, fp=None):
    """Train on the other folds' patients, return the net and the held-out patients' corners in original pixels.
    With `partial` and `fp`, a partial checkpoint is saved after every epoch and whenever core.ckpt.Every is due
    (300 s) inside one; a partial with the same fingerprint is continued from (corners.save_progress / restore)."""
    fold_of, windows, truth, v1 = data["fold"], data["windows"], data["truth"], data["v1"]
    off = window_offsets(v1)
    tr = np.where(fold_of != fold)[0]
    te = np.where(fold_of == fold)[0]
    torch.manual_seed(o["seed"] + fold)
    rng = np.random.default_rng(o["seed"] + fold)
    g_dev = torch.Generator(device=DEV).manual_seed(o["seed"] + fold)
    probe = windows[rng.choice(tr, 256), rng.integers(0, N_CORNERS, 256), SHIFT:SHIFT + CROP, SHIFT:SHIFT + CROP]
    mean, sd = float(probe.mean() / 255.0), max(float(probe.std() / 255.0), 1e-3)
    net = make_net(o["widths"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    valid = data.get("valid")
    if valid is None:
        n_pairs = len(tr) * N_CORNERS
    else:                                                  # landmark pictures lack some corners: only drawn ones are samples
        pairs = np.flatnonzero(valid[tr].ravel())
        n_pairs = len(pairs)
    per_epoch = n_pairs // o["batch"]
    total, warm = o["epochs"] * per_epoch, o["warmup"] * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    print("fold %d: train %d patients (%d crops), held out %d, %d steps an epoch"
          % (fold, len(tr), n_pairs, len(te), per_epoch), flush=True)
    if valid is not None:
        print("fold %d: %d extra landmark pictures trained on" % (fold, int((fold_of[tr] < 0).sum())), flush=True)
    truth_t = torch.from_numpy(truth.astype(np.float32)).to(DEV)             # (N, 22, 2) original pixels
    off_t = torch.from_numpy(off.astype(np.float32)).to(DEV)
    took, start_ep, start_s, order_saved, sums, prior = [], 1, 0, None, (0.0, 0.0), 0.0
    PROBE.clear()
    PROBE.update(steps_run=0, resumed_epoch=None)
    ck = ckpt.load_partial(partial, fp) if partial else None
    if ck is not None and 1 <= ck["epoch"] <= o["epochs"] + 1:
        corners.restore_progress(ck, net, opt, sched, [g_dev])
        rng.bit_generator.state = ck["np_rng"]
        start_ep, start_s, order_saved, took, sums = ck["epoch"], ck["step"], ck["order"], ck["took"], ck["sums"]
        prior = ck["seconds"]
        PROBE.update(resumed_epoch=start_ep, opt=copy.deepcopy(opt.state_dict()), sched=copy.deepcopy(sched.state_dict()))
        print("fold %d: resumed mid-fold at epoch %d of %d (partial checkpoint)" % (fold, start_ep, o["epochs"]), flush=True)
    del ck
    every, t_fold = ckpt.Every(), time.time()
    for ep in range(start_ep, o["epochs"] + 1):
        t0 = time.time()
        net.train()
        if ep == start_ep and order_saved is not None:           # the epoch's draw was made before the save
            order, s_from = order_saved, start_s
            tot_loss, tot_px = (torch.tensor(v, device=DEV) for v in sums)
        else:
            order, s_from = rng.permutation(n_pairs), 0
            tot_loss = tot_px = torch.zeros((), device=DEV)
        for s in range(s_from, per_epoch):
            PROBE["steps_run"] += 1
            began = time.time()
            idx = order[s * o["batch"]:(s + 1) * o["batch"]]
            if valid is None:
                pi, ci = tr[idx // N_CORNERS], idx % N_CORNERS
            else:
                flat = pairs[idx]
                pi, ci = tr[flat // N_CORNERS], flat % N_CORNERS
            win = torch.from_numpy(windows[pi, ci]).to(DEV)
            dxy = torch.randint(-SHIFT, SHIFT + 1, (len(idx), 2), device=DEV, generator=g_dev)
            x = shifted(win, dxy[:, 0], dxy[:, 1]).float().div_(255.0).unsqueeze(1)
            x = nudge(x, g_dev, o)
            pit, cit = torch.from_numpy(pi).to(DEV), torch.from_numpy(ci).to(DEV)
            # drawn corner in the shifted crop's own coordinates: film position minus the crop's top-left
            target = (truth_t[pit, cit] - off_t[pit, cit] - SHIFT - dxy.float())[:, None, :]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                logits = net(net_input((x - mean) / sd, cit))
            loss, px = corners.loss_of(logits, target, o)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss, tot_px = tot_loss + loss.detach(), tot_px + px.detach()
            if DEV.type == "cuda" and (s + 1) % 8 == 0:
                torch.cuda.synchronize()
            pace.rest(time.time() - began)       # core/polite.py: half the card while somebody is at it
            if partial and s + 1 < per_epoch and every.due():
                corners.save_progress(partial, fp, net, opt, sched, [g_dev], epoch=ep, step=s + 1, order=order, took=took,
                                      sums=(float(tot_loss), float(tot_px)), seconds=prior + time.time() - t_fold,
                                      np_rng=rng.bit_generator.state)
                every.mark()
                foldck.stop_after("%d.%d" % (ep, s + 1))
        took.append(time.time() - t0)
        if ep == 1 or ep % 5 == 0 or ep == o["epochs"]:
            print("  epoch %3d  loss %.4f  corner miss on training batches %.2f px  %.0f s"
                  % (ep, float(tot_loss) / per_epoch, float(tot_px) / per_epoch, took[-1]), flush=True)
        if partial and ep < o["epochs"]:
            corners.save_progress(partial, fp, net, opt, sched, [g_dev], epoch=ep + 1, step=0, order=None, took=took,
                                  sums=(0.0, 0.0), seconds=prior + time.time() - t_fold, np_rng=rng.bit_generator.state)
            every.mark()
            foldck.stop_after(ep)
    del opt, sched
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    pred = predict(net, windows, te, mean, sd)
    # the unshifted crop's top-left is the window's plus SHIFT; crop position + top-left = film position
    pred_o = (pred + (off[te] + SHIFT).astype(np.float32)).astype(np.float32)
    return net, te, pred_o, {"mean": mean, "sd": sd, "epoch_seconds": [round(v, 2) for v in took]}


# --- data and judging ------------------------------------------------------------------------------------

def landmark_rows(o, v1, truth):
    """The landmark pictures as training rows (idea 24): dict of ids (-1), fold (-1: in every fold's training, never
    held out), truth, v1 (the window centres: the drawn corner plus a v1 miss), windows, valid (M, 22) bool.
    A picture's miss is one random training patient's whole 22-corner v1 error, so the same corner index gets
    a realistic miss and the corners miss together as they do in a real picture. With landmark_fold (idea 119) the
    films and points are landmarks_test.training_set's (no label-fitted render; fold K left out)."""
    rng = np.random.default_rng(o["seed"])
    err = v1 - truth                                                    # (N, 22, 2) out-of-fold errors
    if o.get("landmark_fold") is None:
        import landmarks
        imgs, pts = [], []
        for r in landmarks.load(part="train"):
            img, p = landmarks.render(r, 1.0)
            imgs.append(img)
            pts.append(p.astype(np.float64))
    else:
        import landmarks_test
        _recs, e = landmarks_test.training_set(o["landmark_fold"])
        imgs, pts = list(e["film"]), [p.astype(np.float64) for p in e["film_pts"]]
    n = len(imgs)
    truths, valids = np.zeros((n, N_CORNERS, 2)), np.zeros((n, N_CORNERS), bool)
    for m, p in enumerate(pts):
        ok = ~np.isnan(p).any(axis=-1)
        p = p.copy()
        p[~ok] = p[ok].mean(0)                                          # a stand-in, never trained on
        truths[m], valids[m] = p, ok
    centre = truths + err[rng.integers(0, len(err), n)]
    windows = np.zeros((n, N_CORNERS, WIN, WIN), np.uint8)
    for m in range(n):
        for c in range(N_CORNERS):
            if valids[m, c]:
                windows[m, c] = crop(imgs[m], centre[m, c, 0], centre[m, c, 1], WIN)[0]
    print("extra_landmarks: %d landmark pictures, %d of %d corners present" % (n, int(valids.sum()), valids.size), flush=True)
    return {"ids": np.full(n, -1), "fold": np.full(n, -1), "truth": truths, "v1": centre, "windows": windows,
            "valid": valids}


def load_everything(version, o=None):
    """Ids, folds, drawn corners, v1's out-of-fold corners (all five folds, original pixels) and the windows;
    with o["extra_landmarks"], the landmark pictures after them (fold -1, `valid` marks the corners that exist)."""
    ids, truth, classes = buu.annotations(VIEW)
    miss = [os.path.join(out_dir(V1_VERSION), "%s-fold%d.npz" % (VIEW, k)) for k in range(buu.FOLDS)]
    miss = [p for p in miss if not os.path.exists(p)]
    if miss:
        raise SystemExit("corners v%d's out-of-fold corners are needed for all %d folds; missing: %s\n"
                         "train them with: python corners.py train --view %s --version %d --folds all"
                         % (V1_VERSION, buu.FOLDS, ", ".join(miss), VIEW, V1_VERSION))
    r = corners.load_oof(VIEW, V1_VERSION, list(range(buu.FOLDS)))
    order = {int(i): n for n, i in enumerate(r["ids"])}
    sel = [order[int(i)] for i in ids]                       # v1's rows in ascending patient id, as `ids`
    v1 = r["pred"][sel].astype(np.float64)
    base, base_size = r["base"][sel], r["base_size"][sel]
    windows = load_windows(ids, v1)
    data = {"ids": ids, "truth": truth, "classes": classes, "fold": buu.folds_of(ids), "v1": v1,
            "base": base, "base_size": base_size, "windows": windows}
    if o and o.get("extra_landmarks"):
        e = landmark_rows(o, v1, truth.astype(np.float64))
        n, m = len(ids), len(e["ids"])
        data = {**data, "windows": np.concatenate([windows, e["windows"]]),
                "ids": np.concatenate([ids, e["ids"]]), "fold": np.concatenate([data["fold"], e["fold"]]),
                "truth": np.concatenate([truth, e["truth"]]), "v1": np.concatenate([v1, e["v1"]]),
                "valid": np.concatenate([np.ones((n, N_CORNERS), bool), e["valid"]]),
                "base": np.concatenate([base, np.zeros((m, N_CORNERS, 2), base.dtype)]),
                "base_size": np.concatenate([base_size, np.zeros((m, N_CORNERS, 2), base_size.dtype)])}
    return data


def write_fold(d, k, o, fp, data, net, te, pred, info, partial):
    """The fold's final files (the npz that marks it done, then the model), then its partial checkpoint goes."""
    np.savez(os.path.join(d, "%s-fold%d.npz" % (VIEW, k)), ids=data["ids"][te], pred=pred,
             truth=data["truth"][te].astype(np.float32), base=data["base"][te], base_size=data["base_size"][te],
             v1=data["v1"][te].astype(np.float32), fold=np.int64(k), fingerprint=fp)
    torch.save({"state": net.state_dict(), "settings": o, "info": info, "view": VIEW, "fingerprint": fp},
               os.path.join(d, "%s-fold%d.pt" % (VIEW, k)))
    ckpt.clear_partial(partial)


def train(version, folds, settings):
    o = {**SETTINGS, **settings}
    corners.check_landmark_fold(VIEW, o)
    card.require_card(DEV, torch)
    # 2026-10-03: leased before the data is loaded (a job waiting with its data held up to 6.9 GB; the machine
    # ran out of committed memory at 23:21 and a training job crashed)
    limit_card_memory()
    data = load_everything(version, o)
    d = out_dir(version)
    os.makedirs(d, exist_ok=True)
    # a setting that is off stays out of the fingerprint, so versions trained before it existed still match
    fp = json.dumps({"view": VIEW, "version": version, "settings": corners.fingerprint_settings(o), "v1": V1_VERSION},
                    sort_keys=True)
    for k in folds:
        npz = os.path.join(d, "%s-fold%d.npz" % (VIEW, k))
        if os.path.exists(npz) and os.environ.get("RAIDIOLOGY_NO_RESUME") != "1":
            try:
                if str(np.load(npz)["fingerprint"]) == fp:
                    print("fold %d: %s already written for these settings, skipped" % (k, npz), flush=True)
                    continue
            except Exception:  # noqa: BLE001  an unreadable file is simply redone
                pass
        with ledger.run("corners", "refine %s fold %d v%d" % (VIEW, k, version)) as led:
            led.settings_from({"view": VIEW, "settings": o})
            partial = os.path.join(d, "%s-fold%d.partial.pt" % (VIEW, k))
            net, te, pred, info = train_fold(k, data, o, polite.Pace(), partial, fp + "|fold%d" % k)
            write_fold(d, k, o, fp, data, net, te, pred, info, partial)
            miss = np.linalg.norm(pred - data["truth"][te], axis=-1).mean()
            miss1 = np.linalg.norm(data["v1"][te] - data["truth"][te], axis=-1).mean()
            led.number("corner_miss_px", miss, "px").number("v1_miss_px", miss1, "px")
            led.number("epochs", o["epochs"], "count").number("train_images", int((data["fold"] != k).sum()), "count")
            if DEV.type == "cuda":
                led.number("peak_memory_gb", torch.cuda.max_memory_allocated() / 2 ** 30)
                torch.cuda.reset_peak_memory_stats()
            print("fold %d: held-out corner miss %.2f px refined, %.2f px v1 -> %s" % (k, miss, miss1, npz), flush=True)
        del net
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


def judge(version, folds, epochs):
    t0 = time.time()
    r2 = corners.load_oof(VIEW, version, folds)
    r1 = corners.load_oof(VIEW, V1_VERSION, folds)
    ids, truth_all, classes = buu.annotations(VIEW)
    fold_all = buu.folds_of(ids)
    label_all = buu.slip_label(classes)
    level_all = np.isin(classes[:, 1:10:2], buu.SLIP_CLASSES).astype(int)
    where = {int(i): n for n, i in enumerate(ids)}
    keep = np.array(sorted(where[int(i)] for i in r2["ids"]))
    o2 = {int(i): n for n, i in enumerate(r2["ids"])}
    o1 = {int(i): n for n, i in enumerate(r1["ids"])}
    ref = r2["pred"][[o2[int(i)] for i in ids[keep]]].astype(np.float64)
    v1 = r1["pred"][[o1[int(i)] for i in ids[keep]]].astype(np.float64)
    truth = truth_all[keep].astype(np.float64)
    pid, fold, label, level_label = ids[keep], fold_all[keep], label_all[keep], level_all[keep]
    n = len(pid)

    # (a) corners
    m_ref = corners.miss_per_patient(ref, truth)
    m_v1 = corners.miss_per_patient(v1, truth)
    print("corners refine: %d held-out patients in folds %s, per-patient mean corner miss at full resolution (px)"
          % (n, ",".join(map(str, folds))))
    for name, m in (("refined", m_ref), ("v1", m_v1)):
        print("  %-8s mean %6.2f  median %6.2f   per fold mean: %s" % (
            name, float(m.mean()), float(np.median(m)),
            ", ".join("%d: %.2f" % (k, float(m[fold == k].mean())) for k in folds)))
    diff = m_ref - m_v1
    dmean = float(diff.mean())
    bm = corners.boot_means(diff[:, None])[:, 0]
    clo, chi = (float(v) for v in corners.ci(bm))
    kept_c = bool(chi < 0)
    print("refined minus v1: %+.2f px (paired patient-bootstrap 95%% CI %+.2f to %+.2f, %d resamples, seed %d)"
          % (dmean, clo, chi, corners.BOOT, corners.BOOT_SEED))
    print("Verdict (corners, refined minus v1): %s" % ("KEPT" if kept_c else "NOT KEPT"))

    # (b) slips
    if len(folds) < 2:
        raise SystemExit("a threshold chosen inside the training folds needs at least two folds scored; got %s" % folds)
    sl, true_sl = slip.slips(ref), slip.slips(truth)
    score = slip.amended(sl, fold)
    dscore = slip.amended(true_sl, fold)
    n_pos = int(label.sum())
    lvl_score = np.zeros((n, 5))
    print("per level (amended slip on the refined corners, marks as BUU's class 3 or 4):")
    for i, name in enumerate(slip.LEVELS):
        pos = int(level_label[:, i].sum())
        lvl = np.zeros(n)
        for k in folds:
            te = fold == k
            lvl[te] = np.abs(sl[te, i] - np.median(sl[~te, i]))
        lvl_score[:, i] = lvl
        lv = slip.auroc(lvl, level_label[:, i]) if pos >= 5 else float("nan")
        print("  %-6s marks %3d  AUROC for the level's marks %s" % (name, pos, "%.3f" % lv if lv == lv else "n/a"))
    call = np.zeros(n, bool)
    cuts = {}
    for k in folds:
        tr_, te_ = fold != k, fold == k
        cuts[k] = slip.youden(score[tr_], label[tr_])
        call[te_] = score[te_] >= cuts[k]
    acc, sens, spec = slip.rates(call, label)
    idx = np.random.default_rng(slip.BOOT_SEED).integers(0, n, (slip.BOOT, n))
    accs, senss, specs = [], [], []
    for rr in idx:
        ac, se, sp = slip.rates(call[rr], label[rr])
        accs.append(ac), senss.append(se), specs.append(sp)

    def boot_auc(sc):
        return np.array([slip.auroc(sc[rr], label[rr]) for rr in idx])

    auc, aucs = slip.auroc(score, label), boot_auc(score)
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    dauc, daucs = slip.auroc(dscore, label), boot_auc(dscore)
    dlo, dhi = np.nanpercentile(daucs, [2.5, 97.5])
    dflo, dfhi = np.nanpercentile(aucs - daucs, [2.5, 97.5])                 # same resample for both
    d = auc - dauc
    print("Refined corners, amended score: AUROC %.4f (patient-bootstrap 95%% CI %.4f to %.4f, %d resamples, seed %d)"
          % (auc, lo, hi, slip.BOOT, slip.BOOT_SEED))
    print("Drawn corners, amended score: AUROC %.4f (%.4f to %.4f)" % (dauc, dlo, dhi))
    print("Refined minus drawn corners: %+.4f (paired patient-bootstrap 95%% CI %+.4f to %+.4f)" % (d, dflo, dfhi))
    print("Refined minus slips v1 (corners v1 measured, AUROC %.4f): %+.4f" % (V1_AUROC, auc - V1_AUROC))
    print("thresholds chosen inside each training fold: %s" % ", ".join("fold %d %.3f" % (k, c) for k, c in cuts.items()))
    print("at those thresholds: accuracy %.3f (%.3f to %.3f; the paper's %.3f), sensitivity %.3f (%.3f to %.3f), "
          "specificity %.3f (%.3f to %.3f); a reader that never calls a slip scores %.3f"
          % (acc, *np.percentile(accs, [2.5, 97.5]), slip.PAPER_ACCURACY, sens, *np.percentile(senss, [2.5, 97.5]),
             spec, *np.percentile(specs, [2.5, 97.5]), 1 - n_pos / n))
    kept_s = bool(dflo >= -slip.MARGIN)
    print("Verdict (slips, refined not worse than drawn corners, margin %.2f): %s"
          % (slip.MARGIN, "KEPT" if kept_s else "NOT KEPT"))
    ver = slip.verified_report(pid, label, level_label, score, call, lvl_score)
    numbers = {**ver, "corner_miss_mean_px": (m_ref.mean(), "px"), "corner_miss_median_px": (np.median(m_ref), "px"),
               "v1_miss_mean_px": (m_v1.mean(), "px"), "v1_miss_median_px": (np.median(m_v1), "px"),
               "miss_diff_px": (dmean, "px"), "miss_diff_low_px": (clo, "px"), "miss_diff_high_px": (chi, "px"),
               "corners_kept": (float(kept_c), ""),
               "auroc": (auc, ""), "auroc_low": (lo, ""), "auroc_high": (hi, ""), "auroc_drawn": (dauc, ""),
               "auroc_diff": (d, ""), "auroc_diff_low": (dflo, ""), "auroc_diff_high": (dfhi, ""),
               "auroc_v1": (V1_AUROC, ""), "accuracy": (acc, ""), "sensitivity": (sens, ""), "specificity": (spec, ""),
               "patients": (n, "count"), "patients_with_slip": (n_pos, "count"), "epochs": (epochs, "count"),
               "slips_kept": (float(kept_s), "")}
    numbers = {k: (float(v), u) for k, (v, u) in numbers.items()}               # numpy floats are not JSON
    settings = {"source": "crops placed by corners v%d out-of-fold corners" % V1_VERSION,
                "folds": ",".join(map(str, folds)), "epochs": epochs, "crop": CROP, "shift_px": SHIFT,
                "corners_verdict": "KEPT" if kept_c else "NOT KEPT", "slips_verdict": "KEPT" if kept_s else "NOT KEPT",
                "bar": "refined minus v1 miss, CI wholly below 0; slips AUROC not worse than drawn, margin %.2f" % slip.MARGIN}
    ledger.write("experiment", "spine xray corners v2: a close look at each corner, v%d out of fold" % version,
                 time.time() - t0, numbers, settings=settings,
                 note="modalities/spine_xray/eval/corners_refine.py; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


def landmark_fold_checks():
    """idea 119, no files, no net: landmark_rows keeps fold K's landmark pictures out (a stand-in for
    landmarks_test.training_set, made with landmarks.split_fold on made-up records), and the setting is checked."""
    import landmarks
    import landmarks_test
    recs = [{"base": "%d-M-050Y1" % (2800 + i)} for i in range(5)] + [{"base": "Case_%d" % i} for i in range(3)]
    table = {r["base"]: i % 5 for i, r in enumerate(recs)}

    def stand_in(exclude):
        train, _held = landmarks.split_fold(recs, table, exclude)
        return train, {"film": [np.zeros((60, 50), np.uint8)] * len(train), "film_pts": np.full((len(train), 22, 2), 20.0)}

    g = np.random.default_rng(0)
    v1 = g.random((4, N_CORNERS, 2)) * 100 + 20
    real = landmarks_test.training_set
    landmarks_test.training_set = stand_in
    try:
        e = landmark_rows({"seed": 0, "landmark_fold": 1}, v1, v1 + 1.0)
    finally:
        landmarks_test.training_set = real
    kept = len([r for r in recs if table[r["base"]] != 1])
    assert len(e["ids"]) == kept and (e["fold"] == -1).all() and e["windows"].shape == (kept, N_CORNERS, WIN, WIN), "fold 1 never trains"
    assert e["valid"].all(), "every corner of a stand-in picture is drawn"
    try:
        corners.check_landmark_fold(VIEW, {**SETTINGS, "landmark_fold": 1, "extra_landmarks": False})
    except SystemExit:
        pass
    else:
        raise AssertionError("landmark_fold without extra_landmarks was allowed")
    assert "landmark_fold" not in corners.fingerprint_settings(SETTINGS)
    print("corners_refine landmark_fold checks ok")


def selftest():
    """CPU, synthetic tiny films, under a minute: python corners_refine.py selftest (CUDA_VISIBLE_DEVICES empty)."""
    import shutil
    import tempfile
    landmark_fold_checks()
    g = np.random.default_rng(0)
    n = 10
    truth = g.random((n, N_CORNERS, 2)) * 150 + 20
    data = {"ids": np.arange(n), "fold": np.arange(n) % 5, "truth": truth, "v1": truth + g.normal(size=truth.shape),
            "windows": g.integers(0, 256, (n, N_CORNERS, WIN, WIN), dtype=np.uint8),
            "base": np.zeros((n, N_CORNERS, 2), np.float32), "base_size": np.zeros((n, N_CORNERS, 2), np.float32)}
    o = {**SETTINGS, "epochs": 4, "batch": 16, "widths": [4, 4, 4, 4, 4], "warmup": 1, "seed": 1}
    root = tempfile.mkdtemp(prefix="refine-selftest-")

    class _NoPace:
        def rest(self, seconds):
            pass

    def run(name, stop):
        if stop:
            os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = stop
        else:
            os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
        partial = os.path.join(root, name + ".partial.pt")
        net, te, pred, info = train_fold(0, data, o, _NoPace(), partial, "fp|fold0")
        os.makedirs(os.path.join(root, name), exist_ok=True)
        write_fold(os.path.join(root, name), 0, o, "fp", data, net, te, pred, info, partial)
        return net

    try:
        corners.resume_checks("corners_refine", run, root, o["epochs"], PROBE)
    finally:
        os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
        os.environ.pop("RAIDIOLOGY_SAVE_EVERY_S", None)
        shutil.rmtree(root, ignore_errors=True)


def main():
    if sys.argv[1:] == ["selftest"]:
        return selftest()
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, required=True, help="names the output folder corners-v<N> (use 2)")
    ap.add_argument("--folds", default="0,1,2,3,4", help="'all' or comma-separated fold numbers (default all five)")
    ap.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    ap.add_argument("--set", action="append", default=[], metavar="key=<json>",
                    help="override: extra_landmarks (add the landmark pictures to every fold's training, default false)")
    a = ap.parse_args()
    folds = corners.parse_folds(a.folds)
    if len(folds) < 2:
        raise SystemExit("the slips verdict needs at least two folds scored; got %s" % folds)
    if a.version == V1_VERSION:
        raise SystemExit("version %d is corners v1's folder; pick another" % V1_VERSION)
    s = {"epochs": a.epochs, **sets.parse(a.set, known=SETTINGS)}
    train(a.version, folds, s)
    judge(a.version, folds, a.epochs)


if __name__ == "__main__":
    main()
