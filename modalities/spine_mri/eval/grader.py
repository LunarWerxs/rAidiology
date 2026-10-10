"""The disc grader (Phase 2): a small multi-task network on crops.py's 2.5D crops, trained from
scratch (no pretrained backbone) on SPIDER development patients only.

Model: 10 input channels (T2 and T1, 5 planes each) -> 4 conv stages -> pooled features plus the
has-T1 flag -> an ordinal head for Pfirrmann (4 cumulative logits, P(grade > k)) and one sigmoid
logit per binary item. Pfirrmann = 1 + the number of cumulative probabilities above 0.5.

Protocol, all on development folds:
- outer 5-fold CV by patient (split.json's folds): model k trains on the other four folds, with 15%
  of its training patients held back for early stopping; its predictions on fold k are the
  out-of-fold (OOF) predictions.
- nested ensembles: for each outer fold k, four inner models (each trained on three of the other
  four folds) are averaged on fold k. These mimic the shipped fold ensemble, so the shipped
  temperatures are fitted on them.
- temperature (one per head output) and thresholds (Youden's J per binary item) are cross-fitted:
  the values applied to fold k are fitted on the other folds' predictions only.
- the shipped grader is the 5 outer models averaged, with temperatures and thresholds fitted on
  all nested-ensemble OOF predictions; the test split is never touched here.
Usage: python grader.py train [--version 1] [--seed N] [--extra lsma] [--set key=<json> ...]
                                [--both-localisations]
       | predict <dev|test> | patient <t2.nii.gz> <labels.nii.gz> [t1.nii.gz]"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import ckpt as ckpt_mod, foldck, ledger, paths, sets  # noqa: E402
paths.use("spine_mri")
from prepare import WORK  # noqa: E402
from spider import ITEMS, SEED, grades  # noqa: E402

MODELS = paths.MODELS
DEV = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")   # none visible: CPU
# A ceiling on the card this process may take: RAIDIOLOGY_GRADER_CARD_GB, 3.5 by default (a wider net sets more). Without one
# the allocator's cache only grows; on 2026-10-03 the wider grader (v108) held 5.1 GB beside the X-ray
# trainer's 4.9 and the desktop, the 12 GB card overfilled, and Windows stalled the X-ray trainer for
# minutes with every thread waiting. At the ceiling the allocator hands back its cache and retries; the
# numbers do not change. Each run logs its peak, so the ceiling can be set from what it really needs.
# The process also waits here for its share of the card (core/card.py lease).
# 3.0 from 2026-10-05 08:30 (v128, v129, v134 and v138 peaked at 2.16 to 2.23 GB allocated under 3.5), back to 3.5
# at 12:15: v131 ran out at 3.0 in its last outer fold. The ceiling counts what the allocator reserves, and the
# 890 MB training tensor each seed puts on the card left 2.17 GB reserved but unused around 0.37 allocated. A
# ceiling is cut from the reserved peak, logged since then, never from the allocated one.
CARD_GB = float(os.environ.get("RAIDIOLOGY_GRADER_CARD_GB", "3.5"))
if DEV.type == "cuda":
    from core import card as _card_lease  # noqa: E402
    _card_lease.lease(CARD_GB, "spine grader")     # before anything starts CUDA, so a waiting grader holds none
    _card = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(
        min(0.95, CARD_GB * 2 ** 30 / torch.cuda.get_device_properties(_card).total_memory), _card)
CFG = {"epochs": 60, "batch": 32, "lr": 2e-3, "weight_decay": 1e-4, "patience": 10, "width": 32, "early_stop_frac": 0.15,
       "amp": False,   # amp: the training step in bfloat16 autocast (a trial switch; version 1 trained in float32)
       # idea 19, all off at these values (docs/efficiency): pool "avgmax" adds a max-pooled copy of the last maps to
       # the head; wear_w weighs grade 4-5 discs in the herniation loss; rank_w > 0 adds a same-grade ranking term
       # of each herniated disc over rank_k herniation-free discs of its grade
       "pool": "avg", "wear_w": 1.0, "rank_w": 0.0, "rank_k": 2}


def fingerprint_cfg(cfg):
    """cfg without the idea-19 keys that sit at their defaults, so every older model's fingerprint is unchanged."""
    c = dict(cfg)
    if c.get("pool", "avg") == "avg":
        c.pop("pool", None)
    if c.get("wear_w", 1.0) == 1.0:
        c.pop("wear_w", None)
    if c.get("rank_w", 0.0) == 0:
        c.pop("rank_w", None)
        c.pop("rank_k", None)
    return c


def build_net(cfg):
    """A Net from a cfg dict (training's CFG or a saved grader.json's); keys an old file lacks mean the old net."""
    return Net(cfg["width"], cfg.get("pool", "avg")).to(DEV)


class Net(nn.Module):
    def __init__(self, width=32, pool="avg"):
        super().__init__()
        self.pool = pool
        chans = [10, width, width * 2, width * 4, width * 8]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, padding=1), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True),
                       nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(chans[-1] * (2 if pool == "avgmax" else 1) + 1, 4 + len(ITEMS))

    def forward(self, x, has_t1):
        f = self.body(x.flatten(1, 2))
        parts = [F.adaptive_avg_pool2d(f, 1).flatten(1)]
        if self.pool == "avgmax":
            parts.append(F.adaptive_max_pool2d(f, 1).flatten(1))
        f = torch.cat([*parts, has_t1[:, None]], 1)
        return self.head(self.drop(f))   # [:, :4] cumulative Pfirrmann logits, [:, 4:] binary items


def targets(df):
    g = df["t_pfirrmann"].to_numpy(int)
    cum = np.stack([(g > k).astype(np.float32) for k in (1, 2, 3, 4)], 1)
    b = np.stack([df[f"t_{i}"].to_numpy(np.float32) for i in ITEMS], 1)
    return np.concatenate([cum, b], 1)


def augment(x, rng):
    """Small shifts, rotation and scale, intensity jitter, and left-right plane reversal."""
    n = x.shape[0]
    theta = torch.zeros(n, 2, 3, device=x.device)
    ang = torch.tensor(rng.uniform(-0.12, 0.12, n), device=x.device, dtype=torch.float32)
    sc = torch.tensor(rng.uniform(0.9, 1.1, n), device=x.device, dtype=torch.float32)
    theta[:, 0, 0] = torch.cos(ang) * sc
    theta[:, 0, 1] = -torch.sin(ang) * sc
    theta[:, 1, 0] = torch.sin(ang) * sc
    theta[:, 1, 1] = torch.cos(ang) * sc
    theta[:, :, 2] = torch.tensor(rng.uniform(-0.08, 0.08, (n, 2)), device=x.device, dtype=torch.float32)
    flat = x.flatten(1, 2)
    flat = F.grid_sample(flat, F.affine_grid(theta, flat.shape, align_corners=False), align_corners=False)
    x = flat.view_as(x)
    gain = torch.tensor(rng.uniform(0.85, 1.15, (n, 2, 1, 1, 1)), device=x.device, dtype=torch.float32)
    x = x * gain
    flip = torch.tensor(rng.random(n) < 0.5, device=x.device)
    x[flip] = x[flip].flip(2)
    return x


PROBE = {}   # the selftest reads what the last train_one did: epochs run, where it resumed, what it loaded


def _hold_out(rng, X):
    """The early-stop patients: a seeded share of the training patients. Advances rng."""
    pts = np.unique(X["patients"])
    hold = set(rng.choice(pts, max(1, int(round(CFG["early_stop_frac"] * len(pts)))), replace=False))
    va = np.isin(X["patients"], list(hold))
    return va, ~va


def _targets(y, tr):
    # A collection graded for Pfirrmann alone (EXTRA) has no items: NaN there, and the item loss
    # counts only the known ones. Without such rows every line below runs exactly as version 1's.
    known = ~torch.isnan(y[:, 4:])
    partial = bool((~known).any())
    y_items = torch.nan_to_num(y[:, 4:])
    if partial:
        k_tr = known[tr].float()
        pos = ((y_items[tr] * k_tr).sum(0) / k_tr.sum(0).clamp_min(1)).clamp(0.02, 0.98)
    else:
        pos = y[tr, 4:].mean(0).clamp(0.02, 0.98)
    pw = ((1 - pos) / pos).sqrt()                   # softened class balance for rare items
    return known, partial, y_items, pw


def _wear_weights(y, tr, known):
    if CFG["wear_w"] == 1.0:
        return None
    # worn discs (grade 4-5) weigh more in the herniation term only, rescaled so its mean over the known
    # training rows is 1; the early-stop loss stays unweighted
    wear_w = 1 + (CFG["wear_w"] - 1) * y[:, 2]
    return wear_w / wear_w[torch.tensor(tr, device=DEV) & known[:, 0]].mean()


def _rank_pools(Y, idx_tr):
    # training rows only, by Pfirrmann grade: the discs whose herniation is known to be absent
    grade_np = 1 + Y[:, :4].sum(1).astype(int)
    herni_np = np.nan_to_num(Y[:, 4], nan=-1)
    grade_t = torch.tensor(grade_np, device=DEV)
    base_rows = np.sort(idx_tr)
    neg_by_grade = {int(g): base_rows[(grade_np[base_rows] == g) & (herni_np[base_rows] == 0)]
                    for g in np.unique(grade_np)}
    return grade_np, herni_np, grade_t, neg_by_grade


def _resume_partial(ck, ckpt, net, opt, sched, rng, idx_tr):
    """Continue a fold from its partial checkpoint when one fits. Returns (best, best_state, stale, idx_tr,
    start epoch, seconds already spent)."""
    best, best_state, stale = np.inf, None, 0
    start, prior_secs = 0, 0.0
    if ck is not None and 0 < ck["epoch"] <= CFG["epochs"]:
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best, best_state, stale, idx_tr = ck["best"], ck["best_state"], ck["stale"], ck["idx_tr"]
        rng.bit_generator.state = ck["np_rng"]
        ckpt_mod.set_rng_state(ck["rng"])
        start, prior_secs = ck["epoch"], ck["seconds"]
        PROBE.update(resumed_epoch=start, opt=copy.deepcopy(opt.state_dict()), best=best, stale=stale)
        print("fold %s: resumed mid-fold at epoch %d of %d (partial checkpoint)"
              % (os.path.basename(ckpt), start, CFG["epochs"]), flush=True)
    return best, best_state, stale, idx_tr, start, prior_secs


def _batch_rows(b, rng, rank):
    """The batch's rows, plus for each herniated row the same-grade herniation-free rows (training rows only)."""
    if rank is None:
        return b
    grade_np, herni_np, _, neg_by_grade = rank
    # same-grade herniation-free rows for each herniated row of the batch (training rows only)
    extra = [rng.choice(neg_by_grade[gi], CFG["rank_k"], replace=len(neg_by_grade[gi]) < CFG["rank_k"])
             for gi in grade_np[b][herni_np[b] == 1] if len(neg_by_grade.get(gi, ())) > 0]
    return np.concatenate([b, *extra]) if extra else b


def _item_loss(out, b, d):
    if d.wear_w is None:
        if d.partial:
            return known_items_loss(out[:, 4:], d.y_items[b], d.known[b], d.pw)
        return F.binary_cross_entropy_with_logits(out[:, 4:], d.y[b, 4:], pos_weight=d.pw)
    if d.partial:
        return known_items_loss(out[:, 4:], d.y_items[b], d.known[b], d.pw, d.wear_w[b])
    return known_items_loss(out[:, 4:], d.y[b, 4:], torch.ones_like(out[:, 4:], dtype=torch.bool), d.pw, d.wear_w[b])


def _rank_loss(out_all, bc, d):
    z = out_all[:, 4]
    hv, gc = d.y[bc, 4], d.rank[2][bc]
    pair = (hv == 1)[:, None] & (hv == 0)[None, :] & (gc[:, None] == gc[None, :])
    if pair.any():
        return CFG["rank_w"] * F.softplus(-(z[:, None] - z[None, :]))[pair].mean()
    return None


def _batch_step(net, opt, b, rng, d):
    nb = len(b)
    bc = _batch_rows(b, rng, d.rank)
    with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=bool(CFG.get("amp")) and DEV.type == "cuda"):
        out_all = net(augment(d.x[bc], rng), d.h[bc]).float()
    out = out_all[:nb]
    loss = F.binary_cross_entropy_with_logits(out[:, :4], d.y[b, :4]) + _item_loss(out, b, d)
    if len(bc) > nb:
        rank_term = _rank_loss(out_all, bc, d)
        if rank_term is not None:
            loss = loss + rank_term
    PROBE["loss"] = float(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()


def _train_epoch(net, opt, idx_tr, rng, d):
    net.train()
    rng.shuffle(idx_tr)
    for s in range(0, len(idx_tr), CFG["batch"]):
        _batch_step(net, opt, idx_tr[s:s + CFG["batch"]], rng, d)


def _val_loss(net, d, va):
    net.eval()
    with torch.no_grad():
        out = net(d.x[va], d.h[va])
        item = known_items_loss(out[:, 4:], d.y_items[va], d.known[va]) if d.partial else \
            F.binary_cross_entropy_with_logits(out[:, 4:], d.y[va, 4:])
        return (F.binary_cross_entropy_with_logits(out[:, :4], d.y[va, :4]) + item).item()


def _save_partial(part, fp, ep, net, opt, sched, idx_tr, rng, best, best_state, stale, seconds):
    ckpt_mod.save_partial(part, {
        "fingerprint": fp, "epoch": ep + 1, "net": net.state_dict(), "opt": opt.state_dict(),
        "sched": sched.state_dict(), "best": best, "best_state": best_state, "stale": stale,
        "idx_tr": idx_tr, "np_rng": rng.bit_generator.state, "rng": ckpt_mod.rng_state(),
        "seconds": seconds})
    foldck.stop_after(ep + 1)


def train_one(X, hasT1, Y, seed, log=None, ckpt=None):
    """Train one model with early stopping on a held-back slice of its own training patients.

    With `ckpt` (a path without extension) the model is <ckpt>.pt, a bare state dict as ever, and the fingerprint of
    its training settings sits in <ckpt>.pt.fp: a model with a matching fingerprint is loaded, not retrained
    (RAIDIOLOGY_NO_RESUME=1 retrains). <ckpt>.partial.pt is saved after every epoch (an epoch is seconds, so there is
    no mid-epoch save) and continued from, early stopping and all; it is removed once the model is saved."""
    final = part = fp = None
    PROBE.clear()
    PROBE.update(epochs_run=0, resumed_epoch=None, cached=False)
    if ckpt:
        final, part = ckpt + ".pt", ckpt + ".partial.pt"
        fp = foldck.digest({"cfg": fingerprint_cfg(CFG), "seed": seed,
                            "data": foldck.array_digest(X["x"], hasT1, Y, X["patients"])})
        if foldck.fp_ok(final, fp):
            net = build_net(CFG)
            net.load_state_dict(torch.load(final, map_location=DEV))
            PROBE["cached"] = True
            if log:
                log(f"    seed {seed}: loaded {os.path.basename(final)} (same settings, no training)")
            return net
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    va, tr = _hold_out(rng, X)
    x = torch.tensor(X["x"], dtype=torch.float32, device=DEV)
    h = torch.tensor(hasT1, dtype=torch.float32, device=DEV)
    y = torch.tensor(Y, dtype=torch.float32, device=DEV)
    known, partial, y_items, pw = _targets(y, tr)
    net = build_net(CFG)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG["epochs"])
    idx_tr = np.nonzero(tr)[0]
    t0 = time.time()
    ck = ckpt_mod.load_partial(part, fp) if part else None
    best, best_state, stale, idx_tr, start, prior_secs = _resume_partial(ck, ckpt, net, opt, sched, rng, idx_tr)
    del ck
    d = argparse.Namespace(x=x, h=h, y=y, y_items=y_items, known=known, partial=partial, pw=pw,
                           wear_w=_wear_weights(y, tr, known),
                           rank=_rank_pools(Y, idx_tr) if CFG["rank_w"] > 0 else None)
    ep = start - 1
    for ep in range(start, CFG["epochs"]):
        PROBE["epochs_run"] += 1
        _train_epoch(net, opt, idx_tr, rng, d)
        sched.step()
        vl = _val_loss(net, d, va)
        if vl < best - 1e-4:
            best, best_state, stale = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= CFG["patience"]:
                break
        if part:
            _save_partial(part, fp, ep, net, opt, sched, idx_tr, rng, best, best_state, stale,
                          prior_secs + time.time() - t0)
    net.load_state_dict(best_state)
    if final:
        foldck.finish(final, lambda p: torch.save(net.state_dict(), p), fp)
        ckpt_mod.clear_partial(part)
    if log:
        log(f"    seed {seed}: {int(tr.sum())} train / {int(va.sum())} early-stop discs, best val loss {best:.4f} at epoch {ep + 1 - stale}")
    return net


def known_items_loss(z, y, known, pos_weight=None, row_w=None):
    """The items' mean log loss over the known (disc, item) cells only. `row_w` (one weight per row) scales the
    herniation column (item 0) alone, as the wear weighting of training asks."""
    per = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight, reduction="none")
    if row_w is not None:
        per = torch.cat([per[:, :1] * row_w[:, None], per[:, 1:]], 1)
    k = known.float()
    return (per * k).sum() / k.sum().clamp_min(1)


@torch.no_grad()
def logits(nets, x, has_t1):
    """Mean logit over the nets (averaging logits keeps temperature scaling a single division)."""
    xt = torch.tensor(x, dtype=torch.float32, device=DEV)
    ht = torch.tensor(has_t1, dtype=torch.float32, device=DEV)
    outs = []
    for n in nets:
        n.eval()
        outs.append(torch.cat([n(xt[s:s + 256], ht[s:s + 256]) for s in range(0, len(xt), 256)]).cpu().numpy())
    return np.mean(outs, 0)


def fit_temperature(z, y):
    """One temperature per output column, minimising log loss on (logits, 0/1 targets)."""
    T = np.ones(z.shape[1])
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 121))
    for j in range(z.shape[1]):
        losses = []
        for t in grid:
            p = 1 / (1 + np.exp(-z[:, j] / t))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            losses.append(-np.mean(y[:, j] * np.log(p) + (1 - y[:, j]) * np.log(1 - p)))
        T[j] = grid[int(np.argmin(losses))]
    return T


def youden(p, t):
    best, thr = -np.inf, 0.5
    for c in np.unique(p):
        y = p >= c
        if t.sum() == 0 or (1 - t).sum() == 0:
            break
        j = y[t == 1].mean() + (~y[t == 0]).mean() - 1
        if j > best + 1e-12:
            best, thr = j, float(c)
    return thr


def ece(p, t, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            e += m.mean() * abs(p[m].mean() - t[m].mean())
    return float(e)


def decode(z, T):
    """Calibrated probabilities from logits: Pfirrmann grade and per-item probabilities."""
    p = 1 / (1 + np.exp(-z / T))
    grade = 1 + (p[:, :4] > 0.5).sum(1)
    cls = np.c_[1 - p[:, 0], p[:, 0] - p[:, 1], p[:, 1] - p[:, 2], p[:, 2] - p[:, 3], p[:, 3]].clip(0, 1)
    return grade, p[:, 4:], cls / cls.sum(1, keepdims=True)


def load_split_crops(split, loc="tss"):
    d = np.load(os.path.join(WORK, split, "crops.npz"), allow_pickle=False)
    idx = pd.read_csv(os.path.join(WORK, split, "crops_index.csv"))
    sel = (idx.localisation == loc).to_numpy()
    idx = idx[sel].reset_index(drop=True)
    g = grades().rename(columns={k: "t_" + k for k in ["pfirrmann", *ITEMS]})
    g["t_modic"] = (g["t_modic"] > 0).astype(int)
    idx = idx.merge(g, on=["patient", "ivd"], how="left")
    return d["x"][sel], idx


def lsma_crops(split):
    import lsma
    return lsma.load_crops(split)


def lsma_items_crops(split):
    import lsma
    return lsma.load_crops(split, items=True)


def lsma_notes_crops(split):
    import lsma
    return lsma.load_crops(split, notes=True)


def lsma_all_crops(split):
    import lsma
    return lsma.load_crops(split, items=True, notes=True)


def lsma_all_clean_crops(split):
    import lsma
    return lsma.load_crops(split, items=True, notes=True, clean=True)


def lsma_all_text_crops(split):
    import lsma
    return lsma.load_crops(split, items=True, notes=True, text=True)


def lsma_all_slip_crops(split):
    import lsma
    return lsma.load_crops(split, items=True, notes=True, slip=True)


def lsma_all_text_slip_crops(split):
    import lsma
    return lsma.load_crops(split, items=True, notes=True, text=True, slip=True)


# Collections beyond SPIDER a trial may train on: name -> (split -> (crops, index)). The index has
# patient (distinct from SPIDER's), ivd, fold, has_t1, t_pfirrmann, and NaN for every item it lacks.
# lsma_items: lsma plus bulging and spondylolisthesis (Multi-Disorder); lsma_notes: plus herniation
# (its structured notes); lsma_all: both; lsma_all_clean: both, with herniation unknown for patients
# whose notes report one but locate none; lsma_all_text: both, herniation from the radiologists'
# free-text notes (lsma_text.py) instead; lsma_all_slip, lsma_all_text_slip: lsma_all and lsma_all_text
# with spondylolisthesis backed by the notes (lsma.slip_items; docs/efficiency/README.md, idea 22).
EXTRA = {"lsma": lsma_crops, "lsma_items": lsma_items_crops, "lsma_notes": lsma_notes_crops, "lsma_all": lsma_all_crops,
         "lsma_all_clean": lsma_all_clean_crops, "lsma_all_text": lsma_all_text_crops,
         "lsma_all_slip": lsma_all_slip_crops, "lsma_all_text_slip": lsma_all_text_slip_crops}


def train(version, seed=0, extra=(), both=False, led=None):
    """`seed` shifts every draw (0 is version 1's); `extra` names collections in EXTRA whose
    development discs join every fold's training. Out-of-fold predictions cover them too, but the
    temperatures, thresholds and calibration are fitted on SPIDER's discs alone, as version 1's.
    `both` adds SPIDER's development discs a second time, cut where SPIDER's own masks put them
    (crops.py's spider_masks), to training only: the out-of-fold rows stay the TotalSpineSeg ones."""
    out_dir = os.path.join(MODELS, f"disc-grader-v{version}")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    x, idx = load_split_crops("dev")
    idx["collection"] = "spider"
    for name in extra:
        xe, ie = EXTRA[name]("dev")
        x, idx = np.concatenate([x, xe]), pd.concat([idx, ie.assign(collection=name)], ignore_index=True)
    Y = targets(idx)
    folds = idx.fold.to_numpy(int)
    xa = None
    if both:
        xa, ia = load_split_crops("dev", "spider_masks")
        Ya, fa = targets(ia), ia.fold.to_numpy(int)

    def pool(keep):
        """The training rows of the folds `keep` accepts, plus their second crops when `both`."""
        m = keep(folds)
        X, h, y = {"x": x[m], "patients": idx.patient[m].to_numpy()}, idx.has_t1[m].to_numpy(), Y[m]
        if xa is not None:
            a = keep(fa)
            X = {"x": np.concatenate([X["x"], xa[a]]), "patients": np.concatenate([X["patients"], ia.patient[a].to_numpy()])}
            h, y = np.concatenate([h, ia.has_t1[a].to_numpy()]), np.concatenate([y, Ya[a]])
        return X, h, y

    base = SEED + 1000 * seed
    log(f"{time.strftime('%Y-%m-%d %H:%M')} training on {len(idx)} discs, {idx.patient.nunique()} patients, device {DEV}, cfg {CFG}"
        + (f", seed {seed}" if seed else "") + (f", with {', '.join(extra)}: {dict(idx.collection.value_counts())}" if extra else "")
        + (f", plus {len(xa)} SPIDER-mask crops for training" if both else ""))
    oof_single = np.zeros_like(Y)
    oof_nested = np.zeros_like(Y)
    outer = []
    for k in range(5):
        tr = folds != k
        log(f"outer fold {k}: train {int(tr.sum())} discs")
        # train_one saves fold{k}.pt itself (a bare state dict, its settings' fingerprint in fold{k}.pt.fp) and
        # loads it instead of training when the fingerprint matches; inner{k}_{j}.pt are the nested models.
        net = train_one(*pool(lambda f: f != k), base + k, log, os.path.join(out_dir, f"fold{k}"))
        outer.append(net)
        oof_single[~tr] = logits([net], x[~tr], idx.has_t1[~tr].to_numpy())
        inner = []
        for j in [f for f in range(5) if f != k]:
            inner.append(train_one(*pool(lambda f: (f != k) & (f != j)), base + 100 * (k + 1) + j, log,
                                   os.path.join(out_dir, f"inner{k}_{j}")))
        oof_nested[~tr] = logits(inner, x[~tr], idx.has_t1[~tr].to_numpy())
    np.savez(os.path.join(out_dir, "oof_logits.npz"), single=oof_single, nested=oof_nested, y=Y,
             patient=idx.patient.to_numpy(), ivd=idx.ivd.to_numpy(), fold=folds,
             collection=idx.collection.to_numpy().astype(str))
    sp = (idx.collection == "spider").to_numpy()
    oof_single, oof_nested, Y, folds = oof_single[sp], oof_nested[sp], Y[sp], folds[sp]
    calib = {}
    for name, z in (("single", oof_single), ("nested", oof_nested)):
        cross_p = np.zeros_like(z)
        for k in range(5):
            T = fit_temperature(z[folds != k], Y[folds != k])
            cross_p[folds == k] = 1 / (1 + np.exp(-z[folds == k] / T))
        calib[name] = {"ece_cross_fitted": {c: round(ece(cross_p[:, j], Y[:, j]), 4) for j, c in enumerate(
            [f"pfirrmann>{k}" for k in (1, 2, 3, 4)] + ITEMS)}}
    T_ship = fit_temperature(oof_nested, Y)
    _, p_ship, _ = decode(oof_nested, T_ship)
    thr_ship = {b: youden(p_ship[:, j], Y[:, 4 + j]) for j, b in enumerate(ITEMS)}
    meta = {"version": version, "cfg": CFG, "trained": time.strftime("%Y-%m-%d"), "n_discs": int(len(idx)),
            **({"seed": seed} if seed else {}), **({"extra": list(extra)} if extra else {}),
            **({"both_localisations": True} if both else {}),
            "n_patients": int(idx.patient.nunique()), "temperature": T_ship.tolist(), "thresholds": thr_ship,
            "calibration": calib, "inputs": "crops.py: T2+T1, 5 planes x 80 x 112 at 0.6 mm, 3.3 mm apart",
            "localisation": "TotalSpineSeg r20260730 disc labels (TSS weights: permission pending)",
            "weights": {f"fold{k}.pt": hashlib.sha256(open(os.path.join(out_dir, f"fold{k}.pt"), "rb").read()).hexdigest()
                        for k in range(5)}}
    if DEV.type == "cuda":
        meta["peak_card_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        meta["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        log("peak card memory %.2f GB allocated, %.2f reserved (ceiling %.1f)"
            % (meta["peak_card_gb"], meta["peak_reserved_gb"], CARD_GB))
    json.dump(meta, open(os.path.join(out_dir, "grader.json"), "w"), indent=1)
    if led is not None:
        led.number("discs", meta["n_discs"], "count").number("patients", meta["n_patients"], "count")
        led.number("peak_card_gb", meta.get("peak_card_gb"), "GB")
        led.number("peak_reserved_memory_gb", meta.get("peak_reserved_gb"), "GB")
        heads = [f"pfirrmann>{k}" for k in (1, 2, 3, 4)] + ITEMS
        for j, c in enumerate(heads):
            led.number(f"temperature.{c}", meta["temperature"][j])
        for b in ITEMS:
            led.number(f"threshold.{b}", meta["thresholds"][b])
        for name in calib:
            for c, v in calib[name]["ece_cross_fitted"].items():
                led.number(f"ece_cross_fitted.{name}.{c}", v)
    log(json.dumps({k: meta[k] for k in ("temperature", "thresholds", "calibration")}, indent=1))


def load_grader(version):
    d = os.path.join(MODELS, f"disc-grader-v{version}")
    meta = json.load(open(os.path.join(d, "grader.json")))
    nets = []
    for k in range(5):
        n = build_net(meta["cfg"])
        n.load_state_dict(torch.load(os.path.join(d, f"fold{k}.pt"), map_location=DEV))
        nets.append(n)
    return nets, meta


def predict_dev(version):
    """OOF development predictions for the scorecard, cross-fitted temperature and thresholds."""
    d = os.path.join(MODELS, f"disc-grader-v{version}")
    o = np.load(os.path.join(d, "oof_logits.npz"))
    meta = json.load(open(os.path.join(d, "grader.json")))
    rows = []
    for loc in ("tss", "spider_masks"):
        x, idx = load_split_crops("dev", loc)
        folds = idx.fold.to_numpy(int)
        nets = None
        for k in range(5):
            tr_mask = o["fold"] != k
            T = fit_temperature(o["single"][tr_mask], o["y"][tr_mask])
            _, p_tr, _ = decode(o["single"][tr_mask], T)
            thr = [youden(p_tr[:, j], o["y"][tr_mask][:, 4 + j]) for j in range(len(ITEMS))]
            sel = folds == k
            if loc == "tss":
                key = dict(zip(zip(o["patient"], o["ivd"]), range(len(o["patient"]))))
                z = np.stack([o["single"][key[(p, i)]] for p, i in zip(idx.patient[sel], idx.ivd[sel])])
            else:
                if nets is None:
                    nets = [build_net(meta["cfg"]) for _ in range(5)]
                    for f, n in enumerate(nets):
                        n.load_state_dict(torch.load(os.path.join(d, f"fold{f}.pt"), map_location=DEV))
                z = logits([nets[k]], x[sel], idx.has_t1[sel].to_numpy())
            grade, p, cls = decode(z, T)
            for r, gr, pp, cc in zip(idx[sel].itertuples(), grade, p, cls):
                row = {"patient": r.patient, "ivd": r.ivd, "localisation": loc, "pfirrmann_pred": int(gr),
                       **{f"c{c + 1}": float(cc[c]) for c in range(5)}}
                for j, b in enumerate(ITEMS):
                    row[f"{b}_score"] = float(pp[j])
                    row[f"{b}_pred"] = int(pp[j] >= thr[j])
                rows.append(row)
    write_preds("dev", pd.DataFrame(rows))


def predict_test(version):
    """Test predictions from the shipped ensemble (final evaluation only)."""
    from spider import guard_test
    guard_test("test")
    nets, meta = load_grader(version)
    T = np.array(meta["temperature"])
    rows = []
    for loc in ("tss", "spider_masks"):
        x, idx = load_split_crops("test", loc)
        grade, p, cls = decode(logits(nets, x, idx.has_t1.to_numpy()), T)
        for r, gr, pp, cc in zip(idx.itertuples(), grade, p, cls):
            row = {"patient": r.patient, "ivd": r.ivd, "localisation": loc, "pfirrmann_pred": int(gr),
                   **{f"c{c + 1}": float(cc[c]) for c in range(5)}}
            for j, b in enumerate(ITEMS):
                row[f"{b}_score"] = float(pp[j])
                row[f"{b}_pred"] = int(pp[j] >= meta["thresholds"][b])
            rows.append(row)
    write_preds("test", pd.DataFrame(rows))


def write_preds(split, P, method="grader"):
    """Join predictions onto every graded row of the split (misses stay empty) in the contract's
    columns -> pred_<method>_<split>.csv; class probabilities c1..c5 and item scores, when present,
    also go to <method>_probs_<split>.csv for the combination rule."""
    m = pd.read_csv(os.path.join(WORK, split, "matches.csv"), keep_default_na=False)
    ccols = [c for c in ("c1", "c2", "c3", "c4", "c5") if c in P]
    if ccols:
        keep = ["patient", "ivd", "localisation", *ccols, *[f"{b}_score" for b in ITEMS]]
        P[keep].to_csv(os.path.join(os.path.dirname(WORK), f"{method}_probs_{split}.csv"), index=False)
        P = P.drop(columns=ccols)
    out = []
    for loc in ("tss", "spider_masks"):
        mm = m.copy()
        mm["localisation"] = loc
        pp = P[P.localisation == loc].drop(columns="localisation")
        mm = mm.merge(pp, on=["patient", "ivd"], how="left")
        if loc == "tss":
            mm.loc[mm.matched != 1, [c for c in pp.columns if c not in ("patient", "ivd")]] = np.nan
        mm.loc[mm.in_scope != 1, [c for c in pp.columns if c not in ("patient", "ivd")]] = np.nan
        out.append(mm)
    D = pd.concat(out, ignore_index=True)
    D["method"], D["split"] = method, split
    cols = ["method", "localisation", "split", "fold", "patient", "ivd", "vendor", "nominal_level", "tss_level", "dice",
            "matched", "in_scope", "drop_reason", "pfirrmann_pred"]
    for b in ITEMS:
        cols += [f"{b}_pred", f"{b}_score"]
    D[cols].to_csv(os.path.join(os.path.dirname(WORK), f"pred_{method}_{split}.csv"), index=False)
    print(f"wrote pred_{method}_{split}", len(D))


def patient(version, t2_p, lab_p, t1_p=None):
    """Grades for a patient disc's lumbar discs (T12-L1..L5-S1) from the shipped ensemble."""
    from crops import LEVEL_ID, Series, crop
    import nibabel as nib
    nets, meta = load_grader(version)
    t2 = Series(t2_p)
    t1 = Series(t1_p) if t1_p else None
    lab = np.asarray(nib.load(lab_p).dataobj).astype(np.int32)
    res = {}
    for level in ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]:
        x, has_t1 = crop(t2, t1, lab, LEVEL_ID[level])
        if x is None:
            continue
        grade, p, cls = decode(logits(nets, x[None].astype(np.float32), np.array([float(has_t1)])), np.array(meta["temperature"]))
        res[level] = {"pfirrmann": int(grade[0]), "pfirrmann_probs": [round(float(c), 3) for c in cls[0]],
                      "items": {b: round(float(p[0, j]), 3) for j, b in enumerate(ITEMS)},
                      "flags": {b: int(p[0, j] >= meta["thresholds"][b]) for j, b in enumerate(ITEMS)}}
    return res


def selftest():
    """CPU, synthetic tiny data, seconds: train_one stopped after epoch 3 of 6 and run again resumes at 3 with the
    saved optimiser and best score, runs only the last 3 epochs, ends on the weights an uninterrupted run reaches,
    leaves no partial; a finished model with its fingerprint is loaded; a corrupt partial is renamed .bad and the
    model trains fresh. Usage: python grader.py selftest (with CUDA_VISIBLE_DEVICES empty)."""
    import shutil
    import tempfile
    os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
    os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
    saved_cfg = dict(CFG)
    CFG.update(epochs=6, patience=100, width=4, batch=16, early_stop_frac=0.25, amp=False)
    g = np.random.default_rng(0)
    n = 64
    X = {"x": g.normal(size=(n, 2, 5, 16, 16)).astype(np.float32), "patients": np.repeat(np.arange(16), 4)}
    hasT1 = g.integers(0, 2, n).astype(np.float32)
    Y = g.integers(0, 2, (n, 4 + len(ITEMS))).astype(np.float32)
    root = tempfile.mkdtemp(prefix="grader-selftest-")

    def run(name, stop=None):
        if stop:
            os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = str(stop)
        else:
            os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
        return train_one(X, hasT1, Y, 7, None, os.path.join(root, name))

    try:
        ref = run("ref")
        assert PROBE["epochs_run"] == 6 and not os.path.exists(os.path.join(root, "ref.partial.pt"))
        # stopped part-way, then resumed
        try:
            run("a", stop=3)
            raise AssertionError("the stop did not fire")
        except foldck.SelftestStop:
            pass
        part = os.path.join(root, "a.partial.pt")
        saved = torch.load(part, map_location="cpu", weights_only=False)
        assert saved["epoch"] == 3 and not os.path.exists(os.path.join(root, "a.pt"))
        got = run("a")
        assert PROBE["resumed_epoch"] == 3, "resumed at the saved epoch"
        assert foldck.same(PROBE["opt"], saved["opt"]), "the optimiser state is the saved one"
        assert PROBE["best"] == saved["best"] and PROBE["stale"] == saved["stale"], "best score and stale counter"
        assert PROBE["epochs_run"] == 3, "only the rest ran"
        assert not os.path.exists(part), "the partial is gone after the model"
        assert foldck.same(got.state_dict(), ref.state_dict()), "ends where an uninterrupted run ends"
        # a finished model with its fingerprint is loaded, not trained
        run("a")
        assert PROBE["cached"] and PROBE["epochs_run"] == 0
        # a different seed is another fingerprint: it trains
        train_one(X, hasT1, Y, 8, None, os.path.join(root, "a"))
        assert not PROBE["cached"] and PROBE["epochs_run"] == 6
        # a corrupt partial is renamed .bad and training starts fresh
        try:
            run("b", stop=2)
        except foldck.SelftestStop:
            pass
        with open(os.path.join(root, "b.partial.pt"), "wb") as f:
            f.write(b"not a checkpoint")
        got = run("b")
        assert os.path.exists(os.path.join(root, "b.partial.pt.bad")), "corrupt partial renamed .bad"
        assert PROBE["resumed_epoch"] is None and PROBE["epochs_run"] == 6, "fresh start"
        assert foldck.same(got.state_dict(), ref.state_dict())
        # the three herniation switches on: finite loss, and the saved model loads back through the load path
        CFG.update(pool="avgmax", wear_w=3.0, rank_w=0.5, rank_k=2)
        net = run("sw")
        assert PROBE["epochs_run"] == 6 and np.isfinite(PROBE["loss"]), "switches on: a finite training loss"
        assert net.pool == "avgmax" and net.head.in_features == 4 * 8 * 2 + 1
        again = build_net({"width": CFG["width"], "pool": CFG["pool"]})
        again.load_state_dict(torch.load(os.path.join(root, "sw.pt"), map_location=DEV))
        assert foldck.same(again.state_dict(), net.state_dict()), "the avgmax model loads back"
        run("sw")
        assert PROBE["cached"] and PROBE["epochs_run"] == 0
        print("grader selftest OK")
    finally:
        os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP", None)
        CFG.clear()
        CFG.update(saved_cfg)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict", "patient", "selftest"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="shifts every draw; 0 is version 1's")
    ap.add_argument("--extra", nargs="*", default=[], choices=sorted(EXTRA),
                    help="collections whose development discs join every fold's training")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override one CFG entry for a trial, e.g. width=48")
    ap.add_argument("--both-localisations", action="store_true",
                    help="also train on SPIDER's discs cut where SPIDER's own masks put them")
    a = ap.parse_args()
    CFG.update(sets.parse(a.set, known=CFG))
    if a.mode == "selftest":
        selftest()
    elif a.mode == "train":
        with ledger.run("grader", f"grader v{a.version} seed {a.seed}") as led:
            led.settings_from({"version": a.version, "seed": a.seed, "extra": list(a.extra), "folds": 5, "split": "dev",
                               "both_localisations": a.both_localisations, "set": list(a.set),
                               **{f"cfg.{k}": v for k, v in CFG.items()}})
            train(a.version, a.seed, tuple(a.extra), a.both_localisations, led)
    elif a.mode == "predict":
        with ledger.run("grader", f"grader v{a.version} predict {a.args[0]}") as led:
            led.settings_from({"version": a.version, "split": a.args[0]})
            (predict_dev if a.args[0] == "dev" else predict_test)(a.version)
    else:
        print(json.dumps(patient(a.version, *a.args), indent=1))
