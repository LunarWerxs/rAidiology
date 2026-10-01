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
Usage: python grader.py train [--version 1] | predict <dev|test> | patient <t2.nii.gz> <labels.nii.gz> [t1.nii.gz]"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from prepare import WORK  # noqa: E402
from spider import ITEMS, SEED, grades  # noqa: E402

MODELS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Models"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = {"epochs": 60, "batch": 32, "lr": 2e-3, "weight_decay": 1e-4, "patience": 10, "width": 32, "early_stop_frac": 0.15}


class Net(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        chans = [10, width, width * 2, width * 4, width * 8]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, padding=1), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True),
                       nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(chans[-1] + 1, 4 + len(ITEMS))

    def forward(self, x, has_t1):
        f = self.body(x.flatten(1, 2))
        f = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1), has_t1[:, None]], 1)
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


def train_one(X, hasT1, Y, seed, log=None):
    """Train one model with early stopping on a held-back slice of its own training patients."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pts = np.unique(X["patients"])
    hold = set(rng.choice(pts, max(1, int(round(CFG["early_stop_frac"] * len(pts)))), replace=False))
    va = np.isin(X["patients"], list(hold))
    tr = ~va
    x = torch.tensor(X["x"], dtype=torch.float32, device=DEV)
    h = torch.tensor(hasT1, dtype=torch.float32, device=DEV)
    y = torch.tensor(Y, dtype=torch.float32, device=DEV)
    pos = y[tr, 4:].mean(0).clamp(0.02, 0.98)
    pw = ((1 - pos) / pos).sqrt()                   # softened class balance for rare items
    net = Net(CFG["width"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG["epochs"])
    best, best_state, stale = np.inf, None, 0
    idx_tr = np.nonzero(tr)[0]
    for ep in range(CFG["epochs"]):
        net.train()
        rng.shuffle(idx_tr)
        for s in range(0, len(idx_tr), CFG["batch"]):
            b = idx_tr[s:s + CFG["batch"]]
            out = net(augment(x[b], rng), h[b])
            loss = F.binary_cross_entropy_with_logits(out[:, :4], y[b, :4]) + \
                F.binary_cross_entropy_with_logits(out[:, 4:], y[b, 4:], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            out = net(x[va], h[va])
            vl = (F.binary_cross_entropy_with_logits(out[:, :4], y[va, :4]) +
                  F.binary_cross_entropy_with_logits(out[:, 4:], y[va, 4:])).item()
        if vl < best - 1e-4:
            best, best_state, stale = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= CFG["patience"]:
                break
    net.load_state_dict(best_state)
    if log:
        log(f"    seed {seed}: {int(tr.sum())} train / {int(va.sum())} early-stop discs, best val loss {best:.4f} at epoch {ep + 1 - stale}")
    return net


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


def train(version):
    out_dir = os.path.join(MODELS, f"disc-grader-v{version}")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    x, idx = load_split_crops("dev")
    Y = targets(idx)
    folds = idx.fold.to_numpy(int)
    log(f"{time.strftime('%Y-%m-%d %H:%M')} training on {len(idx)} discs, {idx.patient.nunique()} patients, device {DEV}, cfg {CFG}")
    oof_single = np.zeros_like(Y)
    oof_nested = np.zeros_like(Y)
    outer = []
    for k in range(5):
        tr = folds != k
        log(f"outer fold {k}: train {int(tr.sum())} discs")
        net = train_one({"x": x[tr], "patients": idx.patient[tr].to_numpy()}, idx.has_t1[tr].to_numpy(), Y[tr], SEED + k, log)
        torch.save(net.state_dict(), os.path.join(out_dir, f"fold{k}.pt"))
        outer.append(net)
        oof_single[~tr] = logits([net], x[~tr], idx.has_t1[~tr].to_numpy())
        inner = []
        for j in [f for f in range(5) if f != k]:
            itr = (folds != k) & (folds != j)
            inner.append(train_one({"x": x[itr], "patients": idx.patient[itr].to_numpy()}, idx.has_t1[itr].to_numpy(),
                                   Y[itr], SEED + 100 * (k + 1) + j, log))
        oof_nested[~tr] = logits(inner, x[~tr], idx.has_t1[~tr].to_numpy())
    np.savez(os.path.join(out_dir, "oof_logits.npz"), single=oof_single, nested=oof_nested, y=Y,
             patient=idx.patient.to_numpy(), ivd=idx.ivd.to_numpy(), fold=folds)
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
            "n_patients": int(idx.patient.nunique()), "temperature": T_ship.tolist(), "thresholds": thr_ship,
            "calibration": calib, "inputs": "crops.py: T2+T1, 5 planes x 80 x 112 at 0.6 mm, 3.3 mm apart",
            "localisation": "TotalSpineSeg r20260730 disc labels (TSS weights: permission pending)",
            "weights": {f"fold{k}.pt": hashlib.sha256(open(os.path.join(out_dir, f"fold{k}.pt"), "rb").read()).hexdigest()
                        for k in range(5)}}
    json.dump(meta, open(os.path.join(out_dir, "grader.json"), "w"), indent=1)
    log(json.dumps({k: meta[k] for k in ("temperature", "thresholds", "calibration")}, indent=1))


def load_grader(version):
    d = os.path.join(MODELS, f"disc-grader-v{version}")
    meta = json.load(open(os.path.join(d, "grader.json")))
    nets = []
    for k in range(5):
        n = Net(meta["cfg"]["width"]).to(DEV)
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
                    nets = [Net(meta["cfg"]["width"]).to(DEV) for _ in range(5)]
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict", "patient"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--version", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "train":
        train(a.version)
    elif a.mode == "predict":
        (predict_dev if a.args[0] == "dev" else predict_test)(a.version)
    else:
        print(json.dumps(patient(a.version, *a.args), indent=1))
