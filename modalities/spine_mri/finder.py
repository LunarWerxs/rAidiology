"""rAidiology's own spine MRI disc finder: where each lumbar disc is on a sagittal T2, with no
TotalSpineSeg weights anywhere in its chain.

Why: the disc grader's crops are centred on disc labels. TotalSpineSeg's released weights (tss.py) learned
sacrum labels from SynthRAD2023 (CC BY-NC 4.0) and other unconfirmed sources, so anything that reads its
labels stays out of the commercial licence (LICENSING.md). This finder replaces it for the grader.

Data and weights: SPIDER's own T2 masks (CC BY 4.0), development patients only (split.json); the net
starts from random numbers (no ImageNet or other pretrained weights). The test split is never read.

Model: a 2D U-Net on sagittal slices resampled to 1 mm in-plane, the slice and its two neighbours as
three channels, four classes (background, vertebra, disc, spinal canal; SPIDER labels 1-99, 201+, 100).
A fixed schedule (no early stopping, so no held-back patients are needed). Discs are the 3D connected
components of the disc class; the lowest is L5-S1 and the rest are counted upward, the way SPIDER
numbers its own discs, and written with TotalSpineSeg's disc ids (prepare.disc_id) so a finder label
file drops in wherever a TotalSpineSeg one went (crops.crop, grader.patient).

Folds: model k trains on the development folds other than k (split.json's, the grader's folds), so every
development patient's discs are found by a model that never saw that patient: those are the crops the
grader trains and is scored on (crops.py finder_main). Model `all` trains on every development patient
and is the one that reads new patients.

Usage: python finder.py train <0|1|2|3|4|all> [...]   -> <MODELS>/spine-disc-finder-v1/<fold>.pt
       python finder.py predict dev [--folds 0,..]    -> <work>/eval/spider/dev/finder/<p>.nii.gz (out of fold)
       python finder.py score dev [--folds 0,..]      -> per-disc Dice, found rate, level agreement (ledger)
       python finder.py cache dev                     -> build finder_cache once, before folds train side by side
       (--folds: only those folds' patients, each by its own model: the one-fold screen)
       python finder.py segment <t2.nii.gz> <out.nii.gz>   (model `all`, any orientation)"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("KMP_BLOCKTIME", "1")

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repository root
from core import ledger, paths  # noqa: E402
paths.use("spine_mri")
from crops import Series  # noqa: E402
from prepare import WORK, disc_id  # noqa: E402

VERSION = 1
OUT = os.path.join(paths.MODELS, f"spine-disc-finder-v{VERSION}")
MM = 1.0                     # in-plane spacing the net sees
CROP = 256                   # training crop, pixels at MM
CLASSES = ("background", "vertebra", "disc", "canal")
CFG = {"iters": int(os.environ.get("FINDER_ITERS", 4000)), "batch": 16, "lr": 1e-3, "weight_decay": 1e-4, "width": 32, "seed": 20261109}
MIN_DISC_MM3 = 300.0         # a disc component smaller than this is noise
CARD_GB = 3.0


def classes(mask):
    """SPIDER mask -> 0 background, 1 vertebra, 2 disc, 3 canal."""
    c = np.zeros(mask.shape, np.uint8)
    c[(mask >= 1) & (mask < 100)] = 1
    c[mask > 200] = 2
    c[mask == 100] = 3
    return c


def to_pir(img):
    """(array, affine, back) of a NIfTI image turned to PIR; back(labels) turns a PIR array to the input's orientation."""
    o = nib.orientations
    src = o.io_orientation(img.affine)
    tr = o.ornt_transform(src, o.axcodes2ornt(("P", "I", "R")))
    a = o.apply_orientation(np.asarray(img.dataobj), tr)
    aff = img.affine @ o.inv_ornt_aff(tr, img.shape)
    inv = o.ornt_transform(o.axcodes2ornt(("P", "I", "R")), src)
    return a, aff, lambda lab: o.apply_orientation(lab, inv)


def scaled(series):
    """The PIR volume scaled the way crops.Series scales it, clipped to the crops' range."""
    return np.clip((series.a - series.lo) / max(series.hi - series.lo, 1e-6), -0.5, 2.0).astype(np.float32)


def zooms(aff):
    return np.linalg.norm(aff[:3, :3], axis=0)


def resample_slices(v, z, order):
    """(AP, SI, LR) -> (LR, AP', SI') at MM in-plane."""
    f = (z[0] / MM, z[1] / MM)
    return np.stack([ndimage.zoom(v[:, :, k], f, order=order) for k in range(v.shape[2])])


def three(v, k):
    """Slice k and its two neighbours (edges repeat), (3, AP', SI')."""
    n = len(v)
    return np.stack([v[max(k - 1, 0)], v[k], v[min(k + 1, n - 1)]])


def build_net(width):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.01, True))

    class UNet(nn.Module):
        def __init__(self, w):
            super().__init__()
            ch = [w, 2 * w, 4 * w, 8 * w, 10 * w]
            self.down = nn.ModuleList([block(3, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(4)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(4)])
            self.dec = nn.ModuleList([block(2 * ch[i], ch[i]) for i in range(4)])
            self.head = nn.Conv2d(ch[0], len(CLASSES), 1)

        def forward(self, x):
            import torch
            skips = []
            for i, d in enumerate(self.down):
                x = d(x if i == 0 else nn.functional.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            for i in reversed(range(4)):
                x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
            return self.head(x)

    return UNet(width)


def split_patients():
    sp = json.load(open(os.path.join(HERE, "eval", "split.json")))
    return {int(k): s["fold"] for k, s in sp["patients"].items() if s["split"] == "dev" and s["t2"]}


def load_case(p, split="dev"):
    """(slices, classes) of a development patient at MM, cached in <work>/<split>/finder_cache (training input only)."""
    base = os.path.join(WORK, split)
    cache = os.path.join(base, "finder_cache", f"{p}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["img"], d["lab"]
    s = Series(os.path.join(base, "t2", f"{p}.nii.gz"))
    z = zooms(s.aff)
    img = resample_slices(scaled(s), z, 1).astype(np.float16)
    m = np.asarray(nib.load(os.path.join(base, "mask", f"{p}.nii.gz")).dataobj).astype(np.int32)
    lab = resample_slices(classes(m), z, 0)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, img=img, lab=lab)
    return img, lab


def train(fold):
    """Model `fold` (0-4: trains on the other development folds; `all`: on every development patient)."""
    import torch
    import torch.nn.functional as F
    from core import card
    meta_p = os.path.join(OUT, "finder.json")
    if os.path.exists(os.path.join(OUT, f"{fold}.pt")) and os.path.exists(meta_p) and json.load(open(meta_p)).get("cfg") == CFG             and str(fold) in json.load(open(meta_p)).get("models", {}):
        print(f"fold {fold}: already trained with these settings", flush=True)
        return 0.0
    card.lease(CARD_GB, f"spine disc finder {fold}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    card.require_card(dev, torch)
    folds = split_patients()
    pts = sorted(p for p, f in folds.items() if fold == "all" or f != int(fold))
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(CFG["seed"] + (99 if fold == "all" else int(fold)))
    torch.manual_seed(CFG["seed"] + (99 if fold == "all" else int(fold)))
    t0 = time.time()
    cases = [load_case(p) for p in pts]
    slices = [(i, k) for i, (img, _) in enumerate(cases) for k in range(len(img))]
    print(f"fold {fold}: {len(pts)} patients, {len(slices)} slices, loaded in {time.time() - t0:.0f}s", flush=True)
    net = build_net(CFG["width"]).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=CFG["iters"], pct_start=0.05)
    part = os.path.join(OUT, f"{fold}.part.pt")      # every 1,000 steps, so a killed run resumes (2026-11-10: three were)
    start = 0
    if os.path.exists(part):
        ck = torch.load(part, map_location=dev, weights_only=False)
        if ck["cfg"] == CFG:
            net.load_state_dict(ck["net"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            rng.bit_generator.state = ck["rng"]
            torch.set_rng_state(ck["torch_rng"])
            start = ck["it"]
            print(f"fold {fold}: resumed at step {start}", flush=True)
    cw = torch.tensor([0.5, 1.0, 2.0, 1.0], device=dev)

    def sample():
        i, k = slices[rng.integers(len(slices))]
        img, lab = cases[i]
        x, y = three(img, k).astype(np.float32), lab[k]
        if rng.random() < 0.5:                                      # left-right: the neighbours swap
            x = x[::-1].copy()
        ph, pw = max(0, CROP - y.shape[0]), max(0, CROP - y.shape[1])
        if ph or pw:
            x = np.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), constant_values=-0.5)
            y = np.pad(y, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        fg = np.argwhere(y == 2)                                    # most crops hold a disc
        if len(fg) and rng.random() < 0.8:
            c = fg[rng.integers(len(fg))]
            a0 = int(np.clip(c[0] - rng.integers(CROP), 0, y.shape[0] - CROP))
            a1 = int(np.clip(c[1] - rng.integers(CROP), 0, y.shape[1] - CROP))
        else:
            a0, a1 = rng.integers(y.shape[0] - CROP + 1), rng.integers(y.shape[1] - CROP + 1)
        x, y = x[:, a0:a0 + CROP, a1:a1 + CROP], y[a0:a0 + CROP, a1:a1 + CROP]
        x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)      # contrast and brightness
        if rng.random() < 0.3:
            x = np.sign(x) * np.abs(x) ** rng.uniform(0.7, 1.4)    # gamma
        return x, y

    net.train()
    log = open(os.path.join(OUT, f"train_{fold}.log"), "a")
    for it in range(start, CFG["iters"]):
        if it and it % 1000 == 0 and it != start:
            torch.save({"cfg": CFG, "it": it, "net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, part)
        b = [sample() for _ in range(CFG["batch"])]
        x = torch.from_numpy(np.stack([s[0] for s in b]).astype(np.float32)).to(dev)
        y = torch.from_numpy(np.stack([s[1] for s in b]).astype(np.int64)).to(dev)
        sc = torch.from_numpy(rng.uniform(0.85, 1.15, len(b)).astype(np.float32)).to(dev)   # size, on the card
        theta = torch.zeros(len(b), 2, 3, device=dev)
        theta[:, 0, 0] = theta[:, 1, 1] = 1 / sc
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z = net(x)
        z = z.float()
        ce = F.cross_entropy(z, y, weight=cw)
        p = z.softmax(1)
        oh = F.one_hot(y, len(CLASSES)).permute(0, 3, 1, 2).float()
        inter = (p * oh).sum((0, 2, 3))[1:]
        dice = 1 - ((2 * inter + 1) / (p.sum((0, 2, 3))[1:] + oh.sum((0, 2, 3))[1:] + 1)).mean()
        loss = ce + dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == CFG["iters"] - 1:
            s = f"{time.strftime('%H:%M:%S')} fold {fold} iter {it} loss {loss.item():.4f} ce {ce.item():.4f} dice {dice.item():.4f}"
            print(s, flush=True)
            log.write(s + "\n")
            log.flush()
    path = os.path.join(OUT, f"{fold}.pt")
    torch.save(net.state_dict(), path)
    if os.path.exists(part):
        os.remove(part)
    meta_p = os.path.join(OUT, "finder.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.update({"version": VERSION, "cfg": CFG, "mm": MM, "classes": list(CLASSES), "min_disc_mm3": MIN_DISC_MM3,
                 "data": "SPIDER development patients' T2 masks (CC BY 4.0), split.json folds; no pretrained weights",
                 "init": "random (torch default), no pretrained weights"})
    meta.setdefault("models", {})[str(fold)] = {"patients": len(pts), "slices": len(slices), "seconds": round(time.time() - t0),
                                                "trained": time.strftime("%Y-%m-%d %H:%M"),
                                                "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(meta, open(meta_p, "w"), indent=1)
    print(f"fold {fold} saved {path} in {time.time() - t0:.0f}s", flush=True)
    return time.time() - t0


_NETS = {}


def load(fold):
    import torch
    if fold not in _NETS:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = build_net(CFG["width"]).to(dev)
        net.load_state_dict(torch.load(os.path.join(OUT, f"{fold}.pt"), map_location=dev))
        _NETS[fold] = net.eval()
    return _NETS[fold]


def probabilities(fold, vol, aff):
    """Class probabilities (4, AP, SI, LR) on the volume's own grid, from a PIR float volume scaled like scaled()."""
    import torch
    net = load(fold)
    dev = next(net.parameters()).device
    z = zooms(aff)
    r = resample_slices(vol, z, 1)
    h, w = r.shape[1:]
    H, W = -(-h // 16) * 16, -(-w // 16) * 16
    out = np.zeros((len(CLASSES), *vol.shape), np.float32)
    with torch.no_grad():
        for k in range(len(r)):
            x = np.full((3, H, W), -0.5, np.float32)
            x[:, :h, :w] = three(r, k)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                p = net(torch.from_numpy(x)[None].to(dev)).float().softmax(1)[:, :, :h, :w]
            p = torch.nn.functional.interpolate(p, size=vol.shape[:2], mode="bilinear", align_corners=False)
            out[:, :, :, k] = p[0].cpu().numpy()
    return out


def discs(prob, aff):
    """Disc label volume with TotalSpineSeg ids: 3D components of the disc class, counted upward from the lowest."""
    seg = prob.argmax(0) == 2
    comp, n = ndimage.label(seg, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros(seg.shape, np.uint16)
    vox_mm3 = float(np.prod(zooms(aff)))
    sizes = ndimage.sum(seg, comp, range(1, n + 1)) * vox_mm3
    keep = [i + 1 for i in range(n) if sizes[i] >= MIN_DISC_MM3]
    if not keep:
        return np.zeros(seg.shape, np.uint16)
    si = ndimage.center_of_mass(seg, comp, keep)                     # axis 1 runs inferior
    order = [keep[j] for j in np.argsort([-c[1] for c in si])]      # lowest first
    out = np.zeros(seg.shape, np.uint16)
    for i, c in enumerate(order, start=1):
        d = disc_id(i)
        if d is None:
            break
        out[comp == c] = d
    return out


def segment(t2_path, out_path, fold="all"):
    """Disc labels (TotalSpineSeg ids) for a sagittal T2 in any orientation, written on its own grid."""
    img = nib.load(t2_path)
    a, aff, back = to_pir(img)
    s = Series(a=a, aff=aff)
    lab = back(discs(probabilities(fold, scaled(s), aff), aff))
    nib.save(nib.Nifti1Image(lab.astype(np.uint16), img.affine), out_path)
    return out_path


def predict_dev(folds=None):
    """Out-of-fold disc labels for every development patient (or those of `folds`): patient p is read by the model of
    its own fold."""
    from core import card
    card.lease(CARD_GB, "spine disc finder predict")
    out = os.path.join(WORK, "dev", "finder")
    os.makedirs(out, exist_ok=True)
    for p, f in sorted(split_patients().items()):
        if folds is not None and f not in folds:
            continue
        t2p = os.path.join(WORK, "dev", "t2", f"{p}.nii.gz")
        s = Series(t2p)
        lab = discs(probabilities(f, scaled(s), s.aff), s.aff)
        nib.save(nib.Nifti1Image(lab, nib.load(t2p).affine), os.path.join(out, f"{p}.nii.gz"))
    return out


def score_dev(led=None, folds=None):
    """The finder's own out-of-fold score on SPIDER's development discs in the grader's scope (matches.csv in_scope):
    found (best finder disc Dice >= match.MIN_DICE, as TotalSpineSeg is matched), Dice, centre distance and
    level agreement (the found disc's id is the one SPIDER's bottom-up count gives), with TotalSpineSeg's Dice
    on the same discs beside it (not under RAIDIOLOGY_NO_TSS=1, where none was read). `folds`: those folds' discs only."""
    import pandas as pd
    from match import MIN_DICE, match_patient, no_tss
    bare = no_tss()
    base = os.path.join(WORK, "dev")
    m = pd.read_csv(os.path.join(base, "matches.csv"), keep_default_na=False)
    m = m[m.in_scope == 1]
    if folds is not None:
        m = m[m.fold.astype(int).isin(folds)]
    rows = []
    for p in sorted(m.patient.unique()):
        mask = np.asarray(nib.load(os.path.join(base, "mask", f"{p}.nii.gz")).dataobj).astype(np.int32)
        img = nib.load(os.path.join(base, "finder", f"{p}.nii.gz"))
        lab = np.asarray(img.dataobj).astype(np.int32)
        z = zooms(img.affine)
        mt = match_patient(mask, lab)
        for _, r in m[m.patient == p].iterrows():
            best, d = mt.get(int(r.ivd), (None, 0.0))
            found = best is not None and d >= MIN_DICE
            dist = np.nan
            if found:
                a = np.array(ndimage.center_of_mass(mask == 200 + int(r.ivd)))
                b = np.array(ndimage.center_of_mass(lab == best))
                dist = float(np.linalg.norm((a - b) * z))
            rows.append({"patient": p, "ivd": int(r.ivd), "fold": r.fold, "found": int(found), "dice": d,
                         "centre_mm": dist, "level_ok": int(found and best == disc_id(int(r.ivd))),
                         "tss_dice": np.nan if bare else (float(r.dice) if r.dice != "" else 0.0)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base, "finder_score.csv"), index=False)
    pts = df.groupby("patient")
    res = {"discs": len(df), "patients": df.patient.nunique(), "found": int(df.found.sum()),
           "found_rate": df.found.mean(), "dice_mean": df.dice.mean(), "dice_median": df.dice.median(),
           **({} if bare else {"tss_dice_mean": df.tss_dice.mean()}), "centre_mm_median": df.centre_mm.median(),
           "centre_mm_p95": df.centre_mm.quantile(0.95), "level_ok_rate": df.level_ok.mean(),
           "patients_all_levels_ok": float((pts.level_ok.min() == 1).mean())}
    rng = np.random.default_rng(0)                                   # 95% interval by patient, 2,000 draws
    ids = df.patient.unique()
    by = {p: g for p, g in pts}
    boot = {"dice_mean": [], "found_rate": [], "level_ok_rate": []}
    for _ in range(2000):
        g = pd.concat([by[p] for p in rng.choice(ids, len(ids))])
        boot["dice_mean"].append(g.dice.mean())
        boot["found_rate"].append(g.found.mean())
        boot["level_ok_rate"].append(g.level_ok.mean())
    for k, v in boot.items():
        res[k + "_low"], res[k + "_high"] = np.percentile(v, [2.5, 97.5])
    print(json.dumps({k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    if led is not None:
        for k, v in res.items():
            led.number(k, float(v), "count" if k in ("discs", "patients", "found") else "")
    return res


def _no_tss():
    from match import no_tss
    return no_tss()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict", "score", "segment", "cache"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--folds", default=None, help="predict/score: only these folds' patients, e.g. 0 or 0,1")
    a = ap.parse_args()
    folds = None if a.folds is None else sorted({int(f) for f in a.folds.split(",")})
    tag = "" if folds is None else " folds " + ",".join(map(str, folds))
    if a.mode == "train":
        for f in a.args:
            with ledger.run("segmenter", f"spine disc finder v{VERSION} fold {f}") as led:
                led.settings_from({"fold": f, "data": "SPIDER dev masks", **{f"cfg.{k}": v for k, v in CFG.items()}})
                train(f)
    elif a.mode == "predict":
        assert a.args == ["dev"], "only the development split: the test split opens through final_eval alone"
        with ledger.run("segmenter", f"spine disc finder v{VERSION} predict dev{tag}") as led:
            print(predict_dev(folds))
    elif a.mode == "score":
        assert a.args == ["dev"], "only the development split"
        with ledger.run("score", f"spine disc finder v{VERSION} dev out of fold{tag}") as led:
            led.settings_from({"split": "dev", "folds": 5 if folds is None else ",".join(map(str, folds)), "min_dice": 0.3,
                               **({"localisation_scope": "nominal level, no TotalSpineSeg"} if _no_tss() else {})})
            score_dev(led, folds)
    elif a.mode == "cache":
        assert a.args == ["dev"], "only the development split"
        t0 = time.time()
        for p in sorted(split_patients()):
            load_case(p)
        print(f"finder_cache: {len(split_patients())} patients in {time.time() - t0:.0f}s", flush=True)
    else:
        print(segment(*a.args))
