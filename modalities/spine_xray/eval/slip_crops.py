"""Slipped vertebrae read from the picture: a CNN classifies a crop around each spinal level, not corner points.

A small CNN classifies a crop around each spinal level of a side-view lumbar X-ray, instead of measuring
the slip from corner points.

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646 (LICENSING.md, BUU-LSPINE row). Plan and pass bar:
docs/efficiency/README.md, "Slip reader v2: reading the joint, not measuring it".

Why. slip.py measured the slip from corners.py v1's corners and was NOT KEPT: AUROC 0.809 against the
drawn corners' 0.871. The corners are close (8 to 12 px) but a slip is a few pixels between two endplates,
so a miss at the wrong corner moves it a whole grade. Here the net reads the joint itself.

The crops. Level i (0..4) is L(i+1) over L(i+2), the last being L5 over S1. Its four corners are the two
ends of the upper vertebra's lower endplate (line 2i + 1, points 4i + 2 and 4i + 3) and the two ends of the
lower vertebra's upper endplate (line 2i + 2, points 4i + 4 and 4i + 5): the same lines slip.py's slips()
uses. The crop is a square centred on the mean of those four points, its side twice the mean length of the
two endplates, resized to 128 px, taken from the 512 x 384 canvas buu.load caches (aspect kept, padded
black); the level goes in beside the picture as a one-hot of five. A side film is never flipped.
  * Training: the drawn corners place the crop, each corner first moved at random (uniformly inside a disc)
    by up to the corner reader's median miss, 8 px at full resolution, so the net learns from crops placed as
    roughly as ours will be. The move is drawn afresh every epoch.
  * Judging: ONLY corners.py v1's out-of-fold corners (work/eval/spine_xray/corners-v1/LA-fold<k>.npz) place
    a crop. The drawn corners never place a judged crop. Judged crops are built once per run.

The net and the training. Four conv blocks, flattened (a slip is where two edges sit against each other,
so the position in the crop is kept), the level's one-hot beside it, then a small head. From scratch.
Binary cross-entropy for "this level is marked class 3 or 4" (the mark sits on the upper vertebra's
lower-edge row, as buu.py says), positives weighted by the training crops' negative to positive ratio.
AdamW, cosine schedule, a fixed 30 epochs, no early stopping: the last epoch is the model. The five
patient folds are corners.py's (split.json). The out-of-fold probabilities of fold k come from a net
trained on the other four folds. The device is corners.py's (RAIDIOLOGY_DEVICE=cpu runs on the processor);
on the card it leases RAIDIOLOGY_SLIP_CARD_GB (default 1.5) as a short job of 15 minutes.

The score is a patient's largest level probability. Bar: KEPT when the out-of-fold patient AUROC minus the
drawn corners' amended score (slip.py's own code, the same patients) has a paired patient-bootstrap 95%
interval (1000 resamples, seed 0, one resample for both, as slip.py does) whose low end is -0.03 or above.
Printed beside it: the AUROC against v1's 0.8093, accuracy, sensitivity and specificity at thresholds
(Youden's J) chosen inside each training fold, and the AUROC of each level.

Usage: python slip_crops.py --version N [--epochs 30] [--folds 0,1,2,3,4]
Outputs: <WORK>/eval/spine_xray/slip-crops-v<N>/oof.npz (patient ids, fold, level probabilities, level labels)
"""
import argparse
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
from core import paths  # noqa: E402
import buu  # noqa: E402
import corners  # noqa: E402
import slip  # noqa: E402

DEV = corners.DEV
VIEW = slip.VIEW
CORNERS_VERSION = 1             # the corner reader whose out-of-fold corners place every judged crop
CROP = 128
SIDE_OVER_ENDPLATE = 2.0        # the crop's side, in mean endplate lengths
JITTER_PX = 8.0                 # the corner reader's median miss at full resolution
EPOCHS = 30
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP = 1
WIDTHS = [24, 48, 96, 128]
SEED = 0
V1_AUROC = 0.8093               # slip.py on corners v1, out of fold, all 2,000 patients
CARD_GB = 1.5
SHORT_MIN = 15


def out_dir(version):
    return os.path.join(paths.WORK, "eval", "spine_xray", "slip-crops-v%d" % version)


def limit_card_memory():
    """Lease the card as a short job before touching CUDA, then cap the process at that size."""
    if DEV.type != "cuda":
        return
    gb = float(os.environ.get("RAIDIOLOGY_SLIP_CARD_GB", CARD_GB))
    card.lease(gb, "spine slip crops", short_min=SHORT_MIN)
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, gb * 2 ** 30 / total), torch.cuda.current_device())


# --- the crops ---------------------------------------------------------------------------------------

def level_geometry(pts):
    """(N, 22, 2) corners -> centres (N, 5, 2) and sides (N, 5). Level i's four corners are points 4i + 2 to
    4i + 5 (the upper vertebra's lower edge, then the lower vertebra's upper edge)."""
    pts = np.asarray(pts, np.float64)
    centre = np.zeros((len(pts), 5, 2))
    side = np.zeros((len(pts), 5))
    for i in range(5):
        four = pts[:, 4 * i + 2:4 * i + 6]
        centre[:, i] = four.mean(axis=1)
        upper = np.linalg.norm(four[:, 0] - four[:, 1], axis=-1)
        lower = np.linalg.norm(four[:, 2] - four[:, 3], axis=-1)
        side[:, i] = SIDE_OVER_ENDPLATE * (upper + lower) / 2
    return centre, side


def cut(pix, rows, centre, side, chunk=256):
    """Crops (M, CROP, CROP) uint8 from pix (N, H, W) uint8: image rows[m], a square at centre[m] (x, y in
    canvas pixels, continuous), side[m] canvas pixels, bilinear. Black outside the canvas."""
    h, w = pix.shape[-2:]
    out = np.zeros((len(rows), CROP, CROP), np.uint8)
    for a in range(0, len(rows), chunk):
        r = torch.from_numpy(rows[a:a + chunk])
        x = pix[r].float().unsqueeze(1)
        c = torch.from_numpy(centre[a:a + chunk]).float()
        s = torch.from_numpy(side[a:a + chunk]).float()
        theta = torch.zeros(len(r), 2, 3)
        theta[:, 0, 0] = s / w
        theta[:, 1, 1] = s / h
        theta[:, 0, 2] = 2 * c[:, 0] / w - 1
        theta[:, 1, 2] = 2 * c[:, 1] / h - 1
        grid = F.affine_grid(theta, [len(r), 1, CROP, CROP], align_corners=False)
        y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        out[a:a + chunk] = y[:, 0].round().clamp(0, 255).to(torch.uint8).numpy()
    return out


def crops_of(pix, rows, pts):
    """Every level's crop of images `rows` (n,) placed by corners pts (n, 22, 2): (n * 5, CROP, CROP), image-major."""
    centre, side = level_geometry(pts)
    return cut(pix, np.repeat(rows, 5), centre.reshape(-1, 2), side.reshape(-1))


def jittered(pts, scale, rng):
    """Every corner moved uniformly inside a disc of JITTER_PX full-resolution pixels (scale: canvas px per
    original px, (n,))."""
    r = JITTER_PX * scale[:, None] * np.sqrt(rng.random(pts.shape[:2]))
    a = rng.random(pts.shape[:2]) * 2 * math.pi
    return pts + np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)


# --- the net -----------------------------------------------------------------------------------------

class Joint(nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers, c = [], 1
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = w
        layers += [nn.Conv2d(c, 32, 1), nn.ReLU(inplace=True)]                        # 128 / 2^4 = 8 cells a side
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(32 * (CROP // 2 ** len(widths)) ** 2 + 5, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x, level):
        h = self.body(x).flatten(1)
        return self.head(torch.cat([h, F.one_hot(level, 5).to(h.dtype)], dim=1))[:, 0]


def prepare(u8):
    """(B, CROP, CROP) uint8 on the device -> (B, 1, CROP, CROP) float, each crop standardised on its own."""
    x = u8.float().div_(255.0).unsqueeze(1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
    return (x - m) / s


def nudge(u8, g):
    """A brightness and contrast nudge per crop, as corners.py's augment (no flip, no warp: the crop's
    placement is already jittered by moving the corners)."""
    b = u8.shape[0]
    u = lambda: (torch.rand(b, 1, 1, device=u8.device, generator=g) * 2 - 1)           # noqa: E731
    x = u8.float()
    m = x.mean(dim=(1, 2), keepdim=True)
    return ((x - m) * (1 + 0.15 * u()) + m + 255 * 0.075 * u()).clamp(0, 255)


def predict(net, crops):
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(crops), 256):
            u8 = torch.from_numpy(crops[a:a + 256]).to(DEV)
            level = (torch.arange(a, a + len(u8), device=DEV) % 5)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                logit = net(prepare(u8), level)
            out.append(torch.sigmoid(logit.float()).cpu())
    return torch.cat(out).numpy()


def train_fold(fold, pix, train_rows, train_pts, train_scale, train_label, judged, epochs, pace):
    """Train on the other folds' patients (corners redrawn with jitter every epoch), return the held-out
    crops' probabilities and the training seconds."""
    t0 = time.time()
    torch.manual_seed(SEED + fold)
    rng = np.random.default_rng(SEED + fold)
    g_cpu = torch.Generator().manual_seed(SEED + fold)
    g_dev = torch.Generator(device=DEV).manual_seed(SEED + fold)
    net = Joint(WIDTHS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_all = train_label.reshape(-1).astype(np.float32)
    pos_weight = torch.tensor((len(y_all) - y_all.sum()) / max(y_all.sum(), 1.0), device=DEV)
    per_epoch = len(y_all) // BATCH
    total, warm = epochs * per_epoch, WARMUP * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    print("fold %d: train %d patients (%d crops, %d marked, positive weight %.1f), %d steps an epoch"
          % (fold, len(train_rows), len(y_all), int(y_all.sum()), float(pos_weight), per_epoch), flush=True)
    for ep in range(1, epochs + 1):
        began_ep = time.time()
        crops = crops_of(pix, train_rows, jittered(train_pts, train_scale, rng))
        net.train()
        order = torch.randperm(len(y_all), generator=g_cpu).numpy()
        tot = torch.zeros((), device=DEV)
        for s in range(per_epoch):
            began = time.time()
            idx = order[s * BATCH:(s + 1) * BATCH]
            u8 = nudge(torch.from_numpy(crops[idx]).to(DEV), g_dev)
            y = torch.from_numpy(y_all[idx]).to(DEV)
            level = torch.from_numpy(idx % 5).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                logit = net(prepare(u8), level)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot = tot + loss.detach()
            if DEV.type == "cuda" and (s + 1) % 8 == 0:
                torch.cuda.synchronize()
            pace.rest(time.time() - began)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print("  epoch %3d  loss %.4f  %.0f s" % (ep, float(tot) / per_epoch, time.time() - began_ep), flush=True)
    seconds = time.time() - t0
    prob = predict(net, judged)
    del net, opt, sched
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return prob, seconds


# --- judging -----------------------------------------------------------------------------------------

def main():
    t0 = time.time()
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, required=True, help="names the output folder slip-crops-v<N>")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--folds", default="0,1,2,3,4", help="'all' or comma-separated fold numbers (default all five)")
    a = ap.parse_args()
    folds = corners.parse_folds(a.folds)
    if len(folds) < 2:
        raise SystemExit("a threshold chosen inside the training folds needs at least two folds scored; got %s" % folds)

    card.require_card(DEV, torch)
    # 2026-10-03: leased before the data is loaded (a job waiting with its data held up to 6.9 GB; the machine
    # ran out of committed memory at 23:21 and a training job crashed)
    limit_card_memory()
    data = buu.load(VIEW)
    ids, truth_pts, classes = data["ids"], data["pts"], data["classes"]
    fold_all = buu.folds_of(ids)
    label_all = buu.slip_label(classes)
    level_all = np.isin(classes[:, 1:10:2], buu.SLIP_CLASSES).astype(np.int64)       # lines 1, 3, 5, 7, 9
    scale_all = data["scale"][:, 0].astype(np.float64)
    pix = torch.from_numpy(data["pix"])

    # corners v1's out-of-fold corners (original pixels) -> canvas pixels, for the judged patients
    r = corners.load_oof(VIEW, CORNERS_VERSION, folds)
    where = {int(i): n for n, i in enumerate(ids)}
    keep = np.array(sorted(where[int(i)] for i in r["ids"]))
    order = {int(i): n for n, i in enumerate(r["ids"])}
    oof_o = r["pred"][[order[int(i)] for i in ids[keep]]].astype(np.float64)
    oof_pts = oof_o * data["scale"][keep][:, None, :] + data["offset"][keep][:, None, :]
    pid, fold, label, level_label = ids[keep], fold_all[keep], label_all[keep], level_all[keep]
    print("slip reader v2: %d judged patients in folds %s, %d with a side-view slip (class 3 or 4); device %s"
          % (len(pid), ",".join(map(str, folds)), int(label.sum()), DEV), flush=True)
    judged = crops_of(pix, keep, oof_pts)                  # placed by corners v1 only, built once
    print("judged crops: %d (%d patients x 5 levels, %d px) placed by corners v%d's out-of-fold corners"
          % (len(judged), len(pid), CROP, CORNERS_VERSION), flush=True)

    prob = np.zeros((len(pid), 5))
    seconds = {}
    for k in folds:
        tr = np.where(fold_all != k)[0]
        te = np.where(fold == k)[0]
        rows = np.concatenate([np.arange(i * 5, i * 5 + 5) for i in te])
        p, took = train_fold(k, pix, tr, truth_pts[tr].astype(np.float64), scale_all[tr], level_all[tr],
                             judged[rows], a.epochs, polite.Pace())
        prob[te] = p.reshape(-1, 5)
        seconds[k] = took
        print("fold %d: trained in %.0f s, %d held-out patients scored" % (k, took, len(te)), flush=True)
    os.makedirs(out_dir(a.version), exist_ok=True)
    npz = os.path.join(out_dir(a.version), "oof.npz")
    np.savez(npz, ids=pid, fold=fold, prob=prob, level_label=level_label, label=label)
    print("wrote %s" % npz, flush=True)

    score = prob.max(axis=1)
    n_pos = int(label.sum())
    print("per level (out-of-fold probability, marks as BUU's class 3 or 4 on the upper vertebra's lower edge):")
    for i, name in enumerate(slip.LEVELS):
        pos = int(level_label[:, i].sum())
        lv = slip.auroc(prob[:, i], level_label[:, i]) if pos >= 5 else float("nan")
        print("  %-6s marks %3d  AUROC for the level's marks %s" % (name, pos, "%.3f" % lv if lv == lv else "n/a"))
    call = np.zeros(len(pid), bool)
    cuts = {}
    for k in folds:
        tr_, te_ = fold != k, fold == k
        cuts[k] = slip.youden(score[tr_], label[tr_])
        call[te_] = score[te_] >= cuts[k]
    acc, sens, spec = slip.rates(call, label)
    idx = np.random.default_rng(slip.BOOT_SEED).integers(0, len(pid), (slip.BOOT, len(pid)))
    accs, senss, specs = [], [], []
    for rr in idx:
        ac, se, sp = slip.rates(call[rr], label[rr])
        accs.append(ac), senss.append(se), specs.append(sp)

    def boot_auc(sc):
        return np.array([slip.auroc(sc[rr], label[rr]) for rr in idx])

    auc, aucs = slip.auroc(score, label), boot_auc(score)
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    drawn_o = (truth_pts[keep] - data["offset"][keep][:, None, :]) / data["scale"][keep][:, None, :]
    dscore = slip.amended(slip.slips(drawn_o), fold)                  # the drawn corners, slip.py's own score
    dauc, daucs = slip.auroc(dscore, label), boot_auc(dscore)
    dlo, dhi = np.nanpercentile(daucs, [2.5, 97.5])
    dflo, dfhi = np.nanpercentile(aucs - daucs, [2.5, 97.5])          # same resample for both
    d = auc - dauc
    print("Crops, largest level probability: AUROC %.4f (patient-bootstrap 95%% CI %.4f to %.4f, %d resamples, seed %d)"
          % (auc, lo, hi, slip.BOOT, slip.BOOT_SEED))
    print("Drawn corners, amended score: AUROC %.4f (%.4f to %.4f)" % (dauc, dlo, dhi))
    print("Crops minus drawn corners: %+.4f (paired patient-bootstrap 95%% CI %+.4f to %+.4f)" % (d, dflo, dfhi))
    print("Crops minus slips v1 (corners v1 measured, AUROC %.4f): %+.4f" % (V1_AUROC, auc - V1_AUROC))
    print("target, context only: AUROC low end %.4f against %.2f" % (lo, slip.BAR))
    print("thresholds chosen inside each training fold: %s" % ", ".join("fold %d %.3f" % (k, c) for k, c in cuts.items()))
    print("at those thresholds: accuracy %.3f (%.3f to %.3f; the paper's %.3f), sensitivity %.3f (%.3f to %.3f), "
          "specificity %.3f (%.3f to %.3f); a reader that never calls a slip scores %.3f"
          % (acc, *np.percentile(accs, [2.5, 97.5]), slip.PAPER_ACCURACY, sens, *np.percentile(senss, [2.5, 97.5]),
             spec, *np.percentile(specs, [2.5, 97.5]), 1 - n_pos / len(pid)))
    print("training seconds per fold: %s" % ", ".join("fold %d %.0f" % (k, s) for k, s in seconds.items()))
    kept = bool(dflo >= -slip.MARGIN)
    print("Verdict (slips v2, not worse than drawn corners, margin %.2f): %s" % (slip.MARGIN, "KEPT" if kept else "NOT KEPT"))
    ver = slip.verified_report(pid, label, level_label, score, call, prob)
    numbers = {**ver, "auroc": (auc, ""), "auroc_low": (lo, ""), "auroc_high": (hi, ""), "auroc_drawn": (dauc, ""),
               "auroc_diff": (d, ""), "auroc_diff_low": (dflo, ""), "auroc_diff_high": (dfhi, ""),
               "auroc_v1": (V1_AUROC, ""), "accuracy": (acc, ""), "sensitivity": (sens, ""), "specificity": (spec, ""),
               "patients": (len(pid), "count"), "patients_with_slip": (n_pos, "count"), "epochs": (a.epochs, "count"),
               "train_seconds_mean": (float(np.mean(list(seconds.values()))), "s"), "kept": (float(kept), "")}
    numbers = {k: (float(v), u) for k, (v, u) in numbers.items()}                 # numpy floats are not JSON
    settings = {"source": "crops placed by corners v%d out-of-fold corners" % CORNERS_VERSION,
                "folds": ",".join(map(str, folds)), "epochs": a.epochs, "crop": CROP, "jitter_px": JITTER_PX,
                "verdict": "KEPT" if kept else "NOT KEPT",
                "bar": "AUROC not worse than drawn corners, margin %.2f" % slip.MARGIN}
    ledger.write("experiment", "spine xray slips v2: crops around each level, v%d out of fold, any class 3 or 4 per patient"
                 % a.version, time.time() - t0, numbers, settings=settings,
                 note="modalities/spine_xray/eval/slip_crops.py; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


if __name__ == "__main__":
    main()
