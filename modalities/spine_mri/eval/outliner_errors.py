"""Where spine outliner v2 loses Dice on SPIDER, out of fold, and which change would win it back: the next-run picker.

Owner, 2026-11-10, after holding the AWS ceiling on a retrain that would only fix the canal: work on what decides the
most accurate next run before paying for it. Every development patient is read by its own fold's model (as
outliner.predict reads it, on the CPU if there is no card) and each class's loss is split into buckets, each owned by
one kind of change:

  labelling  Dice the net drew but the numbering step threw away (a vertebra or disc voxel left id 0: off the disc
             chain, past its ends, or no chain at all)                                     -> code, no training, $0
  edge       wrong voxels within EDGE_MM of the true outline, on the same slice           -> a finer grid or sharper training
  chunk      wrong voxels farther off (a missed or extra piece, a slice drawn empty)     -> capacity, context, iterations
  side       of all wrong voxels, those on the outer quarter of the structure's sagittal slices (overlaps edge and
             chunk: a where, not a what)                                                   -> more through-plane context
  ceiling    the Dice a perfect net could reach at MM in-plane: the true mask taken to MM and back up the way
             probabilities() brings the net's output back (bilinear), 0.5 threshold       -> the grid's own limit

Gains are Dice points (mean over patients) if that bucket were fixed outright; they do not add up exactly (fixing edges
changes what a chunk is). TotalSpineSeg's own outlines, when --tss is given, get the same edge / chunk / side split, so
the gap to it can be read bucket by bucket. TotalSpineSeg trained on SPIDER: its buckets are in-sample, a generous bar.

  python outliner_errors.py [--tss DIR] [--patients N] [--workers K]
      -> <work>/eval/spine_outliner/errors_{patients.csv,summary.json}, the ranked picker printed

The models are read from outliner.OUT (RAIDIOLOGY_MODELS); SPIDER's T2 and masks from prepare.WORK (RAIDIOLOGY_WORK)."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # the repository root
import outliner as O  # noqa: E402

EDGE_MM = 1.0          # SPIDER's in-plane spacing is 0.3-0.9 mm: 1 mm is one or two native pixels, the net's own MM pixel
SIDE = 0.25            # the outer quarter of a structure's sagittal slices, each end, at least one slice
CLASSES = {            # name: (truth from a SPIDER mask, final label test, the net's class)
    "vertebra": (lambda m: (m >= 1) & (m < 100), lambda lab: (lab >= 11) & (lab <= 50), O.VERT),
    "disc": (lambda m: m > 200, lambda lab: lab >= 63, O.DISC),
    "canal": (lambda m: m == 100, lambda lab: (lab == 1) | (lab == 2), None),   # canal-or-cord, as score()
}
# TotalSpineSeg draws the sacrum (50), which SPIDER does not label and v2 never draws: left out, as outliner_vs_tss
TSS_FINAL = {**{k: v[1] for k, v in CLASSES.items()}, "vertebra": lambda lab: (lab >= 11) & (lab <= 45)}
# bucket -> (the change that owns it, its price against one run of today's settings)
CHANGES = {
    "labelling": ("fix the numbering step so it keeps every voxel the net drew", "code only: $0, rescore run 3's models"),
    "edge": ("sharpen outlines: a finer in-plane grid (MM 0.75 or 0.5) or a longer schedule",
             "MM 0.75: ~1.8x the training time; MM 0.5: ~4x; iterations: linear"),
    "chunk": ("draw the whole structure: a wider net (width 48-64) and more iterations",
              "width 48: ~2.2x; width 64: ~4x; iterations: linear"),
    "side": ("see more neighbours: 5 sagittal slices in, not 3", "~1.05x (first layer only); needs a retrain"),
    "ceiling": ("nothing short of a finer grid lifts the ceiling", "MM 0.75: ~1.8x; MM 0.5: ~4x"),
}


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else np.nan


def inplane_dist(target, z):
    """Per sagittal slice, the in-plane distance (mm) of every voxel to the nearest `target` voxel on the same slice;
    inf on a slice where target is empty."""
    out = np.full(target.shape, np.inf, np.float32)
    for k in range(target.shape[2]):
        t = target[:, :, k]
        if t.any():
            out[:, :, k] = ndimage.distance_transform_edt(~t, sampling=(z[0], z[1]))
    return out


def split_loss(gt, pr, z):
    """(dice, edge-forgiven dice, side-forgiven dice, edge share of wrong voxels, side share) of prediction pr."""
    d = dice(gt, pr)
    fn, fp = gt & ~pr, pr & ~gt
    tp = int((gt & pr).sum())
    fn_e = int((fn & (inplane_dist(pr, z) <= EDGE_MM)).sum())
    fp_e = int((fp & (inplane_dist(gt, z) <= EDGE_MM)).sum())
    nfn, nfp = int(fn.sum()), int(fp.sum())
    tp_e = tp + fn_e
    d_edge = 2.0 * tp_e / (2 * tp_e + (nfp - fp_e) + (nfn - fn_e)) if tp_e + nfn + nfp else np.nan
    ks = np.flatnonzero(gt.any((0, 1)))
    side = np.zeros(gt.shape[2], bool)
    n = max(1, int(round(SIDE * len(ks))))
    if len(ks) > 2 * n:            # a structure on one or two slices has no outer slices apart from its middle
        side[ks[:n]] = side[ks[-n:]] = True
    fn_s, fp_s = int(fn[:, :, side].sum()), int(fp[:, :, side].sum())
    tp_s = tp + fn_s
    d_side = 2.0 * tp_s / (2 * tp_s + (nfp - fp_s) + (nfn - fn_s)) if tp_s + nfn + nfp else np.nan
    wrong = max(nfn + nfp, 1)
    return d, d_edge, d_side, (fn_e + fp_e) / wrong, (fn_s + fp_s) / wrong


def ceiling(gt, z):
    """Dice of the true mask after the round trip a perfect net's output makes: down to MM, back up bilinear, 0.5."""
    import torch
    r = O.resample_slices(gt.astype(np.float32), z, 1)              # (LR, AP', SI') at MM
    up = torch.nn.functional.interpolate(torch.from_numpy(r)[:, None], size=gt.shape[:2], mode="bilinear",
                                         align_corners=False)[:, 0].numpy()
    return dice(gt, np.moveaxis(up, 0, -1) > 0.5)


def missed(m, pr, lo, hi):
    """(structures in the truth with ids in [lo, hi], those with under half their voxels drawn)."""
    ids = [int(v) for v in np.unique(m) if lo <= v <= hi]
    return len(ids), sum(1 for v in ids if pr[m == v].mean() < 0.5)


def one(job):
    """Every bucket for one SPIDER development patient."""
    import torch
    torch.set_num_threads(job["threads"])
    p, f, tss_dir = job["patient"], job["fold"], job["tss"]
    img = nib.load(os.path.join(O.SPIDER_WORK, "dev", "t2", f"{p}.nii.gz"))
    a, aff, _ = O.to_pir(img)
    z = O.zooms(aff)
    seg, heat, level = O.probabilities(O.load(str(f)), O.scaled(O.Series(a=a, aff=aff)), aff)
    lab, _ = O.outline(seg, heat, level, aff)
    cls = seg.argmax(0)
    m = O.to_pir(nib.load(os.path.join(O.SPIDER_WORK, "dev", "mask", f"{p}.nii.gz")))[0].astype(np.int32)
    t = None
    if tss_dir and os.path.exists(os.path.join(tss_dir, f"{p}.nii.gz")):
        t = O.to_pir(nib.load(os.path.join(tss_dir, f"{p}.nii.gz")))[0].astype(np.int32)
    row = {"patient": str(p), "fold": int(f), "zoom_ap": round(float(z[0]), 3), "zoom_si": round(float(z[1]), 3),
           "zoom_lr": round(float(z[2]), 3), "slices": int(a.shape[2])}
    for name, (truth, final, c) in CLASSES.items():
        gt = truth(m)
        fin = final(lab)
        raw = (cls == O.CANAL) | (cls == O.CORD) if c is None else cls == c
        row[f"{name}_final"] = dice(gt, fin)
        row[f"{name}_raw"] = dice(gt, raw)
        _, row[f"{name}_edge_forgiven"], row[f"{name}_side_forgiven"], row[f"{name}_edge_share"], \
            row[f"{name}_side_share"] = split_loss(gt, raw, z)
        row[f"{name}_ceiling"] = ceiling(gt, z)
        if t is not None:
            tp = TSS_FINAL[name](t)
            row[f"{name}_tss"], row[f"{name}_tss_edge_forgiven"], row[f"{name}_tss_side_forgiven"], \
                row[f"{name}_tss_edge_share"], row[f"{name}_tss_side_share"] = split_loss(gt, tp, z)
    row["vertebrae"], row["vertebrae_missed"] = missed(m, (cls == O.VERT), 1, 99)
    row["discs"], row["discs_missed"] = missed(m, (cls == O.DISC), 201, 299)
    return row


def summarise(df, tss):
    """{class: {bucket: mean Dice gain}} and the picker: buckets ranked by their gain on vertebra + disc."""
    out = {}
    for name in CLASSES:
        r = {"final": df[f"{name}_final"].mean(), "raw": df[f"{name}_raw"].mean(), "ceiling": df[f"{name}_ceiling"].mean()}
        r["labelling"] = max(0.0, r["raw"] - r["final"])
        r["edge"] = (df[f"{name}_edge_forgiven"] - df[f"{name}_raw"]).mean()
        r["chunk"] = 1.0 - df[f"{name}_edge_forgiven"].mean()
        r["side"] = (df[f"{name}_side_forgiven"] - df[f"{name}_raw"]).mean()
        r["edge_share"] = df[f"{name}_edge_share"].mean()
        r["side_share"] = df[f"{name}_side_share"].mean()
        if tss:
            r["tss"] = df[f"{name}_tss"].mean()
            r["tss_edge"] = (df[f"{name}_tss_edge_forgiven"] - df[f"{name}_tss"]).mean()
            r["tss_chunk"] = 1.0 - df[f"{name}_tss_edge_forgiven"].mean()
            r["tss_side"] = (df[f"{name}_tss_side_forgiven"] - df[f"{name}_tss"]).mean()
        out[name] = {k: round(float(v), 4) for k, v in r.items()}
    vd = ("vertebra", "disc")
    gap = {b: sum(out[n][b] for n in vd) / 2 for b in ("labelling", "edge", "chunk", "side")}
    if tss:            # what we lose beyond what TotalSpineSeg loses in the same bucket: the part a run can close
        for b in ("edge", "chunk", "side"):
            gap[b] = sum(out[n][b] - out[n][f"tss_{b}"] for n in vd) / 2
    gap["ceiling"] = sum(1.0 - out[n]["ceiling"] for n in vd) / 2
    out["missed"] = {"vertebrae": int(df.vertebrae_missed.sum()), "of_vertebrae": int(df.vertebrae.sum()),
                     "discs": int(df.discs_missed.sum()), "of_discs": int(df.discs.sum())}
    out["picker"] = [{"bucket": b, "gain": round(float(g), 4), "change": CHANGES[b][0], "price": CHANGES[b][1]}
                     for b, g in sorted(gap.items(), key=lambda kv: -kv[1])]
    out["patients"] = int(len(df))
    return out


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--tss", default=os.path.join(O.SPIDER_WORK, "dev", "tss", "step2_output"))
    ap.add_argument("--patients", type=int, default=0, help="only the first N per fold (0: every patient)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 6))
    a = ap.parse_args()
    folds = O.split_patients()
    if a.patients:
        folds = {p: f for k in sorted(set(folds.values()))
                 for p, f in [(q, g) for q, g in sorted(folds.items()) if g == k][:a.patients]}
    tss = a.tss if os.path.isdir(a.tss) else ""
    threads = max(1, (os.cpu_count() or 2) // a.workers)
    jobs = [{"patient": p, "fold": f, "tss": tss, "threads": threads} for p, f in sorted(folds.items())]
    from core import ledger
    with ledger.run("score", "spine outliner v2: where SPIDER Dice is lost, out of fold, by bucket (next-run picker)",
                    note="buckets: labelling, edge (<= %g mm in-plane), chunk, side (outer %g of slices), 1 mm ceiling; "
                         "TotalSpineSeg in-sample, sacrum left out" % (EDGE_MM, SIDE)) as led:
        led.settings_from({"models": O.OUT, "patients": len(jobs), "edge_mm": EDGE_MM, "side": SIDE, "mm": O.MM,
                           "tss": bool(tss)})
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, jobs))
        df = pd.DataFrame(rows)
        os.makedirs(O.V2, exist_ok=True)
        df.to_csv(os.path.join(O.V2, "errors_patients.csv"), index=False)
        s = summarise(df, bool(tss))
        with open(os.path.join(O.V2, "errors_summary.json"), "w") as fh:
            json.dump(s, fh, indent=1)
        for name in CLASSES:
            for k, v in s[name].items():
                led.number(f"{name}_{k}", v)
        for r in s["picker"]:
            led.number(f"picker_{r['bucket']}", r["gain"])
    print(json.dumps({k: v for k, v in s.items() if k != "picker"}, indent=1))
    print("next-run picker (Dice points on vertebra + disc, mean of the two%s):" % (", beyond TotalSpineSeg's own" if tss else ""))
    for r in s["picker"]:
        print(f"  {r['bucket']:9s} {r['gain']:+.4f}  {r['change']}  [{r['price']}]")
    return s


if __name__ == "__main__":
    main()
