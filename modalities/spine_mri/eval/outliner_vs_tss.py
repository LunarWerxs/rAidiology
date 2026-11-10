"""Spine outliner v2 against TotalSpineSeg on the same SPIDER development patients, both scored against SPIDER's expert
masks, paired by patient with a bootstrap 95% interval: the evidence for switching the patient pipeline to v2.

TotalSpineSeg is read here only as a comparison (its step-2 labels, already on disk from the earlier SPIDER runs); nothing
it drew trains anything. Both sides are scored by the same rules as outliner.score: vertebra Dice over C1-L5 ids (11-45;
TotalSpineSeg's sacrum, id 50, is left out because SPIDER does not label the sacrum and v2 never draws it), disc Dice
(ids 63+), canal Dice (cord or canal ids against SPIDER's canal, which holds the conus), and per disc: found (Dice 0.3 or
more) and the right level. TotalSpineSeg trained on SPIDER (its README lists it), so its numbers here are on patients it
may have seen: a generous bar; v2 is scored out of fold.

  python outliner_vs_tss.py tss                      -> <work>/eval/spine_outliner/tss_spider_{patients,discs}.csv
  python outliner_vs_tss.py compare <v2_score_dir>   -> paired differences (v2 - TotalSpineSeg), printed and in the ledger,
     over every patient and again over the 143 the level fix was not tuned on (TUNED_ON)
     <v2_score_dir> holds outliner.score's score_spider_patients.csv and score_spider_discs.csv (the box run's copies)."""
import os
import sys

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # the repository root
from core import ledger, paths  # noqa: E402
paths.use("spine_mri")
from prepare import WORK as SPIDER_WORK, disc_id as spider_disc_id  # noqa: E402

TSS_DIR = os.path.join(SPIDER_WORK, "dev", "tss", "step2_output")
MASKS = os.path.join(SPIDER_WORK, "dev", "mask")
OUT = os.path.join(paths.WORK, "eval", "spine_outliner")
SEED = 20261110
# The 25 SPIDER patients (the first five of each fold) whose disc points tuned commit 1eb64d1a's level fix
# (merge_slices, the weighted chain offset, the 1.5-level cut): their scores are not out of sample for that fix, so
# compare also reports the other 143 on their own, as the honest number.
TUNED_ON = frozenset({"11", "13", "37", "44", "47", "12", "23", "28", "31", "34", "10", "16", "17", "19", "35",
                      "3", "8", "9", "30", "51", "2", "15", "18", "20", "32"})


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else np.nan     # as outliner.dice: both empty is no evidence either way


def score_one(m, lab):
    """(patient row, disc rows) for one expert mask `m` and one outline `lab` on the same grid."""
    row = {"dice_vertebra": dice((m >= 1) & (m < 100), (lab >= 11) & (lab <= 45)),
           "dice_disc": dice(m > 200, lab >= 63), "dice_canal": dice(m == 100, (lab == 1) | (lab == 2))}
    discs = []
    for i in sorted({int(v) - 200 for v in np.unique(m) if v > 200}):
        want = spider_disc_id(i)
        if want is None:
            continue
        gt = m == 200 + i
        inside = lab[gt]
        inside = inside[inside >= 63]
        best = int(np.bincount(inside).argmax()) if len(inside) else 0
        d = dice(gt, lab == best) if best else 0.0
        discs.append({"ivd": i, "dice": d, "found": int(d >= 0.3), "level_ok": int(d >= 0.3 and best == want)})
    return row, discs


def tss():
    import pandas as pd
    per, spd = [], []
    for f in sorted(os.listdir(TSS_DIR)):
        if not f.endswith(".nii.gz"):
            continue
        key = f[:-7]
        m = np.asarray(nib.load(os.path.join(MASKS, f)).dataobj).astype(np.int32)
        lab = np.asarray(nib.load(os.path.join(TSS_DIR, f)).dataobj).astype(np.int32)
        assert m.shape == lab.shape, (key, m.shape, lab.shape)
        row, discs = score_one(m, lab)
        per.append({"patient": key, **row})
        spd += [{"patient": key, **d} for d in discs]
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(per).to_csv(os.path.join(OUT, "tss_spider_patients.csv"), index=False)
    pd.DataFrame(spd).to_csv(os.path.join(OUT, "tss_spider_discs.csv"), index=False)
    print(f"TotalSpineSeg scored on {len(per)} SPIDER development patients, {len(spd)} discs -> {OUT}")


def paired(a, b, by, col, n=2000):
    """Mean of a[col] - b[col] over patients present in both (disc rows averaged per patient first), 95% bootstrap CI."""
    pa, pb = a.groupby(by)[col].mean(), b.groupby(by)[col].mean()
    common = pa.index.intersection(pb.index)
    d = (pa[common] - pb[common]).to_numpy(float)
    rng = np.random.default_rng(SEED)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return len(common), float(pa[common].mean()), float(pb[common].mean()), float(d.mean()), *np.percentile(boot, [2.5, 97.5])


def compare(v2_dir, skip=frozenset()):
    """Paired v2 - TotalSpineSeg over the patients both scored, leaving out the patient keys in `skip`."""
    import pandas as pd
    v2p = pd.read_csv(os.path.join(v2_dir, "score_spider_patients.csv"), dtype={"patient": str})
    v2d = pd.read_csv(os.path.join(v2_dir, "score_spider_discs.csv"), dtype={"patient": str})
    tp = pd.read_csv(os.path.join(OUT, "tss_spider_patients.csv"), dtype={"patient": str})
    td = pd.read_csv(os.path.join(OUT, "tss_spider_discs.csv"), dtype={"patient": str})
    missing = set(skip) - set(v2p["patient"])
    assert not missing, f"patients to leave out that v2 never scored (a typo would keep one in): {sorted(missing)}"
    v2p, v2d = v2p[~v2p["patient"].isin(skip)], v2d[~v2d["patient"].isin(skip)]   # paired() drops them from TSS too
    res, lines = {}, []
    for name, a, b, col in (("vertebra_dice", v2p, tp, "dice_vertebra"), ("disc_dice", v2p, tp, "dice_disc"),
                            ("canal_dice", v2p, tp, "dice_canal"), ("disc_found", v2d, td, "found"),
                            ("disc_level_ok", v2d, td, "level_ok")):
        n, va, vb, diff, lo, hi = paired(a, b, "patient", col)
        res[name] = {"patients": n, "v2": round(va, 4), "tss": round(vb, 4), "diff": round(diff, 4),
                     "ci95": [round(lo, 4), round(hi, 4)]}
        verdict = "v2 better" if lo > 0 else "TotalSpineSeg better" if hi < 0 else "same within noise"
        lines.append(f"{name:14s} v2 {va:.4f}  TotalSpineSeg {vb:.4f}  diff {diff:+.4f} ({lo:+.4f} to {hi:+.4f})  "
                     f"{n} patients: {verdict}")
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "tss":
        tss()
    elif len(sys.argv) >= 3 and sys.argv[1] == "compare":
        with ledger.run("score", "spine outliner v2 vs TotalSpineSeg, SPIDER development (out of fold), paired by patient",
                        note="both against SPIDER's expert masks; TotalSpineSeg's sacrum id 50 left out") as led:
            led.settings_from({"v2_scores": sys.argv[2], "bootstrap": 2000, "seed": SEED, "found_min_dice": 0.3,
                               "tuned_on": len(TUNED_ON)})
            for prefix, skip in (("", frozenset()), ("heldout_", TUNED_ON)):
                print(f"-- {'every patient' if not skip else f'leaving out the {len(skip)} the level fix was tuned on'}")
                for name, r in compare(sys.argv[2], skip).items():
                    led.numbers_from({f"{prefix}{name}_{k}": v for k, v in r.items() if k != "ci95"})
                    led.number(f"{prefix}{name}_ci_lo", r["ci95"][0])
                    led.number(f"{prefix}{name}_ci_hi", r["ci95"][1])
    else:
        sys.exit(__doc__)
