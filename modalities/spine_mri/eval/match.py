"""Match every graded SPIDER disc to a TotalSpineSeg disc, and decide which graded discs are scored.

For a graded disc i of patient p: SPIDER's disc mask (label 200+i in the T2's mask) is compared with
every TotalSpineSeg disc label in the same volume; the one with the highest Dice wins if its Dice is
at least 0.3. Its level comes from TotalSpineSeg's label (labels.py).

Scope (one row per graded disc of the split, written to <work>/<split>/matches.csv):
  - patient has no T2                      -> out, "no T2"
  - IVD label 0                            -> out, "IVD label 0"
  - no SPIDER mask for it on the T2, even after prepare.py's same-study fill: the disc is outside
    the T2's field of view, so no T2 method can see it -> out, "not in the T2's field of view"
  - matched, level T12-L1 .. L5-S1         -> scored
  - matched, other level                   -> out, "matched outside T12-L1..L5-S1"
  - unmatched, nominal level in that range -> scored (counts as wrong for every method)
  - unmatched, nominal level outside       -> out, "unmatched, nominal level outside T12-L1..L5-S1"
The nominal level is SPIDER's bottom-up count (spider.NOMINAL); it decides scope only for discs
TotalSpineSeg did not find, and it is the reference for the level-mislabel rate.
Usage: python match.py <dev|test> [work_dir]"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from labels import NAMES, level_of  # noqa: E402
from prepare import WORK  # noqa: E402
from spider import guard_test, NOMINAL, SCORED_LEVELS, grades  # noqa: E402
from split import load as load_split  # noqa: E402

MIN_DICE = 0.3
DISC_IDS = sorted(k for k, v in NAMES.items() if v.startswith("disc_"))


def match_patient(mask, tss):
    """{ivd: (tss_label, dice)} for every SPIDER disc in the mask (best TotalSpineSeg disc, any Dice)."""
    out = {}
    tss_sizes = np.bincount(tss.ravel(), minlength=max(DISC_IDS) + 1)
    for v in np.unique(mask):
        v = int(v)
        if v < 200:
            continue
        sel = mask == v
        over = np.bincount(tss[sel], minlength=max(DISC_IDS) + 1)
        best, best_d = None, 0.0
        for d in DISC_IDS:
            if over[d]:
                dice = 2.0 * over[d] / (sel.sum() + tss_sizes[d])
                if dice > best_d:
                    best, best_d = d, dice
        out[v - 200] = (best, best_d)
    return out


def main(split, work=WORK):
    guard_test(split)
    sp = load_split()
    base = os.path.join(work, split)
    g = grades()
    rows = []
    for key, s in sorted(sp["patients"].items(), key=lambda kv: int(kv[0])):
        if s["split"] != split:
            continue
        p = int(key)
        gp = g[g.patient == p]
        m = None
        if s["t2"]:
            mask = np.asarray(nib.load(os.path.join(base, "mask", f"{p}.nii.gz")).dataobj).astype(np.int32)
            tss = np.asarray(nib.load(os.path.join(base, "tss", "step2_output", f"{p}.nii.gz")).dataobj).astype(np.int32)
            assert mask.shape == tss.shape, (p, mask.shape, tss.shape)
            m = match_patient(mask, np.clip(tss, 0, None))
        for _, r in gp.iterrows():
            i = int(r.ivd)
            nominal = NOMINAL.get(i, "")
            row = {"patient": p, "ivd": i, "fold": s["fold"], "vendor": s["vendor"], "stratum": s["stratum"],
                   "nominal_level": nominal, "tss_label": "", "tss_level": "", "dice": "", "matched": 0,
                   "in_scope": 0, "drop_reason": ""}
            if m is None:
                row["drop_reason"] = "no T2"
            elif i == 0:
                row["drop_reason"] = "IVD label 0"
            elif i not in m:
                row["drop_reason"] = "not in the T2's field of view"
            else:
                lab, dice = m.get(i, (None, 0.0))
                if lab is not None:
                    row.update(tss_label=lab, tss_level=level_of(NAMES[lab]), dice=round(dice, 4))
                if lab is not None and dice >= MIN_DICE:
                    row["matched"] = 1
                    if row["tss_level"] in SCORED_LEVELS:
                        row["in_scope"] = 1
                    else:
                        row["drop_reason"] = "matched outside T12-L1..L5-S1"
                elif nominal in SCORED_LEVELS:
                    row["in_scope"] = 1
                else:
                    row["drop_reason"] = "unmatched, nominal level outside T12-L1..L5-S1"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base, "matches.csv"), index=False)
    elig = df[~df.drop_reason.isin(["no T2", "not in the T2's field of view"]) & (df.ivd >= 1)]
    mt = elig[elig.matched == 1]
    dup = mt.groupby(["patient", "tss_label"]).size()
    print(f"{split}: {len(df)} graded rows, {df.patient.nunique()} patients; eligible {len(elig)}; "
          f"matched {len(mt)} ({len(mt) / max(len(elig), 1):.3f}); "
          f"level differs from SPIDER's count {(mt.tss_level != mt.nominal_level).sum()}; "
          f"in scope {df.in_scope.sum()}; TSS discs matched twice {(dup > 1).sum()}")
    print(df.drop_reason.replace("", "scored").value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or [WORK]))
