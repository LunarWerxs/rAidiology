"""Every SPIDER development patient's disc chain as the numbering step sees it, out of fold, so numbering rules can be
replayed and scored without reading a scan again.

Owner, 2026-11-10, on the next-run picker: decide what helps before paying for it. Disc numbering (outliner.outline)
runs on the disc points and level votes the net gives; reading them takes ~200 s on hydra-aws's CPU, replaying a rule
on them takes milliseconds. Each patient's row holds its points top to bottom (PIR mm), each point's vote and heat, the
nearest expert disc's SCT level and distance, and the expert discs' levels, so a rule is scored per expert disc: right
when a point that stands for it (its nearest expert disc, within FOUND_MM) is numbered its level.

  python outliner_chains.py dump [--workers K]   -> <work>/eval/spine_outliner/chains.json (models: outliner.OUT)
  python outliner_chains.py replay               -> disc-level accuracy from chains.json, today's rule beside the
                                                     candidate (CANDIDATE_ANCHOR), overall, not tuned on and by fold"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # the repository root
import outliner as O  # noqa: E402  floor-ok: the import needs the sys.path lines above it, as in outliner_vs_tss

# A disc point sits at the disc's posterior tip, 9-18 mm from the expert disc's centre (2026-11-10, 168 patients), so a
# point stands for the expert disc it is nearest to, when that is within FOUND_MM (adult lumbar discs sit 30-45 mm apart)
FOUND_MM = 25.0
PATH = os.path.join(O.V2, "chains.json")
LEVEL_OF = {int(v): k for k, v in enumerate(O.DISC_OF) if v}


def one(job):
    import torch
    torch.set_num_threads(job["threads"])
    p, f = job["patient"], job["fold"]
    a, aff, _ = O.to_pir(nib.load(os.path.join(O.SPIDER_WORK, "dev", "t2", f"{p}.nii.gz")))
    z = O.zooms(aff)
    seg, heat, level = O.probabilities(O.load(str(f)), O.scaled(O.Series(a=a, aff=aff)), aff)
    P, votes = O.find_discs(seg.argmax(0), heat, level, z)
    m = O.to_pir(nib.load(os.path.join(O.SPIDER_WORK, "dev", "mask", f"{p}.nii.gz")))[0].astype(np.int32)
    truth = []
    for i in sorted({int(v) - 200 for v in np.unique(m) if v > 200}):
        want = O.spider_disc_id(i)
        if want is not None:
            truth.append({"level": LEVEL_OF[int(want)], "mm": [float(c) for c in np.argwhere(m == 200 + i).mean(0) * z]})
    order = np.argsort(P[:, 1], kind="stable") if len(P) else np.zeros(0, int)
    pts = []
    for k in order:
        q = P[k] * z
        d = [float(np.linalg.norm(q - np.array(t["mm"]))) for t in truth]
        j = int(np.argmin(d)) if d else -1
        pts.append({"mm": [float(c) for c in q], "vote": float(votes[k]),
                    "heat": float(heat[tuple(np.rint(P[k]).astype(int))]),
                    "truth": truth[j]["level"] if j >= 0 else None, "truth_mm": d[j] if j >= 0 else None})
    return {"patient": str(p), "fold": int(f), "points": pts, "truth": truth}


def dump(workers):
    folds = O.split_patients()
    threads = max(1, (os.cpu_count() or 2) // workers)
    jobs = [{"patient": p, "fold": f, "threads": threads} for p, f in sorted(folds.items())]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs))
    os.makedirs(O.V2, exist_ok=True)
    with open(PATH, "w") as fh:
        json.dump(rows, fh)
    print(f"{len(rows)} patients' disc chains -> {PATH}")


def current_rule(pts, anchor=None):
    """outliner.outline's numbering: number_discs, then drop the point furthest from its number while it is over
    OFF_CHAIN, then keep levels 2-25. Returns {point index: level}. With `anchor`, the candidate below: the lowest disc
    left in the chain is numbered L5-S1 (25) when its own vote is `anchor` or more, and the chain moves with it."""
    idx = list(range(len(pts)))
    P = np.array([q["mm"] for q in pts]).reshape(-1, 3)
    v = np.array([q["vote"] for q in pts])
    if not len(P):
        return {}
    levels = O.number_discs(P, v)
    while len(P) > 2:
        off = np.abs(v - levels)
        w = int(off.argmax())
        if off[w] <= O.OFF_CHAIN:
            break
        P, v, idx = np.delete(P, w, 0), np.delete(v, w), idx[:w] + idx[w + 1:]
        levels = O.number_discs(P, v)
    if anchor is not None and v[-1] >= anchor:
        levels = levels + (25 - levels[-1])
    return {i: int(lv) for i, lv in zip(idx, levels, strict=True) if 2 <= lv <= 25}


# The candidate, fixed 2026-11-10 on run 3's chains (spine-outliner-v2c weights, 1eb64d1a numbering) BEFORE run 4's
# models existed, and tested once on run 4's chains: the lowest disc votes L5-S1 at 25.0 or more -> it is L5-S1. On run
# 3's chains it took the discs numbered right from 0.905 to 0.930 (all 168; 0.897 -> 0.927 on the 143 not tuned on),
# patients all right 140 -> 143, fixing 52, 85, 100, 107, 187, 241 and breaking 160, 186, 200, 212 (a point below
# L5-S1, 25 mm off, voting 25.2-25.6). It was the best of nine thresholds and is fragile (24.75: 0.902, 25.25: 0.923),
# so it ships into outliner.outline only if run 4's chains, which it never saw, gain at least 0.01 from it with no fold
# worse by more than 0.01.
# Tested once, 2026-11-10, on run 4's chains (spine-outliner-v2e weights, 168 patients, hydra-aws): FAILED, NOT SHIPPED.
# It gained overall (0.923 -> 0.944; 0.917 -> 0.942 on the 143 not tuned on; patients all right 146 -> 149) but fold 0
# fell 0.957 -> 0.926 (-0.031), past the 0.01 allowed; folds 1-4: +0.034, 0.000, +0.090, +0.008. Do not re-tune the
# threshold on run 4's chains and ship it: those chains are no longer unseen.
CANDIDATE_ANCHOR = 25.0


def score(rows, rule, skip=frozenset()):
    """(expert discs numbered right, expert discs, patients with every disc right, patients) under rule."""
    right = n = whole = pats = 0
    for r in rows:
        if r["patient"] in skip:
            continue
        got = rule(r["points"])
        ok_all = True
        for t in r["truth"]:
            n += 1
            mine = [i for i, q in enumerate(r["points"]) if i in got and q["truth"] == t["level"]
                    and q["truth_mm"] is not None and q["truth_mm"] <= FOUND_MM]
            ok = bool(mine) and any(got[i] == t["level"] for i in mine)
            right += ok
            ok_all &= bool(ok)
        whole += ok_all
        pats += 1
    return right, n, whole, pats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "replay"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 6))
    a = ap.parse_args()
    if a.mode == "dump":
        dump(a.workers)
        return
    with open(PATH) as fh:
        rows = json.load(fh)
    from outliner_vs_tss import TUNED_ON
    for name, rule in (("current", current_rule),
                       (f"anchor {CANDIDATE_ANCHOR}", lambda p: current_rule(p, CANDIDATE_ANCHOR))):
        for tag, skip in (("all", frozenset()), ("not tuned on", TUNED_ON)):
            right, n, whole, pats = score(rows, rule, skip)
            print(f"{name} rule, {tag}: {right}/{n} discs ({right / n:.3f}), {whole}/{pats} patients all right")
        folds = []
        for f in sorted({r["fold"] for r in rows}):
            right, n, _, _ = score([r for r in rows if r["fold"] == f], rule)
            folds.append(f"{right / n:.3f}")
        print(f"{name} rule by fold: {' '.join(folds)}")


if __name__ == "__main__":
    main()
