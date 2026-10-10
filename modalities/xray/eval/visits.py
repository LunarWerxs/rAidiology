"""X-ray: read a visit, not a picture. GRAZPEDWRI-DX children's wrists, development out-of-fold.

A doctor (and BoneView, the sold product on the scores-to-beat list) judges a visit: the front and
side views of one wrist on one day. The detector scores one picture at a time, and a picture whose
view hides the break counts as "no fracture" in the per-picture truth. Here the same out-of-fold
scores are read both ways. A visit is one patient's `study_number` on one side (`laterality`);
its truth is "fractured" when any of its pictures shows the break, and its score combines its
pictures' scores: `max`, `mean`, or `any` (1 minus the product of 1 - p, as if the views were
independent looks).

For each reading: AUROC with score.py's patient bootstrap (the same 2,000 resamples of patients for
every reading, so their intervals are comparable), and the operating points the scores-to-beat
lines use: breaks caught when clearing 98 of 100 healthy (BoneView's clearing), when clearing 94
(ours on the test), healthy cleared when catching 93 (BoneView's catching), and both at the run's
own fitted threshold. The cut-offs for those points are read off the same scores, so they are a
curve's points on development, not a test result. On all visits and on first visits
(`initial_exam`, Lindsey's and Gasmi's setting). Recorded in the ledger.

Usage: python visits.py [run ...]   (default: detector-v102 detector-v103, the 25-epoch recipe at two seeds)"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [HERE, ROOT]
import datasets  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every eval script here
import score  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every eval script here
from core import ledger, paths  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every eval script here

OUT = os.path.join(paths.WORK, "eval", "xray")
COMBINE = {"max": lambda p: p.max(), "mean": lambda p: p.mean(), "any": lambda p: 1 - np.prod(1 - p)}


def frame(run):
    """Development out-of-fold GRAZPEDWRI-DX pictures of a run, with visit, view and first-visit flag."""
    oof = pd.read_csv(os.path.join(OUT, run, "model", "oof_dev.csv"))
    oof = oof[oof.dataset == "grazpedwri"]
    meta = pd.read_csv(os.path.join(datasets.SETS["grazpedwri"]["root"], "dataset.csv"), encoding="utf-8-sig",
                       dtype=str)[["filestem", "study_number", "laterality", "projection", "initial_exam"]]
    df = oof.merge(meta, left_on="image", right_on="filestem", how="left", validate="one_to_one")
    if df.study_number.isna().any():
        raise SystemExit(f"{run}: {int(df.study_number.isna().sum())} pictures not in dataset.csv")
    df["visit"] = df.patient.astype(str) + "/" + df.study_number + "/" + df.laterality
    df["first"] = df.initial_exam == "1"
    return df


def visits(df, how):
    g = df.groupby("visit")
    return pd.DataFrame({"patient": g.patient.first().astype(str), "truth": g.truth.max(),
                         "score": g.score.agg(COMBINE[how]), "first": g["first"].max(), "pictures": g.size()})


def auroc_boot(t, s, patients, P):
    """AUROC and its 95% interval over score.py's patient resamples of the patient list P."""
    pos_i = {p: i for i, p in enumerate(P)}
    unit = np.array([pos_i[p] for p in patients])
    vals, bins = np.unique(s, return_inverse=True)
    from scipy import sparse
    Sp = sparse.csr_matrix((np.ones((t == 1).sum(), np.float32), (unit[t == 1], bins[t == 1])), shape=(len(P), len(vals)))
    Sn = sparse.csr_matrix((np.ones((t == 0).sum(), np.float32), (unit[t == 0], bins[t == 0])), shape=(len(P), len(vals)))
    full = float(score.auroc_from_bins(np.asarray(Sp.sum(0)).ravel(), np.asarray(Sn.sum(0)).ravel()))
    M = score.multiplicity(len(P))
    boots = np.concatenate([score.auroc_from_bins(np.asarray(Sp.T.dot(M[i:i + score.CHUNK].T).T),
                                                  np.asarray(Sn.T.dot(M[i:i + score.CHUNK].T).T))
                            for i in range(0, score.BOOT, score.CHUNK)])
    lo, hi, _ = score.interval(boots)
    return full, lo, hi


def caught_at_cleared(t, s, cleared):
    neg = np.sort(s[t == 0])
    cut = neg[int(np.ceil(cleared * len(neg))) - 1]
    return float((s[t == 1] > cut).mean())


def cleared_at_caught(t, s, caught):
    pos = np.sort(s[t == 1])
    cut = pos[int(np.floor((1 - caught) * len(pos)))]
    return float((s[t == 0] < cut).mean())


def read(t, s, patients, P, thr):
    a, lo, hi = auroc_boot(t, s, patients, P)
    return {"n": len(t), "auroc": a, "auroc_low": lo, "auroc_high": hi,
            "caught_at_cleared_98": caught_at_cleared(t, s, 0.98), "caught_at_cleared_94": caught_at_cleared(t, s, 0.94),
            "cleared_at_caught_93": cleared_at_caught(t, s, 0.93),
            "caught_at_threshold": float((s[t == 1] >= thr).mean()), "cleared_at_threshold": float((s[t == 0] < thr).mean())}


def main(runs):
    for run in runs:
        t0 = time.time()
        df = frame(run)
        thr = pd.read_json(os.path.join(OUT, run, "model", "meta.json"), typ="series")["thresholds"]["grazpedwri"]
        print(f"{run}: GRAZPEDWRI-DX development out-of-fold, {len(df)} pictures, {df.visit.nunique()} visits "
              f"of {df.patient.nunique()} patients; the run's fitted threshold {thr:.4f}")
        print("  %-22s %6s  %-22s %8s %8s %8s %13s" % ("reading", "n", "AUROC (95% CI)", "c@cl98", "c@cl94", "cl@c93", "c/cl at thr"))
        numbers = {}
        for subset in ("all", "first"):
            d = df if subset == "all" else df[df["first"]]
            P = sorted(set(d.patient.astype(str)))
            rows = [("pictures", d.truth.to_numpy(), d.score.to_numpy(), d.patient.astype(str).to_numpy())]
            for how in COMBINE:
                v = visits(d, how)
                rows.append((f"visits, {how}", v.truth.to_numpy(), v.score.to_numpy(), v.patient.to_numpy()))
            for name, t, s, pts in rows:
                r = read(t, s, pts, P, thr)
                print("  %-22s %6d  %.4f (%.4f to %.4f) %8.1f %8.1f %8.1f   %4.1f / %4.1f" % (
                    f"{subset}: {name}", r["n"], r["auroc"], r["auroc_low"], r["auroc_high"], 100 * r["caught_at_cleared_98"],
                    100 * r["caught_at_cleared_94"], 100 * r["cleared_at_caught_93"], 100 * r["caught_at_threshold"],
                    100 * r["cleared_at_threshold"]))
                key = f"{subset}.{name.replace('visits, ', 'visit_').replace(' ', '_')}"
                numbers.update({f"{key}.{k}": (v, "") for k, v in r.items()})
        ledger.write("experiment", f"xray visits: pictures read as visits ({run})", time.time() - t0, numbers,
                     settings={"run": run, "combine": ",".join(COMBINE)}, note="modalities/xray/eval/visits.py")


if __name__ == "__main__":
    main(sys.argv[1:] or ["detector-v102", "detector-v103"])
