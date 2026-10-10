"""Cancer or not, fine-tuned on LIDC's own pathology (idea 29; the bar is written in docs/efficiency/README.md).

LIDC-IDRI's pathology table (tools/lidc_pathology.py diagnoses()) gives 117 patients whose lesions we have scored
(86 cancer, 31 benign; diagnosis 2 or 3 is cancer, 1 benign). Malignancy net v2 was taught by radiologists' ratings and
reads those patients at AUROC 0.707 by its largest p, below the largest nodule's diameter alone (0.755). Wang et al. 2022
fine-tuned on the pathology in five folds and report 0.877. This fine-tunes v2's fold nets on the pathology, leak-free.

Reads: work/eval/lung/malignancy-v<base>/oof_all.csv (tools/malignancy_all.py: every lesion in a scan of a held-out fold)
and fold<k>.pt (v<base>'s fold nets), work/eval/lung/index.csv (scan -> patient), the pathology table, and the lesions'
48 mm windows (all read into memory once).
Writes: work/eval/lung/malignancy-path-v<N>/patients.csv (one row per patient: the labels, the folds, the largest
diameter and p, v2's soft maximum logit, the five seeds' patient logits and their average) and one ledger entry.
Fine-tuned weights are not saved.

Folds, leak-free by construction: a patient's folds are the `fold` values of its lesions' scans; its pathology fold is the
smallest. Fold k starts from a fresh copy of v2's fold-k net (which never saw fold k's scans), fine-tunes on the patients
whose pathology fold is not k and whose fold set does not contain k, and scores the patients whose pathology fold is k.
The read: each lesion's logit from the net, the patient's logit the log-sum-exp of its lesions' logits (a soft maximum),
binary cross-entropy against the pathology, one patient per step with every lesion of the patient (cubes jittered +-2 mm
and flipped or turned as training does). Fixed, never tuned: all layers train, AdamW lr 1e-4, weight decay 1e-4, 30
epochs, five seeds per fold whose patient logits are averaged.

Score: pooled out-of-fold AUROC on the 117 with a 2,000-resample patient bootstrap (seed 0), paired on the same resamples
against the largest diameter alone and against v2's largest p. KEPT when at least 0.03 above the diameter and the paired
2.5th percentile is above 0; then against Wang et al. 2022 (0.877): BEATEN, LEVEL or BEHIND.

Usage: python tools/malignancy_path.py run [--version N] [--base 2] [--epochs 30] [--seeds 5] [--lr 1e-4]
       python tools/malignancy_path.py selftest
RAIDIOLOGY_DEVICE=cpu runs on the processor in float32. On the card: a lease of RAIDIOLOGY_MALIGNANCY_CARD_GB (2.5; 1.0
ran out of memory on 2026-10-05: a patient's lesions go through the net in one step, and one patient has 15).
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "modalities", "ct", "eval")
sys.path.insert(0, ROOT)
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nodule_fpr as nf  # noqa: E402
import nodule_malignancy as nm  # noqa: E402
import lidc_pathology as lpath  # noqa: E402
from core import card, ledger, paths  # noqa: E402

LR, WEIGHT_DECAY, EPOCHS, SEEDS = 1e-4, 1e-4, 30, 5
SHIFT_MM = nm.CFG["shift_mm"]
BOOT, MIN_GAIN, WANG = 2000, 0.03, 0.877
LEASE_GB = 2.5
STEP_FOLDS = nm.N_FOLDS


# ---- patients and folds ---------------------------------------------------------------------------------

def patient_table(oof, idx, diag):
    """(lesions, patients). lesions: oof_all's rows of the patients with a diagnosis in 1, 2, 3, plus patient.
    patients: one row per patient in patient order: patient, y, diagnosis, method, folds (set), path_fold (smallest),
    lesions, diameter (largest, mm), v2_p (largest p), plus `lesion_keys` (the lesions' (seriesuid, x, y, z))."""
    les = oof.merge(idx[["seriesuid", "patient"]], on="seriesuid", how="left")
    d = diag[diag.diagnosis.isin([1, 2, 3])].drop_duplicates("patient")
    les = les.merge(d, on="patient", how="inner").reset_index(drop=True)
    rows = []
    for pat, g in les.groupby("patient", sort=True):
        folds = set(int(f) for f in g.fold)
        rows.append({"patient": pat, "y": int(g.diagnosis.iloc[0] >= 2), "diagnosis": int(g.diagnosis.iloc[0]),
                     "method": g.method.iloc[0], "folds": folds, "path_fold": min(folds), "lesions": len(g),
                     "diameter": float(g.diameter_mm.max()), "v2_p": float(g.p.max()),
                     "lesion_keys": [(r.seriesuid, float(r.coordX), float(r.coordY), float(r.coordZ)) for r in g.itertuples()]})
    return les, pd.DataFrame(rows)


def fold_plan(folds_of, n_folds=STEP_FOLDS):
    """{k: (train, test)} of patient names from {patient: set of folds}. Test: pathology fold (the smallest) is k.
    Train: k is in no fold of the patient (so a patient is never a training patient of a fold it has a scan in)."""
    plan = {}
    for k in range(n_folds):
        test = sorted(p for p, f in folds_of.items() if min(f) == k)
        train = sorted(p for p, f in folds_of.items() if k not in f)
        assert not set(test) & set(train), "fold %d: a patient is both trained and tested" % k
        plan[k] = (train, test)
    return plan


def pool(logits):
    """Soft maximum: log(sum(exp(logits))), stably."""
    z = np.asarray(logits, np.float64)
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


# ---- the score ------------------------------------------------------------------------------------------

def paired_bootstrap(y, scores, n=BOOT, seed=0):
    """AUROC of each read in `scores` ({name: array}) on the same `n` patient resamples: {name: array of n}."""
    y = np.asarray(y)
    idx = np.random.default_rng(seed).integers(0, len(y), (n, len(y)))
    return {k: np.array([lpath.auroc(y[i], np.asarray(s)[i]) for i in idx]) for k, s in scores.items()}


def interval(v):
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def read_against(y, ours, others, n=BOOT):
    """AUROC of `ours` with its interval, and for each name in `others` the paired difference (ours minus other) with
    its interval, all on the same resamples. Returns {'auc', 'ci', 'diff': {name: (point, lo, hi)}, 'other': {name: auc}}."""
    b = paired_bootstrap(y, {"ours": ours, **others}, n)
    res = {"auc": lpath.auroc(y, ours), "ci": interval(b["ours"]), "diff": {}, "other": {}}
    for k, s in others.items():
        res["other"][k] = lpath.auroc(y, s)
        res["diff"][k] = (res["auc"] - res["other"][k], *interval(b["ours"] - b[k]))
    return res


def kept(point, low):
    return point >= MIN_GAIN and low > 0


def wang_word(lo, hi):
    return "BEATEN" if lo > WANG else "BEHIND" if hi < WANG else "LEVEL"


def verdict_lines(point, low, lo, hi):
    return ["Verdict (idea 29, at least 0.03 above the diameter alone, paired low end above 0): "
            + ("KEPT" if kept(point, low) else "NOT KEPT"),
            "Against Wang et al. 2022 (%.3f): %s" % (WANG, wang_word(lo, hi))]


# ---- fine-tuning ----------------------------------------------------------------------------------------

def to_train(net, cfg):
    """Training mode. The norm layers are GroupNorm in v2, which is fine for a batch of one lesion; with batch
    norm the norm layers stay in eval mode (their running statistics are not moved by a handful of lesions)."""
    net.train()
    if cfg.get("norm", "group") == "batch":
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def cubes_of(boxes, keys, rng):
    """(n, 1, D, H, W) uint8: each lesion's cube cut with a uniform jitter of +-SHIFT_MM per axis, then flipped/turned."""
    out = []
    for key in keys:
        jitter = rng.uniform(-SHIFT_MM, SHIFT_MM, 3)
        out.append(np.ascontiguousarray(nf.Sampler.augment(None, nm.ram_cube(boxes[key[0]], key, jitter), rng)))
    return np.stack(out)


def finetune(net, cfg, items, boxes, epochs, lr, rng):
    """Fine-tune `net` (all layers) on items = [(keys, y)] one patient per step, an epoch visiting every patient once
    in a shuffled order. Returns the mean patient loss of each epoch."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    cuda = nf.DEV.type == "cuda"
    losses = []
    for _ in range(epochs):
        tot = 0.0
        for i in rng.permutation(len(items)):
            keys, y = items[int(i)]
            x = nf.net_input(torch.from_numpy(cubes_of(boxes, keys, rng)).to(nf.DEV), net.c_in)
            to_train(net, cfg)
            with torch.autocast(nf.DEV.type, dtype=torch.bfloat16, enabled=cuda):
                lg = net(x).float()
            loss = F.binary_cross_entropy_with_logits(torch.logsumexp(lg, 0), torch.tensor(float(y), device=nf.DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        losses.append(tot / max(len(items), 1))
    return losses


def patient_logits(net, boxes, pts):
    """{patient: logsumexp of its lesions' logits}, no augmentation, eval mode (nm.logits_of)."""
    out = {}
    for _, r in pts.iterrows():
        frame = pd.DataFrame(r.lesion_keys, columns=["seriesuid", "coordX", "coordY", "coordZ"])
        out[r.patient] = pool(nm.logits_of(net, boxes, frame))
    return out


def run_fold(k, ck, pts, plan, boxes, seeds, epochs, lr):
    """Fold k: (v2's unchanged read of fold k's patients, [one {patient: logit} per seed])."""
    train_p, test_p = plan[k]
    by = pts.set_index("patient")
    items = [(by.loc[p, "lesion_keys"], int(by.loc[p, "y"])) for p in train_p]
    test = by.loc[test_p].reset_index()

    def fresh():
        net = nf.build(ck["cfg"])
        net.load_state_dict(ck["state"])
        return net

    base = patient_logits(fresh(), boxes, test)
    per_seed = []
    for s in range(seeds):
        torch.manual_seed(s)
        net = fresh()
        losses = finetune(net, ck["cfg"], items, boxes, epochs, lr, np.random.default_rng([s, k]))
        per_seed.append(patient_logits(net, boxes, test))
        print("  fold %d seed %d: training loss %.3f -> %.3f" % (k, s, losses[0], losses[-1]), flush=True)
    return base, per_seed


# ---- the run --------------------------------------------------------------------------------------------

def limit_card():
    gb = float(os.environ.get("RAIDIOLOGY_MALIGNANCY_CARD_GB", LEASE_GB))
    card.lease(gb, "malignancy pathology")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())
    return gb


def count(pts, names):
    q = pts.set_index("patient").loc[names]
    return "%d (%d cancer / %d benign)" % (len(q), int(q.y.sum()), int((q.y == 0).sum()))


def report(pts, logit_cols, base_logit, final, led, folds_secs):
    """Print the score lines and record the numbers; `final` is the averaged patient logit."""
    y = pts.y.to_numpy()
    others = {"diameter": pts.diameter.to_numpy(), "v2_p": pts.v2_p.to_numpy()}
    res = read_against(y, final, others)
    lo, hi = res["ci"]
    print("All known: %d patients (%d cancer, %d benign)" % (len(pts), int(y.sum()), int((y == 0).sum())))
    print("Pathology-fine-tuned read (all known): AUROC %.3f (%.3f to %.3f)" % (res["auc"], lo, hi))
    names = {"diameter": "the largest diameter", "v2_p": "v2's largest p"}
    led.number("auroc", res["auc"]).number("auroc_low", lo).number("auroc_high", hi)
    for k in others:
        d, dlo, dhi = res["diff"][k]
        print("  %s: AUROC %.3f; paired difference %+.3f (%.3f to %.3f)" % (names[k], res["other"][k], d, dlo, dhi))
        led.number("%s_auroc" % k, res["other"][k]).number("diff_vs_%s" % k, d)
        led.number("diff_vs_%s_low" % k, dlo).number("diff_vs_%s_high" % k, dhi)
    point, low, _ = res["diff"]["diameter"]
    for line in verdict_lines(point, low, lo, hi):
        print(line)
    led.number("kept", int(kept(point, low)), "flag")

    ba = lpath.auroc(y, base_logit)
    print("v2, soft maximum, no fine-tuning: AUROC %.3f" % ba)
    led.number("v2_softmax_auroc", ba)

    m = pts.method.isin([2, 3, 4]).to_numpy()
    if m.sum():
        t = read_against(y[m], final[m], {"diameter": others["diameter"][m], "v2_p": others["v2_p"][m]})
        print("Tissue or progression (%d patients, %d benign; no verdict): AUROC %.3f (%.3f to %.3f); "
              "paired vs diameter %+.3f (%.3f to %.3f), vs v2's largest p %+.3f (%.3f to %.3f)"
              % (m.sum(), int((y[m] == 0).sum()), t["auc"], *t["ci"], *t["diff"]["diameter"], *t["diff"]["v2_p"]))
        led.number("tissue_auroc", t["auc"]).number("tissue_auroc_low", t["ci"][0]).number("tissue_auroc_high", t["ci"][1])
        led.number("tissue_diff_vs_diameter", t["diff"]["diameter"][0]).number("tissue_diff_vs_diameter_low", t["diff"]["diameter"][1])
        led.number("tissue_diff_vs_v2_p", t["diff"]["v2_p"][0]).number("tissue_diff_vs_v2_p_low", t["diff"]["v2_p"][1])
        led.number("tissue_patients", int(m.sum()), "count")
    for k in sorted(pts.path_fold.unique()):
        f = (pts.path_fold == k).to_numpy()
        a = lpath.auroc(y[f], final[f])
        print("Fold %d: %d patients, AUROC %.3f (v2 soft maximum %.3f), %.0f s"
              % (k, f.sum(), a, lpath.auroc(y[f], base_logit[f]), folds_secs.get(int(k), float("nan"))))
        led.number("fold%d_auroc" % k, a).number("fold%d_seconds" % k, folds_secs.get(int(k), float("nan")), "s")
    for i, c in enumerate(logit_cols):
        a = lpath.auroc(y, pts[c].to_numpy())
        print("Seed %d pooled AUROC %.3f" % (i, a))
        led.number("seed%d_auroc" % i, a)


def run(a):
    out = os.path.join(paths.WORK, "eval", "lung", "malignancy-path-v%d" % a.version)
    base = nm.out_root(a.base)
    ck_paths = {k: os.path.join(base, "fold%d.pt" % k) for k in range(nm.N_FOLDS)}
    absent = [p for p in ck_paths.values() if not os.path.exists(p)]
    if absent:
        raise SystemExit("no fold model: " + ", ".join(absent))
    if os.environ.get("RAIDIOLOGY_DEVICE", "").strip() == "cpu":
        nf.DEV = torch.device("cpu")
    card.require_card(nf.DEV, torch)
    gb = limit_card() if nf.DEV.type == "cuda" else float("nan")

    oof = pd.read_csv(os.path.join(base, "oof_all.csv"))
    idx = pd.read_csv(os.path.join(paths.WORK, "eval", "lung", "index.csv"), dtype={"patient": str})
    les, pts = patient_table(oof, idx, lpath.diagnoses())
    print("patients: %d (%d cancer, %d benign), %d lesions" % (len(pts), int(pts.y.sum()), int((pts.y == 0).sum()), len(les)))
    folds_of = dict(zip(pts.patient, pts.folds))
    plan = fold_plan(folds_of)
    for k, (tr, te) in plan.items():
        print("fold %d: train %s; test %s" % (k, count(pts, tr), count(pts, te)))
    assert sorted(p for _, te in plan.values() for p in te) == sorted(pts.patient), "a patient is tested in no fold or two"

    keys = [k for ks in pts.lesion_keys for k in ks]
    boxes = {u: nm.lidc_box(u) for u in dict.fromkeys(k[0] for k in keys)}
    t = time.time()
    n = nm.fill_ram(boxes, keys)
    print("%d lesion windows read into memory in %.0f s" % (n, time.time() - t), flush=True)

    base_logit, seed_logit, secs = {}, [dict() for _ in range(a.seeds)], {}
    for k in range(nm.N_FOLDS):
        t = time.time()
        ck = torch.load(ck_paths[k], map_location=nf.DEV, weights_only=False)
        b, per_seed = run_fold(k, ck, pts, plan, boxes, a.seeds, a.epochs, a.lr)
        base_logit.update(b)
        for s, d in enumerate(per_seed):
            seed_logit[s].update(d)
        secs[k] = time.time() - t
        print("fold %d done in %.0f s" % (k, secs[k]), flush=True)

    pts = pts.drop(columns="lesion_keys")
    cols = []
    for s in range(a.seeds):
        cols.append("logit_s%d" % s)
        pts[cols[-1]] = pts.patient.map(seed_logit[s])
    pts["logit"] = pts[cols].mean(axis=1)
    pts["v2_softmax_logit"] = pts.patient.map(base_logit)
    pts["folds"] = pts.folds.map(lambda f: ";".join(str(x) for x in sorted(f)))
    os.makedirs(out, exist_ok=True)
    keep = ["patient", "y", "diagnosis", "method", "folds", "path_fold", "lesions", "diameter", "v2_p",
            "v2_softmax_logit"] + cols + ["logit"]
    pts[keep].to_csv(os.path.join(out, "patients.csv"), index=False)
    print("wrote %s" % os.path.join(out, "patients.csv"))

    with ledger.run("nodule", "malignancy pathology v%d" % a.version) as led:
        led.settings_from({"version": a.version, "base": a.base, "epochs": a.epochs, "seeds": a.seeds, "lr": a.lr,
                           "weight_decay": WEIGHT_DECAY, "shift_mm": SHIFT_MM, "boot": BOOT, "min_gain": MIN_GAIN,
                           "wang": WANG, "device": nf.DEV.type, "lease_gb": gb, "patients": len(pts)})
        report(pts, cols, pts.v2_softmax_logit.to_numpy(), pts.logit.to_numpy(), led, secs)
    return 0


# ---- selftest -------------------------------------------------------------------------------------------

def selftest():
    """No data, no card: the fold rule, the pooling, the verdicts, the paired bootstrap and a toy fine-tune."""
    # (1) the fold rule: patient c has scans in folds 1 and 3
    folds_of = {"a": {0}, "b": {1}, "c": {1, 3}, "d": {2}, "e": {3}, "f": {4}, "g": {0}, "h": {1}}
    plan = fold_plan(folds_of)
    for k, (tr, te) in plan.items():
        assert not set(tr) & set(te), (k, tr, te)
        assert all(k not in folds_of[p] for p in tr), (k, tr)
        assert all(min(folds_of[p]) == k for p in te)
    assert "c" in plan[1][1] and "c" not in plan[3][1], plan
    assert "c" not in plan[1][0] and "c" not in plan[3][0] and "c" in plan[0][0] and "c" in plan[2][0]
    assert sorted(p for _, te in plan.values() for p in te) == sorted(folds_of)

    # patient_table: smallest fold, y from diagnosis, unknown and missing diagnoses dropped
    oof = pd.DataFrame({"seriesuid": ["s1", "s2", "s3", "s4", "s5"], "coordX": [1.0, 2.0, 3.0, 4.0, 5.0],
                        "coordY": 0.0, "coordZ": 0.0, "diameter_mm": [5.0, 9.0, 7.0, 6.0, 8.0],
                        "p": [0.2, 0.7, 0.4, 0.5, 0.9], "fold": [3, 1, 2, 0, 0]})
    idx = pd.DataFrame({"seriesuid": ["s1", "s2", "s3", "s4", "s5"], "patient": ["P1", "P1", "P2", "P3", "P4"]})
    diag = pd.DataFrame({"patient": ["P1", "P2", "P3"], "diagnosis": [2.0, 1.0, 0.0], "method": [3.0, 2.0, 2.0]})
    les, pts = patient_table(oof, idx, diag)
    assert pts.patient.tolist() == ["P1", "P2"] and pts.y.tolist() == [1, 0], pts
    assert pts.folds.tolist() == [{1, 3}, {2}] and pts.path_fold.tolist() == [1, 2]
    assert pts.diameter.tolist() == [9.0, 7.0] and pts.v2_p.tolist() == [0.7, 0.4] and pts.lesions.tolist() == [2, 1]
    assert pts.lesion_keys[0][0] == ("s1", 1.0, 0.0, 0.0)

    # (2) pooling
    assert abs(pool([0.0, 0.0]) - np.log(2)) < 1e-12
    assert abs(pool([1.0, 2.0, 3.0]) - np.log(np.exp(1) + np.exp(2) + np.exp(3))) < 1e-12
    assert abs(pool([800.0, 800.0]) - (800 + np.log(2))) < 1e-9 and abs(pool([-5.0]) + 5.0) < 1e-12

    # (3) the verdicts at the edges
    assert kept(0.03, 0.001) and not kept(0.0299, 0.5) and not kept(0.1, 0.0) and not kept(0.1, -0.01)
    assert wang_word(0.8771, 0.95) == "BEATEN" and wang_word(0.877, 0.95) == "LEVEL"
    assert wang_word(0.8, 0.877) == "LEVEL" and wang_word(0.8, 0.8769) == "BEHIND"
    v = verdict_lines(0.05, 0.01, 0.9, 0.95)
    assert v[0].endswith(": KEPT") and v[1].endswith(": BEATEN") and "(0.877)" in v[1], v
    v = verdict_lines(0.05, -0.01, 0.7, 0.8)
    assert v[0].endswith(": NOT KEPT") and v[1].endswith(": BEHIND"), v

    # (4) the paired bootstrap: the same resamples for both reads
    rng = np.random.default_rng(1)
    y = (np.arange(60) % 2).astype(int)
    s = y + rng.normal(0, 1, 60)
    r = read_against(y, s, {"same": s.copy(), "noise": rng.normal(size=60)}, n=200)
    assert r["diff"]["same"] == (0.0, 0.0, 0.0), r["diff"]["same"]
    assert r["diff"]["noise"][1] < r["diff"]["noise"][2] and r["auc"] > 0.6
    b = paired_bootstrap(y, {"a": s, "b": s}, n=50)
    assert np.array_equal(b["a"], b["b"])

    # (5) a short fine-tune on a separable toy: two patient classes whose cubes differ in brightness
    real = (nf.DEV, nm.ram_cube)
    nf.DEV = torch.device("cpu")
    nm.ram_cube = lambda scan, key, jitter: np.full((1, 16, 16, 16), 220 if key[0].startswith("hi") else 30, np.uint8)
    try:
        torch.manual_seed(0)
        net = nf.Classifier([4, 8], 2, 1).to(nf.DEV)
        boxes = {"hi%d" % i: None for i in range(6)} | {"lo%d" % i: None for i in range(6)}
        items = [([("hi%d" % i, 0.0, 0.0, 0.0), ("hi%d" % i, 1.0, 0.0, 0.0)], 1) for i in range(6)]
        items += [([("lo%d" % i, 0.0, 0.0, 0.0)], 0) for i in range(6)]
        losses = finetune(net, {"norm": "group"}, items, boxes, 12, 3e-3, np.random.default_rng([0, 0]))
        assert len(losses) == 12 and np.mean(losses[-3:]) < np.mean(losses[:3]), losses
        # the cubes: one per lesion, uint8, 5-D after stacking
        c = cubes_of(boxes, [("hi0", 0.0, 0.0, 0.0), ("hi1", 0.0, 0.0, 0.0)], np.random.default_rng(0))
        assert c.shape == (2, 1, 16, 16, 16) and c.dtype == np.uint8
    finally:
        nf.DEV, nm.ram_cube = real
    print("selftest ok")
    return 0


def main():
    if sys.argv[1:2] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--version", type=int, default=1)
    r.add_argument("--base", type=int, default=2, help="the malignancy version whose fold nets are fine-tuned")
    r.add_argument("--epochs", type=int, default=EPOCHS)
    r.add_argument("--seeds", type=int, default=SEEDS)
    r.add_argument("--lr", type=float, default=LR)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
