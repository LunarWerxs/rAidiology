"""Score EVERY LIDC lesion with the malignancy model of the fold that held its scan out (idea 25, step A).

nodule_malignancy.py trains and scores only the lesions that at least 3 radiologists rated with a median other than
3. LIDC's pathology table (tools/lidc_pathology.py) has patients whose lesions were all left out that way. This reads
lidc.nodules() whole (any number of readers, any median rating, exactly 3 included) and scores each lesion in a scan
with a held-out fold by that fold's saved net, work/eval/lung/malignancy-v<N>/fold<k>.pt. Scans of fold -1 (not in
LUNA16's folds, always training) are skipped. Nothing is trained and nodule_malignancy.py is not edited.

Writes <work>/eval/lung/malignancy-v<N>/oof_all.csv: oof.csv's columns (seriesuid, coordX, coordY, coordZ,
diameter_mm, label, p, fold, logit) plus `readers` (radiologists who marked the lesion) and `median` (their median
malignancy rating). label is 1 for a median above 3, 0 below 3, empty for exactly 3 (or no rating).
For the lesions that are also in oof.csv it prints how far oof_all's p is from oof.csv's (proof that the nets and
crops were loaded as training scored them; expect under 1e-4).

Usage: python tools/malignancy_all.py score --version N
       python tools/malignancy_all.py selftest
The card: a 1.0 GB lease (RAIDIOLOGY_MALIGNANCY_CARD_GB overrides it), 16 crops at a time.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "modalities", "ct", "eval")
sys.path.insert(0, ROOT)
sys.path.insert(0, EVAL)
import lidc  # noqa: E402
import nodule_fpr as nf  # noqa: E402
import nodule_malignancy as nm  # noqa: E402
from core import card, ledger  # noqa: E402
from nodule import scan_folds  # noqa: E402

SCORING_GB, SCORING_BATCH = 1.0, 16
COLS = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "label", "p", "fold", "logit", "readers", "median"]
TOL = 1e-4


def label_of(median, split=3.0):
    """1 above the split, 0 below it, nan when exactly on it or unrated."""
    m = pd.Series(median, dtype=float)
    return pd.Series(np.where(m.isna() | (m == split), np.nan, (m > split).astype(float)), index=m.index)


def lesion_frame(les, folds_of):
    """Every lesion of `les` (lidc.nodules()) in a scan of a held-out fold, in oof_all's columns (no scores yet);
    also the number of lesions skipped for being in a scan of fold -1."""
    fold = les.series_uid.map(folds_of).fillna(-1).astype(int).to_numpy()
    keep = fold >= 0
    les = les[keep]
    df = pd.DataFrame({
        "seriesuid": les.series_uid.to_numpy(), "coordX": les.x_mm.to_numpy(), "coordY": les.y_mm.to_numpy(),
        "coordZ": les.z_mm.to_numpy(), "diameter_mm": les.diameter_mm.to_numpy(),
        "label": label_of(les.malignancy_median).to_numpy(), "fold": fold[keep],
        "readers": les.n_readers.to_numpy(), "median": les.malignancy_median.to_numpy()})
    return df.reset_index(drop=True), int((~keep).sum())


def score_frame(df, nets, boxes, batch=SCORING_BATCH):
    """Add logit and p to `df`: each row scored by the net of its own fold (nets: {fold: net})."""
    df = df.copy()
    df["logit"] = np.float32(0)
    for k, g in df.groupby("fold"):
        df.loc[g.index, "logit"] = nm.logits_of(nets[int(k)], boxes, g.reset_index(drop=True), batch)
    df["p"] = nm.sigmoid64(df.logit)
    return df[COLS]


def compare_with_oof(df, oof):
    """(lesions matched, largest absolute difference of p) of `df` against oof.csv's frame, matched on scan and
    coordinates (rounded to 0.001 mm, as the csv keeps them)."""
    key = ["seriesuid", "cx", "cy", "cz"]

    def keyed(d):
        d = d.copy()
        for a, b in zip(("cx", "cy", "cz"), ("coordX", "coordY", "coordZ")):
            d[a] = d[b].round(3)
        return d[key + ["p"]]
    m = keyed(df).merge(keyed(oof), on=key, suffixes=("", "_oof"))
    if m.empty:
        return 0, float("nan")
    return len(m), float((m.p - m.p_oof).abs().max())


def limit_card():
    gb = float(os.environ.get("RAIDIOLOGY_MALIGNANCY_CARD_GB", SCORING_GB))
    card.lease(gb, "malignancy all")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())
    return gb


def score(a):
    out = nm.out_root(a.version)
    paths_k = {k: os.path.join(out, "fold%d.pt" % k) for k in range(nm.N_FOLDS)}
    absent = [p for p in paths_k.values() if not os.path.exists(p)]
    if absent:
        raise SystemExit("no fold model: " + ", ".join(absent))
    if os.environ.get("RAIDIOLOGY_DEVICE", "").strip() == "cpu":
        # the card is busy: score on the processor, in float32 where training's scoring used bfloat16 on the
        # card, so oof.csv's p differs by about 1e-3 for that reason alone (queue 92, 2026-10-05 04:55)
        nf.DEV = torch.device("cpu")
    card.require_card(nf.DEV, torch)
    gb = limit_card() if nf.DEV.type == "cuda" else float("nan")

    les = lidc.nodules()
    folds_of, _ = scan_folds(False)
    df, skipped = lesion_frame(les, folds_of)
    print("lesions: %d in LIDC-IDRI; %d in scans of a held-out fold (to score); %d skipped (scan not in LUNA16's folds)"
          % (len(les), len(df), skipped))
    boxes = {u: nm.lidc_box(u) for u in df.seriesuid.unique()}
    nets = {}
    for k, p in paths_k.items():
        ck = torch.load(p, map_location=nf.DEV, weights_only=False)
        net = nf.build(ck["cfg"])
        net.load_state_dict(ck["state"])
        nets[k] = net
    with ledger.run("nodule", "malignancy all v%d" % a.version) as led:
        led.settings_from({"version": a.version, "batch": SCORING_BATCH, "lease_gb": gb})
        res = score_frame(df, nets, boxes)
        if nf.DEV.type == "cuda":
            print("scoring's peak card: %.2f GB allocated, %.2f GB reserved (lease %.1f GB)"
                  % (torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30, gb))
        res.to_csv(os.path.join(out, "oof_all.csv"), index=False)
        print("wrote %s: %d lesions of %d patients' scans (%d scans); label 1: %d, 0: %d, empty: %d"
              % (os.path.join(out, "oof_all.csv"), len(res), les[les.series_uid.isin(res.seriesuid)].patient.nunique(),
                 res.seriesuid.nunique(), int((res.label == 1).sum()), int((res.label == 0).sum()),
                 int(res.label.isna().sum())))
        led.number("lesions", len(res), "count").number("label_empty", int(res.label.isna().sum()), "count")
        oof_path = os.path.join(out, "oof.csv")
        if os.path.exists(oof_path):
            n, diff = compare_with_oof(res, pd.read_csv(oof_path))
            print("against oof.csv: %d lesions matched, largest |p difference| %.2e (expect under %.0e)%s"
                  % (n, diff, TOL, "" if diff < TOL else "  <-- LOADING DIFFERS"))
            led.number("oof_matched", n, "count").number("oof_max_abs_p_diff", diff)
        else:
            print("no oof.csv to compare with")
    return 0


def selftest():
    # the label rule
    lab = label_of([4.0, 3.0, 2.0, 3.5, 2.5, np.nan])
    assert lab[[0, 3]].tolist() == [1.0, 1.0] and lab[[2, 4]].tolist() == [0.0, 0.0] and lab[[1, 5]].isna().all(), lab

    # the frame: fold -1 scans skipped, every other lesion kept whatever its readers and median
    les = pd.DataFrame({
        "series_uid": ["a", "a", "b", "c", "d"], "patient": ["p1", "p1", "p2", "p3", "p4"],
        "x_mm": [1.0, 2.0, 3.0, 4.0, 5.0], "y_mm": [0.0] * 5, "z_mm": [0.0] * 5,
        "diameter_mm": [5.0, 8.0, 12.0, 6.0, 9.0], "n_readers": [1, 4, 2, 3, 3],
        "malignancy_median": [3.0, 4.0, 2.0, 5.0, 1.0]})
    df, skipped = lesion_frame(les, {"a": 0, "b": 1, "c": 1})      # d is in no fold
    assert skipped == 1 and len(df) == 4 and "d" not in set(df.seriesuid), (skipped, df)
    assert df.label.isna().tolist() == [True, False, False, False] and df.readers.tolist() == [1, 4, 2, 3]
    assert df["median"].tolist() == [3.0, 4.0, 2.0, 5.0] and df.fold.tolist() == [0, 0, 1, 1]

    # scoring: each lesion by its own fold's net, never the other's
    torch.manual_seed(0)
    dev = nf.DEV
    nets = {k: nf.Classifier([4, 8], 2, 1).to(dev) for k in (0, 1)}
    real_crop = nf.crop_of
    nf.crop_of = lambda scan, p, size=16: np.random.default_rng(int(abs(p[0]) * 10)).integers(0, 255, (16,) * 3).astype(np.uint8)
    try:
        res = score_frame(df, nets, {u: None for u in df.seriesuid}, batch=3)
        assert list(res.columns) == COLS, list(res.columns)
        for k in (0, 1):
            g = df[df.fold == k].reset_index(drop=True)
            own = nm.logits_of(nets[k], {u: None for u in g.seriesuid}, g, 16)
            other = nm.logits_of(nets[1 - k], {u: None for u in g.seriesuid}, g, 16)
            got = res[res.fold == k].logit.to_numpy()
            assert np.allclose(got, own, atol=1e-5), (k, got, own)
            assert not np.allclose(got, other, atol=1e-5), "fold %d lesions scored by the other fold's net" % k
        assert np.allclose(res.p, 1 / (1 + np.exp(-res.logit.astype(float)))) and res.p.between(0, 1).all()
        # oof.csv comparison: matched rows only, and the difference shows
        oof = res[res.label.notna()].iloc[:3].copy()
        oof["p"] = oof.p + [0.0, 0.0, 0.25]
        n, diff = compare_with_oof(res, oof)
        assert n == 3 and abs(diff - 0.25) < 1e-9, (n, diff)
        assert res.to_csv(index=False).splitlines()[0] == ",".join(COLS)
    finally:
        nf.crop_of = real_crop
    print("selftest ok")
    return 0


def main():
    if sys.argv[1:2] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--version", type=int, required=True)
    return score(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
