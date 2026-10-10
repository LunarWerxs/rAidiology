"""Read the spine X-ray corners again with the saved close-look nets: no training, only a second pass and more looks.

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646 (LICENSING.md, BUU-LSPINE row). Plan and bars:
docs/efficiency/README.md, "Corners v2, read again".

Why. corners_refine.py's net was trained on crops centred on v1's misses (11.6 px) and nudged by up to 4 px. Its own
answers miss by less, so a crop centred on the net's answer sits well inside what it knows (a second pass); and the
net's answer moves a little with the crop, so the answers from crops 4 px left, right, up and down are averaged
(five looks). Both are measured here, in original film pixels.

How. Start points are the out-of-fold corners of corners-v<from> (default 2). Each pass, every patient's 22 corners are
cut as CROP x CROP crops from the original jpg (corners_refine.crop) centred on the point plus each look offset, read
by the net of THAT patient's own fold (corners-v2/LA-fold<k>.pt), turned into film pixels (crop position + the crop's
top-left offset), and averaged over the looks; the pass's output is the next pass's centre. Looks: one (0, 0), or five
(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4). Crops go through the net in batches of 256, bfloat16 on the card like
corners_refine.predict. corners_refine's window cache is not read or written. Writes
<WORK>/eval/spine_xray/corners-v<out>/LA-fold<k>.npz with the keys corners_refine writes (v1 holds the from-version's
points here), so corners.load_oof and slip_learned.py --corners-version <out> read it unchanged. Out-version 1 and 2
are refused.

The bar (corners): KEPT when the paired patient-bootstrap 95% interval of the per-patient mean miss, new minus
the from-version's, lies wholly below 0. The slips bar is slip_learned.py's, run on the new folder.

Usage: python corners_reread.py --passes 2 --looks 5 --out-version 3 [--from-version 2] [--against N] [--folds all]
       (--against N judges the re-read against version N's saved points instead of the from-version's: the miss,
       the bootstrap and the sanity line; the ids and truth of the two versions must match.)
       python corners_reread.py --selftest      (no data files, no net)
"""
import argparse
import json
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import ledger  # noqa: E402
import buu  # noqa: E402
import corners  # noqa: E402
import corners_refine as cr  # noqa: E402

DEV = corners.DEV
VIEW = cr.VIEW
BATCH = 256
CHUNK = 8                       # films held open at once; their crops are batched together
LOOKS_1 = [(0, 0)]
LOOKS_5 = [(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4)]


def look_offsets(looks):
    if looks == 1:
        return LOOKS_1
    if looks == 5:
        return LOOKS_5
    raise SystemExit("--looks: 1 or 5")


def reread(read_batch, imgs, pts, passes, offsets):
    """imgs: n films (H, W) uint8; pts (n, 22, 2) film pixels. read_batch(crops (B, CROP, CROP) uint8, corner (B,))
    -> (B, 2) in crop coordinates. Returns (final points, pass 1's centre-look answers), both (n, 22, 2) film pixels."""
    n = len(imgs)
    cur = np.asarray(pts, np.float64).copy()
    first = None
    for p in range(passes):
        patches, tops, who = [], [], []
        for a in range(n):
            for c in range(cr.N_CORNERS):
                for dx, dy in offsets:
                    patch, top = cr.crop(imgs[a], cur[a, c, 0] + dx, cur[a, c, 1] + dy, cr.CROP)
                    patches.append(patch)
                    tops.append(top)
                    who.append((a, c))
        crops, tops = np.stack(patches), np.asarray(tops, np.float64)
        corner = np.array([c for _, c in who])
        ans = np.concatenate([read_batch(crops[s:s + BATCH], corner[s:s + BATCH])
                              for s in range(0, len(crops), BATCH)]).astype(np.float64)
        film = (ans + tops).reshape(n, cr.N_CORNERS, len(offsets), 2)      # crop position + crop top-left
        if p == 0:
            first = film[:, :, 0]
        cur = film.mean(axis=2)
    return cur, first


def net_reader(net, mean, sd):
    def read(crops, corner):
        x = (torch.from_numpy(crops).to(DEV).float().div_(255.0).unsqueeze(1) - mean) / sd
        cc = torch.from_numpy(corner).to(DEV)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
                logits = net(cr.net_input(x, cc))
            return corners.soft_argmax(logits)[0][:, 0].cpu().numpy()
    return read


def load_net(fold):
    z = torch.load(os.path.join(cr.out_dir(2), "%s-fold%d.pt" % (VIEW, fold)), map_location=DEV, weights_only=False)
    net = cr.make_net(z["settings"]["widths"]).to(DEV)
    net.load_state_dict(z["state"])
    net.eval()
    return net, float(z["info"]["mean"]), float(z["info"]["sd"])


def align_against(r, ra):
    """The against-version's points (n, 22, 2) in r's row order, matched by patient id. r, ra: corners.load_oof results."""
    ids, ids_a = np.asarray(r["ids"]), np.asarray(ra["ids"])
    if len(ids) != len(ids_a) or set(ids.tolist()) != set(ids_a.tolist()):
        raise SystemExit("--against: the patient ids of the two versions differ")
    row = {int(i): j for j, i in enumerate(ids_a)}
    order = np.array([row[int(i)] for i in ids])
    t, ta = np.asarray(r["truth"], np.float64), np.asarray(ra["truth"], np.float64)[order]
    if t.shape != ta.shape:
        raise SystemExit("--against: the truth arrays of the two versions differ in shape (%s and %s)" % (t.shape, ta.shape))
    if not np.allclose(t, ta, rtol=0, atol=0.01):          # v1 and v2 store the same corners through other float32 arithmetic
        raise SystemExit("--against: the truth arrays of the two versions differ for the same ids by more than 0.01 px "
                         "(largest difference %.6g px)" % float(np.abs(t - ta).max()))
    return np.asarray(ra["pred"])[order].astype(np.float64)


def summarize(new, first, old, truth, against, folds):
    """Print the miss table, the bootstrap line, the sanity line and the verdict; (n, 22, 2) film-pixel arrays, old = the
    against-version's points. Returns m_new, m_old, dmean, lo, hi, kept, sanity."""
    m_new = corners.miss_per_patient(new, truth)
    m_old = corners.miss_per_patient(old, truth)                           # the against-version's points
    print("corners re-read: %d patients in folds %s, per-patient mean corner miss (px)"
          % (len(m_new), ",".join(map(str, folds))))
    for name, m in (("re-read", m_new), ("v%d" % against, m_old)):
        print("  %-8s mean %6.2f  median %6.2f" % (name, float(m.mean()), float(np.median(m))))
    diff = m_new - m_old
    dmean = float(diff.mean())
    lo, hi = (float(v) for v in corners.ci(corners.boot_means(diff[:, None])[:, 0]))
    kept = bool(hi < 0)
    print("re-read minus v%d: %+.2f px (paired patient-bootstrap 95%% CI %+.2f to %+.2f, %d resamples, seed %d)"
          % (against, dmean, lo, hi, corners.BOOT, corners.BOOT_SEED))
    sanity = float(np.linalg.norm(first - old, axis=-1).mean())
    print("sanity: pass 1 centre look vs v%d saved points, mean absolute difference %.3f px "
          "(this is device and bfloat16 differences only when the run starts where v%d started and reads as it read: "
          "one pass, one look from v%d's own start; otherwise it is how far this read moves from v%d)"
          % (against, sanity, against, against, against))
    print("Verdict (corners, re-read minus v%d): %s" % (against, "KEPT" if kept else "NOT KEPT"))
    return m_new, m_old, dmean, lo, hi, kept, sanity


def run(passes, looks, out_version, from_version, folds, against=None):
    t0 = time.time()
    against = from_version if against is None else against
    from PIL import Image
    offsets = look_offsets(looks)
    if DEV.type == "cuda":
        cr.limit_card_memory()
    r = corners.load_oof(VIEW, from_version, folds)
    old = r["pred"].astype(np.float64) if against == from_version \
        else align_against(r, corners.load_oof(VIEW, against, folds))
    ids_all, _, _ = buu.annotations(VIEW)
    fold_of = dict(zip((int(i) for i in ids_all), buu.folds_of(ids_all)))
    rfold = np.array([fold_of[int(i)] for i in r["ids"]])
    jpgs = buu.files(VIEW, ext="jpg")
    d = corners.out_dir(out_version)
    os.makedirs(d, exist_ok=True)
    fp = json.dumps({"view": VIEW, "from_version": from_version, "passes": passes, "looks": looks}, sort_keys=True)
    new = np.zeros(r["pred"].shape, np.float64)
    first = np.zeros(r["pred"].shape, np.float64)
    for k in folds:
        t1 = time.time()
        rows = np.where(rfold == k)[0]
        net, mean, sd = load_net(k)
        read = net_reader(net, mean, sd)
        for s in range(0, len(rows), CHUNK):
            part = rows[s:s + CHUNK]
            imgs = []
            for i in r["ids"][part]:
                with Image.open(jpgs[int(i)]) as im:                  # once a patient, however many passes
                    imgs.append(np.asarray(im.convert("L"), dtype=np.uint8))
            new[part], first[part] = reread(read, imgs, r["pred"][part].astype(np.float64), passes, offsets)
        np.savez(os.path.join(d, "%s-fold%d.npz" % (VIEW, k)), ids=r["ids"][rows], pred=new[rows].astype(np.float32),
                 truth=r["truth"][rows], base=r["base"][rows], base_size=r["base_size"][rows],
                 v1=r["pred"][rows].astype(np.float32), fold=np.int64(k), fingerprint=fp)
        del net
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        print("fold %d: %d patients re-read, %d passes of %d looks, %.0f s" % (k, len(rows), passes, looks,
                                                                              time.time() - t1), flush=True)

    m_new, m_old, dmean, lo, hi, kept, sanity = summarize(new, first, old, r["truth"].astype(np.float64), against, folds)
    # from_* keys hold the against-version's miss (the from-version's when --against is not given)
    numbers = {"corner_miss_mean_px": (m_new.mean(), "px"), "corner_miss_median_px": (np.median(m_new), "px"),
               "from_miss_mean_px": (m_old.mean(), "px"), "from_miss_median_px": (np.median(m_old), "px"),
               "miss_diff_px": (dmean, "px"), "miss_diff_low_px": (lo, "px"), "miss_diff_high_px": (hi, "px"),
               "corners_kept": (float(kept), ""), "pass1_vs_from_px": (sanity, "px"),
               "patients": (len(m_new), "count"), "passes": (passes, "count"), "looks": (looks, "count")}
    numbers = {k: (float(v), u) for k, (v, u) in numbers.items()}
    settings = {"from_version": from_version, "against_version": against, "out_version": out_version, "folds": ",".join(map(str, folds)),
                "passes": passes, "looks": looks, "crop": cr.CROP,
                "corners_verdict": "KEPT" if kept else "NOT KEPT",
                "bar": "re-read minus v%d miss, CI wholly below 0" % against}
    ledger.write("experiment", "spine xray corners re-read: %d passes, %d looks" % (passes, looks),
                 time.time() - t0, numbers, settings=settings,
                 note="modalities/spine_xray/eval/corners_reread.py; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


def selftest():
    assert look_offsets(1) == [(0, 0)]
    assert look_offsets(5) == [(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4)]
    for bad in (0, 2, 3):
        try:
            look_offsets(bad)
        except SystemExit:
            continue
        raise AssertionError("looks %d accepted" % bad)

    def brightest(crops, corner):                  # stand-in net: the brightest pixel's centre, crop coordinates
        out = []
        for c in crops:
            y, x = np.unravel_index(int(np.argmax(c)), c.shape)
            out.append((x + 0.5, y + 0.5))
        return np.array(out)

    for (px, py) in ((300, 200), (10, 7)):         # the second sits near the film's top-left, crops reach outside
        img = np.zeros((400, 500), np.uint8)
        img[py, px] = 255                          # pixel (px, py) spans [px, px + 1): its centre is (px + .5, py + .5)
        want = np.array([px + 0.5, py + 0.5])
        start = np.tile([px + 2.3, py - 1.7], (cr.N_CORNERS, 1))[None]       # a start point off the bright pixel
        for offsets in (LOOKS_1, LOOKS_5):
            got, first = reread(brightest, [img], start, 1, offsets)
            assert np.allclose(first[0], want), (first[0][0], want)
            if offsets is LOOKS_1:
                assert np.allclose(got[0], want), (got[0][0], want)
        # a second pass centred on the first pass's answer still finds it
        got, _ = reread(brightest, [img], start, 2, LOOKS_1)
        assert np.allclose(got[0], want)
    patch, (ox, oy) = cr.crop(np.zeros((50, 50), np.uint8), 3.4, 3.6, cr.CROP)
    assert (ox, oy) == (3 - cr.CROP // 2, 4 - cr.CROP // 2) and patch.shape == (cr.CROP, cr.CROP)
    # --against: the compared points are the against-version's, matched to the from-version's rows by id
    rng = np.random.default_rng(0)
    n = 6
    truth = rng.uniform(0, 100, (n, cr.N_CORNERS, 2)).astype(np.float32)
    from_pred = truth + 10.0                                   # every corner 10 * sqrt(2) px off
    ids = np.arange(100, 100 + n)
    shuf = rng.permutation(n)                                  # the against-version lists the same ids in another order
    ag_pred = (truth + 3.0)[shuf]                              # every corner 3 * sqrt(2) px off
    r_from = {"ids": ids, "truth": truth, "pred": from_pred}
    r_ag = {"ids": ids[shuf], "truth": truth[shuf], "pred": ag_pred}
    old = align_against(r_from, r_ag)
    assert np.allclose(old, truth + 3.0)
    new = truth + 5.0
    m_old = corners.miss_per_patient(old, truth.astype(np.float64))
    m_from = corners.miss_per_patient(from_pred.astype(np.float64), truth.astype(np.float64))
    assert np.allclose(m_old, 3 * 2 ** 0.5, atol=1e-4) and np.allclose(m_from, 10 * 2 ** 0.5, atol=1e-4)
    diff = corners.miss_per_patient(new.astype(np.float64), truth.astype(np.float64)) - m_old
    assert np.all(diff > 0)                                    # worse than v-against, though better than the from-version
    for bad in ({"ids": ids[shuf][:-1], "truth": truth[shuf][:-1], "pred": ag_pred[:-1]},
                {"ids": ids[shuf] + 1, "truth": truth[shuf], "pred": ag_pred},
                {"ids": ids[shuf], "truth": truth[shuf] + 1.0, "pred": ag_pred}):
        try:
            align_against(r_from, bad)
        except SystemExit:
            continue
        raise AssertionError("mismatched --against accepted")
    # the summary print path, stand-in arrays: a format error there fails here
    first = old + 0.5                                          # pass-1 answers 0.5 * sqrt(2) px from the against points
    got = summarize(new.astype(np.float64), first, old, truth.astype(np.float64), 2, [0, 1])
    assert np.isclose(got[6], 0.5 * 2 ** 0.5) and got[5] is False and got[2] > 0
    near = {"ids": ids[shuf], "truth": truth[shuf] + 1e-4, "pred": ag_pred}    # float32 rounding between versions
    assert np.allclose(align_against(r_from, near), truth + 3.0)
    print("selftest ok")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--looks", type=int, default=5, help="1 or 5")
    ap.add_argument("--out-version", type=int, help="names the output folder corners-v<N> (not 1 or 2)")
    ap.add_argument("--from-version", type=int, default=2)
    ap.add_argument("--against", type=int, default=None,
                    help="compare with this version's saved points (default: the from-version)")
    ap.add_argument("--folds", default="all", help="'all' or comma-separated fold numbers")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.out_version is None:
        raise SystemExit("--out-version is required")
    if a.out_version in (1, 2):
        raise SystemExit("version %d is a trained reader's folder; pick another" % a.out_version)
    if a.passes < 1:
        raise SystemExit("--passes must be at least 1")
    look_offsets(a.looks)
    run(a.passes, a.looks, a.out_version, a.from_version, corners.parse_folds(a.folds), a.against)


if __name__ == "__main__":
    main()
