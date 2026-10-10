"""Pull the far-off spine X-ray corners back to a spine's shape: no net, no training, processor only, seconds.

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646 (LICENSING.md, BUU-LSPINE row). Plan and bars:
docs/efficiency/README.md, "Corners v8, idea 13: a spine shape prior".

Why. The side view's corner miss is 12.0 px on average but 7.3 at the median, so a few corners are far off, and one
corner thrown 40 px makes a slip where there is none. A lumbar spine's 22 corners move together, so a corner far from
where the other 21 put it is probably wrong.

How. For fold k, the shape model is fitted ONLY on the drawn corners of the patients not in fold k: generalized
Procrustes alignment (shift, uniform size, turn; no mirror) to their mean, then PCA, keeping the fewest modes that
explain 98% of the variance. Each patient of fold k has the from-version's 22 predicted corners fitted by the model
robustly: 20 rounds that alternate (a) the similarity transform from the model's frame to the film and (b) the mode
weights (each clipped to +-3 standard deviations of its mode), both weighted by Tukey biweights (c 4.685) of the
per-corner distance between the point and the fit, the scale being 1.4826 x the median distance (floored at 1 px). A
corner whose final weight is 0 (distance beyond c x scale) takes the fitted model's point; every other corner keeps
the net's point. The pure projection (every corner from the fit) is computed too, for the report only. Nothing is
tuned on fold k. Writes <WORK>/eval/spine_xray/corners-v<out>/LA-fold<k>.npz with the keys corners_reread.py writes
(v1 holds the from-version's points), so corners.load_oof and slip_learned.py --corners-version <out> read it
unchanged. Out-version 1 and 2 are refused.

The bar (corners): KEPT when the paired patient-bootstrap 95% interval of the per-patient mean miss, new minus the
from-version's, lies wholly below 0. The slips bar is slip_learned.py's, run on the new folder.

Usage: python corners_shape.py --from-version 2 --out-version 8 [--folds all]
       python corners_shape.py --selftest      (no data files)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import ledger  # noqa: E402
import buu  # noqa: E402
import corners  # noqa: E402

VIEW = "LA"
VARIANCE = 0.98                 # share of the variance the kept modes explain
TUKEY_C = 4.685
ROUNDS = 20
CLIP_SD = 3.0                   # each mode weight is held within this many standard deviations
SCALE_FLOOR_PX = 1.0            # the residual scale never goes below this
GPA_ROUNDS = 20


def _cplx(p):
    return p[..., 0] + 1j * p[..., 1]


def _real(z):
    return np.stack([z.real, z.imag], axis=-1)


def similarity_to(ref, z, w=None):
    """Complex a, t with a * ref + t closest (weighted least squares) to z. ref, z: (n,) complex."""
    w = np.ones(len(z)) if w is None else np.maximum(w, 1e-8)
    sw = w.sum()
    rc, zc = (w * ref).sum() / sw, (w * z).sum() / sw
    a = (w * np.conj(ref - rc) * (z - zc)).sum() / (w * np.abs(ref - rc) ** 2).sum()
    return a, zc - a * rc


def fit_model(shapes):
    """shapes (N, 22, 2) drawn corners. Returns mean (22, 2), modes (K, 22, 2), sd (K,), in the aligned frame."""
    z = _cplx(np.asarray(shapes, np.float64))
    z = z - z.mean(axis=1, keepdims=True)
    z = z / np.sqrt((np.abs(z) ** 2).sum(axis=1, keepdims=True))
    mean = z[0].copy()
    for _ in range(GPA_ROUNDS):
        al = np.empty_like(z)
        for i in range(len(z)):
            b = (np.conj(z[i]) * mean).sum() / (np.abs(z[i]) ** 2).sum()          # turn and size onto the mean
            al[i] = z[i] * b
        new = al.mean(axis=0)
        new = new - new.mean()
        new = new / np.sqrt((np.abs(new) ** 2).sum())
        moved = np.abs(new - mean).max()
        mean = new
        if moved < 1e-12:
            break
    al = np.empty_like(z)
    for i in range(len(z)):
        b = (np.conj(z[i]) * mean).sum() / (np.abs(z[i]) ** 2).sum()
        al[i] = z[i] * b
    flat = _real(al - mean).reshape(len(z), -1)
    _, s, vt = np.linalg.svd(flat, full_matrices=False)
    var = s ** 2
    keep = int(np.searchsorted(np.cumsum(var) / var.sum(), VARIANCE) + 1)
    sd = s[:keep] / np.sqrt(max(len(z) - 1, 1))
    return _real(mean), vt[:keep].reshape(keep, -1, 2), sd


def robust_fit(model, pts):
    """pts (22, 2) predicted corners. Returns (fitted points (22, 2), final Tukey weights (22,))."""
    mean, modes, sd = model
    n = len(pts)
    p = _cplx(np.asarray(pts, np.float64))
    mean_c = _cplx(mean)
    basis = modes.reshape(len(modes), -1).T                                     # (44, K)
    q = mean_c.copy()
    w = np.ones(n)
    fitted = p
    for _ in range(ROUNDS):
        a, t = similarity_to(q, p, w)                                           # (a) model frame -> film
        back = (p - t) / a                                                      # the points in the model frame
        ww = np.repeat(np.maximum(w, 1e-8), 2)                                  # (b) mode weights
        y = _real(back - mean_c).reshape(-1)
        lhs = basis.T @ (ww[:, None] * basis) + 1e-9 * np.eye(basis.shape[1])
        c = np.clip(np.linalg.solve(lhs, basis.T @ (ww * y)), -CLIP_SD * sd, CLIP_SD * sd)
        q = mean_c + _cplx((basis @ c).reshape(n, 2))
        fitted = a * q + t
        r = np.abs(p - fitted)
        scale = max(1.4826 * float(np.median(r)), SCALE_FLOOR_PX)
        u = r / (TUKEY_C * scale)
        w = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    return _real(fitted), w


def apply_shape(model, pred):
    """pred (n, 22, 2). Returns (output, pure projection, replaced mask (n, 22))."""
    out = np.asarray(pred, np.float64).copy()
    proj = np.empty_like(out)
    replaced = np.zeros(out.shape[:2], bool)
    for i in range(len(out)):
        fit, w = robust_fit(model, pred[i])
        proj[i] = fit
        replaced[i] = w == 0.0
        out[i][replaced[i]] = fit[replaced[i]]
    return out, proj, replaced


def run(out_version, from_version, folds):
    t0 = time.time()
    r = corners.load_oof(VIEW, from_version, folds)
    ids_all, pts_all, _ = buu.annotations(VIEW)
    fold_all = buu.folds_of(ids_all)
    fold_of = dict(zip((int(i) for i in ids_all), fold_all))
    rfold = np.array([fold_of[int(i)] for i in r["ids"]])
    d = corners.out_dir(out_version)
    os.makedirs(d, exist_ok=True)
    fp = json.dumps({"view": VIEW, "from_version": from_version, "variance": VARIANCE, "tukey_c": TUKEY_C,
                     "rounds": ROUNDS, "clip_sd": CLIP_SD, "scale_floor_px": SCALE_FLOOR_PX}, sort_keys=True)
    pred = r["pred"].astype(np.float64)
    new = np.zeros(pred.shape, np.float64)
    proj = np.zeros(pred.shape, np.float64)
    replaced = np.zeros(pred.shape[:2], bool)
    modes_kept = []
    for k in folds:
        t1 = time.time()
        rows = np.where(rfold == k)[0]
        model = fit_model(pts_all[fold_all != k])
        modes_kept.append(len(model[1]))
        new[rows], proj[rows], replaced[rows] = apply_shape(model, pred[rows])
        np.savez(os.path.join(d, "%s-fold%d.npz" % (VIEW, k)), ids=r["ids"][rows], pred=new[rows].astype(np.float32),
                 truth=r["truth"][rows], base=r["base"][rows], base_size=r["base_size"][rows],
                 v1=r["pred"][rows].astype(np.float32), fold=np.int64(k), fingerprint=fp)
        print("fold %d: %d patients, %d modes kept, %d corners replaced, %.1f s"
              % (k, len(rows), modes_kept[-1], int(replaced[rows].sum()), time.time() - t1), flush=True)

    truth = r["truth"].astype(np.float64)
    m_new = corners.miss_per_patient(new, truth)
    m_old = corners.miss_per_patient(pred, truth)
    m_proj = corners.miss_per_patient(proj, truth)
    print("corners shape prior: %d patients in folds %s, per-patient mean corner miss (px)"
          % (len(m_new), ",".join(map(str, folds))))
    for name, m in (("shape", m_new), ("v%d" % from_version, m_old)):
        print("  %-8s mean %6.2f  median %6.2f" % (name, float(m.mean()), float(np.median(m))))
    diff = m_new - m_old
    dmean = float(diff.mean())
    lo, hi = (float(v) for v in corners.ci(corners.boot_means(diff[:, None])[:, 0]))
    kept = bool(hi < 0)
    print("shape minus v%d: %+.2f px (paired patient-bootstrap 95%% CI %+.2f to %+.2f, %d resamples, seed %d)"
          % (from_version, dmean, lo, hi, corners.BOOT, corners.BOOT_SEED))
    n_rep, n_pat = int(replaced.sum()), int(replaced.any(axis=1).sum())
    d_before = np.linalg.norm(pred - truth, axis=-1)[replaced]
    d_after = np.linalg.norm(new - truth, axis=-1)[replaced]
    before = float(d_before.mean()) if n_rep else 0.0
    after = float(d_after.mean()) if n_rep else 0.0
    print("replaced: %d of %d corners in %d of %d patients; their mean miss %.2f px before, %.2f px after"
          % (n_rep, replaced.size, n_pat, len(m_new), before, after))
    print("pure projection (every corner from the model): mean %.2f px, median %.2f px"
          % (float(m_proj.mean()), float(np.median(m_proj))))
    print("modes kept per fold: %s" % ", ".join("%d:%d" % kv for kv in zip(folds, modes_kept)))
    secs = time.time() - t0
    print("%.1f s" % secs)
    print("Verdict (corners, shape prior minus v%d): %s" % (from_version, "KEPT" if kept else "NOT KEPT"))
    numbers = {"corner_miss_mean_px": (m_new.mean(), "px"), "corner_miss_median_px": (np.median(m_new), "px"),
               "from_miss_mean_px": (m_old.mean(), "px"), "from_miss_median_px": (np.median(m_old), "px"),
               "miss_diff_px": (dmean, "px"), "miss_diff_low_px": (lo, "px"), "miss_diff_high_px": (hi, "px"),
               "corners_kept": (float(kept), ""), "replaced_corners": (n_rep, "count"),
               "replaced_patients": (n_pat, "count"), "replaced_miss_before_px": (before, "px"),
               "replaced_miss_after_px": (after, "px"), "projection_miss_mean_px": (m_proj.mean(), "px"),
               "projection_miss_median_px": (np.median(m_proj), "px"),
               "modes_kept_mean": (np.mean(modes_kept), "count"), "patients": (len(m_new), "count")}
    numbers = {k: (float(v), u) for k, (v, u) in numbers.items()}
    settings = {"from_version": from_version, "out_version": out_version, "folds": ",".join(map(str, folds)),
                "variance": VARIANCE, "tukey_c": TUKEY_C, "rounds": ROUNDS, "clip_sd": CLIP_SD,
                "scale_floor_px": SCALE_FLOOR_PX, "modes_per_fold": ",".join(map(str, modes_kept)),
                "corners_verdict": "KEPT" if kept else "NOT KEPT",
                "bar": "shape prior minus v%d miss, CI wholly below 0" % from_version}
    ledger.write("experiment", "spine xray corners shape prior: from v%d" % from_version, secs, numbers,
                 settings=settings,
                 note="modalities/spine_xray/eval/corners_shape.py; BUU-LSPINE (Klinwichit et al., Applied Sciences 2023;13(15):8646)")


def _synthetic(rng, n):
    """n plausible 22-point spines: 11 levels, two corners each, a base shape plus two smooth modes and small noise,
    then a random similarity transform."""
    lv = np.linspace(0.0, 1.0, 11)
    base = np.zeros((22, 2))
    base[0::2] = np.stack([-40 + 25 * np.sin(2.5 * lv), 400 * lv], axis=1)
    base[1::2] = np.stack([40 + 25 * np.sin(2.5 * lv), 400 * lv + 8], axis=1)
    lordosis = np.zeros((22, 2))
    lordosis[:, 0] = np.repeat(np.sin(np.pi * lv), 2) * 30
    taper = np.zeros((22, 2))
    taper[:, 0] = np.repeat(lv - 0.5, 2) * np.tile([-1.0, 1.0], 11) * 40
    out = np.empty((n, 22, 2))
    for i in range(n):
        s = base + rng.normal() * lordosis + rng.normal() * taper + rng.normal(0, 0.3, (22, 2))
        th, sc = rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.2)
        rot = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        out[i] = s @ rot.T + rng.uniform(-100, 100, 2) + [500, 300]
    return out


def selftest():
    rng = np.random.default_rng(0)
    shapes = _synthetic(rng, 60)
    model = fit_model(shapes[:50])
    assert 1 <= len(model[1]) <= 8, len(model[1])
    assert model[0].shape == (22, 2) and model[2].shape == (len(model[1]),)
    for held in shapes[50:]:
        # a clean shape comes back exactly as given (nothing replaced)
        out, proj, rep = apply_shape(model, held[None])
        assert not rep.any() and np.array_equal(out[0], held), rep.sum()
        assert np.linalg.norm(proj[0] - held, axis=-1).max() < 4.0
        # one corner thrown 40 px is replaced and lands near its true place; the other 21 are untouched
        for c, direction in ((5, (1.0, 0.0)), (16, (0.0, -1.0)), (0, (0.6, 0.8))):
            bad = held.copy()
            bad[c] += 40.0 * np.array(direction)
            out, _, rep = apply_shape(model, bad[None])
            assert rep[0].sum() == 1 and rep[0, c], np.where(rep[0])[0]
            assert np.linalg.norm(out[0, c] - held[c]) < 5.0, np.linalg.norm(out[0, c] - held[c])
            keep = ~rep[0]
            assert np.array_equal(out[0][keep], bad[keep])
    print("selftest ok")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-version", type=int, default=2)
    ap.add_argument("--out-version", type=int, help="names the output folder corners-v<N> (not 1 or 2)")
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
    run(a.out_version, a.from_version, corners.parse_folds(a.folds))


if __name__ == "__main__":
    main()
