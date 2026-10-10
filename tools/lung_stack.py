"""The lung system's false alarms re-scored by what they look like (idea 58).

The lung nodule system's kept list (seriesuid, coordX, coordY, coordZ, p1, p2, fold; p2 the mean of two
stage 2 nets, every score out of fold) gets a new p2: an out-of-fold logistic regression over eight fixed
features of each place, so plain facts (outside or deep in the lung, bright, crowded) can push false
alarms down. For each fold k a StandardScaler and a LogisticRegression(C=1.0) are fitted on the rows of
the other folds and score fold k; no row is ever fitted in the fold it is scored in.

Features (FEATURES, in this order): lz_v2, lz_v3 (logit of the p2 of fpr-v2 and fpr-v3 alone), lz_p1
(logit of p1), out_mm (mm outside the lung mask, 0 inside), in_mm (mm inside the mask to its edge, 0
outside), vol_mean4 and vol_max4 (mean and max of the 1 mm uint8 volume within 4 mm of the centre),
n_near15 (other rows of the scan within 15 mm). Each --extra-net adds lz_<NAME> (logit of that net's p2)
after these. A NaN feature (no volume or mask) is filled with the
median of the rows the fold is fitted on.

--model trees swaps the logistic regression for gradient-boosted trees: for each fold a
HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
l2_regularization=1.0, early_stopping=False, random_state=64) is fitted on the same rows with the same
median fill and no scaling, and scores fold k. The default, --model logistic, is unchanged.

Labels are LUNA16's, by froc.py's rules: a candidate inside a reference nodule is 1, one inside an
excluded finding (and no nodule) is NaN (scored, never fitted), any other is 0.

    python tools/lung_stack.py run [--candidates PATH] [--out PATH] [--limit-scans N] [--dry]
                                   [--model logistic|trees] [--v2 PATH] [--v3 PATH] [--without-p1]
                                   [--extra-net NAME=PATH ...]
        default candidates <WORK>/eval/lung/fpr-blend23/candidates_oof_sys5_tta8top32_inlung.csv, default out
        <WORK>/eval/lung/fpr-stack58/candidates_oof_sys5_tta8top32_inlung.csv (same columns and rows, p2 new);
        --limit-scans keeps the first N seriesuids in sorted order; --dry records no ledger run;
        --v2 PATH and --v3 PATH specify custom score lists (default: today's V2 and V3 constants) which are
        joined by place as lz_v2 and lz_v3;
        --without-p1 excludes lz_p1 from the model's features (seven features instead of eight, plus any
        --extra-net features); the candidate list need not have a p1 column; without the flag, a candidate
        list with no p1 column stops with a SystemExit naming --without-p1;
        --extra-net (repeatable) reads PATH, a candidates csv, joins its p2 by place (as fpr-v2/v3) and adds
        the feature lz_NAME = logit(p2) after the features above, in the order given; the model is fitted on
        all of them. Every place of the (kept) list must be in PATH or the run stops; nothing is filled.
    python tools/lung_stack.py selftest
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "modalities", "ct", "eval")
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), EVAL, ROOT]
from core import ledger, paths  # noqa: E402
import lung_inside  # noqa: E402

LUNG = os.path.join(paths.WORK, "eval", "lung")
CANDIDATES = os.path.join(LUNG, "fpr-blend23", "candidates_oof_sys5_tta8top32_inlung.csv")
OUT = os.path.join(LUNG, "fpr-stack58", "candidates_oof_sys5_tta8top32_inlung.csv")
V2 = os.path.join(LUNG, "fpr-v2", "candidates_oof_sys5_tta8top32.csv")
V3 = os.path.join(LUNG, "fpr-v3", "candidates_oof_sys5_tta8top32.csv")
KEY = ["seriesuid", "coordX", "coordY", "coordZ"]
XYZ = KEY[1:]
FEATURES = ("lz_v2", "lz_v3", "lz_p1", "out_mm", "in_mm", "vol_mean4", "vol_max4", "n_near15")
BALL_MM = 4.0
NEAR_MM = 15.0
JOBS = 4


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _rounded(df):
    df = df.copy()
    for c in XYZ:
        df[c] = df[c].round(3)
    return df


def scan_feats(uid, xyz, vol_dir=None, mask_fn=None):
    """(n, 3) in_mm, vol_mean4, vol_max4 of world points `xyz` (n x 3) in scan `uid`; NaN where there is no volume.

    Same voxel convention as lung_inside.scan_out. in_mm is 0 for a point whose voxel had to be clipped
    into the crop (it is outside the volume, so outside the mask); an empty or full mask gives NaN."""
    from scipy import ndimage as ndi
    vol_dir = vol_dir or os.path.join(LUNG, "vol")
    npy, js = os.path.join(vol_dir, uid + ".npy"), os.path.join(vol_dir, uid + ".json")
    out = np.full((len(xyz), 3), np.nan)
    if not (os.path.exists(npy) and os.path.exists(js)):
        return out
    with open(js, encoding="utf-8") as f:
        meta = json.load(f)
    spacing = np.asarray(meta["spacing_mm"], dtype=float)[::-1]
    p = (np.asarray(xyz, dtype=float)[:, ::-1] - np.asarray(meta["origin_mm"], dtype=float)[::-1]) / spacing
    mask = mask_fn(uid) if mask_fn else lung_inside._blobs_mask(uid, npy)
    if mask.any() and not mask.all():
        dist = ndi.distance_transform_edt(mask, sampling=spacing)
        c = np.clip(np.rint(p), 0, np.asarray(dist.shape) - 1).astype(int)
        out[:, 0] = np.where((c == np.rint(p)).all(axis=1), dist[c[:, 0], c[:, 1], c[:, 2]], 0.0)
    del mask
    vol = np.load(npy, mmap_mode="r")
    reach = np.ceil(BALL_MM / spacing).astype(int)
    offs = np.stack(np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing="ij"), axis=-1).reshape(-1, 3)
    for i, pt in enumerate(p):
        lo = np.rint(pt).astype(int)
        vox = lo + offs
        near = np.linalg.norm((vox - pt) * spacing, axis=1) <= BALL_MM
        vox = vox[near]
        vox = vox[((vox >= 0) & (vox < np.asarray(vol.shape))).all(axis=1)]
        if len(vox):
            v = vol[vox[:, 0], vox[:, 1], vox[:, 2]].astype(float)
            out[i, 1], out[i, 2] = v.mean(), v.max()
    return out


def _scan_job(args):
    uid, idx, xyz = args
    return idx, scan_feats(uid, xyz)


def n_near(xyz, radius=NEAR_MM):
    """(n,) the number of OTHER points of `xyz` within `radius` mm of each."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    return np.array([len(h) - 1 for h in tree.query_ball_point(xyz, radius)], dtype=float)


def _look_up(cand, other, name):
    """The p2 of `other` for every row of `cand`, by the rounded place; fails when a place is missing."""
    o = _rounded(other)
    if o.duplicated(KEY).any():
        raise SystemExit("%s lists a place twice" % name)
    m = _rounded(cand)[KEY].merge(o[KEY + ["p2"]], on=KEY, how="left", validate="many_to_one")
    if len(m) != len(cand) or m.p2.isna().any():
        raise SystemExit("%s lacks %d of the list's places" % (name, int(m.p2.isna().sum())))
    return m.p2.to_numpy(float)


def parse_extra_nets(specs):
    """[(name, path)] from `NAME=PATH` strings; NAME a simple identifier whose lz_<NAME> clashes with no feature."""
    out = []
    for s in specs or []:
        name, sep, path = s.partition("=")
        if not sep or not path or not name.isascii() or not name.isidentifier():
            raise SystemExit("--extra-net %r: want NAME=PATH with NAME a simple identifier" % s)
        col = "lz_" + name
        if col in FEATURES or col in [("lz_" + n) for n, _ in out]:
            raise SystemExit("--extra-net %r: feature %s already exists" % (s, col))
        out.append((name, path))
    return out


def build_features(cand, v2, v3, out_fn=None, scan_fn=None, jobs=JOBS, extra=None, with_p1=True):
    """`cand` with features added, same rows in the same order. v2, v3: the nets' own score lists.

    extra: {NAME: score list}, each joined by place into the column lz_<NAME> after the base features.
    with_p1: whether to build the lz_p1 feature (default True for backward compatibility)."""
    out_fn = out_fn or (lambda df: lung_inside.out_mm(df, jobs))
    f = cand.reset_index(drop=True).copy()
    for name in (extra or {}):
        if "lz_" + name in f.columns:
            raise SystemExit("feature lz_%s clashes with a column of the candidate list" % name)
    f["lz_v2"] = logit(_look_up(f, v2, "fpr-v2"))
    f["lz_v3"] = logit(_look_up(f, v3, "fpr-v3"))
    if with_p1:
        f["lz_p1"] = logit(f["p1"])
    f["out_mm"] = out_fn(f)
    xyz = f[XYZ].to_numpy(float)
    work = [(uid, np.asarray(ix), xyz[ix]) for uid, ix in f.groupby("seriesuid").indices.items()]
    if scan_fn is None and jobs > 1:
        with ProcessPoolExecutor(min(jobs, JOBS)) as ex:
            res = list(ex.map(_scan_job, work))
    else:
        res = [(idx, (scan_fn or scan_feats)(uid, pts)) for uid, idx, pts in work]
    stats = np.full((len(f), 3), np.nan)
    near = np.zeros(len(f))
    for (uid, idx, pts), (_, s) in zip(work, res):
        stats[idx] = s
        near[idx] = n_near(pts)
    f["in_mm"], f["vol_mean4"], f["vol_max4"] = stats[:, 0], stats[:, 1], stats[:, 2]
    f["n_near15"] = near
    for name, other in (extra or {}).items():
        f["lz_" + name] = logit(_look_up(f, other, "extra net " + name))
    return f


def labels(df, ann, exc):
    """(n,) 1.0 for a candidate inside a reference nodule, NaN inside an excluded finding only, else 0.0.

    froc._match returns only counts and curve inputs, so the per-row call is made here with its geometry
    (froc._radius_sq: strictly inside half the diameter, an unsized finding 1 mm wide; a nodule's hit wins
    over an excluded finding) and then checked against _match's own counts: its false positives, its
    ignored candidates and its hits (found nodules plus double detections) must equal ours."""
    import froc
    uids = sorted(set(df.seriesuid))
    ann = ann[ann.seriesuid.isin(uids)].reset_index(drop=True)
    exc = exc[exc.seriesuid.isin(uids)].reset_index(drop=True)
    ar2, er2 = froc._radius_sq(ann), froc._radius_sq(exc)
    axyz, exyz = ann[froc._XYZ].to_numpy(float), exc[froc._XYZ].to_numpy(float)
    out = np.zeros(len(df))
    xyz = df[froc._XYZ].to_numpy(float)
    for uid, ix in df.groupby("seriesuid").indices.items():
        a, e = (ann.seriesuid == uid).to_numpy(), (exc.seriesuid == uid).to_numpy()
        hit = np.zeros(len(ix), bool)
        if a.any():
            hit = (((xyz[ix][:, None, :] - axyz[a][None]) ** 2).sum(axis=2) < ar2[a][None]).any(axis=1)
        skip = np.zeros(len(ix), bool)
        if e.any():
            skip = (((xyz[ix][:, None, :] - exyz[e][None]) ** 2).sum(axis=2) < er2[e][None]).any(axis=1) & ~hit
        out[ix] = np.where(hit, 1.0, np.where(skip, np.nan, 0.0))
    m = froc._match(df[froc._XYZ + ["seriesuid"]].assign(probability=1.0), ann, exc, uids)
    n_hit = int((out == 1.0).sum())
    if (len(m["fp_prob"]) != int((out == 0.0).sum()) or m["ignored"] != int(np.isnan(out).sum())
            or len(m["tp_prob"]) + m["double"] != n_hit):
        raise SystemExit("labels disagree with froc._match: fp %d vs %d, ignored %d vs %d, hits %d vs %d"
                         % (len(m["fp_prob"]), (out == 0.0).sum(), m["ignored"], np.isnan(out).sum(),
                            len(m["tp_prob"]) + m["double"], n_hit))
    return out


def oof_predict(frame, features, label="label", fold="fold", info=None, model="logistic"):
    """Out-of-fold probabilities, one per row of `frame` in row order.

    For each fold k: StandardScaler + LogisticRegression(C=1.0) fitted on rows with fold != k and a label,
    probability of the label being 1 for every row of fold k. A NaN feature is filled with that column's
    median over the fitting rows. `info` (a list) gets one dict per fold: fold, fitted, scored, filled
    (cells filled in the fitted and in the scored rows) and weights. model "trees" fits a
    HistGradientBoostingClassifier on the same rows, unscaled, instead (weights {}, intercept None, plus
    model and seconds)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    features = list(features)
    x = frame[features].to_numpy(float)
    y = frame[label].to_numpy(float)
    fo = frame[fold].to_numpy(float)
    out = np.full(len(frame), np.nan)
    for k in sorted(set(fo[~np.isnan(fo)])):
        fit = (fo != k) & ~np.isnan(y)
        test = fo == k
        med = np.array([np.median(c[~np.isnan(c)]) if (~np.isnan(c)).any() else 0.0 for c in x[fit].T])
        xf, xt = x[fit].copy(), x[test].copy()
        filled = (int(np.isnan(xf).sum()), int(np.isnan(xt).sum()))
        xf = np.where(np.isnan(xf), med, xf)
        xt = np.where(np.isnan(xt), med, xt)
        if model == "trees":
            import time
            t0 = time.time()
            gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                                l2_regularization=1.0, early_stopping=False, random_state=64).fit(xf, y[fit])
            out[test] = gb.predict_proba(xt)[:, 1]
            if info is not None:
                info.append(dict(fold=int(k), fitted=int(fit.sum()), scored=int(test.sum()), filled=filled,
                                 weights={}, intercept=None, model="trees", seconds=time.time() - t0))
            continue
        sc = StandardScaler().fit(xf)
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(sc.transform(xf), y[fit])
        out[test] = lr.predict_proba(sc.transform(xt))[:, 1]
        if info is not None:
            info.append(dict(fold=int(k), fitted=int(fit.sum()), scored=int(test.sum()), filled=filled,
                             weights=dict(zip(features, lr.coef_[0].tolist())), intercept=float(lr.intercept_[0])))
    return out


def run_cmd(a):
    cand = pd.read_csv(a.candidates)
    if a.limit_scans:
        cand = cand[cand.seriesuid.isin(sorted(set(cand.seriesuid.astype(str)))[:a.limit_scans])]
    cand = cand.reset_index(drop=True)
    print("%d rows in %d scans from %s" % (len(cand), cand.seriesuid.nunique(), a.candidates), flush=True)
    import froc
    ann, exc, ref = froc.load_reference()
    missing = set(cand.seriesuid) - set(ref)
    if missing:
        raise SystemExit("%d scans of the list are not in LUNA16's scan list" % len(missing))

    without_p1 = getattr(a, "without_p1", False)
    if not without_p1 and "p1" not in cand.columns:
        raise SystemExit("candidate list has no p1 column; use --without-p1 to proceed without it")

    v2_path = getattr(a, "v2", None) or V2
    v3_path = getattr(a, "v3", None) or V3
    extra_nets = parse_extra_nets(getattr(a, "extra_net", None))
    extra = {n: pd.read_csv(p) for n, p in extra_nets}

    base_features = [f for f in FEATURES if f != "lz_p1" or not without_p1]
    features = base_features + ["lz_" + n for n, _ in extra_nets]
    f = build_features(cand, pd.read_csv(v2_path), pd.read_csv(v3_path), jobs=JOBS, extra=extra,
                       with_p1=not without_p1)
    f["label"] = labels(f, ann, exc)
    print("labels: %d nodule hits, %d false alarms, %d on excluded findings (scored, not fitted)"
          % ((f.label == 1).sum(), (f.label == 0).sum(), f.label.isna().sum()), flush=True)
    info = []
    p = oof_predict(f, features, "label", "fold", info, model=a.model)
    if np.isnan(p).any():
        raise SystemExit("%d rows got no probability (a NaN fold?)" % int(np.isnan(p).sum()))
    for d in info:
        print("fold %d: fitted %d rows, scored %d rows, NaN features filled %d in fitted rows and %d in scored rows"
              % (d["fold"], d["fitted"], d["scored"], d["filled"][0], d["filled"][1]))
        if a.model == "trees":
            print("   trees: 300 iterations, fit %.1f s" % d["seconds"])
        else:
            print("   " + "  ".join("%s %+.3f" % (n, w) for n, w in d["weights"].items()) + "  (intercept %+.3f)" % d["intercept"])
    res = cand.copy()
    res["p2"] = p
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res.to_csv(a.out, index=False)
    assert list(res.columns) == list(pd.read_csv(a.candidates, nrows=0).columns)
    print("wrote %d rows to %s" % (len(res), a.out))
    return info, res


def selftest():
    import tempfile
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n = 4000
    fold = rng.integers(0, 5, n)
    y = rng.integers(0, 2, n).astype(float)
    noise = rng.normal(size=n)
    # f is the label in fold 0 and carries nothing elsewhere (0), so a fit that never sees fold 0 gives it
    # weight 0; a noisy f elsewhere would not do: any weight at all ranks a 0/1 copy of the label perfectly
    df = pd.DataFrame({"f": np.where(fold == 0, y, 0.0), "g": noise, "label": y, "fold": fold})
    auc = roc_auc_score(y[fold == 0], oof_predict(df, ["f", "g"])[fold == 0])
    assert auc < 0.65, "a feature true only in the scored fold must not help it: %.3f" % auc
    df["f"] = y
    p = oof_predict(df, ["f"])
    auc = roc_auc_score(y[fold == 0], p[fold == 0])
    assert auc > 0.95, "a feature true everywhere must score high: %.3f" % auc
    df["f"] = np.where(rng.random(n) < 0.1, np.nan, y)
    df.loc[df.index[:30], "label"] = np.nan
    info = []
    p = oof_predict(df, ["f"], info=info)
    assert not np.isnan(p).any() and sum(d["scored"] for d in info) == n, "every row, NaN label or feature too, is scored"
    # trees: the same leak test; fold 0's own copy of the label must not reach fold 0, and the fit is repeatable
    df = pd.DataFrame({"f": np.where(fold == 0, y, 0.0), "g": noise, "label": y, "fold": fold})
    pt = oof_predict(df, ["f", "g"], model="trees")
    auc = roc_auc_score(y[fold == 0], pt[fold == 0])
    assert auc < 0.65, "trees: a feature true only in the scored fold must not help it: %.3f" % auc
    assert (pt == oof_predict(df, ["f", "g"], model="trees")).all(), "trees must give the same output twice"
    # the build step on a made-up scan
    with tempfile.TemporaryDirectory() as tmp:
        shape = (20, 20, 20)
        mask = np.zeros(shape, bool)
        mask[5:15, 5:15, 5:15] = True
        vol = np.full(shape, 100, np.uint8)
        vol[10, 10, 10] = 200
        origin = [10.0, 20.0, 30.0]
        np.save(os.path.join(tmp, "a.npy"), vol)
        with open(os.path.join(tmp, "a.json"), "w", encoding="utf-8") as fh:
            json.dump({"origin_mm": origin, "spacing_mm": [1.0, 1.0, 1.0], "shape": list(shape)}, fh)

        def world(z, y_, x):
            return [origin[0] + x, origin[1] + y_, origin[2] + z]
        pts = [world(10, 10, 10), world(10, 10, 12), world(10, 10, 19), world(10, 10, 30)]
        rows = [("a",) + tuple(q) for q in pts] + [("nov", 1.0, 2.0, 3.0), ("nov", 100.0, 2.0, 3.0)]
        cand = pd.DataFrame(rows, columns=KEY)
        cand["p1"] = np.linspace(0.1, 0.9, len(cand))
        cand["p2"] = np.linspace(0.2, 0.8, len(cand))
        cand["fold"] = [0, 1, 2, 3, 4, 0]
        v2, v3 = cand[KEY + ["p2"]].iloc[::-1], cand[KEY + ["p2"]].assign(p2=0.5)
        masks = {"a": mask}
        out_fn = lambda d: lung_inside.out_mm(d, 1, run=lambda uid, q: lung_inside.scan_out(uid, q, tmp, masks.__getitem__))  # noqa: E731
        scan_fn = lambda uid, q: scan_feats(uid, q, tmp, masks.__getitem__)  # noqa: E731
        built = build_features(cand, v2, v3, out_fn=out_fn, scan_fn=scan_fn)
        assert len(built) == len(cand) and (built[KEY] == cand[KEY]).all().all(), "the build step must keep every row in order"
        assert list(built.columns[:len(cand.columns)]) == list(cand.columns) and all(c in built for c in FEATURES)
        assert np.allclose(built.lz_v2, logit(cand.p2)), "fpr-v2 scores must be joined by place, not by row order"
        assert built.in_mm[0] == 5.0 and built.in_mm[1] == 3.0 and built.in_mm[2] == 0.0 and built.in_mm[3] == 0.0, built.in_mm
        assert built.out_mm[0] == 0.0 and abs(built.out_mm[2] - 5.0) < 1e-6, built.out_mm
        ball = 257                                     # voxels within 4 mm of a voxel centre, 1 mm grid
        assert abs(built.vol_mean4[0] - (256 * 100 + 200) / ball) < 1e-9 and built.vol_max4[0] == 200, built.vol_mean4
        assert built.vol_max4[1] == 200 and np.isnan(built.vol_mean4[3]), "a point 10 voxels past the crop has no voxel within 4 mm"
        assert built[list(FEATURES[3:])].iloc[4:].isna().sum().sum() == 8, "a scan with no volume has NaN features"
        assert built.n_near15.tolist() == [2, 2, 3, 1, 0, 0], built.n_near15
        x = built.assign(label=[1.0, 0.0, np.nan, 0.0, 1.0, 0.0])
        p = oof_predict(x, FEATURES)
        assert len(p) == len(cand) and not np.isnan(p).any()
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--candidates", default=CANDIDATES)
    r.add_argument("--out", default=OUT)
    r.add_argument("--limit-scans", type=int, default=0, help="keep only the first N seriesuids (sorted)")
    r.add_argument("--dry", action="store_true", help="record no ledger run")
    r.add_argument("--model", choices=("logistic", "trees"), default="logistic",
                   help="the re-scorer: logistic (default) or trees (gradient-boosted trees)")
    r.add_argument("--v2", metavar="PATH", help="custom fpr-v2 score list (default: today's V2 constant)")
    r.add_argument("--v3", metavar="PATH", help="custom fpr-v3 score list (default: today's V3 constant)")
    r.add_argument("--without-p1", action="store_true", help="exclude lz_p1 from model features")
    r.add_argument("--extra-net", action="append", default=[], metavar="NAME=PATH",
                   help="another net's candidates csv (p2 joined by place) as the feature lz_NAME; repeatable")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    extra_nets = parse_extra_nets(a.extra_net)
    without_p1 = getattr(a, "without_p1", False)
    base_features = [f for f in FEATURES if f != "lz_p1" or not without_p1]
    if a.dry:
        run_cmd(a)
        return 0
    name = "lung stack 58 of " + os.path.basename(a.candidates) + (" (trees)" if a.model == "trees" else "")
    with ledger.run("experiment", name) as led:
        led.settings_from({"candidates": a.candidates, "out": a.out, "limit_scans": a.limit_scans,
                           "features": base_features + ["lz_" + n for n, _ in extra_nets], "C": 1.0,
                           "solver": "lbfgs", "ball_mm": BALL_MM, "near_mm": NEAR_MM, "model": a.model,
                           **({"trees": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
                                         "min_samples_leaf": 200, "l2_regularization": 1.0,
                                         "early_stopping": False, "random_state": 64}} if a.model == "trees" else {}),
                           **({"extra_nets": {n: p for n, p in extra_nets}} if extra_nets else {}),
                           **({"v2": a.v2} if a.v2 else {}), **({"v3": a.v3} if a.v3 else {}),
                           **({"without_p1": True} if without_p1 else {})})
        info, res = run_cmd(a)
        led.number("rows", len(res), "count").number("scans", res.seriesuid.nunique(), "count")
        for d in info:
            led.number("fold%d_fitted" % d["fold"], d["fitted"], "count")
            led.number("fold%d_scored" % d["fold"], d["scored"], "count")
    return 0


if __name__ == "__main__":
    sys.exit(main())
