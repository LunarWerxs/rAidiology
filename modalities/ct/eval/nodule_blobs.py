"""Classic blob candidates for the lung nodule detector: a long list for stage 2 to re-score.

The network's list (nodule.py) misses a quarter of fold 0's nodules outright: at 100 peaks a scan
its v2 reaches 176 of 240, and the other 56 have no peak within four times their radius, so no
re-scoring can find them. LUNA16's own candidate list, made by classic detectors, reaches nearly
all of them, and stage 2 ranks that list at CPM 0.840. This builds such a list itself: bright round
things in the lung found the classic way, the scale-normalised Laplacian of Gaussian over radii of
1.5 to 13 mm (sigma = radius / sqrt 3), its local maxima over a 5 mm cube inside the lung
(lung_prepare's mask grown 4 mm, so nodules on the wall stay in), the strongest `--top` a scan.
Each blob also carries its Hessian's eigenvalues at its own scale (l1 <= l2 <= l3) and its roundness
l3 / l1 (1 for a ball, 0 for a vessel or a wall), so `--score` can rank by shape as well as strength:
vessels are the commonest bright blobs in the lung, and they are tubes.

    python modalities/ct/eval/nodule_blobs.py build [--folds 0 ... | --nlst] [--top 1000] [--jobs 3]
        [--window LOW_HU HIGH_HU]   (--nlst: NLST501's scans, fold -1, for nodule_fpr.py --extra-nlst; use its own --name)
    python modalities/ct/eval/nodule_blobs.py reach [--with <candidates.csv> --with-top 100] [--score <expr>]
    python modalities/ct/eval/nodule_blobs.py join --with <candidates.csv> [--with-top 100] [--top 1000]
        [--merge-mm 4] [--score <expr>] --out <file>
    python modalities/ct/eval/nodule_blobs.py top [--top 300] [--score <expr>] --out <file>
    (every command takes --name <folder under <WORK>/eval/lung>, default blobs)

`build` writes <WORK>/eval/lung/<--name, default blobs>/candidates_oof.csv (seriesuid, coordX, coordY, coordZ,
probability = the response, radius_mm, fold); a scan already in it is kept. `reach` prints how many
nodules each depth reaches (froc.py's rule: a candidate within the nodule's radius), alone and
joined to another list's top K a scan. `join` writes that joined list for stage 2
(`nodule_fpr.py rescore --stage1-file`): the other list's top K a scan as they were, then the
blobs further than `--merge-mm` from all of them, scored below every one of them (their response
times 1e-6 over its maximum), so stage 1's ranking stays the other list's and the blobs come after.
"""
import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(os.path.dirname(HERE)))]
import froc  # noqa: E402
import lung_prepare as lp  # noqa: E402
from core import paths  # noqa: E402

RADII_MM = (1.5, 2.5, 4.0, 6.0, 9.0, 13.0)
HALF_FROM_MM = 6.0      # radii from here up are found at 2 mm (2x2x2 means), 11x faster, then brought back to 1 mm
GROW_MM = 4
PEAK_MM = 5
LUNG = os.path.join(paths.WORK, "eval", "lung")
MASKS = os.path.join(LUNG, "mask")


def lung_of(uid, hu):
    """lung_prepare's lung mask of the scan, kept in <WORK>/eval/lung/mask/<uid>.npz (bits) after the first time."""
    path = os.path.join(MASKS, uid + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        shape = tuple(int(v) for v in z["shape"])
        return np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)
    m = lp.lung_mask(hu)
    os.makedirs(MASKS, exist_ok=True)
    tmp = path[:-4] + ".part.npz"
    np.savez_compressed(tmp, bits=np.packbits(m.ravel()), shape=np.array(m.shape))
    os.replace(tmp, path)
    return m


def _log_max(img, radii, first):
    """Max over `radii` of the scale-normalised Laplacian (bright blobs positive) and the index of the radius."""
    from scipy import ndimage as ndi
    best = np.full(img.shape, -np.inf, np.float32)
    which = np.zeros(img.shape, np.uint8)
    for i, s in enumerate(radii):
        resp = ndi.gaussian_laplace(img, s, truncate=3.0)
        resp *= -(s * s)
        up = resp > best
        best[up] = resp[up]
        which[up] = first + i
        del resp, up
    return best, which


def _kernels(s):
    """The six second-derivative-of-Gaussian kernels (zz, yy, xx, zy, zx, yx) at sigma `s`, times s^2."""
    r = int(math.ceil(3.0 * s))
    t = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-t * t / (2 * s * s))
    g /= g.sum()
    g1 = -t / (s * s) * g
    g2 = (t * t / s ** 4 - 1.0 / (s * s)) * g
    o = np.einsum
    k = [o("i,j,k->ijk", g2, g, g), o("i,j,k->ijk", g, g2, g), o("i,j,k->ijk", g, g, g2),
         o("i,j,k->ijk", g1, g1, g), o("i,j,k->ijk", g1, g, g1), o("i,j,k->ijk", g, g1, g1)]
    return r, (s * s) * np.stack(k).astype(np.float32)


def hessian_at(img, zyx, sigmas):
    """(n, 3) eigenvalues, smallest first, of the scale-normalised Hessian of `img` at each point, each at its sigma.
    A bright ball has all three negative and alike; a vessel has one near 0, a wall two."""
    out = np.zeros((len(zyx), 3), np.float32)
    rmax = int(math.ceil(3.0 * max(sigmas))) if len(sigmas) else 0
    pad = np.pad(img, rmax, mode="edge")
    for s in np.unique(sigmas):
        idx = np.flatnonzero(sigmas == s)
        r, ker = _kernels(float(s))
        for chunk in np.array_split(idx, max(1, len(idx) * (2 * r + 1) ** 3 // 20_000_000 + 1)):
            c = zyx[chunk] + rmax
            patches = np.stack([pad[z - r:z + r + 1, y - r:y + r + 1, x - r:x + r + 1] for z, y, x in c])
            h = np.einsum("nijk,cijk->nc", patches, ker)
            m = np.empty((len(chunk), 3, 3), np.float32)
            m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = h[:, 0], h[:, 1], h[:, 2]
            m[:, 0, 1] = m[:, 1, 0] = h[:, 3]
            m[:, 0, 2] = m[:, 2, 0] = h[:, 4]
            m[:, 1, 2] = m[:, 2, 1] = h[:, 5]
            out[chunk] = np.linalg.eigvalsh(m)
    return out


def scan_blobs(uid, top, window=(-1000.0, 400.0)):
    """(rows, seconds) of the scan's `top` strongest blobs in the lung."""
    from scipy import ndimage as ndi
    t0 = time.time()
    meta = lp.load_meta(uid)
    hu = lp.to_hu(np.load(lp.paths_for(lp.out_dir(), uid)[0])).astype(np.float32)
    lung = ndi.binary_dilation(lung_of(uid, hu), iterations=GROW_MM)
    # the window: HU outside it are clipped. The default sees dense nodules and vessels; a ground-glass
    # window (-1000 to -300) makes a faint nodule as bright against the lung as a dense one
    img = np.clip(hu, float(window[0]), float(window[1]))
    del hu
    sig = [r / math.sqrt(3.0) for r in RADII_MM]
    n_fine = sum(r < HALF_FROM_MM for r in RADII_MM)
    best, which = _log_max(img, sig[:n_fine], 0)
    if n_fine < len(sig):
        # 2x2x2 means, the big radii at half the sigma in 2 mm voxels (the same scale-normalised response),
        # brought back to 1 mm with each 2 mm voxel's area kept in place (grid_mode)
        z, y, x = (n // 2 * 2 for n in img.shape)
        half = img[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean((1, 3, 5), dtype=np.float32)
        hb, hw = _log_max(half, [s / 2.0 for s in sig[n_fine:]], n_fine)
        pad = [(0, n - 2 * h) for n, h in zip(img.shape, hb.shape)]
        hb = np.pad(ndi.zoom(hb, 2, order=1, grid_mode=True, mode="nearest"), pad, mode="edge")
        hw = np.pad(ndi.zoom(hw, 2, order=0, grid_mode=True, mode="nearest"), pad, mode="edge")
        up = hb > best
        best[up] = hb[up]
        which[up] = hw[up]
        del half, hb, hw, up
    peak = (best == ndi.maximum_filter(best, size=PEAK_MM)) & lung & (best > 0)
    zyx = np.argwhere(peak)
    val = best[peak]
    keep = np.argsort(-val, kind="stable")[:top]
    zyx, val = zyx[keep], val[keep]
    radius = np.asarray(RADII_MM)[which[tuple(zyx.T)]]
    lam = hessian_at(img, zyx, radius / math.sqrt(3.0))
    xyz = lp.voxel_to_world(meta, zyx)
    # roundness: the weakest curvature over the strongest, 1 for a ball, 0 for a vessel or a wall (or not a bright blob)
    rnd = np.where(lam[:, 2] < 0, lam[:, 2] / np.minimum(lam[:, 0], -1e-6), 0.0)
    rows = pd.DataFrame({"seriesuid": uid, "coordX": xyz[:, 0], "coordY": xyz[:, 1], "coordZ": xyz[:, 2],
                         "probability": val.astype(np.float64), "radius_mm": radius,
                         "l1": lam[:, 0], "l2": lam[:, 1], "l3": lam[:, 2], "roundness": np.clip(rnd, 0.0, 1.0)})
    return rows, time.time() - t0


def _job(args):
    uid, top, window = args
    return uid, *scan_blobs(uid, top, window)


def build(a):
    if a.nlst:
        if a.folds is not None:
            raise SystemExit("--nlst and --folds cannot be used together: NLST501's scans have no fold")
        import nlst501
        from nodule import scan_folds
        luna, _ = scan_folds(False)
        uids = sorted(u for u, _, _ in nlst501.scans() if nlst501.boxed(u))
        assert not set(uids) & set(luna), "an NLST scan is a LUNA16 scan"
        folds_of = {u: -1 for u in uids}
    else:
        from nodule import scan_folds
        folds_of, _ = scan_folds(False)
        uids = sorted(u for u, f in folds_of.items() if a.folds is None or f in a.folds)
    path = os.path.join(LUNG, a.name, "candidates_oof.csv")
    # every 10 scans go to their own chunk file, written once; the list itself is written once, at the
    # end. Rewriting the whole list every 10 scans wrote about 90 GB for one 710-scan build.
    chunks = os.path.join(LUNG, a.name, "chunks")
    os.makedirs(chunks, exist_ok=True)
    have = [pd.read_csv(path)] if os.path.exists(path) else []
    listed = set(have[0].seriesuid) if have else set()
    for f in sorted(os.listdir(chunks)):
        if f.endswith(".csv"):
            c = pd.read_csv(os.path.join(chunks, f))
            have.append(c[~c.seriesuid.isin(listed)])
    have = [h.rename(columns={"round": "roundness"}) for h in have]
    done = set().union(*(set(h.seriesuid) for h in have))
    todo = [u for u in uids if u not in done]
    print("%d scans, %d already built, %d to build at %d a scan with %d jobs" % (len(uids), len(done), len(todo), a.top, a.jobs),
          flush=True)
    parts, secs = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (uid, rows, s) in enumerate(ex.map(_job, [(u, a.top, a.window) for u in todo]), 1):
            rows["fold"] = folds_of[uid]
            parts.append(rows)
            secs.append(s)
            if i % 10 == 0 or i == len(todo):
                c = os.path.join(chunks, "%s.csv" % parts[0].seriesuid.iloc[0])
                pd.concat(parts, ignore_index=True).to_csv(c + ".part", index=False)
                os.replace(c + ".part", c)
                have.append(pd.concat(parts, ignore_index=True))
                parts = []
                print("  %d of %d, %.1f s a scan" % (i, len(todo), float(np.mean(secs))), flush=True)
    if todo or len(have) > 1:
        # written whole and then swapped in: a reader never sees half a list
        pd.concat(have, ignore_index=True).to_csv(path + ".part", index=False)
        os.replace(path + ".part", path)
    for f in os.listdir(chunks):
        os.remove(os.path.join(chunks, f))
    return 0


def scored(b, expr):
    """The list with `score` = the --score expression over its columns (probability, radius_mm, l1-l3,
    roundness). `round` was the column's first name, and numexpr reads it as its own function."""
    b = b.rename(columns={"round": "roundness"})
    return b.assign(score=b.eval(expr, engine="python").astype(float) if expr else b.probability)


def reach(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "tools"))
    from nodule_depth import reached
    ann, _, _ = froc.load_reference()
    b = scored(pd.read_csv(os.path.join(LUNG, a.name, "candidates_oof.csv")), a.score)
    uids = set(b.seriesuid)
    nod = ann[ann.seriesuid.isin(uids)]
    small = (nod.diameter_mm <= 5).to_numpy()
    b["rank"] = b.groupby("seriesuid").score.rank(ascending=False, method="first")
    other = None
    if a.with_file:
        other = pd.read_csv(a.with_file)
        other = other[other.seriesuid.isin(uids)]
        other = other[other.groupby("seriesuid").probability.rank(ascending=False, method="first") <= a.with_top]
        h = reached(other, nod)
        print("the other list's top %d a scan alone: %d of %d (%.1f in 100), %d of %d at 5 mm or less"
              % (a.with_top, h.sum(), len(nod), 100.0 * h.mean(), h[small].sum(), small.sum()))
    print("%d scans, %d nodules; blobs a scan  reached (in 100)  5 mm or less%s"
          % (len(uids), len(nod), "   joined: reached (in 100)" if other is not None else ""))
    cols = ["seriesuid", "coordX", "coordY", "coordZ"]
    for k in a.depths:
        top = b[b["rank"] <= k]
        h = reached(top, nod)
        line = "  %5d   %4d (%5.1f)   %3d of %3d" % (k, h.sum(), 100.0 * h.mean(), h[small].sum(), small.sum())
        if other is not None:
            j = reached(pd.concat([top[cols], other[cols]]), nod)
            line += "   %4d (%5.1f)" % (j.sum(), 100.0 * j.mean())
        print(line)
    return 0


def join(a):
    from scipy.spatial import cKDTree
    b = scored(pd.read_csv(os.path.join(LUNG, a.name, "candidates_oof.csv")), a.score)
    b = b[b.groupby("seriesuid").score.rank(ascending=False, method="first") <= a.top]
    o = pd.read_csv(a.with_file)
    o = o[o.seriesuid.isin(set(b.seriesuid))]
    # only the scans the list was read on: a scan with blobs and no list would be scored where the
    # list it is compared with has nothing, and stage 2 would train for folds it has no net for
    b = b[b.seriesuid.isin(set(o.seriesuid))]
    o = o[o.groupby("seriesuid").probability.rank(ascending=False, method="first") <= a.with_top]
    cols = ["seriesuid", "coordX", "coordY", "coordZ"]
    xyz = ["coordX", "coordY", "coordZ"]
    scale = 1e-6 / float(b.score.max())
    parts, dropped = [o[cols + ["probability"]].assign(source="list")], 0
    for uid, g in b.groupby("seriesuid"):
        mine = o[o.seriesuid == uid]
        if len(mine):
            d, _ = cKDTree(mine[xyz].to_numpy(float)).query(g[xyz].to_numpy(float))
            far = d > a.merge_mm
            dropped += int((~far).sum())
            g = g[far]
        parts.append(g[cols].assign(probability=g.score * scale, source="blob"))
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    print("%d scans: %d from the list, %d blobs (%d within %.0f mm of the list dropped); %.0f a scan"
          % (out.seriesuid.nunique(), (out.source == "list").sum(), (out.source == "blob").sum(), dropped,
             a.merge_mm, len(out) / out.seriesuid.nunique()))
    return 0


def top_list(a):
    """Each scan's top K blobs by --score, with probability = that score (for nodule_fpr.py --hard-file)."""
    b = scored(pd.read_csv(os.path.join(LUNG, a.name, "candidates_oof.csv")), a.score)
    b = b[b.groupby("seriesuid").score.rank(ascending=False, method="first") <= a.top]
    out = b.drop(columns=["probability"]).rename(columns={"score": "probability"})
    out.to_csv(a.out, index=False)
    print("%d scans, %d blobs (%d a scan) to %s" % (out.seriesuid.nunique(), len(out), a.top, a.out))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--folds", type=int, nargs="+")
    b.add_argument("--nlst", action="store_true",
                   help="build for LIDC-annot-NLST501's prepared scans (fold -1) instead of LUNA16's folds")
    b.add_argument("--top", type=int, default=1000)
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--window", type=float, nargs=2, default=[-1000.0, 400.0], metavar=("LOW_HU", "HIGH_HU"))
    r = sub.add_parser("reach")
    r.add_argument("--depths", type=int, nargs="+", default=[100, 200, 300, 500, 1000])
    r.add_argument("--with", dest="with_file")
    r.add_argument("--with-top", type=int, default=100)
    j = sub.add_parser("join")
    j.add_argument("--with", dest="with_file", required=True)
    j.add_argument("--with-top", type=int, default=100)
    j.add_argument("--top", type=int, default=1000)
    j.add_argument("--merge-mm", type=float, default=4.0)
    j.add_argument("--out", required=True)
    t = sub.add_parser("top")
    t.add_argument("--top", type=int, default=300)
    t.add_argument("--out", required=True)
    for q in (b, r, j, t):
        q.add_argument("--name", default="blobs", help="the list's folder under <WORK>/eval/lung")
    for q in (r, j, t):
        q.add_argument("--score", help="rank blobs by this expression of their columns, e.g. "
                                       "'probability * roundness' (default: probability, the Laplacian's response)")
    a = ap.parse_args()
    return {"build": build, "reach": reach, "join": join, "top": top_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
