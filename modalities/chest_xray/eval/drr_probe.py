"""Idea 43, step 1: LUNA16's CT scans rendered as chest X-rays, read by NIH's chest reader v103 (no training).

A chest film is the body's attenuation summed along each ray; a CT volume holds the same attenuation voxel by
voxel. So summing a lung volume front to back gives a digitally reconstructed radiograph (a DRR). This renders
each of LUNA16's 888 scans (the 1 mm, 8-bit volumes of ct/eval/lung_prepare.py) as a frontal DRR, scores each with
reader-v103's five fold nets, and asks whether the reader tells scans with a nodule of at least 10 mm from scans
with none.

    python modalities/chest_xray/eval/drr_probe.py render [--jobs 6]     processor only
    python modalities/chest_xray/eval/drr_probe.py score [--version N] [--folds 0,1,..]
                                                                         the card, 1.0 GB leased as "drr probe"; another
                                                                         version or folds write scores_vN.csv, report_vN.txt
    python modalities/chest_xray/eval/drr_probe.py selftest              off the card, no real data

render writes, under <WORK>/eval/chest/drr-v1/: drr_384.npy (one uint8 384 x 384 row per scan, in the pixel form
of the cache nih_384.npy, rows in the order of drr_index.csv), drr_index.csv (uid, max_diameter_mm, 0 when none,
and group), and six PNGs (three pos10, three none, drawn with seed 43; the scan's uid is in each name).
Groups: pos10 is a largest nodule of at least 10.0 mm, small is a largest nodule under 10 mm (LUNA16 lists none
under 3 mm), none is a scan with no nodule in annotations.csv.

The picture, per scan:
  1. bytes to HU (0 = -1200, 255 = 600, lung_prepare.to_hu), then to attenuation relative to water,
     mu = max(0, (HU + 1000) / 1000): air 0, lung about 0.2, water 1.0, bone above 1.3 (the 8-bit volume clips it at 600 HU);
  2. mu summed along y, the front-to-back axis of the volume;
  3. oriented as a film is hung: head up, the patient's left on the viewer's right;
  4. windowed into uint16 from the 1st to the 99.5th percentile of the body, the pixels whose summed mu is above
     BODY_MU (5.0, about 5 mm of water; air and scanner padding sum to 0); every other pixel is 0;
  5. xray's own `scaled` and `letterbox` to 384 px, the door every real film goes through (as simsiam.read_png16).

Orientation, from lung_prepare: a volume is (z, y, x), and its docstring and read_scan make the array's x, y and z
rise with the scanner's world x, y and z (a prone scan is flipped to that on reading; slices are sorted by
ascending z; origin_mm is the world place of voxel [0, 0, 0]; world_to_voxel maps world (x, y, z) to voxel (z, y, x)
with only an offset). LUNA16's coordinates are DICOM patient millimetres (LPS): x rises toward the patient's left,
y toward the back, z toward the head. A film is hung with the patient's left on the viewer's right, so the image
columns follow array x as they are (no flip); the head is up, so the image rows run against array z (flip z); y
is the ray. The selftest puts a dense sphere at high x and high z and checks it lands upper right.

score: for each fold net, p_nodule = sigmoid(Nodule logit), p_mass = sigmoid(Mass logit), net score = the larger
of the two; a scan's score is the mean over the five nets (the means of p_nodule and p_mass are kept too), written
to scores.csv. report.txt holds the AUROC of pos10 against none with a 2,000-draw bootstrap 95% CI over scans, the
verdict line, and the reported lines (small against none, Spearman with the largest size, Nodule alone and Mass
alone, and reader-v103's own out-of-fold Nodule and Mass AUROC on NIH as the ceiling). One ledger record.

Idea 43's pass line: the pos10-against-none AUROC is at least 0.60 and its 95% low end is above 0.50. It passes
then step 2 is written; otherwise idea 43 closes. No outside weights and no network are used.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "modalities", "ct", "eval"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from core import paths  # noqa: E402
import lung_prepare as lp  # noqa: E402

# ct/eval has its own prepare.py and datasets.py, and lung_prepare put that folder first: take it out again so
# `import prepare` and reader's `import datasets` find this folder's
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.dirname(lp.__file__))]
sys.path.insert(0, HERE)

SIZE = 384
BODY_MU = 5.0               # summed mu above this is body (see the docstring)
POS_MM = 10.0
PASS_AUROC = 0.60
BOOT = 2000
BOOT_SEED = 43
PNG_SEED = 43
REF_VERSION = 103
VERDICT = "Verdict (idea 43 step 1, DRR AUROC at least 0.60 and its 95% low end above 0.50): "


def out_dir():
    return os.path.join(paths.WORK, "eval", "chest", "drr-v1")


def luna_dir():
    return os.path.join(paths.DATASETS, "public", "luna16")


# ---- the picture ---------------------------------------------------------------------------------------

def mu_table():
    """Attenuation relative to water for each of the 256 bytes: max(0, (HU + 1000) / 1000)."""
    return np.maximum(0.0, (lp.to_hu(np.arange(256, dtype=np.uint8)) + 1000.0) / 1000.0).astype(np.float32)


def project(vol):
    """uint8 volume (z, y, x) to the float32 summed-mu image as a film is hung: rows run head (top) to feet, columns
    run from the patient's right to the patient's left (see the docstring)."""
    lut = mu_table()
    rows = [lut[vol[a:a + 64]].sum(axis=1, dtype=np.float32) for a in range(0, vol.shape[0], 64)]    # (z, x)
    return np.concatenate(rows, axis=0)[::-1]


def to_u16(proj):
    """The summed-mu image windowed into uint16 from the body's 1st to 99.5th percentile; pixels outside the body are 0."""
    body = proj > BODY_MU
    if body.sum() < 100:
        body = np.ones(proj.shape, bool)
    lo, hi = np.percentile(proj[body], [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = np.rint(np.clip((proj - lo) / (hi - lo), 0.0, 1.0) * 65535.0).astype(np.uint16)
    out[~body] = 0
    return out


def xray_prepare():
    import prepare
    return prepare._xray_prepare()


def picture(vol):
    """uint8 volume to the uint8 SIZE x SIZE picture the reader's cache holds."""
    xp = xray_prepare()
    return xp.letterbox(xp.scaled(to_u16(project(vol))), SIZE)


def render_file(npy):
    meta = os.path.splitext(npy)[0] + ".json"
    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    assert m["spacing_mm"] == [1.0, 1.0, 1.0], "%s is not at 1 mm" % npy
    vol = np.load(npy)
    assert list(vol.shape) == m["shape"], "%s: shape differs from its json" % npy
    return picture(vol)


def _render_worker(uid):
    return render_file(lp.paths_for(lp.out_dir(), uid)[0])


# ---- the scans and their groups ---------------------------------------------------------------------------

def groups(ann, uids):
    """DataFrame uid, max_diameter_mm (0 when none), group (pos10, small, none) for each uid, in order."""
    import pandas as pd
    big = ann.groupby("seriesuid").diameter_mm.max()
    d = np.array([float(big.get(u, 0.0)) for u in uids])
    g = np.where(d >= POS_MM, "pos10", np.where(d > 0, "small", "none"))
    return pd.DataFrame({"uid": list(uids), "max_diameter_mm": d, "group": g})


def luna_table():
    import pandas as pd
    sub = pd.read_csv(os.path.join(luna_dir(), "subsets.csv"))
    ann = pd.read_csv(os.path.join(luna_dir(), "annotations.csv"))
    uids = sorted(set(sub.seriesuid))
    assert len(uids) == 888, "subsets.csv lists %d scans, not 888" % len(uids)
    df = groups(ann, uids)
    missing = [u for u in uids if not (os.path.exists(lp.paths_for(lp.out_dir(), u)[0])
                                       and os.path.exists(lp.paths_for(lp.out_dir(), u)[1]))]
    assert not missing, "%d of the 888 scans have no volume (first: %s)" % (len(missing), missing[0])
    return df


# ---- statistics ---------------------------------------------------------------------------------------------

def auroc(y, s):
    """Area under the ROC curve (ties count half); nan when one class is empty."""
    from scipy.stats import rankdata
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_auroc(y, s, n=BOOT, seed=BOOT_SEED):
    """(AUROC, low, high): the AUROC and the 2.5th and 97.5th percentiles of n bootstrap draws of scans."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    g = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        i = g.integers(0, len(y), len(y))
        v = auroc(y[i], s[i])
        if v == v:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auroc(y, s), float(lo), float(hi)


def verdict_line(auc, low):
    return VERDICT + ("PASSES" if auc >= PASS_AUROC and low > 0.5 else "FAILS")


# ---- render -----------------------------------------------------------------------------------------------------

def cmd_render(jobs):
    from PIL import Image
    df = luna_table()
    uids = list(df.uid)
    odir = out_dir()
    os.makedirs(odir, exist_ok=True)
    npy = os.path.join(odir, "drr_384.npy")
    part = npy + ".part"
    arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(len(uids), SIZE, SIZE))
    t0 = time.time()
    if jobs == 1:
        results = map(_render_worker, uids)
        pool = None
    else:
        pool = ProcessPoolExecutor(jobs)
        results = pool.map(_render_worker, uids, chunksize=2)
    try:
        for i, pic in enumerate(results):
            arr[i] = pic
            if (i + 1) % 50 == 0:
                print("%d/%d rendered (%.0f s)" % (i + 1, len(uids), time.time() - t0), flush=True)
    finally:
        if pool:
            pool.shutdown()
    arr.flush()
    del arr
    os.replace(part, npy)
    df.to_csv(os.path.join(odir, "drr_index.csv"), index=False)
    pix = np.load(npy, mmap_mode="r")
    g = np.random.default_rng(PNG_SEED)
    for grp in ("pos10", "none"):
        rows = np.flatnonzero(df.group.to_numpy() == grp)
        for r in sorted(g.choice(rows, size=3, replace=False)):
            Image.fromarray(np.asarray(pix[r])).save(os.path.join(odir, "drr_%s_%s.png" % (grp, uids[r])))
    print("rendered %d scans -> %s (%.0f s); groups %s" % (
        len(uids), npy, time.time() - t0, df.group.value_counts().to_dict()))
    return 0


# ---- score ------------------------------------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


ALL_FOLDS = (0, 1, 2, 3, 4)


def cmd_score(version=REF_VERSION, folds=ALL_FOLDS):
    """Score the DRRs with reader-v<version>'s nets for `folds`. The defaults (v103, all five) write scores.csv and report.txt as
    step 1 did; any other version or folds write scores_v<N>.csv and report_v<N>.txt instead, and leave out the ceiling line, which
    reads the version's five-fold out-of-fold file."""
    import pandas as pd
    from scipy.stats import spearmanr
    folds = tuple(folds)
    full = folds == ALL_FOLDS
    stem = "" if version == REF_VERSION and full else "_v%d" % version
    odir = out_dir()
    df = pd.read_csv(os.path.join(odir, "drr_index.csv"))
    assert len(df) == 888, "drr_index.csv holds %d scans" % len(df)
    pix = np.load(os.path.join(odir, "drr_384.npy"), mmap_mode="r")
    assert pix.shape == (888, SIZE, SIZE) and pix.dtype == np.uint8, "drr_384.npy is %s %s" % (pix.shape, pix.dtype)
    ref = os.path.join(paths.WORK, "eval", "chest", "reader-v%d" % version)
    import reader                       # starts torch; the card is leased before CUDA starts
    import torch
    labels = reader.datasets.LABELS
    j_nod, j_mass = labels.index("Nodule"), labels.index("Mass")
    with open(os.path.join(ref, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["cfg"]["widths"] == [32, 64, 128, 256, 256] and meta["cfg"]["stem_stride"] == 2, meta["cfg"]
    assert meta["cfg"]["widths"] == reader.CFG["widths"] and meta["cfg"]["stem_stride"] == reader.CFG["stem_stride"]
    reader.card.require_card(reader.DEV, torch)
    reader.xd.limit_card_memory(1.0, "drr probe")
    rows = np.arange(len(df))
    cache_row = np.arange(len(df), dtype=np.int64)
    blank = np.zeros((len(df), len(labels)), np.float32)
    p_nod, p_mass, p_net = [], [], []
    for k in folds:
        ck = torch.load(os.path.join(ref, "fold%d.pt" % k), map_location=reader.DEV, weights_only=False)
        net = reader.build_net(ck["settings"], len(labels)).to(reader.DEV)
        net.load_state_dict(ck["state"])
        mean = torch.tensor(ck["info"]["mean"], dtype=torch.float32, device=reader.DEV)
        sd = torch.tensor(ck["info"]["sd"], dtype=torch.float32, device=reader.DEV)
        z = reader.predict(net, pix, cache_row, blank, rows, mean, sd, 64, SIZE)
        a, b = sigmoid(z[:, j_nod]), sigmoid(z[:, j_mass])
        p_nod.append(a)
        p_mass.append(b)
        p_net.append(np.maximum(a, b))
        print("fold %d scored" % k, flush=True)
        del net
    df["score"] = np.mean(p_net, axis=0)
    df["p_nodule"] = np.mean(p_nod, axis=0)
    df["p_mass"] = np.mean(p_mass, axis=0)
    df.to_csv(os.path.join(odir, "scores%s.csv" % stem), index=False)

    pos = (df.group == "pos10").to_numpy()
    none = (df.group == "none").to_numpy()
    small = (df.group == "small").to_numpy()

    def vs_none(mask, col):
        keep = mask | none
        return boot_auroc(mask[keep].astype(int), df[col].to_numpy()[keep])

    auc, lo, hi = vs_none(pos, "score")
    lines = ["pos10 against none (%d against %d scans): AUROC %.4f (95%% CI %.4f to %.4f, %d bootstrap draws over scans)" % (
        pos.sum(), none.sum(), auc, lo, hi, BOOT), verdict_line(auc, lo)]
    sa, slo, shi = vs_none(small, "score")
    lines.append("small (3 to under 10 mm) against none (%d against %d scans): AUROC %.4f (95%% CI %.4f to %.4f)" % (
        small.sum(), none.sum(), sa, slo, shi))
    rho = float(spearmanr(df.score, df.max_diameter_mm)[0])
    lines.append("Spearman correlation of the score with the largest nodule's size, all %d scans: %.4f" % (len(df), rho))
    na, nlo, nhi = vs_none(pos, "p_nodule")
    ma, mlo, mhi = vs_none(pos, "p_mass")
    lines.append("pos10 against none, Nodule alone: AUROC %.4f (95%% CI %.4f to %.4f)" % (na, nlo, nhi))
    lines.append("pos10 against none, Mass alone: AUROC %.4f (95%% CI %.4f to %.4f)" % (ma, mlo, mhi))
    ceil = None
    if full:                        # the out-of-fold file of a version trained on all five folds
        oof = np.load(os.path.join(ref, "oof_logits.npz"))
        ceil = {n: auroc(oof["y"][:, labels.index(n)], oof["logit"][:, labels.index(n)]) for n in ("Nodule", "Mass")}
        lines.append("ceiling, reader-v%d's own out-of-fold AUROC on NIH (%d pictures): Nodule %.4f, Mass %.4f" % (
            version, len(oof["y"]), ceil["Nodule"], ceil["Mass"]))
    else:
        lines.append("read by reader-v%d's fold net(s) %s only (no ceiling line: it needs all five folds)" % (
            version, ", ".join(str(k) for k in folds)))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(odir, "report%s.txt" % stem), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    from core import ledger
    with ledger.run("experiment", "drr probe: LUNA16 DRRs read by reader v%d (idea 43 step 1)" % version) as led:
        led.settings_from({"reader": version, "scans": len(df), "size": SIZE, "body_mu": BODY_MU,
                           "pos_mm": POS_MM, "bootstrap": BOOT, "bootstrap_seed": BOOT_SEED,
                           **({} if full else {"folds": list(folds)})})
        led.number("auroc_pos10", auc).number("auroc_pos10_low", lo).number("auroc_pos10_high", hi)
        led.number("auroc_small", sa).number("auroc_small_low", slo).number("auroc_small_high", shi)
        led.number("spearman_size", rho).number("auroc_pos10_nodule", na).number("auroc_pos10_mass", ma)
        if ceil is not None:
            led.number("ceiling_nodule", ceil["Nodule"]).number("ceiling_mass", ceil["Mass"])
        led.note_is(lines[1])
    return 0


# ---- selftest ---------------------------------------------------------------------------------------------------

def _phantom(sphere):
    """uint8 volume (z 80, y 50, x 100): air round a body box at 40 HU holding lungs at -800 HU, and, when
    asked, a 10 mm sphere at 300 HU at high x (the patient's left) and high z (superior)."""
    shape = (80, 50, 100)
    hu = np.full(shape, -1000.0, np.float32)
    hu[10:70, 5:45, 10:90] = 40.0
    hu[14:66, 9:41, 14:86] = -800.0
    if sphere:
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        hu[(z - 60) ** 2 + (y - 25) ** 2 + (x - 75) ** 2 <= 25.0] = 300.0
    return lp.to_uint8(hu)


def selftest():
    import pandas as pd
    # the bytes-to-mu map
    mu = mu_table()
    assert mu[0] == 0.0
    near0 = int(np.argmin(np.abs(lp.to_hu(np.arange(256, dtype=np.uint8)))))
    assert abs(mu[near0] - 1.0) <= lp.STEP_HU / 1000.0 + 1e-6, mu[near0]
    # orientation: the sphere's own contribution lands upper right
    with_s, without = project(_phantom(True)), project(_phantom(False))
    diff = with_s - without
    r, c = np.unravel_index(int(np.argmax(diff)), diff.shape)
    assert diff[r, c] > 5.0, diff.max()
    assert c >= diff.shape[1] // 2, "the patient's left sphere is in the viewer's left half (column %d of %d)" % (c, diff.shape[1])
    assert r < diff.shape[0] // 2, "the superior sphere is in the bottom half (row %d of %d)" % (r, diff.shape[0])
    # the whole path from a volume file to the cache's pixel form
    with tempfile.TemporaryDirectory(prefix="drr_probe_") as tmp:
        vol = _phantom(True)
        npy = os.path.join(tmp, "toy.npy")
        np.save(npy, vol)
        with open(os.path.join(tmp, "toy.json"), "w", encoding="utf-8") as f:
            json.dump({"spacing_mm": [1.0, 1.0, 1.0], "shape": list(vol.shape)}, f)
        pic = render_file(npy)
        assert pic.shape == (SIZE, SIZE) and pic.dtype == np.uint8 and pic.max() > 0
    # the AUROC and bootstrap helper
    y = np.array([0] * 20 + [1] * 20)
    for s, want in ((np.arange(40.0), 1.0), (-np.arange(40.0), 0.0)):
        a, lo, hi = boot_auroc(y, s, n=200)
        assert a == want and lo <= a <= hi, (a, lo, hi)
    # the groups
    ann = pd.DataFrame({"seriesuid": ["a", "a", "b", "c"], "diameter_mm": [3.5, 10.0, 9.99, 4.0]})
    g = groups(ann, ["a", "b", "c", "d"])
    assert list(g.group) == ["pos10", "small", "small", "none"], list(g.group)
    assert list(g.max_diameter_mm) == [10.0, 9.99, 4.0, 0.0]
    # the verdict line's two thresholds
    assert verdict_line(0.61, 0.52).endswith(": PASSES")
    assert verdict_line(0.61, 0.49).endswith(": FAILS")
    assert verdict_line(0.59, 0.52).endswith(": FAILS")
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--jobs", type=int, default=6)
    s = sub.add_parser("score")
    s.add_argument("--version", type=int, default=REF_VERSION, help="reader version to score with (default %d)" % REF_VERSION)
    s.add_argument("--folds", default="0,1,2,3,4", help="comma-separated fold nets to use (default 0,1,2,3,4)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "render":
        return cmd_render(a.jobs)
    if a.cmd == "score":
        folds = tuple(int(f) for f in a.folds.split(",") if f.strip() != "")
        if not folds or len(set(folds)) != len(folds) or any(not 0 <= f <= 4 for f in folds):
            ap.error("--folds must be distinct fold numbers 0-4, got %r" % a.folds)
        return cmd_score(a.version, tuple(sorted(folds)))
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
