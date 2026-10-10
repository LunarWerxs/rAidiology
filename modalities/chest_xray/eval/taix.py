"""A chest reader for TAIX-Ray's bedside pictures (docs/efficiency idea 26): eight findings, graded 0-4 by the
reporting radiologist, read from one front picture by reader.py's from-scratch trunk. No pretrained weights.

  prepare                 decode the 215,381 pictures once into taix_384.u8 (uint8, N x 384 x 384, aspect kept, padded
                          black) plus taix_384.csv (row, UID, PatientID, Split, the eight grades); resumable
  train --version N       train on the train split minus 10% of its patients (the early-stop set), keep the best epoch,
                          score the val split, print per-finding AUROC with a patient bootstrap and the verdict line
  read --versions 1,2     val verdict of the mean of those versions' saved present probabilities (no card, no training)
  compare --a 1,2 --b 3|4 side B against side A on val: per finding and the mean of the eight, a paired patient bootstrap,
                          verdict (no card, no training); '|' starts a group, each group's mean weighs the same
  test --versions 1,2     the same read on the test split, after the read's val verdict was OPENS TEST (once, as below)
  test --version N        only after N's val verdict was OPENS TEST and only once for all versions: score the test split
                          against the paper (Truhn et al., Sci Data 2026, Table 4)
  selftest                CPU, seconds

A finding is present at any grade above none (heart: above normal): the paper's Fig. 4 rule, BCE on all eight.

Where this differs from reader.py's recipe, and why:
- augmentation is detector.augment WITHOUT its horizontal flip: a flip would swap the left and right findings;
- reader.train_fold is tied to NIH's table, so the loop here is its own (same trunk, loader, normalising, schedule,
  patience, partial checkpoints), with bfloat16 autocast on by default.

Usage: python taix.py prepare [--jobs 6] [--size 512]     (512: taix_512.u8, the pictures' own size; default 384)
       python taix.py train --version 1 [--epochs 12] [--batch 64] [--set key=<json> ...]   (amp, size, renderings, pos_weight, init, cache_size)
       --set cache_size=512 trains and reads from taix_512.u8 (absent is the 384 cache)
       Idea 26 step 3: --set 'init="chest-v103"' starts the trunk from chest reader v103 fold 0.
       python taix.py test --version 1
       python taix.py compare --a 1,2,3,4,6,7 --b "1,2,3,4,6,7|14" [--margin 0.01] [--name TEXT]
       A version trained by mae.py finetune-taix (settings arch 'vit-s16') is rebuilt and read at its own px wherever
       a saved net is loaded (train's scoring-only path, test); the CNN versions read as before.
"""
import argparse
import csv
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from core import paths  # noqa: E402
from core.rows import FileRows  # noqa: E402

DATA = os.path.join(paths.DATASETS, "public", "taix-ray", "data")
CACHE = os.path.join(paths.WORK, "eval", "chest")
SIZE = 384
U8 = os.path.join(CACHE, "taix_384.u8")
CSV = os.path.join(CACHE, "taix_384.csv")
META = os.path.join(CACHE, "taix_384.json")
PROGRESS = os.path.join(CACHE, "taix_384.progress")
CACHE_SIZES = (384, 512)
CSV_384 = CSV


def cache_paths(size):
    """(u8, csv, meta, progress) of the picture cache of this side; 384 is the constants above."""
    return tuple(os.path.join(CACHE, "taix_%d.%s" % (size, e)) for e in ("u8", "csv", "json", "progress"))


def cache_size_of(settings):
    return (settings or {}).get("cache_size", SIZE)


def read_px(settings):
    """The side a saved net reads at: a mae.py ViT-S's own px, else the settings' size, else its cache's side."""
    if settings.get("arch") == "vit-s16":
        return int(settings["px"])
    return int(settings.get("size", cache_size_of(settings)))


MARKER = os.path.join(CACHE, "taix_test_opened.json")
SPLITS = ("train", "val", "test")
# the eight findings, in the order of the logits; column in the parquet files, paper's test AUROC (Table 4)
FINDINGS = [("heart", "HeartSize", 0.87), ("congestion", "PulmonaryCongestion", 0.80),
            ("effusion_left", "PleuralEffusion_Left", 0.89), ("effusion_right", "PleuralEffusion_Right", 0.91),
            ("opacities_left", "PulmonaryOpacities_Left", 0.84), ("opacities_right", "PulmonaryOpacities_Right", 0.85),
            ("atelectasis_left", "Atelectasis_Left", 0.81), ("atelectasis_right", "Atelectasis_Right", 0.83)]
NAMES = [f[0] for f in FINDINGS]
COLUMNS = [f[1] for f in FINDINGS]
PAPER = np.array([f[2] for f in FINDINGS])
OPEN_BAR = 0.85
VAL_SHARE = 0.10
SEED = 20261005
BOOT, BOOT_SEED = 1000, 0
MAX_JOBS = 6
SETTINGS = {
    "amp": "bfloat16 autocast in the training step; on by default here, --set amp=false for float32",
    "size": "train and read at a smaller square (e.g. 256), area-resampled on the card",
    "renderings": "3: reader.py's rank-equalised and high-pass channels; 1 is the recipe's",
    "pos_weight": "BCE pos_weight per finding = negatives/positives of the training rows, clipped to [1, 20]",
    "ordinal": "true: learn the grades as cumulative logits (31 outputs); present = step 0; default false is v1",
    "cache_size": "512: train and read from taix_512.u8 (the pictures' own size); absent is the 384 cache",
    "init": "our own from-scratch chest reader whose trunk starts this net: \"chest-v103\", or idea 40's self-taught trunk "
            "\"chest-ssl-v1\" (last epoch) or \"chest-ssl-v1-e15\" (epoch 15); absent by default; the head stays fresh",
    "train_share": "a share in (0, 1] of the train split's training PATIENTS kept (seeded, the same patients for every version), "
                   "after the early-stop split; the early-stop rows stay as they are; absent is all of them (idea 40's gate A uses 0.1)",
}
# the card share a TAIX process leases and caps itself at: 3.5 GB, or RAIDIOLOGY_TAIX_CARD_GB (idea 39 at 512 px sets it
# from the bench's reserved peak plus 1 GB)
CARD_GB = float(os.environ.get("RAIDIOLOGY_TAIX_CARD_GB", "3.5"))
INITS = {"chest-v103": os.path.join(paths.WORK, "eval", "chest", "reader-v103", "fold0.pt"),
         "chest-ssl-v1": os.path.join(paths.WORK, "eval", "chest", "ssl-v1", "trunk.pt"),
         "chest-ssl-v1-e15": os.path.join(paths.WORK, "eval", "chest", "ssl-v1", "trunk_e15.pt")}
# The ordinal head, finding-major in FINDINGS' order: heart has 3 steps (grades 0-3), the other seven 4 (grades 0-4).
# Output o of finding f's block is the logit of "grade > o"; so the 31 outputs are heart 0-2, congestion 0-3, ...
STEPS = [3] + [4] * 7
OFFSETS = np.concatenate([[0], np.cumsum(STEPS)]).astype(int)
STEP0 = OFFSETS[:-1]                    # column of each finding's first step: its "present" logit
N_ORDINAL = int(OFFSETS[-1])            # 31


def out_dir(version):
    return os.path.join(CACHE, "taix-v%d" % version)


def read_label(versions):
    return "-".join(str(v) for v in versions)


def read_dir(versions):
    """Where a two-net read keeps its val verdict and test logits."""
    return os.path.join(CACHE, "taix-read-%s" % read_label(versions))


def parse_versions(text):
    vs = [int(x) for x in str(text).split(",") if x.strip()]
    if len(vs) < 2 or len(set(vs)) != len(vs):
        raise SystemExit("--versions wants two or more different versions, e.g. 1,2 (got %r)" % text)
    return vs


# ---------------------------------------------------------------- the rule and the splits (pure, selftested)

def present(grades):
    """(n, 8) grades 0-4 to the 0/1 truth: any grade above none (heart: above normal) is present."""
    return (np.asarray(grades) > 0).astype(np.int8)


def ordinal_targets(grades):
    """(n, 8) grades to (n, 31) 0/1 targets: step j of finding f is 1 when grade > j (heart j = 0..2, others 0..3)."""
    g = np.asarray(grades)
    out = np.zeros((len(g), N_ORDINAL), np.int8)
    for f, k in enumerate(STEPS):
        for j in range(k):
            out[:, OFFSETS[f] + j] = g[:, f] > j
    return out


def present_logits(z, ordinal):
    """The eight 'present' logits: the net's output as is, or each finding's step-0 logit (P(grade > 0))."""
    z = np.asarray(z)
    return z[:, STEP0] if ordinal else z


def grades_from_logits(z):
    """(n, 31) cumulative logits to (n, 8) grades: the number of steps whose cumulative probability exceeds 0.5
    (logit > 0). A count, not a walk up the steps, so the read is monotone even if the steps disagree."""
    z = np.asarray(z)
    return np.stack([(z[:, OFFSETS[f]:OFFSETS[f + 1]] > 0).sum(axis=1) for f in range(len(STEPS))], axis=1)


def quadratic_kappa(true, pred, k):
    """Quadratic weighted kappa of integer grades 0..k-1; nan when it is undefined."""
    true, pred = np.asarray(true, int), np.asarray(pred, int)
    if len(true) == 0:
        return float("nan")
    obs = np.zeros((k, k))
    np.add.at(obs, (true, pred), 1.0)
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / len(true)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / float(k - 1) ** 2
    den = (w * exp).sum()
    return float("nan") if den == 0 else float(1.0 - (w * obs).sum() / den)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, np.float64), -60, 60)))


def prob_logit(p):
    """A probability as a logit, clipped away from 0 and 1."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def mean_prob_logits(logit_list):
    """The two-net (or n-net) read: sigmoid each net's present logits, take the mean, return it as a logit."""
    return prob_logit(np.mean([sigmoid(z) for z in logit_list], axis=0))


def load_val(version, cache=CACHE):
    z = np.load(os.path.join(cache, "taix-v%d" % version, "val_logits.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def read_val(versions, cache=CACHE):
    """(UID, PatientID, truth, mean-of-probabilities logits) of the listed versions' saved val logits; stops unless
    their rows are the same rows in the same order."""
    vals = [load_val(v, cache) for v in versions]
    for v, z in zip(versions[1:], vals[1:]):
        for key in ("UID", "PatientID", "truth"):
            assert np.array_equal(vals[0][key], z[key]), "version %d's val %s differ from version %d's" % (v, key, versions[0])
    return vals[0]["UID"], vals[0]["PatientID"], vals[0]["truth"], mean_prob_logits([z["logits"] for z in vals])


def early_stop_split(patients, rows, share=VAL_SHARE, seed=SEED):
    """(train rows, early-stop rows): a seeded `share` of the PATIENTS among `rows` go to early stopping, so no
    patient has pictures on both sides."""
    patients = np.asarray(patients)
    rows = np.asarray(rows)
    uniq = np.array(sorted(set(patients[rows].tolist())))
    held = set(uniq[np.random.default_rng(seed).permutation(len(uniq))[:int(round(share * len(uniq)))]].tolist())
    is_held = np.array([p in held for p in patients[rows].tolist()], dtype=bool)
    return rows[~is_held], rows[is_held]


def share_of_patients(patients, rows, share):
    """The `rows` of a seeded round(share * n) of their sorted unique patients: the same patients every time."""
    patients = np.asarray(patients)
    rows = np.asarray(rows)
    uniq = np.array(sorted(set(patients[rows].tolist())))
    keep = set(uniq[np.random.default_rng(SEED + 40).permutation(len(uniq))[:int(round(share * len(uniq)))]].tolist())
    return rows[np.array([p in keep for p in patients[rows].tolist()], dtype=bool)]


def val_verdict_line(mean):
    return "Verdict (idea 26 step 1, mean val AUROC at least %.2f): %s" % (
        OPEN_BAR, "OPENS TEST" if mean >= OPEN_BAR else "STAYS ON VAL")


def judge(lo, hi, paper):
    """BEATEN when the interval's low end is above the paper's number, BEHIND when its top is below, else LEVEL."""
    return "BEATEN" if lo > paper else "BEHIND" if hi < paper else "LEVEL"


def test_gate(version, out, marker):
    """Stops unless version's val verdict (a version number, or a read's label like "1-2" with its directory as `out`)
    was OPENS TEST and no test marker exists for any version or read."""
    if os.path.exists(marker):
        try:
            who = json.load(open(marker, encoding="utf-8")).get("version")
        except (OSError, ValueError):
            who = "?"
        raise SystemExit("the test split was already opened (version %s, %s); it opens once" % (who, marker))
    vp = os.path.join(out, "val_verdict.json")
    if not os.path.exists(vp):
        raise SystemExit("version %s has no val verdict (%s); run train (or read) first" % (version, vp))
    v = json.load(open(vp, encoding="utf-8"))
    if v.get("verdict") != "OPENS TEST":
        raise SystemExit("version %s's val verdict is %s (mean %.4f), not OPENS TEST" % (version, v.get("verdict"), v.get("mean", float("nan"))))


def write_marker(version, marker):
    """The marker goes down before any test picture is read; creating it fails if one exists."""
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "x", encoding="utf-8") as f:
        json.dump({"version": version, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


# ---------------------------------------------------------------- AUROC with a patient bootstrap

def _auroc_weighted(order, gid, y, w):
    """AUROC of the pictures counted w times each (order: ascending score, gid: tie group of each sorted picture)."""
    ys, ws = y[order], w[order]
    ng = int(gid[-1]) + 1
    gp = np.bincount(gid, weights=ws * ys, minlength=ng)
    gn = np.bincount(gid, weights=ws * (1 - ys), minlength=ng)
    wp, wn = gp.sum(), gn.sum()
    if wp == 0 or wn == 0:
        return float("nan")
    below = np.cumsum(gn) - gn
    return float((gp * (below + 0.5 * gn)).sum() / (wp * wn))


def bootstrap_auroc(logits, truth, patients, n=BOOT, seed=BOOT_SEED):
    """(point, lo, hi) per finding: AUROC and its 95% interval over `n` resamples of the PATIENTS (all of a drawn
    patient's pictures, as many times as drawn)."""
    logits, truth = np.asarray(logits, float), np.asarray(truth)
    _, pid = np.unique(np.asarray(patients), return_inverse=True)
    n_pat = int(pid.max()) + 1
    rng = np.random.default_rng(seed)
    counts = [np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat) for _ in range(n)]
    out = []
    for j in range(logits.shape[1]):
        order = np.argsort(logits[:, j], kind="mergesort")
        s = logits[order, j]
        gid = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        y = truth[:, j].astype(float)
        point = _auroc_weighted(order, gid, y, np.ones(len(y)))
        boots = np.array([_auroc_weighted(order, gid, y, c[pid].astype(float)) for c in counts])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (float("nan"),) * 2
        out.append((point, float(lo), float(hi)))
    return out


def _prep(logits, truth):
    """Per finding: (order, gid, y) with the scores sorted once, for _auroc_weighted."""
    out = []
    for j in range(logits.shape[1]):
        order = np.argsort(logits[:, j], kind="mergesort")
        s = logits[order, j]
        gid = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        out.append((order, gid, truth[:, j].astype(float)))
    return out


def paired_auroc(la, lb, truth, patients, n=BOOT, seed=BOOT_SEED):
    """(points of A, points of B, n resamples of B minus A), each over the eight findings then their mean. A resample
    draws the patients once for both sides (a patient drawn k times counts its pictures k times, as bootstrap_auroc does)."""
    _, pid = np.unique(np.asarray(patients), return_inverse=True)
    n_pat = int(pid.max()) + 1
    rng = np.random.default_rng(seed)
    pa, pb = _prep(la, truth), _prep(lb, truth)
    ones = np.ones(len(pid))
    pts_a = np.array([_auroc_weighted(*p, ones) for p in pa])
    pts_b = np.array([_auroc_weighted(*p, ones) for p in pb])
    diffs = np.empty((n, len(NAMES) + 1))
    for i in range(n):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)[pid].astype(float)
        a_i = np.array([_auroc_weighted(*p, w) for p in pa])
        b_i = np.array([_auroc_weighted(*p, w) for p in pb])
        diffs[i] = np.append(b_i - a_i, b_i.mean() - a_i.mean())
    return np.append(pts_a, pts_a.mean()), np.append(pts_b, pts_b.mean()), diffs


def parse_side(text):
    """'1,2,3|14' to [[1, 2, 3], [14]]: groups split by '|', versions by ','. Stops on an empty group or a repeat."""
    groups = [[int(x) for x in g.split(",") if x.strip()] for g in str(text).split("|")]
    flat = [v for g in groups for v in g]
    if not flat or not all(groups) or len(set(flat)) != len(flat):
        raise SystemExit("a compare side is groups of versions, e.g. 1,2,3|14, each version once (got %r)" % text)
    return groups


def side_probs(groups, vals):
    """(n, 8) probabilities of one side: the mean over its groups of (the mean over a group's versions of the sigmoid of
    their val logits). So '1,2|3' weighs version 3 as much as versions 1 and 2 together."""
    return np.mean([np.mean([sigmoid(vals[v]["logits"]) for v in g], axis=0) for g in groups], axis=0)


def compare_sides(a, b, margin=0.01, cache=CACHE, n=BOOT, seed=BOOT_SEED):
    """(rows, verdict, pictures, patients): rows are (label, A, B, B - A, low, high) for each finding, then their mean.
    Stops unless every version's val rows are the same pictures in the same order with the same truth."""
    ga, gb = parse_side(a), parse_side(b)
    versions = list(dict.fromkeys(v for g in ga + gb for v in g))
    vals = {v: load_val(v, cache) for v in versions}
    ref = versions[0]
    for v in versions[1:]:
        for key in ("UID", "PatientID", "truth"):
            if not np.array_equal(vals[ref][key], vals[v][key]):
                raise SystemExit("version %d's val %s differ from version %d's: not the same pictures" % (v, key, ref))
    truth, patient = vals[ref]["truth"], vals[ref]["PatientID"]
    pts_a, pts_b, diffs = paired_auroc(prob_logit(side_probs(ga, vals)), prob_logit(side_probs(gb, vals)),
                                       truth, patient, n, seed)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5], axis=0)
    labels = NAMES + ["mean of the eight"]
    rows = [(lab, va, vb, vb - va, float(l), float(h)) for lab, va, vb, l, h in zip(labels, pts_a, pts_b, lo, hi)]
    low, top = rows[-1][4], rows[-1][5]
    verdict = "BETTER" if low > 0 else "NOT WORSE" if low > -margin else "WORSE" if top < 0 else "INCONCLUSIVE"
    return rows, verdict, len(patient), len(set(patient.tolist()))


# ---------------------------------------------------------------- prepare

def decode_bytes(data, size):
    """One encoded picture to a size x size uint8: the X-ray module's window and letterbox (as prepare.py does for
    NIH), so chest and bone pictures meet one intensity rule. 16-bit PNGs are what TAIX-Ray ships."""
    from PIL import Image
    import prepare as chest_prepare            # chest_xray/eval/prepare.py; loads the X-ray module's by path
    xp = chest_prepare._xray_prepare()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        arr = np.asarray(im)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1).astype(np.float32) if arr.dtype.kind == "f" else arr[..., 0]
    return xp.letterbox(xp.scaled(arr), size)


def parquet_files(split):
    import glob
    return sorted(glob.glob(os.path.join(DATA, "%s-*.parquet" % split)))


def plan():
    """Row groups in row order: [(path, group, start, n)], and the table rows [(UID, PatientID, Split, 8 grades)].
    Reads the label columns only, never an Image column."""
    import pyarrow.parquet as pq
    tasks, rows = [], []
    for split in SPLITS:
        files = parquet_files(split)
        if not files:
            raise SystemExit("no %s-*.parquet files in %s" % (split, DATA))
        for path in files:
            pf = pq.ParquetFile(path)
            lab = pf.read(columns=["UID", "PatientID", "Split"] + COLUMNS).to_pydict()
            at = len(rows)
            for i in range(len(lab["UID"])):
                rows.append((lab["UID"][i], lab["PatientID"][i], lab["Split"][i], [lab[c][i] for c in COLUMNS]))
            g0 = 0
            for g in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                tasks.append((path, g, at + g0, n))
                g0 += n
            assert g0 == len(rows) - at, path
    return tasks, rows


def check_rows(rows):
    """Prints the counts and stops when a PatientID is in two splits or a UID repeats."""
    uids = [r[0] for r in rows]
    if len(set(uids)) != len(uids):
        raise SystemExit("UIDs repeat: %d rows, %d distinct" % (len(uids), len(set(uids))))
    seen = {}
    for s in SPLITS:
        pats = {r[1] for r in rows if r[2] == s}
        n = sum(1 for r in rows if r[2] == s)
        print("  %-5s %7d pictures  %6d patients" % (s, n, len(pats)), flush=True)
        for p in pats:
            assert p not in seen, "PatientID %s is in both %s and %s" % (p, seen[p], s)
            seen[p] = s
    bad = [r[2] for r in rows if r[2] not in SPLITS]
    assert not bad, "unknown Split values: %s" % sorted(set(bad))
    for r in rows:
        assert all(0 <= g <= 4 for g in r[3]), "grade out of 0-4 in %s" % r[0]


_W = {}


def _work(task):
    path, g, start, n = task
    import pyarrow.parquet as pq
    size = _W["size"]
    if "mm" not in _W:
        _W["mm"] = np.memmap(_W["u8"], dtype=np.uint8, mode="r+", shape=(_W["n"], size, size))
    pf = _W.setdefault("pf", {}).get(path) or _W["pf"].setdefault(path, pq.ParquetFile(path))
    col = pf.read_row_group(g, columns=["Image"]).column("Image").to_pylist()
    bad = []
    for i, im in enumerate(col):
        try:
            _W["mm"][start + i] = decode_bytes(im["bytes"], size)
        except Exception as e:  # noqa: BLE001  one bad picture stays zeros and is listed; the build goes on
            bad.append((start + i, "%s: %s" % (type(e).__name__, e)))
    _W["mm"].flush()
    return start, n, bad


def _init(n, size, u8):
    _W["n"] = n
    _W["size"] = size
    _W["u8"] = u8


def prepare(jobs=MAX_JOBS, size=SIZE):
    from concurrent.futures import ProcessPoolExecutor
    if size not in CACHE_SIZES:
        raise SystemExit("--size must be 384 or 512, not %r" % (size,))
    U8, CSV, META, PROGRESS = cache_paths(size)
    t0 = time.time()
    jobs = max(1, min(int(jobs), MAX_JOBS))
    os.makedirs(CACHE, exist_ok=True)
    tasks, rows = plan()
    n = len(rows)
    print("TAIX-Ray: %d pictures in %d row groups" % (n, len(tasks)), flush=True)
    check_rows(rows)
    with open(CSV + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "UID", "PatientID", "Split"] + COLUMNS)
        for i, r in enumerate(rows):
            w.writerow([i, r[0], r[1], r[2]] + r[3])
    if size != SIZE and os.path.exists(CSV_384):
        import pandas as pd
        if pd.read_csv(CSV_384, usecols=["UID"]).UID.astype(str).tolist() != [str(r[0]) for r in rows]:
            os.remove(CSV + ".part")
            raise SystemExit("the %d cache's UID column differs from %s row for row; a version's rows must mean the same "
                             "pictures at either size, so nothing was built" % (size, CSV_384))
    os.replace(CSV + ".part", CSV)
    want = n * size * size
    if os.path.exists(META) and os.path.exists(U8) and os.path.getsize(U8) == want:
        print("cache already built (%s); delete %s to redo" % (U8, META))
        return
    fresh = not os.path.exists(U8) or os.path.getsize(U8) != want
    if fresh:
        open(PROGRESS, "w").close()
        with open(U8, "wb") as f:
            f.truncate(want)
    done = set()
    if os.path.exists(PROGRESS):
        done = {int(x) for x in open(PROGRESS).read().split()}
    todo = [t for t in tasks if t[2] not in done]
    print("decoding %d of %d row groups to %dx%d in %d processes (%d already done)"
          % (len(todo), len(tasks), size, size, jobs, len(tasks) - len(todo)), flush=True)
    bad, got, last = [], sum(t[3] for t in tasks if t[2] in done), 0
    start_n = got
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(n, size, U8)) as pool, open(PROGRESS, "a") as prog:
        for start, k, b in pool.map(_work, todo, chunksize=1):
            prog.write("%d\n" % start)
            prog.flush()
            bad += b
            got += k
            if got - last >= 5000 or got == n:
                last = got
                el = time.time() - t0
                print("  %d/%d  %.0f s  %.1f pictures/s  %d unreadable"
                      % (got, n, el, (got - start_n) / max(el, 1e-9), len(bad)), flush=True)
    meta = {"size": size, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": n,
            "unreadable": [[rows[r][0], e] for r, e in bad]}
    with open(META + ".part", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(META + ".part", META)
    for r, e in bad[:10]:
        print("  unreadable %s: %s" % (rows[r][0], e), flush=True)
    print("%s  N=%d  unreadable=%d  wall %.1f s" % (U8, n, len(bad), time.time() - t0), flush=True)


def load_table(settings=None):
    """(rows as a dict of arrays, FileRows (N, S, S)); S is the version's cache_size (384 when absent). Stops with the
    build command when prepare has not finished."""
    import pandas as pd
    size = cache_size_of(settings)
    u8, csv_path, meta, _ = cache_paths(size)
    if not (os.path.exists(meta) and os.path.exists(csv_path) and os.path.exists(u8)):
        raise SystemExit("no finished picture cache at %s; build it with: python modalities/chest_xray/eval/taix.py prepare%s"
                         % (u8, "" if size == SIZE else " --size %d" % size))
    df = pd.read_csv(csv_path)
    pix = FileRows(u8, (len(df), size, size))
    t = {"uid": df.UID.to_numpy(str), "patient": df.PatientID.to_numpy(str), "split": df.Split.to_numpy(str),
         "truth": present(df[COLUMNS].to_numpy()), "grades": df[COLUMNS].to_numpy().astype(np.int8)}
    return t, pix


# ---------------------------------------------------------------- training

def augment(x, g, cfg):
    """detector.augment without its horizontal flip (it would swap left and right findings): small rotate and shift,
    a brightness and contrast nudge; all on the card, all seeded."""
    import torch
    import torch.nn.functional as F
    n = x.shape[0]
    hw, wh = x.shape[-2] / x.shape[-1], x.shape[-1] / x.shape[-2]
    ang = (torch.rand(n, device=x.device, generator=g) * 2 - 1) * (cfg["rotate_deg"] * np.pi / 180.0)
    tx = (torch.rand(n, device=x.device, generator=g) * 2 - 1) * cfg["translate"] * 2
    ty = (torch.rand(n, device=x.device, generator=g) * 2 - 1) * cfg["translate"] * 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin * hw, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin * wh, cos, ty
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    j = cfg["jitter"]
    contrast = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device, generator=g) * 2 - 1) * j
    bright = (torch.rand(n, 1, 1, 1, device=x.device, generator=g) * 2 - 1) * j
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * contrast + m + bright).clamp(0.0, 1.0)


def predict(reader, net, pix, truth, rows, mean, sd, batch, px, n_out=len(NAMES)):
    """(len(rows), n_out) logits in `rows` order: 8, or N_ORDINAL for an ordinal net."""
    import torch
    net.eval()
    out = np.zeros((len(rows), n_out), np.float32)
    y = np.asarray(truth, np.float32)
    parts = [rows[a:a + batch] for a in range(0, len(rows), batch)]
    cache_row = np.arange(len(pix))
    at = 0
    with torch.no_grad():
        for sl, x, _y in reader.prefetch(pix, cache_row, y, parts):
            x = (reader.to_float(x, px) - mean) / sd
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(reader.DEV.type == "cuda")):
                z = net(x)
            out[at:at + len(sl)] = z.float().cpu().numpy()
            at += len(sl)
    return out


def n_outputs(settings):
    return N_ORDINAL if settings.get("ordinal") else len(NAMES)


def make_net(reader, cfg, settings):
    if settings.get("arch") == "vit-s16":
        import mae
        enc = mae.ENC
        return mae.ChestViT(mae.make_vit(int(settings["px"]), enc["dim"], enc["depth"], enc["heads"]), n_outputs(settings))
    return reader.ChestNet(reader.widths_of(settings, cfg), cfg["dropout"], cfg["stem_stride"], n_outputs(settings),
                           renderings=int(settings.get("renderings", 1)))


def load_init(net, name, say=print, path=None):
    """Copy the trunk of our own chest reader's checkpoint into `net`; the head stays as built."""
    import torch
    path = path or INITS[name]
    src = torch.load(path, map_location="cpu", weights_only=False)["state"]
    have = net.state_dict()
    new = {}
    for k, v in src.items():
        if k.startswith("trunk."):
            assert k in have and tuple(have[k].shape) == tuple(v.shape), "init tensor %s does not fit the net" % k
            new[k] = v
    n_trunk = sum(1 for k in have if k.startswith("trunk."))
    assert len(new) == n_trunk, "init copied %d trunk tensors, the net has %d" % (len(new), n_trunk)
    r = net.load_state_dict(new, strict=False)
    assert not r.unexpected_keys and all(k.startswith("head.") for k in r.missing_keys), r
    say("init %s: %s, %d trunk tensors copied, head fresh" % (name, path, len(new)))


def make_loss(settings, y, tr, dev=None):
    """BCE over the targets `y` (8 present columns, or the 31 ordinal ones); pos_weight per output when on."""
    import torch
    import torch.nn as nn
    pw = None
    if settings.get("pos_weight"):
        pos = y[tr].sum(axis=0)
        pw = torch.tensor(np.clip((len(tr) - pos) / np.maximum(pos, 1.0), 1.0, 20.0), dtype=torch.float32, device=dev)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def fingerprint_of(version, epochs, batch, settings, cfg, cache):
    return json.loads(json.dumps({"version": version, "epochs": epochs, "batch": batch, "settings": settings, "cfg": cfg,
                                  "seed": SEED, "cache": cache}, sort_keys=True))


def fit(reader, pix, truth, tr, es, settings, epochs, batch, cfg, partial=None, fingerprint=None, grades=None):
    """Train on rows `tr`, keep the epoch with the best mean AUROC on rows `es`; returns (net, info). With `partial`
    a checkpoint is saved after every epoch and mid-epoch when core.ckpt.Every is due, and a matching one is continued.
    `truth` is the (N, 8) present truth; with settings["ordinal"] the targets are ordinal_targets(`grades`) instead."""
    import torch
    import torch.nn as nn
    from core import ckpt, polite
    dev = reader.DEV
    px = int(settings.get("size", pix.shape[-1]))      # the opened cache's side: 384, or 512 with cache_size
    amp = bool(settings.get("amp", True)) and dev.type == "cuda"
    ordinal = bool(settings.get("ordinal"))
    if ordinal and grades is None:
        raise SystemExit("ordinal training needs the grades")
    y = np.asarray(ordinal_targets(grades) if ordinal else truth, np.float32)
    truth = np.asarray(truth)
    n_out = n_outputs(settings)
    cache_row = np.arange(len(pix))
    g_cpu = np.random.default_rng(SEED + 100)
    mean, sd = reader.mean_sd(pix, tr, px, np.random.default_rng(SEED))
    net = make_net(reader, cfg, settings).to(dev)
    if settings.get("init"):
        load_init(net, settings["init"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = make_loss(settings, y, tr, dev)
    g_dev = torch.Generator(device=dev).manual_seed(SEED)
    pace = polite.Pace()
    every = ckpt.Every()
    best, best_state, since, took = -1.0, None, 0, []
    start_ep, skip, order_saved, total0, ep_prior = 1, 0, None, 0.0, 0.0
    print("train %d, early-stop %d pictures, %d epochs at batch %d, %d px, amp %s"
          % (len(tr), len(es), epochs, batch, px, amp), flush=True)

    def save_part(epoch, done_b, stopped=False, order=None, total=0.0, ep_secs=0.0):
        ckpt.save_partial(partial, {
            "state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
            "batch": done_b, "stopped": stopped, "order": order, "total": total, "epoch_seconds_so_far": ep_secs,
            "best": best, "best_state": best_state, "since": since, "took": took, "g_cpu": g_cpu.bit_generator.state,
            "g_dev": g_dev.get_state(), "rng": ckpt.rng_state(), "fingerprint": fingerprint})
        every.mark()

    ck = ckpt.load_partial(partial, fingerprint) if partial else None
    if ck is not None and 0 < ck["epoch"] <= epochs:
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best, since, took = ck["best"], ck["since"], list(ck["took"])
        best_state = None if ck["best_state"] is None else {k: t.to(dev) for k, t in ck["best_state"].items()}
        g_cpu.bit_generator.state = ck["g_cpu"]
        g_dev.set_state(ck["g_dev"].cpu())
        ckpt.set_rng_state(ck["rng"])
        if ck["batch"]:
            start_ep, skip, order_saved = ck["epoch"], ck["batch"], ck["order"]
            total0, ep_prior = ck["total"], ck["epoch_seconds_so_far"]
        else:
            start_ep = epochs + 1 if ck["stopped"] else ck["epoch"] + 1
        print("resumed at epoch %d of %d%s (partial checkpoint)"
              % (min(start_ep, epochs), epochs, ", batch %d" % skip if skip else ""), flush=True)
    del ck
    for ep in range(start_ep, epochs + 1):
        t0 = time.time() - ep_prior
        net.train()
        order = order_saved if order_saved is not None else tr[g_cpu.permutation(len(tr))]
        all_parts = [order[a:a + batch] for a in range(0, len(order), batch)]
        parts = all_parts[skip:]
        total = torch.full((), total0, dtype=torch.float64, device=dev)
        done_b = skip
        skip, order_saved, total0, ep_prior = 0, None, 0.0, 0.0
        began = time.time()
        for rows, x, yb in reader.prefetch(pix, cache_row, y, parts):
            x = (augment(reader.to_float(x, px), g_dev, cfg) - mean) / sd
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = loss_fn(net(x).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach().double() * len(rows)
            pace.rest(time.time() - began)
            began = time.time()
            done_b += 1
            if partial and done_b < len(all_parts) and every.due():
                save_part(ep, done_b, order=order, total=float(total), ep_secs=time.time() - t0)
        sched.step()
        z_es = predict(reader, net, pix, truth, es, mean, sd, batch, px, n_out)
        v, _ = reader.mean_auroc(present_logits(z_es, ordinal), truth[es])
        took.append(time.time() - t0)
        print("  epoch %2d  loss %.4f  early-stop auroc %.4f  %.0f s" % (ep, float(total) / len(tr), v, took[-1]), flush=True)
        stop_now = False
        if v > best + 1e-4:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            since += 1
            if since >= cfg["patience"]:
                print("  stopping: no better early-stop AUROC for %d epochs" % cfg["patience"], flush=True)
                stop_now = True
        if partial:
            save_part(ep, 0, stopped=stop_now)
        if stop_now:
            break
    if best_state is None:
        raise SystemExit("no epoch had a defined early-stop AUROC")
    net.load_state_dict(best_state)
    info = {"epochs": len(took), "early_stop_auroc": best, "mean": float(mean), "sd": float(sd),
            "train_images": int(len(tr)), "early_stop_images": int(len(es)), "seconds": round(sum(took), 1)}
    return net, info


def setup_card(reader, name):
    import torch
    from core import card
    card.require_card(reader.DEV, torch)
    reader.xd.limit_card_memory(CARD_GB, name)
    if reader.DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True


def parse_settings(items):
    from core import sets
    s = sets.parse(items)
    bad = sorted(set(s) - set(SETTINGS))
    if bad:
        raise SystemExit("unknown setting(s) %s: train takes %s" % (", ".join(bad), ", ".join(sorted(SETTINGS))))
    if "init" in s and s["init"] not in INITS:
        raise SystemExit("unknown init %r: known are %s" % (s["init"], ", ".join(sorted(INITS))))
    if "cache_size" in s and (type(s["cache_size"]) is not int or s["cache_size"] not in CACHE_SIZES):
        raise SystemExit("cache_size must be 384 or 512, not %r" % (s["cache_size"],))
    if "train_share" in s and (isinstance(s["train_share"], bool) or not isinstance(s["train_share"], (int, float))
                               or not 0 < s["train_share"] <= 1):
        raise SystemExit("train_share must be a number above 0 and at most 1, not %r" % (s["train_share"],))
    # the defaults stay out of the settings, so v1's fingerprint and saved settings are what they were
    return {k: v for k, v in s.items() if not (k == "renderings" and v == 1) and not (k == "ordinal" and not v)}


def print_table(title, res):
    print(title)
    for name, (p, lo, hi) in zip(NAMES, res):
        print("  %-17s %.4f  (95%% interval %.4f to %.4f)" % (name, p, lo, hi), flush=True)
    m = float(np.mean([r[0] for r in res]))
    print("  %-17s %.4f" % ("mean of the eight", m), flush=True)
    return m


def train(version, epochs, batch, settings):
    import torch
    from core import ledger
    import reader
    out = out_dir(version)
    os.makedirs(out, exist_ok=True)
    setup_card(reader, "taix train v%d" % version)
    cfg = dict(reader.CFG)
    torch.manual_seed(SEED)
    t, pix = load_table(settings)
    split = t["split"]
    tr, es = early_stop_split(t["patient"], np.flatnonzero(split == "train"))
    val = np.flatnonzero(split == "val")
    if settings.get("train_share") is not None:
        tr = share_of_patients(t["patient"], tr, float(settings["train_share"]))
        print("train_share %s: %d training pictures of %d patients kept" % (settings["train_share"], len(tr), len(set(t["patient"][tr]))), flush=True)
    assert not (set(t["patient"][tr]) & set(t["patient"][es])), "a patient is on both sides of the early-stop split"
    print("train split %d pictures: %d to train (%d patients), %d to early-stop (%d patients); val %d pictures"
          % ((split == "train").sum(), len(tr), len(set(t["patient"][tr])), len(es), len(set(t["patient"][es])), len(val)), flush=True)
    st = os.stat(cache_paths(cache_size_of(settings))[0])      # the 384 file unless cache_size says 512
    fp = fingerprint_of(version, epochs, batch, settings, cfg, [st.st_size, st.st_mtime_ns])
    if settings.get("init"):
        si = os.stat(INITS[settings["init"]])
        fp["init_file"] = [si.st_size, si.st_mtime_ns]
    with ledger.run("experiment", "taix train v%d" % version) as led:
        led.settings_from({"epochs": epochs, "batch": batch, "cfg": cfg, "settings": settings})
        model = os.path.join(out, "model.pt")
        ck = torch.load(model, map_location="cpu", weights_only=False) if os.path.exists(model) and \
            os.environ.get("RAIDIOLOGY_NO_RESUME") != "1" else None
        if ck is not None and ck.get("fingerprint") == fp:
            print("training finished earlier (%s); scoring only" % model, flush=True)
            net = make_net(reader, cfg, settings).to(reader.DEV)
            net.load_state_dict(ck["state"])
            info = ck["info"]
        else:
            partial = os.path.join(out, "partial.pt")
            net, info = fit(reader, pix, t["truth"], tr, es, settings, epochs, batch, cfg, partial=partial, fingerprint=fp,
                            grades=t["grades"])
            torch.save({"state": net.state_dict(), "cfg": cfg, "info": info, "settings": settings, "fingerprint": fp}, model)
            from core import ckpt
            ckpt.clear_partial(partial)
        led.number("early_stop_auroc", info["early_stop_auroc"]).number("epochs", info["epochs"], "count")
        led.number("seconds", info["seconds"], "s")
        if reader.DEV.type == "cuda":
            peak = reader.torch.cuda.max_memory_allocated() / 2 ** 30
            held = reader.torch.cuda.max_memory_reserved() / 2 ** 30      # what the cap counts
            led.number("peak_memory_gb", peak).number("peak_reserved_memory_gb", held)
            print("peak card memory %.2f GB allocated, %.2f reserved (share %.1f)" % (peak, held, CARD_GB), flush=True)
        mean = torch.tensor(info["mean"], dtype=torch.float32, device=reader.DEV)
        sd = torch.tensor(info["sd"], dtype=torch.float32, device=reader.DEV)
        px = read_px(settings)
        ordinal = bool(settings.get("ordinal"))
        full = predict(reader, net, pix, t["truth"], val, mean, sd, batch, px, n_outputs(settings))
        logits = present_logits(full, ordinal)
        extra = {"cumulative": full} if ordinal else {}
        np.savez(os.path.join(out, "val_logits.npz"), UID=t["uid"][val], PatientID=t["patient"][val],
                 logits=logits, truth=t["truth"][val], **extra)
        res = bootstrap_auroc(logits, t["truth"][val], t["patient"][val])
        m = print_table("val AUROC (%d pictures, %d patients; patient bootstrap, %d resamples, seed %d)"
                        % (len(val), len(set(t["patient"][val])), BOOT, BOOT_SEED), res)
        for name, (p, lo, hi) in zip(NAMES, res):
            led.number("val.auroc.%s" % name, p).number("val.lo.%s" % name, lo).number("val.hi.%s" % name, hi)
        led.number("val.auroc.mean8", m)
        if ordinal:
            # quadratic weighted kappa of the most likely grade (a monotone read, see grades_from_logits) against the true one
            kap = [quadratic_kappa(t["grades"][val][:, f], grades_from_logits(full)[:, f], STEPS[f] + 1) for f in range(len(NAMES))]
            print("val quadratic weighted kappa of the most likely grade (steps with probability above 0.5)")
            for name, k in zip(NAMES, kap):
                print("  %-17s %.4f" % (name, k), flush=True)
                led.number("val.kappa.%s" % name, k)
            print("  %-17s %.4f" % ("mean of the eight", float(np.nanmean(kap))), flush=True)
        line = write_val_verdict(out, m, version)
        led.note_is(line)
        print(line, flush=True)


def write_val_verdict(out, m, version):
    """Writes out/val_verdict.json (the gate reads it), returns the verdict line."""
    line = val_verdict_line(m)
    with open(os.path.join(out, "val_verdict.json"), "w", encoding="utf-8") as f:
        json.dump({"verdict": line.rsplit(": ", 1)[1], "mean": m, "bar": OPEN_BAR, "version": version,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    return line


def read(versions):
    """The two-net (or n-net) read on val: the mean of the versions' saved present probabilities. No card, no training."""
    from core import ledger
    label = read_label(versions)
    out = read_dir(versions)
    os.makedirs(out, exist_ok=True)
    uid, pat, truth, logits = read_val(versions)
    with ledger.run("experiment", "taix read %s" % label) as led:
        led.settings_from({"versions": list(versions)})
        res = bootstrap_auroc(logits, truth, pat)
        m = print_table("val AUROC of the mean of versions %s's present probabilities (%d pictures, %d patients; "
                        "patient bootstrap, %d resamples, seed %d)" % (label, len(uid), len(set(pat)), BOOT, BOOT_SEED), res)
        for name, (p, lo, hi) in zip(NAMES, res):
            led.number("val.auroc.%s" % name, p).number("val.lo.%s" % name, lo).number("val.hi.%s" % name, hi)
        led.number("val.auroc.mean8", m)
        line = write_val_verdict(out, m, label)
        led.note_is(line)
        print(line, flush=True)


def test(versions):
    """Opens the test split once: one version, or the mean of several versions' present probabilities (a read)."""
    import torch
    from core import ledger
    import reader
    versions = [versions] if isinstance(versions, int) else list(versions)
    label = read_label(versions)
    out = out_dir(versions[0]) if len(versions) == 1 else read_dir(versions)
    test_gate(label, out, MARKER)
    cks = []
    for v in versions:
        model = os.path.join(out_dir(v), "model.pt")
        if not os.path.exists(model):
            raise SystemExit("no trained model at %s" % model)
        cks.append(model)
    setup_card(reader, "taix test v%s" % label)
    cks = [torch.load(m_, map_location="cpu", weights_only=False) for m_ in cks]
    t, pix = load_table(cks[0]["settings"])
    pixes = {cache_size_of(cks[0]["settings"]): pix}
    rows = np.flatnonzero(t["split"] == "test")
    write_marker(label if len(versions) > 1 else versions[0], MARKER)   # first: from here the test split counts as opened, whatever happens next
    print("test split opened (marker %s): %d pictures, %d patients" % (MARKER, len(rows), len(set(t["patient"][rows]))), flush=True)
    with ledger.run("experiment", "taix test v%s" % label) as led:
        led.settings_from({"version": label, "settings": [ck["settings"] for ck in cks]})
        per_net = []
        for ck in cks:
            settings, info = ck["settings"], ck["info"]
            net = make_net(reader, ck["cfg"], settings).to(reader.DEV)
            net.load_state_dict(ck["state"])
            mean = torch.tensor(info["mean"], dtype=torch.float32, device=reader.DEV)
            sd = torch.tensor(info["sd"], dtype=torch.float32, device=reader.DEV)
            cs = cache_size_of(settings)
            if cs not in pixes:
                pixes[cs] = load_table(settings)[1]
            z = predict(reader, net, pixes[cs], t["truth"], rows, mean, sd, ck["cfg"].get("batch", 64),
                        read_px(settings), n_outputs(settings))
            per_net.append(present_logits(z, bool(settings.get("ordinal"))))
            del net
        logits = per_net[0] if len(per_net) == 1 else mean_prob_logits(per_net)
        np.savez(os.path.join(out, "test_logits.npz"), UID=t["uid"][rows], PatientID=t["patient"][rows],
                 logits=logits, truth=t["truth"][rows])
        res = bootstrap_auroc(logits, t["truth"][rows], t["patient"][rows])
        tally = {"BEATEN": 0, "LEVEL": 0, "BEHIND": 0}
        print("test AUROC against the paper (Truhn et al., Sci Data 2026, Table 4; patient bootstrap, %d resamples, seed %d)"
              % (BOOT, BOOT_SEED))
        for name, (p, lo, hi), paper in zip(NAMES, res, PAPER):
            v = judge(lo, hi, paper)
            tally[v] += 1
            print("  %-17s ours %.4f (95%% interval %.4f to %.4f)  paper %.2f  %s" % (name, p, lo, hi, paper, v), flush=True)
            led.number("test.auroc.%s" % name, p).number("test.lo.%s" % name, lo).number("test.hi.%s" % name, hi)
        m = float(np.mean([r[0] for r in res]))
        led.number("test.auroc.mean8", m)
        for k, v in tally.items():
            led.number("test.%s" % k.lower(), v, "count")
        line = "of eight: beaten %d, level %d, behind %d; mean ours %.4f against the paper's %.4f" % (
            tally["BEATEN"], tally["LEVEL"], tally["BEHIND"], m, PAPER.mean())
        led.note_is(line)
        print(line, flush=True)


def compare_run(a, b, margin=0.01, name=""):
    """Prints side B against side A on val and the verdict; the mean of the eight goes in the ledger."""
    from core import ledger
    rows, verdict, n_pic, n_pat = compare_sides(a, b, margin)
    print("val, B (%s) against A (%s): %d pictures, %d patients; paired patient bootstrap, %d resamples, seed %d\n"
          % (b, a, n_pic, n_pat, BOOT, BOOT_SEED))
    print("| label | A | B | B - A (95% CI) |\n|---|---|---|---|")
    for lab, va, vb, d, lo, hi in rows:
        print("| %s | %.4f | %.4f | %+.4f (%+.4f to %+.4f) |" % (lab, va, vb, d, lo, hi))
    line = "Verdict (mean of the eight, margin %.3f): %s" % (margin, verdict)
    print("\n" + line, flush=True)
    with ledger.run("experiment", name or "taix compare %s against %s" % (b, a)) as led:
        led.settings_from({"a": a, "b": b, "margin": margin})
        _, va, vb, d, lo, hi = rows[-1]
        led.number("compare.mean8.a", va).number("compare.mean8.b", vb).number("compare.mean8.diff", d)
        led.number("compare.mean8.lo", lo).number("compare.mean8.hi", hi)
        led.note_is(line)


# ---------------------------------------------------------------- selftest

def selftest():
    import tempfile
    import torch
    from PIL import Image
    import reader
    # the present rule: any grade above none; heart's 0 is normal
    g = np.array([[0, 1, 2, 3, 4, 0, 1, 0], [3, 0, 0, 0, 0, 0, 0, 4]])
    assert present(g).tolist() == [[0, 1, 1, 1, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 1]], "the present rule"
    # the early-stop split is by patient, seeded, and about a tenth of the patients
    rng = np.random.default_rng(1)
    pats = np.array(["p%d" % i for i in rng.integers(0, 200, 1500)])
    rows = np.arange(1500)
    tr, es = early_stop_split(pats, rows)
    assert not (set(pats[tr]) & set(pats[es])), "a patient is on both sides"
    assert len(tr) + len(es) == len(rows) and len(set(pats[es])) == round(0.1 * len(set(pats))), len(set(pats[es]))
    tr2, es2 = early_stop_split(pats, rows)
    assert (es == es2).all() and (tr == tr2).all(), "the split is seeded"
    # train_share: a number in (0, 1], about that share of the patients, the same ones twice
    assert parse_settings(["train_share=0.1"]) == {"train_share": 0.1} and parse_settings(["train_share=1"]) == {"train_share": 1}
    for bad_share in ("0", "1.5", '"x"', "true"):
        try:
            parse_settings(["train_share=" + bad_share])
        except SystemExit:
            pass
        else:
            raise AssertionError("train_share=%s got through" % bad_share)
    s1, s2 = share_of_patients(pats, rows, 0.1), share_of_patients(pats, rows, 0.1)
    assert (s1 == s2).all() and len(set(pats[s1])) == round(0.1 * len(set(pats))), len(set(pats[s1]))
    assert {"chest-ssl-v1", "chest-ssl-v1-e15"} <= set(INITS)
    # init: absent unless given, known names only; the trunk is copied, the head stays fresh
    assert parse_settings([]) == {}
    assert parse_settings(['init="chest-v103"']) == {"init": "chest-v103"}
    try:
        parse_settings(['init="nope"'])
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown init was accepted")
    with tempfile.TemporaryDirectory() as d:
        src = reader.ChestNet([4, 4, 4, 4, 4], 0.3, 2, 15)
        src.train()
        src(torch.randn(2, 1, 64, 64))
        p = os.path.join(d, "fold0.pt")
        torch.save({"state": src.state_dict()}, p)
        net = reader.ChestNet([4, 4, 4, 4, 4], 0.3, 2, 31)
        fresh = {k: v.clone() for k, v in net.state_dict().items() if k.startswith("head.")}
        lines = []
        load_init(net, "chest-v103", say=lines.append, path=p)
        sd, ss = net.state_dict(), src.state_dict()
        n = 0
        for k, v in ss.items():
            if k.startswith("trunk."):
                assert torch.equal(sd[k], v), k
                n += 1
        assert n > 0
        for k, v in fresh.items():
            assert torch.equal(sd[k], v) and sd[k].shape != ss[k].shape, k
        assert len(lines) == 1 and "chest-v103" in lines[0] and "trunk tensors copied" in lines[0], lines
    # the verdict lines and the three-way judgement
    assert val_verdict_line(0.85) == "Verdict (idea 26 step 1, mean val AUROC at least 0.85): OPENS TEST"
    assert val_verdict_line(0.8499) == "Verdict (idea 26 step 1, mean val AUROC at least 0.85): STAYS ON VAL"
    assert (judge(0.88, 0.90, 0.87), judge(0.86, 0.88, 0.87), judge(0.80, 0.86, 0.87)) == ("BEATEN", "LEVEL", "BEHIND")
    assert judge(0.87, 0.9, 0.87) == "LEVEL" and judge(0.8, 0.87, 0.87) == "LEVEL"
    # the marker refusal: no verdict, a STAYS ON VAL verdict, an OPENS TEST verdict, then a marker blocks every version
    with tempfile.TemporaryDirectory() as d:
        marker = os.path.join(d, "taix_test_opened.json")
        for ver, body, why in ((1, None, "no val verdict"), (2, {"verdict": "STAYS ON VAL", "mean": 0.8}, "STAYS ON VAL")):
            o = os.path.join(d, "v%d" % ver)
            os.makedirs(o)
            if body:
                json.dump(body, open(os.path.join(o, "val_verdict.json"), "w"))
            try:
                test_gate(ver, o, marker)
            except SystemExit as e:
                assert why.split()[0] in str(e) or "verdict" in str(e), e
            else:
                raise AssertionError("test_gate let through: " + why)
        o = os.path.join(d, "v3")
        os.makedirs(o)
        json.dump({"verdict": "OPENS TEST", "mean": 0.86}, open(os.path.join(o, "val_verdict.json"), "w"))
        test_gate(3, o, marker)
        write_marker(3, marker)
        assert json.load(open(marker))["version"] == 3
        for ver in (3, 4):
            try:
                test_gate(ver, o, marker)
            except SystemExit as e:
                assert "already opened" in str(e), e
            else:
                raise AssertionError("a second opening got through")
        try:
            write_marker(3, marker)
        except FileExistsError:
            pass
        else:
            raise AssertionError("the marker was overwritten")
    # ordinal targets: heart 2 -> [1,1,0]; effusion 0 -> [0,0,0,0]; 4 -> [1,1,1,1]; 31 outputs in a fixed order
    assert N_ORDINAL == 31 and STEP0.tolist() == [0, 3, 7, 11, 15, 19, 23, 27], (N_ORDINAL, STEP0)
    og = np.array([[2, 0, 0, 4, 1, 0, 0, 0], [3, 4, 0, 0, 0, 0, 0, 0]])
    ot = ordinal_targets(og)
    assert ot.shape == (2, 31) and ot[0, 0:3].tolist() == [1, 1, 0] and ot[0, 7:11].tolist() == [0, 0, 0, 0]
    assert ot[0, 11:15].tolist() == [1, 1, 1, 1] and ot[0, 15:19].tolist() == [1, 0, 0, 0] and ot[1, 0:3].tolist() == [1, 1, 1]
    assert (ot[:, STEP0] == present(og)).all(), "step 0 is the present rule"
    # the present logit is step 0's; v1's eight logits pass through
    zz = np.arange(62, dtype=float).reshape(2, 31)
    assert (present_logits(zz, True) == zz[:, STEP0]).all() and (present_logits(zz[:, :8], False) == zz[:, :8]).all()
    # the monotone grade read: the count of steps above 0.5, even when the steps disagree
    pz = np.full((2, 31), 0.1)
    pz[0, 0:3] = [0.9, 0.6, 0.2]                    # heart: 2
    pz[0, 3:7] = [0.4, 0.9, 0.9, 0.9]               # congestion: 3 (a count, not a walk)
    pz[0, 7:11] = [0.9, 0.9, 0.9, 0.9]              # effusion left: 4
    pz[1, 0:3] = [0.5, 0.5, 0.5]                    # exactly 0.5 is not above
    gz = grades_from_logits(np.log(pz) - np.log1p(-pz))
    assert gz[0].tolist() == [2, 3, 4, 0, 0, 0, 0, 0] and gz[1].tolist() == [0] * 8, gz
    # kappa: perfect 1, an off-by-one pair below it, chance-level 0
    assert quadratic_kappa([0, 1, 2, 3], [0, 1, 2, 3], 4) == 1.0
    assert abs(quadratic_kappa([0, 1, 2, 3, 3], [0, 1, 2, 3, 2], 4) - (1 - 1 / 9 / (np.outer([1, 1, 1, 2], [1, 1, 2, 1]) / 5 * (np.arange(4)[:, None] - np.arange(4)[None, :]) ** 2 / 9).sum())) < 1e-12
    assert abs(quadratic_kappa([0, 0, 1, 1], [0, 1, 0, 1], 2)) < 1e-12
    # the two-net read is the mean of PROBABILITIES, not of logits
    a, b = np.array([[0.0, 4.0]]), np.array([[2.0, -4.0]])
    mp = mean_prob_logits([a, b])
    want = (sigmoid(a) + sigmoid(b)) / 2
    assert np.allclose(sigmoid(mp), want) and not np.allclose(mp, (a + b) / 2), mp
    # the read refuses val rows that differ; accepts and averages rows that match
    with tempfile.TemporaryDirectory() as d:
        uid, pt, tr8 = np.array(["a", "b", "c"]), np.array(["p", "p", "q"]), (rng.random((3, 8)) < 0.5).astype(np.int8)
        lg = [rng.normal(size=(3, 8)) for _ in range(3)]
        for v, ids, l in ((1, uid, lg[0]), (2, uid, lg[1]), (3, uid[::-1].copy(), lg[2])):
            os.makedirs(os.path.join(d, "taix-v%d" % v))
            np.savez(os.path.join(d, "taix-v%d" % v, "val_logits.npz"), UID=ids, PatientID=pt, logits=l, truth=tr8)
        u2, _p2, _t2, ml = read_val([1, 2], d)
        assert u2.tolist() == uid.tolist() and np.allclose(sigmoid(ml), (sigmoid(lg[0]) + sigmoid(lg[1])) / 2)
        try:
            read_val([1, 3], d)
        except AssertionError:
            pass
        else:
            raise AssertionError("the read took val rows in a different order")
        # the test gate takes a read's directory, and the marker still blocks a second opening of any kind
        marker = os.path.join(d, "taix_test_opened.json")
        ro = read_dir([1, 2])
        assert os.path.basename(ro) == "taix-read-1-2"
        rd = os.path.join(d, "taix-read-1-2")
        os.makedirs(rd)
        json.dump({"verdict": "OPENS TEST", "mean": 0.851}, open(os.path.join(rd, "val_verdict.json"), "w"))
        test_gate("1-2", rd, marker)
        write_marker("1-2", marker)
        for lab in ("1-2", "7"):
            try:
                test_gate(lab, rd, marker)
            except SystemExit as e:
                assert "already opened" in str(e), e
            else:
                raise AssertionError("a second opening got through")
        assert write_val_verdict(rd, 0.8499, "1-2").endswith("STAYS ON VAL")
    # ordinal=false changes nothing: the setting leaves the settings, so the fingerprint is the one computed without it
    assert parse_settings(["ordinal=false"]) == {} and parse_settings(["ordinal=true"]) == {"ordinal": True}
    cfg0 = {"lr": 0.1}
    assert fingerprint_of(1, 12, 64, parse_settings(["ordinal=false"]), cfg0, [1, 2]) == \
        fingerprint_of(1, 12, 64, {}, cfg0, [1, 2]) != fingerprint_of(1, 12, 64, {"ordinal": True}, cfg0, [1, 2])
    assert n_outputs({}) == 8 and n_outputs({"ordinal": True}) == 31
    # the cache by size: 384 is the constants, 512 its own files; cache_size takes 384 or 512 only
    assert cache_paths(384) == (U8, CSV, META, PROGRESS), "cache_paths(384) is the existing constants"
    assert [os.path.basename(p) for p in cache_paths(512)] == ["taix_512.u8", "taix_512.csv", "taix_512.json", "taix_512.progress"]
    assert parse_settings(["cache_size=512"]) == {"cache_size": 512}
    for badsize in ("256", "true", "\"512\""):
        try:
            parse_settings(["cache_size=" + badsize])
        except SystemExit:
            pass
        else:
            raise AssertionError("cache_size=%s got through" % badsize)
    old_way = json.loads(json.dumps({"version": 1, "epochs": 12, "batch": 64, "settings": {"amp": False}, "cfg": cfg0,
                                     "seed": SEED, "cache": [1, 2]}, sort_keys=True))
    assert fingerprint_of(1, 12, 64, {"amp": False}, cfg0, [1, 2]) == old_way, "no cache_size: the fingerprint is the old one"
    assert fingerprint_of(1, 12, 64, {"amp": False, "cache_size": 512}, cfg0, [1, 2]) != old_way, "cache_size changes it"
    yl = (rng.random((20, 8)) < 0.3).astype(np.float32)
    zl = torch.tensor(rng.normal(size=(20, 8)), dtype=torch.float32)
    assert torch.equal(make_loss({}, yl, np.arange(20))(zl, torch.tensor(yl)),
                       torch.nn.BCEWithLogitsLoss()(zl, torch.tensor(yl))), "the v1 loss is plain BCE"
    # the bootstrap's point estimate is the plain AUROC; intervals bracket it
    n = 400
    pid = np.array(["q%d" % (i // 4) for i in range(n)])
    truth = (rng.random((n, 8)) < 0.3).astype(np.int8)
    logit = rng.normal(size=(n, 8)) + 1.2 * truth
    logit[:50, 0] = np.round(logit[:50, 0])          # ties
    res = bootstrap_auroc(logit, truth, pid, n=60)
    for j, (p, lo, hi) in enumerate(res):
        assert abs(p - reader.xd.auroc(logit[:, j], truth[:, j])) < 1e-9, (j, p)
        assert lo <= p <= hi or abs(p - lo) < 0.05, (j, lo, p, hi)
    # a picture through the real door: a 16-bit PNG of 200 x 100 lands on a 32 x 32 canvas, aspect kept, black padding
    pic = (np.linspace(0, 1, 200 * 100).reshape(100, 200) * 60000).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(pic).save(buf, format="PNG")
    px = decode_bytes(buf.getvalue(), 32)
    assert px.shape == (32, 32) and px.dtype == np.uint8 and px.max() > 100, (px.shape, px.max())
    assert (px[:8] == 0).all() and (px[-8:] == 0).all() and px[12:20].max() > 0, "aspect kept, padded black"
    # a tiny end-to-end fit on the processor: resumable, finite, and the net has eight logits
    keep_dev = reader.DEV
    reader.DEV = torch.device("cpu")
    try:
        m = 48
        pix = rng.integers(0, 255, (m, 64, 64), dtype=np.uint8)
        y = (rng.random((m, 8)) < 0.5).astype(np.int8)
        cfg = dict(reader.CFG, widths=[4, 4, 4, 4, 4], batch=16, patience=2)
        with tempfile.TemporaryDirectory() as d:
            fp = {"selftest": 1}
            partial = os.path.join(d, "partial.pt")
            net, info = fit(reader, pix, y, np.arange(32), np.arange(32, 48), {"amp": False}, 2, 16, cfg, partial, fp)
            assert np.isfinite(info["early_stop_auroc"]) and info["epochs"] >= 1
            assert os.path.exists(partial), "a partial checkpoint was written"
            net2, info2 = fit(reader, pix, y, np.arange(32), np.arange(32, 48), {"amp": False}, 2, 16, cfg, partial, fp)
            assert info2["early_stop_auroc"] == info["early_stop_auroc"], "a finished partial resumes to the same answer"
            lg = predict(reader, net, pix, y, np.arange(5), torch.tensor(info["mean"]), torch.tensor(info["sd"]), 16, 64)
            assert lg.shape == (5, 8) and np.isfinite(lg).all()
    finally:
        reader.DEV = keep_dev
    # compare: '|' makes groups; B equal to A is a zero difference and NOT WORSE; a perfect B is BETTER; other rows stop it
    assert parse_side("1,2,3|14") == [[1, 2, 3], [14]] and parse_side("7") == [[7]]
    with tempfile.TemporaryDirectory() as d:
        uid, pat = np.array(["u%d" % i for i in range(400)]), np.array(["q%d" % (i // 4) for i in range(400)])
        tr8 = (rng.random((400, 8)) < 0.3).astype(np.int8)
        z = rng.normal(size=(400, 8))
        fakes = {1: (uid, z), 2: (uid, z), 3: (uid, tr8 * 6.0 - 3.0), 4: (uid[::-1].copy(), z)}
        for v, (ids, zz) in fakes.items():
            os.makedirs(os.path.join(d, "taix-v%d" % v))
            np.savez(os.path.join(d, "taix-v%d" % v, "val_logits.npz"), UID=ids, PatientID=pat, logits=zz, truth=tr8)
        rows, verdict, _, _ = compare_sides("1", "2", cache=d, n=60)
        assert all(r[3] == 0 for r in rows) and verdict == "NOT WORSE", (rows, verdict)
        rows, verdict, _, _ = compare_sides("1", "3", cache=d, n=60)
        assert verdict == "BETTER" and rows[-1][4] > 0, (rows[-1], verdict)
        try:
            compare_sides("1", "4", cache=d, n=60)
        except SystemExit as e:
            assert "not the same pictures" in str(e), e
        else:
            raise AssertionError("compare took versions whose rows differ")
    print("taix selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--jobs", type=int, default=MAX_JOBS, help="decoding processes, at most %d" % MAX_JOBS)
    p.add_argument("--size", type=int, default=SIZE, choices=CACHE_SIZES, help="cache side in pixels (512: the pictures' own size)")
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--set", action="append", default=[], help="key=<json>: %s" % ", ".join(sorted(SETTINGS)))
    s = sub.add_parser("test")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--version", type=int)
    g.add_argument("--versions", help="e.g. 1,2: the read of those versions (the mean of their present probabilities)")
    r = sub.add_parser("read")
    r.add_argument("--versions", required=True, help="e.g. 1,2: val verdict of the mean of the saved present probabilities")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="side A: e.g. 1,2,3,4,6,7 (a '|' starts a group, each group weighs the same)")
    c.add_argument("--b", required=True, help='side B: e.g. "1,2,3,4,6,7|14"')
    c.add_argument("--margin", type=float, default=0.01)
    c.add_argument("--name", default="", help="the ledger row's name")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "prepare":
        prepare(a.jobs, a.size)
    elif a.cmd == "train":
        train(a.version, a.epochs, a.batch, parse_settings(a.set))
    elif a.cmd == "read":
        read(parse_versions(a.versions))
    elif a.cmd == "compare":
        compare_run(a.a, a.b, a.margin, a.name)
    else:
        test(a.version if a.version is not None else parse_versions(a.versions))


if __name__ == "__main__":
    main()
