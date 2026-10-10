"""How likely is a lung nodule to be cancer: a small 3D CNN trained on LIDC-IDRI's radiologist ratings.

What. Stage 1 and stage 2 (nodule.py, nodule_fpr.py) find nodules. This module answers the next question,
"is this one malignant?", from the same 48 mm crop around the nodule that stage 2 reads. One network, one
logit; its sigmoid is the chance of cancer.

Why. The project's goal is to beat the radiologists. The LUNGx challenge (Armato et al. 2016, J Med Imaging
3(4):044506) is the public yardstick: 73 test nodules (37 benign, 36 malignant, by pathology or follow-up),
scored by area under the ROC curve (AUROC: the chance that a random cancer gets a higher score than a random
benign nodule; 0.5 is a coin, 1.0 is perfect). Ten computer methods scored 0.50 to 0.68; six radiologists
scored 0.70 to 0.85.

Labels. LIDC-IDRI has no biopsy: each of up to four radiologists rated each nodule's malignancy from 1 (highly
unlikely) to 5 (highly suspicious). So the label is a radiologists' opinion, not a pathology result, and the
model is a student of that opinion. The rule (every number is a --set setting):
  - a lesion from lidc.nodules() (marks of different radiologists matched into one lesion) counts when at
    least `min_readers` = 3 radiologists marked it (a median of one or two ratings is too noisy);
  - label 1 (malignant) when the median malignancy is above `split` = 3, label 0 (benign) when it is below 3,
    and a median of exactly 3 (undecided) is left out of training and of the score;
  - the lesion must lie in a scan that has a prepared 1 mm box (lung_prepare.py; all 1,018 LIDC scans do).
The counts are printed on every run.

Folds. LUNA16's scans keep the fold nodule.scan_folds(False) gives them (five folds, subsets paired). The LIDC
scans LUNA16 left out (lidc_extra.py) are always training and never held out, so they get fold -1 and have no
out-of-fold score. The same patient can have two scans: the run lists every patient whose scans fall in two
different folds, or whose extra scan trains a model that then scores the patient's held-out scan, because such
a patient is a leak between train and test. The bootstrap resamples patients, not nodules, so several nodules
of one patient do not count as independent.

The model. nodule_fpr.crop_of cuts the 48 mm cube (48^3 uint8 at 1 mm) through a small holder object (meta,
path, shape), so the same code reads LIDC boxes from <WORK>/eval/lung/vol and LUNGx boxes from
<WORK>/eval/lung/vol_lungx/vol. The net is nodule_fpr.build(cfg) (four stages of two 3D convolutions, pooled
mean and max, one linear logit). Training is nodule_fpr.train_net: balanced batches (half malignant, half
benign), random flips on all axes, in-plane 90-degree turns, a shift of up to `shift_mm` = 2 mm, a brightness
nudge, binary cross-entropy, AdamW with warmup then a cosine decay. Scores are kept as float32 logits (and
sigmoid p) so a ranking metric has no ties.

The card. The graphics card is shared through the project's card lease, exactly as nodule_fpr does it, with
its own environment variable RAIDIOLOGY_MALIGNANCY_CARD_GB (default 2.5). Scoring LUNGx leases only SCORING_GB
(1.0) of it, reading 16 crops at a time, so it fits beside a training job.

Resuming. Each fold's checkpoint (fold<k>.pt) holds the net and its scored rows, with a fingerprint of the
version, settings, fold and the lesion table; a rerun with the same fingerprint trains and reads nothing.
RAIDIOLOGY_NO_RESUME=1 retrains. Inside a fold, fold<k>.partial.pt (net, optimizer, scheduler, step, seconds,
peak memory; core.ckpt) is saved every RAIDIOLOGY_SAVE_EVERY_S seconds (default 300) and a rerun continues from
it; it is not part of the fingerprint and is removed once the fold's own checkpoint is saved.

    python modalities/ct/eval/nodule_malignancy.py train --version N [--folds 0 1 ...] [--set key=json ...] [--extra-nlst] [--extra-multiomics]
    python modalities/ct/eval/nodule_malignancy.py lungx --version N --set calibration|test [--open-test]

`--extra-nlst` adds nlst501.labels() (LIDC-annot-NLST501: NLST scans whose labels come from the patient's lung
cancer follow-up, not from reader opinion) to every fold's training as fold -1, never held out and never scored,
so the out-of-fold AUROC is still LIDC's alone and a v1 and a v2 with the flag compare directly. The fingerprint
and meta gain "extra_nlst": True only when it is set.

`--extra-multiomics` adds multiomics_table(): the pathology-proven nodules modalities/ct/eval/multiomics.py wrote
under <WORK>/eval/lung/vol_multiomics/ (boxes in vol/, labels in labels.csv; the rows with used true), as fold -1
(training only, never held out or scored, so the out-of-fold AUROC is still LIDC's alone). Their share of cancers
(81%) is far above LIDC's and NLST501's (about a third), so the source must say nothing about the label: each draw
first picks the multiomics source with probability q (that fold's multiomics rows over all its training rows), then
draws half cancer, half benign from that source's own lists (pick); the other draws are exactly the plain ones.
Needs RAM crops and refuses (exit 2, nothing written) without labels.csv, with size_balance, with agreed_first above
0, or with RAM crops off. The fingerprint and meta gain "extra_multiomics": True only when it is set.

`--set size_balance=true` trains on both labels with the same size mix. On LIDC's labels the nodule diameter
alone scores AUROC 0.939, so a net can do well by learning size. Each fold keeps only the bands 6-8, 8-10, 10-12,
12-14, 14-16 and 16-20 mm that hold at least 5 lesions of each label, and draws each band with the share of
training lesions it holds, half cancer and half benign. The fingerprint gains "size_balance" only when true.

`--set agreed_first=0.5` (a fraction of the steps, 0 = off) trains the first half of the steps on the agreed
lesions only: those whose readers all lie on the same side of `split` (every rating under it, or every rating
over it; with split 3, all 1-2 or all 4-5). Later steps draw from every training lesion as usual. NLST501's
labelled nodules (--extra-nlst) are drawn throughout. With size_balance on, the agreed phase uses the same bands
where the fold's agreed lesions hold at least 5 of each label. Each fold prints what the agreed phase used. A
run resumed past the switch draws from the full lists. The fingerprint gains "agreed_first" only when above 0.
`python modalities/ct/eval/nodule_malignancy.py selftest` checks this on a synthetic table (no data, no card).

`--set nlst_fix=true` (needs --extra-nlst, else exit 2) makes --extra-nlst use nlst501.labels(fixed=True): 38 NLST501
labels corrected against the cancers' own NLST-Sybil boxes (12 positives to unknown, 26 confirmed cancers to 1;
modalities/ct/eval/nlst501_fix.csv; docs/efficiency/README.md, idea 23; HSwarm job 20261005-044211-4fe2). The
fingerprint gains "nlst_fix" only when true, and meta gains "nlst_fix": True.

`train` writes <WORK>/eval/lung/malignancy-v<N>/: fold<k>.pt, oof.csv (seriesuid, coordX, coordY, coordZ,
diameter_mm, label, p, fold, then logit), meta.json; it prints the out-of-fold AUROC with a 95% interval
(1000 bootstrap resamples of patients) and the AUROC by diameter (under 10 mm, 10 to 20, over 20).

`lungx` scores <WORK>/eval/lung/lungx/nodules.csv (written by lungx.py; columns set, scan, nodule, patient,
seriesuid, coordX, coordY, coordZ, label, diagnosis, centre_hu; label 1 = malignant) with the mean of the fold
models' logits and prints the AUROC with a bootstrap interval beside the paper's bar. The test set is opened
ONCE: `--set test` refuses without --open-test and refuses when <out>/lungx_test_opened.json exists; when it
runs it writes that file (time, version) first, adds the AUROC to it as soon as it is known, and only then
writes the scores. The calibration set can be scored any time.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import lidc  # noqa: E402
import lung_prepare as lp  # noqa: E402
import nodule_fpr as nf  # noqa: E402
from core import card, ckpt, ledger, paths  # noqa: E402
from nodule import scan_folds, sha, load_resumable, with_settings  # noqa: E402

CFG = {
    "widths": [24, 48, 96, 192], "groups": 8,
    "steps": 4000, "batch": 32, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 300,
    "shift_mm": 2.0, "brightness": 0.05,
    "min_readers": 3, "split": 3.0,
    "seed": 0, "log_every": 200, "queue": 3, "threads": 6,
    "ram_crops": True,      # cut the jittered cubes out of one memory copy per lesion (same voxels; False reads the disk)
    "size_balance": False,  # draw both labels with the same size mix (idea 14); left out of the fingerprint when False
    "agreed_first": 0.0,    # fraction of the steps that draw LIDC lesions only from the agreed ones (idea 16); left out when 0
    "nlst_fix": False,      # --extra-nlst uses nlst501.labels(fixed=True) (idea 23); left out of the fingerprint when False
}
SIZE_BANDS = [(6.0, 8.0), (8.0, 10.0), (10.0, 12.0), (12.0, 14.0), (14.0, 16.0), (16.0, 20.0)]   # mm of diameter_mm
SIZE_MIN_PER_LABEL = 5      # a band is used when training holds at least this many of each label in it
SIZE_LIST_LEN = 20000       # entries per balanced list (the draw shares are matched to ~1/reps of a band)
N_FOLDS = 5
BANDS = [("under 10 mm", 0.0, 10.0), ("10 to 20 mm", 10.0, 20.0), ("over 20 mm", 20.0, float("inf"))]
BOOT = 1000
BAR_METHODS, BAR_RADIOLOGISTS = (0.50, 0.68), (0.70, 0.85)
LUNGX_DIR = os.path.join(paths.WORK, "eval", "lung", "lungx")
LUNGX_VOL = os.path.join(paths.WORK, "eval", "lung", "vol_lungx")
MO_VOL = os.path.join(paths.WORK, "eval", "lung", "vol_multiomics")     # multiomics.py: vol/mo-<SAMPLE>.npy/.json + labels.csv
OOF_COLS = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "label", "p", "fold"]


def out_root(version):
    return os.path.join(paths.WORK, "eval", "lung", "malignancy-v%d" % version)


def card_gb():
    return float(os.environ.get("RAIDIOLOGY_MALIGNANCY_CARD_GB", "2.5"))


# Scoring LUNGx (10 to 73 nodules) needs a fraction of training's 2.14 GB: it leases this much and reads 16 crops
# at a time, so it fits beside a training job instead of waiting an hour for one to finish (2026-10-03, 15:11).
SCORING_GB, SCORING_BATCH = 1.0, 16


_LIMITED = []


def limit_malignancy_card(gb=None):
    """Lease this process's share of the card, then a hard ceiling at it (nodule_fpr's way, own variable);
    `gb` asks for less than the variable's share (scoring), never more."""
    if nf.DEV.type != "cuda" or _LIMITED:
        return
    gb = card_gb() if gb is None else min(gb, card_gb())
    card.lease(gb, "nodule malignancy")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())
    _LIMITED.append(gb)


# ---- the data ------------------------------------------------------------------------------------

class Box:
    """What nodule_fpr.crop_of needs of a scan: meta (origin_mm, spacing_mm), path of the .npy, shape."""

    def __init__(self, npy, js):
        with open(js, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.path = npy
        self.shape = tuple(self.meta["shape"])
        assert list(self.meta["spacing_mm"]) == [1.0, 1.0, 1.0], "the box is not at 1 mm"


def lidc_box(uid):
    return Box(*lp.paths_for(lp.out_dir(), uid))


def lungx_box(uid):
    return Box(*lp.paths_for(LUNGX_VOL, uid))     # lungx.py prepare builds them under <LUNGX_VOL>/vol/


def lesion_table(cfg):
    """(frame, counts): one row per labelled LIDC lesion in a scan with a prepared box, with seriesuid,
    patient, coordX/Y/Z, diameter_mm, n_readers, malignancy_median, label (0/1), agreed (every reader on the
    same side of the split) and fold (-1 = never held out)."""
    les = lidc.nodules()
    counts = {"lesions": len(les)}
    les = les[les.n_readers >= cfg["min_readers"]]
    counts["with_enough_readers"] = len(les)
    les = les[les.malignancy_median.notna()]
    counts["rated"] = len(les)
    split = float(cfg["split"])
    undecided = les.malignancy_median == split
    counts["undecided_left_out"] = int(undecided.sum())
    les = les[~undecided]
    boxed = les.series_uid.map(lambda u: all(os.path.exists(p) for p in lp.paths_for(lp.out_dir(), u)))
    counts["no_prepared_box"] = int((~boxed).sum())
    les = les[boxed]
    df = pd.DataFrame({
        "seriesuid": les.series_uid.to_numpy(), "patient": les.patient.to_numpy(),
        "coordX": les.x_mm.to_numpy(), "coordY": les.y_mm.to_numpy(), "coordZ": les.z_mm.to_numpy(),
        "diameter_mm": les.diameter_mm.to_numpy(), "n_readers": les.n_readers.to_numpy(),
        "malignancy_median": les.malignancy_median.to_numpy(),
        "label": (les.malignancy_median > split).astype(int).to_numpy(),
        "agreed": ((les.mal_max < split) | (les.mal_min > split)).to_numpy()})
    folds_of, missing = scan_folds(False)
    counts["luna16_scans_without_box"] = len(missing)
    df["fold"] = df.seriesuid.map(folds_of).fillna(-1).astype(int)
    counts["labelled"] = len(df)
    counts["malignant"] = int(df.label.sum())
    counts["benign"] = int((df.label == 0).sum())
    counts["agreed"] = int(df.agreed.sum())
    counts["in_luna16_folds"] = int((df.fold >= 0).sum())
    counts["extra_training_only"] = int((df.fold < 0).sum())
    counts["scans"] = int(df.seriesuid.nunique())
    counts["patients"] = int(df.patient.nunique())
    return df.reset_index(drop=True), counts


def nlst_table(fixed=False):
    """nlst501.labels() rows that have a label and a prepared box, in lesion_table's columns, fold -1.
    fixed: nlst501.labels(fixed=True), the 38 labels corrected against the Sybil boxes (idea 23)."""
    import nlst501
    n = nlst501.labels(fixed=fixed)
    n = n[n.label.notna()]
    n = n[n.seriesuid.map(lambda u: all(os.path.exists(p) for p in lp.paths_for(lp.out_dir(), u)))]
    return pd.DataFrame({
        "seriesuid": n.seriesuid.to_numpy(), "patient": ("nlst-" + n.patient.astype(str)).to_numpy(),
        "coordX": n.coordX.to_numpy(), "coordY": n.coordY.to_numpy(), "coordZ": n.coordZ.to_numpy(),
        "diameter_mm": n.diameter_mm.to_numpy(), "n_readers": n.n_readers.to_numpy(),
        "malignancy_median": np.nan, "label": n.label.astype(int).to_numpy(), "agreed": False, "fold": -1})


def multiomics_table(vol=MO_VOL):
    """<vol>/labels.csv rows that are used and have both box files, in lesion_table's columns, fold -1. The
    pathology-proven nodules of multiomics.py: seriesuid and patient start with 'mo-'."""
    m = pd.read_csv(os.path.join(vol, "labels.csv"))
    m = m[m.used.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))]
    m = m[m.seriesuid.map(lambda u: all(os.path.exists(p) for p in lp.paths_for(vol, u)))]
    return pd.DataFrame({
        "seriesuid": m.seriesuid.astype(str).to_numpy(), "patient": m.patient.astype(str).to_numpy(),
        "coordX": m.coordX.to_numpy(), "coordY": m.coordY.to_numpy(), "coordZ": m.coordZ.to_numpy(),
        "diameter_mm": m.diameter_mm.to_numpy(), "n_readers": np.nan, "malignancy_median": np.nan,
        "label": m.label.astype(int).to_numpy(), "agreed": False, "fold": -1})


def patient_leaks(df):
    """Patients with scans in two different folds, and held-out patients whose other scan is an extra
    (so a model trains on one scan of a patient and scores another)."""
    scans = df.drop_duplicates("seriesuid")
    two_folds, extra_leak = [], []
    for patient, g in scans.groupby("patient"):
        held = set(int(f) for f in g.fold if f >= 0)
        if len(held) > 1:
            two_folds.append(patient)
        if held and (g.fold < 0).any():
            extra_leak.append(patient)
    return two_folds, extra_leak


# ---- scoring -------------------------------------------------------------------------------------

@torch.inference_mode()
def logits_of(net, boxes, frame, batch=64):
    """float32 logit of each row of `frame` (seriesuid, coordX/Y/Z), same order."""
    net.eval()
    xyz = frame[["coordX", "coordY", "coordZ"]].to_numpy(float)
    uids = frame.seriesuid.tolist()
    out = np.zeros(len(frame), np.float32)

    def read(i):
        return np.stack([nf.crop_of(boxes[u], p) for u, p in zip(uids[i:i + batch], xyz[i:i + batch])])

    reader = ThreadPoolExecutor(1)
    ahead = reader.submit(read, 0) if len(frame) else None
    for i in range(0, len(frame), batch):
        crops = ahead.result()
        if i + batch < len(frame):
            ahead = reader.submit(read, i + batch)
        x = nf.net_input(torch.from_numpy(crops[:, None]).to(nf.DEV), net.c_in)
        with torch.autocast(nf.DEV.type, dtype=torch.bfloat16, enabled=nf.DEV.type == "cuda"):
            out[i:i + batch] = net(x).float().cpu().numpy()
    reader.shutdown()
    return out


def auroc(y, s):
    """Area under the ROC curve by ranks (ties count half); nan when a class is missing."""
    y, s = np.asarray(y), np.asarray(s, float)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auroc(y, s, groups, n=BOOT, seed=0):
    """(AUROC, low, high): the 95% interval from `n` resamples of whole groups (patients)."""
    y, s = np.asarray(y), np.asarray(s, float)
    point = auroc(y, s)
    idx = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(lambda x: x.to_numpy()).tolist()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[i] for i in rng.integers(len(idx), size=len(idx))])
        v = auroc(y[pick], s[pick])
        if v == v:
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sigmoid64(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


# ---- crops from memory ----------------------------------------------------------------------------
# nodule_fpr.crop_of opens the scan's .npy and reads the 48 mm cube off the disk for every crop of every
# batch: 8.5 ms each, 0.275 s a batch of 32 in one thread, 128,000 reads a fold (2026-10-03). A lesion is
# drawn about 80 times a fold, and its jitter (shift_mm, 1 mm voxels) moves the cube a few voxels only, so
# its (48 + 2 PAD) voxel window is read once and every jittered cube is cut out of it: the same voxels (zeros
# outside the box included) as crop_of for the same draw, and the same random draws in the same order.

PAD = 3                       # voxels the jitter can move the cube by (ceil(shift_mm of 2) + 1)
_RAM = {}                     # (seriesuid, x, y, z) -> the lesion's (48 + 2 PAD)^3 uint8 window, kept all run


def _window_of(scan, key):
    return nf.crop_of(scan, np.array(key[1:], float), nf.CUBE + 2 * PAD)


def fill_ram(boxes, keys, workers=4):
    """Read the window of every lesion in `keys` that is not in memory yet; returns how many were read."""
    todo = [k for k in dict.fromkeys(keys) if k not in _RAM]
    with ThreadPoolExecutor(workers) as pool:
        for k, w in zip(todo, pool.map(lambda k: _window_of(boxes[k[0]], k), todo)):
            _RAM[k] = w
    return len(todo)


def ram_cube(scan, key, jitter):
    """(1, 48, 48, 48) uint8, equal to nf.cube_of(scan, key[1:] + jitter): cut from the lesion's window."""
    centre = np.array(key[1:], float)
    base = np.rint(lp.world_to_voxel(scan.meta, centre.reshape(1, 3))[0]).astype(int)
    moved = np.rint(lp.world_to_voxel(scan.meta, (centre + jitter).reshape(1, 3))[0]).astype(int)
    d = moved - base + PAD
    if key not in _RAM or (d < 0).any() or (d > 2 * PAD).any():      # a shift beyond the window: read it as before
        return nf.cube_of(scan, centre + jitter, 1, None)
    z, y, x = d
    return _RAM[key][z:z + nf.CUBE, y:y + nf.CUBE, x:x + nf.CUBE][None]


def pick(rng, pos, neg, mo=None):
    """(key, label) for one draw: half cancer, half benign from pos and neg. With mo = (q, mo_pos, mo_neg) the draw
    first goes to the multiomics lists with probability q, then makes the same half-and-half choice there. mo None
    makes exactly the draw (the same rng calls in the same order) the sampler always made."""
    if mo is not None and rng.random() < mo[0]:
        pos, neg = mo[1], mo[2]
    if rng.random() < 0.5:
        return pos[int(rng.integers(len(pos)))], 1.0
    return neg[int(rng.integers(len(neg)))], 0.0


class RamSampler(nf.Sampler):
    """nodule_fpr.Sampler with ram_cube for the read; the loop and its random draws are Sampler._run's, except that
    with `mo` set (--extra-multiomics: (q, mo_pos, mo_neg), by train_fold_net) the draw goes through pick."""
    mo = None

    def _run(self, rng):
        try:
            while not self.stop.is_set():
                imgs = np.empty((self.batch, 1, nf.CUBE, nf.CUBE, nf.CUBE), np.uint8)
                labels = np.empty(self.batch, np.float32)
                for b in range(self.batch):
                    key, label = pick(rng, self.pos_list, self.neg_list, self.mo)
                    jitter = rng.uniform(-self.cfg["shift_mm"], self.cfg["shift_mm"], 3)
                    imgs[b] = self.augment(ram_cube(self.scans[key[0]], key, jitter), rng)
                    labels[b] = label
                out, labels = torch.from_numpy(imgs), torch.from_numpy(labels)
                if nf.DEV.type == "cuda":
                    out, labels = out.pin_memory(), labels.pin_memory()
                while not self.stop.is_set():
                    try:
                        self.q.put((out, labels), timeout=0.5)
                        break
                    except queue.Full:
                        pass    # loops to re-check the stop flag
        except Exception as e:
            self.error = e
            self.q.put(None)

    def get(self):
        got = self.q.get()
        if got is None:
            raise RuntimeError("crop loader failed") from self.error
        return got


def ram_ok(cfg):
    """RAM crops cover the plain one-channel, unzoomed read and no hard negatives, which is all this trainer asks."""
    return bool(cfg.get("ram_crops", True)) and int(cfg.get("context", 1)) == 1 and not cfg.get("zoom", 0.0)


# ---- train ---------------------------------------------------------------------------------------

def fingerprint_cfg(cfg):
    """The settings that decide what a fold's net holds: ram_crops only says how crops are read, and size_balance,
    agreed_first and nlst_fix join only when true and above 0, so versions trained before them keep resuming."""
    skip = {"ram_crops"} | ({"size_balance"} if not cfg.get("size_balance") else set()) \
        | ({"agreed_first"} if not cfg.get("agreed_first") else set()) \
        | ({"nlst_fix"} if not cfg.get("nlst_fix") else set())
    return {k: v for k, v in cfg.items() if k not in skip}


def agreed_phase(trn, cfg):
    """(pos, neg, info) for the agreed phase: the fold's agreed LIDC lesions plus every NLST501 row (no
    malignancy_median; follow-up, not opinion), as plain lists or, with size_balance, size_balanced_lists's
    (a band is used when these hold at least 5 of each label). info: lesions, malignant, benign, balance."""
    a = trn[trn.agreed | trn.malignancy_median.isna()]
    info = {"lesions": int(a.agreed.sum()), "extra": int(a.malignancy_median.isna().sum()),
            "malignant": int(a.label.sum()), "benign": int((a.label == 0).sum()), "balance": None}
    if cfg.get("size_balance"):
        pos, neg, info["balance"] = size_balanced_lists(a)
    else:
        cols = ["seriesuid", "coordX", "coordY", "coordZ"]
        pos = [tuple(r) for r in a[a.label == 1][cols].itertuples(index=False)]
        neg = [tuple(r) for r in a[a.label == 0][cols].itertuples(index=False)]
    return pos, neg, info


def agreed_line(info, steps):
    s = ("agreed_first: the first %d steps draw from %d agreed lesions (+ %d NLST501 rows), %d malignant and %d benign"
         % (steps, info["lesions"], info["extra"], info["malignant"], info["benign"]))
    if info["balance"] is not None:
        s += "; " + size_balance_line(info["balance"])
    return s


def size_balanced_lists(trn, bands=SIZE_BANDS, min_per_label=SIZE_MIN_PER_LABEL, total=SIZE_LIST_LEN):
    """(pos, neg, info): lesion keys repeated so that a uniform draw from pos, and from neg, lands in band b
    with probability w_b = (b's training lesions) / (all in used bands). trn needs seriesuid, coordX/Y/Z, label,
    diameter_mm. Pure and deterministic (table order within a band)."""
    cols = ["seriesuid", "coordX", "coordY", "coordZ"]
    d = trn.reset_index(drop=True)
    used, dropped, info_used = [], [], []
    for lo, hi in bands:
        m = (d.diameter_mm >= lo) & (d.diameter_mm < hi)
        n1, n0 = int((m & (d.label == 1)).sum()), int((m & (d.label == 0)).sum())
        if n1 >= min_per_label and n0 >= min_per_label:
            used.append((lo, hi, m, n1, n0))
        else:
            dropped.append({"band": [lo, hi], "malignant": n1, "benign": n0})
    n_used = sum(n1 + n0 for _, _, _, n1, n0 in used)
    pos, neg = [], []
    for lo, hi, m, n1, n0 in used:
        w = (n1 + n0) / n_used
        for lab, out, n in ((1, pos, n1), (0, neg, n0)):
            keys = [tuple(r) for r in d[m & (d.label == lab)][cols].itertuples(index=False)]
            out.extend(keys * max(1, int(round(w * total / n))))
        info_used.append({"band": [lo, hi], "malignant": n1, "benign": n0, "share": w})
    info = {"used": info_used, "dropped": dropped, "left_out": int(len(d) - n_used)}
    return pos, neg, info


def size_balance_line(info):
    u = ", ".join("%g-%g mm %d/%d (%.1f%%)" % (b["band"][0], b["band"][1], b["malignant"], b["benign"], 100 * b["share"])
                  for b in info["used"]) or "none"
    x = ", ".join("%g-%g mm %d/%d" % (b["band"][0], b["band"][1], b["malignant"], b["benign"])
                  for b in info["dropped"]) or "none"
    return "size_balance: used (cancer/benign, share) %s; dropped %s; %d training lesions left out" % (u, x, info["left_out"])


def train_fold_net(scans, pos, neg, cfg, k, path, fp, early=None, mo=None):
    """Train fold k's net (nodule_fpr.train_net), saving fold<k>.partial.pt beside `path` every
    RAIDIOLOGY_SAVE_EVERY_S seconds (core.ckpt, default 300) and continuing from one with the same fingerprint;
    the final checkpoint is saved at `path` and the partial removed. `early` = (pos, neg, steps) draws the first
    `steps` steps from those lists, then from pos and neg. `mo` = (q, mo_pos, mo_neg) (--extra-multiomics) is set on
    RamSampler for the fold's training and cleared after. Returns (net, seconds, peak card GB)."""
    partial = os.path.join(os.path.dirname(path), "fold%d.partial.pt" % k)
    sampler = nf.Sampler
    first, then = (pos, neg), None
    if early is not None:
        first, then = (early[0], early[1]), (early[2], pos, (neg, []))
    if ram_ok(cfg):
        t = time.time()
        n = fill_ram(scans, list(dict.fromkeys(pos + neg + first[0] + first[1] + (mo[1] + mo[2] if mo else []))))
        print("fold %d: %d lesion windows read into memory in %.0f s (%d held)"
              % (k, n, time.time() - t, len(_RAM)), flush=True)
        nf.Sampler = RamSampler
        RamSampler.mo = mo
    try:
        net, secs, peak = nf.train_net(scans, first[0], (first[1], []), cfg, 1000 * int(cfg["seed"]) + k + 1,
                                       "fold %d" % k, 0.0, partial, fp, then)  # prints "fold K: resumed mid-fold at step S of N"
    finally:
        nf.Sampler = sampler
        RamSampler.mo = None
    torch.save({"state": net.state_dict(), "cfg": cfg, "fingerprint": fp, "seconds": secs, "peak_gb": peak}, path)
    ckpt.clear_partial(partial)
    return net, secs, peak


def train(a):
    cfg = with_settings(CFG, a.set)
    if cfg.get("nlst_fix") and not a.extra_nlst:
        print("--set nlst_fix=true needs --extra-nlst (it corrects NLST501's labels)", file=sys.stderr)
        return 2
    if a.extra_multiomics:
        labels_csv = os.path.join(MO_VOL, "labels.csv")
        why = (("%s does not exist (multiomics.py writes it)" % labels_csv) if not os.path.exists(labels_csv)
               else "it cannot be combined with size_balance" if cfg.get("size_balance")
               else "it cannot be combined with agreed_first" if float(cfg.get("agreed_first", 0.0)) > 0
               else "it needs RAM crops (the source-balanced draw lives in RamSampler)" if not ram_ok(cfg) else "")
        if why:
            print("--extra-multiomics: " + why, file=sys.stderr)
            return 2
    card.require_card(nf.DEV, torch)
    limit_malignancy_card()

    df, counts = lesion_table(cfg)
    print("lesions: %(lesions)d matched in LIDC-IDRI; %(with_enough_readers)d marked by enough radiologists" % counts)
    print("  rated: %(rated)d; median exactly 3 (left out): %(undecided_left_out)d; no prepared box: %(no_prepared_box)d"
          % counts)
    print("  labelled: %(labelled)d = %(malignant)d malignant + %(benign)d benign, in %(scans)d scans of %(patients)d"
          " patients" % counts)
    print("  in LUNA16's folds: %(in_luna16_folds)d (scored out of fold); extra scans, training only: "
          "%(extra_training_only)d" % counts)
    for k in range(N_FOLDS):
        d = df[df.fold == k]
        print("  fold %d: %d lesions (%d malignant)" % (k, len(d), int(d.label.sum())))
    two_folds, extra_leak = patient_leaks(df)
    print("patients with scans in two different folds: %d%s" % (len(two_folds), (" (" + ", ".join(two_folds[:10]) + ")") if two_folds else ""))
    print("held-out patients whose other scan is an extra (always training): %d%s"
          % (len(extra_leak), (" (" + ", ".join(extra_leak[:10]) + ")") if extra_leak else ""))
    counts["patients_in_two_folds"] = two_folds
    counts["patients_with_extra_scan_leak"] = extra_leak

    if a.extra_nlst:
        n = nlst_table(bool(cfg.get("nlst_fix")))
        print("extra NLST nodules%s (training only, fold -1): %d = %d malignant + %d benign, in %d scans of %d patients"
              % (" (labels fixed, nlst501_fix.csv)" if cfg.get("nlst_fix") else "", len(n), int(n.label.sum()), int((n.label == 0).sum()), n.seriesuid.nunique(), n.patient.nunique()))
        assert not set(n.seriesuid) & set(df.seriesuid), "an NLST scan is a LIDC scan"
        df = pd.concat([df, n], ignore_index=True)
        counts["extra_nlst"] = len(n)
    if a.extra_multiomics:
        m = multiomics_table()
        print("extra multiomics nodules (training only, fold -1): %d = %d malignant + %d benign, in %d scans of %d patients"
              % (len(m), int(m.label.sum()), int((m.label == 0).sum()), m.seriesuid.nunique(), m.patient.nunique()))
        assert not set(m.seriesuid) & set(df.seriesuid), "a multiomics scan is another table's scan"
        df = pd.concat([df, m], ignore_index=True)
        counts["extra_multiomics"] = len(m)
    run_folds = a.folds if a.folds else list(range(N_FOLDS))
    out = out_root(a.version)
    os.makedirs(out, exist_ok=True)
    table_hash = int(pd.util.hash_pandas_object(
        df[["seriesuid", "coordX", "coordY", "coordZ", "label", "fold"]], index=False).sum()) & (2 ** 63 - 1)
    fp_cfg = fingerprint_cfg(cfg)
    fingerprint = json.loads(json.dumps({"version": a.version, "cfg": fp_cfg, "table": table_hash,
                                         **({"extra_nlst": True} if a.extra_nlst else {}),
                                         **({"extra_multiomics": True} if a.extra_multiomics else {})},
                                        sort_keys=True, default=str))
    boxes = {u: Box(*lp.paths_for(MO_VOL, u)) if u.startswith("mo-") else lidc_box(u) for u in df.seriesuid.unique()}
    name = "nodule malignancy v%d" % a.version
    parts, secs, peaks = [], {}, []
    balance = {}      # fold -> size_balanced_lists info, for folds trained in this run

    with ledger.run("nodule", name) as led:
        led.settings_from({"cfg": cfg, "folds": run_folds})
        for k in run_folds:
            held = df[df.fold == k].reset_index(drop=True)
            trn = df[df.fold != k]
            path = os.path.join(out, "fold%d.pt" % k)
            fp = {**fingerprint, "fold": k}
            ck = load_resumable(path, fp)
            if ck is not None and "oof_rows" in ck:
                rows = pd.DataFrame(ck["oof_rows"])
                secs[k], peak = ck["seconds"], ck.get("peak_gb", float("nan"))
                print("fold %d: resumed with its scored rows (no training, no reading)" % k, flush=True)
            else:
                net = nf.build(cfg)
                if ck is not None:
                    net.load_state_dict(ck["state"])
                    secs[k], peak = ck["seconds"], ck.get("peak_gb", float("nan"))
                    print("fold %d: resumed from %s (no training)" % (k, path), flush=True)
                else:
                    cols = ["seriesuid", "coordX", "coordY", "coordZ"]
                    mo, mo_pos, mo_neg = None, [], []
                    if a.extra_multiomics:
                        is_mo = trn.patient.str.startswith("mo-")
                        mo_trn, trn = trn[is_mo], trn[~is_mo]
                        mo_pos = [tuple(r) for r in mo_trn[mo_trn.label == 1][cols].itertuples(index=False)]
                        mo_neg = [tuple(r) for r in mo_trn[mo_trn.label == 0][cols].itertuples(index=False)]
                        if not mo_pos or not mo_neg:
                            raise SystemExit("fold %d has no multiomics cancers or no multiomics benign nodules" % k)
                        q = len(mo_trn) / (len(mo_trn) + len(trn))
                        mo = (q, mo_pos, mo_neg)
                        print("fold %d: multiomics share of draws q=%.2f (%d cancers, %d benign)"
                              % (k, q, len(mo_pos), len(mo_neg)), flush=True)
                    pos = [tuple(r) for r in trn[trn.label == 1][cols].itertuples(index=False)]
                    neg = [tuple(r) for r in trn[trn.label == 0][cols].itertuples(index=False)]
                    print("fold %d: training on %d malignant and %d benign lesions, scoring %d"
                          % (k, len(pos), len(neg), len(held)), flush=True)
                    if cfg.get("size_balance"):
                        pos, neg, bal = size_balanced_lists(trn)
                        balance[k] = bal
                        print("fold %d: %s" % (k, size_balance_line(bal)), flush=True)
                    if not pos or not neg:
                        raise SystemExit("fold %d has no training data" % k)
                    early = None
                    n_early = min(int(round(float(cfg.get("agreed_first", 0.0)) * cfg["steps"])), int(cfg["steps"]))
                    if n_early > 0:
                        pos_a, neg_a, info_a = agreed_phase(trn, cfg)
                        print("fold %d: %s" % (k, agreed_line(info_a, n_early)), flush=True)
                        if not pos_a or not neg_a:
                            raise SystemExit("fold %d has no agreed training data" % k)
                        early = (pos_a, neg_a, n_early)
                    scans = {u: boxes[u] for u in {key[0] for key in pos + neg + mo_pos + mo_neg}}
                    net, secs[k], peak = train_fold_net(scans, pos, neg, cfg, k, path, fp, early, mo)
                rows = held[["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "label", "fold"]].copy()
                if nf.DEV.type == "cuda":
                    # training's freed blocks stay reserved and split; scoring's 64-crop layers need whole ones
                    # (v2's fold 1 stopped here, 2026-10-03: 973 MiB reserved and unused, 648 MiB asked for)
                    torch.cuda.empty_cache()
                rows["logit"] = logits_of(net, boxes, held)
                rows["p"] = sigmoid64(rows.logit)
                rows["patient"] = held.patient
                torch.save({"state": net.state_dict(), "cfg": cfg, "fingerprint": fp, "seconds": secs[k],
                            "peak_gb": peak, "oof_rows": rows.to_dict("list")}, path)
                del net
                if nf.DEV.type == "cuda":
                    torch.cuda.empty_cache()
            peaks.append(peak)
            parts.append(rows)
            if len(rows) and rows.label.nunique() == 2:
                print("fold %d: AUROC %.4f on %d held-out lesions" % (k, auroc(rows.label, rows.logit), len(rows)))

        oof = pd.concat(parts, ignore_index=True)
        oof[OOF_COLS + ["logit"]].to_csv(os.path.join(out, "oof.csv"), index=False)
        print("distinct scores: %d of %d lesions" % (oof.logit.nunique(), len(oof)))
        point, lo, hi = bootstrap_auroc(oof.label, oof.logit, oof.patient)
        print("out-of-fold AUROC %.4f (95%% interval %.4f to %.4f) on %d lesions of %d patients"
              % (point, lo, hi, len(oof), oof.patient.nunique()))
        led.number("auroc_oof", point).number("auroc_oof_low", lo).number("auroc_oof_high", hi)
        bands = {}
        for i, (label, a_mm, b_mm) in enumerate(BANDS):      # under 10 mm; 10 to 20 inclusive; over 20
            d = oof[(oof.diameter_mm >= a_mm) & (oof.diameter_mm <= b_mm) if i == 1 else
                    (oof.diameter_mm >= a_mm) & (oof.diameter_mm < b_mm) if i == 0 else oof.diameter_mm > a_mm]
            v = auroc(d.label, d.logit)
            bands[label] = {"auroc": v, "lesions": len(d), "malignant": int(d.label.sum())}
            print("  %s: AUROC %s on %d lesions (%d malignant)"
                  % (label, "%.4f" % v if v == v else "n/a (one class only)", len(d), int(d.label.sum())))
            led.number("auroc_band_%d" % i, v)
        led.number("train_seconds_per_fold", float(np.mean(list(secs.values()))), "s")
        led.number("peak_memory_gb", float(np.nanmax(peaks)) if peaks else float("nan"))
        led.number("lesions", len(oof), "count")
        meta = {"name": name, "version": a.version, "folds": run_folds, "cfg": cfg, "counts": counts,
                **({"extra_nlst": True} if a.extra_nlst else {}),
                **({"extra_multiomics": True} if a.extra_multiomics else {}),
                **({"nlst_fix": True} if cfg.get("nlst_fix") else {}),
                **({"size_balance": {str(k): v for k, v in balance.items()}} if cfg.get("size_balance") else {}),
                "auroc_oof": point, "auroc_oof_interval": [lo, hi], "bands": bands,
                "train_seconds_per_fold": {str(k): v for k, v in secs.items()},
                "peak_memory_gb": float(np.nanmax(peaks)) if peaks else float("nan"),
                "commit": ledger._commit(), "torch": torch.__version__, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    return 0


# ---- lungx ---------------------------------------------------------------------------------------

def lungx(a):
    which = a.set
    out = out_root(a.version)
    opened = os.path.join(out, "lungx_test_opened.json")
    if which == "test":
        if not a.open_test:
            raise SystemExit("the LUNGx test set is opened once: add --open-test to open it")
        if os.path.exists(opened):
            raise SystemExit("the LUNGx test set was already opened for this version: " + opened)
    ckpts = sorted(f for f in os.listdir(out) if f.startswith("fold") and f.endswith(".pt")
                   and not f.endswith(".partial.pt")) if os.path.isdir(out) else []
    if not ckpts:
        raise SystemExit("no fold models in " + out)
    nodules_csv = os.path.join(LUNGX_DIR, "nodules.csv")
    if not os.path.exists(nodules_csv):
        raise SystemExit("no " + nodules_csv + " (lungx.py writes it)")
    df = pd.read_csv(nodules_csv)
    df = df[df["set"].astype(str) == which].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no %s nodules in %s" % (which, nodules_csv))
    card.require_card(nf.DEV, torch)
    limit_malignancy_card(SCORING_GB)
    boxes = {u: lungx_box(u) for u in df.seriesuid.unique()}
    nets = []
    for f in ckpts:
        ck = torch.load(os.path.join(out, f), map_location=nf.DEV, weights_only=False)
        net = nf.build(ck["cfg"])
        net.load_state_dict(ck["state"])
        nets.append(net)
    print("LUNGx %s: %d nodules (%d malignant, %d benign) from %d scans; mean of %d fold models' logits"
          % (which, len(df), int(df.label.sum()), int((df.label == 0).sum()), df.seriesuid.nunique(), len(nets)))

    with ledger.run("nodule", "nodule malignancy v%d on LUNGx %s" % (a.version, which)) as led:
        led.settings_from({"version": a.version, "set": which, "models": ckpts})
        if which == "test":
            with open(opened, "w", encoding="utf-8") as f:       # first: the set counts as opened from here
                json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": a.version, "auroc": None}, f, indent=1)
        per_model = np.stack([logits_of(net, boxes, df, SCORING_BATCH) for net in nets])
        if nf.DEV.type == "cuda":
            print("scoring's peak card: %.2f GB allocated, %.2f GB reserved (lease %.1f GB)"
                  % (torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30,
                     min(SCORING_GB, card_gb())))
        df["logit"] = per_model.mean(axis=0)
        df["p"] = sigmoid64(df.logit)
        groups = df["patient"] if "patient" in df else pd.Series(np.arange(len(df)))
        point, lo, hi = bootstrap_auroc(df.label, df.logit, groups)
        if which == "test":
            with open(opened, "w", encoding="utf-8") as f:
                json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": a.version, "auroc": point,
                           "interval": [lo, hi], "nodules": len(df)}, f, indent=1)
        df.to_csv(os.path.join(out, "lungx_%s_scores.csv" % which), index=False)
        singles = [auroc(df.label, z) for z in per_model]
        print("single fold models: AUROC " + ", ".join("%.3f" % v for v in singles))
        print("LUNGx %s AUROC %.4f (95%% interval %.4f to %.4f, %d bootstrap resamples of patients)"
              % (which, point, lo, hi, BOOT))
        print("paper's bar (Armato 2016, 73 test nodules): eleven computer methods %.2f to %.2f, six radiologists %.2f to %.2f"
              % (BAR_METHODS + BAR_RADIOLOGISTS))
        led.number("auroc_lungx_" + which, point).number("auroc_lungx_%s_low" % which, lo)
        led.number("auroc_lungx_%s_high" % which, hi).number("nodules", len(df), "count")
    return 0


def _selftest_resume():
    """Mid-fold resume of train_fold_net on the processor, tiny net, stand-in sampler: a run stopped on purpose right
    after its save at step 3 of 6 (RAIDIOLOGY_CKPT_SELFTEST_STOP) is continued from step 3 with the saved optimizer
    state, running steps 4 to 6 only; the partial is gone once fold<k>.pt is saved; a corrupted partial is renamed
    .bad and training starts fresh."""
    import contextlib
    import io
    import tempfile

    class Stand:                                            # batches that depend only on the seed it is given
        made, drawn, check = [], [0], []

        def __init__(self, scans, positives, negatives, cfg, seed, batch, hard_share=0.5):
            self.rng, self.batch, self.stop = np.random.default_rng(seed), batch, threading.Event()
            Stand.made.append(seed)
            if Stand.check:                                 # right after a resume, before any step
                Stand.check[0](opts[-1])

        def get(self):
            Stand.drawn[0] += 1
            imgs = torch.from_numpy(self.rng.integers(0, 255, (self.batch, 1, nf.CUBE, nf.CUBE, nf.CUBE), dtype=np.uint8))
            return imgs, torch.from_numpy((self.rng.random(self.batch) < 0.5).astype(np.float32))

    opts = []
    real_adamw = torch.optim.AdamW

    def adamw(*args, **kw):
        opts.append(real_adamw(*args, **kw))
        return opts[-1]

    cfg = dict(CFG, widths=[4, 4], groups=2, steps=6, batch=2, warmup=2, log_every=100, ram_crops=False)
    seed = 1000 * int(cfg["seed"]) + 1 + 1                  # train_fold_net's seed for fold 1

    def run(path, fp):
        Stand.drawn[0] = 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = train_fold_net({}, [("a", 0.0, 0.0, 0.0)], [("b", 0.0, 0.0, 0.0)], cfg, 1, path, fp)
        return result, out.getvalue(), Stand.drawn[0] - 1   # the feed reads one batch ahead

    env = {"RAIDIOLOGY_SAVE_EVERY_S": "0", "RAIDIOLOGY_SHARE": "1"}
    old = {k: os.environ.get(k) for k in (*env, "RAIDIOLOGY_CKPT_SELFTEST_STOP", "RAIDIOLOGY_NO_RESUME")}
    os.environ.update(env)
    os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
    real_dev, real_sampler, nf.DEV = nf.DEV, nf.Sampler, torch.device("cpu")
    nf.Sampler, torch.optim.AdamW = Stand, adamw
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fold1.pt")
            partial, fp = os.path.join(tmp, "fold1.partial.pt"), {"fold": 1, "version": 0}
            # stopped on purpose right after the save at step 3
            os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = "3"
            try:
                run(path, fp)
                raise AssertionError("the selftest stop did not stop the run")
            except RuntimeError as e:
                assert "step 3" in str(e), e
            assert os.path.exists(partial) and not os.path.exists(partial + ".tmp") and not os.path.exists(path)
            saved = ckpt.load_partial(partial, fp)
            assert saved["step"] == 3 and saved["opt"]["state"] and saved["seconds"] >= 0, saved.keys()
            assert ckpt.load_partial(partial, {"fold": 2, "version": 0}) is None        # another fingerprint: not used
            os.environ.pop("RAIDIOLOGY_CKPT_SELFTEST_STOP")
            # the rerun resumes at step 3 with the saved optimizer state and runs steps 4 to 6 only
            seen = []

            def same_opt(opt):
                now = opt.state_dict()
                assert now["param_groups"] == saved["opt"]["param_groups"], "optimizer settings differ"
                assert now["state"].keys() == saved["opt"]["state"].keys()
                for i, st in saved["opt"]["state"].items():
                    for key, v in st.items():
                        assert torch.equal(torch.as_tensor(now["state"][i][key]), torch.as_tensor(v)), (i, key)
                seen.append(True)

            Stand.check[:] = [same_opt]
            (_, secs, _), text, n = run(path, fp)
            Stand.check.clear()
            assert "fold 1: resumed mid-fold at step 3 of 6 (partial checkpoint)" in text, text
            assert seen == [True] and n == 3, (seen, n)
            assert Stand.made[-1] == seed * 1_000_003 + 3                                 # re-seeded from (seed, step)
            assert secs >= saved["seconds"]
            # the fold's own checkpoint is saved and the partial is gone
            assert os.listdir(tmp) == ["fold1.pt"], os.listdir(tmp)
            assert torch.load(path, weights_only=False)["fingerprint"] == fp               # the partial left the fingerprint alone
            os.remove(path)
            # a corrupted partial is renamed .bad and training starts fresh
            with open(partial, "wb") as f:
                f.write(b"not a checkpoint")
            _, text, n = run(path, fp)
            assert os.path.exists(partial + ".bad") and "resumed" not in text and "unreadable" in text, text
            assert n == 6 and Stand.made[-1] == seed, (n, Stand.made[-1])
            assert not os.path.exists(partial) and os.path.exists(path)
    finally:
        nf.Sampler, nf.DEV, torch.optim.AdamW = real_sampler, real_dev, real_adamw
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("resume selftest ok")


def selftest():
    """No data, no card: size_balanced_lists and the fingerprint on a synthetic table where size predicts the label."""
    rng = np.random.default_rng(0)
    n = 3000
    d = rng.uniform(4.0, 24.0, n)
    lab = (rng.uniform(size=n) < 1 / (1 + np.exp(-(d - 12.0) * 0.8))).astype(int)
    df = pd.DataFrame({"seriesuid": ["s%d" % (i % 50) for i in range(n)], "coordX": np.arange(n, dtype=float),
                       "coordY": 0.0, "coordZ": 0.0, "label": lab, "diameter_mm": d})
    assert auroc(df.label, df.diameter_mm) > 0.8, "synthetic size should predict the label"
    # a band with 4 cancers and many benign: dropped
    bad = (df.diameter_mm >= 6) & (df.diameter_mm < 8) & (df.label == 1)
    df = df.drop(df[bad].index[4:]).reset_index(drop=True)
    pos, neg, info = size_balanced_lists(df)
    assert [b["band"] for b in info["dropped"]] == [[6.0, 8.0]], info["dropped"]
    assert len(info["used"]) == len(SIZE_BANDS) - 1

    pos_set, neg_set = set(pos), set(neg)
    key = {(r.seriesuid, r.coordX, r.coordY, r.coordZ): r.diameter_mm for r in df.itertuples(index=False)}
    lo_used = min(b["band"][0] for b in info["used"])
    hi_used = max(b["band"][1] for b in info["used"])
    for ks in (pos_set, neg_set):
        assert all(lo_used <= key[k] < hi_used for k in ks), "a lesion outside the used bands was kept"      # (b), (c)
        assert not any(6.0 <= key[k] < 8.0 for k in ks), "dropped band present"
    assert any(k not in pos_set | neg_set for k in key), "nothing was left out"
    for b in info["used"]:
        for ks in (pos, neg):
            share = sum(b["band"][0] <= key[k] < b["band"][1] for k in ks) / len(ks)
            assert abs(share - b["share"]) < 0.01, (b, share)                                                 # (a)
    rows = pd.DataFrame({"d": [key[k] for k in pos + neg], "y": [1] * len(pos) + [0] * len(neg)})
    v = auroc(rows.y, rows.d)
    assert abs(v - 0.5) < 0.06, "size still predicts the label over the balanced lists: %.3f" % v             # (d)
    assert info["left_out"] == len(df) - sum(b["malignant"] + b["benign"] for b in info["used"])
    assert size_balanced_lists(df)[0] == pos, "not deterministic"

    base = {k: v for k, v in CFG.items() if k not in ("size_balance", "agreed_first", "nlst_fix")}
    assert fingerprint_cfg({**base, "size_balance": False}) == fingerprint_cfg(base) == \
        {k: v for k, v in base.items() if k != "ram_crops"}                                                   # (e)
    assert fingerprint_cfg({**base, "size_balance": True}) != fingerprint_cfg(base)
    assert fingerprint_cfg({**base, "agreed_first": 0.0}) == fingerprint_cfg(base)
    assert fingerprint_cfg({**base, "agreed_first": 0.5})["agreed_first"] == 0.5
    assert fingerprint_cfg({**base, "nlst_fix": False}) == fingerprint_cfg(base)
    assert fingerprint_cfg({**base, "nlst_fix": True})["nlst_fix"] is True

    # the agreed phase: only agreed LIDC lesions and the NLST rows; the full lists are the whole fold
    m = rng.integers(1, 6, size=len(df))
    tr = df.assign(malignancy_median=m.astype(float), agreed=rng.uniform(size=len(df)) < 0.3)
    nl = tr.iloc[:40].assign(seriesuid="nlst", malignancy_median=np.nan, agreed=False)
    tr = pd.concat([tr.iloc[40:], nl], ignore_index=True)
    plain = {"size_balance": False}
    pa, na, info = agreed_phase(tr, plain)
    ok = {(r.seriesuid, r.coordX, r.coordY, r.coordZ) for r in tr[tr.agreed | tr.malignancy_median.isna()].itertuples()}
    assert set(pa) | set(na) <= ok and len(pa) + len(na) == len(ok), "the agreed phase drew a split lesion"
    assert any(k[0] == "nlst" for k in pa + na), "NLST rows left out of the agreed phase"
    assert info["lesions"] == int(tr.agreed.sum()) and info["extra"] == 40
    cols = ["seriesuid", "coordX", "coordY", "coordZ"]
    full_pos = [tuple(r) for r in tr[tr.label == 1][cols].itertuples(index=False)]
    assert len(full_pos) > len(pa) and set(pa) < set(full_pos), "the full lists are not the agreed ones' superset"
    pb, nb, infob = agreed_phase(tr, {"size_balance": True})
    assert infob["balance"] is not None and set(pb) | set(nb) <= ok
    assert "agreed_first" in agreed_line(infob, 100)
    _selftest_resume()
    print("selftest ok")
    return 0


def main():
    if sys.argv[1:2] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--version", type=int, required=True)
    t.add_argument("--folds", type=int, nargs="+", choices=range(N_FOLDS))
    t.add_argument("--set", action="append", default=[], help="key=<json>; keys: " + ", ".join(sorted(CFG)))
    t.add_argument("--extra-nlst", action="store_true",
                   help="also train on nlst501.labels() (follow-up labels; never held out or scored)")
    t.add_argument("--extra-multiomics", action="store_true",
                   help="also train on the pathology-proven multiomics nodules (<MO_VOL>/labels.csv; fold -1, training "
                        "only; each draw takes them with probability q, then half cancer, half benign)")
    x = sub.add_parser("lungx")
    x.add_argument("--version", type=int, required=True)
    x.add_argument("--set", required=True, choices=["calibration", "test"])
    x.add_argument("--open-test", action="store_true", help="needed for --set test; the test set opens once")
    a = ap.parse_args()
    return train(a) if a.cmd == "train" else lungx(a)


if __name__ == "__main__":
    sys.exit(main())
