"""The second stage of the lung nodule detector: re-score stage 1's candidates with a small 3D CNN.

Stage 1 (nodule.py) finds the big nodules but ranks many false alarms as high as real nodules.
Stage 2 re-scores every candidate from a 48 mm crop around it with a binary classifier, the classic
LUNA16 false-alarm reducer. The candidate list CPM (froc.py) checks whether re-scoring by p1*p2
beats stage 1 alone.

Why crops. The detection network sees a large context; the classifier needs only the local evidence
and can be tiny. A 48 mm cube at 1 mm per voxel is 48^3 voxels, cheap to read from the prepared .npy
and fast to process. A pool of threads feeds batches into a queue; the card trains on pinned memory
while the threads build the next batch on the CPU.

What the candidate list scores. After re-scoring, candidates_V2's points outside every nodule's
radius (to keep the false-negative rate honest) are scored as a simpler "candidate list" metric: it
captures whether the system would pass a check on a vendor's point cloud alone, without stage 1's
probabilities.

    python modalities/ct/eval/nodule_fpr.py rescore --version M (--stage1 N | --stage1-file F) [--tag T]
        [--hard N | --hard-file F] [--extra-nlst BLOBS --extra-nlst-hard TOP] [--ten] [--skip-v2]
        [--folds ...] [--steps S] [--set key=value]
    python modalities/ct/eval/nodule_fpr.py smoke [--set key=value]   (e.g. --set context=2 --set zoom=0.2)
    python modalities/ct/eval/nodule_fpr.py selftest   (processor only: tta_top's picker, scoring, fingerprint, stamp, mid-fold resume)

Reading speed. `--set tta=8` reads every place through eight mirror images (8x the card time); adding
`--set tta_top=32` reads every place once and only each scan's 32 best by that reading eight ways (the same
score, far less card time). The fold's line says how long the loop waited on the crop reader.

The lung finder's encoder. `--set arch="det"` makes stage 2's net the lung finder's encoder (nodule_det.py's stem and
four levels) with the classifier's head, and `--set init_det=N` starts each fold's encoder from det-vN's net of the
same fold. Both are allowed only at windows 1, context 1, mirror 0 and paste 0; the defaults ("classifier", 0) change
nothing.
"""
import argparse
import hashlib
import json
import math
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
from scipy.ndimage import affine_transform
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)
import froc  # noqa: E402
import lung_prepare as lp  # noqa: E402
from core import card, ckpt, ledger, paths, polite  # noqa: E402
from nodule import Scan, reference, scan_folds, limit_card, sha, load_resumable, with_settings  # noqa: E402

DEV = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
CUBE = 48
LUNA_DIR = os.path.join(paths.DATASETS, "public", "luna16")
CFG = {
    "widths": [24, 48, 96, 192], "groups": 8,
    "steps": 12000, "batch": 64, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 300,
    "nodule_share": 0.5, "shift_mm": 3.0, "hard_share": 0.5, "min_radius_mm": 2.0, "max_radius_mm": 0.0,
    "excluded_min_radius_mm": 3.0, "excluded_margin_mm": 2.0, "brightness": 0.05,
    "seed": 0, "log_every": 200, "queue": 3, "threads": 6,
    "windows": 1, "tta": 1, "tta_top": 0, "context": 1, "zoom": 0.0,
    "norm": "group", "mirror": 0, "paste": 0.0, "init_from": "", "arch": "classifier", "init_det": 0,
}
# `arch` "det" (bar 141): the net is nodule_det.Det's encoder (stem, e1, d2, e2, d3, e3, d4, e4, the same module names,
# so a det checkpoint's tensors load by name, widths nodule_det's) with Classifier's head. It needs windows 1, context 1,
# mirror 0, paste 0, norm "group" (norm means nothing for it: the encoder is BatchNorm throughout) and no init_from.
# `init_det` N (needs arch "det"): fold k's encoder starts from DET_ROOT/det-vN/fold<k>.pt, BatchNorm statistics
# included; the detector's pyramid and heads are not read, and the head starts fresh. det-vN's nets are on nodule.py's
# five folds, so --ten is refused. The fingerprint takes the five files' sha256. At the defaults ("classifier", 0)
# nothing changes, and fingerprint_cfg leaves both out, so saved nets resume.
DET_ROOT = os.path.join(paths.WORK, "eval", "lung")
DET_TRUNK = ("stem", "e1", "d2", "e2", "d3", "e3", "d4", "e4")
# `init_from` PATH (idea 107, docs/efficiency, 'Lung: stage 2 starts from a pretrained trunk'): the classifier's
# stages start from an encoder file nodule_genesis.py writes (its 'state' holds the stages.* tensors, its 'settings'
# the net's shape), and only the head starts fresh. A relative path is taken from the repo root. Only training starts
# from it; a saved net is scored as it was. The fingerprint takes the file's sha256, so a net never resumes from a
# checkpoint trained from another encoder. At the default "" nothing is read, and fingerprint_cfg leaves it out.
# `paste` P (docs/efficiency, 'Lung: stage 2 taught on transplanted nodules'): with chance P a positive draw is a
# transplant, a real annotated nodule of one of the training scans cut out with a soft-edged ball (full weight to its
# radius + 2 mm, then a 2 mm linear ramp) and blended into a negative place as the brighter of the two, label 1. It
# needs context 1, zoom 0 and mirror 0 (check_paste). At the default 0 the sampler draws and reads nothing extra, and
# fingerprint_cfg leaves `paste` out, so saved nets resume.
# `mirror` 1 gives every cube a twin (docs/efficiency, 'Lung: stage 2 sees the mirror place too'): the same read
# (size, context, zoom) at the place mirrored across the scan's left-right midline, flipped along x so its anatomy
# lines up with the cube's, so the net can compare the two sides. The midline is the middle of the lung mask's x
# extent ((min x + max x) / 2 in voxels; a scan with an empty mask uses the volume's centre column and is counted
# in the fold's log line). Training mirrors the jittered centre and uses the same zoom; scoring reads the twin
# without zoom. The twin's channels follow the cube's, so the net sees context x windows x (1 + mirror) channels.
# At the default 0 nothing extra is read or computed, and fingerprint_cfg leaves `mirror` out, so saved nets resume.
# `norm` "batch" puts a BatchNorm3d of the same channel count in place of each GroupNorm (2026-10-04: GroupNorm
# runs in float32 under autocast and the casts around it took 1.1 s of card time in 100 steps, 0.12 s with
# BatchNorm; docs/efficiency 'The card profile'); the default "group" is today's net, and fingerprint_cfg
# leaves `norm` out then, so every saved net still resumes.
# `context` 2 gives each cube a second channel: the 96 mm cube around the same centre (same jitter) at 2 mm a
# voxel, a 96^3 read mean-pooled to 48^3, padded with 0 outside the volume. It multiplies with `windows` (every
# raw channel is expanded). `zoom` Z resamples each training cube about its centre by s, log(s) uniform in
# [-Z, Z] (0.2 is about 0.82 to 1.22), the context channel by the same s; scoring never zooms. At the defaults
# (1 and 0.0) the sampler draws and reads nothing extra, and fingerprint_cfg leaves both out, so saved nets resume.
# `windows` 3 shows the net each cube through three CT windows, like a radiologist switching presets:
# channel 0 is the stored value (HU -1200 to 600), channel 1 the faint window (HU -1000 to -300), channel 2
# the soft-tissue window (HU -160 to 240), each scaled to 0..1. The default 1 is today's single channel, and
# fingerprint_cfg leaves `windows` out of the fingerprint then, so every saved net still resumes.
WINDOWS = ((-1000.0, -300.0), (-160.0, 240.0))
# `tta` N reads each cube through the first N of these mirror images (axes of (n, c, z, y, x)) and averages
# the probabilities. A scoring setting only: never in the fingerprint, so any saved net is read with it.
FLIPS = ((), (4,), (3,), (2,), (3, 4), (2, 4), (2, 3), (2, 3, 4))
# `tta_top` K (with tta above 1) reads every place once, then through the `tta` mirror images only each scan's K
# places with the highest single reading; the rest keep the single reading (docs/efficiency, 'The eight-way reading
# on each scan's top 32 places only': the same score at a fraction of the card time). 0, the default, reads every
# place `tta` ways. Also a scoring setting: never in the fingerprint, named in the scoring stamp.
SYNC_EVERY = 4
SCORING = "float32 head"   # how p2 is computed; saved scores computed another way are scored again


def out_root(version):
    return os.path.join(paths.WORK, "eval", "lung", "fpr-v%d" % version)


def card_gb():
    return float(os.environ.get("RAIDIOLOGY_FPR_CARD_GB", "2.5"))


_LIMITED = []


# A rescore whose every fold resumes a saved net only scores: it leases this much and reads 32 crops at a time,
# so it fits beside training jobs instead of waiting for one to end (the cancer model's scoring, same idea,
# peaked at 0.43 GB reserved at 16 crops; 2026-10-03).
SCORING_GB, SCORING_BATCH = 1.4, 32


def limit_fpr_card(gb=None):
    """Lease this process's share of the card, then a hard ceiling at it; `gb` asks for less than
    RAIDIOLOGY_FPR_CARD_GB's share (scoring only), never more."""
    if DEV.type != "cuda" or _LIMITED:
        return
    gb = card_gb() if gb is None else min(gb, card_gb())
    card.lease(gb, "nodule fpr")
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / total), torch.cuda.current_device())
    _LIMITED.append(gb)


# ---- the data --------------------------------------------------------------------------------

def candidates_frame(candidates_csv, folds_of):
    """Load candidates_V2.csv, filter to scans we have, return with fold column."""
    df = pd.read_csv(candidates_csv)
    df = df[df.seriesuid.isin(folds_of.keys())].copy()
    df["fold"] = df.seriesuid.map(folds_of)
    return df


def _xyz(df):
    return df[["coordX", "coordY", "coordZ"]].to_numpy(float)


def _outside(points, centres, radii):
    """Boolean mask of the (n, 3) points that lie outside every ball (centres (m, 3), radii (m,))."""
    if not len(centres) or not len(points):
        return np.ones(len(points), bool)
    d = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return (d >= radii[None, :]).all(axis=1)


def build_training_data(annotations, excluded, candidates_v2, train_uids, cfg, hard_candidates=None):
    """(positives, negatives, hard_negatives) for ONE fold: lists of (seriesuid, x, y, z) over `train_uids`.

    Positives: every annotation centre and every class-1 point. Negatives: class-0 points outside every
    excluded ball (radius max(diameter/2, 3 mm) + 2 mm) and outside every nodule's radius (max(diameter/2,
    2 mm)). Hard negatives: stage-1 candidate points of those scans outside the same balls."""
    pos, neg, hard = [], [], []
    train = set(train_uids)
    ann_g = {u: g for u, g in annotations[annotations.seriesuid.isin(train)].groupby("seriesuid")}
    exc_g = {u: g for u, g in excluded[excluded.seriesuid.isin(train)].groupby("seriesuid")}
    c2_g = {u: g for u, g in candidates_v2[candidates_v2.seriesuid.isin(train)].groupby("seriesuid")}
    hc_g = {}
    if hard_candidates is not None:
        hc_g = {u: g for u, g in hard_candidates[hard_candidates.seriesuid.isin(train)].groupby("seriesuid")}
    for uid in sorted(train):
        a_g, e_g = ann_g.get(uid, annotations.iloc[:0]), exc_g.get(uid, excluded.iloc[:0])
        a_c = _xyz(a_g)
        a_r = np.maximum(a_g.diameter_mm.to_numpy(float) / 2.0, cfg["min_radius_mm"])
        e_c = _xyz(e_g)
        e_r = (np.maximum(np.maximum(e_g.diameter_mm.to_numpy(float), 0.0) / 2.0, cfg["excluded_min_radius_mm"])
               + cfg["excluded_margin_mm"])
        pos += [(uid, *r) for r in a_c.tolist()]
        g = c2_g.get(uid)
        if g is not None:
            one = g["class"].to_numpy() == 1
            pos += [(uid, *r) for r in _xyz(g)[one].tolist()]
            z = _xyz(g)[~one]
            keep = _outside(z, a_c, a_r) & _outside(z, e_c, e_r)
            neg += [(uid, *r) for r in z[keep].tolist()]
        h = hc_g.get(uid)
        if h is not None:
            z = _xyz(h)
            keep = _outside(z, a_c, a_r) & _outside(z, e_c, e_r)
            hard += [(uid, *r) for r in z[keep].tolist()]
    return pos, neg, hard


# ---- crops --------------------------------------------------------------------------------------

def crop_of(scan, centre, size=CUBE, flip=False):
    """`size` mm cube (size^3 voxels uint8, 48 unless asked) centred on world coordinates, padded with 0 outside;
    `flip` turns it along x (the last axis) as it is read."""
    img = _crop_of(scan, centre, size)
    return img[..., ::-1] if flip else img


def _crop_of(scan, centre, size):
    meta = scan.meta
    c_voxel = lp.world_to_voxel(meta, np.array([centre]).reshape(1, 3))[0]
    start_voxel = np.rint(c_voxel).astype(int) - size // 2

    img = np.zeros((size,) * 3, np.uint8)
    src_slices, dst_slices = [], []
    for s, n in zip(start_voxel, scan.shape):
        a, b = max(s, 0), min(s + size, n)
        if b <= a:              # wholly outside the lung box (LUNA16's points span the whole scan)
            return img
        src_slices.append(slice(a, b))
        dst_slices.append(slice(a - s, b - s))

    vol = np.load(scan.path, mmap_mode="r")
    img[tuple(dst_slices)] = vol[tuple(src_slices)]
    return img


def pool2(img):
    """A uint8 cube mean-pooled 2x2x2 (rounded), in numpy: half the side, the same centre."""
    n = img.shape[0] // 2
    s = img.reshape(n, 2, n, 2, n, 2).sum(axis=(1, 3, 5), dtype=np.uint16)
    return ((s + 4) >> 3).astype(np.uint8)


def zoomed(img, s, side=CUBE):
    """The central `side`^3 of a larger uint8 cube, resampled about its centre by factor s (trilinear): s above 1
    magnifies. The larger cube is centred like crop_of's, voxel len // 2 being the centre."""
    c = img.shape[0] // 2
    out = affine_transform(img, np.eye(3) / s, offset=c - (side // 2) / s, output_shape=(side,) * 3,
                           order=1, mode="constant", cval=0.0, output=np.float32)
    return np.rint(out).astype(np.uint8)


_MID, _MID_EMPTY, _MID_LOCK = {}, set(), threading.Lock()


def midline_of_mask(mask):
    """Middle of a lung mask's left-right extent, (min x + max x) / 2 over its true voxels (x the last axis), in
    voxels; None for an empty mask."""
    xs = np.flatnonzero(mask.any(axis=(0, 1)))
    return None if len(xs) == 0 else (int(xs[0]) + int(xs[-1])) / 2.0


def midline_of(scan):
    """The scan's midline (voxels of the prepared volume), worked out once and kept (the Sampler's threads share it).
    The mask is nodule_blobs.lung_of's; an empty one gives the volume's centre column and the uid goes to _MID_EMPTY."""
    with _MID_LOCK:
        if scan.uid not in _MID:
            import nodule_blobs as nb                       # only a mirror run needs it
            have = os.path.exists(os.path.join(nb.MASKS, scan.uid + ".npz"))
            hu = None if have else lp.to_hu(np.load(scan.path)).astype(np.float32)
            mid = midline_of_mask(nb.lung_of(scan.uid, hu))
            if mid is None:
                mid = (scan.shape[2] - 1) / 2.0
                _MID_EMPTY.add(scan.uid)
            _MID[scan.uid] = mid
        return _MID[scan.uid]


def mirror_centre(scan, centre):
    """World (x, y, z) of the place mirrored across the scan's midline. One voxel is added along x: a flipped crop
    of even size has its centre voxel one before the original's, so this lines the flipped read up with the cube."""
    v = lp.world_to_voxel(scan.meta, np.asarray(centre, float))
    v[2] = 2 * midline_of(scan) - v[2] + 1
    return lp.voxel_to_world(scan.meta, v)


def cube_of(scan, centre, context=1, s=None, mirror=0):
    """(context, 48, 48, 48) uint8: the 48 mm cube, then at context 2 the 96 mm cube around the same centre at
    2 mm a voxel. `s` (training only) resamples every channel about the centre by that factor; the larger reads
    leave a margin so the resampled cube never runs short of voxels. `mirror` 1 appends the twin's channels
    (same reads at the mirrored centre, flipped along x), so (context x (1 + mirror), 48, 48, 48)."""
    cube = _cube_at(scan, centre, context, s, False)
    if not mirror:
        return cube
    return np.concatenate([cube, _cube_at(scan, mirror_centre(scan, centre), context, s, True)])


def _cube_at(scan, centre, context, s, flip):
    if s is None:
        cubes = [crop_of(scan, centre, flip=flip)]
        if context == 2:
            cubes.append(pool2(crop_of(scan, centre, 2 * CUBE, flip)))
        return np.stack(cubes)
    reach = math.ceil(CUBE / 2 * max(s, 1 / s) + 2)         # voxels from the centre the resampling can read
    cubes = [zoomed(crop_of(scan, centre, 2 * reach, flip), s)]
    if context == 2:                                        # the 2 mm read has `reach` voxels each side once pooled
        cubes.append(zoomed(pool2(crop_of(scan, centre, 4 * reach, flip)), s))
    return np.stack(cubes)


def check_paste(cfg):
    """Refuse a `paste` outside 0..1, or above 0 with context, zoom or mirror (a transplant is one plain cube)."""
    p = cfg.get("paste", 0.0)
    if not 0 <= p <= 1:
        raise SystemExit("paste must be between 0 and 1, got %s" % p)
    if p > 0 and (cfg.get("context", 1) != 1 or cfg.get("zoom", 0.0) != 0 or cfg.get("mirror", 0) != 0):
        raise SystemExit("paste above 0 needs context 1, zoom 0 and mirror 0")


def transplant(nodule, place, centre, radius, ramp=2.0):
    """The nodule cube blended into the place cube (uint8, same shape): weight 1 within `radius` voxels of `centre`,
    falling linearly to 0 at radius + ramp. Voxels are 1 mm here. The result is the brighter of the place and that
    blend, so the donor can only add brightness and its darker lung never leaves a ring round the nodule
    (docs/efficiency, idea 73's amendment)."""
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=float) for n in nodule.shape], indexing="ij"), -1)
    d = np.linalg.norm(g - np.asarray(centre, float), axis=-1)
    w = np.clip(1 - (d - radius) / ramp, 0, 1)
    return np.maximum(place, np.rint(place * (1 - w) + nodule * w)).astype(np.uint8)


def transplant_cube(nod_scan, nod_xyz, jitter_mm, radius_mm, place_scan, place_xyz):
    """(cube, centre): the real nodule at world `nod_xyz` (cut at the jittered centre) blended into the place's cube.
    centre is the nodule's own voxel (z, y, x) inside its cube."""
    nod_xyz = np.asarray(nod_xyz, float)
    moved = lp.world_to_voxel(nod_scan.meta, nod_xyz + jitter_mm)
    centre = CUBE // 2 + lp.world_to_voxel(nod_scan.meta, nod_xyz) - np.rint(moved)
    cube = transplant(crop_of(nod_scan, nod_xyz + jitter_mm), crop_of(place_scan, place_xyz), centre, radius_mm + 2.0)
    return cube, centre


class Sampler:
    """Endless batches of augmented crops for classification, built by background threads."""

    def __init__(self, scans, positives, negatives, cfg, seed, batch, hard_share=0.5):
        check_paste(cfg)
        self.scans, self.cfg, self.batch = scans, cfg, batch
        self.hard_share = hard_share
        self.pasted = 0
        self.lock = threading.Lock()
        self.donors = []                    # (uid, world xyz, radius mm) of every annotated nodule, for `paste`
        if cfg.get("paste", 0.0) > 0:
            for uid, sc in scans.items():
                for zyx_r in np.asarray(sc.whole).reshape(-1, 4):
                    self.donors.append((uid, np.asarray(lp.voxel_to_world(sc.meta, zyx_r[:3]), float), float(zyx_r[3])))
            if not self.donors:
                raise SystemExit("paste %s needs annotated nodules in the training scans, found none" % cfg["paste"])
        self.rng = np.random.default_rng(seed)
        self.pos_list = positives
        self.neg_list, self.hard_neg_list = negatives
        self.q = queue.Queue(maxsize=int(cfg["queue"]))
        self.error = None
        self.stop = threading.Event()
        self.threads = []
        for i in range(cfg["threads"]):
            t = threading.Thread(target=self._run, args=(np.random.default_rng([seed, i]),),
                                 daemon=True, name="fpr-crops")
            t.start()
            self.threads.append(t)

    def augment(self, img, rng):
        # img is (channels, z, y, x): every channel gets the same flips and turn
        for ax in range(3):
            if rng.random() < 0.5:
                img = np.flip(img, axis=ax + 1)
        # Random 90-degree rotation in axial plane (axes y, x = axes 2, 3 for czyx order)
        return np.rot90(img, int(rng.integers(4)), axes=(2, 3))   # a view: the caller copies it once

    def _run(self, rng):
        # Each crop is copied straight into the batch, in numpy: per-crop torch calls in six threads
        # ran 3x slower than one thread (work/fpr_io.py)
        context, zoom = int(self.cfg.get("context", 1)), float(self.cfg.get("zoom", 0.0))
        mirror = int(self.cfg.get("mirror", 0))
        paste = float(self.cfg.get("paste", 0.0))
        try:
            while not self.stop.is_set():
                imgs = np.empty((self.batch, context * (1 + mirror), CUBE, CUBE, CUBE), np.uint8)
                labels = np.empty(self.batch, np.float32)

                for b in range(self.batch):
                    # Half positives, half negatives
                    graft = None
                    if rng.random() < 0.5:
                        label = 1.0
                        if paste and rng.random() < paste:      # no draw at the default
                            graft = self.donors[int(rng.integers(len(self.donors)))]
                            if self.hard_neg_list and rng.random() < self.hard_share:
                                uid, x, y, z = self.hard_neg_list[int(rng.integers(len(self.hard_neg_list)))]
                            else:
                                uid, x, y, z = self.neg_list[int(rng.integers(len(self.neg_list)))]
                        else:
                            uid, x, y, z = self.pos_list[int(rng.integers(len(self.pos_list)))]
                    else:
                        # Negatives: uniform or hard
                        if self.hard_neg_list and rng.random() < self.hard_share:
                            uid, x, y, z = self.hard_neg_list[int(rng.integers(len(self.hard_neg_list)))]
                        else:
                            uid, x, y, z = self.neg_list[int(rng.integers(len(self.neg_list)))]
                        label = 0.0

                    scan = self.scans[uid]
                    jitter = rng.uniform(-self.cfg["shift_mm"], self.cfg["shift_mm"], 3)
                    if graft is not None:
                        donor_uid, donor_xyz, donor_r = graft
                        cube, _ = transplant_cube(self.scans[donor_uid], donor_xyz, jitter, donor_r,
                                                  scan, np.array([x, y, z]))
                        imgs[b] = self.augment(cube.reshape(1, CUBE, CUBE, CUBE), rng)
                        labels[b] = label
                        with self.lock:
                            self.pasted += 1
                        continue
                    s = math.exp(rng.uniform(-zoom, zoom)) if zoom else None   # no draw at the default
                    # the twin (mirror) is read at the mirror of the jittered centre, same s, then flipped and
                    # augmented with the cube
                    imgs[b] = self.augment(cube_of(scan, np.array([x, y, z]) + jitter, context, s, mirror), rng)
                    labels[b] = label

                out, labels = torch.from_numpy(imgs), torch.from_numpy(labels)
                if DEV.type == "cuda":
                    out = out.pin_memory()
                    labels = labels.pin_memory()

                while not self.stop.is_set():
                    try:
                        self.q.put((out, labels), timeout=0.5)
                        break
                    except queue.Full:
                        pass
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.q.put(None)

    def get(self):
        got = self.q.get()
        if got is None:
            raise self.error
        return got


class Feed:
    """Batches on the card: the copy of the next batch runs on a side stream while this one trains."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.side = torch.cuda.Stream() if DEV.type == "cuda" else None
        self.ahead = self._fetch()

    def _fetch(self):
        result = self.sampler.get()
        if result is None:
            raise self.sampler.error
        imgs, labels = result
        if self.side is None:
            return (imgs, labels), None
        with torch.cuda.stream(self.side):
            return (imgs.to(DEV, non_blocking=True), labels.to(DEV, non_blocking=True)), (imgs, labels)

    def next(self):
        (gpu_imgs, gpu_labels), _ = self.ahead
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)
            gpu_imgs.record_stream(torch.cuda.current_stream())
            gpu_labels.record_stream(torch.cuda.current_stream())
        self.ahead = self._fetch()
        return gpu_imgs, gpu_labels


# ---- the network and loss -----------------------------------------------------------------------

def net_input(u8, c_in):
    """The net's input from uint8 cubes (n, context, D, H, W): float32 (n, c_in, D, H, W), on the cubes' device.
    Every raw channel (the twin's too) is expanded the same way, so c_in is context x windows x (1 + mirror)."""
    base = u8.float() / 255.0
    if c_in == u8.shape[1]:
        return base
    hu = u8.float() * lp.STEP_HU + lp.HU_LO
    return torch.cat([base] + [((hu - lo) / (hi - lo)).clamp(0.0, 1.0) for lo, hi in WINDOWS], dim=1)


def fingerprint_cfg(cfg):
    """The cfg a checkpoint's fingerprint holds: without `windows` or `context` at 1, `zoom` or `mirror` at 0 or
    `norm` at "group", as before those settings existed, and never `tta` or `tta_top`, which change only how a
    trained net is read."""
    default = {"windows": 1, "context": 1, "zoom": 0.0, "norm": "group", "mirror": 0, "paste": 0.0, "init_from": "",
               "arch": "classifier", "init_det": 0}
    out = {k: v for k, v in cfg.items()
           if k not in ("tta", "tta_top") and not (k in default and v == default[k])}
    if cfg.get("init_from"):
        out["init_from"] = {"path": cfg["init_from"], "sha256": sha(trunk_path(cfg))}
    if cfg.get("init_det"):
        n = int(cfg["init_det"])
        missing = [det_path(n, k) for k in range(5) if not os.path.exists(det_path(n, k))]
        if missing:
            raise SystemExit("init_det %d needs det-v%d's five fold files; missing %s" % (n, n, ", ".join(missing)))
        out["init_det"] = {"version": n, "sha256": [sha(det_path(n, k)) for k in range(5)]}
    return out


def scoring_stamp(cfg):
    """How p2 is computed, saved beside scored rows: rows scored another way are scored again."""
    t, top = int(cfg.get("tta", 1)), int(cfg.get("tta_top", 0))
    if not 1 <= t <= len(FLIPS):
        raise SystemExit("tta must be 1 to %d" % len(FLIPS))
    if top < 0:
        raise SystemExit("tta_top must be 0 or more")
    if t == 1:
        return SCORING
    if top == 0:
        return "%s, %d mirror images" % (SCORING, t)
    return "%s, %d mirror images on each scan's top %d" % (SCORING, t, top)


class Classifier(nn.Module):
    """Small 3D CNN for binary classification: 4 stages of widths [24, 48, 96, 192]."""

    def __init__(self, widths, groups, c_in=1, norm="group"):
        super().__init__()
        if norm == "group":
            make_norm = lambda c: nn.GroupNorm(groups, c)
        elif norm == "batch":
            make_norm = nn.BatchNorm3d
        else:
            raise SystemExit("norm must be group or batch")
        self.c_in = c_in
        self.stages = nn.ModuleList()
        for c_out in widths:
            stage = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, groups=1),
                make_norm(c_out),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, groups=1),
                make_norm(c_out),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2),
            )
            self.stages.append(stage)
            c_in = c_out
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.linear = nn.Linear(2 * widths[-1], 1)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        gap = self.gap(x).view(x.shape[0], -1)
        gmp = self.gmp(x).view(x.shape[0], -1)
        x = torch.cat([gap, gmp], dim=1)
        # in float32: a bfloat16 logit has three significant digits, and fpr-v1's 150,147 V2 points
        # took only 1,312 distinct scores, ties the false-alarm curve cannot rank
        with torch.autocast(x.device.type, enabled=False):
            return self.linear(x.float()).squeeze(-1)


class DetClassifier(nn.Module):
    """The lung finder's encoder (nodule_det.Det's stem and four levels, built the same way and under the same names)
    with Classifier's head: one input channel, widths nodule_det's [32, 64, 128, 256]."""

    def __init__(self):
        super().__init__()
        import nodule_det as nd                             # only an arch det run needs it
        self.widths = list(nd.CFG["widths"])
        self.c_in = 1
        w1, w2, w3, w4 = self.widths
        self.stem = nd.conv(1, w1, 2, 2)
        self.e1 = nd.conv(w1, w1)
        self.d2 = nd.conv(w1, w2, 2, 2)
        self.e2 = nd.conv(w2, w2)
        self.d3 = nd.conv(w2, w3, 2, 2)
        self.e3 = nd.conv(w3, w3)
        self.d4 = nd.conv(w3, w4, 2, 2)
        self.e4 = nd.conv(w4, w4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.linear = nn.Linear(2 * w4, 1)

    def forward(self, x):
        x = self.e4(self.d4(self.e3(self.d3(self.e2(self.d2(self.e1(self.stem(x))))))))
        gap = self.gap(x).view(x.shape[0], -1)
        gmp = self.gmp(x).view(x.shape[0], -1)
        x = torch.cat([gap, gmp], dim=1)
        with torch.autocast(x.device.type, enabled=False):      # in float32, as Classifier's head
            return self.linear(x.float()).squeeze(-1)


def check_arch(cfg):
    """Refuse an unknown `arch`, `init_det` without arch det, and arch det with settings its one-channel net cannot take."""
    arch, n = cfg.get("arch", "classifier"), cfg.get("init_det", 0)
    if arch not in ("classifier", "det"):
        raise SystemExit("arch must be classifier or det, got %r" % (arch,))
    if not isinstance(n, int) or n < 0:
        raise SystemExit("init_det must be 0 or a det version above 0, got %r" % (n,))
    if n and arch != "det":
        raise SystemExit("init_det needs arch det")
    if arch == "det":
        wrong = [k for k, v in (("windows", 1), ("context", 1), ("mirror", 0), ("paste", 0.0), ("norm", "group"),
                                ("init_from", "")) if cfg.get(k, v) != v]
        if wrong:
            raise SystemExit("arch det needs windows 1, context 1, mirror 0, paste 0, norm group and no init_from; "
                             "these differ: %s" % ", ".join(wrong))


def build(cfg):
    if cfg.get("context", 1) not in (1, 2):
        raise SystemExit("context must be 1 or 2")
    check_arch(cfg)
    if cfg.get("arch", "classifier") == "det":
        return DetClassifier().to(DEV)
    return Classifier(cfg["widths"], cfg["groups"], channels_of(cfg), cfg.get("norm", "group")).to(DEV)


def det_path(version, fold):
    return os.path.join(DET_ROOT, "det-v%d" % version, "fold%d.pt" % fold)


def load_det_trunk(net, cfg, fold):
    """Copy det-v<init_det>'s fold `fold` encoder (the eight DET_TRUNK modules, BatchNorm statistics included) into an
    arch det net; the detector's pyramid and heads are not read, and the head stays as built."""
    n = int(cfg["init_det"])
    if fold is None:
        raise SystemExit("init_det needs the fold: each fold starts from det-v%d's net of the same fold" % n)
    path = det_path(n, fold)
    if not os.path.exists(path):
        raise SystemExit("init_det %d: no %s" % (n, path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    have = list(blob.get("cfg", {}).get("widths", []))
    if have != list(net.widths):
        raise SystemExit("det-v%d fold %d has widths %r, this net %r" % (n, fold, have, list(net.widths)))
    trunk = {k: v for k, v in blob["state"].items() if k.split(".")[0] in DET_TRUNK}
    missing, unexpected = net.load_state_dict(trunk, strict=False)
    if unexpected or [k for k in missing if not k.startswith("linear.")]:
        raise SystemExit("det-v%d fold %d does not fit this net: missing %s, unexpected %s" % (n, fold, missing, unexpected))
    print("stage 2 starts from det-v%d fold %d's encoder (%d tensors, sha256 %s)" % (n, fold, len(trunk), sha(path)[:12]),
          flush=True)


def trunk_path(cfg):
    return os.path.join(REPO_ROOT, cfg["init_from"])


def load_trunk(net, cfg):
    """Copy the encoder file named by `init_from` into net's stages; the head stays as built."""
    path = trunk_path(cfg)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    settings = blob.get("settings", {})
    have = {"widths": list(settings.get("widths", [])), **{k: settings.get(k) for k in ("groups", "c_in", "norm")}}
    want = {"widths": list(cfg["widths"]), "groups": cfg["groups"], "c_in": channels_of(cfg), "norm": cfg.get("norm", "group")}
    wrong = ["%s %r (this net %r)" % (k, have[k], want[k]) for k in want if have[k] != want[k]]
    if wrong:
        raise SystemExit("init_from %s was made for other settings: %s" % (cfg["init_from"], "; ".join(wrong)))
    missing, unexpected = net.load_state_dict(blob["state"], strict=False)
    if unexpected or [k for k in missing if not k.startswith("linear.")]:
        raise SystemExit("init_from %s does not fit this net: missing %s, unexpected %s" % (cfg["init_from"], missing, unexpected))
    print("stage 2 starts from %s (%d tensors, sha256 %s)" % (path, len(blob["state"]), sha(path)[:12]), flush=True)


def channels_of(cfg):
    """The net's input channels: context x windows x (1 + mirror)."""
    return cfg.get("context", 1) * cfg.get("windows", 1) * (1 + int(cfg.get("mirror", 0)))


def lr_factor(step, cfg):
    if step < cfg["warmup"]:
        return (step + 1) / cfg["warmup"]
    done = (step - cfg["warmup"]) / max(cfg["steps"] - cfg["warmup"], 1)
    return 0.5 * (1 + math.cos(math.pi * min(done, 1.0)))


def train_net(scans, positives, negatives, cfg, seed, tag, hard_share=0.5, partial=None, fingerprint=None, then=None,
              fold=None):
    """Train a fresh net; returns (net, seconds, peak card GB). `fold` names the det-vN net that `init_det` starts
    from (needed only then; a resumed partial loads its own state instead).

    With `partial` (a path) and `fingerprint`, a partial checkpoint is saved there every RAIDIOLOGY_SAVE_EVERY_S
    seconds (core.ckpt.Every, default 300; never on the last step), and a partial with the same fingerprint is
    continued from its next step.

    `then` = (n, positives, negatives): steps 1 to n draw from the lists above and later steps from these (the
    sampler is rebuilt, seeded from (seed, n) as a resume at step n would be); a run resumed at or past step n
    draws from these from its first step. None (the default) changes nothing."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    net = build(cfg).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            fused=DEV.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg))
    start, prior_secs, prior_peak, sampler_seed = 0, 0.0, 0.0, seed
    ck = ckpt.load_partial(partial, fingerprint) if partial else None
    if ck is not None and 0 < ck.get("step", 0) < cfg["steps"]:
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, prior_secs, prior_peak = ck["step"], ck["seconds"], ck.get("peak_gb", 0.0)
        # The background sampler's exact state cannot be saved: both generators restart from (seed, step).
        sampler_seed = seed * 1_000_003 + start
        torch.manual_seed(sampler_seed)
        gen = torch.Generator(device=DEV).manual_seed(sampler_seed)
        print("fold %s: resumed mid-fold at step %d of %d (partial checkpoint)"
              % (tag.split()[-1], start, cfg["steps"]), flush=True)
    elif cfg.get("init_from"):
        load_trunk(net, cfg)
    elif cfg.get("init_det"):
        load_det_trunk(net, cfg, fold)
    del ck
    if then is not None and start >= then[0]:
        positives, negatives, then = then[1], then[2], None
    sampler = Sampler(scans, positives, negatives, cfg, sampler_seed, cfg["batch"], hard_share)
    feed = Feed(sampler)
    pace = polite.Pace()
    save_due = ckpt.Every()
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    t0 = block = log_t = time.time()
    t0 -= prior_secs                                       # seconds trained add up across a resume
    running, seen = 0.0, 0

    for step in range(start + 1, cfg["steps"] + 1):
        if then is not None and step == then[0] + 1:
            sampler.stop.set()
            sampler = Sampler(scans, then[1], then[2], cfg, seed * 1_000_003 + then[0], cfg["batch"], hard_share)
            feed = Feed(sampler)
            then = None
        imgs, labels = feed.next()
        x = net_input(imgs, net.c_in)
        n = x.shape[0]
        gain = 1 + (torch.rand(n, 1, 1, 1, 1, device=DEV, generator=gen) * 2 - 1) * cfg["brightness"]
        shift = (torch.rand(n, 1, 1, 1, 1, device=DEV, generator=gen) * 2 - 1) * cfg["brightness"]

        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            logits = net(x * gain + shift)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
        opt.step()
        sched.step()

        running = running + loss.detach()
        seen += 1

        if step % SYNC_EVERY == 0 or step == cfg["steps"]:
            running = float(running)
            pace.rest(time.time() - block)
            block = time.time()

        if step % cfg["log_every"] == 0 or step == cfg["steps"]:
            running = float(running)
            now = time.time()
            print("  %s step %d of %d, loss %.4f, %.1f s per 100 steps, using %.0f%% of the card"
                  % (tag, step, cfg["steps"], running / seen, 100 * (now - log_t) / seen, 100 * pace.used()),
                  flush=True)
            log_t, running, seen = now, 0.0, 0
        if partial and step < cfg["steps"] and save_due.due():
            peak_now = max(prior_peak, torch.cuda.max_memory_allocated() / 2 ** 30) if DEV.type == "cuda" else float("nan")
            ckpt.save_partial(partial, {"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                        "step": step, "seconds": time.time() - t0, "peak_gb": peak_now,
                                        "fingerprint": fingerprint})
            save_due.mark()
            if os.environ.get("RAIDIOLOGY_CKPT_SELFTEST_STOP") == str(step):
                sampler.stop.set()
                raise RuntimeError("selftest stop after the partial save at step %d" % step)

    sampler.stop.set()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    peak = max(prior_peak, torch.cuda.max_memory_allocated() / 2 ** 30) if DEV.type == "cuda" else float("nan")
    print("  %s: %d steps in %.0f s, peak card %.2f GB" % (tag, cfg["steps"], seconds, peak), flush=True)
    return net, seconds, peak


# ---- scoring ------------------------------------------------------------------------------------

def scoring_peak():
    """', peak card A GB allocated, R GB reserved' since the last reset (empty on the processor)."""
    if DEV.type != "cuda":
        return ""
    return ", peak card %.2f GB allocated, %.2f GB reserved" % (
        torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30)


# Without inference mode scoring keeps every layer's activations for a backward pass that never comes:
# fpr-v2's first scoring, at 64 crops, ran out of its 2.5 GB (2026-10-03, after scoring_peak was put
# between this decorator and the function it belonged to).
@torch.inference_mode()
def score_points(net, scans, frame, batch=64, tta=1, context=1, waited=None, mirror=0):
    """p2 for each row of `frame` (seriesuid, coordX/Y/Z), same order, the mean over the first `tta` of FLIPS;
    `context` and `mirror` are the net's (never zoomed here); returns (p2 array, crops a second). A list given as `waited`
    gets one number appended: the seconds the loop sat blocked on the reader thread."""
    wait = 0.0
    net.eval()
    t0 = time.time()
    xyz = _xyz(frame)
    uids = frame.seriesuid.tolist()
    out = np.zeros(len(frame), np.float32)

    def read(i):
        return np.stack([cube_of(scans[u], p, context, None, mirror) for u, p in zip(uids[i:i + batch], xyz[i:i + batch])])

    # the next batch is read from disk while the card scores this one
    reader = ThreadPoolExecutor(1)
    ahead = reader.submit(read, 0)
    for i in range(0, len(frame), batch):
        tw = time.time()
        crops = ahead.result()
        wait += time.time() - tw
        if i + batch < len(frame):
            ahead = reader.submit(read, i + batch)
        x = net_input(torch.from_numpy(crops).to(DEV), net.c_in)
        p = 0.0
        for dims in FLIPS[:tta]:
            with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                logit = net(torch.flip(x, dims) if dims else x)
            p = p + torch.sigmoid(logit.float())
        out[i:i + batch] = (p / tta).cpu().numpy()
    reader.shutdown()
    if waited is not None:
        waited.append(wait)
    return out, len(frame) / max(time.time() - t0, 1e-9)


def top_rows(uids, score, k):
    """Row indices (sorted) of each scan's `k` highest `score` rows; equal scores keep the earlier row."""
    score = np.asarray(score, np.float64)
    taken, keep = {}, []
    for i in np.argsort(-score, kind="stable"):
        u = uids[i]
        if taken.get(u, 0) < k:
            taken[u] = taken.get(u, 0) + 1
            keep.append(i)
    return np.sort(np.array(keep, np.int64))


def score_frame(net, scans, frame, batch, tta, tta_top, context, mirror=0):
    """score_points for `rescore`, honouring `tta_top`: with tta above 1 and tta_top above 0, one single reading
    of every row, then the `tta`-way reading of each scan's top `tta_top` rows by it replaces theirs.
    Returns (p2, crops a second, info: n_top, single_s, multi_s, wait_s, total_s)."""
    t0, waited = time.time(), []
    if tta <= 1 or tta_top <= 0:
        p2, _ = score_points(net, scans, frame, batch, tta, context, waited, mirror)
        s = time.time() - t0
        return p2, len(frame) / max(s, 1e-9), {"n_top": 0, "single_s": s, "multi_s": 0.0, "wait_s": waited[0], "total_s": s}
    p2, _ = score_points(net, scans, frame, batch, 1, context, waited, mirror)
    t1 = time.time()
    idx = top_rows(frame.seriesuid.tolist(), p2, tta_top)
    p2 = p2.copy()
    if len(idx):
        p2[idx], _ = score_points(net, scans, frame.iloc[idx].reset_index(drop=True), batch, tta, context, waited, mirror)
    t2 = time.time()
    return p2, len(frame) / max(t2 - t0, 1e-9), {"n_top": len(idx), "single_s": t1 - t0, "multi_s": t2 - t1,
                                                 "wait_s": sum(waited), "total_s": t2 - t0}


def timing_note(info):
    """The reader-wait and (when tta_top is used) per-pass seconds of score_frame, for a fold's printed line."""
    note = "reading waited %.0f s of %.0f s" % (info["wait_s"], info["total_s"])
    if info["n_top"]:
        note += "; %d places read the long way: single pass %.0f s, long pass %.0f s" % (
            info["n_top"], info["single_s"], info["multi_s"])
    return note


# ---- a run --------------------------------------------------------------------------------------

def _saved_fingerprint(path):
    """The fingerprint a saved fold checkpoint carries, read on the processor (None when missing or unreadable,
    or when RAIDIOLOGY_NO_RESUME is 1, as load_resumable treats them)."""
    if os.environ.get("RAIDIOLOGY_NO_RESUME") == "1" or not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("fingerprint")
    except Exception:  # noqa: BLE001
        return None


def rescore(a):
    """Per fold: train on the other folds (or resume), score the fold's stage-1 candidates and V2 points."""
    cfg = with_settings(CFG, a.set)
    if a.steps:
        cfg["steps"] = a.steps
    check_paste(cfg)
    check_arch(cfg)
    if cfg["init_det"] and a.ten:
        raise SystemExit("init_det needs the five folds: det-v%d's nets are on nodule.py's five, not --ten's ten"
                         % cfg["init_det"])
    stamp, tta, tta_top = scoring_stamp(cfg), int(cfg["tta"]), int(cfg["tta_top"])
    card.require_card(DEV, torch)

    folds_of, missing = scan_folds(a.ten)
    if missing:
        print("WARNING: %d of LUNA16's scans have no prepared box: %s"
              % (len(missing), ", ".join(m[-12:] for m in missing[:10])), flush=True)
    ann, exc = reference()
    # training reads LUNA16's answer key plus NLST501's; scoring and every CPM keep `ann`, `exc` (LUNA16 only)
    train_ann, train_exc, n_uids, n_ann, n_exc = ann, exc, [], None, None
    if bool(a.extra_nlst) != bool(a.extra_nlst_hard):
        raise SystemExit("--extra-nlst and --extra-nlst-hard each need the other")

    stage1_file = a.stage1_file or os.path.join(paths.WORK, "eval", "lung", "nodule-v%d" % a.stage1, "candidates_oof.csv")
    stage1 = pd.read_csv(stage1_file)
    stage1 = stage1[stage1.seriesuid.isin(folds_of)].copy()
    stage1["fold"] = stage1.seriesuid.map(folds_of)
    present = sorted(int(f) for f in stage1.fold.unique())
    run_folds = a.folds if a.folds else present
    cand_file = os.path.join(LUNA_DIR, "candidates_V2.csv")
    hard_file = None
    if a.hard or a.hard_file:
        hard_file = a.hard_file or os.path.join(paths.WORK, "eval", "lung", "nodule-v%d" % a.hard, "candidates_oof.csv")

    out = out_root(a.version)
    os.makedirs(out, exist_ok=True)
    fingerprint = {
        "version": a.version, "ten": bool(a.ten), "cfg": fingerprint_cfg(cfg), "hard_version": a.hard, "cand_v2_sha256": sha(cand_file),
        **({"hard_file_sha256": sha(hard_file)} if a.hard_file else {}),
        "annotations_sha256": sha(os.path.join(LUNA_DIR, "annotations.csv")),
        "excluded_sha256": sha(os.path.join(LUNA_DIR, "evaluationScript", "annotations", "annotations_excluded.csv")),
    }
    if a.extra_nlst:
        fingerprint["extra_nlst"] = {"list_sha256": sha(a.extra_nlst), "hard_sha256": sha(a.extra_nlst_hard)}
    fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True, default=str))
    # Training never sees stage 1's candidates (only --hard's), so they are not in the fingerprint: a new
    # stage 1 reuses the trained net and its V2 scores, and only its own candidates are scored again.
    s1_stamp = {"version": a.stage1 if a.stage1_file is None else a.stage1_file, "sha256": sha(stage1_file)}
    what = "stage 1 v%d" % a.stage1 if a.stage1_file is None else os.path.relpath(stage1_file, paths.WORK)
    name = "nodule fpr v%d on %s" % (a.version, what)
    # A --tag run writes its own files (candidates_oof_<tag>.csv, meta_<tag>.json) and leaves the
    # checkpoint as it was, so the untagged list's saved scores stay the ones a plain rescore resumes.
    suffix = "_" + a.tag if a.tag else ""
    s1_parts, v2_parts, train_secs, peaks, rates, info = [], [], {}, [], [], {}
    # the lease is sized once it is known whether any fold trains (a peek on the processor)
    scoring_only = all(_saved_fingerprint(os.path.join(out, "fold%d.pt" % k)) == {**fingerprint, "fold": k}
                       for k in run_folds)
    # 2026-10-03: leased before any scan, candidate list or CUDA is loaded, so a waiting rescore holds
    # next to nothing (it held 2.4 GB waiting, 6.9 GB with its data loaded; the machine ran out of
    # committed memory at 23:21 and a training job crashed)
    limit_fpr_card(SCORING_GB if scoring_only else None)
    batch = SCORING_BATCH if scoring_only else 64
    if scoring_only:
        print("every fold resumes a saved net: scoring only, leasing %.1f GB, %d crops a batch"
              % (min(SCORING_GB, card_gb()), batch), flush=True)

    if a.extra_nlst:
        import nlst501
        n_uids, n_ann, n_exc = nlst501.reference()
        assert not set(n_uids) & set(folds_of), "an NLST scan is a LUNA16 scan"
        train_ann = pd.concat([ann, n_ann], ignore_index=True)
        train_exc = pd.concat([exc, n_exc], ignore_index=True)
    scans = {u: Scan(u, train_ann, train_exc, cfg) for u in [*folds_of, *n_uids]}
    nlst_set = set(n_uids)
    cand_v2 = candidates_frame(cand_file, folds_of)
    hard_cand = pd.read_csv(hard_file) if hard_file else None
    cand_train = cand_v2
    if a.extra_nlst:
        n_all, n_hard = pd.read_csv(a.extra_nlst), pd.read_csv(a.extra_nlst_hard)
        n_all, n_hard = n_all[n_all.seriesuid.isin(nlst_set)], n_hard[n_hard.seriesuid.isin(nlst_set)]
        key = lambda d: list(zip(d.seriesuid, *(d[c].round(3) for c in ("coordX", "coordY", "coordZ"))))  # noqa: E731
        in_hard = set(key(n_hard))
        rest = n_all[[k not in in_hard for k in key(n_all)]]
        cols = ["seriesuid", "coordX", "coordY", "coordZ"]
        cand_train = pd.concat([cand_v2[cols + ["class"]], rest[cols].assign(**{"class": 0})], ignore_index=True)
        hard_cand = n_hard[cols] if hard_cand is None else pd.concat([hard_cand[cols], n_hard[cols]], ignore_index=True)

    with ledger.run("nodule", name) as led:
        led.settings_from({"cfg": cfg, "ten": bool(a.ten), "folds": run_folds, "stage1": a.stage1,
                           "stage1_file": a.stage1_file, "tag": a.tag, "hard": a.hard,
                           "hard_file": a.hard_file, "skip_v2": a.skip_v2,
                           **({"extra_nlst": a.extra_nlst, "extra_nlst_hard": a.extra_nlst_hard} if a.extra_nlst else {})})
        for k in run_folds:
            train_uids = [u for u, f in folds_of.items() if f != k] + list(n_uids)
            held = {u: scans[u] for u, f in folds_of.items() if f == k}
            path = os.path.join(out, "fold%d.pt" % k)
            partial_path = os.path.join(out, "fold%d.partial.pt" % k)
            fp = {**fingerprint, "fold": k}
            ck = load_resumable(path, fp)
            if ck is not None and "s1_rows" in ck and ck.get("stage1") == s1_stamp and ck.get("scoring") == stamp:
                train_secs[k], peak = ck["seconds"], ck.get("peak_gb", float("nan"))
                s1_k, v2_k = pd.DataFrame(ck["s1_rows"]), pd.DataFrame(ck["v2_rows"])
                rates.append(ck.get("crops_per_second", float("nan")))
                print("fold %d: resumed with its scored rows (no training, no reading)" % k, flush=True)
            else:
                net = build(cfg)
                if ck is not None:
                    net.load_state_dict(ck["state"])
                    train_secs[k], peak = ck["seconds"], ck.get("peak_gb", float("nan"))
                    print("fold %d: resumed from %s (no training)" % (k, path), flush=True)
                else:
                    pos, neg, hard = build_training_data(train_ann, train_exc, cand_train, train_uids, cfg, hard_cand)
                    if a.extra_nlst:
                        n = [sum(r[0] in nlst_set for r in lst) for lst in (pos, neg, hard)]
                        print("fold %d: %d positives (LUNA16 %d, NLST %d), %d negatives (LUNA16 %d, NLST %d), "
                              "%d hard negatives (LUNA16 %d, NLST %d)"
                              % (k, len(pos), len(pos) - n[0], n[0], len(neg), len(neg) - n[1], n[1],
                                 len(hard), len(hard) - n[2], n[2]), flush=True)
                    else:
                        print("fold %d: %d positives, %d negatives, %d hard negatives"
                              % (k, len(pos), len(neg), len(hard)), flush=True)
                    if not pos or not neg:
                        raise SystemExit("fold %d has no training data" % k)
                    net, train_secs[k], peak = train_net({u: scans[u] for u in train_uids}, pos, (neg, hard), cfg,
                                                         1000 * int(cfg["seed"]) + k + 1, "fold %d" % k,
                                                         cfg["hard_share"], partial_path, fp, fold=k)
                    torch.save({"state": net.state_dict(), "cfg": cfg, "fingerprint": fp,
                                "seconds": train_secs[k], "peak_gb": peak}, path)
                    ckpt.clear_partial(partial_path)
                s1_k = stage1[stage1.fold == k][["seriesuid", "coordX", "coordY", "coordZ", "probability"]]
                s1_k = s1_k.rename(columns={"probability": "p1"}).reset_index(drop=True)
                if DEV.type == "cuda":
                    # training's freed blocks stay reserved and split; scoring's 64-crop layers need whole ones
                    # (cancer model v2 stopped scoring its fold 1 this way, 2026-10-03: 973 MiB reserved and
                    # unused, 648 MiB asked for, over its 2.5 GB)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()   # what scoring alone holds, to size a scoring-only lease
                s1_k["p2"], r1, i1 = score_frame(net, held, s1_k, batch, tta, tta_top, cfg["context"], cfg["mirror"])
                s1_k["fold"] = k
                if a.skip_v2:
                    v2_k, rate = None, r1
                    print("fold %d: scored %d stage-1 points, %.0f crops a second (V2's points not scored; %s)%s"
                          % (k, len(s1_k), rate, timing_note(i1), scoring_peak()), flush=True)
                elif ck is not None and "v2_rows" in ck and ck.get("scoring") == stamp:
                    v2_k, rate = pd.DataFrame(ck["v2_rows"]), r1
                    print("fold %d: scored %d stage-1 points, %.0f crops a second (V2's scores resumed; %s)%s"
                          % (k, len(s1_k), rate, timing_note(i1), scoring_peak()), flush=True)
                else:
                    v2_k = cand_v2[cand_v2.fold == k][["seriesuid", "coordX", "coordY", "coordZ", "class"]]
                    v2_k = v2_k.reset_index(drop=True)
                    v2_k["p2"], r2, i2 = score_frame(net, held, v2_k, batch, tta, tta_top, cfg["context"], cfg["mirror"])
                    v2_k["fold"] = k
                    rate = (len(s1_k) + len(v2_k)) / (len(s1_k) / r1 + len(v2_k) / r2)
                    both = {key: i1[key] + i2[key] for key in i1}
                    print("fold %d: scored %d stage-1 and %d V2 points, %.0f crops a second (%s)%s"
                          % (k, len(s1_k), len(v2_k), rate, timing_note(both), scoring_peak()), flush=True)
                rates.append(rate)
                if cfg["mirror"]:
                    seen = [u for u in [*train_uids, *held] if u in _MID]
                    print("fold %d: mirror: midlines of %d scans, %d with an empty lung mask (centre column used)"
                          % (k, len(seen), sum(u in _MID_EMPTY for u in seen)), flush=True)
                if not a.tag and not a.skip_v2:
                    torch.save({"state": net.state_dict(), "cfg": cfg, "fingerprint": fp, "seconds": train_secs[k],
                                "peak_gb": peak, "crops_per_second": rate, "stage1": s1_stamp, "scoring": stamp,
                                "s1_rows": s1_k.to_dict("list"), "v2_rows": v2_k.to_dict("list")}, path)
                del net
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
            peaks.append(peak)
            s1_parts.append(s1_k)
            if v2_k is not None:
                v2_parts.append(v2_k)
            info[k] = {"train_scans": len(train_uids), "held_scans": len(held), "train_seconds": train_secs[k]}

        s1_all = pd.concat(s1_parts, ignore_index=True)
        s1_all.to_csv(os.path.join(out, "candidates_oof%s.csv" % suffix), index=False)
        v2_all = pd.concat(v2_parts, ignore_index=True) if v2_parts else None
        if v2_all is not None:
            v2_all.to_csv(os.path.join(out, "v2track_oof%s.csv" % suffix), index=False)

        uids = [u for u, f in folds_of.items() if f in run_folds]
        cols = ["seriesuid", "coordX", "coordY", "coordZ"]
        c1 = s1_all[cols].assign(probability=s1_all.p1)
        c2 = s1_all[cols].assign(probability=s1_all.p2)
        c12 = s1_all[cols].assign(probability=s1_all.p1 * s1_all.p2)
        lists = [("stage1", c1), ("stage2", c2), ("combined", c12)]
        if v2_all is not None:                                # every V2 point, nodules included
            lists.append(("v2list", v2_all[cols].assign(probability=v2_all.p2)))
        results = {key: froc.score(c, ann, exc, uids, boot=1000) for key, c in lists}
        labels = {"stage1": "stage 1 alone (p1)", "stage2": "stage 2 alone (p2)",
                  "combined": "p1 * p2", "v2list": "candidate list (V2 points, p2)"}
        meta = {"name": name, "version": a.version, "stage1": a.stage1, "stage1_file": stage1_file, "tag": a.tag,
                "hard": a.hard, "ten": bool(a.ten),
                "folds": run_folds, "cfg": cfg, "per_fold": info,
                "train_seconds_per_fold": {str(k): v for k, v in train_secs.items()},
                "crops_per_second_scoring": float(np.nanmean(rates)) if rates else float("nan"),
                "peak_memory_gb": float(np.nanmax(peaks)) if peaks else float("nan"),
                "n_scans": len(uids), "commit": ledger._commit(), "torch": torch.__version__,
                "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        for key, res in results.items():
            cpm, lo, hi = res["cpm"], res["boot"]["cpm"][1], res["boot"]["cpm"][2]
            print("CPM %s: %.4f (95%% interval %.4f to %.4f)" % (labels[key], cpm, lo, hi))
            led.number("cpm_" + key, cpm).number("cpm_%s_low" % key, lo).number("cpm_%s_high" % key, hi)
            meta["cpm_" + key] = cpm
            meta["cpm_%s_interval" % key] = [lo, hi]
        led.number("train_seconds_per_fold", float(np.mean(list(train_secs.values()))), "s")
        led.number("crops_per_second", meta["crops_per_second_scoring"], "per s")
        led.number("peak_memory_gb", meta["peak_memory_gb"])
        led.number("scans", len(uids), "count")
        with open(os.path.join(out, "meta%s.json" % suffix), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    return 0


def smoke(a):
    """6 scans on the processor (4 train, 2 scored), 20 steps at batch 8, 2 threads."""
    global DEV
    DEV = torch.device("cpu")
    cfg = with_settings(dict(CFG, steps=20, batch=8, warmup=2, log_every=5, threads=2, queue=2), a.set)
    check_paste(cfg)
    ann, exc = reference()
    have = []
    for uid in sorted(lp.luna16_subsets()):
        npy, js = lp.paths_for(lp.out_dir(), uid)
        if os.path.exists(npy) and os.path.exists(js) and (ann.seriesuid == uid).any():
            have.append(uid)
        if len(have) == 6:
            break
    assert len(have) == 6, "fewer than 6 prepared LUNA16 scans with nodules"
    scans = {u: Scan(u, ann, exc, cfg) for u in have}
    train_uids, held_uids = have[:4], have[4:]
    cand_v2 = candidates_frame(os.path.join(LUNA_DIR, "candidates_V2.csv"), {u: 0 for u in have})
    stage1_file = os.path.join(paths.WORK, "eval", "lung", "nodule-v1", "candidates_oof.csv")
    s1_all = pd.read_csv(stage1_file) if os.path.exists(stage1_file) else None
    pos, neg, hard_n = build_training_data(ann, exc, cand_v2, train_uids, cfg, s1_all)
    print("smoke: %d positives, %d negatives, %d hard negatives" % (len(pos), len(neg), len(hard_n)))
    train_scans = {u: scans[u] for u in train_uids}
    smp = Sampler(train_scans, pos, (neg, hard_n), cfg, 0, cfg["batch"], 0.5)
    bx, by = smp.get()
    smp.stop.set()
    print("crop batch shape %s (z, y, x = %d^3), positives %d of %d" % (tuple(bx.shape), CUBE, int(by.sum()), len(by)))
    net, seconds, peak = train_net(train_scans, pos, (neg, hard_n), cfg, 0, "smoke", 0.5)
    held = {u: scans[u] for u in held_uids}
    v2 = cand_v2[cand_v2.seriesuid.isin(held_uids)].head(300).reset_index(drop=True)
    p2, rate = score_points(net, held, v2, context=cfg["context"], mirror=cfg["mirror"])
    print("smoke: scored %d V2 points, %.0f crops a second; p2 range %.3f to %.3f" % (len(v2), rate, p2.min(), p2.max()))
    if s1_all is not None:
        s1 = s1_all[s1_all.seriesuid.isin(held_uids)].reset_index(drop=True)
        if len(s1):
            q, _ = score_points(net, held, s1, context=cfg["context"], mirror=cfg["mirror"])
            print("smoke: scored %d stage-1 candidates, p2 range %.3f to %.3f" % (len(s1), q.min(), q.max()))
    print("speed: %.3f seconds per step at batch %d" % (seconds / cfg["steps"], cfg["batch"]))
    return 0


def _selftest_resume():
    """Mid-fold resume on the processor with a tiny net and a stand-in sampler: a run stopped on purpose right after
    its save at step 3 of 6 (RAIDIOLOGY_CKPT_SELFTEST_STOP) is continued from step 3, with the saved optimizer
    state, running only steps 4 to 6; a corrupted partial is renamed .bad and training starts fresh."""
    global DEV, Sampler
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
            imgs = torch.from_numpy(self.rng.integers(0, 255, (self.batch, 1, CUBE, CUBE, CUBE), dtype=np.uint8))
            return imgs, torch.from_numpy((self.rng.random(self.batch) < 0.5).astype(np.float32))

    opts = []
    real_adamw = torch.optim.AdamW

    def adamw(*args, **kw):
        opts.append(real_adamw(*args, **kw))
        return opts[-1]

    def run(cfg, tag, partial, fp):
        Stand.drawn[0] = 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = train_net({}, [], ([], []), cfg, 7, tag, 0.5, partial, fp)
        return result, out.getvalue(), Stand.drawn[0] - 1   # the feed reads one batch ahead

    cfg = dict(CFG, widths=[4, 4], groups=2, steps=6, batch=2, warmup=2, log_every=100, brightness=0.05)
    env = {"RAIDIOLOGY_SAVE_EVERY_S": "0", "RAIDIOLOGY_SHARE": "1"}
    old = {k: os.environ.get(k) for k in (*env, "RAIDIOLOGY_CKPT_SELFTEST_STOP", "RAIDIOLOGY_NO_RESUME")}
    os.environ.update(env)
    os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
    real_dev, real_sampler, DEV = DEV, Sampler, torch.device("cpu")
    Sampler, torch.optim.AdamW = Stand, adamw
    try:
        with tempfile.TemporaryDirectory() as tmp:
            partial, fp = os.path.join(tmp, "fold1.partial.pt"), {"fold": 1, "version": 0}
            # a run with no partial path saves nothing
            _, text, n = run(cfg, "fold 1", None, None)
            assert n == 6 and not os.listdir(tmp) and "resumed" not in text, (n, os.listdir(tmp), text)
            # stopped on purpose right after the save at step 3
            os.environ["RAIDIOLOGY_CKPT_SELFTEST_STOP"] = "3"
            try:
                run(cfg, "fold 1", partial, fp)
                raise AssertionError("the selftest stop did not stop the run")
            except RuntimeError as e:
                assert "step 3" in str(e), e
            assert os.path.exists(partial) and not os.path.exists(partial + ".tmp")
            saved = ckpt.load_partial(partial, fp)
            assert saved["step"] == 3 and saved["opt"]["state"] and saved["seconds"] >= 0, saved.keys()
            assert ckpt.load_partial(partial, {"fold": 2, "version": 0}) is None        # another fingerprint: not used
            os.environ.pop("RAIDIOLOGY_NO_RESUME", None)
            os.environ["RAIDIOLOGY_NO_RESUME"] = "1"
            assert ckpt.load_partial(partial, fp) is None
            os.environ.pop("RAIDIOLOGY_NO_RESUME")
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
            _, text, n = run(cfg, "fold 1", partial, fp)
            Stand.check.clear()
            assert "fold 1: resumed mid-fold at step 3 of 6 (partial checkpoint)" in text, text
            assert seen == [True] and n == 3, (seen, n)
            assert Stand.made[-1] == 7 * 1_000_003 + 3                                    # re-seeded from (seed, step)
            after = ckpt.load_partial(partial, fp)
            assert after["step"] == 5 and after["seconds"] >= saved["seconds"]            # no save on the last step
            ckpt.clear_partial(partial)                                                   # what a finished fold does
            assert not os.listdir(tmp), os.listdir(tmp)
            # a corrupted partial is renamed .bad and training starts fresh
            with open(partial, "wb") as f:
                f.write(b"not a checkpoint")
            _, text, n = run(cfg, "fold 1", partial, fp)
            assert os.path.exists(partial + ".bad") and "resumed" not in text and "unreadable" in text, text
            assert n == 6 and Stand.made[-1] == 7, (n, Stand.made[-1])
    finally:
        Sampler, DEV, torch.optim.AdamW = real_sampler, real_dev, real_adamw
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("resume selftest ok")


def selftest(a=None):
    """Processor only, no data and no card: tta_top's picker, its two-pass scoring, the fingerprint and the stamp,
    with a stand-in for score_points whose value depends on tta."""
    global score_points, DEV, DET_ROOT
    # (a) the picker: top 2 of each of 3 scans; a tie keeps the earlier row
    uids = ["a", "b", "a", "c", "a", "b", "a", "c", "b"]
    sc = [0.1, 0.5, 0.9, 0.3, 0.9, 0.5, 0.2, 0.3, 0.7]
    assert top_rows(uids, sc, 2).tolist() == [1, 2, 3, 4, 7, 8], top_rows(uids, sc, 2)
    assert top_rows(uids, sc, 1).tolist() == [2, 3, 8]
    assert top_rows(uids, sc, 9).tolist() == list(range(9))

    frame = pd.DataFrame({"seriesuid": uids, "coordX": sc, "coordY": 0.0, "coordZ": 0.0})
    calls = []

    def fake(net, scans, fr, batch=64, tta=1, context=1, waited=None, mirror=0):
        calls.append((len(fr), tta))
        if waited is not None:
            waited.append(0.0)
        return fr.coordX.to_numpy(np.float32) + 10.0 * (tta - 1), 1.0

    real, score_points = score_points, fake
    try:
        # (b) tta=8, tta_top=2: the top 2 of each scan carry the eight-way value, the rest the single one
        p, _, info = score_frame(None, None, frame, 4, 8, 2, 1)
        top = {1, 2, 3, 4, 7, 8}
        want = [sc[i] + (70.0 if i in top else 0.0) for i in range(9)]
        assert np.allclose(p, want), (p, want)
        assert calls == [(9, 1), (6, 8)] and info["n_top"] == 6, (calls, info)
        # (c) tta_top=0 is the full eight-way reading
        calls.clear()
        p0, _, info0 = score_frame(None, None, frame, 4, 8, 0, 1)
        assert np.allclose(p0, [s + 70.0 for s in sc]) and calls == [(9, 8)] and info0["n_top"] == 0
        # tta=1 ignores tta_top
        p1, _, _ = score_frame(None, None, frame, 4, 1, 32, 1)
        assert np.allclose(p1, sc)
    finally:
        score_points = real
    # (d) the fingerprint leaves tta_top out, and a cfg at the defaults fingerprints as it did before tta_top existed
    base = {k: v for k, v in CFG.items() if k != "tta_top"}
    assert fingerprint_cfg({**CFG, "tta_top": 32}) == fingerprint_cfg({**CFG, "tta_top": 0}) == fingerprint_cfg(base)
    assert "tta_top" not in fingerprint_cfg({**CFG, "tta_top": 32})
    assert fingerprint_cfg({**CFG, "tta": 8, "tta_top": 32}) == fingerprint_cfg(base)
    # (e) the stamp names it; tta=1 is unchanged
    s0, s32 = scoring_stamp({**CFG, "tta": 8, "tta_top": 0}), scoring_stamp({**CFG, "tta": 8, "tta_top": 32})
    assert s0 != s32 and "top 32" in s32, (s0, s32)
    assert scoring_stamp({**CFG, "tta": 1, "tta_top": 32}) == SCORING
    try:
        scoring_stamp({**CFG, "tta": 8, "tta_top": -1})
        raise AssertionError("a negative tta_top passed")
    except SystemExit:
        pass
    # (f) mirror: the midline of a synthetic mask, the mirrored centre, the twin, the channel count, the fingerprint
    import tempfile
    from types import SimpleNamespace
    m = np.zeros((6, 8, 40), bool)
    m[2:4, 3:5, 5:12] = True
    m[1, 1, 30] = True
    assert midline_of_mask(m) == (5 + 30) / 2.0 and midline_of_mask(np.zeros((2, 3, 4), bool)) is None
    shape = (30, 40, 64)
    rng = np.random.default_rng(0)
    half = rng.integers(0, 255, (30, 40, 32), dtype=np.uint8)
    sym = np.concatenate([half, half[..., ::-1]], axis=2)                  # symmetric about x = 31.5
    asym = sym.copy()
    asym[15, 20, 20] = 255                                                  # a dot with no partner at x = 43
    asym[15, 20, 43] = 0
    meta = {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [1.0, 1.0, 1.0], "shape": list(shape)}
    with tempfile.TemporaryDirectory() as tmp:
        scans = {}
        for uid, vol in (("sym", sym), ("asym", asym)):
            path = os.path.join(tmp, uid + ".npy")
            np.save(path, vol)
            scans[uid] = SimpleNamespace(uid=uid, meta=meta, path=path, shape=shape)
            _MID[uid] = 31.5
        c = np.array([20.0, 20.0, 15.0])                                    # world x, y, z = voxel x, y, z
        assert np.allclose(mirror_centre(scans["sym"], c), [44.0, 20.0, 15.0])   # 2 * 31.5 - 20 + 1
        assert np.allclose(mirror_centre(scans["sym"], mirror_centre(scans["sym"], c) - [2, 0, 0]), [22.0, 20.0, 15.0])
        for ctx in (1, 2):
            for s in (None, 1.0):
                both = cube_of(scans["sym"], c, ctx, s, 1)
                assert both.shape == (2 * ctx, CUBE, CUBE, CUBE)
                assert np.array_equal(both[:ctx], both[ctx:]), "a symmetric volume's twin differs from its cube"
                assert np.array_equal(both[:ctx], cube_of(scans["sym"], c, ctx, s, 0))
        both = cube_of(scans["asym"], c, 1, None, 1)
        assert both[0].max() == 255 and (both[0] != both[1]).any()
        # the dot at voxel (15, 20, 20) sits 0 voxels from the cube centre in z, y and 20 - 20 = 0 in x
        assert both[0][CUBE // 2, CUBE // 2, CUBE // 2 + 0] == 255 and both[1][CUBE // 2, CUBE // 2, CUBE // 2] != 255
    for ctx in (1, 2):
        for win in (1, 3):
            for mir in (0, 1):
                assert channels_of({"context": ctx, "windows": win, "mirror": mir}) == ctx * win * (1 + mir)
    assert net_input(torch.zeros(1, 4, 2, 2, 2, dtype=torch.uint8), 12).shape == (1, 12, 2, 2, 2)
    assert fingerprint_cfg({**CFG, "mirror": 0}) == fingerprint_cfg({k: v for k, v in CFG.items() if k != "mirror"})
    assert "mirror" not in fingerprint_cfg(CFG) and fingerprint_cfg({**CFG, "mirror": 1})["mirror"] == 1
    # (g) init_from: the stages come from the encoder file, the head stays as built, a mismatch is refused
    small = {**CFG, "widths": [4, 8], "groups": 2}
    with tempfile.TemporaryDirectory() as tmp:
        trunk = os.path.join(tmp, "encoder.pt")
        torch.manual_seed(0)
        stages = {k: v for k, v in build(small).state_dict().items() if k.startswith("stages.")}
        torch.save({"state": stages, "settings": {"widths": [4, 8], "groups": 2, "c_in": 1, "norm": "group"}}, trunk)
        cfg_i = {**small, "init_from": trunk}
        torch.manual_seed(1)
        net = build(cfg_i)
        head = net.linear.weight.detach().clone()
        load_trunk(net, cfg_i)
        got = net.state_dict()
        assert all(torch.equal(got[k], v) for k, v in stages.items()), "a stage tensor was not copied from the encoder"
        assert torch.equal(net.linear.weight.detach(), head), "the head was overwritten"
        assert fingerprint_cfg(cfg_i)["init_from"]["sha256"] == sha(trunk)
        no_key = {k: v for k, v in CFG.items() if k != "init_from"}
        assert "init_from" not in fingerprint_cfg(CFG) and fingerprint_cfg({**CFG, "init_from": ""}) == fingerprint_cfg(no_key)
        wide = {**cfg_i, "widths": [4, 16]}
        try:
            load_trunk(build(wide), wide)
            raise AssertionError("a trunk of other widths loaded")
        except SystemExit:
            pass
    # (h) arch det and init_det: the defaults fingerprint as before; the det net reads a crop to one float32 logit;
    # its encoder comes from a det-vN fold file (a stand-in in a temp folder), BatchNorm statistics included, and the
    # head stays as built; the refusals
    import nodule_det as nd
    no_det = {k: v for k, v in CFG.items() if k not in ("arch", "init_det")}
    assert "arch" not in fingerprint_cfg(CFG) and "init_det" not in fingerprint_cfg(CFG)
    assert fingerprint_cfg(CFG) == fingerprint_cfg(no_det)
    real_dev, real_root, DEV = DEV, DET_ROOT, torch.device("cpu")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            DET_ROOT = tmp
            cfg9 = {**CFG, "arch": "det", "init_det": 9}
            torch.manual_seed(0)
            net = build(cfg9)
            with torch.no_grad():
                logit = net.eval()(torch.rand(2, 1, CUBE, CUBE, CUBE))
            assert logit.shape == (2,) and logit.dtype == torch.float32, (logit.shape, logit.dtype)
            torch.manual_seed(1)
            det = nd.Det(nd.CFG["widths"], nd.CFG["fpn"], nd.CFG["heat_bias"])
            with torch.no_grad():                           # BatchNorm values unlike a fresh net's
                for mod in det.modules():
                    if isinstance(mod, nn.BatchNorm3d):
                        for t in (mod.weight, mod.bias, mod.running_mean):
                            t.uniform_(-1, 1)
                        mod.running_var.uniform_(0.5, 2)
                        mod.num_batches_tracked.fill_(7)
            os.makedirs(os.path.join(tmp, "det-v9"))
            torch.save({"state": det.state_dict(), "cfg": {"widths": list(nd.CFG["widths"])}}, det_path(9, 0))
            for k in range(1, 5):                           # small stand-ins: only their sha is read
                torch.save({"state": {}, "cfg": {"widths": list(nd.CFG["widths"])}, "fold": k}, det_path(9, k))
            head = [p.detach().clone() for p in net.linear.parameters()]
            load_det_trunk(net, cfg9, 0)
            got = net.state_dict()
            want = {k: v for k, v in det.state_dict().items() if k.split(".")[0] in DET_TRUNK}
            assert want and all(torch.equal(got[k], v) for k, v in want.items()), "a trunk tensor was not copied"
            assert set(got) - set(want) == {"linear.weight", "linear.bias"}, set(got) - set(want)
            assert all(torch.equal(p.detach(), h) for p, h in zip(net.linear.parameters(), head)), "the head was overwritten"
            assert fingerprint_cfg(cfg9)["init_det"] == {"version": 9, "sha256": [sha(det_path(9, k)) for k in range(5)]}
            os.makedirs(os.path.join(tmp, "det-v8"))
            torch.save({"state": {}, "cfg": {"widths": [16, 32, 64, 128]}}, det_path(8, 0))
            refusals = (("arch det with windows 3", lambda: build({**CFG, "arch": "det", "windows": 3})),
                        ("init_det with arch classifier", lambda: build({**CFG, "init_det": 1})),
                        ("an unknown arch", lambda: build({**CFG, "arch": "resnet"})),
                        ("a det file of other widths", lambda: load_det_trunk(net, {**cfg9, "init_det": 8}, 0)),
                        ("init_det without a fold", lambda: load_det_trunk(net, cfg9, None)),
                        ("a det version short of fold files", lambda: fingerprint_cfg({**cfg9, "init_det": 8})))
            refused = []
            for what, refuse in refusals:
                try:
                    refuse()
                except SystemExit as e:
                    refused.append((what, str(e)))
            assert [w for w, _ in refused] == [w for w, _ in refusals], ("not refused", refused)
    finally:
        DEV, DET_ROOT = real_dev, real_root
    _selftest_resume()
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rescore")
    r.add_argument("--version", type=int, required=True)
    s1 = r.add_mutually_exclusive_group(required=True)
    s1.add_argument("--stage1", type=int, help="stage 1 version: its nodule-v<N>/candidates_oof.csv")
    s1.add_argument("--stage1-file", help="any candidate list (seriesuid, coordX, coordY, coordZ, probability)")
    r.add_argument("--tag", help="write candidates_oof_<tag>.csv and meta_<tag>.json; the checkpoint is left as it was")
    hd = r.add_mutually_exclusive_group()
    hd.add_argument("--hard", type=int, help="hard negatives: stage 1 version N's candidates (training folds only)")
    hd.add_argument("--hard-file", help="hard negatives: any candidate list, e.g. nodule_blobs.py top's")
    r.add_argument("--extra-nlst", help="train also on LIDC-annot-NLST501's scans (never held out or scored): "
                                        "their blob list (nodule_blobs.py build --nlst); needs --extra-nlst-hard")
    r.add_argument("--extra-nlst-hard", help="NLST501's hard negatives: that blob list's top K a scan "
                                             "(nodule_blobs.py top); the rest of --extra-nlst's rows are ordinary negatives")
    r.add_argument("--ten", action="store_true")
    r.add_argument("--skip-v2", action="store_true",
                   help="score only the stage 1 list (a trial on it): LUNA16's V2 points are neither scored nor written")
    r.add_argument("--folds", type=int, nargs="+")
    r.add_argument("--steps", type=int)
    r.add_argument("--set", action="append", default=[], help="key=<json>")

    sm = sub.add_parser("smoke")
    sm.add_argument("--set", action="append", default=[], help="key=<json>, on top of smoke's small settings")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"rescore": rescore, "smoke": smoke, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
