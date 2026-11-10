"""rAidiology's spine outliner v2: the whole spine (C1 down to L5) and the spinal cord on a sagittal MRI, with no
TotalSpineSeg anywhere in its chain, written with TotalSpineSeg's step-2 label ids (labels.py: cord 1, canal 2,
C1-C7 11-17, T1-T12 21-32, L1-L5 41-45, discs 63-100 numbered by the vertebra below) so a v2 file drops in wherever
a TotalSpineSeg file went (run_pipeline.py, metrics.py, build_meshes.py, build_stacks.py, qc_overlay.py,
specialist.py read labels.outline_path()).

Why: TotalSpineSeg's released weights learned partly from CC BY-NC data (LICENSING.md, "TotalSpineSeg trained
weights"), and the owner trains and runs only on data and weights he can sell (2026-11-10). v1 (finder.py, lumbar
discs only) freed the disc grader; v2 frees the patient pipeline. finder.py stays as it is for the grader.

Classes: background, vertebra, disc, canal, cord. No sacrum: SPIDER counts vertebra 1 as the most caudal lumbar
vertebra and leaves the sacrum unlabelled (it is background in its masks), and no allowed source labels it
(whole-spine's 6 label-sacrum_seg files name an unidentified "nnUNet3D"). So v2 never writes id 50; metrics.py then
measures L5-S1's mid-body canal from L5 alone (its reference list drops a missing sacrum), and build_meshes.py and
build_stacks.py draw no sacrum.

Label sources (label_verdict(), refused in code; census() counts them):
  SPIDER (Zenodo 10159290, CC BY 4.0), development split only (split.json): lumbar vertebrae, discs, canal on T2.
  Spine Generic data-multi-subject (CC BY 4.0), T1w/T2w: label-SC_seg (cord); label-discs_dlabel (hand-placed disc
    points, SCT's numbering: 2 = C1-C2, 3 = C2-C3 ... 8 = C7-T1 ... 20 = T12-L1 ... 25 = L5-S1). No canal_seg at all:
    two-thirds name TotalSpineSeg, and the rest name SCT's canal model (ivadomed/model-canal-seg), whose training list
    (ressources/canal.json, scripts/data/download_datasets.sh, read 2026-11-10) is the canal_seg files of whole-spine
    (made by TotalSegMRI) and data-multi-subject (mostly TotalSpineSeg), so it learned from TotalSpineSeg's canals.
    The canal is learned from SPIDER's expert masks only; beside the cord it is constrained by the cord's partial label.
  whole-spine (OpenNeuro ds005616, CC0), T1w/T2w: label-SC_seg only where its sidecar names an SCT model. Its README
    says "TotalSpineSeg: Used for segmenting spinal cord, vertebrae, and intervertebral disks", so a sidecar there that
    names only Manual steps may sit on a corrected TotalSpineSeg output: left out. Its canal_seg names TotalSegMRI
    (TotalSpineSeg's earlier name, neuropoly/totalsegmentator-mri): refused. Its labels-discs sidecars are empty: left out.
  Refused whatever the sidecar says: every label-spine_dseg. Refused by GeneratedBy: TotalSpineSeg, TotalSegMRI,
  TotalSegmentator, SynthRAD2023, GoldAtlas, MRSpineSeg, and SCT's canal model (trained on TotalSpineSeg canals). A missing, empty or unidentified GeneratedBy leaves the file
  out. No pretrained weights: the net starts from random numbers. Runs under RAIDIOLOGY_NO_TSS=1 (tss.py refuses).

Model: finder.py's 2D U-Net (a sagittal slice and its two neighbours at 1 mm in-plane), seven outputs: five class
logits, a disc heatmap and a disc level.
  Partial labels: every pixel carries the SET of classes it may be (a bitmask), and the class loss is the marginal
  cross-entropy -log(sum of p over that set) (partial_loss): a SPIDER canal pixel may be canal or cord (SPIDER's canal
  holds the conus), a Spine Generic pixel beside the cord (within the cord label's extent) may be anything but cord,
  a pixel no source labels may be anything and costs nothing. Soft Dice is added per class over the pixels that
  decide that class (their set is exactly it, or leaves it out).
  Disc points: Spine Generic marks each disc by one hand-placed voxel at its posterior tip, valued by its level. The
  heatmap head learns a Gaussian (sigma 3 mm) at every disc's posterior tip: SPIDER gives the tip of each disc mask
  on each slice, so its slices are fully supervised; Spine Generic gives the slices within 2 of its points, rows from
  15 mm above the first point to 15 mm below the last (elsewhere unknown). The level head regresses the level number
  / 25 within 4 pixels of each tip (SPIDER's discs by their nominal level, L5-S1 = 25 counted upward).
  Reading (outline): heatmap peaks (non-maximum suppression at 7 mm), plus any disc component of 300 mm3 or more
  with no peak, are the discs; each takes the level head's vote; number_discs sorts them top to bottom, counts the
  steps between neighbours (a gap 1.7 times the local spacing or more is a missed disc) and fits ONE offset to all the
  votes (their median), so the discs found across the whole field of view number themselves: a neck scan from C2-C3
  down, a lumbar scan up to L5-S1, a whole spine from both ends, never by counting up from L5-S1 alone. Vertebra and
  disc voxels take their level from where they sit along that chain of discs (chain_coord), one level per connected
  piece when 70% of it agrees.

Folds: SPIDER's development folds (split.json, the grader's); Spine Generic and whole-spine subjects by a hash of the
subject id (bids_fold). Model k never saw fold k; `all` reads new patients.

Usage: python outliner.py cache                          -> build the training caches once (before folds train side by side)
       python outliner.py train <0|1|2|3|4|all> [...]    -> <MODELS>/spine-outliner-v2/<fold>.pt
       python outliner.py predict [--folds 0,..]         -> out of fold: SPIDER dev patients, held-out Spine Generic / whole-spine
       python outliner.py score [--folds 0,..]           -> Dice per class, disc-point mm, level accuracy, 95% CIs by patient (ledger)
       python outliner.py segment <in.nii.gz> <out.nii.gz> [--fold all]
       python outliner.py census <spine-generic|whole-spine> <root>   -> kept / refused / unclear label files
       python outliner.py fetchlist <spine-generic|whole-spine> <root> -> the kept label files and their images, to download"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("KMP_BLOCKTIME", "1")

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repository root
from core import ledger, paths  # noqa: E402
paths.use("spine_mri")
from crops import Series  # noqa: E402
from finder import build_net, load_case as finder_case, resample_slices, scaled, split_patients, three, to_pir, zooms  # noqa: E402
from prepare import WORK as SPIDER_WORK, disc_id as spider_disc_id  # noqa: E402

VERSION = 2
OUT = os.path.join(paths.MODELS, f"spine-outliner-v{VERSION}")
V2 = os.path.join(paths.WORK, "eval", "spine_outliner")
MM = 1.0                      # in-plane spacing the net sees
CROP = 256
CLASSES = ("background", "vertebra", "disc", "canal", "cord")
BG, VERT, DISC, CANAL, CORD = range(len(CLASSES))
ALL = (1 << len(CLASSES)) - 1   # a pixel that may be any class: unlabelled, costs nothing
HEAT, LEVEL = len(CLASSES), len(CLASSES) + 1
OUTPUTS = len(CLASSES) + 2
LEVEL_SCALE = 25.0
CFG = {"iters": int(os.environ.get("OUTLINER_ITERS", 12000)), "batch": 16, "lr": 1e-3, "weight_decay": 1e-4, "width": 32,
       "seed": 20261110, "sigma_px": 3.0, "heat_w": 10.0, "level_w": 1.0, "level_radius": 4,
       "share": {"spider": 0.5, "spine-generic": 0.35, "whole-spine": 0.15}}
PEAK, NMS_MM, NEAR_PEAK_MM, MIN_DISC_MM3, CHAIN_MM, GAP = 0.3, 7.0, 12.0, 300.0, 60.0, 1.7
FOLDS = 5
CARD_GB = 4.0
DATASETS = {"spine-generic": os.environ.get("SPINE_GENERIC_ROOT", os.path.join(paths.DATASETS, "public", "spine-generic")),
            "whole-spine": os.environ.get("WHOLE_SPINE_ROOT", os.path.join(paths.DATASETS, "public", "whole-spine"))}
CONTRASTS = ("T1w", "T2w")

# ---------------------------------------------------------------- label ids (TotalSpineSeg step 2, labels.py)


def disc_label(level):
    """TotalSpineSeg id of the disc at SCT level `level` (3 = C2-C3 ... 8 = C7-T1 ... 20 = T12-L1 ... 25 = L5-S1); 0 if none."""
    level = int(level)
    return 60 + level if 3 <= level <= 7 else 63 + level if 8 <= level <= 19 else 71 + level if 20 <= level <= 24 \
        else 100 if level == 25 else 0


def vertebra_label(index):
    """TotalSpineSeg id of vertebra `index` (1 = C1 ... 8 = T1 ... 20 = L1 ... 24 = L5); 0 outside C1-L5 (no sacrum in v2)."""
    index = int(index)
    return 10 + index if 1 <= index <= 7 else 13 + index if 8 <= index <= 19 else 21 + index if 20 <= index <= 24 else 0


DISC_OF = np.array([disc_label(v) for v in range(27)], np.uint16)
VERT_OF = np.array([vertebra_label(v) for v in range(27)], np.uint16)

# ---------------------------------------------------------------- label sources: the sidecar gate

USED = {"label-SC_seg": "cord", "label-canal_seg": "canal", "label-discs_dlabel": "discs", "labels-discs": "discs",
        "labels-disc-manual": "discs"}
REFUSED_KINDS = {"label-spine_dseg": "a spine_dseg file (TotalSpineSeg output: Spine Generic's sidecars name it, "
                                     "whole-spine's README says TotalSpineSeg segmented its vertebrae and discs)"}
# model-canal-seg / sc_canal_t2: SCT's canal model learned from TotalSpineSeg- and TotalSegMRI-made canal_seg files
FORBIDDEN = ("totalspineseg", "totalsegmri", "totalsegmentator", "synthrad", "goldatlas", "mrspineseg",
             "model-canal-seg", "sc_canal_t2")
KNOWN = ("manual", "visual check", "spinalcordtoolbox", "sct_deepseg")
MANUAL_ONLY_UNCLEAR = {"whole-spine": "README says 'TotalSpineSeg: Used for segmenting spinal cord, vertebrae, and "
                                      "intervertebral disks'"}


class Refused(Exception):
    """A label file this build may not read (a refused kind, a forbidden tool in its sidecar, or no clear sidecar)."""


def split_name(path):
    """(stem, contrast, kind) of a BIDS label file: sub-x_T2w_label-SC_seg.nii.gz -> (sub-x_T2w, T2w, label-SC_seg)."""
    b = os.path.basename(path)
    for ext in (".nii.gz", ".nii", ".json"):
        if b.endswith(ext):
            b = b[:-len(ext)]
            break
    i = b.find("_label-")
    i = b.find("_labels-") if i < 0 else i
    if i < 0:
        return b, None, None
    return b[:i], b[:i].split("_")[-1], b[i + 1:]


def sidecar_of(path):
    for ext in (".nii.gz", ".nii"):
        if path.endswith(ext):
            return path[:-len(ext)] + ".json"
    return path


def label_verdict(path, dataset):
    """('keep' | 'refuse' | 'unclear', why) for a label file (its .nii.gz or .json path), read from its JSON sidecar."""
    kind = split_name(path)[2]
    if kind in REFUSED_KINDS:
        return "refuse", REFUSED_KINDS[kind]
    folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))   # labels/<sub>/anat/<file>
    who = os.path.basename(path).split("_")[0]
    if folder.startswith("sub-") and who != folder:   # whole-spine files sub-amuALT's cord under sub-amuAM
        return "unclear", f"named for {who} but filed under {folder}"
    side = sidecar_of(path)
    if not os.path.exists(side):
        return "unclear", "no sidecar"
    try:
        d = json.load(open(side, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return "unclear", f"unreadable sidecar ({e.__class__.__name__})"
    gb = d.get("GeneratedBy") if isinstance(d, dict) else None
    if not isinstance(gb, list) or not gb or not all(isinstance(x, dict) and str(x.get("Name", "")).strip() for x in gb):
        return "unclear", "sidecar names no GeneratedBy"
    hit = [t for t in FORBIDDEN if t in json.dumps(gb).lower()]
    if hit:
        return "refuse", "GeneratedBy names " + ", ".join(hit)
    names = [str(x["Name"]).strip().lower() for x in gb]
    odd = [n for n in names if not n.startswith(KNOWN)]
    if odd:
        return "unclear", "GeneratedBy names an unidentified tool: " + odd[0][:60]
    if dataset in MANUAL_ONLY_UNCLEAR and all(n.startswith(("manual", "visual check")) for n in names):
        return "unclear", "only Manual steps, and the dataset's " + MANUAL_ONLY_UNCLEAR[dataset]
    return "keep", " + ".join(n.split(" -")[0][:48] for n in names)


def read_label(path, dataset):
    """The label image, only when its sidecar clears it; anything else raises Refused (never a silent read)."""
    verdict, why = label_verdict(path, dataset)
    if verdict != "keep":
        raise Refused(f"{os.path.basename(path)}: {verdict}: {why}")
    return nib.load(path)


def census(dataset, root):
    """[(path relative to root, kind, contrast, verdict, why)] for every label file of a kind v2 reads or refuses,
    whether only its sidecar or also its image is on disk."""
    seen = {}
    for dp, _, fs in os.walk(os.path.join(root, "derivatives", "labels")):
        for f in fs:
            stem, contrast, kind = split_name(f)
            if kind in USED or kind in REFUSED_KINDS:
                seen.setdefault(os.path.join(dp, f"{stem}_{kind}.nii.gz"), (kind, contrast))
    return [(os.path.relpath(p, root), k, c, *label_verdict(p, dataset)) for p, (k, c) in sorted(seen.items())]


def image_of(rel_label):
    """The image a label file belongs to, relative to the dataset root."""
    stem = split_name(rel_label)[0]
    d = os.path.dirname(rel_label).split(os.path.join("derivatives", "labels") + os.sep, 1)[-1]
    return os.path.join(d, stem + ".nii.gz")


def fetchlist(dataset, root):
    """The kept label files of the contrasts v2 reads, with their images: the only files the box downloads."""
    out = set()
    for rel, _, contrast, verdict, _ in census(dataset, root):
        if verdict == "keep" and contrast in CONTRASTS:
            out.update((rel, image_of(rel)))
    return sorted(out)


def bids_fold(subject):
    return int(hashlib.sha256(subject.encode()).hexdigest(), 16) % FOLDS


def bids_cases(dataset, root=None):
    """[{dataset, subject, stem, image, cord?, canal?, discs?}]: one per T1w/T2w image with a kept label on disk."""
    root = root or DATASETS[dataset]
    cases = {}
    for rel, kind, contrast, verdict, _ in census(dataset, root):
        img = os.path.join(root, image_of(rel))
        if verdict != "keep" or contrast not in CONTRASTS or not os.path.exists(os.path.join(root, rel)) or not os.path.exists(img):
            continue
        stem = split_name(rel)[0]
        c = cases.setdefault(stem, {"dataset": dataset, "subject": stem.split("_")[0], "stem": stem, "image": img})
        c[USED[kind]] = os.path.join(root, rel)
    return [cases[k] for k in sorted(cases)]

# ---------------------------------------------------------------- training cases


def tips(lab, levels):
    """(N, 4) [slice, row, col, level]: the posterior tip (most posterior 15%) of each disc on each slice of a resampled
    (LR, AP, SI) id volume; levels maps an id to its SCT level."""
    pts = []
    for v, level in levels.items():
        for k in np.nonzero((lab == v).any((1, 2)))[0]:
            yx = np.argwhere(lab[k] == v)
            if len(yx) < 10:
                continue
            post = yx[yx[:, 0] >= np.percentile(yx[:, 0], 85)]
            pts.append((k, *post.mean(0), level))
    return np.array(pts, np.float32).reshape(-1, 4)


def _save_case(d, img, allowed, pts, heat_k, rows, spread):
    os.makedirs(d, exist_ok=True)
    useful = np.nonzero((allowed != ALL).any((1, 2)) | heat_k)[0]
    np.save(os.path.join(d, "img.npy"), img.astype(np.float16))
    np.save(os.path.join(d, "allowed.npy"), allowed.astype(np.uint8))
    np.save(os.path.join(d, "pts.npy"), pts.astype(np.float32))
    np.save(os.path.join(d, "heat_k.npy"), heat_k.astype(bool))
    np.save(os.path.join(d, "useful.npy"), useful.astype(np.int32))
    json.dump({"rows": [int(rows[0]), int(rows[1])], "spread": spread}, open(os.path.join(d, "meta.json"), "w"))


def load_cached(d):
    """A cached case, its volumes memory-mapped (side-by-side folds share the page cache)."""
    m = json.load(open(os.path.join(d, "meta.json")))
    ld = lambda n: np.load(os.path.join(d, n + ".npy"), mmap_mode="r")  # noqa: E731
    return {"img": ld("img"), "allowed": ld("allowed"), "pts": np.asarray(ld("pts")), "heat_k": np.asarray(ld("heat_k")),
            "useful": np.asarray(ld("useful")), "rows": m["rows"], "spread": m["spread"]}


def spider_case(p):
    """A SPIDER development patient: every pixel decided (canal = canal or cord), disc tips on every slice."""
    d = os.path.join(V2, "cache", "spider", str(p))
    if not os.path.exists(os.path.join(d, "meta.json")):
        img, lab = finder_case(p)
        allowed = np.array([1 << BG, 1 << VERT, 1 << DISC, (1 << CANAL) | (1 << CORD)], np.uint8)[lab]
        t2 = os.path.join(SPIDER_WORK, "dev", "t2", f"{p}.nii.gz")
        m = np.asarray(nib.load(os.path.join(SPIDER_WORK, "dev", "mask", f"{p}.nii.gz")).dataobj).astype(np.int32)
        dl = resample_slices(np.where(m > 200, m, 0), zooms(nib.load(t2).affine), 0)
        levels = {200 + i: 26 - i for i in range(1, 18) if spider_disc_id(i)}     # disc 1 (L5-S1) is SCT level 25
        _save_case(d, img, allowed, tips(dl, levels), np.ones(len(img), bool), (0, img.shape[2]), 0)
    return load_cached(d)


def bids_case(c):
    """A Spine Generic / whole-spine image: cord, canal and disc points where its kept labels give them, else unknown."""
    d = os.path.join(V2, "cache", c["dataset"], c["stem"])
    if not os.path.exists(os.path.join(d, "meta.json")):
        img = nib.load(c["image"])
        a, aff, _ = to_pir(img)
        z = zooms(aff)
        vol = resample_slices(scaled(Series(a=a, aff=aff)), z, 1)
        allowed = np.full(vol.shape, ALL, np.uint8)

        def lab(key):
            im = read_label(c[key], c["dataset"])
            if im.shape[:3] != img.shape[:3]:
                raise ValueError(f"{c[key]}: grid {im.shape} is not its image's {img.shape}")
            return to_pir(im)[0]

        def band(m):
            r = np.nonzero(m.any((0, 1)))[0]
            b = np.zeros(m.shape, bool)
            if len(r):
                b[:, :, r.min():r.max() + 1] = True
            return b

        if "cord" in c:
            cord = resample_slices((lab("cord") > 0.5).astype(np.uint8), z, 0) > 0
            allowed[band(cord) & ~cord] &= np.uint8(ALL & ~(1 << CORD))
            allowed[cord] = 1 << CORD
        if "canal" in c:
            canal = resample_slices((lab("canal") > 0.5).astype(np.uint8), z, 0) > 0
            allowed[band(canal) & ~canal] &= np.uint8(ALL & ~((1 << CANAL) | (1 << CORD)))
            allowed[canal] &= np.uint8((1 << CANAL) | (1 << CORD))
        pts = np.zeros((0, 4), np.float32)
        heat_k = np.zeros(len(vol), bool)
        rows = (0, 0)
        if "discs" in c:
            dl = np.rint(np.asarray(lab("discs"), np.float32)).astype(np.int32)
            p = []
            for v in np.unique(dl):
                if 2 <= v <= 25:
                    ap, si, lr = np.argwhere(dl == v).mean(0)
                    p.append((lr, ap * z[0] / MM, si * z[1] / MM, v))
            pts = np.array(p, np.float32).reshape(-1, 4)
            if len(pts):
                for k in np.unique(np.rint(pts[:, 0]).astype(int)):
                    heat_k[max(0, k - 2):k + 3] = True
                rows = (max(0, pts[:, 2].min() - 15 / MM), min(vol.shape[2], pts[:, 2].max() + 15 / MM))
        _save_case(d, vol, allowed, pts, heat_k, rows, 2)
    return load_cached(d)


def training_cases(fold):
    """{source: [case]} for model `fold`: every case outside that fold (all of them for `all`)."""
    out = {"spider": [spider_case(p) for p, f in sorted(split_patients().items()) if fold == "all" or f != int(fold)]}
    for ds in DATASETS:
        out[ds] = [bids_case(c) for c in bids_cases(ds) if fold == "all" or bids_fold(c["subject"]) != int(fold)]
    return {k: v for k, v in out.items() if v}


def sample(case, rng, sigma=CFG["sigma_px"], radius=CFG["level_radius"]):
    """(x (3, CROP, CROP), y (5, CROP, CROP): allowed, heat, heat mask, level, level mask) from one random useful slice."""
    k = int(case["useful"][rng.integers(len(case["useful"]))])
    x = three(case["img"], k).astype(np.float32)
    y = np.asarray(case["allowed"][k])
    ph, pw = max(0, CROP - y.shape[0]), max(0, CROP - y.shape[1])
    o0, o1 = ph // 2, pw // 2
    if ph or pw:
        x = np.pad(x, ((0, 0), (o0, ph - o0), (o1, pw - o1)), constant_values=-0.5)
        y = np.pad(y, ((o0, ph - o0), (o1, pw - o1)))                 # 0: unknown
    pts = case["pts"]
    near = pts[np.abs(pts[:, 0] - k) <= case["spread"] + 0.5] if len(pts) else pts
    fg = np.argwhere((y == 1 << DISC) | (y == 1 << CORD))
    if len(near) and rng.random() < 0.5:
        c = near[rng.integers(len(near)), 1:3] + (o0, o1)
    elif len(fg) and rng.random() < 0.8:
        c = fg[rng.integers(len(fg))]
    else:
        c = None
    if c is None:
        a0, a1 = rng.integers(y.shape[0] - CROP + 1), rng.integers(y.shape[1] - CROP + 1)
    else:
        a0 = int(np.clip(int(c[0]) - rng.integers(CROP), 0, y.shape[0] - CROP))
        a1 = int(np.clip(int(c[1]) - rng.integers(CROP), 0, y.shape[1] - CROP))
    x, y = x[:, a0:a0 + CROP, a1:a1 + CROP], y[a0:a0 + CROP, a1:a1 + CROP]
    t = np.zeros((5, CROP, CROP), np.float32)
    t[0] = y
    if case["heat_k"][k]:
        lo, hi = case["rows"]
        t[2][:, max(0, int(lo) + o1 - a1):max(0, int(hi) + o1 - a1)] = 1
        yy, xx = np.mgrid[0:CROP, 0:CROP]
        for _, r, s, level in near:
            r, s = r + o0 - a0, s + o1 - a1
            if -4 * sigma < r < CROP + 4 * sigma and -4 * sigma < s < CROP + 4 * sigma:
                d2 = (yy - r) ** 2 + (xx - s) ** 2
                t[1] = np.maximum(t[1], np.exp(-d2 / (2 * sigma ** 2)))
                disk = d2 <= radius ** 2
                t[3][disk], t[4][disk] = level / LEVEL_SCALE, 1
    if rng.random() < 0.5:                                          # left-right: the neighbours swap
        x = x[::-1].copy()
    x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.3:
        x = np.sign(x) * np.abs(x) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.3:                                          # T1w and T2w both: the canal may be dark or bright
        x = 1.0 - x
    return x.astype(np.float32), t


def partial_loss(z, allowed):
    """(marginal cross-entropy, soft Dice) of class logits z (B, C, H, W) against `allowed` (B, H, W), each pixel's
    bitmask of the classes it may be. -log(sum of p over the allowed set): one allowed class is ordinary
    cross-entropy, several share the probability, all of them (or 0, padding) cost nothing and pass no gradient.
    Dice per class over the pixels that decide it (the set is exactly that class, or leaves it out)."""
    import torch
    C = z.shape[1]
    a = allowed.long()
    bit = torch.arange(C, device=z.device).view(1, C, 1, 1)
    ok = ((a.unsqueeze(1) >> bit) & 1).bool()
    known = (a > 0) & (a < ALL)
    logp = z.float().log_softmax(1)
    lse = torch.logsumexp(logp.masked_fill(~(ok | ~known.unsqueeze(1)), float("-inf")), 1)   # padding (0) as "any": no -inf
    ce = -(lse * known).sum() / known.sum().clamp(min=1)
    t = (a.unsqueeze(1) == (1 << bit)).float()
    dec = ((~ok) | t.bool()) & known.unsqueeze(1)
    p = logp.exp() * dec
    used = dec[:, 1:].flatten(2).any(2).any(0)
    if not used.any():
        return ce, ce * 0
    inter, den = (p * t).sum((0, 2, 3))[1:], (p + t * dec).sum((0, 2, 3))[1:]
    return ce, 1 - ((2 * inter + 1) / (den + 1))[used].mean()


def train(fold):
    """Model `fold` (0-4: trains on every case outside that fold; `all`: on every case)."""
    import torch
    import torch.nn.functional as F
    from core import card
    meta_p = os.path.join(OUT, "outliner.json")
    if os.path.exists(os.path.join(OUT, f"{fold}.pt")) and os.path.exists(meta_p):
        m = json.load(open(meta_p))
        if m.get("cfg") == CFG and str(fold) in m.get("models", {}):
            print(f"fold {fold}: already trained with these settings", flush=True)
            return 0.0
    card.lease(CARD_GB, f"spine outliner {fold}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    card.require_card(dev, torch)
    seed = CFG["seed"] + (99 if fold == "all" else int(fold))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    cases = training_cases(fold)
    names = sorted(cases)
    share = np.array([CFG["share"][n] for n in names])
    share = share / share.sum()
    counts = {n: len(cases[n]) for n in names}
    print(f"fold {fold}: {counts} loaded in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    net = build_net(CFG["width"], OUTPUTS).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=CFG["iters"], pct_start=0.05)
    part = os.path.join(OUT, f"{fold}.part.pt")
    start = 0
    if os.path.exists(part):
        ck = torch.load(part, map_location=dev, weights_only=False)
        if ck["cfg"] == CFG:
            net.load_state_dict(ck["net"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            rng.bit_generator.state = ck["rng"]
            torch.set_rng_state(ck["torch_rng"])
            start = ck["it"]
            print(f"fold {fold}: resumed at step {start}", flush=True)
    net.train()
    log = open(os.path.join(OUT, f"train_{fold}.log"), "a")
    for it in range(start, CFG["iters"]):
        if it and it % 1000 == 0 and it != start:
            torch.save({"cfg": CFG, "it": it, "net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, part)
        b = []
        for _ in range(CFG["batch"]):
            src = cases[names[rng.choice(len(names), p=share)]]
            b.append(sample(src[rng.integers(len(src))], rng))
        x = torch.from_numpy(np.stack([s[0] for s in b])).to(dev)
        y = torch.from_numpy(np.stack([s[1] for s in b])).to(dev)
        sc = torch.from_numpy(rng.uniform(0.85, 1.15, len(b)).astype(np.float32)).to(dev)   # size, on the card
        theta = torch.zeros(len(b), 2, 3, device=dev)
        theta[:, 0, 0] = theta[:, 1, 1] = 1 / sc
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z = net(x)
        z = z.float()
        ce, dice = partial_loss(z[:, :len(CLASSES)], y[:, 0].round())
        hm, ht = y[:, 2], y[:, 1]
        h = torch.sigmoid(z[:, HEAT])
        heat = CFG["heat_w"] * ((1 + 9 * ht) * hm * (h - ht) ** 2).sum() / hm.sum().clamp(min=1)
        lm = y[:, 4] > 0
        level = CFG["level_w"] * F.smooth_l1_loss(z[:, LEVEL][lm], y[:, 3][lm], beta=0.05) if lm.any() else z.sum() * 0
        loss = ce + dice + heat + level
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == CFG["iters"] - 1:
            s = (f"{time.strftime('%H:%M:%S')} fold {fold} iter {it} loss {loss.item():.4f} ce {ce.item():.4f} "
                 f"dice {dice.item():.4f} heat {heat.item():.4f} level {level.item():.4f}")
            print(s, flush=True)
            log.write(s + "\n")
            log.flush()
    path = os.path.join(OUT, f"{fold}.pt")
    torch.save(net.state_dict(), path)
    if os.path.exists(part):
        os.remove(part)
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.update({"version": VERSION, "cfg": CFG, "mm": MM, "classes": list(CLASSES), "outputs": OUTPUTS,
                 "data": "SPIDER development T2 masks (CC BY 4.0, split.json folds); Spine Generic data-multi-subject "
                         "(CC BY 4.0) and whole-spine ds005616 (CC0) label files that label_verdict kept",
                 "init": "random (torch default), no pretrained weights", "tss": "none (RAIDIOLOGY_NO_TSS=1)"})
    meta.setdefault("models", {})[str(fold)] = {"cases": counts, "seconds": round(time.time() - t0),
                                                "trained": time.strftime("%Y-%m-%d %H:%M"),
                                                "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(meta, open(meta_p, "w"), indent=1)
    print(f"fold {fold} saved {path} in {time.time() - t0:.0f}s", flush=True)
    return time.time() - t0

# ---------------------------------------------------------------- reading


_NETS = {}


def load(fold):
    import torch
    if fold not in _NETS:
        p = os.path.join(OUT, f"{fold}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p}: spine outliner v2 model {fold} is not trained (tools/remote/aws/spine_outliner_box.sh)")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = build_net(CFG["width"], OUTPUTS).to(dev)
        net.load_state_dict(torch.load(p, map_location=dev))
        _NETS[fold] = net.eval()
    return _NETS[fold]


def probabilities(net, vol, aff):
    """(classes (5, AP, SI, LR), heat (AP, SI, LR), level (AP, SI, LR)) on the volume's own grid, from a PIR volume scaled
    like finder.scaled()."""
    import torch
    dev = next(net.parameters()).device
    r = resample_slices(vol, zooms(aff), 1)
    h, w = r.shape[1:]
    H, W = -(-h // 16) * 16, -(-w // 16) * 16
    seg = np.zeros((len(CLASSES), *vol.shape), np.float32)
    heat, level = np.zeros(vol.shape, np.float32), np.zeros(vol.shape, np.float32)
    with torch.no_grad():
        for k in range(len(r)):
            x = np.full((3, H, W), -0.5, np.float32)
            x[:, :h, :w] = three(r, k)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = net(torch.from_numpy(x)[None].to(dev)).float()[:, :, :h, :w]
            o = torch.nn.functional.interpolate(o, size=vol.shape[:2], mode="bilinear", align_corners=False)[0]
            seg[:, :, :, k] = o[:len(CLASSES)].softmax(0).cpu().numpy()
            heat[:, :, k] = torch.sigmoid(o[HEAT]).cpu().numpy()
            level[:, :, k] = (o[LEVEL] * LEVEL_SCALE).cpu().numpy()
    return seg, heat, level


def find_discs(cls, heat, level, z):
    """(points (N, 3) PIR voxels, level votes (N,)): heatmap peaks after 7 mm suppression, plus the posterior tip of any
    disc component of MIN_DISC_MM3 or more with no peak within NEAR_PEAK_MM."""
    size = [2 * max(1, int(round(NMS_MM / zz))) + 1 for zz in z]
    cand = np.argwhere((heat >= PEAK) & (heat == ndimage.maximum_filter(heat, size=size, mode="constant")))
    cand = cand[np.argsort(-heat[tuple(cand.T)])] if len(cand) else cand
    pts = []
    for c in cand:
        if all(np.linalg.norm((c - q) * z) >= NMS_MM for q in pts):
            pts.append(c.astype(np.float64))
    disc = cls == DISC
    comp, n = ndimage.label(disc)
    for i, sl in enumerate(ndimage.find_objects(comp), start=1):
        if sl is None:
            continue
        vox = np.argwhere(comp[sl] == i) + [s.start for s in sl]
        if len(vox) * np.prod(z) < MIN_DISC_MM3:
            continue
        tip = vox[vox[:, 0] >= np.percentile(vox[:, 0], 85)].mean(0)
        if all(np.linalg.norm((tip - q) * z) >= NEAR_PEAK_MM for q in pts):
            pts.append(tip)
    P = np.array(pts, np.float64).reshape(-1, 3)
    votes = []
    for p in P:
        i = np.rint(p).astype(int)
        lo, hi = np.maximum(i - 1, 0), np.minimum(i + 2, level.shape)
        votes.append(float(np.median(level[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])))
    return P, np.array(votes, np.float64)


def number_discs(pos_mm, votes):
    """SCT level of each disc (2 = C1-C2, 3 = C2-C3 ... 25 = L5-S1), in the input order. The discs are sorted top to
    bottom (PIR axis 1 runs inferior), each step between neighbours is one level unless the gap is GAP times the local
    spacing (a disc the reading missed), and one offset, the median of (vote - rank), numbers the whole chain."""
    pos_mm, votes = np.asarray(pos_mm, np.float64).reshape(-1, 3), np.asarray(votes, np.float64)
    if len(pos_mm) == 0:
        return np.zeros(0, int)
    order = np.argsort(pos_mm[:, 1], kind="stable")
    gaps = np.linalg.norm(np.diff(pos_mm[order], axis=0), axis=1)
    steps = []
    for i, g in enumerate(gaps):
        ref = [gaps[j] for j in (i - 1, i + 1) if 0 <= j < len(gaps)]
        ref = float(np.median(ref)) if ref else g
        steps.append(max(1, int(round(g / ref))) if ref > 0 and g / ref >= GAP else 1)
    rank = np.concatenate([[0], np.cumsum(steps)]).astype(int)
    offset = int(np.rint(np.median(votes[order] - rank)))
    out = np.zeros(len(pos_mm), int)
    out[order] = offset + rank
    return out


def chain_coord(X, P):
    """(u, distance mm) of points X (M, 3, mm) along the chain of disc points P (n, 3, mm, top to bottom): u = i + t
    between disc i (t = 0) and disc i + 1 (t = 1), extended past both ends by the end segment's length (one disc alone:
    by 20 mm, its own vertebra's height)."""
    X = np.asarray(X, np.float64)
    if len(P) == 1:
        return (X[:, 1] - P[0, 1]) / 20.0, np.linalg.norm(X - P[0], axis=1)
    best_d, best_u = np.full(len(X), np.inf), np.zeros(len(X))
    for i in range(len(P) - 1):
        d = P[i + 1] - P[i]
        t = (X - P[i]) @ d / max(d @ d, 1e-9)
        dist = np.linalg.norm(X - (P[i] + np.clip(t, 0, 1)[:, None] * d), axis=1)
        better = dist < best_d
        best_d[better], best_u[better] = dist[better], i + t[better]
    return best_u, best_d


def vertebra_index(u, L):
    """Vertebra index (1 = C1 ... 24 = L5) at chain coordinate u between discs of SCT levels L (the vertebra below disc
    level v is vertebra v; a gap of k levels holds k vertebrae)."""
    u, L = np.asarray(u, np.float64), np.asarray(L, int)
    n = len(L)
    if n == 1:
        return L[0] + np.floor(u).astype(int)
    j = np.clip(np.floor(u).astype(int), 0, n - 2)
    inside = L[j] + np.floor((u - j) * (L[j + 1] - L[j])).astype(int)
    return np.where(u < 0, L[0] + np.floor(u).astype(int),
                    np.where(u >= n - 1, L[-1] + np.floor(u - (n - 1)).astype(int), inside))


def _by_piece(mask, vox, ids, share=0.7):
    """One id per connected piece of `mask` where `share` of its voxels agree (a vertebra's posterior elements stay with
    its body); a piece split more evenly keeps its per-voxel ids."""
    comp = ndimage.label(mask)[0][tuple(vox.T)]
    order = np.argsort(comp, kind="stable")
    for grp in np.split(order, np.flatnonzero(np.diff(comp[order])) + 1):
        v = ids[grp]
        v = v[v > 0]
        if len(v):
            vals, cnt = np.unique(v, return_counts=True)
            if cnt.max() >= share * len(grp):
                ids[grp] = vals[cnt.argmax()]
    return ids


def outline(seg, heat, level, aff):
    """(label volume with TotalSpineSeg step-2 ids, [disc dicts]) from v2's outputs on a PIR grid."""
    z = zooms(aff)
    cls = seg.argmax(0)
    lab = np.zeros(cls.shape, np.uint16)
    lab[cls == CANAL] = 2
    lab[cls == CORD] = 1
    P, votes = find_discs(cls, heat, level, z)
    if not len(P):
        return lab, []
    levels = number_discs(P * z, votes)
    order = np.argsort(P[:, 1], kind="stable")
    P, levels, votes = P[order], levels[order], votes[order]
    keep = (levels >= 2) & (levels <= 25)
    P, levels, votes = P[keep], levels[keep], votes[keep]
    discs = [{"pir": [round(float(c), 1) for c in p], "level": int(v), "vote": round(float(w), 2), "id": int(DISC_OF[v])}
             for p, v, w in zip(P, levels, votes)]
    if not len(P):
        return lab, discs
    for c in (VERT, DISC):
        vox = np.argwhere(cls == c)
        if not len(vox):
            continue
        u, dist = chain_coord(vox * z, P * z)
        if c == VERT:
            ids = VERT_OF[np.clip(vertebra_index(u, levels), 0, 26)].astype(np.int64)
        else:
            j = np.clip(np.rint(u).astype(int), 0, len(P) - 1)
            ids = np.where((u >= -0.5) & (u <= len(P) - 0.5), DISC_OF[levels[j]], 0).astype(np.int64)
        ids[dist > CHAIN_MM] = 0
        ids = _by_piece(cls == c, vox, ids)
        lab[tuple(vox.T)] = ids
    return lab, discs


def read_volume(path, fold="all", net=None):
    """(label volume on the input's own grid, discs (PIR voxels), PIR zooms) for a sagittal MRI in any orientation."""
    img = nib.load(path)
    a, aff, back = to_pir(img)
    seg, heat, level = probabilities(net or load(fold), scaled(Series(a=a, aff=aff)), aff)
    lab, discs = outline(seg, heat, level, aff)
    return back(lab), discs, zooms(aff)


def segment(in_path, out_path, fold="all"):
    """v2 labels (TotalSpineSeg step-2 ids) for a sagittal MRI, written on its own grid: run_pipeline.py's outliner."""
    lab, _, _ = read_volume(in_path, fold)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    nib.save(nib.Nifti1Image(lab.astype(np.uint16), nib.load(in_path).affine), out_path)
    return out_path

# ---------------------------------------------------------------- out of fold: predict and score


def _jobs(folds=None):
    """[(source, key, fold, image, out stem)]: SPIDER development patients and Spine Generic / whole-spine images."""
    jobs = [("spider", p, f, os.path.join(SPIDER_WORK, "dev", "t2", f"{p}.nii.gz"), os.path.join(V2, "pred", "spider", str(p)))
            for p, f in sorted(split_patients().items())]
    for ds in DATASETS:
        jobs += [(ds, c, bids_fold(c["subject"]), c["image"], os.path.join(V2, "pred", ds, c["stem"])) for c in bids_cases(ds)]
    return [j for j in jobs if folds is None or j[2] in folds]


def predict(folds=None):
    """Out-of-fold labels: each image read by the model of its own fold -> <work>/eval/spine_outliner/pred/."""
    from core import card
    card.lease(CARD_GB, "spine outliner predict")
    n = 0
    for _, _, f, image, stem in _jobs(folds):
        lab, discs, z = read_volume(image, net=load(str(f)))
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        nib.save(nib.Nifti1Image(lab.astype(np.uint16), nib.load(image).affine), stem + ".nii.gz")
        json.dump({"zooms": [float(v) for v in z], "discs": discs}, open(stem + ".json", "w"), indent=1)
        n += 1
    print(f"predicted {n} images", flush=True)
    return n


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def interval(df, group, col, draws=2000, seed=0):
    """(pooled mean, 2.5%, 97.5%) of `col`, the 95% interval from resampling `group` (patients) 2,000 times."""
    d = df[[group, col]].dropna()
    if d.empty:
        return np.nan, np.nan, np.nan
    g = d.groupby(group)[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = np.random.default_rng(seed).integers(len(g), size=(draws, len(g)))
    lo, hi = np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    return float(s.sum() / n.sum()), float(lo), float(hi)


def score(led=None, folds=None):
    """Out-of-fold scores, each with a 95% interval by patient: SPIDER lumbar Dice per class and disc level agreement
    (nominal level); cord and canal Dice on held-out Spine Generic / whole-spine images (within each label's own
    extent); disc-point distance (mm) and level accuracy against the hand-placed points."""
    import pandas as pd
    per, spd, bids, pts = [], [], [], []
    for src, key, f, image, stem in _jobs(folds):
        if not os.path.exists(stem + ".nii.gz"):
            continue
        lab = to_pir(nib.load(stem + ".nii.gz"))[0].astype(np.int32)
        if src == "spider":
            m = to_pir(nib.load(os.path.join(SPIDER_WORK, "dev", "mask", f"{key}.nii.gz")))[0].astype(np.int32)
            per.append({"patient": key, "fold": f, "dice_vertebra": dice((m >= 1) & (m < 100), (lab >= 11) & (lab <= 50)),
                        "dice_disc": dice(m > 200, lab >= 63), "dice_canal": dice(m == 100, (lab == 1) | (lab == 2))})
            for i in sorted({int(v) - 200 for v in np.unique(m) if v > 200}):
                want = spider_disc_id(i)
                if want is None:
                    continue
                gt = m == 200 + i
                inside = lab[gt]
                inside = inside[inside >= 63]
                best = int(np.bincount(inside).argmax()) if len(inside) else 0
                d = dice(gt, lab == best) if best else 0.0
                spd.append({"patient": key, "fold": f, "ivd": i, "dice": d, "found": int(d >= 0.3), "level_ok": int(d >= 0.3 and best == want)})
            continue
        subject = f"{src}/{key['subject']}"
        row = {"patient": subject, "stem": key["stem"], "fold": f}
        for part, pred in (("cord", lab == 1), ("canal", (lab == 1) | (lab == 2))):
            if part in key:
                gt = to_pir(read_label(key[part], src))[0] > 0.5
                if part == "canal" and "cord" in key:
                    gt |= to_pir(read_label(key["cord"], src))[0] > 0.5
                r = np.nonzero(gt.any((0, 2)))[0]
                if len(r):
                    row[f"dice_{part}"] = dice(gt[:, r.min():r.max() + 1], pred[:, r.min():r.max() + 1])
        bids.append(row)
        if "discs" in key:
            got = json.load(open(stem + ".json"))
            z = np.array(got["zooms"])
            found = np.array([d["pir"] for d in got["discs"]]).reshape(-1, 3)
            flv = np.array([d["level"] for d in got["discs"]], int)
            dl = np.rint(np.asarray(to_pir(read_label(key["discs"], src))[0], np.float32)).astype(int)
            for v in np.unique(dl):
                if 3 <= v <= 25:
                    p = np.argwhere(dl == v).mean(0)
                    dist = np.linalg.norm((found - p) * z, axis=1) if len(found) else np.array([np.inf])
                    j = int(dist.argmin())
                    hit = bool(dist[j] <= 10.0)
                    pts.append({"patient": subject, "level": int(v), "dist_mm": float(dist[j]) if np.isfinite(dist[j]) else np.nan,
                                "found": int(hit), "level_ok": int(hit and flv[j] == v)})
    os.makedirs(V2, exist_ok=True)
    frames = {"spider_patients": pd.DataFrame(per), "spider_discs": pd.DataFrame(spd), "bids": pd.DataFrame(bids),
              "disc_points": pd.DataFrame(pts)}
    for k, df in frames.items():
        df.to_csv(os.path.join(V2, f"score_{k}.csv"), index=False)
    res = {}
    for name, df, cols in (("spider", frames["spider_patients"], ("dice_vertebra", "dice_disc", "dice_canal")),
                           ("spider_disc", frames["spider_discs"], ("found", "level_ok", "dice")),
                           ("heldout", frames["bids"], ("dice_cord", "dice_canal")),
                           ("disc_point", frames["disc_points"], ("found", "level_ok", "dist_mm"))):
        res[f"{name}_patients"] = int(df.patient.nunique()) if len(df) else 0
        for c in cols:
            if c in df:
                res[f"{name}_{c}"], res[f"{name}_{c}_low"], res[f"{name}_{c}_high"] = interval(df, "patient", c)
                res[f"{name}_{c}_n"] = int(df[c].notna().sum())
    if len(frames["disc_points"]):
        res["disc_point_dist_mm_median"] = float(frames["disc_points"].dist_mm.median())
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    if led is not None:
        for k, v in res.items():
            led.number(k, float(v), "count" if k.endswith(("_patients", "_n")) else "mm" if "dist_mm" in k else "")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "train", "predict", "score", "segment", "census", "fetchlist"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--folds", default=None, help="predict/score: only these folds' images, e.g. 0 or 0,1")
    ap.add_argument("--fold", default="all", help="segment: the model to read with")
    a = ap.parse_args()
    folds = None if a.folds is None else sorted({int(f) for f in a.folds.split(",")})
    tag = "" if folds is None else " folds " + ",".join(map(str, folds))
    if a.mode == "cache":
        t0 = time.time()
        n = {k: len(v) for k, v in training_cases("all").items()}
        print(f"outliner cache: {n} in {time.time() - t0:.0f}s", flush=True)
    elif a.mode == "train":
        for f in a.args:
            with ledger.run("segmenter", f"spine outliner v{VERSION} fold {f}") as led:
                led.settings_from({"fold": f, "data": "SPIDER dev masks + Spine Generic + whole-spine kept labels",
                                   **{f"cfg.{k}": v for k, v in CFG.items()}})
                train(f)
    elif a.mode == "predict":
        with ledger.run("segmenter", f"spine outliner v{VERSION} predict out of fold{tag}"):
            predict(folds)
    elif a.mode == "score":
        with ledger.run("score", f"spine outliner v{VERSION} out of fold{tag}") as led:
            led.settings_from({"split": "SPIDER dev + Spine Generic + whole-spine, out of fold",
                               "folds": FOLDS if folds is None else ",".join(map(str, folds)), "found_mm": 10.0,
                               "min_dice": 0.3, "tss": "none"})
            score(led, folds)
    elif a.mode in ("census", "fetchlist"):
        ds, root = a.args
        if a.mode == "fetchlist":
            print("\n".join(fetchlist(ds, root)))
        else:
            from collections import Counter
            rows = census(ds, root)
            for (kind, verdict), n in sorted(Counter((r[1], r[3]) for r in rows).items()):
                print(f"{ds}\t{kind}\t{verdict}\t{n}")
            for r in rows:
                if r[3] != "keep":
                    print("\t".join(map(str, r)), file=sys.stderr)
    else:
        print(segment(*a.args, fold=a.fold))
