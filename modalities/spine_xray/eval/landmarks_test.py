"""Corners on the landmark set, an outside test (idea 32): v7's reading of the spondylolisthesis landmark pictures.

Plan and bar: docs/efficiency/README.md, "Corners on the landmark set, an outside test (idea 32". Dataset: the
Mendeley spondylolisthesis vertebral landmark set (landmarks.py; CC BY 4.0, LICENSING.md). No kept corner net has
seen one picture of it (idea 24 failed), so its 590 distinct pictures test v7 from outside.

The reading. v7 is `corners_reread.py --from-version 1 --passes 1 --looks 5`: corners-v1's fold net places the 22
points on the 512 x 384 canvas (`corners.predict`), then corners-v2's fold net of the SAME fold re-reads each point
from a 160 px window of the film (`corners_reread.reread` with `net_reader` and `load_net`), five looks. Here a
picture is read by all five folds and the reading's 22 points are the mean of the five folds' final points.

No label shapes the input (landmarks.render does scale by the drawn vertebrae and mirror by the drawn lines' slope;
that is a leak for a test, so it is not used; rec['scale'], rec['flip'] and rec['aspect'] are never read for the
input). rec['aspect'] is itself derived from the labels: landmarks._load_train fits it per picture so the median
height / width of the drawn vertebrae matches BUU's, clipped to 1.0 .. 2.4. Here the picture is un-squeezed (the
export squeezed it to 640 x 640) by ONE fixed number, `landmarks.buu_reference()['aspect']`: the median height /
width of BUU's 2,000 side films (the same constant landmarks passes to its first mapping; it comes from film sizes
only). Then buu._fit puts it on the canvas as it puts a BUU film. The refine nets want film pixels (BUU films are
about 2,000 px wide, 160 px windows), so the second look reads a film built from the same un-squeezed picture
resized to BUU's median film width (and that width times the aspect): fixed numbers, no label.

Orientation. The picture is read as it is and mirrored left-right; the reading kept is the one whose 22 points sit
closer to BUU's mean side-view shape after a similarity fit (Procrustes: shift, turn, size; no reflection), the mean
shape being the generalised Procrustes mean of BUU's drawn LA points. No landmark-set label is used.

The measure. Points are mapped back to the exported 640 x 640 frame (canvas -> un-squeezed -> un-mirrored ->
squeezed). The truth is the picture's drawn points in that frame (landmarks.file_frame inverts landmarks' aspect and
mirror), and at run time every mapped truth point is asserted to be one raw label point within 0.5 px. Over every
point both sides define: share within 5 px, within 15 px, mean miss; patient bootstrap 95% intervals (2,000
resamples, seed 0); BUU and other apart; the share of pictures where the mirrored reading was kept; per vertebra
within-5 share. The verdict line is against Bansal et al. 2026 (76.75 of 100 corners within 5 px).

Scored ONCE: `score` refuses to run when work/eval/spine_xray/landmarks-test-v7/result.json exists. The card share is
RAIDIOLOGY_CARD_GB (default 2.0), as corners.py; RAIDIOLOGY_DEVICE=cpu runs on the processor.

Four-fifths arrangement (idea 119, 2026-10-08): `--fold K --from-version A --refine-version B [--baseline]` reads only
fold K's held-out pictures (landmark_folds.csv, landmarks.fold_table) with the A -> B chain exactly as above (each BUU fold's
corners-A first look, then its corners-B refine net, one pass, five looks; the 22 points averaged over the five chains).
With --baseline v7's chain (1 -> 2) reads the same pictures, and the paired comparison (B minus v7) is printed and written:
within 5 px (the pass line's number) and within 15 px, mean miss, BUU and other apart, the mirrored readings' shares, with a
paired patient-bootstrap 95% interval (2,000 resamples, seed 0). Output: work/eval/spine_xray/landmarks-fold<K>-v<A>/, scored
once. The same reading (read_picture, choose) prepares the landmark pictures for training with corners.py and
corners_refine.py `landmark_fold` (training_set): no label shapes their input, and their points go through the same frames.

Usage: python landmarks_test.py score
       python landmarks_test.py score --fold K --from-version A --refine-version B [--baseline]
       python landmarks_test.py selftest      (synthetic, processor, no data file, no net)
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import paths  # noqa: E402
import buu  # noqa: E402
import landmarks  # noqa: E402

FRAME = landmarks.FRAME            # 640: the exported pictures' side
N_POINTS = landmarks.N_POINTS      # 22
CANVAS = (512, 384)                # the corner reader's canvas, height x width
BAR = 76.75                        # Bansal et al. 2026, YOLOv11n-Pose, share of corners within 5 px (x100)
WITHIN = (5.0, 15.0)
BOOT = 2000
BOOT_SEED = 0
TRUTH_TOL = 0.5                    # px: a mapped truth point must be a raw label point this closely
NAMES = ["L%d_%s" % (lev + 1, c) for lev in range(5) for c in ("UL", "UR", "LL", "LR")] + ["S1_L", "S1_R"]
VERTEBRAE = ["L1", "L2", "L3", "L4", "L5", "S1"]
OUT_NAME = "landmarks-test-v7"
FROM_VERSION, REFINE_VERSION = 1, 2
CITATION = "Mendeley spondylolisthesis vertebral landmark dataset, CC BY 4.0 (LICENSING.md); bar: Bansal et al., PLoS One 2026, e0347290"


def out_dir(fold=None, from_version=None):
    if fold is not None:                                          # idea 119: a new folder per fold and first-look version
        return os.path.join(paths.WORK, "eval", "spine_xray", "landmarks-fold%d-v%d" % (fold, from_version))
    return os.path.join(paths.WORK, "eval", "spine_xray", OUT_NAME)


def refuse_if_scored(d):
    """Scored once: a result.json in `d` means the test was already read."""
    if os.path.exists(os.path.join(d, "result.json")):
        raise SystemExit("landmarks_test: %s/result.json exists, so this test was already scored; it is scored once "
                         "and no change is tuned on these pictures" % d)


# --- the frames ------------------------------------------------------------------------------------------------

def mirror_points(p, width):
    """(..., 22, 2) points in a picture `width` px wide -> the same points in its left-right mirror, each line's two
    ends trading places so point 0 of a line stays its left end. Its own inverse; NaN stays NaN."""
    q = np.array(p, np.float64)
    q[..., 0] = width - q[..., 0]
    return q.reshape(-1, 11, 2, 2)[:, :, ::-1, :].reshape(q.shape)


class Frame:
    """One reading's coordinate frames for a picture of `size` = (width, height) px as exported (squeezed). The
    un-squeezed picture ("view") is width x round(width * asp); mirrored readings use its mirror. The canvas is
    buu._fit's fit of the view; the film is the view resized to `film_size` (width, height)."""

    def __init__(self, size, asp, mirrored, film_size, canvas=CANVAS):
        w, h = size
        self.w, self.h_u = int(w), int(round(w * asp))
        self.mirrored = bool(mirrored)
        self.squeeze = np.array([1.0, self.h_u / float(h)])
        from PIL import Image
        _c, _s, off, sc = buu._fit(Image.new("L", (self.w, self.h_u)), canvas[0], canvas[1])
        self.off, self.sc = off, np.asarray(sc, np.float64)
        self.film_size = (int(film_size[0]), int(film_size[1]))
        self.film_k = np.array([self.film_size[0] / float(self.w), self.film_size[1] / float(self.h_u)])

    def to_view(self, p):
        """Exported-frame points -> this reading's un-squeezed (and mirrored) picture's frame."""
        v = np.asarray(p, np.float64) * self.squeeze
        return mirror_points(v, self.w) if self.mirrored else v

    def to_canvas(self, v):
        return np.asarray(v, np.float64) * self.sc + self.off

    def from_canvas(self, c):
        return (np.asarray(c, np.float64) - self.off) / self.sc

    def to_film(self, v):
        return np.asarray(v, np.float64) * self.film_k

    def from_film(self, f):
        return np.asarray(f, np.float64) / self.film_k

    def to_exported(self, v):
        """This reading's view-frame points -> the exported 640 x 640 frame (un-mirrored, squeezed again)."""
        v = mirror_points(v, self.w) if self.mirrored else np.asarray(v, np.float64)
        return v / self.squeeze


def prepare(im, frame):
    """The exported picture (PIL) -> (canvas uint8 (512, 384), film uint8 (H, W)) for this frame, no label involved."""
    from PIL import Image, ImageOps
    u = im.convert("L").resize((frame.w, frame.h_u), Image.BILINEAR)
    if frame.mirrored:
        u = ImageOps.mirror(u)
    canvas, _s, off, sc = buu._fit(u, CANVAS[0], CANVAS[1])
    assert np.allclose(off, frame.off) and np.allclose(sc, frame.sc), "the canvas fit differs from the frame's"
    film = np.asarray(u.resize(frame.film_size, Image.BILINEAR), dtype=np.uint8)
    return canvas, film


# --- choosing the reading by BUU's mean shape -------------------------------------------------------------------

def _unit(s):
    """(..., 22, 2) -> centred, unit Frobenius norm."""
    c = s - s.mean(axis=-2, keepdims=True)
    return c / np.maximum(np.linalg.norm(c, axis=(-2, -1), keepdims=True), 1e-12)


def _rotation(a, b):
    """Rotations (no reflection) R (..., 2, 2) taking unit shapes a onto b: a @ R ~ b."""
    u, s, vt = np.linalg.svd(np.swapaxes(a, -1, -2) @ b)
    d = np.sign(np.linalg.det(u @ vt))
    d = np.where(d == 0, 1.0, d)
    u = u.copy()
    u[..., :, -1] *= d[..., None]
    return u @ vt


def procrustes_distance(a, b):
    """Similarity (shift, turn, size, no reflection) distance between shapes a (..., 22, 2) and b (22, 2): the
    residual after fitting, both shapes at unit size (0 = same shape, up to sqrt(2))."""
    a, b = _unit(np.asarray(a, np.float64)), _unit(np.asarray(b, np.float64))
    r = _rotation(a, np.broadcast_to(b, a.shape))
    return np.linalg.norm(a @ r - b, axis=(-2, -1))


def mean_shape(shapes, rounds=10):
    """The generalised Procrustes mean of (N, 22, 2) shapes, unit size and centred."""
    s = _unit(np.asarray(shapes, np.float64))
    m = s[0]
    for _ in range(rounds):
        r = _rotation(s, np.broadcast_to(m, s.shape))
        m = _unit((s @ r).mean(axis=0))
    return m


def buu_mean_shape():
    """BUU's mean side-view shape from its drawn LA points (no landmark-set label)."""
    _ids, pts, _cls = buu.annotations("LA")
    return mean_shape(pts)


def choose(plain, mirrored, ref):
    """('plain' | 'mirrored', distance plain, distance mirrored): the reading closer to the mean shape `ref`; a tie keeps plain."""
    dp, dm = float(procrustes_distance(plain, ref)), float(procrustes_distance(mirrored, ref))
    return ("mirrored" if dm < dp else "plain"), dp, dm


# --- the reading: v1's first look, then v7's re-read, per fold ---------------------------------------------------

def load_refine(fold, version):
    """Corners-v<version>'s refine net of fold `fold` (corners_refine.make_net), as corners_reread.load_net reads v2's."""
    import torch
    import corners
    import corners_refine as cr
    z = torch.load(os.path.join(cr.out_dir(version), "LA-fold%d.pt" % fold), map_location=corners.DEV, weights_only=False)
    net = cr.make_net(z["settings"]["widths"]).to(corners.DEV)
    net.load_state_dict(z["state"])
    net.eval()
    return net, float(z["info"]["mean"]), float(z["info"]["sd"])


def load_models(from_version=FROM_VERSION, refine_version=REFINE_VERSION):
    """[(first net, mean, sd, refine net, mean, sd)] for folds 0..4: corners-<from> first-look and corners-<refine> refine LA nets."""
    import torch
    import corners
    out = []
    for k in range(buu.FOLDS):
        z = torch.load(os.path.join(corners.out_dir(from_version), "LA-fold%d.pt" % k), map_location=corners.DEV,
                       weights_only=False)
        net1 = corners.UNet(z["settings"]["widths"], int(z["points"])).to(corners.DEV)
        net1.load_state_dict(z["state"])
        net1.eval()
        net2, m2, s2 = load_refine(k, refine_version)
        out.append((net1, float(z["info"]["mean"]), float(z["info"]["sd"]), net2, m2, s2))
    return out


def read_one(models, canvas, film, frame):
    """The 22 points (film pixels of this reading's view) of one prepared picture: per fold, v1's first look on the
    canvas, converted to film pixels, then corners_reread.reread (one pass, five looks) with that fold's refine net;
    the mean over the five folds' final points."""
    import torch
    import corners
    import corners_reread as rr
    pix = torch.from_numpy(canvas[None]).to(corners.DEV)
    finals = []
    for net1, m1, s1, net2, m2, s2 in models:
        first = corners.predict(net1, pix, [0], m1, s1, 1)[0].astype(np.float64)        # canvas pixels
        start = frame.to_film(frame.from_canvas(first))
        cur, _first = rr.reread(rr.net_reader(net2, m2, s2), [film], start[None], 1, rr.LOOKS_5)
        finals.append(cur[0])
    return np.mean(finals, axis=0)


def fixed_film():
    """(un-squeezing aspect, film (width, height)): the one fixed number and BUU's median film size every reading uses."""
    ref_b = landmarks.buu_reference()
    asp = float(ref_b["aspect"])                                  # BUU's films only; no label
    film_w = int(round(float(np.median(ref_b["sizes"][:, 0]))))   # BUU's median film width, and that times the aspect
    return asp, (film_w, int(round(film_w * asp)))


def read_picture(im, chains, asp, film, mean):
    """One exported picture (PIL, 640 x 640) read both ways by each chain (a list of load_models() results). Per chain,
    choose() keeps the reading closer to BUU's mean shape. Returns per chain (way, frame, film points, canvas, film image)
    of the kept reading: the frame maps its film points back to the exported frame (to_exported(from_film(points)))."""
    reads = {}
    for mirrored in (False, True):
        fr = Frame(im.size, asp, mirrored, film)
        canvas, filmpix = prepare(im, fr)
        reads[mirrored] = (fr, canvas, filmpix, [read_one(models, canvas, filmpix, fr) for models in chains])
    out = []
    for c in range(len(chains)):
        way, _dp, _dm = choose(reads[False][3][c], reads[True][3][c], mean)
        fr, canvas, filmpix, pts = reads[way == "mirrored"]
        out.append((way, fr, pts[c], canvas, filmpix))
    return out


def training_pictures(recs, asp, film, mean, chains):
    """The landmark pictures as corners.py / corners_refine.py train on them with landmark_fold (idea 119): each read and
    oriented as the test does (read_picture; no label shapes its input), its drawn points (landmarks.file_frame, the
    picture file's own geometry) mapped through the frame that was kept. Returns a dict: canvas (M, 512, 384) uint8, canvas
    points (M, 22, 2) (NaN where the set has no point or it falls off the canvas), film (M, H, W) uint8, film points
    (M, 22, 2) (NaN where the set has none), and mirrored (M,) bool (the reading kept the mirror)."""
    from PIL import Image
    n, (fw, fh) = len(recs), film
    out = {"canvas": np.zeros((n, CANVAS[0], CANVAS[1]), np.uint8), "canvas_pts": np.full((n, N_POINTS, 2), np.nan),
           "film": np.zeros((n, fh, fw), np.uint8), "film_pts": np.full((n, N_POINTS, 2), np.nan),
           "mirrored": np.zeros(n, bool)}
    for i, rec in enumerate(recs):
        drawn = landmarks.file_frame(rec)
        with Image.open(rec["path"]) as im:
            assert im.size == (int(FRAME), int(FRAME)), "%s is %s, not the 640 x 640 export" % (rec["name"], im.size)
            way, fr, _pts, canvas, filmpix = read_picture(im, chains, asp, film, mean)[0]
        q = fr.to_canvas(fr.to_view(drawn))
        q[(q[:, 0] < 0) | (q[:, 0] >= CANVAS[1]) | (q[:, 1] < 0) | (q[:, 1] >= CANVAS[0])] = np.nan
        out["canvas"][i], out["canvas_pts"][i] = canvas, q
        out["film"][i], out["film_pts"][i] = filmpix, fr.to_film(fr.to_view(drawn))
        out["mirrored"][i] = way == "mirrored"
    return out


def training_set(exclude_fold):
    """training_pictures for the landmark set without fold `exclude_fold` (the held-out one), read by the v7 chain (the
    chain that trained on none of the set). Returns the records and the dict of training_pictures."""
    table = landmarks.fold_table()
    train, _held = landmarks.split_fold(landmarks.load("train"), table, exclude_fold)
    asp, film = fixed_film()
    out = training_pictures(train, asp, film, buu_mean_shape(), [load_models()])
    print("landmark_fold %d: %d pictures trained on (fold %d never trains); %d kept the mirror by the v%d reading"
          % (exclude_fold, len(train), exclude_fold, int(out["mirrored"].sum()), FROM_VERSION), flush=True)
    return train, out


# --- truth -------------------------------------------------------------------------------------------------------

def label_path(rec):
    d = os.path.join(landmarks.root(), "Train", "Keypointrcnn_data", "labels", rec["split"], rec["name"])
    for e in (".json", ".txt"):
        if os.path.exists(d + e):
            return d + e
    raise SystemExit("no label file for %s" % rec["name"])


def check_truth(truth, raw, name="", tol=TRUTH_TOL):
    """Every defined truth point (22, 2) must equal one raw label point (the vertebrae of landmarks.read_label, any
    order) within tol px. Returns the largest distance found; raises AssertionError otherwise."""
    pts = np.concatenate([np.asarray(v, np.float64).reshape(-1, 2) for v in raw])
    worst = 0.0
    for k in range(N_POINTS):
        if np.isnan(truth[k]).any():
            continue
        d = float(np.linalg.norm(pts - truth[k], axis=1).min())
        assert d <= tol, "%s: truth point %s is %.3f px from every raw label point (limit %.1f)" % (name, NAMES[k], d, tol)
        worst = max(worst, d)
    return worst


# --- scoring -----------------------------------------------------------------------------------------------------

patient_of = landmarks.patient_of           # the BUU id (as text) for a BUU-named picture, else the base name (idea 119: one key)


def miss_of(truth, pred):
    """(n, 22) distance in px, NaN where the truth lacks the point (or the prediction does)."""
    return np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(truth, np.float64), axis=-1)


def per_picture(miss):
    """Counts per picture from (n, 22) misses: points defined, within 5, within 15, sum of miss."""
    ok = ~np.isnan(miss)
    m = np.where(ok, miss, 0.0)
    return {"n": ok.sum(1).astype(np.float64), "w5": ((m <= WITHIN[0]) & ok).sum(1).astype(np.float64),
            "w15": ((m <= WITHIN[1]) & ok).sum(1).astype(np.float64), "sum": m.sum(1)}


def pooled(counts):
    n = counts["n"].sum()
    return {"points": int(n), "within5": 100.0 * counts["w5"].sum() / n, "within15": 100.0 * counts["w15"].sum() / n,
            "mean_miss": float(counts["sum"].sum() / n)}


def boot_ci(num, den, patients, n=BOOT, seed=BOOT_SEED):
    """95% interval of sum(num) / sum(den) with patients resampled whole (one draw of `n` resamples). `patients`
    names each picture's patient."""
    _u, inv = np.unique(np.asarray(patients), return_inverse=True)
    pn, pd = np.bincount(inv, weights=num), np.bincount(inv, weights=den)
    idx = np.random.default_rng(seed).integers(0, len(pn), size=(n, len(pn)))
    stat = pn[idx].sum(1) / pd[idx].sum(1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(counts, patients):
    """Within 5 (x100), within 15 (x100) and mean miss, each with a patient-bootstrap interval, for the pictures in
    `counts` (the dict of per_picture's arrays)."""
    p = pooled(counts)
    out = {"pictures": int(len(counts["n"])), "patients": int(len(set(patients))), "points": p["points"]}
    for key, num, scale in (("within5", counts["w5"], 100.0), ("within15", counts["w15"], 100.0),
                            ("mean_miss", counts["sum"], 1.0)):
        lo, hi = boot_ci(num, counts["n"], patients)
        out[key] = {"value": p[key], "low": scale * lo, "high": scale * hi}
    return out


def verdict(lo, hi, bar=BAR):
    """The line against Bansal et al.: BEATEN iff the interval's low end is above the bar, BEHIND iff its high end is
    below, else LEVEL."""
    word = "BEATEN" if lo > bar else "BEHIND" if hi < bar else "LEVEL"
    return "Against Bansal et al. 2026 (%.2f of 100 corners within 5 px): %s" % (bar, word)


# --- the run -----------------------------------------------------------------------------------------------------

def score():
    d = out_dir()
    refuse_if_scored(d)
    t0 = time.time()
    from PIL import Image
    recs = landmarks.load("train")
    got = (len(recs), sum(r["source"] == "buu" for r in recs))
    assert got == (landmarks.EXPECT["bases"], landmarks.EXPECT["buu"]), "landmark set drifted: %s" % (got,)
    import corners
    if corners.DEV.type == "cuda":
        corners.limit_card_memory()
    asp, film = fixed_film()
    mean = buu_mean_shape()
    models = load_models()
    print("landmarks_test: %d pictures; un-squeeze aspect %.4f (BUU's median film height / width), film %d x %d; "
          "device %s" % (len(recs), asp, film[0], film[1], corners.DEV), flush=True)
    truth = np.full((len(recs), N_POINTS, 2), np.nan)
    pred = np.full((len(recs), N_POINTS, 2), np.nan)
    kept = []
    worst = 0.0
    for n, rec in enumerate(recs):
        truth[n] = landmarks.file_frame(rec)
        worst = max(worst, check_truth(truth[n], landmarks.read_label(label_path(rec)), rec["name"]))
        with Image.open(rec["path"]) as im:
            assert im.size == (int(FRAME), int(FRAME)), "%s is %s, not the 640 x 640 export" % (rec["name"], im.size)
            way, fr, pts, _canvas, _film = read_picture(im, [models], asp, film, mean)[0]
        pred[n] = fr.to_exported(fr.from_film(pts))
        kept.append(way)
        if (n + 1) % 50 == 0 or n + 1 == len(recs):
            print("  %d / %d pictures read, %.0f s" % (n + 1, len(recs), time.time() - t0), flush=True)
    assert not np.isnan(pred).any(), "a prediction is NaN"
    print("truth check: all %d pictures' mapped truth points are raw label points (largest distance %.4f px)"
          % (len(recs), worst), flush=True)

    miss = miss_of(truth, pred)
    counts = per_picture(miss)
    patients = np.array([patient_of(r["base"]) for r in recs])
    group = np.array(["buu" if r["source"] == "buu" else "other" for r in recs])
    kept = np.array(kept)
    mirrored_share = {"all": 100.0 * float((kept == "mirrored").mean())}
    res = {"all": summarize(counts, patients)}
    for g in ("buu", "other"):
        sel = group == g
        res[g] = summarize({k: v[sel] for k, v in counts.items()}, patients[sel])
        mirrored_share[g] = 100.0 * float((kept[sel] == "mirrored").mean())
    per_vert = {}
    for v in VERTEBRAE:
        cols = [i for i, nm in enumerate(NAMES) if nm.startswith(v + "_")]
        m = miss[:, cols]
        ok = ~np.isnan(m)
        per_vert[v] = {"points": int(ok.sum()), "within5": 100.0 * float(((m <= WITHIN[0]) & ok).sum() / max(ok.sum(), 1))}

    a = res["all"]
    print("landmarks test, v7's reading, %d pictures (%d BUU, %d other), %d points both sides define; 640 x 640 frame; "
          "patient-bootstrap 95%% intervals, %d resamples, seed %d" % (a["pictures"], res["buu"]["pictures"],
                                                                      res["other"]["pictures"], a["points"], BOOT, BOOT_SEED))
    for name in ("all", "buu", "other"):
        r = res[name]
        print("  %-5s %3d pictures (%d patients): within 5 px %.2f (%.2f to %.2f), within 15 px %.2f (%.2f to %.2f), "
              "mean miss %.2f px (%.2f to %.2f); mirrored reading kept in %.1f%%"
              % (name, r["pictures"], r["patients"], r["within5"]["value"], r["within5"]["low"], r["within5"]["high"],
                 r["within15"]["value"], r["within15"]["low"], r["within15"]["high"], r["mean_miss"]["value"],
                 r["mean_miss"]["low"], r["mean_miss"]["high"], mirrored_share[name]))
    print("  per vertebra within 5 px: " + ", ".join("%s %.1f (%d points)" % (v, per_vert[v]["within5"], per_vert[v]["points"])
                                                      for v in VERTEBRAE))
    line = verdict(a["within5"]["low"], a["within5"]["high"])
    print(line)

    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "points.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["picture", "patient", "group", "kept", "point", "x_true", "y_true", "x_pred", "y_pred", "miss"])
        for n, rec in enumerate(recs):
            for k in range(N_POINTS):
                if np.isnan(miss[n, k]):
                    continue
                w.writerow([rec["name"], patients[n], group[n], kept[n], NAMES[k], "%.3f" % truth[n, k, 0],
                            "%.3f" % truth[n, k, 1], "%.3f" % pred[n, k, 0], "%.3f" % pred[n, k, 1], "%.3f" % miss[n, k]])
    seconds = time.time() - t0
    result = {"verdict_line": line, "bar_within5": BAR, "all": res["all"], "buu": res["buu"], "other": res["other"],
              "mirrored_kept_percent": mirrored_share, "per_vertebra": per_vert, "aspect": asp, "film": list(film),
              "truth_largest_distance_px": worst, "boot": BOOT, "seed": BOOT_SEED, "seconds": seconds,
              "citation": CITATION,
              "reading": "corners-v1 first look, corners-v2 re-read, 1 pass, 5 looks, mean of the 5 folds' final points"}
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)

    from core import ledger
    numbers = {"within5_pct": (a["within5"]["value"], "%"), "within5_pct_low": (a["within5"]["low"], "%"),
               "within5_pct_high": (a["within5"]["high"], "%"), "within15_pct": (a["within15"]["value"], "%"),
               "within15_pct_low": (a["within15"]["low"], "%"), "within15_pct_high": (a["within15"]["high"], "%"),
               "mean_miss_px": (a["mean_miss"]["value"], "px"), "mean_miss_px_low": (a["mean_miss"]["low"], "px"),
               "mean_miss_px_high": (a["mean_miss"]["high"], "px"),
               "buu_within5_pct": (res["buu"]["within5"]["value"], "%"), "other_within5_pct": (res["other"]["within5"]["value"], "%"),
               "mirrored_kept_pct": (mirrored_share["all"], "%"), "pictures": (a["pictures"], "count"),
               "points": (a["points"], "count")}
    ledger.write("corners", "landmark set outside test: v7's reading (idea 32)", seconds,
                 {k: (float(v), u) for k, (v, u) in numbers.items()},
                 settings={"from_version": FROM_VERSION, "refine_version": REFINE_VERSION, "passes": 1, "looks": 5,
                           "bar_within5": BAR, "verdict": line.rsplit(": ", 1)[1], "aspect": round(asp, 4)},
                 note="modalities/spine_xray/eval/landmarks_test.py; " + CITATION)


# --- the four-fifths arrangement (idea 119) ----------------------------------------------------------------------

def paired(ca, cb, patients, num, scale=100.0):
    """Chain A minus chain B on one pooled count: num 'w5' or 'w15' (shares, scale 100) or 'sum' (mean miss, scale 1).
    Returns the value and a paired patient-bootstrap 95% interval: one draw of the patients, the same for both chains."""
    den = ca["n"]
    assert np.array_equal(den, cb["n"]), "the two chains do not define the same points"
    value = scale * (ca[num].sum() / den.sum() - cb[num].sum() / den.sum())
    _u, inv = np.unique(np.asarray(patients), return_inverse=True)
    pa, pb, pd = (np.bincount(inv, weights=x) for x in (ca[num], cb[num], den))
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(pd), size=(BOOT, len(pd)))
    stat = scale * (pa[idx].sum(1) / pd[idx].sum(1) - pb[idx].sum(1) / pd[idx].sum(1))
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return {"value": float(value), "low": float(lo), "high": float(hi)}


def score_fold(k, from_v, refine_v, baseline):
    """Fold k's held-out landmark pictures read by the from_v -> refine_v chain (and, with baseline, by v7's chain on the
    same pictures); prints and writes both and the paired comparison. Scored once: refuse_if_scored on its folder."""
    d = out_dir(k, from_v)
    refuse_if_scored(d)
    t0 = time.time()
    from PIL import Image
    recs = _fold_records(k)
    keys, chains, asp, film, mean, dev = _fold_models(from_v, refine_v, baseline)
    _print_fold_start(k, recs, from_v, refine_v, baseline, asp, film, dev)
    truth, pred, kept, worst = _read_fold(recs, keys, chains, asp, film, mean, t0, Image)
    patients, group, subsets = _fold_groups(recs)
    res, mirror, diff, passed = _fold_results(keys, truth, pred, kept, patients, subsets, baseline)
    line, pass_text = _verdict_texts(res, passed, baseline)
    _print_fold_results(k, from_v, refine_v, baseline, keys, res, mirror, diff, line, pass_text)
    _write_fold_points(d, recs, keys, truth, pred, kept, patients, group)
    seconds = time.time() - t0
    _write_fold_outputs(k, from_v, refine_v, baseline, d, res, mirror, diff, passed, line, pass_text, worst, asp, film, seconds)


def _fold_records(k):
    table = landmarks.fold_table()
    recs_all = landmarks.load("train")
    got = (len(recs_all), sum(r["source"] == "buu" for r in recs_all))
    assert got == (landmarks.EXPECT["bases"], landmarks.EXPECT["buu"]), "landmark set drifted: %s" % (got,)
    _train, recs = landmarks.split_fold(recs_all, table, k)
    return recs


def _fold_models(from_v, refine_v, baseline):
    import corners
    if corners.DEV.type == "cuda":
        corners.limit_card_memory()
    asp, film = fixed_film()
    mean = buu_mean_shape()
    keys, chains = ["B"], [load_models(from_v, refine_v)]
    if baseline:
        keys.append("v7")
        chains.append(load_models(FROM_VERSION, REFINE_VERSION))
    return keys, chains, asp, film, mean, corners.DEV


def _print_fold_start(k, recs, from_v, refine_v, baseline, asp, film, dev):
    chain_text = "B = corners-v%d then v%d" % (from_v, refine_v)
    if baseline:
        chain_text += "; v7 = corners-v%d then v%d" % (FROM_VERSION, REFINE_VERSION)
    print("landmarks_test fold %d: %d held-out pictures (%s); un-squeeze aspect %.4f, film %d x %d; device %s"
          % (k, len(recs), chain_text, asp, film[0], film[1], dev), flush=True)


def _read_fold(recs, keys, chains, asp, film, mean, t0, Image):
    truth = np.full((len(recs), N_POINTS, 2), np.nan)
    pred = {key: np.full((len(recs), N_POINTS, 2), np.nan) for key in keys}
    kept = {key: [] for key in keys}
    worst = 0.0
    for n, rec in enumerate(recs):
        truth[n] = landmarks.file_frame(rec)
        worst = max(worst, check_truth(truth[n], landmarks.read_label(label_path(rec)), rec["name"]))
        with Image.open(rec["path"]) as im:
            assert im.size == (int(FRAME), int(FRAME)), "%s is %s, not the 640 x 640 export" % (rec["name"], im.size)
            reads = read_picture(im, chains, asp, film, mean)
        for key, (way, fr, pts, _canvas, _film) in zip(keys, reads):
            pred[key][n] = fr.to_exported(fr.from_film(pts))
            kept[key].append(way)
        if (n + 1) % 25 == 0 or n + 1 == len(recs):
            print("  %d / %d pictures read, %.0f s" % (n + 1, len(recs), time.time() - t0), flush=True)
    for key in keys:
        assert not np.isnan(pred[key]).any(), "a %s prediction is NaN" % key
    print("truth check: all %d pictures' mapped truth points are raw label points (largest distance %.4f px)"
          % (len(recs), worst), flush=True)
    return truth, pred, kept, worst


def _fold_groups(recs):
    patients = np.array([patient_of(r["base"]) for r in recs])
    group = np.array(["buu" if r["source"] == "buu" else "other" for r in recs])
    subsets = {"all": np.ones(len(recs), bool), "buu": group == "buu", "other": group == "other"}
    return patients, group, subsets


def _fold_results(keys, truth, pred, kept, patients, subsets, baseline):
    counts = {key: per_picture(miss_of(truth, pred[key])) for key in keys}
    res, mirror = {}, {}
    for key in keys:
        kept_a = np.array(kept[key])
        res[key] = {name: summarize({c: v[sel] for c, v in counts[key].items()}, patients[sel]) for name, sel in subsets.items()}
        mirror[key] = {name: 100.0 * float((kept_a[sel] == "mirrored").mean()) for name, sel in subsets.items()}
    diff, passed = {}, None
    if baseline:
        for name, sel in subsets.items():
            ca = {c: v[sel] for c, v in counts["B"].items()}
            cb = {c: v[sel] for c, v in counts["v7"].items()}
            diff[name] = {"within5": paired(ca, cb, patients[sel], "w5"), "within15": paired(ca, cb, patients[sel], "w15"),
                          "mean_miss": paired(ca, cb, patients[sel], "sum", 1.0)}
        d5 = diff["all"]["within5"]
        passed = bool(d5["value"] >= 2.0 and d5["low"] > 0)
    return res, mirror, diff, passed


def _verdict_texts(res, passed, baseline):
    a = res["B"]["all"]
    line = verdict(a["within5"]["low"], a["within5"]["high"])
    pass_text = ("idea 119 pass line: B minus v7 within 5 px at least +2.00, its low end above 0: %s"
                 % ("PASS" if passed else "NOT MET")) if baseline else "idea 119 pass line: not computed (needs --baseline)"

    return line, pass_text


def _print_fold_results(k, from_v, refine_v, baseline, keys, res, mirror, diff, line, pass_text):
    a = res["B"]["all"]
    print("landmarks test, fold %d, B = corners-v%d then v%d: %d pictures (%d BUU, %d other, %d patients); %d points; "
          "640 x 640 frame; patient-bootstrap 95%% intervals, %d resamples, seed %d"
          % (k, from_v, refine_v, a["pictures"], res["B"]["buu"]["pictures"], res["B"]["other"]["pictures"], a["patients"],
             a["points"], BOOT, BOOT_SEED))
    for key in keys:
        for name in ("all", "buu", "other"):
            r = res[key][name]
            print("  %-3s %-5s %3d pictures (%d patients): within 5 px %.2f (%.2f to %.2f), within 15 px %.2f (%.2f to %.2f), "
                  "mean miss %.2f px (%.2f to %.2f); mirrored reading kept in %.1f%%"
                  % (key, name, r["pictures"], r["patients"], r["within5"]["value"], r["within5"]["low"], r["within5"]["high"],
                     r["within15"]["value"], r["within15"]["low"], r["within15"]["high"], r["mean_miss"]["value"],
                     r["mean_miss"]["low"], r["mean_miss"]["high"], mirror[key][name]))
    if baseline:
        for name in ("all", "buu", "other"):
            x = diff[name]
            print("  B minus v7, %-5s: within 5 px %+.2f (%+.2f to %+.2f), within 15 px %+.2f (%+.2f to %+.2f), "
                  "mean miss %+.2f px (%+.2f to %+.2f)"
                  % (name, x["within5"]["value"], x["within5"]["low"], x["within5"]["high"], x["within15"]["value"],
                     x["within15"]["low"], x["within15"]["high"], x["mean_miss"]["value"], x["mean_miss"]["low"],
                     x["mean_miss"]["high"]))
    print("  B against Bansal: " + line)
    print(pass_text)


def _write_fold_points(d, recs, keys, truth, pred, kept, patients, group):
    os.makedirs(d, exist_ok=True)
    cols = ["picture", "patient", "group", "point", "x_true", "y_true"]
    for key in keys:
        cols += ["kept_%s" % key, "x_%s" % key, "y_%s" % key, "miss_%s" % key]
    miss = {key: miss_of(truth, pred[key]) for key in keys}
    with open(os.path.join(d, "points.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for n, rec in enumerate(recs):
            for j in range(N_POINTS):
                if np.isnan(truth[n, j]).any():
                    continue
                row = [rec["name"], patients[n], group[n], NAMES[j], "%.3f" % truth[n, j, 0], "%.3f" % truth[n, j, 1]]
                for key in keys:
                    row += [kept[key][n], "%.3f" % pred[key][n, j, 0], "%.3f" % pred[key][n, j, 1], "%.3f" % miss[key][n, j]]
                w.writerow(row)


def _write_fold_outputs(k, from_v, refine_v, baseline, d, res, mirror, diff, passed, line, pass_text, worst, asp, film,
                        seconds):
    a = res["B"]["all"]
    result = {"fold": k, "from_version": from_v, "refine_version": refine_v, "baseline": baseline,
              "chains": res, "mirrored_kept_percent": mirror, "diff_B_minus_v7": diff or None, "pass_line": passed,
              "verdict_line": line, "pass_line_text": pass_text, "bar_within5": BAR, "aspect": asp, "film": list(film),
              "truth_largest_distance_px": worst, "boot": BOOT, "seed": BOOT_SEED, "seconds": seconds, "citation": CITATION,
              "reading": "corners-v%d first look, corners-v%d refine, 1 pass, 5 looks, mean of the 5 folds' final points"
                         % (from_v, refine_v)}
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)

    from core import ledger
    numbers = {"within5_pct": (a["within5"]["value"], "%"), "within5_pct_low": (a["within5"]["low"], "%"),
               "within5_pct_high": (a["within5"]["high"], "%"), "within15_pct": (a["within15"]["value"], "%"),
               "mean_miss_px": (a["mean_miss"]["value"], "px"), "buu_within5_pct": (res["B"]["buu"]["within5"]["value"], "%"),
               "other_within5_pct": (res["B"]["other"]["within5"]["value"], "%"), "pictures": (a["pictures"], "count")}
    if baseline:
        numbers.update({"v7_within5_pct": (res["v7"]["all"]["within5"]["value"], "%"),
                        "diff_within5_pct": (diff["all"]["within5"]["value"], "%"),
                        "diff_within5_pct_low": (diff["all"]["within5"]["low"], "%"),
                        "diff_within5_pct_high": (diff["all"]["within5"]["high"], "%")})
    ledger.write("corners", "landmark set four-fifths test: fold %d, corners-v%d then v%d (idea 119)" % (k, from_v, refine_v),
                 seconds, {key: (float(v), u) for key, (v, u) in numbers.items()},
                 settings={"fold": k, "from_version": from_v, "refine_version": refine_v, "baseline": baseline, "passes": 1,
                           "looks": 5, "pass_line": passed},
                 note="modalities/spine_xray/eval/landmarks_test.py score --fold; " + CITATION)


# --- selftest ----------------------------------------------------------------------------------------------------

def _synthetic_lines():
    """A made-up side view as 22 points [L1 UL, UR, LL, LR, ..., L5 ..., S1 left, right] with every line's left end
    first and a chiral shape: the upper edges' angle runs -10, -2, 6, 14, 22 degrees, S1's is 36."""
    pts = []
    for lev in range(5):
        cx, cy, w, h = 120 + 6.0 * lev * lev, 60 + 55.0 * lev, 44.0 + 2 * lev, 34.0
        au, al = np.radians(-10 + 8 * lev), np.radians((-10 + 8 * lev) / 2)
        for (ang, y0) in ((au, -h / 2), (al, h / 2)):
            for sgn in (-1, 1):
                pts.append((cx + sgn * w / 2 * np.cos(ang), cy + y0 + sgn * w / 2 * np.sin(ang)))
    for sgn in (-1, 1):
        a = np.radians(36)
        pts.append((140 + 12 + sgn * 25 * np.cos(a), 60 + 55 * 5 + 10 + sgn * 25 * np.sin(a)))
    return np.array(pts, np.float64)


def _raw_vertebrae(pts, rng):
    """22 points -> raw label vertebrae as landmarks.read_label would give: 6 boxes of 4 points in any index order."""
    out = []
    for lev in range(5):
        q = pts[4 * lev:4 * lev + 4].copy()
        out.append(q[rng.permutation(4)])
    s1 = np.concatenate([pts[20:22], pts[20:22] + [3.0, 30.0]])
    out.append(s1[rng.permutation(4)])
    return out


def _forward_like_landmarks(raw_pts, asp, flip):
    """What landmarks.map_picture does to a mapped picture's points: y times the aspect, then (when flipped) x mirrored
    in 640 and each line's two ends traded."""
    p = np.array(raw_pts, np.float64) * np.array([1.0, asp])
    if flip:
        p = mirror_points(p, FRAME)
    return p


def selftest():
    rng = np.random.default_rng(0)
    base = _synthetic_lines()
    _selftest_frames(rng)
    _selftest_procrustes(rng, base)
    _selftest_truth(rng, base)
    _selftest_numbers()
    _selftest_verdict()
    _selftest_scored_once()
    _selftest_paired()
    _selftest_folds()
    print("selftest: all passed")


def _selftest_frames(rng):
    # 1. frame round trip: exported -> un-squeezed -> canvas -> back, exact; film likewise; mirror bookkeeping
    asp = 1.2461
    pts = rng.uniform(5, 635, (3, N_POINTS, 2))
    pts[:, :, 0] = np.sort(pts[:, :, 0].reshape(3, 11, 2), axis=2).reshape(3, N_POINTS)       # each line: left end first
    for mirrored in (False, True):
        fr = Frame((640, 640), asp, mirrored, (1996, 2487))
        v = fr.to_view(pts)
        for back in (fr.to_exported(fr.from_canvas(fr.to_canvas(v))), fr.to_exported(fr.from_film(fr.to_film(v)))):
            assert np.abs(back - pts).max() < 1e-3, "frame round trip, mirrored=%s" % mirrored
        if mirrored:
            ends, old = v.reshape(3, 11, 2, 2), pts.reshape(3, 11, 2, 2)
            assert (ends[:, :, 0, 0] <= ends[:, :, 1, 0]).all(), "a mirrored reading's lines keep their left end first"
            assert np.allclose(ends[:, :, 0, 0], FRAME - old[:, :, 1, 0]), "mirror: a line's left end is the old right end mirrored"
        assert fr.h_u == int(round(640 * asp))
    fr0 = Frame((640, 640), asp, False, (1996, 2487))
    c = fr0.to_canvas(np.array([[0.0, 0.0], [640.0, float(fr0.h_u)]]))
    assert 0 <= c[:, 0].min() and c[:, 0].max() <= CANVAS[1] and 0 <= c[:, 1].min() and c[:, 1].max() <= CANVAS[0], "the picture fits the canvas"
    from PIL import Image
    im = Image.fromarray(rng.integers(0, 255, (640, 640), dtype=np.uint8))
    for mirrored in (False, True):
        fr = Frame(im.size, asp, mirrored, (800, 997))
        cv, fl = prepare(im, fr)
        assert cv.shape == CANVAS and fl.shape == (997, 800)


def _selftest_procrustes(rng, base):
    # 2. the Procrustes chooser
    noise = lambda: rng.normal(0, 1.5, (N_POINTS, 2))  # noqa: E731
    ref = mean_shape(np.stack([base + noise() for _ in range(40)]))
    assert float(procrustes_distance(base, ref)) < 0.05, "the mean shape is near the shape it was made of"
    plain = base * 0.8 + [30.0, -12.0] + noise()
    mirrored_read = mirror_points(plain, 300.0)
    way, dp, dm = choose(plain, mirrored_read, ref)
    assert way == "plain" and dp < dm, (way, dp, dm)
    way2, dp2, dm2 = choose(mirrored_read, plain, ref)
    assert way2 == "mirrored" and dm2 < dp2, (way2, dp2, dm2)
    assert choose(plain, plain, ref)[0] == "plain", "a tie keeps the plain reading"
    turned = (base - base.mean(0)) @ np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]) * 2.0 + 500.0
    assert float(procrustes_distance(turned, base)) < 1e-9, "shift, turn and size do not count"


def _selftest_truth(rng, base):
    # 3. the truth inversion reproduces raw label points (a record built like landmarks' records), with NaN points
    for flip in (False, True):
        for fit_asp in (1.1, 1.6):
            raw_pts = base.copy()
            raw_pts[4:8] = np.nan                                              # L2 missing in this picture
            raw_pts[20:22] = np.nan                                            # and no S1 either
            raw = _raw_vertebrae(np.where(np.isnan(raw_pts), 0.0, raw_pts), rng)
            rec = {"points": _forward_like_landmarks(raw_pts, fit_asp, flip).astype(np.float32), "flip": flip, "aspect": fit_asp}
            # landmarks' records live in the un-stretched frame: raw labels there are the file's y times the aspect;
            # the file's own y is what raw holds, so the label the check sees is the file-frame raw
            truth = landmarks.file_frame(rec)
            worst = check_truth(truth, raw, "synthetic")
            assert worst < 1e-2, worst
            assert np.isnan(truth[4:8]).all() and np.isnan(truth[20:22]).all() and not np.isnan(truth[:4]).any()
            bad = truth.copy()
            bad[0, 0] += 2.0
            try:
                check_truth(bad, raw, "synthetic")
            except AssertionError:
                pass
            else:
                raise AssertionError("a truth point 2 px off every raw label point passed the check")


def _selftest_numbers():
    # 4. within 5 / within 15 / mean miss on hand-made numbers
    truth = np.zeros((2, N_POINTS, 2))
    pred = np.zeros((2, N_POINTS, 2))
    pred[0, :3, 0] = 3.0
    pred[0, 3:7, 0] = 10.0
    pred[0, 7, 1] = 20.0
    truth[1, :4] = np.nan                                                      # picture 1 lacks L1
    pred[1, 4:6, 1] = 6.0
    miss = miss_of(truth, pred)
    cnt = per_picture(miss)
    p = pooled(cnt)
    assert p["points"] == 40 and abs(p["within5"] - 82.5) < 1e-9 and abs(p["within15"] - 97.5) < 1e-9, p
    assert abs(p["mean_miss"] - (9 + 40 + 20 + 12) / 40.0) < 1e-9, p
    lo, hi = boot_ci(np.ones(5), np.ones(5), ["a", "b", "c", "d", "e"])
    assert lo == hi == 1.0
    num, den = np.array([8.0, 2.0, 3.0, 9.0]), np.array([10.0] * 4)
    ci1 = boot_ci(num, den, ["a", "a", "b", "c"])
    assert ci1 == boot_ci(num, den, ["a", "a", "b", "c"]) and 0.0 <= ci1[0] <= ci1[1] <= 1.0
    ci2 = boot_ci(num, den, ["a", "b", "c", "d"])
    assert ci1 != ci2, "pictures of one patient are resampled together"
    assert patient_of("2662-M-040Y1") == "2662" and patient_of("Case_12") == "Case_12"
    s = summarize(cnt, np.array(["p", "q"]))
    assert s["pictures"] == 2 and s["points"] == 40 and abs(s["within5"]["value"] - 82.5) < 1e-9


def _selftest_verdict():
    # 5. the verdict line at its edges
    head = "Against Bansal et al. 2026 (76.75 of 100 corners within 5 px): "
    assert verdict(76.76, 90.0) == head + "BEATEN"
    assert verdict(76.75, 90.0) == head + "LEVEL"
    assert verdict(70.0, 76.75) == head + "LEVEL"
    assert verdict(70.0, 76.74) == head + "BEHIND"
    assert verdict(70.0, 80.0) == head + "LEVEL"


def _selftest_scored_once():
    # 6. scored once
    with tempfile.TemporaryDirectory() as t:
        refuse_if_scored(t)
        with open(os.path.join(t, "result.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        try:
            refuse_if_scored(t)
        except SystemExit:
            pass
        else:
            raise AssertionError("a second scoring was allowed")


def _selftest_paired():
    # 7. the paired bootstrap (idea 119): chain A minus chain B, one draw of the patients for both, seeded
    ca = {"n": np.full(4, 10.0), "w5": np.array([10.0, 5.0, 10.0, 0.0]), "sum": np.array([10.0, 20.0, 10.0, 40.0])}
    cb = {"n": np.full(4, 10.0), "w5": np.full(4, 5.0), "sum": np.full(4, 30.0)}
    pats = ["a", "a", "b", "c"]
    d = paired(ca, cb, pats, "w5")
    assert abs(d["value"] - 12.5) < 1e-9, d                              # (25 - 20) of 40 points, in percent
    assert d["low"] <= d["value"] + 1e-9 and d["low"] < d["high"], d
    assert paired(ca, cb, pats, "w5") == d, "the paired bootstrap is not seeded"
    assert paired(ca, ca, pats, "w5") == {"value": 0.0, "low": 0.0, "high": 0.0}, "a chain against itself"
    flat = paired({"n": np.full(3, 10.0), "w5": np.full(3, 10.0), "sum": np.zeros(3)},
                  {"n": np.full(3, 10.0), "w5": np.zeros(3), "sum": np.zeros(3)}, ["a", "b", "c"], "w5")
    assert flat["low"] == flat["high"] == 100.0, flat                    # every patient +100: every resample is +100
    m = paired(ca, cb, pats, "sum", 1.0)
    assert abs(m["value"] - (80.0 / 40 - 120.0 / 40)) < 1e-9, m          # mean miss: A minus B, px


def _selftest_folds():
    # 8. --fold: the held-out set is fold K's pictures only (split_fold), and the folder is the new one
    tab = {"2662-F-040Y1": 0, "2662-M-040Y1": 0, "2700-F-040Y1": 1, "Case_1": 2, "Case_2": 0}
    recs_s = [{"base": b} for b in ["2662-F-040Y1", "2662-M-040Y1", "2700-F-040Y1", "Case_1", "Case_2"]]
    train_s, held_s = landmarks.split_fold(recs_s, tab, 0)
    assert [r["base"] for r in held_s] == ["2662-F-040Y1", "2662-M-040Y1", "Case_2"], "--fold 0 holds fold 0 only"
    assert len(train_s) + len(held_s) == len(recs_s) and not {r["base"] for r in train_s} & {r["base"] for r in held_s}
    assert out_dir(0, 14).endswith(os.path.join("spine_xray", "landmarks-fold0-v14")), out_dir(0, 14)
    assert out_dir().endswith(os.path.join("spine_xray", OUT_NAME)), "the no-option folder is unchanged"


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["score", "selftest"])
    ap.add_argument("--fold", type=int, choices=range(landmarks.N_FOLDS), help="idea 119: score this landmark fold's held-out pictures")
    ap.add_argument("--from-version", type=int, help="with --fold: the first-look corners version A (folder landmarks-fold<K>-v<A>)")
    ap.add_argument("--refine-version", type=int, help="with --fold: the refine corners version B")
    ap.add_argument("--baseline", action="store_true", help="with --fold: also read v7's chain on the same pictures and compare")
    a = ap.parse_args()
    if a.what == "selftest":
        os.environ["RAIDIOLOGY_DEVICE"] = "cpu"
        try:
            selftest()
        except AssertionError as e:
            print("selftest FAILED: %s" % e)
            return 1
        return 0
    if a.fold is None:
        if a.from_version is not None or a.refine_version is not None or a.baseline:
            raise SystemExit("--from-version, --refine-version and --baseline go with --fold")
        score()
        return 0
    if a.from_version is None or a.refine_version is None:
        raise SystemExit("--fold wants --from-version A and --refine-version B")
    score_fold(a.fold, a.from_version, a.refine_version, a.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
