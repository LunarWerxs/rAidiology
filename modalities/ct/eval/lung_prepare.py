"""Chest CT scans of LIDC-IDRI as compact 3D pictures a lung-nodule detector trains on.

A raw scan is a stack of 100 to 600 DICOM slices, each 512 x 512, with a pixel size of about 0.6 to
0.9 mm and a slice gap of 0.6 to 5 mm, stored as 16-bit numbers. A network cannot learn "how big is a
nodule" from that, because the same 6 mm nodule is 7 voxels wide in one scan and 9 in the next and
only 2 slices thick in the thin one and 1 in the thick one. And it is far too big to hold in memory.
So every scan is turned into one standard picture:

- 1 mm in all three directions (linear interpolation). Now a voxel is a millimetre in every scan, a
  nodule of 6 mm is 6 voxels in any scan, and the network's receptive field means the same thing
  everywhere. 1 mm is about as fine as the thickest slices of the good scans (2.5 mm) can honestly
  give: going finer would only invent detail, going coarser would blur the 3 to 4 mm nodules we are
  here to find. The slice positions come from each slice's ImagePositionPatient z, not from an
  assumed gap, so scans with uneven gaps are interpolated correctly.
- Only the lungs. The air inside the body (below -400 HU, not connected to the air outside the
  patient) is found, the two lungs (and the windpipe they join) are kept, small holes (vessels,
  nodules) are closed, and the picture is cropped to the box around them plus 10 mm. A nodule that
  matters is inside the lung, so nothing the detector must find is thrown away, and the arms, the
  table and the empty air around the patient (half the volume) are gone. A scan where this finds less
  than 0.5 litre of lung is reported and skipped, not guessed at (an adult's lungs hold 3 to 6 litres).
- 8 bits. Hounsfield units (HU: air -1000, water 0, soft tissue 40, bone 300 and up) are clipped to
  [-1200, 600] and stored as uint8, 0 = -1200 and 255 = 600. One step is 1800 / 255 = 7.06 HU. That
  is finer than the noise of a chest CT (10 to 30 HU) and keeps everything from air to dense bone,
  so nothing visible to a radiologist in the lung is lost, and a volume costs a quarter of float32.

Each volume is <WORK>/eval/lung/vol/<SeriesInstanceUID>.npy in axis order (z, y, x), with a
sidecar .json: origin_mm (the world x, y, z of voxel [0, 0, 0] of the crop, in the scanner's
millimetres, the same ones LUNA16's annotations.csv and lidc.py use), spacing_mm [1, 1, 1], shape,
the original pixel spacing, slice spacing and slice count, and the lung volume in litres.
`world_to_voxel` and `voxel_to_world` convert between the scanner's millimetres and voxel indices.

The air outside the body is found slice by slice as the dark region (air and scanner padding) holding
a corner of the picture, not in 3D (the lungs join the outside through the windpipe and neck) and not
by the stack's top and bottom (a scan that stops in the middle of the lungs must keep them). Air a
corner does not reach (under the table) is dropped as well: more than 1% of it within 3 px of the
picture's edge.

    python modalities/ct/eval/lung_prepare.py build [--jobs N] [--only-luna16] [--limit N] [--force] [--uids <file>]
    python modalities/ct/eval/lung_prepare.py check [--uids <file>]

Env RAIDIOLOGY_LUNG_DIR overrides the output folder (default <WORK>/eval/lung).
"""
import argparse
import csv
import glob
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import paths  # noqa: E402

HU_LO, HU_HI = -1200.0, 600.0
STEP_HU = (HU_HI - HU_LO) / 255.0
AIR_HU = -400.0
PADDING_HU = -1500.0        # below this is scanner padding (-2000, -3024, ...), never air
MARGIN_MM = 10
MIN_LITRES = 0.5


def out_dir():
    return os.environ.get("RAIDIOLOGY_LUNG_DIR") or os.path.join(paths.WORK, "eval", "lung")


def to_uint8(hu):
    return np.clip(np.rint((hu - HU_LO) / STEP_HU), 0, 255).astype(np.uint8)


def to_hu(u8):
    return u8.astype(np.float32) * STEP_HU + HU_LO


def world_to_voxel(meta, xyz_mm):
    """World (x, y, z) mm (one point or an N x 3 array) -> float voxel indices (z, y, x) of the crop."""
    p = np.asarray(xyz_mm, dtype=float)[..., ::-1]
    return (p - np.asarray(meta["origin_mm"], dtype=float)[::-1]) / np.asarray(meta["spacing_mm"], dtype=float)[::-1]


def voxel_to_world(meta, zyx):
    """Voxel indices (z, y, x) (one point or N x 3) -> world (x, y, z) mm."""
    p = np.asarray(zyx, dtype=float)
    return (p * np.asarray(meta["spacing_mm"], dtype=float)[::-1]
            + np.asarray(meta["origin_mm"], dtype=float)[::-1])[..., ::-1]


# ---- reading and resampling ------------------------------------------------------------------

def read_scan(folder):
    """(HU float32 volume (z, y, x) sorted by z, z of each slice, (row, col) pixel spacing, x0, y0)."""
    import pydicom
    items = []
    for p in glob.glob(os.path.join(folder, "*.dcm")):
        d = pydicom.dcmread(p)
        if not hasattr(d, "ImagePositionPatient") or not hasattr(d, "PixelData"):
            continue
        items.append((float(d.ImagePositionPatient[2]), d))
    if len(items) < 20:
        raise ValueError("only %d slices" % len(items))
    items.sort(key=lambda t: t[0])
    zs = np.array([t[0] for t in items])
    keep = np.r_[True, np.diff(zs) > 1e-3]          # a repeated z (duplicate slice) is read once
    items = [t for t, k in zip(items, keep) if k]
    zs = zs[keep]
    d0 = items[0][1]
    shape = d0.pixel_array.shape
    vol = np.empty((len(items),) + shape, dtype=np.float32)
    for i, (_, d) in enumerate(items):
        if d.pixel_array.shape != shape:
            raise ValueError("slices of different sizes")
        vol[i] = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) \
            + float(getattr(d, "RescaleIntercept", 0))
    sp = [float(v) for v in d0.PixelSpacing]
    x0, y0 = float(d0.ImagePositionPatient[0]), float(d0.ImagePositionPatient[1])
    # A prone scan (orientation -1 0 0 0 -1 0) has world x and y falling with column and row: flip so
    # the array's x and y rise with world x and y, and x0, y0 are the world position of its [., 0, 0].
    iop = [float(v) for v in getattr(d0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    if iop[0] < 0:
        vol = vol[:, :, ::-1]
        x0 -= (shape[1] - 1) * sp[1]
    if iop[4] < 0:
        vol = vol[:, ::-1, :]
        y0 -= (shape[0] - 1) * sp[0]
    return np.ascontiguousarray(vol), zs, sp, x0, y0


def _lerp_axis(t, ax, pos):
    """Linear interpolation of tensor t along axis ax at fractional indices pos (clamped to the ends)."""
    import torch
    n = t.shape[ax]
    pos = np.clip(pos, 0, n - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = torch.from_numpy((pos - i0).astype(np.float32))
    shp = [1] * t.dim()
    shp[ax] = -1
    w = w.reshape(shp)
    a = t.index_select(ax, torch.from_numpy(i0))
    b = t.index_select(ax, torch.from_numpy(i1))
    return a + (b - a) * w


def resample_1mm(vol, zs, sp):
    """Volume at 1 mm in every direction; voxel [0,0,0] sits at the first slice's x0, y0 and z = zs[0].
    In the slice plane scipy.ndimage.zoom(order=1); across slices a two-tap interpolation at each
    slice's true z (so uneven gaps are right). Timed on one 321-slice 512x512 scan, 0.56 mm pixels, to
    1 mm: scipy zoom 0.85 s, torch two-tap along x, y and z 2.7 s, so zoom is kept for the plane. zoom
    spaces its samples evenly between the first and last pixel, which is 1 mm to within 0.5 mm over the
    whole width."""
    from scipy import ndimage as ndi
    ny = int(round((vol.shape[1] - 1) * sp[0])) + 1
    nx = int(round((vol.shape[2] - 1) * sp[1])) + 1
    t = ndi.zoom(vol, (1, (ny - 1) / (vol.shape[1] - 1), (nx - 1) / (vol.shape[2] - 1)), order=1)
    import torch
    nz = int(np.floor(zs[-1] - zs[0])) + 1
    return _lerp_axis(torch.from_numpy(t), 0, np.interp(zs[0] + np.arange(nz), zs, np.arange(len(zs)))).numpy()


# ---- lungs -----------------------------------------------------------------------------------

def _morph(m, r, grow):
    """Cube dilation (grow) or erosion of radius r, as three 1D passes (a 7x7x7 pool took 40 s a scan)."""
    import torch
    t = torch.from_numpy(m)[None, None].float()
    t = t if grow else -t
    for k in ((2 * r + 1, 1, 1), (1, 2 * r + 1, 1), (1, 1, 2 * r + 1)):
        t = torch.nn.functional.max_pool3d(t, k, 1, tuple(v // 2 for v in k))
    return (t if grow else -t)[0, 0].numpy() > 0.5


def lung_mask(hu):
    """Boolean mask of the lungs at 1 mm (see the module docstring), and nothing else is promised of it."""
    from scipy import ndimage as ndi
    # Outside of the body, slice by slice: the air (and scanner padding, which is far below air) of
    # one slice that holds a corner of the picture. In 3D the lungs join the outside through the
    # windpipe and the neck, so the outside is found in 2D; a lung that only touches the picture's
    # edge (field of view cutting the chest) holds no corner and is kept.
    dark = hu < AIR_HU
    in_plane = np.zeros((3, 3, 3), bool)
    in_plane[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    lab2, n2 = ndi.label(dark, structure=in_plane)
    corners = np.unique(np.concatenate([lab2[:, 0, 0], lab2[:, 0, -1], lab2[:, -1, 0], lab2[:, -1, -1]]))
    outside = np.zeros(n2 + 1, bool)                  # lookup by label: no big temporaries
    outside[corners[corners > 0]] = True
    dark &= hu >= PADDING_HU
    dark &= ~outside[lab2]
    del lab2
    lab, n = ndi.label(dark)
    del dark
    if n == 0:
        return np.zeros(hu.shape, bool)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    # Outside air a corner does not reach (under the table, between table and arms) lies along the
    # picture's edge: 1 to 27% of such a component is within 3 px of the edge, 0.0% of a lung.
    in_frame = (np.bincount(lab[:, :3].ravel(), minlength=n + 1) + np.bincount(lab[:, -3:].ravel(), minlength=n + 1)
                + np.bincount(lab[:, 3:-3, :3].ravel(), minlength=n + 1)
                + np.bincount(lab[:, 3:-3, -3:].ravel(), minlength=n + 1))
    sizes[in_frame > 0.01 * np.maximum(sizes, 1)] = 0
    if sizes.max() == 0:
        return np.zeros(hu.shape, bool)
    # the two lungs (plus the windpipe they join) are the big ones; a bowel gas bubble is far smaller
    big = np.flatnonzero(sizes >= 0.15 * sizes.max())
    big = big[np.argsort(sizes[big])[::-1]][:3]
    keep = np.zeros(n + 1, bool)
    keep[big] = True
    m = keep[lab]
    del lab
    m = _morph(_morph(m, 3, True), 3, False)         # close small holes (vessels, nodules at the edge)
    for i in range(m.shape[0]):
        m[i] = ndi.binary_fill_holes(m[i])
    return m


def crop_to_lungs(hu):
    """(crop of hu, (z, y, x) offset of the crop, lung litres) or None when under 0.5 litre of lung."""
    m = lung_mask(hu)
    litres = float(m.sum()) / 1e6                      # 1 mm voxels: 1e6 of them are a litre
    if litres < MIN_LITRES:
        return None, litres
    idx = np.nonzero(m)
    lo = [max(int(i.min()) - MARGIN_MM, 0) for i in idx]
    hi = [min(int(i.max()) + MARGIN_MM + 1, s) for i, s in zip(idx, hu.shape)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return (hu[sl], lo), litres


# ---- one scan --------------------------------------------------------------------------------

def _atomic(path, write):
    write(path + ".part")
    os.replace(path + ".part", path)


def paths_for(odir, uid):
    return os.path.join(odir, "vol", uid + ".npy"), os.path.join(odir, "vol", uid + ".json")


def process(uid, patient, folder, odir, force=False):
    """-> ('done'|'exists'|'skipped'|'failed', message, bytes written)."""
    npy, js = paths_for(odir, uid)
    if os.path.exists(npy) and os.path.exists(js) and not force:
        return "exists", "", 0
    try:
        import torch
        vol, zs, sp, x0, y0 = read_scan(folder)
        n_slices = len(zs)
        gaps = np.diff(zs)
        hu = resample_1mm(vol, zs, sp)
        del vol
        got, litres = crop_to_lungs(hu)
        if got is None:
            return "skipped", "%.2f litre of lung" % litres, 0
        crop, lo = got
        u8 = to_uint8(crop)
        meta = {"series_uid": uid, "patient": patient, "origin_mm": [x0 + lo[2], y0 + lo[1], float(zs[0]) + lo[0]],
                "spacing_mm": [1.0, 1.0, 1.0], "shape": list(u8.shape),
                "original_pixel_spacing_mm": [sp[0], sp[1]],
                "original_slice_spacing_mm": float(np.median(gaps)), "original_slices": n_slices,
                "lung_litres": round(litres, 3)}
        os.makedirs(os.path.dirname(npy), exist_ok=True)

        def w_npy(p):
            with open(p, "wb") as f:
                np.save(f, u8)

        def w_js(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        _atomic(npy, w_npy)
        _atomic(js, w_js)
        return "done", "", u8.nbytes
    except Exception as e:  # noqa: BLE001  a bad scan must not stop the others
        return "failed", "%s: %s" % (type(e).__name__, e), 0


def _work(a):
    import torch
    torch.set_num_threads(1)
    r = process(*a)
    if r[0] == "failed" and "MemoryError" in r[1]:     # the shared box was briefly out of memory
        import gc
        gc.collect()
        time.sleep(20)
        r = process(*a)
    return (a[0],) + r


def load_meta(uid, odir=None):
    with open(paths_for(odir or out_dir(), uid)[1], encoding="utf-8") as f:
        return json.load(f)


# ---- scan lists ------------------------------------------------------------------------------

def luna16_subsets():
    p = os.path.join(paths.DATASETS, "public", "luna16", "subsets.csv")
    if not os.path.exists(p):
        return {}
    with open(p, newline="") as f:
        return {r["seriesuid"]: int(r["subset"]) for r in csv.DictReader(f)}


def complete_scans():
    import lidc
    ok, _ = lidc.unpacked()
    recs = lidc.series_records()
    return [(u, recs[u]["PatientID"], lidc.scan_dir(recs[u])) for u in sorted(ok)]


# ---- commands --------------------------------------------------------------------------------

def write_index(odir, luna=None):
    """index.csv of every volume in odir (not only the ones this run was given); returns its path."""
    luna = luna if luna is not None else luna16_subsets()
    rows = []
    for jp in sorted(glob.glob(os.path.join(odir, "vol", "*.json"))):
        with open(jp, encoding="utf-8") as f:
            m = json.load(f)
        rows.append([m["series_uid"], m["patient"], luna.get(m["series_uid"], -1),
                     "x".join(str(v) for v in m["shape"]), m["lung_litres"], m["original_slice_spacing_mm"]])
    idx = os.path.join(odir, "index.csv")
    os.makedirs(odir, exist_ok=True)

    def w_idx(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seriesuid", "patient", "luna16_subset", "shape", "lung_litres", "original_slice_spacing"])
            w.writerows(rows)
    _atomic(idx, w_idx)
    return idx


def build(jobs=None, only_luna16=False, limit=None, force=False, odir=None, scans=None):
    odir = odir or out_dir()
    luna = luna16_subsets()
    scans = scans if scans is not None else complete_scans()
    if only_luna16:
        scans = [s for s in scans if s[0] in luna]
    if limit:
        scans = scans[:limit]
    jobs = jobs or max(1, (os.cpu_count() or 4) // 4)
    t0 = time.time()
    counts = {"done": 0, "exists": 0, "skipped": 0, "failed": 0}
    nbytes, finished = 0, 0
    jobs_in = [(u, p, f, odir, force) for u, p, f in scans]
    if jobs == 1:
        results = (_work(a) for a in jobs_in)
        pool = None
    else:
        pool = ProcessPoolExecutor(jobs)
        results = pool.map(_work, jobs_in, chunksize=1)
    try:
        for uid, status, msg, nb in results:
            counts[status] += 1
            nbytes += nb
            finished += 1
            if status in ("skipped", "failed"):
                print("%s %s: %s" % (status.upper(), uid, msg), flush=True)
            if finished % 25 == 0:
                print("%d/%d scans: %d done, %d skipped, %d failed (%.0f s)" % (
                    finished, len(scans), counts["done"], counts["skipped"], counts["failed"], time.time() - t0),
                    flush=True)
    finally:
        if pool:
            pool.shutdown()
    idx = write_index(odir, luna)
    print("scans done %d (already there %d), skipped %d, failed %d; %.0f s; %.2f GB written; index %s" % (
        counts["done"], counts["exists"], counts["skipped"], counts["failed"], time.time() - t0, nbytes / 1e9, idx))
    return counts


def check():
    import pandas as pd
    ann = pd.read_csv(os.path.join(paths.DATASETS, "public", "luna16", "annotations.csv"))
    have = {s[0]: s for s in complete_scans()}
    pick = [u for u in sorted(ann.seriesuid.unique()) if u in have][:3]
    assert len(pick) == 3, "fewer than 3 LUNA16 scans with nodules are unpacked here"
    with tempfile.TemporaryDirectory(prefix="lung_check_") as tmp:
        os.environ["RAIDIOLOGY_LUNG_DIR"] = tmp
        t0 = time.time()
        counts = build(jobs=1, force=True, odir=tmp, scans=[have[u] for u in pick])
        secs = time.time() - t0
        assert counts["done"] == 3, "expected 3 scans built, got %s" % counts
        for u in pick:
            meta = load_meta(u, tmp)
            vol = np.load(paths_for(tmp, u)[0])
            assert list(vol.shape) == meta["shape"]
            for a in ann[ann.seriesuid == u].itertuples():
                v = world_to_voxel(meta, [a.coordX, a.coordY, a.coordZ])
                back = voxel_to_world(meta, v)
                assert np.allclose(back, [a.coordX, a.coordY, a.coordZ], atol=1e-6), "round trip"
                c = np.rint(v).astype(int)
                assert (c >= 1).all() and (c <= np.array(vol.shape) - 2).all(), \
                    "nodule of %s at voxel %s is outside the crop %s" % (u, c.tolist(), vol.shape)
                box = vol[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, c[2] - 1:c[2] + 2]
                hu = float(to_hu(box).mean())
                print("%s nodule %.1f mm at voxel %s: mean HU of the 3x3x3 at its centre %.0f" % (
                    u[-12:], a.diameter_mm, c.tolist(), hu))
                assert hu > -600, "nodule centre is %.0f HU, not brighter than -600" % hu
        print("%.1f seconds per scan (%d scans)" % (secs / 3, 3))
    print("check ok")


def check_uids(uid_file):
    """Build scans from uid_file, verify lungs >= MIN_LITRES, nodules map inside crop with HU > -600."""
    import pandas as pd
    with open(uid_file) as f:
        uids = [line.strip() for line in f if line.strip()]

    ann = pd.read_csv(os.path.join(paths.DATASETS, "public", "luna16", "annotations.csv"))
    have = {s[0]: s for s in complete_scans()}
    real_dir = out_dir()

    with tempfile.TemporaryDirectory(prefix="lung_check_uids_") as tmp:
        os.environ["RAIDIOLOGY_LUNG_DIR"] = tmp
        t0 = time.time()
        scans_to_build = [have[u] for u in uids if u in have]
        counts = build(jobs=1, force=True, odir=tmp, scans=scans_to_build)
        secs = time.time() - t0

        for u in uids:
            assert u in have and os.path.exists(paths_for(tmp, u)[0]), "scan %s was not built" % u

            meta = load_meta(u, tmp)
            vol = np.load(paths_for(tmp, u)[0])
            litres = meta["lung_litres"]

            print("%s: %.2f litres" % (u[-12:], litres))
            assert litres >= MIN_LITRES, "scan %s has only %.2f litres (need >= %.2f)" % (u[-12:], litres, MIN_LITRES)
            assert list(vol.shape) == meta["shape"]

            for a in ann[ann.seriesuid == u].itertuples():
                v = world_to_voxel(meta, [a.coordX, a.coordY, a.coordZ])
                c = np.rint(v).astype(int)
                assert (c >= 1).all() and (c <= np.array(vol.shape) - 2).all(), \
                    "nodule of %s at voxel %s is outside the crop %s" % (u[-12:], c.tolist(), vol.shape)
                box = vol[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, c[2] - 1:c[2] + 2]
                hu = float(to_hu(box).mean())
                print("  nodule %.1f mm: HU %.0f" % (a.diameter_mm, hu))
                assert hu > -600, "nodule centre is %.0f HU, not brighter than -600" % hu

        # scans built before this rule: how much does their lung volume move under it
        old = sorted(glob.glob(os.path.join(real_dir, "vol", "*.json")))
        changes = []
        for jp in old:
            with open(jp, encoding="utf-8") as f:
                m = json.load(f)
            if m["series_uid"] in uids or m["series_uid"] not in have:
                continue
            vol, zs, sp, _, _ = read_scan(have[m["series_uid"]][2])
            new = float(lung_mask(resample_1mm(vol, zs, sp)).sum()) / 1e6
            changes.append((m["series_uid"], m["lung_litres"], new))
            print("old %.2f L -> new %.2f L  %s" % (m["lung_litres"], new, m["series_uid"][-12:]))
            if len(changes) == 20:
                break
        assert len(changes) == 20, "fewer than 20 earlier scans to compare"
        worst = max(changes, key=lambda c: abs(c[2] - c[1]))
        print("largest change over 20 scans: %.2f L -> %.2f L (%s)" % (worst[1], worst[2], worst[0][-12:]))
        print("check ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--jobs", type=int)
    b.add_argument("--only-luna16", action="store_true")
    b.add_argument("--limit", type=int)
    b.add_argument("--force", action="store_true")
    b.add_argument("--uids", type=str, help="file with SeriesInstanceUIDs to build (one per line)")
    c = sub.add_parser("check")
    c.add_argument("--uids", type=str, help="file with SeriesInstanceUIDs to check (one per line)")
    a = ap.parse_args()
    if a.cmd == "build":
        if a.uids:
            with open(a.uids) as f:
                uids = [line.strip() for line in f if line.strip()]
            have = {s[0]: s for s in complete_scans()}
            scans = [have[u] for u in uids if u in have]
            build(a.jobs, False, None, a.force, scans=scans)
        else:
            build(a.jobs, a.only_luna16, a.limit, a.force)
    else:
        if a.uids:
            check_uids(a.uids)
        else:
            check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
