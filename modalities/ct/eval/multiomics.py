"""Multiomics nodule CT set: one outlined lung nodule per patient, cancer or benign, as 1 mm boxes.

2,032 patients, one chest CT each (a NIfTI in one of 36 zips), one 3D Slicer outline of the nodule
(Segmentation.seg.nrrd), and a clinical table with the age and the label (1 cancer, 0 benign). The model
in nodule_malignancy.py trains on 48 mm cubes cut by nodule_fpr.crop_of from "boxes" (a uint8 .npy
volume [z][y][x] at 1 mm and a .json with origin_mm and spacing_mm). This module turns each patient into
such a box, so the set trains the same model as LIDC and LUNGx do.

Licence: scans and labels are CC BY 4.0. Zhao et al., Scientific Data 2025, 12:545; the labels are on
Figshare, 10.6084/m9.figshare.26861077 (v2).

Per sample:
- The scan is read from its zip into memory (never written to disk) and parsed by nibabel. Its affine
  maps voxel indices to RAS millimetres; the boxes are in patient millimetres the DICOM way (LPS: x grows
  to the patient's left, y to the back, z up), so every world point here is LPS = diag(-1, -1, 1) x RAS,
  i.e. the LPS affine is diag(-1, -1, 1, 1) @ img.affine, and a point is mapped into the scan's voxel
  index with its inverse. Only the slab of voxels a request needs is read (the scan is uncompressed in
  memory, so slicing the proxy reads just that slab), HU = stored value x scl_slope + scl_inter.
- The outline is read by SimpleITK (physical coordinates, LPS). Its centroid is the mean index of the
  voxels above 0, taken to millimetres by TransformContinuousIndexToPhysicalPoint; diameter_mm is the
  equivalent sphere of the voxel count x voxel volume.
- The box is a 64 x 64 x 64 cube at 1 mm, axis-aligned in LPS, voxel [32][32][32] on the centroid:
  origin_mm = centroid - 32, so voxel [k][j][i] sits at (ox + i, oy + j, oz + k). It is the scan sampled
  by trilinear interpolation (scipy.ndimage.map_coordinates, order 1); outside the scan it is 0 (uint8).
  Converted with lung_prepare.to_uint8 (HU window -1200..600).
- mask_hu is the mean HU of the outline, shell_hu the mean HU of a 3 mm shell just outside it. Both are
  measured on the outline's NATIVE grid (not a 1 mm one, so a 3 mm nodule is not lost to the 1 mm
  sampling): the mask's voxels, and the voxels within 3 mm of the mask (anisotropic Euclidean distance
  with the mask's own spacing, mask excluded), each sampled from the scan at its LPS centre by
  trilinear interpolation; points outside the scan are left out.
- slice_mm is the coarser of the scan's slice spacing and the outline's (a few outlines were drawn on a
  5 mm reconstruction of a thin-slice scan, and are only that precise).
- used = 4 <= diameter_mm < 30 and slice_mm <= 3.0 and mask_hu - shell_hu > 100 (a real nodule is much
  denser than the lung around it; a lower step means the outline is misplaced). `why` names every rule
  that failed ('' when used). A box is written for every sample that could be read, used or not.

Output (default <WORK>/eval/lung/vol_multiomics): vol/mo-<SAMPLE>.npy and .json (series_uid and patient
are 'mo-<SAMPLE>'), rows/<SAMPLE>.json (one finished sample; a rerun skips it, a failed one is retried),
zips.csv (sample, zip, member, size) and labels.csv (rebuilt whole from the row files after each
prepare: seriesuid, patient, sample, coordX/Y/Z = the LPS centroid, diameter_mm, label, age, slice_mm,
pixel_mm, mask_hu, shell_hu, mask_voxels, used, why).

    python modalities/ct/eval/multiomics.py index [--archive DIR] [--out DIR]
    python modalities/ct/eval/multiomics.py prepare [--out DIR] [--labels DIR] [--archive DIR] [--jobs 2] [--samples S ...] [--limit N] [--force]
    python modalities/ct/eval/multiomics.py check [--out DIR]
    python modalities/ct/eval/multiomics.py selftest

Env RAIDIOLOGY_MULTIOMICS overrides the archive folder (default E:/rAidiology/archive/multiomics-nodule-ct,
read only); the outlines and clinical_data.csv are in <WORK>/ext/multiomics-labels (--labels).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import paths  # noqa: E402
import lung_prepare  # noqa: E402

CUBE = 64
SHELL_MM = 3.0
D_MIN, D_MAX = 4.0, 30.0            # equivalent-sphere diameter of the outline, mm: 4 <= d < 30
SLICE_MAX_MM = 3.0
STEP_MIN_HU = 100.0                 # mask_hu - shell_hu must be above this
ARCHIVE_DEFAULT = "E:/rAidiology/archive/multiomics-nodule-ct"
ROW_FIELDS = ["seriesuid", "patient", "sample", "coordX", "coordY", "coordZ", "diameter_mm", "label", "age",
              "slice_mm", "pixel_mm", "mask_hu", "shell_hu", "mask_voxels", "used", "why"]


def out_dir():
    return os.path.join(paths.WORK, "eval", "lung", "vol_multiomics")


def archive_dir():
    return os.environ.get("RAIDIOLOGY_MULTIOMICS") or ARCHIVE_DEFAULT


def labels_dir():
    return os.path.join(paths.WORK, "ext", "multiomics-labels")


def seg_path(ldir, sample):
    return os.path.join(ldir, "Label", sample, "Segmentation.seg.nrrd")


def _sample_name(s):
    s = str(s).strip()
    return "SAMPLE_%04d" % int(s) if s.isdigit() else s


def read_clinical(ldir):
    """{sample: (label 0/1 or None, age as text)} from clinical_data.csv (SAMPLE_N, Age, Label)."""
    out = {}
    with open(os.path.join(ldir, "clinical_data.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                lab = int(float(r["Label"]))
            except (TypeError, ValueError):
                lab = None
            out[_sample_name(r["SAMPLE_N"])] = (lab, (r.get("Age") or "").strip())
    return out


# ---- the scan: geometry and sampling ----------------------------------------------------------

def lps_affine(img):
    """Voxel index (i, j, k) -> LPS mm: the NIfTI's RAS affine with x and y negated."""
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(img.affine, dtype=float)


def scan_spacing(img):
    """(slice_mm, pixel_mm, [x, y, z spacing]) of the scan: each world axis takes the voxel axis it runs along."""
    a = lps_affine(img)
    norms = np.linalg.norm(a[:3, :3], axis=0)
    ax = [int(np.argmax(np.abs(a[r, :3]))) for r in range(3)]
    if sorted(ax) != [0, 1, 2]:
        ax = [0, 1, 2]
    sp = [float(norms[i]) for i in ax]
    return sp[2], (sp[0] + sp[1]) / 2.0, sp


def sample_hu(img, xyz, fill):
    """HU (float32) of the scan at LPS points xyz (3 x N), trilinear; `fill` where a point is outside the
    scan. Only the slab of voxels the points touch is read."""
    from scipy import ndimage as ndi
    inv = np.linalg.inv(lps_affine(img))
    ijk = inv[:3, :3] @ np.asarray(xyz, dtype=float) + inv[:3, 3:4]
    shape = img.shape[:3]
    lo = np.maximum(np.floor(ijk.min(axis=1)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(ijk.max(axis=1)).astype(int) + 2, shape)
    if (hi <= lo).any():
        return np.full(ijk.shape[1], fill, np.float32)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    if len(img.shape) == 4:
        sl += (0,)
    slab = np.asarray(img.dataobj[sl], dtype=np.float32)
    return ndi.map_coordinates(slab, ijk - lo[:, None], order=1, mode="constant", cval=fill).astype(np.float32)


def cube_of(img, centre):
    """(origin_mm, uint8 cube [k][j][i]) of 64 mm at 1 mm, voxel [32][32][32] on `centre` (LPS mm)."""
    o = np.asarray(centre, dtype=float) - CUBE // 2
    k, j, i = np.meshgrid(*[np.arange(CUBE)] * 3, indexing="ij")
    xyz = np.stack([o[0] + i.ravel(), o[1] + j.ravel(), o[2] + k.ravel()])
    hu = sample_hu(img, xyz, lung_prepare.HU_LO).reshape((CUBE,) * 3)
    return o, lung_prepare.to_uint8(hu)


# ---- the outline ------------------------------------------------------------------------------

def outline_of(seg):
    """The outline (a SimpleITK image) as a dict: centroid (LPS mm), n (voxels), diameter_mm, and the LPS mm
    centres (3 x N) of its voxels (mask_pts) and of the shell of SHELL_MM around it (shell_pts)."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    arr = sitk.GetArrayFromImage(seg)                  # [z][y][x], or [z][y][x][component]
    if arr.ndim == 4:
        arr = arr.any(axis=-1)
    m = arr > 0
    n = int(m.sum())
    if n == 0:
        raise ValueError("empty outline")
    sp = np.asarray(seg.GetSpacing(), dtype=float)     # x, y, z
    origin = np.asarray(seg.GetOrigin(), dtype=float)
    direction = np.asarray(seg.GetDirection(), dtype=float).reshape(3, 3)
    zyx = np.argwhere(m)
    centroid = np.asarray(seg.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in zyx[:, ::-1].mean(axis=0))))

    def physical(zyx_idx):                             # (N, 3) indices z, y, x -> (3, N) LPS mm
        return (origin[:, None] + direction @ (zyx_idx[:, ::-1].T * sp[:, None])).astype(float)

    pad = int(np.ceil(SHELL_MM / sp.min())) + 1
    lo = np.maximum(zyx.min(axis=0) - pad, 0)
    hi = np.minimum(zyx.max(axis=0) + pad + 1, m.shape)
    sub = m[tuple(slice(a, b) for a, b in zip(lo, hi))]
    dist = ndi.distance_transform_edt(~sub, sampling=tuple(sp[::-1]))
    shell = np.argwhere((dist > 0) & (dist <= SHELL_MM)) + lo
    return {"centroid": centroid, "n": n, "diameter_mm": float((6.0 * n * float(np.prod(sp)) / np.pi) ** (1.0 / 3.0)),
            "mask_pts": physical(zyx), "shell_pts": physical(shell), "voxel_mm3": float(np.prod(sp)),
            "slice_mm": float(np.abs(direction * sp[None, :])[2].max()), "spacing_mm": [float(v) for v in sp]}


def _mean(v):
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def geometry(img, seg):
    """Everything one sample needs from its scan (a nibabel image) and outline (a SimpleITK image):
    (origin_mm, uint8 cube, stats dict). Used by process() and by selftest."""
    o = outline_of(seg)
    origin, cube = cube_of(img, o["centroid"])
    scan_slice_mm, pixel_mm, orig_sp = scan_spacing(img)
    # Some outlines were drawn on a thicker reconstruction than the scan in the zip (SAMPLE_2032: a 0.7 mm
    # scan, its outline on 5 mm slices); the centroid and diameter are only as good as the coarser of the two.
    slice_mm = max(scan_slice_mm, o["slice_mm"])
    mask_hu = _mean(sample_hu(img, o["mask_pts"], np.nan))
    shell_hu = _mean(sample_hu(img, o["shell_pts"], np.nan)) if o["shell_pts"].shape[1] else float("nan")
    return origin, cube, {"centroid": o["centroid"], "n": o["n"], "diameter_mm": o["diameter_mm"],
                          "slice_mm": slice_mm, "pixel_mm": pixel_mm, "original_spacing_mm": orig_sp,
                          "outline_spacing_mm": o["spacing_mm"],
                          "mask_hu": mask_hu, "shell_hu": shell_hu}


def why_not(label, st):
    """The rules a sample fails, as text ('' when it is used)."""
    why = []
    d = st["diameter_mm"]
    if not D_MIN <= d < D_MAX:
        why.append("diameter %.1f mm outside %g-%g" % (d, D_MIN, D_MAX))
    if not st["slice_mm"] <= SLICE_MAX_MM:
        why.append("slice %.2f mm over %g" % (st["slice_mm"], SLICE_MAX_MM))
    step = st["mask_hu"] - st["shell_hu"]
    if not step > STEP_MIN_HU:
        why.append("contrast %.0f HU not over %g" % (step, STEP_MIN_HU) if step == step else "contrast not measurable")
    if label is None:
        why.append("no label")
    return "; ".join(why)


# ---- one sample -------------------------------------------------------------------------------

_ZIP = [None, None]                                    # (path, ZipFile) kept open by each worker


def read_member(zip_path, member):
    """The scan as a nibabel image, parsed from the zip's bytes (nothing is written to disk)."""
    import nibabel as nib
    if _ZIP[0] != zip_path:
        if _ZIP[1] is not None:
            _ZIP[1].close()
        _ZIP[:] = [zip_path, zipfile.ZipFile(zip_path)]
    return nib.Nifti1Image.from_bytes(_ZIP[1].read(_ZIP[1].getinfo(member)))


def row_path(odir, sample):
    return os.path.join(odir, "rows", sample + ".json")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def w(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    lung_prepare._atomic(path, w)


def _blank_row(sample, label, age):
    return {"seriesuid": "mo-" + sample, "patient": "mo-" + sample, "sample": sample, "coordX": None,
            "coordY": None, "coordZ": None, "diameter_mm": None, "label": label, "age": age, "slice_mm": None,
            "pixel_mm": None, "mask_hu": None, "shell_hu": None, "mask_voxels": None, "used": False, "why": ""}


def process(task, odir, ldir):
    """task = dict(sample, zip, member, label, age) -> (sample, 'done'|'failed', message)."""
    sample, label, age = task["sample"], task["label"], task["age"]
    try:
        import SimpleITK as sitk
        sp = seg_path(ldir, sample)
        if not os.path.exists(sp):
            raise FileNotFoundError("no outline")
        seg = sitk.ReadImage(sp)
        img = read_member(task["zip"], task["member"])
        origin, cube, st = geometry(img, seg)
        del img
        uid = "mo-" + sample
        npy, js = lung_prepare.paths_for(odir, uid)
        os.makedirs(os.path.dirname(npy), exist_ok=True)
        meta = {"series_uid": uid, "patient": uid, "sample": sample, "origin_mm": [float(v) for v in origin],
                "spacing_mm": [1.0, 1.0, 1.0], "shape": [CUBE] * 3,
                "original_spacing_mm": st["original_spacing_mm"],
                "outline_spacing_mm": st["outline_spacing_mm"], "diameter_mm": st["diameter_mm"], "label": label}

        def w_npy(p):
            with open(p, "wb") as f:
                np.save(f, cube)
        lung_prepare._atomic(npy, w_npy)
        _write_json(js, meta)
        why = why_not(label, st)
        row = _blank_row(sample, label, age)
        row.update({"coordX": float(st["centroid"][0]), "coordY": float(st["centroid"][1]),
                    "coordZ": float(st["centroid"][2]), "diameter_mm": st["diameter_mm"],
                    "slice_mm": st["slice_mm"], "pixel_mm": st["pixel_mm"], "mask_hu": st["mask_hu"],
                    "shell_hu": st["shell_hu"], "mask_voxels": st["n"], "used": not why, "why": why})
        _write_json(row_path(odir, sample), row)
        return sample, "done", why
    except Exception as e:  # noqa: BLE001  a bad scan must not stop the others
        msg = "%s: %s" % (type(e).__name__, e)
        row = _blank_row(sample, label, age)
        row.update({"why": "failed " + msg, "failed": True})
        try:
            _write_json(row_path(odir, sample), row)
        except OSError:
            pass
        return sample, "failed", msg


def _work(a):
    return process(*a)


# ---- commands ---------------------------------------------------------------------------------

def _basename(member):
    return re.split(r"[\\/]", member)[-1]


def index(archive, odir, ldir):
    """List every zip's members from the central directory; write zips.csv; print the oddities."""
    zips = sorted(glob.glob(os.path.join(archive, "*.zip")) + glob.glob(os.path.join(archive, "*", "*.zip")))
    if not zips:
        raise SystemExit("no zip under %s" % archive)
    rows = []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                b = _basename(info.filename)
                if info.is_dir() or not b.lower().endswith(".nii"):
                    continue
                rows.append([b[:-4], os.path.abspath(zp), info.filename, info.file_size])
    os.makedirs(odir, exist_ok=True)

    def w(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["sample", "zip", "member", "size"])
            wr.writerows(rows)
    lung_prepare._atomic(os.path.join(odir, "zips.csv"), w)
    where = {}
    for s, zp, _, _ in rows:
        where.setdefault(s, set()).add(zp)
    twice = sorted(s for s, v in where.items() if len(v) > 1)
    print("%d zips, %d scans, %d distinct samples; zips.csv in %s" % (len(zips), len(rows), len(where), odir))
    if twice:
        print("in two zips (%d): %s" % (len(twice), " ".join(twice[:20])))
    clin = read_clinical(ldir) if os.path.exists(os.path.join(ldir, "clinical_data.csv")) else None
    no_seg = sorted(s for s in where if not os.path.exists(seg_path(ldir, s)))
    print("without an outline (%d): %s" % (len(no_seg), " ".join(no_seg[:20])))
    if clin is not None:
        no_lab = sorted(s for s in where if clin.get(s, (None,))[0] is None)
        print("without a label (%d): %s" % (len(no_lab), " ".join(no_lab[:20])))
        print("labelled but with no scan: %d" % len([s for s in clin if s not in where]))
    else:
        print("no clinical_data.csv under %s" % ldir)
    return rows


def read_rows(odir):
    rows = []
    for p in sorted(glob.glob(os.path.join(odir, "rows", "*.json"))):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def write_labels(odir):
    rows = read_rows(odir)
    path = os.path.join(odir, "labels.csv")

    def w(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, ROW_FIELDS, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
    lung_prepare._atomic(path, w)
    return rows, path


def prepare(odir, ldir, archive, jobs=2, samples=None, limit=None, force=False):
    zp = os.path.join(odir, "zips.csv")
    if not os.path.exists(zp):
        index(archive, odir, ldir)
    with open(zp, newline="", encoding="utf-8") as f:
        todo = list(csv.DictReader(f))                 # zip by zip, in member order: the archive is a spinning disk
    seen, uniq = set(), []
    for r in todo:
        if r["sample"] not in seen:                    # a sample in two zips is read from the first
            seen.add(r["sample"])
            uniq.append(r)
    if samples:
        want = {_sample_name(s) for s in samples}
        uniq = [r for r in uniq if r["sample"] in want]
    clin = read_clinical(ldir)
    tasks, existing = [], 0
    for r in uniq:
        p = row_path(odir, r["sample"])
        if os.path.exists(p) and not force:
            with open(p, encoding="utf-8") as f:
                if not json.load(f).get("failed"):     # a failed one is tried again
                    existing += 1
                    continue
        lab, age = clin.get(r["sample"], (None, ""))
        tasks.append(({"sample": r["sample"], "zip": r["zip"], "member": r["member"], "label": lab, "age": age},
                      odir, ldir))
    if limit:
        tasks = tasks[:limit]
    t0 = time.time()
    done = failed = 0
    pool = None
    try:
        if jobs <= 1:
            results = map(_work, tasks)
        else:
            pool = ProcessPoolExecutor(jobs)
            results = pool.map(_work, tasks, chunksize=1)
        for n, (sample, status, msg) in enumerate(results, 1):
            if status == "done":
                done += 1
            else:
                failed += 1
                print("FAILED %s: %s" % (sample, msg), flush=True)
            if n % 10 == 0 or n == len(tasks):
                print("%d/%d samples: %d done, %d failed (%.0f s)" % (n, len(tasks), done, failed, time.time() - t0),
                      flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
        rows, path = write_labels(odir)
    used = [r for r in rows if r.get("used")]
    print("done %d, skipped as existing %d, failed %d; %.0f s; labels.csv (%d rows) %s" % (
        done, existing, failed, time.time() - t0, len(rows), path))
    print("used %d: %d cancers, %d benign" % (len(used), sum(r["label"] == 1 for r in used),
                                              sum(r["label"] == 0 for r in used)))
    return done, existing, failed


def check(odir):
    p = os.path.join(odir, "labels.csv")
    if not os.path.exists(p):
        print("no labels.csv in %s: run prepare first" % odir)
        return 1
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    used = [r for r in rows if r["used"] == "True"]
    on_disk = glob.glob(os.path.join(odir, "vol", "mo-*.npy"))
    bad = []
    for r in used:
        npy, js = lung_prepare.paths_for(odir, r["seriesuid"])
        if not (os.path.exists(npy) and os.path.exists(js)):
            bad.append("%s: box missing" % r["sample"])
        elif tuple(np.load(npy, mmap_mode="r").shape) != (CUBE,) * 3:
            bad.append("%s: box is not %d^3" % (r["sample"], CUBE))
    failed = [r for r in rows if r["why"].startswith("failed")]
    print("rows %d, boxes on disk %d, used %d (%d cancers, %d benign), failed %d" % (
        len(rows), len(on_disk), len(used), sum(r["label"] == "1" for r in used),
        sum(r["label"] == "0" for r in used), len(failed)))
    for b in bad[:20]:
        print("BAD", b)
    print("check ok" if not bad else "check FAILED: %d used rows without a good box" % len(bad))
    return 1 if bad else 0


def selftest():
    """A synthetic scan (flipped axes, 0.7 x 0.7 x 1.25 mm) with a bright sphere at a known LPS point, three
    markers on its +x, +y and +z sides, and a matching outline on its own grid."""
    import nibabel as nib
    import SimpleITK as sitk
    ras = np.array([[-0.7, 0, 0, 60.0], [0, 0.7, 0, -30.0], [0, 0, -1.25, 40.0], [0, 0, 0, 1.0]])
    shape = (80, 90, 70)
    P = np.array([-30.0, 0.0, 10.0])
    markers = [(P + [12, 0, 0], 3.0), (P + [0, 10, 0], 3.0), (P + [0, 0, 8], 3.0)]
    ijk = np.stack([a.ravel() for a in np.indices(shape)] + [np.ones(int(np.prod(shape)))])
    lps = (np.diag([-1.0, -1.0, 1.0, 1.0]) @ ras @ ijk)[:3]
    data = np.full(lps.shape[1], -800, np.int16)
    for c, r in [(P, 4.0)] + markers:
        data[np.linalg.norm(lps - np.asarray(c)[:, None], axis=0) <= r] = 50
    nii = nib.Nifti1Image(data.reshape(shape), ras)
    nii.header.set_data_dtype(np.int16)
    img = nib.Nifti1Image.from_bytes(nii.to_bytes())

    sp = (0.7, 0.7, 1.25)
    size = (40, 40, 20)
    org = P - [14, 14, 12]
    zyx = np.indices(size[::-1])                       # z, y, x index grids
    pos = [org[a] + zyx[2 - a] * sp[a] for a in range(3)]
    mask = (np.sqrt(sum((pos[a] - P[a]) ** 2 for a in range(3))) <= 4.0).astype(np.uint8)
    seg = sitk.GetImageFromArray(mask)
    seg.SetSpacing(sp)
    seg.SetOrigin(tuple(float(v) for v in org))

    origin, cube, st = geometry(img, seg)
    err = float(np.linalg.norm(st["centroid"] - P))
    assert err < 0.5, "centroid %s is %.2f mm from %s" % (st["centroid"], err, P)
    assert np.allclose(origin, st["centroid"] - CUBE // 2), "origin is not centroid - 32"
    hu = lung_prepare.to_hu(cube)
    assert hu[32, 32, 32] > 0, "the cube's centre is %.0f HU, not bright" % hu[32, 32, 32]
    assert hu[0, 0, 0] < -600, "the cube's corner is %.0f HU, not air" % hu[0, 0, 0]
    for axis, off in ((2, 12), (1, 10), (0, 8)):       # the cube's axes are z, y, x: LPS x is the last one
        for sign, bright in ((1, True), (-1, False)):
            idx = [32, 32, 32]
            idx[axis] += sign * off
            v = hu[tuple(idx)]
            assert (v > -100) if bright else (v < -600), "marker on axis %d side %+d is %.0f HU" % (axis, sign, v)
    assert abs(st["slice_mm"] - 1.25) < 1e-6 and abs(st["pixel_mm"] - 0.7) < 1e-6, (st["slice_mm"], st["pixel_mm"])
    assert 7.0 < st["diameter_mm"] < 9.0, st["diameter_mm"]
    assert st["mask_hu"] - st["shell_hu"] > 600, (st["mask_hu"], st["shell_hu"])
    assert why_not(1, st) == "", why_not(1, st)
    assert "diameter" in why_not(1, dict(st, diameter_mm=61.0)) and "slice" in why_not(0, dict(st, slice_mm=5.0))
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "prepare", "check"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=out_dir())
        if name != "check":
            s.add_argument("--archive", default=archive_dir())
            s.add_argument("--labels", default=labels_dir())
    pp = sub.choices["prepare"]
    pp.add_argument("--jobs", type=int, default=2)
    pp.add_argument("--samples", nargs="+")
    pp.add_argument("--limit", type=int)
    pp.add_argument("--force", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "index":
        index(a.archive, a.out, a.labels)
    elif a.cmd == "prepare":
        prepare(a.out, a.labels, a.archive, a.jobs, a.samples, a.limit, a.force)
    elif a.cmd == "check":
        return check(a.out)
    else:
        selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
