"""Training copies of big public scan collections: what our models read, at a small fraction of the size (owner,
2026-10-05: store it "significantly, borderline 90% smaller" and still train on it).

Each series comes from IDC's public AWS bucket, every object checked by its md5. It is turned into its training copy
at once and its DICOM files are deleted, so a collection many times the free space passes through a few GB:
- **CT:** each axis finer than 1.5 mm resampled to 1.5 mm (linear); an axis already coarser (2.5 mm or 5 mm slices)
  keeps its own spacing, since resampling it would only add interpolated slices. Hounsfield units as int16 clipped
  to [-1024, 3071], one NIfTI (.nii.gz) per evenly spaced stack (a series holding two passes gives two). This is
  LOSSY: detail finer than 1.5 mm is gone. The organ segmenter reads 3 mm (modalities/ct/README.md), so it loses
  nothing it uses; the lung-nodule models read 1 mm (modalities/ct/eval/lung_prepare.py), finer than this copy, so for
  them the copy is coarser than their own input (checked 2026-10-09; an earlier note here said nothing any CT model
  uses is lost). They train on full-resolution sets (LIDC, LIDC-annot-NLST501, nlst-labelled). Any original can be
  fetched again exactly from the .json beside its copy. Measured: 14 to 17 times smaller than the DICOM for thin-slice
  CT. `--exact CT` keeps CT as served instead (JPEG-LS lossless, as MG below): bar 139 (2026-10-10) found lung CT
  "needs exact", the second-look nodule model reading LUNA16's 1.5 mm round trip at 0.7720 against 0.8066.
- **CR, DX (radiographs):** the longest side brought down to 1024 px with Lanczos, the stored values kept as 16-bit
  PNG. The X-ray readers train at 384 px.
- **MG (mammograms): exact.** The DICOM as served, re-encoded as JPEG-LS lossless (`core/store.py`, every pixel
  checked bit for bit), since 2026-10-09: idea 128's crop reader reads the originals' full detail, which a 2048 px
  copy had cut. Lossless already makes a mammogram 9-19 times smaller.
- The labels of the studies kept (SEG, RTSTRUCT) are kept as served: they hold physical coordinates, so they lay
  onto the copy's grid. Raw data (LDCT's projections), screenshots and any other object (SR, MR...) are left out,
  unless `--keep-other`, which keeps them as served.
- Where TCIA restricts a collection's head images, every series that is or may include the head (by body part or
  description: head, brain, neck, face, whole body...) is left out, whatever IDC's own licence field says.
Beside each copy, a .json holds the header fields a model or a split needs (patient, study, series, date, age, sex,
maker, kernel, slice thickness, the original grid) and where the original is (bucket, folder, each object's md5),
so the original can be fetched again at any time. A series that cannot be converted is logged and skipped.

The collection's licence is read at TCIA, its own website, every run (each download's data_license); a series
whose IDC licence is not an allowed one is never fetched. Copies go to `<working disk>/<name>/<patient>/<study>/`
(an SSD, or the hard disk H: once both are short; `--root` sends a collection not needed soon straight to H:),
joined at `<Datasets>/public/<name>`; `E:/rAidiology/archive/<name>/train.json` records every series done, so a
run cut short goes on where it stopped.

Usage:
  python tools/train_copy.py plan  <collection> [--modality CT,CR,DX] [--per-study N] [--no-labels]
  python tools/train_copy.py fetch <collection> [--modality CT,CR,DX] [--per-study N] [--no-labels] [--limit N] [--root DIR]
                                    [--log FILE]
"""
import argparse
import concurrent.futures
import datetime
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time

# numpy's OpenBLAS sets aside buffers for every core when it is imported: 0.8 GB of committed memory per process
# on this 32-core PC, 0.1 GB with one thread (measured 2026-10-06). Nothing here does matrix maths.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import fetch, paths  # noqa: E402
from core.fetch import log  # noqa: E402

ARCHIVE = os.environ.get("RAIDIOLOGY_ARCHIVE", "E:/rAidiology/archive")
WORKING = os.environ.get("RAIDIOLOGY_WORKING", "G:/rAidiology/cold;F:/rAidiology/Datasets/public;"
                         "E:/rAidiology/Datasets/public;D:/rAidiology/Datasets/public;H:/rAidiology/cold;"
                         "O:/rAidiology/cold").split(";")
#                          the spare disk G: first, kept for scan training data (owner, 2026-10-06; `fetch_public.py
#                          warm --to F:` brings a copy to an SSD for a run that needs it fast), then the SSDs, the
#                          hard disk and the Plex server's share; each only down to its fetch.floor
LINKS = os.path.join(paths.DATASETS, "public")
LICENSING = os.path.join(paths.ROOT, "LICENSING.md")
TCIA_PAGE = "https://www.cancerimagingarchive.net/collection/"
ALLOWED = {"cc-by-4.0", "cc-by-3.0", "cc0", "cc-by-sa-4.0", "cc-by-sa-3.0"}
CT_MM = 1.5
HU = (-1024, 3071)
LONG_SIDE = {"CR": 1024, "DX": 1024}
EXACT = {"MG"}                 # kept as served, re-encoded losslessly (core/store.py, bit for bit): the modelling session,
#                                2026-10-09, idea 128's crop reader cuts 512 px crops at 0.0941 mm a pixel from the
#                                originals, about 3,100 px on the long side; a 2048 px copy is too coarse. JPEG-LS
#                                lossless already makes a mammogram 9-19 times smaller (docs/research/data-hunt-2026-10-05.md)
SERIES_AT_ONCE = 8             # series downloading and converting side by side
OBJECTS_AT_ONCE = 8            # each series' objects at once (8 x 8 = IDC's cap of 64 connections)
SHRINK_PROCS = max(2, (os.cpu_count() or 8) // 4)   # processes re-encoding an exact copy's images: the series threads share one interpreter
#                                lock, and the lossless re-encode and its check are mostly Python (2026-10-10, nlst-exact:
#                                8 threads used 1.9 cores, 11 CT slices a second, with 10 cores idle)
_POOL, _POOL_LOCK = None, threading.Lock()


def shrink_pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = concurrent.futures.ProcessPoolExecutor(max_workers=SHRINK_PROCS)
        return _POOL
MIN_CT_SLICES = 20             # fewer is a scout or a single reformatted slab
IMAGE_CLASSES = {"CT Image Storage", "Enhanced CT Image Storage", "Computed Radiography Image Storage",
                 "Digital X-Ray Image Storage - For Presentation", "Digital X-Ray Image Storage - For Processing",
                 "Digital Mammography X-Ray Image Storage - For Presentation",
                 "Digital Mammography X-Ray Image Storage - For Processing"}   # not raw projections or screenshots
LABELS = {"SEG", "RTSTRUCT"}   # kept as served, beside the copies of their study (they hold physical coordinates)
HEAD_WORDS = r"head|brain|neck|skull|face|sinus|orbit|whole.?body|\bwb\b|vertex|\bhn\b|h&n"
RESERVE = 60e9                 # left free on the working disk
HEADER = ["PatientID", "PatientAge", "PatientSex", "StudyDate", "StudyInstanceUID", "SeriesInstanceUID", "Modality",
          "SeriesDescription", "BodyPartExamined", "Manufacturer", "ManufacturerModelName", "ConvolutionKernel",
          "SliceThickness", "KVP", "ContrastBolusAgent", "ViewPosition", "PhotometricInterpretation", "ImageType"]


class Unusable(ValueError):
    """A series whose pictures cannot make a training copy (no evenly spaced stack, an odd pixel array): a property
    of the data, not of the network. Kept in train.json's `unusable` with the reason, not fetched again, and counted
    with the copied ones when a copy is judged whole (NLST, 2026-10-07: 14 of 72,947 series, all re-fetched on
    every run while the record could never say finished)."""


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def tcia_licence(slug, ctx):
    """The collection's licences as TCIA's own site states them, one per download; stops on one not allowed."""
    page = fetch.get_json(fetch.TCIA_SITE + "collections/?slug=" + slug, ctx)
    if not page:
        raise fetch.FetchError("TCIA has no collection %r" % slug)
    seen = {}
    for did in page[0].get("collection_downloads") or []:
        d = fetch.get_json(fetch.TCIA_SITE + "downloads/%s" % did, ctx)
        seen[d.get("download_title") or str(did)] = d.get("data_license") or ""
    return seen


IDC_COLUMNS = ["collection_id", "PatientID", "SeriesInstanceUID", "StudyInstanceUID", "StudyDescription",
               "BodyPartExamined", "Modality", "sop_class_name", "SeriesDescription", "instanceCount",
               "license_short_name", "aws_bucket", "series_aws_url", "series_size_MB"]


def idc_rows(collection, modalities=None):
    """IDC's index (the idc-index-data package's parquet), only this collection's rows (of `modalities`, when given)
    and the columns used here, with the two columns IDCClient derives (crdc_series_uuid, series_size_MB as float)
    derived the same way. IDCClient read the whole index, a million series in 31 columns of Python strings, which stayed
    in the process for the rest of a run that lasts days: 4.1 and 4.6 GB per NLST shard on 2026-10-06, when the
    modelling chat ran short of memory. Arrow's pool hands back what the read used once the frame is made."""
    import idc_index_data  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    where = [("collection_id", "==", collection)] + ([("Modality", "in", list(modalities))] if modalities else [])
    df = pq.read_table(idc_index_data.IDC_INDEX_PARQUET_FILEPATH, columns=IDC_COLUMNS, filters=where).to_pandas()
    pa.default_memory_pool().release_unused()
    df["crdc_series_uuid"] = df["series_aws_url"].str.split("/").str[3]
    df["series_size_MB"] = df["series_size_MB"].astype(float)
    return df


def pick_series(collection, modalities, per_study, no_head, labels=True):
    """The collection's series in IDC: allowed licence, wanted modality; for CT, at most `per_study` per study
    (the most slices first, so the thinnest reconstruction) and none with fewer than MIN_CT_SLICES; with
    `no_head`, nothing that is or may include the head; then, unless `labels` is False, the labels (SEG, RTSTRUCT)
    of the studies kept."""
    import pandas as pd  # noqa: PLC0415
    every = idc_rows(collection, None if labels else modalities)   # the labels are looked up among every modality
    every["lic"] = every["license_short_name"].astype(str).map(fetch.licence_id)
    s = every[every["Modality"].isin(modalities)]
    raw = s[~s["sop_class_name"].isin(IMAGE_CLASSES)]
    if len(raw):
        log("%s: %d series that are not images left out (%s, %.1f GB)" % (
            collection, len(raw), raw["sop_class_name"].value_counts().to_dict(), raw["series_size_MB"].sum() / 1e3))
    s = s[s["sop_class_name"].isin(IMAGE_CLASSES)]
    if no_head:
        text = (s["BodyPartExamined"].astype(str) + " " + s["SeriesDescription"].astype(str) + " " +
                s["StudyDescription"].astype(str))
        head = text.str.contains(HEAD_WORDS, case=False, regex=True, na=False)
        log("%s: TCIA restricts its head images, so %d series that are or may include the head are left out (%.1f GB)" % (
            collection, int(head.sum()), s[head]["series_size_MB"].sum() / 1e3))
        s = s[~head]
    refused = s[~s["lic"].isin(ALLOWED)]
    s = s[s["lic"].isin(ALLOWED)]
    ct = s[s["Modality"] == "CT"]
    ct = ct[ct["instanceCount"] >= MIN_CT_SLICES].sort_values("instanceCount", ascending=False)
    if per_study:
        ct = ct.groupby("StudyInstanceUID", group_keys=False).head(per_study)
    images = pd.concat([s[s["Modality"] != "CT"], ct])
    if not labels:
        return images, refused
    kept = every[every["Modality"].isin(LABELS) & every["StudyInstanceUID"].isin(set(images["StudyInstanceUID"]))
                 & every["lic"].isin(ALLOWED)]
    return pd.concat([images, kept]), refused


class Record:
    def __init__(self, name):
        self.path = os.path.join(ARCHIVE, name, "train.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = fetch._load_json(self.path) or {"name": name, "series": {}}
        self.lock = threading.Lock()
        self.saved = 0.0

    def done(self, uid, row):
        with self.lock:
            self.data["series"][uid] = row
            if time.time() - self.saved > 30:
                fetch._save_json(self.path, self.data)
                self.saved = time.time()

    def save(self):
        with self.lock:
            fetch._save_json(self.path, self.data)


def header(path):
    import pydicom  # noqa: PLC0415
    d = pydicom.dcmread(path, stop_before_pixels=True)
    out = {}
    for k in HEADER:
        v = d.get(k)
        if v is not None and v != "":
            out[k] = [str(x) for x in v] if isinstance(v, (list, pydicom.multival.MultiValue)) else str(v)
    return out


def stacks(files):
    """A CT series' slices as evenly spaced stacks: sorted along the slice normal, split where the orientation
    changes or the step jumps (two passes in one series, a missing slice), so no volume is squashed."""
    import pydicom  # noqa: PLC0415
    groups = {}
    for f in files:
        d = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=[
            "ImagePositionPatient", "ImageOrientationPatient", "AcquisitionNumber", "InstanceNumber"])
        if "ImagePositionPatient" not in d or "ImageOrientationPatient" not in d:
            continue
        o = tuple(round(float(x), 3) for x in d.ImageOrientationPatient)
        n = np.cross(o[:3], o[3:])
        z = round(float(np.dot(n, [float(x) for x in d.ImagePositionPatient])), 3)
        groups.setdefault((o, str(d.get("AcquisitionNumber", ""))), []).append((z, int(d.get("InstanceNumber") or 0), f))
    out = []
    for slices in groups.values():
        # the same position twice (two passes under one acquisition number): the k-th slice at each position,
        # in instance order, goes to the k-th stack
        slices.sort()
        seen = {}
        layers = {}
        for z, _, f in slices:
            k = seen.get(z, 0)
            seen[z] = k + 1
            layers.setdefault(k, []).append((z, f))
        for layer in layers.values():
            if len(layer) < 2:
                continue
            step = float(np.median(np.diff([z for z, _ in layer])))
            run = [layer[0][1]]
            for (a, _), (b, f) in zip(layer, layer[1:]):
                if step <= 0 or abs((b - a) - step) > 0.25 * step:
                    out.append(run)
                    run = []
                run.append(f)
            out.append(run)
    return sorted((r for r in out if len(r) >= MIN_CT_SLICES), key=len, reverse=True)


def ct_copy(files, base):
    """A CT series' files -> one int16 NIfTI per evenly spaced stack (`base`.nii.gz, or `base`_2.nii.gz and on for a
    series holding more than one), at 1.5 mm on each axis finer than that and the stack's own spacing on any axis
    coarser: never upsampled. Returns [(path, the stack's original grid)]."""
    import SimpleITK as sitk  # noqa: PLC0415
    parts = stacks(files)
    if not parts:
        raise Unusable("no evenly spaced stack of %d or more slices" % MIN_CT_SLICES)
    out = []
    for k, names in enumerate(parts, 1):
        r = sitk.ImageSeriesReader()
        r.SetFileNames(names)
        img = r.Execute()
        size, sp = img.GetSize(), img.GetSpacing()
        out_sp = [max(s, CT_MM) for s in sp]
        new = [max(1, int(round(n * s / o))) for n, s, o in zip(size, sp, out_sp)]
        res = sitk.Resample(img, new, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), out_sp, img.GetDirection(),
                            float(HU[0]), sitk.sitkFloat32)
        res = sitk.Cast(sitk.Clamp(sitk.Round(res), sitk.sitkFloat32, *HU), sitk.sitkInt16)
        dst = base + ("" if k == 1 else "_%d" % k) + ".nii.gz"
        sitk.WriteImage(res, dst, useCompression=True)
        out.append((dst, {"original_size": list(size), "original_spacing_mm": [round(x, 4) for x in sp], "slices": len(names),
                          "copy": "%.1f mm on each axis finer than that, its own spacing on coarser ones (%s mm), linear, "
                                  "int16 HU clipped to %s" % (CT_MM, [round(x, 4) for x in out_sp], list(HU))}))
    return out


def planar_copy(path, dst, long_side):
    """One radiograph or mammogram -> its longest side at `long_side` px (never enlarged), 16-bit PNG."""
    import imagecodecs  # noqa: PLC0415
    import pydicom  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    d = pydicom.dcmread(path)
    a = d.pixel_array
    if a.ndim != 2:
        raise Unusable("pixel array of shape %s" % (a.shape,))
    h, w = a.shape
    lo, hi = max(int(a.min()), 0), min(int(a.max()), 65535)
    k = min(1.0, long_side / max(h, w))
    if k < 1.0:
        im = Image.fromarray(a.astype(np.float32), mode="F").resize((round(w * k), round(h * k)), Image.Resampling.LANCZOS)
        a = np.asarray(im)
    a = np.clip(np.round(a), lo, hi).astype(np.uint16)       # Lanczos overshoots at edges; keep the image's own range
    with open(dst, "wb") as f:
        f.write(imagecodecs.png_encode(a))
    return {"original_size": [int(w), int(h)], "copy": "longest side %d px (Lanczos), stored values as 16-bit PNG" % long_side,
            "bits_stored": int(d.get("BitsStored", 16))}


def one_series(row, name, root, tmp_root, ctx, rec, keep_other):
    uid = row.SeriesInstanceUID
    tmp = os.path.join(tmp_root, uid)
    objs = fetch._s3_list(row.aws_bucket, row.crdc_series_uuid + "/", ctx)
    if len(objs) != int(row.instanceCount) or any(md5 is None for _, _, md5 in objs):
        raise fetch.FetchError("%s: IDC folder holds %d objects for %d images" % (uid, len(objs), int(row.instanceCount)))
    entries = [fetch.Entry(key.split("/")[-1], size, "https://%s.s3.amazonaws.com/%s" % (row.aws_bucket, key), "md5:" + md5)
               for key, size, md5 in objs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=OBJECTS_AT_ONCE) as pool:
        list(pool.map(lambda e: fetch.download(e, os.path.join(tmp, e.name), ctx, "%s %s" % (name, e.name)), entries))
    files = sorted(os.path.join(tmp, e.name) for e in entries)
    bytes_in = sum(e.size for e in entries)
    folder = os.path.join(root, fetch._safe_name(str(row.PatientID)), row.StudyInstanceUID)
    os.makedirs(folder, exist_ok=True)
    meta = {"source": {"bucket": row.aws_bucket, "folder": row.crdc_series_uuid, "objects": {e.name: e.checksum for e in entries},
                       "bytes": bytes_in, "licence": row.license_short_name, "collection": row.collection_id},
            "header": header(files[0]), "made": now()}
    out = []
    if row.Modality == "CT" and "CT" not in EXACT:
        for dst, grid in ct_copy(files, os.path.join(folder, uid)):
            meta.setdefault("volumes", {})[os.path.basename(dst)] = grid
            out.append(dst)
        meta["slices_left_out"] = len(files) - sum(v["slices"] for v in meta["volumes"].values())
    elif row.Modality in EXACT:
        from core import store  # noqa: PLC0415
        dsts = []
        for f in files:
            dst = os.path.join(folder, uid, os.path.basename(f))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(f, dst)
            dsts.append(dst)
        for dst, (outcome, before, after) in zip(dsts, shrink_pool().map(store.shrink_file, dsts, chunksize=16)):
            meta.setdefault("images", {})[os.path.basename(dst)] = {
                "copy": "exact: the DICOM as served, %s (JPEG-LS lossless, pixels checked bit for bit)" % outcome,
                "bytes": [before, after]}
            out.append(dst)
    elif row.Modality in LONG_SIDE:
        for f in files:
            dst = os.path.join(folder, "%s_%s.png" % (uid, os.path.splitext(os.path.basename(f))[0]))
            meta.setdefault("images", {})[os.path.basename(dst)] = planar_copy(f, dst, LONG_SIDE[row.Modality])
            out.append(dst)
    elif keep_other or row.Modality in LABELS:
        for f in files:
            dst = os.path.join(folder, uid, os.path.basename(f))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(f, dst)
            out.append(dst)
    side = os.path.join(folder, uid + ".json")
    fetch._save_json(side, meta)
    shutil.rmtree(tmp, ignore_errors=True)
    bytes_out = sum(os.path.getsize(p) for p in out) + os.path.getsize(side)
    rec.done(uid, {"patient": str(row.PatientID), "modality": row.Modality, "bytes_in": bytes_in, "bytes_out": bytes_out,
                   "files": [os.path.relpath(p, root).replace("\\", "/") for p in out]})
    return bytes_in, bytes_out


def write_status(slug, cell):
    """The collection's LICENSING.md Status cell (its row is the one whose link is the TCIA collection page),
    landed the way every session here lands (cycle.py)."""
    link = TCIA_PAGE + slug + "/"
    with open(LICENSING, encoding="utf-8") as f:
        text = f.read()
    rows = list(re.finditer(r"^\| \[[^\]]*\]\(%s\)[^\n]*$" % re.escape(link), text, re.M))
    if len(rows) != 1:
        log("LICENSING.md: %d rows link %s; Status not written" % (len(rows), link))
        return
    line = rows[0].group(0)
    new = " | ".join(line.split(" | ")[:-1] + [cell.replace("|", "/")]) + " |"
    if new == line:
        return
    with open(LICENSING, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace(line, new))
    tool = os.path.expanduser("~/.claude/tools/cycle.py")
    if os.path.exists(tool):
        r = subprocess.run([shutil.which("python") or sys.executable, tool, "--land",
                            "LICENSING: Status of the %s training copies" % slug, "LICENSING.md"],
                           cwd=paths.ROOT, capture_output=True, text=True, creationflags=fetch.NO_WINDOW)
        log("landing LICENSING.md: %s" % (r.stdout or r.stderr).strip()[-300:])


def working_root(name, need, root=None):
    link = os.path.join(LINKS, name)
    if os.path.lexists(link):
        return os.readlink(link).replace("\\\\?\\", "")
    for r in [root] if root else WORKING:
        drive = os.path.splitdrive(r)[0] + "/"
        if os.path.isdir(drive) and min(shutil.disk_usage(drive).free - RESERVE, fetch.room(drive)) > need:
            return os.path.join(r, name)
    raise fetch.FetchError("%s: no working disk has %.0f GB to spare" % (name, need / 1e9))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["plan", "fetch"])
    ap.add_argument("collection", help="the TCIA slug (ct-colonography) or the IDC collection_id (ct_colonography)")
    ap.add_argument("--name", help="the set's name under Datasets/public (default: the TCIA slug plus -train)")
    ap.add_argument("--modality", default="CT,CR,DX,MG")
    ap.add_argument("--per-study", type=int, default=0, help="CT series kept per study, the most slices first (0: all)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many series this run (a trial)")
    ap.add_argument("--keep-other", action="store_true")
    ap.add_argument("--shard", help="I/N: only the series whose UID's md5 is I modulo N, so N processes can share a big "
                                    "collection; give each its own --name (nlst-train-0, nlst-train-1)")
    ap.add_argument("--no-labels", action="store_true", help="leave out the SEG and RTSTRUCT series (NLST's are "
                                                              "128,835 machine-made masks, 20 TB with the CTs)")
    ap.add_argument("--root", help="the folder the copies go under instead of the working disks: H:/rAidiology/cold "
                                   "for a collection not needed soon (owner, 2026-10-05)")
    ap.add_argument("--only", help="a text file of PatientIDs, StudyInstanceUIDs or SeriesInstanceUIDs, one a line: "
                                   "only the picked series they name (the studies a bar trains on)")
    ap.add_argument("--exact", default="", help="modalities kept as served this run, JPEG-LS lossless (CT: bar 139, "
                                                 "2026-10-10, lung CT needs exact); a copy keeps one form, so give it "
                                                 "its own --name (nlst-exact-0)")
    ap.add_argument("--log")
    a = ap.parse_args()
    if a.log:
        sys.stdout = sys.stderr = open(a.log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    EXACT.update(m.strip() for m in a.exact.split(",") if m.strip())
    slug = a.collection if "-" in a.collection else a.collection.replace("_", "-")
    collection = a.collection.replace("-", "_")
    name = a.name or slug + "-train"
    ctx = fetch.SetContext(name)
    lic = tcia_licence(slug, ctx)
    bad = {k: v for k, v in lic.items() if fetch.licence_id(v) not in ALLOWED}
    log("%s: TCIA's licences: %s%s" % (name, sorted(set(lic.values())), "; downloads under another licence: %s" % bad if bad else ""))
    unmapped = [t for t in bad if not re.search(r"head", t, re.I) and re.search(r"image|scan|dicom", t, re.I)]
    if unmapped:
        raise fetch.FetchError("%s: TCIA restricts images %s that cannot be told apart in IDC; not fetched" % (name, unmapped))
    series, refused = pick_series(collection, a.modality.split(","), a.per_study, no_head=bool(bad) and len(unmapped) == 0
                                  and any(re.search(r"head", t, re.I) for t in bad), labels=not a.no_labels)
    if a.only:                                     # a subset a bar named: series, studies or patients
        with open(a.only, encoding="utf-8") as f:
            ids = {x.strip() for x in f if x.strip() and not x.startswith("#")}
        keep = series["SeriesInstanceUID"].isin(ids) | series["StudyInstanceUID"].isin(ids) | \
            series["PatientID"].astype(str).isin(ids)
        log("%s: %d of %d picked series named in %s (%d ids)" % (name, int(keep.sum()), len(series), a.only, len(ids)))
        series = series[keep]
    if a.shard:                                    # one part of a collection too big for one process to finish soon
        part, parts = (int(x) for x in a.shard.split("/"))
        series = series[series["SeriesInstanceUID"].map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % parts
                                                         == part)]
    rec = Record(name)
    unusable = rec.data.setdefault("unusable", {})
    todo = series[~series["SeriesInstanceUID"].isin(list(rec.data["series"]) + list(unusable))]
    if a.limit:
        todo = todo.head(a.limit)
    gb = todo["series_size_MB"].sum() / 1e3
    log("%s: %d series picked (%s), %d left out for their licence; %d to do now, %.1f GB to download" % (
        name, len(series), series["Modality"].value_counts().to_dict(), len(refused), len(todo), gb))
    if a.command == "plan":
        return
    ct = "DICOM, JPEG-LS lossless" if "CT" in EXACT else "%.1f mm (never upsampled) int16 NIfTI" % CT_MM
    was = (rec.data.get("method") or {}).get("CT")
    if was and was != ct and rec.data.get("series"):
        raise fetch.FetchError("%s holds CT as %s; a copy keeps one form, so this one (%s) needs its own --name" % (name, was, ct))
    root = working_root(name, gb * (0.5e9 if "CT" in EXACT else 0.15e9), a.root)   # lossless CT: 2.0-3.2x smaller
    os.makedirs(root, exist_ok=True)
    made = fetch.junction(os.path.join(LINKS, name), root)
    rec.data.update(collection=collection, licences_at_tcia=lic, working_copy=root.replace("\\", "/"), picked=len(series),
                    shard=a.shard, modality=a.modality, per_study=a.per_study, method={"CT": ct,
                                                                       "planar": {k: "%d px 16-bit PNG" % v for k, v in LONG_SIDE.items()},
                                                                       "exact": {k: "DICOM, JPEG-LS lossless" for k in sorted(EXACT)}})
    rec.save()
    tmp_root = os.path.join(ARCHIVE, name, "tmp")
    t0, tot_in, tot_out, n, failed = time.time(), 0, 0, 0, 0
    lock = threading.Lock()

    def run(row):
        nonlocal tot_in, tot_out, n, failed
        try:
            bi, bo = one_series(row, name, root, tmp_root, ctx, rec, a.keep_other)
        except fetch.SetBlocked:
            raise
        except Unusable as e:                      # the data's own shape: kept with its reason, not fetched again
            with lock:
                failed += 1
                unusable[row.SeriesInstanceUID] = str(e)[:200]
            log("%s: series %s unusable: %s" % (name, row.SeriesInstanceUID, str(e)[:200]))
            shutil.rmtree(os.path.join(tmp_root, row.SeriesInstanceUID), ignore_errors=True)
            return
        except Exception as e:  # noqa: BLE001  one series must not stop the rest
            with lock:
                failed += 1
            log("%s: series %s skipped: %s: %s" % (name, row.SeriesInstanceUID, type(e).__name__, str(e)[:200]))
            shutil.rmtree(os.path.join(tmp_root, row.SeriesInstanceUID), ignore_errors=True)
            return
        with lock:
            tot_in += bi
            tot_out += bo
            n += 1
            if n % 50 == 0:
                log("%s: %d of %d series, %.1f GB in, %.2f GB kept (%.1fx), %.0f MB/s" % (
                    name, n, len(todo), tot_in / 1e9, tot_out / 1e9, tot_in / max(tot_out, 1), tot_in / 1e6 / (time.time() - t0)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=SERIES_AT_ONCE) as pool:
        list(pool.map(run, todo.itertuples(index=False)))
    done = rec.data["series"]
    b_in, b_out = sum(v["bytes_in"] for v in done.values()), sum(v["bytes_out"] for v in done.values())
    picked = set(series["SeriesInstanceUID"])
    whole = len(done) + len(picked & set(unusable)) >= len(series)
    rec.data.update(finished=now() if whole else None, bytes_in=b_in, bytes_out=b_out)
    rec.save()
    log("%s: this run %d series, %d skipped; %.1f GB in, %.2f GB kept (%.1fx smaller)%s" % (
        name, n, failed, tot_in / 1e9, tot_out / 1e9, tot_in / max(tot_out, 1), "; junction made" if made else ""))
    if a.limit:
        return
    write_status(slug, status_cell(collection, ", ".join(sorted(set(lic.values()))), rec.data))


def status_cell(collection, licences, current=None):
    """The Status cell for every copy of the collection: each part of a sharded one, and a copy in another form beside
    it (nlst-exact beside nlst-train-0 and -1 since bar 139); a record's "note" says what became of it, and a copy
    deleted since (its record keeps "deleted") is told as gone, never as a junction."""
    recs = [r for r in (fetch._load_json(p) or {} for p in sorted(glob.glob(os.path.join(ARCHIVE, "*", "train.json"))))
            if r.get("collection") == collection and r.get("series")]
    if current and current.get("name") not in {r.get("name") for r in recs}:
        recs.append(current)
    whole = all(r.get("finished") or r.get("deleted") for r in recs)

    def one(r):
        s = r["series"].values()
        b_in, b_out = sum(v["bytes_in"] for v in s), sum(v["bytes_out"] for v in s)
        where = "no longer kept" if r.get("deleted") else "a junction to `%s`" % r["working_copy"]
        return "`Datasets/public/%s` (%s; CT %s; %d of %d series, %.1f GB of DICOM kept as %.1f GB, " \
               "%.0fx smaller%s%s)" % (r["name"], where, (r.get("method") or {}).get("CT", "?"), len(r["series"]),
                                       r.get("picked", 0), b_in / 1e9, b_out / 1e9, b_in / max(b_out, 1),
                                       "; %d unusable, their reasons in train.json" % len(r["unusable"]) if r.get("unusable") else "",
                                       "; " + r["note"] if r.get("note") else "")
    return ("Approved by the owner, 2026-10-05 (\"store the data ... borderline 90%% smaller\"). **Training copies "
            "%s %s:** %s. Licence read at TCIA each run (%s) and each series' IDC licence; every object md5-checked "
            "before conversion; `%s` keeps each series' bucket folder and object md5s, so any original can be fetched "
            "again." % ("made" if whole else "partly made", now()[:10], "; ".join(one(r) for r in recs), licences,
                        "`, `".join(os.path.join(ARCHIVE, r["name"], "train.json").replace("\\", "/") for r in recs)))


if __name__ == "__main__":
    main()
