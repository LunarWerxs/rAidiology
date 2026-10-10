"""Smaller scans that lose nothing: DICOM images stored uncompressed are re-encoded, inside DICOM, as JPEG-LS
lossless, the standard's lossless codec for medical images. Every reader we use opens them (SimpleITK through
GDCM, pydicom through pyjpegls).

Measured 2026-10-05 on our own files, every result decoded and compared value for value:
- CT series stored uncompressed (NLST, 39 to 269 slices): 2.0 to 3.2 times smaller.
- Mammograms stored uncompressed (CMMD, 8-bit): 9 to 19 times smaller.
- An image already compressed (PNG, JPEG, a compressed DICOM transfer syntax, NIfTI .nii.gz) is left alone:
  there is almost nothing left to take out of it losslessly (an NIH chest X-ray PNG: under 10%).

A file is replaced only after its new copy has been read back and found identical: the decoded pixels equal the
original's stored pixel bytes bit for bit (high bits included, so nothing hidden in them is lost), and every other
DICOM element is unchanged, group lengths aside: those (gggg,0000) only count bytes, the standard retired them, and
pydicom 3 no longer writes them. A file with an overlay kept in its pixel bits is left alone. The file's bytes do change
(its transfer syntax), so an archive's md5 for that file no longer applies; the caller records which files were
re-encoded.
"""
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pydicom
from pydicom.uid import JPEGLSLossless, UncompressedTransferSyntaxes

CODEC = JPEGLSLossless
CODEC_NAME = "JPEG-LS lossless"
PIXEL_TAG = 0x7FE00010
SKIP_DIRS = {".git", "__pycache__"}


def _raw(ds):
    """The stored pixel values exactly as the file holds them, as unsigned integers of the stored width."""
    bits = int(ds.BitsAllocated)
    dt = np.dtype("<u%d" % (bits // 8))
    n = int(ds.Rows) * int(ds.Columns) * int(ds.get("NumberOfFrames", 1) or 1) * int(ds.get("SamplesPerPixel", 1))
    return np.frombuffer(ds.PixelData, dtype=dt, count=n)


def _without_group_lengths(ds):
    """The dataset with its group length elements (gggg,0000) removed, at every depth; its pixels are not read."""
    for t in list(ds.keys()):
        if t.element == 0:
            del ds[t]
        elif t != PIXEL_TAG and ds[t].VR == "SQ":
            for item in ds[t].value:
                _without_group_lengths(item)
    return ds


def _reason_to_leave(ds):
    if "PixelData" not in ds:
        return "no pixels"
    if ds.file_meta.get("TransferSyntaxUID") not in UncompressedTransferSyntaxes:
        return "already compressed"
    if ds.get("BitsAllocated") not in (8, 16):
        return "%s-bit pixels" % ds.get("BitsAllocated")
    if int(ds.get("SamplesPerPixel", 1)) != 1 or not str(ds.get("PhotometricInterpretation", "")).startswith("MONOCHROME"):
        return "not a greyscale image"
    if any(0x6000 <= (e.tag >> 16) <= 0x601E and (e.tag >> 16) % 2 == 0 for e in ds if (e.tag & 0xFFFF) == 0x0102):
        return "an overlay in its pixel bits"
    return None


def shrink_file(path):
    """Re-encode one DICOM file losslessly in place. Returns (outcome, bytes before, bytes after), outcome one of
    "shrunk", "left: <why>" or "kept: <why the new copy was not identical>"."""
    old = os.path.getsize(path)
    try:
        ds = pydicom.dcmread(path)
    except Exception as e:  # noqa: BLE001  not DICOM, or a file pydicom cannot read: left as it is
        return "left: not DICOM (%s)" % type(e).__name__, old, old
    why = _reason_to_leave(ds)
    if why:
        return "left: " + why, old, old
    try:
        raw = _raw(ds).copy()
        arr = ds.pixel_array
        if not np.array_equal(arr.reshape(-1).view(raw.dtype), raw):
            return "left: stored bits beyond the image's values", old, old
        # the whole file, elements after the pixels included (Prostate-MRI-US-Biopsy's ultrasound has a private
        # group there); the pixels themselves are left on disk, deferred, and never compared here
        header = _without_group_lengths(pydicom.dcmread(path, defer_size=1 << 20))
        ds.compress(CODEC, arr, generate_instance_uid=False)     # the image keeps its own UID
        tmp = path + ".shrink"
        ds.save_as(tmp, enforce_file_format=True)
    except Exception as e:  # noqa: BLE001  the encoder refused this image: left as it is
        with contextlib.suppress(OSError):
            os.remove(path + ".shrink")
        return "left: encoder refused (%s: %s)" % (type(e).__name__, str(e)[:80]), old, old
    new = os.path.getsize(tmp)
    try:
        back = _without_group_lengths(pydicom.dcmread(tmp))
        same = np.array_equal(back.pixel_array.reshape(-1).view(raw.dtype), raw)
        diff = [str(t) for t in header.keys() if t != PIXEL_TAG and (t not in back or back[t].value != header[t].value)]
        diff += [str(t) for t in back.keys() if t != PIXEL_TAG and t not in header]
    except Exception as e:  # noqa: BLE001
        same, diff = False, ["read back failed: %s" % e]
    if not same or diff or new >= old:
        os.remove(tmp)
        return ("kept: %s" % ("pixels differ" if not same else "elements differ %s" % diff[:5] if diff else "no smaller")), old, old
    os.replace(tmp, path)
    return "shrunk", old, new


def dicom_files(root):
    """Every file under root that may be DICOM: .dcm, or no extension (NBIA and some archives name them so)."""
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            if ext in (".dcm", ".dicom", ".ima") or (ext == "" and not n.startswith(".")):
                yield os.path.join(folder, n)


NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0     # CREATE_NO_WINDOW: a hidden run opens no window


def _worker(list_path, out_path):
    """One worker process: each file on its list, one JSON line per file, flushed as it goes."""
    with open(list_path, encoding="utf-8") as f:
        paths = f.read().splitlines()
    with open(out_path, "a", encoding="utf-8") as out:
        for p in paths:
            out.write(json.dumps((p,) + shrink_file(p)) + "\n")
            out.flush()


def shrink_tree(root, workers=None, log=print):
    """Every DICOM file under root, re-encoded where that loses nothing, by `workers` processes of this module.
    Returns the totals. A run cut short is safe to repeat: a file already re-encoded is left alone."""
    workers = workers or max(1, min(8, (os.cpu_count() or 4) // 8))
    files = list(dicom_files(root))
    tot = {"files": len(files), "shrunk": 0, "left": 0, "kept": 0, "before": 0, "after": 0, "reasons": {}, "examples": {},
           "unchecked": 0, "seconds": 0}
    if not files:
        return tot
    t0 = last = time.time()
    with tempfile.TemporaryDirectory(prefix="shrink-") as tmp:
        procs, outs, errs = [], [], []
        for i in range(min(workers, len(files))):
            lst, out, err = (os.path.join(tmp, "%d.%s" % (i, x)) for x in ("txt", "jsonl", "err"))
            with open(lst, "w", encoding="utf-8") as f:
                f.write("\n".join(files[i::workers]))
            outs.append(out)
            errs.append(err)
            with open(err, "wb") as e:
                procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", lst, out],
                                              stdout=subprocess.DEVNULL, stderr=e, creationflags=NO_WINDOW))
        while any(p.poll() is None for p in procs):
            time.sleep(2)
            if time.time() - last > 60:
                last = time.time()
                n = sum(sum(1 for _ in open(o, encoding="utf-8")) for o in outs if os.path.exists(o))
                log("  %s: %d of %d files re-encoded or checked" % (os.path.basename(root.rstrip("/\\")), n, len(files)))
        failed = [open(e, encoding="utf-8", errors="replace").read()[-400:] for p, e in zip(procs, errs) if p.returncode]
        for o in outs:
            if not os.path.exists(o):
                continue
            with open(o, encoding="utf-8") as f:
                for line in f:
                    path, outcome, before, after = json.loads(line)
                    kind = outcome.split(":")[0]
                    tot[kind] += 1
                    tot["before"] += before
                    tot["after"] += after
                    if kind != "shrunk":
                        why = outcome.split(": ", 1)[1].split(" (")[0]
                        tot["reasons"][why] = tot["reasons"].get(why, 0) + 1
                        tot["examples"].setdefault(why, os.path.relpath(path, root))
    if failed:
        tot["worker_errors"] = failed
    tot["unchecked"] = tot["files"] - tot["shrunk"] - tot["left"] - tot["kept"]
    tot["seconds"] = round(time.time() - t0)
    return tot


if __name__ == "__main__" and sys.argv[1:2] == ["worker"]:
    _worker(sys.argv[2], sys.argv[3])
