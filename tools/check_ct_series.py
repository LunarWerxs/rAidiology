"""Does a CT series from a hospital disc reach the model as the same body the scorecard measured?

    python tools/check_ct_series.py [--scan s0165] [--read]

The scorecard was measured on NIfTI volumes; a person's disc holds DICOM slices, and mcp/server.py
turns those into a volume (`_ct_volume`). Get the geometry wrong there and the model is shown a
mirrored body, so the left kidney is labelled right with full confidence. This writes one development
scan as an ordinary DICOM CT series, one slice a file with the standard patient geometry (LPS), the
files in shuffled order and their image numbers reversed, then reads it back through the server's
own conversion and requires the same voxels in the same place as the original. With `--read` it
also runs read_ct on the series and on the original NIfTI and requires the same answer from both,
and requires the outlines in the series' findings file to lie on the labels they came from, image
by image (a transposed, mirrored or misplaced outline scores far below the 0.9 it needs).

Development scans only (the held-out test is never opened outside the final evaluation).
"""
import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
from skimage.draw import polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "mcp"), ROOT]
import server  # noqa: E402  arkitect-allow: test-floor-loosened - the server's own functions, after its folder is on the path
from core import findings  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path lines, as server is

UID_ROOT = "1.2.826.0.1.3680043.8.498."      # pydicom's root for identifiers that belong to nobody


def ct_module(name):
    """A CT eval module by file: this process already holds the X-ray modules of the same names."""
    spec = importlib.util.spec_from_file_location("ct_" + name, os.path.join(ROOT, "modalities", "ct", "eval", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_series(img, folder, series_uid):
    """`img` as one DICOM file a slice, in LPS patient coordinates as DICOM defines them."""
    vol = np.asanyarray(img.dataobj).astype(np.int16)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ img.affine          # RAS to LPS
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    cols, rows = lps[:3, 0] / spacing[0], lps[:3, 1] / spacing[1]  # array axis 0 runs along a row
    paths = []
    for k in range(vol.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = "%s%d.%d" % (UID_ROOT, 7, k + 1)
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.SeriesInstanceUID = "CT", series_uid
        ds.StudyInstanceUID = UID_ROOT + "8"
        ds.FrameOfReferenceUID = UID_ROOT + "9"
        ds.InstanceNumber = vol.shape[2] - k                    # reversed on purpose
        ds.ImageOrientationPatient = [float(v) for v in list(cols) + list(rows)]
        ds.ImagePositionPatient = [float(v) for v in (lps @ np.array([0, 0, k, 1.0]))[:3]]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]   # between rows, between columns
        ds.SliceThickness = float(spacing[2])
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.PixelData = np.ascontiguousarray(vol[:, :, k].T).tobytes()
        p = os.path.join(folder, "IM%05d" % random.Random(k).randrange(10 ** 5) + "_%d" % k)
        ds.save_as(p, enforce_file_format=True)
        paths.append(p)
    return paths


def same_body(original, roundtrip):
    a, b = nib.as_closest_canonical(original), nib.as_closest_canonical(roundtrip)
    va, vb = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
    return va.shape == vb.shape and np.array_equal(va, vb) and np.allclose(a.affine, b.affine, atol=1e-3)


def filled(paths, shape):
    """The outlines of one structure on one image, filled even-odd, so a hole stays a hole."""
    mask = np.zeros(shape, bool)
    for text in paths:
        pts = findings.decode_path(text)
        rr, cc = polygon([y - 0.5 for _, y in pts], [x - 0.5 for x, _ in pts], shape)
        mask[rr, cc] ^= True
    return mask


def outlines_land(original, source, findings_file, work):
    """Do the outlines in the findings file lie on the right image, in the right place? Each image's
    outlines, filled, against the labels the same reading gives the NIfTI scan on that slice (image k
    of the written series is slice k of the scan). Mean Dice over every structure on every image."""
    doc = findings.read(findings_file)
    keep = tempfile.mkdtemp(dir=work)
    server._run_ct(source, keep)
    lab = nib.load(os.path.join(keep, "labels.nii"))
    turn = nib.orientations.ornt_transform(nib.io_orientation(lab.affine), nib.io_orientation(original.affine))
    labels = nib.orientations.apply_orientation(np.asanyarray(lab.dataobj), turn)
    slice_of = {findings.fingerprint("%s%d.%d" % (UID_ROOT, 7, k + 1)): k for k in range(labels.shape[2])}
    scores = []
    for row in doc["outlines"]["slices"]:
        truth = labels[:, :, slice_of[row["image_fingerprint"]]].T
        by = {}
        for s, text in row["paths"]:
            by.setdefault(s, []).append(text)
        for s, paths in by.items():
            got, want = filled(paths, truth.shape), truth == s + 1
            scores.append(2 * (got & want).sum() / max(got.sum() + want.sum(), 1))
    mean = float(np.mean(scores)) if scores else 0.0
    print("outlines: %d images, %d structure outlines, mean Dice %.4f against the labels they came from"
          % (len(doc["outlines"]["slices"]), len(scores), mean))
    return [] if mean > 0.9 else ["the outlines do not lie where the labels are (mean Dice %.3f)" % mean]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="s0165")
    ap.add_argument("--read", action="store_true", help="also run read_ct both ways (a few minutes)")
    a = ap.parse_args()
    datasets = ct_module("datasets")
    split = json.load(open(os.path.join(ROOT, "modalities", "ct", "eval", "split.json"), encoding="utf-8"))
    if a.scan not in split["dev_folds"]:
        raise SystemExit("%s is not a development scan" % a.scan)
    source = datasets.scan_path(a.scan)
    original = nib.load(source)
    work = tempfile.mkdtemp(prefix="raidiology-check-")
    failed = []
    try:
        series = UID_ROOT + "10"
        write_series(original, work, series)
        t0 = time.time()
        made, numbered, _ = server._ct_volume(work, series, tempfile.mkdtemp(dir=work))
        print("series of %d slices read back in %.1f s" % (len(numbered), time.time() - t0))
        if not same_body(original, nib.load(made)):
            failed.append("the series read back is not the scan it was written from")
        if a.read:
            answers = []
            outlined = tempfile.mkdtemp(dir=work)
            for name, args in (("DICOM series", {"path": work, "series_id": series, "findings": True, "out": outlined}),
                               ("NIfTI", {"file": source})):
                t0 = time.time()
                got = server.tool_read_ct(args)
                print("%s: %d structures in %.0f s on %s" % (name, got["structures_found"], time.time() - t0, got["read_on"]))
                answers.append({s["structure"]: s["millilitres"] for s in got["structures"]})
            # The same labels give the same sizes; the two files' voxel sizes may differ in the
            # sixth decimal, which can move a rounded millilitre by one tenth.
            if answers[0].keys() != answers[1].keys() or any(
                    abs(answers[0][n] - answers[1][n]) > 0.1 + 1e-4 * answers[1][n] for n in answers[1]):
                failed.append("read_ct answered differently for the series than for the NIfTI scan")
            failed += outlines_land(original, source, os.path.join(outlined, "scan" + findings.SUFFIX), work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("FAILED: " + "; ".join(failed) if failed else "OK: the series reaches the model as the scan itself")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
