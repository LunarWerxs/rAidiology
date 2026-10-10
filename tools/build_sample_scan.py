"""Build the sample scan the "Open your own scan" page offers to somebody with no disc to hand.

The first five simulated visitors all stopped at the same place: the page asks for a scan, they did
not have theirs with them, and there was nothing to try. A sample fixes that, and what it is made
of matters more than usual on a page whose whole promise is privacy:

- **It is nobody's private scan and never could be.** It is read from the TotalSegmentator dataset
  (CC BY 4.0, the row in LICENSING.md), which is published, anonymised research data, and from
  nowhere else.
- **Development scans only.** The dataset's held-out test scans are never opened outside the final
  evaluation, and a sample is not a reason to make an exception.
- **Trunk only, no marked pathology.** A scan with no head in it has no face to reconstruct, and a
  visitor frightened about their own result should not be handed a stranger's tumour to look at.
- **No patient tags are written at all.** The files carry what a viewer needs to draw them and
  nothing about a person: no name, no number, no date.

It is written as an ordinary hospital-style series (one DICOM file a slice) inside a ZIP, so the
sample takes exactly the path a visitor's own ZIP takes. Every second slice is left out, which
halves the download and is what many hospital discs do anyway.

The same inputs give the same bytes: the archive's timestamps are fixed. That matters because the
public export refuses every image file except ones whose checksum it has pinned, so a rebuilt sample
has to be looked at and re-pinned on purpose (`PINNED` in tools/public_export.py; this script prints
the checksum).

    python tools/build_sample_scan.py [--scan s0165] [--out site/sample/sample-ct.zip]
"""
import argparse
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "ct", "eval"))
import datasets

UID_ROOT = "1.2.826.0.1.3680043.8.498."      # pydicom's root for identifiers that belong to nobody
EVERY = 2
AIR = -1024


def uid(*parts):
    """A made-up identifier, the same on every machine: digits of a hash under a public root."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return UID_ROOT + str(int(digest[:30], 16))[:30]


def build(scan, out):
    import nibabel as nib
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    split = json.load(open(os.path.join(ROOT, "modalities", "ct", "eval", "split.json"), encoding="utf-8"))
    if scan not in split["dev_folds"]:
        raise SystemExit("%s is not a development scan. The sample is never cut from the test split." % scan)
    row = datasets.index().set_index("image_id").loc[scan]
    if row["pathology"] != "no_pathology":
        raise SystemExit("%s is marked %r. The sample is a scan with no marked pathology." % (scan, row["pathology"]))

    img = nib.as_closest_canonical(nib.load(datasets.scan_path(scan)))
    spacing = [float(v) for v in img.header.get_zooms()[:3]]
    ct = np.asanyarray(img.dataobj)
    x, y, z = ct.shape
    series, study = uid("series", scan), uid("study", scan)
    buffer = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for k, at in enumerate(range(0, z, EVERY)):
            # The array is right-anterior-superior; a DICOM picture is drawn with the patient's
            # right on the left of the screen and their front at the top. So both in-plane axes
            # run backwards, and the first axis becomes the columns.
            picture = ct[::-1, ::-1, at].T
            stored = (np.clip(picture, AIR, 3071) - AIR).astype(np.uint16)
            sop = uid("image", scan, at)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"        # CT image storage
            meta.MediaStorageSOPInstanceUID = sop
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = Dataset()
            ds.file_meta = meta
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = sop
            ds.StudyInstanceUID = study
            ds.SeriesInstanceUID = series
            ds.SeriesNumber = 1
            ds.InstanceNumber = k + 1
            ds.Modality = "CT"
            ds.SeriesDescription = "Sample scan (not yours)"
            ds.Rows, ds.Columns = int(stored.shape[0]), int(stored.shape[1])
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
            ds.PixelRepresentation = 0
            ds.RescaleIntercept, ds.RescaleSlope = AIR, 1
            ds.WindowCenter, ds.WindowWidth = 40, 400                         # the soft-tissue window
            ds.PixelSpacing = [spacing[1], spacing[0]]
            ds.SliceThickness = spacing[2] * EVERY
            ds.SpacingBetweenSlices = spacing[2] * EVERY
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0.0, 0.0, at * spacing[2]]
            ds.PixelData = stored.tobytes()
            one = io.BytesIO()
            ds.save_as(one, enforce_file_format=True)
            entry = zipfile.ZipInfo("sample-ct/IM%04d.dcm" % (k + 1), date_time=(2026, 10, 1, 0, 0, 0))
            zf.writestr(entry, one.getvalue(), zipfile.ZIP_DEFLATED, 9)
            count += 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(buffer.getvalue())
    return count, (y, x), len(buffer.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="s0165")
    ap.add_argument("--out", default=os.path.join(ROOT, "site", "sample", "sample-ct.zip"))
    a = ap.parse_args()
    count, shape, size = build(a.scan, a.out)
    print("%s: %d slices of %dx%d from %s, %.1f MB" % (a.out, count, shape[1], shape[0], a.scan, size / 1e6))
    print("sha256 %s" % hashlib.sha256(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
