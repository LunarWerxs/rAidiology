"""A made-up DICOM series, so the whole chain can be tested without anybody's real scan.

The MCP server, the findings file and the browser viewer have to agree with each other across two
languages: Python hashes a DICOM identifier one way and JavaScript has to hash it the same way, or
a reading silently fails to match the pictures it is about. That is exactly the kind of break that
passes every test written on one side of the line.

So this writes a small series of plain grey radiograph-shaped images with known identifiers. There
is no patient in it and nothing diagnostic about it: the shapes are drawn with arithmetic. It exists
to be matched, not to be read.

    python mcp/demo_scan.py <folder>        writes the series and prints its fingerprints
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SERIES_UID = "1.2.826.0.1.3680043.8.498.10000000000000000000000000000001"
SOP_UIDS = ["1.2.826.0.1.3680043.8.498.2000000000000000000000000000000%d" % i for i in (1, 2, 3)]
SIZE = 192


def _picture(index):
    """A grey field with a brighter bar across it, different in each image.

    Arithmetic rather than a stored image so the repository carries no picture of anybody, and so
    the series is identical on every machine that generates it."""
    rows = []
    for y in range(SIZE):
        row = []
        for x in range(SIZE):
            edge = min(x, y, SIZE - 1 - x, SIZE - 1 - y)
            v = 40 + min(edge, 60)
            if abs(y - (50 + index * 40)) < 10 and 30 < x < SIZE - 30:
                v += 120
            row.append(min(255, v))
        rows.append(row)
    return rows


def write(folder):
    import numpy as np  # noqa: PLC0415
    from pydicom.dataset import Dataset, FileMetaDataset  # noqa: PLC0415
    from pydicom.uid import ExplicitVRLittleEndian  # noqa: PLC0415

    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, sop in enumerate(SOP_UIDS):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"      # computed radiography
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = sop
        ds.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.3000000000000000000000000000001"
        ds.SeriesInstanceUID = SERIES_UID
        ds.SeriesNumber = 1
        ds.InstanceNumber = i + 1
        ds.Modality = "CR"
        ds.SeriesDescription = "DEMO NOT A REAL SCAN"
        ds.BodyPartExamined = "WRIST"
        ds.Rows = SIZE
        ds.Columns = SIZE
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [0.2, 0.2]
        ds.PixelData = np.array(_picture(i), dtype=np.uint8).tobytes()
        path = os.path.join(folder, "demo%d.dcm" % (i + 1))
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def main():
    from core import findings as ff  # noqa: PLC0415
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "tmp", "demo-scan")
    paths = write(folder)
    print("wrote %d file(s) to %s" % (len(paths), os.path.abspath(folder)))
    print("series fingerprint: %s" % ff.fingerprint(SERIES_UID))
    for i, sop in enumerate(SOP_UIDS):
        print("  image %d fingerprint: %s" % (i + 1, ff.fingerprint(sop)))


if __name__ == "__main__":
    main()
