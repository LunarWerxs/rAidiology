"""Self-test for core/store.py: every DICOM it re-encodes as JPEG-LS lossless reads back bit for bit.

Synthetic data only, no network. Run: python tools/selftest_store.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from core import store

CT_SOP = "1.2.840.10008.5.1.4.1.1.2"
JPEG_LS_LOSSLESS = "1.2.840.10008.1.2.4.80"


def write_dicom(path, pixels, bits_stored, photometric="MONOCHROME2"):
    """An uncompressed Explicit VR Little Endian DICOM holding `pixels` (2-D, uint16 or int16) as stored."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CT_SOP
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CT_SOP
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = "Selftest"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = 1 if pixels.dtype.kind == "i" else 0
    ds.RescaleIntercept = "-1024"
    ds.RescaleSlope = "1"
    ds.PixelData = pixels.astype("<i2" if pixels.dtype.kind == "i" else "<u2").tobytes()
    ds.save_as(path, enforce_file_format=True)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def elements(ds):
    """Every element but the pixels and the group lengths, by tag, with its value."""
    return {int(t): ds[t].value for t in ds.keys() if t != store.PIXEL_TAG and (t & 0xFFFF)}


def check_shrunk_identical(path, pixels, label):
    """shrink_file re-encodes the file; the copy decodes to `pixels` exactly and every other element is unchanged."""
    before = pydicom.dcmread(path)
    outcome, _, _ = store.shrink_file(path)
    assert outcome == "shrunk", "%s: outcome %r" % (label, outcome)
    after = pydicom.dcmread(path)
    assert str(after.file_meta.TransferSyntaxUID) == JPEG_LS_LOSSLESS, "%s: transfer syntax %s" % (
        label, after.file_meta.TransferSyntaxUID)
    assert after.pixel_array.dtype == pixels.dtype, "%s: dtype %s" % (label, after.pixel_array.dtype)
    assert np.array_equal(after.pixel_array, pixels), "%s: decoded pixels differ" % label
    b, a = elements(before), elements(after)
    diff = sorted(hex(t) for t in set(a) | set(b) if a.get(t) != b.get(t))
    assert not diff, "%s: elements differ %s" % (label, diff)
    print("ok: %s shrunk, JPEG-LS lossless, pixels and %d other elements identical" % (label, len(b)))


def main():
    rng = np.random.default_rng(20261010)
    with tempfile.TemporaryDirectory(prefix="selftest_store_") as tmp:
        # 1. CT-like, unsigned 12-bit in 16 bits.
        ct = rng.integers(0, 4096, (64, 64), dtype=np.uint16)
        ct_path = os.path.join(tmp, "ct.dcm")
        write_dicom(ct_path, ct, bits_stored=12)
        check_shrunk_identical(ct_path, ct, "CT 12-bit")

        # 6. The same file again: a compressed transfer syntax is left alone (store.py:58-59, docstring line 8).
        before = read_bytes(ct_path)
        outcome, _, _ = store.shrink_file(ct_path)
        assert outcome.startswith("left"), "second run: outcome %r" % outcome
        assert read_bytes(ct_path) == before, "second run changed the file"
        print("ok: JPEG-LS file run again: %r, file unchanged byte for byte" % outcome)

        # 2. Signed CT, int16 with values beyond the usual range.
        signed = rng.integers(-1024, 3072, (64, 64), dtype=np.int16)
        signed[0, :3] = (-2000, 30000, -2000)
        signed_path = os.path.join(tmp, "signed.dcm")
        write_dicom(signed_path, signed, bits_stored=16)
        check_shrunk_identical(signed_path, signed, "signed CT 16-bit")

        # 3. Mammogram-like, 14 bits, MONOCHROME1.
        mammo = rng.integers(0, 16384, (256, 192), dtype=np.uint16)
        mammo_path = os.path.join(tmp, "mammo.dcm")
        write_dicom(mammo_path, mammo, bits_stored=14, photometric="MONOCHROME1")
        check_shrunk_identical(mammo_path, mammo, "mammogram 14-bit MONOCHROME1")

        # 4. Bits above BitsStored are set. store.py:82-84 leaves the file alone when the decoded values are not
        # the stored bits, and its docstring (lines 11-13) promises that a re-encoded file keeps those high bits.
        # So: either it was shrunk and its stored pixel values read back identical, or it is unchanged on disk.
        hi = rng.integers(0, 4096, (64, 64), dtype=np.uint16)
        hi.flat[[5, 6]] = [0xF123, 0x8001]
        hi_path = os.path.join(tmp, "highbits.dcm")
        write_dicom(hi_path, hi, bits_stored=12)
        before = read_bytes(hi_path)
        outcome, _, _ = store.shrink_file(hi_path)
        if outcome == "shrunk":
            assert np.array_equal(pydicom.dcmread(hi_path).pixel_array, hi), "high bits: pixels differ"
            print("ok: high bits re-encoded and read back identical (store.py:82-107)")
        else:
            assert outcome.startswith(("left", "kept")), "high bits: outcome %r" % outcome
            assert read_bytes(hi_path) == before, "high bits: left file changed"
            print("ok: high bits %r, file unchanged byte for byte (store.py:82-84)" % outcome)

        # 5. Not DICOM: left, unchanged. store.py:74-76. No extension, as NBIA names some files.
        noise_path = os.path.join(tmp, "noise")
        with open(noise_path, "wb") as f:
            f.write(rng.bytes(5000))
        before = read_bytes(noise_path)
        outcome, _, _ = store.shrink_file(noise_path)
        assert outcome.startswith("left"), "random bytes: outcome %r" % outcome
        assert read_bytes(noise_path) == before, "random bytes: file changed"
        print("ok: random bytes %r, file unchanged byte for byte (store.py:74-76)" % outcome)


if __name__ == "__main__":
    main()
