"""Self-test for tools/train_copy.py's conversion functions: what the 1.5 mm CT copy keeps and loses, the radiograph
copy's size and bit depth, and that mammograms stay exact. Synthetic data only, no network.
Run: python tools/selftest_train_copy.py   (a failed check raises, so the exit code is non-zero)
"""
import os
import sys
import tempfile

import imagecodecs
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
import train_copy  # noqa: E402

W = H = 128            # pixels per slice
PIX = 0.7              # mm, PixelSpacing
Z0 = -150.0            # mm, the first slice's z
INTERCEPT = -1024      # RescaleIntercept: stored value = HU - INTERCEPT, so ITK's reader gives HU back
CT_SOP = "1.2.840.10008.5.1.4.1.1.2"
CR_SOP = "1.2.840.10008.5.1.4.1.1.1"
SERIES = "1.2.826.0.1.3680043.2.1125.1.1"
STUDY = "1.2.826.0.1.3680043.2.1125.1.2"
FOR = "1.2.826.0.1.3680043.2.1125.1.3"
X, Y = np.meshgrid(np.arange(W) * PIX, np.arange(H) * PIX)   # mm of each pixel's column (X) and row (Y)


def linear(z):
    """HU = 2x + 3y - 1.5z + 100, x, y and z in mm."""
    return 2 * X + 3 * Y - 1.5 * z + 100


def clipped(z):
    """linear, with a 5000 HU block (rows and columns 20-59) and a -1500 HU block (80-119)."""
    hu = linear(z)
    hu[20:60, 20:60] = 5000
    hu[80:120, 80:120] = -1500
    return hu


def write_dcm(path, raw, *, sop, modality, signed, ipp=(0.0, 0.0, 0.0), acq=1, inst=1, thickness=1.25):
    """One uncompressed DICOM file holding `raw` (rows x columns) as 16-bit pixels."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID = sop
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY
    ds.SeriesInstanceUID = SERIES
    ds.FrameOfReferenceUID = FOR
    ds.Modality = modality
    ds.PatientID = "SELFTEST"
    ds.ImagePositionPatient = list(ipp)
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [PIX, PIX]
    ds.SliceThickness = thickness
    ds.AcquisitionNumber = acq
    ds.InstanceNumber = inst
    ds.RescaleSlope = "1"
    ds.RescaleIntercept = str(INTERCEPT) if signed else "0"
    ds.Rows, ds.Columns = raw.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if signed else 0
    ds.PixelData = raw.astype("<i2" if signed else "<u2").tobytes()
    ds.save_as(path)
    return path


def write_series(folder, n, dz, hu=linear, acq=1, tag=""):
    """n axial CT slices at Z0 + k*dz, each HU plane from hu(z); returns the paths."""
    return [write_dcm(os.path.join(folder, "%s%03d.dcm" % (tag, k)), np.round(hu(Z0 + k * dz) - INTERCEPT),
                      sop=CT_SOP, modality="CT", signed=True, ipp=(0.0, 0.0, Z0 + k * dz), acq=acq, inst=k + 1,
                      thickness=dz) for k in range(n)]


def check_geometry_and_values():
    with tempfile.TemporaryDirectory() as d:
        (path, grid), = train_copy.ct_copy(write_series(d, 40, 1.25), os.path.join(d, "copy"))
        img = sitk.ReadImage(path)
        assert np.allclose(img.GetSpacing(), (1.5, 1.5, 1.5)), img.GetSpacing()
        # ct_copy resamples on the first slice's grid (the reader sorts ascending in z): the origin is that position
        assert np.allclose(img.GetOrigin(), (0.0, 0.0, Z0)), img.GetOrigin()
        assert img.GetPixelID() == sitk.sitkInt16, img.GetPixelIDTypeAsString()
        arr = sitk.GetArrayFromImage(img)                       # [z, y, x]
        ox, oy, oz = img.GetOrigin()
        rng = np.random.default_rng(0)
        for _ in range(200):
            i, j, k = (int(rng.integers(n)) for n in arr.shape[::-1])
            x, y, z = ox + i * 1.5, oy + j * 1.5, oz + k * 1.5
            # trilinear reproduces a linear function; each stored pixel and the copy round to a whole HU: 0.5 + 0.5
            assert abs(arr[k, j, i] - (2 * x + 3 * y - 1.5 * z + 100)) <= 1.0 + 1e-6, (i, j, k, arr[k, j, i])
    print("ok: 1.5 mm grid, first-slice origin, int16, linear HU kept within 1 HU at 200 voxels (ct_copy)")


def check_clipping():
    with tempfile.TemporaryDirectory() as d:
        (path, _), = train_copy.ct_copy(write_series(d, 40, 1.25, hu=clipped), os.path.join(d, "copy"))
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        assert arr.max() == 3071 and arr.min() == -1024, (arr.min(), arr.max())
        assert (arr[:, 19, 19] == 3071).all()     # 28.5 mm: inside the 5000 HU block (14-41 mm)
        assert (arr[:, 46, 46] == -1024).all()    # 69 mm: inside the -1500 HU block (56-83 mm)
    print("ok: HU clipped to [-1024, 3071] (ct_copy, HU)")


def check_never_upsampled():
    with tempfile.TemporaryDirectory() as d:
        (path, _), = train_copy.ct_copy(write_series(d, 24, 2.5), os.path.join(d, "copy"))
        assert np.allclose(sitk.ReadImage(path).GetSpacing(), (1.5, 1.5, 2.5)), sitk.ReadImage(path).GetSpacing()
    print("ok: 2.5 mm slices keep their 2.5 mm z spacing, pixels go to 1.5 mm (ct_copy, never upsampled)")


def check_two_passes():
    with tempfile.TemporaryDirectory() as d:
        files = write_series(d, 40, 1.25, acq=1, tag="a") + write_series(d, 40, 1.25, acq=2, tag="b")
        out = train_copy.ct_copy(files, os.path.join(d, "copy"))
        assert sorted(os.path.basename(p) for p, _ in out) == ["copy.nii.gz", "copy_2.nii.gz"], out
        assert all(os.path.exists(p) and g["slices"] == 40 for p, g in out), out
    print("ok: two AcquisitionNumber passes over the same positions give two volumes (ct_copy, stacks)")


def check_planar():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        big = rng.integers(200, 4000, (1500, 2000)).astype(np.uint16)
        src = write_dcm(os.path.join(d, "big.dcm"), big, sop=CR_SOP, modality="CR", signed=False)
        dst = os.path.join(d, "big.png")
        train_copy.planar_copy(src, dst, 1024)
        with open(dst, "rb") as f:
            png = f.read()
        assert png[24] == 16, "IHDR bit depth %d, not 16" % png[24]
        out = imagecodecs.png_decode(png)
        assert out.shape == (768, 1024) and out.dtype == np.uint16, (out.shape, out.dtype)
        assert big.min() <= out.min() and out.max() <= big.max(), "values left the source's range"

        small = rng.integers(0, 4000, (600, 800)).astype(np.uint16)
        src = write_dcm(os.path.join(d, "small.dcm"), small, sop=CR_SOP, modality="CR", signed=False)
        dst = os.path.join(d, "small.png")
        train_copy.planar_copy(src, dst, 1024)
        with open(dst, "rb") as f:
            assert imagecodecs.png_decode(f.read()).shape == (600, 800), "800x600 was resized"
    print("ok: CR 2000x1500 -> longest side 1024 as 16-bit PNG within source range; 800x600 not enlarged (planar_copy)")


def check_mammograms_exact():
    assert "MG" in train_copy.EXACT and "MG" not in train_copy.LONG_SIDE, (train_copy.EXACT, train_copy.LONG_SIDE)
    print("ok: MG is in EXACT and not in LONG_SIDE (mammograms kept exact)")


if __name__ == "__main__":
    for check in [check_geometry_and_values, check_clipping, check_never_upsampled, check_two_passes,
                  check_planar, check_mammograms_exact]:
        check()
