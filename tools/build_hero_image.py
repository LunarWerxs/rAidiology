"""Draw the picture on the front page: one slice of the sample scan, as the scan page shows it.

The front page shows what a visitor gets, not an illustration of it. So the picture is read from
the same sample ZIP the "Try a sample scan" button opens (built by tools/build_sample_scan.py from
the TotalSegmentator dataset, CC BY 4.0), the middle slice, in the usual soft-tissue window
(centre 40, width 400 Hounsfield units). Nothing else goes into it.

The same sample gives the same bytes, and the public export refuses every image whose checksum it
has not pinned, so a rebuilt picture is looked at and re-pinned on purpose (`PINNED` in
tools/public_export.py; this prints the checksum). It also prints which side of the body each edge
of the picture is, worked out from the slice's own orientation the way the scan page does, for the
labels drawn around it.

    python tools/build_hero_image.py [--zip site/sample/sample-ct.zip] [--out site/img/sample-slice.webp]
"""
import argparse
import hashlib
import io
import os
import zipfile

import numpy as np
import pydicom
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDES = [("right side of the body", "left side of the body"), ("front", "back"), ("feet", "head")]
CENTRE, WIDTH = 40.0, 400.0
SIZE = 560


def side(v):
    axis = int(np.argmax(np.abs(v)))
    toward = SIDES[axis][1 if v[axis] > 0 else 0]
    away = SIDES[axis][0 if v[axis] > 0 else 1]
    return toward, away


def middle_slice(path):
    with zipfile.ZipFile(path) as z:
        slices = [pydicom.dcmread(io.BytesIO(z.read(n))) for n in sorted(z.namelist()) if n.endswith(".dcm")]
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    return slices[len(slices) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(ROOT, "site", "sample", "sample-ct.zip"))
    ap.add_argument("--out", default=os.path.join(ROOT, "site", "img", "sample-slice.webp"))
    a = ap.parse_args()
    d = middle_slice(a.zip)
    hu = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
    grey = np.clip((hu - (CENTRE - WIDTH / 2)) / WIDTH, 0, 1) * 255
    img = Image.fromarray(grey.round().astype(np.uint8), "L").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=80, method=6)
    with open(a.out, "wb") as f:
        f.write(buf.getvalue())
    o = [float(v) for v in d.ImageOrientationPatient]
    right, left = side(o[:3])
    bottom, top = side(o[3:])
    print("%s: %d bytes, sha256 %s" % (a.out, len(buf.getvalue()), hashlib.sha256(buf.getvalue()).hexdigest()))
    print("edges: top %s, bottom %s, left %s, right %s" % (top, bottom, left, right))


if __name__ == "__main__":
    main()
