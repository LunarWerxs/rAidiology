"""Per-slice DICOM geometry for each web stack, and projection of an app-frame point
(+X left, +Y superior, +Z anterior, mm) onto the nearest slice of a stack.
    python geometry.py <cd_root> <site_dir> <out.json>   # writes the geometry sidecar"""
import json
import os
import sys

import numpy as np

MR_STORAGE = "1.2.840.10008.5.1.4.1.1.4"


def build(cd, site, out):
    import pydicom
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from series import scan
    index = json.load(open(os.path.join(site, "data", "stacks", "index.json")))
    wanted = {e["source_series"] for e in index}
    geo = {}
    for s in scan(cd):
        if s["name"] not in wanted:
            continue
        ds = [pydicom.dcmread(p, stop_before_pixels=True) for p in s["files"]]  # already in NIfTI slice order
        geo[s["name"]] = [{"pos": [float(x) for x in d.ImagePositionPatient],
                      "orient": [float(x) for x in d.ImageOrientationPatient],
                      "spacing": [float(x) for x in d.PixelSpacing],
                      "rows": int(d.Rows), "cols": int(d.Columns)} for d in ds]
    json.dump(geo, open(out, "w"), indent=0)
    print("geometry for", sorted(geo))


def project(point_app, slices):
    """-> (slice index, x 0..1, y 0..1, distance mm) for the nearest slice."""
    X, Y, Z = point_app
    p = np.array([X, -Z, Y], float)  # app frame -> DICOM LPS
    best = None
    for k, s in enumerate(slices):
        o = np.array(s["orient"])
        row_dir, col_dir = o[:3], o[3:]
        n = np.cross(row_dir, col_dir)
        v = p - np.array(s["pos"])
        dist = float(np.dot(n, v))
        col = float(np.dot(v, row_dir) / s["spacing"][1])
        row = float(np.dot(v, col_dir) / s["spacing"][0])
        cand = (abs(dist), k, col / s["cols"], row / s["rows"], dist)
        if best is None or cand[0] < best[0]:
            best = cand
    return best[1], round(best[2], 4), round(best[3], 4), round(best[4], 1)


if __name__ == "__main__":
    build(*sys.argv[1:4])
