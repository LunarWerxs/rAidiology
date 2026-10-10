"""The answer key of LIDC-IDRI: where four thoracic radiologists marked lung nodules in chest CT.

Each scan was read by four radiologists, who marked every nodule they saw. Nodules of 3 mm or more
were also rated (subtlety, internal structure, calcification, sphericity, margin, lobulation,
spiculation, texture, malignancy 1 to 5); smaller ones are a single point; and a "non-nodule" mark
says the radiologist looked at that spot and decided it was something else. The data is fetched by
`modalities/ct/fetch_tcia.py lidc-idri` into <Datasets>/public/lidc-idri/: `series.json` (TCIA's records),
one folder per scan (<PatientID>/<SeriesInstanceUID>/*.dcm, complete only when it holds the
`.unpacked` marker) and `annotations/` (one XML file per read).

Two things to keep in mind whenever these marks are used as truth:

- These are the UNBLINDED reads. In the second round of reading each radiologist saw the other
  three's marks before finalising their own, so the four are not independent. A reader scored
  against the other three is flattered: the three it is scored against partly shaped its answer.
- The reader numbers (0 to 3) are only the position of a reading session inside one scan's file.
  The same number is not the same radiologist across scans.

Which XML files count: only those whose SeriesInstanceUid is a CT series in `series.json` (a few
files hold chest radiograph reads). When one series has several files, the one with the most
reading sessions is kept, and on a tie the one later in name order. What was dropped is printed.

What comes out:

- `geometry(series_uid)`: from the scan's DICOM headers, SOPInstanceUID -> z, pixel spacing (row,
  column, mm), the x and y of the scan's corner, and the slice spacing. Cached as JSON under
  <Work>/eval/lidc/geometry/ so the headers are read once.
- `marks()`: one row per mark of every radiologist in every kept scan: `kind` is 'nodule' (has
  ratings, 3 mm or more), 'small' (a nodule mark without ratings, under 3 mm) or 'non_nodule'. The
  position is the centroid of the contour points marked "inclusion" over all the mark's slices, in
  millimetres (x = x0 + column * column spacing, y = y0 + row * row spacing, both negated on the 14
  prone scans, `to_mm`; the XML's xCoord is the column, yCoord the row). `diameter_mm` (nodules only) is the largest in-plane extent of any
  one slice's contour.
- `nodules()`: the marks of different radiologists that are the same lesion, matched as below.

How marks are matched into a lesion: within one scan, two nodule marks of different readers match
when the distance between their centroids is at most max(their two diameters) / 2 + 2 mm (the 2 mm
is `tol`). Every such pair is listed, nearest first, and taken greedily; a pair joins the two
groups its marks belong to unless that would put two marks of one reader in a group. A lesion
therefore has at most one mark per reader.

    python modalities/ct/eval/lidc.py
"""
import functools
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from core import paths  # noqa: E402

ROOT = os.path.join(paths.DATASETS, "public", "lidc-idri")
GEO = os.path.join(paths.WORK, "eval", "lidc", "geometry")
CHARS = ["subtlety", "internalStructure", "calcification", "sphericity", "margin", "lobulation",
         "spiculation", "texture", "malignancy"]
TOL_MM = 2.0
LUNA16_AT_3 = 1186          # lesions marked by 3 of 4 in LUNA16's 888 scans (slices of 2.5 mm or thinner)
LUNA16_SPACING = 2.5


def series_records():
    with open(os.path.join(ROOT, "series.json"), encoding="utf-8") as f:
        return {x["SeriesInstanceUID"]: x for x in json.load(f)}


def scan_dir(rec):
    return os.path.join(ROOT, rec["PatientID"], rec["SeriesInstanceUID"])


@functools.lru_cache(maxsize=1)
def unpacked():
    """(uids of the scans whose folder is complete, number skipped as still downloading)."""
    recs = series_records()
    ok = {u for u, r in recs.items() if os.path.exists(os.path.join(scan_dir(r), ".unpacked"))}
    return ok, len(recs) - len(ok)


def _head(path):
    d = __import__("pydicom").dcmread(path, stop_before_pixels=True,
                                      specific_tags=["SOPInstanceUID", "ImagePositionPatient", "PixelSpacing",
                                                     "ImageOrientationPatient"])
    iop = [float(v) for v in getattr(d, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    return (str(d.SOPInstanceUID), [float(v) for v in d.ImagePositionPatient],
            [float(v) for v in d.PixelSpacing], iop)


def _direction(iop):
    """[+1 or -1, +1 or -1]: whether world x rises with the column and world y with the row. 14 of
    LIDC-IDRI's scans are prone (orientation -1 0 0 0 -1 0), where both fall; read as +1, their marks
    came out mirrored in millimetres (2026-10-03, found by lung_prepare.py's check)."""
    return [1 if iop[0] >= 0 else -1, 1 if iop[4] >= 0 else -1]


def to_mm(g, col, row):
    """A mark's (column, row) in the slice to world (x, y) millimetres."""
    return (g["x0"] + g["dir"][0] * np.asarray(col) * g["spacing"][1],
            g["y0"] + g["dir"][1] * np.asarray(row) * g["spacing"][0])


def geometry(series_uid):
    """{'z': {SOPInstanceUID: z mm}, 'spacing': [row, col] mm, 'x0', 'y0' (mm), 'slice_spacing' (mm),
    'dir' (see _direction)}. A cache written before 'dir' existed gets it from one header."""
    cache = os.path.join(GEO, series_uid + ".json")
    # Two processes can build the same cache at once (2026-10-03: the parse and lidc_readers.py). On
    # Windows a file another process is replacing this instant cannot be opened, so a read waits a
    # moment and tries again, and each writer uses its own temporary name.
    for attempt in range(10):
        if not os.path.exists(cache):
            break
        try:
            with open(cache, encoding="utf-8") as f:
                g = json.load(f)
            if "dir" not in g:
                rec = series_records()[series_uid]
                g["dir"] = _direction(_head(sorted(glob.glob(os.path.join(scan_dir(rec), "*.dcm")))[0])[3])
                _write_geometry(cache, g)
            return g
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.2 * (attempt + 1))
    rec = series_records()[series_uid]
    files = sorted(glob.glob(os.path.join(scan_dir(rec), "*.dcm")))
    with ThreadPoolExecutor(8) as pool:
        heads = list(pool.map(_head, files))
    z = {h[0]: h[1][2] for h in heads}
    zs = np.unique(np.round(list(z.values()), 4))
    g = {"z": z,
         "spacing": [float(np.median([h[2][0] for h in heads])), float(np.median([h[2][1] for h in heads]))],
         "x0": float(np.median([h[1][0] for h in heads])),
         "y0": float(np.median([h[1][1] for h in heads])),
         "slice_spacing": float(np.median(np.diff(zs))) if len(zs) > 1 else float("nan"),
         "dir": _direction(heads[0][3])}
    _write_geometry(cache, g)
    return g


def _write_geometry(cache, g):
    os.makedirs(GEO, exist_ok=True)
    part = "%s.%d.part" % (cache, os.getpid())
    with open(part, "w", encoding="utf-8") as f:
        json.dump(g, f)
    try:
        os.replace(part, cache)
    except PermissionError:     # the other process's copy is in place, and the same
        os.remove(part)
    return g


def _strip(root):
    for e in root.iter():
        e.tag = e.tag.split("}")[-1]
    return root


def _read_xml(path):
    root = _strip(ET.parse(path).getroot())
    uid = root.findtext("ResponseHeader/SeriesInstanceUid")
    return (uid.strip() if uid else None), root, len(root.findall("readingSession"))


@functools.lru_cache(maxsize=1)
def kept_files():
    """({series uid: (xml path, root)}, report) after dropping non-CT files and duplicates."""
    recs = series_records()
    by_uid, not_ct, unreadable = {}, [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "annotations", "**", "*.xml"), recursive=True)):
        try:
            uid, root, n = _read_xml(p)
        except ET.ParseError:
            unreadable.append(p)
            continue
        if uid not in recs:
            not_ct.append(p)
            continue
        by_uid.setdefault(uid, []).append((n, os.path.basename(p), p, root))
    kept, dup = {}, []
    for uid, fs in by_uid.items():
        fs.sort(key=lambda t: (t[0], t[1]))
        kept[uid] = (fs[-1][2], fs[-1][3])
        dup += [t[2] for t in fs[:-1]]
    return kept, {"not_ct": not_ct, "duplicate": dup, "unreadable": unreadable}


def _f(e, tag):
    t = e.findtext(tag)
    return float(t) if t not in (None, "") else float("nan")


def _mark(roi_list, g):
    """Centroid (mm), largest in-plane extent (mm) and slice count of one mark's inclusion rois."""
    sr, sc = g["spacing"]
    pts, zs, diam = [], [], 0.0
    for roi in roi_list:
        if (roi.findtext("inclusion") or "").strip().upper() != "TRUE":
            continue
        xy = np.array([[_f(p, "xCoord"), _f(p, "yCoord")] for p in roi.findall("edgeMap")])
        if not len(xy):
            continue
        mm = np.c_[to_mm(g, xy[:, 0], xy[:, 1])]
        pts.append(mm)
        zs.append(_f(roi, "imageZposition"))
        diam = max(diam, float(np.sqrt(((mm[:, None] - mm[None]) ** 2).sum(-1).max())))
    if not pts:
        return None
    p = np.vstack(pts)
    return p[:, 0].mean(), p[:, 1].mean(), float(np.mean(zs)), diam, len(pts)


@functools.lru_cache(maxsize=1)
def _build():
    recs = series_records()
    ok, _ = unpacked()
    files, _ = kept_files()
    rows, scans = [], []
    for uid, (path, root) in sorted(files.items()):
        if uid not in ok:
            continue
        g = geometry(uid)
        patient = recs[uid]["PatientID"]
        sessions = root.findall("readingSession")
        scans.append({"series_uid": uid, "patient": patient, "n_readers": len(sessions),
                      "slice_spacing": g["slice_spacing"], "xml": os.path.basename(path)})
        for reader, s in enumerate(sessions):
            base = {"series_uid": uid, "patient": patient, "reader": reader}
            for n in s.findall("unblindedReadNodule"):
                m = _mark(n.findall("roi"), g)
                if m is None:
                    continue
                ch = n.find("characteristics")
                has = ch is not None and len(ch) > 0
                r = dict(base, kind="nodule" if has else "small", nodule_id=(n.findtext("noduleID") or "").strip(),
                         x_mm=m[0], y_mm=m[1], z_mm=m[2], diameter_mm=m[3] if has else float("nan"), n_slices=m[4])
                r.update({c: _f(ch, c) if has else float("nan") for c in CHARS})
                rows.append(r)
            for n in s.findall("nonNodule"):
                x, y = _f(n, "locus/xCoord"), _f(n, "locus/yCoord")
                if x != x or y != y:
                    continue
                z = _f(n, "imageZposition")
                if z != z:
                    z = g["z"].get((n.findtext("imageSOP_UID") or "").strip(), float("nan"))
                r = dict(base, kind="non_nodule", nodule_id=(n.findtext("nonNoduleID") or "").strip(),
                         x_mm=float(to_mm(g, x, y)[0]), y_mm=float(to_mm(g, x, y)[1]), z_mm=z,
                         diameter_mm=float("nan"), n_slices=1)
                r.update({c: float("nan") for c in CHARS})
                rows.append(r)
    cols = ["series_uid", "patient", "reader", "kind", "nodule_id", "x_mm", "y_mm", "z_mm",
            "diameter_mm", "n_slices"] + CHARS
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(scans)


def marks():
    return _build()[0].copy()


def scans():
    """One row per kept scan: series_uid, patient, n_readers (reading sessions), slice_spacing, xml."""
    return _build()[1].copy()


def match_lesions(marks_frame, tol=None):
    """Match marks in a frame (subset of marks()) into lesion groups greedily by distance.
    Returns: list of lists, each holding indices into marks_frame where marks of a lesion belong.
    Applies the same matching rule as nodules(): distance <= max(radii)/2 + tol_mm, no two from same reader."""
    tol = TOL_MM if tol is None else tol
    if len(marks_frame) == 0:
        return []
    xyz = marks_frame[["x_mm", "y_mm", "z_mm"]].to_numpy()
    rad = marks_frame.diameter_mm.fillna(0).to_numpy() / 2
    rd = marks_frame.reader.to_numpy()
    pairs = []
    for i in range(len(marks_frame)):
        for j in range(i + 1, len(marks_frame)):
            if rd[i] == rd[j]:
                continue
            dist = float(np.linalg.norm(xyz[i] - xyz[j]))
            if dist <= max(rad[i], rad[j]) + tol:
                pairs.append((dist, i, j))
    group = list(range(len(marks_frame)))
    members = {i: [i] for i in range(len(marks_frame))}
    for _, i, j in sorted(pairs):
        a, b = group[i], group[j]
        if a == b or {rd[k] for k in members[a]} & {rd[k] for k in members[b]}:
            continue
        for k in members[b]:
            group[k] = a
        members[a] += members.pop(b)
    return sorted(members.values(), key=lambda ix: min(ix))


def nodules(min_kind="nodule", tol=None):
    """One row per lesion, matched across readers as described in the module docstring.
    min_kind='nodule' uses marks of 3 mm or more; 'small' adds the under-3 mm points too."""
    tol = TOL_MM if tol is None else tol
    kinds = ["nodule"] if min_kind == "nodule" else ["nodule", "small"]
    m = marks()
    m = m[m.kind.isin(kinds)]
    out = []
    for uid, d in m.groupby("series_uid", sort=True):
        d = d.reset_index(drop=True)
        xyz = d[["x_mm", "y_mm", "z_mm"]].to_numpy()
        rad = d.diameter_mm.fillna(0).to_numpy() / 2
        rd = d.reader.to_numpy()
        pairs = []
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if rd[i] == rd[j]:
                    continue
                dist = float(np.linalg.norm(xyz[i] - xyz[j]))
                if dist <= max(rad[i], rad[j]) + tol:
                    pairs.append((dist, i, j))
        group = list(range(len(d)))
        members = {i: [i] for i in range(len(d))}
        for _, i, j in sorted(pairs):
            a, b = group[i], group[j]
            if a == b or {rd[k] for k in members[a]} & {rd[k] for k in members[b]}:
                continue
            for k in members[b]:
                group[k] = a
            members[a] += members.pop(b)
        for lesion, idx in enumerate(sorted(members.values(), key=lambda ix: min(ix))):
            g = d.iloc[idx]
            out.append({"series_uid": uid, "patient": g.patient.iloc[0], "lesion": lesion,
                        "x_mm": g.x_mm.mean(), "y_mm": g.y_mm.mean(), "z_mm": g.z_mm.mean(),
                        "diameter_mm": g.diameter_mm.mean(), "n_readers": len(g),
                        "readers": ",".join(str(r) for r in sorted(g.reader)),
                        "malignancy_median": g.malignancy.median(),
                        "mal_min": g.malignancy.min(), "mal_max": g.malignancy.max()})
    return pd.DataFrame(out, columns=["series_uid", "patient", "lesion", "x_mm", "y_mm", "z_mm", "diameter_mm",
                                      "n_readers", "readers", "malignancy_median", "mal_min", "mal_max"])


LUNA16 = os.path.join(paths.DATASETS, "public", "luna16")


def luna16_match(les):
    """Our lesions marked by at least 3 readers against LUNA16's annotations.csv (CC BY 4.0, Zenodo
    3723295), in its 888 scans (subsets.csv, read from the subset zips' own file lists). Both are in
    the scanner's millimetres, so a LUNA16 nodule is found when one of ours lies within its radius
    (at least 1.5 mm). None when the LUNA16 files are not there."""
    ann_p, sub_p = os.path.join(LUNA16, "annotations.csv"), os.path.join(LUNA16, "subsets.csv")
    if not (os.path.exists(ann_p) and os.path.exists(sub_p)):
        return None
    ann, sub = pd.read_csv(ann_p), pd.read_csv(sub_p)
    luna = set(sub.seriesuid)
    ours = les[(les.n_readers >= 3) & les.series_uid.isin(luna)]
    by = {u: d[["x_mm", "y_mm", "z_mm", "diameter_mm"]].to_numpy() for u, d in ours.groupby("series_uid")}
    have = set(les.series_uid) | set(scans().series_uid)
    found, dist, ddiam, used = 0, [], [], set()
    for a in ann[ann.seriesuid.isin(have)].itertuples():
        o = by.get(a.seriesuid)
        if o is None:
            continue
        d = np.linalg.norm(o[:, :3] - [a.coordX, a.coordY, a.coordZ], axis=1)
        i = int(d.argmin())
        if d[i] <= max(a.diameter_mm / 2, 1.5):
            found += 1
            dist.append(d[i])
            ddiam.append(o[i, 3] - a.diameter_mm)
            used.add((a.seriesuid, i))
    return {"luna_scans": len(luna), "luna_scans_here": len(luna & have),
            "luna_nodules_here": int(ann.seriesuid.isin(have).sum()), "luna_nodules": len(ann),
            "found": found, "ours_at_3": len(ours), "ours_unmatched": len(ours) - len(used),
            "median_distance_mm": float(np.median(dist)) if dist else float("nan"),
            "median_diameter_diff_mm": float(np.median(ddiam)) if ddiam else float("nan")}


def main():
    recs = series_records()
    ok, skipped = unpacked()
    files, rep = kept_files()
    sc, mk = scans(), marks()
    print("LIDC-IDRI scans: %d in series.json, %d kept (unpacked), %d skipped (download still in progress)"
          % (len(recs), len(ok), skipped))
    print("annotation files: %d kept; dropped %d not a CT series in series.json, %d duplicate reads of a series"
          " (fewer readingSessions or earlier name), %d unreadable"
          % (len(files), len(rep["not_ct"]), len(rep["duplicate"]), len(rep["unreadable"])))
    print("  unpacked scans with no annotation file: %d" % (len(ok) - len(sc)))
    n = sc.n_readers.clip(upper=4).value_counts().sort_index()
    print("readers per scan: " + ", ".join("%s%s: %d" % (k, "+" if k == 4 else "", v) for k, v in n.items()))
    print("marks by kind: " + ", ".join("%s %d" % (k, v) for k, v in mk.kind.value_counts().items()))
    les = nodules()
    print("lesions of 3 mm or more marked by at least: " + ", ".join(
        "%d: %d" % (k, (les.n_readers >= k).sum()) for k in (1, 2, 3, 4)))
    thin = set(sc.series_uid[sc.slice_spacing <= LUNA16_SPACING])
    t = les[les.series_uid.isin(thin)]
    print("scans with slices of %.1f mm or thinner: %d; lesions there marked by at least 3: %d (LUNA16 reference %d)"
          % (LUNA16_SPACING, len(thin), (t.n_readers >= 3).sum(), LUNA16_AT_3))
    m = luna16_match(les)
    if m:
        print("LUNA16's own list: %(luna_scans_here)d of its %(luna_scans)d scans here, holding %(luna_nodules_here)d of"
              " its %(luna_nodules)d nodules; %(found)d of those found among our %(ours_at_3)d lesions marked by"
              " 3 or more (%(ours_unmatched)d of ours unmatched); median distance %(median_distance_mm).2f mm,"
              " median diameter ours minus theirs %(median_diameter_diff_mm)+.2f mm" % m)


if __name__ == "__main__":
    main()
