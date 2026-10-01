"""Find and classify every MR series on a patient disc from the DICOM headers alone, so the
pipeline works on any spine MRI export (not just the folder layout of one CD).

Each series gets: region (C / T / L / X), plane (sagittal / axial / coronal / mixed),
sequence (T1 / T2 / STIR / other), a stable name like "L_S03_SAG_T2", and its files in the
order they sit along the slice normal (the same order the NIfTI and the web stacks use)."""
import collections
import os
import re

import numpy as np
import pydicom

MR_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
REGION_WORDS = [("C", ("cervical", "c-spine", "c spine", "neck")),
                ("T", ("thoracic", "t-spine", "t spine", "dorsal")),
                ("L", ("lumbar", "l-spine", "l spine", "lumbosacral", "sacrum"))]


def region_of(d):
    text = " ".join(str(getattr(d, k, "")) for k in ("StudyDescription", "BodyPartExamined", "SeriesDescription", "ProtocolName")).lower()
    for code, words in REGION_WORDS:
        if any(w in text for w in words):
            return code
    return "X"


def sequence_of(d):
    desc = f"{getattr(d, 'SeriesDescription', '')} {getattr(d, 'ProtocolName', '')}".upper()
    if "STIR" in desc or "TIRM" in desc or (float(getattr(d, "InversionTime", 0) or 0) > 50 and float(getattr(d, "InversionTime", 0) or 0) < 250):
        return "STIR"
    if re.search(r"\bT1", desc):
        return "T1"
    if re.search(r"\bT2(?!\*)", desc) or "T2 " in desc:
        return "T2"
    tr, te = float(getattr(d, "RepetitionTime", 0) or 0), float(getattr(d, "EchoTime", 0) or 0)
    if tr and te:
        return "T2" if te > 60 else "T1" if tr < 1000 and te < 30 else "other"
    return "other"


def plane_of(orient):
    n = np.abs(np.cross(orient[:3], orient[3:]))
    return ("sagittal", "coronal", "axial")[int(np.argmax(n))]


def disc_files(cd_root):
    """Paths of the images on the disc: from its DICOMDIR index when present (the standard for
    patient CDs), otherwise every file under the folder."""
    dicomdir = os.path.join(cd_root, "DICOMDIR")
    if os.path.exists(dicomdir):
        from pydicom.fileset import FileSet
        try:
            return [str(inst.path) for inst in FileSet(pydicom.dcmread(dicomdir))]
        except Exception:
            pass
    return [os.path.join(dp, f) for dp, _, fn in os.walk(cd_root) for f in fn]


def scan(cd_root):
    groups = collections.defaultdict(list)
    for p in disc_files(cd_root):
        try:
            d = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception:
            continue
        if str(getattr(d, "SOPClassUID", "")) != MR_STORAGE or not hasattr(d, "ImageOrientationPatient"):
            continue
        groups[str(d.SeriesInstanceUID)].append((d, p))
    out = []
    for items in groups.values():
        d0 = items[0][0]
        planes = {plane_of(np.array(d.ImageOrientationPatient, float)) for d, _ in items}
        plane = planes.pop() if len(planes) == 1 else "mixed"
        o = np.array(d0.ImageOrientationPatient, float)
        n = np.cross(o[:3], o[3:])
        items.sort(key=lambda t: float(np.dot(n, np.array(t[0].ImagePositionPatient, float))))
        region, seq = region_of(d0), sequence_of(d0)
        desc = "".join(ch if ch.isalnum() else "_" for ch in str(getattr(d0, "SeriesDescription", "series"))).strip("_")
        out.append({
            "name": f"{region}_S{int(getattr(d0, 'SeriesNumber', 0) or 0):02d}_{desc}",
            "region": region, "plane": plane, "sequence": seq,
            "description": str(getattr(d0, "SeriesDescription", "")),
            "files": [p for _, p in items], "count": len(items),
            "study": str(getattr(d0, "StudyDescription", "")),
        })
    return sorted(out, key=lambda s: s["name"])


def pick(series, region, plane, sequence):
    """The series for (region, plane, sequence) with the most slices, or None."""
    c = [s for s in series if s["region"] == region and s["plane"] == plane and s["sequence"] == sequence]
    return max(c, key=lambda s: s["count"]) if c else None


REGION_NAME = {"C": "cervical", "T": "thoracic", "L": "lumbar", "X": "spine"}
