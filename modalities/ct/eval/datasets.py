"""The CT dataset: where the scans are, which structures are labelled, and the guard that keeps the
test split shut.

One dataset, for now. The TotalSegmentator dataset v2.0.1 (CC BY 4.0, LICENSING.md) is 1,228
clinical CT scans resampled to 1.5 mm, each with expert masks of 117 anatomical structures. It lives
outside the repository, under `<Datasets>/public/totalsegmentator/`.

Two things here differ from the X-ray sets and both shape everything downstream:

**There is no patient identifier.** The published metadata names the scan, its study type, the
institute, the scanner and the subject's age and sex, and nothing that ties two scans to one person.
So a scan is the resampling unit, exactly as an image is for FracAtlas, and the scorecard has to say
so rather than implying a patient-level split it cannot do.

**The authors published their own split**, 1,082 train, 57 validation, 89 test. We keep their test
set as the locked test rather than cutting a new one. It is the honest choice twice over: it is a
split nobody here chose after seeing any result, and it is the split the published numbers for this
dataset were produced on, so a comparison means something.

Column names, structure names and strata are the contract in METRICS_SPEC.md; nothing here may
change them without changing that document first."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from core import paths  # noqa: E402

SEED = 20261001
PUBLIC = os.path.join(paths.DATASETS, "public")
NAME = "totalsegmentator"

SET = {
    "root": os.environ.get("TOTALSEGMENTATOR_ROOT", os.path.join(PUBLIC, "totalsegmentator")),
    "index": ("meta.csv", "utf-8-sig", ";"),
    "key": "image_id",
    "patient": None,                      # not published: the scan is its own unit
    "scan": "ct.nii.gz",
    "masks": "segmentations",
    "doi": "10.5281/zenodo.10047292",
    "cite": ("Wasserthal, Breit, Meyer, Pradella, Hinck, Sauter, Heye, Boll, Cyriac, Yang, Bach, "
             "Segeroth, 'TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT "
             "Images', Radiology: Artificial Intelligence 5(5) (2023), doi:10.1148/ryai.230024; "
             "dataset v2.0.1 doi:10.5281/zenodo.10047292, CC BY 4.0"),
    # stratum -> (column, "is"|"is not"|"contains", value). `all` is every in-scope scan.
    # Declared here, before anything is trained, because a stratum chosen after a result is not a
    # finding. Each one is a way the scans genuinely differ, not a way to find a flattering subset.
    "strata": {
        "abdomen": ("study_type", "contains", "abdomen"),
        "thorax": ("study_type", "contains", "thorax"),
        "head_or_neck": ("study_type", "contains", "neck"),
        "siemens": ("manufacturer", "is", "siemens"),
        "not_siemens": ("manufacturer", "is not", "siemens"),
        "main_institute": ("institute", "is", "I"),
        "other_institutes": ("institute", "is not", "I"),
        "with_pathology": ("pathology", "is not", "no_pathology"),
    },
}


def root():
    return SET["root"]


def index():
    """The published metadata, one row per scan, with the authors' own split kept as `given_split`.

    Raises rather than returning an empty frame when the dataset is not on this machine: a
    silently empty index would run every script to completion and score nothing."""
    name, encoding, sep = SET["index"]
    f = os.path.join(root(), name)
    if not os.path.exists(f):
        raise SystemExit(
            "The TotalSegmentator dataset is not at %s. Fetch it with "
            "`python modalities/ct/fetch_data.py all`, or point TOTALSEGMENTATOR_ROOT at a copy. "
            "It is CC BY 4.0; the licence row is in LICENSING.md." % root())
    d = pd.read_csv(f, sep=sep, encoding=encoding)
    d = d.rename(columns={"split": "given_split"})
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    for c in ("gender", "institute", "study_type", "manufacturer", "pathology"):
        d[c] = d[c].fillna("").astype(str).str.strip().str.lower()
    d["age_band"] = pd.cut(d["age"], [0, 40, 60, 75, 200],
                           labels=["under40", "40to59", "60to74", "75plus"]).astype(str)
    return d


def scan_path(image_id):
    return os.path.join(root(), str(image_id), SET["scan"])


def mask_path(image_id, structure):
    return os.path.join(root(), str(image_id), SET["masks"], "%s.nii.gz" % structure)


def structures():
    """The 117 structure names, read off disk once and sorted.

    From the data rather than a list typed here, because a list typed here is a list that can drift
    from the masks it claims to name. `inventory.py` checks every scan carries the same set."""
    for d in sorted(os.listdir(root())):
        folder = os.path.join(root(), d, SET["masks"])
        if os.path.isdir(folder):
            return sorted(n[:-7] for n in os.listdir(folder) if n.endswith(".nii.gz"))
    raise SystemExit("No segmentation folders under %s" % root())


def strata(table):
    """Yield (name, mask) pairs, `all` first, over the pre-declared subsets.

    The column is lower-cased and so is the value it is compared against. Both sides, every time:
    the first version lower-cased only the column, so `main_institute`, declared as `I`, never
    matched a single row and `other_institutes` quietly became every scan. Nothing crashed and the
    scorecard printed a plausible number for the wrong set of scans. The independent recomputation
    is what found it."""
    yield "all", pd.Series(True, index=table.index)
    for name, (column, how, value) in SET["strata"].items():
        col = table[column].astype(str).str.lower()
        want = str(value).lower()
        if how == "is":
            yield name, col == want
        elif how == "is not":
            yield name, col != want
        else:
            yield name, col.str.contains(want, regex=False)


def guard_test(split):
    """Refuse to look at the test split unless somebody has deliberately unlocked it.

    The same latch the X-ray work used. The test is opened once, by the final evaluation, and the
    way to open it is to set the variable in that one script rather than anywhere a stray run could
    reach it."""
    if split == "test" and os.environ.get("RAIDIOLOGY_OPEN_TEST") != "yes":
        raise SystemExit(
            "The CT test split is locked. It is opened once, by the final evaluation, which sets "
            "RAIDIOLOGY_OPEN_TEST itself. If you are developing, use `dev`.")
