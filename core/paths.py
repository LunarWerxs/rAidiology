"""Where things live, for every scan type.

The repository holds code and the made-up sample; everything big or private sits outside it, in
folders beside it (never committed): datasets, trained weights and installed tools. The git-ignored
work/ folder inside the repository holds intermediate files. Each location can be moved with an
environment variable.

A script reaches these by putting the repository root on sys.path and importing this module;
`use(<scan type>)` then lets it import its neighbours (and the shared DICOM and reading tools) by
module name."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BESIDE = os.path.dirname(ROOT)

WORK = os.environ.get("RAIDIOLOGY_WORK", os.path.join(ROOT, "work"))
DATASETS = os.environ.get("RAIDIOLOGY_DATASETS", os.path.join(BESIDE, "Datasets"))
MODELS = os.environ.get("RAIDIOLOGY_MODELS", os.path.join(BESIDE, "Models"))
TOOLS = os.environ.get("RAIDIOLOGY_TOOLS", os.path.join(BESIDE, "_tools"))
CASES = os.path.join(ROOT, "cases")          # private per-patient material; never in a public copy
SITE = os.path.join(ROOT, "site")
CORE_DICOM = os.path.join(ROOT, "core", "dicom")
CORE_READING = os.path.join(ROOT, "core", "reading")


def modality(name):
    """A scan type's folder, e.g. modality("spine_mri")."""
    return os.path.join(ROOT, "modalities", name)


def use(name):
    """Put a scan type's folders and the shared tools on the import path."""
    m = modality(name)
    for d in (CORE_DICOM, CORE_READING, m, os.path.join(m, "eval"), os.path.join(m, "sct")):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
