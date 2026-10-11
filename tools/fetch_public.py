"""The public datasets the owner approved on 2026-10-04, and the commands that fetch them.

Each row below has its LICENSING.md row (section "Public datasets approved for download, 2026-10-04", or the
CMMD and CBIS-DDSM row in the X-ray section), written before any download, and the licence ids that row
allows. The downloading itself is core/fetch.py; the plan is docs/research/data-download-plan-2026-10-04.md.

Where things go (owner, 2026-10-05: the two SSDs, E: and F:, take what is in use; later that day, the hard disk H:
may hold "things that are less immediately necessary. Especially during compression"):
- **The download:** `E:/rAidiology/archive/<name>/` holds the files as the archive serves them while they land,
  plus `fetch.json` (the licence as read, every file's size and checksum, what was checked).
- **The working copy:** every set is unpacked as soon as it lands, in the background while the next sets
  download, into `F:/rAidiology/Datasets/public/<name>/`, or onto E: once F: is short of room. Then its
  downloaded files are removed: the two SSDs cannot hold both copies, and fetch.json keeps each file's
  address and checksum, so any of them can be fetched again. A loose file or a DICOM image is moved, not
  copied. The first sets landed on H: before this; a spare copy of their download stays there (`archive_copy`).
- **The hard disk:** a working copy goes to `H:/rAidiology/cold/<name>/` when both SSDs are short of room, rather
  than waiting. `cold` moves a set no model needs soon onto H: and `warm` moves it back; each re-points the junction,
  so readers notice nothing but the speed.
- **Every drive, never full** (owner, 2026-10-05, the downloads outgrowing E:, F: and H:: "Use whatever you need. But
  don't make my drives all be fill"): D: joined the SSDs and the Plex server's movie drive, O: (over the network, a
  symbolic link where a junction cannot go), comes after H:. No drive is taken past 80% full (`fetch.FLOOR`; 90% was "still pretty
  full", owner 2026-10-06; H: is set apart, below); `claim` waits or moves on rather than cross it. `--to D:` sends a `cold`/`warm` move
  to one drive. A cold set's download lands on the first of H:, D: and O: with room for all of it.
- **The spare disk** (owner, 2026-10-06, "90% is still pretty full", then a 6 TB Seagate on USB as G:, kept for scan
  training data alone: "anything that doesn't need speed"): every drive now stops at 80% (H: since 2026-10-09 at 85%, below). G: comes first for every
  working copy, and for a download once H: has no room (below); the SSDs keep what a model reads (the `-train` copies and the sets the code
  names) and take the rest only once G: is short of room. `warm --to F:` brings a set to an SSD before a model
  needs it fast. Moved there: OMAMA's three files, the spare copies, NLST's training copy, and the finished sets
  no code reads.
- **H: as staging** (owner, 2026-10-09: "you can take H up to, like, 95% as long as you know you're gonna bring it
  back down ... I just don't want it to, like, stay over 85% after long periods of time"; `fetch.FLOORS`): what
  stays on H: stops at 85%. A download that passes through (unpacked onto another drive, then removed) may take it
  to 95%. So a download lands on H: first and is unpacked onto G:: G:, a shingled (SMR) disk on USB, slows to a
  crawl when a slow download writes beside an unpack (2026-10-09: 5.3 MB/s against 32.8 without them).
- **Kept packed** (`packed`): a set can stay as served on a cold disk, unpacked when a model needs it. None does
  now: OMAMA-DB's 796 GB `original_2d.tar` did while no drive had room for it twice over; on G: (6 TB, 45% full) it
  unpacks to 59% at most, and the found-crop readers cut their crops from its full-resolution pictures (2026-10-07).
- **The junction:** `<Datasets>/public/<name>` (core/paths.py) points at the working copy, so every reader
  finds the set where it always looks.
- **Smaller, losing nothing** (owner, 2026-10-05: store it far smaller, still fit to train on): once unpacked,
  every uncompressed DICOM image in the working copy is re-encoded as JPEG-LS lossless inside DICOM
  (core/store.py), each one checked bit for bit before it replaces the original; CT comes out 2 to 3 times
  smaller, an uncompressed mammogram 9 to 19 times. fetch.json's `shrunk` says what was done.

Inside a lane (one per archive) the most useful sets go first, the smaller first among equals; Zenodo's lane runs
six sets at once (one connection per file, six allowed) and TCIA's two, so their hosts' connections always have
files to work on.

Usage (`--log FILE` writes the run's log there, line by line):
  python tools/fetch_public.py plan [name ...|all]    read each licence and file list now; download nothing
  python tools/fetch_public.py fetch [name ...|all]   re-read the licence (stop if it changed), download,
                                                      check, write fetch.json, unpack a set under 20 GB
  python tools/fetch_public.py status                 what the archive holds
  python tools/fetch_public.py unpack name ...        unpack fetched sets into working copies and join them
  python tools/fetch_public.py shrink name|folder ... re-encode a set's (or any folder's) DICOM images losslessly
  python tools/fetch_public.py record                 write each landed set's folder into LICENSING.md
  python tools/fetch_public.py cold name ...          move working copies to the hard disk (only while unused)
  python tools/fetch_public.py warm name ...          move them back to an SSD
"""
import argparse
import concurrent.futures
import contextlib
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

# numpy's OpenBLAS sets aside buffers for every core when it is imported: 0.8 GB of committed memory per process
# on this 32-core PC, 0.1 GB with one thread (measured 2026-10-06). Nothing here does matrix maths.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import fetch, paths, store  # noqa: E402
from core.fetch import log  # noqa: E402

ARCHIVE = os.environ.get("RAIDIOLOGY_ARCHIVE", "E:/rAidiology/archive")
COLD = os.environ.get("RAIDIOLOGY_COLD", "H:/rAidiology/cold")
SHARE = os.environ.get("RAIDIOLOGY_SHARE", "O:/rAidiology/cold")   # the Plex server's movie drive, over the network
SPARE = os.environ.get("RAIDIOLOGY_SPARE", "G:/rAidiology/cold")   # the owner's spare 6 TB hard disk, on USB (2026-10-06)
COLD_ROOTS = [SPARE, COLD, SHARE]
WORKING = os.environ.get("RAIDIOLOGY_WORKING", SPARE + ";F:/rAidiology/Datasets/public;E:/rAidiology/Datasets/public;"
                         "D:/rAidiology/Datasets/public;" + COLD + ";" + SHARE).split(";")
#                          the spare disk first (nothing just unpacked needs speed; `warm` brings a set to an SSD for
#                          a model), then the SSDs, the hard disk and the share; each only down to its fetch.floor
CACHE_ROOTS = [os.path.join(r, "archive") for r in (COLD, SPARE, "D:/rAidiology/cold", SHARE)]   # where a download
#                          lands: H: first (staging, up to 95% for a download that passes through; then it is
#                          unpacked onto G:, which never sees the download's slow writes), then the spare disk
LINKS = os.path.join(paths.DATASETS, "public")
LICENSING = os.path.join(paths.ROOT, "LICENSING.md")
ARCHIVE_RESERVE = 40e9       # left free on the download disk
WORKING_RESERVE = 50e9       # left free on a working-copy disk
ROOM_WAIT_S = 6 * 3600       # how long a set waits for another to free room before it gives up
SAVE_S = 30
UNPACKERS = 2                # sets unpacked at once, beside the downloads
# Who goes first on a slow disk (fetch.Turn; 0 first, then the order the owner and the modelling chat gave on
# 2026-10-07): the original 2D mammograms' download (globus_fetch.py), then the mammograms in the order their readers
# want them (the whole-picture readers' 1024 copy, the crop readers' originals, the views nobody reads together yet),
# then everything else, oldest waiter first. A move off a disk within FULL_RANK_AT of its line goes at rank 1
# (move_tier): it frees the disk the models work from in minutes, where an unpack holds G: for hours (2026-10-07:
# D: at 79.7%, its two sets behind OMAMA's five-hour unpack); so does a set staged on H: past its 85% (unpack --via).
RANK = {"omama-db-2d-1024": 1, "omama-db-2d": 2, "omama-db-multiview": 3,
        "mammo-mx": 4}       # the modelling session, 2026-10-09: after queue195's crops, mammo-mx first of the rest
FULL_RANK_AT = 0.01          # of the drive, short of its fetch.floor


def rank(name):
    return RANK.get(name, 5)

BY4, BY3, CC0, SA4 = "cc-by-4.0", "cc-by-3.0", "cc0", "cc-by-sa-4.0"
# A Dataverse guestbook's answer, given per download: the company's public contact and name, nothing personal
# (the owner's leave of 2026-10-05 to sign agreements that allow commercial use).
GUESTBOOK = {"email": "contact@lunarwerx.com", "institution": "LunarWerx LLC"}
HDV = "https://dataverse.harvard.edu"
Z, F, M, T = "zenodo", "figshare", "mendeley", "tcia"


def S(name, area, value, archive, gb, link, licence, **kw):
    return {"name": name, "area": area, "value": value, "archive": archive, "gb": gb, "link": link,
            "licence": set(licence) if isinstance(licence, (set, tuple, list)) else {licence}, **kw}


ZR = "https://zenodo.org/records/"
MD = "https://data.mendeley.com/datasets/"
TC = "https://www.cancerimagingarchive.net/collection/"
TA = "https://www.cancerimagingarchive.net/analysis-result/"
FA = "https://figshare.com/articles/dataset/"
NEW = ("Added 2026-10-05 under the owner's standing go-ahead (\"Download and install whatever you want\"), after the "
       "second hunt's USE verdict")
SETTLED = ("Added 2026-10-05 under the owner's standing go-ahead, once its own record's licence, read at the source, "
           "settled the second hunt's CHECK (the owner's ruling: a dataset's own website licence governs)")
RECORD = ("Added 2026-10-05 under the owner's ruling of that day that the download record's licence governs, once "
          "that record's licence was re-read at the source")
SETTLED_RSNA = ("Added 2026-10-05 under the owner's standing go-ahead, once RSNA's own page, re-read that day, still "
                "allowed commercial use and linked the files on RSNA's public bucket")
RSNA_PNEUMONIA = "https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018"
RSNA_TERMS = ("You may access and use the imaging datasets and annotations for the purposes of academic research and "
              "education, and other commercial or non-commercial purposes as long as you meet the following attribution "
              "requirements")
SWEPT = ("Added 2026-10-05 under the owner's standing go-ahead, after the DataCite sweep's USE verdict, once its own "
         "record's licence and open access were re-read at the source")
HUNT3 = ("Added 2026-10-05 under the owner's standing go-ahead, after the third hunt, once its own record's licence was "
         "re-read at the source")
HUNT4 = ("Added 2026-10-05 after the fourth hunt, under the owner's word that evening (\"download more\"), once its own "
         "record's licence was re-read at the source")
ASKED = ("Added 2026-10-06 on the modelling chat's ask, under the owner's word that this session keeps it stocked, once "
         "its own record's licence was re-read at the source")
GAPS = ("Added 2026-10-11 for the modelling chat's nine weak lines, on the owner's word that day (\"fill the gaps that "
        "it's missing things on\"), once its own record's licence and open download were re-read at the source")
GPL3 = "other:gpl-3.0"   # GitHub's SPDX id as licence_id writes it: copyleft, commercial use allowed (owner, 2026-10-04:
                         # "I would probably assume share alike is okay")
SETS = [
    S("unitochest", "Lung CT", 4, Z, 84.1, ZR + "5797912", BY4, ids=[5797912]),
    S("multiomics-nodule-ct", "Lung CT", 4, Z, 215.0, ZR + "13908015", BY4, ids=[13908015, 13908030, 15043680]),
    S("cstl", "Lung CT", 3, Z, 47.0, ZR + "12582356", BY4, ids=[12582356, 12576633]),
    # Its record holds IDC manifests; the boxes themselves are DICOM SR objects in IDC's public AWS bucket.
    S("nlst-sybil", "Lung CT", 3, Z, None, ZR + "15643335", BY4, ids=[15643335], idc=r"-aws\.s5cmd$"),
    S("nlst-3dplus", "Lung CT", 3, Z, 0.0, ZR + "15320923", BY4, ids=[15320923]),
    # ldfc and hcfmrp-lung-nodule-ct raised 2026-10-06: the modelling chat's weakest line is nodule malignancy on CT,
    # and both carry pathology (ldfc: cancer type per nodule; hcfmrp: biopsy-proven NSCLC/SCLC with TNM stage).
    S("ldfc", "Lung CT", 4, Z, 16.9, ZR + "8422229", BY4, ids=[8422229, 11024613]),
    S("qiba-volct-1b", "Lung CT", 2, T, 0.4, TA + "qiba-volct-1b/", {BY4, BY3}, slug="qiba-volct-1b", kind="analysis-result"),
    S("mediastinal-lymph-node-seg", "Lung CT", 2, T, 35.3, TC + "mediastinal-lymph-node-seg/", BY4, slug="mediastinal-lymph-node-seg"),
    S("rider-lungct-seg", "Lung CT", 2, T, 1.0, TA + "rider-lungct-seg/", {BY3, BY4}, slug="rider-lungct-seg", kind="analysis-result"),
    S("lung-fused-ct-pathology", "Lung CT", 1, T, 27.2, TC + "lung-fused-ct-pathology/", BY3, slug="lung-fused-ct-pathology"),
    S("hcfmrp-lung-nodule-ct", "Lung CT", 3, Z, 36.9, ZR + "11435288", BY4, ids=[11435288]),
    S("iq-oth-nccd", "Lung CT", 1, M, None, MD + "bhmdr45bh2/4", BY4, ids=[("bhmdr45bh2", 4)]),
    S("ct-lymph-nodes", "Lung CT", 1, T, 58.4, TC + "ct-lymph-nodes/", BY3, slug="ct-lymph-nodes"),
    # Its page says CC BY 3.0; NBIA's series records said CC BY 4.0 on 2026-10-05 (LICENSING.md records both).
    S("qin-lungct-seg", "Lung CT", 1, T, 6.4, TA + "qin-lungct-seg/", {BY3, BY4}, slug="qin-lungct-seg", kind="analysis-result"),
    S("taix-ray", "Chest X-ray", 4, "hf", 62.0, "https://huggingface.co/datasets/TLAIM/TAIX-Ray", BY4, repo="TLAIM/TAIX-Ray"),
    S("synthetic-cxr-nodules", "Chest X-ray", 3, Z, 44.3, ZR + "10648433", BY4, ids=[10648433]),
    S("ptx-498", "Chest X-ray", 2, Z, 1.0, ZR + "8266529", BY4, ids=[8266529]),
    S("tbx11k", "Chest X-ray", 2, "page", None, "https://mmcheng.net/tb/", BY4, page="https://mmcheng.net/tb/",
      pattern=r"This dataset belongs to the Media Computing Lab at Nankai University and is licensed under a "
              r"Creative Commons Attribution 4\.0 License"),
    S("hcfmrp-covid-lid", "Chest X-ray", 2, Z, 11.3, ZR + "6841440", BY4, ids=[6841440]),
    S("bdcxr-3257", "Chest X-ray", 1, F, 0.9, FA + "_b_BDCXR-3257_b_A_Comprehensive_Chest_Radiography_Dataset_with_"
      "Multi-Etiological_Pneumonia_Labels_from_a_Clinical_Setting/32086362", BY4, ids=[32086362]),
    S("tb-cxr-kiran", "Chest X-ray", 1, M, None, MD + "8j2g3csprk/3", BY4,
      ids=[("8j2g3csprk", 3), ("7kb49nhfbn", None), ("8cm4ffk7b4", None)]),
    S("ssmu-pneumonia-xray-2", "Chest X-ray", 1, Z, 0.2, ZR + "18241637", BY4, ids=[18241637]),
    S("under5-pneumonia-nigeria", "Chest X-ray", 1, Z, 0.2, ZR + "14185822", BY4, ids=[14185822]),
    S("covid-19-ar", "Chest X-ray", 1, T, 19.9, TC + "covid-19-ar/", BY4, slug="covid-19-ar"),
    S("heart-cxr-czmn5", "Chest X-ray", 1, M, None, MD + "czmn5ypdz5/2", BY4, ids=[("czmn5ypdz5", 2)]),
    S("rsua-cxr", "Chest X-ray", 1, M, None, MD + "2jg8vfdmpm/1", BY4, ids=[("2jg8vfdmpm", 1)]),
    # The same pictures come in three formats (XNAT DICOM 29 GB, MIDS 12.5 GB, processed NumPy 12.2 GB); MIDS and
    # the tables are enough.
    S("cov-caldas", "Chest X-ray", 1, F, 5.6, "https://springernature.figshare.com/collections/Cov-Caldas_A_new_COVID-19_"
      "chest_X-Ray_Dataset_from_state_of_Caldas-Colombia/5833484", CC0, collection=5833484, exclude=[r"XNAT", r"NPY"]),
    S("corda", "Chest X-ray", 1, Z, 27.1, ZR + "7821611", BY4, ids=[7821611]),
    S("tomsk-pneumonia-xray", "Chest X-ray", 1, Z, 4.2, ZR + "5732746", BY4, ids=[5732746]),
    S("nih-ptx-device-cohort", "Chest X-ray", 1, Z, None, ZR + "22839698", BY4, ids=[22839698]),
    S("kauh-covid-cxr", "Chest X-ray", 1, M, None, MD + "xztwjmktrg/2", BY4, ids=[("xztwjmktrg", 2)]),
    S("spondylolisthesis-landmarks", "Lumbar X-ray", 4, M, 0.0, MD + "5jdfdgp762/1", BY4, ids=[("5jdfdgp762", 1)]),
    S("kauh-spine-xray", "Lumbar X-ray", 1, M, 0.1, MD + "xkt857dsxk/1", BY4, ids=[("xkt857dsxk", 1)]),
    S("lumbar-mri-radiologist-notes", "Lumbar MRI", 3, M, None, MD + "s6bgczr8s2/2", BY4, ids=[("s6bgczr8s2", 2)]),
    S("lumbar-mri-label-gt", "Lumbar MRI", 2, M, None, MD + "zbf6b4pttk/2", BY4, ids=[("zbf6b4pttk", 2)]),
    S("lumbar-mri-mid-ivd", "Lumbar MRI", 1, M, None, MD + "ggjtzh452d/1", BY4, ids=[("ggjtzh452d", 1)]),
    # The Zenodo record is a snapshot of pointer files; the images come from the project's own public mirror.
    S("spine-generic-multi", "Lumbar MRI", 1, "annex", None, ZR + "4299140", BY4, ids=[4299140],
      mirror="https://object-arbutus.alliancecan.ca/4f21233170e74c7d8653d790029d4ce2:def-jcohen-data-multi-subject"),
    S("lower-spine-t2-22304", "Lumbar MRI", 1, Z, 0.2, ZR + "22304", CC0, ids=[22304]),
    S("osf-lumbar-seg", "Lumbar MRI", 1, "osf", None, "https://osf.io/qx5rt/", BY4, node="qx5rt"),
    S("tompei-cmmd", "Mammography", 4, T, 0.0, TA + "tompei-cmmd/", BY4, slug="tompei-cmmd", kind="analysis-result",
      skip_sources=True),
    S("mammo-mx", "Mammography", 3, Z, 74.6, ZR + "17740027", BY4, ids=[17740027]),
    S("mexbreast", "Mammography", 3, M, 3.2, MD + "ym4mm9xvhm/2", BY4, ids=[("ym4mm9xvhm", 2)]),
    S("cdd-cesm", "Mammography", 3, T, 1.5, TC + "cdd-cesm/", BY4, slug="cdd-cesm"),
    S("kios-microcalc", "Mammography", 2, Z, 3.0, ZR + "14859694", BY4, ids=[14859694]),
    S("kios-masses", "Mammography", 2, Z, 3.3, ZR + "11446259", BY4, ids=[11446259]),
    S("birads45-dicom", "Mammography", 2, Z, 1.2, ZR + "10433163", BY4, ids=[10433163]),
    S("bcmid", "Mammography", 2, Z, 12.9, ZR + "19402690", BY4, ids=[19402690]),
    S("mammogram-mastery", "Mammography", 1, M, None, MD + "fvjhtskg93/1", BY4, ids=[("fvjhtskg93", 1)]),
    S("breastomo-synth", "Mammography", 1, Z, 0.1, ZR + "21450435", BY4, ids=[21450435]),
    # MIAS-DB.zip/.z01 redistribute MIAS, a CHECK set (owner, 2026-10-04: questionable sets are not fetched).
    S("snubh-mdb-uci", "Mammography", 1, Z, 1.6, ZR + "4950708", CC0, ids=[4950708], exclude=[r"mias"]),
    S("lumbar-vcf-mri", "Other", 2, Z, 1.3, ZR + "13274445", BY4, ids=[13274445]),
    S("cpaisd", "Other", 2, Z, 5.6, ZR + "10892316", BY4, ids=[10892316]),
    S("wisdom-ds", "Other", 2, F, 0.1, FA + "Wisdom_DS/31989366", BY4, ids=[31989366]),
    S("kub-stonex", "Other", 2, F, 20.8, FA + "_b_KUB-StoneX_b_Annotated_Kidney-Ureter-Bladder_X-ray_Dataset_with_Expert_"
      "Stone_Segmentations_and_Radiomic_Features/32049369", BY4, ids=[32049369]),
    S("garden-femoral-neck", "Other", 2, Z, 4.3, ZR + "22807611", BY4, ids=[22807611]),
    S("pengwin-task1", "Other", 2, Z, 7.6, ZR + "10927452", BY4, ids=[10927452]),
    S("knee-xray-gornale", "Other", 2, M, None, MD + "t9ndx37v5h/1", BY4, ids=[("t9ndx37v5h", 1)]),
    S("fumpe", "Other", 2, F, None, "https://figshare.com/collections/FUMPE/4107803", BY4, collection=4107803),
    S("head-cta-aneurysm", "Other", 2, Z, 25.6, ZR + "15697196", BY4, ids=[15697196]),
    S("brar-panoramic", "Other", 2, F, 0.5, FA + "BRAR-anchored_multimodal_dataset/30155974", BY4, ids=[30155974]),
    S("sah-ncct", "Other", 2, Z, 0.6, ZR + "8228847", BY4, ids=[8228847]),
    S("aariz-ceph", "Other", 1, F, 2.1, FA + "Aariz_Cephalometric_Dataset/27986417", BY4, ids=[27986417]),
    S("neurovision-ct", "Other", 1, M, None, MD + "mcgz47rsg2/2", BY4, ids=[("mcgz47rsg2", 2)]),
    S("stroke-multimodal", "Other", 1, Z, 3.9, ZR + "19703144", BY4, ids=[19703144]),
    S("ghana-brain-ct", "Other", 1, M, 0.1, MD + "g3zdkfpk3t/1", BY4, ids=[("g3zdkfpk3t", 1)]),
    S("dental-clinical", "Other", 1, F, 0.4, FA + "dental_clinical_dataset/32219928", BY4, ids=[32219928]),
    S("dental-panoramic-annotated", "Other", 1, F, 0.1, FA + "Annotated_Dental_Panoramic_Radiograph_Dataset/32846093", BY4,
      ids=[32846093]),
    S("dentiro", "Other", 1, F, 0.7, FA + "_b_DentIRO_b_A_High-Quality_Multi-Class_Single-Tooth_Intraoral_Radiograph_"
      "Dataset_for_Automated_Dental_Diagnosis/32086377", BY4, ids=[32086377]),
    S("dental-condition-panoramic", "Other", 1, Z, 1.4, ZR + "15487430", BY4, ids=[15487430]),
    S("spine-mets-ct-seg", "Other", 1, T, 20.4, TC + "spine-mets-ct-seg/", BY4, slug="spine-mets-ct-seg"),
    S("medpelvis3d", "Other", 1, Z, 31.4, ZR + "21757686", BY4, ids=[21757686]),
    S("chestpathct5", "Other", 1, Z, 9.4, ZR + "18256797", CC0, ids=[18256797]),
    S("opg-teeth-akuh", "Other", 1, Z, 0.5, ZR + "10538750", BY4, ids=[10538750]),
    S("sts-tooth", "Other", 1, Z, 31.8, ZR + "10597292", BY4, ids=[10597292]),
    S("periapical-lesion-ankara", "Other", 1, Z, 0.5, ZR + "18824283", BY4, ids=[18824283]),
    S("knee-osteoporosis", "Other", 1, M, 0.2, MD + "fxjm8fb6mw/2", BY4, ids=[("fxjm8fb6mw", 2)]),
    S("panoramic-dental-brahmi", "Other", 0, M, None, MD + "73n3kz2k4k/3", BY4, ids=[("73n3kz2k4k", 3)]),
    S("cmmd", "Mammography", 3, T, 22.9, TC + "cmmd/", BY4, slug="cmmd"),
    S("cbis-ddsm", "Mammography", 3, T, 163.6, TC + "cbis-ddsm/", BY3, slug="cbis-ddsm", row=TC + "cmmd/"),
    # The owner's share-alike ruling covers the data README's CC BY-SA 4.0; the record says CC BY 4.0.
    S("isles-2022", "Other", 2, Z, 1.7, ZR + "7153326", {BY4, SA4}, ids=[7153326]),
    # The second hunt (docs/research/data-hunt-2026-10-05.md): TCIA sets whose every download is CC BY.
    # (Spinal-Multiple-Myeloma-SEG and the EAY131 tumour annotations are held as training copies instead:
    # tools/train_copy.py, with their labels kept as served.)
    S("pediatric-ct-seg", "CT organs", 4, T, 65.7, TC + "pediatric-ct-seg/", BY4, slug="pediatric-ct-seg", approved=NEW),
    S("pancreas-ct", "CT organs", 3, T, 10.0, TC + "pancreas-ct/", BY3, slug="pancreas-ct", approved=NEW),
    S("colorectal-liver-metastases", "CT organs", 3, T, 10.9, TC + "colorectal-liver-metastases/", BY4, slug="colorectal-liver-metastases", approved=NEW),
    S("prostate-anatomical-edge-cases", "CT organs", 3, T, 17.1, TC + "prostate-anatomical-edge-cases/", BY4, slug="prostate-anatomical-edge-cases", approved=NEW),
    S("hcc-tace-seg", "CT organs", 3, T, 28.6, TC + "hcc-tace-seg/", BY4, slug="hcc-tace-seg", approved=NEW),
    S("ct-images-in-covid-19", "Lung CT", 3, T, 14.7, TC + "ct-images-in-covid-19/", BY4, slug="ct-images-in-covid-19", approved=NEW),
    S("cptac-ccrcc-tumor-annotations", "CT lesions", 2, T, 23.7, TA + "cptac-ccrcc-tumor-annotations/", BY4, slug="cptac-ccrcc-tumor-annotations", kind="analysis-result", approved=NEW),
    S("cptac-pda-tumor-annotations", "CT lesions", 2, T, 21.3, TA + "cptac-pda-tumor-annotations/", BY4, slug="cptac-pda-tumor-annotations", kind="analysis-result", approved=NEW),
    S("cptac-ucec-tumor-annotations", "CT lesions", 2, T, 12.0, TA + "cptac-ucec-tumor-annotations/", BY4, slug="cptac-ucec-tumor-annotations", kind="analysis-result", approved=NEW),
    S("stageii-colorectal-ct", "CT organs", 2, T, 7.3, TC + "stageii-colorectal-ct/", BY4, slug="stageii-colorectal-ct", approved=NEW),
    S("pedi-cranial-ct-healthy", "CT organs", 2, T, 3.2, TC + "pedi-cranial-ct-healthy/", BY4, slug="pedi-cranial-ct-healthy", approved=NEW),
    S("adrenal-acc-ki67-seg", "CT lesions", 2, T, 9.9, TC + "adrenal-acc-ki67-seg/", BY4, slug="adrenal-acc-ki67-seg", approved=NEW),
    S("prostatex", "Other", 2, T, 16.7, TC + "prostatex/", BY3, slug="prostatex", approved=NEW),
    S("pretreat-metstobrain-masks", "Other", 2, T, 1.7, TC + "pretreat-metstobrain-masks/", BY4, slug="pretreat-metstobrain-masks", approved=NEW),
    S("vestibular-schwannoma-mc-rc2", "Other", 2, T, 6.0, TC + "vestibular-schwannoma-mc-rc2/", BY4, slug="vestibular-schwannoma-mc-rc2", approved=NEW),
    S("mu-glioma-post", "Other", 2, T, 11.0, TC + "mu-glioma-post/", BY4, slug="mu-glioma-post", approved=NEW),
    S("brain-mets-lung-mri-path-segs", "Other", 2, T, 19.2, TC + "brain-mets-lung-mri-path-segs/", BY4, slug="brain-mets-lung-mri-path-segs", approved=NEW),
    S("prostate-diagnosis", "Other", 1, T, 5.7, TC + "prostate-diagnosis/", BY3, slug="prostate-diagnosis", approved=NEW),
    S("isbi-mr-prostate-2013", "Other", 1, T, 0.6, TA + "isbi-mr-prostate-2013/", BY3, slug="isbi-mr-prostate-2013", kind="analysis-result",
      skip_sources=True, approved=NEW),
    S("prostate-3t", "Other", 1, T, 0.3, TC + "prostate-3t/", BY3, slug="prostate-3t", approved=NEW),
    S("breast-lesions-usg", "Other", 1, T, 0.1, TC + "breast-lesions-usg/", BY4, slug="breast-lesions-usg", approved=NEW),
    # CHECK sets settled at their own source (docs/research/data-hunt-2026-10-05.md): the record's licence governs.
    S("chestdr", "Chest X-ray", 3, F, 15.4, "https://springernature.figshare.com/articles/dataset/ChestDR_Thoracic_Diseases_"
      "Screening_in_Chest_Radiography/22302775", CC0, ids=[22302775], approved=SETTLED),
    S("c4kc-kits", "CT organs", 2, T, 40.7, TC + "c4kc-kits/", BY3, slug="c4kc-kits", approved=SETTLED),
    S("bstt-ct", "Lung CT", 1, Z, 7.3, ZR + "18862454", BY4, ids=[18862454], approved=SETTLED),
    S("platif", "Other", 1, Z, 6.0, ZR + "18007397", BY4, ids=[18007397], approved=SETTLED),
    S("cxr-pneumonia-bhopal", "Chest X-ray", 1, M, 0.2, MD + "kpg5yz77gj/1", BY4, ids=[("kpg5yz77gj", 1)], approved=SETTLED),
    S("chest-imaging-cough-multan", "Chest X-ray", 1, M, 0.0, MD + "y6dvssx73b/1", BY4, ids=[("y6dvssx73b", 1)], approved=SETTLED),
    S("aisslab-breast", "Mammography", 1, M, 0.6, MD + "zp8yfhvndv/2", BY4, ids=[("zp8yfhvndv", 2)], approved=SETTLED),
    S("knee-mri-meniscus", "Other", 1, Z, 0.7, ZR + "16939444", BY4, ids=[16939444], approved=SETTLED),
    S("isles-2015-2018", "Other", 1, Z, 7.4, ZR + "17736412", BY4, ids=[17736412], approved=SETTLED),
    S("periapical-panoramic", "Other", 1, M, 8.9, MD + "kx52tk2ddj/3", BY4, ids=[("kx52tk2ddj", 3)], approved=SETTLED),
    # The DataCite sweep (docs/research/data-hunt-2026-10-05.md, pass 3), each record's licence re-read at its source.
    S("totalsegmentator-v3", "CT organs", 5, Z, 37.4, ZR + "22688904", BY4, ids=[22688904], approved=SWEPT),
    S("amos", "CT organs", 4, Z, 24.2, ZR + "7262581", BY4, ids=[7262581], approved=SWEPT),
    S("ts-liver-lesions", "CT organs", 4, Z, 38.9, ZR + "20272572", BY4, ids=[20272572], approved=SWEPT),
    S("barts-crl", "CT organs", 3, Z, 20.3, ZR + "14004841", BY4, ids=[14004841], approved=SWEPT),
    S("hvm-liver-vessels", "CT organs", 3, Z, 13.5, ZR + "19885789", BY4, ids=[19885789], approved=SWEPT),
    S("radboud-kidney-abnormality", "CT organs", 3, Z, 38.4, ZR + "8014290", BY4, ids=[8014290], approved=SWEPT),
    S("ts-pleural-pericard-effusion", "Lung CT", 3, Z, 9.0, ZR + "20272295", BY4, ids=[20272295], approved=SWEPT),
    # the makers' own Dropbox copy, linked from the record's description, comes faster than Zenodo
    S("totalsegmentator-mri-v3", "Other", 3, Z, 15.2, ZR + "22688334", BY4, ids=[22688334], approved=SWEPT,
      copies={"TotalsegmentatorMRI_dataset_v300.zip": "https://www.dropbox.com/scl/fi/1ihfp6spms0aqd21kxi1a/"
              "TotalsegmentatorMRI_dataset_v300.zip?rlkey=mgsc93715ihzfla9l4pqjwj18&st=n2rr0qln&dl=1"}),
    S("coala-crlm", "CT organs", 2, Z, 4.0, ZR + "14858922", BY4, ids=[14858922], approved=SWEPT),
    S("lumvbcanseg", "CT organs", 2, Z, 10.1, ZR + "8181250", BY4, ids=[8181250], approved=SWEPT),
    S("ts-liver-lesions-mr", "Other", 1, Z, 20.5, ZR + "20272348", BY4, ids=[20272348], approved=SWEPT),
    S("ts-liver-segments-mr", "Other", 1, Z, 1.1, ZR + "20746820", BY4, ids=[20746820], approved=SWEPT),
    S("ts-hip-implant", "CT organs", 1, Z, 9.0, ZR + "20272031", BY4, ids=[20272031], approved=SWEPT),
    S("lumase", "CT organs", 1, Z, 2.9, ZR + "7181338", BY4, ids=[7181338], approved=SWEPT),
    S("covid-ct-md", "Lung CT", 2, F, 11.3, "https://springernature.figshare.com/articles/dataset/COVID-CT-MD_COVID-19_Computed_"
      "Tomography_Scan_Dataset_Applicable_in_Machine_Learning_and_Deep_Learning/12991592", CC0, ids=[12991592], approved=SWEPT),
    S("aaa-ct-siriapisith", "CT organs", 2, F, 2.6, FA + "A_3D_deep_learning_approach_incorporating_coordinate_information_to_improve_the_"
      "segmentation_of_pre-_and_post-operative_abdominal_aortic_aneurysm/19090052", BY4, ids=[19090052], approved=SWEPT),
    S("covid-ct-rate", "Lung CT", 1, F, 0.7, FA + "COVID-CT-Rate_zip/18339416", BY4, ids=[18339416], approved=SWEPT),
    S("spgc-covid", "Lung CT", 1, F, 5.2, FA + "SPGC-COVID_Dataset/16632397", BY4, ids=[16632397], approved=SWEPT),
    S("mammo-density", "Mammography", 2, M, 0.2, MD + "tdx3h2fn9v/5", BY4, ids=[("tdx3h2fn9v", 5)], approved=SWEPT),
    S("coloncancer-ct-2025", "CT organs", 1, M, 1.4, MD + "sz87vvc875/3", BY4, ids=[("sz87vvc875", 3)], approved=SWEPT),
    S("covid19-isl-ct", "Lung CT", 1, M, 4.2, MD + "mbnczn5rw3/1", BY4, ids=[("mbnczn5rw3", 1)], approved=SWEPT),
    # Radiologists' pneumonia boxes on NIH ChestX-ray14 pictures, from RSNA's own bucket (no Kaggle account).
    S("rsna-pneumonia", "Chest X-ray", 3, "page", 7.0, RSNA_PNEUMONIA, {fetch.licence_id(RSNA_TERMS)}, page=RSNA_PNEUMONIA,
      pattern=re.escape(RSNA_TERMS), links=r"https://s3\.amazonaws\.com/east1\.public\.rsna\.org/AI/2018/[^\"'\s<>]+",
      approved=SETTLED_RSNA),
    # Graded lumbar stenosis from Firat University Hospital: 500 patients' axial T2 volumes, and 500 sagittal (v4 only).
    S("lumbar-axial-stenosis-firat", "Lumbar MRI", 5, M, 2.9, MD + "wjxkrkxhb8/1", BY4, ids=[("wjxkrkxhb8", 1)],
      approved=RECORD),
    S("lss-sagittal-aisslab", "Lumbar MRI", 4, M, 2.0, MD + "rgb77xm3jf/4", BY4, ids=[("rgb77xm3jf", 4)],
      approved=RECORD),
    # NLSTSeg from IDC rather than Zenodo: its DICOM re-release (masks and radiomics) and the NLST CTs they are drawn
    # on, from IDC's AWS bucket, much faster than Zenodo's NIfTI copy
    S("nlstseg", "Lung CT", 3, "idc", 55.0, "https://doi.org/10.5281/zenodo.17362624", BY4, result="nlstseg",
      images=True, approved=RECORD),
    S("mopg7-dental", "Other", 2, M, 1.7, MD + "t66ytdrhf2/2", BY4, ids=[("t66ytdrhf2", 2)], approved=RECORD),
    # Dryad keeps its files behind a bot check this client never passes: the owner passes it in a browser window
    # and the files are saved from that window (fetch.json is written beside them), so `fetch` has nothing to do here.
    S("lunit-tb-dryad", "Chest X-ray", 3, "browser", 20.9, "https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbpcv",
      CC0, approved="Added 2026-10-05; Dryad's record says CC0 1.0"),
    # OMAMA-DB on Harvard Dataverse (MIT on the record; guestbook answered per download). The 2D originals land on
    # the hard disk; the 1,000-image sample first, to measure what the originals are before 796 GB is unpacked.
    S("omama-db-sample", "Mammography", 5, "dataverse", 4.6, HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/KXJCIU",
      "other:mit", server=HDV, pid="doi:10.7910/DVN/KXJCIU", only=r"^2d_only_1000_images\.tar$", approved=RECORD),
    S("omama-db-2d", "Mammography", 5, "dataverse", 796.0, HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/KXJCIU",
      "other:mit", server=HDV, pid="doi:10.7910/DVN/KXJCIU", only=r"^original_2d\.tar$", cold=True, approved=RECORD),
    S("omama-db-multiview", "Mammography", 4, "dataverse", 79.5, HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/HYFOWN",
      "other:mit", server=HDV, pid="doi:10.7910/DVN/HYFOWN", only=r"^original_studylevel\.tar$", cold=True,
      row=HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/KXJCIU", approved=RECORD),   # OMAMA-DB's one row
    # The same 2D mammograms at 1024x1024 (105 GB): a usable copy while the originals wait on Harvard's tape recall
    # (the originals sat at 0 GB for over an hour on 2026-10-05; Harvard asks for single recalls under 250 GB)
    S("omama-db-2d-1024", "Mammography", 4, "dataverse", 105.0, HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/KXJCIU",
      "other:mit", server=HDV, pid="doi:10.7910/DVN/KXJCIU", only=r"^1024x1024\.tar$", cold=True,
      row=HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/KXJCIU", approved=RECORD),
    # The four-view studies at 256x256: the one copy of them Harvard keeps outside Globus (its S3 store), so a plain
    # download reaches it; the 512, 1024 and original copies answer a guestbook-signed link with 404 (2026-10-05)
    S("omama-db-multiview-256", "Mammography", 2, "dataverse", 0.7, HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/HYFOWN",
      "other:mit", server=HDV, pid="doi:10.7910/DVN/HYFOWN", only=r"^256x256_studylevel\.tar$",
      row=HDV + "/dataset.xhtml?persistentId=doi:10.7910/DVN/KXJCIU", approved=RECORD),
    # The third hunt (2026-10-05): each record's licence re-read at its source (DataCite, TCIA's API, the AWS registry).
    S("utsw-glioma", "Other", 3, T, 22.9, TC + "utsw-glioma/", BY4, slug="utsw-glioma", approved=HUNT3),
    S("ucsd-ptgbm", "Other", 3, T, 45.0, TC + "ucsd-ptgbm/", BY4, slug="ucsd-ptgbm", approved=HUNT3),
    S("hcuch-recist-ct", "CT lesions", 3, Z, None, ZR + "17788162", BY4, ids=[17788162], approved=HUNT3),
    S("bus-cot", "Mammography", 3, F, 6.9, "https://doi.org/10.6084/m9.figshare.30838715", BY4, ids=[30838715],
      approved=HUNT3),
    # ENHANCE.PET's licence differs by site: AutoPET's subjects are CC BY-NC (left out), Leipzig's and Careggi's
    # CC BY 4.0 (kept, by CT-details.xlsx's Data-Source). Their CT and its CT-derived masks; the PET volumes stay.
    S("enhance-pet-ct", "CT organs", 3, "s3", None, "https://registry.opendata.aws/enhance-pet-1-6k/", BY4,
      bucket="enhance-pet-1-6k",
      page="https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/enhance-pet-1-6k.yaml",
      pattern=r"University Hospital Leipzig: CC BY 4\.0 - Azienda Ospedaliero Universitaria Careggi: CC BY 4\.0",
      prefixes=["CT-details.xlsx", "PT-details.xlsx", "labels.json", "imaging-data/images/CT/",
                "imaging-data/ground-truth/"],
      subjects={"xlsx": "CT-details.xlsx", "id": "Patient", "source": "Data-Source", "key": r"/(\d{4})\.nii\.gz$",
                "keep": ["University Hospital Leipzig", "Azienda Ospedalierio Universitaria Careggi"]},
      approved=HUNT3),
    S("mslesseg", "Other", 2, F, 2.8, "https://doi.org/10.6084/m9.figshare.27919209", BY4, ids=[27919209],
      approved=HUNT3),
    S("ms-consensus-lesions", "Other", 2, M, None, MD + "8bctsm8jz7/1", BY4, ids=[("8bctsm8jz7", 1)], approved=HUNT3),
    S("bus-uclm", "Mammography", 2, M, None, MD + "7fvgj4jsp7/3", BY4, ids=[("7fvgj4jsp7", 3)], approved=HUNT3),
    S("openswisshcc", "Other", 2, Z, None, ZR + "21992461", BY4, ids=[21992461], approved=HUNT3),
    S("fracture-xray-bd", "Other", 2, M, None, MD + "cd6jjsxz44/4", BY4, ids=[("cd6jjsxz44", 4)], approved=HUNT3),
    S("birads-paraguay", "Mammography", 2, Z, None, ZR + "18946198", BY4, ids=[18946198], approved=HUNT3),
    S("pedims", "Other", 1, F, 1.8, "https://doi.org/10.6084/m9.figshare.28701065", BY4, ids=[28701065],
      approved=HUNT3),
    S("endometriosis-us", "Other", 1, F, 0.1, "https://doi.org/10.6084/m9.figshare.31927908", CC0, ids=[31927908],
      approved=HUNT3),
    S("ms-cervical-cord", "Other", 1, M, None, MD + "ydkrtmygjp/1", BY4, ids=[("ydkrtmygjp", 1)], approved=HUNT3),
    S("instrumented-cord-mri", "Other", 1, M, None, MD + "sjpx7md7cf/1", BY4, ids=[("sjpx7md7cf", 1)],
      approved=HUNT3),
    S("dental-rvg", "Other", 1, M, None, MD + "7pz9rrrr9n/1", BY4, ids=[("7pz9rrrr9n", 1)], approved=HUNT3),
    S("tat-det-us", "Other", 1, Z, None, ZR + "21861476", BY4, ids=[21861476], approved=HUNT3),
    S("iugc-2025", "Other", 1, Z, None, ZR + "17355570", BY4, ids=[17355570], approved=HUNT3),
    S("t1-dark-rim-ms", "Other", 1, Z, None, ZR + "17940367", BY4, ids=[17940367], approved=HUNT3),
    S("deepcac2-nlst", "Lung CT", 1, Z, None, ZR + "19946683", BY4, ids=[19946683], approved=HUNT3),
    # The fourth hunt (2026-10-05, evening): prostate MRI in place of UCSF's, and breast ultrasound. Each download's
    # licence is checked against TCIA's API (DataCite for the Zenodo record) by the fetch itself.
    S("prostate-mri-us-biopsy", "Other", 4, T, 80.2, TC + "prostate-mri-us-biopsy/", BY4, slug="prostate-mri-us-biopsy",
      approved=HUNT4),
    S("prostate-mri", "Other", 2, T, 3.6, TC + "prostate-mri/", BY3, slug="prostate-mri", approved=HUNT4),
    S("prostatex-seg-zones", "Other", 2, T, 0.7, TA + "prostatex-seg-zones/", BY3, slug="prostatex-seg-zones",
      kind="analysis-result", skip_collection_downloads=True, approved=HUNT4),
    S("bus-bra", "Mammography", 3, Z, None, ZR + "8231412", BY4, ids=[8231412], approved=HUNT4),
    # IDC's analysis results, read the same evening: expert labels for sets held or being fetched, from IDC's bucket
    S("lung-pet-ct-dx", "Lung CT", 3, T, 124.2, TC + "lung-pet-ct-dx/", BY4, slug="lung-pet-ct-dx", approved=HUNT4),
    S("lung-pet-ct-dx-annotations", "Lung CT", 2, "idc", 0.05, "https://doi.org/10.5281/zenodo.16989819", BY4,
      result="lung_pet_ct_dx_annotations", approved=HUNT4),
    S("prostate-mri-us-biopsy-seg", "Other", 3, "idc", 1.72, "https://doi.org/10.5281/zenodo.10069910", BY4,
      result="prostate_mri_us_biopsy_dicom_annotations", approved=HUNT4),
    S("prostatex-targets", "Other", 2, "idc", 0.01, "https://doi.org/10.5281/zenodo.15643312", BY4,
      result="prostatex_targets", approved=HUNT4),
    S("prostatex-seg-hires", "Other", 1, "idc", 0.12, "https://doi.org/10.7937/tcia.2019.deg7zg1u", BY3,
      result="prostatex_seg_hires", approved=HUNT4),
    # UCSF's portal sets (RMaC kidney CT, ProstateMR, PCNSL) are left out: the AWS registry says CC BY 4.0, but the
    # portal, the only way to the files, makes each downloader accept UCSF's licence, whose clause 1.h bars commercial use
    # The modelling chat's ask for organ labels beyond TotalSegmentator's (2026-10-06). Of the Decathlon only the four
    # CT tasks scanned at MSKCC: its liver task is a subset of LiTS (non-commercial) and its lung task came from TCIA.
    S("msd-mskcc-ct", "CT organs", 3, "s3", 29.5, "https://registry.opendata.aws/msd/", SA4, bucket="msd-for-monai",
      page="https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/msd.yaml",
      pattern=r"CC-BY-SA 4\.0 International",
      prefixes=["Task07_Pancreas.tar", "Task08_HepaticVessel.tar", "Task09_Spleen.tar", "Task10_Colon.tar",
                "license.txt"], approved=ASKED),
    S("lctsc", "CT organs", 3, T, 5.2, TC + "lctsc/", BY3, slug="lctsc", approved=ASKED),
    S("saros", "CT organs", 2, T, 0.1, TA + "saros/", BY4, slug="saros", kind="analysis-result", skip_sources=True,
      approved=ASKED),
    # Its record says CC BY 4.0, and its description that the organ labels come from AbdomenCT-1K ("research purpose
    # only"): the record governs (owner, 2026-10-05). Its 50 CTs are the Decathlon's MSKCC pancreas scans, not LiTS's.
    S("flare22-labelled", "CT organs", 3, Z, None, ZR + "7860267", BY4, ids=[7860267], approved=ASKED),
    S("pancreatic-ct-cbct-seg", "CT organs", 2, T, 14.3, TC + "pancreatic-ct-cbct-seg/", BY4,
      slug="pancreatic-ct-cbct-seg", approved=ASKED),
    # The gap hunt of 2026-10-11 (the modelling chat's nine weak lines), each record re-read at its source that day.
    # LNDb's record says CC BY 4.0 and its description CC BY-NC-ND: the record governs (owner, 2026-10-05); its
    # 2026-10-03 rejection read only the description.
    S("lndb", "Lung CT", 5, Z, 30.3, ZR + "8348419", BY4, ids=[8348419], approved=GAPS),
    S("cradi", "Chest X-ray", 4, Z, 14.6, ZR + "5493595", BY4, ids=[5493595], approved=GAPS),
    S("pediurf", "Other", 4, F, 2.3, "https://doi.org/10.6084/m9.figshare.29998954", BY4, ids=[29998954],
      approved=GAPS),
    # The authors' own repository: 222 patients' sagittal T2 slices with a box per disc (Pfirrmann P1-P5, bulge,
    # protrusion, extrusion, HIZ, Schmorl's nodes), First Affiliated Hospital of Jishou University; only its data.
    S("yolov9-aid-lumbar", "Lumbar MRI", 4, "github", 0.4, "https://github.com/JackChow666/YOLOv9-AID", GPL3,
      repo="JackChow666/YOLOv9-AID", keep=("datasets", "LICENSE.md", "README.md"), approved=GAPS),
    S("irbid-canal-stenosis-labels", "Lumbar MRI", 2, Z, 0.001, ZR + "23298064", BY4, ids=[23298064], approved=GAPS),
]
BY_NAME = {s["name"]: s for s in SETS}


def listing(s, ctx):
    name, a = s["name"], s["archive"]
    if a == Z:
        lst = fetch.zenodo(name, s["ids"], ctx, s.get("exclude", ()))
        if s.get("idc"):                           # the record's licence covers its IDC objects: check it first
            fetch.check_licence(name, lst.licence, s["licence"])
        for e in [e for e in lst.files if s.get("idc") and re.search(s["idc"], e.name)]:
            objs = fetch.idc_objects(name, ctx, fetch.get_bytes(e.url, ctx).decode("utf-8", "replace"))
            lst.files += objs
            lst.notes.append("%s: %d objects in IDC's public bucket" % (e.name, len(objs)))
        fetch.faster_copies(lst, s.get("copies"))
        return lst
    if a == F:
        return fetch.figshare(name, ctx, s.get("ids", ()), s.get("collection"), s.get("exclude", ()))
    if a == "annex":
        return fetch.git_annex(name, ctx, s["ids"][0], s["mirror"])
    if a == M:
        parts = []
        for ds, v in s["ids"]:
            part = fetch.mendeley(name, ctx, ds, v)
            if len(s["ids"]) > 1:
                for e in part.files:
                    e.name = "%s/%s" % (ds, e.name)
            parts.append(part)
        return fetch._merge(name, parts)
    if a == "hf":
        return fetch.huggingface(name, ctx, s["repo"])
    if a == "osf":
        return fetch.osf(name, ctx, s["node"])
    if a == "github":
        return fetch.github(name, ctx, s["repo"], s["keep"])
    if a == "page":
        return fetch.lab_page(name, ctx, s["page"], s["pattern"], s.get("links"))
    if a == T:
        return fetch.tcia(name, ctx, s["slug"], s.get("kind", "collection"), s["licence"], s.get("skip_sources", False))
    if a == "browser":
        raise fetch.FetchError("saved only from a browser window the owner opens, past the archive's bot check")
    if a == "dataverse":
        return fetch.dataverse(name, ctx, s["server"], s["pid"], GUESTBOOK, s.get("only"))
    if a == "idc":
        return fetch.idc_analysis(name, ctx, s["result"], s["licence"], s.get("images", False))
    if a == "s3":
        return fetch.public_bucket(name, ctx, s["bucket"], s["page"], s["pattern"], s["prefixes"], s.get("subjects"))
    raise SystemExit("%s: no reader for archive %r" % (name, a))


def checked_licence(s, lst):
    """The licence the archive states now, against the row; a TCIA or IDC set is checked per download and series."""
    if s["archive"] in (T, "idc"):
        ids = {fetch.licence_id(x) for x in lst.licence.split(" + ") if x}
        if not ids or ids - set(s["licence"]):
            raise fetch.LicenceChanged("%s: %s now says %r (%s); LICENSING.md allows %s. Stopped; check the row."
                                       % (s["name"], s["archive"].upper(), lst.licence, sorted(ids), sorted(s["licence"])))
        return sorted(ids)
    return [fetch.check_licence(s["name"], lst.licence, s["licence"])]


LANE_WIDTH = {Z: 16, T: 2}   # sets at once in a lane; the others run one at a time. Zenodo paces each connection,
                             # so sixteen sets at once, sixteen files each, fill its 100-connection cap (Host enforces it)


AS_GIVEN = False   # --as-given: a lane keeps the order the sets were named in


def lanes(sets):
    """One lane per archive, so the hosts download side by side; inside a lane the most useful sets go first,
    the smaller first among equals, unless --as-given keeps the order they were named in (the owner, 2026-10-08, after
    G: died: the slow downloads first, so a slow host's biggest single file is not left to run alone at the end)."""
    out = {}
    for s in sets:
        out.setdefault(s["archive"], []).append(s)
    for v in out.values():
        if not AS_GIVEN:
            v.sort(key=lambda s: (-s["value"], s["gb"] if s["gb"] is not None else 0.5))
    return out


def chosen(names):
    if not names or names == ["all"]:
        return list(SETS)
    bad = [n for n in names if n not in BY_NAME]
    if bad:
        raise SystemExit("unknown set(s): %s" % bad)
    return [BY_NAME[n] for n in names]


def run_lanes(sets, work):
    groups = lanes(sets)
    results = []
    lock = threading.Lock()

    def lane(queue):
        while True:
            with lock:
                if not queue:
                    return
                s = queue.pop(0)
            try:
                r = work(s)
            except Exception as e:  # noqa: BLE001  one set's bug must not stop the other sets in its lane
                log("%s: %s: %s\n%s" % (s["name"], type(e).__name__, e, traceback.format_exc()))
                r = {"name": s["name"], "archive": s["archive"], "plan_gb": s["gb"], "status": "error",
                     "error": "%s: %s" % (type(e).__name__, e)}
            with lock:
                results.append(r)

    runners = [q for a, q in groups.items() for _ in range(LANE_WIDTH.get(a, 1))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runners)) as pool:
        for fu in [pool.submit(lane, q) for q in runners]:
            fu.result()
    order = {s["name"]: i for i, s in enumerate(SETS)}
    return sorted(results, key=lambda r: order[r["name"]])


def plan(sets):
    def one(s):
        ctx = fetch.SetContext(s["name"])
        row = {"name": s["name"], "archive": s["archive"], "area": s["area"], "plan_gb": s["gb"]}
        try:
            lst = listing(s, ctx)
            row.update(licence=lst.licence, licence_ok=checked_licence(s, lst), files=len(lst.files),
                       gb=round(lst.size() / 1e9, 2), unknown_sizes=sum(f.size is None for f in lst.files),
                       notes=lst.notes, skipped=lst.skipped, status="ok")
        except fetch.LicenceChanged as e:
            row.update(status="LICENCE CHANGED", error=str(e))
        except fetch.FetchError as e:
            row.update(status="error", error=str(e))
        log("plan %-28s %s" % (s["name"], row.get("status")))
        return row

    rows = run_lanes(sets, one)
    os.makedirs(ARCHIVE, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    path = os.path.join(ARCHIVE, "plan-%s.json" % stamp)
    kept = []
    if os.path.exists(path):                       # a partial plan updates its rows in the day's file
        with open(path, encoding="utf-8") as f:
            kept = [r for r in json.load(f) if r["name"] not in {x["name"] for x in rows}]
    fetch._save_json(path, sorted(kept + rows, key=lambda r: [s["name"] for s in SETS].index(r["name"])))
    print("\n%-28s %-9s %-8s %6s %8s %8s  %s" % ("set", "archive", "status", "files", "GB now", "GB plan", "licence"))
    tot, tot_plan = 0.0, 0.0
    for r in rows:
        tot += r.get("gb") or 0
        tot_plan += r.get("plan_gb") or 0
        print("%-28s %-9s %-8s %6s %8s %8s  %s" % (
            r["name"], r["archive"], r["status"][:8], r.get("files", ""), r.get("gb", ""),
            "?" if r["plan_gb"] is None else r["plan_gb"], (r.get("licence") or r.get("error", ""))[:90]))
        for what, why in r.get("skipped", []):
            print("    skipped %s: %s" % (str(what)[:70], why[:120]))
    print("\n%d sets, %.1f GB now (plan %.1f GB); written to %s" % (len(rows), tot, tot_plan, path))
    return 0 if all(r["status"] == "ok" for r in rows) else 1


class Record:
    """A set's fetch.json, saved atomically and not more than every SAVE_S seconds while files land."""

    def __init__(self, name):
        self.path = os.path.join(ARCHIVE, name, "fetch.json")
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)
        self.lock = threading.Lock()
        self.saved = 0.0

    def save(self, force=False):
        with self.lock:
            if force or time.time() - self.saved > SAVE_S:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                fetch._save_json(self.path, self.data)
                self.saved = time.time()


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


_ROOM = threading.Condition()
_CLAIMS = []                 # {"drive", "need", "landed"}: sets still downloading or unpacking
_PLACED = {}                 # set name: the claim place() made for it, until the set claims for itself


def _target(path):
    """Where a junction (or a link to a network share) points, by its drive letter; any other path as it is."""
    return os.readlink(path).replace("\\\\?\\", "") if fetch.is_link(path) else path


def _drive(path):
    return os.path.splitdrive(os.path.abspath(path))[0].upper() + os.sep


def _claimed(drive):
    """Bytes the sets at work on `drive` will still write: what each claimed, less what has landed."""
    return sum(max(0, c["need"] - c["landed"]) for c in _CLAIMS if c["drive"] == drive)


def passes(s):
    """Whether the set's download only passes through the disk it lands on (fetch.floor): unpacked, then removed. A
    packed set's download stays as served."""
    return not s.get("packed")


def claim(path, need, reserve, label, wait=True, passing=False):
    """Claim `need` bytes on path's disk for one set's download or unpack, beside what the sets at work
    will still write. With no room, wait while another set may still free some. The claim (its "landed"
    grows as files land), or None when no room comes. `passing`: a download on its way through (passes())."""
    drive = _drive(path)
    t0 = time.time()
    said = False
    with _ROOM:
        while True:
            # the reserve, and never past what the drive keeps free (fetch.floor), whichever leaves more
            avail = min(fetch.free_bytes(drive) - reserve, fetch.room(drive, passing))
            if avail - _claimed(drive) - need >= 0:
                c = {"drive": drive, "need": need, "landed": 0}
                _CLAIMS.append(c)
                return c
            if not wait or not _CLAIMS or time.time() - t0 > ROOM_WAIT_S:
                return None
            if not said:                           # once, however long the room check itself took
                said = True
                log("%s: waits for room on %s (%.0f GB needed, %.0f GB it can take, %.0f GB still to land for other "
                    "sets)" % (label, drive, need / 1e9, avail / 1e9, _claimed(drive) / 1e9))
            _ROOM.wait(60)


def unclaim(c):
    with _ROOM:
        _CLAIMS.remove(c)
        _ROOM.notify_all()


def fetch_set(s, again=False):
    name = s["name"]
    folder = place(s)
    rec = Record(name)
    d = rec.data
    if d.get("status") == "done" and not again:
        log("%s: already done" % name)
        return {"name": name, "status": "done"}
    ctx = fetch.SetContext(name)
    try:
        lst = listing(s, ctx)
        lic = checked_licence(s, lst)
        if not lst.files and lst.skipped and all(re.search(r"restricted|embargoed|closed", why) for _, why in lst.skipped):
            # every file behind an access request or embargo: no request is made (LICENSING's ruling of 2026-10-05)
            raise fetch.SetBlocked("every file is behind an access request or embargo (%s); no request is made, so it "
                                   "is not fetched" % "; ".join("%s: %s" % w for w in lst.skipped))
        if not lst.files:
            raise fetch.FetchError("the archive lists nothing to fetch (skipped: %s)" % lst.skipped)
    except fetch.LicenceChanged as e:
        d.update(status="licence-changed", error=str(e), checked=now())
        rec.save(True)
        log(str(e))
        return {"name": name, "status": d["status"]}
    except fetch.FetchError as e:
        d.update(status="blocked" if isinstance(e, fetch.SetBlocked) else "failed", error=str(e), checked=now())
        rec.save(True)
        log("%s: %s" % (name, e))
        return {"name": name, "status": d["status"]}
    zipped = {r["meta"]["uid"] for r in (d.get("files") or {}).values()
              if r.get("meta", {}).get("uid") and not r["meta"].get("instance")}
    if zipped:                                     # a series already here as NBIA's zip is not fetched again from IDC
        lst.files = [e for e in lst.files if not (e.kind == "instance" and e.meta["uid"] in zipped)]
    d.update(name=name, archive=s["archive"], link=s["link"], ids=s.get("ids") or s.get("slug") or s.get("repo")
             or s.get("node") or s.get("page"), licence_allowed=sorted(s["licence"]), licence_seen=lst.licence,
             licence_ids=lic, licence_read_at=lst.source, licence_read=now(), notes=lst.notes,
             skipped=[list(x) for x in lst.skipped], status="fetching", error=None,
             unpacked=None, removed=None, archive_copy=None)   # what lands now is unpacked when the set is done
    d.setdefault("started", now())
    files = d.setdefault("files", {})
    pre = {f.name: fetch.on_disk(os.path.join(folder, *f.name.split("/"))) for f in lst.files if f.name not in files}
    need = sum(max(0, (f.size or f.meta.get("bytes") or 0) - pre[f.name]) for f in lst.files if f.name not in files)
    disk = os.path.realpath(folder) if os.path.isdir(folder) else ARCHIVE
    unplace(name)                                         # place()'s whole-set claim, now replaced by the real one
    room = claim(disk, need, ARCHIVE_RESERVE, name, passing=passes(s))   # what is still to come: a resumed part
    if room is None:                                                      # already takes its room
        d.update(status="no-room", error="%.1f GB to fetch, %.1f GB the download disk can take (it keeps %.0f%% "
                 "free)" % (need / 1e9, min(fetch.free_bytes(disk) - ARCHIVE_RESERVE, fetch.room(disk, passes(s))) / 1e9,
                            100 * fetch.floor(disk, passes(s))))
        rec.save(True)
        log("%s: %s" % (name, d["error"]))
        return {"name": name, "status": d["status"]}
    room["pre"] = pre
    try:
        return _fetch_files(s, rec, ctx, lst, need, folder, room)
    finally:
        unclaim(room)


def _fetch_files(s, rec, ctx, lst, need, folder, room):
    name, d = s["name"], rec.data
    files = d["files"]
    rec.save(True)
    log("%s: licence %s, %d files, %.2f GB to fetch" % (name, lst.licence[:80], len(lst.files), need / 1e9))
    errors = {}

    def one(e):
        dst = os.path.join(folder, *e.name.split("/"))
        label = "%s/%s" % (name, e.name)
        r = files.get(e.name)
        if e.kind == "aspera":                     # a package folder, as Aspera brings it
            if r and os.path.isdir(dst):
                return
            r = fetch.download(e, dst, ctx, label)
            with rec.lock:
                files[e.name] = r
                room["landed"] += r["size"] - room["pre"].get(e.name, 0)
            rec.save()
            return
        same = r and e.size in (None, r["size"]) and (not e.checksum or r.get("checksum") in (None, e.checksum))
        if same and os.path.exists(dst) and os.path.getsize(dst) == r["size"] and int(os.path.getmtime(dst)) == r["mtime"]:
            return
        if r and not same:
            log("%s: the archive now lists another size or checksum; fetching again" % label)
            r = None
            if os.path.exists(dst):
                os.remove(dst)
        if os.path.exists(dst):
            r = fetch.recheck(e, dst)
            if r is None:
                log("%s: on disk but fails its check; fetching again" % label)
                os.remove(dst)
        if r is None or not os.path.exists(dst):
            try:
                r = fetch.download(e, dst, ctx, label)
            except fetch.FetchError as x:
                if not e.meta.get("archive_url"):
                    raise
                log("%s: the faster copy failed (%s); from the archive itself" % (label, x))
                e.url = e.meta.pop("archive_url")
                r = fetch.download(e, dst, ctx, label)
            if e.meta.get("archive_url"):
                r["from"] = e.url.split("?")[0]    # a faster copy brought it; the archive's checksum passed
        if e.kind == "series":
            r["meta"] = {k: e.meta[k] for k in ("patient", "uid", "project", "modality")}
        elif e.kind == "instance":
            r["meta"] = {"patient": e.meta["patient"], "uid": e.meta["uid"], "instance": True}
        with rec.lock:
            files[e.name] = r
            room["landed"] += r["size"] - room["pre"].get(e.name, 0)
        rec.save()

    # Files at once, the biggest first so the longest is never started last; each host's cap still bounds its
    # connections (Zenodo's 100, four kept for listings: core/fetch.py SHORT).
    jobs = 64 if any(e.kind == "instance" for e in lst.files) else 16 if s["archive"] == Z else 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(one, e): e for e in sorted(lst.files, key=lambda e: -(e.size or 0))}
        for fu in concurrent.futures.as_completed(futs):
            e = futs[fu]
            try:
                fu.result()
            except fetch.SetBlocked as x:
                errors[e.name] = str(x)
                ctx.stop.set()
            except (fetch.FetchError, OSError) as x:
                errors[e.name] = str(x)
                log("%s/%s: %s" % (name, e.name, x))
    blocked = any("refused" in v for v in errors.values()) and ctx.stop.is_set()
    d.update(status="blocked" if blocked else ("partial" if errors else "done"), errors=errors or None,
             finished=now(), bytes=sum(r["size"] for r in files.values()))
    rec.save(True)
    log("%s: %s, %d files, %.2f GB%s" % (name, d["status"], len(files), d["bytes"] / 1e9,
                                         ", %d failed" % len(errors) if errors else ""))
    return {"name": name, "status": d["status"]}


SPLIT_PART = re.compile(r"\.z\d\d$|\.r\d\d$|\.(zip|7z|rar)\.(?!0*1$)\d+$|\.part(?!0*1\.rar$)\d+\.rar$", re.I)


def first_part(fname, names):
    """The first volume a later split part is unpacked through, or None when the set has none."""
    want = []
    m = re.match(r"(.*)\.([zr])\d\d$", fname, re.I)
    if m:
        want = [m.group(1) + (".zip" if m.group(2).lower() == "z" else ".rar")]
    m = re.match(r"(.*\.(?:zip|7z|rar))\.\d+$", fname, re.I)
    if m:
        want = [m.group(1) + "." + "0" * k + "1" for k in range(4)]
    m = re.match(r"(.*)\.part\d+\.rar$", fname, re.I)
    if m:
        want = [m.group(1) + ".part" + "0" * k + "1.rar" for k in range(4)]
    return next((w for w in want if w in names), None)


def _place(src, dst, size):
    """Move one landed file, or an Aspera package's folder, into the working copy (a rename when both are on
    one disk). One already there whole, from a run cut short after its move, counts as placed."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(src):
        if os.path.isdir(dst) or (os.path.exists(dst) and os.path.getsize(dst) == size):
            return 1
        raise fetch.FetchError("%s is gone and %s is not whole" % (src, dst))
    if os.path.isdir(src) and os.path.exists(dst):
        shutil.rmtree(dst)                         # a part copy from a run cut short
    if _drive(os.path.realpath(src)) == _drive(os.path.realpath(os.path.dirname(dst))):   # a junction's own disk
        os.replace(src, dst)
    else:
        shutil.move(src, dst)                      # a copy, then the source removed
    return 1


def _working_target(name, need, beside=None, src=None, via=None):
    """Where the set's working copy goes, and the room claimed there: its junction's target when it has
    one, else the first working disk with room (waiting on the last one when none has it now), or (via) the
    working root on that disk, under its staging line: the copy only passes through on its way to another disk.
    On the drive holding `src` (the download), `beside` is claimed instead: its big tars give their room back as
    they unpack."""
    link = os.path.join(LINKS, name)
    if os.path.lexists(link):
        if not fetch.is_link(link):
            raise fetch.FetchError("%s is a real folder, not a junction; left alone" % link)
        targets = [os.readlink(link).replace("\\\\?\\", "")]
    elif via:
        targets = [os.path.join(r, name) for r in WORKING if _drive(r) == _drive(via)][:1]
        if not targets:
            raise fetch.FetchError("%s: %s holds no working disk (%s)" % (name, via, ";".join(WORKING)))
    else:
        targets = [os.path.join(r, name) for r in WORKING if os.path.isdir(_drive(r))]
    home = _drive(os.path.realpath(src)) if src and beside is not None else None
    for i, t in enumerate(targets):
        n = beside if home and _drive(os.path.realpath(t)) == home else need
        passing = bool(via) and _drive(os.path.realpath(t)) == _drive(via)
        room = claim(t, n, WORKING_RESERVE, name, passing=passing) if i == len(targets) - 1 else \
            claim(t, n, WORKING_RESERVE, name, wait=False, passing=passing)
        if room is not None:
            return t, room
    raise fetch.FetchError("%s: no working disk has %.0f GB to spare" % (name, need / 1e9))


def _copy_unless_whole(s, d):
    """copy2, unless d already has s whole (same size and time, which copy2 sets only after the data): a move cut
    short by a restart goes on where it stopped instead of copying 48 GB to G: again (2026-10-07)."""
    try:
        a, b = os.stat(s), os.stat(d)
        if a.st_size == b.st_size and abs(a.st_mtime - b.st_mtime) < 0.01:
            return d
    except OSError:
        pass
    return shutil.copy2(s, d)


def _tree(root):
    """Every file under root, by its path relative to root, with its size."""
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = os.path.getsize(p)
    return out


def move_tier(name, cold, to=None, staging=False):
    """A set's working copy moved to the hard disk or the share (cold: no model needs it soon) or back to an SSD,
    or to the working disk `to` names, its junction re-pointed so every reader finds it where it always looks.
    Copied, every file's size checked against the original, then the old copy removed; the record (fetch.json, or
    train.json for a training copy) says where it went. Run it only while nothing reads or writes the set.
    `staging`: the copy only passes through `to` (H: up to its 95% line), to go on when the disk after it can take it
    (2026-10-10: G:, its shingled cache full after the NLST copies, wrote 2 MB/s while D: stood at 86%)."""
    link = os.path.join(LINKS, name)
    if fetch.is_link(link):
        src = os.readlink(link).replace("\\\\?\\", "")
    elif os.path.isdir(link):
        src = link                                 # a set older than the junctions, a real folder where readers look:
        #                                            its copy takes its place behind a junction (2026-10-10, D: at 86%:
        #                                            nih-chestxray14 and six more the modelling chat named)
    else:
        raise fetch.FetchError("%s: %s is neither a junction nor a folder" % (name, link))
    cold_drives = {_drive(r) for r in COLD_ROOTS}
    if to:
        roots = [r for r in WORKING if _drive(r) == _drive(to) and os.path.isdir(_drive(r))]
        if not roots:
            raise fetch.FetchError("%s: %s holds no working disk (%s)" % (name, to, ";".join(WORKING)))
        if _drive(src) == _drive(to):
            log("%s: already on %s" % (name, _drive(src)))
            return {"name": name, "status": "done"}
    elif (_drive(src) in cold_drives) == cold:
        log("%s: already on %s" % (name, _drive(src)))
        return {"name": name, "status": "done"}
    else:
        roots = [r for r in WORKING if (_drive(r) in cold_drives) == cold and os.path.isdir(_drive(r))]
    files = _tree(src)
    need = sum(files.values())
    room = None
    for r in roots:
        room = claim(r, need, WORKING_RESERVE, name, wait=False, passing=staging)
        if room is not None:
            break
    if room is None:
        raise fetch.FetchError("%s: no %s disk has %.0f GB to spare" % (name, "cold" if cold else "SSD", need / 1e9))
    dst = os.path.join(r, os.path.basename(src.rstrip("/\\")))
    u = shutil.disk_usage(src)
    full = u.free < u.total * (fetch.floor(src) + FULL_RANK_AT)   # an SSD near 80%, or H: staged past 85%
    try:
        log("%s: %.1f GB in %d files, %s to %s%s" % (name, need / 1e9, len(files), src, dst,
                                                    " (first on the slow disk: %s is near its line)" % _drive(src).rstrip("\\/")
                                                    if full else ""))
        with fetch.Turn([src, dst], name, 1 if full else rank(name)):
            shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_copy_unless_whole)
            got = _tree(dst)
        bad = [k for k, v in files.items() if got.get(k) != v]
        if bad:
            raise fetch.FetchError("%s: %d files differ after the copy (first %s); the old copy is kept"
                                   % (name, len(bad), bad[0]))
        was = src
        if src == link:                            # the folder set aside (same disk, at once), the junction in its place
            os.rename(link, link + ".moved")
            try:
                fetch.junction(link, dst)
            except BaseException:
                os.rename(link + ".moved", link)
                raise
            src = link + ".moved"
        else:
            fetch.junction(link + ".moving", dst)  # the new link first: should it fail, the old one still stands
            os.rmdir(link)                         # the junction itself, never what it points at
            os.rename(link + ".moving", link)
        for p in (os.path.join(ARCHIVE, name, "fetch.json"), os.path.join(ARCHIVE, name, "train.json")):
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                d["working_copy"] = dst.replace("\\", "/")
                d.setdefault("moved", []).append({"from": was.replace("\\", "/"), "to": d["working_copy"], "at": now()})
                fetch._save_json(p, d)
        shutil.rmtree(src)
        log("%s: now at %s" % (name, dst))
    finally:
        unclaim(room)
    return {"name": name, "status": "done"}


def unpack_set(name, via=None):
    """A landed set into its working copy (archives unpacked, loose files and images moved), the junction to
    it, then the rest of its download removed; fetch.json stays. A run cut short goes on where it stopped.
    `via` (H:): unpacked there to pass through, taking a turn only on the disk written. Its download, read
    from G:, takes no turn there: reading a shingled disk does not make it rewrite its bands as writing does
    (2026-10-10: G:, written by both NLST training copies, unpacked mammo-mx at 6 MB/s while H: sat idle)."""
    rec = Record(name)
    d = rec.data
    if d.get("status") != "done":
        raise fetch.FetchError("%s: not fetched (status %s)" % (name, d.get("status")))
    link = os.path.join(LINKS, name)
    src_root = os.path.join(ARCHIVE, name)
    done = d.setdefault("unpacked_files", [])
    seen = set(done)

    def src_of(f):
        return os.path.join(src_root, *f.split("/"))

    for fname in d["files"]:
        if SPLIT_PART.search(fname) and not first_part(fname, d["files"]):
            raise fetch.FetchError("%s: split part %s has no first volume in the set" % (name, fname))
    left = [src_of(f) for f in d["files"]
            if f not in seen and not SPLIT_PART.search(f) and os.path.exists(src_of(f))]
    need = sum(fetch.unpacked_size(x) for x in left)
    beside = sum(fetch.unpack_need(x, release=True) for x in left)   # its big tars give their bytes back as they go
    target, room = _working_target(name, need, beside, src_root, via)
    count = 0
    t0 = time.time()
    turn = None
    try:
        os.makedirs(target, exist_ok=True)         # in the try: a folder not made gives its room back
        turn = fetch.Turn([target] if via else [target, src_root], name, rank(name))
        # Archives first, loose files after, so a loose file wins over a same-named member (an annexed image
        # over its pointer in the repository snapshot). Smallest archive first: labels and small sets are usable
        # within minutes, not after a day of detection slices onto the hard disk (CSTL, 2026-10-06: its 242 MB of
        # malignancy crops sat behind 31 GB of BMP slices unpacking at one file a second).
        order = sorted(d["files"].items(), key=lambda kv: (not (fetch.is_archive(kv[0]) or kv[1].get("meta")),
                                                           kv[1].get("size") or 0, kv[0]))
        turn.take()
        for fname, r in order:
            if fname in seen:
                continue
            turn.share()                           # between archives, a set ranked first goes next
            src = src_of(fname)
            sub = os.path.join(target, *fname.split("/")[:-1])
            meta = r.get("meta", {})
            if meta.get("instance"):               # one image of a series from IDC, beside its series' others
                count += _place(src, os.path.join(fetch.series_folder(target, meta), os.path.basename(fname)), r["size"])
            elif meta.get("uid"):
                count += fetch.unpack_series(src, target, meta)
            elif SPLIT_PART.search(fname):
                pass                               # unpacked through its first part
            elif fetch.is_archive(fname):
                count += fetch.unpack(src, sub, turn, release=True)  # a tar shares the turn between its members
                                                   # too; a big one gives its bytes back as it goes (removed below)
            else:
                count += _place(src, os.path.join(sub, os.path.basename(fname)), r["size"])
            done.append(fname)
            seen.add(fname)
            rec.save()
    finally:
        if turn:
            turn.give()
        unclaim(room)
    made = fetch.junction(link, target)
    for r in d["files"].values():                  # a check made as it was unpacked (globus_fetch's tars) is made now
        if r.get("checked_when_unpacked"):
            r["checked"] = r.pop("checked_when_unpacked")
    d.update(working_copy=target.replace("\\", "/"), junction=link.replace("\\", "/"), unpacked=now())
    rec.save(True)
    freed = 0
    for fname in d["files"]:                       # where a tar given back as it unpacked would have gone on
        with contextlib.suppress(FileNotFoundError):
            os.remove(src_of(fname) + ".released")
    for fname in d["files"]:                       # the download is in the working copy now
        p = src_of(fname)
        if os.path.isdir(p):
            freed += fetch.unpacked_size(p)
            shutil.rmtree(p)
        elif os.path.exists(p):
            freed += os.path.getsize(p)
            os.remove(p)
    for folder, _, _ in os.walk(src_root, topdown=False):
        if folder != src_root:
            with contextlib.suppress(OSError):    # one still holding something stays
                os.rmdir(folder)
    d["removed"] = now()
    rec.save(True)
    log("%s: unpacked %d files into %s in %.0f s%s; %.2f GB of download removed" % (
        name, count, target, time.time() - t0, "; junction made" if made else "", freed / 1e9))


def shrink_tree(root, label):
    """core.store over one working copy, with its totals logged; the totals are returned for the record."""
    log("%s: re-encoding uncompressed DICOM images losslessly (%s) in %s" % (label, store.CODEC_NAME, root))
    t = store.shrink_tree(root, log=log)
    out = {"at": now(), "codec": store.CODEC_NAME, "files": t["files"], "shrunk": t["shrunk"], "left_alone": t["left"],
           "kept_original": t["kept"], "unchecked": t["unchecked"], "gb_before": round(t["before"] / 1e9, 3),
           "gb_after": round(t["after"] / 1e9, 3), "why_left": t["reasons"], "examples": t["examples"],
           "seconds": t["seconds"]}
    if t.get("worker_errors"):
        out["worker_errors"] = t["worker_errors"]
    log("%s: %d of %d files re-encoded, %.2f GB to %.2f GB (%.0f%% saved) in %d s; left alone %s%s" % (
        label, t["shrunk"], t["files"], t["before"] / 1e9, t["after"] / 1e9,
        100 * (1 - t["after"] / t["before"]) if t["before"] else 0, t["seconds"], t["reasons"] or "none",
        "; NOT IDENTICAL, originals kept: %d" % t["kept"] if t["kept"] else ""))
    return out


def shrink_set(name):
    """A set's working copy re-encoded losslessly (core.store), and what was done written into its fetch.json:
    the re-encoded files' bytes no longer match the archive's checksums, though every pixel and element does."""
    rec = Record(name)
    d = rec.data
    if not d.get("unpacked") or not d.get("working_copy"):
        raise fetch.FetchError("%s: not unpacked yet" % name)
    prev = d.get("shrunk")
    with fetch.Turn([d["working_copy"]], name, rank(name)):
        new = shrink_tree(d["working_copy"], name)
    if prev:                                       # a later run: its "already compressed" holds the earlier runs' work
        mine = prev["shrunk"]
        why = dict(new["why_left"])
        why["already compressed"] = why.get("already compressed", 0) - mine
        if why["already compressed"] <= 0:
            del why["already compressed"]
        new.update(shrunk=new["shrunk"] + mine, left_alone=new["left_alone"] - mine, why_left=why,
                   gb_before=prev["gb_before"], seconds=new["seconds"] + prev["seconds"],
                   earlier=prev.get("earlier", []) + [{k: v for k, v in prev.items() if k != "earlier"}])
    d["shrunk"] = new
    rec.save(True)


def status():
    print("%-28s %-15s %6s %9s  %s" % ("set", "status", "files", "GB", "working copy / error"))
    tot = 0
    for s in SETS:
        p = os.path.join(ARCHIVE, s["name"], "fetch.json")
        if not os.path.exists(p):
            print("%-28s %-15s" % (s["name"], "-"))
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        gb = sum(r["size"] for r in d.get("files", {}).values()) / 1e9
        tot += gb
        print("%-28s %-15s %6d %9.2f  %s" % (s["name"], d.get("status"), len(d.get("files", {})), gb,
                                             d.get("working_copy") or (d.get("error") or "")[:100]))
    print("\n%.1f GB in %s" % (tot, ARCHIVE))


def approval(s):
    """Who approved a set, and when, for its LICENSING.md Status cell."""
    return s.get("approved") or "Approved by the owner, 2026-10-04"


def record_licensing():
    """Each landed set's LICENSING.md Status cell becomes where it is. Rows are found by their link."""
    with open(LICENSING, encoding="utf-8") as f:
        text = f.read()
    changed, missing = 0, []
    cells = {}
    for s in SETS:
        p = os.path.join(ARCHIVE, s["name"], "fetch.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("status") not in ("done", "licence-changed", "blocked"):
            continue
        day = (d.get("finished") or d.get("checked") or now())[:10]
        if d["status"] == "done":
            checks = {c for r in d["files"].values() for c in (r.get("checked") or "").split("; ") if c}
            if "sha256 recorded on arrival" in checks:   # a resumed file's "sha256 recorded" says the same thing
                checks.discard("sha256 recorded")
            checks = sorted(checks)
            record = "`%s/%s/fetch.json`" % (ARCHIVE, s["name"])
            if s.get("packed"):
                where = ("kept packed as served at `%s` (a cold archive, unpacked when a model needs it), its record "
                         "%s" % (_target(os.path.join(ARCHIVE, s["name"])).replace("\\", "/"), record))
            elif not d.get("working_copy"):
                where = "downloaded to `%s/%s`, being unpacked" % (ARCHIVE, s["name"])
            elif d.get("archive_copy"):
                where = ("`Datasets/public/%s` (a junction to `%s`); the download as served stays at `%s`, its "
                         "record %s" % (s["name"], d["working_copy"], d["archive_copy"], record))
            else:
                where = ("`Datasets/public/%s` (a junction to `%s`); the download was removed once unpacked, and %s "
                         "keeps each file's address and checksum" % (s["name"], d["working_copy"], record))
            sh = d.get("shrunk") or {}
            if sh.get("shrunk"):
                where += ("; %d DICOM images re-encoded losslessly (%s, %.2f GB to %.2f GB, each checked bit for bit, so "
                          "their bytes no longer match the archive's checksums)" % (
                              sh["shrunk"], sh["codec"], sh["gb_before"], sh["gb_after"]))
            skipped = "; %s" % "; ".join("%s left out: %s" % (w, y) for w, y in d.get("skipped") or []) if d.get("skipped") else ""
            # a check that is a clause of its own ("to be checked on this PC as it is unpacked: ...", "re-checked on
            # resume") follows the ones that name what checked the file, never "checked by to be checked"
            clause = [c for c in checks if re.match(r"(to be |re-)?checked\b", c)]
            named = [c for c in checks if c not in clause]
            said = "; ".join((["checked by " + "; ".join(named)] if named else []) + clause)
            cell = ("%s. **Downloaded %s:** %s; %d files, %.2f GB, licence re-read at "
                    "download (%s), %s%s." % (approval(s), day, where, len(d["files"]), d["bytes"] / 1e9,
                                              ", ".join(d.get("licence_ids") or []), said, skipped))
        else:
            cell = "%s. **Not downloaded (%s, %s):** %s" % (
                approval(s), d["status"], day, (d.get("error") or "").replace("|", "/")[:400])
        cells.setdefault(s.get("row", s["link"]), {})[s["name"]] = cell
    for link, parts in cells.items():
        names = [s["name"] for s in SETS if s.get("row", s["link"]) == link]
        shared = len(names) > 1
        for n in names:                            # a shared row keeps every set in it, landed or not
            parts.setdefault(n, "%s; **not downloaded yet**." % approval(BY_NAME[n]))
        cell = " ".join("%s: %s" % (n, parts[n]) if shared else parts[n] for n in names).replace("|", "/")
        rows = [m for m in re.finditer(r"^\| \[[^\]]*\]\(%s\)[^\n]*$" % re.escape(link), text, re.M)]
        if len(rows) != 1:
            missing.append(link)
            continue
        line = rows[0].group(0)
        old_cells = line.split(" | ")
        new = " | ".join(old_cells[:-1] + [cell]) + " |"
        if new != line:
            text = text.replace(line, new)
            changed += 1
    if changed:
        with open(LICENSING, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    if changed or missing:
        log("LICENSING.md: %d Status cells written%s" % (changed, "; no single row for %s" % missing if missing else ""))
    return changed


RECORDING = threading.Lock()


def fetched(s, again):
    """fetch_set, then the set's LICENSING.md Status cell, written as soon as the set lands."""
    r = fetch_set(s, again)
    if r["status"] in ("done", "blocked", "licence-changed"):
        with RECORDING:
            record_licensing()
    return r


def settled(name):
    """Unpacked and re-encoded: nothing left for the background unpacker to do."""
    d = Record(name).data
    return bool(d.get("unpacked") and d.get("shrunk"))


def unpacked(s, via=None, to=None):
    """unpack_set and shrink_set under the set's lock (each only if not done yet), then (to) the working copy
    moved to that disk, then its Status cell again (it names the working copy now). `via`: see unpack_set."""
    def work(s):
        if s.get("packed"):
            log("%s: stays packed (a cold archive, unpacked when a model needs it)" % s["name"])
            with RECORDING:
                record_licensing()
            return {"name": s["name"], "status": "done"}
        if not Record(s["name"]).data.get("unpacked"):
            unpack_set(s["name"], via)
        if not Record(s["name"]).data.get("shrunk"):
            shrink_set(s["name"])                  # where it was unpacked: what moves on is the smaller copy
        if to:
            move_tier(s["name"], True, to)
        with RECORDING:
            record_licensing()
        return {"name": s["name"], "status": "done"}
    try:
        return locked(s, work)
    except Exception as e:  # noqa: BLE001  one set's unpack must not stop the others
        log("%s: unpack failed: %s: %s\n%s" % (s["name"], type(e).__name__, e, traceback.format_exc()))
        return {"name": s["name"], "status": "unpack failed: %s" % e}


def fetch_all(sets, again):
    """Every set down its lane; each one that lands (or landed earlier and was never unpacked) is unpacked
    in the background while its lane goes on to the next."""
    unpacker = concurrent.futures.ThreadPoolExecutor(max_workers=UNPACKERS)
    pending = []
    plock = threading.Lock()

    def work(s):
        r = locked(s, lambda s: fetched(s, again))
        if r["status"] == "done" and not settled(s["name"]):
            with plock:
                pending.append(unpacker.submit(unpacked, s))
        return r

    res = run_lanes(sets, work)
    unpack = {r["name"]: r["status"] for r in (fu.result() for fu in pending)}
    unpacker.shutdown()
    return [dict(r, status=unpack.get(r["name"], r["status"])) for r in res]


def land_licensing():
    """After a run, land LICENSING.md the way every session on this machine does (cycle.py), when it is here."""
    tool = os.path.expanduser("~/.claude/tools/cycle.py")
    if not os.path.exists(tool):
        return
    r = subprocess.run([shutil.which("python") or sys.executable, tool, "--land",
                        "LICENSING: Status cells of the sets tools/fetch_public.py landed", "LICENSING.md"],
                       cwd=paths.ROOT, capture_output=True, text=True, creationflags=fetch.NO_WINDOW)
    log("landing LICENSING.md: %s" % (r.stdout or r.stderr).strip()[-300:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["plan", "fetch", "status", "unpack", "shrink", "record", "cold", "warm"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--again", action="store_true", help="fetch a set again even if fetch.json says done; shrink: "
                                                         "scan a set already re-encoded once more")
    ap.add_argument("--to", help="cold/warm: the drive to move to (D:), rather than the first of its kind with room; "
                                 "unpack: the drive the working copy moves to once unpacked and re-encoded")
    ap.add_argument("--staging", action="store_true", help="cold/warm with --to: the copy only passes through that "
                                                             "disk (H:, up to its 95%% line), to go on later")
    ap.add_argument("--via", help="unpack: the drive to unpack and re-encode on, under its staging line (H:), "
                                  "before the move to --to")
    ap.add_argument("--log", help="append everything printed to this file, line by line (a shell's redirect "
                                  "can hold it back in blocks)")
    ap.add_argument("--as-given", action="store_true", help="fetch: inside a lane, take the sets in the order named "
                                                              "(say, the biggest single file first) rather than by use")
    a = ap.parse_args()
    if a.log:
        sys.stdout = sys.stderr = open(a.log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    global AS_GIVEN
    AS_GIVEN = a.as_given
    if a.command == "plan":
        sys.exit(plan(chosen(a.names)))
    if a.command == "status":
        return status()
    if a.command == "record":
        print("LICENSING.md: %d Status cells written" % record_licensing())
        return None
    if a.command == "shrink":
        res = []
        for n in a.names:
            if n in BY_NAME:
                done = Record(n).data.get("shrunk")
                if done and not a.again:           # unpack re-encodes as it goes; 2026-10-11 the Zenodo chain's shrink
                    log("%s: already re-encoded %s (shrink --again scans it once more)" % (n, done.get("at", "")[:16]))
                    res.append({"name": n, "status": "done"})   # step re-read all of 38 such sets on G: for nothing
                    continue
                res.append(locked(BY_NAME[n], lambda s: (shrink_set(s["name"]), {"name": s["name"], "status": "done"})[1]))
            else:                                  # a folder of a set fetched some other way: its record sits beside it
                out = shrink_tree(n, os.path.basename(n.rstrip("/\\")))
                fetch._save_json(os.path.join(n, ".shrunk.json"), out)
                res.append({"name": n, "status": "done" if not out["kept_original"] else "some originals kept"})
        with RECORDING:
            record_licensing()
    elif a.command == "unpack":
        res = [unpacked(BY_NAME[n], a.via, a.to) for n in a.names]
    elif a.command in ("cold", "warm"):                # a training copy (<collection>-train) has no lock of its own
        res = [locked(BY_NAME[n], lambda s: move_tier(s["name"], a.command == "cold", a.to, a.staging)) if n in BY_NAME
               else move_tier(n, a.command == "cold", a.to, a.staging) for n in a.names]
        with RECORDING:
            record_licensing()
    else:
        os.makedirs(ARCHIVE, exist_ok=True)
        res = fetch_all(chosen(a.names), a.again)
        land_licensing()
    print("\n" + "\n".join("%-28s %s" % (r["name"], r["status"]) for r in res))
    sys.exit(0 if all(r["status"] == "done" for r in res) else 1)


def place(s):
    """The set's archive folder, first made a junction to the first disk in CACHE_ROOTS (H: first, then the spare
    disk G:) that can take the whole set above what it keeps free (fetch.floor: on H:, 5% under a download that
    passes through), before anything is written there, its lock file included: a downloaded file waits on the
    network, never on a hard disk (owner, 2026-10-06: G: takes "anything that doesn't need speed"; 2026-10-09: H: is
    "fine for staging"). The room is claimed at once, so sets placed together never count on the same room, and held until
    the set claims for itself (unplace). A folder already made stays where it is. An ordinary set no cold disk can
    take lands on the download disk (ARCHIVE) if it fits there, else stays there while claim() waits for room."""
    folder = os.path.join(ARCHIVE, s["name"])
    with _ROOM:
        if os.path.lexists(folder):
            return folder
        need = (s.get("gb") or 0) * 1e9

        def fits(root):
            d = _drive(root)
            return os.path.isdir(d) and min(fetch.free_bytes(d) - ARCHIVE_RESERVE,
                                            fetch.room(d, passes(s))) - _claimed(d) >= need
        roots = [r for r in CACHE_ROOTS if fits(r)]
        if roots:
            root = roots[0]
            target = os.path.join(root, s["name"])
            os.makedirs(target, exist_ok=True)
            fetch.junction(folder, target)
        elif s.get("cold"):
            raise fetch.FetchError("%s: no cold disk can take %.0f GB and keep what it keeps free" % (s["name"], need / 1e9))
        elif fits(ARCHIVE):
            root = ARCHIVE
        else:
            return folder
        if need:
            c = {"drive": _drive(root), "need": need, "landed": 0}
            _CLAIMS.append(c)
            _PLACED[s["name"]] = c
    return folder


def unplace(name):
    """The room place() claimed for a set, given back: the set now claims what it still needs, or is done."""
    with _ROOM:
        c = _PLACED.pop(name, None)
        if c:
            _CLAIMS.remove(c)
            _ROOM.notify_all()


def locked(s, work):
    """work(s) while holding the set's lock: one process per set at a time (a second run skips a set the
    first is working on), so a phase-B run or a single set can go beside a long `fetch all`. The lock is
    the OS's own, on a byte past the pid the file holds, so it ends with its process however that ends."""
    folder = place(s)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".lock"), "a+b") as f:
        try:
            if sys.platform == "win32":
                import msvcrt  # noqa: PLC0415
                f.seek(1 << 20)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.seek(0)
            log("%s: pid %s is working on it; skipped" % (s["name"], f.read().decode().strip() or "?"))
            return {"name": s["name"], "status": "busy"}
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()).encode())
        f.flush()
        try:
            return work(s)
        finally:
            unplace(s["name"])


if __name__ == "__main__":
    main()
