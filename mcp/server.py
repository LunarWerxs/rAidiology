"""rAidiology as an MCP server: the visitor's own AI opens their scan, on their own machine.

Why this exists. The alternative was a server with a graphics card and an API key, which costs money
per reading and means their scan has to be uploaded to somebody. This way the scan never leaves their
computer, there is nothing to pay for per reading, and the AI they already have does the talking.

**The AI explains, these tools grade.** That division is not a style choice, it is what the numbers
say. On the held-out X-ray test the trained detector scored 0.976 (AUROC) and Claude Opus 5.5 looking
at the same pictures scored 0.690; on spine MRI a Claude reader did not beat even the measurement
rules. So every graded answer here comes from a model that was scored against a radiologist, and the
AI's job is to find the files, choose the series, and put the result into words. `render_slice` hands
the AI a picture on purpose, and says in its own answer that what the AI sees in it is description,
never the verdict.

It speaks MCP over stdin and stdout with no library behind it: newline-delimited JSON-RPC 2.0 with
`initialize`, `tools/list` and `tools/call`. A medical tool is a bad place to add a dependency for
200 lines of protocol, and this way the install is this repository and nothing else.

Add it to Claude Desktop or any MCP client as:
  {"command": "<python>", "args": ["<repo>/mcp/server.py"]}
Try it from a shell with:  python mcp/server.py --selftest
"""
import argparse
import atexit
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "xray", "eval"))

PROTOCOL = "2024-11-05"
NAME = "raidiology"
VERSION = "0.1.0"
NOT_A_DEVICE = ("Not a medical device. rAidiology is for education and research, not for "
                "diagnosis or treatment decisions, and it is not cleared by any regulator. Tell the "
                "person to take anything they are worried about to a doctor.")
MAX_FILES = 20000
MAX_IMAGES = 12        # images scored per series in one call, so a long series cannot stall the client
CT_VERSION = 1
CT_SEGMENTER = os.path.join(ROOT, "modalities", "ct", "eval", "segmenter.py")
CT_FACTS = os.path.join(ROOT, "modalities", "ct", "models", "segmenter-v%d.json" % CT_VERSION)
CT_SECONDS = 3600      # a whole-body CT on a slow processor; the reading itself says how long it took
# Where the viewer is, if whoever installed this server wants to name an address. Empty by default
# and described in words instead: a web address baked in here is one person's own deployment, and
# this file is handed to strangers.
VIEWER = os.environ.get("RAIDIOLOGY_VIEWER", "").strip()


def _lazy():
    """The X-ray reader, imported on first use so an `about` call pays for nothing. It runs the five
    folds as one graph on ONNX Runtime, not torch (modalities/xray/eval/onnx_reader.py): measured
    2026-10-02, a first answer in a fresh process took 1.0 s instead of 2.3, with every one of 400
    development decisions the same as the scored torch reading."""
    import onnx_reader  # noqa: PLC0415
    return onnx_reader


# What a file that can never be a DICOM image starts with: JPEG, PNG, GIF, BMP, TIFF, PDF, ZIP, and a
# Windows program. A DICOM file starts with its 128-byte preamble or, written without one, with a tag
# number, so none of these can be one.
NOT_DICOM = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM", b"II*\x00", b"MM\x00*", b"%PDF", b"PK\x03\x04", b"MZ")


def _worth_reading(path):
    """Whether pydicom should be shown the file. On a real hospital disc, 210 JPEGs beside the
    images cost 78 s of forced pydicom reads that could only fail (365 ms each), and every tool
    that takes a series walks the disc again."""
    try:
        with open(path, "rb") as f:
            head = f.read(132)
    except OSError:
        return False
    return head[128:132] == b"DICM" or not head.startswith(NOT_DICOM)


def _dicom_files(folder):
    out = []
    for dirpath, _dirs, files in os.walk(folder):
        for n in files:
            if n.lower().endswith((".exe", ".dll", ".ini", ".htm", ".html", ".pdf", ".txt", ".xml",
                                   ".css", ".js", ".jar", ".cab", ".msi", ".zip")):
                continue
            p = os.path.join(dirpath, n)
            if not _worth_reading(p):
                continue
            out.append(p)
            if len(out) >= MAX_FILES:
                return out
    return out


_UNPACKED = {}


_HEADERS = {}


def _disc_state(folder):
    """Every file's place, size and time. If none of it changed, nothing on the disc did."""
    out = []
    for dirpath, _dirs, files in os.walk(folder):
        for n in files:
            st = os.stat(os.path.join(dirpath, n))
            out.append((dirpath, n, st.st_size, st.st_mtime_ns))
    return sorted(out)


def _headers(folder):
    """What every image on a disc says about itself, read once a session: (files looked at, [facts]).

    Every tool that takes a series used to walk the disc and read every header again, so a reading
    followed by a findings file and three pictures read the disc five times. This keeps the first
    reading and reads again only when a file on the disc changed by name, size or time."""
    import pydicom  # noqa: PLC0415  arkitect-allow: test-floor-loosened - imported when a disc is read, so `about` pays for nothing, like every tool here
    state = _disc_state(folder)
    kept = _HEADERS.get(folder)
    if kept and kept[0] == state:
        return kept[1], kept[2]
    files = _dicom_files(folder)
    facts = []
    for p in files:
        try:
            d = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if getattr(d, "Rows", None):
                facts.append(_facts_of(d, p))
        except Exception:  # noqa: BLE001  arkitect-allow: test-floor-loosened - as before: a file that is not DICOM, or is broken, is not a scan
            continue
    _HEADERS[folder] = (state, len(files), facts)
    return len(files), facts


def _tag(d, name, default=""):
    return getattr(d, name, default) or default


def _facts_of(d, path):
    """What one image says about itself, as the tools use it. No name, number or date is read."""
    return {
        "path": path, "series": _series_key(d, path), "number": int(_tag(d, "InstanceNumber", 0)),
        "size": "%dx%d" % (int(_tag(d, "Columns", 0)), int(_tag(d, "Rows", 0))),
        "modality": str(_tag(d, "Modality")),
        "description": str(_tag(d, "SeriesDescription") or _tag(d, "StudyDescription") or "untitled"),
        "body_part": str(_tag(d, "BodyPartExamined")),
        "iop": [float(v) for v in _tag(d, "ImageOrientationPatient", [])],
        "ipp": [float(v) for v in _tag(d, "ImagePositionPatient", [])],
        "sop": str(_tag(d, "SOPInstanceUID")), "match": _match_uid(d),
    }


def _cleanup():
    """Take every unpacked copy of somebody's scan off the disk when the server stops. Nothing
    deletes a mkdtemp folder on its own, and these hold medical images."""
    for folder in list(_UNPACKED.values()):
        shutil.rmtree(folder, ignore_errors=True)
    _UNPACKED.clear()


atexit.register(_cleanup)


def _open_folder(path):
    """A folder, or a ZIP unpacked once into a temporary folder. Returns (folder, note).

    Unpacked once, on purpose: every tool that takes a series re-opens the archive, and unpacking
    each time would leave another full copy of the person's scan lying around and fill the disk. The
    copy is keyed by the archive's path, size and timestamp, so an edited archive is unpacked again,
    and `_cleanup` removes them all when the server exits."""
    if os.path.isdir(path):
        return path, ""
    if not zipfile.is_zipfile(path):
        raise ValueError("%s is neither a folder nor a ZIP file." % path)
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, int(st.st_mtime))
    if key in _UNPACKED and os.path.isdir(_UNPACKED[key]):
        return _UNPACKED[key], "The ZIP was already unpacked on this machine for this session."
    tmp = tempfile.mkdtemp(prefix="raidiology-")
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            p = os.path.normpath(os.path.join(tmp, n))
            if not p.startswith(os.path.abspath(tmp) + os.sep):
                continue                          # an entry trying to escape the folder
            z.extract(n, tmp)
    _UNPACKED[key] = tmp
    return tmp, ("The ZIP was unpacked to a temporary folder on this machine, nothing was sent "
                 "anywhere, and it is deleted when this server stops.")


def tool_about(_args):
    return {
        "what": "rAidiology explains a medical scan in plain English. It runs on this machine: no "
                "upload, no account, no server.",
        "how_to_use_me": "Call find_scans on the folder or ZIP from the hospital CD, then read_xray "
                         "on a radiograph. Use render_slice if you want to look at a picture and "
                         "describe it, but take the yes-or-no from read_xray, never from the image. "
                         "Finish with write_findings: it saves a small file the person can drop into "
                         "rAidiology's viewer to see their own images with what was found marked on "
                         "them, which is a far better thing to leave them with than a chat message.",
        "why": "On the held-out test the trained detector scored 0.976 and Claude Opus 5.5 reading "
               "the same pictures scored 0.690. Your eyes are for the explanation; the grade comes "
               "from the model that was measured.",
        "scan_types": {"xray": "a fracture detector, scored and shipping (read_xray)",
                       "spine_mri": "measurements, a disc grader and a 3D view, but they need a "
                                    "graphics card and are not in this server yet",
                       "ct": "an organ and bone labeller, scored and shipping (read_ct): 0.77 mean "
                             "Dice on 89 held-out scans. It says where each organ and bone is, not "
                             "whether anything is wrong with it"},
        "not_a_medical_device": NOT_A_DEVICE,
    }


def _series_key(d, path):
    """The id a series is known by, derived the same way everywhere.

    The whole SeriesInstanceUID, not a short tail of it: two series can share a tail and would then
    be read as one, which would score somebody's images together with a stranger's. A file with no
    UID falls back to its folder, so those series can still be chosen instead of being listed under
    a name nothing matches."""
    uid = str(getattr(d, "SeriesInstanceUID", "") or "").strip()
    return uid if uid else "nouid:" + os.path.basename(os.path.dirname(os.path.abspath(path)))


def tool_find_scans(args):
    folder, note = _open_folder(args["path"])
    looked, facts = _headers(folder)
    series = {}
    for f in facts:
        s = series.setdefault(f["series"], {
            "series_id": f["series"], "modality": f["modality"], "description": f["description"],
            "images": 0, "size": f["size"], "body_part": f["body_part"], "files": [],
        })
        s["images"] += 1
        s["files"].append(f["path"])
    out = []
    for s in sorted(series.values(), key=lambda x: (x["modality"], x["description"])):
        out.append({k: v for k, v in s.items() if k != "files"})
    return {
        "folder": folder, "files_looked_at": looked, "series": out,
        "note": (note + " No name, patient number or date of birth is read from these files.").strip(),
        "next": "read_xray works on a series whose modality is CR, DX or RG, and read_ct on one whose "
                "modality is CT. MR is not in this server yet.",
        "not_a_medical_device": NOT_A_DEVICE,
    }


def _series_files(path, series_id):
    return [f["path"] for f in _series_facts(path, series_id)]


def _series_facts(path, series_id):
    """One series' images in image-number order, with what each says about itself."""
    folder, _ = _open_folder(path)
    _, facts = _headers(folder)
    return sorted((f for f in facts if f["series"] == series_id), key=lambda f: (f["number"], f["path"]))


def tool_read_xray(args):
    reader = _lazy()
    import datasets  # noqa: PLC0415
    profile = str(args.get("body_part_profile") or "grazpedwri").strip()
    if profile not in datasets.NAMES:
        # Checked before anything is scored: an unknown profile would otherwise be found only when
        # the threshold was looked up, after the work, or worse would pick the wrong threshold.
        raise ValueError("body_part_profile must be one of %s; got %r. Use grazpedwri for a wrist "
                         "and fracatlas for a hand, leg, hip or shoulder."
                         % (", ".join(datasets.NAMES), profile))
    if "file" in args and args["file"]:
        paths = [args["file"]]
    else:
        paths = _series_files(args["path"], args["series_id"])
        if not paths:
            raise ValueError("No images found for series %s under %s"
                             % (args.get("series_id"), args.get("path")))
    rows, meta = reader.score_images(paths[:MAX_IMAGES], 1, profile)
    worst = max(rows, key=lambda r: r["score"]) if rows else None
    cut = len(paths) > len(rows)
    return {
        "images_in_series": len(paths),
        "images_read": len(rows),
        "whole_series_read": not cut,
        "note": ("Only the first %d of %d images were read, so this answer is about those, not the "
                 "whole series. Ask again with a file to read a particular one."
                 % (len(rows), len(paths))) if cut else "",
        "per_image": [{"file": os.path.basename(r["path"]), "score": r["score"],
                       "fracture": r["fracture"]} for r in rows],
        "answer": ("A fracture is likely on at least one of the images read." if worst and worst["fracture"]
                   else "No fracture was found on the images read."),
        "highest_score": worst["score"] if worst else None,
        "threshold": meta["thresholds"][profile],
        "how_good_is_this": {
            "measured_on": "4,170 held-out paediatric wrist radiographs from 1,218 patients it had "
                           "never seen, plus 798 mixed hand, leg, hip and shoulder radiographs",
            "auroc": 0.976, "auroc_95ci": [0.970, 0.981],
            "catches": "92.7% of fractures (95% CI 91.2 to 94.1)",
            "clears": "94.4% of unbroken images correctly (92.4 to 96.1)",
            "beat_the_best_simple_method_by": "0.138 AUROC (95% CI 0.125 to 0.154)",
        },
        "who_it_is_not_for": "It was trained mostly on children's wrists, with some hands, legs, "
                             "hips and shoulders. A chest, skull or spine X-ray is outside what it "
                             "was measured on, so say so rather than giving a confident answer.",
        "say_this_too": "It misses about 7 fractures in 100 and calls about 6 in 100 unbroken "
                        "images broken, so a 'no' is not an all-clear.",
        "not_a_medical_device": NOT_A_DEVICE,
    }


def tool_render_slice(args):
    from PIL import Image  # noqa: PLC0415
    import prepare  # noqa: PLC0415
    if args.get("file"):
        path = args["file"]
    else:
        files = _series_files(args["path"], args["series_id"])
        if not files:
            raise ValueError("No images found for that series.")
        path = files[min(int(args.get("index", 0)), len(files) - 1)]
    pix, kind = prepare.read_any(path, prepare.SIZE)
    out = os.path.join(tempfile.mkdtemp(prefix="raidiology-"), "slice.png")
    Image.fromarray(pix).save(out)
    return {
        "png": out, "read_as": kind,
        "this_is_for_describing_only": "Look at it and describe what you see for the person, but the "
                                       "yes or no about a fracture must come from read_xray. A model "
                                       "reading this picture by eye scored 0.690 where the measured "
                                       "detector scored 0.976.",
        "not_a_medical_device": NOT_A_DEVICE,
    }


def _ct_volume(path, series_id, folder):
    """One CT series as a NIfTI file in `folder`. The slices are put in the order they are stacked in
    space (image numbers usually follow it and are not required to, and a volume put together in the
    wrong order is a different body); SimpleITK then reads them and places them in space (ITK's DICOM
    geometry, the reference implementation), and the file it writes holds the pixels and their
    geometry, no tag of the originals. Returns (file, the series' images in image-number order, the
    same in the order the volume stacks them, or None for one file holding every slice)."""
    import SimpleITK as sitk  # noqa: PLC0415  arkitect-allow: test-floor-loosened - a CT-only dependency, imported only when a CT is read
    facts = _series_facts(path, series_id)
    if not facts:
        raise ValueError("No images found for series %s under %s" % (series_id, path))
    if facts[0]["modality"] != "CT":
        raise ValueError("read_ct reads a CT series, and this one's modality is %r." % facts[0]["modality"])
    stacked = None
    if len(facts) == 1:
        img = sitk.ReadImage(facts[0]["path"])          # one file holding every slice
    else:
        for f in facts:
            if len(f["iop"]) != 6 or len(f["ipp"]) != 3:
                raise ValueError("%s says nothing about where its slice is, so the series cannot be put "
                                 "together into a volume." % os.path.basename(f["path"]))
        stacked = sorted(facts, key=lambda f: _along_normal(f["iop"], f["ipp"]))
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([f["path"] for f in stacked])
        img = reader.Execute()
    if img.GetDimension() != 3 or min(img.GetSize()) < 2:
        raise ValueError("This series is not a volume (%s voxels), so there is nothing to label in 3D."
                         % "x".join(str(v) for v in img.GetSize()))
    out = os.path.join(folder, "ct.nii")        # not gzipped: it is read once, a minute later, then deleted
    sitk.WriteImage(img, out)
    return out, facts, stacked


def _along_normal(iop, ipp):
    """How far along the slices' normal (row direction x column direction) a slice's corner is."""
    r, c = iop[:3], iop[3:]
    n = (r[1] * c[2] - r[2] * c[1], r[2] * c[0] - r[0] * c[2], r[0] * c[1] - r[1] * c[0])
    return sum(a * b for a, b in zip(n, ipp))


def _run_ct(nifti, folder):
    """The scored reading, `segmenter.py scan`, in a child process: the CT code has modules named like
    the X-ray ones this process already imported, and torch stays out of the server. Returns what
    the reading printed and the structures it found."""
    missing = [m for m in ("torch", "nibabel", "pandas", "psutil") if importlib.util.find_spec(m) is None]
    if missing:
        raise ValueError("Reading a CT needs %s in the Python that runs this server: pip install %s"
                         % (", ".join(missing), " ".join(missing)))
    labels = os.path.join(folder, "labels.nii")
    run = subprocess.run([sys.executable, CT_SEGMENTER, "scan", nifti, "--out", labels,
                          "--version", str(CT_VERSION)],
                         capture_output=True, text=True, encoding="utf-8", timeout=CT_SECONDS, cwd=ROOT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if run.returncode != 0:
        last = (run.stderr.strip().splitlines() or ["no message"])[-1]
        raise ValueError("The CT reading stopped: %s" % last)
    said, _ = json.JSONDecoder().raw_decode(run.stdout.lstrip())
    with open(os.path.join(folder, "labels.labels.json"), encoding="utf-8") as f:
        return said, json.load(f)["found"]


def _ct_answer(said, found, images, facts):
    per = facts["per_structure_test_dice"]
    structures = sorted(({"structure": r["structure"], "in_plain_words": _plain(r["structure"]),
                          "what_it_is": _about(r["structure"]),
                          "millilitres": r["millilitres"],
                          "outline_dice_on_test": per.get(r["structure"], [None, 0])[0],
                          "test_scans_with_it": per.get(r["structure"], [None, 0])[1]} for r in found),
                        key=lambda s: -s["millilitres"])
    md, inv = facts["mean_dice"], facts["invention_rate"]
    return {
        "structures_found": len(structures), "structures": structures,
        "images_in_series": images, "read_on": said["device"], "seconds": said["seconds"],
        "measured_spacing": said["measured_spacing"],
        "note": "" if said["measured_spacing"] else (
            "This scan is not at 1.5 mm on every axis, the spacing the model was measured on, so it "
            "was resampled first. That step has no scorecard behind it."),
        "what_this_is": "Where each organ, bone and large vessel is: one label per voxel, over the 117 "
                        "structures the model knows. It does not look for disease and cannot say "
                        "whether any organ is healthy.",
        "how_good_is_this": {
            "measured_on": "%d held-out CT scans from the TotalSegmentator dataset that it never saw"
                           % facts["test_scans"],
            "mean_dice": round(md[0], 3), "mean_dice_95ci": [round(md[1], 3), round(md[2], 3)],
            "what_dice_means": "How much its outline overlaps an expert's: 1 is the same outline, 0 "
                               "is no overlap at all.",
            "outlines_something_not_there": "%.1f%% of the time (95%% CI %.1f to %.1f)"
                                            % (100 * inv[0], 100 * inv[1], 100 * inv[2]),
            "per_structure": "Each structure above carries its own Dice on the test scans that "
                             "contain it. Trust the low ones least.",
        },
        "least_reliable_here": [s["structure"] for s in sorted(structures, key=lambda s: s["outline_dice_on_test"] or 0)
                                if s["outline_dice_on_test"] is not None and s["outline_dice_on_test"] < 0.7],
        "say_this_too": "The millilitres come from the outline. They were never scored as "
                        "measurements, so they are a rough size, not a clinical volume.",
        "not_a_medical_device": NOT_A_DEVICE,
    }


# The dataset's names in the words a person uses; the rest only lose their underscores.
PLAIN = {"autochthon": "deep back muscle", "clavicula": "collarbone", "costal_cartilages": "rib cartilages",
         "femur": "thigh bone", "gluteus_maximus": "large buttock muscle", "gluteus_medius": "middle buttock muscle",
         "gluteus_minimus": "small buttock muscle", "humerus": "upper arm bone", "iliac_vena": "iliac vein",
         "iliopsoas": "hip flexor muscle", "scapula": "shoulder blade", "sternum": "breastbone",
         "trachea": "windpipe", "urinary_bladder": "bladder", "esophagus": "food pipe (esophagus)",
         "hip": "hip bone", "portal_vein_and_splenic_vein": "portal and splenic veins",
         "pulmonary_vein": "lung veins", "vertebrae": "vertebra", "sacrum": "sacrum (the base of the spine)",
         "duodenum": "duodenum (the start of the small bowel)", "atrial_appendage": "atrial appendage of the heart",
         "brachiocephalic_trunk": "brachiocephalic artery", "common_carotid_artery": "carotid artery"}


# What each structure is, in one line a person without medical training can use. Round five's three
# visitors (2026-10-02) all stopped at names like "inferior vena cava" and "vertebra T10".
ABOUT = {
    "adrenal_gland": "a small gland on top of the kidney that makes hormones",
    "aorta": "the body's main artery, carrying blood out of the heart",
    "atrial_appendage": "a small pouch on the upper chamber of the heart",
    "autochthon": "the long muscles that run either side of the spine",
    "brachiocephalic_trunk": "an artery from the aorta that feeds the right arm and right side of the head",
    "brachiocephalic_vein": "a large vein bringing blood back from the arm and head",
    "brain": "the organ of thought, inside the skull",
    "clavicula": "the bone across the top of the chest out to the shoulder",
    "colon": "the large bowel, which absorbs water from what is left of food",
    "common_carotid_artery": "the main artery up the neck to the head",
    "costal_cartilages": "the flexible gristle joining the ribs to the breastbone",
    "duodenum": "the first part of the small bowel, just after the stomach",
    "esophagus": "the tube that carries food from the mouth to the stomach",
    "femur": "the long bone between the hip and the knee",
    "gallbladder": "a small pouch under the liver that stores bile",
    "gluteus_maximus": "the large muscle of the buttock",
    "gluteus_medius": "a buttock muscle on the side of the hip",
    "gluteus_minimus": "the smallest, deepest buttock muscle",
    "heart": "the muscle that pumps blood around the body",
    "hip": "one side of the pelvis, the ring of bone the legs join",
    "humerus": "the long bone between the shoulder and the elbow",
    "iliac_artery": "an artery carrying blood to the pelvis and leg",
    "iliac_vena": "a vein bringing blood back from the pelvis and leg",
    "iliopsoas": "a deep muscle that lifts the leg at the hip",
    "inferior_vena_cava": "the large vein that brings blood from the lower body back to the heart",
    "kidney": "a kidney, which filters the blood and makes urine",
    "kidney_cyst": "a fluid-filled pocket in a kidney, usually harmless",
    "liver": "the large organ that cleans the blood and makes bile",
    "lung_lower_lobe": "the lowest part of a lung",
    "lung_middle_lobe": "the middle part of the right lung",
    "lung_upper_lobe": "the top part of a lung",
    "pancreas": "a gland behind the stomach that helps digestion and makes insulin",
    "portal_vein_and_splenic_vein": "veins carrying blood from the gut and spleen to the liver",
    "prostate": "a small gland below the bladder, in men",
    "pulmonary_vein": "veins bringing blood from the lungs to the heart",
    "rib": "a rib; they are counted from the top",
    "sacrum": "the triangular bone at the base of the spine",
    "scapula": "the flat triangular bone at the back of the shoulder",
    "skull": "the bones of the head",
    "small_bowel": "the small bowel, where most food is absorbed",
    "spinal_cord": "the bundle of nerves running down the inside of the spine",
    "spleen": "an organ on the left that filters the blood and helps fight infection",
    "sternum": "the flat bone down the middle of the chest, where the ribs meet",
    "stomach": "where food is broken down after it is swallowed",
    "subclavian_artery": "an artery under the collarbone that feeds the arm",
    "superior_vena_cava": "the large vein that brings blood from the upper body back to the heart",
    "thyroid_gland": "a gland in the neck that controls how fast the body works",
    "trachea": "the windpipe, carrying air to the lungs",
    "urinary_bladder": "the bag that stores urine",
}
SPINE = {"C": "a bone of the spine in the neck", "T": "a bone of the spine at chest level, where a rib joins",
         "L": "a bone of the spine in the lower back", "S": "the top bone of the sacrum, at the base of the spine"}


def _about(name):
    """One line on what a structure is, for somebody with no medical training."""
    words = name.split("_")
    stem = "_".join(w for w in words if w not in ("left", "right") and not w[-1].isdigit())
    if stem == "vertebrae":
        return SPINE.get(words[-1][:1], "a bone of the spine")
    return ABOUT.get(stem, "")


def _plain(name):
    """A structure's name as a person says it: kidney_left is "left kidney", rib_left_5 "left rib 5",
    vertebrae_T9 "vertebra T9", autochthon_right "right deep back muscle"."""
    words = name.split("_")
    side = [w for w in words if w in ("left", "right")]
    rest = [w for w in words if w not in ("left", "right")]
    number = [w for w in rest if w[-1].isdigit()]
    stem = "_".join(w for w in rest if w not in number)
    return " ".join(side + [PLAIN.get(stem, stem.replace("_", " "))] + number)


def _ct_outlines(ct_file, labels_file, stacked, numbered):
    """Each image's outlines, from the labels the reading wrote, and where each structure is largest.

    The reading turns the volume to the model's orientation and writes its labels that way; turned
    back, slice k of them lies on image stacked[k], first axis along its columns, second along its
    rows, as SimpleITK built the volume from the files."""
    import nibabel as nib  # noqa: PLC0415  arkitect-allow: test-floor-loosened - a CT-only dependency, imported only when a CT is outlined
    import numpy as np  # noqa: PLC0415  arkitect-allow: test-floor-loosened - imported when a CT is outlined, so `about` pays for nothing
    from skimage import measure  # noqa: PLC0415  arkitect-allow: test-floor-loosened - a CT-only dependency, imported only when a CT is outlined
    from core import findings as ff  # noqa: PLC0415  arkitect-allow: test-floor-loosened - as in write_findings
    ct, lab = nib.load(ct_file), nib.load(labels_file)
    turn = nib.orientations.ornt_transform(nib.io_orientation(lab.affine), nib.io_orientation(ct.affine))
    labels = nib.orientations.apply_orientation(np.asanyarray(lab.dataobj), turn)
    if labels.shape != ct.shape[:3]:
        raise ValueError("The labels came back in another shape (%s) than the scan (%s)." % (labels.shape, ct.shape))
    index = {f["path"]: i for i, f in enumerate(numbered)}
    rows, biggest = [], {}
    for k, f in enumerate(stacked):
        image = labels[:, :, k].T                        # rows by columns, as the image is stored
        paths = []
        for v in (int(v) for v in np.unique(image) if v):
            mask = np.pad(image == v, 1)                 # a structure at the edge still closes
            for c in measure.find_contours(mask.astype(np.uint8), 0.5):
                c = measure.approximate_polygon(c, tolerance=0.6)
                paths.append([v - 1, ff.encode_path([(x - 0.5, y - 0.5) for y, x in c])])
            area = int(mask.sum())
            if area > biggest.get(v, (0, None))[0]:
                biggest[v] = (area, f)
        if paths:
            rows.append({"image_fingerprint": ff.fingerprint(f["sop"]), "image_index": index[f["path"]],
                         "paths": paths})
    return rows, {v: (f, index[f["path"]]) for v, (_n, f) in biggest.items()}


def _ct_entry(ff, v, name, where, size, dice):
    f, at = where
    what = _about(name)
    trust = (" On scans it had never seen, its outline of this matched a specialist's by about %d out of 100."
             % round(100 * dice)) if dice is not None else ""
    return ff.finding(id="s%d" % v, label=_plain(name), image_fingerprint=ff.fingerprint(f["sop"]),
                      image_index=at, plain="%s%s: about %s ml.%s" % (_plain(name).capitalize(),
                                                                       ", " + what if what else "",
                                                                       format(size, ",.0f"), trust))


def _ct_findings(args, numbered, stacked, folder, found, facts):
    """The reading as a findings file: every structure an entry that opens the image where it is
    largest, and every image's outlines, for rAidiology's viewer to draw over the person's own slices."""
    from core import findings as ff  # noqa: PLC0415  arkitect-allow: test-floor-loosened - as in write_findings
    with open(os.path.join(folder, "labels.labels.json"), encoding="utf-8") as fh:
        names = {int(k): v for k, v in json.load(fh)["labels"].items()}
    rows, biggest = _ct_outlines(os.path.join(folder, "ct.nii"), os.path.join(folder, "labels.nii"), stacked, numbered)
    per = facts["per_structure_test_dice"]
    size = {r["structure"]: r["millilitres"] for r in found}
    entries = [_ct_entry(ff, v, names[v], biggest[v], size.get(names[v], 0), per.get(names[v], [None])[0])
               for v in sorted(biggest, key=lambda v: -size.get(names[v], 0))]
    md, inv = facts["mean_dice"], facts["invention_rate"]
    head = numbered[0]
    doc = ff.document(
        scan={"modality": head["modality"], "description": head["description"], "body_part": head["body_part"],
              "images": len(numbered), "series_fingerprint": ff.fingerprint(head["match"]), "ordered_by": "image number"},
        model={"name": "rAidiology CT segmenter", "version": str(CT_VERSION),
               "measured_on": "%d held-out CT scans from the TotalSegmentator dataset" % facts["test_scans"],
               "mean_dice": round(md[0], 3), "mean_dice_95ci": [round(md[1], 3), round(md[2], 3)],
               "invents": "It outlines a structure that is not there %.1f%% of the time" % (100 * inv[0]),
               "who_it_is_not_for": "It was trained on adult CT. It finds anatomy, not disease.",
               "caveat": "These outlines were drawn by a computer program, not by a doctor. They show "
                         "where each part of the body is, not whether it is healthy."},
        findings=entries,
        summary="%d structures outlined on %d images: where each organ, bone and large vessel is, not "
                "whether anything is wrong with it." % (len(entries), len(rows)),
        tool="rAidiology MCP server %s" % VERSION,
        outlined=ff.outlines(structures=[_plain(names[v]) for v in sorted(names)],
                             about=[_about(names[v]) for v in sorted(names)],
                             dice_on_test=[per.get(names[v], [None])[0] for v in sorted(names)], slices=rows,
                             note="Each path is a structure's outline in the image's own pixels; the "
                                  "number before it is the structure's place in `structures`."))
    return ff.write(doc, _destination(args.get("out"), args.get("path")))


def tool_read_ct(args):
    with open(CT_FACTS, encoding="utf-8") as f:
        facts = json.load(f)
    folder = tempfile.mkdtemp(prefix="raidiology-")
    try:
        if args.get("file"):
            nifti, images = args["file"], None
            if not nifti.lower().endswith((".nii", ".nii.gz")):
                raise ValueError("file must be a NIfTI volume (.nii or .nii.gz); for a DICOM series "
                                 "give path and series_id.")
        else:
            nifti, numbered, stacked = _ct_volume(args["path"], args["series_id"], folder)
            images = len(numbered)
        said, found = _run_ct(nifti, folder)
        wrote = None
        if args.get("findings"):
            if args.get("file") or not stacked:
                raise ValueError("A findings file needs a DICOM series of one image a file (path and "
                                 "series_id): the viewer matches outlines to the images it is shown.")
            wrote = _ct_findings(args, numbered, stacked, folder, found, facts)
    finally:
        shutil.rmtree(folder, ignore_errors=True)       # a copy of somebody's scan and its labels
    answer = _ct_answer(said, found, images, facts)
    if wrote:
        answer["findings_file"] = {"wrote": wrote[0], "bytes": wrote[2], "tell_the_person": (
            "Their reading is saved at %s. They open rAidiology's %s in a browser, drop in the scan "
            "folder and then this file, and see their own slices with each structure outlined. The "
            "file holds no name, patient number or date of birth, and opening it uploads nothing."
            % (wrote[0], VIEWER or "\u201cOpen your own scan\u201d page"))}
    return answer


def _match_uid(d):
    """The series identifier used only for matching this reading to the pictures in the browser.

    Deliberately not `_series_key`: that one's job is to keep two different series apart, and it
    falls back to a folder name, which the browser cannot see. This one falls back the way the
    browser's own reader does, so the fingerprints line up on both sides. It is hashed before it
    goes anywhere near the file."""
    return (str(getattr(d, "SeriesInstanceUID", "") or "").strip()
            or str(getattr(d, "StudyInstanceUID", "") or "").strip()
            or "unknown-series")


def _writable(folder):
    try:
        os.makedirs(folder, exist_ok=True)
        probe = os.path.join(folder, ".raidiology-write-test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("")
        os.remove(probe)
        return True
    except OSError:
        return False                    # a hospital CD is read-only, which is normal, not an error


MODEL_CARD = {
    "name": "rAidiology X-ray fracture detector", "version": "1",
    "measured_on": "4,170 held-out paediatric wrist radiographs from 1,218 patients it had never "
                   "seen, plus 798 mixed hand, leg, hip and shoulder radiographs",
    "auroc": 0.976, "auroc_95ci": [0.970, 0.981],
    "catches": "92.7% of fractures (95% CI 91.2 to 94.1)",
    "clears": "94.4% of unbroken images correctly (92.4 to 96.1)",
    "who_it_is_not_for": "It was trained mostly on children's wrists, with some hands, legs, hips "
                         "and shoulders. A chest, skull or spine X-ray is outside what it was "
                         "measured on.",
    "caveat": "The highlight shows where the score came from, not where a fracture is. Location "
              "was never scored.",
}


def _sop_of(path):
    import pydicom  # noqa: PLC0415
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return str(getattr(d, "SOPInstanceUID", "") or "")
    except Exception:  # noqa: BLE001  a file with no readable header still gets a finding, by index
        return ""


def _findings_from(rows, cut):
    from core import findings as ff  # noqa: PLC0415
    out = []
    for i, r in enumerate(rows):
        if not r["fracture"]:
            continue
        close = r["score"] < cut + 0.10
        out.append(ff.finding(
            id="f%d" % (len(out) + 1), label="possible fracture",
            plain=("Image %d of %d is the kind of picture the detector calls broken%s."
                   % (i + 1, len(rows), ", though only just" if close else "")),
            score=r["score"], confidence="just above the line" if close else "well above the line",
            image_fingerprint=ff.fingerprint(_sop_of(r["path"])), image_index=i,
            where=ff.grid(r["map"], r["map_size"][0], r["map_size"][1])))
    return out


def _destination(out, source):
    """Where to put the file: what they asked for, else beside the scan, else a temporary folder.

    Beside the scan is the right default and often impossible, because a hospital CD is read-only.
    Each candidate is tried rather than assumed, and the temporary folder at the end means the
    reading is never simply lost."""
    choices = [out] if out else []
    if source:
        choices.append(os.path.dirname(os.path.abspath(source)))
    choices.append(tempfile.mkdtemp(prefix="raidiology-"))
    for c in choices:
        if not c:
            continue
        folder = c if (os.path.isdir(c) or not os.path.splitext(c)[1]) else os.path.dirname(c)
        if _writable(folder or "."):
            return c
    raise ValueError("Nowhere to write the file: pass `out` with a folder you can write to.")


def _summary(found, rows, cut, whole):
    top = max((r["score"] for r in rows), default=None)
    if found:
        text = ("%d of the %d image(s) read scored above the line the detector was scored at. That "
                "is a reason to have it looked at, not a diagnosis: it calls about 6 unbroken "
                "images in 100 broken." % (len(found), len(rows)))
    else:
        text = ("Nothing on the %d image(s) read reached the line the detector was scored at. The "
                "highest any of them scored was %.2f against a line of %.2f. It misses about 7 "
                "fractures in 100, so this is not an all-clear." % (len(rows), top or 0.0, cut))
    if not whole:
        text += (" Only the first %d images of this series were read, so this is about those."
                 % len(rows))
    return text


def tool_write_findings(args):
    import pydicom  # noqa: PLC0415
    from core import findings as ff  # noqa: PLC0415
    reader = _lazy()
    import datasets  # noqa: PLC0415
    profile = str(args.get("body_part_profile") or "grazpedwri").strip()
    if profile not in datasets.NAMES:
        raise ValueError("body_part_profile must be one of %s; got %r." % (", ".join(datasets.NAMES), profile))
    source = args.get("path")
    if args.get("file"):
        paths = [args["file"]]
    else:
        paths = _series_files(source, args["series_id"])
        if not paths:
            raise ValueError("No images found for series %s under %s" % (args.get("series_id"), source))
    whole = len(paths) <= MAX_IMAGES
    paths = paths[:MAX_IMAGES]

    head = pydicom.dcmread(paths[0], stop_before_pixels=True, force=True)
    scan = {"modality": str(getattr(head, "Modality", "") or ""),
            "description": str(getattr(head, "SeriesDescription", "") or
                               getattr(head, "StudyDescription", "") or "untitled"),
            "body_part": str(getattr(head, "BodyPartExamined", "") or ""),
            "images": len(paths),
            "series_fingerprint": ff.fingerprint(_match_uid(head)),
            "ordered_by": "image number"}

    rows, meta = reader.score_images_with_maps(paths, 1, profile)
    cut = meta["thresholds"][profile]
    found = _findings_from(rows, cut)
    model = dict(MODEL_CARD, profile=profile, threshold=cut)
    doc = ff.document(scan=scan, model=model, findings=found,
                      summary=_summary(found, rows, cut, whole),
                      tool="rAidiology MCP server %s" % VERSION)
    where, raw, packed = ff.write(doc, _destination(args.get("out"), source))
    return {
        "wrote": where, "bytes": packed, "uncompressed_bytes": raw,
        "findings": len(found), "images_read": len(rows), "whole_series_read": whole,
        "summary": doc["summary"],
        "tell_the_person": ("Their reading is saved at %s. They open rAidiology's %s in a browser, "
                            "drop in the scan folder and then this file, and see the images with "
                            "what was found marked on them. The file holds no name, patient number "
                            "or date of birth, and opening it uploads nothing."
                            % (where, VIEWER or "“Open your own scan” page")),
        "not_a_medical_device": NOT_A_DEVICE,
    }


TOOLS = [
    {"name": "about", "fn": tool_about,
     "description": "What rAidiology is, what it has measured, and how to use these tools. Call it "
                    "first.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "find_scans", "fn": tool_find_scans,
     "description": "List the scan series in a folder or ZIP from a hospital CD. Reads the files on "
                    "this machine; nothing is uploaded, and no name or patient number is read.",
     "inputSchema": {"type": "object", "required": ["path"], "properties": {
         "path": {"type": "string", "description": "A folder or a .zip holding DICOM files"}}}},
    {"name": "read_xray", "fn": tool_read_xray,
     "description": "Run rAidiology's measured fracture detector on a radiograph. This is where the "
                    "yes or no comes from, not from looking at the picture.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "The folder or ZIP, when using series_id"},
         "series_id": {"type": "string", "description": "From find_scans"},
         "file": {"type": "string", "description": "A single image file instead of a series"},
         "body_part_profile": {"type": "string", "enum": ["grazpedwri", "fracatlas"],
                               "description": "grazpedwri for a wrist, fracatlas for a hand, leg, "
                                              "hip or shoulder"}}}},
    {"name": "read_ct", "fn": tool_read_ct,
     "description": "Label the organs, bones and large vessels in a CT with rAidiology's measured "
                    "segmenter: what it found, roughly how large each is, and how reliable that "
                    "kind of outline was on held-out scans. It finds anatomy, not disease. Allow a "
                    "minute or more on a computer without a graphics card.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "The folder or ZIP, when using series_id"},
         "series_id": {"type": "string", "description": "From find_scans: a series whose modality is CT"},
         "file": {"type": "string", "description": "A NIfTI volume (.nii or .nii.gz) instead of a DICOM series"},
         "findings": {"type": "boolean", "description": "Also save a findings file the person can open "
                                                         "in rAidiology's viewer to see every structure "
                                                         "outlined on their own slices"},
         "out": {"type": "string", "description": "Where to save the findings file. Defaults to beside the scan."}}}},
    {"name": "render_slice", "fn": tool_render_slice,
     "description": "Write one image as a PNG you can look at, so you can describe it. Describing "
                    "only: take the verdict from read_xray.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"}, "series_id": {"type": "string"},
         "file": {"type": "string"}, "index": {"type": "integer"}}}},
    {"name": "write_findings", "fn": tool_write_findings,
     "description": "Save the reading as a small file the person can open in rAidiology's own "
                    "viewer, which shows them their images with what was found marked on them. "
                    "Call this after read_xray. The file carries no name, patient number or date "
                    "of birth.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "The folder or ZIP, when using series_id"},
         "series_id": {"type": "string", "description": "From find_scans"},
         "file": {"type": "string", "description": "A single image file instead of a series"},
         "body_part_profile": {"type": "string", "enum": ["grazpedwri", "fracatlas"]},
         "out": {"type": "string", "description": "Where to save it. A folder, or a file name. "
                                                  "Defaults to beside the scan."}}}},
]
BY_NAME = {t["name"]: t for t in TOOLS}


def handle(msg):
    method = msg.get("method")
    if method == "initialize":
        return {"protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                "serverInfo": {"name": NAME, "version": VERSION},
                "instructions": "rAidiology reads a medical scan on this machine. Call about first. "
                                "Every graded answer comes from a model that was scored against a "
                                "radiologist; your part is finding the files and explaining the "
                                "result in plain words. " + NOT_A_DEVICE}
    if method == "tools/list":
        return {"tools": [{k: t[k] for k in ("name", "description", "inputSchema")} for t in TOOLS]}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        if name not in BY_NAME:
            raise ValueError("unknown tool %r" % name)
        result = BY_NAME[name]["fn"](params.get("arguments") or {})
        return {"content": [{"type": "text", "text": json.dumps(result, indent=1)}]}
    if method in ("notifications/initialized", "ping"):
        return {}
    raise ValueError("unknown method %r" % method)


def serve(stdin=None, stdout=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "id" not in msg:                       # a notification: act, answer nothing
            try:
                handle(msg)
            except Exception:  # noqa: BLE001
                pass
            continue
        try:
            out = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
        except Exception as e:  # noqa: BLE001  an error goes back as an answer, never as a crash
            out = {"jsonrpc": "2.0", "id": msg["id"],
                   "error": {"code": -32000, "message": "%s: %s" % (type(e).__name__, e),
                             "data": traceback.format_exc(limit=3)}}
        stdout.write(json.dumps(out) + "\n")
        stdout.flush()


def selftest(path=None):
    """Drive the server the way a client would, so a change that breaks the protocol is obvious.

    With no folder it makes one up (`demo_scan`) and runs the whole chain on it, including writing
    a findings file and reading it back. That is the part worth having automatic: the findings file
    is a contract between this server and a browser written in another language, and a reading that
    quietly stops matching the pictures it is about would otherwise be found by a stranger looking
    at their own scan."""
    import io  # noqa: PLC0415
    made = None
    if not path:
        import demo_scan  # noqa: PLC0415
        made = tempfile.mkdtemp(prefix="raidiology-demo-")
        demo_scan.write(made)
        path = made
    asks = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "about", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "find_scans", "arguments": {"path": path}}}]
    if made:
        import demo_scan  # noqa: PLC0415
        asks += [
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "read_xray", "arguments": {"path": path, "series_id": demo_scan.SERIES_UID}}},
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
             "params": {"name": "write_findings",
                        "arguments": {"path": path, "series_id": demo_scan.SERIES_UID,
                                      "out": os.path.join(made, "demo")}}},
        ]
    out = io.StringIO()
    serve(io.StringIO("\n".join(json.dumps(a) for a in asks) + "\n"), out)
    bad = 0
    for line in out.getvalue().splitlines():
        r = json.loads(line)
        if "error" in r:
            bad += 1
            print("FAIL id %s: %s" % (r.get("id"), r["error"]["message"]))
        elif r["id"] == 2:
            print("tools:", ", ".join(t["name"] for t in r["result"]["tools"]))
        elif r["id"] == 4:
            d = json.loads(r["result"]["content"][0]["text"])
            print("find_scans: %d file(s) looked at, %d series" % (d["files_looked_at"], len(d["series"])))
            for s in d["series"][:6]:
                print("   %-4s %-34s %4d image(s)  %s" % (s["modality"], s["description"][:34],
                                                          s["images"], s["size"]))
        elif r["id"] == 5:
            d = json.loads(r["result"]["content"][0]["text"])
            print("read_xray: %s (highest %.3f against %.3f)"
                  % (d["answer"], d["highest_score"] or 0.0, d["threshold"]))
        elif r["id"] == 6:
            d = json.loads(r["result"]["content"][0]["text"])
            print("write_findings: %d finding(s), %d bytes gzipped from %d"
                  % (d["findings"], d["bytes"], d["uncompressed_bytes"]))
            from core import findings as ff  # noqa: PLC0415
            import demo_scan  # noqa: PLC0415
            doc = ff.read(d["wrote"])
            want = ff.fingerprint(demo_scan.SERIES_UID)
            if doc["scan"]["series_fingerprint"] != want:
                bad += 1
                print("FAIL: the findings file points at a different series than it was written from")
            else:
                print("findings file: reads back, and its series fingerprint is %s" % want)
    print("selftest: %d answer(s), %d failed" % (len(out.getvalue().splitlines()), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", nargs="?", const="", metavar="FOLDER_OR_ZIP")
    a = ap.parse_args()
    if a.selftest is not None:
        sys.exit(selftest(a.selftest or None))
    serve()
