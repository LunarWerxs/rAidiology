"""Download the X-ray datasets from figshare or Mendeley Data, check them, and unpack them.

All are CC BY 4.0 and have a row in LICENSING.md (added before this script was first run):
  fracatlas   FracAtlas, 4,073 musculoskeletal radiographs   doi:10.6084/m9.figshare.22363012
  grazpedwri  GRAZPEDWRI-DX, 20,327 paediatric wrist images  doi:10.6084/m9.figshare.14825193
  orthofrac   OrthoFrac-XR, 1,493 limb radiographs           doi:10.6084/m9.figshare.32021085
  bonefract   BoneFract, 47,931 limb and pelvis radiographs  doi:10.17632/4cr7f2359x.1 (Mendeley)
  pediashf    PediaSHF-DX, 10,325 children's elbow images    doi:10.6084/m9.figshare.29322869

The engine (core/fetch.py) does the work. The file list and the licence come from the archive's API at
run time, not from this file, so a new dataset version is noticed instead of silently skipped. Every
download resumes under an honest agent string, is checked against the archive's own checksum
(figshare's md5, Mendeley's sha256), and is not unpacked unless it matches. Nothing lands in the
repository: everything goes under <Datasets>/public/<name>/ (core/paths.py).

Usage:
  python modalities/xray/fetch_data.py [all|fracatlas|grazpedwri|orthofrac|bonefract|pediashf] [--no-extract] [--jobs 3]
"""
import argparse
import concurrent.futures
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core import fetch, paths  # noqa: E402

SETS = {
    "fracatlas": {"article": 22363012, "dir": "fracatlas", "licence": "CC BY 4.0"},
    "grazpedwri": {"article": 14825193, "dir": "grazpedwri-dx", "licence": "CC BY 4.0"},
    "orthofrac": {"article": 32021085, "dir": "orthofrac-xr", "licence": "CC BY 4.0"},
    "bonefract": {"mendeley": "4cr7f2359x", "version": 1, "dir": "bonefract", "licence": "CC BY 4.0"},
    "pediashf": {"article": 29322869, "dir": "pediashf-dx", "licence": "CC BY 4.0"},
    "diufrac": {"mendeley": "czvd3sdymh", "version": 1, "dir": "diu-multiclass-fracture", "licence": "CC BY 4.0"},
}


def listing(name, spec, ctx):
    if "article" in spec:
        return fetch.figshare(name, ctx, articles=[spec["article"]])
    return fetch.mendeley(name, ctx, spec["mendeley"], spec["version"])


def extract(path, out_dir, label):
    marker = path + ".extracted"
    if os.path.exists(marker):
        print("%s: already unpacked" % label, flush=True)
        return
    n = fetch.unpack(path, out_dir)
    with open(marker, "w") as f:
        f.write("%d entries\n" % n)
    print("%s: unpacked %d entries" % (label, n), flush=True)


def fetch_set(name, extract_zips=True, jobs=3):
    spec = SETS[name]
    ctx = fetch.SetContext(name)
    got = listing(name, spec, ctx)
    lic = fetch.check_licence(name, got.licence, {fetch.licence_id(spec["licence"])})
    out = os.path.join(paths.DATASETS, "public", spec["dir"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "figshare.json" if "article" in spec else "mendeley.json"), "w", encoding="utf-8") as f:
        json.dump({"article": spec.get("article") or spec["mendeley"], "version": spec.get("version"),
                   "source": got.source, "notes": got.notes, "licence": got.licence, "licence_id": lic,
                   "files": [{"name": e.name, "size": e.size, "checksum": e.checksum} for e in got.files]},
                  f, indent=2)
    print("%s: licence %s, %d file(s), %.2f GB" % (name, got.licence, len(got.files), got.size() / 1e9), flush=True)

    def one(e):
        dst = os.path.join(out, *e.name.split("/"))
        label = "%s/%s" % (name, e.name)
        if not os.path.exists(dst) and os.path.exists(dst + ".extracted"):
            print("%s: unpacked earlier and the archive since removed; not downloaded again" % label, flush=True)
            return label
        if os.path.exists(dst) and fetch.recheck(e, dst):
            print("%s: already here and its checksum matches" % label, flush=True)
        else:
            rec = fetch.download(e, dst, ctx, label)
            print("%s: %.1f MB in %d s, %s" % (label, rec["size"] / 1e6, rec["seconds"], rec["checked"]), flush=True)
        if extract_zips and fetch.is_archive(dst):
            extract(dst, out, label)
        return label

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, got.files))
    if name == "grazpedwri" and extract_zips:
        tidy_grazpedwri(out)
    print("%s: done, in %s" % (name, out), flush=True)


def tidy_grazpedwri(out):
    """The four image zips unpack their PNGs into the dataset root; put them where the dataset's own
    folder_structure.zip expects them (yolov5/images/), beside the labels."""
    images = os.path.join(out, "yolov5", "images")
    os.makedirs(images, exist_ok=True)
    moved = 0
    for n in os.listdir(out):
        if n.lower().endswith(".png"):
            os.replace(os.path.join(out, n), os.path.join(images, n))
            moved += 1
    print("grazpedwri: %d image(s) moved into yolov5/images (%d there now)"
          % (moved, len(os.listdir(images))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["all"] + sorted(SETS))
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    for name in (sorted(SETS) if a.which == "all" else [a.which]):
        fetch_set(name, extract_zips=not a.no_extract, jobs=a.jobs)


if __name__ == "__main__":
    main()
