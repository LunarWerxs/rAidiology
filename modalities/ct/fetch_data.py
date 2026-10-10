"""Download the CT datasets from Zenodo, check them, and unpack them.

  totalsegmentator  1,228 clinical CT scans with masks of 117 anatomical structures, CC BY 4.0,
                    doi:10.5281/zenodo.10047292, 23.6 GB in one zip. Its row is in LICENSING.md,
                    written before this script was first run.

The engine (core/fetch.py) does the work. The file list and the licence come from the Zenodo API at
run time, not from this file, so a new version or a changed licence is noticed instead of silently
skipped: a licence that stops saying CC BY 4.0 stops the download. Every file arrives by resumable byte
ranges under an honest agent string, is checked against Zenodo's md5, and is not unpacked unless it
matches. Nothing lands in the repository: everything goes under <Datasets>/public/ (core/paths.py).

Usage: python modalities/ct/fetch_data.py [all|totalsegmentator] [--no-extract]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core import fetch, paths  # noqa: E402

SETS = {
    "totalsegmentator": {"record": 10047292, "dir": "totalsegmentator", "licence": "cc-by-4.0"},
}


def extract(path, out_dir, label):
    marker = path + ".extracted"
    if os.path.exists(marker):
        print("%s: already unpacked" % label, flush=True)
        return
    n = fetch.unpack(path, out_dir)
    with open(marker, "w") as f:
        f.write("%d entries\n" % n)
    print("%s: unpacked %d entries" % (label, n), flush=True)


def fetch_set(name, extract_zips=True):
    spec = SETS[name]
    ctx = fetch.SetContext(name)
    listing = fetch.zenodo(name, [spec["record"]], ctx)
    lic = fetch.check_licence(name, listing.licence, {spec["licence"]})
    out = os.path.join(paths.DATASETS, "public", spec["dir"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "zenodo.json"), "w", encoding="utf-8") as f:
        json.dump({"record": spec["record"], "source": listing.source, "notes": listing.notes, "licence": lic,
                   "files": [{"key": e.name, "size": e.size, "checksum": e.checksum} for e in listing.files]},
                  f, indent=2)
    print("%s: %s, licence %s, %d file(s), %.2f GB"
          % (name, "; ".join(listing.notes), lic, len(listing.files), listing.size() / 1e9), flush=True)
    for e in listing.files:
        dst = os.path.join(out, *e.name.split("/"))
        label = "%s/%s" % (name, e.name)
        if not os.path.exists(dst) and os.path.exists(dst + ".extracted"):
            print("%s: unpacked earlier and the archive since removed; not downloaded again" % label, flush=True)
            continue
        if os.path.exists(dst) and fetch.recheck(e, dst):
            print("%s: already here and its checksum matches" % label, flush=True)
        else:
            rec = fetch.download(e, dst, ctx, label)
            print("%s: %.2f GB in %d s, %s" % (label, rec["size"] / 1e9, rec["seconds"], rec["checked"]), flush=True)
        if extract_zips and fetch.is_archive(dst):
            extract(dst, out, label)
    print("%s: done, in %s" % (name, out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["all"] + sorted(SETS))
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    for name in (sorted(SETS) if a.which == "all" else [a.which]):
        fetch_set(name, extract_zips=not a.no_extract)


if __name__ == "__main__":
    main()
