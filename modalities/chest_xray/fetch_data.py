"""Download NIH ChestX-ray14 from NIH's own Box folder, check it, and unpack it.

112,120 frontal chest X-rays of 30,805 patients from the NIH Clinical Center, each with up to 14
findings mined from its report (NIH's README: over 90% accurate), NIH's own train/test lists by
patient, and 984 boxes drawn by a radiologist. LICENSING.md has the row (added before this was
first run): NIH's FAQ says "The usage of the data set is unrestricted", with a link to the download
site, an acknowledgement of the NIH Clinical Center and the CVPR 2017 citation.

The twelve image archives are the links in NIH's own `batch_download_zips.py`; the four tables come
by Box's shared-file download. Box publishes no checksums through these links, so each file is
checked against the byte size NIH's folder lists, and gzip's own checksums are checked as each
archive unpacks (a damaged archive stops the unpack). Every download resumes. Nothing lands in the
repository: everything goes under <Datasets>/public/nih-chestxray14/ (core/paths.py).

Usage: python modalities/chest_xray/fetch_data.py [--no-extract] [--jobs 3]"""
import argparse
import concurrent.futures
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core import fetch, paths  # noqa: E402

STATIC = "https://nihcc.box.com/shared/static/%s.gz"
SHARED = ("https://nihcc.app.box.com/index.php?rm=box_download_shared_file"
          "&vanity_name=ChestXray-NIHCC&file_id=f_%d")
# (file, link, bytes as NIH's Box folder lists them, 2026-10-03)
ARCHIVES = [
    ("images_001.tar.gz", STATIC % "vfk49d74nhbxq3nqjg0900w5nvkorp5c", 2008470987),
    ("images_002.tar.gz", STATIC % "i28rlmbvmfjbl8p2n3ril0pptcmcu9d1", 3952623504),
    ("images_003.tar.gz", STATIC % "f1t00wrtdk94satdfb9olcolqx20z2jp", 3929234850),
    ("images_004.tar.gz", STATIC % "0aowwzs5lhjrceb3qp67ahp0rd1l1etg", 3838903983),
    ("images_005.tar.gz", STATIC % "v5e3goj22zr6h8tzualxfsqlqaygfbsn", 3935496531),
    ("images_006.tar.gz", STATIC % "asi7ikud9jwnkrnkj99jnpfkjdes7l6l", 3986301172),
    ("images_007.tar.gz", STATIC % "jn1b4mw4n6lnh74ovmcjb8y48h8xj07n", 4016328426),
    ("images_008.tar.gz", STATIC % "tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j", 4018347353),
    ("images_009.tar.gz", STATIC % "upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj", 4111327929),
    ("images_010.tar.gz", STATIC % "l6nilvfa9cg3s28tqv1qc1olm3gnz54p", 4181556296),
    ("images_011.tar.gz", STATIC % "hhq8fkdgvcari67vfhs7ppg2w6ni4jze", 4187084020),
    ("images_012.tar.gz", STATIC % "ioqwiy20ihqwyr8pf4c24eazhh281pbu", 2914187733),
]
TABLES = [
    ("Data_Entry_2017_v2020.csv", SHARED % 219760887468, 9003496),
    ("BBox_List_2017.csv", SHARED % 219760940956, 92416),
    ("train_val_list.txt", SHARED % 256056636701, 1470907),
    ("test_list.txt", SHARED % 256055473534, 435131),
]
OUT = os.path.join(paths.DATASETS, "public", "nih-chestxray14")
CTX = fetch.SetContext("nih-chestxray14")


def unpack(path):
    """Unpack one archive into OUT (its members are images/<name>.png); refuse escaping paths."""
    marker = path + ".extracted"
    if os.path.exists(marker):
        return "already unpacked"
    n = 0
    with tarfile.open(path, "r:gz") as t:
        for m in t:
            p = os.path.normpath(os.path.join(OUT, m.name))
            if not p.startswith(os.path.abspath(OUT) + os.sep) or not (m.isfile() or m.isdir()):
                raise SystemExit("%s: entry escapes the folder or is not a file: %s" % (path, m.name))
            t.extract(m, OUT)
            n += m.isfile()
    with open(marker, "w") as f:
        f.write("%d files\n" % n)
    return "unpacked %d files" % n


def one(item, extract):
    name, url, size = item
    dst = os.path.join(OUT, name)
    if not os.path.exists(dst) and os.path.exists(dst + ".extracted"):
        print("%s: unpacked earlier and the archive since removed; not downloaded again" % name, flush=True)
        return
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        msg = "already here, %d bytes" % size
    else:                                                  # core.fetch retries Box's dropped connections
        rec = fetch.download(fetch.Entry(name, size, url), dst, CTX, name)
        secs = rec["seconds"]
        msg = "%.1f MB in %.0f s (%.1f MB/s)" % (size / 1e6, secs, size / 1e6 / max(secs, 1e-9))
    if os.path.getsize(dst) != size:
        raise SystemExit("%s: %d bytes, NIH lists %d. Delete it and run again." % (name, os.path.getsize(dst), size))
    if extract and name.endswith(".tar.gz"):
        msg += "; " + unpack(dst)
    print("%s: %s" % (name, msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("NIH ChestX-ray14: %d archives and %d tables, %.1f GB, into %s"
          % (len(ARCHIVES), len(TABLES), sum(s for _, _, s in ARCHIVES + TABLES) / 1e9, OUT), flush=True)
    for t in TABLES:
        one(t, False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(lambda it: one(it, not a.no_extract), ARCHIVES))
    pngs = len(os.listdir(os.path.join(OUT, "images"))) if os.path.isdir(os.path.join(OUT, "images")) else 0
    print("nih-chestxray14: done, %d pictures in images/" % pngs, flush=True)


if __name__ == "__main__":
    main()
