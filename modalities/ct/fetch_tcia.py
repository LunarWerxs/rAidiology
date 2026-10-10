"""Download a TCIA collection's CT scans and its answer files.

TCIA's public NBIA API serves every series as one zip, at about 3.6 MB/s per connection (measured
2026-10-03), so eight series come at once. Each collection here has its LICENSING.md row, written
before it was first fetched, and the licence that row names: the series list and every series'
licence are read from the API at run time, so a series that changes or a licence that stops saying
what LICENSING.md says is noticed instead of skipped.

- `lidc-idri`: 1,018 CT series of 1,010 patients (128 GB), each read by four thoracic radiologists
  who marked and rated every nodule (`LIDC-XML-only.zip`); the one public set where the reader can
  be scored beside each radiologist against the other three (docs/VS_DOCTORS.md).
- `spie-aapm`: the SPIE-AAPM Lung CT Challenge, 70 CT series of 70 patients (12.1 GB), with each
  nodule's place and its benign or malignant diagnosis (the two `*NoduleData*.xlsx`).
- `lidc-annot-nlst501`: LIDC-annot-NLST501, 501 NLST low-dose CT series of 501 patients (about 44 GB),
  each read by four of 20 board-certified radiologists who marked and rated every lesion; patient-level
  lung cancer from NLST: 152 cancer, 349 not. Its images are in collection NLST on the NLST host, and the
  series to fetch are the ones its manifest lists (`source-manifest-LIDC-annot-NLST501.tcia`).

A series counts as done when its folder holds as many DICOM files as TCIA lists and a marker says it
was unpacked; a run cut short starts again where it stopped. Nothing lands in the repository:
everything goes under <Datasets>/public/<name>/ (core/paths.py), one folder per series:
<PatientID>/<SeriesInstanceUID>/*.dcm.

Usage: python modalities/ct/fetch_tcia.py lidc-idri|spie-aapm|lidc-annot-nlst501 [--jobs 8] [--limit N]"""
import argparse
import collections
import concurrent.futures
import csv
import json
import os
import shutil
import sys
import time
import urllib.parse
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core import fetch, paths  # noqa: E402

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
SITE = "https://www.cancerimagingarchive.net/wp-content/uploads/"
NLST_API = "https://nlst.cancerimagingarchive.net/nbia-api/services/v1/"
CC_BY_3 = "Creative Commons Attribution 3.0 Unported License"
CC_BY_4 = "Creative Commons Attribution 4.0 International License"
COLLECTIONS = {
    "lidc-idri": {"collection": "LIDC-IDRI", "licence": CC_BY_3,
                  "extras": ["LIDC-XML-only.zip", "lidc-idri-nodule-counts-6-23-2015.xlsx",
                             "tcia-diagnosis-data-2012-04-20.xls"]},
    "spie-aapm": {"collection": "SPIE-AAPM Lung CT Challenge", "licence": CC_BY_3,
                  "extras": ["CalibrationSet_NoduleData.xlsx", "TestSet_NoduleData_PublicRelease_wTruth.xlsx"]},
    "lidc-annot-nlst501": {
        "collection": "NLST", "licence": CC_BY_4, "api": NLST_API,
        "manifest": "source-manifest-LIDC-annot-NLST501.tcia", "series_csv": "LIDC-annot-NLST501-series.csv",
        "correction_csv": "LIDC-annot-NLST501-correction.csv",
        "extras": ["source-manifest-LIDC-annot-NLST501.tcia", "LIDC-annot-NLST501-series.csv",
                   "LIDC-annot-NLST501-annotations.csv", "LIDC-annot-NLST501-patients.csv",
                   "LIDC-annot-NLST501-readers.csv", "LIDC-annot-NLST501-correction.csv",
                   "LIDC-annot-NLST501_TCIA_documentation.pdf", "LIDC-annot-NLST501_CorrectedCT.tcia",
                   "LIDC-annot-NLST501_Seg.tcia"]},
    "nlst-labelled": {                                 # tools/nlst_series.py writes both files first
        "collection": "NLST", "licence": CC_BY_4, "api": NLST_API,
        "manifest": "nlst-labelled.tcia", "series_csv": "nlst-labelled-series.csv", "extras": []},
}
ATTEMPTS = 3
CTX = fetch.SetContext("fetch_tcia")


def get(url, timeout=600):
    """core.fetch's request: the honest agent string, TCIA's connection cap, retries, and a rest on 403/429."""
    return fetch.opened(url, CTX, timeout=timeout)


def manifest_series(c, out):
    """The manifest's series that exist: its UIDs, each looked up under its patient (from the series CSV)."""
    with open(os.path.join(out, c["manifest"]), encoding="utf-8") as f:
        lines = [x.strip() for x in f]
    uids = [x for x in lines[lines.index("ListOfSeriesToDownload=SeriesInstanceUID") + 1:] if x]
    with open(os.path.join(out, c["series_csv"]), encoding="utf-8-sig", newline="") as f:
        pid = {row["series_uid"]: row["pid"] for row in csv.DictReader(f)}
    if "correction_csv" in c:
        with open(os.path.join(out, c["correction_csv"]), encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):              # the series CSV holds the corrected copy of these;
                pid.setdefault(row["series_uid_original"], row["pid"])  # the manifest lists the original
    unknown = [u for u in uids if u not in pid]
    if unknown:
        raise SystemExit("%d manifest series have no patient in %s: %s" % (len(unknown), c["series_csv"], unknown[:5]))
    want = set(uids)

    def patient(p):
        url = "%sgetSeries?Collection=%s&PatientID=%s&format=json" % (
            c.get("api", API), urllib.parse.quote(c["collection"]), urllib.parse.quote(p))
        with get(url, 120) as r:
            body = r.read()                            # NBIA answers an empty body, not [], for no series
        return [x for x in (json.loads(body) if body.strip() else []) if x.get("SeriesInstanceUID") in want]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        found = [x for part in pool.map(patient, sorted({pid[u] for u in uids})) for x in part]
    seen = collections.Counter(x["SeriesInstanceUID"] for x in found)
    bad = [u for u in uids if seen[u] != 1]
    if bad:
        raise SystemExit("%d manifest series not found exactly once on TCIA: %s" % (len(bad), bad))
    return found


def series_list(c, out):
    if "manifest" in c:
        s = manifest_series(c, out)
    else:
        q = urllib.parse.quote(c["collection"])
        with get(c.get("api", API) + "getSeries?Collection=%s&Modality=CT&format=json" % q, 120) as r:
            s = json.load(r)
    wrong = {x.get("LicenseName") for x in s} - {c["licence"]}
    if wrong:
        raise SystemExit("TCIA now lists licence(s) %s for %s CT series; LICENSING.md says %r. "
                         "Stop and check it." % (sorted(wrong), c["collection"], c["licence"]))
    return sorted(s, key=lambda x: (x["PatientID"], x["SeriesInstanceUID"]))


def done(folder, count):
    return (os.path.exists(os.path.join(folder, ".unpacked"))
            and sum(n.endswith(".dcm") for n in os.listdir(folder)) >= count)


def one(out, x, api=API):
    """Fetch and unpack one series; returns (megabytes, seconds) or (0, 0) when already here."""
    folder = os.path.join(out, x["PatientID"], x["SeriesInstanceUID"])
    count = int(x["ImageCount"])
    if os.path.isdir(folder) and done(folder, count):
        return 0.0, 0.0
    os.makedirs(folder, exist_ok=True)
    tmp = folder + ".zip.part"                         # on disk, not in memory: eight series at once
    for attempt in range(1, ATTEMPTS + 1):
        t0 = time.time()
        try:
            with get(api + "getImage?SeriesInstanceUID=" + x["SeriesInstanceUID"]) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            size = os.path.getsize(tmp)
            with zipfile.ZipFile(tmp) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError("damaged member %s" % bad)
                names = [n for n in z.namelist() if not n.endswith("/")]
                for n in names:                        # flat: TCIA's members carry no folders
                    with z.open(n) as src, open(os.path.join(folder, os.path.basename(n)), "wb") as f:
                        shutil.copyfileobj(src, f, 1 << 20)
            os.remove(tmp)
            if sum(n.endswith(".dcm") for n in os.listdir(folder)) < count:
                raise ValueError("%d DICOM files, TCIA lists %d" % (len(names), count))
            with open(os.path.join(folder, ".unpacked"), "w") as f:
                f.write("%d files\n" % len(names))
            return size / 1e6, time.time() - t0
        except Exception as e:                         # a dropped connection or a short zip: once more
            print("%s %s: attempt %d failed (%s)" % (x["PatientID"], x["SeriesInstanceUID"][-12:], attempt, e), flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(5 * attempt)
    raise SystemExit("%s: series %s would not arrive whole after %d attempts"
                     % (x["PatientID"], x["SeriesInstanceUID"], ATTEMPTS))


def extras(out, names):
    for name in names:
        dst = os.path.join(out, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        with get(SITE + name, 300) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        if name.endswith(".zip"):
            with zipfile.ZipFile(dst) as z:
                if z.testzip():
                    raise SystemExit("%s is damaged; delete it and run again" % name)
                z.extractall(os.path.join(out, "annotations"))
        print("%s: %.1f MB" % (name, os.path.getsize(dst) / 1e6), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(COLLECTIONS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N series (a trial)")
    a = ap.parse_args()
    c = COLLECTIONS[a.name]
    out = os.path.join(paths.DATASETS, "public", a.name)
    os.makedirs(out, exist_ok=True)
    extras(out, c["extras"])                           # a manifest source needs its files before listing
    s = series_list(c, out)
    with open(os.path.join(out, "series.json"), "w", encoding="utf-8") as f:
        json.dump(s, f)                                # the whole list, even for a trial: nlst_series.py reads it
    if a.limit:
        s = s[:a.limit]
    total = sum(int(x.get("FileSize", 0)) for x in s) / 1e9
    print("%s: %d CT series, %d patients, %.1f GB, licence %s, into %s"
          % (c["collection"], len(s), len({x["PatientID"] for x in s}), total, c["licence"], out), flush=True)
    t0, got, mb, n = time.time(), 0, 0.0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for size, secs in pool.map(lambda x: one(out, x, c.get("api", API)), s):
            n += 1
            got += size > 0
            mb += size
            if n % 25 == 0 or n == len(s):
                el = time.time() - t0
                print("  %d/%d series (%d fetched now), %.0f MB in %.0f s, %.1f MB/s"
                      % (n, len(s), got, mb, el, mb / max(el, 1e-9)), flush=True)
    print("%s: done, %d series" % (a.name, len(s)), flush=True)


if __name__ == "__main__":
    main()
