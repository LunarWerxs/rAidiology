"""Phase B of the 2026-10-04 download: the NLST CT series that NLST-Sybil and NLST-3D+ label.

Both label sets are labels only. NLST-Sybil's boxes are DICOM SR objects, each naming its patient and the
one CT series it marks; NLST-3D+ is a CSV keyed by series. This lists those series, leaves out the ones
LIDC-annot-NLST501 already brought (`<Datasets>/public/lidc-annot-nlst501/series.json`), finds each
remaining 3D+ series' patient through NBIA's NLST API, and writes a `.tcia` manifest with its series CSV
into `<Datasets>/public/nlst-labelled/` (a junction to the working-copy disk). Then

    python modalities/ct/fetch_tcia.py nlst-labelled

fetches the series. The NLST images are CC BY 4.0 (LICENSING.md, the NLST row).

Usage: python tools/nlst_series.py
"""
import concurrent.futures
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import fetch, paths  # noqa: E402

PUBLIC = os.path.join(paths.DATASETS, "public")
OUT = os.path.join(PUBLIC, "nlst-labelled")
WORKING = "F:/rAidiology/Datasets/public/nlst-labelled"


def sybil():
    import pydicom  # noqa: PLC0415
    out = {}
    for p in glob.glob(os.path.join(PUBLIC, "nlst-sybil", "idc", "*", "*.dcm")):
        d = pydicom.dcmread(p, stop_before_pixels=True)
        for st in d.get("CurrentRequestedProcedureEvidenceSequence") or []:
            for se in st.ReferencedSeriesSequence:
                out[se.SeriesInstanceUID] = str(d.PatientID)
    return out


def three_d_plus():
    rows = []
    for p in glob.glob(os.path.join(PUBLIC, "nlst-3dplus", "*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows += [r["seriesuid"] for r in csv.DictReader(f)]
    return list(dict.fromkeys(rows))


def main():
    with open(os.path.join(PUBLIC, "lidc-annot-nlst501", "series.json"), encoding="utf-8") as f:
        have = {x["SeriesInstanceUID"] for x in json.load(f)}
    sy = sybil()
    td = three_d_plus()
    if not sy or not td:
        raise SystemExit("fetch nlst-sybil and nlst-3dplus first (tools/fetch_public.py fetch nlst-sybil nlst-3dplus)")
    pid = dict(sy)
    ctx = fetch.SetContext("nlst-labelled")
    ask = [u for u in td if u not in pid and u not in have]

    def patient(uid):
        m = fetch.series_meta(uid, ctx)
        return uid, (m[1]["PatientID"] if m else None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for uid, p in pool.map(patient, ask):
            if p:
                pid[uid] = str(p)
    unknown = [u for u in ask if u not in pid]
    want = [u for u in dict.fromkeys(list(sy) + td) if u not in have and u in pid]
    if not os.path.lexists(OUT):
        os.makedirs(WORKING, exist_ok=True)
        fetch.junction(OUT, WORKING)
    with open(os.path.join(OUT, "nlst-labelled.tcia"), "w", encoding="utf-8", newline="\n") as f:
        f.write("downloadServerUrl=https://nlst.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet\n"
                "includeAnnotation=true\nnoOfrRetry=4\ndatabasketId=nlst-labelled.tcia\nmanifestVersion=1.0\n"
                "ListOfSeriesToDownload=SeriesInstanceUID\n" + "\n".join(want) + "\n")
    with open(os.path.join(OUT, "nlst-labelled-series.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series_uid", "pid", "labelled_by"])
        for u in want:
            w.writerow([u, pid[u], "+".join(x for x, s in (("sybil", sy), ("3dplus", set(td))) if u in s)])
    print("NLST-Sybil %d series, NLST-3D+ %d; %d already in lidc-annot-nlst501; %d to fetch, %d patients%s; written to %s"
          % (len(sy), len(td), len((set(sy) | set(td)) & have), len(want), len({pid[u] for u in want}),
             "; %d 3D+ series NBIA does not know: %s" % (len(unknown), unknown[:3]) if unknown else "", OUT))


if __name__ == "__main__":
    main()
