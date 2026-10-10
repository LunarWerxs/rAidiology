"""Search the open data archives for candidate datasets, with each one's licence as the archive
states it (machine-readable metadata, not anyone's reading of a web page).

DataCite holds the DOIs of Zenodo, Figshare, Mendeley Data, Dryad, OSF and most university
repositories, licence included, so one query covers them; Hugging Face is asked separately. Each
hit is marked `commercial` when its licence allows commercial use (CC BY, CC0, PDDL, ODC-BY, MIT,
Apache), `no` when it is non-commercial or no-derivatives, `unknown` otherwise. A `commercial` mark
is where checking starts, not where it ends: LICENSING.md takes a row only after the record and the
data's own terms are read (OrthoFrac-XR's archive carried hospital agreements; MFXR's "good" class
turned out to hold other fractures). Before fetching one, `tools/zip_peek.py` reads its zip's file
list in seconds; names often say what the pictures are.

Found 2026-10-03 this way: OrthoFrac-XR, MFXR, LSMA-PQR and the lumbar MRI it annotates.

Usage: python tools/find_datasets.py "<terms>" ["<terms>" ...] [--out found.csv] [--all]
       (--all keeps the non-commercial and unknown ones in the output)"""
import argparse
import concurrent.futures as cf
import csv
import json
import re
import sys
import urllib.parse
import urllib.request

UA = {"User-Agent": "raidiology-find-datasets/1.0", "Accept": "application/json"}
OPEN = re.compile(r"^(cc-by(-[234]\.\d)?|cc0(-1\.0)?|cc-zero|pddl|odc-by|mit|apache-2\.0|cc-by-sa(-[34]\.0)?)$", re.I)
CLOSED = re.compile(r"nc|nd|noncommercial|non-commercial|restricted|proprietary", re.I)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def verdict(lic):
    ids = [x.strip().lower() for x in lic.split(";") if x.strip()]
    if not ids:
        return "unknown"
    if any(CLOSED.search(i) for i in ids):
        return "no"
    return "commercial" if any(OPEN.match(i) for i in ids) else "unknown"


def datacite(term, n=100):
    q = urllib.parse.urlencode({"query": term, "resource-type-id": "dataset", "page[size]": n})
    out = []
    for it in get("https://api.datacite.org/dois?" + q).get("data", []):
        a = it["attributes"]
        if a.get("isActive") is False:
            continue
        lic = ";".join(r.get("rightsIdentifier") or "" for r in a.get("rightsList", []) if r.get("rightsIdentifier"))
        sizes = ";".join(a.get("sizes") or [])
        out.append({"source": "datacite", "id": it["id"], "title": (a.get("titles") or [{}])[0].get("title", ""),
                    "publisher": a.get("publisher") if isinstance(a.get("publisher"), str) else (a.get("publisher") or {}).get("name", ""),
                    "year": a.get("publicationYear"), "licence": lic, "size": sizes,
                    "url": "https://doi.org/" + it["id"], "term": term})
    return out


def huggingface(term, n=50):
    q = urllib.parse.urlencode({"search": term, "limit": n, "full": "true"})
    out = []
    for d in get("https://huggingface.co/api/datasets?" + q):
        lic = ";".join(t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:"))
        out.append({"source": "huggingface", "id": d.get("id", ""), "title": d.get("id", ""), "publisher": "Hugging Face",
                    "year": (d.get("createdAt") or "")[:4], "licence": lic, "size": "",
                    "url": "https://huggingface.co/datasets/" + d.get("id", ""), "term": term})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    rows, seen = [], set()
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(f, t): (f.__name__, t) for t in a.terms for f in (datacite, huggingface)}
        for fu in cf.as_completed(futs):
            try:
                found = fu.result()
            except Exception as e:  # noqa: BLE001  one archive failing must not lose the others
                print("ERR %s %r: %s" % (*futs[fu], e), file=sys.stderr)
                continue
            for r in found:
                key = re.sub(r"\.v?\d+$", "", r["id"].lower())     # one row per dataset, not per version
                if key in seen:
                    continue
                seen.add(key)
                r["commercial"] = verdict(r["licence"])
                rows.append(r)
    keep = [r for r in rows if a.all or r["commercial"] == "commercial"]
    keep.sort(key=lambda r: (r["term"], -int(r["year"] or 0)))
    if a.out:                               # first, so nothing found is lost to a printing error
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id"])
            w.writeheader()
            w.writerows(keep)
    sys.stdout.reconfigure(errors="replace")   # a Windows console cannot print every title's letters
    for r in keep:
        print("%-10s %-4s %-12s %s | %s" % (r["commercial"], r["year"], r["licence"][:12], r["title"][:90], r["url"]))
    print("%d datasets, %d commercially usable by their licence field" % (len(rows), sum(r["commercial"] == "commercial" for r in rows)))


if __name__ == "__main__":
    main()
