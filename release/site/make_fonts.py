"""Download Inter and Source Serif 4 (variable, OFL) from the google/fonts repository, subset to Latin, write woff2.

    python release/site/make_fonts.py
Writes site/fonts/*.woff2, the two OFL licence files and site/fonts/SOURCES.json (URL and sha256 of every download)."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "site", "fonts")
RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
FAMILIES = [
    ("Inter", "inter/Inter%5Bopsz%2Cwght%5D.ttf", "inter/OFL.txt", "Inter-Variable.woff2", "OFL-Inter.txt"),
    ("SourceSerif4", "sourceserif4/SourceSerif4%5Bopsz%2Cwght%5D.ttf", "sourceserif4/OFL.txt",
     "SourceSerif4-Variable.woff2", "OFL-SourceSerif4.txt"),
]
LATIN = "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    return data, hashlib.sha256(data).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    sources = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, ttf, lic, woff, licname in FAMILIES:
            data, sha = get(RAW + ttf)
            src = os.path.join(tmp, name + ".ttf")
            open(src, "wb").write(data)
            sources.append({"file": woff, "url": RAW + ttf, "sha256": sha})
            ldata, lsha = get(RAW + lic)
            open(os.path.join(OUT, licname), "wb").write(ldata)
            sources.append({"file": licname, "url": RAW + lic, "sha256": lsha})
            subprocess.run([sys.executable, "-m", "fontTools.subset", src, f"--unicodes={LATIN}", "--flavor=woff2",
                            "--layout-features=*", f"--output-file={os.path.join(OUT, woff)}"], check=True)
            print(woff, os.path.getsize(os.path.join(OUT, woff)), "bytes")
    json.dump(sources, open(os.path.join(OUT, "SOURCES.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
