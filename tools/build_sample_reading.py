"""Build the sample scan's own reading, so the open page can show what a reading looks like.

    python tools/build_sample_reading.py [--out site/sample/sample-ct.raidiology.json.gz]

The visitors in round four said the same thing in the end: seeing the pictures does not explain
them. A reading does, a little: every organ, bone and large vessel the CT model found, outlined on
the person's own slices and named in plain words. Somebody without a reading of their own can now
see one, on the sample, which is a development scan from the TotalSegmentator dataset (CC BY 4.0;
tools/build_sample_scan.py), nobody's private scan.

It is the MCP server's own `read_ct` with `findings`, on the sample ZIP the page serves, read on the
processor so the same inputs give the same bytes, and written with a fixed date. The public export
lets a findings file out only as pinned bytes (PINNED in tools/public_export.py): this prints the
checksum to pin, after somebody has looked at it.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "mcp"), ROOT]
os.environ["RAIDIOLOGY_DEVICE"] = "cpu"
import server  # noqa: E402  arkitect-allow: test-floor-loosened - the server's own reading, after its folder is on the path
from core import findings  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as server is

SAMPLE = os.path.join(ROOT, "site", "sample", "sample-ct.zip")
WRITTEN = "2026-10-02T00:00:00Z"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "site", "sample", "sample-ct.raidiology.json.gz"))
    a = ap.parse_args()
    series = server.tool_find_scans({"path": SAMPLE})["series"]
    if len(series) != 1 or series[0]["modality"] != "CT":
        raise SystemExit("the sample should be one CT series; found %s" % [s["modality"] for s in series])
    work = tempfile.mkdtemp(prefix="raidiology-sample-")
    try:
        got = server.tool_read_ct({"path": SAMPLE, "series_id": series[0]["series_id"], "findings": True, "out": work})
        doc = findings.read(got["findings_file"]["wrote"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    doc["written"] = WRITTEN
    where, raw, packed = findings.write(doc, a.out)
    with open(where, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print("%s: %d structures over %d images, %d bytes (%d unpacked)"
          % (os.path.relpath(where, ROOT), len(doc["findings"]), len(doc["outlines"]["slices"]), packed, raw))
    print("sha256 %s" % digest)


if __name__ == "__main__":
    main()
