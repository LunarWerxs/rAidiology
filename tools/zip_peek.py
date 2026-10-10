"""What a dataset's zip holds, read from its own file list, without downloading it.

A zip keeps its list of files at the end, so a few small range requests read every name and size
in a 2 GB archive in seconds. File names often say what a picture shows: the DIU "Multi-Class Bone
Fracture" set (Mendeley czvd3sdymh) was downloaded whole, 1.8 GB, before its names showed that its
"fractures" were bone spurs and chest inflammation and its "normals" mostly chests. Run this on
every candidate before fetching it.

Prints, per zip: the file count and size per folder, and the commonest name shapes per folder
(digits folded to N, so 235 numbered chest views read as one line). A server that ignores range
requests is reported, not downloaded.

Usage: python tools/zip_peek.py mendeley:<id>[:<version>] | figshare:<article> | <zip url> [--shapes 12]"""
import argparse
import collections
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile

UA = {"User-Agent": "rAidiology-fetch/1.0"}   # honest, as core/fetch.py: Zenodo refuses browser-like ones here
BLOCK = 1 << 16


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched by HTTP range in blocks as zipfile asks.
    Zenodo works too: `https://zenodo.org/api/records/<id>/files/<name>.zip/content` (LUNA16's ten
    subsets' 888 scan names read in 14 s, against 64 GB to download)."""

    def __init__(self, url):
        self.url, self.pos, self.blocks, self.requests = url, 0, {}, 0
        r = urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0")), timeout=60)
        with r:
            if r.status != 206:
                raise SystemExit("%s answers range requests with %d, so its list cannot be read without "
                                 "the whole file" % (url[:80], r.status))
            self.url = r.geturl()              # follow the redirect once, to the storage host
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.blocks:
            lo, hi = i * BLOCK, min(self.size, (i + 1) * BLOCK) - 1
            req = urllib.request.Request(self.url, headers=dict(UA, Range="bytes=%d-%d" % (lo, hi)))
            with urllib.request.urlopen(req, timeout=60) as r:
                self.blocks[i] = r.read()
            self.requests += 1
        return self.blocks[i]

    def readinto(self, b):
        n = min(len(b), max(0, self.size - self.pos))
        out, got = memoryview(b), 0
        while got < n:
            i, off = divmod(self.pos + got, BLOCK)
            chunk = self._block(i)[off:off + n - got]
            out[got:got + len(chunk)] = chunk
            got += len(chunk)
        self.pos += n
        return n


def api(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def zips(source):
    """[(name, url)] of the zips a source names."""
    kind, _, rest = source.partition(":")
    if kind == "mendeley":
        ident, _, version = rest.partition(":")
        files = api("https://data.mendeley.com/public-api/datasets/%s/files?folder_id=root&version=%s"
                    % (ident, version or 1))
        return [(f["filename"], f["content_details"]["download_url"]) for f in files
                if f["filename"].lower().endswith(".zip")]
    if kind == "figshare":
        files = api("https://api.figshare.com/v2/articles/%s" % rest)["files"]
        return [(f["name"], f["download_url"]) for f in files if f["name"].lower().endswith(".zip")]
    return [(os.path.basename(source.split("?")[0]), source)]


def shape(name):
    return re.sub(r"\d+", "N", os.path.basename(name))


def peek(name, url, shapes):
    rf = RangeFile(url)
    infos = [i for i in zipfile.ZipFile(io.BufferedReader(rf, BLOCK)).infolist() if not i.is_dir()]
    print("## %s: %.2f GB, %d files, list read in %d requests (%.1f MB)"
          % (name, rf.size / 1e9, len(infos), rf.requests, rf.requests * BLOCK / 1e6))
    by_dir = collections.defaultdict(list)
    for i in infos:
        by_dir[os.path.dirname(i.filename)].append(i)
    for d in sorted(by_dir):
        items = by_dir[d]
        print("%6d files %8.1f MB  %s/" % (len(items), sum(i.file_size for i in items) / 1e6, d))
        for s, n in collections.Counter(shape(i.filename) for i in items).most_common(shapes):
            print("         %6d  %s" % (n, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--shapes", type=int, default=12, help="name shapes to list per folder")
    a = ap.parse_args()
    found = zips(a.source)
    if not found:
        sys.exit("%s: no zip in this record" % a.source)
    for name, url in found:
        peek(name, url, a.shapes)


if __name__ == "__main__":
    main()
