"""Self-test for core/fetch.py: the byte-range downloader and the host limits, against a local HTTP server.

Run: python tools/selftest_fetch.py   (exits non-zero on the first failed check)
"""
import hashlib
import http.server
import math
import os
import random
import re
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import fetch  # noqa: E402

HOST = "127.0.0.1"
SIZE = 5_500_000                        # about 5.5 MB: six 1 MB ranges at CHUNK = 1 MB
DATA = random.Random(20261010).randbytes(SIZE)
MD5 = "md5:" + hashlib.md5(DATA).hexdigest()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        s = self.server
        rng = self.headers.get("Range")
        with s.lock:
            s.seen.append((time.time(), rng))
            s.inflight += 1
            s.peak = max(s.peak, s.inflight)
            refuse = s.refuse > 0
            if refuse:
                s.refuse -= 1
        try:
            time.sleep(s.delay)
            if refuse:
                self.send_response(429)
                self.send_header("Retry-After", "1")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng or "")
            if m:
                a = int(m.group(1))
                b = min(int(m.group(2) or SIZE - 1), SIZE - 1)
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, SIZE))
            else:
                a, b = 0, SIZE - 1
                self.send_response(200)
            body = DATA[a:b + 1]
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with s.lock:
                s.inflight -= 1

    def log_message(self, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__((HOST, 0), Handler)
        self.lock = threading.Lock()
        self.seen = []           # (time, Range header or None) per request, 429s included
        self.inflight = self.peak = 0
        self.refuse = 0          # answer 429 to this many more requests
        self.delay = 0.0         # seconds before each answer, to hold connections open

    @property
    def url(self):
        return "http://%s:%d/set/data.bin" % (HOST, self.server_address[1])

    def ranges(self):
        return [r for _, r in self.seen if r]


def fresh(srv, per_file=None, per_gb=None, gap=None, cap=8):
    """This check's limits for HOST and an empty request log. Host reads its cap and GAP when it is first made,
    so the cached hosts are dropped."""
    for d in (fetch.CAPS, fetch.PER_FILE, fetch.PER_GB, fetch.GAP):
        d.pop(HOST, None)
    fetch.CAPS[HOST] = cap
    if per_file:
        fetch.PER_FILE[HOST] = per_file
    if per_gb:
        fetch.PER_GB[HOST] = per_gb
    if gap:
        fetch.GAP[HOST] = gap
    fetch.Host._all.clear()
    srv.seen.clear()
    srv.peak = 0
    srv.delay = 0.0


def entry(srv):
    return fetch.Entry("data.bin", SIZE, srv.url, MD5)


def ok(msg):
    print("ok: " + msg, flush=True)


def check_one_connection(srv, tmp):
    fresh(srv, per_file=1)
    dst = os.path.join(tmp, "one", "data.bin")
    rec = fetch.download(entry(srv), dst, fetch.SetContext("selftest"), "one")
    assert rec["checked"] == "archive md5", rec
    assert open(dst, "rb").read() == DATA, "one connection: file is not the source"
    assert srv.ranges() == ["bytes=0-%d" % (SIZE - 1)], srv.ranges()
    ok("one connection: byte-identical, one Range request for the whole file (download, _chunked, PER_FILE 1)")


def check_several_connections(srv, tmp):
    fresh(srv, per_file=16, per_gb=0.002)
    srv.delay = 0.3                      # each answer waits, so the three connections overlap
    dst = os.path.join(tmp, "many", "data.bin")
    held = []
    real = fetch._Held
    fetch._Held = lambda: held.append(1) or real()
    try:
        rec = fetch.download(entry(srv), dst, fetch.SetContext("selftest"), "many")
    finally:
        fetch._Held = real
    assert rec["checked"] == "archive md5", rec
    assert not held, "a sparse part went through _Held %d times; its connections should write as they read" % len(held)
    assert open(dst, "rb").read() == DATA, "several connections: file is not the source"
    spans = [tuple(map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", r).groups())) for r in srv.ranges()]
    assert sorted(a for a, _ in spans) == [i * fetch.CHUNK for i in range(6)], spans
    assert all(b - a + 1 == fetch.CHUNK for a, b in spans if b < SIZE - 1), spans
    workers = min(8, 16, math.ceil(SIZE / (0.002 * 1e9)), 6)     # cap, PER_FILE, PER_GB, ranges
    assert workers == 3, workers
    assert srv.peak == workers, "peak connections %d, formula says %d" % (srv.peak, workers)
    ok("several connections: byte-identical, 6 ranges of 1 MB, peak %d connections (formula)" % workers)


def check_not_sparse_in_order(srv, tmp):
    """A part that is not sparse (adopt's sparse=False) takes the in-order path through _Held; a sparse one does not."""
    fresh(srv, per_file=16, per_gb=0.002)
    srv.delay = 0.3
    e = entry(srv)
    dst = os.path.join(tmp, "ordered", "data.bin")
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst))
    fetch._preallocate(part, SIZE, sparse=False)
    assert not fetch._sparse(part), "a part made with sparse=False reads as sparse"
    fetch._save_json(part[:-5] + ".chunks", {"size": SIZE, "chunk": fetch.CHUNK, "url": fetch._ident(e), "done": []})
    held = []
    real = fetch._Held
    fetch._Held = lambda: held.append(1) or real()
    try:
        rec = fetch.download(e, dst, fetch.SetContext("selftest"), "ordered")
    finally:
        fetch._Held = real
    assert rec["checked"] == "archive md5", rec
    assert open(dst, "rb").read() == DATA, "not sparse: file is not the source"
    assert len(held) == 6, "not sparse: %d ranges went through _Held, not 6" % len(held)
    sp = os.path.join(tmp, "ordered", "sparse.bin")
    fetch._preallocate(sp, SIZE)
    assert fetch._sparse(sp) or sys.platform != "win32", "a part made sparse reads as not sparse"
    ok("not sparse: ranges in order through _Held, byte-identical; a sparse part writes as it reads")


def check_resume(srv, tmp):
    fresh(srv, per_file=1)
    e = entry(srv)
    dst = os.path.join(tmp, "resume", "data.bin")
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst))
    have = 3 * fetch.CHUNK
    with open(part, "wb") as f:          # ranges 0-2 right; the rest zeros, which the resume must overwrite
        f.write(DATA[:have] + bytes(SIZE - have))
    fetch._save_json(part[:-5] + ".chunks", {"size": SIZE, "chunk": fetch.CHUNK, "url": fetch._ident(e),
                                             "done": [0, 1, 2]})
    rec = fetch.download(e, dst, fetch.SetContext("selftest"), "resume")
    assert rec["checked"] == "archive md5", rec
    assert srv.ranges() == ["bytes=%d-%d" % (have, SIZE - 1)], srv.ranges()
    assert open(dst, "rb").read() == DATA, "resume: file is not the source"
    ok("resume: only the 3 missing ranges requested (one Range request), byte-identical")


def check_held(srv, tmp):
    h = fetch._Held()
    assert h.b == bytearray(), "_Held must start empty"
    h.write(b"abcdef")
    h.seek(2)
    h.write(b"XY")
    assert bytes(h.b) == b"abXYef", bytes(h.b)
    ok("_Held: starts empty, a write at a earlier point overwrites in place")


def check_gap(srv, tmp):
    fresh(srv, gap=0.2)
    url = srv.url
    starts, lock = [], threading.Lock()

    def one():
        h = fetch.Host.of(url)
        h.take()
        with lock:
            starts.append(time.time())
        h.give()

    threads = [threading.Thread(target=one) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    starts.sort()
    assert len(starts) == 6, starts
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) >= 0.18, gaps
    ok("host GAP: six take()s start at least 0.18 s apart (min %.2f s)" % min(gaps))


def check_429(srv, tmp):
    fresh(srv, per_file=1)
    old = fetch.REST_S
    fetch.REST_S = 2                     # opened() reads REST_S at call time; Retry-After 1 is the smaller
    srv.refuse = 1
    dst = os.path.join(tmp, "rest", "data.bin")
    try:
        rec = fetch.download(entry(srv), dst, fetch.SetContext("selftest"), "rest")
    finally:
        fetch.REST_S = old
    assert rec["checked"] == "archive md5", rec
    assert open(dst, "rb").read() == DATA, "after a 429: file is not the source"
    times = [t for t, _ in srv.seen]
    assert len(times) == 2, times        # the refused request, then its retry
    assert times[1] - times[0] >= 1.9, "retry came %.2f s after the 429" % (times[1] - times[0])
    assert fetch.Host.of(srv.url).until >= times[0] + 1.9, "host did not rest REST_S from the 429"
    ok("429: host rested REST_S (retry %.2f s after it), then the download finished byte-identical"
       % (times[1] - times[0]))


def check_floor(srv, tmp):
    assert fetch.floor("H:/x", passing=True) == 0.05
    assert fetch.floor("H:/x") == 0.15
    assert fetch.floor("\\\\?\\H:\\x", passing=True) == 0.05
    assert fetch.floor("\\\\?\\H:\\x") == 0.15
    assert fetch.floor("G:/x") == fetch.FLOOR and fetch.floor("G:/x", passing=True) == fetch.FLOOR
    ok("floor: H: 0.15 staying / 0.05 passing, also for a \\\\?\\ path; G: the default FLOOR")


def main():
    srv = Server()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    chunk = fetch.CHUNK
    fetch.CHUNK = 1 << 20                # _chunked and download() read CHUNK at call time
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for check in (check_one_connection, check_several_connections, check_not_sparse_in_order, check_resume, check_held,
                          check_gap, check_429, check_floor):
                check(srv, tmp)
    finally:
        fetch.CHUNK = chunk
        for d in (fetch.CAPS, fetch.PER_FILE, fetch.PER_GB, fetch.GAP):
            d.pop(HOST, None)
        fetch.Host._all.clear()
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
    print("all fetch checks ok", flush=True)
