"""Downloading public datasets: the one owner of it.

Every public set rAidiology uses comes from an archive: Zenodo, figshare, Mendeley Data, Hugging Face,
OSF, TCIA or a lab's own page. A reader here asks the archive for the set's licence and its files (name,
size, address, checksum). The downloader then fetches the files, checks them and unpacks them safely.
tools/fetch_public.py holds the list of sets; docs/research/data-download-plan-2026-10-04.md is the plan.

It is polite by construction:

- **It names itself honestly** (`rAidiology-fetch/1.0`). Since 2026-10-05 Zenodo refuses browser-like
  agent strings from this network, and a block is never worked around.
- **Each host has a connection cap** that every file and thread shares (Zenodo: 100, the owner's).
- **A 403 or 429 rests that host for 30 minutes**, longer if its Retry-After says so. The third one
  during one set gives that set up (`SetBlocked`), and the log says what the host answered.
- **Other failures are retried with back-off:** a dropped connection, a time-out, a 5xx.

Zenodo paces each connection, not each file or each client: on 2026-10-09 it gave 0.2 to 0.6 MB/s a
connection, and 8, then 24 connections more beside the refill's six added up (24 on one file 7.2 MB/s, on six files
6.6), every answer a 206, none a 429. So it takes up to 100 connections (the owner, 2026-10-09: "Maybe Zenodo needs
100 connections."): a file under 4 GB gets one, which brings the whole file in one request, and a bigger file one for
each 4 GB it still lacks (`PER_GB`), so the biggest file ends with the rest rather than a day after them. Four are
kept for listing the next set's files (`SHORT`). Its limit is on requests (2,000 an hour for a guest; its answers say
133 a minute), so its requests start at least 1.8 seconds apart (`GAP`). IDC's bucket serves many small images; a
thread keeps its connection to it open from one image to the next (`POOLED`), which saves a TLS handshake
per image.

A big file comes in 64 MB byte ranges, over several connections where its host allows. Each range is
written straight into its place in one preallocated sparse file, so nothing is joined afterwards and no
zeros are written first (`_preallocate`). Ranges are handed out in order, so the file fills from the front. A small
`.chunks` file beside it records the finished ranges, so a run cut short resumes where it stopped.
The archive's own checksum (md5 or sha256) is checked before the file takes its name. Where an archive
publishes none, the sha256 is recorded on arrival.

Unpacking refuses any member whose path would land outside the set's folder."""
import collections
import concurrent.futures
import contextlib
import email.utils
import glob
import hashlib
import html
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field

UA = "rAidiology-fetch/1.0"
CHUNK = 64 << 20               # one byte range
READ = 1 << 20
REST_S = 1800                  # a 403 or 429 rests the host this long
BLOCKS_PER_SET = 3             # the third gives the set up
TRIES = 8                      # transient failures per request before the file fails
ATTEMPTS = 2                   # a file whose checksum fails is fetched once more
PROGRESS_S = 60
RESUME_MIN = 16 << 20          # a smaller file starts over rather than keep a resume note
HELD_EXTRA = 4                 # ranges in memory past one a connection, waiting for a slow range before them (no
#                                more than the file's connections: a Zenodo file on two needs two at most)
IDC_HOSTS = ("idc-open-data.s3.amazonaws.com", "idc-open-data-two.s3.amazonaws.com",
             "idc-open-data-cr.s3.amazonaws.com")
CAPS = {
    "zenodo.org": 100,                               # the owner's cap (2026-10-09; it was 6)
    "services.cancerimagingarchive.net": 8,
    "nlst.cancerimagingarchive.net": 8,
    "www.cancerimagingarchive.net": 2,
    "api.figshare.com": 2,
    "ndownloader.figshare.com": 8,                   # these four hand the bytes to S3 or a CDN
    "data.mendeley.com": 8,
    "huggingface.co": 8,
    "object-arbutus.alliancecan.ca": 8,
    "api.osf.io": 2,
    "files.osf.io": 4,
    "drive.usercontent.google.com": 1,
    "www.dropbox.com": 4,                            # a maker's own copy (faster_copies); it hands bytes to its CDN
    "mmcheng.net": 1,
    "dataverse.harvard.edu": 2,
    "dvn-cloud-iqss.s3.amazonaws.com": 8,            # Harvard Dataverse's store, behind its signed links
    "enhance-pet-1-6k.s3.amazonaws.com": 16,         # an AWS Open Data bucket
    **{h: 64 for h in IDC_HOSTS},                    # AWS's public bucket, built for this
}
PER_FILE = {"zenodo.org": 48}  # connections one file may take of its host's cap (a sparse file's connections write
#                                as they read; only a file that is not sparse holds a range each in memory, _Held)
PER_GB = {"zenodo.org": 1}     # and no more than one for each this many GB the file still lacks: a Zenodo connection
#                                brings about 0.3 MB/s, so 1 GB is about an hour (2026-10-10: at 4 GB the last big
#                                file, unitochest with 40 GB to go, got 10 connections, 3 MB/s at best)
SHORT = {"zenodo.org": 4}      # connections of a host's cap a file body never takes, kept for a set's listing: on
#                                2026-10-06 six Zenodo files, hours each, held all six while two sets' listings waited
GAP = {"zenodo.org": 1.8}      # seconds at least between two requests' starts: under Zenodo's 2,000 an hour
POOLED = set(IDC_HOSTS)        # hosts that never redirect: a thread keeps its connection open
DEFAULT_CAP = 2
SEVEN_ZIP = os.environ.get("RAIDIOLOGY_7Z", r"C:\Program Files\7-Zip\7z.exe")
# IBM's Aspera CLI, portable (github.com/IBM/aspera-cli releases, checked against its sha256), beside the repository
ASCLI = os.environ.get("RAIDIOLOGY_ASCLI", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "_tools", "aspera-cli", "ascli.cmd"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

ZENODO = "https://zenodo.org/api/records/"
FIGSHARE = "https://api.figshare.com/v2/"
MENDELEY = "https://data.mendeley.com/public-api/"
HF = "https://huggingface.co/"
OSF = "https://api.osf.io/v2/"
GH = "https://api.github.com/repos/"
TCIA_SITE = "https://www.cancerimagingarchive.net/api/v1/"
NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
NBIA_NLST = "https://nlst.cancerimagingarchive.net/nbia-api/services/v1/"

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


class FetchError(Exception):
    """A file or a listing cannot be had: a 404, a short file, a checksum that never matches."""


class SetBlocked(FetchError):
    """The set met its third 403 or 429 and is given up."""


class LicenceChanged(FetchError):
    """The archive's licence is no longer the one LICENSING.md recorded."""



def licence_id(text):
    """A licence string as an archive writes it, reduced to one id: cc-by-4.0, cc-by-3.0, cc-by-sa-4.0,
    cc0, or `other:<text>` for anything else. A non-commercial or no-derivatives term never maps to an
    open id, so it can never match an approved row."""
    t = re.sub(r"[\s_]+", " ", str(text or "")).strip().lower()
    if not t:
        return "other:none"
    if re.search(r"\bnc\b|non ?commercial|\bnd\b|no ?deriv", t.replace("-", " ")):
        return "other:" + t
    if re.search(r"cc ?0|cc-zero|zero 1\.0|public domain dedication", t):
        return "cc0"
    sa = bool(re.search(r"\bsa\b|share ?alike", t.replace("-", " ")))
    m = re.search(r"([1-4])\.0", t)
    if m and re.search(r"\bcc\b|cc-by|creative commons|attribution", t):
        return "cc-by-%s%s.0" % ("sa-" if sa else "", m.group(1))
    return "other:" + t


def check_licence(name, seen, allowed):
    """Stop when an archive's licence (raw text) is not one the set's LICENSING.md row allows."""
    got = licence_id(seen)
    if got not in allowed:
        raise LicenceChanged("%s: the archive now says licence %r (%s); LICENSING.md allows %s. Stopped; "
                             "check the row before fetching." % (name, seen, got, sorted(allowed)))
    return got



def cap(host):
    """A host's connection cap. A Globus collection's HTTPS server (`*.data.globus.org`) serves a file staged for us
    in byte ranges: Harvard's gives each connection 2.24 MB/s, and together they add up (measured 2026-10-07 into
    memory: 1 connection 2.2 MB/s, 8 18 MB/s, 32 70 MB/s, 64 about 100 MB/s, the home line's limit). Onto the spare
    disk, with the disk to itself and the ranges written in order, 32 gave 70-72 MB/s (2026-10-07)."""
    return CAPS.get(host) or (32 if host.endswith(".data.globus.org") else DEFAULT_CAP)


def _auth(entry):
    """The Authorization header a file's host wants, from the entry's `auth` (a callable that returns a fresh
    header; a Globus collection's HTTPS server takes a bearer token), or none."""
    auth = entry.meta.get("auth") if isinstance(getattr(entry, "meta", None), dict) else None
    return {"Authorization": auth()} if auth else {}


class Host:
    """One host's connection cap and rest period, shared by every thread."""
    _all = {}
    _lock = threading.Lock()

    def __init__(self, name):
        self.name = name
        self.slots = threading.BoundedSemaphore(cap(name))
        keep = SHORT.get(name, 0)
        self.bulk = threading.BoundedSemaphore(cap(name) - keep) if 0 < keep < cap(name) else None
        self.until = 0.0
        self.gap, self.next_at = GAP.get(name, 0), 0.0
        self.lock = threading.Lock()

    @classmethod
    def of(cls, url):
        name = (urllib.parse.urlsplit(url).hostname or "").lower()
        with cls._lock:
            if name not in cls._all:
                cls._all[name] = cls(name)
            return cls._all[name]

    def rest(self, seconds, why):
        """Rest the host; True when this starts a rest, False when one was already running (a request
        that was in flight when another thread's refusal came back)."""
        with self.lock:
            now = time.time()
            new = now >= self.until
            if now + seconds > self.until:
                self.until = now + seconds
                log("%s: resting %d min (%s)" % (self.name, seconds // 60, why))
            return new

    def wait(self):
        while True:
            left = self.until - time.time()
            if left <= 0:
                return
            time.sleep(min(left, 30))

    def take(self, bulk=False):
        """A connection slot, taken only once the host is not resting (a rest can begin while a thread
        waits for a slot). A file body (bulk) first takes one of the slots SHORT leaves to it. Where the host has a
        GAP, the request then waits its turn to start."""
        if bulk and self.bulk:
            self.bulk.acquire()
        while True:
            self.wait()
            self.slots.acquire()
            if time.time() >= self.until:
                break
            self.slots.release()
        if self.gap:
            with self.lock:
                at = max(time.time(), self.next_at)
                self.next_at = at + self.gap
            time.sleep(max(0.0, at - time.time()))

    def give(self, bulk=False):
        self.slots.release()
        if bulk and self.bulk:
            self.bulk.release()


class SetContext:
    """What one set's requests share: its name and its count of blocks, the rests in a row with no answer
    between them. A host that answers after a rest is rate-limiting, not refusing, so an answer clears it."""

    def __init__(self, name):
        self.name = name
        self.blocks = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def answered(self):
        if self.blocks:
            with self.lock:
                self.blocks = 0

    def blocked(self, host, code, snippet):
        with self.lock:
            self.blocks += 1
            n = self.blocks
        log("%s: %s answered %d (%d of %d for this set): %s" % (self.name, host, code, n, BLOCKS_PER_SET, snippet))
        if n >= BLOCKS_PER_SET:
            self.stop.set()
            raise SetBlocked("%s: %s refused %d times (last: %d %s). Given up; not worked around. If this is "
                             "Zenodo refusing an honest agent string, ask Zenodo support (its 403 page "
                             "invites it)." % (self.name, host, n, code, snippet))


def _retry_after(value):
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        try:
            return max(0, int(email.utils.parsedate_to_datetime(value).timestamp() - time.time()))
        except (TypeError, ValueError):
            return 0


_CONNS = threading.local()     # per thread: {host: an open HTTPSConnection}, for POOLED hosts


def _pooled(url, headers, timeout, method):
    """A request over this thread's open connection to a POOLED host; a connection the host closed while
    it sat idle is opened again once. Any status but 200 or 206 raises HTTPError, as urlopen's would."""
    u = urllib.parse.urlsplit(url)
    pool = _CONNS.__dict__.setdefault("pool", {})
    path = (u.path or "/") + ("?" + u.query if u.query else "")
    c = pool.pop(u.hostname, None)
    while True:
        reused = c is not None
        if c is None:
            c = http.client.HTTPSConnection(u.hostname, timeout=timeout)
        try:
            c.request(method, path, headers=headers)
            r = c.getresponse()
            break
        except (http.client.HTTPException, OSError):
            c.close()
            c = None
            if not reused:
                raise
    r.pooled_conn = c
    if r.status not in (200, 206):
        body = r.read()
        c.close()
        raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, io.BytesIO(body))
    return r


def _close(r):
    """Done with a response: a POOLED connection whose answer was read to its end goes back to the pool."""
    c = getattr(r, "pooled_conn", None)
    if c is not None and r.isclosed() and not r.will_close:
        _CONNS.__dict__.setdefault("pool", {})[c.host] = c
        return
    r.close()
    if c is not None:
        c.close()


@contextlib.contextmanager
def opened(url, ctx, headers=None, timeout=120, method="GET", data=None, bulk=False):
    """Open `url` holding one of its host's connection slots for as long as the caller reads (`data`, a POST
    body, never to a POOLED host). `bulk`: a file body, which leaves the host's SHORT slots to listings.

    Retries what is worth retrying (a dropped connection, a time-out, a 5xx) with back-off. A 403 or 429
    rests the host, and one that starts a rest counts against the set (`SetContext.blocked`). A 404, 410 or
    416 is final."""
    host = Host.of(url)
    delay = 10
    last = None
    for _ in range(TRIES):
        if ctx is not None and ctx.stop.is_set():
            raise FetchError("%s: stopped" % ctx.name)
        host.take(bulk)
        try:
            hdrs = {"User-Agent": UA, **(headers or {})}
            try:
                if host.name in POOLED:
                    r = _pooled(url, hdrs, timeout, method)
                else:
                    r = urllib.request.urlopen(urllib.request.Request(url, data=data, headers=hdrs, method=method),
                                               timeout=timeout)
            except urllib.error.HTTPError as e:
                snippet = re.sub(r"\s+", " ", e.read(400).decode("utf-8", "replace"))[:200]
                if e.code in (403, 429):
                    seconds = max(REST_S, _retry_after(e.headers.get("Retry-After")))
                    where = getattr(e, "url", None) or url
                    answered = Host.of(where)              # after a redirect, the host that refused
                    why = "%d on %s" % (e.code, where[:120])
                    new = host.rest(seconds, why)
                    if answered is not host:
                        answered.rest(seconds, why)
                    if ctx is not None and new:
                        ctx.blocked(answered.name, e.code, snippet)
                    last = "%d %s" % (e.code, snippet)
                    continue
                if e.code in (404, 410, 416) or (400 <= e.code < 500 and e.code not in (408, 409, 425)):
                    raise FetchError("%s: %d %s" % (url[:160], e.code, snippet)) from None
                last = "%d %s" % (e.code, snippet)
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
                last = "%s: %s" % (type(e).__name__, e)
            else:
                if ctx is not None:
                    ctx.answered()
                try:
                    yield r
                finally:
                    _close(r)
                return
        finally:
            host.give(bulk)
        time.sleep(delay)
        delay = min(delay * 2, 600)
    raise FetchError("%s: no answer after %d tries (last: %s)" % (url[:160], TRIES, last))


def get_bytes(url, ctx, headers=None, timeout=120):
    for attempt in range(1, TRIES + 1):
        with opened(url, ctx, headers, timeout) as r:
            try:
                return r.read()
            except (OSError, http.client.HTTPException) as e:  # a body cut short: open it again
                log("%s: read cut short (%s), again" % (url[:100], e))
        time.sleep(min(10 * attempt, 120))
    raise FetchError("%s: the body never arrived whole" % url[:160])


def get_json(url, ctx, timeout=120):
    return json.loads(get_bytes(url, ctx, {"Accept": "application/json"}, timeout))


def range_total(url, ctx):
    """The full size of `url` from a one-byte range request, with the response's file name; None when
    the server ignores ranges."""
    with opened(url, ctx, {"Range": "bytes=0-0"}) as r:
        cr = r.headers.get("Content-Range") or ""
        disp = r.headers.get("Content-Disposition") or ""
        kind = r.headers.get("Content-Type") or ""
        r.read(1)
    m = re.match(r"bytes 0-0/(\d+)", cr)
    fname = None
    m2 = re.search(r"filename\*=UTF-8''([^;]+)", disp) or re.search(r'filename="?([^";]+)', disp)
    if m2:
        fname = urllib.parse.unquote(m2.group(1)).strip()
    return (int(m.group(1)) if m else None), fname, kind



@dataclass
class Entry:
    name: str                       # path inside the set's archive folder, "/"-separated
    size: int | None                # bytes, None when the archive does not say
    url: str
    checksum: str | None = None     # "md5:<hex>" or "sha256:<hex>", the archive's own
    kind: str = "file"              # "file"; "series", a TCIA DICOM series zip; "instance", one of its images from
                                    # IDC; "aspera", a TCIA package folder received over IBM Aspera
    meta: dict = field(default_factory=dict)
    ident: str | None = None        # the archive's own link, when a faster copy serves the bytes (`faster_copies`)


@dataclass
class Listing:
    licence: str                    # the licence as the archive writes it
    source: str                     # where it was read
    files: list
    notes: list = field(default_factory=list)
    skipped: list = field(default_factory=list)   # (what, why)
    est_bytes: int = 0              # for files of unknown size: the archive's own estimate

    def size(self):
        return sum(f.size or 0 for f in self.files) + self.est_bytes


def _merge(name, parts):
    """One listing from several records (several Zenodo records, a figshare collection): their licences
    must agree."""
    lic = {p.licence for p in parts}
    if len({licence_id(x) for x in lic}) > 1:
        raise LicenceChanged("%s: its records carry different licences: %s" % (name, sorted(lic)))
    out = Listing(parts[0].licence, "; ".join(p.source for p in parts), [])
    for p in parts:
        out.files += p.files
        out.notes += p.notes
        out.skipped += p.skipped
        out.est_bytes += p.est_bytes
    return out


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", s).strip(" .")[:120] or "item"



def zenodo(name, ids, ctx, exclude=()):
    parts = []
    for rid in ids:
        r = get_json(ZENODO + str(rid), ctx)
        md = r.get("metadata", {})
        lic = (md.get("license") or {}).get("id") or ""
        files = r.get("files") or []
        if len(files) >= 100:            # a long list may be cut; the files endpoint gives all of it
            files = get_json(ZENODO + "%s/files" % r["id"], ctx).get("entries") or files
        prefix = "%s/" % r["id"] if len(ids) > 1 else ""
        part = Listing(lic, ZENODO + str(r["id"]), [])
        if md.get("access_right", "open") != "open":
            part.skipped.append(("record %s" % r["id"], "access is %s" % md.get("access_right")))
            parts.append(part)
            continue
        for f in files:
            key = f.get("key")
            links = f.get("links") or {}
            url = links.get("content") or links.get("self")
            if url and not url.endswith("/content"):
                url = ZENODO + "%s/files/%s/content" % (r["id"], urllib.parse.quote(key))
            if any(re.search(p, key, re.I) for p in exclude):
                part.skipped.append((key, "left out by the set's row"))
                continue
            part.files.append(Entry(prefix + key, int(f["size"]), url, f.get("checksum")))
        part.notes.append("record %s: %s, version %s" % (r["id"], r.get("doi"), md.get("version")))
        parts.append(part)
    return _merge(name, parts)


def faster_copies(lst, copies):
    """Points each file named in `copies` ({name in the listing: url}) at another copy that its makers serve
    faster than the archive (TotalSegmentator's own Dropbox copy beside its Zenodo record). The archive's size and
    checksum still decide whether what comes is that file, so a file the archive gives no checksum keeps the
    archive's link. The archive's link stays the file's identity, so a .part begun from either resumes, and is
    where the file comes from if the copy fails."""
    for e in lst.files:
        url = (copies or {}).get(e.name)
        if url and e.checksum:
            e.ident, e.meta["archive_url"], e.url = e.url, e.url, url
            lst.notes.append("%s: fetched from %s, checked against the archive's %s" % (
                e.name, url.split("?")[0], e.checksum.split(":")[0]))
        elif url:
            lst.notes.append("%s: the archive gives no checksum, so its own link is used" % e.name)


def git_annex(name, ctx, record, mirror):
    """A git-annex dataset whose Zenodo record is a snapshot of the repository, holding pointer files
    only (Spine Generic): the snapshot, plus every annexed file it points at, from the dataset's own
    public HTTPS mirror (the one `git annex get` uses). Each annex key carries the file's size and sha256."""
    out = zenodo(name, [record], ctx)
    for e in list(out.files):
        if not e.name.lower().endswith(".zip"):
            continue
        z = zipfile.ZipFile(io.BytesIO(get_bytes(e.url, ctx)))
        stem = e.name.rsplit("/", 1)[0] + "/" if "/" in e.name else ""
        found = other = 0
        for info in z.infolist():
            if info.is_dir() or info.file_size > 512:
                continue
            body = z.read(info).strip()
            if not (body.startswith(b"/annex/objects/") or b".git/annex/objects/" in body):
                continue
            m = re.search(rb"(SHA256E?-s(\d+)--([0-9a-f]{64})[^/\s]*)$", body)
            if not m:
                other += 1
                continue
            out.files.append(Entry(stem + info.filename, int(m.group(2)), mirror.rstrip("/") + "/" + m.group(1).decode(),
                                   "sha256:" + m.group(3).decode()))
            found += 1
        out.notes.append("%s: %d annexed files from %s" % (e.name, found, mirror))
        if other:
            out.skipped.append(("%d annexed files" % other, "keyed by a hash other than SHA256, so not checkable"))
    return out


IDC_BUCKET = "https://idc-open-data.s3.amazonaws.com/"


def idc_objects(name, ctx, s5cmd_text):
    """The objects an IDC s5cmd manifest names (`cp s3://idc-open-data/<series>/* .`), from IDC's public
    AWS bucket over plain HTTPS: one listing per series folder, the object's md5 from its ETag."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415
    prefixes = list(dict.fromkeys(re.findall(r"s3://idc-open-data/([0-9a-f-]{36})/", s5cmd_text)))
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"

    def one(prefix):
        root = ET.fromstring(get_bytes(IDC_BUCKET + "?list-type=2&prefix=%s/" % prefix, ctx))
        if root.findtext(ns + "IsTruncated") == "true":
            raise FetchError("%s: IDC folder %s lists more than one page" % (name, prefix))
        got = []
        for c in root.iter(ns + "Contents"):
            key, etag = c.findtext(ns + "Key"), (c.findtext(ns + "ETag") or "").strip('"')
            got.append(Entry("idc/" + key, int(c.findtext(ns + "Size")), IDC_BUCKET + key,
                             "md5:" + etag if re.fullmatch(r"[0-9a-f]{32}", etag) else None))
        return got

    with concurrent.futures.ThreadPoolExecutor(max_workers=CAPS["idc-open-data.s3.amazonaws.com"]) as pool:
        found = [e for part in pool.map(one, prefixes) for e in part]
    empty = len(prefixes) - len({e.name.split("/")[1] for e in found})
    if empty:
        raise FetchError("%s: %d of %d IDC folders are empty" % (name, empty, len(prefixes)))
    return found


_IDC = {}
_IDC_LOCK = threading.RLock()


def _idc_client():
    """IDC's own index of every series it serves (the idc-index package). Loaded once."""
    with _IDC_LOCK:
        if "client" not in _IDC:
            from idc_index import IDCClient  # noqa: PLC0415
            _IDC["client"] = IDCClient()
        return _IDC["client"]


def _idc_index(uids):
    """IDC's index cut to bucket folder, image count and licence, by SeriesInstanceUID, for the series `uids` only,
    read straight from the idc-index-data parquet file IDCClient loads. IDCClient keeps the whole index, a million
    series in 31 columns, about 2 GB for as long as the process runs (two fetch runs at 2.05 GB each, 2026-10-06);
    crdc_series_uuid is derived from series_aws_url as IDCClient derives it."""
    import idc_index_data  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    ix = pq.read_table(idc_index_data.IDC_INDEX_PARQUET_FILEPATH,
                       columns=["SeriesInstanceUID", "series_aws_url", "aws_bucket", "instanceCount", "license_short_name"],
                       filters=[("SeriesInstanceUID", "in", sorted(uids))]).to_pandas()
    pa.default_memory_pool().release_unused()
    ix["crdc_series_uuid"] = ix.pop("series_aws_url").str.split("/").str[3]
    return ix.set_index("SeriesInstanceUID")


def idc_analysis(name, ctx, result, allowed, images=False):
    """An IDC analysis result (its SEG, SR and other series) from IDC's public AWS bucket, each object checked by
    its md5 (the ETag); with `images`, also the image series its segmentations are drawn on. Every series'
    licence in IDC's index must be one `allowed`, and every series must hold the image count the index gives."""
    c = _idc_client()
    ix = c.index
    mine = ix.analysis_result_id == result
    if not mine.any():
        raise FetchError("%s: IDC lists no series for analysis result %r" % (name, result))
    out = Listing("", "IDC index %s, analysis result %s" % (c.get_idc_version(), result), [])
    if images:
        with _IDC_LOCK:
            if "seg" not in _IDC:
                c.fetch_index("seg_index")
                _IDC["seg"] = c.seg_index
        seg = _IDC["seg"]
        ref = set(seg[seg.SeriesInstanceUID.isin(ix[mine].SeriesInstanceUID)].segmented_SeriesInstanceUID)
        rows = ix[mine | ix.SeriesInstanceUID.isin(ref)]
        gone = len(ref - set(rows.SeriesInstanceUID))
        out.notes.append("%d image series the segmentations are drawn on%s" % (len(ref), ", %d not in IDC's index" % gone if gone else ""))
    else:
        rows = ix[mine]
    licences = set()
    for r in rows.itertuples():
        check_licence("%s series %s" % (name, r.SeriesInstanceUID), r.license_short_name, allowed)
        licences.add(r.license_short_name)
    out.licence = " + ".join(sorted(licences))

    def one(r):
        objs = _s3_list(r.aws_bucket, r.crdc_series_uuid + "/", ctx)
        if len(objs) != int(r.instanceCount):
            raise FetchError("%s: IDC series %s holds %d objects, its index says %d"
                             % (name, r.SeriesInstanceUID, len(objs), int(r.instanceCount)))
        meta = {"patient": r.PatientID, "uid": r.SeriesInstanceUID}
        return [Entry("idc/" + key, size, "https://%s.s3.amazonaws.com/%s" % (r.aws_bucket, key),
                      "md5:" + md5 if md5 else None, "instance", dict(meta)) for key, size, md5 in objs]

    with concurrent.futures.ThreadPoolExecutor(max_workers=CAPS["idc-open-data.s3.amazonaws.com"]) as pool:
        for part in pool.map(one, list(rows.itertuples())):
            out.files += part
    out.notes.append("%d series (%s), %d objects from IDC's AWS bucket"
                     % (len(rows), ", ".join("%d %s" % (n, m) for m, n in rows.Modality.value_counts().items()), len(out.files)))
    return out


def _s3_list(bucket, prefix, ctx):
    """Every object under `prefix` in a public S3 bucket over plain HTTPS: (key, size, md5 or None). A folder
    marker (an empty key ending in "/", as some IDC series folders hold) is not an object and is left out."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    got, token = [], None
    while True:
        url = "https://%s.s3.amazonaws.com/?list-type=2&prefix=%s" % (bucket, urllib.parse.quote(prefix))
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(get_bytes(url, ctx))
        for c in root.iter(ns + "Contents"):
            etag = (c.findtext(ns + "ETag") or "").strip('"')
            if c.findtext(ns + "Key").endswith("/"):
                continue
            got.append((c.findtext(ns + "Key"), int(c.findtext(ns + "Size")), etag if re.fullmatch(r"[0-9a-f]{32}", etag) else None))
        if root.findtext(ns + "IsTruncated") != "true":
            return got
        token = root.findtext(ns + "NextContinuationToken")


def public_bucket(name, ctx, bucket, page, pattern, prefixes, subjects=None):
    """A public AWS Open Data bucket: the licence sentence read on its registry entry (`page`, `pattern`), then
    every object under `prefixes` over plain HTTPS. `subjects` keeps only the subjects from some sites, for a
    bucket whose licence differs by site: {"xlsx": its key table in the bucket, "id" and "source": that table's
    columns, "keep": the sources to keep, "key": a regex whose group 1 is an object's subject id}; an object
    the regex does not match (a table, a label map) is kept."""
    text = get_bytes(page, ctx).decode("utf-8", "replace")
    m = re.search(pattern, re.sub(r"\s+", " ", text), re.I)
    if not m:
        raise LicenceChanged("%s: the licence sentence is no longer on %s" % (name, page))
    out = Listing(m.group(0), page, [])
    base = "https://%s.s3.amazonaws.com/" % bucket
    allowed = None
    if subjects:
        rows = _xlsx_rows(get_bytes(base + subjects["xlsx"], ctx))
        allowed = {str(r.get(subjects["id"])) for r in rows if r.get(subjects["source"]) in subjects["keep"]}
        out.notes.append("%d of %d subjects kept, those from %s" % (len(allowed), len(rows), "; ".join(subjects["keep"])))
    for prefix in prefixes:
        for key, size, md5 in _s3_list(bucket, prefix, ctx):
            hit = re.search(subjects["key"], key) if subjects else None
            if key.endswith("/") or (hit and hit.group(1) not in allowed):
                continue
            out.files.append(Entry(key, size, base + urllib.parse.quote(key), "md5:" + md5 if md5 else None))
    if not out.files:
        raise FetchError("%s: nothing under %s in s3://%s" % (name, prefixes, bucket))
    return out


def idc_series(name, ctx, rows, allowed):
    """The series of a TCIA manifest that IDC serves the same: IDC holds the same image count NBIA lists,
    under a licence the row allows. Each comes as its DICOM objects from IDC's public AWS bucket, much faster
    than NBIA's zips, each checked by its md5 (the object's ETag). {uid: [Entry]}; the rest stay with NBIA."""
    ix = _idc_index(rows)
    known = [u for u in rows if u in ix.index]
    picked = {}
    for u in known:
        r = ix.loc[u]
        if isinstance(r, type(ix)):                    # a series IDC lists twice: leave it to NBIA
            continue
        if int(r.instanceCount) != int(rows[u].get("ImageCount") or -1) or licence_id(r.license_short_name) not in allowed:
            continue
        picked[u] = (r.aws_bucket, r.crdc_series_uuid)
    host = "idc-open-data.s3.amazonaws.com"

    def one(u):
        bucket, folder = picked[u]
        objs = _s3_list(bucket, folder + "/", ctx)
        if len(objs) != int(rows[u]["ImageCount"]) or any(md5 is None for _, _, md5 in objs):
            return u, None
        meta = {"patient": rows[u]["PatientID"], "uid": u}
        return u, [Entry("idc/" + key, size, "https://%s.s3.amazonaws.com/%s" % (bucket, key), "md5:" + md5, "instance", dict(meta))
                   for key, size, md5 in objs]

    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=CAPS[host]) as pool:
        for u, entries in pool.map(one, list(picked)):
            if entries:
                out[u] = entries
    return out


def figshare(name, ctx, articles=(), collection=None, exclude=()):
    ids = list(articles)
    if collection:
        ids += [a["id"] for a in get_json(FIGSHARE + "collections/%s/articles?page_size=1000" % collection, ctx)]
    parts = []
    for a in ids:
        d = get_json(FIGSHARE + "articles/%s" % a, ctx)
        prefix = "%s_%s/" % (a, _safe_name(d.get("title", ""))[:60]) if len(ids) > 1 else ""
        lic = d.get("license") or {}
        stated = lic.get("name") or ""
        if stated and not re.search(r"\d\.\d", stated) and re.search(r"creativecommons\.org/licenses/[a-z-]+/\d\.\d", lic.get("url") or ""):
            stated += " (%s)" % lic["url"]          # Springer Nature's figshare names "CC BY" and gives the version in its link
        part = Listing(stated, FIGSHARE + "articles/%s" % a, [])
        # figshare keeps no folders: PediMS's 365 files carry 14 names, listed patient by patient, so the n-th
        # file of a repeated name goes in folder n (the archive's own listing order) instead of overwriting
        # its namesakes, and one patient's scans and masks share a folder
        seen = collections.Counter(f["name"] for f in d.get("files") or [])
        nth = collections.Counter()
        for f in d.get("files") or []:
            if seen[f["name"]] > 1:
                nth[f["name"]] += 1
                f = dict(f, name="%02d/%s" % (nth[f["name"]], f["name"]))
            if f.get("is_link_only"):
                part.skipped.append((f.get("name"), "a link to elsewhere, not a file: %s" % f.get("download_url")))
                continue
            if any(re.search(p, f["name"], re.I) for p in exclude):
                part.skipped.append((f["name"], "left out by the set's row"))
                continue
            md5 = f.get("computed_md5") or f.get("supplied_md5")
            part.files.append(Entry(prefix + f["name"], int(f["size"]), f["download_url"], "md5:" + md5 if md5 else None))
        parts.append(part)
    return _merge(name, parts)


def mendeley(name, ctx, dataset, version=None):
    """A Mendeley Data set. Its API lists only the files at the root (a /folders call is a 404), so a set
    with subfolders comes as Mendeley's own download-all zip, whose every member's CRC is checked."""
    if version is None:
        version = max(v["version"] for v in get_json(MENDELEY + "datasets/%s" % dataset, ctx).get("versions") or [{"version": 1}])
    d = get_json(MENDELEY + "datasets/%s?version=%s" % (dataset, version), ctx)
    lic = (d.get("data_licence") or {}).get("short_name") or (d.get("data_licence") or {}).get("full_name") or ""
    files = get_json(MENDELEY + "datasets/%s/files?folder_id=root&version=%s" % (dataset, version), ctx)
    if not isinstance(files, list):
        files = []
    out = Listing(lic, MENDELEY + "datasets/%s?version=%s" % (dataset, version), [])
    root = sum(int(f.get("size") or 0) for f in files)
    zurl = MENDELEY + "zip/%s/download/%s" % (dataset, version)
    total, fname, kind = range_total(zurl, ctx)
    if files and (total is None or total <= root * 1.02 + 2_000_000):
        for f in files:
            cd = f.get("content_details") or {}
            sha = cd.get("sha256_hash")
            out.files.append(Entry(f["filename"], int(f["size"]), cd["download_url"], "sha256:" + sha if sha else None))
        out.notes.append("v%s: %d root files" % (version, len(files)))
    elif total:
        out.files.append(Entry("%s-%s.zip" % (dataset, version), total, zurl, None, meta={"whole_zip": True}))
        out.notes.append("v%s: subfolders, so Mendeley's download-all zip (%.2f GB); its members' CRCs are "
                         "checked, and its sha256 is recorded" % (version, total / 1e9))
    else:
        raise FetchError("%s: Mendeley lists no root files and its zip gave no size (%s)" % (name, kind))
    return out


class Signed:
    """A file whose link is handed out per request and soon expires. `url` is a fresh link once the last is
    `ttl` seconds old; `ident` stays the same across links, so a part file and its .chunks note resume."""

    def __init__(self, name, size, ident, sign, checksum=None, meta=None, ttl=2400):
        self.name, self.size, self.ident, self.checksum = name, size, ident, checksum
        self.kind, self.meta = "file", meta or {}
        self._sign, self._ttl, self._url, self._at = sign, ttl, None, 0.0
        self._lock = threading.Lock()

    @property
    def url(self):
        with self._lock:
            if self._url is None or time.time() - self._at > self._ttl:
                self._url, self._at = self._sign(), time.time()
            return self._url


def _ident(entry):
    return getattr(entry, "ident", None) or entry.url


def dataverse(name, ctx, server, pid, guestbook, only=None):
    """A Dataverse dataset's latest version. A dataset with a guestbook gives a file only to a POST of the
    guestbook's answer, as a signed link that lasts about a minute and leads to the store's own one-hour link;
    each file is a Signed entry that asks again every 40 minutes (each asking is one guestbook record)."""
    d = get_json("%s/api/datasets/:persistentId/?persistentId=%s" % (server, pid), ctx)["data"]
    v = d["latestVersion"]
    lic = (v.get("license") or {}).get("name") or v.get("termsOfUse") or ""
    out = Listing(lic, "%s/api/datasets/:persistentId/?persistentId=%s (version %s.%s)" % (
        server, pid, v.get("versionNumber"), v.get("versionMinorNumber")), [])

    def signer(fid):
        def sign():
            body = json.dumps({"guestbookResponse": guestbook} if d.get("guestbookId") else {}).encode()
            with opened("%s/api/access/datafile/%d" % (server, fid), ctx, {"Content-Type": "application/json"},
                        method="POST", data=body) as r:
                signed = json.loads(r.read())["data"]["signedUrl"]
            with opened(signed, ctx, {"Range": "bytes=0-0"}) as r:   # its redirect is the store's link
                r.read(1)
                return r.geturl()
        return sign

    for f in v["files"]:
        df = f["dataFile"]
        fn = ((f.get("directoryLabel") or "").strip("/") + "/" if f.get("directoryLabel") else "") + df["filename"]
        if f.get("restricted"):
            out.skipped.append((fn, "restricted"))
            continue
        if only and not re.search(only, fn):
            out.skipped.append((fn, "not taken"))
            continue
        if (df.get("storageIdentifier") or "").startswith("globus"):
            out.skipped.append((fn, "held in the archive's Globus store, which only a Globus transfer reaches"))
            continue
        ck = df.get("checksum") or {}
        kind = (ck.get("type") or "").lower().replace("-", "")
        value = ck.get("value") or ""
        cs = "%s:%s" % (kind, value.lower()) if kind in ("md5", "sha256", "sha1") and re.fullmatch(r"[0-9a-fA-F]+", value) else None
        out.files.append(Signed(fn, int(df["filesize"]), "%s/api/access/datafile/%d" % (server, df["id"]),
                                signer(df["id"]), cs))
    out.notes.append("guestbook %s answered per download" % d.get("guestbookId") if d.get("guestbookId") else "no guestbook")
    return out


def huggingface(name, ctx, repo, dirs=("data",), skip=("original",)):
    info = get_json(HF + "api/datasets/%s" % repo, ctx)
    if info.get("gated"):
        raise FetchError("%s: the Hugging Face repo is gated now; no forms" % name)
    lic = (info.get("cardData") or {}).get("license") or ""
    out = Listing(lic, HF + "api/datasets/%s (card)" % repo, [])
    listing = [x for x in get_json(HF + "api/datasets/%s/tree/main" % repo, ctx) if x.get("type") == "file"]
    for d in dirs:
        listing += [x for x in get_json(HF + "api/datasets/%s/tree/main/%s?recursive=true" % (repo, d), ctx)
                    if x.get("type") == "file"]
    for x in listing:
        p = x["path"]
        if any(p == s or p.startswith(s + "/") for s in skip):
            continue
        lfs = x.get("lfs") or {}
        out.files.append(Entry(p, int(x["size"]), HF + "datasets/%s/resolve/main/%s" % (repo, urllib.parse.quote(p)),
                               "sha256:" + lfs["oid"] if lfs.get("oid") else None))
    out.skipped += [(s + "/", "left out by the set's row") for s in skip]
    return out


def _osf_pages(url, ctx):
    while url:
        page = get_json(url, ctx)
        yield from page.get("data") or []
        url = (page.get("links") or {}).get("next")


def osf(name, ctx, node):
    def licence_of(n):
        rel = ((n.get("relationships") or {}).get("license") or {}).get("data")
        return get_json(OSF + "licenses/%s/" % rel["id"], ctx)["data"]["attributes"]["name"] if rel else None

    top = get_json(OSF + "nodes/%s/" % node, ctx)["data"]
    lic = licence_of(top) or ""
    out = Listing(lic, OSF + "nodes/%s/ (license)" % node, [])

    def walk(url, prefix):
        for x in _osf_pages(url, ctx):
            a = x["attributes"]
            if a["kind"] == "folder":
                walk(x["relationships"]["files"]["links"]["related"]["href"], prefix + _safe_name(a["name"]) + "/")
            else:
                sha = ((a.get("extra") or {}).get("hashes") or {}).get("sha256")
                out.files.append(Entry(prefix + a["name"], int(a["size"]), x["links"]["download"], "sha256:" + sha if sha else None))

    def node_files(nid, prefix):
        walk(OSF + "nodes/%s/files/osfstorage/" % nid, prefix)
        for c in _osf_pages(OSF + "nodes/%s/children/" % nid, ctx):
            own = licence_of(c)
            if own and licence_id(own) != licence_id(lic):
                raise LicenceChanged("%s: component %s says %r, the project %r" % (name, c["id"], own, lic))
            node_files(c["id"], prefix + _safe_name(c["attributes"]["title"]) + "/")

    node_files(node, "")
    return out


def github(name, ctx, repo, keep):
    """A GitHub repository's files under `keep` (folders or single files), each read from raw.githubusercontent.com at
    the commit the default branch points to now, so a push mid-download cannot mix two versions. The licence is the
    repository's own, as GitHub's licence detection names it (its SPDX id). A Git LFS pointer is refused, not followed."""
    info = get_json(GH + repo, ctx)
    lic = (info.get("license") or {}).get("spdx_id") or ""
    sha = get_json(GH + "%s/commits/%s" % (repo, info["default_branch"]), ctx)["sha"]
    tree = get_json(GH + "%s/git/trees/%s?recursive=1" % (repo, sha), ctx)
    if tree.get("truncated"):
        raise FetchError("%s: GitHub cut the file list of %s short; not worked around" % (name, repo))
    out = Listing(lic, GH + repo + " (license)", [], ["%s at commit %s" % (repo, sha)])
    raw = "https://raw.githubusercontent.com/%s/%s/" % (repo, sha)
    for x in tree["tree"]:
        if x["type"] == "blob" and x["path"].rsplit("/", 1)[-1] == ".gitattributes":
            if "filter=lfs" in get_bytes(raw + urllib.parse.quote(x["path"]), ctx).decode("utf-8", "replace"):
                raise FetchError("%s: %s keeps files in Git LFS (%s); not followed" % (name, repo, x["path"]))
    for x in tree["tree"]:
        p = x["path"]
        if x["type"] == "blob" and any(p == k or p.startswith(k + "/") for k in keep):
            out.files.append(Entry(p, int(x["size"]), raw + urllib.parse.quote(p), None, meta={"git_blob": x["sha"]}))
    if not out.files:
        raise FetchError("%s: %s holds nothing under %s" % (name, repo, list(keep)))
    return out


def lab_page(name, ctx, page, pattern, links=None):
    """A lab's own page that states the licence and links its files: on Google Drive (TBX11K), or, with `links`
    (a regex for the files' full addresses), directly (RSNA's challenge pages link their S3 bucket)."""
    text = get_bytes(page, ctx).decode("utf-8", "replace")
    plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
    m = re.search(pattern, plain, re.I)
    if not m:
        raise LicenceChanged("%s: the licence sentence is no longer on %s" % (name, page))
    out = Listing(m.group(0), page, [])
    for url in dict.fromkeys(re.findall(links, html.unescape(text))) if links else ():
        size, fname, kind = range_total(url, ctx)
        if not size or "html" in kind:
            raise FetchError("%s: %s answered %s, not a file; not worked around" % (name, url, kind or "no size"))
        out.files.append(Entry(fname or url.rsplit("/", 1)[-1], size, url, None, meta={}))
    if links and not out.files:
        raise FetchError("%s: %s links no file matching %s any more" % (name, page, links))
    for fid in dict.fromkeys(re.findall(r"drive\.google\.com/file/d/([\w-]+)", text)):
        url = "https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t" % fid
        size, fname, kind = range_total(url, ctx)
        if not size or "html" in kind:
            raise FetchError("%s: Google Drive answered %s for %s, not the file (a quota or a sign-in page); "
                             "not worked around" % (name, kind, fid))
        out.files.append(Entry(fname or fid, size, url, None, meta={"drive_id": fid}))
    return out


def _xlsx_rows(blob):
    """The first sheet of an .xlsx as a list of dicts keyed by its header row (no openpyxl here)."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(io.BytesIO(blob))
    strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(ns + "si"):
            strings.append("".join(t.text or "" for t in si.iter(ns + "t")))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for r in ET.fromstring(z.read(sheet)).iter(ns + "row"):
        vals = {}
        for c in r.iter(ns + "c"):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v, t = c.find(ns + "v"), c.get("t")
            if t == "inlineStr":
                vals[col] = "".join(x.text or "" for x in c.iter(ns + "t"))
            elif v is not None:
                vals[col] = strings[int(v.text)] if t == "s" else v.text
        rows.append(vals)
    if not rows:
        return []
    head = rows[0]
    return [{head[k]: r.get(k) for k in head} for r in rows[1:]]


def _site(url):
    return url.replace("://stage.cancerimagingarchive.net", "://www.cancerimagingarchive.net")


def nbia_series(project, ctx):
    """Every series of a TCIA collection, live from the NBIA API (licence, patient, size, images)."""
    for base in (NBIA, NBIA_NLST):
        body = get_bytes(base + "getSeries?Collection=%s&format=json" % urllib.parse.quote(project), ctx,
                         {"Accept": "application/json"}, timeout=600)
        s = json.loads(body) if body.strip() else []     # an unknown collection is an empty body
        if s:
            return base, {x["SeriesInstanceUID"]: x for x in s}
    return None, {}


def series_meta(uid, ctx):
    """One series' metadata from NBIA, under getSeries's field names; None when neither host knows it."""
    for base in (NBIA, NBIA_NLST):
        body = get_bytes(base + "getSeriesMetaData?SeriesInstanceUID=" + uid, ctx, {"Accept": "application/json"})
        m = json.loads(body) if body.strip() else []
        if m:
            m = m[0]
            return base, {"PatientID": m.get("Subject ID"), "ImageCount": m.get("Number of Images"),
                          "LicenseName": m.get("License Name"), "FileSize": m.get("File Size"),
                          "Collection": m.get("Collection"), "Modality": m.get("Modality")}
    return None


def _download_bytes(d):
    """A TCIA download's own size estimate in bytes (its size and unit), 0 when it gives none."""
    unit = {"kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12}.get(str(d.get("download_size_unit")).lower(), 0)
    try:
        return int(float(d.get("download_size") or 0) * unit)
    except ValueError:
        return 0


def tcia(name, ctx, slug, kind="collection", allowed=(), skip_collection_downloads=False):
    """A TCIA collection or analysis result: its downloads (from the site's API), each with its own
    licence. A `.tcia` manifest becomes one entry per DICOM series (IDC's images, or NBIA's zip with an md5
    per image); a direct file is one entry; a package served only over IBM Aspera is one entry, received
    with the IBM Aspera CLI from its public Faspex link (`_aspera`)."""
    page = get_json(TCIA_SITE + ("collections" if kind == "collection" else "analysis-results") + "/?slug=" + slug, ctx)
    if not page:
        raise FetchError("%s: TCIA has no %s %r" % (name, kind, slug))
    page = page[0]
    ids = list(page.get("result_downloads") or []) if kind != "collection" else []
    sources = list(page.get("collection_downloads") or [])
    out = Listing("", TCIA_SITE + "downloads/<id> (each download's data_license)", [])
    if kind != "collection" and skip_collection_downloads and sources:
        out.skipped.append(("source-image downloads %s" % sources, "fetched with their own collection"))
    else:
        ids += sources
    licences = set()
    live = {}
    links = []
    for did in ids:
        d = get_json(TCIA_SITE + "downloads/%s" % did, ctx)
        title = d.get("download_title") or str(did)
        req = str(d.get("download_requirements") or "")
        f = d.get("download_file") if isinstance(d.get("download_file"), dict) else {}
        url = _site(f.get("guid") or "") or (d.get("download_url") or "")
        if f.get("post_type") == "attachment" and not os.path.splitext(urllib.parse.urlsplit(url).path)[1]:
            # Some downloads name a WordPress attachment page ("…/<slug>/", 404 once the page is gone) instead of
            # the file; the collection's own page links the file under wp-content/uploads, named after that slug
            if not links:
                html = get_bytes(_site(page.get("link") or TCIA_SITE), ctx).decode("utf-8", "replace")
                links = sorted(set(re.findall(r"https?://(?:www|stage)\.cancerimagingarchive\.net/wp-content/uploads/"
                                              r"[^\"'\s<>]+", html)))     # TCIA's own files only
            hit = [u for u in links if os.path.splitext(os.path.basename(urllib.parse.urlsplit(u).path))[0].lower()
                   == (f.get("post_name") or "").lower()]
            if not hit:
                raise FetchError("%s: download %r names attachment page %s, and the collection page links no file "
                                 "called %r" % (name, title, url, f.get("post_name")))
            url = _site(hit[0])
        lic = d.get("data_license") or ""
        check_licence("%s download %r" % (name, title), lic, allowed)
        licences.add(lic)
        if "aspera" in req.lower() or "faspex" in url:
            if "faspex" not in url or not os.path.exists(ASCLI):
                out.skipped.append((title, "an IBM Aspera package with no public Faspex link%s" % (
                    "" if os.path.exists(ASCLI) else ", and no ascli at %s" % ASCLI)))
                continue
            est = _download_bytes(d)
            out.files.append(Entry("aspera/" + _safe_name(title), None, url, None, "aspera", {"bytes": est}))
            out.est_bytes += est
            continue
        if not url:
            out.skipped.append((title, "no file on the record (%s)" % (d.get("download_url") or "none")))
            continue
        if url.lower().endswith(".tcia"):
            text = get_bytes(url, ctx).decode("utf-8", "replace")
            lines = [x.strip() for x in text.splitlines()]
            uids = [x for x in lines[lines.index("ListOfSeriesToDownload=") + 1:] if x] if "ListOfSeriesToDownload=" in lines else []
            meta = d.get("download_metadata") if isinstance(d.get("download_metadata"), dict) else {}
            digest = {}
            if meta.get("guid", "").endswith(".xlsx"):
                digest = {r.get("Series Instance UID"): r for r in _xlsx_rows(get_bytes(_site(meta["guid"]), ctx))}
            projects = sorted({(digest.get(u) or {}).get("Project") for u in uids} - {None})
            for p in projects:
                if p not in live:
                    live[p] = nbia_series(p, ctx)
            rows = {}
            for u in uids:
                for p in projects:
                    if u in live[p][1]:
                        rows[u] = (live[p][0], live[p][1][u])
                        break
            rest = [u for u in uids if u not in rows]
            if rest:     # no digest, or series a collection's list leaves out: one lookup each
                with concurrent.futures.ThreadPoolExecutor(max_workers=CAPS["services.cancerimagingarchive.net"]) as pool:
                    for u, got in zip(rest, pool.map(lambda u: series_meta(u, ctx), rest)):
                        if got:
                            rows[u] = got
            missing = [u for u in uids if u not in rows]
            if missing:
                raise FetchError("%s: %d of %d series in %r are unknown to NBIA, e.g. %s" % (name, len(missing), len(uids), title, missing[0]))
            out.files.append(Entry("manifests/" + os.path.basename(urllib.parse.urlsplit(url).path), None, url))
            for u in uids:
                check_licence("%s series %s" % (name, u), rows[u][1].get("LicenseName"), allowed)
                licences.add(rows[u][1].get("LicenseName"))
            mirrored = idc_series(name, ctx, {u: rows[u][1] for u in uids}, allowed)
            for u, entries in mirrored.items():
                out.files += entries
            out.notes.append("%s: %d of %d series from IDC's AWS bucket, the rest from NBIA" % (title, len(mirrored), len(uids)))
            for u in uids:
                if u in mirrored:
                    continue
                base, row = rows[u]
                out.files.append(Entry("series/%s.zip" % u, None, base + "getImageWithMD5Hash?SeriesInstanceUID=" + u,
                                       None, "series", {"patient": row["PatientID"], "uid": u, "images": int(row["ImageCount"]),
                                                        "project": row.get("Collection"), "modality": row.get("Modality"),
                                                        "bytes": int(row.get("FileSize") or 0)}))
                out.est_bytes += int(row.get("FileSize") or 0)
            out.notes.append("%s: %d series (%d looked up one by one)" % (title, len(uids), len(rest)))
        else:
            out.files.append(Entry("files/" + os.path.basename(urllib.parse.urlsplit(url).path), None, url))
            out.est_bytes += _download_bytes(d)
    ids_seen = {licence_id(x) for x in licences}
    out.licence = " + ".join(sorted(licences)) if licences else ""
    out.notes.append("licences seen: %s" % sorted(ids_seen))
    return out



def hash_file(path, kind):
    h = hashlib.new(kind)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * READ), b""):
            h.update(b)
    return h.hexdigest()


def _save_json(path, obj):
    """Write a state file whole, then swap it in. On Windows the swap is refused while another process has the
    file open (a second run reading a set's record), so it is tried again for a few seconds before giving up."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    for wait in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 0):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if not wait:
                raise
            time.sleep(wait)


def _load_json(path):
    """A state file, or None when it is missing or was cut short by a crash."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _range_start(r):
    """Where a 206 answer's bytes begin, from its Content-Range ('bytes S-E/T'); None when unparsable."""
    m = re.match(r"bytes (\d+)-", r.headers.get("Content-Range") or "")
    return int(m.group(1)) if m else None


def _discard(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def _stream(entry, part, ctx, label, resume):
    """One connection, start to end. With `resume` and a known size, an existing .part continues by a
    byte range; otherwise the file starts over. A .chunks note (url and size) ties the .part to the file it
    began as, so a .part left from a different remote file is never continued."""
    state_path = part[:-5] + ".chunks"
    mine = {"url": _ident(entry), "size": entry.size, "stream": True}
    have = os.path.getsize(part) if (resume and os.path.exists(part)) else 0
    if have and (_load_json(state_path) != mine or entry.size is None or have > entry.size):
        have = 0
    if have and have == entry.size:
        return have                               # whole already; download() checks it
    if resume:
        _save_json(state_path, mine)
    t0, last = time.time(), time.time()
    for _ in range(TRIES):
        headers = {**({"Range": "bytes=%d-" % have} if have else {}), **_auth(entry)}
        try:
            with opened(entry.url, ctx, headers, timeout=300, bulk=True) as r:
                if have and (r.status != 206 or _range_start(r) != have):
                    have = 0                      # the host ignored the range or answered another: start over
                start = have
                length = r.headers.get("Content-Length")
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(READ)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if time.time() - last > PROGRESS_S:
                            last = time.time()
                            log("  %s: %.2f GB, %.1f MB/s" % (label, have / 1e9, have / 1e6 / max(last - t0, 1e-9)))
            short = length is not None and length.isdigit() and have - start < int(length)
            if not short and (entry.size is None or have >= entry.size):
                return have
            if short and entry.size is None:
                log("  %s: body ended at %.1f of %.1f MB, again" % (label, (have - start) / 1e6, int(length) / 1e6))
                have = 0
            else:
                log("  %s: connection ended at %.1f MB, resuming" % (label, have / 1e6))
        except (OSError, http.client.HTTPException) as e:
            log("  %s: %s at %.1f MB, again" % (label, type(e).__name__, have / 1e6))
            if entry.size is None:
                have = 0
        time.sleep(5)
    raise FetchError("%s: never arrived whole" % label)


def _preallocate(path, size, keep=False, sparse=True):
    """A file `size` bytes long with nothing written yet (or, with `keep`, the file's own bytes and nothing
    after them). On Windows it is sparse and its end is set directly: Python's truncate() there writes every
    zero itself (measured 2026-10-05: 10 GB files growing at 350 MB/s each, the SSD at its write limit before a
    byte had arrived), and in a sparse file a range written ahead of the ones before it costs no zero-fill either.
    Not `sparse`, the drive holds the whole size at once (nothing is written; NTFS fills only a gap a write skips),
    so every other writer's room check sees the space as taken."""
    with open(path, "r+b" if keep else "wb") as f:
        if sys.platform != "win32":
            f.truncate(size)
            return
        import ctypes  # noqa: PLC0415
        import msvcrt  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class EndOfFile(ctypes.Structure):
            _fields_ = [("EndOfFile", ctypes.c_longlong)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        k32.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        h = wintypes.HANDLE(msvcrt.get_osfhandle(f.fileno()))
        if sparse and not k32.DeviceIoControl(h, 0x900C4, None, 0, None, 0, ctypes.byref(wintypes.DWORD()),
                                              None):  # FSCTL_SET_SPARSE
            raise OSError(ctypes.get_last_error(), "could not make %s sparse" % path)
        eof = EndOfFile(size)
        if not k32.SetFileInformationByHandle(h, 6, ctypes.byref(eof), ctypes.sizeof(eof)):  # FileEndOfFileInfo
            raise OSError(ctypes.get_last_error(), "could not size %s" % path)


def _sparse(path):
    """Whether a range written past the file's valid data costs no zero-fill: a sparse file on Windows, any file
    elsewhere (truncate() leaves holes)."""
    if sys.platform != "win32":
        return True
    import ctypes  # noqa: PLC0415
    a = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    return a != 0xFFFFFFFF and bool(a & 0x200)     # FILE_ATTRIBUTE_SPARSE_FILE


def adopt(path, entry, part, sparse=True):
    """A file holding the first bytes of `entry` in order (a one-stream copy cut short, such as a cancelled
    Globus transfer's) made `part`, for _chunked to resume: each whole range in it but the last is marked done,
    the last because a write may still have been in flight. The bytes kept are returned. With no file at `path`,
    `part` starts empty; not `sparse`, the drive holds its whole size from the start (see _preallocate)."""
    if os.path.exists(part):
        raise FetchError("%s is already under way; adopt only starts one" % part)
    keep = 0
    if os.path.exists(path):
        keep = max(0, os.path.getsize(path) // CHUNK - 1)
        os.replace(path, part)
    _preallocate(part, entry.size, keep=os.path.exists(part), sparse=sparse)
    _save_json(part[:-5] + ".chunks", {"size": entry.size, "chunk": CHUNK, "url": _ident(entry),
                                       "done": list(range(keep))})
    return keep * CHUNK


class _Held:
    """One range held in memory until every range before it is on disk (_chunked's in-order writes). It grows as its
    bytes come, so a connection still waiting for its host, or a slow one, holds only what it has (2026-10-09: a
    hundred Zenodo connections at 0.3 MB/s each would otherwise hold 64 MB each from the start)."""

    def __init__(self):
        self.b, self.at = bytearray(), 0

    def seek(self, p):
        self.at = p

    def write(self, data):
        self.b[self.at:self.at + len(data)] = data
        self.at += len(data)


def _chunked(entry, part, ctx, label):
    """Byte ranges into one preallocated file, resumable through .chunks, which records each 64 MB range once
    it is on disk. Where a host gives one file one connection (PER_FILE, PER_GB), each run of missing ranges comes in
    one request, so a Zenodo file costs Zenodo one request rather than one per 64 MB.

    In a sparse file (every file this makes) each connection writes its range where it goes as it arrives. A file
    that is not sparse (adopt's, sparse=False) has its ranges brought into memory and written in order instead: NTFS
    fills the gap between such a file's valid data and a write past it with zeros first, so ranges written as they
    arrived were written twice (2026-10-07: the spare disk wrote 52 MB/s for 26 MB/s of mammograms, its valid data
    1.8 GB past the last whole range). There at most HELD_EXTRA ranges more than the connections wait in memory for
    a slow one before them. A sparse file costs no such fill (H:, 2026-10-10: 64 MB written 3 GB into a sparse file
    wrote 67 MB to the disk, into one that was not, 3,289 MB), and in order one slow range held up everything: 21 of
    unitochest's finished ranges (1.9 GB) waited in memory and the file took nothing for six minutes."""
    size = entry.size
    state_path = part[:-5] + ".chunks"
    n = -(-size // CHUNK)
    state = None
    if os.path.exists(part) and os.path.getsize(part) == size:
        state = _load_json(state_path)
        if not isinstance(state, dict) or state.get("size") != size or state.get("chunk") != CHUNK or state.get("url") != _ident(entry):
            state = None
    if state is None:
        _preallocate(part, size)
        state = {"size": size, "chunk": CHUNK, "url": _ident(entry), "done": []}
        _save_json(state_path, state)
    done = set(state["done"])
    todo = [i for i in range(n) if i not in done]
    hn = Host.of(entry.url).name
    lacks = sum(min(size, (i + 1) * CHUNK) - i * CHUNK for i in todo)
    by_size = -(-lacks // int(PER_GB[hn] * 1e9)) if hn in PER_GB else len(todo)
    workers = min(cap(hn), PER_FILE.get(hn, len(todo)), by_size, len(todo)) or 1
    runs = []                                      # [first, last] range indices, fetched in one request
    for i in todo:
        if workers == 1 and runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    lock = threading.Lock()
    nxt = iter(runs)
    got = [sum(min(size, (i + 1) * CHUNK) - i * CHUNK for i in done)]
    base = got[0]
    t0, last = time.time(), [time.time()]
    failed = threading.Event()
    ordered = workers > 1 and not _sparse(part)   # each run one range, handed out in `todo` order
    slots = threading.Semaphore(workers + min(HELD_EXTRA, workers))
    held, hlock, writing, k = {}, threading.Lock(), [False], [0]

    def mark(i, f):
        f.flush()
        os.fsync(f.fileno())                       # on disk before .chunks says so
        with lock:
            done.add(i)
            state["done"] = sorted(done)
            _save_json(state_path, state)
            if time.time() - last[0] > PROGRESS_S:
                last[0] = time.time()
                el = last[0] - t0
                log("  %s: %.1f of %.1f GB, %.1f MB/s" % (label, min(got[0], size) / 1e9, size / 1e9,
                                                          (got[0] - base) / 1e6 / max(el, 1e-9)))

    def one(run, f):
        first, final = run
        start, end = first * CHUNK, min(size, (final + 1) * CHUNK) - 1
        pos, nxt_mark = start, first
        tries, seen = 0, pos
        while tries < TRIES:                       # a drop after progress (one request can run for hours) is free
            if failed.is_set() or ctx.stop.is_set():
                raise FetchError("%s: stopped" % label)
            try:
                with opened(entry.url, ctx, {"Range": "bytes=%d-%d" % (pos, end), **_auth(entry)}, timeout=300,
                            bulk=True) as r:
                    if r.status != 206 or _range_start(r) != pos:
                        raise FetchError("%s: the host ignored a byte range or answered another" % label)
                    total = (r.headers.get("Content-Range") or "").rpartition("/")[2]
                    if total.isdigit() and int(total) != size:
                        raise FetchError("%s: the host now says %s bytes, the record %d" % (label, total, size))
                    f.seek(pos - start if ordered else pos)
                    while pos <= end:
                        b = r.read(min(READ, end + 1 - pos))
                        if not b:
                            break
                        f.write(b)
                        pos += len(b)
                        with lock:
                            got[0] += len(b)
                        while not ordered and nxt_mark <= final and pos >= min(size, (nxt_mark + 1) * CHUNK):
                            mark(nxt_mark, f)      # a whole range is in: record it
                            nxt_mark += 1
                if pos > end:
                    return
                why = "body ended early"
            except (OSError, http.client.HTTPException) as e:
                why = type(e).__name__
            log("  %s: range %d %s at %.0f MB, again" % (label, nxt_mark, why, (pos - start) / 1e6))
            tries = 0 if pos > seen else tries + 1
            seen = pos
            time.sleep(5)
        raise FetchError("%s: range %d never arrived whole" % (label, nxt_mark))

    def put(i, buf, f):
        """A whole range in memory; whoever finds the next range in order there writes the ranges that follow."""
        with hlock:
            held[i] = buf
            if writing[0]:
                return
            writing[0] = True
        while True:
            with hlock:
                j = todo[k[0]] if k[0] < len(todo) else None
                b = held.pop(j, None)
                if b is None:
                    writing[0] = False
                    return
            f.seek(j * CHUNK)
            f.write(b.b)
            mark(j, f)
            with hlock:
                k[0] += 1
            slots.release()

    def worker():
        with open(part, "r+b") as f:
            while not failed.is_set():
                if ordered:
                    while not slots.acquire(timeout=5):
                        if failed.is_set():
                            return
                with lock:
                    run = next(nxt, None)
                if run is None:
                    if ordered:
                        slots.release()
                    return
                try:
                    if ordered:
                        buf = _Held()
                        one(run, buf)
                        put(run[0], buf, f)
                        del buf                    # else it stays in memory while this worker waits for a slot
                    else:
                        one(run, f)
                except BaseException:
                    failed.set()
                    raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(worker) for _ in range(workers)]
        for fu in futs:
            fu.result()
    if len(done) != n:
        raise FetchError("%s: %d of %d ranges done" % (label, len(done), n))
    return size


def check_series_zip(path, entry):
    """A TCIA series zip: every member's CRC, every image's md5 from NBIA's own list, and the image count."""
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise FetchError("damaged member %s" % bad)
        names = [n for n in z.namelist() if not n.endswith("/")]
        dcm = [n for n in names if n.lower().endswith(".dcm")]
        sums = {}
        for n in names:
            if os.path.basename(n).lower() == "md5hashes.csv":
                for line in z.read(n).decode("utf-8", "replace").splitlines()[1:]:
                    k, _, v = line.partition(",")
                    if v.strip():
                        sums[os.path.basename(k.strip())] = v.strip().lower()
        if not sums:
            raise FetchError("no md5hashes.csv in the series zip")
        for n in dcm:
            if hashlib.md5(z.read(n)).hexdigest() != sums.get(os.path.basename(n)):
                raise FetchError("md5 of %s is not NBIA's" % n)
    want = entry.meta.get("images") or 0
    if len(dcm) < want:
        raise FetchError("%d images, NBIA lists %d" % (len(dcm), want))
    return {"images": len(dcm), "md5_checked": len(dcm)}


def _aspera(entry, dst, label):
    """A TCIA package over IBM Aspera: ascli receives it from its public Faspex link into dst.part (a run
    cut short resumes there), every file is checked against the package's own .sums (md5) when it has one,
    and the folder takes its name. Returns what was checked, for fetch.json."""
    part = dst + ".part"
    os.makedirs(part, exist_ok=True)
    t0 = time.time()
    # Not through ascli.cmd: a Faspex link carries '&', which cmd.exe takes as a second command, cutting the link
    # and every argument after it (on 2026-10-05 --to-folder was lost and a 19 GB package landed in the run's
    # working folder, the repository). The portable package's Ruby is called as ascli.cmd would call it, and the
    # run's working folder is the part folder, so nothing it writes can land anywhere else.
    root, cmd, env = os.path.dirname(ASCLI), [ASCLI], None
    if ASCLI.lower().endswith(".cmd") and os.path.exists(os.path.join(root, "ruby", "bin", "ruby.exe")):
        cmd = [os.path.join(root, "ruby", "bin", "ruby.exe"), os.path.join(root, "gems", "bin", "ascli")]
        env = dict(os.environ, GEM_HOME=os.path.join(root, "gems"))
        env.setdefault("ASCLI_SDK_FOLDER", os.path.join(root, "sdk"))
    r = subprocess.run(cmd + ["faspex5", "packages", "receive", "--url=" + entry.url, "--to-folder=" + part,
                              "--interactive=no", "--progress-bar=no", "--transfer=direct"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW, env=env, cwd=part)
    if r.returncode != 0:
        raise FetchError("%s: ascli: %s" % (label, (r.stderr or r.stdout).strip()[-400:]))
    files = [os.path.join(d, f) for d, _, fs in os.walk(part) for f in fs]
    if not files:
        raise FetchError("%s: ascli received nothing" % label)
    checked = 0
    for sums in [p for p in files if p.lower().endswith(".sums")]:
        base = os.path.dirname(sums)
        with open(sums, encoding="utf-8", errors="replace") as f:
            lines = [m for m in (re.match(r"([0-9a-fA-F]{32})\s+\*?(.+?)\s*$", x) for x in f) if m]
        named = {os.path.normcase(os.path.join(base, *m.group(2).split("/"))) for m in lines}
        for m in lines:
            p = os.path.join(base, *m.group(2).split("/"))
            if _inside(base, m.group(2)) and not os.path.isfile(p) and os.path.isdir(os.path.dirname(p)):
                # TCIA renamed a file after writing the .sums (Prostate-MRI-US-Biopsy's overlays, 2026-10-05):
                # take the one unnamed file beside it with that md5
                twin = [q for q in (os.path.join(os.path.dirname(p), x) for x in os.listdir(os.path.dirname(p)))
                        if os.path.isfile(q) and os.path.normcase(q) not in named and hash_file(q, "md5") == m.group(1).lower()]
                p = twin[0] if len(twin) == 1 else p
            if not _inside(base, m.group(2)) or not os.path.isfile(p) or hash_file(p, "md5") != m.group(1).lower():
                raise FetchError("%s: %s is missing or its md5 is not the package's" % (label, m.group(2)))
            checked += 1
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(part, dst)
    return {"size": sum(os.path.getsize(p) for p in [os.path.join(d, f) for d, _, fs in os.walk(dst) for f in fs]),
            "files": len(files), "seconds": round(time.time() - t0),
            "checked": ("md5 of %d files against the package's .sums" % checked if checked
                        else "Aspera's own transfer check (the package has no .sums)"),
            "mtime": int(os.path.getmtime(dst))}


def download(entry, dst, ctx, label):
    """Fetch `entry` to `dst` and check it. Returns what was checked, for fetch.json."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if entry.kind == "aspera":
        return _aspera(entry, dst, label)
    part = dst + ".part"
    for attempt in range(1, ATTEMPTS + 1):
        t0 = time.time()
        if entry.size is not None and entry.size > 2 * CHUNK:
            have = _chunked(entry, part, ctx, label)
        else:
            have = _stream(entry, part, ctx, label, resume=entry.size is not None and entry.size > RESUME_MIN)
        if entry.size is not None and have != entry.size:
            raise FetchError("%s: %d bytes, the record says %d" % (label, have, entry.size))
        record = {"size": os.path.getsize(part), "seconds": round(time.time() - t0)}
        # a read of the whole file: on a slow disk, in its turn, unless the file is small (CHECK_TURN_MIN)
        turn = Turn([part] if record["size"] >= CHECK_TURN_MIN else [], "%s: its check" % label)
        turn.take()                                    # outside the try: a turn not had is no failed check
        try:
            with turn:
                if entry.kind == "series":
                    record.update(check_series_zip(part, entry))
                if entry.checksum:
                    kind, _, want = entry.checksum.partition(":")
                    got = hash_file(part, kind)
                    if got != want.lower():
                        raise FetchError("%s %s, the archive says %s" % (kind, got, want))
                    record["checksum"] = entry.checksum
                    record["checked"] = "archive " + kind
                else:
                    record["sha256"] = hash_file(part, "sha256")
                    record["checked"] = "zip CRCs and NBIA md5s" if entry.kind == "series" else "sha256 recorded on arrival"
                    if entry.meta.get("whole_zip"):
                        with zipfile.ZipFile(part) as z:
                            bad = z.testzip()
                            if bad:
                                raise FetchError("damaged member %s" % bad)
                            record["members"] = len(z.namelist())
                        record["checked"] = "every member's CRC; sha256 recorded on arrival"
        except (FetchError, zipfile.BadZipFile, OSError) as e:
            log("%s: failed its check on attempt %d (%s)" % (label, attempt, e))
            _discard(part, part[:-5] + ".chunks")
            if attempt == ATTEMPTS:
                raise FetchError("%s: failed its check twice (%s)" % (label, e)) from None
            continue
        os.replace(part, dst)
        _discard(part[:-5] + ".chunks")
        record["mtime"] = int(os.path.getmtime(dst))
        return record
    raise FetchError("%s: not fetched" % label)


def recheck(entry, dst):
    """A file already in the archive with no record (a run stopped before saving one): check it again."""
    rec = {"size": os.path.getsize(dst)}
    if entry.size is not None and rec["size"] != entry.size:
        return None
    turn = Turn([dst] if rec["size"] >= CHECK_TURN_MIN else [], "%s: its check again" % os.path.basename(dst))
    turn.take()                                    # outside the try: a turn not had is no failed check
    try:
        with turn:
            if entry.kind == "series":
                rec.update(check_series_zip(dst, entry))
            if entry.checksum:
                kind, _, want = entry.checksum.partition(":")
                if hash_file(dst, kind) != want.lower():
                    return None
                rec["checksum"], rec["checked"] = entry.checksum, "archive " + kind
            else:
                rec["sha256"] = hash_file(dst, "sha256")
                rec["checked"] = "re-checked on resume; sha256 recorded"
    except (FetchError, zipfile.BadZipFile, OSError):
        return None
    rec["mtime"] = int(os.path.getmtime(dst))
    return rec



ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".rar", ".7z")


def is_archive(name):
    n = name.lower()
    return n.endswith(ARCHIVE_EXT) or bool(re.search(r"\.(zip|7z|rar)\.0*1$", n))


def _inside(root, member):
    root = os.path.abspath(root)
    m = member.replace("\\", "/")
    if m.startswith("/") or re.match(r"^[A-Za-z]:", m) or any(p == ".." for p in m.split("/")):
        return False
    p = os.path.abspath(os.path.join(root, m))
    return p == root or p.startswith(root + os.sep)


def _seven_zip_list(path):
    r = subprocess.run([SEVEN_ZIP, "l", "-slt", "-ba", "-p", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise FetchError("7-Zip could not list %s: %s" % (path, (r.stderr or r.stdout)[-300:]))
    links = [line for line in r.stdout.splitlines()
             if re.match(r"(Symbolic Link|Hard Link|Link) = \S", line)]
    if links:
        raise FetchError("%s holds links, which are not unpacked: %s" % (path, links[:3]))
    return [line[7:] for line in r.stdout.splitlines() if line.startswith("Path = ")]


def unpacked_size(path):
    """Bytes `path` takes once unpacked: zip and 7-Zip listings' own sizes; a compressed tar, unread, at
    four times its size; a loose file or a folder (an Aspera package), its size."""
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    n = path.lower()
    if n.endswith(".zip") and not os.path.exists(path[:-4] + ".z01"):
        with zipfile.ZipFile(path) as z:
            return sum(i.file_size for i in z.infolist())
    if n.endswith(".tar"):
        return os.path.getsize(path)
    if n.endswith((".tar.gz", ".tgz", ".tar.bz2")):
        return 4 * os.path.getsize(path)
    if is_archive(n):
        r = subprocess.run([SEVEN_ZIP, "l", "-slt", "-ba", "-p", path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        sizes = [int(x[7:]) for x in r.stdout.splitlines() if x.startswith("Size = ") and x[7:].isdigit()]
        return sum(sizes) if r.returncode == 0 and sizes else 4 * os.path.getsize(path)
    return os.path.getsize(path)


def unpack(path, out_dir, turn=None, release=False):
    """Unpack one archive into out_dir after checking that no member escapes it. Returns the count. A tar shares
    `turn` (a Turn held) between its members. With `release`, a plain tar gives its bytes back to its drive as they
    are unpacked (_untar), so only a caller that removes the archive afterwards passes it."""
    os.makedirs(out_dir, exist_ok=True)
    n = path.lower()
    split_zip = n.endswith(".zip") and os.path.exists(path[:-4] + ".z01")
    if n.endswith(".zip") and not split_zip:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            bad = [x for x in names if not _inside(out_dir, x)]
            if bad:
                raise FetchError("%s: members escape the folder: %s" % (path, bad[:3]))
            z.extractall(out_dir)
        return len(names)
    if n.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        return _untar(path, out_dir, turn, release)
    names = _seven_zip_list(path)
    bad = [x for x in names if not _inside(out_dir, x)]
    if bad:
        raise FetchError("%s: members escape the folder: %s" % (path, bad[:3]))
    r = subprocess.run([SEVEN_ZIP, "x", "-y", "-p", "-aoa", "-o" + out_dir, path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise FetchError("7-Zip could not unpack %s: %s" % (path, (r.stderr or r.stdout)[-300:]))
    return len(names)


def _out_whole(out_dir, member):
    try:
        st = os.stat(os.path.join(out_dir, *member.name.split("/")))
    except OSError:
        return False
    return st.st_size == member.size and int(st.st_mtime) == int(member.mtime)


UNTAR_HELD = 512 << 20         # bytes read ahead of the writes while their .npz checks run on other cores
UNTAR_WHOLE = 64 << 20         # a bigger member is copied through, never held in memory whole
UNTAR_HELD_FILES = 4096        # and at most this many: OMAMA's 163,568 metadata .json (about 400 bytes each) were all
                               # held at once, then written in one drain of five hours on G: that neither said how far
                               # it was nor shared its turn (2026-10-08)
UNTAR_SAY_S = 600              # a progress line at least this often, however small the members
UNTAR_RECHECK = 4 << 30        # on a resume, this much found whole just before the first member not whole is compared
                               # with the tar again: a crash (not a restart) can leave a file's size and time on the
                               # disk without all of its bytes; Windows held at most 1 GB unwritten for G: (2026-10-07)
UNTAR_RELEASE_MIN = 64e9      # a plain tar this big is given back to its drive as it is unpacked, when unpack() is asked
                               # to: omama-db-2d's 796 GB tar beside its 796 GB unpacked would have needed 0.8 TB of G:
                               # more for hours, with other files on the rest of it (2026-10-08)
UNTAR_RELEASE_LAG = 16 << 30   # only bytes this far behind the oldest member not yet written are given back: past a
                               # resume's UNTAR_RECHECK and Windows' 1 GB unwritten, so a crash never takes a member's
                               # only copy
UNTAR_RELEASE_STEP = 4 << 30   # given back this much at a time
DEVICE_NAME = re.compile(r"(?i)^(con|prn|aux|nul|conin\$|conout\$|com[0-9¹²³]|lpt[0-9¹²³])(\..*)?$")


def _npz_bad(src):
    """The first member of a .npz (bytes, or a path) whose CRC-32 fails, or why it cannot be read; None when whole."""
    try:
        with zipfile.ZipFile(io.BytesIO(src) if isinstance(src, bytes) else src) as z:
            return z.testzip()
    except Exception as e:  # noqa: BLE001  whatever stops it being read: a bad CRC, an unknown method, a password
        return "unreadable (%s: %s)" % (type(e).__name__, e)


def untar_releases(path):
    """Whether unpack(path, ..., release=True) gives the archive's bytes back to its drive as it goes."""
    return path.lower().endswith(".tar") and os.path.isfile(path) and os.path.getsize(path) >= UNTAR_RELEASE_MIN


def unpack_need(path, release=False):
    """The room unpacking `path` takes on the archive's own drive: its unpacked size, or for a tar given back as
    it goes (untar_releases, with `release`), only the bytes out ahead of what has been given back."""
    if release and untar_releases(path):
        return UNTAR_RELEASE_LAG + 2 * UNTAR_RELEASE_STEP + UNTAR_HELD
    return unpacked_size(path)


def _released(path):
    """(offset, members before it) from `path`.released: the tar's bytes before offset are given back, and
    offset is a member's header, where its unpack goes on. (0, 0) when none are."""
    try:
        with open(path + ".released", encoding="utf-8") as f:
            d = json.load(f)
        return int(d["offset"]), int(d["members"])
    except FileNotFoundError:
        return 0, 0


def _release(path, end, members):
    """Bytes 0 to `end` of `path` given back to its drive: the place to go on from saved first (so a crash
    between leaves bytes still there, never a gap to read), then the file made sparse and the range zeroed
    (FSCTL_SET_ZERO_DATA, which frees a sparse file's clusters). The file's time is kept, as its record has it."""
    side = path + ".released"
    with open(side + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"offset": end, "members": members}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(side + ".tmp", side)
    if sys.platform != "win32":
        return
    import ctypes  # noqa: PLC0415
    import msvcrt  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class ZeroData(ctypes.Structure):
        _fields_ = [("FileOffset", ctypes.c_longlong), ("BeyondFinalZero", ctypes.c_longlong)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    st = os.stat(path)
    with open(path, "r+b") as f:
        h = wintypes.HANDLE(msvcrt.get_osfhandle(f.fileno()))
        if not k32.DeviceIoControl(h, 0x900C4, None, 0, None, 0, ctypes.byref(wintypes.DWORD()), None):  # FSCTL_SET_SPARSE
            raise OSError(ctypes.get_last_error(), "could not make %s sparse" % path)
        z = ZeroData(0, end)
        if not k32.DeviceIoControl(h, 0x980C8, ctypes.byref(z), ctypes.sizeof(z), None, 0,
                                   ctypes.byref(wintypes.DWORD()), None):  # FSCTL_SET_ZERO_DATA
            raise OSError(ctypes.get_last_error(), "could not give back the start of %s" % path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def _plain_name(name):
    """A member name Windows writes where it says: no device (CON, NUL, COM1...), no stream (a colon), and no part
    ending in a dot or a space, which Windows drops, so two names could land on one file."""
    return all(not DEVICE_NAME.match(x) and ":" not in x and x[-1] not in ". "
               for x in name.split("/") if x not in ("", "."))


def _untar(path, out_dir, turn=None, release=False):
    """A tar unpacked in one pass, read in order: a big tar on a slow disk is read once, not three times (its
    listing, its check and its unpack; OMAMA's 796 GB on the USB disk, 2026-10-07). The Turn it holds is shared
    between two members, what it holds in memory written first, so a job ranked ahead of it need not wait hours
    for one archive's end (NLST's last 42 GB behind OMAMA's five-hour unpack, 2026-10-07). Each member is checked as it
    comes: none may escape out_dir, be a link or a name Windows would put elsewhere (tarfile's "data" filter
    besides), and a .npz member's CRC-32s are checked in memory, on other cores, before it is written; a bad one
    stops the unpack before it lands, its byte range named so it can be fetched again. A member already out whole
    (its size, and the time set only once its data is written) is passed over unread, so a run cut short goes on
    where it stopped; the last UNTAR_RECHECK of those before the first one written are compared with the tar first.
    With `release` (a plain tar of UNTAR_RELEASE_MIN or more), the bytes UNTAR_RELEASE_LAG behind the oldest member
    not yet written are given back to the drive as it goes (_release), so the tar and its unpacked copy never take
    the drive twice over; a run cut short goes on from the last member header given back to (`path`.released).
    A name the tar holds twice is written each time, so its last copy wins. A plain tar must end in zeros within a
    megabyte of its file's end, since tarfile stops quietly at a header it cannot read, as it would at a range never
    written (zeros). Returns the member count."""
    plain = path.lower().endswith(".tar")
    size = os.path.getsize(path)
    name = os.path.basename(path)
    count = npz = got = wrote = 0
    start, count = _released(path) if plain else (0, 0)  # a tar given back in part goes on where that stopped
    release = release and plain and untar_releases(path)
    given, marks = start, collections.deque()      # marks: (header offset, members before it) not yet given back
    held, made, seen = collections.deque(), set(), set()   # held: (member, dest, data, check) in the tar's order
    recent, rechecked = collections.deque(), not plain
    recent_bytes = in_mem = 0
    t0, mark = time.time(), 10e9
    said = t0

    def crc_fault(m, bad):
        return FetchError("%s: %s fails its CRC (%s); bytes %d-%d of the tar, to fetch again" % (
            path, m.name, bad, m.offset_data, m.offset_data + m.size - 1))

    def place(dest):
        parent = os.path.dirname(dest)
        if parent not in made:
            os.makedirs(parent, exist_ok=True)
            made.add(parent)

    def write(item):
        m, dest, data, check = item
        bad = check.result() if check else None
        if bad:
            raise crc_fault(m, bad)
        place(dest)
        with open(dest, "wb") as f:
            f.write(data)
        os.utime(dest, (m.mtime, m.mtime))        # last: a file cut short keeps another time and is written again

    def stream(m, dest):
        """A member copied through from the tar, not held whole: a big one, or one written again."""
        place(dest)
        with t.extractfile(m) as src, open(dest, "wb") as f:
            shutil.copyfileobj(src, f, 8 << 20)
            short = f.tell() != m.size
        if short:
            raise FetchError("%s: %s ends early" % (path, m.name))
        bad = _npz_bad(dest) if m.name.lower().endswith(".npz") else None
        if bad:
            os.remove(dest)
            raise crc_fault(m, bad)
        os.utime(dest, (m.mtime, m.mtime))

    def same(m, dest):
        with t.extractfile(m) as a, open(dest, "rb") as b:
            while True:
                x, y = a.read(8 << 20), b.read(8 << 20)
                if x != y:
                    return False
                if not x:
                    return True

    def recheck():
        again = [(m, dest) for m, dest in recent if not same(m, dest)]
        for m, dest in again:
            stream(m, dest)
        if recent:
            log("  %s: the %d files (%.1f GB) found whole just before this run's first write compared with the tar; "
                "%d written again" % (name, len(recent), recent_bytes / 1e9, len(again)))
        recent.clear()

    if start:
        log("  %s: the first %.0f GB were unpacked and given back to the drive; going on from there" % (
            name, start / 1e9))
    with concurrent.futures.ThreadPoolExecutor(4) as pool, open(path, "rb", buffering=16 << 20) as raw:
        raw.seek(start)                            # tarfile starts at the handle's place, its offsets the file's own
        t = tarfile.open(fileobj=raw, mode="r:" if plain else "r:*")
        for m in t:
            if release:
                marks.append((m.offset, count))
                upto = (held[0][0].offset if held else m.offset) - UNTAR_RELEASE_LAG
                while len(marks) > 1 and marks[1][0] <= upto:
                    marks.popleft()
                if marks[0][0] <= upto and marks[0][0] - given >= UNTAR_RELEASE_STEP:
                    given = marks[0][0]
                    _release(path, given, marks[0][1])
            count += 1
            if not _inside(out_dir, m.name) or m.issym() or m.islnk():
                raise FetchError("%s: member %s escapes the folder or is a link" % (path, m.name))
            if not _plain_name(m.name):
                raise FetchError("%s: member %s names a device or a stream, or a part ends in a dot or a space" % (
                    path, m.name))
            try:
                m = tarfile.data_filter(m, out_dir)
            except (tarfile.FilterError, OSError) as e:
                raise FetchError("%s: member %s: %s" % (path, m.name, e)) from None
            dest = os.path.join(out_dir, *m.name.split("/"))
            if m.isdir():
                os.makedirs(dest, exist_ok=True)
                continue
            if not m.isfile():
                continue
            if turn is not None and turn.due():
                while held:
                    write(held.popleft())
                in_mem = 0
                turn.share()
            twice = m.name in seen
            seen.add(m.name)
            if not twice and _out_whole(out_dir, m):
                if not rechecked:
                    recent.append((m, dest))
                    recent_bytes += m.size
                    while recent_bytes > UNTAR_RECHECK:
                        recent_bytes -= recent.popleft()[0].size
                continue
            if not rechecked:
                recheck()
                rechecked = True
            is_npz = m.name.lower().endswith(".npz")
            if m.size > UNTAR_WHOLE:
                while held:
                    write(held.popleft())
                in_mem = 0
                stream(m, dest)
            else:
                data = t.extractfile(m).read()
                if len(data) != m.size:
                    raise FetchError("%s: %s ends early" % (path, m.name))
                held.append((m, dest, data, pool.submit(_npz_bad, data) if is_npz else None))
                in_mem += len(data)
                while in_mem > UNTAR_HELD or len(held) > UNTAR_HELD_FILES:
                    item = held.popleft()
                    in_mem -= len(item[2])
                    write(item)
            npz += is_npz
            got += m.size
            wrote += 1
            if m.offset_data >= mark or time.time() - said > UNTAR_SAY_S:
                log("  %s: %.0f of %s GB through, %d files out in this run, %d .npz files checked, %.1f MB/s "
                    "unpacked" % (name, m.offset_data / 1e9, "%.0f" % (size / 1e9) if plain else "?",
                                  wrote - len(held), npz, got / 1e6 / max(1, time.time() - t0)))
                if m.offset_data >= mark:
                    mark = m.offset_data + 10e9
                said = time.time()
        if not rechecked:
            recheck()                              # all of it was out: the run before stopped after its last write
        while held:
            write(held.popleft())
        end = t.offset
        t.close()
        if plain:
            left = size - end
            raw.seek(end)
            if not 0 <= left <= 1 << 20 or raw.read(left).strip(b"\0"):
                raise FetchError("%s: the archive stops at byte %d of %d; the rest is not its end-of-archive zeros" % (
                    path, end, size))
    if given > start:
        log("  %s: %.0f GB of it given back to the drive as it was unpacked" % (name, given / 1e9))
    if npz:
        log("  %s: %d members; %d .npz files written in this run, the CRC-32 of each one's every member checked "
            "first; the archive ends in its zeros" % (name, count, npz))
    return count


def series_folder(out_dir, entry_meta):
    """<out_dir>/<PatientID>/<SeriesInstanceUID>, once both are plain folder names."""
    parts = (str(entry_meta["patient"]), str(entry_meta["uid"]))
    if not all(re.fullmatch(r"[A-Za-z0-9._-]+", p) and p not in (".", "..") for p in parts):
        raise FetchError("patient or series id %r is not a plain folder name" % (parts,))
    return os.path.join(out_dir, *parts)


def unpack_series(path, out_dir, entry_meta):
    """A TCIA series zip into <PatientID>/<SeriesInstanceUID>/, flat, as modalities/ct/fetch_tcia.py lays it."""
    folder = series_folder(out_dir, entry_meta)
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        names = [x for x in z.namelist() if x.lower().endswith(".dcm")]
        for x in names:
            with open(os.path.join(folder, os.path.basename(x)), "wb") as f:
                f.write(z.read(x))
    with open(os.path.join(folder, ".unpacked"), "w") as f:
        f.write("%d files\n" % len(names))
    return len(names)



# One heavy job at a time on a slow disk, the one ranked first going next. The owner's spare USB disk G: (a 5,400 rpm
# shingled drive) did 12 MB/s for eight jobs together and 29 MB/s for one alone, which was still held back by its
# connections (measured 2026-10-07): the jobs took turns at the disk's heads anyway, only far worse. Each unpack,
# re-encode, move, checksum read and the mammograms' download takes its turn; a download from a slow server does
# not (it would hold the disk for hours at its server's pace). RAIDIOLOGY_TURN_DRIVES names the disks ("" none).
TURN_DRIVES = {d.strip().upper() for d in os.environ.get("RAIDIOLOGY_TURN_DRIVES", "G:").split(";") if d.strip()}
TURN_POLL_S = 10
TURN_SHARE_S = 300
# A checksum read of a file smaller than this takes no turn: it is over in moments, where a turn cost up to TURN_POLL_S
# a file (2026-10-08: TCIA's small DICOM files checked one every 6 s, then none at all behind the mammograms' download).
# RAIDIOLOGY_CHECK_TURN_MIN, in bytes; the Zenodo refill sets it past any file, so its slow downloads never stand idle
# in the disk's line (the owner, 2026-10-08: the slow ones first).
CHECK_TURN_MIN = float(os.environ.get("RAIDIOLOGY_CHECK_TURN_MIN", 1e9))


def _byte_lock(f, lock=True):
    """The OS's own lock on one byte past anything the file holds: it ends with its process however that ends."""
    f.seek(1 << 20)
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl  # noqa: PLC0415
        fcntl.lockf(f.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if lock else fcntl.LOCK_UN, 1, 1 << 20)


def _alive(ticket):
    """A waiter's ticket its process still holds; a dead process's ticket is removed."""
    try:
        f = open(ticket, "r+b")  # noqa: SIM115   (never "a": that would make again a ticket just removed)
    except FileNotFoundError:
        return False
    except OSError:
        return True                                # being made or removed this moment
    try:
        _byte_lock(f)
    except OSError:
        f.close()
        return True
    _byte_lock(f, False)
    f.close()
    try:
        os.remove(ticket)
    except OSError:                                # still open: its maker has not locked it yet
        return os.path.exists(ticket)
    return False


class Turn:
    """A turn at the slow disks (TURN_DRIVES) that `paths` are on, junctions followed; on no such disk it is always
    there. Each waiter keeps a ticket in <disk>/rAidiology/.turns named by its rank (0 goes first) and the time it
    began to wait, locked by its process so a dead one's ticket is seen and dropped. The waiter whose ticket sorts
    first takes turn.lock, an OS lock it holds while it works, so a job killed mid-work frees the disk at once. A turn
    is never taken from its holder; a long job hands it on between its pieces (share). Not re-entrant: a job holding
    a turn starts no other job that wants one on the same disk."""

    def __init__(self, paths, label, rank=5):
        self.drives = sorted({os.path.splitdrive(os.path.realpath(p))[0].upper() for p in paths} & TURN_DRIVES)
        self.label, self.rank = label, rank
        self.held = []                             # (ticket file, ticket path, turn.lock file) per disk
        self.at = 0.0

    def __enter__(self):
        if not self.held:                          # a turn taken already (take() before a try) is held to the end
            self.take()
        return self

    def __exit__(self, *exc):
        self.give()

    def take(self):
        try:
            for d in self.drives:                  # in one order, so two jobs wanting the same two disks never deadlock
                self.held.append(self._take(d))
        except BaseException:
            self.give()
            raise
        self.at = time.time()

    def give(self):
        while self.held:
            tf, tp, lf = self.held.pop()
            for f in (lf, tf):
                with contextlib.suppress(OSError):
                    _byte_lock(f, False)
                f.close()
            with contextlib.suppress(OSError):
                os.remove(tp)

    def due(self):
        """TURN_SHARE_S has passed since the turn was taken: share() would hand it on now."""
        return bool(self.held) and time.time() - self.at > TURN_SHARE_S

    def share(self):
        """Between two pieces of a long job: once TURN_SHARE_S has passed, the turn handed on and waited for again,
        so a job ranked first, or one of the same rank waiting longer, goes next."""
        if self.due():
            self.give()
            self.take()

    def _take(self, drive):
        q = os.path.join(drive + os.sep, "rAidiology", ".turns")
        os.makedirs(q, exist_ok=True)
        name = "%d-%017.6f-%d-%d.ticket" % (self.rank, time.time(), os.getpid(), threading.get_ident())
        tp = os.path.join(q, name)
        tf = lf = None
        t0, said = time.time(), False

        def ahead():
            return [n for n in sorted(os.listdir(q)) if n.endswith(".ticket") and n < name and _alive(os.path.join(q, n))]
        try:
            tf = open(tp, "a+b")  # noqa: SIM115
            tf.write(self.label.encode("utf-8", "replace"))
            tf.flush()
            for i in range(100):                   # a waiter checking whether this new ticket is alive holds its
                try:                               # lock for a moment (seen 2026-10-07)
                    _byte_lock(tf)
                    break
                except OSError:
                    if i == 99:
                        raise
                    time.sleep(0.05)
            while True:
                before = ahead()
                if not before:
                    lf = open(os.path.join(q, "turn.lock"), "a+b")  # noqa: SIM115
                    try:
                        _byte_lock(lf)
                    except OSError:
                        lf.close()             # its holder is working; it was ahead of us, or came before the line
                        lf = None
                    else:
                        # a waiter that began earlier, whose ticket was not yet on the disk at the first look, goes first
                        before = ahead()
                        if not before:
                            if said:
                                with contextlib.suppress(Exception):
                                    log("%s: has its turn on %s after %.0f min" % (
                                        self.label, drive, (time.time() - t0) / 60))
                            return tf, tp, lf
                        _byte_lock(lf, False)
                        lf.close()
                        lf = None
                if not said:
                    said = True
                    log("%s: waits its turn on %s (%d ahead of it)" % (self.label, drive, len(before)))
                time.sleep(TURN_POLL_S)
        except BaseException:
            for f in (lf, tf):
                if f is not None:
                    with contextlib.suppress(OSError):
                        _byte_lock(f, False)
                    f.close()
            with contextlib.suppress(OSError):
                os.remove(tp)
            raise


FLOOR = 0.20   # of every drive kept free: no drive is taken past 80% full (owner, 2026-10-05: "Use whatever you need.
#                But don't make my drives all be fill"; 2026-10-06, of 90%: "still pretty full")
# A drive the owner set apart: (kept free under what stays there, kept free under a download only passing through).
# H:, the owner's 8 TB CMR disk (2026-10-09): "you can take H up to, like, 95% as long as you know you're gonna bring
# it back down. It's fine for staging. I just don't want it to, like, stay over 85% after long periods of time". A
# download passes through when it is unpacked onto another drive and then removed (fetch_public.passes). Room is
# claimed per process, so the 10% between the two is also the margin for claims two processes cannot see.
FLOORS = {"H:": (0.15, 0.05)}


def floor(path, passing=False):
    """The share of path's drive kept free: under what stays there, or (passing) under a download on its way through."""
    drive = os.path.splitdrive(os.path.abspath(path).replace("\\\\?\\", ""))[0].upper()   # a junction's own target form
    stay, through = FLOORS.get(drive, (FLOOR, FLOOR))
    return through if passing else stay


def free_bytes(path):
    return shutil.disk_usage(path).free


def room(path, passing=False):
    """What path's drive can still take: its free space less what the drive keeps free (floor), and less what the
    training copies at work there will still write (growing)."""
    u = shutil.disk_usage(path)
    return u.free - u.total * floor(path, passing) - growing(path)


ARCHIVE = os.environ.get("RAIDIOLOGY_ARCHIVE", "E:/rAidiology/archive")
AT_WORK_S = 6 * 3600   # a train.json untouched this long is a copy no longer at work: it holds no room
_GROWING = {"at": 0.0, "files": {}}   # train.json path: (drive, bytes still to write)
_GROWING_LOCK = threading.Lock()


def growing(path):
    """Bytes the unfinished training copies (tools/train_copy.py) will still write on path's drive. A copy grows a
    series at a time with no claim another process can see (NLST's two shards write about 0.7 TB to H: over a day),
    so its train.json stands in for one: the bytes it kept so far over the series done, times the series left. A
    record marked "held" (its run paused, its series left to another copy or to none) grows no more (2026-10-10: the
    paused 1.5 mm NLST copies stood for 675 GB on G:, the paused nlst-exact for 82 GB on E:, none of it coming).
    Read at most once a minute; a record caught mid-write keeps the value it had."""
    drive = os.path.splitdrive(os.path.abspath(path))[0].upper()
    with _GROWING_LOCK:
        if time.time() - _GROWING["at"] > 60:
            files = {}
            for p in glob.glob(os.path.join(ARCHIVE, "*", "train.json")):
                try:
                    if time.time() - os.path.getmtime(p) > AT_WORK_S:
                        continue
                    with open(p, encoding="utf-8") as f:
                        d = json.load(f)
                    done = d.get("series") or {}
                    left = (d.get("picked") or 0) - len(done)
                    if not done or left <= 0 or d.get("finished") or d.get("held") or not d.get("working_copy"):
                        continue
                    out = sum(s.get("bytes_out") or 0 for s in done.values())
                    files[p] = (os.path.splitdrive(os.path.abspath(d["working_copy"]))[0].upper(), out // len(done) * left)
                except (OSError, ValueError, TypeError, AttributeError):
                    if p in _GROWING["files"]:
                        files[p] = _GROWING["files"][p]
            _GROWING.update(at=time.time(), files=files)
        return sum(n for k, n in _GROWING["files"].values() if k == drive)


def on_disk(dst):
    """Bytes of dst's unfinished download already written: its .part as streamed, or for a chunked one (a sparse file,
    whose length says nothing) the ranges its .chunks note records as done."""
    part, note = dst + ".part", dst + ".chunks"
    if not os.path.exists(part):
        return 0
    st = _load_json(note) if os.path.exists(note) else None
    if isinstance(st, dict) and isinstance(st.get("done"), list):
        return min(len(st["done"]) * (st.get("chunk") or CHUNK), st.get("size") or 0)
    return os.path.getsize(part)


def network_drive(path):
    """Whether path is on a network share (a mapped drive such as the Plex server's O:)."""
    if sys.platform != "win32":
        return False
    import ctypes  # noqa: PLC0415
    return ctypes.windll.kernel32.GetDriveTypeW(os.path.splitdrive(os.path.abspath(path))[0] + "\\") == 4  # DRIVE_REMOTE


def is_link(path):
    """A junction, or the symbolic link a junction cannot be when its target is on a network share."""
    return os.path.isjunction(path) or os.path.islink(path)


def junction(link, target):
    """Make `link` an NTFS junction to `target`, or confirm it already is one. A target on a network share gets a
    directory symbolic link instead: a junction can only point at a local drive."""
    if os.path.lexists(link):
        if is_link(link) and os.path.normcase(os.path.abspath(os.readlink(link)).replace("\\\\?\\", "")) == os.path.normcase(os.path.abspath(target)):
            return False
        raise FetchError("%s exists and is not a junction to %s; left alone" % (link, target))
    if sys.platform != "win32" or network_drive(target):
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(os.path.abspath(target), link, target_is_directory=True)
        return True
    import _winapi  # noqa: PLC0415
    os.makedirs(os.path.dirname(link), exist_ok=True)
    _winapi.CreateJunction(os.path.abspath(target), os.path.abspath(link))
    return True
