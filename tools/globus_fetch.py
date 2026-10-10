"""Fetch Dataverse files held in a Globus store, which answer a plain (guestbook-signed) download with 404: OMAMA-DB's
originals on Harvard Dataverse. The way in is Globus:

1. The owner's Globus identity: `globus login` opens a browser page; the owner signs in to Globus and allows the CLI.
   This is the one step that is the owner's. If Harvard's store asks for its own consent, the page opens once more.
2. This PC as a Globus endpoint: Globus Connect Personal for Linux, in WSL (Ubuntu) as the unprivileged user
   `globus`, registered by the CLI's `gcp create mapped` (its one-time setup key is handed over in the environment,
   never printed) and allowed to write only under the cold disk's archive folder (/mnt/h/rAidiology/cold/archive).
3. Read access: Dataverse's requestGlobusDownload grants the owner's identity read access to the files. It needs no
   Dataverse account; the dataset's guestbook is answered as for any download (the company's public contact).
4. A Globus transfer per set, checksums verified end to end by Globus, into the set's cold folder (the junction the
   fetcher makes for a cold set). When one succeeds, its fetch.json is written as the fetcher's would be and
   fetch_public.py's unpack, shrink and record follow.
5. A set fetched over the collection's HTTPS server instead (--https, byte ranges on several connections) is then
   compared by a Globus transfer with sync-level checksum, unless its files are tars Dataverse gives no checksum
   for, as Harvard's Globus-held files are ("not available in dataverse"). Those are checked on this PC as they are
   unpacked, in one read of each tar (core.fetch's unpack): every tar header, every .npz member's CRC-32 before it is
   written, and that the archive runs to the file's end. A check of its own first would read the 796 GB tar on the
   slow disk once more (2026-10-07). Globus's comparison reads this PC's copy through WSL; on 2026-10-06 it was reset
   after almost five hours on OMAMA's 105 GB tar on the busy spare disk, and began again.

State (the endpoint's id, each set's task id) is kept in <archive>/_globus/state.json, so a rerun picks up where the
last one stopped. Prints no token, key or signed link.

    python tools/globus_fetch.py omama-db-2d omama-db-multiview
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

# numpy's OpenBLAS sets aside buffers for every core when it is imported: 0.8 GB of committed memory per process
# on this 32-core PC, 0.1 GB with one thread (measured 2026-10-06). Nothing here does matrix maths.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
from core import fetch  # noqa: E402
import fetch_public as fp  # noqa: E402

TOOLS = os.path.join(os.path.dirname(REPO), "_tools")
GLOBUS = os.path.join(TOOLS, "globus-venv", "Scripts", "globus.exe")
GLOBUS_PY = os.path.join(TOOLS, "globus-venv", "Scripts", "python.exe")
# A `globus` command whose browser page is not opened in the default browser (on 2026-10-05 the owner's Chrome froze
# on it) but written to SIGN_IN, for the page to be opened where the owner can use it, and which waits until the
# owner is through: the CLI gives up after an hour
SIGN_IN = "E:/rAidiology/browser/globus-sign-in.txt"
AUTH = """import sys, webbrowser, globus_cli
from globus_cli.login_manager import local_server as m
def wait_for_code(self):
    while True:
        try:
            return self._auth_code_queue.get(block=True, timeout=3600)
        except m.queue.Empty:
            pass
def page(url, *a, **k):
    with open(%r, "w", encoding="utf-8") as f:
        f.write(url)
    return True
m.RedirectHTTPServer.wait_for_code = wait_for_code
webbrowser.open = page
sys.argv = ["globus"] + sys.argv[1:]
globus_cli.main()
""" % SIGN_IN
# The Authorization header for a collection's HTTPS server, written to this script's stdout only (a pipe to
# HttpsAuth, never a log); exits non-zero when the owner has not allowed HTTPS reads of that collection yet
HTTPS_TOKEN = """import sys
from globus_cli.login_manager import LoginManager
sys.stdout.write(LoginManager()._get_client_authorizer(sys.argv[1]).get_authorization_header() or "")
"""
PY = os.path.join(TOOLS, "venv", "Scripts", "python.exe")
DISTRO, WSL_USER = "Ubuntu", "globus"
GCP = "/home/globus/gcp/globusconnectpersonal"           # GCP 3.3.1 for Linux, unpacked there
GCP_NAME = "rAidiology PC"
STATE = os.path.join(fp.ARCHIVE, "_globus", "state.json")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
POLL_S = 600
WATCH_S = 7 * 86400                     # 876 GB at even 5 MB/s is two days; past a week, something needs a person
log = fetch.log


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    fetch._save_json(STATE, st)


def globus(*args, data=None, fmt=True):
    """The Globus CLI; its answer as JSON (fmt) or text. Raises with the CLI's own message on failure."""
    r = subprocess.run([GLOBUS, *args, *(["--format", "json"] if fmt else [])], input=data, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", creationflags=NOWIN)
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip())
    return json.loads(r.stdout) if fmt else r.stdout


def wsl(script, env=None, check=True):
    """A bash script as the WSL user `globus`. `env` names are passed through WSLENV, so values stay off the
    command line."""
    e = dict(os.environ)
    if env:
        e.update(env)
        e["WSLENV"] = ":".join([*(n + "/u" for n in env), *([os.environ["WSLENV"]] if os.environ.get("WSLENV") else [])])
    r = subprocess.run(["wsl.exe", "-d", DISTRO, "-u", WSL_USER, "--", "bash", "-c", script], env=e,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NOWIN)
    if check and r.returncode:
        raise RuntimeError("wsl: %s" % (r.stderr or r.stdout).strip()[-500:])
    return r


def identity():
    """The signed-in Globus identity's id, or None before the owner has signed in."""
    try:
        me = globus("whoami")
    except RuntimeError:
        return None
    return me.get("sub") or me.get("id")


def sign_in():
    """`globus login`: a browser page where the owner signs in and allows the CLI. Waits for it."""
    if identity():
        return identity()
    log("waiting for the owner's Globus sign-in at the page in %s" % SIGN_IN)
    try:
        subprocess.run([GLOBUS_PY, "-c", AUTH, "login"], creationflags=NOWIN, capture_output=True, timeout=WATCH_S)
    except subprocess.TimeoutExpired:
        pass
    who = identity()
    if not who:
        raise RuntimeError("the Globus sign-in did not finish")
    log("signed in to Globus")
    return who


def consent(err):
    """A transfer refused for want of a consent (Harvard's store asking for its own): the CLI names the command;
    run it, which opens the browser page once more, and wait."""
    m = re.search(r"globus session consent ((?:'[^']+'\s*)+)", err)
    if not m:
        return False
    scopes = re.findall(r"'([^']+)'", m.group(1))
    log("Globus asks the owner to allow access to the source store, at the page in %s" % SIGN_IN)
    try:
        subprocess.run([GLOBUS_PY, "-c", AUTH, "session", "consent", *scopes], creationflags=NOWIN,
                       capture_output=True, timeout=WATCH_S)
    except subprocess.TimeoutExpired:
        pass
    return True


def set_dir(name):
    """Where a set's archive folder really is: its junction's (or, on a network share, its link's) target."""
    p = os.path.join(fp.ARCHIVE, name)
    return os.readlink(p).replace("\\\\?\\", "") if fetch.is_link(p) else p


def wsl_path(path):
    """A Windows path as WSL sees it: D:/x is /mnt/d/x (a network drive is mounted there by mount_shares)."""
    drive, rest = os.path.splitdrive(path)
    return "/mnt/%s%s" % (drive[0].lower(), rest.replace("\\", "/"))


def cache_roots():
    """The cold roots whose drive is here now (a network drive may be unmapped, or its server off)."""
    return [r for r in fp.CACHE_ROOTS if os.path.isdir(os.path.splitdrive(r)[0] + "/")]


def mount_shares(roots):
    """Each root's drive mounted in WSL, writable by the GCP user: WSL mounts by itself only the fixed drives there
    when it started, so neither a network drive (the Plex server's O:) nor a disk plugged in since (the spare G:),
    and a mount lasts until WSL shuts down. A drive already mounted is left as it is."""
    for d in sorted({os.path.splitdrive(r)[0].upper() for r in roots}):
        m = "/mnt/" + d[0].lower()
        r2 = subprocess.run(["wsl.exe", "-d", DISTRO, "-u", "root", "--", "sh", "-c",
                             "{ mountpoint -q %s || { mkdir -p %s && mount -t drvfs %s %s -o uid=$(id -u %s),gid=$(id -g %s); }; }"
                             " && su -s /bin/sh %s -c 'test -w %s'" % (m, m, d, m, WSL_USER, WSL_USER, WSL_USER, m)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NOWIN)
        if r2.returncode:
            raise RuntimeError("%s is not mounted in WSL, or %s cannot write it: %s" % (
                d, WSL_USER, (r2.stderr or r2.stdout).strip()[-300:]))


def endpoint(st):
    """This PC's Globus Connect Personal endpoint: registered, set up in WSL, started and connected."""
    # GCP 3.3 writes its client id and key at setup; with them gone, the registration cannot be used again
    configured = wsl("test -s ~/.globusonline/lta/client-id.txt -a -s ~/.globusonline/lta/client-key.pem",
                     check=False).returncode == 0
    if st.get("gcp_id") and not configured:
        # registered, but the setup in WSL never finished: its one-time key is gone, so start again
        globus("endpoint", "delete", st.pop("gcp_id"), fmt=False)
        save_state(st)
    if not st.get("gcp_id"):
        made = globus("gcp", "create", "mapped", GCP_NAME)
        try:
            wsl('"%s" -setup --setup-key "$GCP_SETUP_KEY"' % GCP, env={"GCP_SETUP_KEY": made["globus_connect_setup_key"]})
        except RuntimeError:
            globus("endpoint", "delete", made["id"], fmt=False)
            raise
        st["gcp_id"] = made["id"]
        save_state(st)
        log("this PC is Globus endpoint %r (%s)" % (GCP_NAME, st["gcp_id"]))
    roots = cache_roots()
    paths = ",".join("rw" + wsl_path(r) for r in roots)    # GCP writes only inside these
    connected = globus("endpoint", "show", st["gcp_id"]).get("gcp_connected")
    if connected and st.get("gcp_paths") != paths:
        # started with other paths (a cold disk added or gone since): it would refuse a transfer to the new ones
        wsl('"%s" -stop' % GCP, check=False)
        for _ in range(24):
            time.sleep(5)
            if not globus("endpoint", "show", st["gcp_id"]).get("gcp_connected"):
                break
        log("Globus Connect Personal stopped, to start again with the cold disks %s" % ", ".join(roots))
        connected = False
    if not connected:
        # Started detached from this script's console; it runs until `-stop`, WSL shuts down, or this watcher ends
        # under a job manager (whose job takes its whole tree down with it): a restart resets every transfer's
        # connection, and Globus resumes each from its last restart marker
        out = open(os.path.join(os.path.dirname(STATE), "gcp.log"), "ab")
        mount_shares(roots)
        for r in roots:                                # GCP refuses a restricted path that is not there
            os.makedirs(r, exist_ok=True)
        subprocess.Popen(["wsl.exe", "-d", DISTRO, "-u", WSL_USER, "--", "bash", "-c",
                          'exec "%s" -start -restrict-paths "%s"' % (GCP, paths)],
                         stdout=out, stderr=subprocess.STDOUT, creationflags=NOWIN)
        for _ in range(60):
            time.sleep(5)
            if globus("endpoint", "show", st["gcp_id"]).get("gcp_connected"):
                break
        else:
            raise RuntimeError("Globus Connect Personal did not connect in 5 minutes (see %s)" % out.name)
        st["gcp_paths"] = paths
        save_state(st)
        log("Globus Connect Personal is connected, writing only inside %s" % paths)
    return st["gcp_id"]


def dataverse_files(s):
    """The set's Globus-held files on its Dataverse record (the ones its `only` pattern takes)."""
    with urllib.request.urlopen(urllib.request.Request(
            "%s/api/datasets/:persistentId/?persistentId=%s" % (s["server"], s["pid"]),
            headers={"User-Agent": fetch.UA}), timeout=60) as r:
        v = json.loads(r.read())["data"]["latestVersion"]
    out = []
    for f in v["files"]:
        df = f["dataFile"]
        if re.search(s.get("only") or "", df["filename"]) and (df.get("storageIdentifier") or "").startswith("globus"):
            if not re.fullmatch(r"[\w][\w.-]*", df["filename"]) or f.get("directoryLabel"):
                # it becomes a batch line and a path under the set's folder: plain names only
                raise RuntimeError("%s: file name %r is not a plain name" % (s["name"], df["filename"]))
            ck = df.get("checksum") or {}
            out.append({"id": df["id"], "name": df["filename"], "size": int(df["filesize"]),
                        "checksum": "%s:%s" % (ck.get("type", "").lower().replace("-", ""), ck.get("value", "").lower())
                        if ck.get("value") else None})
    return v, out


def grant(s, who, files):
    """Dataverse's requestGlobusDownload: read access for the owner's identity; returns {file id: (endpoint, path)}."""
    body = json.dumps({"principal": who, "fileIds": [f["id"] for f in files], "guestbookResponse": fp.GUESTBOOK}).encode()
    req = urllib.request.Request("%s/api/datasets/:persistentId/requestGlobusDownload?persistentId=%s" % (
        s["server"], s["pid"]), data=body, method="POST",
        headers={"User-Agent": fetch.UA, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read()).get("data")
    log("%s: Dataverse granted read access; its answer has keys %s" % (s["name"], sorted(data)[:10] if isinstance(data, dict) else type(data).__name__))
    where = {}
    for f in files:
        v = data.get(str(f["id"])) if isinstance(data, dict) else None
        m = re.match(r"(%s)(?::|/)?(/[\w./-]+)$" % UUID, v or "")
        if not m or "/../" in m.group(2) + "/":
            raise RuntimeError("%s: no endpoint and path for %s in Dataverse's answer: %s" % (s["name"], f["name"], json.dumps(data)[:600]))
        where[f["id"]] = (m.group(1), m.group(2))
    return where


def submit(s, gcp, where, files, sync=None):
    """A Globus transfer of the set's files. With sync="checksum" Globus compares each file's checksum on both
    ends and copies only a file that differs (the check after an HTTPS fetch)."""
    srcs = {where[f["id"]][0] for f in files}
    if len(srcs) != 1:
        raise RuntimeError("%s: files on %d source endpoints" % (s["name"], len(srcs)))
    src = srcs.pop()
    # as long as this script watches (a tape-held file can wait hours to be staged), not Globus's default of two days
    deadline = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=WATCH_S)).strftime(
        "%Y-%m-%dT%H:%M:%S")
    batch = "".join('"%s" "%s/%s"\n' % (where[f["id"]][1], wsl_path(set_dir(s["name"])), f["name"]) for f in files)
    for attempt in range(2):
        try:
            t = globus("transfer", src, gcp, "--batch", "-", "--label", "rAidiology %s" % s["name"],
                       "--verify-checksum", "--notify", "off", "--deadline", deadline,
                       *(["--sync-level", sync] if sync else []), data=batch)
            return t["task_id"]
        except RuntimeError as x:
            if attempt or not consent(str(x)):
                raise
    raise RuntimeError("unreachable")


class HttpsAuth:
    """The Authorization header for a Globus collection's HTTPS server, from the CLI's stored consent (its token is
    refreshed through the CLI and never printed). Asked again every 30 minutes."""

    def __init__(self, collection):
        self.collection, self.header, self.at = collection, None, 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            if self.header is None or time.time() - self.at > 1800:
                r = subprocess.run([GLOBUS_PY, "-c", HTTPS_TOKEN, self.collection], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", creationflags=NOWIN)
                if r.returncode or not r.stdout.startswith("Bearer "):
                    raise fetch.FetchError("no Globus HTTPS consent for collection %s" % self.collection)
                self.header, self.at = r.stdout.strip(), time.time()
            return self.header


def https_auth(collection):
    """HttpsAuth for the collection, once the owner has allowed HTTPS reads of it (the page in SIGN_IN)."""
    auth = HttpsAuth(collection)
    try:
        auth()
        return auth
    except fetch.FetchError:
        pass
    log("waiting for the owner to allow HTTPS reads of collection %s, at the page in %s" % (collection, SIGN_IN))
    try:
        subprocess.run([GLOBUS_PY, "-c", AUTH, "session", "consent", "https://auth.globus.org/scopes/%s/https" % collection],
                       creationflags=NOWIN, capture_output=True, timeout=WATCH_S)
    except subprocess.TimeoutExpired:
        pass
    auth()                                  # raises when the consent never came
    log("HTTPS reads of collection %s allowed" % collection)
    return auth


def https_route(s, files, where, gcp, st, live, lock, v):
    """The set's files over the source collection's HTTPS server, in byte ranges on several connections (a Globus
    transfer to this PC is one stream: its endpoint cannot take incoming connections). Its single-stream transfer
    is cancelled only once HTTPS reads are allowed. Then a Globus transfer with sync-level checksum compares each
    file on both ends and copies only one that differs, and the watch records the set as for any transfer."""
    name = s["name"]
    src = {where[f["id"]][0] for f in files}.pop()
    auth = https_auth(src)
    # the Transfer API's own record: `globus endpoint show` refuses a GCSv5 collection
    server = globus("api", "transfer", "GET", "/v0.10/endpoint/" + src).get("https_server")
    if not server:
        raise RuntimeError("%s: collection %s has no HTTPS server" % (name, src))
    def entry(f):
        url = server.rstrip("/") + urllib.parse.quote(where[f["id"]][1])
        return fetch.Entry(f["name"], f["size"], url, None, meta={"auth": auth})

    with lock:
        old = live.pop(name, None)
    if old:
        globus("task", "cancel", old[3], fmt=False)
        log("%s: Globus transfer %s cancelled for the HTTPS route" % (name, old[3]))
        time.sleep(30)                                  # GCP lets go of the file it was writing
    for f in files:                                     # a Globus stream wrote in order: the ranges it holds stay
        dst = os.path.join(fp.ARCHIVE, name, f["name"])
        if os.path.exists(dst) and os.path.getsize(dst) < f["size"] and os.path.exists(dst + ".part"):
            os.remove(dst)
        elif not os.path.exists(dst + ".part") and not (os.path.exists(dst) and os.path.getsize(dst) == f["size"]):
            # the part holds its whole size on the drive from the start, so the other writers' room checks (which
            # keep their claims to themselves) see it taken, and no drive is filled past what it keeps free
            kept = fetch.adopt(dst, entry(f), dst + ".part", sparse=False)
            if kept:
                log("%s/%s: %.1f GB of the Globus copy kept as its first ranges" % (name, f["name"], kept / 1e9))
    ctx = fetch.SetContext(name)
    t0 = time.time()
    for f in files:
        dst = os.path.join(fp.ARCHIVE, name, f["name"])
        if os.path.exists(dst) and os.path.getsize(dst) == f["size"] and not os.path.exists(dst + ".part"):
            continue
        e = entry(f)
        # first on the slow disk (the owner, 2026-10-07: "the mammograms need to go faster"): with eight jobs on it
        # this download wrote 3-5 MB/s of the 22 MB/s Harvard sent; with the disk to itself, 29 MB/s
        with fetch.Turn([dst], "%s/%s download" % (name, f["name"]), 0):
            log("%s/%s: %.1f GB over HTTPS in byte ranges" % (name, f["name"], f["size"] / 1e9))
            have = fetch._chunked(e, dst + ".part", ctx, "%s/%s" % (name, f["name"]))
        if have != f["size"]:
            raise RuntimeError("%s/%s: %d bytes, Dataverse says %d" % (name, f["name"], have, f["size"]))
        os.replace(dst + ".part", dst)
        fetch._discard(dst + ".chunks")
    if unsummed(files):
        log("%s: HTTPS fetch done in %.0f min; Dataverse has no checksum for these tars, so this PC checks them as it "
            "unpacks them" % (name, (time.time() - t0) / 60))
        record(s, v, files, None, https=True)
        steps(name)
        with lock:
            live.pop(name, None)
        return
    log("%s: HTTPS fetch done in %.0f min; Globus now compares checksums on both ends" % (name, (time.time() - t0) / 60))
    task = submit(s, gcp, where, files, sync="checksum")
    with lock:
        st.setdefault("tasks", {})[name] = task
        st.setdefault("https", {})[name] = task
        save_state(st)
        live[name] = (s, v, files, task)
    log("%s: Globus checksum comparison %s submitted" % (name, task))


def https_thread(*args):
    """https_route, its failure logged: the set then needs a rerun (without --https, Globus alone fetches it)."""
    try:
        https_route(*args)
    except Exception as x:  # noqa: BLE001
        log("%s: the HTTPS route stopped: %s: %s; rerun to resume it, or drop --https to let Globus fetch it" % (
            args[0]["name"], type(x).__name__, x))


UNPACK_CHECK = ("every tar header, the CRC-32 of every member of every .npz file before that file is written, and "
                "that the archive ends in its zeros (core.fetch's unpack)")


def unsummed(files):
    """Every file a tar Dataverse gives no checksum for (Harvard's Globus-held files read "not available in
    dataverse"): such a set is checked on this PC as it is unpacked, not by a Globus comparison."""
    return bool(files) and all(f["name"].endswith(".tar") and (not f["checksum"] or "not available" in f["checksum"])
                               for f in files)


def whole(s, f):
    """The file is all on disk: its full size, and no part file left."""
    dst = os.path.join(fp.ARCHIVE, s["name"], f["name"])
    return os.path.exists(dst) and os.path.getsize(dst) == f["size"] and not os.path.exists(dst + ".part")


def steps(name):
    """fetch_public.py's unpack, shrink and record for a set just recorded, logged to a steps file of its own. The
    first that fails stops the rest, and says so here: the unpack is the check of a tar Dataverse gives no checksum
    for, so its failure is this log's to tell (resume_all.ps1's steps chain tries it again after a restart)."""
    steps_log = os.path.join(os.path.dirname(fp.ARCHIVE), "logs", "steps-%s-%s.log" % (
        datetime.date.today().isoformat(), name))
    for step in ("unpack", "shrink", "record"):
        r = subprocess.run([PY, os.path.join(REPO, "tools", "fetch_public.py"), step, name, "--log", steps_log],
                           cwd=REPO, creationflags=NOWIN)
        if r.returncode:
            log("%s: %s FAILED (exit %d: a fault, or another run has the set); the rest not run; see %s" % (
                name, step, r.returncode, steps_log))
            return
    log("%s: recorded, unpacked and checked, shrunk (see %s)" % (name, steps_log))


def record(s, v, files, task, https=False):
    """The set's fetch.json, as the fetcher writes it for a Dataverse download. `https`: the bytes came over the
    collection's HTTPS server and `task` is the Globus checksum comparison that followed, or None when this PC
    checks the tars itself as it unpacks them (core.fetch's unpack)."""
    rec = fp.Record(s["name"])
    t = now()
    task = task or {}
    if https and not task:
        how = ("held in the archive's Globus store; fetched over the source collection's HTTPS server in byte ranges, "
               "to be checked on this PC as it is unpacked, in one read (each file's `checked` says once it is): "
               "%s; a fault stops the unpack. Dataverse publishes no checksum for these files, and Globus's own "
               "comparison reads this PC's copy through WSL too slowly to finish" % UNPACK_CHECK)
    elif https:
        how = ("held in the archive's Globus store; fetched over the source collection's HTTPS server in byte "
               "ranges, then compared by a Globus transfer with sync-level checksum (task %s), which copied %s of "
               "%d files again" % (task["task_id"], task.get("files_transferred"), len(files)))
    else:
        how = ("held in the archive's Globus store; moved by a Globus transfer (task %s) to this PC's Globus Connect "
               "Personal" % task["task_id"])

    def checked(f):
        if https and not task:
            return "to be checked on this PC as it is unpacked: " + UNPACK_CHECK
        return ("Globus compared its checksum on both ends (sync-level checksum)" if https
                else "Globus's end-to-end checksum check (verify_checksum on the transfer)")
    lic = (v.get("license") or {}).get("name") or v.get("termsOfUse") or ""
    rec.data.update({
        "name": s["name"], "archive": "dataverse", "link": s["link"], "ids": None,
        "licence_allowed": sorted(s["licence"]), "licence_seen": lic,
        "licence_ids": [fetch.licence_id(lic)],
        "licence_read_at": "%s/api/datasets/:persistentId/?persistentId=%s (version %s.%s)" % (
            s["server"], s["pid"], v.get("versionNumber"), v.get("versionMinorNumber")),
        "licence_read": t,
        "notes": [how,
                  "read access granted to the owner's Globus identity through Dataverse's requestGlobusDownload, "
                  "guestbook %s answered" % fp.GUESTBOOK],
        "skipped": [], "status": "done", "error": None, "unpacked": None, "removed": None, "archive_copy": None,
        "started": task.get("request_time") or t, "finished": task.get("completion_time") or t, "errors": None,
        "files": {f["name"]: {"size": f["size"], "checksum": f["checksum"], "checked": checked(f),
                              "mtime": int(os.path.getmtime(os.path.join(fp.ARCHIVE, s["name"], f["name"]))),
                              # fetch_public's unpack_set makes this the `checked` once the unpack has gone clean
                              **({"checked_when_unpacked": "checked on this PC as it was unpacked: " + UNPACK_CHECK}
                                 if https and not task else {})}
                  for f in files},
        "bytes": sum(f["size"] for f in files)})
    rec.save(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--log")
    ap.add_argument("--https", nargs="*", default=[], metavar="SET",
                    help="sets fetched over the source collection's HTTPS server in parallel byte ranges (once the owner "
                         "has allowed HTTPS reads of it), then checked by a Globus checksum comparison")
    a = ap.parse_args()
    if a.log:
        sys.stdout = sys.stderr = open(a.log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sets = {s["name"]: s for s in fp.SETS}
    todo = [sets[n] for n in a.sets]
    st = load_state()
    who = sign_in()
    gcp = endpoint(st)
    live, lock, threads = {}, threading.Lock(), []
    for s in todo:
        if fp.Record(s["name"]).data.get("status") == "done":
            log("%s: already done" % s["name"])
            continue
        v, files = dataverse_files(s)
        task = (st.get("tasks") or {}).get(s["name"])
        try:                               # the record's licence, read now, must be one the set's row allows
            fetch.check_licence(s["name"], (v.get("license") or {}).get("name") or v.get("termsOfUse") or "", s["licence"])
        except fetch.LicenceChanged as x:
            log(str(x))
            if task:
                globus("task", "cancel", task, fmt=False)
                log("%s: Globus transfer %s cancelled" % (s["name"], task))
            continue
        if not files:
            log("%s: no Globus-held file to fetch" % s["name"])
            continue
        fp.place(s)                        # the cold junction first, so the transfer lands on the cold disk
        fp.unplace(s["name"])              # this process claims nothing else
        if task:
            t = globus("task", "show", task)
            if t["status"] != "FAILED" and t["destination_endpoint_id"] != gcp:
                # sent to an earlier registration of this PC, which is gone, so it can never finish
                globus("task", "cancel", task, fmt=False)
                log("%s: Globus transfer %s cancelled: it was sent to an earlier registration of this PC" % (
                    s["name"], task))
            if t["status"] == "FAILED" or t["destination_endpoint_id"] != gcp:
                task = None
        if (task and s["name"] in a.https and (st.get("https") or {}).get(s["name"]) == task and unsummed(files)
                and all(whole(s, f) for f in files)):
            # a comparison submitted before this PC checked such tars itself: the HTTPS route below checks them
            globus("task", "cancel", task, fmt=False)
            log("%s: Globus checksum comparison %s cancelled: Dataverse has no checksum for these tars, so this PC "
                "checks them itself" % (s["name"], task))
            task = None
        if not task and s["name"] not in a.https:   # an HTTPS set goes straight to its byte ranges
            where = grant(s, who, files)
            task = submit(s, gcp, where, files)
            st.setdefault("tasks", {})[s["name"]] = task
            save_state(st)
            log("%s: Globus transfer %s submitted: %d files, %.1f GB" % (
                s["name"], task, len(files), sum(f["size"] for f in files) / 1e9))
        if task:
            live[s["name"]] = (s, v, files, task)
        if s["name"] in a.https and (not task or (st.get("https") or {}).get(s["name"]) != task):
            th = threading.Thread(target=https_thread, args=(s, files, grant(s, who, files), gcp, st, live, lock, v),
                                  daemon=True)
            th.start()
            threads.append(th)
    end = time.time() + WATCH_S
    while live or any(th.is_alive() for th in threads):
        with lock:
            watched = list(live.items())
        for name, (s, v, files, task) in watched:
            t = globus("task", "show", task)
            log("%s: Globus %s, %.1f of %.1f GB, %s" % (name, t["status"], (t.get("bytes_transferred") or 0) / 1e9,
                sum(f["size"] for f in files) / 1e9, t.get("nice_status") or ""))
            if t["status"] == "INACTIVE" or t.get("is_paused"):
                # Globus holds the task until someone acts (an expired sign-in or consent, or a pause); it goes on
                # by itself once that is fixed, so keep watching, and say so plainly in the log
                log("%s: NEEDS THE OWNER: Globus has stopped this transfer until a sign-in or consent is renewed "
                    "(%s); `globus login` again, then it resumes" % (name, t.get("nice_status") or "paused"))
            with lock:
                if live.get(name, (None,) * 4)[3] != task:
                    continue                       # the HTTPS route took this set over meanwhile
            if t["status"] == "SUCCEEDED":
                record(s, v, files, t, https=(st.get("https") or {}).get(name) == task)
                steps(name)
                with lock:
                    live.pop(name, None)
            elif t["status"] == "FAILED":
                log("%s: the Globus transfer failed: %s" % (name, t.get("nice_status") or t.get("fatal_error")))
                with lock:
                    live.pop(name, None)
        if (live or any(th.is_alive() for th in threads)) and time.time() > end:
            log("stopped watching after %d days with %s unfinished; the Globus tasks are left running (state in %s)"
                % (WATCH_S // 86400, sorted(live), STATE))
            return
        if live or any(th.is_alive() for th in threads):
            time.sleep(POLL_S)
    wsl('"%s" -stop' % GCP, check=False)
    log("done; Globus Connect Personal stopped")


if __name__ == "__main__":
    main()
