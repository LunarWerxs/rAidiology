"""Every run that produces a number or takes time writes it here, so "is it getting better, and is
it getting faster" is a query instead of an argument.

Two copies of the same thing, on purpose:
- `ledger/<kind>.jsonl` in the repository, append-only, one line per run, committed. It is the
  durable record: it survives a wiped work folder, it diffs in a pull request, and two machines
  merge it by appending.
- `work/ledger.db`, a SQLite mirror rebuilt from those files on demand, so a question like "every
  AUROC the detector has ever scored on the held-out test" is one SELECT.

A run is a kind (`prepare`, `baselines`, `detector`, `segmenter`, `score`, `final_eval`, `check`,
`visitors`, `experiment`), a name inside that kind, the seconds it took, the git commit it STARTED
at, and any number of named numbers. Use it as a context manager:

    with ledger.run("detector", "train fold 2") as r:
        r.settings_from({"lr": 1e-3, "batch": 16})   # what it was run with
        ...
        r.point("loss", step, loss)                  # a curve, one point at a time
        r.number("auroc", 0.84)
        r.number("epochs", 12, "count")

What each part is for:
- **settings** say what a run was run with. Two runs are only compared when their settings match,
  so a smoke test on 24 scans is never reported as "faster" than the real run on 1,139.
- **points** keep the curve, not just where it ended, so "did it plateau or was it stopped early" has
  an answer after the log is gone.
- **a run that is killed still leaves a record.** While a run is open its numbers so far sit in a
  small file under the work folder; the next ledger command finds the ones whose process is gone
  and writes them in as failed. A run that was killed for stalling is a result, and the plain
  context manager lost it.
- **`each.<item>`** numbers are per-item scores (one per structure, one per dataset). `next` lists
  the weakest of them. Record them on development data only: choosing what to fix from held-out
  test scores is training on the test.
- **the card's queue is not run time.** `seconds` is wall time, as ever, but a run that waited for
  the shared graphics card (core/card.py lease/hold) also carries a number `card_wait_s`, and the
  slow flags, the speed comparisons and the slowest/next lists judge `seconds - card_wait_s`.
  2026-10-05: modalities/spine_xray/eval/landmarks_test.py reported 5,421 s and "OVER THE ONE-HOUR
  LIMIT"; about 4,750 s of it was waiting for the lease, and the reading took 11 minutes.
- **experiments** (`record experiment "<what was tried>" -s verdict=kept ...`) are the things that
  were tried and what happened, so the next session reads them before trying one again.

`next` is the point of the whole thing: what to attack, from the record and not from a feeling.

Usage: python core/ledger.py next
       python core/ledger.py status
       python core/ledger.py report [--kind K] [--limit N]
       python core/ledger.py best
       python core/ledger.py slowest [--limit N]
       python core/ledger.py history <kind> [name] [--key K]
       python core/ledger.py curve <kind> <name> [series]
       python core/ledger.py tried
       python core/ledger.py record <kind> <name> [-n key=value[:unit]] [-s key=value] [--note N]
                                    [--seconds S] [--at ISO] [--failed]
       python core/ledger.py time <kind> <name> -- <command ...>
       python core/ledger.py query "<SQL>"
       python core/ledger_selftest.py              (its promises, checked in a throwaway folder)
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402

DIR = os.environ.get("RAIDIOLOGY_LEDGER", os.path.join(ROOT, "ledger"))
# Open runs live outside the repository: they are this machine's state, not a record yet.
OPEN = (os.path.join(DIR, ".open") if "RAIDIOLOGY_LEDGER" in os.environ
        else os.path.join(paths.WORK, "ledger_open"))
DB = os.path.join(paths.WORK, "ledger.db")
BUDGET_S = 600.0            # the target for any one run: ten minutes
CARD_WAIT = "card_wait_s"   # the number a record carries for seconds spent waiting for the card
LIMIT_S = 3600.0           # the hard ceiling: an hour is a defect to fix that session
BEAT_S = 20.0               # how often an open run saves what it has so far
NO_RECORD = "no record: the process was killed or crashed"
# A run that handled a different amount of work is not the same run. These are the numbers that say
# how much, and two runs are compared only when the ones they share agree to within 2%.
SIZE_KEYS = ("scans", "images", "rows", "steps", "pairs", "train_scans", "train_images", "checks",
             "files", "visitors")
# Runs recorded before settings existed wrote them into the note ("jobs=8; first build").
SETTING_IN_NOTE = re.compile(r"(?<![\w.])([a-z_]+)=([^\s;,]+)")
HIGHER = ("dice", "auroc", "accuracy", "kappa", "sensitivity", "specificity", "per_second",
          "speedup", "stars", "trust", "easy", "mb_s", "agree", "minus_reference", "would_come_back",
          "each.", "critic", "where_expected", "confidence", "verdict")
# A visitor's `moments.*` stay plain counts: they grow with how long somebody stays, good or bad.
LOWER = ("invention", "error", "warning", "crash", "unreadable", "loss", "memory", "waiting",
         "never_coming_back", "trash", "problems", "retried")
TIME_KEY = re.compile(r"(^|[._])(seconds|s|ms)([._]|$)")
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY, at TEXT, kind TEXT, name TEXT, commit_sha TEXT, dirty INTEGER,
  seconds REAL, host TEXT, python TEXT, note TEXT, ok INTEGER, settings_id TEXT, ended TEXT
);
CREATE TABLE IF NOT EXISTS numbers (
  run_id INTEGER, key TEXT, value REAL, unit TEXT
);
CREATE TABLE IF NOT EXISTS settings (
  run_id INTEGER, key TEXT, value TEXT
);
CREATE TABLE IF NOT EXISTS points (
  run_id INTEGER, series TEXT, x REAL, y REAL
);
CREATE TABLE IF NOT EXISTS env (
  run_id INTEGER, key TEXT, value TEXT
);
CREATE INDEX IF NOT EXISTS runs_kind ON runs (kind, name, at);
CREATE INDEX IF NOT EXISTS numbers_key ON numbers (key);
CREATE INDEX IF NOT EXISTS points_series ON points (run_id, series);
"""


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def _commit():
    return _git("rev-parse", "--short", "HEAD")


def _dirty():
    """How many tracked files differ from the commit: a number from a dirty tree is not that commit's."""
    return len([line for line in _git("status", "--porcelain", "--untracked-files=no").splitlines()
                if line.strip()])


UNKNOWN_BIRTH = -1        # the process is there, but this user may not ask when it started


def _born(pid):
    """When process `pid` started, UNKNOWN_BIRTH when it runs but may not be asked, None when gone.

    Never `os.kill(pid, 0)`: on Windows that is not a question, it ends the process. The start time
    is kept beside the pid because Windows hands a dead process's pid to the next one that asks.
    Liveness is a zero-wait on the process handle, not its exit code: a process may exit with 259,
    the code that otherwise means "still running"."""
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return 0
        except PermissionError:
            return UNKNOWN_BIRTH
        except OSError:
            return None
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
    h = k.OpenProcess(0x1000 | 0x00100000, False, int(pid))
    if not h:
        return UNKNOWN_BIRTH if ctypes.get_last_error() == 5 else None   # 5: access denied
    try:
        if k.WaitForSingleObject(h, 0) != 0x102:       # WAIT_TIMEOUT: it has not ended
            return None
        made, gone, kern, user = (wintypes.FILETIME() for _ in range(4))
        if not k.GetProcessTimes(h, made, gone, kern, user):
            return UNKNOWN_BIRTH
        return (made.dwHighDateTime << 32) | made.dwLowDateTime
    finally:
        k.CloseHandle(h)


def shown(path):
    """A path as a record shows it: relative to the current folder, or absolute when it is on another drive
    (Windows has no relative path between two drives, and os.path.relpath raises there)."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return os.path.abspath(path).replace(os.sep, "/")


def settings_id(settings):
    if not settings:
        return ""
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:8]


class Run:
    def __init__(self, kind, name, note=""):
        self.kind, self.name, self.note = kind, name, note
        self.numbers, self.settings, self.series = {}, {}, {}
        self.ok, self.t0, self.commit, self.dirty = 1, None, "", 0
        self._marker, self._born, self._closed, self._wait0 = None, None, False, 0.0
        self._lock, self._stop = threading.Lock(), threading.Event()

    def number(self, key, value, unit=""):
        """Record one number. Called twice with the same key, the last one wins."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self               # an undefined metric (n/a) is simply not a number
        if math.isfinite(v):
            with self._lock:
                self.numbers[str(key)] = (v, unit)
        return self

    def numbers_from(self, mapping, unit=""):
        for k, v in dict(mapping).items():
            self.number(k, v, unit)
        return self

    def setting(self, key, value):
        """Record one thing the run was run with. Runs are compared only when all of these match."""
        text = (str(value) if isinstance(value, (str, int, float, bool))
                else json.dumps(value, sort_keys=True, default=str))
        with self._lock:
            self.settings[str(key)] = text
        return self

    def settings_from(self, mapping):
        for k, v in dict(mapping).items():
            self.setting(k, v)
        return self

    def point(self, series, x, y):
        """Add one point to a curve (a loss against a step, a score against a size)."""
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return self
        if math.isfinite(x) and math.isfinite(y):
            with self._lock:
                self.series.setdefault(str(series), []).append([x, y])
        return self

    def note_is(self, text):
        """Replace the note, for a caller that only learns it while the run is happening."""
        self.note = str(text)
        return self

    def _save_open(self):
        """What this run has so far, where a later command can find it if this process dies.

        Written under the lock and never after the run has closed, so a heartbeat that lands while
        the run is finishing cannot leave a marker behind that a later sweep would record twice."""
        with self._lock:
            if self._closed or not self._marker:
                return
            rec = {"kind": self.kind, "name": self.name, "note": self.note, "commit": self.commit,
                   "dirty": self.dirty, "host": socket.gethostname(), "pid": os.getpid(),
                   "born": self._born, "t0": self.t0, "beat": time.time(),
                   "numbers": self.numbers, "settings": self.settings, "series": self.series}
            try:
                os.makedirs(OPEN, exist_ok=True)
                with open(self._marker + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(rec, f)
                os.replace(self._marker + ".tmp", self._marker)
            except (OSError, TypeError, ValueError):
                pass                  # a run is never failed by its own bookkeeping

    def _heartbeat(self):
        while not self._stop.wait(max(BEAT_S, 0.05)):
            self._save_open()

    def __enter__(self):
        sweep()
        self.t0 = time.time()
        self._wait0 = card_waited()
        self.commit, self.dirty = _commit(), _dirty()
        self._born = _born(os.getpid())
        self._marker = os.path.join(OPEN, "%d-%d.json" % (os.getpid(), int(self.t0 * 1000)))
        self._save_open()
        # The beat comes from a thread, so a run that records nothing for an hour still says it is
        # alive, and a killed one is recorded with the time it actually ran.
        threading.Thread(target=self._heartbeat, name="ledger-heartbeat", daemon=True).start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ok = 0 if exc_type else 1
        with self._lock:
            self._closed = True
        self._stop.set()
        try:
            write(self.kind, self.name, time.time() - self.t0, self.numbers, self.note, self.ok,
                  settings=self.settings, series=self.series, commit=self.commit, dirty=self.dirty,
                  card_wait=max(0.0, card_waited() - self._wait0))
        except Exception as e:          # the run's own work must not fail on its record
            # The open marker stays, so the next ledger command still writes this run in, as failed.
            print("ledger: could not record %s/%s (%s); it will be swept in as failed"
                  % (self.kind, self.name, e), file=sys.stderr, flush=True)
            return False
        try:
            os.remove(self._marker)
        except OSError:
            pass
        return False


def run(kind, name, note=""):
    return Run(kind, name, note)


# The libraries whose version can move a number or a time. Only ones the process has already
# imported are read, so recording them costs nothing and imports nothing.
ENV_MODULES = ("torch", "torchvision", "numpy", "pandas", "polars", "pyarrow", "scipy", "sklearn",
               "skimage", "nibabel", "pydicom", "PIL", "timm", "numba", "onnxruntime", "openvino")


def env():
    """What this run ran on: library versions, and the card's stack once torch has used it.

    A library upgrade is an experiment like any other, and this is what lets the history show which
    runs came before it and which after."""
    out = {}
    for name in ENV_MODULES:
        version = getattr(sys.modules.get(name), "__version__", None)
        if version:
            out[name] = str(version)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            out["cuda"] = str(torch.version.cuda)
            out["cudnn"] = str(torch.backends.cudnn.version())
            if torch.cuda.is_initialized():
                out["card"] = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:  # noqa: BLE001  a half-broken driver must not lose the record
            pass
    return out


_MARK = 0.0     # the card wait already accounted for by the records this process wrote


def card_waited():
    """Seconds this process has waited for the card, read without importing card (or torch)."""
    mod = sys.modules.get("core.card") or sys.modules.get("card")
    try:
        return float(mod.waited_seconds()) if mod is not None else 0.0
    except Exception:  # noqa: BLE001  a half-imported card module must not lose a record
        return 0.0


def card_wait_of(rec):
    """The seconds a record's run spent waiting for the card (0 for a record that has none)."""
    return value(rec, CARD_WAIT) or 0.0


def work_seconds(rec):
    """A record's time without its wait for the card: what its speed is judged on."""
    return max(0.0, rec["seconds"] - card_wait_of(rec))


def waited_note(rec):
    w = card_wait_of(rec)
    return "  (waited %.0f s for the card)" % w if w >= 1 else ""


def write(kind, name, seconds, numbers=None, note="", ok=1, *, settings=None, series=None, at=None,
          commit=None, dirty=None, ended="", card_wait=None):
    """Append one run to ledger/<kind>.jsonl. Returns the record.

    card_wait is the seconds of `seconds` spent waiting for the card; left None, it is the wait this
    process accrued since its last record, capped at `seconds`."""
    global _MARK
    os.makedirs(DIR, exist_ok=True)
    now_wait = card_waited()
    if ended:
        wait = 0.0                      # a swept run's wait was its own process's, not this one's
    elif card_wait is None:
        wait = max(0.0, now_wait - _MARK)
    else:
        wait = max(0.0, float(card_wait))
    wait = min(wait, max(0.0, float(seconds)))
    if not ended:
        _MARK = now_wait
    rec = {"at": at or time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "name": name,
           "commit": _commit() if commit is None else commit,
           "dirty": _dirty() if dirty is None else int(dirty),
           "seconds": round(float(seconds), 3),
           "host": socket.gethostname(), "python": platform.python_version(),
           "note": note, "ok": int(ok),
           "numbers": {k: {"value": v, "unit": u} for k, (v, u) in (numbers or {}).items()}}
    if wait >= 1:
        rec["numbers"][CARD_WAIT] = {"value": round(wait, 3), "unit": "s"}
    if settings:
        rec["settings"] = dict(settings)
        rec["settings_id"] = settings_id(rec["settings"])
    if series:
        rec["series"] = {k: v for k, v in series.items() if v}
    # A run written in by the sweep after it died is described by the sweeper's libraries, not its
    # own, so it carries none.
    ran_on = {} if ended else env()
    if ran_on:
        rec["env"] = ran_on
    if ended:
        rec["ended"] = ended
    with open(os.path.join(DIR, "%s.jsonl" % kind), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    work = seconds - wait if wait >= 1 else seconds
    flag = "" if work <= BUDGET_S else (
        "  SLOWER THAN THE TEN-MINUTE TARGET" if work <= LIMIT_S else
        "  OVER THE ONE-HOUR LIMIT: fix this, do not wait for it")
    waited = " (%.1f s of it waiting for the card)" % wait if wait >= 1 else ""
    print("ledger %s/%s: %.1f s%s, %d number(s)%s" % (kind, name, seconds, waited, len(rec["numbers"]),
                                                      flag), flush=True)
    return rec


def open_runs():
    """Every run that has started and not finished, with whether its process is still there."""
    out = []
    if not os.path.isdir(OPEN):
        return out
    for n in sorted(os.listdir(OPEN)):
        if not n.endswith(".json"):
            continue
        path = os.path.join(OPEN, n)
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if rec.get("host") != socket.gethostname():
            continue                  # another machine's run: this one cannot say whether it lives
        born = _born(rec.get("pid", -1))
        # A process this user may not question is left alone: it is there, so it is not swept.
        rec["alive"] = born is not None and (born == rec.get("born") or UNKNOWN_BIRTH in
                                             (born, rec.get("born")))
        rec["path"] = path
        out.append(rec)
    return out


def sweep():
    """Write every run whose process is gone into the ledger as failed. Returns those records."""
    done = []
    for rec in open_runs():
        if rec["alive"]:
            continue
        claimed = rec["path"] + ".swept"
        try:
            os.replace(rec["path"], claimed)    # two commands sweeping at once: one of them wins
        except OSError:
            continue
        numbers = {k: tuple(v) for k, v in rec.get("numbers", {}).items()}
        done.append(write(
            rec["kind"], rec["name"], max(0.0, rec.get("beat", 0) - rec.get("t0", 0)), numbers,
            rec.get("note", ""), 0, settings=rec.get("settings"), series=rec.get("series"),
            at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(rec.get("beat") or time.time())),
            commit=rec.get("commit", ""), dirty=rec.get("dirty", 0), ended=NO_RECORD))
        try:
            os.remove(claimed)
        except OSError:
            pass
    return done


def records(kind=None):
    """Every run ever recorded, oldest first, from the JSONL files."""
    if not os.path.isdir(DIR):
        return []
    out = []
    for n in sorted(os.listdir(DIR)):
        if not n.endswith(".jsonl") or (kind and n != "%s.jsonl" % kind):
            continue
        with open(os.path.join(DIR, n), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    out.sort(key=lambda r: r["at"])
    return out


def by_step(rows):
    out = {}
    for r in rows:
        out.setdefault((r["kind"], r["name"]), []).append(r)
    return out


def value(rec, key):
    """One number of a record, or None when it has none or the one it has is not a number."""
    v = rec.get("numbers", {}).get(key, {}).get("value")
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def database(rebuild=True):
    """The SQLite mirror, rebuilt from the JSONL files so the files stay the source of truth."""
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if rebuild and os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    if rebuild:
        for r in records():
            cur = con.execute(
                "INSERT INTO runs (at, kind, name, commit_sha, dirty, seconds, host, python, note, ok,"
                " settings_id, ended) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["at"], r["kind"], r["name"], r.get("commit", ""), r.get("dirty"), r["seconds"],
                 r.get("host", ""), r.get("python", ""), r.get("note", ""), r.get("ok", 1),
                 settings_id(settings_of(r)), r.get("ended", "")))
            rid = cur.lastrowid
            con.executemany("INSERT INTO numbers (run_id, key, value, unit) VALUES (?,?,?,?)",
                            [(rid, k, v["value"], v.get("unit", ""))
                             for k, v in r.get("numbers", {}).items() if value(r, k) is not None])
            con.executemany("INSERT INTO settings (run_id, key, value) VALUES (?,?,?)",
                            [(rid, k, v) for k, v in settings_of(r).items()])
            con.executemany("INSERT INTO points (run_id, series, x, y) VALUES (?,?,?,?)",
                            [(rid, s, x, y) for s, pts in r.get("series", {}).items() for x, y in pts])
            con.executemany("INSERT INTO env (run_id, key, value) VALUES (?,?,?)",
                            [(rid, k, v) for k, v in r.get("env", {}).items()])
        con.commit()
    return con


def direction(key, unit=""):
    """+1 when a bigger number is better, -1 when a smaller one is, 0 when it is only a count."""
    k = key.lower()
    if any(w in k for w in HIGHER):
        return 1
    if unit in ("s", "ms") or TIME_KEY.search(k) or any(w in k for w in LOWER):
        return -1
    return 0


def settings_of(rec):
    """What a run was run with: its settings, or for an older run the `key=value` words of its note."""
    if "settings" in rec:
        return rec["settings"]
    return dict(SETTING_IN_NOTE.findall(rec.get("note", "")))


def measured(key, unit, v):
    """A zero memory or time reading is a figure nobody took, not the best one ever recorded."""
    k = key.lower()
    return not (v == 0 and ("memory" in k or unit in ("s", "ms") or TIME_KEY.search(k)))


def comparable(a, b):
    """Were these two the same run on the same amount of work? Only then is one faster or better."""
    if settings_id(settings_of(a)) != settings_id(settings_of(b)):
        return False
    for k in SIZE_KEYS:
        x, y = value(a, k), value(b, k)
        if x is not None and y is not None and abs(x - y) > 0.02 * max(abs(x), abs(y)):
            return False
    return True


def like_runs(runs):
    """The finished runs of one step that are comparable to its latest, oldest first."""
    last = runs[-1]
    return [r for r in runs if r.get("ok", 1) and comparable(r, last)]


def arrow(now, before, lower_is_better=False):
    if before is None:
        return "first"
    d = now - before
    if abs(d) < 1e-9:
        return "same"
    good = (d < 0) if lower_is_better else (d > 0)
    return "%s%+.4g" % ("better " if good else "worse ", d)


def moved(last, prev):
    """What changed between two comparable runs, each with the word for which way it went."""
    out = []
    for key, cur in sorted(last.get("numbers", {}).items()):
        now, was = value(last, key), value(prev, key) if prev else None
        if now is None or key == CARD_WAIT:      # the wait is a queue's doing, not the run's
            continue
        if was is None:
            out.append("%s=%.4g" % (key, now))
        elif abs(now - was) > 1e-9:
            way = direction(key, cur.get("unit", ""))
            word = "" if not way else (" better" if (now - was) * way > 0 else " WORSE")
            out.append("%s %.4g->%.4g%s" % (key, was, now, word))
    return out


def report(kind=None, limit=40):
    sweep()
    rows = records(kind)
    if not rows:
        print("nothing recorded yet: %s is empty" % DIR)
        return
    steps = by_step(rows)
    print("%-11s %-34s %10s %-16s %s" % ("kind", "name", "seconds", "versus last", "numbers"))
    for (k, n), rs in sorted(steps.items())[:limit]:
        last = rs[-1]
        like = [r for r in like_runs(rs) if r is not last]
        prev = like[-1] if like else None
        if not last.get("ok", 1):
            speed = "FAILED"
        elif prev is None and len(rs) > 1:
            speed = "no like run"     # earlier runs exist, on other settings or another size
        else:
            speed = arrow(work_seconds(last), work_seconds(prev) if prev else None, lower_is_better=True)
        print("%-11s %-34s %10.1f %-16s %s%s" % (k, n[:34], last["seconds"], speed,
                                                 ", ".join(moved(last, prev)[:6]) or "unchanged",
                                                 waited_note(last)))
    print("\n%d run(s) across %d step(s). The target is 600 s a run; over 3600 s is a defect."
          % (len(rows), len(steps)))


def regressions():
    """Every number whose latest value is worse than the best a comparable run ever recorded."""
    out = []
    for (k, n), rs in sorted(by_step(records()).items()):
        last = rs[-1]
        if not last.get("ok", 1):
            continue
        like = like_runs(rs)
        for key, cur in sorted(last.get("numbers", {}).items()):
            unit = cur.get("unit", "")
            way, now = direction(key, unit), value(last, key)
            had = [(value(r, key), r["at"]) for r in like
                   if value(r, key) is not None and measured(key, unit, value(r, key))]
            if not way or now is None or key == CARD_WAIT or not measured(key, unit, now) or len(had) < 2:
                continue
            best, when = max(had, key=lambda p: p[0] * way)
            if (best - now) * way > 0.01 * max(abs(best), 1e-9):
                out.append((k, n, key, now, best, when))
    return out


def best():
    sweep()
    found = regressions()
    if not found:
        print("Nothing is worse than its own best: every number that has a direction is at, or within "
              "1% of, the best a comparable run recorded.")
        return
    print("Worse than its own best (same settings, same amount of work):")
    for k, n, key, now, top, when in found:
        print("  %-11s %-30s %-28s now %.4g, best %.4g on %s" % (k, n[:30], key[:28], now, top, when))


def slowest(limit=15):
    latest = {}
    for r in records():
        latest[(r["kind"], r["name"])] = r
    worst = sorted(latest.values(), key=lambda r: -work_seconds(r))[:limit]
    if not worst:
        print("nothing recorded yet")
        return
    print("What to make faster next, slowest first:")
    for r in worst:
        work = work_seconds(r)
        state = "ok" if work <= BUDGET_S else ("over the target" if work <= LIMIT_S
                                               else "OVER THE HOUR LIMIT")
        print("  %8.1f s  %-11s %-34s %s  (%s)%s" % (work, r["kind"], r["name"][:34], state,
                                                     r["at"], waited_note(r)))


def ago(seconds):
    return "%.0f s" % seconds if seconds < 120 else ("%.0f min" % (seconds / 60) if seconds < 7200
                                                     else "%.1f h" % (seconds / 3600))


def status():
    died = sweep()
    now = time.time()
    live = [r for r in open_runs() if r["alive"]]
    print("Running now:" if live else "Nothing is running.")
    for r in live:
        print("  %-11s %-34s for %s, last heard from %s ago, %d number(s) and %d curve point(s) so far"
              % (r["kind"], r["name"][:34], ago(now - r["t0"]), ago(now - r["beat"]),
                 len(r.get("numbers", {})), sum(len(v) for v in r.get("series", {}).values())))
    if died:
        print("\nEnded without a record, now written in as failed:")
        for r in died:
            print("  %-11s %-34s after %s, %d number(s) kept" % (r["kind"], r["name"][:34],
                                                                ago(r["seconds"]), len(r["numbers"])))
    rows = records()
    kinds = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    print("\n%-12s %5s %7s  %-19s  %s" % ("kind", "runs", "failed", "last run", "last name"))
    for k, rs in sorted(kinds.items()):
        print("%-12s %5d %7d  %-19s  %s%s" % (k, len(rs), sum(1 for r in rs if not r.get("ok", 1)),
                                               rs[-1]["at"], rs[-1]["name"][:40],
                                               "" if rs[-1].get("ok", 1) else "  (FAILED)"))
    print("\n%d run(s) on record in %s" % (len(rows), DIR))


def weakest(limit=10):
    """The lowest per-item scores of the latest run of every step that recorded any."""
    out = []
    for (k, n), rs in sorted(by_step(records()).items()):
        last = rs[-1]
        each = sorted((value(last, key), key[5:]) for key in last.get("numbers", {})
                      if key.startswith("each.") and value(last, key) is not None)
        if each:
            out.append((k, n, last["at"], len(each), each[:limit]))
    return out


def section(title, empty, lines):
    print(title if lines else empty)
    for line in lines:
        print("   " + line)


def failed_lines(steps):
    failed = sorted((rs[-1] for rs in steps.values() if not rs[-1].get("ok", 1)), key=lambda r: r["at"])
    return ["%-11s %-34s %s  %s" % (r["kind"], r["name"][:34], r["at"],
                                    r.get("ended") or r.get("note", "")[:60]) for r in failed]


def slow_lines(steps, limit):
    slow = sorted((rs[-1] for rs in steps.values() if work_seconds(rs[-1]) > BUDGET_S),
                  key=lambda r: -work_seconds(r))
    return ["%8.1f s  %-11s %-34s%s%s" % (work_seconds(r), r["kind"], r["name"][:34],
                                          "  OVER THE HOUR LIMIT" if work_seconds(r) > LIMIT_S else "",
                                          waited_note(r))
            for r in slow[:limit]]


def weak_lines(limit):
    out = []
    for k, n, at, count, each in weakest(limit):
        out.append("%s / %s (%s, %d items):" % (k, n, at, count))
        out.append("  " + ", ".join("%s %.3f" % (item, v) for v, item in each))
    return out


def next_up(limit=8):
    """What to attack, in the order it should be attacked."""
    sweep()
    steps = by_step(records())
    section("1. The last run of these failed or died:", "1. No step's last run failed.",
            failed_lines(steps))
    section("\n2. Worse than its own best:", "\n2. Nothing is worse than its own best.",
            ["%-11s %-30s %-24s now %.4g, best %.4g on %s" % (k, n[:30], key[:24], now, top, when)
             for k, n, key, now, top, when in regressions()[:limit]])
    section("\n3. Over the ten-minute target, slowest first:",
            "\n3. Every step is inside the ten-minute target.", slow_lines(steps, limit))
    section("\n4. The weakest items, development data only:",
            "\n4. No step has recorded per-item scores yet.", weak_lines(limit))


def history(kind, name=None, key=None, limit=60):
    rows = [r for r in records(kind) if not name or name.lower() in r["name"].lower()]
    if not rows:
        print("no run of kind %r%s" % (kind, " matching %r" % name if name else ""))
        return
    for r in rows[-limit:]:
        keys = [key] if key else sorted(r.get("numbers", {}))[:5]
        nums = ", ".join("%s=%.4g" % (k, value(r, k)) for k in keys if value(r, k) is not None)
        print("%s  %-34s %9.1f s  %-6s %-8s %s%s" % (r["at"], r["name"][:34], r["seconds"],
                                                      "ok" if r.get("ok", 1) else "FAILED",
                                                      settings_id(settings_of(r)), nums, waited_note(r)))


def curve(kind, name, series=None, rows=24):
    found = [r for r in records(kind) if name.lower() in r["name"].lower() and r.get("series")]
    if not found:
        print("no run of kind %r matching %r recorded a curve" % (kind, name))
        return
    last = found[-1]
    print("%s / %s, %s" % (last["kind"], last["name"], last["at"]))
    for s, pts in sorted(last["series"].items()):
        if series and s != series:
            continue
        step = max(1, len(pts) // rows)
        shown = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
        print("  %s (%d points): %s" % (s, len(pts), "  ".join("%g:%.4g" % (x, y) for x, y in shown)))


def tried():
    rows = records("experiment")
    if not rows:
        print("no experiment recorded yet")
        return
    for r in rows:
        s = r.get("settings", {})
        nums = ", ".join("%s=%.4g" % (k, value(r, k)) for k in sorted(r.get("numbers", {}))
                         if value(r, k) is not None)
        print("%s  [%s] %s\n    %s\n    %s" % (r["at"][:16], s.get("verdict", "?"), r["name"], nums,
                                               r.get("note", "")))


def record(a):
    numbers, settings = {}, {}
    for item in a.number or []:
        key, _, rest = item.partition("=")
        val, _, unit = rest.partition(":")
        numbers[key] = (float(val), unit)
    for item in a.setting or []:
        key, _, val = item.partition("=")
        settings[key] = val
    write(a.kind, a.name, a.seconds, numbers, a.note, 0 if a.failed else 1, settings=settings,
          at=a.at)


def timed(kind, name, command):
    """Run a command, record how long it took and whether it passed, and exit as it exited."""
    command = command[1:] if command[:1] == ["--"] else command
    if not command:
        raise SystemExit("time needs a command after --")
    with run(kind, name, note=subprocess.list2cmdline(command)[:200]) as r:
        code = subprocess.run(command).returncode
        r.number("exit_code", code, "count")
        if code:
            raise SystemExit(code)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("next")
    sub.add_parser("status")
    sub.add_parser("best")
    sub.add_parser("tried")
    p = sub.add_parser("report"); p.add_argument("--kind"); p.add_argument("--limit", type=int, default=40)
    s = sub.add_parser("slowest"); s.add_argument("--limit", type=int, default=15)
    h = sub.add_parser("history"); h.add_argument("kind"); h.add_argument("name", nargs="?")
    h.add_argument("--key")
    c = sub.add_parser("curve"); c.add_argument("kind"); c.add_argument("name")
    c.add_argument("series", nargs="?")
    w = sub.add_parser("record"); w.add_argument("kind"); w.add_argument("name")
    w.add_argument("-n", "--number", action="append", help="key=value or key=value:unit")
    w.add_argument("-s", "--setting", action="append", help="key=value")
    w.add_argument("--note", default=""); w.add_argument("--seconds", type=float, default=0.0)
    w.add_argument("--at", help="when it happened, if not now: 2026-10-01T19:07:38")
    w.add_argument("--failed", action="store_true")
    t = sub.add_parser("time"); t.add_argument("kind"); t.add_argument("name")
    t.add_argument("command", nargs=argparse.REMAINDER)
    q = sub.add_parser("query"); q.add_argument("sql")
    a = ap.parse_args()
    if a.cmd == "next":
        next_up()
    elif a.cmd == "status":
        status()
    elif a.cmd == "best":
        best()
    elif a.cmd == "tried":
        tried()
    elif a.cmd == "report":
        report(a.kind, a.limit)
    elif a.cmd == "slowest":
        slowest(a.limit)
    elif a.cmd == "history":
        history(a.kind, a.name, a.key)
    elif a.cmd == "curve":
        curve(a.kind, a.name, a.series)
    elif a.cmd == "record":
        record(a)
    elif a.cmd == "time":
        timed(a.kind, a.name, a.command)
    else:
        sweep()
        con = database()
        for row in con.execute(a.sql):
            print("\t".join("" if v is None else str(v) for v in row))


if __name__ == "__main__":
    main()
