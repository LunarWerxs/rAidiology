"""Training needs the graphics card, and says so instead of running a hundred times slower.

A torch build without CUDA looks exactly like a machine without a card: `torch.cuda.is_available()`
is simply false, every device choice falls back to the processor, and a training run starts, prints
its first steps and takes days. An earlier session here lost hours that way before anybody looked
at `torch.version.cuda` (.claude/memory/torch-build-decides-whether-there-is-a-gpu.md). So a trainer
calls `require_card` first: no card in use means a stop with the reason, unless RAIDIOLOGY_DEVICE=cpu
says the processor is what was meant. Reading is different and never calls this: the shipped reading
runs on the processor on purpose.
"""
import atexit
import json
import os
import sys
import time

_HELD = None    # (gb, name) of the lease this process holds, once it holds one
_WAITED = 0.0   # seconds this process has spent waiting for a lease (see waited_seconds)

# what Windows and the desktop keep of the card when no budget is given, and what CUDA holds per process
# outside a share's ceiling (its context and library workspaces; set_per_process_memory_fraction caps only
# torch's own allocator). History: 1.5, then 1.49 at 03:48 on 2026-10-04 so three 3.5 GB chest shares
# packed into 10.5 GB. They did, while the screen slept. At 13:13, with the owner back at the PC, v103, v110
# and v111 spilled: Windows lets a process use about 11.26 of this card's 11.99 GiB, the desktop held 1.09,
# and the three held 3.50, 3.49 and 3.76 GiB (0.58 GiB of it in ordinary memory). The card then ran at
# "100%" on 65 of its 285 W, and every epoch went from 241-267 s to 1,355-1,487 s for the rest of the run,
# since torch keeps a block that fell back to ordinary memory. Measured: 11.99 - 11.26 + 1.09 = 1.82, so
# 1.9; and the most a job held over its share was 0.26, so 0.3. Three 3.5 GB shares (3 x 3.8 = 11.4 of a
# 10.09 budget) no longer start together; two of them beside a 2.1 GB job, or 3.9 + 1.5 + 3.5, still do.
DESKTOP_GB = 1.9
CUDA_OVERHEAD_GB = 0.3
LOCK_STALE_S = 60     # a lock file older than this was left by a process that died holding it
SHORT_MAX_MIN = 20    # a job that holds the card this many minutes or less may use the short-job lane


def require_card(device, torch):
    if device.type == "cuda" or os.environ.get("RAIDIOLOGY_DEVICE", "").strip() == "cpu":
        return
    raise SystemExit(
        "No graphics card is in use (torch %s, built for CUDA %s). Training on the processor is about "
        "100 times slower. If this machine has a card, this torch build is probably the CPU-only one: "
        "install the +cu build. To train on the processor anyway, set RAIDIOLOGY_DEVICE=cpu."
        % (torch.__version__, torch.version.cuda))


# --- the card lease -------------------------------------------------------------------------------
# Every long GPU process caps itself (set_per_process_memory_fraction), but nothing stopped three of
# them starting together: their caps added up past the 12 GB card, Windows spilled into ordinary
# memory instead of failing, and every job crawled (X-ray epochs went from 13 s alone to 60 s with
# two others; one froze for 30 minutes). A process therefore asks for a lease on its share before it
# takes the card, and waits until the live leases plus its own fit the budget. A lease is a small
# file named after the process, so a crashed process's share frees itself: a lease whose process is
# gone is deleted when seen.

def _lease_dir():
    d = os.environ.get("RAIDIOLOGY_CARD_DIR")
    if d:
        return d
    try:
        from core import paths
    except ImportError:     # run as `python core/card.py`: the repository root is not on the path yet
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core import paths
    return os.path.join(paths.WORK, "card")


def _alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k32.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5     # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _Lock:
    """An exclusive lock file; a lock older than LOCK_STALE_S is stale and removed."""

    def __init__(self, directory):
        self.path = os.path.join(directory, ".lock")

    def __enter__(self):
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
                        os.remove(self.path)
                        continue
                except OSError:
                    continue
                time.sleep(0.5)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def _lease_alive(held):
    """A lease is live while its process, or the child a `hold` wrapper started for it, is alive.
    2026-10-04: so a killed wrapper does not free the share of a child that is still running."""
    if _alive(int(held["pid"])):
        return True
    try:
        return _alive(int(held.get("child") or 0))
    except (ValueError, TypeError):
        return False


def _live_leases(directory, skip_pid=None):
    """The live leases as (name, gb) pairs; dead ones are deleted on the way. Call under the lock."""
    out = []
    for f in os.listdir(directory):
        if not f.endswith(".json"):
            continue
        path = os.path.join(directory, f)
        try:
            with open(path) as fh:
                held = json.load(fh)
            pid = int(held["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue    # being written or unreadable: not ours to judge
        if not _lease_alive(held):
            try:
                os.remove(path)
            except OSError:
                pass
        elif pid != skip_pid:
            out.append((held.get("name") or str(pid), float(held.get("gb", 0))))
    return out


# A process that cannot fit leaves a ticket (<pid>.wait). Without tickets the small jobs kept fitting
# into each other's gaps and the 6.5 GB X-ray trainer never did (2026-10-03: it waited behind a
# nodule detector, a chest reader and a candidate re-scorer that took turns). So a younger request
# goes ahead of the older waiters only when every one of them would still fit beside it afterwards.
# 2026-10-04: the rule looked at the oldest waiter alone. When nodule v3 ended at 03:57, chest v109
# saw fpr-v8 (oldest) fit beside it and started, and v108, older than v109, was left waiting.

def _holds_lease(directory, pid):
    """True when this pid has a live lease file (its own pid, or a child it started, still running)."""
    path = os.path.join(directory, "%d.json" % pid)
    try:
        with open(path) as fh:
            return _lease_alive(json.load(fh))
    except (ValueError, KeyError, TypeError):
        return True         # being written: counts as held
    except OSError:
        return False


def _waiters(directory, skip_pid=None):
    """The live tickets as (since, name, gb) tuples, oldest first; a ticket whose process is gone, or
    now holds a lease, is deleted on the way. Call under the lock."""
    out = []
    for f in os.listdir(directory):
        if not f.endswith(".wait"):
            continue
        path = os.path.join(directory, f)
        try:
            with open(path) as fh:
                t = json.load(fh)
            pid = int(t["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _alive(pid) or _holds_lease(directory, pid):
            try:
                os.remove(path)
            except OSError:
                pass
        elif pid != skip_pid:
            out.append((float(t.get("since", 0)), t.get("name") or str(pid), float(t.get("gb", 0))))
    return sorted(out)


# The same rule made a five-minute reading wait for hours (2026-10-03: the chest soup asked at 21:25
# and was moved to the processor at 22:16, behind a 3.9 GB training run that was itself waiting for
# another long run to end). So a job that says it holds the card SHORT_MAX_MIN minutes or less (its
# short_min) may go ahead of the older waiters when it fits in the room left after the ones that could
# start now, in turn. It only takes room the first blocked waiter could not use yet, so it delays that
# waiter by at most its own short_min. Every other request keeps the rule above.

# A queue runs many python steps one after another, and each step is a new process that asks afresh. So
# a step that asked after a long run had started lined up behind it, and the queue lost its place at every
# step (2026-10-08: a 2-minute speed bench waited over 4 hours). A waiter's place is therefore the earlier of
# its own first ask and RAIDIOLOGY_CARD_SINCE (epoch seconds, a float), when that is set and parses; an empty
# or unparsable value is ignored. A queue exports RAIDIOLOGY_CARD_SINCE=$(date +%s) once at its start, so each
# of its steps keeps the queue's place; a queue meant to go ahead of another may export an earlier value.
# The timeout still runs from the real call, so a since from hours ago never times a call out at once.

def _acquire(gb, name, budget_gb, free_gb, directory, poll_s, timeout_s, pid, say=print, short_min=None,
             since=None):
    """The lease logic with the budget and the free-memory reading injected (`free_gb()` returns the
    card's free GiB, or None to skip that guard). Returns the lease file's path once it is held."""
    if gb > budget_gb + 1e-9:
        raise SystemExit("card lease: %s wants %g GB, more than the whole budget of %g GB" % (name, gb, budget_gb))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "%d.json" % pid)
    ticket = os.path.join(directory, "%d.wait" % pid)
    if since is None:
        env = os.environ.get("RAIDIOLOGY_CARD_SINCE", "").strip()
        try:
            since = float(env) if env else None
        except ValueError:
            since = None
    began = time.time()
    start = began if since is None else min(began, since)
    said = set()
    short = short_min is not None and short_min <= SHORT_MAX_MIN
    as_short = " as a short job (%g min)" % short_min if short else ""

    def wait(key, reason):
        if key not in said:
            said.add(key)
            say(reason)
        left = None if timeout_s is None else timeout_s - (time.time() - began)
        if left is not None and left <= 0:
            release(ticket)
            raise SystemExit(reason)
        time.sleep(poll_s if left is None else max(0.0, min(poll_s, left)))

    while True:         # 1. a place in the budget, checked and taken under the lock
        with _Lock(directory):
            held = _live_leases(directory, skip_pid=pid)
            used = sum(g for _, g in held)
            # a ticket bigger than the budget can never be served, so it holds nobody back
            older = [w for w in _waiters(directory, skip_pid=pid) if w[0] < start and w[2] <= budget_gb + 1e-9]
            head = older[0] if older else None
            # the older waiters that could start now, oldest first, each in the room the ones before it leave
            room, ready = budget_gb - used, 0
            for w in older:
                if w[2] > room + 1e-9:
                    break
                room -= w[2]
                ready += 1
            fits = used + gb <= budget_gb + 1e-9
            if gb <= room + 1e-9 and (ready == len(older) or short):
                with open(path, "w") as fh:
                    json.dump({"pid": pid, "gb": gb, "name": name, "started": time.time(),
                               "short_min": short_min}, fh)
                release(ticket)
                break
            if not os.path.exists(ticket):
                with open(ticket, "w") as fh:
                    json.dump({"pid": pid, "gb": gb, "name": name, "since": start}, fh)
        if not fits:
            wait("budget", "card lease: %s wants %g GB%s and waits for %s (budget %g GB)" % (
                name, gb, as_short, ", ".join("%s %g GB" % h for h in held) or "nothing", budget_gb))
        else:
            wait("turn", "card lease: %s wants %g GB%s and lets %s (%g GB, waiting longer)%s go first" % (
                name, gb, as_short, head[1], head[2],
                " and %d more" % (len(older) - 1) if len(older) > 1 else ""))
    while free_gb is not None:      # 2. the card really has the room (a process that predates leases)
        free = free_gb()
        if free is None or free >= gb:
            break
        try:
            wait("free", "card lease: %s wants %g GB but only %.1f GB of the card is free (budget %g GB)" % (
                name, gb, free, budget_gb))
        except SystemExit:
            release(path)
            raise
    return path


def release(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _smi(field):
    """One number (MiB, first card) from nvidia-smi, read without torch; None on any failure."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=%s" % field, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
            creationflags=0x08000000 if os.name == "nt" else 0)     # CREATE_NO_WINDOW
        if out.returncode != 0:
            return None
        return int(out.stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def card_total_gb():
    """The card's memory in GiB, read without torch.cuda (initialising CUDA costs committed memory, and
    a job that is only waiting must hold none); None when there is no card in use."""
    if os.environ.get("RAIDIOLOGY_DEVICE", "").strip() == "cpu":
        return None
    if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"].strip():
        return None
    mib = _smi("memory.total")
    return None if mib is None else mib / 1024


def _free_gb():
    """The card's free GiB from nvidia-smi. 2026-10-04: this used to import torch and start CUDA, which
    cost a waiting job about 0.6 GB of committed memory (numpy's buffers add 0.74 GB more); the wrapper
    must hold neither."""
    mib = _smi("memory.free")
    return None if mib is None else mib / 1024


_SAID = set()


def _asks_more(gb, name):
    if _HELD is not None and gb > _HELD[0] + 1e-9 and (name, gb) not in _SAID:
        _SAID.add((name, gb))
        print("card lease: %s asks %g GB; this process already holds %g GB as %s, and keeps that" % (
            name, gb, _HELD[0], _HELD[1]))


def waited_seconds():
    """Seconds this process has spent waiting for the card in lease() or hold(); 0 when it never waited.

    2026-10-05: the ledger reads this to keep the wait out of a run's time (landmarks_test reported 5,421 s
    and 'OVER THE ONE-HOUR LIMIT'; about 4,750 s of it was waiting for the lease)."""
    return _WAITED


def _take(gb, name, budget_gb, poll_s, timeout_s, short_min):
    """Wait for the share and return the lease file's path, or None when there is no card."""
    global _WAITED
    total_gb = card_total_gb()
    if total_gb is None:
        return None
    if budget_gb is None:
        env = os.environ.get("RAIDIOLOGY_CARD_BUDGET_GB", "").strip()
        budget_gb = float(env) if env else total_gb - DESKTOP_GB
    if short_min is None:
        env = os.environ.get("RAIDIOLOGY_CARD_SHORT_MIN", "").strip()
        short_min = float(env) if env else None
    # The lease records what the process will really hold on the card: its share plus CUDA's own.
    t0 = time.time()
    try:
        return _acquire(gb + CUDA_OVERHEAD_GB, name, budget_gb, _free_gb, _lease_dir(), poll_s, timeout_s,
                        os.getpid(), short_min=short_min)
    finally:
        _WAITED += max(0.0, time.time() - t0)


def lease(gb, name=None, budget_gb=None, poll_s=30, timeout_s=None, short_min=None):
    """Wait for this process's share of the graphics card, then hold it until the process exits.

    The budget is RAIDIOLOGY_CARD_BUDGET_GB, else the card's memory less DESKTOP_GB for Windows and the
    desktop; the lease counts the share plus CUDA_OVERHEAD_GB, what CUDA holds outside the share's ceiling.
    short_min is the caller's honest estimate of the minutes it will hold the card (else
    RAIDIOLOGY_CARD_SHORT_MIN): at SHORT_MAX_MIN or less it may use room the oldest waiter cannot yet.
    Without a card (no nvidia-smi, CUDA_VISIBLE_DEVICES empty, RAIDIOLOGY_DEVICE=cpu) it returns at once,
    so processor runs are unaffected. Returns the GB this process holds (None without a card).

    2026-10-04: a process holds ONE lease. The chest outline job leased 2.5 GB, then asked again for 6.5
    inside limit_card_memory, waited on its own share and sat on 2.5 GB for 70 minutes. A second call now
    returns the held GB at once (and says so once if it asked for more); a parent that ran this process
    through `hold` passes its share in RAIDIOLOGY_CARD_HELD and nothing is leased here at all.

    2026-10-03 23:21: the machine ran out of committed memory and a training job crashed (numpy 'Unable to
    allocate'); jobs queued here had imported torch, started CUDA and loaded data first and sat holding
    2.1-6.9 GB each. So callers lease before their data and before CUDA, and this reads the card's size
    from nvidia-smi, never torch.cuda, while it waits."""
    global _HELD
    name = name or "pid %d" % os.getpid()
    if _HELD is None:
        env = os.environ.get("RAIDIOLOGY_CARD_HELD", "").strip()
        if env:
            _HELD = (float(env), "its parent")
    if _HELD is not None:
        _asks_more(gb, name)
        return _HELD[0]
    path = _take(gb, name, budget_gb, poll_s, timeout_s, short_min)
    if path is None:
        return None
    _HELD = (float(gb), name)
    atexit.register(release, path)
    return _HELD[0]


def hold(gb, name, command, short_min=None, poll_s=30, timeout_s=None):
    """Lease the share for this process, run `command` as a child that is told it holds it
    (RAIDIOLOGY_CARD_HELD), release, and return the child's exit code.
    2026-10-04: the wrapper waits for the card without importing numpy or torch (1.4 GB of committed
    memory each waiting job held), and the child then starts with the share already in hand."""
    import subprocess
    path = _take(gb, name, None, poll_s, timeout_s, short_min)
    env = dict(os.environ)
    if path is not None:
        env["RAIDIOLOGY_CARD_HELD"] = "%g" % gb
    try:
        # 2026-10-04: the wrapper's own stdout and stderr are handed over by name. Left to inherit, a child made
        # with CREATE_NO_WINDOW under a console-less parent (a detached background job) wrote into a hidden console of
        # its own: queue fifty-three's outline trained for 28 minutes and not one epoch line reached its log.
        sys.stdout.flush()
        sys.stderr.flush()
        child = subprocess.Popen(command, env=env, stdout=sys.stdout, stderr=sys.stderr,
                                 creationflags=0x08000000 if os.name == "nt" else 0)
        if path is not None:
            try:
                with _Lock(os.path.dirname(path)):
                    with open(path) as fh:
                        held = json.load(fh)
                    held["child"] = child.pid
                    with open(path, "w") as fh:
                        json.dump(held, fh)
            except (OSError, ValueError):
                pass
        return child.wait()
    finally:
        if path is not None:
            release(path)


def _selftest_hold():
    global _HELD
    import subprocess
    import tempfile
    keys = ("RAIDIOLOGY_CARD_DIR", "RAIDIOLOGY_CARD_HELD", "RAIDIOLOGY_CARD_BUDGET_GB")
    saved = {k: os.environ.get(k) for k in keys}
    old_held = _HELD
    try:
        with tempfile.TemporaryDirectory() as d:
            os.environ["RAIDIOLOGY_CARD_DIR"] = d
            os.environ.pop("RAIDIOLOGY_CARD_HELD", None)
            me = os.getpid()
            # (m) a second lease in the same process returns the first share and writes no ticket
            _HELD = (2.5, "first")
            assert lease(6.5, "second") == 2.5, "(m) second lease did not return the held share"
            assert os.listdir(d) == [], "(m) second lease wrote a file: %s" % os.listdir(d)
            # (n) a parent's share is returned at once, nothing written
            _HELD = None
            os.environ["RAIDIOLOGY_CARD_HELD"] = "2"
            assert lease(1.0, "child") == 2.0, "(n) the parent's share was not returned"
            assert os.listdir(d) == [], "(n) a file was written under RAIDIOLOGY_CARD_HELD"
            os.environ.pop("RAIDIOLOGY_CARD_HELD")
            _HELD = None
            # (o) a lease whose process is dead but whose child is alive is kept; dropped once both are dead
            dead = 999999
            while _alive(dead) or _alive(dead + 1):
                dead += 2
            lp = os.path.join(d, "%d.json" % dead)
            json.dump({"pid": dead, "child": me, "gb": 3, "name": "wrapper"}, open(lp, "w"))
            assert _live_leases(d) == [("wrapper", 3.0)], "(o) a live child did not keep the lease"
            json.dump({"pid": dead, "child": dead + 1, "gb": 3, "name": "wrapper"}, open(lp, "w"))
            assert _live_leases(d) == [] and not os.path.exists(lp), "(o) a lease with both dead was kept"
            # (p) `hold` end to end. It leases on the real card, so it also waits for that much of the card to be
            # really free: while training fills the card it would wait for good (2026-10-04, a selftest hung
            # behind three chest trainers). So it runs only when the card has the room, and a run that still
            # stalls fails with the reason instead of hanging.
            need = 1.5 + CUDA_OVERHEAD_GB
            free = _free_gb() if card_total_gb() is not None else None
            if card_total_gb() is None:
                print("selftest (p) skipped: no card")
            elif free is not None and free < need:
                print("selftest (p) skipped: card busy (%.1f GB free, the lease waits for %.1f)" % (free, need))
            else:
                os.environ["RAIDIOLOGY_CARD_BUDGET_GB"] = "100"
                try:
                    code = subprocess.run(
                        [sys.executable, os.path.abspath(__file__), "hold", "--gb", "1.5", "--name", "t", "--",
                         sys.executable, "-c",
                         "import os,sys; sys.exit(3 if os.environ.get('RAIDIOLOGY_CARD_HELD') == '1.5' else 4)"],
                        creationflags=0x08000000 if os.name == "nt" else 0, timeout=180).returncode
                except subprocess.TimeoutExpired:
                    raise AssertionError("(p) hold did not finish in 180 s (card free when it started: %s GB)" % free)
                assert code == 3, "(p) hold returned %s" % code
                assert not [f for f in os.listdir(d) if f.endswith(".json")], "(p) lease file left behind"
    finally:
        _HELD = old_held
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.environ["RAIDIOLOGY_CARD_DIR"] = d
        me, parent = os.getpid(), os.getppid()
        lock = os.path.join(d, ".lock")
        say = lambda s: None
        # (a) a lease of 6 registers
        p = _acquire(6, "a", 10, None, _lease_dir(), 0.2, 1, me, say)
        assert json.load(open(p))["gb"] == 6, "lease file wrong"
        assert not os.path.exists(lock), "lock left after (a)"
        # (b) a dead process's 9 GB is ignored and deleted (6 + 3 fits, 6 + 9 would not)
        dead = 999999
        while _alive(dead):
            dead += 1
        dp = os.path.join(d, "%d.json" % dead)
        json.dump({"pid": dead, "gb": 9, "name": "dead", "started": 0}, open(dp, "w"))
        p2 = _acquire(3, "b", 10, None, d, 0.2, 1, me + 4000000, say)
        assert not os.path.exists(dp), "dead lease not deleted"
        release(p2)
        assert not os.path.exists(lock), "lock left after (b)"
        # (c) a live lease of 3 (the parent's) plus ours of 6 leaves no room for another 6
        pp = os.path.join(d, "%d.json" % parent)
        json.dump({"pid": parent, "gb": 3, "name": "parent", "started": 0}, open(pp, "w"))
        try:
            _acquire(6, "c", 10, None, d, 0.2, 1, me + 4000001, say)
        except SystemExit as e:
            assert "c wants 6 GB" in str(e) and "parent 3 GB" in str(e), "reason not named: %s" % e
        else:
            raise AssertionError("(c) should not have fitted")
        assert not os.path.exists(lock), "lock left after (c)"
        # the free-memory guard waits and times out too, and gives its lease back
        release(pp)
        release(p)
        try:
            _acquire(1, "d", 10, lambda: 0.5, d, 0.2, 1, me + 4000002, say)
        except SystemExit as e:
            assert "free" in str(e)
        else:
            raise AssertionError("free-memory guard did not wait")
        assert not os.path.exists(os.path.join(d, "%d.json" % (me + 4000002))), "lease kept after timeout"
        # (e) an older waiter (the parent's ticket for 6) goes first: with 3 held, a younger 3 would fit
        # (6 of 10) but leave no room for the 6, so it waits; a younger 1 leaves room (3 + 1 + 6) and goes
        held = os.path.join(d, "%d.json" % me)
        json.dump({"pid": me, "gb": 3, "name": "held", "started": 0}, open(held, "w"))
        json.dump({"pid": parent, "gb": 6, "name": "big", "since": 0}, open(os.path.join(d, "%d.wait" % parent), "w"))
        try:
            _acquire(3, "e", 10, None, d, 0.2, 1, me + 4000003, say)
        except SystemExit as e:
            assert "lets big (6 GB" in str(e), "reason not named: %s" % e
        else:
            raise AssertionError("(e) jumped the older waiter")
        assert not os.path.exists(os.path.join(d, "%d.wait" % (me + 4000003))), "ticket kept after timeout"
        p5 = _acquire(1, "f", 10, None, d, 0.2, 1, me + 4000004, say)
        release(p5)
        # (f) a ticket whose process now holds a lease is spent and deleted; the younger 1 goes
        json.dump({"pid": parent, "gb": 6, "name": "big", "started": 0}, open(pp, "w"))
        release(held)
        p6 = _acquire(1, "g", 10, None, d, 0.2, 1, me + 4000005, say)
        assert not os.path.exists(os.path.join(d, "%d.wait" % parent)), "spent ticket not deleted"
        release(p6)
        release(pp)
        # (g) more than the whole budget is refused at once, and such a ticket holds nobody back
        try:
            _acquire(11, "h", 10, None, d, 0.2, 60, me + 4000006, say)
        except SystemExit as e:
            assert "more than the whole budget" in str(e)
        else:
            raise AssertionError("(g) an impossible request was not refused")
        json.dump({"pid": parent, "gb": 12, "name": "huge", "since": 0}, open(os.path.join(d, "%d.wait" % parent), "w"))
        p7 = _acquire(9, "i", 10, None, d, 0.2, 1, me + 4000007, say)
        release(p7)
        release(os.path.join(d, "%d.wait" % parent))
        # the short-job lane. State: 6 held (ours), an older 6 waiting (the parent's ticket), budget 10: the 6
        # cannot start (6 + 6 > 10), so a short 2 may use the room beside the held 6
        json.dump({"pid": me, "gb": 6, "name": "held", "started": 0}, open(held, "w"))
        tk = os.path.join(d, "%d.wait" % parent)
        json.dump({"pid": parent, "gb": 6, "name": "big", "since": 0}, open(tk, "w"))
        # (h) a short job (5 min) starts ahead of the older waiter that cannot start yet
        p8 = _acquire(2, "short", 10, None, d, 0.2, 1, me + 4000008, say, short_min=5)
        assert json.load(open(p8))["short_min"] == 5, "short_min not recorded"
        release(p8)
        # (i) a short job does not go ahead of a waiter that could start now (nothing held; 5 + 6 > 10 together)
        release(held)
        try:
            _acquire(5, "short", 10, None, d, 0.2, 1, me + 4000009, say, short_min=5)
        except SystemExit as e:
            assert "lets big (6 GB" in str(e) and "short job" in str(e), "reason not named: %s" % e
        else:
            raise AssertionError("(i) a short job jumped a waiter that could start now")
        # (j) over the cap (30 min) waits as before, and (k) so does a request that gave no estimate
        json.dump({"pid": me, "gb": 6, "name": "held", "started": 0}, open(held, "w"))
        for label, pid_, kw in (("(j) a 30-minute job", me + 4000010, dict(short_min=30)),
                                ("(k) a non-short job", me + 4000011, {})):
            try:
                _acquire(2, "long", 10, None, d, 0.2, 1, pid_, say, **kw)
            except SystemExit as e:
                assert "lets big (6 GB" in str(e), "reason not named: %s" % e
            else:
                raise AssertionError("%s jumped the older waiter" % label)
        release(held)
        release(tk)
        # (m) the 03:57 case: two older waiters (4 and 4) could both start now. A younger 3 fits beside the
        # oldest (3 + 4 of 10) but not beside both (3 + 8), so it waits; a younger 2 fits beside both and goes
        json.dump({"pid": parent, "gb": 4, "name": "first", "since": 0}, open(tk, "w"))
        mine = os.path.join(d, "%d.wait" % me)
        json.dump({"pid": me, "gb": 4, "name": "second", "since": 1}, open(mine, "w"))
        try:
            _acquire(3, "m", 10, None, d, 0.2, 1, me + 4000012, say)
        except SystemExit as e:
            assert "lets first (4 GB, waiting longer) and 1 more" in str(e), "reason not named: %s" % e
        else:
            raise AssertionError("(m) jumped the second-oldest waiter")
        release(_acquire(2, "m2", 10, None, d, 0.2, 1, me + 4000013, say))
        # (n) older waiters 4 and 7: the 4 could start now, the 7 could not (4 + 7 > 10). A short 5 fits in the
        # room the 4 leaves and goes; a non-short 5 would hold the 7 back, so it waits
        json.dump({"pid": me, "gb": 7, "name": "second", "since": 1}, open(mine, "w"))
        release(_acquire(5, "n", 10, None, d, 0.2, 1, me + 4000014, say, short_min=5))
        try:
            _acquire(5, "n2", 10, None, d, 0.2, 1, me + 4000015, say)
        except SystemExit as e:
            assert "lets first (4 GB" in str(e), "reason not named: %s" % e
        else:
            raise AssertionError("(n) a non-short job held back a blocked older waiter")
        release(mine)
        release(tk)
        # (r) RAIDIOLOGY_CARD_SINCE's place in line. We hold 6 of 10, so only one of two 3s fits (room 4). The parent's
        # 3 asked at 100. A call placed at 50 is older than that ticket and goes ahead; placed at 200 it waits behind it
        json.dump({"pid": me, "gb": 6, "name": "held", "started": 0}, open(held, "w"))
        json.dump({"pid": parent, "gb": 3, "name": "queue", "since": 100}, open(tk, "w"))
        release(_acquire(3, "cs1", 10, None, d, 0.2, 1, me + 4000016, say, since=50))
        try:
            _acquire(3, "cs2", 10, None, d, 0.2, 1, me + 4000017, say, since=200)
        except SystemExit as e:
            assert "lets queue (3 GB" in str(e), "reason not named: %s" % e
        else:
            raise AssertionError("(r) a call placed after the ticket jumped it")
        assert not os.path.exists(os.path.join(d, "%d.wait" % (me + 4000017))), "ticket kept after timeout"
        release(held)
        release(tk)
        print("card_since ok")
    # (l) no card is decided without torch.cuda: the processor setting and an empty CUDA_VISIBLE_DEVICES
    for key, val in (("RAIDIOLOGY_DEVICE", "cpu"), ("CUDA_VISIBLE_DEVICES", "")):
        old = os.environ.get(key)
        os.environ[key] = val
        try:
            assert card_total_gb() is None, "%s=%r still reads a card" % (key, val)
        finally:
            if old is None:
                del os.environ[key]
            else:
                os.environ[key] = old
    # (q) the wait is counted: it starts at 0, and a lease without a card leaves it at 0
    assert waited_seconds() == 0.0, "waited_seconds starts at %r" % waited_seconds()
    global _HELD
    keep = {k: os.environ.get(k) for k in ("RAIDIOLOGY_DEVICE", "RAIDIOLOGY_CARD_HELD")}
    held = _HELD
    os.environ["RAIDIOLOGY_DEVICE"] = "cpu"
    os.environ.pop("RAIDIOLOGY_CARD_HELD", None)
    _HELD = None
    try:
        assert lease(1.0, "selftest wait") is None and waited_seconds() == 0.0, "a cpu lease counted a wait"
    finally:
        _HELD = held
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _selftest_hold()
    print("selftest ok")


if __name__ == "__main__":
    usage = "usage: python core/card.py selftest | hold --gb G --name N [--short-min M] -- <command> [args...]"
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif sys.argv[1:2] == ["hold"]:
        import argparse
        ap = argparse.ArgumentParser(prog="card.py hold", usage=usage)
        ap.add_argument("--gb", type=float, required=True)
        ap.add_argument("--name", required=True)
        ap.add_argument("--short-min", type=float, default=None)
        ap.add_argument("command", nargs=argparse.REMAINDER)
        a = ap.parse_args(sys.argv[2:])
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            raise SystemExit(usage)
        raise SystemExit(hold(a.gb, a.name, cmd, short_min=a.short_min))
    else:
        raise SystemExit(usage)
