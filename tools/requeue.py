"""Relaunch the queues a PC restart killed. Meant to be run on 'resume' after a restart.

A restart kills every queue and every waiting tools/after.sh launcher. This finds each queue that
was started or was waiting and never finished (work/after_<q>.log exists, and work/<q>.log lacks
'queue done' / 'FAILED' or is missing) with no live runner, copies the old logs aside
(<q>.before-restart-HHMM.log) and relaunches it through tools/after.sh exactly as it was launched:
no wait pairs if the after-log said 'started', else the same log/pattern pairs (from
work/after_<q>.args if present, else read back from the "waiting for 'P' in L" lines, which only
show the pairs after.sh had reached, so a warning is printed).
A live runner is an after.sh launcher for the queue ('after.sh <q>') or its script (work/<q>.sh,
<q>.cmd); a process that only mentions the queue (tail work/<q>.log, another queue's launcher
waiting on that log) does not count, and neither do this process and its ancestors.
tools/after.sh writes work/after_<q>.args at launch (line 1 the weight, then log and pattern lines)
and every relaunch rewrites it; without one the weight defaults to 3.
The relaunch uses Git's bash explicitly (a bare 'bash' is WSL's System32 one and runs nothing), with
no window, and waits up to 15 s for work/after_<q>.log to be rewritten: else it prints
'relaunch FAILED (after-log not written)' and the exit code is 1.

Usage: python tools/requeue.py [--dry-run] [--all] [--selftest]
       (--all: also queues older than 48 h; --selftest: touches nothing in work/)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, "work")
MAX_AGE_H = 48
DEFAULT_WEIGHT = 3
DONE_RE = re.compile(r"queue done|FAILED")


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def live_command_lines():
    """Command lines of every live process except this one and its ancestors."""
    import psutil
    skip = {os.getpid()}
    try:
        skip |= {p.pid for p in psutil.Process().parents()}
    except psutil.Error:
        pass
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] not in skip and p.info["cmdline"]:
            out.append(" ".join(p.info["cmdline"]))
    return out


def is_running(name, cmdlines):
    """True when a live process IS the runner of the queue: its after.sh launcher or its script.
    A command line that only mentions the name (tail work/<q>.log, another launcher waiting on
    that log) does not count."""
    n = re.escape(name)
    pats = [re.compile(r"after\.sh\s+['\"]?" + n + r"(?![\w])"),
            re.compile(r"[/\\]" + n + r"\.(?:sh|cmd)(?![\w])")]
    return any(p.search(c) for c in cmdlines for p in pats)


def find_bash():
    """Git's bash. A bare 'bash' resolves to WSL's System32 bash.exe on Windows, which runs nothing."""
    for c in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.exists(c):
            return c
    w = shutil.which("bash")
    if w and "system32" not in w.lower().replace("/", "\\"):
        return w
    raise RuntimeError("Git's bash.exe not found (tried C:\\Program Files\\Git\\bin and usr\\bin, PATH); "
                       "not using the System32 (WSL) one")


def launch(cmd):
    """Run a shell command line detached in Git's bash, no window."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen([find_bash(), "-c", cmd], cwd=REPO, creationflags=flags, close_fds=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_args(name, after_text):
    """(weight, [(log, pattern), ...]) for a queue that was waiting."""
    args = read(os.path.join(WORK, f"after_{name}.args")).splitlines()
    if args:
        lines = [a for a in args if a != ""]
        return int(lines[0]), list(zip(lines[1::2], lines[2::2]))
    pairs = []
    for log, pat in ((m.group(2), m.group(1)) for m in
                     re.finditer(r"after: waiting for '(.*)' in (\S+?);", after_text)):
        if (log, pat) not in pairs:
            pairs.append((log, pat))
    return DEFAULT_WEIGHT, pairs


def last_args_if_started(name):
    """The weight a started queue had, if a previous relaunch recorded it."""
    args = read(os.path.join(WORK, f"after_{name}.args")).splitlines()
    return int(args[0]) if args else DEFAULT_WEIGHT


def relaunch(name, weight, pairs, stamp):
    for base in (name, f"after_{name}"):
        src = os.path.join(WORK, base + ".log")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(WORK, f"{base}.before-restart-{stamp}.log"))
    with open(os.path.join(WORK, f"after_{name}.args"), "w", encoding="utf-8", newline="\n") as f:
        f.write(str(weight) + "\n" + "".join(f"{l}\n{p}\n" for l, p in pairs))
    words = " ".join(["tools/after.sh", name, str(weight)] +
                     ["'%s'" % x.replace("'", "'\\''") for lp in pairs for x in lp])
    cmd = f"nohup bash {words} > work/after_{name}.log 2>&1 & disown"
    t0 = time.time() - 2
    launch(cmd)
    path = os.path.join(WORK, f"after_{name}.log")
    for _ in range(30):
        try:
            if os.path.getmtime(path) >= t0:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def selftest():
    import tempfile
    L = ["bash tools/after.sh queue70 3", r"C:\Program Files\Git\bin\bash.exe work/queue70.sh",
         r"cmd /c work\queue70.cmd"]
    assert is_running("queue70", L)
    assert not is_running("queue70", ["tail -f work/queue70.log"])
    assert not is_running("queue70", ["bash tools/after.sh queue78 3 work/queue70.log queue done"])
    assert not is_running("queue7", ["bash tools/after.sh queue70 3"])
    b = find_bash()
    assert os.path.exists(b) and "system32" not in b.lower(), b
    d = tempfile.mkdtemp()
    marker = os.path.join(d, "marker.txt")
    launch("nohup bash -c \"echo ok > '%s'\" > /dev/null 2>&1 & disown" % marker.replace("\\", "/"))
    for _ in range(30):
        if os.path.exists(marker):
            break
        time.sleep(0.5)
    assert os.path.exists(marker), "launched bash did not run"
    shutil.rmtree(d, ignore_errors=True)
    print("requeue selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print what would be done, touch nothing")
    ap.add_argument("--all", action="store_true", help="include queues older than 48 h")
    ap.add_argument("--selftest", action="store_true", help="check the running test, bash and launch; touch nothing in work/")
    opts = ap.parse_args()
    if opts.selftest:
        return selftest()
    failed = False

    cmdlines = live_command_lines()
    stamp = datetime.now().strftime("%H%M")
    names = sorted(m.group(1) for f in os.listdir(WORK)
                   if (m := re.fullmatch(r"after_(queue[A-Za-z0-9_]*)\.log", f)))
    for name in names:
        after_path = os.path.join(WORK, f"after_{name}.log")
        qlog = read(os.path.join(WORK, f"{name}.log"))
        if DONE_RE.search(qlog):
            print(f"{name}: finished")
        elif is_running(name, cmdlines):
            print(f"{name}: running")
        elif not opts.all and time.time() - os.path.getmtime(after_path) > MAX_AGE_H * 3600:
            print(f"{name}: too old")
        elif not os.path.exists(os.path.join(WORK, f"{name}.cmd")):
            print(f"{name}: skipped (work/{name}.cmd does not exist, nothing to run)")
        else:
            after = read(after_path)
            if re.search(rf"after: started {re.escape(name)}\b", after):
                weight, pairs = last_args_if_started(name), []
            else:
                weight, pairs = read_args(name, after)
            how = "started" if not pairs else f"waiting on {len(pairs)} log(s)"
            if opts.dry_run:
                print(f"{name}: would relaunch ({how}, weight {weight})")
            else:
                if pairs and not os.path.exists(os.path.join(WORK, f"after_{name}.args")):
                    print(f"{name}: warning: wait pairs read back from the after-log; waits not reached "
                          f"before the restart may be missing, check work/{name}.sh")
                if relaunch(name, weight, pairs, stamp):
                    print(f"{name}: relaunched ({how}, weight {weight})")
                else:
                    print(f"{name}: relaunch FAILED (after-log not written)")
                    failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
