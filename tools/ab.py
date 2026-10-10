"""A/B: two ways of doing one thing, run in turn, timed, compared, and written into the run ledger.

    python tools/ab.py "<what is being tried>" --a "<command>" --b "<command>" [--repeat 3]
        [--a-env KEY=VALUE ...] [--b-env KEY=VALUE ...] [--cwd DIR]
        [--output FILE]        a file both arms write; 'identical' means its bytes match
        [--metric NAME=REGEX]  a number each arm prints (the regex's first group), better or worse
                               by the ledger's own rule for NAME (dice, auroc ... up; loss, error,
                               memory ... down)
        [--ignore-wall]        judge only the metric; the command's total time is recorded but
                               not judged (a bench whose metric IS the speed, with a compile or a
                               warm-up the real run pays once)
        [--dry]                run and print, write nothing to the ledger

The arms alternate, A B A B, and are judged **pair by pair**: pair i of each arm ran under the same
conditions, so the ratio inside a pair cancels a machine that warms up or gets busier during the
trial. B is **kept** when every pair favours it and the median gain is at least 3%; with five pairs
all one way the chance of that by luck is 1 in 32. It is **dropped** when every pair favours A by
that much, when its output differs (when one is checked), or when an arm fails; otherwise the trial
is **inconclusive**, and the fix is more pairs, not a looser rule. The verdict, both commands and every number go into
the ledger as an `experiment`, so `python core/ledger.py tried` is the history of what was tried.

Accuracy that needs an interval comes from the scorer, not from here: an arm can print the
scorer's mean Dice and this compares it, but whether a difference is real is the scorer's job.
Every arm runs exactly as its command says, so an arm that needs a resource limit carries it in the command.
"""
import argparse
import hashlib
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import ledger, paths, polite  # noqa: E402

FASTER_ENOUGH = 1.03




def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pairs(items):
    out = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit("expected KEY=VALUE, got %r" % item)
        out[key] = val
    return out


def run_arm(arm, command, extra_env, cwd, log_dir, i, output, metrics):
    env = dict(os.environ, **extra_env)
    log = os.path.join(log_dir, "%s%d.log" % (arm, i))
    began = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        code = subprocess.run(command, shell=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    took = time.perf_counter() - began
    text = open(log, encoding="utf-8", errors="replace").read()
    got = {}
    for name, pattern in metrics.items():
        m = re.search(pattern, text)
        got[name] = float(m.group(1)) if m else None
    digest = sha256(output) if output and os.path.exists(output) else None
    print("  %s%d: %.2f s, exit %d%s%s" % (arm, i, took, code,
                                           "".join(", %s %s" % (k, v) for k, v in got.items()),
                                           ", output %s" % digest[:12] if digest else ""), flush=True)
    return {"seconds": took, "code": code, "metrics": got, "digest": digest, "log": log}


def paired(a_vals, b_vals, way):
    """Each pair's gain for B (above 1 is better), the median, and how many pairs favour B.

    The arms alternate so that pair i of A and pair i of B ran under the same conditions; a machine
    that warms up or quietens down during a trial moves both, and the ratio inside a pair cancels
    that. Judging the two arms' spreads instead mistook such a drift for noise (2026-10-02: a 1.17x
    gain that won all five pairs was called inconclusive)."""
    gains = []
    for x, y in zip(a_vals, b_vals):
        if not x or not y:
            continue
        gains.append((y / x) if way > 0 else (x / y))
    if not gains:
        return [], float("nan"), 0
    return gains, statistics.median(gains), sum(g > 1 for g in gains)


class Refused(Exception):
    """A trial that cannot be judged: the arm or the metric it needs is missing."""


def judge_time(a_runs, b_runs, numbers):
    g, med, wins = paired([r["seconds"] for r in a_runs], [r["seconds"] for r in b_runs], -1)
    numbers["speedup"] = (med, "x")
    return "time", g, med, wins


def judge_metric(name, a_runs, b_runs, numbers):
    va = [r["metrics"][name] for r in a_runs]
    vb = [r["metrics"][name] for r in b_runs]
    if any(v is None for v in va + vb):
        raise Refused("%s was not printed by every run" % name)
    way = ledger.direction(name)
    if way == 0:
        raise SystemExit("the ledger has no better-or-worse rule for %r; name it after one" % name)
    numbers["a." + name] = (statistics.median(va), "")
    numbers["b." + name] = (statistics.median(vb), "")
    g, med, wins = paired(va, vb, way)
    numbers["gain." + name] = (med, "x")
    return name, g, med, wins


def outputs_identical(a_runs, b_runs, numbers):
    same = len({r["digest"] for r in a_runs + b_runs}) == 1 and a_runs[0]["digest"] is not None
    numbers["identical"] = (1 if same else 0, "count")
    if not same:
        raise Refused("the output differs between the arms")


def decide(judged, output):
    """Kept when every pair favours B by 3% or more on something judged, dropped when every pair
    favours A by that much on something, inconclusive otherwise."""
    better = ["%s %.2fx in %d of %d pairs" % (lab, med, w, len(g)) for lab, g, med, w in judged
              if g and w == len(g) and med >= FASTER_ENOUGH]
    worse = ["%s %.2fx in %d of %d pairs" % (lab, med, len(g) - w, len(g)) for lab, g, med, w in judged
             if g and w == 0 and med <= 1 / FASTER_ENOUGH]
    if worse:
        return "dropped", "worse: " + "; ".join(worse)
    if better:
        return "kept", "; ".join(better + (["output identical"] if output else []))
    shown = "; ".join("%s %.2fx, %d of %d pairs for B" % (lab, med, w, len(g)) for lab, g, med, w in judged)
    return "inconclusive", "not every pair agrees, or under 3%%: %s" % shown


def verdict(a_runs, b_runs, metrics, output, ignore_wall=False):
    """('kept', 'dropped' or 'inconclusive', why, numbers).

    Kept: every pair favours B on what is judged, and the median gain is at least 3%. Dropped: every
    pair favours A by that much, or B's output differs, or an arm failed. With n pairs all one way
    the chance of that by luck is 1 in 2^n (1 in 32 for five), which is the bar."""
    numbers = {"seconds_a": (statistics.median(r["seconds"] for r in a_runs), "s"),
               "seconds_b": (statistics.median(r["seconds"] for r in b_runs), "s"),
               "pairs": (min(len(a_runs), len(b_runs)), "count")}
    if any(r["code"] for r in a_runs + b_runs):
        return "dropped", "an arm exited non-zero (logs under work/ab/)", numbers
    try:
        judged = [] if ignore_wall else [judge_time(a_runs, b_runs, numbers)]
        judged += [judge_metric(name, a_runs, b_runs, numbers) for name in metrics]
        if output:
            outputs_identical(a_runs, b_runs, numbers)
    except Refused as why:
        return "dropped", str(why), numbers
    kept, why = decide(judged, output)
    return kept, why, numbers


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--a-env", action="append")
    ap.add_argument("--b-env", action="append")
    ap.add_argument("--cwd", default=ROOT)
    ap.add_argument("--output")
    ap.add_argument("--metric", action="append")
    ap.add_argument("--ignore-wall", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    metrics = pairs(a.metric)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(paths.WORK, "ab", stamp)
    os.makedirs(log_dir, exist_ok=True)
    print("A/B %s (%d each, alternating; logs in %s)" % (a.what, a.repeat, log_dir), flush=True)
    a_runs, b_runs = [], []
    t0 = time.perf_counter()
    # Rest between runs as a long job would (core/polite.py): about half the machine while somebody
    # is at it. The rest falls between runs, never inside one, so it moves no measurement.
    pace = polite.Pace()
    for i in range(1, a.repeat + 1):
        for arm, cmd, extra, runs in (("a", a.a, a.a_env, a_runs), ("b", a.b, a.b_env, b_runs)):
            runs.append(run_arm(arm, cmd, pairs(extra), a.cwd, log_dir, i, a.output, metrics))
            pace.rest(runs[-1]["seconds"])
    kept, why, numbers = verdict(a_runs, b_runs, metrics, a.output, a.ignore_wall)
    print("%s: %s" % (kept.upper(), why), flush=True)
    if a.dry:
        return
    settings = {"verdict": kept, "why": why, "a": a.a[:300], "b": a.b[:300], "repeat": str(a.repeat),
                "a_env": " ".join(a.a_env or []), "b_env": " ".join(a.b_env or [])}
    ledger.write("experiment", a.what, time.perf_counter() - t0, numbers, note="tools/ab.py; logs " + log_dir,
                 settings={k: v for k, v in settings.items() if v})


if __name__ == "__main__":
    main()
