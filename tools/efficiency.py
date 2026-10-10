"""Where the time goes and what it runs on: the start of every efficiency pass.

    python tools/efficiency.py            the slowest steps, the library stack, the card
    python tools/efficiency.py --outdated also asks the package index what is newer (network)

The slowest steps come from the run ledger (the latest run of each). The stack is read from the
installed packages' metadata without importing them, so this takes about a second. The pass itself
is recorded in the ledger as a `check`, with the versions as its settings: the day a library is
upgraded is then a visible line in the history, and every run after it carries the new versions in
its own `env` (core/ledger.py).

What to do with the answer is in the standing rule: every change an A/B (`tools/ab.py`), kept only
when it is faster or more accurate and nothing regresses.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from importlib import metadata

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import ledger  # noqa: E402

# The packages whose version can move a training time, a reading time or a number.
STACK = ("torch", "torchvision", "numpy", "pandas", "polars", "pyarrow", "scipy", "scikit-learn",
         "scikit-image", "nibabel", "pydicom", "pillow", "timm", "numba", "triton-windows", "onnxruntime")


def installed():
    out = {}
    for name in STACK:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def outdated():
    """{package: newest} for the stack, from pip's own index query."""
    got = subprocess.run([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                         capture_output=True, text=True, timeout=300)
    if got.returncode:
        raise SystemExit("pip could not reach the index: %s" % got.stderr.strip()[-300:])
    want = {s.lower() for s in STACK}
    return {p["name"].lower(): p["latest_version"] for p in json.loads(got.stdout or "[]")
            if p["name"].lower() in want}


def card():
    try:
        got = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return got.stdout.strip().splitlines()[0] if got.returncode == 0 and got.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdated", action="store_true")
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    ledger.slowest(a.limit)
    stack = installed()
    newer = outdated() if a.outdated else {}
    gpu = card()
    print("\nThe stack (Python %s%s):" % (platform.python_version(), ", " + gpu if gpu else ""))
    for name, version in stack.items():
        print("  %-15s %-14s %s" % (name, version, ("newer: " + newer[name]) if name in newer else ""))
    missing = [s for s in STACK if s not in stack]
    if missing:
        print("  not installed: " + ", ".join(missing))
    settings = {"python": platform.python_version(), **stack}
    if gpu:
        settings["card"] = gpu
    numbers = {"packages": (len(stack), "count")}
    if a.outdated:
        numbers["behind"] = (len(newer), "count")
        settings.update({"newest." + k: v for k, v in newer.items()})
    ledger.write("check", "efficiency inventory", time.time() - t0, numbers, settings=settings)


if __name__ == "__main__":
    main()
