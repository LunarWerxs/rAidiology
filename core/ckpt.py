"""Partial (mid-fold) checkpoints shared by the trainers: a restart or crash costs minutes, not a fold.

    save_partial(path, payload)   atomic write; the payload is a dict that carries "fingerprint"
    load_partial(path, fp)        the payload, or None (missing, unreadable, other fingerprint, no resume)
    rng_state() / set_rng_state(d)  python, numpy, torch CPU and every CUDA device
    Every(seconds=None)           a wall-clock trigger: .due() after that many seconds, .mark() resets it
    clear_partial(path)           removes the partial and its .tmp once the fold's final checkpoint is saved

The partial is never part of a training fingerprint, so folds saved without one still resume.
Env RAIDIOLOGY_SAVE_EVERY_S (default 300) sets Every's seconds; RAIDIOLOGY_NO_RESUME=1 ignores partials.
"""
import os
import random
import time

import numpy as np
import torch


def save_partial(path, payload):
    """Write `payload` to `path` through path + '.tmp' and os.replace, so a cut leaves the old file or the new one."""
    assert "fingerprint" in payload, "a partial checkpoint carries its fingerprint"
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_partial(path, fingerprint):
    """The payload at `path` if its fingerprint equals `fingerprint`; else None. An unreadable file is renamed
    to path + '.bad' with a warning; RAIDIOLOGY_NO_RESUME=1 or a missing file gives None."""
    if os.environ.get("RAIDIOLOGY_NO_RESUME") == "1" or not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict):
            raise ValueError("not a dict")
    except Exception as e:  # noqa: BLE001
        bad = path + ".bad"
        try:
            os.replace(path, bad)
        except OSError:
            bad = "(could not rename)"
        print("WARNING: partial checkpoint %s is unreadable (%s: %s); moved to %s, starting fresh"
              % (path, type(e).__name__, e, bad), flush=True)
        return None
    return ck if ck.get("fingerprint") == fingerprint else None


def rng_state():
    """The random generators' state: python, numpy, torch CPU and every CUDA device (a dict for a checkpoint)."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(d):
    """Put back what rng_state() returned (CUDA states only when this machine has as many devices)."""
    random.setstate(d["python"])
    np.random.set_state(d["numpy"])
    torch.set_rng_state(d["torch"].cpu())
    if d.get("cuda") and torch.cuda.is_available() and len(d["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in d["cuda"]])


class Every:
    """A wall-clock trigger: due() once `seconds` have passed since the last mark() (or since creation)."""

    def __init__(self, seconds=None):
        self.seconds = float(os.environ.get("RAIDIOLOGY_SAVE_EVERY_S") or 300) if seconds is None else float(seconds)
        self.last = time.time()

    def due(self):
        return time.time() - self.last >= self.seconds

    def mark(self):
        self.last = time.time()


def clear_partial(path):
    """Remove the partial checkpoint and its .tmp (a missing file is fine)."""
    for p in (path, path + ".tmp"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
