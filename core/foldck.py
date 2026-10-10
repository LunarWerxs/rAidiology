"""Whole-fold resume and selftest helpers for the trainers that save a plain state dict as a fold's model.

A model file that other code reads as a bare state dict (grader fold{k}.pt, the segmenter's EMA weights) cannot
carry a fingerprint inside, so the fingerprint of the settings that trained it sits beside it, in
<model>.fp. A model without that file is an old one: it is not resumed from (the grader retrains it, the segmenter
keeps treating it as trained, as before). RAIDIOLOGY_NO_RESUME=1 ignores every fingerprint.

    digest(obj)                      a short hash of anything JSON-able (and arrays, by bytes)
    fp_ok(model, fp)                 the model file exists and its .fp holds `fp` (and no NO_RESUME)
    finish(model, save, fp)          write the model through `save(tmp_path)`, atomically, then its .fp
    stop_after(n)                    selftests: raise SelftestStop when RAIDIOLOGY_CKPT_SELFTEST_STOP == str(n)
    same(a, b)                       tensors / arrays / dicts / lists equal, for the selftests
"""
import hashlib
import json
import os

import numpy as np
import torch


class SelftestStop(Exception):
    """Raised by stop_after() so a selftest can stop a trainer part-way, as a crash would."""


def digest(obj):
    h = hashlib.sha256()
    h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return h.hexdigest()[:32]


def array_digest(*arrays, sample=64):
    """A hash of the arrays' shapes and of about `sample` evenly spaced rows of each (cheap on a large array)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(repr((a.shape, str(a.dtype))).encode())
        step = max(1, len(a) // sample) if a.ndim else 1
        h.update(np.ascontiguousarray(a[::step] if a.ndim else a).tobytes())
    return h.hexdigest()[:32]


def _fp_file(model):
    return model + ".fp"


def fp_ok(model, fp):
    if os.environ.get("RAIDIOLOGY_NO_RESUME") == "1" or not os.path.exists(model) or not os.path.exists(_fp_file(model)):
        return False
    try:
        with open(_fp_file(model), encoding="utf-8") as f:
            return f.read().strip() == fp
    except OSError:
        return False


def finish(model, save, fp):
    """The fold's final model: the old .fp goes first, the file arrives whole (tmp + os.replace), the new .fp last,
    so a cut leaves either no .fp (retrained) or a .fp that belongs to the file."""
    try:
        os.remove(_fp_file(model))
    except FileNotFoundError:
        pass
    save(model + ".tmp")
    os.replace(model + ".tmp", model)
    with open(_fp_file(model) + ".tmp", "w", encoding="utf-8") as f:
        f.write(fp)
    os.replace(_fp_file(model) + ".tmp", _fp_file(model))


def stop_after(n):
    if os.environ.get("RAIDIOLOGY_CKPT_SELFTEST_STOP") == str(n):
        raise SelftestStop("selftest stop after %s" % n)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and bool(torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b
