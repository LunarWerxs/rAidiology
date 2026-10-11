"""Keyed, invisible fingerprint for research-model weights.

mark   : flips the least significant mantissa bit of ~1 in N float32 weights, at
         positions and values derived from HMAC-SHA256(key, mark id | shape | k).
verify : tests a weights file against the expected positions/bits of each mark id.

Evidence of whose copy a file is, not a lock. Casting to float16 or quantising
erases the mark (documented limit).
"""
import argparse
import hashlib
import hmac
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

MIN_ELEMENTS = 1024
DEFAULT_RATE = 64
DEFAULT_KEY = Path(os.path.expanduser("~")) / ".raidiology" / "fingerprint.key"
Z_MIN = 8.0
FRAC_MIN = 0.95


# ---------------------------------------------------------------- key / PRNG
def load_key(path):
    key = Path(path or DEFAULT_KEY).read_bytes()
    if len(key) < 16:
        raise SystemExit("key file too short")
    return key


def key_id(key):
    return hashlib.sha256(key).hexdigest()[:12]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected(key, mark_id, shape, k, n, rate):
    """Positions (flat, C order) and bits for one tensor."""
    digest = hmac.new(key, f"{mark_id}|{tuple(shape)}|{k}".encode(), hashlib.sha256).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    m = max(1, n // rate)
    pos = rng.choice(n, size=m, replace=False)
    bits = rng.integers(0, 2, size=m, dtype=np.uint32)
    return pos, bits


def eligible(a):
    return a.dtype == np.float32 and a.size >= MIN_ELEMENTS


def mark_array(a, key, mark_id, k, rate):
    flat = np.ascontiguousarray(a).reshape(-1).copy()
    pos, bits = expected(key, mark_id, a.shape, k, flat.size, rate)
    u = flat.view(np.uint32)
    u[pos] = (u[pos] & np.uint32(0xFFFFFFFE)) | bits
    return flat.reshape(a.shape), len(pos)


# ---------------------------------------------------------------- formats
def _tensor_dict(obj):
    """Find the dict of tensors inside a loaded .pt object (same object, mutable)."""
    import torch
    if isinstance(obj, dict):
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
        for name in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if isinstance(obj.get(name), dict):
                r = _tensor_dict(obj[name])
                if r is not None:
                    return r
        for v in obj.values():
            if isinstance(v, dict):
                r = _tensor_dict(v)
                if r is not None:
                    return r
    return None


def _load_pt(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


class Weights:
    """Format-neutral view: ordered (name, float32 ndarray) plus a way to write back."""

    def __init__(self, path):
        self.path = str(path)
        self.ext = Path(path).suffix.lower()
        self.arrays = []  # (name, ndarray) for all float32 tensors, file order
        if self.ext in (".pt", ".pth"):
            import torch
            self.obj = _load_pt(path)
            self.td = _tensor_dict(self.obj)
            if self.td is None:
                raise SystemExit("no state_dict found in file")
            for n, t in self.td.items():
                if torch.is_tensor(t) and t.dtype == torch.float32:
                    self.arrays.append((n, t.detach().cpu().numpy()))
        elif self.ext == ".safetensors":
            from safetensors import safe_open
            self.tensors, self.meta = {}, None
            with safe_open(path, framework="np") as f:
                self.meta = f.metadata()
                for n in f.keys():
                    a = f.get_tensor(n)
                    self.tensors[n] = a
                    if a.dtype == np.float32:
                        self.arrays.append((n, a))
        elif self.ext == ".onnx":
            import onnx
            from onnx import numpy_helper, TensorProto
            self.model = onnx.load(path)
            for init in self.model.graph.initializer:
                if init.data_type == TensorProto.FLOAT:
                    self.arrays.append((init.name, numpy_helper.to_array(init)))
        else:
            raise SystemExit(f"unsupported file type: {self.ext}")

    def save(self, out, new):
        """new: name -> marked ndarray."""
        if self.ext in (".pt", ".pth"):
            import torch
            for n, a in new.items():
                self.td[n] = torch.from_numpy(a)
            torch.save(self.obj, out)
        elif self.ext == ".safetensors":
            from safetensors.numpy import save_file
            self.tensors.update(new)
            save_file(self.tensors, out, metadata=self.meta)
        else:
            import onnx
            from onnx import numpy_helper
            for init in self.model.graph.initializer:
                if init.name in new:
                    init.CopyFrom(numpy_helper.from_array(new[init.name], init.name))
            onnx.save(self.model, out)


# ---------------------------------------------------------------- commands
def cmd_keygen(a):
    out = Path(a.out) if a.out else DEFAULT_KEY
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    with open(out, "xb") as f:
        f.write(key)
    try:
        os.chmod(out, 0o600)
    except OSError:
        pass
    print(key_id(key))


def cmd_mark(a):
    if os.path.abspath(a.inp) == os.path.abspath(a.out):
        raise SystemExit("refusing to mark in place")
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    key = load_key(a.key_file)
    w = Weights(a.inp)
    seen, new, shapes, elems = {}, {}, [], 0
    for name, arr in w.arrays:
        if not eligible(arr):
            continue
        k = seen.get(arr.shape, 0)
        seen[arr.shape] = k + 1
        new[name], m = mark_array(arr, key, a.mark_id, k, a.rate)
        elems += m
        shapes.append(list(arr.shape))
    w.save(a.out, new)
    rec = {
        "mark_id": a.mark_id, "key_id": key_id(key), "rate": a.rate,
        "input_sha256": sha256_file(a.inp), "output_sha256": sha256_file(a.out),
        "tensors_marked": len(new), "elements_marked": elems, "shapes": shapes,
    }
    Path(str(a.out) + ".mark.json").write_text(json.dumps(rec, indent=2))
    print(f"marked {len(new)} tensors, {elems} elements -> {a.out}")


def best_match(arr, key, mark_id, kmax, rate):
    """Best (matches, total) for one tensor over seeds and the transposed layout."""
    cands = [(arr.shape, arr.reshape(-1))]
    if arr.ndim == 2:
        cands.append((arr.shape[::-1], np.ascontiguousarray(arr.T).reshape(-1)))
    best = (0, 0, -1.0)
    for shape, flat in cands:
        u = flat.view(np.uint32)
        for k in range(kmax):
            pos, bits = expected(key, mark_id, shape, k, flat.size, rate)
            hit = int(np.count_nonzero((u[pos] & np.uint32(1)) == bits))
            f = hit / len(pos)
            if f > best[2]:
                best = (hit, len(pos), f)
    return best


def verify_one(w, key, mark_id, rate):
    arrs = [x for _, x in w.arrays if eligible(x)]
    counts = {}
    for x in arrs:
        counts[x.shape] = counts.get(x.shape, 0) + 1
    hits = total = matched = 0
    for x in arrs:
        kmax = counts[x.shape] + (counts.get(x.shape[::-1], 0) if x.ndim == 2 and x.shape[0] != x.shape[1] else 0)
        h, t, f = best_match(x, key, mark_id, kmax, rate)
        hits += h
        total += t
        matched += f > FRAC_MIN
    frac = hits / total if total else 0.0
    z = (hits - total / 2) / math.sqrt(total / 4) if total else 0.0
    return {"mark_id": mark_id, "tensors": len(arrs), "tensors_matched": matched,
            "bits": total, "fraction": round(frac, 4), "z": round(z, 2),
            "marked": bool(z > Z_MIN and frac > FRAC_MIN)}


def cmd_verify(a):
    ids = list(a.mark_id or [])
    if a.ids_file:
        ids += [l.strip() for l in Path(a.ids_file).read_text().splitlines() if l.strip()]
    if not ids:
        raise SystemExit("no mark ids given")
    key = load_key(a.key_file)
    w = Weights(a.inp)
    res = [verify_one(w, key, i, a.rate) for i in ids]
    if a.json:
        print(json.dumps(res, indent=2))
        return
    for r in res:
        print(f"{r['mark_id']}: {'MARKED' if r['marked'] else 'NOT MARKED'} "
              f"(tensors matched {r['tensors_matched']}/{r['tensors']}, "
              f"fraction {r['fraction']}, z {r['z']})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("keygen")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_keygen)
    s = sub.add_parser("mark")
    s.add_argument("--mark-id", required=True)
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--key-file")
    s.add_argument("--rate", type=int, default=DEFAULT_RATE)
    s.set_defaults(fn=cmd_mark)
    s = sub.add_parser("verify")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--mark-id", action="append")
    s.add_argument("--ids-file")
    s.add_argument("--key-file")
    s.add_argument("--rate", type=int, default=DEFAULT_RATE)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_verify)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
