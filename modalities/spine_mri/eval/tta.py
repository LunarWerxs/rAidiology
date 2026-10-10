"""Each disc read twice, as it is and mirrored left to right (its five planes reversed), the logits
averaged: measured on a run's single-model out-of-fold logits, since only its outer models are saved.

Model k reads fold k's SPIDER discs (TotalSpineSeg crops) both ways. The unmirrored reading must
reproduce the run's saved single-model logits, or the run is not what this reads and it stops.
Writes <run>/oof_logits_tta.npz: a copy of oof_logits.npz with those rows' `single` replaced, for
`compare.py --which single`. The bar is docs/efficiency's ("Read each disc twice").
Usage: python tta.py <version> [<version> ...]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grader import MODELS, load_grader, load_split_crops, logits  # noqa: E402

TOL = 1e-3


def main(version):
    d = os.path.join(MODELS, f"disc-grader-v{version}")
    o = dict(np.load(os.path.join(d, "oof_logits.npz"), allow_pickle=False))
    nets, _ = load_grader(version)
    x, idx = load_split_crops("dev")
    sp = o["collection"] == "spider" if "collection" in o else np.ones(len(o["y"]), bool)
    row = {(int(p), int(i)): r for r, (p, i, s) in enumerate(zip(o["patient"], o["ivd"], sp)) if s}
    plain = np.zeros((len(idx), o["single"].shape[1]), np.float32)
    both = np.zeros_like(plain)
    for k in range(5):
        sel = (idx.fold == k).to_numpy()
        h = idx.has_t1[sel].to_numpy()
        plain[sel] = logits([nets[k]], x[sel], h)
        both[sel] = (plain[sel] + logits([nets[k]], x[sel][:, :, ::-1].copy(), h)) / 2
    at = np.array([row[(int(p), int(i))] for p, i in zip(idx.patient, idx.ivd)])
    err = float(np.abs(plain - o["single"][at]).max())
    if err > TOL:
        raise SystemExit(f"v{version}: the unmirrored reading differs from the saved logits by {err:.4g}; not this run's models")
    o["single"] = o["single"].copy()
    o["single"][at] = both
    out = os.path.join(d, "oof_logits_tta.npz")
    np.savez(out, **o)
    print(f"v{version}: {len(idx)} SPIDER discs read twice (unmirrored matches the saved logits to {err:.1e}) -> {out}")


if __name__ == "__main__":
    for v in sys.argv[1:]:
        main(int(v))
