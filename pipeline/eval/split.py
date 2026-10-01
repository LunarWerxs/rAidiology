"""Patient split for the SPIDER scorecard: 20% held-out test, 80% development in 5 folds.

Seed 20260930. SPIDER's own `subset` column is ignored. SPIDER does not record the hospital, so
patients are stratified by Manufacturer x MagneticFieldStrength as recorded on their T2 series
(their first series when they have no T2). Strata under 10 patients are pooled for stratifying.

Procedure (deterministic): one numpy Generator(seed); for each stratum in sorted name order, the
stratum's patients sorted by id are permuted; the first round(0.2 * n) go to test; the rest of
every stratum, concatenated in that same order, are dealt to folds 0..4 in turn with one running
counter, so small strata spread over the folds.
Usage: python split.py [out.json]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider import SEED, patient_series, rng, vendor  # noqa: E402

MIN_STRATUM = 10


def make():
    ps = patient_series()
    raw = {p: f"{s['row'].Manufacturer} {float(s['row'].MagneticFieldStrength):.1f}T" for p, s in ps.items()}
    counts = {}
    for v in raw.values():
        counts[v] = counts.get(v, 0) + 1
    strat = {p: (v if counts[v] >= MIN_STRATUM else "pooled small strata") for p, v in raw.items()}
    g = rng()
    test, dev_order = [], []
    for name in sorted(set(strat.values())):
        members = sorted(p for p in strat if strat[p] == name)
        perm = [members[i] for i in g.permutation(len(members))]
        k = int(round(0.2 * len(members)))
        test += perm[:k]
        dev_order += perm[k:]
    fold = {p: i % 5 for i, p in enumerate(dev_order)}
    patients = {}
    for p in sorted(ps):
        patients[str(p)] = {"split": "test" if p in test else "dev", "fold": fold.get(p),
                            "stratum": raw[p], "vendor": vendor(ps[p]["row"].Manufacturer),
                            "t2": ps[p]["t2"], "t1": ps[p]["t1"]}
    return {
        "seed": SEED, "source": "SPIDER (van der Graaf, van Hooff, Buckens et al., 'Lumbar spine segmentation in MR images: a dataset and a public benchmark', Scientific Data 11, 264 (2024), doi:10.1038/s41597-024-03090-w; data doi:10.5281/zenodo.10159290, CC BY 4.0), overview.csv",
        "rule": __doc__.split("Usage:")[0].strip(),
        "strata": dict(sorted(counts.items())),
        "n": {"patients": len(ps), "test": len(test), "dev": len(dev_order)},
        "test": sorted(test),
        "dev_folds": {str(k): sorted(p for p, f in fold.items() if f == k) for k in range(5)},
        "patients": patients,
    }


def load(p=None):
    return json.load(open(p or os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.json")))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.json")
    s = make()
    json.dump(s, open(out, "w"), indent=1)
    print(out, s["n"], {k: len(v) for k, v in s["dev_folds"].items()})
