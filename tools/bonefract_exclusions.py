"""BoneFract's pictures left out of its index, from tools/xray_duplicates.py's pairs (r 0.97 or more).

BoneFract files about 17,000 of its 47,931 pictures more than once, under different patient
numbers and often different regions; at r 0.999 or more the copies are the same file. Groups of
copies (pairs joined into connected groups) are handled as one picture:
  - a group whose copies carry both labels is left out whole ("both labels"): its truth is unknown;
  - otherwise the copy whose name sorts first stays and the rest are left out ("copy");
  - a picture matching another collection's ("other collection") is left out, as for OrthoFrac-XR.
Writes modalities/xray/eval/bonefract_excluded.csv (stem, reason, kept, r), which datasets.py reads.
Usage: python tools/bonefract_exclusions.py [work/eval/xray/bonefract_duplicates.csv]"""
import collections
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "modalities", "xray", "eval", "bonefract_excluded.csv")


def label(stem):
    return stem.rsplit("_", 2)[-2]


def main(src):
    d = pd.read_csv(src)
    d = d[d.dataset == "bonefract"]
    rows = {}
    for r in d[d.other_dataset != "bonefract"].itertuples():
        rows[r.image] = {"stem": r.image, "reason": "other collection", "kept": "%s/%s" % (r.other_dataset, r.other_image),
                         "r": r.r}
    w = d[d.other_dataset == "bonefract"]
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    best = collections.defaultdict(float)
    for a, b, r in zip(w.image, w.other_image, w.r):
        parent[find(a)] = find(b)
        best[a] = max(best[a], r)
    groups = collections.defaultdict(list)
    for a in list(parent):
        groups[find(a)].append(a)
    n_both = n_copy = 0
    for g in groups.values():
        g = sorted(g)
        both = len({label(x) for x in g}) > 1
        for x in (g if both else g[1:]):
            rows.setdefault(x, {"stem": x, "reason": "both labels" if both else "copy", "kept": "" if both else g[0],
                                "r": round(best[x], 5)})
        n_both += both
        n_copy += not both
    out = pd.DataFrame(sorted(rows.values(), key=lambda r: r["stem"]))
    out.to_csv(OUT, index=False, encoding="utf-8")
    print("%d groups of copies (%d with both labels); left out: %s -> %s"
          % (len(groups), n_both, out.reason.value_counts().to_dict(), os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "work", "eval", "xray", "bonefract_duplicates.csv"))
