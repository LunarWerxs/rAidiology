"""Which structures are one connected piece in the expert masks, measured on development scans.

    python pieces.py --exclude-fold K [--out pieces_foldK.json]

Keeping only a structure's largest predicted piece removes stray islands, and on version 1's fold 0
(2026-10-02) it did that for the ribs (+0.017 each) while cutting apart the structures that are
several pieces by nature (costal cartilages -0.25, pulmonary veins -0.24), for a net loss. So the
cleanup is offered only for structures that the truth itself shows as one piece: in at least 95% of
the scans that contain it, the largest piece holds at least 99% of its voxels. Counted on the 3 mm
cache, the grid the cleanup works on, and never on the fold being evaluated, so the list it writes
carries no knowledge of the scans it is tried on.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import datasets  # noqa: E402
import prepare  # noqa: E402
from core import ledger  # noqa: E402

ONE_PIECE_SHARE = 0.99      # of a structure's voxels in its largest piece
ONE_PIECE_SCANS = 0.95      # of the scans containing it


def shares(lab, n):
    """For each structure in one scan, the share of its voxels in its largest piece."""
    out = {}
    for c, box in enumerate(ndimage.find_objects(lab, max_label=n), start=1):
        if box is None:
            continue
        pieces, k = ndimage.label(lab[box] == c)
        sizes = np.bincount(pieces.reshape(-1))[1:]
        out[c] = float(sizes.max() / sizes.sum()) if k else 0.0
    return out


def shares_by_scan(scans, n):
    """Every scan's `shares`, its truth read once."""
    out = {}
    for i, scan in enumerate(scans, 1):
        out[scan] = shares(np.load(prepare.lo_files(scan)[1]), n)
        if i % 200 == 0:
            print("  %d of %d scans" % (i, len(scans)), flush=True)
    return out


def one_piece(by_scan, scans, names, scans_share=ONE_PIECE_SCANS):
    """The structures that are one piece in `scans`, and every share seen per class number."""
    seen = {c: [] for c in range(1, len(names) + 1)}
    for scan in scans:
        for c, share in by_scan[scan].items():
            seen[c].append(share)
    return [names[c - 1] for c, v in seen.items()
            if v and np.mean(np.array(v) >= ONE_PIECE_SHARE) >= scans_share], seen


def dev_scans(split):
    skipped = prepare.unprepared("dev")
    return sorted(s for s in split["dev_folds"] if s not in skipped)


def lists(split):
    """The one-piece list for each fold's training scans ("0" to "4") and for every development
    scan ("all"), from one pass over the truth: what segmenter.py records in a model's meta.json."""
    names = datasets.structures()
    dev = dev_scans(split)
    by_scan = shares_by_scan(dev, len(names))
    out = {"all": one_piece(by_scan, dev, names)[0]}
    for k in sorted(set(split["dev_folds"].values())):
        out[str(k)] = one_piece(by_scan, [s for s in dev if split["dev_folds"][s] != k], names)[0]
    return out


def floors(split, q):
    """Each structure's smallest believable size, in voxels at 1.5 mm: the q-th percentile of its
    true size over the scans that contain it. Per fold ("0" to "4", from that fold's training scans)
    and over every development scan ("all"), from the truth counts the cache already holds. A
    prediction smaller than its floor is dropped (segmenter.py's `floor` reading): on version 2's
    development folds, q = 0.01 cut the share of absent structures it outlines from 15.1% to 8.0%
    for 0.0009 mean Dice (min_size.py, 2026-10-02)."""
    names = datasets.structures()
    dev = dev_scans(split)
    truth = {s: np.load(prepare.hi_file(s))["truth"] for s in dev}

    def counted(scans):
        sizes = np.stack([truth[s] for s in scans]).astype(np.float64)
        return {n: float(np.quantile(sizes[:, i][sizes[:, i] > 0], q))
                for i, n in enumerate(names) if (sizes[:, i] > 0).any()}

    out = {"q": q, "all": counted(dev)}
    for k in sorted(set(split["dev_folds"].values())):
        out[str(k)] = counted([s for s in dev if split["dev_folds"][s] != k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-fold", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--scans-share", type=float, default=ONE_PIECE_SCANS,
                    help="the share of scans in which it must be one piece (default %.2f)" % ONE_PIECE_SCANS)
    a = ap.parse_args()
    names = datasets.structures()
    split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    scans = [s for s in dev_scans(split) if split["dev_folds"][s] != a.exclude_fold]
    t0 = time.time()
    found, seen = one_piece(shares_by_scan(scans, len(names)), scans, names, a.scans_share)
    out = a.out or os.path.join(prepare.OUT, "pieces_fold%d_%02d.json" % (a.exclude_fold, round(100 * a.scans_share)))
    json.dump({"exclude_fold": a.exclude_fold, "scans": len(scans), "scans_share": a.scans_share, "one_piece": found,
               "share_in_largest_median": {names[c - 1]: round(float(np.median(v)), 4) for c, v in seen.items() if v}},
              open(out, "w", encoding="utf-8"), indent=1)
    print("%d of %d structures are one piece in >= %.0f%% of %d scans -> %s"
          % (len(found), len(names), 100 * a.scans_share, len(scans), out))
    ledger.write("prepare", "ct one-piece structures, without fold %d" % a.exclude_fold, time.time() - t0,
                 {"scans": (len(scans), "count"), "one_piece_structures": (len(found), "count")})


if __name__ == "__main__":
    main()
