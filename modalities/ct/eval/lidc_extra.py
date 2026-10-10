"""Extra training scans for the lung nodule detector: the LIDC-IDRI scans LUNA16 left out.

What. LIDC-IDRI has 1,018 chest CTs and LUNA16 kept 888 of them (thin slices, no gaps, 3 mm or more
nodules marked by at least 3 of the 4 radiologists). All 1,018 are prepared as 1 mm lung boxes
(lung_prepare.py); the rest, the extras, are unused. `extra_reference()` gives them an answer key
built the way LUNA16 built its own, so nodule.py can train on them (`train --extra-lidc`).

Why. More scans to learn from, never to be scored: an extra is never held out and never read, so
every score is still LUNA16's own 888 scans, from a model that did not train on the scan it reads.

The rule (LUNA16's, from lidc.py's lesions: marks of 3 mm or more matched across readers).
  - nodules: a lesion marked by at least 3 radiologists; centre = the mean of the marks, diameter =
    their mean diameter.
  - excluded findings (what LUNA16 keeps in annotations_excluded.csv, counted neither way): lesions
    marked by 1 or 2 radiologists, every mark under 3 mm ('small', a point), and every mark a
    radiologist called not a nodule ('non_nodule'); diameter 0 when a mark has none.
Both frames use nodule.py's column names: seriesuid, coordX, coordY, coordZ, diameter_mm.

    python modalities/ct/eval/lidc_extra.py check

`check` asserts the extras are 100+ scans, none in LUNA16, all with a prepared box; applies the same
rule to LUNA16's 888 scans and asserts at least 90% of annotations.csv's nodules are reproduced
within the nodule's radius; and asserts nodule.training_uids holds every extra and no held-out scan.
Numbers the check printed: 130 extra scans carrying 207 nodules (in 101 scans) and 4,551 excluded
findings (in 128 scans); the rule reproduces 1,186 of LUNA16's 1,186 annotations.csv nodules (100.0%),
with 1 of our 1,187 lesions at 3 or more readers unmatched; fold 0 trains on 840 scans (710 LUNA16 + 130).
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import lidc  # noqa: E402
import lung_prepare as lp  # noqa: E402

COLS = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
MIN_READERS = 3        # LUNA16: at least 3 of the 4 radiologists
MIN_REPRODUCED = 0.90


def _rows(df, uid, x, y, z, d):
    return pd.DataFrame({"seriesuid": df[uid].to_numpy(), "coordX": df[x].to_numpy(), "coordY": df[y].to_numpy(),
                         "coordZ": df[z].to_numpy(), "diameter_mm": df[d].fillna(0).to_numpy()}, columns=COLS)


def extra_uids():
    """LIDC CT series with a prepared box that are not in LUNA16's 888."""
    luna = set(lp.luna16_subsets())
    out = []
    for uid in sorted(lidc.series_records()):
        if uid in luna:
            continue
        npy, js = lp.paths_for(lp.out_dir(), uid)
        if os.path.exists(npy) and os.path.exists(js):
            out.append(uid)
    return out


def build(uids, les=None, marks=None):
    """(ann, exc) for the given scans by LUNA16's rule."""
    keep = set(uids)
    les = lidc.nodules() if les is None else les
    marks = lidc.marks() if marks is None else marks
    les = les[les.series_uid.isin(keep)]
    marks = marks[marks.series_uid.isin(keep)]
    cols = ("series_uid", "x_mm", "y_mm", "z_mm", "diameter_mm")
    agreed = les[les.n_readers >= MIN_READERS]
    few = les[les.n_readers < MIN_READERS]
    other = marks[marks.kind.isin(["small", "non_nodule"])]
    ann = _rows(agreed, *cols)
    exc = pd.concat([_rows(few, *cols), _rows(other, *cols)], ignore_index=True)
    return ann, exc


def extra_reference():
    """(uids, ann, exc) of the extra scans."""
    uids = extra_uids()
    ann, exc = build(uids)
    return uids, ann, exc


def check():
    luna = lp.luna16_subsets()
    uids, ann, exc = extra_reference()
    assert len(uids) > 100, "only %d extra scans" % len(uids)
    assert not set(uids) & set(luna), "an extra scan is in LUNA16"
    for u in uids:
        npy, js = lp.paths_for(lp.out_dir(), u)
        assert os.path.exists(npy) and os.path.exists(js), "no prepared box for " + u
    assert set(ann.seriesuid) <= set(uids) and set(exc.seriesuid) <= set(uids)
    print("extra scans (LIDC-IDRI, prepared, not in LUNA16): %d" % len(uids))
    print("  nodules (3 mm or more, at least %d readers): %d in %d scans" % (MIN_READERS, len(ann), ann.seriesuid.nunique()))
    print("  excluded findings (1-2 readers, small, non-nodule): %d in %d scans" % (len(exc), exc.seriesuid.nunique()))

    m = lidc.luna16_match(lidc.nodules())
    assert m, "LUNA16's files are not there"
    share = m["found"] / m["luna_nodules_here"]
    print("rule applied to LUNA16's scans: %d of its %d annotations.csv nodules reproduced within the radius (%.1f%%);"
          " %d of our %d lesions at 3 or more readers unmatched"
          % (m["found"], m["luna_nodules_here"], 100 * share, m["ours_unmatched"], m["ours_at_3"]))
    assert share >= MIN_REPRODUCED, "only %.1f%% of LUNA16's nodules reproduced" % (100 * share)

    import nodule
    folds_of, _ = nodule.scan_folds(False)
    got = set(nodule.training_uids(folds_of, 0, uids))
    held = {u for u, f in folds_of.items() if f == 0}
    assert held and set(uids) <= got, "training_uids lacks an extra scan"
    assert not got & held, "training_uids holds a fold-0 scan"
    assert got == set(uids) | (set(folds_of) - held)
    print("training_uids for fold 0: %d scans = %d LUNA16 + %d extras, none of the %d held out"
          % (len(got), len(got) - len(uids), len(uids), len(held)))
    return 0


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        return check()
    print(__doc__.split("\n")[0])
    print("usage: python modalities/ct/eval/lidc_extra.py check")
    return 2


if __name__ == "__main__":
    sys.exit(main())
