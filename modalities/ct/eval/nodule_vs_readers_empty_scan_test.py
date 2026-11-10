"""A scan the detector finds nothing on still counts in the detector-vs-readers comparison (regression).

The comparison used to keep only the four-reader scans that had at least one detector candidate, so a scan the
detector found nothing on dropped out with its lesions: the detector's misses vanished and its sensitivity was
inflated. Here scan "b" has a lesion all four readers mark and no candidate, so it must count: 2 scans, and the
detector catches 1 of the 2 lesions (50), not 1 of 1 (100).

    python modalities/ct/eval/nodule_vs_readers_empty_scan_test.py
"""
import os
import sys
import unittest

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
import nodule_vs_readers as nvr  # noqa: E402
import lidc_readers as lr  # noqa: E402


def _mark(uid, reader, kind, x, y, z, diam=6.0):
    return dict(series_uid=uid, patient="p" + uid, reader=reader, kind=kind, x_mm=x, y_mm=y, z_mm=z,
                diameter_mm=diam if kind == "nodule" else float("nan"))


class ScanWithNoCandidateStillCounts(unittest.TestCase):
    def test_a_four_reader_scan_with_no_candidate_is_counted_and_its_lesion_is_missed(self):
        rows = []
        for u in ("a", "b"):                   # every reader marks the same lesion on both scans
            rows += [_mark(u, r, "nodule", 10, 10, 10) for r in range(4)]
        mk = pd.DataFrame(rows)
        sc = pd.DataFrame({"series_uid": ["a", "b"], "patient": ["pa", "pb"], "n_readers": [4, 4]})
        keys = pd.concat([lr.scan_keys(u, mk[mk.series_uid == u], r) for u in ("a", "b") for r in range(4)])
        cands = pd.DataFrame([dict(seriesuid="a", coordX=10, coordY=10, coordZ=10, probability=0.9)])  # none on "b"
        table, _, n = nvr.compare(cands, keys, mk, sc, n_boot=0)
        self.assertEqual(n, 2)
        pooled = table[(table.key == "2of3") & (table.reader_position == "pooled")].iloc[0]
        self.assertEqual(pooled.reader_sensitivity, 100.0)
        self.assertEqual(pooled.detector_sensitivity, 50.0, pooled)


if __name__ == "__main__":
    unittest.main()
