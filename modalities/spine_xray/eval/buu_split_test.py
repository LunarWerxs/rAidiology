"""split() stratifies on the set's own LA labels, never the merged sample slip marks (regression).

split.json was locked on the set's own labels before the sample's slip marks were merged; running split() must
read annotations('LA', merged=False) so a re-run gives the same folds. The old call used the merged default.

    python modalities/spine_xray/eval/buu_split_test.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import buu  # noqa: E402


class SplitStratifiesOnOwnLabels(unittest.TestCase):
    def test_split_reads_the_sets_own_la_labels_not_the_merged_ones(self):
        calls = []
        ids = np.arange(10, dtype=np.int64)
        cls = np.zeros((10, 11), np.int64)
        cls[::2, 4] = 3  # every other patient has a class 3 line in the set's own labels

        def fake_annotations(view, merged=True):
            calls.append((view, merged))
            return ids, np.zeros((10, 22, 2), np.float32), cls

        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "split.json")
            with mock.patch.object(buu, "annotations", fake_annotations), \
                    mock.patch.object(buu, "SPLIT_PATH", out_path):
                buu.split()
            self.assertTrue(os.path.exists(out_path))
        self.assertEqual(calls, [("LA", False)])


if __name__ == "__main__":
    unittest.main()
