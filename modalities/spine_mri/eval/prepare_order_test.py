"""The disc order check is not run against a vertebra above that the top edge cuts (regression).

A vertebra cut by the top edge of the image has a misleading centroid. The order check must not compare a disc with
the vertebra below it (i - 1) when that vertebra is cut; the old check did, and flagged a correct disc as out of order.

    python modalities/spine_mri/eval/prepare_order_test.py
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
# prepare.py imports these at load time; the order check needs none of them
sys.modules.setdefault("nibabel", mock.MagicMock())
sys.modules.setdefault("SimpleITK", mock.MagicMock())
sys.modules.setdefault("spider", mock.MagicMock())
sys.modules.setdefault("split", mock.MagicMock())
import prepare  # noqa: E402


class OrderCheckSkipsCutVertebraAbove(unittest.TestCase):
    def test_disc_is_not_checked_against_a_vertebra_cut_by_the_top_edge(self):
        m = np.zeros((30, 20), np.int64)  # axis 0 is the column, axis 1 the row (inferior = larger row)
        m[0:10, 0:7] = 1      # vertebra 1: rows 0-6, touches the top edge, so cut; its centroid is near the top
        m[20:30, 5:9] = 2     # vertebra 2: rows 5-8, not cut, above disc 202
        m[0:30, 10:13] = 202  # disc 202: rows 10-12, below vertebra 2 and above vertebra 1
        report = {"order_violations": []}
        prepare._check_order("p", m, [2], report)
        self.assertEqual(report["order_violations"], [])


if __name__ == "__main__":
    unittest.main()
