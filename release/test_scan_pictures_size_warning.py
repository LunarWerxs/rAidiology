"""A picture still over the 150 000 byte target at the lowest quality is kept, and the save says so on stderr.

Before the fix save() kept the quality-56 file without a word, although the blueprint promises 150 KB or less.
The file is still written (the site needs every picture); only the promise is now checked and reported.

    python -m pytest release/test_scan_pictures_size_warning.py
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent / "site"))  # scan_pictures.py lives in release/site/

import scan_pictures  # noqa: E402


def test_save_warns_when_even_the_lowest_quality_is_over_target(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan_pictures, "OUT", str(tmp_path))
    noise = np.random.default_rng(0).integers(0, 256, (900, 900, 3), dtype=np.uint8)
    p = scan_pictures.save(Image.fromarray(noise), "noise")
    assert os.path.getsize(p) > 150_000
    assert "over the 150 000 byte target" in capsys.readouterr().err
