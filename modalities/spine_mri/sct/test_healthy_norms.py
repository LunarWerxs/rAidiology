"""healthy_norms.py and cord_compression.py step 4 on synthetic cords: CPU only, no downloads."""
import json
import os
import re
import sys

import nibabel as nib
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import healthy_norms as hn  # noqa: E402

VOX = 0.5


def tube(ap_of_z, rl=8.0, tilt_deg=0.0, shape=(48, 48, 120)):
    """A cord mask: an elliptical tube (full AP diameter ap_of_z(z mm), full RL diameter rl) whose axis leans
    tilt_deg towards anterior; 0.5 mm voxels, RAS, world z = 0 at the volume's centre."""
    aff = np.diag([VOX, VOX, VOX, 1.0])
    aff[:3, 3] = -np.array(shape) * VOX / 2
    i, j, k = np.indices(shape)
    x, y, z = (aff[:3, :3] @ np.stack([i, j, k]).reshape(3, -1) + aff[:3, 3:4]).reshape(3, *shape)
    t = np.tan(np.radians(tilt_deg))
    c = np.cos(np.radians(tilt_deg))
    # the cross-section perpendicular to the axis: in a z-slice the tube is stretched by 1/cos along y
    yy = (y - t * z) * c
    zz = z / c   # arc position along the axis
    mask = (x / (rl / 2)) ** 2 + (yy / (np.vectorize(ap_of_z)(zz) / 2)) ** 2 <= 1
    return nib.Nifti1Image(mask.astype(np.uint8), aff)


def test_tube_area_and_diameters():
    secs = hn.sections(tube(lambda z: 8.0, tilt_deg=20))
    mid = slice(10, -10)
    area = np.median(secs["area"][mid])
    assert area == pytest.approx(np.pi * 4 * 4, rel=0.05)
    assert np.median(secs["diameter_AP"][mid]) == pytest.approx(8.0, rel=0.05)
    assert np.median(secs["diameter_RL"][mid]) == pytest.approx(8.0, rel=0.05)


def _norms(rows):
    """A reference from [(sex, age, {level: diameter_AP})]; other metrics scale with it."""
    subs = {}
    for n, (sex, age, vals) in enumerate(rows):
        subs[f"sub-{n}"] = {"sex": sex, "age": age, "values": {
            lv: {"area": np.pi * v * 4 / 4, "diameter_AP": v, "diameter_RL": 8.0, "ratio_AP_RL": v / 8.0}
            for lv, v in vals.items()}}
    return {"subjects": subs}


def test_pinched_tube_normalised_ratio():
    import cord_compression as cc
    img = tube(lambda z: 4.0 if abs(z) <= 4 else 8.0)   # AP halved over 8 mm around z = 0
    secs = hn.sections(img)
    pts = {"disc_C5_C6": np.array([0.0, 6.0, 0.0]), "C5": np.array([0.0, 8.0, 15.0]), "C6": np.array([0.0, 8.0, -15.0])}
    # healthy cords as wide at every level: the plain MSCC, (1 - 4/8) * 100 = 50
    flat = _norms([("M", 40, {"C5": 8.0, "C6": 8.0, "C5-C6": 8.0})] * 12)
    r = cc.compression(secs, pts, "C5-C6", flat, "M", 40)
    assert r["cord_compression_ratio_hc"] == pytest.approx(50.0, abs=4.0)
    # healthy cords naturally narrower at the disc (6 mm): normalised (1 - (4/6) / 1) * 100 = 33.3
    narrow = _norms([("M", 40, {"C5": 8.0, "C6": 8.0, "C5-C6": 6.0})] * 12)
    r = cc.compression(secs, pts, "C5-C6", narrow, "M", 40)
    assert r["cord_compression_ratio_hc"] == pytest.approx(100 / 3, abs=4.0)
    assert r["healthy"]["C5-C6"]["n"] == 12 and r["healthy"]["C5-C6"]["fallback"] is None
    assert hn.mscc_norm((8, 8, 4), (8, 8, 6)) == pytest.approx(100 / 3)


def test_match_filters_age_and_sex_and_falls_back():
    rows = [("M", 30, {"C3-C4": 8.0})] * 12 + [("F", 30, {"C3-C4": 7.0})] * 3 + [("M", 60, {"C3-C4": 6.0})] * 5
    norms = _norms(rows)
    m = hn.match(norms, "C3-C4", "M", 35)
    assert (m["n"], m["sex"], m["fallback"], m["mean"]["diameter_AP"]) == (12, "M", None, 8.0)
    f = hn.match(norms, "C3-C4", "F", 30)   # 3 women: both sexes within 20-40
    assert f["n"] == 15 and f["sex"] == "both" and "both sexes" in f["fallback"]
    assert f["mean"]["diameter_AP"] == pytest.approx((12 * 8 + 3 * 7) / 15)
    old = hn.match(norms, "C3-C4", "M", 60)   # only the 5 men aged 60 are within 50-70
    assert old["n"] == 5 and old["fallback"] and old["mean"]["diameter_AP"] == 6.0
    assert hn.match(norms, "C4-C5", "M", 30)["n"] == 0


def test_refused_sidecar_is_never_read(tmp_path, monkeypatch):
    root = tmp_path / "sg"
    anat = root / "derivatives" / "labels" / "sub-x" / "anat"
    anat.mkdir(parents=True)
    (root / "participants.tsv").write_text("participant_id\tsex\tage\tpathology\nsub-x\tM\t30\tHC\n")
    seg = anat / "sub-x_T2w_label-SC_seg.nii.gz"
    seg.write_bytes(b"not a nifti: reading it would raise")
    (anat / "sub-x_T2w_label-SC_seg.json").write_text(json.dumps({"GeneratedBy": [{"Name": "TotalSpineSeg"}]}))
    disc = anat / "sub-x_T2w_label-discs_dlabel.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4)), str(disc))
    (anat / "sub-x_T2w_label-discs_dlabel.json").write_text(json.dumps({"GeneratedBy": [{"Name": "Manual"}]}))
    loaded = []
    real = nib.load
    monkeypatch.setattr(nib, "load", lambda p, *a, **k: loaded.append(str(p)) or real(p, *a, **k))
    norms = hn.build(str(root), str(tmp_path / "out.json"), jobs=1)
    assert str(seg) not in loaded
    assert norms["gate"]["refused"]["label-SC_seg"] == 1 and norms["subjects_n"] == 0
    from outliner import Refused
    with pytest.raises(Refused):
        hn._subject(("sub-x", str(seg), str(disc)))
    assert str(seg) not in loaded


def test_no_pam50_in_cord_compression():
    src = open(os.path.join(HERE, "cord_compression.py"), encoding="utf-8").read()
    for line in src.splitlines():
        if re.search(r"pam50|normalize-hc|normalize_hc", line, re.I):
            assert "no longer used" in line, line
    assert '"sct_compute_compression"' not in src and "PAM50_levels" not in src   # no call, no template file
