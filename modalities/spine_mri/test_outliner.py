"""CPU checks of the spine outliner v2 (outliner.py) on tiny synthetic data, under a minute: the sidecar gate refuses
TotalSpineSeg labels, the partial-label loss ignores what a source does not label, disc numbering from disc points
(neck-only and lumbar-only), TotalSpineSeg's label ids, and an import under RAIDIOLOGY_NO_TSS=1."""
import json
import os
import subprocess
import sys

os.environ["RAIDIOLOGY_NO_TSS"] = "1"

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import outliner as o  # noqa: E402
from labels import NAMES, ORDER  # noqa: E402

GB_TSS = {"GeneratedBy": [{"Name": "TotalSpineSeg", "Description": "active learning"}]}
GB_SEGMRI = {"GeneratedBy": [{"Name": "TotalSegMRI", "Description": "github.com/neuropoly/totalsegmentator-mri"}]}
GB_SCT = {"GeneratedBy": [{"Name": "spinalcordtoolbox: sct_deepseg -task seg_sc_contrast_agnostic"}, {"Name": "Manual"}]}
GB_MANUAL = {"GeneratedBy": [{"Name": "Manual", "Author": "someone"}]}


def _bids(root, files):
    """files: {name: sidecar dict, or None for a label file with no sidecar}; each gets a tiny NIfTI beside it."""
    d = root / "derivatives" / "labels" / "sub-a" / "anat"
    d.mkdir(parents=True, exist_ok=True)
    for name, side in files.items():
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4)), str(d / f"{name}.nii.gz"))
        if side is not None:
            (d / f"{name}.json").write_text(json.dumps(side))
    (root / "sub-a" / "anat").mkdir(parents=True, exist_ok=True)
    for c in ("T1w", "T2w"):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), str(root / "sub-a" / "anat" / f"sub-a_{c}.nii.gz"))
    return d


def test_sidecar_gate_refuses_totalspineseg(tmp_path):
    d = _bids(tmp_path, {"sub-a_T2w_label-spine_dseg": GB_MANUAL,           # refused by kind, whatever the sidecar says
                         "sub-a_T2w_label-canal_seg": GB_TSS,
                         "sub-a_T1w_label-canal_seg": GB_SEGMRI,
                         "sub-a_T2w_label-SC_seg": GB_SCT,
                         "sub-a_T1w_label-SC_seg": {"GeneratedBy": [{"Name": "nnUNet3D"}]},
                         "sub-a_T2w_label-discs_dlabel": GB_MANUAL,
                         "sub-a_T1w_label-discs_dlabel": None})
    verdict = {r[0].split("/")[-1]: r[3] for r in o.census("spine-generic", str(tmp_path))}
    assert verdict == {"sub-a_T1w_label-SC_seg.nii.gz": "unclear", "sub-a_T1w_label-canal_seg.nii.gz": "refuse",
                       "sub-a_T1w_label-discs_dlabel.nii.gz": "unclear", "sub-a_T2w_label-SC_seg.nii.gz": "keep",
                       "sub-a_T2w_label-canal_seg.nii.gz": "refuse", "sub-a_T2w_label-discs_dlabel.nii.gz": "keep",
                       "sub-a_T2w_label-spine_dseg.nii.gz": "refuse"}
    for bad in ("sub-a_T2w_label-spine_dseg", "sub-a_T2w_label-canal_seg", "sub-a_T1w_label-canal_seg", "sub-a_T1w_label-discs_dlabel"):
        with pytest.raises(o.Refused):
            o.read_label(str(d / f"{bad}.nii.gz"), "spine-generic")
    assert o.read_label(str(d / "sub-a_T2w_label-SC_seg.nii.gz"), "spine-generic").shape == (4, 4, 4)
    got = o.fetchlist("spine-generic", str(tmp_path))
    assert not [f for f in got if "spine_dseg" in f or "canal_seg" in f]
    assert "sub-a/anat/sub-a_T2w.nii.gz" in got
    case, = o.bids_cases("spine-generic", str(tmp_path))
    assert set(case) >= {"cord", "discs"} and "canal" not in case


def test_a_label_filed_under_another_subject_is_left_out(tmp_path):
    """whole-spine files sub-amuALT's cord mask under sub-amuAM: its image would be looked for in the wrong folder."""
    d = _bids(tmp_path, {"sub-b_T2w_label-SC_seg": GB_SCT, "sub-a_T2w_label-SC_seg": GB_SCT})
    assert o.label_verdict(str(d / "sub-b_T2w_label-SC_seg.nii.gz"), "spine-generic")[0] == "unclear"
    assert o.label_verdict(str(d / "sub-a_T2w_label-SC_seg.nii.gz"), "spine-generic")[0] == "keep"
    assert not [f for f in o.fetchlist("spine-generic", str(tmp_path)) if "sub-b" in f]


def test_fetchlist_skips_a_label_whose_image_was_never_published(tmp_path):
    """whole-spine labels three subjects it publishes no images for; asking git-annex for one failed the box run."""
    _bids(tmp_path, {"sub-a_T2w_label-SC_seg": GB_SCT})
    (tmp_path / "sub-a" / "anat" / "sub-a_T2w.nii.gz").unlink()
    assert o.fetchlist("spine-generic", str(tmp_path)) == []


def test_sidecar_gate_refuses_scts_canal_model(tmp_path):
    """SCT's canal model learned from TotalSpineSeg/TotalSegMRI canal files (model-canal-seg's canal.json), so a canal
    it drew, even hand-corrected, is TotalSpineSeg's answer at one remove."""
    sct_canal = {"GeneratedBy": [{"Name": "spinalcordtoolbox: sct_deepseg sc_canal_t2 -i x.nii.gz",
                                  "ModelURL": "https://github.com/ivadomed/model-canal-seg/releases/download/r20241126/m.zip"},
                                 {"Name": "Manual", "Author": "someone"}]}
    d = _bids(tmp_path, {"sub-a_T2w_label-canal_seg": sct_canal})
    assert o.label_verdict(str(d / "sub-a_T2w_label-canal_seg.nii.gz"), "spine-generic")[0] == "refuse"
    with pytest.raises(o.Refused):
        o.read_label(str(d / "sub-a_T2w_label-canal_seg.nii.gz"), "spine-generic")


def test_whole_spine_manual_only_and_empty_sidecars_are_left_out(tmp_path):
    _bids(tmp_path, {"sub-a_T1w_label-SC_seg": GB_MANUAL, "sub-a_T2w_label-SC_seg": GB_SCT, "sub-a_T2w_labels-discs": {}})
    verdict = {r[0].split("/")[-1]: r[3] for r in o.census("whole-spine", str(tmp_path))}
    assert verdict == {"sub-a_T1w_label-SC_seg.nii.gz": "unclear", "sub-a_T2w_label-SC_seg.nii.gz": "keep",
                       "sub-a_T2w_labels-discs.nii.gz": "unclear"}


def test_partial_loss_masks_what_a_source_does_not_label():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    z = torch.randn(2, len(o.CLASSES), 8, 8, requires_grad=True)
    unknown = torch.full((2, 8, 8), o.ALL)
    ce, dice = o.partial_loss(z, unknown)
    assert ce.item() == 0 and dice.item() == 0
    a = unknown.clone()
    a[0, :4] = 1 << o.VERT                                     # rows a source labels exactly
    a[1, 0, 0] = (1 << o.CANAL) | (1 << o.CORD)                 # SPIDER's canal: canal or cord
    ce, dice = o.partial_loss(z, a)
    p = z.softmax(1)
    want = torch.cat([-torch.log(p[0, o.VERT, :4]).flatten(), -torch.log(p[1, o.CANAL, 0, 0] + p[1, o.CORD, 0, 0]).view(1)])
    assert torch.allclose(ce, want.mean(), atol=1e-5)
    (ce + dice).backward()
    g = z.grad.abs().sum(1)
    assert float(g[0, 4:].sum()) == 0 and float(g[1].sum() - g[1, 0, 0]) == 0   # unlabelled pixels pass no gradient
    assert float(g[0, :4].min()) > 0
    one = torch.full((1, 8, 8), 1 << o.DISC)
    assert torch.allclose(o.partial_loss(z.detach()[:1], one)[0], F.cross_entropy(z.detach()[:1], torch.full((1, 8, 8), o.DISC)), atol=1e-5)


@pytest.mark.parametrize("start, n, missing", [(3, 6, None), (3, 6, 2), (20, 6, None), (20, 6, 3)])
def test_numbering_from_disc_points(start, n, missing):
    """Neck-only from C2-C3 (SCT 3) and lumbar-only up to L5-S1 (SCT 25), noisy votes, with and without a missed disc."""
    rng = np.random.default_rng(start + n)
    step = 16.0 if start < 10 else 36.0
    pos = np.array([[30.0, 10 + i * step, 5.0] for i in range(n)])
    votes = start + np.arange(n) + rng.uniform(-0.45, 0.45, n)
    keep = [i for i in range(n) if i != missing]
    shuffled = rng.permutation(keep)
    got = o.number_discs(pos[shuffled], votes[shuffled])
    assert list(got) == list(start + shuffled)


def test_numbering_by_votes_not_by_counting_up_from_l5s1():
    pos = np.array([[30.0, 10 + 16 * i, 5.0] for i in range(5)])
    assert list(o.number_discs(pos, [3.1, 4.2, 4.9, 6.3, 7.0])) == [3, 4, 5, 6, 7]   # no L5-S1 in a neck scan


def test_label_ids_are_totalspineseg_step2():
    for v in range(3, 26):
        assert NAMES[o.disc_label(v)] == f"disc_{ORDER[v - 2]}_{ORDER[v - 1]}"
    for i in range(1, 25):
        assert NAMES[o.vertebra_label(i)] == ORDER[i - 1]
    assert o.disc_label(2) == 0 and o.vertebra_label(25) == 0       # no C1-C2 disc id; no sacrum in v2
    assert (o.disc_label(3), o.disc_label(8), o.disc_label(20), o.disc_label(25)) == (63, 71, 91, 100)


def _synthetic(start):
    """Class probabilities, heatmap and level map of a 4-disc sagittal stack whose top disc is SCT level `start`."""
    shape = (40, 140, 5)
    cls = np.zeros(shape, int)
    rows = [20, 50, 80, 110]
    cls[5:28, :, 1:4] = o.VERT
    for r in rows:
        cls[5:31, r - 2:r + 3, 1:4] = o.DISC
    cls[32:36, :, 1:4] = o.CANAL
    cls[33:35, :70, 2] = o.CORD
    seg = np.full((len(o.CLASSES), *shape), 0.025, np.float32)
    np.put_along_axis(seg, cls[None], 0.9, 0)
    heat = np.zeros(shape, np.float32)
    level = np.zeros(shape, np.float32)
    for j, r in enumerate(rows):
        heat[30, r, 2] = 1.0
        level[:, r - 8:r + 9, :] = start + j + (0.3 if j % 2 else -0.3)
    return seg, heat, level, rows


@pytest.mark.parametrize("start, discs, verts", [(3, [63, 64, 65, 66], [12, 13, 14, 15, 16]),
                                                 (22, [93, 94, 95, 100], [42, 43, 44, 45, 0])])
def test_outline_writes_totalspineseg_ids(start, discs, verts):
    seg, heat, level, rows = _synthetic(start)
    lab, found = o.outline(seg, heat, level, np.eye(4))
    assert [d["id"] for d in found] == discs
    assert [int(lab[15, r, 2]) for r in rows] == discs
    assert [int(lab[15, r, 2]) for r in (8, 35, 65, 95, 125)] == verts
    assert int(lab[34, 30, 2]) == 1 and int(lab[33, 100, 1]) == 2
    assert set(np.unique(lab)) - {0} <= set(NAMES)


def test_sample_targets_and_net_shape():
    import torch
    rng = np.random.default_rng(0)
    allowed = np.full((5, 120, 300), o.ALL, np.uint8)
    allowed[:, 50:55, :] = 1 << o.CORD
    case = {"img": np.zeros((5, 120, 300), np.float16), "allowed": allowed, "pts": np.array([[2, 60, 150, 4]], np.float32),
            "heat_k": np.array([False, True, True, True, False]), "useful": np.array([2]), "rows": [100, 200], "spread": 2}
    x, t = o.sample(case, rng)
    assert x.shape == (3, o.CROP, o.CROP) and t.shape == (5, o.CROP, o.CROP)
    assert t[1].max() > 0.99 and np.isclose(t[3][t[4] > 0], 4 / o.LEVEL_SCALE).all()
    assert set(np.unique(t[0])) <= {0, o.ALL, 1 << o.CORD}
    net = o.build_net(8, o.OUTPUTS).eval()
    with torch.no_grad():
        assert net(torch.zeros(1, 3, 64, 64)).shape == (1, o.OUTPUTS, 64, 64)


def test_import_under_no_tss():
    code = ("import sys, outliner\n"
            "assert not [m for m in sys.modules if m == 'tss' or m.startswith('totalspineseg')]\n"
            "try:\n    import tss\nexcept ImportError:\n    print('refused')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env={**os.environ, "RAIDIOLOGY_NO_TSS": "1"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "refused"
