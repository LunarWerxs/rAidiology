"""rAidiology's own healthy-cord reference, built from Spine Generic data-multi-subject (CC BY 4.0), replacing SCT's
PAM50-based `sct_compute_compression -normalize-hc` (the PAM50 template has no licence: LICENSING.md).

Measurement (sections): the cord mask's centreline (the mask's centroid on each slice along the image's
superior-inferior axis, resampled every 1 mm along world z, Gaussian-smoothed, sigma 4 mm, then every 1 mm of arc), and at
each centreline point the mask sampled (trilinear) on a plane perpendicular to the centreline, 0.2 mm grid, +-12 mm:
area = the sampled mask's integral (mm2); AP and transverse (RL) diameters = 4 * sqrt(eigenvalue) of the section's
second moments (the full axes of the ellipse with the same moments, as SCT's diameters), the axis nearer the
anterior direction being AP; ratio = AP / RL.
Levels (healthy): Spine Generic's hand-placed disc points (*_T2w_label-discs_dlabel, SCT numbering: 2 = C1-C2,
3 = C2-C3 ... 8 = C7-T1, 9 = T1-T2), each projected to its nearest centreline point; a disc level's value is the
mean of the sections within +-1 mm of arc, a mid-vertebral level's (C2 ... T1, halfway along the arc between its two
discs) the mean within +-2.5 mm.
Compression (mscc_norm): SCT's normalised MSCC (Bedard, Valosek et al. 2024, medRxiv 10.1101/2024.03.13.24304177;
spinalcordtoolbox metric_ratio_norm): each of the patient's values above (the vertebra above the disc, mid-body),
below (the vertebra below) and at the compressed disc is divided by the healthy value at that level, then
(1 - mi / ((ma + mb) / 2)) * 100. 0 = as expected, positive = narrower.
Subjects: participants.tsv pathology HC only (its MildCompression and DCM participants are left out).
Matching (match): that sex within age +-10 years, as -normalize-hc does; both sexes when fewer than 10 match.

Usage: python healthy_norms.py build <spine_generic_root> <out.json> [--jobs N]"""
import argparse
import csv
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # modalities/spine_mri: outliner.py, the sidecar gate (imported only to build)
NORMS = os.path.join(HERE, "healthy_norms.json")
METRICS = ("area", "diameter_AP", "diameter_RL", "ratio_AP_RL")
DISCS = {2: "C1-C2", 3: "C2-C3", 4: "C3-C4", 5: "C4-C5", 6: "C5-C6", 7: "C6-C7", 8: "C7-T1", 9: "T1-T2"}
MIDS = {(2, 3): "C2", (3, 4): "C3", (4, 5): "C4", (5, 6): "C5", (6, 7): "C6", (7, 8): "C7", (8, 9): "T1"}
LEVELS = ["C1-C2", "C2", "C2-C3", "C3", "C3-C4", "C4", "C4-C5", "C5", "C5-C6", "C6", "C6-C7", "C7", "C7-T1", "T1", "T1-T2"]
DISC_HALF_MM, MID_HALF_MM = 1.0, 2.5
STEP_MM, SMOOTH_MM, PLANE_MM, GRID_MM = 1.0, 4.0, 12.0, 0.2
AGE_SPAN, MIN_N = 10, 10
SOURCE = {"dataset": "Spine Generic Public Database (Multi-Subject), github.com/spine-generic/data-multi-subject",
          "licence": "CC BY 4.0 (its LICENSE file; dataset_description.json says CC0, the stricter one is followed)",
          "credit": "Spine Generic Public Database (Multi-Subject), Cohen-Adad et al., doi:10.5281/zenodo.4299140 "
                    "(the DOI its README carries; the README asks for no other credit line), CC BY 4.0",
          "files": "derivatives/labels/*/anat/*_T2w_label-SC_seg (cord, SCT contrast-agnostic model + manual check) and "
                   "*_T2w_label-discs_dlabel (hand-placed disc points), participants.tsv (age, sex); no images"}


# ---------------------------------------------------------------- measurement


def _si_axis(aff):
    """The array axis closest to world superior-inferior."""
    return int(np.argmax(np.abs(aff[2, :3]) / np.linalg.norm(aff[:3, :3], axis=0)))


def centreline(mask, aff):
    """(points (n, 3) world mm every STEP_MM of arc, unit tangents (n, 3)), superior last; None if under 3 slices."""
    ax = _si_axis(aff)
    pts = []
    for k in range(mask.shape[ax]):
        sl = np.take(mask, k, axis=ax)
        if sl.any():
            ij = np.argwhere(sl).mean(0)
            v = np.insert(ij, ax, k)
            pts.append(aff[:3, :3] @ v + aff[:3, 3])
    if len(pts) < 3:
        return None
    p = np.array(sorted(pts, key=lambda q: q[2]))
    z = np.arange(p[0, 2], p[-1, 2] + 1e-6, STEP_MM)
    if len(z) < 3:
        return None
    c = np.stack([np.interp(z, p[:, 2], p[:, 0]), np.interp(z, p[:, 2], p[:, 1]), z], 1)
    c[:, :2] = ndimage.gaussian_filter1d(c[:, :2], SMOOTH_MM / STEP_MM, axis=0, mode="nearest")
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    u = np.arange(0, s[-1] + 1e-6, STEP_MM)
    c = np.stack([np.interp(u, s, c[:, i]) for i in range(3)], 1)
    t = np.gradient(c, axis=0)
    return c, t / np.linalg.norm(t, axis=1, keepdims=True)


def _section(vol, inv, c, t):
    """(area, AP, RL) of the mask on the plane through c perpendicular to t."""
    r = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = r - (r @ t) * t
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    g = np.arange(-PLANE_MM, PLANE_MM + 1e-6, GRID_MM)
    a, b = np.meshgrid(g, g, indexing="ij")
    w = c + a[..., None] * e1 + b[..., None] * e2
    ijk = inv[:3, :3] @ w.reshape(-1, 3).T + inv[:3, 3:4]
    v = ndimage.map_coordinates(vol, ijk, order=1, mode="constant", cval=0.0).reshape(a.shape)
    m = v.sum()
    if m <= 0:
        return None
    area = m * GRID_MM ** 2
    xy = np.stack([a.ravel(), b.ravel()], 1)
    mu = (v.ravel()[:, None] * xy).sum(0) / m
    d = xy - mu
    cov = (v.ravel()[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / m
    ev, evec = np.linalg.eigh(cov)
    axes = [e1 * evec[0, i] + e2 * evec[1, i] for i in range(2)]
    ant = np.array([0, 1.0, 0])   # RAS+: +y is anterior
    i_ap = int(np.argmax([abs(x @ ant) for x in axes]))
    ap, rl = 4 * np.sqrt(max(ev[i_ap], 0)), 4 * np.sqrt(max(ev[1 - i_ap], 0))
    return area, ap, rl


def sections(img):
    """{'points', 'tangents', 'area', 'diameter_AP', 'diameter_RL', 'ratio_AP_RL'} along a cord mask image; None if empty."""
    mask = np.asarray(img.dataobj) > 0.5
    cl = centreline(mask, img.affine)
    if cl is None:
        return None
    c, t = cl
    # crop to the cord plus the plane's reach, so sampling stays cheap
    pad = int(np.ceil(PLANE_MM * 1.5 / min(img.header.get_zooms()[:3])))
    lo = np.maximum(np.argwhere(mask).min(0) - pad, 0)
    hi = np.minimum(np.argwhere(mask).max(0) + pad + 1, mask.shape)
    vol = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    shift = np.eye(4)
    shift[:3, 3] = lo
    inv = np.linalg.inv(img.affine @ shift)
    out = {k: [] for k in ("points", "tangents", "area", "diameter_AP", "diameter_RL", "ratio_AP_RL")}
    for ci, ti in zip(c, t):
        s = _section(vol, inv, ci, ti)
        if s is None:
            continue
        out["points"].append(ci)
        out["tangents"].append(ti)
        out["area"].append(s[0])
        out["diameter_AP"].append(s[1])
        out["diameter_RL"].append(s[2])
        out["ratio_AP_RL"].append(s[1] / s[2] if s[2] > 0 else np.nan)
    if not out["area"]:
        return None
    return {k: np.asarray(v, float) for k, v in out.items()}


def arc_of(secs, point):
    """Index of the section nearest a world point."""
    return int(np.argmin(np.linalg.norm(secs["points"] - np.asarray(point, float), axis=1)))


def value_at(secs, idx, half_mm):
    """{metric: mean} over the sections within +-half_mm of arc of section idx (sections are STEP_MM apart)."""
    h = int(round(half_mm / STEP_MM))
    sl = slice(max(idx - h, 0), idx + h + 1)
    return {m: float(np.nanmean(secs[m][sl])) for m in METRICS}


def level_values(secs, discs):
    """{level: {metric: value}} from {SCT disc number: world point}: discs and the mid-vertebral levels between them."""
    out, idx = {}, {}
    for n, p in discs.items():
        if n in DISCS:
            i = arc_of(secs, p)
            if 0 < i < len(secs["area"]) - 1:   # a disc beyond the cord's ends is not on it
                idx[n] = i
                out[DISCS[n]] = value_at(secs, i, DISC_HALF_MM)
    for (a, b), name in MIDS.items():
        if a in idx and b in idx:
            out[name] = value_at(secs, (idx[a] + idx[b]) // 2, MID_HALF_MM)
    return out


def mscc_norm(patient, healthy):
    """SCT's normalised MSCC in % from (above, below, at) patient and healthy values (metric_ratio_norm)."""
    ma, mb, mi = (p / h for p, h in zip(patient, healthy))
    return (1 - mi / ((ma + mb) / 2)) * 100


# ---------------------------------------------------------------- the reference


def match(norms, level, sex, age):
    """The healthy values at `level` for `sex` within age +-10 years: {level, sex, age, n, fallback, mean, sd, values}.
    Fewer than 10 of that sex: both sexes, and `fallback` says so."""
    rows = [s for s in norms["subjects"].values() if level in s["values"] and s.get("age") is not None
            and abs(s["age"] - age) <= AGE_SPAN]
    same = [s for s in rows if s["sex"] == sex]
    fallback = None
    if len(same) < MIN_N:
        fallback = f"only {len(same)} healthy {sex} aged {age - AGE_SPAN}-{age + AGE_SPAN} at {level}: both sexes used"
        same = rows
    vals = {m: [s["values"][level][m] for s in same if s["values"][level].get(m) is not None] for m in METRICS}
    return {"level": level, "sex": sex if fallback is None else "both", "age": [age - AGE_SPAN, age + AGE_SPAN],
            "n": len(same), "fallback": fallback,
            "mean": {m: float(np.mean(v)) if v else None for m, v in vals.items()},
            "sd": {m: float(np.std(v, ddof=1)) if len(v) > 1 else None for m, v in vals.items()},
            "values": vals}


def load(path=NORMS):
    return json.load(open(path, encoding="utf-8"))


def _subject(job):
    sub, seg_p, disc_p = job
    from outliner import read_label   # the sidecar gate: refuses anything it does not keep, before any read
    seg = read_label(seg_p, "spine-generic")
    dl = read_label(disc_p, "spine-generic")
    secs = sections(seg)
    if secs is None:
        return sub, None, "empty cord mask"
    lab = np.asarray(dl.dataobj).astype(np.int32)
    discs = {}
    for n in np.unique(lab):
        if int(n) in DISCS:
            v = np.argwhere(lab == n).mean(0)
            discs[int(n)] = dl.affine[:3, :3] @ v + dl.affine[:3, 3]
    vals = level_values(secs, discs)
    return sub, {k: {m: round(x, 4) for m, x in d.items()} for k, d in vals.items()}, None


def _summary(subjects):
    out = {}
    for lv in LEVELS:
        out[lv] = {}
        for grp in ("M", "F", "all"):
            ss = [s for s in subjects.values() if lv in s["values"] and (grp == "all" or s["sex"] == grp)]
            out[lv][grp] = {m: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3), "n": len(v)}
                            if len(v) > 1 else {"mean": None, "sd": None, "n": len(v)}
                            for m in METRICS for v in [[s["values"][lv][m] for s in ss]]}
    return out


def build(root, out_p, jobs=16):
    from outliner import label_verdict
    people = {r["participant_id"]: r for r in csv.DictReader(open(os.path.join(root, "participants.tsv"), encoding="utf-8"), delimiter="\t")}
    gate = {"kept": {"label-SC_seg": 0, "label-discs_dlabel": 0}, "refused": {"label-SC_seg": 0, "label-discs_dlabel": 0},
            "unclear": {"label-SC_seg": 0, "label-discs_dlabel": 0}, "why_not_kept": {}}
    todo, skipped = [], {}
    for sub in sorted(people):
        d = os.path.join(root, "derivatives", "labels", sub, "anat")
        ps = {k: os.path.join(d, f"{sub}_T2w_{k}.nii.gz") for k in ("label-SC_seg", "label-discs_dlabel")}
        ok = True
        for k, p in ps.items():
            verdict, why = label_verdict(p, "spine-generic")
            gate[{"keep": "kept", "refuse": "refused"}.get(verdict, verdict)][k] += 1
            if verdict != "keep":
                gate["why_not_kept"][f"{k}: {why}"] = gate["why_not_kept"].get(f"{k}: {why}", 0) + 1
                ok = False
        if people[sub].get("pathology") != "HC":
            skipped[sub] = f"not a healthy control (participants.tsv pathology: {people[sub].get('pathology')})"
        elif not ok:
            skipped[sub] = "a label file the gate does not keep"
        elif not all(os.path.exists(os.path.realpath(p)) for p in ps.values()):
            skipped[sub] = "label file not downloaded"
        else:
            todo.append((sub, ps["label-SC_seg"], ps["label-discs_dlabel"]))
    if jobs > 1:
        with Pool(jobs) as pool:
            res = pool.map(_subject, todo)
    else:
        res = [_subject(j) for j in todo]
    subjects = {}
    for sub, vals, err in res:
        if err or not vals:
            skipped[sub] = err or "no level measured"
            continue
        p = people[sub]
        age = p.get("age", "")
        subjects[sub] = {"sex": p["sex"], "age": int(float(age)) if age.replace(".", "", 1).isdigit() else None,
                         "pathology": p.get("pathology"), "values": vals}
    try:
        commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    sexes = [s["sex"] for s in subjects.values()]
    norms = {"source": {**SOURCE, "commit": commit, "participants": len(people)},
             "subjects_n": len(subjects), "sex_n": {"M": sexes.count("M"), "F": sexes.count("F")},
             "gate": gate, "skipped_n": len(skipped), "skipped": skipped,
             "method": __doc__.split("\n\nUsage")[0], "metrics": list(METRICS), "levels": LEVELS,
             "summary": _summary(subjects), "subjects": subjects}
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(norms, f, indent=1)
    print(f"{len(subjects)} subjects measured, {len(skipped)} skipped -> {out_p}")
    return norms


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("root")
    b.add_argument("out")
    b.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.root, a.out, a.jobs)


if __name__ == "__main__":
    main()
