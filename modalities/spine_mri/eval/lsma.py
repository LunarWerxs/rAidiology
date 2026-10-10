"""LSMA-PQR (Mendeley Data 10.17632/p3r4xd2488 v2, CC BY 4.0): Pfirrmann grades of L3-L4, L4-L5
and L5-S1 for 515 patients, read on the sagittal MRI of Sudirman et al.'s Lumbar Spine MRI Dataset
(10.17632/k57fr854j2 v2, CC BY 4.0), which holds the scans. A training collection on trial for the
disc grader (grader.py EXTRA), with its own locked split.

Truth: `PfirrmannGrade.csv`, one row per patient, columns D3, D4, D5 = L3-L4, L4-L5, L5-S1, so SPIDER's
bottom-up disc numbers are D5 -> 1, D4 -> 2, D3 -> 3. Grades only: none of the grader's items, which
are NaN here and left out of its item loss.

Where a disc is: LSMA-PQR's mid-sagittal mask (`masks_indexed/sag_T2_<id>_S<slice>.png`, 384 x 384,
class 2 the discs, 3 the sacrum, read off its quality-check pictures) marks every disc on one slice.
`<slice>` is the number in the DICOM file's name (`..._008.ima`), not its place in the stack, and
the picture is that slice resized to 384 x 384: every patient's picture is correlated with it, and
a patient whose picture matches no slice (r < MATCH_R) is left out and listed. Which mask disc is
which level comes from `IVDHeights.csv`, whose two height points per disc (384-pixel x, y) lie in
it; the levels must then run D3, D4, D5 downward. The disc's centre and long axis set the crop the
way a TotalSpineSeg label does for SPIDER (crops.py), on the full T2 and T1 volumes, T1 being the
series whose own picture (`sag_T1_...`) matches. Some series hold every slice twice (two files per
position); one file per position is kept.

Split: the unit is the patient; stratified by the patient's worst grade, strata under 10 pooled;
20% test, the rest dealt to five folds; seed spider.SEED. Locked in `lsma_split.json` before
anything was fitted, and the test split opens only through final_eval (guard_test).

Usage: python lsma.py split            (once: writes lsma_split.json; refuses to overwrite)
       python lsma.py crops <dev|test> [workers]  -> <work>/eval/lsma/<split>/crops.npz, crops_index.csv,
                                                    left_out.csv
       python lsma.py score <headline version> <version> [<version> ...]
              (RAIDIOLOGY_FINAL_EVAL=1, after `crops test`: the test split, opened once)"""
import functools
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402
from spider import ITEMS, SEED, guard_test  # noqa: E402

ROOT = os.environ.get("LSMA_ROOT", os.path.join(paths.DATASETS, "public", "lsma-pqr", "LSMA-PQR"))
SCANS = os.environ.get("SUDIRMAN_ROOT", os.path.join(paths.DATASETS, "public", "lumbar-spine-mri-sudirman"))
SPLIT = os.path.join(HERE, "lsma_split.json")
LEVEL_IVD = {"D5": 1, "D4": 2, "D3": 3}         # SPIDER's bottom-up disc number
PATIENT_OFFSET = 100000                          # so a patient never shares a number with SPIDER's
MIN_STRATUM, FOLDS, TEST_SHARE = 10, 5, 0.2
DISC, SACRUM = 2, 3
PNG, MATCH_R = 384, 0.98
OUT = os.path.join(paths.WORK, "eval", "lsma")


def grades():
    """One row per graded disc: patient (LSMA-PQR's id), ivd, pfirrmann."""
    g = pd.read_csv(os.path.join(ROOT, "PfirrmannGrade.csv"))
    rows = [{"patient": int(r.Patient_ID), "ivd": ivd, "pfirrmann": int(getattr(r, col))}
            for r in g.itertuples() for col, ivd in LEVEL_IVD.items()]
    return pd.DataFrame(rows)


def make_split():
    g = grades()
    worst = g.groupby("patient").pfirrmann.max()
    counts = worst.value_counts()
    stratum = {p: ("worst %d" % w if counts[w] >= MIN_STRATUM else "pooled small strata") for p, w in worst.items()}
    rng = np.random.default_rng(SEED)
    test, dev_order = [], []
    for name in sorted(set(stratum.values())):
        members = sorted(p for p in stratum if stratum[p] == name)
        perm = [members[i] for i in rng.permutation(len(members))]
        k = int(round(TEST_SHARE * len(members)))
        test += perm[:k]
        dev_order += perm[k:]
    assign = {p: "test" for p in test}
    for i, p in enumerate(dev_order):
        assign[p] = i % FOLDS
    return {"source": "LSMA-PQR, Mendeley Data 10.17632/p3r4xd2488 v2, CC BY 4.0; scans: Sudirman et al., "
                      "Lumbar Spine MRI Dataset, 10.17632/k57fr854j2 v2, CC BY 4.0",
            "seed": SEED, "folds": FOLDS, "test_share": TEST_SHARE, "min_stratum": MIN_STRATUM,
            "rule": __doc__.split("Split:")[1].split("Usage:")[0].strip(),
            "n": {"patients": len(assign), "test": len(test), "dev": len(dev_order)},
            "strata": {k: int(v) for k, v in sorted(counts.items())},
            "assignment": {str(p): assign[p] for p in sorted(assign)}}


def assignment():
    return {int(p): v for p, v in json.load(open(SPLIT, encoding="utf-8"))["assignment"].items()}


def frame(split):
    """grades() rows of that split, with fold ('' for test)."""
    guard_test(split)
    a = assignment()
    g = grades()
    g["where"] = g.patient.map(a)
    g = g[(g["where"] == "test") if split == "test" else (g["where"] != "test")].copy()
    g["fold"] = ["" if w == "test" else int(w) for w in g["where"]]
    return g.drop(columns="where").reset_index(drop=True)


def load_crops(split, items=False, notes=False, clean=False, text=False, slip=False):
    """(crops, index) for grader.py: the index has patient (offset past SPIDER's), ivd, fold, has_t1,
    t_pfirrmann and NaN for every item; with `items`, bulging and spondylolisthesis from the
    Multi-Disorder Annotations where they say (disorder_items); with `notes`, herniation from
    LSMA-PQR's own structured notes (note_items; `clean` passes through to it); with `notes` and
    `text`, herniation instead from the radiologists' free-text notes (text_items); with `items` and
    `slip`, spondylolisthesis instead from slip_items (backed by the notes)."""
    guard_test(split)
    base = os.path.join(OUT, split)
    d = np.load(os.path.join(base, "crops.npz"), allow_pickle=False)
    idx = pd.read_csv(os.path.join(base, "crops_index.csv"))
    idx["t_pfirrmann"] = idx.pfirrmann
    for i in ITEMS:
        idx["t_" + i] = np.nan
    for wanted, table in ((items, disorder_items), (items and slip, slip_items), (notes, text_items if text else functools.partial(note_items, clean))):
        if wanted:
            it = table()
            idx = idx.drop(columns=list(it.columns[2:])).merge(it, on=["patient", "ivd"], how="left")
    idx["patient"] = idx.patient + PATIENT_OFFSET
    return d["x"], idx


NOTE_LEVEL = {1: "L5_S1", 2: "L4_L5", 3: "L3_L4"}


@functools.lru_cache(maxsize=2)
def note_items(clean=False):
    """Herniation for LSMA-PQR's graded discs, from its structured_notes.csv (the radiologists'
    reports, structured by its authors). L5-S1 and L4-L5 carry a disc type: "Central Protrusion" is a
    herniation (1), "Diffuse Bulge" is not (0). A level the notes mark not involved (<level>_Involvement
    0) has none (0); an involved level with no type, and L3-L4 when involved (it has no type column),
    are unknown. With `clean`, a patient whose notes report a herniation, protrusion or extrusion
    (Disc_Herniation, Disc_Protrusion, Disc_Extrusion) but type no level "Central Protrusion" has
    every disc unknown: the notes never say where it is, so the not-involved zeros are not trusted."""
    n = pd.read_csv(os.path.join(ROOT, "structured_notes.csv")).set_index("Patient_ID")
    rows = []
    for pid, r in n.iterrows():
        first = len(rows)
        for ivd, lv in NOTE_LEVEL.items():
            kind = r.get(f"{lv}_Disc_Type")
            if kind == "Central Protrusion":
                h = 1.0
            elif kind == "Diffuse Bulge" or r[f"{lv}_Involvement"] == 0:
                h = 0.0
            else:
                h = np.nan
            rows.append({"patient": int(pid), "ivd": ivd, "t_herniation": h})
        if clean:
            reported = any(r.get(f) == 1 for f in ("Disc_Herniation", "Disc_Protrusion", "Disc_Extrusion"))
            located = any(r.get(f"{lv}_Disc_Type") == "Central Protrusion" for lv in NOTE_LEVEL.values())
            if reported and not located:
                for row in rows[first:]:
                    row["t_herniation"] = np.nan
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def text_items():
    """Herniation for LSMA-PQR's graded discs from the radiologists' free-text notes: patient, ivd,
    t_herniation, read from lsma_text_labels.csv (lsma_text.py build); a blank is unknown (NaN)."""
    t = pd.read_csv(os.path.join(HERE, "lsma_text_labels.csv"))
    t = t[t.ivd.notna()].rename(columns={"herniation": "t_herniation"})
    return t.assign(ivd=t.ivd.astype(int), t_herniation=t.t_herniation.astype(float))[["patient", "ivd", "t_herniation"]].reset_index(drop=True)


MD_ROOT = os.environ.get("LUMBAR_MD_ROOT", os.path.join(
    paths.DATASETS, "public", "lumbar-multi-disorder", "Multi-Disorder Annotations for Lumbar Spine Mid-Sagittal Images",
    "labeled_images"))
MD_CLASSES = {"normal": "Normal IVD", "bulge": "Lumbar Disc Bulge (LDB)", "listhesis": "Spondylolisthesis"}
MD_COVER = 0.5


@functools.lru_cache(maxsize=1)
def disorder_items():
    """Two of the grader's items for LSMA-PQR's graded discs, from the Multi-Disorder Annotations
    (Mendeley 9j9hmy8kbn, CC BY 4.0: the same 515 patients' mid-sagittal pictures, pixel for pixel,
    with a region per disc and condition). A disc is marked by a region covering at least half of it
    on the T2 picture (regions cover a disc wholly or not at all: 0.9 or more for 95% of those that
    touch one): bulging 1 under a bulge region, 0 under a normal one, unknown under neither (those
    marked only degenerated); spondylolisthesis 1 under a spondylolisthesis region, else 0."""
    from PIL import Image
    rows = []
    for (seq, pid), s in sorted(slices().items()):
        if seq != "T2":
            continue
        comps, _ = disc_components(pid, s)
        masks = {k: os.path.join(MD_ROOT, c, f"Sagittal_{pid:04d}_T2_mask.png") for k, c in MD_CLASSES.items()}
        masks = {k: np.asarray(Image.open(f)) if os.path.exists(f) else None for k, f in masks.items()}
        if comps is None or all(m is None for m in masks.values()):
            continue
        for ivd, d in comps.items():
            cover = {k: 0.0 if m is None else max(((d & (m == j)).sum() / d.sum() for j in np.unique(m)[1:]), default=0.0)
                     for k, m in masks.items()}
            rows.append({"patient": pid, "ivd": ivd,
                         "t_bulging": 1.0 if cover["bulge"] >= MD_COVER else 0.0 if cover["normal"] >= MD_COVER else np.nan,
                         "t_spondylolisthesis": float(cover["listhesis"] >= MD_COVER)})
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def slip_regions():
    """({(patient, ivd): the largest cover of that disc by any one slip region}, {(patient, ivd) marked}):
    every Sagittal_<id>_T2*_mask.png of the slip class (second files included) is read, each connected
    component of each nonzero value is one region, and a region marks the one disc it covers most
    when that cover (overlap / disc pixels, as disorder_items computes it) is at least MD_COVER."""
    from PIL import Image
    from scipy import ndimage
    covers, marked = {}, set()
    folder = os.path.join(MD_ROOT, MD_CLASSES["listhesis"])
    for (seq, pid), s in sorted(slices().items()):
        if seq != "T2":
            continue
        files = sorted(glob.glob(os.path.join(folder, f"Sagittal_{pid:04d}_T2*_mask.png")))
        if not files:
            continue
        comps, _ = disc_components(pid, s)
        if comps is None:
            continue
        for f in files:
            m = np.asarray(Image.open(f))
            for v in np.unique(m)[1:]:
                lab, n = ndimage.label(m == v)
                for j in range(1, n + 1):
                    r = lab == j
                    cover = {ivd: (d & r).sum() / d.sum() for ivd, d in comps.items()}
                    for ivd, c in cover.items():
                        covers[(pid, ivd)] = max(covers.get((pid, ivd), 0.0), float(c))
                    best = max(cover, key=cover.get)
                    if cover[best] >= MD_COVER:
                        marked.add((pid, best))
    return covers, marked


def slip_covers():
    return slip_regions()[0]


def structured_slip():
    """{patient: the structured notes' Spondylolisthesis (0/1)}."""
    n = pd.read_csv(os.path.join(ROOT, "structured_notes.csv"))
    return {int(p): int(v) for p, v in zip(n.Patient_ID, n.Spondylolisthesis.fillna(0))}


@functools.lru_cache(maxsize=1)
def slip_items():
    """Spondylolisthesis for the graded discs of every patient disorder_items covers (patient, ivd,
    t_spondylolisthesis), backed by the radiologists' notes (docs/efficiency/README.md, "Slip labels
    the notes back up", idea 22). A slip region marks a disc (slip_regions). The notes report a slip
    when the structured Spondylolisthesis column is 1 or the free text says so (lsma_text.slip_read);
    they place it at the levels that text names, or nowhere. Rules: 1 for a marked disc when the notes
    report a slip and name no level or this disc's; NaN (unknown) for a marked disc whose notes report
    no slip or name only other levels, and for an unmarked disc at a level the notes name as slipped;
    0 everywhere else."""
    import lsma_text
    base = disorder_items()[["patient", "ivd"]]
    marked = slip_regions()[1]
    structured, text = structured_slip(), lsma_text.notes()
    read = {}
    rows = []
    for pid, ivd in zip(base.patient, base.ivd):
        if pid not in read:
            reported, levels = lsma_text.slip_read(text.get(pid, ""))
            read[pid] = (reported or structured.get(pid, 0) == 1, levels)
        reported, levels = read[pid]
        level = NOTE_LEVEL[ivd].replace("_", "-")
        named = bool(levels)
        if (pid, ivd) in marked:
            t = 1.0 if reported and (not named or level in levels) else np.nan
        else:
            t = np.nan if reported and level in levels else 0.0
        rows.append({"patient": pid, "ivd": ivd, "t_spondylolisthesis": t})
    return pd.DataFrame(rows)


def correlation(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(((a - a.mean()) / (a.std() + 1e-9)) @ ((b - b.mean()) / (b.std() + 1e-9)) / a.size)


def to_nib(img):
    """(array, affine) of a SimpleITK image as nibabel reads the NIfTI SimpleITK would write: voxel
    axes x, y, z, world RAS (SimpleITK's is LPS)."""
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    aff = np.eye(4)
    aff[:3, :3] = np.array(img.GetDirection()).reshape(3, 3) * np.array(img.GetSpacing())
    aff[:3, 3] = img.GetOrigin()
    return a, np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff


def read_series(pid, seq, slice_no):
    """The sagittal series of `seq` whose file numbered `slice_no` is LSMA-PQR's picture: (SimpleITK
    volume, one file per position in order through the stack; the picture's place in it; series
    name; r), or None with r when no series matches."""
    import SimpleITK as sitk
    from PIL import Image
    png = np.asarray(Image.open(os.path.join(ROOT, "images_png", f"sag_{seq}_{pid:04d}_S{slice_no}.png")).convert("L"))
    best = (-1.0, None, None)
    for d in sorted(glob.glob(os.path.join(SCANS, "01_MRI_Data", f"{pid:04d}", "*", seq + "_TSE_SAG*"))):
        for f in glob.glob(os.path.join(d, f"*_{slice_no:03d}.ima")):
            s = sitk.GetArrayFromImage(sitk.ReadImage(f))[0].astype(np.float32)
            r = correlation(np.asarray(Image.fromarray(s).resize((PNG, PNG), Image.BILINEAR)), png)
            if r > best[0]:
                best = (r, d, f)
    r, d, f = best
    if r < MATCH_R:
        return None, r
    reader = sitk.ImageFileReader()
    where = {}
    for g in sorted(glob.glob(os.path.join(d, "*.ima"))):
        reader.SetFileName(g)
        reader.ReadImageInformation()
        normal = np.array(reader.GetDirection()).reshape(3, 3)[:, 2]
        where.setdefault(round(float(np.dot(reader.GetOrigin(), normal)), 2), g)   # the first file at a position
    order = sorted(where)
    files = [where[k] for k in order]
    reader.SetFileName(f)
    reader.ReadImageInformation()
    normal = np.array(reader.GetDirection()).reshape(3, 3)[:, 2]
    k = order.index(round(float(np.dot(reader.GetOrigin(), normal)), 2))
    return (sitk.ReadImage(files), k, os.path.basename(d)), r


def disc_components(pid, slice_no):
    """The 384 x 384 mask's disc components, one per level: {ivd: boolean 384 x 384}, or a reason."""
    from PIL import Image
    from scipy import ndimage
    m = np.asarray(Image.open(os.path.join(ROOT, "masks_indexed", f"sag_T2_{pid:04d}_S{slice_no}.png")))
    comp, _ = ndimage.label(m == DISC)
    h = heights().loc[pid]
    out, rows = {}, {}
    for col, ivd in LEVEL_IVD.items():
        (x1, y1), (x2, y2) = [map(float, p.split(",")) for p in h[col + "_Coord"].split(";")]
        x, y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        c = comp[y, x]
        if c == 0:                                           # the midpoint between two edges: nearest disc pixel
            yy, xx = np.nonzero(comp)
            j = np.argmin((yy - y) ** 2 + (xx - x) ** 2)
            if (yy[j] - y) ** 2 + (xx[j] - x) ** 2 > 10 ** 2:
                return None, f"{col}: no disc within 10 px of its height points"
            c = comp[yy[j], xx[j]]
        out[ivd] = comp == c
        rows[ivd] = np.nonzero(out[ivd])[0].mean()
    if not rows[1] > rows[2] > rows[3] or any((out[a] & out[b]).any() for a, b in ((1, 2), (2, 3), (1, 3))):
        return None, "levels not three distinct discs running D3, D4, D5 downward"
    return out, ""


def patient_crops(job):
    """One patient's crops: (list of (crop, index row), reason left out or '')."""
    import SimpleITK as sitk
    from PIL import Image
    from crops import Series, crop
    from prepare import disc_id
    pid, fold, slice_t2, slice_t1, graded = job
    discs, why = disc_components(pid, slice_t2)
    if discs is None:
        return [], why
    t2, r2 = read_series(pid, "T2", slice_t2)
    if t2 is None:
        return [], f"T2 picture matches no slice (best r {r2:.3f})"
    img, k, name2 = t2
    lab = np.zeros(sitk.GetArrayFromImage(img).shape, np.uint16)   # (slices, rows, columns)
    for ivd, m in discs.items():
        big = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((lab.shape[2], lab.shape[1]), Image.NEAREST)) > 0
        lab[k][big] = disc_id(ivd)
    lab_img = sitk.GetImageFromArray(lab)
    lab_img.CopyInformation(img)
    a2, aff2 = to_nib(sitk.DICOMOrient(img, "PIR"))
    al, affl = to_nib(sitk.DICOMOrient(lab_img, "PIR"))
    assert np.allclose(aff2, affl) and "".join(__import__("nibabel").aff2axcodes(aff2)) == "PIR"
    t1, r1 = read_series(pid, "T1", slice_t1) if slice_t1 else (None, float("nan"))
    t1s = Series(None, *to_nib(sitk.DICOMOrient(t1[0], "PIR"))) if t1 else None
    t2s = Series(a=a2, aff=aff2)
    out = []
    for ivd, grade, note in graded:
        x, has_t1 = crop(t2s, t1s, al, disc_id(ivd))
        if x is None:
            continue
        out.append((x, {"patient": pid, "ivd": ivd, "fold": fold, "localisation": "lsma_mask", "has_t1": int(has_t1),
                        "pfirrmann": grade, "note_disc_type": note, "t2_series": name2, "r_t2": round(r2, 4),
                        "t1_series": t1[2] if t1 else "", "r_t1": round(r1, 4), "mask_pixels": int(discs[ivd].sum())}))
    return out, "" if out else "no disc crop"


def slices():
    """{(seq, patient): the slice number of its LSMA-PQR picture}."""
    out = {}
    for f in os.listdir(os.path.join(ROOT, "images_png")):
        _, seq, pid, s = f[:-4].split("_")
        out[(seq, int(pid))] = int(s[1:])
    return out


def make_crops(split, workers=max(2, (os.cpu_count() or 8) // 4)):
    """Crops of every graded disc of the split's patients, a patient at a time across `workers`
    (about 0.4 s a patient each; run it in the background)."""
    g = frame(split)
    notes = pd.read_csv(os.path.join(ROOT, "structured_notes.csv")).set_index("Patient_ID")
    note = {1: "L5_S1_Disc_Type", 2: "L4_L5_Disc_Type"}
    s = slices()
    def disc_type(p, ivd):                                   # the report's disc type, L4-L5 and L5-S1 only
        v = notes.at[p, note[ivd]] if ivd in note else None
        return str(v) if pd.notna(v) else ""
    jobs = [(int(p), "" if split == "test" else int(d.fold.iloc[0]), s[("T2", p)], s.get(("T1", p)),
             [(int(r.ivd), int(r.pfirrmann), disc_type(p, r.ivd)) for r in d.itertuples()])
            for p, d in g.groupby("patient")]
    t0 = time.time()
    X, rows, left = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for job, (crops_, why) in zip(jobs, ex.map(patient_crops, jobs, chunksize=4)):
            for x, row in crops_:
                X.append(x)
                rows.append(row)
            if why:
                left.append({"patient": job[0], "reason": why})
    base = os.path.join(OUT, split)
    os.makedirs(base, exist_ok=True)
    np.savez_compressed(os.path.join(base, "crops.npz"), x=np.stack(X))      # the index is crops_index.csv
    idx = pd.DataFrame(rows)
    idx.to_csv(os.path.join(base, "crops_index.csv"), index=False)
    pd.DataFrame(left, columns=["patient", "reason"]).to_csv(os.path.join(base, "left_out.csv"), index=False)
    print(f"lsma {split}: {len(X)} crops of {idx.patient.nunique()} patients ({len(jobs)} in the split, "
          f"{len(left)} left out: left_out.csv), no T1 {int((idx.has_t1 == 0).sum())}, {time.time() - t0:.0f} s")


def score_test(versions, headline):
    """LSMA-PQR's test split, opened once: each grader version's grades (1 plus its ensemble's
    cumulative logits above 0, as compare.py reads them) against the grades, exact, within one and
    quadratic weighted kappa (the authors' 0.51 is that kappa) with patient-bootstrap intervals.
    `headline` is the version named before opening as the one the scores-to-beat line reports; the
    others are reported beside it, never swapped in. A version may be "103+104": those runs' models
    averaged. Refuses a second opening (test/opened.json)."""
    guard_test("test")
    from grader import load_grader, logits
    from score import bootstrap, m_exact, m_qwk, m_within_one
    from core import ledger
    opened = os.path.join(OUT, "test", "opened.json")
    if os.path.exists(opened):
        raise SystemExit(f"LSMA-PQR's test split was opened already: {opened}")
    if headline not in versions:
        raise SystemExit("the headline version must be one of those scored")
    t0 = time.time()
    x, idx = load_crops("test")
    t = idx.t_pfirrmann.to_numpy(int)
    pts = idx.patient.to_numpy()
    out = {"opened": time.strftime("%Y-%m-%d %H:%M:%S"), "headline": headline, "discs": int(len(idx)),
           "patients": int(idx.patient.nunique()), "versions": {}}
    json.dump(out, open(opened, "w"), indent=1)                 # written first: a crash still counts as the opening
    numbers = {}
    for v in versions:
        nets = [n for part in str(v).split("+") for n in load_grader(int(part))[0]]
        g = 1 + (logits(nets, x, idx.has_t1.to_numpy())[:, :4] > 0).sum(1)
        r = {}
        for name, fn in (("exact", m_exact), ("within_one", m_within_one), ("qwk", m_qwk)):
            lo, hi, _ = bootstrap(pts, fn, t, g)
            r[name] = [round(fn(t, g), 4), round(lo, 4), round(hi, 4)]
            numbers[f"v{v}.{name}"] = (fn(t, g), "")
        out["versions"][str(v)] = r
        print(f"v{v}{' (headline)' if v == headline else ''}: " + ", ".join(
            f"{k} {a:.4f} ({lo:.4f} to {hi:.4f})" for k, (a, lo, hi) in r.items()))
    json.dump(out, open(opened, "w"), indent=1)
    ledger.write("experiment", f"spine: LSMA-PQR test, opened once (headline v{headline})", time.time() - t0, numbers,
                 settings={"versions": ",".join(map(str, versions)), "headline": headline},
                 note="modalities/spine_mri/eval/lsma.py score")
    print(f"LSMA-PQR test: {out['discs']} discs of {out['patients']} patients; {opened}")


@functools.lru_cache(None)
def heights():
    return pd.read_csv(os.path.join(ROOT, "IVDHeights.csv")).set_index("Patient_ID")


if __name__ == "__main__":
    if sys.argv[1:2] == ["split"]:
        if os.path.exists(SPLIT):
            raise SystemExit("%s exists; a locked split is never regenerated" % SPLIT)
        s = make_split()
        with open(SPLIT, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
        print("lsma split:", s["n"], "strata", s["strata"])
    elif sys.argv[1:2] == ["crops"] and sys.argv[2:3] in (["dev"], ["test"]):
        make_crops(sys.argv[2], *map(int, sys.argv[3:4]))
    elif sys.argv[1:2] == ["score"] and len(sys.argv) >= 4:
        score_test(sys.argv[3:], sys.argv[2])
    else:
        raise SystemExit(__doc__)
