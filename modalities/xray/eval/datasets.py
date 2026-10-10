"""The two X-ray datasets: their index files, the truth column, where each image is, the
pre-declared strata, and the guard that keeps the test split shut.

Both are CC BY 4.0 (LICENSING.md) and live outside the repository, under <Datasets>/public/.

  grazpedwri  GRAZPEDWRI-DX, 20,327 paediatric wrist radiographs of 6,091 patients, 16-bit PNG.
              Truth: `fracture_visible`. The primary set, because the patient identifier is
              published, so the split can be by patient.
  fracatlas   FracAtlas, 4,083 musculoskeletal radiographs (hand, leg, hip, shoulder), 8-bit JPEG.
              Truth: `fractured`. No patient identifier is published, so each image is its own
              resampling unit and the scorecard says so.
  orthofrac   OrthoFrac-XR, 1,493 limb radiographs of about 1,300 patients (Bangladesh, ages 1 to
              86, median 11), JPEG, each validated by an orthopaedic surgeon against its report.
              Truth: `fracture_type` distal or proximal (558) against non_fracture (574). The 349
              post_fracture images (healing, callus, hardware) are left out: whether "a fracture is
              visible" on them has no single answer; so are 12 that are another collection's
              pictures (ORTHOFRAC_DUPLICATES). No patient identifier is published either.
              A training collection on trial (added 2026-10-03), not one the shipped model reads.
  bonefract   BoneFract, 47,931 radiographs of 47,922 patients (arm, wrist, hand, pelvis, thigh,
              lower leg, foot; Nawabshah, Pakistan), 512 x 512 PNG, letterboxed by its authors.
              Truth: the `Positive` folder. No index file is published, so one is written beside it
              on first use (bonefract_index). Its labeller is not stated, and a look at the pictures
              finds chest films under "Arm" and old hardware under "Positive": a noisy collection,
              so it is on trial like orthofrac, judged only by what it does to the other two.

`NAMES` are the two collections every shipped model was trained and scored on; `ALL` adds the ones
on trial, which a trainer takes only when its settings name them.

Column names, values and strata are the contract in METRICS_SPEC.md; nothing here may change them
without changing that document first."""
import functools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from core import paths  # noqa: E402

SEED = 20261001
PUBLIC = os.path.join(paths.DATASETS, "public")

SETS = {
    "grazpedwri": {
        "root": os.environ.get("GRAZPEDWRI_ROOT", os.path.join(PUBLIC, "grazpedwri-dx")),
        "index": ("dataset.csv", "utf-8-sig"),
        "key": "filestem",
        "truth": "fracture_visible",
        "patient": "patient_id",
        "images": ("yolov5/images", ".png"),
        "doi": "10.6084/m9.figshare.14825193",
        "cite": ("Nagy, Janisch, Hrzic, Sorantin, Tschauner, 'A pediatric wrist trauma X-ray dataset "
                 "(GRAZPEDWRI-DX) for machine learning', Scientific Data 9, 222 (2022), "
                 "doi:10.1038/s41597-022-01328-z; data doi:10.6084/m9.figshare.14825193, CC BY 4.0"),
        # stratum -> (column, "is"|"is not", value). `all` is every in-scope row.
        "strata": {"initial_exam": ("initial_exam", "is", "1"),
                   "no_cast": ("cast", "is not", "1"),
                   "certain": ("diagnosis_uncertain", "is not", "1")},
        "stratify_split": ("age_band", "sex", "any_fracture"),
    },
    "fracatlas": {
        "root": os.environ.get("FRACATLAS_ROOT", os.path.join(PUBLIC, "fracatlas", "FracAtlas")),
        "index": ("dataset.csv", "utf-8"),
        "key": "image_id",
        "truth": "fractured",
        "patient": None,                     # not published: the image is its own unit
        "images": ("images", ""),            # under images/Fractured or images/Non_fractured
        "doi": "10.6084/m9.figshare.22363012",
        "cite": ("Abedeen, Rahman, Prottyasha, Ahmed, Chowdhury, Shatabda, 'FracAtlas: A Dataset for "
                 "Fracture Classification, Localization and Segmentation of Musculoskeletal "
                 "Radiographs', Scientific Data 10, 521 (2023), doi:10.1038/s41597-023-02432-4; "
                 "data doi:10.6084/m9.figshare.22363012, CC BY 4.0"),
        "strata": {"hand": ("hand", "is", "1"), "leg": ("leg", "is", "1"),
                   "hip": ("hip", "is", "1"), "shoulder": ("shoulder", "is", "1")},
        "stratify_split": ("body_part", "view", "any_fracture"),
    },
    "orthofrac": {
        "root": os.environ.get("ORTHOFRAC_ROOT", os.path.join(PUBLIC, "orthofrac-xr", "OrthoFrac-XR")),
        "index": ("Bone_metadata.csv", "utf-8-sig"),
        "key": "id",
        "truth": "fracture_type",            # distal or proximal is 1, non_fracture 0 (see index)
        "patient": None,                     # not published: the image is its own unit
        "images": ("", ""),                  # under <fracture_type>_fracture/ or non_fracture/
        "doi": "10.6084/m9.figshare.32021085",
        "cite": ("OrthoFrac-XR: A Clinically Validated Multimodal High-Resolution X-ray Imaging Dataset "
                 "for Bone Fracture Detection and Localization, Health Informatics Research Lab, "
                 "Daffodil International University (2025); data doi:10.6084/m9.figshare.32021085, "
                 "CC BY 4.0"),
        "strata": {"adult": ("age_group", "is", "adult"), "child": ("age_group", "is", "child"),
                   "upper_limb": ("limb", "is", "upper"), "lower_limb": ("limb", "is", "lower")},
        "stratify_split": ("age_group", "limb", "any_fracture"),
    },
    "bonefract": {
        "root": os.environ.get("BONEFRACT_ROOT", os.path.join(PUBLIC, "bonefract")),
        "index": ("rAidiology_index.csv", "utf-8"),   # written by bonefract_index() on first use
        "key": "stem",
        "truth": "label",                    # Positive is 1 (see index)
        "patient": "patient_id",
        "images": ("BoneFract A Bone Fracture Dataset", ".png"),
        "doi": "10.17632/4cr7f2359x.1",
        "cite": ("Noor R, Ali A. BoneFract: A Bone Fracture Dataset. Mendeley Data v1 (2026); "
                 "doi:10.17632/4cr7f2359x.1, CC BY 4.0"),
        "strata": {"upper_limb": ("limb", "is", "upper"), "lower_limb": ("limb", "is", "lower"),
                   "pelvis": ("limb", "is", "pelvis")},
        "stratify_split": ("limb", "any_fracture"),
    },
}
NAMES = ["fracatlas", "grazpedwri"]
ALL = sorted(SETS)
ORTHOFRAC_TRUTH = {"distal": 1, "proximal": 1, "non_fracture": 0}   # post_fracture: out of scope
# OrthoFrac-XR pictures that are another collection's: nine exact copies of FracAtlas pictures
# (r 1.000) and three at r 0.97 or more with a FracAtlas or GRAZPEDWRI-DX one, the top 1% of best
# matches (tools/xray_duplicates.py, 2026-10-03). Left out so no picture is on both sides of a split.
ORTHOFRAC_DUPLICATES = {"distal_184", "distal_185", "distal_186", "distal_187", "distal_188",
                        "distal_189", "distal_190", "distal_257", "non_fracture_209", "non_fracture_3",
                        "proximal_133", "proximal_134"}
UPPER_LIMB = {"elbow", "ulna", "radius", "humerus", "carpal", "shoulder", "metacarpal"}
BONEFRACT_LIMB = {"Arm": "upper", "Hand": "upper", "wrist": "upper", "Foot": "lower", "Lower leg": "lower",
                  "Thigh": "lower", "pelvis": "pelvis"}
BONEFRACT_EXCLUDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bonefract_excluded.csv")


@functools.lru_cache(maxsize=1)
def bonefract_excluded():
    """Stems left out of BoneFract's index: its copies and other collections' pictures, written by
    tools/bonefract_exclusions.py from tools/xray_duplicates.py's pairs."""
    return frozenset(pd.read_csv(BONEFRACT_EXCLUDED, dtype=str, keep_default_na=False).stem)


@functools.lru_cache(maxsize=1)
def bonefract_folders():
    """stem -> its folders (given split, region, patient, label): a stem's region is not always its
    folder's (Arm_patient26944_Positive_001 is filed under wrist)."""
    idx = pd.read_csv(bonefract_index(), dtype=str, keep_default_na=False)
    return {r.stem: (r.given_split, r.region, r.patient_id, r.label) for r in idx.itertuples()}


def bonefract_index():
    """Write BoneFract's index (it publishes none): one row per PNG, from its folders."""
    s = SETS["bonefract"]
    out = os.path.join(s["root"], s["index"][0])
    if os.path.exists(out):
        return out
    base = os.path.join(s["root"], s["images"][0])
    rows = []
    for given in ("train", "valid", "test"):
        for region in sorted(os.listdir(os.path.join(base, given))):
            for pid in sorted(os.listdir(os.path.join(base, given, region))):
                for label in sorted(os.listdir(os.path.join(base, given, region, pid))):
                    for f in sorted(os.listdir(os.path.join(base, given, region, pid, label))):
                        if f.lower().endswith(".png"):
                            rows.append({"stem": f[:-4], "patient_id": pid, "region": region, "label": label,
                                         "given_split": given})
    tmp = out + ".part"
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, out)
    return out
AGE_BANDS = [(0, 6, "0-5"), (6, 11, "6-10"), (11, 15, "11-14"), (15, 100, "15-19")]


def guard_test(split):
    """The test split is opened once, by final_eval.py, which sets RAIDIOLOGY_FINAL_EVAL=1."""
    if split == "test" and os.environ.get("RAIDIOLOGY_FINAL_EVAL") != "1":
        raise SystemExit("The test split is opened only by final_eval.py (once).")


def root(dataset):
    return SETS[dataset]["root"]


def age_band(age):
    a = float(age)
    for lo, hi, name in AGE_BANDS:
        if lo <= a < hi:
            return name
    return AGE_BANDS[-1][2]


def index(dataset):
    """The dataset's own index, with the columns every other script agrees on added:

    image (the key), patient (the resampling unit), truth (0/1), and the facts the split is
    stratified by. Every original column is kept, as a string, so the strata read the source."""
    s = SETS[dataset]
    name, enc = s["index"]
    if dataset == "bonefract":
        bonefract_index()
    df = pd.read_csv(os.path.join(s["root"], name), encoding=enc, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if dataset == "orthofrac":
        keep = df.fracture_type.str.strip().isin(ORTHOFRAC_TRUTH) & ~df.id.str.strip().isin(ORTHOFRAC_DUPLICATES)
        df = df[keep].reset_index(drop=True)
    if dataset == "bonefract":
        df = df[~df.stem.isin(bonefract_excluded())].reset_index(drop=True)
    df["image"] = df[s["key"]].str.strip()
    if dataset == "orthofrac":
        df["truth"] = df.fracture_type.str.strip().map(ORTHOFRAC_TRUTH).astype(int)
    elif dataset == "bonefract":
        df["truth"] = (df.label == "Positive").astype(int)
    else:
        df["truth"] = (df[s["truth"]].str.strip() == "1").astype(int)
    if s["patient"]:
        df["patient"] = df[s["patient"]].str.strip()
    else:
        df["patient"] = df["image"]
    if dataset == "grazpedwri":
        df["age_band"] = df.age.map(age_band)
        df["sex"] = df.gender.str.strip()
    elif dataset == "orthofrac":
        # Two ages carry no number; they count as children, the commoner group (1,025 of 1,491).
        age = pd.to_numeric(df.age.str.strip(), errors="coerce")
        df["age_group"] = ["adult" if a >= 18 else "child" for a in age.fillna(0)]
        df["limb"] = ["upper" if b.strip() in UPPER_LIMB else "lower" for b in df.bone_type]
        df["sex"] = df.gender.str.strip()
    elif dataset == "bonefract":
        df["limb"] = df.region.map(BONEFRACT_LIMB)
        df["sex"] = ""
    else:
        parts = ["hip", "shoulder", "leg", "hand"]          # first flag in this order; else "other"
        df["body_part"] = [next((p for p in parts if r[p] == "1"), "other") for _, r in df.iterrows()]
        views = ["frontal", "lateral", "oblique"]
        df["view"] = [next((v for v in views if r[v] == "1"), "other") for _, r in df.iterrows()]
        df["sex"] = ""
    any_fx = df.groupby("patient").truth.transform("max")
    df["any_fracture"] = any_fx.map({0: "no", 1: "yes"})
    if df.image.duplicated().any():
        raise SystemExit("%s: duplicate image keys in its index" % dataset)
    return df


def strata(dataset):
    """{stratum name: a boolean Series on index(dataset)}, `all` first, exactly as METRICS_SPEC.md."""
    df = index(dataset)
    out = {"all": pd.Series(True, index=df.index)}
    for name, (col, op, val) in SETS[dataset]["strata"].items():
        v = df[col].str.strip()
        out[name] = (v == val) if op == "is" else (v != val)
    return out


def image_path(dataset, image):
    """Where that image's file is, or None when it is not on disk."""
    s = SETS[dataset]
    sub, ext = s["images"]
    if dataset == "fracatlas":
        for d in ("Fractured", "Non_fractured"):
            p = os.path.join(s["root"], sub, d, image)
            if os.path.exists(p):
                return p
        return None
    if dataset == "orthofrac":
        folder = "non_fracture" if image.startswith("non_fracture") else image.rsplit("_", 1)[0] + "_fracture"
        for e in (".jpg", ".jpeg"):
            p = os.path.join(s["root"], folder, image + e)
            if os.path.exists(p):
                return p
        return None
    if dataset == "bonefract":
        folders = bonefract_folders().get(image)
        p = os.path.join(s["root"], sub, *folders, image + ext) if folders else ""
        return p if os.path.exists(p) else None
    p = os.path.join(s["root"], sub, image + ext)
    return p if os.path.exists(p) else None


if __name__ == "__main__":
    for d in ALL:
        df = index(d)
        st = strata(d)
        print("%-11s %5d images, %5d units, %5d with a fracture (%.3f)"
              % (d, len(df), df.patient.nunique(), int(df.truth.sum()), df.truth.mean()))
        print("            strata: " + ", ".join("%s %d" % (k, int(v.sum())) for k, v in st.items()))
        missing = sum(1 for i in df.image if image_path(d, i) is None)
        print("            image files missing: %d" % missing)
