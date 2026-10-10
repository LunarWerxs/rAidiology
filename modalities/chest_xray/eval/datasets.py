"""The chest X-ray dataset: its index, the truth columns, where each picture is, and the guard that
keeps the test split shut.

NIH ChestX-ray14 (LICENSING.md), 112,120 frontal chest X-rays of 30,805 patients, lives outside the
repository, under <Datasets>/public/nih-chestxray14/ (modalities/chest_xray/fetch_data.py puts it
there). `NIH_CXR_ROOT` points somewhere else.

Why each choice:
- One row per picture, sorted by image name, so a row number is stable and the picture cache
  (prepare.py) can be keyed by it.
- NIH's own split is kept as `nih_split` (train_val_list.txt and test_list.txt). It is by patient,
  and it is the split every published result uses, so ours compares with theirs.
- One 0/1 column per finding, spelt as NIH spells it (Pleural_Thickening), in NIH's order. `Abnormal`
  is the question a radiologist is asked first: 1 unless the label is exactly "No Finding".
- The index stops if a picture is in neither list or in both, rather than guessing a split.

Column names and values are the contract the other chest X-ray scripts read; change them here and
in those together."""
import functools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from core import paths  # noqa: E402

ROOT = os.environ.get("NIH_CXR_ROOT", os.path.join(paths.DATASETS, "public", "nih-chestxray14"))
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
LABELS = FINDINGS + ["Abnormal"]
NO_FINDING = "No Finding"


def guard_test(split):
    """The test split is opened once, by the final evaluation, which sets RAIDIOLOGY_FINAL_EVAL=1."""
    if split == "test" and os.environ.get("RAIDIOLOGY_FINAL_EVAL") != "1":
        raise SystemExit("The test split is opened only by final_eval.py (once).")


def _names(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@functools.lru_cache(maxsize=1)
def index():
    """One row per picture, sorted by image: image, patient, follow_up, age, sex, view, nih_split,
    then an int8 column for each of LABELS."""
    raw = pd.read_csv(os.path.join(ROOT, "Data_Entry_2017_v2020.csv"), dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    df = pd.DataFrame({
        "image": raw["Image Index"].str.strip(),
        "patient": raw["Patient ID"].astype(int),
        "follow_up": raw["Follow-up #"].astype(int),
        "age": raw["Patient Age"].astype(int),
        "sex": raw["Patient Sex"].str.strip(),
        "view": raw["View Position"].str.strip(),
    })
    if df.image.duplicated().any():
        raise SystemExit("nih-chestxray14: duplicate image names in Data_Entry_2017_v2020.csv")
    train_val, test = set(_names("train_val_list.txt")), set(_names("test_list.txt"))
    if train_val & test:
        raise SystemExit("nih-chestxray14: %d pictures are in both list files" % len(train_val & test))
    df["nih_split"] = df.image.map(lambda i: "train_val" if i in train_val else ("test" if i in test else None))
    if df.nih_split.isna().any():
        raise SystemExit("nih-chestxray14: %d pictures are in neither list file" % int(df.nih_split.isna().sum()))
    found = raw["Finding Labels"].str.strip()
    parts = found.str.split("|")
    for name in FINDINGS:
        df[name] = parts.map(lambda p, n=name: n in p).astype("int8")
    df["Abnormal"] = (found != NO_FINDING).astype("int8")
    return df.sort_values("image").reset_index(drop=True)


def image_path(image):
    return os.path.join(ROOT, "images", image)


if __name__ == "__main__":
    df = index()
    print("%d images, %d patients, %d test images, %d No Finding"
          % (len(df), df.patient.nunique(), int((df.nih_split == "test").sum()), int((df.Abnormal == 0).sum())))
    print("  " + ", ".join("%s %d" % (n, int(df[n].sum())) for n in FINDINGS))
    print("  image files missing: %d" % sum(1 for i in df.image if not os.path.exists(image_path(i))))
