"""Where X-ray working files go, and the one writer for prediction files.

Every method writes `work/eval/xray/pred_<method>_<dataset>_<split>.csv` with exactly the columns
METRICS_SPEC.md names, in this order. `collect()` concatenates them into `predictions.csv`, which is
the only prediction file score.py and recompute.py read."""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from core import paths  # noqa: E402

WORK = os.path.join(paths.WORK, "eval", "xray")
CACHE = os.path.join(WORK, "cache")
BUNDLES = os.path.join(WORK, "reader", "bundles")
COLUMNS = ["method", "dataset", "split", "fold", "patient", "image", "in_scope", "drop_reason",
           "fracture_pred", "fracture_score"]
METHODS = ["prevalence", "meta", "features", "reader", "cnn"]
UNREADABLE = "unreadable image"


def path(method, dataset, split, folder=None):
    return os.path.join(folder or WORK, "pred_%s_%s_%s.csv" % (method, dataset, split))


def write(method, dataset, split, rows, folder=None):
    """rows: dicts with at least patient, image, in_scope; the rest default to empty."""
    if method not in METHODS:
        raise SystemExit("unknown method %r; METRICS_SPEC.md lists %s" % (method, METHODS))
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df["method"], df["dataset"], df["split"] = method, dataset, split
    df = df[COLUMNS]
    os.makedirs(folder or WORK, exist_ok=True)
    out = path(method, dataset, split, folder)
    df.to_csv(out, index=False)
    n = int((df.in_scope == 1).sum())
    print("%s: %d rows, %d in scope, %d with a prediction"
          % (os.path.basename(out), len(df), n, int(df.fracture_pred.astype(str).str.len().gt(0).sum())),
          flush=True)
    return out


def collect(folder=None):
    folder = folder or WORK
    parts = sorted(glob.glob(os.path.join(folder, "pred_*_*_*.csv")))
    if not parts:
        raise SystemExit("no pred_*.csv in %s" % folder)
    df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in parts], ignore_index=True)
    out = os.path.join(folder, "predictions.csv")
    df[COLUMNS].to_csv(out, index=False)
    print("predictions.csv: %d rows from %s" % (len(df), [os.path.basename(p) for p in parts]), flush=True)
    return out


if __name__ == "__main__":
    collect()
