"""One command per split: SPIDER -> NIfTI (prepare.py) -> TotalSpineSeg -> Dice matching (match.py)
-> baseline (a) rules (rules.py) -> work/eval/predictions.csv (every pred_*.csv present).
It needs a GPU and takes hours: python pipeline/eval/run_split.py dev.
The test split is run once, by final_eval.py, never by hand.
Usage: python run_split.py <dev|test> [--skip-prepare] [--skip-tss] | --collect-only"""
import glob
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import match  # noqa: E402
import prepare  # noqa: E402
from spider import guard_test  # noqa: E402
import rules  # noqa: E402
from tss import segment_chunked  # noqa: E402

TSS_DATA = os.path.normpath(os.path.join(HERE, "..", "..", "work", "tss_data"))
EVAL = os.path.dirname(prepare.WORK)


def collect():
    parts = sorted(glob.glob(os.path.join(EVAL, "pred_*_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    out = os.path.join(EVAL, "predictions.csv")
    df.to_csv(out, index=False)
    print("predictions.csv:", len(df), "rows from", [os.path.basename(p) for p in parts])


def main(split, skip_tss=False, skip_prepare=False):
    guard_test(split)
    t0 = time.time()
    if not skip_prepare:
        prepare.main(split)
    print(f"[{time.time() - t0:.0f}s] prepared", flush=True)
    base = os.path.join(prepare.WORK, split)
    if not skip_tss:
        t2 = os.path.join(base, "t2")
        segment_chunked(t2, os.path.join(base, "tss"), TSS_DATA)
        print(f"[{time.time() - t0:.0f}s] TotalSpineSeg done on {len(os.listdir(t2))} volumes", flush=True)
    match.main(split)
    rules.main(split)
    collect()
    print(f"[{time.time() - t0:.0f}s] {split} finished", flush=True)


if __name__ == "__main__":
    if "--collect-only" in sys.argv:
        collect()
    else:
        main(sys.argv[1], "--skip-tss" in sys.argv, "--skip-prepare" in sys.argv)
