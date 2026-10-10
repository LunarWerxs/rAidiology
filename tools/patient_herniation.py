"""Herniation per patient on SPIDER's development patients, from grader runs' out-of-fold logits.

A patient's score is the highest herniation logit among its discs (nested ensembles); its truth is
whether any disc is herniated. Prints the per-patient AUROC and, with the threshold slid so it clears
as many patients without a herniation as Gemini 3.1 Pro did (74 of 100, VS_DOCTORS), how many with
one it catches (Gemini: 58). Runs that do not exist yet are skipped.
Usage: python tools/patient_herniation.py <version> [<version> ...]"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "spine_mri", "eval"))
from core import ledger  # noqa: E402
from score import m_auroc  # noqa: E402
from spider import ITEMS  # noqa: E402

MODELS = os.path.join(os.path.dirname(ROOT), "Models")
CLEAR = 0.74


def per_patient(version, which="nested"):
    d = np.load(os.path.join(MODELS, f"disc-grader-v{version}", "oof_logits.npz"), allow_pickle=False)
    sp = d["collection"] == "spider" if "collection" in d.files else np.ones(len(d["y"]), bool)
    j = 4 + ITEMS.index("herniation")
    df = pd.DataFrame({"patient": d["patient"][sp], "s": d[which][sp, j], "t": d["y"][sp, j]})
    return df.groupby("patient").agg(s=("s", "max"), t=("t", "max"))


def main(versions):
    with ledger.run("measurement", "herniation per patient " + " ".join("v%s" % v for v in versions)) as led:
        led.settings_from({"versions": versions, "clear": CLEAR})
        for v in versions:
            if not os.path.exists(os.path.join(MODELS, f"disc-grader-v{v}", "oof_logits.npz")):
                continue
            pt = per_patient(v)
            neg = np.sort(pt.s[pt.t == 0].values)
            thr = neg[int(np.ceil(CLEAR * len(neg))) - 1]
            auc = m_auroc(pt.t.values.astype(int), pt.s.values)
            cleared, caught = (pt.s[pt.t == 0] <= thr).mean(), (pt.s[pt.t == 1] > thr).mean()
            print("v%s: herniation per patient, %d patients (%d with one): AUROC %.3f; clearing %.2f it catches %.2f"
                  % (v, len(pt), int(pt.t.sum()), auc, cleared, caught))
            k = "v%s" % v
            led.number(k + ".patients", len(pt), "count").number(k + ".with_herniation", int(pt.t.sum()), "count")
            led.number(k + ".auroc", auc).number(k + ".cleared", cleared).number(k + ".sensitivity", caught)


if __name__ == "__main__":
    main(sys.argv[1:])
