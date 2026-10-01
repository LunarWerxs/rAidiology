"""Lumbar "compared with N people" references, shown only where a commercial-OK cohort with age and
sex can match the patient.

Cohort: SPIDER development patients (CC BY 4.0), labelled what it is: people with low back pain in
the Netherlands, never "healthy". Measurements come from metrics.py on TotalSpineSeg labels
(TSS weights: permission pending). The test split is never read.
Gate: at least MIN_N development patients of the same sex within AGE_BAND years of the patient's
age with that level measured. SPIDER records an age for only 41 of its 218 patients, so the gate
decides from the data whether anything may be shown; when it fails, nothing is shown.
Usage: python percentiles.py <age> <M|F> [metrics_json] -> prints the decision (and percentiles)"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare import WORK  # noqa: E402
from spider import SCORED_LEVELS, overview  # noqa: E402
from split import load as load_split  # noqa: E402

MIN_N, AGE_BAND = 30, 10
MEASURES = ("disc_height_mm", "disc_to_csf_signal")
COHORT = "SPIDER back-pain cohort (Netherlands, CC BY 4.0), not healthy people"


def cohort(age, sex):
    sp = load_split()
    o = overview().groupby("patient").first()
    out = []
    for p, r in o.iterrows():
        s = sp["patients"][str(p)]
        if s["split"] != "dev" or not s["t2"] or r.sex != sex or np.isnan(r.birth_date):
            continue
        if abs(float(r.birth_date) - age) <= AGE_BAND:   # SPIDER's birth_date column holds the age in years
            out.append(int(p))
    return out


def reference(age, sex):
    pts = cohort(age, sex)
    ref = {lv: {m: [] for m in MEASURES} for lv in SCORED_LEVELS}
    for p in pts:
        f = os.path.join(WORK, "dev", "metrics", f"{p}_tss.json")
        if not os.path.exists(f):
            continue
        for lv in json.load(open(f)).get("levels", []):
            if lv["level"] in ref:
                for m in MEASURES:
                    if lv.get(m) is not None:
                        ref[lv["level"]][m].append(float(lv[m]))
    return pts, ref


def decide(age, sex, metrics=None):
    pts, ref = reference(age, sex)
    out = {"cohort": COHORT, "age_band": [age - AGE_BAND, age + AGE_BAND], "sex": sex,
           "matched_patients": len(pts), "min_n": MIN_N, "levels": {}}
    for lv, ms in ref.items():
        for m, vals in ms.items():
            if len(vals) < MIN_N:
                continue
            row = {"n": len(vals)}
            if metrics and lv in metrics and metrics[lv].get(m) is not None:
                row["percentile"] = int(round(100 * np.mean(np.array(vals) <= metrics[lv][m])))
            out["levels"].setdefault(lv, {})[m] = row
    out["shown"] = bool(out["levels"])
    if not out["shown"]:
        out["why_not"] = (f"SPIDER gives an age for 41 of 218 patients; {len(pts)} development patients of sex {sex} "
                          f"are within {AGE_BAND} years of {age}, short of {MIN_N}. No comparison is shown.")
    return out


if __name__ == "__main__":
    age, sex = int(sys.argv[1]), sys.argv[2]
    m = None
    if len(sys.argv) > 3:
        m = {lv["level"]: lv for lv in json.load(open(sys.argv[3]))["levels"]}
    print(json.dumps(decide(age, sex, m), indent=1))
