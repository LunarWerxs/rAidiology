"""The specialist model on a patient disc, written only when a method passed the ship rule.

Reads work/eval/final_eval.json (the ship decision) and pipeline/eval/SCORECARD.md (the held-out
test numbers the app may quote), runs the shipped grader on the patient's lumbar sagittal T2/T1 with
its TotalSpineSeg labels, and writes work/specialist.json for make_report.py. When nothing shipped
it writes nothing and removes a stale file, so the app shows no model.
A scan that feeds a mistake-log case (pipeline/eval/cases/) is development data: its re-run is
marked not blind.
Usage: python specialist.py <work_dir> [--grader-version 1]"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))

PLAIN = {"herniation": "herniation", "narrowing": "narrowing", "bulging": "bulge", "spondylolisthesis": "slipped vertebra",
         "up_endplate": "upper endplate change", "low_endplate": "lower endplate change", "modic": "Modic change"}


def headline(method):
    """Pfirrmann exact agreement on the held-out test split, tss localisation, from SCORECARD.md."""
    card = open(os.path.join(HERE, "eval", "SCORECARD.md"), encoding="utf-8").read()
    pat = rf"\| test \| pfirrmann \| exact \| {method} \| tss \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| \d+ \|"
    m = re.search(pat, card)
    if not m:
        return None
    n_discs, n_pts, v, lo, hi = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
    return {"n_scans": n_pts, "n_discs": n_discs,
            "headline": {"item": "pfirrmann", "metric": "exact", "value": v, "ci": [lo, hi],
                         "text": f"SuprSkan's disc model (its grades are in Levels) was tested on {n_pts} "
                                 f"expert-graded scans it never trained on: it gave the same disc-drying grade "
                                 f"as the expert {round(100 * v)}% of the time "
                                 f"(95% CI {round(100 * lo)}-{round(100 * hi)}%)."}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--grader-version", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(a.work, "specialist.json")
    state_p = os.path.join(a.work, "eval", "final_eval.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {}
    shipped = [m for m, s in state.get("ship", {}).items() if s.get("ships")]
    if "grader" not in shipped:
        if os.path.exists(out):
            os.remove(out)
        print("No grader shipped (final evaluation:", state.get("ship", "not run"), "); no specialist data.")
        return
    from grader import patient
    series = json.load(open(os.path.join(a.work, "nifti", "series.json")))
    pick = lambda seq: max((s for s in series if s["region"] == "L" and s["plane"] == "sagittal" and s["sequence"] == seq),
                           key=lambda s: s["count"], default=None)
    t2, t1 = pick("T2"), pick("T1")
    nii = lambda s: os.path.join(a.work, "nifti", s["name"] + ".nii.gz")
    grades = patient(a.grader_version, nii(t2), os.path.join(a.work, "tss_out", "step2_output", t2["name"] + ".nii.gz"),
                   nii(t1) if t1 else None)
    levels, rows = {}, {}
    for level, g in grades.items():
        conf = g["pfirrmann_probs"][g["pfirrmann"] - 1]
        # A patient sees an item flagged only when the model calls it more likely than not. The
        # scorecard's sensitivity/specificity use thresholds tuned for screening (Youden), which sit as
        # low as a few percent for rare items: fine for measuring, alarming on a patient's own disc.
        flags = sorted((b for b, p in g["items"].items() if p >= 0.5), key=lambda b: -g["items"][b])
        levels[level] = {"pfirrmann": g["pfirrmann"], "pfirrmann_confidence": round(conf, 2),
                         "pfirrmann_probs": g["pfirrmann_probs"], "items": g["items"], "flags": flags}
        named = ", ".join(f"{PLAIN[b]} {round(100 * g['items'][b])}%" for b in flags)
        rows[level] = f"Grade {g['pfirrmann']}/5 ({round(100 * conf)}% sure)" + (f"; {named}" if named else "")
    sc = headline("grader")
    sc.update(method=f"SuprSkan disc grader v{a.grader_version}", split="held-out test, SPIDER",
              permission="TSS weights: permission pending", source="pipeline/eval/SCORECARD.md")
    not_blind = bool(glob.glob(os.path.join(HERE, "eval", "cases", "*.json")))
    json.dump({"method": sc["method"], "not_blind": not_blind, "levels": levels, "scorecard": sc,
               "comparison": {"model_label": "Specialist model" + (" (not blind)" if not_blind else ""), "rows": rows}},
              open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, json.dumps(levels, indent=1))


if __name__ == "__main__":
    main()
