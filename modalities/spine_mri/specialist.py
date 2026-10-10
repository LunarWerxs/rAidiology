"""The specialist model on a patient disc, written only when a method passed the ship rule.

Reads work/eval/final_eval.json (the ship decision) and modalities/spine_mri/eval/SCORECARD.md (the held-out
test numbers the app may quote), runs the shipped grader on the patient's lumbar sagittal T2/T1 with
its TotalSpineSeg labels, and writes work/specialist.json for make_report.py. When nothing shipped
it writes nothing and removes a stale file, so the app shows no model.
A scan that feeds a mistake-log case (cases/<patient>/mistake-log/) is development data: its re-run is
marked not blind.
--dev-version N: the same reading from a DEVELOPMENT grader vN, written to work/specialist_dev.json, with no
ship rule and, instead of the held-out scorecard, a dev_scorecard: vN's out-of-fold numbers on SPIDER's
development discs (../Models/disc-grader-vN/oof_logits.npz) beside the shipped v1's on the same discs.
It never reads or writes specialist.json.
Usage: python specialist.py <work_dir> [--grader-version 1] [--dev-version N]
       python specialist.py --selftest"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "eval"))
sys.path.insert(0, ROOT)

PLAIN = {"herniation": "herniation", "narrowing": "narrowing", "bulging": "bulge", "spondylolisthesis": "slipped vertebra",
         "up_endplate": "upper endplate change", "low_endplate": "lower endplate change", "modic": "Modic change"}
DRAWS, SEED = 1000, 0


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
                         "text": f"rAidiology's disc model (its grades are in Levels) was tested on {n_pts} "
                                 f"expert-graded scans it never trained on: it gave the same disc-drying grade "
                                 f"as the expert {round(100 * v)}% of the time "
                                 f"(95% CI {round(100 * lo)}-{round(100 * hi)}%)."}}


def pick_levels(grades):
    """Per-level reading and the comparison row text, from grader.patient()'s output."""
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
    return levels, rows


def is_not_blind():
    return bool(glob.glob(os.path.join(ROOT, "cases", "*", "mistake-log", "*.json")))


def read_patient(work, version):
    """The grader's per-level grades on the patient's lumbar sagittal T2/T1 (needs the scan, TSS labels and a GPU)."""
    from grader import patient
    series = json.load(open(os.path.join(work, "nifti", "series.json")))
    pick = lambda seq: max((s for s in series if s["region"] == "L" and s["plane"] == "sagittal" and s["sequence"] == seq),
                           key=lambda s: s["count"], default=None)
    t2, t1 = pick("T2"), pick("T1")
    nii = lambda s: os.path.join(work, "nifti", s["name"] + ".nii.gz")
    return patient(version, nii(t2), os.path.join(work, "tss_out", "step2_output", t2["name"] + ".nii.gz"),
                   nii(t1) if t1 else None)


def write_shipped(work, version, grades_fn=read_patient):
    """The default path: work/specialist.json only for a grader that passed the ship rule."""
    out = os.path.join(work, "specialist.json")
    state_p = os.path.join(work, "eval", "final_eval.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {}
    shipped = [m for m, s in state.get("ship", {}).items() if s.get("ships")]
    if "grader" not in shipped:
        if os.path.exists(out):
            os.remove(out)
        print("No grader shipped (final evaluation:", state.get("ship", "not run"), "); no specialist data.")
        return None
    levels, rows = pick_levels(grades_fn(work, version))
    sc = headline("grader")
    sc.update(method=f"rAidiology disc grader v{version}", split="held-out test, SPIDER",
              permission="TSS weights: permission pending", source="modalities/spine_mri/eval/SCORECARD.md")
    not_blind = is_not_blind()
    json.dump({"method": sc["method"], "not_blind": not_blind, "levels": levels, "scorecard": sc,
               "comparison": {"model_label": "Specialist model" + (" (not blind)" if not_blind else ""), "rows": rows}},
              open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, json.dumps(levels, indent=1))
    return out


# ---- the development model's numbers: out-of-fold on SPIDER's development discs ----

def qwk(t, y):
    n = len(t)
    O = [[0.0] * 5 for _ in range(5)]
    for a, b in zip(t, y):
        O[int(a) - 1][int(b) - 1] += 1
    rs = [sum(r) for r in O]
    cs = [sum(O[i][j] for i in range(5)) for j in range(5)]
    num = sum((i - j) ** 2 / 16.0 * O[i][j] for i in range(5) for j in range(5))
    den = sum((i - j) ** 2 / 16.0 * rs[i] * cs[j] / n for i in range(5) for j in range(5))
    return 1 - num / den if den > 0 else float("nan")


def metrics():
    """The same three functions compare.py uses (score.py's)."""
    from score import m_exact, m_qwk, m_within_one
    return {"exact": m_exact, "within_one": m_within_one, "kappa": m_qwk}


def boot_ci(patients, fn, *arrays):
    """95% percentile CI by patient, 1000 resamples, seed 0 (score.py's recipe with this card's draw count)."""
    import numpy as np
    P = np.array(sorted(set(int(p) for p in patients)))
    rows = {p: np.nonzero(patients == p)[0] for p in P}
    draws = np.random.default_rng(SEED).integers(0, len(P), size=(DRAWS, len(P)))
    vals = []
    for d in draws:
        idx = np.concatenate([rows[P[k]] for k in d])
        v = fn(*(a[idx] for a in arrays))
        if v == v:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def card_numbers(truth, grade, patients, fns):
    return {k: {"value": round(float(fn(truth, grade)), 4), "ci": boot_ci(patients, fn, truth, grade)} for k, fn in fns.items()}


def dev_scorecard(truth, g_dev, g_v1, patients, version):
    """vN's and v1's exact, within-one and quadratic weighted kappa on the same discs, with the one plain sentence."""
    import numpy as np
    truth, g_dev, g_v1, patients = (np.asarray(x) for x in (truth, g_dev, g_v1, patients))
    fns = metrics()
    dev, v1 = card_numbers(truth, g_dev, patients, fns), card_numbers(truth, g_v1, patients, fns)
    n_pts = len(set(int(p) for p in patients))
    text = (f"A newer version of the disc model, still in development: on {n_pts} expert-graded development scans it gave "
            f"the expert's exact grade {round(100 * dev['exact']['value'])}% of the time (v1: {round(100 * v1['exact']['value'])}%) "
            f"and is not yet tested on scans it never saw.")
    return {"method": f"rAidiology disc grader v{version} (in development)", "split": "SPIDER development, out of fold",
            "n_patients": n_pts, "n_discs": int(len(truth)), **dev, "v1": v1, "text": text}


def dev_scorecard_from_oof(version, which="nested"):
    """Load both runs' oof_logits.npz as compare.py does; only SPIDER discs both predicted."""
    from compare import frame
    from core import paths
    models = paths.MODELS
    fd = frame(os.path.join(models, f"disc-grader-v{version}", "oof_logits.npz"), which)
    f1 = frame(os.path.join(models, "disc-grader-v1", "oof_logits.npz"), which)
    both = fd[fd.collection == "spider"].merge(f1[f1.collection == "spider"], on=["patient", "ivd"], suffixes=("_d", "_1"))
    if (both.truth_d != both.truth_1).any():
        raise SystemExit("the two runs disagree on the truth of a disc; they were not trained on the same grades")
    return dev_scorecard(both.truth_d.values, both.grade_d.values, both.grade_1.values, both.patient.values, version)


def write_dev(work, version, grades, card):
    """work/specialist_dev.json from a development grader's grades; never touches specialist.json."""
    out = os.path.join(work, "specialist_dev.json")
    levels, rows = pick_levels(grades)
    not_blind = is_not_blind()
    json.dump({"method": card["method"], "not_blind": not_blind, "levels": levels, "dev_scorecard": card,
               "comparison": {"model_label": "Newer model, in development" + (" (not blind)" if not_blind else ""), "rows": rows}},
              open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, json.dumps(levels, indent=1))
    print(card["text"])
    return out


def selftest():
    import tempfile
    import numpy as np
    # 1. the dev_scorecard maths on a tiny fake out-of-fold set with a known exact share and kappa
    truth = np.array([1, 2, 3, 4, 5, 3, 2, 2])
    g_dev = np.array([1, 2, 3, 4, 4, 3, 2, 3])       # 6 of 8 exact, 8 of 8 within one
    g_v1 = np.array([1, 2, 2, 4, 5, 2, 3, 2])       # 5 of 8 exact
    pts = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    card = dev_scorecard(truth, g_dev, g_v1, pts, 120)
    assert card["exact"]["value"] == 0.75 and card["within_one"]["value"] == 1.0, card
    assert abs(card["kappa"]["value"] - round(qwk(truth, g_dev), 4)) < 1e-9, (card["kappa"], qwk(truth, g_dev))
    assert card["v1"]["exact"]["value"] == 0.625, card["v1"]
    assert card["n_patients"] == 4 and card["n_discs"] == 8
    for k in ("exact", "within_one", "kappa"):
        v = card[k]
        assert 0 <= v["ci"][0] <= v["value"] + 1e-9 and v["ci"][0] <= v["ci"][1] <= 1.0 + 1e-9, (k, v)
    assert card == dev_scorecard(truth, g_dev, g_v1, pts, 120), "bootstrap is not deterministic"
    assert "75%" in card["text"] and "(v1: 62%)" in card["text"] and "4 expert-graded" in card["text"], card["text"]
    assert "not yet tested on scans it never saw" in card["text"]
    assert card["method"] == "rAidiology disc grader v120 (in development)"
    # 2. --dev-version never touches specialist.json; 3. the default path is unchanged
    fake = {"L4-L5": {"pfirrmann": 3, "pfirrmann_probs": [0.02, 0.21, 0.62, 0.13, 0.02],
                      "items": {"herniation": 0.08, "bulging": 0.71, "modic": 0.05}}}
    levels, rows = pick_levels(fake)
    assert levels["L4-L5"]["flags"] == ["bulging"] and levels["L4-L5"]["pfirrmann_confidence"] == 0.62
    assert rows["L4-L5"] == "Grade 3/5 (62% sure); bulge 71%", rows
    with tempfile.TemporaryDirectory() as w:
        shipped_p = os.path.join(w, "specialist.json")
        open(shipped_p, "w").write('{"shipped": true}')
        out = write_dev(w, 120, fake, card)
        assert os.path.basename(out) == "specialist_dev.json" and open(shipped_p).read() == '{"shipped": true}'
        d = json.load(open(out))
        assert "scorecard" not in d and d["dev_scorecard"] == card and d["method"] == card["method"]
        assert d["comparison"]["model_label"].startswith("Newer model, in development") and d["comparison"]["rows"] == rows
        assert d["levels"]["L4-L5"]["flags"] == ["bulging"] and "not_blind" in d
        # default path, nothing shipped: removes the stale file and never calls the grader
        os.makedirs(os.path.join(w, "eval"))
        json.dump({"ship": {"grader": {"ships": False}}}, open(os.path.join(w, "eval", "final_eval.json"), "w"))
        boom = lambda *a: (_ for _ in ()).throw(AssertionError("grader called"))
        assert write_shipped(w, 1, boom) is None and not os.path.exists(shipped_p)
        assert os.path.exists(out), "the default path removed specialist_dev.json"
        # default path, shipped: writes specialist.json with a scorecard, no dev_scorecard
        json.dump({"ship": {"grader": {"ships": True}}}, open(os.path.join(w, "eval", "final_eval.json"), "w"))
        if headline("grader"):
            p = write_shipped(w, 1, lambda *a: fake)
            s = json.load(open(p))
            assert s["method"] == "rAidiology disc grader v1" and "scorecard" in s and "dev_scorecard" not in s
            assert s["comparison"]["model_label"].startswith("Specialist model")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", nargs="?")
    ap.add_argument("--grader-version", type=int, default=1)
    ap.add_argument("--dev-version", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.work:
        ap.error("work_dir is required")
    if a.dev_version is not None:
        card = dev_scorecard_from_oof(a.dev_version)
        write_dev(a.work, a.dev_version, read_patient(a.work, a.dev_version), card)
    else:
        write_shipped(a.work, a.grader_version)


if __name__ == "__main__":
    main()
