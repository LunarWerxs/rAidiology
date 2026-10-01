"""The final evaluation: opens the held-out test split ONCE and scores every candidate together.

Order (each step is recorded in work/eval/final_eval.json before the next starts):
1. Refuse if final_eval.json already says the test split was opened.
2. Choose the reference baseline by development Pfirrmann kappa (localisation tss) among the
   baselines present in the development predictions (rules, panel). Recorded before step 3.
3. Test split: SPIDER -> NIfTI -> TotalSpineSeg -> matching -> rules (thresholds from all
   development folds) -> crops -> the shipped grader ensemble -> the AI panel's bundles for 30 test
   patients (seed 20260930, half per scanner maker). The panel runs as Claude Opus sub-agents (no
   API key), so the run stops here; after their answers are ingested, `--finish` continues.
   MedGemma and the combination run only if they qualified on development folds (--with-medgemma).
4. score.py with the reference -> SCORECARD.md, including the paired differences.
5. Ship rule: a method ships only if the paired patient-level bootstrap 95% CI of its Pfirrmann
   kappa minus the reference's lies above zero, on the patients both scored.
Then a separate agent session that has not seen score.py runs recompute.py.
Usage: python final_eval.py --panel agents|none [--with-medgemma] [--grader-version 1]
       python final_eval.py --finish        (after the panel bundles are answered and ingested)"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import score  # noqa: E402
from prepare import WORK  # noqa: E402

EVAL = os.path.dirname(WORK)
STATE = os.path.join(EVAL, "final_eval.json")
PY = sys.executable


def record(state, **kw):
    state.update(kw)
    state.setdefault("log", []).append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw})
    json.dump(state, open(STATE, "w"), indent=1)


def dev_kappa(df, method):
    g = df[(df.split == "dev") & (df.method == method) & (df.localisation == "tss") & (df.in_scope == 1)]
    if g.empty or not g.pfirrmann_pred.notna().any():
        return None
    t, y, _ = score.filled(g, "pfirrmann")
    return score.m_qwk(t, y)


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=HERE, env={**os.environ, "SUPRSKAN_FINAL_EVAL": "1"})


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--panel", choices=["agents", "none"])
    g.add_argument("--finish", action="store_true")
    ap.add_argument("--with-medgemma", action="store_true")
    ap.add_argument("--grader-version", type=int, default=1)
    a = ap.parse_args()
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if a.finish:
        return finish(state)
    if state.get("test_opened"):
        sys.exit(f"The test split was already opened on {state['test_opened']}; it is opened once. See {STATE}.")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    df = score.load(os.path.join(EVAL, "predictions.csv"), score.path("radiological_gradings.csv"))
    kappas = {m: dev_kappa(df, m) for m in ("rules", "panel")}
    kappas = {m: k for m, k in kappas.items() if k is not None}
    reference = max(kappas, key=kappas.get)
    if a.with_medgemma:
        q = os.path.join(EVAL, "medgemma_qualify.json")
        if not (os.path.exists(q) and json.load(open(q)).get("qualifies")):
            sys.exit("MedGemma did not qualify on development folds (medgemma_qualify.json); run without --with-medgemma.")
    record(state, commit=commit, dev_pfirrmann_kappa=kappas, reference=reference, panel=a.panel,
           with_medgemma=a.with_medgemma)
    record(state, test_opened=time.strftime("%Y-%m-%d %H:%M:%S"))
    run("run_split.py", "test")
    run("crops.py", "test")
    run("grader.py", "predict", "test", "--version", str(a.grader_version))
    if a.with_medgemma:
        run("medgemma_runner.py", "test")
    if a.panel == "agents":
        run("panel_runner.py", "spider", "test", "--sample-test", "30", "--backend", "agents")
        record(state, awaiting="panel answers: run the Opus readers on the new bundles, ingest, then --finish")
        print("Panel bundles written; answer them, ingest, then run final_eval.py --finish")
        return
    finish(state)


def finish(state):
    """Score every method on the test split together and apply the ship rule."""
    if not state.get("test_opened") or state.get("finished"):
        sys.exit("Nothing to finish: the test split is not open, or the evaluation already finished.")
    reference, kappas = state["reference"], state["dev_pfirrmann_kappa"]
    if state.get("panel") == "agents":
        run("panel_runner.py", "spider", "test", "--sample-test", "30", "--backend", "agents")
        if not os.path.exists(os.path.join(EVAL, "pred_panel_test.csv")):
            sys.exit("Panel answers are missing; ingest them first.")
    record(state, awaiting=None)
    run("run_split.py", "--collect-only")   # predictions.csv with every method's test rows
    run("score.py", "--reference", reference)
    card = open(os.path.join(HERE, "SCORECARD.md"), encoding="utf-8").read()
    ship = {}
    for line in card.splitlines():
        m = re.match(r"\| test \| pfirrmann \| qwk \| (\w+) \| (\w+) \| (\d+) \| (\d+) \| (\S+) \| (\S+) \| (\S+) \| (\d+) \|", line)
        if m and m.group(2) == reference:
            lo = m.group(6)
            ship[m.group(1)] = {"difference": m.group(5), "ci_low": lo, "ci_high": m.group(7),
                                "n_discs": int(m.group(3)), "n_patients": int(m.group(4)),
                                "ships": lo != "n/a" and float(lo) > 0}
    lines = ["", "## Ship decision", "",
             f"Reference baseline: **{reference}** (highest development Pfirrmann kappa: "
             + ", ".join(f"{k} {v:.3f}" for k, v in kappas.items()) + ").",
             "A method ships only if the 95% CI of its Pfirrmann kappa minus the reference's, on the patients "
             "both scored, lies above zero.", ""]
    for meth, s in ship.items():
        lines.append(f"- **{meth}**: difference {s['difference']} (95% CI {s['ci_low']} to {s['ci_high']}), "
                     f"{s['n_discs']} discs, {s['n_patients']} patients: "
                     + ("**ships**." if s["ships"] else "does not ship (the interval reaches zero or below)."))
    open(os.path.join(HERE, "SCORECARD.md"), "a", encoding="utf-8").write("\n".join(lines) + "\n")
    record(state, ship=ship, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(ship, indent=1))


if __name__ == "__main__":
    main()
