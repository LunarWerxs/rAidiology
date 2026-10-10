"""The X-ray final evaluation: opens the held-out test split ONCE and scores every candidate together.

Order, and each step is written into work/eval/xray/final_eval.json before the next one starts:
1. Refuse if final_eval.json already says the test split was opened.
2. Choose the reference baseline by development AUROC on GRAZPEDWRI-DX, stratum `all`, among the
   baselines actually present in the development predictions (prevalence, meta, features, reader).
   Recorded before step 3, so the thing the detector must beat is fixed before the test is seen.
3. Test split: baselines (the models fitted on all five development folds), then the detector
   ensemble, then the AI reader's bundles. The reader runs as Claude Opus sub-agents, with no API
   key, so the run stops there; after their answers are ingested, `--finish` continues.
4. score.py with that reference, which adds the paired differences to SCORECARD.md.
5. The ship rule, declared in modalities/xray/README.md before any of this: the detector ships only
   if the 95% patient-level bootstrap interval of its AUROC minus the reference's, on the GRAZPEDWRI-DX
   test images both scored, lies above zero.
Then a separate agent session that has not seen score.py runs recompute.py.

Usage: python final_eval.py --reader agents|none [--version 1]
       python final_eval.py --finish        (after the reader bundles are answered and ingested)

The second look (decided 2026-10-07, docs/HANDOFF.md): a detector kept on development, chosen with no
test number, may read the spent test split ONCE more, labelled a second look, with the test's count of
looks. Only when its development gain over the model the test last read is beyond noise (compare.py
says KEPT, run BEFORE anything touches the test); once per kept model per test set; no test number
chooses anything; a look that comes out lower is reported, not re-run. The candidate is the versions
read together: an image's score is the mean over versions of that version's five-net mean (so ten nets
weigh equally); on development, the mean of the versions' oof_dev.csv scores. Each version's development
rows must equal the first look's (the split file's hash may differ: it gained other collections; the rows decide).

       python final_eval.py --second-look --versions 140,141 [--batch 64] [--card-gb 1.5]
       python final_eval.py --selftest      CPU, seconds, temp files, no data, no card

Steps, each recorded in the look's own 'done' list in final_eval.json (state['looks']); a look that is
opened but unfinished is resumed by running the same command again: done steps are skipped, its
recorded thresholds are kept, there is no new gate and no new latch. Everything the look writes goes
to work/eval/xray/look<N>/ (the first look's files in work/eval/xray/ are never touched), and a section
is appended to SCORECARD.md. The reference stays the first look's.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402
import preds  # noqa: E402
import score  # noqa: E402

STATE = os.path.join(preds.WORK, "final_eval.json")
SHIP_DATASET = "grazpedwri"
SHIP_STRATUM = "all"
SHIP_METRIC = "auroc"
CANDIDATE = "cnn"
BASELINES = ["prevalence", "meta", "features", "reader"]
PY = sys.executable
CARD = os.path.join(HERE, "SCORECARD.md")
# The seed noise measured for the detector, per collection (docs/efficiency/README.md, idea 55): the
# second look's development gate asks for a gain beyond it, not just above zero.
NOISE = {"fracatlas": 0.0106, "grazpedwri": 0.0030}


def record(state, scope=None, **kw):
    """Write into the state (or into one look's entry, `scope`) and save the state file."""
    (scope if scope is not None else state).update(kw)
    entry = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    if scope is not None:
        entry["look"] = scope["look"]
    state.setdefault("log", []).append(entry)
    os.makedirs(preds.WORK, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def dev_auroc():
    """{baseline: its development AUROC on the ship dataset}, for the ones that have rows."""
    df = score.load()
    gs, _ = score.groups(df)
    out = {}
    for m in BASELINES:
        g = gs.get((m, SHIP_DATASET, "dev"))
        if g is None:
            continue
        r = score.number(g, SHIP_STRATUM, SHIP_METRIC)
        if r and r[0] is not None:
            out[m] = r[0]
    return out


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, "-u", *args], check=True, cwd=HERE,
                   env={**os.environ, "RAIDIOLOGY_FINAL_EVAL": "1"})


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reader", choices=["agents", "none"])
    g.add_argument("--finish", action="store_true")
    g.add_argument("--second-look", action="store_true", dest="second_look")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--versions", default=None, help="second look: the versions read together, 140,141")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--card-gb", type=float, default=1.5, dest="card_gb")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
    if a.second_look:
        if not a.versions:
            sys.exit("--second-look needs --versions 140,141")
        try:
            versions = sorted({int(v) for v in a.versions.split(",")})
        except ValueError:
            sys.exit("--versions is a comma list of integers, got %r" % a.versions)
        return second_look(state, begin(state, versions), a.batch, a.card_gb)
    if a.finish:
        return finish(state)
    if state.get("test_opened"):
        sys.exit("The test split was already opened on %s; it is opened once. See %s."
                 % (state["test_opened"], STATE))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=HERE).stdout.strip()
    aurocs = dev_auroc()
    if not aurocs:
        sys.exit("No baseline has a development AUROC yet: run baselines.py dev (and the reader) first.")
    reference = max(aurocs, key=aurocs.get)
    record(state, commit=commit, version=a.version, dev_auroc=aurocs, reference=reference,
           reader=a.reader, ship={"dataset": SHIP_DATASET, "stratum": SHIP_STRATUM,
                                  "metric": SHIP_METRIC, "candidate": CANDIDATE})
    record(state, test_opened=time.strftime("%Y-%m-%d %H:%M:%S"))
    run("baselines.py", "test")
    run("detector.py", "predict", "test", "--version", str(a.version))
    if a.reader == "agents":
        run("reader.py", "test", "--backend", "agents")
        record(state, awaiting="reader answers: run the Opus readers on the new bundles, ingest, then --finish")
        print("Reader bundles written. Answer them, ingest them, then run final_eval.py --finish",
              flush=True)
        return
    finish(state)


def ship_row(card, reference):
    """The paired difference row the ship rule reads, from the scorecard itself."""
    want = r"\| %s \| test \| %s \| %s \| %s \| %s \| (\d+) \| (\d+) \| (\S+) \| (\S+) \| (\S+) \| (\d+) \|" % (
        SHIP_DATASET, SHIP_STRATUM, SHIP_METRIC, CANDIDATE, reference)
    m = re.search(want, card)
    if not m:
        return None
    return {"n_images": int(m.group(1)), "n_patients": int(m.group(2)), "difference": m.group(3),
            "ci_low": m.group(4), "ci_high": m.group(5), "n_boot": int(m.group(6))}


def finish(state):
    if not state.get("test_opened") or state.get("finished"):
        sys.exit("Nothing to finish: the test split is not open, or the evaluation already finished.")
    reference = state["reference"]
    if state.get("reader") == "agents" and not os.path.exists(
            preds.path("reader", SHIP_DATASET, "test")):
        sys.exit("The reader's test answers are missing; ingest them first (reader.py ingest <answers.json>).")
    record(state, awaiting=None)
    run("preds.py")
    run("score.py", "--reference", reference)
    card = open(os.path.join(HERE, "SCORECARD.md"), encoding="utf-8").read()
    row = ship_row(card, reference)
    if row is None:
        sys.exit("SCORECARD.md has no paired %s row for %s against %s; score.py did not write one."
                 % (SHIP_METRIC, CANDIDATE, reference))
    ships = row["ci_low"] != "n/a" and float(row["ci_low"]) > 0
    lines = ["", "## Ship decision", "",
             "Reference baseline: **%s**, the best development AUROC on %s (%s)."
             % (reference, SHIP_DATASET,
                ", ".join("%s %.3f" % (k, v) for k, v in sorted(state["dev_auroc"].items()))),
             "", "The rule, written down in `modalities/xray/README.md` before the test split was "
             "opened: the detector ships only if the 95%% interval of its AUROC minus the reference's, "
             "on the %s test images both scored, lies above zero." % SHIP_DATASET, "",
             "- **%s**: difference %s (95%% CI %s to %s), %d images, %d patients: %s"
             % (CANDIDATE, row["difference"], row["ci_low"], row["ci_high"], row["n_images"],
                row["n_patients"],
                "**it ships**." if ships else "it does not ship (the interval reaches zero or below)."),
             ""]
    with open(os.path.join(HERE, "SCORECARD.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    record(state, ship_result={**row, "ships": ships}, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps({"reference": reference, **row, "ships": ships}, indent=1), flush=True)


# ---- the second look -------------------------------------------------------------------------

def _detector():
    import detector  # noqa: PLC0415  imports torch, which the first look's paths do not need
    return detector


def looks_of(state):
    return state.get("looks", [])


def look_folder(number):
    return os.path.join(preds.WORK, "look%d" % number)


def label(versions):
    return "+".join("v%d" % v for v in versions)


def refuse_second(state, versions):
    """Refuse, or return the unfinished look of these versions to resume (None for a new look)."""
    if not (state.get("test_opened") and state.get("finished")):
        sys.exit("The first look has not finished (see %s): a second look needs it finished." % STATE)
    resume = None
    for l in looks_of(state):
        same = sorted(l["versions"]) == versions
        if same and l.get("finished"):
            sys.exit("%s already had its look (look %d, finished %s)." % (label(versions), l["look"], l["finished"]))
        if not l.get("finished"):
            if same and l.get("test_opened"):
                resume = l
            else:
                sys.exit("Look %d (%s) is open and unfinished: run its own command again to carry it on."
                         % (l["look"], label(l["versions"])))
    return resume


def oof_path(version):
    return os.path.join(_detector().model_dir(version), "oof_dev.csv")


def checks(state, versions):
    """Everything that has to be true of the versions before the test is seen, or exit."""
    det = _detector()
    first_meta = json.load(open(os.path.join(det.model_dir(state["version"]), "meta.json"), encoding="utf-8"))
    first_split_sha = first_meta.get("split_sha256")
    key = ["dataset", "image", "patient", "fold", "truth"]
    first_oof = set(map(tuple, pd.read_csv(oof_path(state["version"]), dtype=str, keep_default_na=False)[key].values))
    oof_sets = []
    for v in versions:
        if v < det.TRIAL_FROM:
            sys.exit("version %d is below %d: a second look reads kept development trials." % (v, det.TRIAL_FROM))
        d = det.model_dir(v)
        need = ["meta.json", "oof_dev.csv"] + ["fold%d.pt" % k for k in range(5)]
        gone = [n for n in need if not os.path.exists(os.path.join(d, n))]
        if gone:
            sys.exit("version %d lacks %s in %s." % (v, gone, d))
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        if meta.get("datasets") != datasets.NAMES:
            sys.exit("version %d's datasets %s are not %s." % (v, meta.get("datasets"), datasets.NAMES))
        v_oof = set(map(tuple, pd.read_csv(oof_path(v), dtype=str, keep_default_na=False)[key].values))
        oof_sets.append(v_oof)
        if v_oof != first_oof:
            sys.exit("version %d's development rows differ from version %d's (%d vs %d rows, %d not shared)."
                     % (v, state["version"], len(v_oof), len(first_oof), len(v_oof ^ first_oof)))
        if meta.get("split_sha256") != first_split_sha:
            print("version %d's split file hash differs from version %d's (the file gained other collections); its development rows are the same, so the split of the shipped collections is the same" % (v, state["version"]), flush=True)
    for v, s in zip(versions[1:], oof_sets[1:]):
        if s != oof_sets[0]:
            sys.exit("version %d's development rows differ from version %d's (%d vs %d rows, %d not shared)."
                     % (v, versions[0], len(s), len(oof_sets[0]), len(s ^ oof_sets[0])))


def candidate_dev(versions):
    """The versions' out-of-fold scores averaged, one row per development image."""
    key = ["dataset", "image", "patient", "fold", "truth"]
    frames = []
    for v in versions:
        f = pd.read_csv(oof_path(v), dtype={c: str for c in key})
        frames.append(f.set_index(["dataset", "image"]).score)
    out = pd.read_csv(oof_path(versions[0]), dtype={c: str for c in key})[key].copy()
    idx = pd.MultiIndex.from_frame(out[["dataset", "image"]])
    out["score"] = np.mean([s.reindex(idx).values for s in frames], axis=0)
    return out


def last_read(state):
    """(label, development rows file) of the model the test last read."""
    done = [l for l in looks_of(state) if l.get("finished")]
    if done:
        return ("look %d (%s)" % (done[-1]["look"], label(done[-1]["versions"])),
                os.path.join(done[-1]["folder"], "oof_dev.csv"))
    return "v%d" % state["version"], oof_path(state["version"])


def parse_compare(stdout, returncode):
    """Extract the verdict and per-collection lines from compare.py output.

    The verdict is the last line whose stripped text is exactly a verdict word
    (KEPT, DROPPED, INCONCLUSIVE), or 'ERROR' if there is no such line or returncode is non-zero.
    """
    lines = [l.rstrip() for l in stdout.splitlines() if l.strip()]
    per = [l.strip() for l in lines if l.startswith("  ")]

    if returncode != 0:
        return per, "ERROR"

    verdict_words = {"KEPT", "DROPPED", "INCONCLUSIVE"}
    for line in reversed(lines):
        if line.strip() in verdict_words:
            return per, line.strip()

    return per, "ERROR"


def run_compare(what, a, b):
    """compare.py as a subprocess: (its per-collection lines, its verdict line)."""
    noise = ",".join("%s=%s" % kv for kv in NOISE.items())
    got = subprocess.run([PY, "-u", "compare.py", what, "--a", a, "--b", b, "--noise", noise], cwd=HERE,
                         capture_output=True, text=True, check=False)
    per, verdict = parse_compare(got.stdout, got.returncode)
    if got.returncode:
        per.append((got.stderr or "").strip()[-300:])
    return per, verdict


def begin(state, versions):
    """Refusals, checks, the development gate, the thresholds, then the latch. No test pixel is read here."""
    resume = refuse_second(state, versions)
    if resume is not None:
        print("resuming look %d (%s): its gate and thresholds stay as recorded" % (resume["look"], label(versions)),
              flush=True)
        return resume
    checks(state, versions)
    number = len(looks_of(state)) + 2
    folder = look_folder(number)
    os.makedirs(folder, exist_ok=True)
    dev = candidate_dev(versions)
    cand = os.path.join(folder, "oof_dev.csv")
    dev.to_csv(cand, index=False)
    last_label, last_file = last_read(state)
    what = "second look %d gate: %s against %s" % (number, label(versions), last_label)
    per, verdict = run_compare(what, last_file, cand)
    gate = {"what": what, "lines": per, "verdict": verdict, "passed": verdict == "KEPT", "against": last_label}
    print(what + ":\n" + "\n".join("  " + l for l in per) + "\n" + verdict, flush=True)
    if not gate["passed"]:
        record(state, gate_refusals=state.get("gate_refusals", []) + [
            {"versions": versions, "gate": gate, "at": time.strftime("%Y-%m-%d %H:%M:%S")}])
        sys.exit("The development gain of %s over %s is not beyond noise (compare.py says %s). The test is not opened."
                 % (label(versions), last_label, verdict))
    det = _detector()
    thresholds = {}
    for d in datasets.NAMES:
        m = dev[dev.dataset == d]
        thresholds[d] = det.youden(m.score.values, m.truth.values.astype(int))
    entry = {"look": number, "versions": versions, "folder": folder, "gate": gate, "thresholds": thresholds,
             "test_opened": None, "finished": None, "done": []}
    record(state, looks=looks_of(state) + [entry])
    record(state, entry, test_opened=time.strftime("%Y-%m-%d %H:%M:%S"))   # the latch, before any test pixel
    return entry


def step(state, entry, name, fn):
    if name in entry["done"]:
        print("= %s: already done" % name, flush=True)
        return
    print("+ %s" % name, flush=True)
    began = time.time()
    fn()
    record(state, entry, done=entry["done"] + [name],
           step_seconds={**entry.get("step_seconds", {}), name: round(time.time() - began, 1)})


def copy_baselines(entry):
    """The first look's prediction files that are not the detector's, byte for byte."""
    n = 0
    for m in BASELINES:
        for f in sorted(glob.glob(os.path.join(preds.WORK, "pred_%s_*_*.csv" % m))):
            shutil.copy2(f, os.path.join(entry["folder"], os.path.basename(f)))
            n += 1
    if not n:
        sys.exit("No first-look baseline prediction files in %s." % preds.WORK)


def write_dev_preds(entry):
    det = _detector()
    dev = pd.read_csv(os.path.join(entry["folder"], "oof_dev.csv"), dtype={"dataset": str, "image": str})
    key = {(r.dataset, r.image): r.score for r in dev.itertuples()}
    df, _, _ = det.table(("dev",))
    scores = pd.Series([key.get((r.dataset, str(r.image)), np.nan) for r in df.itertuples()], index=df.index)
    if scores[df.readable].isna().any():
        sys.exit("The candidate's development scores do not cover every development image.")
    det.write_split(df, "dev", scores, entry["thresholds"], entry["versions"][0], folder=entry["folder"])


def read_test(entry, batch, card_gb):
    """The one read: every fold net of every version on the test images, as detector.predict does."""
    os.environ["RAIDIOLOGY_FINAL_EVAL"] = "1"
    det = _detector()
    det.limit_card_memory(gb=card_gb, name="xray second look")
    n = len(datasets.NAMES)
    df, arrays, _ = det.table(("test",))
    pix = det.stack(df, arrays)
    rows = df.index.values[df.readable.values]
    per_version = []
    for v in entry["versions"]:
        meta, nets = det.load_folds(v, n)
        acc = np.zeros(len(rows), np.float64)
        for net, mean, sd in nets:
            acc += det.predict_rows(net, pix, df, rows, det.torch.tensor(mean, device=det.DEV),
                                    det.torch.tensor(sd, device=det.DEV), n, batch)
        per_version.append(acc / len(nets))
    scores = pd.Series(np.nan, index=df.index)
    scores.iloc[rows] = np.mean(per_version, axis=0)
    det.write_split(df, "test", scores, entry["thresholds"], entry["versions"][0], folder=entry["folder"])


def score_look(state, entry):
    folder = entry["folder"]
    preds.collect(folder)
    out = os.path.join(folder, "SCORECARD.md")
    subprocess.run([PY, "-u", "score.py", "--predictions", os.path.join(folder, "predictions.csv"),
                    "--out", out, "--reference", state["reference"]], cwd=HERE, check=True)
    row = ship_row(open(out, encoding="utf-8").read(), state["reference"])
    if row is None:
        sys.exit("%s has no paired %s row for %s against %s." % (out, SHIP_METRIC, CANDIDATE, state["reference"]))
    record(state, entry, ship_result=row)


def paired_table(gs_look, gs_first, names):
    """The look's cnn test group minus the first look's, per collection and stratum, on images both scored."""
    out = []
    for d in datasets.NAMES:
        a, b = gs_look.get((CANDIDATE, d, "test")), gs_first.get((CANDIDATE, d, "test"))
        if a is None or b is None:
            continue
        for st in score.stratum_order(names[d]):
            for metric in ("auroc", "sensitivity", "specificity"):
                r = score.paired(a, b, st, metric)
                if r is None:
                    continue
                v, lo, hi, _nb, ni, npat = r
                out.append({"dataset": d, "stratum": st, "metric": metric, "difference": v,
                            "ci_low": lo, "ci_high": hi, "images": ni, "patients": npat})
    return out


def against_first(entry):
    gs_look, names = score.groups(score.load(os.path.join(entry["folder"], "predictions.csv")))
    gs_first, _ = score.groups(score.load(os.path.join(preds.WORK, "predictions.csv")))
    return paired_table(gs_look, gs_first, names)


def section(state, entry):
    n, g = entry["look"], entry["gate"]

    def f(x):
        return "n/a" if x is None else "%+.4f" % x

    row = entry["ship_result"]
    lines = ["This is look %d of the X-ray test split. The first look (version %d, opened %s) is above, unchanged; "
             "this one is labelled a second look, and the result is reported whether higher or lower."
             % (n, state["version"], state["test_opened"]), "",
             "**Versions read together:** %s (each image's score is the mean of the versions' five-net means)."
             % label(entry["versions"]), "",
             "**Development gate**, run before the test was touched (%s): **%s**" % (g["what"], g["verdict"]), ""]
    lines += ["- " + l for l in g["lines"]] + [""]
    lines += ["**Thresholds** (Youden on the candidate's development scores, fixed before the read): "
              + ", ".join("%s %.4f" % (d, entry["thresholds"][d]) for d in datasets.NAMES), "",
              "**Ship row against the reference** (%s, fixed at the first look): %s AUROC difference %s "
              "(95%% CI %s to %s), %d images, %d patients. The look decides nothing about shipping; it reports."
              % (state["reference"], SHIP_DATASET, row["difference"], row["ci_low"], row["ci_high"],
                 row["n_images"], row["n_patients"]), "",
              "**Against the first look** (look %d minus look 1, on the test images both scored):" % n, "",
              "| collection | stratum | metric | difference | 95% CI | images | patients |",
              "|---|---|---|---|---|---|---|"]
    for r in entry["paired"]:
        lines.append("| %s | %s | %s | %s | %s to %s | %d | %d |" % (
            r["dataset"], r["stratum"], r["metric"], f(r["difference"]), f(r["ci_low"]), f(r["ci_high"]),
            r["images"], r["patients"]))
    lines += ["", "No test number chose anything here: the versions, the thresholds and the reference were fixed "
              "before the test was read.", "",
              "Files: `%s`." % os.path.relpath(entry["folder"], preds.WORK).replace(os.sep, "/")]
    return lines


def title_of(entry):
    return "## Second look %d (look %d of this test set)" % (entry["look"], entry["look"])


def write_md(state, entry):
    record(state, entry, paired=against_first(entry))
    with open(os.path.join(entry["folder"], "second_look.md"), "w", encoding="utf-8") as f:
        f.write("\n".join([title_of(entry), ""] + section(state, entry)) + "\n")


def append_card(state, entry):
    text = open(CARD, encoding="utf-8").read() if os.path.exists(CARD) else ""
    if title_of(entry) in text:    # a crash between the write and the record must not append twice
        return
    with open(CARD, "a", encoding="utf-8") as f:
        f.write("\n" + title_of(entry) + "\n\n" + "\n".join(section(state, entry)) + "\n")


def second_look_steps(state, entry, batch=64, card_gb=1.5):
    step(state, entry, "copy baselines", lambda: copy_baselines(entry))
    step(state, entry, "candidate dev predictions", lambda: write_dev_preds(entry))
    step(state, entry, "read test", lambda: read_test(entry, batch, card_gb))
    step(state, entry, "score", lambda: score_look(state, entry))
    step(state, entry, "write second_look", lambda: write_md(state, entry))
    step(state, entry, "append scorecard", lambda: append_card(state, entry))


def second_look(state, entry, batch=64, card_gb=1.5):
    t0 = time.time()
    second_look_steps(state, entry, batch, card_gb)
    record(state, entry, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps({"look": entry["look"], "versions": entry["versions"], **entry["ship_result"],
                      "seconds": round(time.time() - t0, 1)}, indent=1), flush=True)


def selftest():
    """The second look's logic with stubs, in temp folders: no data, no card, no torch, seconds."""
    import tempfile  # noqa: PLC0415
    import types  # noqa: PLC0415

    g = globals()
    keep = {k: g[k] for k in ("STATE", "CARD", "_detector", "parse_compare", "run_compare", "read_test", "write_dev_preds",
                              "score_look", "against_first")}
    keep_work, keep_paired, keep_order = preds.WORK, score.paired, score.stratum_order
    failures = []

    def check(name, ok):
        if not ok:
            failures.append(name)

    def exits(fn, *args):
        try:
            fn(*args)
        except SystemExit as e:
            return str(e) or "exit"
        return None

    class Stop(Exception):
        pass

    # parse_compare selftest
    exact_text = """second look 2 gate: v140+v141 against v1:
  fracatlas, 3226 pictures of 3226 patients: AUROC A 0.8682, B 0.8983, B minus A +0.0300 (95% CI +0.0182 to +0.0429), against a seed noise of 0.0106
  grazpedwri, 16157 pictures of 4873 patients: AUROC A 0.9654, B 0.9745, B minus A +0.0091 (95% CI +0.0072 to +0.0109), against a seed noise of 0.0030
KEPT
ledger experiment/second look 2 gate: v140+v141 against v1: 3.9 s, 10 number(s)"""

    per, verdict = parse_compare(exact_text, 0)
    check("parse_compare exact text gives KEPT", verdict == "KEPT")
    check("parse_compare exact text gives 2 per-collection lines", len(per) == 2 and per[0].startswith("fracatlas") and per[1].startswith("grazpedwri"))

    exact_text_dropped = exact_text.replace("KEPT", "DROPPED")
    per, verdict = parse_compare(exact_text_dropped, 0)
    check("parse_compare with DROPPED gives DROPPED", verdict == "DROPPED")

    no_verdict = """second look 2 gate: v140+v141 against v1:
  fracatlas, 3226 pictures of 3226 patients: AUROC A 0.8682, B 0.8983, B minus A +0.0300 (95% CI +0.0182 to +0.0429), against a seed noise of 0.0106
  grazpedwri, 16157 pictures of 4873 patients: AUROC A 0.9654, B 0.9745, B minus A +0.0091 (95% CI +0.0072 to +0.0109), against a seed noise of 0.0030
ledger experiment/second look 2 gate: v140+v141 against v1: 3.9 s, 10 number(s)"""
    per, verdict = parse_compare(no_verdict, 0)
    check("parse_compare without verdict line gives ERROR", verdict == "ERROR")

    per, verdict = parse_compare(exact_text, 1)
    check("parse_compare with non-zero returncode gives ERROR", verdict == "ERROR")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            preds.WORK = os.path.join(tmp, "work")
            g["STATE"], g["CARD"] = os.path.join(tmp, "final_eval.json"), os.path.join(tmp, "SCORECARD.md")
            os.makedirs(preds.WORK)
            names = datasets.NAMES
            rows = []
            for d in names:
                for i in range(6):
                    rows.append({"dataset": d, "image": "%s%d" % (d, i), "patient": "p%d" % i, "fold": i % 5,
                                 "truth": i % 2, "score": 0.2 + 0.1 * i})
            oof = pd.DataFrame(rows)
            models = {}
            for v in (1, 140, 141):
                d = os.path.join(tmp, "m%d" % v)
                os.makedirs(d)
                with open(os.path.join(d, "meta.json"), "w") as fh:
                    json.dump({"split_sha256": "s", "datasets": names}, fh)
                for k in range(5):
                    open(os.path.join(d, "fold%d.pt" % k), "w").close()
                oof.assign(score=oof.score + (0.05 if v == 141 else 0)).to_csv(os.path.join(d, "oof_dev.csv"), index=False)
                models[v] = d
            fake = types.SimpleNamespace(TRIAL_FROM=100, model_dir=lambda v: models[v],
                                         youden=lambda s, t: float(np.mean(s)))
            g["_detector"] = lambda: fake
            first = {"version": 1, "reference": "meta", "test_opened": "t", "finished": "t2"}

            # (a) refusals
            check("(a) first look unfinished", exits(begin, {**first, "finished": None}, [140]) is not None)
            check("(a) same versions finished", exits(begin, {**first, "looks": [
                {"look": 2, "versions": [140, 141], "finished": "x", "test_opened": "x"}]}, [140, 141]) is not None)
            check("(a) another look open", exits(begin, {**first, "looks": [
                {"look": 2, "versions": [142], "finished": None, "test_opened": "x"}]}, [140, 141]) is not None)
            check("(a) trial floor", exits(checks, first, [99]) is not None)

            # (b) a failed gate records a refusal, writes no look, reads nothing
            calls = []
            g["read_test"] = lambda *a: calls.append("read")
            g["run_compare"] = lambda what, a, b: (["fracatlas ..."], "INCONCLUSIVE")
            st = dict(first)
            check("(b) failed gate exits", exits(begin, st, [140, 141]) is not None)
            check("(b) refusal recorded, no look, no read",
                  len(st.get("gate_refusals", [])) == 1 and not st.get("looks") and not calls)

            # (c) entry, thresholds and latch on disk before the read
            seen = {}

            def stub_read(entry, batch, card_gb):
                disk = json.load(open(g["STATE"], encoding="utf-8"))["looks"][-1]
                seen.update(ok=bool(disk["thresholds"] and disk["test_opened"] and disk["gate"]["passed"]
                                    and disk["versions"] == [140, 141]))
                raise Stop()

            g["read_test"], g["write_dev_preds"] = stub_read, lambda e: None
            g["run_compare"] = lambda what, a, b: (["fracatlas ..."], "KEPT")
            with open(os.path.join(preds.WORK, "pred_meta_fracatlas_test.csv"), "w") as fh:
                fh.write("x\n")
            st = dict(first)
            entry = begin(st, [140, 141])
            check("(c) numbered 2, oof written",
                  entry["look"] == 2 and os.path.exists(os.path.join(entry["folder"], "oof_dev.csv")))
            try:
                second_look_steps(st, entry)
            except Stop:
                seen["reached"] = True
            check("(c) on disk before the read", seen.get("reached") and seen.get("ok"))
            check("(c) baseline copied",
                  os.path.exists(os.path.join(entry["folder"], "pred_meta_fracatlas_test.csv")))

            # (d) resume: done steps skipped, no new gate
            def boom(*a):
                raise AssertionError("gated again")

            g["run_compare"] = boom
            ran = []
            g["read_test"] = lambda e, b, c: ran.append("read")
            g["score_look"] = lambda s, e: ran.append("score") or record(s, e, ship_result={})
            g["against_first"], g["write_dev_preds"] = lambda e: [], lambda e: ran.append("dev")
            st2 = json.load(open(g["STATE"], encoding="utf-8"))
            again = begin(st2, [140, 141])
            thr = dict(again["thresholds"])
            step(st2, again, "copy baselines", lambda: ran.append("copy"))
            step(st2, again, "candidate dev predictions", lambda: ran.append("dev"))
            step(st2, again, "read test", lambda: ran.append("read"))
            check("(d) resumed same look, thresholds kept",
                  again["look"] == 2 and again["thresholds"] == thr and len(st2["looks"]) == 1)
            check("(d) done steps skipped", "copy" not in ran and "copy baselines" in again["done"])

            # (e) folders
            sub = os.path.join(tmp, "sub")
            before = set(os.listdir(preds.WORK))
            out = preds.write("cnn", "grazpedwri", "test", [{"patient": "p", "image": "i", "in_scope": 1}], folder=sub)
            preds.collect(sub)
            check("(e) folder write only inside it", os.path.dirname(out) == sub
                  and set(os.listdir(preds.WORK)) == before and os.path.exists(os.path.join(sub, "predictions.csv")))
            check("(e) default path unchanged",
                  preds.path("cnn", "a", "test") == os.path.join(preds.WORK, "pred_cnn_a_test.csv"))

            # (f) the paired section
            fa = pd.DataFrame({"image": ["a", "b", "c"], "s": [1.0, 2.0, 3.0]})
            fb = pd.DataFrame({"image": ["a", "b", "c"], "s": [0.5, 1.5, 2.0]})
            score.paired = lambda a, b, st, m: (float((a.s - b.s).mean()), -0.1, 0.9, 10, len(a), 3)
            score.stratum_order = lambda n: ["all"]
            table = paired_table({(CANDIDATE, d, "test"): fa for d in names},
                                 {(CANDIDATE, d, "test"): fb for d in names}, {d: ["all"] for d in names})
            check("(f) paired rows", len(table) == 2 * 3 and abs(table[0]["difference"] - 2.0 / 3) < 1e-9)
            entry.update(ship_result={"difference": "0.1", "ci_low": "0.0", "ci_high": "0.2", "n_images": 3,
                                      "n_patients": 3}, paired=table)
            text = "\n".join(section(first, entry))
            check("(f) section text", "No test number chose anything" in text
                  and "| fracatlas | all | auroc |" in text and "KEPT" in text)

            # (g) split_sha256 differs but oof rows match: passes checks with a message
            d = os.path.join(tmp, "m142")
            os.makedirs(d)
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump({"split_sha256": "different_hash", "datasets": names}, fh)
            for k in range(5):
                open(os.path.join(d, "fold%d.pt" % k), "w").close()
            oof.to_csv(os.path.join(d, "oof_dev.csv"), index=False)
            models[142] = d
            exit_msg = exits(checks, first, [142])
            check("(g) split differs but rows match", exit_msg is None)

            # oof rows differ: should fail
            d = os.path.join(tmp, "m143")
            os.makedirs(d)
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump({"split_sha256": "s", "datasets": names}, fh)
            for k in range(5):
                open(os.path.join(d, "fold%d.pt" % k), "w").close()
            oof_changed = oof.copy()
            oof_changed.loc[oof_changed.fold == 0, "truth"] = (1 - oof_changed.loc[oof_changed.fold == 0, "truth"]).astype(int)
            oof_changed.to_csv(os.path.join(d, "oof_dev.csv"), index=False)
            models[143] = d
            exit_msg = exits(checks, first, [143])
            check("(g) oof rows differ", exit_msg is not None)
    except Exception as e:  # noqa: BLE001
        failures.append("exception %r" % e)
    finally:
        g.update(keep)
        preds.WORK, score.paired, score.stratum_order = keep_work, keep_paired, keep_order
    if failures:
        raise SystemExit("selftest: FAILED: " + "; ".join(failures))
    print("selftest ok")


if __name__ == "__main__":
    main()
