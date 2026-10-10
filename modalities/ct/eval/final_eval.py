"""The CT final evaluation: opens the held-out test split ONCE and scores every method on it together.

The order, and each step is written into work/eval/ct/final_eval.json before the next one starts:

1. Refuse if the test split has been opened before. Two latches say so: the state file in the work
   folder, and the run ledger, which is committed and so survives a wiped work folder.
2. Refuse unless everything that must exist before the test is seen does exist: five fold models
   whose files match the checksums written when they were trained, and out-of-fold development
   rows for the segmenter and the baselines covering the same scans.
3. Choose the reference: the baseline with the highest development mean Dice. Recorded before step
   4, so the thing the segmenter must beat is fixed before the test is seen.
4. Open the test. Baselines first (their priors and atlas built from all of development), then the
   test scans into the cache, then the segmenter, all five fold models averaged.
5. Score both splits with that reference, and have the independent recomputation check every cell
   of both. If the two disagree on any cell, nothing is published: the run stops with the test
   open and the disagreement on record.
6. The ship rule, declared in METRICS_SPEC.md before anything was trained: the segmenter ships only
   if the 95% interval of its mean Dice minus the reference's, on the test scans, lies above zero.

    python modalities/ct/eval/final_eval.py [--version 1]
    python modalities/ct/eval/final_eval.py --finish      carry on after a step that crashed

The second look (decided 2026-10-07, docs/HANDOFF.md): a model kept on development, chosen with no test
number, may read the spent test set ONCE more, labelled a second look, only when its development gain
over the model the test last read is beyond noise. The result is reported whether higher or lower,
with the test's count of looks; the first look's files and SCORECARD numbers stay as they are.

    python modalities/ct/eval/final_eval.py --second-look --version N
    python modalities/ct/eval/final_eval.py --second-look --version N --with-version M --tag T --noise X
                                                          a PAIR: version N's and version M's nets read together
    python modalities/ct/eval/final_eval.py --finish      also carries a crashed second look on
    python modalities/ct/eval/final_eval.py --selftest    CPU, seconds, temp folders, no real data

It refuses unless the first look finished, a look of the same read has not finished, and that read is
not the one the test last read. A read is version N alone, or the pair N with M: version N's look does
not block the pair, and a finished pair look blocks a repeat of the pair.

THE DEVELOPMENT GATE runs before anything touches the test: per-scan mean Dice of the read's out-of-fold
rows minus the last-read model's development rows, paired over the shared scans, 95% interval from 2000
scan resamples (seed 0). A single version reads work_dir(N)/oof_fold0..4.csv; a pair reads
work_dir(N)/oof_fold0..4_T.csv (--tag T names them, as `predict dev --tag` does), all five of each. The
gate passes only if the low end of the interval is above --noise X (default 0). On a pass its numbers,
with the noise, versions and tag, go into state['looks'] with the ledger latch ("ct test opened second
look vN", or "vN+M" for a pair) before any test scan is read; on a fail they go into
state['gate_refusals'] and nothing opens. A pair's test read is version N's five nets and version M's
five, every net averaged with equal weight (ten nets); both versions' fold files must match their written
checksums before anything opens. Every output goes to work/eval/ct/look<L>-v<N>/ or look<L>-v<N>+<M>/ (pred
files, cards, tsv files, second_look.md) and a section is appended to SCORECARD.md. The reference stays
the first look's. Each step is recorded in the look's own 'done' list.

A FINE version (its last phase at 1.5 mm) reads the test at 1.5 mm, so after the latch, and before its own read, the
look first builds the test scans' 1.5 mm cache ('test 1.5 mm cache', prepare.py hi15 --split test --look L) and, when
it has a coarse prior P, that prior's maps on the test ('coarse maps on test', segmenter.py coarse --split test --look L,
all five of P's nets averaged). Both refuse unless the look is open.

`--finish` does not open anything a second time. It picks up at the first step not yet recorded as
done, and exists because a crash halfway through should cost a re-run of that step, not the test.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import datasets  # noqa: E402
import score  # noqa: E402
from core import ledger, paths  # noqa: E402

OUT = os.path.join(paths.WORK, "eval", "ct")
STATE = os.path.join(OUT, "final_eval.json")
CANDIDATE = "segmenter"
SHIP = ("all", CANDIDATE, "headline_dice_minus_reference")
LEDGER_NAME = "ct test opened"
PY = sys.executable


def record(state, scope=None, **kw):
    """Write into the state (or into one look's entry, `scope`) and save the state file."""
    (scope if scope is not None else state).update(kw)
    entry = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    if scope is not None:
        entry["look"] = scope["look"]
    state.setdefault("log", []).append(entry)
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def run(state, step, *args, log=None, scope=None, fn=None):
    """One step as its own process, with the test unlocked for it. Skipped if already done.

    `scope` is a look's entry: its 'done' list and timings live there. `fn` runs in this process
    instead of a subprocess (copies and write-ups)."""
    holder = scope if scope is not None else state
    if step in holder.get("done", []):
        print("= %s: already done" % step, flush=True)
        return
    began = time.time()
    if fn is not None:
        print("+ %s" % step, flush=True)
        fn()
    else:
        print("+ %s: %s" % (step, " ".join(args)), flush=True)
        env = {**os.environ, "RAIDIOLOGY_OPEN_TEST": "yes"}
        if log:
            with open(log, "w", encoding="utf-8") as out:
                got = subprocess.run([PY, "-u", *args], cwd=HERE, env=env, stdout=out, stderr=subprocess.PIPE,
                                     text=True, check=False)
            with open(log, "a", encoding="utf-8") as out:
                out.write(got.stderr or "")
            if got.returncode:
                raise SystemExit("%s failed (exit %d). Its output is in %s" % (step, got.returncode, log))
        else:
            subprocess.run([PY, "-u", *args], cwd=HERE, env=env, check=True)
    took = {**holder.get("step_seconds", {}), step: round(time.time() - began, 1)}
    record(state, scope, done=holder.get("done", []) + [step], step_seconds=took)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cells(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 6:
                out[tuple(p[:3])] = tuple(float(v) for v in p[3:])
    return out


def fold_files(version):
    """A version's five fold models, each file's checksum the one written at training. Returns meta, or exits."""
    import segmenter  # noqa: PLC0415  imports torch, which the latch check above does not need

    d = segmenter.model_dir(version)
    meta_file = os.path.join(d, "meta.json")
    if not os.path.exists(meta_file):
        raise SystemExit("No trained segmenter at %s." % d)
    meta = json.load(open(meta_file, encoding="utf-8"))
    folds = {f["fold"]: f for f in meta["folds"]}
    if sorted(folds) != list(range(segmenter.FOLDS)):
        raise SystemExit("Folds trained: %s. All %d are needed." % (sorted(folds), segmenter.FOLDS))
    for k, f in folds.items():
        path = os.path.join(d, f["file"])
        if not os.path.exists(path):
            raise SystemExit("fold %d's weights are missing (%s)." % (k, path))
        if sha256(path) != f["sha256"]:
            raise SystemExit("fold %d's weights are not the file that was trained (checksum differs)." % k)
    return meta


def ready(version):
    """Everything that has to be true before the test is seen. Returns what was found, or exits."""
    meta = fold_files(version)
    split = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    want = set(split["dev_folds"])
    for name in ("baselines", CANDIDATE):
        f = os.path.join(OUT, "pred_%s_dev.csv" % name)
        if not os.path.exists(f):
            raise SystemExit("No development rows for %s (%s)." % (name, f))
        got = set(pd.read_csv(f, usecols=["scan"], dtype=str)["scan"])
        if got != want:
            raise SystemExit("%s covers %d development scans, the split has %d." % (f, len(got), len(want)))
    return meta


def choose_reference():
    """The baseline with the highest development mean Dice, from a fresh scoring of development."""
    tsv = os.path.join(OUT, "reference_choice.tsv")
    subprocess.run([PY, "-u", "score.py", "dev", "--out", os.path.join(OUT, "reference_choice.md"),
                    "--tsv", tsv], cwd=HERE, check=True)
    dice = {k[1]: v[0] for k, v in cells(tsv).items()
            if k[0] == "all" and k[2] == "headline_dice" and k[1] != CANDIDATE}
    if not dice:
        raise SystemExit("No baseline has a development mean Dice: run baselines.py predict dev first.")
    return max(dice, key=dice.get), dice


def card(path, title):
    """One split's scorecard as a section of the published one: headings one level down."""
    out = []
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# CT scorecard"):
            line = "## " + title
        elif line.startswith("#"):
            line = "#" + line
        elif line.startswith("---") or line.startswith("Not a medical device."):
            continue
        out.append(line)
    return out


def publish(state, out_path=None):
    """SCORECARD.md: the decision, then the held-out test, then development, from the scored files.

    `out_path` is for rehearsing the assembly on stand-in files without touching the real card."""
    test = cells(os.path.join(OUT, "scorecard_test.tsv"))
    value, low, high = test[SHIP]
    ships = low > 0
    reference = state["reference"]
    lines = [
        "# CT scorecard", "",
        "**Where is each organ and bone in this scan?** One label per voxel over the 117 structures "
        "the TotalSegmentator dataset marks, scored per structure by Dice overlap against the "
        "dataset's own expert masks, exactly as [METRICS_SPEC.md](METRICS_SPEC.md) defines it. Every "
        "number below was computed by `score.py` and recomputed by `recompute.py`, written from the "
        "specification alone by a worker that never saw the scorer; the two agree on all %d cells of "
        "the test split and all %d of development." % (state["cells_test"], state["cells_dev"]), "",
        "**The dataset publishes no patient identifier**, so the unit is the scan and two scans of "
        "one person could be treated as independent. Read every interval with that in mind.", "",
        "## Ship decision", "",
        "The reference is **%s**, the strongest baseline on development (%s), chosen and recorded "
        "before the test split was opened on %s." % (
            reference, ", ".join("%s %.4f" % kv for kv in sorted(state["dev_dice"].items())),
            state["test_opened"]), "",
        "The rule, written down before anything was trained: the segmenter ships only if the 95% "
        "interval of its mean Dice minus the reference's, on the held-out test scans, lies above zero.", "",
        "- **segmenter minus %s: %+.4f (95%% CI %+.4f to %+.4f)** on %d test scans: %s"
        % (reference, value, low, high, state["test_scans"],
           "**it ships.**" if ships else "**it does not ship** (the interval reaches zero or below)."), ""]
    lines += card(os.path.join(OUT, "card_test.md"), "Held-out test, opened once") + [""]
    lines += card(os.path.join(OUT, "card_dev.md"),
                  "Development, each scan scored by a model that never saw it") + [""]
    lines += ["---", "", "Not a medical device. These are measurements on a public research dataset, "
              "not a claim about anybody's scan."]
    with open(out_path or os.path.join(HERE, "SCORECARD.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"difference": value, "ci_low": low, "ci_high": high, "ships": bool(ships)}


# ---- the second look -------------------------------------------------------------------------

RESAMPLES = 2000


def looks_of(state):
    return state.get("looks", [])


def read_of(entry):
    """A read's identity: (version, with_version). An entry from before pairs is a single version's read."""
    return entry["version"], entry.get("with_version")


def name_of(version, with_version=None):
    """A read's name in files and the ledger: 'v116', or the pair 'v116+115'."""
    return "v%d" % version if with_version is None else "v%d+%d" % (version, with_version)


def read_words(version, with_version=None):
    """A read's name in the scorecard's words: 'version 116', or 'version 116 with version 115'."""
    if with_version is None:
        return "version %d" % version
    return "version %d with version %d" % (version, with_version)


def look_folder(entry):
    return os.path.join(OUT, "look%d-%s" % (entry["look"], name_of(*read_of(entry))))


def last_read(state):
    """(version, with_version, development rows file) the test last read: the first look's, or the last finished look's."""
    done = [l for l in looks_of(state) if l.get("finished")]
    if done:
        last = done[-1]
        return last["version"], last.get("with_version"), os.path.join(look_folder(last), "pred_%s_dev.csv" % CANDIDATE)
    return state["version"], None, os.path.join(OUT, "pred_%s_dev.csv" % CANDIDATE)


def last_dev_rows(state):
    return pd.read_csv(last_read(state)[2], dtype={"scan": str}, keep_default_na=False)


def new_dev_rows(version, tag=None):
    """Version N's development rows: the five out-of-fold files (a pair's, named by its tag), all of them or an exit."""
    import segmenter  # noqa: PLC0415

    parts = []
    suffix = "" if tag is None else "_" + tag
    for k in range(segmenter.FOLDS):
        f = os.path.join(segmenter.work_dir(version), "oof_fold%d%s.csv" % (k, suffix))
        if not os.path.exists(f):
            raise SystemExit("Version %d lacks %s: all %d folds' development rows are needed."
                             % (version, f, segmenter.FOLDS))
        parts.append(pd.read_csv(f, dtype={"scan": str}, keep_default_na=False))
    return pd.concat(parts, ignore_index=True)


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=HERE).stdout.strip()


def rows_of(version, with_version, tag):
    """The read's development rows: a single version's five files, or a pair's five tagged ones."""
    return new_dev_rows(version) if with_version is None else new_dev_rows(version, tag=tag)


def dev_gate(last, new, resamples=RESAMPLES, noise=0.0):
    """Paired per-scan mean Dice, new minus last, with a 95% interval from scan resamples (seed 0).
    It passes only when the low end of the interval is above `noise`."""
    a_scans, b_scans = set(last["scan"]), set(new["scan"])
    if a_scans != b_scans:
        raise SystemExit("The development rows cover different scans: %d in the last-read model's, %d in "
                         "the new one's, %d shared. No comparison is made." % (
                             len(a_scans), len(b_scans), len(a_scans & b_scans)))
    a, b = score.scan_dice(last), score.scan_dice(new)
    shared = a.index.intersection(b.index)
    if not len(shared):
        raise SystemExit("No scan has a mean Dice in both sets of development rows.")
    diff = (b.loc[shared] - a.loc[shared]).to_numpy()
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    low, high = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    return {"scans": int(len(diff)), "last_mean": float(a.loc[shared].mean()),
            "new_mean": float(b.loc[shared].mean()), "difference": float(diff.mean()),
            "ci_low": low, "ci_high": high, "resamples": resamples, "noise": float(noise),
            "passed": bool(low > noise)}


def refuse_second(state, version, with_version=None):
    name = name_of(version, with_version)
    if not state.get("finished"):
        raise SystemExit("The first look has not finished (see %s): a second look needs it finished." % STATE)
    for l in looks_of(state):
        if read_of(l) == (version, with_version) and l.get("finished"):
            raise SystemExit("%s already had its look (look %d, finished %s)." % (name, l["look"], l["finished"]))
        if not l.get("finished"):
            raise SystemExit("Look %d (%s) is open and unfinished: --finish carries it on."
                             % (l["look"], name_of(*read_of(l))))
    if last_read(state)[:2] == (version, with_version):
        raise SystemExit("%s is the read the test last read; a look of it again is no new look." % name)


def start_look(state, version, with_version=None, tag=None, noise=0.0):
    """Checks, the development gate, then the latch and the looks entry. Nothing here reads a test scan.

    With `with_version` the read is a pair: version's and with_version's nets together, its development
    rows named by `tag`. The gate's low end must be above `noise`."""
    if version is None:
        raise SystemExit("--second-look needs --version N.")
    if (with_version is None) != (tag is None):
        raise SystemExit("A pair needs both --with-version M and --tag T (the development rows' tag).")
    if with_version == version:
        raise SystemExit("--with-version must name the other version of the pair.")
    refuse_second(state, version, with_version)
    last = last_read(state)
    if with_version is not None:
        fold_files(version)
        fold_files(with_version)
    gate = {**dev_gate(last_dev_rows(state), rows_of(version, with_version, tag), noise=noise),
            "with_version": with_version, "tag": tag}
    number = len(looks_of(state)) + 2
    print("development gate, %s minus %s: %+.4f (95%% CI %+.4f to %+.4f) on %d scans, noise %.4f: %s" % (
        name_of(version, with_version), name_of(last[0], last[1]), gate["difference"], gate["ci_low"],
        gate["ci_high"], gate["scans"], noise, "PASS" if gate["passed"] else "FAIL"), flush=True)
    entry = {"look": number, "version": version, "with_version": with_version, "tag": tag,
             "compared_with_version": last[0], "compared_with_with_version": last[1],
             "gate": gate, "commit": git_head(), "done": []}
    if not gate["passed"]:
        # Recorded, but it is not a look: the test stays unopened and the next look keeps its number.
        refusal = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   **{k: entry[k] for k in ("version", "with_version", "tag", "compared_with_version",
                                            "compared_with_with_version", "gate")}}
        record(state, gate_refusals=state.get("gate_refusals", []) + [refusal])
        raise SystemExit("The development gain of %s over %s is not beyond noise (low end %+.4f, needs above %+.4f). "
                         "The test is not opened." % (name_of(version, with_version), name_of(last[0], last[1]),
                                                      gate["ci_low"], noise))
    ledger.write("final_eval", "%s second look %s" % (LEDGER_NAME, name_of(version, with_version)), 0.0,
                 {"version": (version, "count"), "look": (number, "count")},
                 note="look %d, gate %+.4f low %+.4f, noise %.4f, reference %s" % (
                     number, gate["difference"], gate["ci_low"], noise, state["reference"]))
    entry["opened"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record(state, looks=looks_of(state) + [entry])
    return state["looks"][-1]


def section(state, entry):
    """The second look's results as markdown lines, from the two scorecard tsv files."""
    ref = state["reference"]
    mine = cells(os.path.join(look_folder(entry), "scorecard_test.tsv"))
    first = cells(os.path.join(OUT, "scorecard_test.tsv"))
    g = entry["gate"]
    mean_key = ("all", CANDIDATE, "headline_dice")

    def fmt(c, key):
        v, lo, hi = c[key]
        return "%.4f (95%% CI %.4f to %.4f)" % (v, lo, hi)

    def fmt_d(c):
        v, lo, hi = c[SHIP]
        return "%+.4f (95%% CI %+.4f to %+.4f)" % (v, lo, hi)

    n, m = entry["version"], entry.get("with_version")
    last_n, last_m = entry["compared_with_version"], entry.get("compared_with_with_version")
    agreed = "recompute test" in entry["done"] and "recompute dev" in entry["done"]
    lines = [
        "This is look %d of the CT test split. The first look (version %d, opened %s) is above, unchanged; "
        "this one is labelled a second look, and the result is reported whether it is higher or lower."
        % (entry["look"], state["version"], state["test_opened"]), ""]
    if m is not None:
        lines += [
            "This read is a **pair**: version %d's five nets and version %d's five, read together with every "
            "net averaged with equal weight (ten nets), under version %d's reading. Its development rows are "
            "version %d's out-of-fold files tagged `%s`." % (n, m, n, n, entry["tag"]), ""]
    if entry.get("fine"):
        lines += ["This read is at **1.5 mm**: each test scan was read from its own 1.5 mm cache%s." % (
            "" if entry.get("prior") is None else ", with version %d's 3 mm coarse map as the prior" % entry["prior"]), ""]
    return lines + [
        "**Development gate**, passed before the test was touched: %s minus %s on %d development scans, "
        "paired, per-scan mean Dice: %+.4f (95%% CI %+.4f to %+.4f, %d resamples), passing only with a low end "
        "above noise %.4f. %s: %.4f. %s: %.4f." % (
            read_words(n, m), read_words(last_n, last_m), g["scans"], g["difference"], g["ci_low"], g["ci_high"],
            g["resamples"], g.get("noise", 0.0), read_words(n, m), g["new_mean"],
            read_words(last_n, last_m), g["last_mean"]), "",
        "The reference is still **%s**, fixed at the first look." % ref, "",
        "| | test mean Dice | minus %s |" % ref, "|---|---|---|",
        "| %s (look %d) | %s | %s |" % (read_words(n, m), entry["look"], fmt(mine, mean_key), fmt_d(mine)),
        "| version %d (look 1, from its scorecard) | %s | %s |" % (state["version"], fmt(first, mean_key),
                                                                  fmt_d(first)), "",
        "The independent recomputation (`recompute.py`) %s on all %d test cells and %d development cells of "
        "this look." % ("agrees" if agreed else "has not agreed", entry.get("cells_test", 0),
                        entry.get("cells_dev", 0)), "",
        "Files: `%s`." % os.path.relpath(look_folder(entry), paths.WORK).replace(os.sep, "/")]


def fine_of(version):
    """Whether a version reads at 1.5 mm (a fine version), and the version whose coarse maps it reads as its prior
    (None without one): the standing prior_from when set, since the second look builds on what the model reads."""
    import segmenter  # noqa: PLC0415  imports torch, which the gate above does not need
    meta = json.load(open(os.path.join(segmenter.model_dir(version), "meta.json"), encoding="utf-8"))
    return segmenter.is_fine_version(meta), segmenter.reading_prior(meta)


def second_look_steps(state, entry):
    folder, version = look_folder(entry), str(entry["version"])
    ref = state["reference"]
    words = read_words(entry["version"], entry.get("with_version"))
    os.makedirs(folder, exist_ok=True)

    def copy_baselines():
        for split in ("test", "dev"):
            shutil.copy2(os.path.join(OUT, "pred_baselines_%s.csv" % split),
                         os.path.join(folder, "pred_baselines_%s.csv" % split))

    def dev_rows():
        rows_of(entry["version"], entry.get("with_version"), entry.get("tag")).to_csv(
            os.path.join(folder, "pred_%s_dev.csv" % CANDIDATE), index=False)

    def write_md():
        record(state, entry,
               cells_test=len(cells(os.path.join(folder, "scorecard_test.tsv"))),
               cells_dev=len(cells(os.path.join(folder, "scorecard_dev.tsv"))))
        head = ["# Second look, %s (look %d of the CT test)" % (words, entry["look"]), ""]
        with open(os.path.join(folder, "second_look.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(head + section(state, entry)) + "\n")

    def append_card():
        path = os.path.join(HERE, "SCORECARD.md")
        title = "## Second look, %s (look %d of the test)" % (words, entry["look"])
        text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if title in text:          # a crash between the write and the record must not append twice
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + title + "\n\n" + "\n".join(section(state, entry)) + "\n")

    # A pair's test read names its partner, and a fine version's names the look: segmenter refuses either without an open look.
    run(state, "copy baselines rows", scope=entry, fn=copy_baselines)
    run(state, "version dev rows", scope=entry, fn=dev_rows)
    fine, prior = fine_of(entry["version"])
    partner = [] if entry.get("with_version") is None else ["--with-version", str(entry["with_version"])]
    look = ["--look", str(entry["look"])] if partner or fine else []
    if fine:
        record(state, entry, fine=True, prior=prior)
        run(state, "test 1.5 mm cache", "prepare.py", "hi15", "--split", "test", *look, scope=entry)
        if prior is not None:
            run(state, "coarse maps on test", "segmenter.py", "coarse", "--version", str(prior), "--split", "test",
                *look, scope=entry)
    run(state, "segmenter on test", "segmenter.py", "predict", "test", "--version", version,
        "--out-dir", folder, *partner, *look, scope=entry)
    for split in ("test", "dev"):
        run(state, "score %s" % split, "score.py", split, "--reference", ref, "--dir", folder,
            "--out", os.path.join(folder, "card_%s.md" % split),
            "--tsv", os.path.join(folder, "scorecard_%s.tsv" % split), scope=entry)
    for split in ("test", "dev"):
        run(state, "recompute %s" % split, "recompute.py", split, "--reference", ref, "--dir", folder,
            "--compare", os.path.join(folder, "scorecard_%s.tsv" % split),
            log=os.path.join(folder, "recompute_%s.log" % split), scope=entry)
    run(state, "write second_look", scope=entry, fn=write_md)
    run(state, "append scorecard", scope=entry, fn=append_card)


def second_look(state, entry):
    t0 = time.time()
    name = name_of(entry["version"], entry.get("with_version"))
    with ledger.run("final_eval", "ct second look %s (look %d)" % (name, entry["look"])) as led:
        second_look_steps(state, entry)
        record(state, entry, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        got = cells(os.path.join(look_folder(entry), "scorecard_test.tsv"))[SHIP]
        led.number("difference", got[0]).number("ci_low", got[1]).number("ci_high", got[2])
        led.number("gate_difference", entry["gate"]["difference"]).number("gate_low", entry["gate"]["ci_low"])
        led.number("look", entry["look"], "count")
        led.note_is("look %d, %s, reference %s" % (entry["look"], name, state["reference"]))
    print(json.dumps({"look": entry["look"], "version": entry["version"], "with_version": entry.get("with_version"),
                      "difference": got[0],
                      "ci_low": got[1], "ci_high": got[2], "seconds": round(time.time() - t0, 1)}, indent=1),
          flush=True)


def selftest():
    """The second look's logic on synthetic rows in temp folders: no data, no subprocess, seconds."""
    import tempfile  # noqa: PLC0415

    import recompute  # noqa: PLC0415

    g = globals()
    keep = {k: g[k] for k in ("OUT", "STATE", "ledger", "run", "new_dev_rows", "last_dev_rows", "git_head")}
    failures = []

    def check(name, ok):
        if not ok:
            failures.append(name)

    def refused(fn, *args):
        try:
            fn(*args)
        except SystemExit as e:
            return str(e)
        return None

    def rows(scans, gain, structures=("a", "b")):
        out = []
        for i, s in enumerate(scans):
            for st in structures:
                out.append({"scan": s, "structure": st, "in_scope": 1, "truth_voxels": 1000,
                            "pred_voxels": 1000, "both_voxels": 600 + 10 * (i % 5) + gain(i)})
        return pd.DataFrame(out)

    class Stop(Exception):
        pass

    class FakeLedger:
        def __init__(self):
            self.written = []

        def write(self, *a, **k):
            self.written.append((a, k))

    try:
        scans = ["s%02d" % i for i in range(14)]
        last = rows(scans, lambda i: 0)
        better = rows(scans, lambda i: 20 + (i % 3))
        same = rows(scans, lambda i: 0)
        worse = rows(scans, lambda i: -15)
        check("(a) better passes", dev_gate(last, better)["passed"])
        check("(a) equal fails", not dev_gate(last, same)["passed"])
        check("(a) worse fails", not dev_gate(last, worse)["passed"])
        check("(b) mismatched scans refused", refused(dev_gate, last, rows(scans[:-1], lambda i: 20)) is not None)

        with tempfile.TemporaryDirectory() as tmp:
            g["OUT"], g["STATE"] = tmp, os.path.join(tmp, "final_eval.json")
            fake = FakeLedger()
            g["ledger"], g["git_head"] = fake, lambda: "abc"
            g["last_dev_rows"], g["new_dev_rows"] = lambda st: last, lambda v: better
            base = {"version": 1, "reference": "ref", "test_opened": "t", "finished": None, "looks": []}
            check("(c) first look unfinished refused", refused(start_look, dict(base), 116) is not None)
            fin = {**base, "finished": "t2"}
            check("(c) a finished look of N refused",
                  refused(start_look, {**fin, "looks": [{"look": 2, "version": 116, "finished": "t3"}]}, 116)
                  is not None)
            check("(c) last-read version refused", refused(start_look, dict(fin), 1) is not None)
            check("(c) an open look blocks another",
                  refused(start_look, {**fin, "looks": [{"look": 2, "version": 117, "finished": None}]}, 116)
                  is not None)
            check("(c) refusals wrote no latch", not fake.written)

            def stub(state, step, *a, **k):
                check("(e) latch before first step",
                      len(fake.written) == 1 and "second look v116" in fake.written[0][0][1])
                disk = json.load(open(g["STATE"], encoding="utf-8"))
                check("(e) looks entry on disk before first step",
                      bool(disk["looks"]) and disk["looks"][-1]["version"] == 116
                      and disk["looks"][-1]["gate"]["passed"])
                raise Stop()

            g["run"] = stub
            state = dict(fin)
            entry = start_look(state, 116)
            check("(e) look numbered 2", entry["look"] == 2)
            ran = False
            try:
                second_look_steps(state, entry)
            except Stop:
                ran = True
            check("(e) the stubbed runner was reached", ran)

            fake.written.clear()
            g["new_dev_rows"] = lambda v: same
            state2 = dict(fin, looks=[])
            check("(a) a failed gate exits", refused(start_look, state2, 116) is not None)
            check("(a) a failed gate wrote no latch and no look", not fake.written and not state2["looks"])

            frame = rows(scans[:3], lambda i: 0)
            frame.insert(0, "method", "m")
            frame["split"], frame["fold"], frame["drop_reason"] = "dev", "0", ""
            os.makedirs(os.path.join(tmp, "sub"))
            frame.to_csv(os.path.join(tmp, "pred_m_dev.csv"), index=False)
            frame.to_csv(os.path.join(tmp, "sub", "pred_m_dev.csv"), index=False)
            keep_work, keep_out, keep_idx = recompute.WORK, score.OUT, score.datasets.index
            try:
                recompute.WORK, score.OUT = tmp, tmp
                score.datasets.index = lambda: pd.DataFrame({score.datasets.SET["key"]: scans})
                check("(d) recompute: no --dir is the work folder", recompute.load("dev").equals(recompute.load("dev", tmp)))
                check("(d) score: no --dir is the eval folder", score.load("dev").equals(score.load("dev", tmp)))
                check("(d) --dir reads its own folder", len(score.load("dev", os.path.join(tmp, "sub"))) == len(frame))
            finally:
                recompute.WORK, score.OUT, score.datasets.index = keep_work, keep_out, keep_idx

        # The pair: fold files are read from fake model folders (segmenter's prepare.OUT points at the temp folder).
        import segmenter  # noqa: PLC0415

        def make_folds(d):
            os.makedirs(d, exist_ok=True)
            folds = []
            for k in range(segmenter.FOLDS):
                path = os.path.join(d, "fold%d.pt" % k)
                with open(path, "wb") as f:
                    f.write(b"net %d" % k)
                folds.append({"fold": k, "file": "fold%d.pt" % k, "sha256": sha256(path)})
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"folds": folds, "trained": "selftest"}, f)

        with tempfile.TemporaryDirectory() as tmp:
            keep_prep = segmenter.prepare.OUT
            segmenter.prepare.OUT = tmp
            try:
                g["OUT"], g["STATE"] = tmp, os.path.join(tmp, "final_eval.json")
                fake = FakeLedger()
                g["ledger"], g["git_head"] = fake, lambda: "abc"
                g["last_dev_rows"], g["new_dev_rows"] = lambda st: last, lambda v, tag=None: better
                for v in (116, 115):
                    make_folds(segmenter.model_dir(v))
                fin = {"version": 1, "reference": "ref", "test_opened": "t", "finished": "t2", "looks": []}
                alone = {"look": 2, "version": 116, "finished": "t3"}
                check("(f) v116 alone's finished look does not block the pair",
                      refused(start_look, {**fin, "looks": [alone]}, 116, 115, "pair115") is None
                      and fake.written[-1][0][1] == "ct test opened second look v116+115")
                fake.written.clear()
                check("(f) a finished pair look blocks its repeat",
                      refused(start_look, {**fin, "looks": [{**alone, "with_version": 115}]}, 116, 115, "pair115")
                      is not None and not fake.written)
                check("(f) a pair without its tag, or a tag without a pair, is refused",
                      refused(start_look, dict(fin), 116, 115, None) is not None
                      and refused(start_look, dict(fin), 116, None, "pair115") is not None
                      and refused(start_look, dict(fin), 116, 116, "pair115") is not None)
                check("(f) the pair's look folder is named for both versions",
                      os.path.basename(look_folder({"look": 3, "version": 116, "with_version": 115}))
                      == "look3-v116+115"
                      and os.path.basename(look_folder({"look": 2, "version": 116})) == "look2-v116")

                fake.written.clear()
                low = dev_gate(last, better)["ci_low"]
                check("(g) the pair's gate passes at noise 0", refused(start_look, dict(fin), 116, 115, "pair115")
                      is None and fake.written)
                fake.written.clear()
                blocked = dict(fin)
                check("(g) the pair's gate fails above its low end: recorded, nothing opened",
                      refused(start_look, blocked, 116, 115, "pair115", low + 0.001) is not None
                      and not fake.written and not blocked.get("looks")
                      and blocked["gate_refusals"][-1]["with_version"] == 115
                      and blocked["gate_refusals"][-1]["tag"] == "pair115"
                      and blocked["gate_refusals"][-1]["gate"]["noise"] == low + 0.001)

                d115 = segmenter.model_dir(115)
                fake.written.clear()
                os.remove(os.path.join(d115, "fold2.pt"))
                check("(h) a pair whose M fold file is missing is refused before the latch",
                      refused(start_look, dict(fin), 116, 115, "pair115") is not None and not fake.written)
                make_folds(d115)
                with open(os.path.join(d115, "fold3.pt"), "wb") as f:
                    f.write(b"tampered")
                check("(h) a pair whose M checksum differs is refused before the latch",
                      refused(start_look, dict(fin), 116, 115, "pair115") is not None and not fake.written)
                make_folds(d115)
                check("(h) and the intact pair is let through again",
                      refused(start_look, dict(fin), 116, 115, "pair115") is None)
            finally:
                segmenter.prepare.OUT = keep_prep

        # A fine version (1.5 mm, prior 116) through the whole look: the gate, the latch, the 1.5 mm cache and the prior's
        # coarse maps in that order, its own read, the scoring. The subprocesses are a fake that writes the scorecard
        # files; a crash in the coarse step is resumed by a second pass that opens nothing again.
        before = len(failures)
        order = ["copy baselines rows", "version dev rows", "test 1.5 mm cache", "coarse maps on test",
                 "segmenter on test", "score test", "score dev", "recompute test", "recompute dev",
                 "write second_look", "append scorecard"]
        with tempfile.TemporaryDirectory() as tmp:
            keep_prep, keep_here, keep_sub, keep_work = segmenter.prepare.OUT, g["HERE"], g["subprocess"], paths.WORK
            real = keep_sub
            paths.WORK = tmp                          # second_look.md names the look's folder relative to it
            segmenter.prepare.OUT = tmp
            try:
                g["OUT"], g["STATE"], g["HERE"] = tmp, os.path.join(tmp, "final_eval.json"), tmp
                g["run"] = keep["run"]                  # case (e)'s stub is still installed: this one runs the real steps
                fake = FakeLedger()
                g["ledger"], g["git_head"] = fake, lambda: "abc"
                g["last_dev_rows"], g["new_dev_rows"] = lambda st: last, lambda v, tag=None: better
                d121 = segmenter.model_dir(121)
                os.makedirs(d121, exist_ok=True)
                folds = []
                for k in range(segmenter.FOLDS):
                    path = os.path.join(d121, "fold%d.pt" % k)
                    with open(path, "wb") as f:
                        f.write(b"net %d" % k)
                    folds.append({"fold": k, "file": "fold%d.pt" % k, "sha256": sha256(path)})
                with open(os.path.join(d121, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"cfg": {"phases": [{"mm": 3, "steps": 1}, {"mm": 1.5, "steps": 1}],
                                       "prior": {"version": 116, "dim": 16}},
                               "structures": ["a", "b"], "folds": folds, "trained": "selftest"}, f)

                def write_tsv(path, shipped, mean):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("all\tsegmenter\theadline_dice\t%g\t%g\t%g\n" % mean)
                        f.write("all\tsegmenter\theadline_dice_minus_reference\t%g\t%g\t%g\n" % shipped)

                write_tsv(os.path.join(tmp, "scorecard_test.tsv"), (0.01, 0.002, 0.02), (0.8, 0.79, 0.81))
                for split in ("test", "dev"):
                    with open(os.path.join(tmp, "pred_baselines_%s.csv" % split), "w", encoding="utf-8") as f:
                        f.write("method\n")

                class FakeProc:
                    returncode, stderr = 0, ""

                class FakeSubprocess:
                    PIPE, CalledProcessError = real.PIPE, real.CalledProcessError

                    def __init__(self):
                        self.calls, self.crash = [], None

                    def run(self, cmd, **kw):
                        args = cmd[2:]                      # [python, -u, script, arguments...]
                        self.calls.append(args)
                        if self.crash is not None and self.crash in args:
                            raise real.CalledProcessError(1, cmd)
                        if args[0] == "score.py":
                            write_tsv(args[args.index("--tsv") + 1], (0.02, 0.01, 0.03), (0.85, 0.84, 0.86))
                        return FakeProc()

                fake_sp = g["subprocess"] = FakeSubprocess()
                state = {"version": 1, "reference": "ref", "test_opened": "t", "finished": "t2", "looks": []}
                entry = start_look(state, 121)
                check("(i) the fine version's look is number 2, latched under its own name",
                      entry["look"] == 2 and len(fake.written) == 1 and "second look v121" in fake.written[0][0][1])
                fake_sp.crash = "coarse"
                try:
                    second_look_steps(state, entry)
                    crashed = False
                except real.CalledProcessError:
                    crashed = True
                check("(i) the coarse step crashed", crashed)
                check("(i) the steps before it are done",
                      entry["done"] == ["copy baselines rows", "version dev rows", "test 1.5 mm cache"])
                fake_sp.crash = None
                second_look_steps(state, entry)
                check("(i) the look finished, its steps in order", entry["done"] == order)
                check("(i) the 1.5 mm cache was built once, inside the look",
                      [c for c in fake_sp.calls if c[0] == "prepare.py"]
                      == [["prepare.py", "hi15", "--split", "test", "--look", "2"]])
                coarse = [c for c in fake_sp.calls if c[:2] == ["segmenter.py", "coarse"]]
                check("(i) the prior's coarse maps: all five nets, inside the look",
                      len(coarse) == 2 and all(c == ["segmenter.py", "coarse", "--version", "116", "--split", "test",
                                                     "--look", "2"] for c in coarse))
                check("(i) the fine read names the look and no partner",
                      [c for c in fake_sp.calls if c[:2] == ["segmenter.py", "predict"]]
                      == [["segmenter.py", "predict", "test", "--version", "121", "--out-dir", look_folder(entry),
                           "--look", "2"]])
                check("(i) the steps wrote no second latch, and the first look stands",
                      len(fake.written) == 1 and state["test_opened"] == "t" and state["finished"] == "t2")
                check("(i) the look's scoring is recorded", entry["cells_test"] == 2 and entry["cells_dev"] == 2)
                with open(os.path.join(look_folder(entry), "second_look.md"), encoding="utf-8") as f:
                    text = f.read()
                check("(i) second_look.md says it was read at 1.5 mm, with the prior",
                      "at **1.5 mm**" in text and "version 116's 3 mm coarse map" in text)
                # a standing prior_from (`segmenter.py prior`) replaces the cfg prior, so the coarse-maps step names it
                with open(os.path.join(d121, "meta.json"), encoding="utf-8") as f:
                    meta121 = json.load(f)
                meta121["reading"] = {"prior_from": 124}
                with open(os.path.join(d121, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta121, f)
                os.makedirs(segmenter.model_dir(124), exist_ok=True)
                with open(os.path.join(segmenter.model_dir(124), "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"version": 124, "cfg": {"phases": [{"mm": 3, "steps": 1}]}, "structures": ["a", "b"]}, f)
                check("(i) a standing prior_from is the prior the coarse-maps step names", fine_of(121)[1] == 124)
            finally:
                segmenter.prepare.OUT, g["HERE"], g["subprocess"], paths.WORK = keep_prep, keep_here, keep_sub, keep_work
        if len(failures) == before:
            print("fine look ok")
    except Exception as e:  # noqa: BLE001
        failures.append("exception %r" % e)
    finally:
        g.update(keep)
    if failures:
        raise SystemExit("selftest: FAILED: " + "; ".join(failures))
    print("selftest: all passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--second-look", action="store_true", dest="second_look")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--with-version", dest="with_version", type=int, default=None,
                    help="with --second-look: the pair's other version (its nets are read with --version's)")
    ap.add_argument("--tag", default=None, help="with --second-look --with-version: the pair's development rows' tag")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="with --second-look: the development gate passes only when its low end is above this")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.with_version is not None and not a.second_look:
        raise SystemExit("--with-version belongs to --second-look.")
    state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
    opened_before = [r for r in ledger.records("final_eval") if r.get("name") == LEDGER_NAME]

    if a.second_look or (a.finish and any(not l.get("finished") for l in looks_of(state))):
        if a.finish:
            open_looks = [l for l in looks_of(state) if not l.get("finished")]
            if not open_looks:
                raise SystemExit("Nothing to finish: no second look is open.")
            return second_look(state, open_looks[-1])
        return second_look(state, start_look(state, a.version, a.with_version, a.tag, a.noise))

    if a.version is None:
        a.version = 1
    if a.finish:
        if not state.get("test_opened") or state.get("finished"):
            raise SystemExit("Nothing to finish: the test split is not open, or the evaluation finished.")
    else:
        if state.get("test_opened"):
            raise SystemExit("The CT test split was already opened on %s; it is opened once. See %s. "
                             "If a step crashed, --finish carries on without opening anything again."
                             % (state["test_opened"], STATE))
        if opened_before:
            raise SystemExit("The run ledger says the CT test split was opened on %s. It is opened once, "
                             "and a missing work folder does not make it unseen." % opened_before[-1]["at"])
        meta = ready(a.version)
        reference, dice = choose_reference()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True,
                               cwd=HERE).stdout.strip()
        record(state, version=a.version, commit=commit, uncommitted=dirty.splitlines(),
               reference=reference, dev_dice=dice, trained=meta["trained"],
               weights={str(f["fold"]): f["sha256"] for f in meta["folds"]})
        # The latch goes into the committed ledger before a single test scan is read.
        ledger.write("final_eval", LEDGER_NAME, 0.0, {"version": (a.version, "count")},
                     note="reference %s" % reference)
        record(state, test_opened=time.strftime("%Y-%m-%d %H:%M:%S"))

    reference, version = state["reference"], str(state["version"])
    t0 = time.time()
    with ledger.run("final_eval", "ct final evaluation") as led:
        run(state, "baselines on test", "baselines.py", "predict", "test")
        run(state, "test scans into the cache", "prepare.py", "test")
        run(state, "segmenter on test", "segmenter.py", "predict", "test", "--version", version)
        run(state, "score test", "score.py", "test", "--reference", reference,
            "--out", os.path.join(OUT, "card_test.md"), "--tsv", os.path.join(OUT, "scorecard_test.tsv"))
        run(state, "score dev", "score.py", "dev", "--reference", reference,
            "--out", os.path.join(OUT, "card_dev.md"), "--tsv", os.path.join(OUT, "scorecard_dev.tsv"))
        for split in ("test", "dev"):
            run(state, "recompute %s" % split, "recompute.py", split, "--reference", reference,
                "--compare", os.path.join(OUT, "scorecard_%s.tsv" % split),
                log=os.path.join(HERE, "recompute_%s.log" % split))
        test_rows = pd.read_csv(os.path.join(OUT, "pred_%s_test.csv" % CANDIDATE), usecols=["scan"], dtype=str)
        record(state, cells_test=len(cells(os.path.join(OUT, "scorecard_test.tsv"))),
               cells_dev=len(cells(os.path.join(OUT, "scorecard_dev.tsv"))),
               test_scans=int(test_rows["scan"].nunique()))
        result = publish(state)
        record(state, ship_result=result, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        led.number("difference", result["difference"]).number("ci_low", result["ci_low"])
        led.number("ci_high", result["ci_high"]).number("ships", 1 if result["ships"] else 0, "count")
        led.number("cells_test", state["cells_test"], "count").number("cells_dev", state["cells_dev"], "count")
        for step, took in state.get("step_seconds", {}).items():
            led.number("seconds.%s" % step.replace(" ", "_"), took, "s")
        led.note_is("reference %s, %s" % (reference, "ships" if result["ships"] else "does not ship"))
    print(json.dumps({"reference": reference, **result, "seconds": round(time.time() - t0, 1)}, indent=1),
          flush=True)


if __name__ == "__main__":
    main()
