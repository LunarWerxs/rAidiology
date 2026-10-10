"""How many times each model has been trained, from the ledgers.
Reads the *.jsonl ledgers in RAIDIOLOGY_LEDGER (default <repo>/ledger) and counts the version folders each model's
globs name. Prints the table that sits between the train-counts markers in docs/VS_DOCTORS.md; --write replaces only
the text between the markers. The as-of time is the newest counted ledger line.
A model's round is its count of trainings: round N is the N-th distinct version trained, in time order. A version's
time is the ledger's first `at` for it, or, for a version the ledger never names (trained before it began, or beside
it), the time its folder was made (the earliest of ctime and mtime over its folders). Version folders are named by
the model's globs: `models:<glob>` is under the Models folder (RAIDIOLOGY_MODELS, else `Models` beside the repo), any
other glob is under <work>/eval (RAIDIOLOGY_WORK overrides work/). A folder counts when it is a directory whose name
ends in v<number> (`v101.failed1` does not). A glob ending `|<file glob>` (`corners-v*|LA-*`) counts only folders that
hold a file matching it, so a front-only folder is never a side round. A folder-only version younger than the model's newest ledger version is a
run in progress and gets no round. Ties are broken by version number.
--rounds lists the rounds; --write also rewrites docs/ROUNDS.md whole.
  python tools/train_counts.py
  python tools/train_counts.py --rounds
  python tools/train_counts.py --write [--doc docs/VS_DOCTORS.md] [--rounds-doc docs/ROUNDS.md]"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START, END = "<!-- train-counts:start -->", "<!-- train-counts:end -->"

# (ledger file, regex on name, row name, line it serves, tuple of folder globs: under work/eval, or `models:` = under
# the Models folder). A glob may end in `|<file glob>` (e.g. "spine_xray/corners-v*|LA-*"): the folder then counts only
# when it holds at least one file matching the file glob. Without `|` a glob behaves as before.
MODELS = [
    ("detector.jsonl", r"^train fold \d+ v(\d+)", "X-ray fracture detector",
     "Kids' wrists and limbs, adults' mixed bones", ("xray/detector-v*", "models:xray-fracture-detector-v*")),
    ("segmenter.jsonl", r"^train fold \d+ v(\d+)", "CT body-part segmenter", "Mean overlap, 3 mm and finest",
     ("ct/segmenter-v*", "models:ct-segmenter-v*")),
    ("reader.jsonl", r"^train v(\d+) folds", "NIH chest reader (14 findings)",
     "Fourteen findings; the start of other chest nets", ("chest/reader-v*",)),
    ("experiment.jsonl", r"^foramen v(\d+)", "Spine MRI foramen grader", "Foraminal stenosis, four grades",
     ("spine_mri/foramen/v*",)),
    ("experiment.jsonl", r"^mammo v(\d+) train", "CMMD mammography", "CMMD cancer or benign", ("mammo/mammo-v*",)),
    ("experiment.jsonl", r"^cbis v(\d+) (mass|calc) train", "CBIS-DDSM masses and calcifications",
     "CBIS-DDSM masses; CMMD's start", ("mammo/cbis/v*",)),
    ("experiment.jsonl", r"^taix train v(\d+)", "TAIX bedside chest (8 findings)", "TAIX-Ray, eight findings",
     ("chest/taix-v*",)),
    ("nodule.jsonl", r"^nodule fpr v(\d+)", "Lung nodule false-alarm ranker", "LUNA16 CPM and candidate ranking",
     ("lung/fpr-v*",)),
    ("grader.jsonl", r"^grader v(\d+)", "Spine MRI disc grader", "Exactly right, agreement, two steps off",
     ("models:disc-grader-v*",)),
    ("nodule.jsonl", r"^nodule v(\d+) \d+-fold", "Lung nodule finder", "LUNA16 CPM; one radiologist against three",
     ("lung/nodule-v*",)),
    ("nodule.jsonl", r"^nodule malignancy v(\d+)", "Lung cancer or not", "Cancer or not, LIDC pathology",
     ("lung/malignancy-v*",)),
    ("experiment.jsonl", r"^axial_stenosis v(\d+)", "Spine MRI axial stenosis", "Canal, recesses and foramina",
     ("spine_mri/axial_stenosis/v*",)),
    ("corners.jsonl", r"^train LA fold \d+ v(\d+)", "Spine X-ray vertebra corners, side view",
     "Corners; spondylolisthesis", ("spine_xray/corners-v*|LA-*",)),
    ("corners.jsonl", r"^train AP fold \d+ v(\d+)", "Spine X-ray vertebra corners, front view",
     "Corners and vertebrae found, front view", ("spine_xray/corners-v*|AP-*",)),
]


def read_ledger(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                rows.append(d)
    return rows


def compress(versions):
    """Sorted distinct numbers -> 'a to b' runs where neighbours are at most 20 apart."""
    runs = []
    for v in sorted(versions):
        if runs and v - runs[-1][1] <= 20:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return ", ".join(str(a) if a == b else "%d to %d" % (a, b) for a, b in runs)


def scan():
    """Every model's ledger versions (first time, rows matched), folder versions (folder time), and the newest ledger time."""
    ledger = os.environ.get("RAIDIOLOGY_LEDGER") or os.path.join(ROOT, "ledger")
    work = os.environ.get("RAIDIOLOGY_WORK") or os.path.join(ROOT, "work")
    evaldir = os.path.join(work, "eval")
    modelsdir = os.environ.get("RAIDIOLOGY_MODELS") or os.path.join(os.path.dirname(ROOT), "Models")
    cache = {}
    found, newest = [], ""
    for fname, pat, name, serves, disks in MODELS:
        if fname not in cache:
            cache[fname] = read_ledger(os.path.join(ledger, fname))
        rx = re.compile(pat)
        versions = {}  # version -> [first at, rows matched]
        for d in cache[fname]:
            m = rx.match(str(d.get("name", "")))
            if not m:
                continue
            v = versions.setdefault(int(m.group(1)), ["", 0])
            v[1] += 1
            at = str(d.get("at", ""))
            if at:
                newest = max(newest, at)
                v[0] = min(v[0], at) if v[0] else at
        on_disk = {}  # version -> local ISO time its earliest folder was made
        for disk in disks:
            disk, _, need = disk.partition("|")
            base, g = (modelsdir, disk[len("models:"):]) if disk.startswith("models:") else (evaldir, disk)
            for p in glob.glob(os.path.join(base, g)):
                m = re.search(r"v(\d+)$", os.path.basename(p))
                if os.path.isdir(p) and m and (not need or glob.glob(os.path.join(glob.escape(p), need))):
                    st = os.stat(p)
                    t = datetime.datetime.fromtimestamp(min(st.st_ctime, st.st_mtime)).isoformat(timespec="seconds")
                    v = int(m.group(1))
                    on_disk[v] = min(on_disk[v], t) if v in on_disk else t
        found.append((name, serves, versions, on_disk))
    return found, newest


def rounds(found=None):
    """Row name -> its rounds, every version in time order: the ledger's first time for it, or for a folder-only
    version its folder time; ties by number. A folder-only version younger than the model's newest ledger version
    is a run in progress: no round."""
    out = {}
    for name, _, versions, on_disk in (found or scan()[0]):
        latest = max((v[0] for v in versions.values()), default="")
        order = [(versions[v][0] or on_disk.get(v, ""), v, "ledger", versions[v][1]) for v in versions]
        order += [(on_disk[v], v, "folder", 0) for v in on_disk
                  if v not in versions and (not latest or on_disk[v] < latest)]
        order.sort(key=lambda o: (o[0], o[1]))
        out[name] = [{"round": i, "version": v, "first": first, "source": source, "runs": runs}
                     for i, (first, v, source, runs) in enumerate(order, 1)]
    return out


def round_of(name, version):
    for r in rounds()[name]:
        if r["version"] == version:
            return r["round"]
    raise KeyError((name, version))


def when(iso):
    return iso[:10] + " " + iso[11:16]


def round_line(r):
    if r["source"] == "folder":
        return "round %d = v%d (folder made %s)" % (r["round"], r["version"], when(r["first"]))
    return "round %d = v%d (first trained %s, %d runs)" % (r["round"], r["version"], when(r["first"]), r["runs"])


def build():
    found, newest = scan()
    allrounds = rounds(found)
    rows = []
    for name, serves, versions, on_disk in found:
        rs = allrounds[name]
        count = str(len(rs))
        folders = sum(1 for r in rs if r["source"] == "folder")
        if folders:
            count += " (%d from folders, before or beside the ledger)" % folders
        last = "%d (v%d)" % (rs[-1]["round"], rs[-1]["version"]) if rs else "none"
        runs = sum(v[1] for v in versions.values())
        rows.append((len(rs), name, count, last, runs, compress([r["version"] for r in rs]), serves))
    rows.sort(key=lambda r: (-r[0], r[1]))
    asof = when(newest) if newest else "unknown"
    out = ["As of %s (local time), from each model's ledger." % asof, "",
           "| Model | Versions trained | Latest round | Training runs | Version numbers | Line it serves |",
           "|---|---:|---:|---:|---|---|"]
    for _, name, count, last, runs, nums, serves in rows:
        out.append("| %s | %s | %s | %d | %s | %s |" % (name, count, last, runs, nums, serves))
    return "\n".join(out)


def rounds_doc():
    newest = scan()[1]
    out = ["# Training rounds", "",
           "A model's round is its count of trainings: round N is the N-th distinct version of that model trained, "
           "in time order. The version number is the stored name. A version the ledger never names is dated by its "
           "folder and marked `folder`. Reads of several rounds together, or other "
           "non-training steps, are written N.1, N.2 in the docs. Generated by `python tools/train_counts.py "
           "--write`, do not edit by hand.", "",
           "As of %s (local time), from each model's ledger." % (when(newest) if newest else "unknown")]
    for name, rs in rounds().items():
        out += ["", "## " + name, "", "| Round | Version | First trained | Source | Training runs |",
                "|---:|---:|---|---|---:|"]
        for r in rs:
            out.append("| %d | v%d | %s | %s | %d |" % (r["round"], r["version"], when(r["first"]), r["source"],
                                                       r["runs"]))
    return "\n".join(out) + "\n"


def write(doc, block):
    with open(doc, "rb") as f:
        raw = f.read().decode("utf-8")
    if raw.count(START) != 1 or raw.count(END) != 1:
        sys.exit("train_counts: %s needs exactly one %s and one %s (found %d and %d)"
                 % (doc, START, END, raw.count(START), raw.count(END)))
    head, rest = raw.split(START)
    mid, tail = rest.split(END)
    if raw.index(START) > raw.index(END):
        sys.exit("train_counts: the end marker comes before the start marker in %s" % doc)
    nl = "\r\n" if "\r\n" in raw else "\n"
    new = head + START + nl + block.replace("\n", nl) + nl + END + tail
    with open(doc, "wb") as f:
        f.write(new.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Rebuild VS_DOCTORS' training-count table from the ledgers.")
    ap.add_argument("--write", action="store_true", help="replace the text between the markers in --doc")
    ap.add_argument("--doc", default=os.path.join(ROOT, "tools", "VS_DOCTORS.md"))
    ap.add_argument("--rounds", action="store_true", help="print every model's rounds")
    ap.add_argument("--rounds-doc", default=os.path.join(ROOT, "tools", "ROUNDS.md"),
                    help="the generated rounds document that --write rewrites")
    a = ap.parse_args()
    if a.rounds:
        for name, rs in rounds().items():
            print(name)
            for r in rs:
                print(round_line(r))
        return
    block = build()
    if a.write:
        write(a.doc, block)
        with open(a.rounds_doc, "w", encoding="utf-8", newline="\n") as f:
            f.write(rounds_doc())
    else:
        print(block)


if __name__ == "__main__":
    main()
