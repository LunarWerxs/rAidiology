"""Baseline `reader`: Claude Opus 5.5 looking at one radiograph and saying yes or no to a fracture,
run as sub-agents because the owner has no Anthropic API key. There is no API-calling path here and
the anthropic SDK is never imported.

How it works, in three steps:
  1. `dev` or `test` draws a sample of images and turns each into a self-contained bundle folder
     (preds.BUNDLES/<name>/). A sub-agent is handed one folder and may read nothing else.
  2. The orchestrator hands the folders out (`bundles` lists the ones still waiting) and gathers the
     answers into one JSON file.
  3. `ingest` stores the answers where an API call's answer would have gone, writes
     pred_reader_<dataset>_<split>.csv in the exact METRICS_SPEC.md columns, and runs preds.collect().

The sample, and why it is drawn this way
  dev 120 images, test 200, split between the two datasets in proportion to how many images each has
  in that split (so about 100 GRAZPEDWRI-DX and 20 FracAtlas for dev). Units are chosen first and then
  one image per chosen unit, so no patient contributes twice: a patient's images look alike and would
  quietly count the same case twice. The label is never used to choose anything, because the reader
  must meet the real prevalence; a balanced sample would flatter its sensitivity and say nothing
  about how it behaves in a casualty department. One numpy generator, seeded from datasets.SEED,
  makes every draw, so re-running reproduces the same sample.

What is inside a bundle, and why nothing else
  image.png (written from the cached array, so the reader sees exactly the pixels the detector
  sees), INSTRUCTIONS.md (READER_BRIEF.md, then the question and the exact answer format) and
  answer.schema.json. The folder name is 'case-' plus eight hex characters of a hash, which carries
  no hint of the dataset, the patient or the answer. No dataset name, patient id, age, cast flag or
  source-bearing file name goes in. The private key, which maps a bundle name back to its image, is
  work/eval/xray/reader/manifest.json and never goes inside a bundle.

The answer: {"fracture": "yes" or "no", "confidence": an integer 0 to 100, "where": at most 15 words}.
  fracture_score = 50 + confidence / 2 for yes and 50 - confidence / 2 for no, so one number orders
  every answer; fracture_pred = 1 for yes, 0 for no. A bundle with no answer, or one that cannot be
  parsed, becomes a row with an empty prediction and score, which METRICS_SPEC.md counts as a miss
  (the worst answer), not a dropped row. Only the sampled images get rows: an image the reader was
  never shown has no row, so the group is scored on what the reader saw. A sampled image that is
  unreadable gets no bundle and a row with in_scope 0 and the reason preds.UNREADABLE.

The test split is opened only through datasets.guard_test(split).
Usage:
  python reader.py dev  [--sample N] --backend agents [--rebuild]
  python reader.py test [--sample N] --backend agents [--rebuild]      (final_eval.py only)
  python reader.py bundles [dev|test]       the pending bundle folders, one per line
  python reader.py ingest <answers.json>    {bundle: answer} or [{"bundle": ..., ...}]; re-runnable"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402,F401
import datasets  # noqa: E402
import preds  # noqa: E402
import prepare  # noqa: E402
import split as splitmod  # noqa: E402

READER_DIR = os.path.join(preds.WORK, "reader")
MANIFEST = os.path.join(READER_DIR, "manifest.json")
ANSWERS = os.path.join(READER_DIR, "answers.json")
UNREADABLE = os.path.join(READER_DIR, "unreadable.json")
BRIEF = os.path.join(os.path.dirname(HERE), "READER_BRIEF.md")
SAMPLE_SIZE = {"dev": 120, "test": 200}
MODEL = "claude-opus-5-5"

QUESTION = """## Your task

Look at `image.png` in this folder. It is the only thing you may open: do not read any other file
or folder, and do not look for anything outside this folder.

**Question: does this radiograph show a fracture?**

Answer yes or no about THIS image. An uncertain yes is still a yes: give it a low confidence.
rAidiology is not a medical device; this is a reading exercise, not advice.

## Answer format

Reply with one JSON object and nothing else:

{"fracture": "yes" or "no", "confidence": an integer from 0 to 100, "where": "at most 15 words"}

- `fracture`: exactly `yes` or `no`.
- `confidence`: how sure you are of that answer, 0 (a guess) to 100 (certain).
- `where`: where you see the fracture, or what you checked if you answered no. At most 15 words.
"""

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["fracture", "confidence", "where"],
    "properties": {
        "fracture": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "where": {"type": "string", "maxLength": 120},
    },
}


def load_json(p, default):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, p)


def bundle_name(dataset, split, image):
    """Neutral token: 'case-' and eight hex characters of a hash. Nothing in it names the answer."""
    h = hashlib.sha256(repr((dataset, split, image, str(datasets.SEED))).encode("utf-8")).hexdigest()
    return "case-" + h[:8]


def allocate(counts, n):
    """Split n over the datasets in proportion to counts (largest remainder, ties by name)."""
    total = sum(counts.values())
    raw = {d: n * c / total for d, c in counts.items()}
    out = {d: min(int(raw[d]), counts[d]) for d in counts}
    for d in sorted(counts, key=lambda d: (-(raw[d] - int(raw[d])), d)):
        if sum(out.values()) >= n:
            break
        if out[d] < counts[d]:
            out[d] += 1
    return out


def sample(split, n):
    """[(dataset, row)] : units first (never balanced by label), then one image per chosen unit."""
    frames = {d: splitmod.frame(d, split) for d in datasets.NAMES}
    counts = {d: len(f) for d, f in frames.items() if len(f)}
    if not counts:
        raise SystemExit("no rows in split %r" % split)
    want = allocate(counts, n)
    rng = np.random.default_rng(datasets.SEED)
    chosen = []
    for d in sorted(counts):
        f = frames[d]
        units = sorted({str(u).strip() for u in f.patient})
        k = min(want[d], len(units))
        picked = sorted(int(i) for i in rng.choice(len(units), size=k, replace=False))
        by_unit = {}
        for _, r in f.sort_values("image").iterrows():
            by_unit.setdefault(str(r.patient).strip(), []).append(r)
        for i in picked:
            rows = by_unit[units[i]]
            chosen.append((d, rows[int(rng.integers(len(rows)))]))
    return chosen


def open_cache(dataset):
    """The cached 8-bit array for that dataset, building it first if it is not there."""
    try:
        return prepare.load(dataset, prepare.SIZE)
    except (FileNotFoundError, OSError):
        prepare.build(dataset, prepare.SIZE, False)
        return prepare.load(dataset, prepare.SIZE)


def instructions():
    with open(BRIEF, encoding="utf-8") as f:
        brief = f.read().rstrip()
    return brief + "\n\n" + QUESTION


def make_bundles(split, n, rebuild):
    """Draw the sample, write one folder per readable image, and record the private key."""
    t0 = time.time()
    chosen = sample(split, n)
    print("sample %s: %d images (%s) in %.1fs" % (
        split, len(chosen),
        ", ".join("%s %d" % (d, sum(1 for x, _ in chosen if x == d)) for d in datasets.NAMES),
        time.time() - t0), flush=True)
    t0 = time.time()
    caches, bad = {}, {}
    for d in datasets.NAMES:
        if any(x == d for x, _ in chosen):
            arr, index, _ = open_cache(d)
            caches[d] = (arr, index)
            bad[d] = set(prepare.unreadable(d, prepare.SIZE))
    manifest = load_json(MANIFEST, {})
    old = {k for k, v in manifest.items() if v["split"] == split}
    manifest = {k: v for k, v in manifest.items() if v["split"] != split}
    unread = load_json(UNREADABLE, {})
    unread[split] = []
    text, names = instructions(), set()
    for d, r in chosen:
        img = str(r.image)
        fold = "" if r.fold == "" else int(r.fold)
        facts = {"dataset": d, "split": split, "image": img, "patient": str(r.patient).strip(), "fold": fold}
        arr, index = caches[d]
        if img in bad[d] or img not in index:
            unread[split].append(facts)
            continue
        name = bundle_name(d, split, img)
        if name in names or name in manifest:
            raise SystemExit("bundle name collision on %s" % name)
        names.add(name)
        manifest[name] = facts
        folder = os.path.join(preds.BUNDLES, name)
        os.makedirs(folder, exist_ok=True)
        png = os.path.join(folder, "image.png")
        if rebuild or not os.path.exists(png):
            Image.fromarray(np.asarray(arr[index[img]], dtype=np.uint8)).save(png)
        with open(os.path.join(folder, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(folder, "answer.schema.json"), "w", encoding="utf-8") as f:
            json.dump(SCHEMA, f, indent=1)
    for stale in old - names:
        shutil.rmtree(os.path.join(preds.BUNDLES, stale), ignore_errors=True)
    save_json(MANIFEST, manifest)
    save_json(UNREADABLE, unread)
    print("%d bundles in %s (%d unreadable, no bundle) in %.1fs"
          % (len(names), preds.BUNDLES, len(unread[split]), time.time() - t0), flush=True)


def pending(split=None):
    """Bundle folders that exist, are in the manifest and have no stored answer yet."""
    manifest = load_json(MANIFEST, {})
    answers = load_json(ANSWERS, {})
    out = []
    for name in sorted(manifest):
        if split and manifest[name]["split"] != split:
            continue
        folder = os.path.join(preds.BUNDLES, name)
        if name not in answers and os.path.isdir(folder):
            out.append(folder)
    return out


def parse(ans):
    """(fracture_pred, fracture_score) from an answer, or None when it cannot be read."""
    if isinstance(ans, str):
        try:
            ans = json.loads(ans)
        except ValueError:
            m = re.search(r"\{.*\}", ans, re.S)
            try:
                ans = json.loads(m.group(0)) if m else None
            except ValueError:
                ans = None
    if not isinstance(ans, dict):
        return None
    f = ans.get("fracture")
    if isinstance(f, bool):
        f = "yes" if f else "no"
    f = str(f).strip().lower()
    if f not in ("yes", "no"):
        return None
    c = ans.get("confidence")
    if isinstance(c, bool) or c is None:
        return None
    try:
        c = float(c)
    except (TypeError, ValueError):
        return None
    if not (0 <= c <= 100) or c != c:
        return None
    c = int(round(c))
    return (1, 50 + c / 2) if f == "yes" else (0, 50 - c / 2)


def read_answers(p):
    """{bundle: answer} from either file shape."""
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, list):
        raise SystemExit("%s: expected an object of {bundle: answer} or a list of {\"bundle\": ...}" % p)
    out = {}
    for item in raw:
        if not isinstance(item, dict) or "bundle" not in item:
            raise SystemExit("%s: a list entry has no \"bundle\" key" % p)
        out[item["bundle"]] = item["answer"] if "answer" in item else {k: v for k, v in item.items() if k != "bundle"}
    return out


def ingest(answers_p):
    t0 = time.time()
    manifest = load_json(MANIFEST, {})
    new = read_answers(answers_p)
    unknown = sorted(k for k in new if k not in manifest)
    if unknown:
        raise SystemExit("%d answer key(s) are not in the manifest, e.g. %s" % (len(unknown), unknown[:3]))
    for s in sorted({manifest[k]["split"] for k in new}):
        datasets.guard_test(s)
    answers = load_json(ANSWERS, {})
    answers.update(new)
    save_json(ANSWERS, answers)
    groups = sorted({(manifest[k]["dataset"], manifest[k]["split"]) for k in answers if k in manifest})
    unread = load_json(UNREADABLE, {})
    for d, s in groups:
        datasets.guard_test(s)
        rows, got, miss = [], 0, 0
        for name in sorted((k for k, v in manifest.items() if v["dataset"] == d and v["split"] == s),
                           key=lambda k: manifest[k]["image"]):
            m = manifest[name]
            row = {"fold": m["fold"], "patient": m["patient"], "image": m["image"], "in_scope": 1,
                   "drop_reason": "", "fracture_pred": "", "fracture_score": ""}
            p = parse(answers[name]) if name in answers else None
            if p is None:
                miss += 1
            else:
                row["fracture_pred"], row["fracture_score"] = p
                got += 1
            rows.append(row)
        for m in unread.get(s, []):
            if m["dataset"] == d:
                rows.append({"fold": m["fold"], "patient": m["patient"], "image": m["image"], "in_scope": 0,
                             "drop_reason": preds.UNREADABLE, "fracture_pred": "", "fracture_score": ""})
        print("reader %s %s: %d answered, %d without a usable answer (counted as misses)" % (d, s, got, miss),
              flush=True)
        preds.write("reader", d, s, rows)
    preds.collect()
    print("ingest done in %.1fs" % (time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip().split("\n")[0])
    ap.add_argument("mode", choices=["dev", "test", "bundles", "ingest"])
    ap.add_argument("target", nargs="?", help="ingest: the answers JSON; bundles: dev or test")
    ap.add_argument("--sample", type=int, help="images to sample (default 120 dev, 200 test)")
    ap.add_argument("--backend", choices=["agents"], help="agents: Claude Opus sub-agents, no API key")
    ap.add_argument("--rebuild", action="store_true", help="rewrite every bundle image")
    a = ap.parse_args()
    if a.mode == "ingest":
        if not a.target:
            ap.error("ingest needs <answers.json>")
        ingest(a.target)
    elif a.mode == "bundles":
        if a.target not in (None, "dev", "test"):
            ap.error("bundles takes dev or test, or nothing")
        if a.target:
            datasets.guard_test(a.target)
        elif any(v["split"] == "test" for v in load_json(MANIFEST, {}).values()):
            datasets.guard_test("test")
        for p in pending(a.target):
            print(p)
    else:
        datasets.guard_test(a.mode)
        if a.backend != "agents":
            ap.error("--backend agents is required (there is no API path)")
        make_bundles(a.mode, a.sample or SAMPLE_SIZE[a.mode], a.rebuild)


if __name__ == "__main__":
    main()
