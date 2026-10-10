"""Write modalities/xray/models/fracture-detector.md, the model card for the X-ray fracture detector,
from what is on disk and nothing typed into this script.

Every number in the card comes from one of four places, and the card says which:
- the trained detector's record, paths.MODELS/xray-fracture-detector-v<N>/meta.json;
- the locked split, modalities/xray/eval/split.json (units, images, folds, the rule in words);
- the two datasets' own index files, read through datasets.index (images, fractures, casts,
  follow-ups, uncertain labels), so the card's counts are the data's counts, not the README's;
- the scored rows of modalities/xray/eval/SCORECARD.md, in the table formats METRICS_SPEC.md fixes.

Why it refuses rather than guesses: a model card with a blank where a number should be reads as if
the number were fine. So if any of those files is missing the script stops and names the file, and
writes nothing. The one gap it fills on purpose is the held-out test before final_eval.py has run:
then the card says, in so many words, that the held-out test has not been opened.

The test split is never read here. The development counts come through split.frame(dataset, "dev"),
behind datasets.guard_test, and the held-out numbers only from SCORECARD.md, which final_eval.py
writes once.

Nothing here is random, so there is no seed to set: the same files give the same card, byte for byte.

Usage: python model_card.py [--version 1] [--out <path>]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402

import datasets  # noqa: E402
import split  # noqa: E402

MODELS = paths.MODELS
SCORECARD = os.path.join(HERE, "SCORECARD.md")
SPLIT = split.PATH
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "models", "fracture-detector.md")
METHOD = "cnn"                     # the detector's name in predictions.csv and SCORECARD.md
PRIMARY = "grazpedwri"             # the ship decision rests on the set split by patient
METRIC_ORDER = ["auroc", "sensitivity", "specificity", "balanced_accuracy", "kappa"]
BASELINES = ["prevalence", "meta", "features", "reader"]
NOT_OPENED = "The held-out test has not been opened."

H_METRICS = "| dataset | split | stratum | metric | method | n_images | n_patients | value | ci_low | ci_high | n_boot |"
H_PAIRED = ("| dataset | split | stratum | metric | method | reference | n_images | n_patients | difference "
            "| ci_low | ci_high | n_boot |")
H_ACCOUNT = "| dataset | split | quantity | value |"


def log(msg):
    print(msg, flush=True)


def refuse(what, path):
    raise SystemExit("model_card.py: refusing to write a card with holes: %s is missing (%s)" % (what, path))


def need(path, what):
    if not os.path.exists(path):
        refuse(what, path)
    return path


def tables(text):
    """{header: [rows as lists of cells]} for the three checked SCORECARD.md tables.

    Read by header, exactly as recompute.py does, so prose and other tables are ignored and a
    reworded heading cannot hide a row."""
    out = {H_METRICS: [], H_PAIRED: [], H_ACCOUNT: []}
    lines = [ln.rstrip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if head in out:
            width = head.count("|") - 1
            i += 2                                   # the header and its |---| line
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) == width:
                    out[head].append(cells)
                i += 1
            continue
        i += 1
    return out


def metric_rows(t, dataset, split_name, method):
    """Table 2 rows of one (dataset, split, method), as dicts, `all` first and metrics in spec order."""
    keys = H_METRICS.strip("| ").split(" | ")
    rows = [dict(zip(keys, r)) for r in t[H_METRICS]]
    rows = [r for r in rows if r["dataset"] == dataset and r["split"] == split_name and r["method"] == method]
    order = list(datasets.SETS[dataset]["strata"])
    rows.sort(key=lambda r: ((["all"] + order).index(r["stratum"]) if r["stratum"] in ["all"] + order else 99,
                             METRIC_ORDER.index(r["metric"]) if r["metric"] in METRIC_ORDER else 99))
    return rows


def find(rows, stratum, metric):
    return next((r for r in rows if r["stratum"] == stratum and r["metric"] == metric), None)


def ci(r):
    return "%s (95%% CI %s to %s)" % (r["value"], r["ci_low"], r["ci_high"])


def metric_table(rows):
    L = ["| stratum | metric | images | units | value | 95% CI |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s to %s |" % (r["stratum"], r["metric"], r["n_images"],
                                                          r["n_patients"], r["value"], r["ci_low"], r["ci_high"]))
    return L


def facts(dataset):
    """Counts from the dataset's own index, whole set and development split."""
    s = datasets.SETS[dataset]
    need(os.path.join(s["root"], s["index"][0]), "the %s index file" % dataset)
    df = datasets.index(dataset)
    datasets.guard_test("dev")
    dev = split.frame(dataset, "dev")
    f = {"images": len(df), "units": int(df.patient.nunique()), "fractures": int(df.truth.sum()),
         "dev_images": len(dev), "dev_units": int(dev.patient.nunique()), "dev_fractures": int(dev.truth.sum())}
    if dataset == "grazpedwri":
        for col in ("cast", "initial_exam", "diagnosis_uncertain"):
            f[col] = int((df[col].str.strip() == "1").sum())
    else:
        for col in ("hand", "leg", "hip", "shoulder"):
            f[col] = int((df[col].str.strip() == "1").sum())
    return f


def shortcut_lines(rows, f):
    """What the pre-declared subsets showed, from the development rows."""
    a = find(rows, "all", "auroc")
    L = []
    for stratum, what in (("initial_exam", "initial examinations only, so no follow-up of a known fracture"),
                          ("no_cast", "images with no cast"),
                          ("certain", "images whose diagnosis is not flagged uncertain")):
        r = find(rows, stratum, "auroc")
        if r is None or a is None:
            L.append("- %s: no development AUROC row for this subset in SCORECARD.md." % what.capitalize())
            continue
        try:
            gap = float(r["value"]) - float(a["value"])
            delta = "%+.3f against all images" % gap
        except ValueError:
            delta = "not comparable (n/a)"
        L.append("- %s (%s images): AUROC %s, %s." % (what.capitalize(), r["n_images"], ci(r), delta))
    return L


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Write the X-ray fracture detector's model card from files on disk.")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    # stage 1: the files the card is made of, all of them, before anything is written
    t = time.time()
    model_dir = os.path.join(MODELS, "xray-fracture-detector-v%d" % a.version)
    meta = json.load(open(need(os.path.join(model_dir, "meta.json"), "the detector's meta.json"), encoding="utf-8"))
    sp = split.load(need(SPLIT, "the locked split split.json"))
    card_text = open(need(SCORECARD, "the scorecard SCORECARD.md"), encoding="utf-8").read()
    tb = tables(card_text)
    log("read meta.json, split.json, SCORECARD.md  %.1f s" % (time.time() - t))

    # stage 2: the datasets' own counts
    t = time.time()
    F = {d: facts(d) for d in datasets.NAMES}
    log("counted both datasets' index files  %.1f s" % (time.time() - t))

    # stage 3: the scored rows
    t = time.time()
    dev = {d: metric_rows(tb, d, "dev", METHOD) for d in datasets.NAMES}
    test = {d: metric_rows(tb, d, "test", METHOD) for d in datasets.NAMES}
    missing_dev = [d for d in datasets.NAMES if not dev[d]]
    if missing_dev:
        refuse("development rows for method %r on %s" % (METHOD, ", ".join(missing_dev)), SCORECARD)
    opened = any(test[d] for d in datasets.NAMES)
    pk = H_PAIRED.strip("| ").split(" | ")
    paired = [dict(zip(pk, r)) for r in tb[H_PAIRED] if r[4] == METHOD]
    log("parsed SCORECARD.md: %d metric rows, %d paired rows, held-out %s  %.1f s"
        % (len(tb[H_METRICS]), len(paired), "opened" if opened else "not opened", time.time() - t))

    # stage 4: the card
    t = time.time()
    g, fa = F["grazpedwri"], F["fracatlas"]
    sg, sf = sp["datasets"]["grazpedwri"], sp["datasets"]["fracatlas"]
    trained_on = meta.get("datasets", datasets.NAMES)
    if isinstance(trained_on, str):
        trained_on = [trained_on]
    thr = meta.get("threshold")
    L = ["# Model card: rAidiology X-ray fracture detector v%d" % a.version, "",
         "Generated by `modalities/xray/eval/model_card.py` from the model's `meta.json`, "
         "`modalities/xray/eval/split.json`, the two datasets' own index files and "
         "`modalities/xray/eval/SCORECARD.md`. No number in it was typed by hand.", "",
         "**Not a medical device.** It is a research tool, not a diagnosis (`LICENSING.md`).", "",
         "## What it answers", "",
         "One question per radiograph, fixed before any model was trained: **does this radiograph show a "
         "fracture?** The output is a score from 0 to 1 per image, where higher means a fracture is more likely, "
         "and a yes or no from that score at a threshold fitted on development folds only"
         + (" (%s)" % thr if thr is not None else "") + ". It does not say where the fracture is, "
         "what kind it is, or how bad.", "",
         "## What it was trained on", "",
         "Development images only, from %s. The held-out test images were never used for training, the "
         "threshold, or any choice." % " and ".join("`%s`" % d for d in trained_on), ""]
    L += ["| | GRAZPEDWRI-DX | FracAtlas |", "|---|---|---|",
          "| images in the dataset | %d | %d |" % (g["images"], fa["images"]),
          "| with a fracture | %d (%.0f%%) | %d (%.0f%%) |" % (g["fractures"], 100.0 * g["fractures"] / g["images"],
                                                          fa["fractures"], 100.0 * fa["fractures"] / fa["images"]),
          "| resampling unit | patient (%d) | image (%d) |" % (g["units"], fa["units"]),
          "| development images (units) | %d (%d) | %d (%d) |" % (g["dev_images"], g["dev_units"],
                                                              fa["dev_images"], fa["dev_units"]),
          "| development images with a fracture | %d | %d |" % (g["dev_fractures"], fa["dev_fractures"]),
          "| held-out images (units), never trained on | %d (%d) | %d (%d) |"
          % (sg["n"]["test_images"], sg["n"]["test_units"], sf["n"]["test_images"], sf["n"]["test_units"]),
          ""]
    L += ["Citations, both under the **CC BY 4.0** licence:", "",
          "- GRAZPEDWRI-DX: %s." % datasets.SETS["grazpedwri"]["cite"],
          "- FracAtlas: %s." % datasets.SETS["fracatlas"]["cite"], "",
          "Because both training sets are CC BY 4.0, this model carries **no 'permission pending'** the way the "
          "spine disc grader does: nothing in it depends on weights or labels whose commercial terms are "
          "unconfirmed. The attribution above is what the licence asks for.", "",
          "**Trained from scratch, with no pretrained backbone.** No ImageNet weights, no model zoo: ImageNet's "
          "terms are non-commercial, and weights learned from it would carry that restriction into a paid product.", ""]
    L += ["## Model", ""]
    for k in ("inputs", "network", "trained"):
        if k in meta:
            L.append("- %s: %s." % (k.capitalize(), meta[k]))
    if "cfg" in meta:
        L.append("- Training settings: `%s`." % json.dumps(meta["cfg"], sort_keys=True))
    L += ["- %d fold models (%d-fold cross-validation on development units); each development score below "
          "comes from the fold model that never saw that image." % (sp["folds"], sp["folds"]), ""]
    L += ["## The split", "",
          "Seed %d, %d%% of the units held out as test, the rest in %d folds, written once and committed "
          "(`split.json`). The rule, as recorded there:" % (sp["seed"], round(100 * sp["test_share"]), sp["folds"]), ""]
    L += ["> " + ln if ln else ">" for ln in sp["rule"].splitlines()]
    L += ["",
          "- GRAZPEDWRI-DX is split by **%s**, so a child's images never fall on both sides." % sg["unit"],
          "- FracAtlas is split by **%s**: FracAtlas publishes no patient identifier, so two radiographs of the "
          "same person could fall on both sides. That is why the ship decision rests on GRAZPEDWRI-DX and "
          "FracAtlas is an external check." % sf["unit"], ""]
    L += ["## Development numbers", "",
          "Out of fold on development images (95% CI, 2,000 bootstrap resamples of the unit: the patient for "
          "GRAZPEDWRI-DX, the image for FracAtlas).", ""]
    for d in datasets.NAMES:
        L += ["### %s" % d, ""] + metric_table(dev[d]) + [""]
    base = []
    for m in BASELINES:
        r = find(metric_rows(tb, PRIMARY, "dev", m), "all", "auroc")
        if r is not None:
            base.append("| %s | %s | %s to %s |" % (m, r["value"], r["ci_low"], r["ci_high"]))
    if base:
        L += ["Baselines on all GRAZPEDWRI-DX development images, for scale:", "",
              "| method | AUROC | 95% CI |", "|---|---|---|"] + base + [""]
    L += ["## Held-out numbers", ""]
    if not opened:
        L += [NOT_OPENED + " These numbers are written here once `final_eval.py` has run, "
              "and it runs once.", ""]
    else:
        L += ["The shipped model, scored once in the final evaluation:", ""]
        for d in datasets.NAMES:
            L += ["### %s" % d, ""] + (metric_table(test[d]) if test[d] else ["No held-out rows for this dataset."]) + [""]
        ship = next((p for p in paired if p["dataset"] == PRIMARY and p["stratum"] == "all"), None)
        if ship:
            try:
                verdict = "**ships**" if float(ship["ci_low"]) > 0 else "**does not ship**"
            except ValueError:
                verdict = "**does not ship** (the interval is undefined)"
            L += ["Ship rule (AUROC minus the reference `%s`'s on the same images, paired by patient, all "
                  "GRAZPEDWRI-DX test images): difference %s, 95%% CI %s to %s: %s."
                  % (ship["reference"], ship["difference"], ship["ci_low"], ship["ci_high"], verdict), ""]
    L += ["## Shortcuts it could learn instead of the fracture", "",
          "- **A cast.** %d GRAZPEDWRI-DX images show a plaster cast, which almost always means a fracture "
          "somebody already diagnosed." % g["cast"],
          "- **A follow-up.** Only %d of the %d images are initial examinations; the rest are check-ups of a "
          "known fracture." % (g["initial_exam"], g["images"]),
          "- **An unsure label.** %d images are flagged 'diagnosis uncertain'; none were thrown away."
          % g["diagnosis_uncertain"], "",
          "What the pre-declared subsets showed (GRAZPEDWRI-DX, development, out of fold):", ""]
    L += shortcut_lines(dev[PRIMARY], g)
    if opened and test[PRIMARY]:
        L += ["", "And on the held-out test:", ""] + shortcut_lines(test[PRIMARY], g)
    L += ["", "A subset AUROC well below the all-images AUROC would mean part of the score came from the "
          "shortcut, not the break.", ""]
    L += ["## Who it is NOT for", "",
          "- It is paediatric-wrist-heavy: %d of the %d training-pool images are GRAZPEDWRI-DX children's wrists. "
          "An adult hip is out of distribution (FracAtlas has only %d hip images in total), and so is any body "
          "part neither dataset covers, such as the skull, spine, ribs or pelvis."
          % (g["images"], g["images"] + fa["images"], fa["hip"]),
          "- Not for anyone deciding on treatment: a radiologist's report decides, not this.",
          "- Not for images other than plain radiographs (no CT, MRI or ultrasound), or for photographs of a screen.",
          "- Not a medical device.", ""]
    if isinstance(meta.get("weights"), dict):
        L += ["## Files", "", "Weights in `Models/xray-fracture-detector-v%d/` (outside the repository):" % a.version, ""]
        for f, h in sorted(meta["weights"].items()):
            L.append("- `%s` SHA-256 `%s`" % (f, h))
        L.append("")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    log("wrote the card  %.1f s" % (time.time() - t))
    log("%s  %d lines  (total %.1f s)" % (os.path.normpath(out), len(L), time.time() - t0))


if __name__ == "__main__":
    main()
