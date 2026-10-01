"""Baseline (b), the AI panel: one reader prompt per patient through the Claude API (claude-opus-5-5),
grading every marked disc. Also re-runs mistake-log cases on a patient's own scan.

Images: each series is rendered the way reading/render.py renders a patient disc (same windows,
same PNG layout, a series_meta.json so gridcrop.py works unchanged). The reader sees the mid-sagittal
T2 with its discs marked A, B, C... (from TotalSpineSeg, or from SPIDER's masks for discs only
SPIDER located), the two T2 slices either side, the mid T2 in the fluid and marrow windows, and the
T1 slices nearest the mid T2 in physical space. The prompt is READER_BRIEF.md plus a per-disc
grading instruction; the answer is structured JSON, one entry per marked disc.

Money: every live call needs --cap-usd. Spend is logged per call to work/eval/panel/ledger.jsonl and
the runner stops before any call whose worst case (estimated input + max_tokens of output) would
take the total past the cap. --dry-run renders and builds every request and estimates its cost
without calling the API. No server-side model fallback: the scorecard measures claude-opus-5-5
itself, so a refusal is recorded as a miss.

The API key comes from ANTHROPIC_API_KEY, the repo's .env, or the Windows user environment
(git-ignored); it is never printed. Without a key, use --backend agents.
Usage:
  python panel_runner.py spider dev --dev-set tune|score | --patients 1,2,3 [--cap-usd X] [--dry-run]
  python panel_runner.py spider test --sample-test 30 --cap-usd X      (final_eval.py only)
  python panel_runner.py case <case.json> --cd <cd_folder> [--cap-usd X] [--dry-run]
  python panel_runner.py spend"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PIPE)
sys.path.insert(0, os.path.join(PIPE, "reading"))
from labels import NAMES, level_of  # noqa: E402
from prepare import WORK  # noqa: E402
from spider import ITEMS, SEED, grades  # noqa: E402
from split import load as load_split  # noqa: E402

MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 16000
PRICE = {"input": 4.0, "cache_write": 5.0, "cache_read": 0.20, "output": 20.0}  # $ per million tokens
PANEL = os.path.join(os.path.dirname(WORK), "panel")
LEDGER = os.path.join(PANEL, "ledger.jsonl")
BRIEF = os.path.join(PIPE, "reading", "READER_BRIEF.md")
SCORED = ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]

GRADING = """## Your task: grade every marked disc

This message is a different case from the example series table in the brief above: an adult
lumbar spine MRI from a back-pain clinic, sagittal images only, no history. The zoom tools are not
available here; read the attached images. Image 1 marks each disc to grade with a letter at its
front edge (A, B, ...); its level name is the localiser's guess and may be off by one.

For each marked disc answer:
- `pfirrmann` 1-5 (Pfirrmann 2001, on T2; judge the nucleus's signal against the CSF in the same
  image): 1 = homogeneous, bright white, hyperintense and isointense to CSF, clear nucleus-annulus
  boundary, normal height; 2 = inhomogeneous with or without horizontal bands, still hyperintense
  and isointense to CSF, clear boundary, normal height; 3 = inhomogeneous grey, intermediate
  signal, boundary unclear, height normal to slightly decreased; 4 = inhomogeneous grey to black,
  intermediate to hypointense, boundary lost, height normal to moderately decreased; 5 =
  inhomogeneous black, hypointense, boundary lost, collapsed disc space.
- `pfirrmann_probs`: five integers (0-100, summing to 100), your probability for grades 1 to 5.
- for each of these, `<item>` = 1 if present else 0, and `<item>_p` = your probability (0-100)
  that it is present:
  - `herniation`: disc material displaced focally beyond the disc space (protrusion, extrusion
    or sequestration).
  - `narrowing`: disc height reduced compared with the discs above it.
  - `bulging`: a broad, symmetric extension of the disc beyond the vertebral body margins
    (not a focal herniation).
  - `spondylolisthesis`: the vertebra above this disc slipped forward or backward on the one below.
  - `up_endplate`: endplate change or Schmorl node in the endplate above this disc (the lower
    endplate of the vertebra above).
  - `low_endplate`: endplate change or Schmorl node in the endplate below this disc (the upper
    endplate of the vertebra below).
  - `modic`: Modic change of any type in the bone marrow next to this disc (type 1: dark on T1,
    bright on T2; type 2: bright on T1 and T2; type 3: dark on both).
Grade what the images show, in the same way at every level. Answer every marked disc."""


def dotenv(name):
    """A value from the repo's git-ignored .env, or None."""
    p = os.path.join(PIPE, "..", ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            if k == name and v:
                return v.strip().strip('"').strip("'")
    return None


def api_key():
    k = os.environ.get("ANTHROPIC_API_KEY") or dotenv("ANTHROPIC_API_KEY")
    if not k and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
                k = winreg.QueryValueEx(h, "ANTHROPIC_API_KEY")[0]
        except OSError:
            k = None
    return k


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in open(LEDGER) if line.strip())


def cost(u):
    return (u.get("input_tokens", 0) * PRICE["input"] + u.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]
            + u.get("cache_read_input_tokens", 0) * PRICE["cache_read"] + u.get("output_tokens", 0) * PRICE["output"]) / 1e6


def image_tokens(w, h):
    return int(np.ceil(w * h / 750))


def font(sz):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def sagittal_stack(nifti_path):
    """(slices, rows, cols) from a PIR NIfTI: rows run head to feet, cols front to back,
    slice 0 = patient left (the layout render.py writes for a patient disc)."""
    img = nib.load(nifti_path)
    a = np.asarray(img.dataobj).astype(np.float32)
    z = img.header.get_zooms()
    return np.stack([a[:, :, k].T for k in range(a.shape[2])]), img.affine, (float(z[1]), float(z[0]), float(z[2]))


def render_case(p, split, loc_dir_name):
    """Render patient p's T2 and T1 to <panel>/<split>/png like render.py does; returns the series."""
    from render import write_series
    base = os.path.join(WORK, split)
    out = os.path.join(PANEL, split, "png")
    meta_p = os.path.join(out, "series_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    series = {}
    for seq in ("t2", "t1"):
        src = os.path.join(base, seq, f"{p}.nii.gz")
        if not os.path.exists(src):
            continue
        key = f"{p}_SAG_{seq.upper()}"
        vol, aff, (row_mm, col_mm, thick) = sagittal_stack(src)
        if key not in meta:
            used = write_series(out, key, vol)
            meta[key] = {"n": len(vol), "pixel_spacing": [row_mm, col_mm], "thickness": thick,
                         "window": used["standard"], "windows": used}
        series[seq] = {"key": key, "vol": vol, "affine": aff}
    json.dump(meta, open(meta_p, "w"), indent=1)
    return out, series


def disc_markers(lab_path, wanted):
    """{label: (level, row, col)} centroids on the sagittal stack layout for disc labels in `wanted`."""
    a = np.asarray(nib.load(lab_path).dataobj).astype(np.int32)
    out = {}
    for v in wanted:
        xx, yy, _ = np.nonzero(a == v)  # PIR: x = posterior column, y = inferior row
        if len(xx):
            out[v] = (level_of(NAMES[v]), float(yy.mean()), float(xx.min() + 0.15 * (xx.max() - xx.min())))
    return out


TILE_PX = 260     # width of each magnified disc-margin tile (the zoom adapts to the pixel size)
ZOOM_MM = (24.0, 20.0)   # crop size (front-to-back, head-to-foot) around each disc's back edge


def disc_backs(lab_path, wanted):
    """{label: (level, row, back_col, pixel_mm)}: each disc's back edge on the sagittal stack layout
    (the mean row of its most posterior 15% of voxels and their column)."""
    img = nib.load(lab_path)
    a = np.asarray(img.dataobj).astype(np.int32)
    z = img.header.get_zooms()
    out = {}
    for v in wanted:
        xx, yy, _ = np.nonzero(a == v)  # PIR: x = posterior column, y = inferior row
        if len(xx):
            back = xx >= np.percentile(xx, 85)
            out[v] = (level_of(NAMES[v]), float(yy[back].mean()), float(xx[back].mean()), (float(z[1]), float(z[0])))
    return out


def margin_montages(sources, backs, names=None):
    """[(label, image)]: for every disc, one montage of magnified crops of its back edge (annulus,
    ligament line, front of the canal) from each (series name, png path function, slices) source in
    the standard and fluid windows. Every disc gets the same treatment, so a reader can resolve a
    1-3 mm focus without being pointed at any level. `names` maps label -> how to call the disc."""
    out = []
    for lab, (level, r, c, (row_mm, col_mm)) in sorted(backs.items(), key=lambda kv: kv[1][1]):
        w, h = ZOOM_MM[0] / col_mm, ZOOM_MM[1] / row_mm
        box = (int(c - 0.6 * w), int(r - h / 2), int(c + 0.4 * w), int(r + h / 2))
        zoom = max(2, int(round(TILE_PX / w)))
        tiles = []
        for name, path_for, slices in sources:
            for s in slices:
                for win in ("", "_fluid"):
                    p = path_for(s, win)
                    if not os.path.exists(p):
                        continue
                    im = Image.open(p).convert("RGB").crop(box)
                    im = im.resize((im.width * zoom, im.height * zoom), Image.BICUBIC)
                    d = ImageDraw.Draw(im)
                    d.text((4, 2), f"{name} {s} {'fluid' if win else 'std'}", fill=(255, 255, 0), font=font(14))
                    d.line([(6, im.height - 8), (6 + 5 / col_mm * zoom, im.height - 8)], fill=(0, 255, 255), width=2)
                    tiles.append(im)
        if not tiles:
            continue
        per_row = 5   # keeps the montage under ~1,500 px wide, so it is not downscaled on the way in
        tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
        n_rows = (len(tiles) + per_row - 1) // per_row
        mont = Image.new("RGB", (min(len(tiles), per_row) * (tw + 4) - 4, n_rows * (th + 4) - 4), (40, 40, 40))
        for i, t in enumerate(tiles):
            mont.paste(t, ((i % per_row) * (tw + 4), (i // per_row) * (th + 4)))
        who = (names or {}).get(lab, level)
        out.append((f"Zoom x{zoom} on the back edge of disc {who}: each tile is one slice and window "
                    "(yellow caption), front of the patient to the left, cyan bar = 5 mm.", mont))
    return out


def nearest_slice(t2_aff, t2_shape, t2_k, t1_aff, t1_n):
    """The T1 slice nearest the centre of T2 slice t2_k (the centre, so a small angle between the
    two series does not move the answer)."""
    c = t2_aff @ np.array([(t2_shape[0] - 1) / 2, (t2_shape[1] - 1) / 2, t2_k, 1.0])
    k1 = (np.linalg.inv(t1_aff) @ c)[2]
    return int(np.clip(round(k1), 0, t1_n - 1))


def png_block(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def build_content(png_dir, series, mid, markers, backs=None):
    """User-message content: labelled images, a zoom montage of each marked disc's back edge, then
    the marker list. Returns (content, image tokens, letters)."""
    t2 = series["t2"]
    n = len(t2["vol"])
    letters = {}
    items, tokens = [], 0

    def add(label, path_or_img):
        nonlocal tokens
        im = Image.open(path_or_img).convert("RGB") if isinstance(path_or_img, str) else path_or_img
        tokens += image_tokens(*im.size)
        items.append({"type": "text", "text": label})
        items.append(png_block(im))

    k = t2["key"]
    im = Image.open(os.path.join(png_dir, k, f"{mid:02d}.png")).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, (lab, (level, r, c)) in enumerate(sorted(markers.items(), key=lambda kv: kv[1][1])):
        L = chr(ord("A") + i)
        letters[L] = (lab, level)
        d.line([(c - 14, r), (c - 3, r)], fill=(255, 220, 0), width=2)
        d.text((max(c - 60, 2), r - 9), f"{L} {level}", fill=(255, 220, 0), font=font(13))
    add(f"Image 1: T2 sagittal, slice {mid} of {n} (midline), standard window, discs marked.", im)
    for s in (mid - 2, mid - 1, mid + 1, mid + 2):
        if 0 <= s < n:
            add(f"T2 sagittal, slice {s} (0 = patient left), standard window.", os.path.join(png_dir, k, f"{s:02d}.png"))
    for w, what in (("fluid", "fluid and disc window"), ("marrow", "bone and marrow window")):
        add(f"T2 sagittal, slice {mid} (midline), {what}.", os.path.join(png_dir, k, f"{mid:02d}_{w}.png"))
    if "t1" in series:
        t1 = series["t1"]
        n_rows, n_cols = t2["vol"].shape[1], t2["vol"].shape[2]
        k1 = nearest_slice(t2["affine"], (n_cols, n_rows), mid, t1["affine"], len(t1["vol"]))
        for s in (k1 - 1, k1, k1 + 1):
            if 0 <= s < len(t1["vol"]):
                add(f"T1 sagittal, slice {s} (nearest the T2 midline is {k1}), standard window.",
                    os.path.join(png_dir, t1["key"], f"{s:02d}.png"))
    else:
        items.append({"type": "text", "text": "No T1 series exists for this patient."})
    if backs:
        names = {lab: f"{L} ({lev})" for L, (lab, lev) in letters.items()}
        src = [("T2", lambda s, win: os.path.join(png_dir, k, f"{s:02d}{win}.png"), [s for s in (mid - 1, mid, mid + 1) if 0 <= s < n])]
        for label, im in margin_montages(src, {lab: v for lab, v in backs.items() if lab in names}, names):
            add(label, im)
    lines = [f"{L}: {lev}" for L, (_, lev) in letters.items()]
    items.append({"type": "text", "text": "Marked discs to grade: " + "; ".join(lines) + "."})
    return items, tokens, letters


def disc_schema(letters):
    props = {"marker": {"type": "string", "enum": sorted(letters)}, "pfirrmann": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
             "pfirrmann_probs": {"type": "array", "items": {"type": "integer"}, "minItems": 5, "maxItems": 5},
             "notes": {"type": "string"}}
    for b in ITEMS:
        props[b] = {"type": "integer", "enum": [0, 1]}
        props[b + "_p"] = {"type": "integer"}
    return {"type": "object", "additionalProperties": False, "required": ["discs"],
            "properties": {"discs": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}}


def system_prompt(extra):
    """READER_BRIEF.md plus the task. Mistake-log references are stripped, so a case re-run is never
    told which level or finding it is being tested on."""
    brief = re.sub(r"\s*\((?:mistake-log )?case [^)]*\)", "", open(BRIEF, encoding="utf-8").read())
    return [{"type": "text", "text": brief + "\n\n" + extra, "cache_control": {"type": "ephemeral"}}]


BACKEND = "api"   # or "agents": no API key; each call becomes a bundle that Claude Opus 5.5 sub-agents read
BUNDLES = os.path.join(PANEL, "bundles")
PENDING = "PENDING"


def bundle_name(tag):
    """A neutral folder name: a case id or patient number in the path would hint at the answer."""
    import hashlib
    return "b" + hashlib.sha1(tag.encode()).hexdigest()[:10]


def write_bundle(tag, system, content, schema, cache, letters=None):
    """The agents backend's stand-in for an API call: the same instructions, images (as numbered
    PNGs, in order, each with its label) and output schema. The agent sees only that folder; which
    case or patient it is and where the answer belongs live in bundles/index.json, never shown to it."""
    d = os.path.join(BUNDLES, bundle_name(tag))
    os.makedirs(d, exist_ok=True)
    idx_p = os.path.join(BUNDLES, "index.json")
    idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    idx[bundle_name(tag)] = {"tag": tag, "cache": cache, "letters": letters}
    json.dump(idx, open(idx_p, "w"), indent=1)
    msgs, n, label = [], 0, None
    for b in content:
        if b["type"] == "text":
            msgs.append({"text": b["text"]})
        else:
            n += 1
            f = f"{n:02d}.png"
            Image.open(io.BytesIO(base64.standard_b64decode(b["source"]["data"]))).save(os.path.join(d, f))
            msgs.append({"image": f})
    json.dump({"instructions": system[0]["text"], "messages": msgs, "schema": schema},
              open(os.path.join(d, "request.json"), "w"), indent=1)
    return d


def ingest(answers_p):
    """Store sub-agent answers ([{"bundle": "<folder name>", "answer": {...}}]) where the API
    backend would have stored them."""
    idx = json.load(open(os.path.join(BUNDLES, "index.json")))
    n = 0
    for a in json.load(open(answers_p, encoding="utf-8")):
        req = idx[a["bundle"]]
        rec = {"tag": req["tag"], "backend": "claude-code-subagent", "model": MODEL}
        os.makedirs(os.path.dirname(req["cache"]), exist_ok=True)
        json.dump({"letters": req["letters"], "answer": a["answer"], "call": rec}, open(req["cache"], "w"), indent=1)
        n += 1
    print("ingested", n, "answers")


def call(client, system, content, schema, cap, tag, dry, cache=None, letters=None):
    if BACKEND == "agents" and not dry:
        return PENDING, {"bundle": write_bundle(tag, system, content, schema, cache, letters)}
    msgs = [{"role": "user", "content": content}]
    if client is not None:  # live: the API's own count, plus room for the output schema
        est_in = client.messages.count_tokens(model=MODEL, system=system, messages=msgs).input_tokens + 2000
    else:  # offline: a deliberately high guess (2 characters per token, images by area, fixed overheads)
        est_in = int(np.ceil(len(json.dumps(system)) / 2 + sum(len(b.get("text", "")) for b in content) / 2
                             + sum(image_tokens_from_block(b) + 50 for b in content) + len(json.dumps(schema)) / 2 + 500))
    worst = (est_in * max(PRICE["input"], PRICE["cache_write"]) + MAX_TOKENS * PRICE["output"]) / 1e6
    if dry:
        return None, {"estimated_input_tokens": est_in, "worst_case_usd": round(worst, 4)}
    if cap is None:
        sys.exit("A live call needs --cap-usd (your spending cap).")
    if spent() + worst > cap:
        sys.exit(f"Stopping: spent ${spent():.2f}, the next call could cost up to ${worst:.2f}, cap ${cap:.2f}.")
    t0 = time.time()
    r = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=msgs,
                               output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}})
    u = {k: getattr(r.usage, k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    rec = {"tag": tag, "model": r.model, "stop_reason": r.stop_reason, "usage": u, "cost_usd": cost(u),
           "seconds": round(time.time() - t0, 1), "request_id": r._request_id}
    os.makedirs(PANEL, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    text = next((b.text for b in r.content if b.type == "text"), None) if r.stop_reason == "end_turn" else None
    try:
        return (json.loads(text) if text else None), rec
    except json.JSONDecodeError:
        return None, rec   # recorded as a miss; the call is in the ledger


def image_tokens_from_block(b):
    if b.get("type") != "image":
        return 0
    im = Image.open(io.BytesIO(base64.standard_b64decode(b["source"]["data"])))
    return image_tokens(*im.size)


def client_or_none(dry):
    if dry or BACKEND == "agents":
        return None
    import anthropic
    k = api_key()
    if not k:
        sys.exit("No ANTHROPIC_API_KEY: use --backend agents (Claude Opus sub-agents, no key).")
    return anthropic.Anthropic(api_key=k)


def test_sample(n=30):
    """n test patients at random (seed 20260930), half per manufacturer (vendor)."""
    sp = load_split()
    g = np.random.default_rng(SEED)
    out = []
    by = {}
    for p in sp["test"]:
        by.setdefault(sp["patients"][str(p)]["vendor"], []).append(p)
    vendors = sorted(by)
    for i, v in enumerate(vendors):
        k = n // len(vendors) + (1 if i < n % len(vendors) else 0)
        members = sorted(by[v])
        out += [members[j] for j in sorted(g.permutation(len(members))[:k])]
    return sorted(out)


def dev_samples():
    """Prompt tuning uses 10 fold-0 patients; the panel's development score uses 30 other development
    patients (folds 1-4, half per scanner maker), so the patients it is scored on never shaped its prompt.
    Both drawn with seed 20260930 among patients with a T2."""
    sp = load_split()
    g = np.random.default_rng(SEED)
    has = lambda p: bool(sp["patients"][str(p)]["t2"])
    f0 = sorted(p for p in sp["dev_folds"]["0"] if has(p))
    tune = sorted(f0[i] for i in g.permutation(len(f0))[:10])
    rest = sorted(p for k in "1234" for p in sp["dev_folds"][k] if has(p))
    by = {}
    for p in rest:
        by.setdefault(sp["patients"][str(p)]["vendor"], []).append(p)
    score = []
    for v in sorted(by):
        score += [by[v][i] for i in sorted(g.permutation(len(by[v]))[:15])]
    return {"tune": tune, "score": sorted(score)}


CALIB = os.path.join(PANEL, "dev", "calibration.json")


def expected_grade(d):
    """The reader's mean Pfirrmann grade from its five grade probabilities (its single grade if absent)."""
    p = d.get("pfirrmann_probs")
    if not p or len(p) != 5 or sum(max(x, 0) for x in p) <= 0:
        return float(d["pfirrmann"])
    p = np.clip(np.asarray(p, float), 0, None)
    return float((np.arange(1, 6) * p).sum() / p.sum())


def calibrate(tune):
    """Fit on the tuning patients only, the way every other method gets its thresholds from development
    data: four cut-points on the reader's mean grade (most agreement in quadratic weighted kappa) and,
    per yes/no item, a Youden threshold on its probability (kept as the reader's own call when the
    tuning discs hold fewer than 5 positives or negatives)."""
    from rules import fit_cuts, fit_youden
    t = tune[(tune.localisation == "tss") & (tune.in_scope == 1)].merge(grades(), on=["patient", "ivd"])
    g = t.dropna(subset=["pfirrmann", "pfirrmann_expect"])
    thr = {}
    for b in ITEMS:
        s = t.dropna(subset=[b, b + "_score"])
        y = (s[b] > 0).astype(int)
        thr[b] = fit_youden(s[b + "_score"], y) if min(y.sum(), (1 - y).sum()) >= 5 else None
    c = {"fitted_on": "panel tuning patients (dev fold 0)", "patients": sorted(int(p) for p in t.patient.unique()),
         "n_discs": int(len(g)), "pfirrmann_cuts_on_negative_mean_grade": fit_cuts(-g.pfirrmann_expect, g.pfirrmann.astype(int)),
         "thresholds": thr}
    json.dump(c, open(CALIB, "w"), indent=1)
    print("panel calibration:", c)


def calibrated(df):
    if not os.path.exists(CALIB):
        sys.exit("No panel calibration: run `spider dev --dev-set tune` first.")
    c = json.load(open(CALIB))
    from rules import predict_grade
    out = df.copy()
    has = out.pfirrmann_expect.notna()
    out.loc[has, "pfirrmann_pred"] = predict_grade(-out.loc[has, "pfirrmann_expect"].values,
                                                   c["pfirrmann_cuts_on_negative_mean_grade"])
    for b, thr in c["thresholds"].items():
        s = out[b + "_score"]
        if thr is not None:
            out.loc[s.notna(), b + "_pred"] = (s[s.notna()] >= thr).astype(int)
    return out.drop(columns=["pfirrmann_expect"])


def run_spider(split, patients, cap, dry, out_name=None):
    base = os.path.join(WORK, split)
    m = pd.read_csv(os.path.join(base, "matches.csv"), keep_default_na=False)
    client = client_or_none(dry)
    raw_dir = os.path.join(PANEL, split, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    rows, est, pending = [], [], 0
    for p in patients:
        mp = m[m.patient == p]
        if mp.empty or (mp.drop_reason == "no T2").all():
            continue
        png_dir, series = render_case(p, split, "tss")
        mid = json.load(open(os.path.join(base, "metrics", f"{p}_tss.json"))).get("mid_slice", len(series["t2"]["vol"]) // 2)
        tss_ids = [k for k, v in NAMES.items() if v.startswith("disc_") and level_of(v) in SCORED]
        answers = {}
        for loc, lab_dir, ids in (("tss", "tss/step2_output", tss_ids), ("spider_masks", "spiderlab", None)):
            if loc == "spider_masks":
                # discs TotalSpineSeg did not find get a second call, marked from SPIDER's masks
                miss = mp[(mp.in_scope == 1) & (mp.matched == 0)]
                if miss.empty:
                    break
                from prepare import disc_id
                ids = [disc_id(int(i)) for i in miss.ivd]
            markers = disc_markers(os.path.join(base, lab_dir, f"{p}.nii.gz"), ids)
            if not markers:
                continue
            backs = disc_backs(os.path.join(base, lab_dir, f"{p}.nii.gz"), list(markers))
            content, _, letters = build_content(png_dir, series, mid, markers, backs)
            cache = os.path.join(raw_dir, f"{p}_{loc}.json")
            if os.path.exists(cache) and not dry:
                res = json.load(open(cache))
            else:
                ans, rec = call(client, system_prompt(GRADING), content, disc_schema(letters), cap, f"{split} {p} {loc}", dry,
                                cache=cache, letters=letters)
                res = {"letters": letters, "answer": ans, "call": rec}
                if dry:
                    est.append(rec["worst_case_usd"])
                    continue
                if ans == PENDING:
                    pending += 1
                    continue
                json.dump(res, open(cache, "w"), indent=1)
            by_label = {}
            for d in (res["answer"] or {}).get("discs", []):
                lab = res["letters"].get(d["marker"])
                if lab:
                    by_label[int(lab[0])] = d
            answers[loc] = by_label
        if dry:
            continue
        for _, r in mp.iterrows():
            for loc in ("tss", "spider_masks"):
                d = None
                if r.in_scope == 1 and r.matched == 1 and r.tss_label != "":
                    d = answers.get("tss", {}).get(int(r.tss_label))
                elif r.in_scope == 1 and loc == "spider_masks":
                    from prepare import disc_id
                    d = answers.get("spider_masks", {}).get(disc_id(int(r.ivd)))
                row = {c: r[c] for c in ("patient", "ivd", "fold", "vendor", "nominal_level", "tss_level", "dice",
                                         "matched", "in_scope", "drop_reason")}
                row.update(method="panel", localisation=loc, split=split,
                           pfirrmann_pred=d["pfirrmann"] if d else np.nan,
                           pfirrmann_expect=expected_grade(d) if d else np.nan)
                for b in ITEMS:
                    row[f"{b}_pred"] = d[b] if d else np.nan
                    row[f"{b}_score"] = min(max(d[b + "_p"], 0), 100) / 100 if d else np.nan
                rows.append(row)
    if dry:
        print(f"dry run: {len(est)} calls, worst case ${sum(est):.2f} total, ${np.mean(est) if est else 0:.3f} per call")
        return
    if pending:
        print(f"{pending} bundles written to {BUNDLES}; answer them with sub-agents, `ingest`, then run this again")
        return
    out = os.path.join(PANEL, split, out_name) if out_name else os.path.join(os.path.dirname(WORK), f"pred_panel_{split}.csv")
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    new = pd.DataFrame(rows)
    if out_name is None:  # a scored output: the reader's calls through the cut-points fitted on the tuning patients
        new = calibrated(new)
    elif out_name == "tune_preds.csv":
        calibrate(new)
    if not old.empty:
        old = old[~old.patient.isin(new.patient.unique())]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)
    print("wrote", out, "spent so far $%.2f" % spent())


def run_case(case_p, cd, cap, dry):
    """Re-run one mistake-log case on a patient's own disc; prints PASS or FAIL per expected answer."""
    from render import main as render_disc
    case = json.load(open(case_p, encoding="utf-8"))
    reader = os.path.join(PANEL, "cases", "reader")
    png = os.path.join(reader, "png")
    if not os.path.exists(os.path.join(png, "series_meta.json")):
        render_disc(cd, reader)
    content = []
    for s, slices in case["series"].items():
        for k in slices:
            for w in ("", "_fluid", "_marrow"):
                content.append({"type": "text", "text": f"{s} slice {k}{' ' + w[1:] + ' window' if w else ' standard window'}."})
                content.append(png_block(Image.open(os.path.join(png, s, f"{k:02d}{w}.png")).convert("RGB")))
    # the same back-edge zoom for every disc of the region (never only the case's level)
    sag_t2 = [s for s in case["series"] if s.endswith("SAG_T2")]
    lab_p = os.path.join(os.path.dirname(WORK), "..", "tss_out", "step2_output", f"{sag_t2[0]}.nii.gz") if sag_t2 else None
    if lab_p and os.path.exists(lab_p):
        region_levels = {"lumbar": SCORED, "cervical": ["C2-C3", "C3-C4", "C4-C5", "C5-C6", "C6-C7", "C7-T1"]}[case["region"]]
        ids = [k for k, v in NAMES.items() if v.startswith("disc_") and level_of(v) in region_levels]
        src = [(s.split("_")[-1], (lambda s_: lambda k, win: os.path.join(png, s_, f"{k:02d}{win}.png"))(s), sl)
               for s, sl in case["series"].items()]
        for label, im in margin_montages(src, disc_backs(lab_p, ids)):
            content.append({"type": "text", "text": label})
            content.append(png_block(im))
    t = case["test"]
    content.append({"type": "text", "text": t["question"]})
    schema = {"type": "object", "additionalProperties": False, "required": ["answers"], "properties": {"answers": {
        "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["level", t["field"], "evidence"],
                                   "properties": {"level": {"type": "string", "enum": t["levels"]},
                                                  t["field"]: {"type": "string", "enum": t["choices"]},
                                                  "evidence": {"type": "string"}}}}}}
    extra = "## This task\nA single question about this scan (the series named in the brief). Read the attached images."
    cache = os.path.join(PANEL, "cases", case["id"] + ".answer.json")
    if os.path.exists(cache) and not dry:
        c = json.load(open(cache))
        ans, rec = c["answer"], c["call"]
    else:
        ans, rec = call(client_or_none(dry), system_prompt(extra), content, schema, cap, f"case {case['id']}", dry, cache=cache)
    if dry:
        print(case["id"], "dry run:", rec)
        return True
    if ans == PENDING:
        print(case["id"], "bundle written:", rec["bundle"])
        return True
    got = {a["level"]: a[t["field"]] for a in (ans or {}).get("answers", [])}
    ok = all(got.get(k) == v for k, v in t["expect"].items())
    res = {"case": case["id"], "pass": ok, "expect": t["expect"], "got": got, "call": rec, "answer": ans}
    os.makedirs(os.path.join(PANEL, "cases"), exist_ok=True)
    json.dump(res, open(os.path.join(PANEL, "cases", case["id"] + ".result.json"), "w"), indent=1)
    print(case["id"], "PASS" if ok else "FAIL", {k: got.get(k) for k in t["expect"]})
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["spider", "case", "spend", "ingest"])
    ap.add_argument("--backend", choices=["api", "agents"], default="api")
    ap.add_argument("target", nargs="?")
    ap.add_argument("--patients")
    ap.add_argument("--sample-test", type=int)
    ap.add_argument("--dev-set", choices=["tune", "score"], help="the panel's development patients (dev_samples)")
    ap.add_argument("--cd")
    ap.add_argument("--cap-usd", type=float)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global BACKEND
    BACKEND = a.backend
    if a.mode == "ingest":
        ingest(a.target)
    elif a.mode == "spend":
        print(f"spent ${spent():.2f} over {sum(1 for _ in open(LEDGER)) if os.path.exists(LEDGER) else 0} calls")
    elif a.mode == "spider":
        pts = (test_sample(a.sample_test) if a.sample_test else dev_samples()[a.dev_set] if a.dev_set
               else [int(x) for x in a.patients.split(",")])
        # only the development scoring sample (and the final test sample) feed predictions.csv;
        # prompt tuning and ad-hoc runs (e.g. the 3-patient cost measurement) are kept apart
        keep = a.dev_set == "score" or a.target == "test"
        run_spider(a.target, pts, a.cap_usd, a.dry_run,
                   None if keep else ("tune_preds.csv" if a.dev_set == "tune" else "adhoc_preds.csv"))
    else:
        sys.exit(0 if run_case(a.target, a.cd, a.cap_usd, a.dry_run) else 1)


if __name__ == "__main__":
    main()
