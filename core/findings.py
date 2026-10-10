"""The small file a reading leaves behind, so the viewer can show it.

The route rAidiology ships is the visitor's own machine: their AI runs the MCP server, the measured
model scores their scan there, and nothing is uploaded. That leaves one gap. The reading happens in
a chat window, and the pictures live in the browser. This file is the bridge: a few kilobytes of
plain facts that the person can drop onto the viewer, which then shows them the image the model was
worried about with the region highlighted.

Three rules the format exists to keep:

**It carries no identity.** Not as a policy, as a mechanism: this module builds the dictionary
itself out of its own allow-list, so a caller cannot pass a name, a patient number or a date of
birth through it even by accident. The series and each image are referenced by a one-way fingerprint
of their DICOM identifier, never the identifier, which is enough for the browser to recognise the
scan it is already holding and useless to anybody else.

**It is small and boring.** Plain JSON, gzipped, a few kilobytes. The highlight is a 6x6 grid of
bytes, because that is genuinely all the model has: everything in it can be read by any program
anybody wants to write, and the whole thing fits in a message.

**It says what it is not.** Every file carries the held-out accuracy, who the model is not for, and
that rAidiology is not a medical device, so a file that outlives this conversation still arrives
with its limits attached.

    from core import findings
    doc = findings.document(scan=..., model=..., findings=[...], summary="...")
    path = findings.write(doc, "C:/Users/me/Downloads/wrist")

`read()` is the same contract in reverse and is what the tests use: a file this module wrote must
survive a round trip, and a file with anything unexpected in it is refused rather than guessed at.
"""
import base64
import gzip
import hashlib
import io
import json
import os
import time

VERSION = 1
SUFFIX = ".raidiology.json.gz"
NOT_A_DEVICE = ("Not a medical device. rAidiology is for education and research, not for "
                "diagnosis. Always talk to a doctor about your results.")

# Every key a findings document may hold, by section. A key that is not here does not reach the
# file. That is the whole privacy mechanism: there is no pass-through dictionary anywhere below.
SCAN_KEYS = ("modality", "description", "body_part", "images", "series_fingerprint", "ordered_by")
MODEL_KEYS = ("name", "version", "profile", "threshold", "measured_on", "auroc", "auroc_95ci",
              "catches", "clears", "who_it_is_not_for", "caveat", "mean_dice", "mean_dice_95ci",
              "invents")
FINDING_KEYS = ("id", "image_fingerprint", "image_index", "label", "plain", "score", "confidence",
                "where")
WHERE_KEYS = ("grid", "cells", "note")
TOP_KEYS = ("raidiology_findings", "written", "tool", "scan", "model", "summary", "findings",
            "not_a_medical_device", "what_this_file_is", "outlines")
# Outlines: what a labeller (CT) drew on each image, as closed paths in the image's own pixels.
OUTLINE_KEYS = ("structures", "about", "dice_on_test", "slices", "note")
SLICE_KEYS = ("image_fingerprint", "image_index", "paths")

GRID_NOTE = ("A coarse map of which part of the image moved the score, not a measured location of "
             "anything. Each cell covers about a sixteenth of the width. The model was scored on "
             "whether a fracture is present, never on where it is, so treat this as a place to "
             "look rather than an answer.")


def fingerprint(uid):
    """A one-way, 16-character fingerprint of a DICOM identifier.

    The browser computes the same thing from the scan it already has, so the two can be matched
    without this file ever carrying an identifier somebody could look up. SHA-256 truncated: the
    full digest is 64 characters of noise in a file meant to stay small, and 16 hex characters is
    64 bits, which will not collide across the handful of series on one disc."""
    text = str(uid or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _pick(mapping, keys, what):
    """Copy across only the keys that belong, and say plainly when one does not.

    A caller passing something unexpected is a bug worth hearing about, not something to drop
    quietly: quietly dropping it is how a field nobody meant to publish gets added later and nobody
    notices it was being ignored."""
    unknown = sorted(set(mapping) - set(keys))
    if unknown:
        raise ValueError("%s may not carry %s. The allowed keys are %s, and that list is the "
                         "privacy rule: anything else would have to be added here on purpose."
                         % (what, ", ".join(unknown), ", ".join(keys)))
    return {k: mapping[k] for k in keys if k in mapping and mapping[k] is not None}


def grid(values, width, height):
    """A heat map as bytes, normalised so the strongest cell in this image is 255.

    Normalised per image on purpose. The raw numbers are a logit decomposition whose absolute scale
    means nothing to a reader, and a map shaded against its own strongest cell answers the only
    question it can honestly answer: within this picture, where did the score come from."""
    vals = [float(v) for v in values]
    if len(vals) != width * height:
        raise ValueError("a %dx%d grid needs %d values, got %d" % (width, height, width * height, len(vals)))
    lo, hi = min(vals), max(vals)
    span = hi - lo
    cells = [0] * len(vals) if span <= 0 else [int(round((v - lo) / span * 255)) for v in vals]
    return {"grid": [int(width), int(height)], "cells": cells, "note": GRID_NOTE}


def encode_path(points):
    """A closed outline as text: the points in half-pixel steps, each one the difference from the last,
    zigzag-varint coded, base64. A point is (x, y) in the image's pixels, (0, 0) its top-left corner, so
    a pixel's centre is at (column + 0.5, row + 0.5). About a byte a point, which is what keeps a
    whole CT's outlines to a few hundred kilobytes."""
    out = bytearray()
    px = py = 0
    for x, y in points:
        hx, hy = int(round(2 * x)), int(round(2 * y))
        for d in (hx - px, hy - py):
            z = (d << 1) ^ (d >> 63)            # zigzag: small negative numbers stay small
            while z >= 0x80:
                out.append((z & 0x7F) | 0x80)
                z >>= 7
            out.append(z)
        px, py = hx, hy
    return base64.b64encode(bytes(out)).decode("ascii")


def decode_path(text):
    """`encode_path` in reverse: the points, in pixels."""
    raw = base64.b64decode(text)
    vals, z, shift = [], 0, 0
    for b in raw:
        z |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            vals.append((z >> 1) ^ -(z & 1))
            z, shift = 0, 0
    pts, x, y = [], 0, 0
    for i in range(0, len(vals) - 1, 2):
        x, y = x + vals[i], y + vals[i + 1]
        pts.append((x / 2.0, y / 2.0))
    return pts


def _paths(row):
    """An outlined image's paths, each [structure number, outline text], and nothing else."""
    for item in row.get("paths", []):
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[0], int)
                and isinstance(item[1], str)):
            raise ValueError("an outline path is [structure number, encoded outline], got %r" % (item,))
    return row


def _where(f):
    w = f.get("where")
    if w is not None:
        _pick(w, WHERE_KEYS, "a finding's where")
        if not all(isinstance(v, int) for v in list(w.get("grid", [])) + list(w.get("cells", []))):
            raise ValueError("a finding's where holds numbers only")
    return f


def outlines(structures, dice_on_test, slices, note, about=None):
    """The outlines section: structure names, a line on what each one is (`about`, for somebody with
    no medical training), each one's Dice on the held-out test (how far that kind of outline can be
    trusted), and per image the paths drawn on it as [structure index, path]."""
    rows = [_paths(_pick(dict(s), SLICE_KEYS, "an outlined image")) for s in slices]
    return _pick({"structures": list(structures), "about": list(about) if about else None,
                  "dice_on_test": list(dice_on_test), "slices": rows, "note": str(note)},
                 OUTLINE_KEYS, "the outlines section")


def finding(id, label, plain, score=None, confidence=None, image_fingerprint=None,
            image_index=None, where=None):
    """One thing the model found, in the shape the file stores it."""
    return _pick({"id": str(id), "label": str(label), "plain": str(plain), "score": score,
                  "confidence": confidence, "image_fingerprint": image_fingerprint,
                  "image_index": image_index, "where": where},
                 FINDING_KEYS, "a finding")


def document(scan, model, findings, summary, written=None, tool=None, outlined=None):
    """The whole file, assembled and checked. `outlined` is what `outlines()` built, for a labeller.

    `written` is passed in rather than taken from the clock so a caller that wants a reproducible
    file can have one; the MCP server passes the real time."""
    rows = list(findings)
    for f in rows:
        _where(_pick(f, FINDING_KEYS, "a finding"))
    extra = {"outlines": _pick(dict(outlined), OUTLINE_KEYS, "the outlines section")} if outlined else {}
    return {**extra, 
        "raidiology_findings": VERSION,
        "written": written or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": str(tool or "rAidiology MCP server"),
        "scan": _pick(dict(scan), SCAN_KEYS, "the scan section"),
        "model": _pick(dict(model), MODEL_KEYS, "the model section"),
        "summary": str(summary),
        "findings": rows,
        "not_a_medical_device": NOT_A_DEVICE,
        # No web address in here on purpose. This string travels with somebody's reading and
        # outlives the conversation that made it, and a hard-coded host is a thing that rots, moves
        # or turns out to belong to one person's own deployment rather than to rAidiology.
        "what_this_file_is": "A reading of one scan, written by rAidiology on the machine the scan "
                             "was opened on. It holds no name, patient number or date of birth. "
                             "Open it on rAidiology's “Open your own scan” page together "
                             "with the scan itself to see the images with the findings marked.",
    }


def write(doc, path):
    """Write the document, gzipped, and return where it landed.

    `path` may be a folder, a name, or a full file name. A name without the suffix gets it, so a
    caller cannot accidentally produce a file the viewer will not recognise."""
    target = str(path)
    if os.path.isdir(target):
        target = os.path.join(target, "scan" + SUFFIX)
    elif not target.endswith(SUFFIX):
        target = target[:-3] if target.endswith(".gz") else target
        target = target[:-5] if target.endswith(".json") else target
        target += SUFFIX
    folder = os.path.dirname(os.path.abspath(target))
    os.makedirs(folder, exist_ok=True)
    body = json.dumps(doc, indent=1, sort_keys=True).encode("utf-8")
    # mtime=0 so the same reading written twice is the same bytes: a file that differs only by a
    # timestamp looks like a different reading to anybody diffing two of them.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(body)
    with open(target, "wb") as f:
        f.write(buf.getvalue())
    return target, len(body), os.path.getsize(target)


def read(path):
    """Read one back, refusing anything this module would not have written.

    Refusing rather than guessing: the viewer shows this to somebody about their own body, and a
    file from a future version whose meaning has changed is worse than a file it declines to open."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    doc = json.loads(raw.decode("utf-8"))
    got = doc.get("raidiology_findings")
    if got != VERSION:
        raise ValueError("this is not a rAidiology findings file version %d (it says %r)"
                         % (VERSION, got))
    _pick(doc, TOP_KEYS, "a findings file")
    _pick(doc.get("scan", {}), SCAN_KEYS, "the scan section")
    _pick(doc.get("model", {}), MODEL_KEYS, "the model section")
    for f in doc.get("findings", []):
        _where(_pick(f, FINDING_KEYS, "a finding"))
    if "outlines" in doc:
        _pick(doc["outlines"], OUTLINE_KEYS, "the outlines section")
        for row in doc["outlines"].get("slices", []):
            _paths(_pick(row, SLICE_KEYS, "an outlined image"))
    return doc
