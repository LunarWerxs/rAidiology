# The findings file

A reading happens on the person's own computer: their AI, rAidiology's MCP server, their scan,
nothing uploaded. The pictures, though, are best looked at in a browser. This file is what crosses
between the two. It is a few kilobytes of plain facts that the person can drop onto
rAidiology's "Open your own scan" page alongside the scan itself, and the page then
shows them their own images with what was found marked on them.

Written by `core/findings.py`. Read by `site/js/open/findings.js`. Named `<something>.raidiology.json.gz`.

## What it is

Gzipped JSON. A real one measures about a kilobyte. That size is the point: everything in it can be
read by any program anybody cares to write, it fits in a chat message, and there is nothing in it
heavy enough to need a server.

```jsonc
{
  "raidiology_findings": 1,              // refuse anything that is not exactly this
  "written": "2026-10-01T09:14:02Z",
  "tool": "rAidiology MCP server 0.1.0",
  "scan": {
    "modality": "CR",
    "description": "WRIST",
    "body_part": "WRIST",
    "images": 3,
    "series_fingerprint": "59926243e3754218",
    "ordered_by": "image number"
  },
  "model": {
    "name": "rAidiology X-ray fracture detector",
    "version": "1", "profile": "grazpedwri", "threshold": 0.563,
    "measured_on": "4,170 held-out paediatric wrist radiographs from 1,218 patients…",
    "auroc": 0.976, "auroc_95ci": [0.970, 0.981],
    "catches": "92.7% of fractures (95% CI 91.2 to 94.1)",
    "clears": "94.4% of unbroken images correctly (92.4 to 96.1)",
    "who_it_is_not_for": "…",
    "caveat": "The highlight shows where the score came from, not where a fracture is."
  },
  "summary": "1 of the 3 images read scored above the line the detector was scored at. …",
  "findings": [
    {
      "id": "f1",
      "label": "possible fracture",
      "plain": "Image 2 of 3 is the kind of picture the detector calls broken.",
      "score": 0.81,
      "confidence": "well above the line",
      "image_fingerprint": "7c1e0b4a9d2f5531",
      "image_index": 1,
      "where": { "grid": [6, 6], "cells": [0, 12, …], "note": "A coarse map of …" }
    }
  ],
  "not_a_medical_device": "…"
}
```

## The three things the format is for

**It carries no identity, by construction.** Not as a policy somebody has to remember: `findings.py`
assembles the document from its own allow-list of keys, so a name, a patient number or a date of
birth cannot pass through it even if a caller tries. The series and each image are referred to by
`sha256(uid)[:16]`, never the identifier. The browser computes the same fingerprint from the scan it
is already holding, so the matching happens on the visitor's own device out of two things they
brought themselves, and the file on its own tells a stranger nothing about whose scan it is.

**It says what it is not.** The held-out accuracy, who the model is not for, and the
not-a-medical-device line are in every file. A reading that outlives the conversation it came from
still arrives with its limits attached, and the viewer prints them under the findings rather than
tucking them away.

**It never claims more than was measured.** `where` is a 6x6 grid of bytes, and that is not a
shortcut: it is the whole spatial resolution the detector has. The network averages its feature map
and then applies one linear layer, so the per-cell value is literally the decomposition of the
number the model produced, and averaging the grid returns the logit exactly. What it is *not* is a
location: the detector was scored on whether a fracture is present, never on where it is. Both the
file and the page say so in those words.

## Writing one

```python
from core import findings as ff

doc = ff.document(
    scan={"modality": "CR", "images": 3,
          "series_fingerprint": ff.fingerprint(series_instance_uid)},
    model={"name": "…", "threshold": 0.563},
    findings=[ff.finding("f1", "possible fracture", "Image 2 of 3 …", score=0.81,
                         image_fingerprint=ff.fingerprint(sop_instance_uid), image_index=1,
                         where=ff.grid(cells, 6, 6))],
    summary="…")
path, raw_bytes, packed_bytes = ff.write(doc, "C:/Users/me/Downloads/wrist")
```

`ff.read(path)` is the same contract backwards and refuses anything `document()` would not have
produced. A file from a future version is declined rather than guessed at, because the person
reading it is reading about their own body.

## Outlines: a CT reading

A labeller does not find one thing on one image; it puts every organ, bone and large vessel
somewhere on every image. `read_ct` with `findings` writes the same file with one more section,
`outlines`, and the page draws each structure's outline over the person's own slices, with a key of
what is outlined on the picture in view. A file without the section is an X-ray reading exactly as
before, so the version stays 1.

```jsonc
"outlines": {
  "structures": ["spleen", "right kidney", "left kidney", …],    // in the model's order, plain words
  "dice_on_test": [0.9427, 0.8695, 0.8757, …],                   // each one's Dice on the held-out test
  "slices": [
    {"image_fingerprint": "3b7c…", "image_index": 22,
     "paths": [[6, "KlICAAIG…"], [0, "HhwCAg…"], …]}             // [structure, outline]
  ],
  "note": "…"
}
```

An outline is a closed path in the image's own pixels, (0, 0) the top-left corner, so a pixel's
centre is at (column + 0.5, row + 0.5). It is written in half-pixel steps, each point the difference
from the one before, zigzag-varint coded, base64 (`findings.encode_path`, and `decodePath` in
`site/js/open/findings.js`). That is about a byte a point: the 46-image sample scan's reading, 54
structures, is 47 KB. The outlines come from the labels the model wrote, turned back from the
model's orientation to the volume SimpleITK built from the files, slice k on the k-th image in
space; `tools/check_ct_series.py --read` fills them back in and requires them to lie on those
labels, image by image (mean Dice 0.98 on a development scan; mirrored on purpose, 0.12).

Every structure is also an ordinary finding, `label` in plain words, pointing at the image where it
is largest, so the page's list of findings is a list of what was outlined, each a button that opens
the image where it is biggest. The model section carries `mean_dice`, `mean_dice_95ci` and `invents`
instead of an AUROC, and its `caveat`: an outline shows where the model put a structure, not whether
it is healthy.

## Matching, and what happens when it fails

The page matches on `series_fingerprint` first. Inside the series it matches each finding on
`image_fingerprint`, falling back to `image_index` for a disc whose files carry no image identifier
at all. Three failures are possible and all three are shown to the visitor in plain words rather
than swallowed:

| what went wrong | what the page says |
|---|---|
| the reading is of a different scan | "That reading is about a different scan from the one open here." |
| the version is not 1 | "That file says it is rAidiology findings version N, and this page reads version 1." |
| it is not readable | "That file is not a rAidiology reading: it is not readable as JSON." |

## Checking the whole chain

    python mcp/server.py --selftest

With no folder it makes up a DICOM series (`mcp/demo_scan.py`), runs find → read → write over it,
and checks the file reads back pointing at the series it was written from. That last check is the
one that matters: the fingerprint is computed in Python on one side and in JavaScript on the other,
and a reading that quietly stops matching its own pictures is a break no single-language test would
ever catch.
