# Data contract: what the analysis pipeline writes and the web app reads

The site is static (any static host; a real scan goes behind access control, see the README).
Everything the app shows comes from `site/data/`. The pipeline (Python, `pipeline/`) writes these files; the app must render gracefully when
any optional file is missing (show a tasteful empty state, never a blank page or a console crash).

## `data/report.json` (required)

```json
{
  "version": 1,
  "patient": { "label": "Anonymous", "age": 52, "sex": "F" },
  "study": { "month": "2025-03", "field_strength": "3T", "contrast": false,
             "exams": ["Cervical spine MRI", "Lumbar spine MRI"] },
  "summary": {
    "status": "ok | watch | see-doctor",
    "headline": "One sentence, plain English.",
    "plain": "Three or four sentences an 18-year-old gets in one read.",
    "technical": "Radiology-style impression paragraph."
  },
  "regions": [
    {
      "id": "cervical | lumbar",
      "name": "Neck (cervical spine)",
      "anchor": "neck | lower_back",
      "levels_present": ["C1", "C2", "...", "T1"],
      "summary": "plain-English one-paragraph summary",
      "meshes": [
        { "name": "L4", "kind": "vertebra | disc | cord | canal | sacrum | csf",
          "level": "L4", "file": "data/meshes/lumbar/L4.glb",
          "status": "normal | mild | moderate | severe" }
      ]
    }
  ],
  "levels": [
    { "region": "lumbar", "level": "L4-L5", "status": "normal | mild | moderate | severe",
      "disc": "text", "canal": "none | mild | moderate | severe",
      "foramen_left": "none | mild | moderate | severe",
      "foramen_right": "none | mild | moderate | severe", "note": "text" }
  ],
  "findings": [
    {
      "id": "lumbar-l4l5-disc",
      "region": "lumbar",
      "level": "L4-L5",
      "title": "Short title (plain English)",
      "technical_title": "Radiology wording",
      "severity": "normal-variant | incidental | mild | moderate | severe",
      "confidence": "high | medium | low",
      "plain": "What it is, in plain English.",
      "technical": "What it is, in radiology wording.",
      "why_it_matters": "What it can cause / what it usually means at 52.",
      "what_to_do": "Sensible next step (non-prescriptive).",
      "anchor": { "mesh": "disc_L4_L5", "point": [0.0, 0.0, 0.0] },
      "views": [
        { "stack": "L_SAG_T2", "slice": 6, "caption": "Side view",
          "marks": [ { "type": "circle", "x": 0.52, "y": 0.61, "r": 0.04 },
                     { "type": "arrow", "x": 0.52, "y": 0.61, "dx": -0.08, "dy": -0.05 } ] }
      ]
    }
  ],
  "glossary": [ { "term": "Disc", "plain": "..." } ],
  "questions_for_doctor": ["..."],
  "method": { "ai_models": ["..."], "readers": "text", "limitations": ["..."] },
  "disclaimer": "Not a diagnosis. ..."
}
```

### Optional per-level fields

`levels[].measure` (optional): automated measurements, only the keys that were measured.

| key | meaning |
|---|---|
| `disc_height_mm` | disc height on the mid-sagittal T2 (`metrics.py`) |
| `disc_to_csf_signal` | disc signal divided by spinal-fluid signal on the same slices (higher = more water) |
| `canal_ap_at_disc_mm` | front-to-back width of the fluid sac at the disc |
| `cord_ap_at_disc_mm` | front-to-back width of the cord at the disc |
| `cord_compression_ratio_hc` | cervical only, in percent: how much narrower front to back the cord is at this level than expected from healthy adults of the same sex and age +-10 years (Spinal Cord Toolbox `sct_compute_compression -normalize-hc 1`, its `diameter_AP_ratio_PAM50_normalized`; database of 203 healthy adults, `pipeline/sct/`). 0 = as expected, positive = narrower, negative = wider. Present only where the reading reports cord or canal narrowing. PAM50 template: permission pending. |

`levels[].specialist` (optional, lumbar T12-L1 to L5-S1 only): the shipped specialist model's
grades. Present only when a method passed the ship rule on the held-out test split
(`pipeline/eval/SCORECARD.md`); otherwise absent and the app shows nothing.

```json
{ "model": "SuprSkan disc grader v1",
  "pfirrmann": 3, "pfirrmann_confidence": 0.62, "pfirrmann_probs": [0.02, 0.21, 0.62, 0.13, 0.02],
  "items": { "herniation": 0.08, "narrowing": 0.31, "bulging": 0.71, "spondylolisthesis": 0.01,
             "up_endplate": 0.22, "low_endplate": 0.18, "modic": 0.05 },
  "flags": ["bulging"],
  "not_blind": true }
```

Confidences are calibrated probabilities: of discs given 0.8, about 80% matched the expert on
development data. `flags` lists the items the model puts at 50% or more (more likely than not),
most likely first; `items` holds every item's probability. `not_blind` is
true when the scan already feeds a mistake-log rule (its re-run reading is development data).

### Optional top-level `scorecard`

Present only with `levels[].specialist`. The app shows `headline.text` as one line on the
Overview and never claims more than it says.

```json
{ "method": "SuprSkan disc grader v1", "split": "held-out test, SPIDER",
  "n_scans": 44, "n_discs": 264,
  "headline": { "item": "pfirrmann", "metric": "exact", "value": 0.61, "ci": [0.52, 0.68],
                "text": "Tested on 44 expert-graded scans: agrees with the expert on disc drying 61% of the time (95% CI 52-68%)." },
  "permission": "TSS weights: permission pending",
  "source": "pipeline/eval/SCORECARD.md" }
```

### Optional `comparison.model_label` and `comparison.rows[].model_short`

When a specialist model has an opinion on a comparison row, `rows[].model_short` holds it in a few
words and `comparison.model_label` names the column (for example "Specialist model (not blind)").
Rows without `model_short` show nothing in that column.

Severity colours (use everywhere, including 3D): normal/normal-variant = teal/green,
incidental = slate blue, mild = amber, moderate = orange, severe = red.

## 3D coordinate frame (meshes and `anchor.point`)
Units are millimetres. three.js axes: **+Y = up (toward the head), +Z = toward the viewer
(patient's front), +X = patient's LEFT** (so a camera on +Z looking at the patient's front sees the
patient's left on screen right). Each region's meshes are in that region's own scanner frame; the
regions are usually scanned as separate series, so their absolute positions are unrelated. The app places
each region group into the body model by its `anchor` (fit the group's bounding box to that
anatomical slot of the procedural spine, uniform scale, keep orientation). `anchor.point` of a
finding is in the same frame as its region's meshes, so it moves with the group.

Meshes are `.glb` (binary glTF), one mesh per file, no materials the app must keep (the app
applies its own materials by `kind` and `status`).

## Image stacks (the 2D scan viewer)
`data/stacks/index.json`:
```json
[ { "id": "L_SAG_T2", "label": "Lower back, side view (T2)", "region": "lumbar",
    "plane": "sagittal | axial", "sequence": "T1 | T2 | STIR",
    "count": 15, "width": 512, "height": 512, "pixel_mm": [0.5, 0.5],
    "path": "data/stacks/L_SAG_T2/{i}.jpg", "default_slice": 6,
    "orientation": { "left": "Front", "right": "Back", "top": "Head" } } ]
```
`{i}` is the zero-padded two-digit slice index (`00.jpg`, `01.jpg` ...). Marks in `findings[].views`
use coordinates normalised 0..1 of the image width and height.
