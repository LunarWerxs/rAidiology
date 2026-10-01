# Changelog

## [disc-grader-v1] - 2026-10-01

First public release, under the PolyForm Noncommercial License 1.0.0.

- **The web app:** a plain-English, 3D walkthrough of a spine MRI reading, with made-up sample data so
  it opens without a scan. On phones the results panel is a bottom sheet you can drag, tuck away or
  open fully.
- **The pipeline:** DICOM to NIfTI, TotalSpineSeg labels, per-level measurements, 3D meshes and image
  stacks for the app.
- **The SPIDER scorecard:** every method scored against one expert's grades on public scans, with 95%
  confidence intervals, and recomputed independently (`pipeline/eval/SCORECARD.md`).
- **The trained disc grader** (attached as `disc-grader-v1.zip`): on 44 held-out SPIDER patients it gave
  the expert's Pfirrmann grade 61% of the time (95% CI 52-68%), against 44% for the measurement rules.
  Unzip it into `../Models/disc-grader-v1/`, beside the repository. What it is and is not:
  `pipeline/models/disc-grader.md`.

Not a medical device, and for noncommercial use only (`LICENSING.md`). The disc finder it relies on,
TotalSpineSeg, has trained weights whose commercial terms are not settled.
