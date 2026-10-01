# Changelog

## [Unreleased]

- **New name:** SuprSkan is now rAidiology, everywhere in the app, the docs and the code.
- **Organised by scan type:** shared parts live in `core/` (DICOM handling, image tools for AI readers,
  and one file that says where data, weights and tools live); spine MRI lives in
  `modalities/spine_mri/`; X-ray and CT get their own folders with their plans. Every command moved
  with its script; the scorecard and its independent recomputation reproduce exactly.

## [disc-grader-v1] - 2026-10-01

First public release, under the PolyForm Noncommercial License 1.0.0.

- **The web app:** a plain-English, 3D walkthrough of a spine MRI reading, with made-up sample data so
  it opens without a scan. On phones the results panel is a bottom sheet you can drag, tuck away or
  open fully.
- **The pipeline:** DICOM to NIfTI, TotalSpineSeg labels, per-level measurements, 3D meshes and image
  stacks for the app.
- **The SPIDER scorecard:** every method scored against one expert's grades on public scans, with 95%
  confidence intervals, and recomputed independently (`modalities/spine_mri/eval/SCORECARD.md`).
- **The trained disc grader** (attached as `disc-grader-v1.zip`): on 44 held-out SPIDER patients it gave
  the expert's Pfirrmann grade 61% of the time (95% CI 52-68%), against 44% for the measurement rules.
  Unzip it into `../Models/disc-grader-v1/`, beside the repository. What it is and is not:
  `modalities/spine_mri/models/disc-grader.md`.

Not a medical device, and for noncommercial use only (`LICENSING.md`). The disc finder it relies on,
TotalSpineSeg, has trained weights whose commercial terms are not settled.
