# Changelog

## [Unreleased]

- **The sample scan names what you point at.** "Try a sample scan" opens a real CT with every organ,
  bone and blood vessel outlined; pointing at an outline, or tapping it on a phone, names it in plain
  words beside the point ("liver: the large organ that cleans the blood and makes bile"). The outlines
  are a findings file written by the MCP server's new `read_ct`, which reads a hospital CT series on
  the person's own machine and answers with each structure, its size and its own test accuracy. The
  page still never says whether anything is wrong. Five simulated visitors with no disc of their own
  gave it 4 stars each, against 2.67 the round before; the list of names folds away, brightness and
  contrast are one click away, and on a phone Previous and Next step through the pictures.
- **A reading now leaves something behind, and the viewer opens it.** When somebody's own AI reads
  their scan through rAidiology's MCP server, `write_findings` saves a file of about a kilobyte
  beside the scan. They drop it onto `/open.html` along with the scan itself and see their own
  pictures with the flagged images marked, the part of the picture that moved the score shaded, and
  the model's held-out accuracy and limits printed underneath. It holds no name, patient number or
  date of birth: the series and each image are referred to by a one-way fingerprint that the browser
  recomputes from the scan it is already holding, so the matching happens on the visitor's device
  and the file on its own identifies nobody. The shading is exact rather than estimated, because the
  network averages its feature map before a single linear layer, so the per-cell value recombines to
  the score; it is labelled a place to look and not a location, because location was never scored.
  Format in `docs/FINDINGS_FILE.md`; `python mcp/server.py --selftest` drives the whole chain on a
  made-up series and checks the two languages still agree on the fingerprint.
- **What a reading costs, with a stopwatch rather than a guess.** `docs/COST_MODEL.md`: an
  eight-image wrist series is 2.45 seconds of ordinary processor time and about a hundredth of a
  cent to rent, a graphics card costs roughly three times as much to go 2.7 times faster, and more
  processor threads bought nothing at all. So a paid server, if one is ever offered, is not blocked
  on the compute bill; it is decided by the monthly floor. Two improvements came out of measuring
  it: the five fold weights are loaded once per process instead of on every question, and images
  are decoded four at a time, measured at 1.91x.
- **A reading no longer depends on the machine's mood.** cuDNN's algorithm choice is pinned, which
  took a graphics card from disagreeing with its own previous answer to exactly repeatable, at no
  cost in speed. The gap that remains, between a processor and a graphics card, is 0.0090 on a
  score from 0 to 1 and is written down with how often it could change an answer: 6 of 4,170
  held-out wrist images sit that close to the line. `tools/device_agreement.py` is the measurement.
- **CT, step four.** The held-out test is the dataset authors' own 89 scans, kept rather than
  re-cut, so nobody here chose it after seeing a result; their other 1,139 are development in five
  folds. `METRICS_SPEC.md` fixes the contract before anything is trained: a method hands in three
  voxel counts per scan and structure, so the scorer never opens a scan, and the two questions a
  segmentation metric must not mix are kept apart - the overlap on structures that are there, and
  how often a method invents one that is not. Four baselines are built and scored first, and the
  number a trained segmenter has to beat is **0.1847 mean Dice (0.1798 to 0.1894)**, set by a
  Hounsfield threshold inside a learned region. A second implementation of the scorer, written from
  the contract by somebody told not to read the first, disagreed on its opening run and was right:
  a stratum filter lower-cased the column but not the value it compared against, so one institute
  stratum matched nothing, disappeared from the scorecard, and its complement quietly became every
  scan while printing a plausible number. Nothing crashed. After that and three smaller corrections
  the two agree on all 144 cells.

- **X-ray fracture detection, and it ships.** A second scan type that answers one question, "does
  this radiograph show a fracture?", on two commercially usable datasets: GRAZPEDWRI-DX (20,327
  paediatric wrist images, split by patient) and FracAtlas (4,083 mixed musculoskeletal images, a
  second test set). On the held-out test, opened once, the detector scores AUROC 0.976
  (95% CI 0.970-0.981) against the best baseline's 0.838, a difference of +0.138 (0.125 to 0.154);
  it catches 92.7% of fractures and calls 94.4% of unbroken images correctly. The gap widens on
  initial examinations only (+0.239) and with no cast (+0.216), so it is reading the bone rather
  than the plaster. Trained from scratch, no pretrained weights, both datasets CC BY 4.0.
- **A public landing page, and "open your own scan".** `/` explains rAidiology to anyone, and
  `/open.html` opens the folder or ZIP from a hospital CD entirely in the visitor's own browser:
  the DICOM is parsed in JavaScript, the series are sorted along the scan direction, and there is
  brightness, contrast, zoom and a 1 cm scale bar. Nothing is uploaded, there is no account and no
  third-party code, and the name, ID and date-of-birth tags are never read. The one private reading
  moved to `/report.html` and is the only thing the password now covers.
- **A run ledger.** Every run that produces a number or takes time is recorded, so a score that
  moved or a step that got slower is a query rather than a memory. Its first result: the image
  cache now decodes in worker processes instead of threads, 4.5 times faster with identical output.

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
