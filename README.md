# rAidiology

**Not a medical device.** rAidiology is not cleared or approved by the FDA or any other regulator. It
is for education and a second look, never for diagnosis or treatment decisions; always discuss
your results with a doctor ([LICENSING.md](LICENSING.md#not-a-medical-device)).

rAidiology turns a spine MRI reading into a calm, plain-English web page. It shows a status
("Looks OK", "Worth watching" or "See a doctor"), a short summary, and each finding explained in
everyday words. Findings sit on an interactive 3D model: a glass human body with the spine inside
it, where the scanned regions glow in colours that match how much each finding matters. You can
also scroll through the scan images with the findings marked on them, see every spinal level on a
colour-coded diagram, and look up words and questions to ask your doctor.

It is a static site: no build step, no npm. The page is `site/index.html` with plain ES modules
and CSS; three.js is loaded from jsDelivr (pinned to 0.170.0).

Spine MRI is the first scan type; X-ray and CT are planned, each in its own folder.

## Repository layout

| Folder | What it holds |
|---|---|
| `site/` | the viewer (static web app) and its data format, `site/data/CONTRACT.md` |
| `core/` | parts every scan type shares: DICOM handling (`core/dicom/`), image tools for AI readers (`core/reading/`), and `core/paths.py`, which says where data, weights and tools live |
| `modalities/` | one folder per scan type, each with its pipeline, reader brief, scorecard (`eval/`) and model cards. How a scan type plugs in: [modalities/README.md](modalities/README.md) |
| `tools/` | repository tooling, such as the builder of the reference skeleton the viewer shows |

Datasets, trained weights and installed tools sit beside the repository (`../Datasets`,
`../Models`, `../_tools`), never in it; working files go to the git-ignored `work/`.

## Run locally

```sh
cd site
python -m http.server 8000
```

Then open http://localhost:8000. Opening `index.html` straight from disk does not work, because
browsers block module and data loading from `file://`.

## Data

Everything the page shows comes from `site/data/`; the format is in
[`site/data/CONTRACT.md`](site/data/CONTRACT.md).

- `data/report.json`: the real reading. `make_report.py` builds it from `reading.json`, which AI
  readers and a person write; that step is not a script (see `modalities/spine_mri/README.md`, "The reading step").
- `data/report.sample.json`: made-up demo data. The page uses it only when `report.json` is
  missing, and then shows a "Sample data" badge.
- `data/meshes/...`: one `.glb` per vertebra/disc/cord mesh, listed in `report.json`.
- `data/stacks/index.json` plus `data/stacks/<id>/00.jpg ...`: the image series for the scan viewer.

Anything optional that is missing shows a friendly empty state instead.

## Deploy

`site/` is a static site; any static host serves it. A deployment that shows a real scan needs
access control in front of it (for example Cloudflare Access, or a password gate as a Pages
Function); never put a real scan on a public host. All URLs are relative, so the site also works
under a `/<repo>/` sub-path.

## Making the data (pipeline)

`modalities/spine_mri/` turns a patient MRI disc into everything in `site/data/`: DICOM to NIfTI, the
TotalSpineSeg model (code LGPL-3.0; its trained weights are permission pending, see LICENSING.md) (vertebrae, discs, cord, canal), per-level measurements, smooth
3D meshes and de-identified image stacks. The written reading comes from an AI reading panel with
adversarial verification. See [`modalities/spine_mri/README.md`](modalities/spine_mri/README.md).

For local preview with caching off (so edits show on reload): `python serve.py`.

## Privacy

The page shows only `patient.label`, `age` and `sex`. Names, birth dates, accession numbers and
institutions are dropped when the report is loaded and never shown.

This repository holds no patient data. `site/data/` has only the made-up `report.sample.json`, the
format description and the reference skeleton, so the page opens on the sample with a "Sample
data" badge. Running the pipeline on your own disc writes your `report.json`, meshes and image
stacks; `.gitignore` keeps them out of commits. Keep them out of any public repository, because
git history keeps every committed file.

Not a diagnosis. Always discuss results with a doctor.

## Licence

PolyForm Noncommercial 1.0.0, verbatim in [LICENSE](LICENSE). It is free for personal use and for
charitable, educational, public-research, public-health and government organisations. Any other
use with a commercial application, such as private clinics, insurers and legal work, needs a paid
licence. Who needs a licence and the third-party licences: [LICENSING.md](LICENSING.md).

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

## Where it is going

rAidiology is becoming a measured specialist: a scorecard on expert-graded public scans (SPIDER), a
disc grader trained from scratch, healthy-population references and a mistake log. On 44 held-out
SPIDER patients the disc grader gave the expert's Pfirrmann grade 61% of the time (95% CI 52-68%),
against 44% for the measurement rules. Every number, method and caveat is in
[modalities/spine_mri/eval/SCORECARD.md](modalities/spine_mri/eval/SCORECARD.md), and the grader's
[model card](modalities/spine_mri/models/disc-grader.md) says what it is and is not. Its trained weights are a
download on this repository's Releases page (`disc-grader-v1`; unzip into `../Models/disc-grader-v1/`).
Development used private data that is not part of this repository. Scored runs used an earlier
wording of `modalities/spine_mri/READER_BRIEF.md`.

Next: X-ray and CT ([modalities/](modalities/README.md)), and a landing page where anyone can open
their own scan in the browser.

## Built on, and what already exists

rAidiology's viewer, report format and reading workflow were written for this project; no existing
app was copied. It stands on these open-source pieces:

- [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) (NeuroPoly; code LGPL-3.0, trained weights permission pending): labels every vertebra, disc, the cord and the canal (nnU-Net).
- [SpineReport](https://github.com/ivadomed/SpineReport) (NeuroPoly, LGPL-3.0): 3D canal, disc and foramen morphometrics, used as an independent measurement.
- [BodyParts3D](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/) (DBCLS, CC BY 4.0): the anatomical reference skeleton, modified (parts merged and simplified).
- [three.js](https://threejs.org/), pydicom, SimpleITK, nibabel, scikit-image, trimesh.

Closest existing things, none of which combine a patient-facing explanation, a 3D model of the
patient's own spine, a blind AI reading with adversarial checks, and a side-by-side with the
radiologist's report:

- [CoLumbo](https://blackfordanalysis.com/ai-portfolio-columbo-spine-copilot): FDA-cleared commercial AI that pre-reads lumbar MRI for radiologists (measurements, draft report).
- [SpineNet](http://zeus.robots.ox.ac.uk/spinenet/) (Oxford): research model for Pfirrmann grade, disc narrowing and canal stenosis.
- [SpineReport](https://arxiv.org/abs/2606.10021): research reports of 3D spine morphometrics against a control group, for clinicians.
- General web and desktop viewers such as [EPAM MRI Viewer](https://github.com/epam/mriviewer) and [OpenMRI](https://github.com/yairmcaudillo-cell/OpenMRI): viewing only, no interpretation.
