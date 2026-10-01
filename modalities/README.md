# Scan types

rAidiology is organised by scan type. Each one has its own folder here and plugs into the same
shared parts:

| Shared part | What it gives every scan type |
|---|---|
| `core/dicom/` | finding the series on a disc (`series.py`), DICOM to NIfTI (`to_nifti.py`), per-slice geometry for marking images (`geometry.py`) |
| `core/reading/` | rendering images for AI readers and zooming into them (`render.py`, `crop.py`, `gridcrop.py`, `windows.py`) |
| `core/paths.py` | where the work folder, datasets, trained weights and tools live (each can be moved with an environment variable) |
| `site/` | the viewer. A scan type fills `site/data/report.json` in the format of `site/data/CONTRACT.md` |
| `cases/<patient>/` | one patient's private readings, comparison and mistake log. Never in a public copy |

A scan type's folder holds:

- `README.md`: what it reads and how to run it;
- its pipeline from a disc to `site/data/` (segmentation, measurements, 3D meshes, image stacks, the report);
- `READER_BRIEF.md`: how AI readers should look at this kind of image;
- `eval/`: its scorecard on expert-labelled public data, with a held-out test opened once and an
  independent recomputation;
- `models/`: a model card for anything it trains.

| Scan type | Status |
|---|---|
| [spine_mri](spine_mri/) | Working: AI reading, 3D view, and a disc grader that beat the measurement rules on a held-out test (`spine_mri/eval/SCORECARD.md`) |
| [xray](xray/) | Planned |
| [ct](ct/) | Planned |

## How a new scan type gets a trained model

The same six steps that produced the spine disc grader:

1. **Pick a question an expert can answer on the image**, narrow enough to score: "is this wrist
   broken", "how much has this vertebra collapsed", not "read this X-ray".
2. **Find expert-labelled scans you may use.** The licence decides: only data and weights that allow
   commercial use (add a row to `LICENSING.md` before downloading anything).
3. **Split by patient**: development folds for building, and a test set locked away and opened once.
4. **Measure the baselines first** (simple measurements, an AI reader with no training) on the
   development folds, with patient-level 95% confidence intervals.
5. **Train on the development folds only**, from scratch or from commercially licensed weights, and
   score it by cross-validation against the same baselines.
6. **Open the test once.** It ships into the app only if it beats the best baseline with the
   confidence interval of the difference above zero; an independent script recomputes every number.

Not a medical device, in any scan type (`LICENSING.md`).
