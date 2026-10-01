# CT (planned)

Nothing is built here yet. This is the plan and the data that decides it. Steps follow
[../README.md](../README.md) ("How a new scan type gets a trained model").

## How CT fits

A CT is a 3D stack like an MRI, so most of the spine MRI path carries over: `core/dicom/` turns the
disc into NIfTI volumes, a segmentation model finds the structures (the job TotalSpineSeg does for
spine MRI), measurements and 3D meshes follow, and the viewer shows them. Two differences: CT
brightness is in fixed Hounsfield units (so bone and soft-tissue windows are standard), and the
segmenter must be one we may use commercially.

## First questions worth training for

1. **A segmenter for the 3D view**, trained on the TotalSegmentator dataset (below), so every CT gets
   its bones and organs labelled the way spine MRI gets its vertebrae and discs.
2. **Spine CT**: vertebra labelling and shape, trained on VerSe, reusing the spine reading and 3D
   view.
3. **Lung nodules**, scored against LIDC-IDRI, where four thoracic radiologists read every scan, so
   the scorecard can compare with more than one expert.

## Candidate data and models (licences read at the source on 2026-10-01)

Commercially usable:

| Dataset or model | What it is | Expert labels | Licence |
|---|---|---|---|
| [TotalSegmentator dataset v2](https://zenodo.org/records/10047292) | 1,228 clinical CT scans | masks of 117 anatomical structures | CC BY 4.0 |
| [VerSe](https://github.com/anjany/verse) ('19 + '20) | 374 spine CT scans, 355 patients | vertebra masks and centre points | CC BY-SA 4.0 |
| [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | chest CT, 1,010 patients | lung nodule marks and ratings by 4 thoracic radiologists | CC BY 3.0 (plus TCIA's usage policy) |
| [AMOS](https://zenodo.org/records/7155725) | 500 CT + 100 MRI abdominal scans | masks of 15 organs | CC BY 4.0 |
| [MedSAM](https://github.com/bowang-lab/MedSAM) weights | general medical segmentation model | none (a model) | Apache-2.0 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) weights | CT segmentation models | none (models) | Apache-2.0 for the open tasks; some tasks need a separate licence. Check per task |

Not usable in a paid product: CTSpine1K and KiTS23 (CC BY-NC-SA), NVIDIA's NV-Segment-CTMR weights
(non-commercial licence); RSNA's cervical spine fracture and intracranial haemorrhage sets, CQ500,
DeepLesion, AbdomenCT-1K and the MONAI bundles (per-bundle licences) have not been confirmed.

Before any download: add the row to `LICENSING.md`.
