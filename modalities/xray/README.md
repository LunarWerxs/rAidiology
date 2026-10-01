# X-ray (planned)

Nothing is built here yet. This is the plan and the data that decides it. Steps follow
[../README.md](../README.md) ("How a new scan type gets a trained model").

## First questions worth training for

An X-ray is one 2D image, so the shared parts mostly carry over: `core/dicom/series.py` finds the
images, `core/reading/` renders them for AI readers, the viewer shows them. No 3D segmentation step
is needed to start; a 2D model trains in hours on the development machine's GPU.

1. **Is a bone broken, and where?** (limbs and wrists). This has the best commercially usable data
   (below) and a clear expert answer.
2. **Pneumonia on a chest X-ray.** Usable labels exist (RSNA), but check the underlying NIH images'
   terms first.
3. **Spine X-rays** (alignment, slipped or collapsed vertebrae) would reuse the spine know-how, but no
   commercially usable expert-labelled set was found yet (VinDr-SpineXR is restricted).

## Candidate data (licences read at the source on 2026-10-01)

Commercially usable, which is the gate for anything in a paid product:

| Dataset | What it is | Expert labels | Licence |
|---|---|---|---|
| [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) | about 4,000 musculoskeletal radiographs | fracture yes/no, location boxes, segmentation | CC BY 4.0 |
| [GRAZPEDWRI-DX](https://doi.org/10.6084/m9.figshare.14825193) | 20,327 paediatric wrist trauma radiographs | fractures and other findings marked by paediatric radiologists | CC BY 4.0 |
| [RSNA Pneumonia Detection](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) | chest radiographs from NIH ChestX-ray14 | pneumonia boxes by radiologists | RSNA terms allow "commercial or non-commercial purposes" with attribution; the NIH images' own terms are not yet confirmed |

Not usable in a paid product (research-only, credentialed or not confirmed): MIMIC-CXR and VinDr-CXR
(PhysioNet credentialed licence), VinDr-SpineXR (PhysioNet restricted licence), PadChest (research
use agreement), LERA and RSNA Bone Age (non-commercial terms); CheXpert, MURA, NIH ChestX-ray14,
the OAI knee X-rays, BUU-LSPINE and the Shenzhen/Montgomery TB sets have not been confirmed.

Before any download: add the row to `LICENSING.md`.
