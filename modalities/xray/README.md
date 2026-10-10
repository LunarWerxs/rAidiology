# X-ray: is a fracture visible?

An X-ray is one 2D image, so the shared parts carry over: `core/dicom/series.py` finds the images,
`core/reading/` renders them for AI readers, `site/` shows them. No 3D segmentation step is needed.

The scored question, fixed before any model was trained:

> **Does this radiograph show a fracture?** One yes or no per image.

That is the question a person actually asks on the way out of A&E, and it is the one the two
commercially usable datasets below have a radiologist's answer for.

Steps follow [../README.md](../README.md) ("How a new scan type gets a trained model"). The numbers
live in [eval/SCORECARD.md](eval/SCORECARD.md); the metric contract that defines every one of them is
[eval/METRICS_SPEC.md](eval/METRICS_SPEC.md). Not a medical device (`LICENSING.md`).

## The data

Both are CC BY 4.0, both have a row in `LICENSING.md` written before the download, and both stay
outside the repository, in `../Datasets/public/` (`python fetch_data.py all` puts them there, checks
figshare's md5 for every file, and refuses to run if figshare's licence field ever stops saying
CC BY 4.0).

| | GRAZPEDWRI-DX | FracAtlas |
|---|---|---|
| What | paediatric wrist trauma radiographs | mixed musculoskeletal radiographs (hand, leg, hip, shoulder) |
| Images | 20,327 | 4,083 |
| Fracture visible | 13,550 (67%) | 719 (18%) |
| Labelled by | paediatric radiologists, with boxes per finding | the dataset's authors, with boxes and masks |
| Patient identifier published | **yes**: 6,091 patients, 1 to 30 images each | **no** |
| Extra facts per image | age, sex, side, projection, cast, metal, osteopenia, "diagnosis uncertain", scanner maker, pixel spacing | body part, view, hardware, multiscan, fracture count |
| Role here | the primary set: the split is by patient and the ship decision rests on it | a second, independent test set: different hospitals, different body parts |

**FracAtlas publishes no patient identifier**, so its split can only be by image, and two radiographs
of the same person could fall on both sides of it. That is why the ship decision is made on
GRAZPEDWRI-DX, where the split is by patient, and FracAtlas is reported beside it as an external
check with that limitation stated on the scorecard.

### Two things that could be learned instead of the fracture

Both are reported, not hidden, because a number that quietly includes them is not an honest number:

- **A cast.** 5,776 GRAZPEDWRI-DX images show a plaster cast, which almost always means a fracture
  somebody already diagnosed. A model can learn "cast" and score well without ever seeing a break.
- **A follow-up.** Only 10,861 of the images are initial examinations; the rest are check-ups of a
  known fracture.
- **A label the radiologist was unsure of.** 537 images are flagged "diagnosis uncertain". None are
  thrown away: throwing away the hard cases is how a scorecard flatters a model.

So every metric is also reported on three pre-declared subsets: **initial examinations only**,
**images with no cast**, and **images whose diagnosis is not flagged uncertain**.

## Order

Data and tools live outside the repository: the datasets in `../Datasets/public/`, trained weights in
`../Models/<name>-v<N>/`. Working files go to `work/eval/xray/` (git-ignored). Python is the
environment in `modalities/spine_mri/README.md` plus pandas and scikit-learn. Heavy steps need a GPU.

| step | command | needs |
|---|---|---|
| download the datasets | `python fetch_data.py all` | 17 GB of disk |
| split (committed) | `python eval/split.py` | the two dataset indexes |
| prepare images once into a memory-mapped cache | `python eval/prepare.py all` | the images |
| baselines: prevalence, metadata-only, classical image features | `python eval/baselines.py dev` | the cache |
| AI reader bundles, then its development score | `python eval/reader.py dev --backend agents` writes bundles; Claude Opus sub-agents answer them; `python eval/reader.py ingest <answers.json>` | no API key |
| train the detector (5 folds) and predict out of fold | `python eval/detector.py train --version 1`, `python eval/detector.py predict dev --version 1` | GPU |
| development scorecard | `python eval/score.py` | all of the above |
| independent recomputation | `python eval/recompute.py > eval/recompute.log` | a separate agent session that has not seen `score.py` |
| **final evaluation (once)** | `python eval/final_eval.py --reader agents`, answer the test bundles, ingest, `python eval/final_eval.py --finish` | everything above |
| model card | `python eval/model_card.py` | a trained detector |

## What is measured against what

The baselines are scored **before** the detector is trained, on development folds only.

| | what it is | why it is here |
|---|---|---|
| `prevalence` | always answers with the development majority | the floor: what guessing scores |
| `meta` | logistic regression on age, sex, side, projection and scanner maker only, no pixels | the **no-image floor**. A detector that does not beat this has learned nothing from the image |
| `features` | logistic regression on classical image statistics (intensity, gradient and texture summaries over a fixed grid), no deep learning and no pretrained weights | the simple-measurement baseline, the X-ray analogue of the spine measurement rules |
| `reader` | Claude Opus 5.5 reading the image, as sub-agents with no API key | what a general model scores with no training |
| `cnn` | a small convolutional network trained from scratch on development folds | the candidate |

## What it scored (held-out test, opened once on 2026-10-01)

| GRAZPEDWRI-DX test, 4,170 images, 1,218 patients | AUROC |
|---|---|
| **Detector (ships)** | **0.976** (95% CI 0.970-0.981) |
| Classical image features (the reference) | 0.838 (0.820-0.853) |
| Claude Opus 5.5 reading the image (167 images) | 0.690 (0.608-0.769) |
| Metadata only, no pixels | 0.686 (0.653-0.715) |
| Prevalence | 0.500 |

The detector's AUROC minus the reference's, on the images both scored, is **+0.138 (95% CI 0.125 to
0.154)**: above zero, so it ships. At its development-fitted threshold it catches 92.7% of the
fractures (95% CI 91.2-94.1) and calls 94.4% of the unbroken ones correctly (92.4-96.1), kappa 0.849.

It is not leaning on the plaster. Where the reference baseline falls away, the gap widens: **+0.239**
on initial examinations only and **+0.216** on images with no cast. On the second, independent test
set (FracAtlas, different hospitals and body parts) it scores 0.893 (0.859-0.924) against 0.750.

Every number was recomputed from `eval/METRICS_SPEC.md` alone by an agent that never saw `score.py`,
in a clean room holding only the contract, the card and the data: 2,978 cells, none differing
(`eval/recompute.log`).

### Where it runs

`python eval/detector.py image <file> [--profile grazpedwri|fracatlas]` scores one radiograph, DICOM
or ordinary image, with the exact five-fold ensemble the scorecard measured, through the same
windowing and letterboxing. That is what a server-side worker calls. It does **not** run in the
visitor's browser on `site/open.html`: the measured artefact is five models and about 47 MB, and
shipping a smaller or quantised one would mean shipping something whose held-out accuracy nobody has
measured, which the test set can no longer answer because it is closed.

## The ship rule, declared before the test set is opened

- The primary metric is **AUROC** on all GRAZPEDWRI-DX test images, with a patient-level bootstrap.
- The reference is the best of `prevalence`, `meta`, `features` and `reader` by development AUROC,
  chosen and written down before the test split is opened.
- The detector ships into the app only if the 95% confidence interval of its AUROC **minus** the
  reference's, on the images both scored, lies above zero.
- Sensitivity, specificity, balanced accuracy and Cohen's kappa are reported too, at a threshold
  fitted on development folds only, but they do not decide shipping.
- The test split is opened once, by `eval/final_eval.py`, which refuses to run twice.

## The answer depends a little on the machine

Measured on 2026-10-01 over twenty held-out radiographs (`tools/device_agreement.py`, in the
ledger). On one machine a reading is now exactly repeatable: the same image scored twice gives the
same number, on a processor and on a graphics card alike, because `detector.steady_inference()`
pins cuDNN's algorithm choice. Before it, a card could disagree with itself.

Across kinds of machine it is not zero and cannot be: a processor and a graphics card differ by up
to **0.0090** on a score from 0 to 1, which is different convolution kernels adding up in a
different order rather than a precision setting. **0.14%** of the held-out wrist test (6 of 4,170)
and 1.84% of the mixed-body-part test sit that close to the threshold, so that is roughly how often
the kind of machine could change the yes or no.

Which is one of the reasons the shipped route reads on the processor, the same thing a laptop, a
desktop and a rented container all do. A graphics-card server would need its threshold recalibrated
on that hardware rather than inheriting this one. The arithmetic is in
[docs/COST_MODEL.md](../../docs/COST_MODEL.md), which also measures what a reading costs to serve.

## Later, not now

Already-licence-checked ideas that are not part of this first model: pneumonia on a chest X-ray
(RSNA labels are usable but the underlying NIH images' terms are not confirmed), and spine X-rays
(no commercially usable expert-labelled set found: VinDr-SpineXR is restricted). The full
not-usable list is in `LICENSING.md`.
