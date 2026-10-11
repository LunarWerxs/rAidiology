# Model card: rAidiology CT segmenter v121

**Not a medical device. Not for diagnosis or treatment decisions.**

## What it does

It outlines organs and bones in a CT scan. You give it a 3D CT scan with 1.5 mm spacing. It gives back one label for every voxel (a voxel is a 3D pixel). The label is one of 117 organs and bones, such as the liver, spleen, aorta, ribs, vertebrae and hips, or "nothing".

It works in two steps. A coarser model (version 116) reads the scan at 3 mm and finds roughly where everything is. This model (version 121) then reads the scan at 1.5 mm, with that first read as an extra input, and sharpens the edges. It says where a structure is. It does not say whether the structure is healthy, and it does not look for disease.

### What it is not for

- **Diagnosis.** It does not tell you what is wrong with anyone.
- **Triage or screening decisions.** Do not use it to decide which scans are read first, who is called back or who is sent home.
- **Treatment planning.** Do not use its outlines or volumes to plan surgery, radiotherapy or any other treatment. Its outlines are wrong in places, and the tables below show where.
- **Children.** No score is split by age, so nobody has checked children. Adults have not been checked as a separate group either.
- **Other scan types.** It reads CT only. It does not read MRI, X-ray, PET or ultrasound, and it does not read a single 2D slice.
- **Other spacings.** It was scored on scans at 1.5 mm only. A scan at any other spacing has no score behind it.
- **Other scanners, hospitals and protocols.** Most of the test scans come from one institute and one scanner maker. The group tables below are the only evidence about anything else, and the groups are small.
- **Structures outside the 117.** It cannot outline anything that is not on its list.

## How well it does

**Dice** is the overlap between the model's outline and the answer key. 1.0 means they match exactly. 0 means they do not touch. It is worked out for each structure over the scans that contain it, then averaged over structures, each counting once. A structure found in fewer than 10 scans of the split is left out of that average. The **95% interval** is the range the true score probably falls in, given only 89 test scans.

The answer key is the dataset's own masks. They were drawn with the help of a model and approved by people. They are not outlines drawn by hand by a panel of radiologists, so this model has not been compared with a radiologist. A model that is better than the key would be marked wrong where it is right.

| Measure | Ours (95% interval) | Test set | Published score to beat, and who holds it | Same pictures? |
|---|---|---|---|---|
| Mean Dice over the 115 structures found in at least 10 test scans, read at 1.5 mm (the two kidney cysts, in 4 and 3 scans, are reported but left out of the mean) | **0.8668** (0.8538 to 0.8781) | Held-out test, 89 scans. This is the fourth time these scans were read (2026-10-08). | 0.943, TotalSegmentator's 1.5 mm model (Wasserthal et al., Radiology: Artificial Intelligence 2023) | Same dataset family, but not the same list of scans or structures. Their paper scored 104 structures on its own test scans. **We are behind by 0.08.** |
| The same read on the development scans, each read by models that never saw it | 0.8156 (0.8116 to 0.8194), read with the same size floor as the test | 1,139 development scans | No published score | Not applicable |
| Gain from the 1.5 mm step over the 3 mm read it is compared with (version 116 with version 115) | +0.0332 (+0.0307 to +0.0357) on test, from 0.8336 to 0.8668. +0.0148 (+0.0136 to +0.0160) on development, from 0.8008 to 0.8156, read with the same size floor. The scorecard's development gate section prints +0.0153 (+0.0140 to +0.0168), from 0.8017 to 0.8171. The sources do not say why the two differ. This card uses the first, because it is the same read as the other development rows. | Test, 89 scans, and development, 1,139 scans | No published score | Not applicable |
| Mean Dice minus the simple bone-window method | +0.6892 (+0.6728 to +0.7062). Bone-window scores 0.1776. | Test, 89 scans | No published score. Bone-window is a simple method with no learning, used as a floor. | Not applicable |
| How often it outlines a structure that is not in the scan | 0.0410 (0.0301 to 0.0543), about 4 times in 100 | Test, 89 scans | No published score | Not applicable |
| The same, on development scans | 0.1311 (0.1242 to 0.1384), about 13 times in 100 | 1,139 development scans | No published score | Not applicable |

For reference, the 3 mm read it is compared with scores 0.8336 (0.8214 to 0.8442) on the same test scans, against 0.84 for TotalSegmentator's 3 mm model. At 3 mm the two are level. At 1.5 mm we are behind.

The test scans have now been read four times. Each later read was allowed only after the new model beat the old one on development scans, and no test number was used to choose anything. Even so, treat the test score as one that has been looked at more than once.

### By group (test scans, mean Dice)

| Group | Scans | Mean Dice (95% interval) |
|---|---:|---|
| All | 89 | 0.8668 (0.8538 to 0.8781) |
| Abdomen scans | 47 | 0.8802 (0.8680 to 0.8914) |
| Thorax scans | 32 | 0.8883 (0.8759 to 0.8986) |
| Head or neck scans | 19 | 0.8766 (0.8554 to 0.8950) |
| Siemens scanners | 58 | 0.8709 (0.8527 to 0.8858) |
| Other scanner makers | 31 | 0.8642 (0.8449 to 0.8815) |
| The dataset's main institute | 55 | 0.8573 (0.8362 to 0.8737) |
| All other institutes | 34 | 0.8866 (0.8733 to 0.8981) |
| Scans with marked pathology | 73 | 0.8663 (0.8514 to 0.8793) |

A scan can sit in more than one group, so the rows overlap.

### By structure (test scans, mean Dice over the scans that contain the structure)

- **Weakest.** Both kidney cysts score 0.0000 (left cyst in 4 scans, right cyst in 3). Next are vertebra C6 at 0.6699 (34 scans), the gallbladder at 0.7079 (52), the portal vein and splenic vein together at 0.7189 (65), the left adrenal gland at 0.7375 (65) and the duodenum at 0.7375 (65).
- **Strongest.** The left hip at 0.9691 (58 scans), the spleen at 0.9656 (65) and the right hip at 0.9534 (59).

The full scorecard, with every structure for version 1 and the later looks at the test, is in [modalities/ct/eval/SCORECARD.md](../../modalities/ct/eval/SCORECARD.md). How each number is worked out is in [modalities/ct/eval/METRICS_SPEC.md](../../modalities/ct/eval/METRICS_SPEC.md).

## Data it learned from

One dataset. Nothing else.

| Dataset | What it was used for | Licence | Commercial use allowed? |
|---|---|---|---|
| TotalSegmentator dataset v2.0.1: 1,228 clinical CT scans with masks of 117 structures | **Training, labels and calibration:** 1,139 "development" scans, split into five folds. Each fold's model learned from 911 or 912 scans and the rest were held out for scoring. The masks are the labels. The size floor (see Limitations) was worked out from the mask sizes in the development scans only. **Testing only:** the other 89 scans, the dataset authors' own test split. They were never used for training or for any choice. | CC BY 4.0 | Yes |

**Credit this licence asks for.** Cite the paper and the dataset, and link the licence (<https://creativecommons.org/licenses/by/4.0/>):

> Wasserthal, Breit, Meyer, Pradella, Hinck, Sauter, Heye, Boll, Cyriac, Yang, Bach, Segeroth, 'TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images', Radiology: Artificial Intelligence 5(5) (2023), doi:10.1148/ryai.230024; dataset v2.0.1 doi:10.5281/zenodo.10047292, CC BY 4.0

The dataset is at <https://zenodo.org/records/10047292>.

What was not used:

- **No pretrained weights of any kind.** No ImageNet and no model zoo. Version 116 was trained from scratch on the dataset above. Version 121 starts from version 116's weights for the same fold, so it also has no other source.
- **TotalSegmentator's own released model weights were not used.**
- **No other dataset** was used for training, pretraining, labels or calibration.

The dataset does not publish a patient identifier. The scan is the unit here, not the patient, so two scans of one person could fall on both sides of a split.

## Limitations and known failure modes

- **It is behind the published model at this detail.** 0.8668 against 0.943 at 1.5 mm, a gap of 0.08.
- **It draws structures that are not there.** About 4 times in 100 on the test scans and about 13 times in 100 on the development scans. The sources do not say why the two differ. Dice does not show this, which is why it has its own row.
- **Some structures are unreliable.** Kidney cysts scored 0.0000 on the test scans. Vertebra C6, the gallbladder, the portal and splenic veins, the adrenal glands and the duodenum are the next weakest.
- **One dataset.** Most test scans come from one institute (55 of 89) and one scanner maker (58 of 89). The score is lower for the main institute (0.8573) than for the others (0.8866). Other hospitals and scanners are mostly unchecked.
- **Scans at other spacings are unscored.** Only 1.5 mm was measured.
- **A small structure can be removed.** The reading drops any structure it draws smaller than 1 in 100 of that structure's true sizes in the development scans. This cuts false drawings, and it can also remove a real structure that is much smaller than usual.
- **The second step depends on the first.** The 1.5 mm model was trained with the 3 mm read as an input. It needs that read, so it needs both sets of weights.
- **A perfect score is under 1.** The dataset's masks overlap in 0.15% of marked voxels, and this model gives each voxel one label.
- **No comparison with a radiologist.** The answer key is not a set of outlines drawn by hand by a panel of doctors. Do not read the Dice as agreement with a radiologist.
- **Intervals treat each scan as independent.** Because the scan is the unit, they may be narrower than they should be.

## Licence

The weights and the code are under the [PolyForm Noncommercial License 1.0.0](../../LICENSE). This line travels with every copy:

> Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

You may use them for noncommercial purposes under that licence. Commercial use is available under the rAidiology Commercial Licence ([COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)), with a free 60-day evaluation (500 studies, internal testing only, no customers, no patients' care), priced on the rAidiology site's pricing page. Not for diagnosis or treatment. Not cleared by the FDA or any regulator.

The weights are not stored in this repository. Downloads are gated: you accept the licence before you download. Official weights are trained and published by LunarWerx only.

The training dataset is CC BY 4.0, which asks for the credit given above wherever the model or its outputs are described.

## How to run it

The weights are ten files in two sets. Version 121 is five files, `fold0.pt` to `fold4.pt`, 124,959,555 bytes each. Version 116, the 3 mm first step it needs, is five files with the same names, 124,896,325 bytes each. Each set comes with its `meta.json`, which holds the checksum of every fold file. The `spacing_mm` field in version 121's `meta.json` reads 3.0. The model reads at 1.5 mm, as this card says. The read averages the five fold models.

Put each set in its own folder under the work folder (`work/` by default, or the folder `RAIDIOLOGY_WORK` points to):

    work/eval/ct/segmenter-v116/model/
    work/eval/ct/segmenter-v121/model/

Then read the dataset's development scans with the cascade:

    python modalities/ct/fetch_data.py totalsegmentator
    python modalities/ct/eval/prepare.py dev
    python modalities/ct/eval/prepare.py hi15
    python modalities/ct/eval/segmenter.py coarse --version 116
    python modalities/ct/eval/segmenter.py predict dev --version 121

The last line writes the out-of-fold rows that [modalities/ct/eval/score.py](../../modalities/ct/eval/score.py) reads.

**Reading one scan of your own with version 121 is not supported yet.** The `scan` command stops with "a 1.5 mm version reads prepared scans only for now". For your own file, `python modalities/ct/eval/segmenter.py scan <ct.nii.gz> --out labels.nii.gz --version 1` reads with version 1 instead, a 3 mm model with a lower score (0.7667, 95% interval 0.7517 to 0.7811, on the same 89 test scans, first look). Its card is [modalities/ct/models/segmenter.md](../../modalities/ct/models/segmenter.md).

---

Not a medical device. These are measurements on a public research dataset, not a claim about anybody's scan.
