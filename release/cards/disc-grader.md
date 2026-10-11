# Model card: rAidiology disc grader v133 + v134 + v140

**Not a medical device. Not for diagnosis or treatment decisions.**

Written 2026-10-10. These are three versions of one recipe, trained with three different random seeds (0, 1 and 2). The scores below are for the three read together. An earlier version, v1, was published before them; its held-out test scores are given too.

## What it does

It grades the discs of the lower spine on a sagittal MRI. You give it a sagittal T2 scan, a sagittal T1 scan if you have one, and the disc labels that TotalSpineSeg draws on the T2 (they show it where each disc is). For each disc from T12-L1 down to L5-S1 it cuts out a small block of pictures and returns a Pfirrmann grade from 1 to 5, the chance of each grade, and the chance of seven findings: herniation, narrowing, bulging, a slipped vertebra (spondylolisthesis), change in the upper endplate, change in the lower endplate, and any Modic change. Each version is five small networks (about 4.7 MB each) whose answers are averaged.

A Pfirrmann grade says how worn a disc looks on MRI. Grade 1 is a healthy, plump, watery disc. Grade 5 is a disc worn flat and dried out.

### What it is not for

- Diagnosis, treatment decisions, triage (who is seen first) or screening. A person must read the scan.
- Children and teenagers. It was not tested on them.
- People without back pain. SPIDER, the main training set, holds people with low back pain only.
- Other pictures. It reads sagittal T2 (with sagittal T1 if present) of the lumbar spine. It does not read axial or coronal pictures, neck (cervical) discs, chest (thoracic) discs, or other body parts.
- Spines after surgery or with implants. It was not checked on them.
- Other scanners and hospitals, without your own check. Its skill dropped sharply when an earlier version met a hospital it had never seen (see Limitations).
- Any paid or commercial use. None is offered for this model (see Licence).

## How well it does

Every score here is agreement with a doctor's grades, or with the grades a dataset's authors made. It is not agreement with the truth. For scale: always guessing grade 3 (the commonest) gets 30 of 100 SPIDER development discs exactly right.

**These three versions have not been scored on held-out test patients.** They were chosen using development patients only. The held-out test patients were opened once for earlier versions (v1 on SPIDER, v103 on LSMA-PQR). Opening them again for these versions would be a second look, so it has not been done. The first table is therefore a development result, "out of fold": every patient is scored by networks that never saw that patient. The five networks you download have, between them, seen every development patient, so they cannot be scored this way. For each fifth of the patients, four stand-in networks trained without that fifth were scored instead. The stand-ins are not part of this download. No test patient was used to train or tune these three versions.

### The three versions read together (average of their probabilities)

| Measure | Ours (95% CI where recorded) | Test set | Best published, and who holds it | Same pictures? |
|---|---|---|---|---|
| Exact grade, out of 100 discs | 59.6 (55.5 to 63.4) | SPIDER development patients, out of fold: 168 patients, 1,004 discs | 79: SpineNet against a panel of experts (Spine 2023), which counts grades 1 and 2 as one. 84: the scale's authors, three observers on 300 discs (Pfirrmann et al., Spine 2001) | No. Counted the way SpineNet counts, ours is 69.1 (65.3 to 72.9) |
| Within one grade, out of 100 | 96.41 | the same | 99: SpineNet, and the scale's authors | No |
| Two or more steps off, out of 100 | 3.59 | the same | 1: the same two | No |
| Agreement score (quadratic weighted kappa; 0 is guessing, 1 is always agreeing) | 0.7997 | the same | 0.83 (0.74 to 0.91): three radiologists and three spine surgeons (Urrutia et al., Eur Spine J 2016; not re-read at the source). Doctors reach 0.66 to 0.83 with each other on this scale | No |
| Patients with a herniation caught, out of 100, when clearing 74 of 100 patients without one | 53 (38 to 70) | the same patients: 168, of whom 49 have a herniation | 58 (41 to 74): Gemini 3.1 Pro, untrained, on the middle sagittal T1 alone, 119 SPIDER patients (Eur Spine J 2026) | Same dataset, other patients |
| LSMA-PQR: exact grade, within one grade (out of 100), kappa | 71.68 (68.85 to 74.51), 98.30 (97.49 to 99.03), 0.7590 (0.7214 to 0.7903) | LSMA-PQR development patients, out of fold: 412 patients, 1,236 discs | Kappa 0.51: the dataset's own baseline grader (J Imaging Inform Med 2026) | Same dataset, other split. Its abstract does not say which patients scored 0.51 |

Each version alone scored 58.67 (v133), 60.06 (v134) and 58.07 (v140) exact on the same SPIDER discs. Changing only the random seed moves the exact grade by about 2 points.

### Other findings (area under the curve; 0.5 is a coin flip, 1.0 is always right)

| Finding | Three versions together, SPIDER development, out of fold | Version 1, SPIDER held-out test (95% CI) |
|---|---|---|
| Herniation (per disc) | 0.8836 | 0.818 (0.710 to 0.912) |
| Narrowing | not recorded | 0.956 (0.930 to 0.980) |
| Bulging | 0.8878 | 0.874 (0.824 to 0.919) |
| Slipped vertebra | 0.8642 | 0.887 (0.789 to 0.964) |
| Upper endplate change | not recorded | 0.817 (0.734 to 0.893) |
| Lower endplate change | not recorded | 0.850 (0.784 to 0.908) |
| Any Modic change | not recorded | 0.832 (0.743 to 0.911) |

### The earlier version, v1, on held-out test patients

These patients were never used to train or tune v1. The full tables are in [the SPIDER scorecard](../../modalities/spine_mri/eval/SCORECARD.md).

| Measure | Ours (95% CI) | Test set | Best published, and who holds it | Same pictures? |
|---|---|---|---|---|
| Exact grade, out of 100 discs | 60.6 (52.3 to 68.2) | SPIDER test: 44 patients, 264 discs, opened once; discs found by TotalSpineSeg | 79 (SpineNet) and 84 (the scale's authors) | No |
| Exact grade, discs found by SPIDER's own masks | 61.7 (52.7 to 70.1) | the same | 60 (54 to 68): radiomics plus image features with TabPFN, 20% of SPIDER's patients held out (Sci Rep 2026) | Same dataset, other split. Level |
| Within one grade, out of 100 | 95.8 (92.0 to 98.9) | the same | 99 | No |
| Agreement score (kappa) | 0.773 (0.664 to 0.853); 0.800 (0.695 to 0.873) with SPIDER's own masks | the same | 0.83 (doctors); 0.81 for the ordinal LightGBM in the same Sci Rep 2026 paper | No; same dataset, other split |
| Better than simple measurement rules? | kappa difference 0.135 (0.032 to 0.251), paired by patient: it passed the ship rule | the same | | |
| Version 103 on a second collection: kappa; exact grade and within one grade, out of 100 | 0.78 (0.72 to 0.84); 77 and 98 | LSMA-PQR test: 103 patients, 309 discs, opened once | Kappa 0.51: the dataset's own baseline grader | Same dataset, other split |
| Version 1 on those same discs, never having seen that collection | kappa 0.42; 30 of 100 exact | the same | | |

### Where it is behind

- Exact grade: 59.6 of 100 against 79 and 84. Counted the way SpineNet counts, 69.1 against 79.
- Two or more steps off: 3.59 of 100 against 1.
- Herniation per patient: 53 caught against 58. The gap is not statistically significant, but no version tried so far has reached 58. Single versions catch from 27 to 57.
- Kappa 0.7997 sits inside the 0.66 to 0.83 that doctors reach with each other. It is below the top of that range, and the intervals overlap, so it is not shown to be below doctors either.
- It learned from 2,240 discs of 580 patients. SpineNet's 2017 version learned from 12,018 discs.
- It has not been shown to beat doctors, and these tests could not show it. The answer key is one doctor's grades, so a reader better than that doctor is marked wrong exactly where it is right.

## Data it learned from

No dataset was used for pretraining. The networks start from random numbers (no ImageNet or other pretrained weights). The sets below were used for training, for labels and for tuning, in the roles named. The temperatures that turn scores into probabilities, and the yes/no cut-offs, were fitted on out-of-fold predictions for these same development patients.

- **SPIDER**, development patients only (168 scored patients, 1,004 discs). Used for training, labels (one musculoskeletal radiologist's grade for every disc) and tuning. Each disc was cut out twice, once placed by TotalSpineSeg and once by SPIDER's own masks. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: van der Graaf, van Hooff, Buckens et al., "Lumbar spine segmentation in MR images: a dataset and a public benchmark", Scientific Data 11, 264 (2024), doi:10.1038/s41597-024-03090-w; data doi:10.5281/zenodo.10159290, CC BY 4.0.
- **LSMA-PQR** v2, development patients only (412 patients, 1,236 discs, levels L3-L4, L4-L5 and L5-S1). Used for training and tuning: its Pfirrmann grades are the labels, and its disc masks place the crops. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Masood, Taj, Talha and Khan, "LSMA-PQR: Lumbar Spine Multi-view Annotations with Pfirrmann grading, Quantitative measurements and structured Radiological reports", Mendeley Data v2, doi:10.17632/p3r4xd2488.2, CC BY 4.0.
- **Lumbar Spine MRI Dataset** v2, the scans that LSMA-PQR graded. Used for the pictures of those 412 patients. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Sudirman, Al Kafri, Natalia, Meidia, Afriliana, Al-Rashdan, Bashtawi and Al-Jumaily, "Lumbar Spine MRI Dataset", Mendeley Data v2, doi:10.17632/k57fr854j2.2, CC BY 4.0.
- **Multi-Disorder Annotations for Lumbar Spine Mid-Sagittal Images** v1. Used for the bulging and slipped-vertebra labels of the LSMA-PQR discs. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Masood, Taj, Talha and Khan, "Multi-Disorder Annotations for Lumbar Spine Mid-Sagittal Images", Mendeley Data v1, doi:10.17632/9j9hmy8kbn.1, CC BY 4.0.
- **Radiologists Notes for Lumbar Spine MRI Dataset** v2 (free-text notes from radiologists, one per patient). Used for the herniation labels of the LSMA-PQR discs. A written rule reads each note. A second reading by AI language models flagged where the rule might be wrong. In 505 patients the two agreed, and 10 notes where they differed were settled by hand. 60 patients have no note, so their herniation labels are left unknown. No model's output was used directly as a label. LSMA-PQR's own structured table was not used for herniation, because it marked "no evidence of disc herniation" as a herniation. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Sudirman, Al Kafri, Natalia, Meidia, Afriliana, Al-Rashdan, Bashtawi and Al-Jumaily, "Radiologists Notes for Lumbar Spine MRI Dataset", Mendeley Data v2, doi:10.17632/s6bgczr8s2.2, CC BY 4.0.

**Used only for testing.** SPIDER's 44 held-out test patients (opened once, for v1) and LSMA-PQR's 103 held-out test patients (opened once, for v103). These three versions never trained on them and were not tuned on them.

**Not used.** ImageNet or any pretrained backbone; the RSNA LumbarDISC data and the SpineNet model (both noncommercial). Other AI models were run as comparison readers, or as a checker on the herniation labels. None of their answers is a label.

**Tool it depends on.** TotalSpineSeg (NeuroPoly; code under LGPL-3.0; release r20260730) finds and labels the discs, both when the training pictures were cut and when you read a scan. It is not included here. The licence of its trained weights is unsettled, and parts of its own training data are noncommercial or have no clear terms (sacrum labels from SynthRAD2023, CC BY-NC 4.0; MRSpineSeg and GoldAtlas, terms not confirmed). NeuroPoly has not cleared the weights for commercial use. Anything that depends on them, this model included, is for research and noncommercial use only. This is why commercial use of this model is not allowed (see Licence).

Credits for the sets above stay with every copy of the weights.

## Limitations and known failure modes

- **Scores on SPIDER are optimistic.** TotalSpineSeg was trained on SPIDER's pictures, so it finds SPIDER's discs better than it will find yours. Expect lower scores on new scans. On SPIDER's test patients TotalSpineSeg found 293 of 294 eligible discs.
- **One doctor's grades, one kind of patient.** The truth is one radiologist's grades of a Dutch back-pain group (SPIDER) and the grades of one other dataset's authors (LSMA-PQR). Other doctors would disagree on some discs. This card gives no results split by age, sex or scanner brand.
- **A new hospital can cost it most of its skill.** Version 1 scored kappa 0.42 on LSMA-PQR's test discs. Version 103, trained with that collection's development discs, scored 0.78 on the same discs. These three versions have seen two collections only: SPIDER's four Dutch hospitals and LSMA-PQR. On any other scanner or hospital, measure it before trusting any number here.
- **Herniation is the weak spot.** Asked "does this patient have a herniation anywhere?", it catches 53 of 100 while clearing 74 of 100 patients without one. Earlier versions mostly read wear. A healthy patient's worn L5-S1 disc scored as high as a real herniation. The herniation labels were rebuilt from the radiologists' own notes to fix part of this. The per-disc yes/no flag is set low so that it catches most herniations, and it raises many false alarms. For version 1 on SPIDER's test discs the flag had sensitivity 0.824 and specificity 0.785. Treat a flag as a reason to look, not as a finding.
- **Upper levels have less data.** LSMA-PQR covers L3-L4, L4-L5 and L5-S1 only. T12-L1, L1-L2 and L2-L3 learned from SPIDER's 168 patients alone.
- **Rare findings are noisy.** Modic types 1 and 3 are too rare to score on their own (4 and 7 discs in all of SPIDER), so only "any Modic change" is given. The slipped-vertebra score rests on few cases and has wide intervals.
- **It needs the disc finder.** If TotalSpineSeg misses a disc or labels the wrong level, you get no grade for that disc or a grade for the wrong disc. A T2 scan is required. A missing T1 is allowed (it is replaced by zeros and a flag).
- **Three seeds differ.** One version alone moves the exact grade by about 2 points. The averaged scores above need all three.
- **No held-out test yet.** These versions were chosen on development patients. See the note at the top of "How well it does".

## Licence

The weights and the code are under the [PolyForm Noncommercial License 1.0.0](../../LICENSE). This line travels with every copy:

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

You may use them free for noncommercial purposes the licence lists, such as personal study and use by educational or public research organisations. This model is not covered by the commercial licence: it is for noncommercial use only. The reason: it runs on TotalSpineSeg to find and label the discs, and its authors were never asked for commercial use. Permission for commercial use was never given. It depends on TotalSpineSeg's weights, which are not cleared for commercial use (see 'Tool it depends on'). It is still not a medical device and may not be used for diagnosis. You must accept the licence before you download the weights. The datasets keep their own CC BY 4.0 licences and credits, listed above. TotalSpineSeg has its own terms. More in [LICENSING.md](../../LICENSING.md).

## How to run it

Put the version folders you want (see Files) in one folder, set `RAIDIOLOGY_MODELS` to that folder, and run the reader on one version:

```
# environment variable RAIDIOLOGY_MODELS = the folder that holds disc-grader-v133, disc-grader-v134, disc-grader-v140
python modalities/spine_mri/eval/grader.py patient <t2.nii.gz> <labels.nii.gz> [<t1.nii.gz>] --version 140
```

`<labels.nii.gz>` is TotalSpineSeg's step 2 label file for the same T2 scan (see `modalities/spine_mri/tss.py` for the call). The T2 and T1 scans are NIfTI files. It prints one block of JSON per level that TotalSpineSeg found: the grade, the probability of each grade, each finding's probability, and the yes/no flags. The temperatures and cut-offs it applies are in each version's `grader.json`. It runs on a processor or a graphics card.

The command reads one version at a time. The scores above are for the three versions averaged, taking the mean of their probabilities. This repository does not yet have a command that does the averaging. Running the command once per version and averaging the probabilities gives a close read, but that exact route was not the one scored.

### Files

Each version is a folder of five fold networks and one `grader.json`. Only these 18 files are needed to read a scan.

| File | Bytes | SHA-256 |
|---|---:|---|
| `disc-grader-v133/fold0.pt` | 4,742,438 | `b9be076447f945bcf356dcc6815ce9fec9885ee6afa83bfe8d139855d4f0462a` |
| `disc-grader-v133/fold1.pt` | 4,742,438 | `e20a7a12d56d3cfc1fec1d10b32f76e60af70f9e1ad7258ff46f91446a00eb94` |
| `disc-grader-v133/fold2.pt` | 4,742,438 | `32bb0ee92abe192e751dbfc0e83b52dc9f8cb39da7b2ea712f6f8b11cce110ab` |
| `disc-grader-v133/fold3.pt` | 4,742,438 | `d4c1d27a173652a6a4b67942bcfed3b5648f884f402057caf88c623e90d3e2a9` |
| `disc-grader-v133/fold4.pt` | 4,742,438 | `88207cb1cce80cbef5006dfac19554a78b8f57d686221821d47e503a60e706be` |
| `disc-grader-v133/grader.json` | 2,312 | `911d8f3eba968733d7b5a51df7031d332cedc6e179070ba0d9ac20b769d26e78` |
| `disc-grader-v134/fold0.pt` | 4,742,438 | `0f6b62125a2f69ca78be2daddbf3d41229049e366be3c80e1cb084ea652ffa1b` |
| `disc-grader-v134/fold1.pt` | 4,742,438 | `fd2ed057a35bc586017ce4f4041d745ac159dc9c75c0f39dec8680aab3d746a1` |
| `disc-grader-v134/fold2.pt` | 4,742,438 | `9cc1c836fb40e00dcb29b56525520954c42838ce2ad248e024fa402b6a1c7da5` |
| `disc-grader-v134/fold3.pt` | 4,742,438 | `623749109d0ffe688e7a31c013802b6b6de2462f0081718e3528225b7d0b5ee5` |
| `disc-grader-v134/fold4.pt` | 4,742,438 | `83d7bc7fa0f444921ef52006b2ed555d7ee08c3a2c80113531765849a62784e9` |
| `disc-grader-v134/grader.json` | 2,319 | `1b01883676f6c94abfebf2dbf6e5864e984ac143f29531c117c5c08d339f9230` |
| `disc-grader-v140/fold0.pt` | 4,742,438 | `e7d6dd92bf8e4c0d657fa3e041d4542993b82cc34470541eea930144e290c984` |
| `disc-grader-v140/fold1.pt` | 4,742,438 | `92e8eecf670edc2567baebdfb89038b483dbee6621e6bb555803950c3f30bf32` |
| `disc-grader-v140/fold2.pt` | 4,742,438 | `f501eea6fc5fb9212ca72bdaad9da3ff171ab9d2b99006aa3a19017ff2e28000` |
| `disc-grader-v140/fold3.pt` | 4,742,438 | `fc2929563693710e02df52d95900b1a88d2a6023a753d6be64153c8041cf96c5` |
| `disc-grader-v140/fold4.pt` | 4,742,438 | `546753fcdf2db6dbb8df51ef7d0c628278ccb2826ecbb8d0ae8d74f049e7d814` |
| `disc-grader-v140/grader.json` | 2,348 | `321119f8096d9d0656c84e252a3d84bf73d3f6ee8090922cd587847c3651b798` |

Code: `modalities/spine_mri/eval/grader.py` (network and reader), `modalities/spine_mri/eval/crops.py` (how a disc is cut out). The earlier version's card is `modalities/spine_mri/models/disc-grader.md`.
