# Model card: rAidiology disc grader v144 + v145 + v146, with the rAidiology disc finder v1

**Not a medical device. Not for diagnosis or treatment decisions.**

Written 2026-11-10. These are three versions of one recipe, trained with three different random seeds (0, 1 and 2), on discs placed by rAidiology's own disc finder. The scores below are for the three read together. They replace v133, v134 and v140, which placed the discs with TotalSpineSeg; the scores are the same within the seed noise. An earlier version, v1, was published before them; its held-out test scores are given too.

## What it does

It grades the discs of the lower spine on a sagittal MRI. You give it a sagittal T2 scan and a sagittal T1 scan if you have one. Its own disc finder, included in the download, first finds and numbers each disc on the T2. For each disc from T12-L1 down to L5-S1 it cuts out a small block of pictures and returns a Pfirrmann grade from 1 to 5, the chance of each grade, and the chance of seven findings: herniation, narrowing, bulging, a slipped vertebra (spondylolisthesis), change in the upper endplate, change in the lower endplate, and any Modic change. Each version is five small networks (about 4.7 MB each) whose answers are averaged.

A Pfirrmann grade says how worn a disc looks on MRI. Grade 1 is a healthy, plump, watery disc. Grade 5 is a disc worn flat and dried out.

### What it is not for

- Diagnosis, treatment decisions, triage (who is seen first) or screening. A person must read the scan.
- Children and teenagers. It was not tested on them.
- People without back pain. SPIDER, the main training set, holds people with low back pain only.
- Other pictures. It reads sagittal T2 (with sagittal T1 if present) of the lumbar spine. It does not read axial or coronal pictures, neck (cervical) discs, chest (thoracic) discs, or other body parts.
- Spines after surgery or with implants. It was not checked on them.
- Other scanners and hospitals, without your own check. Its skill dropped sharply when an earlier version met a hospital it had never seen (see Limitations).

## How well it does

Every score here is agreement with a doctor's grades, or with the grades a dataset's authors made. It is not agreement with the truth. For scale: always guessing grade 3 (the commonest) gets 30 of 100 SPIDER development discs exactly right.

**These three versions have not been scored on held-out test patients.** They were chosen using development patients only. The held-out test patients were opened once for earlier versions (v1 on SPIDER, v103 on LSMA-PQR). Opening them again for these versions would be a second look, so it has not been done. The first table is therefore a development result, "out of fold": every patient is scored by networks that never saw that patient. The five networks you download have, between them, seen every development patient, so they cannot be scored this way. For each fifth of the patients, four stand-in networks trained without that fifth were scored instead. The stand-ins are not part of this download. No test patient was used to train or tune these three versions.

### The three versions read together (average of their probabilities)

| Measure | Ours (95% CI where recorded) | Test set | Best published, and who holds it | Same pictures? |
|---|---|---|---|---|
| Exact grade, out of 100 discs | 60.1 (55.8 to 64.2) | SPIDER development patients, out of fold: 168 patients, 1,004 discs | 79: SpineNet against a panel of experts (Spine 2023), which counts grades 1 and 2 as one. 84: the scale's authors, three observers on 300 discs (Pfirrmann et al., Spine 2001) | No. Counted the way SpineNet counts, ours is 69.2 (65.3 to 73.0) |
| Within one grade, out of 100 | 96.31 (94.52 to 97.90) | the same | 99: SpineNet, and the scale's authors | No |
| Two or more steps off, out of 100 | 3.69 | the same | 1: the same two | No |
| Agreement score (quadratic weighted kappa; 0 is guessing, 1 is always agreeing) | 0.8012 (0.7550 to 0.8368) | the same | 0.83 (0.74 to 0.91): three radiologists and three spine surgeons (Urrutia et al., Eur Spine J 2016; not re-read at the source). Doctors reach 0.66 to 0.83 with each other on this scale | No |
| Patients with a herniation caught, out of 100, when clearing 75 of 100 patients without one | 51 (37 to 65) | the same patients: 168, of whom 49 have a herniation | 58 (41 to 74): Gemini 3.1 Pro, untrained, on the middle sagittal T1 alone, 119 SPIDER patients (Eur Spine J 2026) | Same dataset, other patients |
| LSMA-PQR: exact grade, within one grade (out of 100), kappa | 72.17 (69.34 to 75.08), 98.22 (97.41 to 98.95), 0.7612 (0.7271 to 0.7928) | LSMA-PQR development patients, out of fold: 412 patients, 1,236 discs | Kappa 0.51: the dataset's own baseline grader (J Imaging Inform Med 2026) | Same dataset, other split. Its abstract does not say which patients scored 0.51 |

Each version alone scored 58.37 (v144), 59.06 (v145) and 58.67 (v146) exact on the same SPIDER discs. Changing only the random seed moves the exact grade by about 2 points. The three versions they replace (v133 + v134 + v140, discs placed by TotalSpineSeg) scored 59.6 exact and kappa 0.7997 together; paired by patient, the difference is +0.5 points exact (95% interval -1.6 to +2.5) and +0.0014 kappa (-0.0096 to +0.0122), both inside the seed noise.

### Other findings (area under the curve; 0.5 is a coin flip, 1.0 is always right)

| Finding | Three versions together, SPIDER development, out of fold | Version 1, SPIDER held-out test (95% CI) |
|---|---|---|
| Herniation (per disc) | 0.8732 | 0.818 (0.710 to 0.912) |
| Narrowing | not recorded | 0.956 (0.930 to 0.980) |
| Bulging | 0.8872 | 0.874 (0.824 to 0.919) |
| Slipped vertebra | 0.8660 | 0.887 (0.789 to 0.964) |
| Upper endplate change | not recorded | 0.817 (0.734 to 0.893) |
| Lower endplate change | not recorded | 0.850 (0.784 to 0.908) |
| Any Modic change | not recorded | 0.832 (0.743 to 0.911) |

### The earlier version, v1, on held-out test patients

These patients were never used to train or tune v1. The full tables are in [the SPIDER scorecard](https://github.com/LunarWerxs/rAidiology/blob/main/modalities/spine_mri/eval/SCORECARD.md).

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

- Exact grade: 60.1 of 100 against 79 and 84. Counted the way SpineNet counts, 69.2 against 79.
- Two or more steps off: 3.69 of 100 against 1.
- Herniation per patient: 51 caught against 58. The gap is not statistically significant, but no version tried so far has reached 58. Earlier single versions caught from 27 to 57.
- Kappa 0.8012 sits inside the 0.66 to 0.83 that doctors reach with each other. It is below the top of that range, and the intervals overlap, so it is not shown to be below doctors either.
- It learned from 2,240 discs of 580 patients. SpineNet's 2017 version learned from 12,018 discs.
- It has not been shown to beat doctors, and these tests could not show it. The answer key is one doctor's grades, so a reader better than that doctor is marked wrong exactly where it is right.

## Data it learned from

No dataset was used for pretraining. The networks start from random numbers (no ImageNet or other pretrained weights). The sets below were used for training, for labels and for tuning, in the roles named. The temperatures that turn scores into probabilities, and the yes/no cut-offs, were fitted on out-of-fold predictions for these same development patients.

- **SPIDER**, development patients only (168 scored patients, 1,004 discs). Used for training, labels (one musculoskeletal radiologist's grade for every disc) and tuning. Each disc was cut out twice, once placed by rAidiology's disc finder (out of fold: by a finder that never saw that patient) and once by SPIDER's own masks. SPIDER's masks of the same development patients also trained the disc finder. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: van der Graaf, van Hooff, Buckens et al., "Lumbar spine segmentation in MR images: a dataset and a public benchmark", Scientific Data 11, 264 (2024), doi:10.1038/s41597-024-03090-w; data doi:10.5281/zenodo.10159290, CC BY 4.0.
- **LSMA-PQR** v2, development patients only (412 patients, 1,236 discs, levels L3-L4, L4-L5 and L5-S1). Used for training and tuning: its Pfirrmann grades are the labels, and its disc masks place the crops. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Masood, Taj, Talha and Khan, "LSMA-PQR: Lumbar Spine Multi-view Annotations with Pfirrmann grading, Quantitative measurements and structured Radiological reports", Mendeley Data v2, doi:10.17632/p3r4xd2488.2, CC BY 4.0.
- **Lumbar Spine MRI Dataset** v2, the scans that LSMA-PQR graded. Used for the pictures of those 412 patients. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Sudirman, Al Kafri, Natalia, Meidia, Afriliana, Al-Rashdan, Bashtawi and Al-Jumaily, "Lumbar Spine MRI Dataset", Mendeley Data v2, doi:10.17632/k57fr854j2.2, CC BY 4.0.
- **Multi-Disorder Annotations for Lumbar Spine Mid-Sagittal Images** v1. Used for the bulging and slipped-vertebra labels of the LSMA-PQR discs. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Masood, Taj, Talha and Khan, "Multi-Disorder Annotations for Lumbar Spine Mid-Sagittal Images", Mendeley Data v1, doi:10.17632/9j9hmy8kbn.1, CC BY 4.0.
- **Radiologists Notes for Lumbar Spine MRI Dataset** v2 (free-text notes from radiologists, one per patient). Used for the herniation labels of the LSMA-PQR discs. A written rule reads each note. A second reading by an AI language model (Claude Sonnet 5.5 over Anthropic's API, whose terms give the output to the user) flagged where the rule might be wrong. In 505 patients the two agreed, and 10 notes where they differed were settled by hand. 60 patients have no note, so their herniation labels are left unknown. No model's output was used directly as a label. LSMA-PQR's own structured table was not used for herniation, because it marked "no evidence of disc herniation" as a herniation. Licence: CC BY 4.0. Commercial use: allowed with credit. Credit: Sudirman, Al Kafri, Natalia, Meidia, Afriliana, Al-Rashdan, Bashtawi and Al-Jumaily, "Radiologists Notes for Lumbar Spine MRI Dataset", Mendeley Data v2, doi:10.17632/s6bgczr8s2.2, CC BY 4.0.

**Used only for testing.** SPIDER's 44 held-out test patients (opened once, for v1) and LSMA-PQR's 103 held-out test patients (opened once, for v103). These three versions never trained on them and were not tuned on them.

**Not used.** ImageNet or any pretrained backbone; TotalSpineSeg's weights and labels (not cleared for commercial use; `RAIDIOLOGY_NO_TSS=1` makes the code refuse to load it); the RSNA LumbarDISC data and the SpineNet model (both noncommercial). Other AI models were run as comparison readers, or as a checker on the herniation labels. None of their answers is a label.

**The disc finder.** rAidiology's own (`modalities/spine_mri/finder.py`, folder `spine-disc-finder-v1`): a small 2D network trained from random numbers on SPIDER's development patients' T2 masks (CC BY 4.0) and nothing else. Out of fold on the 168 development patients it found all 1,004 discs, with Dice 0.885 (0.878 to 0.892) and a median centre miss of 0.72 mm, and gave 96.8 of 100 discs (94.0 to 99.2) their right level; every level was right in 96.4 of 100 patients. It ships with the grader, so nothing outside this download is needed to read a scan.

Credits for the sets above stay with every copy of the weights.

## Limitations and known failure modes

- **Scores on SPIDER are optimistic.** The disc finder learned from SPIDER's pictures. Each patient was read by a finder that never saw that patient, but the patients come from the same four hospitals, so it finds SPIDER's discs better than it will find yours. Expect lower scores on new scans. The finder has not been scored on SPIDER's held-out test patients.
- **One doctor's grades, one kind of patient.** The truth is one radiologist's grades of a Dutch back-pain group (SPIDER) and the grades of one other dataset's authors (LSMA-PQR). Other doctors would disagree on some discs. This card gives no results split by age, sex or scanner brand.
- **A new hospital can cost it most of its skill.** Version 1 scored kappa 0.42 on LSMA-PQR's test discs. Version 103, trained with that collection's development discs, scored 0.78 on the same discs. These three versions have seen two collections only: SPIDER's four Dutch hospitals and LSMA-PQR. On any other scanner or hospital, measure it before trusting any number here.
- **Herniation is the weak spot.** Asked "does this patient have a herniation anywhere?", it catches 53 of 100 while clearing 74 of 100 patients without one. Earlier versions mostly read wear. A healthy patient's worn L5-S1 disc scored as high as a real herniation. The herniation labels were rebuilt from the radiologists' own notes to fix part of this. The per-disc yes/no flag is set low so that it catches most herniations, and it raises many false alarms. For version 1 on SPIDER's test discs the flag had sensitivity 0.824 and specificity 0.785. Treat a flag as a reason to look, not as a finding.
- **Upper levels have less data.** LSMA-PQR covers L3-L4, L4-L5 and L5-S1 only. T12-L1, L1-L2 and L2-L3 learned from SPIDER's 168 patients alone.
- **Rare findings are noisy.** Modic types 1 and 3 are too rare to score on their own (4 and 7 discs in all of SPIDER), so only "any Modic change" is given. The slipped-vertebra score rests on few cases and has wide intervals.
- **It needs the disc finder.** If the finder misses a disc or labels the wrong level (it gave 3.2 of 100 SPIDER development discs the wrong level), you get no grade for that disc or a grade for the wrong disc. A T2 scan is required. A missing T1 is allowed (it is replaced by zeros and a flag).
- **Three seeds differ.** One version alone moves the exact grade by about 2 points. The averaged scores above need all three.
- **No held-out test yet.** These versions were chosen on development patients. See the note at the top of "How well it does".

## Licence

The weights and the code are under the [PolyForm Noncommercial License 1.0.0](https://github.com/LunarWerxs/rAidiology/blob/main/LICENSE). This line travels with every copy:

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

- Free (a plain reading; the LICENSE text is what binds): personal study, research, and use by charities, schools, public research and health organisations, and governments.
- Commercial use is available under the rAidiology Commercial Licence ([COMMERCIAL-LICENSE.md](https://github.com/LunarWerxs/rAidiology/blob/main/COMMERCIAL-LICENSE.md)), with a free 60-day evaluation (500 studies, internal testing only, no customers, no patients' care), priced on the rAidiology site's pricing page. Commercial use includes companies using it in paid work, in internal tools, and in any product or hosted service built on it. Not for diagnosis or treatment. Not cleared by the FDA or any regulator.
- Downloads are on request: universities, schools, public research and health bodies and governments are approved from their email address; a company gets the commercial licence and its free 60-day evaluation copy.
- The data credits above also travel with the weights. Its whole chain (training data, the disc finder, the labels) allows commercial use; more in [LICENSING.md](https://github.com/LunarWerxs/rAidiology/blob/main/LICENSING.md).

## How to run it

Put the version folders you want and the `spine-disc-finder-v1` folder (see Files) in one folder, set `RAIDIOLOGY_MODELS` to that folder, find the discs, then read the scan with one version:

```
# environment variable RAIDIOLOGY_MODELS = the folder that holds spine-disc-finder-v1 and disc-grader-v144, -v145, -v146
# environment variable RAIDIOLOGY_NO_TSS = 1 (optional: the code then refuses to load TotalSpineSeg anywhere)
python modalities/spine_mri/finder.py segment <t2.nii.gz> <labels.nii.gz>
python modalities/spine_mri/eval/grader.py patient <t2.nii.gz> <labels.nii.gz> [<t1.nii.gz>] --version 146
```

`<labels.nii.gz>` is the disc finder's label file for the same T2 scan, written by the first command. The T2 and T1 scans are NIfTI files. It prints one block of JSON per level the finder found: the grade, the probability of each grade, each finding's probability, and the yes/no flags. The temperatures and cut-offs it applies are in each version's `grader.json`. It runs on a processor or a graphics card; on a processor the finder took about 10 seconds for one SPIDER scan.

The command reads one version at a time. The scores above are for the three versions averaged, taking the mean of their probabilities. This repository does not yet have a command that does the averaging. Running the command once per version and averaging the probabilities gives a close read, but that exact route was not the one scored.

### Files

Each version is a folder of five fold networks and one `grader.json`; the disc finder is one network (`all.pt`) and its `finder.json`. Only these 20 files are needed to read a scan.

| File | Bytes | SHA-256 |
|---|---:|---|
| `disc-grader-v144/fold0.pt` | 4,742,837 | `d49ba29377bd09b279b402c84749fb19188a5f25218e3c8090d68e66d16935f4` |
| `disc-grader-v144/fold1.pt` | 4,742,837 | `378d3b22e6e2242f3e1e80fdd78d2e20d5aa1392fc46c1cf1b746aba87ca45c9` |
| `disc-grader-v144/fold2.pt` | 4,742,837 | `36a8ac7173aa811b347062af5c067753b39e0a0beb2237709449a146f4878b33` |
| `disc-grader-v144/fold3.pt` | 4,742,837 | `b9d2c07f2a334324a0216e5cd3bb9000d1d0454a87d1363e38f893d717f5c0dc` |
| `disc-grader-v144/fold4.pt` | 4,742,837 | `479f424a911f7711e9622ee4decedd3b80aab6ec246598497eeeced8cb275913` |
| `disc-grader-v144/grader.json` | 2,328 | `b91564a1bcaf3dbbfb3ec7e73e565ceeb2877764aa47f43cceca71ed2d2ba629` |
| `disc-grader-v145/fold0.pt` | 4,742,837 | `746a1ef329a686d8cf9b47f681185071e885ff306e6fd08c236419a57da90137` |
| `disc-grader-v145/fold1.pt` | 4,742,837 | `eea2bc591d5df759b6c59f87b80d28517577ba5c9cc5c03069a518b97c8b52ca` |
| `disc-grader-v145/fold2.pt` | 4,742,837 | `e818802e6b68bde20fa53a67e348728ff9dd6e8441bf53b26616afaf5031014a` |
| `disc-grader-v145/fold3.pt` | 4,742,837 | `d9f565ed8a3efff7a81afbec018d52c37cd6959fc11a6a6da8ee038b39e8a6c1` |
| `disc-grader-v145/fold4.pt` | 4,742,837 | `b62e5651cd60f855ea284f84e4611962155aaaacf4defe6d7e1c0f7cb8aa5ce4` |
| `disc-grader-v145/grader.json` | 2,340 | `cfc8de72814879545ccac2ea7b5c96a90d50c43545bda121d4a2af35b128de18` |
| `disc-grader-v146/fold0.pt` | 4,742,837 | `dd4001868777a24f73896ae6e80e029935ceaf1bcea57b3125aba6f1e29801fd` |
| `disc-grader-v146/fold1.pt` | 4,742,837 | `1ba865d7fd50ecf95d549780cfd9de8b0496f96e125a5f9439d003dd4902ad42` |
| `disc-grader-v146/fold2.pt` | 4,742,837 | `72a30b424dda2aa7e1fabfe09ebdf5d78ec6c23fce1cc9b0efe457bb7547a308` |
| `disc-grader-v146/fold3.pt` | 4,742,837 | `39dd47fc90436e5c3756d7a82d4b6736e07665c3e30b7897b370ac281925120a` |
| `disc-grader-v146/fold4.pt` | 4,742,837 | `5d6248801f02752a97ef4901ec313359f98fb18dcbe6b757b64b9d2f62a6ed3e` |
| `disc-grader-v146/grader.json` | 2,339 | `434252eac4f0548d7e51277f9306fbdc1063e0aacea63dee832c9a5795c10799` |
| `spine-disc-finder-v1/all.pt` | 22,760,779 | `da7f9fc19bb27bbd2fbb0270279400935ba128c16e36fc99b97201228c0e4ded` |
| `spine-disc-finder-v1/finder.json` | 1,537 | `1b28ad5eff74478ca1e34427924e70e5aa77f50d4e48e0b5a6fde69279e54c57` |

Code: `modalities/spine_mri/eval/grader.py` (network and reader), `modalities/spine_mri/eval/crops.py` (how a disc is cut out). The earlier version's card is `modalities/spine_mri/models/disc-grader.md`.
