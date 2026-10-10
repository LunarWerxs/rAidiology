# Model card: rAidiology Lung nodule finder and malignancy reader (finder det-v1, second look v2 and v3, cancer-or-not v5)

**Not a medical device. Not for diagnosis or treatment decisions.**

Written 2026-10-10 from results recorded up to that day.

## What it does

It looks for small round spots (nodules) in a chest CT scan, and it can give one nodule a cancer-or-not score. You give it one chest CT series with slices 2.5 mm thick or thinner. The scan is first turned into a 1 mm picture cropped to the lungs. The finder and the second look return a list of places in the scan (x, y, z in millimetres). Each place has a score for how likely a nodule is there. The cancer-or-not reader takes one nodule's centre and returns a score for how likely it is cancer, from a 48 mm cube around it.

It is a chain of parts, not one network:

| Part | What it does | Files in the download | Size of each file |
|---|---|---|---|
| Finder, det-v1 | A 3D network that scores every 2 mm cell of the lungs for "a nodule is centred here". Keeps up to 100 places a scan. | `det-v1/fold0.pt` to `fold4.pt` | about 11 MB |
| Blob finder | A classic method with no learning (no weights). Adds the 300 roundest bright spots a scan. | none | none |
| Second look, v2 and v3 | Two small 3D networks. Each re-scores a place from a 48 mm cube. v3 sees three CT windows. Each reads a scan's 32 likeliest places eight ways (mirror images) and the scores are averaged. | `fpr-v2/fold0.pt` to `fold4.pt`, `fpr-v3/fold0.pt` to `fold4.pt` | about 7.9 MB (fpr-v2 fold 0 is 18.3 MB until a clean net is exported) |
| Lung step | Drops every place more than 14.55 mm outside the lung outline. The outline comes from CT thresholds, not a network. | a setting, `inlung_D.json` (14.55 mm) | none |
| Re-scorer and soft rule | A logistic model over eight inputs: the two second-look scores, the finder's score, and five plain facts about the place (how far outside or inside the lung, how bright, how crowded). Then each place's score is lowered by how close a stronger place sits. | fitted from the out-of-fold lists | none |
| Cancer-or-not reader, v5 | One small 3D network on a 48 mm cube around a nodule. One score. | `malignancy-v5/fold0.pt` to `fold4.pt` | about 7.9 MB |

Every network comes in five copies, one per fold. Each copy was trained on four fifths of the scans and never saw the other fifth. Every score below was made by reading each scan with the copy that had never seen it.

### What it is not for

- **Diagnosis.** It does not tell anyone whether they have cancer. The cancer-or-not score is a model's guess, and a nodule's size alone predicts about as well (see the tables).
- **Triage or screening decisions.** Do not use it to decide who is called back, who is sent home or which scan is read first. It has not been tested against a screening standard such as Lung-RADS.
- **Treatment planning.** Do not use its places or scores to plan any treatment.
- **Children.** Age was not used and was not checked. No scans of children were chosen or tested.
- **Other scan types.** It reads chest CT only. It does not read chest X-ray, MRI, PET or a single slice.
- **Thick slices.** The scans it was scored on are 2.5 mm or thinner. Nothing was scored on thicker slices.
- **Other scanners, reconstruction settings and hospitals.** Its detection scores come from one collection of scans (LIDC-IDRI). Results on scans from elsewhere are not known.
- **Anything but nodules.** It does not look for other chest findings such as emphysema, infection, fluid or enlarged lymph nodes.
- **Reading one new scan with the five copies averaged.** That has not been scored (see Limitations).

## How well it does

**CPM** is how many of 100 real nodules are found, averaged over seven allowances of false alarms per scan (1/8, 1/4, 1/2, 1, 2, 4 and 8). It is shown out of 100. Higher is better. A nodule counts as found when a place lies closer to its centre than its radius.

The answer key is the 1,186 nodules of 3 mm or more that at least 3 of 4 radiologists marked in LUNA16's 888 thin-slice LIDC-IDRI scans. LUNA16 keeps a second list of findings it counts neither way (under 3 mm, or marked by too few radiologists). One kind of those findings has no size on file. Our scorer treats such a finding as 1 mm wide. The script that LUNA16 ships treats it as 10 mm wide, so more false alarms land on it and are not charged. We show both readings.

LUNA16 has no separate test set. Its protocol is ten fixed groups of scans, each scored by a model trained on the other nine. We use five folds of two groups, so each model trains on 80% of the other scans where the protocol gives 90%. That is a disadvantage for us. Many designs were compared on these same folds, so treat the detection scores as development scores, not as scores from an untouched test.

| Measure | Ours (95% interval) | Test set | Published score to compare, and who holds it | Same pictures? |
|---|---|---|---|---|
| Nodules found, whole system, our strict reading (CPM) | **88.0** (86.0 to 89.6). It holds 1,178 of the 1,186 nodules. | LUNA16's 888 scans, five folds, every scan read by a model that never trained on it | 81.1, ZNET, a detection system from LUNA16's challenge paper (Setio et al., Med Image Anal 2017). 93.0, nnDetection (Baumgartner et al., MICCAI 2021). 95.1, PAtech, an entry on LUNA16's public leaderboard (2 January 2018). | Yes, the same 888 scans. Ahead of 81.1. Behind 93.0 and 95.1 under this reading. |
| The same system, read by LUNA16's own script as shipped (the way nnDetection's README runs it) | **92.4** (90.8 to 93.9). 82.3 found at one false alarm in eight scans, where the strict reading finds 70.5. | The same 888 scans | 93.0, nnDetection | Yes. Level with 93.0, which sits inside our interval. Behind 95.1. |
| Finder alone (det-v1) | 75.6 (72.9 to 77.9) | The same 888 scans | No published score for this part alone | Not applicable |
| Second look (v2 and v3 averaged, then the soft rule), on LUNA16's own list of 754,975 candidate places (CPM) | 85.4 (83.1 to 87.3). v2 alone 82.1 (79.8 to 84.3). | The same 888 scans, five folds | 90.8, CUMedVis (Dou et al.). 96.8, PATech, an entry on LUNA16's false-positive-reduction leaderboard (20 December 2017). | Yes, the same list. Behind both. |
| One radiologist against the other three: nodules caught when raising no more false alarms than the radiologists did | 57.6 caught, on LUNA16's 886 four-reader scans. 20.0 behind the radiologists (15.4 to 28.8). | The four LIDC radiologists, each scored against the other three | The radiologists catch 77.6 on the same 886 scans; ours is 20.0 behind (15.4 to 28.8). On all 1,016 LIDC scans they catch 77.5 (76.0 to 79.0), at 0.03 false alarms per scan. | Yes. Behind. At the radiologists' own catch rate the system raises 0.14 false alarms per pair against their 0.03, about 5 times as many. |
| Cancer or not, per patient, against LIDC's pathology table (the patient's most suspicious nodule, AUROC) | **0.717** (0.599 to 0.827), v5 | 117 patients (86 cancers, 31 benign), every nodule scored out of fold | 0.877 (give or take 0.036), a network trained on the pathology labels (Wang et al., 2022). The radiologists' ratings score 0.762. | Yes, the same 117 patients. Behind 0.877. The largest nodule's diameter alone scores 0.755, so per patient the model adds nothing over size yet. |
| Cancer or not, per nodule, against the radiologists' ratings (AUROC) | 0.9485 (0.9300 to 0.9652), v5 | 749 LIDC nodules, out of fold, patients resampled for the interval | None. These labels are the radiologists' opinion, not pathology. | Not applicable. The nodule's diameter alone scores 0.939 on LIDC's labels. |
| Cancer or not, 36 cancers and 37 benign nodules in the LUNGx test scans | not run | The LUNGx test set has not been opened | 0.85, one of six radiologists (their range is 0.70 to 0.85). Computer methods 0.50 to 0.68 (Armato et al., J Med Imaging 2016). | Not run |

Notes on the table:

- AUROC is the chance that a random cancer scores higher than a random benign nodule. 0.5 is a coin flip and 1.0 is perfect.
- The radiologists' row uses the unblinded LIDC reads, where each radiologist had seen the others' marks first, so their false alarms are almost none. To beat them a system must catch more nodules at nearly zero false alarms. This system does not.
- 77.6 is the radiologists' rate on the 886 four-reader scans; 77.5 (76.0 to 79.0) is their rate on all 1,016 LIDC scans.
- Inside the 0.9485 row, by size: under 10 mm 0.818 (364 nodules, 41 malignant); 10 to 20 mm 0.852 (248 nodules, 186 malignant). Over 20 mm, 136 of 137 nodules are labelled malignant, so the one benign nodule cannot support a score there.
- Where it is behind: 93.0 and 95.1 on detection under the strict reading, 90.8 and 96.8 on LUNA16's own list, the radiologists on false alarms, and 0.877 on cancer or not.

## Data it learned from

Nothing was pretrained on other data. All networks started from random weights. The blob finder and the lung outline use no learning.

| Dataset | Used for | Licence | Credit it requires | Commercial use |
|---|---|---|---|---|
| LIDC-IDRI (1,018 chest CT scans of 1,010 patients, with nodule marks and ratings by four thoracic radiologists) | **Training** the finder, the second look and the cancer-or-not reader. Its 130 scans that LUNA16 left out set the lung step's 14.55 mm limit (207 nodule centres) and add 133 nodules to the cancer-or-not reader's training. **Checking (development):** its pathology table (117 patients) has scored several rounds of the cancer-or-not reader and helped choose v5; nothing in v5 trains on it. | CC BY 3.0, under TCIA's Data Usage Policy | Cite Armato et al., The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI), Medical Physics 38(2), 2011; data doi:10.7937/K9/TCIA.2015.LO9QL9SX; and Clark et al., The Cancer Imaging Archive (TCIA), J Digit Imaging 2013. Never try to identify a participant. | Allowed, with the credit |
| LUNA16 (the answer key of 1,186 nodules in 888 LIDC-IDRI scans, its list of findings counted neither way, its ten groups of scans, and its scoring script) | **Labels** for the finder and the second look, the folds, and all detection scoring. Its list of 754,975 candidate places (`candidates_V2.csv`) also supplies the second look's training places. | CC BY 4.0 for the answer key, the scan groups and the scoring script (both Zenodo records' licence field). **To confirm before release:** the licence of `candidates_V2.csv`, which LICENSING.md does not list yet. | Cite Setio et al., Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge, Medical Image Analysis 42, 2017. The scans come from LIDC-IDRI (row above). | Allowed, with the credit, for the answer key, the scan groups and the scoring script. Not yet confirmed for `candidates_V2.csv`. |
| LIDC-annot-NLST501 (501 low-dose chest CT scans from the National Lung Screening Trial, each read by 4 of 20 radiologists; patient-level lung cancer from NLST follow-up) | **Training** the cancer-or-not reader only: 858 nodules, used for training alone, never held out and never scored | CC BY 4.0, under TCIA's Data Usage Policy | Cite Renoust et al. 2026, DOI 10.7937/Y2DX-E334, and TCIA | Allowed, with the credit |
| NLST (the National Lung Screening Trial images behind the set above) | The 501 scans above | Images CC BY 4.0 on TCIA | Cite the collection's DOI, 10.7937/TCIA.HMQ8-J677, and TCIA, as TCIA's Data Usage Policy asks. Only the images on TCIA were used. The cancer labels came with LIDC-annot-NLST501 under its own CC BY 4.0 licence. Nothing was requested from NLST's controlled-access clinical release. | Allowed, with the credit |
| NLST-Sybil (expert tumour boxes on NLST scans, labels only) | **Label correction** only: 38 of the NLST501 cancer labels were corrected against its boxes (12 positives to unknown, 26 confirmed cancers to positive). No scans. | CC BY 4.0 | Credit "NLST-Sybil: Expert annotations of tumor regions in the NLST CT images (IDC, v1)", Zenodo record 15643335, CC BY 4.0, with the record's authors | Allowed, with the credit |
| SPIE-AAPM Lung CT Challenge (LUNGx) | **Checking only.** Its 10 practice scans checked that nodule coordinates land on the nodules. Its 60 test scans have not been opened. Nothing trains on it. | CC BY 3.0, under TCIA's Data Usage Policy | Cite DOI 10.7937/K9/TCIA.2015.UZLSU3FL and Armato et al., J Med Imaging 2016, 3(4):044506 | Allowed, with the credit |

Used only for testing or checking: the LIDC pathology table, and the LUNGx scans. LUNA16's scoring script is used only to score.

Not used, so no licence from them applies: LUNA25 (its nodule list is non-commercial), the Shanghai multiomics nodule set (tried as training for the cancer-or-not reader and not kept, so it is not in v5), Lung-PET-CT-Dx, NSCLC-Radiomics, LNDb, ImageNet weights, and any set under a data use agreement.

The cancer-or-not labels are two kinds of opinion. For LIDC they are the radiologists' median malignancy rating (1 to 5): above 3 is malignant, below 3 is benign, exactly 3 is left out. At least 3 radiologists must have marked the nodule. That gives 882 labelled LIDC nodules (432 malignant, 450 benign). The 858 NLST nodules are labelled from the patient's cancer follow-up instead.

## Limitations and known failure modes

- **Development scores.** The detection numbers come from the same 888 scans that were used to choose between designs. There is no untouched test. The cancer-or-not model's LUNGx test is unopened. The 0.717 per-patient score is also a development score: the same 117 patients were used to compare rounds.
- **One source of scans.** Detection was trained and scored on LIDC-IDRI scans only. How it reads other scanners, hospitals, reconstruction kernels, contrast and low-dose screening scans has not been scored.
- **The five copies averaged are unscored.** Every score above has each scan read by the one copy that never saw it. How well the five copies do together on a brand-new scan has not been measured.
- **False alarms.** At one false alarm in eight scans it finds 70.5 of 100 nodules under the strict reading, and 82.3 under LUNA16's script. At the radiologists' own catch rate it raises about 5 times as many false alarms (0.14 per pair against their 0.03).
- **Misses.** Its list holds 1,178 of the 1,186 nodules, so 8 never get a place near them. An early look at the first network, on one fold, found that nodules touching the lung wall were missed more often (29 of 54 reached, against 147 of 186 inside the lung). That has not been re-measured for the current finder.
- **The lung step can drop a real nodule.** Places more than 14.55 mm outside the lung outline are removed. The limit is the farthest of 207 nodule centres (12.55 mm out) plus 2 mm. The outline is found with CT thresholds. A scan where it finds less than 0.5 litre of lung is skipped and reported, not guessed at.
- **Full detail needed.** The second look lost accuracy when scans were first stored as smaller 1.5 mm copies and read from those: the v2 net alone read 0.7720 against 0.8066 on the original scans, and v3 read 0.0373 lower (0.0275 to 0.0452). The finder and the cancer-or-not reader were not made worse. Read the original scans at full detail.
- **Cancer or not leans on size.** Per patient, the largest nodule's diameter alone (0.755) scores higher than the model (0.717). The LIDC labels are radiologists' opinion. The NLST labels are patient-level follow-up, which says which patient had cancer and not which nodule, so the most suspicious nodule of a cancer patient is assumed to be the cancer. Small nodules are hard: within the 0.9485 row, nodules under 10 mm score 0.818.
- **Types of nodule.** Scores are not split by solid, part-solid and ground-glass nodules, so how it does on each is not known.
- **People.** Scores are not split by age, sex or ethnicity. Nobody has checked for differences between groups.
- **No growth over time.** It reads one scan at a time. It does not compare a scan with an earlier one.

## Licence

The weights and the code are under the **PolyForm Noncommercial License 1.0.0** ([LICENSE](../../LICENSE)).

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

That line travels with every copy. Commercial use needs a separate licence from LunarWerx. Downloads are gated: you accept the licence before you download the weights. The datasets above keep their own licences and credits, which travel with the weights as well. Selling a product that uses this model for diagnosis would also need regulatory clearance first. See [LICENSING.md](../../LICENSING.md) for the third-party list.

## How to run it

There is no single command yet that takes one new scan and returns a read. The pieces below are the ones that made the scores above.

They read scans laid out as LUNA16 and LIDC-IDRI are laid out. They need LUNA16's annotation and candidate files in the datasets folder. They look for the weights under `$RAIDIOLOGY_WORK/eval/lung/<name>/fold<k>.pt` (`RAIDIOLOGY_WORK` and `RAIDIOLOGY_DATASETS` are read in `core/paths.py`), so put the downloaded folders `det-v1`, `fpr-v2`, `fpr-v3` and `malignancy-v5` there. `nodule_fpr.py rescore` trains a fold itself when no saved net matches the settings and inputs it is given, so give it the settings the nets were saved with.

1. Make the 1 mm lung pictures: `python modalities/ct/eval/lung_prepare.py build`
2. Finder: `python modalities/ct/eval/nodule_det.py read --version 1`
3. Blob places: `python modalities/ct/eval/nodule_blobs.py build`, then `join` to merge them with the finder's places.
4. Second look: `python modalities/ct/eval/nodule_fpr.py rescore --version 2 ...` and `--version 3 --set windows=3 ...`, both with `--set tta=8 --set tta_top=32`, run on the joined list (`--stage1-file`).
5. Average the two: `python tools/stage2_blend.py A.csv B.csv OUT.csv`
6. Lung step: `python tools/lung_inside.py filter --candidates OUT.csv --d-file <file holding 14.55>`
7. Re-scorer: `python tools/lung_stack.py run --candidates <filtered list> --out <file> --v2 A.csv --v3 B.csv`
8. Cancer or not: `python modalities/ct/eval/nodule_malignancy.py lungx --version 5 --set calibration` scores the nodules in a table that `modalities/ct/eval/lungx.py` writes (a scan, a centre in millimetres and a label for each), using the average of the five copies' scores.

The soft rule that follows step 7 is not in this repository yet, so the last step of the 88.0 system cannot be rerun from here. No time or cost per scan has been measured for this chain.
