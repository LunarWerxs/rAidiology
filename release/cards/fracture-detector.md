# Model card: rAidiology X-ray fracture detector v140 + v141

**Not a medical device. Not for diagnosis or treatment decisions.**

## What it does

You give it one X-ray picture and say which kind of picture it is: a wrist, or a hand, leg, hip or shoulder. It answers one question: does this radiograph show a fracture?

The answer is a score from 0 to 1 and a yes or no. A higher score means a fracture is more likely. The yes or no comes from a cut-off that was fitted on development data only: 0.5208 for the wrist profile and 0.1261 for the other-bones profile. The two cut-offs differ because the two collections differ: about 18 in 100 FracAtlas pictures show a fracture, against about 67 in 100 of the wrist pictures.

It does not say where the fracture is, what kind it is, or how bad it is.

The model is two sets of five small convolutional networks. Both sets use the same recipe and differ only in the random seed (v140 uses seed 0, v141 uses seed 1). A picture's score is the average of the two sets' scores. Each set was trained by 5-fold cross-validation on the development pictures, so each set holds five networks.

The two sets were chosen on development data, before the held-out test was read again. On development pictures, read out of fold, the pair scored higher than version 1 (the first version of this detector):

- FracAtlas: AUROC difference +0.0300 (95% CI +0.0182 to +0.0429), against a seed-to-seed noise of 0.0106.
- GRAZPEDWRI-DX: AUROC difference +0.0091 (95% CI +0.0072 to +0.0109), against a seed-to-seed noise of 0.0030.

An older version, version 1, has its own card: [fracture-detector.md](https://github.com/LunarWerxs/rAidiology/blob/main/modalities/xray/models/fracture-detector.md).

## What it is not for

- **Diagnosis, triage or screening.** A radiologist's report decides. Do not use it to decide who is seen first, who is sent home, or what treatment someone gets.
- **A medical device.** It is not cleared or approved by the FDA or any other regulator. A commercial licence does not change that.
- **Children, other than wrists.** The children's pictures it learned from and was tested on are wrists. Other children's bones are not tested.
- **Any age group on the other-bones pictures.** FracAtlas gives hand, leg, hip and shoulder pictures. FracAtlas publishes no ages or sexes, so this card makes no claim for an age group on them.
- **Bones neither collection covers**, such as the skull, spine, ribs or pelvis.
- **Anything that is not a plain radiograph.** No CT, MRI or ultrasound, and no photograph of a screen.
- **Other scanners or views.** It was only tested on pictures from the two collections below. It has not been tested on pictures from other hospitals or other scanners.

## How well it does

All scores below are on held-out test pictures, unless a row says otherwise. These are 20% of the patients (GRAZPEDWRI-DX) or 20% of the pictures (FracAtlas), set aside before any model was trained and never used for training or for any choice about the model.

Words used:

- **AUROC** runs from 0.5 (a coin) to 1.0 (perfect).
- **Catches** is sensitivity: of 100 pictures that show a fracture, how many it flags.
- **Clears** is specificity: of 100 pictures that show none, how many it leaves unflagged.
- **First visit** is the dataset's `initial_exam` flag. It marks the first exam of an injury, not a child's first visit ever.
- **95% CI** is a 95% percentile bootstrap interval from 2,000 resamples of the patient (GRAZPEDWRI-DX) or the picture (FracAtlas).

### Against the numbers other people have published

The published numbers come from other pictures and other methods, so none of these is a head-to-head test. "Not like for like" says so.

| Measure | Ours | Test set | Published number | Held by | Same pictures? |
|---|---|---|---|---|---|
| Children's wrists, overall skill (AUROC), pictures with a certain diagnosis | 0.979 (0.974 to 0.984) | GRAZPEDWRI-DX held-out test, 4,067 pictures of 1,176 patients | 0.940 | EfficientNet-B4, 10-fold cross-validation on 4,588 pictures of this dataset, no patient split described (Eur Radiol 2025) | Same dataset, other split. **Ahead.** |
| Mixed bones, overall skill (AUROC) | 0.918 (0.888 to 0.945) | FracAtlas held-out test, 798 pictures | 0.833 | FAD-MIL on FracAtlas (Sci Rep 2026) | Same dataset, other split (ours is split by picture). **Ahead.** |
| Mixed bones, breaks caught when clearing as many healthy as FAD-MIL (93 of 100) | 76 caught while clearing 94 on test; 68 on development folds | FracAtlas held-out test, and development folds (out of fold) | 49 | FAD-MIL on FracAtlas (Sci Rep 2026) | Same dataset, other split. **Ahead.** No interval in the source. |
| Children's wrists, first visit: breaks caught when raising as few false alarms as an emergency doctor | 93.5 | GRAZPEDWRI-DX first visits, development folds (out of fold) | 81 | Emergency clinicians (doctors and physician assistants) reading wrist films at a US orthopaedic hospital, ages not stated (Lindsey, PNAS 2018) | No, and not children. **Ahead, not like for like.** No interval in the source. |
| Children's wrists, first visit: healthy cleared when catching as many as an emergency doctor | 98.8 | GRAZPEDWRI-DX first visits, development folds (out of fold) | 87.5 | The same emergency clinicians | No, and not children. **Ahead, not like for like.** No interval in the source. |
| Children's limbs: breaks caught | 88 on first visits (0.882, 0.857 to 0.907); 94 on all pictures (0.942, 0.929 to 0.953) | GRAZPEDWRI-DX held-out test, 2,172 first-visit and 4,170 pictures | 98 | Paediatric radiologists, every limb bone, not only the wrist (Gasmi, Pediatric Radiology 2023) | No. **Behind.** |
| Children's forearms: caught and cleared, both at once | 94 and 93 (0.942, 0.929 to 0.953 and 0.934, 0.917 to 0.949) | GRAZPEDWRI-DX held-out test, all 4,170 wrist pictures | 93 and 98 | BoneView, a sold product, against a children's-radiology consensus (Altmann-Schneider, Pediatric Radiology 2024) | No. **Behind on clearing.** |
| Mixed bones, breaks caught when raising as few false alarms as an emergency doctor | 76 | FracAtlas development folds (out of fold) | 81 | Emergency clinicians, on wrists only (Lindsey, PNAS 2018) | No, and wrists only. **Behind, not like for like.** No interval in the source. |

For scale, on the same held-out test a model that sees no pixels scores AUROC 0.69 on the children's wrists (from age, sex, side, projection and scanner maker) and 0.68 on FracAtlas (from body part and view only; FracAtlas publishes no age or sex). Classical image statistics score 0.84 on the children's wrists and 0.75 on FracAtlas.

### The held-out test numbers behind it

The pair was scored at the cut-offs above. Version 1 is shown beside it. Each cell is the value and its 95% CI.

| Test set | Measure | v140 + v141 (second look, 2026-10-07) | Version 1 (first look, 2026-10-01) |
|---|---|---|---|
| GRAZPEDWRI-DX, all pictures (4,170 pictures, 1,218 patients) | AUROC | 0.978 (0.972 to 0.982) | 0.976 (0.970 to 0.981) |
| | Catches | 0.942 (0.929 to 0.953) | 0.927 (0.912 to 0.941) |
| | Clears | 0.934 (0.917 to 0.949) | 0.944 (0.924 to 0.961) |
| GRAZPEDWRI-DX, first visits (2,172 pictures, 990 patients) | AUROC | 0.969 (0.960 to 0.978) | 0.967 (0.957 to 0.976) |
| | Catches | 0.882 (0.857 to 0.907) | 0.839 (0.810 to 0.868) |
| | Clears | 0.973 (0.963 to 0.982) | 0.985 (0.978 to 0.991) |
| FracAtlas, all readable pictures (798 pictures) | AUROC | 0.918 (0.888 to 0.945) | 0.893 (0.859 to 0.924) |
| | Catches | 0.817 (0.750 to 0.881) | 0.810 (0.745 to 0.872) |
| | Clears | 0.832 (0.803 to 0.861) | 0.835 (0.808 to 0.863) |

The change from the first look to the second look, on the pictures both scored, is on the [scorecard](https://github.com/LunarWerxs/rAidiology/blob/main/modalities/xray/eval/SCORECARD.md) (section "Second look 2"):

- FracAtlas AUROC: +0.0256 (+0.0093 to +0.0415). This is a clear gain.
- GRAZPEDWRI-DX AUROC: +0.0016 (-0.0005 to +0.0038). The interval includes zero, so on wrists the overall skill did not change in any way the test can show.
- GRAZPEDWRI-DX first visits, catches: +0.0431 (+0.0292 to +0.0584). Clears: -0.0120 (-0.0205 to -0.0041). So the pair flags more fractures on first visits and raises more false alarms.

The held-out test has now been read twice: version 1 on 2026-10-01, and this pair on 2026-10-07 (labelled a second look on the scorecard). No test number chose the versions, the cut-offs or the comparison. They were all fixed before the second read.

The full scorecard, with every number defined in [METRICS_SPEC.md](https://github.com/LunarWerxs/rAidiology/blob/main/modalities/xray/eval/METRICS_SPEC.md), is [modalities/xray/eval/SCORECARD.md](https://github.com/LunarWerxs/rAidiology/blob/main/modalities/xray/eval/SCORECARD.md). The absolute second-look scores are given in the table above, not in the scorecard, which holds the first look and the change from it.

## Data it learned from

| Collection | Role | Licence | Commercial use allowed? |
|---|---|---|---|
| GRAZPEDWRI-DX v2 (20,327 paediatric wrist radiographs of 6,091 patients, labelled by paediatric radiologists) | Training: 16,157 development pictures of 4,873 patients. Cut-offs fitted on these, out of fold. Test only: 4,170 pictures of 1,218 patients, never trained on | CC BY 4.0 | Yes, with credit |
| FracAtlas v7 (4,083 musculoskeletal radiographs in its index: hand, leg, hip, shoulder; fracture yes or no, with boxes and masks, from the dataset's authors) | Training: 3,268 development pictures, of which 3,226 could be read. Cut-offs fitted on these, out of fold. Test only: 815 pictures, of which 798 could be read, never trained on | CC BY 4.0 | Yes, with credit |
| NIH ChestX-ray14 (112,120 frontal chest X-rays of 30,805 patients) | Pretraining only. The two sets of networks start from the convolutional trunk of one network of rAidiology's own chest reader. That reader was trained from scratch on pictures from NIH's training and validation list (about 86,000 pictures in all) with NIH's report-mined labels, which NIH says are over 90% accurate. NIH's own test list was not opened. No chest picture was used to train, tune or test the fracture answer | NIH states the images carry no restriction; credit asked | Yes, with credit |

No other collection was used. No ImageNet weights and no other outside pretrained weights were used. The chest reader itself was trained from scratch.

Google's expert labels for 4,374 of NIH's pictures were not used. Their terms allow research and education only.

### Credit these licences require

CC BY 4.0 asks you to give credit, link the licence and say what was changed. Each picture's grey levels were windowed to the 0.5th to 99.5th percentiles of the pixels above its own 1st percentile, and the picture was scaled so its longer side is 384 pixels and centred on a black 384 by 384 square, without stretching. No picture is included with the weights.

- **GRAZPEDWRI-DX**: Nagy, Janisch, Hrzic, Sorantin, Tschauner, "A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for machine learning", Scientific Data 9, 222 (2022), doi:10.1038/s41597-022-01328-z; data doi:10.6084/m9.figshare.14825193, CC BY 4.0.
- **FracAtlas**: Abedeen, Rahman, Prottyasha, Ahmed, Chowdhury, Shatabda, "FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs", Scientific Data 10, 521 (2023), doi:10.1038/s41597-023-02432-4; data doi:10.6084/m9.figshare.22363012, CC BY 4.0.
- **NIH ChestX-ray14**: link to NIH's download site (https://nihcc.app.box.com/v/ChestXray-NIHCC), cite Wang, Peng, Lu, Lu, Bagheri, Summers, "ChestX-ray8", IEEE CVPR 2017, pp. 3462-3471, and acknowledge the NIH Clinical Center as the data provider.

### How it was trained

- Five stages of two convolutions each (widths 32, 64, 128, 256 and 256; batch norm, ReLU, max-pool), a stride-2 stem, dropout 0.3 and one linear output. Each network is 9,445,016 bytes.
- 25 epochs, batch 64, learning rate 0.0003, weight decay 0.0001. Augmentation: left-right flip, rotation up to 10 degrees, shift up to 0.05, and a brightness and contrast change (jitter 0.1). No early-stopping set (`val_share` 0).
- Started from the trunk of one network of the chest reader (its fold 0); all 60 tensors fit. Seed 0 for v140, seed 1 for v141. Trained on 2026-10-06.
- Split: seed 20261001, 20% of the units held out as test, the rest in 5 folds, written once and never changed (`modalities/xray/eval/split.json`). The unit is the patient for GRAZPEDWRI-DX and the picture for FracAtlas.
- Fold k's network never saw the pictures in fold k. Every development score in this card is from the network that never saw that picture.

## Limitations and known failure modes

- **Behind the specialists.** On children's limbs it catches 88 to 94 of 100 where paediatric radiologists catch 98. On forearms it clears 93 of 100 where BoneView clears 98. On mixed bones it catches 76 of 100 at an emergency doctor's false-alarm rate, where the doctors catch 81 (wrist films, not the same pictures).
- **Mostly children's wrists.** 16,157 of the 19,383 pictures it was trained on for fractures are GRAZPEDWRI-DX wrists; the rest are FracAtlas. Its starting point was learned from NIH chest X-rays, not bones. FracAtlas has only 338 hip pictures in total.
- **Shoulders are weak.** On FracAtlas shoulders (65 test pictures) it catches 0.417 of the fractures (0.143 to 0.714), with AUROC 0.692 (0.494 to 0.863). That is fewer than half, from very few pictures, so the true figure is uncertain. It clears 0.755 of healthy shoulders (0.636 to 0.869), fewer than version 1: a change of -0.0943 (-0.1800 to -0.0196) on the same pictures.
- **Hands raise many false alarms.** On FracAtlas hands (299 test pictures) it clears 0.645 (0.579 to 0.705), so roughly a third of healthy hands are flagged. It catches 0.852 (0.773 to 0.926) of the fractured ones.
- **Other bones by part (FracAtlas test).** Leg: 445 pictures, AUROC 0.922 (0.877 to 0.961), catches 0.769 (0.652 to 0.879), clears 0.924 (0.897 to 0.948). Hip: 67 pictures, AUROC 0.933 (0.856 to 0.988), catches 0.833 (0.600 to 1.000), clears 0.818 (0.709 to 0.917). The hip and shoulder intervals are wide because there are few pictures.
- **First visits are harder at a fixed cut-off.** On GRAZPEDWRI-DX first visits it catches 0.882 (0.857 to 0.907), against 0.942 (0.929 to 0.953) on all pictures. First visits are the setting a clinic cares about, and the later check-ups are easier.
- **Shortcuts.** 5,776 GRAZPEDWRI-DX pictures show a plaster cast, which almost always means a fracture was already found. Only 10,861 of the 20,327 pictures are first exams. 537 are flagged "diagnosis uncertain", and none were thrown away. To check that the score is not just reading the cast, the same scores were taken on three subsets of the held-out test. AUROC on first visits only is 0.969 (0.960 to 0.978), with no cast 0.973 (0.966 to 0.980), and with a certain diagnosis 0.979 (0.974 to 0.984), against 0.978 (0.972 to 0.982) on all pictures. The score does not collapse without the cast.
- **FracAtlas has no patient number.** It is split by picture, so two pictures of the same person could land on both sides. That could make the FracAtlas numbers a little high. A check on development folds found this adds about 0 to AUROC, and at most +0.02. This is why the ship decision rests on GRAZPEDWRI-DX, where the split is by patient.
- **Pictures that could not be read.** 42 development and 17 test FracAtlas files are cut short on disk. They are left out of the scores above.
- **The same picture can score slightly differently on different machines.** A processor and a graphics card differed by up to 0.0090 on a score from 0 to 1 over twenty held-out radiographs. On version 1's held-out test, 6 of 4,170 wrist pictures (0.14%) and 15 of 815 FracAtlas pictures (1.84%) were that close to the cut-off, so the kind of machine could flip their answer. It was not re-measured for v140 and v141.
- **The test is not fresh.** It has been read twice (see above). The second read is labelled and nothing was tuned on it, but it is not a first look.
- **The labels are the labellers' labels.** The model learns what the paediatric radiologists (GRAZPEDWRI-DX) and the FracAtlas authors marked. It cannot be better than their marks.

## Licence

The weights and the code are under the PolyForm Noncommercial License 1.0.0 (see [LICENSE](LICENSE)). The notice below travels with every copy:

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

Free use is for personal study and research and for noncommercial organisations, as the licence says. Commercial use needs a separate licence from LunarWerx. Contact LunarWerx at https://github.com/LunarWerxs.

Downloads are gated: you accept the licence before you can download the weights.

The three collections above each allow commercial use with credit, so the data does not stand in the way of a commercial licence. Selling this as a diagnostic tool is a different matter and needs regulatory clearance first. The dataset licence ledger is [LICENSING.md](https://github.com/LunarWerxs/rAidiology/blob/main/LICENSING.md).

## How to run it

Read one picture with one of the two sets:

```
python modalities/xray/eval/detector.py image PATH_TO_RADIOGRAPH --version 140 --profile grazpedwri
python modalities/xray/eval/detector.py image PATH_TO_RADIOGRAPH --version 141 --profile grazpedwri
```

Use `--profile grazpedwri` for a wrist and `--profile fracatlas` for a hand, leg, hip or shoulder. Each call prints one line of JSON per picture with the score, the yes or no at that set's own cut-off, and the version, and then a "not a medical device" line.

To read with the pair, average the two scores and compare the average with the pair's cut-offs: 0.5208 (wrist) or 0.1261 (other bones). The repository has no one-step command for this yet. The way the pair was scored on the held-out test is `modalities/xray/eval/final_eval.py --second-look --versions 140,141`, which re-opens the test split and is for the project's own scoring, not for reading your pictures.

Files, for each of v140 and v141: `fold0.pt` to `fold4.pt` (9,445,016 bytes each) and `meta.json`. The two sets use the same file names, so keep them in two separate folders. The code looks for them under the repository: v140's files in `work/eval/xray/detector-v140/model/` (`work/eval/xray/detector-v140/model/fold0.pt` and so on) and v141's files in `work/eval/xray/detector-v141/model/` (`work/eval/xray/detector-v141/model/fold0.pt` and so on). Inside each folder the files must be called `fold0.pt` to `fold4.pt` and `meta.json`. `RAIDIOLOGY_WORK` moves `work/`. No ONNX graph has been exported for v140 or v141. The exporter, `tools/export_detector_onnx.py --version N`, takes a version number, but the ONNX reader (`modalities/xray/eval/onnx_reader.py`) looks only in the folder for released versions, which is version 1 today.
