# Model card: rAidiology Mammogram reader v29, v30, v34, v35 and v38 (read together)

**Not a medical device. Not for diagnosis or treatment decisions.**

The CMMD scores below were measured on 2026-10-10. The CBIS-DDSM test score was measured on 2026-10-05.

## What it does

You give it one mammogram picture of one breast, one view (CC or MLO), where a lesion has already been found and sent for biopsy. It gives back one score between 0 and 1. A higher score means the reader leans more toward "malignant" and less toward "benign". The score ranks pictures. Nobody has checked that a score of 0.8 means an 80% chance.

The reader is five versions whose scores are averaged, each counting the same. Each version is five small convolutional networks, one for each fold of patients it was trained on, so there are 25 networks in the read, plus 5 more in a helper called the finder (version 6).

- Versions 29 and 30 read the whole picture. The breast is cut out, turned so the chest wall is on the left, and shrunk to 1024 pixels tall and 640 wide.
- Versions 34, 35 and 38 read three close-up crops, each 512 by 512 pixels at full detail. The finder (version 6) picks where the three crops are cut. A picture's score is the highest of its three crop scores.

For a whole breast, average the scores of its pictures. That is how the per-breast figure below is made.

Every network was trained from scratch or from one of our own networks. No ImageNet weights and no other outside weights are used anywhere in the chain (see "Data it learned from").

## What it is not for

- **Diagnosis.** It does not tell you what is wrong with anyone.
- **Decisions about a patient or about your own health.** Talk to a doctor about any scan.
- **Screening or triage.** It was trained and scored only on breasts that already had a lesion and a biopsy. In that data 2,632 of 3,744 pictures are malignant. Screening pictures are mostly healthy, and that case was never tested. Do not use it to decide who is called back or who is cleared.
- **Finding a lesion.** It does not say whether a lesion is there. It scores a lesion that someone already found.
- **People outside the data.** The patients in the data are aged 17 to 87. Nothing here tests anyone under 17. The clinical table lists age but not sex, so men are untested.
- **Other kinds of picture.** Tomosynthesis (3D mammography), contrast-enhanced mammography, ultrasound, MRI and photos of film are untested. Scanned film is untested for this five-version read.
- **Other scanners, hospitals and countries.** This read was scored only on one collection, CMMD, from China. Nothing here shows how it behaves anywhere else.
- **Claims that it is as good as a doctor.** It has not been compared with radiologists reading the same pictures.

## How well it does

The score is **AUROC**. It is the chance that a malignant picture gets a higher score than a benign one. 0.5 is a coin toss and 1.0 is perfect. The **95% interval** is the range the true score probably falls in. It comes from resampling patients 2,000 times, so one patient's pictures never get split.

| Measure | Ours (95% interval) | Test set | Best published, and who holds it | Same pictures? |
|---|---|---|---|---|
| CMMD, cancer or benign, per picture, five-version read | **0.793** (0.774 to 0.811); to four places 0.7925 (0.7739 to 0.8114) | CMMD: 3,744 pictures of 1,872 breasts of 1,775 patients, five folds split by patient. Each picture is read by networks that never trained on it. This is a development score, not a held-out test opened once. | 0.885 (0.849 to 0.918): Takahashi et al., J Imaging Inform Med 2025, doi 10.1007/s10278-025-01773-3, a vision transformer whose attention is masked to the breast outline, on 2,436 CMMD pictures in three folds split 70/30 by patient. Also 0.848 (give or take 0.015): Bobowicz et al., Cancers 2023, 15:2704, which starts from ImageNet weights and from a transfer from the authors' own private mammograms. With ImageNet weights alone the same paper reads 0.772. | No. Same collection, other split and other picture count. |
| The same, per breast (the average of a breast's pictures) | **0.8147** (0.7945 to 0.8346) | Same | No per-breast number found. Bobowicz et al. do not say whether their figure is per picture or per breast. | Not comparable |
| Version 38 alone, per picture | 0.7920 (0.7744 to 0.8100) | Same | Same as the first row | Same as the first row |
| Version 38 alone, per breast (the average of a breast's pictures) | 0.8209 (0.8017 to 0.8401) | Same | Same as the per-breast row above | Not comparable |
| CBIS-DDSM, mass benign or malignant, per lesion and view (a separate reader, version 3 of the mass reader, trained from scratch for 200 epochs) | **0.829** (0.775 to 0.873) | CBIS-DDSM's official test list, 378 masses, opened once | 0.804 with an ImageNet start and 0.716 from scratch: Tsochatzidis, Costaridou and Pratikakis, J Imaging 2019, 5:37, doi 10.3390/jimaging5030037, best of eight networks | Yes, the same 378 test masses. We leave out of training the 31 patients that appear in both of CBIS-DDSM's lists. Their 0.804 started from ImageNet, which ours does not use. |

Where it stands:

- **CMMD: behind.** 0.793 is below both 0.848 and 0.885. It is 0.055 under 0.848. The whole interval, 0.774 to 0.811, sits under both.
- **CMMD against ImageNet only: above, but not like for like.** Bobowicz et al.'s 0.772 with ImageNet alone sits under our interval. Their net started from ImageNet weights. Ours started from none.
- **CBIS-DDSM masses: above the from-scratch score, level with the ImageNet one.** 0.829 against 0.716, and 0.804 sits inside our interval. This row is a separate reader. It is not the five-version read, and the test list is now used up, so a newer CBIS-DDSM reader has no test score.
- **The five versions add almost nothing to version 38 alone.** Per picture the five versions add 0.0005 over version 38 alone (0.7925 against 0.7920). Per breast, version 38 alone scores higher (0.8209 against 0.8147).
- **No doctor comparison.** The "best published" numbers are other computer models.

Scores by group, five-version read, same out-of-fold scores (the groups are small, so their intervals are wide, and we did not compute them):

| Group | Pictures | AUROC |
|---|---|---|
| Lesion is a mass | 2,298 | 0.7371 |
| Lesion is calcifications | 524 | 0.8424 |
| Mass and calcifications | 922 | 0.8797 |
| CC view | 1,872 | 0.7964 |
| MLO view | 1,872 | 0.7899 |
| Age 40 or under | 944 | 0.7462 |
| Age 41 to 46 | 944 | 0.7469 |
| Age 47 to 54 | 952 | 0.7749 |
| Age over 54 | 904 | 0.8043 |

## Data it learned from

The chain, in order. Each step starts from the network before it.

1. A **chest X-ray reader** (our version 103) trained on NIH ChestX-ray14.
2. A **CBIS-DDSM mass reader** (our version 5) starts from step 1 and trains on CBIS-DDSM masses.
3. An **OMAMA-DB reader** (our version 1) starts from step 2. It learns cancer or not from 26,519 OMAMA-DB screening pictures, minus one fifth (5,290 pictures) held back to check it. A second one, the **OMAMA-DB crop reader**, starts from it and learns the same task at the crop scale: three 512 by 512 crops per picture, from the same pictures.
4. The **finder** (version 6) starts from step 2 and trains on CMMD, with lesion outlines as a training-only extra.
5. **Versions 29 and 30** start from the OMAMA-DB reader and train on CMMD. **Version 32** (lesion crops) starts from it too, and **versions 34 and 35** start from version 32. **Version 37** (lesion crops) starts from the OMAMA-DB crop reader, and **version 38** starts from version 37. Versions 34, 35 and 38 train on CMMD's pictures cut at the finder's three places.

Each CMMD network is scored on one fifth of the patients. Those patients are never in its training. It trains on the other four fifths, minus a tenth of those patients kept aside to pick when to stop.

| Dataset | Used for | Licence | Commercial use allowed? |
|---|---|---|---|
| NIH ChestX-ray14 (112,120 chest X-rays) | Training the chest reader that starts the chain. Its labels were mined from radiology reports by software and NIH says over 90% are accurate. Google's expert labels were not used. | NIH says use is unrestricted if you credit it | Yes, with the credit below |
| CBIS-DDSM (1,566 participants, scanned film mammograms with pathology-verified lesions) | Training the mass reader (official training masses only). Separately, its official test list was opened once to score the standalone mass reader (version 3), which is not part of the five-version read. | CC BY 3.0 | Yes, with the credit below |
| OMAMA-DB (Harvard Dataverse; about 158,000 patients, 7,351 with pathology-confirmed cancer) | Training the OMAMA-DB reader and crop reader. Labels: cancer or not, confirmed by pathology. | MIT (the dataset record's licence) | Yes, keep the MIT notice |
| CMMD (1,775 patients, mammograms with biopsy-confirmed benign or malignant) | Training and scoring everything in the read, in five folds split by patient. | CC BY 4.0 | Yes, with the credit below |
| TOMPEI-CMMD (lesion outlines, called segmentations, on 1,385 MLO breasts of CMMD) | Training only. A side task that draws the outline helps the networks learn, and the outlines place the lesion crops used for versions 32 and 37. The outlines are never used when reading a picture. | CC BY 4.0 | Yes, with the credit below |

**Used only for testing:** none of these sets, for the five-version read. CMMD is both training and test data, kept apart by patient. CBIS-DDSM's official test list (378 masses) is used only for testing, once, for the standalone mass reader.

**Not used:** ImageNet weights or any other outside weights (ImageNet's terms are non-commercial). CBIS-DDSM and CDD-CESM pictures were tried as extra CMMD training pictures. That made scores worse, so no network in this read uses them.

The credit each licence asks for:

- **NIH ChestX-ray14.** Link NIH's download site, https://nihcc.app.box.com/v/ChestXray-NIHCC. Cite Wang, Peng, Lu, Lu, Bagheri and Summers, "ChestX-ray8", IEEE CVPR 2017, pages 3462 to 3471. Acknowledge the NIH Clinical Center as the data provider.
- **CBIS-DDSM** (CC BY 3.0, from The Cancer Imaging Archive): https://www.cancerimagingarchive.net/collection/cbis-ddsm/. Credit the collection and cite it exactly as that page asks.
- **OMAMA-DB** (MIT): doi:10.7910/DVN/KXJCIU on Harvard Dataverse, https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KXJCIU. Keep the record's copyright and permission notice with the weights, and cite the record as its page asks.
- **CMMD** (CC BY 4.0, from The Cancer Imaging Archive): https://www.cancerimagingarchive.net/collection/cmmd/. Credit the collection and cite it exactly as that page asks.
- **TOMPEI-CMMD** (CC BY 4.0, from The Cancer Imaging Archive): https://www.cancerimagingarchive.net/analysis-result/tompei-cmmd/. Credit the collection and cite it exactly as that page asks.

Anyone who shares these weights must pass these credits on with them.

## Limitations and known failure modes

- **A development score only.** Every CMMD number comes from CMMD itself. There is no held-out test and no second hospital. The recipe was tuned over many rounds on these same out-of-fold scores, and a round was kept when its gain beat the noise on the same pictures. The number may be a little high for new pictures.
- **It scores lesions that were already biopsied.** The mix is 2,632 malignant pictures to 1,112 benign. A different mix, such as screening, changes what a score means. It was never tested.
- **Masses are the weak spot.** Masses score 0.7371 against 0.8424 for calcifications. Masses are 2,298 of the 3,744 pictures.
- **It is close to a coin toss where no lesion outline exists.** TOMPEI-CMMD has no outline for 1,002 of the pictures. On those, version 38 scores 0.5726 and the other four versions score between 0.4792 and 0.5420. The cause is not settled.
- **Whether TOMPEI-CMMD outlined a breast is linked to the answer.** Outlined breasts are mostly cancers (82 to 91 in 100, by lesion type), the rest mostly not (28 to 41 in 100). That fact alone scores about 0.757. Part of the score may come from lesions that are easy to see, and pictures like the unoutlined ones read near chance.
- **Age leans on the score.** Age alone scores 0.6813 on these pictures. Version 38's score and age correlate at 0.265. The read does better in older patients (0.8043 over age 54) than in the youngest two groups (0.7462 and 0.7469). Adding age to the read scores 0.8045 out of fold. That is reported only and is not part of the reader.
- **No tested cut-off.** We report AUROC only. We have not measured sensitivity or specificity at a chosen cut-off, or whether scores are calibrated.
- **The shipped read is 25 fold networks, not one final network.** Each network saw four fifths of CMMD's patients. Averaging all five networks of a version on a new picture is the natural way to use them, but that exact use is not what was scored. The score above has one network per version reading each picture.
- **It needs the finder.** Versions 34, 35 and 38 need the finder's three places. If the finder points at the wrong place, those versions read the wrong crop.
- **Picture preparation is simple.** The breast is found with a brightness threshold, and the picture is cut and flipped from that. Pictures that look unlike CMMD's (other vendors, film scans, markers or labels in the picture) were not tested at this step.
- **Behind the best published CMMD scores.** See the table.

## Licence

The weights and the code are under the **PolyForm Noncommercial License 1.0.0** ([LICENSE](../../LICENSE)). The notice `Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)` travels with every copy.

- Free (a plain reading; the LICENSE text is what binds): personal study, research, and use by charities, schools, public research and health organisations, and governments.
- Commercial use needs a separate licence from LunarWerx. That includes clinics and companies using it in paid work, and any product or hosted service built on it.
- Downloads are gated: you accept the licence before you download.
- The data credits above also travel with the weights.

More detail is in [LICENSING.md](../../LICENSING.md). The licence does not change the first line of this card: this is not a medical device.

## How to run it

The code is in [modalities/mammo/eval/mammo.py](../../modalities/mammo/eval/mammo.py). It needs Python with PyTorch and the CMMD files from The Cancer Imaging Archive, placed in `Datasets/public/cmmd` beside the repository folder.

Put the downloaded files where the code looks for them: `work/eval/mammo/mammo-v<N>/fold<k>.pt`, with N one of 29, 30, 34, 35, 38 and 6, and k from 0 to 4. Each file is about 9.4 MB. The list:

| Version | Files | Bytes each |
|---|---|---|
| 29 | fold0.pt to fold4.pt | 9,442,318 |
| 30 | fold0.pt to fold4.pt | 9,442,318 |
| 34 | fold0.pt to fold4.pt | 9,441,304 |
| 35 | fold0.pt to fold4.pt | 9,442,318 |
| 38 | fold0.pt to fold4.pt | 9,441,240 |
| 6 (finder) | fold0.pt to fold4.pt | 9,707,376 |

That is 30 files and 284,584,370 bytes in all.

To work out the score again from per-picture scores:

```
python modalities/mammo/eval/mammo.py gate
python modalities/mammo/eval/mammo.py prepare
python modalities/mammo/eval/mammo.py read --versions 29,30,38,34,35
```

`read` prints the per-picture and per-breast AUROC with the 95% interval. `gate` and `prepare` build the CMMD index and pictures it reads from. `read` also needs each version's `oof.csv`, the per-picture scores that training writes into that version's folder. They are not part of the weights download. Rebuilding them means retraining the whole chain above, with every dataset listed. Each version's settings are stored inside its `fold<k>.pt`.

To load one network in your own code, call `load_fold_net(version, k, dev)` in `mammo.py`. It returns the network, the mean and the standard deviation to normalise pixels with. `window_to_u8` and `breast_crop` in the same file prepare a picture the way the reader expects.

**Not written yet:** a single command that takes a new mammogram and returns the five-version score. These weights are also not exported to ONNX, so they do not run in a web browser yet.

## To finish before this card is published

Delete this section once both items are done.

- **Read OMAMA-DB's other terms.** The dataset licence search flagged OMAMA-DB for a narrower statement next to its MIT record (see LICENSING.md, the section on sets settled by the download record's licence). It was cleared by taking the record's licence as the one that governs. Each download also answered a Harvard Dataverse guestbook. The "Yes" for commercial use in the data table rests on that choice, not on there being no other terms. Read that statement and the guestbook terms before selling a commercial licence.
- **Paste in the credit text.** The credit list above says where each citation lives, but it does not hold the text. CC BY asks for credit, and MIT asks for its notice to travel with the weights. Paste the citation lines from the CMMD, CBIS-DDSM and TOMPEI-CMMD pages at The Cancer Imaging Archive, and OMAMA-DB's MIT copyright line, into this card. Until then a copy of the weights would not carry the credit each licence requires.
