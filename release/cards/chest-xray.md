# Model card: rAidiology Chest X-ray reader v216

**Not a medical device. Not for diagnosis or treatment decisions.**

## What it does

You give it one frontal chest X-ray. It gives back 15 scores. Fourteen are the findings on the NIH ChestX-ray14 list: atelectasis, cardiomegaly (enlarged heart), effusion (fluid around the lung), infiltration, mass, nodule, pneumonia, pneumothorax (collapsed lung), consolidation, edema (fluid in the lung), emphysema, fibrosis, pleural thickening and hernia. The fifteenth, "Abnormal", is true for any picture that NIH did not label "No Finding". A higher score means the model rates that finding as more likely. The scores rank pictures. No cut-off has been chosen or tested.

Version 216 is a team of ten networks whose scores are averaged, each network counting the same. Five are convolutional networks (version 118, about 8 million numbers each, reading the picture at 384 by 384 pixels). Five are vision transformers (version 215, the ViT-S/16 design, about 21 million numbers each, reading it at 224 by 224). Each network learned from about four fifths of NIH's development patients (between 62,055 and 62,745 pictures, plus 6,637 to 7,093 pictures used to decide when to stop). Each network left out a different fifth of the patients (17,064 to 17,712 pictures), and that fifth is where its development score comes from. The five convolutional networks started from random numbers. The five transformers started from an encoder that was first taught on unlabelled chest X-rays (see "Data it learned from"), which itself started from random numbers. No network started from ImageNet or from any other outside model. The ten weight files add up to 589,209,450 bytes.

## What it is not for

- **Diagnosis.** It does not tell you what is wrong with anyone.
- **Decisions about a patient or about your own health.** Talk to a doctor about any scan.
- **Triage or screening decisions.** Do not use it to decide which scans are read first, who is called back or who is cleared. It has no tested cut-off, and a low score does not show that a finding is absent.
- **Children.** No score is split by age, so nobody has checked children. Adults have not been checked as a separate group either.
- **Other kinds of picture.** Side-view (lateral) X-rays, photos or phone pictures of film, X-rays of other body parts, CT and MRI are all untested.
- **Other hospitals and machines.** The labelled pictures it learned from and was scored on all come from one hospital. Nothing here shows how it behaves on other scanners or other patient groups.
- **Bedside X-rays of intensive care patients.** Some were used for pre-training without labels. It has not been scored on them.
- **Claims that it is as good as a doctor.** It has not been compared with radiologists reading the same pictures (see below).

## How well it does

The score is **AUROC**. It is the chance that a picture with the finding gets a higher score than a picture without it. 0.5 is a coin toss and 1.0 is perfect. The **95% interval** is the range the true score probably falls in. It comes from resampling patients 2,000 times.

The test is NIH's official test list: 25,596 pictures of 2,797 patients, where 61.5 of every 100 pictures are abnormal. The labels are NIH's own. The list was kept out of all training, including pre-training.

This list has been read three times, each time by a newer reader. The scores were 0.7965, then 0.7995, then 0.8175 for this model. A new read was allowed only after the new reader had done better on development patients. No test score was used to choose a model.

| Measure | Ours (95% interval) | Test set | Highest published score we found, and who holds it | Same pictures? |
|---|---|---|---|---|
| Mean AUROC of the 14 findings | **0.8175** (0.8110 to 0.8231) | NIH test list, 25,596 pictures | 0.830: a ViT-B/16 pre-trained by masked autoencoding on 0.5 million chest X-rays only (Xiao et al., WACV 2023, arXiv 2210.12843, Table 6) | Yes, the same list. **We are behind by 0.007, measured from the top of our interval.** |
| Mean AUROC of the 14 findings, against networks trained on NIH alone from random weights | **0.8175** (0.8110 to 0.8231) | NIH test list | 0.804: DenseNet-121 from random weights, 224 px, picture alone (Xiao et al., WACV 2023, Table 1). Also 0.806: Baltruschat et al., Scientific Reports 2019, 448 px with age, sex and view as extra inputs | Yes, the same list. **We are ahead of both.** Our low end, 0.8110, is above each. Not like for like: half our team was first pre-trained without labels on 254,987 chest X-rays, most of them from other hospitals. Without that pre-training, our earlier read of the same list was 0.7995 (0.793 to 0.806), level with 0.804. |
| Each of the 14 findings | 0.7071 (infiltration) to 0.9232 (hernia) | NIH test list | 0.734 (infiltration) to 0.985 (hernia): the average of nine radiologists (Rajpurkar et al., PLOS Medicine 2018, Table 1) | No. That study used 420 other NIH pictures with labels from a three-radiologist panel, and the panel's labels are private. The numbers cannot be compared. |
| "Abnormal" | 0.7363 (0.7241 to 0.7477) | NIH test list | No comparable number | Not applicable |
| Normal or not, 810 NIH pictures labelled by a majority of three radiologists | **Not scored** | None | Two US board-certified radiologists caught 87 of 100 abnormal pictures and cleared 89 of 100 normal ones, or caught 81 and cleared 94 (Nabulsi et al., Scientific Reports 2021, Table 3) | Not scored. The 810 labels come from Google under terms that allow research and education only. We did not use them. |
| Mean AUROC of the 14 findings, development patients | 0.8382 | 86,524 NIH development pictures, each read by networks that never trained on its patient | No published score | Not applicable. This is a development number, not a final one. |

Scores for each finding on the NIH test list:

| Finding | AUROC | 95% interval | Pictures with the finding |
|---|---|---|---|
| Atelectasis | 0.7747 | 0.7622 to 0.7878 | 3,279 |
| Cardiomegaly | 0.9016 | 0.8864 to 0.9142 | 1,069 |
| Effusion | 0.8388 | 0.8282 to 0.8487 | 4,658 |
| Infiltration | 0.7071 | 0.6961 to 0.7175 | 6,112 |
| Mass | 0.8235 | 0.8032 to 0.8424 | 1,748 |
| Nodule | 0.7544 | 0.7333 to 0.7743 | 1,623 |
| Pneumonia | 0.7267 | 0.7021 to 0.7497 | 555 |
| Pneumothorax | 0.8614 | 0.8447 to 0.8773 | 2,665 |
| Consolidation | 0.7610 | 0.7459 to 0.7750 | 1,815 |
| Edema | 0.8542 | 0.8383 to 0.8704 | 925 |
| Emphysema | 0.8960 | 0.8801 to 0.9096 | 1,093 |
| Fibrosis | 0.8382 | 0.8149 to 0.8608 | 435 |
| Pleural thickening | 0.7845 | 0.7661 to 0.8036 | 1,143 |
| Hernia | 0.9232 | 0.8776 to 0.9574 | 86 |
| Abnormal | 0.7363 | 0.7241 to 0.7477 | 15,735 |
| **Mean of the 14 findings** | **0.8175** | 0.8110 to 0.8231 | |

Where it is behind: the highest published score we found on this list is 0.830, and ours is 0.8175. The 0.830 model was pre-trained on 0.5 million chest X-rays. Our pre-training pool holds 254,987 pictures.

## Data it learned from

Two kinds of learning went into the weights.

1. **Pre-training, for the five vision transformers only.** A network learned to rebuild the hidden 90% of unlabelled chest X-rays (a masked autoencoder, 200 epochs, 254,987 pictures, no labels). Only pictures were used.
2. **Training on labels, for all ten networks.** Every network learned the 15 labels from NIH ChestX-ray14's development part, using the labels NIH made by reading the radiology reports with software.

| Dataset | Used for | Pictures | Licence | Commercial use allowed? |
|---|---|---|---|---|
| NIH ChestX-ray14, development part (train_val_list) | Training on labels (all ten networks). Also pictures for pre-training. | 86,524 | NIH calls use of the set unrestricted and asks for credit | Yes, with the credit below |
| TAIX-Ray, training part only (University Hospital Aachen, bedside intensive care X-rays) | Pre-training pictures only | 137,593 | CC BY 4.0 | Yes, with the credit below |
| COVID-19-NY-SBU (The Cancer Imaging Archive; Stony Brook University) | Pre-training pictures only | 13,634 | CC BY 4.0 | Yes, with the credit below |
| Tuberculosis chest X-ray sets (Kiran and Jabeen; three Mendeley Data sets) | Pre-training pictures only | 13,717 | CC BY 4.0 | Yes, with the credit below |
| BDCXR-3257 (children's chest X-rays from a hospital in Dhaka) | Pre-training pictures only | 3,257 | CC BY 4.0 (the figshare record). The download also carries the hospital's data use agreement, which allows non-commercial academic research only and forbids commercial exploitation, resale, licensing or integration into clinical decision-support systems without written authorization. LunarWerx follows the download record's CC BY 4.0 licence (LICENSING.md decision D26), and the commercial licence's dataset schedule discloses this. The agreement also asks for the acknowledgement below. | Yes, under the record's CC BY 4.0, with the credit and acknowledgement below |
| COVID-19-AR (The Cancer Imaging Archive) | Pre-training pictures only | 262 | CC BY 4.0 | Yes, with the credit below |

The pre-training counts add up to the pool's 254,987 pictures. We resized the pictures, and we did not use their labels.

**Used only for testing or checking, not to train these weights**

- **NIH ChestX-ray14, official test list.** 25,596 pictures. Used to score the model. Its pictures are not in the pre-training pool.
- **ChestX-Det** (Apache-2.0, commercial use allowed). Boxes and outlines drawn by three radiologists on 3,578 NIH pictures. Used only as a check on one fold during development. Credit: Lian et al., "A Structure-Aware Relation Network for Thoracic Diseases Detection and Segmentation", IEEE Transactions on Medical Imaging 2021.

**Looked at, not part of these weights**

- **CheXmask** (heart and lung outlines, CC BY 4.0). It trained a separate heart-and-lung outline model and some training trials that were not kept. None of that is in these weights.
- **Google's radiologist labels for NIH pictures.** Not used. Their terms allow research and education only.

**Credit that the licences ask for**

- **NIH ChestX-ray14.** Data provided by the NIH Clinical Center. Download site: https://nihcc.app.box.com/v/ChestXray-NIHCC. Please cite Wang, Peng, Lu, Lu, Bagheri and Summers, "ChestX-ray8", IEEE CVPR 2017, pp. 3462-3471.
- **TAIX-Ray** (CC BY 4.0, https://huggingface.co/datasets/TLAIM/TAIX-Ray). Truhn, D., Geiger, D., Siepmann, R., von der Stück, M. S., Bressem, K. K., Kather, J. N., Kuhl, C., Müller-Franzes, G., & Nebelung, S. (2026). A comprehensive bedside chest radiography dataset with structured, itemized and graded radiologic reports. Scientific Data, 13, 632. https://doi.org/10.1038/s41597-026-07271-7. We resized the pictures and used them without labels.
- **COVID-19-NY-SBU** (CC BY 4.0). The Cancer Imaging Archive's required data citation: Saltz, J., Saltz, M., Prasanna, P., Moffitt, R., Hajagos, J., Bremer, E., Balsamo, J., & Kurc, T. (2021). Stony Brook University COVID-19 Positive Cases [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.BBAG-2923. We resized the pictures and used them without labels.
- **COVID-19-AR** (CC BY 4.0). The Cancer Imaging Archive's required data citation: Desai, S., Baghal, A., Wongsurawat, T., Al-Shukri, S., Gates, K., Farmer, P., Rutherford, M., Blake, G.D., Nolan, T., Powell, T., Sexton, K., Bennett, W., Prior, F. (2020). Data from Chest Imaging with Clinical and Genomic Correlates Representing a Rural COVID-19 Positive Population [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2020.py71-5978. The paper the authors recommend: Desai, S., et al. (2020). Chest imaging representing a COVID-19 positive rural U.S. population. Scientific Data, 7(1), 414. https://doi.org/10.1038/s41597-020-00741-6. We resized the pictures and used them without labels.
- **Tuberculosis chest X-ray sets** (each CC BY 4.0). Kiran, S., & Jabeen, I. (2026). Dataset of Tuberculosis Chest X-rays Images (Version 3) [Dataset]. Mendeley Data. https://doi.org/10.17632/8j2g3csprk.3; Kiran, S., & Jabeen, I. (2026). Large-Scale Chest X-ray Dataset of 9,364 Normal (Healthy) Cases for Automated Lung Analysis [Dataset]. Mendeley Data. https://doi.org/10.17632/7kb49nhfbn.1; Kiran, S., Jabeen, I., & Mehmood, Q. (2026). Pneumonia Chest X-ray Dataset (2023-2024) [Dataset]. Mendeley Data. https://doi.org/10.17632/8cm4ffk7b4.1. We resized the pictures and used them without labels.
- **BDCXR-3257** (CC BY 4.0). Durjoy, S. H., Shikder, M. E., Bijoy, M. H. I., Shoib, M. M. H., Rahman, A., Hasan, M. Z., Ferdoucy, S. A., & Islam, S. M. M. (2026). BDCXR-3257: A Comprehensive Chest Radiography Dataset with Multi-Etiological Pneumonia Labels from a Clinical Setting. figshare. Dataset. https://doi.org/10.6084/m9.figshare.32086362.v1. Its data use agreement asks that work built on it say the radiographs were collected at Dr. M R Khan Shishu Hospital & Institute of Child Health, Dhaka, Bangladesh, under ethical clearance REC-FSIT-2025/No: 11928.

The citations above were read from each source on 2026-10-10 (LICENSING.md keeps where each was read).

## Limitations and known failure modes

- **The answer key is noisy.** NIH made its labels by reading radiology reports with software. NIH says they are over 90% accurate. The test labels have the same noise. A score here measures agreement with those labels, not with a radiologist looking at the picture.
- **Some findings are read poorly.** The weakest are infiltration (0.7071), pneumonia (0.7267), nodule (0.7544), consolidation (0.7610), atelectasis (0.7747) and pleural thickening (0.7845). The "Abnormal" score is 0.7363.
- **Rare findings have wide intervals.** Hernia has only 86 test pictures, and its interval runs from 0.8776 to 0.9574. Fibrosis has 435 and pneumonia 555.
- **The test list has been read three times.** Each later read was allowed only after the new reader improved on development patients, and no test score picked a model. Even so, a list read several times is a weaker test than one read once, by an amount we cannot measure.
- **The development number is not a clean one.** The vision transformers' pre-training saw the unlabelled pictures of all development patients, including those a network was later scored on. The test list was never in the pre-training pool. Trust the test number.
- **Different mix of patients.** In the development pictures, 41.6 of every 100 are abnormal. In the test list, 61.5 of every 100 are. The 0.8382 and the 0.8175 are not on the same mix of patients, so do not compare them closely.
- **No cut-off and no catch rate.** Only AUROC was measured. We cannot say how many abnormal pictures it would catch or how many normal ones it would clear at any setting.
- **No comparison with radiologists on the same pictures.** The one test that could do it needs labels we are not allowed to use. The published radiologist numbers in the table come from other pictures.
- **One hospital for labels.** Other scanners, hospitals, patient groups, bedside X-rays and children have no score.

## Licence

The weights and the code are under the PolyForm Noncommercial License 1.0.0 (see LICENSE). This line travels with every copy:

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

Commercial use is available under the rAidiology Commercial Licence ([COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)), with a free 60-day evaluation (500 studies, internal testing only, no customers, no patients' care), priced on the rAidiology site's pricing page. Not for diagnosis or treatment. Not cleared by the FDA or any regulator. Downloads are gated: you accept the licence before you download.

The credits above, BDCXR-3257's hospital acknowledgement included, must stay with the weights.

This is not a medical device. It is not cleared or approved by the FDA or any other regulator.

## How to run it

There is no one-picture command yet. The pieces are in the repository:

- `modalities/chest_xray/eval/prepare.py` prepares a picture: grey, scaled to run from the 0.5th to the 99.5th percentile of the pixels above the picture's own 1st percentile, padded to a square and resized to 384 pixels.
- `modalities/chest_xray/eval/final_eval.py`: `build_reader_net` rebuilds a network from one weight file, and `read_all` averages the scores of every network in the team.
- `modalities/chest_xray/eval/reader.py`: `ChestNet` (the convolutional network) and `predict` (subtracts the mean and divides by the sd saved in the weight file, and shrinks the picture to the network's own size).
- `modalities/chest_xray/eval/mae.py`: `ChestViT` (the vision transformer).

To read a picture, rebuild all ten networks and read the prepared picture with each. Each network gives 15 raw outputs (logits, before any sigmoid). Average those raw outputs over all ten networks. Every network counts the same. That is what was scored. If you want a 0 to 1 number, apply the sigmoid after averaging. Averaging probabilities instead gives a reader that was not tested.

The command that reads NIH's test list with version 118's five networks and version 215's five networks is:

```
python modalities/chest_xray/eval/final_eval.py --second-look --versions 118,215 --read 216
```

It only runs inside the full working folder, because it checks the development records first, and it refuses to read the test list twice for the same reader. NIH's pictures come from `python modalities/chest_xray/fetch_data.py`.

The weight files are PyTorch files that hold Python objects. Load only files you downloaded from the gated page.
