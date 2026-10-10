# Model card: rAidiology Lumbar spine X-ray corner reader, side view v1 + v2 (read five ways, stored as v7), front view v10

**Not a medical device. Not for diagnosis or treatment decisions.**

## What it does

It finds the corners of the lower-back vertebrae on a spine X-ray. You give it one lumbar spine X-ray picture (a JPG). It gives back the position of each corner, counted in pixels of that picture. The lower-back vertebrae are called L1 to L5. For each one it marks two points on the top edge and two points on the bottom edge. A front-view picture (AP) gets 20 points. A side-view picture (LA, patient facing left) gets 22 points, because it also marks two points on the top edge of the sacrum (S1).

There is one reader for each view. The front reader is five small neural networks (U-Nets). The side reader is five pairs: a first-look network and a close-look network, ten in all. Each network (or pair) trained on a different four fifths of the patients.

The side reader works in two steps:

1. The first look shrinks the whole picture to 512 by 384 pixels and places all 22 points.
2. The close look cuts a 160 by 160 piece out of the full-size picture around each point and places that point again. It does this five times, with the piece moved 4 pixels left, right, up and down, and averages the five answers.

The front reader has the first look only. Every network was trained from scratch. No outside pretrained weights were used.

It says where the corners are. It does not say whether a spine is healthy, and it does not look for disease.

### What it is not for

- **Diagnosis.** It does not tell you what is wrong with anyone.
- **Treatment decisions.** Do not use its points to plan surgery, bracing or any other treatment.
- **Triage or screening decisions.** Do not use it to decide which pictures are read first or who is called back.
- **Calling a slipped vertebra (spondylolisthesis).** A slip call built on an earlier reading of these corners was tested and was not good enough to keep. See the table below. Do not use it.
- **Measurements in millimetres.** The pictures carry no pixel size. Every distance is in pixels.
- **Children, older adults, men or women as separate groups.** No score is split by age or sex, so this card makes no claim for any group.
- **Other body parts and other views.** It reads the lower-back spine, front or side. It does not read the neck or the mid-back (thoracic) spine, and it does not read angled (oblique) views.
- **Other hospitals, other countries and other machines.** The training pictures all come from one dataset collected in Thailand. On pictures from a second set, accuracy fell, and for one group it fell a lot. See the table.
- **Spines with metal implants or fractures.** No result is reported for them.
- **Files other than JPG.** It was trained and tested on JPG pictures. DICOM files were not used.

## How well it does

Numbers are as of 10 October 2026.

**Corner miss** is how far a placed corner is from the hand-drawn corner, in pixels. Lower is better. The average is taken over a picture's corners, then over patients.

**Vertebrae found** works like this. We draw one box around the four corners we placed for a vertebra, and compare it with the box around the four hand-drawn corners. The box counts as found when the two overlap by at least half (called IoU 0.5). The stricter reading averages ten overlap levels, from 0.50 to 0.95. We draw one box for each vertebra, so "right when it says one is there" and "finds the ones that are there" are the same number.

**Within 5 px** is the share of corners placed no more than 5 pixels from the hand-drawn corner. On the outside set, pixels are those of the set's pictures, which come squeezed to 640 x 640. A BUU-LSPINE side film is about 2,000 pixels wide and about 2,500 tall, so these pixels are about three times wider and nearly four times taller than the ones in the corner-miss rows.

**Out of fold** means each patient was read only by a network that did not train on that patient. The BUU-LSPINE numbers are all out of fold. The same patients were also used to pick this version, so they are development scores. They are not a test that was opened once. The outside set in the second half of the table is the one test of that kind, and it was scored once.

The answer key is the hand-drawn lines that come with the dataset. A reader that was better than the drawing would be marked wrong where it is right.

| Measure | Ours (95% interval) | Test set | Published number we compare with, and who holds it | Same pictures? |
|---|---|---|---|---|
| Vertebrae found, side view, IoU 0.5 | 97.9 (97.3 to 98.4) | All 2,000 BUU-LSPINE patients, out of fold | Precision 83.5 and recall 95.7: YOLOv5 in the BUU-LSPINE paper (Klinwichit et al., Table 8) | Same dataset, other patients (the paper's split is not known) |
| Vertebrae found, side view, stricter reading (0.50 to 0.95) | 89.4 (88.8 to 89.9) | The same 2,000 patients | The same numbers. The paper says its scores average over 0.50 to 0.95, but its precision and recall columns look like IoU 0.5, so we give both. | The same |
| Vertebrae found, front view, IoU 0.5 | 95.8 (95.0 to 96.6) | All 2,000 BUU-LSPINE patients, out of fold | Precision 81.9 and recall 95.8: the same YOLOv5 | Same dataset, other patients |
| Vertebrae found, front view, stricter reading (0.50 to 0.95) | 81.2 (80.4 to 81.9) | The same 2,000 patients | The same numbers | The same |
| Corner miss, side view | 11.78 px, median patient 7.10 px (no interval recorded) | All 2,000 BUU-LSPINE patients, out of fold | 4.91 mm: DenseNet201 in the BUU-LSPINE paper | Same dataset, other patients. The paper counts millimetres and we count pixels, so the two cannot be compared yet. |
| Corner miss, front view | 17.33 px (15.90 to 18.86), median patient 9.45 px | All 2,000 BUU-LSPINE patients, out of fold | 4.63 mm: ResNet152V2 in the BUU-LSPINE paper | The same, and not comparable yet |
| Outside set, side view: corners within 5 px | 64.3 (61.7 to 66.9) of 12,548 corners on 590 pictures. The 501 Burapha-patient pictures: 71.6. The 89 Honduran-patient pictures: 18.3. | The Spondylolisthesis Vertebral Landmark set, pictures scaled to 640 x 640. No network ever trained on it. Scored once. | 76.75: YOLOv11n-Pose trained on 80% of this set (Bansal et al., PLoS One 2026, e0347290, Table 4) | Same dataset, but they trained on it and we did not. Whether their split is the folders the set ships with, the paper does not say. |
| Outside set, side view: corners within 15 px | 81.7 (78.8 to 84.3) | The same 590 pictures, scaled to 640 x 640 | About 100: the same paper | The same |
| Slipped vertebra called right, side view (not part of this release) | Accuracy 75.5 (73.7 to 77.3); sensitivity 80.0; specificity 75.1. Overall skill (AUROC) 0.83 (0.79 to 0.86), against 0.89 when the hand-drawn corners are used. | All 2,000 BUU-LSPINE patients, out of fold, on the close-look corners read once (an earlier reading of the same close-look networks, not the five-look reading in this card), graded against the corrected slip labels | Accuracy 92.3: a support vector machine on the corners, in the BUU-LSPINE paper | Same dataset, other patients. A reader that never calls a slip scores 92.2 here, so the paper's number needs its own labels before it can be compared. Not kept. |

### Where it is behind

- **Outside pictures.** On the second set, whose pictures are scaled to 640 x 640, it places 64.3 of 100 corners within 5 px. The published number is 76.75. This is not a like-for-like test: the published reader trained on part of that set, and ours trained on none of it.
- **Different place, different result.** On that set, the Burapha-patient pictures score 71.6 and the Honduran-patient pictures score 18.3. A reader that fits a film by the shape of the training pictures fails on films that look different.
- **Strict reading of vertebrae found.** On the side view, the looser reading (97.9) is above the paper's recall of 95.7. The stricter reading (89.4) is below it. On the front view, the stricter reading (81.2) is just under the paper's precision of 81.9, and its interval reaches that number, so all that can be said is level. The front view's stricter reading (81.2) is well below the paper's recall of 95.8. At IoU 0.5 the front view (95.8, 95.0 to 96.6) is level with that recall.
- **Corners in millimetres.** We cannot say yet how we stand against the published 4.91 mm and 4.63 mm.
- **Slips.** The slip call is behind the published accuracy, and it was not kept.

### A newer version, not in this card

A newer pair of networks also learned from four fifths of the outside set. On the 118 pictures of that set it had not seen (scaled to 640 x 640), it placed 73.2 of 100 corners (66.9 to 78.9) within 5 px. The reading in this card places 67.7 on the same 118. That is level with the published 76.75, because the interval reaches it. Only one fifth of the set has been read so far, so this is not the version described here.

## Data it learned from

| Dataset | What it was used for | Licence | Credit the licence asks for | Commercial use allowed? |
|---|---|---|---|---|
| BUU-LSPINE, the 2,000-patient release (Burapha University, Thailand): 4,000 lumbar spine X-ray pictures, front and side, with hand-drawn lines on the upper and lower edge of each vertebra | **Training.** Each network trained on four fifths of the patients by patient (1,600 patients), five networks for the front view and five pairs for the side view. The held-out fifth was used to score it. | The dataset's End-User License Agreement from Burapha University, accepted before download | Do not try to identify or contact the patients. Acknowledge the Burapha Spine Dataset, and data derived from it, when presenting results or algorithms that used it. Cite: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis Detection. Applied Sciences 2023;13(15):8646. Dataset page: https://services.informatics.buu.ac.th/spine/. Do not use Burapha University's name to suggest it endorses anything. No warranty. | No clause limits use to research or rules out commercial use. The agreement does not say in so many words that commercial use is allowed. The article in Applied Sciences is CC BY 4.0, and that covers the article only. |
| Spondylolisthesis Vertebral Landmark Dataset v1 (Reyes, VSB-TU Ostrava), Mendeley Data: https://data.mendeley.com/datasets/5jdfdgp762/1 | **Testing only.** 590 side-view pictures, scored once. No network trained on any of it. | CC BY 4.0 | Name the dataset, its author and the source link, and give the licence. | Yes |

Other things you might ask about:

- **Pretraining:** none. No ImageNet weights and no other outside weights.
- **Labels from another model:** none. All labels are hand-drawn lines from the dataset.
- **Calibration sets:** none.
- **BUU-LSPINE 400-patient sample:** these are the first 400 patients of the 2,000. They are not extra patients. They were used only to check slip labels, and the corner reader does not use those labels.

For the licences of the other datasets used by other rAidiology models, see [LICENSING.md](../../LICENSING.md).

## Limitations and known failure modes

- **The average miss is bigger than the typical miss.** On the side view the average is 11.78 px and the median patient is 7.10 px. On the front view it is 17.33 px and 9.45 px. Most pictures are placed well, and a minority are placed badly.
- **One source of training pictures.** Everything it learned from comes from one dataset collected in Thailand. On a second set, the Honduran-patient pictures scored 18.3 within 5 px, against 71.6 for the Burapha patients.
- **Side films must face left.** All side films in the training set face left. The reader never flips a film. For the outside test, each picture was read as it was and mirrored, and the reading closer to the average training shape was kept. The mirrored reading was kept for 4.6 of 100 pictures overall.
- **Scores are development scores.** The same 2,000 patients were used to choose this version. Choosing among versions can make the scores a little more hopeful than a fresh test would.
- **The answer key is one set of hand-drawn lines.** If the drawing is off at a corner, our score cannot show it.
- **Pixels, not millimetres.** The pictures carry no pixel size.
- **The slip call was not kept.** On the 2,000 patients, a slip call built on an earlier reading of the corners (the close-look networks read once) had an accuracy of 75.5, and a reader that never calls a slip scores 92.2. The slip labels outside the first 400 patients are probably missing some real slips, so even that comparison is rough.
- **Checked on JPG pictures only,** and only on the two datasets named above.
- **The five-network average on new pictures.** On BUU-LSPINE, each patient was read by one network. On new pictures the side reader averages the points from all five networks, and that average was scored on the outside set. The front reader's five-network average has not been scored.

## Licence

The weights and the code are under the [PolyForm Noncommercial License 1.0.0](../../LICENSE).

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

That notice travels with every copy. The licence is free for noncommercial use. Commercial use needs a separate licence from LunarWerx. Contact LunarWerx at https://github.com/LunarWerxs.

Downloads are gated: you accept the licence before you download.

Please keep the Burapha University credit above when you share results made with this reader.

This is not a medical device. It is not cleared or approved by the FDA or any other regulator.

## How to run it

Run from the repository root. These commands need the BUU-LSPINE files in place, and the networks in `work/eval/spine_xray/` (`RAIDIOLOGY_WORK` moves that folder). The scripts read the networks from the folders named below.

The files, five networks each:

| Reader | Folder and files | Size of each file |
|---|---|---|
| Side view, first look | `corners-v1/LA-fold0.pt` to `LA-fold4.pt` | 11,086,402 bytes |
| Side view, close look | `corners-v2/LA-fold0.pt` to `LA-fold4.pt` | 7,575,746 bytes |
| Front view | `corners-v10/AP-fold0.pt` to `AP-fold4.pt` | 11,086,530 bytes |

All fifteen files together are 148,743,390 bytes. The side reader needs both its first-look and close-look files.

To read the side view the way this card describes, use the repository's modules:

- [`modalities/spine_xray/eval/corners_reread.py`](../../modalities/spine_xray/eval/corners_reread.py) places the points with the close-look networks, five looks. It starts from the first-look points saved in `corners-v1`. This is the command that made the side numbers above: `python modalities/spine_xray/eval/corners_reread.py --from-version 1 --passes 1 --looks 5 --out-version <a new number> --against 2`
- [`modalities/spine_xray/eval/corners.py`](../../modalities/spine_xray/eval/corners.py) is the first look. `python modalities/spine_xray/eval/corners.py eval --view AP --version 10 --folds all` prints the front-view corner miss.
- [`modalities/spine_xray/eval/vertebra_boxes.py`](../../modalities/spine_xray/eval/vertebra_boxes.py) scores vertebrae found: `python modalities/spine_xray/eval/vertebra_boxes.py score --view AP --version 10`
- [`modalities/spine_xray/eval/landmarks_test.py`](../../modalities/spine_xray/eval/landmarks_test.py) holds `load_models` and `read_picture`, which read a picture from outside the BUU-LSPINE set with all five side-view chains. It is the closest thing to a one-picture reader. It was written for the outside test, which uses 640 by 640 pictures.

There is no command yet that takes any single X-ray and prints its corners. The modules above are what exist today. See also the [spine X-ray reader overview](../../modalities/spine_xray/README.md).
