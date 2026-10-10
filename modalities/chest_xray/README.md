# Chest X-ray

**Status (2026-10-07 13:40):** the reader (`eval/reader.py`, trained from no outside weights on the five
development folds) read NIH's official test list once on 2026-10-07 01:42: mean AUROC of the fourteen findings
**0.797** (0.790 to 0.803), versions 103 and 106 read together (v192). That is a hair behind a DenseNet-121 trained
from scratch on the same list (0.804) and behind 0.830 from a net pretrained on 0.5 million chest X-rays
(docs/VS_DOCTORS.md). A wider reader (v118, the last two stages 512 wide) read +0.0079 over v103 on fold 0 and is
being trained on all five folds (idea 99 in docs/efficiency). `eval/final_eval.py --second-look --versions A,B
--read R` reads the spent test list once more, labelled a second look, only when `compare.py` says the new read is
BETTER than the read the test last saw on development; no test number chooses anything.

## The questions

1. **Is this chest X-ray normal or not?** The question two US board-certified radiologists were
   scored on, on 810 NIH chest X-rays labelled by a majority of three other radiologists: 87 of 100
   abnormal ones caught with 89 of 100 normal ones cleared, and 81 caught with 94 cleared (Nabulsi et
   al., Scientific Reports 2021, Table 3). Google released those 810 images' labels.
2. **Which of fourteen findings does it show** (atelectasis, cardiomegaly, effusion, infiltration,
   mass, nodule, pneumonia, pneumothorax, consolidation, edema, emphysema, fibrosis, pleural
   thickening, hernia)? Nine radiologists' AUROC per finding on 420 NIH X-rays is the bar (CheXNeXt,
   PLOS Medicine 2018, Table 1); that panel's labels are private, so this one is scored on NIH's own
   test list instead.

Both are lines in the maintainers' comparison with published doctors' scores, which is not part of this copy.

## The data

NIH ChestX-ray14 (LICENSING.md): 112,120 frontal chest X-rays of 30,805 patients from the NIH
Clinical Center, 1024 x 1024 PNG, each with up to fourteen findings mined from its report (NIH: over
90% accurate). NIH splits it by patient into `train_val_list.txt` (86,524 pictures) and
`test_list.txt` (25,596). Here the first is the development set (five folds by patient) and the
second the held-out test, opened once.

Google's radiologist labels for 4,374 of these pictures (four findings, each image read by three
radiologists and adjudicated) and for 810 (normal or not, three radiologists) come through a Google
form, and their README says they are for research and education only (read on a public mirror
2026-10-06; LICENSING's NIH row). So rAidiology uses them neither for training nor for scoring, and the
doctors' normal-or-not line waits for labels with clean terms.

```
python modalities/chest_xray/fetch_data.py        # 45 GB into <Datasets>/public/nih-chestxray14/
```

Not a medical device.
