# Spine X-ray reader

Reads a lumbar spine X-ray: places the corners of each vertebra's upper and lower edge (L1 to L5, plus
S1's top on the side view) and, from the side view's corners, measures each level's slip (Meyerding) to
flag a slipped vertebra. Trained from scratch (no pretrained weights) on BUU-LSPINE, 2,000 Thai patients'
front (AP) and side (LA) films. Plan and pass bars: `docs/efficiency/README.md`, "Spine X-ray: vertebra
corners and slipped vertebrae from BUU-LSPINE".

Dataset: Klinwichit P, et al. BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis
Detection. Applied Sciences 2023;13(15):8646. Used under its End-User License Agreement (LICENSING.md).

All scripts are in `eval/`. Run from the repository root. Outputs go under
`work/eval/spine_xray/` (`RAIDIOLOGY_WORK` moves it); the card share asked for is about 2 GB
(`RAIDIOLOGY_CARD_GB` changes it).

1. `python modalities/spine_xray/eval/buu.py check`: counts patients, images, rows and classes, compares
   the 400-patient sample with the set; exits 0 only for 2,000 patients, 4,000 images, 10 rows an AP file
   and 11 an LA file. `python modalities/spine_xray/eval/buu.py split` writes `eval/split.json` (five
   folds by patient, seed 0, stratified on any side-view class 3 or 4). Row order, which end of a line is
   front, and the point order are in `buu.py`'s docstring.
2. `python modalities/spine_xray/eval/corners.py train --view LA --version 1 --folds 0` (then `--folds all`
   for the slip reader's five out-of-fold sets), and the same for `--view AP`. `--epochs E` and
   `--set key=<json>` change the recipe (`--help` lists them). `... corners.py eval --view LA --version 1`
   prints the mean corner miss in pixels for the model and the mean-shape baseline with patient-bootstrap
   95% intervals, the line `Verdict (corners, half the baseline's miss)`, and writes a ledger row.
3. `python modalities/spine_xray/eval/slip.py --version 1` reads the out-of-fold LA corners of
   `corners.py` version 1 (needs `--folds all` trained) and prints per-level slips, the amended score's
   per-patient AUROC for any anterolisthesis or retrolisthesis (each level's slip minus its median over the
   other folds, then the largest absolute value) for the model and for the drawn corners with intervals, their
   paired difference, accuracy, sensitivity and specificity at fold-internal thresholds, and
   `Verdict (slips, not worse than drawn corners, margin 0.03)`. The written score (largest raw slip) and the
   0.85 target are printed as context. `slip.py --truth` scores the drawn corners only and prints
   `Drawn corners, amended score: AUROC ...`, with no verdict.
