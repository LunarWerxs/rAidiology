# The SPIDER scorecard and the specialist models

Everything here measures SuprSkan against expert grades on public scans before anything ships.
The rules: aim for commercial-OK licences only (open items are marked in `LICENSING.md`), honest
numbers, no fitting to a radiologist's report; the metric definitions are in [METRICS_SPEC.md](METRICS_SPEC.md); the results are in
[SCORECARD.md](SCORECARD.md).

Data and tools live outside the repository, beside it: SPIDER in `../Datasets/public/spider/` (or set
`SPIDER_ROOT`), trained weights in `../Models/<name>-v<N>/`, the Spinal Cord Toolbox in
`../_tools/sct_7.3/` (`pipeline/sct/install_sct.ps1`), and MedGemma's transformers in
`../_tools/medgemma_libs/` (kept out of the main environment). Working files go to `work/eval/`
(git-ignored). Python: the virtual environment from `pipeline/README.md`, plus pandas and
scikit-learn. Heavy steps need a GPU and can take hours.

## Order

| step | command | needs |
|---|---|---|
| split (done, committed) | `python split.py` | `overview.csv` |
| development data, TotalSpineSeg, matching, rules baseline | `python run_split.py dev` | SPIDER, GPU |
| grader crops | `python crops.py dev` | the step above |
| grader training (5 folds + 20 nested models) | `python grader.py train --version 1` | GPU |
| grader development predictions | `python grader.py predict dev` | trained grader |
| AI panel, prompt tuning then development score | `python panel_runner.py spider dev --dev-set tune --backend agents` writes bundles; Claude Opus sub-agents answer them; `python panel_runner.py ingest <answers.json>`; re-run the first command for predictions, which also fits the panel's cut-points and item thresholds on those 10 patients (`work/eval/panel/dev/calibration.json`); then the same with `--dev-set score` (calibrated) | no API key needed: the reader runs as Claude Opus 5.5 sub-agents (`--backend api` uses a key instead) |
| MedGemma on development, then the qualify check | `python medgemma_runner.py dev`, `python medgemma_runner.py qualify` | weights from google/medgemma-1.5-4b-it in `../Models/medgemma-1.5-4b-it`, after you accept Google's Health AI Developer Foundations terms |
| development scorecard | `python run_split.py --collect-only` (re-collects `predictions.csv`), `python score.py` | |
| independent recomputation | `python recompute.py > recompute.log` | run by a separate agent session that has not seen `score.py` |
| mistake-log cases | `python panel_runner.py case cases/<id>.json --cd <cd_folder> --backend agents`, answer, ingest, run again for PASS/FAIL | |
| **final evaluation (once)** | `python final_eval.py --panel agents [--with-medgemma]`, answer the 30 test bundles with Opus readers, `python panel_runner.py ingest <answers.json>`, then `python final_eval.py --finish` | everything above |
| model card | `python model_card.py` | trained grader |
| app | `python ../specialist.py ../../work`, then `make_report.py` | a method that passed the ship rule |

The AI panel runs without an API key: `--backend agents` turns each request into a folder of
images and instructions with a neutral name (`work/eval/panel/bundles/`), Claude Opus sub-agents
answer them reading only that folder, and `ingest` stores the answers where an API call's would go.

## What each file is

- `spider.py`: SPIDER's files, the T2 rule, the bottom-up label convention, PIR conversion.
- `split.py` / `split.json`: the patient split (seed 20260930).
- `prepare.py`: NIfTI conversion, label-convention checks, the SPIDER-mask label maps.
- `match.py`: Dice matching of graded discs to TotalSpineSeg discs, scope and drop reasons.
- `rules.py`: baseline (a). `panel_runner.py`: baseline (b) and the mistake-log case runner.
- `crops.py`, `grader.py`: the disc grader. `model_card.py` writes `pipeline/models/disc-grader.md`.
- `medgemma_runner.py`: MedGemma and the combination. `percentiles.py`: the lumbar reference gate.
- `score.py` writes the scorecard; `recompute.py` (written from METRICS_SPEC.md alone) checks it.
- `final_eval.py`: the one-time test run and the ship decision.
- `cases/`: the mistake log, one confirmed miss per JSON file: `id`, `region` (`lumbar` or
  `cervical`), `series` ({series name: [slice numbers]}) and `test` ({`question`, `field`, `levels`,
  `choices`, `expect`: {level: the right answer}}). Any case file makes a re-run of that scan "not
  blind". The development cases are private data and are never in a public copy.
