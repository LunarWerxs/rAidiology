# X-ray scorecard: the metric contract

`score.py` writes `SCORECARD.md` from `work/eval/xray/predictions.csv` and the two datasets' own
index files. `recompute.py` is written separately from this document alone, reads only those files,
recomputes every number in `SCORECARD.md`, and exits non-zero if any differs by more than 0.005.
Everything a reimplementation needs is here; if this document and `score.py` disagree, this document
is right and `score.py` has a bug.

## Inputs

### The datasets' own index files (unchanged)

| dataset | file | read it with |
|---|---|---|
| `grazpedwri` | `<Datasets>/public/grazpedwri-dx/dataset.csv` | encoding `utf-8-sig` (the file starts with a byte-order mark, so its first column name is `filestem`) |
| `fracatlas` | `<Datasets>/public/fracatlas/FracAtlas/dataset.csv` | encoding `utf-8` |

`<Datasets>` is `../Datasets` beside the repository, or `$RAIDIOLOGY_DATASETS`.

The key is (`dataset`, `image`): `filestem` for `grazpedwri`, `image_id` for `fracatlas`.

Truth, for both datasets the column is read as a string and compared to `"1"`:

| dataset | truth = 1 when |
|---|---|
| `grazpedwri` | `fracture_visible` == `"1"` |
| `fracatlas` | `fractured` == `"1"` |

Every other value, including an empty cell, is truth 0.

**A collection on trial: `orthofrac`** (OrthoFrac-XR, added 2026-10-03; `datasets.ALL`, not
`datasets.NAMES`). It trains a trial model only, and nothing on this scorecard reads it until it is
promoted. Its index is `<Datasets>/public/orthofrac-xr/OrthoFrac-XR/Bone_metadata.csv` (encoding
`utf-8-sig`), keyed by `id`. Truth is 1 when `fracture_type` is `distal` or `proximal` and 0 when it
is `non_fracture`; rows with `post_fracture`, and the 12 ids in `datasets.ORTHOFRAC_DUPLICATES`
(another collection's pictures), are not in the index at all. Strata: `all`, `adult` (age 18 or
more), `child`, `upper_limb` and `lower_limb` (`datasets.UPPER_LIMB`). The unit is the image (no
patient identifier is published).

**A collection on trial: `bonefract`** (BoneFract, added 2026-10-03; `datasets.ALL`). It trains a
trial model only. It ships no index file, so `datasets.py` writes one beside it on first use,
`<Datasets>/public/bonefract/rAidiology_index.csv`, one row per PNG under
`BoneFract A Bone Fracture Dataset/<train|valid|test>/<region>/patient<NNNNN>/<Positive|Negative>/`,
keyed by the file's stem (`<region>_patient<NNNNN>_<label>_001`). Truth is 1 under `Positive`, 0 under
`Negative`. The unit is the patient (`patient<NNNNN>`; after the exclusions below, 1 of 36,914 has two pictures). The authors'
own train, valid and test folders are kept as a column (`given_split`) but not used: the split is
this project's, locked like the others. Strata: `all`, `upper_limb` (Arm, Hand, wrist),
`lower_limb` (Foot, Lower leg, Thigh) and `pelvis`, by the region folder a picture is filed under
(its name's region can differ). The stems in `modalities/xray/eval/bonefract_excluded.csv` are not
in the index: BoneFract's own copies (groups of pictures at r 0.97 or more: a group carrying both
labels is left out whole, otherwise all but the first name in sort order) and its pictures matching
another collection's. 36,915 pictures remain.

### `predictions.csv`

One row per (`method`, `dataset`, `split`, `image`). Every image of every (dataset, split) a method
covers has a row, scored or not. Columns:

| column | meaning |
|---|---|
| `method` | `prevalence`, `meta`, `features`, `reader`, `cnn` |
| `dataset` | `grazpedwri` or `fracatlas` |
| `split` | `dev` or `test` |
| `fold` | 0-4 for `dev` rows (the fold the row was predicted out of), empty for `test` |
| `patient` | the resampling unit. GRAZPEDWRI-DX's `patient_id`; for FracAtlas, which publishes no patient identifier, the image's own `image_id`, so there each image is its own unit |
| `image` | `filestem` (`grazpedwri`) or `image_id` (`fracatlas`) |
| `in_scope` | 1 if the row is scored |
| `drop_reason` | why `in_scope` is 0, else empty |
| `fracture_pred` | predicted 0/1, or empty |
| `fracture_score` | a number where higher means "fracture more likely", or empty |

`in_scope` is 0 only when the image itself could not be used; the one reason in use is
`unreadable image`. A row whose `image` is not a key of the dataset's index file is an error, not a
drop.

## Groups, strata and what is scored

A **group** is the set of rows sharing (`method`, `dataset`, `split`). Only rows with `in_scope` = 1
are scored.

A **stratum** selects in-scope rows of a group by their image's row in the dataset index file. The
strata are fixed:

| dataset | stratum | the image's index row |
|---|---|---|
| `grazpedwri` | `all` | every in-scope row |
| `grazpedwri` | `initial_exam` | `initial_exam` == `"1"` |
| `grazpedwri` | `no_cast` | `cast` != `"1"` |
| `grazpedwri` | `certain` | `diagnosis_uncertain` != `"1"` |
| `fracatlas` | `all` | every in-scope row |
| `fracatlas` | `hand` | `hand` == `"1"` |
| `fracatlas` | `leg` | `leg` == `"1"` |
| `fracatlas` | `hip` | `hip` == `"1"` |
| `fracatlas` | `shoulder` | `shoulder` == `"1"` |

FracAtlas's body-part columns overlap: an image flagged `mixed` can appear in more than one stratum.

Within a group:

- The group is **scored** if at least one in-scope row has a non-empty `fracture_pred`. Otherwise the
  method makes no prediction for that group and no number is reported for it.
- `auroc` is reported for a group if at least one in-scope row has a non-empty `fracture_score`.
- An in-scope row with an empty prediction is a **miss**. It stays in n and is filled with the worst
  possible answer:
  - prediction: 1 minus the truth;
  - score (for `auroc` only): (minimum of the group's non-empty `fracture_score`) minus 1 when the
    truth is 1; (maximum of them) plus 1 when the truth is 0.

The minimum and maximum are taken over the **whole group's** in-scope non-empty scores, once, before
any stratum is selected and before any bootstrap. A group whose every in-scope score is empty gets
no `auroc` row at all, so the fill values are never needed in that case.

## Metrics

With n in-scope rows of the stratum, truth t and filled prediction y, and
TP = #(t=1, y=1), FP = #(t=0, y=1), FN = #(t=1, y=0), TN = #(t=0, y=0):

- **sensitivity**: TP / (TP + FN); undefined when TP + FN = 0.
- **specificity**: TN / (TN + FP); undefined when TN + FP = 0.
- **balanced_accuracy**: (sensitivity + specificity) / 2; undefined when either is undefined.
- **kappa**: Cohen's kappa. po = (TP + TN) / n;
  pe = ((TP + FP)(TP + FN) + (FN + TN)(FP + TN)) / n^2; kappa = (po - pe) / (1 - pe).
  Undefined when n = 0 or pe = 1.
- **auroc**: with filled scores s: (number of (positive, negative) pairs with s_pos > s_neg, plus
  half the pairs with s_pos = s_neg) / (n_pos * n_neg); undefined without both classes.

`n_images` is the number of in-scope rows of the stratum; `n_patients` the number of distinct
`patient` values among them.

The five metrics are reported in this order: `auroc`, `sensitivity`, `specificity`,
`balanced_accuracy`, `kappa`.

## Confidence intervals

95% percentile bootstrap, 2,000 resamples, resampling the `patient` column. For **each reported
number separately**:

1. `P` = the sorted distinct `patient` values of the rows the number is computed on, sorted as
   strings by Python's default string order after stripping surrounding whitespace.
2. `rng = numpy.random.default_rng(20261001)`, created fresh for this number.
3. `draws = rng.integers(0, len(P), size=(2000, len(P)))`.
4. For each of the 2,000 rows of `draws`, the resample is the concatenation, in draw order, of all
   rows of patient `P[k]` for each k in that draw row (a patient drawn twice appears twice). The
   metric is computed on it with the same fill values as the full data (fill values are not
   recomputed per resample).
5. Resamples where the metric is undefined are dropped. `ci_low`, `ci_high` =
   `numpy.percentile(values, [2.5, 97.5])` (linear interpolation) over the rest. `n_boot` is how
   many were kept.

Paired differences (final evaluation): the same recipe on the rows whose (`dataset`, `split`,
`image`) is in scope for both methods, with the difference of the two methods' metric computed on
each resample. `P` comes from those shared rows.

### An equivalent fast form

A resample built row by row, 2,000 times, for each of a couple of hundred numbers over 16,000 rows
is too slow to be worth writing. Both of these are the same arithmetic in a different order, and
either may be used:

- Every metric but `auroc` depends on the resample only through its TP, FP, FN and TN totals. Count
  each patient's own TP, FP, FN, TN once into a matrix `C` with one row per patient of `P`; let `M`
  be the 2,000 x len(P) matrix of how often each patient appears in each resample (that is,
  `M[i, k]` = the number of times `k` occurs in draw row `i`). Then the 2,000 resamples' totals are
  `M @ C`, and the metric of each row of that is the metric of that resample.
- `auroc` depends on the resample only through, for each distinct score value in ascending order,
  how many positives and how many negatives carry it. Build those as two len(P) x V matrices
  `S_pos`, `S_neg` (patient by score value), and the resamples' per-value counts are `M @ S_pos` and
  `M @ S_neg`. With `pos` and `neg` as a resample's rows of those,
  `auroc = sum(pos * (cumulative neg strictly below this value + 0.5 * neg)) / (sum(pos) * sum(neg))`,
  which is the pair-counting definition above.

`draws` depends only on the seed and `len(P)`, so for a given `len(P)` it may be computed once and
reused for every number with that many patients: a fresh `default_rng(20261001)` asked for the same
size always returns the same draws.

## Data accounting

From rows with `method` = `prevalence` and the (dataset, split) shown:

- `rows`: all rows; `patients`: distinct `patient`.
- `in_scope_rows`: rows with `in_scope` = 1; `unreadable_rows`: rows with
  `drop_reason` = `unreadable image`.
- `fracture_rows`: in-scope rows whose truth is 1. `prevalence` = fracture_rows / in_scope_rows,
  3 decimals.
- for each stratum of that dataset except `all`: `<stratum>_rows`, the in-scope rows in it, and
  `<stratum>_prevalence` (3 decimals), the share of those whose truth is 1.
- `dev` only: `fold_0_patients` .. `fold_4_patients`, the distinct `patient` values with that `fold`.

Counts are integers; anything named `*prevalence*` or `*rate*` has 3 decimals.

## `SCORECARD.md` format

Every number `recompute.py` checks sits in a markdown table whose header row is exactly one of
these; other tables and prose are not checked.

1. `| dataset | split | quantity | value |`: data accounting; `quantity` is a name from the list
   above.
2. `| dataset | split | stratum | metric | method | n_images | n_patients | value | ci_low | ci_high | n_boot |`:
   every reported metric of every scored group, for every stratum of that dataset.
3. `| dataset | split | stratum | metric | method | reference | n_images | n_patients | difference | ci_low | ci_high | n_boot |`:
   paired differences, final evaluation only. `split` = `test`, `metric` = `auroc`, one `reference`
   shared by every row (the baseline chosen before the test split was opened), and exactly one row
   for every (dataset, stratum) and every other method that has `test` rows in `predictions.csv`.

Rows are written sorted by their columns left to right, as strings, except that `metric` follows the
order given above and `stratum` puts `all` first.

Values have 3 decimals. An undefined value is written `n/a` and must be `n/a` in the recomputation
too. A difference over 0.005 in any cell, a missing row, or an extra row is a failure.

Every (dataset, split) that has rows in `predictions.csv` must be reported in tables 1 and 2. The one
exception is `test` before the final evaluation, when the scorecard's held-out section says "Not yet
run" and `predictions.csv` holds no test rows. A (dataset, split) in `predictions.csv` that the
scorecard leaves out is reported as missing rows.
