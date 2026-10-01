# SPIDER scorecard: the metric contract

`score.py` writes `SCORECARD.md` from `work/eval/predictions.csv` and SPIDER's
`radiological_gradings.csv`. `recompute.py` is written separately from this document alone, reads
only those two files, recomputes every number in `SCORECARD.md`, and exits non-zero if any
differs by more than 0.005. Everything a reimplementation needs is here; if this document and
`score.py` disagree, this document is right and `score.py` has a bug.

## Inputs

### `radiological_gradings.csv` (SPIDER, unchanged)

Columns `Patient, IVD label, Modic, UP endplate, LOW endplate, Spondylolisthesis, Disc herniation,
Disc narrowing, Disc bulging, Pfirrman grade`. The key is (`Patient`, `IVD label`). Truth:

| item | truth |
|---|---|
| `pfirrmann` | `Pfirrman grade` (1-5) |
| `herniation` | `Disc herniation` (0/1) |
| `narrowing` | `Disc narrowing` (0/1) |
| `bulging` | `Disc bulging` (0/1) |
| `spondylolisthesis` | `Spondylolisthesis` (0/1) |
| `up_endplate` | `UP endplate` (0/1) |
| `low_endplate` | `LOW endplate` (0/1) |
| `modic` | 1 if `Modic` > 0 else 0 (Modic types are counted, not scored) |

### `predictions.csv`

One row per (method, localisation, split, patient, IVD label). Every graded disc of every patient a
method covers has a row, scored or not. Columns:

| column | meaning |
|---|---|
| `method` | `rules`, `panel`, `grader`, `medgemma`, `combo` |
| `localisation` | `tss` (discs located by TotalSpineSeg) or `spider_masks` (located by SPIDER's own masks) |
| `split` | `dev` or `test` |
| `fold` | 0-4 for dev rows (the out-of-fold fold), empty for test |
| `patient`, `ivd` | SPIDER `Patient` and `IVD label` |
| `vendor` | `Philips` or `Siemens` |
| `nominal_level` | SPIDER's bottom-up level (`L5-S1` for IVD 1 ... `T12-L1` for IVD 6), empty past IVD 9 |
| `tss_level` | level of the best TotalSpineSeg disc, empty if none overlaps |
| `dice` | that disc's Dice, empty if none |
| `matched` | 1 if `dice` >= 0.3 |
| `in_scope` | 1 if the disc is scored |
| `drop_reason` | why `in_scope` is 0, else empty |
| `pfirrmann_pred` | predicted grade 1-5, or empty |
| `<item>_pred` | for each binary item above: predicted 0/1, or empty |
| `<item>_score` | for each binary item: a score where higher means "present", or empty |

## Groups and what is scored

A **group** is the set of rows sharing (`method`, `localisation`, `split`). Only rows with
`in_scope` = 1 are scored. Within a group:

- An item is **scored** for the group if at least one in-scope row has a non-empty `<item>_pred`
  (`pfirrmann_pred` for Pfirrmann). Otherwise the method makes no prediction for that item and no
  number is reported.
- AUROC is reported for an item if at least one in-scope row has a non-empty `<item>_score`.
- Within a scored item, an in-scope row with an empty prediction is a **miss** (a disc the method
  could not locate or measure). It stays in n and is filled with the worst possible answer:
  - Pfirrmann: 1 if the true grade is 3 or more, else 5.
  - binary prediction: 1 minus the truth.
  - score (for AUROC only): (minimum of the group's non-empty scores for that item) minus 1 when
    the truth is 1; (maximum of them) plus 1 when the truth is 0.

The minimum and maximum are taken over the group's in-scope non-empty scores (the whole group, not
the vendor subset), once, before any bootstrap.

## Metrics

With n discs (in-scope rows), truth t and filled prediction y:

- **exact** (Pfirrmann): mean of (y = t).
- **within_one** (Pfirrmann): mean of (|y - t| <= 1).
- **qwk** (Pfirrmann): quadratic weighted kappa over the fixed categories 1..5.
  O = 5x5 count matrix O[t, y]; E = outer(row sums of O, column sums of O) / n;
  W[i, j] = (i - j)^2 / 16; qwk = 1 - sum(W * O) / sum(W * E). Undefined when sum(W * E) = 0.
- **sensitivity**: TP / (TP + FN); undefined when there are no positives.
- **specificity**: TN / (TN + FP); undefined when there are no negatives.
- **auroc**: with filled scores s: (number of (positive, negative) pairs with s_pos > s_neg, plus
  half the pairs with s_pos = s_neg) / (n_pos * n_neg); undefined without both classes.

`n_discs` is the number of in-scope rows; `n_patients` the number of distinct patients among them.

## Confidence intervals

95% percentile bootstrap, 2,000 resamples, resampling patients. For **each reported number
separately**:

1. `P` = the sorted (ascending, as integers) distinct patients of the rows the number is computed
   on.
2. `rng = numpy.random.default_rng(20260930)`, created fresh for this number.
3. `draws = rng.integers(0, len(P), size=(2000, len(P)))`.
4. For each of the 2,000 rows of `draws`, the resample is the concatenation, in draw order, of all
   rows of patient `P[k]` for each k in that draw row (a patient drawn twice appears twice). The
   metric is computed on it with the same fill values as the full data (fill values are not
   recomputed per resample).
5. Resamples where the metric is undefined are dropped. `ci_low`, `ci_high` =
   `numpy.percentile(values, [2.5, 97.5])` (linear interpolation) over the rest. `n_boot` is how
   many were kept.

Paired differences (final evaluation): the same recipe on the patients both methods scored, with
the difference of the two methods' metric computed on each resample.

## Data accounting

From rows with `method` = `rules`, `localisation` = `tss` and the split shown:

- `graded_rows`: all rows; `patients`: distinct patients.
- `excluded_no_t2_rows`, `excluded_no_t2_patients`: rows (patients) with `drop_reason` = `no T2`.
- `ivd0_rows`: rows with `drop_reason` = `IVD label 0`.
- `outside_t2_view_rows`: rows with `drop_reason` = `not in the T2's field of view` (a graded disc
  with no SPIDER mask on the patient's T2, so no T2-based method can see it).
- `eligible`: rows with `ivd` >= 1 and `drop_reason` neither `no T2` nor
  `not in the T2's field of view`.
- `matched`: eligible rows with `matched` = 1. `match_rate` = matched / eligible.
- `level_differs`: matched rows whose `tss_level` differs from `nominal_level`.
  `level_mislabel_rate` = level_differs / matched.
- `dropped_matched_outside`: `drop_reason` = `matched outside T12-L1..L5-S1`.
- `dropped_unmatched_outside`: `drop_reason` = `unmatched, nominal level outside T12-L1..L5-S1`.
- `scored`: rows with `in_scope` = 1; `scored_unmatched`: of those, `matched` = 0.
- `modic_type_0` .. `modic_type_3`: in-scope rows by the gradings' `Modic` value.

## `SCORECARD.md` format

Every number `recompute.py` checks sits in a markdown table whose header row is exactly one of
these; other tables and prose are not checked.

1. `| split | quantity | value |`: data accounting; `quantity` is a name from the list above.
   Counts are integers; rates have 3 decimals.
2. `| split | item | metric | method | localisation | n_discs | n_patients | value | ci_low | ci_high | n_boot |`:
   every scored metric of every group, all vendors together.
3. `| split | vendor | item | metric | method | localisation | n_discs | n_patients | value | ci_low | ci_high | n_boot |`:
   the same per vendor, for `localisation` = `tss` only; P is that vendor's patients.
4. `| split | item | metric | method | reference | n_discs | n_patients | difference | ci_low | ci_high | n_boot |`:
   paired differences (final evaluation only), on rows whose (patient, ivd) is in scope for both
   groups, `localisation` = `tss`. It holds `split` = `test`, `item` = `pfirrmann`, `metric` = `qwk`,
   one `reference` shared by every row (the baseline chosen before the test split was opened), and
   exactly one row for every other method that has test rows in `predictions.csv`.

Values have 3 decimals. An undefined value is written `n/a` and must be `n/a` in the recomputation
too. A difference over 0.005 in any cell, a missing row, or an extra row is a failure.

Every split that has rows in `predictions.csv` must be reported in tables 1-3. The one exception is
`test` before the final evaluation, when the scorecard's held-out section says "Not yet run" and
`predictions.csv` holds no test rows. A split in `predictions.csv` that the scorecard leaves out is
reported as missing rows.
