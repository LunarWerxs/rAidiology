# How a CT segmenter is scored

This is the contract. The scorer and the independent recomputation are written from this page and
not from each other, which is the whole point of writing it down before either exists: two people
reading one specification is a check, two people reading one implementation is a typo waiting to be
agreed with.

Fixed on 2026-10-01, before anything was trained.

## The question

> **Where is each organ and bone in this scan?** One label per voxel, over the 117 structures the
> TotalSegmentator dataset marks.

## What a method hands in

One CSV, `predictions.csv`, with exactly these columns:

| column | meaning |
|---|---|
| `method` | the name of the method, e.g. `always-empty`, `bone-threshold`, `segmenter` |
| `split` | `dev` or `test` |
| `fold` | the development fold the scan belongs to; empty for test |
| `scan` | the dataset's `image_id`, e.g. `s0042` |
| `structure` | one of the 117 names, exactly as the mask files are named |
| `in_scope` | `1` or `0`, see below |
| `drop_reason` | why `in_scope` is `0`; empty otherwise |
| `truth_voxels` | voxels the expert mask marks for this structure in this scan |
| `pred_voxels` | voxels the method marks |
| `both_voxels` | voxels both mark |

**Voxel counts, not volumes.** Everything scored here is computable from those three counts, so the
scorer never opens a scan and the recomputation never has to agree with the scorer about how a file
was read. The file is about 143,000 rows and a few megabytes, which is a thing two people can
actually both check.

A row must exist for every (scan, structure) pair in the split, including the pairs where the
structure is not in the scan. **A missing row is an error, not a zero**: silently treating it as
zero is how a method that skipped the hard scans scores well.

## In scope

`in_scope = 0` only for these reasons, written in `drop_reason`:

- `unreadable` - the scan or its masks could not be read at all.
- `geometry_mismatch` - the prediction is not on the same voxel grid as the truth, so the counts do
  not describe the same space.

Nothing else. In particular a structure the scan does not contain is **in scope**: it is most of the
data and the thing a careless method gets wrong.

## The two things measured, and why it is two

A structure is either **present** in a scan (`truth_voxels > 0`) or **absent**. Those ask different
questions and mixing them is the mistake this section exists to prevent.

### Present: Dice overlap

For a present structure,

    dice = 2 * both_voxels / (truth_voxels + pred_voxels)

and if `truth_voxels + pred_voxels` is 0 the pair is present-but-empty, which cannot happen by
definition, so that case is an error rather than a 1.

**Per structure**, the score is the mean Dice over the scans that contain it. **The headline is the
mean of those 117 numbers**, each structure counting once. It is never a mean over rows and never
weighted by volume: a voxel-weighted average is a report on the liver and the lungs with the other
115 structures rounded off.

A structure present in fewer than 10 scans of the split is reported with its count and excluded from
the headline mean, because a mean of three numbers is not a measurement. Which ones those are is
printed, never quietly dropped.

### Absent: how often it invents one

For an absent structure (`truth_voxels == 0`), Dice is undefined - the denominator is `pred_voxels`
alone and the numerator is always 0 - so it is not a Dice and is not averaged in as one. It is
reported as:

    invention rate = share of absent (scan, structure) pairs where pred_voxels > 0
    invented size  = median pred_voxels over those pairs

A segmenter that outlines a femur in a scan of a head is wrong in a way no Dice column would show,
and on this dataset the absent pairs are the majority: a scan contains about 69 of the 117
structures at the median.

## Strata, declared now

Declared in `datasets.py` before anything was trained, so none of them is a subset chosen after
seeing a result. `all` first, then: `abdomen`, `thorax`, `head_or_neck`, `siemens`, `not_siemens`,
`main_institute`, `other_institutes`, `with_pathology`.

`main_institute` against `other_institutes` is the one to watch: 898 of the 1,228 scans come from
one institute, so a method can look good by learning that institute's scanner. The gap between
those two strata is the honest measure of whether it generalises.

## Confidence intervals

The unit is the **scan**. The dataset publishes no patient identifier, so two scans of one person
could be treated as independent; that is a real limitation and it goes on the scorecard.

95% percentile bootstrap, **2,000 resamples, seed 20261001**: draw scans with replacement to the
same count, recompute every number from the drawn scans' rows, and take the 2.5th and 97.5th
percentiles. Two methods are compared by resampling the **same** scans for both and taking
percentiles of the difference, which is what makes the interval of the difference meaningful.

An equivalent fast form: for a fixed structure, the per-scan quantities `both`, `truth` and `pred`
are a vector per scan, so a resample is a matrix product between a 2,000 x n count matrix of how
often each scan was drawn and the n-vector of per-scan values. The whole scorecard is a handful of
matrix products rather than 2,000 passes over the rows. Either form is allowed; they must agree.

## The baselines, and the ship rule

Declared before the test is opened.

The reference is the best of these on the development split by headline mean Dice, chosen and
written down before the test split is unlocked:

- `always-empty` - predicts nothing, anywhere. Its Dice is 0 and its invention rate is 0. It is the
  floor that proves a metric is not rewarding silence.
- `where-it-usually-is` - for each structure, the cells of a normalised box where it turns up most
  often across development scans, as many cells as it usually occupies, mapped onto this scan's
  grid. A shape prior with no image in it.
- `nearest-scan` - classical single-atlas segmentation: find the development scan whose brightness
  profile and field of view most resemble this one, and copy its expert masks across. Decades old,
  no training, and the thing that makes this a contest rather than a formality.
- `bone-window` - take the better of those two regions and keep only the voxels whose Hounsfield
  brightness fits what the structure is made of. CT brightness means the same thing on every
  scanner, so this is physics rather than a guess.

All of them are computed **out of fold**: a development scan in fold 2 is predicted from folds 0, 1,
3 and 4. A prior or an atlas that has seen the scan it predicts has seen the answer.

**Changed on 2026-10-01, before anything was trained and before any scorecard existed.** This page
first said the prior would be "the voxels where the structure appears in more than half of the
development scans". Built, that rule left **all 117 structures completely empty**: every scan covers
a different part of a body, so once each is squashed into the same box, no cell holds the same organ
in half of them. A rule that produces nothing is not a baseline, and a reference that is too weak
makes the ship rule a formality, so the threshold became "the cells it turns up in most often, as
many as it usually occupies" and `nearest-scan` was added as a reference with some teeth. The
original wording is kept here rather than quietly replaced, because what a specification said before
a result is the only part of it worth anything.

The segmenter ships into the app only if, on the held-out test, the 95% confidence interval of its
headline mean Dice **minus** the reference's lies above zero, with both computed on the same scans.
Per-structure Dice, the invention rate and the stratum table are reported too, but they do not
decide shipping.

The test split is opened once, by `final_eval.py`, which refuses to run twice.

## The recomputation

Every number on the scorecard is recomputed by a second script, written by somebody who has read
this page and not the scorer, run against the same `predictions.csv`. The scorecard is only
published if the two agree on every cell. That is how the X-ray scorecard was checked, and it
caught nothing, which is the result worth having.

**So that two implementations can be compared by a machine rather than by eye, the names are fixed
here.** Both write a tab-separated line per number:

    stratum <TAB> method <TAB> metric <TAB> value <TAB> low <TAB> high

with `metric` one of exactly `headline_dice`, `invention_rate`, `invented_size_median` and
`n_eligible_structures` (and, when a reference is named, `headline_dice_minus_reference`, defined
below). Cells must agree to within 1e-9, and `nan` must match `nan`.

**Which numbers carry an interval:** `headline_dice` and `invention_rate` do (and
`headline_dice_minus_reference`, when it is written), and nothing else does.
`invented_size_median` and `n_eligible_structures` print `nan` for both bounds: they describe the
study rather than estimate a quantity, and a bootstrapped median needs a tie-breaking convention
that two implementations will not guess the same way. A method that never invents anything has no
median at all, and prints `nan` for the value too - never `0`, which would claim it invented things
of no size.

**A declared stratum that matches no scan is reported as empty**, by name, rather than left out.

This paragraph exists because the first time the two were written, they agreed on every number and
disagreed on what to call three of them, which is a comparison nobody can run.

## The difference the ship rule reads

**Added on 2026-10-01, before any segmenter had been scored on any split and before the test was
opened.** The ship rule above reads one number, the segmenter's headline mean Dice minus the
reference's, and until now that number had no name, so only one of the two implementations could
have produced it. A ship decision resting on the one number nobody recomputes would make the
recomputation a ceremony. So it gets a fifth metric name:

    headline_dice_minus_reference

- It is written only when a reference method is named, with `--reference <method>` on both
  scripts, and then for every stratum and for **every method other than the reference**.
- `value` is that method's `headline_dice` minus the reference's `headline_dice`, on the observed
  scans of the stratum. Each of the two is computed exactly as `headline_dice` is, with its own
  eligible structures fixed on the observed scans.
- `low` and `high` are the 2.5th and 97.5th percentiles of the same difference over the resamples.
  Each resample uses **the same drawn scans for both methods**: the difference is taken inside the
  resample and the percentiles are taken of the differences, never the other way round. A resample
  in which either headline is undefined is left out.
- With no reference named, neither script writes this metric, and the other four are unchanged.

The ship rule, restated in these terms: on the test split, stratum `all`, method `segmenter`,
`headline_dice_minus_reference` has `low` above zero.

## One label per voxel, and masks that overlap

**Noted on 2026-10-01, when the training cache was built, before any segmenter had been scored.**
`truth_voxels` is counted in each structure's own expert mask, as this page has always said. The
masks overlap a little: across the development scans, 0.15% of marked voxels sit in more than one
mask. So a scan's truth counts sum to slightly more than its marked voxels, and a method that
answers the question as asked, one label per voxel, cannot score a Dice of exactly 1 on every
structure at once. That is a property of the question and the data, not an error in the counts, and
nothing about the counting changes: `both_voxels` for a structure is the voxels the method gave that
structure's label which that structure's mask contains.
