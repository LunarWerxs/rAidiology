#!/usr/bin/env python3
"""Independent recomputation of `SCORECARD.md` from `METRICS_SPEC.md` alone.

Reads the predictions and the two datasets' own index files, recomputes every number of every
table whose header the contract names, compares each cell against the card with a tolerance of
0.005, and exits non-zero if anything differs, is missing or is extra.

This file was written without ever reading the program that produced the card: it was written in a
clean room holding only the contract, the card and the three data files, by an agent that never saw
`score.py`. The two implementations are only worth comparing while they stay independent, so when
`score.py` changes, this file is not "fixed" to match it: either the contract is wrong, or one of the
two has a bug, and that is the thing to find.

Only the four default paths below were changed after it was written, to point at where those files
live in the repository. Nothing that computes a number was touched.

Usage: python recompute.py > recompute.log
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SEED = 20261001
N_BOOT = 2000
TOLERANCE = 0.005
DECIMALS = 3
# Resamples are processed in blocks: the per-score-value matrices of a large stratum are far too
# big to hold for all 2,000 resamples at once.
BLOCK = 250

METRIC_ORDER = ["auroc", "sensitivity", "specificity", "balanced_accuracy", "kappa"]

STRATA = {
    "grazpedwri": ["all", "initial_exam", "no_cast", "certain"],
    "fracatlas": ["all", "hand", "leg", "hip", "shoulder"],
}

IMAGE_KEY = {"grazpedwri": "filestem", "fracatlas": "image_id"}
TRUTH_COLUMN = {"grazpedwri": "fracture_visible", "fracatlas": "fractured"}

ACCOUNTING_HEADER = ("dataset", "split", "quantity", "value")
SCORES_HEADER = (
    "dataset", "split", "stratum", "metric", "method",
    "n_images", "n_patients", "value", "ci_low", "ci_high", "n_boot",
)
PAIRED_HEADER = (
    "dataset", "split", "stratum", "metric", "method", "reference",
    "n_images", "n_patients", "difference", "ci_low", "ci_high", "n_boot",
)

ACCOUNTING_KEYS = ("dataset", "split", "quantity")
ACCOUNTING_VALUES = ("value",)
SCORES_KEYS = ("dataset", "split", "stratum", "metric", "method")
SCORES_VALUES = ("n_images", "n_patients", "value", "ci_low", "ci_high", "n_boot")
PAIRED_KEYS = ("dataset", "split", "stratum", "metric", "method", "reference")
PAIRED_VALUES = ("n_images", "n_patients", "difference", "ci_low", "ci_high", "n_boot")



def read_strings(path: Path, encoding: str) -> pd.DataFrame:
    """Every field as a string, blanks as empty strings: a blank turned into NaN would silently
    change truth, scope and fill decisions."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding=encoding)


def index_frame(raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    key = IMAGE_KEY[dataset]
    if key not in raw.columns:
        raise SystemExit(f"{dataset}: index file has no column {key!r}")
    out = pd.DataFrame({"image": raw[key].to_numpy()})
    out["truth"] = (raw[TRUTH_COLUMN[dataset]].to_numpy() == "1").astype(np.int8)
    if dataset == "grazpedwri":
        out["st_all"] = True
        out["st_initial_exam"] = raw["initial_exam"].to_numpy() == "1"
        out["st_no_cast"] = raw["cast"].to_numpy() != "1"
        out["st_certain"] = raw["diagnosis_uncertain"].to_numpy() != "1"
    else:
        out["st_all"] = True
        for part in ("hand", "leg", "hip", "shoulder"):
            out["st_" + part] = raw[part].to_numpy() == "1"
    if out["image"].duplicated().any():
        raise SystemExit(f"{dataset}: index file has duplicate image keys")
    return out


def joined_predictions(preds: pd.DataFrame, indices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for dataset, index in indices.items():
        sub = preds[preds["dataset"] == dataset]
        if sub.empty:
            continue
        unknown = set(sub["image"]) - set(index["image"])
        if unknown:
            sample = ", ".join(sorted(unknown)[:5])
            raise SystemExit(
                f"{dataset}: {len(unknown)} predicted image(s) are not keys of the dataset index "
                f"file, which the contract calls an error, not a drop: {sample}"
            )
        parts.append(sub.merge(index, on="image", how="left", validate="many_to_one"))
    if not parts:
        raise SystemExit("predictions.csv holds no rows of a known dataset")
    out = pd.concat(parts, ignore_index=True)
    out["patient"] = out["patient"].str.strip()
    return out



def metric_values(name: str, tp, fp, fn, tn):
    """The metric of every row of a set of confusion totals, plus a mask of where it is defined."""
    tp, fp, fn, tn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    pos, neg = tp + fn, tn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(pos > 0, tp / np.where(pos > 0, pos, 1.0), np.nan)
        spec = np.where(neg > 0, tn / np.where(neg > 0, neg, 1.0), np.nan)
        if name == "sensitivity":
            value = sens
        elif name == "specificity":
            value = spec
        elif name == "balanced_accuracy":
            value = (sens + spec) / 2.0
        elif name == "kappa":
            safe_n = np.where(n > 0, n, 1.0)
            po = np.where(n > 0, (tp + tn) / safe_n, np.nan)
            pe = np.where(
                n > 0,
                ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (safe_n * safe_n),
                np.nan,
            )
            usable = (n > 0) & (pe != 1.0)
            value = np.where(usable, (po - pe) / np.where(usable, 1.0 - pe, 1.0), np.nan)
        else:
            raise ValueError(f"not a counting metric: {name}")
    return value, np.isfinite(value)


def auroc_values(pos, neg):
    """AUROC of every column of per-score-value positive and negative counts, values ascending.

    This is the contract's pair counting: each positive beats every negative strictly below its
    value and ties with half of those at it.
    """
    cumulative = np.cumsum(neg, axis=0)
    below = cumulative - neg
    numerator = (pos * (below + 0.5 * neg)).sum(axis=0)
    n_pos = pos.sum(axis=0)
    n_neg = neg.sum(axis=0)
    denominator = n_pos * n_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(
            denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), np.nan
        )
    return value, np.isfinite(value)



_counts_cache: dict[int, np.ndarray] = {}


def resample_counts(n_units: int) -> np.ndarray:
    """How often each unit appears in each of the 2,000 resamples.

    A fresh `default_rng(SEED)` asked for the same size always returns the same draws, so the
    matrix depends only on the number of units and is cached across numbers.
    """
    cached = _counts_cache.get(n_units)
    if cached is not None:
        return cached
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_units, size=(N_BOOT, n_units))
    offsets = (np.arange(N_BOOT, dtype=np.int64) * n_units)[:, None]
    counts = (
        np.bincount((draws + offsets).ravel(), minlength=N_BOOT * n_units)
        .reshape(N_BOOT, n_units)
        .astype(np.int32)
    )
    if len(_counts_cache) >= 3:
        _counts_cache.pop(next(iter(_counts_cache)))
    _counts_cache[n_units] = counts
    return counts


def percentile_interval(kept: np.ndarray):
    if kept.size == 0:
        return None, None, 0
    low, high = np.percentile(kept, [2.5, 97.5])
    return float(low), float(high), int(kept.size)


def unit_index(patients: np.ndarray):
    """The sorted distinct patients and each row's position among them."""
    units, position = np.unique(patients, return_inverse=True)
    return units, position.astype(np.int64)


def counting_cells(name: str, truth: np.ndarray, pred: np.ndarray, patients: np.ndarray) -> dict:
    """Point value and interval of a counting metric, resampling patients.

    Every metric but AUROC sees a resample only through its confusion totals, so each patient's
    own totals are counted once and the resamples' totals are a single matrix product.
    """
    units, position = unit_index(patients)
    n_units = units.size
    category = np.where(truth == 1, np.where(pred == 1, 0, 2), np.where(pred == 1, 1, 3))
    per_unit = (
        np.bincount(position * 4 + category, minlength=n_units * 4)
        .reshape(n_units, 4)
        .astype(np.float64)
    )
    totals = per_unit.sum(axis=0)[None, :]
    point, defined = metric_values(name, totals[:, 0], totals[:, 1], totals[:, 2], totals[:, 3])
    value = float(point[0]) if defined[0] else None
    if n_units == 0:
        return {"value": value, "ci_low": None, "ci_high": None, "n_boot": 0}
    counts = resample_counts(n_units)
    kept = []
    for start in range(0, N_BOOT, BLOCK):
        block = counts[start : start + BLOCK].astype(np.float64)
        resampled = block @ per_unit
        values, ok = metric_values(
            name, resampled[:, 0], resampled[:, 1], resampled[:, 2], resampled[:, 3]
        )
        kept.append(values[ok])
    low, high, n_boot = percentile_interval(np.concatenate(kept))
    return {"value": value, "ci_low": low, "ci_high": high, "n_boot": n_boot}


def value_count_matrix(n_values, n_units, value_index, position, mask):
    """Score value by unit counts, transposed so a block of resamples is one sparse product."""
    rows = value_index[mask]
    cols = position[mask]
    data = np.ones(rows.size, dtype=np.float64)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_values, n_units)).tocsr()


def score_profile(scores: np.ndarray, truth: np.ndarray, patients: np.ndarray):
    units, position = unit_index(patients)
    values = np.unique(scores)
    value_index = np.searchsorted(values, scores)
    positives = value_count_matrix(values.size, units.size, value_index, position, truth == 1)
    negatives = value_count_matrix(values.size, units.size, value_index, position, truth == 0)
    return units.size, positives, negatives


def auroc_cells(truth: np.ndarray, scores: np.ndarray, patients: np.ndarray) -> dict:
    n_units, positives, negatives = score_profile(scores, truth, patients)
    total_pos = np.asarray(positives.sum(axis=1))
    total_neg = np.asarray(negatives.sum(axis=1))
    point, defined = auroc_values(total_pos, total_neg)
    value = float(point[0]) if defined[0] else None
    if n_units == 0:
        return {"value": value, "ci_low": None, "ci_high": None, "n_boot": 0}
    counts = resample_counts(n_units)
    kept = []
    for start in range(0, N_BOOT, BLOCK):
        block = np.ascontiguousarray(counts[start : start + BLOCK].T.astype(np.float64))
        values, ok = auroc_values(positives.dot(block), negatives.dot(block))
        kept.append(values[ok])
    low, high, n_boot = percentile_interval(np.concatenate(kept))
    return {"value": value, "ci_low": low, "ci_high": high, "n_boot": n_boot}


def paired_auroc_cells(truth, scores_a, scores_b, patients) -> dict:
    """Difference of two methods' AUROC on the shared rows, both computed on each resample."""
    units, position = unit_index(patients)
    n_units = units.size
    sides = []
    for scores in (scores_a, scores_b):
        values = np.unique(scores)
        value_index = np.searchsorted(values, scores)
        sides.append(
            (
                value_count_matrix(values.size, n_units, value_index, position, truth == 1),
                value_count_matrix(values.size, n_units, value_index, position, truth == 0),
            )
        )
    point = []
    for positives, negatives in sides:
        one, ok = auroc_values(np.asarray(positives.sum(axis=1)), np.asarray(negatives.sum(axis=1)))
        point.append((one[0], ok[0]))
    value = float(point[0][0] - point[1][0]) if point[0][1] and point[1][1] else None
    if n_units == 0:
        return {"value": value, "ci_low": None, "ci_high": None, "n_boot": 0}
    counts = resample_counts(n_units)
    kept = []
    for start in range(0, N_BOOT, BLOCK):
        block = np.ascontiguousarray(counts[start : start + BLOCK].T.astype(np.float64))
        first, ok_first = auroc_values(sides[0][0].dot(block), sides[0][1].dot(block))
        second, ok_second = auroc_values(sides[1][0].dot(block), sides[1][1].dot(block))
        both = ok_first & ok_second
        kept.append((first - second)[both])
    low, high, n_boot = percentile_interval(np.concatenate(kept))
    return {"value": value, "ci_low": low, "ci_high": high, "n_boot": n_boot}



class Group:
    """The in-scope rows of one (method, dataset, split), with misses already filled."""

    def __init__(self, method: str, dataset: str, split: str, rows: pd.DataFrame):
        self.method = method
        self.dataset = dataset
        self.split = split
        in_scope = rows[rows["in_scope"] == "1"]
        self.image = in_scope["image"].to_numpy()
        self.patient = in_scope["patient"].to_numpy()
        self.truth = in_scope["truth"].to_numpy().astype(np.int8)
        self.strata = {
            stratum: in_scope["st_" + stratum].to_numpy().astype(bool)
            for stratum in STRATA[dataset]
        }

        raw_pred = in_scope["fracture_pred"].to_numpy()
        missing_pred = raw_pred == ""
        stated = np.where(missing_pred, "0", raw_pred).astype(np.float64)
        # A miss stays in n and is answered as badly as possible, so refusing cannot pay.
        self.pred = np.where(missing_pred, 1 - self.truth, stated).astype(np.int8)
        self.scored = bool((~missing_pred).any())

        raw_score = in_scope["fracture_score"].to_numpy()
        missing_score = raw_score == ""
        self.has_score = bool((~missing_score).any())
        if self.has_score:
            stated_score = np.where(missing_score, "0", raw_score).astype(np.float64)
            # One minimum and maximum for the whole group, before any stratum or resample.
            low = stated_score[~missing_score].min()
            high = stated_score[~missing_score].max()
            self.score = np.where(
                missing_score, np.where(self.truth == 1, low - 1.0, high + 1.0), stated_score
            )
        else:
            self.score = None


def build_groups(joined: pd.DataFrame) -> dict[tuple[str, str, str], Group]:
    groups = {}
    for (method, dataset, split), rows in joined.groupby(["method", "dataset", "split"], sort=True):
        groups[(method, dataset, split)] = Group(method, dataset, split, rows)
    return groups



def accounting_table(joined: pd.DataFrame, problems: list[str]) -> dict:
    expected = {}
    for (dataset, split), rows in joined.groupby(["dataset", "split"], sort=True):
        book = rows[rows["method"] == "prevalence"]
        if book.empty:
            problems.append(
                f"data accounting: ({dataset}, {split}) has prediction rows but no `prevalence` "
                "rows to account for them"
            )
            continue
        in_scope = book[book["in_scope"] == "1"]
        truth = in_scope["truth"].to_numpy().astype(np.int8)
        n_in_scope = len(in_scope)
        quantities = [
            ("rows", float(len(book))),
            ("patients", float(book["patient"].nunique())),
            ("in_scope_rows", float(n_in_scope)),
            ("unreadable_rows", float((book["drop_reason"] == "unreadable image").sum())),
            ("fracture_rows", float((truth == 1).sum())),
            ("prevalence", float((truth == 1).sum() / n_in_scope) if n_in_scope else None),
        ]
        for stratum in STRATA[dataset][1:]:
            mask = in_scope["st_" + stratum].to_numpy().astype(bool)
            quantities.append((f"{stratum}_rows", float(mask.sum())))
            quantities.append(
                (
                    f"{stratum}_prevalence",
                    float((truth[mask] == 1).sum() / mask.sum()) if mask.sum() else None,
                )
            )
        if split == "dev":
            for fold in range(5):
                fold_rows = book[book["fold"] == str(fold)]
                quantities.append((f"fold_{fold}_patients", float(fold_rows["patient"].nunique())))
        for quantity, value in quantities:
            expected[(dataset, split, quantity)] = {"value": value}
    return expected


def scores_table(groups: dict) -> dict:
    expected = {}
    for (method, dataset, split), group in groups.items():
        if not group.scored:
            continue
        for stratum in STRATA[dataset]:
            mask = group.strata[stratum]
            patients = group.patient[mask]
            truth = group.truth[mask]
            common = {
                "n_images": float(mask.sum()),
                "n_patients": float(np.unique(patients).size),
            }
            for metric in METRIC_ORDER:
                if metric == "auroc":
                    if not group.has_score:
                        continue
                    cells = auroc_cells(truth, group.score[mask], patients)
                else:
                    cells = counting_cells(metric, truth, group.pred[mask], patients)
                expected[(dataset, split, stratum, metric, method)] = {**common, **cells}
    return expected


def paired_table(groups: dict, card_rows: list[dict], problems: list[str]) -> dict:
    """Paired differences of the final evaluation, against the reference the card names.

    The contract fixes everything about these rows but which method is the baseline, which it
    says was chosen before the test split was opened; the card carries that choice, so it is read
    from there rather than guessed.
    """
    test_methods = sorted({key[0] for key in groups if key[2] == "test"})
    test_datasets = sorted({key[1] for key in groups if key[2] == "test"})
    if not test_methods:
        return {}
    references = {row["reference"] for row in card_rows}
    if not card_rows:
        problems.append(
            "paired differences: predictions.csv holds test rows but the scorecard has no "
            "paired-difference table to check"
        )
        return {}
    if len(references) != 1:
        problems.append(
            "paired differences: the scorecard's rows name "
            f"{len(references)} references ({sorted(references)}), but the contract allows one"
        )
        return {}
    reference = references.pop()
    expected = {}
    for dataset in test_datasets:
        base = groups.get((reference, dataset, "test"))
        if base is None or not base.has_score:
            problems.append(
                f"paired differences: no scored reference `{reference}` for ({dataset}, test)"
            )
            continue
        for method in test_methods:
            if method == reference:
                continue
            other = groups.get((method, dataset, "test"))
            if other is None or not other.has_score:
                continue
            shared = np.intersect1d(base.image, other.image)
            base_at = {image: i for i, image in enumerate(base.image)}
            other_at = {image: i for i, image in enumerate(other.image)}
            base_rows = np.array([base_at[image] for image in shared], dtype=np.int64)
            other_rows = np.array([other_at[image] for image in shared], dtype=np.int64)
            for stratum in STRATA[dataset]:
                keep = base.strata[stratum][base_rows]
                rows_a = base_rows[keep]
                rows_b = other_rows[keep]
                patients = base.patient[rows_a]
                cells = paired_auroc_cells(
                    base.truth[rows_a], other.score[rows_b], base.score[rows_a], patients
                )
                expected[(dataset, "test", stratum, "auroc", method, reference)] = {
                    "n_images": float(rows_a.size),
                    "n_patients": float(np.unique(patients).size),
                    "difference": cells["value"],
                    "ci_low": cells["ci_low"],
                    "ci_high": cells["ci_high"],
                    "n_boot": float(cells["n_boot"]),
                }
    return expected



SEPARATOR = re.compile(r"^:?-{1,}:?$")


def split_row(line: str):
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    body = stripped[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def is_separator(cells) -> bool:
    return bool(cells) and all(SEPARATOR.match(cell) for cell in cells)


def parse_card(text: str) -> dict[tuple, list[list[str]]]:
    tables: dict[tuple, list[list[str]]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = split_row(lines[i])
        following = split_row(lines[i + 1]) if i + 1 < len(lines) else None
        if header is None or following is None or not is_separator(following):
            i += 1
            continue
        rows = []
        j = i + 2
        while j < len(lines):
            cells = split_row(lines[j])
            if cells is None:
                break
            rows.append(cells)
            j += 1
        tables.setdefault(tuple(header), []).extend(rows)
        i = j
    return tables


def card_rows(tables: dict, header: tuple, key_columns: tuple, problems: list[str]) -> dict:
    rows = {}
    for cells in tables.get(header, []):
        if len(cells) != len(header):
            problems.append(
                f"scorecard: a row of the {header[2]} table has {len(cells)} cells, "
                f"not {len(header)}: {cells}"
            )
            continue
        row = dict(zip(header, cells))
        key = tuple(row[column] for column in key_columns)
        if key in rows:
            problems.append(f"scorecard: duplicate row {key}")
            continue
        rows[key] = row
    return rows



def show(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{DECIMALS}f}"


def compare_table(name, expected, card, key_columns, value_columns, problems):
    checked = 0
    differed = 0
    for key in sorted(expected, key=lambda k: sort_key(key_columns, k)):
        row = card.get(key)
        if row is None:
            problems.append(f"{name}: row {' | '.join(key)} is recomputed but not in the scorecard")
            continue
        for column in value_columns:
            checked += 1
            mine = expected[key][column]
            theirs = row[column]
            if not cells_agree(mine, theirs):
                differed += 1
                problems.append(
                    f"{name}: {' | '.join(key)} | {column}: "
                    f"scorecard {theirs} vs recomputed {show(mine)}"
                )
    only_card = [key for key in card if key not in expected]
    for key in sorted(only_card):
        problems.append(f"{name}: row {' | '.join(key)} is in the scorecard but not recomputed")
    return {
        "checked": checked,
        "differed": differed,
        "only_recomputed": sum(1 for key in expected if key not in card),
        "only_card": len(only_card),
    }


def cells_agree(mine, theirs: str) -> bool:
    theirs = theirs.strip()
    if mine is None or theirs == "n/a":
        return mine is None and theirs == "n/a"
    try:
        stated = float(theirs)
    except ValueError:
        return False
    return abs(stated - mine) <= TOLERANCE


def sort_key(key_columns, key):
    """The card's order: columns left to right as strings, but `all` first and metrics in the
    contract's order."""
    parts = []
    for column, value in zip(key_columns, key):
        if column == "metric":
            parts.append(
                f"{METRIC_ORDER.index(value):02d}" if value in METRIC_ORDER else f"99{value}"
            )
        elif column == "stratum":
            parts.append("0" if value == "all" else "1" + value)
        else:
            parts.append(value)
    return tuple(parts)



def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(here).parents[2]                      # the repository root
    beside = root.parent / "Datasets" / "public"      # core/paths.py: datasets sit beside the repo
    work = root / "work" / "eval" / "xray"
    parser.add_argument("--predictions", type=Path, default=work / "predictions.csv")
    parser.add_argument("--scorecard", type=Path, default=here / "SCORECARD.md")
    parser.add_argument("--grazpedwri", type=Path, default=beside / "grazpedwri-dx" / "dataset.csv")
    parser.add_argument("--fracatlas", type=Path, default=beside / "fracatlas" / "FracAtlas" / "dataset.csv")
    args = parser.parse_args(argv)

    started = time.perf_counter()

    preds = read_strings(args.predictions, "utf-8")
    indices = {
        "grazpedwri": index_frame(read_strings(args.grazpedwri, "utf-8-sig"), "grazpedwri"),
        "fracatlas": index_frame(read_strings(args.fracatlas, "utf-8"), "fracatlas"),
    }
    joined = joined_predictions(preds, indices)
    groups = build_groups(joined)

    tables = parse_card(args.scorecard.read_text(encoding="utf-8"))
    problems: list[str] = []
    card_accounting = card_rows(tables, ACCOUNTING_HEADER, ACCOUNTING_KEYS, problems)
    card_scores = card_rows(tables, SCORES_HEADER, SCORES_KEYS, problems)
    card_paired = card_rows(tables, PAIRED_HEADER, PAIRED_KEYS, problems)

    expected_accounting = accounting_table(joined, problems)
    expected_scores = scores_table(groups)
    expected_paired = paired_table(groups, list(card_paired.values()), problems)

    reports = [
        (
            "table 1 (data accounting)",
            compare_table(
                "table 1", expected_accounting, card_accounting,
                ACCOUNTING_KEYS, ACCOUNTING_VALUES, problems,
            ),
        ),
        (
            "table 2 (scores)",
            compare_table(
                "table 2", expected_scores, card_scores, SCORES_KEYS, SCORES_VALUES, problems
            ),
        ),
        (
            "table 3 (paired differences)",
            compare_table(
                "table 3", expected_paired, card_paired, PAIRED_KEYS, PAIRED_VALUES, problems
            ),
        ),
    ]

    for label, report in reports:
        print(
            f"{label}: {report['checked']} cells checked, {report['differed']} differing, "
            f"{report['only_card']} row(s) only in the scorecard, "
            f"{report['only_recomputed']} row(s) only recomputed"
        )

    if problems:
        print()
        for problem in problems:
            print(f"MISMATCH {problem}")

    total_checked = sum(report["checked"] for _, report in reports)
    total_differed = sum(report["differed"] for _, report in reports)
    total_rows = sum(report["only_card"] + report["only_recomputed"] for _, report in reports)
    other = len(problems) - total_differed - total_rows
    print()
    print(
        f"TOTAL: {total_checked} cells checked, {total_differed} differing, "
        f"{total_rows} row discrepancies, {max(other, 0)} other problem(s)"
    )
    print(f"wall time {time.perf_counter() - started:.1f} s")
    verdict = not problems
    print("PASS: the scorecard agrees with the recomputation" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
