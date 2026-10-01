"""Independent recomputation of every checked number in SCORECARD.md.

Written from METRICS_SPEC.md alone, without seeing score.py: it shares no code with the scorer and
reads only predictions.csv, SPIDER's radiological_gradings.csv and SCORECARD.md. Every row of the
four checked tables is recomputed with numpy, pandas and the standard library; a cell that differs
by more than 0.005 (counts: at all), a missing row or an extra row is a failure. Every split with
rows in predictions.csv must be reported in tables 1-3, and predictions.csv itself must follow the
spec's input rules.

    python recompute.py [--predictions P] [--gradings G] [--scorecard S]

Exit code 0 when everything matches, 1 otherwise.
"""

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def find_repo(start):
    """The nearest folder at or above `start` holding core/paths.py (found, never imported)."""
    for folder in (start, *start.parents):
        if (folder / 'core' / 'paths.py').is_file():
            return folder
    sys.exit(f'recompute: FAILED, no repository root (a folder with core/paths.py) above {start}')


REPO = find_repo(HERE)
DEFAULT_PREDICTIONS = REPO / 'work' / 'eval' / 'predictions.csv'
DEFAULT_GRADINGS = REPO.parent / 'Datasets' / 'public' / 'spider' / 'radiological_gradings.csv'
DEFAULT_SCORECARD = HERE / 'SCORECARD.md'

SEED = 20260930
N_RESAMPLES = 2000
TOLERANCE = 0.005

# binary item -> gradings column ('modic' is present when the Modic type is above 0)
BINARY_ITEMS = {
    'herniation': 'Disc herniation',
    'narrowing': 'Disc narrowing',
    'bulging': 'Disc bulging',
    'spondylolisthesis': 'Spondylolisthesis',
    'up_endplate': 'UP endplate',
    'low_endplate': 'LOW endplate',
    'modic': 'Modic',
}
ITEMS = ['pfirrmann'] + list(BINARY_ITEMS)
NUMERIC_COLUMNS = {'patient', 'ivd', 'fold', 'dice', 'matched', 'in_scope'}
REQUIRED_COLUMNS = ['method', 'localisation', 'split', 'fold', 'patient', 'ivd', 'vendor',
                    'nominal_level', 'tss_level', 'dice', 'matched', 'in_scope', 'drop_reason']
ALLOWED_VALUES = {
    'method': {'rules', 'panel', 'grader', 'medgemma', 'combo'},
    'localisation': {'tss', 'spider_masks'},
    'split': {'dev', 'test'},
    'vendor': {'Philips', 'Siemens'},
}
GROUP_COLUMNS = ['method', 'localisation', 'split']
ROW_KEY = GROUP_COLUMNS + ['patient', 'ivd']
# SPIDER's bottom-up level per IVD label; past IVD 9 the level is empty
NOMINAL_LEVELS = {1: 'L5-S1', 2: 'L4-L5', 3: 'L3-L4', 4: 'L2-L3', 5: 'L1-L2', 6: 'T12-L1'}
MATCH_DICE = 0.3
SPLIT_TABLES = ['accounting', 'overall', 'vendor']

TABLE_HEADERS = {
    'accounting': '| split | quantity | value |',
    'overall': '| split | item | metric | method | localisation | n_discs | n_patients | value '
               '| ci_low | ci_high | n_boot |',
    'vendor': '| split | vendor | item | metric | method | localisation | n_discs | n_patients | value '
              '| ci_low | ci_high | n_boot |',
    'paired': '| split | item | metric | method | reference | n_discs | n_patients | difference '
              '| ci_low | ci_high | n_boot |',
}
KEY_COLUMNS = {
    'accounting': ['split', 'quantity'],
    'overall': ['split', 'item', 'metric', 'method', 'localisation'],
    'vendor': ['split', 'vendor', 'item', 'metric', 'method', 'localisation'],
    'paired': ['split', 'item', 'metric', 'method', 'reference'],
}


def display_path(path):
    """A path relative to the repository root; a file outside it by its name alone."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name


def fatal(message):
    print(f'recompute: FAILED, {message}')
    sys.exit(1)


def load_gradings(path):
    """Truth per (patient, ivd), plus the raw Modic type for the accounting counts."""
    raw = pd.read_csv(path)
    truth = pd.DataFrame({
        'patient': raw['Patient'].astype(int),
        'ivd': raw['IVD label'].astype(int),
        'pfirrmann_truth': raw['Pfirrman grade'],
        'modic_raw': raw['Modic'],
    })
    for item, column in BINARY_ITEMS.items():
        if item == 'modic':
            truth['modic_truth'] = (raw['Modic'] > 0).astype(float).where(raw['Modic'].notna())
        else:
            truth[f'{item}_truth'] = raw[column]
    return truth


def load_predictions(path, truth):
    """predictions.csv joined to the truth; only empty cells are missing."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        fatal(f'predictions.csv lacks the columns {", ".join(missing)}')
    for column in raw.columns:
        if column in NUMERIC_COLUMNS or column.endswith('_pred') or column.endswith('_score'):
            raw[column] = pd.to_numeric(raw[column])
        else:
            raw[column] = raw[column].fillna('')
    for column in ('in_scope', 'matched'):
        # an empty cell is an input error; remember it so validate_predictions reports it
        raw[f'{column}_empty'] = raw[column].isna()
        raw[column] = raw[column].fillna(0).astype(int)
    raw['patient'] = raw['patient'].astype(int)
    raw['ivd'] = raw['ivd'].astype(int)
    return raw.merge(truth, on=['patient', 'ivd'], how='left', validate='many_to_one')


def validate_predictions(predictions, truth, problems):
    """The predictions.csv rules of the spec that every recomputed number rests on."""
    def flag(what, mask):
        if mask.any():
            examples = predictions.loc[mask, ROW_KEY].head(3).astype(str).agg('/'.join, axis=1)
            problems.append(f'INPUT {what}: {int(mask.sum())} rows (e.g. {", ".join(examples)})')

    for column, allowed in ALLOWED_VALUES.items():
        flag(f'{column} outside {sorted(allowed)}', ~predictions[column].isin(allowed))
    flag('duplicate (method, localisation, split, patient, ivd)',
         predictions.duplicated(ROW_KEY, keep=False))
    graded = predictions[['patient', 'ivd']].merge(truth[['patient', 'ivd']], how='left',
                                                   indicator=True)['_merge'] == 'both'
    flag('disc not in radiological_gradings.csv', ~graded.to_numpy())

    # every graded disc of every patient a group covers has a row
    covered = predictions[GROUP_COLUMNS + ['patient']].drop_duplicates().merge(
        truth[['patient', 'ivd']], on='patient')
    absent = covered.merge(predictions[ROW_KEY].drop_duplicates(), on=ROW_KEY, how='left',
                           indicator=True)
    absent = absent[absent['_merge'] == 'left_only']
    if len(absent):
        examples = absent[ROW_KEY].head(3).astype(str).agg('/'.join, axis=1)
        problems.append(f'INPUT graded discs of a covered patient without a row: {len(absent)} '
                        f'(e.g. {", ".join(examples)})')

    dev = predictions['split'] == 'dev'
    fold = predictions['fold']
    flag('dev row without a fold in 0-4', dev & ~fold.isin([0, 1, 2, 3, 4]))
    flag('test row with a fold', (predictions['split'] == 'test') & fold.notna())
    flag('empty in_scope', predictions['in_scope_empty'])
    flag('empty matched', predictions['matched_empty'])
    flag('in_scope not 0/1', ~predictions['in_scope'].isin([0, 1]))
    flag(f'matched differs from dice >= {MATCH_DICE}',
         predictions['matched'] != (predictions['dice'] >= MATCH_DICE).astype(int))
    has_reason = predictions['drop_reason'] != ''
    flag('in_scope = 1 with a drop_reason', (predictions['in_scope'] == 1) & has_reason)
    flag('in_scope = 0 without a drop_reason', (predictions['in_scope'] == 0) & ~has_reason)
    nominal = predictions['ivd'].map(NOMINAL_LEVELS)
    flag('nominal_level differs from the IVD label',
         nominal.notna() & (predictions['nominal_level'] != nominal))
    flag('nominal_level given past IVD 9',
         (predictions['ivd'] > 9) & (predictions['nominal_level'] != ''))
    # predictions are exact grades (1-5) or 0/1, never rounded into range; scores are unbounded
    for item in ITEMS:
        column = f'{item}_pred'
        if column not in predictions:
            continue
        allowed = [1, 2, 3, 4, 5] if item == 'pfirrmann' else [0, 1]
        prediction = predictions[column]
        flag(f'{column} not one of {allowed}', prediction.notna() & ~prediction.isin(allowed))


def split_cells(line):
    text = line.strip()
    if text.startswith('|'):
        text = text[1:]
    if text.endswith('|'):
        text = text[:-1]
    return [cell.strip().strip('`').strip() for cell in text.split('|')]


def format_key(key):
    return '/'.join(key)


def collect_scorecard_rows(text, problems):
    """Rows of every table whose header is exactly one of the four checked headers, by key."""
    rows = {name: {} for name in TABLE_HEADERS}
    names_by_header = {header: name for name, header in TABLE_HEADERS.items()}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        name = names_by_header.get(lines[i].strip())
        i += 1
        if name is None:
            continue
        columns = split_cells(TABLE_HEADERS[name])
        separator = lines[i].strip() if i < len(lines) else ''
        if separator.startswith('|') and set(separator) <= set('|-: '):
            i += 1
        while i < len(lines) and lines[i].strip().startswith('|'):
            cells = split_cells(lines[i])
            line_number = i + 1
            i += 1
            if len(cells) != len(columns):
                problems.append(f'MALFORMED {name} line {line_number}: {len(cells)} cells, '
                                f'expected {len(columns)}')
                continue
            row = dict(zip(columns, cells))
            key = tuple(row[column] for column in KEY_COLUMNS[name])
            if key in rows[name]:
                problems.append(f'DUPLICATE {name} {format_key(key)} (line {line_number})')
                continue
            rows[name][key] = row
    return rows


QWK_WEIGHTS = np.array([[(i - j) ** 2 / 16 for j in range(5)] for i in range(5)])


def ratio(sums):
    """Column 0 over column 1; undefined (NaN) where column 1 is 0."""
    result = np.full(len(sums), np.nan)
    defined = sums[:, 1] > 0
    result[defined] = sums[defined, 0] / sums[defined, 1]
    return result


def qwk(sums):
    """Quadratic weighted kappa from flattened 5x5 counts O[t, y] over grades 1..5."""
    observed = sums.reshape(-1, 5, 5)
    n = observed.sum(axis=(1, 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = (observed.sum(axis=2)[:, :, None] * observed.sum(axis=1)[:, None, :]
                    / n[:, None, None])
    numerator = (QWK_WEIGHTS * observed).sum(axis=(1, 2))
    denominator = (QWK_WEIGHTS * expected).sum(axis=(1, 2))
    result = np.full(len(sums), np.nan)
    defined = denominator > 0
    result[defined] = 1 - numerator[defined] / denominator[defined]
    return result


def auroc(sums):
    """Pair-counting AUROC from per-score-value counts: positives in the first half of the
    columns, negatives in the second, both ordered by ascending score. A positive beats every
    negative at a lower score and ties count one half."""
    k = sums.shape[1] // 2
    positives, negatives = sums[:, :k], sums[:, k:]
    negatives_below = np.cumsum(negatives, axis=1) - negatives
    numerator = (positives * (negatives_below + 0.5 * negatives)).sum(axis=1)
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)
    result = np.full(len(sums), np.nan)
    defined = denominator > 0
    result[defined] = numerator[defined] / denominator[defined]
    return result


def row_statistics(metric, t, v):
    """Additive per-disc statistics for a metric, and the function of their sums giving it.

    Every metric depends only on the multiset of discs, so summing the statistics of a resample
    equals computing the metric on the concatenated resample."""
    n = len(t)
    if metric in ('exact', 'within_one'):
        hit = (v == t) if metric == 'exact' else (np.abs(v - t) <= 1)
        return np.column_stack([hit, np.ones(n)]).astype(float), ratio
    if metric == 'sensitivity':
        return np.column_stack([(t == 1) & (v == 1), t == 1]).astype(float), ratio
    if metric == 'specificity':
        return np.column_stack([(t == 0) & (v == 0), t == 0]).astype(float), ratio
    if metric == 'qwk':
        rows = np.zeros((n, 25))
        rows[np.arange(n), (t - 1) * 5 + (v - 1)] = 1
        return rows, qwk
    if metric == 'auroc':
        values, rank = np.unique(v, return_inverse=True)
        rank = rank.ravel()
        k = len(values)
        rows = np.zeros((n, 2 * k))
        rows[np.arange(n), np.where(t == 1, rank, k + rank)] = 1
        return rows, auroc
    raise ValueError(f'unknown metric {metric}')


def sum_by_patient(rows, patient_index, n_patients):
    membership = np.zeros((n_patients, len(patient_index)))
    membership[patient_index, np.arange(len(patient_index))] = 1
    return membership @ rows


@functools.lru_cache(maxsize=None)
def patient_draw_counts(n_patients):
    """counts[b, k]: how many times patient P[k] is drawn in resample b.

    A fresh default_rng(SEED) per reported number gives the same draws for the same len(P), so
    the counts are cached by len(P)."""
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_patients, size=(N_RESAMPLES, n_patients))
    flat = (draws + n_patients * np.arange(N_RESAMPLES)[:, None]).ravel()
    counts = np.bincount(flat, minlength=N_RESAMPLES * n_patients)
    return counts.reshape(N_RESAMPLES, n_patients).astype(float)


def reported(x):
    return 'n/a' if np.isnan(x) else float(x)


def metric_numbers(metric, parts, patients):
    """Cells of one reported number. `parts` holds one (truth, filled prediction) pair, or two for
    a paired difference (first minus second); all are aligned with `patients`."""
    if len(patients) == 0:
        return {'n_discs': 0, 'n_patients': 0, 'value': 'n/a', 'ci_low': 'n/a', 'ci_high': 'n/a',
                'n_boot': 0}
    distinct, patient_index = np.unique(patients, return_inverse=True)
    patient_index = patient_index.ravel()
    per_patient, functions = [], []
    for t, v in parts:
        rows, function = row_statistics(metric, t, v)
        per_patient.append(sum_by_patient(rows, patient_index, len(distinct)))
        functions.append(function)

    def combine(sums):
        result = functions[0](sums[0])
        if len(functions) == 2:
            result = result - functions[1](sums[1])
        return result

    value = combine([s.sum(axis=0, keepdims=True) for s in per_patient])[0]
    counts = patient_draw_counts(len(distinct))
    boot = combine([counts @ s for s in per_patient])
    kept = boot[~np.isnan(boot)]
    ci_low, ci_high = np.percentile(kept, [2.5, 97.5]) if kept.size else (np.nan, np.nan)
    return {'n_discs': int(len(patients)), 'n_patients': int(len(distinct)),
            'value': reported(value), 'ci_low': reported(ci_low), 'ci_high': reported(ci_high),
            'n_boot': int(kept.size)}


def prepare_group(rows, name):
    """In-scope rows of one (method, localisation, split) group with each scored item's truth
    and worst-case-filled predictions and scores (fill values from the whole group)."""
    scored = rows[rows['in_scope'] == 1].reset_index(drop=True)
    group = {'patient': scored['patient'].to_numpy(dtype=int),
             'ivd': scored['ivd'].to_numpy(dtype=int),
             'vendor': scored['vendor'].to_numpy(dtype=str),
             'items': {}}
    for item in ITEMS:
        prediction = scored.get(f'{item}_pred')
        if prediction is None or not prediction.notna().any():
            continue
        truth = scored[f'{item}_truth']
        if truth.isna().any():
            fatal(f'{format_key(name)}: in-scope rows without a {item} grading')
        t = truth.to_numpy(dtype=float).astype(int)
        p = prediction.to_numpy(dtype=float)
        worst = np.where(t >= 3, 1, 5) if item == 'pfirrmann' else 1 - t
        y = np.where(np.isnan(p), worst, np.rint(p)).astype(int)
        allowed = (1, 2, 3, 4, 5) if item == 'pfirrmann' else (0, 1)
        if not np.isin(y, allowed).all():
            fatal(f'{format_key(name)}: {item}_pred outside {allowed}')
        entry = {'t': t, 'y': y, 'metrics': (['exact', 'within_one', 'qwk'] if item == 'pfirrmann'
                                             else ['sensitivity', 'specificity'])}
        score = scored.get(f'{item}_score')
        if item != 'pfirrmann' and score is not None and score.notna().any():
            s = score.to_numpy(dtype=float)
            present = s[~np.isnan(s)]
            fill = np.where(t == 1, present.min() - 1, present.max() + 1)
            entry['s'] = np.where(np.isnan(s), fill, s)
            entry['metrics'].append('auroc')
        group['items'][item] = entry
    return group


def item_number(entry, metric, mask, patients):
    values = entry['s'] if metric == 'auroc' else entry['y']
    return metric_numbers(metric, [(entry['t'][mask], values[mask])], patients[mask])


def paired_number(groups, split, item, metric, method, reference):
    """Difference method minus reference on the (patient, ivd) in scope for both tss groups;
    None when the scorecard row cannot exist."""
    first, second = groups.get((method, 'tss', split)), groups.get((reference, 'tss', split))
    if first is None or second is None:
        return None
    a, b = first['items'].get(item), second['items'].get(item)
    if a is None or b is None or metric not in a['metrics'] or metric not in b['metrics']:
        return None
    position_in_second = {key: i for i, key in enumerate(zip(second['patient'], second['ivd']))}
    in_first, in_second = [], []
    for i, key in enumerate(zip(first['patient'], first['ivd'])):
        if key in position_in_second:
            in_first.append(i)
            in_second.append(position_in_second[key])
    in_first, in_second = np.array(in_first, dtype=int), np.array(in_second, dtype=int)
    field = 's' if metric == 'auroc' else 'y'
    row = metric_numbers(metric,
                         [(a['t'][in_first], a[field][in_first]),
                          (b['t'][in_second], b[field][in_second])],
                         first['patient'][in_first])
    row['difference'] = row.pop('value')
    return row


def accounting(rows):
    """Data accounting quantities from the rules/tss rows of one split."""
    drop = rows['drop_reason']
    no_t2 = drop == 'no T2'
    outside_view = drop == "not in the T2's field of view"
    eligible = (rows['ivd'] >= 1) & ~no_t2 & ~outside_view
    matched = eligible & (rows['matched'] == 1)
    level_differs = matched & (rows['tss_level'] != rows['nominal_level'])
    scored = rows['in_scope'] == 1

    def rate(numerator, denominator):
        return numerator / denominator if denominator else 'n/a'

    quantities = {
        'graded_rows': len(rows),
        'patients': rows['patient'].nunique(),
        'excluded_no_t2_rows': no_t2.sum(),
        'excluded_no_t2_patients': rows.loc[no_t2, 'patient'].nunique(),
        'ivd0_rows': (drop == 'IVD label 0').sum(),
        'outside_t2_view_rows': outside_view.sum(),
        'eligible': eligible.sum(),
        'matched': matched.sum(),
        'level_differs': level_differs.sum(),
        'dropped_matched_outside': (drop == 'matched outside T12-L1..L5-S1').sum(),
        'dropped_unmatched_outside':
            (drop == 'unmatched, nominal level outside T12-L1..L5-S1').sum(),
        'scored': scored.sum(),
        'scored_unmatched': (scored & (rows['matched'] == 0)).sum(),
    }
    for modic_type in range(4):
        quantities[f'modic_type_{modic_type}'] = (scored & (rows['modic_raw'] == modic_type)).sum()
    quantities = {name: int(value) for name, value in quantities.items()}
    quantities['match_rate'] = rate(quantities['matched'], quantities['eligible'])
    quantities['level_mislabel_rate'] = rate(quantities['level_differs'], quantities['matched'])
    return quantities


def recompute_tables(predictions, splits, paired_keys):
    """Every row tables 1-3 should hold for the reported splits, and the required table-4 rows,
    keyed like the scorecard."""
    expected = {name: {} for name in TABLE_HEADERS}
    groups = {}
    for name, rows in predictions.groupby(['method', 'localisation', 'split'], sort=True):
        if name[2] in splits:
            groups[name] = prepare_group(rows, name)

    for split in sorted(splits):
        base = predictions[(predictions['method'] == 'rules')
                           & (predictions['localisation'] == 'tss')
                           & (predictions['split'] == split)]
        if len(base):
            for quantity, value in accounting(base).items():
                expected['accounting'][(split, quantity)] = {'value': value}

    for (method, localisation, split), group in groups.items():
        everyone = np.ones(len(group['patient']), dtype=bool)
        vendors = sorted(set(group['vendor']))
        for item, entry in group['items'].items():
            for metric in entry['metrics']:
                expected['overall'][(split, item, metric, method, localisation)] = item_number(
                    entry, metric, everyone, group['patient'])
                if localisation != 'tss':
                    continue
                for vendor in vendors:
                    expected['vendor'][(split, vendor, item, metric, method, localisation)] = (
                        item_number(entry, metric, group['vendor'] == vendor, group['patient']))

    for key in paired_keys:
        row = paired_number(groups, *key)
        if row is not None:
            expected['paired'][key] = row
    return expected


def required_paired_keys(predictions, found_paired, problems):
    """The table-4 rows the spec requires: test/pfirrmann/qwk, one row per method with test rows
    against one shared reference. The reference (the baseline chosen before the test split was
    opened) is not in predictions.csv, so it is read from the scorecard and must be unique."""
    test_methods = set(predictions.loc[predictions['split'] == 'test', 'method'])
    if not test_methods:
        return []
    references = sorted({key[4] for key in found_paired})
    if not references:
        problems.append('MISSING table paired: predictions.csv has test rows but the scorecard '
                        'has no paired-difference rows')
        return []
    if len(references) > 1:
        problems.append(f'MISMATCH table paired: one shared reference expected, the scorecard '
                        f'has {", ".join(references)}')
        return []
    reference = references[0]
    if reference not in test_methods:
        problems.append(f'MISMATCH table paired: reference {reference} has no test rows in '
                        f'predictions.csv')
        return []
    return [('test', 'pfirrmann', 'qwk', method, reference)
            for method in sorted(test_methods - {reference})]


def cell_matches(cell, value):
    """'n/a' only matches 'n/a'; counts must be equal; other numbers within TOLERANCE."""
    if isinstance(value, str):
        return cell == value
    if cell == 'n/a':
        return False
    try:
        number = float(cell)
    except ValueError:
        return False
    if isinstance(value, int):
        return number == value
    return abs(number - value) <= TOLERANCE + 1e-9


def shown(value):
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def compare(expected, found, problems):
    """Append a problem per missing, extra or differing row/cell; return how many cells were checked."""
    checked = 0
    for name in TABLE_HEADERS:
        wanted, present = expected[name], found[name]
        rows_checked = 0
        for key in wanted:
            if key not in present:
                problems.append(f'MISSING {name} {format_key(key)}: recomputed, not in the scorecard')
        for key, cells in present.items():
            if key not in wanted:
                problems.append(f'EXTRA {name} {format_key(key)}: in the scorecard, '
                                f'not produced by the recomputation')
                continue
            rows_checked += 1
            for column, value in wanted[key].items():
                checked += 1
                if not cell_matches(cells[column], value):
                    problems.append(f'MISMATCH {name} {format_key(key)} {column}: '
                                    f'scorecard={cells[column]} recomputed={shown(value)}')
        print(f'table {name}: {rows_checked} rows checked '
              f'({len(present)} in scorecard, {len(wanted)} recomputed)')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--gradings', type=Path, default=DEFAULT_GRADINGS)
    parser.add_argument('--scorecard', type=Path, default=DEFAULT_SCORECARD)
    args = parser.parse_args()

    for label, path in (('predictions', args.predictions), ('gradings', args.gradings),
                        ('scorecard', args.scorecard)):
        if not path.is_file():
            fatal(f'{label} not found: {path}')
        print(f'{label}: {display_path(path)} ({path.stat().st_size} bytes)')

    truth = load_gradings(args.gradings)
    predictions = load_predictions(args.predictions, truth)
    text = args.scorecard.read_text(encoding='utf-8')

    problems = []
    validate_predictions(predictions, truth, problems)
    found = collect_scorecard_rows(text, problems)
    splits = {key[0] for rows in found.values() for key in rows}
    if not splits:
        problems.append('MISSING tables: the scorecard has no rows under any checked header')

    # Every split with rows in predictions.csv is recomputed, so one the scorecard leaves out shows
    # up as missing rows. test is exempt only while predictions.csv holds no test rows.
    prediction_splits = set(predictions['split'])
    paired_keys = required_paired_keys(predictions, found['paired'], problems)
    expected = recompute_tables(predictions, splits | prediction_splits, paired_keys)
    for key in paired_keys:
        if key not in expected['paired']:
            problems.append(f'UNDEFINED paired {format_key(key)}: required by the spec, but the '
                            f'method or reference has no tss pfirrmann predictions on test')
    for split in sorted(prediction_splits):
        for name in SPLIT_TABLES:
            if (any(key[0] == split for key in expected[name])
                    and not any(key[0] == split for key in found[name])):
                note = (' (the scorecard says "Not yet run", which only holds while '
                        'predictions.csv has no test rows)'
                        if split == 'test' and 'Not yet run' in text else '')
                problems.append(f'MISSING split {split} in table {name}: predictions.csv has '
                                f'{int((predictions["split"] == split).sum())} {split} rows'
                                f'{note}')
    checked = compare(expected, found, problems)

    for problem in problems:
        print(problem)
    if problems:
        print(f'recompute: FAILED, {len(problems)} problems in {checked} numbers')
        return 1
    print(f'recompute: OK, {checked} numbers checked')
    return 0


if __name__ == '__main__':
    sys.exit(main())
