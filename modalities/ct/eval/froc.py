"""FROC scoring for lung-nodule detection, a Python 3 port of the official LUNA16 evaluation.

This is a port of the official LUNA16 scoring script (noduleCADEvaluationLUNA16.py, distributed with
the challenge under CC BY 4.0). Cite: Setio et al., "Validation, comparison, and combination of
algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16
challenge", Medical Image Analysis 2017. The semantics are the official ones; only the speed
differs (numpy per scan instead of Python loops), and `selfcheck` proves the two agree on the
official sample submission.

What is scored. A submission is a list of candidates, each a point in a scan (seriesuid, x, y, z in
millimetres) with a probability. A reference nodule counts as found when a candidate lies strictly
closer to its centre than its radius (half its diameter). Several candidates on one nodule count
once, with the highest score; the others are "double detections", ignored. A candidate that hits no
reference nodule is a false positive. A nodule no candidate hits is a miss (it still counts in the
denominator, at the lowest possible score).

Excluded findings. LUNA16 keeps a second list of findings that are not clear enough to be required
(for example nodules under 3 mm, or marks only one or two radiologists made). A candidate on one of
these is neither a hit nor a false alarm: it is dropped. A system is not punished for finding
something the readers were unsure about, and not rewarded either. A candidate already claimed by a
reference nodule is a hit, even if it also sits on an excluded finding.

FROC and CPM. Sweeping a threshold over the probabilities gives a curve of sensitivity (share of
reference nodules found) against false positives per scan. CPM, the competition performance metric,
is the mean sensitivity read off that curve at 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives per
scan. Confidence intervals resample SCANS with replacement (as the official script does), rebuild
the curve each time, and take the 2.5th and 97.5th percentiles.

Official details kept: distance test is strict (<); a finding with no size (diameter < 0) is
treated as 1 mm wide (see UNSIZED_DIAMETER_MM: the kit's script text says 10, its published output
says 1); with no false positives at all the curve's false-positive axis is all zeros;
`max_marks` (official runs call it with 100; the bundled sample was scored without) keeps only the
top-scoring candidates per scan. The seven false-positive rates are interpolated linearly
(np.interp), flat beyond the curve's ends.

    python modalities/ct/eval/froc.py score <submission.csv> [--boot 1000]
    python modalities/ct/eval/froc.py selfcheck
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from core import paths  # noqa: E402

LUNA_DIR = os.path.join(paths.DATASETS, 'public', 'luna16')
EVAL_DIR = os.path.join(LUNA_DIR, 'evaluationScript')
FP_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
_XYZ = ['coordX', 'coordY', 'coordZ']
# Diameter used for a finding with no size (diameter_mm < 0, i.e. most excluded findings). The script
# shipped in the kit says 10.0, but with 10 the sample submission loses 6555 candidates to excluded
# findings, not the 1294 its own CADAnalysis.txt reports; 1.0 reproduces 1294 exactly (and every other
# count), so the published results were made with a radius of 0.5 mm there. We follow the published output.
UNSIZED_DIAMETER_MM = 1.0


def load_reference():
    """The official annotations, excluded findings and scan list."""
    d = os.path.join(EVAL_DIR, 'annotations')
    ann = pd.read_csv(os.path.join(d, 'annotations.csv'))
    exc = pd.read_csv(os.path.join(d, 'annotations_excluded.csv'))
    uids = pd.read_csv(os.path.join(d, 'seriesuids.csv'), header=None)[0].astype(str).tolist()
    return ann, exc, uids


def _radius_sq(df):
    d = df['diameter_mm'].to_numpy(dtype=np.float64) if 'diameter_mm' in df else np.full(len(df), -1.0)
    d = np.where(d < 0.0, UNSIZED_DIAMETER_MM, d)
    return (d / 2.0) ** 2


def _by_scan(df, index):
    """Rows of df grouped by scan index: (xyz, extra, bounds) with rows sorted by scan, file order kept."""
    sid = df['seriesuid'].astype(str).map(index)
    keep = sid.notna().to_numpy()
    sid = sid.to_numpy()[keep].astype(np.int64)
    order = np.argsort(sid, kind='stable')
    rows = np.flatnonzero(keep)[order]
    bounds = np.searchsorted(sid[order], np.arange(len(index) + 1))
    return rows, bounds


def _match(candidates, annotations, excluded, seriesuids, max_marks=None):
    """The official per-scan matching. Returns the pieces both the curve and the counts are built from."""
    uids = list(dict.fromkeys(str(u) for u in seriesuids))
    index = {u: i for i, u in enumerate(uids)}
    n = len(uids)
    cxyz = candidates[_XYZ].to_numpy(dtype=np.float64)
    cprob = candidates['probability'].to_numpy(dtype=np.float64)
    arows, abounds = _by_scan(annotations, index)
    axyz = annotations[_XYZ].to_numpy(dtype=np.float64)[arows]
    ar2 = _radius_sq(annotations)[arows]
    erows, ebounds = _by_scan(excluded, index)
    exyz = excluded[_XYZ].to_numpy(dtype=np.float64)[erows]
    er2 = _radius_sq(excluded)[erows]
    crows, cbounds = _by_scan(candidates, index)

    tp_prob, tp_scan, fp_prob, fp_scan = [], [], [], []
    lesions = np.zeros(n, dtype=np.int64)
    n_cand = ignored = double = 0
    for s in range(n):
        rows = crows[cbounds[s]:cbounds[s + 1]]
        if max_marks and max_marks > 0 and len(rows) > max_marks:
            p = cprob[rows]
            thr = np.sort(p)[::-1][max_marks]
            rows = rows[p > thr][:max_marks]
        n_cand += len(rows)
        a0, a1 = abounds[s], abounds[s + 1]
        lesions[s] = a1 - a0
        m = len(rows)
        if m == 0:
            continue
        xyz, prob = cxyz[rows], cprob[rows]
        claimed = np.zeros(m, dtype=bool)
        if a1 > a0:
            d2 = ((xyz[:, None, :] - axyz[None, a0:a1, :]) ** 2).sum(axis=2)
            hit = d2 < ar2[None, a0:a1]
            found = hit.any(axis=0)
            double += int(np.maximum(hit.sum(axis=0) - 1, 0).sum())
            if found.any():
                tp_prob.append(np.where(hit[:, found], prob[:, None], -np.inf).max(axis=0))
                tp_scan.append(np.full(int(found.sum()), s))
            claimed = hit.any(axis=1)
        e0, e1 = ebounds[s], ebounds[s + 1]
        on_excluded = np.zeros(m, dtype=bool)
        if e1 > e0:
            d2 = ((xyz[:, None, :] - exyz[None, e0:e1, :]) ** 2).sum(axis=2)
            on_excluded = (d2 < er2[None, e0:e1]).any(axis=1) & ~claimed
            ignored += int(on_excluded.sum())
        fp = ~claimed & ~on_excluded
        fp_prob.append(prob[fp])
        fp_scan.append(np.full(int(fp.sum()), s))
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    return dict(
        uids=uids, lesions=lesions, n_cand=n_cand, ignored=ignored, double=double,
        tp_prob=cat(tp_prob, np.float64), tp_scan=cat(tp_scan, np.int64),
        fp_prob=cat(fp_prob, np.float64), fp_scan=cat(fp_scan, np.int64))


def _curve_inputs(m):
    """Entries of the curve (hits and false alarms; misses are dropped as in the official code), sorted by score."""
    prob = np.concatenate([m['tp_prob'], m['fp_prob']])
    gt = np.concatenate([np.ones(len(m['tp_prob'])), np.zeros(len(m['fp_prob']))])
    scan = np.concatenate([m['tp_scan'], m['fp_scan']])
    order = np.argsort(-prob, kind='stable')
    prob, gt, scan = prob[order], gt[order], scan[order]
    last = np.ones(len(prob), dtype=bool)
    last[:-1] = prob[1:] != prob[:-1]  # the end of each run of equal scores: one curve point per threshold
    return prob, gt, scan, np.flatnonzero(last)


def _curve(gt, w, last, total_lesions, n_images):
    """(false positives per scan, sensitivity) at every threshold, as the official computeFROC."""
    tps = np.concatenate([[0.0], np.cumsum(w * gt)[last]])
    fps = np.concatenate([[0.0], np.cumsum(w * (1.0 - gt))[last]])
    sens = tps / total_lesions if total_lesions > 0 else np.zeros(len(tps))
    if fps[-1] == 0:  # official: no false positives, the ROC is undefined, its false-positive axis is zeros
        return np.zeros(len(fps)), sens
    return fps / n_images, sens


def _sens_at(fps, sens, rates=FP_RATES):
    return np.interp(np.asarray(rates, dtype=np.float64), fps, sens)


def _defaults(annotations, excluded, seriesuids):
    if annotations is None or excluded is None or seriesuids is None:
        a, e, u = load_reference()
        annotations = a if annotations is None else annotations
        excluded = e if excluded is None else excluded
        seriesuids = u if seriesuids is None else seriesuids
    return annotations, excluded, seriesuids


def score(candidates, annotations=None, excluded=None, seriesuids=None, boot=0, seed=0, max_marks=None):
    """Score candidates (seriesuid, coordX, coordY, coordZ, probability) the way LUNA16 does.

    annotations / excluded: frames with seriesuid, coordX/Y/Z and diameter_mm (default: the official
    files). seriesuids: the scans being scored (default: the official 888); candidates on other scans
    are ignored. boot > 0 adds 95% intervals by resampling scans. max_marks: keep only that many
    top-scoring candidates per scan (the official runs use 100; default no cap).
    Returns a dict: cpm, sens_at (seven values at FP_RATES), tp, fp, fn, n_candidates,
    ignored_excluded, ignored_double, n_scans, n_nodules, fps/sens (the curve), and with boot > 0
    boot (dict: cpm, sens_at as (mean, lower, upper)).
    """
    annotations, excluded, seriesuids = _defaults(annotations, excluded, seriesuids)
    m = _match(candidates, annotations, excluded, seriesuids, max_marks)
    n = len(m['uids'])
    total = int(m['lesions'].sum())
    prob, gt, scan, last = _curve_inputs(m)
    fps, sens = _curve(gt, np.ones(len(gt)), last, total, n)
    at = _sens_at(fps, sens)
    tp = len(m['tp_prob'])
    res = dict(cpm=float(at.mean()), sens_at=[float(v) for v in at], tp=tp, fp=len(m['fp_prob']),
               fn=total - tp, n_candidates=m['n_cand'], ignored_excluded=m['ignored'],
               ignored_double=m['double'], n_scans=n, n_nodules=total, fps=fps, sens=sens)
    if boot and boot > 0:
        rng = np.random.default_rng(seed)
        sa = np.zeros((boot, len(FP_RATES)))
        for b in range(boot):
            counts = np.bincount(rng.integers(n, size=n), minlength=n).astype(np.float64)
            f, s = _curve(gt, counts[scan], last, float(counts @ m['lesions']), n)
            sa[b] = _sens_at(f, s)
        res['boot'] = _summarise(sa, boot)
    return res


def _summarise(sa, boot):
    """Official compute_mean_ci: mean, and the sorted values at floor(0.025 B) and floor(0.975 B)."""
    lo, hi = math.floor(0.025 * boot), min(math.floor(0.975 * boot), boot - 1)
    cpm = np.sort(sa.mean(axis=1))
    srt = np.sort(sa, axis=0)
    return dict(cpm=(float(cpm.mean()), float(cpm[lo]), float(cpm[hi])),
                sens_at=[(float(srt[:, i].mean()), float(srt[lo, i]), float(srt[hi, i])) for i in range(sa.shape[1])])


def operating_point(candidates, threshold, annotations=None, excluded=None, seriesuids=None, max_marks=None):
    """Sensitivity and false positives per scan of the candidates scoring at or above threshold."""
    annotations, excluded, seriesuids = _defaults(annotations, excluded, seriesuids)
    kept = candidates[candidates['probability'].to_numpy(dtype=np.float64) >= threshold]
    m = _match(kept, annotations, excluded, seriesuids, max_marks)
    total, n = int(m['lesions'].sum()), len(m['uids'])
    tp, fp = len(m['tp_prob']), len(m['fp_prob'])
    return dict(sensitivity=tp / total if total else 0.0, fp_per_scan=fp / n if n else 0.0,
                tp=tp, fp=fp, n_candidates=m['n_cand'], threshold=threshold)


def sensitivity_at_fp(result, fp_per_scan):
    """Sensitivity of a score() result at the given false positives per scan (interpolated on the curve)."""
    return float(np.interp(fp_per_scan, result['fps'], result['sens']))


def _fmt(res):
    lines = ['CPM %.4f' % res['cpm']]
    boot = res.get('boot')
    for i, r in enumerate(FP_RATES):
        ci = '   95%% CI %.4f-%.4f' % boot['sens_at'][i][1:] if boot else ''
        lines.append('  sensitivity at %-5g FP/scan: %.4f%s' % (r, res['sens_at'][i], ci))
    if boot:
        lines.insert(1, '  CPM 95%% CI %.4f-%.4f (bootstrap mean %.4f)' % (boot['cpm'][1], boot['cpm'][2], boot['cpm'][0]))
    lines.append('TP %d  FP %d  FN %d  nodules %d  candidates %d  ignored on excluded %d  double detections %d  scans %d' % (
        res['tp'], res['fp'], res['fn'], res['n_nodules'], res['n_candidates'], res['ignored_excluded'],
        res['ignored_double'], res['n_scans']))
    return '\n'.join(lines)


def selfcheck():
    ex = os.path.join(EVAL_DIR, 'exampleFiles')
    sub = pd.read_csv(os.path.join(ex, 'submission', 'sampleSubmission.csv'))
    t0 = time.time()
    res = score(sub)
    secs = time.time() - t0
    print('sample submission (%d candidates) scored in %.2f s' % (len(sub), secs))
    print(_fmt(res))
    bad = []
    for key, want in [('tp', 1120), ('fp', 548420), ('fn', 66), ('ignored_excluded', 1294),
                      ('ignored_double', 231), ('n_candidates', 551065)]:
        if res[key] != want:
            bad.append('%s: got %s, official %s' % (key, res[key], want))
    ref = os.path.join(ex, 'evaluation', 'froc_sampleSubmission.txt')
    cur = np.loadtxt(ref, delimiter=',')
    want = np.interp(FP_RATES, cur[:, 0], cur[:, 1])
    for r, g, w in zip(FP_RATES, res['sens_at'], want):
        if abs(g - w) > 0.001:
            bad.append('sensitivity at %g FP/scan: got %.5f, official curve %.5f' % (r, g, w))
    ann, exc, uids = load_reference()
    perfect = ann[['seriesuid'] + _XYZ].assign(probability=1.0)
    p = score(perfect)
    print('perfect submission: CPM %.4f, TP %d, FP %d' % (p['cpm'], p['tp'], p['fp']))
    if p['fp'] != 0 or p['tp'] != len(ann) or any(abs(v - 1.0) > 1e-9 for v in p['sens_at']):
        bad.append('perfect submission: TP %d FP %d sens %s' % (p['tp'], p['fp'], p['sens_at']))
    op = operating_point(sub, -np.inf)
    if op['tp'] != 1120 or op['fp'] != 548420:
        bad.append('operating_point at -inf: tp %d fp %d' % (op['tp'], op['fp']))
    if bad:
        print('selfcheck FAILED:\n  ' + '\n  '.join(bad))
        return 1
    print('selfcheck ok')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('score', help='score a LUNA16 submission csv against the official reference')
    s.add_argument('submission')
    s.add_argument('--boot', type=int, default=0)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--max-marks', type=int, default=None)
    sp.add_parser('selfcheck', help='reproduce the official output for the sample submission')
    a = ap.parse_args(argv)
    if a.cmd == 'selfcheck':
        return selfcheck()
    print(_fmt(score(pd.read_csv(a.submission), boot=a.boot, seed=a.seed, max_marks=a.max_marks)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
