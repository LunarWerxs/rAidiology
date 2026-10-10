"""Tests for the A/B judges. Every "kept" or "not kept" in this repository is what one of them printed.

Why. A trial's verdict is whatever these scripts print, and nothing checked them. A judge that paired
rows by position instead of by case, took A minus B instead of B minus A, resampled the two sides
apart, or read its interval against the wrong threshold would flip kept and not kept without a sound.
So each judge is run here through its real command line, on small made-up inputs whose answer was
worked out by hand beforehand, and the test reads what it prints:

- A against an identical B: a difference of exactly 0 with an interval of exactly [0, 0] (only a
  paired resample gives that), and the verdict the judge's own rule gives for it.
- A against a B that is better by a known amount: the printed values equal the hand computation,
  and the verdict is the better one.
- The two swapped: the same difference and interval with the sign flipped, and the worse verdict.
- B's rows shuffled (and A's): every printed number the same as before.
- The lung nodule FROC judge: a few scans with candidates inside and outside the hit radius, a double
  detection and a candidate on an excluded finding, with the CPM worked out by hand.

Nothing here reads real data or writes the real ledger. Each judge reads files written into a scratch
folder in the system's temporary folder (on this PC another drive than the repository's, so the judges
that record their inputs as paths relative to the repository meet two drives, as they did not until
2026-10-10: os.path.relpath raised after the verdict was printed and nothing reached the ledger), and
RAIDIOLOGY_LEDGER, and where a judge reads them RAIDIOLOGY_WORK and RAIDIOLOGY_DATASETS, point into it.

    python tools/test_judges.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(os.path.dirname(ROOT), "_tools", "venv", "Scripts", "python.exe")
PYTHON = VENV if os.path.exists(VENV) else sys.executable
SCRATCH = tempfile.gettempdir()
N = r"([+-]?\d+\.\d+)"          # one printed number
EXACT = 6e-5                     # a number printed to 4 decimals is within half of the last one
MIRROR = 1.1e-4                  # two numbers printed to 4 decimals: one unit of the last apart at most


# The four-case block most of these tests are made of: two positives and two negatives, scored
#   label   1  1  0  0
#   A       4  2  3  1   AUROC by hand: of the 4 (positive, negative) pairs, 4>3, 4>1 and 2>1 win
#                        and 2<3 loses, so 3/4. More whole blocks only add pairs of the same four
#                        kinds in the same proportions, so any number of blocks gives 0.75 too.
#   B       4  3  2  1   every positive above every negative: 1.0
# so B minus A is +0.25. The block's patients are split across it: patient X holds the 4 and the 3,
# patient Y the 2 and the 1. A resample that draws x X's and y Y's gives A an AUROC of
# (x*x + x*y + y*y) / (x + y)^2 (from 0.75 to 1) while B stays at 1: the paired difference moves
# from draw to draw but is never below 0, and is 0 only when one kind of patient is drawn alone.
def blocks(n):
    """label, A's score, B's score and patient of n blocks (4 n cases, 2 n patients)."""
    y = np.tile([1, 1, 0, 0], n)
    a = np.tile([4.0, 2.0, 3.0, 1.0], n)
    b = np.tile([4.0, 3.0, 2.0, 1.0], n)
    patient = np.repeat(np.arange(n) * 2, 4) + np.tile([0, 1, 0, 1], n)
    return y, a, b, patient


def shuffled(n, seed):
    return np.random.default_rng(seed).permutation(n)


def judge(args, tmp, **env):
    """Run one judge from the repository root; its printed lines, without the ledger's timing line."""
    full = dict(os.environ, RAIDIOLOGY_LEDGER=os.path.join(tmp, "ledger"), PYTHONIOENCODING="utf-8")
    full.pop("NIH_CXR_ROOT", None)          # the chest X-ray index must come from the scratch datasets
    full.update(env)
    p = subprocess.run([PYTHON] + list(args), cwd=ROOT, env=full, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    if p.returncode:
        raise AssertionError("%s exited %d\n--- stdout\n%s\n--- stderr\n%s"
                             % (" ".join(args), p.returncode, p.stdout, p.stderr))
    return "\n".join(line for line in p.stdout.splitlines() if not line.startswith("ledger "))


def numbers(pattern, text):
    m = re.search(pattern, text, re.M)
    if not m:
        raise AssertionError("no line matching %r in:\n%s" % (pattern, text))
    return [float(g) for g in m.groups()]


def word(pattern, text):
    m = re.search(pattern, text, re.M)
    if not m:
        raise AssertionError("no line matching %r in:\n%s" % (pattern, text))
    return m.group(1).strip()


def last_line(text):
    return text.strip().splitlines()[-1].strip()


class JudgeCase(unittest.TestCase):
    """A scratch folder per judge, made before its runs and removed after them."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(SCRATCH, exist_ok=True)
        cls._scratch = tempfile.TemporaryDirectory(prefix="test_judges_", dir=SCRATCH, ignore_cleanup_errors=True)
        cls.tmp = cls._scratch.name
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls._scratch.cleanup()

    def path(self, *p):
        return os.path.join(self.tmp, *p)

    def assertClose(self, got, want, what=""):
        self.assertAlmostEqual(got, want, delta=EXACT, msg=what)

    def assertZero(self, d, lo, hi, what=""):
        """Identical sides: exactly 0 in every resample, so the interval is [0, 0], not merely near it."""
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0), what)

    def assertMirrored(self, swapped, better, what=""):
        """(B - A, low, high) with A and B swapped: the same numbers with the sign flipped."""
        d, lo, hi = swapped
        d0, lo0, hi0 = better
        self.assertAlmostEqual(d, -d0, delta=MIRROR, msg=what)
        self.assertAlmostEqual(lo, -hi0, delta=MIRROR, msg=what)
        self.assertAlmostEqual(hi, -lo0, delta=MIRROR, msg=what)


class ChestXrayCompare(JudgeCase):
    """modalities/chest_xray/eval/compare.py: AUROC per label, paired by patient; the verdict is on the
    mean of the 14 findings, with the margin rule (BETTER / NOT WORSE / WORSE / INCONCLUSIVE)."""

    ROW = rf"^\| (.+?) \| {N} \| {N} \| {N} \({N} to {N}\) \|$"

    @classmethod
    def build(cls):
        work = os.path.join(cls.tmp, "work")

        def reader(version, image, patient, y, score):
            d = os.path.join(work, "eval", "chest", "reader-v%d" % version)
            os.makedirs(d)
            np.savez(os.path.join(d, "oof_logits.npz"), image=np.asarray(image), patient=np.asarray(patient),
                     y=y, logit=score)
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:   # read when both are reader runs
                json.dump({"folds": [{"epoch_seconds": [10.0, 12.0]}]}, f)

        y, a, b, patient = blocks(300)        # 1,200 pictures: the judge stops under 1,000 shared
        image = np.array(["img%05d.png" % i for i in range(len(y))])

        def cols(v):
            """The same column for all 15 labels."""
            return np.repeat(v[:, None], 15, axis=1)
        reader(1, image, patient, cols(y), cols(a))
        reader(2, image, patient, cols(y), cols(b))
        pa, pb = shuffled(len(y), 1), shuffled(len(y), 2)
        reader(3, image[pb], patient[pb], cols(y)[pb], cols(b)[pb])
        reader(4, image[pa], patient[pa], cols(y)[pa], cols(a)[pa])
        rng = np.random.default_rng(3)
        ry, rs = (rng.random((len(y), 15)) < 0.3).astype(np.int8), rng.normal(size=(len(y), 15))
        reader(5, image, np.arange(len(y)) // 3, ry, rs)
        reader(6, image, np.arange(len(y)) // 3, ry, rs)

        def run(x, z):
            return judge(["modalities/chest_xray/eval/compare.py", x, z, "--boot", "200", "--no-ledger"],
                         cls.tmp, RAIDIOLOGY_WORK=work)
        cls.better, cls.swapped, cls.shuffled, cls.same = run("1", "2"), run("2", "1"), run("4", "3"), run("5", "6")

    def rows(self, text):
        found = re.findall(self.ROW, text, re.M)
        self.assertEqual(len(found), 16, text)      # 14 findings, their mean, Abnormal
        return {r[0]: [float(v) for v in r[1:]] for r in found}

    def verdict(self, text):
        return word(r"^Verdict \(mean of 14 findings, margin [\d.]+\): (.+)$", text)

    def test_identical(self):
        for name, (va, vb, d, lo, hi) in self.rows(self.same).items():
            self.assertEqual(va, vb, name)
            self.assertZero(d, lo, hi, name)
        self.assertEqual(self.verdict(self.same), "NOT WORSE")    # [0, 0]: the low end is above minus the margin

    def test_better(self):
        for name, (va, vb, d, lo, hi) in self.rows(self.better).items():
            self.assertClose(va, 0.75, name)
            self.assertClose(vb, 1.0, name)
            self.assertClose(d, 0.25, name)
            self.assertGreater(lo, 0, name)
            self.assertLessEqual(hi, 0.25 + EXACT, name)    # A is never below 0.75 in a resample
        self.assertEqual(self.verdict(self.better), "BETTER")

    def test_swapped(self):
        better, swapped = self.rows(self.better), self.rows(self.swapped)
        for name, (va, vb, d, lo, hi) in swapped.items():
            self.assertClose(va, 1.0, name)
            self.assertClose(vb, 0.75, name)
            self.assertMirrored((d, lo, hi), better[name][2:], name)
        self.assertEqual(self.verdict(self.swapped), "WORSE")

    def test_shuffled(self):
        self.assertEqual(self.rows(self.shuffled), self.rows(self.better))
        self.assertEqual(self.verdict(self.shuffled), self.verdict(self.better))


class CtCompare(JudgeCase):
    """modalities/ct/eval/compare.py: mean Dice of segmentation rows, paired by scan. Its rule has no
    margin: KEPT when the interval is above 0 (and beyond --noise), DROPPED when below, else
    INCONCLUSIVE."""

    @classmethod
    def build(cls):
        scans = ["s%02d" % i for i in range(1, 13)]     # 12: the judge wants 10 in common, a structure 10 scans

        def rows(liver, spleen):
            # Dice = 2 both / (truth + pred) = both / 100 here
            return pd.DataFrame({"scan": scans * 2, "structure": ["liver"] * 12 + ["spleen"] * 12, "split": "dev",
                                 "in_scope": 1, "truth_voxels": 100, "pred_voxels": 100,
                                 "both_voxels": list(liver) + list(spleen)})

        # By hand: A's liver is 0.5 on every other scan and 1 on the rest, its spleen 1: per structure
        # 0.75 and 1, mean Dice 0.875. B is 1 everywhere, so B minus A is +0.125. A resample cannot lift
        # A above B (B is 1 on any draw), and leaves them level only when no 0.5 scan is drawn.
        a = rows([50 if i % 2 else 100 for i in range(12)], [100] * 12)
        b = rows([100] * 12, [100] * 12)
        rng = np.random.default_rng(4)
        r = rows(rng.integers(0, 101, 12), rng.integers(0, 101, 12))
        for name, df in (("a", a), ("b", b), ("a_shuffled", a.sample(frac=1, random_state=1)),
                         ("b_shuffled", b.sample(frac=1, random_state=2)), ("r1", r), ("r2", r)):
            df.to_csv(os.path.join(cls.tmp, name + ".csv"), index=False)

        def run(x, z):
            return judge(["modalities/ct/eval/compare.py", "test_judges", "--a", os.path.join(cls.tmp, x + ".csv"),
                          "--b", os.path.join(cls.tmp, z + ".csv"), "--resamples", "200", "--dry"], cls.tmp)
        cls.better, cls.swapped = run("a", "b"), run("b", "a")
        cls.shuffled, cls.same = run("a_shuffled", "b_shuffled"), run("r1", "r2")

    def diff(self, text):
        return numbers(rf"B minus A\s+{N} \(95% CI {N} to {N}\)", text)

    def dice(self, side, text):
        return numbers(rf"mean Dice {side} {N} \(", text)[0]

    def test_identical(self):
        self.assertZero(*self.diff(self.same))
        self.assertEqual(self.dice("A", self.same), self.dice("B", self.same))
        self.assertEqual(last_line(self.same), "INCONCLUSIVE")    # a difference of 0 is not beyond a noise of 0

    def test_better(self):
        self.assertClose(self.dice("A", self.better), 0.875)
        self.assertClose(self.dice("B", self.better), 1.0)
        d, lo, hi = self.diff(self.better)
        self.assertClose(d, 0.125)
        self.assertGreater(lo, 0)
        self.assertEqual(last_line(self.better), "KEPT")

    def test_swapped(self):
        self.assertMirrored(self.diff(self.swapped), self.diff(self.better))
        self.assertEqual(last_line(self.swapped), "DROPPED")

    def test_shuffled(self):
        self.assertEqual(self.shuffled, self.better)


class SpineMriCompare(JudgeCase):
    """modalities/spine_mri/eval/compare.py: exact grade, within one grade and quadratic weighted kappa
    of disc grades, paired by patient (score.py's 2,000 draws, fixed). KEPT when all three are up,
    DROPPED when any is down, else INCONCLUSIVE."""

    @classmethod
    def build(cls):
        n_pat = 20
        patient = np.repeat(np.arange(1, n_pat + 1), 5)
        ivd = np.tile([1, 2, 3, 4, 5], n_pat)
        truth = np.tile([1, 2, 3, 4, 5], n_pat)
        # Odd patients' A grades are off on two discs: truth 1 read as 3 (two grades out) and truth 5
        # read as 4 (one out); even patients' A grades are right. B is right everywhere.
        off = np.where(np.repeat(np.arange(1, n_pat + 1) % 2 == 1, 5), np.tile([3, 2, 3, 4, 4], n_pat), truth)

        def disc_file(name, grade, truth=truth, order=None):
            order = np.arange(len(grade)) if order is None else order
            y, z = np.zeros((len(grade), 11)), np.zeros((len(grade), 11))     # 4 grade steps, then 7 items
            y[:, :4] = np.arange(4)[None, :] < (truth - 1)[:, None]             # grade g: the first g - 1 steps
            z[:, :4] = np.where(np.arange(4)[None, :] < (grade - 1)[:, None], 1.0, -1.0)
            np.savez(os.path.join(cls.tmp, name + ".npz"), nested=z[order], y=y[order],
                     patient=patient[order], ivd=ivd[order])

        disc_file("a", off)
        disc_file("b", truth)
        disc_file("a_shuffled", off, order=shuffled(len(truth), 1))
        disc_file("b_shuffled", truth, order=shuffled(len(truth), 2))
        rng = np.random.default_rng(5)
        rt, rg = rng.integers(1, 6, len(truth)), rng.integers(1, 6, len(truth))
        disc_file("r1", rg, truth=rt)
        disc_file("r2", rg, truth=rt)

        def run(x, z):
            return judge(["modalities/spine_mri/eval/compare.py", "test_judges", "--a", os.path.join(cls.tmp, x + ".npz"),
                          "--b", os.path.join(cls.tmp, z + ".npz")], cls.tmp)
        cls.better, cls.swapped = run("a", "b"), run("b", "a")
        cls.shuffled, cls.same = run("a_shuffled", "b_shuffled"), run("r1", "r2")

    def metric(self, name, text):
        """A, B, B - A, low, high of one metric."""
        return numbers(rf"^\s+{name}\s+A {N}\s+B {N}\s+B minus A {N} \(95% CI {N} to {N}\)", text)

    def test_identical(self):
        for name in ("exact", "within_one", "qwk"):
            va, vb, d, lo, hi = self.metric(name, self.same)
            self.assertEqual(va, vb, name)
            self.assertZero(d, lo, hi, name)
        self.assertEqual(last_line(self.same), "INCONCLUSIVE")    # every metric level

    def test_better(self):
        # By hand, over one odd and one even patient (10 discs; every pair of patients alike):
        #   exact      A 8/10 = 0.8, B 1, B - A +0.2
        #   within one A 9/10 = 0.9, B 1, B - A +0.1
        #   kappa      A: the confusion matrix is the identity plus (1->3) and (5->4). Rows sum to 2,
        #              columns to 1, 2, 3, 3, 1. Weights (i - j)^2 / 16. Observed: (4 + 1) / 16 = 5/16.
        #              Expected: sum over columns of (column / 5) * sum_i (i - j)^2 / 16
        #              = (1*30 + 2*15 + 3*10 + 3*15 + 1*30) / 80 = 165/80. Kappa 1 - 25/165 = 28/33.
        #              B 1, B - A +5/33.
        for name, a, d in (("exact", 0.8, 0.2), ("within_one", 0.9, 0.1), ("qwk", 28 / 33, 5 / 33)):
            va, vb, dd, lo, hi = self.metric(name, self.better)
            self.assertClose(va, a, name)
            self.assertClose(vb, 1.0, name)
            self.assertClose(dd, d, name)
            self.assertGreater(lo, 0, name)
        self.assertEqual(last_line(self.better), "KEPT")

    def test_swapped(self):
        for name in ("exact", "within_one", "qwk"):
            self.assertMirrored(self.metric(name, self.swapped)[2:], self.metric(name, self.better)[2:], name)
        self.assertEqual(last_line(self.swapped), "DROPPED")

    def test_shuffled(self):
        self.assertEqual(self.shuffled, self.better)


class XrayCompare(JudgeCase):
    """modalities/xray/eval/compare.py: AUROC per collection (fracatlas, grazpedwri), paired by patient
    through mirror.paired_auroc. KEPT when every collection is up, DROPPED when any is down."""
    # UNVERIFIED: the judge imports mirror.py, which imports torch and detector.py at module level
    # (modalities/xray/eval/mirror.py:24, :33); this assumes those imports read no data files.

    LINE = rf"^\s+(\w+), (\d+) pictures of (\d+) patients: AUROC A {N}, B {N}, B minus A {N} \(95% CI {N} to {N}\)"

    @classmethod
    def build(cls):
        y, a, b, patient = blocks(6)
        rng = np.random.default_rng(6)
        random = rng.normal(size=2 * len(y))

        def table(score):
            return pd.concat([pd.DataFrame({"dataset": d, "image": ["%s_%03d.png" % (d, i) for i in range(len(y))],
                                            "patient": ["%s_P%03d" % (d, p) for p in patient], "fold": 0,
                                            "truth": y, "score": score[k * len(y):(k + 1) * len(y)]})
                              for k, d in enumerate(("fracatlas", "grazpedwri"))], ignore_index=True)
        ta, tb, tr = table(np.tile(a, 2)), table(np.tile(b, 2)), table(random)
        for name, df in (("a", ta), ("b", tb), ("a_shuffled", ta.sample(frac=1, random_state=1)),
                         ("b_shuffled", tb.sample(frac=1, random_state=2)), ("r1", tr), ("r2", tr)):
            df.to_csv(os.path.join(cls.tmp, name + ".csv"), index=False)

        def run(x, z):
            return judge(["modalities/xray/eval/compare.py", "test_judges", "--a", os.path.join(cls.tmp, x + ".csv"),
                          "--b", os.path.join(cls.tmp, z + ".csv")], cls.tmp)
        cls.better, cls.swapped = run("a", "b"), run("b", "a")
        cls.shuffled, cls.same = run("a_shuffled", "b_shuffled"), run("r1", "r2")

    def collections(self, text):
        found = re.findall(self.LINE, text, re.M)
        self.assertEqual([f[0] for f in found], ["fracatlas", "grazpedwri"], text)
        return {f[0]: [float(v) for v in f[1:]] for f in found}

    def test_identical(self):
        for name, (n, n_p, va, vb, d, lo, hi) in self.collections(self.same).items():
            self.assertEqual(va, vb, name)
            self.assertZero(d, lo, hi, name)
        self.assertEqual(last_line(self.same), "INCONCLUSIVE")    # 0 is not beyond a noise of 0

    def test_better(self):
        for name, (n, n_p, va, vb, d, lo, hi) in self.collections(self.better).items():
            self.assertEqual((n, n_p), (24, 12), name)
            self.assertClose(va, 0.75, name)        # the block's hand computation
            self.assertClose(vb, 1.0, name)
            self.assertClose(d, 0.25, name)
            self.assertGreater(lo, 0, name)
        self.assertEqual(last_line(self.better), "KEPT")

    def test_swapped(self):
        better = self.collections(self.better)
        for name, row in self.collections(self.swapped).items():
            self.assertMirrored(row[4:], better[name][4:], name)
        self.assertEqual(last_line(self.swapped), "DROPPED")

    def test_shuffled(self):
        self.assertEqual(self.shuffled, self.better)


class FrocCompare(JudgeCase):
    """tools/froc_compare.py: CPM of two lung nodule candidate lists (modalities/ct/eval/froc.py's
    matching and curve), paired by scan, with the margin rule."""

    @classmethod
    def build(cls):
        ref = os.path.join(cls.tmp, "datasets", "public", "luna16", "evaluationScript", "annotations")
        os.makedirs(ref)
        # Five copies of three scans. Every scan has one nodule at the origin, 10 mm wide (a hit is
        # strictly closer than 5 mm); scan 3 also has an excluded finding of no size at (50, 50, 50),
        # which froc.py treats as 1 mm wide (strictly closer than 0.5 mm).
        uids = ["r%ds%d" % (r, s) for r in range(5) for s in (1, 2, 3)]
        pd.DataFrame({"seriesuid": uids, "coordX": 0.0, "coordY": 0.0, "coordZ": 0.0,
                      "diameter_mm": 10.0}).to_csv(os.path.join(ref, "annotations.csv"), index=False)
        pd.DataFrame({"seriesuid": ["r%ds3" % r for r in range(5)], "coordX": 50.0, "coordY": 50.0, "coordZ": 50.0,
                      "diameter_mm": -1.0}).to_csv(os.path.join(ref, "annotations_excluded.csv"), index=False)
        pd.Series(uids).to_csv(os.path.join(ref, "seriesuids.csv"), index=False, header=False)

        def candidates(name, per_copy, order_seed=None):
            df = pd.DataFrame([{"seriesuid": "r%d%s" % (r, s), "coordX": x, "coordY": y, "coordZ": z, "probability": p}
                               for r in range(5) for s, x, y, z, p in per_copy])
            if order_seed is not None:
                df = df.sample(frac=1, random_state=order_seed)
            df.to_csv(os.path.join(cls.tmp, name + ".csv"), index=False)

        # A, per copy: scan 1 a hit at 0.9; scan 2 a false positive at 0.95; scan 3 a false positive at 0.8.
        a = [("s1", 1, 1, 1, 0.9), ("s2", 30, 30, 30, 0.95), ("s3", 20, 20, 20, 0.8)]
        # B, per copy:
        #   scan 1: a hit at 0.9, and a second candidate on the same nodule at 0.2 (a double detection:
        #           the nodule counts once, at its highest score);
        #   scan 2: the candidate exactly 5 mm out, at 0.95, is not strictly inside: a false positive;
        #           the one 4.9 mm out, at 0.8, is inside: a hit;
        #   scan 3: the one 0.2 mm from the excluded finding, at 0.99, is dropped, neither hit nor
        #           false positive; the one far from both, at 0.3, is a false positive. Its nodule is missed.
        b = [("s1", 1, 1, 1, 0.9), ("s1", 2, 0, 0, 0.2), ("s2", 0, 0, 5, 0.95), ("s2", 0, 0, 4.9, 0.8),
             ("s3", 50, 50, 50.2, 0.99), ("s3", 20, 20, 20, 0.3)]
        candidates("a", a)
        candidates("b", b)
        candidates("b_copy", b)
        candidates("a_shuffled", a, order_seed=1)
        candidates("b_shuffled", b, order_seed=2)

        def run(x, z):
            return judge(["tools/froc_compare.py", os.path.join(cls.tmp, x + ".csv"), os.path.join(cls.tmp, z + ".csv"),
                          "--boot", "200", "--dry"], cls.tmp, RAIDIOLOGY_DATASETS=os.path.join(cls.tmp, "datasets"))
        cls.better, cls.swapped = run("a", "b"), run("b", "a")
        cls.shuffled, cls.same = run("a_shuffled", "b_shuffled"), run("b", "b_copy")

    def cpm(self, text):
        """A, B, B - A, low, high."""
        return numbers(rf"^CPM  A {N}  B {N}  B - A {N} \({N} to {N}\)", text)

    def verdict(self, text):
        return word(r"^Verdict \(CPM, margin [\d.]+\): (.+)$", text)

    def test_identical(self):
        ca, cb, d, lo, hi = self.cpm(self.same)
        self.assertEqual(ca, cb)
        self.assertZero(d, lo, hi)
        self.assertEqual(self.verdict(self.same), "NOT WORSE")

    def test_hand_computed_cpm(self):
        # 15 scans, 15 nodules. The curve steps at each distinct score, highest first (each step is
        # 5 entries, one per copy); false positives per scan = false positives / 15.
        #   B: 0.95 FP, 0.9 hit, 0.8 hit, 0.3 FP. Points (FP/scan, sensitivity):
        #      (0, 0) (1/3, 0) (1/3, 1/3) (1/3, 2/3) (2/3, 2/3).
        #      At 1/8 and 1/4: 0 (between (0, 0) and (1/3, 0)); at 1/2: 2/3; at 1, 2, 4, 8: past the
        #      end, 2/3. CPM = 5 * (2/3) / 7 = 10/21.
        #   A: 0.95 FP, 0.9 hit, 0.8 FP. Points (0, 0) (1/3, 0) (1/3, 1/3) (2/3, 1/3).
        #      At 1/8 and 1/4: 0; from 1/2 on: 1/3. CPM = 5 * (1/3) / 7 = 5/21.
        # Rules this catches: the 0.99 candidate on the excluded finding counted as a false positive
        # gives B 8/21; the 5 mm candidate counted as a hit (<= instead of <) gives 2/3; the double
        # detection's 0.2 taken instead of the nodule's best 0.9 gives 3/7.
        self.assertIn("15 scans, 15 nodules; 200 paired resamples of scans", self.better)
        ca, cb, d, lo, hi = self.cpm(self.better)
        self.assertClose(ca, 5 / 21)
        self.assertClose(cb, 10 / 21)
        self.assertClose(d, 5 / 21)
        # B is never below A on a resample (on a draw with c2 copies of scan 2, B's sensitivity from
        # there on is (c1 + c2) / L against A's c1 / L), and level only when no scan 2 is drawn.
        self.assertGreater(lo, 0)
        self.assertEqual(self.verdict(self.better), "BETTER")

    def test_swapped(self):
        self.assertMirrored(self.cpm(self.swapped)[2:], self.cpm(self.better)[2:])
        self.assertEqual(self.verdict(self.swapped), "WORSE")

    def test_shuffled(self):
        self.assertEqual(self.shuffled, self.better)


def lesions(score, order=None):
    """The blocks as lung lesions: one scan per patient, a distinct centre per lesion, all 8 mm."""
    y, a, b, patient = blocks(6)
    df = pd.DataFrame({"seriesuid": ["scan%02d" % p for p in patient], "coordX": np.arange(len(y)) * 10.0,
                       "coordY": 0.0, "coordZ": 0.0, "diameter_mm": 8.0, "label": y,
                       "logit": {"a": a, "b": b}[score] if isinstance(score, str) else score,
                       "patient": ["P%02d" % p for p in patient]})
    return df if order is None else df.iloc[order]


class MalignancyCompare(JudgeCase):
    """tools/malignancy_compare.py: AUROC of two versions' out-of-fold logits on the same lesions, paired
    by patient. Its WORSE needs the high end below minus the margin (malignancy_compare.py:135), not
    below 0 as in froc_compare."""

    @classmethod
    def build(cls):
        lung = os.path.join(cls.tmp, "work", "eval", "lung")
        random = np.random.default_rng(7).normal(size=24)
        for version, df in ((1, lesions("a")), (2, lesions("b")), (3, lesions("b", shuffled(24, 2))),
                            (4, lesions("a", shuffled(24, 1))), (5, lesions(random)), (6, lesions(random))):
            d = os.path.join(lung, "malignancy-v%d" % version)
            os.makedirs(d)
            df.drop(columns="patient").to_csv(os.path.join(d, "oof.csv"), index=False)
        lesions("a")[["seriesuid", "patient"]].drop_duplicates().to_csv(os.path.join(lung, "index.csv"), index=False)

        def run(x, z):
            return judge(["tools/malignancy_compare.py", x, z, "--boot", "200"], cls.tmp,
                         RAIDIOLOGY_WORK=os.path.join(cls.tmp, "work"))
        cls.better, cls.swapped, cls.shuffled, cls.same = run("1", "2"), run("2", "1"), run("4", "3"), run("5", "6")

    def auroc(self, text):
        return numbers(rf"^AUROC  A {N}  B {N}  B - A {N} \({N} to {N}\)", text)

    def verdict(self, text):
        return word(r"^Verdict \(AUROC, margin [\d.]+\): (.+)$", text)

    def test_identical(self):
        va, vb, d, lo, hi = self.auroc(self.same)
        self.assertEqual(va, vb)
        self.assertZero(d, lo, hi)
        self.assertEqual(self.verdict(self.same), "NOT WORSE")

    def test_better(self):
        self.assertIn("24 shared lesions of 12 patients (12 cancer-looking)", self.better)
        va, vb, d, lo, hi = self.auroc(self.better)
        self.assertClose(va, 0.75)                  # the block's hand computation
        self.assertClose(vb, 1.0)
        self.assertClose(d, 0.25)
        self.assertGreater(lo, 0)
        self.assertEqual(self.verdict(self.better), "BETTER")

    def test_swapped(self):
        self.assertMirrored(self.auroc(self.swapped)[2:], self.auroc(self.better)[2:])
        self.assertLess(self.auroc(self.swapped)[4], -0.01)     # below minus the margin, so its rule says WORSE
        self.assertEqual(self.verdict(self.swapped), "WORSE")

    def test_shuffled(self):
        self.assertEqual(self.shuffled, self.better)


class RoundtripMaligCompare(JudgeCase):
    """tools/roundtrip15.py malig-compare: AUROC of two lesion score files, paired by patient, with
    froc_compare's rule."""

    @classmethod
    def build(cls):
        random = np.random.default_rng(8).normal(size=24)
        for name, df in (("a", lesions("a")), ("b", lesions("b")), ("a_shuffled", lesions("a", shuffled(24, 1))),
                         ("b_shuffled", lesions("b", shuffled(24, 2))), ("r1", lesions(random)), ("r2", lesions(random))):
            df.to_csv(os.path.join(cls.tmp, name + ".csv"), index=False)

        def run(x, z):
            return judge(["tools/roundtrip15.py", "malig-compare", os.path.join(cls.tmp, x + ".csv"),
                          os.path.join(cls.tmp, z + ".csv"), "--boot", "200"], cls.tmp)
        cls.better, cls.swapped = run("a", "b"), run("b", "a")
        cls.shuffled, cls.same = run("a_shuffled", "b_shuffled"), run("r1", "r2")

    def auroc(self, text):
        return numbers(rf"^AUROC  A {N}  B {N}  B - A {N} \({N} to {N}\)", text)

    def verdict(self, text):
        return word(r"^Verdict \(AUROC, margin [\d.]+\): (.+)$", text)

    def test_identical(self):
        va, vb, d, lo, hi = self.auroc(self.same)
        self.assertEqual(va, vb)
        self.assertZero(d, lo, hi)
        self.assertEqual(self.verdict(self.same), "NOT WORSE")

    def test_better(self):
        self.assertIn("24 lesions of 12 patients", self.better)
        va, vb, d, lo, hi = self.auroc(self.better)
        self.assertClose(va, 0.75)                  # the block's hand computation
        self.assertClose(vb, 1.0)
        self.assertClose(d, 0.25)
        self.assertGreater(lo, 0)
        self.assertEqual(self.verdict(self.better), "BETTER")

    def test_swapped(self):
        self.assertMirrored(self.auroc(self.swapped)[2:], self.auroc(self.better)[2:])
        self.assertEqual(self.verdict(self.swapped), "WORSE")

    def test_shuffled(self):
        self.assertEqual(self.shuffled, self.better)


class ChestXDetEval(JudgeCase):
    """tools/chestxdet_eval.py with two versions: B - A of the mean AUROC over the 10 findings shared
    with ChestX-Det, against the radiologists' labels, paired by patient (1,000 draws, fixed). Its rule
    has no margin: BETTER above 0, WORSE below, else INCONCLUSIVE."""

    CXD = ["Atelectasis", "Cardiomegaly", "Consolidation", "Effusion", "Emphysema", "Fibrosis", "Mass",
           "Nodule", "Pleural Thickening", "Pneumothorax"]
    NIH = ["Atelectasis", "Cardiomegaly", "Consolidation", "Effusion", "Emphysema", "Fibrosis", "Mass",
           "Nodule", "Pleural_Thickening", "Pneumothorax"]

    @classmethod
    def build(cls):
        y, a, b, patient = blocks(6)
        n = len(y)
        image = np.array(["nih_%03d.png" % i for i in range(n)])
        work = os.path.join(cls.tmp, "work")

        def reader(version, score, order=np.arange(n)):
            d = os.path.join(work, "eval", "chest", "reader-v%d" % version)
            os.makedirs(d)
            # every one of the 15 label columns gets the same score, so their order does not matter
            np.savez(os.path.join(d, "oof_logits.npz"), image=image[order],
                     logit=np.repeat(np.asarray(score, float)[order][:, None], 15, axis=1))

        def datasets(root, order):
            """ChestX-Det's match list and boxes (a positive picture holds all 10 findings, a negative
            none), and the NIH index files datasets.index() reads, rows in `order`."""
            cxd = os.path.join(root, "public", "chestx-det")
            nih = os.path.join(root, "public", "nih-chestxray14")
            os.makedirs(cxd)
            os.makedirs(nih)
            files = ["cxd_%03d.png" % i for i in range(n)]
            pd.DataFrame({"image": image, "file_name": files, "patient": patient + 1}).iloc[order].to_csv(
                os.path.join(cxd, "nih_match.csv"), index=False)
            boxes = [{"file_name": files[i], "syms": cls.CXD if y[i] else []} for i in order]
            for part, half in (("train", boxes[:n // 2]), ("test", boxes[n // 2:])):
                with open(os.path.join(cxd, "ChestX_Det_%s.json" % part), "w", encoding="utf-8") as f:
                    json.dump(half, f)
            pd.DataFrame({"Image Index": image, "Finding Labels": ["|".join(cls.NIH) if v else "No Finding" for v in y],
                          "Follow-up #": 0, "Patient ID": patient + 1, "Patient Age": 50, "Patient Sex": "M",
                          "View Position": "PA"}).iloc[order].to_csv(os.path.join(nih, "Data_Entry_2017_v2020.csv"),
                                                                     index=False)
            with open(os.path.join(nih, "train_val_list.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(image) + "\n")
            open(os.path.join(nih, "test_list.txt"), "w", encoding="utf-8").close()

        reader(1, a)
        reader(2, b)
        reader(3, b, shuffled(n, 2))
        reader(4, a, shuffled(n, 1))
        random = np.random.default_rng(9).normal(size=n)
        reader(5, random)
        reader(6, random)
        datasets(os.path.join(cls.tmp, "datasets"), np.arange(n))
        datasets(os.path.join(cls.tmp, "datasets_shuffled"), shuffled(n, 3))

        def run(x, z, data="datasets"):
            return judge(["tools/chestxdet_eval.py", x, z], cls.tmp, RAIDIOLOGY_WORK=work,
                         RAIDIOLOGY_DATASETS=os.path.join(cls.tmp, data))
        cls.better, cls.swapped, cls.same = run("1", "2"), run("2", "1"), run("5", "6")
        cls.shuffled = run("4", "3", "datasets_shuffled")

    def paired(self, text):
        return numbers(rf"radiologists: B - A {N} \({N} to {N}\)", text)

    def verdict(self, text):
        return word(r"^Verdict \(radiologists' labels\): (\w+)$", text)

    def test_identical(self):
        self.assertZero(*self.paired(self.same))
        self.assertEqual(self.verdict(self.same), "INCONCLUSIVE")     # [0, 0] is neither above nor below 0

    def test_better(self):
        self.assertIn("v2 against v1 on 24 pictures of 12 patients", self.better)
        # each version's own report: the mean AUROC against the radiologists, 3 decimals
        means = [float(v) for v in re.findall(rf"^mean over the 10 findings: {N} against the radiologists",
                                              self.better, re.M)]
        self.assertEqual(len(means), 2, self.better)
        self.assertAlmostEqual(means[0], 0.75, delta=6e-4)    # the block's hand computation
        self.assertAlmostEqual(means[1], 1.0, delta=6e-4)
        d, lo, hi = self.paired(self.better)
        self.assertClose(d, 0.25)
        self.assertGreater(lo, 0)
        self.assertEqual(self.verdict(self.better), "BETTER")

    def test_swapped(self):
        self.assertMirrored(self.paired(self.swapped), self.paired(self.better))
        self.assertEqual(self.verdict(self.swapped), "WORSE")

    def test_shuffled(self):
        # B's and A's rows, the match list and the box files all in another order
        self.assertEqual(self.paired(self.shuffled), self.paired(self.better))
        self.assertEqual(self.verdict(self.shuffled), self.verdict(self.better))


if __name__ == "__main__":
    unittest.main(verbosity=2)
