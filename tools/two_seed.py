"""The two-seed bar, computed from two compare.py outputs instead of by hand.

    python tools/two_seed.py <seed 0 compare output> <seed 1 compare output> [--noise name=value ...]

Each file is what modalities/{xray,spine_mri}/eval/compare.py printed (or tee'd). Per measure: the
two seeds' B minus A, their mean, the bar (mean no worse than minus the seed noise over the square
root of two) and a proven gain (both seeds above the noise). The noise is the one compare.py printed
beside the measure; --noise sets or overrides it. Kept when neither file says DROPPED and every
measure with a noise passes; a measure with no noise is reported, not judged. Exit 0 kept, 1 not.
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import ledger  # noqa: E402

SPINE = re.compile(r"^\s*(\S+)\s+A [\d.]+\s+B [\d.]+\s+B minus A ([+-][\d.]+) \(95% CI ([+-][\d.]+) to ([+-][\d.]+)\)"
                   r"(?:, against a seed noise of ([\d.]+))?")
XRAY = re.compile(r"^\s*(\S+), \d+ pictures of \d+ patients: AUROC A [\d.]+, B [\d.]+, B minus A ([+-][\d.]+) "
                  r"\(95% CI ([+-][\d.]+) to ([+-][\d.]+)\)(?:, against a seed noise of ([\d.]+))?")


def parse(path):
    rows, verdict = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SPINE.match(line) or XRAY.match(line)
            if m:
                name, d, lo, hi, n = m.groups()
                rows[name] = (float(d), float(lo), float(hi), float(n) if n else None)
            elif line.strip() in ("DROPPED", "KEPT", "INCONCLUSIVE"):
                verdict = line.strip()
    return rows, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed0")
    ap.add_argument("seed1")
    ap.add_argument("--noise", nargs="*", default=[], help="name=value")
    x = ap.parse_args()
    name = "two-seed bar of %s and %s" % (os.path.basename(x.seed0), os.path.basename(x.seed1))
    with ledger.run("experiment", name) as led:
        led.settings_from({"seed0": x.seed0, "seed1": x.seed1, "noise": x.noise})
        return bar_of(x, led)


def bar_of(x, led):
    (r0, v0), (r1, v1) = parse(x.seed0), parse(x.seed1)
    over = {k: float(v) for k, v in (s.split("=", 1) for s in x.noise)}
    if not r0 or not r1:
        sys.exit("no measures found in %s" % (x.seed0 if not r0 else x.seed1))
    print("seed 0 %s, seed 1 %s" % (v0, v1))
    print("%-24s %9s %9s %9s %9s  %s" % ("measure", "seed 0", "seed 1", "mean", "bar", "verdict"))
    ok, gains = v0 != "DROPPED" and v1 != "DROPPED", []
    for name in r0:
        if name not in r1:
            continue
        d0, d1 = r0[name][0], r1[name][0]
        noise = over.get(name, r0[name][3] if r0[name][3] is not None else r1[name][3])
        mean = (d0 + d1) / 2.0
        if noise is None:
            print("%-24s %+9.4f %+9.4f %+9.4f %9s  (no noise: reported)" % (name, d0, d1, mean, "-"))
            led.number(name + ".seed0_diff", d0).number(name + ".seed1_diff", d1).number(name + ".mean_diff", mean)
            continue
        bar = -noise / math.sqrt(2.0)
        good = mean >= bar
        gain = d0 > noise and d1 > noise
        ok &= good
        if gain:
            gains.append(name)
        print("%-24s %+9.4f %+9.4f %+9.4f %+9.4f  %s%s" % (name, d0, d1, mean, bar, "pass" if good else "FAIL",
                                                          ", proven gain" if gain else ""))
        led.number(name + ".seed0_diff", d0).number(name + ".seed1_diff", d1).number(name + ".mean_diff", mean)
        led.number(name + ".noise", noise).number(name + ".bar", bar)
    line = "%s %s" % ("KEPT by the two-seed bar" if ok else "NOT KEPT by the two-seed bar",
                      "(proven gain: %s)" % ", ".join(gains) if gains else "")
    print(line)
    led.number("kept", 1 if ok else 0, "flag").number("proven_gains", len(gains), "count")
    led.note_is(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
