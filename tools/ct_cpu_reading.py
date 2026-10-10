"""The CT reading as the MCP server does it, on the processor, over development scans: how long a
scan takes, the scorer's rows, and a fingerprint of every scan's labels.

    python tools/ct_cpu_reading.py <out.csv> [--scans 60] [--version 1] [--fold 0] [--set key=<json>]

Reads the first `--scans` scans of the fold's held-out development scans with that fold's model,
exactly as `segmenter.py scan` reads one (`read_volume`, `labels_on_grid`, the floor). The
environment picks the arm: RAIDIOLOGY_CPU_LAYOUT (plain or channels_last), RAIDIOLOGY_LABEL_TILE (0
for whole slabs). Writes `<out.csv>` (rows for `modalities/ct/eval/compare.py --a <the card's
oof_fold0.csv> --b <out.csv>`) and `<out>.labels.json` (scan -> sha256 of its labels), so two arms
that should agree to the voxel can be checked to: `--same-as <other.labels.json>`.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("RAIDIOLOGY_DEVICE", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "modalities", "ct", "eval"), ROOT]
import numpy as np  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import pandas as pd  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import datasets  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import prepare  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import segmenter as s  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
from core import ledger  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--scans", type=int, default=60)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--same-as", help="another run's .labels.json: count the scans whose labels differ")
    ap.add_argument("--set", action="append", help="key=<json>, a reading setting to try (overlap)")
    a = ap.parse_args()
    datasets.guard_test("dev")
    s.steady_inference()
    models, meta = s.load_models(a.version, [a.fold])
    names = meta["structures"]
    split = s.load_split()
    skipped = prepare.unprepared("dev")
    scans = sorted(k for k, f in split["dev_folds"].items() if f == a.fold and k not in skipped)[:a.scans]
    reading = s.reading_of(meta, a.set, part=a.fold)
    arm = {"layout": s.CPU_LAYOUT, "label_tile": s.LABEL_TILE, "half": str(s.READ_HALF), "scans": len(scans),
           **({"reading": a.set} if a.set else {})}
    if s.CPU_ENGINE != "torch":
        arm.update(engine=s.CPU_ENGINE, precision=s.OV_PRECISION)
    rows, prints, seconds = [], {}, []
    with ledger.run("segmenter", "read on a processor v%d fold %d" % (a.version, a.fold)) as led:
        led.settings_from(arm)
        for i, scan in enumerate(scans, 1):
            t0 = time.time()
            acc, weight = s.read_volume(models, np.load(prepare.lo_files(scan)[0]), meta["cfg"]["patch"], reading, names)
            pred = s.under_the_floor(s.labels_on_grid(acc, weight, s._native_shape(scan)), reading["floor"], names)
            seconds.append(time.time() - t0)
            del acc, weight
            prints[scan] = hashlib.sha256(pred.cpu().numpy().tobytes()).hexdigest()
            truth, said, both = s.counts_against_truth(pred, scan, len(names))
            rows += s.rows_for(scan, "dev", a.fold, names, truth, said, both)
            if i % 10 == 0 or i == len(scans):
                print("  %d of %d, %.1f s a scan so far" % (i, len(scans), float(np.mean(seconds))), flush=True)
        pd.DataFrame(rows)[s.COLUMNS].to_csv(a.out, index=False)
        with open(os.path.splitext(a.out)[0] + ".labels.json", "w", encoding="utf-8") as f:
            json.dump(prints, f, indent=0)
        led.number("seconds_per_scan", float(np.mean(seconds)), "s").number("scans", len(scans), "count")
        differ = None
        if a.same_as:
            other = json.load(open(a.same_as, encoding="utf-8"))
            both_read = [k for k in prints if k in other]
            differ = sum(prints[k] != other[k] for k in both_read)
            led.number("scans_with_different_labels", differ, "count")
    print("%s: %d scans, %.2f s a scan (median %.2f)%s -> %s" % (
        json.dumps(arm), len(scans), float(np.mean(seconds)), float(np.median(seconds)),
        "" if differ is None else ", labels differ on %d of %d against %s" % (differ, len(both_read), a.same_as),
        a.out))


if __name__ == "__main__":
    main()
