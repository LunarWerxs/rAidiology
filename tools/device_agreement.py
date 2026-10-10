"""Does the same radiograph get the same answer on a different machine?

A score that moves with the hardware is a quiet way for a medical-ish tool to be wrong: two people
with the same picture get different words, and nothing in the scorecard shows it. So it is measured
rather than assumed, and the numbers go in the ledger with everything else.

Two questions, separately:

- **Repeatable here?** The same image, scored twice on this machine. Anything but zero is a defect,
  and `detector.steady_inference()` exists because the answer used to be no on a graphics card.
- **The same across machine kinds?** The same image on the processor and on the graphics card. This
  one is not zero and cannot be made zero: different convolution kernels add up in a different
  order. What matters is how big it is against the threshold, so this also counts how many images in
  the held-out test sit inside that distance of the line and could therefore flip.

    python tools/device_agreement.py [--images 20]

It needs a graphics card to compare two kinds of machine; with none it still checks repeatability
and says plainly that the comparison was not run rather than reporting agreement.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "xray", "eval"))
from core import ledger  # noqa: E402

CHILD = """
import json, os, sys
sys.path.insert(0, r"%s"); sys.path.insert(0, os.path.join(r"%s", "modalities", "xray", "eval"))
import detector, datasets
table = datasets.index("grazpedwri")
paths = [datasets.image_path("grazpedwri", v) for v in table["image"].head(%d)]
rows, _ = detector.score_images(paths, 1)
again, _ = detector.score_images(paths, 1)
print("@@" + json.dumps({"first": [r["score"] for r in rows], "again": [r["score"] for r in again]}))
"""


def on(device, n):
    """Score the same images twice in a fresh process on one device.

    A fresh process because the weights are loaded onto whichever device was chosen at import, so
    the two devices cannot be compared inside one run without reloading everything anyway."""
    env = dict(os.environ, RAIDIOLOGY_DEVICE=device)
    out = subprocess.run([sys.executable, "-c", CHILD % (ROOT, ROOT, n)], capture_output=True,
                         text=True, env=env, cwd=ROOT, check=False)
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise SystemExit("no scores came back from the %s run:\n%s" % (device, out.stderr[-800:]))


def near_the_line(band):
    """How much of the held-out test sits inside `band` of the threshold, per dataset."""
    import pandas as pd  # noqa: PLC0415
    from core import paths as P  # noqa: PLC0415
    meta = json.load(open(os.path.join(P.MODELS, "xray-fracture-detector-v1", "meta.json"),
                          encoding="utf-8"))
    out = {}
    for name, cut in meta["thresholds"].items():
        f = os.path.join(P.WORK, "eval", "xray", "pred_cnn_%s_test.csv" % name)
        if not os.path.exists(f):
            continue
        s = pd.read_csv(f)["fracture_score"]
        out[name] = (int(((s - cut).abs() <= band).sum()), len(s), cut)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=20)
    a = ap.parse_args()
    import torch  # noqa: PLC0415

    cpu = on("cpu", a.images)
    repeat_cpu = max(abs(x - y) for x, y in zip(cpu["first"], cpu["again"]))
    print("processor, the same image twice:   %.2e" % repeat_cpu)

    gap = repeat_gpu = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = on("cuda", a.images)
        repeat_gpu = max(abs(x - y) for x, y in zip(gpu["first"], gpu["again"]))
        gap = max(abs(x - y) for x, y in zip(cpu["first"], gpu["first"]))
        print("graphics card, the same image twice: %.2e" % repeat_gpu)
        print("processor against graphics card:     %.2e" % gap)
    else:
        print("no graphics card here, so the two kinds of machine were NOT compared. "
              "This is not agreement; it is an unanswered question.")

    band = gap if gap else 0.03
    for name, (n, total, cut) in near_the_line(band).items():
        print("%-11s within %.3f of the line (%.3f): %d of %d, %.2f%%"
              % (name, band, cut, n, total, 100.0 * n / total))

    with ledger.run("measurement", "device agreement") as r:
        r.number("repeat_cpu", repeat_cpu).number("images", a.images, "count")
        if gap is not None:
            r.number("repeat_gpu", repeat_gpu).number("cpu_vs_gpu", gap)
        for name, (n, total, _cut) in near_the_line(band).items():
            r.number("%s_share_within_the_gap" % name, n / total)
        r.note_is("cpu vs gpu %s" % ("%.2e" % gap if gap is not None else "not measured"))


if __name__ == "__main__":
    main()
