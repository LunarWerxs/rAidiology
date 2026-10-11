"""CPU selftest for handler.py: names only, pathology dropped, no measurements. Prints 'selftest ok', exit 0, else 1."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RAIDIOLOGY_DEVICE"] = "cpu"
os.environ["RAID_SERVE_REPO"] = REPO
os.environ["RAIDIOLOGY_MODELS"] = os.path.abspath(os.path.join(REPO, "..", "Models"))
os.environ["RAID_CT_VERSION"] = "2"
SCAN = os.environ.get("RAID_SELFTEST_SCAN", os.path.abspath(os.path.join(
    REPO, "..", "Datasets", "public", "totalsegmentator", "s0654", "ct.nii.gz")))
sys.path.insert(0, HERE)

import handler  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

BAD_WORDS = ("millilit", "volume", "size", "count", "voxel", "density", "score")
fails = []


def check(ok, what):
    if not ok:
        fails.append(what)


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield path + "/" + str(k), str(k)
            yield from walk(v, path + "/" + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, path + "[%d]" % i)
    elif isinstance(x, str):
        yield path, x


listing, measured, shape, labels, affine = handler.read(SCAN)
with tempfile.TemporaryDirectory() as d:
    out_path = os.path.join(d, "labels.nii.gz")
    handler.write_labels(labels, affine, out_path)
    written = np.asanyarray(nib.load(out_path).dataobj)
out = {"structures": listing, "labels_written": True, "seconds": {"read": 0.0}, "shape": shape,
       "measured_spacing": measured, "version": handler.VERSION, "not_a_medical_device": True, "notice": handler.NOTICE}
json.dumps(out)
names = [s["name"] for s in listing]
labs = [s["label"] for s in listing]
check(len(listing) >= 20, "fewer than 20 structures: %d" % len(listing))
check(labs == sorted(labs), "not sorted by label")
check(handler.EXCLUDED and not set(names) & set(handler.EXCLUDED.values()), "excluded name listed")
check(all(set(s) == {"label", "name"} for s in listing), "structure entries carry more than label and name")
for p, text in walk(out):
    low = text.lower()
    if p.startswith("/shape") or p.startswith("/seconds"):
        continue
    if any(w in low for w in BAD_WORDS) and p != "/notice":
        fails.append("measurement word at %s: %s" % (p, text))
present = {int(v) for v in np.unique(written)}
check(present <= set(labs) | {0}, "label map holds unlisted values: %s" % sorted(present - set(labs) - {0}))
check(not present & set(handler.EXCLUDED), "excluded labels occur in the map")
if fails:
    print("selftest FAILED:\n" + "\n".join(fails))
    sys.exit(1)
print("selftest ok (%d structures; excluded: %s)" % (len(listing), ", ".join(handler.EXCLUDED.values())))
