"""One command from a patient disc (any folder of DICOM files) to the web app's data folder.

    python run_pipeline.py <cd_root> <site_dir> [--work <dir>]

Steps: find and classify every MR series from its headers (series.py), DICOM -> NIfTI,
TotalSpineSeg on each region's sagittal T2, per-level measurements, smooth 3D meshes (.glb),
de-identified JPEG stacks, and the per-slice geometry used to place image marks.
The written reading (reading.json) comes from the reading step (AI readers + adversarial
verification + a checked synthesis); make_report.py merges it in. This script does not invent it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
sys.path.insert(0, HERE)
from metrics import main as measure_to_json  # noqa: E402
from series import REGION_NAME  # noqa: E402
from tss import segment  # noqa: E402


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cd_root")
    ap.add_argument("site_dir")
    ap.add_argument("--work", default=os.path.join(HERE, "..", "work"))
    a = ap.parse_args()
    nii = os.path.join(a.work, "nifti")
    tss_in = os.path.join(a.work, "tss_in")
    tss_out = os.path.join(a.work, "tss_out")
    run(PY, os.path.join(HERE, "to_nifti.py"), a.cd_root, nii)
    series = json.load(open(os.path.join(nii, "series.json")))
    # one sagittal T2 per region: the one with the most slices
    sag_t2 = {}
    for s in series:
        if s["plane"] == "sagittal" and s["sequence"] == "T2":
            if s["region"] not in sag_t2 or s["count"] > sag_t2[s["region"]]["count"]:
                sag_t2[s["region"]] = s
    if not sag_t2:
        sys.exit("No sagittal T2 series found; the 3D model and measurements need one per region.")
    os.makedirs(tss_in, exist_ok=True)
    for s in sag_t2.values():
        shutil.copy(os.path.join(nii, s["name"] + ".nii.gz"), tss_in)
    step2 = segment(tss_in, tss_out, os.path.join(a.work, "tss_data"))
    for code, s in sag_t2.items():
        img = os.path.join(nii, s["name"] + ".nii.gz")
        lab = os.path.join(step2, s["name"] + ".nii.gz")
        measure_to_json(img, lab, os.path.join(a.work, f"metrics_{s['name']}.json"))
        run(PY, os.path.join(HERE, "build_meshes.py"), lab, REGION_NAME[code], a.site_dir)
    run(PY, os.path.join(HERE, "build_stacks.py"), nii, step2, a.site_dir)
    run(PY, os.path.join(HERE, "geometry.py"), a.cd_root, a.site_dir, os.path.join(a.work, "geometry.json"))


if __name__ == "__main__":
    main()
