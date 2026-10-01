"""Convert every single-plane MR series on a patient disc into a NIfTI volume (pixels + geometry
only; no patient tags survive), plus series.json describing what was found.
Usage: python to_nifti.py <cd_root> <out_dir>"""
import json
import os
import sys

import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from series import scan  # noqa: E402


def main(cd, out):
    os.makedirs(out, exist_ok=True)
    found = scan(cd)
    for s in found:
        if s["plane"] == "mixed":
            continue  # multi-plane localizers are not one volume
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(s["files"])
        try:
            img = reader.Execute()
        except RuntimeError as e:
            print("skipped", s["name"], str(e).splitlines()[-1][:120])
            continue
        sitk.WriteImage(img, os.path.join(out, f"{s['name']}.nii.gz"))
        print(s["name"], s["plane"], s["sequence"], img.GetSize(), [round(x, 3) for x in img.GetSpacing()])
    json.dump([{k: v for k, v in s.items() if k != "files"} for s in found],
              open(os.path.join(out, "series.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
