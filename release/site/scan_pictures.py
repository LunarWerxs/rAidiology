"""Build the seven scan pictures on the website (site/img/scans/*.webp) and their credits
(site/data/scan-pictures.json) from public research datasets under CC BY or NIH terms.

Deterministic: every file is named in CHOICES, nothing is random. Run from anywhere:
    python release/site/scan_pictures.py
Reads Datasets/public/<folder> (outside the repository). Never reads anything but those public sets.
"""
import csv
import io
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from modalities.xray.eval.prepare import read_grey  # noqa: E402

DATA = os.environ.get("RAIDIOLOGY_PUBLIC_DATA", os.path.join(ROOT, "..", "Datasets", "public"))
OUT = os.path.join(ROOT, "site", "img", "scans")
NAVY = (5, 16, 28)

# Fixed picks. Change a value here, never add randomness.
CHOICES = {
    "ct-organs": {"scan_id": "s0739", "size": (720, 900)},
    "mri-spine": {"image": "100_t2.mha", "size": (640, 900)},
    "xray-wrist": {"stem": "0076_0818101753_01_WRI-R1_M014", "zip": "images_part1.zip", "size": (720, 900)},
    "xray-chest": {"image": "00001015_000.png", "size": (800, 800), "trim": (0.07, 0, 0, 0.1)},
    "ct-lungs": {"patient": "LIDC-IDRI-0001", "size": (800, 800)},
    "mammogram": {"patient": "D1-0032", "file": 0, "size": (720, 900)},
    "xray-spine": {"image": "Dataset/Test/3753-F-049Y1.jpg", "size": (640, 900), "trim": (0.1, 0, 0, 0)},
}

# Normal structures only (no pathology class is ever listed).
ORGAN_COLOURS = {
    "liver": (230, 120, 90), "spleen": (170, 110, 200), "kidney_left": (240, 190, 70), "kidney_right": (240, 190, 70),
    "pancreas": (240, 160, 120), "stomach": (110, 190, 160), "gallbladder": (120, 200, 90),
    "lung_upper_lobe_left": (110, 170, 230), "lung_lower_lobe_left": (90, 150, 215),
    "lung_upper_lobe_right": (110, 170, 230), "lung_middle_lobe_right": (130, 190, 240),
    "lung_lower_lobe_right": (90, 150, 215), "heart": (225, 90, 120), "aorta": (240, 80, 70),
    "inferior_vena_cava": (90, 110, 220), "small_bowel": (230, 170, 190), "colon": (200, 160, 110),
    "urinary_bladder": (240, 220, 100), "esophagus": (150, 210, 200), "adrenal_gland_left": (255, 210, 120),
    "adrenal_gland_right": (255, 210, 120), "sacrum": (200, 200, 120), "hip_left": (180, 220, 140),
    "hip_right": (180, 220, 140), "femur_left": (150, 210, 170), "femur_right": (150, 210, 170),
}
BONE_GROUPS = {  # shown name -> (prefix test, colours cycled)
    "vertebrae": (lambda n: n.startswith("vertebrae_"), [(235, 225, 150), (205, 190, 120)]),
    "ribs": (lambda n: n.startswith("rib_"), [(235, 150, 190)]),
}
SPIDER_DISC_COLOURS = [(240, 170, 80), (90, 200, 220)]
SPIDER_VERT_COLOURS = [(150, 120, 230), (110, 160, 235)]

CREDITS = {
    "ct-organs": dict(
        title="CT, organs and bones coloured", scan="CT",
        dataset="TotalSegmentator dataset v2.0.1", licence="CC BY 4.0",
        citation="Wasserthal, Breit, Meyer, Pradella, Hinck, Sauter et al., \"TotalSegmentator: Robust Segmentation of 104 "
                 "Anatomic Structures in CT Images\", Radiology: Artificial Intelligence 5(5) (2023), doi:10.1148/ryai.230024; "
                 "data doi:10.5281/zenodo.10047292, CC BY 4.0.",
        url="https://doi.org/10.5281/zenodo.10047292",
        changes="windowed (width 400, level 40), cropped and coloured by LunarWerx, using the dataset's own masks"),
    "mri-spine": dict(
        title="MRI of the lumbar spine, vertebrae and discs coloured", scan="MRI",
        dataset="SPIDER lumbar spine MRI", licence="CC BY 4.0",
        citation="van der Graaf, van Hooff, Buckens et al., \"Lumbar spine segmentation in MR images: a dataset and a public "
                 "benchmark\", Scientific Data 11, 264 (2024), doi:10.1038/s41597-024-03090-w; data doi:10.5281/zenodo.10159290, CC BY 4.0.",
        url="https://doi.org/10.5281/zenodo.10159290",
        changes="middle sagittal slice, contrast stretched, cropped and coloured by LunarWerx, using the dataset's own masks"),
    "xray-wrist": dict(
        title="X-ray of a child's wrist", scan="X-ray",
        dataset="GRAZPEDWRI-DX v2", licence="CC BY 4.0",
        citation="Nagy, Janisch, Hrzic, Sorantin, Tschauner, \"A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for machine "
                 "learning\", Scientific Data 9, 222 (2022), doi:10.1038/s41597-022-01328-z; data doi:10.6084/m9.figshare.14825193, CC BY 4.0.",
        url="https://doi.org/10.6084/m9.figshare.14825193",
        changes="16-bit picture contrast stretched, cropped and resized by LunarWerx"),
    "xray-chest": dict(
        title="Chest X-ray", scan="X-ray",
        dataset="NIH ChestX-ray14", licence="No restrictions (NIH), attribution requested",
        citation="Wang, Peng, Lu, Lu, Bagheri, Summers, \"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks\", "
                 "IEEE CVPR 2017, pp. 3462-3471. Provided by the NIH Clinical Center (https://nihcc.app.box.com/v/ChestXray-NIHCC).",
        url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        changes="contrast stretched, cropped and resized by LunarWerx"),
    "ct-lungs": dict(
        title="CT of the chest, lung window", scan="CT",
        dataset="LIDC-IDRI", licence="CC BY 3.0",
        citation="Armato et al., The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI), "
                 "Medical Physics 38(2), 2011; data doi:10.7937/K9/TCIA.2015.LO9QL9SX; Clark et al., The Cancer Imaging Archive (TCIA), "
                 "J Digit Imaging 2013.",
        url="https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX",
        changes="windowed (width 1500, level -600), cropped and resized by LunarWerx"),
    "mammogram": dict(
        title="Mammogram", scan="Mammogram",
        dataset="CMMD (Chinese Mammography Database)", licence="CC BY 4.0",
        citation="Cui, C., Li, L., Cai, H., Fan, Z., Zhang, L., Dan, T., Li, J., & Wang, J. (2021). The Chinese Mammography Database "
                 "(CMMD): An online mammography database with biopsy confirmed types for machine diagnosis of breast. The Cancer "
                 "Imaging Archive. https://doi.org/10.7937/tcia.eqde-4b16.",
        url="https://doi.org/10.7937/tcia.eqde-4b16",
        changes="contrast stretched, cropped and resized by LunarWerx"),
    "xray-spine": dict(
        title="Side X-ray of the lower spine", scan="X-ray",
        dataset="Spondylolisthesis Vertebral Landmark Dataset v1 (Reyes, VSB-TU Ostrava)", licence="CC BY 4.0",
        citation="Spondylolisthesis Vertebral Landmark Dataset, version 1, Reyes, VSB-TU Ostrava, Mendeley Data, "
                 "doi:10.17632/5jdfdgp762.1, CC BY 4.0 (the record's licence field, read at the page listed in LICENSING.md).",
        url="https://data.mendeley.com/datasets/5jdfdgp762/1",
        changes="contrast stretched, cropped and resized by LunarWerx"),
}


def pub(*p):
    return os.path.join(DATA, *p)


def stretch(a, lo_p=1.0, hi_p=99.5):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [lo_p, hi_p])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def window(a, width, level):
    lo = level - width / 2
    return np.clip((a.astype(np.float32) - lo) / width, 0, 1)


def to_img(a01):
    return Image.fromarray((a01 * 255 + 0.5).astype(np.uint8)).convert("RGB")


def letterbox(im, size):
    """Fit im inside size on the navy background, centred."""
    w, h = size
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    bg = Image.new("RGB", size, NAVY)
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


def crop_to(a01, pad=0.03, thresh=0.04):
    """Crop a 2-D picture to where it is brighter than the background."""
    m = a01 > thresh
    ys, xs = np.where(m)
    if not len(ys):
        return a01
    py, px = int(pad * a01.shape[0]), int(pad * a01.shape[1])
    return a01[max(0, ys.min() - py):ys.max() + py + 1, max(0, xs.min() - px):xs.max() + px + 1]


def trim(a, fr):
    """Cut fractions (top, bottom, left, right) off a picture, to drop a corner label."""
    t, b, l, r = fr
    h, w = a.shape[:2]
    return a[int(h * t):h - int(h * b), int(w * l):w - int(w * r)]


def save(im, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".webp")
    for q in (80, 76, 72, 68, 62, 56):
        im.save(p, "WEBP", quality=q, method=6)
        if os.path.getsize(p) <= 150_000:
            break
    return p


def overlay(rgb, layers, alpha=0.5):
    """layers: list of (mask, colour). Blend colour at alpha with a thin darker outline."""
    from scipy import ndimage as ndi
    out = rgb.astype(np.float32)
    for m, c in layers:
        if not m.any():
            continue
        c = np.array(c, np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    for m, c in layers:
        if not m.any():
            continue
        edge = m & ~ndi.binary_erosion(m, iterations=1)
        out[edge] = np.array(c, np.float32) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def ct_organs(c):
    import nibabel as nib
    from scipy import ndimage as ndi
    d = pub("totalsegmentator", c["scan_id"])
    img = nib.as_closest_canonical(nib.load(os.path.join(d, "ct.nii.gz")))
    ct = np.asarray(img.dataobj, dtype=np.float32)
    sp = img.header.get_zooms()[:3]
    segdir = os.path.join(d, "segmentations")
    names = [f[:-7] for f in sorted(os.listdir(segdir)) if f.endswith(".nii.gz")]
    wanted = [n for n in names if n in ORGAN_COLOURS or any(t(n) for t, _ in BONE_GROUPS.values())]
    masks = {}
    for n in wanted:
        m = nib.as_closest_canonical(nib.load(os.path.join(segdir, n + ".nii.gz")))
        masks[n] = np.asarray(m.dataobj) > 0
    # coronal plane through the middle of the trunk: the y with the most coloured structure in the
    # band of the kidneys, spleen and liver
    core = np.zeros(ct.shape, bool)
    for n in ("kidney_left", "kidney_right", "liver", "spleen"):
        if n in masks:
            core |= masks[n]
    ys = np.where(core.any(axis=(0, 2)))[0]
    y = int(np.argmax(core.sum(axis=(0, 2)))) if len(ys) else ct.shape[1] // 2
    sl = ct[:, y, :]
    rows = {n: m[:, y, :] for n, m in masks.items()}

    def disp(a):  # [x, z] -> [z, x], head up, patient's right on the left of the picture
        return a[::-1, ::-1].T
    base = window(disp(sl), 400, 40)
    zoom_y, zoom_x = sp[2], sp[0]
    H = base.shape[0]
    new_h = int(round(H * zoom_y)); new_w = int(round(base.shape[1] * zoom_x))
    scale = 3.0 / min(zoom_x, zoom_y)  # upsample so masks outline smoothly
    big = (int(round(new_w * scale / 3.0)), int(round(new_h * scale / 3.0)))
    rgb = np.asarray(Image.fromarray((base * 255).astype(np.uint8)).resize(big, Image.BICUBIC).convert("RGB"))
    layers, shown = [], []
    ci = {k: 0 for k in BONE_GROUPS}
    for n in wanted:
        m = disp(rows[n]).astype(np.uint8)
        if not m.any():
            continue
        m = np.asarray(Image.fromarray(m * 255).resize(big, Image.NEAREST)) > 0
        if n in ORGAN_COLOURS:
            layers.append((m, ORGAN_COLOURS[n])); shown.append(n)
            continue
        for g, (t, cols) in BONE_GROUPS.items():
            if t(n):
                layers.append((m, cols[ci[g] % len(cols)])); ci[g] += 1
                if g not in shown:
                    shown.append(g)
    out = overlay(rgb, layers)
    # crop to the body outline
    body = ndi.binary_fill_holes(np.asarray(Image.fromarray((base * 255).astype(np.uint8)).resize(big)) > 40)
    ys2, xs2 = np.where(body)
    pad = 12
    out = out[max(0, ys2.min() - pad):ys2.max() + pad, max(0, xs2.min() - pad):xs2.max() + pad]
    return letterbox(Image.fromarray(out), c["size"]), pretty(shown), f"totalsegmentator/{c['scan_id']}/ct.nii.gz"


def pretty(names):
    seen, out = set(), []
    for n in names:
        n = n.replace("_left", "").replace("_right", "").replace("_upper_lobe", "").replace("_middle_lobe", "").replace("_lower_lobe", "")
        n = n.replace("_", " ")
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def mri_spine(c):
    import SimpleITK as sitk
    ip = pub("spider", "images", c["image"])
    img = sitk.ReadImage(ip)
    msk = sitk.ReadImage(pub("spider", "masks", c["image"]))
    a = sitk.GetArrayFromImage(img)[:, :, :]   # [z, y, x] of the file; sagittal slices run along x
    mk = sitk.GetArrayFromImage(msk)
    mid = a.shape[2] // 2
    sl, ms = a[:, :, mid], mk[:, :, mid]
    sp = img.GetSpacing()  # (x, y, z)
    sl, ms = sl[::-1, :], ms[::-1, :]          # head up
    base = stretch(sl, 1, 99.7)
    layers, shown = [], set()
    labs = [v for v in np.unique(ms) if v]
    vi = di = 0
    for v in labs:
        m = ms == v
        if v == 100:
            continue
        if 1 <= v < 100:
            layers.append((m, SPIDER_VERT_COLOURS[vi % 2])); vi += 1; shown.add("vertebrae")
        elif v >= 200:
            layers.append((m, SPIDER_DISC_COLOURS[di % 2])); di += 1; shown.add("discs")
    h, w = base.shape
    W = int(round(w * sp[1])); Hh = int(round(h * sp[2]))
    k = 3.0 / min(sp[1], sp[2])
    big = (int(round(W * k / 3)), int(round(Hh * k / 3)))
    rgb = np.asarray(to_img(base).resize(big, Image.BICUBIC))
    lay = [(np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(big, Image.NEAREST)) > 0, col) for m, col in layers]
    out = overlay(rgb, lay)
    ys, xs = np.where(np.any([m for m, _ in lay], axis=0))
    pad = 60
    out = out[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad * 3):xs.max() + pad * 3]
    return letterbox(Image.fromarray(out), c["size"]), sorted(shown), f"spider/images/{c['image']}"


def xray_wrist(c):
    z = zipfile.ZipFile(pub("grazpedwri-dx", c["zip"]))
    a = read_grey(io.BytesIO(z.read(c["stem"] + ".png")))
    with open(pub("grazpedwri-dx", "dataset.csv"), encoding="utf-8-sig") as f:
        row = next(r for r in csv.DictReader(f) if r["filestem"] == c["stem"])
    assert not any(row[k].strip() for k in ("fracture_visible", "cast", "metal", "osteopenia", "diagnosis_uncertain")), row
    b = crop_to(stretch(a, 0.5, 99.8), thresh=0.03)
    return letterbox(to_img(b), c["size"]), ["none"], f"grazpedwri-dx/{c['zip']}/{c['stem']}.png"


def xray_chest(c):
    with open(pub("nih-chestxray14", "Data_Entry_2017_v2020.csv"), encoding="utf-8-sig") as f:
        row = next(r for r in csv.DictReader(f) if r["Image Index"] == c["image"])
    assert row["Finding Labels"] == "No Finding" and row["View Position"] == "PA", row
    a = read_grey(pub("nih-chestxray14", "images", c["image"]))
    b = crop_to(trim(stretch(a, 0.5, 99.7), c["trim"]), thresh=0.03)
    return letterbox(to_img(b), c["size"]), ["none"], f"nih-chestxray14/images/{c['image']}"


def dicom(path):
    import pydicom
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    return ds, a * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))


def ct_lungs(c):
    import pydicom
    root = pub("lidc-idri", c["patient"])
    series = os.path.join(root, sorted(os.listdir(root))[0])
    files = sorted(x for x in os.listdir(series) if x.endswith('.dcm'))
    zs = []
    for f in files:
        ds = pydicom.dcmread(os.path.join(series, f), stop_before_pixels=True)
        zs.append((float(ds.ImagePositionPatient[2]), f))
    zs.sort()
    # the slice with the most lung in the middle half of the scan (lung = air inside the body)
    from scipy import ndimage as ndi
    best = None
    for _, fn in zs[len(zs) // 4: 3 * len(zs) // 4: 4]:
        _, h = dicom(os.path.join(series, fn))
        lab, _n = ndi.label(h < -300)
        outside = np.isin(lab, np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        n = int(((h < -800) & ~outside).sum())
        if best is None or n > best[0]:
            best = (n, fn)
    f = best[1]
    ds, hu = dicom(os.path.join(series, f))
    lab, _n = ndi.label(hu < -300)
    outside = np.isin(lab, np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    b = window(hu, 1500, -600)
    b[outside] = 0
    # the table and the corners are not the body: keep the biggest bright blob's box via row/col counts
    cols = (hu > -300).sum(axis=0); rows_ = (hu > -300).sum(axis=1)
    xs = np.where(cols > 20)[0]; ys = np.where(rows_ > 20)[0]
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # lung-window colour of the air outside the body is dark grey: set it to the navy so it blends
    return letterbox(to_img(b), c["size"]), ["none"], f"lidc-idri/{c['patient']}/{os.path.basename(series)}/{f}"


def mammogram(c):
    root = pub("cmmd", c["patient"])
    series = os.path.join(root, sorted(os.listdir(root))[0])
    f = sorted(os.listdir(series))[c["file"]]
    ds, a = dicom(os.path.join(series, f))
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    b = stretch(a, 1, 99.8)
    if b[:, : b.shape[1] // 2].mean() < b[:, b.shape[1] // 2:].mean():
        b = b[:, ::-1]                     # chest wall on the left
    b = crop_to(b, pad=0.02, thresh=0.08)
    return letterbox(to_img(b), c["size"]), ["none"], f"cmmd/{c['patient']}/{os.path.basename(series)}/{f}"


def xray_spine(c):
    a = np.asarray(Image.open(pub("spondylolisthesis-landmarks", c["image"])).convert("L"))
    b = crop_to(trim(stretch(a, 0.5, 99.7), c["trim"]), thresh=0.04)
    return letterbox(to_img(b), c["size"]), ["none"], f"spondylolisthesis-landmarks/{c['image']}"


BUILDERS = {"ct-organs": ct_organs, "mri-spine": mri_spine, "xray-wrist": xray_wrist, "xray-chest": xray_chest,
            "ct-lungs": ct_lungs, "mammogram": mammogram, "xray-spine": xray_spine}


def main(only=None):
    credits = []
    for name, fn in BUILDERS.items():
        if only and name not in only:
            continue
        im, anatomy, src = fn(CHOICES[name])
        p = save(im, name)
        print(name, im.size, os.path.getsize(p), src)
        credits.append({"file": f"img/scans/{name}.webp", **CREDITS[name], "anatomy": anatomy, "source_file": src})
    if not only:
        os.makedirs(os.path.join(ROOT, "site", "data"), exist_ok=True)
        with open(os.path.join(ROOT, "site", "data", "scan-pictures.json"), "w", encoding="utf-8") as f:
            json.dump(credits, f, indent=2, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
