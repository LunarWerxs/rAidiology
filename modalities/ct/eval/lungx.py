"""SPIE-AAPM Lung CT Challenge dataset: nodules with benign/cancer labels in 1 mm lung boxes.

The SPIE-AAPM Lung CT Challenge (CC BY 3.0, 70 CT series of 70 patients, 12.1 GB) contains
two sets: 10 calibration scans with 10 nodules and 60 test scans with 73 nodules. Each nodule
has a location in pixel coordinates from two Excel sheets (CalibrationSet_NoduleData.xlsx and
TestSet_NoduleData_PublicRelease_wTruth.xlsx), an image number, and a diagnosis.

This module:
- Reads the Excel sheets using the standard library (zipfile and xml.etree) and parses:
  * Calibration set: 10 nodules (its other rows are footnotes) with Scan Number (CT-Training-{lc,be}NNN), x,y in pixel
    coordinates (packed as a 6-digit number: e.g. 120325 = x:120, y:325), image number,
    and diagnosis (malignant or benign, as string indices into the Excel shared strings).
  * Test set: 73 nodules with Scan Number (LUNGx-CTXXX), nodule number, x,y as "x, y"
    strings, image number, and diagnosis (Benign nodule, Primary lung cancer, etc.).
- Maps Scan Number to PatientID from series.json, which is TCIA's PatientID field.
- Interprets the image number as the DICOM InstanceNumber (checked: it puts 81 of the 83
  nodule centres on soft tissue, the file-name order 48; see _get_dicom_headers).
- Converts pixel (x, y) to world millimetres using the first image's PixelSpacing and
  ImagePositionPatient, without orientation handling (SPIE-AAPM scans are not prone).
- Converts world (x, y, z) mm to voxel indices using lung_prepare.world_to_voxel().
- Writes 1 mm lung boxes to <WORK>/eval/lung/vol_lungx/<SeriesInstanceUID>.npy using
  lung_prepare.build(), and a CSV table of nodules with world coordinates and HU values.

Nodule diagnosis labels (from both sets, listed with their label value):
- Benign nodule: 0
- Primary lung cancer: 1
- Primary lung cancer (with space): 1
- Suspicious malignant nodule: 1

    python modalities/ct/eval/lungx.py prepare [--jobs N]
    python modalities/ct/eval/lungx.py nodules
    python modalities/ct/eval/lungx.py check
"""
import argparse
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ledger, paths  # noqa: E402
import lung_prepare  # noqa: E402


def _read_xlsx(xlsx_path):
    """Read Excel sheet and return (shared_strings, rows_data).

    rows_data is a list of dicts, where each dict maps column letter to cell value
    (string indices are returned as ints, numeric values as floats or strings as-is).
    """
    with zipfile.ZipFile(xlsx_path) as z:
        # Read shared strings
        with z.open('xl/sharedStrings.xml') as f:
            root = ET.parse(f).getroot()
        ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        strings = []
        for t in root.findall('.//x:t', ns):
            if t.text:
                strings.append(t.text)

        # Read sheet1
        with z.open('xl/worksheets/sheet1.xml') as f:
            root = ET.parse(f).getroot()

        rows = root.findall('.//x:row', ns)
        rows_data = []
        for row in rows:
            row_dict = {}
            for cell in row.findall('x:c', ns):
                ref = cell.get('r', '')
                col = ref.rstrip('0123456789')

                # Get cell type and value
                cell_type = cell.get('t', 'n')
                val_elem = cell.find('x:v', ns)
                val = val_elem.text if val_elem is not None else None

                if val is None:
                    pass  # val stays None
                elif cell_type == 's':
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                else:
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass

                row_dict[col] = val
            rows_data.append(row_dict)

        return strings, rows_data


def _get_series_lookup():
    """Load series.json and create a map from PatientID to series info (case-insensitive)."""
    series_file = os.path.join(paths.DATASETS, 'public', 'spie-aapm', 'series.json')
    with open(series_file) as f:
        series = json.load(f)
    # Create a case-insensitive lookup: convert all keys to uppercase
    return {s['PatientID'].upper(): s for s in series}


def _get_dicom_headers(scan_dir, image_num):
    """Geometry of the slice whose DICOM InstanceNumber is `image_num` (None when there is none).

    The InstanceNumber, not the file-name order: measured 2026-10-03 on all 83 nodules, the HU at
    (x, y) on the slice of that InstanceNumber is above -500 (soft tissue) for 81, on the n-th file by
    name for 48, on the n-th slice by rising z for 41."""
    import pydicom
    for path in sorted(glob.glob(os.path.join(scan_dir, "*.dcm"))):
        d = pydicom.dcmread(path, stop_before_pixels=True)
        if int(getattr(d, "InstanceNumber", -1)) != int(image_num):
            continue
        if not hasattr(d, "ImagePositionPatient") or not hasattr(d, "PixelSpacing"):
            return None
        return {
            'x0': float(d.ImagePositionPatient[0]),
            'y0': float(d.ImagePositionPatient[1]),
            'z': float(d.ImagePositionPatient[2]),
            'pixel_spacing_row': float(d.PixelSpacing[0]),
            'pixel_spacing_col': float(d.PixelSpacing[1]),
        }
    return None


def _parse_calibration_set():
    """Parse CalibrationSet_NoduleData.xlsx.

    Returns list of dicts with keys: set, scan, nodule, patient, seriesuid,
    coordX, coordY, coordZ (world mm), label, diagnosis.
    """
    xlsx_path = os.path.join(paths.DATASETS, 'public', 'spie-aapm', 'CalibrationSet_NoduleData.xlsx')
    strings, rows = _read_xlsx(xlsx_path)

    series_lookup = _get_series_lookup()
    nodules = []

    # Parse data rows (skip header at row 0)
    for row_idx, row in enumerate(rows[1:], 1):
        if 'A' not in row or 'B' not in row or 'C' not in row or 'D' not in row:
            continue

        scan_idx = row['A']
        xy_packed = row['B']
        image_num = row['C']
        diag_idx = row['D']

        if scan_idx is None or xy_packed is None or image_num is None or diag_idx is None:
            continue

        # Resolve scan name from index
        if isinstance(scan_idx, int) and scan_idx < len(strings):
            scan_name = strings[scan_idx]
        else:
            continue

        # Resolve diagnosis from index
        if isinstance(diag_idx, int) and diag_idx < len(strings):
            diagnosis = strings[diag_idx]
        else:
            continue

        # Decode x,y: packed number where y is always 3 digits, x is remaining
        # e.g., 120325 -> x=120, y=325; 95328 -> x=95, y=328
        xy_str = str(int(xy_packed))
        if len(xy_str) >= 4:
            x_pix = int(xy_str[:-3])
            y_pix = int(xy_str[-3:])
        else:
            continue  # Skip malformed coordinates

        # Look up series (case-insensitive)
        if scan_name.upper() not in series_lookup:
            continue
        series_info = series_lookup[scan_name.upper()]
        patient = series_info['PatientID']
        seriesuid = series_info['SeriesInstanceUID']
        scan_dir = os.path.join(paths.DATASETS, 'public', 'spie-aapm', patient, seriesuid)

        # Get DICOM headers to convert pixel coords to world mm
        geom = _get_dicom_headers(scan_dir, int(image_num))
        if geom is None:
            continue

        # Convert pixel to world mm: (1,1) is upper-left, x is column, y is row
        # In DICOM: x = x0 + (pixel_column - 1) * pixel_spacing_col
        #           y = y0 + (pixel_row - 1) * pixel_spacing_row
        x_mm = geom['x0'] + (x_pix - 1) * geom['pixel_spacing_col']
        y_mm = geom['y0'] + (y_pix - 1) * geom['pixel_spacing_row']
        z_mm = geom['z']

        # Parse diagnosis
        label = 1 if 'malignant' in diagnosis.lower() else 0

        nodules.append({
            'set': 'calibration',
            'scan': scan_name,
            'nodule': row_idx,
            'patient': patient,
            'seriesuid': seriesuid,
            'coordX': x_mm,
            'coordY': y_mm,
            'coordZ': z_mm,
            'label': label,
            'diagnosis': diagnosis,
        })

    return nodules


def _parse_test_set():
    """Parse TestSet_NoduleData_PublicRelease_wTruth.xlsx.

    Returns list of dicts with keys: set, scan, nodule, patient, seriesuid,
    coordX, coordY, coordZ (world mm), label, diagnosis.
    """
    xlsx_path = os.path.join(paths.DATASETS, 'public', 'spie-aapm', 'TestSet_NoduleData_PublicRelease_wTruth.xlsx')
    strings, rows = _read_xlsx(xlsx_path)

    series_lookup = _get_series_lookup()
    nodules = []

    # Parse data rows (skip header at row 0)
    for row_idx, row in enumerate(rows[1:], 1):
        if 'A' not in row or 'C' not in row or 'D' not in row or 'E' not in row:
            continue

        scan_idx = row['A']
        xy_idx = row['C']
        image_num = row['D']
        diag_idx = row['E']

        if scan_idx is None or xy_idx is None or image_num is None or diag_idx is None:
            continue

        # Resolve scan name from index
        if isinstance(scan_idx, int) and scan_idx < len(strings):
            scan_name = strings[scan_idx]
        else:
            continue

        # Resolve x,y from index
        if isinstance(xy_idx, int) and xy_idx < len(strings):
            xy_str = strings[xy_idx]
            parts = xy_str.split(',')
            if len(parts) != 2:
                continue
            try:
                x_pix = float(parts[0].strip())
                y_pix = float(parts[1].strip())
            except ValueError:
                continue
        else:
            continue

        # Resolve diagnosis from index
        if isinstance(diag_idx, int) and diag_idx < len(strings):
            diagnosis = strings[diag_idx]
        else:
            continue

        # Look up series (case-insensitive)
        if scan_name.upper() not in series_lookup:
            continue
        series_info = series_lookup[scan_name.upper()]
        patient = series_info['PatientID']
        seriesuid = series_info['SeriesInstanceUID']
        scan_dir = os.path.join(paths.DATASETS, 'public', 'spie-aapm', patient, seriesuid)

        # Get DICOM headers to convert pixel coords to world mm
        geom = _get_dicom_headers(scan_dir, int(image_num))
        if geom is None:
            continue

        # Convert pixel to world mm
        x_mm = geom['x0'] + (x_pix - 1) * geom['pixel_spacing_col']
        y_mm = geom['y0'] + (y_pix - 1) * geom['pixel_spacing_row']
        z_mm = geom['z']

        # Parse diagnosis
        label = 0 if 'benign' in diagnosis.lower() else 1

        nodules.append({
            'set': 'test',
            'scan': scan_name,
            'nodule': row_idx,
            'patient': patient,
            'seriesuid': seriesuid,
            'coordX': x_mm,
            'coordY': y_mm,
            'coordZ': z_mm,
            'label': label,
            'diagnosis': diagnosis,
        })

    return nodules


def prepare(jobs=None):
    """Build 1 mm lung boxes for all complete SPIE-AAPM scans."""
    series_lookup = _get_series_lookup()

    scans = []
    for series_info in series_lookup.values():
        patient = series_info['PatientID']
        seriesuid = series_info['SeriesInstanceUID']
        scan_dir = os.path.join(paths.DATASETS, 'public', 'spie-aapm', patient, seriesuid)

        # Check if scan is complete
        if os.path.exists(os.path.join(scan_dir, '.unpacked')):
            scans.append((seriesuid, patient, scan_dir))

    if not scans:
        print("No complete SPIE-AAPM scans found")
        return

    odir = os.path.join(paths.WORK, 'eval', 'lung', 'vol_lungx')
    lung_prepare.build(jobs=jobs, odir=odir, scans=scans)


def nodules(led=None):
    """Write nodules.csv with world coordinates and HU values."""
    cal_nodules = _parse_calibration_set()
    test_nodules = _parse_test_set()
    all_nodules = cal_nodules + test_nodules

    odir = os.path.join(paths.WORK, 'eval', 'lung', 'lungx')
    os.makedirs(odir, exist_ok=True)

    csv_path = os.path.join(odir, 'nodules.csv')

    # Load volumes and metadata to compute centre_hu
    vol_dir = os.path.join(paths.WORK, 'eval', 'lung', 'vol_lungx')

    rows = []
    for n in all_nodules:
        # Get volume and metadata if available
        centre_hu = float('nan')
        vol_path, meta_path = lung_prepare.paths_for(vol_dir, n['seriesuid'])
        if os.path.exists(vol_path) and os.path.exists(meta_path):
            try:
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                vol = np.load(vol_path)

                # Convert world coords to voxel indices
                v = lung_prepare.world_to_voxel(meta, [n['coordX'], n['coordY'], n['coordZ']])
                c = np.rint(v).astype(int)

                # Extract 3x3x3 box around centre and compute mean HU
                if (c >= 1).all() and (c <= np.array(vol.shape) - 2).all():
                    box = vol[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, c[2] - 1:c[2] + 2]
                    centre_hu = float(lung_prepare.to_hu(box).mean())
            except Exception:
                pass

        rows.append([
            n['set'],
            n['scan'],
            n['nodule'],
            n['patient'],
            n['seriesuid'],
            n['coordX'],
            n['coordY'],
            n['coordZ'],
            n['label'],
            n['diagnosis'],
            centre_hu,
        ])

    # Atomic write
    def write_csv(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['set', 'scan', 'nodule', 'patient', 'seriesuid',
                        'coordX', 'coordY', 'coordZ', 'label', 'diagnosis', 'centre_hu'])
            w.writerows(rows)

    part_path = csv_path + '.part'
    write_csv(part_path)
    os.replace(part_path, csv_path)
    print(f"Wrote {csv_path}: {len(rows)} nodules")
    if led is not None:
        led.number("nodules", len(rows), "count")
        led.number("nodules_without_centre_hu", sum(1 for r in rows if r[-1] != r[-1]), "count")


def check(led=None):
    """Assert counts and properties of nodules."""
    cal_nodules = _parse_calibration_set()
    test_nodules = _parse_test_set()

    assert len(cal_nodules) == 10, f"Expected 10 calibration nodules, got {len(cal_nodules)}"
    assert len(test_nodules) == 73, f"Expected 73 test nodules, got {len(test_nodules)}"

    test_labels = [n['label'] for n in test_nodules]
    assert sum(l == 1 for l in test_labels) == 36, f"Expected 36 malignant test nodules, got {sum(l == 1 for l in test_labels)}"
    assert sum(l == 0 for l in test_labels) == 37, f"Expected 37 benign test nodules, got {sum(l == 0 for l in test_labels)}"

    # Check that nodules are inside their boxes (when boxes are built)
    vol_dir = os.path.join(paths.WORK, 'eval', 'lung', 'vol_lungx')
    centre_hus = []
    for n in cal_nodules + test_nodules:
        vol_path, meta_path = lung_prepare.paths_for(vol_dir, n['seriesuid'])
        if os.path.exists(vol_path) and os.path.exists(meta_path):
            try:
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                vol = np.load(vol_path)

                v = lung_prepare.world_to_voxel(meta, [n['coordX'], n['coordY'], n['coordZ']])
                c = np.rint(v).astype(int)

                assert (c >= 1).all() and (c <= np.array(vol.shape) - 2).all(), \
                    f"Nodule {n['scan']}/{n['nodule']} at voxel {c} outside crop {vol.shape}"

                box = vol[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, c[2] - 1:c[2] + 2]
                hu = float(lung_prepare.to_hu(box).mean())
                centre_hus.append(hu)
                if hu <= -500:      # a faint (ground-glass) nodule can be this low; many would mean a wrong slice
                    print(f"  {n['scan']}/{n['nodule']}: centre {hu:.0f} HU")
            except FileNotFoundError:
                pass

    assert centre_hus, "no LUNGx box built yet: run prepare first"
    dense = sum(h > -500 for h in centre_hus)
    assert dense >= 0.9 * len(centre_hus) and np.median(centre_hus) > -500,         f"only {dense} of {len(centre_hus)} centres above -500 HU: the coordinates are wrong"
    if centre_hus:
        print(f"{dense} of {len(centre_hus)} centres above -500 HU. Centre HU: median {float(np.median(centre_hus)):.0f}, min {float(np.min(centre_hus)):.0f}, "
              f"max {float(np.max(centre_hus)):.0f}")
    print(f"Calibration: {len(cal_nodules)} nodules; Test: {len(test_nodules)} nodules "
          f"({sum(l == 1 for l in test_labels)} malignant, {sum(l == 0 for l in test_labels)} benign)")
    print("check ok")
    if led is not None:
        led.number("calibration_nodules", len(cal_nodules), "count").number("test_nodules", len(test_nodules), "count")
        led.number("test_malignant", sum(l == 1 for l in test_labels), "count")
        led.number("test_benign", sum(l == 0 for l in test_labels), "count")
        led.number("centres_checked", len(centre_hus), "count").number("centres_above_minus_500_hu", dense, "count")
        led.number("centre_hu_median", float(np.median(centre_hus)), "HU")
        led.number("centre_hu_min", float(np.min(centre_hus)), "HU").number("centre_hu_max", float(np.max(centre_hus)), "HU")
        led.note_is("check ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("prepare")
    b.add_argument("--jobs", type=int)
    sub.add_parser("nodules")
    sub.add_parser("check")
    a = ap.parse_args()

    with ledger.run("measurement", "lungx " + a.cmd) as led:
        led.settings_from({"command": a.cmd, "jobs": getattr(a, "jobs", None)})
        if a.cmd == "prepare":
            prepare(a.jobs)
        elif a.cmd == "nodules":
            nodules(led)
        else:
            check(led)


if __name__ == "__main__":
    sys.exit(main())
