# Spine MRI

rAidiology's first scan type.

From a patient MRI disc (a folder with a `DICOMDIR`) to the data the web app shows.

```
DICOM  ->  core/dicom/to_nifti.py   one NIfTI volume per series (pixels + geometry, no patient tags)
       ->  TotalSpineSeg (tss.py)   vertebrae C1..L5 + sacrum, discs, spinal cord, spinal canal (nnU-Net, GPU)
       ->  metrics.py               per level: disc height, disc-to-CSF signal (hydration), canal and cord width
       ->  build_meshes.py          smooth .glb mesh per structure + anchor points (app frame, mm)
       ->  build_stacks.py          de-identified JPEG slice stacks, plus "AI outlines" overlays
       ->  core/dicom/geometry.py   per-slice DICOM geometry (kept in work/, never published)
reading (AI readers + skeptics + human-checked synthesis) -> cases/<patient>/reading.json
       ->  specialist.py            the shipped disc grader's grades (only if it passed the ship rule)
       ->  sct/cord_compression.py  cervical cord against healthy adults (only where the reading reports narrowing)
       ->  make_report.py           report.json: anchors resolved to 3D points, image marks placed by geometry
```

The reading step is not a runnable tool: AI readers (an agent orchestrating them, or a person)
follow `READER_BRIEF.md` (this folder) and write `reading.json` in the format `make_report.py` reads
(its docstring, and `site/data/CONTRACT.md` for what reaches the app). Run everything else, from the
repository root:

```bash
python modalities/spine_mri/run_pipeline.py <cd_folder> site
python core/dicom/geometry.py <cd_folder> site work/geometry.json
python modalities/spine_mri/specialist.py work
python modalities/spine_mri/sct/cord_compression.py cases/<patient>/reading.json work --age <years> --sex <M|F>
python modalities/spine_mri/make_report.py cases/<patient>/reading.json site work/geometry.json
```

A patient's own material (the reading, the comparison with the radiologist's report, mistake-log
cases) lives in `cases/<patient>/` at the repository root, never next to the code.

## Setup (Windows, NVIDIA GPU)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install totalspineseg nnunetv2 "kornia<0.8" SimpleITK nibabel pydicom scikit-image trimesh fast-simplification scipy pillow
```

`kornia<0.8` matters: TotalSpineSeg's augmentation package imports `kornia.core.Tensor`, which
kornia 0.8 removed, and the error only shows when the model loads.
TotalSpineSeg downloads its weights from its GitHub release on first run (about 300 MB).

## The reading step

The shared image tools in `core/reading/` are what the AI readers used: `render.py <cd_folder> <reader_dir>` (every series to
PNG slices and montages), `crop.py` / `gridcrop.py` (zoom the same box across slices/sequences,
reading slices from `$READER_PNG`) and `READER_BRIEF.md` (orientation
conventions and level mapping, which the readers must be given; without them they mix up left and
right on axial images). The panel is two independent readers per region with different strategies
(sagittal-first, axial-first), one "everything else" reader, a reconciler, a missed-findings hunter,
and three skeptics per finding (is it real, is the level right, is the grade right). The final
synthesis into `reading.json` is checked against the images and the measurements.

Each slice is also rendered in a fluid/disc window and a bone/marrow window from the 16-bit data
(`core/reading/windows.py`); readers are told to look at both.

## Scorecard and specialist models

`eval/` measures every reading method against expert grades on the public SPIDER dataset, trains
the disc grader and decides what ships: see [eval/README.md](eval/README.md). The cervical cord
comparison with healthy adults uses Spinal Cord Toolbox (`sct/install_sct.ps1`,
`sct/cord_compression.py`).

## Measurement caveats

Everything is measured on the AI segmentation of 3 to 5 mm thick sagittal slices: +-1 pixel is
+-0.4 to 0.55 mm. The canal label is the dural sac (fluid + cord), not the bony canal. Disc signal
is a ratio to spinal fluid on the same slices, so it is comparable between levels of one scan, not
between scanners.

## Optional: SpineReport (independent 3D morphometrics)

```bash
.venv/Scripts/python -m pip install "git+https://github.com/ivadomed/SpineReport.git"
totalspineseg <in> <tss_iso> --iso   # inputs named sub-<id>_T2w.nii.gz
spinereport -t <tss_iso> -c <tss_iso> -o <reports>
```

The pip release of TotalSpineSeg lacks `totalspineseg/resources/labels_maps/*.json`, which SpineReport
reads; copy that folder from the TotalSpineSeg GitHub repo into the installed package first.
Its disc metrics work on any spine region; its canal metrics follow the cord, so below the conus
(the lower lumbar spine) they come out as -1 and cannot be used.
