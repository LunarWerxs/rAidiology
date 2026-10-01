# SuprSkan pipeline

From a patient MRI disc (a folder with a `DICOMDIR`) to the data the web app shows.

```
DICOM  ->  to_nifti.py      one NIfTI volume per series (pixels + geometry, no patient tags)
       ->  TotalSpineSeg    vertebrae C1..L5 + sacrum, discs, spinal cord, spinal canal (nnU-Net, GPU)
       ->  metrics.py       per level: disc height, disc-to-CSF signal (hydration), canal and cord width
       ->  build_meshes.py  smooth .glb mesh per structure + anchor points (app frame, mm)
       ->  build_stacks.py  de-identified JPEG slice stacks, plus "AI outlines" overlays
       ->  geometry.py      per-slice DICOM geometry (kept in work/, never published)
reading (AI readers + skeptics + human-checked synthesis) -> reading.json
       ->  make_report.py   report.json: anchors resolved to 3D points, image marks placed by geometry
```

The reading step is not a runnable tool: AI readers (an agent orchestrating them, or a person)
follow `reading/READER_BRIEF.md` and write `reading.json` in the format `make_report.py` reads
(its docstring, and `site/data/CONTRACT.md` for what reaches the app). Run everything else:

```bash
python pipeline/run_pipeline.py <cd_folder> site
python pipeline/geometry.py <cd_folder> site work/geometry.json
python pipeline/make_report.py pipeline/reading.json site work/geometry.json
```

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

`reading/` holds what the AI readers used: `render.py <cd_folder> <reader_dir>` (every series to
PNG slices and montages), `crop.py` / `gridcrop.py` (zoom the same box across slices/sequences,
reading slices from `$READER_PNG`) and `READER_BRIEF.md` (orientation
conventions and level mapping, which the readers must be given; without them they mix up left and
right on axial images). The panel is two independent readers per region with different strategies
(sagittal-first, axial-first), one "everything else" reader, a reconciler, a missed-findings hunter,
and three skeptics per finding (is it real, is the level right, is the grade right). The final
synthesis into `reading.json` is checked against the images and the measurements.

Each slice is also rendered in a fluid/disc window and a bone/marrow window from the 16-bit data
(`reading/windows.py`); readers are told to look at both.

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
