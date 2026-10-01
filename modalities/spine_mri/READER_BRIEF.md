# Reader brief: spine MRI, images only

Example case: adult spine MRI (field strength: fill in). Replace the series table and level
mapping for each new case. Studies: fill in (for example, MRI lumbar spine without contrast).
No clinical history is available. **Do NOT open any report, scanned documents, viewer folders
(such as `IHE_PDI` or `INDEX.HTM`), or anything outside `<READER_DIR>`.** Read the images only.

`<READER_DIR>` is the folder `render.py <cd_folder> <READER_DIR>` wrote; the orchestrator fills in
the real path before handing this brief out.

## Where the images are
`<READER_DIR>/png/<SERIES>/<NN>.png`: every slice at its native
matrix size, windowed 0.5-99.7 percentile per series.
Each slice also comes in two more windows made from the full 16-bit data, at the same size:
`<NN>_fluid.png` (fluid and disc: the bright end stretched, so fluid, a hydrated nucleus and a small
bright annular fissure separate) and `<NN>_marrow.png` (bone and marrow: the middle stretched, so
marrow, endplates and Modic-type changes show). **Look at both** before calling a disc's signal, a
bright spot in the annulus, or an endplate normal.
`png/<SERIES>__montage.png`: overview grid, slice index in yellow.
`png/series_meta.json`: pixel spacing, thickness, orientation and positions per slice.

| Series | What | Slices | Pixel (mm) | Thick (mm) |
|---|---|---|---|---|
| (one row per series, filled in for each case from `png/series_meta.json`) | e.g. lumbar sagittal T2 | | | |

## Orientation conventions (check them against each case's DICOM headers)
- **Sagittal**: image left = anterior, image right = posterior, top = superior. T1, T2 and STIR
  usually share slice positions, so the same index is the same plane; confirm it in
  `series_meta.json`. Find the midline slice yourself.
- **Axial**: standard radiological view: image left = patient RIGHT, image right = patient LEFT,
  image top = anterior.
- Map an axial slice to its level by its z position against the mid-sagittal T2
  (`series_meta.json` holds every slice's position). Count the lumbar vertebrae up from the
  sacrum yourself.
- `png/C_axial_localizer.png` and `png/L_axial_localizer.png` draw the axial slice lines on the
  mid-sagittal T2 (approximate).

## Reading rules

- **R1 annular fissure**. A small T2- or STIR-bright focus
  inside the dark outer ring at the back of a disc, seen on two neighbouring sagittal slices or on
  both T2 and STIR, is an annular fissure (high-intensity zone): call it and name the level. Put it
  down to the epidural veins only when it lies behind the posterior longitudinal ligament line,
  outside the annulus, or runs along a vessel over more than one level. Check the fluid window.
- **R2 segmental alignment**. Judge the curve segment
  by segment, not only overall: for each segment from C2-C3 to C7-T1 (and T12-L1 to L5-S1), compare
  the planes of the two neighbouring discs or vertebral back walls. Two or more neighbouring
  segments within about 3 degrees of parallel are segmental straightening, and must be reported
  even when the overall C2-C7 (or L1-S1) angle is normal.

## Zoom tool
With `READER_PNG=<READER_DIR>/png` set:
`python core/reading/crop.py OUT.png x0 y0 x1 y1 scale SERIES:SLICE [SERIES:SLICE ...]`
(add `:fluid` or `:marrow` after a slice for that window, e.g. `L_S03_SAG_T2:6:fluid`)
crops the same pixel box from each listed slice, upsamples by `scale`, lays them side by side and
writes `<READER_DIR>/crops/OUT.png` (prefix OUT with your reader id so files do not collide).
`gridcrop.py` takes the same arguments and adds a 1 mm grid. Keep the output under ~1500 px wide
or it gets downscaled when you view it. Measure distances as pixels x pixel spacing.
