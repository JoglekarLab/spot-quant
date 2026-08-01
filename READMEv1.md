# Spot Quant — interactive fluorescent spot quantification

A modular napari GUI for detecting and measuring fluorescent spots in
microscopy images (tif / tiff / nd2).

## Run

```bash
conda activate img-env
python -m spot_quant
```

The GUI is a single dock widget with two tabs: **Files & detection** (file IO
and the detection controls together) and **Measurement**.

## Panels

**File IO**
- *Select folder…* lists every `.tif`, `.tiff` and `.nd2` file in a folder.
- Double-click a file to open it. The transmitted / brightfield / phase
  channel is shown in grayscale as the **bottom** layer; fluorescence channels
  sit on top as coloured additive layers. The **reference** channel's contrast
  is set to [min, 0.7·max] and other fluorescence channels to [min, 0.8·max].
  (The white top-hat result is no longer displayed as a layer.)
- *Edit image metadata…* reads channel names, pixel size, magnification,
  time step and **z-step** from the file where available and lets you edit them
  (defaults are used when a value is missing).
- If a z-stack's file has no z-step / time metadata, opening it **prompts** for
  the z-step (in nanometres) or time interval so downstream defaults are set
  correctly.

**Spot detection** (same tab as File IO)
Each algorithm stage is its own control group. Global detection is disabled:
preview spots appear **only inside the ROIs** once they are drawn. Tuning a
control updates the live ROI-limited single-plane preview, which draws the
**outline of the thresholded masks** (a Labels contour) inside the ROIs. The
**Detect spots** button then runs the full-stack pipeline; it hides the
preview outlines and shows only the finalised detected spots (points,
colour-coded by region). Changing any parameter or ROI returns to preview
mode.

Each ROI's threshold is computed from that ROI's **stack histogram** (top-hat
pooled over all planes within the ROI). The calculated thresholds are shown in
the status line, recorded per spot in a `roi_threshold` column, and exposed on
the result.

1. **Smoothing** — Gaussian, Median, Kuwahara, or Gaussian low-pass, with an
   editable filter size.
2. **White top-hat** — editable structuring-element size; suppresses
   background and keeps spot-sized features.
3. **Thresholding** — Otsu, Li, or *ilastik* (browse for an `.ilp` pixel
   classification project at runtime; the foreground-probability map is
   thresholded as in `memQuant.py`). Thresholded foreground objects smaller
   than **Min mask size** (default 10 px, per plane) are removed.
3b. **Detection ROIs** *(required)* — draw one or more rectangles. Detection
   and measurement happen **only inside these ROIs**, and spots outside every
   ROI are ignored. For robust "global" detection, a single threshold is
   computed from one intensity histogram pooled across **all** ROIs in the
   session (per channel), then applied to every ROI. Spots are tagged with the
   ROI they fall in (`ROI1`, `ROI2`, …). Detect does nothing until at least one
   ROI is drawn.
4. **Maxima detection & linking** — minimum distance and relative threshold
   control the per-plane maxima search; the link distance sets how far (in XY
   pixels) a spot may move between successive planes and still be treated as
   the same spot. **Min Z-linkage** is a QC filter: spots linked across fewer
   than this many planes are treated as spurious and not quantified (capped at
   the stack depth). Its start-up value is derived from the z-step —
   `ceil(1000 / z_step_nm)`, i.e. the number of planes spanning ~1 µm — and can
   be overridden.
5. **Measurement** — each spot is a fixed-radius disk. Signal pixels are the
   thresholded foreground pixels within the disk; the background ring (gap +
   width beyond the disk) excludes foreground pixels. Intensities are read from
   the raw image and `corrected_integrated = integrated − background`
   (`background = ring_median × spot_pixels`).

### Multi-channel measurement

Spots are detected **only in the reference channel** (the one selected in the
Channel box). Every other measurable channel (brightfield / trans / phase are
skipped) is then measured at the reference spot's centroid and in-focus
z-plane — no independent re-detection — after going through its own
smoothing → top-hat → ROI-histogram threshold so the masked disk + ring use
that channel's foreground. A per-axis **chromatic offset** (Δz, Δy, Δx;
default 0) is added for non-reference channels to correct chromatic
aberration. The combined table carries a `channel` column; spots keep the same
`spot_id` across channels.

### Incremental ROIs

If you draw more ROIs after detecting, the next Detect analyzes **only the new
ROIs** — already-analyzed ROIs are skipped and their results stay on screen and
in the table. **Deleting an ROI removes its measurements** from the stored data
(and its outline/label from the view). Clearing the ROIs (or opening another
file) resets this. ROI labels are stable across deletions.

### Boundary spots & sorting

Spots whose disk touches the boundary of their ROI are dropped. The final
measurement table is sorted by ROI number (then channel).

When spots are detected, each spot's **measurement region** (the disk) is
outlined on its in-focus plane (coloured by ROI number); each ROI's label is
drawn once in a corner of the ROI rectangle rather than on every spot.

### Multi-file session & export

Every Detect compiles that file's spots into a session table, tagged with the
`filename` and the in-focus depth `z_um` (`z_plane × z-step`). Analysing more
files (each with its own ROIs) keeps appending; re-detecting a file replaces
its rows. The Measurement tab shows the compiled long-format table and exports
all of it to a single CSV/XLSX — one tidy row per spot per channel, easy to
re-import. *Clear session* resets the accumulator.

### Session-wide threshold at export

ROIs are remembered per file (and restored when you re-open a file). When you
**export/save**, the whole session is re-run: one intensity histogram is
compiled from the top-hat pixels of **every ROI in every file** (per channel),
a single global threshold is computed from it, and detection/measurement are
recomputed for all files with that threshold before writing. This guarantees
every ROI in the session is included and thresholded consistently. The typical
flow is: open a file → draw ROIs → open the next file → draw ROIs → … →
export.

### In-focus selection

Spots are detected in every plane, then linked across successive planes when
their centroids lie within the link distance. Each linked track keeps only its
**in-focus** plane — the plane of highest intensity — and the spot is measured
and annotated only there. Detected spots appear in a `spots` points layer at
their `(z, y, x)` position (so each shows only on its in-focus plane),
colour-coded by region.

Per-spot measurements (in-focus z-plane, centroid in pixels and physical
units, peak / mean / integrated intensity, ring background, region, planes
linked) plus spot count and density are shown on the Measurement tab and
export to CSV or XLSX.

## Two-spot line-scan analysis (`scan.py`)

For ROIs with exactly two detected reference-channel spots:

- `integrated_intensity_scan(image, p1, p2)` rotates the ROI so the line
  through the two centroids is horizontal (by sampling along the line with
  bilinear interpolation) and sums intensity in a ±2-row band centred on that
  line, returning the `integrated_intensity_scan` vector. ROIs whose spot count
  isn't 2 are excluded (`roi_scans`).
- `classify_lengths(lengths)` labels ROIs **data-drivenly from the length
  distribution** — no hard-coded limits. It clusters the pooled scan lengths
  with 1-D k-means and picks the number of groups by the best silhouette (a
  single group if no real separation), numbering `group1..groupK` by increasing
  length. Every scan gets a label. (On the example session this yields two
  groups — short and long — split near ~2.7 µm.)
- `normalize_group` / `normalize_groups` resample each group's scans (linear /
  1-D bilinear interpolation) to the group's **mean pixel count**, so the
  normalized coordinate corresponds to 1 pixel of the mean-length scan.

The line connecting the centroids is **elongated by 5 px on either side** and
the scan is taken on the reference channel's **max-projection**.

The scan always starts at the **brighter** of the two spots.

**Export.** A *line-scan* checkbox on the Measurement tab enables the analysis
at export time. When checked, a separate `<name>_linescans.xlsx` is written
alongside the measurements, with labelled sheets: `raw_intensity` (raw scan
vectors), `corrected_intensity` (raw minus the ROI's ring background × band
rows), and one normalized sheet **per group** (`normalized_group1`,
`normalized_group2`, …) — per-group resampled scans with a normalized spatial
coordinate in mean pixels and fractional intensity summing to 1. A **Summary**
sheet reports, per group, the number of scans (`n_scans`) and the mean and
standard deviation of the fractional-intensity profile at each normalized pixel
(an average curve ± std). Only ROIs with exactly two reference-channel spots are
included.

**Report layout.** The exported measurement table is *paired by channel*: one
row per spot with each channel's intensities side by side
(`<channel>_<metric>`). A `num_spots` column gives the ROI's detected-spot
count, and ROIs with `num_spots != 2` are reported **last**.

## Module layout

| File | Responsibility |
|------|----------------|
| `app.py` | builds the viewer and the tabbed dock widget |
| `state.py` | shared `AppState` (image, metadata, results, signals) |
| `io_panel.py` | folder/file list, image loading, metadata dialog |
| `detection_panel.py` | grouped pipeline controls, live update |
| `measurement_panel.py` | per-spot table, count/density, CSV/XLSX export |
| `pipeline.py` | smooth → top-hat → threshold → maxima → measure |
| `filters.py` | smoothing + top-hat implementations |
| `metadata.py` | nd2/tif loading and metadata extraction |

## Dependencies

`napari`, `scikit-image`, `scipy`, `numpy`, `pandas`, `nd2`, `tifffile`,
and (for the ilastik threshold) `ilastik` + `xarray`.
