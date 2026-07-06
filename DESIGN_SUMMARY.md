# Spot Quant — Design Summary

A modular **napari** GUI for detecting and measuring fluorescent spots in
microscopy z-stacks (`.tif` / `.tiff` / `.nd2`), with a multi-file / multi-ROI
workflow and a session-wide global threshold computed at export.

Environment: conda `img-env`; run with `python -m spot_quant`.

---

## 1. Module map

| Module | Responsibility |
|--------|----------------|
| `app.py` | Builds the napari viewer + one tabbed dock widget (Files & detection / Measurement). |
| `state.py` | `AppState`: shared image, metadata, per-file ROIs + reference channel, accumulated session records, latest params; Qt signals; `recompute_session()`. |
| `metadata.py` | `ImageMeta` + `load_image()` for nd2/tif; reads channel names, pixel size, magnification, time step, z-step (+ `*_known` flags). |
| `filters.py` | Smoothing (Gaussian, Median, Kuwahara, Gaussian low-pass) + white top-hat. |
| `pipeline.py` | Detection/measurement core (pure, testable): per-plane detect, cross-plane linking, in-focus selection, disk+ring measurement, ROI/global thresholding, multichannel, whole-session run, paired report. |
| `scan.py` | Two-spot line-scan analysis: rotated ±2-row integrated scan, length classification, per-group normalization, export sheets. |
| `io_panel.py` | Folder/file list, image loading + display rules, metadata dialogs. |
| `detection_panel.py` | Grouped algorithm controls, live ROI preview, Detect, ROI persistence, committed-spot display. |
| `measurement_panel.py` | Compiled session table, export (paired report + optional line-scans), Clear session. |

Pure logic lives in `pipeline.py` / `scan.py` / `filters.py` / `metadata.py`
(no Qt) so it is unit-testable headless. GUI panels and `state.py` import Qt.

---

## 2. Detection & measurement pipeline (per file)

1. **Smooth** (chosen method + size) → **white top-hat** (size), per plane.
2. **Threshold** inside ROIs only (global detection disabled outside ROIs).
   Otsu / Li from a histogram, or ilastik `.ilp` foreground probability.
   Objects smaller than **Min mask size** (default 10 px) are removed per plane.
3. **Detect** local maxima per plane (min distance, relative threshold).
4. **Link** across successive planes (centroids within *link distance*, XY).
5. **QC**: drop tracks shorter than **Min Z-linkage** (start-up value =
   `ceil(1000 / z_step_nm)` ≈ planes spanning 1 µm; capped at stack depth).
6. **In-focus plane** = the track's brightest plane; the spot is measured and
   annotated only there.
7. **Measure** on a fixed-radius disk (thresholded foreground pixels within the
   disk = signal; a concentric ring, excluding foreground, = background).
   `background = ring_median × spot_pixels`;
   `corrected_integrated = integrated − background`. Intensities from raw image.
8. Drop spots whose disk touches the ROI boundary. Sort by ROI number.

### Multichannel (reference-anchored)
Spots are detected **only in the reference channel**. Every other measurable
channel (brightfield / trans / phase are skipped by name) is measured at the
reference spot's centroid and in-focus z, plus a per-axis **chromatic offset**
(Δz, Δy, Δx; default 0). Spots keep the same `spot_id` across channels.

---

## 3. Session workflow & thresholding

- **Per file**: open → (prompt for z-step in nm / time interval if missing) →
  draw ROIs → they are remembered per file and restored on re-open. The
  reference channel selection persists across file opens.
- **Global threshold at export**: `run_session` reloads every file with saved
  ROIs, compiles **one histogram per channel from the top-hat pixels of every
  ROI of every file**, computes a single global threshold per channel, and
  recomputes all files with it (`compute_session_thresholds` + threshold
  overrides threaded through `run_stack` / `_channel_foreground` /
  `run_multichannel`).
- **Incremental ROIs**: re-detecting analyses only *new* ROIs; deleting an ROI
  removes its stored records (labels are stable, storage keyed by region).

---

## 4. Display conventions (on open)

- Transmitted / brightfield / phase channel → grayscale, **bottom** layer.
- Fluorescence channels on top, additive; **reference** channel contrast
  `[min, 0.7·max]`, others `[min, 0.8·max]`.
- White top-hat is **not** shown as a layer.
- Preview outlines the thresholded masks inside un-analyzed ROIs; Detect shows
  each spot's measurement-region **disk outline** on its in-focus plane
  (colour-coded by ROI); each ROI label sits in a corner of its rectangle.

### GUI layout notes
- Smoothing / white-top-hat / thresholding controls live in a **pop-up**
  ("Spot detection controls…" button); default smoothing is **Median, size 2**.
- The **chromatic offset** (Δz, Δy, Δx) is edited in the **Edit-metadata**
  dialog (stored on `state`, session-level).
- The reference channel selection **persists** across image opens.
- **ROI labels are session-global** (increment across files, never repeat);
  stored per file as `(r0,r1,c0,c1,label)` and reused on re-open.

---

## 5. Reports & exports

- **Measurements** export (`build_report`): *paired by channel* — one row per
  spot with `<channel>_<metric>` columns; a `num_spots` column; ROIs with
  `num_spots != 2` reported **last**.
- **Line scans** (checkbox → `<name>_linescans.xlsx`): for two-spot ROIs only.
  The connecting line is elongated by **5 px** each side and scanned on the
  reference **max-projection**, summing ±2 rows; the scan starts at the
  **brighter** spot. Sheets: `raw_intensity`, `corrected_intensity`
  (raw − band_rows × ring background), and `normalized_group1` /
  `normalized_group2` (resampled to the group's mean pixel count; fractional
  intensity summing to 1).
- **Length groups**: assigned data-drivenly from the length distribution
  (`classify_lengths`: 1-D k-means + silhouette k-selection); no hard-coded
  limits. Groups numbered by increasing length; every scan is labelled.

---

## 6. Testing

~79 headless unit tests under `tests/` (see `tests/README.md`), covering
filters, the full stack pipeline, linking/in-focus/QC, ROI & session
thresholds, reference-anchoring + offset, session compilation, the paired
report, and the line-scan module. GUI/IO paths (napari layers, dialogs, file
reload) are compile-verified only — confirm those in `img-env`.

`tests/conftest.py` loads pure modules directly (no Qt/napari), so the suite
runs without the GUI stack.

---

## 7. Known gotchas

- **Google-Drive sync reverts**: the project folder is cloud-synced;
  `pipeline._filter_small` has reverted from `min_size=` to `max_size=`
  (an invalid kwarg for `skimage.remove_small_objects`) more than once after a
  sync conflict. If detection suddenly raises
  `TypeError: ... unexpected keyword argument 'max_size'`, re-apply
  `min_size=` and re-run the suite.
- **ilastik + ROI histogram**: ilastik has no histogram, so ROI/global
  histogram thresholds don't apply; its foreground is the probability map
  masked to the ROI.
- **±2 rows** = 5 rows total (offsets −2…+2).
