# Tests

Unit tests for the pure-logic modules of `spot_quant` (filters, pipeline) and
a regression test for the file-switch crash guard. These don't require napari
or qtpy — `conftest.py` loads the modules directly, so they run in a headless
environment.

## Run

```bash
conda activate img-env
cd /path/to/fluorescent-spot-measurement
pytest tests/ -v
```

## What's covered

- `test_filters.py` — every smoothing method preserves shape and finiteness;
  Kuwahara is edge-preserving; white top-hat suppresses flat background and
  keeps small bright features; unknown method raises.
- `test_pipeline.py` — single-plane *preview*: Otsu and Li recover three
  implanted spots at the right locations; intermediates
  (raw/smoothed/tophat/foreground) are returned; flat image yields no spots;
  ilastik without a model raises.
- `test_stack.py` — full-stack pipeline: cross-plane linking collapses a spot
  spanning several planes to one row; the kept plane is the brightest
  (in-focus) one; 3-D centroids; measurement columns; the linking primitive
  (`link_tracks`) chains consecutive planes and breaks on large gaps; the
  disk + concentric-ring background math (`background = ring_median ×
  n_pixels`, `corrected = integrated − background`); ROI region assignment;
  2-D input promoted to a single plane.
- `test_preview.py` — ROI-limited preview: nothing is detected without ROIs
  (global detection disabled); foreground/spots stay inside the ROI; per-ROI
  thresholds are recorded and come from the ROI's stack histogram (a threshold
  exists even on an out-of-focus plane); thresholded masks below the
  min-mask-size are removed (preview and stack).
- `test_multichannel.py` — brightfield/trans/phase channels are excluded;
  spots are detected only in the reference channel and other channels are
  measured at those centroids (a spot-free channel still gets one row per
  reference spot); the chromatic offset shifts non-reference measurement
  positions (default 0 keeps them); a `channel` column; ROI-sorted; an
  all-brightfield input raises.
- `test_session.py` — multi-file compilation: `tag_filename` adds the
  `filename` column (idempotent), `compile_records` concatenates all files
  into one long-format table; `ImageMeta` carries the new `z_step` field.
- `test_scan.py` — two-spot line-scan analysis: line length + the data-driven
  `classify_lengths` (finds 2 or 3 clusters, one group for a single blob, empty
  input); the ±2-row integrated scan
  (length matches the line, sums 5 rows on a uniform image, rotation-invariant,
  peaks on a ridge, requires distinct points); `roi_scans` excludes ROIs
  without exactly 2 spots; per-group normalization resamples to the mean pixel
  count via linear interpolation.
- `test_roi_labels.py` — `next_roi_labels` gives session-global ROI labels that
  increment across files (never repeat) and preserves existing labels by bounds.
- `test_report.py` — paired-channel report: `build_report` pivots to one row
  per spot with side-by-side `<channel>_<metric>` columns, adds `num_spots`,
  and orders ROIs with != 2 spots last.
- `test_linescan_export.py` — measurement-driven line scans (now also: both
  group1 and group2 get their own `normalized_<group>` sheet; the scan starts
  at the brighter spot): `linescans_from_
  measurements` uses the reference max-projection, keeps only two-spot ROIs,
  background-corrects (raw − band_rows × ring background); `build_linescan_
  sheets` emits the three labelled sheets (raw / corrected / normalized) with
  per-scan fractional intensity summing to 1 and a shared group length.
- `test_session_threshold.py` — session-wide thresholding: `compute_session_
  thresholds` pools top-hat pixels across every ROI of every file per channel
  (brightfield excluded); `run_session` applies one threshold per channel to
  all files, sorts by filename, keeps reference-anchored measurement, and
  returns empty without ROIs.
- `test_global_threshold.py` — pooled global thresholding: the threshold is
  Otsu over pixels concatenated across all ROI boxes; every ROI shares one
  threshold; ilastik pools to None; `threshold_rois` controls the pool.
- `test_qc.py` — Z-linkage QC: a 2-plane spot is dropped at the default
  min-linkage of 3 but kept when lowered; the threshold is capped at the stack
  depth so 2-D input isn't wiped; all kept spots meet the minimum.
- `test_load_guard.py` — regression for the napari crash on switching files:
  the `loading` / `_running` guards make `run()` a no-op during the layer
  swap so it never mutates the layer list re-entrantly.
