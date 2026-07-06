"""Spot-detection pipeline: smooth -> white_tophat -> threshold -> maxima.

Two entry points:

* :func:`run` operates on a single 2-D plane and returns every intermediate
  stage.  It is used for the live preview while parameters are tuned.

* :func:`run_stack` operates on a whole z-stack.  It detects spots in every
  plane, links them across successive planes (centroids within
  ``link_max_dist`` pixels in XY), keeps only the in-focus plane of each
  linked track (the plane of highest intensity), and measures each spot in
  that plane.  Spot footprints are fixed-radius disks; the local background
  is the median of a concentric ring scaled by the spot's pixel count.

Detection and measurement are limited to user-defined ROIs: the foreground is
zero outside them and, within each ROI, the threshold is taken from that
region's stack histogram.  (With no ROIs the pipeline falls back to a single
global threshold over the whole image.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.filters import threshold_li, threshold_otsu
from skimage.morphology import remove_small_objects

from . import filters

THRESHOLD_METHODS = ("Otsu", "Li", "ilastik")

# An ROI is (row_min, row_max, col_min, col_max, label).
ROI = Tuple[int, int, int, int, str]


@dataclass
class PipelineParams:
    smoothing_method: str = "Median"
    smoothing_size: float = 2.0
    tophat_size: float = 9.0
    threshold_method: str = "Otsu"
    # Thresholding
    min_mask_size: int = 10             # drop foreground objects < this (pixels)
    # Maxima detection
    min_distance: int = 3
    peak_rel_threshold: float = 0.1     # fraction of top-hat max
    measure_radius: int = 3             # spot disk radius (pixels)
    # Local background ring (pixels, measured from the disk edge outward)
    bkg_gap: int = 1                    # gap between spot disk and ring
    bkg_width: int = 3                  # ring thickness
    # Cross-plane linking
    link_max_dist: float = 2.0          # max XY centroid distance to link
    min_link_planes: int = 3            # QC: drop tracks shorter than this
    # Chromatic-aberration offset applied to NON-reference channels (pixels)
    offset_z: int = 0
    offset_y: int = 0
    offset_x: int = 0
    # ilastik threshold support
    ilastik_model_path: Optional[Path] = None
    ilastik_prob_threshold: float = 0.5


@dataclass
class PipelineResult:
    """Single-plane preview result."""
    raw: np.ndarray
    smoothed: np.ndarray
    tophat: np.ndarray
    foreground: np.ndarray              # boolean mask
    centroids: np.ndarray              # (N, 2) row, col
    measurements: pd.DataFrame = field(default_factory=pd.DataFrame)
    roi_thresholds: dict = field(default_factory=dict)


@dataclass
class StackResult:
    """Full-stack detection result (one row per in-focus spot)."""
    centroids: np.ndarray              # (N, 3) z, row, col -- in-focus planes
    measurements: pd.DataFrame
    tophat: np.ndarray                 # (Z, Y, X)
    foreground: np.ndarray             # (Z, Y, X) bool
    plane_shape: Tuple[int, int]
    n_planes: int
    roi_thresholds: dict = field(default_factory=dict)   # label -> threshold


@dataclass
class MultiChannelResult:
    """Per-channel detection results plus a combined, ROI-sorted table.

    Exposes the display fields of the reference channel (``centroids``,
    ``tophat``, ``foreground`` ...) so the same viewers/panels work, while
    ``measurements`` holds every measurable channel's spots with a ``channel``
    column.
    """
    measurements: pd.DataFrame
    per_channel: dict                  # channel name -> StackResult
    centroids: np.ndarray              # reference channel, for annotation
    tophat: np.ndarray
    foreground: np.ndarray
    plane_shape: Tuple[int, int]
    n_planes: int
    roi_thresholds: dict = field(default_factory=dict)   # channel -> {label: t}


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #
def _threshold_value(values: np.ndarray, method: str) -> Optional[float]:
    """Otsu/Li threshold of a 1-D set of values, or None if degenerate."""
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.ptp(finite) == 0:
        return None
    if method == "Otsu":
        return float(threshold_otsu(finite))
    if method == "Li":
        return float(threshold_li(finite))
    raise ValueError(f"Unknown histogram threshold method: {method!r}")


def _filter_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected foreground objects smaller than *min_size* pixels.

    Operates per-plane (2-D connectivity) for a stack so objects are judged
    within each plane, consistent with the 2-D detection/measurement.
    """
    if min_size is None or min_size <= 1:
        return mask
    if mask.ndim == 2:
        return remove_small_objects(mask, min_size=min_size)
    out = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        out[z] = remove_small_objects(mask[z], min_size=min_size)
    return out


def _ilastik_foreground(raw: np.ndarray, params: PipelineParams) -> np.ndarray:
    """Foreground via an ilastik pixel-classification project (.ilp).

    Mirrors ``memQuant.py``: a 2-D plane is fed as a ``DataArray`` with
    ``["y", "x"]`` dims and the foreground channel (index 1) of the
    probability map is thresholded.
    """
    if params.ilastik_model_path is None:
        raise ValueError("No ilastik .ilp model selected.")
    from ilastik.experimental.api import from_project_file  # lazy import
    from xarray import DataArray

    model = from_project_file(Path(params.ilastik_model_path))
    prob = model.predict(DataArray(raw.astype(np.float32), dims=["y", "x"]))
    prob = np.asarray(prob)[:, :, 1]
    return prob > params.ilastik_prob_threshold


def apply_threshold(tophat: np.ndarray, raw: np.ndarray,
                    params: PipelineParams) -> np.ndarray:
    """Boolean foreground mask for a single plane (no ROIs)."""
    if params.threshold_method == "ilastik":
        return _ilastik_foreground(raw, params)
    t = _threshold_value(tophat, params.threshold_method)
    if t is None:
        return np.zeros_like(tophat, dtype=bool)
    return tophat > t


# --------------------------------------------------------------------------- #
# Disk / ring measurement
# --------------------------------------------------------------------------- #
def _disk_ring_masks(shape, r, c, radius, gap, width):
    """Boolean (spot_disk, background_ring) masks centred at (r, c)."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (yy - r) ** 2 + (xx - c) ** 2
    disk = d2 <= radius ** 2
    r_in = radius + gap
    r_out = radius + gap + width
    ring = (d2 > r_in ** 2) & (d2 <= r_out ** 2)
    return disk, ring


def measure_spot(raw: np.ndarray, foreground: np.ndarray, r: int, c: int,
                 params: PipelineParams) -> dict:
    """Intensity stats + local-ring background for one spot.

    Signal pixels are the thresholded foreground pixels within the spot disk;
    the background ring excludes foreground pixels.  Intensities are read from
    the *raw* image.  Background is the ring median times the spot pixel count,
    giving a contribution directly comparable to the integrated spot
    intensity.  (Mirrors memQuant's signal/background logic.)
    """
    disk, ring = _disk_ring_masks(raw.shape, r, c, params.measure_radius,
                                  params.bkg_gap, params.bkg_width)
    signal_mask = disk & foreground
    bkg_mask = ring & ~foreground
    spot_vals = raw[signal_mask]
    ring_vals = raw[bkg_mask]
    n_pixels = int(spot_vals.size)
    bkg_median = float(np.median(ring_vals)) if ring_vals.size else 0.0
    integrated = float(spot_vals.sum())
    background = bkg_median * n_pixels
    return {
        "peak_intensity": float(spot_vals.max()) if n_pixels else 0.0,
        "mean_intensity": float(spot_vals.mean()) if n_pixels else 0.0,
        "integrated_intensity": integrated,
        "n_pixels": n_pixels,
        "bkg_median": bkg_median,
        "background": background,
        "corrected_integrated": integrated - background,
    }


def _disk_peak(raw: np.ndarray, r: int, c: int, radius: int) -> float:
    """Peak raw intensity within a disk -- the metric used to pick in-focus."""
    disk, _ = _disk_ring_masks(raw.shape, r, c, radius, 0, 1)
    vals = raw[disk]
    return float(vals.max()) if vals.size else 0.0


# --------------------------------------------------------------------------- #
# Per-plane detection + cross-plane linking
# --------------------------------------------------------------------------- #
def detect_plane(tophat_plane: np.ndarray, foreground_plane: np.ndarray,
                 params: PipelineParams) -> np.ndarray:
    """Local maxima of the top-hat restricted to the foreground (M, 2)."""
    peak_img = np.where(foreground_plane, tophat_plane, 0.0)
    if np.ptp(peak_img) <= 0:
        return np.empty((0, 2), dtype=int)
    return peak_local_max(
        peak_img,
        min_distance=max(int(params.min_distance), 1),
        threshold_rel=params.peak_rel_threshold,
        labels=foreground_plane.astype(int),
    )


def link_tracks(dets_by_plane: List[List[tuple]], max_dist: float) -> List[list]:
    """Link detections across successive planes into tracks.

    ``dets_by_plane[z]`` is a list of ``(row, col, intensity)``.  A detection
    links to a track only if the track was extended in the immediately
    preceding plane and the XY centroid distance is < ``max_dist``.  Greedy
    nearest-neighbour matching.  Returns a list of tracks, each a list of
    ``(z, row, col, intensity)``.
    """
    tracks: List[list] = []
    active: List[list] = []          # tracks whose last point is in plane z-1
    for z, plane in enumerate(dets_by_plane):
        used = set()
        next_active: List[list] = []
        for tr in active:
            _, lr, lc, _ = tr[-1]
            best_i, best_d = None, max_dist
            for i, (r, c, inten) in enumerate(plane):
                if i in used:
                    continue
                d = float(np.hypot(r - lr, c - lc))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                r, c, inten = plane[best_i]
                tr.append((z, r, c, inten))
                used.add(best_i)
                next_active.append(tr)
            else:
                tracks.append(tr)       # track ends here
        for i, (r, c, inten) in enumerate(plane):
            if i not in used:
                next_active.append([(z, r, c, inten)])
        active = next_active
    tracks.extend(active)
    return tracks


# --------------------------------------------------------------------------- #
# ROI-aware foreground over the whole stack
# --------------------------------------------------------------------------- #
def _clip_roi(roi: ROI, shape) -> ROI:
    h, w = shape
    r0, r1, c0, c1, label = roi
    r0, r1 = max(0, int(min(r0, r1))), min(h, int(max(r0, r1)))
    c0, c1 = max(0, int(min(c0, c1))), min(w, int(max(c0, c1)))
    return r0, r1, c0, c1, label


def global_roi_threshold(tophat: np.ndarray, params: PipelineParams,
                         rois: Optional[List[ROI]]) -> Optional[float]:
    """One threshold from the pixels pooled across *all* ROIs.

    Compiles a single intensity histogram from the top-hat values inside every
    ROI (across all planes) and thresholds it once, so the same "global"
    threshold applies to every ROI.  Pooling more pixels makes Otsu/Li more
    robust.  Returns ``None`` for the ilastik method (no histogram) or when the
    pooled data is degenerate.
    """
    if params.threshold_method == "ilastik":
        return None
    plane_shape = tophat.shape[1:]
    pooled = []
    for roi in (rois or []):
        r0, r1, c0, c1, _ = _clip_roi(roi, plane_shape)
        if r1 <= r0 or c1 <= c0:
            continue
        pooled.append(tophat[:, r0:r1, c0:c1].ravel())
    if not pooled:
        return None
    return _threshold_value(np.concatenate(pooled), params.threshold_method)


def compute_foreground_stack(tophat: np.ndarray, raw_stack: np.ndarray,
                             params: PipelineParams,
                             rois: Optional[List[ROI]] = None,
                             threshold: Optional[float] = None,
                             threshold_rois: Optional[List[ROI]] = None
                             ) -> np.ndarray:
    """(Z, Y, X) boolean foreground.

    Detection is limited to *rois*: the foreground is zero outside them.  A
    single global threshold (pooled from *threshold_rois*, defaulting to all
    *rois*) is applied inside every ROI, giving consistent "global" detection.
    With no ROIs, a global threshold over the whole image is used.
    """
    method = params.threshold_method
    plane_shape = tophat.shape[1:]

    if rois:
        fg = np.zeros_like(tophat, dtype=bool)
        if method == "ilastik":
            ilastik_fg = np.stack([_ilastik_foreground(raw_stack[z], params)
                                   for z in range(raw_stack.shape[0])])
            for roi in rois:
                r0, r1, c0, c1, _ = _clip_roi(roi, plane_shape)
                if r1 > r0 and c1 > c0:
                    fg[:, r0:r1, c0:c1] = ilastik_fg[:, r0:r1, c0:c1]
            return _filter_small(fg, params.min_mask_size)

        gt = threshold
        if gt is None:
            gt = global_roi_threshold(tophat, params, threshold_rois or rois)
        if gt is not None:
            for roi in rois:
                r0, r1, c0, c1, _ = _clip_roi(roi, plane_shape)
                if r1 > r0 and c1 > c0:
                    fg[:, r0:r1, c0:c1] = tophat[:, r0:r1, c0:c1] > gt
        return _filter_small(fg, params.min_mask_size)

    # No ROIs: global threshold over the whole image.
    if method == "ilastik":
        fg = np.stack([_ilastik_foreground(raw_stack[z], params)
                       for z in range(raw_stack.shape[0])])
    else:
        gt = _threshold_value(tophat.ravel(), method)
        fg = (tophat > gt) if gt is not None else np.zeros_like(tophat, bool)
    return _filter_small(fg, params.min_mask_size)


def _assign_region(r: int, c: int, rois: Optional[List[ROI]]) -> str:
    for roi in (rois or []):
        r0, r1, c0, c1, label = roi
        if min(r0, r1) <= r < max(r0, r1) and min(c0, c1) <= c < max(c0, c1):
            return label
    return "global"


def _roi_bounds(label: str, rois: Optional[List[ROI]]):
    """(r0, r1, c0, c1) for *label*, or None for the global region."""
    for roi in (rois or []):
        r0, r1, c0, c1, lab = roi
        if lab == label:
            return min(r0, r1), max(r0, r1), min(c0, c1), max(c0, c1)
    return None


def _touches_boundary(r: int, c: int, radius: int, bounds) -> bool:
    """True if a disk of *radius* at (r, c) reaches/exceeds the ROI bounds."""
    if bounds is None:
        return False
    r0, r1, c0, c1 = bounds
    return (r - radius < r0 or r + radius > r1
            or c - radius < c0 or c + radius > c1)


def _roi_number(label: str) -> int:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return int(digits) if digits else 0


def sort_by_roi(df: pd.DataFrame, extra: Optional[List[str]] = None) -> pd.DataFrame:
    """Sort a measurement table by ROI number (then any *extra* columns)."""
    if df.empty or "region" not in df.columns:
        return df.reset_index(drop=True)
    keys = ["__roi__"] + [c for c in (extra or []) if c in df.columns]
    out = df.assign(__roi__=df["region"].map(_roi_number))
    out = out.sort_values(keys, kind="stable").drop(columns="__roi__")
    return out.reset_index(drop=True)


def min_link_for_zstep(z_step_um: float) -> int:
    """Start-up min Z-linkage from the z-step: ceil(1000 / step_nm).

    ``step_nm`` is the z-step in nanometres, so this is the number of planes
    spanning ~1 micron (a spot dimmer/shorter than that is treated as noise).
    """
    step_nm = float(z_step_um) * 1000.0
    if step_nm <= 0:
        return 1
    return int(max(1, np.ceil(1000.0 / step_nm)))


def next_roi_labels(prior: dict, bounds_list, counter: int):
    """Assign session-global ROI labels, preserving existing ones by bounds.

    ``prior`` maps ``(r0,r1,c0,c1) -> label`` for already-labelled rectangles;
    ``counter`` is the highest ROI index used so far.  New rectangles get
    ``ROI{counter+1}`` (incrementing across files so labels never repeat).
    Returns ``(labelled_list, new_counter)`` where each item is
    ``(r0, r1, c0, c1, label)``.
    """
    labelled = []
    for b in bounds_list:
        key = tuple(int(x) for x in b[:4])
        label = prior.get(key)
        if label is None:
            counter += 1
            label = f"ROI{counter}"
        labelled.append((*key, label))
    return labelled, counter


def tag_filename(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Return a copy of *df* with a leading ``filename`` column."""
    out = df.copy()
    if "filename" not in out.columns:
        out.insert(0, "filename", filename)
    return out


def compile_records(records: dict) -> pd.DataFrame:
    """Concatenate per-file measurement tables into one long-format table."""
    if not records:
        return pd.DataFrame()
    return pd.concat(records.values(), ignore_index=True)


def build_report(table: pd.DataFrame) -> pd.DataFrame:
    """Pair channels per spot and order abnormal ROIs last.

    Pivots the long measurement table into one row per spot with each channel's
    intensities side by side (``<channel>_<metric>``).  Adds ``num_spots`` (the
    number of detected spots in the ROI) and sorts so ROIs with exactly two
    spots come first and any ROI with ``num_spots != 2`` is reported last.
    """
    if table is None or getattr(table, "empty", True):
        return pd.DataFrame()
    id_cols = [c for c in ("filename", "region", "spot_id") if c in table.columns]
    if "channel" not in table.columns or len(id_cols) < 2:
        return table.reset_index(drop=True)

    channels = list(dict.fromkeys(table["channel"]))
    val_cols = [c for c in table.columns if c not in id_cols + ["channel"]]

    wide = table.pivot_table(index=id_cols, columns="channel",
                             values=val_cols, aggfunc="first")
    wide.columns = [f"{ch}_{val}" for val, ch in wide.columns]
    wide = wide.reset_index()

    # Channel-major column order: identity, then each channel's metrics.
    ordered = list(id_cols)
    for ch in channels:
        for val in val_cols:
            col = f"{ch}_{val}"
            if col in wide.columns:
                ordered.append(col)
    wide = wide[[c for c in ordered if c in wide.columns]]

    # Spots per ROI; abnormal (!= 2) reported last.
    grp = ["filename", "region"] if "filename" in wide.columns else ["region"]
    wide["num_spots"] = wide.groupby(grp)["spot_id"].transform("count")
    wide["__abnormal__"] = (wide["num_spots"] != 2).astype(int)
    wide["__roi__"] = wide["region"].map(_roi_number)
    sort_cols = ["__abnormal__"] + \
        (["filename"] if "filename" in wide.columns else []) + \
        ["__roi__", "spot_id"]
    wide = (wide.sort_values(sort_cols, kind="stable")
            .drop(columns=["__abnormal__", "__roi__"]).reset_index(drop=True))
    return wide


# Channels whose names contain any of these tokens are not measured.
_NON_FLUOR_TOKENS = ("brightfield", "bright field", "trans", "phase")


def is_measurable_channel(name: str) -> bool:
    """False for brightfield/transmitted/phase channels."""
    low = str(name).lower()
    return not any(tok in low for tok in _NON_FLUOR_TOKENS)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def run(raw: np.ndarray, params: PipelineParams,
        pixel_size: float = 1.0, pixel_unit: str = "px") -> PipelineResult:
    """Single-plane preview: smooth -> top-hat -> threshold -> maxima."""
    raw = np.asarray(raw, dtype=float)
    smoothed = filters.smooth(raw, params.smoothing_method, params.smoothing_size)
    tophat = filters.white_tophat(smoothed, params.tophat_size)
    foreground = apply_threshold(tophat, raw, params)
    centroids = detect_plane(tophat, foreground, params)
    return PipelineResult(raw=raw, smoothed=smoothed, tophat=tophat,
                          foreground=foreground, centroids=centroids)


def _stack_tophat(stack: np.ndarray, params: PipelineParams) -> np.ndarray:
    """Per-plane smoothing + white top-hat for a (Z, Y, X) stack."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[None]
    out = np.empty_like(stack)
    for z in range(stack.shape[0]):
        out[z] = filters.white_tophat(
            filters.smooth(stack[z], params.smoothing_method,
                           params.smoothing_size), params.tophat_size)
    return out


def run_stack(stack: np.ndarray, params: PipelineParams,
              pixel_size: float = 1.0, pixel_unit: str = "px",
              rois: Optional[List[ROI]] = None,
              threshold_rois: Optional[List[ROI]] = None,
              threshold: Optional[float] = None) -> StackResult:
    """Full-stack detection with cross-plane linking and in-focus measurement.

    Spots are detected within *rois*.  The threshold is *threshold* if given
    (e.g. a session-wide value), otherwise pooled from *threshold_rois*
    (default: all *rois*) for robust global detection.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[None]
    Z = stack.shape[0]
    plane_shape = stack.shape[1:]

    tophat = _stack_tophat(stack, params)

    # One global threshold (explicit, or pooled across all ROIs).
    if rois:
        gt = (threshold if threshold is not None
              else global_roi_threshold(tophat, params, threshold_rois or rois))
    else:
        gt = threshold
    thresholds = {roi[4]: gt for roi in (rois or [])}
    foreground = compute_foreground_stack(tophat, stack, params, rois,
                                          threshold=gt)

    # Detect per plane and tag each detection with a peak intensity.
    dets_by_plane: List[List[tuple]] = []
    for z in range(Z):
        cents = detect_plane(tophat[z], foreground[z], params)
        plane = [(int(r), int(c), _disk_peak(stack[z], int(r), int(c),
                                             params.measure_radius))
                 for r, c in cents]
        dets_by_plane.append(plane)

    tracks = link_tracks(dets_by_plane, params.link_max_dist)

    # QC: require a minimum Z-linkage (capped at the stack depth so single- or
    # few-plane stacks aren't wiped out).
    min_link = min(max(int(params.min_link_planes), 1), Z)

    rows = []
    centroids = []
    sid = 0
    for tr in tracks:
        if len(tr) < min_link:                     # spurious detection
            continue
        z, r, c, _ = max(tr, key=lambda t: t[3])   # in-focus = max intensity
        region = _assign_region(r, c, rois)
        # Drop spots whose disk touches the boundary of their ROI.
        if _touches_boundary(r, c, params.measure_radius,
                             _roi_bounds(region, rois)):
            continue
        stats = measure_spot(stack[z], foreground[z], r, c, params)
        rows.append({
            "spot_id": sid,
            "z_plane": int(z),
            "row_px": int(r),
            "col_px": int(c),
            f"x_{pixel_unit}": c * pixel_size,
            f"y_{pixel_unit}": r * pixel_size,
            **stats,
            "region": region,
            "roi_threshold": thresholds.get(region),
            "n_planes_linked": len(tr),
        })
        centroids.append((z, r, c))
        sid += 1

    columns = ["spot_id", "z_plane", "row_px", "col_px",
               f"x_{pixel_unit}", f"y_{pixel_unit}",
               "peak_intensity", "mean_intensity", "integrated_intensity",
               "n_pixels", "bkg_median", "background", "corrected_integrated",
               "region", "roi_threshold", "n_planes_linked"]
    measurements = sort_by_roi(pd.DataFrame(rows, columns=columns))
    centroids = (np.array(centroids, dtype=float).reshape(-1, 3)
                 if centroids else np.empty((0, 3)))
    return StackResult(centroids=centroids, measurements=measurements,
                       tophat=tophat, foreground=foreground,
                       plane_shape=plane_shape, n_planes=Z,
                       roi_thresholds=thresholds)


def _channel_foreground(stack: np.ndarray, params: PipelineParams,
                        rois: Optional[List[ROI]],
                        threshold_rois: Optional[List[ROI]] = None,
                        threshold: Optional[float] = None):
    """Per-plane top-hat + global-ROI-thresholded foreground for one channel."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[None]
    tophat = _stack_tophat(stack, params)
    if rois:
        gt = (threshold if threshold is not None
              else global_roi_threshold(tophat, params, threshold_rois or rois))
    else:
        gt = threshold
    thr = {roi[4]: gt for roi in (rois or [])}
    fg = compute_foreground_stack(tophat, stack, params, rois, threshold=gt)
    return stack, fg, thr


def run_multichannel(stacks: dict, ref_channel: str, params: PipelineParams,
                     pixel_size: float = 1.0, pixel_unit: str = "px",
                     rois: Optional[List[ROI]] = None,
                     threshold_rois: Optional[List[ROI]] = None,
                     channel_thresholds: Optional[dict] = None
                     ) -> MultiChannelResult:
    """Detect spots in the reference channel only; measure all channels there.

    ``stacks`` maps channel name -> (Z, Y, X) volume.  Brightfield / trans /
    phase channels are skipped.  Spots are detected (with cross-plane linking
    and in-focus selection) only in ``ref_channel``.  Every measurable channel
    is then measured at the reference spot's centroid and in-focus z-plane; for
    non-reference channels a per-axis ``offset`` (``offset_z/y/x``, default 0)
    is added to correct chromatic aberration.  Each channel still goes through
    smoothing, white top-hat and ROI stack-histogram thresholding so the masked
    disk + ring measurement uses that channel's own foreground.
    """
    measurable = {n: s for n, s in stacks.items() if is_measurable_channel(n)}
    if not measurable:
        raise ValueError("No measurable (non-brightfield) channels found.")
    if ref_channel not in measurable:
        ref_channel = next(iter(measurable))

    # Detect in the reference channel. Use a session-wide threshold if given,
    # else pool across all ROIs of this image.
    ref_thresh = channel_thresholds.get(ref_channel) if channel_thresholds else None
    ref_res = run_stack(measurable[ref_channel], params,
                        pixel_size, pixel_unit, rois, threshold_rois,
                        threshold=ref_thresh)
    ref_df = ref_res.measurements.copy()
    ref_df.insert(1, "channel", ref_channel)
    columns = list(ref_df.columns)

    per_channel = {ref_channel: ref_res}
    thresholds = {ref_channel: ref_res.roi_thresholds}
    frames = [ref_df]

    dz, dy, dx = params.offset_z, params.offset_y, params.offset_x
    for name, stack in measurable.items():
        if name == ref_channel:
            continue
        ch_thresh = channel_thresholds.get(name) if channel_thresholds else None
        ch_stack, fg, thr = _channel_foreground(stack, params, rois,
                                                threshold_rois, threshold=ch_thresh)
        Z, H, W = ch_stack.shape
        rows = []
        for _, rs in ref_res.measurements.iterrows():
            z = int(np.clip(int(rs["z_plane"]) + dz, 0, Z - 1))
            r = int(np.clip(int(rs["row_px"]) + dy, 0, H - 1))
            c = int(np.clip(int(rs["col_px"]) + dx, 0, W - 1))
            stats = measure_spot(ch_stack[z], fg[z], r, c, params)
            rows.append({
                "spot_id": rs["spot_id"], "channel": name,
                "z_plane": z, "row_px": r, "col_px": c,
                f"x_{pixel_unit}": c * pixel_size,
                f"y_{pixel_unit}": r * pixel_size,
                **stats,
                "region": rs["region"], "roi_threshold": thr.get(rs["region"]),
                "n_planes_linked": rs["n_planes_linked"],
            })
        frames.append(pd.DataFrame(rows, columns=columns))
        per_channel[name] = thr
        thresholds[name] = thr

    combined = sort_by_roi(pd.concat(frames, ignore_index=True),
                           extra=["channel", "spot_id"])
    return MultiChannelResult(
        measurements=combined, per_channel=per_channel,
        centroids=ref_res.centroids, tophat=ref_res.tophat,
        foreground=ref_res.foreground, plane_shape=ref_res.plane_shape,
        n_planes=ref_res.n_planes, roi_thresholds=thresholds)


def compute_session_thresholds(files_data: list, params: PipelineParams) -> dict:
    """One threshold per channel, pooled over *all* ROIs of *all* files.

    ``files_data`` is a list of dicts with ``"stacks"`` (channel name ->
    (Z, Y, X)) and ``"rois"`` (list of ROI tuples).  For each measurable
    channel, the top-hat pixels inside every ROI of every file are compiled
    into one histogram and thresholded once -> ``{channel: threshold}``.  Empty
    for the ilastik method (no histogram).
    """
    if params.threshold_method == "ilastik":
        return {}
    from collections import defaultdict
    pooled = defaultdict(list)
    for f in files_data:
        for ch, stack in f["stacks"].items():
            if not is_measurable_channel(ch):
                continue
            tophat = _stack_tophat(stack, params)
            ps = tophat.shape[1:]
            for roi in f.get("rois", []):
                r0, r1, c0, c1, _ = _clip_roi(roi, ps)
                if r1 > r0 and c1 > c0:
                    pooled[ch].append(tophat[:, r0:r1, c0:c1].ravel())
    return {ch: _threshold_value(np.concatenate(v), params.threshold_method)
            for ch, v in pooled.items() if v}


def run_session(files_data: list, params: PipelineParams):
    """Re-run detection/measurement over the whole session with one global
    per-channel threshold pooled across all ROIs of all files.

    Returns ``(combined_table, channel_thresholds)``.  Each entry of
    ``files_data`` needs ``filename``, ``stacks``, ``ref_channel``, ``rois``
    and optionally ``pixel_size`` / ``pixel_unit``.
    """
    thresholds = compute_session_thresholds(files_data, params)
    frames = []
    for f in files_data:
        if not f.get("rois"):
            continue
        res = run_multichannel(
            f["stacks"], f["ref_channel"], params,
            pixel_size=f.get("pixel_size", 1.0),
            pixel_unit=f.get("pixel_unit", "px"),
            rois=f["rois"], channel_thresholds=thresholds)
        frames.append(tag_filename(res.measurements, f["filename"]))
    if not frames:
        return pd.DataFrame(), thresholds
    combined = pd.concat(frames, ignore_index=True)
    combined["__roi__"] = combined["region"].map(_roi_number)
    keys = [k for k in ["filename", "__roi__", "channel", "spot_id"]
            if k in combined.columns]
    combined = combined.sort_values(keys).drop(columns="__roi__")
    return combined.reset_index(drop=True), thresholds


def run_preview(stack: np.ndarray, z: int, params: PipelineParams,
                rois: Optional[List[ROI]] = None,
                threshold_rois: Optional[List[ROI]] = None) -> PipelineResult:
    """ROI-limited single-plane preview for the displayed Z-plane.

    Spots are found only inside the ROIs, using one global threshold pooled
    across all *threshold_rois* (default: all *rois*) stack histograms, matching
    :func:`run_stack`.  The current plane is displayed; the foreground and
    detected maxima are confined to the ROIs.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[None]
    Z = stack.shape[0]
    z = max(0, min(int(z), Z - 1))
    plane = stack[z]

    # Current-plane top-hat for display.
    th_full = filters.white_tophat(
        filters.smooth(plane, params.smoothing_method, params.smoothing_size),
        params.tophat_size)

    fg = np.zeros(plane.shape, dtype=bool)
    method = params.threshold_method

    if method == "ilastik":
        fgi = _ilastik_foreground(plane, params)
        for roi in (rois or []):
            r0, r1, c0, c1, _ = _clip_roi(roi, plane.shape)
            if r1 > r0 and c1 > c0:
                fg[r0:r1, c0:c1] = fgi[r0:r1, c0:c1]
        thresholds = {roi[4]: None for roi in (rois or [])}
    else:
        # One global threshold from all ROIs' stack histograms (pooled).
        pooled = []
        for roi in (threshold_rois or rois or []):
            r0, r1, c0, c1, _ = _clip_roi(roi, plane.shape)
            if r1 <= r0 or c1 <= c0:
                continue
            sub = stack[:, r0:r1, c0:c1]
            pooled.append(np.stack([
                filters.white_tophat(
                    filters.smooth(sub[k], params.smoothing_method,
                                   params.smoothing_size), params.tophat_size)
                for k in range(Z)]).ravel())
        gt = _threshold_value(np.concatenate(pooled), method) if pooled else None
        thresholds = {roi[4]: gt for roi in (rois or [])}
        if gt is not None:
            for roi in (rois or []):
                r0, r1, c0, c1, _ = _clip_roi(roi, plane.shape)
                if r1 > r0 and c1 > c0:
                    fg[r0:r1, c0:c1] = th_full[r0:r1, c0:c1] > gt

    fg = _filter_small(fg, params.min_mask_size)
    centroids = detect_plane(th_full, fg, params)
    return PipelineResult(raw=plane, smoothed=th_full, tophat=th_full,
                          foreground=fg, centroids=centroids,
                          roi_thresholds=thresholds)
