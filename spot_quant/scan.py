"""Line-scan analysis for two-spot ROIs.

Given the two detected reference-channel spots of an ROI, these functions:

* rotate the ROI so the line through the two centroids is horizontal and sum
  the intensity in a +/-2-row band centred on that line
  (:func:`integrated_intensity_scan`);
* label ROIs by clustering the line-length *distribution*
  (:func:`classify_lengths`);
* normalise each group's scans onto a common length by resampling
  (:func:`normalize_group`, :func:`normalize_groups`).

The scan is computed by sampling the image along the connecting line with
bilinear interpolation, which is equivalent to rotating the ROI so the line is
horizontal and then summing rows -- but without resampling the whole image.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

# Group labels are assigned data-drivenly from the length distribution
# (see :func:`classify_lengths`); no hard-coded length limits.
MAX_LENGTH_GROUPS = 4        # most groups the auto-classifier will consider
MIN_GROUP_COUNT = 2          # a group must contain at least this many scans


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def line_length_px(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean distance between two (row, col) points, in pixels."""
    (r1, c1), (r2, c2) = p1, p2
    return float(np.hypot(r2 - r1, c2 - c1))


def line_length_um(p1, p2, pixel_size: float) -> float:
    """Length of the connecting line in microns."""
    return line_length_px(p1, p2) * float(pixel_size)


# --------------------------------------------------------------------------- #
# (1) Rotated +/- 2-row integrated intensity scan
# --------------------------------------------------------------------------- #
def max_projection(stack: np.ndarray) -> np.ndarray:
    """Maximum-intensity projection over z for a (Z, Y, X) stack (or 2-D)."""
    stack = np.asarray(stack, dtype=float)
    return stack.max(axis=0) if stack.ndim == 3 else stack


def integrated_intensity_scan(image: np.ndarray, p1: Sequence[float],
                              p2: Sequence[float], half_width: int = 2,
                              extend: float = 5.0, order: int = 1) -> np.ndarray:
    """Integrated intensity along the line through *p1* and *p2*.

    The line is *elongated by ``extend`` pixels on either side* of the two
    centroids, then sampled at 1-pixel steps; at each step the image is summed
    over a band of ``2*half_width + 1`` rows perpendicular to the line (i.e.
    +/-2 rows by default) using bilinear interpolation (``order=1``).  This is
    the "rotate the ROI so the line is horizontal, then sum +/-2 rows"
    operation.

    Returns a 1-D vector of length ``round(line_length_px + 2*extend) + 1``.
    """
    image = np.asarray(image, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    length = float(np.hypot(*d))
    if length == 0:
        raise ValueError("The two spots coincide; cannot define a line.")

    u = d / length                       # unit vector along the line (dr, dc)
    v = np.array([-u[1], u[0]])          # unit perpendicular
    n = int(round(length + 2 * extend)) + 1
    t = np.linspace(-extend, length + extend, n)

    scan = np.zeros(n, dtype=float)
    for s in range(-half_width, half_width + 1):
        rr = p1[0] + t * u[0] + s * v[0]
        cc = p1[1] + t * u[1] + s * v[1]
        scan += map_coordinates(image, [rr, cc], order=order,
                                mode="constant", cval=0.0)
    return scan


def roi_scans(image: np.ndarray, spots_by_roi: Dict[str, list],
              pixel_size: float = 1.0, half_width: int = 2,
              groups: Optional[dict] = None) -> Dict[str, dict]:
    """Compute the scan, line length and group for every 2-spot ROI.

    ``spots_by_roi`` maps ROI label -> list of (row, col) reference-channel
    centroids.  ROIs whose spot count is not exactly 2 are skipped.  ``image``
    may be a z-stack (it is max-projected) or a 2-D image.
    Returns ``{label: {"scan", "length_px", "length_um", "group"}}``.  Group
    labels are assigned from the pooled length distribution of the two-spot
    ROIs (see :func:`classify_lengths`).
    """
    image = max_projection(image)
    out = {}
    for label, spots in spots_by_roi.items():
        if len(spots) != 2:                      # exclude non-two-spot ROIs
            continue
        p1, p2 = spots[0], spots[1]
        out[label] = {
            "scan": integrated_intensity_scan(image, p1, p2, half_width),
            "length_px": line_length_px(p1, p2),
            "length_um": line_length_um(p1, p2, pixel_size),
        }
    keys = list(out)
    labels, _ = classify_lengths([out[k]["length_um"] for k in keys])
    for k, lab in zip(keys, labels):
        out[k]["group"] = lab
    return out


# --------------------------------------------------------------------------- #
# (2) Classify ROIs by line length
# --------------------------------------------------------------------------- #
def _kmeans_1d(x: np.ndarray, k: int, iters: int = 100):
    """Lloyd's k-means on 1-D data; returns (labels-by-increasing-center)."""
    xs = np.sort(x)
    centers = (np.quantile(xs, np.linspace(0, 1, k + 2)[1:-1])
               if k > 1 else np.array([x.mean()]))
    centers = np.asarray(centers, dtype=float)
    for _ in range(iters):
        idx = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new = np.array([x[idx == j].mean() if np.any(idx == j) else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    idx = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
    # Relabel clusters by increasing centre so group numbering follows length.
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[i] for i in idx]), centers[order]


def _cluster_thresholds(x: np.ndarray, idx: np.ndarray) -> list:
    """Midpoint cut-points between consecutive (length-ordered) clusters."""
    thr = []
    for j in range(idx.max()):
        lo = x[idx == j].max()
        hi = x[idx == j + 1].min()
        thr.append(float((lo + hi) / 2.0))
    return thr


def _silhouette_1d(x: np.ndarray, idx: np.ndarray) -> float:
    """Mean silhouette score for a 1-D clustering (higher = better separated)."""
    clusters = np.unique(idx)
    if clusters.size < 2:
        return -1.0
    per_cluster = {c: x[idx == c] for c in clusters}
    scores = []
    for xi, ci in zip(x, idx):
        same = per_cluster[ci]
        a = np.mean(np.abs(same - xi)) if same.size > 1 else 0.0
        b = min(np.mean(np.abs(per_cluster[c] - xi))
                for c in clusters if c != ci)
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def classify_lengths(lengths: Sequence[float],
                     max_groups: int = MAX_LENGTH_GROUPS,
                     min_count: int = MIN_GROUP_COUNT,
                     min_silhouette: float = 0.6):
    """Assign a group label to each length from the length *distribution*.

    Clusters the lengths with 1-D k-means (no hard-coded limits).  The number
    of groups is chosen data-drivenly: among all ``k`` (2 .. *max_groups*) whose
    groups each hold >= *min_count* scans, the one with the best mean
    **silhouette** is kept, provided it clears *min_silhouette* (otherwise the
    lengths form a single group).  This favours genuinely separated clusters and
    won't split one broad cluster just because its halves are populous.  Groups
    are numbered ``group1..groupK`` by increasing length.  Returns
    ``(labels, thresholds)`` (the length cut-points between groups).
    """
    lengths = np.asarray(list(lengths), dtype=float)
    n = lengths.size
    if n == 0:
        return [], []
    uniq = np.unique(lengths)
    if uniq.size < 2:
        return ["group1"] * n, []

    best_idx, best_thr, best_sil = None, None, -np.inf
    for k in range(2, min(int(max_groups), uniq.size) + 1):
        idx, _ = _kmeans_1d(lengths, k)
        sizes = [int((idx == i).sum()) for i in range(idx.max() + 1)]
        if len(sizes) < k or min(sizes) < min_count:
            continue
        sil = _silhouette_1d(lengths, idx)
        if sil > best_sil:
            best_idx, best_thr, best_sil = idx, _cluster_thresholds(lengths, idx), sil

    if best_idx is None or best_sil < min_silhouette:
        return ["group1"] * n, []
    labels = [f"group{int(i) + 1}" for i in best_idx]
    return labels, best_thr


# --------------------------------------------------------------------------- #
# (3) Normalise scans within a group by resampling
# --------------------------------------------------------------------------- #
def normalize_group(scans: List[np.ndarray],
                    target_len: Optional[int] = None
                    ) -> Tuple[List[np.ndarray], int]:
    """Resample every scan in a group to a common length.

    The common length defaults to the mean number of pixels across the group's
    scans (so the normalised spatial coordinate corresponds to 1 pixel of the
    mean-length scan).  Resampling uses linear (1-D bilinear) interpolation.
    Returns ``(resampled_scans, target_len)``.
    """
    scans = [np.asarray(s, dtype=float) for s in scans if len(s) > 0]
    if not scans:
        return [], 0
    if target_len is None:
        target_len = int(round(np.mean([len(s) for s in scans])))
    target_len = max(int(target_len), 1)

    x_new = np.linspace(0.0, 1.0, target_len)
    out = []
    for s in scans:
        if len(s) == 1:
            out.append(np.full(target_len, s[0]))
            continue
        x_old = np.linspace(0.0, 1.0, len(s))
        out.append(np.interp(x_new, x_old, s))
    return out, target_len


def normalize_groups(group_scans: Dict[str, List[np.ndarray]]
                     ) -> Dict[str, dict]:
    """Normalise each group independently.

    ``group_scans`` maps group label -> list of scans.  Returns
    ``{group: {"scans": [...], "target_len": int}}``.
    """
    result = {}
    for group, scans in group_scans.items():
        resampled, target = normalize_group(scans)
        result[group] = {"scans": resampled, "target_len": target}
    return result


def linescans_from_measurements(files_data: list, table, half_width: int = 2,
                                extend: float = 5.0,
                                groups: Optional[dict] = None) -> list:
    """Per-ROI line scans from a measurement table + the loaded stacks.

    For every (file, ROI) with exactly two reference-channel spots, computes
    the raw ±(half_width)-row scan along the (elongated) connecting line on the
    reference channel's max-projection, and a background-corrected scan using
    the ROI's per-spot ring background.  ``files_data`` items carry
    ``filename``, ``stacks``, ``ref_channel`` and ``pixel_size``; ``table`` is
    the combined measurement DataFrame.  Returns a list of dicts.
    """
    n_rows = 2 * half_width + 1
    by_file = {f["filename"]: f for f in files_data}
    out = []
    for fname, f in by_file.items():
        ref = f["ref_channel"]
        proj = max_projection(f["stacks"][ref])
        px = f.get("pixel_size", 1.0)
        sub = table[(table["filename"] == fname) & (table["channel"] == ref)]
        for region, g in sub.groupby("region"):
            if len(g) != 2:
                continue
            # Always start the scan at the brighter spot.
            bright = next((c for c in ("peak_intensity", "mean_intensity",
                                       "integrated_intensity") if c in g.columns),
                          None)
            if bright is not None:
                g = g.sort_values(bright, ascending=False, kind="stable")
            else:
                g = g.sort_values(["col_px", "row_px"])
            p1 = (float(g.iloc[0]["row_px"]), float(g.iloc[0]["col_px"]))
            p2 = (float(g.iloc[1]["row_px"]), float(g.iloc[1]["col_px"]))
            raw = integrated_intensity_scan(proj, p1, p2, half_width, extend)
            roi_bkg = float(g["bkg_median"].mean()) if "bkg_median" in g else 0.0
            corrected = raw - n_rows * roi_bkg
            out.append({
                "filename": fname, "roi": str(region),
                "length_um": line_length_um(p1, p2, px),
                "raw": raw, "corrected": corrected})

    # Assign group labels from the pooled length distribution of all scans.
    labels, _ = classify_lengths([o["length_um"] for o in out])
    for o, lab in zip(out, labels):
        o["group"] = lab
    return out


def build_linescan_sheets(linescans: list) -> Dict[str, "object"]:
    """Build the three export sheets (as long-format DataFrames).

    * ``raw_intensity`` — raw integrated scan vectors.
    * ``corrected_intensity`` — background-corrected vectors.
    * ``normalized`` — per-group resampled vectors with a normalized spatial
      coordinate (in mean pixels) and fractional intensity (each vector summed
      to 1).
    """
    import pandas as pd

    def group_label(ls):
        return ls["group"] if ls["group"] is not None else "unclassified"

    raw_rows, corr_rows = [], []
    for ls in linescans:
        g = group_label(ls)
        for i, val in enumerate(ls["raw"]):
            raw_rows.append({"filename": ls["filename"], "roi": ls["roi"],
                             "group": g, "position_px": i,
                             "raw_intensity": float(val)})
        for i, val in enumerate(ls["corrected"]):
            corr_rows.append({"filename": ls["filename"], "roi": ls["roi"],
                              "group": g, "position_px": i,
                              "corrected_intensity": float(val)})

    # Normalize the corrected vectors within each classified group.
    by_group: Dict[str, list] = {}
    for ls in linescans:
        if ls["group"] is None:
            continue
        by_group.setdefault(ls["group"], []).append(ls)

    sheets = {
        "raw_intensity": pd.DataFrame(
            raw_rows, columns=["filename", "roi", "group", "position_px",
                               "raw_intensity"]),
        "corrected_intensity": pd.DataFrame(
            corr_rows, columns=["filename", "roi", "group", "position_px",
                                "corrected_intensity"]),
    }

    # One normalized sheet PER group (group1 and group2 both reported), plus a
    # per-group Summary of the mean/std fractional profile and scan count.
    summary_rows = []
    for group in sorted(by_group):
        items = by_group[group]
        resampled, target = normalize_group([it["corrected"] for it in items])
        fracs = []
        rows = []
        for it, vec in zip(items, resampled):
            total = float(np.nansum(vec))
            frac = (vec / total) if total > 0 else np.zeros_like(vec)
            fracs.append(frac)
            for i, val in enumerate(frac):
                rows.append({
                    "filename": it["filename"], "roi": it["roi"],
                    "group": group, "norm_pixel": i,
                    "fractional_intensity": float(val)})
        sheets[f"normalized_{group}"] = pd.DataFrame(
            rows, columns=["filename", "roi", "group", "norm_pixel",
                           "fractional_intensity"])

        # Average / std of the fractional profile across scans in this group.
        stack = np.vstack(fracs) if fracs else np.zeros((0, target))
        n = stack.shape[0]
        mean = stack.mean(axis=0) if n else np.zeros(target)
        std = stack.std(axis=0, ddof=1) if n > 1 else np.zeros(target)
        for i in range(target):
            summary_rows.append({
                "group": group, "n_scans": n, "norm_pixel": i,
                "mean_fractional_intensity": float(mean[i]),
                "std_fractional_intensity": float(std[i])})

    sheets["Summary"] = pd.DataFrame(
        summary_rows, columns=["group", "n_scans", "norm_pixel",
                               "mean_fractional_intensity",
                               "std_fractional_intensity"])
    return sheets


def group_and_normalize(roi_results: Dict[str, dict]) -> Dict[str, dict]:
    """Group :func:`roi_scans` output by ``group`` and normalise each group."""
    group_scans: Dict[str, List[np.ndarray]] = {}
    group_labels: Dict[str, List[str]] = {}
    for label, info in roi_results.items():
        g = info.get("group")
        if g is None:
            continue
        group_scans.setdefault(g, []).append(info["scan"])
        group_labels.setdefault(g, []).append(label)
    out = normalize_groups(group_scans)
    for g in out:
        out[g]["rois"] = group_labels[g]
    return out
