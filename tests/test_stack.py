"""Tests for full-stack detection: linking, in-focus, background, ROIs."""
import numpy as np
import pytest


def _nearest_spot(row, spots):
    return min(spots, key=lambda s: abs(s["y"] - row["row_px"])
              + abs(s["x"] - row["col_px"]))


def test_links_to_one_row_per_spot(pipeline, spotty_stack):
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(
        threshold_method="Otsu", min_distance=5, peak_rel_threshold=0.3,
        link_max_dist=2.0)
    result = pipeline.run_stack(stack, params)
    # Two physical spots -> two in-focus rows (not one per plane).
    assert len(result.centroids) == len(spots)


def test_in_focus_plane_is_brightest(pipeline, spotty_stack):
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(
        threshold_method="Otsu", min_distance=5, peak_rel_threshold=0.3)
    result = pipeline.run_stack(stack, params)
    for _, row in result.measurements.iterrows():
        s = _nearest_spot(row, spots)
        assert int(row["z_plane"]) == s["z_focus"]


def test_centroids_are_3d(pipeline, spotty_stack):
    stack, _ = spotty_stack
    result = pipeline.run_stack(stack, pipeline.PipelineParams(
        threshold_method="Otsu", peak_rel_threshold=0.3))
    assert result.centroids.shape[1] == 3        # z, y, x
    assert result.foreground.shape == stack.shape


def test_measurement_columns(pipeline, spotty_stack):
    stack, _ = spotty_stack
    result = pipeline.run_stack(
        stack, pipeline.PipelineParams(threshold_method="Otsu",
                                       peak_rel_threshold=0.3),
        pixel_size=0.1, pixel_unit="um")
    assert list(result.measurements.columns) == [
        "spot_id", "z_plane", "row_px", "col_px", "x_um", "y_um",
        "peak_intensity", "mean_intensity", "integrated_intensity",
        "n_pixels", "bkg_median", "background", "corrected_integrated",
        "region", "roi_threshold", "n_planes_linked"]


def test_roi_thresholds_recorded(pipeline, spotty_stack):
    stack, spots = spotty_stack
    rois = [(s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, f"ROI{i + 1}")
            for i, s in enumerate(spots)]
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    result = pipeline.run_stack(stack, params, rois=rois)
    # A threshold is recorded for every ROI...
    assert set(result.roi_thresholds) == {"ROI1", "ROI2"}
    assert all(t is not None for t in result.roi_thresholds.values())
    # ...and each spot row carries its ROI's threshold.
    for _, row in result.measurements.iterrows():
        assert row["roi_threshold"] == result.roi_thresholds[row["region"]]


# -- linking unit test ------------------------------------------------------ #
def test_link_tracks_chains_consecutive_planes(pipeline):
    # Same XY across 3 planes -> one track; a far detection -> its own track.
    dets = [
        [(10, 10, 5.0), (50, 50, 1.0)],
        [(11, 10, 9.0)],                 # links to (10,10): dist 1 < 2
        [(10, 11, 4.0)],                 # links again: dist ~1.4 < 2
    ]
    tracks = pipeline.link_tracks(dets, max_dist=2.0)
    lengths = sorted(len(t) for t in tracks)
    assert lengths == [1, 3]
    # In-focus pick = max intensity element of the 3-chain (plane 1, inten 9).
    chain = max(tracks, key=len)
    z, r, c, inten = max(chain, key=lambda t: t[3])
    assert (z, inten) == (1, 9.0)


def test_link_breaks_when_gap_too_large(pipeline):
    dets = [[(10, 10, 1.0)], [(20, 20, 1.0)]]   # 14 px apart -> no link
    tracks = pipeline.link_tracks(dets, max_dist=2.0)
    assert len(tracks) == 2


# -- background ring math (threshold-masked, read from raw) ----------------- #
def test_background_ring_math(pipeline):
    # Disk pixels = 100 (foreground), everything else (incl. ring) = 10.
    img = np.full((40, 40), 10.0)
    r, c, radius = 20, 20, 3
    yy, xx = np.ogrid[:40, :40]
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2
    img[disk] = 100.0
    fg = img > 50            # foreground = the bright disk only
    params = pipeline.PipelineParams(measure_radius=radius, bkg_gap=2,
                                     bkg_width=3)
    m = pipeline.measure_spot(img, fg, r, c, params)
    n = int(disk.sum())
    assert m["n_pixels"] == n
    assert m["bkg_median"] == pytest.approx(10.0)
    assert m["integrated_intensity"] == pytest.approx(100.0 * n)
    assert m["background"] == pytest.approx(10.0 * n)
    assert m["corrected_integrated"] == pytest.approx(90.0 * n)


def test_signal_excludes_non_foreground_pixels(pipeline):
    # Only half the disk is foreground -> only those pixels count as signal.
    img = np.full((40, 40), 10.0)
    r, c, radius = 20, 20, 4
    yy, xx = np.ogrid[:40, :40]
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2
    img[disk] = 100.0
    fg = disk.copy()
    fg[:, c:] = False        # drop the right half of the disk from foreground
    params = pipeline.PipelineParams(measure_radius=radius, bkg_gap=2, bkg_width=3)
    m = pipeline.measure_spot(img, fg, r, c, params)
    assert m["n_pixels"] == int((disk & fg).sum())
    assert m["n_pixels"] < int(disk.sum())


# -- ROI restricts detection ------------------------------------------------ #
def test_detection_limited_to_rois(pipeline, spotty_stack):
    stack, spots = spotty_stack
    # ROI tightly around the first spot only -> only that spot is detected.
    s = spots[0]
    roi = (s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, "ROI1")
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    result = pipeline.run_stack(stack, params, rois=[roi])
    # The spot outside the ROI must not be detected at all.
    assert len(result.centroids) == 1
    row = result.measurements.iloc[0]
    assert row["region"] == "ROI1"
    assert _nearest_spot(row, spots)["z_focus"] == spots[0]["z_focus"]
    # No detected spot may lie outside the ROI bounds.
    assert ((result.centroids[:, 1] >= s["y"] - 8).all()
            and (result.centroids[:, 1] <= s["y"] + 8).all())


def test_two_rois_detect_both_spots(pipeline, spotty_stack):
    stack, spots = spotty_stack
    rois = [(s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, f"ROI{i + 1}")
            for i, s in enumerate(spots)]
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    result = pipeline.run_stack(stack, params, rois=rois)
    assert len(result.centroids) == len(spots)
    assert set(result.measurements["region"]) == {"ROI1", "ROI2"}


# -- boundary removal & sorting --------------------------------------------- #
def test_boundary_touching_spots_removed(pipeline, spotty_stack):
    stack, spots = spotty_stack
    s = spots[0]
    # ROI whose lower row edge clips the spot disk (radius 3) -> removed.
    roi_clip = (s["y"] - 8, s["y"] + 1, s["x"] - 8, s["x"] + 8, "ROI1")
    # Same spot with a comfortably larger ROI -> kept.
    roi_ok = (s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, "ROI1")
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3, measure_radius=3)
    clipped = pipeline.run_stack(stack, params, rois=[roi_clip])
    kept = pipeline.run_stack(stack, params, rois=[roi_ok])
    assert len(clipped.centroids) == 0
    assert len(kept.centroids) == 1


def test_measurements_sorted_by_roi(pipeline, spotty_stack):
    stack, spots = spotty_stack
    # Give the first spot's ROI a HIGHER number to test the sort.
    roi_a = (spots[0]["y"] - 8, spots[0]["y"] + 8,
             spots[0]["x"] - 8, spots[0]["x"] + 8, "ROI2")
    roi_b = (spots[1]["y"] - 8, spots[1]["y"] + 8,
             spots[1]["x"] - 8, spots[1]["x"] + 8, "ROI1")
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    result = pipeline.run_stack(stack, params, rois=[roi_a, roi_b])
    regions = list(result.measurements["region"])
    assert regions == ["ROI1", "ROI2"]


def test_2d_input_promoted_to_single_plane(pipeline, spotty_image):
    img, _ = spotty_image
    result = pipeline.run_stack(img, pipeline.PipelineParams(
        threshold_method="Otsu", peak_rel_threshold=0.3))
    assert result.n_planes == 1
    assert (result.measurements["z_plane"] == 0).all()
