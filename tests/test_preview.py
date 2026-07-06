"""Tests for the ROI-limited single-plane preview (global detection disabled)."""
import numpy as np


def _rois(spots):
    return [(s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, f"ROI{i + 1}")
            for i, s in enumerate(spots)]


def test_preview_without_rois_finds_nothing(pipeline, spotty_stack):
    stack, _ = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    # In-focus plane of spot 0 is z=2, yet with no ROI nothing is detected.
    result = pipeline.run_preview(stack, z=2, params=params, rois=[])
    assert len(result.centroids) == 0
    assert not result.foreground.any()
    assert result.roi_thresholds == {}


def test_preview_detects_only_inside_roi(pipeline, spotty_stack):
    stack, spots = spotty_stack
    s = spots[0]
    roi = (s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, "ROI1")
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    result = pipeline.run_preview(stack, z=s["z_focus"], params=params,
                                  rois=[roi])
    # Foreground is confined to the ROI box.
    outside = result.foreground.copy()
    outside[s["y"] - 8:s["y"] + 8, s["x"] - 8:s["x"] + 8] = False
    assert not outside.any()
    # The spot in this plane is found and lies inside the ROI.
    assert len(result.centroids) >= 1
    for r, c in result.centroids:
        assert s["y"] - 8 <= r <= s["y"] + 8 and s["x"] - 8 <= c <= s["x"] + 8


def test_preview_records_roi_thresholds(pipeline, spotty_stack):
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    result = pipeline.run_preview(stack, z=2, params=params, rois=_rois(spots))
    assert set(result.roi_thresholds) == {"ROI1", "ROI2"}
    assert all(t is not None for t in result.roi_thresholds.values())


def test_small_masks_removed(pipeline):
    # A 3x3 (9 px) blob and a large blob inside one ROI. With min_mask_size=10
    # only the large blob survives the thresholding cleanup.
    Z, H, W = 3, 60, 60
    stack = np.full((Z, H, W), 10.0)
    stack[:, 10:13, 10:13] = 300.0          # 9-pixel blob -> removed
    stack[:, 30:40, 30:40] = 300.0          # 100-pixel blob -> kept
    roi = [(0, H, 0, W, "ROI1")]
    big_only = pipeline.PipelineParams(threshold_method="Otsu",
                                       peak_rel_threshold=0.2, min_mask_size=10,
                                       measure_radius=2)
    res = pipeline.run_preview(stack, z=1, params=big_only, rois=roi)
    # The small blob's pixels are gone from the foreground.
    assert not res.foreground[10:13, 10:13].any()
    assert res.foreground[30:40, 30:40].any()

    # Lowering the threshold to <=9 keeps the small blob.
    keep_small = pipeline.PipelineParams(threshold_method="Otsu",
                                         peak_rel_threshold=0.2, min_mask_size=5,
                                         measure_radius=2)
    res2 = pipeline.run_preview(stack, z=1, params=keep_small, rois=roi)
    assert res2.foreground[10:13, 10:13].any()


def test_small_masks_removed_in_stack(pipeline):
    Z, H, W = 3, 60, 60
    stack = np.full((Z, H, W), 10.0)
    stack[:, 10:13, 10:13] = 300.0
    stack[:, 30:40, 30:40] = 300.0
    roi = [(0, H, 0, W, "ROI1")]
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.2, min_mask_size=10,
                                     measure_radius=2)
    res = pipeline.run_stack(stack, params, rois=roi)
    assert not res.foreground[:, 10:13, 10:13].any()
    assert res.foreground[:, 30:40, 30:40].any()


def test_preview_threshold_uses_stack_not_single_plane(pipeline, spotty_stack):
    # An out-of-focus plane still yields a threshold (pooled over the stack),
    # so the ROI histogram is genuinely stack-based.
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    roi = _rois(spots)[:1]
    far = pipeline.run_preview(stack, z=6, params=params, rois=roi)  # off-focus
    assert far.roi_thresholds["ROI1"] is not None
