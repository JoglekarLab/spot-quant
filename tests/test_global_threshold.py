"""Global (pooled-ROI) thresholding tests."""
import numpy as np
from skimage.filters import threshold_otsu


def _rois(spots):
    return [(s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, f"ROI{i + 1}")
            for i, s in enumerate(spots)]


def test_global_threshold_pools_all_rois(pipeline):
    # Build a top-hat-like volume; two ROIs with different local content.
    Z, H, W = 3, 40, 80
    top = np.zeros((Z, H, W))
    top[:, 10:20, 10:20] = 50.0     # ROI1 region has some signal
    top[:, 10:20, 50:60] = 200.0    # ROI2 region brighter
    rois = [(5, 30, 5, 30, "ROI1"), (5, 30, 45, 70, "ROI2")]
    params = pipeline.PipelineParams(threshold_method="Otsu")
    gt = pipeline.global_roi_threshold(top, params, rois)
    # Expected: Otsu over the pooled pixels of both ROI boxes.
    pooled = np.concatenate([top[:, 5:30, 5:30].ravel(),
                             top[:, 5:30, 45:70].ravel()])
    assert gt == threshold_otsu(pooled)


def test_all_rois_share_one_threshold(pipeline, spotty_stack):
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    res = pipeline.run_stack(stack, params, rois=_rois(spots))
    vals = set(res.roi_thresholds.values())
    assert len(vals) == 1                      # one global threshold for all
    assert next(iter(vals)) is not None


def test_ilastik_global_threshold_is_none(pipeline):
    top = np.ones((2, 10, 10))
    params = pipeline.PipelineParams(threshold_method="ilastik")
    assert pipeline.global_roi_threshold(top, params,
                                         [(0, 10, 0, 10, "ROI1")]) is None


def test_threshold_rois_override_pool(pipeline, spotty_stack):
    # Pooling over only one ROI can differ from pooling over both.
    stack, spots = spotty_stack
    rois = _rois(spots)
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    both = pipeline.run_stack(stack, params, rois=rois, threshold_rois=rois)
    one = pipeline.run_stack(stack, params, rois=rois, threshold_rois=[rois[0]])
    t_both = next(iter(both.roi_thresholds.values()))
    t_one = next(iter(one.roi_thresholds.values()))
    assert t_both is not None and t_one is not None
    # Both still apply a single global threshold across all ROIs.
    assert len(set(both.roi_thresholds.values())) == 1
    assert len(set(one.roi_thresholds.values())) == 1
