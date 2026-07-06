"""QC: spots with short Z-linkage are dropped."""
import numpy as np


def _two_plane_spot_stack():
    """Z=5 stack with a spot present in only planes 1 and 2 (linkage = 2)."""
    Z, H, W = 5, 60, 60
    rng = np.random.default_rng(3)
    stack = rng.normal(100, 3, (Z, H, W))
    yy, xx = np.ogrid[:H, :W]
    blob = 500 * np.exp(-((yy - 30) ** 2 + (xx - 30) ** 2) / 8.0)
    stack[1] += blob
    stack[2] += blob
    roi = [(20, 40, 20, 40, "ROI1")]
    return stack, roi


def test_short_linkage_dropped_by_default(pipeline):
    stack, roi = _two_plane_spot_stack()
    # Default min_link_planes = 3 -> the 2-plane spot is discarded.
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    assert params.min_link_planes == 3
    res = pipeline.run_stack(stack, params, rois=roi)
    assert len(res.centroids) == 0


def test_short_linkage_kept_when_threshold_lowered(pipeline):
    stack, roi = _two_plane_spot_stack()
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3, min_link_planes=2)
    res = pipeline.run_stack(stack, params, rois=roi)
    assert len(res.centroids) == 1
    assert (res.measurements["n_planes_linked"] >= 2).all()


def test_min_link_capped_at_stack_depth(pipeline, spotty_image):
    # A 2-D image (one plane) must not be wiped by a min_link of 3.
    img, _ = spotty_image
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3, min_link_planes=3)
    roi = [(0, img.shape[0], 0, img.shape[1], "ROI1")]
    res = pipeline.run_stack(img, params, rois=roi)
    assert len(res.centroids) >= 1


def test_all_kept_spots_meet_min_link(pipeline, spotty_stack):
    stack, _ = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3, min_link_planes=3)
    rois = [(0, stack.shape[1], 0, stack.shape[2], "ROI1")]
    res = pipeline.run_stack(stack, params, rois=rois)
    assert (res.measurements["n_planes_linked"] >= 3).all()
