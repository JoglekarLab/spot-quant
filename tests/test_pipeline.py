"""Tests for the single-plane preview pipeline (smooth->tophat->thresh->maxima)."""
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["Otsu", "Li"])
def test_preview_recovers_three_spots(pipeline, spotty_image, method):
    img, centers = spotty_image
    params = pipeline.PipelineParams(
        threshold_method=method, min_distance=5, peak_rel_threshold=0.3)
    result = pipeline.run(img, params)
    assert len(result.centroids) >= len(centers)


def test_centroids_land_on_implanted_spots(pipeline, spotty_image):
    img, centers = spotty_image
    params = pipeline.PipelineParams(
        threshold_method="Otsu", min_distance=5, peak_rel_threshold=0.3)
    result = pipeline.run(img, params)
    found = {tuple(c) for c in result.centroids}
    for cy, cx in centers:
        assert any(abs(cy - fy) <= 2 and abs(cx - fx) <= 2
                   for fy, fx in found), (cy, cx)


def test_preview_carries_intermediates(pipeline, spotty_image):
    img, _ = spotty_image
    result = pipeline.run(img, pipeline.PipelineParams(threshold_method="Otsu"))
    assert result.raw.shape == img.shape
    assert result.smoothed.shape == img.shape
    assert result.tophat.shape == img.shape
    assert result.foreground.dtype == bool


def test_flat_image_yields_no_spots(pipeline):
    result = pipeline.run(np.ones((32, 32)),
                          pipeline.PipelineParams(threshold_method="Otsu"))
    assert len(result.centroids) == 0


def test_ilastik_without_model_raises(pipeline):
    params = pipeline.PipelineParams(threshold_method="ilastik")
    with pytest.raises(ValueError):
        pipeline.apply_threshold(np.zeros((8, 8)), np.zeros((8, 8)), params)
