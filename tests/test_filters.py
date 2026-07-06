"""Tests for the smoothing and top-hat filters."""
import numpy as np


def test_all_smoothing_methods_preserve_shape_and_finite(filters, spotty_image):
    img, _ = spotty_image
    for method in filters.SMOOTHING_METHODS:
        out = filters.smooth(img, method, 2)
        assert out.shape == img.shape, method
        assert np.isfinite(out).all(), method


def test_smoothing_methods_list():
    # Guard against accidental edits to the public method set.
    from conftest import load_module
    assert load_module("filters").SMOOTHING_METHODS == (
        "Gaussian", "Median", "Kuwahara", "Gaussian low pass")


def test_kuwahara_is_edge_preserving(filters):
    # A step edge should stay sharp-ish: output values cluster near the two
    # input levels rather than blurring to the mean between them.
    img = np.zeros((64, 64))
    img[:, 32:] = 100.0
    out = filters.kuwahara(img, 3)
    mid = out[20:44, 20:44]
    near_levels = np.minimum(np.abs(mid - 0), np.abs(mid - 100))
    assert near_levels.mean() < 25.0


def test_white_tophat_suppresses_flat_background(filters):
    # White top-hat of a constant image is ~zero everywhere.
    flat = np.full((32, 32), 50.0)
    out = filters.white_tophat(flat, 5)
    assert np.allclose(out, 0.0)


def test_white_tophat_keeps_small_bright_feature(filters):
    img = np.zeros((64, 64))
    img[32, 32] = 200.0
    out = filters.white_tophat(img, 5)
    assert out[32, 32] > 100.0


def test_unknown_method_raises(filters):
    import pytest
    with pytest.raises(ValueError):
        filters.smooth(np.zeros((8, 8)), "Nope", 1)
