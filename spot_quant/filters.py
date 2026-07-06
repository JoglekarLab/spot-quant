"""Smoothing and morphological filters for the spot-detection pipeline.

All functions operate on 2-D float images and return a float image of the
same shape.  ``size`` is interpreted as a radius (pixels) for spatial
filters and as a cutoff for the frequency-domain low-pass.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, median
from skimage.morphology import disk, white_tophat as _white_tophat

SMOOTHING_METHODS = ("Gaussian", "Median", "Kuwahara", "Gaussian low pass")


def _as_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=float)


def gaussian_smooth(img: np.ndarray, size: float) -> np.ndarray:
    """Spatial Gaussian blur; ``size`` is the standard deviation in pixels."""
    return gaussian(_as_float(img), sigma=max(size, 1e-6), preserve_range=True)


def median_smooth(img: np.ndarray, size: float) -> np.ndarray:
    """Median filter over a disk footprint of radius ``size``."""
    radius = max(int(round(size)), 1)
    return median(_as_float(img), footprint=disk(radius)).astype(float)


def kuwahara(img: np.ndarray, size: float) -> np.ndarray:
    """Edge-preserving Kuwahara filter.

    Each pixel is replaced by the mean of whichever of its four quadrant
    sub-windows has the lowest intensity variance.  ``size`` sets the
    quadrant radius, so the full window is ``2*size + 1`` on a side.
    """
    img = _as_float(img)
    radius = max(int(round(size)), 1)
    # Box mean / variance over a (radius+1) square via uniform filters.
    win = radius + 1
    mean = ndi.uniform_filter(img, win)
    sqr_mean = ndi.uniform_filter(img * img, win)
    var = np.clip(sqr_mean - mean * mean, 0, None)

    # Shifts that place each quadrant's box-stat at the central pixel.
    shifts = {
        "nw": (-radius, -radius),
        "ne": (-radius, radius),
        "sw": (radius, -radius),
        "se": (radius, radius),
    }
    means = []
    variances = []
    for dy, dx in shifts.values():
        means.append(ndi.shift(mean, (dy, dx), order=0, mode="nearest"))
        variances.append(ndi.shift(var, (dy, dx), order=0, mode="nearest"))

    means = np.stack(means)          # (4, H, W)
    variances = np.stack(variances)  # (4, H, W)
    best = np.argmin(variances, axis=0)
    return np.take_along_axis(means, best[None], axis=0)[0]


def gaussian_lowpass(img: np.ndarray, size: float) -> np.ndarray:
    """Frequency-domain Gaussian low-pass filter.

    ``size`` is the cutoff radius (in pixels) of the Gaussian envelope in
    the frequency domain: larger ``size`` keeps more high-frequency detail.
    """
    img = _as_float(img)
    f = np.fft.fftshift(np.fft.fft2(img))
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma = max(size, 1e-6)
    envelope = np.exp(-d2 / (2.0 * sigma * sigma))
    filtered = np.fft.ifft2(np.fft.ifftshift(f * envelope))
    return np.real(filtered)


def smooth(img: np.ndarray, method: str, size: float) -> np.ndarray:
    """Dispatch to the requested smoothing method."""
    if method == "Gaussian":
        return gaussian_smooth(img, size)
    if method == "Median":
        return median_smooth(img, size)
    if method == "Kuwahara":
        return kuwahara(img, size)
    if method == "Gaussian low pass":
        return gaussian_lowpass(img, size)
    raise ValueError(f"Unknown smoothing method: {method!r}")


def white_tophat(img: np.ndarray, size: float) -> np.ndarray:
    """White top-hat with a disk structuring element of radius ``size``.

    Suppresses slowly varying background and keeps bright features smaller
    than the structuring element (i.e. the spots).
    """
    radius = max(int(round(size)), 1)
    return _white_tophat(_as_float(img), footprint=disk(radius)).astype(float)
