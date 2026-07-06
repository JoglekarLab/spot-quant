"""Test fixtures and import shims for the spot_quant test suite.

The pipeline/filters modules depend only on numpy/scipy/skimage/pandas, but
importing the ``spot_quant`` package normally pulls in napari and qtpy via
``__init__`` -> ``app``.  To test the pure-logic modules without a GUI stack
installed, we register ``spot_quant`` as a namespace package *without* running
its ``__init__`` and load the individual modules from their files.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

PKG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spot_quant"))


def _ensure_namespace_package():
    if "spot_quant" not in sys.modules:
        pkg = types.ModuleType("spot_quant")
        pkg.__path__ = [PKG_DIR]
        sys.modules["spot_quant"] = pkg


def load_module(name):
    """Load ``spot_quant.<name>`` from file without triggering __init__."""
    _ensure_namespace_package()
    full = f"spot_quant.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(PKG_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def filters():
    return load_module("filters")


@pytest.fixture(scope="session")
def pipeline():
    return load_module("pipeline")


@pytest.fixture
def spotty_image():
    """128x128 image: noisy background plus 3 bright Gaussian spots."""
    rng = np.random.default_rng(0)
    img = rng.normal(100, 5, (128, 128))
    yy, xx = np.ogrid[:128, :128]
    centers = [(30, 40), (70, 90), (100, 20)]
    for y, x in centers:
        img += 400 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 8.0)
    return img, centers


@pytest.fixture
def spotty_stack():
    """Z-stack with two spots, each peaking (in-focus) at a known plane.

    Returns ``(stack, spots)`` where ``spots`` is a list of dicts giving the
    fixed XY centre and the in-focus z-plane of each spot.  Each spot keeps a
    constant XY position across planes (so it links) and its amplitude follows
    a Gaussian along z that peaks at ``z_focus``.
    """
    rng = np.random.default_rng(1)
    Z, H, W = 7, 96, 96
    stack = rng.normal(100, 4, (Z, H, W))
    yy, xx = np.ogrid[:H, :W]
    spots = [
        {"y": 30, "x": 40, "z_focus": 2},
        {"y": 70, "x": 60, "z_focus": 5},
    ]
    for s in spots:
        for z in range(Z):
            amp = 500 * np.exp(-((z - s["z_focus"]) ** 2) / (2 * 1.2 ** 2))
            stack[z] += amp * np.exp(
                -((yy - s["y"]) ** 2 + (xx - s["x"]) ** 2) / 8.0)
    return stack, spots
