"""Tests for measurement-driven line-scan extraction and export sheets."""
import numpy as np
import pandas as pd
import pytest

from conftest import load_module


@pytest.fixture(scope="module")
def scan():
    return load_module("scan")


def _table():
    # Two length clusters: 3 short ROIs (~20 px) and 3 long ROIs (~50 px), so
    # the data-driven classifier forms group1 (short) and group2 (long).
    # mCherry rows present but must be ignored.
    rows = []
    short = {"R1": (20, 40), "R2": (20, 41), "R3": (20, 39)}   # ~20 px apart
    long = {"R4": (10, 60), "R5": (10, 61), "R6": (10, 59)}    # ~50 px apart
    for roi, (c0, c1) in {**short, **long}.items():
        rows.append(("a.nd2", "GFP", roi, 30, c0, 10.0))
        rows.append(("a.nd2", "GFP", roi, 30, c1, 10.0))
    rows.append(("a.nd2", "mCherry", "R1", 30, 20, 5.0))       # ignored
    return pd.DataFrame(rows, columns=[
        "filename", "channel", "region", "row_px", "col_px", "bkg_median"])


def _files():
    stack = np.full((3, 60, 80), 100.0)     # uniform reference stack
    return [{"filename": "a.nd2", "stacks": {"GFP": stack},
             "ref_channel": "GFP", "pixel_size": 0.1}]


def test_linescans_from_measurements(scan):
    out = scan.linescans_from_measurements(_files(), _table())
    assert len(out) == 6                      # six 2-spot ROIs (mCherry ignored)
    by_roi = {d["roi"]: d for d in out}
    assert by_roi["R1"]["group"] == "group1"      # short cluster
    assert by_roi["R4"]["group"] == "group2"      # long cluster
    # corrected = raw - n_rows(=5) * bkg(=10) on a uniform-100 image interior.
    ls = by_roi["R1"]
    mid = len(ls["raw"]) // 2
    assert ls["raw"][mid] == pytest.approx(500.0, abs=1e-6)
    assert ls["corrected"][mid] == pytest.approx(450.0, abs=1e-6)


def test_build_linescan_sheets_reports_both_groups(scan):
    out = scan.linescans_from_measurements(_files(), _table())
    sheets = scan.build_linescan_sheets(out)
    # Both group1 and group2 get their own normalized sheet, plus a Summary.
    assert set(sheets) == {"raw_intensity", "corrected_intensity",
                           "normalized_group1", "normalized_group2", "Summary"}
    assert list(sheets["normalized_group1"].columns) == [
        "filename", "roi", "group", "norm_pixel", "fractional_intensity"]
    assert not sheets["normalized_group1"].empty
    assert not sheets["normalized_group2"].empty
    # Each normalized line-scan's fractional intensity sums to ~1.
    for name in ("normalized_group1", "normalized_group2"):
        for _, g in sheets[name].groupby(["filename", "roi"]):
            assert g["fractional_intensity"].sum() == pytest.approx(1.0, abs=1e-6)


def test_brighter_spot_is_first(scan):
    # ROI with a bright spot on the right and a dim spot on the left; the scan
    # must start at the brighter (right) spot.
    img = np.zeros((40, 60))
    img[20, 15] = 50.0     # dim, left
    img[20, 45] = 500.0    # bright, right
    stack = img[None]
    files = [{"filename": "a.nd2", "stacks": {"GFP": stack},
              "ref_channel": "GFP", "pixel_size": 0.1}]
    tbl = pd.DataFrame([
        ("a.nd2", "GFP", "ROI1", 20, 15, 5.0, 50.0),
        ("a.nd2", "GFP", "ROI1", 20, 45, 5.0, 500.0),
    ], columns=["filename", "channel", "region", "row_px", "col_px",
                "bkg_median", "peak_intensity"])
    out = scan.linescans_from_measurements(files, tbl)
    raw = out[0]["raw"]
    # First non-extended sample (index 5) should sit on the bright spot.
    assert raw[:len(raw) // 2].max() > raw[len(raw) // 2:].max()


def test_group_column_always_labelled(scan):
    # Every scan gets a data-driven group label (no blanks / NaN).
    sheets = scan.build_linescan_sheets(
        scan.linescans_from_measurements(_files(), _table()))
    col = sheets["raw_intensity"]["group"]
    assert col.notna().all()
    assert col.map(lambda g: str(g).startswith("group")).all()


def test_summary_sheet(scan):
    out = scan.linescans_from_measurements(_files(), _table())
    summary = scan.build_linescan_sheets(out)["Summary"]
    assert list(summary.columns) == [
        "group", "n_scans", "norm_pixel", "mean_fractional_intensity",
        "std_fractional_intensity"]
    # Three scans per group here.
    for group, g in summary.groupby("group"):
        assert (g["n_scans"] == 3).all()
        # The mean fractional profile sums to ~1.
        assert g["mean_fractional_intensity"].sum() == pytest.approx(1.0, abs=1e-6)


def test_normalized_uses_group_mean_length(scan):
    # Two group1 ROIs of different lengths -> both resampled to the mean length.
    files = _files()
    tbl = pd.DataFrame([
        ("a.nd2", "GFP", "ROI1", 30, 20, 10.0),
        ("a.nd2", "GFP", "ROI1", 30, 36, 10.0),   # len 16 px -> 1.6 um group1
        ("a.nd2", "GFP", "ROI2", 30, 20, 10.0),
        ("a.nd2", "GFP", "ROI2", 30, 44, 10.0),   # len 24 px -> 2.4 um group1
    ], columns=["filename", "channel", "region", "row_px", "col_px",
                "bkg_median"])
    out = scan.linescans_from_measurements(files, tbl)
    sheets = scan.build_linescan_sheets(out)
    lengths = sheets["normalized_group1"].groupby(["filename", "roi"]).size()
    assert lengths.nunique() == 1             # all normalized to one length
