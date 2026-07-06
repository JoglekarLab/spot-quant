"""Tests for the two-spot line-scan analysis (spot_quant.scan)."""
import numpy as np
import pytest

from conftest import load_module


@pytest.fixture(scope="module")
def scan():
    return load_module("scan")


# -- geometry / classification ---------------------------------------------- #
def test_line_length(scan):
    assert scan.line_length_px((0, 0), (0, 10)) == 10
    assert scan.line_length_px((0, 0), (6, 8)) == 10
    assert scan.line_length_um((0, 0), (0, 10), 0.1) == pytest.approx(1.0)


def test_classify_lengths_finds_clusters(scan):
    # Two clear clusters -> group1 (short) and group2 (long), data-driven.
    lengths = [1.9, 2.0, 2.1, 5.0, 5.1, 5.2]
    labels, thr = scan.classify_lengths(lengths)
    assert labels == ["group1", "group1", "group1",
                      "group2", "group2", "group2"]
    assert len(thr) == 1 and 2.1 < thr[0] < 5.0     # one cut-point in the gap


def test_classify_lengths_three_clusters(scan):
    # Three well-separated clusters -> three groups, numbered by length.
    lengths = [1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2]
    labels, thr = scan.classify_lengths(lengths)
    assert sorted(set(labels)) == ["group1", "group2", "group3"]
    assert len(thr) == 2
    assert labels[0] == "group1" and labels[-1] == "group3"


def test_classify_lengths_single_cluster(scan):
    # No real separation -> everything is group1.
    labels, thr = scan.classify_lengths([3.0, 3.0, 3.1, 2.9])
    assert set(labels) == {"group1"} and thr == []


def test_classify_lengths_empty(scan):
    assert scan.classify_lengths([]) == ([], [])


# -- integrated intensity scan ---------------------------------------------- #
def test_scan_length_includes_extension(scan):
    img = np.ones((40, 40))
    # Line length 20, elongated by 5 px each side -> 20 + 10 + 1 = 31 samples.
    s = scan.integrated_intensity_scan(img, (10, 5), (10, 25), half_width=2)
    assert len(s) == 31
    # extend=0 recovers the bare line length.
    s0 = scan.integrated_intensity_scan(img, (10, 5), (10, 25), extend=0)
    assert len(s0) == 21


def test_extension_samples_beyond_endpoints(scan):
    # A ridge only between the two spots; the elongated ends fall on background.
    img = np.zeros((40, 60))
    img[20, 10:30] = 100.0                  # bright only between cols 10..30
    s = scan.integrated_intensity_scan(img, (20, 10), (20, 30), half_width=0,
                                       extend=5)
    assert s[0] == pytest.approx(0.0, abs=1e-6)     # extended-before end
    assert s[-1] == pytest.approx(0.0, abs=1e-6)    # extended-after end
    assert s[len(s) // 2] == pytest.approx(100.0, abs=1e-6)


def test_scan_sums_five_rows_of_ones(scan):
    # Uniform image: each sample sums 2*2+1 = 5 rows of value 1.
    img = np.ones((60, 60))
    s = scan.integrated_intensity_scan(img, (30, 10), (30, 50), half_width=2)
    assert np.allclose(s[2:-2], 5.0, atol=1e-6)


def test_scan_is_rotation_invariant_on_uniform_image(scan):
    # A diagonal line on a uniform image gives the same +/-2-row sum (~5).
    img = np.ones((60, 60))
    s = scan.integrated_intensity_scan(img, (10, 10), (40, 40), half_width=2)
    assert np.allclose(s[3:-3], 5.0, atol=1e-6)


def test_scan_picks_up_a_ridge_along_the_line(scan):
    # Bright ridge along a row; scan (summing the ridge + neighbours) peaks
    # well above the background band sum.
    img = np.zeros((40, 40))
    img[20, :] = 100.0                                  # ridge on row 20
    s = scan.integrated_intensity_scan(img, (20, 5), (20, 35), half_width=2)
    assert s.mean() == pytest.approx(100.0, rel=0.1)    # one bright row summed


def test_scan_requires_distinct_points(scan):
    with pytest.raises(ValueError):
        scan.integrated_intensity_scan(np.ones((10, 10)), (5, 5), (5, 5))


# -- roi_scans excludes non-two-spot ROIs ----------------------------------- #
def test_roi_scans_excludes_non_two_spot(scan):
    img = np.ones((50, 50))
    spots = {
        "ROI1": [(10, 5), (10, 25)],       # 2 spots -> included
        "ROI2": [(10, 10)],                # 1 spot  -> excluded
        "ROI3": [(1, 1), (2, 2), (3, 3)],  # 3 spots -> excluded
    }
    out = scan.roi_scans(img, spots, pixel_size=0.1)
    assert set(out) == {"ROI1"}
    assert out["ROI1"]["length_px"] == pytest.approx(20.0)
    assert out["ROI1"]["length_um"] == pytest.approx(2.0)
    assert out["ROI1"]["group"] == "group1"


# -- normalization ---------------------------------------------------------- #
def test_normalize_group_resamples_to_mean_length(scan):
    scans = [np.linspace(0, 1, 10), np.linspace(0, 1, 20)]
    out, target = scan.normalize_group(scans)
    assert target == 15                                 # mean(10, 20)
    assert all(len(s) == 15 for s in out)
    # Endpoints preserved by linear interpolation.
    for s_in, s_out in zip(scans, out):
        assert s_out[0] == pytest.approx(s_in[0])
        assert s_out[-1] == pytest.approx(s_in[-1])


def test_normalize_group_values_are_linear_interp(scan):
    # A ramp resampled to a longer length stays a ramp.
    ramp = np.array([0.0, 10.0])
    out, target = scan.normalize_group([ramp], target_len=5)
    assert target == 5
    assert np.allclose(out[0], np.linspace(0, 10, 5))


def test_group_and_normalize_end_to_end(scan):
    roi_results = {
        "ROI1": {"scan": np.linspace(0, 1, 10), "group": "group1"},
        "ROI2": {"scan": np.linspace(0, 1, 20), "group": "group1"},
        "ROI3": {"scan": np.linspace(0, 1, 50), "group": "group2"},
        "ROI4": {"scan": np.linspace(0, 1, 5), "group": None},   # unclassified
    }
    out = scan.group_and_normalize(roi_results)
    assert set(out) == {"group1", "group2"}
    assert out["group1"]["target_len"] == 15
    assert all(len(s) == 15 for s in out["group1"]["scans"])
    assert out["group2"]["target_len"] == 50
    assert set(out["group1"]["rois"]) == {"ROI1", "ROI2"}
