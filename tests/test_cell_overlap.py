"""A cell must really sit in the ROI before it can be its mother or bud.

Overlap is counted in absolute pixels, not as a fraction: an ROI is drawn
around the spot pair and is usually smaller than a whole cell, so most of a
real mother or bud lies outside the box and a percentage would penalise
exactly the cells that matter.
"""
import numpy as np
import pytest


@pytest.fixture
def pair_plus_sliver():
    """Mother + bud well inside the ROI, and a neighbour barely clipping it."""
    H = W = 160
    labels = np.zeros((H, W), dtype=np.int32)
    yy, xx = np.ogrid[:H, :W]
    labels[(yy - 80) ** 2 + (xx - 60) ** 2 <= 34 ** 2] = 1        # mother
    labels[((yy - 80) ** 2 + (xx - 118) ** 2 <= 20 ** 2)
           & (labels == 0)] = 2                                   # bud
    labels[((yy - 20) ** 2 + (xx - 20) ** 2 <= 25 ** 2)
           & (labels == 0)] = 3                                   # neighbour
    roi = np.zeros((H, W), dtype=bool)
    roi[35:125, 15:150] = True          # cell 3 only just reaches the corner
    return labels, roi


def test_overlap_counts_are_pixels_inside_the_roi(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    assert set(counts) == {1, 2, 3}
    assert counts[1] > counts[2] > counts[3]
    assert counts[3] < 0.2 * (labels == 3).sum()   # only a corner is inside
    assert counts[1] == int(((labels == 1) & roi).sum())


def test_threshold_filters_the_sliver_out(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    assert set(cells.cells_touching_roi(labels, roi, 0)) == {1, 2, 3}
    cut = counts[3] + 1
    assert set(cells.cells_touching_roi(labels, roi, cut)) == {1, 2}
    assert cells.cells_touching_roi(labels, roi, 10 ** 6) == []


def test_touching_is_ordered_by_overlap(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    assert cells.cells_touching_roi(labels, roi, 0) == [1, 2, 3]


def test_pair_by_size_ignores_the_sliver(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    # Without a threshold the sliver is a candidate but loses on area anyway.
    assert cells.pair_by_size(labels, roi, min_overlap=counts[3] + 1) == (1, 2)


def test_pair_by_size_needs_two_qualifying_cells(cells, pair_plus_sliver):
    """One cell in the ROI means no pair, so nothing is named."""
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    cut = counts[2] + 1                          # only the mother clears it
    assert cells.pair_by_size(labels, roi, min_overlap=cut) == (None, None)


def test_pair_by_size_with_no_cells_at_all(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    empty = np.zeros_like(roi)
    assert cells.pair_by_size(labels, empty) == (None, None)


def test_a_split_pair_is_refused_when_a_cell_barely_clips(cells,
                                                          pair_plus_sliver):
    """One dot in the mother, one in the barely-overlapping neighbour."""
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    spots = [(80, 60), (38, 30)]                 # second dot is in cell 3
    assert cells.cell_at(labels, *spots[1]) == 3

    loose = cells.assign_pair(labels, spots, roi_mask=roi, min_overlap=0)
    assert loose["ok"] and loose["spot_roles"] == ["mom", "bud"]

    strict = cells.assign_pair(labels, spots, roi_mask=roi,
                               min_overlap=counts[3] + 1)
    assert not strict["ok"]
    assert strict["spot_roles"] == ["", ""]
    assert "under" in strict["reason"] and "px inside the ROI" in strict["reason"]


def test_a_genuine_pair_still_passes_the_threshold(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    res = cells.assign_pair(labels, [(80, 60), (80, 118)], roi_mask=roi,
                            pixel_size=0.1, min_overlap=counts[3] + 1)
    assert res["ok"]
    assert res["spot_roles"] == ["mom", "bud"]
    assert res["pair_type"] == cells.PAIR_SPLIT


def test_both_in_mom_refuses_when_the_bud_only_clips(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    both = [(80, 45), (80, 75)]                  # both dots inside the mother
    cut = counts[2] + 1                          # excludes the real bud too
    res = cells.assign_pair(labels, both, roi_mask=roi, pixel_size=0.1,
                            min_overlap=cut)
    assert not res["ok"]
    assert res["needs_daughter"] is True
    assert "no bud overlapping this ROI" in res["reason"]


def test_outlines_drop_the_sliver_and_keep_the_pair(cells, pair_plus_sliver):
    labels, roi = pair_plus_sliver
    counts = cells.overlap_counts(labels, roi)
    recs = cells.outlines_for_roi(labels, roi, {1: "mom", 2: "bud", 3: "bud"},
                                  min_overlap=counts[3] + 1)
    assert [(r["cell"], r["role"]) for r in recs] == [(1, "mom"), (2, "bud")]


def test_default_threshold_constant_is_exposed(cells):
    assert cells.DEFAULT_MIN_OVERLAP_PX == 50


# --------------------------------------------------------------------------- #
# Regression: a big neighbour clipping the box must not win the bud slot
# --------------------------------------------------------------------------- #
@pytest.fixture
def bud_vs_big_neighbour():
    """The real bud is small but fully inside; the neighbour is huge but barely in.

    Ranking candidates by whole-cell area picks the neighbour; ranking by how
    much of each cell is inside the ROI picks the bud, which is correct.
    """
    H = W = 220
    labels = np.zeros((H, W), dtype=np.int32)
    yy, xx = np.ogrid[:H, :W]
    labels[(yy - 100) ** 2 + (xx - 70) ** 2 <= 30 ** 2] = 1        # mother
    labels[((yy - 100) ** 2 + (xx - 120) ** 2 <= 18 ** 2)
           & (labels == 0)] = 2                                    # real bud
    labels[((yy - 100) ** 2 + (xx - 205) ** 2 <= 55 ** 2)
           & (labels == 0)] = 3                                    # big neighbour
    roi = np.zeros((H, W), dtype=bool)
    roi[60:140, 35:155] = True
    return labels, roi


def test_the_real_bud_beats_a_bigger_neighbour(cells, bud_vs_big_neighbour):
    labels, roi = bud_vs_big_neighbour
    areas = cells.cell_areas(labels)
    overlap = cells.overlap_counts(labels, roi)

    # The premise: neighbour is the bigger cell, but the bud is more inside.
    assert areas[3] > areas[2]
    assert overlap[2] > overlap[3]

    mom, bud = cells.pair_by_size(labels, roi, min_overlap=0)
    assert (mom, bud) == (1, 2), "the bud must win on overlap, not lose on area"


def test_the_bud_also_wins_when_both_dots_are_in_the_mother(
        cells, bud_vs_big_neighbour):
    labels, roi = bud_vs_big_neighbour
    found = cells.other_cell_in_roi(labels, roi, exclude=1, min_overlap=0)
    assert found == 2


def test_the_mother_is_still_the_larger_of_the_chosen_two(cells):
    """Selection is by overlap; mom vs bud is still decided by total size."""
    H = W = 200
    labels = np.zeros((H, W), dtype=np.int32)
    yy, xx = np.ogrid[:H, :W]
    labels[(yy - 100) ** 2 + (xx - 60) ** 2 <= 34 ** 2] = 1        # bigger
    labels[((yy - 100) ** 2 + (xx - 120) ** 2 <= 20 ** 2)
           & (labels == 0)] = 2                                    # smaller
    # An ROI shifted onto the small cell, so IT has the most overlap.
    roi = np.zeros((H, W), dtype=bool)
    roi[80:120, 85:145] = True
    overlap = cells.overlap_counts(labels, roi)
    assert overlap[1] > 0 and overlap[2] > overlap[1]   # small cell overlaps more
    mom, bud = cells.pair_by_size(labels, roi, min_overlap=0)
    assert (mom, bud) == (1, 2)                    # but the bigger is the mother
