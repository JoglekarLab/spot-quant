"""Mom / bud roles: stored outlines, and roles recomputed from spot positions.

The export re-runs detection with a session-wide threshold, so it can find a
different set of spots and number them differently.  Roles therefore cannot be
saved against a spot id -- they are re-derived from each dot's position using
the cell outlines kept for its ROI.
"""
import numpy as np


def test_outline_round_trips_to_the_same_cell(cells, budding_pair):
    labels, _, info = budding_pair
    verts = cells.cell_outline(labels, info["mom"])
    assert verts is not None and len(verts) >= 3
    rebuilt, _ = cells.labels_from_outlines(
        [{"role": "mom", "verts": verts}], labels.shape)
    original = labels == info["mom"]
    got = rebuilt == 1
    overlap = (original & got).sum() / (original | got).sum()
    assert overlap > 0.97                      # same cell, a few boundary px


def test_only_the_tagged_pair_is_kept(cells, budding_pair):
    """An ROI is one dividing cell, so only its mother and bud are stored."""
    labels, roi, info = budding_pair
    # A third cell clipping the corner of the ROI, and one fully outside it.
    yy, xx = np.ogrid[:160, :160]
    labels = labels.copy()
    labels[((yy - 40) ** 2 + (xx - 25) ** 2 <= 14 ** 2) & (labels == 0)] = 3
    labels[((yy - 150) ** 2 + (xx - 150) ** 2 <= 8 ** 2) & (labels == 0)] = 4

    assert set(cells.cells_touching_roi(labels, roi)) == {1, 2, 3}

    recs = cells.outlines_for_roi(labels, roi, {1: "mom", 2: "bud"})
    assert len(recs) == 2                               # never more than two
    assert [r["cell"] for r in recs] == [1, 2]          # mother first, then bud
    assert [r["role"] for r in recs] == ["mom", "bud"]
    assert all(r["verts"] is not None for r in recs)
    assert recs[0]["area"] > recs[1]["area"]


def test_nothing_is_stored_when_no_pair_is_tagged(cells, budding_pair):
    labels, roi, _ = budding_pair
    assert cells.outlines_for_roi(labels, roi, {}) == []


def test_only_one_of_each_role_is_kept(cells, budding_pair):
    """Two cells tagged 'mom' in one ROI: the larger one wins."""
    labels, roi, _ = budding_pair
    recs = cells.outlines_for_roi(labels, roi, {1: "mom", 2: "mom"})
    assert len(recs) == 1
    assert recs[0]["cell"] == 1                         # the bigger cell


def test_tags_from_other_rois_are_ignored(cells, budding_pair):
    """cell_roles is image-wide; only cells touching THIS ROI may be used."""
    labels, roi, info = budding_pair
    elsewhere = np.zeros_like(roi)
    elsewhere[0:20, 0:20] = True                        # nowhere near the pair
    assert cells.outlines_for_roi(labels, elsewhere, {1: "mom", 2: "bud"}) == []


def test_an_explicit_pair_overrides_the_tags(cells, budding_pair):
    labels, roi, _ = budding_pair
    recs = cells.outlines_for_roi(labels, roi, {1: "mom", 2: "bud"},
                                  mom_cell=2, daughter_cell=1)
    assert [(r["cell"], r["role"]) for r in recs] == [(2, "mom"), (1, "bud")]


def _records(cells, budding_pair, tagged=True):
    labels, roi, info = budding_pair
    roles = {info["mom"]: "mom", info["bud"]: "bud"} if tagged else {}
    return cells.outlines_for_roi(labels, roi, roles), info


def test_split_pair_is_mom_and_bud(cells, budding_pair):
    recs, info = _records(cells, budding_pair)
    res = cells.roles_for_spots(recs, [info["mom_centre"], info["bud_centre"]],
                                pixel_size=0.1, gap_um=0.3)
    assert res["ok"]
    assert res["spot_roles"] == ["mom", "bud"]
    assert res["pair_type"] == cells.PAIR_SPLIT
    assert res["distances_um"] is not None


def test_both_in_mother_one_clearly_nearer_the_bud(cells, budding_pair):
    recs, info = _records(cells, budding_pair)
    near = (80, 88)                            # inside mom, close to the bud
    far = (80, 35)                             # inside mom, far from the bud
    res = cells.roles_for_spots(recs, [near, far], pixel_size=0.1, gap_um=0.3)
    assert res["ok"]
    assert res["spot_roles"] == ["toward_bud", "mom"]
    assert res["pair_type"] == cells.PAIR_BOTH_IN_MOM
    d = res["distances_um"]
    assert d[0] < d[1]


def test_role_order_follows_the_dots_not_their_listing(cells, budding_pair):
    recs, info = _records(cells, budding_pair)
    near, far = (80, 88), (80, 35)
    a = cells.roles_for_spots(recs, [near, far], pixel_size=0.1, gap_um=0.3)
    b = cells.roles_for_spots(recs, [far, near], pixel_size=0.1, gap_um=0.3)
    assert a["spot_roles"] == ["toward_bud", "mom"]
    assert b["spot_roles"] == ["mom", "toward_bud"]


def test_both_in_mother_but_too_close_to_call(cells, budding_pair):
    recs, info = _records(cells, budding_pair)
    # Two dots the same distance from the bud -> neither is "toward" it.
    res = cells.roles_for_spots(recs, [(66, 60), (94, 60)],
                                pixel_size=0.1, gap_um=0.3)
    assert res["ok"]
    assert res["spot_roles"] == ["mom", "mom"]
    assert "neither clearly nearer" in res["reason"]


def test_both_in_mother_with_no_bud_is_not_scored(cells, budding_pair):
    """An ROI only counts when a mother AND a bud are there."""
    labels, roi, info = budding_pair
    mom_only = np.where(labels == info["mom"], labels, 0)
    recs = cells.outlines_for_roi(mom_only, roi, {info["mom"]: "mom"})
    res = cells.roles_for_spots(recs, [(80, 88), (80, 35)],
                                pixel_size=0.1, gap_um=0.3)
    assert res["spot_roles"] == ["", ""]
    assert not res["ok"]
    assert "no mother-and-bud pair" in res["reason"]


def test_assign_pair_alone_also_refuses_without_a_bud(cells, budding_pair):
    labels, roi, info = budding_pair
    mom_only = np.where(labels == info["mom"], labels, 0)
    res = cells.assign_pair(mom_only, [(80, 88), (80, 35)], roi_mask=roi,
                            pixel_size=0.1, gap_um=0.3)
    assert res["spot_roles"] == ["", ""]
    assert not res["ok"]
    assert res["needs_daughter"] is True            # "Pick bud" can fix it
    assert "no bud overlapping this ROI" in res["reason"]


def test_an_untagged_pair_stores_nothing_so_roles_stay_blank(cells,
                                                             budding_pair):
    """With no mom/bud tag there is nothing to store, and no role to give."""
    recs, info = _records(cells, budding_pair, tagged=False)
    assert recs == []
    res = cells.roles_for_spots(recs, [info["mom_centre"], info["bud_centre"]],
                                pixel_size=0.1, gap_um=0.3)
    assert res["spot_roles"] == ["", ""]
    assert "no segmented cells" in res["reason"]


def test_unresolvable_cases_report_why(cells, budding_pair):
    recs, info = _records(cells, budding_pair)
    one = cells.roles_for_spots(recs, [info["mom_centre"]])
    assert one["spot_roles"] == [""] and "exactly two" in one["reason"]

    none_stored = cells.roles_for_spots([], [(10, 10), (20, 20)])
    assert none_stored["spot_roles"] == ["", ""]
    assert "no segmented cells" in none_stored["reason"]

    outside = cells.roles_for_spots(recs, [(5, 5), (6, 6)])
    assert outside["spot_roles"] == ["", ""]
    assert not outside["ok"]
