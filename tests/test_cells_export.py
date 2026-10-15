"""Mom / bud reaching the exported table, and surviving a save-reload cycle.

Roles are re-derived from the stored cell outlines rather than carried on a
spot id, so they stay attached to the right dot even when the export re-run
finds a different set of spots.
"""
import numpy as np
import pandas as pd


def _cells_for(cells, budding_pair):
    labels, roi, info = budding_pair
    recs = cells.outlines_for_roi(labels, roi,
                                  {info["mom"]: "mom", info["bud"]: "bud"})
    return {"f1.nd2": {"ROI1": recs}}, info


def _table(info, spot_ids=(0, 1), positions=None):
    """Two spots, two channels, one ROI -- the shape run_session produces."""
    pos = positions or [info["mom_centre"], info["bud_centre"]]
    rows = []
    for sid, (r, c) in zip(spot_ids, pos):
        for ch in ("mCherry", "FITC"):
            rows.append({"filename": "f1.nd2", "region": "ROI1", "channel": ch,
                         "spot_id": sid, "row_px": r, "col_px": c,
                         "integrated_intensity": 100.0 + sid})
    return pd.DataFrame(rows)


def test_roles_land_on_every_channel_of_the_spot(cells, budding_pair):
    store, info = _cells_for(cells, budding_pair)
    out = cells.apply_roles(_table(info), store,
                            ref_by_file={"f1.nd2": "mCherry"},
                            pixel_by_file={"f1.nd2": 0.1})
    assert set(cells.ROLE_COLUMNS) <= set(out.columns)
    assert list(out[out.spot_id == 0]["spot_role"]) == ["mom", "mom"]
    assert list(out[out.spot_id == 1]["spot_role"]) == ["bud", "bud"]
    assert (out["pair_type"] == cells.PAIR_SPLIT).all()
    assert out["dist_to_bud_um"].notna().all()


def test_roles_follow_position_not_spot_id(cells, budding_pair):
    """The re-run may renumber spots; the role must track the dot."""
    store, info = _cells_for(cells, budding_pair)
    swapped = _table(info, spot_ids=(7, 3),
                     positions=[info["bud_centre"], info["mom_centre"]])
    out = cells.apply_roles(swapped, store,
                            ref_by_file={"f1.nd2": "mCherry"},
                            pixel_by_file={"f1.nd2": 0.1})
    assert set(out[out.spot_id == 7]["spot_role"]) == {"bud"}
    assert set(out[out.spot_id == 3]["spot_role"]) == {"mom"}


def test_toward_bud_reaches_the_table(cells, budding_pair):
    store, info = _cells_for(cells, budding_pair)
    tbl = _table(info, positions=[(80, 88), (80, 35)])
    out = cells.apply_roles(tbl, store, ref_by_file={"f1.nd2": "mCherry"},
                            pixel_by_file={"f1.nd2": 0.1})
    assert set(out[out.spot_id == 0]["spot_role"]) == {"toward_bud"}
    assert set(out[out.spot_id == 1]["spot_role"]) == {"mom"}
    assert (out["pair_type"] == cells.PAIR_BOTH_IN_MOM).all()
    near = out[out.spot_id == 0]["dist_to_bud_um"].iloc[0]
    far = out[out.spot_id == 1]["dist_to_bud_um"].iloc[0]
    assert near < far


def test_an_roi_without_two_spots_is_blank_with_a_reason(cells, budding_pair):
    store, info = _cells_for(cells, budding_pair)
    tbl = _table(info)
    extra = tbl[tbl.spot_id == 0].copy()
    extra["spot_id"] = 2
    out = cells.apply_roles(pd.concat([tbl, extra], ignore_index=True), store,
                            ref_by_file={"f1.nd2": "mCherry"},
                            pixel_by_file={"f1.nd2": 0.1})
    assert (out["spot_role"] == "").all()
    assert "exactly two spots" in out["role_note"].iloc[0]


def test_an_roi_with_no_stored_cells_is_blank_with_a_reason(cells, budding_pair):
    _, info = _cells_for(cells, budding_pair)
    out = cells.apply_roles(_table(info), {},
                            ref_by_file={"f1.nd2": "mCherry"},
                            pixel_by_file={"f1.nd2": 0.1})
    assert (out["spot_role"] == "").all()
    assert "no segmented cells" in out["role_note"].iloc[0]


def test_an_empty_table_still_gets_the_columns(cells):
    out = cells.apply_roles(pd.DataFrame(), {})
    assert set(cells.ROLE_COLUMNS) <= set(out.columns)


def test_cells_sheet_round_trips_through_a_workbook(cells, session_io,
                                                    budding_pair, tmp_path):
    store, info = _cells_for(cells, budding_pair)
    df = session_io.cells_dataframe(store)
    assert len(df) == 2 and set(df["role"]) == {"mom", "bud"}

    path = tmp_path / "session.xlsx"
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name=session_io.CELLS_SHEET, index=False)
    back = session_io.cells_from_sheet(
        pd.ExcelFile(path).parse(session_io.CELLS_SHEET))

    assert set(back) == {"f1.nd2"} and set(back["f1.nd2"]) == {"ROI1"}
    recs = back["f1.nd2"]["ROI1"]
    assert [r["role"] for r in recs] == ["mom", "bud"]
    # And the reloaded outlines still produce the same roles.
    res = cells.roles_for_spots(recs, [info["mom_centre"], info["bud_centre"]],
                                pixel_size=0.1, gap_um=0.3)
    assert res["spot_roles"] == ["mom", "bud"]


def test_untagged_cells_are_not_saved(cells, session_io, budding_pair):
    """No tag, no row -- the sheet stays at two rows per region."""
    labels, roi, _ = budding_pair
    recs = cells.outlines_for_roi(labels, roi, {})       # nothing tagged
    assert recs == []
    assert session_io.cells_dataframe({"f1.nd2": {"ROI1": recs}}).empty


def test_the_sheet_holds_at_most_two_rows_per_region(cells, session_io,
                                                     budding_pair):
    store, _ = _cells_for(cells, budding_pair)
    df = session_io.cells_dataframe(store)
    assert df.groupby(["filename", "region"]).size().max() <= 2
    assert list(df["role"]) == ["mom", "bud"]


def test_session_from_sheets_picks_up_the_cells_sheet(cells, session_io,
                                                      pipeline, budding_pair):
    store, _ = _cells_for(cells, budding_pair)
    roi_df = pd.DataFrame([{"filename": "f1.nd2", "region": "ROI1",
                            "shape_type": "rectangle", "r0": 0, "r1": 10,
                            "c0": 0, "c1": 10, "verts": "",
                            "ref_channel": "mCherry"}])
    data = session_io.session_from_sheets(
        roi_df, None, pipeline.PipelineParams,
        session_io.cells_dataframe(store))
    assert "f1.nd2" in data["session_cells"]
    assert len(data["session_cells"]["f1.nd2"]["ROI1"]) == 2


def test_report_keeps_one_role_column_not_one_per_channel(cells, pipeline,
                                                          budding_pair):
    store, info = _cells_for(cells, budding_pair)
    tagged = cells.apply_roles(_table(info), store,
                               ref_by_file={"f1.nd2": "mCherry"},
                               pixel_by_file={"f1.nd2": 0.1})
    rep = pipeline.build_report(tagged)
    assert "spot_role" in rep.columns
    assert "mCherry_spot_role" not in rep.columns
    assert "FITC_spot_role" not in rep.columns
    assert "mCherry_integrated_intensity" in rep.columns   # still per channel
    assert sorted(rep["spot_role"]) == ["bud", "mom"]
