"""Removing detected spots, and saving added/removed spots in the session."""
import numpy as np
import pandas as pd


def _rois(spots):
    return [(s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, f"ROI{i + 1}")
            for i, s in enumerate(spots)]


def _params(pipeline):
    return pipeline.PipelineParams(threshold_method="Otsu",
                                   peak_rel_threshold=0.3)


def test_drop_removed_spots_matches_region_and_radius(pipeline):
    df = pd.DataFrame({"region": ["ROI1", "ROI1", "ROI2"],
                       "row_px": [10, 40, 10], "col_px": [10, 40, 10]})
    out = pipeline.drop_removed_spots(df, {"ROI1": [(11, 9)]}, radius=3)
    # Only the ROI1 spot near (11, 9) goes; ROI2's spot at the same place stays.
    assert list(zip(out["region"], out["row_px"])) == [("ROI1", 40), ("ROI2", 10)]
    assert pipeline.drop_removed_spots(df, None, 3) is df


def test_drop_removed_centroids(pipeline):
    cents = np.array([[0, 10, 10], [1, 40, 40]])
    out = pipeline.drop_removed_centroids(cents, {"ROI1": [(10, 12)]}, 3)
    assert out.tolist() == [[1, 40, 40]]


def test_removed_spot_is_dropped_in_every_channel(pipeline, spotty_stack):
    stack, spots = spotty_stack
    stacks = {"GFP": stack, "mCherry": stack * 0.5 + 5}
    params = _params(pipeline)
    rois = _rois(spots)
    full = pipeline.run_multichannel(stacks, "GFP", params, rois=rois)
    ref = full.measurements[full.measurements["channel"] == "GFP"]
    target = ref.iloc[0]
    region = str(target["region"])
    removed = {region: [(int(target["row_px"]), int(target["col_px"]))]}

    res = pipeline.run_multichannel(stacks, "GFP", params, rois=rois,
                                    removed_spots=removed)
    m = res.measurements
    for ch in ("GFP", "mCherry"):
        before = full.measurements[(full.measurements["channel"] == ch)
                                   & (full.measurements["region"] == region)]
        after = m[(m["channel"] == ch) & (m["region"] == region)]
        assert len(after) == len(before) - 1
    assert len(res.centroids) == len(full.centroids) - 1
    # Other ROIs are untouched.
    others = full.measurements[full.measurements["region"] != region]
    assert len(m[m["region"] != region]) == len(others)


def test_removed_spot_stays_removed_with_global_threshold(pipeline,
                                                          spotty_stack):
    stack, spots = spotty_stack
    rois = _rois(spots)
    params = _params(pipeline)
    files = [{"filename": "a.nd2", "stacks": {"GFP": stack},
              "ref_channel": "GFP", "rois": rois}]
    table, _ = pipeline.run_session(files, params)
    first = table.iloc[0]
    files[0]["removed_spots"] = {
        str(first["region"]): [(int(first["row_px"]), int(first["col_px"]))]}
    table2, _ = pipeline.run_session(files, params)
    assert len(table2) == len(table) - 1


def test_manual_spot_is_never_dropped(pipeline, spotty_stack):
    stack, spots = spotty_stack
    rois = _rois(spots)
    s = spots[0]
    pt = (int(s["y"]), int(s["x"]))
    res = pipeline.run_multichannel(
        {"GFP": stack}, "GFP", _params(pipeline), rois=rois,
        manual_spots={"ROI1": [pt]}, removed_spots={"ROI1": [pt]})
    man = res.measurements[res.measurements["n_planes_linked"] == 0]
    assert len(man) == 1


def test_spot_edits_round_trip(session_io, pipeline):
    manual = {"a.nd2": {"ROI1": [(5, 6)]}}
    removed = {"a.nd2": {"ROI1": [(20, 21), (30, 31)]},
               "b.nd2": {"ROI4": [(1, 2)]}}
    df = session_io.spot_edits_dataframe(manual, removed)
    assert list(df.columns) == ["filename", "region", "edit", "row_px",
                                "col_px"]
    assert sorted(df["edit"]) == ["added", "removed", "removed", "removed"]
    m2, r2 = session_io.spot_edits_from_sheet(df)
    assert m2 == manual
    assert r2 == removed


def test_session_from_sheets_restores_spot_edits(session_io, pipeline):
    shapes = {"a.nd2": [{"bounds": (0, 50, 0, 50), "verts": None,
                         "label": "ROI1"}]}
    roi_df = session_io.rois_dataframe(shapes, {"a.nd2": "GFP"})
    edits = session_io.spot_edits_dataframe({}, {"a.nd2": {"ROI1": [(9, 9)]}})
    data = session_io.session_from_sheets(roi_df, None,
                                          pipeline.PipelineParams,
                                          None, edits)
    assert data["removed_spots"] == {"a.nd2": {"ROI1": [(9, 9)]}}
    assert data["manual_spots"] == {}
    # Old exports without the sheet still import, with no edits.
    old = session_io.session_from_sheets(roi_df, None, pipeline.PipelineParams)
    assert old["manual_spots"] == {} and old["removed_spots"] == {}
