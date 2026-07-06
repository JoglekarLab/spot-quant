"""Tests for the paired-channel report (pipeline.build_report)."""
import pandas as pd


def _long():
    # ROI1: 2 spots (normal). ROI2: 1 spot. ROI3: 3 spots (both abnormal).
    rows = []
    def add(region, sid, ch, integ):
        rows.append(("a.nd2", ch, region, sid, integ, 10 * sid))
    for sid in (0, 1):
        add("ROI1", sid, "GFP", 100 + sid); add("ROI1", sid, "mCherry", 50 + sid)
    add("ROI2", 0, "GFP", 200); add("ROI2", 0, "mCherry", 80)
    for sid in (0, 1, 2):
        add("ROI3", sid, "GFP", 300 + sid); add("ROI3", sid, "mCherry", 60 + sid)
    return pd.DataFrame(rows, columns=[
        "filename", "channel", "region", "spot_id",
        "integrated_intensity", "row_px"])


def test_pairs_channels_one_row_per_spot(pipeline):
    rep = pipeline.build_report(_long())
    # One row per (filename, region, spot_id): 2 + 1 + 3 = 6 spots.
    assert len(rep) == 6
    # Paired channel columns exist side by side.
    assert "GFP_integrated_intensity" in rep.columns
    assert "mCherry_integrated_intensity" in rep.columns
    r1 = rep[(rep["region"] == "ROI1") & (rep["spot_id"] == 0)].iloc[0]
    assert r1["GFP_integrated_intensity"] == 100
    assert r1["mCherry_integrated_intensity"] == 50


def test_num_spots_and_abnormal_last(pipeline):
    rep = pipeline.build_report(_long())
    ns = dict(zip(rep["region"], rep["num_spots"]))
    assert ns["ROI1"] == 2 and ns["ROI2"] == 1 and ns["ROI3"] == 3
    # The two-spot ROI is reported first; ROIs with != 2 spots come last.
    regions = list(rep["region"])
    assert regions[0] == "ROI1" and regions[1] == "ROI1"
    assert set(regions[2:]) == {"ROI2", "ROI3"}


def test_build_report_empty(pipeline):
    assert pipeline.build_report(pd.DataFrame()).empty
