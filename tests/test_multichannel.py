"""Tests for multi-channel measurement and brightfield exclusion."""


def _rois(spots):
    return [(s["y"] - 8, s["y"] + 8, s["x"] - 8, s["x"] + 8, f"ROI{i + 1}")
            for i, s in enumerate(spots)]


def test_is_measurable_channel(pipeline):
    assert pipeline.is_measurable_channel("GFP")
    assert pipeline.is_measurable_channel("mCherry")
    assert not pipeline.is_measurable_channel("Brightfield")
    assert not pipeline.is_measurable_channel("Trans")
    assert not pipeline.is_measurable_channel("Phase contrast")


def test_multichannel_excludes_brightfield(pipeline, spotty_stack):
    stack, spots = spotty_stack
    stacks = {
        "GFP": stack,
        "mCherry": stack * 0.5 + 5,
        "Brightfield": stack,
    }
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    res = pipeline.run_multichannel(stacks, "GFP", params, rois=_rois(spots))
    assert set(res.per_channel) == {"GFP", "mCherry"}
    assert "Brightfield" not in res.per_channel
    assert set(res.measurements["channel"]) <= {"GFP", "mCherry"}


def test_multichannel_has_channel_column_and_is_sorted(pipeline, spotty_stack):
    stack, spots = spotty_stack
    stacks = {"GFP": stack, "mCherry": stack * 0.5 + 5}
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    res = pipeline.run_multichannel(stacks, "GFP", params, rois=_rois(spots))
    assert "channel" in res.measurements.columns
    nums = [int(r[3:]) for r in res.measurements["region"]]
    assert nums == sorted(nums)                       # sorted by ROI number


def test_multichannel_reference_drives_annotation(pipeline, spotty_stack):
    stack, spots = spotty_stack
    stacks = {"GFP": stack, "mCherry": stack * 0.5 + 5}
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    res = pipeline.run_multichannel(stacks, "mCherry", params, rois=_rois(spots))
    # Annotation centroids come from the chosen reference channel.
    assert len(res.centroids) == len(res.per_channel["mCherry"].centroids)


def test_nonref_measured_at_ref_spots(pipeline, spotty_stack):
    import numpy as np
    stack, spots = spotty_stack
    # mCherry has NO spots of its own (just noise); only the reference (GFP)
    # carries spots. Reference-anchored measurement still yields one mCherry
    # row per GFP spot at the same coordinates.
    rng = np.random.default_rng(7)
    flat = rng.normal(100, 4, stack.shape)
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    res = pipeline.run_multichannel({"GFP": stack, "mCherry": flat},
                                    "GFP", params, rois=_rois(spots))
    g = res.measurements[res.measurements["channel"] == "GFP"]
    m = res.measurements[res.measurements["channel"] == "mCherry"]
    assert len(g) == len(m) == len(spots)        # not independently detected
    gi = g.set_index("spot_id"); mi = m.set_index("spot_id")
    for sid in gi.index:
        assert mi.loc[sid, "row_px"] == gi.loc[sid, "row_px"]
        assert mi.loc[sid, "col_px"] == gi.loc[sid, "col_px"]


def test_chromatic_offset_shifts_nonref(pipeline, spotty_stack):
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3,
                                     offset_x=3, offset_y=-2, offset_z=1)
    res = pipeline.run_multichannel({"GFP": stack, "mCherry": stack * 0.6 + 3},
                                    "GFP", params, rois=_rois(spots))
    g = res.measurements[res.measurements["channel"] == "GFP"].set_index("spot_id")
    m = res.measurements[res.measurements["channel"] == "mCherry"].set_index("spot_id")
    for sid in g.index:
        assert m.loc[sid, "col_px"] == g.loc[sid, "col_px"] + 3
        assert m.loc[sid, "row_px"] == g.loc[sid, "row_px"] - 2
        assert m.loc[sid, "z_plane"] == g.loc[sid, "z_plane"] + 1


def test_offset_default_zero_keeps_positions(pipeline, spotty_stack):
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    assert (params.offset_x, params.offset_y, params.offset_z) == (0, 0, 0)
    res = pipeline.run_multichannel({"GFP": stack, "mCherry": stack * 0.6 + 3},
                                    "GFP", params, rois=_rois(spots))
    g = res.measurements[res.measurements["channel"] == "GFP"].set_index("spot_id")
    m = res.measurements[res.measurements["channel"] == "mCherry"].set_index("spot_id")
    for sid in g.index:
        assert m.loc[sid, "col_px"] == g.loc[sid, "col_px"]
        assert m.loc[sid, "z_plane"] == g.loc[sid, "z_plane"]


def test_multichannel_raises_without_measurable_channel(pipeline, spotty_stack):
    import pytest
    stack, spots = spotty_stack
    params = pipeline.PipelineParams(threshold_method="Otsu")
    with pytest.raises(ValueError):
        pipeline.run_multichannel({"Brightfield": stack}, "Brightfield",
                                  params, rois=_rois(spots))
