"""Tests for multi-file session compilation and the Z-step metadata field."""
import pandas as pd

from conftest import load_module


def test_tag_filename_adds_column(pipeline):
    df = pd.DataFrame({"spot_id": [0, 1], "region": ["ROI1", "ROI1"]})
    out = pipeline.tag_filename(df, "img1.nd2")
    assert list(out.columns)[0] == "filename"
    assert (out["filename"] == "img1.nd2").all()
    # Original is untouched.
    assert "filename" not in df.columns


def test_tag_filename_is_idempotent(pipeline):
    df = pd.DataFrame({"filename": ["x"], "spot_id": [0]})
    out = pipeline.tag_filename(df, "y")
    assert (out["filename"] == "x").all()      # existing column preserved


def test_compile_records_concatenates_all_files(pipeline):
    records = {
        "a.nd2": pipeline.tag_filename(
            pd.DataFrame({"spot_id": [0, 1], "region": ["ROI1", "ROI2"]}), "a.nd2"),
        "b.nd2": pipeline.tag_filename(
            pd.DataFrame({"spot_id": [0], "region": ["ROI1"]}), "b.nd2"),
    }
    table = pipeline.compile_records(records)
    assert len(table) == 3
    assert set(table["filename"]) == {"a.nd2", "b.nd2"}
    # Long-format: one flat table easy to re-import.
    assert {"filename", "spot_id", "region"} <= set(table.columns)


def test_compile_records_empty(pipeline):
    assert pipeline.compile_records({}).empty


# -- Z-step metadata -------------------------------------------------------- #
def test_imagemeta_has_z_step_default():
    meta = load_module("metadata")
    m = meta.ImageMeta()
    assert m.z_step == meta.DEFAULT_Z_STEP
    assert "z_step" in m.as_dict()
    # Freshly constructed metadata is not "known" until read from a file.
    assert m.z_step_known is False
    assert m.time_step_known is False


def test_min_link_for_zstep(pipeline):
    import math
    # ceil(1000 / step_nm), step_nm = z_step_um * 1000  ==  ceil(1 / z_step_um)
    assert pipeline.min_link_for_zstep(0.5) == 2      # 500 nm -> ceil(2)
    assert pipeline.min_link_for_zstep(0.25) == 4     # 250 nm -> ceil(4)
    assert pipeline.min_link_for_zstep(0.2) == 5      # 200 nm -> ceil(5)
    assert pipeline.min_link_for_zstep(0.3) == math.ceil(1000 / 300)  # -> 4
    assert pipeline.min_link_for_zstep(1.0) == 1      # 1000 nm -> 1
    assert pipeline.min_link_for_zstep(0.0) == 1      # guard
