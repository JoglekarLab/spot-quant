"""Restoring a file's cells from an imported session.

The export keeps each ROI's mom/bud outlines but not the micro-sam label
image.  Reopening a session has to rebuild that image from the outlines,
otherwise the cells are invisible and roles can't be assigned without
re-segmenting.
"""
import numpy as np
import pandas as pd


def _two_rois(cells, labels, shape):
    """Two ROIs, each with its own mom/bud pair, as stored on the session."""
    roi_a = np.zeros(shape, dtype=bool); roi_a[35:125, 15:150] = True
    roi_b = np.zeros(shape, dtype=bool); roi_b[160:240, 15:150] = True
    return {
        "ROI1": cells.outlines_for_roi(labels, roi_a, {1: "mom", 2: "bud"}),
        "ROI2": cells.outlines_for_roi(labels, roi_b, {3: "mom", 4: "bud"}),
    }


def _four_cell_image():
    H, W = 280, 200
    labels = np.zeros((H, W), dtype=np.int32)
    yy, xx = np.ogrid[:H, :W]
    labels[(yy - 80) ** 2 + (xx - 60) ** 2 <= 34 ** 2] = 1
    labels[((yy - 80) ** 2 + (xx - 118) ** 2 <= 20 ** 2) & (labels == 0)] = 2
    labels[((yy - 200) ** 2 + (xx - 60) ** 2 <= 30 ** 2) & (labels == 0)] = 3
    labels[((yy - 200) ** 2 + (xx - 112) ** 2 <= 18 ** 2) & (labels == 0)] = 4
    return labels, (H, W)


def test_label_image_is_rebuilt_with_the_same_cell_ids(cells):
    labels, shape = _four_cell_image()
    per_roi = _two_rois(cells, labels, shape)

    rebuilt, roles = cells.labels_from_session(per_roi, shape)
    assert set(np.unique(rebuilt)) - {0} == {1, 2, 3, 4}
    assert roles == {1: "mom", 2: "bud", 3: "mom", 4: "bud"}

    for cell in (1, 2, 3, 4):
        a, b = labels == cell, rebuilt == cell
        assert (a & b).sum() / (a | b).sum() > 0.97


def test_restored_cells_still_assign_the_same_roles(cells):
    labels, shape = _four_cell_image()
    per_roi = _two_rois(cells, labels, shape)
    rebuilt, roles = cells.labels_from_session(per_roi, shape)

    roi_a = np.zeros(shape, dtype=bool); roi_a[35:125, 15:150] = True
    before = cells.assign_pair(labels, [(80, 60), (80, 118)], roi_mask=roi_a,
                               pixel_size=0.1)
    after = cells.assign_pair(rebuilt, [(80, 60), (80, 118)], roi_mask=roi_a,
                              pixel_size=0.1)
    assert before["spot_roles"] == after["spot_roles"] == ["mom", "bud"]
    assert after["mom_cell"] == 1 and after["daughter_cell"] == 2


def test_nothing_stored_rebuilds_an_empty_image(cells):
    labels, shape = _four_cell_image()
    empty, roles = cells.labels_from_session({}, shape)
    assert not empty.any() and roles == {}
    assert not cells.labels_from_session(None, shape)[0].any()


def test_a_cell_shared_by_two_rois_is_drawn_once(cells):
    labels, shape = _four_cell_image()
    roi = np.zeros(shape, dtype=bool); roi[35:125, 15:150] = True
    recs = cells.outlines_for_roi(labels, roi, {1: "mom", 2: "bud"})
    rebuilt, roles = cells.labels_from_session({"ROI1": recs, "ROI2": recs},
                                               shape)
    assert set(np.unique(rebuilt)) - {0} == {1, 2}
    assert roles == {1: "mom", 2: "bud"}


def test_full_round_trip_through_the_workbook(cells, session_io, pipeline,
                                              tmp_path):
    """Export the cells sheet, import it, rebuild the image, keep the roles."""
    labels, shape = _four_cell_image()
    store = {"img.nd2": _two_rois(cells, labels, shape)}

    path = tmp_path / "session.xlsx"
    with pd.ExcelWriter(path) as w:
        session_io.cells_dataframe(store).to_excel(
            w, sheet_name=session_io.CELLS_SHEET, index=False)
        pd.DataFrame([{"filename": "img.nd2", "region": "ROI1",
                       "shape_type": "rectangle", "r0": 35, "r1": 125,
                       "c0": 15, "c1": 150, "verts": "",
                       "ref_channel": "mCherry"}]).to_excel(
            w, sheet_name=session_io.ROI_SHEET, index=False)

    xls = pd.ExcelFile(path)
    data = session_io.session_from_sheets(
        xls.parse(session_io.ROI_SHEET), None, pipeline.PipelineParams,
        xls.parse(session_io.CELLS_SHEET))

    per_roi = data["session_cells"]["img.nd2"]
    assert set(per_roi) == {"ROI1", "ROI2"}

    rebuilt, roles = cells.labels_from_session(per_roi, shape)
    assert roles == {1: "mom", 2: "bud", 3: "mom", 4: "bud"}
    res = cells.roles_for_spots(per_roi["ROI1"], [(80, 60), (80, 118)],
                                pixel_size=0.1)
    assert res["spot_roles"] == ["mom", "bud"]
