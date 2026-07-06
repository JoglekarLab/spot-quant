"""Session-global ROI labelling (pipeline.next_roi_labels)."""


def test_labels_increment_across_files(pipeline):
    # File A: two ROIs -> ROI1, ROI2.
    a_bounds = [(0, 10, 0, 10), (20, 30, 20, 30)]
    a, counter = pipeline.next_roi_labels({}, a_bounds, 0)
    assert [x[4] for x in a] == ["ROI1", "ROI2"]
    assert counter == 2

    # File B: two new ROIs continue the count -> ROI3, ROI4 (never repeat).
    b_bounds = [(0, 10, 0, 10), (40, 50, 40, 50)]
    b, counter = pipeline.next_roi_labels({}, b_bounds, counter)
    assert [x[4] for x in b] == ["ROI3", "ROI4"]
    assert counter == 4


def test_existing_labels_preserved_by_bounds(pipeline):
    prior = {(0, 10, 0, 10): "ROI1", (20, 30, 20, 30): "ROI2"}
    # Re-visit the file, one ROI unchanged + one new rectangle.
    bounds = [(0, 10, 0, 10), (60, 70, 60, 70)]
    out, counter = pipeline.next_roi_labels(prior, bounds, 2)
    assert out[0][4] == "ROI1"       # preserved
    assert out[1][4] == "ROI3"       # new, continues global count
    assert counter == 3


def test_labels_are_bounds_tuples(pipeline):
    out, _ = pipeline.next_roi_labels({}, [(1, 2, 3, 4)], 0)
    assert out[0] == (1, 2, 3, 4, "ROI1")
