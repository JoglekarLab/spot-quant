"""Session-wide (all files, all ROIs) global threshold + run_session."""
import numpy as np


def _mk(rng):
    Z, H, W = 7, 96, 96
    s = rng.normal(100, 4, (Z, H, W))
    yy, xx = np.ogrid[:H, :W]
    for y, x, zf in [(30, 40, 2), (70, 60, 5)]:
        for z in range(Z):
            a = 500 * np.exp(-((z - zf) ** 2) / (2 * 1.2 ** 2))
            s[z] += a * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 8.0)
    return s


def _files(rng):
    rois = [(22, 38, 32, 48, "ROI1"), (62, 78, 52, 68, "ROI2")]
    return [
        {"filename": "a.nd2",
         "stacks": {"GFP": _mk(rng), "mCherry": _mk(rng) * 0.6 + 3,
                    "Brightfield": _mk(rng)},
         "ref_channel": "GFP", "rois": rois, "pixel_size": 0.1,
         "pixel_unit": "um"},
        {"filename": "b.nd2",
         "stacks": {"GFP": _mk(rng) * 1.3, "mCherry": _mk(rng) * 0.6 + 3},
         "ref_channel": "GFP", "rois": rois[:1], "pixel_size": 0.1,
         "pixel_unit": "um"},
    ]


def test_session_threshold_pools_all_files(pipeline):
    from skimage.filters import threshold_otsu
    files = _files(np.random.default_rng(1))
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    thr = pipeline.compute_session_thresholds(files, params)
    assert set(thr) == {"GFP", "mCherry"}          # brightfield excluded
    # GFP threshold == Otsu over GFP top-hat pixels pooled from every ROI/file.
    pooled = []
    for f in files:
        top = pipeline._stack_tophat(f["stacks"]["GFP"], params)
        for r0, r1, c0, c1, _ in f["rois"]:
            pooled.append(top[:, r0:r1, c0:c1].ravel())
    assert thr["GFP"] == threshold_otsu(np.concatenate(pooled))


def test_run_session_applies_one_threshold_per_channel(pipeline):
    files = _files(np.random.default_rng(2))
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    table, thr = pipeline.run_session(files, params)
    assert set(table["filename"]) == {"a.nd2", "b.nd2"}
    # Every GFP row across both files carries the single session GFP threshold.
    g = table[table["channel"] == "GFP"]["roi_threshold"].round(6)
    assert set(g) == {round(thr["GFP"], 6)}
    # Rows are sorted by filename first.
    files_col = list(table["filename"])
    assert files_col == sorted(files_col)


def test_run_session_only_reference_detected(pipeline):
    # A spot-free mCherry still yields one row per GFP spot (anchored).
    rng = np.random.default_rng(3)
    files = _files(rng)
    for f in files:
        f["stacks"]["mCherry"] = rng.normal(100, 4, f["stacks"]["GFP"].shape)
    params = pipeline.PipelineParams(threshold_method="Otsu",
                                     peak_rel_threshold=0.3)
    table, _ = pipeline.run_session(files, params)
    for fname in ("a.nd2", "b.nd2"):
        sub = table[table["filename"] == fname]
        g = sub[sub["channel"] == "GFP"]
        m = sub[sub["channel"] == "mCherry"]
        assert len(g) == len(m)


def test_run_session_empty_without_rois(pipeline):
    params = pipeline.PipelineParams(threshold_method="Otsu")
    table, thr = pipeline.run_session(
        [{"filename": "a.nd2", "stacks": {"GFP": _mk(np.random.default_rng(0))},
          "ref_channel": "GFP", "rois": []}], params)
    assert table.empty
