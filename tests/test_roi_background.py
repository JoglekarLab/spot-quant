"""Background ring and boundary rejection follow the ROI's real outline.

For a polygon or a tilted box the upright bounding box can be twice the ROI's
area, so a bounds-only boundary test lets through spots whose disk hangs
outside, and an unclipped ring takes its background from whatever sits beyond
the ROI -- often the neighbouring cell.
"""
import numpy as np


def _tilted(cy, cx, length, width, deg):
    t = np.radians(deg)
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-u[1], u[0]])
    c = np.array([float(cy), float(cx)])
    return np.array([c - u * length / 2 - v * width / 2,
                     c + u * length / 2 - v * width / 2,
                     c + u * length / 2 + v * width / 2,
                     c - u * length / 2 + v * width / 2])


def test_region_masks_covers_rectangles_and_polygons(pipeline):
    verts = _tilted(100, 100, 80, 24, 45)
    rois = [(0, 20, 0, 20, "ROI1"), (60, 140, 60, 140, "ROI2")]
    masks = {"ROI2": pipeline.roi_mask(None, verts, (200, 200))}
    out = pipeline.region_masks(rois, masks, (200, 200))
    assert set(out) == {"ROI1", "ROI2"}
    assert out["ROI1"].sum() == 400                  # built from the rectangle
    assert out["ROI2"] is masks["ROI2"]              # polygon passed through


def test_boundary_uses_the_mask_not_the_bounding_box(pipeline):
    verts = _tilted(100, 100, 80, 24, 45)
    mask = pipeline.roi_mask(None, verts, (200, 200))
    r0, r1 = int(verts[:, 0].min()), int(verts[:, 0].max())
    c0, c1 = int(verts[:, 1].min()), int(verts[:, 1].max())
    bounds = (r0, r1, c0, c1)
    # A point in the corner of the upright box but outside the tilted ROI.
    assert not pipeline._touches_boundary(r0 + 6, c0 + 6, 3, bounds)
    assert pipeline._touches_boundary(r0 + 6, c0 + 6, 3, bounds, mask)
    # The ROI centre is safely inside either way.
    assert not pipeline._touches_boundary(100, 100, 3, bounds, mask)


def test_boundary_rejects_a_disk_running_off_the_image(pipeline):
    mask = np.ones((50, 50), dtype=bool)
    assert pipeline._touches_boundary(1, 1, 3, None, mask)
    assert not pipeline._touches_boundary(25, 25, 3, None, mask)


def _thin_roi_scene():
    """A 5-px-tall ROI on a bright field: the ring escapes it top and bottom.

    Mirrors the real case -- a ~22 px wide tilted box with a ring reaching 8 px
    from the spot, so most of the ring sits outside the ROI.
    """
    raw = np.full((60, 60), 5000.0)            # bright neighbours all around
    raw[28:33, :] = 100.0                      # the ROI's own quiet background
    fg = np.zeros((60, 60), dtype=bool)
    roi = np.zeros((60, 60), dtype=bool)
    roi[28:33, :] = True
    return raw, fg, roi


def test_ring_is_clipped_to_the_roi(pipeline):
    """Bright pixels outside the ROI must not raise the background."""
    raw, fg, roi = _thin_roi_scene()
    p = pipeline.PipelineParams(measure_radius=3, bkg_gap=1, bkg_width=4)

    leaky = pipeline.measure_spot(raw, fg, 30, 30, p)
    tight = pipeline.measure_spot(raw, fg, 30, 30, p, roi=roi)
    assert leaky["bkg_median"] > 1000          # dragged up by the neighbours
    assert tight["bkg_median"] == 100.0        # clean local background
    assert tight["bkg_pixels"] < leaky["bkg_pixels"]
    assert tight["bkg_pixels"] >= pipeline.MIN_RING_PIXELS


def test_clipping_can_be_turned_off(pipeline):
    raw, fg, roi = _thin_roi_scene()
    off = pipeline.PipelineParams(measure_radius=3, bkg_gap=1, bkg_width=4,
                                  clip_bkg_to_roi=False)
    on = pipeline.PipelineParams(measure_radius=3, bkg_gap=1, bkg_width=4)
    assert pipeline.measure_spot(raw, fg, 30, 30, off, roi=roi)["bkg_median"] > 1000
    assert pipeline.measure_spot(raw, fg, 30, 30, on, roi=roi)["bkg_median"] == 100.0


def test_clipping_falls_back_when_too_few_pixels_remain(pipeline):
    """A sliver of a ring is a worse estimate than a slightly leaky one."""
    raw = np.full((60, 60), 100.0)
    fg = np.zeros((60, 60), dtype=bool)
    roi = np.zeros((60, 60), dtype=bool)
    roi[30:32, 30:32] = True                   # far smaller than the ring
    p = pipeline.PipelineParams(measure_radius=3, bkg_gap=1, bkg_width=3)
    full = pipeline.measure_spot(raw, fg, 30, 30, p)
    kept = pipeline.measure_spot(raw, fg, 30, 30, p, roi=roi)
    assert kept["bkg_pixels"] == full["bkg_pixels"]     # fell back
    assert kept["bkg_pixels"] >= pipeline.MIN_RING_PIXELS


def test_bkg_pixels_reported_and_background_scales_with_signal(pipeline):
    raw = np.full((40, 40), 50.0)
    fg = np.zeros((40, 40), dtype=bool)
    fg[19:22, 19:22] = True
    p = pipeline.PipelineParams(measure_radius=3, bkg_gap=1, bkg_width=3)
    m = pipeline.measure_spot(raw, fg, 20, 20, p)
    assert m["bkg_pixels"] > 0
    assert m["background"] == m["bkg_median"] * m["n_pixels"]
