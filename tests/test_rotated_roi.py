"""Tilted-rectangle ROIs: is_rotated_rect + masking on the real outline.

A box turned with napari's rotation handle is a true rectangle at an angle.
It must be recognised as such (so it isn't flagged as deformed) and, like any
polygon, measured on its own outline rather than its upright bounding box.
"""
import numpy as np


def _rot_rect(cy, cx, length, width, deg):
    """Four corners of a rectangle of ``length`` x ``width`` turned ``deg``."""
    t = np.radians(deg)
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-u[1], u[0]])
    c = np.array([float(cy), float(cx)])
    return np.array([c - u * length / 2 - v * width / 2,
                     c + u * length / 2 - v * width / 2,
                     c + u * length / 2 + v * width / 2,
                     c - u * length / 2 + v * width / 2])


def test_rotated_rectangle_recognised_at_every_angle(pipeline):
    for deg in (0, 15, 30, 45, 67, 90, 123):
        v = _rot_rect(100, 100, 66, 22, deg)
        assert pipeline.is_rotated_rect(v), f"failed at {deg} degrees"


def test_axis_aligned_box_is_both(pipeline):
    v = _rot_rect(100, 100, 60, 20, 0)
    assert pipeline.is_axis_rect(v)
    assert pipeline.is_rotated_rect(v)


def test_tilted_box_is_not_axis_aligned(pipeline):
    v = _rot_rect(100, 100, 60, 20, 40)
    assert not pipeline.is_axis_rect(v)
    assert pipeline.is_rotated_rect(v)


def test_deformed_quads_are_rejected(pipeline):
    trapezoid = np.array([[0., 0.], [0., 60.], [40., 50.], [40., 10.]])
    parallelogram = np.array([[0., 0.], [0., 60.], [40., 75.], [40., 15.]])
    assert not pipeline.is_rotated_rect(trapezoid)
    assert not pipeline.is_rotated_rect(parallelogram)


def test_non_quadrilaterals_are_rejected(pipeline):
    pentagon = np.array([[0., 0.], [0., 50.], [30., 60.], [50., 30.], [20., 0.]])
    triangle = np.array([[0., 0.], [0., 40.], [30., 20.]])
    assert not pipeline.is_rotated_rect(pentagon)
    assert not pipeline.is_rotated_rect(triangle)
    assert not pipeline.is_rotated_rect(np.array([[0., 0.], [0., 10.]]))


def test_tolerances_are_configurable(pipeline):
    """A box a few degrees out of square passes the default 5-degree slack."""
    v = np.array([[0., 0.], [0., 60.], [40., 62.], [40., 2.]])
    assert pipeline.is_rotated_rect(v, angle_tol=10.0, side_tol=5.0)
    assert not pipeline.is_rotated_rect(v, angle_tol=1.0, side_tol=1.0)


def test_tilted_roi_masks_its_outline_not_its_bounding_box(pipeline):
    """A 45-degree box must exclude the corners of its upright box."""
    verts = _rot_rect(100, 100, 80, 24, 45)
    shape = (200, 200)
    poly = pipeline.roi_mask(None, verts, shape)
    r0, r1 = int(verts[:, 0].min()), int(verts[:, 0].max())
    c0, c1 = int(verts[:, 1].min()), int(verts[:, 1].max())
    box = pipeline.roi_mask((r0, r1, c0, c1), None, shape)
    assert poly.sum() < 0.75 * box.sum()      # far smaller than its box
    # Rectangle masks are half-open [r0, r1); polygon2mask includes the
    # boundary, so allow the one-pixel edge when checking containment.
    padded = pipeline.roi_mask((r0, r1 + 1, c0, c1 + 1), None, shape)
    assert (poly & ~padded).sum() == 0         # and wholly inside it


def test_detection_is_confined_to_a_tilted_roi(pipeline):
    """A spot inside the upright box but outside the tilted box is ignored."""
    rng = np.random.default_rng(0)
    Z, H, W = 7, 200, 200
    stack = rng.normal(100, 3, (Z, H, W))
    yy, xx = np.ogrid[:H, :W]

    def plant(z, r, c, amp=900):
        for k in range(Z):
            stack[k] += (amp * np.exp(-((k - z) ** 2) / 2.0)
                         * np.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / 4.0)))

    verts = _rot_rect(100, 100, 90, 26, 45)
    plant(3, 100, 100)                        # centre: inside the tilted box
    r0, r1 = int(verts[:, 0].min()), int(verts[:, 0].max())
    c0, c1 = int(verts[:, 1].min()), int(verts[:, 1].max())
    plant(3, r0 + 6, c0 + 6)                  # corner: in the box, not the ROI

    params = pipeline.PipelineParams(min_link_planes=3)
    roi = [(r0, r1, c0, c1, "ROI1")]
    masks = {"ROI1": pipeline.roi_mask(None, verts, (H, W))}

    with_mask = pipeline.run_stack(stack, params, rois=roi, masks=masks)
    without = pipeline.run_stack(stack, params, rois=roi)
    assert len(with_mask.measurements) == 1
    assert len(without.measurements) == 2
    row = with_mask.measurements.iloc[0]
    assert abs(int(row["row_px"]) - 100) <= 2
    assert abs(int(row["col_px"]) - 100) <= 2
