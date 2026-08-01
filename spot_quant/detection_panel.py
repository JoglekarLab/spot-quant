"""Spot detection & measurement dock widget.

Grouped controls for each algorithm stage.  Tuning a control updates a live
single-plane *preview* (top-hat, foreground, current-plane maxima).  The
**Detect spots** button runs the full-stack pipeline: per-plane detection,
cross-plane linking, in-focus selection, disk measurement with a concentric
background ring, and region assignment for any local-threshold ROIs.

ROIs are rectangles by default.  A separate **Draw polygon ROI** button draws
true polygons, whose pixels are selected with a mask instead of a bounding
box.  A shape drawn as a rectangle but reshaped so it is no longer an upright
rectangle is rejected (Detect is blocked with a message) rather than silently
treated as its bounding box.
"""
from __future__ import annotations

import napari.utils.notifications as notifications
import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from . import cells, pipeline
from .filters import SMOOTHING_METHODS
from .pipeline import THRESHOLD_METHODS, PipelineParams
from .state import AppState

ROI_LAYER = "local_ROI"
CELLS_LAYER = "cells"
OVAL_LAYER = "new cell"
# Distinct border colours cycled per region for spot annotation.
_REGION_COLORS = ["red", "cyan", "yellow", "lime", "orange", "magenta"]

# Descriptions kept for reference (the old ⓘ hover icons were removed).
_PARAM_TIPS = {
    "channel":
        "Reference channel that spots are detected in. Every other fluorescence "
        "channel is then measured at those same spots. Brightfield / "
        "transmitted / phase channels are skipped.",
    "smooth_method":
        "Noise-reduction filter applied before detection. Median is a solid "
        "default; Gaussian blurs more; Kuwahara smooths while preserving edges.",
    "smooth_size":
        "Size (radius, px) of the smoothing filter. Larger = smoother, but "
        "blurs small spots together.",
    "tophat_size":
        "White top-hat size (px). Removes background structures larger than "
        "this and keeps spot-sized features. Set a little larger than your "
        "spots.",
    "thresh_method":
        "How the bright-vs-background cutoff is chosen from the pooled ROI "
        "pixels. Otsu and Li are automatic; ilastik uses a trained .ilp model.",
    "ilastik_prob":
        "ilastik method only: pixels whose foreground probability is above "
        "this value (0–1) are kept as signal.",
    "min_mask_size":
        "Foreground blobs smaller than this many pixels are dropped as noise "
        "(judged per plane).",
    "min_distance":
        "Minimum spacing (px) between two spots. Peaks closer than this merge "
        "into one and the brighter wins. Lower it if close spots merge; raise "
        "it if one spot splits into two.",
    "peak_rel":
        "How bright a peak must be to count, as a fraction (0–1) of the "
        "brightest peak in the region. This is the main dial for finding more "
        "or fewer spots — lower it to pick up dimmer puncta.",
    "link_dist":
        "Maximum sideways drift (XY px) a spot may move between adjacent "
        "z-slices and still be treated as the same spot across depth.",
    "min_link":
        "A spot must appear across at least this many consecutive z-slices to "
        "be kept (drops single-slice noise). Starts from your z-step and is "
        "capped at the number of slices.",
    "measure_radius":
        "Radius (px) of the disk drawn on each spot; its bright pixels are "
        "summed as the signal. Note: a spot whose disk reaches the ROI edge is "
        "dropped.",
    "bkg_gap":
        "Gap (px) between the spot disk and the background ring, so the spot's "
        "own glow doesn't leak into the background estimate.",
    "bkg_width":
        "Thickness (px) of the background ring outside the gap. Its median "
        "(ignoring bright pixels) is the local background that gets subtracted.",
}


def _roi_number(label: str) -> int:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return int(digits) if digits else 0


class DetectionPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.viewer = state.viewer
        self.params = PipelineParams()
        self._running = False
        # Analyzed ROIs for the current file: list of dicts with
        # {"key", "bounds": (r0,r1,c0,c1), "verts": ndarray|None,
        #  "label": "ROIn", "centroids": (N,3) array}.
        self._analyzed = []
        self._spot_vol = None            # accumulated detected-region label vol
        # Active sticky ROI tool: None / "add_rectangle" / "add_polygon" /
        # "select".  Kept asserted across preview/detect layer updates so the
        # user can draw or delete many ROIs until Esc or toggling the button.
        self._draw_mode = None
        # True while a "Pick mom"/"Pick daughter"/"Remove pick" tool is armed.
        self._pick_daughter = False
        self._pick_mom = False
        self._pick_remove = False
        # "mom"/"daughter"/None while a "Draw ... cell" polygon tool is armed.
        self._oval_role = None
        # True while the "Add spot" tool is armed (click to place/remove a spot).
        self._add_spot_mode = False

        layout = QVBoxLayout(self)

        # -- channel -------------------------------------------------------- #
        chan_box = QGroupBox("Channel")
        chan_form = QFormLayout(chan_box)
        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self._on_param_changed)
        chan_form.addRow("Measure channel",
                         self._with_info(self.channel_selector,
                                         _PARAM_TIPS["channel"]))
        layout.addWidget(chan_box)

        # -- smoothing / top-hat / thresholding live in a pop-up dialog ----- #
        smooth_box = QGroupBox("1. Smoothing")
        smooth_form = QFormLayout(smooth_box)
        self.smooth_method = QComboBox()
        self.smooth_method.addItems(SMOOTHING_METHODS)
        self.smooth_method.setCurrentText(self.params.smoothing_method)  # Median
        self.smooth_size = QDoubleSpinBox()
        self.smooth_size.setRange(0.1, 100.0)
        self.smooth_size.setValue(self.params.smoothing_size)
        smooth_form.addRow("Method", self._with_info(
            self.smooth_method, _PARAM_TIPS["smooth_method"]))
        smooth_form.addRow("Filter size", self._with_info(
            self.smooth_size, _PARAM_TIPS["smooth_size"]))

        tophat_box = QGroupBox("2. White top-hat")
        tophat_form = QFormLayout(tophat_box)
        self.tophat_size = QDoubleSpinBox()
        self.tophat_size.setRange(1.0, 200.0)
        self.tophat_size.setValue(self.params.tophat_size)
        tophat_form.addRow("Filter size", self._with_info(
            self.tophat_size, _PARAM_TIPS["tophat_size"]))

        thresh_box = QGroupBox("3. Thresholding")
        thresh_form = QFormLayout(thresh_box)
        self.thresh_method = QComboBox()
        self.thresh_method.addItems(THRESHOLD_METHODS)
        thresh_form.addRow("Method", self._with_info(
            self.thresh_method, _PARAM_TIPS["thresh_method"]))

        ilastik_row = QHBoxLayout()
        self.ilastik_label = QLabel("(no model)")
        self.ilastik_label.setWordWrap(True)
        self.ilastik_btn = QPushButton("Browse .ilp…")
        self.ilastik_btn.clicked.connect(self._choose_ilastik)
        self.ilastik_btn.setEnabled(False)
        ilastik_row.addWidget(self.ilastik_btn)
        ilastik_row.addWidget(self.ilastik_label, 1)
        thresh_form.addRow("ilastik model", ilastik_row)
        self.ilastik_prob = QDoubleSpinBox()
        self.ilastik_prob.setRange(0.0, 1.0)
        self.ilastik_prob.setSingleStep(0.05)
        self.ilastik_prob.setValue(self.params.ilastik_prob_threshold)
        thresh_form.addRow("ilastik prob.", self._with_info(
            self.ilastik_prob, _PARAM_TIPS["ilastik_prob"]))
        self.min_mask_size = QSpinBox()
        self.min_mask_size.setRange(1, 100000)
        self.min_mask_size.setValue(self.params.min_mask_size)
        thresh_form.addRow("Min mask size", self._with_info(
            self.min_mask_size, _PARAM_TIPS["min_mask_size"]))

        self._controls_dialog = QDialog(self)
        self._controls_dialog.setWindowTitle("Spot detection controls")
        cd_layout = QVBoxLayout(self._controls_dialog)
        for box in (smooth_box, tophat_box, thresh_box):
            cd_layout.addWidget(box)
        cd_layout.addStretch(1)

        controls_btn = QPushButton("Spot detection controls…")
        controls_btn.clicked.connect(self._open_controls)
        layout.addWidget(controls_btn)

        # -- ROIs (required) ------------------------------------------------ #
        roi_box = QGroupBox("3b. Detection ROIs (required)")
        roi_layout = QVBoxLayout(roi_box)
        roi_layout.addWidget(QLabel(
            "Draw rectangles, or use 'Draw polygon ROI' for a custom shape.\n"
            "A draw/select button stays on: keep drawing (or deleting) ROIs\n"
            "until you press Esc or click the button again. Detection happens\n"
            "ONLY inside these ROIs; a rectangle reshaped into a non-rectangle\n"
            "is rejected — redraw it or draw it as a polygon."))
        roi_btn_row = QHBoxLayout()
        self.draw_btn = QPushButton("Draw ROI(s)")
        self.poly_btn = QPushButton("Draw polygon ROI")
        self.select_btn = QPushButton("Select / delete ROI")
        clear_btn = QPushButton("Clear ROIs")
        for b in (self.draw_btn, self.poly_btn, self.select_btn):
            b.setCheckable(True)          # stays pressed while its mode is on
        self.draw_btn.clicked.connect(self._draw_rois)
        self.poly_btn.clicked.connect(self._draw_polys)
        self.select_btn.clicked.connect(self._select_rois)
        clear_btn.clicked.connect(self._clear_rois)
        for b in (self.draw_btn, self.poly_btn, self.select_btn, clear_btn):
            roi_btn_row.addWidget(b)
        roi_layout.addLayout(roi_btn_row)
        layout.addWidget(roi_box)

        # -- maxima --------------------------------------------------------- #
        maxima_box = QGroupBox("4. Maxima detection & linking")
        maxima_form = QFormLayout(maxima_box)
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 200)
        self.min_distance.setValue(self.params.min_distance)
        self.peak_rel = QDoubleSpinBox()
        self.peak_rel.setRange(0.0, 1.0)
        self.peak_rel.setSingleStep(0.05)
        self.peak_rel.setValue(self.params.peak_rel_threshold)
        self.link_dist = QDoubleSpinBox()
        self.link_dist.setRange(0.5, 50.0)
        self.link_dist.setSingleStep(0.5)
        self.link_dist.setValue(self.params.link_max_dist)
        self.min_link = QSpinBox()
        self.min_link.setRange(1, 100)
        self.min_link.setValue(self.params.min_link_planes)
        maxima_form.addRow("Min. distance", self._with_info(
            self.min_distance, _PARAM_TIPS["min_distance"]))
        maxima_form.addRow("Rel. threshold", self._with_info(
            self.peak_rel, _PARAM_TIPS["peak_rel"]))
        maxima_form.addRow("Link dist. (XY px)", self._with_info(
            self.link_dist, _PARAM_TIPS["link_dist"]))
        maxima_form.addRow("Min Z-linkage", self._with_info(
            self.min_link, _PARAM_TIPS["min_link"]))
        layout.addWidget(maxima_box)

        # -- spot disk + background ring ------------------------------------ #
        meas_box = QGroupBox("5. Measurement (disk + ring background)")
        meas_form = QFormLayout(meas_box)
        self.measure_radius = QSpinBox()
        self.measure_radius.setRange(1, 50)
        self.measure_radius.setValue(self.params.measure_radius)
        self.bkg_gap = QSpinBox()
        self.bkg_gap.setRange(0, 50)
        self.bkg_gap.setValue(self.params.bkg_gap)
        self.bkg_width = QSpinBox()
        self.bkg_width.setRange(1, 50)
        self.bkg_width.setValue(self.params.bkg_width)
        meas_form.addRow("Spot radius", self._with_info(
            self.measure_radius, _PARAM_TIPS["measure_radius"]))
        meas_form.addRow("Ring gap", self._with_info(
            self.bkg_gap, _PARAM_TIPS["bkg_gap"]))
        meas_form.addRow("Ring width", self._with_info(
            self.bkg_width, _PARAM_TIPS["bkg_width"]))
        layout.addWidget(meas_box)

        # Chromatic offset now lives in the Edit-metadata dialog (File IO).

        # -- detect --------------------------------------------------------- #
        self.detect_btn = QPushButton("Detect spots (stack)")
        self.detect_btn.clicked.connect(self.detect)
        layout.addWidget(self.detect_btn)
        self.status_label = QLabel("No spots detected yet")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # -- manual spot --------------------------------------------------- #
        spot_box = QGroupBox("Manual spot")
        spot_v = QVBoxLayout(spot_box)
        spot_v.addWidget(QLabel(
            "Detection missed a spot? Click 'Add spot', then click the middle "
            "of\nwhere the spot should be, inside its ROI. Click an added spot "
            "again\nto remove it. Added spots are measured (full disk) and "
            "exported."))
        self.add_spot_btn = QPushButton("Add spot")
        self.add_spot_btn.setCheckable(True)
        self.add_spot_btn.clicked.connect(self._toggle_add_spot)
        spot_v.addWidget(self.add_spot_btn)
        layout.addWidget(spot_box)

        # -- 6. cells: mom / daughter via micro-sam ------------------------- #
        cell_box = QGroupBox("6. Cells: mom / bud (micro-sam)")
        cell_layout = QVBoxLayout(cell_box)
        cell_layout.addWidget(QLabel(
            "Segment brightfield cells with micro-sam. 'Auto: all ROIs' colours "
            "the\nbigger cell in each ROI as mom, the smaller as bud (no "
            "spots\nneeded). If spots are detected it also tags each dot "
            "(mom / bud /\ntoward-bud). Or set a cell by hand with "
            "Pick mom / Pick bud.\nMom cells outline cyan, buds "
            "yellow."))
        self.segment_btn = QPushButton("Segment cells (micro-sam)")
        self.segment_btn.clicked.connect(self._segment_cells)
        cell_layout.addWidget(self.segment_btn)
        gap_form = QFormLayout()
        self.toward_gap = QDoubleSpinBox()
        self.toward_gap.setRange(0.0, 100.0)
        self.toward_gap.setDecimals(2)
        self.toward_gap.setSingleStep(0.05)
        self.toward_gap.setValue(0.3)
        gap_form.addRow("Toward-bud gap (µm)", self._with_info(
            self.toward_gap,
            "When both dots are in the mother, one must be at least this many "
            "microns closer to the bud than the other to be tagged "
            "'toward bud'; otherwise both are just 'mom'."))
        cell_layout.addLayout(gap_form)
        crow = QHBoxLayout()
        self.auto_all_btn = QPushButton("Auto: all ROIs")
        self.auto_all_btn.clicked.connect(self._auto_all_rois)
        crow.addWidget(self.auto_all_btn)
        cell_layout.addLayout(crow)
        mrow = QHBoxLayout()
        self.pick_mom_btn = QPushButton("Pick mom")
        self.pick_dau_btn = QPushButton("Pick bud")
        self.pick_remove_btn = QPushButton("Remove pick")
        self.pick_mom_btn.setCheckable(True)
        self.pick_dau_btn.setCheckable(True)
        self.pick_remove_btn.setCheckable(True)
        self.pick_mom_btn.clicked.connect(self._toggle_pick_mom)
        self.pick_dau_btn.clicked.connect(self._toggle_pick_daughter)
        self.pick_remove_btn.clicked.connect(self._toggle_pick_remove)
        mrow.addWidget(QLabel("Manual (click a cell):"))
        mrow.addWidget(self.pick_mom_btn)
        mrow.addWidget(self.pick_dau_btn)
        mrow.addWidget(self.pick_remove_btn)
        cell_layout.addLayout(mrow)
        orow = QHBoxLayout()
        self.oval_mom_btn = QPushButton("Draw mom cell")
        self.oval_dau_btn = QPushButton("Draw bud cell")
        self.oval_mom_btn.setCheckable(True)
        self.oval_dau_btn.setCheckable(True)
        self.oval_mom_btn.clicked.connect(lambda: self._arm_oval("mom"))
        self.oval_dau_btn.clicked.connect(lambda: self._arm_oval("bud"))
        orow.addWidget(QLabel("micro-sam missed one? Draw a polygon:"))
        orow.addWidget(self.oval_mom_btn)
        orow.addWidget(self.oval_dau_btn)
        cell_layout.addLayout(orow)
        layout.addWidget(cell_box)

        layout.addStretch(1)

        # A parameter change invalidates any detection -> back to preview.
        for w in (self.smooth_method, self.thresh_method):
            w.currentIndexChanged.connect(self._on_param_changed)
        for w in (self.smooth_size, self.tophat_size, self.ilastik_prob,
                  self.peak_rel):
            w.valueChanged.connect(self._on_param_changed)
        self.min_distance.valueChanged.connect(self._on_param_changed)
        self.min_mask_size.valueChanged.connect(self._on_param_changed)
        self.min_link.valueChanged.connect(self._on_param_changed)
        self.thresh_method.currentTextChanged.connect(self._on_threshold_changed)

        self.state.image_loaded.connect(self._on_image_loaded)
        self.state.metadata_changed.connect(self._on_param_changed)
        self.state.session_imported.connect(self._on_session_imported)
        # Z-scrolling refreshes the preview but keeps any detection on screen.
        self.viewer.dims.events.current_step.connect(lambda e: self._preview())

        self.setEnabled(False)

    # ------------------------------------------------------------------ #
    def _with_info(self, field, tip):
        """Return *field* as-is. The old hover-help 'ⓘ' icon was removed; the
        *tip* argument is ignored so existing call sites keep working."""
        return field

    def _on_image_loaded(self):
        # Keep the previously chosen reference channel if it still exists.
        prev = self.channel_selector.currentText()
        self.channel_selector.blockSignals(True)
        self.channel_selector.clear()
        self.channel_selector.addItems(self.state.meta.channel_names)
        if prev and prev in self.state.meta.channel_names:
            self.channel_selector.setCurrentText(prev) \
                if hasattr(self.channel_selector, "setCurrentText") else None
            self.channel_selector.setCurrentIndex(
                self.state.meta.channel_names.index(prev))
        self.channel_selector.blockSignals(False)
        # Start-up min Z-linkage depends on the z-step size.
        min_link = pipeline.min_link_for_zstep(self.state.meta.z_step)
        self.min_link.blockSignals(True)
        self.min_link.setValue(min_link)
        self.min_link.blockSignals(False)
        self.params.min_link_planes = min_link
        self.setEnabled(True)
        # New file: drop any previous cell segmentation display.
        self._pick_daughter = False
        self._pick_mom = False
        self._pick_remove = False
        self.pick_dau_btn.setChecked(False)
        self.pick_mom_btn.setChecked(False)
        self.pick_remove_btn.setChecked(False)
        self._add_spot_mode = False
        self.add_spot_btn.setChecked(False)
        self._disarm_draw(commit=False)
        for n in (CELLS_LAYER, "mom cells", "bud cells", "spot roles",
                  "cell labels", OVAL_LAYER):
            self._clear_layer(n)
        self._reset_analysis()
        # Restoring ROIs and running the first preview touches the shapes layer
        # right as the image layers settle; doing it synchronously here can race
        # napari's layer model and segfault. Defer to the next event-loop tick,
        # once the viewer is in a consistent state.
        QTimer.singleShot(0, self._restore_and_preview)

    def _restore_and_preview(self):
        if self.state.image is None:
            return
        self._restore_rois()
        self._preview()

    def _on_session_imported(self):
        """Sync controls to imported settings, then restore ROIs for the open
        file so the user can keep adding ROIs."""
        if self.state.params is not None:
            self._apply_params_to_controls(self.state.params)
        if self.state.image is not None:
            self._reset_analysis()
            QTimer.singleShot(0, self._restore_and_preview)

    def _apply_params_to_controls(self, p):
        """Push an imported ``PipelineParams`` onto the widgets (no re-preview)."""
        widgets = [self.smooth_method, self.smooth_size, self.tophat_size,
                   self.thresh_method, self.min_mask_size, self.min_distance,
                   self.peak_rel, self.link_dist, self.min_link,
                   self.measure_radius, self.bkg_gap, self.bkg_width,
                   self.ilastik_prob]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.smooth_method.setCurrentText(p.smoothing_method)
            self.smooth_size.setValue(float(p.smoothing_size))
            self.tophat_size.setValue(float(p.tophat_size))
            self.thresh_method.setCurrentText(p.threshold_method)
            self.min_mask_size.setValue(int(p.min_mask_size))
            self.min_distance.setValue(int(p.min_distance))
            self.peak_rel.setValue(float(p.peak_rel_threshold))
            self.link_dist.setValue(float(p.link_max_dist))
            self.min_link.setValue(int(p.min_link_planes))
            self.measure_radius.setValue(int(p.measure_radius))
            self.bkg_gap.setValue(int(p.bkg_gap))
            self.bkg_width.setValue(int(p.bkg_width))
            self.ilastik_prob.setValue(float(p.ilastik_prob_threshold))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.params = p

    def _reset_analysis(self):
        """Forget analyzed ROIs and remove committed detection layers.

        The ROI label counter is session-global (on state) and is NOT reset
        here, so ROI indices keep incrementing across files.
        """
        self._analyzed = []
        self._spot_vol = None
        self.state.pending_rois = 0
        self._draw_mode = None
        self._sync_mode_buttons()
        for lyr in ("spots", "ROI labels", "spots (preview)"):
            self._clear_layer(lyr)

    def _on_param_changed(self, *args):
        self._preview()

    def _on_threshold_changed(self, method):
        self.ilastik_btn.setEnabled(method == "ilastik")
        self._on_param_changed()

    def _choose_ilastik(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ilastik project", "", "ilastik project (*.ilp)")
        if path:
            from pathlib import Path
            self.params.ilastik_model_path = Path(path)
            self.ilastik_label.setText(Path(path).name)
            self._on_param_changed()

    # -- ROI handling --------------------------------------------------- #
    def _connect_roi_layer(self, layer):
        """Wire an ROI shapes layer's edit + mode events."""
        layer.events.data.connect(lambda e: self._on_rois_changed())
        layer.events.mode.connect(lambda e: self._on_roi_mode_changed())
        layer.mouse_drag_callbacks.append(self._on_add_spot_click)

    def _ensure_roi_layer(self):
        if ROI_LAYER not in self.viewer.layers:
            layer = self.viewer.add_shapes(
                name=ROI_LAYER, face_color="transparent", edge_color="yellow",
                edge_width=2)
            self._connect_roi_layer(layer)
        return self.viewer.layers[ROI_LAYER]

    # -- sticky draw / select modes ------------------------------------- #
    def _set_draw_mode(self, mode):
        """Enter a sticky ROI mode, or toggle it off if already active.

        ``mode`` is 'add_rectangle', 'add_polygon' or 'select'.  The mode stays
        on (kept asserted after preview/detect redraw layers) until the user
        presses Esc in the canvas or clicks the same button again.
        """
        if self._draw_mode == mode:
            self._stop_draw_mode()
            return
        layer = self._ensure_roi_layer()
        self._draw_mode = mode
        self.viewer.layers.selection.active = layer
        layer.mode = mode
        self._sync_mode_buttons()
        if mode == "select":
            notifications.show_info(
                "Click an ROI to select it, press Delete to remove it. "
                "Esc or click the button again to stop.")
        else:
            notifications.show_info(
                "Keep drawing ROIs; press Esc or click the button again to stop.")

    def _stop_draw_mode(self):
        self._draw_mode = None
        if ROI_LAYER in self.viewer.layers:
            try:
                self.viewer.layers[ROI_LAYER].mode = "pan_zoom"
            except Exception:  # noqa: BLE001
                pass
        self._sync_mode_buttons()

    def _draw_rois(self, *args):
        self._set_draw_mode("add_rectangle")

    def _draw_polys(self, *args):
        self._set_draw_mode("add_polygon")

    def _select_rois(self, *args):
        self._set_draw_mode("select")

    def _on_roi_mode_changed(self):
        """Drop the sticky mode when the user leaves it (e.g. presses Esc)."""
        if self._draw_mode is None or ROI_LAYER not in self.viewer.layers:
            return
        if self.viewer.layers[ROI_LAYER].mode != self._draw_mode:
            self._draw_mode = None
            self._sync_mode_buttons()

    def _reassert_draw_mode(self):
        """Keep the ROI layer active + in its mode after aux layers redraw."""
        if not self._draw_mode or ROI_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[ROI_LAYER]
        try:
            if self.viewer.layers.selection.active is not layer:
                self.viewer.layers.selection.active = layer
            if layer.mode != self._draw_mode:
                layer.mode = self._draw_mode
        except Exception:  # noqa: BLE001
            pass

    def _sync_mode_buttons(self):
        self.draw_btn.setChecked(self._draw_mode == "add_rectangle")
        self.poly_btn.setChecked(self._draw_mode == "add_polygon")
        self.select_btn.setChecked(self._draw_mode == "select")

    def _clear_rois(self, *args):
        if ROI_LAYER in self.viewer.layers:
            self.viewer.layers.remove(ROI_LAYER)
        self._draw_mode = None
        self._sync_mode_buttons()
        self._reset_analysis()
        self._preview()

    def _read_shapes(self):
        """Every ROI shape as a record.

        Returns a list of dicts:
          ``{"bounds": (r0,r1,c0,c1), "verts": ndarray|None, "valid": bool,
             "key": hashable}``
        ``verts`` is ``None`` for a valid rectangle (bounding-box path); it is
        the polygon corners for a polygon.  A shape drawn as a rectangle but no
        longer axis-aligned is marked ``valid=False`` (corners kept for the
        message).
        """
        out = []
        if ROI_LAYER not in self.viewer.layers:
            return out
        layer = self.viewer.layers[ROI_LAYER]
        try:
            shape_types = list(layer.shape_type)
        except Exception:  # noqa: BLE001
            shape_types = []
        for i, shp in enumerate(layer.data):
            coords = np.asarray(shp, dtype=float)
            yx = coords[:, -2:]
            ys, xs = yx[:, 0], yx[:, 1]
            bounds = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
            st = shape_types[i] if i < len(shape_types) else "polygon"
            if st == "rectangle" and pipeline.is_axis_rect(yx):
                out.append({"bounds": bounds, "verts": None, "valid": True,
                            "key": pipeline.shape_key(bounds, None)})
            elif st == "rectangle":
                # Rectangle-typed but reshaped: reject, don't silently box it.
                out.append({"bounds": bounds, "verts": yx, "valid": False,
                            "key": pipeline.shape_key(bounds, yx)})
            else:
                out.append({"bounds": bounds, "verts": yx, "valid": True,
                            "key": pipeline.shape_key(bounds, yx)})
        return out

    def _valid_shapes(self):
        return [s for s in self._read_shapes() if s["valid"]]

    def _invalid_positions(self):
        return [i for i, s in enumerate(self._read_shapes(), 1) if not s["valid"]]

    def _analyzed_keys(self):
        return {e["key"] for e in self._analyzed}

    def _unanalyzed(self):
        done = self._analyzed_keys()
        return [s for s in self._valid_shapes() if s["key"] not in done]

    def _update_pending(self):
        """Publish how many valid ROIs are still awaiting analysis."""
        self.state.pending_rois = len(self._unanalyzed())

    def _plane_shape(self):
        return self.state.channel_stack(
            self.channel_selector.currentIndex()).shape[1:]

    def _masks_for(self, labelled_shapes, plane_shape):
        """Build ``{label: mask}`` for any polygon shapes (rectangles skipped)."""
        masks = {}
        for label, verts in labelled_shapes:
            if verts is not None:
                masks[label] = pipeline.roi_mask(None, verts, plane_shape)
        return masks

    # -- two-spot warning highlight ------------------------------------- #
    def _committed_counts(self):
        """Spot count per analyzed ROI, keyed by its shape key."""
        return {e["key"]: len(e["centroids"]) for e in self._analyzed}

    def _apply_roi_warnings(self, counts):
        """Colour each ROI outline by spot count; return the number flagged.

        Green = exactly 2 spots; red = 0, 1 or 3+ (can't be analysed as a pair,
        which is what the line-scan and paired report need); yellow = not counted
        yet.  ``counts`` maps a shape key -> spot count.
        """
        if ROI_LAYER not in self.viewer.layers:
            return 0
        layer = self.viewer.layers[ROI_LAYER]
        rgba = {"ok": (0.0, 1.0, 0.0, 1.0), "warn": (1.0, 0.0, 0.0, 1.0),
                "none": (1.0, 1.0, 0.0, 1.0)}
        colors, flagged = [], 0
        for s in self._read_shapes():
            cnt = counts.get(s["key"])
            if cnt is None:
                colors.append(rgba["none"])
            elif cnt == 2:
                colors.append(rgba["ok"])
            else:
                colors.append(rgba["warn"])
                flagged += 1
        if colors:
            for value in (np.array(colors, dtype=float), colors):
                try:
                    layer.edge_color = value
                    break
                except Exception:  # noqa: BLE001
                    continue
            try:
                layer.refresh()
            except Exception:  # noqa: BLE001
                pass
        return flagged

    def _save_rois(self):
        """Persist the current file's valid ROIs with session-global labels."""
        if self.state.current_path is None:
            return self.state.assign_roi_labels("", [], "")
        return self.state.assign_roi_labels(
            self.state.current_path.name, self._valid_shapes(),
            self.channel_selector.currentText())

    def _restore_rois(self):
        """Recreate the ROI shapes layer for a file that has saved ROIs.

        Each ROI is added independently so one bad shape (e.g. an imported ROI
        with an off-image coordinate) can't silently wipe out all of them; any
        that fail are reported instead of swallowed.
        """
        if self.state.current_path is None:
            return
        saved = self.state.session_shapes.get(self.state.current_path.name)
        if not saved:
            return
        if ROI_LAYER in self.viewer.layers:
            try:
                self.viewer.layers.remove(ROI_LAYER)
            except Exception:  # noqa: BLE001
                pass
        try:
            H, W = self._plane_shape()
        except Exception:  # noqa: BLE001
            H, W = None, None
        layer = self.viewer.add_shapes(
            name=ROI_LAYER, face_color="transparent", edge_color="yellow",
            edge_width=2)
        self._connect_roi_layer(layer)
        bad = []
        for rec in saved:
            try:
                if rec.get("verts") is not None:
                    shp = np.asarray(rec["verts"], dtype=float)[:, -2:].copy()
                    if H is not None:                 # clip to the image
                        shp[:, 0] = np.clip(shp[:, 0], 0, H)
                        shp[:, 1] = np.clip(shp[:, 1], 0, W)
                    if len(shp) < 3 or not np.isfinite(shp).all():
                        bad.append(str(rec.get("label", "?")))
                        continue
                    layer.add(shp, shape_type="polygon")
                else:
                    r0, r1, c0, c1 = (int(v) for v in rec["bounds"])
                    if H is not None:                 # clip off-image corners
                        r0, r1 = max(0, min(r0, H)), max(0, min(r1, H))
                        c0, c1 = max(0, min(c0, W)), max(0, min(c1, W))
                    r0, r1 = sorted((r0, r1))
                    c0, c1 = sorted((c0, c1))
                    if r1 <= r0 or c1 <= c0:          # degenerate after clipping
                        bad.append(str(rec.get("label", "?")))
                        continue
                    shp = np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]],
                                   dtype=float)
                    layer.add(shp, shape_type="rectangle")
            except Exception:  # noqa: BLE001
                bad.append(str(rec.get("label", "?")))
        if bad:
            notifications.show_warning(
                "Imported, but couldn't draw ROI(s): " + ", ".join(bad))

    # ------------------------------------------------------------------ #
    def _collect_params(self):
        self.params.smoothing_method = self.smooth_method.currentText()
        self.params.smoothing_size = self.smooth_size.value()
        self.params.tophat_size = self.tophat_size.value()
        self.params.threshold_method = self.thresh_method.currentText()
        self.params.ilastik_prob_threshold = self.ilastik_prob.value()
        self.params.min_mask_size = self.min_mask_size.value()
        self.params.min_distance = self.min_distance.value()
        self.params.peak_rel_threshold = self.peak_rel.value()
        self.params.link_max_dist = self.link_dist.value()
        self.params.min_link_planes = self.min_link.value()
        self.params.measure_radius = self.measure_radius.value()
        self.params.bkg_gap = self.bkg_gap.value()
        self.params.bkg_width = self.bkg_width.value()
        # Chromatic offset is edited in the metadata dialog and stored on state.
        self.params.offset_z = int(getattr(self.state, "offset_z", 0))
        self.params.offset_y = int(getattr(self.state, "offset_y", 0))
        self.params.offset_x = int(getattr(self.state, "offset_x", 0))
        # Share the latest params so the session can be re-run at export.
        self.state.params = self.params
        self.state.ref_channel = self.channel_selector.currentText()

    def _open_controls(self):
        self._controls_dialog.show()
        self._controls_dialog.raise_()
        self._controls_dialog.activateWindow()

    def _ready(self):
        if self.state.loading or self._running:
            return False
        if self.state.image is None or not self.isEnabled():
            return False
        if self.channel_selector.currentIndex() < 0:
            return False
        if (self.thresh_method.currentText() == "ilastik"
                and self.params.ilastik_model_path is None):
            self.status_label.setText("Select an .ilp model to use ilastik")
            return False
        return True

    def _on_rois_changed(self, *args):
        """Handle ROI edits: purge deleted ROIs' data, then refresh preview."""
        current = {s["key"] for s in self._valid_shapes()}
        removed = [e for e in self._analyzed if e["key"] not in current]
        if removed:
            fname = (self.state.current_path.name
                     if self.state.current_path else "unknown")
            for e in removed:
                self.state.remove_region(fname, e["label"])
                self._analyzed.remove(e)
            self._rebuild_committed()
            notifications.show_info(
                f"Removed {len(removed)} deleted ROI(s) from the data.")
        self._save_rois()
        self._preview()

    # -- live ROI-limited single-plane preview -------------------------- #
    def _preview(self, *args):
        if not self._ready():
            return
        self._collect_params()
        channel = self.channel_selector.currentIndex()

        # Block on any rectangle that isn't actually rectangular.
        invalid = self._invalid_positions()
        if invalid:
            self._clear_layer("spots (preview)")
            self.detect_btn.setEnabled(False)
            self.status_label.setText(
                f"ROI shape(s) {invalid} aren't rectangles. Redraw them, or "
                "use 'Draw polygon ROI'.")
            self._reassert_draw_mode()
            return
        self.detect_btn.setEnabled(True)

        new_shapes = self._unanalyzed()
        self._update_pending()
        self._running = True
        try:
            z = (int(self.viewer.dims.current_step[0])
                 if self.viewer.dims.ndim >= 3 else 0)
            if not new_shapes:
                self._clear_layer("spots (preview)")
                if not self._analyzed:
                    self.status_label.setText("Draw ROIs to preview/detect spots")
                return
            preview_rois = [(*s["bounds"], f"tmp{i}")
                            for i, s in enumerate(new_shapes)]
            # Pool the threshold across every drawn ROI (analyzed + new).
            all_rois = ([(*e["bounds"], e["label"]) for e in self._analyzed]
                        + preview_rois)
            stack = self.state.channel_stack(channel)
            plane_shape = stack.shape[1:]
            masks = self._masks_for(
                [(f"tmp{i}", s["verts"]) for i, s in enumerate(new_shapes)]
                + [(e["label"], e.get("verts")) for e in self._analyzed],
                plane_shape)
            result = pipeline.run_preview(stack, z, self.params,
                                          rois=preview_rois,
                                          threshold_rois=all_rois,
                                          masks=masks or None,
                                          project=self.state.is_stack)
            # Outline the thresholded masks in the new ROIs (no top-hat layer).
            self._set_mask_outline("spots (preview)", result.foreground)
            # Per-ROI spot counts -> warning colours. Analyzed ROIs use their
            # committed count; the new ROIs use this preview's count.
            counts = self._committed_counts()
            cents = np.asarray(result.centroids).reshape(-1, 2)
            for i, s in enumerate(new_shapes):
                m = masks.get(f"tmp{i}") if s["verts"] is not None else None
                if m is not None:
                    n = int(sum(1 for r, c in cents
                                if 0 <= int(r) < m.shape[0]
                                and 0 <= int(c) < m.shape[1]
                                and m[int(r), int(c)]))
                else:
                    r0, r1, c0, c1 = s["bounds"]
                    n = int(sum(1 for r, c in cents
                                if r0 <= r < r1 and c0 <= c < c1))
                counts[s["key"]] = n
            flagged = self._apply_roi_warnings(counts)
            warn = (f"  |  ⚠ {flagged} ROI(s) ≠2 spots"
                    if flagged else "")
            self.status_label.setText(
                f"Preview: {len(result.centroids)} spot(s) in "
                f"{len(new_shapes)} new ROI(s)  |  "
                f"{self._format_thresholds(result.roi_thresholds)}{warn}")
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Preview failed: {exc}")
        finally:
            self._running = False
            self._reassert_draw_mode()

    # -- reference-anchored, multi-channel detection -------------------- #
    def detect(self, *args):
        if not self._ready():
            return
        self._collect_params()

        invalid = self._invalid_positions()
        if invalid:
            notifications.show_warning(
                f"ROI shape(s) {invalid} aren't rectangles. Redraw them or use "
                "'Draw polygon ROI' before detecting.")
            self.status_label.setText(
                f"Fix ROI shape(s) {invalid}, then Detect")
            return
        if not self._valid_shapes():
            notifications.show_warning("Draw at least one ROI before detecting.")
            self.status_label.setText("Draw an ROI, then Detect")
            return
        new_shapes = self._unanalyzed()
        if not new_shapes:
            notifications.show_info("All drawn ROIs have already been analyzed.")
            return

        # Session-global labels (increment across files, never repeated);
        # assigned/kept by state, matched by shape key.
        ref = self.channel_selector.currentText()
        shapes = self._valid_shapes()
        labelled = self._save_rois()          # aligned with `shapes`
        info = {s["key"]: {"roi": roi, "verts": s["verts"]}
                for s, roi in zip(shapes, labelled)}
        new_rois = [info[s["key"]]["roi"] for s in new_shapes]

        names = self.state.meta.channel_names
        stacks = {name: self.state.channel_stack(i)
                  for i, name in enumerate(names)
                  if pipeline.is_measurable_channel(name)}
        if not stacks:
            notifications.show_warning("No measurable (non-brightfield) channels.")
            return

        # Pool the threshold across all drawn ROIs (analyzed + new) so the
        # "global" threshold uses every user-selected ROI in the session.
        threshold_rois = ([(*e["bounds"], e["label"]) for e in self._analyzed]
                          + new_rois)
        plane_shape = next(iter(stacks.values())).shape[1:]
        masks = self._masks_for(
            [(info[s["key"]]["roi"][4], s["verts"]) for s in shapes
             if s["key"] in info]
            + [(e["label"], e.get("verts")) for e in self._analyzed],
            plane_shape)
        self._running = True
        try:
            result = pipeline.run_multichannel(
                stacks, ref, self.params,
                pixel_size=self.state.meta.pixel_size,
                pixel_unit=self.state.meta.pixel_unit, rois=new_rois,
                threshold_rois=threshold_rois, masks=masks or None)
            self._register(result, [(info[s["key"]]["roi"], s["verts"])
                                    for s in new_shapes])
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Detection failed: {exc}")
            return
        finally:
            self._running = False

        self.state.last_result = result
        records = result.measurements.copy()
        if "z_plane" in records.columns:
            records["z_um"] = records["z_plane"] * self.state.meta.z_step
        fname = (self.state.current_path.name
                 if self.state.current_path else "unknown")
        self.state.record_measurements(fname, records)

        self._save_rois()
        self._update_pending()
        ref_thr = result.roi_thresholds.get(ref, {})
        n_flag = sum(1 for e in self._analyzed if len(e["centroids"]) != 2)
        warn = f"  |  ⚠ {n_flag} ROI(s) ≠2 spots" if n_flag else ""
        self.status_label.setText(
            f"+{len(result.measurements)} measurements from {len(new_rois)} new "
            f"ROI(s); {len(self._analyzed)} analyzed total  |  "
            f"{ref}: {self._format_thresholds(ref_thr)}{warn}")
        self.state.result_updated.emit()

    def _register(self, result, new_items):
        """Record new ROIs' centroids per label and refresh committed display.

        ``new_items`` is a list of ``((r0,r1,c0,c1,label), verts)``.  A spot is
        assigned to an ROI by its polygon mask (or bounding box for a
        rectangle).
        """
        self._clear_layer("spots (preview)")
        plane_shape = self._plane_shape()
        for roi, verts in new_items:
            r0, r1, c0, c1, label = roi
            if verts is not None:
                m = pipeline.roi_mask(None, verts, plane_shape)
                pts = [(int(z), int(r), int(c)) for z, r, c in result.centroids
                       if 0 <= int(r) < m.shape[0] and 0 <= int(c) < m.shape[1]
                       and m[int(r), int(c)]]
            else:
                pts = [(int(z), int(r), int(c)) for z, r, c in result.centroids
                       if r0 <= r < r1 and c0 <= c < c1]
            self._analyzed.append({
                "key": pipeline.shape_key((r0, r1, c0, c1), verts),
                "bounds": (r0, r1, c0, c1), "verts": verts, "label": label,
                "centroids": np.array(pts, dtype=int).reshape(-1, 3)})
        self._rebuild_committed()

    def _rebuild_committed(self):
        """Redraw committed spot outlines + ROI labels from analyzed ROIs."""
        if not self._analyzed:
            self._spot_vol = None
            self._clear_layer("spots")
            self._clear_layer("ROI labels")
            self._reassert_draw_mode()
            return
        Z, H, W = self.state.channel_stack(
            self.channel_selector.currentIndex()).shape
        vol = np.zeros((Z, H, W), dtype=np.uint16)
        radius = self.params.measure_radius
        yy, xx = np.ogrid[:H, :W]
        for e in self._analyzed:
            val = _roi_number(e["label"]) or 1
            for z, r, c in e["centroids"]:
                z, r, c = int(z), int(r), int(c)
                vol[z][(yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2] = val
        self._spot_vol = vol
        # Display is a 2-D max projection, so flatten the per-plane spot volume:
        # every spot's in-focus disk shows at once on the projection.
        self._set_mask_outline("spots", vol.max(axis=0))
        counts_by_label = {e["label"]: len(e["centroids"])
                           for e in self._analyzed}
        self._set_roi_labels([(*e["bounds"], e["label"])
                              for e in self._analyzed], counts_by_label)
        self._apply_roi_warnings(self._committed_counts())
        self._reassert_draw_mode()

    # -- cells: micro-sam segmentation + mom / daughter ----------------- #
    def _brightfield_index(self):
        names = self.state.meta.channel_names or []
        for i, n in enumerate(names):
            if not pipeline.is_measurable_channel(n):
                return i
        return None

    def _segment_cells(self, *args):
        if self.state.image is None:
            return
        idx = self._brightfield_index()
        if idx is None:
            notifications.show_warning(
                "No brightfield / transmitted channel found to segment.")
            return
        stack = self.state.channel_stack(idx)
        image2d = np.asarray(stack.max(axis=0) if stack.ndim == 3 else stack)
        try:
            from micro_sam.automatic_segmentation import (
                get_predictor_and_segmenter, automatic_instance_segmentation)
        except Exception:  # noqa: BLE001
            notifications.show_error(
                "micro-sam is not installed. Install it with "
                "`conda install -c conda-forge micro_sam` to segment cells.")
            return
        self.segment_btn.setEnabled(False)
        self.segment_btn.setText("Segmenting… (first run downloads a model)")
        try:
            predictor, segmenter = get_predictor_and_segmenter(
                model_type="vit_b_lm")
            labels = automatic_instance_segmentation(
                predictor=predictor, segmenter=segmenter, input_path=image2d,
                ndim=2, verbose=False)
            labels = np.asarray(labels).astype(np.int32)
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"micro-sam segmentation failed: {exc}")
            return
        finally:
            self.segment_btn.setEnabled(True)
            self.segment_btn.setText("Segment cells (micro-sam)")
        self.state.cell_labels = labels
        self.state.cell_roles = {}
        for e in self._analyzed:
            e.pop("daughter_cell", None)
            e.pop("spot_roles", None)
        self._show_cells_layer()
        self._recolor_cells()
        notifications.show_info(
            f"micro-sam found {int(labels.max())} cell(s). "
            "Detect spots, then Auto mom/bud.")

    def _show_cells_layer(self):
        if self.state.cell_labels is None:
            return
        self._clear_layer(CELLS_LAYER)
        layer = self.viewer.add_labels(self.state.cell_labels, name=CELLS_LAYER,
                                       opacity=0.35)
        layer.mouse_drag_callbacks.append(self._on_cell_clicked)

    def _click_rc(self, layer, event):
        """(row, col) of a click in image coordinates, or None."""
        try:
            dp = layer.world_to_data(event.position)
            return int(round(dp[-2])), int(round(dp[-1]))
        except Exception:  # noqa: BLE001
            return None

    def _entry_at_point(self, r, c):
        """Analyzed ROI entry whose region contains pixel (r, c), or None."""
        for e in self._analyzed:
            r0, r1, c0, c1 = e["bounds"]
            if not (r0 <= r <= r1 and c0 <= c <= c1):
                continue
            if e.get("verts") is not None:
                m = self._roi_mask_for(e)
                if m is not None and 0 <= r < m.shape[0] and 0 <= c < m.shape[1] \
                        and not m[r, c]:
                    continue
            return e
        return None

    def _on_cell_clicked(self, layer, event):
        if not (self._pick_mom or self._pick_daughter or self._pick_remove):
            return
        rc = self._click_rc(layer, event)
        if rc is None:
            return
        r, c = rc
        cell = cells.cell_at(self.state.cell_labels, r, c)
        if not cell:
            return
        if self._pick_remove:
            self._remove_cell_pick(cell)
        else:
            role = "mom" if self._pick_mom else "bud"
            self.state.cell_roles[cell] = role             # colour it even w/o spots
            entry = self._entry_at_point(r, c)
            if entry is not None:
                entry["mom_cell" if role == "mom" else "daughter_cell"] = cell
                self._assign_roi(entry)
        self._recolor_cells()
        self._annotate_spot_roles()

    def _remove_cell_pick(self, cell):
        """Un-tag a wrongly picked cell: drop its role/colour, clear any ROI
        override pointing at it, and blank the dot tags that came from it."""
        self.state.cell_roles.pop(cell, None)
        labels = self.state.cell_labels
        for e in self._analyzed:
            if e.get("mom_cell") == cell:
                e.pop("mom_cell", None)
            if e.get("daughter_cell") == cell:
                e.pop("daughter_cell", None)
            roles = e.get("spot_roles")
            if not roles:
                continue
            new = list(roles)
            for i, (z, r, c2) in enumerate(e["centroids"]):
                if cells.cell_at(labels, r, c2) == cell:
                    new[i] = ""
            e["spot_roles"] = new

    def _arm_pick(self):
        self._restore_roi_active()
        if self._pick_remove:
            notifications.show_info("Click a cell to remove its mom/bud tag.")
        elif self._pick_mom or self._pick_daughter:
            notifications.show_info(
                f"Click the {'mother' if self._pick_mom else 'bud'} cell.")

    def _toggle_pick_mom(self, *args):
        self._pick_mom = self.pick_mom_btn.isChecked()
        if self._pick_mom:
            self._pick_daughter = self._pick_remove = False
            self.pick_dau_btn.setChecked(False)
            self.pick_remove_btn.setChecked(False)
            self._disarm_draw()
        self._arm_pick()

    def _toggle_pick_daughter(self, *args):
        self._pick_daughter = self.pick_dau_btn.isChecked()
        if self._pick_daughter:
            self._pick_mom = self._pick_remove = False
            self.pick_mom_btn.setChecked(False)
            self.pick_remove_btn.setChecked(False)
            self._disarm_draw()
        self._arm_pick()

    def _toggle_pick_remove(self, *args):
        self._pick_remove = self.pick_remove_btn.isChecked()
        if self._pick_remove:
            self._pick_mom = self._pick_daughter = False
            self.pick_mom_btn.setChecked(False)
            self.pick_dau_btn.setChecked(False)
            self._disarm_draw()
        self._arm_pick()

    # -- draw a cell polygon when micro-sam misses one ------------------ #
    def _ensure_cell_labels(self):
        """Make sure there's a label image to paint into (blank if none yet)."""
        if self.state.cell_labels is None:
            self.state.cell_labels = np.zeros(self._plane_shape(), dtype=np.int32)
            self._show_cells_layer()

    def _ensure_draw_layer(self):
        if OVAL_LAYER in self.viewer.layers:
            return self.viewer.layers[OVAL_LAYER]
        return self.viewer.add_shapes(
            name=OVAL_LAYER, face_color="transparent", edge_color="white",
            edge_width=1)

    def _disarm_draw(self, commit=True):
        """Turn the draw tool off, baking any pending polygon(s) first."""
        if commit and self._oval_role:
            self._bake_pending(self._oval_role)
        self._oval_role = None
        self.oval_mom_btn.setChecked(False)
        self.oval_dau_btn.setChecked(False)

    def _arm_oval(self, role):
        if self._oval_role == role:              # toggle the active tool off
            self._disarm_draw(commit=True)
            self._restore_roi_active()
            return
        if self._oval_role:                      # switching role: commit first
            self._bake_pending(self._oval_role)
        self._oval_role = role
        self.oval_mom_btn.setChecked(role == "mom")
        self.oval_dau_btn.setChecked(role == "bud")
        self._pick_mom = self._pick_daughter = False
        self.pick_mom_btn.setChecked(False)
        self.pick_dau_btn.setChecked(False)
        self._ensure_cell_labels()
        layer = self._ensure_draw_layer()
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "add_polygon"
        except Exception:  # noqa: BLE001
            pass
        notifications.show_info(
            f"Draw a polygon around the {role} cell (double-click to finish). "
            f"Click 'Draw {role} cell' again to add it.")

    def _bake_pending(self, role):
        """Paint each drawn polygon into the label image as a new tagged cell."""
        if OVAL_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[OVAL_LAYER]
        labels = self.state.cell_labels
        baked = 0
        if labels is not None:
            for shp in list(layer.data):
                yx = np.asarray(shp, dtype=float)[:, -2:]
                if len(yx) < 3:
                    continue
                mask = pipeline.roi_mask(None, yx, labels.shape)
                if not mask.any():
                    continue
                new_label = int(labels.max()) + 1
                labels[mask] = new_label
                self.state.cell_roles[new_label] = role
                cr, cc = int(yx[:, 0].mean()), int(yx[:, 1].mean())
                entry = self._entry_at_point(cr, cc)
                if entry is not None:
                    entry["mom_cell" if role == "mom" else "daughter_cell"] = new_label
                    self._assign_roi(entry)
                baked += 1
        layer.data = []
        self._show_cells_layer()
        self._recolor_cells()
        self._annotate_spot_roles()
        if baked:
            notifications.show_info(f"Added {baked} {role} cell(s).")

    # -- manual spots --------------------------------------------------- #
    def _toggle_add_spot(self, *args):
        self._add_spot_mode = self.add_spot_btn.isChecked()
        if self._add_spot_mode:
            self._pick_mom = self._pick_daughter = self._pick_remove = False
            self.pick_mom_btn.setChecked(False)
            self.pick_dau_btn.setChecked(False)
            self.pick_remove_btn.setChecked(False)
            self._disarm_draw()
            self._stop_draw_mode()
            self._sync_mode_buttons()
            if ROI_LAYER in self.viewer.layers:
                try:
                    self.viewer.layers.selection.active = self.viewer.layers[ROI_LAYER]
                    self.viewer.layers[ROI_LAYER].mode = "pan_zoom"
                except Exception:  # noqa: BLE001
                    pass
            notifications.show_info(
                "Click the middle of a missed spot, inside its ROI.")
        else:
            self._restore_roi_active()

    def _on_add_spot_click(self, layer, event):
        if not self._add_spot_mode:
            return
        rc = self._click_rc(layer, event)
        if rc is None:
            return
        r, c = rc
        entry = self._entry_at_point(r, c)
        if entry is None:
            notifications.show_info("Click inside an analyzed ROI.")
            return
        region = entry["label"]
        fname = (self.state.current_path.name
                 if self.state.current_path else "unknown")
        pts = self.state.manual_spots.setdefault(fname, {}).setdefault(region, [])
        # Toggle: click near an existing manual spot removes it, else add one.
        rad = max(int(self.params.measure_radius), 3)
        hit = next((i for i, (pr, pc) in enumerate(pts)
                    if (pr - r) ** 2 + (pc - c) ** 2 <= rad * rad), None)
        if hit is not None:
            pts.pop(hit)
        else:
            pts.append((int(r), int(c)))
        if not pts:
            self.state.manual_spots[fname].pop(region, None)
        self._remeasure_roi(entry)

    def _remeasure_roi(self, entry):
        """Re-run one ROI including its manual spots; update display + table."""
        ref = self.channel_selector.currentText()
        names = self.state.meta.channel_names
        stacks = {nm: self.state.channel_stack(i) for i, nm in enumerate(names)
                  if pipeline.is_measurable_channel(nm)}
        if not stacks:
            return
        plane_shape = next(iter(stacks.values())).shape[1:]
        label = entry["label"]
        roi = (*entry["bounds"], label)
        threshold_rois = [(*e["bounds"], e["label"]) for e in self._analyzed]
        masks = self._masks_for(
            [(e["label"], e.get("verts")) for e in self._analyzed], plane_shape)
        fname = (self.state.current_path.name
                 if self.state.current_path else "unknown")
        region_manual = self.state.manual_spots.get(fname, {}).get(label)
        manual = {label: region_manual} if region_manual else None
        self._running = True
        try:
            result = pipeline.run_multichannel(
                stacks, ref, self.params,
                pixel_size=self.state.meta.pixel_size,
                pixel_unit=self.state.meta.pixel_unit,
                rois=[roi], threshold_rois=threshold_rois, masks=masks or None,
                manual_spots=manual)
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Re-measure failed: {exc}")
            return
        finally:
            self._running = False

        recs = result.measurements
        ref_rows = recs[recs["channel"] == ref] if "channel" in recs.columns \
            else recs
        pts = [(int(z), int(r), int(c)) for z, r, c in
               zip(ref_rows["z_plane"], ref_rows["row_px"], ref_rows["col_px"])]
        entry["centroids"] = np.array(pts, dtype=int).reshape(-1, 3)

        self.state.remove_region(fname, label)
        records = recs.copy()
        if not records.empty:
            if "z_plane" in records.columns:
                records["z_um"] = records["z_plane"] * self.state.meta.z_step
            self.state.record_measurements(fname, records)

        self._rebuild_committed()
        if self.state.cell_labels is not None:
            self._assign_roi(entry)
            self._recolor_cells()
            self._annotate_spot_roles()
        self.state.result_updated.emit()

    def _restore_roi_active(self):
        """Keep the right layer active after cell layers redraw: the oval layer
        while a Draw tool is armed, the cells layer while a Pick tool is armed
        (so clicks land on cells), otherwise the ROI layer (so the Select tool
        keeps working)."""
        if self._oval_role and OVAL_LAYER in self.viewer.layers:
            try:
                self.viewer.layers.selection.active = self.viewer.layers[OVAL_LAYER]
                self.viewer.layers[OVAL_LAYER].mode = "add_polygon"
            except Exception:  # noqa: BLE001
                pass
            return
        if (self._pick_mom or self._pick_daughter or self._pick_remove) \
                and CELLS_LAYER in self.viewer.layers:
            try:
                self.viewer.layers.selection.active = self.viewer.layers[CELLS_LAYER]
            except Exception:  # noqa: BLE001
                pass
            return
        self._reassert_draw_mode()
        if not self._draw_mode and ROI_LAYER in self.viewer.layers:
            try:
                self.viewer.layers.selection.active = self.viewer.layers[ROI_LAYER]
            except Exception:  # noqa: BLE001
                pass

    def _roi_mask_for(self, entry):
        if self.state.cell_labels is None:
            return None
        return pipeline.roi_mask(entry["bounds"], entry.get("verts"),
                                 self.state.cell_labels.shape)

    def _assign_roi(self, entry):
        labels = self.state.cell_labels
        if labels is None:
            return None
        spots = [(int(r), int(c)) for z, r, c in entry["centroids"]]
        res = cells.assign_pair(
            labels, spots, roi_mask=self._roi_mask_for(entry),
            daughter_cell=entry.get("daughter_cell"),
            mom_cell=entry.get("mom_cell"),
            pixel_size=float(self.state.meta.pixel_size or 1.0),
            gap_um=float(self.toward_gap.value()))
        if res["ok"]:
            entry["spot_roles"] = res["spot_roles"]
            if res["mom_cell"]:
                self.state.cell_roles[res["mom_cell"]] = "mom"
            if res["daughter_cell"]:
                self.state.cell_roles[res["daughter_cell"]] = "bud"
        return res

    def _auto_roles(self, entries):
        if self.state.cell_labels is None:
            notifications.show_warning("Segment cells first.")
            return
        done = need = bad = 0
        for e in entries:
            res = self._assign_roi(e)
            if res is None or not res["ok"]:
                bad += 1
            elif res["needs_daughter"]:
                need += 1
            else:
                done += 1
        self._recolor_cells()
        self._annotate_spot_roles()
        notifications.show_info(
            f"mom/bud: {done} tagged, {need} both-in-mom (pick the bud to "
            f"refine), {bad} skipped.")

    def _color_pair_by_size(self, bounds, verts):
        """Colour the two biggest cells in one ROI as mom / daughter. Returns
        True if at least a mother cell was found."""
        labels = self.state.cell_labels
        mask = pipeline.roi_mask(bounds, verts, labels.shape)
        mom, dau = cells.pair_by_size(labels, mask)
        if mom:
            self.state.cell_roles[mom] = "mom"
        if dau:
            self.state.cell_roles[dau] = "bud"
        return mom is not None

    def _auto_all_rois(self, *args):
        if self.state.cell_labels is None:
            notifications.show_warning("Segment cells first.")
            return
        if self._analyzed:                       # spots detected: tag dots too
            self._auto_roles(self._analyzed)
            return
        # No spots yet: just colour mom / daughter cells from the drawn ROIs.
        shapes = self._valid_shapes()
        if not shapes:
            notifications.show_warning(
                "Draw at least one ROI (or detect spots) first.")
            return
        n = sum(self._color_pair_by_size(s["bounds"], s["verts"]) for s in shapes)
        self._recolor_cells()
        notifications.show_info(
            f"Coloured mom / bud cells in {n} ROI(s). "
            "Detect spots to tag the dots.")

    def _recolor_cells(self):
        for n in ("mom cells", "bud cells", "cell labels"):
            self._clear_layer(n)
        labels = self.state.cell_labels
        if labels is None:
            return
        from skimage.segmentation import find_boundaries
        mom, dau = cells.role_masks(labels, self.state.cell_roles)
        if mom.any():
            self.viewer.add_image(
                find_boundaries(mom, mode="outer").astype(float),
                name="mom cells", colormap="cyan", blending="additive")
        if dau.any():
            self.viewer.add_image(
                find_boundaries(dau, mode="outer").astype(float),
                name="bud cells", colormap="yellow", blending="additive")
        # Text label ("mom"/"bud") at each tagged cell's centroid, so the roles
        # are readable even without spot tags.
        pts, txt = [], []
        for cell, role in self.state.cell_roles.items():
            ys, xs = np.where(labels == cell)
            if len(ys):
                pts.append([float(ys.mean()), float(xs.mean())])
                txt.append(role)
        if pts:
            self.viewer.add_points(
                np.array(pts, dtype=float), name="cell labels", size=1,
                face_color="transparent", border_color="transparent",
                properties={"role": txt},
                text={"string": "{role}", "size": 11, "color": "white",
                      "anchor": "center"})
        self._restore_roi_active()

    def _annotate_spot_roles(self):
        name = "spot roles"
        self._clear_layer(name)
        pts, txt = [], []
        for e in self._analyzed:
            roles = e.get("spot_roles")
            if not roles:
                continue
            for (z, r, c), role in zip(e["centroids"], roles):
                if role:
                    pts.append([int(r), int(c)])
                    txt.append(role)
        if not pts:
            return
        self.viewer.add_points(
            np.array(pts, dtype=float), name=name, size=1,
            face_color="transparent", border_color="transparent",
            properties={"role": txt},
            text={"string": "{role}", "size": 8, "color": "white",
                  "anchor": "lower_left"})
        self._restore_roi_active()

    # -- viewer helpers ------------------------------------------------- #
    def _set_image_layer(self, name, data, cmap, visible):
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if layer.data.shape == data.shape:
                layer.data = data
                return
            self.viewer.layers.remove(name)
        self.viewer.add_image(data, name=name, colormap=cmap, visible=visible,
                              blending="additive")

    def _set_points(self, name, data, border_color="red", text=None):
        data = np.asarray(data)
        kwargs = dict(size=8, face_color="transparent", border_color=border_color)
        if text is not None and len(text) == len(data):
            kwargs["properties"] = {"region": list(text)}
            kwargs["text"] = {"string": "{region}", "size": 8, "color": "white"}
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        if len(data):
            self.viewer.add_points(data, name=name, **kwargs)

    def _set_mask_outline(self, name, mask):
        """Show the outline of a 2-D mask as a Labels contour layer."""
        data = np.asarray(mask).astype(np.uint8)
        if not data.any():
            self._clear_layer(name)
            return
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if hasattr(layer, "contour") and layer.data.shape == data.shape:
                layer.data = data
                return
            self.viewer.layers.remove(name)
        layer = self.viewer.add_labels(data, name=name, opacity=0.9)
        layer.contour = 2

    def _set_roi_labels(self, rois, counts=None):
        """Place each ROI's label (with spot count) at a corner of its box."""
        name = "ROI labels"
        if not rois:
            self._clear_layer(name)
            return
        points = np.array([[min(r0, r1), min(c0, c1)]
                           for r0, r1, c0, c1, _ in rois], dtype=float)
        labels = []
        for *_, lab in rois:
            if counts is not None and lab in counts:
                n = counts[lab]
                labels.append(f"{lab}: {n}" + (" ⚠" if n != 2 else ""))
            else:
                labels.append(str(lab))
        self._clear_layer(name)
        self.viewer.add_points(
            points, name=name, size=1, face_color="transparent",
            border_color="transparent",
            properties={"label": labels},
            text={"string": "{label}", "size": 11, "color": "yellow",
                  "anchor": "upper_left"})

    def _clear_layer(self, name):
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)

    @staticmethod
    def _format_thresholds(thresholds):
        if not thresholds:
            return "thresholds: –"
        parts = [f"{lab}={t:.3g}" if t is not None else f"{lab}=n/a"
                 for lab, t in thresholds.items()]
        return "thresholds: " + ", ".join(parts)
