"""Spot detection & measurement dock widget.

Grouped controls for each algorithm stage.  Tuning a control updates a live
single-plane *preview* (top-hat, foreground, current-plane maxima).  The
**Detect spots** button runs the full-stack pipeline: per-plane detection,
cross-plane linking, in-focus selection, disk measurement with a concentric
background ring, and region assignment for any local-threshold ROIs.
"""
from __future__ import annotations

import napari.utils.notifications as notifications
import numpy as np
from qtpy.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from . import pipeline
from .filters import SMOOTHING_METHODS
from .pipeline import THRESHOLD_METHODS, PipelineParams
from .state import AppState

ROI_LAYER = "local_ROI"
# Distinct border colours cycled per region for spot annotation.
_REGION_COLORS = ["red", "cyan", "yellow", "lime", "orange", "magenta"]


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
        # {"bounds": (r0,r1,c0,c1), "label": "ROIn", "centroids": (N,3) array}.
        self._analyzed = []
        self._spot_vol = None            # accumulated detected-region label vol

        layout = QVBoxLayout(self)

        # -- channel -------------------------------------------------------- #
        chan_box = QGroupBox("Channel")
        chan_form = QFormLayout(chan_box)
        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self._on_param_changed)
        chan_form.addRow("Measure channel", self.channel_selector)
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
        smooth_form.addRow("Method", self.smooth_method)
        smooth_form.addRow("Filter size", self.smooth_size)

        tophat_box = QGroupBox("2. White top-hat")
        tophat_form = QFormLayout(tophat_box)
        self.tophat_size = QDoubleSpinBox()
        self.tophat_size.setRange(1.0, 200.0)
        self.tophat_size.setValue(self.params.tophat_size)
        tophat_form.addRow("Filter size", self.tophat_size)

        thresh_box = QGroupBox("3. Thresholding")
        thresh_form = QFormLayout(thresh_box)
        self.thresh_method = QComboBox()
        self.thresh_method.addItems(THRESHOLD_METHODS)
        thresh_form.addRow("Method", self.thresh_method)

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
        thresh_form.addRow("ilastik prob.", self.ilastik_prob)
        self.min_mask_size = QSpinBox()
        self.min_mask_size.setRange(1, 100000)
        self.min_mask_size.setValue(self.params.min_mask_size)
        thresh_form.addRow("Min mask size", self.min_mask_size)

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
            "Draw one or more rectangles. Detection and measurement happen\n"
            "ONLY inside these ROIs; each is thresholded from its own stack\n"
            "histogram. Spots outside every ROI are ignored."))
        roi_btn_row = QHBoxLayout()
        draw_btn = QPushButton("Draw ROI(s)")
        clear_btn = QPushButton("Clear ROIs")
        draw_btn.clicked.connect(self._draw_rois)
        clear_btn.clicked.connect(self._clear_rois)
        roi_btn_row.addWidget(draw_btn)
        roi_btn_row.addWidget(clear_btn)
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
        maxima_form.addRow("Min. distance", self.min_distance)
        maxima_form.addRow("Rel. threshold", self.peak_rel)
        maxima_form.addRow("Link dist. (XY px)", self.link_dist)
        maxima_form.addRow("Min Z-linkage", self.min_link)
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
        meas_form.addRow("Spot radius", self.measure_radius)
        meas_form.addRow("Ring gap", self.bkg_gap)
        meas_form.addRow("Ring width", self.bkg_width)
        layout.addWidget(meas_box)

        # Chromatic offset now lives in the Edit-metadata dialog (File IO).

        # -- detect --------------------------------------------------------- #
        self.detect_btn = QPushButton("Detect spots (stack)")
        self.detect_btn.clicked.connect(self.detect)
        layout.addWidget(self.detect_btn)
        self.status_label = QLabel("No spots detected yet")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
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
        # Z-scrolling refreshes the preview but keeps any detection on screen.
        self.viewer.dims.events.current_step.connect(lambda e: self._preview())

        self.setEnabled(False)

    # ------------------------------------------------------------------ #
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
        self._reset_analysis()
        self._restore_rois()
        self._preview()

    def _reset_analysis(self):
        """Forget analyzed ROIs and remove committed detection layers.

        The ROI label counter is session-global (on state) and is NOT reset
        here, so ROI indices keep incrementing across files.
        """
        self._analyzed = []
        self._spot_vol = None
        self.state.pending_rois = 0
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
    def _draw_rois(self):
        if ROI_LAYER not in self.viewer.layers:
            layer = self.viewer.add_shapes(
                name=ROI_LAYER, face_color="transparent", edge_color="yellow",
                edge_width=2)
            # React to ROI edits: drop deleted ROIs' data, preview new ones.
            layer.events.data.connect(lambda e: self._on_rois_changed())
        layer = self.viewer.layers[ROI_LAYER]
        self.viewer.layers.selection.active = layer
        layer.mode = "add_rectangle"
        notifications.show_info("Draw rectangles; preview spots appear inside them.")

    def _clear_rois(self):
        if ROI_LAYER in self.viewer.layers:
            self.viewer.layers.remove(ROI_LAYER)
        self._reset_analysis()
        self._preview()

    def _roi_bounds_list(self):
        """Current ROI rectangles as (r0, r1, c0, c1) bounds, in layer order."""
        bounds = []
        if ROI_LAYER not in self.viewer.layers:
            return bounds
        for shp in self.viewer.layers[ROI_LAYER].data:
            coords = np.asarray(shp)
            ys, xs = coords[:, -2], coords[:, -1]
            bounds.append((int(ys.min()), int(ys.max()),
                           int(xs.min()), int(xs.max())))
        return bounds

    def _unanalyzed_bounds(self):
        done = {e["bounds"] for e in self._analyzed}
        return [b for b in self._roi_bounds_list() if b not in done]

    def _update_pending(self):
        """Publish how many drawn ROIs are still awaiting analysis."""
        self.state.pending_rois = len(self._unanalyzed_bounds())

    def _save_rois(self):
        """Persist the current file's ROIs with session-global labels."""
        if self.state.current_path is None:
            return self.state.assign_roi_labels("", [], "")
        return self.state.assign_roi_labels(
            self.state.current_path.name, self._roi_bounds_list(),
            self.channel_selector.currentText())

    def _restore_rois(self):
        """Recreate the ROI shapes layer for a file that has saved ROIs."""
        if self.state.current_path is None:
            return
        saved = self.state.session_rois.get(self.state.current_path.name)
        if not saved:
            return
        rects = [np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]], dtype=float)
                 for (r0, r1, c0, c1, _label) in saved]
        try:
            if ROI_LAYER in self.viewer.layers:
                self.viewer.layers.remove(ROI_LAYER)
            layer = self.viewer.add_shapes(
                rects, shape_type="rectangle", name=ROI_LAYER,
                face_color="transparent", edge_color="yellow", edge_width=2)
            layer.events.data.connect(lambda e: self._on_rois_changed())
        except Exception:  # noqa: BLE001
            pass

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
        current = set(self._roi_bounds_list())
        removed = [e for e in self._analyzed if e["bounds"] not in current]
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
        new_bounds = self._unanalyzed_bounds()
        self._update_pending()
        self._running = True
        try:
            z = (int(self.viewer.dims.current_step[0])
                 if self.viewer.dims.ndim >= 3 else 0)
            if not new_bounds:
                self._clear_layer("spots (preview)")
                if not self._analyzed:
                    self.status_label.setText("Draw ROIs to preview/detect spots")
                return
            preview_rois = [(*b, f"tmp{i}") for i, b in enumerate(new_bounds)]
            # Pool the threshold across every drawn ROI (analyzed + new).
            all_rois = ([(*e["bounds"], e["label"]) for e in self._analyzed]
                        + preview_rois)
            stack = self.state.channel_stack(channel)
            result = pipeline.run_preview(stack, z, self.params,
                                          rois=preview_rois,
                                          threshold_rois=all_rois)
            # Outline the thresholded masks in the new ROIs (no top-hat layer).
            self._set_mask_outline("spots (preview)", result.foreground)
            self.status_label.setText(
                f"Preview: {len(result.centroids)} spot(s) in "
                f"{len(new_bounds)} new ROI(s)  |  "
                f"{self._format_thresholds(result.roi_thresholds)}")
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Preview failed: {exc}")
        finally:
            self._running = False

    # -- reference-anchored, multi-channel detection -------------------- #
    def detect(self, *args):
        if not self._ready():
            return
        self._collect_params()
        if not self._roi_bounds_list():
            notifications.show_warning("Draw at least one ROI before detecting.")
            self.status_label.setText("Draw an ROI, then Detect")
            return
        new_bounds = self._unanalyzed_bounds()
        if not new_bounds:
            notifications.show_info("All drawn ROIs have already been analyzed.")
            return
        # Session-global labels (increment across files, never repeated);
        # assigned/kept by state, matched by bounds.
        ref = self.channel_selector.currentText()
        labelled = self._save_rois()          # [(r0,r1,c0,c1,label), ...]
        by_bounds = {roi[:4]: roi for roi in labelled}
        new_rois = [by_bounds[b] for b in new_bounds if b in by_bounds]

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
        self._running = True
        try:
            result = pipeline.run_multichannel(
                stacks, ref, self.params,
                pixel_size=self.state.meta.pixel_size,
                pixel_unit=self.state.meta.pixel_unit, rois=new_rois,
                threshold_rois=threshold_rois)
            self._register(result, new_rois)
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
        self.status_label.setText(
            f"+{len(result.measurements)} measurements from {len(new_rois)} new "
            f"ROI(s); {len(self._analyzed)} analyzed total  |  "
            f"{ref}: {self._format_thresholds(ref_thr)}")
        self.state.result_updated.emit()

    def _register(self, result, new_rois):
        """Record new ROIs' centroids per label and refresh committed display."""
        self._clear_layer("spots (preview)")
        for r0, r1, c0, c1, label in new_rois:
            pts = [(int(z), int(r), int(c)) for z, r, c in result.centroids
                   if r0 <= r < r1 and c0 <= c < c1]
            self._analyzed.append({
                "bounds": (r0, r1, c0, c1), "label": label,
                "centroids": np.array(pts, dtype=int).reshape(-1, 3)})
        self._rebuild_committed()

    def _rebuild_committed(self):
        """Redraw committed spot outlines + ROI labels from analyzed ROIs."""
        if not self._analyzed:
            self._spot_vol = None
            self._clear_layer("spots")
            self._clear_layer("ROI labels")
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
        self._set_mask_outline("spots", vol)
        self._set_roi_labels([(*e["bounds"], e["label"]) for e in self._analyzed])

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

    def _set_roi_labels(self, rois):
        """Place each ROI's label at one corner (top-left) of the rectangle."""
        name = "ROI labels"
        if not rois:
            self._clear_layer(name)
            return
        points = np.array([[min(r0, r1), min(c0, c1)]
                           for r0, r1, c0, c1, _ in rois], dtype=float)
        labels = [lab for *_, lab in rois]
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
