"""File-IO dock widget: folder browsing, file list and metadata editing."""
from __future__ import annotations

from pathlib import Path

import napari.utils.notifications as notifications
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from . import pipeline, session_io
from .metadata import list_image_files, load_image
from .state import AppState

# Default per-channel colormaps cycled through for fluorescence channels.
_CMAPS = ["green", "magenta", "cyan", "yellow", "red", "blue"]


class FileIOPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.viewer = state.viewer

        layout = QVBoxLayout(self)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        select_btn = QPushButton("Select folder…")
        select_btn.clicked.connect(self._choose_folder)

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self._open_selected)

        self.meta_btn = QPushButton("Edit image metadata…")
        self.meta_btn.setEnabled(False)
        self.meta_btn.clicked.connect(self._edit_metadata)

        self.import_btn = QPushButton("Import ROIs / session…")
        self.import_btn.clicked.connect(self._import_session)

        files_box = QGroupBox("Images (tif / tiff / nd2)")
        box_layout = QVBoxLayout(files_box)
        box_layout.addWidget(QLabel("Double-click a file to open it"))
        box_layout.addWidget(self.file_list)

        layout.addWidget(select_btn)
        layout.addWidget(self.folder_label)
        layout.addWidget(files_box)
        layout.addWidget(self.meta_btn)
        layout.addWidget(self.import_btn)
        layout.addStretch(1)

    # ------------------------------------------------------------------ #
    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not path:
            return
        folder = Path(path)
        files = list_image_files(folder)
        self.file_list.clear()
        if not files:
            notifications.show_warning(f"No tif/tiff/nd2 files in {folder}")
        for f in files:
            self.file_list.addItem(f.name)
        self.state.folder = folder
        self.folder_label.setText(str(folder))

    def _open_selected(self, item):
        if self.state.folder is None:
            return
        path = self.state.folder / item.text()
        try:
            image, meta = load_image(path)
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Could not open {path.name}: {exc}")
            return

        # Prompt for acquisition spacing if it wasn't in the file, so the
        # z-step-dependent defaults (e.g. min Z-linkage) are set correctly.
        is_stack = image.shape[0] >= 1 and image[0].ndim == 3
        if is_stack and not meta.z_step_known:
            _MissingMetaDialog(meta, self).exec_()

        # Guard reactive panels while we swap the viewer's layers: clearing
        # layers fires dims events, and a reactive re-run that adds layers
        # mid-clear crashes napari.
        self.state.loading = True
        try:
            self._display(image, meta)
        finally:
            self.state.loading = False

        # Now that the viewer is in a consistent state, publish the image so
        # the detection panel runs exactly once.
        self.state.set_image(path, image, meta)
        self.meta_btn.setEnabled(True)
        notifications.show_info(f"Opened {path.name}")

    def _display(self, image, meta):
        self.viewer.layers.clear()
        names = [names_i if names_i else f"Ch{i}"
                 for i, names_i in enumerate(
                     meta.channel_names or [None] * image.shape[0])]

        fluor_idx = [i for i in range(image.shape[0])
                     if pipeline.is_measurable_channel(names[i])]
        other_idx = [i for i in range(image.shape[0]) if i not in fluor_idx]
        # Reference channel = the persisted selection if present, else the
        # first fluorescence channel.
        ref_name = getattr(self.state, "ref_channel", None)
        if ref_name in names and pipeline.is_measurable_channel(ref_name):
            ref_i = names.index(ref_name)
        else:
            ref_i = fluor_idx[0] if fluor_idx else None

        # Z-stacks are shown flattened as a **max projection** (display only);
        # the full stack stays on ``state`` for detection/measurement.
        def _project(data):
            return data.max(axis=0) if data.ndim == 3 else data

        # Add transmitted / brightfield / phase FIRST so it sits at the bottom.
        for i in other_idx:
            self.viewer.add_image(_project(image[i]), name=names[i],
                                  colormap="gray")

        # Then fluorescence channels on top.
        for j, i in enumerate(fluor_idx):
            data = _project(image[i])
            lo, hi = float(np.min(data)), float(np.max(data))
            # Auto-contrast upper limit: a low fraction of max so dim spots stay
            # visible. Reference channel -> [min, 0.3*max]; others -> [min, 0.4*max].
            frac = 0.3 if i == ref_i else 0.4
            hi_lim = frac * hi if hi > lo else lo + 1.0
            if hi_lim <= lo:
                hi_lim = lo + 1.0
            self.viewer.add_image(
                data, name=names[i], colormap=_CMAPS[j % len(_CMAPS)],
                blending="additive", contrast_limits=[lo, hi_lim])

    # ------------------------------------------------------------------ #
    def _import_session(self):
        """Load ROIs (and settings) from a previous export to resume work.

        An XLSX with an 'ROIs' sheet restores exactly; a plain measurements CSV
        (or an XLSX without that sheet) rebuilds approximate rectangles from the
        spot positions and warns that it is lossy.
        """
        start = str(self.state.folder) if self.state.folder else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import ROIs / session", start,
            "Session / measurements (*.xlsx *.csv)")
        if not path:
            return
        try:
            data, approx = self._read_session(path)
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Import failed: {exc}")
            return
        n_files = len([v for v in data.get("session_rois", {}).values() if v])
        n_rois = sum(len(v) for v in data.get("session_rois", {}).values())
        if n_rois == 0:
            notifications.show_warning("No ROIs found to import.")
            return
        if approx:
            QMessageBox.warning(
                self, "Approximate import",
                f"Rebuilt {n_rois} rectangular ROI(s) from spot positions in "
                f"{n_files} file(s).\n\nThis is approximate: ROIs with no spots, "
                "polygon shapes and the original detection settings could not "
                "be recovered. Re-check the boxes before detecting.")
        self.state.import_session(data)
        notifications.show_info(
            f"Imported {n_rois} ROI(s) across {n_files} file(s)"
            + (" (approximate)" if approx else "")
            + ". Open a file to see its ROIs.")

    def _read_session(self, path):
        """Return ``(session_dict, approx)`` for an xlsx/csv import."""
        suffix = Path(path).suffix.lower()
        if suffix in (".xlsx", ".xlsm"):
            xls = pd.ExcelFile(path)
            if session_io.ROI_SHEET in xls.sheet_names:
                roi_df = xls.parse(session_io.ROI_SHEET)
                settings_df = (xls.parse(session_io.SETTINGS_SHEET)
                               if session_io.SETTINGS_SHEET in xls.sheet_names
                               else None)
                return session_io.session_from_sheets(
                    roi_df, settings_df, pipeline.PipelineParams), False
            df = xls.parse(xls.sheet_names[0])
            return session_io.session_from_measurements(df), True
        df = pd.read_csv(path)
        return session_io.session_from_measurements(df), True

    def _edit_metadata(self):
        if self.state.image is None:
            return
        dlg = _MetadataDialog(self.state.meta, self.state.n_channels,
                              self.state, self)
        if dlg.exec_() == QDialog.Accepted:
            dlg.apply_to(self.state.meta)
            self.state.metadata_changed.emit()
            notifications.show_info("Metadata updated")


class _MetadataDialog(QDialog):
    """Editable form for channel names, pixel size, magnification, time step,
    and the chromatic-aberration offset for non-reference channels."""

    def __init__(self, meta, n_channels, state=None, parent=None):
        self.state = state
        super().__init__(parent)
        self.setWindowTitle("Edit image metadata")
        form = QFormLayout(self)

        self.channel_edits = []
        names = list(meta.channel_names) + [
            f"Ch{i}" for i in range(len(meta.channel_names), n_channels)
        ]
        for i in range(n_channels):
            edit = QLineEdit(names[i])
            self.channel_edits.append(edit)
            form.addRow(f"Channel {i} name", edit)

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(5)
        self.pixel_size.setRange(0.0, 1e6)
        self.pixel_size.setValue(meta.pixel_size)
        self.pixel_size.setSuffix(f" {meta.pixel_unit}/px")
        form.addRow("Pixel size", self.pixel_size)

        self.magnification = QDoubleSpinBox()
        self.magnification.setRange(0.0, 1e4)
        self.magnification.setValue(meta.magnification)
        self.magnification.setSuffix("x")
        form.addRow("Magnification", self.magnification)

        self.time_step = QDoubleSpinBox()
        self.time_step.setDecimals(4)
        self.time_step.setRange(0.0, 1e6)
        self.time_step.setValue(meta.time_step)
        self.time_step.setSuffix(" s")
        form.addRow("Time step", self.time_step)

        self.z_step = QDoubleSpinBox()
        self.z_step.setDecimals(5)
        self.z_step.setRange(0.0, 1e6)
        self.z_step.setValue(meta.z_step)
        self.z_step.setSuffix(f" {meta.pixel_unit}/plane")
        form.addRow("Z-step", self.z_step)

        # Chromatic-aberration offset applied to non-reference channels.
        form.addRow(QLabel("— Chromatic offset (non-reference channels) —"))
        self.offset_z = QSpinBox(); self.offset_z.setRange(-100, 100)
        self.offset_y = QSpinBox(); self.offset_y.setRange(-100, 100)
        self.offset_x = QSpinBox(); self.offset_x.setRange(-100, 100)
        self.offset_z.setValue(int(getattr(state, "offset_z", 0)))
        self.offset_y.setValue(int(getattr(state, "offset_y", 0)))
        self.offset_x.setValue(int(getattr(state, "offset_x", 0)))
        form.addRow("Δz (planes)", self.offset_z)
        form.addRow("Δy (px)", self.offset_y)
        form.addRow("Δx (px)", self.offset_x)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def apply_to(self, meta):
        meta.channel_names = [e.text() for e in self.channel_edits]
        meta.pixel_size = self.pixel_size.value()
        meta.magnification = self.magnification.value()
        meta.time_step = self.time_step.value()
        meta.z_step = self.z_step.value()
        if self.state is not None:
            self.state.offset_z = self.offset_z.value()
            self.state.offset_y = self.offset_y.value()
            self.state.offset_x = self.offset_x.value()


class _MissingMetaDialog(QDialog):
    """Prompt for Z-step (nm) / time interval when the file lacks them."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.setWindowTitle("Enter acquisition spacing")
        form = QFormLayout(self)
        form.addRow(QLabel(
            "This file has no z-step / time metadata.\n"
            "Enter the values used during acquisition."))

        self.z_step_nm = QDoubleSpinBox()
        self.z_step_nm.setDecimals(1)
        self.z_step_nm.setRange(0.0, 1e7)
        self.z_step_nm.setValue(meta.z_step * 1000.0)   # microns -> nm
        self.z_step_nm.setSuffix(" nm")
        form.addRow("Z-step", self.z_step_nm)

        self.time_step = QDoubleSpinBox()
        self.time_step.setDecimals(4)
        self.time_step.setRange(0.0, 1e6)
        self.time_step.setValue(meta.time_step)
        self.time_step.setSuffix(" s")
        form.addRow("Time interval", self.time_step)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _accept(self):
        if self.z_step_nm.value() > 0:
            self.meta.z_step = self.z_step_nm.value() / 1000.0   # nm -> microns
            self.meta.z_step_known = True
        self.meta.time_step = self.time_step.value()
        self.meta.time_step_known = True
        self.accept()
