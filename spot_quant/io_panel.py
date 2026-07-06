"""File-IO dock widget: folder browsing, file list and metadata editing."""
from __future__ import annotations

from pathlib import Path

import napari.utils.notifications as notifications
import numpy as np
from qtpy.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QListWidget, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from . import pipeline
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

        files_box = QGroupBox("Images (tif / tiff / nd2)")
        box_layout = QVBoxLayout(files_box)
        box_layout.addWidget(QLabel("Double-click a file to open it"))
        box_layout.addWidget(self.file_list)

        layout.addWidget(select_btn)
        layout.addWidget(self.folder_label)
        layout.addWidget(files_box)
        layout.addWidget(self.meta_btn)
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

        # Add transmitted / brightfield / phase FIRST so it sits at the bottom.
        for i in other_idx:
            self.viewer.add_image(image[i], name=names[i], colormap="gray")

        # Then fluorescence channels on top.
        for j, i in enumerate(fluor_idx):
            data = image[i]
            lo, hi = float(np.min(data)), float(np.max(data))
            # Reference channel -> [min, 0.7*max]; others -> [min, 0.8*max].
            frac = 0.7 if i == ref_i else 0.8
            hi_lim = frac * hi if hi > lo else lo + 1.0
            if hi_lim <= lo:
                hi_lim = lo + 1.0
            self.viewer.add_image(
                data, name=names[i], colormap=_CMAPS[j % len(_CMAPS)],
                blending="additive", contrast_limits=[lo, hi_lim])

    # ------------------------------------------------------------------ #
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
