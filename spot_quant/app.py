"""Assemble the napari viewer with a tabbed dock widget.

File IO and spot detection share one tab; measurement has its own.
"""
from __future__ import annotations

import napari
from qtpy.QtWidgets import (
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from .detection_panel import DetectionPanel
from .io_panel import FileIOPanel
from .measurement_panel import MeasurementPanel
from .state import AppState


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


def launch():
    viewer = napari.Viewer(title="Spot Quant")
    state = AppState()
    state.viewer = viewer

    # File IO + detection share one scrollable tab.
    io_detect = QWidget()
    layout = QVBoxLayout(io_detect)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(FileIOPanel(state))
    layout.addWidget(DetectionPanel(state))

    tabs = QTabWidget()
    tabs.addTab(_scroll(io_detect), "Files & detection")
    tabs.addTab(_scroll(MeasurementPanel(state)), "Measurement")

    viewer.window.add_dock_widget(tabs, name="Spot Quant", area="right")

    napari.run()
    return viewer


if __name__ == "__main__":
    launch()
