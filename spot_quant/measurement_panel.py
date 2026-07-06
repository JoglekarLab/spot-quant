"""Measurement tab: compiled session table, summary and export.

Records from every analysed file accumulate into one long-format table (one
row per spot per channel, tagged with ``filename`` and ``region``), so a
multi-file / multi-ROI workflow exports to a single CSV/XLSX that is trivial
to re-import for downstream analysis.
"""
from __future__ import annotations

from pathlib import Path

import napari.utils.notifications as notifications
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from . import pipeline, scan
from .state import AppState


class MeasurementPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)

        summary_box = QGroupBox("Session summary")
        summary_layout = QVBoxLayout(summary_box)
        self.summary_label = QLabel("No data recorded yet")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(summary_box)

        layout.addWidget(QLabel("Compiled measurements (all analysed files):"))
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV…")
        self.export_xlsx_btn = QPushButton("Export XLSX…")
        self.clear_btn = QPushButton("Clear session")
        self.export_csv_btn.clicked.connect(lambda: self._export("csv"))
        self.export_xlsx_btn.clicked.connect(lambda: self._export("xlsx"))
        self.clear_btn.clicked.connect(self.state.clear_session)
        btn_row.addWidget(self.export_csv_btn)
        btn_row.addWidget(self.export_xlsx_btn)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.linescan_check = QCheckBox(
            "Also export line-scans (two-spot ROIs) → *_linescans.xlsx")
        layout.addWidget(self.linescan_check)

        self.state.result_updated.connect(self._refresh)
        self.state.session_changed.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------ #
    def _refresh(self):
        df = self.state.session_table()
        n_files = len(self.state.session_records)
        if df.empty:
            self.summary_label.setText("No data recorded yet")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        n_spots = (df.groupby(["filename", "channel"]).ngroups
                   if {"filename", "channel"} <= set(df.columns) else len(df))
        n_rois = df["region"].nunique() if "region" in df.columns else 0
        self.summary_label.setText(
            f"{n_files} file(s)  |  {len(df)} measurement rows  |  "
            f"{n_rois} distinct ROI label(s) across files")
        self._fill_table(df)

    def _fill_table(self, df):
        self.table.clear()
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                val = df.iloc[r, c]
                text = f"{val:.4g}" if isinstance(val, float) else str(val)
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _recompute_or_fallback(self, with_linescans):
        """Re-run the session with a global threshold; fall back to the
        incrementally recorded table if a re-run isn't possible.

        Returns ``(dataframe, linescans_or_None)``.
        """
        n_rois = sum(len(b) for b in self.state.session_rois.values())
        if n_rois:
            n_files = len([b for b in self.state.session_rois.values() if b])
            reply = QMessageBox.question(
                self, "Re-analyze session",
                f"Re-run detection on {n_rois} ROI(s) across {n_files} file(s) "
                "using one global threshold pooled from all ROIs, then export?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return None, None
            try:
                df, thresholds, linescans = self.state.recompute_session(
                    with_linescans=with_linescans)
            except Exception as exc:  # noqa: BLE001
                notifications.show_error(f"Session re-run failed: {exc}")
                return self.state.session_table(), None
            if df is not None and not df.empty:
                notifications.show_info(
                    "Global thresholds: "
                    + ", ".join(f"{k}={v:.3g}" for k, v in thresholds.items()))
                return df, linescans
        # Fallback: the per-file records already accumulated (no line scans).
        return self.state.session_table(), None

    def _export(self, fmt):
        do_linescan = self.linescan_check.isChecked()
        # Re-run the whole session so the threshold is global across every ROI
        # of every file, then export that result.
        df, linescans = self._recompute_or_fallback(with_linescans=do_linescan)
        if df is None or df.empty:
            notifications.show_warning("No measurements to export.")
            return
        # Pair each spot's per-channel intensities and push non-two-spot ROIs
        # to the end of the report.
        df = pipeline.build_report(df)
        default = f"spot_measurements.{fmt}"
        if self.state.folder is not None:
            default = str(Path(self.state.folder) / default)
        filt = "CSV (*.csv)" if fmt == "csv" else "Excel (*.xlsx)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export compiled measurements", default, filt)
        if not path:
            return
        try:
            if fmt == "xlsx":
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Export failed: {exc}")
            return
        msg = (f"Saved {len(df)} rows from {len(self.state.session_records)} "
               f"file(s) to {Path(path).name}")

        # Line-scan analysis -> separate *_linescans.xlsx with labelled sheets.
        if do_linescan:
            if not linescans:
                notifications.show_warning(
                    "No two-spot ROIs found for line-scan analysis.")
            else:
                ls_path = str(Path(path).with_suffix("")) + "_linescans.xlsx"
                try:
                    sheets = scan.build_linescan_sheets(linescans)
                    with __import__("pandas").ExcelWriter(ls_path) as writer:
                        for name, sheet in sheets.items():
                            sheet.to_excel(writer, sheet_name=name, index=False)
                    msg += f"; {len(linescans)} line-scan(s) → {Path(ls_path).name}"
                except Exception as exc:  # noqa: BLE001
                    notifications.show_error(f"Line-scan export failed: {exc}")
        notifications.show_info(msg)
