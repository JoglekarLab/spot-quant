"""Measurement tab: compiled session table, summary and export.

Records from every analysed file accumulate into one long-format table (one
row per spot per channel, tagged with ``filename`` and ``region``), so a
multi-file / multi-ROI workflow exports to a single CSV/XLSX that is trivial
to re-import for downstream analysis.
"""
from __future__ import annotations

from pathlib import Path

import napari.utils.notifications as notifications
import pandas as pd
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from . import pipeline, scan, session_io
from .state import AppState


class MeasurementPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._export_cancelled = False

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
        self.export_xlsx_btn = QPushButton("Export XLSX…")
        self.clear_btn = QPushButton("Clear session")
        self.export_xlsx_btn.clicked.connect(lambda: self._export("xlsx"))
        self.clear_btn.clicked.connect(self.state.clear_session)
        btn_row.addWidget(self.export_xlsx_btn)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.linescan_check = QCheckBox(
            "Also export line-scans (two-spot ROIs) → *_linescans.xlsx")
        self.linescan_check.setChecked(True)
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

    def _export_global_result(self, with_linescans):
        """Export the last global re-detect as it stands, without re-running.

        If something changed after it (new/moved/deleted ROIs, settings, a
        per-file Detect), warn and let the user export anyway or cancel.
        """
        stale = list(self.state.global_stale)
        if stale:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Results may be out of date")
            box.setText("These changed after the last "
                        "'Re-detect all files (global)':")
            box.setInformativeText(
                "\n".join(f"• {s}" for s in stale)
                + "\n\nExport will use the results as they are now, without "
                "re-running. To update them first, click Cancel and run "
                "'Re-detect all files (global)' again.")
            export_btn = box.addButton("Export anyway", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() is not export_btn:
                self._export_cancelled = True
                return None, None
        try:
            df, linescans = self.state.table_without_rerun(
                with_linescans=with_linescans)
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Export failed: {exc}")
            self._export_cancelled = True
            return None, None
        notifications.show_info(
            "Using the results from 'Re-detect all files (global)' "
            "(no re-run).")
        return df, linescans

    def _recompute_or_fallback(self, with_linescans):
        """Re-run the session with a global threshold; fall back to the
        incrementally recorded table if a re-run isn't possible.

        If "Re-detect all files (global)" was already run, its result is
        exported as is (see :meth:`_export_global_result`).

        Returns ``(dataframe, linescans_or_None)``.
        """
        if self.state.global_done:
            return self._export_global_result(with_linescans)
        n_rois = sum(len(b) for b in self.state.session_rois.values())
        if n_rois:
            n_files = len([b for b in self.state.session_rois.values() if b])
            reply = QMessageBox.question(
                self, "Re-analyze session",
                f"Re-run detection on {n_rois} ROI(s) across {n_files} file(s) "
                "using one global threshold pooled from all ROIs, then export?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                self._export_cancelled = True
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
        self._export_cancelled = False
        df, linescans = self._recompute_or_fallback(with_linescans=do_linescan)
        if self._export_cancelled:
            return
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
        # Resume tables: exact ROIs (incl. empty ones + polygons) and settings,
        # so the session can be re-imported to add more ROIs later.
        roi_df = session_io.rois_dataframe(self.state.session_shapes,
                                           self.state.session_ref)
        settings_df = session_io.settings_dataframe(
            self.state.params, getattr(self.state, "roi_counter", 0))
        cells_df = session_io.cells_dataframe(
            getattr(self.state, "session_cells", {}))
        edits_df = session_io.spot_edits_dataframe(
            self.state.manual_spots, self.state.removed_spots)
        companion = str(Path(path).with_suffix("")) + "_session.xlsx"
        try:
            if fmt == "xlsx":
                with pd.ExcelWriter(path) as writer:
                    df.to_excel(writer, sheet_name="measurements", index=False)
                    if not roi_df.empty:
                        roi_df.to_excel(writer, sheet_name=session_io.ROI_SHEET,
                                        index=False)
                        settings_df.to_excel(
                            writer, sheet_name=session_io.SETTINGS_SHEET,
                            index=False)
                    if not cells_df.empty:
                        cells_df.to_excel(
                            writer, sheet_name=session_io.CELLS_SHEET,
                            index=False)
                    if not edits_df.empty:
                        edits_df.to_excel(
                            writer, sheet_name=session_io.SPOT_EDITS_SHEET,
                            index=False)
            else:
                df.to_csv(path, index=False)
                if not roi_df.empty:
                    with pd.ExcelWriter(companion) as writer:
                        roi_df.to_excel(writer, sheet_name=session_io.ROI_SHEET,
                                        index=False)
                        settings_df.to_excel(
                            writer, sheet_name=session_io.SETTINGS_SHEET,
                            index=False)
                        if not cells_df.empty:
                            cells_df.to_excel(
                                writer, sheet_name=session_io.CELLS_SHEET,
                                index=False)
                        if not edits_df.empty:
                            edits_df.to_excel(
                                writer,
                                sheet_name=session_io.SPOT_EDITS_SHEET,
                                index=False)
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Export failed: {exc}")
            return
        msg = (f"Saved {len(df)} rows from {len(self.state.session_records)} "
               f"file(s) to {Path(path).name}")
        if not roi_df.empty:
            msg += ("; ROIs+settings sheets included (re-importable)"
                    if fmt == "xlsx"
                    else f"; ROIs → {Path(companion).name} (re-importable)")

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
