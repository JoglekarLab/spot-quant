"""Shared application state passed between the GUI panels."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Signal

from . import pipeline
from .metadata import ImageMeta, load_image


class AppState(QObject):
    """Holds the loaded image and metadata and notifies panels of changes.

    ``image`` is channel-first: ``(C, Y, X)`` or ``(C, Z, Y, X)``.
    """

    image_loaded = Signal()        # emitted after a new image is set
    metadata_changed = Signal()    # emitted after metadata is edited
    result_updated = Signal()      # emitted after the pipeline re-runs
    session_changed = Signal()     # emitted after the session table changes

    def __init__(self):
        super().__init__()
        self.viewer = None
        self.folder: Optional[Path] = None
        self.current_path: Optional[Path] = None
        self.image: Optional[np.ndarray] = None
        self.meta: ImageMeta = ImageMeta()
        # True while a new file is being loaded / layers swapped, so reactive
        # panels don't touch the viewer mid-clear (avoids a napari crash).
        self.loading: bool = False
        # Most recent pipeline result, shared with the Measurement tab.
        self.last_result = None
        # Accumulated measurement tables, nested filename -> region -> table.
        # Keying by region lets incremental ROI detection add new regions and
        # re-detection replace a region without duplicating rows.
        self.session_records: "dict[str, dict[str, pd.DataFrame]]" = {}
        # Number of ROIs drawn on the current image but not yet analyzed.
        self.pending_rois: int = 0
        # Per-file ROIs as labelled tuples (r0,r1,c0,c1,label) + reference
        # channel, so the whole session can be re-run at export.  ROI labels
        # are session-global (increment across files, never repeat).
        self.session_rois: "dict[str, list]" = {}
        self.session_ref: "dict[str, str]" = {}
        self.roi_counter: int = 0
        # Latest detection parameters (set by the detection panel).
        self.params = None
        # Reference channel name (persisted across image loads).
        self.ref_channel: Optional[str] = None
        # Chromatic-aberration offset (edited via the metadata dialog).
        self.offset_z: int = 0
        self.offset_y: int = 0
        self.offset_x: int = 0

    # -- per-file ROI persistence ------------------------------------------- #
    def assign_roi_labels(self, filename: str, bounds_list, ref_channel: str):
        """Store the file's ROIs with session-global labels and return them.

        Labels already assigned to a rectangle (matched by bounds) are kept;
        new rectangles get the next global ``ROIn``.  Returns the labelled list
        ``[(r0, r1, c0, c1, label), ...]``.
        """
        prior = {tuple(int(x) for x in r[:4]): r[4]
                 for r in self.session_rois.get(filename, [])}
        labelled, self.roi_counter = pipeline.next_roi_labels(
            prior, bounds_list, self.roi_counter)
        if labelled:
            self.session_rois[filename] = labelled
            self.session_ref[filename] = ref_channel
        else:
            self.session_rois.pop(filename, None)
            self.session_ref.pop(filename, None)
        return labelled

    def _build_files_data(self):
        """Reload every file that has saved ROIs into pipeline-ready dicts."""
        files_data = []
        if self.folder is None:
            return files_data
        for fname, bounds in self.session_rois.items():
            if not bounds:
                continue
            try:
                image, meta = load_image(Path(self.folder) / fname)
            except Exception:
                continue
            names = meta.channel_names or [f"Ch{i}" for i in range(image.shape[0])]
            stacks = {}
            for i, nm in enumerate(names):
                ch = image[i]
                stacks[nm] = ch if ch.ndim == 3 else ch[None]
            ref = self.session_ref.get(fname) or next(
                (n for n in names if pipeline.is_measurable_channel(n)), names[0])
            # session_rois already stores (r0,r1,c0,c1,label) with global labels.
            rois = [tuple(r) for r in bounds]
            files_data.append({
                "filename": fname, "stacks": stacks, "ref_channel": ref,
                "rois": rois, "pixel_size": meta.pixel_size,
                "pixel_unit": meta.pixel_unit})
        return files_data

    def recompute_session(self, with_linescans: bool = False):
        """Reload every file with saved ROIs and re-run the whole session with
        one global per-channel threshold pooled across all ROIs of all files.

        Returns ``(combined_table, channel_thresholds, linescans)``; the last
        is ``None`` unless *with_linescans* is set.
        """
        if self.params is None or self.folder is None or not self.session_rois:
            return None, {}, None
        files_data = self._build_files_data()
        if not files_data:
            return None, {}, None
        table, thresholds = pipeline.run_session(files_data, self.params)
        linescans = None
        if with_linescans:
            from . import scan
            linescans = scan.linescans_from_measurements(files_data, table)
        return table, thresholds, linescans

    # -- session accumulation ----------------------------------------------- #
    def record_measurements(self, filename: str, df: pd.DataFrame):
        """Add/replace the records for each region present in *df*."""
        tagged = pipeline.tag_filename(df, filename)
        per = self.session_records.setdefault(filename, {})
        if "region" in tagged.columns:
            for region, sub in tagged.groupby("region"):
                per[str(region)] = sub
        else:
            per["_all"] = tagged
        self.session_changed.emit()

    def remove_region(self, filename: str, region: str):
        """Drop a single ROI's records (e.g. when its ROI is deleted)."""
        per = self.session_records.get(filename)
        if not per or region not in per:
            return
        del per[region]
        if not per:
            del self.session_records[filename]
        self.session_changed.emit()

    def clear_session(self):
        self.session_records.clear()
        self.session_rois.clear()
        self.session_ref.clear()
        self.roi_counter = 0
        self.session_changed.emit()

    def session_table(self) -> pd.DataFrame:
        """All recorded files/regions concatenated into one long-format table."""
        frames = [sub for per in self.session_records.values()
                  for sub in per.values()]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # -- channel / plane helpers -------------------------------------------- #
    @property
    def n_channels(self) -> int:
        return 0 if self.image is None else self.image.shape[0]

    @property
    def is_stack(self) -> bool:
        return self.image is not None and self.image.ndim == 4

    def channel_plane(self, channel: int) -> np.ndarray:
        """Return the 2-D plane for *channel* at the viewer's current Z."""
        if self.image is None:
            raise RuntimeError("No image loaded.")
        ch = self.image[channel]
        if ch.ndim == 2:
            return ch
        z = 0
        if self.viewer is not None and self.viewer.dims.ndim >= 3:
            z = int(self.viewer.dims.current_step[0])
        z = max(0, min(z, ch.shape[0] - 1))
        return ch[z]

    def channel_stack(self, channel: int) -> np.ndarray:
        """Return the full (Z, Y, X) volume for *channel* (1 plane if 2-D)."""
        if self.image is None:
            raise RuntimeError("No image loaded.")
        ch = self.image[channel]
        return ch if ch.ndim == 3 else ch[None]

    def set_image(self, path: Path, image: np.ndarray, meta: ImageMeta):
        self.current_path = Path(path)
        self.image = image
        self.meta = meta
        self.image_loaded.emit()
