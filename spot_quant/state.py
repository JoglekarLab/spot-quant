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
    session_imported = Signal()    # emitted after ROIs/settings are imported

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
        # Labels of the ROIs that have been through detection, per file.  An
        # ROI with zero spots has no rows in session_records, so this is what
        # tells "analyzed, found nothing" apart from "drawn, not detected yet"
        # when a file is reopened and its saved spots are redrawn.
        self.analyzed_rois: "dict[str, set]" = {}
        # Manually placed spots, filename -> region label -> list of (row, col).
        # Measured full-disk and merged into detection at record/export time.
        self.manual_spots: "dict[str, dict[str, list]]" = {}
        # Detected spots the user removed, filename -> region -> list of
        # (row, col).  Every detection path (Detect, Re-detect, global, export)
        # drops a detected spot that lands within pipeline.removal_radius of one.
        self.removed_spots: "dict[str, dict[str, list]]" = {}
        # Result of the last "Re-detect all files (global)".  While it is still
        # current, Export uses the session table as is instead of re-running
        # the whole session.  global_stale lists what changed since (shown in
        # the export warning); an empty list means the result is current.
        self.global_done: bool = False
        self.global_thresholds: dict = {}
        self.global_ref: "dict[str, str]" = {}
        self.global_pixel: "dict[str, float]" = {}
        self.global_stale: "list[str]" = []
        # Number of ROIs drawn on the current image but not yet analyzed.
        self.pending_rois: int = 0
        # Per-file ROIs as labelled tuples (r0,r1,c0,c1,label) + reference
        # channel, so the whole session can be re-run at export.  ROI labels
        # are session-global (increment across files, never repeat).
        self.session_rois: "dict[str, list]" = {}
        # Full per-file ROI shape records: list of
        # {"bounds": (r0,r1,c0,c1), "verts": ndarray|None, "label": str}.
        # ``verts`` is None for rectangles and the polygon corners otherwise.
        self.session_shapes: "dict[str, list]" = {}
        # Per-file polygon corners keyed by label ({label: verts}); rectangles
        # are absent.  Used to rebuild masks for the export re-run.
        self.session_polys: "dict[str, dict]" = {}
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
        # micro-sam cell segmentation for the current file: 2-D instance-label
        # image, and the mom/bud role of each cell label ({label: "mom"/"bud"}).
        # Both are per-file and dropped when another file opens.
        self.cell_labels = None
        self.cell_roles: "dict[int, str]" = {}
        # Session-wide cell record: filename -> ROI label -> list of
        # {"cell", "role", "area", "verts"} for EVERY segmented cell touching
        # that ROI.  Outlines rather than label images, so the whole session
        # fits in memory and can be written to (and read back from) the export.
        # This is what lets mom/bud survive switching files and the export
        # re-run, which recomputes roles from these shapes.
        self.session_cells: "dict[str, dict[str, list]]" = {}

    # -- per-file ROI persistence ------------------------------------------- #
    def assign_roi_labels(self, filename: str, shapes, ref_channel: str):
        """Store the file's ROIs with session-global labels and return them.

        ``shapes`` is a list of records
        ``{"bounds": (r0,r1,c0,c1), "verts": ndarray|None, "key": hashable}``.
        Labels already assigned to a shape (matched by its ``key``, which
        includes polygon corners so two shapes with the same bounding box stay
        distinct) are kept; new shapes get the next global ``ROIn``.  Returns
        the labelled list ``[(r0, r1, c0, c1, label), ...]`` aligned with
        ``shapes``.
        """
        prior = {}
        for rec in self.session_shapes.get(filename, []):
            prior[pipeline.shape_key(rec["bounds"], rec.get("verts"))] = \
                rec["label"]
        labelled, records, new_keys = [], [], set()
        for s in shapes:
            key = s.get("key") or pipeline.shape_key(s["bounds"], s.get("verts"))
            new_keys.add(key)
            label = prior.get(key)
            if label is None:
                self.roi_counter += 1
                label = f"ROI{self.roi_counter}"
            r0, r1, c0, c1 = s["bounds"]
            labelled.append((r0, r1, c0, c1, label))
            records.append({"bounds": tuple(s["bounds"]),
                            "verts": s.get("verts"), "label": label})
        if filename and new_keys != set(prior):
            self.mark_global_stale(f"ROIs added, moved or deleted in {filename}")
        if records:
            self.session_rois[filename] = labelled
            self.session_shapes[filename] = records
            self.session_polys[filename] = {
                r["label"]: r["verts"] for r in records
                if r["verts"] is not None}
            self.session_ref[filename] = ref_channel
        else:
            self.session_rois.pop(filename, None)
            self.session_shapes.pop(filename, None)
            self.session_polys.pop(filename, None)
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
            # Rebuild polygon masks (keyed by label) from stored corners so the
            # export re-run respects each ROI's real shape.
            plane_shape = next(iter(stacks.values())).shape[1:]
            polys = self.session_polys.get(fname, {})
            masks = {label: pipeline.roi_mask(None, verts, plane_shape)
                     for label, verts in polys.items() if verts is not None}
            files_data.append({
                "filename": fname, "stacks": stacks, "ref_channel": ref,
                "rois": rois, "masks": masks or None,
                "manual_spots": self.manual_spots.get(fname) or None,
                "removed_spots": self.removed_spots.get(fname) or None,
                "pixel_size": meta.pixel_size,
                "pixel_unit": meta.pixel_unit,
                "z_step": meta.z_step})
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
        # Re-derive mom/bud from each dot's position: the re-run may have found
        # a different set of spots, so a role saved against a spot_id could
        # land on the wrong dot.
        from . import cells as cells_mod
        table = cells_mod.apply_roles(
            table, self.session_cells,
            ref_by_file={f["filename"]: f["ref_channel"] for f in files_data},
            pixel_by_file={f["filename"]: f.get("pixel_size", 1.0)
                           for f in files_data},
            gap_um=float(getattr(self.params, "toward_bud_gap_um", 0.3)))
        linescans = None
        if with_linescans:
            from . import scan
            linescans = scan.linescans_from_measurements(files_data, table)
        return table, thresholds, linescans

    # -- session accumulation ----------------------------------------------- #
    def roles_for_file(self, filename: str, df: pd.DataFrame,
                       ref_channel: str) -> pd.DataFrame:
        """Add the mom/bud columns to one file's freshly measured table."""
        from . import cells as cells_mod
        return cells_mod.apply_roles(
            df, self.session_cells, ref_by_file={filename: ref_channel},
            pixel_by_file={filename: self.meta.pixel_size},
            gap_um=float(getattr(self.params, "toward_bud_gap_um", 0.3)))

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

    # -- global re-detect bookkeeping -------------------------------------- #
    def set_global_result(self, thresholds: dict, ref_by_file: dict,
                          pixel_by_file: dict):
        """Remember that the session table now holds a global re-detect."""
        self.global_done = True
        self.global_thresholds = dict(thresholds or {})
        self.global_ref = dict(ref_by_file or {})
        self.global_pixel = dict(pixel_by_file or {})
        self.global_stale = []

    def mark_global_stale(self, reason: str):
        """Note that something changed after the last global re-detect."""
        if self.global_done and reason not in self.global_stale:
            self.global_stale.append(reason)

    def clear_global(self):
        self.global_done = False
        self.global_thresholds = {}
        self.global_ref = {}
        self.global_pixel = {}
        self.global_stale = []

    def table_without_rerun(self, with_linescans: bool = False):
        """The session table as it stands (from the last global re-detect),
        with mom/bud roles refreshed, for export without re-running detection.

        Returns ``(table, linescans)``; line scans need the images, so they are
        reloaded (no detection is re-run).
        """
        from . import cells as cells_mod
        table = self.session_table()
        if table.empty:
            return table, None
        ref_by_file = dict(self.global_ref)
        ref_by_file.update({k: v for k, v in self.session_ref.items()
                            if k not in ref_by_file})
        table = cells_mod.apply_roles(
            table, self.session_cells, ref_by_file=ref_by_file,
            pixel_by_file=self.global_pixel,
            gap_um=float(getattr(self.params, "toward_bud_gap_um", 0.3)))
        linescans = None
        if with_linescans:
            from . import scan
            files_data = self._build_files_data()
            linescans = scan.linescans_from_measurements(files_data, table)
        return table, linescans

    def mark_analyzed(self, filename: str, labels, replace: bool = False):
        """Record that these ROI labels have been through detection."""
        if replace:
            self.analyzed_rois[filename] = {str(l) for l in labels}
        else:
            self.analyzed_rois.setdefault(filename, set()).update(
                str(l) for l in labels)

    def remove_region(self, filename: str, region: str):
        """Drop a single ROI's records (e.g. when its ROI is deleted)."""
        self.analyzed_rois.get(filename, set()).discard(str(region))
        per = self.session_records.get(filename)
        if not per or region not in per:
            return
        del per[region]
        if not per:
            del self.session_records[filename]
        self.session_changed.emit()

    def import_session(self, data: dict):
        """Load ROIs (and optionally settings) from an imported session.

        ``data`` is the dict returned by ``session_io.session_from_sheets`` /
        ``session_from_measurements``.  Replaces the per-file ROI state so that
        opening a file shows its restored ROIs and new ROIs keep numbering.
        Prior measurements aren't restored (the table repopulates when you
        re-detect or export, which re-runs the whole session).
        """
        self.session_shapes = dict(data.get("session_shapes", {}))
        self.session_rois = dict(data.get("session_rois", {}))
        self.session_polys = dict(data.get("session_polys", {}))
        self.session_ref = dict(data.get("session_ref", {}))
        self.session_cells = dict(data.get("session_cells", {}))
        self.manual_spots = dict(data.get("manual_spots", {}))
        self.removed_spots = dict(data.get("removed_spots", {}))
        self.roi_counter = int(data.get("roi_counter", 0))
        # Measurements aren't imported, so there is no global result to reuse.
        self.clear_global()
        params = data.get("params")
        if params is not None:
            self.params = params
            self.offset_z = int(getattr(params, "offset_z", 0))
            self.offset_y = int(getattr(params, "offset_y", 0))
            self.offset_x = int(getattr(params, "offset_x", 0))
        self.session_imported.emit()
        self.session_changed.emit()

    def clear_session(self):
        self.session_records.clear()
        self.analyzed_rois.clear()
        self.manual_spots.clear()
        self.removed_spots.clear()
        self.clear_global()
        self.session_rois.clear()
        self.session_shapes.clear()
        self.session_polys.clear()
        self.session_ref.clear()
        self.session_cells.clear()
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
        # Cell segmentation is per-file; drop it when a new file opens.
        self.cell_labels = None
        self.cell_roles = {}
        self.image_loaded.emit()
