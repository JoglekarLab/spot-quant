"""Serialise / restore a session so ROIs (and settings) can be re-imported.

The measurements export is a *spot* table — one row per detected spot — so it
cannot represent an ROI that had no spots, a polygon's corners, or the
detection settings.  To let a user reopen a session and keep adding ROIs, the
export also carries two small tables:

* an **ROIs** sheet — one row per ROI (rectangle or polygon), including empty
  ones, with its bounding box, polygon corners and reference channel;
* a **settings** sheet — the detection parameters and the ROI-label counter.

``session_from_sheets`` rebuilds the session exactly from those two tables.
``session_from_measurements`` is the lossy fallback: it rebuilds an
*approximate* rectangle around each ROI's spots from a plain measurements table
(no empty ROIs, no polygons, no settings).

This module is pure (numpy / pandas / json only) so it is unit-testable without
the GUI stack.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

ROI_SHEET = "ROIs"
SETTINGS_SHEET = "settings"

# Detection-parameter fields persisted to the settings sheet.
_PARAM_FLOAT = ("smoothing_size", "tophat_size", "peak_rel_threshold",
                "link_max_dist", "ilastik_prob_threshold")
_PARAM_INT = ("min_mask_size", "min_distance", "measure_radius", "bkg_gap",
              "bkg_width", "min_link_planes", "offset_z", "offset_y", "offset_x")
_PARAM_STR = ("smoothing_method", "threshold_method")


def _roi_number(label) -> int:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return int(digits) if digits else 0


# --------------------------------------------------------------------------- #
# Serialise
# --------------------------------------------------------------------------- #
def rois_dataframe(session_shapes: dict, session_ref: dict) -> pd.DataFrame:
    """One row per ROI (including empty ones) from the per-file shape records."""
    cols = ["filename", "region", "shape_type", "r0", "r1", "c0", "c1",
            "verts", "ref_channel"]
    rows = []
    for fname, records in (session_shapes or {}).items():
        ref = (session_ref or {}).get(fname, "")
        for rec in records:
            verts = rec.get("verts")
            r0, r1, c0, c1 = rec["bounds"]
            rows.append({
                "filename": fname,
                "region": rec["label"],
                "shape_type": "polygon" if verts is not None else "rectangle",
                "r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1),
                "verts": (json.dumps(np.asarray(verts).tolist())
                          if verts is not None else ""),
                "ref_channel": ref or "",
            })
    return pd.DataFrame(rows, columns=cols)


def settings_dataframe(params, roi_counter: int) -> pd.DataFrame:
    """Key/value table of detection parameters + the ROI-label counter."""
    rows = []
    fields = _PARAM_FLOAT + _PARAM_INT + _PARAM_STR + ("ilastik_model_path",)
    for f in fields:
        val = getattr(params, f, None) if params is not None else None
        if f == "ilastik_model_path" and val is not None:
            val = str(val)
        rows.append({"key": f, "value": "" if val is None else val})
    rows.append({"key": "roi_counter", "value": int(roi_counter)})
    return pd.DataFrame(rows, columns=["key", "value"])


# --------------------------------------------------------------------------- #
# Restore (exact) from the ROIs + settings sheets
# --------------------------------------------------------------------------- #
def session_from_sheets(roi_df: Optional[pd.DataFrame],
                        settings_df: Optional[pd.DataFrame],
                        params_factory) -> dict:
    """Rebuild the session dicts + params from the two sheets.

    ``params_factory`` builds a params object from keyword arguments (pass
    ``pipeline.PipelineParams``).  Returns a dict with ``session_shapes``,
    ``session_rois``, ``session_polys``, ``session_ref``, ``roi_counter`` and
    ``params`` (``approx`` is ``False``).
    """
    session_shapes, session_rois, session_polys, session_ref = {}, {}, {}, {}
    if roi_df is not None and not roi_df.empty:
        for fname, grp in roi_df.groupby("filename"):
            fname = str(fname)
            recs, tuples, polys = [], [], {}
            for _, row in grp.iterrows():
                label = str(row["region"])
                bounds = (int(row["r0"]), int(row["r1"]),
                          int(row["c0"]), int(row["c1"]))
                verts = None
                vraw = row.get("verts", "")
                if (str(row.get("shape_type", "")) == "polygon"
                        and isinstance(vraw, str) and vraw.strip()):
                    try:
                        verts = np.asarray(json.loads(vraw), dtype=float)
                    except Exception:  # noqa: BLE001
                        verts = None
                recs.append({"bounds": bounds, "verts": verts, "label": label})
                tuples.append((*bounds, label))
                if verts is not None:
                    polys[label] = verts
            session_shapes[fname] = recs
            session_rois[fname] = tuples
            if polys:
                session_polys[fname] = polys
            ref = grp["ref_channel"].iloc[0] if "ref_channel" in grp else ""
            if isinstance(ref, str) and ref:
                session_ref[fname] = ref

    params, roi_counter = None, 0
    if settings_df is not None and not settings_df.empty:
        kv = {str(k): v for k, v in zip(settings_df["key"], settings_df["value"])}
        roi_counter = _as_int(kv.get("roi_counter", 0), 0)
        params = _build_params(kv, params_factory)

    # roi_counter must be at least the highest label seen.
    for recs in session_shapes.values():
        for rec in recs:
            roi_counter = max(roi_counter, _roi_number(rec["label"]))

    return {"session_shapes": session_shapes, "session_rois": session_rois,
            "session_polys": session_polys, "session_ref": session_ref,
            "roi_counter": roi_counter, "params": params, "approx": False}


def _as_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(round(float(x)))
    except Exception:  # noqa: BLE001
        return default


def _as_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:  # noqa: BLE001
        return default


def _build_params(kv: dict, params_factory):
    kwargs = {}
    for name in _PARAM_FLOAT:
        v = _as_float(kv.get(name))
        if v is not None:
            kwargs[name] = v
    for name in _PARAM_INT:
        v = _as_int(kv.get(name))
        if v is not None:
            kwargs[name] = v
    for name in _PARAM_STR:
        v = kv.get(name)
        if isinstance(v, str) and v:
            kwargs[name] = v
    params = params_factory(**kwargs)
    mp = kv.get("ilastik_model_path")
    if isinstance(mp, str) and mp:
        from pathlib import Path
        params.ilastik_model_path = Path(mp)
    return params


# --------------------------------------------------------------------------- #
# Restore (lossy) from a measurements table
# --------------------------------------------------------------------------- #
def _find_pos_cols(df: pd.DataFrame):
    rcols = [c for c in df.columns if str(c).endswith("_row_px")]
    ccols = [c for c in df.columns if str(c).endswith("_col_px")]
    if rcols and ccols:
        return rcols[0], ccols[0]
    if "row_px" in df.columns and "col_px" in df.columns:
        return "row_px", "col_px"
    return None


def session_from_measurements(df: pd.DataFrame, pad: int = 12) -> dict:
    """Lossy fallback: an approximate rectangle around each ROI's spots.

    Uses the first channel's ``*_row_px`` / ``*_col_px`` columns as spot
    positions and pads the bounding box by ``pad`` px (so re-detected spots
    don't sit on the ROI edge).  Empty ROIs, polygons and detection settings
    cannot be recovered.  Same return shape as :func:`session_from_sheets` with
    ``params=None`` and ``approx=True``.
    """
    empty = {"session_shapes": {}, "session_rois": {}, "session_polys": {},
             "session_ref": {}, "roi_counter": 0, "params": None, "approx": True}
    if df is None or df.empty:
        return empty
    pos = _find_pos_cols(df)
    if pos is None or "filename" not in df.columns or "region" not in df.columns:
        return empty
    rcol, ccol = pos
    session_shapes, session_rois = {}, {}
    roi_counter = 0
    for fname, fgrp in df.groupby("filename"):
        recs, tuples = [], []
        for region, grp in fgrp.groupby("region"):
            rs = grp[rcol].astype(float)
            cs = grp[ccol].astype(float)
            r0 = max(0, int(np.floor(rs.min())) - pad)
            r1 = int(np.ceil(rs.max())) + pad
            c0 = max(0, int(np.floor(cs.min())) - pad)
            c1 = int(np.ceil(cs.max())) + pad
            label = str(region)
            recs.append({"bounds": (r0, r1, c0, c1), "verts": None,
                         "label": label})
            tuples.append((r0, r1, c0, c1, label))
            roi_counter = max(roi_counter, _roi_number(label))
        session_shapes[str(fname)] = recs
        session_rois[str(fname)] = tuples
    return {"session_shapes": session_shapes, "session_rois": session_rois,
            "session_polys": {}, "session_ref": {}, "roi_counter": roi_counter,
            "params": None, "approx": True}
