"""Mom / bud (daughter) role assignment from a cell instance-label image.

Cells (yeast) are segmented into an integer label image — for example by
micro-sam's automatic instance segmentation.  Each detected spot falls inside
one cell.  For a two-spot ROI we label the pair:

* dots in two different cells  → the bigger cell's dot is ``mom``, the smaller
  cell's dot is ``daughter``;
* both dots in the mother cell → find the daughter (the other cell in the ROI,
  or one the user picks).  With a daughter known, the dot at least ``gap_um``
  microns closer to it is ``toward_bud`` and the other is plain ``mom``; if
  neither is clearly closer, or no daughter is known, both dots are ``mom``.

The mother/daughter *cell* roles drive the on-screen colouring; the per-*dot*
roles above are what a spot carries into analysis.

Cell shapes are also stored and restored as plain polygon outlines
(:func:`cell_outline`, :func:`labels_from_outlines`) so a session can carry its
mom/bud assignment without keeping a full segmentation label image, and
:func:`roles_for_spots` re-derives the roles from spot positions after a
re-run.

Pure module (numpy / scipy / skimage only) — unit-testable without the GUI or
micro-sam.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# A cell must have at least this many pixels inside an ROI before it can be
# called that ROI's mother or bud.  Guards against a sliver of a neighbouring
# cell clipping the box and being mistaken for the bud.
DEFAULT_MIN_OVERLAP_PX = 50

MOM = "mom"
DAUGHTER = "bud"                      # role value shown to the user is "bud"
TOWARD_DAUGHTER = "toward_bud"        # in the mother, clearly nearer the bud
# The dot that is NOT nearer the bud is plain ``mom``, the same value a dot in
# a mother cell gets when the pair is split across two cells.
MOM_TOWARD_DAUGHTER = TOWARD_DAUGHTER   # backwards-compatible aliases
MOM_TOWARD_MOM = MOM

# How an ROI's two dots sit relative to the cells.
PAIR_SPLIT = "split"                  # one dot in the mother, one in the bud
PAIR_BOTH_IN_MOM = "both_in_mom"      # both dots inside the mother


def cell_at(labels: Optional[np.ndarray], r: float, c: float) -> int:
    """Cell label at pixel ``(r, c)``, or 0 (background / out of range)."""
    if labels is None:
        return 0
    ri, ci = int(round(r)), int(round(c))
    if 0 <= ri < labels.shape[0] and 0 <= ci < labels.shape[1]:
        return int(labels[ri, ci])
    return 0


def cell_areas(labels: Optional[np.ndarray]) -> Dict[int, int]:
    """Pixel area of every non-zero cell label."""
    if labels is None:
        return {}
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v): int(n) for v, n in zip(vals, counts) if v != 0}


def other_cell_in_roi(labels: Optional[np.ndarray], roi_mask: Optional[np.ndarray],
                      exclude: int, areas: Optional[Dict[int, int]] = None,
                      min_overlap: int = 0) -> Optional[int]:
    """The bud in ``roi_mask``: the cell other than ``exclude`` with the most
    pixels inside the ROI.

    Used to auto-pick the bud when both dots are in the mother.  Only cells
    with at least ``min_overlap`` pixels inside are considered, and they are
    ranked by that overlap rather than by whole-cell area -- otherwise a large
    neighbour clipping the corner of the box would beat the real bud.  Returns
    ``None`` when the ROI holds no qualifying second cell.
    """
    if labels is None or roi_mask is None:
        return None
    for cell in cells_touching_roi(labels, roi_mask, min_overlap):
        if cell != exclude:
            return cell                    # already ordered by overlap
    return None


def pair_by_size(labels: Optional[np.ndarray], roi_mask: Optional[np.ndarray],
                 areas: Optional[Dict[int, int]] = None,
                 min_overlap: int = 0
                 ) -> Tuple[Optional[int], Optional[int]]:
    """The ROI's mother and bud from segmentation alone, as ``(mom, bud)``.

    Two separate questions, answered by two different measures:

    * **Which cells are in this ROI?**  The two with the most pixels *inside*
      ``roi_mask`` (and at least ``min_overlap`` of them).  Ranking candidates
      by whole-cell area instead would let a large neighbour that merely clips
      the corner of the box beat the real bud sitting squarely inside it.
    * **Which of those two is the mother?**  The one with the larger total
      area, since that is what actually distinguishes a mother from its bud.

    An ROI only counts when **both** are found.  With one qualifying cell (or
    none) this returns ``(None, None)`` rather than naming a lone mother --
    a region without a mother-and-bud pair is not a dividing cell to score.
    """
    if labels is None or roi_mask is None:
        return (None, None)
    if areas is None:
        areas = cell_areas(labels)
    present = cells_touching_roi(labels, roi_mask, min_overlap)
    if len(present) < 2:
        return (None, None)
    a, b = present[0], present[1]          # the two best-overlapping cells
    return (a, b) if areas.get(a, 0) >= areas.get(b, 0) else (b, a)


def _dist_to_cell(labels: np.ndarray, cell: int,
                  points: Sequence[Sequence[float]]) -> List[float]:
    """Euclidean distance (px) from each point to the nearest pixel of *cell*."""
    from scipy.ndimage import distance_transform_edt
    mask = labels == cell
    if not mask.any():
        return [float("inf")] * len(points)
    dt = distance_transform_edt(~mask)
    out = []
    for r, c in points:
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < dt.shape[0] and 0 <= ci < dt.shape[1]:
            out.append(float(dt[ri, ci]))
        else:
            out.append(float("inf"))
    return out


def assign_pair(labels: Optional[np.ndarray],
                spots: Sequence[Sequence[float]],
                roi_mask: Optional[np.ndarray] = None,
                daughter_cell: Optional[int] = None,
                mom_cell: Optional[int] = None,
                pixel_size: float = 1.0,
                gap_um: float = 0.3,
                areas: Optional[Dict[int, int]] = None,
                min_overlap: int = 0) -> dict:
    """Assign mom/daughter roles to a two-spot ROI's dots.

    ``spots`` is ``[(row, col), (row, col)]``.  ``roi_mask`` (bool, image-sized)
    lets the daughter be auto-picked as the other cell in the ROI; pass
    ``daughter_cell`` to force it.  ``gap_um`` is the "clearly closer" distance
    threshold in microns.

    An ROI is scored only when a **mother and a bud** are both present: each
    must have at least ``min_overlap`` pixels inside ``roi_mask``.  With only
    one qualifying cell nothing is assigned -- the dots come back with no role
    and a reason, rather than being labelled from a lone cell.

    Returns::

        {"spot_roles": [role, role],     # aligned with `spots`
         "mom_cell": int|None,
         "daughter_cell": int|None,
         "distances_um": [float, float]|None,   # each dot to the bud
         "pair_type": "split"|"both_in_mom"|"",
         "needs_daughter": bool,         # both in mom but no daughter found
         "ok": bool,
         "reason": str}
    """
    if areas is None:
        areas = cell_areas(labels)
    cells = [cell_at(labels, r, c) for r, c in spots]
    out = {"spot_roles": ["", ""], "mom_cell": None, "daughter_cell": None,
           "distances_um": None, "pair_type": "",
           "needs_daughter": False, "ok": False, "reason": ""}
    if len(spots) != 2:
        out["reason"] = "ROI does not have exactly two spots"
        return out
    a, b = cells
    if a == 0 or b == 0:
        out["reason"] = "a spot is not inside any segmented cell"
        return out

    # Both cells of the pair must really sit in the ROI, not just clip it.
    qualifying = (set(cells_touching_roi(labels, roi_mask, min_overlap))
                  if (roi_mask is not None and min_overlap > 0) else None)

    if a != b:                                    # dots in two cells
        if qualifying is not None and not {a, b} <= qualifying:
            out["reason"] = (
                f"a cell has under {min_overlap} px inside the ROI")
            return out
        if mom_cell in (a, b):                     # user forced the mother
            mom = mom_cell
            dau = b if mom == a else a
        elif daughter_cell in (a, b):              # user forced the daughter
            dau = daughter_cell
            mom = b if dau == a else a
        else:                                      # size guess: bigger = mother
            mom, dau = (a, b) if areas.get(a, 0) >= areas.get(b, 0) else (b, a)
        out["mom_cell"], out["daughter_cell"] = mom, dau
        out["spot_roles"] = [MOM if c == mom else DAUGHTER for c in cells]
        out["pair_type"] = PAIR_SPLIT
        out["distances_um"] = [x * float(pixel_size)
                               for x in _dist_to_cell(labels, dau, spots)]
        out["ok"] = True
        return out

    # both dots in the same (mother) cell
    mom = a
    out["mom_cell"] = mom
    out["pair_type"] = PAIR_BOTH_IN_MOM
    if qualifying is not None and mom not in qualifying:
        out["mom_cell"] = None
        out["pair_type"] = ""
        out["reason"] = f"the mother has under {min_overlap} px inside the ROI"
        return out
    dau = daughter_cell
    if dau is None:
        dau = other_cell_in_roi(labels, roi_mask, exclude=mom, areas=areas,
                                min_overlap=min_overlap)
    if dau is None or dau == mom:
        # No qualifying bud, so this is not a mother-and-bud pair and is not
        # scored.  needs_daughter tells the GUI it can be fixed with "Pick bud".
        out["needs_daughter"] = True
        out["reason"] = "no bud overlapping this ROI; nothing assigned"
        return out

    out["daughter_cell"] = dau
    d = [x * float(pixel_size) for x in _dist_to_cell(labels, dau, spots)]
    out["distances_um"] = d
    if abs(d[0] - d[1]) >= float(gap_um):
        near = 0 if d[0] < d[1] else 1
        roles = [MOM, MOM]
        roles[near] = TOWARD_DAUGHTER
        out["spot_roles"] = roles
    else:
        # The two dots are not separated enough to call one of them nearer.
        out["spot_roles"] = [MOM, MOM]
        out["reason"] = "both dots in the mother; neither clearly nearer the bud"
    out["ok"] = True
    return out


def role_masks(labels: Optional[np.ndarray],
               cell_roles: Dict[int, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean ``(mom_mask, daughter_mask)`` over the label image for display."""
    if labels is None:
        empty = np.zeros((1, 1), dtype=bool)
        return empty, empty
    mom_ids = [c for c, role in cell_roles.items() if role == MOM]
    dau_ids = [c for c, role in cell_roles.items() if role == DAUGHTER]
    mom = np.isin(labels, mom_ids) if mom_ids else np.zeros(labels.shape, bool)
    dau = np.isin(labels, dau_ids) if dau_ids else np.zeros(labels.shape, bool)
    return mom, dau


# --------------------------------------------------------------------------- #
# Storing cell shapes: outlines instead of a whole segmentation
# --------------------------------------------------------------------------- #
def cell_outline(labels: Optional[np.ndarray], cell: int,
                 tolerance: float = 1.0) -> Optional[np.ndarray]:
    """Simplified ``(N, 2)`` outline of one cell, in ``(row, col)`` pixels.

    A micro-sam label image is far too big to keep for every file, but role
    assignment only needs each cell's *shape*.  An outline is a few dozen
    numbers, small enough to hold in memory for a whole session and to write
    into the exported session sheets.  ``tolerance`` is the Douglas-Peucker
    simplification in pixels; 0 keeps every boundary point.
    """
    if labels is None:
        return None
    mask = np.asarray(labels) == cell
    if not mask.any():
        return None
    from skimage.measure import approximate_polygon, find_contours
    # Pad so a cell touching the image edge still yields a closed contour.
    contours = find_contours(np.pad(mask, 1).astype(float), 0.5)
    if not contours:
        return None
    verts = max(contours, key=len) - 1.0
    if tolerance:
        verts = approximate_polygon(verts, tolerance)
    if len(verts) > 3 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    return verts if len(verts) >= 3 else None


def overlap_counts(labels: Optional[np.ndarray],
                   roi_mask: Optional[np.ndarray]) -> Dict[int, int]:
    """``{cell: number of that cell's pixels inside roi_mask}``."""
    if labels is None or roi_mask is None:
        return {}
    inside = np.asarray(labels)[np.asarray(roi_mask, dtype=bool)]
    vals, counts = np.unique(inside, return_counts=True)
    return {int(v): int(n) for v, n in zip(vals, counts) if v != 0}


def cells_touching_roi(labels: Optional[np.ndarray],
                       roi_mask: Optional[np.ndarray],
                       min_overlap: int = 0) -> List[int]:
    """Cells with at least ``min_overlap`` pixels inside ``roi_mask``.

    Measured in absolute pixels, not as a fraction: an ROI is typically drawn
    around the spot pair and is smaller than a whole cell, so most of a real
    mother or bud lies outside the box.  A percentage would penalise exactly
    the large cells that matter.  Ordered by decreasing overlap.
    """
    counts = overlap_counts(labels, roi_mask)
    kept = [c for c, n in counts.items() if n >= max(int(min_overlap), 0)]
    return sorted(kept, key=lambda c: -counts[c])


def outlines_for_roi(labels: Optional[np.ndarray],
                     roi_mask: Optional[np.ndarray],
                     cell_roles: Optional[Dict[int, str]] = None,
                     mom_cell: Optional[int] = None,
                     daughter_cell: Optional[int] = None,
                     tolerance: float = 1.0,
                     min_overlap: int = 0) -> List[dict]:
    """Outline record for the ROI's tagged mother and bud -- at most two.

    Returns ``[{"cell": int, "role": str, "area": int, "verts": ndarray}, ...]``
    with the mother first, the bud second, and nothing else: an ROI is a
    dividing cell, so only that pair carries meaning.  A cell overlapping the
    ROI without a mom/bud tag is deliberately dropped, which keeps the stored
    record and the exported ``cells`` sheet to two rows per region.

    ``mom_cell`` / ``daughter_cell`` name the pair explicitly; otherwise the
    tags in ``cell_roles`` are used, restricted to cells with at least
    ``min_overlap`` pixels inside this ROI (``cell_roles`` is image-wide, so it
    also holds other ROIs' pairs).  If several qualifying cells carry the same
    tag, the one with the most overlap wins.
    """
    roles = dict(cell_roles or {})
    areas = cell_areas(labels)
    ordered = cells_touching_roi(labels, roi_mask, min_overlap)  # by overlap
    touching = set(ordered)

    def _pick(role_name, explicit):
        if explicit is not None and int(explicit) != 0:
            return int(explicit)
        return next((c for c in ordered if roles.get(c) == role_name), None)

    mom = _pick(MOM, mom_cell)
    bud = _pick(DAUGHTER, daughter_cell)
    if bud is not None and bud == mom:
        bud = None

    out = []
    for cell, role in ((mom, MOM), (bud, DAUGHTER)):
        if cell is None:
            continue
        verts = cell_outline(labels, cell, tolerance)
        if verts is None:
            continue
        out.append({"cell": int(cell), "role": role,
                    "area": int(areas.get(cell, 0)), "verts": verts})
    return out


def labels_from_outlines(records, shape):
    """Rebuild a label image from stored outline records.

    ``records`` is the list :func:`outlines_for_roi` produces.  Cells are drawn
    mother first, then bud, then the rest, each only where nothing has been
    drawn yet, so overlapping hand-drawn outlines never eat into the mother.
    Returns ``(labels, {label: role})``; the label value is the record's
    position + 1.
    """
    from skimage.draw import polygon2mask
    labels = np.zeros(shape, dtype=np.int32)
    roles = {}
    for idx, rec in enumerate(records or [], start=1):
        verts = rec.get("verts")
        if verts is None:
            continue
        m = polygon2mask(shape, np.asarray(verts, dtype=float))
        labels[m & (labels == 0)] = idx
        if rec.get("role"):
            roles[idx] = rec["role"]
    return labels, roles


def roles_for_spots(records, spots: Sequence[Sequence[float]],
                    pixel_size: float = 1.0, gap_um: float = 0.3) -> dict:
    """Roles for one ROI's dots, from stored outlines rather than a labelling.

    Rasterises the ROI's cells into a window just big enough to hold them and
    the dots, then runs :func:`assign_pair` on that.  This is what lets roles
    be recomputed after the export re-run: a role follows the dot's *position*,
    so it stays correct even when re-detection changes which spots are found or
    how they are numbered.  Only the ROI's tagged mother and bud are stored, so
    a dot that sits in neither gets no role.
    """
    blank = {"spot_roles": ["" for _ in spots], "distances_um": None,
             "pair_type": "", "mom_cell": None, "daughter_cell": None,
             "needs_daughter": False, "ok": False, "reason": ""}
    if len(spots) != 2:
        blank["reason"] = "ROI does not have exactly two spots"
        return blank
    shapes = [np.asarray(r["verts"], dtype=float) for r in (records or [])
              if r.get("verts") is not None]
    if not shapes:
        blank["reason"] = "no segmented cells stored for this ROI"
        return blank

    pts = np.asarray([[float(r), float(c)] for r, c in spots])
    coords = np.vstack(shapes + [pts])
    off = np.floor(coords.min(axis=0)).astype(int) - 2
    span = np.ceil(coords.max(axis=0)).astype(int) + 3 - off
    window = (int(span[0]), int(span[1]))

    shifted = [{**r, "verts": (None if r.get("verts") is None
                               else np.asarray(r["verts"], dtype=float) - off)}
               for r in (records or [])]
    labels, roles = labels_from_outlines(shifted, window)
    mom_id = next((i for i, r in roles.items() if r == MOM), None)
    bud_id = next((i for i, r in roles.items() if r == DAUGHTER), None)
    if mom_id is None or bud_id is None:
        blank["reason"] = "no mother-and-bud pair stored for this ROI"
        return blank
    local = [(p[0] - off[0], p[1] - off[1]) for p in pts]
    res = assign_pair(labels, local, roi_mask=(labels > 0),
                      daughter_cell=bud_id, mom_cell=mom_id,
                      pixel_size=pixel_size, gap_um=gap_um)
    if not res["ok"] and not res["reason"]:
        res["reason"] = "a dot is not inside any segmented cell"
    return res

# --------------------------------------------------------------------------- #
# Applying roles to a measurement table
# --------------------------------------------------------------------------- #
ROLE_COLUMNS = ("spot_role", "dist_to_bud_um", "pair_type", "role_note")


def apply_roles(table, session_cells: Optional[dict] = None,
                ref_by_file: Optional[dict] = None,
                pixel_by_file: Optional[dict] = None,
                gap_um: float = 0.3):
    """Add the mom/bud columns to a measurement table.

    Roles are worked out per ``(filename, region)`` from the reference
    channel's two spots and the cell outlines stored for that ROI, then copied
    onto every channel's row for the same ``spot_id``.  Because the lookup is
    by position, this is safe to run after the export re-run, which may find a
    different set of spots and renumber them.

    Adds ``spot_role`` (``mom`` / ``bud`` / ``toward_bud``, blank when it
    can't be decided), ``dist_to_bud_um``, ``pair_type`` (``split`` /
    ``both_in_mom``) and ``role_note`` explaining any blank.
    """
    import pandas as pd

    if table is None or getattr(table, "empty", True):
        out = table if table is not None else pd.DataFrame()
        for col in ROLE_COLUMNS:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    out = table.copy()
    for col in ROLE_COLUMNS:
        out[col] = "" if col != "dist_to_bud_um" else np.nan
    if "region" not in out.columns or "spot_id" not in out.columns:
        return out

    has_file = "filename" in out.columns
    keys = (["filename", "region"] if has_file else ["region"])
    for key, group in out.groupby(keys, sort=False):
        fname = (key[0] if has_file else "") if isinstance(key, tuple) else (
            key if has_file else "")
        region = str(key[-1] if isinstance(key, tuple) else key)
        ref = (ref_by_file or {}).get(fname)
        if "channel" in group.columns and ref is not None:
            ref_rows = group[group["channel"] == ref]
        elif "channel" in group.columns:
            ref_rows = group[group["channel"] == group["channel"].iloc[0]]
        else:
            ref_rows = group
        # A manually added spot carries n_planes_linked == 0 but is still a dot.
        ref_rows = ref_rows.drop_duplicates(subset="spot_id")
        records = (session_cells or {}).get(fname, {}).get(region)
        spots = [(float(r["row_px"]), float(r["col_px"]))
                 for _, r in ref_rows.iterrows()]
        res = roles_for_spots(records, spots,
                              pixel_size=float((pixel_by_file or {}).get(
                                  fname, 1.0)),
                              gap_um=gap_um)
        note = res.get("reason", "")
        dists = res.get("distances_um")
        for i, (_, r) in enumerate(ref_rows.iterrows()):
            sid = r["spot_id"]
            rows = group.index[group["spot_id"] == sid]
            role = res["spot_roles"][i] if i < len(res["spot_roles"]) else ""
            out.loc[rows, "spot_role"] = role
            out.loc[rows, "pair_type"] = res.get("pair_type", "")
            out.loc[rows, "role_note"] = note
            if dists is not None and i < len(dists):
                out.loc[rows, "dist_to_bud_um"] = float(dists[i])
    return out

def labels_from_session(per_roi: Optional[dict], shape):
    """Rebuild one file's cell label image from its stored ROI records.

    ``per_roi`` maps ROI label -> the outline records kept for that ROI.  The
    stored ``cell`` ids are reused as the label values, so a restored session
    refers to the same cells as the segmentation it was saved from and manual
    picks still line up.  Mothers are drawn before buds and each cell only
    where nothing has been drawn yet, so overlapping outlines never eat into
    the mother.  Returns ``(labels, {cell: role})``.
    """
    from skimage.draw import polygon2mask
    labels = np.zeros(shape, dtype=np.int32)
    roles: Dict[int, str] = {}
    records = [rec for recs in (per_roi or {}).values()
               for rec in (recs or []) if rec.get("verts") is not None]
    records.sort(key=lambda r: 0 if r.get("role") == MOM else 1)
    drawn = set()
    for rec in records:
        cell = int(rec.get("cell", 0))
        if cell <= 0 or cell in drawn:
            continue
        drawn.add(cell)
        m = polygon2mask(shape, np.asarray(rec["verts"], dtype=float))
        labels[m & (labels == 0)] = cell
        if rec.get("role"):
            roles[cell] = rec["role"]
    return labels, roles
