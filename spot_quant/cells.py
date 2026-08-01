"""Mom / bud (daughter) role assignment from a cell instance-label image.

Cells (yeast) are segmented into an integer label image — for example by
micro-sam's automatic instance segmentation.  Each detected spot falls inside
one cell.  For a two-spot ROI we label the pair:

* dots in two different cells  → the bigger cell's dot is ``mom``, the smaller
  cell's dot is ``daughter``;
* both dots in the mother cell → find the daughter (the other cell in the ROI,
  or one the user picks).  With a daughter known, the dot at least ``gap_um``
  microns closer to it is ``mom_toward_daughter`` and the other
  ``mom_toward_mom``; if neither is clearly closer, or no daughter is known,
  both dots are just ``mom``.

The mother/daughter *cell* roles drive the on-screen colouring; the per-*dot*
roles above are what a spot carries into analysis.

Pure module (numpy / scipy only) — unit-testable without the GUI or micro-sam.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MOM = "mom"
DAUGHTER = "bud"                      # role value shown to the user is "bud"
MOM_TOWARD_DAUGHTER = "mom_toward_bud"
MOM_TOWARD_MOM = "mom_toward_mom"


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
                      exclude: int, areas: Optional[Dict[int, int]] = None
                      ) -> Optional[int]:
    """Largest cell inside ``roi_mask`` other than ``exclude`` (the mother).

    Used to auto-pick the daughter when both dots are in the mother.  Returns
    ``None`` when the ROI holds no second cell.
    """
    if labels is None or roi_mask is None:
        return None
    if areas is None:
        areas = cell_areas(labels)
    vals = {int(v) for v in np.unique(labels[roi_mask]) if v not in (0, exclude)}
    if not vals:
        return None
    return max(vals, key=lambda v: areas.get(v, 0))


def pair_by_size(labels: Optional[np.ndarray], roi_mask: Optional[np.ndarray],
                 areas: Optional[Dict[int, int]] = None
                 ) -> Tuple[Optional[int], Optional[int]]:
    """The two largest cells overlapping ``roi_mask`` as ``(mom, daughter)``.

    Used to colour mom vs daughter from segmentation alone, before any spots are
    detected: the biggest cell in the ROI is the mother, the next biggest the
    daughter. Returns ``(None, None)`` when no cell overlaps the ROI.
    """
    if labels is None or roi_mask is None:
        return (None, None)
    if areas is None:
        areas = cell_areas(labels)
    present = sorted((int(v) for v in np.unique(labels[roi_mask]) if v != 0),
                     key=lambda v: areas.get(v, 0), reverse=True)
    mom = present[0] if present else None
    dau = present[1] if len(present) > 1 else None
    return (mom, dau)


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
                areas: Optional[Dict[int, int]] = None) -> dict:
    """Assign mom/daughter roles to a two-spot ROI's dots.

    ``spots`` is ``[(row, col), (row, col)]``.  ``roi_mask`` (bool, image-sized)
    lets the daughter be auto-picked as the other cell in the ROI; pass
    ``daughter_cell`` to force it.  ``gap_um`` is the "clearly closer" distance
    threshold in microns.

    Returns::

        {"spot_roles": [role, role],     # aligned with `spots`
         "mom_cell": int|None,
         "daughter_cell": int|None,
         "needs_daughter": bool,         # both in mom but no daughter found
         "ok": bool,
         "reason": str}
    """
    if areas is None:
        areas = cell_areas(labels)
    cells = [cell_at(labels, r, c) for r, c in spots]
    out = {"spot_roles": ["", ""], "mom_cell": None, "daughter_cell": None,
           "needs_daughter": False, "ok": False, "reason": ""}
    if len(spots) != 2:
        out["reason"] = "ROI does not have exactly two spots"
        return out
    a, b = cells
    if a == 0 or b == 0:
        out["reason"] = "a spot is not inside any segmented cell"
        return out

    if a != b:                                    # dots in two cells
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
        out["ok"] = True
        return out

    # both dots in the same (mother) cell
    mom = a
    out["mom_cell"] = mom
    dau = daughter_cell
    if dau is None:
        dau = other_cell_in_roi(labels, roi_mask, exclude=mom, areas=areas)
    if dau is None or dau == mom:
        out["spot_roles"] = [MOM, MOM]
        out["needs_daughter"] = True
        out["ok"] = True
        out["reason"] = "both dots in the mother; no daughter cell found"
        return out

    out["daughter_cell"] = dau
    d = [x * float(pixel_size) for x in _dist_to_cell(labels, dau, spots)]
    if abs(d[0] - d[1]) >= float(gap_um):
        near = 0 if d[0] < d[1] else 1
        roles = [MOM_TOWARD_MOM, MOM_TOWARD_MOM]
        roles[near] = MOM_TOWARD_DAUGHTER
        out["spot_roles"] = roles
    else:
        out["spot_roles"] = [MOM, MOM]
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
