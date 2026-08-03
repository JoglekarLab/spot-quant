"""Image loading and metadata read/write for nd2 and tif/tiff files.

A single :class:`ImageMeta` describes the editable fields exposed in the GUI:
channel names, pixel size, magnification and time step.  Values are read
from the file where available and otherwise fall back to defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

SUPPORTED_SUFFIXES = (".tif", ".tiff", ".nd2")

DEFAULT_PIXEL_SIZE = 0.1        # microns / pixel
DEFAULT_MAGNIFICATION = 60.0    # objective magnification
DEFAULT_TIME_STEP = 1.0         # seconds between frames
DEFAULT_Z_STEP = 0.5            # microns between z-planes


@dataclass
class ImageMeta:
    channel_names: List[str] = field(default_factory=list)
    pixel_size: float = DEFAULT_PIXEL_SIZE       # microns / pixel
    pixel_unit: str = "um"
    magnification: float = DEFAULT_MAGNIFICATION
    time_step: float = DEFAULT_TIME_STEP         # seconds
    z_step: float = DEFAULT_Z_STEP               # microns between z-planes
    # Whether the value was read from the file (vs. a fallback default).
    z_step_known: bool = False
    time_step_known: bool = False

    def as_dict(self) -> dict:
        return {
            "channel_names": list(self.channel_names),
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "magnification": self.magnification,
            "time_step": self.time_step,
            "z_step": self.z_step,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def list_image_files(folder: Path) -> List[Path]:
    """All tif/tiff/nd2 files in *folder*, sorted by name."""
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=lambda p: p.name.lower())


def load_image(path: Path):
    """Return ``(array, ImageMeta)``.

    The array is returned with a leading channel axis when channels are
    present: shape ``(C, Y, X)`` for 2-D data or ``(C, Z, Y, X)`` for a
    z-stack.  A single-channel image is returned as ``(1, ...)``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".nd2":
        return _load_nd2(path)
    if suffix in (".tif", ".tiff"):
        return _load_tif(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_nd2(path: Path):
    import nd2

    arr = nd2.imread(path)  # typically (Z, C, Y, X) or (C, Y, X)
    meta = ImageMeta()
    try:
        with nd2.ND2File(path) as f:
            md = f.metadata
            ch = md.contents.channelCount
            meta.channel_names = [md.channels[i].channel.name for i in range(ch)]
            # voxel size in microns
            vox = f.voxel_size()
            if vox is not None and vox.x:
                meta.pixel_size = float(vox.x)
                meta.pixel_unit = "um"
            if vox is not None and getattr(vox, "z", None):
                meta.z_step = float(vox.z)
                meta.z_step_known = True
    except Exception:
        pass

    arr, n_channels = _to_channel_first(arr, len(meta.channel_names) or None)
    if not meta.channel_names:
        meta.channel_names = [f"Ch{i}" for i in range(n_channels)]
    return arr, meta


def _load_tif(path: Path):
    import tifffile

    meta = ImageMeta()
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        # Pixel size from resolution tags (pixels per unit -> unit per pixel).
        try:
            page = tf.pages[0]
            xres = page.tags.get("XResolution")
            if xres is not None:
                num, den = xres.value
                if num:
                    meta.pixel_size = float(den) / float(num)
        except Exception:
            pass
        # ImageJ metadata may carry channel/finterval info.
        ij = getattr(tf, "imagej_metadata", None) or {}
        if "finterval" in ij:
            meta.time_step = float(ij["finterval"])
            meta.time_step_known = True
        if "spacing" in ij:                 # ImageJ z-spacing (microns)
            meta.z_step = float(ij["spacing"])
            meta.z_step_known = True
        n_ch_hint = int(ij.get("channels", 0)) or None

    arr, n_channels = _to_channel_first(arr, n_ch_hint)
    if not meta.channel_names:
        meta.channel_names = [f"Ch{i}" for i in range(n_channels)]
    return arr, meta


def _to_channel_first(arr: np.ndarray, n_channels_hint=None):
    """Normalise an array to channel-first layout and report channel count.

    Returns ``(array, n_channels)`` where the array is ``(C, Y, X)`` or
    ``(C, Z, Y, X)``.  The channel axis is guessed as the smallest of the
    non-spatial axes, optionally constrained by *n_channels_hint*.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:                      # (Y, X)
        return arr[None], 1
    if arr.ndim == 3:                      # (C, Y, X) or (Z, Y, X)
        c0 = arr.shape[0]
        if n_channels_hint and c0 == n_channels_hint:
            return arr, c0
        # Heuristic: a small leading axis is channels, else treat as 1-channel Z.
        if c0 <= 8:
            return arr, c0
        return arr[None], 1
    if arr.ndim == 4:                      # (Z, C, Y, X) or (C, Z, Y, X)
        # Move the smaller of the first two axes to the front as channels.
        if arr.shape[1] <= arr.shape[0]:
            arr = np.moveaxis(arr, 1, 0)   # -> (C, Z, Y, X)
        return arr, arr.shape[0]
    raise ValueError(f"Unsupported array shape: {arr.shape}")
