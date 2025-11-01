from __future__ import annotations
import os
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

# Optional: Pillow gives us clean RGB, EXIF orientation, and multi-page TIFF handling
try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# Optional: imageio as a fallback
try:
    import imageio.v3 as iio  # type: ignore
    _HAS_IMAGEIO_V3 = True
except Exception:
    _HAS_IMAGEIO_V3 = False
    try:
        import imageio as iio  # type: ignore
        _HAS_IMAGEIO = True
    except Exception:
        _HAS_IMAGEIO = False

# default data root (same as FastAPI DATA_DIR). You can override via env.
DATA_ROOT = Path(os.environ.get("DATA_DIR", "/tmp/lcteller")).resolve()

def _resolve_to_data_root(p: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    If p is absolute and exists, return it.
    Otherwise, treat p as a relative path under base_dir (or DATA_ROOT).
    Also normalizes backslashes to forward slashes first.
    """
    s = str(p).replace("\\", "/")
    cand = Path(s)
    if cand.is_absolute() and cand.exists():
        return cand
    base = (base_dir or DATA_ROOT)
    return (base / cand).resolve()

@dataclass
class LoadReport:
    path: str
    shape: Tuple[int, ...]
    dtype: str
    mode: Optional[str] = None          # PIL image mode if Pillow was used
    pages: Optional[int] = None         # number of frames/pages (TIFF etc.)
    used_backend: str = "unknown"       # "Pillow" | "imageio" | "numpy"
    warnings: list[str] = field(default_factory=list)


def _to_uint8(arr: np.ndarray, report: LoadReport) -> np.ndarray:
    """Convert various dtypes to uint8 safely."""
    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        # common cases: [0, 1] or [0, 255]
        finite = np.isfinite(arr)
        vmin = float(np.nanmin(arr[finite])) if finite.any() else 0.0
        vmax = float(np.nanmax(arr[finite])) if finite.any() else 1.0
        if vmax <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
            report.warnings.append("float image assumed in [0,1] → scaled to [0,255].")
        else:
            arr = np.clip(arr, 0.0, 255.0)
            report.warnings.append("float image assumed in [0,255] → clipped to [0,255].")
        return arr.astype(np.uint8)

    if arr.dtype == np.uint16:
        # scale 16-bit to 8-bit (simple >> 8 keeps speed and avoids overflow)
        report.warnings.append("uint16 image downscaled to uint8 (>> 8).")
        return (arr >> 8).astype(np.uint8)

    if arr.dtype == np.int16:
        report.warnings.append("int16 image converted to uint8 by clipping to [0,255].")
        return np.clip(arr, 0, 255).astype(np.uint8)

    # last resort
    report.warnings.append(f"image dtype {arr.dtype} converted to uint8 by clipping.")
    return np.clip(arr.astype(np.float32), 0, 255).astype(np.uint8)


def _ensure_rgb(arr: np.ndarray, report: LoadReport) -> np.ndarray:
    """
    Ensure HxWx3 RGB uint8.
    - Gray → stack to 3 channels
    - RGBA → drop alpha with straight alpha (no premultiply correction)
    - Extra channels → keep first 3
    """
    if arr.ndim == 2:
        report.warnings.append("grayscale → RGB by channel stacking.")
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape {arr.shape}; expected HxW or HxWxC.")

    H, W, C = arr.shape
    if C == 3:
        return arr.astype(np.uint8, copy=False)

    if C >= 4:
        # drop alpha; if you need proper alpha compositing later, we can add it
        report.warnings.append("alpha channel dropped → RGB.")
        arr = arr[..., :3]
        return arr.astype(np.uint8, copy=False)

    # C == 1 (should have been covered above), but just in case
    if C == 1:
        report.warnings.append("single-channel image expanded to RGB.")
        arr = np.repeat(arr, 3, axis=-1)
        return arr.astype(np.uint8, copy=False)

    # C == 2 or other odd cases
    report.warnings.append(f"{C}-channel image reduced/expanded to RGB.")
    if C > 3:
        arr = arr[..., :3]
    else:
        # pad with last channel
        pads = [arr[..., -1]] * (3 - C)
        arr = np.concatenate([arr] + pads, axis=-1)
    return arr.astype(np.uint8, copy=False)


def load_image(
    path: Union[str, Path],
    *,
    page: int = 0,
    exif_orient: bool = True,
    base_dir: Optional[Union[str, Path]] = None,
    max_mp: Optional[float] = 200.0,
) -> Tuple[np.ndarray, LoadReport]:
    """
    Load an image from disk and return (rgb_uint8, report).

    - Always returns uint8 HxWx3 RGB.
    - Handles grayscale, RGBA, float/16-bit, and EXIF orientation.
    - For multi-page images (TIFF), loads the first page by default.
    - Optional pixel cap via `max_mp` (mega-pixels) to avoid reading huge images by mistake.

    Args:
        path: file path
        page: page/frame index for multi-page files
        exif_orient: apply EXIF orientation (Pillow only)
        max_mp: if set, raises if image exceeds this many mega-pixels

    Raises:
        FileNotFoundError, ValueError on bad shapes or too-large image.
    """
    base = Path(base_dir).resolve() if base_dir else None
    p = _resolve_to_data_root(path, base)
    if not p.exists():
        raise FileNotFoundError(str(p))

    report = LoadReport(path=str(p), shape=(), dtype="", mode=None, pages=None, used_backend="unknown")

    # Prefer Pillow for clean handling
    if _HAS_PIL:
        with Image.open(p) as im:
            # Get number of frames (TIFF, GIF, etc.)
            n_frames = getattr(im, "n_frames", 1)
            report.pages = int(n_frames)
            # Seek desired page
            if page and page < n_frames:
                im.seek(page)
            # EXIF orientation
            if exif_orient:
                try:
                    im = ImageOps.exif_transpose(im)
                except Exception:
                    report.warnings.append("EXIF transpose failed or not present.")

            report.mode = im.mode
            report.used_backend = "Pillow"

            # Convert to RGB directly via Pillow
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)

    elif _HAS_IMAGEIO_V3 or _HAS_IMAGEIO:
        report.used_backend = "imageio"
        # imageio v3 can auto-RGB with 'rgb=True' param; not always available on v2
        try:
            if _HAS_IMAGEIO_V3:
                arr = iio.imread(p, index=page, extension=p.suffix, plugin=None)  # type: ignore
            else:
                arr = iio.imread(p)  # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to read image with imageio: {e}")

        report.mode = None
        # standardize dtype and channels
        arr = _to_uint8(arr, report)
        arr = _ensure_rgb(arr, report)

    else:
        raise RuntimeError("No Pillow or imageio installed. Please install 'pillow' or 'imageio'.")

    # size guard
    H, W = arr.shape[:2]
    mp = (H * W) / 1_000_000.0
    report.shape = arr.shape
    report.dtype = str(arr.dtype)

    if max_mp is not None and mp > max_mp:
        raise ValueError(f"Image too large ({mp:.1f} MP > {max_mp} MP): {p.name}")

    return arr, report
