from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from .utils import (
    LoadReport,
    SourceType,
    detect_bitdepth_u16,
    ensure_rgb_anydepth,
    resolve_to_data_root,
)


def infer_bit_depth_and_normalize_scale(
    arr: np.ndarray,
    report: LoadReport,
    *,
    fast: bool = True,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Convert array to float32 in [0, 1].

    Returns:
      scaled_arr, bit_depth
    """
    dtype = arr.dtype
    report.dtype = str(dtype)

    if np.issubdtype(dtype, np.bool_):
        report.bit_depth = 1
        report.white_level = 1
        report.shifted = False
        return arr.astype(np.float32), 1

    if np.issubdtype(dtype, np.uint8):
        report.bit_depth = 8
        report.white_level = 255
        report.shifted = False
        return arr.astype(np.float32) / 255.0, 8

    if np.issubdtype(dtype, np.uint16):
        bit_depth, white, shifted = detect_bitdepth_u16(
            arr,
            use_percentile=not fast,
        )

        report.bit_depth = bit_depth
        report.white_level = white
        report.shifted = shifted

        img_u16 = arr

        if shifted and bit_depth < 16:
            shift = 16 - bit_depth
            report.warnings.append(
                f"uint16 image appears left-shifted ({bit_depth}-bit); "
                f"shifting right by {shift}."
            )
            img_u16 = (img_u16 >> shift).astype(np.uint16)

        scaled = img_u16.astype(np.float32) / float(white)
        scaled = np.clip(scaled, 0.0, 1.0)

        return scaled, bit_depth

    if np.issubdtype(dtype, np.int16):
        amin = float(arr.min())
        amax = float(arr.max())

        if amax <= 0:
            report.warnings.append("int16 image has non-positive range; clipping to [0,1].")
            scaled = np.clip(arr.astype(np.float32), 0, 1)
            report.bit_depth = None
            report.white_level = None
            report.shifted = False
            return scaled, None

        if amin < 0:
            arr_shifted = arr.astype(np.float32) - amin
            amax = float(arr_shifted.max())
        else:
            arr_shifted = arr.astype(np.float32)

        scaled = arr_shifted / max(amax, 1.0)

        report.warnings.append("int16 image scaled to [0,1] based on value range.")
        report.bit_depth = None
        report.white_level = None
        report.shifted = False

        return scaled, None

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        max_val = float(info.max)

        scaled = arr.astype(np.float32) / max_val

        report.bit_depth = int(np.log2(info.max + 1))
        report.white_level = int(info.max)
        report.shifted = False
        report.warnings.append(f"integer image dtype {dtype} scaled by max {info.max}.")

        return scaled, report.bit_depth

    if np.issubdtype(dtype, np.floating):
        finite = np.isfinite(arr)

        if not finite.any():
            report.warnings.append("float image has no finite values; returning zeros.")
            report.bit_depth = None
            report.white_level = None
            report.shifted = False
            return np.zeros_like(arr, dtype=np.float32), None

        vmin = float(np.nanmin(arr[finite]))
        vmax = float(np.nanmax(arr[finite]))
        arr32 = arr.astype(np.float32)

        if vmin >= 0.0 and vmax <= 1.0:
            scaled = np.clip(arr32, 0.0, 1.0)
            report.bit_depth = None
            report.white_level = 1
            report.shifted = False
            return scaled, None

        if vmin >= 0.0 and vmax <= 255.0:
            report.warnings.append("float image assumed in [0,255] -> scaled by 255.")
            scaled = np.clip(arr32, 0.0, 255.0) / 255.0
            report.bit_depth = 8
            report.white_level = 255
            report.shifted = False
            return scaled, 8

        if vmin >= 0.0 and vmax <= 65535.0:
            report.warnings.append("float image assumed in [0,65535] -> scaled by 65535.")
            scaled = np.clip(arr32, 0.0, 65535.0) / 65535.0
            report.bit_depth = 16
            report.white_level = 65535
            report.shifted = False
            return scaled, 16

        report.warnings.append(
            "float image scaled to [0,1] based on min/max; "
            f"original range [{vmin:.3g}, {vmax:.3g}]."
        )

        arr32 = arr32 - vmin
        vmax2 = float(arr32.max())

        if vmax2 <= 0:
            report.bit_depth = None
            report.white_level = None
            report.shifted = False
            return np.zeros_like(arr32), None

        scaled = arr32 / vmax2

        report.bit_depth = None
        report.white_level = None
        report.shifted = False

        return scaled, None

    report.warnings.append(f"image dtype {dtype} scaled to [0,1] via min/max.")

    arr32 = arr.astype(np.float32)
    vmin = float(arr32.min())
    vmax = float(arr32.max())

    if vmax <= vmin:
        report.bit_depth = None
        report.white_level = None
        report.shifted = False
        return np.zeros_like(arr32), None

    scaled = (arr32 - vmin) / (vmax - vmin)

    report.bit_depth = None
    report.white_level = None
    report.shifted = False

    return scaled, None

def read_cv2_from_path(path: Path, page: int, report: LoadReport) -> np.ndarray:
    """
    Read from disk using OpenCV.

    For TIFF and page > 0, uses imreadmulti.
    Otherwise uses imread.
    """
    report.path = str(path.resolve())
    ext = path.suffix.lower()

    if ext in (".tif", ".tiff") and page != 0:
        ok, imgs = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)

        if not ok or not imgs:
            raise RuntimeError(f"Failed to read TIFF with OpenCV: {path}")

        report.pages = len(imgs)

        if page >= len(imgs):
            raise ValueError(f"Requested page {page}, but TIFF has {len(imgs)} pages.")

        arr = imgs[page]

    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if arr is None:
            raise RuntimeError(f"Failed to read image with OpenCV: {path}")

        report.pages = 1

    report.used_backend = "opencv"
    return arr

def read_cv2_from_bytes_or_file(
    src: SourceType,
    page: int,
    report: LoadReport,
) -> np.ndarray:
    """
    Read from bytes or file-like object using cv2.imdecode.

    Only the first page is available for in-memory multi-page TIFFs.
    """
    if isinstance(src, (bytes, bytearray)):
        buf = bytes(src)
        report.path = "<bytes>"
    else:
        if hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass

        buf = src.read()  # type: ignore[attr-defined]
        report.path = "<file-like>"

    data = np.frombuffer(buf, dtype=np.uint8)
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if arr is None:
        raise RuntimeError("Failed to decode image from memory with OpenCV.")

    report.pages = 1
    report.used_backend = "opencv"

    if page != 0:
        report.warnings.append(
            "page > 0 requested for in-memory image; only first page is returned."
        )

    return arr


def open_image(
    source: SourceType,
    *,
    page: int = 0,
    base_dir: Optional[Union[str, Path]] = None,
    max_mp: Optional[float] = 200.0,
) -> Tuple[np.ndarray, LoadReport]:
    """
    Low-level image loader using OpenCV.

    Accepts:
      - str / Path
      - raw bytes
      - bytearray
      - file-like object

    Returns raw image data with original dtype.
    No scaling or channel reordering is done here.
    """
    resolved_source: Any = source
    base = Path(base_dir).resolve() if base_dir else None

    if isinstance(source, (str, Path)):
        resolved_source = resolve_to_data_root(source, base)
        p = Path(resolved_source)

        if not p.exists():
            raise FileNotFoundError(str(p))

        path_for_report = str(p)
    else:
        path_for_report = "<memory>"

    report = LoadReport(
        path=path_for_report,
        shape=(),
        dtype="",
        mode=None,
        pages=None,
        used_backend="opencv",
    )

    if isinstance(resolved_source, Path):
        arr = read_cv2_from_path(resolved_source, page=page, report=report)
    else:
        arr = read_cv2_from_bytes_or_file(resolved_source, page=page, report=report)

    if arr.ndim < 2:
        raise ValueError(f"Unsupported image ndim {arr.ndim}; expected at least 2D.")

    h, w = arr.shape[:2]
    mp = (h * w) / 1_000_000.0

    report.shape = arr.shape
    report.dtype = str(arr.dtype)

    if max_mp is not None and mp > max_mp:
        raise ValueError(f"Image too large ({mp:.1f} MP > {max_mp} MP).")

    return arr, report


def scale_image(
    arr: np.ndarray,
    report: Optional[LoadReport] = None,
    *,
    fast: bool = False,
) -> Tuple[np.ndarray, LoadReport]:
    """
    Scale a raw image array to float32 in [0,1].

    fast=True skips percentile-based uint16 bit-depth detection.
    """
    if report is None:
        report = LoadReport(
            path="<unknown>",
            shape=arr.shape,
            dtype=str(arr.dtype),
            mode=None,
            pages=None,
            used_backend="opencv",
        )

    scaled, bit_depth = infer_bit_depth_and_normalize_scale(
        arr,
        report,
        fast=fast,
    )

    report.shape = scaled.shape
    report.bit_depth = bit_depth

    return scaled, report

def load_image(
    source: SourceType,
    *,
    page: int = 0,
    base_dir: Optional[Union[str, Path]] = None,
    max_mp: Optional[float] = 200.0,
    as_chw: bool = True,
    scale: bool = True,
    fast_scale: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, LoadReport]:
    """
    High-level image loader.

    - Opens with OpenCV
    - Converts BGR -> RGB for color images
    - Ensures RGB
    - Optionally scales to [0,1]
    - Optionally returns CHW instead of HWC
    """
    arr, report = open_image(
        source,
        page=page,
        base_dir=base_dir,
        max_mp=max_mp,
    )

    if arr.ndim == 3 and arr.shape[2] >= 3:
        bgr = arr[..., :3]
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    arr = ensure_rgb_anydepth(arr, report)

    if scale:
        arr, report = scale_image(arr, report, fast=fast_scale)

    if as_chw:
        arr = np.moveaxis(arr, -1, 0)

    return arr, report

def load_images(
    filenames: List[str],
    data_dir: str,
    scale: bool = True,
    **kwargs: Any,
) -> List[np.ndarray]:
    """
    Load several images from one data directory.
    """
    res: list[np.ndarray] = []

    for file in filenames:
        img, _ = load_image(
            file,
            base_dir=data_dir,
            scale=scale,
            **kwargs,
        )
        res.append(img)

    return res
