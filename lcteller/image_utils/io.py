from __future__ import annotations
import os
import math
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, BinaryIO, Any, List

import numpy as np
import cv2

# default data root (same as FastAPI DATA_DIR). You can override via env.
DATA_ROOT = Path(os.environ.get("DATA_DIR", "/tmp/lcteller")).resolve()

SourceType = Union[str, Path, bytes, bytearray, BinaryIO]

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
    mode: Optional[str] = None          # not really used with cv2, kept for compat
    pages: Optional[int] = None         # number of frames/pages (TIFF etc.)
    used_backend: str = "opencv"
    bit_depth: Optional[int] = None     # 8, 10, 12, 14, 16, ...
    white_level: Optional[int] = None   # nominal white level before scaling
    shifted: Optional[bool] = None      # did it look left-shifted in uint16?
    warnings: list[str] = field(default_factory=list)


def _ensure_rgb_anydepth(arr: np.ndarray, report: LoadReport) -> np.ndarray:
    """
    Ensure HxWx3, keep dtype as-is.

    - Gray → stack to 3 channels
    - RGBA / BGRA → drop alpha later (we already converted color channels)
    - >3 channels → first 3
    """
    if arr.ndim == 2:
        report.warnings.append("grayscale → RGB by channel stacking.")
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape {arr.shape}; expected HxW or HxWxC.")

    h, w, c = arr.shape
    if c == 3:
        return arr

    if c >= 4:
        report.warnings.append("alpha channel dropped → RGB.")
        return arr[..., :3]

    if c == 1:
        report.warnings.append("single-channel image expanded to RGB.")
        return np.repeat(arr, 3, axis=-1)

    # c == 2 or odd cases
    report.warnings.append(f"{c}-channel image reduced/expanded to RGB.")
    if c > 3:
        arr = arr[..., :3]
    else:
        pads = [arr[..., -1]] * (3 - c)
        arr = np.concatenate([arr] + pads, axis=-1)
    return arr


def _detect_bitdepth_u16(a: np.ndarray, p: float = 99.9) -> Tuple[int, int, bool]:
    """
    Detect bit depth (8/10/12/14/16) for a uint16 image.

    Returns:
      bit_depth : int, e.g. 12
      white     : int, nominal white level (unshifted), e.g. 4095
      shifted   : bool, True if data looks left-shifted in uint16
    """
    if a.dtype != np.uint16:
        raise TypeError("Expected uint16 array for _detect_bitdepth_u16")

    vmax = int(a.max())
    if vmax == 0:
        # no information; choose 16-bit as a safe default
        return 16, (1 << 16) - 1, False

    # look for left-shifted patterns:
    # low bits all zero across the image
    # use smaller bit depths first so 10/12-bit shifted are not mis-labeled as 14-bit
    for b in (8, 10, 12, 14):
        shift = 16 - b
        low_mask = (1 << shift) - 1
        if (a & low_mask).max() == 0:
            white = (1 << b) - 1
            return b, white, True

    # fallback: estimate from a high percentile
    sample = float(np.percentile(a, p))
    sample = max(1.0, sample)
    est_bits = int(math.ceil(math.log2(sample + 1.0)))
    est_bits = min(16, max(2, est_bits))

    allowed = (8, 10, 12, 14, 16)
    b = min(allowed, key=lambda k: abs(k - est_bits))
    white = (1 << b) - 1

    # if actual max exceeds the nominal white, treat as full 16-bit
    if vmax > white:
        return 16, (1 << 16) - 1, False

    return b, white, False


def _infer_bit_depth_and_normalize_scale(arr: np.ndarray, report: LoadReport) -> Tuple[np.ndarray, Optional[int]]:
    """
    Convert array to float32 in [0, 1].

    Returns (scaled_arr, bit_depth).

    bit_depth is the nominal bit depth based on dtype, when possible.
    """
    dtype = arr.dtype
    report.dtype = str(dtype)

    # bool → [0,1]
    if np.issubdtype(dtype, np.bool_):
        report.bit_depth = 1
        report.white_level = 1
        report.shifted = False
        return arr.astype(np.float32), 1

    # plain 8-bit
    if np.issubdtype(dtype, np.uint8):
        report.bit_depth = 8
        report.white_level = 255
        report.shifted = False
        return (arr.astype(np.float32) / 255.0), 8

    # uint16 with possible 10/12/14/16-bit encoding
    if np.issubdtype(dtype, np.uint16):
        bit_depth, white, shifted = _detect_bitdepth_u16(arr)
        report.bit_depth = bit_depth
        report.white_level = white
        report.shifted = shifted

        img_u16 = arr
        if shifted and bit_depth < 16:
            shift = 16 - bit_depth
            report.warnings.append(
                f"uint16 image appears left-shifted ({bit_depth}-bit); shifting right by {shift}."
            )
            img_u16 = (img_u16 >> shift).astype(np.uint16)

        scaled = img_u16.astype(np.float32) / float(white)
        scaled = np.clip(scaled, 0.0, 1.0)
        return scaled, bit_depth

    # signed 16-bit: we do not know real bit depth, so treat as range scaling
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

    # other integer types
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        max_val = float(info.max)
        scaled = arr.astype(np.float32) / max_val
        report.bit_depth = int(np.log2(info.max + 1))
        report.white_level = info.max
        report.shifted = False
        report.warnings.append(f"integer image dtype {dtype} scaled by max {info.max}.")
        return scaled, report.bit_depth

    # floating types
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

        # already [0,1]
        if vmin >= 0.0 and vmax <= 1.0:
            scaled = np.clip(arr32, 0.0, 1.0)
            report.bit_depth = None
            report.white_level = 1
            report.shifted = False
            return scaled, None

        # common [0,255]
        if vmin >= 0.0 and vmax <= 255.0:
            report.warnings.append("float image assumed in [0,255] → scaled by 255.")
            scaled = np.clip(arr32, 0.0, 255.0) / 255.0
            report.bit_depth = 8
            report.white_level = 255
            report.shifted = False
            return scaled, 8

        # common [0,65535]
        if vmin >= 0.0 and vmax <= 65535.0:
            report.warnings.append("float image assumed in [0,65535] → scaled by 65535.")
            scaled = np.clip(arr32, 0.0, 65535.0) / 65535.0
            report.bit_depth = 16
            report.white_level = 65535
            report.shifted = False
            return scaled, 16

        # general case: scale by min/max
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

    # last resort: cast to float and scale by min/max
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




def _read_cv2_from_path(path: Path, page: int, report: LoadReport) -> np.ndarray:
    """
    Read from disk using cv2.

    - For TIFF and page>0, use imreadmulti.
    - For all others or page==0, use imread (first page only).
    """
    report.path = str(path.resolve())
    ext = path.suffix.lower()

    if ext in (".tif", ".tiff") and page != 0:
        ok, imgs = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
        if not ok or not imgs:
            raise RuntimeError(f"Failed to read TIFF (multi-page) with OpenCV: {path}")
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


def _read_cv2_from_bytes_or_file(src: SourceType, page: int, report: LoadReport) -> np.ndarray:
    """
    Read from bytes or file-like using cv2.imdecode.

    Only first page is available for multi-page TIFFs here.
    """
    if isinstance(src, (bytes, bytearray)):
        buf = bytes(src)
        report.path = "<bytes>"
    else:
        # file-like
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
        report.warnings.append("page>0 requested for in-memory image; only first page is returned.")
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

    - Accepts paths (str/Path), raw bytes, or file-like objects (e.g. FastAPI UploadFile.file).
    - Supports common formats: TIFF, PNG, JPEG, and others supported by OpenCV.
    - Returns an array in HxW or HxWxC with the original dtype.
    - Does not scale or reorder channels. That is handled by `scale_image` / `load_image`.
    """
    resolved_source: Any = source
    base = Path(base_dir).resolve() if base_dir else None

    if isinstance(source, (str, Path)):
        resolved_source = _resolve_to_data_root(source, base)
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
        arr = _read_cv2_from_path(resolved_source, page=page, report=report)
    else:
        arr = _read_cv2_from_bytes_or_file(resolved_source, page=page, report=report)

    if arr.ndim < 2:
        raise ValueError(f"Unsupported image ndim {arr.ndim}; expected at least 2D.")

    h, w = arr.shape[:2]
    mp = (h * w) / 1_000_000.0
    report.shape = arr.shape

    if max_mp is not None and mp > max_mp:
        raise ValueError(f"Image too large ({mp:.1f} MP > {max_mp} MP).")

    return arr, report


def scale_image(
    arr: np.ndarray,
    report: Optional[LoadReport] = None,
) -> Tuple[np.ndarray, Optional[LoadReport]]:
    """
    Scale a raw image array to float32 in [0,1].

    - Keeps shape as-is.
    - If a report is given, updates dtype/bit_depth/white_level/shifted and adds warnings.

    Returns (scaled_arr, report).
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
    scaled, bit_depth = _infer_bit_depth_and_normalize_scale(arr, report)
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
) -> Tuple[np.ndarray, Optional[LoadReport]]:
    """
    High-level helper for most use cases.

    - Opens the image (path/bytes/file-like) with OpenCV.
    - Converts BGR → RGB for color images.
    - Ensures RGB (3 channels).
    - Optionally scales to [0,1] float32.
    - Optionally returns as [3, H, W] instead of [H, W, 3].

    This is handy for FastAPI endpoints: pass `UploadFile.file` directly as `source`.
    """
    arr, report = open_image(
        source,
        page=page,
        base_dir=base_dir,
        max_mp=max_mp,
    )

    # OpenCV gives BGR for color images; convert to RGB before channel handling.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # only use first 3 channels for color conversion; alpha handled later
        bgr = arr[..., :3]
        arr_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = arr_rgb

    arr = _ensure_rgb_anydepth(arr, report)

    if scale:
        arr, report = scale_image(arr, report)

    if as_chw:
        # [H, W, 3] -> [3, H, W]
        arr = np.moveaxis(arr, -1, 0)

    return arr, report

def load_images(filenames: List[str],
                data_dir: str) -> List[np.ndarray]:
    res = []
    for file in filenames:
        img, _ = load_image(file, base_dir = data_dir)
        res.append(img)
    return res

