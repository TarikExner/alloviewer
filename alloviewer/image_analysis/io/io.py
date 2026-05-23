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
    """Infer image bit depth and scale image data to ``float32`` in ``[0, 1]``.

    Parameters
    ----------
    arr : numpy.ndarray
        Raw image array.
    report : LoadReport
        Load report updated in place with dtype, bit-depth, white-level,
        shift, and warning information.
    fast : bool, optional
        If ``True``, skips percentile-based uint16 bit-depth detection. The
        default is ``True``.

    Returns
    -------
    scaled_arr : numpy.ndarray
        Image array converted to ``float32`` and scaled to ``[0, 1]``.
    bit_depth : int or None
        Inferred bit depth. Returns ``None`` when no meaningful bit depth can
        be assigned.

    Notes
    -----
    Boolean, uint8, uint16, integer, and floating-point arrays are handled
    separately. For floating-point arrays outside common ranges, min-max
    scaling is used and a warning is added to ``report``.
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
    """Read an image from disk with OpenCV.

    Parameters
    ----------
    path : pathlib.Path
        Image path.
    page : int
        Page index for multi-page TIFF files. For non-TIFF files, this value is
        ignored.
    report : LoadReport
        Load report updated in place with path, page count, and backend
        information.

    Returns
    -------
    numpy.ndarray
        Raw image array with original dtype.

    Raises
    ------
    RuntimeError
        If OpenCV fails to read the image.
    ValueError
        If a requested TIFF page does not exist.
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
    """Read an image from bytes or a file-like object with OpenCV.

    Parameters
    ----------
    src : SourceType
        Raw bytes, bytearray, or file-like object with a ``read`` method.
    page : int
        Requested page index. Only page ``0`` is available for in-memory
        decoding.
    report : LoadReport
        Load report updated in place with path, page count, backend, and
        warnings.

    Returns
    -------
    numpy.ndarray
        Raw image array with original dtype.

    Raises
    ------
    RuntimeError
        If OpenCV fails to decode the image from memory.

    Notes
    -----
    OpenCV's in-memory decoder returns only the first page for multi-page TIFF
    data.
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
    """Open an image with OpenCV without scaling or channel normalization.

    Parameters
    ----------
    source : SourceType
        Image source. Supported inputs are ``str``, ``Path``, ``bytes``,
        ``bytearray``, and file-like objects.
    page : int, optional
        Page index for multi-page TIFF files on disk. The default is ``0``.
    base_dir : str or pathlib.Path or None, optional
        Optional root directory used to resolve relative paths.
    max_mp : float or None, optional
        Maximum allowed image size in megapixels. If ``None``, no size check is
        applied. The default is ``200.0``.

    Returns
    -------
    arr : numpy.ndarray
        Raw image array with original dtype and OpenCV channel order.
    report : LoadReport
        Metadata and warnings collected during loading.

    Raises
    ------
    FileNotFoundError
        If a path source does not exist.
    ValueError
        If the image has fewer than two dimensions or exceeds ``max_mp``.
    RuntimeError
        If OpenCV fails to read or decode the image.
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
    """Scale a raw image array to ``float32`` in ``[0, 1]``.

    Parameters
    ----------
    arr : numpy.ndarray
        Raw image array.
    report : LoadReport or None, optional
        Existing load report to update. If ``None``, a new report is created.
    fast : bool, optional
        If ``True``, skips percentile-based uint16 bit-depth detection. The
        default is ``False``.

    Returns
    -------
    scaled : numpy.ndarray
        Scaled image array with dtype ``float32``.
    report : LoadReport
        Updated load report.
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
    """Load an image and return RGB data.

    Parameters
    ----------
    source : SourceType
        Image source. Supported inputs are ``str``, ``Path``, ``bytes``,
        ``bytearray``, and file-like objects.
    page : int, optional
        Page index for multi-page TIFF files on disk. The default is ``0``.
    base_dir : str or pathlib.Path or None, optional
        Optional root directory used to resolve relative paths.
    max_mp : float or None, optional
        Maximum allowed image size in megapixels. If ``None``, no size check is
        applied. The default is ``200.0``.
    as_chw : bool, optional
        If ``True``, return data as ``(C, H, W)``. If ``False``, return data as
        ``(H, W, C)``. The default is ``True``.
    scale : bool, optional
        If ``True``, scale image values to ``float32`` in ``[0, 1]``. The
        default is ``True``.
    fast_scale : bool, optional
        If ``True``, skips percentile-based uint16 bit-depth detection during
        scaling. The default is ``True``.
    **kwargs : Any
        Accepted for caller compatibility. Values are not used.

    Returns
    -------
    arr : numpy.ndarray
        RGB image array, optionally scaled and optionally moved to CHW layout.
    report : LoadReport
        Metadata and warnings collected during loading.

    Notes
    -----
    Color images loaded by OpenCV are converted from BGR to RGB before channel
    normalization. Single-channel images are expanded to RGB by
    ``ensure_rgb_anydepth``.
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
    """Load multiple images from one directory.

    Parameters
    ----------
    filenames : list of str
        Image filenames or relative paths.
    data_dir : str
        Base directory used to resolve image paths.
    scale : bool, optional
        If ``True``, scale image values to ``float32`` in ``[0, 1]``. The
        default is ``True``.
    **kwargs : Any
        Additional keyword arguments passed to :func:`load_image`.

    Returns
    -------
    list of numpy.ndarray
        Loaded image arrays in the order of ``filenames``.

    Raises
    ------
    FileNotFoundError
        If any requested file does not exist.
    ValueError
        If any image is invalid or exceeds the configured size limit.
    RuntimeError
        If OpenCV fails to read or decode any image.
    """
    res: List[np.ndarray] = []

    for file in filenames:
        img, _ = load_image(
            file,
            base_dir=data_dir,
            scale=scale,
            **kwargs,
        )
        res.append(img)

    return res
