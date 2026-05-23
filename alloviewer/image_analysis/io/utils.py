from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

import numpy as np


DATA_ROOT = Path(os.environ.get("DATA_DIR", "/tmp/lcteller")).resolve()

SourceType = Union[str, Path, bytes, bytearray, BinaryIO]

IMAGE_EXTS: set[str] = {
    ".tif", ".tiff",
    ".png",
    ".jpg", ".jpeg",
    ".bmp",
    ".webp",
}

# Channel markers must be standalone filename tokens separated by one of these.
CHANNEL_SEPARATORS: tuple[str, ...] = ("_", "-", ".", " ")


@dataclass
class LoadReport:
    """Metadata collected while loading and scaling an image.

    Parameters
    ----------
    path : str
        Source path or placeholder for in-memory sources.
    shape : tuple of int
        Image shape after the current processing step.
    dtype : str
        Image dtype after the current processing step.
    mode : str or None, optional
        Optional image mode, if provided by a backend.
    pages : int or None, optional
        Number of pages in the image file, when known.
    used_backend : str, optional
        Backend used for loading. The default is ``"opencv"``.
    bit_depth : int or None, optional
        Inferred bit depth.
    white_level : int or None, optional
        Maximum expected white value for the inferred bit depth.
    shifted : bool or None, optional
        Whether the data appears left-shifted in uint16 storage.
    warnings : list of str, optional
        Non-fatal loading or scaling warnings.
    """

    path: str
    shape: Tuple[int, ...]
    dtype: str
    mode: Optional[str] = None
    pages: Optional[int] = None
    used_backend: str = "opencv"
    bit_depth: Optional[int] = None
    white_level: Optional[int] = None
    shifted: Optional[bool] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChannelCombineReport:
    """Metadata collected while combining monochrome channels into RGB.

    Parameters
    ----------
    group_key : str
        Name of the matched image group.
    output_path : str
        Suggested or written output path.
    input_paths : dict
        Mapping from channel name to source path or source name.
    shape : tuple of int
        Shape of the combined RGB image.
    dtype : str
        Dtype of the combined RGB image.
    bit_depth : int or None
        Inferred shared bit depth for the input channels.
    white_level : int or None
        Shared white level for the inferred bit depth.
    shifted : bool or None
        Whether all non-zero uint16 channels appear left-shifted.
    warnings : list of str, optional
        Non-fatal channel-combination warnings.
    """

    group_key: str
    output_path: str
    input_paths: Dict[str, str]
    shape: Tuple[int, int, int]
    dtype: str
    bit_depth: Optional[int]
    white_level: Optional[int]
    shifted: Optional[bool]
    warnings: list[str] = field(default_factory=list)


def resolve_to_data_root(
    p: Union[str, Path],
    base_dir: Optional[Path] = None,
) -> Path:
    """Resolve a path against a base directory or the configured data root.

    Parameters
    ----------
    p : str or pathlib.Path
        Input path. Backslashes are normalized before resolution.
    base_dir : pathlib.Path or None, optional
        Base directory used for relative paths. If omitted, ``DATA_ROOT`` is
        used.

    Returns
    -------
    pathlib.Path
        Absolute resolved path.

    Notes
    -----
    Existing absolute paths are returned unchanged. Non-existing absolute paths
    are treated like relative paths under ``base_dir`` or ``DATA_ROOT``.
    """
    s = str(p).replace("\\", "/")
    candidate = Path(s)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    base = base_dir or DATA_ROOT
    return (base / candidate).resolve()


def ensure_rgb_anydepth(arr: np.ndarray, report: LoadReport) -> np.ndarray:
    """Ensure that an image has RGB layout while keeping its dtype unchanged.

    Parameters
    ----------
    arr : numpy.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    report : LoadReport
        Load report updated in place with channel-conversion warnings.

    Returns
    -------
    numpy.ndarray
        Image with shape ``(H, W, 3)`` and original dtype.

    Raises
    ------
    ValueError
        If the input cannot be converted to ``(H, W, 3)``.

    Notes
    -----
    Grayscale images are stacked into three channels. Four-channel images drop
    the alpha channel. Images with more than three channels keep the first
    three channels.
    """
    if arr.ndim == 2:
        report.warnings.append("grayscale -> RGB by channel stacking.")
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape {arr.shape}; expected HxW or HxWxC.")

    _, _, c = arr.shape

    if c == 3:
        return arr

    if c >= 4:
        report.warnings.append("alpha channel dropped -> RGB.")
        return arr[..., :3]

    if c == 1:
        report.warnings.append("single-channel image expanded to RGB.")
        return np.repeat(arr, 3, axis=-1)

    report.warnings.append(f"{c}-channel image reduced/expanded to RGB.")

    pads = [arr[..., -1]] * (3 - c)
    return np.concatenate([arr] + pads, axis=-1)


def as_single_channel_raw(arr: np.ndarray, name: Union[str, Path]) -> np.ndarray:
    """Return one raw monochrome channel from loaded image data.

    Parameters
    ----------
    arr : numpy.ndarray
        Input image array. Supported shapes are ``(H, W)``, ``(H, W, 1)``, or
        ``(H, W, C)`` where all channels are identical.
    name : str or pathlib.Path
        Image name used in error messages.

    Returns
    -------
    numpy.ndarray
        Two-dimensional raw image channel.

    Raises
    ------
    ValueError
        If the image has multiple non-identical channels or an unsupported
        shape.

    Notes
    -----
    No scaling, clipping, shifting, or normalization is applied.
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[..., 0]

        first = arr[..., 0]

        if all(np.array_equal(first, arr[..., c]) for c in range(1, arr.shape[2])):
            return first

        raise ValueError(
            f"Expected monochrome image for {name}, but got non-identical "
            f"{arr.shape[2]}-channel data."
        )

    raise ValueError(f"Unsupported image shape {arr.shape} for {name}.")


def detect_bitdepth_u16(
    a: np.ndarray,
    p: float = 99.9,
    use_percentile: bool = True,
    sample_stride: int = 16,
) -> Tuple[int, int, bool]:
    """Infer nominal bit depth for a uint16 image.

    Parameters
    ----------
    a : numpy.ndarray
        Input image with dtype ``uint16``.
    p : float, optional
        Percentile used for bit-depth estimation when ``use_percentile`` is
        ``True``. The default is ``99.9``.
    use_percentile : bool, optional
        If ``True``, estimate bit depth from a percentile of the full array.
        If ``False``, use a strided sample and maximum-value checks.
    sample_stride : int, optional
        Stride used when ``use_percentile`` is ``False``. The default is ``16``.

    Returns
    -------
    bit_depth : int
        Inferred nominal bit depth.
    white : int
        White level for the inferred bit depth.
    shifted : bool
        Whether the image appears left-shifted in uint16 storage.

    Raises
    ------
    TypeError
        If ``a`` is not a uint16 array.

    Notes
    -----
    Left-shift detection checks whether the lower unused bits are all zero for
    candidate bit depths ``8``, ``10``, ``12``, and ``14``.
    """
    if a.dtype != np.uint16:
        raise TypeError("Expected uint16 array for detect_bitdepth_u16")

    flat = a.ravel()
    vmax = int(flat.max())

    if vmax == 0:
        return 16, 65535, False

    if use_percentile:
        sample = flat
    else:
        sample = flat[::sample_stride]

    for b in (8, 10, 12, 14):
        shift = 16 - b
        low_mask = (1 << shift) - 1

        if (sample & low_mask).max() == 0:
            white = (1 << b) - 1
            return b, white, True

    if not use_percentile:
        for b in (8, 10, 12, 14, 16):
            white = (1 << b) - 1
            if vmax <= white:
                return b, white, False

        return 16, 65535, False

    sample_val = float(np.percentile(flat, p))
    sample_val = max(1.0, sample_val)

    est_bits = int(math.ceil(math.log2(sample_val + 1.0)))
    est_bits = min(16, max(2, est_bits))

    allowed = (8, 10, 12, 14, 16)
    b = min(allowed, key=lambda k: abs(k - est_bits))
    white = (1 << b) - 1

    if vmax > white:
        return 16, 65535, False

    return b, white, False


def infer_group_bit_depth(
    channels: Dict[str, np.ndarray],
) -> Tuple[Optional[int], Optional[int], Optional[bool], list[str]]:
    """Infer a shared bit depth for RGB channel arrays.

    Parameters
    ----------
    channels : dict
        Mapping from channel name to image array.

    Returns
    -------
    bit_depth : int or None
        Inferred shared bit depth.
    white_level : int or None
        White level for the inferred bit depth.
    shifted : bool or None
        Whether all non-zero uint16 channels appear left-shifted.
    warnings : list of str
        Non-fatal warnings about ambiguous or mixed channel data.

    Notes
    -----
    No input values are modified. All-zero uint16 channels are ignored for bit
    depth decisions when at least one non-zero uint16 channel is present.
    """
    warnings: list[str] = []
    dtypes = {arr.dtype for arr in channels.values()}

    if all(np.issubdtype(dt, np.uint8) for dt in dtypes):
        return 8, 255, False, warnings

    if all(np.issubdtype(dt, np.uint16) for dt in dtypes):
        detected: list[tuple[int, int, bool, str]] = []

        for ch, arr in channels.items():
            if int(arr.max()) == 0:
                warnings.append(
                    f"Channel {ch!r} is all zero; bit depth inferred from other channels."
                )
                continue

            bit_depth, white, shifted = detect_bitdepth_u16(
                arr,
                use_percentile=False,
            )
            detected.append((bit_depth, white, shifted, ch))

        if not detected:
            warnings.append("All channels are zero uint16; assuming 16-bit storage.")
            return 16, 65535, False, warnings

        bit_depths = {x[0] for x in detected}
        whites = {x[1] for x in detected}
        shifted_flags = {x[2] for x in detected}

        if len(bit_depths) > 1:
            warnings.append(
                "Channels appear to have different nominal bit depths: "
                + ", ".join(f"{ch}={bd}" for bd, _, _, ch in detected)
                + ". Raw values were kept unchanged."
            )

        if len(shifted_flags) > 1:
            warnings.append(
                "Some uint16 channels look left-shifted and others do not. "
                "Raw values were kept unchanged."
            )

        bit_depth = max(bit_depths)
        white = max(whites)
        shifted = True if shifted_flags == {True} else False

        return bit_depth, white, shifted, warnings

    if all(np.issubdtype(dt, np.integer) for dt in dtypes):
        max_info = max(np.iinfo(dt).max for dt in dtypes)
        bit_depth = int(np.ceil(np.log2(max_info + 1)))

        warnings.append(
            f"Mixed integer dtypes {sorted(str(dt) for dt in dtypes)}; "
            "output will be cast to uint16."
        )

        return bit_depth, int(max_info), False, warnings

    warnings.append(
        f"Non-integer or mixed dtypes {sorted(str(dt) for dt in dtypes)}; "
        "bit depth is not well-defined."
    )

    return None, None, None, warnings


def common_output_dtype(channels: Dict[str, np.ndarray]) -> np.dtype:
    """Choose an output dtype for combined channel data.

    Parameters
    ----------
    channels : dict
        Mapping from channel name to image array.

    Returns
    -------
    numpy.dtype
        Output dtype. Returns ``uint8`` for all-uint8 input, ``uint16`` for
        integer input with any non-uint8 dtype, and ``float32`` otherwise.

    Notes
    -----
    This function selects a safe storage dtype without scaling the values.
    """
    dtypes = [arr.dtype for arr in channels.values()]

    if all(dt == np.uint8 for dt in dtypes):
        return np.dtype(np.uint8)

    if all(np.issubdtype(dt, np.integer) for dt in dtypes):
        return np.dtype(np.uint16)

    return np.dtype(np.float32)


def detect_channel_from_name(
    filename: str,
) -> Tuple[Optional[str], Optional[str], list[str]]:
    """Detect a standalone RGB channel token from a filename.

    Parameters
    ----------
    filename : str
        Input filename. Only the stem is inspected.

    Returns
    -------
    channel : str or None
        Detected channel, one of ``"r"``, ``"g"``, or ``"b"``. Returns
        ``None`` if no valid channel token is found.
    group_key : str or None
        Group key formed from the remaining filename tokens. Returns
        ``None`` if channel detection fails.
    warnings : list of str
        Warning messages explaining skipped or ambiguous filenames.

    Raises
    ------
    ValueError
        If ``CHANNEL_SEPARATORS`` is empty.

    Notes
    -----
    The channel token must be a standalone filename token separated by one of
    ``CHANNEL_SEPARATORS``. Names like ``sample_r.tif`` are accepted; names like
    ``sampleR.tif`` or ``sample_red.tif`` are rejected.
    """
    warnings: list[str] = []

    stem = Path(filename).stem
    lower = stem.strip().lower()

    if not CHANNEL_SEPARATORS:
        raise ValueError("CHANNEL_SEPARATORS must contain at least one separator.")

    sep_class = "[" + re.escape("".join(CHANNEL_SEPARATORS)) + "]"
    tokens = [t for t in re.split(sep_class + "+", lower) if t]

    hits: list[tuple[int, str]] = [
        (i, tok)
        for i, tok in enumerate(tokens)
        if tok in {"r", "g", "b"}
    ]

    if len(hits) == 0:
        warnings.append(
            f"No standalone r/g/b channel token found in filename {filename!r}; "
            f"allowed separators are {CHANNEL_SEPARATORS!r}; skipping."
        )
        return None, None, warnings

    if len(hits) > 1:
        warnings.append(
            f"Ambiguous r/g/b channel tokens in filename {filename!r}; "
            f"allowed separators are {CHANNEL_SEPARATORS!r}; skipping."
        )
        return None, None, warnings

    idx, ch = hits[0]
    remaining = tokens[:idx] + tokens[idx + 1:]
    group_key = "_".join(remaining) if remaining else "combined"

    return ch, group_key, warnings
