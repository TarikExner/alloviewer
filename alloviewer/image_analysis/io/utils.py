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
#
# Accepted:
#   sample_r.tif
#   sample-R.tif
#   sample.g.tif
#   sample b.tif
#   r_sample.tif
#   G-sample.tif
#   b.sample.tif
#
# Rejected:
#   sampleR.tif
#   Rsample.tif
#   sample_red.tif
#   sample_blue.tif
#   bright_sample.tif
CHANNEL_SEPARATORS: tuple[str, ...] = ("_", "-", ".", " ")


@dataclass
class LoadReport:
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
    """
    If p is absolute and exists, return it.

    Otherwise, treat p as a relative path under base_dir or DATA_ROOT.
    Backslashes are normalized first.
    """
    s = str(p).replace("\\", "/")
    candidate = Path(s)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    base = base_dir or DATA_ROOT
    return (base / candidate).resolve()

def ensure_rgb_anydepth(arr: np.ndarray, report: LoadReport) -> np.ndarray:
    """
    Ensure HxWx3 while keeping dtype unchanged.

    - Gray -> stacked to 3 channels
    - RGBA/BGRA -> alpha dropped
    - More than 3 channels -> first 3
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
    """
    Convert loaded image data to one raw monochrome channel.

    No scaling is done.
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
    """
    Detect nominal bit depth for a uint16 image.

    Returns:
      bit_depth : int
      white     : int
      shifted   : bool
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
    """
    Infer a shared bit depth for r/g/b channels.

    All-zero uint16 channels are ambiguous, so non-zero channels decide.
    No values are modified here.
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
    """
    Pick a safe output dtype without scaling.
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
    """
    Detect r/g/b only when it appears as a standalone filename token
    separated by one of CHANNEL_SEPARATORS.

    Returns:
      channel, group_key, warnings
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
