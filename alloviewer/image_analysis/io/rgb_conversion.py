from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .io import open_image
from .utils import (
    ChannelCombineReport,
    IMAGE_EXTS,
    SourceType,
    as_single_channel_raw,
    common_output_dtype,
    detect_channel_from_name,
    infer_group_bit_depth,
    resolve_to_data_root,
)


@dataclass
class NamedImageSource:
    """
    Backend-friendly image input.

    name:
        Original filename. Used for channel detection and output naming.

    source:
        Path, bytes, bytearray, or file-like object.
        FastAPI UploadFile.file works here.
    """
    name: str
    source: SourceType


@dataclass
class CombinedRGBImage:
    """
    In-memory combined RGB image.

    array:
        RGB image in HxWx3 layout.
        Raw values are preserved. No scaling is done.

    output_name:
        Suggested output filename.

    report:
        Metadata and warnings.
    """
    array: np.ndarray
    output_name: str
    report: ChannelCombineReport

def group_named_sources_by_rgb_channel(
    sources: Sequence[NamedImageSource],
    *,
    error_on_unmatched: bool = False,
) -> tuple[dict[str, dict[str, NamedImageSource]], list[str]]:
    """
    Group named image sources into RGB channel groups.

    Returns:
      grouped[group_key][channel] = NamedImageSource
      unmatched_warnings
    """
    grouped: dict[str, dict[str, NamedImageSource]] = defaultdict(dict)
    unmatched_warnings: list[str] = []

    for item in sources:
        ch, group_key, warnings = detect_channel_from_name(item.name)

        if ch is None or group_key is None:
            unmatched_warnings.extend([f"{item.name}: {w}" for w in warnings])
            continue

        if ch in grouped[group_key]:
            raise ValueError(
                f"Duplicate {ch!r} channel for group {group_key!r}: "
                f"{grouped[group_key][ch].name!r} and {item.name!r}"
            )

        grouped[group_key][ch] = item

    if unmatched_warnings and error_on_unmatched:
        raise ValueError(
            "Some image files could not be assigned to r/g/b channels:\n"
            + "\n".join(unmatched_warnings)
        )

    return grouped, unmatched_warnings


def combine_mono_rgb_sources(
    sources: Sequence[NamedImageSource],
    *,
    output_ext: str = ".tif",
    max_mp: Optional[float] = 200.0,
    require_complete_rgb: bool = True,
    error_on_unmatched: bool = False,
) -> list[CombinedRGBImage]:
    """
    Combine named monochrome r/g/b image sources into RGB images.

    This is the main backend-friendly function.

    It accepts:
      - disk paths
      - bytes
      - bytearray
      - file-like objects
      - FastAPI UploadFile.file via:
        NamedImageSource(name=file.filename, source=file.file)

    No scaling is done.
    No right-shifting is done.
    No intensity normalization is done.

    The returned image arrays are RGB, HxWx3, and keep raw values.
    """
    if not output_ext.startswith("."):
        output_ext = "." + output_ext

    grouped, _ = group_named_sources_by_rgb_channel(
        sources,
        error_on_unmatched=error_on_unmatched,
    )

    if not grouped:
        raise ValueError("No valid r/g/b image groups found.")

    combined_images: list[CombinedRGBImage] = []

    for group_key, channel_items in grouped.items():
        warnings: list[str] = []

        missing = [ch for ch in ("r", "g", "b") if ch not in channel_items]

        if missing and require_complete_rgb:
            warnings.append(
                f"Skipping group {group_key!r}; missing channels: {missing}."
            )
            continue

        channels: dict[str, np.ndarray] = {}
        input_paths: dict[str, str] = {}
        reference_shape: Optional[Tuple[int, int]] = None

        for ch in ("r", "g", "b"):
            if ch not in channel_items:
                continue

            item = channel_items[ch]

            arr, _ = open_image(
                item.source,
                max_mp=max_mp,
            )

            mono = as_single_channel_raw(arr, item.name)

            if reference_shape is None:
                reference_shape = mono.shape
            elif mono.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch in group {group_key!r}: "
                    f"channel {ch!r} has shape {mono.shape}, "
                    f"expected {reference_shape}."
                )

            channels[ch] = mono
            input_paths[ch] = item.name

        if reference_shape is None:
            continue

        for ch in ("r", "g", "b"):
            if ch not in channels:
                ref = next(iter(channels.values()))
                channels[ch] = np.zeros(reference_shape, dtype=ref.dtype)
                warnings.append(
                    f"Channel {ch!r} missing in group {group_key!r}; filled with zeros."
                )

        bit_depth, white_level, shifted, bit_warnings = infer_group_bit_depth(channels)
        warnings.extend(bit_warnings)

        out_dtype = common_output_dtype(channels)

        rgb = np.stack(
            [
                channels["r"].astype(out_dtype, copy=False),
                channels["g"].astype(out_dtype, copy=False),
                channels["b"].astype(out_dtype, copy=False),
            ],
            axis=-1,
        )

        output_name = f"{group_key}{output_ext}"

        report = ChannelCombineReport(
            group_key=group_key,
            output_path=output_name,
            input_paths=input_paths,
            shape=rgb.shape,
            dtype=str(rgb.dtype),
            bit_depth=bit_depth,
            white_level=white_level,
            shifted=shifted,
            warnings=warnings,
        )

        combined_images.append(
            CombinedRGBImage(
                array=rgb,
                output_name=output_name,
                report=report,
            )
        )

    if not combined_images:
        raise ValueError("No RGB images were created from the provided sources.")

    return combined_images


def save_rgb_image(
    rgb: np.ndarray,
    output_path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """
    Save an RGB image array to disk.

    OpenCV writes color images as BGR, so RGB -> BGR conversion is done here.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}.")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(str(output_path), bgr)

    if not ok:
        raise RuntimeError(f"Failed to write RGB image: {output_path}")

    return output_path


def encode_rgb_image(
    rgb: np.ndarray,
    *,
    ext: str = ".tif",
) -> bytes:
    """
    Encode an RGB image array to bytes.

    Useful for FastAPI responses or object storage.
    """
    if not ext.startswith("."):
        ext = "." + ext

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}.")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode(ext, bgr)

    if not ok:
        raise RuntimeError(f"Failed to encode RGB image as {ext!r}.")

    return encoded.tobytes()

def combine_mono_rgb_folder(
    folder: Union[str, Path],
    output_folder: Union[str, Path],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    output_ext: str = ".tif",
    overwrite: bool = False,
    max_mp: Optional[float] = 200.0,
    require_complete_rgb: bool = True,
    error_on_unmatched: bool = False,
) -> list[ChannelCombineReport]:
    """
    Folder-based wrapper around combine_mono_rgb_sources().

    Use this for batch processing from disk.

    For FastAPI, prefer combine_mono_rgb_sources().
    """
    base = Path(base_dir).resolve() if base_dir else None

    in_dir = resolve_to_data_root(folder, base)
    out_dir = resolve_to_data_root(output_folder, base)

    if not in_dir.exists():
        raise FileNotFoundError(str(in_dir))

    if not in_dir.is_dir():
        raise NotADirectoryError(str(in_dir))

    out_dir.mkdir(parents=True, exist_ok=True)

    sources: list[NamedImageSource] = []

    for path in sorted(in_dir.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        sources.append(
            NamedImageSource(
                name=path.name,
                source=path,
            )
        )

    combined = combine_mono_rgb_sources(
        sources,
        output_ext=output_ext,
        max_mp=max_mp,
        require_complete_rgb=require_complete_rgb,
        error_on_unmatched=error_on_unmatched,
    )

    reports: list[ChannelCombineReport] = []

    for item in combined:
        out_path = out_dir / item.output_name

        saved_path = save_rgb_image(
            item.array,
            out_path,
            overwrite=overwrite,
        )

        item.report.output_path = str(saved_path)
        reports.append(item.report)

    return reports
