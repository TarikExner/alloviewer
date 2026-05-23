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
    common_output_dtype,
    detect_channel_from_name,
    infer_group_bit_depth,
    resolve_to_data_root,
)


@dataclass
class NamedImageSource:
    """Named image input used for channel grouping.

    Parameters
    ----------
    name : str
        Original filename or display name. Used for channel detection and
        output naming.
    source : SourceType
        Image source. Supported values are paths, bytes, bytearray objects, and
        file-like objects.
    """

    name: str
    source: SourceType


@dataclass
class CombinedRGBImage:
    """Combined in-memory RGB image.

    Parameters
    ----------
    array : numpy.ndarray
        RGB image in ``(H, W, 3)`` layout. Raw values are preserved.
    output_name : str
        Suggested output filename.
    report : ChannelCombineReport
        Metadata and warnings for the channel-combination step.
    """

    array: np.ndarray
    output_name: str
    report: ChannelCombineReport


def extract_declared_mono_channel(
    arr: np.ndarray,
    *,
    declared_channel: str,
    image_name: str,
    warnings: list[str],
) -> np.ndarray:
    """Extract a declared monochrome channel from an image array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input image. Supported shapes are ``(H, W)``, ``(H, W, 1)``,
        ``(H, W, 3)``, and ``(H, W, 4)``.
    declared_channel : {'r', 'g', 'b'}
        Channel declared by the filename.
    image_name : str
        Image name used in warning messages.
    warnings : list of str
        Warning list updated in place.

    Returns
    -------
    numpy.ndarray
        Two-dimensional channel image.

    Raises
    ------
    ValueError
        If ``declared_channel`` is invalid or the image shape is unsupported.

    Notes
    -----
    Three- and four-channel inputs are treated as OpenCV-native BGR or BGRA
    arrays, converted to RGB, and then reduced to the filename-declared channel.
    No scaling, clipping, bit shifting, or normalization is applied.
    """
    declared_channel = declared_channel.lower()

    if declared_channel not in {"r", "g", "b"}:
        raise ValueError(
            f"Invalid declared_channel {declared_channel!r}; expected 'r', 'g', or 'b'."
        )

    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported image shape {arr.shape} for {image_name!r}; "
            "expected HxW, HxWx1, HxWx3, or HxWx4."
        )

    n_channels = arr.shape[2]

    if n_channels == 1:
        return arr[..., 0]

    if n_channels not in (3, 4):
        raise ValueError(
            f"Unsupported {n_channels}-channel image for {image_name!r}; "
            "expected HxW, HxWx1, HxWx3, or HxWx4."
        )

    bgr = arr[..., :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    channel_index = {
        "r": 0,
        "g": 1,
        "b": 2,
    }[declared_channel]

    selected = rgb[..., channel_index]

    nonzero_counts = {
        "r": int(np.count_nonzero(rgb[..., 0])),
        "g": int(np.count_nonzero(rgb[..., 1])),
        "b": int(np.count_nonzero(rgb[..., 2])),
    }

    warnings.append(
        f"Image {image_name!r} has shape {arr.shape}. "
        f"Expected a monochrome image, but got {n_channels} channels. "
        f"Using fallback extraction because this RGB-combination function expects "
        f"single-channel information. Filename declares channel {declared_channel!r}; "
        f"using RGB channel index {channel_index} after BGR->RGB interpretation. "
        f"Non-zero pixel counts: {nonzero_counts}."
    )

    if n_channels == 4:
        warnings.append(
            f"Image {image_name!r} has an alpha channel. Alpha was ignored."
        )

    if nonzero_counts[declared_channel] == 0:
        warnings.append(
            f"Declared channel {declared_channel!r} in {image_name!r} contains only "
            "zero pixels after BGR->RGB interpretation."
        )

    ignored_nonzero_channels = [
        ch
        for ch in ("r", "g", "b")
        if ch != declared_channel and nonzero_counts[ch] > 0
    ]

    if ignored_nonzero_channels:
        warnings.append(
            f"Image {image_name!r} contains non-zero pixels in non-declared channel(s) "
            f"{ignored_nonzero_channels}. These channels were ignored."
        )

    return selected


def group_named_sources_by_rgb_channel(
    sources: Sequence[NamedImageSource],
    *,
    error_on_unmatched: bool = False,
) -> tuple[dict[str, dict[str, NamedImageSource]], list[str]]:
    """Group named sources by RGB channel and group key.

    Parameters
    ----------
    sources : sequence of NamedImageSource
        Image sources with filenames used for channel detection.
    error_on_unmatched : bool, optional
        If ``True``, raise an error when any source cannot be assigned to an
        RGB channel. The default is ``False``.

    Returns
    -------
    grouped : dict
        Nested mapping ``grouped[group_key][channel] = NamedImageSource``.
    unmatched_warnings : list of str
        Warnings for files that could not be assigned to a channel.

    Raises
    ------
    ValueError
        If duplicate channels are found in one group, or if unmatched files are
        present and ``error_on_unmatched`` is ``True``.
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
    """Combine named monochrome R/G/B sources into RGB images.

    Parameters
    ----------
    sources : sequence of NamedImageSource
        Named image sources. Filenames are used to assign files to ``r``,
        ``g``, and ``b`` channels.
    output_ext : str, optional
        Output filename extension. The default is ``".tif"``.
    max_mp : float or None, optional
        Maximum allowed image size in megapixels. Passed to :func:`open_image`.
        If ``None``, no size limit is applied.
    require_complete_rgb : bool, optional
        If ``True``, groups missing any RGB channel are skipped. If ``False``,
        missing channels are filled with zeros.
    error_on_unmatched : bool, optional
        If ``True``, raise an error for files that cannot be assigned to a
        channel.

    Returns
    -------
    list of CombinedRGBImage
        Combined RGB images and reports.

    Raises
    ------
    ValueError
        If no valid groups are found, no images are created, shapes mismatch
        within a group, or channel detection fails under strict settings.

    Notes
    -----
    No scaling, right shifting, or intensity normalization is applied. Output
    arrays are RGB with shape ``(H, W, 3)`` and preserve raw channel values.
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

            mono = extract_declared_mono_channel(
                arr,
                declared_channel=ch,
                image_name=item.name,
                warnings=warnings,
            )

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
    """Save an RGB image array to disk.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB image with shape ``(H, W, 3)``.
    output_path : str or pathlib.Path
        Output file path.
    overwrite : bool, optional
        If ``True``, replace an existing file. The default is ``False``.

    Returns
    -------
    pathlib.Path
        Resolved output path.

    Raises
    ------
    FileExistsError
        If the output file exists and ``overwrite`` is ``False``.
    ValueError
        If ``rgb`` is not shaped ``(H, W, 3)``.
    RuntimeError
        If OpenCV fails to write the image.

    Notes
    -----
    OpenCV writes color images in BGR order, so the input RGB image is converted
    to BGR before writing.
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
    """Encode an RGB image array to bytes.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB image with shape ``(H, W, 3)``.
    ext : str, optional
        Image extension passed to OpenCV, for example ``".tif"`` or ``".png"``.
        The default is ``".tif"``.

    Returns
    -------
    bytes
        Encoded image bytes.

    Raises
    ------
    ValueError
        If ``rgb`` is not shaped ``(H, W, 3)``.
    RuntimeError
        If OpenCV fails to encode the image.

    Notes
    -----
    OpenCV encodes color images in BGR order, so the input RGB image is
    converted to BGR before encoding.
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
    """Combine monochrome RGB channel files from a folder and save outputs.

    Parameters
    ----------
    folder : str or pathlib.Path
        Input folder containing channel images.
    output_folder : str or pathlib.Path
        Folder where combined RGB images are written.
    base_dir : str or pathlib.Path or None, optional
        Optional root directory used to resolve relative input and output
        folders.
    output_ext : str, optional
        Output image extension. The default is ``".tif"``.
    overwrite : bool, optional
        If ``True``, replace existing output files. The default is ``False``.
    max_mp : float or None, optional
        Maximum allowed image size in megapixels. Passed to :func:`open_image`.
    require_complete_rgb : bool, optional
        If ``True``, groups missing any RGB channel are skipped. If ``False``,
        missing channels are filled with zeros.
    error_on_unmatched : bool, optional
        If ``True``, raise an error for files that cannot be assigned to a
        channel.

    Returns
    -------
    list of ChannelCombineReport
        Reports for all written RGB images.

    Raises
    ------
    FileNotFoundError
        If the input folder does not exist.
    NotADirectoryError
        If ``folder`` is not a directory.
    FileExistsError
        If an output file exists and ``overwrite`` is ``False``.
    ValueError
        If no valid channel groups are found or no RGB images are created.
    RuntimeError
        If an image cannot be read, encoded, or written.

    Notes
    -----
    This function is intended for batch processing from disk. For web backends
    or in-memory workflows, use :func:`combine_mono_rgb_sources`.
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
