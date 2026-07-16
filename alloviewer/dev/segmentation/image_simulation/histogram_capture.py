
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

from alloviewer.image_analysis.io import load_image


EXT_IMAGES_FOLDERS = [
    "./ext_images/20251106_25065441_iPhone_XR_JPEG",
    "./ext_images/20251106_25722169_iPhone_XR_JPEG",
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251107_25065521_GooglePixel",
    "./ext_images/20251107_25722332_GooglePixel",
    "./ext_images/20251014_25719960",
    "./ext_images/20251014_25720084",
    "./ext_images/20251107_25065521",
    "./ext_images/20251107_25722332",
    "./ext_images/20260504_AM1_mono_rgb",
    "./ext_images/20260504_Auto1_mono_rgb",
    "./ext_images/20260507_XM1_+DTT_mono_rgb",
    "./ext_images/20260507_XM1_mono_rgb",
]

NEW_EXT_IMAGES_FOLDERS = [
    './ext_images/RUN_E3D485B679B8',
    './ext_images/RUN_0E41007D6AD1',
    './ext_images/RUN_9F17E3445802',
    './ext_images/RUN_A0BA087114AF',
    './ext_images/RUN_EF4293E06F29',
    './ext_images/RUN_E44B44BD58D3',
    './ext_images/RUN_17A7AE91A22F',
    './ext_images/RUN_609C1D30150F',
    './ext_images/RUN_3CE14774E520',
    './ext_images/RUN_FCEC3886DD87',
    './ext_images/RUN_2B97E768DF92',
    './ext_images/RUN_C7AAF95B2EF9',
    './ext_images/RUN_7C1C97246793'
]

EXT_IMAGES_FOLDER_MAPPING = dict(
    zip(EXT_IMAGES_FOLDERS, NEW_EXT_IMAGES_FOLDERS, strict=True)
)

MONOCHROME_IMAGES_FOLDERS_INPUT = [
    "./ext_images/20260504_AM1_mono",
    "./ext_images/20260504_Auto1_mono",
    "./ext_images/20260507_XM1_+DTT_mono",
    "./ext_images/20260507_XM1_mono",
]

MONOCHROME_IMAGES_FOLDERS_OUTPUT = [
    "./ext_images/20260504_AM1_mono_rgb",
    "./ext_images/20260504_Auto1_mono_rgb",
    "./ext_images/20260507_XM1_+DTT_mono_rgb",
    "./ext_images/20260507_XM1_mono_rgb",
]

STYLE_CACHE_PATH = Path("./results/style_cache.cache")
STYLE_QUANTILE_CACHE_PATH = STYLE_CACHE_PATH.with_name(
    "camera_quantile_band_cache.pkl"
)
DEFAULT_ANNOTATION_DIR = Path("./region_annotations")

LABEL_IGNORE = np.uint8(0)
LABEL_BACKGROUND = np.uint8(1)
LABEL_FOREGROUND = np.uint8(2)
LABEL_OUTSIDE_WELL = np.uint8(3)

REGION_NAMES = (
    "all", # "all" means all reviewed foreground and background pixels inside the well.
    "foreground",
    "background",
    "local_background",
    "far_background",
)

FOREGROUND_QUANTILES = {
    "iphone": 0.92,
    "googlepixel": 0.92,
    "microscope": 0.96,
    "monochrome_generic": 0.96,
    "monochrome_real": 0.96,
    "default": 0.92,
}

BACKGROUND_QUANTILES = {
    "iphone": 0.80,
    "googlepixel": 0.80,
    "microscope": 0.80,
    "monochrome_generic": 0.80,
    "monochrome_real": 0.80,
    "default": 0.80,
}

PAIRED_CACHE_VERSION = 3
PAIRED_CACHE_TYPE = "manual_annotations_per_image"
PAIRED_REFERENCE_SAMPLING = "paired_per_image"


def _find_device_label(file_path: str | Path) -> str:
    text = str(file_path).lower()

    if "mono_rgb" in text or "mono_real" in text:
        return "monochrome_real"

    if "iphone" in text:
        return "iphone"

    if "googlepixel" in text or "pixel" in text:
        return "googlepixel"

    return "microscope"


def _collect_image_paths(
    folders: Optional[Sequence[str | Path]] = None,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    recursive: bool = True,
) -> list[Path]:
    if folders is None:
        folders = EXT_IMAGES_FOLDERS

    exts = tuple(ext.lower() for ext in exts)
    image_paths: list[Path] = []

    for folder in dict.fromkeys(Path(folder) for folder in folders):
        if not folder.exists():
            continue

        walker = folder.rglob("*") if recursive else folder.glob("*")

        for path in walker:
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)

    image_paths = sorted(dict.fromkeys(image_paths))

    if not image_paths:
        raise RuntimeError("No image files were found.")

    return image_paths


def _as_hwc_rgb_float01(
    img: np.ndarray,
    preserve_copy: bool = False,
) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(img, dtype=np.float32)
    input_chw = False

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {arr.shape}.")

    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
        input_chw = True
    elif arr.shape[-1] != 3:
        raise ValueError(
            f"Expected CHW or HWC RGB image, got shape {img.shape}."
        )

    if preserve_copy:
        arr = arr.copy()

    return np.clip(arr, 0.0, 1.0), input_chw


def _annotation_paths_for_image(
    image_path: str | Path,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
) -> Dict[str, Path]:
    """
    Return annotation paths using the reviewer's output layout.

    Example
    -------
    Source:
        ext_images/folder_a/IMG_1.jpeg

    Annotation:
        region_annotations/folder_a/IMG_1_regions.npy
    """
    image_path = Path(image_path)
    folder = Path(annotation_dir) / image_path.parent.name
    stem = image_path.stem

    return {
        "regions": folder / f"{stem}_regions.npy",
        "well_mask": folder / f"{stem}_well_mask.npy",
        "metadata": folder / f"{stem}_regions.json",
        "preview": folder / f"{stem}_preview.png",
    }


def _annotation_is_complete(
    image_path: str | Path,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
) -> bool:
    paths = _annotation_paths_for_image(image_path, annotation_dir)

    return (
        paths["regions"].is_file()
        and paths["well_mask"].is_file()
        and paths["metadata"].is_file()
    )


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _annotation_inventory(
    image_paths: Sequence[Path],
    annotation_dir: str | Path,
) -> Dict[str, Any]:
    complete = []
    latest_mtime = 0.0

    for image_path in image_paths:
        paths = _annotation_paths_for_image(image_path, annotation_dir)

        if not _annotation_is_complete(image_path, annotation_dir):
            continue

        complete.append(str(image_path))

        for key in ("regions", "well_mask", "metadata"):
            try:
                latest_mtime = max(
                    latest_mtime,
                    paths[key].stat().st_mtime,
                )
            except OSError:
                pass

    return {
        "n_complete": len(complete),
        "complete_images": complete,
        "latest_mtime": float(latest_mtime),
    }



def _resolve_device_value(
    value: Any,
    device: str,
    default: Any,
) -> Any:
    """Compatibility helper retained from the automatic-mask cache."""
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(
            device,
            value.get("default", default),
        )

    return value


def make_foreground_background_masks(
    img_hwc: np.ndarray,
    foreground_quantile: float = 0.92,
    background_quantile: float = 0.80,
    min_region_pixels: int = 2048,
    signal_mode: str = "max",
    morph_open: bool = False,
    morph_kernel_size: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Legacy automatic mask helper.

    The reviewed annotation cache does not call this function, but it remains
    available for notebooks and older code that import it directly.
    """
    image = np.clip(
        np.asarray(img_hwc, dtype=np.float32),
        0.0,
        1.0,
    )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected HWC RGB image, got {image.shape}."
        )

    if not 0.0 <= foreground_quantile <= 1.0:
        raise ValueError(
            "foreground_quantile must be in [0, 1]."
        )

    if not 0.0 <= background_quantile <= 1.0:
        raise ValueError(
            "background_quantile must be in [0, 1]."
        )

    if background_quantile >= foreground_quantile:
        raise ValueError(
            "background_quantile must be smaller than "
            "foreground_quantile."
        )

    if signal_mode == "max":
        signal = image.max(axis=-1)
    elif signal_mode == "luma":
        signal = (
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        )
    elif signal_mode == "green":
        signal = image[..., 1]
    elif signal_mode == "red_green_max":
        signal = np.maximum(
            image[..., 0],
            image[..., 1],
        )
    else:
        raise ValueError(
            "signal_mode must be one of: "
            "'max', 'luma', 'green', 'red_green_max'."
        )

    finite = np.isfinite(signal)
    all_mask = finite.copy()

    if not finite.any():
        empty = np.zeros(signal.shape, dtype=bool)
        return {
            "all": empty,
            "foreground": empty,
            "background": empty,
        }

    values = signal[finite]
    foreground_threshold = float(
        np.quantile(values, foreground_quantile)
    )
    background_threshold = float(
        np.quantile(values, background_quantile)
    )

    foreground = finite & (
        signal >= foreground_threshold
    )
    background = finite & (
        signal <= background_threshold
    )

    if morph_open and foreground.any():
        kernel_size = max(1, int(morph_kernel_size))

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )
        foreground = cv2.morphologyEx(
            foreground.astype(np.uint8),
            cv2.MORPH_OPEN,
            kernel,
        ).astype(bool)

    if int(foreground.sum()) < int(min_region_pixels):
        foreground = np.zeros_like(
            foreground,
            dtype=bool,
        )

    if int(background.sum()) < int(min_region_pixels):
        background = np.zeros_like(
            background,
            dtype=bool,
        )

    return {
        "all": all_mask,
        "foreground": foreground,
        "background": background,
    }



def _load_reviewed_regions(
    image_path: Path,
    annotation_dir: str | Path,
    *,
    default_ring_inner_radius: int = 2,
    default_ring_outer_radius: int = 6,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load one reviewed mask pair and derive the analysis regions.

    Main labels
    -----------
    0: ignore
    1: valid background
    2: foreground
    3: outside well

    Derived regions
    ---------------
    all:
        Reviewed foreground and background pixels inside the well.

    local_background:
        Reviewed background between the inner and outer dilation radii.

    far_background:
        Reviewed background beyond the outer dilation radius.
    """
    annotation_paths = _annotation_paths_for_image(
        image_path,
        annotation_dir,
    )

    labels = np.load(annotation_paths["regions"]).astype(
        np.uint8,
        copy=False,
    )
    well_mask = np.load(annotation_paths["well_mask"]).astype(
        bool,
        copy=False,
    )
    metadata = _load_json(annotation_paths["metadata"])

    if labels.ndim != 2:
        raise ValueError(
            f"Expected a 2D region mask for {image_path}, got {labels.shape}."
        )

    if well_mask.shape != labels.shape:
        raise ValueError(
            f"Well-mask shape {well_mask.shape} does not match "
            f"region-mask shape {labels.shape} for {image_path}."
        )

    inside = well_mask & (labels != LABEL_OUTSIDE_WELL)
    foreground = inside & (labels == LABEL_FOREGROUND)
    background = inside & (labels == LABEL_BACKGROUND)
    valid_all = foreground | background

    ring_settings = metadata.get("ring_settings", {})
    inner_radius = max(
        0,
        int(
            ring_settings.get(
                "inner_radius",
                default_ring_inner_radius,
            )
        ),
    )
    outer_radius = max(
        inner_radius + 1,
        int(
            ring_settings.get(
                "outer_radius",
                default_ring_outer_radius,
            )
        ),
    )

    if foreground.any():
        inner = ndi.binary_dilation(
            foreground,
            iterations=inner_radius,
        )
        outer = ndi.binary_dilation(
            foreground,
            iterations=outer_radius,
        )
    else:
        inner = np.zeros_like(foreground)
        outer = np.zeros_like(foreground)

    local_background = outer & ~inner & background
    far_background = background & ~outer

    masks = {
        "all": valid_all,
        "foreground": foreground,
        "background": background,
        "local_background": local_background,
        "far_background": far_background,
    }

    info = {
        "annotation_paths": {
            key: str(value)
            for key, value in annotation_paths.items()
        },
        "ring_inner_radius": int(inner_radius),
        "ring_outer_radius": int(outer_radius),
        "metadata": metadata,
    }

    return masks, info

def _sample_pixels_from_masked_region(
    img_hwc: np.ndarray,
    mask_hw: Optional[np.ndarray],
    sample_pixels: Optional[int],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    img = np.clip(
        np.asarray(img_hwc, dtype=np.float32),
        0.0,
        1.0,
    )

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {img.shape}.")

    if mask_hw is None:
        pixels = img.reshape(-1, 3)
    else:
        mask = np.asarray(mask_hw, dtype=bool)

        if mask.shape != img.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"image shape {img.shape[:2]}."
            )

        if not mask.any():
            return None

        pixels = img[mask]

    n_total = int(pixels.shape[0])

    if n_total == 0:
        return None

    if sample_pixels is not None and n_total > int(sample_pixels):
        indices = rng.choice(
            n_total,
            size=int(sample_pixels),
            replace=False,
        )
        pixels = pixels[indices]

    return pixels.astype(np.float32, copy=False)


def _compute_pixels_quantiles(
    pixels: np.ndarray,
    q_probs: np.ndarray,
) -> np.ndarray:
    channel_quantiles = []

    for channel in range(3):
        values = np.clip(pixels[:, channel], 0.0, 1.0)

        quantiles = np.quantile(
            values,
            q_probs,
        ).astype(np.float32)

        channel_quantiles.append(
            np.maximum.accumulate(quantiles)
        )

    return np.stack(channel_quantiles, axis=0).astype(np.float32)


def _build_region_payload_from_records(
    records: Sequence[Dict[str, Any]],
    region: str,
    *,
    q_band_lo: float,
    q_band_hi: float,
    store_q_images: bool,
) -> Optional[Dict[str, Any]]:
    quantiles = []
    image_ids = []

    for record in records:
        region_payload = record.get("regions", {}).get(region)

        if region_payload is None:
            continue

        quantiles.append(
            np.asarray(region_payload["q"], dtype=np.float32)
        )
        image_ids.append(str(record["id"]))

    if not quantiles:
        return None

    quantile_stack = np.stack(quantiles, axis=0).astype(np.float32)

    q_center = np.quantile(
        quantile_stack,
        0.50,
        axis=0,
    ).astype(np.float32)
    q_lo = np.quantile(
        quantile_stack,
        q_band_lo,
        axis=0,
    ).astype(np.float32)
    q_hi = np.quantile(
        quantile_stack,
        q_band_hi,
        axis=0,
    ).astype(np.float32)

    q_center = np.maximum.accumulate(q_center, axis=1)
    q_lo = np.maximum.accumulate(q_lo, axis=1)
    q_hi = np.maximum.accumulate(q_hi, axis=1)
    q_center = np.clip(q_center, q_lo, q_hi)

    payload: Dict[str, Any] = {
        "n_images": int(quantile_stack.shape[0]),
        "image_ids": tuple(image_ids),
        "q_center": q_center,
        "q_lo": q_lo,
        "q_hi": q_hi,
    }

    if store_q_images:
        payload["q_images"] = quantile_stack

    return payload


def _region_minimum_pixels(
    region: str,
    *,
    min_region_pixels: int,
    min_local_background_pixels: int,
) -> int:
    if region == "local_background":
        return int(min_local_background_pixels)

    return int(min_region_pixels)


def build_annotated_quantile_cache(
    folders: Optional[Sequence[str | Path]] = None,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    devices: Sequence[str] = (
        "iphone",
        "googlepixel",
        "microscope",
        "monochrome_generic",
        "monochrome_real",
    ),
    recursive: bool = True,
    sample_pixels_per_region: Optional[int] = 100_000,
    n_quantiles: int = 1024,
    q_band_lo: float = 0.025,
    q_band_hi: float = 0.975,
    rng_seed: int = 0,
    cache_path: Optional[str | Path] = STYLE_QUANTILE_CACHE_PATH,
    force_recompute: bool = False,
    min_region_pixels: int = 2048,
    min_local_background_pixels: int = 512,
    require_foreground_background: bool = True,
    store_q_images: bool = True,
    default_ring_inner_radius: int = 2,
    default_ring_outer_radius: int = 6,
) -> Dict[str, Any]:
    """
    Build a paired per-image quantile cache from reviewed annotations.

    Each real image contributes one record containing all available RGB region
    curves. During augmentation, one complete record is sampled, so foreground,
    background, and all three channels always come from the same real image.

    Pixels marked ignore or outside-well are never used.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)

        if cache_path.exists() and not force_recompute:
            with cache_path.open("rb") as handle:
                cache = pickle.load(handle)

            if _is_paired_annotation_cache(cache):
                return cache

    annotation_dir = Path(annotation_dir)

    if not annotation_dir.exists():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {annotation_dir}"
        )

    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")

    if not (0.0 <= q_band_lo < q_band_hi <= 1.0):
        raise ValueError(
            "Require 0 <= q_band_lo < q_band_hi <= 1."
        )

    image_paths = _collect_image_paths(
        folders=folders,
        exts=exts,
        recursive=recursive,
    )

    q_probs = np.linspace(
        0.0,
        1.0,
        int(n_quantiles),
        dtype=np.float32,
    )
    rng = np.random.default_rng(rng_seed)

    records_by_device: Dict[str, list[Dict[str, Any]]] = {
        device: []
        for device in devices
    }

    skipped: list[Dict[str, str]] = []
    n_missing_annotations = 0

    for image_path in tqdm(
        image_paths,
        desc="Building paired quantile cache",
        unit="image",
    ):
        device = _find_device_label(image_path)

        if device not in records_by_device:
            continue

        if not _annotation_is_complete(image_path, annotation_dir):
            n_missing_annotations += 1
            continue

        try:
            image, report = load_image(
                image_path.name,
                base_dir=image_path.parent,
                as_chw=False,
                scale=True,
                fast_scale=True,
            )
        except Exception as error:
            skipped.append({
                "path": str(image_path),
                "reason": f"image load failed: {error!r}",
            })
            continue

        image = np.clip(
            np.asarray(image, dtype=np.float32),
            0.0,
            1.0,
        )

        if image.ndim != 3 or image.shape[2] != 3:
            skipped.append({
                "path": str(image_path),
                "reason": f"bad image shape {image.shape}",
            })
            continue

        try:
            masks, annotation_info = _load_reviewed_regions(
                image_path,
                annotation_dir,
                default_ring_inner_radius=default_ring_inner_radius,
                default_ring_outer_radius=default_ring_outer_radius,
            )
        except Exception as error:
            skipped.append({
                "path": str(image_path),
                "reason": f"annotation load failed: {error!r}",
            })
            continue

        if masks["all"].shape != image.shape[:2]:
            skipped.append({
                "path": str(image_path),
                "reason": (
                    f"annotation shape {masks['all'].shape} does not match "
                    f"image shape {image.shape[:2]}"
                ),
            })
            continue

        region_payloads: Dict[str, Dict[str, Any]] = {}

        for region in REGION_NAMES:
            mask = masks[region]
            n_pixels = int(mask.sum())
            minimum = _region_minimum_pixels(
                region,
                min_region_pixels=min_region_pixels,
                min_local_background_pixels=min_local_background_pixels,
            )

            if n_pixels < minimum:
                continue

            pixels = _sample_pixels_from_masked_region(
                image,
                mask,
                sample_pixels_per_region,
                rng,
            )

            if pixels is None or pixels.shape[0] == 0:
                continue

            region_payloads[region] = {
                "q": _compute_pixels_quantiles(
                    pixels,
                    q_probs,
                ),
                "n_pixels": n_pixels,
                "n_sampled_pixels": int(pixels.shape[0]),
            }

        if "all" not in region_payloads:
            skipped.append({
                "path": str(image_path),
                "reason": "too few reviewed pixels for region 'all'",
            })
            continue

        if require_foreground_background and (
            "foreground" not in region_payloads
            or "background" not in region_payloads
        ):
            skipped.append({
                "path": str(image_path),
                "reason": (
                    "paired foreground/background regions are missing "
                    "or below the minimum pixel count"
                ),
            })
            continue

        image_id = f"{image_path.parent.name}/{image_path.name}"

        record = {
            "id": image_id,
            "path": str(image_path),
            "source_folder": image_path.parent.name,
            "file_name": image_path.name,
            "device": device,
            "image_shape": tuple(int(v) for v in image.shape),
            "regions": region_payloads,
            "ring_settings": {
                "inner_radius": int(
                    annotation_info["ring_inner_radius"]
                ),
                "outer_radius": int(
                    annotation_info["ring_outer_radius"]
                ),
            },
            "annotation_paths": annotation_info["annotation_paths"],
            "loader_report": {
                "path": getattr(report, "path", str(image_path)),
                "dtype": getattr(report, "dtype", str(image.dtype)),
                "bit_depth": getattr(report, "bit_depth", None),
                "white_level": getattr(report, "white_level", None),
                "warnings": list(
                    getattr(report, "warnings", [])
                ),
            },
        }

        records_by_device[device].append(record)

    devices_payload: Dict[str, Any] = {}

    for device in devices:
        records = records_by_device[device]

        if not records:
            continue

        aggregate_regions: Dict[str, Any] = {}

        for region in REGION_NAMES:
            payload = _build_region_payload_from_records(
                records,
                region,
                q_band_lo=q_band_lo,
                q_band_hi=q_band_hi,
                store_q_images=store_q_images,
            )

            if payload is not None:
                aggregate_regions[region] = payload

        if "all" not in aggregate_regions:
            continue

        device_payload: Dict[str, Any] = {
            "n_images": int(len(records)),
            "images": records,
            "regions": aggregate_regions,
        }

        # Backward-compatible all-region aggregate fields.
        all_payload = aggregate_regions["all"]
        device_payload["q_center"] = all_payload["q_center"]
        device_payload["q_lo"] = all_payload["q_lo"]
        device_payload["q_hi"] = all_payload["q_hi"]

        if "q_images" in all_payload:
            device_payload["q_images"] = all_payload["q_images"]

        devices_payload[device] = device_payload

    inventory = _annotation_inventory(
        image_paths,
        annotation_dir,
    )

    cache: Dict[str, Any] = {
        "version": PAIRED_CACHE_VERSION,
        "cache_type": PAIRED_CACHE_TYPE,
        "reference_sampling": PAIRED_REFERENCE_SAMPLING,
        "devices": devices_payload,
        "q_probs": q_probs,
        "regions": REGION_NAMES,
        "devices_requested": tuple(devices),
        "folders": [
            str(folder)
            for folder in (
                folders
                if folders is not None
                else EXT_IMAGES_FOLDERS
            )
        ],
        "annotation_dir": str(annotation_dir),
        "annotation_inventory": inventory,
        "sample_pixels_per_region": sample_pixels_per_region,
        "n_quantiles": int(n_quantiles),
        "q_band_lo": float(q_band_lo),
        "q_band_hi": float(q_band_hi),
        "min_region_pixels": int(min_region_pixels),
        "min_local_background_pixels": int(
            min_local_background_pixels
        ),
        "require_foreground_background": bool(
            require_foreground_background
        ),
        "store_q_images": bool(store_q_images),
        "default_ring_inner_radius": int(
            default_ring_inner_radius
        ),
        "default_ring_outer_radius": int(
            default_ring_outer_radius
        ),
        "n_source_images": int(len(image_paths)),
        "n_missing_annotations": int(n_missing_annotations),
        "skipped": skipped,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with cache_path.open("wb") as handle:
            pickle.dump(
                cache,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    return cache


def _is_paired_annotation_cache(cache: Any) -> bool:
    return (
        isinstance(cache, dict)
        and int(cache.get("version", 0)) >= PAIRED_CACHE_VERSION
        and cache.get("cache_type") == PAIRED_CACHE_TYPE
        and cache.get("reference_sampling")
        == PAIRED_REFERENCE_SAMPLING
        and isinstance(cache.get("devices"), dict)
        and "q_probs" in cache
    )


def _cache_needs_rebuild(
    cache: Dict[str, Any],
    *,
    cache_path: Path,
    image_paths: Sequence[Path],
    annotation_dir: str | Path,
    rebuild_if_annotations_newer: bool,
) -> bool:
    if not _is_paired_annotation_cache(cache):
        return True

    if not rebuild_if_annotations_newer:
        return False

    inventory = _annotation_inventory(
        image_paths,
        annotation_dir,
    )
    cached_inventory = cache.get("annotation_inventory", {})

    if int(inventory["n_complete"]) != int(
        cached_inventory.get("n_complete", -1)
    ):
        return True

    if tuple(inventory["complete_images"]) != tuple(
        cached_inventory.get("complete_images", ())
    ):
        return True

    cached_annotation_dir = cache.get("annotation_dir")
    if cached_annotation_dir is not None and Path(
        cached_annotation_dir
    ) != Path(annotation_dir):
        return True

    try:
        cache_mtime = cache_path.stat().st_mtime
    except OSError:
        return True

    return float(inventory["latest_mtime"]) > float(cache_mtime)


def load_or_build_quantile_band_cache(
    folders: Optional[Sequence[str | Path]] = None,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    cache_path: str | Path = STYLE_QUANTILE_CACHE_PATH,
    force_recompute: bool = False,
    rebuild_if_annotations_newer: bool = False,
    sample_pixels_per_image: Optional[int] = 100_000,
    n_quantiles: int = 1024,
    q_band_lo: float = 0.025,
    q_band_hi: float = 0.975,
    min_region_pixels: int = 2048,
    min_local_background_pixels: int = 512,
    require_foreground_background: bool = True,
    store_q_images: bool = True,
    default_ring_inner_radius: int = 2,
    default_ring_outer_radius: int = 6,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    """
    Load or build the manually reviewed, paired per-image quantile cache.

    Existing version-2 population caches are rejected and rebuilt. The cache is
    also rebuilt when new or changed annotation files are newer than the cache.

    ``sample_pixels_per_image`` is retained for caller compatibility; it is used
    as the per-region sampling limit.
    """
    del legacy_kwargs

    cache_path = Path(cache_path)

    image_paths = _collect_image_paths(
        folders=folders,
    )

    if cache_path.exists() and not force_recompute:
        try:
            with cache_path.open("rb") as handle:
                cache = pickle.load(handle)
        except Exception:
            cache = None

        if (
            cache is not None
            and not _cache_needs_rebuild(
                cache,
                cache_path=cache_path,
                image_paths=image_paths,
                annotation_dir=annotation_dir,
                rebuild_if_annotations_newer=(
                    rebuild_if_annotations_newer
                ),
            )
        ):
            return cache

    return build_annotated_quantile_cache(
        folders=folders,
        annotation_dir=annotation_dir,
        sample_pixels_per_region=sample_pixels_per_image,
        n_quantiles=n_quantiles,
        q_band_lo=q_band_lo,
        q_band_hi=q_band_hi,
        cache_path=cache_path,
        force_recompute=True,
        min_region_pixels=min_region_pixels,
        min_local_background_pixels=(
            min_local_background_pixels
        ),
        require_foreground_background=(
            require_foreground_background
        ),
        store_q_images=store_q_images,
        default_ring_inner_radius=(
            default_ring_inner_radius
        ),
        default_ring_outer_radius=(
            default_ring_outer_radius
        ),
    )


# Alias for code that uses the shorter name.
def load_or_build_quantile_cache(
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    return load_or_build_quantile_band_cache(*args, **kwargs)


# Existing public builder name now builds the reviewed paired cache.
def build_target_quantile_band_cache(
    folders: Optional[Sequence[str | Path]] = None,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for the old public builder name.

    Old automatic-mask-only arguments are accepted and ignored because the new
    cache is built from reviewed masks.
    """
    if (
        "sample_pixels_per_image" in kwargs
        and "sample_pixels_per_region" not in kwargs
    ):
        kwargs["sample_pixels_per_region"] = kwargs.pop(
            "sample_pixels_per_image"
        )

    for obsolete_name in (
        "use_regions",
        "foreground_quantile",
        "background_quantile",
        "foreground_signal_mode",
    ):
        kwargs.pop(obsolete_name, None)

    return build_annotated_quantile_cache(
        folders=folders,
        annotation_dir=annotation_dir,
        **kwargs,
    )


def _compute_image_channel_quantiles(
    channel: np.ndarray,
    q_probs: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    channel = np.clip(
        np.asarray(channel, dtype=np.float32),
        0.0,
        1.0,
    )

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != channel.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"channel shape {channel.shape}."
            )

        if not mask.any():
            raise ValueError(
                "Cannot calculate quantiles from an empty mask."
            )

        channel = channel[mask]

    quantiles = np.quantile(
        channel,
        q_probs,
    ).astype(np.float32)

    return np.maximum.accumulate(quantiles)


def _strictly_increasing_knots(
    values: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    knots = np.asarray(values, dtype=np.float32).copy()
    knots = np.maximum.accumulate(knots)

    for index in range(1, knots.size):
        if knots[index] <= knots[index - 1]:
            knots[index] = knots[index - 1] + eps

    return knots


def _minimal_band_projection(
    source_quantiles: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> np.ndarray:
    projected = np.clip(
        source_quantiles,
        target_low,
        target_high,
    ).astype(np.float32)

    return np.maximum.accumulate(projected)


def _get_region_ref(
    device_ref: Dict[str, Any],
    region: str,
    fallback_to_all: bool = False,
) -> Dict[str, Any]:
    regions = device_ref.get("regions")

    if not isinstance(regions, dict):
        raise KeyError("Device reference is missing 'regions'.")

    if region in regions:
        return regions[region]

    if fallback_to_all and "all" in regions:
        return regions["all"]

    raise KeyError(
        f"Region '{region}' is missing. "
        f"Available regions: {tuple(regions.keys())}"
    )


def _sample_paired_reference_record(
    device_ref: Dict[str, Any],
    required_regions: Sequence[str],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    records = device_ref.get("images")

    if not isinstance(records, list) or not records:
        raise ValueError(
            "This cache has no paired per-image records. "
            "Rebuild it from the reviewed annotations."
        )

    candidates = [
        record
        for record in records
        if all(
            region in record.get("regions", {})
            for region in required_regions
        )
    ]

    if not candidates:
        raise ValueError(
            "No real reference image contains all required regions: "
            f"{tuple(required_regions)}."
        )

    index = int(rng.integers(0, len(candidates)))
    return candidates[index]


def _record_region_quantiles(
    record: Dict[str, Any],
    region: str,
) -> np.ndarray:
    try:
        quantiles = record["regions"][region]["q"]
    except KeyError as error:
        raise KeyError(
            f"Reference image '{record.get('id')}' is missing "
            f"region '{region}'."
        ) from error

    quantiles = np.asarray(quantiles, dtype=np.float32)

    if quantiles.ndim != 2 or quantiles.shape[0] != 3:
        raise ValueError(
            f"Invalid quantile shape for region '{region}': "
            f"{quantiles.shape}."
        )

    return quantiles


def _make_soft_cell_weight(
    cell_mask: np.ndarray,
    sigma: float = 1.5,
) -> np.ndarray:
    mask = np.asarray(cell_mask, dtype=np.float32)

    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(
                f"Expected a 2D cell mask, got {cell_mask.shape}."
            )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D cell mask, got {cell_mask.shape}."
        )

    mask = np.clip(mask, 0.0, 1.0)

    if sigma > 0:
        mask = cv2.GaussianBlur(
            mask,
            ksize=(0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
        )

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _apply_quantile_map_1d(
    values: np.ndarray,
    source_quantiles: np.ndarray,
    destination_quantiles: np.ndarray,
) -> np.ndarray:
    source_knots = _strictly_increasing_knots(
        source_quantiles
    )
    destination = np.maximum.accumulate(
        np.asarray(
            destination_quantiles,
            dtype=np.float32,
        )
    )

    mapped = np.interp(
        np.asarray(values, dtype=np.float32),
        source_knots,
        destination,
        left=destination[0],
        right=destination[-1],
    ).astype(np.float32)

    return np.clip(mapped, 0.0, 1.0)


def _adjust_image_to_region_reference(
    arr_hwc: np.ndarray,
    q_probs: np.ndarray,
    region_ref: Dict[str, Any],
    strength: float,
    match_mode: str,
    source_mask: Optional[np.ndarray] = None,
    sampled_quantiles: Optional[np.ndarray] = None,
) -> np.ndarray:
    output = arr_hwc.copy()

    sample_modes = {
        "sample_real_curve",
        "sample_real_image",
        "paired_real_image",
    }

    if match_mode in sample_modes:
        if sampled_quantiles is None:
            raise ValueError(
                "A sampled per-image quantile curve is required "
                f"for match_mode='{match_mode}'."
            )

        sampled_quantiles = np.asarray(
            sampled_quantiles,
            dtype=np.float32,
        )

        if (
            sampled_quantiles.ndim != 2
            or sampled_quantiles.shape[0] != 3
        ):
            raise ValueError(
                "sampled_quantiles must have shape [3, n_quantiles], "
                f"got {sampled_quantiles.shape}."
            )

    for channel in range(3):
        source_q = _compute_image_channel_quantiles(
            arr_hwc[..., channel],
            q_probs,
            mask=source_mask,
        )

        if match_mode == "project_to_band":
            target_q_full = _minimal_band_projection(
                source_q,
                region_ref["q_lo"][channel],
                region_ref["q_hi"][channel],
            )
        elif match_mode in sample_modes:
            target_q_full = sampled_quantiles[channel]
        elif match_mode == "center":
            target_q_full = np.asarray(
                region_ref["q_center"][channel],
                dtype=np.float32,
            )
        else:
            raise ValueError(
                "match_mode must be one of: "
                "'project_to_band', 'sample_real_curve', "
                "'sample_real_image', 'paired_real_image', 'center'."
            )

        destination_q = (
            (1.0 - strength) * source_q
            + strength * target_q_full
        ).astype(np.float32)
        destination_q = np.maximum.accumulate(destination_q)

        output[..., channel] = _apply_quantile_map_1d(
            arr_hwc[..., channel],
            source_q,
            destination_q,
        )

    return np.clip(output, 0.0, 1.0).astype(np.float32)


def apply_device_quantile_band_match(
    img: np.ndarray,
    target_device: str,
    quantile_band_cache: Dict[str, Any],
    strength: float = 1.0,
    preserve_input_layout: bool = True,
    rng: Optional[np.random.Generator] = None,
    cell_mask: Optional[np.ndarray] = None,
    region_mode: str = "all",
    match_mode: str = "project_to_band",
    mask_blur_sigma: float = 1.5,
    fallback_to_all: bool = False,
) -> np.ndarray:
    """
    Move an RGB image toward one real device reference.

    For sample-based modes, one complete real-image record is sampled once per
    call. Its foreground, background, and all RGB channels remain paired.

    Supported region modes
    ----------------------
    all:
        Apply the sampled image's reviewed all-pixel curve globally.

    foreground_background:
        Apply the sampled image's foreground and background curves separately,
        then blend them with a soft simulated-cell mask.
    """
    if target_device in {"simulated_raw", "raw_simulated"}:
        return np.clip(
            np.asarray(img, dtype=np.float32).copy(),
            0.0,
            1.0,
        )

    if "devices" not in quantile_band_cache:
        raise KeyError(
            "quantile_band_cache is missing 'devices'."
        )

    if "q_probs" not in quantile_band_cache:
        raise KeyError(
            "quantile_band_cache is missing 'q_probs'."
        )

    if target_device not in quantile_band_cache["devices"]:
        raise KeyError(
            f"Target device '{target_device}' is not in the cache."
        )

    strength = float(np.clip(strength, 0.0, 1.0))
    arr, input_chw = _as_hwc_rgb_float01(
        img,
        preserve_copy=True,
    )

    if strength <= 0.0:
        output = arr

        if preserve_input_layout and input_chw:
            output = np.moveaxis(output, -1, 0)

        return output.astype(np.float32)

    if rng is None:
        rng = np.random.default_rng()

    q_probs = np.asarray(
        quantile_band_cache["q_probs"],
        dtype=np.float32,
    )
    device_ref = quantile_band_cache["devices"][
        target_device
    ]

    sample_modes = {
        "sample_real_curve",
        "sample_real_image",
        "paired_real_image",
    }

    sampled_record: Optional[Dict[str, Any]] = None

    if region_mode == "all":
        region_ref = _get_region_ref(
            device_ref,
            "all",
            fallback_to_all=False,
        )

        sampled_q = None

        if match_mode in sample_modes:
            sampled_record = _sample_paired_reference_record(
                device_ref,
                required_regions=("all",),
                rng=rng,
            )
            sampled_q = _record_region_quantiles(
                sampled_record,
                "all",
            )

        output = _adjust_image_to_region_reference(
            arr_hwc=arr,
            q_probs=q_probs,
            region_ref=region_ref,
            strength=strength,
            match_mode=match_mode,
            source_mask=None,
            sampled_quantiles=sampled_q,
        )

    elif region_mode == "foreground_background":
        if cell_mask is None:
            if not fallback_to_all:
                raise ValueError(
                    "cell_mask is required for "
                    "region_mode='foreground_background'."
                )

            region_ref = _get_region_ref(
                device_ref,
                "all",
                fallback_to_all=False,
            )
            sampled_q = None

            if match_mode in sample_modes:
                sampled_record = _sample_paired_reference_record(
                    device_ref,
                    required_regions=("all",),
                    rng=rng,
                )
                sampled_q = _record_region_quantiles(
                    sampled_record,
                    "all",
                )

            output = _adjust_image_to_region_reference(
                arr_hwc=arr,
                q_probs=q_probs,
                region_ref=region_ref,
                strength=strength,
                match_mode=match_mode,
                source_mask=None,
                sampled_quantiles=sampled_q,
            )

        else:
            hard_fg = np.asarray(cell_mask)

            if hard_fg.ndim == 3:
                if hard_fg.shape[-1] == 1:
                    hard_fg = hard_fg[..., 0]
                elif hard_fg.shape[0] == 1:
                    hard_fg = hard_fg[0]
                else:
                    raise ValueError(
                        f"Expected a 2D cell mask, got {cell_mask.shape}."
                    )

            hard_fg = np.asarray(hard_fg > 0.5, dtype=bool)

            if hard_fg.shape != arr.shape[:2]:
                raise ValueError(
                    f"cell_mask shape {hard_fg.shape} does not match "
                    f"image shape {arr.shape[:2]}."
                )

            hard_bg = ~hard_fg

            if not hard_fg.any():
                raise ValueError(
                    "cell_mask contains no foreground pixels."
                )

            if not hard_bg.any():
                raise ValueError(
                    "cell_mask contains no background pixels."
                )

            fg_ref = _get_region_ref(
                device_ref,
                "foreground",
                fallback_to_all=fallback_to_all,
            )
            bg_ref = _get_region_ref(
                device_ref,
                "background",
                fallback_to_all=fallback_to_all,
            )

            sampled_fg_q = None
            sampled_bg_q = None

            if match_mode in sample_modes:
                try:
                    sampled_record = (
                        _sample_paired_reference_record(
                            device_ref,
                            required_regions=(
                                "foreground",
                                "background",
                            ),
                            rng=rng,
                        )
                    )
                except ValueError:
                    if not fallback_to_all:
                        raise

                    sampled_record = (
                        _sample_paired_reference_record(
                            device_ref,
                            required_regions=("all",),
                            rng=rng,
                        )
                    )

                    all_ref = _get_region_ref(
                        device_ref,
                        "all",
                        fallback_to_all=False,
                    )
                    sampled_all_q = _record_region_quantiles(
                        sampled_record,
                        "all",
                    )

                    output = _adjust_image_to_region_reference(
                        arr_hwc=arr,
                        q_probs=q_probs,
                        region_ref=all_ref,
                        strength=strength,
                        match_mode=match_mode,
                        source_mask=None,
                        sampled_quantiles=sampled_all_q,
                    )

                    if preserve_input_layout and input_chw:
                        output = np.moveaxis(output, -1, 0)

                    return np.clip(
                        output,
                        0.0,
                        1.0,
                    ).astype(np.float32)

                sampled_fg_q = _record_region_quantiles(
                    sampled_record,
                    "foreground",
                )
                sampled_bg_q = _record_region_quantiles(
                    sampled_record,
                    "background",
                )

            output_fg = _adjust_image_to_region_reference(
                arr_hwc=arr,
                q_probs=q_probs,
                region_ref=fg_ref,
                strength=strength,
                match_mode=match_mode,
                source_mask=hard_fg,
                sampled_quantiles=sampled_fg_q,
            )
            output_bg = _adjust_image_to_region_reference(
                arr_hwc=arr,
                q_probs=q_probs,
                region_ref=bg_ref,
                strength=strength,
                match_mode=match_mode,
                source_mask=hard_bg,
                sampled_quantiles=sampled_bg_q,
            )

            soft_weight = _make_soft_cell_weight(
                cell_mask,
                sigma=mask_blur_sigma,
            )[..., None]

            output = (
                (1.0 - soft_weight) * output_bg
                + soft_weight * output_fg
            )
            output = np.clip(
                output,
                0.0,
                1.0,
            ).astype(np.float32)

    else:
        raise ValueError(
            "region_mode must be one of: "
            "'all', 'foreground_background'."
        )

    if preserve_input_layout and input_chw:
        output = np.moveaxis(output, -1, 0)

    return np.clip(
        output,
        0.0,
        1.0,
    ).astype(np.float32)


def cache_uses_paired_image_sampling(
    quantile_band_cache: Optional[Dict[str, Any]],
) -> bool:
    """Return True for the new reviewed per-image cache."""
    return bool(
        quantile_band_cache
        and quantile_band_cache.get("reference_sampling")
        == PAIRED_REFERENCE_SAMPLING
    )

