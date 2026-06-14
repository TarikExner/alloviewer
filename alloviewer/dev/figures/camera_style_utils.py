from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch

from alloviewer.image_analysis.io import load_image
from alloviewer.dev.segmentation import UNET_MEAN, UNET_STD
from alloviewer.dev.segmentation.image_simulation.histogram_capture import (
    STYLE_CACHE_PATH,
    EXT_IMAGES_FOLDERS,
    DEFAULT_ANNOTATION_DIR,
    LABEL_IGNORE,
    LABEL_BACKGROUND,
    LABEL_FOREGROUND,
    LABEL_OUTSIDE_WELL,
    _find_device_label,
    _collect_image_paths,
    _annotation_paths_for_image,
    _annotation_is_complete,
)


DEFAULT_DEVICE_ORDER = (
    "iphone",
    "googlepixel",
    "microscope",
    "monochrome_generic",
    "monochrome_real",
    "simulated_raw",
    "synthetic",
)

FEATURE_CACHE_VERSION = 2
FEATURE_CACHE_TYPE = "reviewed_valid_pixels"
FEATURE_REGION_NAME = "foreground_and_background_inside_well"


def _load_reviewed_valid_mask(
    image_path: str | Path,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
) -> tuple[np.ndarray, dict]:
    """
    Load the reviewed valid-pixel mask for one real image.

    Used pixels are exactly:
      - inside the saved well mask
      - class 1 (background) or class 2 (foreground)

    Ignored pixels and outside-well pixels are excluded.
    """
    image_path = Path(image_path)
    paths = _annotation_paths_for_image(
        image_path,
        annotation_dir=annotation_dir,
    )

    labels = np.load(paths["regions"]).astype(np.uint8, copy=False)
    well_mask = np.load(paths["well_mask"]).astype(bool, copy=False)

    if labels.ndim != 2:
        raise ValueError(
            f"Expected a 2D reviewed mask for {image_path}, got {labels.shape}."
        )

    if well_mask.shape != labels.shape:
        raise ValueError(
            f"Well-mask shape {well_mask.shape} does not match "
            f"region-mask shape {labels.shape} for {image_path}."
        )

    valid_mask = (
        well_mask
        & (
            (labels == LABEL_BACKGROUND)
            | (labels == LABEL_FOREGROUND)
        )
        & (labels != LABEL_IGNORE)
        & (labels != LABEL_OUTSIDE_WELL)
    )

    if not valid_mask.any():
        raise ValueError(
            f"No reviewed foreground/background pixels remain for {image_path}."
        )

    metadata = {
        "regions_path": str(paths["regions"]),
        "well_mask_path": str(paths["well_mask"]),
        "metadata_path": str(paths["metadata"]),
        "n_valid_pixels": int(valid_mask.sum()),
        "valid_fraction": float(valid_mask.mean()),
    }

    return valid_mask, metadata


def _build_feature_source_inventory(
    image_paths: Sequence[Path],
    annotation_dir: str | Path,
    use_reviewed_regions: bool,
    require_annotations: bool,
) -> dict:
    """
    Build a stable inventory used to decide whether a feature cache is stale.
    """
    entries = []
    latest_mtime = 0.0

    for image_path in image_paths:
        annotation_complete = _annotation_is_complete(
            image_path,
            annotation_dir=annotation_dir,
        )

        if (
            use_reviewed_regions
            and require_annotations
            and not annotation_complete
        ):
            continue

        entry = {
            "image_path": str(image_path),
        }

        try:
            image_mtime = float(image_path.stat().st_mtime)
        except OSError:
            image_mtime = 0.0

        entry["image_mtime"] = image_mtime
        latest_mtime = max(latest_mtime, image_mtime)

        if use_reviewed_regions and annotation_complete:
            paths = _annotation_paths_for_image(
                image_path,
                annotation_dir=annotation_dir,
            )

            annotation_mtimes = {}

            for key in ("regions", "well_mask", "metadata"):
                try:
                    value = float(paths[key].stat().st_mtime)
                except OSError:
                    value = 0.0

                annotation_mtimes[key] = value
                latest_mtime = max(latest_mtime, value)

            entry["annotation_mtimes"] = annotation_mtimes

        entries.append(entry)

    return {
        "n_images": int(len(entries)),
        "entries": entries,
        "latest_mtime": float(latest_mtime),
    }


def _feature_cache_is_current(
    payload: dict,
    *,
    final_cache_path: Path,
    image_paths: Sequence[Path],
    annotation_dir: str | Path,
    normalize_imgs: bool,
    use_reviewed_regions: bool,
    require_annotations: bool,
    hist_bins: int,
    percentiles: Sequence[float],
    sample_pixels: Optional[int],
    normalized_hist_range: Tuple[float, float],
) -> bool:
    """
    Check cache version, settings, source list, and modification times.
    """
    if not isinstance(payload, dict):
        return False

    if payload.get("cache_version") != FEATURE_CACHE_VERSION:
        return False

    if payload.get("cache_type") != FEATURE_CACHE_TYPE:
        return False

    expected = {
        "normalize_imgs": bool(normalize_imgs),
        "use_reviewed_regions": bool(use_reviewed_regions),
        "require_annotations": bool(require_annotations),
        "hist_bins": int(hist_bins),
        "percentiles": tuple(float(p) for p in percentiles),
        "sample_pixels": sample_pixels,
        "normalized_hist_range": tuple(float(v) for v in normalized_hist_range),
        "annotation_dir": str(Path(annotation_dir)),
    }

    for key, value in expected.items():
        cached_value = payload.get(key)

        if key in {"percentiles", "normalized_hist_range"}:
            cached_value = tuple(cached_value)

        if cached_value != value:
            return False

    current_inventory = _build_feature_source_inventory(
        image_paths,
        annotation_dir=annotation_dir,
        use_reviewed_regions=use_reviewed_regions,
        require_annotations=require_annotations,
    )
    cached_inventory = payload.get("source_inventory", {})

    current_paths = tuple(
        entry["image_path"]
        for entry in current_inventory["entries"]
    )
    cached_paths = tuple(
        entry.get("image_path")
        for entry in cached_inventory.get("entries", [])
    )

    if current_paths != cached_paths:
        return False

    try:
        cache_mtime = float(final_cache_path.stat().st_mtime)
    except OSError:
        return False

    if current_inventory["latest_mtime"] > cache_mtime:
        return False

    return True


def _unet_mean_std_numpy() -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(UNET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(UNET_STD, dtype=np.float32).reshape(1, 1, 3)
    return mean, std


def normalize_rgb_image_with_unet(img_hwc: np.ndarray) -> np.ndarray:
    """
    img_hwc : float32 RGB image in [0,1], shape [H,W,3]
    returns : normalized float32 RGB image, shape [H,W,3]
    """
    img_hwc = np.asarray(img_hwc, dtype=np.float32)
    if img_hwc.ndim != 3 or img_hwc.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {img_hwc.shape}")

    mean, std = _unet_mean_std_numpy()
    return (img_hwc - mean) / std


def denormalize_rgb_image_with_unet(img_hwc: np.ndarray) -> np.ndarray:
    """
    img_hwc : normalized float32 RGB image, shape [H,W,3]
    returns : denormalized float32 RGB image, shape [H,W,3]
    """
    img_hwc = np.asarray(img_hwc, dtype=np.float32)
    if img_hwc.ndim != 3 or img_hwc.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {img_hwc.shape}")

    mean, std = _unet_mean_std_numpy()
    return img_hwc * std + mean


def denormalize_dataset_imgs(imgs_norm: torch.Tensor) -> torch.Tensor:
    """
    imgs_norm: [T,3,H,W] or [3,H,W], already normalized with UNET_MEAN/UNET_STD
    returns : same rank in image space
    """
    if imgs_norm.ndim == 3:
        mean = torch.as_tensor(
            UNET_MEAN,
            dtype=imgs_norm.dtype,
            device=imgs_norm.device,
        ).view(3, 1, 1)
        std = torch.as_tensor(
            UNET_STD,
            dtype=imgs_norm.dtype,
            device=imgs_norm.device,
        ).view(3, 1, 1)
        return imgs_norm * std + mean

    if imgs_norm.ndim == 4:
        mean = torch.as_tensor(
            UNET_MEAN,
            dtype=imgs_norm.dtype,
            device=imgs_norm.device,
        ).view(1, 3, 1, 1)
        std = torch.as_tensor(
            UNET_STD,
            dtype=imgs_norm.dtype,
            device=imgs_norm.device,
        ).view(1, 3, 1, 1)
        return imgs_norm * std + mean

    raise ValueError(f"Expected [3,H,W] or [T,3,H,W], got shape {tuple(imgs_norm.shape)}")


def _ensure_tile_tensor(imgs_t: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor shape is [T,3,H,W].
    """
    if imgs_t.ndim == 3:
        if imgs_t.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W], got {tuple(imgs_t.shape)}")
        return imgs_t.unsqueeze(0)

    if imgs_t.ndim == 4:
        if imgs_t.shape[1] != 3:
            raise ValueError(f"Expected [T,3,H,W], got {tuple(imgs_t.shape)}")
        return imgs_t

    raise ValueError(f"Expected [3,H,W] or [T,3,H,W], got {tuple(imgs_t.shape)}")


def _safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0

    m = float(x.mean())
    s = float(x.std())
    if s < 1e-12:
        return 0.0

    z = (x - m) / s
    return float(np.mean(z ** 3))


def _build_feature_names(
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
) -> list[str]:
    feature_names: list[str] = []

    for ch in ("r", "g", "b"):
        feature_names.append(f"{ch}_mean")
        feature_names.append(f"{ch}_std")
        feature_names.append(f"{ch}_skew")

        for p in percentiles:
            p_name = str(float(p)).replace(".", "_")
            feature_names.append(f"{ch}_p{p_name}")

        for b in range(hist_bins):
            feature_names.append(f"{ch}_hist_{b:02d}")

    feature_names.extend([
        "gray_mean",
        "gray_std",
        "gray_skew",
    ])

    for p in percentiles:
        p_name = str(float(p)).replace(".", "_")
        feature_names.append(f"gray_p{p_name}")

    feature_names.extend([
        "sat_mean",
        "sat_std",
        "sat_skew",
        "dark_frac",
        "bright_frac",
        "n_pixels_used",
        "aspect_ratio",
        "height",
        "width",
    ])

    return feature_names


def extract_feature_row_from_rgb_image(
    img: np.ndarray,
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    rng: Optional[np.random.Generator] = None,
    phone_label: str = "synthetic",
    path_label: str = "<in_memory>",
    hist_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    dark_threshold: Optional[float] = 0.02,
    bright_threshold: Optional[float] = 0.98,
    clip_input: bool = True,
    pixel_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Extract one feature row from one RGB image.

    Supports:
      - HWC [H,W,3]
      - CHW [3,H,W]

    If ``pixel_mask`` is provided, all statistics and histograms are calculated
    only from True pixels. The mask must match the original H x W image shape.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 3:
        raise ValueError(f"img must be 3D, got shape {img.shape}")

    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)

    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {img.shape}")

    if hist_range is not None and clip_input:
        lo, hi = hist_range
        img = np.clip(img, lo, hi)

    H, W, _ = img.shape
    percentiles = tuple(float(p) for p in percentiles)

    def _norm_hist(x: np.ndarray, bins: int, value_range: Tuple[float, float]) -> np.ndarray:
        h, _ = np.histogram(x, bins=bins, range=value_range)
        h = h.astype(np.float32)
        h /= (h.sum() + 1e-8)
        return h

    if pixel_mask is None:
        pixels = img.reshape(-1, 3)
    else:
        mask = np.asarray(pixel_mask, dtype=bool)

        if mask.shape != (H, W):
            raise ValueError(
                f"pixel_mask shape {mask.shape} does not match image shape {(H, W)}."
            )

        if not mask.any():
            raise ValueError("pixel_mask contains no True pixels.")

        pixels = img[mask]

    n_total = pixels.shape[0]

    if sample_pixels is not None and n_total > sample_pixels:
        idx = rng.choice(n_total, size=int(sample_pixels), replace=False)
        pixels_use = pixels[idx]
    else:
        pixels_use = pixels

    r = pixels_use[:, 0]
    g = pixels_use[:, 1]
    b = pixels_use[:, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    rgb_max = np.max(pixels_use, axis=1)
    rgb_min = np.min(pixels_use, axis=1)
    sat = np.where(
        np.abs(rgb_max) > 1e-8,
        (rgb_max - rgb_min) / (np.abs(rgb_max) + 1e-8),
        0.0,
    ).astype(np.float32)

    row = {
        "path": path_label,
        "filename": Path(path_label).name if not str(path_label).startswith("<") else path_label,
        "phone": str(phone_label).lower(),
        "height": int(H),
        "width": int(W),
        "aspect_ratio": float(W / max(H, 1)),
        "n_pixels_used": int(pixels_use.shape[0]),
        "n_pixels_available": int(n_total),
        "pixel_fraction_available": float(n_total / max(H * W, 1)),
    }

    for ch_name, x in zip(("r", "g", "b"), (r, g, b)):
        row[f"{ch_name}_mean"] = float(np.mean(x))
        row[f"{ch_name}_std"] = float(np.std(x))
        row[f"{ch_name}_skew"] = float(_safe_skew(x))

        pvals = np.percentile(x, percentiles)
        for p, val in zip(percentiles, pvals):
            p_name = str(p).replace(".", "_")
            row[f"{ch_name}_p{p_name}"] = float(val)

        if hist_range is None:
            x_lo = float(np.min(x))
            x_hi = float(np.max(x))
            if x_hi <= x_lo:
                x_hi = x_lo + 1e-6
            hist = _norm_hist(x, bins=hist_bins, value_range=(x_lo, x_hi))
        else:
            hist = _norm_hist(x, bins=hist_bins, value_range=hist_range)

        for i, val in enumerate(hist):
            row[f"{ch_name}_hist_{i:02d}"] = float(val)

    row["gray_mean"] = float(np.mean(gray))
    row["gray_std"] = float(np.std(gray))
    row["gray_skew"] = float(_safe_skew(gray))

    gray_pvals = np.percentile(gray, percentiles)
    for p, val in zip(percentiles, gray_pvals):
        p_name = str(p).replace(".", "_")
        row[f"gray_p{p_name}"] = float(val)

    row["sat_mean"] = float(np.mean(sat))
    row["sat_std"] = float(np.std(sat))
    row["sat_skew"] = float(_safe_skew(sat))

    row["dark_frac"] = (
        float(np.mean(gray <= dark_threshold))
        if dark_threshold is not None
        else np.nan
    )
    row["bright_frac"] = (
        float(np.mean(gray >= bright_threshold))
        if bright_threshold is not None
        else np.nan
    )

    return row


def get_feature_cache_path(
    cache_path: str | Path = STYLE_CACHE_PATH,
    normalized: bool = False,
) -> Path:
    cache_path = Path(cache_path)
    if normalized:
        return cache_path.with_name(f"{cache_path.stem}_normalized{cache_path.suffix}")
    return cache_path


def extract_real_image_feature_table(
    folders: Optional[Sequence[str | Path]] = None,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    recursive: bool = True,
    ignore_failures: bool = True,
    cache_path: str | Path = STYLE_CACHE_PATH,
    force_recompute: bool = False,
    normalize_imgs: bool = False,
    normalized_hist_range: Tuple[float, float] = (-3.0, 3.0),
    rng_seed: int = 0,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    use_reviewed_regions: bool = True,
    require_annotations: bool = True,
):
    """
    Build and cache feature statistics from real images.

    The default input folders are exactly ``EXT_IMAGES_FOLDERS`` imported from
    ``histogram_capture``.

    When ``use_reviewed_regions=True`` (recommended), statistics are calculated
    only from pixels that are:
      - inside the saved well mask
      - labelled background (1) or foreground (2)

    Ignore pixels (0) and outside-well pixels (3) are excluded.

    Two separate caches are supported:
      - normalize_imgs=False: image-space features
      - normalize_imgs=True: UNet-normalized image-space features

    In normalized mode, the image is normalized first and then the same reviewed
    spatial mask is applied. This matches the pixels presented to the model while
    avoiding the dark phone area outside the well.
    """
    final_cache_path = get_feature_cache_path(
        cache_path=cache_path,
        normalized=normalize_imgs,
    )

    annotation_dir = Path(annotation_dir)
    percentiles = tuple(float(p) for p in percentiles)

    image_paths = _collect_image_paths(
        folders=folders,
        exts=exts,
        recursive=recursive,
    )

    if final_cache_path.exists() and not force_recompute:
        try:
            with open(final_cache_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            payload = None

        if payload is not None and _feature_cache_is_current(
            payload,
            final_cache_path=final_cache_path,
            image_paths=image_paths,
            annotation_dir=annotation_dir,
            normalize_imgs=normalize_imgs,
            use_reviewed_regions=use_reviewed_regions,
            require_annotations=require_annotations,
            hist_bins=hist_bins,
            percentiles=percentiles,
            sample_pixels=sample_pixels,
            normalized_hist_range=normalized_hist_range,
        ):
            return payload["rows"], payload["feature_names"], payload["X"]

    rng = np.random.default_rng(rng_seed)

    feature_names = _build_feature_names(
        hist_bins=hist_bins,
        percentiles=percentiles,
    )

    rows = []
    skipped: list[dict] = []
    n_missing_annotations = 0

    for path in tqdm(
        image_paths,
        desc=(
            "Extracting normalized reviewed-image features"
            if normalize_imgs
            else "Extracting reviewed-image features"
        ),
    ):
        valid_mask = None
        annotation_metadata = None

        if use_reviewed_regions:
            if not _annotation_is_complete(
                path,
                annotation_dir=annotation_dir,
            ):
                n_missing_annotations += 1

                if require_annotations:
                    skipped.append({
                        "path": str(path),
                        "reason": "missing reviewed annotation",
                    })
                    continue
            else:
                try:
                    valid_mask, annotation_metadata = _load_reviewed_valid_mask(
                        path,
                        annotation_dir=annotation_dir,
                    )
                except Exception as error:
                    if ignore_failures:
                        skipped.append({
                            "path": str(path),
                            "reason": f"annotation load failed: {error!r}",
                        })
                        continue
                    raise

        try:
            img, _ = load_image(
                path.name,
                base_dir=path.parent,
                as_chw=False,
                scale=True,
                fast_scale=True,
            )
        except Exception as error:
            if ignore_failures:
                skipped.append({
                    "path": str(path),
                    "reason": f"image load failed: {error!r}",
                })
                continue
            raise

        img = np.asarray(img, dtype=np.float32)

        if img.ndim != 3 or img.shape[2] != 3:
            message = f"Bad image shape: {path} -> {img.shape}"

            if ignore_failures:
                skipped.append({
                    "path": str(path),
                    "reason": message,
                })
                continue

            raise RuntimeError(message)

        if valid_mask is not None and valid_mask.shape != img.shape[:2]:
            message = (
                f"Reviewed mask shape {valid_mask.shape} does not match "
                f"image shape {img.shape[:2]} for {path}."
            )

            if ignore_failures:
                skipped.append({
                    "path": str(path),
                    "reason": message,
                })
                continue

            raise RuntimeError(message)

        device_label = _find_device_label(path)

        if normalize_imgs:
            img_work = normalize_rgb_image_with_unet(
                np.clip(img, 0.0, 1.0)
            )

            row = extract_feature_row_from_rgb_image(
                img=img_work,
                hist_bins=hist_bins,
                percentiles=percentiles,
                sample_pixels=sample_pixels,
                rng=rng,
                phone_label=device_label,
                path_label=str(path),
                hist_range=normalized_hist_range,
                dark_threshold=None,
                bright_threshold=None,
                clip_input=True,
                pixel_mask=valid_mask,
            )
        else:
            img_work = np.clip(
                img.astype(np.float32, copy=False),
                0.0,
                1.0,
            )

            row = extract_feature_row_from_rgb_image(
                img=img_work,
                hist_bins=hist_bins,
                percentiles=percentiles,
                sample_pixels=sample_pixels,
                rng=rng,
                phone_label=device_label,
                path_label=str(path),
                hist_range=(0.0, 1.0),
                dark_threshold=0.02,
                bright_threshold=0.98,
                clip_input=True,
                pixel_mask=valid_mask,
            )

        row["filename"] = path.name
        row["feature_region"] = (
            FEATURE_REGION_NAME
            if valid_mask is not None
            else "full_image"
        )
        row["annotation_used"] = bool(valid_mask is not None)

        if annotation_metadata is not None:
            row.update(annotation_metadata)

        rows.append(row)

    if not rows:
        raise RuntimeError("No usable real images were processed.")

    X = np.array(
        [
            [row[name] for name in feature_names]
            for row in rows
        ],
        dtype=np.float32,
    )

    source_inventory = _build_feature_source_inventory(
        image_paths,
        annotation_dir=annotation_dir,
        use_reviewed_regions=use_reviewed_regions,
        require_annotations=require_annotations,
    )

    payload = {
        "cache_version": FEATURE_CACHE_VERSION,
        "cache_type": FEATURE_CACHE_TYPE,
        "feature_region": (
            FEATURE_REGION_NAME
            if use_reviewed_regions
            else "full_image"
        ),
        "rows": rows,
        "feature_names": feature_names,
        "X": X,
        "folders": [
            str(folder)
            for folder in (
                folders
                if folders is not None
                else EXT_IMAGES_FOLDERS
            )
        ],
        "annotation_dir": str(annotation_dir),
        "use_reviewed_regions": bool(use_reviewed_regions),
        "require_annotations": bool(require_annotations),
        "hist_bins": int(hist_bins),
        "percentiles": percentiles,
        "sample_pixels": sample_pixels,
        "normalize_imgs": bool(normalize_imgs),
        "normalized_hist_range": tuple(
            float(value)
            for value in normalized_hist_range
        ),
        "source_inventory": source_inventory,
        "n_source_images": int(len(image_paths)),
        "n_processed_images": int(len(rows)),
        "n_missing_annotations": int(n_missing_annotations),
        "skipped": skipped,
    }

    final_cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(final_cache_path, "wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return rows, feature_names, X


def rebuild_style_feature_cache(
    folders: Optional[Sequence[str | Path]] = None,
    cache_path: str | Path = STYLE_CACHE_PATH,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    normalize_imgs: bool = False,
    normalized_hist_range: Tuple[float, float] = (-3.0, 3.0),
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    use_reviewed_regions: bool = True,
    require_annotations: bool = True,
):
    """
    Force-rebuild one of the two caches.
    """
    return extract_real_image_feature_table(
        folders=folders,
        exts=exts,
        hist_bins=hist_bins,
        percentiles=percentiles,
        sample_pixels=sample_pixels,
        cache_path=cache_path,
        force_recompute=True,
        normalize_imgs=normalize_imgs,
        normalized_hist_range=normalized_hist_range,
        annotation_dir=annotation_dir,
        use_reviewed_regions=use_reviewed_regions,
        require_annotations=require_annotations,
    )


def summarize_features_by_phone(
    rows: list[dict],
    feature_names: Sequence[str],
    phones: Sequence[str] = DEFAULT_DEVICE_ORDER,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
):
    if not rows:
        raise ValueError("rows is empty")

    q_lo = float(q_lo)
    q_hi = float(q_hi)
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError("Need 0 <= q_lo < q_hi <= 1")

    summaries: Dict[str, dict] = {}
    total_n = len(rows)

    for phone in phones:
        subset = [row for row in rows if str(row.get("phone", "")).lower() == phone]
        if len(subset) == 0:
            continue

        X = np.array(
            [[row[f] for f in feature_names] for row in subset],
            dtype=np.float64,
        )

        means = X.mean(axis=0)
        stds = X.std(axis=0)
        meds = np.median(X, axis=0)
        qlos = np.quantile(X, q_lo, axis=0)
        qhis = np.quantile(X, q_hi, axis=0)

        summaries[phone] = {
            "phone": phone,
            "n_images": len(subset),
            "fraction": len(subset) / total_n,
            "feature_mean": {f: float(v) for f, v in zip(feature_names, means)},
            "feature_std": {f: float(v) for f, v in zip(feature_names, stds)},
            "feature_median": {f: float(v) for f, v in zip(feature_names, meds)},
            "feature_q_lo": {f: float(v) for f, v in zip(feature_names, qlos)},
            "feature_q_hi": {f: float(v) for f, v in zip(feature_names, qhis)},
        }

    return summaries


def compare_real_and_synthetic_feature_tables(
    real_rows: Sequence[dict],
    synthetic_rows: Sequence[dict],
    feature_names: Sequence[str],
    group_names: Sequence[str] = DEFAULT_DEVICE_ORDER,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
) -> list[dict]:
    out = []

    for group in group_names:
        real_sub = [
            r for r in real_rows
            if str(r.get("phone", "")).lower() == group
        ]
        syn_sub = [
            r for r in synthetic_rows
            if str(r.get("phone", "")).lower() == group
        ]

        if len(real_sub) == 0 or len(syn_sub) == 0:
            continue

        XR = np.array(
            [[r[f] for f in feature_names] for r in real_sub],
            dtype=np.float64,
        )
        XS = np.array(
            [[r[f] for f in feature_names] for r in syn_sub],
            dtype=np.float64,
        )

        real_med = np.median(XR, axis=0)
        syn_med = np.median(XS, axis=0)

        real_std = np.std(XR, axis=0) + 1e-8
        syn_std = np.std(XS, axis=0) + 1e-8

        real_lo = np.quantile(XR, q_lo, axis=0)
        real_hi = np.quantile(XR, q_hi, axis=0)
        syn_lo = np.quantile(XS, q_lo, axis=0)
        syn_hi = np.quantile(XS, q_hi, axis=0)

        median_diff_z = (syn_med - real_med) / real_std
        std_ratio = np.maximum(syn_std / real_std, real_std / syn_std)

        real_range = np.maximum(real_hi - real_lo, 1e-8)
        syn_range = np.maximum(syn_hi - syn_lo, 1e-8)
        range_ratio = np.maximum(syn_range / real_range, real_range / syn_range)

        overlap_lo = np.maximum(real_lo, syn_lo)
        overlap_hi = np.minimum(real_hi, syn_hi)
        overlap = np.maximum(0.0, overlap_hi - overlap_lo)

        union_lo = np.minimum(real_lo, syn_lo)
        union_hi = np.maximum(real_hi, syn_hi)
        union = np.maximum(1e-8, union_hi - union_lo)

        overlap_frac = overlap / union

        mismatch_score = (
            np.abs(median_diff_z)
            + 0.35 * np.abs(np.log(std_ratio))
            + 0.35 * np.abs(np.log(range_ratio))
            + 2.0 * (1.0 - overlap_frac)
        )

        for i, feat in enumerate(feature_names):
            out.append(
                {
                    "group": group,
                    "feature": feat,
                    "real_median": float(real_med[i]),
                    "syn_median": float(syn_med[i]),
                    "median_diff_z": float(median_diff_z[i]),
                    "std_ratio": float(std_ratio[i]),
                    "range_ratio": float(range_ratio[i]),
                    "range_overlap_frac": float(overlap_frac[i]),
                    "mismatch_score": float(mismatch_score[i]),
                }
            )

    out.sort(key=lambda x: (x["group"], -x["mismatch_score"]))
    return out


def _extract_style_name_from_extras(extras: Dict[str, Any]) -> str:
    if not isinstance(extras, dict):
        return "synthetic"

    meta = extras.get("meta", {})
    if not isinstance(meta, dict):
        return "synthetic"

    full_meta = meta.get("full", {})
    if isinstance(full_meta, dict):
        cam_meta = full_meta.get("camera_style", {})
        if isinstance(cam_meta, dict):
            style_name = cam_meta.get("style_name", None)
            if style_name is not None:
                return str(style_name).lower()

    tiles_meta = meta.get("tiles", None)
    if isinstance(tiles_meta, list):
        for t in tiles_meta:
            if not isinstance(t, dict):
                continue

            cam_meta = t.get("camera_style", {})
            if isinstance(cam_meta, dict):
                style_name = cam_meta.get("style_name", None)
                if style_name is not None:
                    return str(style_name).lower()

    return "synthetic"


def collect_synthetic_feature_rows_from_dataset(
    dataset,
    n_synthetic: int = 100,
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    rng_seed: int = 0,
    use_first_tile_only: bool = True,
    normalized_features: bool = False,
    normalized_hist_range: Tuple[float, float] = (-3.0, 3.0),
) -> list[dict]:
    """
    Collect feature rows from a dataset whose stored images are already normalized.

    normalized_features=False:
        dataset images are first denormalized, then image-space features are extracted.

    normalized_features=True:
        dataset images are used as-is, and normalized-space features are extracted.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []

    n_items = min(int(n_synthetic), len(dataset))

    for i in range(n_items):
        item = dataset[i]

        if not isinstance(item, tuple) or len(item) < 3:
            raise ValueError(
                "Expected dataset[i] to return at least "
                "(imgs_t, target, extras)"
            )

        imgs_t, _, extras = item[:3]
        style_name = _extract_style_name_from_extras(extras)

        imgs_t = _ensure_tile_tensor(imgs_t.detach())

        if normalized_features:
            imgs_work = imgs_t
        else:
            imgs_work = denormalize_dataset_imgs(imgs_t)

        if use_first_tile_only:
            tile_indices = [0]
        else:
            tile_indices = list(range(imgs_work.shape[0]))

        for t_idx in tile_indices:
            img = (
                imgs_work[t_idx]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if normalized_features:
                row = extract_feature_row_from_rgb_image(
                    img=img,
                    hist_bins=hist_bins,
                    percentiles=percentiles,
                    sample_pixels=sample_pixels,
                    rng=rng,
                    phone_label=style_name,
                    path_label=f"<synthetic_{i}_tile_{t_idx}>",
                    hist_range=normalized_hist_range,
                    dark_threshold=None,
                    bright_threshold=None,
                    clip_input=True,
                )
            else:
                img = np.clip(img, 0.0, 1.0)
                row = extract_feature_row_from_rgb_image(
                    img=img,
                    hist_bins=hist_bins,
                    percentiles=percentiles,
                    sample_pixels=sample_pixels,
                    rng=rng,
                    phone_label=style_name,
                    path_label=f"<synthetic_{i}_tile_{t_idx}>",
                    hist_range=(0.0, 1.0),
                    dark_threshold=0.02,
                    bright_threshold=0.98,
                    clip_input=True,
                )

            rows.append(row)

    return rows


def _ordered_labels_present(labels: Sequence[str]) -> list[str]:
    labels_set = {str(x).lower() for x in labels}

    ordered = [x for x in DEFAULT_DEVICE_ORDER if x in labels_set]
    extras = sorted(labels_set.difference(ordered))

    return ordered + extras


def _select_pca_features(
    feature_names: Sequence[str],
    normalized: bool,
    feature_subset: Optional[Sequence[str]] = None,
    drop_size_features: bool = True,
) -> list[str]:
    if feature_subset is not None:
        return list(feature_subset)

    if drop_size_features:
        drop = {"height", "width", "aspect_ratio", "n_pixels_used"}
    else:
        drop = set()

    if normalized:
        drop = set(drop)
        drop.update({"dark_frac", "bright_frac", "sat_mean", "sat_std", "sat_skew"})

    return [f for f in feature_names if f not in drop]


def plot_real_and_synthetic_pca(
    dataset,
    folders: Optional[Sequence[str | Path]] = None,
    cache_path: str | Path = STYLE_CACHE_PATH,
    n_synthetic: int = 100,
    normalized: bool = False,
    feature_subset: Optional[Sequence[str]] = None,
    drop_size_features: bool = True,
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    rng_seed: int = 0,
    alpha_real: float = 0.45,
    alpha_syn: float = 0.80,
    s_real: float = 16,
    s_syn: float = 36,
    use_first_tile_only: bool = True,
    normalized_hist_range: Tuple[float, float] = (-3.0, 3.0),
    title: Optional[str] = None,
    force_recompute_real_cache: bool = False,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    use_reviewed_regions: bool = True,
    require_annotations: bool = True,
):
    """
    PCA comparison between real images and dataset images.

    normalized=False:
        real cache is image-space cache.
        dataset images are denormalized before feature extraction.

    normalized=True:
        real cache is normalized-image cache.
        dataset images are used as stored.
    """
    real_rows, feature_names, _ = extract_real_image_feature_table(
        folders=folders,
        hist_bins=hist_bins,
        percentiles=percentiles,
        sample_pixels=sample_pixels,
        cache_path=cache_path,
        force_recompute=force_recompute_real_cache,
        normalize_imgs=normalized,
        normalized_hist_range=normalized_hist_range,
        rng_seed=rng_seed,
        annotation_dir=annotation_dir,
        use_reviewed_regions=use_reviewed_regions,
        require_annotations=require_annotations,
    )

    feature_names_used = _select_pca_features(
        feature_names=feature_names,
        normalized=normalized,
        feature_subset=feature_subset,
        drop_size_features=drop_size_features,
    )

    if not feature_names_used:
        raise ValueError("No features selected for PCA")

    synthetic_rows = collect_synthetic_feature_rows_from_dataset(
        dataset=dataset,
        n_synthetic=n_synthetic,
        hist_bins=hist_bins,
        percentiles=percentiles,
        sample_pixels=sample_pixels,
        rng_seed=rng_seed,
        use_first_tile_only=use_first_tile_only,
        normalized_features=normalized,
        normalized_hist_range=normalized_hist_range,
    )

    if len(synthetic_rows) == 0:
        raise RuntimeError("No synthetic rows were collected from the dataset.")

    all_rows = list(real_rows) + list(synthetic_rows)

    X = np.array(
        [[row[f] for f in feature_names_used] for row in all_rows],
        dtype=np.float64,
    )

    finite_cols = np.all(np.isfinite(X), axis=0)
    if not finite_cols.all():
        feature_names_used = [
            f for f, keep in zip(feature_names_used, finite_cols)
            if keep
        ]
        X = X[:, finite_cols]

    if X.shape[0] < 2:
        raise ValueError("Need at least two images for PCA.")

    if X.shape[1] < 2:
        raise ValueError("Need at least two finite features for PCA.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=rng_seed)
    X_pca = pca.fit_transform(X_scaled)

    labels = [str(row["phone"]).lower() for row in all_rows]
    is_real = np.array(
        [not str(row["path"]).startswith("<synthetic_") for row in all_rows],
        dtype=bool,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    present_labels = _ordered_labels_present(labels)

    syn_markers = {
        "iphone": "x",
        "googlepixel": "^",
        "microscope": "s",
        "monochrome_generic": "P",
        "monochrome_real": "v",
        "simulated_raw": "D",
        "synthetic": "D",
    }

    for dev in present_labels:
        idx = [
            i for i, lab in enumerate(labels)
            if lab == dev and is_real[i]
        ]
        if idx:
            pts = X_pca[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                alpha=alpha_real,
                s=s_real,
                label=f"real {dev}",
            )

    for dev in present_labels:
        idx = [
            i for i, lab in enumerate(labels)
            if lab == dev and not is_real[i]
        ]
        if idx:
            pts = X_pca[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                alpha=alpha_syn,
                s=s_syn,
                marker=syn_markers.get(dev, "x"),
                label=f"synthetic {dev}",
            )

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100.0 * evr[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100.0 * evr[1]:.1f}% var)")

    if title is None:
        title = (
            "PCA of normalized real vs normalized synthetic image features"
            if normalized
            else "PCA of real vs denormalized synthetic image features"
        )

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "cache_path_used": str(get_feature_cache_path(cache_path, normalized=normalized)),
        "normalized": normalized,
        "real_rows": real_rows,
        "synthetic_rows": synthetic_rows,
        "feature_names_used": feature_names_used,
        "X_scaled": X_scaled.astype(np.float32),
        "X_pca": X_pca.astype(np.float32),
        "labels": labels,
        "is_real": is_real,
        "scaler": scaler,
        "pca": pca,
    }

def plot_real_image_pca(
    folders: Optional[Sequence[str | Path]] = None,
    cache_path: str | Path = STYLE_CACHE_PATH,
    normalized: bool = False,
    feature_subset: Optional[Sequence[str]] = None,
    drop_size_features: bool = True,
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    rng_seed: int = 0,
    normalized_hist_range: Tuple[float, float] = (-3.0, 3.0),
    force_recompute_real_cache: bool = False,
    annotation_dir: str | Path = DEFAULT_ANNOTATION_DIR,
    use_reviewed_regions: bool = True,
    require_annotations: bool = True,
    title: Optional[str] = None,
    point_size: float = 28,
    alpha: float = 0.75,
):
    """
    Plot PCA for external/real images only.
    """
    real_rows, feature_names, _ = extract_real_image_feature_table(
        folders=folders,
        hist_bins=hist_bins,
        percentiles=percentiles,
        sample_pixels=sample_pixels,
        cache_path=cache_path,
        force_recompute=force_recompute_real_cache,
        normalize_imgs=normalized,
        normalized_hist_range=normalized_hist_range,
        rng_seed=rng_seed,
        annotation_dir=annotation_dir,
        use_reviewed_regions=use_reviewed_regions,
        require_annotations=require_annotations,
    )

    feature_names_used = _select_pca_features(
        feature_names=feature_names,
        normalized=normalized,
        feature_subset=feature_subset,
        drop_size_features=drop_size_features,
    )

    if not feature_names_used:
        raise ValueError("No features selected for PCA")

    X = np.array(
        [[row[f] for f in feature_names_used] for row in real_rows],
        dtype=np.float64,
    )

    finite_cols = np.all(np.isfinite(X), axis=0)
    if not finite_cols.all():
        feature_names_used = [
            f for f, keep in zip(feature_names_used, finite_cols)
            if keep
        ]
        X = X[:, finite_cols]

    if X.shape[0] < 2:
        raise ValueError("Need at least two real images for PCA.")

    if X.shape[1] < 2:
        raise ValueError("Need at least two finite features for PCA.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=rng_seed)
    X_pca = pca.fit_transform(X_scaled)

    labels = [str(row["phone"]).lower() for row in real_rows]
    present_labels = _ordered_labels_present(labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    for dev in present_labels:
        idx = [i for i, lab in enumerate(labels) if lab == dev]
        if idx:
            pts = X_pca[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=point_size,
                alpha=alpha,
                label=dev,
            )

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100.0 * evr[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100.0 * evr[1]:.1f}% var)")

    if title is None:
        title = (
            "PCA of normalized external image features"
            if normalized
            else "PCA of external image features"
        )

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "cache_path_used": str(get_feature_cache_path(cache_path, normalized=normalized)),
        "normalized": normalized,
        "real_rows": real_rows,
        "feature_names_used": feature_names_used,
        "X_scaled": X_scaled.astype(np.float32),
        "X_pca": X_pca.astype(np.float32),
        "labels": labels,
        "scaler": scaler,
        "pca": pca,
    }
