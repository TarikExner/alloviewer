import os
import pickle
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple, Union

import cv2
import numpy as np

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
    "./ext_images/20260507_XM1_mono_rgb"
]

MONOCHROME_IMAGES_FOLDERS_INPUT = [
    "./ext_images/20260504_AM1_mono",
    "./ext_images/20260504_Auto1_mono",
    "./ext_images/20260507_XM1_+DTT_mono",
    "./ext_images/20260507_XM1_mono"
]

MONOCHROME_IMAGES_FOLDERS_OUTPUT = [
    "./ext_images/20260504_AM1_mono_rgb",
    "./ext_images/20260504_Auto1_mono_rgb",
    "./ext_images/20260507_XM1_+DTT_mono_rgb",
    "./ext_images/20260507_XM1_mono_rgb"
]

STYLE_CACHE_PATH = Path("./results/style_cache.cache")
STYLE_QUANTILE_CACHE_PATH = STYLE_CACHE_PATH.with_name("camera_quantile_band_cache.pkl")

REGION_NAMES = ("all", "foreground", "background")

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

def _resolve_device_value(value, device: str, default):
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(device, value.get("default", default))
    return value

def _find_device_label(file_path: str | Path) -> str:
    s = str(file_path).lower()

    if "mono_rgb" in s or "mono_real" in s:
        return "monochrome_real"

    if "iphone" in s:
        return "iphone"

    if "googlepixel" in s or "pixel" in s:
        return "googlepixel"

    return "microscope"

def _collect_image_paths(
    folders: Optional[Sequence[str | Path]] = None,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    recursive: bool = True,
) -> list[Path]:
    if folders is None:
        folders = EXT_IMAGES_FOLDERS

    exts = tuple(e.lower() for e in exts)
    image_paths: list[Path] = []

    # Deduplicate folders while preserving order.
    folders_unique = list(dict.fromkeys(Path(f) for f in folders))

    for folder in folders_unique:
        if not folder.exists():
            continue

        walker = folder.rglob("*") if recursive else folder.glob("*")
        for path in walker:
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)

    # Deduplicate paths while preserving order, then sort for stable builds.
    image_paths = sorted(dict.fromkeys(image_paths))

    if not image_paths:
        raise RuntimeError("No image files found.")

    return image_paths


def _as_hwc_rgb_float01(
    img: np.ndarray,
    preserve_copy: bool = False,
) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(img, dtype=np.float32)
    input_chw = False

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")

    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
        input_chw = True
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected CHW or HWC RGB image, got shape {img.shape}")

    if preserve_copy:
        arr = arr.copy()

    arr = np.clip(arr, 0.0, 1.0)
    return arr, input_chw


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
    Create simple automatic foreground/background masks for real reference images.

    Intended use:
      - foreground = bright cell-like pixels
      - background = non-cell pixels

    This function is deliberately kept standalone so that masks can be inspected
    by hand during development.

    Parameters
    ----------
    img_hwc:
        RGB image, HWC, float-like. Values are clipped to [0, 1].

    foreground_quantile:
        Pixels with signal >= this image-level quantile are foreground.

    background_quantile:
        Pixels with signal <= this image-level quantile are background.

    min_region_pixels:
        If a region has fewer pixels than this, that region is returned as empty.
        This avoids unstable quantile curves from tiny foreground regions.

    signal_mode:
        "max":
            signal = max(R, G, B)
        "luma":
            signal = 0.2126 R + 0.7152 G + 0.0722 B
        "green":
            signal = G
        "red_green_max":
            signal = max(R, G)

    morph_open:
        If True, apply a small morphological opening to foreground.
        Default False because tiny cells may be removed too aggressively.

    morph_kernel_size:
        Kernel size for optional foreground opening.

    Returns
    -------
    dict with boolean masks:
        {
            "all": all pixels,
            "foreground": bright cell-like pixels,
            "background": non-cell pixels,
        }
    """
    img = np.clip(np.asarray(img_hwc, dtype=np.float32), 0.0, 1.0)

    if not (0.0 <= foreground_quantile <= 1.0):
        raise ValueError(f"foreground_quantile must be in [0, 1], got {foreground_quantile}")

    if not (0.0 <= background_quantile <= 1.0):
        raise ValueError(f"background_quantile must be in [0, 1], got {background_quantile}")

    if background_quantile >= foreground_quantile:
        raise ValueError(
            "background_quantile should be smaller than foreground_quantile, "
            f"got background_quantile={background_quantile}, "
            f"foreground_quantile={foreground_quantile}"
        )

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {img.shape}")

    if signal_mode == "max":
        signal = np.max(img, axis=-1)
    elif signal_mode == "luma":
        signal = (
            0.2126 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2]
        )
    elif signal_mode == "green":
        signal = img[..., 1]
    elif signal_mode == "red_green_max":
        signal = np.maximum(img[..., 0], img[..., 1])
    else:
        raise ValueError(
            "signal_mode must be one of: 'max', 'luma', 'green', 'red_green_max'"
        )

    signal = np.asarray(signal, dtype=np.float32)
    finite = np.isfinite(signal)

    all_mask = np.ones(signal.shape, dtype=bool)

    if not finite.any():
        empty = np.zeros(signal.shape, dtype=bool)
        return {
            "all": all_mask,
            "foreground": empty,
            "background": empty,
        }

    valid_signal = signal[finite]

    fg_thr = float(np.quantile(valid_signal, foreground_quantile))
    bg_thr = float(np.quantile(valid_signal, background_quantile))

    foreground = finite & (signal >= fg_thr)
    background = finite & (signal <= bg_thr)

    if morph_open and foreground.any():
        k = int(morph_kernel_size)
        k = max(1, k)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)
        fg_u8 = foreground.astype(np.uint8)
        foreground = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel).astype(bool)

    if int(foreground.sum()) < int(min_region_pixels):
        foreground = np.zeros_like(foreground, dtype=bool)

    if int(background.sum()) < int(min_region_pixels):
        background = np.zeros_like(background, dtype=bool)

    return {
        "all": all_mask,
        "foreground": foreground,
        "background": background,
    }


def _sample_pixels_from_masked_region(
    img_hwc: np.ndarray,
    mask_hw: Optional[np.ndarray],
    sample_pixels: Optional[int],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    img = np.clip(np.asarray(img_hwc, dtype=np.float32), 0.0, 1.0)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {img.shape}")

    if mask_hw is None:
        pixels = img.reshape(-1, 3)
    else:
        mask = np.asarray(mask_hw, dtype=bool)
        if mask.shape != img.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {img.shape[:2]}"
            )
        if not mask.any():
            return None
        pixels = img[mask]

    n_total = pixels.shape[0]
    if n_total == 0:
        return None

    if sample_pixels is not None and n_total > sample_pixels:
        idx = rng.choice(n_total, size=int(sample_pixels), replace=False)
        pixels = pixels[idx]

    return pixels.astype(np.float32, copy=False)


def _compute_pixels_quantiles(
    pixels: np.ndarray,
    q_probs: np.ndarray,
) -> np.ndarray:
    q_img = []
    for c in range(3):
        qc = np.quantile(
            np.clip(pixels[:, c], 0.0, 1.0),
            q_probs,
        ).astype(np.float32)
        qc = np.maximum.accumulate(qc)
        q_img.append(qc)

    return np.stack(q_img, axis=0).astype(np.float32)


def _build_region_payload(
    quantile_list: list[np.ndarray],
    q_band_lo: float,
    q_band_hi: float,
    store_q_images: bool = True,
) -> Optional[Dict[str, Any]]:
    if len(quantile_list) == 0:
        return None

    Q = np.stack(quantile_list, axis=0).astype(np.float32)

    q_center = np.quantile(Q, 0.50, axis=0).astype(np.float32)
    q_lo = np.quantile(Q, q_band_lo, axis=0).astype(np.float32)
    q_hi = np.quantile(Q, q_band_hi, axis=0).astype(np.float32)

    q_center = np.maximum.accumulate(q_center, axis=1)
    q_lo = np.maximum.accumulate(q_lo, axis=1)
    q_hi = np.maximum.accumulate(q_hi, axis=1)
    q_center = np.clip(q_center, q_lo, q_hi)

    payload: Dict[str, Any] = {
        "n_images": int(Q.shape[0]),
        "q_center": q_center,
        "q_lo": q_lo,
        "q_hi": q_hi,
    }

    if store_q_images:
        payload["q_images"] = Q

    return payload


def build_target_quantile_band_cache(
    folders: Optional[Sequence[str | Path]] = None,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    devices: Sequence[str] = (
        "iphone",
        "googlepixel",
        "microscope",
        "monochrome_generic",
        "monochrome_real",
    ),
    recursive: bool = True,
    sample_pixels_per_image: Optional[int] = 100_000,
    n_quantiles: int = 256,
    q_band_lo: float = 0.025,
    q_band_hi: float = 0.975,
    rng_seed: int = 0,
    cache_path: Optional[str | Path] = STYLE_QUANTILE_CACHE_PATH,
    force_recompute: bool = False,
    use_regions: bool = True,
    foreground_quantile: Union[float, Dict[str, float], None] = None,
    background_quantile: Union[float, Dict[str, float], None] = None,
    min_region_pixels: int = 2048,
    foreground_signal_mode: str = "max",
    store_q_images: bool = True,
) -> Dict[str, Any]:
    """
    Build a device/style histogram cache from real reference images.

    The cache supports both old global matching and newer region-aware matching.

    Foreground/background masks are estimated from image brightness. Foreground
    means bright cell-like pixels.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    if foreground_quantile is None:
        foreground_quantile = FOREGROUND_QUANTILES

    if background_quantile is None:
        background_quantile = BACKGROUND_QUANTILES

    rng = np.random.default_rng(rng_seed)
    image_paths = _collect_image_paths(folders=folders, exts=exts, recursive=recursive)
    q_probs = np.linspace(0.0, 1.0, int(n_quantiles), dtype=np.float32)

    per_device_region_quantiles: Dict[str, Dict[str, list[np.ndarray]]] = {
        d: {region: [] for region in REGION_NAMES}
        for d in devices
    }
    counts = {d: 0 for d in devices}
    skipped: list[Dict[str, str]] = []

    for path in tqdm(image_paths, desc="Building quantile-band cache", unit="image"):
        device = _find_device_label(path)
        if device not in devices:
            continue

        file_name = os.path.basename(path)
        folder = os.path.dirname(path)

        try:
            img, report = load_image(
                file_name,
                base_dir=folder,
                as_chw=False,
                scale=True,
                fast_scale=True,
            )
        except Exception as e:
            skipped.append({"path": str(path), "reason": repr(e)})
            continue

        if img.ndim != 3 or img.shape[2] != 3:
            skipped.append({"path": str(path), "reason": f"bad shape {img.shape}"})
            continue

        img = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)

        if use_regions:
            fg_q = float(
                _resolve_device_value(
                    foreground_quantile,
                    device=device,
                    default=0.92,
                )
            )
            bg_q = float(
                _resolve_device_value(
                    background_quantile,
                    device=device,
                    default=0.80,
                )
            )

            masks = make_foreground_background_masks(
                img_hwc=img,
                foreground_quantile=fg_q,
                background_quantile=bg_q,
                min_region_pixels=min_region_pixels,
                signal_mode=foreground_signal_mode,
            )
        else:
            masks = {
                "all": np.ones(img.shape[:2], dtype=bool),
                "foreground": np.zeros(img.shape[:2], dtype=bool),
                "background": np.zeros(img.shape[:2], dtype=bool),
            }

        # Always build all-region cache.
        for region in REGION_NAMES:
            mask = masks[region]
            if region != "all" and not mask.any():
                continue

            pixels = _sample_pixels_from_masked_region(
                img_hwc=img,
                mask_hw=mask,
                sample_pixels=sample_pixels_per_image,
                rng=rng,
            )
            if pixels is None or pixels.shape[0] == 0:
                continue

            q_img = _compute_pixels_quantiles(pixels, q_probs)
            per_device_region_quantiles[device][region].append(q_img)

        counts[device] += 1

    devices_payload: Dict[str, Any] = {}

    for device in devices:
        regions_payload: Dict[str, Any] = {}

        for region in REGION_NAMES:
            payload = _build_region_payload(
                per_device_region_quantiles[device][region],
                q_band_lo=q_band_lo,
                q_band_hi=q_band_hi,
                store_q_images=store_q_images,
            )
            if payload is not None:
                regions_payload[region] = payload

        if "all" not in regions_payload:
            continue

        device_payload: Dict[str, Any] = {
            "n_images": int(counts[device]),
            "regions": regions_payload,
        }

        # Backward-compatible top-level fields.
        device_payload["q_center"] = regions_payload["all"]["q_center"]
        device_payload["q_lo"] = regions_payload["all"]["q_lo"]
        device_payload["q_hi"] = regions_payload["all"]["q_hi"]
        if "q_images" in regions_payload["all"]:
            device_payload["q_images"] = regions_payload["all"]["q_images"]

        devices_payload[device] = device_payload

    cache = {
        "version": 2,
        "devices": devices_payload,
        "q_probs": q_probs,
        "regions": REGION_NAMES,
        "devices_requested": tuple(devices),
        "folders": [str(f) for f in (folders if folders is not None else EXT_IMAGES_FOLDERS)],
        "sample_pixels_per_image": sample_pixels_per_image,
        "n_quantiles": int(n_quantiles),
        "q_band_lo": float(q_band_lo),
        "q_band_hi": float(q_band_hi),
        "use_regions": bool(use_regions),
        "foreground_quantile": foreground_quantile,
        "background_quantile": background_quantile,
        "min_region_pixels": int(min_region_pixels),
        "foreground_signal_mode": str(foreground_signal_mode),
        "store_q_images": bool(store_q_images),
        "skipped": skipped,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    return cache


def load_or_build_quantile_band_cache(
    folders: Optional[Sequence[str | Path]] = None,
    cache_path: str | Path = STYLE_QUANTILE_CACHE_PATH,
    force_recompute: bool = False,
    sample_pixels_per_image: Optional[int] = 100_000,
    n_quantiles: int = 256,
    q_band_lo: float = 0.025,
    q_band_hi: float = 0.975,
    use_regions: bool = True,
    foreground_quantile: Union[float, Dict[str, float], None] = None,
    background_quantile: Union[float, Dict[str, float], None] = None,
    min_region_pixels: int = 2048,
    foreground_signal_mode: str = "max",
    store_q_images: bool = True,
) -> Dict[str, Any]:
    return build_target_quantile_band_cache(
        folders=folders,
        cache_path=cache_path,
        force_recompute=force_recompute,
        sample_pixels_per_image=sample_pixels_per_image,
        n_quantiles=n_quantiles,
        q_band_lo=q_band_lo,
        q_band_hi=q_band_hi,
        use_regions=use_regions,
        foreground_quantile=foreground_quantile,
        background_quantile=background_quantile,
        min_region_pixels=min_region_pixels,
        foreground_signal_mode=foreground_signal_mode,
        store_q_images=store_q_images,
    )


def _compute_image_channel_quantiles(
    x: np.ndarray,
    q_probs: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != x.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match channel shape {x.shape}")
        if not mask.any():
            return np.zeros_like(q_probs, dtype=np.float32)
        x = x[mask]

    q = np.quantile(x, q_probs).astype(np.float32)
    return np.maximum.accumulate(q)


def _strictly_increasing_knots(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    y = np.maximum.accumulate(y)

    for i in range(1, y.size):
        if y[i] <= y[i - 1]:
            y[i] = y[i - 1] + eps

    return y


def _minimal_band_projection(
    src_q: np.ndarray,
    tgt_q_lo: np.ndarray,
    tgt_q_hi: np.ndarray,
) -> np.ndarray:
    proj = np.clip(src_q, tgt_q_lo, tgt_q_hi).astype(np.float32)
    return np.maximum.accumulate(proj)


def _sample_real_curve(
    region_ref: Dict[str, Any],
    channel: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if "q_images" not in region_ref:
        # Fall back safely if old cache was loaded.
        return region_ref["q_center"][channel].astype(np.float32)

    q_images = np.asarray(region_ref["q_images"], dtype=np.float32)
    if q_images.ndim != 3 or q_images.shape[1] != 3:
        raise ValueError(f"Invalid q_images shape: {q_images.shape}")

    idx = int(rng.integers(0, q_images.shape[0]))
    return q_images[idx, channel].astype(np.float32)


def _get_region_ref(
    device_ref: Dict[str, Any],
    region: str,
    fallback_to_all: bool = False,
) -> Dict[str, Any]:
    if "regions" not in device_ref:
        raise KeyError("device_ref is missing 'regions'.")

    regions = device_ref["regions"]

    if region in regions:
        return regions[region]

    if fallback_to_all and "all" in regions:
        return regions["all"]

    raise KeyError(
        f"Region '{region}' is missing. Available regions: {tuple(regions.keys())}"
    )

def _make_soft_cell_weight(
    cell_mask: np.ndarray,
    sigma: float = 1.5,
) -> np.ndarray:
    m = np.asarray(cell_mask, dtype=np.float32)

    if m.ndim == 3:
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif m.shape[0] == 1:
            m = m[0]
        else:
            raise ValueError(f"Expected 2D cell mask, got shape {cell_mask.shape}")

    if m.ndim != 2:
        raise ValueError(f"Expected 2D cell mask, got shape {cell_mask.shape}")

    m = np.clip(m, 0.0, 1.0)

    if sigma > 0:
        m = cv2.GaussianBlur(m, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))

    return np.clip(m.astype(np.float32), 0.0, 1.0)


def _apply_quantile_map_1d(
    x: np.ndarray,
    src_quantiles: np.ndarray,
    dst_quantiles: np.ndarray,
) -> np.ndarray:
    src_q = _strictly_increasing_knots(src_quantiles)
    dst_q = np.maximum.accumulate(np.asarray(dst_quantiles, dtype=np.float32))

    y = np.interp(
        np.asarray(x, dtype=np.float32),
        src_q,
        dst_q,
        left=dst_q[0],
        right=dst_q[-1],
    ).astype(np.float32)

    return np.clip(y, 0.0, 1.0)


def _adjust_image_to_region_reference(
    arr_hwc: np.ndarray,
    q_probs: np.ndarray,
    region_ref: Dict[str, Any],
    strength: float,
    rng: np.random.Generator,
    match_mode: str,
    source_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = arr_hwc.copy()

    for c in range(3):
        src_q = _compute_image_channel_quantiles(
            arr_hwc[..., c],
            q_probs,
            mask=source_mask,
        )

        if match_mode == "project_to_band":
            dst_q_full = _minimal_band_projection(
                src_q,
                region_ref["q_lo"][c],
                region_ref["q_hi"][c],
            )
        elif match_mode == "sample_real_curve":
            dst_q_full = _sample_real_curve(region_ref, c, rng=rng)
        elif match_mode == "center":
            dst_q_full = region_ref["q_center"][c].astype(np.float32)
        else:
            raise ValueError(
                "match_mode must be one of: "
                "'project_to_band', 'sample_real_curve', 'center'"
            )

        dst_q = ((1.0 - strength) * src_q + strength * dst_q_full).astype(np.float32)
        dst_q = np.maximum.accumulate(dst_q)
        out[..., c] = _apply_quantile_map_1d(arr_hwc[..., c], src_q, dst_q)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


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
    Move an RGB image toward the real-image intensity distribution of one device.

    Assumes a fresh region-aware cache:
        cache["devices"][device]["regions"][region]

    region_mode:
        "all":
            Global all-pixel histogram matching.

        "foreground_background":
            Requires cell_mask unless fallback_to_all=True.
            Foreground and background references are applied separately and
            softly blended with a blurred cell mask.
    """
    if target_device in {"simulated_raw", "raw_simulated"}:
        out = np.asarray(img, dtype=np.float32).copy()
        return np.clip(out, 0.0, 1.0)

    if "devices" not in quantile_band_cache:
        raise KeyError("quantile_band_cache is missing 'devices'.")

    if "q_probs" not in quantile_band_cache:
        raise KeyError("quantile_band_cache is missing 'q_probs'.")

    if target_device not in quantile_band_cache["devices"]:
        raise KeyError(f"Target device '{target_device}' not in quantile_band_cache.")

    strength = float(np.clip(strength, 0.0, 1.0))
    arr, input_chw = _as_hwc_rgb_float01(img, preserve_copy=True)

    if strength <= 0.0:
        out = arr
        if preserve_input_layout and input_chw:
            out = np.moveaxis(out, -1, 0)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    if rng is None:
        rng = np.random.default_rng()

    q_probs = np.asarray(quantile_band_cache["q_probs"], dtype=np.float32)
    device_ref = quantile_band_cache["devices"][target_device]

    if region_mode == "all":
        region_ref = _get_region_ref(
            device_ref,
            "all",
            fallback_to_all=False,
        )

        out = _adjust_image_to_region_reference(
            arr_hwc=arr,
            q_probs=q_probs,
            region_ref=region_ref,
            strength=strength,
            rng=rng,
            match_mode=match_mode,
            source_mask=None,
        )

    elif region_mode == "foreground_background":
        if cell_mask is None:
            if not fallback_to_all:
                raise ValueError(
                    "cell_mask is required for region_mode='foreground_background'."
                )

            region_ref = _get_region_ref(
                device_ref,
                "all",
                fallback_to_all=False,
            )

            out = _adjust_image_to_region_reference(
                arr_hwc=arr,
                q_probs=q_probs,
                region_ref=region_ref,
                strength=strength,
                rng=rng,
                match_mode=match_mode,
                source_mask=None,
            )

        else:
            w_fg = _make_soft_cell_weight(cell_mask, sigma=mask_blur_sigma)

            if w_fg.shape != arr.shape[:2]:
                raise ValueError(
                    f"cell_mask shape {w_fg.shape} does not match image shape {arr.shape[:2]}"
                )

            hard_fg = w_fg > 0.5
            hard_bg = ~hard_fg

            if not hard_fg.any():
                raise ValueError(
                    "cell_mask contains no foreground pixels; "
                    "cannot use foreground_background histogram matching."
                )

            if not hard_bg.any():
                raise ValueError(
                    "cell_mask contains no background pixels; "
                    "cannot use foreground_background histogram matching."
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

            out_fg = _adjust_image_to_region_reference(
                arr_hwc=arr,
                q_probs=q_probs,
                region_ref=fg_ref,
                strength=strength,
                rng=rng,
                match_mode=match_mode,
                source_mask=hard_fg,
            )

            out_bg = _adjust_image_to_region_reference(
                arr_hwc=arr,
                q_probs=q_probs,
                region_ref=bg_ref,
                strength=strength,
                rng=rng,
                match_mode=match_mode,
                source_mask=hard_bg,
            )

            w = w_fg[..., None].astype(np.float32)
            out = (1.0 - w) * out_bg + w * out_fg
            out = np.clip(out, 0.0, 1.0).astype(np.float32)

    else:
        raise ValueError("region_mode must be one of: 'all', 'foreground_background'.")

    if preserve_input_layout and input_chw:
        out = np.moveaxis(out, -1, 0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
