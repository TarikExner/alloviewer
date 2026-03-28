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

from ...image_analysis.io import load_image
from .config import STYLE_CACHE_PATH, UNET_MEAN, UNET_STD
from .camera_styles import EXT_IMAGES_FOLDERS


# -----------------------------------------------------------------------------
# labels / paths
# -----------------------------------------------------------------------------

def _find_device_label(file_path: str | Path) -> str:
    s = str(file_path).lower()
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

    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue

        walker = folder.rglob("*") if recursive else folder.glob("*")
        for path in walker:
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)

    if not image_paths:
        raise RuntimeError("No image files found.")

    return sorted(image_paths)


# -----------------------------------------------------------------------------
# normalization helpers
# -----------------------------------------------------------------------------

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
    imgs_norm: [T,3,H,W], already normalized with UNET_MEAN/UNET_STD
    returns : [T,3,H,W] in image space
    """
    mean = torch.as_tensor(UNET_MEAN, dtype=imgs_norm.dtype, device=imgs_norm.device).view(1, 3, 1, 1)
    std = torch.as_tensor(UNET_STD, dtype=imgs_norm.dtype, device=imgs_norm.device).view(1, 3, 1, 1)
    return imgs_norm * std + mean


# -----------------------------------------------------------------------------
# feature extraction
# -----------------------------------------------------------------------------

def _safe_skew(x: np.ndarray) -> float:
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
) -> dict:
    """
    Extract one feature row from one RGB image.

    Supports:
      - HWC [H,W,3]
      - CHW [3,H,W]

    Parameters
    ----------
    hist_range
        Histogram range for channel histograms.
        Use (0,1) for image-space features.
        Use something like (-3,3) for normalized-image features.
        Use None to adapt range to each image/channel.
    dark_threshold, bright_threshold
        Thresholds for gray-level fractions.
        Set to None to disable in normalized space.
    clip_input
        If True and hist_range is not None, clip the image to hist_range first.
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

    pixels = img.reshape(-1, 3)
    n_total = pixels.shape[0]

    if sample_pixels is not None and n_total > sample_pixels:
        idx = rng.choice(n_total, size=sample_pixels, replace=False)
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
        "filename": path_label,
        "phone": phone_label,
        "height": int(H),
        "width": int(W),
        "aspect_ratio": float(W / max(H, 1)),
        "n_pixels_used": int(pixels_use.shape[0]),
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

    row["dark_frac"] = float(np.mean(gray <= dark_threshold)) if dark_threshold is not None else np.nan
    row["bright_frac"] = float(np.mean(gray >= bright_threshold)) if bright_threshold is not None else np.nan

    return row


# -----------------------------------------------------------------------------
# cache path helpers
# -----------------------------------------------------------------------------

def get_feature_cache_path(
    cache_path: str | Path = STYLE_CACHE_PATH,
    normalized: bool = False,
) -> Path:
    cache_path = Path(cache_path)
    if normalized:
        return cache_path.with_name(f"{cache_path.stem}_normalized{cache_path.suffix}")
    return cache_path


# -----------------------------------------------------------------------------
# real-image feature tables
# -----------------------------------------------------------------------------

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
):
    """
    Build and cache feature stats from real images.

    Two separate caches are supported:
      - normalize_imgs=False : image-space features
      - normalize_imgs=True  : normalized-space features
    """
    final_cache_path = get_feature_cache_path(cache_path=cache_path, normalized=normalize_imgs)

    if final_cache_path.exists() and not force_recompute:
        with open(final_cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload["rows"], payload["feature_names"], payload["X"]

    rng = np.random.default_rng(rng_seed)
    percentiles = tuple(float(p) for p in percentiles)
    image_paths = _collect_image_paths(folders=folders, exts=exts, recursive=recursive)
    feature_names = _build_feature_names(hist_bins=hist_bins, percentiles=percentiles)

    rows = []

    for path in tqdm(image_paths, desc="Extracting real-image features"):
        try:
            file_name = os.path.basename(path)
            folder = os.path.dirname(path)

            img, _ = load_image(
                file_name,
                base_dir=folder,
                as_chw=False,
                scale=True,
                fast_scale=True,
            )
        except Exception:
            if ignore_failures:
                continue
            raise

        if img.ndim != 3 or img.shape[2] != 3:
            if ignore_failures:
                continue
            raise RuntimeError(f"Bad image shape: {path} -> {img.shape}")

        if normalize_imgs:
            img = normalize_rgb_image_with_unet(img)
            row = extract_feature_row_from_rgb_image(
                img=img,
                hist_bins=hist_bins,
                percentiles=percentiles,
                sample_pixels=sample_pixels,
                rng=rng,
                phone_label=_find_device_label(path),
                path_label=str(path),
                hist_range=normalized_hist_range,
                dark_threshold=None,
                bright_threshold=None,
                clip_input=True,
            )
        else:
            row = extract_feature_row_from_rgb_image(
                img=img,
                hist_bins=hist_bins,
                percentiles=percentiles,
                sample_pixels=sample_pixels,
                rng=rng,
                phone_label=_find_device_label(path),
                path_label=str(path),
                hist_range=(0.0, 1.0),
                dark_threshold=0.02,
                bright_threshold=0.98,
                clip_input=True,
            )

        row["filename"] = path.name
        rows.append(row)

    if not rows:
        raise RuntimeError("No usable images were processed.")

    X = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)

    final_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_cache_path, "wb") as f:
        pickle.dump(
            {
                "rows": rows,
                "feature_names": feature_names,
                "X": X,
                "folders": [str(f) for f in (folders if folders is not None else EXT_IMAGES_FOLDERS)],
                "hist_bins": hist_bins,
                "percentiles": percentiles,
                "sample_pixels": sample_pixels,
                "normalize_imgs": normalize_imgs,
                "normalized_hist_range": normalized_hist_range,
            },
            f,
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
    )


# -----------------------------------------------------------------------------
# summaries / mismatch tables
# -----------------------------------------------------------------------------

def summarize_features_by_phone(
    rows: list[dict],
    feature_names: Sequence[str],
    phones: Sequence[str] = ("iphone", "googlepixel", "microscope"),
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
        subset = [row for row in rows if row.get("phone") == phone]
        if len(subset) == 0:
            continue

        X = np.array([[row[f] for f in feature_names] for row in subset], dtype=np.float64)

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
    group_names: Sequence[str] = ("iphone", "googlepixel", "microscope"),
    q_lo: float = 0.10,
    q_hi: float = 0.90,
) -> list[dict]:
    out = []

    for group in group_names:
        real_sub = [r for r in real_rows if str(r.get("phone", "")).lower() == group]
        syn_sub = [r for r in synthetic_rows if str(r.get("phone", "")).lower() == group]

        if len(real_sub) == 0 or len(syn_sub) == 0:
            continue

        XR = np.array([[r[f] for f in feature_names] for r in real_sub], dtype=np.float64)
        XS = np.array([[r[f] for f in feature_names] for r in syn_sub], dtype=np.float64)

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


# -----------------------------------------------------------------------------
# dataset helpers
# -----------------------------------------------------------------------------

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

    for i in range(int(n_synthetic)):
        imgs_t, _, extras = dataset[i]
        style_name = _extract_style_name_from_extras(extras)

        if normalized_features:
            imgs_work = imgs_t.detach()
        else:
            imgs_work = denormalize_dataset_imgs(imgs_t.detach())

        if use_first_tile_only:
            tile_indices = [0]
        else:
            tile_indices = list(range(imgs_work.shape[0]))

        for t_idx in tile_indices:
            img = imgs_work[t_idx].permute(1, 2, 0).cpu().numpy().astype(np.float32)

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


# -----------------------------------------------------------------------------
# PCA plotting
# -----------------------------------------------------------------------------

def plot_real_and_synthetic_pca(
    dataset,
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
):
    """
    PCA comparison between real images and dataset images.

    Parameters
    ----------
    normalized
        False:
            real cache is image-space cache
            dataset images are denormalized before feature extraction

        True:
            real cache is normalized-image cache
            dataset images are used as stored
    """
    final_cache_path = get_feature_cache_path(cache_path=cache_path, normalized=normalized)

    if not final_cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {final_cache_path}")

    with open(final_cache_path, "rb") as f:
        payload = pickle.load(f)

    real_rows = payload["rows"]
    feature_names = payload["feature_names"]

    if feature_subset is None:
        if drop_size_features:
            drop = {"height", "width", "aspect_ratio", "n_pixels_used"}
            feature_names_used = [f for f in feature_names if f not in drop]
        else:
            feature_names_used = list(feature_names)
    else:
        feature_names_used = list(feature_subset)

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

    all_rows = list(real_rows) + list(synthetic_rows)
    X = np.array([[row[f] for f in feature_names_used] for row in all_rows], dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=rng_seed)
    X_pca = pca.fit_transform(X_scaled)

    labels = [str(row["phone"]).lower() for row in all_rows]
    is_real = np.array([not str(row["path"]).startswith("<synthetic_") for row in all_rows])

    fig, ax = plt.subplots(figsize=(10, 8))

    real_devices = ["iphone", "googlepixel", "microscope"]
    syn_devices = ["iphone", "googlepixel", "microscope", "synthetic"]

    for dev in real_devices:
        idx = [i for i, lab in enumerate(labels) if lab == dev and is_real[i]]
        if idx:
            pts = X_pca[idx]
            ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha_real, s=s_real, label=f"real {dev}")

    syn_markers = {
        "iphone": "x",
        "googlepixel": "^",
        "microscope": "s",
        "synthetic": "D",
    }

    for dev in syn_devices:
        idx = [i for i, lab in enumerate(labels) if lab == dev and not is_real[i]]
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
        title = "PCA of normalized real vs normalized synthetic image features" if normalized \
            else "PCA of real vs denormalized synthetic image features"

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "cache_path_used": str(final_cache_path),
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
