import numpy as np
import os
from typing import Optional, Tuple, Sequence, Dict

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import pickle

import cv2

from .config import STYLE_CACHE_PATH

from ...image_analysis.io import load_image

RNG = np.random.Generator


# folders with images for the mean+-STD calculation
EXT_IMAGES_FOLDERS = [
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251106_25722169_iPhone_XR_JPEG",
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251107_25065521_GooglePixel",
    "./ext_images/20251107_25722332_GooglePixel",
    "./ext_images/20251014_25719960",
    "./ext_images/20251014_25720084",
    "./ext_images/20251107_25065521",
    "./ext_images/20251107_25722332"
]

def _pair_from_center_width(center: float, width: float, lo: float, hi: float) -> Tuple[float, float]:
    a = max(lo, center - width)
    b = min(hi, center + width)
    if a > b:
        a, b = b, a
    return (float(a), float(b))


def _safe_range(a: float, b: float, lo: Optional[float] = None, hi: Optional[float] = None) -> Tuple[float, float]:
    x0 = float(min(a, b))
    x1 = float(max(a, b))
    if lo is not None:
        x0 = max(lo, x0)
        x1 = max(lo, x1)
    if hi is not None:
        x0 = min(hi, x0)
        x1 = min(hi, x1)
    return (float(x0), float(x1))

@dataclass
class CameraStyleParams:
    name: str

    # basic tone
    exposure_range: Tuple[float, float] = (1.0, 1.0)
    c_range: Tuple[float, float] = (1.0, 1.0)
    b_range: Tuple[float, float] = (0.0, 0.0)
    gamma_range: Tuple[float, float] = (1.0, 1.0)

    # nonlinear tone
    shadow_lift_range: Tuple[float, float] = (0.0, 0.0)
    highlight_rolloff_range: Tuple[float, float] = (0.0, 0.0)
    midtone_contrast_range: Tuple[float, float] = (0.0, 0.0)

    # color
    mix_range: Tuple[float, float] = (0.0, 0.0)
    wb_range: Tuple[float, float] = (1.0, 1.0)
    saturation_range: Tuple[float, float] = (1.0, 1.0)
    green_magenta_shift_range: Tuple[float, float] = (0.0, 0.0)
    blue_yellow_shift_range: Tuple[float, float] = (0.0, 0.0)

    # blur / sharpening / noise
    blur_sigma_range: Tuple[float, float] = (0.0, 0.0)
    sharpen_strength_range: Tuple[float, float] = (0.0, 0.0)
    noise_std_base_range: Tuple[float, float] = (0.0, 0.0)

    # uneven field
    vignette_amp_range: Tuple[float, float] = (0.0, 0.0)
    illum_amp_range: Tuple[float, float] = (0.0, 0.0)

    # clipping / compression
    clip_prob: float = 0.0
    jpeg_prob: float = 0.0
    jpeg_quality_range: Tuple[int, int] = (60, 95)

    # resampling artifacts
    resize_prob: float = 0.0
    resize_scale_range: Tuple[float, float] = (1.0, 1.0)


@dataclass
class CameraStyleConfig:
    styles: Sequence[str] = ("microscope", "iphone", "googlepixel")
    probs: Optional[Sequence[float]] = None

    def sample_style(self, rng: RNG) -> str:
        if len(self.styles) == 1:
            return self.styles[0]

        if self.probs is None:
            idx = int(rng.integers(0, len(self.styles)))
            return self.styles[idx]

        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        idx = int(rng.choice(len(self.styles), p=p))
        return self.styles[idx]

def build_camera_style_from_summary(
    phone: str,
    summary: dict,
) -> CameraStyleParams:
    """
    Map robust feature summaries to a CameraStyleParams.
    """
    med = summary["feature_median"]
    qlo = summary["feature_q_lo"]
    qhi = summary["feature_q_hi"]

    gray_mean_med = med["gray_mean"]
    gray_std_med = med["gray_std"]
    sat_mean_med = med["sat_mean"]
    dark_frac_med = med["dark_frac"]
    bright_frac_med = med["bright_frac"]

    r_mean = med["r_mean"]
    g_mean = med["g_mean"]
    b_mean = med["b_mean"]

    gray_p05 = med.get("gray_p5", gray_mean_med)
    gray_p50 = med.get("gray_p50", gray_mean_med)
    gray_p95 = med.get("gray_p95", gray_mean_med)

    p05_spread = qhi.get("gray_p5", gray_p05) - qlo.get("gray_p5", gray_p05)
    p50_spread = qhi.get("gray_p50", gray_p50) - qlo.get("gray_p50", gray_p50)
    p95_spread = qhi.get("gray_p95", gray_p95) - qlo.get("gray_p95", gray_p95)

    tonal_spread = max(1e-6, gray_p95 - gray_p05)

    # color bias signals
    rg_bias = r_mean - g_mean
    bg_bias = b_mean - g_mean
    rb_bias = r_mean - b_mean

    # exposure
    # brighter families get slightly wider exposure range near 1
    exposure_center = 1.0 + 0.35 * (gray_mean_med - 0.35)
    exposure_width = 0.06 + 0.45 * p50_spread
    exposure_range = _pair_from_center_width(exposure_center, exposure_width, 0.75, 1.35)

    # contrast
    contrast_center = 1.0 + 0.8 * (gray_std_med - 0.18)
    contrast_width = 0.04 + 0.30 * max(0.01, qhi["gray_std"] - qlo["gray_std"])
    c_range = _pair_from_center_width(contrast_center, contrast_width, 0.80, 1.25)

    # brightness offset
    b_center = 0.15 * (gray_mean_med - 0.35)
    b_width = 0.01 + 0.10 * p50_spread
    b_range = _pair_from_center_width(b_center, b_width, -0.08, 0.08)

    # gamma
    # broad tonal compression handled here
    gamma_center = 1.0 - 0.5 * (gray_mean_med - 0.35) + 0.2 * (0.18 - gray_std_med)
    gamma_width = 0.05 + 0.30 * p95_spread
    gamma_range = _pair_from_center_width(gamma_center, gamma_width, 0.75, 1.30)

    # shadow lift
    # high dark_frac or elevated low percentile floor => more lift
    shadow_center = 0.03 + 0.25 * dark_frac_med + 0.12 * gray_p05
    shadow_width = 0.01 + 0.08 * p05_spread
    shadow_lift_range = _pair_from_center_width(shadow_center, shadow_width, 0.0, 0.25)

    # highlight rolloff
    highlight_center = 0.02 + 0.35 * bright_frac_med + 0.25 * max(0.0, gray_p95 - 0.80)
    highlight_width = 0.01 + 0.10 * p95_spread
    highlight_rolloff_range = _pair_from_center_width(highlight_center, highlight_width, 0.0, 0.35)

    # midtone contrast
    midtone_center = 1.5 * (gray_std_med - 0.18)
    midtone_width = 0.03 + 0.25 * max(0.01, qhi["gray_std"] - qlo["gray_std"])
    midtone_contrast_range = _pair_from_center_width(midtone_center, midtone_width, -0.25, 0.40)

    # RG mixing
    mix_center = 0.04 + 0.15 * abs(rg_bias)
    mix_width = 0.01 + 0.05 * abs(qhi["r_mean"] - qlo["r_mean"])
    mix_range = _pair_from_center_width(mix_center, mix_width, 0.0, 0.25)

    # WB spread
    wb_center = 1.0
    wb_width = 0.02 + 0.6 * max(
        abs(qhi["r_mean"] - qlo["r_mean"]),
        abs(qhi["g_mean"] - qlo["g_mean"]),
        abs(qhi["b_mean"] - qlo["b_mean"]),
    )
    wb_range = _pair_from_center_width(wb_center, wb_width, 0.80, 1.25)

    # saturation
    sat_center = 1.0 + 0.7 * (sat_mean_med - 0.30)
    sat_width = 0.05 + 0.8 * max(0.01, qhi["sat_mean"] - qlo["sat_mean"])
    saturation_range = _pair_from_center_width(sat_center, sat_width, 0.70, 1.35)

    # explicit color-axis shifts
    green_magenta_center = np.clip(-0.8 * (0.5 * (r_mean + b_mean) - g_mean), -0.12, 0.12)
    green_magenta_width = 0.01 + 0.03 * (
        abs(qhi["g_mean"] - qlo["g_mean"]) +
        0.5 * abs(qhi["r_mean"] - qlo["r_mean"]) +
        0.5 * abs(qhi["b_mean"] - qlo["b_mean"])
    )
    green_magenta_shift_range = _pair_from_center_width(green_magenta_center, green_magenta_width, -0.15, 0.15)

    blue_yellow_center = np.clip(-0.8 * rb_bias, -0.15, 0.15)
    blue_yellow_width = 0.01 + 0.04 * (
        abs(qhi["r_mean"] - qlo["r_mean"]) +
        abs(qhi["b_mean"] - qlo["b_mean"])
    )
    blue_yellow_shift_range = _pair_from_center_width(blue_yellow_center, blue_yellow_width, -0.18, 0.18)

    # clip probability
    clip_prob = float(np.clip(
        0.05 + 1.4 * bright_frac_med + 0.2 * max(0.0, gray_p95 - 0.90),
        0.0, 0.75
    ))

    # The parts below are not well estimated from histograms alone.
    # Keep them device-sensitive but conservative.
    if phone == "iphone":
        blur_sigma_range = (0.7, 1.6)
        sharpen_strength_range = (0.35, 0.85)
        noise_std_base_range = (0.008, 0.020)
        vignette_amp_range = (0.04, 0.14)
        illum_amp_range = (0.04, 0.10)
        jpeg_prob = 0.3
        jpeg_quality_range = (70, 95)
        resize_prob = 0.30
        resize_scale_range = (0.60, 0.90)
    elif phone == "googlepixel":
        blur_sigma_range = (0.5, 1.4)
        sharpen_strength_range = (0.25, 0.70)
        noise_std_base_range = (0.006, 0.016)
        vignette_amp_range = (0.03, 0.12)
        illum_amp_range = (0.03, 0.09)
        jpeg_prob = 0.30
        jpeg_quality_range = (70, 95)
        resize_prob = 0.25
        resize_scale_range = (0.65, 0.92)
    elif phone == "microscope":
        # broad cloud, but shifted toward the real microscope region
        exposure_range = (0.72, 1.02)
        c_range = (0.82, 1.06)
        b_range = (-0.05, 0.01)
        gamma_range = (1.02, 1.30)
        shadow_lift_range = (0.00, 0.08)
        highlight_rolloff_range = (0.00, 0.08)
        midtone_contrast_range = (-0.12, 0.08)
        mix_range = (0.00, 0.05)
        wb_range = (0.90, 1.06)
        saturation_range = (0.72, 0.98)
        green_magenta_shift_range = (-0.05, 0.08)
        blue_yellow_shift_range = (-0.08, 0.03)
        blur_sigma_range = (0.20, 0.90)
        sharpen_strength_range = (0.00, 0.18)
        noise_std_base_range = (0.001, 0.008)
        vignette_amp_range = (0.00, 0.03)
        illum_amp_range = (0.00, 0.05)
        clip_prob = 0.02
        jpeg_prob = 0.0
        jpeg_quality_range = (95, 100)
        resize_prob = 0.0
        resize_scale_range = (1.0, 1.0)
    else:
        blur_sigma_range = (0.5, 1.2)
        sharpen_strength_range = (0.10, 0.50)
        noise_std_base_range = (0.005, 0.015)
        vignette_amp_range = (0.02, 0.10)
        illum_amp_range = (0.03, 0.08)
        jpeg_prob = 0.30
        jpeg_quality_range = (70, 95)
        resize_prob = 0.20
        resize_scale_range = (0.70, 0.95)

    return CameraStyleParams(
        name=phone,
        exposure_range=_safe_range(*exposure_range, lo=0.75, hi=1.35),
        c_range=_safe_range(*c_range, lo=0.80, hi=1.25),
        b_range=_safe_range(*b_range, lo=-0.08, hi=0.08),
        gamma_range=_safe_range(*gamma_range, lo=0.75, hi=1.30),
        shadow_lift_range=_safe_range(*shadow_lift_range, lo=0.0, hi=0.25),
        highlight_rolloff_range=_safe_range(*highlight_rolloff_range, lo=0.0, hi=0.35),
        midtone_contrast_range=_safe_range(*midtone_contrast_range, lo=-0.25, hi=0.40),
        mix_range=_safe_range(*mix_range, lo=0.0, hi=0.25),
        wb_range=_safe_range(*wb_range, lo=0.80, hi=1.25),
        saturation_range=_safe_range(*saturation_range, lo=0.70, hi=1.35),
        green_magenta_shift_range=_safe_range(*green_magenta_shift_range, lo=-0.15, hi=0.15),
        blue_yellow_shift_range=_safe_range(*blue_yellow_shift_range, lo=-0.18, hi=0.18),
        blur_sigma_range=blur_sigma_range,
        sharpen_strength_range=sharpen_strength_range,
        noise_std_base_range=noise_std_base_range,
        vignette_amp_range=vignette_amp_range,
        illum_amp_range=illum_amp_range,
        clip_prob=clip_prob,
        jpeg_prob=jpeg_prob,
        jpeg_quality_range=jpeg_quality_range,
        resize_prob=resize_prob,
        resize_scale_range=resize_scale_range,
    )

def extract_real_image_feature_table_cv2(
    folders: Optional[Sequence[str | Path]] = None,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    recursive: bool = True,
    ignore_failures: bool = True,
    cache_path: Optional[str | Path] = STYLE_CACHE_PATH,
    force_recompute: bool = False,
    rng_seed: int = 0,
):
    """
    Extract per-image appearance features from real images.

    Returns
    -------
    rows : list[dict]
    feature_names : list[str]
    X : np.ndarray
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            return payload["rows"], payload["feature_names"], payload["X"]

    if folders is None:
        folders = EXT_IMAGES_FOLDERS
        if folders is None:
            raise ValueError("folders must be provided")

    rng = np.random.default_rng(rng_seed)
    folders = [Path(f) for f in folders]
    exts = tuple(e.lower() for e in exts)
    percentiles = tuple(float(p) for p in percentiles)

    image_paths = []
    for folder in folders:
        if not folder.exists():
            continue
        walker = folder.rglob("*") if recursive else folder.glob("*")
        for path in walker:
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)

    if not image_paths:
        raise RuntimeError("No image files found in the given folders.")

    channel_names = ("r", "g", "b")
    feature_names = []

    for ch in channel_names:
        feature_names.append(f"{ch}_mean")
        feature_names.append(f"{ch}_std")
        feature_names.append(f"{ch}_skew")
        for p in percentiles:
            p_name = str(p).replace(".", "_")
            feature_names.append(f"{ch}_p{p_name}")
        for b in range(hist_bins):
            feature_names.append(f"{ch}_hist_{b:02d}")

    feature_names.extend([
        "gray_mean",
        "gray_std",
        "gray_skew",
    ])
    for p in percentiles:
        p_name = str(p).replace(".", "_")
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

    rows = []

    def _safe_skew(x: np.ndarray) -> float:
        m = float(x.mean())
        s = float(x.std())
        if s < 1e-12:
            return 0.0
        z = (x - m) / s
        return float(np.mean(z ** 3))

    def _norm_hist_01(x: np.ndarray, bins: int) -> np.ndarray:
        h, _ = np.histogram(x, bins=bins, range=(0.0, 1.0))
        h = h.astype(np.float32)
        h /= (h.sum() + 1e-8)
        return h

    def _find_phone(file_path: str | Path) -> str:
        s = str(file_path).lower()
        if "iphone" in s:
            return "iphone"
        if "googlepixel" in s or "pixel" in s:
            return "googlepixel"
        else:
            return "microscope"

    for path in tqdm(image_paths, desc="Extracting real-image features"):
        file_name = os.path.basename(path)
        folder = os.path.dirname(path)

        phone = _find_phone(path)

        img, _ = load_image(file_name, base_dir = folder)

        H, W, _ = img.shape

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
        sat = np.where(rgb_max > 1e-8, (rgb_max - rgb_min) / (rgb_max + 1e-8), 0.0).astype(np.float32)

        row = {
            "path": str(path),
            "filename": path.name,
            "phone": phone,
            "height": int(H),
            "width": int(W),
            "aspect_ratio": float(W / max(H, 1)),
            "n_pixels_used": int(pixels_use.shape[0]),
        }

        for ch_name, x in zip(channel_names, (r, g, b)):
            row[f"{ch_name}_mean"] = float(np.mean(x))
            row[f"{ch_name}_std"] = float(np.std(x))
            row[f"{ch_name}_skew"] = float(_safe_skew(x))

            pvals = np.percentile(x, percentiles)
            for p, val in zip(percentiles, pvals):
                p_name = str(p).replace(".", "_")
                row[f"{ch_name}_p{p_name}"] = float(val)

            hist = _norm_hist_01(x, bins=hist_bins)
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

        row["dark_frac"] = float(np.mean(gray <= 0.02))
        row["bright_frac"] = float(np.mean(gray >= 0.98))

        rows.append(row)

    if not rows:
        raise RuntimeError("No usable images were processed.")

    X = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "rows": rows,
                    "feature_names": feature_names,
                    "X": X,
                    "folders": [str(f) for f in folders],
                    "hist_bins": hist_bins,
                    "percentiles": percentiles,
                    "sample_pixels": sample_pixels,
                },
                f,
            )

    return rows, feature_names, X

def summarize_features_by_phone(
    rows: list[dict],
    feature_names: Sequence[str],
    phones: Sequence[str] = ("iphone", "googlepixel", "microscope"),
    q_lo: float = 0.10,
    q_hi: float = 0.90,
):
    """
    Build robust per-phone summaries.

    Returns
    -------
    summaries : dict[str, dict]
        summaries[phone] contains:
        - n_images
        - fraction
        - mean / std / median / q_lo / q_hi for each feature
    """
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


def build_style_registry_from_real_images(
    folders: Optional[Sequence[str | Path]],
    cache_path: str | Path = STYLE_CACHE_PATH,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    phones: Sequence[str] = ("iphone", "googlepixel", "microscope"),
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    force_recompute: bool = False,
):
    rows, feature_names, _ = extract_real_image_feature_table_cv2(
        folders=folders,
        exts=exts,
        hist_bins=hist_bins,
        percentiles=percentiles,
        sample_pixels=sample_pixels,
        cache_path=cache_path,
        force_recompute=force_recompute,
    )

    summaries = summarize_features_by_phone(
        rows=rows,
        feature_names=feature_names,
        phones=phones,
        q_lo=q_lo,
        q_hi=q_hi,
    )

    style_registry: Dict[str, CameraStyleParams] = {}
    for phone in phones:
        if phone not in summaries:
            continue
        style_registry[phone] = build_camera_style_from_summary(phone, summaries[phone])

    style_registry["simulated_raw"] = CameraStyleParams(
        name="simulated_raw",
        exposure_range=(1.0, 1.0),
        c_range=(1.0, 1.0),
        b_range=(0.0, 0.0),
        gamma_range=(1.0, 1.0),
        shadow_lift_range=(0.0, 0.0),
        highlight_rolloff_range=(0.0, 0.0),
        midtone_contrast_range=(0.0, 0.0),
        mix_range=(0.0, 0.0),
        wb_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        green_magenta_shift_range=(0.0, 0.0),
        blue_yellow_shift_range=(0.0, 0.0),
        blur_sigma_range=(0.0, 0.0),
        sharpen_strength_range=(0.0, 0.0),
        noise_std_base_range=(0.0, 0.0),
        vignette_amp_range=(0.0, 0.0),
        illum_amp_range=(0.0, 0.0),
        clip_prob=0.0,
        jpeg_prob=0.0,
        jpeg_quality_range=(100, 100),
        resize_prob=0.0,
        resize_scale_range=(1.0, 1.0),
    )

    return style_registry, summaries, rows, feature_names

STYLE_PARAMS_REGISTRY: Dict[str, CameraStyleParams] = {}

def load_or_build_default_style_registry(
    folders: Optional[Sequence[str | Path]] = None,
    cache_path: str | Path = STYLE_CACHE_PATH,
    force_recompute: bool = False,
):
    global STYLE_PARAMS_REGISTRY
    STYLE_PARAMS_REGISTRY, summaries, rows, feature_names = build_style_registry_from_real_images(
        folders=folders,
        cache_path=cache_path,
        force_recompute=force_recompute,
    )
    return STYLE_PARAMS_REGISTRY, summaries, rows, feature_names

def phone_mix_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("microscope", "iphone", "googlepixel"),
        probs=None,
    )

def microscope_only_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("microscope",),
        probs=None,
    )

def simulated_raw_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("simulated_raw",),
        probs=None,
    )
