from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np

from .config import STYLE_CACHE_PATH
from ...image_analysis.io import load_image

RNG = np.random.Generator


EXT_IMAGES_FOLDERS = [
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251106_25722169_iPhone_XR_JPEG",
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251107_25065521_GooglePixel",
    "./ext_images/20251107_25722332_GooglePixel",
    "./ext_images/20251014_25719960",
    "./ext_images/20251014_25720084",
    "./ext_images/20251107_25065521",
    "./ext_images/20251107_25722332",
]


# -----------------------------
# cache paths
# -----------------------------

STYLE_CACHE_PATH = Path(STYLE_CACHE_PATH)
STYLE_QUANTILE_CACHE_PATH = STYLE_CACHE_PATH.with_name("camera_quantile_band_cache.pkl")


# -----------------------------
# style params
# -----------------------------

@dataclass
class CameraStyleParams:
    name: str

    # global tone
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

    # blur / sharpen / noise
    blur_sigma_range: Tuple[float, float] = (0.0, 0.0)
    sharpen_strength_range: Tuple[float, float] = (0.0, 0.0)
    noise_std_base_range: Tuple[float, float] = (0.0, 0.0)

    # uneven field
    vignette_amp_range: Tuple[float, float] = (0.0, 0.0)
    illum_amp_range: Tuple[float, float] = (0.0, 0.0)

    # compression
    clip_prob: float = 0.0
    jpeg_prob: float = 0.0
    jpeg_quality_range: Tuple[int, int] = (60, 95)

    # resize artifacts
    resize_prob: float = 0.0
    resize_scale_range: Tuple[float, float] = (1.0, 1.0)

    # soft histogram band match
    histogram_match_strength_range: Tuple[float, float] = (0.0, 0.0)
    use_histogram_match: bool = True

    # median matching
    median_match_strength: float = 0.0


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


# -----------------------------
# fixed style presets
# -----------------------------

IPHONE_STYLE = CameraStyleParams(
    name="iphone",
    exposure_range=(0.98, 1.04),
    c_range=(0.90, 0.99),
    b_range=(0.01, 0.035),
    gamma_range=(1.00, 1.03),

    shadow_lift_range=(0.05, 0.10),
    highlight_rolloff_range=(0.10, 0.18),
    midtone_contrast_range=(-0.03, 0.02),

    mix_range=(0.01, 0.04),
    wb_range=(0.98, 1.03),
    saturation_range=(0.76, 0.96),
    green_magenta_shift_range=(-0.015, 0.015),
    blue_yellow_shift_range=(-0.015, 0.015),

    blur_sigma_range=(0.18, 0.45),
    sharpen_strength_range=(0.02, 0.08),
    noise_std_base_range=(0.002, 0.005),

    vignette_amp_range=(0.00, 0.02),
    illum_amp_range=(0.00, 0.015),

    clip_prob=0.00,
    jpeg_prob=0.03,
    jpeg_quality_range=(92, 99),

    resize_prob=0.00,
    resize_scale_range=(0.95, 1.00),

    histogram_match_strength_range=(0.35, 0.70),
    use_histogram_match=True,

)

GOOGLEPIXEL_STYLE = CameraStyleParams(
    name="googlepixel",
    exposure_range=(0.90, 1.08),
    c_range=(0.90, 1.10),
    b_range=(-0.01, 0.03),
    gamma_range=(0.90, 1.08),

    shadow_lift_range=(0.02, 0.08),
    highlight_rolloff_range=(0.03, 0.10),
    midtone_contrast_range=(-0.02, 0.10),

    mix_range=(0.02, 0.10),
    wb_range=(0.94, 1.08),
    saturation_range=(0.75, 1.00),
    green_magenta_shift_range=(-0.04, 0.04),
    blue_yellow_shift_range=(-0.05, 0.05),

    blur_sigma_range=(0.35, 1.00),
    sharpen_strength_range=(0.12, 0.40),
    noise_std_base_range=(0.004, 0.012),

    vignette_amp_range=(0.02, 0.08),
    illum_amp_range=(0.02, 0.06),

    clip_prob=0.08,
    jpeg_prob=0.15,
    jpeg_quality_range=(82, 98),

    resize_prob=0.10,
    resize_scale_range=(0.80, 0.96),

    histogram_match_strength_range=(0.35, 0.70),
    use_histogram_match=True,
)

MICROSCOPE_STYLE = CameraStyleParams(
    name="microscope",
    exposure_range=(0.76, 0.98),
    c_range=(0.90, 1.02),
    b_range=(-0.03, 0.005),
    gamma_range=(0.98, 1.12),

    shadow_lift_range=(0.00, 0.015),
    highlight_rolloff_range=(0.00, 0.025),
    midtone_contrast_range=(-0.04, 0.025),

    mix_range=(0.00, 0.015),
    wb_range=(0.985, 1.015),
    saturation_range=(0.24, 0.52),
    green_magenta_shift_range=(-0.01, 0.01),
    blue_yellow_shift_range=(-0.05, -0.01),

    blur_sigma_range=(0.08, 0.28),
    sharpen_strength_range=(0.00, 0.035),
    noise_std_base_range=(0.0002, 0.0012),

    vignette_amp_range=(0.00, 0.008),
    illum_amp_range=(0.00, 0.01),

    clip_prob=0.00,
    jpeg_prob=0.0,
    jpeg_quality_range=(95, 100),

    resize_prob=0.0,
    resize_scale_range=(1.0, 1.0),

    histogram_match_strength_range=(0.99, 1.0),
    use_histogram_match=True,

    median_match_strength = 0.35
)

SIMULATED_RAW_STYLE = CameraStyleParams(
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

    histogram_match_strength_range=(0.0, 0.0),
    use_histogram_match=False,
)

STYLE_PARAMS_REGISTRY: Dict[str, CameraStyleParams] = {
    "iphone": IPHONE_STYLE,
    "googlepixel": GOOGLEPIXEL_STYLE,
    "microscope": MICROSCOPE_STYLE,
    "simulated_raw": SIMULATED_RAW_STYLE,
}


# -----------------------------
# style config helpers
# -----------------------------

def phone_mix_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("microscope", "iphone", "googlepixel"),
        probs=None,
    )

def googlepixel_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("googlepixel",),
        probs=None,
    )

def iphone_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("iphone",),
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


# -----------------------------
# quantile band cache
# -----------------------------

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


def _sample_pixels_from_image(
    img_hwc: np.ndarray,
    sample_pixels: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    pixels = img_hwc.reshape(-1, 3)
    n_total = pixels.shape[0]

    if sample_pixels is not None and n_total > sample_pixels:
        idx = rng.choice(n_total, size=sample_pixels, replace=False)
        pixels = pixels[idx]

    return pixels.astype(np.float32, copy=False)


def build_target_quantile_band_cache(
    folders: Optional[Sequence[str | Path]] = None,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    devices: Sequence[str] = ("iphone", "googlepixel", "microscope"),
    recursive: bool = True,
    sample_pixels_per_image: Optional[int] = 300_000,
    n_quantiles: int = 1024,
    q_band_lo: float = 0.025,
    q_band_hi: float = 0.975,
    rng_seed: int = 0,
    cache_path: Optional[str | Path] = STYLE_QUANTILE_CACHE_PATH,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    rng = np.random.default_rng(rng_seed)
    image_paths = _collect_image_paths(folders=folders, exts=exts, recursive=recursive)
    q_probs = np.linspace(0.0, 1.0, int(n_quantiles), dtype=np.float32)

    per_device_quantiles: Dict[str, list[np.ndarray]] = {d: [] for d in devices}
    counts = {d: 0 for d in devices}

    for path in image_paths:
        device = _find_device_label(path)
        if device not in devices:
            continue

        file_name = os.path.basename(path)
        folder = os.path.dirname(path)

        img, _ = load_image(
            file_name,
            base_dir=folder,
            as_chw=False,
            scale=True,
            fast_scale=True,
        )

        if img.ndim != 3 or img.shape[2] != 3:
            continue

        pixels = _sample_pixels_from_image(
            img_hwc=img,
            sample_pixels=sample_pixels_per_image,
            rng=rng,
        )

        q_img = []
        for c in range(3):
            qc = np.quantile(np.clip(pixels[:, c], 0.0, 1.0), q_probs).astype(np.float32)
            q_img.append(qc)
        q_img = np.stack(q_img, axis=0)

        per_device_quantiles[device].append(q_img)
        counts[device] += 1

    devices_payload: Dict[str, Any] = {}
    for device in devices:
        if len(per_device_quantiles[device]) == 0:
            continue

        Q = np.stack(per_device_quantiles[device], axis=0)

        q_center = np.quantile(Q, 0.50, axis=0).astype(np.float32)
        q_lo = np.quantile(Q, q_band_lo, axis=0).astype(np.float32)
        q_hi = np.quantile(Q, q_band_hi, axis=0).astype(np.float32)

        q_center = np.maximum.accumulate(q_center, axis=1)
        q_lo = np.maximum.accumulate(q_lo, axis=1)
        q_hi = np.maximum.accumulate(q_hi, axis=1)
        q_center = np.clip(q_center, q_lo, q_hi)

        devices_payload[device] = {
            "n_images": counts[device],
            "q_center": q_center,
            "q_lo": q_lo,
            "q_hi": q_hi,
        }

    cache = {
        "devices": devices_payload,
        "q_probs": q_probs,
        "devices_requested": tuple(devices),
        "folders": [str(f) for f in (folders if folders is not None else EXT_IMAGES_FOLDERS)],
        "sample_pixels_per_image": sample_pixels_per_image,
        "n_quantiles": int(n_quantiles),
        "q_band_lo": float(q_band_lo),
        "q_band_hi": float(q_band_hi),
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
    sample_pixels_per_image: Optional[int] = 300_000,
    n_quantiles: int = 1024,
    q_band_lo: float = 0.025,
    q_band_hi: float = 0.975,
) -> Dict[str, Any]:
    return build_target_quantile_band_cache(
        folders=folders,
        cache_path=cache_path,
        force_recompute=force_recompute,
        sample_pixels_per_image=sample_pixels_per_image,
        n_quantiles=n_quantiles,
        q_band_lo=q_band_lo,
        q_band_hi=q_band_hi,
    )


# -----------------------------
# quantile band helpers
# -----------------------------

def _compute_image_channel_quantiles(
    x: np.ndarray,
    q_probs: np.ndarray,
) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
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


def apply_device_quantile_band_match(
    img: np.ndarray,
    target_device: str,
    quantile_band_cache: Dict[str, Any],
    strength: float = 1.0,
    preserve_input_layout: bool = True,
) -> np.ndarray:
    """
    Move an RGB image toward the real-image intensity distribution of one device,
    using per-channel quantile-band matching.

    This function does not force the image to exactly match one fixed histogram.
    Instead, for each channel separately, it computes the image's quantile curve
    and projects it into the allowed quantile band learned from real images of
    the chosen device. The amount of movement is controlled by `strength`.

    In plain terms:
      - `strength=0.0` leaves the image unchanged
      - `strength=1.0` applies the full minimal correction needed to place the
        channel quantiles inside the real-device band
      - intermediate values apply only part of that correction

    The operation is monotone per channel:
      - darker pixels stay darker than brighter pixels within the same channel
      - it changes the global channel intensity distribution
      - it does not explicitly model local structures or semantic regions

    Parameters
    ----------
    img : np.ndarray
        RGB image to transform.
        Notes:
          - The function clips values to [0, 1] before processing.
          - The function assumes RGB order, not BGR.

    target_device : str
        Name of the target device/domain whose real-image quantile band should
        be used.

        Typical values:
          - "microscope"
          - "iphone"
          - "googlepixel"

    quantile_band_cache : Dict[str, Any]
        Cache dictionary produced by your quantile-band builder, for example
        `build_target_quantile_band_cache(...)`.

    strength : float, default=1.0
        How strongly the image should be moved toward the target device band.

        Interpretation:
          - 0.0:
                no change
          - 1.0:
                full minimal projection into the target quantile band
          - 0.0 < strength < 1.0:
                partial movement toward that projected curve

    preserve_input_layout : bool, default=True
        Whether to return the result in the same layout as the input.

        Behavior:
          - if input was HWC, output is HWC
          - if input was CHW and this is True, output is converted back to CHW
          - if input was CHW and this is False, output stays HWC internally

        In most cases you want this set to True.

    Returns
    -------
    np.ndarray
        Transformed RGB image as float32 in [0, 1].

    """
    if target_device not in quantile_band_cache["devices"]:
        raise KeyError(f"Target device '{target_device}' not in quantile_band_cache")

    arr = np.asarray(img, dtype=np.float32)
    input_chw = False

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")

    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
        input_chw = True
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected CHW or HWC RGB image, got shape {img.shape}")

    arr = np.clip(arr, 0.0, 1.0)
    out = arr.copy()

    q_probs = quantile_band_cache["q_probs"]
    device_ref = quantile_band_cache["devices"][target_device]
    q_lo = device_ref["q_lo"]
    q_hi = device_ref["q_hi"]

    for c in range(3):
        src_q = _compute_image_channel_quantiles(arr[..., c], q_probs)
        dst_q_full = _minimal_band_projection(src_q, q_lo[c], q_hi[c])
        dst_q = ((1.0 - strength) * src_q + strength * dst_q_full).astype(np.float32)
        dst_q = np.maximum.accumulate(dst_q)
        out[..., c] = _apply_quantile_map_1d(arr[..., c], src_q, dst_q)

    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    if preserve_input_layout and input_chw:
        out = np.moveaxis(out, -1, 0)

    return out
