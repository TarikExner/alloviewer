import cv2
import numpy as np
from typing import Any, Dict, Optional

from .types import RNG

from .camera_style_config import (
    CameraStyleConfig,
    CameraStyleParams,
)

from .histogram_capture import (
    apply_device_quantile_band_match,
)

from .utils import (
    apply_s_curve,
    lift_shadows,
    compress_highlights,
    sample_channel_values,
    apply_channel_median_match,
    apply_read_noise,
    apply_global_blur,
    apply_photon_noise,
)

def apply_camera_style(
    img: np.ndarray,
    rng: RNG,
    style_cfg: CameraStyleConfig,
    style_registry: Dict[str, CameraStyleParams],
    quantile_band_cache: Optional[Dict[str, Any]] = None,
    cell_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply a sampled camera style, then optionally move the result toward the
    corresponding real-device histogram distribution.

    Parameters
    ----------
    img:
        RGB HWC float image in [0, 1].

    rng:
        Random number generator.

    style_cfg:
        Camera style sampler.

    style_registry:
        Mapping from style name to CameraStyleParams.

    quantile_band_cache:
        Optional histogram cache.

    cell_mask:
        Optional simulated cell mask. If given, foreground/background histogram
        matching can use the true simulated cells instead of guessing.

    Returns
    -------
    np.ndarray:
        RGB float32 image in [0, 1].
    """
    assert img.ndim == 3 and img.shape[2] == 3

    img = np.clip(img.astype(np.float32).copy(), 0.0, 1.0)

    style_name = style_cfg.sample_style(rng)
    if style_name not in style_registry:
        raise KeyError(f"Style '{style_name}' not found in style_registry")

    params = style_registry[style_name]

    if style_name in {"simulated_raw", "raw_simulated"}:
        return img

    H, W, _ = img.shape

    # 1) exposure
    exposure = rng.uniform(*params.exposure_range)
    img = np.clip(img * exposure, 0.0, 1.0)

    # 1b) channel-specific exposure / underillumination
    channel_gains = sample_channel_values(rng, params.channel_gain_range)
    channel_shifts = sample_channel_values(rng, params.channel_shift_range)
    img = np.clip(
        img * channel_gains[None, None, :] + channel_shifts[None, None, :],
        0.0,
        1.0,
    )

    # 2) global contrast / brightness
    c = rng.uniform(*params.c_range)
    b = rng.uniform(*params.b_range)
    img = np.clip(img * c + b, 0.0, 1.0)

    # 3) white balance
    wb = rng.uniform(
        params.wb_range[0],
        params.wb_range[1],
        size=3,
    ).astype(np.float32)
    wb = wb / (wb.mean() + 1e-8)
    img = np.clip(img * wb[None, None, :], 0.0, 1.0)

    # 4) explicit color-axis shifts
    gm = rng.uniform(*params.green_magenta_shift_range)
    by = rng.uniform(*params.blue_yellow_shift_range)

    color_shift = np.array(
        [
            1.0 - 0.35 * gm - 0.50 * by,
            1.0 + 1.00 * gm,
            1.0 - 0.35 * gm + 0.50 * by,
        ],
        dtype=np.float32,
    )
    color_shift = color_shift / (color_shift.mean() + 1e-8)
    img = np.clip(img * color_shift[None, None, :], 0.0, 1.0)

    # 5) saturation
    sat = rng.uniform(*params.saturation_range)
    gray = img.mean(axis=2, keepdims=True)
    img = np.clip(gray + sat * (img - gray), 0.0, 1.0)

    # 6) R/G mixing
    a = rng.uniform(*params.mix_range)
    M = np.array(
        [
            [1.0 - a, a,       0.0],
            [a,       1.0 - a, 0.0],
            [0.0,     0.0,     1.0],
        ],
        dtype=np.float32,
    )
    img = np.clip(img @ M.T, 0.0, 1.0)

    # 7) uneven illumination
    illum_amp = rng.uniform(*params.illum_amp_range)
    if illum_amp > 0:
        scale = 64
        h_small = max(1, H // scale)
        w_small = max(1, W // scale)

        field_small = rng.normal(
            0.0,
            1.0,
            size=(h_small, w_small),
        ).astype(np.float32)

        field = cv2.resize(
            field_small,
            (W, H),
            interpolation=cv2.INTER_CUBIC,
        )
        field = field - field.mean()
        field = field / (field.std() + 1e-6)
        field = 1.0 + illum_amp * field
        field = np.clip(
            field,
            1.0 - 2.0 * illum_amp,
            1.0 + 2.0 * illum_amp,
        )

        img = np.clip(img * field[..., None], 0.0, 1.0)

    # 8) vignette
    vignette_amp = rng.uniform(*params.vignette_amp_range)
    if vignette_amp > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H / 2.0, W / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = rr / (0.72 * max(H, W))

        vignette = 1.0 - vignette_amp * (r_norm ** 2)
        vignette = np.clip(vignette, 1.0 - vignette_amp, 1.0)

        img = np.clip(img * vignette[..., None], 0.0, 1.0)

    # 9) S-curve / midtone contrast
    s = rng.uniform(*params.midtone_contrast_range)
    img = apply_s_curve(img, s)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 10) shadow lift
    shadow_lift = rng.uniform(*params.shadow_lift_range)
    img = lift_shadows(img, shadow_lift)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 11) highlight compression
    highlight_rolloff = rng.uniform(*params.highlight_rolloff_range)
    img = compress_highlights(img, highlight_rolloff)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 12) gamma
    gamma = rng.uniform(*params.gamma_range)
    img = np.clip(img, 1e-6, 1.0) ** gamma
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 13) clean global blur
    global_blur_sigma = rng.uniform(*params.global_blur_sigma_range)
    img = apply_global_blur(
        img=img,
        sigma=float(global_blur_sigma),
    )

    # 14) photon shot noise
    photon_level = rng.uniform(*params.photon_level_range)
    img = apply_photon_noise(
        img=img,
        rng=rng,
        photon_level=float(photon_level),
    )

    # 15) read noise
    read_noise = rng.uniform(*params.read_noise_range)
    img = apply_read_noise(
        img=img,
        rng=rng,
        read_noise=float(read_noise),
    )

    # 16) clipping event
    if rng.random() < params.clip_prob:
        gain = rng.uniform(1.03, 1.25)
        img = np.clip(img * gain, 0.0, 1.0)

    # 17) resampling artifacts
    if rng.random() < params.resize_prob:
        resize_scale = rng.uniform(*params.resize_scale_range)
        h2 = max(8, int(round(H * resize_scale)))
        w2 = max(8, int(round(W * resize_scale)))

        tmp = cv2.resize(
            img,
            (w2, h2),
            interpolation=cv2.INTER_AREA,
        )
        img = cv2.resize(
            tmp,
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )
        img = np.clip(img, 0.0, 1.0)

    # 18) blur + sharpen
    sigma = rng.uniform(*params.blur_sigma_range)
    sharpen_strength = rng.uniform(*params.sharpen_strength_range)

    if sigma > 0.0 or sharpen_strength > 0.0:
        img_float = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)

        if sigma > 0.0:
            ksize = max(3, int(2 * round(sigma) + 1))
            if ksize % 2 == 0:
                ksize += 1

            blurred = cv2.GaussianBlur(
                img_float,
                (ksize, ksize),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
            )
        else:
            blurred = img_float

        if sharpen_strength > 0.0:
            img = cv2.addWeighted(
                img_float,
                1.0 + float(sharpen_strength),
                blurred,
                -float(sharpen_strength),
                0.0,
            )
        else:
            img = blurred

        img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 19) soft histogram band match
    if (
        params.use_histogram_match
        and quantile_band_cache is not None
        and style_name in quantile_band_cache.get("devices", {})
    ):
        hist_strength = rng.uniform(*params.histogram_match_strength_range)

        if hist_strength > 0:
            img = apply_device_quantile_band_match(
                img=img,
                target_device=style_name,
                quantile_band_cache=quantile_band_cache,
                strength=float(hist_strength),
                preserve_input_layout=True,
                rng=rng,
                cell_mask=cell_mask,
                region_mode=params.histogram_region_mode,
                match_mode=params.histogram_match_mode,
                mask_blur_sigma=float(params.histogram_mask_blur_sigma),
                fallback_to_all=True,
            )
            img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 20) optional per-channel median correction
    if (
        params.use_histogram_match
        and params.use_median_match
        and quantile_band_cache is not None
        and style_name in quantile_band_cache.get("devices", {})
    ):
        median_strength = rng.uniform(*params.median_match_strength)

        if median_strength > 0:
            if style_name == "microscope":
                channel_strength = np.array([0.35, 0.35, 1.00], dtype=np.float32)
            elif style_name == "monochrome_real":
                channel_strength = np.array([1.00, 1.00, 0.15], dtype=np.float32)
            else:
                channel_strength = np.array([1.0, 1.0, 1.0], dtype=np.float32)

            img = apply_channel_median_match(
                img=img,
                target_device=style_name,
                quantile_band_cache=quantile_band_cache,
                strength=float(median_strength),
                per_channel_strength=channel_strength,
            )
            img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 21) JPEG as final phone/output artifact
    if rng.random() < params.jpeg_prob:
        tmp = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        tmp_bgr = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)

        quality = int(
            rng.integers(
                params.jpeg_quality_range[0],
                params.jpeg_quality_range[1] + 1,
            )
        )

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, enc = cv2.imencode(".jpg", tmp_bgr, encode_param)

        if ok:
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.clip(img, 0.0, 1.0).astype(np.float32)

    return img.astype(np.float32)
