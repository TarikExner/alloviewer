from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Dict, Tuple

import numpy as np

from .types import RNG, ChannelRange

@dataclass
class CameraStyleParams:
    name: str

    # global tone
    exposure_range: Tuple[float, float] = (1.0, 1.0)
    c_range: Tuple[float, float] = (1.0, 1.0)
    b_range: Tuple[float, float] = (0.0, 0.0)
    gamma_range: Tuple[float, float] = (1.0, 1.0)

    # per-channel acquisition exposure
    channel_gain_range: ChannelRange = (
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
    )
    channel_shift_range: ChannelRange = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )

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

    # blur / sharpen
    blur_sigma_range: Tuple[float, float] = (0.0, 0.0)
    sharpen_strength_range: Tuple[float, float] = (0.0, 0.0)

    # acquisition-style effects migrated from simulate_image / CameraDimension
    global_blur_sigma_range: Tuple[float, float] = (0.0, 0.0)
    photon_level_range: Tuple[float, float] = (0.0, 0.0)
    read_noise_range: Tuple[float, float] = (0.0, 0.0)

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
    histogram_region_mode: str = "all"
    histogram_match_mode: str = "project_to_band"
    histogram_mask_blur_sigma: float = 1.5

    # median matching
    median_match_strength: Tuple[float, float] = (0.0, 1.0)
    use_median_match: bool = True


@dataclass
class CameraStyleConfig:
    styles: Sequence[str] = (
        "microscope",
        "iphone",
        "googlepixel",
        "simulated_raw",
    )
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

def with_histogram_adherence(
    registry: Dict[str, CameraStyleParams],
    mode: str = "default",
) -> Dict[str, CameraStyleParams]:
    """
    Return a modified style registry with stronger or weaker histogram adherence.

    mode:
      - "strict": for PCA/showoff datasets; strongly follows real histogram cache
      - "default": unchanged
      - "lenient": for training; keeps more synthetic variation
      - "off": disables histogram and median matching
    """
    mode = str(mode).lower()

    if mode == "default":
        return dict(registry)

    if mode == "strict":
        hist_range = (0.99, 1.00)
        median_range = (0.70, 0.70)
        use_hist = True
        use_median = True

    elif mode == "lenient":
        hist_range = (0.5, 0.55)
        median_range = (0.7, 0.7)
        use_hist = True
        use_median = True

    elif mode == "off":
        hist_range = (0.0, 0.0)
        median_range = (0.0, 0.0)
        use_hist = False
        use_median = False

    else:
        raise ValueError("mode must be one of: 'strict', 'default', 'lenient', 'off'")

    out: Dict[str, CameraStyleParams] = {}

    for name, params in registry.items():
        if name == "simulated_raw":
            out[name] = params
            continue

        out[name] = replace(
            params,
            histogram_match_strength_range=hist_range,
            median_match_strength=median_range,
            use_histogram_match=use_hist,
            use_median_match=use_median,
        )

    return out

IPHONE_STYLE = CameraStyleParams(
    name="iphone",
    exposure_range=(0.98, 1.04),
    c_range=(0.90, 0.99),
    b_range=(0.01, 0.035),
    gamma_range=(1.00, 1.03),

    channel_gain_range=(
        (0.95, 1.05),
        (0.95, 1.05),
        (0.95, 1.05),
    ),
    channel_shift_range=(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ),

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

    global_blur_sigma_range=(0.0, 0.25),
    photon_level_range=(2000.0, 12000.0),
    read_noise_range=(0.0002, 0.005),

    vignette_amp_range=(0.00, 0.02),
    illum_amp_range=(0.00, 0.015),

    clip_prob=0.00,
    jpeg_prob=0.3,
    jpeg_quality_range=(80, 100),

    resize_prob=0.00,
    resize_scale_range=(0.95, 1.00),

    histogram_match_strength_range=(0.35, 1.00),
    use_histogram_match=True,
    histogram_region_mode="foreground_background",
    histogram_match_mode="sample_real_curve",
    histogram_mask_blur_sigma=1.5,

    median_match_strength=(0.0, 0.7),
    use_median_match=True,
)


GOOGLEPIXEL_STYLE = CameraStyleParams(
    name="googlepixel",
    exposure_range=(0.90, 1.08),
    c_range=(0.90, 1.10),
    b_range=(-0.01, 0.03),
    gamma_range=(0.90, 1.08),

    channel_gain_range=(
        (0.92, 1.08),
        (0.92, 1.08),
        (0.92, 1.08),
    ),
    channel_shift_range=(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ),

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

    global_blur_sigma_range=(0.0, 0.25),
    photon_level_range=(2000.0, 12000.0),
    read_noise_range=(0.0002, 0.005),

    vignette_amp_range=(0.02, 0.08),
    illum_amp_range=(0.02, 0.06),

    clip_prob=0.08,
    jpeg_prob=0.3,
    jpeg_quality_range=(80, 100),

    resize_prob=0.10,
    resize_scale_range=(0.80, 0.96),

    histogram_match_strength_range=(0.35, 1.00),
    use_histogram_match=True,
    histogram_region_mode="foreground_background",
    histogram_match_mode="sample_real_curve",
    histogram_mask_blur_sigma=1.5,

    median_match_strength=(0.0, 0.7),
    use_median_match=True,
)


MICROSCOPE_STYLE = CameraStyleParams(
    name="microscope",
    exposure_range=(0.76, 0.98),
    c_range=(0.90, 1.02),
    b_range=(-0.03, 0.005),
    gamma_range=(0.98, 1.12),

    channel_gain_range=(
        (0.90, 1.05),
        (0.90, 1.05),
        (0.90, 1.05),
    ),
    channel_shift_range=(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ),

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

    global_blur_sigma_range=(0.0, 0.25),
    photon_level_range=(2000.0, 12000.0),
    read_noise_range=(0.0002, 0.005),

    vignette_amp_range=(0.00, 0.008),
    illum_amp_range=(0.00, 0.01),

    clip_prob=0.00,
    jpeg_prob=0.3,
    jpeg_quality_range=(80, 100),

    resize_prob=0.0,
    resize_scale_range=(1.0, 1.0),

    histogram_match_strength_range=(0.95, 1.00),
    use_histogram_match=True,
    histogram_region_mode="foreground_background",
    histogram_match_mode="sample_real_curve",
    histogram_mask_blur_sigma=1.0,

    median_match_strength=(0.0, 0.7),
    use_median_match=True,
)


MONOCHROME_GENERIC_STYLE = CameraStyleParams(
    name="monochrome_generic",

    # Generic separate-channel microscope look.
    # Each channel can be dimmed independently.
    exposure_range=(0.80, 1.05),
    c_range=(0.90, 1.05),
    b_range=(-0.025, 0.005),
    gamma_range=(0.95, 1.12),

    channel_gain_range=(
        (0.0, 1.10),
        (0.0, 1.10),
        (0.0, 1.10),
    ),
    channel_shift_range=(
        (-0.015, 0.005),
        (-0.015, 0.005),
        (-0.015, 0.005),
    ),

    shadow_lift_range=(0.00, 0.015),
    highlight_rolloff_range=(0.00, 0.025),
    midtone_contrast_range=(-0.04, 0.025),

    mix_range=(0.00, 0.005),
    wb_range=(0.995, 1.005),
    saturation_range=(0.85, 1.00),
    green_magenta_shift_range=(0.0, 0.0),
    blue_yellow_shift_range=(0.0, 0.0),

    blur_sigma_range=(0.08, 0.30),
    sharpen_strength_range=(0.00, 0.025),

    global_blur_sigma_range=(0.0, 0.25),
    photon_level_range=(2000.0, 12000.0),
    read_noise_range=(0.0002, 0.006),

    vignette_amp_range=(0.00, 0.01),
    illum_amp_range=(0.00, 0.012),

    clip_prob=0.00,
    jpeg_prob=0.00,
    jpeg_quality_range=(90, 100),

    resize_prob=0.0,
    resize_scale_range=(1.0, 1.0),

    histogram_match_strength_range=(0.70, 1.00),
    use_histogram_match=True,
    histogram_region_mode="foreground_background",
    histogram_match_mode="project_to_band",
    histogram_mask_blur_sigma=1.0,

    median_match_strength=(0.0, 0.15),
    use_median_match=True,
)


MONOCHROME_REAL_STYLE = CameraStyleParams(
    name="monochrome_real",

    # Real same-exposure monochrome case:
    # blue is near absent, red is underilluminated, green is main signal.
    exposure_range=(0.80, 1.05),
    c_range=(0.90, 1.04),
    b_range=(-0.025, 0.005),
    gamma_range=(0.95, 1.12),

    channel_gain_range=(
        (0.25, 0.65),   # red underilluminated
        (0.80, 1.10),   # green retained
        (0.00, 0.02),   # blue practically disabled
    ),
    channel_shift_range=(
        (-0.010, 0.000),
        (-0.005, 0.005),
        (0.000, 0.000),
    ),

    shadow_lift_range=(0.00, 0.015),
    highlight_rolloff_range=(0.00, 0.025),
    midtone_contrast_range=(-0.04, 0.025),

    # Keep color operations almost disabled here. This is not phone RGB color.
    mix_range=(0.00, 0.003),
    wb_range=(0.995, 1.005),
    saturation_range=(0.95, 1.00),
    green_magenta_shift_range=(0.0, 0.0),
    blue_yellow_shift_range=(0.0, 0.0),

    blur_sigma_range=(0.08, 0.30),
    sharpen_strength_range=(0.00, 0.025),

    global_blur_sigma_range=(0.0, 0.25),
    photon_level_range=(2000.0, 12000.0),
    read_noise_range=(0.0002, 0.006),

    vignette_amp_range=(0.00, 0.01),
    illum_amp_range=(0.00, 0.012),

    clip_prob=0.00,
    jpeg_prob=0.00,
    jpeg_quality_range=(90, 100),

    resize_prob=0.0,
    resize_scale_range=(1.0, 1.0),

    histogram_match_strength_range=(0.80, 1.00),
    use_histogram_match=True,
    histogram_region_mode="foreground_background",
    histogram_match_mode="sample_real_curve",
    histogram_mask_blur_sigma=1.0,

    median_match_strength=(0.0, 0.25),
    use_median_match=True,
)


SIMULATED_RAW_STYLE = CameraStyleParams(
    name="simulated_raw",
    exposure_range=(1.0, 1.0),
    c_range=(1.0, 1.0),
    b_range=(0.0, 0.0),
    gamma_range=(1.0, 1.0),

    channel_gain_range=(
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
    ),
    channel_shift_range=(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ),

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

    global_blur_sigma_range=(0.0, 0.0),
    photon_level_range=(0.0, 0.0),
    read_noise_range=(0.0, 0.0),

    vignette_amp_range=(0.0, 0.0),
    illum_amp_range=(0.0, 0.0),

    clip_prob=0.0,
    jpeg_prob=0.0,
    jpeg_quality_range=(80, 100),

    resize_prob=0.0,
    resize_scale_range=(1.0, 1.0),

    histogram_match_strength_range=(0.0, 0.0),
    use_histogram_match=False,
    histogram_region_mode="all",
    histogram_match_mode="project_to_band",
    histogram_mask_blur_sigma=0.0,

    median_match_strength=(0.0, 0.0),
    use_median_match=False,
)


STYLE_PARAMS_REGISTRY: Dict[str, CameraStyleParams] = {
    "iphone": IPHONE_STYLE,
    "googlepixel": GOOGLEPIXEL_STYLE,
    "microscope": MICROSCOPE_STYLE,
    "monochrome_generic": MONOCHROME_GENERIC_STYLE,
    "monochrome_real": MONOCHROME_REAL_STYLE,
    "simulated_raw": SIMULATED_RAW_STYLE,
}


###############
### PRESETS ###
###############

def diverse_cameras() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=(
            "microscope",
            "iphone",
            "googlepixel",
            "monochrome_generic",
            "simulated_raw",
        ),
        probs=None,
    )

def diverse_cameras_showoff() -> CameraStyleConfig:
    """without generic monochrome, just for fig S2"""
    return CameraStyleConfig(
        styles=(
            "microscope",
            "iphone",
            "googlepixel",
            "monochrome_real",
            "simulated_raw",
        ),
        probs=None,
    )


def diverse_cameras_with_monochrome() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=(
            "microscope",
            "iphone",
            "googlepixel",
            "monochrome_generic",
            "monochrome_real",
            "simulated_raw",
        ),
        probs=None,
    )


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


def monochrome_generic_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("monochrome_generic",),
        probs=None,
    )


def monochrome_real_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("monochrome_real",),
        probs=None,
    )


def simulated_raw_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("simulated_raw",),
        probs=None,
    )
