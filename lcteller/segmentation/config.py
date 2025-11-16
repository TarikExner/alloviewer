# config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Sequence
import numbers
import numpy as np

Param = Any
RNG = np.random.Generator
NumOrRange = Union[int, float, Tuple[float, float], Tuple[int, int]]

PASS_THROUGH_RANGES: Tuple[str, ...] = (
    # in-focus/out-of-focus blur ranges
    "sigma_in", "sigma_out",
    # debris ranges
    "dirt_size", "dirt_sigma", "dirt_alpha",
    # ring / reflect ranges
    "ring_sigma_range", "ring_alpha_range",
    "reflect_offset_range", "reflect_alpha_range",
    # ghosts sampled inside the simulator
    "ghost_sigma", "ghost_intensity",
)

INT_KEYS: Tuple[str, ...] = (
    "H", "W",
    "n_cells",
    "boundary_width",
    "rim_min_sep_px",
    "pack_iters",
    "ghost_trail",
    "reflect_n",
    "ring_artifacts",
    "reflect_harmonics",
)

# -----------------------------
# Sampling helpers
# -----------------------------

def _is_pair(x) -> bool:
    return isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(v, numbers.Number) for v in x)

def _sample_number(rng: RNG, spec: Union[float, int, Tuple[float, float], Tuple[int, int]], integer: bool = False) -> Union[float, int]:
    if _is_pair(spec):
        lo, hi = spec  # inclusive-exclusive for floats; inclusive-inclusive for ints below
        if integer:
            return int(rng.integers(int(np.floor(lo)), int(np.ceil(hi)) + 1))
        else:
            return float(rng.uniform(float(lo), float(hi)))
    # scalar
    return int(spec) if integer else float(spec)

def _sample_bool(rng: RNG, spec: Union[bool, float, Tuple[float, float]]) -> bool:
    """
    spec can be:
      - bool: returned directly
      - float p in [0,1]: Bernoulli(p)
      - (p_lo, p_hi): sample p ~ U[p_lo, p_hi], then Bernoulli(p)
    """
    if isinstance(spec, bool):
        return spec
    if _is_pair(spec):
        p = float(rng.uniform(float(spec[0]), float(spec[1])))
        p = np.clip(p, 0.0, 1.0)
        return bool(rng.random() < p)
    # scalar prob
    p = float(spec)
    p = np.clip(p, 0.0, 1.0)
    return bool(rng.random() < p)

def _round_to_multiple(x: float, m: int) -> int:
    return int(max(m, round(x / m) * m))

def _choose_ratio(rng: RNG, ratios: Sequence[Tuple[int, int]], portrait_prob: float) -> float:
    num, den = ratios[int(rng.integers(0, len(ratios)))]
    r = num / den  # width / height
    if portrait_prob > 0 and rng.random() < portrait_prob:
        r = 1.0 / r  # swap to height / width
    return r

@dataclass
class CameraStyleParams:
    name: str

    # global contrast / brightness
    c_range: Tuple[float, float]
    b_range: Tuple[float, float]

    # gamma
    gamma_range: Tuple[float, float]

    # R/G mixing
    mix_range: Tuple[float, float]

    # blur + sharpen
    blur_sigma_range: Tuple[float, float]
    sharpen_strength: float

    # noise
    noise_std_base: float

    # white balance / color cast
    wb_range: Tuple[float, float]

    # uneven illumination + vignette
    vignette_amp: float
    illum_amp: float

    # clipping
    clip_prob: float


# concrete style presets (all numbers live here)

MICROSCOPE_STYLE = CameraStyleParams(
    name="microscope",
    c_range=(0.9, 1.1),
    b_range=(-0.03, 0.03),
    gamma_range=(0.9, 1.1),
    mix_range=(0.02, 0.08),
    blur_sigma_range=(0.4, 1.0),
    sharpen_strength=0.3,
    noise_std_base=0.010,
    wb_range=(0.95, 1.05),
    vignette_amp=0.05,
    illum_amp=0.05,
    clip_prob=0.2,
)

IPHONE_STYLE = CameraStyleParams(
    name="iphone",
    c_range=(0.9, 1.2),
    b_range=(-0.05, 0.05),
    gamma_range=(0.8, 1.3),
    mix_range=(0.08, 0.25),
    blur_sigma_range=(0.8, 1.8),
    sharpen_strength=0.7,
    noise_std_base=0.015,
    wb_range=(0.85, 1.20),
    vignette_amp=0.12,
    illum_amp=0.12,
    clip_prob=0.4,
)

PIXEL_STYLE = CameraStyleParams(
    name="pixel",
    c_range=(0.9, 1.2),
    b_range=(-0.04, 0.04),
    gamma_range=(0.85, 1.2),
    mix_range=(0.06, 0.20),
    blur_sigma_range=(0.6, 1.5),
    sharpen_strength=0.5,
    noise_std_base=0.012,
    wb_range=(0.9, 1.15),
    vignette_amp=0.10,
    illum_amp=0.10,
    clip_prob=0.3,
)

SIMULATED_RAW_STYLE = CameraStyleParams(
    name="simulated_raw",
    c_range=(1.0, 1.0),
    b_range=(0.0, 0.0),
    gamma_range=(1.0, 1.0),
    mix_range=(0.0, 0.0),
    blur_sigma_range=(0.0, 0.0),
    sharpen_strength=0.0,
    noise_std_base=0.0,
    wb_range=(1.0, 1.0),
    vignette_amp=0.0,
    illum_amp=0.0,
    clip_prob=0.0,
)

STYLE_PARAMS_REGISTRY: Dict[str, CameraStyleParams] = {
    "microscope": MICROSCOPE_STYLE,
    "iphone": IPHONE_STYLE,
    "pixel": PIXEL_STYLE,
    "raw": SIMULATED_RAW_STYLE,
}

@dataclass
class CameraStyleConfig:
    """
    Controls which camera look(s) to sample in apply_camera_style.
    """
    styles: Sequence[str] = ("microscope", "iphone", "pixel")
    probs: Optional[Sequence[float]] = None
    jpeg_prob: float = 0.3

    def sample_style(self, rng: RNG) -> str:
        if len(self.styles) == 1:
            return self.styles[0]

        if self.probs is None:
            idx = int(rng.integers(0, len(self.styles)))
            return self.styles[idx]

        p = np.asarray(self.probs, dtype=float)
        p = p / p.sum()
        idx = int(rng.choice(len(self.styles), p=p))
        return self.styles[idx]


@dataclass
class CameraSetup:
    """
    Sample width first, then compute height from an aspect ratio.
    """
    name: str = "Camera"

    # width range (int)
    W: NumOrRange = (512, 2500)
    H: Optional[int] = None

    # aspect ratios as (width, height)
    aspect_ratios: Sequence[Tuple[int, int]] = ((16, 9), (16, 10), (3, 2), (4, 3))
    portrait_prob: float = 0.5   # 0..1, chance to flip to portrait
    size_multiple: int = 32      # round H to this multiple

    # --- noise / camera ---
    blur_sigma_global: Union[float, Tuple[float, float]] = (0.0, 1.0)
    photon_level: Union[float, Tuple[float, float]] = (100, 4000)
    read_noise: Union[float, Tuple[float, float]] = (0.0, 0.1)

    # background hue (0=orange, 1=green)
    bg_hue: NumOrRange = (0.0, 1.0)

    def sample(self, rng: RNG) -> Dict[str, Any]:
        W = _sample_number(rng, self.W, integer=True)
        if not self.H:
            ratio = _choose_ratio(rng, self.aspect_ratios, self.portrait_prob)  # width/height
            H = _round_to_multiple(W / ratio, self.size_multiple)
        else:
            H = self.H

        return {
            "H": int(H),
            "W": int(W),
            "photon_level": _sample_number(rng, self.photon_level, integer=False),
            "read_noise": _sample_number(rng, self.read_noise, integer=False),
            "blur_sigma_global": _sample_number(rng, self.blur_sigma_global, integer=False),
            "bg_hue": _sample_number(rng, self.bg_hue, integer=False),
        }


@dataclass
class SimulatorConfig:
    # --- size / geometry ---
    well_radius_frac: Union[float, Tuple[float, float]] = (0.3, 0.5)
    well_center_jitter: Union[float, Tuple[float, float]] = (0.0, 0.04)

    # --- radial look of the well ---
    background_level: Union[float, Tuple[float, float]] = (0, 0.3)
    edge_boost: Union[float, Tuple[float, float]] = 0.25
    radial_gamma: Union[float, Tuple[float, float]] = 1.2
    vignette_strength: Union[float, Tuple[float, float]] = 0.20

    # --- cells ---
    n_cells: Union[int, Tuple[int, int]] = (10, 2000)
    cell_diameter: Union[float, Tuple[float, float]] = (2.0, 12.0)
    frac_positive: Union[float, Tuple[float, float]] = (0.0, 1.0)
    color_jitter: Union[float, Tuple[float, float]] = (0.0, 0.2)
    sigma_in: Union[Tuple[float, float], Tuple[float, float]] = (0.5, 1.5)   # pass-through
    sigma_out: Union[Tuple[float, float], Tuple[float, float]] = (0.5, 1.5) # pass-through
    focus_frac_in: Union[float, Tuple[float, float]] = (0.0, 1.0)
    in_focus_sigma_thresh: Optional[Union[float, Tuple[float, float]]] = None

    boundary_width: Union[int, Tuple[int, int]] = 2

    # crowd near outer wall
    rim_bias: Union[float, Tuple[float, float]] = (0.5, 0.95)
    rim_band: Union[float, Tuple[float, float]] = (0.1, 0.5)
    edge_clamp: Union[float, Tuple[float, float]] = (0.1, 0.65)

    # --- collision / packing ---
    min_cell_sep_px: Optional[Union[float, Tuple[float, float]]] = None    # if None -> 0.9 * cell_diameter
    rim_min_sep_px: Union[int, Tuple[int, int]] = (4, 20)
    pack_iters: Union[int, Tuple[int, int]] = (10, 20)
    pack_strength: Union[float, Tuple[float, float]] = (0.0, 1.0)
    wall_margin_px: Union[float, Tuple[float, float]] = (2.0, 20.0)

    # --- sidedness ---
    side_bias_enable: Union[bool, float, Tuple[float, float]] = True
    side_bias_theta: Union[float, Tuple[float, float]] = (0.0, 3.141)
    side_bias_strength: Union[float, Tuple[float, float]] = (0.5, 0.9)
    side_bias_kappa: Union[float, Tuple[float, float]] = (0.0, 10.0)
    side_bias_inner_frac: Union[float, Tuple[float, float]] = (0.0, 1.0)

    # --- visual wall (soft rim) ---
    wall_blur_sigma: Union[float, Tuple[float, float]] = (6.0, 18.0)
    ring_artifacts: Union[int, Tuple[int, int]] = (0, 2)
    ring_sigma_range: Union[Tuple[float, float], Tuple[float, float]] = (6.0, 18.0)  # pass-through
    ring_alpha_range: Union[Tuple[float, float], Tuple[float, float]] = (0.01, 0.10) # pass-through

    # --- ghosts (outside) ---
    ghost_enable: Union[bool, float, Tuple[float, float]] = True
    ghost_density: Union[float, Tuple[float, float]] = (0.0, 1.0)
    ghost_offset_px: Union[float, Tuple[float, float]] = (10.0, 50.0)
    ghost_offset_jitter: Union[float, Tuple[float, float]] = (1.0, 10.0)
    ghost_sigma: Union[Tuple[float, float], Tuple[float, float]] = (2.5, 6.0) # pass-through (minor axis)
    ghost_dilate: Union[float, Tuple[float, float]] = 1.0
    ghost_intensity: Union[float, Tuple[float, float]] = (0.02, 0.3)            # pass-through OR scalar

    # outward elongation / trail
    ghost_stretch: Union[float, Tuple[float, float]] = (1.0, 3.0)
    ghost_trail: Union[int, Tuple[int, int]] = (1,3)
    ghost_trail_decay: Union[float, Tuple[float, float]] = 0.6

    # --- debris (inside well) ---
    dirt_density: Union[float, Tuple[float, float]] = (0.0001, 0.001)
    dirt_size: Union[Tuple[int, int], Tuple[int, int]] = (4, 12)  # pass-through (discrete range)
    dirt_sigma: Union[Tuple[float, float], Tuple[float, float]] = (0.0, 2.0)  # pass-through
    dirt_alpha: Union[Tuple[float, float], Tuple[float, float]] = (0.1, 1.0) # pass-through

    # --- radial reflections (outside) ---
    reflect_enable: Union[bool, float, Tuple[float, float]] = True
    reflect_n: Union[int, Tuple[int, int]] = (1, 10)
    reflect_theta_sigma: Union[float, Tuple[float, float]] = (0.05, 0.2)
    reflect_radial_sigma: Union[float, Tuple[float, float]] = (6.0, 18.0)
    reflect_offset_range: Union[Tuple[float, float], Tuple[float, float]] = (10, 100.0)  # pass-through
    reflect_alpha_range: Union[Tuple[float, float], Tuple[float, float]] = (0.05, 0.20)  # pass-through
    reflect_wobble: Union[float, Tuple[float, float]] = (0.0, 1.0)
    reflect_harmonics: Union[int, Tuple[int, int]] = 2
    reflect_harmonic_decay: Union[float, Tuple[float, float]] = 0.55

    def sample_kwargs(self, rng: RNG, camera: Optional[CameraSetup] = None) -> Dict[str, Any]:
        """
        Sample a full kwargs dict for simulate_image, optionally merging camera overrides.
        - Scalars/ranges are sampled to scalars,
        - PASS_THROUGH_RANGES are left as ranges (tuples),
        - Booleans support prob specs.
        """
        out: Dict[str, Any] = {}

        # helper to set any key with type awareness
        def set_num(key: str, integer: bool = False):
            val = getattr(self, key)
            out[key] = _sample_number(rng, val, integer=integer)

        def set_bool(key: str):
            val = getattr(self, key)
            out[key] = _sample_bool(rng, val)

        def set_passthrough(key: str):
            out[key] = getattr(self, key)

        # camera overrides (if provided)
        if camera is not None:
            cam = camera.sample(rng)
            out.update(cam)

        # size / geometry
        set_num("well_radius_frac")
        set_num("well_center_jitter")

        # radial background look
        set_num("background_level")
        set_num("edge_boost")
        set_num("radial_gamma")
        set_num("vignette_strength")

        # cells
        set_num("n_cells", integer=True)
        set_num("cell_diameter")
        set_num("frac_positive")
        set_num("color_jitter")
        set_passthrough("sigma_in")
        set_passthrough("sigma_out")
        set_num("focus_frac_in")
        if self.in_focus_sigma_thresh is None:
            out["in_focus_sigma_thresh"] = None
        else:
            set_num("in_focus_sigma_thresh")
        set_num("boundary_width", integer=True)

        # crowd near wall
        set_num("rim_bias")
        set_num("rim_band")
        set_num("edge_clamp")

        # collision / packing
        if self.min_cell_sep_px is None:
            out["min_cell_sep_px"] = None
        else:
            set_num("min_cell_sep_px")
        set_num("rim_min_sep_px", integer=True)
        set_num("pack_iters", integer=True)
        set_num("pack_strength")
        set_num("wall_margin_px")

        # sidedness
        set_bool("side_bias_enable")
        set_num("side_bias_theta")
        set_num("side_bias_strength")
        set_num("side_bias_kappa")
        set_num("side_bias_inner_frac")

        # wall
        set_num("wall_blur_sigma")
        set_num("ring_artifacts", integer=True)
        set_passthrough("ring_sigma_range")
        set_passthrough("ring_alpha_range")

        # ghosts
        set_bool("ghost_enable")
        set_num("ghost_density")
        set_num("ghost_offset_px")
        set_num("ghost_offset_jitter")
        set_passthrough("ghost_sigma")
        set_num("ghost_dilate")
        # ghost_intensity: allow scalar OR range; pass-through keeps range behavior
        if _is_pair(self.ghost_intensity):
            set_passthrough("ghost_intensity")
        else:
            set_num("ghost_intensity")

        set_num("ghost_stretch")
        set_num("ghost_trail", integer=True)
        set_num("ghost_trail_decay")

        # debris
        set_num("dirt_density")
        set_passthrough("dirt_size")
        set_passthrough("dirt_sigma")
        set_passthrough("dirt_alpha")

        # reflections
        set_bool("reflect_enable")
        set_num("reflect_n", integer=True)
        set_num("reflect_theta_sigma")
        set_num("reflect_radial_sigma")
        set_passthrough("reflect_offset_range")
        set_passthrough("reflect_alpha_range")
        set_num("reflect_wobble")
        set_num("reflect_harmonics", integer=True)
        set_num("reflect_harmonic_decay")


        # enforce int casting for int keys (in case of overrides)
        for k in INT_KEYS:
            if k in out:
                out[k] = int(round(out[k]))

        # simulator expects return_targets=True for training
        out.setdefault("return_targets", True)
        return out


# -----------------------------
# Example presets
# -----------------------------

def default_camera() -> CameraSetup:
    return CameraSetup()

def default_scene() -> SimulatorConfig:
    return SimulatorConfig()

def test_camera() -> CameraSetup:
    return CameraSetup(
        name = "test_cam",
        W = 2160,
        H = 1620,

        aspect_ratios = ((4,3), (4,3)),
        portrait_prob = 0,
        blur_sigma_global = 0,
        photon_level = 2500,
        read_noise = 0.01
    )

def train_camera() -> CameraSetup:
    return default_camera()

def img_export_camera() -> CameraSetup:
    return CameraSetup(
        name = "img_export_cam",
        W = 2160,
        H = 1620,
        blur_sigma_global = 0,
        photon_level = 2500,
        read_noise = 0.01
    )

def train_scene() -> SimulatorConfig:
    return default_scene()

def test_scene() -> SimulatorConfig:
    return SimulatorConfig(
        well_radius_frac = (0.3, 0.5),
        well_center_jitter = (0.0, 0.04),

        # --- radial look of the well ---
        background_level = (0, 0.3),
        edge_boost = (0, 0.5),
        radial_gamma = 1.2,
        vignette_strength= (0, 0.20),

        # --- cells ---
        n_cells = (10, 2000),
        cell_diameter = (6.0, 10.0),
        frac_positive = (0.0, 1.0),
        color_jitter = (0.0, 0.2),
        sigma_in = (0.9, 1.2),   # pass-through
        sigma_out = (0.9, 1.2), # pass-through
        focus_frac_in = (0.0, 1.0),
        in_focus_sigma_thresh = None,

        boundary_width = 2,

        # crowd near outer wall
        rim_bias = (0.5, 0.95),
        rim_band = (0.1, 0.5),
        edge_clamp = (0.1, 0.3),

        # --- collision / packing ---
        min_cell_sep_px = None,
        rim_min_sep_px = (4, 20),
        pack_iters = (10, 20),
        pack_strength = (0.0, 1.0),
        wall_margin_px = (2.0, 20.0),

        # --- ghosts (outside) ---
        ghost_enable = True,
        ghost_intensity = (0.02, 0.1),
    )

def img_export_scene() -> SimulatorConfig:
    return test_scene()

def phone_mix_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("microscope", "iphone", "pixel"),
        probs=None,
        jpeg_prob=0.3,
    )

def microscope_only_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("microscope",),
        probs=None,
        jpeg_prob=0.3,
    )

def simulated_raw_style() -> CameraStyleConfig:
    return CameraStyleConfig(
        styles=("simulated_raw",),
        probs=None,
        jpeg_prob=0.0,
    )
