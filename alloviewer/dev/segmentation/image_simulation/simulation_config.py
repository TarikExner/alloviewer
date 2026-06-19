from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from .types import RNG
from .utils import (
    sample_number,
    sample_bool,
    is_pair
)
from .camera_dimension_config import CameraDimension

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
    "cluster_seed_tries",
    "cluster_member_tries",
    "cluster_packed_candidate_count",
)

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

    # --- multiplicative background texture inside the well ---
    background_texture_enable: Union[bool, float, Tuple[float, float]] = True
    background_texture_sigma_fine: Union[float, Tuple[float, float]] = (0.25, 0.8)
    background_texture_sigma_coarse: Union[float, Tuple[float, float]] = (1.0, 2.5)
    background_texture_fine_weight: Union[float, Tuple[float, float]] = (0.90, 1.0)
    background_texture_coarse_weight: Union[float, Tuple[float, float]] = (0.0, 0.08)
    background_texture_strength: Union[float, Tuple[float, float]] = (0.01, 0.04)
    background_texture_clip: Tuple[float, float] = (0.1, 1.6)

    # --- cells ---
    n_cells: Union[int, Tuple[int, int]] = (10, 2000)

    # Image-size-calibrated core-diameter model. Each row is:
    #   (image short side in px, minimum core diameter, maximum core diameter)
    # Values outside the measured image-size interval are clamped.
    cell_diameter_bounds_by_short_side: Optional[
        Tuple[Tuple[float, float, float], ...]
    ] = (
        (1620.0, 7.0, 11.0),
        (3024.0, 11.0, 15.0),
    )
    cell_diameter_center_margin_frac: Union[float, Tuple[float, float]] = 0.20
    cell_diameter_sigma_frac: Union[float, Tuple[float, float]] = 0.18
    cell_diameter_min_sigma_px: Union[float, Tuple[float, float]] = 0.25

    # Legacy diameter model. It is used only when
    # cell_diameter_bounds_by_short_side is None.
    cell_diameter: Union[float, Tuple[float, float]] = (7.0, 12.0)

    # Scale cell diameter with image size while keeping direct control over
    # cell_diameter. Scaling uses the short image side: min(H, W).
    cell_diameter_reference_short_side: Union[float, Tuple[float, float]] = 1620.0
    cell_diameter_size_exponent: Union[float, Tuple[float, float]] = 0.75
    cell_diameter_scale_clip: Optional[Tuple[float, float]] = (0.85, 2.0)

    large_cell_frac: Union[float, Tuple[float, float]] = (0.0, 0.2)
    large_cell_diameter_factor: Union[float, Tuple[float, float]] = (1.15, 1.5)

    cell_ellipse_enable: Union[bool, float, Tuple[float, float]] = True
    cell_axis_jitter = (0.0,0.5)          # ±50% axis ratio
    cell_random_rotation: Union[bool, float, Tuple[float, float]] = True
    cell_intensity_range = (0.70, 1.05)  # per-cell brightness multiplier

    frac_positive: Union[float, Tuple[float, float]] = (0.0, 1.0)
    color_jitter: Union[float, Tuple[float, float]] = (0.0, 0.2)
    sigma_in: Union[Tuple[float, float], Tuple[float, float]] = (0.06, 0.1)
    sigma_out: Union[Tuple[float, float], Tuple[float, float]] = (0.08, 0.16)
    focus_frac_in: Union[float, Tuple[float, float]] = (0.0, 1.0)
    in_focus_sigma_thresh: Optional[Union[float, Tuple[float, float]]] = None

    boundary_width: Union[int, Tuple[int, int]] = 2

    # crowd near outer wall
    rim_bias: Union[float, Tuple[float, float]] = (0.5, 0.95)
    rim_band: Union[float, Tuple[float, float]] = (0.1, 0.5)
    edge_clamp: Union[float, Tuple[float, float]] = (0.1, 0.65)

    # --- clustering ---
    cluster_enable: Union[bool, float, Tuple[float, float]] = True
    clustered_cell_frac: Union[float, Tuple[float, float]] = (0.30, 0.80)
    cluster_size_range: Tuple[int, int] = (2, 35)
    cluster_contact_factor_range: Tuple[float, float] = (1.02, 1.5)
    cluster_core_min_sep_factor: Union[float, Tuple[float, float]] = (0.90, 0.94)
    cluster_chain_probability: Union[float, Tuple[float, float]] = (0.25, 0.75)
    cluster_angle_jitter: Union[float, Tuple[float, float]] = (0.55, 1.10)

    # Explicit lengthy-versus-packed cluster geometry. Small clusters are
    # usually lengthy; larger clusters approach cluster_packed_probability.
    cluster_packed_probability: Union[float, Tuple[float, float]] = (0.35, 0.70)
    cluster_packed_size_bias_range: Tuple[int, int] = (3, 15)
    cluster_packed_contact_factor_range: Tuple[float, float] = (1.02, 1.5)
    cluster_packed_candidate_count: Union[int, Tuple[int, int]] = (6, 12)
    cluster_packed_contact_bonus: Union[float, Tuple[float, float]] = (1.0, 2.0)

    # A packed cluster can seed beside an earlier packed cluster. Different
    # cluster IDs may then appear as one larger packed region.
    cluster_packed_region_join_probability: Union[
        float, Tuple[float, float]
    ] = (0.20, 0.45)
    cluster_packed_region_contact_factor_range: Tuple[float, float] = (0.95, 1.12)

    cluster_seed_tries: Union[int, Tuple[int, int]] = 120
    cluster_member_tries: Union[int, Tuple[int, int]] = 32
    cluster_pack_min_sep_factor: Union[float, Tuple[float, float]] = (0.90, 0.94)

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
    ghost_stretch: Union[float, Tuple[float, float]] = (0.2, 3.0)
    ghost_trail: Union[int, Tuple[int, int]] = (1,3)
    ghost_trail_decay: Union[float, Tuple[float, float]] = 0.6

    # --- debris (inside well) ---
    dirt_density: Union[float, Tuple[float, float]] = (0.0, 0.00005)
    dirt_size: Union[Tuple[int, int], Tuple[int, int]] = (1, 8)  # pass-through (discrete range)
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

    def sample_kwargs(self, rng: RNG, camera: Optional[CameraDimension] = None) -> Dict[str, Any]:
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
            out[key] = sample_number(rng, val, integer=integer)

        def set_bool(key: str):
            val = getattr(self, key)
            out[key] = sample_bool(rng, val)

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

        # multiplicative background texture inside the well
        set_bool("background_texture_enable")
        set_num("background_texture_sigma_fine")
        set_num("background_texture_sigma_coarse")
        set_num("background_texture_fine_weight")
        set_num("background_texture_coarse_weight")
        set_num("background_texture_strength")
        set_passthrough("background_texture_clip")

        # cells
        set_num("n_cells", integer=True)
        set_passthrough("cell_diameter_bounds_by_short_side")
        set_num("cell_diameter_center_margin_frac")
        set_num("cell_diameter_sigma_frac")
        set_num("cell_diameter_min_sigma_px")
        set_num("cell_diameter")
        set_num("cell_diameter_reference_short_side")
        set_num("cell_diameter_size_exponent")
        set_passthrough("cell_diameter_scale_clip")
        set_bool("cell_ellipse_enable")
        set_num("cell_axis_jitter")
        set_bool("cell_random_rotation")
        set_passthrough("cell_intensity_range")

        set_num("large_cell_frac")
        set_num("large_cell_diameter_factor")
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

        # clustering
        set_bool("cluster_enable")
        set_num("clustered_cell_frac")
        set_passthrough("cluster_size_range")
        set_passthrough("cluster_contact_factor_range")
        set_num("cluster_core_min_sep_factor")
        set_num("cluster_chain_probability")
        set_num("cluster_angle_jitter")
        set_num("cluster_packed_probability")
        set_passthrough("cluster_packed_size_bias_range")
        set_passthrough("cluster_packed_contact_factor_range")
        set_num("cluster_packed_candidate_count", integer=True)
        set_num("cluster_packed_contact_bonus")
        set_num("cluster_packed_region_join_probability")
        set_passthrough("cluster_packed_region_contact_factor_range")
        set_num("cluster_seed_tries", integer=True)
        set_num("cluster_member_tries", integer=True)
        set_num("cluster_pack_min_sep_factor")

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
        if is_pair(self.ghost_intensity):
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

###############
### PRESETS ###
###############

def default_scene() -> SimulatorConfig:
    return SimulatorConfig()

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

        # --- multiplicative background texture inside the well ---
        background_texture_enable = True,
        background_texture_sigma_fine = (0.25, 0.8),
        background_texture_sigma_coarse = (1.0, 2.5),
        background_texture_fine_weight = (0.90, 1.0),
        background_texture_coarse_weight = (0.0, 0.08),
        background_texture_strength = (0.01, 0.04),
        background_texture_clip = (0.1, 1.6),

        # --- cells ---
        n_cells = (10, 2000),
        cell_diameter = (6.5, 13.5),
        cell_diameter_reference_short_side = 1620.0,
        cell_diameter_size_exponent = 0.95,
        cell_diameter_scale_clip = (0.60, 2.20),
        frac_positive = (0.0, 1.0),
        color_jitter = (0.0, 0.2),
        sigma_in = (0.06, 0.1),   # pass-through
        sigma_out = (0.08, 0.16), # pass-through
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
