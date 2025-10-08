import torch
from pathlib import Path

HERE = Path(__file__).resolve()
PKG_DIR= HERE.parent
MODELS_DIR = PKG_DIR / "models"
DEVICE = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CALIB_RG_GAUSS = {
    "method": "pc_nc_gaussian_rg_default",
    "mu_pc": 1.60,   # orange mean R/G
    "sd_pc": 0.25,   # some spread from jitter/blur
    "mu_nc": 0.06,   # green mean R/G
    "sd_nc": 0.02,   # small spread
}

INSTANCE_CONFIG = {
    "min_object_area": 80,
    "min_hole_area": 20,
    "min_instance_area": 100,

    "distance_smooth_sigma": 0.0,
    "use_boundary": True,
    "gamma": 3.0,
    "smooth_boundary_sigma": 0.0,

    "use_edge_term": True,
    "edge_sigma": 1.0,
    "edge_weight": 1.5,

    "seed_method": "hmax",          # or "spacing"
    "h_maxima": 0.6,                # lower -> more seeds
    "min_peak_distance": 8,         # used if seed_method="spacing"
    "marker_erosion_radius": 1,

    "compactness": 0.0,
    "watershed_line": True,
}

UNET_CONFIG = {
    "unet_mode": "small",
    "model_dir": MODELS_DIR,
    "model_file": None,
    "device": DEVICE,
    "thr_cell": 0.5,
    "thr_bound": 0.5,
    "use_amp": True if DEVICE == "cuda" else False,
    "return_logits": False,
    "compute_instances": True,
}
