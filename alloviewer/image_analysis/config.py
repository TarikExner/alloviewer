import torch
from pathlib import Path

from .segmenter import SegmenterConfig, InstanceSegmenterConfig
from .qc import QCMonitorConfig

HERE = Path(__file__).resolve()
PKG_DIR= HERE.parent.parent
MODELS_DIR = PKG_DIR / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CALIB_RG_GAUSS = {
    "method": "pc_nc_gaussian_rg_default",
    "mu_pc": 1.60,   # orange mean R/G
    "sd_pc": 0.25,   # some spread from jitter/blur
    "mu_nc": 0.06,   # green mean R/G
    "sd_nc": 0.02,   # small spread
}

WELL_QC_CONFIG = QCMonitorConfig()

INSTANCE_CONFIG = InstanceSegmenterConfig()
INSTANCE_CONFIG_DICT = INSTANCE_CONFIG.to_dict()

UNET_CONFIG = SegmenterConfig.from_dict({
    "device": DEVICE,
    "use_amp": DEVICE == "cuda",
    "model_dir": MODELS_DIR,
    "model_file": None,
}).to_dict()

CDC_SUMMARY_CONFIG = {
    "positive_cutoff": 20.0,
    "borderline_low": 15.0,
    "borderline_high": 25.0,
    "min_rois": 50,
    "max_uncertain_fraction": 0.25,
    "min_dynamic_range": 30.0,
    "max_replicate_range": 20.0,
    "weak_positive": 20.0,
    "moderate_positive": 40.0,
    "strong_positive": 70.0,
}
    
