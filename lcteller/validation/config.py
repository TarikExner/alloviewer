import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class TrainingValidationConfig:
    h5_path: str
    out_csv: Optional[str] = None
    out_summary_json: Optional[str] = None
    batch_size: int = 8
    workers: int = 4
    cell_thr: float = 0.5
    center_peak_thr: float = 0.2
    center_nms_dist: int = 3
    center_match_radius: int = 10
    ap_thr_list: Sequence[float] = tuple(np.linspace(0.05, 0.7, 14))
    oks_thresholds: Sequence[float] = (0.5, 0.75, 0.9)
    boundary_thr: float = 0.9
    boundary_tol: int = 2
    boundary_sweep: bool = False
    energy_frac_delta: float = 0.05  # as fraction of GT range
