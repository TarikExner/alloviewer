import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class TrainingValidationConfig:
    """
    Configure validation data loading, segmentation thresholds, and metric tolerances.

    Attributes
    ----------
    h5_path : str
        Path to the simulated-image HDF5 dataset.
    out_csv : Optional[str]
        Optional destination path for per-record validation metrics. Default is ``None``.
    out_summary_json : Optional[str]
        Optional destination path for validation summary statistics. Default is ``None``.
    batch_size : int
        Number of dataset records per DataLoader batch. Default is ``8``.
    workers : int
        Number of DataLoader worker processes. Default is ``12``.
    cell_thr : float
        Threshold used to binarize cell probabilities. Default is ``0.5``.
    center_peak_thr : float
        Minimum probability accepted as a predicted center peak. Default is ``0.2``.
    center_nms_dist : int
        Center nonmaximum-suppression radius in pixels. Default is ``3``.
    center_match_radius : int
        Largest center-assignment distance in pixels. Default is ``10``.
    ap_thr_list : Sequence[float]
        Probability thresholds used to compute center average precision. Default is
        ``tuple(np.linspace(0.05, 0.7, 14))``.
    oks_thresholds : Sequence[float]
        Object-keypoint similarity thresholds evaluated for center matching. Default is
        ``(0.5, 0.75, 0.9)``.
    boundary_thr : float
        Threshold used to binarize boundary probabilities. Default is ``0.9``.
    boundary_tol : int
        Allowed boundary-matching distance in pixels. Default is ``2``.
    boundary_sweep : bool
        Whether to evaluate multiple boundary thresholds. Default is ``False``.
    energy_frac_delta : float
        Fraction of the reference energy range used as an error tolerance. Default is
        ``0.05``.
    """
    h5_path: str
    out_csv: Optional[str] = None
    out_summary_json: Optional[str] = None
    batch_size: int = 8
    workers: int = 12
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
