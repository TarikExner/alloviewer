from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .igg import IgGControlStats


@dataclass
class QCResult:
    events: np.ndarray
    mask_edge: np.ndarray
    mask_sing: Optional[np.ndarray]
    mask_qc: np.ndarray
    notes: List[str]


@dataclass
class LymphResult:
    mask_lymph: np.ndarray
    info: Dict[str, Any]

@dataclass(frozen=True)
class PopulationResult:
    label: str
    n_events: int
    igg_pos_fraction: float
    igg_median_raw: float
    igg_median_t: float
    igg_median_shift: float
    igg_median_ratio: float
    igg_fluorescence_index: float
    igg_cutoff_t: float
    igg_nc_median_raw: float
    igg_pc_median_raw: Optional[float]


@dataclass(frozen=True)
class FittedGater:
    panel: Any
    config: Any
    marker_thresholds: Dict[str, float]
    marker_cofactors: Dict[str, float]
    feature_scaler: Any
    clusterer: Any
    outlier_score_threshold: float
    cluster_to_type: Dict[int, str]
    gate_options: List[str]
    igg_cutoff_by_gate: Dict[str, float]
    igg_control_stats_by_gate: Dict[str, IgGControlStats]
    marker_calibration_info: Any


@dataclass
class FileResult:
    file_name: str
    populations: List[PopulationResult]
    notes: List[str]


@dataclass
class SampleResult:
    sample_name: str
    role: str
    n_files: int
    per_file: List[FileResult]
    combined: List[PopulationResult]
    notes: List[str]


@dataclass
class FileAnalysis:
    events: np.ndarray

    # pure gating masks (full length)
    mask_edge: np.ndarray
    mask_sing: Optional[np.ndarray]
    mask_qc: np.ndarray

    # lymph gating
    mask_lymph_raw: np.ndarray
    mask_lymph: np.ndarray

    # populations (full length), only within cleaned lymph
    m_by_marker: Dict[str, np.ndarray]

    notes: List[str]
