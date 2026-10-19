from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .igg import IgGControlStats


@dataclass
class QCResult:
    """
    Store events, QC masks, and diagnostic notes for one file.

    Attributes
    ----------
    events : numpy.ndarray
        Event matrix with one row per event and one column per measurement channel.
    mask_edge : numpy.ndarray
        Boolean mask selecting events that pass detector-edge filtering.
    mask_sing : Optional[numpy.ndarray]
        Boolean singlet mask, or None when singlet gating is unavailable.
    mask_qc : numpy.ndarray
        Boolean mask selecting events retained by QC.
    notes : List[str]
        Diagnostic messages collected while processing the file or sample.
    """
    events: np.ndarray
    mask_edge: np.ndarray
    mask_sing: Optional[np.ndarray]
    mask_qc: np.ndarray
    notes: List[str]


@dataclass
class LymphResult:
    """
    Store the lymphocyte mask and its fitting diagnostics.

    Attributes
    ----------
    mask_lymph : numpy.ndarray
        Boolean mask selecting the retained lymphocyte events.
    info : Dict[str, Any]
        Diagnostic information recorded while selecting the lymphocyte population.
    """
    mask_lymph: np.ndarray
    info: Dict[str, Any]

@dataclass(frozen=True)
class PopulationResult:
    """
    Store the event count and IgG readouts for one population.

    Attributes
    ----------
    label : str
        Display label or identifier for the corresponding item.
    n_events : int
        Number of events in the selected population.
    igg_pos_fraction : float
        Fraction of population events strictly above the transformed IgG cutoff.
    igg_median_raw : float
        Median IgG intensity before the asinh transform.
    igg_median_t : float
        Median IgG intensity after the asinh transform.
    igg_median_shift : float
        Raw IgG median difference relative to the negative-control median.
    igg_median_ratio : float
        Raw IgG median divided by the negative-control median, when defined.
    igg_fluorescence_index : float
        Control-normalized IgG fluorescence index recorded for the population.
    igg_cutoff_t : float
        IgG positivity threshold in asinh-transformed intensity units.
    igg_nc_median_raw : float
        Raw median IgG intensity of the negative-control reference.
    igg_pc_median_raw : Optional[float]
        Raw median IgG intensity of the optional positive-control reference.
    """
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
    """
    Store fitted calibration, scaling, clustering, and control statistics.

    Attributes
    ----------
    panel : Any
        Mapping of scatter, population-marker, and IgG channels.
    config : Any
        Flow-cytometry gating and clustering configuration.
    marker_thresholds : Dict[str, float]
        Per-marker positive thresholds in transformed units.
    marker_cofactors : Dict[str, float]
        Per-marker asinh scale factors fitted on the dataset.
    feature_scaler : Any
        Fitted feature transformation used to standardize new events.
    clusterer : Any
        Clusterer configuration or fitted clusterer object appropriate to the enclosing
        type.
    outlier_score_threshold : float
        Learned outlier-score threshold used when assigning new events.
    cluster_to_type : Dict[int, str]
        Mapping from integer cluster labels to biological population names.
    gate_options : List[str]
        Available gate labels in display order.
    igg_cutoff_by_gate : Dict[str, float]
        Mapping from gate names to IgG thresholds in transformed units.
    igg_control_stats_by_gate : Dict[str, IgGControlStats]
        Mapping from gate names to negative- and positive-control IgG statistics.
    marker_calibration_info : Any
        Diagnostics recorded while fitting marker cofactors and thresholds.
    """
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
    """
    Store population results and notes for one FCS file.

    Attributes
    ----------
    file_name : str
        File path or unique basename identifying the selected FCS file.
    populations : List[PopulationResult]
        Per-population metrics for the analyzed FCS file.
    notes : List[str]
        Diagnostic messages collected while processing the file or sample.
    """
    file_name: str
    populations: List[PopulationResult]
    notes: List[str]


@dataclass
class SampleResult:
    """
    Store per-file and combined population results for a sample.

    Attributes
    ----------
    sample_name : str
        Name identifying the sample whose events or measurements are shown.
    role : str
        Sample role: NC for negative control, PC for positive control, or SAMPLE.
    n_files : int
        Number of FCS files belonging to the sample.
    per_file : List[FileResult]
        Individual analysis results for the sample's FCS files.
    combined : List[PopulationResult]
        Population metrics pooled across the sample's FCS files.
    notes : List[str]
        Diagnostic messages collected while processing the file or sample.
    """
    sample_name: str
    role: str
    n_files: int
    per_file: List[FileResult]
    combined: List[PopulationResult]
    notes: List[str]


@dataclass
class FileAnalysis:
    """
    Cache events and QC, lymphocyte, and marker masks for one file.

    Attributes
    ----------
    events : numpy.ndarray
        Event matrix with one row per event and one column per measurement channel.
    mask_edge : numpy.ndarray
        Boolean mask selecting events that pass detector-edge filtering.
    mask_sing : Optional[numpy.ndarray]
        Boolean singlet mask, or None when singlet gating is unavailable.
    mask_qc : numpy.ndarray
        Boolean mask selecting events retained by QC.
    mask_lymph_raw : numpy.ndarray
        Lymphocyte mask before the final cleanup step.
    mask_lymph : numpy.ndarray
        Boolean mask selecting the retained lymphocyte events.
    m_by_marker : Dict[str, numpy.ndarray]
        Mapping from marker names to event-level population masks.
    notes : List[str]
        Diagnostic messages collected while processing the file or sample.
    """
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
