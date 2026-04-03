from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union


SingletMode = Literal["mad", "gmm", "hybrid"]
EventSource = Literal["raw", "comp"]


@dataclass
class TransformConfig:
    default_cofactor: float = 150.0
    igg_cofactor: float = 150.0
    scatter_use_log1p: bool = True


@dataclass
class EdgeConfig:
    hi_frac: float = 0.95


@dataclass
class SingletConfig:
    mode: SingletMode = "mad"

    min_fsc_quantile: float = 0.02
    min_events: int = 300
    min_keep_fraction: float = 0.05
    max_keep_fraction: float = 0.98

    k_mad: float = 4.0
    prefilter_k_mad: float = 5.0

    gmm_components: int = 2
    gmm_covariance_type: Literal["tied", "full"] = "tied"
    gmm_resp_threshold: float = 0.8
    gmm_min_events: int = 500
    gmm_subsample: int = 5000
    gmm_reg_covar: float = 1e-6
    gmm_med_a_tie_delta: float = 0.05

    hybrid_k_mad: float = 3.5
    hybrid_min_keep_rel: float = 0.50
    hybrid_max_keep_rel: float = 0.999


@dataclass
class DebrisConfig:
    gmm_components: int = 4
    quantile_cut: float = 0.2
    max_component_fraction: float = 0.35
    resp_threshold: float = 0.70
    min_keep_fraction: float = 0.05
    subsample: int = 100_000


@dataclass
class QCConfig:
    event_source: EventSource = "comp"
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    singlet: SingletConfig = field(default_factory=SingletConfig)
    debris: DebrisConfig = field(default_factory=DebrisConfig)


@dataclass
class MarkerCofactorConfig:
    max_events_per_marker: int = 120_000
    max_events_per_file: int = 5_000
    bins: int = 256
    smooth: int = 7
    tail_frac_of_peak: float = 0.15
    tail_consecutive_bins: int = 4
    target_t: float = 1.0
    min: float = 100.0
    max: float = 2000.0

    iter_max: int = 6
    tol_rel: float = 0.02
    bg_q: float = 0.99
    min_bg_weight: float = 0.05
    min_mean_sep_t: float = 0.30
    clip_t: float = 8.0


@dataclass
class MarkerThresholdConfig:
    gmm_min_events: int = 2000
    fallback_quantile: float = 0.80
    max_events_per_marker: int = 120_000
    max_events_per_file: int = 5_000


@dataclass
class LymphocyteConfig:
    gmm_components: int = 3
    subsample: int = 80_000
    resp_threshold: float = 0.60
    min_keep_fraction: float = 0.02

    min_guide_events: int = 500
    guide_coverage: float = 0.85

    exclude_low_fsc_in_guide: bool = True
    guide_fsc_gmm_min_events: int = 500
    guide_fsc_fallback_q: float = 0.05

    fsc_low_q: float = 0.10
    ssc_high_q: float = 0.99


@dataclass
class FeatureScalingConfig:
    z_eps: float = 1e-6
    z_clip: float = 8.0


@dataclass
class ClusterSamplingConfig:
    subsample_per_file: int = 5000
    tail_boost_q: float = 0.99
    tail_boost_per_marker: int = 250


@dataclass
class PredictionConfig:
    pred_prob_min: float = 0.2
    outlier_q: float = 0.99
    cluster_assign_margin: float = 0.10


@dataclass
class DebrisClusterLabelConfig:
    fsc_q: float = 0.10
    ssc_q: float = 0.10


@dataclass
class ClusterLabelConfig:
    min_frac_above: float = 0.60
    min_median_margin: float = 0.0
    frac_weight: float = 0.0
    assign_margin: float = 0.10

@dataclass
class IgGCutoffConfig:
    min_events_nc: int = 2000
    min_events_pc: int = 2000
    hist_bins: int = 256
    hist_smooth: int = 7
    tail_frac_of_peak: float = 0.15
    tail_consecutive_bins: int = 4
    range_q_lo: float = 0.001
    range_q_hi: float = 0.999
    fallback_quantile: float = 0.90
    buffer: float = 0.0


@dataclass
class HDBSCANConfig:
    algorithm: Literal["hdbscan"] = "hdbscan"
    min_cluster_size: int = 15
    min_samples: Optional[int] = 10
    max_noise: float = 0.15
    cluster_selection_method: Literal["leaf", "eom"] = "leaf"
    prediction_data: bool = True


@dataclass
class PARCConfig:
    algorithm: Literal["parc"] = "parc"
    dist_std_local: float = 3.0
    jac_std_global: Union[float, Literal["median"]] = "median"
    keep_all_local_dist: Union[bool, Literal["auto"]] = "auto"
    too_big_factor: float = 0.4
    small_pop: int = 10
    jac_weighted_edges: bool = True
    knn: int = 30
    n_iter_leiden: int = 5
    num_threads: int = -1
    distance: Literal["l2", "ip", "cosine"] = "l2"
    time_smallpop: int = 15
    partition_type: Literal["ModularityVP", "RBVP"] = "ModularityVP"
    resolution_parameter: float = 1.0
    hnsw_param_ef_construction: int = 150
    allow_oos_prediction: bool = True
    prediction_k: Optional[int] = None


@dataclass
class FlowSOMConfig:
    algorithm: Literal["flowsom"] = "flowsom"
    n_clusters: int = 30
    seed: int = 187
    label_kind: Literal["cluster", "metacluster"] = "metacluster"
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)


ClustererConfig = Union[HDBSCANConfig, PARCConfig, FlowSOMConfig]


@dataclass
class GatingConfig:
    random_state: int = 187

    transform: TransformConfig = field(default_factory=TransformConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    marker_cofactor: MarkerCofactorConfig = field(default_factory=MarkerCofactorConfig)
    marker_threshold: MarkerThresholdConfig = field(default_factory=MarkerThresholdConfig)
    lymphocyte: LymphocyteConfig = field(default_factory=LymphocyteConfig)
    feature_scaling: FeatureScalingConfig = field(default_factory=FeatureScalingConfig)
    cluster_sampling: ClusterSamplingConfig = field(default_factory=ClusterSamplingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    debris_cluster_label: DebrisClusterLabelConfig = field(default_factory=DebrisClusterLabelConfig)
    igg_cutoff: IgGCutoffConfig = field(default_factory=IgGCutoffConfig)
    cluster_label: ClusterLabelConfig = field(default_factory=ClusterLabelConfig)

    clusterer: ClustererConfig = field(default_factory=PARCConfig)

    @classmethod
    def with_hdbscan(cls, **clusterer_kwargs) -> "GatingConfig":
        return cls(clusterer=HDBSCANConfig(**clusterer_kwargs))

    @classmethod
    def with_parc(cls, **clusterer_kwargs) -> "GatingConfig":
        return cls(clusterer=PARCConfig(**clusterer_kwargs))

    @classmethod
    def with_flowsom(cls, **clusterer_kwargs) -> "GatingConfig":
        return cls(clusterer=FlowSOMConfig(**clusterer_kwargs))
