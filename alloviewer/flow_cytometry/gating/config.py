from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union


SingletMode = Literal["mad", "gmm", "hybrid"]
EventSource = Literal["raw", "comp"]


@dataclass
class TransformConfig:
    """
    Set fluorescence cofactors and the scatter log-transform option.

    Attributes
    ----------
    default_cofactor : float
        Default fluorescence asinh scale factor.
    igg_cofactor : float
        Asinh scale factor used for IgG values. Default is ``150.0``.
    scatter_use_log1p : bool
        Whether scatter values use a log1p transform. Default is ``True``.
    """
    default_cofactor: float = 150.0
    igg_cofactor: float = 150.0
    scatter_use_log1p: bool = True


@dataclass
class EdgeConfig:
    """
    Set the fraction of the detector range used to reject edge events.

    Attributes
    ----------
    hi_frac : float
        Fraction of the detector range above which events are rejected. Default is ``0.95``.
    """
    hi_frac: float = 0.95


@dataclass
class SingletConfig:
    """
    Configure MAD, Gaussian-mixture, and hybrid singlet selection.

    Attributes
    ----------
    mode : SingletMode
        Singlet-selection method: mad, gmm, or hybrid. Default is ``'mad'``.
    min_fsc_quantile : float
        Lower FSC quantile excluded before singlet fitting. Default is ``0.02``.
    min_events : int
        Minimum event count required by the fitting step. Default is ``300``.
    min_keep_fraction : float
        Minimum retained fraction accepted by the gate. Default is ``0.05``.
    max_keep_fraction : float
        Maximum retained fraction accepted by the gate. Default is ``0.98``.
    k_mad : float
        Median-absolute-deviation multiplier for the singlet residual gate. Default is
        ``4.0``.
    prefilter_k_mad : float
        Robust residual cutoff applied before Gaussian-mixture singlet fitting. Default is
        ``5.0``.
    gmm_components : int
        Number of Gaussian-mixture components. Default is ``2``.
    gmm_covariance_type : Literal['tied', 'full']
        Covariance structure passed to the Gaussian mixture used for singlet selection.
        Default is ``'tied'``.
    gmm_resp_threshold : float
        Minimum component responsibility for retaining a mixture-selected singlet. Default
        is ``0.8``.
    gmm_min_events : int
        Minimum event count required for a Gaussian-mixture fit. Default is ``500``.
    gmm_subsample : int
        Maximum number of events used for mixture fitting. Default is ``5000``.
    gmm_reg_covar : float
        Diagonal covariance regularization for the mixture fit. Default is ``1e-06``.
    gmm_med_a_tie_delta : float
        Tolerance for treating component median FSC-area values as tied. Default is
        ``0.05``.
    hybrid_k_mad : float
        Robust residual multiplier applied by the hybrid singlet gate. Default is ``3.5``.
    hybrid_min_keep_rel : float
        Minimum fraction of the prefiltered population retained by the hybrid gate. Default
        is ``0.5``.
    hybrid_max_keep_rel : float
        Maximum fraction of the prefiltered population retained by the hybrid gate. Default
        is ``0.999``.
    """
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
    """
    Configure mixture-based debris removal and retention limits.

    Attributes
    ----------
    gmm_components : int
        Number of Gaussian-mixture components. Default is ``4``.
    quantile_cut : float
        Low-scatter quantile used to identify a candidate debris component. Default is
        ``0.2``.
    max_component_fraction : float
        Largest mixture component fraction accepted as debris. Default is ``0.35``.
    resp_threshold : float
        Minimum mixture responsibility accepted for the selected component. Default is
        ``0.7``.
    min_keep_fraction : float
        Minimum retained fraction accepted by the gate. Default is ``0.05``.
    subsample : int
        Maximum number of events used when fitting the corresponding mixture model. Default
        is ``100000``.
    """
    gmm_components: int = 4
    quantile_cut: float = 0.2
    max_component_fraction: float = 0.35
    resp_threshold: float = 0.70
    min_keep_fraction: float = 0.05
    subsample: int = 100_000


@dataclass
class QCConfig:
    """
    Group the event source and edge, singlet, and debris QC settings.

    Attributes
    ----------
    event_source : EventSource
        Event source used for QC, such as compensated or raw events. Default is ``'comp'``.
    edge : EdgeConfig
        Configuration for excluding events near detector limits. Default is
        ``field(default_factory=EdgeConfig)``.
    singlet : SingletConfig
        Configuration for selecting singlet events. Default is
        ``field(default_factory=SingletConfig)``.
    debris : DebrisConfig
        Configuration for removing low-scatter debris events. Default is
        ``field(default_factory=DebrisConfig)``.
    """
    event_source: EventSource = "comp"
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    singlet: SingletConfig = field(default_factory=SingletConfig)
    debris: DebrisConfig = field(default_factory=DebrisConfig)


@dataclass
class MarkerCofactorConfig:
    """
    Configure estimation of marker-specific asinh cofactors.

    Attributes
    ----------
    max_events_per_marker : int
        Upper limit on events pooled for a marker calibration. Default is ``120000``.
    max_events_per_file : int
        Upper limit on calibration events sampled from each file. Default is ``5000``.
    bins : int
        Number of histogram bins, or explicit bin edges where supported by the called
        histogram routine. Default is ``256``.
    smooth : int
        Smoothing-window size used for the marker intensity histogram. Default is ``7``.
    tail_frac_of_peak : float
        Histogram height fraction used to identify the background peak's tail. Default is
        ``0.15``.
    tail_consecutive_bins : int
        Number of consecutive low histogram bins required to end the background tail.
        Default is ``4``.
    target_t : float
        Target transformed intensity used when deriving the marker cofactor. Default is
        ``1.0``.
    min : float
        Lower permitted marker-cofactor value. Default is ``100.0``.
    max : float
        Upper permitted marker-cofactor value. Default is ``2000.0``.
    iter_max : int
        Maximum cofactor-refinement iterations. Default is ``6``.
    tol_rel : float
        Relative cofactor change below which refinement stops. Default is ``0.02``.
    bg_q : float
        Background-component quantile used for cofactor estimation. Default is ``0.99``.
    min_bg_weight : float
        Minimum mixture weight required for a plausible background component. Default is
        ``0.05``.
    min_mean_sep_t : float
        Minimum separation of mixture means in transformed intensity units. Default is
        ``0.3``.
    clip_t : float
        Absolute transformed-intensity limit used during cofactor fitting. Default is
        ``8.0``.
    """
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
    """
    Configure marker-positivity threshold fitting and fallback quantiles.

    Attributes
    ----------
    gmm_min_events : int
        Minimum event count required for a Gaussian-mixture fit. Default is ``2000``.
    fallback_quantile : float
        Quantile used when the primary threshold fit cannot be used. Default is ``0.8``.
    max_events_per_marker : int
        Upper limit on events pooled for a marker calibration. Default is ``120000``.
    max_events_per_file : int
        Upper limit on calibration events sampled from each file. Default is ``5000``.
    """
    gmm_min_events: int = 2000
    fallback_quantile: float = 0.80
    max_events_per_marker: int = 120_000
    max_events_per_file: int = 5_000


@dataclass
class LymphocyteConfig:
    """
    Configure scatter-based lymphocyte selection and marker guidance.

    Attributes
    ----------
    gmm_components : int
        Number of Gaussian-mixture components. Default is ``3``.
    subsample : int
        Maximum number of events used when fitting the corresponding mixture model. Default
        is ``80000``.
    resp_threshold : float
        Minimum mixture responsibility accepted for the selected component. Default is
        ``0.6``.
    min_keep_fraction : float
        Minimum retained fraction accepted by the gate. Default is ``0.02``.
    min_guide_events : int
        Minimum marker-guided events required to guide the lymphocyte gate. Default is
        ``500``.
    guide_coverage : float
        Target coverage of marker-guided events by the selected scatter components. Default
        is ``0.85``.
    exclude_low_fsc_in_guide : bool
        Whether low-FSC events are removed from the marker-derived guide population. Default
        is ``True``.
    guide_fsc_gmm_min_events : int
        Minimum guide events needed to fit the FSC mixture filter. Default is ``500``.
    guide_fsc_fallback_q : float
        Fallback low-FSC quantile for filtering guide events. Default is ``0.05``.
    fsc_low_q : float
        Lower FSC quantile used for the final scatter restriction. Default is ``0.1``.
    ssc_high_q : float
        Upper SSC quantile used for the final scatter restriction. Default is ``0.99``.
    """
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
    """
    Set the MAD floor and clipping bound for robust feature scaling.

    Attributes
    ----------
    z_eps : float
        Minimum MAD used when scaling features. Default is ``1e-06``.
    z_clip : float
        Absolute bound for scaled feature values. Default is ``8.0``.
    """
    z_eps: float = 1e-6
    z_clip: float = 8.0


@dataclass
class ClusterSamplingConfig:
    """
    Set per-file sampling and marker-tail oversampling for clustering.

    Attributes
    ----------
    subsample_per_file : int
        Number of events sampled per file for clustering. Default is ``5000``.
    tail_boost_q : float
        Quantile defining the high-marker tail used for extra sampling. Default is ``0.99``.
    tail_boost_per_marker : int
        Maximum extra events sampled from each marker tail. Default is ``250``.
    """
    subsample_per_file: int = 5000
    tail_boost_q: float = 0.99
    tail_boost_per_marker: int = 250


@dataclass
class PredictionConfig:
    """
    Set confidence, outlier, and assignment limits for event prediction.

    Attributes
    ----------
    pred_prob_min : float
        Minimum prediction confidence for keeping a cluster assignment. Default is ``0.2``.
    outlier_q : float
        Quantile used to set the training outlier-score cutoff. Default is ``0.99``.
    cluster_assign_margin : float
        Reserved configuration field; the supplied gating code does not read this setting.
        Default is ``0.1``.
    """
    pred_prob_min: float = 0.2
    outlier_q: float = 0.99
    cluster_assign_margin: float = 0.10


@dataclass
class DebrisClusterLabelConfig:
    """
    Set scatter quantiles used to identify debris-like clusters.

    Attributes
    ----------
    fsc_q : float
        Low FSC quantile used to label clusters as debris. Default is ``0.1``.
    ssc_q : float
        Low SSC quantile used to label clusters as debris. Default is ``0.1``.
    """
    fsc_q: float = 0.10
    ssc_q: float = 0.10


@dataclass
class ClusterLabelConfig:
    """
    Set marker-positivity and separation criteria for cluster labels.

    Attributes
    ----------
    min_frac_above : float
        Minimum fraction of cluster events above a marker threshold for that marker to be a
        label candidate. Default is ``0.6``.
    min_median_margin : float
        Minimum cluster median excess over the transformed marker threshold. Default is
        ``0.0``.
    frac_weight : float
        Weight of the above-threshold event fraction in the cluster label score. Default is
        ``0.0``.
    assign_margin : float
        Minimum score difference between the best and second-best marker labels. Default is
        ``0.1``.
    """
    min_frac_above: float = 0.60
    min_median_margin: float = 0.0
    frac_weight: float = 0.0
    assign_margin: float = 0.10

@dataclass
class IgGCutoffConfig:
    """
    Configure IgG control histograms and gate-specific positive cutoffs.

    Attributes
    ----------
    min_events_nc : int
        Minimum negative-control events required by IgG cutoff estimation. Default is
        ``2000``.
    min_events_pc : int
        Minimum positive-control events required by control-based IgG estimation. Default is
        ``2000``.
    hist_bins : int
        Number of histogram bins. Default is ``256``.
    hist_smooth : int
        Smoothing-window size for the IgG intensity histogram. Default is ``7``.
    tail_frac_of_peak : float
        Histogram height fraction used to identify the background peak's tail. Default is
        ``0.15``.
    tail_consecutive_bins : int
        Number of consecutive low histogram bins required to end the background tail.
        Default is ``4``.
    range_q_lo : float
        Lower event-intensity quantile defining the histogram range. Default is ``0.001``.
    range_q_hi : float
        Upper event-intensity quantile defining the histogram range. Default is ``0.999``.
    fallback_quantile : float
        Quantile used when the primary threshold fit cannot be used. Default is ``0.9``.
    buffer : float
        Additional offset added to the estimated IgG cutoff in transformed units. Default is
        ``0.0``.
    """
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
    """
    Store HDBSCAN fitting and prediction options.

    Attributes
    ----------
    algorithm : Literal['hdbscan']
        Name selecting the clustering backend. Default is ``'hdbscan'``.
    min_cluster_size : int
        Minimum cluster size passed to HDBSCAN. Default is ``15``.
    min_samples : Optional[int]
        HDBSCAN density-neighborhood size, or None to use the estimator default.
    max_noise : float
        Maximum tolerated fraction of events assigned to noise during cluster selection.
        Default is ``0.15``.
    cluster_selection_method : Literal['leaf', 'eom']
        HDBSCAN cluster extraction method, such as leaf or excess of mass. Default is
        ``'leaf'``.
    prediction_data : bool
        Whether HDBSCAN retains the data needed for approximate predictions. Default is
        ``True``.
    """
    algorithm: Literal["hdbscan"] = "hdbscan"
    min_cluster_size: int = 15
    min_samples: Optional[int] = 10
    max_noise: float = 0.15
    cluster_selection_method: Literal["leaf", "eom"] = "leaf"
    prediction_data: bool = True


@dataclass
class PARCConfig:
    """
    Store PARC graph-clustering and out-of-sample prediction options.

    Attributes
    ----------
    algorithm : Literal['parc']
        Name selecting the clustering backend. Default is ``'parc'``.
    dist_std_local : float
        Local distance-pruning parameter forwarded to PARC. Default is ``3.0``.
    jac_std_global : Union[float, Literal['median']]
        Global Jaccard-pruning parameter forwarded to PARC. Default is ``'median'``.
    keep_all_local_dist : Union[bool, Literal['auto']]
        PARC option controlling retention of local-distance graph edges. Default is
        ``'auto'``.
    too_big_factor : float
        PARC threshold controlling further processing of oversized clusters. Default is
        ``0.4``.
    small_pop : int
        Small-population size threshold forwarded to PARC. Default is ``10``.
    jac_weighted_edges : bool
        Whether PARC uses Jaccard-weighted graph edges. Default is ``True``.
    knn : int
        Number of nearest neighbors used to construct the PARC graph. Default is ``30``.
    n_iter_leiden : int
        Number of Leiden iterations requested from PARC. Default is ``5``.
    num_threads : int
        Thread-count setting passed to PARC; -1 delegates the choice to the estimator.
        Default is ``-1``.
    distance : Literal['l2', 'ip', 'cosine']
        Distance metric used by the PARC nearest-neighbor graph. Default is ``'l2'``.
    time_smallpop : int
        Small-population processing limit forwarded to PARC. Default is ``15``.
    partition_type : Literal['ModularityVP', 'RBVP']
        Leiden partition type forwarded to PARC. Default is ``'ModularityVP'``.
    resolution_parameter : float
        Resolution parameter passed to PARC's community-detection configuration. Default is
        ``1.0``.
    hnsw_param_ef_construction : int
        HNSW index-construction parameter forwarded to PARC. Default is ``150``.
    allow_oos_prediction : bool
        Whether to allow assignment of events not used during PARC fitting. Default is
        ``True``.
    prediction_k : Optional[int]
        Neighbor count for out-of-sample prediction; None uses the adapter default.
    """
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
    """
    Store FlowSOM cluster counts, label selection, and estimator options.

    Attributes
    ----------
    algorithm : Literal['flowsom']
        Name selecting the clustering backend. Default is ``'flowsom'``.
    n_clusters : int
        Requested number of clusters. Default is ``30``.
    seed : int
        Random seed for reproducible sampling. Default is ``187``.
    label_kind : Literal['cluster', 'metacluster']
        Whether FlowSOM labels represent individual clusters or metaclusters. Default is
        ``'metacluster'``.
    estimator_kwargs : dict[str, Any]
        Additional options forwarded to the clustering estimator. Default is
        ``field(default_factory=dict)``.
    """
    algorithm: Literal["flowsom"] = "flowsom"
    n_clusters: int = 30
    seed: int = 187
    label_kind: Literal["cluster", "metacluster"] = "metacluster"
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)


ClustererConfig = Union[HDBSCANConfig, PARCConfig, FlowSOMConfig]


@dataclass
class GatingConfig:
    """
    Group the configuration for QC, calibration, clustering, and IgG readouts.

    Attributes
    ----------
    random_state : int
        Random seed used by fitting and sampling steps. Default is ``187``.
    transform : TransformConfig
        Configuration of fluorescence and scatter transformations. Default is
        ``field(default_factory=TransformConfig)``.
    qc : QCConfig
        If True, compute QC metrics and filter rejected instances before ROI extraction.
        Default is ``field(default_factory=QCConfig)``.
    marker_cofactor : MarkerCofactorConfig
        Configuration for learning per-marker asinh cofactors. Default is
        ``field(default_factory=MarkerCofactorConfig)``.
    marker_threshold : MarkerThresholdConfig
        Configuration for estimating transformed marker thresholds. Default is
        ``field(default_factory=MarkerThresholdConfig)``.
    lymphocyte : LymphocyteConfig
        Configuration for selecting the lymphocyte scatter population. Default is
        ``field(default_factory=LymphocyteConfig)``.
    feature_scaling : FeatureScalingConfig
        Configuration for robust feature standardization and clipping. Default is
        ``field(default_factory=FeatureScalingConfig)``.
    cluster_sampling : ClusterSamplingConfig
        Configuration for event subsampling and marker-tail enrichment. Default is
        ``field(default_factory=ClusterSamplingConfig)``.
    prediction : PredictionConfig
        Configuration for assigning new events to learned clusters. Default is
        ``field(default_factory=PredictionConfig)``.
    debris_cluster_label : DebrisClusterLabelConfig
        Configuration for identifying debris clusters by scatter medians. Default is
        ``field(default_factory=DebrisClusterLabelConfig)``.
    igg_cutoff : IgGCutoffConfig
        Configuration for estimating control-based IgG positivity cutoffs. Default is
        ``field(default_factory=IgGCutoffConfig)``.
    cluster_label : ClusterLabelConfig
        Configuration for assigning biological labels to learned clusters. Default is
        ``field(default_factory=ClusterLabelConfig)``.
    clusterer : ClustererConfig
        Clusterer configuration or fitted clusterer object appropriate to the enclosing
        type. Default is ``field(default_factory=PARCConfig)``.
    """
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
        """
        Create default gating settings with an HDBSCAN clusterer configuration.

        Parameters
        ----------
        **clusterer_kwargs : dict
            Keyword arguments passed to the selected clusterer configuration.

        Returns
        -------
        GatingConfig
            Default gating settings with an HDBSCAN clusterer configuration.
        """
        return cls(clusterer=HDBSCANConfig(**clusterer_kwargs))

    @classmethod
    def with_parc(cls, **clusterer_kwargs) -> "GatingConfig":
        """
        Create default gating settings with a PARC clusterer configuration.

        Parameters
        ----------
        **clusterer_kwargs : dict
            Keyword arguments passed to the selected clusterer configuration.

        Returns
        -------
        GatingConfig
            Default gating settings with a PARC clusterer configuration.
        """
        return cls(clusterer=PARCConfig(**clusterer_kwargs))

    @classmethod
    def with_flowsom(cls, **clusterer_kwargs) -> "GatingConfig":
        """
        Create default gating settings with a FlowSOM clusterer configuration.

        Parameters
        ----------
        **clusterer_kwargs : dict
            Keyword arguments passed to the selected clusterer configuration.

        Returns
        -------
        GatingConfig
            Default gating settings with a FlowSOM clusterer configuration.
        """
        return cls(clusterer=FlowSOMConfig(**clusterer_kwargs))
