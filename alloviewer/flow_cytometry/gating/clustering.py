from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clusterers import BaseClusterer
from .config import (
    ClusterSamplingConfig,
    FeatureScalingConfig,
    PredictionConfig,
    TransformConfig,
)
from .scaling import MADScaler
from ..fcs_file import FCSFile
from ..panel import Panel


@dataclass
class ClusterPrediction:
    """Cluster prediction result mapped back to full event length.

    Parameters
    ----------
    idx_used : numpy.ndarray
        Event indices used for prediction.
    y : numpy.ndarray
        Predicted cluster labels for ``idx_used``.
    strength : numpy.ndarray
        Prediction strengths or probabilities.
    outlier_score : numpy.ndarray
        Outlier scores for predicted events.
    is_out : numpy.ndarray
        Boolean mask over predicted events marking outliers.
    is_debris : numpy.ndarray
        Boolean mask over predicted events marking debris clusters.
    mask_all_in_lymph : numpy.ndarray
        Full-length mask for non-outlier, non-debris events inside the
        lymphocyte gate.
    mask_by_marker : dict
        Full-length marker masks keyed by marker name.
    mask_lymph_union : numpy.ndarray
        Full-length union mask across marker-assigned lymphocyte events.
    """

    idx_used: np.ndarray
    y: np.ndarray
    strength: np.ndarray
    outlier_score: np.ndarray
    is_out: np.ndarray
    is_debris: np.ndarray
    mask_all_in_lymph: np.ndarray
    mask_by_marker: Dict[str, np.ndarray]
    mask_lymph_union: np.ndarray


def fit_clustering(
    *,
    transform_cfg: TransformConfig,
    feature_scaling_cfg: FeatureScalingConfig,
    cluster_sampling_cfg: ClusterSamplingConfig,
    prediction_cfg: PredictionConfig,
    panel: Panel,
    file_records: List[Dict[str, Any]],
    marker_cofactors: Dict[str, float],
    clusterer: BaseClusterer,
    rng: np.random.Generator,
) -> Tuple[MADScaler, BaseClusterer, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the global clustering model.

    Parameters
    ----------
    transform_cfg : TransformConfig
        Transformation settings for scatter and marker channels.
    feature_scaling_cfg : FeatureScalingConfig
        Feature-scaling settings.
    cluster_sampling_cfg : ClusterSamplingConfig
        Settings for building the pooled clustering training set.
    prediction_cfg : PredictionConfig
        Prediction and outlier-threshold settings.
    panel : Panel
        Panel definition containing scatter and marker channels.
    file_records : list of dict
        Per-file records containing FCS objects, event matrices, and lymphocyte
        masks.
    marker_cofactors : dict
        Marker-specific transformation cofactors.
    clusterer : BaseClusterer
        Clusterer backend to fit.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    feature_scaler : MADScaler
        Fitted feature scaler.
    clusterer : BaseClusterer
        Fitted clusterer.
    outlier_thr : float
        Outlier-score threshold used during prediction.
    X_train_scatter : numpy.ndarray
        Raw scatter feature block used for training.
    T_train : numpy.ndarray
        Transformed marker feature block used for training.
    y_train : numpy.ndarray
        Cluster labels assigned to training events.

    Raises
    ------
    ValueError
        If fewer than 2000 lymphocyte events are available for clustering.
    """
    X_train_raw, X_train_scatter, T_train = build_training_pool(
        transform_cfg=transform_cfg,
        cluster_sampling_cfg=cluster_sampling_cfg,
        panel=panel,
        file_records=file_records,
        marker_cofactors=marker_cofactors,
        rng=rng,
    )

    if X_train_raw.shape[0] < 2000:
        raise ValueError("Too few lymph events to fit clustering. Add data or relax gating.")

    feature_scaler = MADScaler.fit(
        X_train_raw,
        eps=float(feature_scaling_cfg.z_eps),
        clip=float(feature_scaling_cfg.z_clip),
    )
    X_train = feature_scaler.transform(X_train_raw)

    clusterer = clusterer.fit(X_train)
    y_train = np.asarray(clusterer.labels_, dtype=int)

    outlier_thr = compute_outlier_threshold(
        prediction_cfg=prediction_cfg,
        clusterer=clusterer,
    )

    return feature_scaler, clusterer, outlier_thr, X_train_scatter, T_train, y_train


def compute_outlier_threshold(
    *,
    prediction_cfg: PredictionConfig,
    clusterer: BaseClusterer,
) -> float:
    """Compute the outlier threshold for prediction.

    Parameters
    ----------
    prediction_cfg : PredictionConfig
        Prediction configuration containing the outlier quantile.
    clusterer : BaseClusterer
        Fitted clusterer.

    Returns
    -------
    float
        Quantile-based outlier threshold for HDBSCAN-like models with
        ``outlier_scores_``. Returns positive infinity when no training outlier
        scores are available.
    """
    model = getattr(clusterer, "model", None)
    out_train = getattr(model, "outlier_scores_", None) if model is not None else None

    if out_train is None:
        return float("inf")

    out_train = np.asarray(out_train, dtype=float)

    if out_train.size == 0:
        return float("inf")

    return float(np.quantile(out_train, float(prediction_cfg.outlier_q)))


def _log1p_nonneg(x: np.ndarray) -> np.ndarray:
    """Apply ``log1p`` to non-negative float32 values.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.

    Returns
    -------
    numpy.ndarray
        ``log1p(max(x, 0))`` as ``float32``.
    """
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32, copy=False)


def build_feature_blocks(
    *,
    transform_cfg: TransformConfig,
    panel: Panel,
    fcs: FCSFile,
    events: np.ndarray,
    mask_in: np.ndarray,
    marker_cofactors: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build clustering feature blocks for one file.

    Parameters
    ----------
    transform_cfg : TransformConfig
        Transformation settings.
    panel : Panel
        Panel definition containing scatter and marker channels.
    fcs : FCSFile
        FCS file used for channel index lookup.
    events : numpy.ndarray
        Event matrix.
    mask_in : numpy.ndarray
        Boolean mask selecting events used for feature extraction.
    marker_cofactors : dict
        Marker-specific transformation cofactors.

    Returns
    -------
    X_raw : numpy.ndarray
        Raw feature matrix with shape ``(n, 2 + m)`` containing scatter and
        transformed marker features before scaling.
    idx : numpy.ndarray
        Original event indices selected by ``mask_in``.
    X_scatter : numpy.ndarray
        Scatter feature matrix with shape ``(n, 2)``.
    T_markers : numpy.ndarray
        Transformed marker feature matrix with shape ``(n, m)``.

    Raises
    ------
    ValueError
        If required scatter channels are missing from the panel.
    """
    idx = np.flatnonzero(np.asarray(mask_in, dtype=bool))

    if idx.size == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            idx,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    if not (panel.fsc_a and panel.ssc_a):
        raise ValueError("Need panel.fsc_a and panel.ssc_a for clustering features.")

    jf = int(fcs.get_channel_index(panel.fsc_a))
    js = int(fcs.get_channel_index(panel.ssc_a))
    fsc = events[:, jf].astype(np.float32, copy=False)[idx]
    ssc = events[:, js].astype(np.float32, copy=False)[idx]

    if bool(transform_cfg.scatter_use_log1p):
        xf = _log1p_nonneg(fsc)
        xs = _log1p_nonneg(ssc)
    else:
        xf = fsc
        xs = ssc

    X_scatter = np.column_stack([xf, xs]).astype(np.float32, copy=False)

    t_cols: List[np.ndarray] = []

    for mname, ch in (panel.markers or {}).items():
        j = int(fcs.get_channel_index(ch))
        raw = events[:, j].astype(np.float32, copy=False)[idx]
        cof = float(marker_cofactors.get(mname, transform_cfg.default_cofactor))
        t_cols.append(
            np.arcsinh(raw / max(cof, 1e-12)).astype(np.float32, copy=False)
        )

    T_markers = (
        np.column_stack(t_cols).astype(np.float32, copy=False)
        if t_cols
        else np.zeros((idx.size, 0), dtype=np.float32)
    )

    X_raw = np.column_stack([X_scatter, T_markers]).astype(np.float32, copy=False)

    return X_raw, idx, X_scatter, T_markers


def build_training_pool(
    *,
    transform_cfg: TransformConfig,
    cluster_sampling_cfg: ClusterSamplingConfig,
    panel: Panel,
    file_records: List[Dict[str, Any]],
    marker_cofactors: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pooled clustering training data from file records.

    Parameters
    ----------
    transform_cfg : TransformConfig
        Transformation settings.
    cluster_sampling_cfg : ClusterSamplingConfig
        Sampling settings for training data.
    panel : Panel
        Panel definition containing scatter and marker channels.
    file_records : list of dict
        File records containing ``"fcs"``, ``"events"``, and ``"mask_lymph"``.
    marker_cofactors : dict
        Marker-specific transformation cofactors.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    X_train_raw : numpy.ndarray
        Pooled raw feature matrix with shape ``(N, 2 + m)``.
    X_train_scatter : numpy.ndarray
        Pooled scatter feature matrix with shape ``(N, 2)``.
    T_train : numpy.ndarray
        Pooled transformed marker feature matrix with shape ``(N, m)``.
    """
    X_raw_parts: List[np.ndarray] = []
    X_scatter_parts: List[np.ndarray] = []
    T_parts: List[np.ndarray] = []

    k_per_file = int(cluster_sampling_cfg.subsample_per_file)
    q_tail = float(cluster_sampling_cfg.tail_boost_q)
    tail_per_marker = int(cluster_sampling_cfg.tail_boost_per_marker)

    for rec in file_records:
        fcs: FCSFile = rec["fcs"]
        ev = rec["events"]
        mask_lymph = np.asarray(rec["mask_lymph"], dtype=bool)

        if not np.any(mask_lymph):
            continue

        idx_lymph = np.flatnonzero(mask_lymph)

        if idx_lymph.size > k_per_file > 0:
            base_idx = rng.choice(idx_lymph, size=k_per_file, replace=False)
        else:
            base_idx = idx_lymph

        extra: List[int] = []

        for _mname, ch in (panel.markers or {}).items():
            j = int(fcs.get_channel_index(ch))
            vals_full = ev[:, j].astype(np.float32, copy=False)
            vals_lymph = vals_full[idx_lymph]
            vals_lymph = vals_lymph[np.isfinite(vals_lymph)]

            if vals_lymph.size < 200:
                continue

            thr = float(np.quantile(vals_lymph, q_tail))
            pick = idx_lymph[vals_full[idx_lymph] >= thr]

            if pick.size:
                kk = min(tail_per_marker, pick.size)
                extra.extend(rng.choice(pick, size=kk, replace=False).tolist())

        train_idx = np.unique(
            np.concatenate([base_idx, np.asarray(extra, dtype=int)])
        ).astype(int)

        X_raw_all, idx_all, X_scatter_all, T_markers_all = build_feature_blocks(
            transform_cfg=transform_cfg,
            panel=panel,
            fcs=fcs,
            events=ev,
            mask_in=mask_lymph,
            marker_cofactors=marker_cofactors,
        )

        if X_raw_all.size == 0 or idx_all.size == 0:
            continue

        order = np.argsort(idx_all)
        idx_sorted = idx_all[order]
        pos = np.searchsorted(idx_sorted, train_idx)
        pos = pos[
            (pos >= 0)
            & (pos < idx_sorted.size)
            & (idx_sorted[pos] == train_idx)
        ]

        if pos.size == 0:
            continue

        sel = order[pos]
        X_raw_parts.append(X_raw_all[sel, :])
        X_scatter_parts.append(X_scatter_all[sel, :])
        T_parts.append(T_markers_all[sel, :])

    if not X_raw_parts:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    return (
        np.vstack(X_raw_parts).astype(np.float32, copy=False),
        np.vstack(X_scatter_parts).astype(np.float32, copy=False),
        np.vstack(T_parts).astype(np.float32, copy=False),
    )


def predict_in_mask(
    *,
    prediction_cfg: PredictionConfig,
    clusterer: BaseClusterer,
    feature_scaler: Optional[MADScaler],
    outlier_thr: float,
    X_raw: np.ndarray,
    idx: np.ndarray,
    cluster_to_type: Dict[int, str],
    marker_names: List[str],
    n_events_total: int,
) -> ClusterPrediction:
    """Predict clusters for selected events and build full-length masks.

    Parameters
    ----------
    prediction_cfg : PredictionConfig
        Prediction settings.
    clusterer : BaseClusterer
        Fitted clusterer.
    feature_scaler : MADScaler or None
        Fitted feature scaler. If ``None``, raw features are used directly.
    outlier_thr : float
        Outlier-score threshold.
    X_raw : numpy.ndarray
        Feature matrix before scaling.
    idx : numpy.ndarray
        Original event indices corresponding to rows in ``X_raw``.
    cluster_to_type : dict
        Mapping from cluster ID to marker type or ``"Debris"``.
    marker_names : list of str
        Marker names used to initialize marker masks.
    n_events_total : int
        Number of events in the original file.

    Returns
    -------
    ClusterPrediction
        Predicted labels, outlier/debris flags, and full-length gate masks.
    """
    if X_raw.size == 0 or idx.size == 0:
        z = np.zeros(n_events_total, dtype=bool)

        return ClusterPrediction(
            idx_used=np.asarray(idx, dtype=int),
            y=np.zeros(0, dtype=int),
            strength=np.zeros(0, dtype=float),
            outlier_score=np.zeros(0, dtype=float),
            is_out=np.zeros(0, dtype=bool),
            is_debris=np.zeros(0, dtype=bool),
            mask_all_in_lymph=z.copy(),
            mask_by_marker={m: z.copy() for m in marker_names},
            mask_lymph_union=z.copy(),
        )

    X = feature_scaler.transform(X_raw) if feature_scaler is not None else X_raw
    pred = clusterer.predict(X)

    y = np.asarray(pred.labels, dtype=int)
    prob = pred.prob
    out_score = pred.outlier_score

    is_out = y == -1

    if prob is not None:
        is_out |= np.asarray(prob, dtype=float) < float(prediction_cfg.pred_prob_min)

    if out_score is not None and np.isfinite(float(outlier_thr)):
        is_out |= np.asarray(out_score, dtype=float) > float(outlier_thr)

    is_debris = np.zeros_like(is_out, dtype=bool)

    for i, cid in enumerate(y):
        if is_out[i]:
            continue

        if cluster_to_type.get(int(cid), "Unknown") == "Debris":
            is_debris[i] = True

    keep = (~is_out) & (~is_debris)

    mask_all_in_lymph = np.zeros(n_events_total, dtype=bool)
    mask_all_in_lymph[idx] = keep

    mask_by_marker: Dict[str, np.ndarray] = {
        m: np.zeros(n_events_total, dtype=bool)
        for m in marker_names
    }
    mask_union = np.zeros(n_events_total, dtype=bool)

    for i, cid in enumerate(y):
        if not keep[i]:
            continue

        t = cluster_to_type.get(int(cid), "Unknown")

        if t in mask_by_marker:
            mask_by_marker[t][idx[i]] = True
            mask_union[idx[i]] = True

    return ClusterPrediction(
        idx_used=np.asarray(idx, dtype=int),
        y=y,
        strength=(
            np.asarray(prob, dtype=float)
            if prob is not None
            else np.zeros_like(y, dtype=float)
        ),
        outlier_score=(
            np.asarray(out_score, dtype=float)
            if out_score is not None
            else np.zeros_like(y, dtype=float)
        ),
        is_out=np.asarray(is_out, dtype=bool),
        is_debris=np.asarray(is_debris, dtype=bool),
        mask_all_in_lymph=mask_all_in_lymph,
        mask_by_marker=mask_by_marker,
        mask_lymph_union=mask_union,
    )


def predict_file_in_mask(
    *,
    transform_cfg: TransformConfig,
    prediction_cfg: PredictionConfig,
    panel: Panel,
    fcs: FCSFile,
    events: np.ndarray,
    mask_in: np.ndarray,
    marker_cofactors: Dict[str, float],
    clusterer: BaseClusterer,
    feature_scaler: Any,
    outlier_thr: float,
    cluster_to_type: Dict[int, str],
    marker_names: List[str],
) -> ClusterPrediction:
    """Predict clusters for selected events in one FCS file.

    Parameters
    ----------
    transform_cfg : TransformConfig
        Transformation settings.
    prediction_cfg : PredictionConfig
        Prediction settings.
    panel : Panel
        Panel definition containing scatter and marker channels.
    fcs : FCSFile
        FCS file used for channel lookup.
    events : numpy.ndarray
        Event matrix.
    mask_in : numpy.ndarray
        Boolean mask selecting events to predict.
    marker_cofactors : dict
        Marker-specific transformation cofactors.
    clusterer : BaseClusterer
        Fitted clusterer.
    feature_scaler : Any
        Fitted feature scaler.
    outlier_thr : float
        Outlier-score threshold.
    cluster_to_type : dict
        Mapping from cluster ID to marker type or ``"Debris"``.
    marker_names : list of str
        Marker names used to initialize marker masks.

    Returns
    -------
    ClusterPrediction
        Predicted labels, outlier/debris flags, and full-length masks.
    """
    X_raw, idx, _, _ = build_feature_blocks(
        transform_cfg=transform_cfg,
        panel=panel,
        fcs=fcs,
        events=events,
        mask_in=mask_in,
        marker_cofactors=marker_cofactors,
    )

    return predict_in_mask(
        prediction_cfg=prediction_cfg,
        clusterer=clusterer,
        feature_scaler=feature_scaler,
        outlier_thr=outlier_thr,
        X_raw=X_raw,
        idx=idx,
        cluster_to_type=cluster_to_type,
        marker_names=marker_names,
        n_events_total=int(events.shape[0]),
    )
