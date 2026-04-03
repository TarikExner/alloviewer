# clustering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    TransformConfig,
    FeatureScalingConfig,
    ClusterSamplingConfig,
    PredictionConfig,
)
from .scaling import MADScaler
from .clusterers import BaseClusterer
from ..panel import Panel
from ..fcs_file import FCSFile


@dataclass
class ClusterPrediction:
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
    """
    Returns:
      feature_scaler, clusterer, outlier_thr, X_train_scatter, T_train, y_train
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
    """
    For HDBSCAN: uses model.outlier_scores_ and prediction_cfg.outlier_q.
    For non-HDBSCAN clusterers: returns +inf.
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
    """
    Returns:
      X_raw:     (n, 2+m) [scatter, markers] in raw feature space (no scaling)
      idx:       (n,)
      X_scatter: (n, 2) scatter in raw feature space
      T_markers: (n, m) markers in asinh(raw/cofactor) space
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
        t_cols.append(np.arcsinh(raw / max(cof, 1e-12)).astype(np.float32, copy=False))

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
    """
    Builds pooled training data from lymph masks stored in file_records:
      - X_train_raw:     (N, 2+m)
      - X_train_scatter: (N, 2)
      - T_train:         (N, m)
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

        train_idx = np.unique(np.concatenate([base_idx, np.asarray(extra, dtype=int)])).astype(int)

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
        pos = pos[(pos >= 0) & (pos < idx_sorted.size) & (idx_sorted[pos] == train_idx)]
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
    """
    Predict clusters for events indexed by idx, remove outliers and debris clusters,
    return full-length masks for later counting.
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

    is_out = (y == -1)
    if prob is not None:
        is_out |= (np.asarray(prob, dtype=float) < float(prediction_cfg.pred_prob_min))
    if out_score is not None and np.isfinite(float(outlier_thr)):
        is_out |= (np.asarray(out_score, dtype=float) > float(outlier_thr))

    is_debris = np.zeros_like(is_out, dtype=bool)
    for i, cid in enumerate(y):
        if is_out[i]:
            continue
        if cluster_to_type.get(int(cid), "Unknown") == "Debris":
            is_debris[i] = True

    keep = (~is_out) & (~is_debris)

    mask_all_in_lymph = np.zeros(n_events_total, dtype=bool)
    mask_all_in_lymph[idx] = keep

    mask_by_marker: Dict[str, np.ndarray] = {m: np.zeros(n_events_total, dtype=bool) for m in marker_names}
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
        strength=np.asarray(prob, dtype=float) if prob is not None else np.zeros_like(y, dtype=float),
        outlier_score=np.asarray(out_score, dtype=float) if out_score is not None else np.zeros_like(y, dtype=float),
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
