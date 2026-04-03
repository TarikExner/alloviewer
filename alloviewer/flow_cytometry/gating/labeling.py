from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import DebrisClusterLabelConfig, ClusterLabelConfig


def label_clusters(
    *,
    debris_cfg: DebrisClusterLabelConfig,
    cluster_label_cfg: ClusterLabelConfig,
    X_scatter: np.ndarray,               # shape (n, 2) in the SAME rows/order as y_train
    T_markers: np.ndarray,               # shape (n, m_count) asinh(marker/cofactor), SAME rows/order as y_train
    y_train: np.ndarray,
    marker_names: List[str],
    marker_thresholds: Dict[str, float], # thresholds in transformed-asinh space
) -> Dict[int, str]:
    """
    Label clusters using learned per-marker thresholds in transformed space.

    Rules (cluster-level):
      - Debris: low FSC/SSC medians (scatter-only).
      - Otherwise, for each marker:
          frac_above = fraction of cluster events with T_markers[:, mi] > thr[marker]
          med = median of T_markers[:, mi]
        marker is a candidate if:
          frac_above >= cluster_label_cfg.min_frac_above
          AND med >= thr + cluster_label_cfg.min_median_margin
      - Pick candidate with highest score:
          score = (med - thr) + cluster_label_cfg.frac_weight * frac_above
      - If no candidate: Unknown
      - Optional ambiguity handling via cluster_label_cfg.assign_margin.
    """
    marker_names = list(marker_names)
    m_count = len(marker_names)

    if X_scatter.ndim != 2 or X_scatter.shape[1] != 2:
        raise ValueError("X_scatter must be (n,2) [FSC,SSC] for debris detection.")
    if T_markers.ndim != 2 or T_markers.shape[1] != m_count:
        raise ValueError("T_markers must be (n,m_count) aligned to marker_names.")
    if T_markers.shape[0] != y_train.shape[0] or X_scatter.shape[0] != y_train.shape[0]:
        raise ValueError("X_scatter, T_markers, and y_train must have same n rows.")

    cluster_ids = np.unique(y_train)
    cluster_ids = cluster_ids[cluster_ids != -1]

    # debris cuts based on global scatter distribution
    fsc_cut = float(np.quantile(X_scatter[:, 0], float(debris_cfg.fsc_q)))
    ssc_cut = float(np.quantile(X_scatter[:, 1], float(debris_cfg.ssc_q)))

    min_frac = float(cluster_label_cfg.min_frac_above)
    min_med_margin = float(cluster_label_cfg.min_median_margin)
    frac_w = float(cluster_label_cfg.frac_weight)
    assign_margin = float(cluster_label_cfg.assign_margin)

    thr_vec = np.array([float(marker_thresholds.get(m, np.nan)) for m in marker_names], dtype=float)

    cluster_to_type: Dict[int, str] = {}

    for cid in cluster_ids:
        rows = (y_train == cid)
        if not np.any(rows):
            continue

        # debris?
        med_f = float(np.median(X_scatter[rows, 0]))
        med_s = float(np.median(X_scatter[rows, 1]))
        if (med_f < fsc_cut) and (med_s < ssc_cut):
            cluster_to_type[int(cid)] = "Debris"
            continue

        # label by thresholds in transformed marker space
        T = T_markers[rows, :]

        med = np.median(T, axis=0).astype(float)
        frac = np.mean(T > thr_vec.reshape(1, -1), axis=0).astype(float)

        ok_thr = np.isfinite(thr_vec)
        cand = ok_thr & (frac >= min_frac) & (med >= (thr_vec + min_med_margin))

        if not np.any(cand):
            cluster_to_type[int(cid)] = "Unknown"
            continue

        score = (med - thr_vec) + (frac_w * frac)
        score[~cand] = -1e18

        best_i = int(np.argmax(score))
        best_score = float(score[best_i])

        tmp = score.copy()
        tmp[best_i] = -1e18
        second_score = float(np.max(tmp))

        if (second_score > -1e17) and ((best_score - second_score) < assign_margin):
            cluster_to_type[int(cid)] = "Unknown"
        else:
            cluster_to_type[int(cid)] = str(marker_names[best_i])

    return cluster_to_type
