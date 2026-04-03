# marker_calibration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
from sklearn.mixture import GaussianMixture

from .config import TransformConfig, MarkerCofactorConfig, MarkerThresholdConfig
from ..panel import Panel
from ..fcs_file import FCSFile


@dataclass(frozen=True)
class MarkerCalibrationResult:
    marker_cofactors: Dict[str, float]
    marker_thresholds: Dict[str, float]
    marker_info: Dict[str, Dict[str, Any]]


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    if x.size == 0:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float("nan")

    t = q * cw[-1]
    j = int(np.searchsorted(cw, t, side="left"))
    j = min(max(j, 0), x.size - 1)
    return float(x[j])


def _gmm_intersection_1d(means: np.ndarray, vars_: np.ndarray, weights: np.ndarray) -> float:
    m0, m1 = float(means[0]), float(means[1])
    v0, v1 = float(vars_[0]), float(vars_[1])
    w0, w1 = float(weights[0]), float(weights[1])

    v0 = max(v0, 1e-12)
    v1 = max(v1, 1e-12)
    w0 = max(w0, 1e-12)
    w1 = max(w1, 1e-12)

    A = (1.0 / (2.0 * v0)) - (1.0 / (2.0 * v1))
    B = (-m0 / v0) + (m1 / v1)
    C = (m0 * m0) / (2.0 * v0) - (m1 * m1) / (2.0 * v1) - math.log((w0 / math.sqrt(v0)) / (w1 / math.sqrt(v1)))

    if abs(A) < 1e-10:
        if abs(B) < 1e-12:
            return 0.5 * (m0 + m1)
        return float(-C / B)

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return 0.5 * (m0 + m1)

    rdisc = math.sqrt(disc)
    x1 = (-B + rdisc) / (2.0 * A)
    x2 = (-B - rdisc) / (2.0 * A)

    lo = min(m0, m1)
    hi = max(m0, m1)
    mid = 0.5 * (m0 + m1)

    cand = []
    if lo <= x1 <= hi:
        cand.append(x1)
    if lo <= x2 <= hi:
        cand.append(x2)

    if cand:
        if len(cand) == 1:
            return float(cand[0])
        return float(cand[np.argmin([abs(c - mid) for c in cand])])

    return float(x1 if abs(x1 - mid) < abs(x2 - mid) else x2)


def _pool_marker_raw(
    *,
    cofactor_cfg: MarkerCofactorConfig,
    random_state: int,
    panel: Panel,
    file_records: List[Dict[str, Any]],
    marker_name: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))

    max_total = int(cofactor_cfg.max_events_per_marker)
    max_per_file = int(cofactor_cfg.max_events_per_file)

    parts: List[np.ndarray] = []
    total = 0

    ch = (panel.markers or {}).get(marker_name)
    if ch is None:
        return np.zeros(0, dtype=np.float32)

    for rec in file_records:
        if total >= max_total:
            break

        fcs: FCSFile = rec["fcs"]
        ev = rec["events"]
        mask_qc = np.asarray(rec["mask_qc"], dtype=bool)
        if not np.any(mask_qc):
            continue

        j = int(fcs.get_channel_index(ch))
        v = ev[:, j].astype(np.float32, copy=False)[mask_qc]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue

        k = min(v.size, max_per_file, max_total - total)
        if k <= 0:
            break

        if v.size > k:
            v = v[rng.choice(v.size, size=k, replace=False)]
        else:
            v = v[:k]

        parts.append(v)
        total += int(v.size)

    if not parts:
        return np.zeros(0, dtype=np.float32)

    x = np.concatenate(parts).astype(np.float32, copy=False)
    x = x[np.isfinite(x)]
    return x


def _calibrate_marker(
    *,
    transform_cfg: TransformConfig,
    cofactor_cfg: MarkerCofactorConfig,
    threshold_cfg: MarkerThresholdConfig,
    random_state: int,
    x_raw: np.ndarray,
    cofactor_init: float,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns:
      c_final: raw units divisor
      t_cut: threshold in asinh(marker / c) space
      info: details and fallback info
    """
    info: Dict[str, Any] = {"fallback": None, "iters": 0}

    x = np.asarray(x_raw, dtype=np.float32)
    x = x[np.isfinite(x)]

    if x.size < int(threshold_cfg.gmm_min_events):
        info["fallback"] = "too_few_events"
        c = float(cofactor_init)
        t = np.arcsinh(x / max(c, 1e-12)) if x.size else np.zeros(0, dtype=float)
        t_cut = float(np.quantile(t, float(threshold_cfg.fallback_quantile))) if t.size else float("nan")
        info["c_final"] = c
        info["t_cut"] = t_cut
        return c, t_cut, info

    c = float(cofactor_init)
    target_t = float(cofactor_cfg.target_t)
    denom = float(np.sinh(target_t)) if target_t > 0 else 1.0

    for it in range(int(cofactor_cfg.iter_max)):
        info["iters"] = it + 1

        t = np.arcsinh(x / max(c, 1e-12)).astype(np.float32, copy=False)

        clip_t = float(cofactor_cfg.clip_t)
        if clip_t > 0:
            t = np.clip(t, -clip_t, clip_t)

        gmm = GaussianMixture(n_components=2, random_state=int(random_state))
        gmm.fit(t.reshape(-1, 1))

        means = gmm.means_.ravel().astype(float)
        bg = int(np.argmin(means))
        fg = 1 - bg

        resp = gmm.predict_proba(t.reshape(-1, 1)).astype(np.float32, copy=False)
        w_bg = resp[:, bg].astype(float)

        bg_weight = float(np.mean(w_bg))
        mean_sep = float(abs(means[fg] - means[bg]))

        if (bg_weight < float(cofactor_cfg.min_bg_weight)) or (mean_sep < float(cofactor_cfg.min_mean_sep_t)):
            info["fallback"] = "gmm_unstable"
            xm = x[x <= np.median(x)]
            x_bg = float(np.quantile(xm, float(cofactor_cfg.bg_q))) if xm.size else float(np.quantile(x, 0.20))
            c_new = float(x_bg / max(denom, 1e-12))
        else:
            t_bg = _weighted_quantile(t.astype(float), w_bg, float(cofactor_cfg.bg_q))
            if not np.isfinite(t_bg):
                info["fallback"] = "weighted_quantile_failed"
                xm = x[x <= np.median(x)]
                x_bg = float(np.quantile(xm, float(cofactor_cfg.bg_q))) if xm.size else float(np.quantile(x, 0.20))
                c_new = float(x_bg / max(denom, 1e-12))
            else:
                x_bg = float(c * np.sinh(float(t_bg)))
                c_new = float(x_bg / max(denom, 1e-12))

        c_new = float(np.clip(c_new, float(cofactor_cfg.min), float(cofactor_cfg.max)))
        if not np.isfinite(c_new) or c_new <= 0:
            info["fallback"] = info["fallback"] or "invalid_c"
            c_new = float(cofactor_init)

        rel = abs(c_new - c) / max(c, 1e-12)
        c = c_new
        if rel < float(cofactor_cfg.tol_rel):
            break

    t_final = np.arcsinh(x / max(c, 1e-12)).astype(np.float32, copy=False)

    clip_t = float(cofactor_cfg.clip_t)
    if clip_t > 0:
        t_final = np.clip(t_final, -clip_t, clip_t)

    gmm2 = GaussianMixture(n_components=2, random_state=int(random_state))
    gmm2.fit(t_final.reshape(-1, 1))

    means2 = gmm2.means_.ravel().astype(float)
    vars2 = gmm2.covariances_.ravel().astype(float)
    weights2 = gmm2.weights_.ravel().astype(float)

    order = np.argsort(means2)
    means2 = means2[order]
    vars2 = vars2[order]
    weights2 = weights2[order]

    t_cut = _gmm_intersection_1d(means2, vars2, weights2)
    if not np.isfinite(t_cut):
        t_cut = float(0.5 * (means2[0] + means2[1]))

    info["c_final"] = float(c)
    info["t_cut"] = float(t_cut)
    info["means_t"] = [float(m) for m in means2]
    info["weights"] = [float(w) for w in weights2]
    return float(c), float(t_cut), info


def calibrate_markers(
    *,
    transform_cfg: TransformConfig,
    cofactor_cfg: MarkerCofactorConfig,
    threshold_cfg: MarkerThresholdConfig,
    random_state: int,
    panel: Panel,
    file_records: List[Dict[str, Any]],
    marker_names: Optional[List[str]] = None,
) -> MarkerCalibrationResult:
    """
    Public API:
      - pools raw values per marker from QC events
      - estimates per-marker cofactor and threshold in asinh space
    """
    marker_names = list(marker_names) if marker_names is not None else list((panel.markers or {}).keys())
    cof0 = float(transform_cfg.default_cofactor)

    marker_cofactors: Dict[str, float] = {}
    marker_thresholds: Dict[str, float] = {}
    marker_info: Dict[str, Dict[str, Any]] = {}

    for mname in marker_names:
        x = _pool_marker_raw(
            cofactor_cfg=cofactor_cfg,
            random_state=random_state,
            panel=panel,
            file_records=file_records,
            marker_name=mname,
        )

        if x.size == 0:
            marker_cofactors[mname] = float(cof0)
            marker_thresholds[mname] = float("nan")
            marker_info[mname] = {
                "fallback": "no_data",
                "c_final": float(cof0),
                "t_cut": float("nan"),
            }
            continue

        c, t_cut, info = _calibrate_marker(
            transform_cfg=transform_cfg,
            cofactor_cfg=cofactor_cfg,
            threshold_cfg=threshold_cfg,
            random_state=random_state,
            x_raw=x,
            cofactor_init=cof0,
        )
        marker_cofactors[mname] = float(c)
        marker_thresholds[mname] = float(t_cut)
        marker_info[mname] = dict(info)

    return MarkerCalibrationResult(
        marker_cofactors=marker_cofactors,
        marker_thresholds=marker_thresholds,
        marker_info=marker_info,
    )
