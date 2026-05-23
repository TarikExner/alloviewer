from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.mixture import GaussianMixture

from ._utils import fit_1d_gmm2, log1p_nonneg, subsample
from .config import LymphocyteConfig, TransformConfig
from .types import LymphResult
from ..fcs_file import FCSFile
from ..panel import Panel


def gate_lymphocytes(
    *,
    lymph_cfg: LymphocyteConfig,
    transform_cfg: TransformConfig,
    random_state: int,
    panel: Panel,
    fcs: FCSFile,
    events: np.ndarray,
    mask_qc: np.ndarray,
    marker_thresholds: Dict[str, float],
    marker_cofactors: Dict[str, float],
) -> LymphResult:
    """Gate lymphocytes using FSC/SSC with marker-informed back-gating.

    Parameters
    ----------
    lymph_cfg : LymphocyteConfig
        Lymphocyte-gating configuration.
    transform_cfg : TransformConfig
        Transformation configuration used for marker channels.
    random_state : int
        Random seed used for subsampling and mixture-model fitting.
    panel : Panel
        Panel definition with scatter and marker channels.
    fcs : FCSFile
        FCS file object used for channel index lookup.
    events : numpy.ndarray
        Event matrix with shape ``(n_events, n_channels)``.
    mask_qc : numpy.ndarray
        Boolean QC mask defining events eligible for lymphocyte gating.
    marker_thresholds : dict
        Marker thresholds in transformed space, keyed by marker name.
    marker_cofactors : dict
        Marker transformation cofactors, keyed by marker name.

    Returns
    -------
    LymphResult
        Lymphocyte mask and diagnostic information.

    Notes
    -----
    Lymphocytes are selected in FSC/SSC space. Marker-positive events are used
    as a guide to choose FSC/SSC mixture components, but marker positivity alone
    does not define the lymphocyte gate. If marker guidance is insufficient or
    the fitted model keeps too few events, the function falls back to FSC/SSC
    quantile rails.
    """
    lc = lymph_cfg
    tc = transform_cfg

    mask_qc = np.asarray(mask_qc, dtype=bool)
    idx_qc = np.flatnonzero(mask_qc)

    if idx_qc.size == 0:
        return LymphResult(
            mask_lymph=np.zeros(events.shape[0], dtype=bool),
            info={"skipped": True, "reason": "no_qc"},
        )

    if not (panel.fsc_a and panel.ssc_a):
        return LymphResult(
            mask_lymph=mask_qc.copy(),
            info={"skipped": True, "reason": "no_scatter_channels"},
        )

    jf = int(fcs.get_channel_index(panel.fsc_a))
    js = int(fcs.get_channel_index(panel.ssc_a))

    fsc = events[:, jf].astype(np.float32, copy=False)
    ssc = events[:, js].astype(np.float32, copy=False)

    xf = log1p_nonneg(fsc[idx_qc])
    xs = log1p_nonneg(ssc[idx_qc])
    X = np.column_stack([xf, xs])

    rng = np.random.default_rng(int(random_state))

    guide = np.zeros(idx_qc.size, dtype=bool)
    guide_counts: Dict[str, int] = {}

    for mname, ch in (panel.markers or {}).items():
        thr = float(marker_thresholds.get(mname, np.nan))

        if not np.isfinite(thr):
            continue

        j = int(fcs.get_channel_index(ch))
        raw = events[:, j].astype(np.float32, copy=False)[idx_qc]
        cof = float(marker_cofactors.get(mname, tc.default_cofactor))
        vt = np.arcsinh(raw / max(cof, 1e-12)).astype(np.float32, copy=False)

        pos = np.isfinite(vt) & (vt > thr)
        guide |= pos
        guide_counts[mname] = int(pos.sum())

    info: Dict[str, Any] = {
        "skipped": False,
        "guide_total": int(guide.sum()),
        "guide_counts": dict(guide_counts),
    }

    if int(guide.sum()) < int(lc.min_guide_events):
        f_lo = float(np.quantile(fsc[idx_qc], float(lc.fsc_low_q)))
        s_hi = float(np.quantile(ssc[idx_qc], float(lc.ssc_high_q)))
        m = mask_qc.copy()
        m[idx_qc] &= (fsc[idx_qc] >= f_lo) & (ssc[idx_qc] <= s_hi)
        info["fallback"] = "qc_quantile_rails"
        info["fsc_low"] = f_lo
        info["ssc_high"] = s_hi

        return LymphResult(mask_lymph=m, info=info)

    guide_keep = guide.copy()

    if bool(lc.exclude_low_fsc_in_guide) and int(guide.sum()) >= int(
        lc.guide_fsc_gmm_min_events
    ):
        g1 = fit_1d_gmm2(
            log1p_nonneg(fsc[idx_qc][guide]),
            rs=int(random_state),
        )

        if g1 is not None:
            gmm1, order = g1
            resp1 = gmm1.predict_proba(
                log1p_nonneg(fsc[idx_qc][guide]).reshape(-1, 1)
            )
            high_k = int(order[-1])
            keep_high = resp1[:, high_k] >= 0.5

            gi = np.flatnonzero(guide)
            guide_keep[:] = False
            guide_keep[gi] = keep_high
            info["guide_low_fsc_removed"] = True
            info["guide_keep"] = int(guide_keep.sum())
        else:
            fthr = float(
                np.quantile(
                    fsc[idx_qc][guide],
                    float(lc.guide_fsc_fallback_q),
                )
            )
            guide_keep = guide & (fsc[idx_qc] >= fthr)
            info["guide_low_fsc_removed"] = True
            info["guide_keep"] = int(guide_keep.sum())

    if int(guide_keep.sum()) < int(lc.min_guide_events):
        guide_keep = guide
        info["guide_low_fsc_removed"] = False

    sub = subsample(X, int(lc.subsample), rng)
    X_sub = X[sub]

    K = int(max(2, lc.gmm_components))
    gmm2 = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=int(random_state),
    )
    gmm2.fit(X_sub)

    resp = gmm2.predict_proba(X)

    guide_idx = np.flatnonzero(guide_keep)
    guide_resp_sum = resp[guide_idx].sum(axis=0)

    order = np.argsort(-guide_resp_sum)
    picked = []
    covered = 0.0
    total = float(np.sum(guide_resp_sum)) + 1e-12

    target = float(lc.guide_coverage)

    for k in order:
        picked.append(int(k))
        covered += float(guide_resp_sum[k]) / total

        if covered >= target:
            break

    picked = np.asarray(picked, dtype=int)
    info["picked_components"] = picked.tolist()
    info["guide_coverage"] = float(covered)

    p = (
        resp[:, picked].sum(axis=1)
        if picked.size
        else np.zeros(resp.shape[0], dtype=float)
    )
    keep_local = p >= float(lc.resp_threshold)

    keep_frac = float(np.mean(keep_local))
    info["keep_frac"] = keep_frac

    if keep_frac < float(lc.min_keep_fraction):
        f_lo = float(np.quantile(fsc[idx_qc], float(lc.fsc_low_q)))
        s_hi = float(np.quantile(ssc[idx_qc], float(lc.ssc_high_q)))
        keep_local = (fsc[idx_qc] >= f_lo) & (ssc[idx_qc] <= s_hi)
        info["fallback"] = "qc_quantile_rails_after_gmm"
        info["fsc_low"] = f_lo
        info["ssc_high"] = s_hi

    f_lo = float(np.quantile(fsc[idx_qc], float(lc.fsc_low_q)))
    s_hi = float(np.quantile(ssc[idx_qc], float(lc.ssc_high_q)))
    keep_local &= (fsc[idx_qc] >= f_lo) & (ssc[idx_qc] <= s_hi)
    info["fsc_low_final"] = f_lo
    info["ssc_high_final"] = s_hi

    mask_lymph = np.zeros(events.shape[0], dtype=bool)
    mask_lymph[idx_qc] = keep_local

    return LymphResult(mask_lymph=mask_lymph, info=info)
