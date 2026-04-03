from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from .config import SingletMode, QCConfig
from .types import QCResult
from ._utils import freeze_mapping
from alloviewer.flow_cytometry.fcs_file import FCSFile
from alloviewer.flow_cytometry.panel import Panel


class QCGater:
    """
    QC stage = edge -> singlets -> broad debris removal.
    """

    def __init__(self, panel: Panel, config: QCConfig) -> None:
        self.panel = panel
        self.config = config

        self._cfg_token = freeze_mapping(asdict(self.config))
        self._cache: Dict[Tuple[int, Tuple], QCResult] = {}

    def _events(self, fcs: FCSFile) -> np.ndarray:
        return fcs.get_events(self.config.event_source)

    def _idx(self, fcs: FCSFile, channel_label: str) -> int:
        return int(fcs.get_channel_index(channel_label))

    def _col(self, events: np.ndarray, idx: int) -> np.ndarray:
        return events[:, idx].astype(np.float32, copy=False)

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        med = np.median(x)
        return float(np.median(np.abs(x - med)))

    @staticmethod
    def _subsample_idx(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
        if k <= 0 or k >= n:
            return np.arange(n)
        return rng.choice(n, size=k, replace=False)

    @staticmethod
    def _log1p_f32(x: np.ndarray) -> np.ndarray:
        # explicit log transform used by singlet/debris gates
        return np.log1p(np.asarray(x, dtype=np.float32))

    def gate_edge_events(
        self,
        fcs: FCSFile,
        events: np.ndarray,
        mask_in: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        m = (np.ones(events.shape[0], dtype=bool) if mask_in is None else np.asarray(mask_in, dtype=bool).copy())
        pnl = self.panel

        for ch in filter(None, [pnl.fsc_a, pnl.ssc_a, getattr(pnl, "fsc_h", None), getattr(pnl, "ssc_h", None)]):
            j = self._idx(fcs, ch)
            hi = float(fcs.channels.loc[ch, "pnr"] / fcs.channels.loc[ch, "png"])
            m &= (events[:, j] > 0.0) & (events[:, j] < (float(self.config.edge.hi_frac) * hi))
        return m

    def gate_singlets(
        self,
        fsc_a: np.ndarray,
        fsc_h: np.ndarray,
        *,
        mode: Optional[SingletMode] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns:
          mask_singlets_full_length, info
        """
        cfg = self.config
        mode_use: SingletMode = mode or cfg.singlet.mode

        eps = 1e-12
        valid = (fsc_a > eps) & (fsc_h > eps)

        if np.any(valid):
            q = float(cfg.singlet.min_fsc_quantile)
            cut = float(np.quantile(fsc_a[valid], q))
            valid &= (fsc_a >= cut)

        n_valid = int(np.sum(valid))
        if n_valid < int(cfg.singlet.min_events):
            return valid, {"mode_used": "fallback_valid", "reason": "too_few_valid", "n_valid": n_valid}

        if mode_use == "hybrid":
            try:
                m, info = self.gate_singlets_hybrid(fsc_a, fsc_h, valid=valid)
                return m, info
            except Exception as e:
                m, info2 = self.gate_singlets_mad(fsc_a, fsc_h, valid=valid)
                info2["fallback_from"] = "hybrid"
                info2["fallback_error"] = str(e)
                return m, info2

        if mode_use == "gmm":
            try:
                m, info = self.gate_singlets_gmm(fsc_a, fsc_h, valid=valid)
                return m, info
            except Exception as e:
                m, info2 = self.gate_singlets_mad(fsc_a, fsc_h, valid=valid)
                info2["fallback_from"] = "gmm"
                info2["fallback_error"] = str(e)
                return m, info2

        return self.gate_singlets_mad(fsc_a, fsc_h, valid=valid)

    # -------------------------
    # MAD singlets
    # -------------------------
    def gate_singlets_mad(
        self,
        fsc_a: np.ndarray,
        fsc_h: np.ndarray,
        *,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Stable for FSC-A/FSC-H:
          r = log1p(FSC-H) - log1p(FSC-A)
        """
        cfg = self.config
        x = self._log1p_f32(fsc_a[valid])
        y = self._log1p_f32(fsc_h[valid])
        r = (y - x).astype(np.float32, copy=False)

        if r.size < 100:
            return valid, {"mode_used": "mad", "reason": "too_few_events"}

        r_med = float(np.median(r))
        r_mad = float(self._mad(r))
        if r_mad <= 0.0 or not np.isfinite(r_mad):
            return valid, {"mode_used": "mad", "reason": "mad_zero_or_bad"}

        k = float(cfg.singlet.k_mad)
        lo = r_med - k * r_mad
        hi = r_med + k * r_mad

        out = np.zeros_like(valid, dtype=bool)
        idx = np.flatnonzero(valid)
        out[idx] = (r >= lo) & (r <= hi)

        keep_frac = float(np.mean(out[valid])) if np.any(valid) else 0.0
        return out, {
            "mode_used": "mad",
            "r_med": r_med,
            "r_mad": r_mad,
            "k": k,
            "keep_frac": keep_frac,
        }

    # -------------------------
    # GMM singlets
    # -------------------------
    def gate_singlets_gmm(
        self,
        fsc_a: np.ndarray,
        fsc_h: np.ndarray,
        *,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        2D GMM in log space.
        Robust component selection for your plots:
          - pick the component with higher median log(FSC-A)
          - use diagonal alignment only as a tie-break
        """
        cfg = self.config
        rng = np.random.default_rng(int(getattr(cfg, "random_state", 0)))

        x = self._log1p_f32(fsc_a[valid])
        y = self._log1p_f32(fsc_h[valid])
        X = np.column_stack([x, y]).astype(np.float32, copy=False)

        n = int(X.shape[0])
        if n < int(cfg.singlet.gmm_min_events):
            return valid, {"mode_used": "gmm_fallback_valid", "reason": "too_few_events", "n": n}

        sub_k = int(cfg.singlet.gmm_subsample)
        sub_idx = self._subsample_idx(n, sub_k, rng)
        X_sub = X[sub_idx]

        K = int(cfg.singlet.gmm_components)
        cov_type = str(cfg.singlet.gmm_covariance_type)
        reg = float(cfg.singlet.gmm_reg_covar)

        gmm = GaussianMixture(
            n_components=K,
            covariance_type=cov_type,
            random_state=int(getattr(cfg, "random_state", 0)),
            reg_covar=reg,
        )
        gmm.fit(X_sub)

        resp = gmm.predict_proba(X)  # (n, K)
        hard = np.argmax(resp, axis=1)

        # per-component metrics
        med_a = np.full(K, -np.inf, dtype=np.float32)
        frac = np.zeros(K, dtype=np.float32)
        align = np.full(K, -np.inf, dtype=np.float32)

        diag = np.array([1.0, 1.0], dtype=np.float32)
        diag /= np.linalg.norm(diag)

        for k in range(K):
            sel = (hard == k)
            if not np.any(sel):
                continue
            frac[k] = float(np.mean(sel))
            med_a[k] = float(np.median(X[sel, 0]))

            # diagonal alignment from covariance (component-specific covariance is best,
            # but we can estimate from assigned points; stable enough)
            C = np.cov(X[sel].T)
            w, v = np.linalg.eigh(C)
            v_max = v[:, int(np.argmax(w))]
            v_max = v_max / (np.linalg.norm(v_max) + 1e-12)
            align[k] = float(abs(np.dot(v_max, diag)))

        # choose component:
        # mainly highest med_a; if very close, use align.
        k_star = int(np.argmax(med_a))
        if K > 1:
            best = float(med_a[k_star])
            # if another component within delta, pick higher alignment
            delta = float(cfg.singlet.gmm_med_a_tie_delta)
            close = np.flatnonzero((med_a >= (best - delta)) & np.isfinite(med_a))
            if close.size > 1:
                k_star = int(close[np.argmax(align[close])])

        resp_thr = float(cfg.singlet.gmm_resp_threshold)
        keep_local = resp[:, k_star] >= resp_thr

        keep_frac_local = float(np.mean(keep_local)) if keep_local.size else 0.0
        min_keep = float(cfg.singlet.min_keep_fraction)
        max_keep = float(cfg.singlet.max_keep_fraction)

        # if responsibility threshold is too strict/lenient, fall back to hard assignment
        if (keep_frac_local < min_keep) or (keep_frac_local > max_keep):
            keep_local = (hard == k_star)
            keep_frac_local = float(np.mean(keep_local)) if keep_local.size else 0.0

        # if still bad, fall back to valid
        if (keep_frac_local < min_keep) or (keep_frac_local > max_keep):
            return valid, {
                "mode_used": "gmm_fallback_valid",
                "reason": "keep_frac_out_of_bounds",
                "keep_frac": keep_frac_local,
                "min_keep": min_keep,
                "max_keep": max_keep,
                "K": K,
                "cov_type": cov_type,
                "reg_covar": reg,
                "chosen_component": k_star,
                "med_a": med_a.tolist(),
                "align": align.tolist(),
                "frac": frac.tolist(),
                "resp_thr": resp_thr,
            }

        out = np.zeros_like(valid, dtype=bool)
        idx = np.flatnonzero(valid)
        out[idx] = keep_local

        return out, {
            "mode_used": "gmm",
            "K": K,
            "cov_type": cov_type,
            "reg_covar": reg,
            "chosen_component": int(k_star),
            "med_a": med_a.tolist(),
            "align": align.tolist(),
            "frac": frac.tolist(),
            "resp_thr": resp_thr,
            "keep_frac": keep_frac_local,
            "thresholded": True,
        }

    # -------------------------
    # Hybrid: MAD screen then GMM
    # -------------------------
    def gate_singlets_hybrid(
        self,
        fsc_a: np.ndarray,
        fsc_h: np.ndarray,
        *,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Hybrid (reversed):
          Stage 1: GMM in log space to remove the "bad cloud" (coarse).
          Stage 2: tight MAD band (log-diff) on the GMM-kept events (fine).

        Fallbacks:
          - If GMM fails -> MAD on `valid`.
          - If GMM result looks unreasonable -> MAD on `valid`.
          - If MAD-on-GMM is unreasonable -> return GMM-only mask.
        """
        cfg = self.config

        # -------------------------
        # Stage 1: GMM (coarse)
        # -------------------------
        try:
            mask_gmm, info_gmm = self.gate_singlets_gmm(fsc_a, fsc_h, valid=valid)
        except Exception as e:
            # hard fallback: MAD on valid
            m_mad, info_mad = self.gate_singlets_mad(fsc_a, fsc_h, valid=valid)
            return m_mad, {
                "mode_used": "hybrid",
                "stage": "fallback_mad_only",
                "reason": "gmm_failed",
                "error": str(e),
                "mad_info": info_mad,
            }

        # sanity check for stage1 keep fraction
        keep1 = float(np.mean(mask_gmm[valid])) if np.any(valid) else 0.0
        min_keep = float(cfg.singlet.min_keep_fraction)
        max_keep = float(cfg.singlet.max_keep_fraction)

        if (keep1 < min_keep) or (keep1 > max_keep):
            # fallback: MAD on valid (GMM did something weird)
            m_mad, info_mad = self.gate_singlets_mad(fsc_a, fsc_h, valid=valid)
            return m_mad, {
                "mode_used": "hybrid",
                "stage": "fallback_mad_only",
                "reason": "gmm_keep_frac_out_of_bounds",
                "keep_frac_gmm": keep1,
                "min_keep": min_keep,
                "max_keep": max_keep,
                "gmm_info": info_gmm,
                "mad_info": info_mad,
            }

        # If too few points after GMM, don't attempt MAD tightening.
        min_after = int(cfg.singlet.min_events)
        n_after = int(np.sum(mask_gmm))
        if n_after < min_after:
            return mask_gmm, {
                "mode_used": "hybrid",
                "stage": "fallback_gmm_only",
                "reason": "too_few_after_gmm",
                "n_after_gmm": n_after,
                "min_after": min_after,
                "keep_frac_gmm": keep1,
                "gmm_info": info_gmm,
            }

        # -------------------------
        # Stage 2: tight MAD (fine)
        # -------------------------
        # Use a separate knob if present; otherwise use singlet_k_mad (tight).
        k_tight = float(cfg.singlet.hybrid_k_mad)

        x = self._log1p_f32(fsc_a[mask_gmm])
        y = self._log1p_f32(fsc_h[mask_gmm])
        r = (y - x).astype(np.float32, copy=False)

        if r.size < 100:
            return mask_gmm, {
                "mode_used": "hybrid",
                "stage": "fallback_gmm_only",
                "reason": "too_few_for_mad_stage2",
                "n_stage2": int(r.size),
                "gmm_info": info_gmm,
            }

        r_med = float(np.median(r))
        r_mad = float(self._mad(r))
        if r_mad <= 0.0 or not np.isfinite(r_mad):
            return mask_gmm, {
                "mode_used": "hybrid",
                "stage": "fallback_gmm_only",
                "reason": "mad_zero_or_bad_stage2",
                "r_mad": r_mad,
                "gmm_info": info_gmm,
            }

        lo = r_med - k_tight * r_mad
        hi = r_med + k_tight * r_mad

        # Map stage2 result back to full length mask
        out = np.zeros_like(valid, dtype=bool)
        idx_gmm = np.flatnonzero(mask_gmm)
        keep2_local = (r >= lo) & (r <= hi)
        out[idx_gmm] = keep2_local

        # Sanity check: stage2 should not destroy everything
        keep2_rel = float(np.mean(out[mask_gmm])) if np.any(mask_gmm) else 0.0
        # Accept stage2 if it keeps a reasonable fraction of the GMM-kept points
        min_keep2_rel = float(cfg.singlet.hybrid_min_keep_rel)
        max_keep2_rel = float(cfg.singlet.hybrid_max_keep_rel)

        if (keep2_rel < min_keep2_rel) or (keep2_rel > max_keep2_rel):
            return mask_gmm, {
                "mode_used": "hybrid",
                "stage": "fallback_gmm_only",
                "reason": "mad_stage2_keep_rel_out_of_bounds",
                "keep_rel_stage2": keep2_rel,
                "min_keep_rel": min_keep2_rel,
                "max_keep_rel": max_keep2_rel,
                "k_tight": k_tight,
                "r_med": r_med,
                "r_mad": r_mad,
                "gmm_info": info_gmm,
            }

        # Final overall keep fraction (relative to `valid`)
        keep_final = float(np.mean(out[valid])) if np.any(valid) else 0.0

        return out, {
            "mode_used": "hybrid",
            "stage1": "gmm",
            "stage2": "mad_tight",
            "keep_frac_gmm": keep1,
            "keep_rel_stage2": keep2_rel,
            "keep_frac_final": keep_final,
            "k_tight": k_tight,
            "r_med": r_med,
            "r_mad": r_mad,
            "gmm_info": info_gmm,
        }


    def compute_qc(
        self,
        fcs: FCSFile, *,
        singlet_mode: Optional[SingletMode] = None
    ) -> QCResult:
        key = (id(fcs), self._cfg_token, singlet_mode)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        notes: List[str] = []
        events = self._events(fcs)
        if events is None:
            raise ValueError(f"get_events('{self.config.event_source}') returned None.")

        mask_edge = self.gate_edge_events(fcs, events)

        mask_sing: Optional[np.ndarray] = None
        if self.panel.fsc_a and self.panel.fsc_h:
            try:
                ia = self._idx(fcs, self.panel.fsc_a)
                ih = self._idx(fcs, self.panel.fsc_h)

                mode_use: SingletMode = singlet_mode or self.config.singlet.mode

                mask_sing_raw, info = self.gate_singlets(
                    self._col(events, ia),
                    self._col(events, ih),
                    mode=mode_use,
                )
                mask_sing = np.asarray(mask_sing_raw, dtype=bool) & mask_edge

                # optional: keep your fallback preference visible
                notes.append(f"Singlets: {info.get('mode_used')}")
                if info.get("fallback_from"):
                    notes.append(f"Singlets fallback from {info.get('fallback_from')}: {info.get('fallback_error')}")

            except Exception as e:
                notes.append(f"Singlets skipped: {e}")
                mask_sing = None

        mask_qc = (mask_sing.copy() if mask_sing is not None else mask_edge.copy())

        out = QCResult(
            events=events,
            mask_edge=mask_edge,
            mask_sing=mask_sing,
            mask_qc=mask_qc,
            notes=notes,
        )
        self._cache[key] = out
        return out

