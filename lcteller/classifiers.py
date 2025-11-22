import math
from typing import List, Dict, Any, Optional

from .config import DEFAULT_CALIB_RG_GAUSS


def _phi_cdf(z: float) -> float:
    # standard normal CDF
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _normalize_gauss_calib(calib: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill missing keys from defaults and clamp stds."""
    base = DEFAULT_CALIB_RG_GAUSS
    c = {**base, **(calib or {})}
    c["sd_pc"] = max(float(c["sd_pc"]), 1e-6)
    c["sd_nc"] = max(float(c["sd_nc"]), 1e-6)
    c.setdefault("method", "pc_nc_gaussian_rg")
    return c


class ROIClassifier:
    """
    Binary classifier using an R/G threshold.
    - Expects calib with 'rg_thresh' (from PCNCMedianCalibrator).
    - If missing, falls back to midpoint of default means.
    """
    def __init__(self, calib: Optional[Dict[str, Any]] = None):
        if calib is None or "rg_thresh" not in calib:
            # fallback: midpoint between default R/G means (PC high, NC low)
            thr = 0.5 * (float(DEFAULT_CALIB_RG_GAUSS["mu_pc"]) +
                         float(DEFAULT_CALIB_RG_GAUSS["mu_nc"]))
            self.thr = float(thr)
            self.method = "pc_nc_median_rg_fallback"
        else:
            self.thr = float(calib["rg_thresh"])
            self.method = calib.get("method", "pc_nc_median_rg")

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rois:
            score = r["mean_r"] / max(1e-6, r["mean_g"])  # R/G
            rr = dict(r)
            rr["score"] = float(score)
            rr["label"] = "pos" if score >= self.thr else "neg"
            rr["method"] = self.method
            out.append(rr)
        return out


class ROIClassifierMedianRG:
    """
    Binary classifier using a single R/G threshold.
    Expects calib dict with 'rg_thresh'. If missing, falls back to midpoint of
    DEFAULT_CALIB_RG_GAUSS means.
    """
    def __init__(self, calib: Optional[Dict[str, Any]] = None):
        if calib is None or "rg_thresh" not in calib:
            thr = 0.5 * (float(DEFAULT_CALIB_RG_GAUSS["mu_pc"]) + float(DEFAULT_CALIB_RG_GAUSS["mu_nc"]))
            self.thr = float(thr)
            self.method = "pc_nc_median_rg_fallback"
        else:
            self.thr = float(calib["rg_thresh"])
            self.method = calib.get("method", "pc_nc_median_rg")

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rois:
            score = r["mean_r"] / max(1e-6, r["mean_g"])   # R/G
            rr = dict(r)
            rr["score"] = float(score)
            rr["label"] = "pos" if score >= self.thr else "neg"
            rr["method"] = self.method
            out.append(rr)
        return out


class ROIClassifierNCUpper:
    """
    One-sided upper-tail test vs NC on R/G.
    z_nc = (rg - mu_nc)/sd_nc; label 'pos' if z_nc >= k, else 'uncertain'.
    """
    def __init__(self, calib: Optional[Dict[str, Any]] = None, k: float = 3.0):
        c = _normalize_gauss_calib(calib)
        self.mu_nc = float(c["mu_nc"])
        self.sd_nc = float(c["sd_nc"])
        self.k = float(k)
        self.method = c.get("method", "pc_nc_gaussian_rg") + f"_NCUpper_k{self.k:g}"

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rois:
            rg = r["mean_r"] / max(1e-6, r["mean_g"])
            z_nc = (rg - self.mu_nc) / self.sd_nc
            p_nc_upper = 1.0 - _phi_cdf(z_nc)
            rr = dict(r)
            rr["score_rg"] = float(rg)
            rr["z_nc"] = float(z_nc)
            rr["p_nc_upper"] = float(p_nc_upper)
            rr["label"] = "pos" if z_nc >= self.k else "uncertain"
            rr["method"] = self.method
            out.append(rr)
        return out


class ROIClassifierPCLower:
    """
    One-sided lower-tail test vs PC on R/G.
    z_pc = (rg - mu_pc)/sd_pc; label 'neg' if z_pc <= -k, else 'uncertain'.
    """
    def __init__(self, calib: Optional[Dict[str, Any]] = None, k: float = 3.0):
        c = _normalize_gauss_calib(calib)
        self.mu_pc = float(c["mu_pc"])
        self.sd_pc = float(c["sd_pc"])
        self.k = float(k)
        self.method = c.get("method", "pc_nc_gaussian_rg") + f"_PCLower_k{self.k:g}"

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rois:
            rg = r["mean_r"] / max(1e-6, r["mean_g"])
            z_pc = (rg - self.mu_pc) / self.sd_pc
            # lower tail p = CDF(z_pc)
            p_pc_lower = _phi_cdf(z_pc)
            rr = dict(r)
            rr["score_rg"] = float(rg)
            rr["z_pc"] = float(z_pc)
            rr["p_pc_lower"] = float(p_pc_lower)
            rr["label"] = "neg" if z_pc <= -self.k else "uncertain"
            rr["method"] = self.method
            out.append(rr)
        return out


class ROIClassifierGaussian3Way:
    """
    Combine both tails on R/G:
      - 'pos' if z_nc >= k
      - 'neg' if z_pc <= -k
      - else 'uncertain'
    """
    def __init__(self, calib: Optional[Dict[str, Any]] = None, k: float = 2.0):
        c = _normalize_gauss_calib(calib)
        self.mu_nc = float(c["mu_nc"])
        self.sd_nc = float(c["sd_nc"])
        self.mu_pc = float(c["mu_pc"])
        self.sd_pc = float(c["sd_pc"])
        self.k = float(k)
        self.method = c.get("method", "pc_nc_gaussian_rg") + f"_3way_k{self.k:g}"

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from math import erf, sqrt
        out: List[Dict[str, Any]] = []
        for r in rois:
            rg = r["mean_r"] / max(1e-6, r["mean_g"])
            z_nc = (rg - self.mu_nc) / self.sd_nc
            z_pc = (rg - self.mu_pc) / self.sd_pc
            p_nc_upper = 1.0 - (0.5 * (1.0 + erf(z_nc / sqrt(2.0))))
            p_pc_lower = 0.5 * (1.0 + erf(z_pc / sqrt(2.0)))

            if z_nc >= self.k:
                label = "pos"
            elif z_pc <= -self.k:
                label = "neg"
            else:
                label = "uncertain"

            rr = dict(r)
            rr["score_rg"] = float(rg)
            rr["z_nc"] = float(z_nc)
            rr["z_pc"] = float(z_pc)
            rr["p_nc_upper"] = float(p_nc_upper)
            rr["p_pc_lower"] = float(p_pc_lower)
            rr["label"] = label
            rr["method"] = self.method
            out.append(rr)
        return out


