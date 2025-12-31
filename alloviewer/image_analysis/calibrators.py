import numpy as np
from typing import List, Dict, Any

def _rg_array(wells: List[List[Dict[str, Any]]]) -> np.ndarray:
    xs = []
    for rois in wells:
        for r in rois:
            xs.append(r["mean_r"] / max(1e-6, r["mean_g"]))
    return np.array(xs, dtype=np.float64) if xs else np.array([1.0], dtype=np.float64)

def _safe_std(x: np.ndarray, floor: float = 1e-6) -> float:
    s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return max(s, floor)

class PCNCMedianCalibrator:
    """
    Simple R/G median split:
      - pc_med_rg = median R/G over PC ROIs
      - nc_med_rg = median R/G over NC ROIs
      - rg_thresh = midpoint between medians
    """
    def fit(
        self,
        pc_wells: List[List[Dict[str, Any]]],
        nc_wells: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        pc = _rg_array(pc_wells)  # orange, higher R/G
        nc = _rg_array(nc_wells)  # green, lower R/G
        pc_med = float(np.median(pc))
        nc_med = float(np.median(nc))
        thr = 0.5 * (pc_med + nc_med)
        return {
            "method": "pc_nc_median_rg",
            "rg_thresh": float(thr),
            "pc_med_rg": pc_med,
            "nc_med_rg": nc_med,
        }

class PCNCMeanCalibrator:
    """
    Simple R/G mean split:
      - pc_med_rg = mean R/G over PC ROIs
      - nc_med_rg = mean R/G over NC ROIs
      - rg_thresh = midpoint between means
    """
    def fit(
        self,
        pc_wells: List[List[Dict[str, Any]]],
        nc_wells: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        pc = _rg_array(pc_wells)  # orange, higher R/G
        nc = _rg_array(nc_wells)  # green, lower R/G
        pc_med = float(np.mean(pc))
        nc_med = float(np.mean(nc))
        thr = 0.5 * (pc_med + nc_med)
        return {
            "method": "pc_nc_mean_rg",
            "rg_thresh": float(thr),
            "pc_med_rg": pc_med,
            "nc_med_rg": nc_med,
        }

class PCNCGaussianRGCalibrator:
    """
    Fit normals on R/G for PC (orange) and NC (green).
    Returns means and stds for both groups.
    """
    def fit(
        self,
        pc_wells: List[List[Dict[str, Any]]],
        nc_wells: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        pc = _rg_array(pc_wells)
        nc = _rg_array(nc_wells)
        mu_pc = float(np.mean(pc))
        sd_pc = _safe_std(pc)
        mu_nc = float(np.mean(nc))
        sd_nc = _safe_std(nc)
        return {
            "method": "pc_nc_gaussian_rg",
            "mu_pc": mu_pc, "sd_pc": sd_pc,
            "mu_nc": mu_nc, "sd_nc": sd_nc,
        }


