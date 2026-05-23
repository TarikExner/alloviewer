import numpy as np
from typing import List, Dict, Any


def _rg_array(wells: List[List[Dict[str, Any]]]) -> np.ndarray:
    """Return ROI-level red/green ratios from nested well data.

    Parameters
    ----------
    wells : list of list of dict
        Nested well structure. Each inner list contains ROI dictionaries.
        Each ROI must contain ``"mean_r"`` and ``"mean_g"`` values.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of R/G ratios as ``float64``. If no ROIs are
        present, returns ``array([1.0])`` to provide a neutral fallback value.

    Notes
    -----
    The green channel denominator is clipped to at least ``1e-6`` to avoid
    division by zero.
    """
    xs = []
    for rois in wells:
        for r in rois:
            xs.append(r["mean_r"] / max(1e-6, r["mean_g"]))
    return np.array(xs, dtype=np.float64) if xs else np.array([1.0], dtype=np.float64)


def _safe_std(x: np.ndarray, floor: float = 1e-6) -> float:
    """Return a standard deviation with a lower bound.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.
    floor : float, optional
        Minimum standard deviation returned. The default is ``1e-6``.

    Returns
    -------
    float
        Sample standard deviation using ``ddof=1`` when at least two values are
        present. Returns at least ``floor``.
    """
    s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return max(s, floor)

def _xy_array(wells: List[List[Dict[str, Any]]]) -> np.ndarray:
    """Return ROI-level red and green channel means as a 2D array.

    Parameters
    ----------
    wells : list of list of dict
        Nested well structure. Each ROI dictionary must contain ``"mean_r"``
        and ``"mean_g"`` values.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_rois, 2)``. The first column contains red-channel
        means and the second column contains green-channel means.
    """
    rows = []
    for well in wells:
        for r in well:
            rows.append([float(r["mean_r"]), float(r["mean_g"])])
    return np.asarray(rows, dtype=float)


class PCNCMedianCalibrator:
    """Calibrate an R/G threshold from PC and NC median ratios.

    The threshold is the midpoint between the median R/G ratio of positive
    control ROIs and the median R/G ratio of negative control ROIs.
    """

    def fit(
        self,
        pc_wells: List[List[Dict[str, Any]]],
        nc_wells: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Fit the median-based R/G calibration.

        Parameters
        ----------
        pc_wells : list of list of dict
            Positive control wells. ROIs are expected to have higher R/G values.
        nc_wells : list of list of dict
            Negative control wells. ROIs are expected to have lower R/G values.

        Returns
        -------
        dict
            Calibration parameters with the following keys:

            ``"method"``
                Name of the calibration method.
            ``"rg_thresh"``
                Midpoint between PC and NC median R/G ratios.
            ``"pc_med_rg"``
                Median R/G ratio of positive control ROIs.
            ``"nc_med_rg"``
                Median R/G ratio of negative control ROIs.
        """
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
    """Calibrate an R/G threshold from PC and NC mean ratios.

    The threshold is the midpoint between the mean R/G ratio of positive
    control ROIs and the mean R/G ratio of negative control ROIs.
    """

    def fit(
        self,
        pc_wells: List[List[Dict[str, Any]]],
        nc_wells: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Fit the mean-based R/G calibration.

        Parameters
        ----------
        pc_wells : list of list of dict
            Positive control wells. ROIs are expected to have higher R/G values.
        nc_wells : list of list of dict
            Negative control wells. ROIs are expected to have lower R/G values.

        Returns
        -------
        dict
            Calibration parameters with the following keys:

            ``"method"``
                Name of the calibration method.
            ``"rg_thresh"``
                Midpoint between PC and NC mean R/G ratios.
            ``"pc_med_rg"``
                Mean R/G ratio of positive control ROIs. The key name is kept
                for compatibility, although the value is a mean.
            ``"nc_med_rg"``
                Mean R/G ratio of negative control ROIs. The key name is kept
                for compatibility, although the value is a mean.
        """
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
    """Fit one-dimensional Gaussian R/G models for PC and NC ROIs.

    This calibrator estimates the mean and standard deviation of the R/G ratio
    separately for positive and negative control ROIs.
    """

    def fit(
        self,
        pc_wells: List[List[Dict[str, Any]]],
        nc_wells: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Fit Gaussian R/G parameters for positive and negative controls.

        Parameters
        ----------
        pc_wells : list of list of dict
            Positive control wells. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.
        nc_wells : list of list of dict
            Negative control wells. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.

        Returns
        -------
        dict
            Gaussian calibration parameters with the following keys:

            ``"method"``
                Name of the calibration method.
            ``"mu_pc"``
                Mean R/G ratio of positive control ROIs.
            ``"sd_pc"``
                Standard deviation of positive control R/G ratios.
            ``"mu_nc"``
                Mean R/G ratio of negative control ROIs.
            ``"sd_nc"``
                Standard deviation of negative control R/G ratios.

        Notes
        -----
        Standard deviations are lower-bounded by ``1e-6``.
        """
        pc = _rg_array(pc_wells)
        nc = _rg_array(nc_wells)
        mu_pc = float(np.mean(pc))
        sd_pc = _safe_std(pc)
        mu_nc = float(np.mean(nc))
        sd_nc = _safe_std(nc)
        return {
            "method": "pc_nc_gaussian_rg",
            "mu_pc": mu_pc,
            "sd_pc": sd_pc,
            "mu_nc": mu_nc,
            "sd_nc": sd_nc,
        }



class PCNCGaussian2DCalibrator:
    """Fit two-dimensional Gaussian models for PC and NC ROIs.

    This calibrator models each ROI by its red- and green-channel mean
    intensities. It estimates one mean vector and one covariance matrix for
    positive controls and one pair for negative controls.
    """

    def fit(
        self,
        pc_wells: List[List[Dict[str, Any]]],
        nc_wells: List[List[Dict[str, Any]]],
        eps: float = 1e-6,
    ) -> Dict[str, Any]:
        """Fit 2D Gaussian parameters for positive and negative controls.

        Parameters
        ----------
        pc_wells : list of list of dict
            Positive control wells. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.
        nc_wells : list of list of dict
            Negative control wells. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.
        eps : float, optional
            Diagonal regularization added to each covariance matrix. The
            default is ``1e-6``.

        Returns
        -------
        dict
            Two-dimensional Gaussian calibration parameters with the following
            keys:

            ``"method"``
                Name of the calibration method.
            ``"mu_pc"``
                Positive control mean vector with shape ``(2,)``.
            ``"cov_pc"``
                Positive control covariance matrix with shape ``(2, 2)``.
            ``"mu_nc"``
                Negative control mean vector with shape ``(2,)``.
            ``"cov_nc"``
                Negative control covariance matrix with shape ``(2, 2)``.

        Notes
        -----
        The mean vectors and covariance matrices are based on ``mean_r`` and
        ``mean_g`` directly, not on the R/G ratio.
        """
        pc = _xy_array(pc_wells)
        nc = _xy_array(nc_wells)

        mu_pc = pc.mean(axis=0)
        mu_nc = nc.mean(axis=0)

        cov_pc = np.cov(pc.T) + eps * np.eye(2)
        cov_nc = np.cov(nc.T) + eps * np.eye(2)

        return {
            "method": "pc_nc_gaussian_2d",
            "mu_pc": mu_pc,
            "cov_pc": cov_pc,
            "mu_nc": mu_nc,
            "cov_nc": cov_nc,
        }
