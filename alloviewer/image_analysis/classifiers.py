import math
from typing import List, Dict, Any, Optional

import numpy as np

from .config import DEFAULT_CALIB_RG_GAUSS


def _phi_cdf(z: float) -> float:
    """Return the standard normal cumulative distribution value.

    Parameters
    ----------
    z : float
        Standardized input value.

    Returns
    -------
    float
        Probability mass of the standard normal distribution up to ``z``.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normalize_gauss_calib(calib: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill missing Gaussian calibration values and clamp standard deviations.

    Parameters
    ----------
    calib : dict or None
        Calibration dictionary. Missing values are filled from
        ``DEFAULT_CALIB_RG_GAUSS``.

    Returns
    -------
    dict
        Calibration dictionary containing Gaussian R/G parameters. The keys
        ``"sd_pc"`` and ``"sd_nc"`` are forced to be at least ``1e-6``.

    Notes
    -----
    This function returns a shallow merged copy and does not modify the input
    dictionary.
    """
    base = DEFAULT_CALIB_RG_GAUSS
    c = {**base, **(calib or {})}
    c["sd_pc"] = max(float(c["sd_pc"]), 1e-6)
    c["sd_nc"] = max(float(c["sd_nc"]), 1e-6)
    c.setdefault("method", "pc_nc_gaussian_rg")
    return c


class ROIClassifier:
    """Classify ROIs with a binary R/G threshold.

    The classifier labels an ROI as ``"pos"`` when its red/green ratio is
    greater than or equal to the threshold, and ``"neg"`` otherwise.

    """

    def __init__(self, calib: Optional[Dict[str, Any]] = None):
        """Initialize the threshold classifier.

        Parameters
        ----------
        calib : dict or None, optional
            Calibration dictionary containing ``"rg_thresh"``. If absent, the
            threshold is set to the midpoint between the default positive and
            negative Gaussian R/G means.
        """
        if calib is None or "rg_thresh" not in calib:
            # fallback: midpoint between default R/G means (PC high, NC low)
            thr = 0.5 * (
                float(DEFAULT_CALIB_RG_GAUSS["mu_pc"])
                + float(DEFAULT_CALIB_RG_GAUSS["mu_nc"])
            )
            self.thr = float(thr)
            self.method = "pc_nc_median_rg_fallback"
        else:
            self.thr = float(calib["rg_thresh"])
            self.method = calib.get("method", "pc_nc_median_rg")

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify ROIs.

        Parameters
        ----------
        rois : list of dict
            ROI dictionaries. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.

        Returns
        -------
        list of dict
            Copies of the input ROI dictionaries with added keys:

            ``"score"``
                R/G ratio.
            ``"label"``
                ``"pos"`` or ``"neg"``.
            ``"method"``
                Name of the classification method.

        Notes
        -----
        The green channel denominator is clipped to at least ``1e-6``.
        """
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
    """Classify ROIs with a median-derived R/G threshold.

    The classifier expects a calibration dictionary with ``"rg_thresh"``. ROIs
    with R/G ratios greater than or equal to the threshold are labeled
    ``"pos"``; all others are labeled ``"neg"``.
    """

    def __init__(self, calib: Optional[Dict[str, Any]] = None):
        """Initialize the median R/G threshold classifier.

        Parameters
        ----------
        calib : dict or None, optional
            Calibration dictionary containing ``"rg_thresh"``. If absent, the
            threshold is set to the midpoint between the default positive and
            negative Gaussian R/G means.
        """
        if calib is None or "rg_thresh" not in calib:
            thr = 0.5 * (
                float(DEFAULT_CALIB_RG_GAUSS["mu_pc"])
                + float(DEFAULT_CALIB_RG_GAUSS["mu_nc"])
            )
            self.thr = float(thr)
            self.method = "pc_nc_median_rg_fallback"
        else:
            self.thr = float(calib["rg_thresh"])
            self.method = calib.get("method", "pc_nc_median_rg")

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify ROIs.

        Parameters
        ----------
        rois : list of dict
            ROI dictionaries. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.

        Returns
        -------
        list of dict
            Copies of the input ROI dictionaries with added R/G score, binary
            label, and method name.
        """
        out: List[Dict[str, Any]] = []
        for r in rois:
            score = r["mean_r"] / max(1e-6, r["mean_g"])  # R/G
            rr = dict(r)
            rr["score"] = float(score)
            rr["label"] = "pos" if score >= self.thr else "neg"
            rr["method"] = self.method
            out.append(rr)
        return out


class ROIClassifierNCUpper:
    """Classify ROIs by testing against the NC upper tail.

    ROIs are labeled ``"pos"`` when their R/G ratio is at least ``k`` standard
    deviations above the negative control mean. Other ROIs are labeled
    ``"uncertain"``.
    """

    def __init__(self, calib: Optional[Dict[str, Any]] = None, k: float = 3.0):
        """Initialize the NC upper-tail classifier.

        Parameters
        ----------
        calib : dict or None, optional
            Gaussian R/G calibration dictionary with ``"mu_nc"`` and
            ``"sd_nc"``. Missing values are filled from defaults.
        k : float, optional
            Z-score cutoff above the negative control mean. The default is
            ``3.0``.
        """
        c = _normalize_gauss_calib(calib)
        self.mu_nc = float(c["mu_nc"])
        self.sd_nc = float(c["sd_nc"])
        self.k = float(k)
        self.method = c.get("method", "pc_nc_gaussian_rg") + f"_NCUpper_k{self.k:g}"

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify ROIs using the NC upper-tail rule.

        Parameters
        ----------
        rois : list of dict
            ROI dictionaries. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.

        Returns
        -------
        list of dict
            Copies of the input ROI dictionaries with added keys:

            ``"score_rg"``
                R/G ratio.
            ``"z_nc"``
                Z-score relative to the negative control distribution.
            ``"p_nc_upper"``
                Upper-tail probability under the negative control model.
            ``"label"``
                ``"pos"`` or ``"uncertain"``.
            ``"method"``
                Name of the classification method.
        """
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
    """Classify ROIs by testing against the PC lower tail.

    ROIs are labeled ``"neg"`` when their R/G ratio is at least ``k`` standard
    deviations below the positive control mean. Other ROIs are labeled
    ``"uncertain"``.
    """

    def __init__(self, calib: Optional[Dict[str, Any]] = None, k: float = 3.0):
        """Initialize the PC lower-tail classifier.

        Parameters
        ----------
        calib : dict or None, optional
            Gaussian R/G calibration dictionary with ``"mu_pc"`` and
            ``"sd_pc"``. Missing values are filled from defaults.
        k : float, optional
            Z-score cutoff below the positive control mean. The default is
            ``3.0``.
        """
        c = _normalize_gauss_calib(calib)
        self.mu_pc = float(c["mu_pc"])
        self.sd_pc = float(c["sd_pc"])
        self.k = float(k)
        self.method = c.get("method", "pc_nc_gaussian_rg") + f"_PCLower_k{self.k:g}"

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify ROIs using the PC lower-tail rule.

        Parameters
        ----------
        rois : list of dict
            ROI dictionaries. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.

        Returns
        -------
        list of dict
            Copies of the input ROI dictionaries with added keys:

            ``"score_rg"``
                R/G ratio.
            ``"z_pc"``
                Z-score relative to the positive control distribution.
            ``"p_pc_lower"``
                Lower-tail probability under the positive control model.
            ``"label"``
                ``"neg"`` or ``"uncertain"``.
            ``"method"``
                Name of the classification method.
        """
        out: List[Dict[str, Any]] = []
        for r in rois:
            rg = r["mean_r"] / max(1e-6, r["mean_g"])
            z_pc = (rg - self.mu_pc) / self.sd_pc
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
    """Classify ROIs into positive, negative, or uncertain by Gaussian R/G tests.

    The classifier combines two one-sided tests:

    - ``"pos"`` if the R/G ratio is high relative to the negative control.
    - ``"neg"`` if the R/G ratio is low relative to the positive control.
    - ``"uncertain"`` otherwise.
    """

    def __init__(self, calib: Optional[Dict[str, Any]] = None, k: float = 2.0):
        """Initialize the three-way Gaussian R/G classifier.

        Parameters
        ----------
        calib : dict or None, optional
            Gaussian R/G calibration dictionary with ``"mu_nc"``, ``"sd_nc"``,
            ``"mu_pc"``, and ``"sd_pc"``. Missing values are filled from
            defaults.
        k : float, optional
            Absolute Z-score cutoff for assigning positive or negative labels.
            The default is ``2.0``.
        """
        c = _normalize_gauss_calib(calib)
        self.mu_nc = float(c["mu_nc"])
        self.sd_nc = float(c["sd_nc"])
        self.mu_pc = float(c["mu_pc"])
        self.sd_pc = float(c["sd_pc"])
        self.k = float(k)
        self.method = c.get("method", "pc_nc_gaussian_rg") + f"_3way_k{self.k:g}"

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify ROIs into ``"pos"``, ``"neg"``, or ``"uncertain"``.

        Parameters
        ----------
        rois : list of dict
            ROI dictionaries. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.

        Returns
        -------
        list of dict
            Copies of the input ROI dictionaries with added R/G score, PC and
            NC Z-scores, tail probabilities, label, and method name.
        """
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


class ROIClassifierGaussian2D3Way:
    """Classify ROIs with two-dimensional Gaussian PC and NC models.

    Each ROI is represented by ``mean_r`` and ``mean_g``. The classifier
    compares the log-density under the positive control model with the
    log-density under the negative control model.

    Labels are assigned from the log-density difference:

    - ``"pos"`` if ``logp_pc - logp_nc > margin``.
    - ``"neg"`` if ``logp_pc - logp_nc < -margin``.
    - ``"uncertain"`` otherwise.
    """

    def __init__(self, calib: Dict[str, Any], margin: float = 1.0):
        """Initialize the two-dimensional Gaussian classifier.

        Parameters
        ----------
        calib : dict
            Calibration dictionary containing ``"mu_pc"``, ``"cov_pc"``,
            ``"mu_nc"``, and ``"cov_nc"``.
        margin : float, optional
            Minimum absolute log-density difference required for a positive or
            negative label. The default is ``1.0``.

        Raises
        ------
        KeyError
            If a required calibration key is missing.
        numpy.linalg.LinAlgError
            If either covariance matrix cannot be inverted.

        Notes
        -----
        The normalizing constant ``-log(2*pi)`` is omitted in the log-density
        because it cancels when comparing two two-dimensional Gaussian models.
        """
        self.mu_pc = np.asarray(calib["mu_pc"], dtype=float)
        self.cov_pc = np.asarray(calib["cov_pc"], dtype=float)
        self.mu_nc = np.asarray(calib["mu_nc"], dtype=float)
        self.cov_nc = np.asarray(calib["cov_nc"], dtype=float)
        self.margin = float(margin)
        self.method = (
            calib.get("method", "pc_nc_gaussian_2d")
            + f"_3way_margin{self.margin:g}"
        )

        self.inv_pc = np.linalg.inv(self.cov_pc)
        self.inv_nc = np.linalg.inv(self.cov_nc)
        self.logdet_pc = np.linalg.slogdet(self.cov_pc)[1]
        self.logdet_nc = np.linalg.slogdet(self.cov_nc)[1]

    def _logpdf(
        self,
        x: np.ndarray,
        mu: np.ndarray,
        inv: np.ndarray,
        logdet: float,
    ) -> float:
        """Return the Gaussian log-density up to an additive constant.

        Parameters
        ----------
        x : numpy.ndarray
            Observation vector with shape ``(2,)``.
        mu : numpy.ndarray
            Mean vector with shape ``(2,)``.
        inv : numpy.ndarray
            Inverse covariance matrix with shape ``(2, 2)``.
        logdet : float
            Log-determinant of the covariance matrix.

        Returns
        -------
        float
            Gaussian log-density without the constant term shared by both
            classes.
        """
        d = x - mu
        return -0.5 * (d @ inv @ d + logdet)

    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify ROIs into ``"pos"``, ``"neg"``, or ``"uncertain"``.

        Parameters
        ----------
        rois : list of dict
            ROI dictionaries. Each ROI must contain ``"mean_r"`` and
            ``"mean_g"``.

        Returns
        -------
        list of dict
            Copies of the input ROI dictionaries with added keys:

            ``"logp_pc"``
                Log-density under the positive control model.
            ``"logp_nc"``
                Log-density under the negative control model.
            ``"score"``
                Difference ``logp_pc - logp_nc``.
            ``"label"``
                ``"pos"``, ``"neg"``, or ``"uncertain"``.
            ``"method"``
                Name of the classification method.
        """
        out = []
        for r in rois:
            x = np.array([float(r["mean_r"]), float(r["mean_g"])], dtype=float)

            logp_pc = self._logpdf(x, self.mu_pc, self.inv_pc, self.logdet_pc)
            logp_nc = self._logpdf(x, self.mu_nc, self.inv_nc, self.logdet_nc)
            delta = logp_pc - logp_nc

            if delta > self.margin:
                label = "pos"
            elif delta < -self.margin:
                label = "neg"
            else:
                label = "uncertain"

            rr = dict(r)
            rr["logp_pc"] = float(logp_pc)
            rr["logp_nc"] = float(logp_nc)
            rr["score"] = float(delta)
            rr["label"] = label
            rr["method"] = self.method
            out.append(rr)

        return out
