import numpy as np
from sklearn.mixture import GaussianMixture

from typing import Optional, Tuple, Any
from .config import GatingConfig
from ..sample import FCSFile

def fcs_display_name(fcs: FCSFile) -> str:
    """
    Choose the original FCS filename, file path, or a fallback name.

    Parameters
    ----------
    fcs : FCSFile
        Loaded FCSFile or compatible event and channel container.

    Returns
    -------
    str
        The original FCS filename, file path, or a fallback name.
    """
    return str(
        getattr(fcs, "original_filename", None)
        or getattr(fcs, "path", None)
        or "unknown.fcs"
    )

def freeze_mapping(x: Any) -> Any:
    """
    Recursively turn mappings and sequences into tuples for cache keys.

    Parameters
    ----------
    x : Any
        Mapping or nested value to convert to immutable containers.

    Returns
    -------
    Any
        Depending on the input and branch, returns ``tuple(sorted(((k, freeze_mapping(v))
        for k, v in x.items())))``, ``tuple((freeze_mapping(v) for v in x))``, ``x``.
    """
    if isinstance(x, dict):
        return tuple(sorted((k, freeze_mapping(v)) for k, v in x.items()))
    if isinstance(x, list):
        return tuple(freeze_mapping(v) for v in x)
    if isinstance(x, tuple):
        return tuple(freeze_mapping(v) for v in x)
    return x


def log1p_nonneg(x: np.ndarray) -> np.ndarray:
    """
    Clip negative values to zero and return float32 log1p values.

    Parameters
    ----------
    x : numpy.ndarray
        Numeric observations used to compute the statistic or transformation.

    Returns
    -------
    numpy.ndarray
        Computed value of ``np.log1p(np.clip(x, 0.0, None)).astype(np.float32,
        copy=False)``.
    """
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32, copy=False)


def subsample(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return sampled row indices without replacement, or all row indices.

    Parameters
    ----------
    x : numpy.ndarray
        Numeric observations used to compute the statistic or transformation.
    k : int
        Requested number of sampled rows.
    rng : np.random.Generator
        NumPy random-number generator used for sampling.

    Returns
    -------
    numpy.ndarray
        One-dimensional row indices; k <= 0 or k >= the row count returns all indices.
    """
    if k <= 0 or x.shape[0] <= k:
        return np.arange(x.shape[0])
    return rng.choice(x.shape[0], size=k, replace=False)


def fit_1d_gmm2(x: np.ndarray, rs: int) -> Optional[Tuple[GaussianMixture, np.ndarray]]:
    """
    Fit a two-component mixture to at least fifty finite values.

    Parameters
    ----------
    x : numpy.ndarray
        Numeric observations used to compute the statistic or transformation.
    rs : int
        Random seed for the Gaussian-mixture fit.

    Returns
    -------
    Optional[Tuple[GaussianMixture, numpy.ndarray]]
        Depending on the input and branch, returns ``None``, ``(g, order)``.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return None
    g = GaussianMixture(n_components=2, random_state=int(rs))
    g.fit(x.reshape(-1, 1))
    means = g.means_.ravel()
    order = np.argsort(means)
    return g, order

def mad(x: np.ndarray) -> float:
    """
    Calculate the median absolute deviation from the median.

    Parameters
    ----------
    x : numpy.ndarray
        Numeric observations used to compute the statistic or transformation.

    Returns
    -------
    float
        The median absolute deviation from the median.
    """
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_zscore(cfg: GatingConfig, X: np.ndarray) -> np.ndarray:
    """
    Scale each feature by its median and MAD and apply configured clipping.

    Parameters
    ----------
    cfg : GatingConfig
        Object exposing z_eps and z_clip directly. The current GatingConfig stores these in
        feature_scaling; pass that subconfiguration when calling this helper.
    X : numpy.ndarray
        Numeric array with shape (n_events, n_features).

    Returns
    -------
    numpy.ndarray
        Computed value of ``Z.astype(np.float32, copy=False)``.

    Notes
    -----
    Uses (X - median) / MAD without a normal-consistency factor.
    """
    X = X.astype(np.float32, copy=False)
    med_arr = np.median(X, axis=0)
    mad_arr = np.array([
        mad(X[:, j])
        for j in range(X.shape[1])
    ], dtype=np.float32)
    mad_arr = np.maximum(mad_arr, float(cfg.z_eps))
    Z = (X - med_arr) / mad_arr
    clip = float(cfg.z_clip)
    if clip > 0:
        Z = np.clip(Z, -clip, clip)
    return Z.astype(np.float32, copy=False)

def thr_from_cluster_meds(cfg: GatingConfig, v: np.ndarray) -> float:
    """
    Estimate a threshold between two mixture means, with a robust fallback.

    Parameters
    ----------
    cfg : GatingConfig
        Configuration object controlling this component.
    v : numpy.ndarray
        One-dimensional finite-value vector used to estimate a threshold.

    Returns
    -------
    float
        A threshold between two mixture means, with a robust fallback.
    """
    v = v[np.isfinite(v)]
    if v.size < 4:
        return float(np.median(v)) if v.size else 0.0
    try:
        g = GaussianMixture(n_components=2, random_state=int(cfg.random_state))
        g.fit(v.reshape(-1, 1))
        means = np.sort(g.means_.ravel())
        return float(0.5 * (means[0] + means[1]))
    except Exception:
        med_arr = float(np.median(v))
        mad_arr = float(mad(v))
        return float(med_arr + 3.0 * mad_arr)

