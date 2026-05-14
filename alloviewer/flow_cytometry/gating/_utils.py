import numpy as np
from sklearn.mixture import GaussianMixture

from typing import Optional, Tuple, Any
from .config import GatingConfig
from ..sample import FCSFile

def fcs_display_name(fcs: FCSFile) -> str:
    return str(
        getattr(fcs, "original_filename", None)
        or getattr(fcs, "path", None)
        or "unknown.fcs"
    )

def freeze_mapping(x: Any) -> Any:
    if isinstance(x, dict):
        return tuple(sorted((k, freeze_mapping(v)) for k, v in x.items()))
    if isinstance(x, list):
        return tuple(freeze_mapping(v) for v in x)
    if isinstance(x, tuple):
        return tuple(freeze_mapping(v) for v in x)
    return x


def log1p_nonneg(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32, copy=False)


def subsample(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or x.shape[0] <= k:
        return np.arange(x.shape[0])
    return rng.choice(x.shape[0], size=k, replace=False)


def fit_1d_gmm2(x: np.ndarray, rs: int) -> Optional[Tuple[GaussianMixture, np.ndarray]]:
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
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_zscore(cfg: GatingConfig, X: np.ndarray) -> np.ndarray:
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

