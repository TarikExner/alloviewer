from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import IgGCutoffConfig, TransformConfig
from ..fcs_file import FCSFile


@dataclass(frozen=True)
class IgGControlStats:
    """IgG control statistics for one gate.

    Parameters
    ----------
    gate : str
        Gate label.
    nc_n_events : int
        Number of negative-control events used for cutoff estimation.
    pc_n_events : int
        Number of positive-control events used for cutoff estimation.
    nc_median_raw : float
        Median raw IgG value in the negative control.
    nc_median_t : float
        Median transformed IgG value in the negative control.
    pc_median_raw : float or None
        Median raw IgG value in the positive control, when available.
    pc_median_t : float or None
        Median transformed IgG value in the positive control, when available.
    cutoff_t : float
        IgG positivity cutoff in transformed space.
    """

    gate: str
    nc_n_events: int
    pc_n_events: int
    nc_median_raw: float
    nc_median_t: float
    pc_median_raw: Optional[float]
    pc_median_t: Optional[float]
    cutoff_t: float


def _smooth_counts(y: np.ndarray, w: int) -> np.ndarray:
    """Smooth histogram counts with a centered moving average.

    Parameters
    ----------
    y : numpy.ndarray
        Histogram counts.
    w : int
        Smoothing window size. Values below one are clipped to one. Even values
        are increased by one.

    Returns
    -------
    numpy.ndarray
        Smoothed counts as ``float32``.
    """
    w = int(max(1, w))

    if w == 1:
        return y.astype(np.float32, copy=False)

    if w % 2 == 0:
        w += 1

    k = np.ones(w, dtype=np.float32) / float(w)

    return np.convolve(y.astype(np.float32, copy=False), k, mode="same")


def _safe_median(x: np.ndarray) -> float:
    """Return the median of finite values.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.

    Returns
    -------
    float
        Median of finite values, or ``nan`` if no finite values are present.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan")

    return float(np.median(x))


def _safe_ratio(num: float, den: float) -> float:
    """Return a finite ratio.

    Parameters
    ----------
    num : float
        Numerator.
    den : float
        Denominator.

    Returns
    -------
    float
        ``num / den`` when both values are finite and the denominator is not
        near zero. Otherwise returns ``nan``.
    """
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")

    if abs(den) < 1e-12:
        return float("nan")

    return float(num / den)


def _safe_fi(sample: float, neg: float, pos: Optional[float]) -> float:
    """Compute a fluorescence index.

    Parameters
    ----------
    sample : float
        Sample median raw IgG value.
    neg : float
        Negative-control median raw IgG value.
    pos : float or None
        Positive-control median raw IgG value.

    Returns
    -------
    float
        Fluorescence index in percent, or ``nan`` if it cannot be computed.
    """
    if pos is None or not np.isfinite(pos):
        return float("nan")

    if not np.isfinite(sample) or not np.isfinite(neg):
        return float("nan")

    den = pos - neg

    if abs(den) < 1e-12:
        return float("nan")

    return float(((sample - neg) / den) * 100.0)


def igg_cutoff_from_hist(
    *,
    igg_cfg: IgGCutoffConfig,
    transform_cfg: TransformConfig,
    nc_vals: np.ndarray,
    pc_vals: Optional[np.ndarray] = None,
) -> float:
    """Estimate an IgG positivity cutoff from histograms.

    Parameters
    ----------
    igg_cfg : IgGCutoffConfig
        IgG cutoff configuration.
    transform_cfg : TransformConfig
        Transformation configuration. Included for API consistency.
    nc_vals : numpy.ndarray
        Negative-control IgG values in transformed space.
    pc_vals : numpy.ndarray or None, optional
        Positive-control IgG values in transformed space.

    Returns
    -------
    float
        Cutoff in transformed space.

    Notes
    -----
    If enough negative-control events are available, the cutoff is estimated
    from the right tail of the negative-control histogram. If enough
    positive-control events are also available, the cutoff may instead be placed
    where the positive-control density reaches or exceeds the negative-control
    density in the right tail.
    """
    nc = np.asarray(nc_vals, dtype=np.float32)
    nc = nc[np.isfinite(nc)]

    if nc.size < int(igg_cfg.min_events_nc):
        return (
            float(
                np.quantile(nc, float(igg_cfg.fallback_quantile))
                + float(igg_cfg.buffer)
            )
            if nc.size
            else 0.0
        )

    pc = None

    if pc_vals is not None:
        pc0 = np.asarray(pc_vals, dtype=np.float32)
        pc0 = pc0[np.isfinite(pc0)]

        if pc0.size >= int(igg_cfg.min_events_pc):
            pc = pc0

    def _robust_range(x: np.ndarray) -> Tuple[float, float]:
        """Return a padded robust plotting range.

        Parameters
        ----------
        x : numpy.ndarray
            Input values.

        Returns
        -------
        tuple of float
            Lower and upper histogram range.
        """
        lo = float(np.quantile(x, float(igg_cfg.range_q_lo)))
        hi = float(np.quantile(x, float(igg_cfg.range_q_hi)))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(x))
            hi = float(np.max(x))

        if hi <= lo:
            hi = lo + 1.0

        pad = 0.02 * (hi - lo)

        return lo - pad, hi + pad

    lo, hi = _robust_range(nc if pc is None else np.concatenate([nc, pc]))
    bins = int(max(32, igg_cfg.hist_bins))

    cn, edges = np.histogram(nc, bins=bins, range=(lo, hi))
    cn_s = _smooth_counts(cn, int(igg_cfg.hist_smooth))
    peak_i = int(np.argmax(cn_s))
    peak_v = float(cn_s[peak_i])

    tail_thr = float(igg_cfg.tail_frac_of_peak) * peak_v
    need = int(max(1, igg_cfg.tail_consecutive_bins))

    def _tail_cutoff_from_nc() -> float:
        """Estimate cutoff from the negative-control right tail.

        Returns
        -------
        float
            Tail-based cutoff in transformed space.
        """
        run = 0

        for i in range(peak_i + 1, len(cn_s)):
            if float(cn_s[i]) <= tail_thr:
                run += 1
            else:
                run = 0

            if run >= need:
                j = i - need + 1
                x = 0.5 * (edges[j] + edges[j + 1])

                return float(x + float(igg_cfg.buffer))

        return float(
            np.quantile(nc, float(igg_cfg.fallback_quantile))
            + float(igg_cfg.buffer)
        )

    if pc is None:
        return _tail_cutoff_from_nc()

    cp, _ = np.histogram(pc, bins=edges)
    cn_d = cn_s / max(1.0, float(np.sum(cn_s)))
    cp_s = _smooth_counts(cp, int(igg_cfg.hist_smooth))
    cp_d = cp_s / max(1.0, float(np.sum(cp_s)))

    for i in range(peak_i + 1, len(cn_d) - 1):
        if cp_d[i] >= cn_d[i] and float(cn_s[i]) <= (0.25 * peak_v):
            x = 0.5 * (edges[i] + edges[i + 1])
            return float(x + float(igg_cfg.buffer))

    return _tail_cutoff_from_nc()


def get_igg_values_from_mask(
    *,
    transform_cfg: TransformConfig,
    fcs: FCSFile,
    events: np.ndarray,
    mask: np.ndarray,
    igg_channel: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw and transformed IgG values for masked events.

    Parameters
    ----------
    transform_cfg : TransformConfig
        Transformation configuration containing the IgG cofactor.
    fcs : FCSFile
        FCS file used for channel lookup.
    events : numpy.ndarray
        Event matrix.
    mask : numpy.ndarray
        Boolean event mask.
    igg_channel : str
        IgG channel label.

    Returns
    -------
    raw_values : numpy.ndarray
        Raw finite IgG values as ``float32``.
    transformed_values : numpy.ndarray
        ``arcsinh(raw / igg_cofactor)`` values as ``float32``.

    Notes
    -----
    The returned arrays are kept aligned. Non-finite raw or transformed values
    are removed.
    """
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        empty = np.array([], dtype=np.float32)
        return empty, empty

    j = int(fcs.get_channel_index(igg_channel))
    igg_raw = events[:, j].astype(np.float32, copy=False)
    igg_raw = igg_raw[mask]
    igg_raw = igg_raw[np.isfinite(igg_raw)]

    if igg_raw.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty

    igg_t = np.arcsinh(igg_raw / max(float(transform_cfg.igg_cofactor), 1e-12))
    igg_t = igg_t[np.isfinite(igg_t)]

    if igg_t.size != igg_raw.size:
        finite_mask = np.isfinite(
            np.arcsinh(igg_raw / max(float(transform_cfg.igg_cofactor), 1e-12))
        )
        igg_raw = igg_raw[finite_mask]
        igg_t = np.arcsinh(igg_raw / max(float(transform_cfg.igg_cofactor), 1e-12))

    return (
        igg_raw.astype(np.float32, copy=False),
        igg_t.astype(np.float32, copy=False),
    )


def pool_igg_values_from_mask(
    *,
    transform_cfg: TransformConfig,
    fcs: FCSFile,
    events: np.ndarray,
    mask: np.ndarray,
    igg_channel: str,
) -> np.ndarray:
    """Return transformed IgG values for masked events.

    Parameters
    ----------
    transform_cfg : TransformConfig
        Transformation configuration containing the IgG cofactor.
    fcs : FCSFile
        FCS file used for channel lookup.
    events : numpy.ndarray
        Event matrix.
    mask : numpy.ndarray
        Boolean event mask.
    igg_channel : str
        IgG channel label.

    Returns
    -------
    numpy.ndarray
        ``arcsinh(raw / igg_cofactor)`` values as ``float32``.
    """
    _, igg_t = get_igg_values_from_mask(
        transform_cfg=transform_cfg,
        fcs=fcs,
        events=events,
        mask=mask,
        igg_channel=igg_channel,
    )

    return igg_t


def build_igg_control_stats(
    *,
    gate: str,
    igg_cfg: IgGCutoffConfig,
    transform_cfg: TransformConfig,
    nc_raw: np.ndarray,
    nc_t: np.ndarray,
    pc_raw: Optional[np.ndarray] = None,
    pc_t: Optional[np.ndarray] = None,
) -> IgGControlStats:
    """Build IgG control statistics for one gate.

    Parameters
    ----------
    gate : str
        Gate label.
    igg_cfg : IgGCutoffConfig
        IgG cutoff configuration.
    transform_cfg : TransformConfig
        Transformation configuration.
    nc_raw : numpy.ndarray
        Negative-control raw IgG values.
    nc_t : numpy.ndarray
        Negative-control transformed IgG values.
    pc_raw : numpy.ndarray or None, optional
        Positive-control raw IgG values.
    pc_t : numpy.ndarray or None, optional
        Positive-control transformed IgG values.

    Returns
    -------
    IgGControlStats
        Control statistics and cutoff for the gate.
    """
    nc_raw = np.asarray(nc_raw, dtype=np.float32)
    nc_t = np.asarray(nc_t, dtype=np.float32)

    pc_raw_arr = None if pc_raw is None else np.asarray(pc_raw, dtype=np.float32)
    pc_t_arr = None if pc_t is None else np.asarray(pc_t, dtype=np.float32)

    cutoff_t = igg_cutoff_from_hist(
        igg_cfg=igg_cfg,
        transform_cfg=transform_cfg,
        nc_vals=nc_t,
        pc_vals=pc_t_arr,
    )

    return IgGControlStats(
        gate=gate,
        nc_n_events=int(nc_t.size),
        pc_n_events=int(0 if pc_t_arr is None else pc_t_arr.size),
        nc_median_raw=_safe_median(nc_raw),
        nc_median_t=_safe_median(nc_t),
        pc_median_raw=(
            None
            if pc_raw_arr is None or pc_raw_arr.size == 0
            else _safe_median(pc_raw_arr)
        ),
        pc_median_t=(
            None
            if pc_t_arr is None or pc_t_arr.size == 0
            else _safe_median(pc_t_arr)
        ),
        cutoff_t=float(cutoff_t),
    )


def compute_igg_readouts(
    *,
    raw_vals: np.ndarray,
    t_vals: np.ndarray,
    control: IgGControlStats,
) -> Dict[str, float]:
    """Compute IgG readout metrics for one sample or file.

    Parameters
    ----------
    raw_vals : numpy.ndarray
        Raw IgG values.
    t_vals : numpy.ndarray
        Transformed IgG values aligned to ``raw_vals``.
    control : IgGControlStats
        Control statistics and transformed-space cutoff for the gate.

    Returns
    -------
    dict
        IgG readout metrics with event count, positive fraction, medians,
        median shift, median ratio, and fluorescence index.
    """
    raw_vals = np.asarray(raw_vals, dtype=np.float32)
    t_vals = np.asarray(t_vals, dtype=np.float32)

    raw_vals = raw_vals[np.isfinite(raw_vals)]
    t_vals = t_vals[np.isfinite(t_vals)]

    n_events = int(min(raw_vals.size, t_vals.size))

    if n_events == 0:
        return {
            "n_events": 0,
            "igg_pos_fraction": 0.0,
            "igg_median_raw": float("nan"),
            "igg_median_t": float("nan"),
            "igg_median_shift": float("nan"),
            "igg_median_ratio": float("nan"),
            "igg_fluorescence_index": float("nan"),
        }

    raw_vals = raw_vals[:n_events]
    t_vals = t_vals[:n_events]

    sample_median_raw = _safe_median(raw_vals)
    sample_median_t = _safe_median(t_vals)

    n_pos = int(np.sum(t_vals > float(control.cutoff_t)))
    frac_pos = float(n_pos / n_events)

    shift = (
        float(sample_median_raw - control.nc_median_raw)
        if np.isfinite(sample_median_raw) and np.isfinite(control.nc_median_raw)
        else float("nan")
    )

    ratio = _safe_ratio(sample_median_raw, control.nc_median_raw)
    fi = _safe_fi(sample_median_raw, control.nc_median_raw, control.pc_median_raw)

    return {
        "n_events": int(n_events),
        "igg_pos_fraction": float(frac_pos),
        "igg_median_raw": float(sample_median_raw),
        "igg_median_t": float(sample_median_t),
        "igg_median_shift": float(shift),
        "igg_median_ratio": float(ratio),
        "igg_fluorescence_index": float(fi),
    }
