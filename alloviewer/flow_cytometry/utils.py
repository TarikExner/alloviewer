from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


CORE_GATES = ("All Cells", "Singlets", "Lymphocytes")


def norm_path(p: str) -> str:
    """Normalize a path for stable dictionary lookups.

    Parameters
    ----------
    p : str
        Input path.

    Returns
    -------
    str
        Normalized absolute path when resolution succeeds. If resolution fails,
        returns a normalized version of the original path string.
    """
    try:
        return os.path.normcase(os.path.normpath(str(Path(p).resolve())))
    except Exception:
        return os.path.normcase(os.path.normpath(str(p)))


def sanitize_array(a: np.ndarray) -> np.ndarray:
    """Convert an array to finite float32 values.

    Parameters
    ----------
    a : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Float32 array where NaN, positive infinity, and negative infinity are
        replaced by ``0.0``.
    """
    return np.nan_to_num(
        a.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def as_list_finite(a: np.ndarray) -> List[float]:
    """Convert an array to a finite Python list.

    Parameters
    ----------
    a : numpy.ndarray
        Input array.

    Returns
    -------
    list of float
        Sanitized array values as a list.
    """
    return sanitize_array(a).tolist()


def downsample_idx(
    n: int,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices for optional random downsampling.

    Parameters
    ----------
    n : int
        Total number of available points.
    max_points : int
        Maximum number of returned indices.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    numpy.ndarray
        Integer indices. If ``n <= max_points``, all indices are returned.
    """
    if n <= max_points:
        return np.arange(n, dtype=int)

    return rng.choice(n, size=max_points, replace=False).astype(int)


def simpoints(
    x: np.ndarray,
    y: np.ndarray,
    in_gate: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build serializable scatter-plot points.

    Parameters
    ----------
    x : numpy.ndarray
        X-axis values.
    y : numpy.ndarray
        Y-axis values.
    in_gate : numpy.ndarray
        Boolean mask indicating whether each point is in the selected gate.

    Returns
    -------
    list of dict
        List of point dictionaries with ``"x"``, ``"y"``, and ``"inGate"``.
    """
    x2 = sanitize_array(x)
    y2 = sanitize_array(y)
    g2 = in_gate.astype(bool, copy=False)

    return [
        {"x": float(xx), "y": float(yy), "inGate": bool(gg)}
        for xx, yy, gg in zip(x2, y2, g2)
    ]


def population_result_to_dict(p, display_label_fn) -> Dict[str, Any]:
    """Convert a population result to a serializable dictionary.

    Parameters
    ----------
    p : Any
        Population result object with IgG metric attributes.
    display_label_fn : callable
        Function used to convert internal labels to display labels.

    Returns
    -------
    dict
        Serialized population metrics.
    """
    return {
        "label": display_label_fn(p.label),
        "n_events": int(p.n_events),
        "igg_pos_fraction": float(p.igg_pos_fraction),
        "igg_median_raw": float(p.igg_median_raw),
        "igg_median_t": float(p.igg_median_t),
        "igg_median_shift": float(p.igg_median_shift),
        "igg_median_ratio": float(p.igg_median_ratio),
        "igg_fluorescence_index": float(p.igg_fluorescence_index),
        "igg_cutoff_t": float(p.igg_cutoff_t),
        "igg_nc_median_raw": float(p.igg_nc_median_raw),
        "igg_pc_median_raw": (
            None if p.igg_pc_median_raw is None else float(p.igg_pc_median_raw)
        ),
    }


def build_label_maps(
    fitted,
    marker_to_population: Dict[str, str],
) -> tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Build display labels and marker lookup maps.

    Parameters
    ----------
    fitted : Any
        Fitted analysis object containing a panel with marker mappings.
    marker_to_population : dict
        Mapping from internal marker names to requested population labels.

    Returns
    -------
    pop_labels : list of str
        Population labels for display.
    label_to_marker : dict
        Mapping from display population label to internal marker name.
    marker_to_label : dict
        Mapping from internal marker name to display population label.
    marker_to_pop : dict
        Mapping from internal marker name to display population label.

    Notes
    -----
    Duplicate population labels are made unique by appending the marker name.
    """
    marker_names = list((fitted.panel.markers or {}).keys())

    label_to_marker: Dict[str, str] = {}
    pop_labels: List[str] = []
    seen = set()

    for marker_name in marker_names:
        label = (marker_to_population.get(marker_name) or "").strip() or marker_name

        if label in seen:
            label = f"{label} ({marker_name})"

        seen.add(label)
        pop_labels.append(label)
        label_to_marker[label] = marker_name

    marker_to_label: Dict[str, str] = {
        marker_name: label
        for label, marker_name in label_to_marker.items()
    }

    marker_to_pop: Dict[str, str] = {
        marker_name: label
        for label, marker_name in label_to_marker.items()
    }

    return pop_labels, label_to_marker, marker_to_label, marker_to_pop


def build_gate_options(
    fitted,
    pop_labels: List[str],
) -> List[str]:
    """Build selectable gate labels.

    Parameters
    ----------
    fitted : Any
        Fitted analysis object containing panel scatter-channel assignments.
    pop_labels : list of str
        Population labels to add after core gates.

    Returns
    -------
    list of str
        Gate labels for display and downstream selection.
    """
    can_singlets = bool(fitted.panel.fsc_a and fitted.panel.fsc_h)

    gate_options: List[str] = ["All Cells"]

    if can_singlets:
        gate_options.append("Singlets")

    gate_options.append("Lymphocytes")
    gate_options.extend(pop_labels)

    return gate_options


def make_display_label_fn(marker_to_label: Dict[str, str]):
    """Create a function that maps internal labels to display labels.

    Parameters
    ----------
    marker_to_label : dict
        Mapping from internal marker names to display labels.

    Returns
    -------
    callable
        Function accepting a label string and returning its display label.
    """

    def display_label(label: str) -> str:
        """Return a display label for an internal population label.

        Parameters
        ----------
        label : str
            Internal label.

        Returns
        -------
        str
            Display label.
        """
        if label in CORE_GATES:
            return label

        return marker_to_label.get(label, label)

    return display_label


def build_cutoff_by_gate(
    *,
    fitted,
    gate_options: List[str],
    label_to_marker: Dict[str, str],
) -> Dict[str, float]:
    """Build IgG cutoff values keyed by display gate label.

    Parameters
    ----------
    fitted : Any
        Fitted analysis object with ``igg_cutoff_by_gate``.
    gate_options : list of str
        Display gate labels.
    label_to_marker : dict
        Mapping from display population labels to internal marker names.

    Returns
    -------
    dict
        Mapping from display gate label to IgG cutoff.
    """
    cutoff_by_gate: Dict[str, float] = {}

    for gate in gate_options:
        if gate in CORE_GATES:
            cutoff_by_gate[gate] = float(fitted.igg_cutoff_by_gate.get(gate, 0.0))
        else:
            marker_name = label_to_marker.get(gate)
            cutoff_by_gate[gate] = float(
                fitted.igg_cutoff_by_gate.get(marker_name or "", 0.0)
            )

    return cutoff_by_gate


def build_metrics_maps(
    *,
    ds,
    results,
    marker_to_label: Dict[str, str],
) -> tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Build sample-level and file-level metric lookup maps.

    Parameters
    ----------
    ds : Any
        Dataset object with samples and file paths.
    results : iterable
        Analysis results aligned with ``ds.samples``.
    marker_to_label : dict
        Mapping from internal marker names to display labels.

    Returns
    -------
    sample_combined_metrics_by_name : dict
        Nested mapping from sample name to display population label to metrics.
    file_metrics_by_key : dict
        Nested mapping from normalized file path to display population label to
        metrics.
    """
    display_label = make_display_label_fn(marker_to_label)

    sample_combined_metrics_by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    file_metrics_by_key: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ds_sample, sample_result in zip(ds.samples, results):
        sample_combined_metrics_by_name[ds_sample.name] = {
            display_label(pop.label): population_result_to_dict(pop, display_label)
            for pop in sample_result.combined
        }

        for fp, file_result in zip(ds_sample.file_paths, sample_result.per_file):
            file_metrics_by_key[norm_path(fp)] = {
                display_label(pop.label): population_result_to_dict(
                    pop,
                    display_label,
                )
                for pop in file_result.populations
            }

    return sample_combined_metrics_by_name, file_metrics_by_key


def gate_mask_downsampled(entry: Dict[str, Any], gate: str) -> np.ndarray:
    """Return the downsampled mask for a selected gate.

    Parameters
    ----------
    entry : dict
        Plot-cache entry containing masks and marker-positive arrays.
    gate : str
        Selected gate label.

    Returns
    -------
    numpy.ndarray
        Boolean mask for the selected gate. Unknown population gates fall back
        to the lymphocyte mask.
    """
    n = len(entry["igg"])
    all_true = np.ones(n, dtype=bool)

    gate = (gate or "").strip() or "All Cells"

    mask_all = np.asarray(entry.get("mask_all", all_true.tolist()), dtype=bool)
    mask_sing = entry.get("mask_sing")
    mask_lymph = np.asarray(entry.get("mask_lymph", all_true.tolist()), dtype=bool)

    if gate == "All Cells":
        return mask_all

    if gate == "Singlets":
        if mask_sing is None:
            return mask_all

        return np.asarray(mask_sing, dtype=bool)

    if gate == "Lymphocytes":
        return mask_lymph

    marker_pos = entry.get("marker_pos", {}).get(gate)

    if marker_pos is None:
        marker_to_pop = entry.get("marker_to_pop", {}) or {}
        mapped_gate = (marker_to_pop.get(gate) or "").strip()

        if mapped_gate:
            marker_pos = entry.get("marker_pos", {}).get(mapped_gate)

    if marker_pos is None:
        return mask_lymph

    return np.asarray(marker_pos, dtype=bool)


def collect_plot_series(
    *,
    plot_cache: Dict[str, Any],
    selected_key: str,
    gate: str,
    role: str,
    label: str,
    color: str,
    only_selected: bool,
    max_points_final: int,
    max_line_values: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Collect scatter and line-series data for a plot.

    Parameters
    ----------
    plot_cache : dict
        Cache of downsampled per-file plot data.
    selected_key : str
        Cache key for the selected file.
    gate : str
        Selected gate label.
    role : str
        Sample role used when aggregating across files.
    label : str
        Display label for the returned series.
    color : str
        Display color for the returned series.
    only_selected : bool
        If ``True``, use only ``selected_key``. If ``False``, combine all cache
        entries with the requested role.
    max_points_final : int
        Maximum number of scatter points returned.
    max_line_values : int
        Maximum number of line values returned.

    Returns
    -------
    scatter : dict
        Scatter-series data with points and summary counts.
    line : dict
        Line-series data with IgG values and summary counts.
    """
    if only_selected:
        keys = [selected_key]
    else:
        keys = [
            key
            for key, cache_entry in plot_cache.items()
            if cache_entry.get("role") == role
        ]

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    pos: List[np.ndarray] = []

    for key in keys:
        cache_entry = plot_cache[key]
        mask = gate_mask_downsampled(cache_entry, gate)

        igg = np.asarray(cache_entry["igg"], dtype=float)
        ssc = (
            np.asarray(cache_entry["ssc_a"], dtype=float)
            if cache_entry.get("ssc_a") is not None
            else np.zeros_like(igg)
        )
        igg_positive = np.asarray(
            (cache_entry.get("igg_pos_by_gate", {}) or {}).get(
                gate,
                [False] * len(igg),
            ),
            dtype=bool,
        )

        xs.append(igg[mask])
        ys.append(ssc[mask])
        pos.append(igg_positive[mask])

    if xs:
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        p = np.concatenate(pos)
    else:
        x = np.zeros(0, dtype=float)
        y = np.zeros(0, dtype=float)
        p = np.zeros(0, dtype=bool)

    n_total = int(x.shape[0])
    n_pos = int(p.sum())
    pos_pct = (100.0 * n_pos / n_total) if n_total > 0 else 0.0

    if n_total > max_points_final:
        rng = np.random.default_rng(0)
        take = rng.choice(n_total, size=max_points_final, replace=False)
        x = x[take]
        y = y[take]
        p = p[take]

    line_vals = x

    if line_vals.shape[0] > max_line_values:
        rng = np.random.default_rng(1)
        q = np.quantile(line_vals, 0.99)
        tail = line_vals[line_vals >= q]
        rest = line_vals[line_vals < q]

        need = max(0, max_line_values - tail.shape[0])

        if rest.shape[0] > need:
            pick = rng.choice(rest.shape[0], size=need, replace=False)
            rest = rest[pick]

        line_vals = np.concatenate([rest, tail])

    scatter = {
        "label": label,
        "color": color,
        "points": simpoints(x, y, p),
        "n_total": n_total,
        "n_pos": n_pos,
        "pos_pct": float(pos_pct),
    }

    line = {
        "label": label,
        "color": color,
        "values": as_list_finite(line_vals),
        "n_total": n_total,
        "n_pos": n_pos,
        "pos_pct": float(pos_pct),
    }

    return scatter, line


def population_x_label(pop_label: str, pop_to_marker: Dict[str, str]) -> str:
    """Build an x-axis label for a population marker plot.

    Parameters
    ----------
    pop_label : str
        Display population label.
    pop_to_marker : dict
        Mapping from display population label to internal marker name.

    Returns
    -------
    str
        Marker label for the x-axis.
    """
    pop_label = (pop_label or "").strip()
    marker = (pop_to_marker.get(pop_label) or "").strip()

    if not marker:
        return f"{pop_label} marker" if pop_label else "marker"

    if not pop_label or pop_label == marker:
        return marker

    return f"{marker} ({pop_label})"
