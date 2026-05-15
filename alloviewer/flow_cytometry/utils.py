from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


CORE_GATES = ("All Cells", "Singlets", "Lymphocytes")


def norm_path(p: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(p).resolve())))
    except Exception:
        return os.path.normcase(os.path.normpath(str(p)))


def sanitize_array(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        a.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def as_list_finite(a: np.ndarray) -> List[float]:
    return sanitize_array(a).tolist()


def downsample_idx(
    n: int,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)

    return rng.choice(n, size=max_points, replace=False).astype(int)


def simpoints(
    x: np.ndarray,
    y: np.ndarray,
    in_gate: np.ndarray,
) -> List[Dict[str, Any]]:
    x2 = sanitize_array(x)
    y2 = sanitize_array(y)
    g2 = in_gate.astype(bool, copy=False)

    return [
        {"x": float(xx), "y": float(yy), "inGate": bool(gg)}
        for xx, yy, gg in zip(x2, y2, g2)
    ]


def population_result_to_dict(p, display_label_fn) -> Dict[str, Any]:
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
    """
    Returns:
      - pop_labels
      - label_to_marker: frontend population label -> internal marker name
      - marker_to_label: internal marker name -> frontend population label
      - marker_to_pop: internal marker name -> frontend population label
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
        marker_name: label for label, marker_name in label_to_marker.items()
    }

    marker_to_pop: Dict[str, str] = {
        marker_name: label for label, marker_name in label_to_marker.items()
    }

    return pop_labels, label_to_marker, marker_to_label, marker_to_pop


def build_gate_options(
    fitted,
    pop_labels: List[str],
) -> List[str]:
    can_singlets = bool(fitted.panel.fsc_a and fitted.panel.fsc_h)

    gate_options: List[str] = ["All Cells"]

    if can_singlets:
        gate_options.append("Singlets")

    gate_options.append("Lymphocytes")
    gate_options.extend(pop_labels)

    return gate_options


def make_display_label_fn(marker_to_label: Dict[str, str]):
    def display_label(label: str) -> str:
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
    """
    Downsampled mask for the selected end-gate.

    Gate options include:
      - All Cells
      - Singlets
      - Lymphocytes
      - population labels
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
    pop_label = (pop_label or "").strip()
    marker = (pop_to_marker.get(pop_label) or "").strip()

    if not marker:
        return f"{pop_label} marker" if pop_label else "marker"

    if not pop_label or pop_label == marker:
        return marker

    return f"{marker} ({pop_label})"
