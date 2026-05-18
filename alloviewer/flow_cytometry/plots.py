from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pathlib import Path

from .gating.gater import FittedGater
from .utils import (
    CORE_GATES,
    as_list_finite,
    build_cutoff_by_gate,
    build_gate_options,
    build_label_maps,
    build_metrics_maps,
    collect_plot_series,
    downsample_idx,
    make_display_label_fn,
    norm_path,
    population_result_to_dict,
    population_x_label,
    simpoints,
)


ProgressEvent = Dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]

def _build_payload(
    *,
    fitted: FittedGater,
    results,
    marker_to_population: Dict[str, str],
    gate_options: List[str],
    label_to_marker: Dict[str, str],
    marker_to_label: Dict[str, str],
) -> Dict[str, Any]:
    display_label = make_display_label_fn(marker_to_label)

    payload: Dict[str, Any] = {
        "ok": True,
        "results": [
            {
                "sample_name": result.sample_name,
                "role": result.role,
                "n_files": result.n_files,
                "combined": [
                    population_result_to_dict(pop, display_label)
                    for pop in result.combined
                ],
                "per_file": [
                    {
                        "file_name": file_result.file_name,
                        "populations": [
                            population_result_to_dict(pop, display_label)
                            for pop in file_result.populations
                        ],
                    }
                    for file_result in result.per_file
                ],
            }
            for result in results
        ],
        "panel_used": {
            "fsc_a": fitted.panel.fsc_a,
            "fsc_h": fitted.panel.fsc_h,
            "ssc_a": fitted.panel.ssc_a,
            "igg": fitted.panel.igg,
            "markers": dict(fitted.panel.markers or {}),
            "marker_to_population": dict(marker_to_population),
        },
    }

    payload["panel_used"]["cutoff_by_gate"] = build_cutoff_by_gate(
        fitted=fitted,
        gate_options=gate_options,
        label_to_marker=label_to_marker,
    )

    return payload


def build_plot_cache_entry_for_file(
    *,
    sample,
    fp: str,
    fcs,
    gater,
    fitted: FittedGater,
    marker_names: List[str],
    label_to_marker: Dict[str, str],
    marker_to_pop: Dict[str, str],
    gate_options: List[str],
    cutoff_by_gate: Dict[str, float],
    file_metrics_by_key: Dict[str, Dict[str, Dict[str, Any]]],
    sample_combined_metrics_by_name: Dict[str, Dict[str, Dict[str, Any]]],
    max_points: int,
    rng: np.random.Generator,
) -> tuple[str, Dict[str, Any]]:
    """
    Build plot-cache data for one FCS file.

    This is the file-level unit used for progress tracking.
    It uses gater.analyze_file_cached(fcs, fitted).
    """
    fa = gater.analyze_file_cached(fcs, fitted)

    events = fa.events
    if events is None:
        raise ValueError(f"{fp}: file analysis has no events.")

    n = int(events.shape[0])
    idx = downsample_idx(n, max_points=max_points, rng=rng)
    n_ds = int(idx.size)

    fsc_a = None
    fsc_h = None
    ssc_a = None

    if fitted.panel.fsc_a:
        ia = fcs.get_channel_index(fitted.panel.fsc_a)
        fsc_a = events[:, ia].astype(np.float32, copy=False)

    if fitted.panel.fsc_h:
        ih = fcs.get_channel_index(fitted.panel.fsc_h)
        fsc_h = events[:, ih].astype(np.float32, copy=False)

    if fitted.panel.ssc_a:
        issc = fcs.get_channel_index(fitted.panel.ssc_a)
        ssc_a = events[:, issc].astype(np.float32, copy=False)

    jigg = fcs.get_channel_index(fitted.panel.igg)
    igg_raw = events[:, jigg].astype(np.float32, copy=False)
    igg = gater.transform_channel(igg_raw)
    igg_ds = igg[idx]
    igg_raw_ds = igg_raw[idx]

    mask_all_ds = np.asarray(fa.mask_edge, dtype=bool)[idx].copy()

    if fa.mask_sing is not None:
        mask_sing_ds = np.asarray(fa.mask_sing, dtype=bool)[idx].copy().tolist()
    else:
        mask_sing_ds = None

    mask_lymph_ds_raw = np.asarray(fa.mask_lymph_raw, dtype=bool)[idx].copy()
    mask_lymph_ds_clean = np.asarray(fa.mask_lymph, dtype=bool)[idx].copy()

    m_by_marker_full = fa.m_by_marker or {}

    m_by_marker_ds: Dict[str, np.ndarray] = {
        marker_name: np.asarray(
            m_by_marker_full.get(marker_name, np.zeros(n, dtype=bool)),
            dtype=bool,
        )[idx].copy()
        for marker_name in marker_names
    }

    marker_pos_by_label_ds: Dict[str, np.ndarray] = {}

    for pop_label, marker_name in label_to_marker.items():
        marker_pos_by_label_ds[pop_label] = m_by_marker_ds.get(
            marker_name,
            np.zeros(n_ds, dtype=bool),
        )

    marker_vals_ds: Dict[str, List[float]] = {}

    for pop_label, marker_name in label_to_marker.items():
        channel = fitted.panel.markers[marker_name]
        j = fcs.get_channel_index(channel)

        cofactor = float(
            getattr(fitted, "marker_cofactors", {}).get(
                marker_name,
                fitted.config.transform.igg_cofactor,
            )
        )

        values = gater.transform_channel(
            events[:, j].astype(np.float32, copy=False),
            cofactor=cofactor,
        )

        marker_vals_ds[pop_label] = as_list_finite(values[idx])

    igg_pos_by_gate: Dict[str, List[bool]] = {}

    for gate in gate_options:
        cutoff = float(cutoff_by_gate.get(gate, 0.0))
        igg_pos_by_gate[gate] = (igg_ds > cutoff).tolist()

    for marker_name in marker_names:
        marker_mask = np.asarray(
            fa.m_by_marker.get(marker_name, np.zeros(n, dtype=bool)),
            dtype=bool,
        )
        if np.any(marker_mask & ~np.asarray(fa.mask_lymph, dtype=bool)):
            raise RuntimeError(
                f"{fp}: marker mask {marker_name} leaks outside cleaned lymph"
            )

    key = norm_path(fp)

    entry = {
        "file_key": key,
        "file_key_raw": fp,
        "sample_name": sample.name,
        "role": sample.role,
        "gate_options": gate_options,
        "cutoff_by_gate": dict(cutoff_by_gate),

        # Downsampled scatter and IgG.
        "fsc_a": as_list_finite(fsc_a[idx]) if fsc_a is not None else None,
        "fsc_h": as_list_finite(fsc_h[idx]) if fsc_h is not None else None,
        "ssc_a": as_list_finite(ssc_a[idx]) if ssc_a is not None else None,
        "igg": as_list_finite(igg_ds),
        "igg_raw": as_list_finite(igg_raw_ds),

        # Downsampled masks.
        "mask_all": mask_all_ds.tolist(),
        "mask_sing": mask_sing_ds,
        "mask_lymph_raw": mask_lymph_ds_raw.tolist(),
        "mask_lymph": mask_lymph_ds_clean.tolist(),

        # Populations.
        "marker_pos": {
            key_: marker_pos_by_label_ds[key_].tolist()
            for key_ in marker_pos_by_label_ds
        },
        "marker_vals": marker_vals_ds,

        "pop_to_marker": dict(label_to_marker),
        "marker_to_pop": dict(marker_to_pop),

        # IgG positivity.
        "igg_pos_by_gate": dict(igg_pos_by_gate),

        # Metrics for selected gate display.
        "selected_file_metrics_by_gate": file_metrics_by_key.get(key, {}),
        "sample_combined_metrics_by_gate": sample_combined_metrics_by_name.get(
            sample.name,
            {},
        ),
    }

    return key, entry


def make_results_payload(
    ds,
    gater,
    fitted: FittedGater,
    results,
    marker_to_population: Dict[str, str],
    max_points: int = 20_000,
    seed: int = 0,
    progress_cb: Optional[ProgressCallback] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build:
      - JSON payload for the frontend
      - plot_cache with one entry per file

    File-level work is handled by build_plot_cache_entry_for_file().
    This allows the pipeline to report progress per file.
    """
    rng = np.random.default_rng(seed)

    marker_names = list((fitted.panel.markers or {}).keys())

    (
        pop_labels,
        label_to_marker,
        marker_to_label,
        marker_to_pop,
    ) = build_label_maps(
        fitted=fitted,
        marker_to_population=marker_to_population,
    )

    gate_options = build_gate_options(
        fitted=fitted,
        pop_labels=pop_labels,
    )

    payload = _build_payload(
        fitted=fitted,
        results=results,
        marker_to_population=marker_to_population,
        gate_options=gate_options,
        label_to_marker=label_to_marker,
        marker_to_label=marker_to_label,
    )

    cutoff_by_gate = dict(payload["panel_used"]["cutoff_by_gate"])

    (
        sample_combined_metrics_by_name,
        file_metrics_by_key,
    ) = build_metrics_maps(
        ds=ds,
        results=results,
        marker_to_label=marker_to_label,
    )

    plot_cache: Dict[str, Any] = {}

    if not hasattr(gater, "analyze_file_cached"):
        raise AttributeError(
            "Gater must implement analyze_file_cached(fcs, fitted) for plot caching."
        )

    for sample in ds.samples:
        for fp, fcs in zip(sample.file_paths, sample.files):
            key, entry = build_plot_cache_entry_for_file(
                sample=sample,
                fp=fp,
                fcs=fcs,
                gater=gater,
                fitted=fitted,
                marker_names=marker_names,
                label_to_marker=label_to_marker,
                marker_to_pop=marker_to_pop,
                gate_options=gate_options,
                cutoff_by_gate=cutoff_by_gate,
                file_metrics_by_key=file_metrics_by_key,
                sample_combined_metrics_by_name=sample_combined_metrics_by_name,
                max_points=max_points,
                rng=rng,
            )

            plot_cache[key] = entry

            if progress_cb is not None:
                progress_cb(
                    {
                        "stage": "plot_cache",
                        "sample_name": sample.name,
                        "role": sample.role,
                        "file_name": getattr(
                            fcs,
                            "original_filename",
                            str(fp),
                        ),
                        "file_path": fp,
                    }
                )

    return payload, plot_cache

def _as_bool_mask(values, fallback_len: int, fallback: bool = True) -> np.ndarray:
    if values is None:
        return np.full(fallback_len, fallback, dtype=bool)

    arr = np.asarray(values, dtype=bool)

    if arr.shape[0] != fallback_len:
        return np.full(fallback_len, fallback, dtype=bool)

    return arr


def _mask_for_final_gate(entry: Dict[str, Any], gate: str, n: int) -> np.ndarray:
    """
    Return the event mask used for final IgG display for one cached file.

    The plot cache stores downsampled values only, so this mask works on the
    downsampled IgG vector.
    """
    gate = (gate or "").strip()

    if gate in ("", "All Cells"):
        return _as_bool_mask(entry.get("mask_all"), n, fallback=True)

    if gate == "Singlets":
        mask_sing = entry.get("mask_sing")
        if mask_sing is not None:
            return _as_bool_mask(mask_sing, n, fallback=True)
        return _as_bool_mask(entry.get("mask_all"), n, fallback=True)

    if gate == "Lymphocytes":
        return _as_bool_mask(entry.get("mask_lymph"), n, fallback=True)

    marker_pos = (entry.get("marker_pos", {}) or {}).get(gate)
    if marker_pos is not None:
        return _as_bool_mask(marker_pos, n, fallback=False)

    return _as_bool_mask(entry.get("mask_lymph"), n, fallback=True)


def _sample_line_values(values: np.ndarray, max_values: int) -> List[float]:
    values = values[np.isfinite(values)]

    if values.size <= max_values:
        return values.astype(float).tolist()

    idx = np.linspace(0, values.size - 1, max_values).astype(int)
    return values[idx].astype(float).tolist()


def _short_file_label(entry: Dict[str, Any]) -> str:
    raw = entry.get("file_key_raw") or entry.get("file_key") or ""
    name = Path(str(raw)).name
    return name or "file"


def _build_single_file_line_series(
    *,
    entry: Dict[str, Any],
    gate: str,
    label: str,
    color: str,
    cutoff: float,
    max_line_values: int,
) -> Dict[str, Any]:

    # Curves use transformed IgG; cards/report can still use raw IgG.
    igg_values = entry.get("igg", [])
    igg_raw_values = entry.get("igg_raw")

    if igg_raw_values is None:
        igg_raw_values = igg_values

    igg = np.asarray(igg_values, dtype=float)
    igg_raw = np.asarray(igg_raw_values, dtype=float)

    n = int(min(igg.shape[0], igg_raw.shape[0]))
    igg = igg[:n]
    igg_raw = igg_raw[:n]

    # Use the same event gate as the final plot.
    mask = _mask_for_final_gate(entry, gate, n)

    if mask.shape[0] != n:
        mask = np.full(n, True, dtype=bool)

    selected = igg[mask]
    selected_raw = igg_raw[mask]

    selected = selected[np.isfinite(selected)]
    selected_raw = selected_raw[np.isfinite(selected_raw)]

    # Use the same event gate as the final plot.
    mask = _mask_for_final_gate(entry, gate, n)

    if mask.shape[0] != n:
        mask = np.full(n, True, dtype=bool)

    selected = igg[mask]
    selected_raw = igg_raw[mask]

    selected = selected[np.isfinite(selected)]
    selected_raw = selected_raw[np.isfinite(selected_raw)]

    # Positivity should NOT be recomputed from raw IgG with the transformed cutoff.
    # Use the precomputed transformed-space positivity mask for this gate.
    igg_pos = (entry.get("igg_pos_by_gate", {}) or {}).get(gate)
    if igg_pos is not None:
      pos_mask = np.asarray(igg_pos, dtype=bool)
      if pos_mask.shape[0] == mask.shape[0]:
          gated_pos = pos_mask[mask]
          n_pos = int(np.sum(gated_pos))
      else:
          n_pos = 0
    else:
      n_pos = 0

    n_total = int(selected.size)
    pos_pct = float((n_pos / n_total) * 100.0) if n_total > 0 else 0.0

    values = _sample_line_values(selected, max_line_values)
    values_raw = _sample_line_values(selected_raw, max_line_values)

    raw_median = (
        float(np.median(selected_raw))
        if selected_raw.size > 0
        else None
    )

    transformed_median = (
        float(np.median(selected))
        if selected.size > 0
        else None
    )

    return {
        "label": label,
        "color": color,

        # Main curve values are transformed IgG.
        "values": values,
        "value_scale": "asinh",
        "x_label": "asinh IgG",

        # Raw values for cards and report.
        "values_raw": values_raw,
        "raw_median": raw_median,
        "transformed_median": transformed_median,

        "n_total": n_total,
        "n_pos": n_pos,
        "pos_pct": pos_pct,

        "filename": _short_file_label(entry),
        "sample_name": str(entry.get("sample_name") or ""),
        "role": str(entry.get("role") or ""),
    }

def _build_control_file_line_series(
    *,
    plot_cache: Dict[str, Any],
    gate: str,
    role: str,
    role_label: str,
    colors: List[str],
    cutoff: float,
    max_line_values: int,
) -> List[Dict[str, Any]]:
    """
    Build one histogram curve per control file.

    This is intentionally file-level. Do not combine controls here, because
    combined NC/PC curves hide replicate-level shifts.
    """
    out: List[Dict[str, Any]] = []

    entries = [
        entry
        for entry in plot_cache.values()
        if str(entry.get("role", "")).upper() == role.upper()
    ]

    entries.sort(
        key=lambda e: (
            str(e.get("sample_name", "")),
            _short_file_label(e),
        )
    )

    for idx, entry in enumerate(entries):
        file_label = _short_file_label(entry)

        out.append(
            _build_single_file_line_series(
                entry=entry,
                gate=gate,
                label=f"{role_label} · {_short_file_label(entry)}",
                color=colors[idx % len(colors)],
                cutoff=cutoff,
                max_line_values=max_line_values,
            )
        )

    return out

def build_results_response_from_cache(
    plot_cache: Dict[str, Any],
    selected_key: str,
    selected_gate: str,
    max_points_final: int = 2000,
    max_line_values: int = 2000,
) -> Dict[str, Any]:
    """
    Returns:
      - gating_plots: full gating strategy for selected file
      - final_scatter_series + line_series: overlay for selected end-gate
      - selected_file_metrics / selected_sample_metrics for the selected gate
    """
    entry = plot_cache[selected_key]
    gate_options = list(entry.get("gate_options", []))

    gate = (selected_gate or "").strip()
    if not gate:
        gate = gate_options[0] if gate_options else "All Cells"

    cutoff_by_gate = entry.get("cutoff_by_gate", {}) or {}
    cutoff = float(cutoff_by_gate.get(gate, 0.0))

    selected_file_metrics_by_gate = (
        entry.get("selected_file_metrics_by_gate", {}) or {}
    )
    sample_combined_metrics_by_gate = (
        entry.get("sample_combined_metrics_by_gate", {}) or {}
    )

    selected_file_metrics = selected_file_metrics_by_gate.get(gate)
    selected_sample_metrics = sample_combined_metrics_by_gate.get(gate)

    gating_plots: List[Dict[str, Any]] = []

    fsc_a = entry.get("fsc_a")
    fsc_h = entry.get("fsc_h")
    ssc_a = entry.get("ssc_a")

    mask_all = np.asarray(
        entry.get("mask_all", [True] * len(entry.get("igg", []))),
        dtype=bool,
    )
    mask_sing = entry.get("mask_sing")

    mask_lymph = np.asarray(
        entry.get("mask_lymph", [True] * len(entry.get("igg", []))),
        dtype=bool,
    )
    mask_lymph_raw = np.asarray(
        entry.get("mask_lymph_raw", mask_lymph.tolist()),
        dtype=bool,
    )

    if fsc_a is not None and fsc_h is not None and mask_sing is not None:
        x = np.asarray(fsc_a, dtype=float)
        y = np.asarray(fsc_h, dtype=float)
        ms = np.asarray(mask_sing, dtype=bool)

        gating_plots.append(
            {
                "title": "Singlets (/All Cells)",
                "x_label": "FSC-A",
                "y_label": "FSC-H",
                "points": simpoints(x[mask_all], y[mask_all], ms[mask_all]),
            }
        )

    if fsc_a is not None and ssc_a is not None:
        x = np.asarray(fsc_a, dtype=float)
        y = np.asarray(ssc_a, dtype=float)

        base = mask_all.copy()
        if mask_sing is not None:
            base = np.asarray(mask_sing, dtype=bool)

        parent_gate = "Singlets" if mask_sing is not None else "All Cells"

        gating_plots.append(
            {
                "title": f"Lymphocytes (/{parent_gate})",
                "x_label": "FSC-A",
                "y_label": "SSC-A",
                "points": simpoints(
                    x[base],
                    y[base],
                    mask_lymph_raw[base],
                ),
            }
        )

    pop_to_marker = entry.get("pop_to_marker", {}) or {}

    for pop in gate_options:
        if pop in CORE_GATES:
            continue

        marker_values = entry.get("marker_vals", {}).get(pop)
        marker_pos = entry.get("marker_pos", {}).get(pop)

        if marker_values is None or marker_pos is None:
            continue

        marker_values = np.asarray(marker_values, dtype=float)
        marker_pos = np.asarray(marker_pos, dtype=bool)

        y_base = (
            np.asarray(ssc_a, dtype=float)
            if ssc_a is not None
            else np.zeros_like(marker_values)
        )

        gating_plots.append(
            {
                "title": f"{pop} (/Lymphocytes)",
                "x_label": population_x_label(pop, pop_to_marker),
                "y_label": "SSC-A" if ssc_a is not None else "0",
                "points": simpoints(
                    marker_values[mask_lymph],
                    y_base[mask_lymph],
                    marker_pos[mask_lymph],
                ),
            }
        )

    sc_nc, _ln_nc_combined = collect_plot_series(
        plot_cache=plot_cache,
        selected_key=selected_key,
        gate=gate,
        role="NC",
        label="Negative control",
        color="#22c55e",
        only_selected=False,
        max_points_final=max_points_final,
        max_line_values=max_line_values,
    )

    sc_pc, _ln_pc_combined = collect_plot_series(
        plot_cache=plot_cache,
        selected_key=selected_key,
        gate=gate,
        role="PC",
        label="Positive control",
        color="#ef4444",
        only_selected=False,
        max_points_final=max_points_final,
        max_line_values=max_line_values,
    )

    sc_sel, ln_sel = collect_plot_series(
        plot_cache=plot_cache,
        selected_key=selected_key,
        gate=gate,
        role="__SEL__",
        label="Selected file",
        color="#3b82f6",
        only_selected=True,
        max_points_final=max_points_final,
        max_line_values=max_line_values,
    )

    negative_control_lines = _build_control_file_line_series(
        plot_cache=plot_cache,
        gate=gate,
        role="NC",
        role_label="Negative control",
        colors=[
            "#16a34a",
            "#22c55e",
            "#15803d",
            "#86efac",
            "#166534",
        ],
        cutoff=cutoff,
        max_line_values=max_line_values,
    )

    positive_control_lines = _build_control_file_line_series(
        plot_cache=plot_cache,
        gate=gate,
        role="PC",
        role_label="Positive control",
        colors=[
            "#dc2626",
            "#ef4444",
            "#b91c1c",
            "#fca5a5",
            "#991b1b",
        ],
        cutoff=cutoff,
        max_line_values=max_line_values,
    )

    selected_entry = plot_cache[selected_key]
    selected_role = str(selected_entry.get("role", "")).upper()

    line_series = [
        *negative_control_lines,
        *positive_control_lines,
    ]

    # Avoid drawing the same control file twice when the selected file itself is NC or PC.
    if selected_role not in {"NC", "PC"}:
        line_series.append(ln_sel)

    return {
        "gate_options": gate_options,
        "selected_gate": gate,
        "gating_plots": gating_plots,
        "final_scatter_series": [sc_nc, sc_pc, sc_sel],
        "line_series": line_series,
        "cutoff": cutoff,
        "selected_file_metrics": selected_file_metrics,
        "selected_sample_metrics": selected_sample_metrics,
    }
