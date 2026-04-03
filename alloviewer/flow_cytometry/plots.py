from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .gating.gater import FittedGater

import numpy as np


# -------------------------
# utilities
# -------------------------

def _norm_path(p: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(p).resolve())))
    except Exception:
        return os.path.normcase(os.path.normpath(str(p)))


def _sanitize(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _as_list_finite(a: np.ndarray) -> List[float]:
    return _sanitize(a).tolist()


def _downsample_idx(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return rng.choice(n, size=max_points, replace=False).astype(int)


def _simpoints(x: np.ndarray, y: np.ndarray, ing: np.ndarray) -> List[Dict[str, Any]]:
    x2 = _sanitize(x)
    y2 = _sanitize(y)
    g2 = ing.astype(bool, copy=False)
    return [{"x": float(xx), "y": float(yy), "inGate": bool(gg)} for xx, yy, gg in zip(x2, y2, g2)]


def _population_result_to_dict(p, display_label_fn) -> Dict[str, Any]:
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


def _gate_mask_ds(entry: Dict[str, Any], gate: str) -> np.ndarray:
    """
    Downsampled stepwise mask for the selected end-gate.
    gate options include: All Cells, Singlets, Lymphocytes, and each population label.

    Semantics:
      - mask_all       = edge-only
      - mask_sing      = edge+singlets (if available)
      - mask_lymph_raw = pure lymph gate BEFORE cluster cleanup (used for gating plot)
      - mask_lymph     = lymphocytes AFTER cluster cleanup (outliers + debris clusters removed)
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

    mp = entry.get("marker_pos", {}).get(gate)
    if mp is None:
        mtp = entry.get("marker_to_pop", {}) or {}
        g2 = (mtp.get(gate) or "").strip()
        if g2:
            mp = entry.get("marker_pos", {}).get(g2)

    if mp is None:
        return mask_lymph

    return np.asarray(mp, dtype=bool)


# -------------------------
# PUBLIC API (keep names / args stable)
# -------------------------

def make_results_payload(
    ds,
    gater,
    fitted: FittedGater,
    results,
    marker_to_population: Dict[str, str],
    max_points: int = 20_000,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    This module only:
      - builds JSON payloads for the frontend
      - downsamples arrays into plot_cache

    IMPORTANT PERFORMANCE RULE:
      - Do NOT run clustering prediction here.
      - Use gater.analyze_file_cached(fcs, fitted) to reuse per-file results
        (QC + lymph + clustering prediction).
    """

    rng = np.random.default_rng(seed)

    # --- build frontend labels ---
    marker_names = list((fitted.panel.markers or {}).keys())
    label_to_marker: Dict[str, str] = {}
    pop_labels: List[str] = []
    seen = set()
    for m in marker_names:
        lbl = (marker_to_population.get(m) or "").strip() or m
        if lbl in seen:
            lbl = f"{lbl} ({m})"
        seen.add(lbl)
        pop_labels.append(lbl)
        label_to_marker[lbl] = m

    can_singlets = bool(fitted.panel.fsc_a and fitted.panel.fsc_h)

    gate_options: List[str] = ["All Cells"]
    if can_singlets:
        gate_options.append("Singlets")
    gate_options.append("Lymphocytes")
    gate_options.extend(pop_labels)

    # --- remap summary results labels (marker name -> population label) ---
    marker_to_label: Dict[str, str] = {v: k for k, v in label_to_marker.items()}

    def _display_label(lbl: str) -> str:
        if lbl in ("All Cells", "Singlets", "Lymphocytes"):
            return lbl
        return marker_to_label.get(lbl, lbl)

    payload: Dict[str, Any] = {
        "ok": True,
        "results": [
            {
                "sample_name": r.sample_name,
                "role": r.role,
                "n_files": r.n_files,
                "combined": [
                    _population_result_to_dict(p, _display_label)
                    for p in r.combined
                ],
                "per_file": [
                    {
                        "file_name": fr.file_name,
                        "populations": [
                            _population_result_to_dict(p, _display_label)
                            for p in fr.populations
                        ],
                    }
                    for fr in r.per_file
                ],
            }
            for r in results
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

    # --- build cutoff_by_gate in frontend label space ---
    cutoff_by_gate: Dict[str, float] = {}
    for g in gate_options:
        if g in ("All Cells", "Singlets", "Lymphocytes"):
            cutoff_by_gate[g] = float(fitted.igg_cutoff_by_gate.get(g, 0.0))
        else:
            m = label_to_marker.get(g)
            cutoff_by_gate[g] = float(fitted.igg_cutoff_by_gate.get(m or "", 0.0))

    payload["panel_used"]["cutoff_by_gate"] = dict(cutoff_by_gate)

    # --- metrics lookup maps for plot_cache ---
    sample_combined_metrics_by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    file_metrics_by_key: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ds_sample, sample_result in zip(ds.samples, results):
        sample_combined_metrics_by_name[ds_sample.name] = {
            _display_label(p.label): _population_result_to_dict(p, _display_label)
            for p in sample_result.combined
        }

        for fp, file_result in zip(ds_sample.file_paths, sample_result.per_file):
            file_metrics_by_key[_norm_path(fp)] = {
                _display_label(p.label): _population_result_to_dict(p, _display_label)
                for p in file_result.populations
            }

    # --- plot_cache per file ---
    plot_cache: Dict[str, Any] = {}

    if not hasattr(gater, "analyze_file_cached"):
        raise AttributeError("Gater must implement analyze_file_cached(fcs, fitted) for plots caching.")

    for s in ds.samples:
        for fp, fcs in zip(s.file_paths, s.files):
            fa = gater.analyze_file_cached(fcs, fitted)

            events = fa.events
            if events is None:
                continue

            n = int(events.shape[0])
            idx = _downsample_idx(n, max_points=max_points, rng=rng)
            n_ds = int(idx.size)

            # -------------------------
            # 1) Scatter (RAW) and IgG (transformed) for plotting
            # -------------------------
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

            # -------------------------
            # 2) Downsample masks (from cached analysis)
            # -------------------------
            mask_all_ds = np.asarray(fa.mask_edge, dtype=bool)[idx].copy()

            if fa.mask_sing is not None:
                mask_sing_ds = np.asarray(fa.mask_sing, dtype=bool)[idx].copy().tolist()
            else:
                mask_sing_ds = None

            mask_lymph_ds_raw = np.asarray(fa.mask_lymph_raw, dtype=bool)[idx].copy()
            mask_lymph_ds_clean = np.asarray(fa.mask_lymph, dtype=bool)[idx].copy()

            m_by_marker_full = fa.m_by_marker or {}
            m_by_marker_ds: Dict[str, np.ndarray] = {
                mn: (np.asarray(m_by_marker_full.get(mn, np.zeros(n, dtype=bool)), dtype=bool)[idx].copy())
                for mn in marker_names
            }

            marker_pos_by_label_ds: Dict[str, np.ndarray] = {}
            for pop_label, marker_name in label_to_marker.items():
                marker_pos_by_label_ds[pop_label] = m_by_marker_ds.get(marker_name, np.zeros(n_ds, dtype=bool))

            marker_vals_ds: Dict[str, List[float]] = {}
            for pop_label, marker_name in label_to_marker.items():
                ch = fitted.panel.markers[marker_name]
                j = fcs.get_channel_index(ch)
                cof = float(getattr(fitted, "marker_cofactors", {}).get(marker_name, fitted.config.transform.igg_cofactor))
                v = gater.transform_channel(events[:, j].astype(np.float32, copy=False), cofactor=cof)
                marker_vals_ds[pop_label] = _as_list_finite(v[idx])

            igg_pos_by_gate: Dict[str, List[bool]] = {}
            for g in gate_options:
                c = float(cutoff_by_gate.get(g, 0.0))
                igg_pos_by_gate[g] = (igg_ds > c).tolist()

            key = _norm_path(fp)
            marker_to_pop = {marker: pop for pop, marker in label_to_marker.items()}

            for mn in marker_names:
                mm = np.asarray(fa.m_by_marker.get(mn, np.zeros(n, dtype=bool)), dtype=bool)
                if np.any(mm & ~np.asarray(fa.mask_lymph, dtype=bool)):
                    raise RuntimeError(f"{fp}: marker mask {mn} leaks outside cleaned lymph")

            plot_cache[key] = {
                "file_key": key,
                "file_key_raw": fp,
                "sample_name": s.name,
                "role": s.role,
                "gate_options": gate_options,
                "cutoff_by_gate": dict(cutoff_by_gate),

                # downsampled scatter and IgG
                "fsc_a": _as_list_finite(fsc_a[idx]) if fsc_a is not None else None,
                "fsc_h": _as_list_finite(fsc_h[idx]) if fsc_h is not None else None,
                "ssc_a": _as_list_finite(ssc_a[idx]) if ssc_a is not None else None,
                "igg": _as_list_finite(igg_ds),

                # downsampled masks
                "mask_all": mask_all_ds.tolist(),
                "mask_sing": mask_sing_ds,
                "mask_lymph_raw": mask_lymph_ds_raw.tolist(),
                "mask_lymph": mask_lymph_ds_clean.tolist(),

                # populations
                "marker_pos": {k: marker_pos_by_label_ds[k].tolist() for k in marker_pos_by_label_ds},
                "marker_vals": marker_vals_ds,

                "pop_to_marker": dict(label_to_marker),
                "marker_to_pop": dict(marker_to_pop),

                # IgG positivity
                "igg_pos_by_gate": dict(igg_pos_by_gate),

                # metrics for selected gate display
                "selected_file_metrics_by_gate": file_metrics_by_key.get(key, {}),
                "sample_combined_metrics_by_gate": sample_combined_metrics_by_name.get(s.name, {}),
            }

    return payload, plot_cache


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

    selected_file_metrics_by_gate = entry.get("selected_file_metrics_by_gate", {}) or {}
    sample_combined_metrics_by_gate = entry.get("sample_combined_metrics_by_gate", {}) or {}

    selected_file_metrics = selected_file_metrics_by_gate.get(gate)
    selected_sample_metrics = sample_combined_metrics_by_gate.get(gate)

    gating_plots: List[Dict[str, Any]] = []

    fsc_a = entry.get("fsc_a")
    fsc_h = entry.get("fsc_h")
    ssc_a = entry.get("ssc_a")

    mask_all = np.asarray(entry.get("mask_all", [True] * len(entry.get("igg", []))), dtype=bool)
    mask_sing = entry.get("mask_sing")

    mask_lymph = np.asarray(entry.get("mask_lymph", [True] * len(entry.get("igg", []))), dtype=bool)
    mask_lymph_raw = np.asarray(entry.get("mask_lymph_raw", mask_lymph.tolist()), dtype=bool)

    if fsc_a is not None and fsc_h is not None and mask_sing is not None:
        x = np.asarray(fsc_a, dtype=float)
        y = np.asarray(fsc_h, dtype=float)
        ms = np.asarray(mask_sing, dtype=bool)
        gating_plots.append(
            {
                "title": "Singlets (/All Cells)",
                "x_label": "FSC-A",
                "y_label": "FSC-H",
                "points": _simpoints(x[mask_all], y[mask_all], ms[mask_all]),
            }
        )

    if fsc_a is not None and ssc_a is not None:
        x = np.asarray(fsc_a, dtype=float)
        y = np.asarray(ssc_a, dtype=float)

        base = mask_all.copy()
        if mask_sing is not None:
            base = np.asarray(mask_sing, dtype=bool)

        gating_plots.append(
            {
                "title": "Lymphocytes (/" + ("Singlets" if mask_sing is not None else "All Cells") + ")",
                "x_label": "FSC-A",
                "y_label": "SSC-A",
                "points": _simpoints(x[base], y[base], mask_lymph_raw[base]),
            }
        )

    pop_to_marker = entry.get("pop_to_marker", {}) or {}

    def _pop_xlabel(pop_label: str) -> str:
        pop_label = (pop_label or "").strip()
        marker = (pop_to_marker.get(pop_label) or "").strip()
        if not marker:
            return f"{pop_label} marker" if pop_label else "marker"
        if not pop_label or pop_label == marker:
            return marker
        return f"{marker} ({pop_label})"

    for pop in gate_options:
        if pop in ("All Cells", "Singlets", "Lymphocytes"):
            continue
        mv = entry.get("marker_vals", {}).get(pop)
        mp = entry.get("marker_pos", {}).get(pop)
        if mv is None or mp is None:
            continue

        mv = np.asarray(mv, dtype=float)
        mp = np.asarray(mp, dtype=bool)

        y_base = np.asarray(ssc_a, dtype=float) if ssc_a is not None else np.zeros_like(mv)
        gating_plots.append(
            {
                "title": f"{pop} (/Lymphocytes)",
                "x_label": _pop_xlabel(pop),
                "y_label": "SSC-A" if ssc_a is not None else "0",
                "points": _simpoints(mv[mask_lymph], y_base[mask_lymph], mp[mask_lymph]),
            }
        )

    def collect_series(
        role: str,
        label: str,
        color: str,
        only_selected: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        keys: List[str] = []
        if only_selected:
            keys = [selected_key]
        else:
            for k, e in plot_cache.items():
                if e.get("role") == role:
                    keys.append(k)

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        pos: List[np.ndarray] = []

        for k in keys:
            e = plot_cache[k]
            m = _gate_mask_ds(e, gate)

            igg_k = np.asarray(e["igg"], dtype=float)
            ssc_k = (
                np.asarray(e["ssc_a"], dtype=float)
                if e.get("ssc_a") is not None
                else np.zeros_like(igg_k)
            )
            pos_k = np.asarray(
                (e.get("igg_pos_by_gate", {}) or {}).get(gate, [False] * len(igg_k)),
                dtype=bool,
            )

            xs.append(igg_k[m])
            ys.append(ssc_k[m])
            pos.append(pos_k[m])

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
            rr = np.random.default_rng(0)
            take = rr.choice(n_total, size=max_points_final, replace=False)
            x = x[take]
            y = y[take]
            p = p[take]

        line_vals = x
        if line_vals.shape[0] > max_line_values:
            rr = np.random.default_rng(1)
            q = np.quantile(line_vals, 0.99)
            tail = line_vals[line_vals >= q]
            rest = line_vals[line_vals < q]
            need = max(0, max_line_values - tail.shape[0])
            if rest.shape[0] > need:
                pick = rr.choice(rest.shape[0], size=need, replace=False)
                rest = rest[pick]
            line_vals = np.concatenate([rest, tail])

        scatter = {
            "label": label,
            "color": color,
            "points": _simpoints(x, y, p),
            "n_total": n_total,
            "n_pos": n_pos,
            "pos_pct": float(pos_pct),
        }
        line = {
            "label": label,
            "color": color,
            "values": _as_list_finite(line_vals),
            "n_total": n_total,
            "n_pos": n_pos,
            "pos_pct": float(pos_pct),
        }
        return scatter, line

    sc_nc, ln_nc = collect_series("NC", "Negative control", "#22c55e", only_selected=False)
    sc_pc, ln_pc = collect_series("PC", "Positive control", "#ef4444", only_selected=False)
    sc_sel, ln_sel = collect_series("__SEL__", "Selected file", "#3b82f6", only_selected=True)

    return {
        "gate_options": gate_options,
        "selected_gate": gate,
        "gating_plots": gating_plots,
        "final_scatter_series": [sc_nc, sc_pc, sc_sel],
        "line_series": [ln_nc, ln_pc, ln_sel],
        "cutoff": cutoff,
        "selected_file_metrics": selected_file_metrics,
        "selected_sample_metrics": selected_sample_metrics,
    }
