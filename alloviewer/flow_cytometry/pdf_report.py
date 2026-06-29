from __future__ import annotations

import io
import os
import textwrap
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    NextPageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.platypus.doctemplate import ActionFlowable

from .scoring import ScoreRule, RatioRule
from .plots import build_results_response_from_cache


# Gating dot plots (match frontend defaults)
PLOT_POINT_COLOR = "#64748b"  # out-of-gate fill
PLOT_GATE_COLOR = "#22c55e"   # in-gate fill
PLOT_POINT_STROKE_COLOR = "#0f172a"
PLOT_POINT_STROKE_WIDTH = 0.35

PLOT_AXIS_COLOR = "#94a3b8"
PLOT_TEXT_COLOR = "#64748b"

# Line plots (match app colors)
PLOT_LINE_WIDTH = 1.8
PLOT_CUTOFF_COLOR = "#ef4444"
PLOT_LINE_NC_COLOR = "#22c55e"
PLOT_LINE_PC_COLOR = "#ef4444"
PLOT_LINE_SELECTED_COLOR = "#3b82f6"
PLOT_LINE_OTHER_COLOR = "#111827"

# Subsampling to reduce clutter (gating plots)
MAX_EVENTS_PER_GATING_PLOT = 12_000
GATING_SUBSAMPLE_SEED = 0

# Page size mode: "SAFE" fits within both A4 and Letter.
PAGE_SIZE_MODE_DEFAULT = "SAFE"  # "SAFE" | "A4" | "LETTER"


@dataclass
class ReportMeta:
    job_id: str = ""
    positivity_metric: Optional[str] = None
    positivity_threshold: Optional[float] = None


METRIC_ORDER: List[Tuple[str, str, str]] = [
    ("frac_pos", "frac_pos", "frac_pos (rel. to NC)"),
    ("median_ratio", "median ratio", "median ratio (rel. to NC)"),
    ("median_shift", "median shift", "median shift (rel. to NC)"),
    ("fluorescence_index", "fluorescence index", "fluorescence index (rel. to NC)"),
]

METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "frac_pos": (
        "frac_pos",
        "fracPos",
        "fraction_positive",
        "fraction_pos",
        "positive_fraction",
        "pos_frac",
        "igg_pos_frac",
        "igg_positive_fraction",
        "pos_pct",
        "igg_pos_pct",
        "igg_pos_fraction",
    ),
    "median_ratio": (
        "median_ratio",
        "medianRatio",
        "median ratio",
        "mfi_ratio",
        "mfiRatio",
        "ratio",
        "median_ratio_rel_to_nc",
        "mfi_ratio_s_nc",
        "igg_median_ratio",
    ),
    "median_shift": (
        "median_shift",
        "medianShift",
        "median shift",
        "mfi_shift",
        "mfiShift",
        "shift",
        "median_delta",
        "delta_median",
        "igg_median_shift",
    ),
    "fluorescence_index": (
        "fluorescence_index",
        "fluorescenceIndex",
        "fluorescence index",
        "fi",
        "FI",
        "stain_index",
        "stainIndex",
        "stimulation_index",
        "stimulationIndex",
        "igg_fluorescence_index",
    ),
}

METRIC_DISPLAY = {key: display for key, display, _header in METRIC_ORDER}


def _safe_pagesizes() -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (portrait, landscape) that fits within both A4 and Letter."""
    a4w, a4h = A4
    lw, lh = LETTER
    portrait = (min(a4w, lw), min(a4h, lh))
    landscape = (portrait[1], portrait[0])
    return portrait, landscape


def _pagesizes(mode: str) -> tuple[tuple[float, float], tuple[float, float]]:
    m = (mode or PAGE_SIZE_MODE_DEFAULT).upper()
    if m == "A4":
        return (A4, (A4[1], A4[0]))
    if m == "LETTER":
        return (LETTER, (LETTER[1], LETTER[0]))
    return _safe_pagesizes()


def _fig_to_png_bytes(
    fig: plt.Figure,
    dpi: int = 150,
    *,
    tight: bool = True,
) -> bytes:
    if tight:
        try:
            fig.tight_layout(pad=0.4)
        except Exception:
            pass
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def _basename(path_or_name: str) -> str:
    return os.path.basename(str(path_or_name or ""))


def _build_file_key_map(plot_cache: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, e in (plot_cache or {}).items():
        raw = e.get("file_key_raw") or e.get("file_key") or k
        out[_basename(raw)] = k
    return out


def _safe_decode_meta(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return value.decode(enc).strip()
            except Exception:
                pass
        return value.decode(errors="ignore").strip()

    return str(value).strip()


def _normalize_meta_key(key: Any) -> str:
    return str(key).strip().lower().replace("_", " ").replace("-", " ")


def _extract_tube_name_from_meta(meta: Dict[str, Any], fallback: str) -> str:
    if not meta:
        return fallback

    normalized = {
        _normalize_meta_key(k): _safe_decode_meta(v)
        for k, v in meta.items()
        if _safe_decode_meta(v)
    }

    candidate_keys = [
        "$tube name",
        "tube name",
        "$tubename",
        "tubename",
        "tube",
        "tube name:",
        "sample name",
        "$sample",
        "sample",
        "$src",
        "src",
        "name",
    ]

    for key in candidate_keys:
        value = normalized.get(_normalize_meta_key(key))
        if value:
            return value

    return fallback


def _metadata_from_path(path_like: Any) -> Dict[str, Any]:
    try:
        path = os.fspath(path_like)
        if not path or not os.path.exists(path) or not str(path).lower().endswith(".fcs"):
            return {}
        from flowio import FlowData
        fd = FlowData(str(path))
        return dict(getattr(fd, "text", {}) or {})
    except Exception:
        return {}


def _tube_name_from_entry(entry: Dict[str, Any], *, fallback: str) -> str:
    for key in ("tube_name", "display_name", "fcs_display_name"):
        value = str(entry.get(key) or "").strip()
        if value:
            return value

    for key in ("file_key_raw", "file_key"):
        value = entry.get(key)
        if not value:
            continue
        meta = _metadata_from_path(value)
        tube = _extract_tube_name_from_meta(meta, fallback)
        if tube != fallback:
            return tube

    return fallback


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, dict):
        for key in ("value", "score", "metric", "raw", "rel_to_nc"):
            if key in value:
                return _safe_float(value.get(key))
        return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _fmt_num(x: Any, nd: int = 2) -> str:
    v = _safe_float(x)
    return "-" if v != v else f"{v:.{nd}f}"


def _fmt_threshold(x: Any) -> str:
    v = _safe_float(x)
    if v != v:
        return "-"
    return f"{v:g}"


def _median_mfi(values: Any) -> float:
    v = np.asarray(values if values is not None else [], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(v))


def _ratio_value(mfi_sample: float, mfi_nc: float) -> float:
    if (mfi_sample != mfi_sample) or (mfi_nc != mfi_nc) or float(mfi_nc) == 0.0:
        return float("nan")
    r = float(mfi_sample) / float(mfi_nc)
    if not np.isfinite(r):
        return float("nan")
    return r




def _raw_cutoff_from_transformed(cutoff_t: float, cofactor: float = 150.0) -> float:
    """Invert the asinh(raw/cofactor) transform used for IgG display cutoffs."""
    try:
        c = float(cutoff_t)
        k = float(cofactor)
    except Exception:
        return float("nan")
    if not np.isfinite(c) or not np.isfinite(k) or k <= 0:
        return float("nan")
    return float(np.sinh(c) * k)


def _cell_par(text: str, style: ParagraphStyle) -> Paragraph:
    safe = (str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    return Paragraph(safe, style)


def _normalize_metric_key(metric: Optional[str]) -> Optional[str]:
    if not metric:
        return None
    raw = str(metric).strip()
    low = raw.casefold().replace("-", "_").replace(" ", "_")

    for canonical, aliases in METRIC_ALIASES.items():
        if raw in aliases:
            return canonical
        if low == canonical.casefold():
            return canonical
        for alias in aliases:
            alias_low = str(alias).casefold().replace("-", "_").replace(" ", "_")
            if low == alias_low:
                return canonical
    return low


def _metric_from_metrics(metrics: Any, canonical: str) -> float:
    if not isinstance(metrics, dict):
        return float("nan")

    aliases = METRIC_ALIASES.get(canonical, (canonical,))

    for key in aliases:
        if key in metrics:
            return _safe_float(metrics.get(key))

    # Case-insensitive fallback. This keeps the report robust to frontend-style keys.
    wanted = {
        str(key).casefold().replace("-", "_").replace(" ", "_")
        for key in aliases
    }
    for key, value in metrics.items():
        key_norm = str(key).casefold().replace("-", "_").replace(" ", "_")
        if key_norm in wanted:
            return _safe_float(value)

    return float("nan")




def _value_from_aliases(metrics: Any, aliases: Tuple[str, ...]) -> float:
    if not isinstance(metrics, dict):
        return float("nan")

    for key in aliases:
        if key in metrics:
            return _safe_float(metrics.get(key))

    wanted = {
        str(key).casefold().replace("-", "_").replace(" ", "_")
        for key in aliases
    }
    for key, value in metrics.items():
        key_norm = str(key).casefold().replace("-", "_").replace(" ", "_")
        if key_norm in wanted:
            return _safe_float(value)

    return float("nan")


def _mfi_values_for_row(
    *,
    rr: Dict[str, Any],
    line_series: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Return raw MFI values: NC, PC, selected file."""
    metrics = rr.get("selected_file_metrics")

    mfi_sel = _value_from_aliases(metrics, ("igg_median_raw", "mfi_raw", "median_raw", "median_mfi_raw"))
    mfi_nc = _value_from_aliases(metrics, ("igg_nc_median_raw", "nc_median_raw", "mfi_nc_raw"))
    mfi_pc = _value_from_aliases(metrics, ("igg_pc_median_raw", "pc_median_raw", "mfi_pc_raw"))

    c_nc, c_pc, c_sel = _compute_mfi_triplet(line_series)

    if mfi_nc != mfi_nc:
        mfi_nc = c_nc
    if mfi_pc != mfi_pc:
        mfi_pc = c_pc
    if mfi_sel != mfi_sel:
        mfi_sel = c_sel

    return mfi_nc, mfi_pc, mfi_sel

def _selected_pos_pct(line_series: List[Dict[str, Any]]) -> float:
    for s in line_series:
        if str(s.get("label", "")).casefold() == "selected file":
            return _safe_float(s.get("pos_pct"))
    return float("nan")


def _metric_values_for_row(
    *,
    rr: Dict[str, Any],
    line_series: List[Dict[str, Any]],
    mfi_nc: float,
    mfi_sel: float,
) -> Dict[str, float]:
    metrics = rr.get("selected_file_metrics")

    values = {
        canonical: _metric_from_metrics(metrics, canonical)
        for canonical, _display, _header in METRIC_ORDER
    }

    # Conservative fallbacks for values that can be derived unambiguously here.
    if values["frac_pos"] != values["frac_pos"]:
        values["frac_pos"] = _selected_pos_pct(line_series)

    if values["median_ratio"] != values["median_ratio"]:
        values["median_ratio"] = _ratio_value(mfi_sel, mfi_nc)

    if values["median_shift"] != values["median_shift"]:
        if mfi_sel == mfi_sel and mfi_nc == mfi_nc:
            values["median_shift"] = float(mfi_sel - mfi_nc)

    return values


def _score_text(metric_values: Dict[str, float], meta: ReportMeta) -> str:
    metric_key = _normalize_metric_key(meta.positivity_metric)
    threshold = _safe_float(meta.positivity_threshold)

    if not metric_key or threshold != threshold:
        return "-"

    value = metric_values.get(metric_key, float("nan"))
    if value != value:
        return "-"

    return "Positive" if float(value) > float(threshold) else "Negative"


def _score_rule_text(meta: ReportMeta) -> str:
    metric_key = _normalize_metric_key(meta.positivity_metric)
    threshold = _safe_float(meta.positivity_threshold)

    if not metric_key or threshold != threshold:
        return "* Score calculated by the metric and threshold selected in the app."

    metric_label = METRIC_DISPLAY.get(metric_key, str(meta.positivity_metric or metric_key).replace("_", " "))
    return f"* Score calculated by {metric_label} > {_fmt_threshold(threshold)}."


def _metric_cell_style(
    canonical: str,
    meta: ReportMeta,
    normal: ParagraphStyle,
    bold: ParagraphStyle,
) -> ParagraphStyle:
    return bold if _normalize_metric_key(meta.positivity_metric) == canonical else normal


def _extract_positive_marker(raw_label: str) -> Optional[str]:
    """
    From labels like "CD3 PerCP-A+ CD19 APC-Cy7-A-" return "CD3".
    Uses the first marker with a '+' channel.
    """
    toks = str(raw_label or "").split()
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i + 1 < len(toks):
        marker = toks[i]
        chan = toks[i + 1]
        if chan.endswith("+") or chan.endswith("-"):
            pairs.append((marker, chan))
            i += 2
        else:
            i += 1

    positives = [(m, c) for (m, c) in pairs if c.endswith("+")]
    if not positives:
        return None
    return positives[0][0]


def _find_population_name(marker: str, marker_to_population: Optional[Dict[str, str]]) -> str:
    """
    marker_to_population comes from build_panel_from_rows(panel_rows):
      key = antibody string (user-defined, often "CD3", "CD19", ...)
      value = population name (user-defined, e.g. "T-cells", "B-cells")
    We match case-insensitively and also allow simple fallback matching.
    """
    if not marker_to_population:
        return ""

    if marker in marker_to_population:
        return str(marker_to_population.get(marker) or "")

    m_low = marker.lower()
    for k, v in marker_to_population.items():
        if str(k).lower() == m_low:
            return str(v or "")

    for k, v in marker_to_population.items():
        ks = str(k)
        if not ks:
            continue
        if ks.lower() in m_low or m_low in ks.lower():
            return str(v or "")

    return ""


def population_label_for_table(raw_label: str, marker_to_population: Optional[Dict[str, str]]) -> str:
    """
    Required output: "{marker} ({population})", based on user panel definitions.
    If population name not found, returns marker only.
    If parsing fails, returns the raw label.
    """
    marker = _extract_positive_marker(raw_label)
    if not marker:
        return str(raw_label or "")
    pop = _find_population_name(marker, marker_to_population)
    if pop:
        return f"{marker} ({pop})"
    return marker


def _subsample_points(points: List[Dict[str, Any]], max_points: int, seed: int) -> List[Dict[str, Any]]:
    n = len(points)
    if max_points <= 0 or n <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    idx = np.sort(idx)
    return [points[int(i)] for i in idx]


def _smart_axis(ax: plt.Axes, *, tick_label_size: float = 7.5) -> None:
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 4))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _scatter_png(
    points: List[Dict[str, Any]],
    title: str,
    x_label: str,
    y_label: str,
    width_in: float,
    height_in: float,
    max_points: int,
    seed: int,
) -> bytes:
    pts = _subsample_points(points, max_points=max_points, seed=seed)

    x = np.asarray([p.get("x", 0.0) for p in pts], dtype=float)
    y = np.asarray([p.get("y", 0.0) for p in pts], dtype=float)
    ing = np.asarray([bool(p.get("inGate", False)) for p in pts], dtype=bool)

    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_subplot(1, 1, 1)

    if np.any(~ing):
        ax.scatter(
            x[~ing], y[~ing],
            s=2, alpha=0.18,
            c=PLOT_POINT_COLOR,
            linewidths=0,
        )

    if np.any(ing):
        ax.scatter(
            x[ing], y[ing],
            s=6, alpha=0.80,
            c=PLOT_GATE_COLOR,
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_title(str(title or ""), fontsize=9)
    ax.set_xlabel(str(x_label or ""), fontsize=10)
    ax.set_ylabel(str(y_label or ""), fontsize=10)

    _smart_axis(ax, tick_label_size=7.5)

    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    return _fig_to_png_bytes(fig)


def _wrap_label(text: str, width: int = 30) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "-"
    parts = textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False)
    if not parts:
        return raw
    return "\n".join(parts)


def _series_name(s: Dict[str, Any]) -> str:
    tube = str(s.get("tube_name") or "").strip()
    if tube:
        return tube
    return str(s.get("filename") or "file").strip()


def _role_label_for_series(s: Dict[str, Any]) -> str:
    label = str(s.get("label", ""))
    role = str(s.get("role", "")).upper()

    if label.casefold() == "selected file":
        return "Selected file"
    if role == "NC" or label.casefold().startswith("negative control"):
        return "Negative Control"
    if role == "PC" or label.casefold().startswith("positive control"):
        return "Positive Control"
    return label or role or "File"


def _legend_label_for_series(s: Dict[str, Any]) -> str:
    role_label = _role_label_for_series(s)
    return f"{role_label}\n{_wrap_label(_series_name(s), width=30)}"


def _style_for_series(s: Dict[str, Any]) -> Dict[str, Any]:
    label = str(s.get("label", ""))
    role = str(s.get("role", "")).upper()

    if label.casefold() == "selected file":
        return dict(color=PLOT_LINE_SELECTED_COLOR, linestyle="--", zorder=5)
    if role == "NC" or label.casefold().startswith("negative control"):
        return dict(color=PLOT_LINE_NC_COLOR, linestyle="-", zorder=3)
    if role == "PC" or label.casefold().startswith("positive control"):
        return dict(color=PLOT_LINE_PC_COLOR, linestyle="-", zorder=3)
    return dict(color=PLOT_LINE_OTHER_COLOR, linestyle="-", zorder=2)


def _hist_line_png(
    series: List[Dict[str, Any]],
    cutoff: float,
    title: str,
    width_in: float,
    height_in: float,
    bins: int = 80,
) -> bytes:
    vals_all: List[np.ndarray] = []
    for s in series:
        v = np.asarray(s.get("values", []), dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            vals_all.append(v)

    if vals_all:
        pooled = np.concatenate(vals_all)
        vmin = float(np.quantile(pooled, 0.01))
        vmax = float(np.quantile(pooled, 0.99))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    edges = np.linspace(vmin, vmax, int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_subplot(1, 1, 1)

    for s in series:
        v = np.asarray(s.get("values", []), dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            y = np.zeros_like(centers)
        else:
            hist, _ = np.histogram(v, bins=edges)
            hist = hist.astype(float)
            area = float(hist.sum())
            bin_w = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
            y = (hist / (area * bin_w)) if (area > 0 and bin_w > 0) else np.zeros_like(centers)

        st = _style_for_series(s)
        ax.plot(
            centers, y,
            linewidth=PLOT_LINE_WIDTH,
            label=_legend_label_for_series(s),
            color=st["color"],
            linestyle=st["linestyle"],
            zorder=st["zorder"],
        )

    if np.isfinite(float(cutoff)):
        ax.axvline(float(cutoff), linestyle=":", linewidth=1.4, color=PLOT_CUTOFF_COLOR)

    ax.set_xscale("symlog", linthresh=250)
    ax.set_title(str(title or ""), fontsize=8.5, pad=2)
    ax.set_xlabel("IgG (transformed)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.6)
    _smart_axis(ax, tick_label_size=8)

    ax.legend(
        fontsize=7.2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=2.6,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.10, right=0.66, top=0.68, bottom=0.20)
    return _fig_to_png_bytes(fig, tight=False)


def _compute_mfi_triplet(line_series: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    nc_vals: List[float] = []
    pc_vals: List[float] = []
    sel_vals: List[float] = []

    for s in line_series:
        label = str(s.get("label", ""))
        role = str(s.get("role", "")).upper()
        raw_values = s.get("values_raw")
        if raw_values is None:
            raw_values = s.get("values", [])
        values = np.asarray(raw_values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        if label.casefold() == "selected file":
            sel_vals.extend(values.tolist())
        elif role == "NC" or label.casefold().startswith("negative control"):
            nc_vals.extend(values.tolist())
        elif role == "PC" or label.casefold().startswith("positive control"):
            pc_vals.extend(values.tolist())

    return (
        _median_mfi(nc_vals),
        _median_mfi(pc_vals),
        _median_mfi(sel_vals),
    )


def _metrics_title(pop_gate: str, metric_values: Dict[str, float]) -> str:
    row1 = str(pop_gate)

    row2_parts = []
    row3_parts = []

    for i, (canonical, display, _header) in enumerate(METRIC_ORDER):
        part = f"{display} {_fmt_num(metric_values.get(canonical))}"

        if i < 2:
            row2_parts.append(part)
        else:
            row3_parts.append(part)

    row2 = " | ".join(row2_parts)
    row3 = " | ".join(row3_parts)

    return f"{row1}\n{row2}\n{row3}"


def _summary_header(meta: ReportMeta, style_th: ParagraphStyle, style_th_bold: ParagraphStyle) -> List[Any]:
    row: List[Any] = [
        _cell_par("File", style_th),
        _cell_par("Tube", style_th),
        _cell_par("IgG+ (%)", style_th),
        _cell_par("MFI sample", style_th),
        _cell_par("MFI NC", style_th),
        _cell_par("MFI PC", style_th),
    ]
    for canonical, _display, header in METRIC_ORDER:
        row.append(_cell_par(header, _metric_cell_style(canonical, meta, style_th, style_th_bold)))
    row.append(_cell_par("Score*", style_th_bold))
    return row


def _detail_header(meta: ReportMeta, style_th: ParagraphStyle, style_th_bold: ParagraphStyle) -> List[Any]:
    row: List[Any] = [
        _cell_par("Population", style_th),
        _cell_par("IgG+ (%)", style_th),
        _cell_par("MFI sample", style_th),
        _cell_par("MFI NC", style_th),
        _cell_par("MFI PC", style_th),
    ]
    for canonical, _display, header in METRIC_ORDER:
        row.append(_cell_par(header, _metric_cell_style(canonical, meta, style_th, style_th_bold)))
    row.append(_cell_par("Score*", style_th_bold))
    return row


def _row_metric_cells(
    metric_values: Dict[str, float],
    meta: ReportMeta,
    style_cell: ParagraphStyle,
    style_cell_bold: ParagraphStyle,
) -> List[Any]:
    cells: List[Any] = []
    for canonical, _display, _header in METRIC_ORDER:
        st = _metric_cell_style(canonical, meta, style_cell, style_cell_bold)
        cells.append(_cell_par(_fmt_num(metric_values.get(canonical)), st))
    return cells


def _summary_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def _detail_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def build_fcxm_summary_pdf(
    payload: Dict[str, Any],
    plot_cache: Dict[str, Any],
    meta: Optional[ReportMeta] = None,
    score_rules: Optional[Iterable[ScoreRule]] = None,
    ratio_score_rules: Optional[Iterable[RatioRule]] = None,
    page_size_mode: str = PAGE_SIZE_MODE_DEFAULT,
) -> bytes:
    from reportlab.platypus.doctemplate import ActionFlowable

    _ = score_rules
    _ = ratio_score_rules

    meta = meta or ReportMeta()
    portrait_size, landscape_size = _pagesizes(page_size_mode)

    buf = io.BytesIO()

    left_margin = 16 * mm
    right_margin = 16 * mm
    top_margin = 12 * mm
    bottom_margin = 12 * mm

    doc = BaseDocTemplate(
        buf,
        pagesize=portrait_size,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="FCXM Summary",
    )

    class StartOnOddPage(ActionFlowable):
        """
        Force the next file block to start on an odd page.

        Odd page = front side.
        Even page = back side.
        """

        def __init__(self, template_id: str = "details_landscape"):
            super().__init__(())
            self.template_id = template_id

        def apply(self, doc):
            doc.handle_nextPageTemplate(self.template_id)

            # In this ActionFlowable context, doc.page is still the current page.
            # If current page is odd, one break would put the next content on even;
            # therefore insert an additional blank page.
            if doc.page % 2 == 1:
                doc.handle_pageBreak()
                doc.handle_pageBreak()
            else:
                doc.handle_pageBreak()

    styles = getSampleStyleSheet()
    style_h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
    )
    style_small = ParagraphStyle(
        "small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.black,
    )
    style_cell = ParagraphStyle(
        "cell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=7.8,
        textColor=colors.black,
    )
    style_cell_bold = ParagraphStyle(
        "cell_bold",
        parent=style_cell,
        fontName="Helvetica-Bold",
    )
    style_th = ParagraphStyle(
        "th",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.3,
        leading=7.2,
        textColor=colors.black,
    )
    style_th_bold = ParagraphStyle(
        "th_bold",
        parent=style_th,
        fontName="Helvetica-Bold",
    )

    marker_to_population: Dict[str, str] = {}
    if isinstance(payload, dict):
        pu = payload.get("panel_used", {})
        if isinstance(pu, dict):
            mtp = pu.get("marker_to_population", {})
            if isinstance(mtp, dict):
                marker_to_population = {str(k): str(v) for k, v in mtp.items()}

    def footer(canvas, _doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(doc.leftMargin, 8 * mm, f"Job: {meta.job_id or '-'}")
        canvas.drawRightString(
            canvas._pagesize[0] - doc.rightMargin,
            8 * mm,
            f"Page {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    def _draw_title(canvas, text: str):
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(
            doc.leftMargin,
            canvas._pagesize[1] - doc.topMargin + 1 * mm,
            text,
        )
        canvas.setFont("Helvetica", 9)

    def _draw_editable_fields(canvas):
        form = canvas.acroForm
        field_h = 6.5 * mm
        label_w = 30 * mm
        field_w = 70 * mm
        row_gap = 6.5 * mm

        x_left = doc.leftMargin
        x_right = (
            doc.leftMargin
            + (canvas._pagesize[0] - doc.leftMargin - doc.rightMargin) / 2
            + 4 * mm
        )
        y0 = canvas._pagesize[1] - doc.topMargin - 14 * mm

        def field(label: str, name: str, x: float, y: float, value: str = ""):
            canvas.drawString(x, y + 1.5 * mm, label)
            form.textfield(
                name=name,
                tooltip=label,
                x=x + label_w,
                y=y,
                width=field_w,
                height=field_h,
                fontName="Helvetica",
                fontSize=9,
                borderStyle="inset",
                borderWidth=1,
                borderColor=colors.black,
                fillColor=None,
                textColor=colors.black,
                value=value,
            )

        field("Report date:", "report_date", x_left, y0, value=date.today().isoformat())
        field("Laboratory:", "laboratory", x_right, y0, value="")
        y1 = y0 - (field_h + row_gap)
        field("Examiner:", "examiner", x_left, y1, value="")
        field("2nd reviewer:", "second_reviewer", x_right, y1, value="")

    def summary_on_page(canvas, _doc_obj):
        footer(canvas, _doc_obj)
        canvas.saveState()
        if canvas.getPageNumber() == 1:
            _draw_title(canvas, "Flow Cytometry Crossmatch (FCXM) Summary")
            _draw_editable_fields(canvas)
        else:
            _draw_title(canvas, "Flow Cytometry Crossmatch (FCXM) Summary (continued)")
        canvas.restoreState()

    def details_on_page(canvas, _doc_obj):
        footer(canvas, _doc_obj)
        canvas.saveState()
        _draw_title(canvas, "Flow Cytometry Crossmatch (FCXM) Details")
        canvas.restoreState()

    lw, lh = landscape_size
    pw, ph = portrait_size

    frame_land = Frame(
        left_margin,
        bottom_margin,
        lw - left_margin - right_margin,
        lh - top_margin - bottom_margin,
        id="frame_land",
    )
    frame_port = Frame(
        left_margin,
        bottom_margin,
        pw - left_margin - right_margin,
        ph - top_margin - bottom_margin,
        id="frame_port",
    )

    tpl_summary_land = PageTemplate(
        id="summary_landscape",
        frames=[frame_land],
        onPage=summary_on_page,
        pagesize=landscape_size,
    )
    tpl_details_land = PageTemplate(
        id="details_landscape",
        frames=[frame_land],
        onPage=details_on_page,
        pagesize=landscape_size,
    )
    tpl_port = PageTemplate(
        id="portrait",
        frames=[frame_port],
        onPage=details_on_page,
        pagesize=portrait_size,
    )
    doc.addPageTemplates([tpl_summary_land, tpl_details_land, tpl_port])

    results = (payload or {}).get("results", [])
    key_map = _build_file_key_map(plot_cache or {})

    story: List[Any] = []

    story.append(Spacer(1, 32 * mm))
    story.append(Paragraph("Results overview", style_h2))

    has_singlets_any = False
    for _k, e in (plot_cache or {}).items():
        opts = list(e.get("gate_options", []))
        if "Singlets" in opts:
            has_singlets_any = True
            break

    gate_rank = {"All Cells": 0, "Singlets": 1, "Lymphocytes": 2}

    def gate_path_label(gate_label: str) -> str:
        g = str(gate_label or "")
        if g == "All Cells":
            return "All Cells"
        if g == "Singlets":
            return "Singlets (/All Cells)"
        if g == "Lymphocytes":
            return "Lymphocytes (/" + ("Singlets" if has_singlets_any else "All Cells") + ")"
        return g + " (/" + (("Singlets/" if has_singlets_any else "") + "Lymphocytes") + ")"

    def _gate_sort_key(g: str) -> Tuple[int, str]:
        return (int(gate_rank.get(g, 3)), str(g).casefold())

    gate_to_rows: Dict[str, List[Tuple[str, List[Any]]]] = {}

    for sample in results:
        for fr in sample.get("per_file", []):
            file_name = str(fr.get("file_name", ""))
            file_base = _basename(file_name)
            key = key_map.get(file_base)
            if not key:
                continue

            entry = plot_cache.get(key, {})
            tube_name = _tube_name_from_entry(entry, fallback=file_base)
            gate_options = list(entry.get("gate_options", []))

            for gate_label in gate_options:
                gate_str = str(gate_label or "")

                rr = build_results_response_from_cache(
                    plot_cache=plot_cache,
                    selected_key=key,
                    selected_gate=gate_str,
                )
                line_series = rr.get("line_series", [])
                mfi_nc, mfi_pc, mfi_sel = _mfi_values_for_row(
                    rr=rr,
                    line_series=line_series,
                )
                sel_pct = _selected_pos_pct(line_series)
                metric_values = _metric_values_for_row(
                    rr=rr,
                    line_series=line_series,
                    mfi_nc=mfi_nc,
                    mfi_sel=mfi_sel,
                )

                row = [
                    _cell_par(file_base, style_cell),
                    _cell_par(tube_name, style_cell),
                    _cell_par(_fmt_num(sel_pct), style_cell),
                    _cell_par(_fmt_num(mfi_sel), style_cell),
                    _cell_par(_fmt_num(mfi_nc), style_cell),
                    _cell_par(_fmt_num(mfi_pc), style_cell),
                    *_row_metric_cells(metric_values, meta, style_cell, style_cell_bold),
                    _cell_par(_score_text(metric_values, meta), style_cell_bold),
                ]

                gate_to_rows.setdefault(gate_str, []).append(
                    (f"{file_base.casefold()}\t{tube_name.casefold()}", row)
                )

    sorted_gates = sorted(gate_to_rows.keys(), key=_gate_sort_key)

    usable_w = lw - left_margin - right_margin
    col_widths = [
        0.13 * usable_w,  # file
        0.13 * usable_w,  # tube
        0.07 * usable_w,  # IgG+
        0.08 * usable_w,  # MFI sample
        0.08 * usable_w,  # MFI NC
        0.08 * usable_w,  # MFI PC
        0.085 * usable_w, # frac_pos
        0.085 * usable_w, # median ratio
        0.085 * usable_w, # median shift
        0.085 * usable_w, # fluorescence index
        0.09 * usable_w,  # score
    ]

    any_tables = False

    for gate_str in sorted_gates:
        recs = gate_to_rows.get(gate_str, [])
        if not recs:
            continue

        recs.sort(key=lambda t: t[0])
        story.append(Paragraph(gate_path_label(gate_str), style_h2))

        rows: List[List[Any]] = [_summary_header(meta, style_th, style_th_bold)] + [
            r for _f, r in recs
        ]

        tbl = Table(rows, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(_summary_table_style())
        story.append(tbl)
        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(_score_rule_text(meta), style_small))
        story.append(Spacer(1, 3 * mm))
        any_tables = True

    if not any_tables:
        story.append(Paragraph("No results.", style_small))
        story.append(Spacer(1, 4 * mm))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("MFI is computed as median of raw IgG values.", style_small))
    story.append(Spacer(1, 6 * mm))

    sig_rows = [
        ["Examiner signature:", "_______________________________", "Date:", "____________"],
        ["Reviewer signature:", "_______________________________", "Date:", "____________"],
    ]
    sig_tbl = Table(sig_rows, colWidths=[35 * mm, 80 * mm, 12 * mm, 30 * mm])
    sig_tbl.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ]
        )
    )
    story.append(sig_tbl)

    # ---------- Detail pages ----------
    detail_usable_w = lw - left_margin - right_margin

    gate_plot_size = 55 * mm

    # Two IgG distribution plots per row.
    pop_plot_gap = 4 * mm
    pop_plot_width = (detail_usable_w - pop_plot_gap) / 2.0
    pop_plot_height = 52 * mm

    max_gate_plots_per_row = 4
    max_pops_per_page = 4

    gate_in = float(gate_plot_size) / 72.0
    pop_w_in = float(pop_plot_width) / 72.0
    pop_h_in = float(pop_plot_height) / 72.0

    for sample in results:
        role = str(sample.get("role", ""))

        for fr in sample.get("per_file", []):
            file_name = str(fr.get("file_name", ""))
            file_base = _basename(file_name)
            key = key_map.get(file_base)
            if not key:
                continue

            story.append(StartOnOddPage("details_landscape"))

            entry = plot_cache.get(key, {})
            tube_name = _tube_name_from_entry(entry, fallback=file_base)

            cutoff_raw_by_gate = entry.get("cutoff_raw_by_gate", {}) or {}
            cutoff_by_gate = entry.get("cutoff_by_gate", {}) or {}
            cutoff_t = float(cutoff_by_gate.get("All Cells", 0.0))
            cutoff = float(cutoff_raw_by_gate.get("All Cells", float("nan")))
            if not np.isfinite(cutoff):
                cutoff = _raw_cutoff_from_transformed(
                    cutoff_t,
                    float(entry.get("igg_cofactor", 150.0)),
                )

            story.append(Spacer(1, 6 * mm))
            story.append(Paragraph(f"Sample: {tube_name} | Role: {role}", style_h2))
            story.append(Paragraph(f"File: {file_base} | Raw IgG cutoff: {cutoff:.4f}", style_small))
            story.append(Spacer(1, 2 * mm))

            gate_options = list(entry.get("gate_options", []))
            pop_rows: List[List[Any]] = [_detail_header(meta, style_th, style_th_bold)]

            for gate_label in gate_options:
                gate_str = str(gate_label or "")
                rr = build_results_response_from_cache(
                    plot_cache=plot_cache,
                    selected_key=key,
                    selected_gate=gate_str,
                )
                line_series = rr.get("line_series", [])
                mfi_nc, mfi_pc, mfi_sel = _mfi_values_for_row(
                    rr=rr,
                    line_series=line_series,
                )
                sel_pct = _selected_pos_pct(line_series)
                metric_values = _metric_values_for_row(
                    rr=rr,
                    line_series=line_series,
                    mfi_nc=mfi_nc,
                    mfi_sel=mfi_sel,
                )

                pop_rows.append(
                    [
                        _cell_par(str(gate_label), style_cell),
                        _cell_par(_fmt_num(sel_pct), style_cell),
                        _cell_par(_fmt_num(mfi_sel), style_cell),
                        _cell_par(_fmt_num(mfi_nc), style_cell),
                        _cell_par(_fmt_num(mfi_pc), style_cell),
                        *_row_metric_cells(metric_values, meta, style_cell, style_cell_bold),
                        _cell_par(_score_text(metric_values, meta), style_cell_bold),
                    ]
                )

            pf_col_widths = [
                0.20 * detail_usable_w, # population
                0.07 * detail_usable_w, # IgG+
                0.08 * detail_usable_w, # MFI sample
                0.08 * detail_usable_w, # MFI NC
                0.08 * detail_usable_w, # MFI PC
                0.09 * detail_usable_w, # frac_pos
                0.09 * detail_usable_w, # median ratio
                0.09 * detail_usable_w, # median shift
                0.09 * detail_usable_w, # fluorescence index
                0.08 * detail_usable_w, # score
            ]

            pop_tbl = Table(pop_rows, colWidths=pf_col_widths, repeatRows=1)
            pop_tbl.setStyle(_detail_table_style())
            story.append(pop_tbl)
            story.append(Spacer(1, 1.5 * mm))
            story.append(Paragraph(_score_rule_text(meta), style_small))
            story.append(Spacer(1, 4 * mm))

            # Page 1 of each file: population IgG curves.
            story.append(Paragraph("IgG distributions by population", style_h2))

            pop_gates = [
                g for g in gate_options
                if g not in ("All Cells", "Singlets", "Lymphocytes")
            ]

            pop_image_rows: List[List[Any]] = []
            current_pop_row: List[Any] = []
            pops_on_page = 0

            def flush_pop_image_rows() -> None:
                nonlocal pop_image_rows
                if not pop_image_rows:
                    return

                tbl = Table(
                    pop_image_rows,
                    colWidths=[pop_plot_width, pop_plot_width],
                    hAlign="LEFT",
                )
                tbl.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 3 * mm))
                pop_image_rows = []

            for pop_gate in pop_gates:
                if pops_on_page >= max_pops_per_page:
                    if current_pop_row:
                        while len(current_pop_row) < 2:
                            current_pop_row.append(Spacer(1, pop_plot_height))
                        pop_image_rows.append(current_pop_row)
                        current_pop_row = []

                    flush_pop_image_rows()

                    story.append(PageBreak())
                    story.append(Spacer(1, 6 * mm))
                    story.append(
                        Paragraph(
                            f"Sample: {tube_name} | Role: {role} (continued)",
                            style_h2,
                        )
                    )
                    story.append(Paragraph(f"File: {file_base} | Raw IgG cutoff: {cutoff:.4f}", style_small))
                    story.append(Spacer(1, 2 * mm))
                    story.append(Paragraph("IgG distributions by population", style_h2))
                    pops_on_page = 0

                rr = build_results_response_from_cache(
                    plot_cache=plot_cache,
                    selected_gate=pop_gate,
                    selected_key=key,
                )
                line_series = rr.get("line_series", [])

                # Passing NaN suppresses the cutoff line in _hist_line_png().
                cutoff_local = float("nan")

                mfi_nc, _mfi_pc, mfi_sel = _mfi_values_for_row(
                    rr=rr,
                    line_series=line_series,
                )
                metric_values = _metric_values_for_row(
                    rr=rr,
                    line_series=line_series,
                    mfi_nc=mfi_nc,
                    mfi_sel=mfi_sel,
                )

                png = _hist_line_png(
                    series=line_series,
                    cutoff=cutoff_local,
                    title=_metrics_title(str(pop_gate), metric_values),
                    width_in=pop_w_in,
                    height_in=pop_h_in,
                )

                current_pop_row.append(
                    Image(
                        io.BytesIO(png),
                        width=pop_plot_width,
                        height=pop_plot_height,
                    )
                )

                if len(current_pop_row) == 2:
                    pop_image_rows.append(current_pop_row)
                    current_pop_row = []

                pops_on_page += 1

            if current_pop_row:
                while len(current_pop_row) < 2:
                    current_pop_row.append(Spacer(1, pop_plot_height))
                pop_image_rows.append(current_pop_row)

            flush_pop_image_rows()

            # Page 2 of each file: gating strategy.
            story.append(PageBreak())
            story.append(Spacer(1, 6 * mm))
            story.append(Paragraph(f"Sample: {tube_name} | Role: {role} (gating)", style_h2))
            story.append(Paragraph(f"File: {file_base}", style_small))
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph("Gating strategy", style_h2))

            rr0 = build_results_response_from_cache(
                plot_cache=plot_cache,
                selected_gate="All Cells",
                selected_key=key,
            )
            gating_plots = rr0.get("gating_plots", [])

            gate_imgs: List[Image] = []
            for gp in gating_plots:
                png = _scatter_png(
                    points=gp.get("points", []),
                    title=str(gp.get("title", "")),
                    x_label=str(gp.get("x_label", "")),
                    y_label=str(gp.get("y_label", "")),
                    width_in=gate_in,
                    height_in=gate_in,
                    max_points=MAX_EVENTS_PER_GATING_PLOT,
                    seed=GATING_SUBSAMPLE_SEED,
                )
                gate_imgs.append(
                    Image(
                        io.BytesIO(png),
                        width=gate_plot_size,
                        height=gate_plot_size,
                    )
                )

            grid_rows: List[List[Any]] = []
            row: List[Any] = []

            for im in gate_imgs:
                row.append(im)
                if len(row) == max_gate_plots_per_row:
                    grid_rows.append(row)
                    row = []

            if row:
                while len(row) < max_gate_plots_per_row:
                    row.append(Spacer(1, gate_plot_size))
                grid_rows.append(row)

            if grid_rows:
                grid = Table(
                    grid_rows,
                    colWidths=[gate_plot_size] * max_gate_plots_per_row,
                )
                grid.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ]
                    )
                )
                story.append(grid)
                story.append(Spacer(1, 4 * mm))

    doc.build(story)
    return buf.getvalue()
