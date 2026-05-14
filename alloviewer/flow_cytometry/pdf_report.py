from __future__ import annotations

import io
import os
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

from .scoring import ScoreRule, RatioRule, pct_to_score_verdict, ratio_to_score_verdict
from .plots import build_results_response_from_cache


# -------------------------
# Config (edit these later)
# -------------------------

# Gating dot plots (match frontend defaults)
PLOT_POINT_COLOR = "#64748b"  # out-of-gate fill
PLOT_GATE_COLOR = "#22c55e"   # in-gate fill
PLOT_POINT_STROKE_COLOR = "#0f172a"
PLOT_POINT_STROKE_WIDTH = 0.35

PLOT_AXIS_COLOR = "#94a3b8"
PLOT_TEXT_COLOR = "#64748b"

# Line plots (color + markers)
PLOT_LINE_WIDTH = 1.6
PLOT_CUTOFF_COLOR = "#ef4444"

LINE_SERIES_STYLE = {
    "Negative control": dict(color="#22c55e", marker="o", linestyle="-"),
    "Positive control": dict(color="#ef4444", marker="s", linestyle="--"),
    "Selected file": dict(color="#3b82f6", marker="^", linestyle="-."),  # sample
}

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


# -------------------------
# Page sizes
# -------------------------

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


# -------------------------
# Formatting helpers
# -------------------------

def _fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
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


def _median_mfi(values: Any) -> float:
    v = np.asarray(values if values is not None else [], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(v))


def _fmt_num(x: float, nd: int = 2) -> str:
    return "-" if (x != x) else f"{x:.{nd}f}"


def _ratio_value(mfi_sample: float, mfi_nc: float) -> float:
    if (mfi_sample != mfi_sample) or (mfi_nc != mfi_nc) or float(mfi_nc) == 0.0:
        return float("nan")
    r = float(mfi_sample) / float(mfi_nc)
    if (r != r) or (r == float("inf")) or (r == float("-inf")):
        return float("nan")
    return r


def _fmt_ratio(mfi_sample: float, mfi_nc: float, nd: int = 2) -> str:
    r = _ratio_value(mfi_sample, mfi_nc)
    return "-" if (r != r) else f"{r:.{nd}f}"


def _cell_par(text: str, style: ParagraphStyle) -> Paragraph:
    safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


# -------------------------
# Population label helpers
# -------------------------

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

    # Exact
    if marker in marker_to_population:
        return str(marker_to_population.get(marker) or "")

    # Case-insensitive exact
    m_low = marker.lower()
    for k, v in marker_to_population.items():
        if str(k).lower() == m_low:
            return str(v or "")

    # Fallback: key contained in marker (rare) or marker contained in key (e.g. "CD3" in "CD3 PerCP-A")
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


# -------------------------
# Plot helpers
# -------------------------

def _subsample_points(points: List[Dict[str, Any]], max_points: int, seed: int) -> List[Dict[str, Any]]:
    n = len(points)
    if max_points <= 0 or n <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    idx = np.sort(idx)
    return [points[int(i)] for i in idx]


def _smart_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=7)
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
            linewidths=0
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
    ax.set_xlabel(str(x_label or ""), fontsize=8)
    ax.set_ylabel(str(y_label or ""), fontsize=8)

    _smart_axis(ax)

    # Square axes box (not equal scaling)
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    return _fig_to_png_bytes(fig)


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

    markevery = max(1, bins // 16)

    for s in series:
        label = str(s.get("label", ""))
        v = np.asarray(s.get("values", []), dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            y = np.zeros_like(centers)
        else:
            hist, _ = np.histogram(v, bins=edges)
            hist = hist.astype(float)
            area = float(hist.sum())
            bin_w = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0

            # Density: integrates to 1 over x
            y = (hist / (area * bin_w)) if (area > 0 and bin_w > 0) else np.zeros_like(centers)

        st = LINE_SERIES_STYLE.get(label, dict(color="#111827", marker=None, linestyle="-"))
        ax.plot(
            centers, y,
            linewidth=PLOT_LINE_WIDTH,
            label=label,
            color=st.get("color", "#111827"),
            linestyle=st.get("linestyle", "-"),
            marker=st.get("marker", None),
            markersize=3.5,
            markevery=markevery,
        )

    ax.axvline(float(cutoff), linestyle=":", linewidth=1.4, color=PLOT_CUTOFF_COLOR)

    ax.set_title(str(title or ""), fontsize=9)
    ax.set_xlabel("IgG (transformed)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.6)
    _smart_axis(ax)
    ax.legend(fontsize=7, loc="upper right", frameon=False)

    return _fig_to_png_bytes(fig)


def _compute_mfi_triplet(line_series: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    mfi_nc = float("nan")
    mfi_pc = float("nan")
    mfi_sel = float("nan")
    for s in line_series:
        label = str(s.get("label", ""))
        if label == "Negative control":
            mfi_nc = _median_mfi(s.get("values", []))
        elif label == "Positive control":
            mfi_pc = _median_mfi(s.get("values", []))
        elif label == "Selected file":
            mfi_sel = _median_mfi(s.get("values", []))
    return mfi_nc, mfi_pc, mfi_sel


def build_fcxm_summary_pdf(
    payload: Dict[str, Any],
    plot_cache: Dict[str, Any],
    meta: Optional[ReportMeta] = None,
    score_rules: Optional[Iterable[ScoreRule]] = None,
    ratio_score_rules: Optional[Iterable[RatioRule]] = None,
    page_size_mode: str = PAGE_SIZE_MODE_DEFAULT,
) -> bytes:
    meta = meta or ReportMeta()
    portrait_size, landscape_size = _pagesizes(page_size_mode)

    buf = io.BytesIO()

    # Layout
    left_margin = 16 * mm
    right_margin = 16 * mm
    top_margin = 12 * mm
    bottom_margin = 12 * mm

    doc = BaseDocTemplate(
        buf,
        pagesize=portrait_size,  # templates override
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="FCXM Summary",
    )

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
        fontSize=7.4,
        leading=8.6,
        textColor=colors.black,
    )
    style_th = ParagraphStyle(
        "th",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.0,
        leading=8.0,
        textColor=colors.black,
    )

    # marker_to_population is stored in payload["panel_used"]["marker_to_population"]
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
        canvas.drawRightString(canvas._pagesize[0] - doc.rightMargin, 8 * mm, f"Page {canvas.getPageNumber()}")
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
        x_right = doc.leftMargin + (canvas._pagesize[0] - doc.leftMargin - doc.rightMargin) / 2 + 4 * mm
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

    # Summary: page 1 with fields, later pages without fields
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

    # Templates
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

    tpl_land = PageTemplate(id="landscape", frames=[frame_land], onPage=summary_on_page, pagesize=landscape_size)
    tpl_port = PageTemplate(id="portrait", frames=[frame_port], onPage=details_on_page, pagesize=portrait_size)
    doc.addPageTemplates([tpl_land, tpl_port])

    results = (payload or {}).get("results", [])
    key_map = _build_file_key_map(plot_cache or {})

    # Cache MFI per (file_key, gate_label)
    mfi_cache: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

    def get_mfi_for(key: str, gate: str) -> Tuple[float, float, float]:
        ck = (key, gate)
        if ck in mfi_cache:
            return mfi_cache[ck]
        rr = build_results_response_from_cache(plot_cache=plot_cache, selected_gate=gate, selected_key=key)
        mfi_cache[ck] = _compute_mfi_triplet(rr.get("line_series", []))
        return mfi_cache[ck]

    story: List[Any] = []

    # Leave space for title + fields on summary page 1
    story.append(Spacer(1, 32 * mm))

    # ---------- Summary tables (landscape), grouped by population ----------
    story.append(Paragraph("Results overview", style_h2))

    # Headline shows the gate + gating path, so the table does NOT need a "Population" column.
    header_row: List[Any] = [
        _cell_par("File", style_th),
        _cell_par("IgG+ (%)", style_th),
        _cell_par("MFI (IgG) sample", style_th),
        _cell_par("MFI (IgG) NC", style_th),
        _cell_par("MFI (IgG) PC", style_th),
        _cell_par("MFI ratio (S/NC)", style_th),
        _cell_par("Score % (pct)", style_th),
        _cell_par("Score (ratio)", style_th),
    ]

    # Determine if singlets gate exists (usually consistent across dataset)
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
        # population gate
        return g + " (/" + (("Singlets/" if has_singlets_any else "") + "Lymphocytes") + ")"

    # Collect rows per gate
    gate_to_rows: Dict[str, List[Tuple[str, List[Any]]]] = {}  # gate -> list[(file_sort_key, row)]
    gate_seen_order: Dict[str, int] = {}

    for sample in results:
        for fr in sample.get("per_file", []):
            file_name = str(fr.get("file_name", ""))
            file_base = _basename(file_name)
            key = key_map.get(file_base)
            if not key:
                continue

            entry = plot_cache.get(key, {})
            gate_options = list(entry.get("gate_options", []))

            for gate_label in gate_options:
                gate_str = str(gate_label or "")

                rr = build_results_response_from_cache(
                    plot_cache=plot_cache,
                    selected_key=key,
                    selected_gate=gate_str,
                )
                line_series = rr.get("line_series", [])
                mfi_nc, mfi_pc, mfi_sel = _compute_mfi_triplet(line_series)

                sel_pct = 0.0
                for s in line_series:
                    if str(s.get("label", "")) == "Selected file":
                        sel_pct = float(s.get("pos_pct", 0.0))
                        break

                pct_score, pct_verdict = pct_to_score_verdict(sel_pct, rules=score_rules)

                ratio = _ratio_value(mfi_sel, mfi_nc)
                ratio_str = _fmt_ratio(mfi_sel, mfi_nc)
                if ratio == ratio:
                    r_score, r_verdict = ratio_to_score_verdict(ratio, rules=ratio_score_rules)
                    ratio_score_str = f"{r_score} - {r_verdict}"
                else:
                    ratio_score_str = "-"

                row = [
                    _cell_par(file_base, style_cell),
                    _cell_par(f"{sel_pct:.2f}", style_cell),
                    _cell_par(_fmt_num(mfi_sel), style_cell),
                    _cell_par(_fmt_num(mfi_nc), style_cell),
                    _cell_par(_fmt_num(mfi_pc), style_cell),
                    _cell_par(ratio_str, style_cell),
                    _cell_par(f"{pct_score} - {pct_verdict}", style_cell),
                    _cell_par(ratio_score_str, style_cell),
                ]

                gate_to_rows.setdefault(gate_str, []).append((file_base.casefold(), row))

                if gate_str not in gate_seen_order:
                    # stable ordering: first by "step" gates, then alphabetic
                    gate_seen_order[gate_str] = gate_rank.get(gate_str, 3)

    # Sort gates: step gates first, then alphabetically
    def _gate_sort_key(g: str) -> Tuple[int, str]:
        return (int(gate_rank.get(g, 3)), str(g).casefold())

    sorted_gates = sorted(gate_to_rows.keys(), key=_gate_sort_key)

    usable_w = lw - left_margin - right_margin
    col_widths = [
        0.22 * usable_w,  # file
        0.08 * usable_w,  # IgG
        0.12 * usable_w,  # MFI sample
        0.12 * usable_w,  # MFI NC
        0.12 * usable_w,  # MFI PC
        0.10 * usable_w,  # ratio
        0.12 * usable_w,  # score pct
        0.12 * usable_w,  # score ratio
    ]

    any_tables = False

    for gate_str in sorted_gates:
        recs = gate_to_rows.get(gate_str, [])
        if not recs:
            continue

        # Sort within table by file name
        recs.sort(key=lambda t: t[0])

        # Gate headline (includes gating path)
        story.append(Paragraph(gate_path_label(gate_str), style_h2))

        rows: List[List[Any]] = [header_row] + [r for _f, r in recs]

        tbl = Table(rows, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 3 * mm))
        any_tables = True

    if not any_tables:
        # keep page structure stable even if no data
        story.append(Paragraph("No results.", style_small))
        story.append(Spacer(1, 4 * mm))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("MFI is computed as median of transformed IgG values.", style_small))
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

    # Switch to portrait template for detail pages
    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())

    # ---------- Detail pages ----------
    gate_plot_size = 55 * mm
    pop_plot_width = 155 * mm
    pop_plot_height = 50 * mm
    max_gate_plots_per_row = 3
    max_pops_per_page = 5

    gate_in = float(gate_plot_size) / 72.0
    pop_w_in = float(pop_plot_width) / 72.0
    pop_h_in = float(pop_plot_height) / 72.0

    for sample in results:
        sample_name = str(sample.get("sample_name", ""))
        role = str(sample.get("role", ""))

        for fr in sample.get("per_file", []):
            file_name = str(fr.get("file_name", ""))
            key = key_map.get(_basename(file_name))
            if not key:
                continue

            entry = plot_cache[key]
            cutoff = float(entry.get("cutoff", 0.0))

            story.append(Paragraph(f"Sample: {sample_name} | Role: {role}", style_h2))
            story.append(Paragraph(f"File: {_basename(file_name)} | IgG cutoff: {cutoff:.4f}", style_small))
            story.append(Spacer(1, 2 * mm))

            # Per-file summary table
            pf_header_row: List[Any] = [
                _cell_par("Population", style_th),
                _cell_par("IgG+ (%)", style_th),
                _cell_par("MFI (IgG) sample", style_th),
                _cell_par("MFI (IgG) NC", style_th),
                _cell_par("MFI (IgG) PC", style_th),
                _cell_par("MFI ratio (S/NC)", style_th),
                _cell_par("Score % (pct)", style_th),
                _cell_par("Score (ratio)", style_th),
            ]
            pop_rows: List[List[Any]] = [pf_header_row]

            entry = plot_cache.get(key, {})
            gate_options = list(entry.get("gate_options", []))

            for gate_label in gate_options:
                rr = build_results_response_from_cache(
                    plot_cache=plot_cache,
                    selected_key=key,
                    selected_gate=gate_label,
                )
                line_series = rr.get("line_series", [])
                mfi_nc, mfi_pc, mfi_sel = _compute_mfi_triplet(line_series)

                sel_pct = 0.0
                for s in line_series:
                    if str(s.get("label", "")) == "Selected file":
                        sel_pct = float(s.get("pos_pct", 0.0))
                        break

                pct_score, pct_verdict = pct_to_score_verdict(sel_pct, rules=score_rules)

                ratio = _ratio_value(mfi_sel, mfi_nc)
                ratio_str = _fmt_ratio(mfi_sel, mfi_nc)
                if ratio == ratio:
                    r_score, r_verdict = ratio_to_score_verdict(ratio, rules=ratio_score_rules)
                    ratio_score_str = f"{r_score} - {r_verdict}"
                else:
                    ratio_score_str = "-"

                pop_rows.append(
                    [
                        _cell_par(str(gate_label), style_cell),  # EXACTLY as frontend
                        _cell_par(f"{sel_pct:.2f}", style_cell),
                        _cell_par(_fmt_num(mfi_sel), style_cell),
                        _cell_par(_fmt_num(mfi_nc), style_cell),
                        _cell_par(_fmt_num(mfi_pc), style_cell),
                        _cell_par(ratio_str, style_cell),
                        _cell_par(f"{pct_score} - {pct_verdict}", style_cell),
                        _cell_par(ratio_score_str, style_cell),
                    ]
                )

            usable_pw = pw - left_margin - right_margin
            pf_col_widths = [
                0.26 * usable_pw,
                0.08 * usable_pw,
                0.12 * usable_pw,
                0.12 * usable_pw,
                0.12 * usable_pw,
                0.10 * usable_pw,
                0.10 * usable_pw,
                0.10 * usable_pw,
            ]

            pop_tbl = Table(pop_rows, colWidths=pf_col_widths, repeatRows=1)
            pop_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            story.append(pop_tbl)
            story.append(Spacer(1, 4 * mm))

            # Gating plots
            story.append(Paragraph("Gating strategy", style_h2))
            rr0 = build_results_response_from_cache(plot_cache=plot_cache, selected_gate="All Cells", selected_key=key)
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
                gate_imgs.append(Image(io.BytesIO(png), width=gate_plot_size, height=gate_plot_size))

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
                grid = Table(grid_rows, colWidths=[gate_plot_size] * max_gate_plots_per_row)
                grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
                story.append(grid)
                story.append(Spacer(1, 4 * mm))

            # Population IgG curves
            story.append(Paragraph("IgG distributions by population", style_h2))

            gate_options = list(entry.get("gate_options", []))
            pop_gates = [g for g in gate_options if g not in ("All Cells", "Singlets", "Lymphocytes")]

            pops_on_page = 0
            for pop_gate in pop_gates:
                if pops_on_page >= max_pops_per_page:
                    story.append(PageBreak())
                    story.append(Paragraph(f"Sample: {sample_name} | Role: {role} (continued)", style_h2))
                    story.append(Paragraph(f"File: {_basename(file_name)} | IgG cutoff: {cutoff:.4f}", style_small))
                    story.append(Spacer(1, 2 * mm))
                    story.append(Paragraph("IgG distributions by population", style_h2))
                    pops_on_page = 0

                rr = build_results_response_from_cache(plot_cache=plot_cache, selected_gate=pop_gate, selected_key=key)
                line_series = rr.get("line_series", [])
                cutoff_local = float(rr.get("cutoff", cutoff))

                sel_pct = 0.0
                for s in line_series:
                    if str(s.get("label", "")) == "Selected file":
                        sel_pct = float(s.get("pos_pct", 0.0))
                        break

                pct_score, pct_verdict = pct_to_score_verdict(sel_pct, rules=score_rules)

                png = _hist_line_png(
                    series=line_series,
                    cutoff=cutoff_local,
                    title=f"{pop_gate} | IgG+ {sel_pct:.2f}% | Score% {pct_score} - {pct_verdict}",
                    width_in=pop_w_in,
                    height_in=pop_h_in,
                )
                story.append(Image(io.BytesIO(png), width=pop_plot_width, height=pop_plot_height))
                story.append(Spacer(1, 2 * mm))
                pops_on_page += 1

            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()

