from __future__ import annotations

import io
from datetime import date
from typing import Any, Dict, Iterable, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Flowable
)


PAGE_SIZE_MODE_DEFAULT = "SAFE"

class EditableTextBox(Flowable):
    def __init__(
        self,
        *,
        name: str,
        label: str,
        width: float,
        height: float,
        label_width: float = 38 * mm,
        value: str = "",
        tooltip: str | None = None,
        font_size: int = 8,
    ):
        super().__init__()
        self.name = name
        self.label = label
        self.width = width
        self.height = height
        self.label_width = label_width
        self.value = value
        self.tooltip = tooltip or label
        self.font_size = font_size

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        field_x = self.label_width
        field_w = self.width - self.label_width

        canvas.saveState()

        canvas.setStrokeColor(colors.grey)
        canvas.setLineWidth(0.25)
        canvas.rect(0, 0, self.width, self.height, stroke=1, fill=0)
        canvas.line(self.label_width, 0, self.label_width, self.height)

        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawString(5, self.height - 8, self.label)

        canvas.acroForm.textfieldRelative(
            name=self.name,
            tooltip=self.tooltip,
            value=self.value,
            x=field_x + 2,
            y=2,
            width=field_w - 4,
            height=self.height - 4,
            fontName="Helvetica",
            fontSize=self.font_size,
            borderStyle="solid",
            borderWidth=0,
            borderColor=None,
            fillColor=None,
            textColor=colors.black,
            fieldFlags="multiline",
        )

        canvas.restoreState()

def _safe_pagesizes() -> tuple[tuple[float, float], tuple[float, float]]:
    a4w, a4h = A4
    lw, lh = LETTER
    portrait = (min(a4w, lw), min(a4h, lh))
    landscape = (portrait[1], portrait[0])
    return portrait, landscape


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    if x != x:
        return "-"

    return f"{x:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"

    try:
        x = float(value)
    except Exception:
        return str(value)

    if x != x:
        return "-"

    return f"{x:.{digits}f}%"


def _safe_text(value: Any) -> str:
    text = "" if value is None else str(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _par(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_safe_text(value), style)


def _role_color(role: Optional[str]) -> colors.Color:
    r = str(role or "").lower()

    if r == "positive":
        return colors.HexColor("#fee2e2")

    if r == "negative":
        return colors.HexColor("#dcfce7")

    if r == "igm":
        return colors.HexColor("#fef3c7")

    if r == "sample":
        return colors.white

    return colors.HexColor("#f5f5f5")


def _well_value(well: Dict[str, Any]) -> Any:
    value = well.get("frac_pos_corrected")

    if value is None:
        value = well.get("corrected_frac_pos")

    if value is None:
        value = well.get("frac_pos")

    return value


def _is_positive_well(
    well_id: str,
    result: Dict[str, Any],
    threshold: Optional[float],
) -> bool:
    wells = result.get("wells", {}) or {}
    well = wells.get(well_id, {}) or {}

    value = _well_value(well)

    if value is None or threshold is None:
        return False

    try:
        return float(value) >= float(threshold)
    except Exception:
        return False


def _metric_table(
    rows: list[tuple[str, Any]],
    style_cell: ParagraphStyle,
    style_th: ParagraphStyle,
    col_widths: Optional[list[float]] = None,
) -> Table:
    data = [
        [_par("Metric", style_th), _par("Value", style_th)],
        *[
            [_par(label, style_cell), _par(value, style_cell)]
            for label, value in rows
        ],
    ]

    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    return tbl


def _well_cell(
    well_id: str,
    value: Any,
    role: Any,
    style_well_id: ParagraphStyle,
    style_cell: ParagraphStyle,
    style_role: ParagraphStyle,
) -> list[Any]:
    return [
        _par(well_id, style_well_id),
        _par(_pct(value, 1), style_cell),
        _par(role or "-", style_role),
    ]


def _well_layout_table(
    result: Dict[str, Any],
    style_cell: ParagraphStyle,
    style_th: ParagraphStyle,
    style_well_id: ParagraphStyle,
    style_role: ParagraphStyle,
    threshold: Optional[float],
) -> Table:
    wells = result.get("wells", {}) or {}

    plate_rows = ["A", "B", "C", "D", "E", "F"]
    plate_cols = list(range(1, 11))

    data: list[list[Any]] = [
        [_par("", style_th)] + [_par(str(c), style_th) for c in plate_cols]
    ]

    backgrounds: list[tuple] = []

    for r_idx, row in enumerate(plate_rows, start=1):
        out_row: list[Any] = [_par(row, style_th)]

        for c_idx, col in enumerate(plate_cols, start=1):
            well_id = f"{row}{col}"
            well = wells.get(well_id, {}) or {}

            role = well.get("role") or "-"
            value = _well_value(well)

            out_row.append(
                _well_cell(
                    well_id=well_id,
                    value=value,
                    role=role,
                    style_well_id=style_well_id,
                    style_cell=style_cell,
                    style_role=style_role,
                )
            )

            bg = _role_color(role)

            if (
                _is_positive_well(well_id, result, threshold)
                and str(role).lower() == "sample"
            ):
                bg = colors.HexColor("#fee2e2")

            backgrounds.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), bg))

        data.append(out_row)

    tbl = Table(
        data,
        colWidths=[12 * mm] + [22 * mm for _ in plate_cols],
        rowHeights=[8 * mm] + [17 * mm for _ in plate_rows],
    )

    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                *backgrounds,
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    return tbl


def _allele_table(
    alleles: Iterable[Dict[str, Any]],
    style_cell: ParagraphStyle,
    style_th: ParagraphStyle,
    max_rows: int = 35,
) -> Table:
    sorted_alleles = sorted(
        list(alleles or []),
        key=lambda a: (
            float(a.get("positive_fraction") or 0),
            int(a.get("positive_well_count") or 0),
            int(a.get("total_well_count") or 0),
        ),
        reverse=True,
    )[:max_rows]

    data: list[list[Any]] = [
        [
            _par("Allele", style_th),
            _par("Positive score", style_th),
            _par("Fraction", style_th),
            _par("Positive wells", style_th),
            _par("Negative wells", style_th),
        ]
    ]

    for allele in sorted_alleles:
        positive_wells = ", ".join(allele.get("positive_wells") or [])

        positive_fraction = 0.0
        try:
            positive_fraction = float(allele.get("positive_fraction") or 0) * 100.0
        except Exception:
            positive_fraction = 0.0

        data.append(
            [
                _par(allele.get("allele_key", "-"), style_cell),
                _par(allele.get("positive_ratio", "-"), style_cell),
                _par(_pct(positive_fraction, 1), style_cell),
                _par(positive_wells or "-", style_cell),
                _par(str(allele.get("negative_well_count", "-")), style_cell),
            ]
        )

    if len(data) == 1:
        data.append(
            [
                _par("-", style_cell),
                _par("-", style_cell),
                _par("-", style_cell),
                _par("-", style_cell),
                _par("-", style_cell),
            ]
        )

    tbl = Table(
        data,
        colWidths=[30 * mm, 28 * mm, 24 * mm, 105 * mm, 28 * mm],
        repeatRows=1,
    )

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

    return tbl


def build_cdc_summary_pdf(
    result: Dict[str, Any],
    job_id: str,
) -> bytes:
    summary = result.get("summary", {}) or {}
    assay_type = str(
        summary.get("assay_type") or result.get("assay_type") or ""
    ).lower()

    run = summary.get("run_validity", {}) or {}
    assay = summary.get("assay_result", {}) or {}
    qc = summary.get("qc", {}) or {}
    pra = result.get("pra_analysis", {}) or {}

    threshold = pra.get("positivity_threshold")
    if threshold is None:
        threshold = 20.0

    _portrait_size, landscape_size = _safe_pagesizes()
    page_w, page_h = landscape_size

    buf = io.BytesIO()

    left_margin = 14 * mm
    right_margin = 14 * mm
    top_margin = 12 * mm
    bottom_margin = 12 * mm

    doc = BaseDocTemplate(
        buf,
        pagesize=landscape_size,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="CDC Summary",
    )

    frame = Frame(
        left_margin,
        bottom_margin,
        page_w - left_margin - right_margin,
        page_h - top_margin - bottom_margin,
        id="main",
    )

    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        spaceAfter=4,
    )

    style_h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=12,
        spaceBefore=6,
        spaceAfter=4,
    )

    style_small = ParagraphStyle(
        "small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
    )

    style_cell = ParagraphStyle(
        "cell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=8.4,
    )

    style_th = ParagraphStyle(
        "th",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.0,
        leading=8.0,
    )

    style_well_id = ParagraphStyle(
        "well_id",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=8.2,
        alignment=1,
    )

    style_role = ParagraphStyle(
        "role",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.6,
        leading=7.4,
        textColor=colors.HexColor("#404040"),
        alignment=1,
    )

    report_title = (
        "CDC PRA Summary" if assay_type == "pra" else "CDC Crossmatch Summary"
    )

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(doc.leftMargin, 7 * mm, f"Job: {job_id or '-'}")
        canvas.drawRightString(
            canvas._pagesize[0] - doc.rightMargin,
            7 * mm,
            f"Page {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    def page_canvas(canvas, _doc):
        footer(canvas, _doc)

        if canvas.getPageNumber() != 1:
            return

        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 15)
        canvas.drawString(
            doc.leftMargin,
            canvas._pagesize[1] - 10 * mm,
            report_title,
        )

        form = canvas.acroForm

        field_h = 6.0 * mm
        label_w = 25 * mm
        field_w = 54 * mm
        y = canvas._pagesize[1] - 22 * mm

        fields = [
            ("Report date:", "report_date", date.today().isoformat()),
            ("Patient ID:", "patient_id", ""),
            ("Sample ID:", "sample_id", ""),
            ("Laboratory:", "laboratory", ""),
            ("Examiner:", "examiner", ""),
            ("Reviewer:", "reviewer", ""),
        ]

        x0 = doc.leftMargin
        col_gap = 8 * mm

        for i, (label, name, value) in enumerate(fields):
            col = i % 3
            row = i // 3
            x = x0 + col * (label_w + field_w + col_gap)
            yy = y - row * 9 * mm

            canvas.setFont("Helvetica", 8)
            canvas.drawString(x, yy + 1.7 * mm, label)

            form.textfield(
                name=name,
                tooltip=label,
                x=x + label_w,
                y=yy,
                width=field_w,
                height=field_h,
                fontName="Helvetica",
                fontSize=8,
                borderStyle="inset",
                borderWidth=1,
                borderColor=colors.black,
                fillColor=None,
                textColor=colors.black,
                value=value,
            )

        canvas.restoreState()

    doc.addPageTemplates(
        [
            PageTemplate(
                id="landscape",
                frames=[frame],
                onPage=page_canvas,
                pagesize=landscape_size,
            )
        ]
    )

    story: list[Any] = []

    # ---------------------------------------------------------------------
    # Page 1: editable fields + result summary only
    # ---------------------------------------------------------------------
    story.append(Spacer(1, 24 * mm))
    story.append(Paragraph(report_title, style_title))

    if assay_type == "pra":
        reactivity = pra.get("reactivity_score", {}) or {}

        result_rows = [
            ("PRA", _pct(assay.get("pra_percent"), 1)),
            (
                "Positive panel wells",
                f"{assay.get('positive_panel_wells', '-')} / "
                f"{assay.get('valid_panel_wells', '-')}",
            ),
            (
                "Full panel reactivity",
                f"{_pct(reactivity.get('score_percent'), 1)} "
                f"({reactivity.get('positive_well_count', '-')} / "
                f"{reactivity.get('total_well_count', '-')})",
            ),
            (
                "Mean corrected fraction positive",
                _pct(assay.get("mean_corrected_frac_pos"), 1),
            ),
            (
                "Median corrected fraction positive",
                _pct(assay.get("median_corrected_frac_pos"), 1),
            ),
            (
                "Max corrected fraction positive",
                _pct(assay.get("max_corrected_frac_pos"), 1),
            ),
            ("Weak positives", assay.get("n_weak_positive", "-")),
            ("Moderate positives", assay.get("n_moderate_positive", "-")),
            ("Strong positives", assay.get("n_strong_positive", "-")),
            ("Positivity threshold", _pct(threshold, 1)),
        ]
    else:
        result_rows = [
            ("Final call", assay.get("final_call", "-")),
            (
                "Sample corrected fraction positive",
                _pct(assay.get("sample_corrected_frac_pos"), 1),
            ),
            (
                "Sample raw fraction positive",
                _pct(assay.get("sample_raw_frac_pos"), 1),
            ),
            ("Margin from cutoff", _fmt(assay.get("margin_from_cutoff"), 2)),
            ("Replicate SD", _fmt(assay.get("replicate_sd"), 2)),
            ("Replicate range", _fmt(assay.get("replicate_range"), 2)),
            ("Replicate discordant", assay.get("replicate_discordant", "-")),
            ("Sample wells", ", ".join(assay.get("sample_wells") or []) or "-"),
        ]

    run_rows = [
        ("Status", run.get("status", "-")),
        ("PC mean raw", _pct(run.get("pc_mean_raw"), 1)),
        ("NC mean raw", _pct(run.get("nc_mean_raw"), 1)),
        ("Dynamic range", _pct(run.get("dynamic_range"), 1)),
        ("Positive controls", run.get("n_positive_controls", "-")),
        ("Negative controls", run.get("n_negative_controls", "-")),
        ("PC replicate range", _fmt(run.get("pc_replicate_range"), 2)),
        ("NC replicate range", _fmt(run.get("nc_replicate_range"), 2)),
    ]

    qc_rows = [
        ("Total wells", qc.get("total_wells", "-")),
        ("Valid wells", qc.get("valid_wells", "-")),
        ("Mean ROI count", _fmt(qc.get("mean_n_rois"), 1)),
        ("Mean uncertain fraction", _fmt(qc.get("mean_uncertain_fraction"), 3)),
        ("Low ROI wells", len(qc.get("low_roi_wells") or [])),
        ("High uncertain wells", len(qc.get("high_uncertain_wells") or [])),
    ]

    usable_w = page_w - left_margin - right_margin
    third = usable_w / 3.0

    story.append(Paragraph("Result overview", style_h2))

    overview = Table(
        [
            [
                _metric_table(
                    result_rows,
                    style_cell,
                    style_th,
                    [34 * mm, third - 34 * mm],
                ),
                _metric_table(
                    run_rows,
                    style_cell,
                    style_th,
                    [32 * mm, third - 32 * mm],
                ),
                _metric_table(
                    qc_rows,
                    style_cell,
                    style_th,
                    [38 * mm, third - 38 * mm],
                ),
            ]
        ],
        colWidths=[third, third, third],
    )

    overview.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    story.append(overview)

    # ---------------------------------------------------------------------
    # Page 2: plate only
    # ---------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Well layout", style_title))
    story.append(
        Paragraph(
            "Cells show well ID, corrected fraction positive, and assigned role. "
            "Sample wells above threshold are highlighted.",
            style_small,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        _well_layout_table(
            result=result,
            style_cell=style_cell,
            style_th=style_th,
            style_well_id=style_well_id,
            style_role=style_role,
            threshold=threshold,
        )
    )

    # ---------------------------------------------------------------------
    # Last page: PRA HLA allele evidence only
    # Crossmatch reports intentionally stop after the plate page.
    # ---------------------------------------------------------------------
    if assay_type == "pra":
        story.append(PageBreak())
        story.append(Paragraph("HLA allele evidence", style_title))
        story.append(
            Paragraph(
                "Positive score means positive sample wells carrying the allele "
                "divided by all tested sample wells carrying the allele. Controls "
                "are excluded.",
                style_small,
            )
        )
        story.append(Spacer(1, 3 * mm))
        story.append(_allele_table(pra.get("alleles") or [], style_cell, style_th))

    story.append(PageBreak())
    story.append(Paragraph("Interpretation and sign-off", style_title))
    story.append(
        Paragraph(
            "This section can be completed after technical and clinical review.",
            style_small,
        )
    )
    story.append(Spacer(1, 4 * mm))

    story.append(
        EditableTextBox(
            name=f"{job_id}_interpretation",
            label="Interpretation:",
            width=usable_w,
            height=34 * mm,
            tooltip="Interpretation",
        )
    )

    story.append(Spacer(1, 4 * mm))

    story.append(
        EditableTextBox(
            name=f"{job_id}_comments",
            label="Comments:",
            width=usable_w,
            height=46 * mm,
            tooltip="Comments",
        )
    )

    story.append(Spacer(1, 5 * mm))

    signoff_rows = [
        [
            _par("Examiner signature:", style_th),
            _par(
                "______________________________    Date: ____________",
                style_cell,
            ),
        ],
        [
            _par("Reviewer signature:", style_th),
            _par(
                "______________________________    Date: ____________",
                style_cell,
            ),
        ],
    ]

    signoff_tbl = Table(
        signoff_rows,
        colWidths=[38 * mm, usable_w - 38 * mm],
        rowHeights=[14 * mm, 14 * mm],
    )

    signoff_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(signoff_tbl)

    doc.build(story)

    return buf.getvalue()
