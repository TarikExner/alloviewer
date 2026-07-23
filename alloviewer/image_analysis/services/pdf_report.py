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



def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number:
        return None

    return number


def _control_percent_with_range(
    value: Any,
    *,
    min_value: Any = None,
    max_value: Any = None,
    range_value: Any = None,
    digits: int = 1,
) -> str:
    main = _pct(value, digits)
    minimum = _finite_float(min_value)
    maximum = _finite_float(max_value)

    if minimum is not None and maximum is not None:
        return f"{main} (range: {minimum:.{digits}f}-{maximum:.{digits}f}%)"

    range_number = _finite_float(range_value)

    if range_number is not None:
        return f"{main} (range: {range_number:.{digits}f}%)"

    return main


def _crossmatch_call_label(value: Any) -> str:
    key = str(value or "not_available")

    return {
        "positive": "Positive",
        "negative": "Negative",
        "borderline": "Borderline",
        "needs_review": "Needs review",
        "not_available": "Not available",
    }.get(key, key)


def _normalized_column_modes(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[int, str] = {}

    for raw_column, raw_mode in value.items():
        try:
            column = int(raw_column)
        except (TypeError, ValueError):
            continue

        mode = str(raw_mode)

        if 1 <= column <= 10 and mode in {"T", "B", "T/B", "empty"}:
            normalized[column] = mode

    return normalized


def _safe_text(value: Any) -> str:
    text = "" if value is None else str(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _par(value: Any, style: ParagraphStyle) -> Paragraph:
    # ReportLab paragraphs ignore plain newline characters. Escape the text
    # first, then convert intentional newlines to explicit line breaks.
    text = _safe_text(value).replace("\n", "<br/>")
    return Paragraph(text, style)


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

    return value


def _well_raw_value(well: Dict[str, Any]) -> Any:
    return well.get("frac_pos")


def _well_manual_override(well: Dict[str, Any]) -> Dict[str, Any] | None:
    override = well.get("manual_override")

    if (
        isinstance(override, dict)
        and override.get("call") in {"positive", "negative"}
    ):
        return override

    return None


def _well_effective_call(
    well: Dict[str, Any],
    threshold: Optional[float],
) -> str:
    call = well.get("effective_call")

    if call in {"positive", "negative", "not_available"}:
        return str(call)

    override = _well_manual_override(well)

    if override is not None:
        return str(override["call"])

    value = _well_value(well)

    if value is None or threshold is None:
        return "not_available"

    try:
        return "positive" if float(value) >= float(threshold) else "negative"
    except Exception:
        return "not_available"


def _well_is_borderline(well: Dict[str, Any]) -> bool:
    explicit = well.get("borderline")

    if isinstance(explicit, bool):
        return explicit

    value = _finite_float(_well_value(well))

    return value is not None and 15.0 <= value <= 25.0


def _active_manual_overrides(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    wells = result.get("wells", {}) or {}
    rows: list[Dict[str, Any]] = []

    for well_id, well in wells.items():
        if not isinstance(well, dict):
            continue

        override = _well_manual_override(well)

        if override is None:
            continue

        rows.append(
            {
                "well_id": str(well_id),
                "automated_call": str(
                    well.get("automated_call")
                    or override.get("automated_call")
                    or "not_available"
                ),
                "effective_call": str(override.get("call")),
                "corrected": _well_value(well),
                "raw": _well_raw_value(well),
                "created_at": str(override.get("created_at") or "-"),
            }
        )

    return sorted(rows, key=lambda row: row["well_id"])


def _is_positive_well(
    well_id: str,
    result: Dict[str, Any],
    threshold: Optional[float],
) -> bool:
    wells = result.get("wells", {}) or {}
    well = wells.get(well_id, {}) or {}
    return _well_effective_call(well, threshold) == "positive"


def _well_column_number(well_id: Any) -> int | None:
    digits = ""

    for character in reversed(str(well_id)):
        if not character.isdigit():
            break
        digits = character + digits

    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def _positive_well_counts_by_role(
    result: Dict[str, Any],
    threshold: Optional[float],
    *,
    columns: Iterable[int] | None = None,
) -> list[tuple[str, int, int]]:
    """Count effective positive calls by role, optionally within plate columns.

    Manual declarations are included because ``_well_effective_call`` resolves
    the effective call before falling back to the measured value. For crossmatch
    reports, ``columns`` limits the count to one cell type. IgM is omitted only
    when the selected columns contain no IgM control wells.
    """
    wells = result.get("wells", {}) or {}
    selected_columns = None if columns is None else {int(column) for column in columns}
    role_specs = [
        ("positive", "PC", False),
        ("igm", "IgM", True),
        ("negative", "NC", False),
        ("sample", "Samples", False),
    ]
    counts: list[tuple[str, int, int]] = []

    for role_key, label, omit_when_empty in role_specs:
        role_wells = [
            well
            for well_id, well in wells.items()
            if isinstance(well, dict)
            and str(well.get("role") or "").lower() == role_key
            and (
                selected_columns is None
                or _well_column_number(well_id) in selected_columns
            )
        ]

        if omit_when_empty and not role_wells:
            continue

        positive_count = sum(
            _well_effective_call(well, threshold) == "positive"
            for well in role_wells
        )
        counts.append((label, positive_count, len(role_wells)))

    return counts


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


def _manual_override_table(
    overrides: list[Dict[str, Any]],
    style_cell: ParagraphStyle,
    style_th: ParagraphStyle,
    usable_width: float,
) -> Table:
    displayed = overrides[:12]
    data = [
        [
            _par("Well", style_th),
            _par("Automated", style_th),
            _par("User declaration", style_th),
            _par("Corrected", style_th),
            _par("Raw", style_th),
            _par("Timestamp (UTC)", style_th),
        ]
    ]

    for item in displayed:
        data.append(
            [
                _par(item["well_id"], style_cell),
                _par(_crossmatch_call_label(item["automated_call"]), style_cell),
                _par(_crossmatch_call_label(item["effective_call"]), style_cell),
                _par(_pct(item["corrected"], 1), style_cell),
                _par(_pct(item["raw"], 1), style_cell),
                _par(item["created_at"], style_cell),
            ]
        )

    table = Table(
        data,
        colWidths=[
            16 * mm,
            28 * mm,
            34 * mm,
            24 * mm,
            24 * mm,
            usable_width - 126 * mm,
        ],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ede9fe")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#7c3aed")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _well_cell(
    well_id: str,
    corrected_value: Any,
    raw_value: Any,
    role: Any,
    effective_call: str,
    borderline: bool,
    manual_override: Dict[str, Any] | None,
    style_well_id: ParagraphStyle,
    style_cell: ParagraphStyle,
    style_role: ParagraphStyle,
) -> list[Any]:
    corrected_text = _pct(corrected_value, 1)
    raw_text = _pct(raw_value, 1)
    role_key = str(role or "-").lower()
    role_label = {
        "positive": "Positive",
        "negative": "Negative",
        "igm": "IgM",
        "sample": "Sample",
    }.get(role_key, str(role or "-"))

    cell = [
        _par(well_id, style_well_id),
        _par(corrected_text, style_cell),
        _par(f"(raw: {raw_text})", style_role),
        _par(f"Role: {role_label}", style_role),
        _par(f"Call: {_crossmatch_call_label(effective_call)}", style_role),
    ]

    if borderline:
        cell.append(
            _par(
                "BORDERLINE",
                style_role,
            )
        )

    if manual_override is not None:
        cell.append(
            _par(
                f"OVERRIDE: {str(manual_override.get('call')).upper()}",
                style_role,
            )
        )

    return cell


def _well_layout_table(
    result: Dict[str, Any],
    style_cell: ParagraphStyle,
    style_th: ParagraphStyle,
    style_well_id: ParagraphStyle,
    style_role: ParagraphStyle,
    threshold: Optional[float],
    flip_vertical: bool = False,
    column_modes: Optional[Dict[int, str]] = None,
) -> Table:
    wells = result.get("wells", {}) or {}
    plate_rows = ["A", "B", "C", "D", "E", "F"]

    if flip_vertical:
        plate_rows.reverse()

    plate_cols = list(range(1, 11))
    normalized_modes = _normalized_column_modes(column_modes)
    has_column_labels = any(
        mode in {"T", "B", "T/B"}
        for mode in normalized_modes.values()
    )

    data: list[list[Any]] = [
        [_par("", style_th)] + [_par(str(column), style_th) for column in plate_cols],
        [
            _par("Cell type" if has_column_labels else "", style_th),
            *[
                _par(
                    normalized_modes.get(column, "")
                    if normalized_modes.get(column) != "empty"
                    else "",
                    style_th,
                )
                for column in plate_cols
            ],
        ],
    ]

    backgrounds: list[tuple] = []
    override_borders: list[tuple] = []

    for row_index, row in enumerate(plate_rows, start=2):
        output_row: list[Any] = [_par(row, style_th)]

        for column_index, column in enumerate(plate_cols, start=1):
            well_id = f"{row}{column}"
            well = wells.get(well_id, {}) or {}
            role = well.get("role") or "-"
            corrected_value = _well_value(well)
            raw_value = _well_raw_value(well)
            manual_override = _well_manual_override(well)
            effective_call = _well_effective_call(well, threshold)
            borderline = _well_is_borderline(well)

            output_row.append(
                _well_cell(
                    well_id=well_id,
                    corrected_value=corrected_value,
                    raw_value=raw_value,
                    role=role,
                    effective_call=effective_call,
                    borderline=borderline,
                    manual_override=manual_override,
                    style_well_id=style_well_id,
                    style_cell=style_cell,
                    style_role=style_role,
                )
            )

            background = _role_color(role)

            if (
                _is_positive_well(well_id, result, threshold)
                and str(role).lower() == "sample"
            ):
                background = colors.HexColor("#fee2e2")

            # Borderline is a separate flag from the binary positive/negative
            # call and takes visual priority over the role/call background.
            if borderline:
                background = colors.HexColor("#fff7cc")

            if manual_override is not None:
                override_borders.append(
                    (
                        "BOX",
                        (column_index, row_index),
                        (column_index, row_index),
                        1.5,
                        colors.HexColor("#7c3aed"),
                    )
                )

            backgrounds.append(
                (
                    "BACKGROUND",
                    (column_index, row_index),
                    (column_index, row_index),
                    background,
                )
            )

        data.append(output_row)

    table = Table(
        data,
        colWidths=[12 * mm] + [22 * mm for _ in plate_cols],
        rowHeights=[7 * mm, 7 * mm] + [22 * mm for _ in plate_rows],
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 1), colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                *backgrounds,
                *override_borders,
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

    return table

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
    flip_vertical: bool = False,
) -> bytes:
    summary = result.get("summary", {}) or {}
    assay_type = str(
        summary.get("assay_type") or result.get("assay_type") or ""
    ).lower()

    run = summary.get("run_validity", {}) or {}
    assay = summary.get("assay_result", {}) or {}
    qc = summary.get("qc", {}) or {}
    pra = result.get("pra_analysis", {}) or {}
    column_modes = _normalized_column_modes(
        result.get("column_modes") or summary.get("column_modes")
    )
    manual_overrides = _active_manual_overrides(result)

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
    usable_w = page_w - left_margin - right_margin

    # ---------------------------------------------------------------------
    # Page 1: editable fields + result summary only
    # ---------------------------------------------------------------------
    story.append(Spacer(1, 24 * mm))
    story.append(Paragraph(report_title, style_title))

    if manual_overrides:
        story.append(
            Table(
                [[
                    _par(
                        "MANUAL CLASSIFICATION OVERRIDES APPLIED - categorical "
                        "results use the user-declared calls; measured raw and "
                        "corrected fractions remain unchanged.",
                        style_cell,
                    )
                ]],
                colWidths=[usable_w],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ede9fe")),
                        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#7c3aed")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            )
        )
        story.append(Spacer(1, 1.5 * mm))
        override_details = []

        for item in manual_overrides[:10]:
            timestamp = item["created_at"].replace("T", " ")
            override_details.append(
                f"{item['well_id']}: "
                f"{_crossmatch_call_label(item['automated_call'])} -> "
                f"{_crossmatch_call_label(item['effective_call'])} "
                f"({timestamp})"
            )

        if len(manual_overrides) > 10:
            override_details.append(
                f"+{len(manual_overrides) - 10} additional override(s)"
            )

        story.append(
            _par(
                "Active overrides: " + "; ".join(override_details),
                style_small,
            )
        )
        story.append(Spacer(1, 2 * mm))

    positive_well_counts = _positive_well_counts_by_role(result, threshold)

    run_rows = [
        ("Status", run.get("status", "-")),
        (
            "Positive Control % positive",
            _control_percent_with_range(
                run.get("pc_mean_raw"),
                min_value=run.get("pc_min_raw") or run.get("pc_replicate_min"),
                max_value=run.get("pc_max_raw") or run.get("pc_replicate_max"),
                range_value=run.get("pc_replicate_range"),
                digits=1,
            ),
        ),
        (
            "Negative Control % positive",
            _control_percent_with_range(
                run.get("nc_mean_raw"),
                min_value=run.get("nc_min_raw") or run.get("nc_replicate_min"),
                max_value=run.get("nc_max_raw") or run.get("nc_replicate_max"),
                range_value=run.get("nc_replicate_range"),
                digits=1,
            ),
        ),
        (
            "Dynamic range between Positive and Negative Control",
            _pct(run.get("dynamic_range"), 1),
        ),
    ]

    qc_rows = [
        ("Total wells", qc.get("total_wells", "-")),
        ("Valid wells", qc.get("valid_wells", "-")),
        ("Mean cell count", _fmt(qc.get("mean_n_rois"), 1)),
        ("Mean uncertain fraction", _fmt(qc.get("mean_uncertain_fraction"), 3)),
        ("Low cell count wells", len(qc.get("low_roi_wells") or [])),
        ("High uncertain wells", len(qc.get("high_uncertain_wells") or [])),
    ]

    if assay_type == "pra":
        positive_well_count_text = "\n".join(
            f"{label} {positive} / {total}"
            for label, positive, total in positive_well_counts
        )
        run_rows.append(("Positive wells", positive_well_count_text or "-"))

        reactivity = pra.get("reactivity_score", {}) or {}

        result_rows = [
            (
                "Panel reactivity",
                f"{_pct(assay.get('pra_percent'), 1)} "
                f"({assay.get('positive_panel_wells', '-')} / "
                f"{assay.get('valid_panel_wells', '-')} panel wells)",
            ),
            (
                "Full panel reactivity",
                f"{_pct(reactivity.get('score_percent'), 1)} "
                f"({reactivity.get('positive_well_count', '-')} / "
                f"{reactivity.get('total_well_count', '-')} sample wells)",
            ),
            ("Mean corrected", _fmt(assay.get("mean_corrected_frac_pos"), 1)),
            ("Median corrected", _fmt(assay.get("median_corrected_frac_pos"), 1)),
            ("Max corrected", _fmt(assay.get("max_corrected_frac_pos"), 1)),
            ("Threshold", _fmt(threshold, 1)),
            (
                "Borderline wells",
                ", ".join(assay.get("borderline_wells") or []) or "None",
            ),
            ("Weak", assay.get("n_weak_positive", "-")),
            ("Moderate", assay.get("n_moderate_positive", "-")),
            ("Strong", assay.get("n_strong_positive", "-")),
            (
                "Manual overrides",
                ", ".join(summary.get("manual_override_wells") or []) or "None",
            ),
        ]

        third = usable_w / 3.0
        story.append(Paragraph("Summary values", style_h2))

        overview = Table(
            [
                [
                    [
                        Paragraph("Summary values", style_h2),
                        _metric_table(
                            result_rows,
                            style_cell,
                            style_th,
                            [34 * mm, third - 34 * mm - 6],
                        ),
                    ],
                    [
                        Paragraph("Run validity", style_h2),
                        _metric_table(
                            run_rows,
                            style_cell,
                            style_th,
                            [42 * mm, third - 42 * mm - 6],
                        ),
                    ],
                    [
                        Paragraph("QC", style_h2),
                        _metric_table(
                            qc_rows,
                            style_cell,
                            style_th,
                            [38 * mm, third - 38 * mm - 6],
                        ),
                    ],
                ]
            ],
            colWidths=[third, third, third],
        )
    else:
        by_cell_mode = assay.get("by_cell_mode", {}) or {}
        mode_specs = [
            ("T", "T-cell crossmatch"),
            ("B", "B-cell crossmatch"),
            ("T/B", "T/B-cell crossmatch"),
        ]
        mode_cells: list[Any] = []

        for mode, title in mode_specs:
            mode_result = by_cell_mode.get(mode, {}) or {}
            mode_run = mode_result.get("run_validity", {}) or {}
            columns = mode_result.get("columns") or []
            title_with_columns = (
                f"{title} (columns {', '.join(map(str, columns))})"
                if columns
                else title
            )
            mode_positive_counts = _positive_well_counts_by_role(
                result,
                threshold,
                columns=columns,
            )
            mode_positive_count_text = "\n".join(
                f"{label} {positive} / {total}"
                for label, positive, total in mode_positive_counts
            )

            mode_rows = [
                ("Run status", mode_run.get("status", "-")),
                (
                    "Positive Control % positive",
                    _control_percent_with_range(
                        mode_run.get("pc_mean_raw"),
                        range_value=mode_run.get("pc_replicate_range"),
                        digits=1,
                    ),
                ),
                (
                    "Negative Control % positive",
                    _control_percent_with_range(
                        mode_run.get("nc_mean_raw"),
                        range_value=mode_run.get("nc_replicate_range"),
                        digits=1,
                    ),
                ),
                (
                    "Dynamic range between Positive and Negative Control",
                    _pct(mode_run.get("dynamic_range"), 1),
                ),
                (
                    "Positive wells",
                    mode_positive_count_text or "-",
                ),
                (
                    "Final call",
                    (
                        f"{_crossmatch_call_label(mode_result.get('final_call'))} "
                        "(borderline)"
                        if mode_result.get("final_borderline")
                        else _crossmatch_call_label(mode_result.get("final_call"))
                    ),
                ),
                (
                    "Borderline wells",
                    ", ".join(mode_result.get("borderline_wells") or []) or "None",
                ),
                (
                    "User-adjusted wells",
                    ", ".join(mode_result.get("manual_override_wells") or []) or "None",
                ),
                (
                    "Corrected % positive",
                    _pct(mode_result.get("sample_corrected_frac_pos"), 1),
                ),
                (
                    "Raw % positive",
                    _pct(mode_result.get("sample_raw_frac_pos"), 1),
                ),
                (
                    "Margin from cutoff",
                    f"{_fmt(mode_result.get('margin_from_cutoff'), 1)} pp",
                ),
                (
                    "Replicate range",
                    f"{_fmt(mode_result.get('replicate_range'), 1)} pp",
                ),
                ("Replicate SD", _fmt(mode_result.get("replicate_sd"), 2)),
                (
                    "Sample wells",
                    ", ".join(mode_result.get("sample_wells") or []) or "-",
                ),
            ]

            mode_cells.append(
                [
                    Paragraph(title_with_columns, style_h2),
                    _metric_table(
                        mode_rows,
                        style_cell,
                        style_th,
                        [34 * mm, usable_w / 3.0 - 34 * mm - 6],
                    ),
                ]
            )

        story.append(Paragraph("Crossmatch results by cell type", style_h2))

        cell_results = Table(
            [mode_cells],
            colWidths=[usable_w / 3.0] * 3,
        )
        cell_results.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(cell_results)
        story.append(Spacer(1, 3 * mm))

        half = usable_w / 2.0
        overview = Table(
            [
                [
                    [
                        Paragraph("Run validity", style_h2),
                        _metric_table(
                            run_rows,
                            style_cell,
                            style_th,
                            [62 * mm, half - 62 * mm - 6],
                        ),
                    ],
                    [
                        Paragraph("QC", style_h2),
                        _metric_table(
                            qc_rows,
                            style_cell,
                            style_th,
                            [48 * mm, half - 48 * mm - 6],
                        ),
                    ],
                ]
            ],
            colWidths=[half, half],
        )

    overview.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
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
            "Cells show well ID, corrected fraction positive, the raw fraction on "
            "the following line in brackets, and the assigned role. Crossmatch "
            "column headers show T-cell, B-cell, or combined T/B-cell assignments. "
            "Borderline wells have a light yellow background. Non-borderline sample "
            "wells with an effective positive call are highlighted in light red. "
            "User-overridden cells carry a purple border and an explicit override "
            "label.",
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
            flip_vertical=flip_vertical,
            column_modes=column_modes,
        )
    )
    story.append(Spacer(1, 1.5 * mm))
    story.append(
        Paragraph(
            "Fractions positive were calibrated using the negative- and "
            "positive-control reference values. Raw fractions are shown in "
            "brackets on a separate line for interpretability. Manual "
            "positive/negative declarations supersede the automated call only for "
            "categorical downstream interpretation; measured fractions, control "
            "calibration, run validity, and QC values remain unchanged.",
            style_small,
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
                "are excluded. Active user declarations are applied to the positive/"
                "negative carrier-well classification and are listed on page 1.",
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
