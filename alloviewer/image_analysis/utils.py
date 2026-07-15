from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage import feature, measure, morphology
from skimage.segmentation import find_boundaries

from .structs import Plate, PlateLayout, WellImage, WellResult
from ..dev.segmentation import UNET_MEAN, UNET_STD


PRA_GENERIC_LAYOUT = PlateLayout(
    wells={
        "A1": "negative", "A2": "sample",  "A3": "sample",  "A4": "sample",  "A5": "sample",
        "A6": "sample",   "A7": "sample",  "A8": "sample",  "A9": "sample",  "A10": "positive",

        "B1": "negative", "B2": "sample",  "B3": "sample",  "B4": "sample",  "B5": "sample",
        "B6": "sample",   "B7": "sample",  "B8": "sample",  "B9": "sample",  "B10": "positive",

        "C1": "sample",   "C2": "sample",  "C3": "sample",  "C4": "sample",  "C5": "sample",
        "C6": "sample",   "C7": "sample",  "C8": "sample",  "C9": "sample",  "C10": "sample",

        "D1": "sample",   "D2": "sample",  "D3": "sample",  "D4": "sample",  "D5": "sample",
        "D6": "sample",   "D7": "sample",  "D8": "sample",  "D9": "sample",  "D10": "sample",

        "E1": "sample",   "E2": "sample",  "E3": "sample",  "E4": "sample",  "E5": "sample",
        "E6": "sample",   "E7": "sample",  "E8": "sample",  "E9": "sample",  "E10": "sample",

        "F1": "sample",   "F2": "sample",  "F3": "sample",  "F4": "sample",  "F5": "sample",
        "F6": "sample",   "F7": "sample",  "F8": "sample",  "F9": "sample",  "F10": "sample",
    }
)

PRA_GENERIC_IMAGE_ORDER = [
    "A1", "B1", "C1", "D1", "E1", "F1",
    "F2", "E2", "D2", "C2", "B2", "A2",
    "A3", "B3", "C3", "D3", "E3", "F3",
    "F4", "E4", "D4", "C4", "B4", "A4",
    "A5", "B5", "C5", "D5", "E5", "F5",
    "F6", "E6", "D6", "C6", "B6", "A6",
    "A7", "B7", "C7", "D7", "E7", "F7",
    "F8", "E8", "D8", "C8", "B8", "A8",
    "A9", "B9", "C9", "D9", "E9", "F9",
    "F10", "E10", "D10", "C10", "B10", "A10",
]


def create_plate(
    layout: PlateLayout,
    images: List[np.ndarray],
    image_order: List[str],
    image_paths: List[str],
) -> Plate:
    """Create a plate object from images and a well layout.

    Parameters
    ----------
    layout : PlateLayout
        Plate layout containing a role for each well ID.
    images : list of numpy.ndarray
        Images in acquisition order.
    image_order : list of str
        Well IDs matching the order of ``images``.
    image_paths : list of str
        Image file paths matching the order of ``images``.

    Returns
    -------
    Plate
        Plate containing one :class:`WellImage` per image.

    Raises
    ------
    KeyError
        If a well ID in ``image_order`` is missing from ``layout``.
    IndexError
        If ``images`` or ``image_paths`` is shorter than ``image_order``.
    """
    plate = Plate(plate_id="SIM001")

    for i, well_id in enumerate(image_order):
        role = layout.wells[well_id]
        plate.add(
            WellImage(
                well_id,
                role=role,
                image=images[i],
                path=image_paths[i],
            )
        )

    return plate


def frac_pos_raw(wr: WellResult) -> float:
    """Compute the raw positive ROI fraction in percent.

    Parameters
    ----------
    wr : WellResult
        Well result containing classified ROIs.

    Returns
    -------
    float
        Percentage of positive ROIs among positive and negative ROIs.
        Uncertain ROIs are ignored. Returns ``numpy.nan`` if no positive or
        negative ROIs are present.
    """
    n_pos = sum(1 for r in wr.rois if r.label == "pos")
    n_neg = sum(1 for r in wr.rois if r.label == "neg")
    n_total = n_pos + n_neg

    if n_total == 0:
        return np.nan

    return 100.0 * (n_pos / n_total)


def convert_frac_pos_to_score(frac_pos: int) -> int:
    """Convert a positive fraction to a discrete score.

    Parameters
    ----------
    frac_pos : int
        Positive fraction in percent.

    Returns
    -------
    int
        Score in ``{1, 2, 4, 6, 8}``.
    """
    if frac_pos <= 10:
        return 1
    if frac_pos <= 20:
        return 2
    if frac_pos <= 50:
        return 4
    if frac_pos <= 80:
        return 6

    return 8


def _safe_float(x: Any, default: float = np.nan) -> float:
    """Convert a value to float with a fallback.

    Parameters
    ----------
    x : Any
        Input value.
    default : float, optional
        Value returned when conversion fails. The default is ``numpy.nan``.

    Returns
    -------
    float
        Converted value or ``default``.
    """
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _mean_or_nan(xs: List[float]) -> float:
    """Return the mean of finite values or NaN.

    Parameters
    ----------
    xs : list of float
        Input values.

    Returns
    -------
    float
        Mean of non-NaN values, or ``nan`` if no values remain.
    """
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _median_or_nan(xs: List[float]) -> float:
    """Return the median of finite values or NaN.

    Parameters
    ----------
    xs : list of float
        Input values.

    Returns
    -------
    float
        Median of non-NaN values, or ``nan`` if no values remain.
    """
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.median(xs)) if xs else float("nan")


def _sd_or_nan(xs: List[float]) -> float:
    """Return the sample standard deviation of finite values or NaN.

    Parameters
    ----------
    xs : list of float
        Input values.

    Returns
    -------
    float
        Sample standard deviation using ``ddof=1``. Returns ``nan`` when fewer
        than two non-NaN values are present.
    """
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")


def _range_or_nan(xs: List[float]) -> float:
    """Return the range of finite values or NaN.

    Parameters
    ----------
    xs : list of float
        Input values.

    Returns
    -------
    float
        Difference between maximum and minimum non-NaN values. Returns ``nan``
        if no non-NaN values are present.
    """
    xs = [x for x in xs if not np.isnan(x)]
    return float(max(xs) - min(xs)) if xs else float("nan")


def _roi_label_counts(wr: WellResult) -> Dict[str, int]:
    """Count positive, negative, and uncertain ROIs.

    Parameters
    ----------
    wr : WellResult
        Well result containing ROI labels.

    Returns
    -------
    dict
        Counts with keys ``"n_total"``, ``"n_pos"``, ``"n_neg"``, and
        ``"n_uncertain"``.
    """
    n_pos = sum(1 for r in wr.rois if (r.label or "").lower() == "pos")
    n_neg = sum(1 for r in wr.rois if (r.label or "").lower() == "neg")
    n_total = len(wr.rois)
    n_uncertain = max(0, n_total - n_pos - n_neg)

    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_uncertain": n_uncertain,
    }


def _uncertain_fraction(wr: WellResult) -> float:
    """Compute the fraction of uncertain ROIs.

    Parameters
    ----------
    wr : WellResult
        Well result containing ROI labels.

    Returns
    -------
    float
        Fraction of ROIs that are neither positive nor negative. Returns
        ``nan`` if the well has no ROIs.
    """
    counts = _roi_label_counts(wr)

    if counts["n_total"] == 0:
        return float("nan")

    return counts["n_uncertain"] / counts["n_total"]


def build_pra_result(
    sample_ids: List[str],
    sample_corr: List[float],
    positive_cutoff: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a PRA assay result summary.

    Parameters
    ----------
    sample_ids : list of str
        Sample well IDs.
    sample_corr : list of float
        Corrected positive fractions for sample wells.
    positive_cutoff : float
        Cutoff used to define positive panel wells.
    config : dict
        Assay configuration. Must contain ``"weak_positive"``,
        ``"moderate_positive"``, and ``"strong_positive"``.

    Returns
    -------
    dict
        PRA summary with positive well count, valid well count, PRA percentage,
        corrected fraction summaries, intensity class counts, and positive well
        IDs.
    """
    valid = [
        (wid, val)
        for wid, val in zip(sample_ids, sample_corr)
        if not np.isnan(val)
    ]

    valid_panel_wells = len(valid)
    positive = [(wid, val) for wid, val in valid if val >= positive_cutoff]

    weak_cutoff = float(config["weak_positive"])
    moderate_cutoff = float(config["moderate_positive"])
    strong_cutoff = float(config["strong_positive"])

    n_weak = sum(1 for _, v in valid if weak_cutoff <= v < moderate_cutoff)
    n_moderate = sum(1 for _, v in valid if moderate_cutoff <= v < strong_cutoff)
    n_strong = sum(1 for _, v in valid if v >= strong_cutoff)

    pra_percent = (
        100.0 * len(positive) / valid_panel_wells
        if valid_panel_wells > 0
        else float("nan")
    )

    return {
        "pra_percent": pra_percent,
        "positive_panel_wells": len(positive),
        "valid_panel_wells": valid_panel_wells,
        "mean_corrected_frac_pos": _mean_or_nan([v for _, v in valid]),
        "median_corrected_frac_pos": _median_or_nan([v for _, v in valid]),
        "max_corrected_frac_pos": max([v for _, v in valid], default=float("nan")),
        "n_weak_positive": n_weak,
        "n_moderate_positive": n_moderate,
        "n_strong_positive": n_strong,
        "positive_wells": [wid for wid, _ in positive],
    }


def _call_from_value(
    value: float,
    borderline_low: float,
    borderline_high: float,
) -> str:
    """Convert a corrected fraction into a categorical assay call.

    Parameters
    ----------
    value : float
        Corrected positive fraction.
    borderline_low : float
        Lower threshold of the borderline interval.
    borderline_high : float
        Upper threshold of the borderline interval.

    Returns
    -------
    str
        One of ``"not_available"``, ``"negative"``, ``"borderline"``, or
        ``"positive"``.
    """
    if np.isnan(value):
        return "not_available"

    if value < borderline_low:
        return "negative"

    if value <= borderline_high:
        return "borderline"

    return "positive"


def build_crossmatch_result(
    sample_ids: List[str],
    sample_raw: List[float],
    sample_corr: List[float],
    positive_cutoff: float,
    borderline_low: float,
    borderline_high: float,
    max_replicate_range: float,
) -> Dict[str, Any]:
    """Build a crossmatch assay result summary.

    Parameters
    ----------
    sample_ids : list of str
        Sample well IDs.
    sample_raw : list of float
        Raw positive fractions for sample wells.
    sample_corr : list of float
        Corrected positive fractions for sample wells.
    positive_cutoff : float
        Positive cutoff used for margin reporting.
    borderline_low : float
        Lower threshold of the borderline interval.
    borderline_high : float
        Upper threshold of the borderline interval.
    max_replicate_range : float
        Maximum allowed range between replicate corrected fractions.

    Returns
    -------
    dict
        Crossmatch summary containing the final call, raw and corrected sample
        means, cutoff margin, replicate metrics, discordance flag, and sample
        well IDs.
    """
    valid_corr = [v for v in sample_corr if not np.isnan(v)]
    valid_raw = [v for v in sample_raw if not np.isnan(v)]

    mean_corr = _mean_or_nan(valid_corr)
    mean_raw = _mean_or_nan(valid_raw)

    replicate_range = _range_or_nan(valid_corr)
    replicate_sd = _sd_or_nan(valid_corr)

    replicate_discordant = (
        not np.isnan(replicate_range)
        and replicate_range > max_replicate_range
    )

    final_call = _call_from_value(
        mean_corr,
        borderline_low=borderline_low,
        borderline_high=borderline_high,
    )

    if replicate_discordant:
        final_call = "needs_review"

    return {
        "final_call": final_call,
        "sample_corrected_frac_pos": mean_corr,
        "sample_raw_frac_pos": mean_raw,
        "margin_from_cutoff": (
            mean_corr - positive_cutoff
            if not np.isnan(mean_corr)
            else float("nan")
        ),
        "replicate_sd": replicate_sd,
        "replicate_range": replicate_range,
        "replicate_discordant": replicate_discordant,
        "sample_wells": sample_ids,
    }


def _well_column(well_id: str) -> int | None:
    digits = ""

    for character in reversed(str(well_id)):
        if not character.isdigit():
            break
        digits = character + digits

    if not digits:
        return None

    return int(digits)


def _normalize_crossmatch_column_modes(
    column_modes: Dict[int, str] | None,
) -> Dict[int, str]:
    if not column_modes:
        return {}

    allowed = {"T", "B", "T/B", "empty"}
    normalized: Dict[int, str] = {}

    for raw_column, raw_mode in column_modes.items():
        column = int(raw_column)
        mode = str(raw_mode)

        if column < 1 or column > 10:
            raise ValueError(
                f"Crossmatch column index must be between 1 and 10: {column}"
            )

        if mode not in allowed:
            raise ValueError(
                f"Unsupported crossmatch cell mode for column {column}: {mode}"
            )

        normalized[column] = mode

    return normalized


def _build_run_validity_summary(
    *,
    pc_ids: List[str],
    nc_ids: List[str],
    raw_frac: Callable[[str], float],
    min_dynamic_range: float,
    max_replicate_range: float,
    qc_warnings: List[str] | None = None,
) -> Dict[str, Any]:
    pc_raw = [raw_frac(well_id) for well_id in pc_ids]
    nc_raw = [raw_frac(well_id) for well_id in nc_ids]
    pc_mean = _mean_or_nan(pc_raw)
    nc_mean = _mean_or_nan(nc_raw)
    dynamic_range = (
        pc_mean - nc_mean
        if not np.isnan(pc_mean) and not np.isnan(nc_mean)
        else float("nan")
    )
    pc_range = _range_or_nan(pc_raw)
    nc_range = _range_or_nan(nc_raw)
    control_warnings: List[str] = []

    if not pc_ids:
        control_warnings.append("No positive control wells.")

    if not nc_ids:
        control_warnings.append("No negative control wells.")

    if np.isnan(dynamic_range) or dynamic_range < min_dynamic_range:
        control_warnings.append("Poor positive/negative control separation.")

    if not np.isnan(pc_range) and pc_range > max_replicate_range:
        control_warnings.append("Positive control replicates differ strongly.")

    if not np.isnan(nc_range) and nc_range > max_replicate_range:
        control_warnings.append("Negative control replicates differ strongly.")

    status = "invalid" if control_warnings else "warning" if qc_warnings else "valid"

    return {
        "status": status,
        "pc_mean_raw": pc_mean,
        "nc_mean_raw": nc_mean,
        "dynamic_range": dynamic_range,
        "pc_replicate_range": pc_range,
        "nc_replicate_range": nc_range,
        "n_positive_controls": len(pc_ids),
        "n_negative_controls": len(nc_ids),
        "positive_control_wells": pc_ids,
        "negative_control_wells": nc_ids,
        "control_warnings": control_warnings,
    }


def build_cdc_summary(
    per_well: Dict[str, WellResult],
    plate: Plate,
    config: Dict[str, Any],
    assay_type: str = "pra",
    column_modes: Dict[int, str] | None = None,
) -> Dict[str, Any]:
    """Build CDC run validity, assay-result, and QC summaries.

    For crossmatch assays, ``column_modes`` groups sample wells into T-cell,
    B-cell, and combined T/B-cell result summaries while retaining the overall
    crossmatch result for compatibility and auditing.
    """
    positive_cutoff = float(config["positive_cutoff"])
    borderline_low = float(config["borderline_low"])
    borderline_high = float(config["borderline_high"])
    min_rois = int(config["min_rois"])
    max_uncertain_fraction = float(config["max_uncertain_fraction"])
    min_dynamic_range = float(config["min_dynamic_range"])
    max_replicate_range = float(config["max_replicate_range"])

    pc_ids = [well.well_id for well in plate.get("positive")]
    nc_ids = [well.well_id for well in plate.get("negative")]
    sample_ids = [well.well_id for well in plate.get("sample")]

    def raw_frac(well_id: str) -> float:
        return (
            frac_pos_raw(per_well[well_id])
            if well_id in per_well
            else float("nan")
        )

    def corr_frac(well_id: str) -> float:
        if well_id not in per_well:
            return float("nan")

        return _safe_float(per_well[well_id].corrected_frac_pos)

    sample_raw = [raw_frac(well_id) for well_id in sample_ids]
    sample_corr = [corr_frac(well_id) for well_id in sample_ids]

    all_ids = list(per_well.keys())
    low_roi_wells: List[str] = []
    high_uncertain_wells: List[str] = []

    for well_id, well_result in per_well.items():
        counts = _roi_label_counts(well_result)
        uncertain_fraction = _uncertain_fraction(well_result)

        if counts["n_total"] < min_rois:
            low_roi_wells.append(well_id)

        if (
            not np.isnan(uncertain_fraction)
            and uncertain_fraction > max_uncertain_fraction
        ):
            high_uncertain_wells.append(well_id)

    qc_warnings: List[str] = []

    if low_roi_wells:
        qc_warnings.append(f"{len(low_roi_wells)} well(s) have low ROI count.")

    if high_uncertain_wells:
        qc_warnings.append(
            f"{len(high_uncertain_wells)} well(s) have high uncertain fraction."
        )

    run_validity = _build_run_validity_summary(
        pc_ids=pc_ids,
        nc_ids=nc_ids,
        raw_frac=raw_frac,
        min_dynamic_range=min_dynamic_range,
        max_replicate_range=max_replicate_range,
        qc_warnings=qc_warnings,
    )

    qc_summary = {
        "total_wells": len(all_ids),
        "valid_wells": len(all_ids) - len(low_roi_wells),
        "low_roi_wells": low_roi_wells,
        "high_uncertain_wells": high_uncertain_wells,
        "mean_n_rois": _mean_or_nan(
            [float(len(well_result.rois)) for well_result in per_well.values()]
        ),
        "mean_uncertain_fraction": _mean_or_nan(
            [_uncertain_fraction(well_result) for well_result in per_well.values()]
        ),
        "warnings": qc_warnings,
    }

    normalized_column_modes = _normalize_crossmatch_column_modes(column_modes)

    if assay_type == "crossmatch":
        assay_result = build_crossmatch_result(
            sample_ids=sample_ids,
            sample_raw=sample_raw,
            sample_corr=sample_corr,
            positive_cutoff=positive_cutoff,
            borderline_low=borderline_low,
            borderline_high=borderline_high,
            max_replicate_range=max_replicate_range,
        )

        by_cell_mode: Dict[str, Dict[str, Any]] = {}

        for cell_mode in ("T", "B", "T/B"):
            columns = sorted(
                column
                for column, mode in normalized_column_modes.items()
                if mode == cell_mode
            )

            if not columns:
                continue

            mode_sample_ids = [
                well_id
                for well_id in sample_ids
                if _well_column(well_id) in columns
            ]
            mode_pc_ids = [
                well_id
                for well_id in pc_ids
                if _well_column(well_id) in columns
            ]
            mode_nc_ids = [
                well_id
                for well_id in nc_ids
                if _well_column(well_id) in columns
            ]

            mode_result = build_crossmatch_result(
                sample_ids=mode_sample_ids,
                sample_raw=[raw_frac(well_id) for well_id in mode_sample_ids],
                sample_corr=[corr_frac(well_id) for well_id in mode_sample_ids],
                positive_cutoff=positive_cutoff,
                borderline_low=borderline_low,
                borderline_high=borderline_high,
                max_replicate_range=max_replicate_range,
            )
            mode_run_validity = _build_run_validity_summary(
                pc_ids=mode_pc_ids,
                nc_ids=mode_nc_ids,
                raw_frac=raw_frac,
                min_dynamic_range=min_dynamic_range,
                max_replicate_range=max_replicate_range,
            )

            by_cell_mode[cell_mode] = {
                "cell_mode": cell_mode,
                "columns": columns,
                "run_validity": mode_run_validity,
                **mode_result,
            }

        assay_result["by_cell_mode"] = by_cell_mode
    else:
        assay_type = "pra"
        assay_result = build_pra_result(
            sample_ids=sample_ids,
            sample_corr=sample_corr,
            positive_cutoff=positive_cutoff,
            config=config,
        )

    return {
        "assay_type": assay_type,
        "run_validity": run_validity,
        "assay_result": assay_result,
        "qc": qc_summary,
        "column_modes": {
            str(column): mode
            for column, mode in normalized_column_modes.items()
        },
    }


def _roi_label_to_rgb(label: Any) -> Tuple[int, int, int]:
    """Map an ROI label to an RGB color.

    Parameters
    ----------
    label : Any
        ROI label.

    Returns
    -------
    tuple of int
        RGB color. Positive is orange, negative is green, and all other labels
        are blue.
    """
    label = (label or "").strip().lower()

    if label in {"pos", "positive"}:
        return (255, 165, 0)

    if label in {"neg", "negative"}:
        return (0, 170, 0)

    return (65, 105, 225)


def _safe_int(value: Any) -> Optional[int]:
    """Convert a value to int with a ``None`` fallback.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    int or None
        Converted integer, or ``None`` if conversion fails.
    """
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _roi_instance_id(roi: Any, fallback_id: int) -> int:
    """Return an instance ID from an ROI object or dictionary.

    Parameters
    ----------
    roi : Any
        ROI object or dictionary.
    fallback_id : int
        ID returned when no valid instance ID field is found.

    Returns
    -------
    int
        Positive instance ID.
    """
    keys = (
        "instance_id",
        "roi_id",
        "label_id",
        "object_id",
        "id",
        "instance",
    )

    if isinstance(roi, dict):
        for key in keys:
            value = _safe_int(roi.get(key))
            if value is not None and value > 0:
                return value
        return fallback_id

    for key in keys:
        value = _safe_int(getattr(roi, key, None))
        if value is not None and value > 0:
            return value

    return fallback_id


def save_segmented_preview(
    instance_labels: np.ndarray,
    rois: Any,
    out_path: str | Path,
    max_size: int = 900,
) -> None:
    """Save a color-coded segmentation preview with black ROI outlines.

    Parameters
    ----------
    instance_labels : numpy.ndarray
        Two-dimensional instance label image. Background is expected to be
        label ``0``.
    rois : iterable
        ROI dictionaries or objects containing labels and, when available,
        instance IDs.
    out_path : str or pathlib.Path
        Output path. The suffix is forced to ``.png``.
    max_size : int, optional
        Maximum output width or height in pixels. The default is ``900``.

    Raises
    ------
    ValueError
        If ``instance_labels`` is not two-dimensional.

    Notes
    -----
    Background is white. Positive ROIs are orange, negative ROIs are green, and
    uncertain or unknown ROIs are blue. Each ROI is outlined with a roughly
    2-pixel-wide black border in the final preview.
    """
    out_path = Path(out_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(instance_labels)

    if labels.ndim != 2:
        raise ValueError(f"instance_labels must be 2D, got shape {labels.shape}")

    labels = labels.astype(np.int32, copy=False)
    max_label = int(labels.max(initial=0))

    # White background by default
    lut = np.full((max_label + 1, 3), 255, dtype=np.uint8)

    # Keep your existing color coding
    for i, roi in enumerate(rois):
        inst_id = _roi_instance_id(roi, fallback_id=i + 1)

        if inst_id <= 0 or inst_id > max_label:
            continue

        roi_label = (
            roi.get("label")
            if isinstance(roi, dict)
            else getattr(roi, "label", None)
        )

        lut[inst_id] = _roi_label_to_rgb(roi_label)

    rgb = lut[labels]

    # Resize first, so the outline width is applied to the final preview size
    h, w = labels.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        rgb = np.array(
            Image.fromarray(rgb, mode="RGB").resize(
                (new_w, new_h),
                Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
            copy=True,
        )

        labels_for_border = np.asarray(
            Image.fromarray(labels.astype(np.int32), mode="I").resize(
                (new_w, new_h),
                Image.Resampling.NEAREST,
            )
        ).astype(np.int32, copy=False)
    else:
        labels_for_border = labels

    # Build a border mask from the instance labels
    # "inner" gives a 1 px border; one dilation step makes it about 2 px
    borders = find_boundaries(
        labels_for_border,
        connectivity=1,
        mode="inner",
        background=0,
    )

    # Paint only the border pixels black, keep all ROI fill colors
    rgb[borders] = (0, 0, 0)

    Image.fromarray(rgb, mode="RGB").save(out_path, compress_level=1)


def to_jsonable(obj: Any) -> Any:
    """Convert common NumPy objects into JSON-compatible Python objects.

    Parameters
    ----------
    obj : Any
        Input object.

    Returns
    -------
    Any
        Object converted to dictionaries, lists, Python scalars, or unchanged
        values when no conversion is needed.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def build_unet_by_mode(
    mode: str,
    builders: Dict[str, Callable[..., torch.nn.Module]],
) -> Callable[..., torch.nn.Module]:
    """Return a UNet builder for a given model size.

    Parameters
    ----------
    mode : str
        Model size key.
    builders : dict
        Mapping from model size keys to builder callables.

    Returns
    -------
    callable
        UNet builder.

    Raises
    ------
    ValueError
        If ``mode`` is not present in ``builders``.
    """
    try:
        return builders[mode]
    except KeyError as exc:
        valid = ", ".join(sorted(builders))
        raise ValueError(f"Unknown unet_mode: {mode}. Expected one of: {valid}") from exc


def to_chw_numpy(img: np.ndarray) -> np.ndarray:
    """Convert an image to ``(3, H, W)`` float32 in ``[0, 1]``.

    Parameters
    ----------
    img : numpy.ndarray
        Input image with shape ``(H, W, 3)``, ``(3, H, W)``,
        ``(1, 3, H, W)``, or ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        Contiguous image array with shape ``(3, H, W)`` and dtype
        ``float32``.

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a three-channel image.
    """
    if img.ndim == 3 and img.shape[0] == 3 and img.dtype == np.float32:
        return np.ascontiguousarray(img)

    if img.ndim == 4 and img.shape[0] == 1 and img.shape[1] == 3:
        img = img[0]

    if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image with 3 channels, got shape {img.shape}")

    x = img.astype(np.float32, copy=False)

    if x.max() > 1.0:
        x = x / 255.0

    x = np.transpose(x, (2, 0, 1))
    return np.ascontiguousarray(x, dtype=np.float32)


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Normalize an image tensor with UNet training statistics.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with shape ``(B, 3, H, W)``.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    mean = torch.as_tensor(
        UNET_MEAN,
        dtype=x.dtype,
        device=x.device,
    ).view(1, 3, 1, 1)

    std = torch.as_tensor(
        UNET_STD,
        dtype=x.dtype,
        device=x.device,
    ).view(1, 3, 1, 1)

    return (x - mean) / std


def to_tensor_tiles(
    tiles: np.ndarray,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert tile arrays to a torch tensor.

    Parameters
    ----------
    tiles : numpy.ndarray
        Tile batch with shape ``(T, 3, H, W)`` or ``(T, H, W, 3)``.
    device : torch.device
        Target torch device.
    normalize : bool, optional
        Whether to normalize with UNet training statistics. The default is
        ``True``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(T, 3, H, W)`` on ``device``.

    Raises
    ------
    ValueError
        If ``tiles`` is not a 4D array with three channels.
    """
    if tiles.ndim != 4:
        raise ValueError(f"Expected 4D tiles array, got shape {tiles.shape}")

    if tiles.shape[1] == 3:
        x = tiles.astype(np.float32, copy=False)
    elif tiles.shape[-1] == 3:
        x = np.transpose(tiles, (0, 3, 1, 2)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Expected tiles with 3 channels, got shape {tiles.shape}")

    if x.max() > 1.0:
        x = x / 255.0

    x = np.ascontiguousarray(x, dtype=np.float32)
    t = torch.from_numpy(x).to(device, non_blocking=True)

    if normalize:
        t = normalize_tensor(t)

    return t


def iter_sliding_windows(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield sliding-window coordinates for an image.

    Parameters
    ----------
    height : int
        Image height.
    width : int
        Image width.
    tile_size : int
        Tile size in pixels.
    overlap : int
        Overlap between neighboring tiles.

    Yields
    ------
    tuple of int
        Coordinates ``(y0, y1, x0, x1)``.

    Raises
    ------
    ValueError
        If ``overlap`` is greater than or equal to ``tile_size``.
    """
    stride = int(tile_size) - int(overlap)
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if ys[-1] + tile_size < height:
        ys.append(height - tile_size)

    xs = list(range(0, max(1, width - tile_size + 1), stride))
    if xs[-1] + tile_size < width:
        xs.append(width - tile_size)

    for y0 in ys:
        y1 = y0 + tile_size
        for x0 in xs:
            x1 = x0 + tile_size
            yield y0, y1, x0, x1


def tile_image_numpy(
    image_chw: np.ndarray,
    tile_size: int,
    overlap: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Split a CHW image into fixed-size tiles.

    Parameters
    ----------
    image_chw : numpy.ndarray
        Image with shape ``(3, H, W)``.
    tile_size : int
        Tile size in pixels.
    overlap : int
        Overlap between neighboring tiles.
    pad_value : float, optional
        Value used for bottom and right padding. The default is ``0.0``.

    Returns
    -------
    tiles : numpy.ndarray
        Tile batch with shape ``(T, 3, tile_size, tile_size)``.
    orig_hw : tuple of int
        Original image size as ``(H, W)``.

    Raises
    ------
    AssertionError
        If ``image_chw`` is not shaped ``(3, H, W)``.
    RuntimeError
        If no tiles are produced.
    """
    assert image_chw.ndim == 3 and image_chw.shape[0] == 3, "image must be [3, H, W]"

    _, height, width = image_chw.shape
    windows = list(iter_sliding_windows(height, width, tile_size, overlap))

    if not windows:
        raise RuntimeError("No tiles produced. Check tile_size, overlap, and image size.")

    tiles_arr = np.empty(
        (len(windows), 3, tile_size, tile_size),
        dtype=image_chw.dtype,
    )

    for idx, (y0, y1, x0, x1) in enumerate(windows):
        crop = image_chw[:, y0:y1, x0:x1]
        tile = tiles_arr[idx]
        tile.fill(pad_value)

        th, tw = crop.shape[1], crop.shape[2]
        tile[:, :th, :tw] = crop

    return tiles_arr, (height, width)


def reconstruct_from_tiles_probability(
    tiles: np.ndarray,
    orig_hw: Tuple[int, int],
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """Reconstruct full probability maps by averaging overlapping tiles.

    Parameters
    ----------
    tiles : numpy.ndarray
        Tile predictions with shape ``(T, C, tile_size, tile_size)``.
    orig_hw : tuple of int
        Original image size as ``(H, W)``.
    tile_size : int
        Tile size in pixels.
    overlap : int
        Overlap between neighboring tiles.

    Returns
    -------
    numpy.ndarray
        Reconstructed probability maps with shape ``(C, H, W)``.

    Raises
    ------
    AssertionError
        If ``tiles`` has the wrong shape.
    RuntimeError
        If the number of tiles does not match the tiling scheme.
    """
    assert tiles.ndim == 4, "tiles must be [N, C, tile_size, tile_size]"

    n_tiles, n_channels, tile_h, tile_w = tiles.shape
    assert tile_h == tile_size and tile_w == tile_size, "tile_size mismatch"

    height, width = orig_hw

    acc = np.zeros((n_channels, height, width), dtype=np.float32)
    acc_w = np.zeros((1, height, width), dtype=np.float32)

    idx = 0
    for y0, _, x0, _ in iter_sliding_windows(height, width, tile_size, overlap):
        if idx >= n_tiles:
            raise RuntimeError("Not enough tiles for the requested reconstruction.")

        th = min(tile_size, height - y0)
        tw = min(tile_size, width - x0)
        patch = tiles[idx, :, :th, :tw]

        acc[:, y0:y0 + th, x0:x0 + tw] += patch
        acc_w[:, y0:y0 + th, x0:x0 + tw] += 1.0

        idx += 1

    if idx != n_tiles:
        raise RuntimeError(
            f"Number of tiles ({n_tiles}) does not match tiling scheme ({idx})."
        )

    acc_w[acc_w == 0] = 1.0
    return (acc / acc_w).astype(np.float32)


def as_contiguous_f32(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert an array to contiguous float32.

    Parameters
    ----------
    x : numpy.ndarray or None
        Input array.

    Returns
    -------
    numpy.ndarray or None
        Contiguous ``float32`` array, or ``None`` if the input is ``None``.
    """
    if x is None:
        return None

    if x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]:
        return x

    return np.ascontiguousarray(x, dtype=np.float32)


def hysteresis_mask(
    p_cell: np.ndarray,
    low_thr: float,
    high_thr: float,
    close_selem: Optional[np.ndarray],
    min_hole_area: int,
    min_object_area: int,
) -> np.ndarray:
    """Create a binary cell mask using two-threshold hysteresis.

    Parameters
    ----------
    p_cell : numpy.ndarray
        Cell probability map with shape ``(H, W)``.
    low_thr : float
        Low probability threshold.
    high_thr : float
        High probability threshold.
    close_selem : numpy.ndarray or None
        Structuring element for binary closing.
    min_hole_area : int
        Minimum hole area retained.
    min_object_area : int
        Minimum object area retained.

    Returns
    -------
    numpy.ndarray
        Boolean cell mask.
    """
    strong = p_cell >= float(high_thr)
    weak = p_cell >= float(low_thr)

    lab = measure.label(weak, connectivity=1)

    if strong.any():
        strong_ids = np.unique(lab[strong])
        strong_ids = strong_ids[strong_ids != 0]

        if strong_ids.size > 0:
            lut = np.zeros(lab.max() + 1, dtype=bool)
            lut[strong_ids] = True
            mask = lut[lab]
        else:
            mask = np.zeros_like(weak, dtype=bool)
    else:
        mask = np.zeros_like(weak, dtype=bool)

    if close_selem is not None:
        mask = morphology.closing(mask, close_selem)

    mask = ndi.binary_fill_holes(mask)

    if min_hole_area > 0:
        mask = morphology.remove_small_holes(
            mask,
            area_threshold=int(min_hole_area),
        )

    if min_object_area > 0:
        mask = morphology.remove_small_objects(
            mask,
            min_size=int(min_object_area),
        )

    return mask.astype(bool, copy=False)


def smooth01(x: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth and clip a probability map.

    Parameters
    ----------
    x : numpy.ndarray
        Input probability map.
    sigma : float
        Gaussian smoothing sigma. If non-positive, the input is returned
        unchanged.

    Returns
    -------
    numpy.ndarray
        Smoothed probability map clipped to ``[0, 1]``.
    """
    if sigma and sigma > 0:
        y = ndi.gaussian_filter(x, float(sigma))
        return np.clip(y, 0.0, 1.0)

    return x


def make_markers(
    mask: np.ndarray,
    p_center: Optional[np.ndarray],
    dist_s: np.ndarray,
    use_centers: bool,
    center_seed_method: str,
    center_min_distance: int,
    center_thr: float,
) -> np.ndarray:
    """Create watershed markers from center probabilities or mask components.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean watershed mask.
    p_center : numpy.ndarray or None
        Center probability map.
    dist_s : numpy.ndarray
        Smoothed distance transform. The value is accepted to keep the call
        signature aligned with distance-based segmentation code.
    use_centers : bool
        Whether to use center probabilities for seed creation.
    center_seed_method : str
        Seed creation method. Supported values are ``"nms"`` and ``"thr"``.
    center_min_distance : int
        Minimum distance between local-maximum center seeds.
    center_thr : float
        Center probability threshold.

    Returns
    -------
    numpy.ndarray
        Integer marker image with shape matching ``mask``.
    """
    _ = dist_s

    work_mask = mask if mask.dtype == bool else mask.astype(bool, copy=False)
    seeds_bool = np.zeros_like(work_mask, dtype=bool)

    if use_centers and p_center is not None:
        if center_seed_method == "nms":
            coords = feature.peak_local_max(
                p_center,
                min_distance=int(center_min_distance),
                threshold_abs=float(center_thr),
                labels=work_mask,
                exclude_border=False,
            )

            if coords.size:
                seeds_bool[tuple(coords.T)] = True
        else:
            seeds_bool |= (p_center >= float(center_thr)) & work_mask

    markers = measure.label(seeds_bool, connectivity=1).astype(np.int32)

    if markers.max() == 0:
        markers = measure.label(work_mask, connectivity=1).astype(np.int32)

    return markers

def add_roi_borders(
    preview: np.ndarray,
    instance_labels: np.ndarray,
) -> np.ndarray:
    """
    Add a two-pixel black border around every labeled ROI.

    Boundaries between touching instances are also marked.
    """
    output = preview.copy()

    labels = np.asarray(instance_labels)

    if labels.shape != output.shape[:2]:
        raise ValueError(
            "Instance labels and preview have different image dimensions."
        )

    # mode="thick" creates a boundary approximately two pixels wide.
    borders = find_boundaries(
        labels,
        connectivity=1,
        mode="thick",
        background=0,
    )

    if output.ndim == 2:
        output[borders] = 0

    elif output.ndim == 3 and output.shape[2] >= 3:
        output[borders, :3] = 0

        # Keep the black border fully visible for RGBA previews.
        if output.shape[2] == 4:
            if np.issubdtype(output.dtype, np.integer):
                output[borders, 3] = np.iinfo(output.dtype).max
            else:
                output[borders, 3] = 1.0

    else:
        raise ValueError(
            f"Unsupported preview shape: {output.shape}"
        )

    return output
