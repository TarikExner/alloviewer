import numpy as np
from pathlib import Path
from typing import Any, List

from PIL import Image

from .structs import Plate, PlateLayout, WellImage, WellResult


PRA_GENERIC_LAYOUT = PlateLayout(
    wells={
        'A1': 'negative', 'A2': 'sample',  'A3': 'sample',  'A4': 'sample',  'A5': 'sample',
        'A6': 'sample',   'A7': 'sample',  'A8': 'sample',  'A9': 'sample',  'A10': 'positive',

        'B1': 'negative', 'B2': 'sample',  'B3': 'sample',  'B4': 'sample',  'B5': 'sample',
        'B6': 'sample',   'B7': 'sample',  'B8': 'sample',  'B9': 'sample',  'B10': 'positive',

        'C1': 'sample',   'C2': 'sample',  'C3': 'sample',  'C4': 'sample',  'C5': 'sample',
        'C6': 'sample',   'C7': 'sample',  'C8': 'sample',  'C9': 'sample',  'C10': 'sample',

        'D1': 'sample',   'D2': 'sample',  'D3': 'sample',  'D4': 'sample',  'D5': 'sample',
        'D6': 'sample',   'D7': 'sample',  'D8': 'sample',  'D9': 'sample',  'D10': 'sample',

        'E1': 'sample',   'E2': 'sample',  'E3': 'sample',  'E4': 'sample',  'E5': 'sample',
        'E6': 'sample',   'E7': 'sample',  'E8': 'sample',  'E9': 'sample',  'E10': 'sample',

        'F1': 'sample',   'F2': 'sample',  'F3': 'sample',  'F4': 'sample',  'F5': 'sample',
        'F6': 'sample',   'F7': 'sample',  'F8': 'sample',  'F9': 'sample',  'F10': 'sample',
    }
)

PRA_GENERIC_IMAGE_ORDER=[
    'A1', 'B1', 'C1', 'D1', 'E1', 'F1',
    'F2', 'E2', 'D2', 'C2', 'B2', 'A2',
    'A3', 'B3', 'C3', 'D3', 'E3', 'F3',
    'F4', 'E4', 'D4', 'C4', 'B4', 'A4',
    'A5', 'B5', 'C5', 'D5', 'E5', 'F5',
    'F6', 'E6', 'D6', 'C6', 'B6', 'A6',
    'A7', 'B7', 'C7', 'D7', 'E7', 'F7',
    'F8', 'E8', 'D8', 'C8', 'B8', 'A8',
    'A9', 'B9', 'C9', 'D9', 'E9', 'F9',
    'F10', 'E10', 'D10', 'C10', 'B10', 'A10',
]


def create_plate(
    layout: PlateLayout,
    images: List[np.ndarray],
    image_order: List[str],
    image_paths: List[str],
) -> Plate:
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
    """
    Raw fraction positive in percent.

    Only classified positive and negative ROIs are used in the denominator.
    Uncertain ROIs are ignored here.
    """
    n_pos = sum(1 for r in wr.rois if r.label == "pos")
    n_neg = sum(1 for r in wr.rois if r.label == "neg")
    n_total = n_pos + n_neg

    if n_total == 0:
        return np.nan

    return 100.0 * (n_pos / n_total)


def convert_frac_pos_to_score(frac_pos: int) -> int:
    if frac_pos <= 10:
        return 1
    if frac_pos <= 20:
        return 2
    if frac_pos <= 50:
        return 4
    if frac_pos <= 80:
        return 6

    return 8


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _mean_or_nan(xs: list[float]) -> float:
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _median_or_nan(xs: list[float]) -> float:
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.median(xs)) if xs else float("nan")


def _sd_or_nan(xs: list[float]) -> float:
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")


def _range_or_nan(xs: list[float]) -> float:
    xs = [x for x in xs if not np.isnan(x)]
    return float(max(xs) - min(xs)) if xs else float("nan")


def _roi_label_counts(wr: WellResult) -> dict[str, int]:
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
    counts = _roi_label_counts(wr)

    if counts["n_total"] == 0:
        return float("nan")

    return counts["n_uncertain"] / counts["n_total"]


def build_pra_result(
    sample_ids: list[str],
    sample_corr: list[float],
    positive_cutoff: float,
    config: dict,
) -> dict[str, Any]:
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
    if np.isnan(value):
        return "not_available"

    if value < borderline_low:
        return "negative"

    if value <= borderline_high:
        return "borderline"

    return "positive"


def build_crossmatch_result(
    sample_ids: list[str],
    sample_raw: list[float],
    sample_corr: list[float],
    positive_cutoff: float,
    borderline_low: float,
    borderline_high: float,
    max_replicate_range: float,
) -> dict[str, Any]:
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


def build_cdc_summary(
    per_well: dict[str, WellResult],
    plate: Plate,
    config: dict,
    assay_type: str = "pra",
) -> dict[str, Any]:
    positive_cutoff = float(config["positive_cutoff"])
    borderline_low = float(config["borderline_low"])
    borderline_high = float(config["borderline_high"])
    min_rois = int(config["min_rois"])
    max_uncertain_fraction = float(config["max_uncertain_fraction"])
    min_dynamic_range = float(config["min_dynamic_range"])
    max_replicate_range = float(config["max_replicate_range"])

    pc_ids = [w.well_id for w in plate.get("positive")]
    nc_ids = [w.well_id for w in plate.get("negative")]
    sample_ids = [w.well_id for w in plate.get("sample")]

    def raw_frac(wid: str) -> float:
        return frac_pos_raw(per_well[wid]) if wid in per_well else float("nan")

    def corr_frac(wid: str) -> float:
        if wid not in per_well:
            return float("nan")
        return _safe_float(per_well[wid].corrected_frac_pos)

    pc_raw = [raw_frac(wid) for wid in pc_ids]
    nc_raw = [raw_frac(wid) for wid in nc_ids]
    sample_raw = [raw_frac(wid) for wid in sample_ids]
    sample_corr = [corr_frac(wid) for wid in sample_ids]

    pc_mean = _mean_or_nan(pc_raw)
    nc_mean = _mean_or_nan(nc_raw)

    dynamic_range = (
        pc_mean - nc_mean
        if not np.isnan(pc_mean) and not np.isnan(nc_mean)
        else float("nan")
    )

    all_ids = list(per_well.keys())

    low_roi_wells = []
    high_uncertain_wells = []

    for wid, wr in per_well.items():
        counts = _roi_label_counts(wr)
        unc_frac = _uncertain_fraction(wr)

        if counts["n_total"] < min_rois:
            low_roi_wells.append(wid)

        if not np.isnan(unc_frac) and unc_frac > max_uncertain_fraction:
            high_uncertain_wells.append(wid)

    control_warnings = []

    if len(pc_ids) == 0:
        control_warnings.append("No positive control wells.")

    if len(nc_ids) == 0:
        control_warnings.append("No negative control wells.")

    if np.isnan(dynamic_range) or dynamic_range < min_dynamic_range:
        control_warnings.append("Poor positive/negative control separation.")

    pc_range = _range_or_nan(pc_raw)
    nc_range = _range_or_nan(nc_raw)

    if not np.isnan(pc_range) and pc_range > max_replicate_range:
        control_warnings.append("Positive control replicates differ strongly.")

    if not np.isnan(nc_range) and nc_range > max_replicate_range:
        control_warnings.append("Negative control replicates differ strongly.")

    qc_warnings = []

    if low_roi_wells:
        qc_warnings.append(f"{len(low_roi_wells)} well(s) have low ROI count.")

    if high_uncertain_wells:
        qc_warnings.append(
            f"{len(high_uncertain_wells)} well(s) have high uncertain fraction."
        )

    run_status = "valid"

    if control_warnings:
        run_status = "invalid"
    elif qc_warnings:
        run_status = "warning"

    run_validity = {
        "status": run_status,
        "pc_mean_raw": pc_mean,
        "nc_mean_raw": nc_mean,
        "dynamic_range": dynamic_range,
        "pc_replicate_range": pc_range,
        "nc_replicate_range": nc_range,
        "n_positive_controls": len(pc_ids),
        "n_negative_controls": len(nc_ids),
        "control_warnings": control_warnings,
    }

    qc_summary = {
        "total_wells": len(all_ids),
        "valid_wells": len(all_ids) - len(low_roi_wells),
        "low_roi_wells": low_roi_wells,
        "high_uncertain_wells": high_uncertain_wells,
        "mean_n_rois": _mean_or_nan(
            [float(len(wr.rois)) for wr in per_well.values()]
        ),
        "mean_uncertain_fraction": _mean_or_nan(
            [_uncertain_fraction(wr) for wr in per_well.values()]
        ),
        "warnings": qc_warnings,
    }

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
    }


def _roi_label_to_rgb(label) -> tuple[int, int, int]:
    label = (label or "").strip().lower()

    if label in {"pos", "positive"}:
        return (255, 165, 0)

    if label in {"neg", "negative"}:
        return (0, 170, 0)

    return (65, 105, 225)


def _safe_int(value) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _roi_instance_id(roi, fallback_id: int) -> int:
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
    rois,
    out_path: str | Path,
    max_size: int = 900,
) -> None:
    """
    Save a classified segmentation preview.

    Background: white.
    Positive ROIs: orange.
    Negative ROIs: green.
    Uncertain/unknown ROIs: blue.
    """
    out_path = Path(out_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(instance_labels)

    if labels.ndim != 2:
        raise ValueError(f"instance_labels must be 2D, got shape {labels.shape}")

    labels = labels.astype(np.int32, copy=False)
    max_label = int(labels.max(initial=0))

    lut = np.full((max_label + 1, 3), 255, dtype=np.uint8)

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

    pil_img = Image.fromarray(rgb, mode="RGB")
    pil_img.thumbnail((max_size, max_size), Image.Resampling.NEAREST)
    pil_img.save(out_path, compress_level=1)


def to_jsonable(obj):
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
