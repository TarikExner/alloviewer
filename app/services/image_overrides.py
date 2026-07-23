from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from alloviewer.image_analysis.config import CDC_SUMMARY_CONFIG
from alloviewer.image_analysis.services.analysis import (
    calculate_allele_reactivity_evidence,
    calculate_pra_reactivity_score,
)
from alloviewer.image_analysis.structs import parsed_plate_from_dict
from alloviewer.image_analysis.utils import (
    automated_well_call,
    build_crossmatch_result,
    build_pra_result,
    is_borderline_value,
)

from app.services.job_paths import get_job_paths
from app.services.job_registry import read_json


ALLOWED_MANUAL_CALLS = {"positive", "negative"}
ALLOWED_EFFECTIVE_CALLS = {
    "positive",
    "negative",
    "not_available",
}


def _finite_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")

    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")

    return number if np.isfinite(number) else float("nan")


def _corrected_fraction(well: Mapping[str, Any]) -> float:
    value = well.get("frac_pos_corrected")

    if value is None:
        value = well.get("corrected_frac_pos")

    return _finite_or_nan(value)


def _raw_fraction(well: Mapping[str, Any]) -> float:
    return _finite_or_nan(well.get("frac_pos"))


def _well_column(well_id: str) -> int | None:
    digits = ""

    for character in reversed(str(well_id)):
        if not character.isdigit():
            break
        digits = character + digits

    return int(digits) if digits else None


def _normalize_column_modes(value: Any) -> dict[int, str]:
    if not isinstance(value, Mapping):
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


def _load_pra_threshold(job_id: str) -> float:
    request_path = get_job_paths(job_id).request

    if not request_path.exists():
        return float(CDC_SUMMARY_CONFIG["positive_cutoff"])

    request = read_json(request_path)

    if not isinstance(request, Mapping):
        return float(CDC_SUMMARY_CONFIG["positive_cutoff"])

    return float(
        request.get(
            "pra_positivity_threshold",
            CDC_SUMMARY_CONFIG["positive_cutoff"],
        )
    )


def _sample_well_ids(wells: Mapping[str, Any]) -> list[str]:
    return [
        str(well_id)
        for well_id, well in wells.items()
        if isinstance(well, Mapping)
        and str(well.get("role") or "").lower() == "sample"
    ]


def _manual_override_wells(result: Mapping[str, Any]) -> set[str]:
    overrides = result.get("manual_overrides")

    if not isinstance(overrides, Mapping):
        return set()

    return {
        str(well_id).upper()
        for well_id, override in overrides.items()
        if isinstance(override, Mapping)
        and override.get("call") in ALLOWED_MANUAL_CALLS
    }


def _effective_calls(wells: Mapping[str, Any]) -> dict[str, str]:
    calls: dict[str, str] = {}

    for well_id, well in wells.items():
        if not isinstance(well, Mapping):
            continue

        call = well.get("effective_call")

        if call in ALLOWED_EFFECTIVE_CALLS:
            calls[str(well_id).upper()] = str(call)

    return calls


def _synchronize_well_calls(
    *,
    result: dict[str, Any],
    pra_threshold: float,
) -> None:
    assay_type = str(result.get("assay_type") or "pra").lower()
    wells = result.get("wells")

    if not isinstance(wells, dict):
        raise ValueError("The analysis result does not contain well results.")

    normalized_overrides: dict[str, dict[str, Any]] = {}
    existing_overrides = result.get("manual_overrides")

    if not isinstance(existing_overrides, Mapping):
        existing_overrides = {}

    for well_id, well in wells.items():
        if not isinstance(well, dict):
            continue

        corrected_fraction = _corrected_fraction(well)
        automated_call = automated_well_call(
            corrected_fraction,
            assay_type=assay_type,
            pra_positive_cutoff=pra_threshold,
            config=CDC_SUMMARY_CONFIG,
        )

        well["automated_call"] = automated_call
        well["borderline"] = is_borderline_value(
            corrected_fraction,
            borderline_low=float(CDC_SUMMARY_CONFIG["borderline_low"]),
            borderline_high=float(CDC_SUMMARY_CONFIG["borderline_high"]),
        )
        override = existing_overrides.get(well_id)

        if override is None:
            override = existing_overrides.get(str(well_id).upper())

        if (
            isinstance(override, Mapping)
            and override.get("call") in ALLOWED_MANUAL_CALLS
        ):
            normalized_override = {
                "call": str(override["call"]),
                "automated_call": str(automated_call),
                "created_at": str(
                    override.get("created_at")
                    or datetime.now(timezone.utc).isoformat()
                ),
            }
            normalized_overrides[str(well_id)] = normalized_override
            well["manual_override"] = normalized_override
            well["effective_call"] = normalized_override["call"]
        else:
            well["manual_override"] = None
            well["effective_call"] = automated_call

    result["manual_overrides"] = normalized_overrides


def _recalculate_pra(
    *,
    job_id: str,
    result: dict[str, Any],
    pra_threshold: float,
) -> None:
    wells = result["wells"]
    sample_ids = _sample_well_ids(wells)
    corrected = [_corrected_fraction(wells[well_id]) for well_id in sample_ids]
    calls = _effective_calls(wells)
    override_wells = _manual_override_wells(result)
    summary = dict(result.get("summary") or {})
    summary["assay_result"] = build_pra_result(
        sample_ids=sample_ids,
        sample_corr=corrected,
        positive_cutoff=pra_threshold,
        config=CDC_SUMMARY_CONFIG,
        effective_calls=calls,
        manual_override_wells=override_wells,
    )
    result["summary"] = summary

    paths = get_job_paths(job_id)

    if not paths.plate_layout.exists():
        raise FileNotFoundError(
            "The parsed PRA plate-layout file is missing; override recalculation cannot continue."
        )

    layout_data = read_json(paths.plate_layout)

    if not isinstance(layout_data, dict):
        raise ValueError("The persisted PRA plate layout is invalid.")

    hla_layout = parsed_plate_from_dict(layout_data)
    included_wells = {well_id.upper() for well_id in sample_ids}
    result["pra_analysis"] = {
        "positivity_threshold": pra_threshold,
        "included_well_type": "sample",
        "included_wells": sorted(included_wells),
        "reactivity_score": calculate_pra_reactivity_score(
            per_well=wells,
            hla_layout=hla_layout,
            positivity_threshold=pra_threshold,
            include_well_ids=included_wells,
            effective_calls=calls,
        ),
        "alleles": calculate_allele_reactivity_evidence(
            per_well=wells,
            hla_layout=hla_layout,
            positivity_threshold=pra_threshold,
            include_well_ids=included_wells,
            effective_calls=calls,
        ),
    }


def _recalculate_crossmatch(result: dict[str, Any]) -> None:
    wells = result["wells"]
    sample_ids = _sample_well_ids(wells)
    raw = [_raw_fraction(wells[well_id]) for well_id in sample_ids]
    corrected = [_corrected_fraction(wells[well_id]) for well_id in sample_ids]
    calls = _effective_calls(wells)
    override_wells = _manual_override_wells(result)
    config = CDC_SUMMARY_CONFIG
    summary = dict(result.get("summary") or {})
    previous_assay = dict(summary.get("assay_result") or {})
    previous_by_mode = previous_assay.get("by_cell_mode") or {}
    assay_result = build_crossmatch_result(
        sample_ids=sample_ids,
        sample_raw=raw,
        sample_corr=corrected,
        positive_cutoff=float(config["positive_cutoff"]),
        borderline_low=float(config["borderline_low"]),
        borderline_high=float(config["borderline_high"]),
        max_replicate_range=float(config["max_replicate_range"]),
        effective_calls=calls,
        manual_override_wells=override_wells,
    )
    column_modes = _normalize_column_modes(
        result.get("column_modes") or summary.get("column_modes")
    )
    by_cell_mode: dict[str, dict[str, Any]] = {}

    for cell_mode in ("T", "B", "T/B"):
        columns = sorted(
            column
            for column, mode in column_modes.items()
            if mode == cell_mode
        )

        if not columns:
            continue

        mode_sample_ids = [
            well_id
            for well_id in sample_ids
            if _well_column(well_id) in columns
        ]
        mode_result = build_crossmatch_result(
            sample_ids=mode_sample_ids,
            sample_raw=[_raw_fraction(wells[well_id]) for well_id in mode_sample_ids],
            sample_corr=[
                _corrected_fraction(wells[well_id])
                for well_id in mode_sample_ids
            ],
            positive_cutoff=float(config["positive_cutoff"]),
            borderline_low=float(config["borderline_low"]),
            borderline_high=float(config["borderline_high"]),
            max_replicate_range=float(config["max_replicate_range"]),
            effective_calls=calls,
            manual_override_wells=override_wells,
        )
        previous_mode = (
            previous_by_mode.get(cell_mode, {})
            if isinstance(previous_by_mode, Mapping)
            else {}
        )
        by_cell_mode[cell_mode] = {
            "cell_mode": cell_mode,
            "columns": columns,
            "run_validity": dict(previous_mode.get("run_validity") or {}),
            **mode_result,
        }

    assay_result["by_cell_mode"] = by_cell_mode
    summary["assay_result"] = assay_result
    result["summary"] = summary


def recalculate_after_manual_overrides(
    *,
    job_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(result)
    pra_threshold = _load_pra_threshold(job_id)
    _synchronize_well_calls(
        result=updated,
        pra_threshold=pra_threshold,
    )
    override_wells = sorted(_manual_override_wells(updated))
    summary = dict(updated.get("summary") or {})
    summary["manual_override_applied"] = bool(override_wells)
    summary["manual_override_count"] = len(override_wells)
    summary["manual_override_wells"] = override_wells
    updated["summary"] = summary

    assay_type = str(updated.get("assay_type") or "pra").lower()

    if assay_type == "pra":
        _recalculate_pra(
            job_id=job_id,
            result=updated,
            pra_threshold=pra_threshold,
        )
    elif assay_type == "crossmatch":
        _recalculate_crossmatch(updated)
    else:
        raise ValueError(f"Unsupported image-analysis assay type: {assay_type}")

    return updated


def apply_well_classification_override(
    *,
    job_id: str,
    result: dict[str, Any],
    well_id: str,
    call: str | None,
) -> dict[str, Any]:
    if call is not None and call not in ALLOWED_MANUAL_CALLS:
        raise ValueError(f"Unsupported manual well call: {call}")

    updated = deepcopy(result)
    wells = updated.get("wells")

    if not isinstance(wells, dict):
        raise ValueError("The analysis result does not contain well results.")

    normalized_well_id = str(well_id).upper()
    actual_well_id = next(
        (
            candidate
            for candidate in wells
            if str(candidate).upper() == normalized_well_id
        ),
        None,
    )

    if actual_well_id is None:
        raise KeyError(f"Well not found in analysis result: {well_id}")

    well = wells[actual_well_id]

    if not isinstance(well, dict):
        raise ValueError(f"Well result is invalid: {actual_well_id}")

    if np.isnan(_corrected_fraction(well)):
        raise ValueError(
            "This well has no calibrated fraction positive and cannot be manually classified."
        )

    pra_threshold = _load_pra_threshold(job_id)
    _synchronize_well_calls(
        result=updated,
        pra_threshold=pra_threshold,
    )
    well = updated["wells"][actual_well_id]
    automated_call = str(well.get("automated_call") or "not_available")
    previous_override = well.get("manual_override")
    previous_call = (
        str(previous_override.get("call"))
        if isinstance(previous_override, Mapping)
        and previous_override.get("call") in ALLOWED_MANUAL_CALLS
        else None
    )
    created_at = datetime.now(timezone.utc).isoformat()
    overrides = dict(updated.get("manual_overrides") or {})

    if call is None:
        overrides.pop(actual_well_id, None)
        well["manual_override"] = None
        well["effective_call"] = automated_call
        action = "cleared"
    else:
        override = {
            "call": call,
            "automated_call": automated_call,
            "created_at": created_at,
        }
        overrides[actual_well_id] = override
        well["manual_override"] = override
        well["effective_call"] = call
        action = "set"

    updated["manual_overrides"] = overrides
    history = list(updated.get("manual_override_history") or [])
    history.append(
        {
            "well_id": actual_well_id,
            "action": action,
            "previous_call": previous_call,
            "new_call": call,
            "automated_call": automated_call,
            "created_at": created_at,
        }
    )
    updated["manual_override_history"] = history

    return recalculate_after_manual_overrides(
        job_id=job_id,
        result=updated,
    )
