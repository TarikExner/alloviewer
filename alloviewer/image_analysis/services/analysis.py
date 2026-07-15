from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Iterable

import numpy as np

from ..structs import ParsedPlateLayout, WellResult


@dataclass
class AlleleReactivityEvidence:
    allele_key: str
    locus: str
    allele: str

    positive_well_count: int
    total_well_count: int
    negative_well_count: int

    positive_fraction: float
    positive_ratio: str

    positive_wells: list[str]
    negative_wells: list[str]
    missing_result_wells: list[str]

    well_values: dict[str, Optional[float]]


@dataclass
class PraReactivityScore:
    positive_well_count: int
    total_well_count: int
    positive_fraction: float
    score_percent: float
    threshold: float
    positive_wells: list[str]
    negative_wells: list[str]
    missing_result_wells: list[str]


def _allele_key(locus: str, allele: str) -> str:
    return f"{locus}:{allele}"


def _normalize_well_set(well_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if well_ids is None:
        return None

    return {str(well_id).upper() for well_id in well_ids}


def _well_positive_value(wr: WellResult | Mapping[str, Any]) -> Optional[float]:
    if isinstance(wr, Mapping):
        value = wr.get("frac_pos_corrected")

        if value is None:
            value = wr.get("corrected_frac_pos")
    else:
        value = getattr(wr, "corrected_frac_pos", None)

    if value is None:
        return None

    value = float(value)

    if np.isnan(value):
        return None

    return value


def _well_effective_call(
    well_id: str,
    wr: WellResult | Mapping[str, Any],
    effective_calls: Optional[Mapping[str, str]],
) -> Optional[str]:
    normalized_well_id = str(well_id).upper()

    if effective_calls is not None:
        call = effective_calls.get(normalized_well_id)

        if call is None:
            call = effective_calls.get(str(well_id))

        if call in {"positive", "negative"}:
            return str(call)

    if isinstance(wr, Mapping):
        call = wr.get("effective_call")

        if call in {"positive", "negative"}:
            return str(call)

        manual_override = wr.get("manual_override")

        if isinstance(manual_override, Mapping):
            call = manual_override.get("call")

            if call in {"positive", "negative"}:
                return str(call)

    return None


def _well_is_positive(
    well_id: str,
    wr: WellResult | Mapping[str, Any],
    positivity_threshold: float,
    effective_calls: Optional[Mapping[str, str]],
) -> Optional[bool]:
    call = _well_effective_call(well_id, wr, effective_calls)

    if call == "positive":
        return True

    if call == "negative":
        return False

    value = _well_positive_value(wr)

    if value is None:
        return None

    return value >= positivity_threshold


def calculate_allele_reactivity_evidence(
    per_well: Mapping[str, WellResult],
    hla_layout: ParsedPlateLayout,
    positivity_threshold: float,
    include_well_ids: Optional[Iterable[str]] = None,
    effective_calls: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    """Report per-allele PRA reactivity.

    Only wells in include_well_ids are considered when provided.
    For CDC PRA, pass sample wells only. Positive and negative controls should
    not contribute to allele evidence.
    """
    included = _normalize_well_set(include_well_ids)

    allele_to_wells: dict[str, dict[str, Any]] = {}

    for well_id, well_layout in hla_layout.wells.items():
        normalized_well_id = well_id.upper()

        if included is not None and normalized_well_id not in included:
            continue

        for locus, alleles in well_layout.loci.data.items():
            for allele in alleles:
                allele_key = _allele_key(locus, allele)

                if allele_key not in allele_to_wells:
                    allele_to_wells[allele_key] = {
                        "locus": locus,
                        "allele": allele,
                        "wells": [],
                    }

                allele_to_wells[allele_key]["wells"].append(normalized_well_id)

    result_by_well = {
        well_id.upper(): wr
        for well_id, wr in per_well.items()
    }

    evidence: list[AlleleReactivityEvidence] = []

    for allele_key, item in allele_to_wells.items():
        carrier_wells = sorted(set(item["wells"]))

        positive_wells: list[str] = []
        negative_wells: list[str] = []
        missing_result_wells: list[str] = []
        well_values: dict[str, Optional[float]] = {}

        for well_id in carrier_wells:
            wr = result_by_well.get(well_id)

            if wr is None:
                missing_result_wells.append(well_id)
                well_values[well_id] = None
                continue

            value = _well_positive_value(wr)
            well_values[well_id] = value

            is_positive = _well_is_positive(
                well_id,
                wr,
                positivity_threshold,
                effective_calls,
            )

            if is_positive is None:
                missing_result_wells.append(well_id)
            elif is_positive:
                positive_wells.append(well_id)
            else:
                negative_wells.append(well_id)

        total_well_count = len(carrier_wells)
        positive_well_count = len(positive_wells)
        negative_well_count = len(negative_wells)

        positive_fraction = (
            positive_well_count / total_well_count
            if total_well_count > 0
            else 0.0
        )

        evidence.append(
            AlleleReactivityEvidence(
                allele_key=allele_key,
                locus=item["locus"],
                allele=item["allele"],
                positive_well_count=positive_well_count,
                total_well_count=total_well_count,
                negative_well_count=negative_well_count,
                positive_fraction=positive_fraction,
                positive_ratio=f"{positive_well_count}/{total_well_count}",
                positive_wells=positive_wells,
                negative_wells=negative_wells,
                missing_result_wells=missing_result_wells,
                well_values=well_values,
            )
        )

    evidence.sort(
        key=lambda e: (
            e.positive_well_count,
            e.positive_fraction,
            e.total_well_count,
        ),
        reverse=True,
    )

    return [asdict(e) for e in evidence]


def calculate_pra_reactivity_score(
    per_well: Mapping[str, WellResult],
    hla_layout: ParsedPlateLayout,
    positivity_threshold: float,
    include_well_ids: Optional[Iterable[str]] = None,
    effective_calls: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Calculate temporary overall PRA reactivity.

    Current rule:

        positive tested sample HLA wells / all tested sample HLA wells

    Positive and negative controls are excluded when include_well_ids contains
    sample wells only.
    """
    included = _normalize_well_set(include_well_ids)

    hla_well_ids = {
        well_id.upper()
        for well_id, well_layout in hla_layout.wells.items()
        if well_layout.loci.data
        and (included is None or well_id.upper() in included)
    }

    result_by_well = {
        well_id.upper(): wr
        for well_id, wr in per_well.items()
    }

    positive_wells: list[str] = []
    negative_wells: list[str] = []
    missing_result_wells: list[str] = []

    for well_id in sorted(hla_well_ids):
        wr = result_by_well.get(well_id)

        if wr is None:
            missing_result_wells.append(well_id)
            continue

        is_positive = _well_is_positive(
            well_id,
            wr,
            positivity_threshold,
            effective_calls,
        )

        if is_positive is None:
            missing_result_wells.append(well_id)
        elif is_positive:
            positive_wells.append(well_id)
        else:
            negative_wells.append(well_id)

    total_well_count = len(hla_well_ids)
    positive_well_count = len(positive_wells)

    positive_fraction = (
        positive_well_count / total_well_count
        if total_well_count > 0
        else 0.0
    )

    score = PraReactivityScore(
        positive_well_count=positive_well_count,
        total_well_count=total_well_count,
        positive_fraction=positive_fraction,
        score_percent=positive_fraction * 100.0,
        threshold=positivity_threshold,
        positive_wells=positive_wells,
        negative_wells=negative_wells,
        missing_result_wells=missing_result_wells,
    )

    return asdict(score)
