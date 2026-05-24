from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

from ..structs import ParsedPlateLayout, WellResult


@dataclass
class AlleleReactivity:
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


def _well_positive_value(wr: WellResult) -> Optional[float]:
    """Return the value used for PRA positivity.

    Prefer corrected_frac_pos because the pipeline already calibrates each well
    against positive and negative controls. If it is missing or NaN, return None.
    """
    value = getattr(wr, "corrected_frac_pos", None)

    if value is None:
        return None

    value = float(value)

    if np.isnan(value):
        return None

    return value


def calculate_allele_reactivity_evidence(
    per_well: Mapping[str, WellResult],
    hla_layout: ParsedPlateLayout,
    positivity_threshold: float,
) -> list[dict[str, Any]]:
    """Calculate per-allele PRA reactivity.

    For each HLA allele in the parsed Excel layout, this counts:

    positive wells containing allele / all wells containing allele

    Example:
        B:27 appears in 3 wells.
        1 of those wells is positive.
        Result: B:27 = 1/3.

    This function does not claim antibody fidelity. It reports supporting
    positive carrier wells and negative carrier wells.
    """
    allele_to_wells: dict[str, dict[str, Any]] = {}

    for well_id, well_layout in hla_layout.wells.items():
        normalized_well_id = well_id.upper()

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

    evidence: list[AlleleReactivity] = []

    # Normalize result well IDs once. The image pipeline and Excel parser should
    # both use IDs like A1, B12, etc., but this avoids case errors.
    result_by_well = {
        well_id.upper(): wr
        for well_id, wr in per_well.items()
    }

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

            if value is None:
                missing_result_wells.append(well_id)
            elif value >= positivity_threshold:
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
            AlleleReactivity(
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
) -> dict[str, Any]:
    """Calculate a simple overall PRA reactivity score.

    This is intentionally simple for now:

        positive tested HLA wells / all tested HLA wells

    Replace this later once the assay rule is clear.
    """
    hla_well_ids = {
        well_id.upper()
        for well_id, well_layout in hla_layout.wells.items()
        if well_layout.loci.data
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

        value = _well_positive_value(wr)

        if value is None:
            missing_result_wells.append(well_id)
        elif value >= positivity_threshold:
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
