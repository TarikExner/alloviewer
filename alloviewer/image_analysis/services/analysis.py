from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Iterable

import numpy as np

from ..structs import ParsedPlateLayout, WellResult


@dataclass
class AlleleReactivityEvidence:
    """
    Store measured positive and negative well evidence for one HLA allele.

    Attributes
    ----------
    allele_key : str
        Combined locus and allele identifier separated by a colon.
    locus : str
        HLA locus name.
    allele : str
        HLA allele identifier.
    positive_well_count : int
        Number of included wells classified as positive.
    total_well_count : int
        Number of included wells, including wells with missing results.
    negative_well_count : int
        Number of included wells classified as negative.
    positive_fraction : float
        Positive-well count divided by the total included-well count; zero when that total
        is zero.
    positive_ratio : str
        Formatted ratio of positive wells to total included wells.
    positive_wells : list[str]
        Identifiers of included wells classified as positive.
    negative_wells : list[str]
        Identifiers of included wells classified as negative.
    missing_result_wells : list[str]
        Identifiers of included wells without a usable result or effective call.
    well_values : dict[str, Optional[float]]
        Mapping from well identifiers to corrected positive percentages, or None when
        unavailable.
    """
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
    """
    Store overall PRA reactivity counts, fractions, and missing-well information.

    Attributes
    ----------
    positive_well_count : int
        Number of included wells classified as positive.
    total_well_count : int
        Number of included wells, including wells with missing results.
    positive_fraction : float
        Positive-well count divided by the total included-well count; zero when that total
        is zero.
    score_percent : float
        Percentage of included wells classified as positive.
    threshold : float
        Corrected positive-percentage cutoff used when an effective well call is
        unavailable.
    positive_wells : list[str]
        Identifiers of included wells classified as positive.
    negative_wells : list[str]
        Identifiers of included wells classified as negative.
    missing_result_wells : list[str]
        Identifiers of included wells without a usable result or effective call.
    """
    positive_well_count: int
    total_well_count: int
    positive_fraction: float
    score_percent: float
    threshold: float
    positive_wells: list[str]
    negative_wells: list[str]
    missing_result_wells: list[str]


def _allele_key(locus: str, allele: str) -> str:
    """
    Join an HLA locus and allele with a colon.

    Parameters
    ----------
    locus : str
        HLA locus name.
    allele : str
        HLA allele identifier.

    Returns
    -------
    str
        Computed value of ``f'{locus}:{allele}'``.
    """
    return f"{locus}:{allele}"


def _normalize_well_set(well_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    """
    Uppercase well identifiers while preserving None as no restriction.

    Parameters
    ----------
    well_ids : Optional[Iterable[str]]
        Well identifiers to normalize; None represents no selection restriction.

    Returns
    -------
    Optional[set[str]]
        Depending on the input and branch, returns ``None``, ``{str(well_id).upper() for
        well_id in well_ids}``.
    """
    if well_ids is None:
        return None

    return {str(well_id).upper() for well_id in well_ids}


def _well_positive_value(wr: WellResult | Mapping[str, Any]) -> Optional[float]:
    """
    Read a corrected positive percentage, returning None for missing values or NaN.

    Parameters
    ----------
    wr : WellResult | Mapping[str, Any]
        Well result object or mapping containing corrected values and optional calls.

    Returns
    -------
    Optional[float]
        A corrected positive percentage, returning None for missing values or NaN.
    """
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
    """
    Resolve an explicit effective call before checking the well's own manual call.

    Parameters
    ----------
    well_id : str
        Plate well identifier, such as ``A1``.
    wr : WellResult | Mapping[str, Any]
        Well result object or mapping containing corrected values and optional calls.
    effective_calls : Optional[Mapping[str, str]]
        Optional mapping of well IDs to explicit positive or negative calls.

    Returns
    -------
    Optional[str]
        Depending on the input and branch, returns ``str(call)``, ``None``.
    """
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
    """
    Use an effective call or compare the corrected percentage against the threshold.

    Parameters
    ----------
    well_id : str
        Plate well identifier, such as ``A1``.
    wr : WellResult | Mapping[str, Any]
        Well result object or mapping containing corrected values and optional calls.
    positivity_threshold : float
        Positive-call cutoff in the units of the selected assay metric.
    effective_calls : Optional[Mapping[str, str]]
        Optional mapping of well IDs to explicit positive or negative calls.

    Returns
    -------
    Optional[bool]
        Depending on the input and branch, returns ``True``, ``False``, ``None``, ``value >=
        positivity_threshold``.
    """
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
    """
    Report per-allele PRA reactivity.

    Parameters
    ----------
    per_well : Mapping[str, WellResult]
        Mapping of well IDs to ROI measurements and well results.
    hla_layout : ParsedPlateLayout
        Parsed HLA plate layout mapping wells to donor alleles.
    positivity_threshold : float
        Positive-call cutoff in the units of the selected assay metric.
    include_well_ids : Optional[Iterable[str]]
        Well IDs to include; None considers all wells in the HLA layout. Default is
        ``None``.
    effective_calls : Optional[Mapping[str, str]]
        Optional mapping of well IDs to explicit positive or negative calls. Default is
        ``None``.

    Returns
    -------
    list[dict[str, Any]]
        Computed value of ``[asdict(e) for e in evidence]``.

    Notes
    -----
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
    """
    Calculate temporary overall PRA reactivity.

    Parameters
    ----------
    per_well : Mapping[str, WellResult]
        Mapping of well IDs to ROI measurements and well results.
    hla_layout : ParsedPlateLayout
        Parsed HLA plate layout mapping wells to donor alleles.
    positivity_threshold : float
        Positive-call cutoff in the units of the selected assay metric.
    include_well_ids : Optional[Iterable[str]]
        Well IDs to include; None considers all wells in the HLA layout. Default is
        ``None``.
    effective_calls : Optional[Mapping[str, str]]
        Optional mapping of well IDs to explicit positive or negative calls. Default is
        ``None``.

    Returns
    -------
    dict[str, Any]
        Temporary overall PRA reactivity.

    Notes
    -----
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
