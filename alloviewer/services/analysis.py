from __future__ import annotations
from typing import Dict, List, Any
from ..structs import (
    ParsedPlateLayout, AnalysisRequest, AnalysisResult,
    AlleleEvidence, Stringency, PositivityEntry
)

def _allele_key(locus: str, allele: str) -> str:
    return f"{locus}:{allele}"

def analyze_culprit_alleles(
    layout: ParsedPlateLayout,
    positives: List[PositivityEntry],
    req: AnalysisRequest
) -> AnalysisResult:
    pos_map: Dict[str, float] = {p.well_id.upper(): p.positive_fraction for p in positives}

    carrier_wells: Dict[str, List[str]] = {}
    for wid, w in layout.wells.items():
        for locus, alleles in w.loci.data.items():
            for a in alleles:
                carrier_wells.setdefault(_allele_key(locus, a), []).append(wid)

    ev_list: List[AlleleEvidence] = []
    s: Stringency = req.stringency
    th = req.positivity_threshold
    alt_th = s.min_alt_threshold if s.allow_relaxed else None

    per_well_echo: Dict[str, Dict[str, Any]] = {
        wid: {"positive_fraction": pos_map.get(wid, 0.0), "loci": w.loci.data}
        for wid, w in layout.wells.items()
    }

    for allele_key, c_wells in carrier_wells.items():
        total_with = len(c_wells)
        pos_count = 0
        neg_count = 0
        weighted_support = 0.0

        for wid in c_wells:
            frac = pos_map.get(wid, 0.0)
            if frac >= th:
                pos_count += 1
                weighted_support += 1.0
            elif alt_th is not None and frac >= alt_th:
                weighted_support += 0.5
            else:
                neg_count += 1

        positive_fraction = (pos_count / total_with) if total_with else 0.0
        score = weighted_support - (s.negative_penalty * neg_count)

        ev_list.append(AlleleEvidence(
            allele_key=allele_key,
            supports=pos_count,
            supports_weighted=weighted_support,
            total_with_allele=total_with,
            positive_with_allele=pos_count,
            negative_with_allele=neg_count,
            positive_fraction=positive_fraction,
            score=score
        ))

    kept = [
        e for e in ev_list
        if e.supports >= s.min_positive_support
        and e.positive_fraction >= s.min_positive_fraction
        and e.score > 0
    ]
    kept.sort(key=lambda e: (e.score, e.positive_fraction, e.supports), reverse=True)

    notes = [
        f"Threshold: {th:.2f}",
        *( [f"Relaxed threshold: {alt_th:.2f} (half weight)"] if alt_th is not None else [] ),
        f"Min positive fraction: {s.min_positive_fraction:.2f}",
        f"Min positive support: {s.min_positive_support}",
        f"Negative penalty: {s.negative_penalty:.2f}",
    ]

    return AnalysisResult(
        upload_id=layout.upload_id,
        positivity_threshold=th,
        inferred_positive_alleles=kept,
        per_well=per_well_echo,
        notes=notes
    )

