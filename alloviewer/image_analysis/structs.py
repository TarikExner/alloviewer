from dataclasses import dataclass, field, asdict
import numpy as np
from typing import Dict, List, Literal, Optional, Any, Union, Iterable

WellType = Literal["positive", "negative", "IgM", "sample", "empty"]
WellID = str

@dataclass(frozen=True)
class WellImage:
    well_id: str
    role: str = ""
    path: str = ""
    image: np.ndarray = np.array([])
    result_ids: List[str] = field(default_factory=list)

@dataclass
class Plate:
    plate_id: str
    wells: Dict[str, WellImage] = field(default_factory=dict)

    def add(self, w: WellImage): self.wells[w.well_id] = w

    def get(self, role: Optional[str]=None) -> List[WellImage]:
        if role is None:
            return list(self.wells.values())
        return [w for w in self.wells.values() if w.role == role]

    def subset(self, wells: Union[str, Iterable[str]]) -> "Plate":
        """
        Return a new Plate containing only the wells with the given IDs.

        Parameters
        ----------
        wells : str or iterable of str
            A single well ID (e.g. "A01") or a list/tuple/set of well IDs.
        """
        # Normalize to a set of IDs
        if isinstance(wells, str):
            ids = {wells}
        else:
            ids = set(wells)

        new_wells = {wid: w for wid, w in self.wells.items() if wid in ids}
        return Plate(plate_id=self.plate_id, wells=new_wells)
    

@dataclass
class PlateLayout:
  wells: Dict[WellID, WellType]


@dataclass
class ROIResult:
    roi_id: int
    mean_r: float
    mean_g: float
    mean_b: float
    area: int

    label: Optional[str] = None            # "pos" | "neg" | "uncertain"
    score: Optional[float] = None          # primary score (e.g., R/G threshold score)

    # optional diagnostics for Gaussian/median variants
    score_rg: Optional[float] = None       # R/G used by classifiers
    z_nc: Optional[float] = None           # z vs NC (upper-tail test)
    z_pc: Optional[float] = None           # z vs PC (lower-tail test)
    p_nc_upper: Optional[float] = None     # 1 - CDF(z_nc)
    p_pc_lower: Optional[float] = None     # CDF(z_pc)
    method: Optional[str] = None           # classifier/calibrator tag

    # future-proof bucket
    extras: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WellResult:
    well_id: str
    rois: List[ROIResult] = field(default_factory=list)
    qc: Dict[str, Any] = field(default_factory=dict)
    store_paths: Dict[str, str] = field(default_factory=dict)
    preview_path: Optional[str] = None

    corrected_frac_pos: Optional[float] = None

    def summary(self) -> Dict[str, Any]:
        n_pos = sum(1 for r in self.rois if r.label == "pos")
        n_rois = len(self.rois)

        if n_rois == 0:
            frac_pos = 0.0
        else:
            frac_pos = 100.0 * (n_pos / n_rois)

        return {
            "well_id": self.well_id,
            "n_rois": n_rois,
            "n_pos": n_pos,
            "frac_pos": frac_pos,
            "frac_pos_corrected": self.corrected_frac_pos,
            "qc": self.qc,
            "store_paths": self.store_paths,
            "preview_path": self.preview_path,
        }

@dataclass
class LociMap:
    # locus -> list of alleles/markers (flexible, future-proof)
    data: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class WellLayout:
    well_id: WellID
    combo_id: Optional[str] = None
    race: Optional[str] = None
    loci: LociMap = field(default_factory=LociMap)

@dataclass
class ParsedPlateLayout:
    upload_id: str
    schema_version: str
    sha256: str
    lot_no: Optional[str]
    compl_no: Optional[str]
    plate_format: Optional[str]
    wells: Dict[WellID, WellLayout]
    custom_loci: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    valid: bool = True

@dataclass
class PositivityEntry:
    well_id: WellID
    positive_fraction: float  # 0..1

@dataclass
class Stringency:
    min_positive_fraction: float = 2/3
    min_alt_threshold: Optional[float] = None
    min_positive_support: int = 2
    negative_penalty: float = 0.5
    allow_relaxed: bool = True

@dataclass
class AnalysisRequest:
    upload_id: str
    image_batch_id: Optional[str] = None
    positivity_threshold: float = 0.20
    stringency: Stringency = field(default_factory=Stringency)

@dataclass
class AlleleEvidence:
    allele_key: str
    supports: int
    supports_weighted: float
    total_with_allele: int
    positive_with_allele: int
    negative_with_allele: int
    positive_fraction: float
    score: float

@dataclass
class AnalysisResult:
    upload_id: str
    positivity_threshold: float
    inferred_positive_alleles: List[AlleleEvidence]
    per_well: Dict[WellID, Dict[str, Any]]
    notes: List[str] = field(default_factory=list)

# ---- helpers for (de)serialization ----

def dc_to_dict(obj: Any) -> Any:
    """Convert dataclass objects (nested) to plain dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dc_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: dc_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dc_to_dict(v) for v in obj]
    return obj

def parsed_plate_from_dict(d: Dict[str, Any]) -> ParsedPlateLayout:
    wells: Dict[str, WellLayout] = {}
    for wid, w in d["wells"].items():
        wells[wid] = WellLayout(
            well_id=w["well_id"],
            combo_id=w.get("combo_id"),
            race=w.get("race"),
            loci=LociMap(data=w.get("loci", {}).get("data", w.get("loci", {})))
        )
    return ParsedPlateLayout(
        upload_id=d["upload_id"],
        schema_version=d["schema_version"],
        sha256=d["sha256"],
        lot_no=d.get("lot_no"),
        compl_no=d.get("compl_no"),
        plate_format=d.get("plate_format"),
        wells=wells,
        custom_loci=d.get("custom_loci", []),
        warnings=d.get("warnings", []),
        valid=d.get("valid", True),
    )
