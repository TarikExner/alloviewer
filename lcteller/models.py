from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel

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
class SegmentationResults:
    """Big arrays from segmentation. Often saved to disk; keep paths in WellResult."""
    instances: np.ndarray                # int32 [H,W]
    cell_mask: np.ndarray                # bool/uint8 [H,W]
    bound_mask: np.ndarray               # bool/uint8 [H,W]
    probs: Optional[np.ndarray] = None   # float32 [2,H,W] (optional / large)

    def to_shapes(self) -> Dict[str, Any]:
        return {
            "instances": tuple(self.instances.shape),
            "cell_mask": tuple(self.cell_mask.shape),
            "bound_mask": tuple(self.bound_mask.shape),
            "probs": None if self.probs is None else tuple(self.probs["cell"].shape),
        }

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
    cfg_hash: str
    rois: List[ROIResult] = field(default_factory=list)
    results: Optional[SegmentationResults] = None
    qc: Dict[str, Any] = field(default_factory=dict)
    store_paths: Dict[str, str] = field(default_factory=dict)
    preview_path: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        n_pos = sum(1 for r in self.rois if r.label == "pos")
        n_rois = len(self.rois)
        return {
            "well_id": self.well_id,
            "cfg_hash": self.cfg_hash,
            "n_rois": n_rois,
            "n_pos": n_pos,
            "frac_pos": int((n_pos/n_rois)*100),
            "qc": self.qc,
            "results_shapes": None if self.results is None else self.results.to_shapes(),
            "store_paths": self.store_paths,
            "preview_path": self.preview_path,
        }

@dataclass
class Plate:
    plate_id: str
    wells: Dict[str, WellImage] = field(default_factory=dict)

    def add(self, w: WellImage): self.wells[w.well_id] = w

    def get(self, role: Optional[str]=None) -> List[WellImage]:
        if role is None:
            return list(self.wells.values())
        return [w for w in self.wells.values() if w.role == role]


class PlateLayout(BaseModel):
  wells: Dict[WellID, WellType]


