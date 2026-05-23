from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import numpy as np


WellType = Literal["positive", "negative", "IgM", "sample", "empty"]
WellID = str


@dataclass(frozen=True)
class WellImage:
    """Image and metadata for a single plate well.

    Parameters
    ----------
    well_id : str
        Well identifier, for example ``"A1"``.
    role : str, optional
        Well role, such as ``"positive"``, ``"negative"``, ``"sample"``, or
        ``"empty"``.
    path : str, optional
        Source image path.
    image : numpy.ndarray, optional
        Image array for the well.
    result_ids : list of str, optional
        IDs of results linked to this image.
    """

    well_id: str
    role: str = ""
    path: str = ""
    image: np.ndarray = field(default_factory=lambda: np.array([]))
    result_ids: List[str] = field(default_factory=list)


@dataclass
class Plate:
    """Collection of well images belonging to one plate.

    Parameters
    ----------
    plate_id : str
        Plate identifier.
    wells : dict, optional
        Mapping from well ID to :class:`WellImage`.
    """

    plate_id: str
    wells: Dict[str, WellImage] = field(default_factory=dict)

    def add(self, w: WellImage) -> None:
        """Add or replace a well image.

        Parameters
        ----------
        w : WellImage
            Well image to store. Existing entries with the same well ID are
            replaced.
        """
        self.wells[w.well_id] = w

    def get(self, role: Optional[str] = None) -> List[WellImage]:
        """Return wells, optionally filtered by role.

        Parameters
        ----------
        role : str or None, optional
            Role to filter by. If ``None``, all wells are returned.

        Returns
        -------
        list of WellImage
            Matching well images.
        """
        if role is None:
            return list(self.wells.values())

        return [w for w in self.wells.values() if w.role == role]

    def subset(self, wells: Union[str, Iterable[str]]) -> "Plate":
        """Return a plate containing only selected wells.

        Parameters
        ----------
        wells : str or iterable of str
            Single well ID or iterable of well IDs.

        Returns
        -------
        Plate
            New plate with the same ``plate_id`` and selected well images.

        Notes
        -----
        Well IDs not present in the plate are silently ignored.
        """
        if isinstance(wells, str):
            ids = {wells}
        else:
            ids = set(wells)

        new_wells = {wid: w for wid, w in self.wells.items() if wid in ids}
        return Plate(plate_id=self.plate_id, wells=new_wells)


@dataclass
class PlateLayout:
    """Role layout for a plate.

    Parameters
    ----------
    wells : dict
        Mapping from well ID to well type.
    """

    wells: Dict[WellID, WellType]


@dataclass
class ROIResult:
    """Measurement and classification result for one ROI.

    Parameters
    ----------
    roi_id : int
        ROI or instance identifier.
    mean_r : float
        Mean red-channel intensity.
    mean_g : float
        Mean green-channel intensity.
    mean_b : float
        Mean blue-channel intensity.
    area : int
        ROI area in pixels.
    label : str or None, optional
        Classification label, commonly ``"pos"``, ``"neg"``, or
        ``"uncertain"``.
    score : float or None, optional
        Main classifier score.
    score_rg : float or None, optional
        Red/green ratio score.
    z_nc : float or None, optional
        Z-score relative to the negative control distribution.
    z_pc : float or None, optional
        Z-score relative to the positive control distribution.
    p_nc_upper : float or None, optional
        Upper-tail probability under the negative control model.
    p_pc_lower : float or None, optional
        Lower-tail probability under the positive control model.
    logp_pc : float or None, optional
        Log probability under the positive control model.
    logp_nc : float or None, optional
        Log probability under the negative control model.
    mahal_pc : float or None, optional
        Mahalanobis distance to the positive control model.
    mahal_nc : float or None, optional
        Mahalanobis distance to the negative control model.
    delta_logp : float or None, optional
        Difference ``logp_pc - logp_nc``.
    method : str or None, optional
        Name of the classification method.
    extras : dict, optional
        Extra method-specific values.
    """

    roi_id: int
    mean_r: float
    mean_g: float
    mean_b: float
    area: int

    label: Optional[str] = None
    score: Optional[float] = None

    score_rg: Optional[float] = None
    z_nc: Optional[float] = None
    z_pc: Optional[float] = None
    p_nc_upper: Optional[float] = None
    p_pc_lower: Optional[float] = None

    logp_pc: Optional[float] = None
    logp_nc: Optional[float] = None
    mahal_pc: Optional[float] = None
    mahal_nc: Optional[float] = None
    delta_logp: Optional[float] = None

    method: Optional[str] = None

    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WellResult:
    """Analysis result for a single well.

    Parameters
    ----------
    well_id : str
        Well identifier.
    rois : list of ROIResult, optional
        ROI-level results for the well.
    qc : dict, optional
        Quality-control values.
    store_paths : dict, optional
        Paths to stored intermediate or output files.
    preview_path : str or None, optional
        Path to a preview image.
    corrected_frac_pos : float or None, optional
        Control-corrected positive ROI fraction.
    """

    well_id: str
    rois: List[ROIResult] = field(default_factory=list)
    qc: Dict[str, Any] = field(default_factory=dict)
    store_paths: Dict[str, str] = field(default_factory=dict)
    preview_path: Optional[str] = None

    corrected_frac_pos: Optional[float] = None

    def summary(self) -> Dict[str, Any]:
        """Return a compact well-level summary.

        Returns
        -------
        dict
            Summary containing well ID, ROI count, positive ROI count, raw
            positive fraction, corrected positive fraction, QC data, stored
            paths, and preview path.
        """
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
    """Mapping from locus names to alleles or markers.

    Parameters
    ----------
    data : dict, optional
        Mapping from locus name to a list of allele or marker strings.
    """

    data: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class WellLayout:
    """Parsed layout information for one well.

    Parameters
    ----------
    well_id : str
        Well identifier.
    combo_id : str or None, optional
        Combination ID assigned to the well.
    race : str or None, optional
        Race or source group value from the layout file.
    loci : LociMap, optional
        Locus-to-marker mapping for the well.
    """

    well_id: WellID
    combo_id: Optional[str] = None
    race: Optional[str] = None
    loci: LociMap = field(default_factory=LociMap)


@dataclass
class ParsedPlateLayout:
    """Parsed plate layout metadata and well definitions.

    Parameters
    ----------
    upload_id : str
        Upload identifier.
    schema_version : str
        Layout schema version.
    sha256 : str
        SHA-256 checksum of the uploaded layout source.
    lot_no : str or None
        Lot number.
    compl_no : str or None
        Complement number.
    plate_format : str or None
        Plate format descriptor.
    wells : dict
        Mapping from well ID to :class:`WellLayout`.
    custom_loci : list of str, optional
        Custom locus names found in the layout.
    warnings : list of str, optional
        Non-fatal parsing warnings.
    valid : bool, optional
        Whether the parsed layout is valid.
    """

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
    """Positive fraction assigned to one well.

    Parameters
    ----------
    well_id : str
        Well identifier.
    positive_fraction : float
        Positive fraction in the range ``0..1``.
    """

    well_id: WellID
    positive_fraction: float


@dataclass
class Stringency:
    """Thresholds and penalties for allele inference.

    Parameters
    ----------
    min_positive_fraction : float, optional
        Minimum fraction of wells with an allele that must be positive.
    min_alt_threshold : float or None, optional
        Optional alternative positive-fraction threshold.
    min_positive_support : int, optional
        Minimum number of positive wells supporting an allele.
    negative_penalty : float, optional
        Penalty applied for negative wells carrying an allele.
    allow_relaxed : bool, optional
        Whether relaxed fallback rules may be used.
    """

    min_positive_fraction: float = 2 / 3
    min_alt_threshold: Optional[float] = None
    min_positive_support: int = 2
    negative_penalty: float = 0.5
    allow_relaxed: bool = True


@dataclass
class AnalysisRequest:
    """Input request for allele-level analysis.

    Parameters
    ----------
    upload_id : str
        Upload identifier of the parsed plate layout.
    image_batch_id : str or None, optional
        Optional image batch identifier.
    positivity_threshold : float, optional
        Threshold used to call a well positive.
    stringency : Stringency, optional
        Allele inference thresholds.
    """

    upload_id: str
    image_batch_id: Optional[str] = None
    positivity_threshold: float = 0.20
    stringency: Stringency = field(default_factory=Stringency)


@dataclass
class AlleleEvidence:
    """Evidence summary for one inferred positive allele.

    Parameters
    ----------
    allele_key : str
        Allele identifier.
    supports : int
        Number of positive supporting wells.
    supports_weighted : float
        Weighted support score.
    total_with_allele : int
        Number of wells containing the allele.
    positive_with_allele : int
        Number of allele-containing wells called positive.
    negative_with_allele : int
        Number of allele-containing wells called negative.
    positive_fraction : float
        Fraction of allele-containing wells called positive.
    score : float
        Final evidence score.
    """

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
    """Result of allele-level analysis.

    Parameters
    ----------
    upload_id : str
        Upload identifier of the parsed plate layout.
    positivity_threshold : float
        Threshold used to call wells positive.
    inferred_positive_alleles : list of AlleleEvidence
        Inferred positive alleles with evidence fields.
    per_well : dict
        Per-well analysis data.
    notes : list of str, optional
        Analysis notes or warnings.
    """

    upload_id: str
    positivity_threshold: float
    inferred_positive_alleles: List[AlleleEvidence]
    per_well: Dict[WellID, Dict[str, Any]]
    notes: List[str] = field(default_factory=list)


def dc_to_dict(obj: Any) -> Any:
    """Convert nested dataclass objects to plain Python containers.

    Parameters
    ----------
    obj : Any
        Dataclass object, dictionary, list, or scalar value.

    Returns
    -------
    Any
        Plain Python representation. Dataclasses become dictionaries, lists
        remain lists, dictionaries remain dictionaries, and scalar values are
        returned unchanged.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dc_to_dict(v) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {k: dc_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [dc_to_dict(v) for v in obj]

    return obj


def parsed_plate_from_dict(d: Dict[str, Any]) -> ParsedPlateLayout:
    """Create a parsed plate layout from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary representation of a parsed plate layout.

    Returns
    -------
    ParsedPlateLayout
        Parsed plate layout object.

    Raises
    ------
    KeyError
        If required top-level fields or required well fields are missing.

    Notes
    -----
    The ``loci`` field accepts either ``{"data": ...}`` or a direct
    locus-to-marker mapping.
    """
    wells: Dict[str, WellLayout] = {}

    for wid, w in d["wells"].items():
        wells[wid] = WellLayout(
            well_id=w["well_id"],
            combo_id=w.get("combo_id"),
            race=w.get("race"),
            loci=LociMap(data=w.get("loci", {}).get("data", w.get("loci", {}))),
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
