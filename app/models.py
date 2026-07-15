from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


WellType = Literal["positive", "negative", "sample", "empty", "igm"]
WellID = str
CrossmatchCellMode = Literal["T", "B", "T/B", "empty"]

ChannelRole = Literal["Scatter", "Population Marker", "IgG Marker"]
SampleRole = Literal["NC", "PC", "SAMPLE"]
JobType = Literal["pra", "crossmatch", "fcxm"]


class CreateJobRequest(BaseModel):
    job_type: JobType


class JobResponse(BaseModel):
    job_id: str
    job_type: JobType
    status: str
    stage: str | None = None
    created_at: float
    updated_at: float


class PlateLayout(BaseModel):
    wells: Dict[WellID, WellType]


class ProcessRequest(BaseModel):
    layout: PlateLayout
    image_order: List[WellID]
    image_filenames: list[str]
    pra_positivity_threshold: float = Field(default=20.0, ge=0.0, le=100.0)
    column_modes: Dict[int, CrossmatchCellMode] = Field(default_factory=dict)
    flip_vertical: bool = False


class WellClassificationOverrideRequest(BaseModel):
    call: Literal["positive", "negative"] | None = None


class ProcessStartResponse(BaseModel):
    job_id: str


class WellResult(BaseModel):
    well: WellID
    role: WellType
    score: float
    status: str


class ProcessResponse(BaseModel):
    results: List[WellResult]
    summary: Dict[str, Any]


class ProgressResponse(BaseModel):
    status: str
    stage: str | None = None
    done: int = 0
    total: int = 0
    current_well: str | None = None
    done_wells: list[str] = Field(default_factory=list)
    done_filenames: list[str] = Field(default_factory=list)
    result: Any | None = None
    error: str | None = None
    error_type: str | None = None
    failed_stage: str | None = None
    failed_well: str | None = None
    support_id: str | None = None


class SimPoint(BaseModel):
    x: float
    y: float
    inGate: bool = False


class GatingPlot(BaseModel):
    title: str
    x_label: str
    y_label: str
    points: List[SimPoint]


class PlotSeries(BaseModel):
    label: str
    color: str
    points: List[SimPoint]
    n_total: int
    n_pos: int
    pos_pct: float


class LineSeries(BaseModel):
    label: str
    color: str
    values: list[float]
    values_raw: list[float] | None = None
    raw_median: float | None = None
    transformed_median: float | None = None
    value_scale: str | None = None
    x_label: str | None = None
    n_total: int
    n_pos: int
    pos_pct: float
    filename: str | None = None
    sample_name: str | None = None
    role: str | None = None


class FCXMResultsRequest(BaseModel):
    fcs_filename: str
    gate: str = ""


class FCXMGateMetrics(BaseModel):
    label: str
    n_events: int
    igg_pos_fraction: float
    igg_median_raw: float
    igg_median_t: float
    igg_median_shift: float
    igg_median_ratio: float
    igg_fluorescence_index: float
    igg_cutoff_t: float
    igg_nc_median_raw: float
    igg_pc_median_raw: Optional[float] = None


class FCXMResultsResponse(BaseModel):
    gate_options: List[str]
    selected_gate: Optional[str] = None
    gating_plots: List[GatingPlot]
    final_scatter_series: List[PlotSeries]
    line_series: List[LineSeries]
    cutoff: float
    selected_file_metrics: Optional[FCXMGateMetrics] = None
    selected_sample_metrics: Optional[FCXMGateMetrics] = None


class FCXMPanelRow(BaseModel):
    channel: str
    role: ChannelRole
    antibody: str
    population: str


class FCXMSample(BaseModel):
    id: str
    name: str
    role: SampleRole
    file_paths: List[str]


class FCXMRunRequest(BaseModel):
    panel_rows: List[FCXMPanelRow]
    samples: List[FCXMSample]


class FCXMRunStartResponse(BaseModel):
    job_id: str


class FCXMRunProgressResponse(BaseModel):
    status: Literal["queued", "running", "done", "error"]
    message: str | None = None
    stage: str | None = None
    result: Any | None = None
    total_files: int | None = None
    done_files: int | None = None
    current_file: str | None = None
    done_filenames: list[str] = Field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    failed_stage: str | None = None
    failed_file: str | None = None
    support_id: str | None = None


class FcsDisplayNamesRequest(BaseModel):
    filenames: list[str]
    mode: Literal["filename", "tube_name"] = "filename"


class FcsDisplayNamesResponse(BaseModel):
    names: dict[str, str]
