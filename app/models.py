import time
import threading

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field

WellType = Literal["positive", "negative", "sample", "empty", "igm"]
WellID = str

ChannelRole = Literal["Scatter", "Population Marker", "IgG Marker"]
SampleRole = Literal["NC", "PC", "SAMPLE"]

IMAGE_JOB_PROGRESS: Dict[str, Dict[str, Any]] = {}
IMAGE_JOB_RESULTS: Dict[str, Any] = {}

FCXM_JOB_PROGRESS: Dict[str, Dict[str, Any]] = {}
FCXM_JOB_RESULTS: Dict[str, Any] = {}
FCXM_JOB_PLOTS: Dict[str, Any] = {}
FCXM_JOB_RUN_REQUESTS: Dict[str, Dict[str, Any]] = {}

# One lock for all FCXM job dicts (requests + background tasks share these)
FCXM_JOB_LOCK = threading.RLock()

# Time-to-live since last access (seconds). Adjust as needed.
FCXM_JOB_TTL_SECONDS = 6 * 60 * 10  # 1 hour

def fcxm_job_touch(job_id: str) -> None:
    """
    Update last_access (and set created_at if missing).
    Safe to call even if job_id is unknown.
    """
    now = time.time()
    prog = FCXM_JOB_PROGRESS.get(job_id)
    if prog is None:
        return
    prog.setdefault("created_at", now)
    prog["last_access"] = now


def fcxm_job_cleanup() -> int:
    """
    Remove FCXM jobs that have not been accessed for FCXM_JOB_TTL_SECONDS.
    Returns number of deleted jobs.
    """
    now = time.time()
    deleted = 0

    # Decide staleness based on progress timestamps (single source of truth)
    stale_ids: List[str] = []
    for job_id, prog in list(FCXM_JOB_PROGRESS.items()):
        last_access = prog.get("last_access") or prog.get("created_at")
        if last_access is None:
            # If no timestamps, treat as stale
            stale_ids.append(job_id)
            continue
        if (now - float(last_access)) > FCXM_JOB_TTL_SECONDS:
            stale_ids.append(job_id)

    for job_id in stale_ids:
        FCXM_JOB_PROGRESS.pop(job_id, None)
        FCXM_JOB_RESULTS.pop(job_id, None)
        FCXM_JOB_PLOTS.pop(job_id, None)
        FCXM_JOB_RUN_REQUESTS.pop(job_id, None)
        deleted += 1

    return deleted

class PlateLayout(BaseModel):
    wells: Dict[WellID, WellType]

class ProcessRequest(BaseModel):
    layout: PlateLayout
    image_order: List[WellID]
    image_filenames: list[str]
    template_filename: Optional[str] = None
    assay_type: Literal["pra", "crossmatch"] = "pra"

    hla_layout_upload_id: Optional[str] = None

    pra_positivity_threshold: float = Field(default=20.0, ge=0.0, le=100.0)

class ProcessStartResponse(BaseModel):
    job_id: str

class WellResult(BaseModel):
    well: WellID
    role: WellType
    score: float
    status: str

class ProcessResponse(BaseModel):
    results: List[WellResult]
    summary: Dict[str, float]

class ProgressResponse(BaseModel):
    status: str  # "queued" | "running" | "done"
    done: int
    total: int
    current_well: Optional[str]
    done_wells: List[str]
    result: Optional[dict] = None

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
    pos_pct: float  # 0..100

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
    job_id: str
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
    message: Optional[str] = None
    stage: Optional[str] = None
    result: Optional[Any] = None
    total_files: int | None = None
    done_files: int | None = None
    current_file: str | None = None
    done_filenames: list[str] = []

class FcsDisplayNamesRequest(BaseModel):
    filenames: list[str]
    mode: Literal["filename", "tube_name"] = "filename"


class FcsDisplayNamesResponse(BaseModel):
    names: dict[str, str]
