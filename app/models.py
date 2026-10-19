from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


WellType = Literal["positive", "negative", "sample", "empty", "igm"]
WellID = str
CrossmatchCellMode = Literal["T", "B", "T/B", "empty"]

ChannelRole = Literal["Scatter", "Population Marker", "IgG Marker"]
SampleRole = Literal["NC", "PC", "SAMPLE"]
JobType = Literal["pra", "crossmatch", "fcxm"]


class CreateJobRequest(BaseModel):
    """
    Specify the assay type for a new analysis job.

    Attributes
    ----------
    job_type : JobType
        Assay type: ``pra``, ``crossmatch``, or ``fcxm``.
    """
    job_type: JobType


class JobResponse(BaseModel):
    """
    Describe a job's identifier, type, state, and timestamps.

    Attributes
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    job_type : JobType
        Assay type: ``pra``, ``crossmatch``, or ``fcxm``.
    status : str
        Current processing state.
    stage : str | None
        Current pipeline stage, or None. Default is ``None``.
    created_at : float
        Job creation time as seconds since the Unix epoch.
    updated_at : float
        Last metadata update time as seconds since the Unix epoch.
    """
    job_id: str
    job_type: JobType
    status: str
    stage: str | None = None
    created_at: float
    updated_at: float


class PlateLayout(BaseModel):
    """
    Map plate well identifiers to their assay roles.

    Attributes
    ----------
    wells : Dict[WellID, WellType]
        Mapping from well IDs to per-well data or assay roles, as required by the model.
    """
    wells: Dict[WellID, WellType]


class ProcessRequest(BaseModel):
    """
    Specify image order, plate roles, and assay settings for an image run.

    Attributes
    ----------
    layout : PlateLayout
        Plate layout mapping well positions to their assay roles.
    image_order : List[WellID]
        Well IDs in the same order as the uploaded image filenames.
    image_filenames : list[str]
        Image paths relative to the job root, paired with image_order.
    pra_positivity_threshold : float
        Positive-call cutoff for corrected PRA well values, from 0 to 100 percent. Default
        is ``Field(default=20.0, ge=0.0, le=100.0)``.
    column_modes : Dict[int, CrossmatchCellMode]
        Mapping from one-based column numbers to ``T``, ``B``, ``T/B``, or ``empty``.
        Default is ``Field(default_factory=dict)``.
    flip_vertical : bool
        If True, reverse the displayed plate row order. Default is ``False``.
    """
    layout: PlateLayout
    image_order: List[WellID]
    image_filenames: list[str]
    pra_positivity_threshold: float = Field(default=20.0, ge=0.0, le=100.0)
    column_modes: Dict[int, CrossmatchCellMode] = Field(default_factory=dict)
    flip_vertical: bool = False


class WellClassificationOverrideRequest(BaseModel):
    """
    Set a manual well call or clear it with None.

    Attributes
    ----------
    call : Literal['positive', 'negative'] | None
        Manual call, ``positive`` or ``negative``; None clears an existing override. Default
        is ``None``.
    """
    call: Literal["positive", "negative"] | None = None


class ProcessStartResponse(BaseModel):
    """
    Return the identifier of a submitted image job.

    Attributes
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    """
    job_id: str


class WellResult(BaseModel):
    """
    Describe one well's role, score, and analysis status.

    Attributes
    ----------
    well : WellID
        Mapping of measurements and call information for one well.
    role : WellType
        Well role, such as sample, positive control, negative control, empty, or IgM
        control.
    score : float
        Computed score assigned to the well.
    status : str
        Current processing state.
    """
    well: WellID
    role: WellType
    score: float
    status: str


class ProcessResponse(BaseModel):
    """
    Return per-well results and their assay summary.

    Attributes
    ----------
    results : List[WellResult]
        Per-well image-analysis results included in the response.
    summary : Dict[str, Any]
        Aggregate assay metrics included in the process response.
    """
    results: List[WellResult]
    summary: Dict[str, Any]


class ProgressResponse(BaseModel):
    """
    Describe image-job progress, results, and any failure details.

    Attributes
    ----------
    status : str
        Current processing state.
    stage : str | None
        Current pipeline stage, or None. Default is ``None``.
    done : int
        Updated completed-well count; None leaves the count unchanged. Default is ``0``.
    total : int
        Total number of items expected by the progress tracker. Default is ``0``.
    current_well : str | None
        Well currently being processed, or None. Default is ``None``.
    done_wells : list[str]
        Well IDs whose processing has completed. Default is ``Field(default_factory=list)``.
    done_filenames : list[str]
        File names whose processing has completed. Default is
        ``Field(default_factory=list)``.
    result : Any | None
        Analysis result containing per-well values and assay summaries. Default is ``None``.
    error : str | None
        Optional public error message describing a failed job. Default is ``None``.
    error_type : str | None
        Exception type recorded for the failed job. Default is ``None``.
    failed_stage : str | None
        Pipeline stage at which the job failed, if known. Default is ``None``.
    failed_well : str | None
        Well at which processing failed, if known. Default is ``None``.
    support_id : str | None
        Job identifier supplied with a failure for support lookup. Default is ``None``.
    """
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
    """
    Store one scatter point and whether it belongs to the gate.

    Attributes
    ----------
    x : float
        Horizontal coordinate in the plot axis scale.
    y : float
        Vertical coordinate in the plot axis scale.
    inGate : bool
        Whether the displayed event point belongs to the indicated gate. Default is
        ``False``.
    """
    x: float
    y: float
    inGate: bool = False


class GatingPlot(BaseModel):
    """
    Describe a gating scatter plot and its axis labels.

    Attributes
    ----------
    title : str
        Text displayed as the plot or panel title.
    x_label : str
        Horizontal-axis label.
    y_label : str
        Vertical-axis label.
    points : List[SimPoint]
        Scatter-point mappings containing x, y, and optional inGate values.
    """
    title: str
    x_label: str
    y_label: str
    points: List[SimPoint]


class PlotSeries(BaseModel):
    """
    Store a labeled scatter series and its positive-event counts.

    Attributes
    ----------
    label : str
        Display label or identifier for the corresponding item.
    color : str
        Color used to render the corresponding graphical element.
    points : List[SimPoint]
        Scatter-point mappings containing x, y, and optional inGate values.
    n_total : int
        Total events represented by the plotted series.
    n_pos : int
        Events in the plotted series classified as IgG positive.
    pos_pct : float
        Percentage of plotted events classified as IgG positive.
    """
    label: str
    color: str
    points: List[SimPoint]
    n_total: int
    n_pos: int
    pos_pct: float


class LineSeries(BaseModel):
    """
    Store IgG values, plotting metadata, medians, and positive-event counts.

    Attributes
    ----------
    label : str
        Display label or identifier for the corresponding item.
    color : str
        Color used to render the corresponding graphical element.
    values : list[float]
        Numeric values used to compute the requested statistic or transformation.
    values_raw : list[float] | None
        Optional raw intensity values corresponding to the transformed series. Default is
        ``None``.
    raw_median : float | None
        Median intensity before transformation. Default is ``None``.
    transformed_median : float | None
        Median intensity in the scale used for plotting. Default is ``None``.
    value_scale : str | None
        Identifier of the intensity scale represented by the series. Default is ``None``.
    x_label : str | None
        Horizontal-axis label. Default is ``None``.
    n_total : int
        Total events represented by the plotted series.
    n_pos : int
        Events in the plotted series classified as IgG positive.
    pos_pct : float
        Percentage of plotted events classified as IgG positive.
    filename : str | None
        Name or path of the file selected by this operation. Default is ``None``.
    sample_name : str | None
        Name identifying the sample whose events or measurements are shown. Default is
        ``None``.
    role : str | None
        Sample role: NC for negative control, PC for positive control, or SAMPLE. Default is
        ``None``.
    """
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
    """
    Select an uploaded FCS file and a population gate for plotting.

    Attributes
    ----------
    fcs_filename : str
        FCS path relative to the job root.
    gate : str
        Population gate label; an empty string delegates selection to the result-view
        builder. Default is ``''``.
    """
    fcs_filename: str
    gate: str = ""


class FCXMGateMetrics(BaseModel):
    """
    Store event counts and IgG metrics for one population gate.

    Attributes
    ----------
    label : str
        Display label or identifier for the corresponding item.
    n_events : int
        Number of events in the selected population.
    igg_pos_fraction : float
        Fraction of population events strictly above the transformed IgG cutoff.
    igg_median_raw : float
        Median IgG intensity before the asinh transform.
    igg_median_t : float
        Median IgG intensity after the asinh transform.
    igg_median_shift : float
        Raw IgG median difference relative to the negative-control median.
    igg_median_ratio : float
        Raw IgG median divided by the negative-control median, when defined.
    igg_fluorescence_index : float
        Control-normalized IgG fluorescence index recorded for the population.
    igg_cutoff_t : float
        IgG positivity threshold in asinh-transformed intensity units.
    igg_nc_median_raw : float
        Raw median IgG intensity of the negative-control reference.
    igg_pc_median_raw : Optional[float]
        Raw median IgG intensity of the optional positive-control reference. Default is
        ``None``.
    """
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
    """
    Return available gates, plots, and selected file and sample metrics.

    Attributes
    ----------
    gate_options : List[str]
        Available gate labels in display order.
    selected_gate : Optional[str]
        Name of the selected end gate; the function supplies its fallback when empty.
        Default is ``None``.
    gating_plots : List[GatingPlot]
        Plots showing the successive event-gating stages.
    final_scatter_series : List[PlotSeries]
        Scatter layers for the selected final population gate.
    line_series : List[LineSeries]
        Per-file IgG values and plotting metadata for controls and the selected file.
    cutoff : float
        IgG positivity cutoff in the units of the supplied values.
    selected_file_metrics : Optional[FCXMGateMetrics]
        Population metrics for the selected individual FCS file. Default is ``None``.
    selected_sample_metrics : Optional[FCXMGateMetrics]
        Population metrics pooled across the selected sample's files. Default is ``None``.
    """
    gate_options: List[str]
    selected_gate: Optional[str] = None
    gating_plots: List[GatingPlot]
    final_scatter_series: List[PlotSeries]
    line_series: List[LineSeries]
    cutoff: float
    selected_file_metrics: Optional[FCXMGateMetrics] = None
    selected_sample_metrics: Optional[FCXMGateMetrics] = None


class FCXMPanelRow(BaseModel):
    """
    Map a detector to its assay role, antibody, and population.

    Attributes
    ----------
    channel : str
        FCS detector or channel name.
    role : ChannelRole
        Channel role: Scatter, Population Marker, or IgG Marker.
    antibody : str
        Antibody name assigned to the measurement channel.
    population : str
        Population label associated with the channel's marker.
    """
    channel: str
    role: ChannelRole
    antibody: str
    population: str


class FCXMSample(BaseModel):
    """
    Group FCS files under a named sample and control role.

    Attributes
    ----------
    id : str
        Client-provided identifier for the sample.
    name : str
        Name identifying the item being processed.
    role : SampleRole
        Sample role: NC for negative control, PC for positive control, or SAMPLE.
    file_paths : List[str]
        Paths to the FCS files belonging to this sample.
    """
    id: str
    name: str
    role: SampleRole
    file_paths: List[str]


class FCXMRunRequest(BaseModel):
    """
    Supply panel assignments and sample groups for an FCXM run.

    Attributes
    ----------
    panel_rows : List[FCXMPanelRow]
        Channel roles, antibodies, and populations defining the FCS panel.
    samples : List[FCXMSample]
        Sample objects included in the dataset.
    """
    panel_rows: List[FCXMPanelRow]
    samples: List[FCXMSample]


class FCXMRunStartResponse(BaseModel):
    """
    Return the identifier of a submitted FCXM job.

    Attributes
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    """
    job_id: str


class FCXMRunProgressResponse(BaseModel):
    """
    Describe FCXM progress, completed files, results, and failure details.

    Attributes
    ----------
    status : Literal['queued', 'running', 'done', 'error']
        Current processing state.
    message : str | None
        Human-readable status or error message. Default is ``None``.
    stage : str | None
        Current pipeline stage, or None. Default is ``None``.
    result : Any | None
        Analysis result containing per-well values and assay summaries. Default is ``None``.
    total_files : int | None
        Number of input files before expansion into progress work units. Default is
        ``None``.
    done_files : int | None
        Updated completed-file count; None leaves the count unchanged. Default is ``None``.
    current_file : str | None
        File currently being processed, or None. Default is ``None``.
    done_filenames : list[str]
        File names whose processing has completed. Default is
        ``Field(default_factory=list)``.
    error : str | None
        Optional public error message describing a failed job. Default is ``None``.
    error_type : str | None
        Exception type recorded for the failed job. Default is ``None``.
    failed_stage : str | None
        Pipeline stage at which the job failed, if known. Default is ``None``.
    failed_file : str | None
        File at which processing failed, if known. Default is ``None``.
    support_id : str | None
        Job identifier supplied with a failure for support lookup. Default is ``None``.
    """
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
    """
    Select FCS files and the source of their display names.

    Attributes
    ----------
    filenames : list[str]
        Uploaded FCS filenames whose display names are requested.
    mode : Literal['filename', 'tube_name']
        Display-name source: filename uses uploaded names and tube_name uses FCS metadata.
        Default is ``'filename'``.
    """
    filenames: list[str]
    mode: Literal["filename", "tube_name"] = "filename"


class FcsDisplayNamesResponse(BaseModel):
    """
    Map submitted FCS paths to their display names.

    Attributes
    ----------
    names : dict[str, str]
        Mapping from submitted FCS filenames to their display names.
    """
    names: dict[str, str]
