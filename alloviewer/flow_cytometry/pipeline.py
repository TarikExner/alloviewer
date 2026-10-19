from __future__ import annotations

import logging
from typing import Any, Callable

from alloviewer.flow_cytometry.sample import Dataset, Sample
from alloviewer.flow_cytometry.panel_utils import (
    build_panel_from_rows,
)
from alloviewer.flow_cytometry.gating import (
    Gater,
    GatingConfig,
)

from .plots import make_results_payload


logger = logging.getLogger(__name__)

ProgressEvent = dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    event: ProgressEvent,
) -> None:
    """
    Send one normalized progress event to the application layer.

    Parameters
    ----------
    progress_cb : ProgressCallback | None
        Callback receiving progress-event dictionaries; None disables progress events.
    event : ProgressEvent
        Progress event emitted by the analysis pipeline.
    """
    if progress_cb is not None:
        progress_cb(event)


def _count_files(req_dict: dict[str, Any]) -> int:
    """
    Count file paths across all submitted sample groups.

    Parameters
    ----------
    req_dict : dict[str, Any]
        FCXM request mapping with panel rows and samples with resolved file paths.

    Returns
    -------
    int
        File paths across all submitted sample groups.
    """
    return sum(
        len(sample.get("file_paths", []) or [])
        for sample in req_dict.get("samples", [])
    )


def _event_file_name(
    event: ProgressEvent,
) -> str | None:
    """
    Read a file name from the supported progress-event keys.

    Parameters
    ----------
    event : ProgressEvent
        Progress event emitted by the analysis pipeline.

    Returns
    -------
    str | None
        A file name from the supported progress-event keys.
    """
    value = (
        event.get("file_name")
        or event.get("file_path")
        or event.get("current_file")
    )

    if value is None:
        return None

    return str(value)


def _stage_message(stage: str) -> str:
    """
    Map a pipeline stage to its progress message, with a readable fallback.

    Parameters
    ----------
    stage : str
        Current pipeline stage, or None.

    Returns
    -------
    str
        Computed value of ``messages.get(stage, stage.replace('_', ' ').capitalize() +
        '.')``.
    """
    messages = {
        "starting": "Starting flow cytometry analysis.",
        "build_dataset": "Building dataset.",
        "build_panel": "Building panel.",
        "fit_qc": "Running file QC.",
        "fit_marker_calibration": "Calibrating markers.",
        "fit_lymphocytes": "Gating lymphocytes.",
        "fit_clustering": "Fitting clustering model.",
        "fit_cluster_labels": "Labeling clusters.",
        "fit_control_stats": "Building control statistics.",
        "apply_file": "Applying gates to files.",
        "payload": "Building result payload.",
        "plot_cache": "Building plot cache.",
        "done": "Analysis done.",
    }

    return messages.get(
        stage,
        stage.replace("_", " ").capitalize() + ".",
    )


def _make_progress_callback(
    *,
    total_files: int,
    progress_cb: ProgressCallback | None,
) -> ProgressCallback:
    """
    Build a callback used by the FCXM analysis components.

    Parameters
    ----------
    total_files : int
        Number of input files before expansion into progress work units.
    progress_cb : ProgressCallback | None
        Callback receiving progress-event dictionaries; None disables progress events.

    Returns
    -------
    ProgressCallback
        A callback used by the FCXM analysis components.

    Notes
    -----
    Progress is represented as file-level work units.

    Counted stages
    --------------
    - fit_qc
    - fit_lymphocytes
    - fit_control_stats
    - apply_file
    - plot_cache

    Global stages
    -------------
    These are displayed but do not increment the work-unit counter:

    - build_dataset
    - build_panel
    - fit_marker_calibration
    - fit_clustering
    - fit_cluster_labels
    - payload
    """
    counted_stages = {
        "fit_qc",
        "fit_lymphocytes",
        "fit_control_stats",
        "apply_file",
        "plot_cache",
    }

    total_work = max(
        1,
        total_files * len(counted_stages),
    )

    done_work = 0
    done_filenames: list[str] = []

    def progress(event: ProgressEvent) -> None:
        """
        Update file-stage work counts and emit normalized FCXM progress.

        Parameters
        ----------
        event : ProgressEvent
            Progress event emitted by the analysis pipeline.
        """
        nonlocal done_work
        nonlocal done_filenames

        stage = str(
            event.get("stage") or "running"
        )

        file_name = _event_file_name(event)

        if stage in counted_stages:
            done_work = min(
                done_work + 1,
                total_work,
            )

        # A file is considered fully processed only after its plot cache has
        # been built.
        if stage == "plot_cache" and file_name:
            if file_name not in done_filenames:
                done_filenames.append(file_name)

        _emit_progress(
            progress_cb,
            {
                "status": "running",
                "message": _stage_message(stage),
                "stage": stage,
                "total_files": total_work,
                "done_files": done_work,
                "current_file": file_name,
                "done_filenames": done_filenames.copy(),
            },
        )

    return progress


def _build_dataset(
    req_dict: dict[str, Any],
) -> Dataset:
    """
    Construct Sample objects and load their FCS files from the request.

    Parameters
    ----------
    req_dict : dict[str, Any]
        FCXM request mapping with panel rows and samples with resolved file paths.

    Returns
    -------
    Dataset
        Sample objects and load their FCS files from the request.
    """
    samples: list[Sample] = []

    for sample_dict in req_dict["samples"]:
        samples.append(
            Sample(
                name=sample_dict["name"],
                role=sample_dict["role"],
                file_paths=sample_dict["file_paths"],
            )
        )

    return Dataset(samples=samples)


def run_fcxm_pipeline(
    req_dict: dict[str, Any],
    *,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Run FCXM analysis.

    Parameters
    ----------
    req_dict : dict[str, Any]
        Dictionary created from ``FCXMRunRequest.model_dump()``. It contains
        panel rows and samples with resolved file paths.
    progress_cb : ProgressCallback | None
        Optional callback receiving normalized progress dictionaries. Default is ``None``.

    Returns
    -------
    dict
        Dictionary containing the frontend payload and plot cache.

    Notes
    -----
    This function does not write job state to Redis and does not mark a job as
    completed or failed. Those actions belong to the application service.
    """
    total_files = _count_files(req_dict)

    logger.info(
        "Starting FCXM analysis with %d files.",
        total_files,
    )

    counted_stage_count = 5
    total_work = max(
        1,
        total_files * counted_stage_count,
    )

    _emit_progress(
        progress_cb,
        {
            "status": "running",
            "message": _stage_message("starting"),
            "stage": "starting",
            "total_files": total_work,
            "done_files": 0,
            "current_file": None,
            "done_filenames": [],
        },
    )

    progress = _make_progress_callback(
        total_files=total_files,
        progress_cb=progress_cb,
    )

    progress({"stage": "build_dataset"})
    dataset = _build_dataset(req_dict)

    progress({"stage": "build_panel"})
    panel, marker_to_population = build_panel_from_rows(
        req_dict["panel_rows"]
    )

    gater = Gater(
        panel,
        GatingConfig(),
    )

    fitted = gater.fit(
        dataset,
        progress_cb=progress,
    )

    results = gater.apply(
        dataset,
        fitted,
        progress_cb=progress,
    )

    progress({"stage": "payload"})

    payload, plot_cache = make_results_payload(
        ds=dataset,
        gater=gater,
        fitted=fitted,
        results=results,
        marker_to_population=marker_to_population,
        max_points=5000,
        seed=0,
        progress_cb=progress,
    )

    logger.info(
        "Finished FCXM analysis with %d files.",
        total_files,
    )

    return {
        "payload": payload,
        "plot_cache": plot_cache,
    }


def run_fcxm_analysis(
    req_dict: dict[str, Any],
    *,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Public FCXM analysis entry point.

    Parameters
    ----------
    req_dict : dict[str, Any]
        FCXM request mapping with panel rows and samples with resolved file paths.
    progress_cb : ProgressCallback | None
        Callback receiving progress-event dictionaries; None disables progress events.
        Default is ``None``.

    Returns
    -------
    dict[str, Any]
        Computed value of ``run_fcxm_pipeline(req_dict=req_dict, progress_cb=progress_cb)``.
    """
    return run_fcxm_pipeline(
        req_dict=req_dict,
        progress_cb=progress_cb,
    )
