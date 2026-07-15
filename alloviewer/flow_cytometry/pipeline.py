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
    """
    if progress_cb is not None:
        progress_cb(event)


def _count_files(req_dict: dict[str, Any]) -> int:
    return sum(
        len(sample.get("file_paths", []) or [])
        for sample in req_dict.get("samples", [])
    )


def _event_file_name(
    event: ProgressEvent,
) -> str | None:
    value = (
        event.get("file_name")
        or event.get("file_path")
        or event.get("current_file")
    )

    if value is None:
        return None

    return str(value)


def _stage_message(stage: str) -> str:
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
    req_dict
        Dictionary created from ``FCXMRunRequest.model_dump()``. It contains
        panel rows and samples with resolved file paths.
    progress_cb
        Optional callback receiving normalized progress dictionaries.

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
    """
    return run_fcxm_pipeline(
        req_dict=req_dict,
        progress_cb=progress_cb,
    )
