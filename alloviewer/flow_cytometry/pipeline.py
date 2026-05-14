# alloviewer/flow_cytometry/pipeline.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from alloviewer.flow_cytometry.sample import Dataset, Sample
from alloviewer.flow_cytometry.panel_utils import build_panel_from_rows
from alloviewer.flow_cytometry.gating import Gater, GatingConfig

from .plots import make_results_payload

from app.models import FCXM_JOB_PROGRESS


ProgressEvent = Dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]


def _count_files(req_dict: Dict[str, Any]) -> int:
    return sum(
        len(sample.get("file_paths", []) or [])
        for sample in req_dict.get("samples", [])
    )


def _event_file_name(event: ProgressEvent) -> str | None:
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

    return messages.get(stage, stage.replace("_", " ").capitalize() + ".")


def _make_progress_callback(
    *,
    job_id: Optional[str],
    total_files: int,
) -> ProgressCallback:
    """
    Progress is tracked as file-level work units.

    Counted stages:
      - fit_qc
      - fit_lymphocytes
      - fit_control_stats
      - apply_file
      - plot_cache

    Global stages are shown as messages but do not increment done_files:
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

    total_work = max(1, total_files * len(counted_stages))
    done_work = 0
    done_filenames: list[str] = []

    def progress(event: ProgressEvent) -> None:
        nonlocal done_work, done_filenames

        if not job_id:
            return

        stage = str(event.get("stage") or "running")
        file_name = _event_file_name(event)

        if stage in counted_stages:
            done_work = min(done_work + 1, total_work)

        if stage == "plot_cache" and file_name:
            done_filenames.append(file_name)

        previous = FCXM_JOB_PROGRESS.get(job_id, {})

        FCXM_JOB_PROGRESS[job_id] = {
            **previous,
            "status": "running",
            "message": _stage_message(stage),
            "stage": stage,
            "total_files": total_work,
            "done_files": done_work,
            "current_file": file_name,
            "done_filenames": done_filenames,
        }

    return progress


def _init_progress(job_id: Optional[str], total_files: int) -> None:
    if not job_id:
        return

    total_work = max(1, total_files * 5)

    FCXM_JOB_PROGRESS[job_id] = {
        **FCXM_JOB_PROGRESS.get(job_id, {}),
        "status": "running",
        "message": "Starting flow cytometry analysis.",
        "stage": "starting",
        "total_files": total_work,
        "done_files": 0,
        "current_file": None,
        "done_filenames": [],
    }


def _mark_done(job_id: Optional[str]) -> None:
    if not job_id:
        return

    previous = FCXM_JOB_PROGRESS.get(job_id, {})
    total = int(previous.get("total_files") or 1)

    FCXM_JOB_PROGRESS[job_id] = {
        **previous,
        "status": "done",
        "message": "Analysis done.",
        "stage": "done",
        "done_files": total,
        "total_files": total,
        "current_file": None,
    }


def _mark_error(job_id: Optional[str], error: Exception) -> None:
    if not job_id:
        return

    previous = FCXM_JOB_PROGRESS.get(job_id, {})

    FCXM_JOB_PROGRESS[job_id] = {
        **previous,
        "status": "error",
        "message": "Analysis failed.",
        "stage": "error",
        "error": repr(error),
        "current_file": None,
    }


def _build_dataset(req_dict: Dict[str, Any]) -> Dataset:
    samples = []

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
    req_dict: Dict[str, Any],
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run FCXM analysis.

    req_dict is created from FCXMRunRequest.model_dump().

    It contains:
      - panel_rows: list of dicts
      - samples: list of dicts with file_paths relative to DATA_DIR

    Progress is tracked by file-level work units:
      - QC
      - lymphocyte gating
      - control stats
      - apply file
      - plot cache
    """
    total_files = _count_files(req_dict)
    _init_progress(job_id, total_files)

    progress = _make_progress_callback(
        job_id=job_id,
        total_files=total_files,
    )

    try:
        progress({"stage": "build_dataset"})
        ds = _build_dataset(req_dict)

        progress({"stage": "build_panel"})
        panel, marker_to_population = build_panel_from_rows(req_dict["panel_rows"])

        gater = Gater(panel, GatingConfig())

        fitted = gater.fit(
            ds,
            progress_cb=progress,
        )

        results = gater.apply(
            ds,
            fitted,
            progress_cb=progress,
        )

        progress({"stage": "payload"})

        payload, plot_cache = make_results_payload(
            ds=ds,
            gater=gater,
            fitted=fitted,
            results=results,
            marker_to_population=marker_to_population,
            max_points=5000,
            seed=0,
            progress_cb=progress,
        )

        _mark_done(job_id)

        return {
            "payload": payload,
            "plot_cache": plot_cache,
        }

    except Exception as e:
        _mark_error(job_id, e)
        print(f"FCXM analysis failed for job {job_id}: {repr(e)}")
        raise


def run_fcxm_analysis(
    req_dict: Dict[str, Any],
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    return run_fcxm_pipeline(
        req_dict=req_dict,
        job_id=job_id,
    )
