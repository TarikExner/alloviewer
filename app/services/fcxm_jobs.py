import logging
import os
from pathlib import Path

from fastapi import HTTPException

from alloviewer.flow_cytometry.pipeline import run_fcxm_analysis

from app.models import FCXMRunRequest
from app.core.settings import settings
from app.core.paths import resolve_under_base_dir

from app.services.job_state import (
    update_fcxm_progress,
    get_fcxm_progress,
    set_fcxm_result,
    set_fcxm_request,
    set_fcxm_plot_cache_path,
)

from app.services.fcxm_cache_storage import save_fcxm_plot_cache


logger = logging.getLogger(__name__)


def norm_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(Path(p).resolve())))


def collect_original_filenames(req_dict: dict) -> list[str]:
    files: list[str] = []

    for sample in req_dict.get("samples", []):
        for fp in sample.get("file_paths", []) or []:
            if fp:
                files.append(str(fp))

    return files


def resolve_request_file_paths(req_dict: dict) -> dict[str, str]:
    """
    Converts request file paths to absolute paths in-place.

    Returns:
        Mapping from normalized absolute path to original uploaded filename.
    """
    abs_to_original: dict[str, str] = {}

    for sample in req_dict.get("samples", []):
        abs_paths = []

        for fp in sample.get("file_paths", []) or []:
            dest = resolve_under_base_dir(settings.data_dir, fp)

            if dest.suffix.lower() != ".fcs":
                raise HTTPException(status_code=415, detail=f"Not an .fcs file: {fp}")

            if not dest.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {fp}")

            abs_path = str(dest)
            abs_paths.append(abs_path)
            abs_to_original[norm_path(abs_path)] = str(fp)

        sample["file_paths"] = abs_paths

    return abs_to_original


def run_fcxm_job(job_id: str, req: FCXMRunRequest) -> None:
    logger.info("Starting FCXM job %s", job_id)

    try:
        req_dict = req.model_dump()
        original_files = collect_original_filenames(req_dict)
        total_files = len(original_files)

        update_fcxm_progress(
            job_id,
            status="running",
            message="Starting flow cytometry analysis.",
            stage="starting",
            total_files=max(1, total_files * 5),
            done_files=0,
            current_file=None,
            done_filenames=[],
        )

        abs_to_original = resolve_request_file_paths(req_dict)

        req_dict["_abs_to_original"] = abs_to_original
        req_dict["_original_files"] = original_files
        set_fcxm_request(job_id, req_dict)

        result = run_fcxm_analysis(
            req_dict=req_dict,
            job_id=job_id,
        )

        set_fcxm_result(job_id, result["payload"])

        plot_cache_path = save_fcxm_plot_cache(
            job_id=job_id,
            plot_cache=result["plot_cache"],
        )
        set_fcxm_plot_cache_path(job_id, plot_cache_path)

        previous = get_fcxm_progress(job_id) or {}
        total_work = int(previous.get("total_files") or max(1, total_files * 5))

        update_fcxm_progress(
            job_id,
            status="done",
            message="Done.",
            stage="done",
            total_files=total_work,
            done_files=total_work,
            current_file=None,
            done_filenames=original_files,
        )

        logger.info("Finished FCXM job %s", job_id)

    except Exception as e:
        logger.exception("FCXM job failed: %s", job_id)

        update_fcxm_progress(
            job_id,
            status="error",
            message=str(e),
            stage="error",
            error=repr(e),
            current_file=None,
        )

        raise
