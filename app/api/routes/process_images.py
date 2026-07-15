from __future__ import annotations

import logging
import re
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse

from alloviewer.image_analysis.services.pdf_report import build_cdc_summary_pdf

from app.config import IMAGE_EXTENSIONS
from app.models import ProcessRequest, ProcessStartResponse, ProgressResponse
from app.services.job_paths import JobPathError, get_job_paths, resolve_job_path
from app.services.job_registry import require_job_type, update_job, write_json_atomic
from app.services.job_state import (
    delete_image_job,
    get_image_progress,
    get_image_result,
    set_image_progress,
)
from app.workers.tasks_image import run_image_analysis_task


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/jobs/{job_id}/image",
    tags=["image-analysis"],
)

WELL_ID_PATTERN = re.compile(r"^[A-Za-z]+\d+$")
ALLOWED_CROSSMATCH_CELL_MODES = {"T", "B", "T/B", "empty"}


def _require_image_job(job_id: str) -> dict[str, Any]:
    try:
        return require_job_type(job_id, {"pra", "crossmatch"})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


def _validate_image_files(*, job_id: str, image_filenames: list[str]) -> None:
    if not image_filenames:
        raise HTTPException(
            status_code=400,
            detail="At least one image file is required.",
        )

    paths = get_job_paths(job_id)

    for filename in image_filenames:
        try:
            resolved = resolve_job_path(
                job_id,
                filename,
                required_root=paths.image_uploads,
                must_exist=True,
            )
        except JobPathError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image path '{filename}': {exc}",
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail=f"Image file not found in this job: {filename}",
            )

        if not resolved.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Image path is not a file: {filename}",
            )

        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type: {filename}",
            )


def _well_column(well_id: str) -> int:
    match = re.search(r"(\d+)$", well_id)

    if match is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid well ID in image order: {well_id}",
        )

    column = int(match.group(1))

    if column < 1 or column > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Well column is outside the supported range: {well_id}",
        )

    return column


def _validated_column_modes(
    *,
    job_type: str,
    image_order: list[str],
    column_modes: dict[int, str],
) -> dict[int, str]:
    if job_type != "crossmatch":
        return {}

    normalized: dict[int, str] = {}

    for raw_column, raw_mode in column_modes.items():
        column = int(raw_column)
        mode = str(raw_mode)

        if column < 1 or column > 10:
            raise HTTPException(
                status_code=400,
                detail=f"Crossmatch column index must be between 1 and 10: {column}",
            )

        if mode not in ALLOWED_CROSSMATCH_CELL_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported crossmatch cell mode for column {column}: {mode}",
            )

        normalized[column] = mode

    used_columns = sorted({_well_column(str(well_id)) for well_id in image_order})
    missing_columns = [column for column in used_columns if column not in normalized]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Crossmatch cell-type labels are missing for column(s): "
                + ", ".join(map(str, missing_columns))
            ),
        )

    empty_used_columns = [
        column
        for column in used_columns
        if normalized.get(column) == "empty"
    ]

    if empty_used_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Columns containing uploaded wells cannot use the empty cell mode: "
                + ", ".join(map(str, empty_used_columns))
            ),
        )

    return normalized


def _clear_previous_image_outputs(job_id: str) -> None:
    """
    Remove outputs from an earlier failed attempt while preserving uploads,
    parsed plate layouts, thumbnails, metadata, and logs.
    """
    paths = get_job_paths(job_id, create=True)

    if paths.segmented.exists():
        shutil.rmtree(paths.segmented)

    paths.segmented.mkdir(parents=True, exist_ok=True)
    paths.result.unlink(missing_ok=True)
    paths.summary_pdf.unlink(missing_ok=True)

    for summary_path in paths.reports.glob("summary*.pdf"):
        summary_path.unlink(missing_ok=True)

    delete_image_job(job_id)


@router.post("/run", response_model=ProcessStartResponse)
async def process(job_id: str, req: ProcessRequest):
    job = _require_image_job(job_id)
    paths = get_job_paths(job_id, create=True)
    current_status = str(job.get("status") or "draft")
    job_type = str(job.get("job_type") or "")

    if current_status in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail="This job is already queued or running.",
        )

    if current_status == "done":
        raise HTTPException(
            status_code=409,
            detail=(
                "This job has already completed. "
                "Create a new job to run another analysis."
            ),
        )

    if len(req.image_order) != len(req.image_filenames):
        raise HTTPException(
            status_code=400,
            detail=(
                "The number of image files must match the number "
                "of wells in the image order."
            ),
        )

    if len(set(req.image_order)) != len(req.image_order):
        raise HTTPException(
            status_code=400,
            detail="The image order contains duplicate wells.",
        )

    _validate_image_files(
        job_id=job_id,
        image_filenames=req.image_filenames,
    )

    if job_type == "pra" and not paths.plate_layout.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "PRA analysis requires a parsed plate layout. "
                "Upload the Excel layout before starting the analysis."
            ),
        )

    column_modes = _validated_column_modes(
        job_type=job_type,
        image_order=[str(well_id) for well_id in req.image_order],
        column_modes={int(key): str(value) for key, value in req.column_modes.items()},
    )

    _clear_previous_image_outputs(job_id)

    request_payload = {
        "schema_version": 2,
        "layout": jsonable_encoder(req.layout),
        "image_order": jsonable_encoder(req.image_order),
        "image_filenames": list(req.image_filenames),
        "pra_positivity_threshold": req.pra_positivity_threshold,
        "column_modes": {str(column): mode for column, mode in column_modes.items()},
        "flip_vertical": bool(req.flip_vertical),
    }

    write_json_atomic(paths.request, request_payload)

    queued_progress = {
        "status": "queued",
        "stage": "queued",
        "done": 0,
        "total": 0,
        "current_well": None,
        "done_wells": [],
        "error": None,
        "error_type": None,
        "failed_stage": None,
        "failed_well": None,
        "support_id": job_id,
    }

    set_image_progress(job_id, queued_progress)

    update_job(
        job_id,
        status="queued",
        stage="queued",
        error=None,
        progress=queued_progress,
    )

    try:
        run_image_analysis_task.delay(job_id)
    except Exception:
        logger.exception("Could not queue image-analysis job %s.", job_id)

        error_progress = {
            **queued_progress,
            "status": "error",
            "stage": "queued",
            "error": "The analysis could not be added to the worker queue.",
            "error_type": "QueueError",
            "failed_stage": "queued",
        }

        set_image_progress(job_id, error_progress)

        update_job(
            job_id,
            status="error",
            stage="queued",
            error={
                "message": error_progress["error"],
                "type": error_progress["error_type"],
            },
            progress=error_progress,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The analysis could not be added to the worker queue. "
                f"Job ID: {job_id}"
            ),
        )

    return {"job_id": job_id}


@router.get("/progress", response_model=ProgressResponse)
async def get_process(job_id: str):
    _require_image_job(job_id)

    progress = get_image_progress(job_id)

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail="Progress information was not found for this job.",
        )

    result = get_image_result(job_id)

    return {
        **progress,
        "support_id": progress.get("support_id") or job_id,
        "result": result,
    }


@router.get("/segmented/{well_id}.png")
async def get_segmented_image(job_id: str, well_id: str):
    _require_image_job(job_id)

    if not WELL_ID_PATTERN.fullmatch(well_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid well ID.",
        )

    path = get_job_paths(job_id).segmented / f"{well_id}.png"

    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Segmented image not found.",
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{well_id}.png",
    )


@router.get("/summary.pdf")
async def get_cdc_summary_pdf(job_id: str, flip_vertical: bool = False):
    _require_image_job(job_id)
    progress = get_image_progress(job_id)

    if progress is None:
        raise HTTPException(status_code=404, detail="Progress information was not found for this job.")

    if progress.get("status") != "done":
        raise HTTPException(status_code=409, detail="The summary PDF is only available after the analysis has completed.")

    result = get_image_result(job_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    paths = get_job_paths(job_id, create=True)
    summary_path = paths.reports / ("summary_flipped.pdf" if flip_vertical else "summary_standard.pdf")

    try:
        pdf = build_cdc_summary_pdf(result=result, job_id=job_id, flip_vertical=flip_vertical)
        summary_path.write_bytes(pdf)
    except Exception:
        logger.exception("Could not generate CDC summary for job %s.", job_id)
        raise HTTPException(status_code=500, detail=f"The summary PDF could not be generated. Job ID: {job_id}")

    return FileResponse(
        summary_path,
        media_type="application/pdf",
        filename=f"cdc_summary_{job_id}.pdf",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )
