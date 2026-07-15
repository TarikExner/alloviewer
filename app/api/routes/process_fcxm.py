from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from alloviewer.flow_cytometry.pdf_report import (
    ReportMeta,
    build_fcxm_summary_pdf,
)
from alloviewer.flow_cytometry.plots import (
    build_results_response_from_cache,
)

from app.models import (
    FCXMResultsRequest,
    FCXMResultsResponse,
    FCXMRunProgressResponse,
    FCXMRunRequest,
    FCXMRunStartResponse,
    FcsDisplayNamesRequest,
    FcsDisplayNamesResponse,
)
from app.services.fcxm_cache_storage import (
    load_fcxm_plot_cache,
)
from app.services.job_paths import (
    JobPathError,
    get_job_paths,
    resolve_job_path,
)
from app.services.job_registry import (
    require_job_type,
    write_json_atomic,
)
from app.services.job_state import (
    delete_fcxm_job,
    get_fcxm_progress,
    get_fcxm_result,
    set_fcxm_progress,
)
from app.workers.tasks_fcxm import run_fcxm_job_task


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/jobs/{job_id}/fcxm",
    tags=["fcxm"],
)


def _require_fcxm_job(
    job_id: str,
) -> dict[str, Any]:
    try:
        return require_job_type(
            job_id,
            {"fcxm"},
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )


def _norm_path(
    value: str | Path,
) -> str:
    return os.path.normcase(
        os.path.normpath(
            str(Path(value).resolve())
        )
    )


def _safe_decode(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        for encoding in (
            "utf-8",
            "latin-1",
            "cp1252",
        ):
            try:
                return value.decode(
                    encoding
                ).strip()
            except Exception:
                continue

        return value.decode(
            errors="ignore"
        ).strip()

    return str(value).strip()


def _normalize_meta_key(
    key: Any,
) -> str:
    return (
        str(key)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _extract_tube_name_from_meta(
    metadata: dict[Any, Any],
    fallback: str,
) -> str:
    if not metadata:
        return fallback

    normalized: dict[str, str] = {}

    for key, value in metadata.items():
        decoded = _safe_decode(value)

        if decoded:
            normalized[
                _normalize_meta_key(key)
            ] = decoded

    candidate_keys = [
        "$tube name",
        "tube name",
        "$tubename",
        "tubename",
        "tube",
        "tube name:",
        "sample name",
        "$sample",
        "sample",
        "$src",
        "src",
        "name",
    ]

    for key in candidate_keys:
        value = normalized.get(
            _normalize_meta_key(key)
        )

        if value:
            return value

    return fallback


def _read_fcs_metadata(
    path: Path,
) -> dict[Any, Any]:
    from flowio import FlowData

    flow_data = FlowData(
        str(path)
    )

    return dict(
        getattr(
            flow_data,
            "text",
            {},
        )
        or {}
    )


def _resolve_fcs_upload_path(
    *,
    job_id: str,
    filename: str,
) -> Path:
    paths = get_job_paths(job_id)

    try:
        resolved = resolve_job_path(
            job_id,
            filename,
            required_root=paths.fcs_uploads,
            must_exist=True,
        )
    except JobPathError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid FCS path "
                f"'{filename}': {exc}"
            ),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=(
                "FCS file was not found "
                f"in this job: {filename}"
            ),
        )

    if not resolved.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                "FCS path is not a file: "
                f"{filename}"
            ),
        )

    if resolved.suffix.lower() != ".fcs":
        raise HTTPException(
            status_code=415,
            detail=(
                "File is not an .fcs file: "
                f"{filename}"
            ),
        )

    return resolved


def _collect_request_filenames(
    request_dict: dict[str, Any],
) -> list[str]:
    filenames: list[str] = []

    for sample in request_dict.get(
        "samples",
        [],
    ):
        for filename in (
            sample.get("file_paths", [])
            or []
        ):
            if filename:
                filenames.append(
                    str(filename)
                )

    return filenames


def _validate_request_files(
    *,
    job_id: str,
    request_dict: dict[str, Any],
) -> list[str]:
    filenames = _collect_request_filenames(
        request_dict
    )

    if not filenames:
        raise HTTPException(
            status_code=400,
            detail=(
                "The FCXM request contains "
                "no FCS files."
            ),
        )

    for filename in filenames:
        _resolve_fcs_upload_path(
            job_id=job_id,
            filename=filename,
        )

    return filenames


def _find_plot_cache_key(
    *,
    cache: dict[Any, Any],
    resolved_path: Path,
) -> str:
    expected = _norm_path(
        resolved_path
    )

    for raw_key in cache:
        key = str(raw_key)

        if _norm_path(key) == expected:
            return key

    # Basename fallback supports caches produced before every internal path
    # was normalized consistently. It is accepted only when unambiguous.
    basename_matches = [
        str(raw_key)
        for raw_key in cache
        if Path(
            str(raw_key)
        ).name
        == resolved_path.name
    ]

    if len(basename_matches) == 1:
        return basename_matches[0]

    examples = [
        str(key)
        for key in list(
            cache.keys()
        )[:5]
    ]

    raise HTTPException(
        status_code=404,
        detail=(
            "File was not found in the "
            "FCXM plot cache: "
            f"{resolved_path.name}. "
            f"Example cache keys: {examples}"
        ),
    )


def _clear_previous_outputs(
    job_id: str,
) -> None:
    paths = get_job_paths(
        job_id,
        create=True,
    )

    paths.result.unlink(
        missing_ok=True
    )
    paths.plot_cache.unlink(
        missing_ok=True
    )
    paths.summary_pdf.unlink(
        missing_ok=True
    )

    delete_fcxm_job(job_id)


@router.post(
    "/fcs-display-names",
    response_model=FcsDisplayNamesResponse,
)
async def fcxm_fcs_display_names(
    job_id: str,
    req: FcsDisplayNamesRequest,
):
    _require_fcxm_job(job_id)

    names: dict[str, str] = {}

    for filename in req.filenames:
        fallback = Path(
            filename
        ).name

        if req.mode == "filename":
            names[filename] = fallback
            continue

        try:
            path = _resolve_fcs_upload_path(
                job_id=job_id,
                filename=filename,
            )

            metadata = _read_fcs_metadata(
                path
            )

            names[filename] = (
                _extract_tube_name_from_meta(
                    metadata,
                    fallback,
                )
            )

        except HTTPException:
            raise

        except Exception:
            logger.exception(
                "Could not read FCS metadata: "
                "job_id=%s file=%s",
                job_id,
                filename,
            )

            names[filename] = fallback

    return FcsDisplayNamesResponse(
        names=names
    )


@router.post(
    "/run",
    response_model=FCXMRunStartResponse,
)
async def fcxm_run(
    job_id: str,
    req: FCXMRunRequest,
):
    job = _require_fcxm_job(
        job_id
    )

    current_status = str(
        job.get("status")
        or "draft"
    )

    if current_status in {
        "queued",
        "running",
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                "This FCXM job is already "
                "queued or running."
            ),
        )

    if current_status == "done":
        raise HTTPException(
            status_code=409,
            detail=(
                "This FCXM job has already "
                "completed. Create a new job "
                "to run another analysis."
            ),
        )

    request_dict = req.model_dump()

    original_filenames = (
        _validate_request_files(
            job_id=job_id,
            request_dict=request_dict,
        )
    )

    _clear_previous_outputs(
        job_id
    )

    paths = get_job_paths(
        job_id,
        create=True,
    )

    write_json_atomic(
        paths.request,
        request_dict,
    )

    total_work = max(
        1,
        len(original_filenames) * 5,
    )

    queued_progress = {
        "status": "queued",
        "message": "Queued.",
        "stage": "queued",
        "total_files": total_work,
        "done_files": 0,
        "current_file": None,
        "done_filenames": [],
        "error": None,
        "error_type": None,
        "failed_stage": None,
        "failed_file": None,
        "support_id": job_id,
    }

    set_fcxm_progress(
        job_id,
        queued_progress,
    )

    try:
        run_fcxm_job_task.delay(
            job_id
        )

    except Exception:
        logger.exception(
            "Could not queue FCXM job %s.",
            job_id,
        )

        error_progress = {
            **queued_progress,
            "status": "error",
            "message": (
                "The FCXM analysis could not "
                "be added to the worker queue."
            ),
            "stage": "error",
            "failed_stage": "queued",
            "error": (
                "The analysis could not be "
                "added to the worker queue."
            ),
            "error_type": "QueueError",
        }

        set_fcxm_progress(
            job_id,
            error_progress,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The FCXM analysis could not "
                "be added to the worker queue. "
                f"Job ID: {job_id}"
            ),
        )

    return FCXMRunStartResponse(
        job_id=job_id
    )


@router.get(
    "/progress",
    response_model=FCXMRunProgressResponse,
)
async def fcxm_run_progress(
    job_id: str,
):
    _require_fcxm_job(job_id)

    progress = get_fcxm_progress(
        job_id
    )

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Progress information was not "
                "found for this job."
            ),
        )

    result = get_fcxm_result(
        job_id
    )

    return {
        "status": progress.get(
            "status",
            "queued",
        ),
        "message": progress.get(
            "message"
        ),
        "stage": progress.get(
            "stage"
        ),
        "result": result,
        "total_files": progress.get(
            "total_files"
        ),
        "done_files": progress.get(
            "done_files"
        ),
        "current_file": progress.get(
            "current_file"
        ),
        "done_filenames": progress.get(
            "done_filenames",
            [],
        ),
        "error": progress.get(
            "error"
        ),
        "error_type": progress.get(
            "error_type"
        ),
        "failed_stage": progress.get(
            "failed_stage"
        ),
        "failed_file": progress.get(
            "failed_file"
        ),
        "support_id": (
            progress.get("support_id")
            or job_id
        ),
    }


@router.post(
    "/results",
    response_model=FCXMResultsResponse,
)
async def fcxm_results(
    job_id: str,
    req: FCXMResultsRequest,
):
    _require_fcxm_job(job_id)

    progress = get_fcxm_progress(
        job_id
    )

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if progress.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail=(
                "FCXM results are only available "
                "after analysis has completed."
            ),
        )

    try:
        cache = load_fcxm_plot_cache(
            job_id=job_id
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "The FCXM plot cache was "
                "not found."
            ),
        )
    except Exception:
        logger.exception(
            "Could not load FCXM plot cache: %s",
            job_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The FCXM plot cache could "
                "not be loaded."
            ),
        )

    resolved_path = _resolve_fcs_upload_path(
        job_id=job_id,
        filename=req.fcs_filename,
    )

    selected_key = _find_plot_cache_key(
        cache=cache,
        resolved_path=resolved_path,
    )

    data = build_results_response_from_cache(
        plot_cache=cache,
        selected_gate=req.gate,
        selected_key=selected_key,
    )

    return FCXMResultsResponse(
        **data
    )


@router.get("/summary.pdf")
async def fcxm_summary_pdf(
    job_id: str,
    positivity_metric: str | None = None,
    positivity_threshold: float | None = None,
):
    _require_fcxm_job(job_id)

    progress = get_fcxm_progress(
        job_id
    )

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if progress.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail=(
                "The FCXM summary is only "
                "available after analysis "
                "has completed."
            ),
        )

    payload = get_fcxm_result(
        job_id
    )

    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "The FCXM result payload "
                "was not found."
            ),
        )

    try:
        plot_cache = load_fcxm_plot_cache(
            job_id=job_id
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "The FCXM plot cache was "
                "not found."
            ),
        )
    except Exception:
        logger.exception(
            "Could not load FCXM plot cache: %s",
            job_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The FCXM plot cache could "
                "not be loaded."
            ),
        )

    try:
        pdf_bytes = build_fcxm_summary_pdf(
            payload=payload,
            plot_cache=plot_cache,
            meta=ReportMeta(
                job_id=job_id,
                positivity_metric=positivity_metric,
                positivity_threshold=positivity_threshold,
            ),
        )

        paths = get_job_paths(
            job_id,
            create=True,
        )

        paths.summary_pdf.write_bytes(
            pdf_bytes
        )

    except Exception:
        logger.exception(
            "Could not build FCXM summary: %s",
            job_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The FCXM summary PDF could "
                "not be generated."
            ),
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                "attachment; "
                f'filename="fcxm_summary_{job_id}.pdf"'
            ),
        },
    )
