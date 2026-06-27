import os
import time
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from alloviewer.flow_cytometry.plots import build_results_response_from_cache
from alloviewer.flow_cytometry.pdf_report import (
    build_fcxm_summary_pdf,
    ReportMeta,
)

from ...models import (
    FCXMResultsRequest,
    FCXMResultsResponse,
    FCXMRunRequest,
    FCXMRunStartResponse,
    FCXMRunProgressResponse,
    FcsDisplayNamesRequest,
    FcsDisplayNamesResponse,
)

from app.services.job_state import (
    set_fcxm_progress,
    get_fcxm_progress,
    set_fcxm_request,
    get_fcxm_request,
    get_fcxm_result,
    get_fcxm_plot_cache_path,
)

from app.services.fcxm_cache_storage import load_fcxm_plot_cache
from app.services.fcxm_jobs import collect_original_filenames
from app.workers.tasks_fcxm import run_fcxm_job_task

from ...core.settings import settings
from ...core.paths import resolve_under_base_dir


router = APIRouter(prefix="/api/fcxm", tags=["fcxm"])


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(Path(p).resolve())))


def _safe_decode(value) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return value.decode(enc).strip()
            except Exception:
                pass
        return value.decode(errors="ignore").strip()

    return str(value).strip()


def _normalize_meta_key(key) -> str:
    return str(key).strip().lower().replace("_", " ").replace("-", " ")


def _extract_tube_name_from_meta(meta: dict, fallback: str) -> str:
    if not meta:
        return fallback

    normalized = {
        _normalize_meta_key(k): _safe_decode(v)
        for k, v in meta.items()
        if _safe_decode(v)
    }

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
        value = normalized.get(_normalize_meta_key(key))
        if value:
            return value

    return fallback


def _read_fcs_metadata(path: Path) -> dict:
    from flowio import FlowData

    fd = FlowData(str(path))
    return dict(getattr(fd, "text", {}) or {})


def _resolve_fcs_upload_path(filename: str) -> Path:
    dest = resolve_under_base_dir(settings.data_dir, filename)

    if dest.suffix.lower() != ".fcs":
        raise HTTPException(status_code=415, detail=f"Not an .fcs file: {filename}")

    if not dest.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {filename}")

    return dest


def _public_filename(
    value: str | None,
    abs_to_original: dict[str, str],
) -> str | None:
    if not value:
        return None

    text = str(value)
    norm = _norm_path(text)

    if norm in abs_to_original:
        return abs_to_original[norm]

    return text


def _public_filenames(
    values: list[str] | None,
    abs_to_original: dict[str, str],
) -> list[str]:
    out: list[str] = []

    for value in values or []:
        public = _public_filename(value, abs_to_original)
        if public:
            out.append(public)

    return out


@router.post("/fcs-display-names", response_model=FcsDisplayNamesResponse)
async def fcxm_fcs_display_names(req: FcsDisplayNamesRequest):
    names: dict[str, str] = {}

    for filename in req.filenames:
        fallback = Path(filename).name

        if req.mode == "filename":
            names[filename] = fallback
            continue

        try:
            path = _resolve_fcs_upload_path(filename)
            meta = _read_fcs_metadata(path)
            names[filename] = _extract_tube_name_from_meta(meta, fallback)
        except HTTPException:
            raise
        except Exception:
            names[filename] = fallback

    return FcsDisplayNamesResponse(names=names)


@router.post("/run", response_model=FCXMRunStartResponse)
async def fcxm_run(req: FCXMRunRequest):
    job_id = str(uuid.uuid4())
    now = time.time()

    req_dict = req.model_dump()
    original_files = collect_original_filenames(req_dict)
    total_files = len(original_files)

    set_fcxm_progress(
        job_id,
        {
            "status": "queued",
            "message": "Queued.",
            "stage": "queued",
            "created_at": now,
            "last_access": now,
            "total_files": max(1, total_files * 5),
            "done_files": 0,
            "current_file": None,
            "done_filenames": [],
        },
    )

    set_fcxm_request(job_id, req_dict)

    run_fcxm_job_task.delay(job_id, req_dict)

    return {"job_id": job_id}


@router.get("/run/{job_id}", response_model=FCXMRunProgressResponse)
async def fcxm_run_progress(job_id: str):
    prog = get_fcxm_progress(job_id)

    if prog is None:
        raise HTTPException(status_code=404, detail="Job not found")

    res = get_fcxm_result(job_id)

    req_dict = get_fcxm_request(job_id) or {}
    abs_to_original = req_dict.get("_abs_to_original", {}) or {}

    current_file = _public_filename(
        prog.get("current_file"),
        abs_to_original,
    )

    done_filenames = _public_filenames(
        prog.get("done_filenames", []),
        abs_to_original,
    )

    return {
        "status": prog.get("status", "queued"),
        "message": prog.get("message"),
        "stage": prog.get("stage"),
        "result": res,
        "total_files": prog.get("total_files"),
        "done_files": prog.get("done_files"),
        "current_file": current_file,
        "done_filenames": done_filenames,
    }


@router.post("/results", response_model=FCXMResultsResponse)
async def fcxm_results(req: FCXMResultsRequest):
    plot_cache_path = get_fcxm_plot_cache_path(req.job_id)

    if not plot_cache_path:
        raise HTTPException(status_code=404, detail="Job not found")

    cache = load_fcxm_plot_cache(plot_cache_path)

    abs_path = str(resolve_under_base_dir(settings.data_dir, req.fcs_filename))
    key = _norm_path(abs_path)

    file_cache = cache.get(key)
    if file_cache is None:
        some = list(cache.keys())[:5]
        raise HTTPException(
            status_code=404,
            detail=(
                f"File not found in plot cache: {req.fcs_filename} "
                f"(resolved: {abs_path}) "
                f"(norm key: {key}) "
                f"(example keys: {some})"
            ),
        )

    data = build_results_response_from_cache(
        plot_cache=cache,
        selected_gate=req.gate,
        selected_key=key,
    )

    return FCXMResultsResponse(**data)


@router.get("/summary/{job_id}")
async def fcxm_summary_pdf(
    job_id: str,
    positivity_metric: str | None = None,
    positivity_threshold: float | None = None,
):
    prog = get_fcxm_progress(job_id)

    if prog is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if prog.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job status is {prog.get('status')}",
        )

    payload = get_fcxm_result(job_id)
    plot_cache_path = get_fcxm_plot_cache_path(job_id)

    if payload is None or not plot_cache_path:
        raise HTTPException(status_code=404, detail="Job not found")

    plot_cache = load_fcxm_plot_cache(plot_cache_path)

    pdf_bytes = build_fcxm_summary_pdf(
        payload=payload,
        plot_cache=plot_cache,
        meta=ReportMeta(
            job_id=job_id,
            positivity_metric=positivity_metric,
            positivity_threshold=positivity_threshold,
        ),
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="fcxm_summary_{job_id}.pdf"'
        },
    )
