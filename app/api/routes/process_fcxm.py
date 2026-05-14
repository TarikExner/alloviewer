# api/routes/fcxm.py
import inspect
import os
import time
from pathlib import Path
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import Response

from alloviewer.flow_cytometry.pipeline import run_fcxm_analysis
from alloviewer.flow_cytometry.plots import build_results_response_from_cache
from alloviewer.flow_cytometry.pdf_report import build_fcxm_summary_pdf, ReportMeta

from ...models import (
    FCXMResultsRequest,
    FCXMResultsResponse,
    FCXMRunRequest,
    FCXMRunStartResponse,
    FCXMRunProgressResponse,
    FcsDisplayNamesRequest,
    FcsDisplayNamesResponse,
    FCXM_JOB_PROGRESS,
    FCXM_JOB_RESULTS,
    FCXM_JOB_PLOTS,
    FCXM_JOB_RUN_REQUESTS,
    fcxm_job_touch,
    fcxm_job_cleanup,
)

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
    """
    FCS tube names are not stored under one universal key.
    Try common keys and fall back to filename.
    """
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
    """
    Read FCS metadata with flowio.
    """
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


def _collect_original_filenames(req_dict: dict) -> list[str]:
    files: list[str] = []

    for sample in req_dict.get("samples", []):
        for fp in sample.get("file_paths", []) or []:
            if fp:
                files.append(str(fp))

    return files


def _resolve_request_file_paths(req_dict: dict) -> dict[str, str]:
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
            abs_to_original[_norm_path(abs_path)] = str(fp)

        sample["file_paths"] = abs_paths

    return abs_to_original


def _public_filename(value: str | None, abs_to_original: dict[str, str]) -> str | None:
    """
    Converts an absolute path reported by analysis code back to the original
    uploaded filename used by the frontend.
    """
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

def run_fcxm_job(job_id: str, req: FCXMRunRequest):
    try:
        req_dict = req.model_dump()
        original_files = _collect_original_filenames(req_dict)
        total_files = len(original_files)

        FCXM_JOB_PROGRESS[job_id] = {
            **FCXM_JOB_PROGRESS.get(job_id, {}),
            "status": "running",
            "message": "Starting flow cytometry analysis.",
            "stage": "starting",
            "total_files": max(1, total_files * 5),
            "done_files": 0,
            "current_file": None,
            "done_filenames": [],
            "last_access": time.time(),
        }

        abs_to_original = _resolve_request_file_paths(req_dict)

        # Store the mapping privately in the progress dict.
        # The GET endpoint uses it to convert absolute paths back to frontend filenames.
        FCXM_JOB_PROGRESS[job_id]["_abs_to_original"] = abs_to_original
        FCXM_JOB_PROGRESS[job_id]["_original_files"] = original_files

        result = run_fcxm_analysis(
            req_dict=req_dict,
            job_id=job_id,
        )

        FCXM_JOB_RESULTS[job_id] = result["payload"]
        FCXM_JOB_PLOTS[job_id] = result["plot_cache"]

        previous = FCXM_JOB_PROGRESS.get(job_id, {})
        total_work = int(previous.get("total_files") or max(1, total_files * 5))

        FCXM_JOB_PROGRESS[job_id] = {
            **previous,
            "status": "done",
            "message": "Done.",
            "stage": "done",
            "total_files": total_work,
            "done_files": total_work,
            "current_file": None,
            "done_filenames": original_files,
            "last_access": time.time(),
        }

        fcxm_job_touch(job_id)

    except Exception as e:
        FCXM_JOB_PROGRESS[job_id] = {
            **FCXM_JOB_PROGRESS.get(job_id, {}),
            "status": "error",
            "message": str(e),
            "stage": "error",
            "error": repr(e),
            "current_file": None,
            "last_access": time.time(),
        }
        fcxm_job_touch(job_id)
        print(f"FCXM job failed for {job_id}: {repr(e)}")

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
async def fcxm_run(req: FCXMRunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    now = time.time()

    req_dict = req.model_dump()
    original_files = _collect_original_filenames(req_dict)
    total_files = len(original_files)

    fcxm_job_cleanup()

    FCXM_JOB_PROGRESS[job_id] = {
        "status": "queued",
        "message": "Queued.",
        "stage": "queued",
        "created_at": now,
        "last_access": now,
        "total_files": max(1, total_files * 5),
        "done_files": 0,
        "current_file": None,
        "done_filenames": [],
    }

    FCXM_JOB_RUN_REQUESTS[job_id] = req_dict

    background_tasks.add_task(run_fcxm_job, job_id, req)
    return {"job_id": job_id}


@router.get("/run/{job_id}", response_model=FCXMRunProgressResponse)
async def fcxm_run_progress(job_id: str):
    fcxm_job_cleanup()

    if job_id not in FCXM_JOB_PROGRESS:
        raise HTTPException(status_code=404, detail="Job not found")

    fcxm_job_touch(job_id)

    prog = FCXM_JOB_PROGRESS[job_id]
    res = FCXM_JOB_RESULTS.get(job_id)

    abs_to_original = prog.get("_abs_to_original", {}) or {}

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
    fcxm_job_cleanup()

    cache = FCXM_JOB_PLOTS.get(req.job_id)
    if cache is None:
        raise HTTPException(status_code=404, detail="Job not found")

    fcxm_job_touch(req.job_id)

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
    fcxm_job_cleanup()

    prog = FCXM_JOB_PROGRESS.get(job_id)
    if prog is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if prog.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job status is {prog.get('status')}",
        )

    payload = FCXM_JOB_RESULTS.get(job_id)
    plot_cache = FCXM_JOB_PLOTS.get(job_id)
    if payload is None or plot_cache is None:
        raise HTTPException(status_code=404, detail="Job not found")

    fcxm_job_touch(job_id)

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
