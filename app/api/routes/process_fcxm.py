# api/routes/fcxm.py
import os
import time
from pathlib import Path
import random
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import Response

from alloviewer.main import run_fcxm_analysis
from alloviewer.flow_cytometry.plots import build_results_response_from_cache
from alloviewer.flow_cytometry.pdf_report import build_fcxm_summary_pdf, ReportMeta

from ...models import (
    FCXMResultsRequest,
    FCXMResultsResponse,
    FCXMRunRequest,
    FCXMRunStartResponse,
    FCXMRunProgressResponse,
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


def _seed_from(s: str) -> int:
    return sum(ord(c) for c in s) % (2**31 - 1)


def _make_cloud(
    rng: random.Random, n: int, cx: float, cy: float, sx: float, sy: float
):
    return [(rng.gauss(cx, sx), rng.gauss(cy, sy)) for _ in range(n)]


def run_fcxm_job(job_id: str, req: FCXMRunRequest):
    try:
        FCXM_JOB_PROGRESS[job_id] = {
            "status": "running",
            "message": "Running…",
            "created_at": time.time(),
            "last_access": time.time(),
        }

        req_dict = req.model_dump()

        # Resolve all file paths to absolute paths under settings.data_dir
        for s in req_dict.get("samples", []):
            abs_paths = []
            for fp in s.get("file_paths", []):
                dest = resolve_under_base_dir(settings.data_dir, fp)
                if dest.suffix.lower() != ".fcs":
                    raise HTTPException(
                        status_code=415, detail=f"Not an .fcs file: {fp}"
                    )
                if not dest.exists():
                    raise HTTPException(
                        status_code=400, detail=f"File not found: {fp}"
                    )
                abs_paths.append(str(dest))
            s["file_paths"] = abs_paths

        result = run_fcxm_analysis(req_dict)
        print("result is returned")

        FCXM_JOB_RESULTS[job_id] = result["payload"]
        FCXM_JOB_PLOTS[job_id] = result["plot_cache"]

        FCXM_JOB_PROGRESS[job_id] = {
            **FCXM_JOB_PROGRESS.get(job_id, {}),
            "status": "done",
            "message": "Done.",
        }
        fcxm_job_touch(job_id)

    except Exception as e:
        FCXM_JOB_PROGRESS[job_id] = {
            "status": "error",
            "message": str(e),
            **FCXM_JOB_PROGRESS.get(job_id, {}),
        }
        fcxm_job_touch(job_id)


@router.post("/run", response_model=FCXMRunStartResponse)
async def fcxm_run(req: FCXMRunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    now = time.time()

    fcxm_job_cleanup()
    FCXM_JOB_PROGRESS[job_id] = {
        "status": "queued",
        "message": "Queued.",
        "created_at": now,
        "last_access": now,
    }
    FCXM_JOB_RUN_REQUESTS[job_id] = req.model_dump()

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

    return {
        "status": prog.get("status", "queued"),
        "message": prog.get("message"),
        "result": res,
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
                f"File not found in plot cache: {req.fcs_filename} (resolved: {abs_path}) "
                f"(norm key: {key}) (example keys: {some})"
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
        raise HTTPException(status_code=409, detail=f"Job status is {prog.get('status')}")

    payload = FCXM_JOB_RESULTS.get(job_id)
    plot_cache = FCXM_JOB_PLOTS.get(job_id)
    if payload is None or plot_cache is None:
        raise HTTPException(status_code=404, detail="Job not found")

    fcxm_job_touch(job_id)
    
    import pickle
    with open("C:/Users/tarik/Lab/LCTeller/payload.data", "wb") as file:
        pickle.dump(payload, file)
    with open("C:/Users/tarik/Lab/LCTeller/plot_cache.data", "wb") as file:
        pickle.dump(plot_cache, file)
    with open("C:/Users/tarik/Lab/LCTeller/meta.data", "wb") as file:
        pickle.dump(
            ReportMeta(
                job_id=job_id,
                positivity_metric=positivity_metric,
                positivity_threshold=positivity_threshold,
            ),
            file
        )


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
        headers={"Content-Disposition": f'attachment; filename="fcxm_summary_{job_id}.pdf"'},
    )

