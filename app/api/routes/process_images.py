import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse

from alloviewer.image_analysis.pipeline import run_image_analysis

from ...models import (
    IMAGE_JOB_PROGRESS,
    IMAGE_JOB_RESULTS,
    ProcessRequest,
    ProcessStartResponse,
    ProgressResponse,
)
from ...core.settings import settings
from .plate_layouts import get_repo

from alloviewer.image_analysis.storage.repo import LayoutRepo

router = APIRouter(tags=["process"])


@router.post("/api/process", response_model=ProcessStartResponse)
async def process(
    req: ProcessRequest,
    background_tasks: BackgroundTasks,
    repo: LayoutRepo = Depends(get_repo),
):
    job_id = str(uuid.uuid4())

    hla_layout = None

    if req.assay_type == "pra":
        if not req.hla_layout_upload_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "PRA analysis requires hla_layout_upload_id. "
                    "Upload and parse the Excel layout first via /parse, then pass "
                    "the returned upload_id to /api/process."
                ),
            )

        hla_layout = repo.get_by_id(req.hla_layout_upload_id)

        if hla_layout is None:
            raise HTTPException(
                status_code=404,
                detail=f"HLA layout not found: {req.hla_layout_upload_id}",
            )

    IMAGE_JOB_PROGRESS[job_id] = {
        "status": "queued",
        "done": 0,
        "total": 0,
        "current_well": None,
        "done_wells": [],
    }

    background_tasks.add_task(
        run_image_analysis,
        job_id=job_id,
        layout=req.layout,
        image_order=req.image_order,
        image_filenames=req.image_filenames,
        data_dir=str(settings.data_dir),
        template_filename=req.template_filename,
        assay_type=req.assay_type,
        hla_layout=hla_layout,
        pra_positivity_threshold=req.pra_positivity_threshold,
    )

    return {"job_id": job_id}


@router.get("/api/process/{job_id}", response_model=ProgressResponse)
async def get_process(job_id: str):
    if job_id not in IMAGE_JOB_PROGRESS:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = IMAGE_JOB_PROGRESS[job_id]
    result = IMAGE_JOB_RESULTS.get(job_id)

    return {**progress, "result": result}


@router.get("/api/process/{job_id}/segmented/{well_id}.png")
async def get_segmented_image(job_id: str, well_id: str):
    path = Path(settings.data_dir) / "segmented" / job_id / f"{well_id}.png"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Segmented image not found")

    return FileResponse(path, media_type="image/png")
