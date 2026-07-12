import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.encoders import jsonable_encoder

from ...models import (
    ProcessRequest,
    ProcessStartResponse,
    ProgressResponse,
)
from ...core.settings import settings
from .plate_layouts import get_repo

from alloviewer.image_analysis.storage.repo import LayoutRepo
from alloviewer.image_analysis.services.pdf_report import build_cdc_summary_pdf

from app.services.job_state import (
    set_image_progress,
    get_image_progress,
    get_image_result,
)
from app.workers.tasks_image import run_image_analysis_task

def to_celery_payload(value):
    """
    Convert Pydantic/dataclass/custom nested objects into JSON-safe data
    before sending them through Celery.
    """
    return jsonable_encoder(value)

router = APIRouter(tags=["process"])


@router.post("/api/process", response_model=ProcessStartResponse)
async def process(
    req: ProcessRequest,
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

    set_image_progress(
        job_id,
        {
            "status": "queued",
            "stage": "queued",
            "done": 0,
            "total": 0,
            "current_well": None,
            "done_wells": [],
        },
    )

    layout_payload = to_celery_payload(req.layout)
    hla_layout_payload = to_celery_payload(hla_layout)

    run_image_analysis_task.delay(
        job_id=job_id,
        layout=layout_payload,
        image_order=to_celery_payload(req.image_order),
        image_filenames=to_celery_payload(req.image_filenames),
        data_dir=str(settings.data_dir),
        template_filename=req.template_filename,
        assay_type=req.assay_type,
        hla_layout=hla_layout_payload,
        pra_positivity_threshold=req.pra_positivity_threshold,
    )

    return {"job_id": job_id}


@router.get("/api/process/{job_id}", response_model=ProgressResponse)
async def get_process(job_id: str):
    progress = get_image_progress(job_id)

    if progress is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result = get_image_result(job_id)

    return {
        **progress,
        "support_id": progress.get("support_id") or job_id,
        "result": result,
    }


@router.get("/api/process/{job_id}/segmented/{well_id}.png")
async def get_segmented_image(job_id: str, well_id: str):
    path = Path(settings.data_dir) / "segmented" / job_id / f"{well_id}.png"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Segmented image not found")

    return FileResponse(path, media_type="image/png")

@router.get("/api/process/{job_id}/summary.pdf")
async def get_cdc_summary_pdf(job_id: str):
    progress = get_image_progress(job_id)

    if progress is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if progress.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail="Summary PDF is only available after the analysis is done.",
        )

    result = get_image_result(job_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")

    pdf = build_cdc_summary_pdf(result=result, job_id=job_id)

    filename = f"cdc_summary_{job_id}.pdf"

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
