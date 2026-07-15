from fastapi import APIRouter, HTTPException, Response

from app.models import CreateJobRequest, JobResponse
from app.services.job_registry import (
    create_job,
    require_job,
    delete_job_directory,
)
from app.services.job_state import delete_all_job_state


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse)
async def create_job_route(req: CreateJobRequest):
    return create_job(req.job_type)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_route(job_id: str):
    try:
        return require_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@router.delete("/{job_id}", status_code=204)
async def delete_job_route(job_id: str):
    try:
        job = require_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail="A queued or running job cannot be deleted.",
        )

    delete_all_job_state(job_id)
    delete_job_directory(job_id)

    return Response(status_code=204)
