from __future__ import annotations

from app.services.image_jobs import run_image_job
from app.workers.celery_app import celery_app


@celery_app.task(
    name="app.workers.tasks_image.run_image_analysis_task",
)
def run_image_analysis_task(job_id: str) -> None:
    """
    Execute one persisted image-analysis job.

    Only the job ID crosses the Celery boundary. The request and all inputs
    are loaded from the corresponding job directory.
    """
    run_image_job(job_id=job_id)
