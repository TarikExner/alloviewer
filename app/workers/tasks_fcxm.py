from __future__ import annotations

from app.services.fcxm_jobs import run_fcxm_job
from app.workers.celery_app import celery_app


@celery_app.task(
    name="app.workers.tasks_fcxm.run_fcxm_job_task",
)
def run_fcxm_job_task(job_id: str) -> None:
    """
    Execute one previously prepared FCXM job.

    The request is loaded from the job directory by ``run_fcxm_job``.
    Only the job ID is passed through Celery.
    """
    run_fcxm_job(job_id=job_id)
