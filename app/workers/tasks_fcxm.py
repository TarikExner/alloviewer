from app.workers.celery_app import celery_app
from app.models import FCXMRunRequest
from app.services.fcxm_jobs import run_fcxm_job


@celery_app.task(name="app.workers.tasks_fcxm.run_fcxm_job_task")
def run_fcxm_job_task(job_id: str, req_dict: dict) -> None:
    req = FCXMRunRequest(**req_dict)
    run_fcxm_job(job_id=job_id, req=req)
