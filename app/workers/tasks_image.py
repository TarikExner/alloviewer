from app.workers.celery_app import celery_app
from app.services.image_jobs import run_image_job


@celery_app.task(name="app.workers.tasks_image.run_image_analysis_task")
def run_image_analysis_task(
    job_id: str,
    layout: dict,
    image_order: list[str],
    image_filenames: list[str],
    data_dir: str,
    template_filename: str | None,
    assay_type: str,
    hla_layout: dict | None,
    pra_positivity_threshold: float,
) -> None:
    run_image_job(
        job_id=job_id,
        layout=layout,
        image_order=image_order,
        image_filenames=image_filenames,
        data_dir=data_dir,
        template_filename=template_filename,
        assay_type=assay_type,
        hla_layout=hla_layout,
        pra_positivity_threshold=pra_positivity_threshold,
    )
