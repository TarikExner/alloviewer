import logging

from app.scripts.cleanup_jobs import cleanup_runtime_data
from app.workers.celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.tasks_cleanup.cleanup_runtime_data_task")
def cleanup_runtime_data_task() -> dict[str, int]:
    """
    Scheduled cleanup for old runtime files.

    Deletes:
      - generated artifacts older than 24h
      - uploaded source files older than 24h

    This should run on the maintenance queue.
    """
    logger.info("Starting scheduled runtime cleanup.")

    result = cleanup_runtime_data(
        artifacts_older_than_hours=24.0,
        uploads_older_than_hours=24.0,
        include_uploads=True,
        dry_run=False,
    )

    logger.info("Finished scheduled runtime cleanup: %s", result)

    return result
