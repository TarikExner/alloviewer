import logging
import os

from app.scripts.cleanup_jobs import (
    GIBIBYTE,
    cleanup_jobs_until_minimum_free_space,
    cleanup_runtime_data,
)
from app.workers.celery_app import celery_app


logger = logging.getLogger(__name__)


def _minimum_free_disk_bytes() -> int:
    raw_value = os.getenv("DISK_CLEANUP_MIN_FREE_GB", "2").strip()

    try:
        minimum_free_gb = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "DISK_CLEANUP_MIN_FREE_GB must be a number."
        ) from exc

    if minimum_free_gb < 0:
        raise ValueError(
            "DISK_CLEANUP_MIN_FREE_GB must be greater than or equal to zero."
        )

    return int(minimum_free_gb * GIBIBYTE)


@celery_app.task(name="app.workers.tasks_cleanup.cleanup_runtime_data_task")
def cleanup_runtime_data_task() -> dict[str, int]:
    """
    Scheduled cleanup for old runtime files.

    Deletes complete job directories that have been inactive for more than
    24 hours. This task runs on the maintenance queue.
    """
    logger.info("Starting scheduled runtime cleanup.")

    result = cleanup_runtime_data(
        jobs_older_than_hours=24.0,
        dry_run=False,
    )

    logger.info("Finished scheduled runtime cleanup: %s", result)
    return result


@celery_app.task(name="app.workers.tasks_cleanup.cleanup_low_disk_space_task")
def cleanup_low_disk_space_task() -> dict[str, int | bool]:
    """
    Keep at least the configured amount of free space on the filesystem that
    contains DATA_DIR.

    The default threshold is 2 GiB. When free space drops below the threshold,
    the oldest non-running jobs are removed until the threshold is met or no
    safe deletion candidates remain.
    """
    minimum_free_bytes = _minimum_free_disk_bytes()

    logger.info(
        "Starting low-disk-space cleanup with minimum free bytes=%s.",
        minimum_free_bytes,
    )

    result = cleanup_jobs_until_minimum_free_space(
        minimum_free_bytes=minimum_free_bytes,
        dry_run=False,
    )

    logger.info("Finished low-disk-space cleanup: %s", result)
    return result
