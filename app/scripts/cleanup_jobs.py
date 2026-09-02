from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from app.core.settings import settings
from app.services.job_paths import normalize_job_id
from app.services.job_registry import read_json
from app.services.job_state import delete_all_job_state


logger = logging.getLogger(__name__)


GIBIBYTE = 1024 ** 3
ACTIVE_JOB_STATUSES = {"queued", "running"}


def _format_gibibytes(value: int) -> str:
    return f"{value / GIBIBYTE:.2f} GiB"


def _read_job_metadata(job_dir: Path) -> dict[str, Any] | None:
    """
    Read ``job.json`` from a job directory.

    Invalid or missing metadata is reported and handled as an orphaned job
    directory by the cleanup routine.
    """
    metadata_path = job_dir / "job.json"

    if not metadata_path.exists():
        logger.warning(
            "Job directory has no job.json: %s",
            job_dir,
        )
        return None

    if not metadata_path.is_file():
        logger.warning(
            "Job metadata path is not a file: %s",
            metadata_path,
        )
        return None

    try:
        metadata = read_json(metadata_path)
    except Exception:
        logger.exception(
            "Could not read job metadata: %s",
            metadata_path,
        )
        return None

    if not isinstance(metadata, dict):
        logger.warning(
            "Job metadata is not a JSON object: %s",
            metadata_path,
        )
        return None

    return metadata


def _job_last_updated(
    job_dir: Path,
    metadata: dict[str, Any] | None,
) -> float | None:
    """
    Determine the last relevant activity timestamp.

    ``job.json.updated_at`` is preferred. Directory modification time is used
    only for orphaned or malformed job directories.
    """
    if metadata is not None:
        updated_at = metadata.get("updated_at")

        if isinstance(updated_at, (int, float)):
            return float(updated_at)

        if isinstance(updated_at, str):
            try:
                return float(updated_at)
            except ValueError:
                logger.warning(
                    "Invalid updated_at value in %s: %r",
                    job_dir / "job.json",
                    updated_at,
                )

    try:
        return job_dir.stat().st_mtime
    except FileNotFoundError:
        return None
    except OSError:
        logger.exception(
            "Could not read job directory timestamp: %s",
            job_dir,
        )
        return None


def _remove_job_directory(
    job_dir: Path,
    *,
    job_id: str,
    dry_run: bool,
) -> bool:
    """
    Delete Redis state and the complete job directory.
    """
    if not job_dir.exists():
        return False

    if not job_dir.is_dir():
        logger.warning(
            "Skipping non-directory item in jobs root: %s",
            job_dir,
        )
        return False

    if dry_run:
        logger.info(
            "[dry-run] Would delete job %s: %s",
            job_id,
            job_dir,
        )
        return True

    try:
        delete_all_job_state(job_id)
    except Exception:
        # Redis cleanup failure should not prevent removal of stale files.
        logger.exception(
            "Could not delete Redis state for job %s.",
            job_id,
        )

    try:
        logger.info(
            "Deleting stale job %s: %s",
            job_id,
            job_dir,
        )
        shutil.rmtree(job_dir)
    except FileNotFoundError:
        return False
    except Exception:
        logger.exception(
            "Could not delete job directory: %s",
            job_dir,
        )
        return False

    return True


def cleanup_runtime_data(
    *,
    jobs_older_than_hours: float,
    dry_run: bool,
) -> dict[str, int]:
    """
    Delete complete job directories that have been inactive beyond the
    configured threshold.

    All job-owned uploads, intermediate files, logs, results, thumbnails,
    segmented images, caches, and reports are removed together.

    Parameters
    ----------
    jobs_older_than_hours
        Delete jobs whose ``updated_at`` value is older than this threshold.
    dry_run
        Report matching jobs without deleting files or Redis state.

    Returns
    -------
    dict
        Cleanup counters.
    """
    if jobs_older_than_hours < 0:
        raise ValueError(
            "jobs_older_than_hours must be greater than or equal to zero."
        )

    data_dir = Path(settings.data_dir).resolve()
    jobs_root = data_dir / "jobs"

    cutoff_timestamp = (
        time.time()
        - jobs_older_than_hours * 60 * 60
    )

    result = {
        "job_dirs_checked": 0,
        "job_dirs_deleted": 0,
        "job_dirs_retained": 0,
        "orphan_job_dirs_deleted": 0,
        "invalid_items_skipped": 0,
        "deletion_errors": 0,
    }

    logger.info(
        "Cleaning job data under: %s",
        jobs_root,
    )
    logger.info(
        "Job inactivity threshold, hours: %s",
        jobs_older_than_hours,
    )
    logger.info(
        "Cutoff timestamp: %s",
        cutoff_timestamp,
    )
    logger.info(
        "Dry run: %s",
        dry_run,
    )

    if not jobs_root.exists():
        logger.info(
            "Jobs directory does not exist; nothing to clean."
        )
        return result

    if not jobs_root.is_dir():
        logger.error(
            "Jobs path is not a directory: %s",
            jobs_root,
        )
        result["invalid_items_skipped"] += 1
        return result

    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            logger.warning(
                "Skipping non-directory item in jobs root: %s",
                job_dir,
            )
            result["invalid_items_skipped"] += 1
            continue

        result["job_dirs_checked"] += 1

        try:
            job_id = normalize_job_id(job_dir.name)
        except ValueError:
            logger.warning(
                "Skipping directory with invalid job ID: %s",
                job_dir,
            )
            result["invalid_items_skipped"] += 1
            continue

        metadata = _read_job_metadata(job_dir)

        status = (
            str(metadata.get("status") or "").strip().lower()
            if metadata is not None
            else ""
        )

        if status in ACTIVE_JOB_STATUSES:
            logger.info(
                "Retaining active job %s because its status is %s.",
                job_id,
                status,
            )
            result["job_dirs_retained"] += 1
            continue

        last_updated = _job_last_updated(
            job_dir,
            metadata,
        )

        if last_updated is None:
            logger.warning(
                "Could not determine job age; retaining: %s",
                job_dir,
            )
            result["job_dirs_retained"] += 1
            continue

        if last_updated >= cutoff_timestamp:
            result["job_dirs_retained"] += 1
            continue

        is_orphan = metadata is None

        # Re-read job state immediately before deletion. The job may have
        # become queued or running since the first metadata check.
        current_metadata = _read_job_metadata(job_dir)
        current_status = (
            str(current_metadata.get("status") or "").strip().lower()
            if current_metadata is not None
            else ""
        )

        if current_status in ACTIVE_JOB_STATUSES:
            logger.info(
                "Retaining active job %s because its current status is %s.",
                job_id,
                current_status,
            )
            result["job_dirs_retained"] += 1
            continue

        deleted = _remove_job_directory(
            job_dir,
            job_id=job_id,
            dry_run=dry_run,
        )

        if deleted:
            result["job_dirs_deleted"] += 1

            if is_orphan:
                result["orphan_job_dirs_deleted"] += 1
        else:
            result["deletion_errors"] += 1

    logger.info(
        "Finished runtime cleanup: %s",
        result,
    )

    return result



def cleanup_jobs_until_minimum_free_space(
    *,
    minimum_free_bytes: int,
    dry_run: bool,
) -> dict[str, int | bool]:
    """
    Remove the oldest non-running job directories until the filesystem that
    contains ``DATA_DIR`` has at least ``minimum_free_bytes`` available.

    Jobs with status ``queued`` or ``running`` are never removed. Completed,
    failed, draft, orphaned, and malformed jobs are considered in oldest-first
    order. Job age is based on ``job.json.updated_at`` with directory mtime as
    a fallback.

    Parameters
    ----------
    minimum_free_bytes
        Required free space on the filesystem that contains ``DATA_DIR``.
    dry_run
        Report the deletion order without removing files or Redis state.

    Returns
    -------
    dict
        Disk-space and cleanup counters. All byte values are integers so the
        result can be serialized by Celery without conversion.
    """
    if minimum_free_bytes < 0:
        raise ValueError(
            "minimum_free_bytes must be greater than or equal to zero."
        )

    data_dir = Path(settings.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs_root = data_dir / "jobs"

    usage_before = shutil.disk_usage(data_dir)
    free_before = int(usage_before.free)

    result: dict[str, int | bool] = {
        "minimum_free_bytes": int(minimum_free_bytes),
        "free_bytes_before": free_before,
        "free_bytes_after": free_before,
        "estimated_free_bytes_after": free_before,
        "job_dirs_checked": 0,
        "candidate_job_dirs": 0,
        "active_job_dirs_skipped": 0,
        "invalid_items_skipped": 0,
        "job_dirs_deleted": 0,
        "deletion_errors": 0,
        "bytes_reclaimed_estimate": 0,
        "target_met": free_before >= minimum_free_bytes,
        "dry_run": dry_run,
    }

    logger.info(
        "Disk-space cleanup check for %s: free=%s, required=%s, dry_run=%s",
        data_dir,
        _format_gibibytes(free_before),
        _format_gibibytes(minimum_free_bytes),
        dry_run,
    )

    if free_before >= minimum_free_bytes:
        logger.info("Disk-space cleanup is not required.")
        return result

    if not jobs_root.exists():
        logger.warning(
            "Free space is below the threshold, but the jobs directory does "
            "not exist: %s",
            jobs_root,
        )
        return result

    if not jobs_root.is_dir():
        logger.error("Jobs path is not a directory: %s", jobs_root)
        result["invalid_items_skipped"] += 1
        return result

    candidates: list[
        tuple[float, Path, str, dict[str, Any] | None]
    ] = []

    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            logger.warning(
                "Skipping non-directory item in jobs root: %s",
                job_dir,
            )
            result["invalid_items_skipped"] += 1
            continue

        result["job_dirs_checked"] += 1

        try:
            job_id = normalize_job_id(job_dir.name)
        except ValueError:
            logger.warning(
                "Skipping directory with invalid job ID: %s",
                job_dir,
            )
            result["invalid_items_skipped"] += 1
            continue

        metadata = _read_job_metadata(job_dir)
        status = (
            str(metadata.get("status") or "").strip().lower()
            if metadata is not None
            else ""
        )

        if status in ACTIVE_JOB_STATUSES:
            result["active_job_dirs_skipped"] += 1
            continue

        last_updated = _job_last_updated(job_dir, metadata)

        if last_updated is None:
            logger.warning(
                "Could not determine job age; retaining: %s",
                job_dir,
            )
            result["invalid_items_skipped"] += 1
            continue

        candidates.append(
            (last_updated, job_dir, job_id, metadata)
        )

    candidates.sort(key=lambda item: (item[0], item[1].name))
    result["candidate_job_dirs"] = len(candidates)

    estimated_free = free_before

    for _, job_dir, job_id, _ in candidates:
        if dry_run:
            try:
                job_size = sum(
                    item.stat().st_size
                    for item in job_dir.rglob("*")
                    if item.is_file()
                )
            except OSError:
                logger.exception(
                    "Could not estimate job directory size: %s",
                    job_dir,
                )
                job_size = 0

            logger.info(
                "[dry-run] Would delete oldest job %s: %s",
                job_id,
                job_dir,
            )
            result["job_dirs_deleted"] += 1
            result["bytes_reclaimed_estimate"] += int(job_size)
            estimated_free += int(job_size)
            result["estimated_free_bytes_after"] = estimated_free

            if estimated_free >= minimum_free_bytes:
                result["target_met"] = True
                break

            continue

        # Re-read metadata immediately before deletion. This prevents a job
        # that became queued or running after candidate collection from being
        # removed.
        current_metadata = _read_job_metadata(job_dir)
        current_status = (
            str(current_metadata.get("status") or "").strip().lower()
            if current_metadata is not None
            else ""
        )

        if current_status in ACTIVE_JOB_STATUSES:
            logger.info(
                "Retaining job %s because its current status is %s.",
                job_id,
                current_status,
            )
            result["active_job_dirs_skipped"] += 1
            continue

        deleted = _remove_job_directory(
            job_dir,
            job_id=job_id,
            dry_run=False,
        )

        if deleted:
            result["job_dirs_deleted"] += 1
        else:
            result["deletion_errors"] += 1

        free_now = int(shutil.disk_usage(data_dir).free)
        result["free_bytes_after"] = free_now
        result["estimated_free_bytes_after"] = free_now

        logger.info(
            "Free space after cleanup step: %s",
            _format_gibibytes(free_now),
        )

        if free_now >= minimum_free_bytes:
            result["target_met"] = True
            break

    if not dry_run:
        final_free = int(shutil.disk_usage(data_dir).free)
        result["free_bytes_after"] = final_free
        result["estimated_free_bytes_after"] = final_free
        result["target_met"] = final_free >= minimum_free_bytes

    if not result["target_met"]:
        logger.warning(
            "Free space is still below the required threshold after all safe "
            "job deletion candidates were processed: free=%s, required=%s",
            _format_gibibytes(int(result["free_bytes_after"])),
            _format_gibibytes(minimum_free_bytes),
        )

    logger.info("Finished disk-space cleanup: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete stale AlloViewer job directories and their Redis state."
        )
    )

    parser.add_argument(
        "--jobs-older-than-hours",
        type=float,
        default=24.0,
        help=(
            "Delete jobs that have not been updated for this many hours. "
            "Default: 24."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print matching jobs without deleting files or Redis state."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        ),
    )

    result = cleanup_runtime_data(
        jobs_older_than_hours=args.jobs_older_than_hours,
        dry_run=args.dry_run,
    )

    logger.info(
        "Cleanup result: %s",
        result,
    )


if __name__ == "__main__":
    main()
