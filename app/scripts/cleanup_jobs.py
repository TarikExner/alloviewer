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
