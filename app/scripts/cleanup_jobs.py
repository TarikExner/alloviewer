import argparse
import logging
import shutil
import time
from pathlib import Path

from app.core.settings import settings


logger = logging.getLogger(__name__)


UPLOAD_FILE_EXTENSIONS = {
    ".fcs",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".xlsx",
    ".xls",
    ".csv",
}

ARTIFACT_DIR_NAMES = {
    "segmented",
    "jobs",
    "_thumbs",
}

EXCLUDED_RECURSIVE_DIR_NAMES = {
    "segmented",
    "jobs",
    "_thumbs",
}


def _is_older_than(path: Path, cutoff_timestamp: float) -> bool:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False

    return mtime < cutoff_timestamp


def _remove_dir(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False

    if not path.is_dir():
        logger.warning("Skipping non-directory path: %s", path)
        return False

    if dry_run:
        logger.info("[dry-run] Would delete directory: %s", path)
        return True

    logger.info("Deleting directory: %s", path)
    shutil.rmtree(path)
    return True


def _remove_file(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False

    if not path.is_file():
        logger.warning("Skipping non-file path: %s", path)
        return False

    if dry_run:
        logger.info("[dry-run] Would delete file: %s", path)
        return True

    logger.info("Deleting file: %s", path)
    path.unlink()
    return True


def _is_under_excluded_recursive_dir(path: Path, data_dir: Path) -> bool:
    try:
        relative_parts = path.relative_to(data_dir).parts
    except ValueError:
        return True

    return any(part in EXCLUDED_RECURSIVE_DIR_NAMES for part in relative_parts)


def cleanup_artifact_dirs(
    *,
    data_dir: Path,
    cutoff_timestamp: float,
    dry_run: bool,
) -> dict[str, int]:
    result = {
        "segmented_dirs_deleted": 0,
        "job_dirs_deleted": 0,
        "thumbnail_dirs_deleted": 0,
    }

    segmented_root = data_dir / "segmented"
    if segmented_root.exists():
        for job_dir in segmented_root.iterdir():
            if not job_dir.is_dir():
                continue

            if not _is_older_than(job_dir, cutoff_timestamp):
                continue

            if _remove_dir(job_dir, dry_run=dry_run):
                result["segmented_dirs_deleted"] += 1

    jobs_root = data_dir / "jobs"
    if jobs_root.exists():
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue

            if not _is_older_than(job_dir, cutoff_timestamp):
                continue

            if _remove_dir(job_dir, dry_run=dry_run):
                result["job_dirs_deleted"] += 1

    thumbs_root = data_dir / "_thumbs"
    if thumbs_root.exists():
        for thumb_item in thumbs_root.iterdir():
            if not _is_older_than(thumb_item, cutoff_timestamp):
                continue

            if thumb_item.is_dir():
                if _remove_dir(thumb_item, dry_run=dry_run):
                    result["thumbnail_dirs_deleted"] += 1
            elif thumb_item.is_file():
                if _remove_file(thumb_item, dry_run=dry_run):
                    result["thumbnail_dirs_deleted"] += 1

    return result


def cleanup_uploaded_files(
    *,
    data_dir: Path,
    cutoff_timestamp: float,
    dry_run: bool,
) -> int:
    deleted = 0

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        if _is_under_excluded_recursive_dir(path, data_dir):
            continue

        if path.suffix.lower() not in UPLOAD_FILE_EXTENSIONS:
            continue

        if not _is_older_than(path, cutoff_timestamp):
            continue

        if _remove_file(path, dry_run=dry_run):
            deleted += 1

    return deleted


def cleanup_empty_upload_dirs(
    *,
    data_dir: Path,
    dry_run: bool,
) -> int:
    deleted = 0

    dirs = sorted(
        [p for p in data_dir.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for path in dirs:
        if path == data_dir:
            continue

        if path.name in ARTIFACT_DIR_NAMES:
            continue

        if _is_under_excluded_recursive_dir(path, data_dir):
            continue

        try:
            next(path.iterdir())
            continue
        except StopIteration:
            pass

        if dry_run:
            logger.info("[dry-run] Would delete empty upload directory: %s", path)
            deleted += 1
            continue

        logger.info("Deleting empty upload directory: %s", path)
        path.rmdir()
        deleted += 1

    return deleted


def cleanup_runtime_data(
    *,
    artifacts_older_than_hours: float,
    uploads_older_than_hours: float,
    include_uploads: bool,
    dry_run: bool,
) -> dict[str, int]:
    data_dir = Path(settings.data_dir).resolve()

    artifact_cutoff = time.time() - artifacts_older_than_hours * 60 * 60
    upload_cutoff = time.time() - uploads_older_than_hours * 60 * 60

    logger.info("Cleaning runtime data under: %s", data_dir)
    logger.info("Artifact threshold, hours: %s", artifacts_older_than_hours)
    logger.info("Upload threshold, hours: %s", uploads_older_than_hours)
    logger.info("Include uploads: %s", include_uploads)
    logger.info("Dry run: %s", dry_run)

    artifact_result = cleanup_artifact_dirs(
        data_dir=data_dir,
        cutoff_timestamp=artifact_cutoff,
        dry_run=dry_run,
    )

    uploaded_files_deleted = 0
    empty_upload_dirs_deleted = 0

    if include_uploads:
        uploaded_files_deleted = cleanup_uploaded_files(
            data_dir=data_dir,
            cutoff_timestamp=upload_cutoff,
            dry_run=dry_run,
        )

        empty_upload_dirs_deleted = cleanup_empty_upload_dirs(
            data_dir=data_dir,
            dry_run=dry_run,
        )

    return {
        **artifact_result,
        "uploaded_files_deleted": uploaded_files_deleted,
        "empty_upload_dirs_deleted": empty_upload_dirs_deleted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean old runtime data from settings.data_dir."
    )

    parser.add_argument(
        "--artifacts-older-than-hours",
        type=float,
        default=24.0,
        help="Delete generated artifact directories older than this many hours.",
    )

    parser.add_argument(
        "--uploads-older-than-hours",
        type=float,
        default=24.0,
        help="Delete uploaded source files older than this many hours.",
    )

    parser.add_argument(
        "--include-uploads",
        action="store_true",
        help="Also delete uploaded source files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without deleting anything.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    result = cleanup_runtime_data(
        artifacts_older_than_hours=args.artifacts_older_than_hours,
        uploads_older_than_hours=args.uploads_older_than_hours,
        include_uploads=args.include_uploads,
        dry_run=args.dry_run,
    )

    logger.info("Cleanup result: %s", result)


if __name__ == "__main__":
    main()
