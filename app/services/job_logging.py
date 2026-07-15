from __future__ import annotations

import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Iterator

from app.services.job_paths import get_job_paths


class JobIdFilter(logging.Filter):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = self.job_id
        return True


@contextmanager
def job_log_context(
    *,
    job_id: str,
    job_type: str,
) -> Iterator[None]:
    paths = get_job_paths(job_id, create=True)

    handler = RotatingFileHandler(
        paths.log,
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )

    handler.setLevel(logging.INFO)
    handler.addFilter(JobIdFilter(job_id))

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s "
            "[job=%(job_id)s] %(name)s - %(message)s"
        )
    )

    root = logging.getLogger()
    root.addHandler(handler)

    logger = logging.getLogger(__name__)

    try:
        logger.info(
            "Job logging started: type=%s",
            job_type,
        )
        yield
    finally:
        logger.info("Job logging stopped.")
        root.removeHandler(handler)
        handler.close()
