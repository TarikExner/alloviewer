from __future__ import annotations

import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Iterator

from app.services.job_paths import get_job_paths


class JobIdFilter(logging.Filter):
    """
    Attach a job identifier to each log record passed through the filter.
    """
    def __init__(self, job_id: str):
        """
        Store the job identifier used to label log records.

        Parameters
        ----------
        job_id : str
            UUID string identifying the job within the configured data directory.
        """
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Set the record's job_id and allow the record through.

        Parameters
        ----------
        record : logging.LogRecord
            Log record updated in place with the job identifier.

        Returns
        -------
        bool
            The constant ``True``.
        """
        record.job_id = self.job_id
        return True


@contextmanager
def job_log_context(
    *,
    job_id: str,
    job_type: str,
) -> Iterator[None]:
    """
    Attach a rotating job log handler for the duration of a context.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    job_type : str
        Assay type: ``pra``, ``crossmatch``, or ``fcxm``.

    Yields
    ------
    None
        Control to the caller while the job log handler is active.

    Notes
    -----
    Adds a handler to the root logger, with a 5 MiB limit and two backup files. The handler is removed and closed in a finally block.
    """
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
