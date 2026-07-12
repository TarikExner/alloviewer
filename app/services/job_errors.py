from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException


@dataclass(frozen=True)
class PublicJobError:
    message: str
    error_type: str


def describe_job_error(exc: Exception) -> PublicJobError:
    """
    Convert an internal exception into information that is safe and useful
    for the frontend.

    Full exception details and tracebacks should remain in the server logs.
    """
    error_type = type(exc).__name__

    if isinstance(exc, HTTPException):
        detail = exc.detail

        if isinstance(detail, str) and detail.strip():
            return PublicJobError(
                message=detail.strip(),
                error_type=error_type,
            )

        return PublicJobError(
            message=f"The server rejected the request with status {exc.status_code}.",
            error_type=error_type,
        )

    if isinstance(exc, FileNotFoundError):
        if exc.filename:
            filename = Path(exc.filename).name
            message = f"Required file was not found: {filename}"
        else:
            message = "A required file could not be found."

        return PublicJobError(message=message, error_type=error_type)

    if isinstance(exc, PermissionError):
        return PublicJobError(
            message="The server could not access a required file.",
            error_type=error_type,
        )

    if isinstance(exc, MemoryError):
        return PublicJobError(
            message=(
                "The analysis ran out of available memory. "
                "Try fewer or smaller files."
            ),
            error_type=error_type,
        )

    if isinstance(exc, ValueError):
        message = str(exc).strip() or "The analysis received invalid input."
        return PublicJobError(message=message, error_type=error_type)

    if isinstance(exc, RuntimeError):
        message = str(exc).strip() or "The analysis could not be completed."
        return PublicJobError(message=message, error_type=error_type)

    return PublicJobError(
        message=(
            "An unexpected internal error occurred. Try again. "
            "If the error persists, please report the job ID to the AlloViewer team."
        ),
        error_type=error_type,
    )
