from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from alloviewer.flow_cytometry.pipeline import (
    run_fcxm_analysis,
)

from app.models import FCXMRunRequest
from app.services.fcxm_cache_storage import (
    save_fcxm_plot_cache,
)
from app.services.job_errors import (
    describe_job_error,
)
from app.services.job_paths import (
    JobPathError,
    get_job_paths,
    resolve_job_path,
)
from app.services.job_registry import (
    read_json,
    require_job_type,
)
from app.services.job_state import (
    get_fcxm_progress,
    set_fcxm_result,
    update_fcxm_progress,
)


logger = logging.getLogger(__name__)

ProgressCallback = Callable[
    [dict[str, Any]],
    None,
]


def _normalize_absolute_path(
    path: str | Path,
) -> str:
    return os.path.normcase(
        os.path.normpath(
            str(Path(path).resolve())
        )
    )


def _collect_original_filenames(
    request_dict: dict[str, Any],
) -> list[str]:
    filenames: list[str] = []

    for sample in request_dict.get(
        "samples",
        [],
    ):
        for filename in (
            sample.get("file_paths", [])
            or []
        ):
            if filename:
                filenames.append(
                    str(filename)
                )

    return filenames


def _resolve_request_file_paths(
    *,
    job_id: str,
    request_dict: dict[str, Any],
) -> dict[str, str]:
    """
    Resolve job-relative FCS paths to absolute paths for the analysis library.

    The request stored on disk remains unchanged. Only the in-memory copy is
    modified.
    """
    paths = get_job_paths(job_id)

    absolute_to_original: dict[
        str,
        str,
    ] = {}

    for sample in request_dict.get(
        "samples",
        [],
    ):
        absolute_paths: list[str] = []

        for raw_path in (
            sample.get("file_paths", [])
            or []
        ):
            original_path = str(
                raw_path
            )

            try:
                resolved = resolve_job_path(
                    job_id,
                    original_path,
                    required_root=paths.fcs_uploads,
                    must_exist=True,
                )
            except JobPathError as exc:
                raise ValueError(
                    "Invalid FCS file path: "
                    f"{original_path}"
                ) from exc
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "FCS file was not found: "
                    f"{Path(original_path).name}"
                ) from exc

            if not resolved.is_file():
                raise ValueError(
                    "FCS path is not a file: "
                    f"{original_path}"
                )

            if (
                resolved.suffix.lower()
                != ".fcs"
            ):
                raise ValueError(
                    "File is not an FCS file: "
                    f"{original_path}"
                )

            absolute_path = str(
                resolved
            )

            absolute_paths.append(
                absolute_path
            )

            absolute_to_original[
                _normalize_absolute_path(
                    absolute_path
                )
            ] = original_path

        sample["file_paths"] = (
            absolute_paths
        )

    return absolute_to_original


def _public_filename(
    value: Any,
    absolute_to_original: dict[
        str,
        str,
    ],
) -> str | None:
    if value is None:
        return None

    text = str(value)

    if (
        text
        in absolute_to_original.values()
    ):
        return text

    path = Path(text)

    if path.is_absolute():
        mapped = (
            absolute_to_original.get(
                _normalize_absolute_path(
                    path
                )
            )
        )

        if mapped:
            return mapped

        # Do not expose internal server paths.
        return path.name

    return text.replace(
        "\\",
        "/",
    )


def _make_progress_callback(
    *,
    job_id: str,
    absolute_to_original: dict[
        str,
        str,
    ],
) -> ProgressCallback:
    def progress(
        event: dict[str, Any],
    ) -> None:
        payload = dict(event)

        payload["current_file"] = (
            _public_filename(
                payload.get(
                    "current_file"
                ),
                absolute_to_original,
            )
        )

        done_filenames = payload.get(
            "done_filenames"
        )

        if isinstance(
            done_filenames,
            list,
        ):
            public_filenames: list[str] = []

            for value in done_filenames:
                public_name = (
                    _public_filename(
                        value,
                        absolute_to_original,
                    )
                )

                if (
                    public_name
                    and public_name
                    not in public_filenames
                ):
                    public_filenames.append(
                        public_name
                    )

            payload["done_filenames"] = (
                public_filenames
            )

        update_fcxm_progress(
            job_id,
            **payload,
        )

    return progress


def run_fcxm_job(
    *,
    job_id: str,
) -> None:
    """
    Execute an FCXM job from its persisted request.

    Only ``job_id`` crosses the Celery boundary. The request is loaded from
    ``DATA_DIR/jobs/<job_id>/request.json``.
    """
    logger.info(
        "Starting FCXM job %s.",
        job_id,
    )

    absolute_to_original: dict[
        str,
        str,
    ] = {}

    try:
        require_job_type(
            job_id,
            {"fcxm"},
        )

        paths = get_job_paths(
            job_id
        )

        if not paths.request.exists():
            raise FileNotFoundError(
                "The FCXM request file "
                "is missing."
            )

        raw_request = read_json(
            paths.request
        )

        if not isinstance(
            raw_request,
            dict,
        ):
            raise ValueError(
                "The FCXM request file "
                "does not contain a JSON object."
            )

        request = FCXMRunRequest(
            **raw_request
        )

        request_dict = (
            request.model_dump()
        )

        original_filenames = (
            _collect_original_filenames(
                request_dict
            )
        )

        if not original_filenames:
            raise ValueError(
                "The FCXM request contains "
                "no FCS files."
            )

        total_files = len(
            original_filenames
        )

        total_work = max(
            1,
            total_files * 5,
        )

        update_fcxm_progress(
            job_id,
            status="running",
            message=(
                "Starting flow cytometry "
                "analysis."
            ),
            stage="starting",
            total_files=total_work,
            done_files=0,
            current_file=None,
            done_filenames=[],
            error=None,
            error_type=None,
            failed_stage=None,
            failed_file=None,
            support_id=job_id,
        )

        absolute_to_original = (
            _resolve_request_file_paths(
                job_id=job_id,
                request_dict=request_dict,
            )
        )

        progress_callback = (
            _make_progress_callback(
                job_id=job_id,
                absolute_to_original=(
                    absolute_to_original
                ),
            )
        )

        result = run_fcxm_analysis(
            req_dict=request_dict,
            progress_cb=progress_callback,
        )

        if "payload" not in result:
            raise RuntimeError(
                "FCXM analysis did not "
                "return a result payload."
            )

        if "plot_cache" not in result:
            raise RuntimeError(
                "FCXM analysis did not "
                "return a plot cache."
            )

        set_fcxm_result(
            job_id,
            result["payload"],
        )

        save_fcxm_plot_cache(
            job_id=job_id,
            plot_cache=(
                result["plot_cache"]
            ),
        )

        previous = (
            get_fcxm_progress(
                job_id
            )
            or {}
        )

        final_total = int(
            previous.get(
                "total_files"
            )
            or total_work
        )

        update_fcxm_progress(
            job_id,
            status="done",
            message="Analysis done.",
            stage="done",
            total_files=final_total,
            done_files=final_total,
            current_file=None,
            done_filenames=(
                original_filenames
            ),
            error=None,
            error_type=None,
            failed_stage=None,
            failed_file=None,
            support_id=job_id,
        )

        logger.info(
            "Finished FCXM job %s.",
            job_id,
        )

    except Exception as exc:
        previous = (
            get_fcxm_progress(
                job_id
            )
            or {}
        )

        failed_stage = previous.get(
            "stage"
        )

        failed_file = _public_filename(
            previous.get(
                "current_file"
            ),
            absolute_to_original,
        )

        public_error = (
            describe_job_error(exc)
        )

        logger.exception(
            "FCXM job failed: "
            "job_id=%s stage=%s file=%s",
            job_id,
            failed_stage,
            failed_file,
        )

        update_fcxm_progress(
            job_id,
            **{
                **previous,
                "status": "error",
                "message": (
                    "Flow cytometry analysis "
                    "failed."
                ),
                "stage": "error",
                "failed_stage": (
                    failed_stage
                ),
                "failed_file": (
                    failed_file
                ),
                "error": (
                    public_error.message
                ),
                "error_type": (
                    public_error.error_type
                ),
                "support_id": job_id,
                "current_file": None,
            },
        )

        raise
