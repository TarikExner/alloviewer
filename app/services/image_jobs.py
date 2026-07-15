from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from alloviewer.image_analysis.pipeline import (
    run_image_analysis,
)
from alloviewer.image_analysis.structs import (
    PlateLayout,
)

from app.services.job_errors import (
    describe_job_error,
)
from app.services.job_paths import (
    get_job_paths,
)
from app.services.job_registry import (
    read_json,
    require_job_type,
)
from app.services.job_state import (
    get_image_progress,
    set_image_result,
    update_image_progress,
)


logger = logging.getLogger(__name__)

ProgressCallback = Callable[
    [dict[str, Any]],
    None,
]


class AttrDict(dict):
    """
    Dictionary supporting both item access and attribute access.

    Parsed PRA layouts are currently consumed in both forms:

        layout["wells"]
        layout.wells

    The persisted JSON is converted recursively before being passed to the
    analysis library.
    """

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                name
            ) from exc

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> None:
        self[name] = value

    def __delattr__(
        self,
        name: str,
    ) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(
                name
            ) from exc


def to_attrdict(
    value: Any,
) -> Any:
    """
    Recursively convert JSON dictionaries to ``AttrDict`` instances.
    """
    if isinstance(value, dict):
        return AttrDict(
            {
                key: to_attrdict(
                    inner_value
                )
                for key, inner_value
                in value.items()
            }
        )

    if isinstance(value, list):
        return [
            to_attrdict(item)
            for item in value
        ]

    return value


def _load_request(
    job_id: str,
) -> dict[str, Any]:
    paths = get_job_paths(
        job_id
    )

    if not paths.request.exists():
        raise FileNotFoundError(
            "The image-analysis request file is missing."
        )

    request = read_json(
        paths.request
    )

    if not isinstance(
        request,
        dict,
    ):
        raise ValueError(
            "The image-analysis request file does not contain a JSON object."
        )

    return request


def _load_plate_layout(
    job_id: str,
    assay_type: str,
) -> Any | None:
    """
    Load the parsed PRA plate layout.

    Crossmatch jobs do not use an HLA plate layout.
    """
    if assay_type != "pra":
        return None

    paths = get_job_paths(
        job_id
    )

    if not paths.plate_layout.exists():
        raise FileNotFoundError(
            "The parsed PRA plate-layout file is missing."
        )

    parsed_layout = read_json(
        paths.plate_layout
    )

    if not isinstance(
        parsed_layout,
        dict,
    ):
        raise ValueError(
            "The parsed PRA plate layout does not contain a JSON object."
        )

    return to_attrdict(
        parsed_layout
    )


def _make_progress_callback(
    job_id: str,
) -> ProgressCallback:
    """
    Persist progress events emitted by the analysis library.
    """

    def progress(
        event: dict[str, Any],
    ) -> None:
        payload = dict(event)

        if not payload.get(
            "status"
        ):
            payload["status"] = (
                "running"
            )

        update_image_progress(
            job_id,
            **payload,
        )

    return progress


def _validate_request(
    request: dict[str, Any],
) -> tuple[
    PlateLayout,
    list[str],
    list[str],
    float,
]:
    layout_data = request.get(
        "layout"
    )

    if not isinstance(
        layout_data,
        dict,
    ):
        raise ValueError(
            "The image-analysis request is missing its plate layout."
        )

    layout = PlateLayout(
        **layout_data
    )

    image_order_raw = request.get(
        "image_order"
    )

    image_filenames_raw = request.get(
        "image_filenames"
    )

    if not isinstance(
        image_order_raw,
        list,
    ):
        raise ValueError(
            "The image-analysis request contains an invalid image order."
        )

    if not isinstance(
        image_filenames_raw,
        list,
    ):
        raise ValueError(
            "The image-analysis request contains an invalid image-file list."
        )

    image_order = [
        str(value)
        for value in image_order_raw
    ]

    image_filenames = [
        str(value)
        for value in image_filenames_raw
    ]

    if not image_order:
        raise ValueError(
            "The image-analysis request contains no wells."
        )

    if not image_filenames:
        raise ValueError(
            "The image-analysis request contains no image files."
        )

    if (
        len(image_order)
        != len(image_filenames)
    ):
        raise ValueError(
            "The image order and image-file list have different lengths."
        )

    positivity_threshold = float(
        request.get(
            "pra_positivity_threshold",
            20.0,
        )
    )

    return (
        layout,
        image_order,
        image_filenames,
        positivity_threshold,
    )


def run_image_job(
    *,
    job_id: str,
) -> None:
    """
    Execute a persisted image-analysis job.

    Only ``job_id`` crosses the Celery boundary. The request, uploaded images,
    parsed PRA layout, results, and generated outputs are all read from or
    written to the corresponding job directory.
    """
    logger.info(
        "Starting image-analysis job %s.",
        job_id,
    )

    try:
        job = require_job_type(
            job_id,
            {
                "pra",
                "crossmatch",
            },
        )

        assay_type = str(
            job.get("job_type")
        )

        if assay_type not in {
            "pra",
            "crossmatch",
        }:
            raise ValueError(
                "Unsupported image-analysis job type: "
                f"{assay_type}"
            )

        paths = get_job_paths(
            job_id,
            create=True,
        )

        request = _load_request(
            job_id
        )

        (
            layout,
            image_order,
            image_filenames,
            positivity_threshold,
        ) = _validate_request(
            request
        )

        hla_layout = _load_plate_layout(
            job_id,
            assay_type,
        )

        total = len(
            image_order
        )

        update_image_progress(
            job_id,
            status="running",
            stage="starting",
            done=0,
            total=total,
            current_well=None,
            done_wells=[],
            error=None,
            error_type=None,
            failed_stage=None,
            failed_well=None,
            support_id=job_id,
        )

        progress_callback = (
            _make_progress_callback(
                job_id
            )
        )

        result = run_image_analysis(
            layout=layout,
            image_order=image_order,
            image_filenames=image_filenames,

            # Filenames are job-relative paths such as:
            # uploads/images/folder/image.tif
            input_root=paths.root,

            segmented_output_dir=(
                paths.segmented
            ),
            segmented_url_prefix=(
                f"/api/jobs/{job_id}"
                "/image/segmented"
            ),

            progress_cb=progress_callback,
            assay_type=assay_type,
            hla_layout=hla_layout,
            pra_positivity_threshold=(
                positivity_threshold
            ),
        )

        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "Image analysis did not return a result dictionary."
            )

        set_image_result(
            job_id,
            result,
        )

        update_image_progress(
            job_id,
            status="done",
            stage="done",
            done=total,
            total=total,
            current_well=None,
            done_wells=image_order,
            error=None,
            error_type=None,
            failed_stage=None,
            failed_well=None,
            support_id=job_id,
        )

        logger.info(
            "Finished image-analysis job %s.",
            job_id,
        )

    except Exception as exc:
        previous = (
            get_image_progress(
                job_id
            )
            or {}
        )

        failed_stage = (
            previous.get("stage")
        )

        failed_well = (
            previous.get(
                "current_well"
            )
        )

        public_error = (
            describe_job_error(exc)
        )

        logger.exception(
            "Image-analysis job failed: "
            "job_id=%s stage=%s well=%s",
            job_id,
            failed_stage,
            failed_well,
        )

        try:
            update_image_progress(
                job_id,
                **{
                    **previous,
                    "status": "error",
                    "stage": (
                        failed_stage
                        or "unknown"
                    ),
                    "failed_stage": (
                        failed_stage
                    ),
                    "failed_well": (
                        failed_well
                    ),
                    "error": (
                        public_error.message
                    ),
                    "error_type": (
                        public_error.error_type
                    ),
                    "support_id": job_id,
                    "current_well": None,
                },
            )
        except Exception:
            logger.exception(
                "Could not persist the image-job failure state: %s",
                job_id,
            )

        raise
