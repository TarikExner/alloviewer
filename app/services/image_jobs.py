import logging
from typing import Any

from alloviewer.image_analysis.pipeline import run_image_analysis
from alloviewer.image_analysis.structs import PlateLayout

from app.services.job_errors import describe_job_error
from app.services.job_state import (
    get_image_progress,
    update_image_progress,
)

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """
    Dict that also supports attribute access.

    Needed because parsed HLA layouts are used in both styles:
        hla_layout.wells.items()
        well_layout.loci.data

    After Celery JSON serialization, everything arrives as normal dicts.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def to_attrdict(value: Any) -> Any:
    """
    Recursively convert JSON-decoded dicts into objects that behave like both
    dicts and attribute-access objects.
    """
    if isinstance(value, dict):
        return AttrDict(
            {
                key: to_attrdict(inner_value)
                for key, inner_value in value.items()
            }
        )

    if isinstance(value, list):
        return [to_attrdict(item) for item in value]

    return value


def run_image_job(
    *,
    job_id: str,
    layout: dict,
    image_order: list[str],
    image_filenames: list[str],
    data_dir: str,
    template_filename: str | None,
    assay_type: str,
    hla_layout: dict | None,
    pra_positivity_threshold: float,
) -> None:
    """
    Rebuild Python-side objects from Celery-safe JSON payloads and execute the
    image analysis pipeline.
    """
    logger.info("Starting image job %s", job_id)

    try:
        layout_obj = PlateLayout(**layout)

        hla_layout_obj = None
        if hla_layout is not None:
            hla_layout_obj = to_attrdict(hla_layout)

        run_image_analysis(
            job_id=job_id,
            layout=layout_obj,
            image_order=image_order,
            image_filenames=image_filenames,
            data_dir=data_dir,
            template_filename=template_filename,
            assay_type=assay_type,
            hla_layout=hla_layout_obj,
            pra_positivity_threshold=pra_positivity_threshold,
        )

        logger.info("Finished image job %s", job_id)

    except Exception as exc:
        previous = get_image_progress(job_id) or {}

        failed_stage = previous.get("stage")
        failed_well = previous.get("current_well")

        public_error = describe_job_error(exc)

        logger.exception(
            "Image job failed: job_id=%s stage=%s well=%s",
            job_id,
            failed_stage,
            failed_well,
        )

        update_image_progress(
            job_id,
            status="error",

            # Keep the real stage instead of replacing it with "error".
            stage=failed_stage or "unknown",
            failed_stage=failed_stage,
            failed_well=failed_well,

            error=public_error.message,
            error_type=public_error.error_type,
            support_id=job_id,

            current_well=None,
        )

        raise
