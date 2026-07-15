from __future__ import annotations

import copy
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from . import load_images
from .calibrators import PCNCGaussian2DCalibrator
from .classifiers import ROIClassifierGaussian2D3Way
from .config import (
    CDC_SUMMARY_CONFIG,
    INSTANCE_CONFIG,
    UNET_CONFIG,
)
from .extractor import RGBExtractor
from .qc import QCMonitor
from .segmenter import SegmenterUNetInference
from .services.analysis import (
    calculate_allele_reactivity_evidence,
    calculate_pra_reactivity_score,
)
from .structs import (
    ParsedPlateLayout,
    PlateLayout,
    ROIResult,
    WellResult,
)
from .utils import (
    automated_well_call,
    build_cdc_summary,
    create_plate,
    frac_pos_raw,
    save_segmented_preview,
    to_jsonable,
)


logger = logging.getLogger(__name__)

ProgressEvent = dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]


ALLOWED_CROSSMATCH_CELL_MODES = {"T", "B", "T/B", "empty"}


def _normalize_column_modes(
    column_modes: dict[int, str] | None,
) -> dict[int, str]:
    if not column_modes:
        return {}

    normalized: dict[int, str] = {}

    for raw_column, raw_mode in column_modes.items():
        column = int(raw_column)
        mode = str(raw_mode)

        if column < 1 or column > 10:
            raise ValueError(
                f"Crossmatch column index must be between 1 and 10: {column}"
            )

        if mode not in ALLOWED_CROSSMATCH_CELL_MODES:
            raise ValueError(
                f"Unsupported crossmatch cell mode for column {column}: {mode}"
            )

        normalized[column] = mode

    return normalized


def _well_column(well_id: str) -> int | None:
    digits = ""

    for character in reversed(str(well_id)):
        if not character.isdigit():
            break
        digits = character + digits

    if not digits:
        return None

    return int(digits)


def _emit_progress(
    progress_cb: ProgressCallback | None,
    **values: Any,
) -> None:
    """
    Send a progress event to the application layer.

    The analysis package does not know how progress is persisted. The
    application callback may write to Redis, update job metadata, or ignore
    the event.
    """
    if progress_cb is not None:
        progress_cb(values)


def _emit_stage_progress(
    *,
    progress_cb: ProgressCallback | None,
    stage: str,
    done: int,
    total: int,
    done_wells: list[str],
    current_well: str | None = None,
) -> None:
    """
    Emit one complete image-analysis progress state.

    Supplying all well-related fields with every event prevents stale or
    partially merged progress states in the application layer.
    """
    _emit_progress(
        progress_cb,
        status="running",
        stage=stage,
        done=done,
        total=total,
        current_well=current_well,
        done_wells=done_wells.copy(),
    )


def _env_int(
    name: str,
    default: int,
) -> int:
    raw = os.getenv(name)

    if raw is None or raw == "":
        return default

    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer environment value: %s=%r; using %d",
            name,
            raw,
            default,
        )
        return default

    return max(1, value)


def _segmenter_device_type(
    segmenter: SegmenterUNetInference,
) -> str:
    device = getattr(
        segmenter,
        "device",
        None,
    )

    device_type = getattr(
        device,
        "type",
        None,
    )

    return str(
        device_type or "unknown"
    )


def _segmenter_is_cuda(
    segmenter: SegmenterUNetInference,
) -> bool:
    return (
        _segmenter_device_type(segmenter)
        == "cuda"
    )


def _cpu_segmentation_workers(
    segmenter: SegmenterUNetInference,
    total: int,
) -> int:
    if _segmenter_is_cuda(segmenter):
        return 1

    workers = _env_int(
        "IMAGE_ANALYSIS_CPU_WORKERS",
        4,
    )

    return max(
        1,
        min(
            workers,
            max(1, total),
        ),
    )


def _prepare_cpu_parallel_runtime(
    max_workers: int,
) -> None:
    if max_workers <= 1:
        return

    ## Avoid nested CPU oversubscription. Python worker threads multiplied by
    ## Torch/OpenMP worker threads can otherwise consume substantially more
    ## CPU resources than requested.

    try:
        torch.set_num_threads(1)
    except Exception:
        logger.debug(
            "Could not set Torch thread count.",
            exc_info=True,
        )

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        logger.debug(
            "Could not set Torch interop thread count.",
            exc_info=True,
        )


def _extract_roi_from_image(
    image: np.ndarray,
    segmenter: SegmenterUNetInference,
    qc_monitor: QCMonitor | None,
    extractor: RGBExtractor,
    well_id: str,
    qc: bool = False,
) -> tuple[WellResult, dict[str, Any]]:
    segmentation_results: dict[str, Any] = segmenter(
        image
    )

    if qc:
        if qc_monitor is None:
            raise RuntimeError(
                "QC was requested but no QC monitor was provided."
            )

        qc_output = qc_monitor(
            instance_labels=segmentation_results[
                "instance_labels"
            ],
            probs=segmentation_results.get(
                "probs"
            ),
            image=image,
        )

        segmentation_results["qc"] = {
            "well": qc_output["well"],
            "roi_table": qc_output[
                "roi_table"
            ],
        }

        segmentation_results[
            "instance_labels_qc"
        ] = qc_output[
            "instances_filtered"
        ]

        labels_for_rois = (
            segmentation_results[
                "instance_labels_qc"
            ]
        )

    else:
        labels_for_rois = (
            segmentation_results[
                "instance_labels"
            ]
        )

    rois_dict = extractor(
        image,
        labels_for_rois,
    )

    segmentation_results[
        "instance_labels_for_rois"
    ] = labels_for_rois

    rois = [
        ROIResult(**value)
        for value in rois_dict
    ]

    well_result = WellResult(
        well_id=well_id,
        rois=rois,
        qc=segmentation_results.get(
            "qc",
            {},
        ),
    )

    return (
        well_result,
        segmentation_results,
    )


def _segment_one_well(
    *,
    well: Any,
    segmenter: SegmenterUNetInference,
    qc: bool,
) -> tuple[
    str,
    WellResult,
    np.ndarray,
]:
    thread_name = (
        threading.current_thread().name
    )

    started_at = time.perf_counter()

    logger.info(
        "Starting segmentation for well %s on %s.",
        well.well_id,
        thread_name,
    )

    try:
        image = well.image

        if image is None:
            raise ValueError(
                f"No image provided for well {well.well_id}."
            )

        if (
            not hasattr(image, "shape")
            or image.size == 0
        ):
            raise ValueError(
                f"Image for well {well.well_id} is empty."
            )

        ## Keep these objects local to each worker. This avoids shared mutable
        ## state in ROI extraction and QC processing.
        
        extractor = RGBExtractor()
        qc_monitor = (
            QCMonitor()
            if qc
            else None
        )

        (
            well_result,
            segmentation_results,
        ) = _extract_roi_from_image(
            image=image,
            extractor=extractor,
            segmenter=segmenter,
            qc_monitor=qc_monitor,
            well_id=well.well_id,
            qc=qc,
        )

        labels_for_preview = (
            segmentation_results[
                "instance_labels_for_rois"
            ].astype(
                np.uint16,
                copy=False,
            )
        )

        ## Probability maps and temporary inference arrays are no longer
        ## required after ROI extraction.
        
        segmentation_results.clear()

        duration = (
            time.perf_counter()
            - started_at
        )

        logger.info(
            "Finished segmentation for well %s in %.1f seconds on %s.",
            well.well_id,
            duration,
            thread_name,
        )

        return (
            well.well_id,
            well_result,
            labels_for_preview,
        )

    except Exception:
        duration = (
            time.perf_counter()
            - started_at
        )

        logger.exception(
            "Segmentation failed for well %s after %.1f seconds on %s.",
            well.well_id,
            duration,
            thread_name,
        )

        raise


def _segment_plate_wells(
    *,
    wells_list: list[Any],
    segmenter: SegmenterUNetInference,
    qc: bool,
    progress_cb: ProgressCallback | None,
) -> tuple[
    dict[str, WellResult],
    dict[str, np.ndarray],
    list[str],
]:
    total = len(wells_list)

    max_workers = (
        _cpu_segmentation_workers(
            segmenter,
            total,
        )
    )

    _prepare_cpu_parallel_runtime(
        max_workers
    )

    device_type = (
        _segmenter_device_type(
            segmenter
        )
    )

    mode = (
        "sequential"
        if max_workers == 1
        else "parallel"
    )

    logger.info(
        "Segmentation mode=%s device=%s max_workers=%d total_wells=%d.",
        mode,
        device_type,
        max_workers,
        total,
    )

    completed: dict[
        str,
        tuple[
            WellResult,
            np.ndarray,
        ],
    ] = {}

    done = 0
    done_wells: list[str] = []

    _emit_stage_progress(
        progress_cb=progress_cb,
        stage="segmenting",
        done=0,
        total=total,
        current_well=None,
        done_wells=[],
    )

    if max_workers == 1:
        logger.info(
            "Running sequential segmentation; CUDA available=%s.",
            torch.cuda.is_available(),
        )

        for well in wells_list:
            ## This event remains active for the duration of inference and is
            ## what makes the current well light up in the frontend.
            
            _emit_stage_progress(
                progress_cb=progress_cb,
                stage="segmenting",
                current_well=well.well_id,
                done=done,
                total=total,
                done_wells=done_wells,
            )

            (
                well_id,
                well_result,
                labels,
            ) = _segment_one_well(
                well=well,
                segmenter=segmenter,
                qc=qc,
            )

            completed[well_id] = (
                well_result,
                labels,
            )

            done += 1

            if well_id not in done_wells:
                done_wells.append(
                    well_id
                )

            
            # Clear current_well after completion. The finished well is now
            # represented exclusively through done_wells, allowing the
            # frontend to change it from running to done immediately.
            
            _emit_stage_progress(
                progress_cb=progress_cb,
                stage="segmenting",
                current_well=None,
                done=done,
                total=total,
                done_wells=done_wells,
            )

    else:
        logger.info(
            "Running parallel segmentation; CUDA available=%s.",
            torch.cuda.is_available(),
        )

        # Several wells are processed concurrently, so there is no accurate
        # single current_well. Completed wells still light up through
        # done_wells as futures finish.
        
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="image-seg",
        ) as pool:
            futures = {
                pool.submit(
                    _segment_one_well,
                    well=well,
                    segmenter=segmenter,
                    qc=qc,
                ): well.well_id
                for well in wells_list
            }

            try:
                for future in as_completed(
                    futures
                ):
                    scheduled_well_id = (
                        futures[future]
                    )

                    try:
                        (
                            well_id,
                            well_result,
                            labels,
                        ) = future.result()

                    except Exception as exc:
                        for other_future in futures:
                            other_future.cancel()

                        raise RuntimeError(
                            "Segmentation failed for well "
                            f"{scheduled_well_id}."
                        ) from exc

                    completed[well_id] = (
                        well_result,
                        labels,
                    )

                    done += 1

                    if (
                        well_id
                        not in done_wells
                    ):
                        done_wells.append(
                            well_id
                        )

                    _emit_stage_progress(
                        progress_cb=progress_cb,
                        stage="segmenting",
                        current_well=None,
                        done=done,
                        total=total,
                        done_wells=done_wells,
                    )

            finally:
                _emit_stage_progress(
                    progress_cb=progress_cb,
                    stage="segmenting",
                    current_well=None,
                    done=done,
                    total=total,
                    done_wells=done_wells,
                )

    missing = [
        well.well_id
        for well in wells_list
        if well.well_id
        not in completed
    ]

    if missing:
        raise RuntimeError(
            "Segmentation did not return results for wells: "
            + ", ".join(missing)
        )

    # Keep completion order deterministic for all subsequent stages and for
    # the final progress response.
    
    ordered_done_wells = [
        well.well_id
        for well in wells_list
    ]

    _emit_stage_progress(
        progress_cb=progress_cb,
        stage="segmenting",
        current_well=None,
        done=total,
        total=total,
        done_wells=ordered_done_wells,
    )

    per_well = {
        well.well_id: completed[
            well.well_id
        ][0]
        for well in wells_list
    }

    per_well_instance_labels = {
        well.well_id: completed[
            well.well_id
        ][1]
        for well in wells_list
    }

    return (
        per_well,
        per_well_instance_labels,
        ordered_done_wells,
    )


def run_image_analysis(
    layout: PlateLayout,
    image_order: list[str],
    image_filenames: list[str],
    *,
    input_root: str | Path,
    segmented_output_dir: str | Path,
    segmented_url_prefix: str,
    progress_cb: ProgressCallback | None = None,
    unet_config: dict[str, Any] | None = None,
    qc: bool = False,
    assay_type: str = "pra",
    hla_layout: ParsedPlateLayout | None = None,
    pra_positivity_threshold: float = 20.0,
    column_modes: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Run CDC image analysis.

    Parameters
    ----------
    layout
        Plate well-role layout.
    image_order
        Well IDs in image acquisition order.
    image_filenames
        File paths relative to ``input_root``.
    input_root
        Root directory used to resolve input files.
    segmented_output_dir
        Directory where segmented preview images are written.
    segmented_url_prefix
        URL prefix used in the returned well results.
    progress_cb
        Optional callback receiving complete progress events.
    unet_config
        Optional UNet configuration. The default configuration is copied when
        omitted.
    qc
        Whether to run ROI QC filtering.
    assay_type
        Either ``"pra"`` or ``"crossmatch"``.
    hla_layout
        Parsed HLA layout required for PRA analysis.
    pra_positivity_threshold
        Positivity threshold used for PRA calculations.
    column_modes
        Optional mapping from plate column number to ``"T"``, ``"B"``,
        ``"T/B"``, or ``"empty"``. Used for CDC crossmatch summaries.

    Returns
    -------
    dict
        JSON-compatible analysis result.

    Notes
    -----
    This function does not write job state to Redis and does not mark the job
    as completed or failed. Those actions belong to the application service.
    """
    if assay_type not in {
        "pra",
        "crossmatch",
    }:
        raise ValueError(
            "assay_type must be either 'pra' or 'crossmatch'."
        )

    normalized_column_modes = _normalize_column_modes(column_modes)

    if assay_type == "crossmatch" and not normalized_column_modes:
        raise ValueError(
            "Crossmatch analysis requires column_modes."
        )

    if (
        len(image_order)
        != len(image_filenames)
    ):
        raise ValueError(
            "image_order and image_filenames must have the same length."
        )

    if not image_filenames:
        raise ValueError(
            "At least one image file is required."
        )

    logger.info(
        "Starting image analysis: assay_type=%s images=%d.",
        assay_type,
        len(image_filenames),
    )

    input_root_path = Path(
        input_root
    ).resolve()

    segmented_dir = Path(
        segmented_output_dir
    ).resolve()

    segmented_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = copy.deepcopy(
        unet_config
        if unet_config is not None
        else UNET_CONFIG
    )

    config.setdefault(
        "instance_cfg",
        INSTANCE_CONFIG.to_dict(),
    )

    initial_total = len(
        image_order
    )

    _emit_stage_progress(
        progress_cb=progress_cb,
        stage="loading_images",
        done=0,
        total=initial_total,
        current_well=None,
        done_wells=[],
    )

    segmenter = (
        SegmenterUNetInference.from_config(
            config
        )
    )

    calibrator = (
        PCNCGaussian2DCalibrator()
    )

    classifier_constructor = (
        ROIClassifierGaussian2D3Way
    )

    images: list[np.ndarray] = (
        load_images(
            image_filenames,
            str(input_root_path),
            scale=True,
        )
    )

    if (
        len(images)
        != len(image_filenames)
    ):
        raise RuntimeError(
            "The image loader returned a different number of images than "
            "requested."
        )

    plate = create_plate(
        layout,
        images,
        image_order,
        image_filenames,
    )

    wells_list = list(
        plate.get()
    )

    total = len(
        wells_list
    )

    if total == 0:
        raise ValueError(
            "The plate contains no processable wells."
        )

    (
        per_well,
        per_well_instance_labels,
        completed_wells,
    ) = _segment_plate_wells(
        wells_list=wells_list,
        segmenter=segmenter,
        qc=qc,
        progress_cb=progress_cb,
    )

    _emit_stage_progress(
        progress_cb=progress_cb,
        stage="calibrating",
        done=total,
        total=total,
        current_well=None,
        done_wells=completed_wells,
    )

    positive_control_rois = [
        per_well[
            well.well_id
        ].rois
        for well in plate.get(
            "positive"
        )
    ]

    negative_control_rois = [
        per_well[
            well.well_id
        ].rois
        for well in plate.get(
            "negative"
        )
    ]

    if not positive_control_rois:
        raise ValueError(
            "No positive-control wells were available for calibration."
        )

    if not negative_control_rois:
        raise ValueError(
            "No negative-control wells were available for calibration."
        )

    calibration = calibrator.fit(
        pc_wells=[
            [
                roi.__dict__
                for roi in rois
            ]
            for rois
            in positive_control_rois
        ],
        nc_wells=[
            [
                roi.__dict__
                for roi in rois
            ]
            for rois
            in negative_control_rois
        ],
    )

    _emit_stage_progress(
        progress_cb=progress_cb,
        stage="classifying",
        done=total,
        total=total,
        current_well=None,
        done_wells=completed_wells,
    )

    classifier = (
        classifier_constructor(
            calibration
        )
    )

    for well_result in per_well.values():
        updated_rois = classifier(
            [
                roi.__dict__
                for roi
                in well_result.rois
            ]
        )

        well_result.rois = [
            ROIResult(**value)
            for value
            in updated_rois
        ]

    _emit_stage_progress(
        progress_cb=progress_cb,
        stage="saving_previews",
        done=total,
        total=total,
        current_well=None,
        done_wells=completed_wells,
    )

    for (
        well_id,
        well_result,
    ) in per_well.items():
        segmented_path = (
            segmented_dir
            / f"{well_id}.png"
        )

        save_segmented_preview(
            instance_labels=(
                per_well_instance_labels[
                    well_id
                ]
            ),
            rois=well_result.rois,
            out_path=segmented_path,
        )

        well_result.preview_path = str(
            segmented_path
        )

        well_result.store_paths[
            "segmented_preview"
        ] = str(
            segmented_path
        )

    del per_well_instance_labels

    _emit_stage_progress(
        progress_cb=progress_cb,
        stage="finalizing",
        done=total,
        total=total,
        current_well=None,
        done_wells=completed_wells,
    )

    positive_control_well_ids = [
        well.well_id
        for well in plate.get(
            "positive"
        )
    ]

    negative_control_well_ids = [
        well.well_id
        for well in plate.get(
            "negative"
        )
    ]

    positive_control_fractions = [
        frac_pos_raw(
            per_well[well_id]
        )
        for well_id
        in positive_control_well_ids
    ]

    negative_control_fractions = [
        frac_pos_raw(
            per_well[well_id]
        )
        for well_id
        in negative_control_well_ids
    ]

    positive_reference = float(
        np.nanmean(
            positive_control_fractions
        )
    )

    negative_reference = float(
        np.nanmean(
            negative_control_fractions
        )
    )

    for well_result in per_well.values():
        raw_fraction = frac_pos_raw(
            well_result
        )

        if (
            np.isnan(raw_fraction)
            or np.isnan(
                positive_reference
            )
            or np.isnan(
                negative_reference
            )
            or positive_reference
            == negative_reference
        ):
            corrected_fraction = np.nan

        else:
            corrected_fraction = (
                (
                    raw_fraction
                    - negative_reference
                )
                / (
                    positive_reference
                    - negative_reference
                )
                * 100.0
            )

            corrected_fraction = float(
                np.clip(
                    corrected_fraction,
                    0.0,
                    100.0,
                )
            )

        well_result.corrected_frac_pos = (
            corrected_fraction
        )

    summary = build_cdc_summary(
        per_well=per_well,
        plate=plate,
        config=CDC_SUMMARY_CONFIG,
        assay_type=assay_type,
        column_modes=normalized_column_modes,
    )

    pra_analysis = None

    if assay_type == "pra":
        if hla_layout is None:
            raise ValueError(
                "PRA analysis requires hla_layout. "
                "Pass the parsed Excel layout into run_image_analysis."
            )

        sample_well_ids = {
            well.well_id.upper()
            for well in plate.get(
                "sample"
            )
        }

        pra_analysis = {
            "positivity_threshold": (
                pra_positivity_threshold
            ),
            "included_well_type": (
                "sample"
            ),
            "included_wells": sorted(
                sample_well_ids
            ),
            "reactivity_score": (
                calculate_pra_reactivity_score(
                    per_well=per_well,
                    hla_layout=hla_layout,
                    positivity_threshold=(
                        pra_positivity_threshold
                    ),
                    include_well_ids=(
                        sample_well_ids
                    ),
                )
            ),
            "alleles": (
                calculate_allele_reactivity_evidence(
                    per_well=per_well,
                    hla_layout=hla_layout,
                    positivity_threshold=(
                        pra_positivity_threshold
                    ),
                    include_well_ids=(
                        sample_well_ids
                    ),
                )
            ),
        }

    role_map = (
        getattr(
            layout,
            "wells",
            {},
        )
        or {}
    )

    url_prefix = (
        segmented_url_prefix.rstrip(
            "/"
        )
    )

    def serialize_well(
        well_id: str,
        well_result: WellResult,
    ) -> dict[str, Any]:
        role = role_map.get(well_id)

        if role == "positive":
            automated_call = "positive"
        elif role == "negative":
            automated_call = "negative"
        elif role == "sample":
            automated_call = automated_well_call(
                well_result.corrected_frac_pos,
                assay_type=assay_type,
                pra_positive_cutoff=pra_positivity_threshold,
                config=CDC_SUMMARY_CONFIG,
            )
        else:
            automated_call = None

        return {
            **well_result.summary(),
            "role": role,
            "cell_mode": normalized_column_modes.get(
                _well_column(well_id)
            ),
            "segmented_image_url": f"{url_prefix}/{well_id}.png",
            "automated_call": automated_call,
            "effective_call": automated_call,
            "manual_override": None,
        }

    result = {
        "assay_type": assay_type,
        "calib": calibration,
        "column_modes": {
            str(column): mode
            for column, mode in normalized_column_modes.items()
        },
        "wells": {
            well_id: serialize_well(well_id, well_result)
            for well_id, well_result in per_well.items()
        },
        "summary": summary,
        "pra_analysis": pra_analysis,
        "manual_overrides": {},
        "manual_override_history": [],
    }

    json_result = to_jsonable(
        result
    )

    logger.info(
        "Finished image analysis: assay_type=%s wells=%d.",
        assay_type,
        total,
    )

    return json_result