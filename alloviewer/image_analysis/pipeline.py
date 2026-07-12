import os
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Any, Callable
import logging

import numpy as np
import torch

from . import load_images
from .structs import PlateLayout, ROIResult, WellResult, ParsedPlateLayout
from .segmenter import SegmenterUNetInference
from .extractor import RGBExtractor
from .calibrators import PCNCGaussian2DCalibrator
from .classifiers import ROIClassifierGaussian2D3Way
from .qc import QCMonitor
from .config import UNET_CONFIG, INSTANCE_CONFIG, CDC_SUMMARY_CONFIG
from .utils import (
    build_cdc_summary,
    create_plate,
    frac_pos_raw,
    save_segmented_preview,
    to_jsonable,
)
from .services.analysis import (
    calculate_allele_reactivity_evidence,
    calculate_pra_reactivity_score,
)


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)

    print(f"RAW: {raw}")

    if raw is None or raw == "":
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return max(1, value)


def _emit_progress(
    callback: ProgressCallback | None,
    **values: Any,
) -> None:
    if callback is not None:
        callback(values)


def _segmenter_device_type(segmenter: SegmenterUNetInference) -> str:
    device = getattr(segmenter, "device", None)
    device_type = getattr(device, "type", None)
    return str(device_type or "unknown")


def _segmenter_is_cuda(segmenter: SegmenterUNetInference) -> bool:
    return _segmenter_device_type(segmenter) == "cuda"


def _cpu_segmentation_workers(
    segmenter: SegmenterUNetInference,
    total: int,
) -> int:
    if _segmenter_is_cuda(segmenter):
        return 1

    workers = _env_int("IMAGE_ANALYSIS_CPU_WORKERS", 4)
    return max(1, min(workers, total))


def _prepare_cpu_parallel_runtime(max_workers: int) -> None:
    if max_workers <= 1:
        return

    # Avoid nested CPU oversubscription:
    # 4 Python worker threads x N Torch/OpenMP threads can otherwise explode.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _update_segmentation_progress(
    *,
    progress_cb: ProgressCallback | None,
    current_well: str | None,
    done: int,
    total: int,
    done_wells: list[str],
) -> None:
    _emit_progress(
        progress_cb,
        status="running",
        stage="segmenting",
        current_well=current_well,
        done=done,
        total=total,
        done_wells=done_wells.copy(),
    )


def _extract_roi_from_image(
    image: np.ndarray,
    segmenter: SegmenterUNetInference,
    qc_monitor: Optional[QCMonitor],
    extractor: RGBExtractor,
    well_id: str,
    qc: bool = False,
) -> tuple[WellResult, dict]:
    segmentation_results: dict = segmenter(image)

    if qc:
        assert qc_monitor is not None

        qc_out = qc_monitor(
            instance_labels=segmentation_results["instance_labels"],
            probs=segmentation_results.get("probs"),
            image=image,
        )

        segmentation_results["qc"] = {
            "well": qc_out["well"],
            "roi_table": qc_out["roi_table"],
        }
        segmentation_results["instance_labels_qc"] = qc_out["instances_filtered"]

        labels_for_rois = segmentation_results["instance_labels_qc"]
        rois_dict = extractor(image, labels_for_rois)
    else:
        labels_for_rois = segmentation_results["instance_labels"]
        rois_dict = extractor(image, labels_for_rois)

    segmentation_results["instance_labels_for_rois"] = labels_for_rois

    rois = [ROIResult(**d) for d in rois_dict]

    wr = WellResult(
        well_id=well_id,
        rois=rois,
        qc=segmentation_results.get("qc", {}),
    )

    return wr, segmentation_results


def _segment_one_well(
    *,
    well,
    segmenter: SegmenterUNetInference,
    qc: bool,
    job_id: str,
) -> tuple[str, WellResult, np.ndarray]:
    thread_name = threading.current_thread().name
    t0 = time.perf_counter()

    print(
        f"[image-job {job_id}] START well {well.well_id} on {thread_name}",
        flush=True,
    )

    try:
        image = well.image

        if image.shape[0] == 0:
            raise ValueError(f"No image provided for well {well.well_id}.")

        # Keep these local to each thread. Cheap to create, avoids shared
        # mutable state in the ROI extraction and QC code.
        extractor = RGBExtractor()
        qc_monitor = QCMonitor() if qc else None

        wr, segmentation_results = _extract_roi_from_image(
            image=image,
            extractor=extractor,
            segmenter=segmenter,
            qc_monitor=qc_monitor,
            well_id=well.well_id,
            qc=qc,
        )

        labels_for_preview = segmentation_results["instance_labels_for_rois"].astype(
            np.uint16,
            copy=False,
        )

        # Drop the large probability maps and temporary arrays as soon as the
        # only needed output has been copied/referenced.
        segmentation_results.clear()

        dt = time.perf_counter() - t0
        print(
            f"[image-job {job_id}] DONE well {well.well_id} "
            f"in {dt:.1f}s on {thread_name}",
            flush=True,
        )

        return well.well_id, wr, labels_for_preview

    except Exception as exc:
        dt = time.perf_counter() - t0
        print(
            f"[image-job {job_id}] FAILED well {well.well_id} "
            f"after {dt:.1f}s on {thread_name}: {repr(exc)}",
            flush=True,
        )
        raise


def _segment_plate_wells(
    *,
    wells_list,
    segmenter: SegmenterUNetInference,
    qc: bool,
    job_id: str,
) -> tuple[dict[str, WellResult], dict[str, np.ndarray]]:
    total = len(wells_list)
    max_workers = _cpu_segmentation_workers(segmenter, total)
    _prepare_cpu_parallel_runtime(max_workers)

    device_type = _segmenter_device_type(segmenter)
    mode = "sequential" if max_workers == 1 else "parallel"

    print(
        f"[image-job {job_id}] segmentation mode={mode}, "
        f"device={device_type}, max_workers={max_workers}, total_wells={total}",
        flush=True,
    )

    completed: dict[str, tuple[WellResult, np.ndarray]] = {}
    done = 0
    done_wells: list[str] = []

    if max_workers == 1:
        print(
            f"[image-job {job_id}] one worker, "
            f"cuda_available={torch.cuda.is_available()}",
            flush=True,
        )

        for well in wells_list:
            # In sequential mode this marks the well that is currently running.
            update_image_progress(
                job_id,
                stage="segmenting",
                current_well=well.well_id,
                done=done,
                total=total,
                done_wells=done_wells.copy(),
            )

            wid, wr, labels = _segment_one_well(
                well=well,
                segmenter=segmenter,
                qc=qc,
                job_id=job_id,
            )

            completed[wid] = (wr, labels)
            done += 1
            done_wells.append(wid)

            _update_segmentation_progress(
                job_id=job_id,
                current_well=wid,
                done=done,
                total=total,
                done_wells=done_wells,
            )

    else:
        print(
            f"[image-job {job_id}] multiple workers, "
            f"cuda_available={torch.cuda.is_available()}",
            flush=True,
        )

        _update_segmentation_progress(
            job_id=job_id,
            current_well=None,
            done=done,
            total=total,
            done_wells=done_wells,
        )

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
                    job_id=job_id,
                ): well.well_id
                for well in wells_list
            }

            try:
                for future in as_completed(futures):
                    scheduled_wid = futures[future]

                    try:
                        wid, wr, labels = future.result()
                    except Exception as exc:
                        for other in futures:
                            other.cancel()

                        raise RuntimeError(
                            f"Segmentation failed for well {scheduled_wid}."
                        ) from exc

                    completed[wid] = (wr, labels)
                    done += 1
                    done_wells.append(wid)

                    # In parallel mode, current_well means the latest completed
                    # well, not the only active well.
                    _update_segmentation_progress(
                        job_id=job_id,
                        current_well=wid,
                        done=done,
                        total=total,
                        done_wells=done_wells,
                    )

            finally:
                update_image_progress(job_id, current_well=None)

    missing = [well.well_id for well in wells_list if well.well_id not in completed]
    if missing:
        raise RuntimeError(
            "Segmentation did not return results for wells: "
            + ", ".join(missing)
        )

    per_well = {
        well.well_id: completed[well.well_id][0]
        for well in wells_list
    }

    per_well_instance_labels = {
        well.well_id: completed[well.well_id][1]
        for well in wells_list
    }

    return per_well, per_well_instance_labels



def run_image_analysis(
    layout: PlateLayout,
    image_order: List[str],
    image_filenames: List[str],
    data_dir: str,
    template_filename: Optional[str],
    job_id: Optional[str] = None,
    unet_config: Optional[dict] = UNET_CONFIG,
    qc: bool = False,
    assay_type: str = "pra",
    hla_layout: Optional[ParsedPlateLayout] = None,
    pra_positivity_threshold: float = 20.0,
):
    if not job_id:
        job_id = "MY_JOB"

    print(f"[image-job {job_id}] started", flush=True)


    try:
        if not unet_config:
            unet_config = copy.deepcopy(UNET_CONFIG)
            unet_config["instance_cfg"] = INSTANCE_CONFIG.to_dict()

        segmenter = SegmenterUNetInference.from_config(unet_config)
        calibrator = PCNCGaussian2DCalibrator()
        classifier_ctor = ROIClassifierGaussian2D3Way

        segmented_dir = Path(data_dir) / "segmented" / job_id

        images: List[np.ndarray] = load_images(
            image_filenames,
            data_dir,
            scale=True,
        )

        plate = create_plate(layout, images, image_order, image_filenames)

        wells_list = list(plate.get())
        total = len(wells_list)

        set_image_progress(
            job_id,
            {
                "status": "running",
                "stage": "segmenting",
                "done": 0,
                "total": total,
                "current_well": None,
                "done_wells": [],
            },
        )

        per_well, per_well_instance_labels = _segment_plate_wells(
            wells_list=wells_list,
            segmenter=segmenter,
            qc=qc,
            job_id=job_id,
        )

        update_image_progress(job_id, current_well=None)

        update_image_progress(job_id, stage="calibrating")

        pc = [per_well[w.well_id].rois for w in plate.get("positive")]
        nc = [per_well[w.well_id].rois for w in plate.get("negative")]

        calib = calibrator.fit(
            pc_wells=[[r.__dict__ for r in rs] for rs in pc],
            nc_wells=[[r.__dict__ for r in rs] for rs in nc],
        )

        update_image_progress(job_id, stage="classifying")

        clf = classifier_ctor(calib)

        for wr in per_well.values():
            updated = clf([r.__dict__ for r in wr.rois])
            wr.rois = [ROIResult(**d) for d in updated]

        update_image_progress(job_id, stage="saving_previews")

        for wid, wr in per_well.items():
            segmented_path = segmented_dir / f"{wid}.png"

            save_segmented_preview(
                instance_labels=per_well_instance_labels[wid],
                rois=wr.rois,
                out_path=segmented_path,
            )

            wr.preview_path = str(segmented_path)
            wr.store_paths["segmented_preview"] = str(segmented_path)

        del per_well_instance_labels

        update_image_progress(job_id, stage="finalizing")

        pc_well_ids = [w.well_id for w in plate.get("positive")]
        nc_well_ids = [w.well_id for w in plate.get("negative")]

        pc_fracs = [frac_pos_raw(per_well[wid]) for wid in pc_well_ids]
        nc_fracs = [frac_pos_raw(per_well[wid]) for wid in nc_well_ids]

        pc_ref = float(np.nanmean(pc_fracs))
        nc_ref = float(np.nanmean(nc_fracs))

        for wr in per_well.values():
            raw = frac_pos_raw(wr)

            if (
                np.isnan(raw)
                or np.isnan(pc_ref)
                or np.isnan(nc_ref)
                or pc_ref == nc_ref
            ):
                corr = np.nan
            else:
                corr = (raw - nc_ref) / (pc_ref - nc_ref) * 100.0
                corr = float(np.clip(corr, 0.0, 100.0))

            wr.corrected_frac_pos = corr

        summary = build_cdc_summary(
            per_well=per_well,
            plate=plate,
            config=CDC_SUMMARY_CONFIG,
            assay_type=assay_type,
        )

        pra_analysis = None

        if assay_type == "pra":
            if hla_layout is None:
                raise ValueError(
                    "PRA analysis requires hla_layout. "
                    "Pass the parsed Excel layout into run_image_analysis."
                )

            sample_well_ids = {
                w.well_id.upper()
                for w in plate.get("sample")
            }

            pra_analysis = {
                "positivity_threshold": pra_positivity_threshold,
                "included_well_type": "sample",
                "included_wells": sorted(sample_well_ids),
                "reactivity_score": calculate_pra_reactivity_score(
                    per_well=per_well,
                    hla_layout=hla_layout,
                    positivity_threshold=pra_positivity_threshold,
                    include_well_ids=sample_well_ids,
                ),
                "alleles": calculate_allele_reactivity_evidence(
                    per_well=per_well,
                    hla_layout=hla_layout,
                    positivity_threshold=pra_positivity_threshold,
                    include_well_ids=sample_well_ids,
                ),
            }

        role_map = getattr(layout, "wells", {}) or {}

        result = {
            "assay_type": assay_type,
            "calib": calib,
            "wells": {
                wid: {
                    **wr.summary(),
                    "role": role_map.get(wid),
                    "segmented_image_url": f"/api/process/{job_id}/segmented/{wid}.png",
                }
                for wid, wr in per_well.items()
            },
            "summary": summary,
            "pra_analysis": pra_analysis,
        }

        set_image_result(job_id, to_jsonable(result))

        update_image_progress(
            job_id,
            status="done",
            stage="done",
            done=total,
            current_well=None,
        )

        print(f"[image-job {job_id}] done", flush=True)

        return result

    except Exception as e:
        update_image_progress(
            job_id,
            status="error",
            stage="error",
            error=repr(e),
            current_well=None,
        )

        print(f"[image-job {job_id}] failed: {repr(e)}", flush=True)
        raise

