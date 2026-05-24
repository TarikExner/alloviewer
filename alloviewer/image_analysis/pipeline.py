import copy
from pathlib import Path
from typing import List, Optional, Literal

import numpy as np

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
from .services.analysis import calculate_allele_reactivity_evidence, calculate_pra_reactivity_score

from app.models import IMAGE_JOB_PROGRESS, IMAGE_JOB_RESULTS

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

    try:
        if not unet_config:
            unet_config = copy.deepcopy(UNET_CONFIG)
            unet_config["instance_cfg"] = INSTANCE_CONFIG.to_dict()

        segmenter = SegmenterUNetInference.from_config(unet_config)
        qc_monitor = QCMonitor()
        extractor = RGBExtractor()
        calibrator = PCNCGaussian2DCalibrator()
        classifier_ctor = ROIClassifierGaussian2D3Way

        per_well: dict[str, WellResult] = {}
        per_well_instance_labels: dict[str, np.ndarray] = {}

        segmented_dir = Path(data_dir) / "segmented" / job_id

        images: List[np.ndarray] = load_images(
            image_filenames,
            data_dir,
            scale=True,
        )

        plate = create_plate(layout, images, image_order, image_filenames)

        wells_list = list(plate.get())
        total = len(wells_list)

        IMAGE_JOB_PROGRESS[job_id] = {
            "status": "running",
            "stage": "segmenting",
            "done": 0,
            "total": total,
            "current_well": None,
            "done_wells": [],
        }

        for well_idx, well in enumerate(plate.get()):
            IMAGE_JOB_PROGRESS[job_id]["stage"] = "segmenting"
            IMAGE_JOB_PROGRESS[job_id]["current_well"] = well.well_id

            print(f"Calculating well {well.well_id}")

            image = well.image

            if image.shape[0] == 0:
                raise ValueError("No image provided.")

            wr, segmentation_results = _extract_roi_from_image(
                image=image,
                extractor=extractor,
                segmenter=segmenter,
                qc_monitor=qc_monitor,
                well_id=well.well_id,
                qc=qc,
            )

            per_well[well.well_id] = wr

            per_well_instance_labels[well.well_id] = segmentation_results[
                "instance_labels_for_rois"
            ].astype(np.uint16, copy=False)

            IMAGE_JOB_PROGRESS[job_id]["done_wells"].append(well.well_id)
            IMAGE_JOB_PROGRESS[job_id]["done"] = well_idx + 1

        IMAGE_JOB_PROGRESS[job_id]["current_well"] = None

        IMAGE_JOB_PROGRESS[job_id]["stage"] = "calibrating"

        pc = [per_well[w.well_id].rois for w in plate.get("positive")]
        nc = [per_well[w.well_id].rois for w in plate.get("negative")]

        calib = calibrator.fit(
            pc_wells=[[r.__dict__ for r in rs] for rs in pc],
            nc_wells=[[r.__dict__ for r in rs] for rs in nc],
        )

        IMAGE_JOB_PROGRESS[job_id]["stage"] = "classifying"

        clf = classifier_ctor(calib)

        for wr in per_well.values():
            updated = clf([r.__dict__ for r in wr.rois])
            wr.rois = [ROIResult(**d) for d in updated]

        IMAGE_JOB_PROGRESS[job_id]["stage"] = "saving_previews"

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

        IMAGE_JOB_PROGRESS[job_id]["stage"] = "finalizing"

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

        result = {
            "calib": calib,
            "wells": {
                wid: {
                    **wr.summary(),
                    "segmented_image_url": f"/api/process/{job_id}/segmented/{wid}.png",
                }
                for wid, wr in per_well.items()
            },
            "summary": summary,
            "pra_analysis": pra_analysis,
        }

        print(result)

        IMAGE_JOB_RESULTS[job_id] = to_jsonable(result)

        IMAGE_JOB_PROGRESS[job_id]["status"] = "done"
        IMAGE_JOB_PROGRESS[job_id]["stage"] = "done"
        IMAGE_JOB_PROGRESS[job_id]["done"] = total
        IMAGE_JOB_PROGRESS[job_id]["current_well"] = None

        return result

    except Exception as e:
        IMAGE_JOB_PROGRESS[job_id] = {
            **IMAGE_JOB_PROGRESS.get(job_id, {}),
            "status": "error",
            "error": repr(e),
            "current_well": None,
        }

        print(f"Image analysis failed for job {job_id}: {repr(e)}")
        raise
