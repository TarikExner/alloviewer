"""

layout=PlateLayout(
    wells={'A1': 'positive', 'A2': 'negative', 'A3': 'sample', 'A4': 'sample', 'A5': 'sample'}
)

image_order=['A1', 'A2', 'A3', 'A4', 'A5']
template_filename='Figure_4.pdf'
image_filenames=[
    'test_images/NC.jpg',
    'test_images/PC.jpg',
    'test_images/Sample_0.1.jpg',
    'test_images/Sample_0.3.jpg',
    'test_images/Sample_0.6.jpg'
]

"""
import numpy as np
from typing import List, Optional, Dict, Any
import copy
from .image_analysis import load_images
from .image_analysis.structs import (
    PlateLayout,
    ROIResult, 
    WellResult
)
from .image_analysis.segmenter import SegmenterUNetInference
from .image_analysis.extractor import RGBExtractor
from .image_analysis.calibrators import PCNCMedianCalibrator, PCNCGaussianRGCalibrator
from .image_analysis.classifiers import ROIClassifier, ROIClassifierGaussian3Way

from .image_analysis.qc import QCMonitor

from .image_analysis.config import UNET_CONFIG, INSTANCE_CONFIG

from .image_analysis.utils import create_plate, frac_pos_raw

from app.models import JOB_PROGRESS, JOB_RESULTS

def _extract_roi_from_image(image: np.ndarray,
                            segmenter: SegmenterUNetInference,
                            qc_monitor: Optional[QCMonitor],
                            extractor: RGBExtractor,
                            well_id: str,
                            qc: bool = False) -> WellResult:
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

        rois_dict = extractor(image, segmentation_results["instance_labels_qc"])
    else:
        rois_dict = extractor(image, segmentation_results["instance_labels"])

    rois = [ROIResult(**d) for d in rois_dict]

    return WellResult(
        well_id=well_id,
        rois=rois,
        qc=segmentation_results.get("qc", {}),
    )


def run_job(
    layout: PlateLayout,
    image_order: List[str],
    image_filenames: List[str],
    data_dir: str,
    template_filename: Optional[str],
    job_id: Optional[str] = None,
    unet_config: Optional[dict] = UNET_CONFIG,
    qc: bool = False,
):
    if not job_id:
        job_id = "MY_JOB"

    if not unet_config:
        unet_config = copy.deepcopy(UNET_CONFIG)
        unet_config["instance_cfg"] = INSTANCE_CONFIG

    # use the (possibly updated) config
    segmenter = SegmenterUNetInference.from_config(unet_config)
    qc_monitor = QCMonitor()
    extractor = RGBExtractor()
    calibrator = PCNCGaussianRGCalibrator()
    classifier_ctor = ROIClassifierGaussian3Way
    per_well: dict[str, WellResult] = {}

    # 1) load images and build plate
    images: List[np.ndarray] = load_images(
        image_filenames, data_dir, scale=True
    )
    plate = create_plate(layout, images, image_order, image_filenames)

    wells_list = list(plate.get())
    total = len(wells_list)

    JOB_PROGRESS[job_id] = {
      "status": "running",
      "done": 0,
      "total": total,
      "current_well": None,
      "done_wells": [],
    }

    # 2) segmentation + feature extraction
    for well_idx, well in enumerate(plate.get()):
        JOB_PROGRESS[job_id]["current_well"] = well.well_id
        JOB_PROGRESS[job_id]["done"] = well_idx
        print(f"Calculating well {well.well_id}")
        image = well.image

        if image.shape[0] == 0:
            raise ValueError("No image provided!")

        per_well[well.well_id] = _extract_roi_from_image(
            image = image,
            extractor = extractor,
            segmenter = segmenter,
            qc_monitor = qc_monitor,
            well_id = well.well_id,
            qc = qc
        )
        JOB_PROGRESS[job_id]["done_wells"].append(well.well_id)
        

    # 3) build PC / NC sets for calibration
    pc = [per_well[w.well_id].rois for w in plate.get("positive")]
    nc = [per_well[w.well_id].rois for w in plate.get("negative")]

    calib = calibrator.fit(
        pc_wells=[[r.__dict__ for r in rs] for rs in pc],
        nc_wells=[[r.__dict__ for r in rs] for rs in nc],
    )

    # 4) classify ROIs
    clf = classifier_ctor(calib)
    for wr in per_well.values():
        updated = clf([r.__dict__ for r in wr.rois])
        wr.rois = [ROIResult(**d) for d in updated]

    # 6) compute reference values from PC / NC wells
    pc_well_ids = [w.well_id for w in plate.get("positive")]
    nc_well_ids = [w.well_id for w in plate.get("negative")]

    pc_fracs = [frac_pos_raw(per_well[wid]) for wid in pc_well_ids]
    nc_fracs = [frac_pos_raw(per_well[wid]) for wid in nc_well_ids]

    pc_ref = float(np.nanmean(pc_fracs))  # mean % positive in PC wells
    nc_ref = float(np.nanmean(nc_fracs))  # mean % positive in NC wells

    # 7) compute corrected_frac_pos for all wells
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
            # map NC mean -> 0, PC mean -> 100
            corr = (raw - nc_ref) / (pc_ref - nc_ref) * 100.0
            corr = float(np.clip(corr, 0.0, 100.0))

        wr.corrected_frac_pos = corr


    result = {
        "calib": calib,
        "wells": {wid: wr.summary() for wid, wr in per_well.items()},
    }

    JOB_RESULTS[job_id] = result
    JOB_PROGRESS[job_id]["status"] = "done"
    JOB_PROGRESS[job_id]["done"] = total
    JOB_PROGRESS[job_id]["current_well"] = None

    return result

