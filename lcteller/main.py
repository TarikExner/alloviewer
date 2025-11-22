
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
from typing import List
import copy
from .image_utils import load_images
from .structs import (
    PlateLayout,
    Plate,
    WellImage,
    ROIResult, 
    WellResult
)
from .segmenter import SegmenterUNetInference
from .extractor import RGBExtractor
from .calibrators import PCNCMedianCalibrator
from .classifiers import ROIClassifier

from .qc import QCMonitor

from .config import UNET_CONFIG, INSTANCE_CONFIG, WELL_QC_CONFIG


def create_plate(layout: PlateLayout,
                 images: List[np.ndarray]) -> Plate:
    plate = Plate(plate_id="SIM001")
    for i, (well_id, role) in enumerate(layout.wells.items()):
        plate.add(WellImage(well_id, role=role, image=images[i]))

    return plate

def run_job(layout: PlateLayout,
            image_order: List[str],
            template_filename: str,
            image_filenames: List[str],
            data_dir: str):
    
    unet_config = copy.deepcopy(UNET_CONFIG)
    unet_config["instance_cfg"] = INSTANCE_CONFIG

    segmenter = SegmenterUNetInference.from_config(UNET_CONFIG)
    qc_monitor = QCMonitor()
    extractor = RGBExtractor()
    calibrator = PCNCMedianCalibrator()
    classifier_ctor = ROIClassifier
    per_well: dict[str, WellResult] = {}

    # Function start: Load images
    images: List[np.ndarray] = load_images(image_filenames, data_dir, scale = True)
    plate = create_plate(layout, images)

    for well in plate.get():
        print(f"Calculating well {well.well_id}")
        image = well.image

        if image.shape[0] == 0:
            raise ValueError("No image provided!")

        segmentation_results: dict = segmenter(image)
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

        rois = [ROIResult(**d) for d in rois_dict]

        wr = WellResult(
            well_id=well.well_id,
            rois=rois,
            qc=segmentation_results.get("qc", {}),
        )
        per_well[well.well_id] = wr

    pc = [per_well[w.well_id].rois for w in plate.get("positive")]
    nc = [per_well[w.well_id].rois for w in plate.get("negative")]

    calib = calibrator.fit(
        pc_wells=[[r.__dict__ for r in rs] for rs in pc],
        nc_wells=[[r.__dict__ for r in rs] for rs in nc],
    )

    clf = classifier_ctor(calib)
    for wr in per_well.values():
        updated = clf([r.__dict__ for r in wr.rois])
        wr.rois = [ROIResult(**d) for d in updated]

    return {
        "calib": calib,
        "wells": {wid: wr.summary() for wid, wr in per_well.items()},
    }

