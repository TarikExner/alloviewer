
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
from .image_utils import (
    load_images, tile_images 
)
from .structs import (
    PlateLayout, Plate, WellImage,
    ROIResult, SegmentationResults,
    WellResult
)
from .segmenter import SegmenterUNet, InstanceSegmenter
from .extractor import RGBExtractor
from .calibrators import PCNCMedianCalibrator
from .classifiers import ROIClassifier

from .qc import QCMonitor

from .config import UNET_CONFIG, INSTANCE_CONFIG, WELL_QC_CONFIG



def tile_images(imgs: List[np.ndarray]) -> List[np.ndarray]:
    return [
        tile_image_numpy(img)
        for img in imgs
    ]

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

    segmenter = SegmenterUNet.from_config(UNET_CONFIG)
    instance_segmenter= InstanceSegmenter.from_config(INSTANCE_CONFIG)
    extractor = RGBExtractor()
    calibrator = PCNCMedianCalibrator()
    classifier_ctor = ROIClassifier
    qc_monitor = QCMonitor()
    per_well: dict[str, WellResult] = {}

    # Function start: Load images
    images: List[np.ndarray] = load_images(image_filenames, data_dir)
    images: List[np.ndarray] = tile_images(images)
    plate = create_plate(layout, images)


    cfg_id = "123"


    for well in plate.get():
        print(f"Calculating well {well.well_id}")
        image = well.image

        if image.shape[0] == 0:
            raise ValueError("No image provided!")

        segmentation_results = segmenter(image)
        segmentation_results = instance_segmenter(segmentation_results)
        # qc_out = qc_monitor(
        #     instance_labels=segmentation_results["instance_labels"],
        #     probs=segmentation_results.get("probs"),
        #     image=image,
        #     markers=segmentation_results.get("markers"),
        # )

        # segmentation_results["qc"] = {
        #     "well": qc_out["well"],
        #     "roi_table": qc_out["roi_table"],
        # }
        # 
        # segmentation_results["instance_labels_qc"] = qc_out["instances_filtered"]

        # rois_dict = extractor(image, segmentation_results["instance_labels_qc"])
        rois_dict = extractor(image, segmentation_results["instance_labels"])
        rois = [ROIResult(**d) for d in rois_dict]

        probs = segmentation_results.get("probs", None)

        results = SegmentationResults(
            instances=segmentation_results["instance_labels"].astype(np.int32),
            cell_mask=segmentation_results["cell_mask"].astype(np.uint8),
            bound_mask=segmentation_results["boundary"].astype(np.uint8),
            probs=probs
        )
        wr = WellResult(
            well_id=well.well_id,
            cfg_hash=cfg_id,
            rois=rois,
            results=results,
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
        "cfg_hash": cfg_id,
        "calib": calib,
        "wells": {wid: wr.summary() for wid, wr in per_well.items()},
    }

