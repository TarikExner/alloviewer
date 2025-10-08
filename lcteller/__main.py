import numpy as np
from . import simulate_image

from . import Plate, DummyByOrderSegmenter, WellImage

from . import RGBExtractor, PCNCMedianCalibrator, ROIClassifier, ROIClassifierGaussian3Way, PCNCGaussianRGCalibrator
from . import PCNCMeanCalibrator

from typing import Tuple
from .models import WellResult, SegmentationResults, ROIResult
from .contracts import ISegmenter, IFeatureExtractor, ICalibrator
from pathlib import Path

import json
import hashlib

from typing import Dict, Any, Optional, List

def get_images_from_user():
    kwargs = dict(
        cell_diameter = 13,
        n_cells = 300,
    )
    images = []
    instance_labels = []
    cell_masks = []
    boundaries = []

    # PC
    pc_img, _, pc_targets = simulate_image(frac_positive = 1, **kwargs)
    images.append(pc_img)
    instance_labels.append(pc_targets["instance_labels"])
    cell_masks.append(pc_targets["cell_mask"])
    boundaries.append(pc_targets["boundary"])

    # NC
    nc_img, _, nc_targets = simulate_image(frac_positive = 0, **kwargs)
    images.append(nc_img)
    instance_labels.append(nc_targets["instance_labels"])
    cell_masks.append(nc_targets["cell_mask"])
    boundaries.append(nc_targets["boundary"])

    for i in range(3):
        frac_positive = (0.15 * (i+1))
        n_positive = frac_positive*kwargs["n_cells"]
        print(n_positive)
        img, _, targets = simulate_image(frac_positive = (0.15* (i+1)), **kwargs)
        images.append(img)
        instance_labels.append(targets["instance_labels"])
        cell_masks.append(targets["cell_mask"])
        boundaries.append(targets["boundary"])

    return images, instance_labels, cell_masks, boundaries


def get_image_order_from_user():
    return ["A1", "A2", "A3", "A4", "A5"]

def get_assignments_from_user():
    return ["PC", "NC", "SA01", "SA02", "SA03"]

def build_plate_from_user() -> Tuple[Plate, DummyByOrderSegmenter]:
    images, insts, cells, bounds = get_images_from_user()
    order = get_image_order_from_user()
    roles = get_assignments_from_user()

    assert len(images) == len(order) == len(roles) == len(cells) == len(bounds), "length mismatch"

    plate = Plate(plate_id="SIM001")
    for i, (well_id, role) in enumerate(zip(order, roles)):
        plate.add(WellImage(well_id, role=role, image=images[i]))

    seg = DummyByOrderSegmenter(
        cell_masks=cells,
        bound_masks=bounds,
        instance_labels=insts,
        min_size=16,
    )
    return plate, seg

def _cfg_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:10]


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_plate(
    plate: Plate,
    segmenter: ISegmenter,
    extractor: IFeatureExtractor,
    calibrator: ICalibrator,
    classifier_ctor,                          # a class/callable taking calib -> classifier
    cfg: Dict[str, Any],
    *,
    save_dir: Optional[str | Path] = None,    # where to write npz/previews; None = no writes
    save_probs: bool = False,                 # probs can be large; keep False unless needed
    preview_size: int = 512,                  # overlay PNG size
) -> Dict[str, Any]:
    """
    Runs segmentation -> features -> calibration (PC/NC) -> classification.
    Returns a dict with cfg hash, calibration, and per-well summaries.
    """
    cfg_id = _cfg_hash(cfg)
    out_dir = _ensure_dir(save_dir) if save_dir else None

    per_well: dict[str, WellResult] = {}

    # 1) segment + extract
    for w in plate.get():
        # load image
        if w.image is not None:
            img = w.image
        else:
            raise ValueError(f"No image or path for well {w.well_id}")

        seg = segmenter(img)  # expects dict with 'instances','cell_mask','bound_mask','probs'(opt)
        rois_dicts = extractor(img, seg["instances"])
        rois = [ROIResult(**d) for d in rois_dicts]

        probs = seg.get("probs", None)
        if (probs is not None) and (not save_probs):
            probs = None

        results = SegmentationResults(
            instances=seg["instances"].astype(np.int32),
            cell_mask=seg["cell_mask"].astype(np.uint8),
            bound_mask=seg["bound_mask"].astype(np.uint8),
            probs=(probs.astype(np.float32) if probs is not None else None),
        )

        wr = WellResult(
            well_id=w.well_id,
            cfg_hash=cfg_id,
            rois=rois,
            results=results,
            qc=seg.get("qc", {}),
        )

        # optional: persist arrays + preview
        if out_dir:
            base = f"{w.well_id}_{cfg_id}"

            np.savez_compressed(
                out_dir / f"{base}_seg.npz",
                instances=results.instances,
                cell_mask=results.cell_mask,
                bound_mask=results.bound_mask,
                probs=(results.probs if results.probs is not None else np.array([])),
            )
            wr.store_paths["seg"] = str(out_dir / f"{base}_seg.npz")

            try:
                _ = 1/0
            except Exception as e:
                wr.qc["preview_error"] = str(e)

        per_well[w.well_id] = wr

    # 2) calibration from PC/NC wells
    pc = [per_well[w.well_id].rois for w in plate.get("PC")]
    nc = [per_well[w.well_id].rois for w in plate.get("NC")]
    # pass lists of ROI dicts to the calibrator
    calib = calibrator.fit(
        pc_wells=[[r.__dict__ for r in rs] for rs in pc],
        nc_wells=[[r.__dict__ for r in rs] for rs in nc],
    )
    print(calib)

    # 3) classify all wells
    clf = classifier_ctor(calib)
    for wr in per_well.values():
        updated = clf([r.__dict__ for r in wr.rois])  # list[dict] with label/score
        wr.rois = [ROIResult(**d) for d in updated]

    # 4) final light summary for UI/API
    return {
        "cfg_hash": cfg_id,
        "calib": calib,
        "wells": {wid: wr.summary() for wid, wr in per_well.items()},
    }


def create_plate():
    plate, segmenter = build_plate_from_user()
    return plate

def job():
    plate, segmenter = build_plate_from_user()
    extractor = RGBExtractor()
    calibrator = PCNCMedianCalibrator()
    classifier_ctor = ROIClassifier

    cfg = {"seg": "dummy_by_order", "feat": "rgb", "calib": "median_v1"}

    result = run_plate(
        plate=plate,
        segmenter=segmenter,
        extractor=extractor,
        calibrator=calibrator,
        classifier_ctor=classifier_ctor,
        cfg=cfg,
        save_dir=None,       # set to None to skip disk writes
        preview_size=512
    )
    return result

def DEVELOPMENT(images: list[np.ndarray]):

    print(images[0].shape)

