from __future__ import annotations

import gc
import os
from typing import Optional, Sequence, Tuple

import pandas as pd

from alloviewer.dev.segmentation import TiledH5Dataset
from alloviewer.dev.segmentation.utils import collate_no_meta
from alloviewer.image_analysis.segmenter import SegmenterUNet

from .config import TrainingValidationConfig
from .utils import (
    validation_make_segmenter_and_config,
    validation_validate_tiled_h5,
)


def validate_unet_segmentation(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "inst_seg",
) -> Tuple[pd.DataFrame, dict]:
    """
    Tile-level validation for saved training/validation H5 datasets.

    Ground-truth policy
    -------------------
    - Stored /inst labels are the source of truth for pixel-level instance metrics.
    - meta['tiles'][tile_idx]['sim_meta']['n_cells'] is kept separately as
      n_cells_gt_meta_tile for grouping and center-in-tile count analyses.
    - GT instances are not reconstructed from target heads.
    """
    return validation_validate_tiled_h5(
        segmenter=segmenter,
        cfg=cfg,
        dataset_cls=TiledH5Dataset,
        collate_fn=collate_no_meta,
        indices=indices,
        segmentation_method=segmentation_method,
        progress_desc=f"Validating UNet ({segmentation_method}, tile-level)",
    )


def run_single_training_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    unet_mode: str,
    dataset_mode: str,
    seg_method: str = "inst_seg",
    force: bool = False,
) -> pd.DataFrame:
    """
    Run one training-validation combination.

    Output:
        training_val_<unet_mode>_<dataset_mode>_<seg_method>.csv
    """
    if dataset_mode not in ("crop_well_resize", "pad_resize", "tiles"):
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    if seg_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown seg_method: {seg_method}")

    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"training_val_{unet_mode}_{dataset_mode}_{seg_method}.csv",
    )
    out_summary_json = os.path.join(
        out_dir,
        f"training_val_{unet_mode}_{dataset_mode}_{seg_method}_summary.json",
    )

    if os.path.isfile(out_csv) and not force:
        return pd.read_csv(out_csv, index_col=None)

    h5_path = os.path.join(h5_dir, f"{dataset_mode}_val.h5")
    model_file = f"best_{unet_mode}_{dataset_mode}_S512_seed187.pth"

    segmenter, cfg = validation_make_segmenter_and_config(
        h5_path=h5_path,
        out_csv=out_csv,
        out_summary_json=out_summary_json,
        unet_mode=unet_mode,
        model_dir=model_dir,
        model_file=model_file,
        seg_method=seg_method,
        validation_config_cls=TrainingValidationConfig,
    )

    df, _ = validate_unet_segmentation(
        segmenter=segmenter,
        cfg=cfg,
        indices=None,
        segmentation_method=seg_method,
    )

    gc.collect()
    return df


def run_training_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    seg_method: str = "inst_seg",
    force: bool = False,
) -> None:
    """
    Sequential fallback.

    For cluster use, prefer run_single_training_validation through a SLURM array.
    """
    results = []

    for unet_mode in ("large", "medium", "small"):
        for dataset_mode in ("crop_well_resize", "pad_resize", "tiles"):
            print(
                f"... Starting calculations for UNet {unet_mode}, "
                f"dataset {dataset_mode}, seg_method {seg_method}"
            )

            df = run_single_training_validation(
                out_dir=out_dir,
                model_dir=model_dir,
                h5_dir=h5_dir,
                unet_mode=unet_mode,
                dataset_mode=dataset_mode,
                seg_method=seg_method,
                force=force,
            )

            results.append(df)
            gc.collect()

    if results:
        final = pd.concat(results, axis=0)
        final.to_csv(
            os.path.join(out_dir, f"training_val_combined_{seg_method}.csv"),
            index=False,
        )

    return

