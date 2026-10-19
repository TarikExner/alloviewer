from __future__ import annotations

import gc
import os
from typing import Optional, Sequence, Tuple

import pandas as pd

from alloviewer.dev.segmentation.utils import collate_no_meta
from alloviewer.image_analysis.segmenter import SegmenterUNet

from ..segmentation import TiledH5Dataset
from .config import TrainingValidationConfig
from .utils import (
    validation_make_segmenter_and_config,
    validation_validate_tiled_h5,
)


def validate_unet_on_tiled_h5(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "inst_seg",
    stop: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Tile-level validation for simulated or external tiled H5 datasets.

    Parameters
    ----------
    segmenter : SegmenterUNet
        Callable UNet instance segmenter used to process the image.
    cfg : TrainingValidationConfig
        Configuration object controlling this component.
    indices : Optional[Sequence[int]]
        Optional subset of dataset record indices. Default is ``None``.
    segmentation_method : str
        Segmentation-method selector used to choose the configured implementation. Default
        is ``'inst_seg'``.
    stop : Optional[int]
        Optional stop indicator checked between validation items. Default is ``None``.

    Returns
    -------
    tuple
        Per-tile validation dataframe and aggregate summary dictionary, in that order.

    Notes
    -----
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
        stop=stop,
    )


def run_single_ext_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    unet_mode: str,
    dataset_mode: str,
    seg_method: str,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run validation for one combination.

    Parameters
    ----------
    out_dir : str
        Directory in which generated outputs are stored.
    model_dir : str
        Directory containing model checkpoint files.
    h5_dir : str
        Directory containing the input HDF5 datasets.
    unet_mode : str
        UNet model size selected by the pipeline.
    dataset_mode : str
        Dataset sampling mode used to select the corresponding training or validation data.
    seg_method : str
        Segmentation-method selector used to choose the configured implementation.
    force : bool
        Whether to recompute results even when the expected output already exists. Default
        is ``False``.

    Returns
    -------
    pandas.DataFrame
        Depending on the input and branch, returns ``pd.read_csv(out_csv, index_col=None)``,
        ``df``.

    Raises
    ------
    ValueError
        Unknown dataset_mode: {dataset_mode} Unknown seg_method: {seg_method}
    FileNotFoundError
        H5 file not found: {h5_path}

    Notes
    -----
    dataset_mode:
        - external_images -> external_images_test.h5
        - tiles           -> tiles_test.h5
    """
    if dataset_mode not in ("external_images", "tiles"):
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    if seg_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown seg_method: {seg_method}")

    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"testing_val_{unet_mode}_{dataset_mode}_{seg_method}.csv",
    )
    out_summary_json = os.path.join(
        out_dir,
        f"testing_val_{unet_mode}_{dataset_mode}_{seg_method}_summary.json",
    )

    if os.path.isfile(out_csv) and not force:
        return pd.read_csv(out_csv, index_col=None)

    h5_path = os.path.join(h5_dir, f"{dataset_mode}_test.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    model_file = f"best_{unet_mode}_tiles_S512_seed187.pth"

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

    df, _ = validate_unet_on_tiled_h5(
        segmenter=segmenter,
        cfg=cfg,
        segmentation_method=seg_method,
    )

    gc.collect()
    return df


def run_test_ext_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    force: bool = False,
) -> None:
    """
    Sequential fallback.

    Parameters
    ----------
    out_dir : str
        Directory in which generated outputs are stored.
    model_dir : str
        Directory containing model checkpoint files.
    h5_dir : str
        Directory containing the input HDF5 datasets.
    force : bool
        Whether to recompute results even when the expected output already exists. Default
        is ``False``.

    Notes
    -----
    For cluster use, prefer the SLURM array script.
    """
    results = []

    for unet_mode in ("large", "medium", "small"):
        for dataset_mode in ("external_images", "tiles"):
            for seg_method in ("conventional", "inst_seg"):
                print(
                    f"... Starting calculations for UNet {unet_mode}, "
                    f"dataset {dataset_mode}, method {seg_method}"
                )

                df = run_single_ext_validation(
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
        final.to_csv(os.path.join(out_dir, "testing_val_combined.csv"), index=False)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--unet-mode", required=True)
    parser.add_argument(
        "--dataset-mode",
        required=True,
        choices=["external_images", "tiles"],
    )
    parser.add_argument(
        "--seg-method",
        required=True,
        choices=["conventional", "inst_seg"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = run_single_ext_validation(
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        h5_dir=args.h5_dir,
        unet_mode=args.unet_mode,
        dataset_mode=args.dataset_mode,
        seg_method=args.seg_method,
        force=args.force,
    )
    print(f"Done. Wrote {len(df)} rows.")

