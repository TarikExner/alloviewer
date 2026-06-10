from __future__ import annotations

import os
import json
import gc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel

from alloviewer.dev.segmentation import TiledH5Dataset
from alloviewer.dev.segmentation.utils import collate_no_meta
from alloviewer.image_analysis.segmenter import (
    SegmenterUNet,
    SegmenterConfig,
    InstanceSegmenterConfig,
)

from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,
    center_metrics_hungarian,
    energy_metrics_extended_full,
    get_dataset_mode,
)
from .config import TrainingValidationConfig


def _count_positive_labels(inst: np.ndarray) -> int:
    vals = np.unique(inst)
    vals = vals[vals > 0]
    return int(vals.size)


def _extract_full_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    full_meta = meta.get("full", {})
    if isinstance(full_meta, dict):
        return full_meta
    return {}


def _extract_tile_meta(meta: Dict[str, Any], tile_idx: int) -> Dict[str, Any]:
    tiles = meta.get("tiles", [])
    if isinstance(tiles, list) and tile_idx < len(tiles):
        tm = tiles[tile_idx]
        if isinstance(tm, dict):
            return tm
    return {}


def _extract_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    full_meta = _extract_full_meta(meta)

    params = full_meta.get("params", None)
    if isinstance(params, dict):
        return params

    params = meta.get("params", None)
    if isinstance(params, dict):
        return params

    return {}


def _jsonify_params(params: Dict[str, Any]) -> Dict[str, Any]:
    params_out: Dict[str, Any] = {}

    for k, v in params.items():
        col = f"param_{k}"

        if isinstance(v, (int, float, np.floating, np.integer)):
            params_out[col] = float(v)
        elif isinstance(v, (list, tuple, np.ndarray)):
            try:
                params_out[col] = json.dumps([float(x) for x in list(v)])
            except Exception:
                continue
        else:
            continue

    return params_out


def _get_original_sample_idx(
    loop_idx: int,
    indices: Optional[Sequence[int]],
) -> int:
    if indices is None:
        return int(loop_idx)
    return int(indices[loop_idx])


def _segment_tile_instances(
    out: Dict[str, Any],
    tile_idx: int,
    segmentation_method: str,
    cell_prob: np.ndarray,
    cfg: TrainingValidationConfig,
) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
      cell_pred_bin
      n_components
      n_instances
    """
    if segmentation_method == "conventional":
        cell_pred_bin = (cell_prob >= cfg.cell_thr).astype(np.uint8)
        n_components = int(sklabel(cell_pred_bin, connectivity=1).max())

        # Fill this column so Figure 1B does not break.
        # For conventional mode, instance count is the connected-component count.
        n_instances = n_components

        return cell_pred_bin, n_components, n_instances

    if segmentation_method == "inst_seg":
        inst_list = out.get("instance_labels", None)
        if inst_list is None:
            raise RuntimeError(
                "segmentation_method='inst_seg' but segmenter output has no instance_labels."
            )

        inst_pred = np.asarray(inst_list[tile_idx], dtype=np.int32)
        cell_pred_bin = (inst_pred > 0).astype(np.uint8)
        n_instances = _count_positive_labels(inst_pred)
        n_components = int(sklabel((cell_prob >= cfg.cell_thr).astype(np.uint8), connectivity=1).max())

        return cell_pred_bin, n_components, n_instances

    raise ValueError(f"Unknown segmentation_method: {segmentation_method}")


def validate_unet_segmentation(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "inst_seg",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tile-level validation for simulated training/testing datasets.

    This works for:
      - crop_well_resize
      - pad_resize
      - tiles

    It returns one row per tile.

    Important:
      - Metrics are tile-level.
      - n_cells_gt_instances is the number of unique GT instance labels in that tile.
      - n_cells_pred_instances is always present.
    """
    if segmentation_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown segmentation_method: {segmentation_method}")

    if segmentation_method == "inst_seg" and segmenter.inst_seg is None:
        raise ValueError(
            "segmentation_method='inst_seg' requires SegmenterUNet with compute_instances=True"
        )

    ds = TiledH5Dataset(cfg.h5_path, indices=indices)

    # Keep batch_size=1 because tile count T can vary across samples.
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta,
    )

    per_rows: List[Dict[str, Any]] = []

    with tqdm(
        total=len(ds),
        desc=f"Validating UNet ({segmentation_method}, tile-level)",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for loop_idx, batch in enumerate(dl):
            original_idx = _get_original_sample_idx(loop_idx, indices)

            imgs_t, tgts_t, extras = batch

            # imgs_t: [1,T,3,H,W]
            # tgts_t: [1,T,4,H,W]
            imgs_t = imgs_t[0]
            tgts_t = tgts_t[0]

            inst_all = extras["instance_labels"][0]  # [T,H,W]
            meta = extras["meta"][0]

            tiles_meta = meta.get("tiles", [])
            if isinstance(tiles_meta, list) and len(tiles_meta) > 0:
                T = len(tiles_meta)
            else:
                T = int(imgs_t.shape[0])

            imgs_np = imgs_t[:T].detach().cpu().numpy().astype(np.float32)
            tgts_np = tgts_t[:T].detach().cpu().numpy().astype(np.float32)
            inst_np = inst_all[:T].detach().cpu().numpy().astype(np.int32)

            # Run all tiles from the sample in one call.
            out = segmenter(imgs_np)

            cell_probs = out["probs"]["cell"]
            bound_probs = out["probs"]["bound"]
            center_probs = out["probs"]["center"]
            energy_probs = out["probs"]["energy"]

            full_meta = _extract_full_meta(meta)
            n_cells_full = full_meta.get("n_cells", np.nan)
            frac_positive = full_meta.get("frac_positive", np.nan)
            src_path = full_meta.get("src_path", "")

            params_out = _jsonify_params(_extract_params(meta))

            dataset_mode = get_dataset_mode(cfg.h5_path)

            for t in range(T):
                tgt = tgts_np[t]       # [4,H,W]
                inst_gt = inst_np[t]   # [H,W]

                cell_gt = (tgt[0] > 0.5).astype(np.uint8)
                energy_gt = tgt[3].astype(np.float32)

                cell_prob = np.asarray(cell_probs[t], dtype=np.float32)
                bound_prob = np.asarray(bound_probs[t], dtype=np.float32)
                center_pred = np.asarray(center_probs[t], dtype=np.float32)
                energy_pred = np.asarray(energy_probs[t], dtype=np.float32)

                n_gt = _count_positive_labels(inst_gt)

                (
                    cell_pred_bin,
                    n_cc,
                    n_pred_instances,
                ) = _segment_tile_instances(
                    out=out,
                    tile_idx=t,
                    segmentation_method=segmentation_method,
                    cell_prob=cell_prob,
                    cfg=cfg,
                )

                peaks = nms_peaks_np(
                    center_pred,
                    thr=cfg.center_peak_thr,
                    min_dist=cfg.center_nms_dist,
                )
                n_centers_pred = int(len(peaks))

                mask_stats = iou_dice_overlap(cell_pred_bin, cell_gt)

                boundary_f1 = boundary_f1_skeletonized(
                    bound_prob,
                    inst_gt,
                    tol=cfg.boundary_tol,
                    thr=cfg.boundary_thr,
                    sweep=cfg.boundary_sweep,
                )

                center_stats = center_metrics_hungarian(
                    center_pred,
                    inst_gt,
                    peak_thr=cfg.center_peak_thr,
                    nms_dist=cfg.center_nms_dist,
                    match_radius=cfg.center_match_radius,
                    ap_thr_list=cfg.ap_thr_list,
                    oks_thresholds=cfg.oks_thresholds,
                )

                energy_stats = energy_metrics_extended_full(
                    energy_pred,
                    energy_gt,
                    cell_gt,
                    frac_delta=cfg.energy_frac_delta,
                )

                tile_meta = _extract_tile_meta(meta, t)
                tile_xy = tile_meta.get("tile_xy", (np.nan, np.nan))

                if isinstance(tile_xy, (list, tuple)) and len(tile_xy) == 2:
                    tile_y = tile_xy[0]
                    tile_x = tile_xy[1]
                else:
                    tile_y = np.nan
                    tile_x = np.nan

                row: Dict[str, Any] = {
                    "sample_idx": int(original_idx),
                    "idx": int(original_idx),  # backward-compatible alias
                    "tile_idx": int(t),
                    "metric_level": "tile",
                    "dataset_mode": dataset_mode,
                    "segmentation_method": segmentation_method,
                    "unet_mode": segmenter.cfg.unet_mode,
                    "src_path": src_path if src_path is not None else "",
                    "tile_y": int(tile_y) if not pd.isna(tile_y) else np.nan,
                    "tile_x": int(tile_x) if not pd.isna(tile_x) else np.nan,
                    "n_cells_per_img": (
                        int(n_cells_full)
                        if isinstance(n_cells_full, (int, np.integer))
                        else float(n_cells_full)
                        if isinstance(n_cells_full, (float, np.floating))
                        else np.nan
                    ),
                    "frac_positive": (
                        float(frac_positive)
                        if isinstance(frac_positive, (int, float, np.integer, np.floating))
                        else np.nan
                    ),
                    "n_cells_gt_instances": int(n_gt),
                    "n_cells_pred_components_thr0p5": int(n_cc),
                    "n_cells_pred_centers": int(n_centers_pred),
                    "n_cells_pred_instances": int(n_pred_instances),
                    "count_error_components": int(n_cc - n_gt),
                    "count_error_centers": int(n_centers_pred - n_gt),
                    "count_error_instances": int(n_pred_instances - n_gt),
                    "abs_count_error_instances": int(abs(n_pred_instances - n_gt)),
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                    "boundary_f1": float(boundary_f1),
                    **{f"center_{k}": v for k, v in center_stats.items()},
                    **{f"energy_{k}": v for k, v in energy_stats.items()},
                    **params_out,
                }

                per_rows.append(row)

            pbar.update(1)

    df = pd.DataFrame(per_rows)

    metric_cols = [
        c for c in df.columns
        if any(
            c.startswith(pfx)
            for pfx in (
                "mask_",
                "boundary_",
                "center_",
                "energy_",
                "count_error_",
                "abs_count_error_",
            )
        )
    ]

    summary = {
        "n_images": int(df["sample_idx"].nunique()) if len(df) else 0,
        "n_tiles": int(len(df)),
        "means": {
            c: float(np.nanmean(df[c].values.astype(np.float64)))
            for c in metric_cols
        },
        "stds": {
            c: float(np.nanstd(df[c].values.astype(np.float64)))
            for c in metric_cols
        },
    }

    if cfg.out_csv:
        os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
        df.to_csv(cfg.out_csv, index=False)

    if cfg.out_summary_json:
        os.makedirs(os.path.dirname(cfg.out_summary_json) or ".", exist_ok=True)
        with open(cfg.out_summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    return df, summary


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

    cfg = TrainingValidationConfig(
        h5_path=h5_path,
        cell_thr=0.1,
        batch_size=1,
        out_csv=out_csv,
        out_summary_json=out_summary_json,
    )

    seg_params: Dict[str, Any] = dict(
        unet_mode=unet_mode,
        model_dir=model_dir,
        model_file=model_file,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=torch.cuda.is_available(),
        normalize=False,  # TiledH5Dataset/H5CellsDataset already normalizes
    )

    if seg_method == "conventional":
        segmenter_cfg = SegmenterConfig(
            compute_instances=False,
            **seg_params,
        ).to_dict()
    else:
        segmenter_cfg = SegmenterConfig(
            instance_cfg=InstanceSegmenterConfig().to_dict(),
            compute_instances=True,
            **seg_params,
        ).to_dict()

    segmenter = SegmenterUNet.from_config(segmenter_cfg)

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
    res = []

    for unet_mode in ["large", "medium", "small"]:
        for dataset_mode in ["crop_well_resize", "pad_resize", "tiles"]:
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

            res.append(df)
            gc.collect()

    if len(res) > 0:
        final = pd.concat(res, axis=0)
        final.to_csv(
            os.path.join(out_dir, f"training_val_combined_{seg_method}.csv"),
            index=False,
        )

    return
