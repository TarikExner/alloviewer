from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel

from ..segmentation.utils import collate_no_meta
from alloviewer.image_analysis.segmenter import SegmenterUNet, SegmenterConfig, InstanceSegmenterConfig

import torch

from tqdm import tqdm

from ..segmentation import TiledH5Dataset
from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,           
    center_metrics_hungarian,
    energy_metrics_extended_full,
)
from .config import TrainingValidationConfig

import gc

def validate_unet_vs_imagej_on_fullres_h5(
    segmenter: SegmenterUNet,
    gt_segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    imagej_h5_path: str,
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "conventional",  # same options as tiled version
    stop: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate on full-res simulated images with two "prediction" sources:

    1) UNet predictions vs ground-truth targets from cfg.h5_path
    2) ImageJ targets (from imagej_h5_path) vs the same ground-truth targets

    For each tile we compute the same metrics as in validate_unet_on_tiled_h5
    and add two rows to the output DataFrame:

        - dataset_mode = "UNet"   for UNet vs GT
        - dataset_mode = "imageJ" for ImageJ vs GT
    """
    if segmentation_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown segmentation_method: {segmentation_method}")

    if segmentation_method == "inst_seg" and segmenter.inst_seg is None:
        raise ValueError(
            "segmentation_method='inst_seg' requires SegmenterUNet with compute_instances=True"
        )

    # ground-truth simulated data
    ds_gt = TiledH5Dataset(cfg.h5_path, indices=indices)
    # ImageJ-segmented data with identical imgs but different targets
    ds_ij = TiledH5Dataset(imagej_h5_path, indices=indices)

    if len(ds_gt) != len(ds_ij):
        raise RuntimeError(
            f"Ground-truth and ImageJ datasets have different lengths: "
            f"{len(ds_gt)} vs {len(ds_ij)}"
        )

    dl_gt = DataLoader(
        ds_gt,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta
    )
    dl_ij = DataLoader(
        ds_ij,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta
    )

    rows: List[Dict[str, Any]] = []

    with tqdm(total=len(ds_gt), desc="validate fullres GT vs UNet & imageJ", unit="img") as pbar:
        for sample_idx, (batch_gt, batch_ij) in enumerate(zip(dl_gt, dl_ij)):
            if stop is not None and sample_idx == stop:
                break

            imgs_t, tgts_gt_t, extras_gt = batch_gt
            _, tgts_ij_t, extras_ij = batch_ij  # imgs are identical, can ignore here

            # remove batch dim
            imgs_t = imgs_t[0]             # [T,3,H,W]
            tgts_gt_t = tgts_gt_t[0]       # [T,4,H,W]
            tgts_ij_t = tgts_ij_t[0]       # [T,4,H,W]

            meta = extras_gt.get("meta", [{}])[0]

            imgs_np = imgs_t.numpy().astype(np.float32)
            tgts_gt_np = tgts_gt_t.numpy().astype(np.float32)
            tgts_ij_np = tgts_ij_t.numpy().astype(np.float32)
            T, _, H, W = imgs_np.shape

            # run UNet once per sample (all tiles at once)
            out = segmenter(imgs_np)
            cell_prob_unet = out["probs"]["cell"]    # [T,H,W]
            bound_prob_unet = out["probs"]["bound"]
            center_prob_unet = out["probs"]["center"]
            energy_prob_unet = out["probs"]["energy"]

            inst_pred_list = out.get("instance_labels", None)
            if segmentation_method == "inst_seg":
                if inst_pred_list is None:
                    raise RuntimeError(
                        "segmentation_method='inst_seg' but segmenter did not return instance_labels"
                    )
                # inst_pred_list is a list of [H,W] arrays, length T

            # common meta (same for both UNet and imageJ rows)
            full_meta = meta.get("full", {})
            n_cells_from_meta = full_meta.get("n_cells", None)
            frac_positive_from_meta = full_meta.get("frac_positive", None)
            src_path = full_meta.get("src_path", None)

            params = full_meta.get("params", {})
            params_out = {}
            for k, v in params.items():
                col = f"param_{k}"
                if isinstance(v, (int, float, np.floating)):
                    params_out[col] = float(v)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    params_out[col] = json.dumps([float(x) for x in list(v)])
                else:
                    continue

            for t in range(T):
                # ground-truth targets per tile
                tgt_gt = tgts_gt_np[t]      # [4,H,W]
                # ImageJ targets per tile (will be treated as "predictions")
                tgt_ij = tgts_ij_np[t]      # [4,H,W]

                cell_gt = (tgt_gt[0] > 0.5).astype(np.uint8)
                energy_gt = tgt_gt[3].astype(np.float32)

                # build instance labels for ground truth using gt_segmenter
                gt_inst_seg_dict = {
                    "probs": {
                        "cell":   (tgt_gt[0] > gt_segmenter.cfg.thr_cell).astype(np.uint8),
                        "bound":  (tgt_gt[1] > gt_segmenter.cfg.thr_bound).astype(np.uint8),
                        "center": tgt_gt[2],
                        "energy": tgt_gt[3],
                    },
                    "cell_mask": (tgt_gt[0] >= gt_segmenter.cfg.thr_cell).astype(np.uint8),
                    "boundary":  (tgt_gt[1] >= gt_segmenter.cfg.thr_bound).astype(np.uint8),
                    "instance_labels": None,
                    "meta": {},
                }
                inst_seg_dict = gt_segmenter.inst_seg(gt_inst_seg_dict, update_cell_mask=True)
                inst_gt = inst_seg_dict["instance_labels"]
                n_gt = int(inst_gt.max())

                # pick n_cells: meta wins, else GT
                if n_cells_from_meta is not None:
                    n_cells = int(n_cells_from_meta)
                else:
                    n_cells = n_gt

                # ------------------------------------------------------------------
                # 1) UNet vs ground truth
                # ------------------------------------------------------------------
                cell_p_unet = cell_prob_unet[t]
                bound_p_unet = bound_prob_unet[t]
                center_p_unet = center_prob_unet[t]
                energy_p_unet = energy_prob_unet[t]

                if segmentation_method == "conventional":
                    cell_pred_bin_unet = (cell_p_unet >= cfg.cell_thr).astype(np.uint8)
                    n_cc_unet = int(sklabel(cell_pred_bin_unet, connectivity=1).max())
                    n_cells_pred_instances_unet = np.nan
                else:  # "inst_seg"
                    inst_pred = np.asarray(inst_pred_list[t], dtype=np.int32)
                    cell_pred_bin_unet = (inst_pred > 0).astype(np.uint8)
                    n_cells_pred_instances_unet = int(inst_pred.max())
                    n_cc_unet = n_cells_pred_instances_unet

                peaks_unet = nms_peaks_np(
                    center_p_unet,
                    thr=cfg.center_peak_thr,
                    min_dist=cfg.center_nms_dist,
                )
                n_centers_unet = int(len(peaks_unet))

                mask_stats_unet = iou_dice_overlap(cell_pred_bin_unet, cell_gt)

                boundary_f1_unet = boundary_f1_skeletonized(
                    bound_p_unet,
                    inst_gt,
                    tol=cfg.boundary_tol,
                    thr=cfg.boundary_thr,
                    sweep=cfg.boundary_sweep,
                )

                center_stats_unet = center_metrics_hungarian(
                    center_p_unet,
                    inst_gt,
                    peak_thr=cfg.center_peak_thr,
                    nms_dist=cfg.center_nms_dist,
                    match_radius=cfg.center_match_radius,
                    ap_thr_list=cfg.ap_thr_list,
                    oks_thresholds=cfg.oks_thresholds,
                )

                energy_stats_unet = energy_metrics_extended_full(
                    energy_p_unet,
                    energy_gt,
                    cell_gt,
                    frac_delta=cfg.energy_frac_delta,
                )

                row_unet: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "tile_idx": int(t),
                    "n_cells_per_img": n_cells,
                    "frac_positive": (
                        float(frac_positive_from_meta)
                        if frac_positive_from_meta is not None else np.nan
                    ),
                    "n_cells_gt_instances": n_gt,
                    "n_cells_pred_components_thr0p5": n_cc_unet,
                    "n_cells_pred_centers": n_centers_unet,
                    "n_cells_pred_instances": (
                        float(n_cells_pred_instances_unet)
                        if not np.isnan(n_cells_pred_instances_unet) else np.nan
                    ),
                    "count_error_components": int(n_cc_unet - n_gt),
                    "count_error_centers": int(n_centers_unet - n_gt),
                    **{f"mask_{k}": v for k, v in mask_stats_unet.items()},
                    "boundary_f1": float(boundary_f1_unet),
                    **{f"center_{k}": v for k, v in center_stats_unet.items()},
                    **{f"energy_{k}": v for k, v in energy_stats_unet.items()},
                    "src_path": src_path if src_path is not None else "",
                    "segmentation_method": segmentation_method,
                    "dataset_mode": "UNet",
                    **params_out,
                }
                rows.append(row_unet)

                # ------------------------------------------------------------------
                # 2) ImageJ targets vs ground truth
                #    Here we treat tgt_ij as the "prediction".
                # ------------------------------------------------------------------
                cell_p_ij = tgt_ij[0]
                bound_p_ij = tgt_ij[1]
                center_p_ij = tgt_ij[2]
                energy_p_ij = tgt_ij[3]

                # simple threshold on ImageJ cell map; use 0.5 since it is a mask
                cell_pred_bin_ij = (cell_p_ij >= 0.5).astype(np.uint8)
                n_cc_ij = int(sklabel(cell_pred_bin_ij, connectivity=1).max())
                # no real instance segmentation from ImageJ here
                n_cells_pred_instances_ij = np.nan

                ij_inst_seg_dict = {
                    "probs": {
                        "cell":   (cell_p_ij > gt_segmenter.cfg.thr_cell).astype(np.uint8),
                        "bound":  (bound_p_ij > gt_segmenter.cfg.thr_bound).astype(np.uint8),
                        "center": center_p_ij,
                        "energy": energy_p_ij,
                    },
                    "cell_mask": (cell_p_ij >= gt_segmenter.cfg.thr_cell).astype(np.uint8),
                    "boundary":  (bound_p_ij >= gt_segmenter.cfg.thr_bound).astype(np.uint8),
                    "instance_labels": None,
                    "meta": {},
                }
                ij_inst_seg_dict = gt_segmenter.inst_seg(ij_inst_seg_dict, update_cell_mask=True)
                ij_inst = ij_inst_seg_dict["instance_labels"]
                ij_inst_n = int(ij_inst.max())

                peaks_ij = nms_peaks_np(
                    center_p_ij,
                    thr=cfg.center_peak_thr,
                    min_dist=cfg.center_nms_dist,
                )
                n_centers_ij = int(len(peaks_ij))

                mask_stats_ij = iou_dice_overlap(cell_pred_bin_ij, cell_gt)

                boundary_f1_ij = boundary_f1_skeletonized(
                    bound_p_ij,
                    inst_gt,
                    tol=cfg.boundary_tol,
                    thr=cfg.boundary_thr,
                    sweep=cfg.boundary_sweep,
                )

                center_stats_ij = center_metrics_hungarian(
                    center_p_ij,
                    inst_gt,
                    peak_thr=cfg.center_peak_thr,
                    nms_dist=cfg.center_nms_dist,
                    match_radius=cfg.center_match_radius,
                    ap_thr_list=cfg.ap_thr_list,
                    oks_thresholds=cfg.oks_thresholds,
                )

                energy_stats_ij = energy_metrics_extended_full(
                    energy_p_ij,
                    energy_gt,
                    cell_gt,
                    frac_delta=cfg.energy_frac_delta,
                )

                row_ij: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "tile_idx": int(t),
                    "n_cells_per_img": n_cells,
                    "frac_positive": (
                        float(frac_positive_from_meta)
                        if frac_positive_from_meta is not None else np.nan
                    ),
                    "n_cells_gt_instances": n_gt,
                    "n_cells_pred_components_thr0p5": n_cc_ij,
                    "n_cells_pred_centers": n_centers_ij,
                    "n_cells_pred_instances": ij_inst_n if segmentation_method == "inst_seg" else np.nan,
                    "count_error_components": int(n_cc_ij - n_gt),
                    "count_error_centers": int(n_centers_ij - n_gt),
                    **{f"mask_{k}": v for k, v in mask_stats_ij.items()},
                    "boundary_f1": float(boundary_f1_ij),
                    **{f"center_{k}": v for k, v in center_stats_ij.items()},
                    **{f"energy_{k}": v for k, v in energy_stats_ij.items()},
                    "src_path": src_path if src_path is not None else "",
                    # segmentation_method is still "conventional" here
                    "segmentation_method": "conventional",
                    "dataset_mode": "imageJ",
                    **params_out,
                }
                rows.append(row_ij)

            pbar.update(1)

    df = pd.DataFrame(rows)
    df["unet_mode"] = segmenter.cfg.unet_mode

    metric_cols = [
        c for c in df.columns
        if any(c.startswith(pfx) for pfx in ("mask_", "boundary_", "center_", "energy_", "count_error_"))
    ]
    summary = {
        "n_images": int(len(df)),
        "means": {c: float(np.nanmean(df[c].values.astype(np.float64))) for c in metric_cols},
        "stds":  {c: float(np.nanstd(df[c].values.astype(np.float64)))  for c in metric_cols},
    }

    if cfg.out_csv:
        os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
        df.to_csv(cfg.out_csv, index=False)

    if cfg.out_summary_json:
        os.makedirs(os.path.dirname(cfg.out_summary_json) or ".", exist_ok=True)
        with open(cfg.out_summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    return df, summary


def run_fullres_unet_vs_imagej_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    unet_mode: str,
    seg_method: str,
    gt_h5_name: str = "fullres_ground_truth.h5",
    imagej_h5_name: str = "fullres_imageJ.h5",
) -> pd.DataFrame:
    """
    Run validation on full-res simulated images, comparing:

        - UNet vs ground-truth targets (from fullres_ground_truth.h5)
        - ImageJ vs ground-truth targets (fullres_imageJ.h5 vs fullres_ground_truth.h5)

    Rows are labeled with dataset_mode='UNet' or 'imageJ'.
    """
    if seg_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown seg_method: {seg_method}")

    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"testing_val_imageJ_{unet_mode}_{seg_method}.csv"
    )

    # cache
    if os.path.isfile(out_csv):
        df = pd.read_csv(out_csv, index_col=None)
        return df

    gt_h5_path = os.path.join(h5_dir, gt_h5_name)
    imagej_h5_path = os.path.join(h5_dir, imagej_h5_name)

    cfg = TrainingValidationConfig(
        h5_path=gt_h5_path,
        cell_thr=0.1,
        out_csv=out_csv,
    )

    seg_params: Dict[str, Any] = dict(
        unet_mode=unet_mode,
        model_dir=model_dir,
        model_file=f"best_{unet_mode}_tiles_S512_seed187.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=torch.cuda.is_available(),
        normalize=False,  # DiskSimCellsDataset already normalizes
    )

    if seg_method == "conventional":
        segmenter_cfg = SegmenterConfig(
            compute_instances=False,
            **seg_params,
        ).to_dict()
    else:  # "inst_seg"
        segmenter_cfg = SegmenterConfig(
            instance_cfg=InstanceSegmenterConfig().to_dict(),
            compute_instances=True,
            **seg_params,
        ).to_dict()

    gt_segmenter_cfg = SegmenterConfig(
        instance_cfg=InstanceSegmenterConfig().to_dict(),
        compute_instances=True,
        **seg_params,
    ).to_dict()

    gt_segmenter = SegmenterUNet.from_config(gt_segmenter_cfg)
    segmenter = SegmenterUNet.from_config(segmenter_cfg)

    df, _ = validate_unet_vs_imagej_on_fullres_h5(
        segmenter=segmenter,
        gt_segmenter=gt_segmenter,
        cfg=cfg,
        imagej_h5_path=imagej_h5_path,
        segmentation_method=seg_method,
    )
    gc.collect()
    return df
