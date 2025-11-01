from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel
from tqdm import tqdm

from ..segmentation.utils import collate_no_meta
from ..segmenter import SegmenterUNet

from .validation_datasets import DiskValidationDataset

from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    center_metrics_hungarian,
    nms_peaks_np,
    energy_metrics_extended_full,
    flatten_meta
)
from .config import TrainingValidationConfig


def validate_unet_external_tiles(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    mode: Literal["group", "flat"] = "group",
    indices: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Same as above but for external data (HDF5 built from real images).
    These don't have frac_pos; we only have n_cells from the mask.
    """
    segmenter.cfg.input_is_tiles = True

    ds = DiskValidationDataset(cfg.h5_path, mode=mode)
    if indices is not None:
        ds = torch.utils.data.Subset(ds, indices)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta,
    )

    per_rows: List[Dict[str, Any]] = []

    with tqdm(total=len(ds), desc="Validating UNet (external-tiles)", unit="tile") as pbar:
        for batch in dl:
            imgs_t, tgts_t, extras = batch  # [B,3,S,S], [B,4,S,S]
            B = int(imgs_t.shape[0])

            probs = segmenter.predict_tiles(imgs_t)  # [B,4,S,S] on CPU
            probs = probs.numpy().astype(np.float32)

            for b in range(B):
                tgt = tgts_t[b].numpy().astype(np.float32)
                cell_gt = (tgt[0] > 0.5).astype(np.uint8)
                energy_gt = tgt[3].astype(np.float32)

                inst_gt = extras["instance_labels"][b].numpy().astype(np.int32)
                meta = extras["meta"][b]

                cell_prob = probs[b, 0]
                bound_prob = probs[b, 1]
                center_pred = probs[b, 2]
                energy_pred = probs[b, 3]

                cell_pred_bin = (cell_prob >= cfg.cell_thr).astype(np.uint8)
                n_cc = int(sklabel(cell_pred_bin, connectivity=1).max())
                n_gt = int(inst_gt.max())  # from mask

                # centers
                peaks = nms_peaks_np(center_pred,
                                     thr=cfg.center_peak_thr,
                                     min_dist=cfg.center_nms_dist)
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

                # external tiles: we don’t have simulated n_cells, so use GT instances
                n_cells_ext = n_gt

                row: Dict[str, Any] = {
                    "idx": int(len(per_rows)),
                    "n_cells_gt_instances": n_cells_ext,
                    "n_cells_pred_components_thr0p5": n_cc,
                    "n_cells_pred_centers": n_centers_pred,
                    "count_error_components": int(n_cc - n_cells_ext),
                    "count_error_centers": int(n_centers_pred - n_cells_ext),
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                    "boundary_f1": float(boundary_f1),
                    **{f"center_{k}": v for k, v in center_stats.items()},
                    **{f"energy_{k}": v for k, v in energy_stats.items()},
                }

                # flatten meta
                meta_cols = flatten_meta(meta)
                row.update(meta_cols)

                per_rows.append(row)
                pbar.update(1)

    df = pd.DataFrame(per_rows)
    df["unet_mode"] = segmenter.cfg.unet_mode
    df["dataset_mode"] = "external_tiles"

    metric_cols = [
        c for c in df.columns
        if any(c.startswith(pfx) for pfx in ("mask_", "boundary_", "center_", "energy_", "count_error_"))
    ]
    summary = {
        "n_tiles": int(len(df)),
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


