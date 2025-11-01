from __future__ import annotations

import os
import json
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




def validate_unet_sim_tiles(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    mode: Literal["group", "flat"] = "group",
    indices: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Same idea as validate_unet_segmentation, but for *tiled simulated* HDF5.
    HDF5 rows are tiles, not full images.
    """
    # tell segmenter we will feed tiles
    segmenter.cfg.input_is_tiles = True

    ds = DiskValidationDataset(cfg.h5_path, mode=mode)
    if indices is not None:
        ds = torch.utils.data.Subset(ds, indices)  # tiny helper

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

    with tqdm(total=len(ds), desc="Validating UNet (sim-tiles)", unit="tile") as pbar:
        for batch in dl:
            imgs_t, tgts_t, extras = batch  # imgs: [B,3,S,S], tgts: [B,4,S,S]
            B = int(imgs_t.shape[0])

            # run model on the whole batch
            # we already told the segmenter that inputs are tiles,
            # but imgs_t is torch -> we can go straight to device
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
                n_gt = int(inst_gt.max())

                peaks = nms_peaks_np(center_pred,
                                     thr=cfg.center_peak_thr,
                                     min_dist=cfg.center_nms_dist)
                n_centers_pred = int(len(peaks))

                # metrics
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

                # sim tiles DO have n_cells in meta["full"] but we stored per-tile too
                # try tile-level first, then full-level
                if "tile" in meta:  # if you loaded straight from HDF5, meta is already nested
                    # this would be the case if you didn't unwrap it, but in our exporter
                    # we stored plain dict, so usually it's flat
                    tile_meta = meta["tile"]
                    full_meta = meta.get("full", {})
                    n_sim = int(tile_meta.get("n_cells", full_meta.get("n_cells", n_gt)))
                else:
                    # our DiskValidationDataset (flat) gives you the JSON as-is,
                    # so it will have "tile" and "full" on top
                    tile_meta = meta.get("tile", {})
                    full_meta = meta.get("full", {})
                    n_sim = int(tile_meta.get("n_cells", full_meta.get("n_cells", n_gt)))

                row: Dict[str, Any] = {
                    "idx": int(len(per_rows)),
                    "n_cells_simulated": n_sim,
                    "n_cells_gt_instances": n_gt,
                    "n_cells_pred_components_thr0p5": n_cc,
                    "n_cells_pred_centers": n_centers_pred,
                    "count_error_components": int(n_cc - n_gt),
                    "count_error_centers": int(n_centers_pred - n_gt),
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                    "boundary_f1": float(boundary_f1),
                    **{f"center_{k}": v for k, v in center_stats.items()},
                    **{f"energy_{k}": v for k, v in energy_stats.items()},
                }

                # flatten the whole meta (tile + full) so we keep everything
                meta_cols = flatten_meta(meta)
                row.update(meta_cols)

                per_rows.append(row)
                pbar.update(1)

    df = pd.DataFrame(per_rows)
    df["unet_mode"] = segmenter.cfg.unet_mode
    df["dataset_mode"] = "sim_tiles"

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

