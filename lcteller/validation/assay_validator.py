# validate_unet_segmentation.py

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel

from ..segmentation.image_dataset import DiskSimCellsDataset

from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,           
    center_metrics_hungarian,
    energy_metrics_extended_full,
)


@dataclass
class ValidationConfig:
    h5_path: str
    out_csv: Optional[str] = None
    out_summary_json: Optional[str] = None
    batch_size: int = 8
    workers: int = 4
    cell_thr: float = 0.5
    center_peak_thr: float = 0.2
    center_nms_dist: int = 3
    center_match_radius: int = 10
    ap_thr_list: Sequence[float] = tuple(np.linspace(0.05, 0.7, 14))
    oks_thresholds: Sequence[float] = (0.5, 0.75, 0.9)
    boundary_thr: float = 0.9
    boundary_tol: int = 2
    boundary_sweep: bool = False
    energy_frac_delta: float = 0.05  # as fraction of GT range


def _flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten your simulate_image meta into DataFrame-friendly columns.
    - meta['params'] (dict) is expanded under 'param__*'
    - other nested types are JSON-encoded under 'meta__*'
    """
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k == "params" and isinstance(v, dict):
            for pk, pv in v.items():
                out[f"param__{pk}"] = pv
        elif isinstance(v, (dict, list, tuple)):
            out[f"meta__{k}"] = json.dumps(v)
        else:
            out[f"meta__{k}"] = v
    return out


def validate_unet_segmentation(
    segmenter,
    cfg: ValidationConfig,
    indices: Optional[Sequence[int]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs inference using SegmenterUNet on DiskSimCellsDataset and computes metrics.
    Returns (per_image_df, summary_dict).
    """
    ds = DiskSimCellsDataset(cfg.h5_path, indices=indices)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
    )

    per_rows: List[Dict[str, Any]] = []

    for batch in dl:
        imgs_t, tgts_t, extras = batch  # imgs: [B,3,S,S], tgts: [B,4,S,S]
        B = int(imgs_t.shape[0])

        for b in range(B):
            # ---- inputs (numpy) ----
            img_chw = imgs_t[b].numpy().astype(np.float32)        # [3,S,S] in [0,1]
            img_hwc = np.transpose(img_chw, (1, 2, 0))            # [H,W,3]

            tgt = tgts_t[b].numpy().astype(np.float32)            # [4,S,S]
            cell_gt = (tgt[0] > 0.5).astype(np.uint8)
            # bound_prob_gt = tgt[1].astype(np.float32)           # not used directly
            # center_gt_heat = tgt[2].astype(np.float32)          # GT centers heat (not used)
            energy_gt = tgt[3].astype(np.float32)

            inst_gt = extras["instance_labels"][b].numpy().astype(np.int32)
            meta = extras["meta"][b]

            # ---- model forward ----
            out = segmenter(img_hwc)
            cell_prob = out["probs"]["cell"].astype(np.float32)
            bound_prob = out["probs"]["bound"].astype(np.float32)
            center_pred = out["probs"]["center"].astype(np.float32)
            energy_pred = out["probs"]["energy"].astype(np.float32)

            # ---- binarize for components ----
            cell_pred_bin = (cell_prob >= cfg.cell_thr).astype(np.uint8)
            n_cc = int(sklabel(cell_pred_bin, connectivity=1).max())
            n_gt = int(inst_gt.max())

            # ---- centers (pred count via peaks) ----
            peaks = nms_peaks_np(center_pred, thr=cfg.center_peak_thr, min_dist=cfg.center_nms_dist)
            n_centers_pred = int(len(peaks))

            # ---- metrics ----
            # masks
            mask_stats = iou_dice_overlap(cell_pred_bin, cell_gt)

            # boundary F1 vs thin GT boundary
            boundary_f1 = boundary_f1_skeletonized(
                bound_prob,
                inst_gt,
                tol=cfg.boundary_tol,
                thr=cfg.boundary_thr,
                sweep=cfg.boundary_sweep,
            )

            # centers (Hungarian + AP + OKS)
            center_stats = center_metrics_hungarian(
                center_pred,
                inst_gt,
                peak_thr=cfg.center_peak_thr,
                nms_dist=cfg.center_nms_dist,
                match_radius=cfg.center_match_radius,
                ap_thr_list=cfg.ap_thr_list,
                oks_thresholds=cfg.oks_thresholds,
            )

            # energy (inside cells) — full set (includes SSIM & grad corr)
            energy_stats = energy_metrics_extended_full(
                energy_pred,
                energy_gt,
                cell_gt,
                frac_delta=cfg.energy_frac_delta,
            )

            # ---- counts & meta ----
            n_sim = int(meta.get("n_cells", n_gt))  # fallback to GT instances if n_cells missing

            row: Dict[str, Any] = {
                "idx": int(len(per_rows)),
                # counts
                "n_cells_simulated": n_sim,
                "n_cells_gt_instances": n_gt,
                "n_cells_pred_components_thr0p5": n_cc,
                "n_cells_pred_centers": n_centers_pred,
                "count_error_components": int(n_cc - n_gt),
                "count_error_centers": int(n_centers_pred - n_gt),
                # mask metrics
                **{f"mask_{k}": v for k, v in mask_stats.items()},
                # boundary
                "boundary_f1": float(boundary_f1),
                # center metrics
                **{f"center_{k}": v for k, v in center_stats.items()},
                # energy metrics
                **{f"energy_{k}": v for k, v in energy_stats.items()},
            }

            # flatten meta/params to columns
            meta_cols = _flatten_meta(meta)
            row.update(meta_cols)

            per_rows.append(row)

    df = pd.DataFrame(per_rows)

    # ---- dataset-level summary ----
    metric_cols = [
        c for c in df.columns
        if any(c.startswith(pfx) for pfx in ("mask_", "boundary_", "center_", "energy_", "count_error_"))
    ]
    summary = {
        "n_images": int(len(df)),
        "means": {c: float(np.nanmean(df[c].values.astype(np.float64))) for c in metric_cols},
        "stds":  {c: float(np.nanstd(df[c].values.astype(np.float64)))  for c in metric_cols},
    }

    # ---- save outputs ----
    if cfg.out_csv:
        os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
        df.to_csv(cfg.out_csv, index=False)

    if cfg.out_summary_json:
        os.makedirs(os.path.dirname(cfg.out_summary_json) or ".", exist_ok=True)
        with open(cfg.out_summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    return df, summary

