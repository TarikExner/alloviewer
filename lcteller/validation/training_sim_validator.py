from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel

from ..segmentation.image_dataset import DiskSimCellsDataset
from ..segmentation.utils import collate_no_meta
from ..segmenter import SegmenterUNet, SegmenterConfig
import gc

from tqdm import tqdm

from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,           
    center_metrics_hungarian,
    energy_metrics_extended_full,
    get_dataset_mode
)
from .config import TrainingValidationConfig

def validate_unet_segmentation(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    indices: Optional[Sequence[int]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs inference using SegmenterUNet on DiskSimCellsDataset and computes metrics.
    Works with datasets that return [B,1,3,H,W] / [B,1,4,H,W].
    """
    ds = DiskSimCellsDataset(cfg.h5_path, indices=indices)
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


    with tqdm(total=len(ds), desc="Validating UNet (images)", unit="img") as pbar:
        for batch in dl:
            imgs_t, tgts_t, extras = batch
            # imgs_t: [B,1,3,H,W] or [B,3,H,W]
            # tgts_t: [B,1,4,H,W] or [B,4,H,W]

            # squeeze middle dim if present
            if imgs_t.ndim == 5 and imgs_t.shape[1] == 1:
                imgs_t = imgs_t[:, 0, ...]          # [B,3,H,W]
            if tgts_t.ndim == 5 and tgts_t.shape[1] == 1:
                tgts_t = tgts_t[:, 0, ...]          # [B,4,H,W]

            B = int(imgs_t.shape[0])

            # instance labels
            inst_all = extras["instance_labels"]
            if inst_all.ndim == 4 and inst_all.shape[1] == 1:
                inst_all = inst_all[:, 0, ...]      # [B,H,W]

            metas = extras["meta"]

            for b in range(B):
                img_chw = imgs_t[b].numpy().astype(np.float32)     # [3,H,W]
                img_hwc = np.transpose(img_chw, (1, 2, 0))         # [H,W,3]

                tgt = tgts_t[b].numpy().astype(np.float32)         # [4,H,W]
                cell_gt = (tgt[0] > 0.5).astype(np.uint8)
                energy_gt = tgt[3].astype(np.float32)

                inst_gt = inst_all[b].numpy().astype(np.int32)
                meta = metas[b]
                full_meta = meta["full"]

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
                peaks = nms_peaks_np(
                    center_pred,
                    thr=cfg.center_peak_thr,
                    min_dist=cfg.center_nms_dist,
                )
                n_centers_pred = int(len(peaks))

                # ---- metrics ----
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

                n_cells = int(full_meta.get("n_cells", n_gt))
                frac_positive = float(full_meta.get("frac_positive", np.nan))
                
                params = meta.get("params", {})
                params_out = {}
                for k, v in params.items():
                    col = f"param_{k}"
                    if isinstance(v, (int, float, np.floating)):
                        params_out[col] = float(v)
                    elif isinstance(v, (list, tuple, np.ndarray)):
                        # keep lists compact and consistent in CSV
                        params_out[col] = json.dumps([float(x) for x in list(v)])
                    else:
                        # ignore unexpected types
                        continue

                row: Dict[str, Any] = {
                    "idx": int(len(per_rows)),
                    "n_cells_simulated": n_cells,
                    "frac_positive": frac_positive,
                    "n_cells_gt_instances": n_gt,
                    "n_cells_pred_components_thr0p5": n_cc,
                    "n_cells_pred_centers": n_centers_pred,
                    "count_error_components": int(n_cc - n_gt),
                    "count_error_centers": int(n_centers_pred - n_gt),
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                    "boundary_f1": float(boundary_f1),
                    **{f"center_{k}": v for k, v in center_stats.items()},
                    **{f"energy_{k}": v for k, v in energy_stats.items()},
                    **params_out
                }

                per_rows.append(row)

                pbar.update(1)

    df = pd.DataFrame(per_rows)
    df["unet_mode"] = segmenter.cfg.unet_mode
    df["dataset_mode"] = get_dataset_mode(cfg.h5_path)

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

def run_training_validation(out_dir: str,
                            model_dir: str,
                            h5_dir: str) -> None:
    res = []
    for unet_mode in ["large", "medium", "small"]:
        for dataset_mode in ["crop_well_resize", "pad_resize", "tiles"]:
            print(f"... Starting calculations for UNet {unet_mode} and dataset {dataset_mode}")

            cfg = TrainingValidationConfig(
                h5_path = os.path.join(h5_dir, f"{dataset_mode}_val.h5"),
                cell_thr = 0.1,
                out_csv = os.path.join(out_dir, f"training_val_{unet_mode}_{dataset_mode}.csv")
            )
            segmenter_cfg = SegmenterConfig(
                unet_mode = unet_mode,
                model_dir = model_dir,
                model_file = f"best_{unet_mode}_{dataset_mode}_S512_seed187.pth",
                device = "cuda" if torch.cuda.is_available() else "cpu",
                use_amp = torch.cuda.is_available()
            ).to_dict()
            segmenter = SegmenterUNet.from_config(segmenter_cfg)
            df, _ = validate_unet_segmentation(segmenter, cfg)
            res.append(df)
            gc.collect()

    final = pd.concat(res, axis = 0)
    final.to_csv(os.path.join(out_dir, "training_val_combined.csv"), index = False)

    return



