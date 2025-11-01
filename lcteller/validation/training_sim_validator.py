from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel

import matplotlib.pyplot as plt

from ..segmentation.image_dataset import DiskSimCellsDataset
from ..segmentation.utils import collate_no_meta
from ..segmenter import SegmenterUNet

from tqdm import tqdm

from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,           
    center_metrics_hungarian,
    energy_metrics_extended_full,
)
from .config import TrainingValidationConfig


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
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
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
        collate_fn=collate_no_meta
    )

    per_rows: List[Dict[str, Any]] = []

    # progress bar over images
    with tqdm(total=len(ds), desc="Validating UNet (images)", unit="img") as pbar:
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

                # update progress bar per image
                pbar.update(1)

    def _get_dataset_mode(h5_path):
        if "tile" in h5_path:
            return "tiling"
        elif "crop_well" in h5_path:
            return "crop_well_resize"
        elif "pad_resize" in h5_path:
            return "pad_resize"
        else:
            raise ValueError("Unknown DatasetMode")

    df = pd.DataFrame(per_rows)
    df["unet_mode"] = segmenter.cfg.unet_mode
    df["dataset_mode"] = _get_dataset_mode(cfg.h5_path)

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

def visualize_unet_sample(
    segmenter,
    cfg: ValidationConfig,
    idx: int,
    indices: Optional[Sequence[int]] = None,
    figsize=(14, 6),
):
    """
    Show one sample from DiskSimCellsDataset:
    row 1: RGB, GT cell, GT boundary, GT center, GT energy
    row 2: RGB, Pred cell/prob, Pred boundary, Pred center, Pred energy

    All GT/pred pairs share the same color scale.
    """

    # --- load dataset and single sample ---
    ds = DiskSimCellsDataset(cfg.h5_path, indices=indices)
    if idx < 0 or idx >= len(ds):
        raise IndexError(f"idx {idx} out of range for dataset of length {len(ds)}")

    # ds item should match what your DataLoader+collate give you
    img_chw, tgt_chw, extra = ds[idx]  # img: [3,S,S], tgt: [4,S,S]

    # to numpy
    img_chw = img_chw.numpy().astype(np.float32)   # [3,H,W]
    img_hwc = np.transpose(img_chw, (1, 2, 0))     # [H,W,3]

    tgt = tgt_chw.numpy().astype(np.float32)       # [4,H,W]
    cell_gt = (tgt[0] > 0.5).astype(np.float32)    # keep float for viz
    boundary_gt = tgt[1].astype(np.float32)
    center_gt = tgt[2].astype(np.float32)
    energy_gt = tgt[3].astype(np.float32)

    inst_gt = extra["instance_labels"].numpy().astype(np.int32)
    meta = extra["meta"]

    # --- forward pass (use same style as in validate) ---
    out = segmenter(img_hwc)
    cell_prob = out["probs"]["cell"].astype(np.float32)
    bound_prob = out["probs"]["bound"].astype(np.float32)
    center_pred = out["probs"]["center"].astype(np.float32)
    energy_pred = out["probs"]["energy"].astype(np.float32)

    # if you want to also binarize the cell prob using cfg.cell_thr:
    cell_pred_bin = (cell_prob >= cfg.cell_thr).astype(np.float32)

    # --- helper to get common vmin/vmax for GT vs pred ---
    def common_range(a, b):
        # handle empty / constant maps
        vmin = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
        vmax = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
        if vmax == vmin:
            vmax = vmin + 1e-6
        return vmin, vmax

    # ranges
    cell_vmin, cell_vmax = common_range(cell_gt, cell_prob)
    bound_vmin, bound_vmax = common_range(boundary_gt, bound_prob)
    center_vmin, center_vmax = common_range(center_gt, center_pred)
    energy_vmin, energy_vmax = common_range(energy_gt, energy_pred)

    # --- plot ---
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    ax = axes

    # row 1 (GT/simulated)
    ax[0, 0].imshow(img_hwc)
    ax[0, 0].set_title("Input (sim)")
    ax[0, 0].axis("off")

    im1 = ax[0, 1].imshow(cell_gt, vmin=cell_vmin, vmax=cell_vmax, cmap="viridis")
    ax[0, 1].set_title("GT cell")
    ax[0, 1].axis("off")

    im2 = ax[0, 2].imshow(boundary_gt, vmin=bound_vmin, vmax=bound_vmax, cmap="viridis")
    ax[0, 2].set_title("GT boundary")
    ax[0, 2].axis("off")

    im3 = ax[0, 3].imshow(center_gt, vmin=center_vmin, vmax=center_vmax, cmap="viridis")
    ax[0, 3].set_title("GT center")
    ax[0, 3].axis("off")

    im4 = ax[0, 4].imshow(energy_gt, vmin=energy_vmin, vmax=energy_vmax, cmap="viridis")
    ax[0, 4].set_title("GT energy")
    ax[0, 4].axis("off")

    # row 2 (predicted)
    ax[1, 0].imshow(img_hwc)
    ax[1, 0].set_title("Input (dup)")
    ax[1, 0].axis("off")

    # you can switch cell_prob to cell_pred_bin here if you want to show hard mask
    ax[1, 1].imshow(cell_prob, vmin=cell_vmin, vmax=cell_vmax, cmap="viridis")
    ax[1, 1].set_title(f"Pred cell (thr={cfg.cell_thr})")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(bound_prob, vmin=bound_vmin, vmax=bound_vmax, cmap="viridis")
    ax[1, 2].set_title("Pred boundary")
    ax[1, 2].axis("off")

    ax[1, 3].imshow(center_pred, vmin=center_vmin, vmax=center_vmax, cmap="viridis")
    ax[1, 3].set_title("Pred center")
    ax[1, 3].axis("off")

    ax[1, 4].imshow(energy_pred, vmin=energy_vmin, vmax=energy_vmax, cmap="viridis")
    ax[1, 4].set_title("Pred energy")
    ax[1, 4].axis("off")

    plt.tight_layout()
    plt.show()

    # could also return stuff if you want to inspect
    return {
        "img_hwc": img_hwc,
        "gt": {
            "cell": cell_gt,
            "boundary": boundary_gt,
            "center": center_gt,
            "energy": energy_gt,
            "inst": inst_gt,
            "meta": meta,
        },
        "pred": {
            "cell_prob": cell_prob,
            "cell_bin": cell_pred_bin,
            "boundary": bound_prob,
            "center": center_pred,
            "energy": energy_pred,
        },
    }
