from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel

from ..segmentation.utils import collate_no_meta
from ..segmenter import SegmenterUNet, SegmenterConfig, InstanceSegmenterConfig

import torch

from tqdm import tqdm

from ..segmentation import TiledH5Dataset
from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,           
    center_metrics_hungarian,
    energy_metrics_extended_full,
    get_dataset_mode
)
from .config import TrainingValidationConfig

import gc

def validate_unet_on_tiled_h5(
    segmenter: SegmenterUNet,
    gt_segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,                    # your validation cfg (has cell_thr, boundary_thr, etc.)
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "conventional",        # "conventional" | "inst_seg"
    stop: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the UNet on every sample in a tiled H5 (sim or external),
    using meta["tiles"] to know how many tiles to keep.

    For each tile we compute the same metrics you already use.

    segmentation_method:
        - "conventional": use simple thresholding on P(cell)
        - "inst_seg":     use the InstanceSegmenter output (instances > 0)
    """
    if segmentation_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown segmentation_method: {segmentation_method}")

    if segmentation_method == "inst_seg" and segmenter.inst_seg is None:
        raise ValueError(
            "segmentation_method='inst_seg' requires SegmenterUNet with compute_instances=True"
        )

    ds = TiledH5Dataset(cfg.h5_path, indices=indices)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta
    )

    rows: List[Dict[str, Any]] = []

    with tqdm(total=len(ds), desc="validate tiled h5", unit="img") as pbar:
        for sample_idx, batch in enumerate(dl):
            if sample_idx == stop:
                break
            imgs_t, tgts_t, extras = batch
            # remove batch dim
            imgs_t = imgs_t[0]                    # [T,3,S,S]
            tgts_t = tgts_t[0]                    # [T,4,S,S]
            inst_t = extras["instance_labels"][0] # [T,S,S]
            meta = extras["meta"][0]

            imgs_np = imgs_t.numpy().astype(np.float32)
            tgts_np = tgts_t.numpy().astype(np.float32)
            inst_np = inst_t.numpy().astype(np.int32)
            T, _, H, W = imgs_np.shape

            # run all tiles in one go
            out = segmenter(imgs_np)
            cell_prob = out["probs"]["cell"]    # [T,H,W]
            bound_prob = out["probs"]["bound"]
            center_pred = out["probs"]["center"]
            energy_pred = out["probs"]["energy"]


            # optional instance predictions (only present if compute_instances=True)
            inst_pred_list = out.get("instance_labels", None)
            if segmentation_method == "inst_seg":
                if inst_pred_list is None:
                    raise RuntimeError(
                        "segmentation_method='inst_seg' but segmenter did not return instance_labels"
                    )
                # inst_pred_list is a list of [H,W] arrays, length T

            # figure out what kind of source this is
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
                tgt = tgts_np[t]           # [4,H,W]
                cell_gt = (tgt[0] > 0.5).astype(np.uint8)
                energy_gt = tgt[3].astype(np.float32)
                inst_gt = inst_np[t]

                cell_p = cell_prob[t]
                bound_p = bound_prob[t]
                center_p = center_pred[t]
                energy_p = energy_pred[t]

                gt_inst_seg_dict = {
                    "probs": {
                        "cell":   (tgt[0]>gt_segmenter.cfg.thr_cell).astype(np.uint8),
                        "bound":  (tgt[1]>gt_segmenter.cfg.thr_bound).astype(np.uint8),
                        "center": (tgt[2]),
                        "energy": (tgt[3]),
                    },
                    "cell_mask": (tgt[0]>= segmenter.cfg.thr_cell).astype(np.uint8),
                    "boundary":  (tgt[1]>= segmenter.cfg.thr_bound).astype(np.uint8),
                    "instance_labels": None,
                    "meta": {},

                }
                inst_seg_dict = gt_segmenter.inst_seg(gt_inst_seg_dict, update_cell_mask = True)
                instances = inst_seg_dict["instance_labels"]
                n_cells_tile = instances.max()

                # --- choose prediction mask / counts depending on method ---
                if segmentation_method == "conventional":
                    # simple threshold on P(cell)
                    cell_pred_bin = (cell_p >= cfg.cell_thr).astype(np.uint8)
                    # connected components on binary mask
                    n_cc = int(sklabel(cell_pred_bin, connectivity=1).max())
                    n_cells_pred_instances = np.nan  # no true instance output here
                else:  # "inst_seg"
                    inst_pred = np.asarray(inst_pred_list[t], dtype=np.int32)
                    # union of predicted instances as foreground mask
                    cell_pred_bin = (inst_pred > 0).astype(np.uint8)
                    # count instances directly
                    n_cells_pred_instances = int(inst_pred.max())
                    # for compatibility, use instance count as "components" count
                    n_cc = n_cells_pred_instances

                n_gt = int(inst_gt.max())

                # centers (count)
                peaks = nms_peaks_np(
                    center_p,
                    thr=cfg.center_peak_thr,
                    min_dist=cfg.center_nms_dist,
                )
                n_centers_pred = int(len(peaks))

                # mask metrics
                mask_stats = iou_dice_overlap(cell_pred_bin, cell_gt)

                # boundary
                boundary_f1 = boundary_f1_skeletonized(
                    bound_p,
                    inst_gt,
                    tol=cfg.boundary_tol,
                    thr=cfg.boundary_thr,
                    sweep=cfg.boundary_sweep,
                )

                # centers (Hungarian + AP)
                center_stats = center_metrics_hungarian(
                    center_p,
                    inst_gt,
                    peak_thr=cfg.center_peak_thr,
                    nms_dist=cfg.center_nms_dist,
                    match_radius=cfg.center_match_radius,
                    ap_thr_list=cfg.ap_thr_list,
                    oks_thresholds=cfg.oks_thresholds,
                )

                # energy
                energy_stats = energy_metrics_extended_full(
                    energy_p,
                    energy_gt,
                    cell_gt,
                    frac_delta=cfg.energy_frac_delta,
                )

                # pick n_cells: meta wins, else GT
                if n_cells_from_meta is not None:
                    n_cells = int(n_cells_from_meta)
                else:
                    n_cells = n_gt

                row: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "tile_idx": int(t),
                    "n_cells": n_cells_tile,
                    "n_cells_per_img": n_cells,
                    "frac_positive": (
                        float(frac_positive_from_meta)
                        if frac_positive_from_meta is not None else np.nan
                    ),
                    "n_cells_gt_instances": n_gt,
                    "n_cells_pred_components_thr0p5": n_cc,
                    "n_cells_pred_centers": n_centers_pred,
                    "n_cells_pred_instances": (
                        float(n_cells_pred_instances)
                        if not np.isnan(n_cells_pred_instances) else np.nan
                    ),
                    "count_error_components": int(n_cc - n_gt),
                    "count_error_centers": int(n_centers_pred - n_gt),
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                    "boundary_f1": float(boundary_f1),
                    **{f"center_{k}": v for k, v in center_stats.items()},
                    **{f"energy_{k}": v for k, v in energy_stats.items()},
                    "src_path": src_path if src_path is not None else "",
                    "segmentation_method": segmentation_method,
                    **params_out
                }

                rows.append(row)

            pbar.update(1)

    df = pd.DataFrame(rows)
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

def run_single_ext_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    unet_mode: str,
    dataset_mode: str,
    seg_method: str,
) -> pd.DataFrame:
    """
    Run validation for a single combination of:
        - unet_mode       in {"large","medium","small", ...}
        - dataset_mode    e.g. "external_images" or "tiles"
        - seg_method      in {"conventional", "inst_seg"}

    Writes a CSV for this combo and returns the DataFrame.
    """
    if seg_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown seg_method: {seg_method}")

    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"testing_val_{unet_mode}_{dataset_mode}_{seg_method}.csv"
    )

    # cache: if CSV already exists, just read and return
    if os.path.isfile(out_csv):
        df = pd.read_csv(out_csv, index_col=None)
        return df

    h5_path = os.path.join(h5_dir, f"{dataset_mode}_test.h5")

    cfg = TrainingValidationConfig(
        h5_path=h5_path,
        cell_thr=0.1,
        out_csv=out_csv,
    )

    seg_params: Dict[str, Any] = dict(
        unet_mode=unet_mode,
        model_dir=model_dir,
        model_file=f"best_{unet_mode}_tiles_S512_seed187.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=torch.cuda.is_available(),
        normalize=False #DiskSimCellsDataset already normalizes
    )

    if seg_method == "conventional":
        # if you want to skip instance segmentation for speed:
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

    # ground truth labels for external images need to be segmented properly
    gt_segmenter_cfg = SegmenterConfig(
        instance_cfg=InstanceSegmenterConfig().to_dict(),
        compute_instances=True,
        **seg_params,
    ).to_dict()


    gt_segmenter = SegmenterUNet.from_config(gt_segmenter_cfg)

    segmenter = SegmenterUNet.from_config(segmenter_cfg)

    df, _ = validate_unet_on_tiled_h5(
        segmenter,
        gt_segmenter,
        cfg,
        segmentation_method=seg_method,
    )
    gc.collect()

    return df


def run_test_ext_validation(out_dir: str,
                            model_dir: str,
                            h5_dir: str) -> None:
    res = []
    for unet_mode in ["large", "medium", "small"]:
        for dataset_mode in ["external_images", "tiles"]:
            for seg_method in ["conventional", "inst_seg"]:
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
                )
                res.append(df)

    final = pd.concat(res, axis=0)
    final.to_csv(os.path.join(out_dir, "testing_val_combined.csv"), index=False)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--unet-mode", required=True)
    parser.add_argument("--dataset-mode", required=True)
    parser.add_argument(
        "--seg-method",
        required=True,
        choices=["conventional", "inst_seg"],
    )
    args = parser.parse_args()

    df = run_single_ext_validation(
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        h5_dir=args.h5_dir,
        unet_mode=args.unet_mode,
        dataset_mode=args.dataset_mode,
        seg_method=args.seg_method,
    )
    print(f"Done. Wrote {len(df)} rows.")

