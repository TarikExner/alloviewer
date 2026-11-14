from __future__ import annotations

"""Validate UNet checkpoints on saved simulated validation H5 tiles.

Drop-in replacement for alloviewer/dev/validation/training_sim_validator.py.

The existing training_val_<mode>_<dataset>_<method>.csv filenames and
legacy metrics are preserved. Adds pixel TP/TN/FP/FN; one-to-one cell TP/FP/FN
at IoU thresholds 0.10/0.25/0.30/0.50/0.70/0.75/0.90 and center radii 4/6/8 px; precision,
recall, F1; mean matched-cell Dice/Jaccard, with missed-cell-zero versions;
localization error; center-in-predicted-instance F1; and split/merge indicators. Every row evaluates one tile.

Ground truth is the stored H5 /inst instance-label map. It is *not* inferred
from UNet target heads. Metadata 'n_cells' is kept separately and must not be
used as the source of object TP/FP/FN.

This script evaluates SIMULATED validation data. It does not, by itself,
address a reviewer's requirement for independently annotated REAL images.

Example:
  from alloviewer.dev.validation.training_sim_validator import run_single_training_validation
  df = run_single_training_validation(
      out_dir='./results', model_dir='./alloviewer/models', h5_dir='./data',
      unet_mode='small', dataset_mode='tiles', seg_method='inst_seg', force=True)
"""

import gc
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from skimage.measure import label as sk_label
from torch.utils.data import DataLoader
from tqdm import tqdm

from alloviewer.dev.segmentation import TiledH5Dataset
from alloviewer.dev.segmentation.utils import collate_no_meta
from alloviewer.image_analysis.segmenter import SegmenterUNet
from .config import TrainingValidationConfig
from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    center_metrics_hungarian,
    energy_metrics_extended_full,
    nms_peaks_np,
    validation_make_segmenter_and_config,
    validation_extract_full_meta,
    validation_extract_tile_meta,
    validation_extract_tile_xy,
    validation_extract_tile_n_cells_meta,
    validation_extract_tile_frac_positive,
    validation_extract_params,
    validation_jsonify_params,
    validation_as_float_or_nan,
    validation_as_int_or_nan,
    validation_locally_relabel_instances,
    validation_count_positive_labels,
    validation_prediction_masks_and_counts,
    validation_count_error_vs_meta,
    validation_summary_from_dataframe,
    get_dataset_mode,
    VALIDATION_VERSION_STORED_INST_GT,
)
# Shared measurement definitions for *all simulated comparisons*. Importing
# this function does not load Cellpose or construct any neural network.
from .simulated_segmentation_comparison import compare_simulated_instances


def validate_unet_segmentation(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "inst_seg",
) -> Tuple[pd.DataFrame, dict]:
    """Validate one H5 set. Existing metric columns plus instance metrics.

    Predictions come from exactly one UNet call per image. The legacy head
    metrics (cell/boundary/center/energy) remain separate from instance metrics.
    For conventional mode, connected components of cell_prob >= cell_thr
    constitute the predicted instances. This convention is recorded explicitly.
    """
    if segmentation_method not in {"inst_seg", "conventional"}:
        raise ValueError(f"Unknown segmentation_method {segmentation_method!r}")
    if segmentation_method == "inst_seg" and segmenter.inst_seg is None:
        raise ValueError("inst_seg requires compute_instances=True")

    dataset = TiledH5Dataset(cfg.h5_path, indices=indices)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        collate_fn=collate_no_meta,
    )
    rows: list[dict[str, Any]] = []
    dataset_mode = get_dataset_mode(cfg.h5_path)
    try:
        for loop_idx, batch in enumerate(tqdm(loader, desc=f"UNet {segmentation_method}", unit="image")):
            sample_idx = loop_idx if indices is None else int(indices[loop_idx])
            image_batch, target_batch, extras = batch
            image_batch = image_batch[0]
            target_batch = target_batch[0]
            stored_instances = extras["instance_labels"][0]
            meta = extras["meta"][0]
            tile_metas = meta.get("tiles") or []
            n_tiles = len(tile_metas) if tile_metas else int(image_batch.shape[0])
            inputs = image_batch[:n_tiles].detach().cpu().numpy().astype(np.float32)
            targets = target_batch[:n_tiles].detach().cpu().numpy().astype(np.float32)
            stored = stored_instances[:n_tiles].detach().cpu().numpy().astype(np.int32)

            with torch.inference_mode():
                pred = segmenter(inputs)
            probs = pred["probs"]
            full_meta = validation_extract_full_meta(meta)
            params = validation_jsonify_params(validation_extract_params(meta))
            for t in range(n_tiles):
                target = targets[t]
                gt_stored = stored[t]
                gt_local = validation_locally_relabel_instances(gt_stored)
                gt_count = validation_count_positive_labels(gt_stored)

                p_cell = np.asarray(probs["cell"][t], dtype=np.float32)
                p_bound = np.asarray(probs["bound"][t], dtype=np.float32)
                p_center = np.asarray(probs["center"][t], dtype=np.float32)
                p_energy = np.asarray(probs["energy"][t], dtype=np.float32)
                gt_cell_head = (target[0] > 0.5).astype(np.uint8)
                gt_energy = np.asarray(target[3], dtype=np.float32)
                pred_binary, n_components, n_pred = validation_prediction_masks_and_counts(
                    out=pred, tile_idx=t, segmentation_method=segmentation_method,
                    cell_prob=p_cell, cfg=cfg,
                )
                if segmentation_method == "inst_seg":
                    predicted_instances = np.asarray(pred["instance_labels"][t], dtype=np.int32)
                else:
                    predicted_instances = sk_label(pred_binary > 0, connectivity=1).astype(np.int32)
                # These new metrics use stored instance GT for ALL methods.
                instance_metrics = compare_simulated_instances(gt_local, predicted_instances,
                                                                boundary_tolerance=cfg.boundary_tol)
                mask_stats = iou_dice_overlap(pred_binary, gt_cell_head)
                boundary_f1 = boundary_f1_skeletonized(
                    p_bound, gt_local, tol=cfg.boundary_tol,
                    thr=cfg.boundary_thr, sweep=cfg.boundary_sweep,
                )
                legacy_center = center_metrics_hungarian(
                    p_center, gt_local, peak_thr=cfg.center_peak_thr,
                    nms_dist=cfg.center_nms_dist,
                    match_radius=cfg.center_match_radius,
                    ap_thr_list=cfg.ap_thr_list,
                    oks_thresholds=cfg.oks_thresholds,
                )
                legacy_energy = energy_metrics_extended_full(
                    p_energy, gt_energy, gt_cell_head,
                    frac_delta=cfg.energy_frac_delta,
                )
                n_centers = len(nms_peaks_np(
                    p_center, thr=cfg.center_peak_thr, min_dist=cfg.center_nms_dist,
                ))
                tile_meta = validation_extract_tile_meta(meta, t)
                tile_y, tile_x = validation_extract_tile_xy(tile_meta)
                meta_count = validation_extract_tile_n_cells_meta(tile_meta)
                errors = {
                    "components": n_components - gt_count,
                    "centers": n_centers - gt_count,
                    "instances": n_pred - gt_count,
                }
                row: dict[str, Any] = {
                    "validation_version": VALIDATION_VERSION_STORED_INST_GT + "+cell_level_v4",
                    "sample_idx": sample_idx,
                    "tile_idx": t,
                    "metric_level": "tile",
                    "level": "tile",
                    "dataset_mode": dataset_mode,
                    "segmentation_method": segmentation_method,
                    "method": "AlloViewer_UNet",
                    "instance_definition": (
                        "UNet_instance_segmenter" if segmentation_method == "inst_seg"
                        else "connected_components_of_cell_probability_threshold"
                    ),
                    "unet_mode": segmenter.cfg.unet_mode,
                    "src_path": full_meta.get("src_path", "") or "",
                    "tile_y": int(tile_y) if np.isfinite(tile_y) else np.nan,
                    "tile_x": int(tile_x) if np.isfinite(tile_x) else np.nan,
                    "n_cells_per_img": validation_as_int_or_nan(full_meta.get("n_cells")),
                    "n_cells_gt_meta_tile": validation_as_int_or_nan(meta_count),
                    "n_cells_gt_instances_stored_labels": gt_count,
                    "n_cells_gt_instances": gt_count,
                    "frac_positive_per_img": validation_as_float_or_nan(full_meta.get("frac_positive")),
                    "frac_positive_tile": validation_extract_tile_frac_positive(tile_meta),
                    "n_cells_pred_components_cell_thr": n_components,
                    "cell_component_thr": cfg.cell_thr,
                    "n_cells_pred_centers": n_centers,
                    "n_cells_pred_instances": n_pred,
                    "count_error_components_vs_stored_labels": errors["components"],
                    "count_error_centers_vs_stored_labels": errors["centers"],
                    "count_error_instances_vs_stored_labels": errors["instances"],
                    "abs_count_error_components_vs_stored_labels": abs(errors["components"]),
                    "abs_count_error_centers_vs_stored_labels": abs(errors["centers"]),
                    "abs_count_error_instances_vs_stored_labels": abs(errors["instances"]),
                    "mask_iou": mask_stats["iou"],
                    "mask_dice": mask_stats["dice"],
                    "mask_overlap": mask_stats["overlap"],
                    "boundary_f1": boundary_f1,
                    **{f"center_{k}": v for k, v in legacy_center.items()},
                    **{f"energy_{k}": v for k, v in legacy_energy.items()},
                    **params,
                }
                for what, n_predicted in (("components", n_components),
                                          ("centers", n_centers),
                                          ("instances", n_pred)):
                    diff = validation_count_error_vs_meta(n_predicted, meta_count)
                    row[f"count_error_{what}_vs_meta_tile"] = diff
                    row[f"abs_count_error_{what}_vs_meta_tile"] = abs(diff) if np.isfinite(diff) else np.nan
                # Keep legacy 'center_*' head scores and add geometric center
                # scores with *different names*: center_f1_r6, center_tp_r6.
                row.update(instance_metrics)
                rows.append(row)
    finally:
        dataset.close()

    result = pd.DataFrame(rows)
    summary = validation_summary_from_dataframe(result)
    summary["validation_version"] = VALIDATION_VERSION_STORED_INST_GT + "+cell_level_v4"
    summary["notes"] = [
        "Metrics use simulator-stored H5 /inst reference labels.",
        "Cell-level TN is undefined; pixel_tn is background correctly predicted.",
        "IoU 0.5 matching is one-to-one; unmatched GT counts as FN, unmatched predictions as FP.",
        "mean_instance_dice_iou_0p5 conditions on matched TP; all_gt variant penalizes FN with zero.",
        "Primary center F1 is F1_center_4px; 6/8 px are sensitivity checks.",
        "center_in_mask_f1 uses a representative pixel contained in the stored GT instance.",
        "tile rows can overlap and are not independent biological observations.",
        "This is simulated data, not independently annotated real microscopy.",
    ]
    # The original summary helper only scans legacy prefixes. Explicitly add
    # all new cell-level outputs, with NaNs excluded from finite means.
    new_metric_prefixes = (
        "pixel_", "instance_", "center_in_mask_", "center_tp_r", "center_fp_r", "center_fn_r",
        "center_f1_r", "F1_center_4px", "center_precision_r", "center_recall_r", "center_mean_error_px_r",
        "matched_", "mean_instance_", "pq_", "aji", "possible_",
    )
    summary["cell_level_metrics"] = {}
    for col in result.select_dtypes(include="number"):
        if col.startswith(new_metric_prefixes):
            values = result[col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                summary["cell_level_metrics"][col] = {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "n": int(len(values)),
                }
    if cfg.out_csv:
        destination = Path(cfg.out_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        # Small per-configuration supplementary table, per *tile*.
        cols = (
            "F1_center_4px", "center_f1_r4", "center_mean_error_px_r4",
            "center_in_mask_f1", "center_in_mask_precision", "center_in_mask_recall",
            *(f"instance_f1_iou_{str(x).replace('.', 'p')}"
              for x in (0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.90)),
            "instance_f1_iou_0p5", "instance_precision_iou_0p5",
            "instance_recall_iou_0p5",
            "mean_instance_dice_iou_0p5", "mean_instance_jaccard_iou_0p5",
            "mean_instance_dice_all_gt_iou_0p5", "mean_instance_jaccard_all_gt_iou_0p5",
            "pixel_dice", "pixel_jaccard", "count_error_absolute", "pq_iou_0p5",
        )
        supp: dict[str, Any] = {"method": "AlloViewer_UNet", "level": "tile", "n_tiles": len(result)}
        for col in cols:
            v = pd.to_numeric(result.get(col, pd.Series(dtype=float)), errors="coerce")
            supp[f"{col}_mean"] = v.mean()
            supp[f"{col}_sd"] = v.std()
        for col in ("pixel_tp", "pixel_tn", "pixel_fp", "pixel_fn",
                    "instance_tp_iou_0p5", "instance_fp_iou_0p5", "instance_fn_iou_0p5",
                    "center_tp_r4", "center_fp_r4", "center_fn_r4",
                    "center_in_mask_tp", "center_in_mask_fp", "center_in_mask_fn",
                    *(f"instance_{name}_iou_{str(x).replace('.', 'p')}"
                      for x in (0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.90)
                      for name in ("tp", "fp", "fn"))):
            supp[f"{col}_total"] = int(pd.to_numeric(result.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        for metric, tp_key, fp_key, fn_key in (
            ("center_4px", "center_tp_r4", "center_fp_r4", "center_fn_r4"),
            ("center_in_mask", "center_in_mask_tp", "center_in_mask_fp", "center_in_mask_fn"),
            *((f"instance_iou_{str(x).replace('.', 'p')}",
               f"instance_tp_iou_{str(x).replace('.', 'p')}",
               f"instance_fp_iou_{str(x).replace('.', 'p')}",
               f"instance_fn_iou_{str(x).replace('.', 'p')}")
              for x in (0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.90)),
        ):
            tp_count, fp_count, fn_count = (
                supp[f"{k}_total"] for k in (tp_key, fp_key, fn_key)
            )
            denom_p, denom_r = tp_count + fp_count, tp_count + fn_count
            denom_f = 2 * tp_count + fp_count + fn_count
            supp[f"{metric}_micro_precision"] = (tp_count / denom_p if denom_p else
                                                    1.0 if not denom_r else 0.0)
            supp[f"{metric}_micro_recall"] = (tp_count / denom_r if denom_r else
                                                 1.0 if not denom_p else 0.0)
            supp[f"{metric}_micro_f1"] = (2 * tp_count / denom_f if denom_f else 1.0)
        pd.DataFrame([supp]).to_csv(
            destination.with_name(destination.stem + "_supplementary_summary.csv"), index=False
        )
    if cfg.out_summary_json:
        p = Path(cfg.out_summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    return result, summary


def run_single_training_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    unet_mode: str,
    dataset_mode: str,
    seg_method: str = "inst_seg",
    force: bool = False,
) -> pd.DataFrame:
    """Evaluate one checkpoint/data-mode pair using the original file names."""
    if dataset_mode not in ("crop_well_resize", "pad_resize", "tiles"):
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")
    if seg_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown seg_method: {seg_method}")
    os.makedirs(out_dir, exist_ok=True)
    basename = f"training_val_{unet_mode}_{dataset_mode}_{seg_method}"
    output = os.path.join(out_dir, basename + ".csv")
    summary_path = os.path.join(out_dir, basename + "_summary.json")
    if os.path.isfile(output) and not force:
        df = pd.read_csv(output)
        if not {"F1_center_4px", "center_in_mask_f1", "instance_f1_iou_0p9"}.issubset(df.columns):
            raise RuntimeError(
                f"Old output lacks new instance metrics: {output}. Pass force=True to recompute."
            )
        return df

    h5_path = os.path.join(h5_dir, f"{dataset_mode}_val.h5")
    model_file = f"best_{unet_mode}_{dataset_mode}_S512_seed187.pth"
    segmenter, cfg = validation_make_segmenter_and_config(
        h5_path=h5_path, out_csv=output, out_summary_json=summary_path,
        unet_mode=unet_mode, model_dir=model_dir, model_file=model_file,
        seg_method=seg_method, validation_config_cls=TrainingValidationConfig,
    )
    df, _ = validate_unet_segmentation(
        segmenter=segmenter, cfg=cfg, indices=None,
        segmentation_method=seg_method,
    )
    del segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def run_training_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    seg_method: str = "inst_seg",
    force: bool = False,
) -> None:
    """Run the original 3x3 grid of checkpoints/dataset modes."""
    frames = []
    for unet_mode in ("large", "medium", "small"):
        for dataset_mode in ("crop_well_resize", "pad_resize", "tiles"):
            print(f"UNet {unet_mode}; dataset {dataset_mode}; {seg_method}", flush=True)
            df = run_single_training_validation(
                out_dir=out_dir, model_dir=model_dir, h5_dir=h5_dir,
                unet_mode=unet_mode, dataset_mode=dataset_mode,
                seg_method=seg_method, force=force,
            )
            frames.append(df)
            gc.collect()
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(out_dir, f"training_val_combined_{seg_method}.csv"), index=False
        )
