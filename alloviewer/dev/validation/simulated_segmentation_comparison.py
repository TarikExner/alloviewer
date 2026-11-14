#!/usr/bin/env python3
"""Compare UNet, Cellpose, and ImageJ with simulated instance ground truth.

Install as alloviewer/dev/validation/simulated_segmentation_comparison.py.

Ground truth is ``/inst`` in ``fullres_ground_truth.h5``. These are actual
simulator instance labels, *not* masks inferred from network target heads.
The stored tiles have global instance IDs, so they can be stitched exactly.
ImageJ is reconstructed from the target heads in ``fullres_imageJ.h5`` using
the existing AlloViewer instance segmenter (as in imageJ_comparison.py).

Predictions are reconstructed at full image size and scored in full-image
and (optionally) per-tile crops, using the same segmentation outputs. Image
rows are independent image observations; tile rows may overlap and must not
be treated as independent measurements. UNet probability maps are averaged
in overlaps before instance segmentation. Cellpose receives the reconstructed original RGB image
(denormalized from the dataset's UNET_MEAN/STD), NOT the simulator's labels.

The original imageJ_comparison.py columns are retained for UNet and ImageJ.
Cellpose has no UNet-like boundary/center/energy probability heads, so their
head-specific metrics are missing for Cellpose. All three methods additionally
receive the *same* pixel/object/centroid metrics as the manual comparison,
using the actual simulated instances rather than approximate circular ROIs.

Output: testing_val_CPIJ_<unet_mode>_<seg_method>.csv
        testing_val_CPIJ_<unet_mode>_<seg_method>_supplementary_summary.csv
        testing_val_CPIJ_<unet_mode>_<seg_method>_imagewide_summary.json

Existing imageJ code is reused. Metrics use sparse one-to-one matching so
whole images containing thousands of cells can be scored without allocating
a dense cells-by-cells overlap matrix. No changes to existing modules are required.

Example (from the project root)::

    python -m alloviewer.dev.validation.simulated_segmentation_comparison \
      --h5-dir ./data --model-dir ./alloviewer/models --out-dir ./results \
      --unet-mode small --seg-method inst_seg --device cuda \
      --cellpose-diameter 10 --limit 5 --force

Instance F1 is scored at IoU 0.10/0.25/0.30/0.50/0.70/0.75/0.90.
Ground-truth-center-pixel inclusion provides another one-to-one F1 metric.

Cellpose 3 (cyto3) and Cellpose 4 (Cellpose-SAM) are supported with the same
version-dependent settings as manual_segmentation_comparison.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage as ndi
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..segmentation import TiledH5Dataset, UNET_MEAN, UNET_STD
from ..segmentation.utils import collate_no_meta
from .config import TrainingValidationConfig
from .imageJ_comparison import (
    _assert_same_sample_identity,
    _assert_same_tile_grid,
    _extract_full_hw,
    _jsonify_params,
    _make_segmenter_and_validation_config,
    _row_metrics_for_prediction,
    _segment_full_probs,
    stitch_float_tiles_mean,
    stitch_instance_tiles,
)
from .manual_segmentation_comparison import (
    _make_cellpose,
    _predict_cellpose,
)
from .utils import iou_dice_overlap

VALIDATION_VERSION = "simulated_fullres_CPIJ_stored_instance_gt_v4"
METHODS = ("UNet", "Cellpose", "imageJ")


def _unet_normalized_to_rgb(tiles: np.ndarray) -> np.ndarray:
    """Convert [T,3,H,W] dataset-normalized tiles back to RGB in [0, 1]."""
    if tiles.ndim != 4 or tiles.shape[1] != 3:
        raise ValueError(f"Expected UNet-normalized RGB [T,3,H,W], got {tiles.shape}")
    mean = np.asarray(UNET_MEAN, dtype=np.float32)[None, :, None, None]
    std = np.asarray(UNET_STD, dtype=np.float32)[None, :, None, None]
    rgb = tiles.astype(np.float32, copy=False) * std + mean
    if not np.all(np.isfinite(rgb)):
        raise ValueError("Non-finite pixel values after denormalizing H5 image")
    # H5 source images were in [0,1] before H5CellsDataset normalized them.
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def _full_rgb_from_tiles(
    normalized_tiles: np.ndarray,
    tile_metas: list[dict[str, Any]],
    full_hw: tuple[int, int],
) -> np.ndarray:
    """Reconstruct original RGB full image without using GT masks."""
    rgb_chw = stitch_float_tiles_mean(
        _unet_normalized_to_rgb(normalized_tiles), tile_metas, full_hw
    )
    return np.ascontiguousarray(np.moveaxis(rgb_chw, 0, -1))


def _compact(labels: np.ndarray) -> np.ndarray:
    """Map nonzero global instance IDs onto 1..N, without joining instances."""
    x = np.asarray(labels)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.integer) or np.any(x < 0):
        raise ValueError("Instances must be a nonnegative 2D integer array")
    unique = np.unique(x)
    unique = unique[unique != 0]
    out = np.zeros(x.shape, dtype=np.int32)
    foreground = x != 0
    out[foreground] = np.searchsorted(unique, x[foreground]).astype(np.int32) + 1
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == fp == fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def _match_sparse_edges(
    gt_idx: np.ndarray, pred_idx: np.ndarray, score: np.ndarray,
    ng: int, npred: int, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum-cardinality one-to-one matching, then maximum overlap score.

    Uses connected components of the sparse bipartite candidate graph instead
    of materializing an N_gt x N_prediction dense matrix. This matters for
    dense, full-resolution simulated wells containing many thousands of cells.
    """
    gi, pi, values = gt_idx[valid], pred_idx[valid], score[valid]
    if not len(gi):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=float)
    # Two directions are needed for an undirected bipartite graph.
    vertices_a = np.concatenate([gi, ng + pi])
    vertices_b = np.concatenate([ng + pi, gi])
    adjacency = sparse.coo_matrix(
        (np.ones(len(vertices_a), dtype=np.uint8), (vertices_a, vertices_b)),
        shape=(ng + npred, ng + npred),
    ).tocsr()
    _, component = connected_components(adjacency, directed=False)
    edge_components = component[gi]
    chosen_gt: list[np.ndarray] = []
    chosen_pred: list[np.ndarray] = []
    chosen_scores: list[np.ndarray] = []
    for component_id in np.unique(edge_components):
        where = edge_components == component_id
        this_g, this_p, this_score = gi[where], pi[where], values[where]
        gt_nodes, row_idx = np.unique(this_g, return_inverse=True)
        pred_nodes, col_idx = np.unique(this_p, return_inverse=True)
        weights = np.zeros((len(gt_nodes), len(pred_nodes)), dtype=np.float64)
        weights[row_idx, col_idx] = this_score
        # The cardinality bonus exceeds the largest possible total score
        # difference, so matches are maximized before overlap quality.
        bonus = min(len(gt_nodes), len(pred_nodes)) + 1.0
        weights = np.where(weights > 0, weights + bonus, 0.0)
        rows, cols = linear_sum_assignment(-weights)
        matched = weights[rows, cols] > 0
        if not matched.any():
            continue
        rows, cols = rows[matched], cols[matched]
        chosen_gt.append(gt_nodes[rows])
        chosen_pred.append(pred_nodes[cols])
        chosen_scores.append(weights[rows, cols] - bonus)
    return (
        np.concatenate(chosen_gt), np.concatenate(chosen_pred),
        np.concatenate(chosen_scores),
    )


def _centers(labels: np.ndarray, n_labels: int) -> np.ndarray:
    if not n_labels:
        return np.empty((0, 2), dtype=float)
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    mass = np.bincount(ids, minlength=n_labels + 1)[1:]
    sum_y = np.bincount(ids, weights=ys, minlength=n_labels + 1)[1:]
    sum_x = np.bincount(ids, weights=xs, minlength=n_labels + 1)[1:]
    return np.column_stack((sum_y / mass, sum_x / mass))


def _boundary_f1(gt_fg: np.ndarray, pred_fg: np.ndarray, tolerance: int):
    structure = np.ones((3, 3), dtype=bool)
    gt_boundary = gt_fg & ~ndi.binary_erosion(gt_fg, structure=structure)
    pred_boundary = pred_fg & ~ndi.binary_erosion(pred_fg, structure=structure)
    if not gt_boundary.any() and not pred_boundary.any():
        return 1.0, 1.0, 1.0
    if not gt_boundary.any() or not pred_boundary.any():
        return 0.0, 0.0, 0.0
    distance_gt = ndi.distance_transform_edt(~gt_boundary)
    distance_pred = ndi.distance_transform_edt(~pred_boundary)
    p = float((distance_gt[pred_boundary] <= tolerance).mean())
    r = float((distance_pred[gt_boundary] <= tolerance).mean())
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def compare_simulated_instances(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    boundary_tolerance: int = 2,
) -> dict[str, Any]:
    """Cell-level, pixel-level and object-overlap metrics on exact GT masks.

    Counts: pixel TP/TN/FP/FN are binary foreground/background counts.
    Object TP/FP/FN use one-to-one matching at each stated IoU cutoff;
    there is no meaningful object-level TN. At IoU=0.5, mean matched-object
    Dice/Jaccard exclude unmatched cells, while `all_gt` variants explicitly
    score missed GT cells as zero. Empty-set conventions are stated in summary.

    Both image-wide and tile-crop rows use this exact same metric function.
    """
    if boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be nonnegative")
    gt = _compact(reference)
    pred = _compact(predicted)
    if gt.shape != pred.shape:
        raise ValueError(f"Mismatched shapes: {gt.shape}, {pred.shape}")
    ng, npred = int(gt.max()), int(pred.max())
    gt_fg, pred_fg = gt > 0, pred > 0
    tp = int(np.count_nonzero(gt_fg & pred_fg))
    fp = int(np.count_nonzero(~gt_fg & pred_fg))
    fn = int(np.count_nonzero(gt_fg & ~pred_fg))
    tn = int(np.count_nonzero(~gt_fg & ~pred_fg))
    p, r, f1 = _prf(tp, fp, fn)
    iou_binary = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    gt_area = np.bincount(gt[gt_fg], minlength=ng + 1)[1:]
    pred_area = np.bincount(pred[pred_fg], minlength=npred + 1)[1:]

    if tp:
        codes = (gt[gt_fg & pred_fg].astype(np.int64) - 1) * npred + (
            pred[gt_fg & pred_fg].astype(np.int64) - 1
        )
        codes, intersection = np.unique(codes, return_counts=True)
        gi = codes // npred
        pi = codes % npred
        intersection = intersection.astype(np.float64)
        union = gt_area[gi] + pred_area[pi] - intersection
        pair_iou = intersection / union
        pair_dice = 2.0 * intersection / (gt_area[gi] + pred_area[pi])
    else:
        gi = pi = np.empty(0, dtype=np.int64)
        codes = np.empty(0, dtype=np.int64)
        intersection = union = pair_iou = pair_dice = np.empty(0, dtype=float)

    metrics: dict[str, Any] = {
        "n_gt_instances": ng,
        "n_pred_instances": npred,
        "count_error_signed": npred - ng,
        "count_error_absolute": abs(npred - ng),
        "count_error_relative": (npred - ng) / ng if ng else np.nan,
        "pixel_tp": tp, "pixel_tn": tn, "pixel_fp": fp, "pixel_fn": fn,
        "pixel_precision": p, "pixel_recall": r, "pixel_f1": f1,
        "pixel_dice": f1, "pixel_jaccard": iou_binary,
        "pixel_iou": iou_binary,
        "pixel_specificity": tn / (tn + fp) if tn + fp else 1.0,
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn),
        "evaluation_pixels": int(gt.size),
    }

    # Instance quality: matching by IoU. The IoU cutoff is part of each name.
    for cutoff in (0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.90):
        suffix = str(cutoff).replace(".", "p")
        mg, mp, overlaps = _match_sparse_edges(
            gi, pi, pair_iou, ng, npred, pair_iou >= cutoff
        )
        matched_count = len(mg)
        i_tp, i_fp, i_fn = matched_count, npred - matched_count, ng - matched_count
        i_p, i_r, i_f1 = _prf(i_tp, i_fp, i_fn)
        if matched_count:
            # Dice = 2*IoU/(1+IoU), same matching pairs and cutoff.
            matched_mean_iou = float(overlaps.mean())
            matched_mean_dice = float((2.0 * overlaps / (1.0 + overlaps)).mean())
            matched_sum_iou = float(overlaps.sum())
            matched_sum_dice = float((2.0 * overlaps / (1.0 + overlaps)).sum())
        else:
            matched_mean_iou = matched_mean_dice = np.nan
            matched_sum_iou = matched_sum_dice = 0.0
        empty_perfect = ng == npred == 0
        metrics.update({
            f"instance_tp_iou_{suffix}": i_tp,
            f"instance_fp_iou_{suffix}": i_fp,
            f"instance_fn_iou_{suffix}": i_fn,
            f"instance_precision_iou_{suffix}": i_p,
            f"instance_recall_iou_{suffix}": i_r,
            f"instance_f1_iou_{suffix}": i_f1,
            f"matched_mean_iou_{suffix}": matched_mean_iou,
            f"matched_mean_dice_{suffix}": matched_mean_dice,
            f"mean_instance_jaccard_iou_{suffix}": matched_mean_iou,
            f"mean_instance_dice_iou_{suffix}": matched_mean_dice,
            f"mean_instance_jaccard_all_gt_iou_{suffix}":
                matched_sum_iou / ng if ng else (1.0 if empty_perfect else 0.0),
            f"mean_instance_dice_all_gt_iou_{suffix}":
                matched_sum_dice / ng if ng else (1.0 if empty_perfect else 0.0),
            f"pq_iou_{suffix}":
                matched_sum_iou / (i_tp + 0.5 * i_fp + 0.5 * i_fn)
                if (i_tp + i_fp + i_fn) else 1.0,
        })

    # Aggregated Jaccard Index, using nonzero-overlap, one-to-one matches.
    mg, mp, _ = _match_sparse_edges(gi, pi, pair_iou, ng, npred, pair_iou > 0)
    if len(mg):
        matching_codes = mg.astype(np.int64) * npred + mp
        ix = np.searchsorted(codes, matching_codes)
        # codes were sorted by np.unique; all matched edges exist in codes.
        numerator = float(intersection[ix].sum())
        denominator = float(union[ix].sum())
        matched_g = np.ones(ng, dtype=bool)
        matched_p = np.ones(npred, dtype=bool)
        matched_g[mg] = False
        matched_p[mp] = False
        denominator += float(gt_area[matched_g].sum() + pred_area[matched_p].sum())
        metrics["aji"] = numerator / denominator if denominator else 1.0
    else:
        metrics["aji"] = 1.0 if ng == npred == 0 else 0.0

    # Independent GT-center-pixel-in-predicted-instance detection. For stored
    # instance masks, use the nearest integer pixel to the GT centroid. Each
    # predicted instance can match at most one GT cell, including in merges.
    # This is intentionally different from 4/6/8-pixel centroid-distance F1.
    gt_centers = _centers(gt, ng)
    rounded = np.rint(gt_centers).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, gt.shape[0] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, gt.shape[1] - 1)
    # A concave / ring-shaped instance may have its centroid in background.
    # Use the closest GT-positive pixel in that instance instead: the exact
    # GT mask must match itself perfectly under center-in-mask scoring.
    needs_relocation = np.flatnonzero(
        gt[rounded[:, 0], rounded[:, 1]] != np.arange(1, ng + 1)
    )
    if len(needs_relocation):
        slices = ndi.find_objects(gt)
        for i in needs_relocation:
            sl = slices[int(i)]
            if sl is None:
                raise ValueError(f"Missing ground-truth instance {int(i) + 1}")
            y_local, x_local = np.nonzero(gt[sl] == int(i) + 1)
            y_global, x_global = y_local + sl[0].start, x_local + sl[1].start
            nearest = np.argmin(
                (y_global - gt_centers[i, 0]) ** 2 +
                (x_global - gt_centers[i, 1]) ** 2
            )
            rounded[i] = (y_global[nearest], x_global[nearest])
    hit_ids = pred[rounded[:, 0], rounded[:, 1]] if ng else np.empty(0, dtype=np.int32)
    n_hit = int(np.unique(hit_ids[hit_ids > 0]).size)
    hit_fp, hit_fn = npred - n_hit, ng - n_hit
    hit_p, hit_r, hit_f1 = _prf(n_hit, hit_fp, hit_fn)
    metrics.update({
        "center_in_mask_tp": n_hit,
        "center_in_mask_fp": hit_fp,
        "center_in_mask_fn": hit_fn,
        "center_in_mask_precision": hit_p,
        "center_in_mask_recall": hit_r,
        "center_in_mask_f1": hit_f1,
    })
    # Center matching at same radii as manual segmentation comparison.
    pred_centers = _centers(pred, npred)
    if ng and npred:
        tree = cKDTree(pred_centers)
        neighborhoods = tree.query_ball_point(gt_centers, r=8.0)
        cg, cp, distance = [], [], []
        for a, neighbors in enumerate(neighborhoods):
            for b in neighbors:
                cg.append(a)
                cp.append(b)
                distance.append(float(np.linalg.norm(gt_centers[a] - pred_centers[b])))
        cg = np.asarray(cg, dtype=np.int64)
        cp = np.asarray(cp, dtype=np.int64)
        distance = np.asarray(distance, dtype=float)
    else:
        cg = cp = np.empty(0, dtype=np.int64)
        distance = np.empty(0, dtype=float)
    for radius in (4.0, 6.0, 8.0):
        valid = distance <= radius
        match_g, match_p, _ = _match_sparse_edges(
            cg, cp, 1.0 - distance / (radius + 1.0), ng, npred, valid
        )
        n_match = len(match_g)
        c_tp, c_fp, c_fn = n_match, npred - n_match, ng - n_match
        c_p, c_r, c_f1 = _prf(c_tp, c_fp, c_fn)
        mean_dist = (float(np.linalg.norm(
            gt_centers[match_g] - pred_centers[match_p], axis=1
        ).mean()) if n_match else np.nan)
        suffix = str(int(radius))
        metrics.update({
            f"center_tp_r{suffix}": c_tp,
            f"center_fp_r{suffix}": c_fp,
            f"center_fn_r{suffix}": c_fn,
            f"center_precision_r{suffix}": c_p,
            f"center_recall_r{suffix}": c_r,
            f"center_f1_r{suffix}": c_f1,
            f"center_mean_error_px_r{suffix}": mean_dist,
        })

    # Primary center-distance F1: 4 pixels (~half the reference circle diameter).
    metrics["F1_center_4px"] = metrics["center_f1_r4"]
    boundary_p, boundary_r, boundary_f1 = _boundary_f1(gt_fg, pred_fg, boundary_tolerance)
    metrics.update({
        "boundary_precision_tol_px": boundary_p,
        "boundary_recall_tol_px": boundary_r,
        "boundary_f1_tol_px": boundary_f1,
        "boundary_tolerance_px": boundary_tolerance,
    })
    if tp:
        splits = np.bincount(
            gi, weights=(intersection / gt_area[gi] >= 0.10).astype(int), minlength=ng
        )
        merges = np.bincount(
            pi, weights=(intersection / pred_area[pi] >= 0.10).astype(int), minlength=npred
        )
        metrics["possible_splits"] = int((splits > 1).sum())
        metrics["possible_merges"] = int((merges > 1).sum())
    else:
        metrics["possible_splits"] = metrics["possible_merges"] = 0
    return metrics


def _common_instance_metrics(
    gt_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    boundary_tolerance: int,
) -> dict[str, Any]:
    return compare_simulated_instances(
        gt_labels, predicted_labels, boundary_tolerance=boundary_tolerance
    )


def _cellpose_legacy_metrics(
    *,
    sample_idx: int,
    inst_pred: np.ndarray,
    inst_gt: np.ndarray,
    cell_gt: np.ndarray,
    cfg: TrainingValidationConfig,
    src_path: str,
    n_cells_full_meta: Any,
    frac_positive: Any,
    params_out: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatible legacy columns that are meaningful for instance-only CP.

    We deliberately DO NOT invent probabilities or a center/energy heatmap.
    The legacy center_* and energy_* metrics are left NaN by pandas, while
    the directly comparable center_f1_r{4,6,8} metrics are computed separately.
    The primary center-distance score is F1_center_4px (alias of center_f1_r4).
    """
    n_gt = int(np.unique(inst_gt[inst_gt > 0]).size)
    n_pred = int(np.unique(inst_pred[inst_pred > 0]).size)
    mask_stats = iou_dice_overlap(inst_pred > 0, cell_gt > 0)
    return {
        "validation_version": VALIDATION_VERSION,
        "sample_idx": sample_idx,
        "tile_idx": -1,
        "metric_level": "image",
        "dataset_mode": "Cellpose",
        "segmentation_method": "instance",
        "src_path": src_path,
        "n_cells_full_meta": n_cells_full_meta,
        "frac_positive": frac_positive,
        "n_cells_gt_instances": n_gt,
        "n_cells_pred_instances": n_pred,
        "count_error_instances_vs_gt_instances": n_pred - n_gt,
        "abs_count_error_instances_vs_gt_instances": abs(n_pred - n_gt),
        **{f"mask_{key}": val for key, val in mask_stats.items()},
        **params_out,
    }


def _summary_for_results(
    df: pd.DataFrame,
    *,
    gt_h5: Path,
    imagej_h5: Path,
    cellpose_name: str,
    cellpose_version: str,
    cellpose_diameter: float,
) -> dict[str, Any]:
    """Per-method/per-level macro means plus explicitly pooled confusion counts."""
    excluded_cols = {
        "sample_idx", "tile_idx", "n_cells_full_meta", "frac_positive",
        "full_height", "full_width", "tile_x0", "tile_y0", "tile_height", "tile_width",
        "evaluation_pixels", "boundary_tolerance_px", "cellpose_diameter",
        "cellpose_min_size", "cellpose_flow_threshold", "cellpose_cellprob_threshold",
    }
    numeric_cols = [c for c in df.columns if c not in excluded_cols
                    and pd.api.types.is_numeric_dtype(df[c])
                    and (c.startswith(("mask_", "pixel_", "instance_", "center_", "boundary_",
                                      "pq_", "aji", "matched_", "mean_instance_", "count_error_",
                                      "abs_count_error_", "possible_")))]
    comparisons: dict[str, Any] = {}
    for (level, method), frame in df.groupby(["metric_level", "dataset_mode"], sort=False):
        metric_summary: dict[str, Any] = {}
        for col in numeric_cols:
            series = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
            finite = series[np.isfinite(series)]
            if len(finite):
                metric_summary[col] = {"mean": float(finite.mean()),
                                       "std": float(finite.std(ddof=1)) if len(finite)>1 else 0.0,
                                       "n": int(len(finite))}
        confusion = {}
        for prefix in ("pixel", "instance_iou_0p5", "center_r4", "center_in_mask"):
            fields = (("pixel_tp", "pixel_tn", "pixel_fp", "pixel_fn") if prefix == "pixel" else
                      ("instance_tp_iou_0p5", "instance_fp_iou_0p5", "instance_fn_iou_0p5")
                      if prefix == "instance_iou_0p5" else
                      ("center_tp_r4", "center_fp_r4", "center_fn_r4")
                      if prefix == "center_r4" else
                      ("center_in_mask_tp", "center_in_mask_fp", "center_in_mask_fn"))
            confusion[prefix] = {name: int(pd.to_numeric(frame[name], errors="coerce").fillna(0).sum())
                                 for name in fields}
            tp = confusion[prefix][fields[0]]
            fp = confusion[prefix][fields[-2]]
            fn = confusion[prefix][fields[-1]]
            precision, recall, f1 = _prf(tp, fp, fn)
            confusion[prefix].update({"micro_precision": precision,
                                      "micro_recall": recall, "micro_f1": f1})
        comparisons.setdefault(str(level), {})[str(method)] = {
            "n_rows": int(len(frame)), "n_images": int(frame["sample_idx"].nunique()),
            "metrics": metric_summary, "pooled_confusion": confusion,
        }
    return {
        "validation_version": VALIDATION_VERSION,
        "ground_truth": "stored /inst global instance IDs in fullres_ground_truth.h5",
        "gt_h5": str(gt_h5), "imagej_h5": str(imagej_h5),
        "n_images": int(df["sample_idx"].nunique()) if len(df) else 0,
        "n_rows": int(len(df)),
        "cellpose_model": cellpose_name, "cellpose_version": cellpose_version,
        "cellpose_diameter_px": cellpose_diameter,
        "by_level_and_method": comparisons,
        "notes": [
            "Primary statistical unit: full image; tile crops overlap and are not independent.",
            "Tile-level instances crossing borders are counted separately in each crop.",
            "All methods evaluated against the same stored simulator /inst labels.",
            "pixel_tp/tn/fp/fn refer to pixels; instance_tp/fp/fn refer to cells; object TN is undefined.",
            "mean_instance_dice/jaccard_iou_0p5 average only matched cells at IoU >=0.5.",
            "mean_instance_dice/jaccard_all_gt_iou_0p5 additionally assign zero to missed GT cells.",
            "Empty GT and prediction use score 1; unmatched-pair mean Dice/Jaccard are NaN.",
            "F1_center_4px is the primary center-distance metric (center_f1_r4 alias).",
            "Center matching at 6/8 px is a sensitivity check; centers-in-mask are separate.",
            "The GT center-in-mask representative pixel is always inside its GT instance.",
            "ImageJ and UNet head-specific scores are retained from the original comparison.",
            "This is simulated-image validation, not independent real-image annotation.",
        ],
    }


def _supplementary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One readable summary row per method and evaluation level.

    Report both macro averages of per-image (or per-tile) scores and pooled
    cell/pixel confusion counts. The image rows are the independent units.
    """
    summary_columns = (
        "F1_center_4px", "center_f1_r4", "center_mean_error_px_r4",
        "center_in_mask_precision", "center_in_mask_recall", "center_in_mask_f1",
        *(f"instance_f1_iou_{str(x).replace('.', 'p')}"
          for x in (0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.90)),
        "pixel_dice", "pixel_jaccard", "pixel_precision", "pixel_recall",
        "pixel_specificity", "instance_precision_iou_0p5",
        "instance_recall_iou_0p5", "instance_f1_iou_0p5",
        "center_precision_r6", "center_recall_r6", "center_f1_r6",
        "center_mean_error_px_r6", "mean_instance_dice_iou_0p5",
        "mean_instance_jaccard_iou_0p5", "mean_instance_dice_all_gt_iou_0p5",
        "mean_instance_jaccard_all_gt_iou_0p5", "pq_iou_0p5", "aji",
        "count_error_signed", "count_error_absolute",
    )
    counts = (
        "pixel_tp", "pixel_tn", "pixel_fp", "pixel_fn",
        "instance_tp_iou_0p5", "instance_fp_iou_0p5", "instance_fn_iou_0p5",
        "center_tp_r4", "center_fp_r4", "center_fn_r4",
        "center_in_mask_tp", "center_in_mask_fp", "center_in_mask_fn",
        *(f"instance_{kind}_iou_{str(x).replace('.', 'p')}"
          for x in (0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.90)
          for kind in ("tp", "fp", "fn")),
    )
    rows = []
    for (level, method), group in df.groupby(["metric_level", "dataset_mode"], sort=False):
        row = {"metric_level": level, "dataset_mode": method,
               "n_images": int(group.sample_idx.nunique()),
               "n_rows": int(len(group))}
        for name in summary_columns:
            values = pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{name}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{name}_n"] = int(len(values))
        for name in counts:
            row[f"{name}_total"] = int(pd.to_numeric(group[name], errors="coerce").fillna(0).sum())
        for tag, tp_key, fp_key, fn_key in (
            ("pixel", "pixel_tp", "pixel_fp", "pixel_fn"),
            ("instance_iou_0p5", "instance_tp_iou_0p5",
             "instance_fp_iou_0p5", "instance_fn_iou_0p5"),
            ("center_r4", "center_tp_r4", "center_fp_r4", "center_fn_r4"),
            ("center_in_mask", "center_in_mask_tp", "center_in_mask_fp", "center_in_mask_fn"),
            *((f"instance_iou_{str(x).replace('.', 'p')}",
               f"instance_tp_iou_{str(x).replace('.', 'p')}",
               f"instance_fp_iou_{str(x).replace('.', 'p')}",
               f"instance_fn_iou_{str(x).replace('.', 'p')}")
              for x in (0.10, 0.30, 0.70, 0.90)),
        ):
            pp, rr, ff = _prf(
                row[f"{tp_key}_total"], row[f"{fp_key}_total"], row[f"{fn_key}_total"]
            )
            row[f"{tag}_micro_precision"] = pp
            row[f"{tag}_micro_recall"] = rr
            row[f"{tag}_micro_f1"] = ff
        rows.append(row)
    return pd.DataFrame(rows)


def validate_simulated_segmentation(
    segmenter: Any,
    imagej_segmenter: Any,
    cellpose_engine: Any,
    *,
    cfg: TrainingValidationConfig,
    imagej_h5_path: str | Path,
    cellpose_major: int,
    cellpose_version: str,
    cellpose_model_label: str,
    cellpose_diameter: float = 10.0,
    cellpose_min_size: int = 3,
    cellpose_flow_threshold: float = 0.4,
    cellpose_cellprob_threshold: float = 0.0,
    boundary_tolerance: int = 2,
    indices: Optional[Sequence[int]] = None,
    stop: Optional[int] = None,
    metric_level: str = "both",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare all three engines against simulated GT at full-image and tile level.

    ``segmenter`` and ``imagej_segmenter`` follow the same API as in the
    existing imageJ_comparison.py. ``cellpose_engine`` must implement eval().
    These injected engines allow checks without model checkpoints.
    """
    if metric_level not in {"image", "tile", "both"}:
        raise ValueError("metric_level must be image, tile or both")
    if cellpose_diameter <= 0 or boundary_tolerance < 0:
        raise ValueError("Require positive Cellpose diameter and nonnegative boundary tolerance")
    if stop is not None and stop < 1:
        raise ValueError("stop must be >= 1")
    if imagej_segmenter.inst_seg is None:
        raise ValueError("ImageJ comparison requires a configured instance segmenter")

    gt_h5 = Path(cfg.h5_path).expanduser().resolve(strict=True)
    ij_h5 = Path(imagej_h5_path).expanduser().resolve(strict=True)
    gt_ds = TiledH5Dataset(str(gt_h5), indices=indices)
    ij_ds = TiledH5Dataset(str(ij_h5), indices=indices)
    if len(gt_ds) != len(ij_ds):
        raise RuntimeError(f"GT and ImageJ sample counts differ: {len(gt_ds)} vs {len(ij_ds)}")

    make_loader = lambda ds: DataLoader(  # noqa: E731
        ds, batch_size=1, shuffle=False, num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        collate_fn=collate_no_meta,
    )
    rows: list[dict[str, Any]] = []
    try:
        iterator = zip(make_loader(gt_ds), make_loader(ij_ds))
        with tqdm(total=min(len(gt_ds), stop or len(gt_ds)), desc="Simulated: UNet / Cellpose / ImageJ", unit="image") as bar:
            for position, (batch_gt, batch_ij) in enumerate(iterator):
                if stop is not None and position >= stop:
                    break
                sample_idx = position if indices is None else int(indices[position])
                imgs_t, tgts_gt_t, extras_gt = batch_gt
                _, tgts_ij_t, extras_ij = batch_ij
                meta_gt = extras_gt["meta"][0]
                meta_ij = extras_ij["meta"][0]
                _assert_same_tile_grid(meta_gt, meta_ij, sample_idx)
                _assert_same_sample_identity(meta_gt, meta_ij, sample_idx)

                tile_metas = meta_gt["tiles"]
                n_tiles = len(tile_metas)
                if not n_tiles:
                    raise ValueError(f"No tiles at sample {sample_idx}")
                imgs = imgs_t[0, :n_tiles].detach().cpu().numpy().astype(np.float32)
                gt_targets = tgts_gt_t[0, :n_tiles].detach().cpu().numpy().astype(np.float32)
                ij_targets = tgts_ij_t[0, :n_tiles].detach().cpu().numpy().astype(np.float32)
                gt_instances = extras_gt["instance_labels"][0, :n_tiles].detach().cpu().numpy().astype(np.int32)
                full_hw = _extract_full_hw(meta_gt, tile_metas)

                # Stored /inst uses global object IDs across overlapping crops.
                gt_full = stitch_instance_tiles(gt_instances, tile_metas, full_hw)
                gt_targets_full = stitch_float_tiles_mean(gt_targets, tile_metas, full_hw)
                ij_targets_full = stitch_float_tiles_mean(ij_targets, tile_metas, full_hw)
                cell_gt = (gt_targets_full[0] > 0.5).astype(np.uint8)
                energy_gt = gt_targets_full[3].astype(np.float32)
                image_full_rgb = _full_rgb_from_tiles(imgs, tile_metas, full_hw)

                # UNet receives exactly the normalized tiles used in the old script.
                with torch.inference_mode():
                    unet_out = segmenter(imgs)
                probs_unet_tiles = np.stack(
                    [unet_out["probs"][channel] for channel in ("cell", "bound", "center", "energy")],
                    axis=1,
                ).astype(np.float32)
                probs_unet = stitch_float_tiles_mean(probs_unet_tiles, tile_metas, full_hw)
                inst_unet = _segment_full_probs(
                    segmenter.inst_seg, *probs_unet,
                    cell_thr=cfg.cell_thr, update_cell_mask=True,
                )
                inst_imagej = _segment_full_probs(
                    imagej_segmenter.inst_seg, *ij_targets_full,
                    cell_thr=cfg.cell_thr, update_cell_mask=True,
                )
                inst_cellpose = _predict_cellpose(
                    cellpose_engine, image_full_rgb,
                    major=cellpose_major,
                    diameter=cellpose_diameter,
                    min_size=cellpose_min_size,
                    flow_threshold=cellpose_flow_threshold,
                    cellprob_threshold=cellpose_cellprob_threshold,
                )
                if inst_cellpose.shape != full_hw:
                    raise ValueError(f"Cellpose result shape {inst_cellpose.shape} != full image {full_hw}")
                predictions = {
                    "UNet": np.asarray(inst_unet, dtype=np.int32),
                    "Cellpose": np.asarray(inst_cellpose, dtype=np.int32),
                    "imageJ": np.asarray(inst_imagej, dtype=np.int32),
                }
                for method, predicted in predictions.items():
                    if predicted.shape != full_hw:
                        raise ValueError(f"{method} result shape {predicted.shape} != GT shape {full_hw}")
                    if np.any(predicted < 0):
                        raise ValueError(f"{method} returned negative instance identifiers")

                full_meta = meta_gt.get("full", {})
                if not isinstance(full_meta, dict):
                    full_meta = {}
                src_path = str(full_meta.get("src_path", ""))
                n_cells_meta = full_meta.get("n_cells")
                frac_positive = full_meta.get("frac_positive")
                params = full_meta.get("params", {})
                params_out = _jsonify_params(params if isinstance(params, dict) else {})

                for method in METHODS:
                    predicted = predictions[method]
                    shared = {
                        "validation_version": VALIDATION_VERSION,
                        "sample_idx": sample_idx,
                        "dataset_mode": method,
                        "method": "AlloViewer_UNet" if method == "UNet" else method,
                        "src_path": src_path,
                        "n_tiles": n_tiles,
                        "full_height": full_hw[0],
                        "full_width": full_hw[1],
                        "gt_source": "stored_global_instance_labels_in_gt_H5_inst",
                        "unet_mode": segmenter.cfg.unet_mode,
                        "cellpose_model": cellpose_model_label,
                        "cellpose_version": cellpose_version,
                        "cellpose_diameter": cellpose_diameter,
                        "cellpose_min_size": cellpose_min_size,
                        "cellpose_flow_threshold": cellpose_flow_threshold,
                        "cellpose_cellprob_threshold": cellpose_cellprob_threshold,
                        "boundary_tolerance": boundary_tolerance,
                    }
                    if metric_level in {"image", "both"}:
                        if method == "Cellpose":
                            legacy = _cellpose_legacy_metrics(
                                sample_idx=sample_idx, inst_pred=predicted,
                                inst_gt=gt_full, cell_gt=cell_gt, cfg=cfg,
                                src_path=src_path, n_cells_full_meta=n_cells_meta,
                                frac_positive=frac_positive, params_out=params_out,
                            )
                        else:
                            heads = probs_unet if method == "UNet" else ij_targets_full
                            legacy = _row_metrics_for_prediction(
                                sample_idx=sample_idx, dataset_mode=method,
                                segmentation_method=("inst_seg" if method == "imageJ" else
                                                     "inst_seg" if segmenter.inst_seg is not None else "conventional"),
                                cell_pred=heads[0], bound_pred=heads[1],
                                center_pred=heads[2], energy_pred=heads[3],
                                inst_pred=predicted, cell_gt=cell_gt, energy_gt=energy_gt,
                                inst_gt=gt_full, cfg=cfg, src_path=src_path,
                                n_cells_full_meta=n_cells_meta, frac_positive=frac_positive,
                                params_out=params_out,
                            )
                        comparable = _common_instance_metrics(
                            gt_full, predicted, boundary_tolerance=boundary_tolerance
                        )
                        rows.append({
                            **legacy, **comparable, **shared,
                            "metric_level": "image", "level": "image", "tile_idx": -1,
                            "tile_x0": np.nan, "tile_y0": np.nan,
                            "tile_height": full_hw[0], "tile_width": full_hw[1],
                        })
                    if metric_level in {"tile", "both"}:
                        # These are CROPS OF THE STITCHED FULL-IMAGE prediction.
                        # This makes them comparable to image-wide results,
                        # rather than running a different inference per tile.
                        for tile_idx, tile_meta in enumerate(tile_metas):
                            y0, x0 = (int(v) for v in tile_meta["tile_xy"])
                            h, w = (int(v) for v in tile_meta["tile_hw"])
                            y1, x1 = min(full_hw[0], y0 + h), min(full_hw[1], x0 + w)
                            if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
                                raise ValueError(f"Invalid tile {tile_idx}: {tile_meta}")
                            gt_crop = gt_full[y0:y1, x0:x1]
                            predicted_crop = predicted[y0:y1, x0:x1]
                            tile_metrics = _common_instance_metrics(
                                gt_crop, predicted_crop, boundary_tolerance=boundary_tolerance
                            )
                            rows.append({
                                **shared, **tile_metrics,
                                "metric_level": "tile", "level": "tile",
                                "tile_idx": tile_idx,
                                "tile_x0": x0, "tile_y0": y0,
                                "tile_height": y1 - y0, "tile_width": x1 - x0,
                                "segmentation_method": ("inst_seg" if method != "UNet"
                                                        else "inst_seg" if segmenter.inst_seg is not None
                                                        else "conventional"),
                            })
                bar.update(1)
    finally:
        gt_ds.close()
        ij_ds.close()

    df = pd.DataFrame(rows)
    summary = _summary_for_results(
        df, gt_h5=gt_h5, imagej_h5=ij_h5,
        cellpose_name=cellpose_model_label, cellpose_version=cellpose_version,
        cellpose_diameter=cellpose_diameter,
    )
    if cfg.out_csv:
        out = Path(cfg.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        _supplementary_table(df).to_csv(
            out.with_name(out.stem + "_supplementary_summary.csv"), index=False
        )
    if cfg.out_summary_json:
        p = Path(cfg.out_summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return df, summary


def run_simulated_segmentation_comparison(
    out_dir: str | Path,
    model_dir: str | Path,
    h5_dir: str | Path,
    unet_mode: str = "small",
    seg_method: str = "inst_seg",
    *,
    gt_h5_name: str = "fullres_ground_truth.h5",
    imagej_h5_name: str = "fullres_imageJ.h5",
    model_file: Optional[str] = None,
    force: bool = False,
    device: Optional[str] = None,
    workers: int = 0,
    stop: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    cellpose_model: Optional[str] = None,
    cellpose_diameter: float = 10.0,
    cellpose_min_size: int = 3,
    cellpose_flow_threshold: float = 0.4,
    cellpose_cellprob_threshold: float = 0.0,
    boundary_tolerance: int = 2,
    metric_level: str = "both",
) -> pd.DataFrame:
    """Run one CPIJ validation configuration and save CSV + JSON summary.

    Existing output is reused unless force=True. Avoid reusing results from
    changed model/Cellpose settings; pass force=True or a separate output dir.
    """
    if seg_method not in {"conventional", "inst_seg"}:
        raise ValueError("seg_method must be 'conventional' or 'inst_seg'")
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if cellpose_diameter <= 0:
        raise ValueError("cellpose_diameter must be >0")
    if device is not None and not (device == "cpu" or device.startswith("cuda")):
        raise ValueError("device must be cpu or cuda[:index]")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_h5_path = Path(h5_dir).expanduser().resolve() / gt_h5_name
    ij_h5_path = Path(h5_dir).expanduser().resolve() / imagej_h5_name
    out_csv = out_dir / f"testing_val_CPIJ_{unet_mode}_{seg_method}.csv"
    summary_json = out_dir / f"testing_val_CPIJ_{unet_mode}_{seg_method}_imagewide_summary.json"
    if out_csv.exists() and not force:
        print(f"Reusing existing result: {out_csv} (use --force to recompute)")
        return pd.read_csv(out_csv)

    selected_file = model_file or f"best_{unet_mode}_tiles_S512_seed187.pth"
    segmenter, imagej_segmenter, cfg = _make_segmenter_and_validation_config(
        gt_h5_path=str(gt_h5_path), out_csv=str(out_csv),
        out_summary_json=str(summary_json), unet_mode=unet_mode,
        model_dir=str(Path(model_dir).expanduser().resolve()),
        model_file=selected_file, seg_method=seg_method,
    )
    if device is not None:
        # The original helper selects its device at construction, so the
        # requested device must agree with the already-created objects.
        actual = str(segmenter.device)
        if device.startswith("cuda") and not actual.startswith("cuda"):
            raise RuntimeError(f"Requested {device}, but the UNet uses {actual}")
        if device == "cpu" and actual != "cpu":
            raise RuntimeError("Requested CPU, but original UNet helper selected CUDA")
    cfg.workers = workers
    use_gpu = str(segmenter.device).startswith("cuda")
    cellpose_engine, cp_major, cp_version, cp_label = _make_cellpose(cellpose_model, use_gpu)
    try:
        df, _ = validate_simulated_segmentation(
            segmenter, imagej_segmenter, cellpose_engine,
            cfg=cfg, imagej_h5_path=ij_h5_path,
            cellpose_major=cp_major, cellpose_version=cp_version,
            cellpose_model_label=cp_label,
            cellpose_diameter=cellpose_diameter,
            cellpose_min_size=cellpose_min_size,
            cellpose_flow_threshold=cellpose_flow_threshold,
            cellpose_cellprob_threshold=cellpose_cellprob_threshold,
            boundary_tolerance=boundary_tolerance,
            indices=indices, stop=stop, metric_level=metric_level,
        )
        return df
    finally:
        del cellpose_engine, imagej_segmenter, segmenter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--model-dir", default="./models")
    p.add_argument("--h5-dir", required=True, help="Directory with both fullres_*.h5 files")
    p.add_argument("--unet-mode", choices=("small", "medium", "large"), default="small")
    p.add_argument("--seg-method", choices=("conventional", "inst_seg"), default="inst_seg")
    p.add_argument("--model-file", default=None)
    p.add_argument("--gt-h5-name", default="fullres_ground_truth.h5")
    p.add_argument("--imagej-h5-name", default="fullres_imageJ.h5")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="First N full images")
    p.add_argument("--force", action="store_true", help="Overwrite results")
    p.add_argument("--cellpose-model", default=None)
    p.add_argument("--cellpose-diameter", type=float, default=10.0)
    p.add_argument("--cellpose-min-size", type=int, default=3)
    p.add_argument("--cellpose-flow-threshold", type=float, default=0.4)
    p.add_argument("--cellpose-cellprob-threshold", type=float, default=0.0)
    p.add_argument("--boundary-tolerance", type=int, default=2)
    p.add_argument("--metric-level", choices=("both", "image", "tile"), default="both",
                   help="Default both: image rows plus per-tile crop rows")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    df = run_simulated_segmentation_comparison(
        out_dir=args.out_dir, model_dir=args.model_dir, h5_dir=args.h5_dir,
        unet_mode=args.unet_mode, seg_method=args.seg_method,
        gt_h5_name=args.gt_h5_name, imagej_h5_name=args.imagej_h5_name,
        model_file=args.model_file, force=args.force, device=args.device,
        workers=args.workers, stop=args.limit, cellpose_model=args.cellpose_model,
        cellpose_diameter=args.cellpose_diameter, cellpose_min_size=args.cellpose_min_size,
        cellpose_flow_threshold=args.cellpose_flow_threshold,
        cellpose_cellprob_threshold=args.cellpose_cellprob_threshold,
        boundary_tolerance=args.boundary_tolerance, metric_level=args.metric_level,
    )
    print(f"Saved/loaded {len(df)} rows; {df['sample_idx'].nunique() if len(df) else 0} images")
    print(df[df["metric_level"] == "image"].groupby("dataset_mode")[
        ["instance_f1_iou_0p5", "F1_center_4px", "pixel_dice",
         "mean_instance_dice_iou_0p5", "mean_instance_jaccard_iou_0p5"]
    ].mean().round(4)) if "image" in set(df["metric_level"]) else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
