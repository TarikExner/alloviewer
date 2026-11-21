#!/usr/bin/env python3
"""Compare Cellpose and AlloViewer UNet instances against manual circle ROIs.

Install as ``alloviewer/dev/validation/manual_segmentation_comparison.py``.
Run from the project root, for example:

    python -m alloviewer.dev.validation.manual_segmentation_comparison --dry-run
    python -m alloviewer.dev.validation.manual_segmentation_comparison --limit 3 --save-masks
    python -m alloviewer.dev.validation.manual_segmentation_comparison

Defaults:
    tiles:       ./expert_scoring_tile_exports/score_*/*.tif
                 (also supports ./expert_scoring_tile_exports/zips/score_*.zip)
    circles:     ./manual_segmentations/csv/*.csv
    result:      ./results/manual_segmentation_comparison_v2.csv

Two levels in the same CSV: one row per (tile, human annotator, method),
plus one aggregate per (original image, human annotator, method). The latter
pools confusion counts across MANUALLY ANNOTATED tiles, and separately reconstructs
unique pixels in the annotated image region when the tile manifest is available. Tiles overlap: an aggregate is NOT an
independent reconstructed full-image segmentation and can count the same
cell twice if a cell is annotated in neighboring tiles. Images are never resized. Annotated-region Dice equals binary pixel F1.
A true whole-image score is only valid if the evaluated region covers the entire
original image and the annotations there are exhaustive.
One-to-one IoU F1 is scored at 0.10/0.25/0.30/0.50/0.70/0.75/0.90.
The center-in-mask F1 requires the annotated GT center pixel to lie within
a predicted cell instance (one match per predicted instance).
Manual circles are proxies for cell positions and approximate extents, NOT
hand-traced masks. Use F1_center_4px (4-pixel tolerance) as the primary center-distance metric
for the approximately 8-pixel hand-annotated circles; 6/8 px are sensitivity
checks. Do not confuse this with the independent center_in_mask_f1.
Prefer center-detection F1 for the main comparison; report
pixel Dice/IoU and instance overlap scores as *circle agreement* only.

IMPORTANT: Exhaustiveness of the manually annotated region cannot be inferred
from a list of circles. --region full assumes every cell in the tile was
annotated. --region bbox (default) evaluates only a bounding rectangle around
manual circles, including a configurable margin. A bbox is not proof that
annotation was exhaustive within the rectangle. Check this before publication.

Cellpose 3 uses cyto3 by default; Cellpose 4 uses its installed default
Cellpose-SAM model. Set --cellpose-model to explicitly choose a model, and
record the version/model in the output. A fixed --cellpose-diameter avoids
feeding manual annotations into inference (ground-truth leakage).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import warnings
import zipfile
import pandas as pd
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


REQUIRED_COLUMNS = {
    "image_name", "cell_id", "roi_shape", "roi_x_px", "roi_y_px",
    "diameter_px", "center_x_px", "center_y_px",
}
METHODS = ("AlloViewer_UNet", "Cellpose")
# A fairly small object can span only ~50 pixels in the human reference.
IOU_THRESHOLDS = (0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.90)
CENTER_RADII = (4.0, 6.0, 8.0)


@dataclass(frozen=True)
class TileSource:
    filename: str
    path: Path
    zip_member: str | None = None

    @property
    def display_path(self) -> str:
        return str(self.path) if self.zip_member is None else f"{self.path}::{self.zip_member}"

    def read_image(self) -> tuple[np.ndarray, np.ndarray]:
        """Return raw RGB and AlloViewer-normalized float RGB images."""
        # Lazy import: metric functions and --dry-run do not require PyTorch.
        from ...image_analysis.io.io import load_image, scale_image

        if self.zip_member is None:
            source: Any = self.path.expanduser().resolve(strict=True)
        else:
            with zipfile.ZipFile(self.path) as archive:
                source = io.BytesIO(archive.read(self.zip_member))
        raw, report = load_image(source, scale=False, as_chw=False)
        scaled, _ = scale_image(raw, report, fast=True)
        if raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError(f"Expected RGB TIFF for {self.display_path}, got {raw.shape}")
        return raw, scaled


def find_tiles(root: Path) -> dict[str, TileSource]:
    """Index exported TIFF tiles; prefer extracted folders over ZIP copies."""
    if not root.is_dir():
        raise NotADirectoryError(f"Missing tile directory: {root}")
    found: dict[str, TileSource] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        if path.name in found:
            raise ValueError(f"Duplicate tile filename: {path.name} ({found[path.name].path}, {path})")
        found[path.name] = TileSource(path.name, path)

    zip_root = root / "zips"
    if zip_root.is_dir():
        for archive_path in sorted(zip_root.glob("score_*.zip")):
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    if Path(member).suffix.lower() not in (".tif", ".tiff"):
                        continue
                    name = Path(member).name
                    if name not in found:
                        found[name] = TileSource(name, archive_path, member)
    return found


def list_annotation_groups(csv_dir: Path) -> list[tuple[Path, str, str, list[dict[str, str]]]]:
    """One annotation group per CSV image and annotator; no mixing experts."""
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"Missing annotation directory: {csv_dir}")
    groups: list[tuple[Path, str, str, list[dict[str, str]]]] = []
    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV annotations in {csv_dir}")
    for path in csv_paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
            by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in reader:
                name = Path(row["image_name"].strip()).name
                annotator = row.get("annotator", "unknown").strip() or "unknown"
                if not name:
                    raise ValueError(f"{path}: row lacks image_name")
                by_key.setdefault((name, annotator), []).append(row)
        for (name, annotator), rows in sorted(by_key.items()):
            groups.append((path, name, annotator, rows))
    # Same annotator/image in multiple CSVs must not silently be double-counted.
    seen: dict[tuple[str, str], Path] = {}
    for path, name, annotator, _ in groups:
        key = (name, annotator)
        if key in seen:
            raise ValueError(f"Duplicate annotation group {key}: {seen[key]} and {path}")
        seen[key] = path
    return groups

def _sanitize_manual_annotations(df: pd.DataFrame, image_shape: tuple[int, int]) -> pd.DataFrame:
    """Remove clearly invalid manual circles without touching the raw files.

    Rules:
    - Remove circles that are completely outside the image.
    - Remove exact duplicate circles (same geometry), keeping the first.
    """
    h, w = image_shape[:2]
    out = df.copy()

    # Ensure numeric columns
    for col in ("roi_x_px", "roi_y_px", "center_x_px", "center_y_px", "diameter_px"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # 1) Remove circles completely outside the image
    x0 = out["roi_x_px"].to_numpy(dtype=float)
    y0 = out["roi_y_px"].to_numpy(dtype=float)
    d = out["diameter_px"].to_numpy(dtype=float)

    x1 = x0 + d
    y1 = y0 + d

    fully_outside = (x1 <= 0) | (y1 <= 0) | (x0 >= w) | (y0 >= h)
    out = out.loc[~fully_outside].copy()

    # 2) Remove exact duplicate circles, keep first occurrence
    # "identical" = same circle geometry
    dup_cols = ["roi_x_px", "roi_y_px", "center_x_px", "center_y_px", "diameter_px"]
    out = out.drop_duplicates(subset=dup_cols, keep="first").copy()

    return out

def manual_circle_instances(rows: list[Mapping[str, str]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rasterize ImageJ-style circle/oval ROIs using pixel centers.

    Example: x=464, y=381, diameter=8 gives 52 pixels, consistent with the
    provided CSV. Overlapping circles are assigned to the nearest center so
    each pixel belongs to at most one human instance.
    """
    h, w = shape
    if h < 1 or w < 1:
        raise ValueError(f"Invalid shape: {shape}")
    label_map = np.zeros(shape, dtype=np.int32)
    best_dist = np.full(shape, np.inf, dtype=np.float32)
    used: set[str] = set()
    centers: list[tuple[float, float]] = []  # (y,x)
    diameters: list[float] = []
    labels = Counter()
    clipped = 0
    overlapping_pixels = 0
    for index, row in enumerate(rows, 1):
        cell_id = str(row["cell_id"]).strip()
        if row.get("image_width_px") and row.get("image_height_px"):
            expected_w = int(float(row["image_width_px"]))
            expected_h = int(float(row["image_height_px"]))
            if (expected_h, expected_w) != shape:
                raise ValueError(
                    f"Annotation size {(expected_h, expected_w)} != image {shape}"
                )
        if cell_id in used:
            raise ValueError(f"Repeated cell_id={cell_id} for one annotator/image")
        used.add(cell_id)
        if row["roi_shape"].strip().lower() not in {"circle", "oval", "ellipse"}:
            raise ValueError(f"Unsupported ROI shape {row['roi_shape']!r} for cell {cell_id}")
        x0, y0 = float(row["roi_x_px"]), float(row["roi_y_px"])
        d = float(row["diameter_px"])
        cx, cy = float(row["center_x_px"]), float(row["center_y_px"])
        if not all(map(math.isfinite, (x0, y0, d, cx, cy))) or d <= 0:
            raise ValueError(f"Invalid circle coordinates for cell {cell_id}")
        # center_x and center_y in the sample CSV are x0 + diameter/2.
        if abs(cx - (x0 + d / 2)) > 1.0 or abs(cy - (y0 + d / 2)) > 1.0:
            raise ValueError(f"Inconsistent circle bounds and center for cell {cell_id}")
        left, right = max(0, int(math.floor(x0))), min(w, int(math.ceil(x0 + d)))
        top, bottom = max(0, int(math.floor(y0))), min(h, int(math.ceil(y0 + d)))
        if left >= right or top >= bottom:
            raise ValueError(f"Circle outside image: {cell_id}")
        if x0 < 0 or y0 < 0 or x0 + d > w or y0 + d > h:
            clipped += 1
        yy, xx = np.ogrid[top:bottom, left:right]
        distance = ((xx + 0.5 - cx) / (d / 2)) ** 2 + ((yy + 0.5 - cy) / (d / 2)) ** 2
        within = distance <= 1.0
        if not within.any():
            raise ValueError(f"Circle has no pixels: {cell_id}")
        prior = label_map[top:bottom, left:right]
        best = best_dist[top:bottom, left:right]
        overlapping_pixels += int((within & (prior > 0)).sum())
        update = within & ((prior == 0) | (distance < best))
        prior[update] = index
        best[update] = distance[update]
        centers.append((cy, cx))
        diameters.append(d)
        labels[str(row.get("annotation_class", "unknown")).lower()] += 1
    stats = {
        "n_manual_cells": len(rows),
        "n_manual_clipped_circles": clipped,
        "n_manual_overlapping_pixels": overlapping_pixels,
        "manual_diameter_mean": float(np.mean(diameters)) if diameters else np.nan,
        "manual_annotation_classes": json.dumps(dict(labels), sort_keys=True),
    }
    return label_map, np.asarray(centers, dtype=float).reshape(-1, 2), stats


def evaluation_region(gt: np.ndarray, *, mode: str, margin: int) -> np.ndarray:
    if mode not in {"full", "bbox"}:
        raise ValueError("region must be full or bbox")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if mode == "full":
        return np.ones(gt.shape, dtype=bool)
    ys, xs = np.nonzero(gt)
    if not len(ys):
        raise ValueError("Cannot derive a bounding box from zero human cells")
    top = max(0, int(ys.min()) - margin)
    left = max(0, int(xs.min()) - margin)
    bottom = min(gt.shape[0], int(ys.max()) + margin + 1)
    right = min(gt.shape[1], int(xs.max()) + margin + 1)
    region = np.zeros(gt.shape, dtype=bool)
    region[top:bottom, left:right] = True
    return region


def compact_labels(labels: np.ndarray) -> np.ndarray:
    """Remove negative IDs and map positive instance IDs to consecutive 1..N."""
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label map, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Instance labels must be integer, got {arr.dtype}")
    if (arr < 0).any():
        raise ValueError("Instance labels must not contain negative values")
    vals = np.unique(arr)
    positive = vals[vals > 0]
    return (np.searchsorted(positive, arr, side="left").astype(np.int32) + 1) * (arr != 0)


def _overlap(gt: np.ndarray, pred: np.ndarray):
    """Fast intersection and union matrices for disjoint instance masks."""
    if gt.shape != pred.shape:
        raise ValueError(f"Ground truth/prediction shape mismatch: {gt.shape} vs {pred.shape}")
    ng, npred = int(gt.max()), int(pred.max())
    sizes_gt = np.bincount(gt.ravel(), minlength=ng + 1)[1:]
    sizes_pred = np.bincount(pred.ravel(), minlength=npred + 1)[1:]
    both = (gt > 0) & (pred > 0)
    intersections = np.bincount(
        (gt[both].astype(np.int64) - 1) * npred + (pred[both].astype(np.int64) - 1),
        minlength=ng * npred,
    ).reshape(ng, npred)
    unions = sizes_gt[:, None] + sizes_pred[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros_like(unions, dtype=float), where=unions > 0)
    return iou, intersections, unions, sizes_gt, sizes_pred


def _optimal_pairs(score: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matching: maximize number of valid pairs, then score sum."""
    nr, nc = score.shape
    if nr == 0 or nc == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    # Valid matches carry a cardinality bonus larger than any summed score.
    weights = np.where(valid, (min(nr, nc) + 1) + score, 0.0)
    rows, cols = linear_sum_assignment(-weights)
    keep = valid[rows, cols]
    return rows[keep], cols[keep]


def _centroids(labels: np.ndarray) -> np.ndarray:
    ids = np.arange(1, int(labels.max()) + 1)
    if not len(ids):
        return np.zeros((0, 2), dtype=float)
    mass = np.bincount(labels.ravel(), minlength=len(ids) + 1)[1:]
    yy, xx = np.indices(labels.shape)
    sums_y = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=len(ids) + 1)[1:]
    sums_x = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=len(ids) + 1)[1:]
    return np.column_stack([sums_y / mass, sums_x / mass])


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    # For genuinely empty images, both empty is perfect (nothing to detect).
    if tp == fp == fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return p, r, f1


def _boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance: int) -> tuple[float, float, float]:
    """Binary foreground-boundary agreement within tolerance (circle proxy)."""
    g, p = gt > 0, pred > 0
    structure = np.ones((3, 3), dtype=bool)
    gb = g & ~ndi.binary_erosion(g, structure=structure)
    pb = p & ~ndi.binary_erosion(p, structure=structure)
    if not gb.any() and not pb.any():
        return 1.0, 1.0, 1.0
    if not gb.any() or not pb.any():
        return 0.0, 0.0, 0.0
    dist_gt = ndi.distance_transform_edt(~gb)
    dist_pred = ndi.distance_transform_edt(~pb)
    precision = float((dist_gt[pb] <= tolerance).mean())
    recall = float((dist_pred[gb] <= tolerance).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compare_instances(
    manual_labels: np.ndarray,
    predicted_labels: np.ndarray,
    manual_centers: np.ndarray,
    *,
    region: np.ndarray | None = None,
    boundary_tolerance: int = 2,
    split_merge_fraction: float = 0.10,
) -> dict[str, Any]:
    """Compare instance maps via pixels, object overlaps and cell centers.

    Matching is one-to-one Hungarian assignment. GT circle centers are the
    recorded ROI centers, not mask centroids (allows partially clipped ROIs).
    Instance predictions outside *region* are not scored, and any predicted
    object having at least one pixel inside region is considered for scoring.
    """
    if manual_labels.shape != predicted_labels.shape:
        raise ValueError("Manual and predicted labels have different shapes")
    if region is None:
        region = np.ones(manual_labels.shape, dtype=bool)
    if region.shape != manual_labels.shape or region.dtype != bool:
        raise ValueError("Region must be a boolean mask of the image shape")
    if not 0 <= split_merge_fraction <= 1 or boundary_tolerance < 0:
        raise ValueError("Invalid split/merge fraction or boundary tolerance")
    manual_labels = compact_labels(np.where(region, manual_labels, 0))
    predicted_labels = compact_labels(np.where(region, predicted_labels, 0))
    manual_centers = np.asarray(manual_centers, dtype=float).reshape(-1, 2)
    if len(manual_centers) != manual_labels.max():
        raise ValueError("Manual centers and rasterized instances do not match")

    ng = int(manual_labels.max())
    npred = int(predicted_labels.max())
    gt_pix, pd_pix = manual_labels > 0, predicted_labels > 0
    tp_pix = int((gt_pix & pd_pix).sum())
    fp_pix = int((~gt_pix & pd_pix & region).sum())
    fn_pix = int((gt_pix & ~pd_pix).sum())
    tn_pix = int((~gt_pix & ~pd_pix & region).sum())
    pixel_p, pixel_r, pixel_f1 = _prf(tp_pix, fp_pix, fn_pix)
    pixel_iou = tp_pix / (tp_pix + fp_pix + fn_pix) if (tp_pix + fp_pix + fn_pix) else 1.0
    if ng and npred:
        iou, intersections, unions, garea, parea = _overlap(manual_labels, predicted_labels)
    else:
        iou = np.zeros((ng, npred), float)
        intersections = np.zeros((ng, npred), int)
        unions = np.zeros((ng, npred), int)
        garea = np.bincount(manual_labels.ravel(), minlength=ng + 1)[1:]
        parea = np.bincount(predicted_labels.ravel(), minlength=npred + 1)[1:]

    result: dict[str, Any] = {
        "n_manual": ng, "n_predicted": npred,
        "count_error_signed": npred - ng,
        "count_error_absolute": abs(npred - ng),
        "count_error_relative": (npred - ng) / ng if ng else np.nan,
        "pixel_tp": tp_pix, "pixel_fp": fp_pix, "pixel_fn": fn_pix, "pixel_tn": tn_pix,
        "pixel_precision": pixel_p, "pixel_recall": pixel_r,
        "pixel_dice": pixel_f1, "pixel_jaccard": pixel_iou, "pixel_iou": pixel_iou,
        "pixel_f1": pixel_f1, "pixel_specificity": (tn_pix / (tn_pix + fp_pix)
                                                   if tn_pix + fp_pix else 1.0),
        "pixel_accuracy": ((tp_pix + tn_pix) / region.sum()
                           if region.sum() else np.nan),
        "evaluation_pixels": int(region.sum()),
        "evaluated_fraction": float(region.mean()),
    }
    for thr in IOU_THRESHOLDS:
        r, c = _optimal_pairs(iou, iou >= thr)
        tp, fp, fn = len(r), npred - len(r), ng - len(r)
        p, rec, f1 = _prf(tp, fp, fn)
        suffix = str(thr).replace(".", "p")
        # Matched-object overlap is defined only on true-positive one-to-one
        # pairs. The all_gt variants penalize every missed reference cell with 0.
        matched_iou = iou[r, c]
        matched_dice = 2.0 * matched_iou / (1.0 + matched_iou)
        mean_iou = float(matched_iou.mean()) if len(r) else np.nan
        mean_dice = float(matched_dice.mean()) if len(r) else np.nan
        result.update({
            f"instance_tp_iou_{suffix}": tp,
            f"instance_fp_iou_{suffix}": fp,
            f"instance_fn_iou_{suffix}": fn,
            f"instance_precision_iou_{suffix}": p,
            f"instance_recall_iou_{suffix}": rec,
            f"instance_f1_iou_{suffix}": f1,
            f"matched_mean_iou_{suffix}": mean_iou,
            f"matched_mean_dice_{suffix}": mean_dice,
            f"mean_instance_jaccard_iou_{suffix}": mean_iou,
            f"mean_instance_dice_iou_{suffix}": mean_dice,
            f"mean_instance_jaccard_all_gt_iou_{suffix}":
                float(matched_iou.sum() / ng) if ng else (1.0 if npred == 0 else 0.0),
            f"mean_instance_dice_all_gt_iou_{suffix}":
                float(matched_dice.sum() / ng) if ng else (1.0 if npred == 0 else 0.0),
            f"pq_iou_{suffix}": float(matched_iou.sum() / (tp + 0.5 * fp + 0.5 * fn))
            if (tp + fp + fn) else 1.0,
        })
    # Panoptic quality = recognition quality * segmentation quality;
    # the product is computed directly above (avoids 0/0 cases).
    if ng and npred:
        r, c = _optimal_pairs(iou, iou > 0)
        agi = int(intersections[r, c].sum())
        agu = int(unions[r, c].sum())
        unmatched_g = set(range(ng)) - set(r)
        unmatched_p = set(range(npred)) - set(c)
        agu += int(garea[list(unmatched_g)].sum()) if unmatched_g else 0
        agu += int(parea[list(unmatched_p)].sum()) if unmatched_p else 0
        result["aji_circle"] = agi / agu if agu else 1.0
        result["matched_instance_mean_dice"] = (
            float(np.mean(2 * intersections[r, c] / (garea[r] + parea[c]))) if len(r) else np.nan
        )
    else:
        result["aji_circle"] = 1.0 if ng == npred == 0 else 0.0
        result["matched_instance_mean_dice"] = np.nan

    # Ground-truth-center-in-predicted-instance detection. ImageJ ROI center
    # coordinates use pixel boundaries, so floor(cx), floor(cy) identifies the
    # pixel containing that center. If several GT centers fall in one predicted
    # instance, it contributes at most one TP: merges cannot inflate recall.
    # Conversely, additional fragments without a GT center count as FP.
    center_yx = np.floor(manual_centers).astype(np.int64)
    center_yx[:, 0] = np.clip(center_yx[:, 0], 0, manual_labels.shape[0] - 1)
    center_yx[:, 1] = np.clip(center_yx[:, 1], 0, manual_labels.shape[1] - 1)
    hit_ids = predicted_labels[center_yx[:, 0], center_yx[:, 1]] if ng else np.empty(0, dtype=np.int32)
    # Each GT center maps to at most one predicted object: the number of
    # distinct positive IDs is the maximum-cardinality one-to-one matching.
    hit_count = int(np.unique(hit_ids[hit_ids > 0]).size)
    hit_fp, hit_fn = npred - hit_count, ng - hit_count
    hit_p, hit_r, hit_f1 = _prf(hit_count, hit_fp, hit_fn)
    result.update({
        "center_in_mask_tp": hit_count,
        "center_in_mask_fp": hit_fp,
        "center_in_mask_fn": hit_fn,
        "center_in_mask_precision": hit_p,
        "center_in_mask_recall": hit_r,
        "center_in_mask_f1": hit_f1,
    })

    predicted_centers = _centroids(predicted_labels)
    dist = np.linalg.norm(manual_centers[:, None, :] - predicted_centers[None, :, :], axis=-1)
    for radius in CENTER_RADII:
        r, c = _optimal_pairs(1.0 - dist / (radius + 1.0), dist <= radius)
        tp, fp, fn = len(r), npred - len(r), ng - len(r)
        p, rec, f1 = _prf(tp, fp, fn)
        suffix = str(int(radius))
        result.update({
            f"center_tp_r{suffix}": tp,
            f"center_fp_r{suffix}": fp,
            f"center_fn_r{suffix}": fn,
            f"center_precision_r{suffix}": p,
            f"center_recall_r{suffix}": rec,
            f"center_f1_r{suffix}": f1,
            f"center_mean_error_px_r{suffix}": float(dist[r, c].mean()) if len(r) else np.nan,
        })
    # Primary center-distance metric for ~8-pixel manual circles.
    # Retain center_f1_r4 for backward compatibility with existing analyses.
    result["F1_center_4px"] = result["center_f1_r4"]
    bp, br, bf = _boundary_f1(manual_labels, predicted_labels, boundary_tolerance)
    result.update({
        "boundary_precision_tol_px": bp,
        "boundary_recall_tol_px": br,
        "boundary_f1_tol_px": bf,
        "boundary_tolerance_px": boundary_tolerance,
    })
    if ng and npred:
        fraction_of_gt = intersections / garea[:, None]
        fraction_of_pred = intersections / parea[None, :]
        result["possible_splits"] = int(((fraction_of_gt >= split_merge_fraction).sum(axis=1) > 1).sum())
        result["possible_merges"] = int(((fraction_of_pred >= split_merge_fraction).sum(axis=0) > 1).sum())
    else:
        result["possible_splits"] = 0
        result["possible_merges"] = 0
    return result


def _predict_unet(model: Any, image_scaled: np.ndarray) -> np.ndarray:
    output = model(image_scaled)
    masks = output.get("instance_labels")
    if masks is None:
        raise ValueError("AlloViewer UNet did not return instance_labels")
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return np.asarray(masks)


def _predict_cellpose(model: Any, image_scaled: np.ndarray, *, major: int, diameter: float,
                      min_size: int, flow_threshold: float, cellprob_threshold: float) -> np.ndarray:
    kwargs = {
        "diameter": diameter,
        "channel_axis": -1,
        "normalize": True,
        "min_size": min_size,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
    }
    if major < 4:
        # Cellpose 3 channels indices: 0 grayscale, 1 red, 2 green, 3 blue.
        # Use [0,0] so the same full-color RGB data are mapped to grayscale.
        kwargs["channels"] = [0, 0]
    prediction = model.eval(image_scaled, **kwargs)
    masks = prediction[0] if isinstance(prediction, tuple) else prediction
    if isinstance(masks, (list, tuple)):
        if len(masks) != 1:
            raise ValueError("Cellpose returned more than one mask for one image")
        masks = masks[0]
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return np.asarray(masks)


def _make_unet(unet_mode: str, model_dir: str, model_file: str | None, device: str):
    from ...image_analysis.config import INSTANCE_CONFIG, UNET_CONFIG
    from ...image_analysis.segmenter import SegmenterUNetInference
    config = dict(UNET_CONFIG)
    config.update({
        "unet_mode": unet_mode, "model_dir": model_dir, "model_file": model_file,
        "device": device, "input_is_tiles": False, "compute_instances": True,
        "instance_cfg": INSTANCE_CONFIG.to_dict(), "tile_size": 512, "tile_overlap": 64,
    })
    return SegmenterUNetInference.from_config(config)


def _make_cellpose(model_name: str | None, use_gpu: bool):
    from cellpose import models
    version = metadata.version("cellpose")
    major = int(version.split(".")[0])
    if major < 3:
        raise RuntimeError(f"Cellpose >=3 required, got {version}")
    if major >= 4:
        kwargs: dict[str, Any] = {"gpu": use_gpu}
        if model_name is not None:
            kwargs["pretrained_model"] = model_name
        model = models.CellposeModel(**kwargs)
        model_label = model_name or "installed_cellpose4_default"
    else:
        model_label = model_name or "cyto3"
        model = models.CellposeModel(gpu=use_gpu, model_type=model_label)
    return model, major, version, model_label


# The same schema serves tile rows and source-image aggregate rows.
# Image-level rows report pooled tile confusion counts, not stitched masks.
BASE_FIELDS = (
    "level", "center", "source_image_key", "source_image", "source_image_path",
    "image_name", "image_path", "tile_index", "tile_x0", "tile_y0",
    "n_tiles_annotated", "n_tiles_evaluated", "n_tiles_total",
    "aggregation_note", "annotation_csv", "human_annotator",
    "region_mode", "bbox_margin", "unet_checkpoint", "unet_mode",
    "cellpose_model", "cellpose_version", "cellpose_diameter",
    "status", "error", "method",
)
MANUAL_FIELDS = (
    "n_manual_cells", "n_manual_clipped_circles", "n_manual_overlapping_pixels",
    "manual_diameter_mean", "manual_annotation_classes",
)
# Determine all metric fieldnames from a deterministic one-object perfect example.
METRIC_FIELDS = tuple(compare_instances(
    np.array([[1]], dtype=np.int32), np.array([[1]], dtype=np.int32),
    np.array([[0.5, 0.5]]),
).keys())
IMAGE_STITCH_FIELDS = (
    "image_stitch_status", "annotated_image_dice", "annotated_image_jaccard",
    "annotated_image_pixel_f1", "annotated_image_pixel_tp", "annotated_image_pixel_tn",
    "annotated_image_pixel_fp", "annotated_image_pixel_fn",
    "annotated_image_evaluated_pixels", "annotated_image_coverage_fraction",
    "annotated_image_overlap_pixels", "annotated_image_gt_overlap_conflicts",
    "annotated_image_pred_overlap_conflicts",
    "whole_image_dice", "whole_image_jaccard", "whole_image_pixel_f1",
    "image_pixel_scope",
    "pooled_annotated_image_center_f1_4px",
    "pooled_annotated_image_instance_f1_iou_0p5",
)
OUTPUT_FIELDS = (*BASE_FIELDS, *MANUAL_FIELDS, *METRIC_FIELDS, *IMAGE_STITCH_FIELDS)



def _read_tile_manifest(root: Path, manifest_path: Path | None = None) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Read exporter metadata from CSV or manifests.zip, if present."""
    # Some exports move all TIFFs into an `images/` folder. Look next to it
    # for the original manifest, which is required for overlap-safe image Dice.
    if manifest_path is not None:
        csv_path = Path(manifest_path)
        zip_path = root / "zips" / "manifests.zip"
    else:
        candidates = [root / "manifests" / "tile_manifest.csv"]
        if root.name == "images":
            candidates.append(root.parent / "manifests" / "tile_manifest.csv")
        csv_path = next((q for q in candidates if q.is_file()), candidates[0])
        zip_root = root.parent if root.name == "images" else root
        zip_path = zip_root / "zips" / "manifests.zip"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            candidates = [n for n in archive.namelist()
                          if Path(n).name == "tile_manifest.csv"]
            if len(candidates) != 1:
                raise ValueError(f"Expected one tile_manifest.csv in {zip_path}")
            rows = list(csv.DictReader(io.StringIO(
                archive.read(candidates[0]).decode("utf-8-sig")
            )))
    else:
        warnings.warn("Tile manifest missing: inferring source image from tile filename. "
                      "Exact source-image names and coordinates may be unavailable.")
        return {}, {}

    tile_info: dict[str, dict[str, str]] = {}
    total_by_image: Counter[str] = Counter()
    for row in rows:
        name = Path(row.get("output_relative_path") or row.get("output_name") or "").name
        if not name:
            raise ValueError("Tile manifest has no output_relative_path/output_name")
        if name in tile_info:
            raise ValueError(f"Duplicate filename in tile manifest: {name}")
        site = row["site"].strip()
        source_key = f"{site}/{row['Folder']}/{row['image_name']}"
        row["_source_image_key"] = source_key
        tile_info[name] = row
        total_by_image[source_key] += 1
    return tile_info, dict(total_by_image)


def _tile_identity(name: str, tile_meta: dict[str, str],
                   totals: Mapping[str, int]) -> dict[str, Any]:
    """Identify original microscopy image (not the 512px tile)."""
    if tile_meta:
        source_key = tile_meta["_source_image_key"]
        return dict(
            center=tile_meta["site"], source_image_key=source_key,
            source_image=tile_meta["image_name"],
            source_image_path=tile_meta.get("source_path", ""),
            tile_index=tile_meta.get("tile_index", ""),
            tile_x0=tile_meta.get("x0", ""), tile_y0=tile_meta.get("y0", ""),
            n_tiles_total=totals.get(source_key, ""),
        )
    stem = Path(name).stem
    prefix = re.sub(r"__tile_\d+$", "", stem)
    code = re.match(r"^\d+_(BER|FRA)_", prefix)
    center = {"BER": "Berlin", "FRA": "Frankfurt"}.get(
        code.group(1) if code else "", "unknown"
    )
    index = re.search(r"__tile_(\d+)$", stem)
    return dict(center=center, source_image_key=prefix,
                source_image=prefix, source_image_path="",
                tile_index=index.group(1) if index else "",
                tile_x0="", tile_y0="", n_tiles_total="")


def _safe_float(v: Any) -> float:
    try:
        value = float(v)
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _weighted_mean(rows: list[dict[str, Any]], key: str,
                   weight_key: str | None = None) -> float:
    values = []
    weights = []
    for row in rows:
        value = _safe_float(row.get(key))
        weight = _safe_float(row.get(weight_key)) if weight_key else 1.0
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            values.append(value)
            weights.append(weight)
    return float(np.average(values, weights=weights)) if values else math.nan


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, dtype=bool).ravel())


def _unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def _store_stitch_tile(
    storage: dict[tuple[str, str, str], list[dict[str, Any]]],
    source_key: str, annotator: str, method: str,
    metadata: Mapping[str, Any], gt: np.ndarray,
    predicted: np.ndarray, region: np.ndarray,
) -> None:
    """Compress binary masks for one tile; no duplicate overlap pixels are counted."""
    try:
        h, w = gt.shape
        sh = int(metadata["source_height"])
        sw = int(metadata["source_width"])
        x = int(metadata["x0"])
        y = int(metadata["y0"])
        fallback = str(metadata.get("fallback_pad_resize", "False")).lower()
        if fallback in {"true", "1", "yes"}:
            raise ValueError("source tile was padded/resized")
        if min(sh, sw) <= 0 or min(x, y) < 0 or y + h > sh or x + w > sw:
            raise ValueError("invalid source size/tile coordinates")
        payload = {
            "status": "ok", "src_shape": (sh, sw), "tile_shape": (h, w),
            "x": x, "y": y,
            "gt": _pack_mask(gt > 0),
            "pred": _pack_mask(predicted > 0),
            "region": _pack_mask(region),
        }
    except (ValueError, KeyError, TypeError) as exc:
        payload = {"status": f"skipped_invalid_tile_metadata: {exc}"}
    storage.setdefault((source_key, annotator, method), []).append(payload)


def stitch_annotated_image_pixels(
    tiles: list[dict[str, Any]], *, annotations_exhaustive: bool = False,
) -> dict[str, Any]:
    """Pixel agreement over the UNION of annotated image regions.

    GT circles are unioned in areas shared by tiles. Predictions use the
    first tile (input tile order) for shared evaluated pixels; disagreement
    between predictions in those areas is reported. No unannotated image
    background is added to the confusion counts.
    """
    result: dict[str, Any] = {field: np.nan for field in IMAGE_STITCH_FIELDS}
    if not tiles:
        result["image_stitch_status"] = "skipped_no_tile_masks"
        result["image_pixel_scope"] = "not_evaluated"
        return result
    for tile in tiles:
        if tile.get("status") != "ok":
            result["image_stitch_status"] = tile.get("status") or "skipped_invalid_tile"
            result["image_pixel_scope"] = "not_evaluated"
            return result
    shape = tiles[0]["src_shape"]
    if any(tile["src_shape"] != shape for tile in tiles):
        result["image_stitch_status"] = "skipped_source_dimensions_inconsistent"
        result["image_pixel_scope"] = "not_evaluated"
        return result
    h, w = shape
    evaluated = np.zeros(shape, bool)
    ground_truth = np.zeros(shape, bool)
    predicted = np.zeros(shape, bool)
    overlap = gt_conflicts = pred_conflicts = 0
    for tile in sorted(tiles, key=lambda t: (t["y"], t["x"])):
        th, tw = tile["tile_shape"]
        y, x = tile["y"], tile["x"]
        region = _unpack_mask(tile["region"], (th, tw))
        gt = _unpack_mask(tile["gt"], (th, tw))
        pred = _unpack_mask(tile["pred"], (th, tw))
        view = np.s_[y:y+th, x:x+tw]
        already = evaluated[view]
        old_gt = ground_truth[view]
        old_pred = predicted[view]
        shared = region & already
        overlap += int(shared.sum())
        gt_conflicts += int((shared & (old_gt != gt)).sum())
        pred_conflicts += int((shared & (old_pred != pred)).sum())
        # A manually marked cell may be visible in two overlapping tiles.
        # GT is the union of circle masks, not a sum of tile-level counts.
        old_gt |= gt & region
        new = region & ~already
        old_pred[new] = pred[new]
        already |= region
    tp = int((evaluated & ground_truth & predicted).sum())
    tn = int((evaluated & ~ground_truth & ~predicted).sum())
    fp = int((evaluated & ~ground_truth & predicted).sum())
    fn = int((evaluated & ground_truth & ~predicted).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    is_complete = bool(evaluated.all())
    result.update({
        "image_stitch_status": "ok_complete_pixel_coverage" if is_complete
        else "ok_partial_annotated_region",
        "image_pixel_scope": (
            "entire_original_image_exhaustively_annotated" if is_complete and annotations_exhaustive
            else "entire_original_image_geometry_exhaustiveness_unverified" if is_complete
            else "original_image_annotated_pixels_only"
        ),
        "annotated_image_dice": dice,
        "annotated_image_jaccard": iou,
        "annotated_image_pixel_f1": dice,  # Binary pixel F1 == binary Dice.
        "annotated_image_pixel_tp": tp,
        "annotated_image_pixel_tn": tn,
        "annotated_image_pixel_fp": fp,
        "annotated_image_pixel_fn": fn,
        "annotated_image_evaluated_pixels": int(evaluated.sum()),
        "annotated_image_coverage_fraction": float(evaluated.mean()),
        "annotated_image_overlap_pixels": overlap,
        "annotated_image_gt_overlap_conflicts": gt_conflicts,
        "annotated_image_pred_overlap_conflicts": pred_conflicts,
        "whole_image_dice": dice if is_complete and annotations_exhaustive else np.nan,
        "whole_image_jaccard": iou if is_complete and annotations_exhaustive else np.nan,
        "whole_image_pixel_f1": dice if is_complete and annotations_exhaustive else np.nan,
    })
    return result


def assign_source_image_shards(
    groups: list[tuple[Path, str, str, list[dict[str, str]]]],
    tile_metadata: Mapping[str, dict[str, str]],
    totals: Mapping[str, int],
    num_shards: int,
) -> dict[str, int]:
    """Greedy deterministic group-count balance; all tiles of an image stay together."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    counts = Counter(
        _tile_identity(name, tile_metadata.get(name, {}), totals)["source_image_key"]
        for _, name, _, _ in groups
    )
    assignments: dict[str, int] = {}
    loads = [0] * num_shards
    for source_key, n in sorted(counts.items(), key=lambda it: (-it[1], it[0])):
        shard = min(range(num_shards), key=lambda i: (loads[i], i))
        assignments[source_key] = shard
        loads[shard] += n
    return assignments


def aggregate_image_rows(
    tile_rows: list[dict[str, Any]],
    stitch_payloads: Mapping[tuple[str, str, str], list[dict[str, Any]]] | None = None,
    *, annotations_exhaustive: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate annotated tiles by original image, method and human annotator.

    Sum additive pixel/object confusion counts, then recompute micro-averaged
    precision/recall/F1. Nonadditive metrics are explicitly macro-averaged.
    Overlapping tiles are NOT deduplicated: do not use the resulting number of
    cells as an independent whole-image count.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in tile_rows:
        if row.get("level") != "tile":
            continue
        key = (row["source_image_key"], row["human_annotator"], row["method"])
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    additive = (
        "n_manual_cells", "n_manual_clipped_circles", "n_manual_overlapping_pixels",
        "n_manual", "n_predicted", "count_error_signed", "count_error_absolute",
        "pixel_tp", "pixel_fp", "pixel_fn", "pixel_tn", "evaluation_pixels",
        "possible_splits", "possible_merges",
    )
    for group_key, group in sorted(grouped.items()):
        example = group[0]
        ok = [r for r in group if r.get("status") == "ok"]
        unique_tiles = {r["image_name"] for r in group}
        result: dict[str, Any] = {field: example.get(field, "") for field in BASE_FIELDS}
        result.update({
            "level": "image", "image_name": example["source_image"],
            "image_path": example["source_image_path"],
            "annotation_csv": "", "tile_index": "", "tile_x0": "", "tile_y0": "",
            "n_tiles_annotated": len(unique_tiles),
            "n_tiles_evaluated": len(ok),
            "aggregation_note": (
                "Pooled counts across annotated tiles; no overlap deduplication; "
                "other metrics are tile macro means; not whole-image mask evaluation"
            ),
            "status": "ok" if len(ok) == len(group) else ("partial" if ok else "error"),
            "error": "; ".join(sorted(set(str(r.get("error", "")) for r in group
                                            if r.get("status") != "ok"))),
        })
        if ok:
            for field in additive:
                result[field] = sum(int(float(r[field])) for r in ok)
            # Absolute per-tile count error is summed, not the absolute net error.
            result["count_error_relative"] = (
                result["count_error_signed"] / result["n_manual"]
                if result["n_manual"] else math.nan
            )
            precision, recall, f1 = _prf(result["pixel_tp"], result["pixel_fp"], result["pixel_fn"])
            result["pixel_precision"] = precision
            result["pixel_recall"] = recall
            result["pixel_dice"] = f1
            result["pixel_f1"] = f1
            denom = result["pixel_tp"] + result["pixel_fp"] + result["pixel_fn"]
            result["pixel_iou"] = result["pixel_tp"] / denom if denom else 1.0
            result["pixel_jaccard"] = result["pixel_iou"]
            spec_den = result["pixel_tn"] + result["pixel_fp"]
            result["pixel_specificity"] = result["pixel_tn"] / spec_den if spec_den else 1.0
            result["pixel_accuracy"] = (
                (result["pixel_tp"] + result["pixel_tn"]) / result["evaluation_pixels"]
                if result["evaluation_pixels"] else math.nan
            )
            # Coverage is the fraction of total *tile* pixels evaluated. It is
            # not the fraction of the original full-resolution image reviewed.
            tile_pixels = sum(
                int(float(r["evaluation_pixels"])) / float(r["evaluated_fraction"])
                for r in ok if float(r["evaluated_fraction"]) > 0
            )
            result["evaluated_fraction"] = (
                result["evaluation_pixels"] / tile_pixels if tile_pixels else math.nan
            )
            result["manual_diameter_mean"] = _weighted_mean(
                ok, "manual_diameter_mean", "n_manual_cells"
            )
            manual_classes: Counter[str] = Counter()
            for r in ok:
                for name, count in json.loads(r.get("manual_annotation_classes") or "{}").items():
                    manual_classes[name] += int(count)
            result["manual_annotation_classes"] = json.dumps(dict(manual_classes), sort_keys=True)
            for thr in IOU_THRESHOLDS:
                suffix = str(thr).replace(".", "p")
                for k in ("tp", "fp", "fn"):
                    column = f"instance_{k}_iou_{suffix}"
                    result[column] = sum(int(float(r[column])) for r in ok)
                tp = result[f"instance_tp_iou_{suffix}"]
                fp = result[f"instance_fp_iou_{suffix}"]
                fn = result[f"instance_fn_iou_{suffix}"]
                prec, rec, score = _prf(tp, fp, fn)
                result[f"instance_precision_iou_{suffix}"] = prec
                result[f"instance_recall_iou_{suffix}"] = rec
                result[f"instance_f1_iou_{suffix}"] = score
                result[f"matched_mean_iou_{suffix}"] = _weighted_mean(
                    ok, f"matched_mean_iou_{suffix}", f"instance_tp_iou_{suffix}"
                )
                result[f"matched_mean_dice_{suffix}"] = _weighted_mean(
                    ok, f"matched_mean_dice_{suffix}", f"instance_tp_iou_{suffix}"
                )
                result[f"mean_instance_jaccard_iou_{suffix}"] = result[f"matched_mean_iou_{suffix}"]
                result[f"mean_instance_dice_iou_{suffix}"] = result[f"matched_mean_dice_{suffix}"]
                # True global mean across reference circles (unmatched = zero).
                # Only image aggregates of annotated tiles; overlapping tiles may double count.
                n_gt_total = result["n_manual"]
                result[f"mean_instance_jaccard_all_gt_iou_{suffix}"] = (
                    sum(float(r[f"mean_instance_jaccard_all_gt_iou_{suffix}"]) * int(float(r["n_manual"]))
                        for r in ok if int(float(r["n_manual"])) > 0) / n_gt_total
                    if n_gt_total else (1.0 if result["n_predicted"] == 0 else 0.0)
                )
                result[f"mean_instance_dice_all_gt_iou_{suffix}"] = (
                    sum(float(r[f"mean_instance_dice_all_gt_iou_{suffix}"]) * int(float(r["n_manual"]))
                        for r in ok if int(float(r["n_manual"])) > 0) / n_gt_total
                    if n_gt_total else (1.0 if result["n_predicted"] == 0 else 0.0)
                )
                # PQ denominator and matched IoU sum are additive over tiles.
                pq_denom = tp + 0.5 * fp + 0.5 * fn
                summed_iou = sum(
                    float(r[f"pq_iou_{suffix}"]) * (
                        int(float(r[f"instance_tp_iou_{suffix}"])) +
                        0.5 * int(float(r[f"instance_fp_iou_{suffix}"])) +
                        0.5 * int(float(r[f"instance_fn_iou_{suffix}"]))
                    ) for r in ok
                )
                result[f"pq_iou_{suffix}"] = summed_iou / pq_denom if pq_denom else 1.0
            for key in ("tp", "fp", "fn"):
                result[f"center_in_mask_{key}"] = sum(
                    int(float(r[f"center_in_mask_{key}"])) for r in ok
                )
            p, r, f = _prf(
                result["center_in_mask_tp"], result["center_in_mask_fp"],
                result["center_in_mask_fn"],
            )
            result.update(center_in_mask_precision=p, center_in_mask_recall=r,
                          center_in_mask_f1=f)
            for radius in CENTER_RADII:
                suffix = str(int(radius))
                for k in ("tp", "fp", "fn"):
                    col = f"center_{k}_r{suffix}"
                    result[col] = sum(int(float(r[col])) for r in ok)
                tp = result[f"center_tp_r{suffix}"]
                fp = result[f"center_fp_r{suffix}"]
                fn = result[f"center_fn_r{suffix}"]
                p, r, f = _prf(tp, fp, fn)
                result[f"center_precision_r{suffix}"] = p
                result[f"center_recall_r{suffix}"] = r
                result[f"center_f1_r{suffix}"] = f
                result[f"center_mean_error_px_r{suffix}"] = _weighted_mean(
                    ok, f"center_mean_error_px_r{suffix}", f"center_tp_r{suffix}"
                )
            result["F1_center_4px"] = result["center_f1_r4"]
            for col in ("aji_circle", "matched_instance_mean_dice",
                        "boundary_precision_tol_px", "boundary_recall_tol_px",
                        "boundary_f1_tol_px"):
                result[col] = _weighted_mean(ok, col)
            result["boundary_tolerance_px"] = example.get("boundary_tolerance_px", "")
        if stitch_payloads is not None:
            result.update(stitch_annotated_image_pixels(
                stitch_payloads.get(group_key, []), annotations_exhaustive=annotations_exhaustive
            ))
        else:
            result.update(stitch_annotated_image_pixels([]))
        result["pooled_annotated_image_center_f1_4px"] = result.get("F1_center_4px", np.nan)
        result["pooled_annotated_image_instance_f1_iou_0p5"] = result.get(
            "instance_f1_iou_0p5", np.nan
        )
        aggregates.append(result)
    return aggregates


def supplementary_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per model and level; use image rows for inference.

    Report unweighted macro mean/SD across image rows, and micro totals from
    the explicit confusion columns. A source image may have several manual
    annotators: those observations share an image and are not independent.
    """
    group_by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") == "ok":
            group_by.setdefault((r["level"], r["method"]), []).append(r)
    metrics = (
        "center_in_mask_precision", "center_in_mask_recall", "center_in_mask_f1",
        *(f"instance_f1_iou_{str(x).replace('.', 'p')}" for x in IOU_THRESHOLDS),
        "F1_center_4px", "center_f1_r4", "center_mean_error_px_r4",
        "instance_f1_iou_0p5",
        "instance_precision_iou_0p5", "instance_recall_iou_0p5",
        "mean_instance_dice_iou_0p5", "mean_instance_jaccard_iou_0p5",
        "mean_instance_dice_all_gt_iou_0p5", "mean_instance_jaccard_all_gt_iou_0p5",
        "pixel_dice", "pixel_jaccard", "pq_iou_0p5", "count_error_absolute",
        "annotated_image_dice", "annotated_image_jaccard", "annotated_image_pixel_f1",
        "whole_image_dice", "whole_image_pixel_f1",
    )
    confusion = (
        "pixel_tp", "pixel_tn", "pixel_fp", "pixel_fn",
        "instance_tp_iou_0p5", "instance_fp_iou_0p5", "instance_fn_iou_0p5",
        "center_tp_r4", "center_fp_r4", "center_fn_r4",
        "center_in_mask_tp", "center_in_mask_fp", "center_in_mask_fn",
        *(f"instance_{kind}_iou_{str(x).replace('.', 'p')}"
          for x in IOU_THRESHOLDS for kind in ("tp", "fp", "fn")),
    )
    out = []
    for (level, method), data in sorted(group_by.items()):
        summary: dict[str, Any] = {
            "level": level, "method": method, "n_rows": len(data),
            "n_source_images": len({r["source_image_key"] for r in data}),
            "note": ("Cell F1 pools tile counts; annotated_image_dice/F1 counts each "
                     "evaluated source-image pixel once; no whole-image GT outside "
                     "the reviewed area"),
        }
        for k in metrics:
            values = np.array([_safe_float(r.get(k)) for r in data], dtype=float)
            values = values[np.isfinite(values)]
            summary[f"{k}_mean"] = float(values.mean()) if len(values) else np.nan
            summary[f"{k}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        for k in confusion:
            summary[f"{k}_total"] = sum(int(float(r.get(k, 0))) for r in data)
        # Micro F1 uses pooled cell confusion, not an average of tile scores.
        for metric, tp_key, fp_key, fn_key in (
            ("center_4px", "center_tp_r4", "center_fp_r4", "center_fn_r4"),
            ("center_in_mask", "center_in_mask_tp", "center_in_mask_fp", "center_in_mask_fn"),
            *((f"instance_iou_{str(x).replace('.', 'p')}",
               f"instance_tp_iou_{str(x).replace('.', 'p')}",
               f"instance_fp_iou_{str(x).replace('.', 'p')}",
               f"instance_fn_iou_{str(x).replace('.', 'p')}")
              for x in IOU_THRESHOLDS),
        ):
            precision, recall, f1 = _prf(
                summary[f"{tp_key}_total"],
                summary[f"{fp_key}_total"],
                summary[f"{fn_key}_total"],
            )
            summary[f"{metric}_micro_precision"] = precision
            summary[f"{metric}_micro_recall"] = recall
            summary[f"{metric}_micro_f1"] = f1
        out.append(summary)
    return out


def _write_rows(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(OUTPUT_FIELDS)
    if append and path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            previous = next(reader)
        if previous != columns:
            raise ValueError("Existing CSV schema differs. Use a new --output path")
    with path.open("a" if append and path.exists() else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if f.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", type=Path, default=Path("./expert_scoring_tile_exports"))
    p.add_argument("--annotations", type=Path, default=Path("./manual_segmentations/csv"))
    p.add_argument("--output", type=Path, default=Path("./results/manual_segmentation_comparison_v2.csv"))
    p.add_argument("--region", choices=("bbox", "full"), default="bbox")
    p.add_argument("--bbox-margin", type=int, default=12)
    p.add_argument("--unet-mode", default="small", choices=("small", "medium", "large"))
    p.add_argument("--model-dir", default="./models")
    p.add_argument("--model-file", default="best_small_tiles_S512_seed187.pth")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--cellpose-model", default=None,
                   help="CP3 defaults to cyto3; CP4 uses its installed default model")
    p.add_argument("--cellpose-diameter", type=float, default=10.0,
                   help="Fixed diameter in pixels; do not fit this from the validation CSV")
    p.add_argument("--cellpose-min-size", type=int, default=3)
    p.add_argument("--cellpose-flow-threshold", type=float, default=0.4)
    p.add_argument("--cellpose-cellprob-threshold", type=float, default=0.0)
    p.add_argument("--boundary-tolerance", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Process first N annotation groups")
    p.add_argument("--dry-run", action="store_true", help="List matches without loading models/images")
    p.add_argument("--save-masks", action="store_true", help="Export npz instance masks for audit")
    p.add_argument("--resume", action="store_true", help="Append to CSV, skipping completed image/annotator groups")
    p.add_argument("--fail-fast", action="store_true", help="Raise immediately instead of logging errors")
    p.add_argument("--tile-manifest", type=Path, default=None,
                   help="Optional explicit tile_manifest.csv (needed for overlap-corrected image metrics)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--annotations-exhaustive", action="store_true",
                   help="Only use when EVERY source-image cell has been marked and all pixels evaluated")
    p.add_argument("--merge-shards", type=str, default=None,
                   help="Glob of completed shard CSVs; combine without running inference")
    return p


def run(
    *,
    tiles: Path = Path("./expert_scoring_tile_exports"),
    annotations: Path = Path("./manual_segmentations/csv"),
    output: Path = Path("./results/manual_segmentation_comparison_v2.csv"),
    region_mode: str = "bbox",
    bbox_margin: int = 12,
    unet_mode: str = "small",
    model_dir: str = "./models",
    model_file: str | None = "best_small_tiles_S512_seed187.pth",
    device: str = "cuda",
    cellpose_model: str | None = None,
    cellpose_diameter: float = 10.0,
    cellpose_min_size: int = 3,
    cellpose_flow_threshold: float = 0.4,
    cellpose_cellprob_threshold: float = 0.0,
    boundary_tolerance: int = 2,
    limit: int | None = None,
    dry_run: bool = False,
    save_masks: bool = False,
    resume: bool = False,
    fail_fast: bool = False,
    unet_engine: Any | None = None,
    cellpose_engine: Any | None = None,
    cellpose_major: int | None = None,
    tile_manifest: Path | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    annotations_exhaustive: bool = False,
) -> list[dict[str, Any]]:
    """Run comparison. Optional injected engines permit testing without checkpoints."""
    tiles, annotations, output = Path(tiles), Path(annotations), Path(output)
    if cellpose_diameter <= 0 or bbox_margin < 0 or boundary_tolerance < 0:
        raise ValueError("Invalid diameter, bbox margin, or boundary tolerance")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >=1")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    all_groups = list_annotation_groups(annotations)
    sources = find_tiles(tiles)
    tile_metadata, total_by_image = _read_tile_manifest(tiles, tile_manifest)
    if num_shards > 1:
        assignments = assign_source_image_shards(
            all_groups, tile_metadata, total_by_image, num_shards
        )
        all_groups = [g for g in all_groups if assignments[
            _tile_identity(g[1], tile_metadata.get(g[1], {}), total_by_image)[
                "source_image_key"]] == shard_index]
        print(f"Shard {shard_index + 1}/{num_shards}: {len(all_groups)} annotation groups")
    missing = [(p, n, a) for p, n, a, _ in all_groups if n not in sources]
    print(f"Annotation groups: {len(all_groups)}; exported tiles: {len(sources)}; missing tile matches: {len(missing)}")
    for p, name, annotator in missing[:15]:
        print(f"  MISSING: {name} (annotator={annotator}; {p})")
    if limit is not None:
        all_groups = all_groups[:limit]
    if dry_run:
        for p, name, annotator, rows in all_groups[:25]:
            print(f"  {name} / {annotator}: {len(rows)} circles, {'FOUND' if name in sources else 'MISSING'}")
        return []

    existing_tiles: list[dict[str, Any]] = []
    if resume and output.exists():
        with output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or ()) != list(OUTPUT_FIELDS):
                raise ValueError("Existing output CSV has a different schema; use a new path")
            existing_tiles = [r for r in reader if r.get("level") == "tile"]
        completed: dict[tuple[str, str, str], set[str]] = {}
        for row in existing_tiles:
            if row.get("status") == "ok":
                key = (row["annotation_csv"], row["image_name"], row["human_annotator"])
                completed.setdefault(key, set()).add(row["method"])
        done = {key for key, methods in completed.items() if set(METHODS) <= methods}
        new_groups = [g for g in all_groups if (str(g[0]), g[1], g[2]) not in done]
        rerun = {(str(g[0]), g[1], g[2]) for g in new_groups}
        existing_tiles = [r for r in existing_tiles
                          if (r["annotation_csv"], r["image_name"], r["human_annotator"]) not in rerun]
        all_groups = new_groups
        print(f"After --resume filtering: {len(all_groups)} remaining groups")
    if not all_groups:
        if existing_tiles:
            stitch_payloads: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
            restore_saved_mask_payloads(existing_tiles, output, tile_metadata, stitch_payloads)
            image_rows = aggregate_image_rows(
                existing_tiles, stitch_payloads, annotations_exhaustive=annotations_exhaustive
            )
            _write_rows(output, [*existing_tiles, *image_rows], append=False)
            print(f"Already complete. Wrote tile and image rows to {output}")
        else:
            print("Nothing to process on this shard")
            # Emit a header-only file so shard merging can check completeness.
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(OUTPUT_FIELDS)
        return []
    # Resume rewrites the existing tile-only part first. Old image aggregates
    # are never used as an input to new aggregates.
    if resume and output.exists():
        if existing_tiles:
            _write_rows(output, existing_tiles, append=False)
        else:
            output.unlink()

    if unet_engine is None:
        unet_engine = _make_unet(unet_mode, model_dir, model_file, device)
    if cellpose_engine is None:
        cellpose_engine, cp_major, cp_version, cp_label = _make_cellpose(cellpose_model, device == "cuda")
    else:
        cp_major = cellpose_major if cellpose_major is not None else 4
        cp_version, cp_label = "injected_test_model", "injected_test_model"
    records: list[dict[str, Any]] = []
    stitch_payloads: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    first_batch = not (resume and bool(existing_tiles))
    # Process different human annotators on the same image without rerunning models.
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, (annotation_path, name, annotator, rows) in enumerate(all_groups, 1):
        source = sources.get(name)
        identity = _tile_identity(name, tile_metadata.get(name, {}), total_by_image)
        base: dict[str, Any] = {
            "level": "tile", **identity,
            "annotation_csv": str(annotation_path),
            "image_name": name,
            "human_annotator": annotator,
            "image_path": source.display_path if source else "",
            "n_tiles_annotated": 1,
            "n_tiles_evaluated": 1,
            "aggregation_note": "Direct mask comparison within manually reviewed tile region",
            "region_mode": region_mode,
            "bbox_margin": bbox_margin,
            "unet_checkpoint": str(Path(model_dir) / (model_file or f"best_{unet_mode}.pth")),
            "unet_mode": unet_mode,
            "cellpose_model": cp_label,
            "cellpose_version": cp_version,
            "cellpose_diameter": cellpose_diameter,
            "status": "ok",
            "error": "",
            "method": "",
        }
        outrows: list[dict[str, Any]] = []
        try:
            if source is None:
                raise FileNotFoundError(f"Tile not found for {name}")
            if name not in cache:
                raw, scaled = source.read_image()
                unet_labels = _predict_unet(unet_engine, scaled)
                cp_labels = _predict_cellpose(
                    cellpose_engine, scaled, major=cp_major, diameter=cellpose_diameter,
                    min_size=cellpose_min_size, flow_threshold=cellpose_flow_threshold,
                    cellprob_threshold=cellpose_cellprob_threshold,
                )
                if unet_labels.shape != raw.shape[:2] or cp_labels.shape != raw.shape[:2]:
                    raise ValueError(f"Prediction dimensions differ from image {raw.shape}")
                cache.clear()
                cache[name] = (raw, unet_labels, cp_labels)
            raw, unet_labels, cp_labels = cache[name]

            # Remove identical circles and circles completely outside the tile.
            n_before = len(rows)
            rows = _sanitize_manual_annotations(
                pd.DataFrame(rows), raw.shape[:2]
            ).to_dict("records")

            if len(rows) != n_before:
                print(
                    f"Sanitized {name} / {annotator}: "
                    f"removed {n_before - len(rows)} annotations",
                    flush=True,
                )

            gt, centers, manual_stats = manual_circle_instances(rows, raw.shape[:2])
            coverage = evaluation_region(gt, mode=region_mode, margin=bbox_margin)
            for method, pred in zip(METHODS, (unet_labels, cp_labels)):
                metrics = compare_instances(
                    gt, pred, centers, region=coverage, boundary_tolerance=boundary_tolerance
                )
                outrows.append({
                    **base, "method": method, **manual_stats, **metrics,
                })
                _store_stitch_tile(
                    stitch_payloads, identity["source_image_key"], annotator,
                    method, tile_metadata.get(name, {}), gt, pred, coverage,
                )
            if save_masks:
                mask_dir = output.parent / "manual_segmentation_masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                safe_name = Path(name).stem
                safe_annotator = "".join(c if c.isalnum() or c in "_-" else "_" for c in annotator)
                dest = mask_dir / f"{safe_name}__{safe_annotator}.npz"
                np.savez_compressed(dest, manual=gt, unet=unet_labels,
                                    cellpose=cp_labels, evaluation_region=coverage)
        except Exception as exc:
            if fail_fast:
                raise
            print(f"ERROR: {name} / {annotator}: {exc}", file=sys.stderr)
            outrows = [{**base, "method": method, "status": "error", "error": str(exc),
                        "n_tiles_evaluated": 0} for method in METHODS]
        _write_rows(output, outrows, append=resume or not first_batch)
        first_batch = False
        records.extend(outrows)
        if idx == 1 or idx % 10 == 0 or idx == len(all_groups):
            print(f"Processed {idx}/{len(all_groups)} groups -> {output}", flush=True)
    # Write both levels to the requested single results CSV. No full-image
    # pixels are reread; aggregates are computed from annotated tile rows.
    tile_rows = existing_tiles + records
    if existing_tiles:
        restore_saved_mask_payloads(existing_tiles, output, tile_metadata, stitch_payloads)
    image_rows = aggregate_image_rows(
        tile_rows, stitch_payloads, annotations_exhaustive=annotations_exhaustive
    )
    _write_rows(output, tile_rows + image_rows, append=False)
    summary_rows = supplementary_summary(tile_rows + image_rows)
    if summary_rows:
        summary_path = output.with_name(output.stem + "_supplementary_summary.csv")
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            cols = list(dict.fromkeys(k for r in summary_rows for k in r))
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"DONE: {len(tile_rows)} tile-method rows; {len(image_rows)} "
          f"source-image-method rows -> {output}", flush=True)
    return tile_rows + image_rows


def restore_saved_mask_payloads(
    existing_tiles: list[dict[str, Any]], output: Path,
    tile_metadata: Mapping[str, dict[str, str]],
    stitch_payloads: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> None:
    """Restore unique image/annotator masks for image-level scoring on resume."""
    restored = set()
    for old in existing_tiles:
        identity_key = (old["image_name"], old["human_annotator"])
        if identity_key in restored:
            continue
        restored.add(identity_key)
        safe_name = Path(old["image_name"]).stem
        a = "".join(c if c.isalnum() or c in "_-" else "_"
                    for c in old["human_annotator"])
        mask_path = (output.parent / "manual_segmentation_masks" /
                     f"{safe_name}__{a}.npz")
        if not mask_path.is_file():
            continue
        with np.load(mask_path) as saved:
            for method, field in (("AlloViewer_UNet", "unet"), ("Cellpose", "cellpose")):
                _store_stitch_tile(
                    stitch_payloads, old["source_image_key"], old["human_annotator"],
                    method, tile_metadata.get(old["image_name"], {}),
                    saved["manual"], saved[field], saved["evaluation_region"],
                )


def merge_shard_results(pattern: str, output: Path) -> Path:
    """Combine successful, non-overlapping shard outputs and recalculate summary."""
    import glob
    paths = [Path(x) for x in sorted(glob.glob(pattern))
             if not x.endswith("_supplementary_summary.csv")]
    if not paths:
        raise FileNotFoundError(f"No shard CSV files match: {pattern}")
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing result: {output}")
    rows = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or ()) != list(OUTPUT_FIELDS):
                raise ValueError(f"Shard has wrong CSV schema: {path}")
            for row in reader:
                key = (row["level"], row["source_image_key"], row["image_name"],
                       row["human_annotator"], row["method"])
                if key in seen:
                    raise ValueError(f"Same result in multiple shards: {key}")
                seen.add(key)
                rows.append(row)
    _write_rows(output, rows, append=False)
    supplements = supplementary_summary(rows)
    if supplements:
        summary_path = output.with_name(output.stem + "_supplementary_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            columns = list(dict.fromkeys(k for r in supplements for k in r))
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(supplements)
    print(f"Merged {len(paths)} shards, {len(rows)} rows -> {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    a = make_parser().parse_args(argv)
    if a.merge_shards:
        merge_shard_results(a.merge_shards, a.output)
        return 0
    run(
        tiles=a.tiles, annotations=a.annotations, output=a.output,
        region_mode=a.region, bbox_margin=a.bbox_margin,
        unet_mode=a.unet_mode, model_dir=a.model_dir, model_file=a.model_file,
        device=a.device, cellpose_model=a.cellpose_model,
        cellpose_diameter=a.cellpose_diameter, cellpose_min_size=a.cellpose_min_size,
        cellpose_flow_threshold=a.cellpose_flow_threshold,
        cellpose_cellprob_threshold=a.cellpose_cellprob_threshold,
        boundary_tolerance=a.boundary_tolerance, limit=a.limit,
        dry_run=a.dry_run, save_masks=a.save_masks, resume=a.resume,
        fail_fast=a.fail_fast, tile_manifest=a.tile_manifest,
        num_shards=a.num_shards, shard_index=a.shard_index,
        annotations_exhaustive=a.annotations_exhaustive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
