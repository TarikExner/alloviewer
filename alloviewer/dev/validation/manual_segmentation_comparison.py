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
from collections import Counter
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
IOU_THRESHOLDS = (0.25, 0.50, 0.75)
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
            source: Any = self.path
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
        if not any(part.startswith("score_") for part in path.relative_to(root).parts[:-1]):
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
        "pixel_dice": pixel_f1, "pixel_iou": pixel_iou,
        "evaluation_pixels": int(region.sum()),
        "evaluated_fraction": float(region.mean()),
    }
    for thr in IOU_THRESHOLDS:
        r, c = _optimal_pairs(iou, iou >= thr)
        tp, fp, fn = len(r), npred - len(r), ng - len(r)
        p, rec, f1 = _prf(tp, fp, fn)
        suffix = str(thr).replace(".", "p")
        result.update({
            f"instance_tp_iou_{suffix}": tp,
            f"instance_fp_iou_{suffix}": fp,
            f"instance_fn_iou_{suffix}": fn,
            f"instance_precision_iou_{suffix}": p,
            f"instance_recall_iou_{suffix}": rec,
            f"instance_f1_iou_{suffix}": f1,
            f"matched_mean_iou_{suffix}": float(iou[r, c].mean()) if len(r) else np.nan,
            f"pq_iou_{suffix}": float(iou[r, c].sum() / (tp + 0.5 * fp + 0.5 * fn))
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
OUTPUT_FIELDS = (*BASE_FIELDS, *MANUAL_FIELDS, *METRIC_FIELDS)


def _read_tile_manifest(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Read exporter metadata from CSV or manifests.zip, if present."""
    csv_path = root / "manifests" / "tile_manifest.csv"
    zip_path = root / "zips" / "manifests.zip"
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
        name = Path(row["output_relative_path"]).name
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


def aggregate_image_rows(tile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    for key, group in sorted(grouped.items()):
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
            denom = result["pixel_tp"] + result["pixel_fp"] + result["pixel_fn"]
            result["pixel_iou"] = result["pixel_tp"] / denom if denom else 1.0
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
            for col in ("aji_circle", "matched_instance_mean_dice",
                        "boundary_precision_tol_px", "boundary_recall_tol_px",
                        "boundary_f1_tol_px"):
                result[col] = _weighted_mean(ok, col)
            result["boundary_tolerance_px"] = example.get("boundary_tolerance_px", "")
        aggregates.append(result)
    return aggregates


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
    p.add_argument("--output", type=Path, default=Path("./results/manual_segmentation_comparison.csv"))
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
    return p


def run(
    *,
    tiles: Path = Path("./expert_scoring_tile_exports"),
    annotations: Path = Path("./manual_segmentations/csv"),
    output: Path = Path("./results/manual_segmentation_comparison.csv"),
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
) -> list[dict[str, Any]]:
    """Run comparison. Optional injected engines permit testing without checkpoints."""
    tiles, annotations, output = Path(tiles), Path(annotations), Path(output)
    if cellpose_diameter <= 0 or bbox_margin < 0 or boundary_tolerance < 0:
        raise ValueError("Invalid diameter, bbox margin, or boundary tolerance")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >=1")
    all_groups = list_annotation_groups(annotations)
    sources = find_tiles(tiles)
    tile_metadata, total_by_image = _read_tile_manifest(tiles)
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
            _write_rows(output, [*existing_tiles, *aggregate_image_rows(existing_tiles)], append=False)
            print(f"Already complete. Wrote tile and image rows to {output}")
        else:
            print("Nothing to process")
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
            gt, centers, manual_stats = manual_circle_instances(rows, raw.shape[:2])
            coverage = evaluation_region(gt, mode=region_mode, margin=bbox_margin)
            for method, pred in zip(METHODS, (unet_labels, cp_labels)):
                metrics = compare_instances(
                    gt, pred, centers, region=coverage, boundary_tolerance=boundary_tolerance
                )
                outrows.append({
                    **base, "method": method, **manual_stats, **metrics,
                })
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
    image_rows = aggregate_image_rows(tile_rows)
    _write_rows(output, tile_rows + image_rows, append=False)
    print(f"DONE: {len(tile_rows)} tile-method rows; {len(image_rows)} "
          f"source-image-method rows -> {output}", flush=True)
    return tile_rows + image_rows


def main(argv: list[str] | None = None) -> int:
    a = make_parser().parse_args(argv)
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
        fail_fast=a.fail_fast,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

