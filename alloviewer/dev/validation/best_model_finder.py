from __future__ import annotations

import copy
import glob
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from alloviewer.image_analysis.segmenter import SegmenterConfig, SegmenterUNet

from . import utils as val_utils


UNetMode = Literal["small", "medium", "large"]
DatasetMode = Literal["crop_well_resize", "pad_resize", "tiles"]
CoordinateRounding = Literal["round", "floor", "ceil"]

DEFAULT_UNET_MODE: UNetMode = "small"
DEFAULT_DATASET_MODE: DatasetMode = "tiles"

# Save result CSV/JSON files directly into ./results.
DEFAULT_OUTPUT_DIR = "./results"

DEFAULT_MODEL_DIR = "./models"
DEFAULT_H5_PATH = "./image_datasets/human_annotated_images.h5"
DEFAULT_HUMAN_CSV_DIR = "./human_annotations"
DEFAULT_HUMAN_COUNTS_CSV_FILE = "human_annotations.csv"
DEFAULT_IMAGEJ_POINTS_CSV_FILE = "results.csv"

# In ./human_annotations/results.csv, ImageJ coordinates are stored as X/Y.
DEFAULT_IMAGEJ_X_COL = "X"
DEFAULT_IMAGEJ_Y_COL = "Y"
DEFAULT_IMAGEJ_FILE_COL = "file_name"

KNOWN_DATASET_MODES: Sequence[str] = (
    "crop_well_resize",
    "pad_resize",
    "tiles",
)

# Epoch filter defaults.
#
# The default helper run only considers checkpoints up to epoch 100.
# Set DEFAULT_MAX_EPOCH = None, or pass max_epoch=None, to include all epochs.
DEFAULT_INCLUDE_EPOCHS: Optional[Sequence[int]] = None
DEFAULT_MIN_EPOCH: Optional[int] = None
DEFAULT_MAX_EPOCH: Optional[int] = 250

_EPOCH_RE = re.compile(r"(?:^|[_-])epoch[_-]?(\d+)(?:\D|$)", re.IGNORECASE)


# -------------------------------------------------------------------------
# Small compatibility wrappers around .utils
# -------------------------------------------------------------------------

def _decode_json_maybe(x: Any) -> Dict[str, Any]:
    return val_utils.decode_json_maybe(x)


def _load_human_roi_counts(csv_dir: str) -> pd.DataFrame:
    return val_utils.load_human_roi_counts(csv_dir)


def _count_positive_labels(labels: np.ndarray) -> int:
    fn = getattr(val_utils, "count_positive_labels", None)
    if fn is None:
        fn = getattr(val_utils, "_count_positive_labels", None)
    if fn is not None:
        return int(fn(labels))
    return int(np.unique(labels[labels > 0]).size)


def _extract_image_identity(meta: Dict[str, Any]) -> Tuple[str, str]:
    fn = getattr(val_utils, "extract_image_identity", None)
    if fn is None:
        fn = getattr(val_utils, "_extract_image_identity", None)
    if fn is None:
        raise AttributeError(
            ".utils must define extract_image_identity or _extract_image_identity."
        )
    folder, image_name = fn(meta)
    return str(folder), str(image_name)


def _extract_full_hw(
    meta: Dict[str, Any],
    tile_metas: Sequence[Dict[str, Any]],
    tile_hw: Tuple[int, int],
) -> Tuple[int, int]:
    fn = getattr(val_utils, "extract_full_hw", None)
    if fn is None:
        fn = getattr(val_utils, "_extract_full_hw", None)
    if fn is None:
        raise AttributeError(".utils must define extract_full_hw or _extract_full_hw.")
    full_hw = fn(meta, tile_metas, tile_hw)
    return int(full_hw[0]), int(full_hw[1])


def _stitch_prob_tiles(
    prob_tiles: np.ndarray,
    tile_metas: Sequence[Dict[str, Any]],
    full_hw: Tuple[int, int],
) -> np.ndarray:
    fn = getattr(val_utils, "stitch_prob_tiles", None)
    if fn is None:
        raise AttributeError(".utils must define stitch_prob_tiles.")
    return fn(prob_tiles, tile_metas, full_hw)


def _as_numpy_prediction(x: Any) -> np.ndarray:
    fn = getattr(val_utils, "as_numpy_prediction", None)
    if fn is None:
        fn = getattr(val_utils, "_as_numpy_prediction", None)
    if fn is not None:
        return np.asarray(fn(x))

    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()

    return np.asarray(x)


# -------------------------------------------------------------------------
# Checkpoint discovery
# -------------------------------------------------------------------------

def parse_epoch_from_model_file(path: str) -> Optional[int]:
    """Return epoch number parsed from a checkpoint filename."""
    name = os.path.basename(path)
    m = _EPOCH_RE.search(name)
    if m is None:
        return None
    return int(m.group(1))


def detect_dataset_mode_from_model_file(
    path: str,
    *,
    known_dataset_modes: Sequence[str] = KNOWN_DATASET_MODES,
) -> Optional[str]:
    """
    Return dataset mode detected from a checkpoint filename.

    The function uses substring matching against known dataset mode names.
    This is intentional because checkpoint filenames already encode modes such as
    'crop_well_resize', 'pad_resize', or 'tiles'.
    """
    name = os.path.basename(path)
    matches = [mode for mode in known_dataset_modes if mode in name]

    if not matches:
        return None

    # Prefer the longest match if a future mode name overlaps another.
    matches = sorted(matches, key=len, reverse=True)
    return matches[0]


def epoch_model_pattern(unet_mode: str) -> str:
    """Checkpoint pattern for one model size."""
    return f"{unet_mode}_*_epoch_*.pth"


def _sort_model_paths_by_epoch(paths: Sequence[str]) -> List[str]:
    """Sort checkpoint paths by parsed epoch, then filename."""
    return sorted(
        [str(p) for p in paths],
        key=lambda p: (
            parse_epoch_from_model_file(p)
            if parse_epoch_from_model_file(p) is not None
            else 10**12,
            os.path.basename(p),
        ),
    )


def filter_model_paths_by_dataset_mode(
    model_paths: Sequence[str],
    *,
    dataset_mode: Optional[str] = DEFAULT_DATASET_MODE,
    known_dataset_modes: Sequence[str] = KNOWN_DATASET_MODES,
) -> List[str]:
    """
    Filter checkpoint paths by dataset mode encoded in the filename.

    Parameters
    ----------
    model_paths:
        Checkpoint paths.
    dataset_mode:
        Dataset mode to keep. Defaults to 'tiles'.
        Pass None to keep all dataset modes.
    known_dataset_modes:
        Dataset mode names expected in checkpoint filenames.
    """
    if dataset_mode is None:
        return _sort_model_paths_by_epoch(model_paths)

    dataset_mode = str(dataset_mode)
    if dataset_mode not in set(known_dataset_modes):
        raise ValueError(
            f"Unknown dataset_mode={dataset_mode!r}. "
            f"Expected one of {list(known_dataset_modes)} or None."
        )

    out = [
        str(p)
        for p in model_paths
        if detect_dataset_mode_from_model_file(
            str(p),
            known_dataset_modes=known_dataset_modes,
        )
        == dataset_mode
    ]

    out = _sort_model_paths_by_epoch(out)

    if not out:
        raise FileNotFoundError(
            "No checkpoint files remained after dataset-mode filtering. "
            f"dataset_mode={dataset_mode!r}"
        )

    return out


def filter_model_paths_by_epoch(
    model_paths: Sequence[str],
    *,
    include_epochs: Optional[Sequence[int]] = None,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
) -> List[str]:
    """
    Filter checkpoint paths by parsed epoch.

    Parameters
    ----------
    model_paths:
        Checkpoint paths.
    include_epochs:
        Optional explicit set/list of epochs to keep, e.g. [5, 10, 15, 20, 25, 30, 35, 40].
        If provided, only these epochs are kept.
    min_epoch:
        Optional lower epoch bound, inclusive.
    max_epoch:
        Optional upper epoch bound, inclusive.
    """
    include_set = None
    if include_epochs is not None:
        include_set = {int(e) for e in include_epochs}

    out: List[str] = []

    for p in model_paths:
        ep = parse_epoch_from_model_file(p)
        if ep is None:
            continue

        if include_set is not None and ep not in include_set:
            continue

        if min_epoch is not None and ep < int(min_epoch):
            continue

        if max_epoch is not None and ep > int(max_epoch):
            continue

        out.append(str(p))

    out = _sort_model_paths_by_epoch(out)

    if not out:
        raise FileNotFoundError(
            "No checkpoint files remained after epoch filtering. "
            f"include_epochs={include_epochs}, min_epoch={min_epoch}, max_epoch={max_epoch}"
        )

    return out


def find_epoch_model_files(
    model_dir: str,
    *,
    unet_mode: UNetMode = DEFAULT_UNET_MODE,
    dataset_mode: Optional[DatasetMode] = DEFAULT_DATASET_MODE,
    model_pattern: Optional[str] = None,
    include_epochs: Optional[Sequence[int]] = None,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
) -> List[str]:
    """
    Find saved epoch checkpoints for one UNet size and dataset mode.

    By default this searches only files such as:
      small_*_epoch_0050.pth

    It does not include best_*.pth checkpoints.

    Dataset-mode filtering is applied after file discovery.
    Epoch filtering is then applied after dataset-mode filtering.
    """
    pattern = model_pattern or epoch_model_pattern(unet_mode)
    paths = glob.glob(os.path.join(model_dir, pattern))

    epoch_paths = [p for p in paths if parse_epoch_from_model_file(p) is not None]
    epoch_paths = _sort_model_paths_by_epoch(epoch_paths)

    if not epoch_paths:
        raise FileNotFoundError(
            f"No epoch model files matching pattern '{pattern}' were found in {model_dir}."
        )

    epoch_paths = filter_model_paths_by_dataset_mode(
        epoch_paths,
        dataset_mode=dataset_mode,
    )
    epoch_paths = filter_model_paths_by_epoch(
        epoch_paths,
        include_epochs=include_epochs,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    return epoch_paths


# -------------------------------------------------------------------------
# ImageJ coordinate loading
# -------------------------------------------------------------------------

def _first_existing_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c

    lower_to_original = {str(c).lower(): c for c in columns}
    for c in candidates:
        key = str(c).lower()
        if key in lower_to_original:
            return lower_to_original[key]

    return None


def load_imagej_coordinate_points(
    csv_dir: str,
    *,
    csv_filename: str = DEFAULT_IMAGEJ_POINTS_CSV_FILE,
    folder_col: str = "Folder",
    image_col: Optional[str] = DEFAULT_IMAGEJ_FILE_COL,
    x_col: Optional[str] = DEFAULT_IMAGEJ_X_COL,
    y_col: Optional[str] = DEFAULT_IMAGEJ_Y_COL,
) -> pd.DataFrame:
    """
    Load ImageJ-derived coordinate detections from ./human_annotations/results.csv.

    Expected columns in the current file:
        Folder
        file_name
        X
        Y

    The resulting dataframe standardizes names to:
        Folder, image_name, imagej_point_id, imagej_x, imagej_y

    These are not human point annotations. They are an ImageJ-derived spatial
    proxy and should be reported as such.
    """
    csv_path = os.path.join(csv_dir, csv_filename)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"ImageJ coordinate CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if folder_col not in df.columns:
        raise ValueError(
            f"ImageJ coordinate CSV is missing folder column {folder_col!r}. "
            f"Available columns: {list(df.columns)}"
        )

    if image_col is None or image_col not in df.columns:
        image_col = _first_existing_column(df.columns, ["file_name", "image_name", "filename", "file"])
    if image_col is None or image_col not in df.columns:
        raise ValueError(
            "Could not infer image-name column in ImageJ coordinate CSV. "
            f"Available columns: {list(df.columns)}"
        )

    if x_col is None or x_col not in df.columns:
        x_col = _first_existing_column(df.columns, ["X", "x", "XM", "x_px", "centroid_x", "center_x"])
    if y_col is None or y_col not in df.columns:
        y_col = _first_existing_column(df.columns, ["Y", "y", "YM", "y_px", "centroid_y", "center_y"])

    missing = []
    if x_col is None or x_col not in df.columns:
        missing.append("x coordinate")
    if y_col is None or y_col not in df.columns:
        missing.append("y coordinate")
    if missing:
        raise ValueError(
            "Could not infer ImageJ coordinate columns. "
            f"Missing: {missing}. Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out["Folder"] = out[folder_col].astype(str)
    out["image_name"] = out[image_col].astype(str)
    out["imagej_x"] = pd.to_numeric(out[x_col], errors="coerce")
    out["imagej_y"] = pd.to_numeric(out[y_col], errors="coerce")
    out = out.dropna(subset=["imagej_x", "imagej_y"]).copy()

    out["imagej_point_id"] = out.groupby(["Folder", "image_name"]).cumcount().astype(int)
    out["imagej_x_col"] = str(x_col)
    out["imagej_y_col"] = str(y_col)
    out["imagej_source_csv"] = csv_filename

    keep_cols = [
        "Folder",
        "image_name",
        "imagej_point_id",
        "imagej_x",
        "imagej_y",
        "imagej_x_col",
        "imagej_y_col",
        "imagej_source_csv",
    ]

    optional_cols = [
        c for c in ["mean_red", "mean_green", "mean_blue"]
        if c in out.columns and c not in keep_cols
    ]

    return out[keep_cols + optional_cols].reset_index(drop=True)


def make_imagej_points_lookup(points_df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Map (Folder, image_name) to the corresponding ImageJ point dataframe."""
    lookup: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, grp in points_df.groupby(["Folder", "image_name"], dropna=False):
        folder, image_name = key
        lookup[(str(folder), str(image_name))] = grp.reset_index(drop=True)
    return lookup


# -------------------------------------------------------------------------
# Segmentation and coordinate matching
# -------------------------------------------------------------------------

def _segmenter_cfg_for_model(
    *,
    model_path: str,
    unet_mode: UNetMode,
    base_cfg: Optional[SegmenterConfig],
) -> SegmenterConfig:
    """Copy/create a SegmenterConfig and point it to one checkpoint."""
    if base_cfg is None:
        cfg = SegmenterConfig(
            unet_mode=unet_mode,
            compute_instances=True,
            input_is_tiles=True,
        )
    else:
        cfg = copy.deepcopy(base_cfg)

    cfg.unet_mode = unet_mode
    cfg.compute_instances = True
    cfg.input_is_tiles = True
    cfg.model_dir = os.path.dirname(os.path.abspath(model_path))
    cfg.model_file = os.path.basename(model_path)
    return cfg


def segment_one_h5_entry_with_labels(
    segmenter: SegmenterUNet,
    imgs_tiles: np.ndarray,
    meta: Dict[str, Any],
    *,
    update_cell_mask: bool = False,
) -> Dict[str, Any]:
    """
    Segment one tiled H5 entry and return full-image instance labels.

    This mirrors the count-only segmentation path, but keeps instance_labels so
    ImageJ coordinate hits can be evaluated.
    """
    if imgs_tiles.ndim != 4 or imgs_tiles.shape[1] != 3:
        raise ValueError(
            "Expected imgs_tiles with shape [n_tiles, 3, tile_h, tile_w], "
            f"got {imgs_tiles.shape}."
        )

    tile_metas = meta.get("tiles", None)
    if not isinstance(tile_metas, list) or len(tile_metas) == 0:
        raise ValueError("Metadata must contain a non-empty list under key 'tiles'.")

    n_tiles = len(tile_metas)
    imgs_tiles = imgs_tiles[:n_tiles]

    tiles_t = segmenter._to_tensor_tiles(imgs_tiles)
    probs_t = _as_numpy_prediction(segmenter.predict_tiles(tiles_t))

    full_hw = _extract_full_hw(meta, tile_metas, imgs_tiles.shape[-2:])
    probs_full = _stitch_prob_tiles(probs_t, tile_metas, full_hw)

    if probs_full.shape[0] < 4:
        raise ValueError(
            "Expected at least 4 probability channels: cell, bound, center, energy. "
            f"Got shape {probs_full.shape}."
        )

    if segmenter.inst_seg is None:
        raise RuntimeError(
            "Segmenter has no instance segmenter. Set cfg.compute_instances=True."
        )

    seg_out: Dict[str, Any] = {
        "probs": {
            "cell": probs_full[0],
            "bound": probs_full[1],
            "center": probs_full[2],
            "energy": probs_full[3],
        },
        "instance_labels": None,
        "meta": {},
    }

    seg_out = segmenter.inst_seg(seg_out, update_cell_mask=update_cell_mask)
    labels = np.asarray(seg_out["instance_labels"])

    folder, image_name = _extract_image_identity(meta)

    return {
        "Folder": folder,
        "image_name": image_name,
        "instance_labels": labels,
        "unet_roi_count": _count_positive_labels(labels),
    }


def _round_coordinate(value: float, mode: CoordinateRounding) -> int:
    if mode == "round":
        return int(np.rint(value))
    if mode == "floor":
        return int(np.floor(value))
    if mode == "ceil":
        return int(np.ceil(value))
    raise ValueError(f"Unknown coordinate rounding mode: {mode!r}")


def _safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def _find_instance_at_point(
    labels: np.ndarray,
    *,
    x: float,
    y: float,
    radius_px: int = 0,
    coordinate_rounding: CoordinateRounding = "round",
) -> Dict[str, Any]:
    """
    Find the predicted instance label at or near an ImageJ coordinate.

    Coordinates use image convention:
        x = column
        y = row

    If radius_px == 0, only the rounded coordinate is tested.
    If radius_px > 0, labels are searched inside a circular radius around the
    rounded coordinate, and the nearest positive label pixel is used.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D instance labels, got shape {labels.shape}.")

    h, w = labels.shape
    px = _round_coordinate(float(x), coordinate_rounding)
    py = _round_coordinate(float(y), coordinate_rounding)

    in_bounds = 0 <= px < w and 0 <= py < h

    base = {
        "imagej_px": int(px),
        "imagej_py": int(py),
        "point_in_bounds": bool(in_bounds),
        "point_on_instance": False,
        "matched_instance_id": 0,
        "matched_pixel_x": np.nan,
        "matched_pixel_y": np.nan,
        "match_distance_px": np.nan,
    }

    if not in_bounds:
        return base

    radius_px = int(max(0, radius_px))

    if radius_px == 0:
        lab = int(labels[py, px])
        if lab > 0:
            base.update(
                {
                    "point_on_instance": True,
                    "matched_instance_id": lab,
                    "matched_pixel_x": int(px),
                    "matched_pixel_y": int(py),
                    "match_distance_px": 0.0,
                }
            )
        return base

    y0 = max(0, py - radius_px)
    y1 = min(h, py + radius_px + 1)
    x0 = max(0, px - radius_px)
    x1 = min(w, px + radius_px + 1)

    patch = labels[y0:y1, x0:x1]
    if patch.size == 0:
        return base

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    disk = dist2 <= radius_px ** 2
    positive = (patch > 0) & disk

    if not np.any(positive):
        return base

    candidate_y, candidate_x = np.where(positive)
    candidate_dist2 = dist2[positive]
    best_i = int(np.argmin(candidate_dist2))

    yy_abs = int(candidate_y[best_i] + y0)
    xx_abs = int(candidate_x[best_i] + x0)
    lab = int(labels[yy_abs, xx_abs])

    base.update(
        {
            "point_on_instance": True,
            "matched_instance_id": lab,
            "matched_pixel_x": xx_abs,
            "matched_pixel_y": yy_abs,
            "match_distance_px": float(np.sqrt(candidate_dist2[best_i])),
        }
    )
    return base


def match_imagej_points_to_labels(
    *,
    labels: np.ndarray,
    points_df: pd.DataFrame,
    folder: str,
    image_name: str,
    image_index: int,
    unet_mode: str,
    dataset_mode: Optional[str],
    epoch: int,
    model_file: str,
    model_path: str,
    point_match_radius_px: int = 3,
    coordinate_rounding: CoordinateRounding = "round",
) -> pd.DataFrame:
    """Return one row per ImageJ coordinate with the matched predicted instance id."""
    rows: List[Dict[str, Any]] = []

    for _, point in points_df.iterrows():
        hit = _find_instance_at_point(
            labels,
            x=float(point["imagej_x"]),
            y=float(point["imagej_y"]),
            radius_px=point_match_radius_px,
            coordinate_rounding=coordinate_rounding,
        )

        row = {
            "Folder": folder,
            "image_name": image_name,
            "image_index": int(image_index),
            "unet_mode": unet_mode,
            "dataset_mode": dataset_mode,
            "epoch": int(epoch),
            "model_file": model_file,
            "model_path": model_path,
            "imagej_point_id": point["imagej_point_id"],
            "imagej_x": float(point["imagej_x"]),
            "imagej_y": float(point["imagej_y"]),
            "point_match_radius_px": int(point_match_radius_px),
            "coordinate_rounding": coordinate_rounding,
        }
        row.update(hit)

        for c in ["mean_red", "mean_green", "mean_blue", "imagej_x_col", "imagej_y_col", "imagej_source_csv"]:
            if c in point.index:
                row[c] = point[c]

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_imagej_matches_for_image(
    *,
    labels: np.ndarray,
    point_matches: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Summarize ImageJ-coordinate-to-instance matching for one image.

    The output keeps object-level counts to derive precision, recall, F1/Dice,
    and Jaccard later.

    Definitions:
        predicted instance = one positive label id in the UNet instance label image
        matched predicted instance = positive label id hit by at least one ImageJ point

    Object-level:
        TP = number of unique UNet instances hit by ImageJ coordinates
        FP = UNet instances not hit by any ImageJ coordinate
        FN = ImageJ points not assigned to a unique matched UNet instance

    This penalizes both over-splitting and merging:
        one predicted instance hit by two ImageJ points gives TP=1, FN=1
        two predicted instances for one ImageJ point gives TP=1, FP=1
    """
    pred_ids = np.unique(labels[labels > 0])
    n_pred_instances = int(pred_ids.size)

    n_points = int(point_matches.shape[0])
    if n_points == 0:
        return {
            "imagej_n_points": 0,
            "imagej_n_points_on_instance": 0,
            "imagej_point_recall": np.nan,
            "imagej_n_pred_instances": n_pred_instances,
            "imagej_n_unique_matched_instances": 0,
            "imagej_object_tp": 0,
            "imagej_object_fp": n_pred_instances,
            "imagej_object_fn": 0,
            "imagej_object_precision": np.nan if n_pred_instances == 0 else 0.0,
            "imagej_object_recall": np.nan,
            "imagej_object_f1": np.nan,
            "imagej_object_dice": np.nan,
            "imagej_object_jaccard": np.nan,
        }

    matched = point_matches[point_matches["matched_instance_id"].astype(int) > 0]
    n_points_on_instance = int(matched.shape[0])
    matched_ids = np.unique(matched["matched_instance_id"].to_numpy(dtype=int))
    matched_ids = matched_ids[matched_ids > 0]
    n_unique_matched = int(matched_ids.size)

    tp = n_unique_matched
    fp = max(0, n_pred_instances - tp)
    fn = max(0, n_points - tp)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    jaccard = _safe_div(tp, tp + fp + fn)

    return {
        "imagej_n_points": n_points,
        "imagej_n_points_on_instance": n_points_on_instance,
        "imagej_point_recall": _safe_div(n_points_on_instance, n_points),
        "imagej_n_pred_instances": n_pred_instances,
        "imagej_n_unique_matched_instances": n_unique_matched,
        "imagej_object_tp": int(tp),
        "imagej_object_fp": int(fp),
        "imagej_object_fn": int(fn),
        "imagej_object_precision": precision,
        "imagej_object_recall": recall,
        "imagej_object_f1": f1,
        "imagej_object_dice": f1,
        "imagej_object_jaccard": jaccard,
    }


# -------------------------------------------------------------------------
# Model evaluation
# -------------------------------------------------------------------------

def run_one_model_on_h5(
    *,
    h5_path: str,
    model_path: str,
    unet_mode: UNetMode = DEFAULT_UNET_MODE,
    dataset_mode: Optional[str] = None,
    segmenter_cfg: Optional[SegmenterConfig] = None,
    imagej_points_lookup: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    point_match_radius_px: int = 3,
    coordinate_rounding: CoordinateRounding = "round",
    update_cell_mask: bool = False,
    desc: Optional[str] = None,
    return_point_matches: bool = False,
) -> Any:
    """Run one checkpoint on all entries in the human-count H5 file."""
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    epoch = parse_epoch_from_model_file(model_path)
    if epoch is None:
        raise ValueError(f"Could not parse epoch from model file: {model_path}")

    detected_dataset_mode = detect_dataset_mode_from_model_file(model_path)
    if dataset_mode is None:
        dataset_mode_for_rows = detected_dataset_mode
    else:
        dataset_mode_for_rows = str(dataset_mode)

    cfg = _segmenter_cfg_for_model(
        model_path=model_path,
        unet_mode=unet_mode,
        base_cfg=segmenter_cfg,
    )
    segmenter = SegmenterUNet(cfg)

    image_rows: List[Dict[str, Any]] = []
    point_tables: List[pd.DataFrame] = []

    model_file = os.path.basename(model_path)
    model_path_abs = os.path.abspath(model_path)

    with h5py.File(h5_path, "r") as f:
        if "imgs" not in f or "meta" not in f:
            raise KeyError("H5 file must contain datasets '/imgs' and '/meta'.")

        imgs_ds = f["imgs"]
        meta_ds = f["meta"]

        n_total = int(imgs_ds.shape[0])
        n_written = int(f.attrs.get("written", n_total))
        n_use = min(n_total, n_written)

        iterator = tqdm(
            range(n_use),
            desc=desc or f"Human-count set | {unet_mode} {dataset_mode_for_rows} epoch {epoch}",
            dynamic_ncols=True,
        )

        for i in iterator:
            meta = _decode_json_maybe(meta_ds[i])
            imgs_tiles = imgs_ds[i]

            seg = segment_one_h5_entry_with_labels(
                segmenter,
                imgs_tiles,
                meta,
                update_cell_mask=update_cell_mask,
            )

            folder = str(seg["Folder"])
            image_name = str(seg["image_name"])
            labels = np.asarray(seg["instance_labels"])

            row = {
                "Folder": folder,
                "image_name": image_name,
                "image_index": int(i),
                "unet_mode": unet_mode,
                "dataset_mode": dataset_mode_for_rows,
                "epoch": int(epoch),
                "model_file": model_file,
                "model_path": model_path_abs,
                "unet_roi_count": int(seg["unet_roi_count"]),
            }

            if imagej_points_lookup is not None:
                points = imagej_points_lookup.get(
                    (folder, image_name),
                    pd.DataFrame(
                        columns=[
                            "Folder",
                            "image_name",
                            "imagej_point_id",
                            "imagej_x",
                            "imagej_y",
                        ]
                    ),
                )

                point_df = match_imagej_points_to_labels(
                    labels=labels,
                    points_df=points,
                    folder=folder,
                    image_name=image_name,
                    image_index=int(i),
                    unet_mode=unet_mode,
                    dataset_mode=dataset_mode_for_rows,
                    epoch=int(epoch),
                    model_file=model_file,
                    model_path=model_path_abs,
                    point_match_radius_px=point_match_radius_px,
                    coordinate_rounding=coordinate_rounding,
                )

                row.update(
                    summarize_imagej_matches_for_image(
                        labels=labels,
                        point_matches=point_df,
                    )
                )

                if return_point_matches:
                    point_tables.append(point_df)

            image_rows.append(row)

    del segmenter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_df = pd.DataFrame(image_rows)

    if not return_point_matches:
        return image_df

    if point_tables:
        point_match_df = pd.concat(point_tables, ignore_index=True)
    else:
        point_match_df = pd.DataFrame()

    return image_df, point_match_df


def _safe_corr(x: np.ndarray, y: np.ndarray, *, method: str = "pearson") -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")

    if method == "spearman":
        xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        return float(np.corrcoef(xr, yr)[0, 1])

    return float(np.corrcoef(x, y)[0, 1])


def summarize_imagej_coordinate_agreement(
    per_image_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize ImageJ-coordinate proxy metrics for each checkpoint."""
    required = {
        "unet_mode",
        "dataset_mode",
        "epoch",
        "model_file",
        "model_path",
        "imagej_object_tp",
        "imagej_object_fp",
        "imagej_object_fn",
    }
    missing = required - set(per_image_df.columns)
    if missing:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_cols = ["unet_mode", "dataset_mode", "epoch", "model_file", "model_path"]

    for key, group in per_image_df.groupby(group_cols, dropna=False):
        unet_mode, dataset_mode, epoch, model_file, model_path = key

        tp = int(group["imagej_object_tp"].fillna(0).sum())
        fp = int(group["imagej_object_fp"].fillna(0).sum())
        fn = int(group["imagej_object_fn"].fillna(0).sum())

        n_points = int(group.get("imagej_n_points", pd.Series(dtype=float)).fillna(0).sum())
        n_pred = int(group.get("imagej_n_pred_instances", pd.Series(dtype=float)).fillna(0).sum())
        n_hits = int(group.get("imagej_n_points_on_instance", pd.Series(dtype=float)).fillna(0).sum())
        n_unique = int(group.get("imagej_n_unique_matched_instances", pd.Series(dtype=float)).fillna(0).sum())

        rows.append(
            {
                "unet_mode": unet_mode,
                "dataset_mode": dataset_mode,
                "epoch": epoch,
                "model_file": model_file,
                "model_path": model_path,
                "n_images": int(group.shape[0]),
                "imagej_total_points": n_points,
                "imagej_total_pred_instances": n_pred,
                "imagej_total_points_on_instance": n_hits,
                "imagej_total_unique_matched_instances": n_unique,
                "imagej_object_tp_micro": tp,
                "imagej_object_fp_micro": fp,
                "imagej_object_fn_micro": fn,
                "imagej_point_recall_micro": _safe_div(n_hits, n_points),
                "imagej_object_precision_micro": _safe_div(tp, tp + fp),
                "imagej_object_recall_micro": _safe_div(tp, tp + fn),
                "imagej_object_f1_micro": _safe_div(2 * tp, 2 * tp + fp + fn),
                "imagej_object_dice_micro": _safe_div(2 * tp, 2 * tp + fp + fn),
                "imagej_object_jaccard_micro": _safe_div(tp, tp + fp + fn),
                "imagej_point_recall_macro": float(np.nanmean(group["imagej_point_recall"]))
                if "imagej_point_recall" in group.columns
                else np.nan,
                "imagej_object_precision_macro": float(np.nanmean(group["imagej_object_precision"]))
                if "imagej_object_precision" in group.columns
                else np.nan,
                "imagej_object_recall_macro": float(np.nanmean(group["imagej_object_recall"]))
                if "imagej_object_recall" in group.columns
                else np.nan,
                "imagej_object_f1_macro": float(np.nanmean(group["imagej_object_f1"]))
                if "imagej_object_f1" in group.columns
                else np.nan,
                "imagej_object_jaccard_macro": float(np.nanmean(group["imagej_object_jaccard"]))
                if "imagej_object_jaccard" in group.columns
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset_mode", "epoch"]).reset_index(drop=True)


def summarize_count_errors(
    per_image_df: pd.DataFrame,
    *,
    selection_metric: str = "mae",
) -> pd.DataFrame:
    """Summarize manual-count agreement and ImageJ-coordinate proxy metrics."""
    required = {
        "unet_mode",
        "dataset_mode",
        "epoch",
        "model_file",
        "model_path",
        "human_roi_count",
        "unet_roi_count",
    }
    missing = required - set(per_image_df.columns)
    if missing:
        raise ValueError(f"per_image_df is missing required columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    group_cols = ["unet_mode", "dataset_mode", "epoch", "model_file", "model_path"]

    for key, group in per_image_df.groupby(group_cols, dropna=False):
        unet_mode, dataset_mode, epoch, model_file, model_path = key
        matched = group.dropna(subset=["human_roi_count", "unet_roi_count"]).copy()

        base = {
            "unet_mode": unet_mode,
            "dataset_mode": dataset_mode,
            "epoch": epoch,
            "model_file": model_file,
            "model_path": model_path,
            "n_images_predicted": int(group["unet_roi_count"].notna().sum()),
            "n_images_with_human_count": int(group["human_roi_count"].notna().sum()),
            "n_matched": int(matched.shape[0]),
        }

        if matched.empty:
            row = {
                **base,
                "mean_human_count": np.nan,
                "mean_unet_count": np.nan,
                "bias": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "median_abs_error": np.nan,
                "mean_abs_relative_error": np.nan,
                "pearson_r": np.nan,
                "spearman_r": np.nan,
            }
        else:
            human = matched["human_roi_count"].to_numpy(dtype=float)
            pred = matched["unet_roi_count"].to_numpy(dtype=float)
            err = pred - human
            abs_err = np.abs(err)

            denom = human.copy()
            denom[denom == 0] = np.nan
            abs_rel = abs_err / denom

            row = {
                **base,
                "mean_human_count": float(np.mean(human)),
                "mean_unet_count": float(np.mean(pred)),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "median_abs_error": float(np.median(abs_err)),
                "mean_abs_relative_error": float(np.nanmean(abs_rel)),
                "pearson_r": _safe_corr(human, pred, method="pearson"),
                "spearman_r": _safe_corr(human, pred, method="spearman"),
            }

        rows.append(row)

    summary = pd.DataFrame(rows)

    imagej_summary = summarize_imagej_coordinate_agreement(per_image_df)
    if not imagej_summary.empty:
        summary = summary.merge(
            imagej_summary,
            on=["unet_mode", "dataset_mode", "epoch", "model_file", "model_path"],
            how="left",
            validate="one_to_one",
        )

    if selection_metric not in summary.columns:
        raise ValueError(
            f"selection_metric='{selection_metric}' is not present in summary columns: "
            f"{list(summary.columns)}"
        )

    higher_is_better = {
        "pearson_r",
        "spearman_r",
        "imagej_point_recall_micro",
        "imagej_point_recall_macro",
        "imagej_object_precision_micro",
        "imagej_object_precision_macro",
        "imagej_object_recall_micro",
        "imagej_object_recall_macro",
        "imagej_object_f1_micro",
        "imagej_object_f1_macro",
        "imagej_object_dice_micro",
        "imagej_object_jaccard_micro",
        "imagej_object_jaccard_macro",
    }
    lower_is_better = selection_metric not in higher_is_better
    valid = summary.dropna(subset=[selection_metric]).copy()
    summary["selected"] = False

    if not valid.empty:
        idx = valid[selection_metric].idxmin() if lower_is_better else valid[selection_metric].idxmax()
        summary.loc[idx, "selected"] = True

    return summary.sort_values(
        ["selected", "dataset_mode", "epoch"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def compare_human_counts_for_epoch_models(
    *,
    h5_path: str = DEFAULT_H5_PATH,
    human_csv_dir: str = DEFAULT_HUMAN_CSV_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    unet_mode: UNetMode = DEFAULT_UNET_MODE,
    dataset_mode: Optional[DatasetMode] = DEFAULT_DATASET_MODE,
    model_pattern: Optional[str] = None,
    model_paths: Optional[Sequence[str]] = None,
    include_epochs: Optional[Sequence[int]] = DEFAULT_INCLUDE_EPOCHS,
    min_epoch: Optional[int] = DEFAULT_MIN_EPOCH,
    max_epoch: Optional[int] = DEFAULT_MAX_EPOCH,
    segmenter_cfg: Optional[SegmenterConfig] = None,
    selection_metric: str = "mae",
    update_cell_mask: bool = False,
    imagej_points_csv_filename: str = DEFAULT_IMAGEJ_POINTS_CSV_FILE,
    imagej_folder_col: str = "Folder",
    imagej_image_col: Optional[str] = DEFAULT_IMAGEJ_FILE_COL,
    imagej_x_col: Optional[str] = DEFAULT_IMAGEJ_X_COL,
    imagej_y_col: Optional[str] = DEFAULT_IMAGEJ_Y_COL,
    point_match_radius_px: int = 3,
    coordinate_rounding: CoordinateRounding = "round",
) -> Dict[str, Any]:
    """
    Run saved epoch checkpoints for one UNet size and dataset mode on the human-count image set.

    Writes directly to output_dir:
      - per_image_counts_by_epoch.csv
      - per_imagej_point_matches_by_epoch.csv
      - per_image_imagej_coordinate_metrics_by_epoch.csv
      - summary_imagej_coordinate_agreement_by_epoch.csv
      - summary_by_epoch.csv
      - selected_model.json

    The default selects the checkpoint with lowest MAE versus manual human counts.

    Manual count reference
    ----------------------
    Manual counts are loaded through .utils.load_human_roi_counts from:
        ./human_annotations/human_annotations.csv

    ImageJ coordinate proxy
    -----------------------
    ImageJ coordinate detections are loaded from:
        ./human_annotations/results.csv

    These points are used as a spatial proxy only. The resulting F1/Dice/Jaccard
    metrics should be described as agreement with ImageJ-derived coordinates,
    not as F1 against manual human annotations.
    """
    os.makedirs(output_dir, exist_ok=True)

    if model_paths is None:
        model_paths = find_epoch_model_files(
            model_dir,
            unet_mode=unet_mode,
            dataset_mode=dataset_mode,
            model_pattern=model_pattern,
            include_epochs=include_epochs,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
        )
    else:
        model_paths = _sort_model_paths_by_epoch([str(p) for p in model_paths])
        model_paths = filter_model_paths_by_dataset_mode(
            model_paths,
            dataset_mode=dataset_mode,
        )
        model_paths = filter_model_paths_by_epoch(
            model_paths,
            include_epochs=include_epochs,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
        )

    human_df = _load_human_roi_counts(human_csv_dir).copy()
    human_df["Folder"] = human_df["Folder"].astype(str)
    human_df["image_name"] = human_df["image_name"].astype(str)

    imagej_points = load_imagej_coordinate_points(
        human_csv_dir,
        csv_filename=imagej_points_csv_filename,
        folder_col=imagej_folder_col,
        image_col=imagej_image_col,
        x_col=imagej_x_col,
        y_col=imagej_y_col,
    )
    imagej_points_lookup = make_imagej_points_lookup(imagej_points)

    image_tables: List[pd.DataFrame] = []
    point_tables: List[pd.DataFrame] = []

    print(
        "Running human-count checkpoint comparison with "
        f"{len(model_paths)} checkpoint(s). "
        f"unet_mode={unet_mode}, dataset_mode={dataset_mode}, "
        f"include_epochs={include_epochs}, min_epoch={min_epoch}, max_epoch={max_epoch}, "
        f"output_dir={output_dir}, imagej_points_csv={imagej_points_csv_filename}, "
        f"point_match_radius_px={point_match_radius_px}, "
        f"coordinate_rounding={coordinate_rounding}"
    )

    for model_path in model_paths:
        image_df, point_df = run_one_model_on_h5(
            h5_path=h5_path,
            model_path=model_path,
            unet_mode=unet_mode,
            dataset_mode=dataset_mode,
            segmenter_cfg=segmenter_cfg,
            imagej_points_lookup=imagej_points_lookup,
            point_match_radius_px=point_match_radius_px,
            coordinate_rounding=coordinate_rounding,
            update_cell_mask=update_cell_mask,
            return_point_matches=True,
        )
        image_tables.append(image_df)
        point_tables.append(point_df)

    unet_df = pd.concat(image_tables, ignore_index=True)
    unet_df["Folder"] = unet_df["Folder"].astype(str)
    unet_df["image_name"] = unet_df["image_name"].astype(str)

    if point_tables:
        point_matches = pd.concat(point_tables, ignore_index=True)
    else:
        point_matches = pd.DataFrame()

    per_image = unet_df.merge(
        human_df,
        on=["Folder", "image_name"],
        how="left",
        validate="many_to_one",
    )

    per_image["count_error"] = per_image["unet_roi_count"] - per_image["human_roi_count"]
    per_image["abs_count_error"] = per_image["count_error"].abs()
    per_image["relative_count_error"] = (
        per_image["count_error"] / per_image["human_roi_count"].replace(0, np.nan)
    )
    per_image["abs_relative_count_error"] = per_image["relative_count_error"].abs()

    imagej_summary = summarize_imagej_coordinate_agreement(per_image)
    summary = summarize_count_errors(per_image, selection_metric=selection_metric)

    per_image_csv = os.path.join(output_dir, "per_image_counts_by_epoch.csv")
    point_matches_csv = os.path.join(output_dir, "per_imagej_point_matches_by_epoch.csv")
    imagej_per_image_csv = os.path.join(output_dir, "per_image_imagej_coordinate_metrics_by_epoch.csv")
    imagej_summary_csv = os.path.join(output_dir, "summary_imagej_coordinate_agreement_by_epoch.csv")
    summary_csv = os.path.join(output_dir, "summary_by_epoch.csv")
    selected_json = os.path.join(output_dir, "selected_model.json")

    per_image.sort_values(["dataset_mode", "epoch", "Folder", "image_name"]).to_csv(
        per_image_csv,
        index=False,
    )

    imagej_cols = [
        c for c in per_image.columns
        if c.startswith("imagej_")
        or c in [
            "Folder", "image_name", "image_index", "unet_mode", "dataset_mode",
            "epoch", "model_file", "model_path", "unet_roi_count",
        ]
    ]
    per_image[imagej_cols].sort_values(["dataset_mode", "epoch", "Folder", "image_name"]).to_csv(
        imagej_per_image_csv,
        index=False,
    )

    if not point_matches.empty:
        point_matches.sort_values(
            ["dataset_mode", "epoch", "Folder", "image_name", "imagej_point_id"]
        ).to_csv(point_matches_csv, index=False)
    else:
        point_matches.to_csv(point_matches_csv, index=False)

    imagej_summary.to_csv(imagej_summary_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    selected = summary.loc[summary["selected"]].head(1)
    filter_info = {
        "dataset_mode": dataset_mode,
        "include_epochs": None if include_epochs is None else [int(e) for e in include_epochs],
        "min_epoch": None if min_epoch is None else int(min_epoch),
        "max_epoch": None if max_epoch is None else int(max_epoch),
        "n_model_paths": int(len(model_paths)),
        "model_files": [os.path.basename(p) for p in model_paths],
        "manual_count_csv": DEFAULT_HUMAN_COUNTS_CSV_FILE,
        "imagej_points_csv_filename": imagej_points_csv_filename,
        "imagej_folder_col": imagej_folder_col,
        "imagej_image_col": imagej_image_col,
        "imagej_x_col": imagej_x_col,
        "imagej_y_col": imagej_y_col,
        "point_match_radius_px": int(point_match_radius_px),
        "coordinate_rounding": coordinate_rounding,
    }

    if selected.empty:
        selected_info: Dict[str, Any] = {
            "selection_metric": selection_metric,
            "unet_mode": unet_mode,
            "filter": filter_info,
            "selected": None,
            "reason": "No model had a finite selection metric.",
        }
    else:
        row = selected.iloc[0].to_dict()
        selected_info = {
            "selection_metric": selection_metric,
            "unet_mode": unet_mode,
            "dataset_mode": row.get("dataset_mode"),
            "filter": filter_info,
            "selected_epoch": None if pd.isna(row.get("epoch")) else int(row["epoch"]),
            "selected_model_file": row.get("model_file"),
            "selected_model_path": row.get("model_path"),
            "selected_metric_value": (
                None if pd.isna(row.get(selection_metric)) else float(row[selection_metric])
            ),
            "row": {
                k: (None if pd.isna(v) else v)
                for k, v in row.items()
                if k != "selected"
            },
        }

    with open(selected_json, "w", encoding="utf-8") as f:
        json.dump(selected_info, f, indent=2)

    return {
        "per_image": per_image,
        "imagej_point_matches": point_matches,
        "imagej_summary": imagej_summary,
        "summary": summary,
        "selected": selected_info,
        "paths": {
            "per_image_csv": per_image_csv,
            "imagej_point_matches_csv": point_matches_csv,
            "imagej_per_image_csv": imagej_per_image_csv,
            "imagej_summary_csv": imagej_summary_csv,
            "summary_csv": summary_csv,
            "selected_json": selected_json,
        },
    }


def run_epoch_model_human_count_comparison(
    unet_mode: UNetMode = DEFAULT_UNET_MODE,
    dataset_mode: Optional[DatasetMode] = DEFAULT_DATASET_MODE,
    include_epochs: Optional[Sequence[int]] = DEFAULT_INCLUDE_EPOCHS,
    min_epoch: Optional[int] = DEFAULT_MIN_EPOCH,
    max_epoch: Optional[int] = DEFAULT_MAX_EPOCH,
    selection_metric: str = "mae",
    point_match_radius_px: int = 3,
    coordinate_rounding: CoordinateRounding = "round",
):
    """
    Convenience runner.

    Defaults:
      unet_mode='small'
      dataset_mode='tiles'
      max_epoch=100
      output_dir='./results'
      ImageJ coordinate file='./human_annotations/results.csv'
      ImageJ x/y columns='X'/'Y'

    Pass dataset_mode=None to include all dataset modes.
    Pass max_epoch=None to include all epoch checkpoints.

    For coordinate matching tolerance, set point_match_radius_px.
    """
    cfg = SegmenterConfig(
        unet_mode=unet_mode,
        model_dir=DEFAULT_MODEL_DIR,
        model_file="",  # overwritten for each epoch checkpoint
        device="cuda",
        use_amp=True,
        compute_instances=True,
        input_is_tiles=True,
        normalize=True,
        instance_cfg={},
        thr_cell=0.1,
        thr_bound=0.1,
    )

    out = compare_human_counts_for_epoch_models(
        h5_path=DEFAULT_H5_PATH,
        human_csv_dir=DEFAULT_HUMAN_CSV_DIR,
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        unet_mode=unet_mode,
        dataset_mode=dataset_mode,
        model_pattern=None,  # defaults to f"{unet_mode}_*_epoch_*.pth"
        include_epochs=include_epochs,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        segmenter_cfg=cfg,
        selection_metric=selection_metric,
        update_cell_mask=False,
        imagej_points_csv_filename=DEFAULT_IMAGEJ_POINTS_CSV_FILE,
        imagej_folder_col="Folder",
        imagej_image_col=DEFAULT_IMAGEJ_FILE_COL,
        imagej_x_col=DEFAULT_IMAGEJ_X_COL,
        imagej_y_col=DEFAULT_IMAGEJ_Y_COL,
        point_match_radius_px=point_match_radius_px,
        coordinate_rounding=coordinate_rounding,
    )

    print(out["summary"].head())
    print("\nSelected model:")
    print(json.dumps(out["selected"], indent=2))
    print("\nSaved:")
    for p in out["paths"].values():
        print(p)

    return out

