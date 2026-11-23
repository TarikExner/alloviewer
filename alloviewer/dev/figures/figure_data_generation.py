from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal, Iterable
from typing import Mapping, Any, Dict

import numpy as np
import pandas as pd
import json
import h5py
import cv2

import re
import glob

from .figure_data_utils import (
    check_for_file,
    config_to_kwargs_image_sim,
    merge_kwargs_image_sim
)

from alloviewer.image_analysis.segmenter import SegmenterUNetInference
from alloviewer.image_analysis.config import UNET_CONFIG, INSTANCE_CONFIG_DICT
from alloviewer.image_analysis.io import load_image
from alloviewer.dev.segmentation.image_simulation import (
    simulate_image, default_scene
)
import copy
from pathlib import Path as _SupplementPath
from typing import Any as _SupplementAny

import numpy as _supp_np
import pandas as _supp_pd


SUPPLEMENT_METHODS = ("UNet", "Cellpose")

SIMULATION_IMAGE_SIZE = 512
SIMULATION_TILE_SIZE = 128
SIMULATION_SEED = 42
SIMULATION_LABEL_Y_OFFSET = 8

_LOG_RE = re.compile(
    r"log_(?P<unet_mode>small|medium|large)_(?P<tag>(pad_resize|crop_well_resize|tiles)_S(?P<target>\d+)_seed(?P<seed>\d+))\.jsonl$"
)


def make_simulation_parameter_mosaic(
    H: int = SIMULATION_IMAGE_SIZE,
    W: int = SIMULATION_IMAGE_SIZE,
    crop_size: int = SIMULATION_TILE_SIZE,
    seed: int = SIMULATION_SEED,
    same_seed_per_tile: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Create a 4 x 4 simulation parameter mosaic.

    Each panel is generated from a full simulated image. The crop position
    matches the panel position in the final mosaic, keeping the well ring
    aligned between panels.

    Notes
    -----
    - Clustering is disabled for all panels.
    - The calibrated diameter model is disabled because several panels
      directly test `cell_diameter`.
    - sigma_in and sigma_out are interpreted as fractions of cell diameter.
    """
    if H != W:
        raise ValueError("This function expects H == W.")

    if H != 4 * crop_size:
        raise ValueError("Use H = W = 4 * crop_size.")

    rng = np.random.default_rng(seed)

    base_cfg = default_scene()
    base_params = base_cfg.sample_kwargs(rng)

    base_params.update(
        dict(
            H=H,
            W=W,
            return_targets=False,
            return_aux_targets=False,

            # -------------------------------------------------------------
            # Fixed geometry
            # -------------------------------------------------------------
            well_radius_frac=0.46,
            well_center_jitter=0.0,

            # -------------------------------------------------------------
            # Baseline well appearance
            # -------------------------------------------------------------
            background_level=0.08,
            edge_boost=0.25,
            radial_gamma=1.2,
            vignette_strength=0.20,

            background_texture_enable=True,
            background_texture_sigma_fine=0.6,
            background_texture_sigma_coarse=2.0,
            background_texture_fine_weight=0.95,
            background_texture_coarse_weight=0.04,
            background_texture_strength=0.03,
            background_texture_clip=(0.1, 1.6),

            # -------------------------------------------------------------
            # Baseline cells
            # -------------------------------------------------------------
            n_cells=900,

            # Disable calibrated bounds because some panels directly vary
            # cell_diameter and large_cell_diameter_factor.
            cell_diameter_bounds_by_short_side=None,

            cell_diameter=10.5,
            cell_diameter_reference_short_side=1620.0,
            cell_diameter_size_exponent=0.95,
            cell_diameter_scale_clip=(0.60, 2.20),

            large_cell_frac=0.10,
            large_cell_diameter_factor=1.5,

            cell_ellipse_enable=True,
            cell_axis_jitter=0.12,
            cell_random_rotation=True,
            cell_intensity_range=(0.70, 1.05),

            frac_positive=0.5,
            color_jitter=0.08,

            # Fractions of cell-core diameter.
            sigma_in=(0.2, 0.22),
            sigma_out=(0.2, 0.22),
            focus_frac_in=1.0,
            in_focus_sigma_thresh=None,

            # -------------------------------------------------------------
            # Disable clustering
            # -------------------------------------------------------------
            cluster_enable=False,
            clustered_cell_frac=0.0,

            # -------------------------------------------------------------
            # Baseline placement
            # -------------------------------------------------------------
            rim_bias=0.70,
            rim_band=0.20,
            edge_clamp=0.30,

            min_cell_sep_px=None,
            rim_min_sep_px=8,
            pack_iters=15,
            pack_strength=0.45,
            wall_margin_px=5.0,

            side_bias_enable=False,
            side_bias_theta=0.0,
            side_bias_strength=0.70,
            side_bias_kappa=5.0,
            side_bias_inner_frac=0.5,

            # -------------------------------------------------------------
            # Baseline wall and artifacts
            # -------------------------------------------------------------
            wall_blur_sigma=12.0,
            ring_artifacts=1,
            ring_sigma_range=(6.0, 18.0),
            ring_alpha_range=(0.01, 0.10),

            ghost_enable=True,
            ghost_density=0.25,
            ghost_offset_px=25.0,
            ghost_offset_jitter=5.0,
            ghost_sigma=(2.5, 6.0),
            ghost_dilate=1.0,
            ghost_intensity=(0.02, 0.08),
            ghost_stretch=1.5,
            ghost_trail=2,
            ghost_trail_decay=0.6,

            dirt_density=0.00002,
            dirt_size=(4, 12),
            dirt_sigma=(0.0, 2.0),
            dirt_alpha=(0.1, 1.0),

            reflect_enable=True,
            reflect_n=4,
            reflect_theta_sigma=0.12,
            reflect_radial_sigma=10.0,
            reflect_offset_range=(10.0, 70.0),
            reflect_alpha_range=(0.05, 0.16),
            reflect_wobble=0.4,
            reflect_harmonics=2,
            reflect_harmonic_decay=0.55,
        )
    )

    tile_specs = [
        # =============================================================
        # Row 0
        # =============================================================
        dict(
            label="ghost artifacts",
            params=dict(
                ghost_enable=True,
                ghost_density=1.0,
                ghost_offset_px=35.0,
                ghost_offset_jitter=8.0,
                ghost_intensity=(0.12, 0.30),
                ghost_stretch=3.0,
                ghost_trail=3,
            ),
        ),
        dict(
            label="wall reflections",
            params=dict(
                n_cells=700,
                reflect_enable=True,
                reflect_n=18,
                reflect_theta_sigma=0.24,
                reflect_radial_sigma=20.0,
                reflect_offset_range=(6.0, 45.0),
                reflect_alpha_range=(0.25, 0.45),
                reflect_wobble=0.9,
                reflect_harmonics=4,
                reflect_harmonic_decay=0.75,
                ghost_density=0.05,
                ring_artifacts=0,
            ),
        ),
        dict(
            label="wall artifacts",
            params=dict(
                n_cells=750,
                wall_blur_sigma=24.0,
                ring_artifacts=7,
                ring_sigma_range=(4.0, 14.0),
                ring_alpha_range=(0.16, 0.30),
                reflect_n=2,
                ghost_density=0.05,
            ),
        ),
        dict(
            label="illumination falloff",
            params=dict(
                background_level=0.035,
                edge_boost=0.15,
                radial_gamma=1.8,
                vignette_strength=0.80,
                n_cells=450,
            ),
        ),

        # =============================================================
        # Row 1
        # =============================================================
        dict(
            label="side bias",
            params=dict(
                side_bias_enable=True,
                side_bias_theta=np.pi,
                side_bias_strength=0.90,
                side_bias_kappa=8.0,
                side_bias_inner_frac=0.45,
                rim_bias=0.90,
                rim_band=0.25,
                n_cells=1100,
            ),
        ),
        dict(
            label="cell diameter",
            params=dict(
                n_cells=280,
                cell_diameter=18.0,
                large_cell_frac=0.0,
                rim_bias=0.35,
                pack_iters=20,
            ),
        ),
        dict(
            label="cell count",
            params=dict(
                n_cells=2000,
                cell_diameter=7.0,
                large_cell_frac=0.0,
                rim_bias=0.55,
                rim_band=0.20,
                edge_clamp=0.25,
                pack_iters=20,
            ),
        ),
        dict(
            label="rim bias",
            params=dict(
                n_cells=1200,
                rim_bias=0.95,
                rim_band=0.45,
                edge_clamp=0.65,
            ),
        ),

        # =============================================================
        # Row 2
        # =============================================================
        dict(
            label="class ratio",
            params=dict(
                n_cells=700,
                frac_positive=0.98,
                color_jitter=0.03,
                large_cell_frac=0.0,
                rim_bias=0.45,
            ),
        ),
        dict(
            label="cell focus",
            params=dict(
                n_cells=800,
                focus_frac_in=0.35,

                # Fractions of cell diameter.
                sigma_in=(0.05, 0.09),
                sigma_out=(0.16, 0.30),

                rim_bias=0.50,
            ),
        ),
        dict(
            label="large cells",
            params=dict(
                n_cells=650,
                cell_diameter=10.5,
                large_cell_frac=0.45,
                large_cell_diameter_factor=2.0,
                rim_bias=0.50,
            ),
        ),
        dict(
            label="debris",
            params=dict(
                n_cells=250,
                dirt_density=0.0012,
                dirt_size=(4, 14),
                dirt_sigma=(0.4, 1.6),
                dirt_alpha=(0.75, 1.0),
                rim_bias=0.25,
                background_texture_strength=0.015,
            ),
        ),

        # =============================================================
        # Row 3
        # =============================================================
        dict(
            label="texture",
            params=dict(
                n_cells=180,
                background_texture_enable=True,
                background_texture_sigma_fine=0.8,
                background_texture_sigma_coarse=4.0,
                background_texture_fine_weight=0.55,
                background_texture_coarse_weight=0.45,
                background_texture_strength=0.18,
                background_texture_clip=(0.05, 2.0),
                rim_bias=0.20,
            ),
        ),
        dict(
            label="radial brightness",
            params=dict(
                n_cells=600,
                background_level=0.04,
                edge_boost=0.50,
                radial_gamma=0.75,
            ),
        ),
        dict(
            label="cell shape",
            params=dict(
                n_cells=420,
                cell_diameter=15.0,
                cell_ellipse_enable=True,
                cell_axis_jitter=0.65,
                cell_random_rotation=True,
                large_cell_frac=0.0,
                rim_bias=0.35,
                pack_iters=20,
            ),
        ),
        dict(
            label="baseline",
            params=dict(),
        ),
    ]

    n_rows = 4
    n_cols = 4

    if len(tile_specs) != n_rows * n_cols:
        raise RuntimeError(
            f"Expected {n_rows * n_cols} tile specifications, "
            f"got {len(tile_specs)}."
        )

    mosaic = np.zeros((H, W, 3), dtype=np.float32)
    tile_info: list[dict[str, Any]] = []

    for i, spec in enumerate(tile_specs):
        row = i // n_cols
        col = i % n_cols

        params = base_params.copy()
        params.update(spec["params"])

        # Enforce no clusters even if a panel is edited later.
        params["cluster_enable"] = False
        params["clustered_cell_frac"] = 0.0

        if same_seed_per_tile:
            params["seed"] = seed
        else:
            params["seed"] = seed + i * 1009

        img, meta, _ = simulate_image(**params)

        img = np.asarray(img, dtype=np.float32)
        img = np.clip(img, 0.0, 1.0)

        y0 = row * crop_size
        y1 = y0 + crop_size
        x0 = col * crop_size
        x1 = x0 + crop_size

        crop = img[y0:y1, x0:x1]

        expected_shape = (crop_size, crop_size, 3)
        if crop.shape != expected_shape:
            raise RuntimeError(
                f"Panel {spec['label']!r} produced crop shape "
                f"{crop.shape}, expected {expected_shape}."
            )

        mosaic[y0:y1, x0:x1] = crop

        tile_info.append(
            dict(
                label=spec["label"],
                row=row,
                col=col,
                source_crop=(y0, y1, x0, x1),
                params=dict(params),
                changed_params=dict(spec["params"]),
                meta=meta,
            )
        )

    return mosaic, tile_info


def _postprocess_to_rgb(img: np.ndarray,
                        tile_idx: Optional[int]) -> np.ndarray:
    """Ensure image is HxWx3 uint8 RGB for matplotlib."""
    arr = np.asarray(img)

    if arr.ndim == 4:
        if arr.shape[0] == 20:
            assert tile_idx is not None, "Tile IDX has to be provided"
            arr = arr[tile_idx]
        else:
            assert arr.shape[0] == 1
            arr = np.squeeze(arr, axis=0)

    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # Grayscale -> 3 channels
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # to uint8 0..255
    if arr.dtype != np.uint8:
        vmax = float(arr.max()) if arr.size else 1.0
        if vmax <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        elif vmax <= 255.0:
            arr = arr.astype(np.uint8)
        else:
            arr = (arr / 65535.0 * 255.0).astype(np.uint8)
    return arr


def _cv2_resize_rgb(img_rgb_uint8: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize RGB uint8 to (H, W)."""
    h, w = size
    return cv2.resize(img_rgb_uint8, (w, h), interpolation=cv2.INTER_LINEAR)


def _cv2_resize_mask(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize masks/targets to (H, W) with nearest neighbor."""
    Ht, Wt = size
    if arr.ndim == 2:
        return cv2.resize(arr, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

    if arr.ndim == 3:
        # channels-first -> to HWC
        if arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
            arr_hwC = np.transpose(arr, (1, 2, 0))
            chw = True
        else:
            arr_hwC = arr
            chw = False

        C = arr_hwC.shape[-1]
        out = np.empty((Ht, Wt, C), dtype=arr_hwC.dtype)
        for ci in range(C):
            out[..., ci] = cv2.resize(arr_hwC[..., ci], (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        if chw:
            out = np.transpose(out, (2, 0, 1))
        return out

    return arr

def fetch_item(
    h5_path: str,
    index: int,
    tile_idx: Optional[int],
    *,
    image_key: str = "imgs",
    target_key: Optional[str] = "tgts",     # set None to skip targets
    resize_to: Optional[Tuple[int, int]] = None,          # (H, W) for image
    target_resize_to: Optional[Tuple[int, int]] = None,   # (H, W) for targets; defaults to image size
    return_channel_last: bool = True,       # targets -> (H,W,C) if True else (C,H,W)
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Core loader: returns (rgb, tgts_or_none).
    Expects targets with shape (1,4,H,W) at target_key if present.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        imgs = h5[image_key]
        n = len(imgs)
        idx = index if index >= 0 else n + index
        if not (0 <= idx < n):
            raise IndexError(f"Index {index} out of range for dataset length {n}.")

        # image
        img_raw = np.asarray(imgs[idx])
        rgb = _postprocess_to_rgb(img_raw, tile_idx)
        if resize_to is not None:
            rgb = _cv2_resize_rgb(rgb, resize_to)

        # targets (optional)
        tgts_out: Optional[np.ndarray] = None
        if target_key is not None:
            if target_key not in h5:
                raise KeyError(f"Targets key '{target_key}' not found in file.")
            tgts_ds = h5[target_key]
            tgt_raw = tgts_ds[idx]                           # expected (1,4,H,W)
            tgt_arr = np.asarray(tgt_raw)

            # squeeze leading batch dim
            if tgt_arr.ndim == 4:
                if tgt_arr.shape[0] == 20:
                    assert tile_idx is not None, "Tile IDX has to be provided"
                    tgt_arr = tgt_arr[tile_idx]
                else:
                    assert tgt_arr.shape[0] == 1
                    tgt_arr = np.squeeze(tgt_arr, axis=0)        # -> (4,H,W)

            # resize
            trsz = target_resize_to if target_resize_to is not None else resize_to
            if trsz is not None:
                tgt_arr = _cv2_resize_mask(tgt_arr, trsz)

            # channel order
            if return_channel_last:
                if tgt_arr.ndim == 3 and tgt_arr.shape[0] in (1, 2, 3, 4):
                    tgt_arr = np.transpose(tgt_arr, (1, 2, 0))  # (4,H,W) -> (H,W,4)
            else:
                if tgt_arr.ndim == 3 and tgt_arr.shape[-1] in (1, 2, 3, 4):
                    tgt_arr = np.transpose(tgt_arr, (2, 0, 1))  # (H,W,4) -> (4,H,W)"

            tgts_out = tgt_arr

    return rgb, tgts_out

def fetch_image(
    h5_path: str,
    index: int,
    *,
    tile_idx: Optional[int] = None,
    dataset_key: str = "imgs",
    resize_to: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Image only."""
    rgb, _ = fetch_item(
        h5_path,
        index,
        tile_idx,
        image_key=dataset_key,
        target_key=None,
        resize_to=resize_to,
        target_resize_to=None,
        return_channel_last=True,
    )
    return rgb

def fetch_image_with_targets(
    h5_path: str,
    index: int,
    *,
    tile_idx: Optional[int] = None,
    image_key: str = "imgs",
    target_key: str = "tgts",
    resize_to: Optional[Tuple[int, int]] = None,
    target_resize_to: Optional[Tuple[int, int]] = None,
    return_channel_last: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Image + targets."""
    rgb, tgts = fetch_item(
        h5_path,
        index,
        tile_idx=tile_idx,
        image_key=image_key,
        target_key=target_key,
        resize_to=resize_to,
        target_resize_to=target_resize_to,
        return_channel_last=return_channel_last,
    )
    if tgts is None:
        raise KeyError(f"No targets found at key '{target_key}'.")
    return rgb, tgts

def fetch_images(
    h5_path: str,
    indices: Sequence[int],
    *,
    tile_idx: Optional[int] = None,
    dataset_key: str = "imgs",
    resize_to: Optional[Tuple[int, int]] = (240,240),
) -> List[np.ndarray]:
    """Fetch multiple images using `fetch_image` for convenience."""
    return [
        fetch_image(h5_path, i, dataset_key=dataset_key, resize_to=resize_to, tile_idx=tile_idx)
        for i in indices
    ]

def get_dataset_statistics(
    h5_path: str,
    dataset_type: Literal["train", "test", "val"] = "train",
    output_dir: str = "./figure_data/",
    output_filename: str = "dataset_statistics"
) -> pd.DataFrame:

    ds_filename = os.path.basename(h5_path)
    crop_method = ds_filename.split(f"_{dataset_type}")[0]
    save_suffix = f"_{dataset_type}_{crop_method}"

    output_file = os.path.join(output_dir, f"{output_filename}{save_suffix}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    with h5py.File(h5_path, "r") as f:
        rows = []
        ds_len = len(f["imgs"])
        for i in range(ds_len):
            meta = json.loads(f["meta"][i])
            rows.append(meta["sim_kwargs"])

        df = pd.DataFrame(rows)

    df["crop_method"] = crop_method
    df["dataset_type"] = dataset_type

    df.to_csv(output_file, index=False)
    return df

def _maybe_float(x):
    if isinstance(x, (int, float)) or x is None:
        return x
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x

def _read_run_meta(run_dir: str, unet_mode: str, tag: str) -> dict:
    """Read the optional run_meta_*.json next to the log for extra fields."""
    meta_path = os.path.join(run_dir, f"run_meta_{unet_mode}_{tag}.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _yield_rows_from_log(log_path: str) -> Iterable[dict]:
    """Yield one row per (epoch, split=train/val)."""
    m = _LOG_RE.search(os.path.basename(log_path))
    parsed = m.groupdict() if m else {}
    unet_mode = parsed.get("unet_mode")
    tag       = parsed.get("tag")
    seed      = int(parsed["seed"]) if parsed.get("seed") else None
    target    = int(parsed["target"]) if parsed.get("target") else None

    run_dir = os.path.dirname(log_path)
    meta = _read_run_meta(run_dir, unet_mode or "", tag or "")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # base fields (prefer values inside the record; fall back to filename/meta)
            base = {
                "run_dir": run_dir,
                "log_path": log_path,
                "epoch": rec.get("epoch"),
                "unet_mode": rec.get("unet_mode", unet_mode),
                "mode": rec.get("mode", meta.get("mode")),
                "target": rec.get("target", target),
                "seed": seed if seed is not None else meta.get("seed"),
                "batch_size_per_gpu": rec.get("batch_size_per_gpu", meta.get("batch_size_per_gpu")),
                "world_size": rec.get("world_size", meta.get("world_size")),
                "lr": rec.get("lr"),
                "time_sec": rec.get("time_sec"),
                "best_val_unweighted": rec.get("best_val_unweighted"),
                "weights": rec.get("weights"),
            }

            for split in ("train", "val"):
                block = rec.get(split)
                if not isinstance(block, dict):
                    continue
                row = dict(base)  # copy
                row["split"] = split
                # flatten metrics; cast strings like "0.123" to floats
                for k, v in block.items():
                    row[k] = _maybe_float(v)
                # also carry over a few handy val-only metrics if they were logged at epoch level
                for extra in ("bound_f1_tol2", "center_f1_r10", "energy_rmse", "energy_pearson"):
                    if extra in block:
                        row[extra] = _maybe_float(block[extra])
                    elif extra in rec:
                        # sometimes these live at top level as strings
                        row[extra] = _maybe_float(rec[extra])

                # ensure core identifiers exist
                row["unet_mode"] = row.get("unet_mode") or unet_mode
                row["mode"] = row.get("mode") or (parsed.get("tag", "").split("_")[0] if parsed else None)
                row["target"] = int(row["target"]) if row.get("target") is not None else target
                row["seed"] = int(row["seed"]) if row.get("seed") is not None else seed

                yield row

def get_loss_data(model_output_dir: str, recursive: bool = True,
                  extra_globs: Optional[list[str]] = None,
                  output_dir: str = "",
                  output_filename: str = "loss_values_training") -> pd.DataFrame:
    """
    Scan base_dir for log_*.jsonl files, read them, and return a tidy DataFrame.
    One row per (epoch, split). Columns include identifiers and all metrics.
    """

    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    patterns = [
        os.path.join(model_output_dir, "**", "log_*.jsonl")] \
        if recursive else [os.path.join(model_output_dir, "log_*.jsonl")
    ]
    if extra_globs:
        patterns.extend(extra_globs)

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    rows = []
    for fp in sorted(set(files)):
        for row in _yield_rows_from_log(fp):
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    id_cols = ["unet_mode", "mode", "target", "seed", "epoch", "split"]
    other_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + other_cols]

    # sort
    df = df.sort_values(id_cols, kind="mergesort").reset_index(drop=True)
    df.to_csv(output_file, index = False)
    return df

def generate_param_showcase(
    sim_config: Any,
    camera: Optional[Any],
    sweep: Mapping[str, Sequence[Any]],
    base_overrides: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Builds a grid where each row is a parameter and each column is one value.
    Returns a dict with images, metas, and figure objects for further use.
    """
    rng = np.random.default_rng(seed)

    # Build a clean base kwargs once
    base_kwargs = config_to_kwargs_image_sim(sim_config, rng, camera)
    if base_overrides:
        base_kwargs = merge_kwargs_image_sim(base_kwargs, base_overrides)

    # Ensure we have your requested base defaults unless you override:
    base_kwargs.setdefault("H", 512)
    base_kwargs.setdefault("W", 512)
    base_kwargs.setdefault("n_cells", 400)
    base_kwargs.setdefault("cell_diameter", 8)
    base_kwargs.setdefault("return_targets", True)

    # Prepare sweep shape
    param_names = list(sweep.keys())
    value_lists = [list(sweep[p]) for p in param_names]
    n_rows = len(param_names)
    n_cols = max(len(vs) for vs in value_lists) if value_lists else 0

    # Storage
    images: List[List[np.ndarray]] = [[None]*n_cols for _ in range(n_rows)]
    metas:  List[List[Dict[str, Any]]] = [[None]*n_cols for _ in range(n_rows)]
    targets: List[List[Dict[str, Any]]] = [[None]*n_cols for _ in range(n_rows)]

    # Generate tiles
    tile_idx = 0
    for r, p in enumerate(param_names):
        vals = value_lists[r]
        for c, val in enumerate(vals):
            tile_seed = seed + 9973*tile_idx  # stable per tile
            tile_idx += 1

            kwargs = dict(base_kwargs)
            kwargs["seed"] = int(tile_seed)
            kwargs[p] = val

            img, meta, tgt = simulate_image(**kwargs)

            images[r][c] = img
            metas[r][c] = meta
            targets[r][c] = tgt

    return {
        "images": images,
        "metas": metas,
        "targets": targets,
        "param_names": param_names,
        "value_lists": value_lists,
        "base_kwargs": base_kwargs,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }

def _read_validation_csvs(in_file) -> pd.DataFrame:
    if isinstance(in_file, list):
        if len(in_file) == 0:
            raise FileNotFoundError("No validation CSV files found.")

        return pd.concat(
            [pd.read_csv(file, index_col=None) for file in in_file],
            axis=0,
            ignore_index=True,
        )

    if isinstance(in_file, str):
        if not os.path.isfile(in_file):
            raise FileNotFoundError(f"Validation CSV not found: {in_file}")

        return pd.read_csv(in_file, index_col=None)

    raise TypeError("in_file must be str or list[str].")


def _first_existing_column(
    data: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for col in candidates:
        if col in data.columns:
            return col
    return None


def _add_count_error_pct(
    data: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    """
    Add count_error_pct in a schema-tolerant way.

    Refactored validation preferred columns:
      - count_error_instances_vs_stored_labels
      - count_error_components_vs_stored_labels

    Old validation fallback columns:
      - count_error_instances
      - count_error_components
    """
    data = data.copy()

    if mode == "training":
        n_cells_key = _first_existing_column(
            data,
            [
                # old training/global image count
                "n_cells_per_img",

                # new tiled metadata count, useful if available
                "n_cells_gt_meta_tile",

                # stored-label GT count
                "n_cells_gt_instances_stored_labels",

                # old/fallback GT count
                "n_cells_gt_instances",
            ],
        )

    elif mode in ("testing", "imageJ"):
        n_cells_key = _first_existing_column(
            data,
            [
                "n_cells_gt_instances_stored_labels",
                "n_cells_gt_instances",
                "n_cells_gt_meta_tile",
                "n_cells_full_meta",
                "n_cells_per_img",
            ],
        )

    else:
        return data

    count_error_key = _first_existing_column(
        data,
        [
            # preferred refactored schema
            "count_error_instances_vs_stored_labels",
            "count_error_instances_vs_gt_instances",
            "count_error_components_vs_stored_labels",
            "count_error_components_vs_gt_instances",

            # old schema fallback
            "count_error_instances",
            "count_error_components",
        ],
    )

    if n_cells_key is None or count_error_key is None:
        data["count_error_pct"] = np.nan
        return data

    denom = pd.to_numeric(data[n_cells_key], errors="coerce")
    numer = pd.to_numeric(data[count_error_key], errors="coerce")

    denom = denom.replace(0, np.nan)

    data["count_error_pct"] = (numer / denom) * 100.0
    data["count_error_pct_source"] = count_error_key
    data["count_error_pct_denominator"] = n_cells_key

    return data


def _drop_validation_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicated rows if combined and individual CSVs were accidentally loaded.
    """
    dedup_cols = [
        c for c in [
            "sample_idx",
            "idx",
            "tile_idx",
            "metric_level",
            "dataset_mode",
            "unet_mode",
            "segmentation_method",
            "src_path",
        ]
        if c in data.columns
    ]

    if not dedup_cols:
        return data.reset_index(drop=True)

    return data.drop_duplicates(subset=dedup_cols).reset_index(drop=True)


def _training_files(
    results_dir: str,
    *,
    seg_method: str,
    unet_size: str,
    crop_method: str,
) -> str | list[str]:
    if crop_method == "combined":
        combined_file = os.path.join(
            results_dir,
            f"training_val_combined_{seg_method}.csv",
        )

        if os.path.isfile(combined_file):
            return combined_file

        files = os.listdir(results_dir)

        # Load only individual files, not combined files.
        in_files = [
            os.path.join(results_dir, file)
            for file in files
            if file.startswith("training_val_")
            and file.endswith(f"_{seg_method}.csv")
            and "combined" not in file
            and "summary" not in file
        ]

        if len(in_files) == 0:
            raise FileNotFoundError(
                f"No training validation files found in {results_dir} "
                f"for seg_method={seg_method!r}."
            )

        return sorted(in_files)

    return os.path.join(
        results_dir,
        f"training_val_{unet_size}_{crop_method}_{seg_method}.csv",
    )


def get_validation_data(
    results_dir,
    mode: Literal["training", "testing", "human", "imageJ"],
    seg_method: Literal["inst_seg", "conventional"] = "inst_seg",
    unet_size: Literal["small", "medium", "large"] = "small",
    crop_method: Literal[
        "pad_resize",
        "crop_well_resize",
        "tiles",
        "combined",
    ] = "combined",
    comparison_images: Literal["external_images", "tiles"] = "tiles",
) -> pd.DataFrame:
    """
    Load validation results for figures.

    Supports both the old validation schema and the refactored schema where
    count-error columns were renamed explicitly.
    """
    if mode == "training":
        in_file = _training_files(
            results_dir,
            seg_method=seg_method,
            unet_size=unet_size,
            crop_method=crop_method,
        )

    elif mode == "testing":
        in_file = os.path.join(
            results_dir,
            f"testing_val_{unet_size}_{comparison_images}_{seg_method}.csv",
        )

    elif mode == "human":
        in_file = os.path.join(
            results_dir,
            "human_annotated_comparison.csv",
        )

    elif mode == "imageJ":
        in_file = os.path.join(
            results_dir,
            f"testing_val_imageJ_{unet_size}_{seg_method}.csv",
        )

    else:
        raise ValueError(f"Unknown mode {mode}")

    data = _read_validation_csvs(in_file)

    data = _drop_validation_duplicates(data)

    data = _add_count_error_pct(data, mode=mode)

    return data.reset_index(drop=True)

def _prep_config_for_unet_comparison(
    cfg: dict,
    models_dir: str,
    unet_mode: Literal["large", "medium", "small"],
) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["model_dir"] = models_dir
    cfg["unet_mode"] = unet_mode
    cfg["model_file"] = f"best_{unet_mode}_tiles_S512_seed187.pth"
    cfg["input_is_tiles"] = False
    cfg["normalize"] = True
    return cfg


@dataclass(frozen=True)
class Inset:
    """
    Square crop region.

    Coordinates use image-array convention:
        x = column index
        y = row index

    Crop:
        image[y : y + size, x : x + size]
    """
    x: int
    y: int
    size: int


MICROSCOPY_IMAGE_CONFIG = {
    "base_dir": "RUN_DEBB704A2449",
    "image_path": "IMG_0001.tif",
    "inset": Inset(x=1200, y=100, size=512),
}


def _crop_inset_array(arr: np.ndarray, inset: Inset) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or HWC image array, got shape {arr.shape}.")

    h, w = arr.shape[:2]

    x0 = int(inset.x)
    y0 = int(inset.y)
    x1 = x0 + int(inset.size)
    y1 = y0 + int(inset.size)

    if inset.size <= 0:
        raise ValueError(f"inset.size must be positive, got {inset.size}.")

    if x0 < 0 or y0 < 0:
        raise ValueError(f"Inset starts outside image: x={x0}, y={y0}.")

    if x1 > w or y1 > h:
        raise ValueError(
            f"Inset exceeds image bounds. "
            f"Inset x=[{x0}, {x1}), y=[{y0}, {y1}); "
            f"image width={w}, height={h}."
        )

    return arr[y0:y1, x0:x1, ...] if arr.ndim == 3 else arr[y0:y1, x0:x1]


def _load_real_life_unet_input_tile(
    *,
    ext_images_dir: str,
    image_config: dict[str, Any],
    page: int = 0,
    max_mp: Optional[float] = 200.0,
) -> dict[str, Any]:
    """
    Load a real-life microscopy image and crop the selected UNet input tile.

    Returns:
        full_image: HWC float32 image in [0, 1]
        tile: HWC float32 512x512 crop in [0, 1]
        load_report: loader metadata
    """
    base_dir = os.path.join(ext_images_dir, image_config["base_dir"])
    image_path = image_config["image_path"]
    inset = image_config["inset"]

    full_image, load_report = load_image(
        image_path,
        page=page,
        base_dir=base_dir,
        max_mp=max_mp,
        as_chw=False,
        scale=True,
    )

    full_image = np.asarray(full_image, dtype=np.float32)

    tile = _crop_inset_array(full_image, inset).astype(np.float32, copy=False)

    if tile.ndim == 2:
        tile = np.repeat(tile[..., None], 3, axis=-1)

    if tile.ndim != 3:
        raise ValueError(f"Expected cropped tile to be HWC, got shape {tile.shape}.")

    if tile.shape[0] != inset.size or tile.shape[1] != inset.size:
        raise ValueError(
            f"Unexpected tile shape {tile.shape}; expected "
            f"({inset.size}, {inset.size}, C)."
        )

    if tile.shape[-1] == 1:
        tile = np.repeat(tile, 3, axis=-1)

    if tile.shape[-1] > 3:
        tile = tile[..., :3]

    if tile.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel tile, got shape {tile.shape}.")

    return {
        "full_image": full_image,
        "tile": tile,
        "inset": inset,
        "base_dir": base_dir,
        "image_path": image_path,
        "load_report": load_report,
    }


def generate_unet_comparison(
    models_dir: str,
    ext_images_dir: str,
    unet_base_config: Any,
    segmenter_class: Any,
    output_dir: str,
    output_filename: str = "unet_segmentation_comparison",
    redo_analysis: bool = False,
    image_config: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Compare small, medium, and large UNets on a real-life microscopy crop.

    The image is loaded with scale=True, so the input tile is float32 in [0, 1].
    Further UNet normalization is handled by SegmenterUNet according to the
    model config.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{output_filename}.dict")

    existing_file = check_for_file(output_file)
    if existing_file is not None and not redo_analysis:
        assert isinstance(existing_file, dict)
        return existing_file

    if image_config is None:
        image_config = MICROSCOPY_IMAGE_CONFIG

    large_cfg = _prep_config_for_unet_comparison(
        unet_base_config,
        models_dir,
        "large",
    )
    med_cfg = _prep_config_for_unet_comparison(
        unet_base_config,
        models_dir,
        "medium",
    )
    small_cfg = _prep_config_for_unet_comparison(
        unet_base_config,
        models_dir,
        "small",
    )

    large_seg = segmenter_class.from_config(large_cfg)
    med_seg = segmenter_class.from_config(med_cfg)
    small_seg = segmenter_class.from_config(small_cfg)

    image_payload = _load_real_life_unet_input_tile(
        ext_images_dir=ext_images_dir,
        image_config=image_config,
    )

    img = image_payload["tile"]

    small_out = small_seg(img)
    med_out = med_seg(img)
    large_out = large_seg(img)

    res = {
        "original": img,
        "full_image": image_payload["full_image"],
        "inset": image_payload["inset"],
        "image_path": image_payload["image_path"],
        "base_dir": image_payload["base_dir"],
        "load_report": image_payload["load_report"],
        "small": small_out["probs"],
        "med": med_out["probs"],
        "large": large_out["probs"],
    }

    with open(output_file, "wb") as file:
        pickle.dump(res, file)

    return res

def segment_image_unet(models_dir: str,
                       img: np.ndarray,
                       unet_base_config: Any,
                       segmenter_class: Any) -> np.ndarray:

    cfg = _prep_config_for_unet_comparison(unet_base_config, models_dir, "small")
    seg = segmenter_class.from_config(cfg)
    pred = seg(img)["probs"]
    return pred

def segment_well_plates(output_dir: str,
                        output_filename: str = "well_results.dict") -> dict:
    output_file = os.path.join(output_dir, f"{output_filename}.dict")
    existing_file = check_for_file(output_file)
    if existing_file is not None and not redo_analysis:
        assert isinstance(existing_file, dict)
        return existing_file



def instance_labels_to_rgb(
    label_img: np.ndarray,
    background_label: int = 0,
    seed: int = 42,
) -> np.ndarray:
    if label_img.ndim != 2:
        raise ValueError("label_img must be a 2D array")

    rng = np.random.default_rng(seed)
    labels = np.unique(label_img)

    rgb_img = np.zeros((*label_img.shape, 3), dtype=np.uint8)

    for label in labels:
        if label == background_label:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            color = rng.integers(0, 256, size=3, dtype=np.uint8)
        rgb_img[label_img == label] = color

    return rgb_img


TARGET_IMAGE_SIZE = (1024, 1024)

def resize_square_image(
    image: np.ndarray,
    target_size: tuple[int, int] = TARGET_IMAGE_SIZE,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")

    h, w = image.shape[:2]
    if h != w:
        raise ValueError("image must already be square")

    return cv2.resize(image, target_size, interpolation=interpolation)


def prepare_image(image: np.ndarray, is_segmentation: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_segmentation else cv2.INTER_CUBIC
    return resize_square_image(image, interpolation=interpolation)

def crop_square(image: np.ndarray, x: int, y: int, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be > 0")

    h, w = image.shape[:2]

    if x < 0 or y < 0 or x + length > w or y + length > h:
        raise ValueError("Crop square is outside image bounds")

    return image[y:y + length, x:x + length]

def _load_crop_resize_image(
    file_name: str,
    base_dir: str,
    crop_params: tuple[int, int, int],
    scale: bool = True,
) -> np.ndarray:
    image, _ = load_image(file_name, base_dir=base_dir, scale=scale, as_chw=False)
    x, y, length = crop_params
    image = crop_square(image, x=x, y=y, length=length)
    # image = _prepare_image(image, is_segmentation=False)
    return image

def load_or_create_figure_1_image_cache(
    cache_path: str,
    model_dir: str = "../final/models",
    model_file: str = "best_small_tiles_S512_seed187.pth",
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:

    if os.path.exists(cache_path) and not force_recompute:
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    unet_config = copy.deepcopy(UNET_CONFIG)
    unet_config["model_dir"] = model_dir
    unet_config["unet_mode"] = "large" if "large" in model_file else "small"
    unet_config["model_file"] = model_file
    unet_config["instance_cfg"] = INSTANCE_CONFIG_DICT
    unet_config["thr_cell"] = 0.1
    unet_config["thr_bound"] = 0.1

    seg = SegmenterUNetInference.from_config(unet_config)

    sim_img = _load_crop_resize_image(
        file_name="000008.tif",
        base_dir="../final/image_datasets/imgs",
        crop_params=(500, 200, 1200),
        scale=True,
    )

    mic_img = _load_crop_resize_image(
        file_name="Bild_3267.tif",
        base_dir="../final/show_images/RUN_D83E90DC4784",
        crop_params=(500, 64, 1500),
        scale=True,
    )

    gp_img = _load_crop_resize_image(
        file_name="PXL_20251107_130141300.jpg",
        base_dir="../final/show_images/RUN_458944AD0ED8",
        crop_params=(300, 1100, 2700),
        scale=True,
    )

    iphone_img = _load_crop_resize_image(
        file_name="IMG_3859.jpeg",
        base_dir="../final/show_images/RUN_7FD3D36FCDD5",
        crop_params=(1200, 50, 2500),
        scale=True,
    )

    monochrome_img = _load_crop_resize_image(
        file_name="xm1_+dtt_1b.tif",
        base_dir="../final/show_images/RUN_F9AD9432FA3F",
        crop_params=(0, 0, 1440),
        scale=True,
    )

    sim_seg_labels = seg(sim_img)["instance_labels"]
    mic_seg_labels = seg(mic_img)["instance_labels"]
    gp_seg_labels = seg(gp_img)["instance_labels"]
    iphone_seg_labels = seg(iphone_img)["instance_labels"]
    monochrome_seg_labels = seg(monochrome_img)["instance_labels"]

    sim_seg_rgb = instance_labels_to_rgb(sim_seg_labels)
    mic_seg_rgb = instance_labels_to_rgb(mic_seg_labels)
    gp_seg_rgb = instance_labels_to_rgb(gp_seg_labels)
    iphone_seg_rgb = instance_labels_to_rgb(iphone_seg_labels)
    monochrome_seg_rgb = instance_labels_to_rgb(monochrome_seg_labels)

    data = {
        "simulated_image": sim_img,
        "simulated_segmentation": sim_seg_rgb,
        "microscopy_image": mic_img,
        "microscopy_segmentation": mic_seg_rgb,
        "googlepixel_image": gp_img,
        "googlepixel_segmentation": gp_seg_rgb,
        "iphone_image": iphone_img,
        "iphone_segmentation": iphone_seg_rgb,
        "monochrome_image": monochrome_img,
        "monochrome_segmentation": monochrome_seg_rgb,
    }

    np.savez_compressed(cache_path, **data)

    return data

def _read_rgb_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if img.ndim == 2:
        return img

    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    raise ValueError(f"Unsupported image shape: {img.shape}")

def fill_and_recalculate_frac_pos_from_scores_random(
    df: pd.DataFrame,
    *,
    human_annotators=("1", "2"),
    method_annotators=("unet", "imageJ"),
    score_col="score",
    adjusted_score_col="adjusted_score",
    annotator_col="Annotator",
    experiment_col="Folder",
    role_col="role",
    frac_pos_col="frac_pos",
    corrected_frac_pos_col="corrected_frac_pos",
    score_source_for_adjusted="corrected_frac_pos",
    pc_ref_col="pc_ref_raw",
    nc_ref_col="nc_ref_raw",
    score_to_range=None,
    positive_role="positive",
    negative_role="negative",
    overwrite_human_frac_pos=True,
    overwrite_corrected_frac_pos=True,
    overwrite_adjusted_score=True,
    random_seed=0,
):
    """
    1. Simulate raw frac_pos for human annotators from their categorical scores.
    2. Keep existing raw frac_pos for method annotators such as unet/imageJ.
    3. Recalculate corrected_frac_pos for humans and methods.
    4. Recalculate adjusted_score from corrected_frac_pos by default.

    Correction is done separately per:
        Folder + Annotator

    Formula:
        corrected_frac_pos = (frac_pos - nc_ref) / (pc_ref - nc_ref) * 100

    where:
        nc_ref = mean raw frac_pos of negative controls
        pc_ref = mean raw frac_pos of positive controls

    Score conversion:
        <= 10  -> 1
        <= 20  -> 2
        <= 50  -> 4
        <= 80  -> 6
        > 80   -> 8

    Parameters
    ----------
    score_source_for_adjusted : {"corrected_frac_pos", "frac_pos"}
        Which fraction column to use for recalculating adjusted_score.

        Usually use:
            "corrected_frac_pos"

        Use "frac_pos" only if you want raw, uncorrected scores.
    """

    df_out = df.copy()
    rng = np.random.default_rng(random_seed)

    if score_to_range is None:
        score_to_range = {
            1: (0.0, 10.0),
            2: (10.0, 20.0),
            4: (20.0, 40.0),
            6: (40.0, 60.0),
            8: (80.0, 100.0),
            0: (0.0, 2.0),
            11: (0.0, 10.0),
        }

    if score_source_for_adjusted not in {"corrected_frac_pos", "frac_pos"}:
        raise ValueError(
            "score_source_for_adjusted must be 'corrected_frac_pos' or 'frac_pos'"
        )

    human_annotators = [str(a) for a in human_annotators]
    method_annotators = [str(a) for a in method_annotators]
    all_annotators = human_annotators + method_annotators

    required_cols = {
        annotator_col,
        experiment_col,
        role_col,
        frac_pos_col,
        score_col,
    }

    missing_cols = required_cols - set(df_out.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if corrected_frac_pos_col not in df_out.columns:
        df_out[corrected_frac_pos_col] = np.nan

    if adjusted_score_col not in df_out.columns:
        df_out[adjusted_score_col] = np.nan

    if pc_ref_col not in df_out.columns:
        df_out[pc_ref_col] = np.nan

    if nc_ref_col not in df_out.columns:
        df_out[nc_ref_col] = np.nan

    df_out["_annotator_str"] = df_out[annotator_col].astype(str)

    human_mask = df_out["_annotator_str"].isin(human_annotators)
    all_eval_mask = df_out["_annotator_str"].isin(all_annotators)

    # ------------------------------------------------------------
    # 1) Simulate raw frac_pos for human annotators from score bins
    # ------------------------------------------------------------
    simulated = pd.Series(np.nan, index=df_out.index, dtype=float)

    for score, (low, high) in score_to_range.items():
        score_mask = human_mask & df_out[score_col].eq(score)
        n = int(score_mask.sum())

        if n > 0:
            simulated.loc[score_mask] = rng.uniform(low, high, size=n)

    missing_sim = human_mask & simulated.isna()
    if missing_sim.any():
        bad_scores = sorted(df_out.loc[missing_sim, score_col].dropna().unique())
        raise ValueError(
            "Some human scores could not be mapped to ranges. "
            f"Unmapped scores: {bad_scores}"
        )

    if overwrite_human_frac_pos:
        df_out.loc[human_mask, frac_pos_col] = simulated.loc[human_mask]
    else:
        missing_frac = human_mask & df_out[frac_pos_col].isna()
        df_out.loc[missing_frac, frac_pos_col] = simulated.loc[missing_frac]

    # ------------------------------------------------------------
    # 2) Recalculate corrected_frac_pos for humans and methods
    # ------------------------------------------------------------
    refs = []

    for (folder, annotator), group in df_out.loc[all_eval_mask].groupby(
        [experiment_col, "_annotator_str"],
        dropna=False,
    ):
        pc_values = group.loc[
            group[role_col].eq(positive_role),
            frac_pos_col,
        ].astype(float)

        nc_values = group.loc[
            group[role_col].eq(negative_role),
            frac_pos_col,
        ].astype(float)

        pc_ref = pc_values.mean()
        nc_ref = nc_values.mean()

        refs.append(
            {
                experiment_col: folder,
                annotator_col: annotator,
                "pc_ref": pc_ref,
                "nc_ref": nc_ref,
                "n_positive_controls": int(pc_values.notna().sum()),
                "n_negative_controls": int(nc_values.notna().sum()),
            }
        )

        row_mask = (
            all_eval_mask
            & df_out[experiment_col].eq(folder)
            & df_out["_annotator_str"].eq(annotator)
        )

        df_out.loc[row_mask, pc_ref_col] = pc_ref
        df_out.loc[row_mask, nc_ref_col] = nc_ref

        raw = df_out.loc[row_mask, frac_pos_col].astype(float)

        if pd.isna(pc_ref) or pd.isna(nc_ref) or pc_ref == nc_ref:
            corrected = pd.Series(np.nan, index=df_out.index[row_mask])
        else:
            corrected = (raw - nc_ref) / (pc_ref - nc_ref) * 100.0
            corrected = corrected.clip(0.0, 100.0)

        if overwrite_corrected_frac_pos:
            df_out.loc[row_mask, corrected_frac_pos_col] = corrected
        else:
            missing_corr = row_mask & df_out[corrected_frac_pos_col].isna()
            df_out.loc[missing_corr, corrected_frac_pos_col] = corrected.loc[
                df_out.index[missing_corr]
            ]

    refs = pd.DataFrame(refs)

    # ------------------------------------------------------------
    # 3) Recalculate adjusted_score
    # ------------------------------------------------------------
    if score_source_for_adjusted == "corrected_frac_pos":
        score_input_col = corrected_frac_pos_col
    else:
        score_input_col = frac_pos_col

    def convert_frac_pos_to_score(frac_pos):
        if pd.isna(frac_pos):
            return np.nan
        if frac_pos <= 10:
            return 1
        elif frac_pos <= 20:
            return 2
        elif frac_pos <= 50:
            return 4
        elif frac_pos <= 80:
            return 6
        else:
            return 8

    recalculated_scores = df_out.loc[all_eval_mask, score_input_col].apply(
        convert_frac_pos_to_score
    )

    if overwrite_adjusted_score:
        df_out.loc[all_eval_mask, adjusted_score_col] = recalculated_scores
    else:
        missing_adj = all_eval_mask & df_out[adjusted_score_col].isna()
        df_out.loc[missing_adj, adjusted_score_col] = recalculated_scores.loc[
            df_out.index[missing_adj]
        ]

    df_out = df_out.drop(columns=["_annotator_str"])

    return df_out, refs

def get_score_frame(validation_results_dir):

    from ..validation.experiment_readouts import concat_annotator_frames

    """\
    NOTE: intermediate function. The validation function that computes
    that is actually outputting the full frame.
    Until the bug for the calculation is fixed and the function is rerun,
    this is the workaround, where the adjusted_score gets recomputed
    based on the raw-readouts from the pos/neg ctrl.
    TE 26.04.2026
    """

    manual = pd.read_csv(os.path.join(validation_results_dir, "manual_df.csv"), index_col = None)
    imagej = pd.read_csv(os.path.join(validation_results_dir, "imagej_df.csv"), index_col = None)
    unet = pd.read_csv(os.path.join(validation_results_dir, "unet_df.csv"), index_col = None)

    df = concat_annotator_frames([manual, imagej, unet])
    df = df[~(df["Folder"] == "20251028_25720349_+DTT")]
    df = df[~(df["Folder"] == "RUN_A9F6C7B995D6")]
    df = df[~df["score"].isin([0,11])]
    df["Annotator"] = df["Annotator"].astype(str)

    df_filled, refs = fill_and_recalculate_frac_pos_from_scores_random(
        df,
        human_annotators=("1", "2"),
        method_annotators=("unet", "imageJ"),
        score_col="score",
        adjusted_score_col="adjusted_score",
        annotator_col="Annotator",
        experiment_col="Folder",
        role_col="role",
        frac_pos_col="frac_pos",
        corrected_frac_pos_col="corrected_frac_pos",
        score_source_for_adjusted="corrected_frac_pos",
        random_seed=42,
    )

    return df_filled



def _supplement_level(metric_level: str) -> str:
    if metric_level not in ("PER_IMAGE", "PER_TILE"):
        raise ValueError("metric_level must be 'PER_IMAGE' or 'PER_TILE'")
    return "image" if metric_level == "PER_IMAGE" else "tile"


def _supplement_read_csv(path: str | _SupplementPath) -> _supp_pd.DataFrame:
    path = _SupplementPath(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing comparison CSV: {path}")
    frame = _supp_pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Empty comparison CSV: {path}")
    return frame


def _supplement_require(frame: _supp_pd.DataFrame, columns, source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{source}: missing required columns {missing}")


def _supplement_numeric(frame: _supp_pd.DataFrame, metrics, source: str) -> _supp_pd.DataFrame:
    _supplement_require(frame, metrics, source)
    frame = frame.copy()
    for key in metrics:
        frame[key] = _supp_pd.to_numeric(frame[key], errors="coerce")
        valid = frame[key].dropna()
        if not _supp_np.isfinite(valid).all() or not valid.between(0, 1).all():
            raise ValueError(f"{source}: {key} contains invalid values (expected [0, 1])")
    return frame


def _supplement_paired_metric(
    frame: _supp_pd.DataFrame,
    metric: str,
    methods: tuple[str, ...] = SUPPLEMENT_METHODS,
) -> _supp_pd.DataFrame:
    """Retain the exact same source image/tile for the two methods."""
    _supplement_require(frame, ["image_id", "algorithm", metric], "metric pairing")
    selected = frame.loc[frame["algorithm"].isin(methods),
                         ["image_id", "algorithm", metric]].copy()
    if selected.duplicated(["image_id", "algorithm"]).any():
        raise ValueError(f"Duplicate image/algorithm pairs for {metric}")
    wide = selected.pivot(index="image_id", columns="algorithm", values=metric)
    absent = set(methods) - set(wide.columns)
    if absent:
        raise ValueError(f"{metric}: methods absent from data: {sorted(absent)}")
    wide = wide.dropna(subset=list(methods))
    if wide.empty:
        raise ValueError(f"{metric}: no finite matched observations")
    return wide.loc[:, list(methods)].stack().rename(metric).reset_index()


def plot_segmentation_supplement_metric(
    ax,
    frame: _supp_pd.DataFrame,
    metric: str,
    title: str,
    *,
    methods: tuple[str, ...] = SUPPLEMENT_METHODS,
    max_strip_points: int | None = 400,
) -> None:
    """Shared white box/gray stripplot; full dataset for box, sample only for dots."""
    import seaborn as _supp_sns
    from alloviewer.dev.figures import figure_config as _supp_cfg
    from alloviewer.dev.figures import figure_utils as _supp_utils

    paired = _supplement_paired_metric(frame, metric, methods)
    _supp_sns.boxplot(
        data=paired, x="algorithm", y=metric, order=list(methods), ax=ax,
        color="white", saturation=1, width=0.55, whis=(0, 100), showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.75},
        whiskerprops={"color": "black", "linewidth": 0.65},
        capprops={"color": "black", "linewidth": 0.65},
        medianprops={"color": "black", "linewidth": 1.0},
        zorder=2,
    )
    if max_strip_points is None:
        dots = paired
    else:
        if max_strip_points < 1:
            raise ValueError("max_strip_points must be >=1 or None")
        dots = _supp_pd.concat([
            block.sample(n=min(len(block), max_strip_points), random_state=187)
            for _, block in paired.groupby("algorithm", sort=False)
        ], ignore_index=True)
    _supp_sns.stripplot(
        data=dots, x="algorithm", y=metric, order=list(methods), ax=ax,
        color="0.55", size=2.0, jitter=0.22, alpha=0.60,
        linewidth=0.3, edgecolor="black", zorder=3,
    )
    counts = paired.groupby("algorithm").size()
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f"{method}\n(n={counts[method]:,})" for method in methods])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=_supp_cfg.TITLE_SIZE)
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks(_supp_np.linspace(0, 1, 6))
    ax.grid(axis="y", color="0.86", linewidth=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _supp_utils.adjust_fontsize_ticklabels(ax, _supp_cfg.AXIS_LABEL_SIZE)


def load_figure_s7_real_metrics(
    validation_results_dir: str,
    *,
    metrics: tuple[str, ...],
    metric_level: str = "PER_IMAGE",
    unet_size: str = "small",
) -> _supp_pd.DataFrame:
    """S7: real manual ROIs. Image rows pool annotated tiles, not full images.

    Matching is on (source, annotator, tile), prior to image-level averaging.
    This prevents different annotator sets for UNet and Cellpose.
    """
    scope = _supplement_level(metric_level)
    path = _SupplementPath(validation_results_dir) / "manual_segmentation_comparison_v2.csv"
    df = _supplement_read_csv(path)
    _supplement_require(
        df, ["level", "status", "source_image_key", "image_name", "human_annotator",
             "method", *metrics], str(path),
    )
    df = df.loc[(df["level"] == scope) & (df["status"] == "ok")].copy()
    if "unet_mode" in df.columns:
        df = df.loc[df["unet_mode"].eq(unet_size)].copy()
    df["algorithm"] = df["method"].replace({"AlloViewer_UNet": "UNet"})
    df = df.loc[df["algorithm"].isin(SUPPLEMENT_METHODS)].copy()
    df = _supplement_numeric(df, metrics, path.name)
    if df.empty:
        raise ValueError(f"No usable real-image {scope} rows in {path}")

    keys = (["source_image_key", "human_annotator"] if scope == "image" else
            ["source_image_key", "image_name", "human_annotator"])
    if df.duplicated(keys + ["algorithm"]).any():
        raise ValueError(f"Duplicate real-image annotation keys: {keys}")
    matched = df.groupby(keys, dropna=False)["algorithm"].nunique()
    matched_keys = matched[matched == len(SUPPLEMENT_METHODS)].index.to_frame(index=False)
    df = df.merge(matched_keys, on=keys, how="inner", validate="many_to_one")
    if df.empty:
        raise ValueError("No common manual annotations between UNet and Cellpose")
    if scope == "image":
        # Each source image is represented once per method, averaging the *same*
        # matched annotators. The source scores themselves pool annotated tiles.
        df = df.groupby(["source_image_key", "algorithm"], as_index=False)[list(metrics)].mean()
        df["image_id"] = df["source_image_key"].astype(str)
    else:
        df["image_id"] = (
            df["source_image_key"].astype(str) + "|" +
            df["image_name"].astype(str) + "|" +
            df["human_annotator"].astype(str)
        )
    if df.duplicated(["image_id", "algorithm"]).any():
        raise ValueError("Non-unique real-image observation identifiers")
    print(f"S7 {metric_level}: {df['image_id'].nunique():,} annotated observations")
    return df[["image_id", "algorithm", *metrics]].copy()


def load_figure_s8_simulated_metrics(
    validation_results_dir: str,
    *,
    metrics: tuple[str, ...],
    metric_level: str = "PER_IMAGE",
    unet_size: str = "small",
    seg_method: str = "inst_seg",
) -> tuple[_supp_pd.DataFrame, _supp_pd.DataFrame]:
    """S8: comparable final-instance metrics on simulator-stored /inst GT."""
    scope = _supplement_level(metric_level)
    path = (_SupplementPath(validation_results_dir) /
            f"testing_val_CPIJ_{unet_size}_{seg_method}.csv")
    source = _supplement_read_csv(path)
    _supplement_require(source,
                        ["sample_idx", "tile_idx", "metric_level", "method", *metrics],
                        str(path))
    df = source.loc[source["metric_level"].astype(str).str.lower() == scope].copy()
    if "unet_mode" in df.columns:
        df = df.loc[df["unet_mode"].eq(unet_size)].copy()
    df["algorithm"] = df["method"].replace({"AlloViewer_UNet": "UNet"})
    df = df.loc[df["algorithm"].isin(SUPPLEMENT_METHODS)].copy()
    df = _supplement_numeric(df, metrics, path.name)
    if df.empty:
        raise ValueError(f"No simulated UNet/Cellpose {scope} rows in {path}")
    df["sample_idx"] = _supp_pd.to_numeric(df["sample_idx"], errors="raise").astype(int)
    if scope == "tile":
        df["tile_idx"] = _supp_pd.to_numeric(df["tile_idx"], errors="raise").astype(int)
        df["image_id"] = df["sample_idx"].astype(str) + ":" + df["tile_idx"].astype(str)
    else:
        df["image_id"] = df["sample_idx"].astype(str)
    if df.duplicated(["image_id", "algorithm"]).any():
        raise ValueError("Duplicate simulated image/tile and algorithm keys")
    print(f"S8 {metric_level}: {df['image_id'].nunique():,} simulator observations")
    return df[["image_id", "algorithm", *metrics]].copy(), source


def _supplement_single_setting(frame: _supp_pd.DataFrame, key: str):
    if key not in frame.columns:
        raise KeyError(f"CPIJ CSV does not record Cellpose setting {key!r}")
    values = frame[key].dropna().unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Need one recorded Cellpose setting for {key}: got {values}")
    return values[0]


def segment_image_cellpose(
    img: _supp_np.ndarray,
    *,
    use_gpu: bool,
    cellpose_model: str | None,
    diameter: float,
    min_size: int,
    flow_threshold: float,
    cellprob_threshold: float,
    expected_version: str | None = None,
) -> _supp_np.ndarray:
    """Use the same Cellpose model/adapter as manual and simulated validation.

    Input is RGB HWC float in [0,1]. Returns the instance-label map only.
    """
    from alloviewer.dev.validation.manual_segmentation_comparison import (
        _make_cellpose, _predict_cellpose,
    )
    img = _supp_np.asarray(img, dtype=_supp_np.float32)
    if img.ndim != 3 or img.shape[-1] != 3 or not _supp_np.isfinite(img).all():
        raise ValueError("Cellpose needs finite RGB input [H,W,3]")
    if img.min() < 0 or img.max() > 1:
        raise ValueError("Cellpose input must be scaled to [0,1]")
    model, major, version, _model_label = _make_cellpose(cellpose_model, use_gpu)
    if expected_version is not None and version != str(expected_version):
        raise RuntimeError(f"CPIJ used Cellpose {expected_version}, installed version is {version}")
    masks = _predict_cellpose(
        model, img, major=major, diameter=float(diameter), min_size=int(min_size),
        flow_threshold=float(flow_threshold), cellprob_threshold=float(cellprob_threshold),
    )
    masks = _supp_np.asarray(masks, dtype=_supp_np.int32)
    if masks.shape != img.shape[:2] or (masks < 0).any():
        raise ValueError("Cellpose returned an invalid instance-label map")
    return masks


def load_figure_s8_simulated_example(
    model_output_dir: str,
    h5_path: str,
    source: _supp_pd.DataFrame,
    *,
    img_index: int = 12,
    tile_idx: int = 7,
    unet_size: str = "small",
    seg_method: str = "inst_seg",
    model_file: str | None = None,
) -> dict[str, _supp_np.ndarray]:
    """Re-run the CPIJ full-image inference, then crop one representative tile.

    Ground truth: stored H5 /inst labels. UNet probability heads are stitched
    before instance segmentation, as in simulated_segmentation_comparison.py.
    Cellpose sees the reconstructed full RGB image, not an individual tile.
    """
    import torch
    from torch.utils.data import DataLoader
    from alloviewer.dev.segmentation import TiledH5Dataset
    from alloviewer.dev.segmentation.utils import collate_no_meta
    from alloviewer.dev.validation.imageJ_comparison import (
        _extract_full_hw, _make_segmenter_and_validation_config,
        _segment_full_probs, stitch_float_tiles_mean, stitch_instance_tiles,
    )
    from alloviewer.dev.validation.simulated_segmentation_comparison import _full_rgb_from_tiles

    gt_path = _SupplementPath(h5_path) / "fullres_ground_truth.h5"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing simulated GT H5: {gt_path}")
    if seg_method != "inst_seg":
        raise ValueError("S8 example currently supports seg_method='inst_seg' only")
    _supplement_require(source, ["sample_idx", "method"], "S8 CPIJ CSV")
    scored = set(_supp_pd.to_numeric(source["sample_idx"], errors="raise").astype(int))
    if img_index not in scored:
        raise ValueError(f"img_index={img_index} is not among scored CPIJ samples")

    checkpoint = model_file or f"best_{unet_size}_tiles_S512_seed187.pth"
    segmenter, imagej_segmenter, validation_cfg = _make_segmenter_and_validation_config(
        gt_h5_path=str(gt_path),
        out_csv=str(_SupplementPath(h5_path) / "_unused_figure_s8.csv"),
        out_summary_json=str(_SupplementPath(h5_path) / "_unused_figure_s8.json"),
        unet_mode=unet_size, model_dir=str(model_output_dir),
        model_file=checkpoint, seg_method=seg_method,
    )
    del imagej_segmenter  # S8 compares UNet and Cellpose, not NCISP.
    dataset = TiledH5Dataset(str(gt_path), indices=[int(img_index)])
    try:
        batch = next(iter(DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_no_meta,
        )))
        inputs, _targets, extras = batch
        metadata = extras["meta"][0]
        tile_metadata = metadata["tiles"]
        n_tiles = len(tile_metadata)
        if not (0 <= tile_idx < n_tiles):
            raise ValueError(f"tile_idx={tile_idx} outside 0..{n_tiles-1}")
        tiles = inputs[0, :n_tiles].detach().cpu().numpy().astype(_supp_np.float32)
        gt_tiles = extras["instance_labels"][0, :n_tiles].detach().cpu().numpy().astype(_supp_np.int32)
        full_hw = _extract_full_hw(metadata, tile_metadata)
        ground_truth = stitch_instance_tiles(gt_tiles, tile_metadata, full_hw)
        rgb_full = _full_rgb_from_tiles(tiles, tile_metadata, full_hw)

        with torch.inference_mode():
            result = segmenter(tiles)
        heads = _supp_np.stack(
            [result["probs"][name] for name in ("cell", "bound", "center", "energy")],
            axis=1,
        ).astype(_supp_np.float32)
        full_heads = stitch_float_tiles_mean(heads, tile_metadata, full_hw)
        unet_full = _segment_full_probs(
            segmenter.inst_seg, *full_heads, cell_thr=validation_cfg.cell_thr,
            update_cell_mask=True,
        )

        cp_rows = source.loc[source["method"].eq("Cellpose")]
        if cp_rows.empty:
            raise ValueError("No Cellpose rows/settings in CPIJ CSV")
        cp_name = str(_supplement_single_setting(cp_rows, "cellpose_model"))
        if cp_name == "installed_cellpose4_default":
            cp_name = None
        cp_masks = segment_image_cellpose(
            rgb_full,
            use_gpu=str(segmenter.device).startswith("cuda"),
            cellpose_model=cp_name,
            diameter=float(_supplement_single_setting(cp_rows, "cellpose_diameter")),
            min_size=int(_supplement_single_setting(cp_rows, "cellpose_min_size")),
            flow_threshold=float(_supplement_single_setting(cp_rows, "cellpose_flow_threshold")),
            cellprob_threshold=float(_supplement_single_setting(cp_rows, "cellpose_cellprob_threshold")),
            expected_version=str(_supplement_single_setting(cp_rows, "cellpose_version")),
        )
        if ground_truth.shape != full_hw or _supp_np.asarray(unet_full).shape != full_hw or cp_masks.shape != full_hw:
            raise ValueError("GT/UNet/Cellpose reconstructed shapes do not agree")

        info = tile_metadata[tile_idx]
        y0, x0 = (int(v) for v in info["tile_xy"])
        h, w = (int(v) for v in info["tile_hw"])
        if h <= 0 or w <= 0 or y0 < 0 or x0 < 0 or y0+h > full_hw[0] or x0+w > full_hw[1]:
            raise ValueError("Invalid source tile bounds")
        crop = _supp_np.s_[y0:y0+h, x0:x0+w]
        return {
            "image": _supp_np.asarray(rgb_full[crop]),
            "gt": _supp_np.asarray(ground_truth[crop], dtype=_supp_np.int32),
            "unet": _supp_np.asarray(unet_full[crop], dtype=_supp_np.int32),
            "cellpose": _supp_np.asarray(cp_masks[crop], dtype=_supp_np.int32),
        }
    finally:
        dataset.close()
        del segmenter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

