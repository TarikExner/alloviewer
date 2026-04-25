from __future__ import annotations

import os
import pickle
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
import copy

_LOG_RE = re.compile(
    r"log_(?P<unet_mode>small|medium|large)_(?P<tag>(pad_resize|crop_well_resize|tiles)_S(?P<target>\d+)_seed(?P<seed>\d+))\.jsonl$"
)

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
    dataset_key: Optional[str] = "imgs",
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
    simulate_image_fn,
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

            img, meta, tgt = simulate_image_fn(**kwargs)

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

def get_validation_data(results_dir,
                        mode: Literal["training", "testing", "human", "imageJ"],
                        seg_method: Literal["inst_seg", "conventional"] = "inst_seg",
                        unet_size: Literal["small", "medium", "large"] = "small",
                        crop_method: Literal["pad_resize", "crop_well_resize", "tiles", "combined"] = "combined",
                        comparison_images: Literal["external_images", "tiles"] = "tiles"
                        ) -> pd.DataFrame:

    if mode == "training":
        if crop_method == "combined":
            in_file = os.path.join(results_dir, f"{mode}_val_combined.csv")
        else:
            in_file = os.path.join(results_dir, f"{mode}_val_{unet_size}_{crop_method}.csv")
    elif mode == "testing":
        in_file = os.path.join(results_dir, f"{mode}_val_{unet_size}_{comparison_images}_{seg_method}.csv")
    elif mode == "human":
        in_file = os.path.join(results_dir, "human_annotated_comparison.csv")
    elif mode == "imageJ":
        in_file = os.path.join(results_dir, "testing_val_imageJ_small_inst_seg.csv")
    else:
        raise ValueError(f"Unknown mode {mode}")

    data = pd.read_csv(in_file, index_col = None)

    if mode == "training":
        n_cells_key = "n_cells_per_img"
        data["count_error_pct"] = (data["count_error_components"] / data[n_cells_key]) * 100

    return data

def _prep_config_for_unet_comparison(
    cfg: dict,
    models_dir: str,
    unet_mode: Literal["large", "medium", "small"]
) -> dict:
    cfg["model_dir"] = models_dir
    cfg["unet_mode"] = unet_mode
    cfg["model_file"] = f"best_{unet_mode}_tiles_S512_seed187.pth"
    return cfg

def generate_unet_comparison(models_dir: str,
                             h5_path: str,
                             unet_base_config: Any,
                             segmenter_class: Any,
                             output_dir: str,
                             output_filename: str = "unet_segmentation_comparison",
                             redo_analysis: bool = False) -> dict:

    output_file = os.path.join(output_dir, f"{output_filename}.dict")
    existing_file = check_for_file(output_file)
    if existing_file is not None and not redo_analysis:
        assert isinstance(existing_file, dict)
        return existing_file

    large_cfg = _prep_config_for_unet_comparison(unet_base_config, models_dir, "large")
    med_cfg = _prep_config_for_unet_comparison(unet_base_config, models_dir, "medium")
    small_cfg = _prep_config_for_unet_comparison(unet_base_config, models_dir, "small")

    large_seg = segmenter_class.from_config(large_cfg)
    med_seg = segmenter_class.from_config(med_cfg)
    small_seg = segmenter_class.from_config(small_cfg)

    img_idx = 47

    h5file = os.path.join(h5_path, "tiles_train.h5")
    with h5py.File(h5file, "r") as f:
        img = np.asarray(f["imgs"][img_idx])


    large_pred = large_seg(img)["probs"]
    med_pred = med_seg(img)["probs"]
    small_pred = small_seg(img)["probs"]

    res = {
        "original": img,
        "small": small_pred,
        "med": med_pred,
        "large": large_pred
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


def _prepare_image(image: np.ndarray, is_segmentation: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_segmentation else cv2.INTER_LINEAR
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
    image = _prepare_image(image, is_segmentation=False)
    return image

def load_or_create_figure_1_image_cache(
    cache_path: str = "./figure_data/figure_1_image_cache.npz",
    model_dir: str = "../scripts/models",
    model_file: str = "best_small_tiles_S512_seed187.pth",
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:

    if os.path.exists(cache_path) and not force_recompute:
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # -------------------------
    # Build segmenter
    # -------------------------
    unet_config = copy.deepcopy(UNET_CONFIG)
    unet_config["model_dir"] = model_dir
    unet_config["model_file"] = model_file
    unet_config["instance_cfg"] = INSTANCE_CONFIG_DICT
    unet_config["thr_cell"] = 0.1
    unet_config["thr_bound"] = 0.1

    seg = SegmenterUNetInference.from_config(unet_config)

    # -------------------------
    # Hardcoded image loading
    # -------------------------
    sim_img = _load_crop_resize_image(
        file_name="000006.tif",
        base_dir="../scripts/image_datasets/imgs",
        crop_params=(500, 200, 1200),
        scale=True,
    )

    mic_img = _load_crop_resize_image(
        file_name="Bild_696.tif",
        base_dir="../scripts/experiment_readout_images/20251021_25720330",
        crop_params=(600, 150, 1450),
        scale=True,
    )

    gp_img = _load_crop_resize_image(
        file_name="PXL_20251107_130141300.jpg",
        base_dir="../scripts/ext_images/20251107_25065521_GooglePixel/",
        crop_params=(300, 1100, 2700),
        scale=True,
    )

    iphone_img = _load_crop_resize_image(
        file_name="IMG_3859.jpeg",
        base_dir="../scripts/ext_images/20251106_25065441_iPhone_XR_JPEG/",
        crop_params=(1200, 50, 2500),
        scale=True,
    )

    # -------------------------
    # Expensive part: inference
    # -------------------------
    sim_seg_labels = seg(sim_img)["instance_labels"]
    mic_seg_labels = seg(mic_img)["instance_labels"]
    gp_seg_labels = seg(gp_img)["instance_labels"]
    iphone_seg_labels = seg(iphone_img)["instance_labels"]

    # -------------------------
    # Convert labels to display RGB
    # -------------------------
    sim_seg_rgb = instance_labels_to_rgb(sim_seg_labels)
    mic_seg_rgb = instance_labels_to_rgb(mic_seg_labels)
    gp_seg_rgb = instance_labels_to_rgb(gp_seg_labels)
    iphone_seg_rgb = instance_labels_to_rgb(iphone_seg_labels)

    data = {
        "simulated_image": sim_img,
        "simulated_segmentation": sim_seg_rgb,
        "microscopy_image": mic_img,
        "microscopy_segmentation": mic_seg_rgb,
        "googlepixel_image": gp_img,
        "googlepixel_segmentation": gp_seg_rgb,
        "iphone_image": iphone_img,
        "iphone_segmentation": iphone_seg_rgb,
    }

    np.savez_compressed(cache_path, **data)

    return data

def _prepare_segmentation_for_display(segmentation: np.ndarray) -> np.ndarray:
    seg = _prepare_image(segmentation, is_segmentation=True)

    if seg.ndim == 3 and seg.shape[2] == 3:
        if seg.dtype != np.uint8:
            seg = np.clip(seg, 0, 255).astype(np.uint8)
        return seg

    if seg.ndim == 2:
        return instance_labels_to_rgb(seg)

    raise ValueError("Unsupported segmentation shape")


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


