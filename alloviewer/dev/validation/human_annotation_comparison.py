# human_annotation_comparison.py

import os
import glob
import json
import copy
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm

from alloviewer.image_analysis.segmenter import SegmenterConfig, SegmenterUNet


# ----------------------------
# small helpers
# ----------------------------

def _decode_json_maybe(x: Any) -> Dict[str, Any]:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, np.bytes_):
        x = x.tobytes().decode("utf-8")
    if isinstance(x, str):
        return json.loads(x)
    if isinstance(x, dict):
        return x
    raise TypeError(f"Could not decode meta entry of type {type(x)}")


def _first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _as_int_pair(x: Any) -> Optional[Tuple[int, int]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return int(x[0]), int(x[1])
    return None


def _count_positive_labels(lbl: np.ndarray) -> int:
    vals = np.unique(lbl)
    vals = vals[vals > 0]
    return int(vals.size)


# ----------------------------
# human csv loading
# ----------------------------

def load_human_roi_counts(csv_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(csv_dir, "human_annotations.csv"))

    needed = {"Folder", "image_name", "roi_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    return (
        df.loc[:, ["Folder", "image_name", "roi_id"]]
        .dropna(subset=["Folder", "image_name", "roi_id"])
        .groupby(["Folder", "image_name"], as_index=False)["roi_id"]
        .nunique()
        .rename(columns={"roi_id": "human_roi_count"})
    )


# ----------------------------
# meta parsing
# ----------------------------

def _extract_image_identity(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract (Folder, image_name) from H5 meta.

    Priority:
      1) direct fields if present
      2) parse from full["src_path"]
      3) parse from first tile's full_meta["src_path"]
    """
    full = meta.get("full", {}) if isinstance(meta.get("full", {}), dict) else {}

    # direct fields first
    folder = _first_present(
        full,
        ["Folder", "folder", "subfolder", "dir_name", "dirname", "source_folder"],
        default=None,
    )
    image_name = _first_present(
        full,
        ["image_name", "filename", "file_name", "img_name", "name"],
        default=None,
    )

    if folder is not None and image_name is not None:
        return str(folder), str(image_name)

    # next: parse src_path from full
    src_path = _first_present(full, ["src_path", "image_path", "img_path", "path"], default=None)

    # fallback: some datasets duplicate full meta inside each tile
    if src_path is None:
        tiles = meta.get("tiles", [])
        if isinstance(tiles, list) and len(tiles) > 0:
            full_meta = tiles[0].get("full_meta", {})
            if isinstance(full_meta, dict):
                src_path = _first_present(
                    full_meta,
                    ["src_path", "image_path", "img_path", "path"],
                    default=None,
                )

    if src_path is not None:
        src_path = os.path.normpath(str(src_path))
        image_name = os.path.basename(src_path)
        folder = os.path.basename(os.path.dirname(src_path))
        if folder and image_name:
            return folder, image_name

    raise KeyError(
        "Could not extract Folder/image_name from H5 meta. "
        f"Available top-level keys: {list(meta.keys())}, "
        f"full keys: {list(full.keys()) if isinstance(full, dict) else 'n/a'}"
    )

def _extract_full_hw(meta: Dict[str, Any], tile_metas: List[Dict[str, Any]], tile_hw: Tuple[int, int]) -> Tuple[int, int]:
    """
    Tries to get full image size from:
      1) meta["full"]
      2) first tile's full_meta
      3) fallback from tile extents
    """
    full = meta.get("full", {}) if isinstance(meta.get("full", {}), dict) else {}

    # fallback: many datasets keep the real full meta inside each tile entry
    if not full and len(tile_metas) > 0:
        fm = tile_metas[0].get("full_meta", {})
        if isinstance(fm, dict):
            full = fm

    H = _first_present(full, ["height", "H", "img_h", "image_height"], default=None)
    W = _first_present(full, ["width", "W", "img_w", "image_width"], default=None)
    if H is not None and W is not None:
        return int(H), int(W)

    shape = _first_present(full, ["shape", "image_shape", "full_shape", "hw"], default=None)
    hw = _as_int_pair(shape)
    if hw is not None:
        return int(hw[0]), int(hw[1])

    # fallback from tile boxes
    max_y = 0
    max_x = 0
    for tm in tile_metas:
        y0, y1, x0, x1 = _extract_tile_box(tm, tile_hw)
        max_y = max(max_y, y1)
        max_x = max(max_x, x1)

    if max_y <= 0 or max_x <= 0:
        raise ValueError("Could not infer full image size from H5 metadata.")

    return max_y, max_x

def _extract_tile_box(tile_meta: Dict[str, Any], tile_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Returns (y0, y1, x0, x1) for one tile.
    Supports the actual metadata format with:
      - tile_xy
      - tile_hw
    """

    # --- your actual format ---
    if "tile_xy" in tile_meta:
        xy = tile_meta["tile_xy"]
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            raise ValueError(f"tile_xy has invalid format: {xy}")

        # assuming tile_xy = [x0, y0]
        x0 = int(xy[0])
        y0 = int(xy[1])

        hw = tile_meta.get("tile_hw", tile_hw)
        if not isinstance(hw, (list, tuple)) or len(hw) < 2:
            raise ValueError(f"tile_hw has invalid format: {hw}")

        th = int(hw[0])
        tw = int(hw[1])

        return y0, y0 + th, x0, x0 + tw

    # --- old generic fallbacks ---
    y0 = _first_present(tile_meta, ["y0", "top", "row0", "r0"], default=None)
    y1 = _first_present(tile_meta, ["y1", "bottom", "row1", "r1"], default=None)
    x0 = _first_present(tile_meta, ["x0", "left", "col0", "c0"], default=None)
    x1 = _first_present(tile_meta, ["x1", "right", "col1", "c1"], default=None)

    if None not in (y0, y1, x0, x1):
        return int(y0), int(y1), int(x0), int(x1)

    origin = _first_present(tile_meta, ["origin", "xy0", "yx0", "start"], default=None)
    shape = _first_present(tile_meta, ["shape", "hw", "tile_shape", "size"], default=None)

    if origin is not None:
        if isinstance(origin, (list, tuple)) and len(origin) >= 2:
            oy, ox = int(origin[0]), int(origin[1])
        else:
            oy, ox = None, None
    else:
        oy = _first_present(tile_meta, ["y", "row", "top"], default=None)
        ox = _first_present(tile_meta, ["x", "col", "left"], default=None)

    if shape is not None and isinstance(shape, (list, tuple)) and len(shape) >= 2:
        th, tw = int(shape[0]), int(shape[1])
    else:
        th = _first_present(tile_meta, ["h", "height", "tile_h"], default=tile_hw[0])
        tw = _first_present(tile_meta, ["w", "width", "tile_w"], default=tile_hw[1])

    if oy is not None and ox is not None:
        return int(oy), int(oy) + int(th), int(ox), int(ox) + int(tw)

    raise KeyError(
        "Could not extract tile box from tile metadata. "
        f"Available tile keys: {list(tile_meta.keys())}"
    )

# ----------------------------
# stitching
# ----------------------------

def stitch_prob_tiles(
    prob_tiles: np.ndarray,
    tile_metas: List[Dict[str, Any]],
    full_hw: Tuple[int, int],
) -> np.ndarray:
    """
    prob_tiles: [T, 4, tile_h, tile_w]
    returns:    [4, H, W]
    """
    if prob_tiles.ndim != 4:
        raise ValueError(f"Expected prob_tiles [T,4,h,w], got {prob_tiles.shape}")

    T, C, tile_h, tile_w = prob_tiles.shape
    if C != 4:
        raise ValueError(f"Expected 4 channels, got {C}")

    if len(tile_metas) != T:
        raise ValueError(
            f"Number of tile metas ({len(tile_metas)}) does not match number of tiles ({T})"
        )

    H, W = full_hw
    acc = np.zeros((C, H, W), dtype=np.float32)
    wgt = np.zeros((1, H, W), dtype=np.float32)

    for i, tm in enumerate(tile_metas):
        y0, y1, x0, x1 = _extract_tile_box(tm, (tile_h, tile_w))

        th = max(0, min(y1, H) - max(y0, 0))
        tw = max(0, min(x1, W) - max(x0, 0))
        if th == 0 or tw == 0:
            continue

        yy0 = max(y0, 0)
        xx0 = max(x0, 0)

        acc[:, yy0:yy0 + th, xx0:xx0 + tw] += prob_tiles[i, :, :th, :tw]
        wgt[:, yy0:yy0 + th, xx0:xx0 + tw] += 1.0

    wgt[wgt == 0] = 1.0
    return (acc / wgt).astype(np.float32)


# ----------------------------
# per-image comparison
# ----------------------------

def segment_one_h5_entry(
    segmenter: SegmenterUNet,
    imgs_tiles: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    imgs_tiles: [T, 3, S, S]
    meta: decoded JSON dict
    """
    if imgs_tiles.ndim != 4 or imgs_tiles.shape[1] != 3:
        raise ValueError(f"Expected imgs_tiles [T,3,H,W], got {imgs_tiles.shape}")

    tile_metas = meta.get("tiles", None)
    if not isinstance(tile_metas, list) or len(tile_metas) == 0:
        raise ValueError("H5 meta['tiles'] is missing or empty.")

    T = len(tile_metas)
    imgs_tiles = imgs_tiles[:T]

    # tile inference
    tiles_t = segmenter._to_tensor_tiles(imgs_tiles)
    probs_t = segmenter.predict_tiles(tiles_t).numpy()   # [T,4,h,w]

    full_hw = _extract_full_hw(meta, tile_metas, imgs_tiles.shape[-2:])
    probs_full = stitch_prob_tiles(probs_t, tile_metas, full_hw)

    cell_p = probs_full[0]
    bound_p = probs_full[1]
    center_p = probs_full[2]
    energy_p = probs_full[3]

    seg_out = {
        "probs": {
            "cell": cell_p,
            "bound": bound_p,
            "center": center_p,
            "energy": energy_p,
        },
        "instance_labels": None,
        "meta": {},
    }

    if segmenter.inst_seg is None:
        raise RuntimeError(
            "Segmenter has no instance segmenter. "
            "Set cfg.compute_instances=True."
        )

    seg_out = segmenter.inst_seg(seg_out, update_cell_mask=False)
    labels = seg_out["instance_labels"]

    folder, image_name = _extract_image_identity(meta)

    return {
        "Folder": folder,
        "image_name": image_name,
        "unet_roi_count": _count_positive_labels(labels),
    }


# ----------------------------
# main entry
# ----------------------------

def compare_human_annotations(
    h5_path: str = "./image_datasets/human_annotated_images.h5",
    human_csv_dir: str = "./human_annotations",
    output_csv: str = "./results/human_annotated_comparison.csv",
    segmenter_cfg: Optional[SegmenterConfig] = None,
) -> pd.DataFrame:
    """
    Runs UNet on all images in the H5 file, counts ROIs, joins with human counts,
    and writes the result CSV.
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # load human counts
    human_df = load_human_roi_counts(human_csv_dir)

    # build segmenter
    if segmenter_cfg is None:
        segmenter_cfg = SegmenterConfig(
            compute_instances=True,
            input_is_tiles=True,
        )
    else:
        segmenter_cfg = copy.deepcopy(segmenter_cfg)
        segmenter_cfg.compute_instances = True
        segmenter_cfg.input_is_tiles = True

    segmenter = SegmenterUNet(segmenter_cfg)

    rows: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as f:
        if "imgs" not in f or "meta" not in f:
            raise KeyError("H5 file must contain datasets '/imgs' and '/meta'.")

        imgs_ds = f["imgs"]
        meta_ds = f["meta"]

        n_total = int(imgs_ds.shape[0])
        n_written = int(f.attrs.get("written", n_total))
        n_use = min(n_total, n_written)

        for i in tqdm(range(n_use), desc="Comparing human vs UNet", dynamic_ncols=True):
            meta = _decode_json_maybe(meta_ds[i])
            imgs_tiles = imgs_ds[i]   # [Tmax, 3, S, S]
            row = segment_one_h5_entry(segmenter, imgs_tiles, meta)
            rows.append(row)

    unet_df = pd.DataFrame(rows)

    out_df = human_df.merge(
        unet_df,
        on=["Folder", "image_name"],
        how="outer",
        validate="one_to_one",
    )

    out_df = out_df.sort_values(["Folder", "image_name"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    return out_df

# ----------------------------
# simple script use
# ----------------------------

def run_human_annotation_comparison():
    cfg = SegmenterConfig(
        unet_mode="small",          # change as needed
        model_dir="./models",
        model_file="best_small_tiles_S512_seed187.pth",            # uses best_small.pth / best_medium.pth / best_large.pth
        device="cuda",
        use_amp=True,
        compute_instances=True,
        input_is_tiles=True,
        normalize=True,
        instance_cfg={},            # fill if you want custom instance settings
    )

    df = compare_human_annotations(
        h5_path="./image_datasets/human_annotated_images.h5",
        human_csv_dir="./human_annotations",
        output_csv="./results/human_annotated_comparison.csv",
        segmenter_cfg=cfg,
    )

    print(df.head())
    print(f"\nSaved: ./results/human_annotated_comparison.csv")
