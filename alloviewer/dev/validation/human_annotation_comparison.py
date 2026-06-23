import os
import copy
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm

from alloviewer.image_analysis.segmenter import SegmenterConfig, SegmenterUNet

from .utils import (
    count_positive_labels,
    decode_json_maybe,
    extract_full_hw,
    extract_image_identity,
    extract_tile_box,
    load_human_roi_counts,
    stitch_prob_tiles,
    segment_one_h5_entry
)


def _as_numpy_prediction(x) -> np.ndarray:
    """
    Convert segmenter output to a NumPy array.

    This handles both CPU/GPU torch tensors and already-materialized NumPy arrays.
    """
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def load_imageJ_roi_counts(csv_dir: str) -> pd.DataFrame:
    """
    Load ImageJ/reference ROI counts from results.csv.

    Kept local because this function was not moved to utils.py.
    """
    df = pd.read_csv(os.path.join(csv_dir, "results.csv"))

    needed = {"Folder", "file_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    out = (
        df.loc[:, ["Folder", "file_name"]]
        .dropna(subset=["Folder", "file_name"])
        .copy()
    )

    out["Folder"] = out["Folder"].astype(str)
    out["file_name"] = out["file_name"].astype(str)

    return (
        out.groupby(["Folder", "file_name"], as_index=False)
        .size()
        .rename(columns={"size": "imageJ_roi_count", "file_name": "image_name"})
    )


def compare_human_annotations(
    h5_path: str = "./image_datasets/human_annotated_images.h5",
    human_csv_dir: str = "./human_annotations",
    output_csv: str = "./results/human_annotated_comparison.csv",
    segmenter_cfg: Optional[SegmenterConfig] = None,
    include_imagej_counts: bool = True,
) -> pd.DataFrame:
    """
    Run UNet on all images in the H5 file, count ROIs, join with human counts,
    optionally join ImageJ/reference counts, and write the result CSV.
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    human_df = load_human_roi_counts(human_csv_dir)

    imagej_df = None
    if include_imagej_counts:
        imagej_df = load_imageJ_roi_counts(human_csv_dir)

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
            meta = decode_json_maybe(meta_ds[i])
            imgs_tiles = imgs_ds[i]
            row = segment_one_h5_entry(segmenter, imgs_tiles, meta)
            rows.append(row)

    unet_df = pd.DataFrame(rows)

    out_df = human_df.merge(
        unet_df,
        on=["Folder", "image_name"],
        how="outer",
        validate="one_to_one",
    )

    if imagej_df is not None:
        out_df = out_df.merge(
            imagej_df,
            on=["Folder", "image_name"],
            how="outer",
            validate="one_to_one",
        )

    out_df = out_df.sort_values(["Folder", "image_name"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    return out_df


def run_human_annotation_comparison():
    cfg = SegmenterConfig(
        unet_mode="small",
        model_dir="./models",
        model_file="best_small_tiles_S512_seed187.pth",
        device="cuda",
        use_amp=True,
        compute_instances=True,
        input_is_tiles=True,
        normalize=True,
        instance_cfg={},
        thr_cell=0.1,
        thr_bound=0.1,
    )

    df = compare_human_annotations(
        h5_path="./image_datasets/human_annotated_images.h5",
        human_csv_dir="./human_annotations",
        output_csv="./results/human_annotated_comparison.csv",
        segmenter_cfg=cfg,
        include_imagej_counts=True,
    )

    print(df.head())
    print("\nSaved: ./results/human_annotated_comparison.csv")


def visualize_human_vs_unet_tile(
    image_idx: int,
    tile_idx: int,
    h5_path: str = "./image_datasets/human_annotated_images.h5",
    segmenter_cfg: Optional[SegmenterConfig] = None,
    figsize: Tuple[int, int] = (20, 10),
):
    """
    Visualize one chosen tile region using full-image stitched UNet inference
    followed by full-image instance segmentation.

    Returns a dict with the cropped arrays.
    """
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    if segmenter_cfg is None:
        segmenter_cfg = SegmenterConfig(
            unet_mode="small",
            model_dir="./models",
            model_file="best_small_tiles_S512_seed187.pth",
            device="cuda",
            use_amp=True,
            compute_instances=True,
            input_is_tiles=True,
            normalize=True,
            instance_cfg={},
            thr_cell=0.1,
            thr_bound=0.1,
        )
    else:
        segmenter_cfg = copy.deepcopy(segmenter_cfg)
        segmenter_cfg.compute_instances = True
        segmenter_cfg.input_is_tiles = True

    segmenter = SegmenterUNet(segmenter_cfg)

    if segmenter.inst_seg is None:
        raise RuntimeError("InstanceSegmenter is missing. Set compute_instances=True.")

    with h5py.File(h5_path, "r") as f:
        if "imgs" not in f or "meta" not in f or "inst" not in f:
            raise KeyError("H5 file must contain '/imgs', '/meta', and '/inst'.")

        imgs_ds = f["imgs"]
        inst_ds = f["inst"]
        meta_ds = f["meta"]

        n_total = int(imgs_ds.shape[0])
        if image_idx < 0 or image_idx >= n_total:
            raise IndexError(f"image_idx={image_idx} out of range [0, {n_total - 1}]")

        meta = decode_json_maybe(meta_ds[image_idx])
        tile_metas = meta.get("tiles", None)

        if not isinstance(tile_metas, list) or len(tile_metas) == 0:
            raise ValueError("H5 meta['tiles'] is missing or empty.")

        if tile_idx < 0 or tile_idx >= len(tile_metas):
            raise IndexError(
                f"tile_idx={tile_idx} out of range [0, {len(tile_metas) - 1}]"
            )

        T = len(tile_metas)
        imgs_tiles = imgs_ds[image_idx, :T]
        ref_inst_tile = inst_ds[image_idx, tile_idx].astype(np.int32)

        tiles_t = segmenter._to_tensor_tiles(imgs_tiles)
        probs_t = _as_numpy_prediction(segmenter.predict_tiles(tiles_t))

        full_hw = extract_full_hw(meta, tile_metas, imgs_tiles.shape[-2:])
        probs_full = stitch_prob_tiles(probs_t, tile_metas, full_hw)

        if probs_full.shape[0] < 4:
            raise ValueError(
                f"Expected at least 4 probability channels after stitching, got {probs_full.shape}"
            )

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

        seg_out = segmenter.inst_seg(seg_out, update_cell_mask=True)

        full_instances = seg_out["instance_labels"].astype(np.int32)
        full_cell_mask = seg_out["cell_mask"].astype(np.uint8)

        if "boundary" in seg_out:
            full_boundary_mask = seg_out["boundary"].astype(np.uint8)
        else:
            full_boundary_mask = (bound_p >= segmenter_cfg.thr_bound).astype(np.uint8)

        y0, y1, x0, x1 = extract_tile_box(
            tile_metas[tile_idx],
            imgs_tiles.shape[-2:],
        )

        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(full_hw[0], y1)
        x1 = min(full_hw[1], x1)

        pred_inst_crop = full_instances[y0:y1, x0:x1]

        cell_p_crop = cell_p[y0:y1, x0:x1]
        bound_p_crop = bound_p[y0:y1, x0:x1]
        center_p_crop = center_p[y0:y1, x0:x1]
        energy_p_crop = energy_p[y0:y1, x0:x1]

        cell_mask_crop = full_cell_mask[y0:y1, x0:x1]
        boundary_mask_crop = full_boundary_mask[y0:y1, x0:x1]

        img_tile = imgs_tiles[tile_idx]
        img_rgb = np.transpose(img_tile, (1, 2, 0)).astype(np.float32)

        if img_rgb.max() > 1.0:
            img_rgb = img_rgb / 255.0

        img_rgb = np.clip(img_rgb, 0.0, 1.0)

        h_crop, w_crop = pred_inst_crop.shape
        img_rgb = img_rgb[:h_crop, :w_crop]
        ref_inst_tile = ref_inst_tile[:h_crop, :w_crop]

        ref_overlay = label2rgb(
            ref_inst_tile,
            image=img_rgb,
            bg_label=0,
            alpha=0.45,
        )

        pred_overlay = label2rgb(
            pred_inst_crop,
            image=img_rgb,
            bg_label=0,
            alpha=0.45,
        )

        folder, image_name = extract_image_identity(meta)

        fig, axes = plt.subplots(2, 4, figsize=figsize)

        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title(f"Original\nimage={image_idx}, tile={tile_idx}")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(ref_overlay)
        axes[0, 1].set_title(
            f"ImageJ overlay\ncount={count_positive_labels(ref_inst_tile)}"
        )
        axes[0, 1].axis("off")

        axes[0, 2].imshow(pred_overlay)
        axes[0, 2].set_title(
            f"UNET instance labels\ncount={count_positive_labels(pred_inst_crop)}"
        )
        axes[0, 2].axis("off")

        axes[0, 3].axis("off")

        im_cell = axes[1, 0].imshow(cell_p_crop, vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("UNET cell mask/head")
        axes[1, 0].axis("off")
        fig.colorbar(im_cell, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im_bound = axes[1, 1].imshow(bound_p_crop, vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("UNET cell bounds/head")
        axes[1, 1].axis("off")
        fig.colorbar(im_bound, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im_center = axes[1, 2].imshow(center_p_crop, vmin=0.0, vmax=1.0)
        axes[1, 2].set_title("UNET centers/head")
        axes[1, 2].axis("off")
        fig.colorbar(im_center, ax=axes[1, 2], fraction=0.046, pad=0.04)

        im_energy = axes[1, 3].imshow(energy_p_crop, vmin=0.0, vmax=1.0)
        axes[1, 3].set_title("UNET distances / energy head")
        axes[1, 3].axis("off")
        fig.colorbar(im_energy, ax=axes[1, 3], fraction=0.046, pad=0.04)

        fig.suptitle(f"{folder} / {image_name}", fontsize=14)
        plt.tight_layout()
        plt.show()

        return {
            "image_idx": image_idx,
            "tile_idx": tile_idx,
            "Folder": folder,
            "image_name": image_name,
            "tile_box": (y0, y1, x0, x1),
            "img_tile": img_rgb,
            "reference_instance_labels": ref_inst_tile,
            "imagej_overlay": ref_overlay,
            "unet_overlay": pred_overlay,
            "unet_cell_probability": cell_p_crop,
            "unet_boundary_probability": bound_p_crop,
            "unet_center_probability": center_p_crop,
            "unet_energy_probability": energy_p_crop,
            "unet_cell_mask": cell_mask_crop,
            "unet_boundary_mask": boundary_mask_crop,
            "unet_instance_labels": pred_inst_crop,
            "seg_out": seg_out,
        }

