from __future__ import annotations

import os
import json
import gc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.measure import label as sklabel

from alloviewer.dev.segmentation.utils import collate_no_meta
from alloviewer.image_analysis.segmenter import (
    SegmenterUNet,
    SegmenterConfig,
    InstanceSegmenterConfig,
)

from ..segmentation import TiledH5Dataset
from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,
    center_metrics_hungarian,
    energy_metrics_extended_full,
    get_dataset_mode,
)
from .config import TrainingValidationConfig


def _count_positive_labels(inst: np.ndarray) -> int:
    vals = np.unique(inst)
    vals = vals[vals > 0]
    return int(vals.size)


def _get_original_sample_idx(
    loop_idx: int,
    indices: Optional[Sequence[int]],
) -> int:
    if indices is None:
        return int(loop_idx)
    return int(indices[loop_idx])


def _extract_full_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    full_meta = meta.get("full", {})
    if isinstance(full_meta, dict):
        return full_meta
    return {}


def _extract_tile_meta(meta: Dict[str, Any], tile_idx: int) -> Dict[str, Any]:
    tiles = meta.get("tiles", [])
    if isinstance(tiles, list) and tile_idx < len(tiles):
        tile_meta = tiles[tile_idx]
        if isinstance(tile_meta, dict):
            return tile_meta
    return {}


def _jsonify_params(params: Dict[str, Any]) -> Dict[str, Any]:
    params_out: Dict[str, Any] = {}

    for k, v in params.items():
        col = f"param_{k}"

        if isinstance(v, (int, float, np.integer, np.floating)):
            params_out[col] = float(v)
        elif isinstance(v, (list, tuple, np.ndarray)):
            try:
                params_out[col] = json.dumps([float(x) for x in list(v)])
            except Exception:
                continue
        else:
            continue

    return params_out


def _make_gt_seg_dict(
    tgt: np.ndarray,
    gt_segmenter: SegmenterUNet,
) -> Dict[str, Any]:
    return {
        "probs": {
            "cell": (tgt[0] > gt_segmenter.cfg.thr_cell).astype(np.uint8),
            "bound": (tgt[1] > gt_segmenter.cfg.thr_bound).astype(np.uint8),
            "center": tgt[2].astype(np.float32),
            "energy": tgt[3].astype(np.float32),
        },
        "cell_mask": (tgt[0] >= gt_segmenter.cfg.thr_cell).astype(np.uint8),
        "boundary": (tgt[1] >= gt_segmenter.cfg.thr_bound).astype(np.uint8),
        "instance_labels": None,
        "meta": {},
    }


def _get_prediction_mask_and_counts(
    *,
    out: Dict[str, Any],
    tile_idx: int,
    segmentation_method: str,
    cell_prob: np.ndarray,
    cfg: TrainingValidationConfig,
) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
      cell_pred_bin
      n_components
      n_pred_instances

    For conventional mode, n_pred_instances is set equal to the connected
    component count so plotting code always has a usable column.
    """
    if segmentation_method == "conventional":
        cell_pred_bin = (cell_prob >= cfg.cell_thr).astype(np.uint8)
        n_components = int(sklabel(cell_pred_bin, connectivity=1).max())
        n_pred_instances = n_components
        return cell_pred_bin, n_components, n_pred_instances

    if segmentation_method == "inst_seg":
        inst_pred_list = out.get("instance_labels", None)
        if inst_pred_list is None:
            raise RuntimeError(
                "segmentation_method='inst_seg' but segmenter did not return instance_labels"
            )

        inst_pred = np.asarray(inst_pred_list[tile_idx], dtype=np.int32)
        cell_pred_bin = (inst_pred > 0).astype(np.uint8)

        n_pred_instances = _count_positive_labels(inst_pred)
        n_components = n_pred_instances

        return cell_pred_bin, n_components, n_pred_instances

    raise ValueError(f"Unknown segmentation_method: {segmentation_method}")


def validate_unet_on_tiled_h5(
    segmenter: SegmenterUNet,
    gt_segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "conventional",
    stop: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tile-level validation for simulated or external tiled H5 datasets.

    One output row is written per tile.

    segmentation_method:
        - "conventional": threshold P(cell), count connected components
        - "inst_seg": use InstanceSegmenter output
    """
    if segmentation_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown segmentation_method: {segmentation_method}")

    if segmentation_method == "inst_seg" and segmenter.inst_seg is None:
        raise ValueError(
            "segmentation_method='inst_seg' requires SegmenterUNet with compute_instances=True"
        )

    if gt_segmenter.inst_seg is None:
        raise ValueError("gt_segmenter must have compute_instances=True")

    ds = TiledH5Dataset(cfg.h5_path, indices=indices)

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta,
    )

    rows: List[Dict[str, Any]] = []
    dataset_mode = get_dataset_mode(cfg.h5_path)

    with tqdm(
        total=len(ds),
        desc=f"validate tiled h5 ({dataset_mode}, {segmentation_method})",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for loop_idx, batch in enumerate(dl):
            if stop is not None and loop_idx == stop:
                break

            original_idx = _get_original_sample_idx(loop_idx, indices)

            imgs_t, tgts_t, extras = batch

            imgs_t = imgs_t[0]                    # [T,3,S,S]
            tgts_t = tgts_t[0]                    # [T,4,S,S]
            inst_t = extras["instance_labels"][0] # [T,S,S]
            meta = extras["meta"][0]

            tile_metas = meta.get("tiles", [])
            if isinstance(tile_metas, list) and len(tile_metas) > 0:
                T = len(tile_metas)
            else:
                T = int(imgs_t.shape[0])

            imgs_np = imgs_t[:T].detach().cpu().numpy().astype(np.float32)
            tgts_np = tgts_t[:T].detach().cpu().numpy().astype(np.float32)
            inst_np = inst_t[:T].detach().cpu().numpy().astype(np.int32)

            out = segmenter(imgs_np)

            cell_prob = out["probs"]["cell"]      # [T,H,W]
            bound_prob = out["probs"]["bound"]
            center_pred = out["probs"]["center"]
            energy_pred = out["probs"]["energy"]

            if segmentation_method == "inst_seg" and out.get("instance_labels", None) is None:
                raise RuntimeError(
                    "segmentation_method='inst_seg' but segmenter did not return instance_labels"
                )

            full_meta = _extract_full_meta(meta)
            n_cells_from_meta = full_meta.get("n_cells", None)
            frac_positive_from_meta = full_meta.get("frac_positive", None)
            src_path = full_meta.get("src_path", "")

            params = full_meta.get("params", {})
            params_out = _jsonify_params(params if isinstance(params, dict) else {})

            for t in range(T):
                tgt = tgts_np[t]           # [4,H,W]
                inst_gt_stored = inst_np[t]

                cell_gt = (tgt[0] > 0.5).astype(np.uint8)
                energy_gt = tgt[3].astype(np.float32)

                cell_p = np.asarray(cell_prob[t], dtype=np.float32)
                bound_p = np.asarray(bound_prob[t], dtype=np.float32)
                center_p = np.asarray(center_pred[t], dtype=np.float32)
                energy_p = np.asarray(energy_pred[t], dtype=np.float32)

                # Reconstruct GT instances from target heads. This keeps behavior
                # compatible with the previous version while counting labels safely.
                gt_inst_seg_dict = _make_gt_seg_dict(tgt, gt_segmenter)
                gt_inst_seg_dict = gt_segmenter.inst_seg(
                    gt_inst_seg_dict,
                    update_cell_mask=True,
                )
                inst_gt = np.asarray(
                    gt_inst_seg_dict["instance_labels"],
                    dtype=np.int32,
                )
                n_gt = _count_positive_labels(inst_gt)

                (
                    cell_pred_bin,
                    n_cc,
                    n_cells_pred_instances,
                ) = _get_prediction_mask_and_counts(
                    out=out,
                    tile_idx=t,
                    segmentation_method=segmentation_method,
                    cell_prob=cell_p,
                    cfg=cfg,
                )

                peaks = nms_peaks_np(
                    center_p,
                    thr=cfg.center_peak_thr,
                    min_dist=cfg.center_nms_dist,
                )
                n_centers_pred = int(len(peaks))

                mask_stats = iou_dice_overlap(cell_pred_bin, cell_gt)

                boundary_f1 = boundary_f1_skeletonized(
                    bound_p,
                    inst_gt,
                    tol=cfg.boundary_tol,
                    thr=cfg.boundary_thr,
                    sweep=cfg.boundary_sweep,
                )

                center_stats = center_metrics_hungarian(
                    center_p,
                    inst_gt,
                    peak_thr=cfg.center_peak_thr,
                    nms_dist=cfg.center_nms_dist,
                    match_radius=cfg.center_match_radius,
                    ap_thr_list=cfg.ap_thr_list,
                    oks_thresholds=cfg.oks_thresholds,
                )

                energy_stats = energy_metrics_extended_full(
                    energy_p,
                    energy_gt,
                    cell_gt,
                    frac_delta=cfg.energy_frac_delta,
                )

                if n_cells_from_meta is not None:
                    try:
                        n_cells_per_img = int(n_cells_from_meta)
                    except Exception:
                        n_cells_per_img = np.nan
                else:
                    n_cells_per_img = np.nan

                tile_meta = _extract_tile_meta(meta, t)
                tile_xy = tile_meta.get("tile_xy", (np.nan, np.nan))
                if isinstance(tile_xy, (list, tuple)) and len(tile_xy) == 2:
                    tile_y = tile_xy[0]
                    tile_x = tile_xy[1]
                else:
                    tile_y = np.nan
                    tile_x = np.nan

                row: Dict[str, Any] = {
                    "sample_idx": int(original_idx),
                    "idx": int(original_idx),  # backward-compatible alias
                    "tile_idx": int(t),
                    "metric_level": "tile",
                    "dataset_mode": dataset_mode,
                    "segmentation_method": segmentation_method,
                    "unet_mode": segmenter.cfg.unet_mode,
                    "src_path": src_path if src_path is not None else "",
                    "tile_y": int(tile_y) if not pd.isna(tile_y) else np.nan,
                    "tile_x": int(tile_x) if not pd.isna(tile_x) else np.nan,
                    "n_cells_per_img": n_cells_per_img,
                    "frac_positive": (
                        float(frac_positive_from_meta)
                        if frac_positive_from_meta is not None else np.nan
                    ),
                    "n_cells_gt_instances": int(n_gt),
                    "n_cells_gt_instances_stored_labels": _count_positive_labels(inst_gt_stored),
                    "n_cells_pred_components_thr0p5": int(n_cc),
                    "n_cells_pred_centers": int(n_centers_pred),
                    "n_cells_pred_instances": int(n_cells_pred_instances),
                    "count_error_components": int(n_cc - n_gt),
                    "count_error_centers": int(n_centers_pred - n_gt),
                    "count_error_instances": int(n_cells_pred_instances - n_gt),
                    "abs_count_error_instances": int(abs(n_cells_pred_instances - n_gt)),
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                    "boundary_f1": float(boundary_f1),
                    **{f"center_{k}": v for k, v in center_stats.items()},
                    **{f"energy_{k}": v for k, v in energy_stats.items()},
                    **params_out,
                }

                rows.append(row)

            pbar.update(1)

    df = pd.DataFrame(rows)

    metric_cols = [
        c for c in df.columns
        if any(
            c.startswith(pfx)
            for pfx in (
                "mask_",
                "boundary_",
                "center_",
                "energy_",
                "count_error_",
                "abs_count_error_",
            )
        )
    ]

    summary = {
        "n_images": int(df["sample_idx"].nunique()) if len(df) else 0,
        "n_tiles": int(len(df)),
        "means": {
            c: float(np.nanmean(df[c].values.astype(np.float64)))
            for c in metric_cols
        },
        "stds": {
            c: float(np.nanstd(df[c].values.astype(np.float64)))
            for c in metric_cols
        },
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
    force: bool = False,
) -> pd.DataFrame:
    """
    Run validation for one combination.

    dataset_mode:
      - "external_images" -> external_images_test.h5
      - "tiles"           -> tiles_test.h5
    """
    if dataset_mode not in ("external_images", "tiles"):
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    if seg_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown seg_method: {seg_method}")

    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"testing_val_{unet_mode}_{dataset_mode}_{seg_method}.csv",
    )
    out_summary_json = os.path.join(
        out_dir,
        f"testing_val_{unet_mode}_{dataset_mode}_{seg_method}_summary.json",
    )

    if os.path.isfile(out_csv) and not force:
        return pd.read_csv(out_csv, index_col=None)

    h5_path = os.path.join(h5_dir, f"{dataset_mode}_test.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    model_file = f"best_{unet_mode}_tiles_S512_seed187.pth"

    cfg = TrainingValidationConfig(
        h5_path=h5_path,
        cell_thr=0.1,
        batch_size=1,
        out_csv=out_csv,
        out_summary_json=out_summary_json,
    )

    seg_params: Dict[str, Any] = dict(
        unet_mode=unet_mode,
        model_dir=model_dir,
        model_file=model_file,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=torch.cuda.is_available(),
        normalize=False,  # TiledH5Dataset/H5CellsDataset already normalizes
    )

    if seg_method == "conventional":
        segmenter_cfg = SegmenterConfig(
            compute_instances=False,
            **seg_params,
        ).to_dict()
    else:
        segmenter_cfg = SegmenterConfig(
            instance_cfg=InstanceSegmenterConfig().to_dict(),
            compute_instances=True,
            **seg_params,
        ).to_dict()

    gt_segmenter_cfg = SegmenterConfig(
        instance_cfg=InstanceSegmenterConfig().to_dict(),
        compute_instances=True,
        **seg_params,
    ).to_dict()

    gt_segmenter = SegmenterUNet.from_config(gt_segmenter_cfg)
    segmenter = SegmenterUNet.from_config(segmenter_cfg)

    df, _ = validate_unet_on_tiled_h5(
        segmenter=segmenter,
        gt_segmenter=gt_segmenter,
        cfg=cfg,
        segmentation_method=seg_method,
    )

    gc.collect()
    return df


def run_test_ext_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    force: bool = False,
) -> None:
    """
    Sequential fallback.

    For cluster use, prefer the SLURM array script.
    """
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
                    force=force,
                )
                res.append(df)

    if len(res) > 0:
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
    parser.add_argument("--dataset-mode", required=True, choices=["external_images", "tiles"])
    parser.add_argument("--seg-method", required=True, choices=["conventional", "inst_seg"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = run_single_ext_validation(
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        h5_dir=args.h5_dir,
        unet_mode=args.unet_mode,
        dataset_mode=args.dataset_mode,
        seg_method=args.seg_method,
        force=args.force,
    )
    print(f"Done. Wrote {len(df)} rows.")
