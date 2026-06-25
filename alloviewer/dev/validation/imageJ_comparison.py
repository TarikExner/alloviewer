from __future__ import annotations

import gc
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from skimage.measure import label as sklabel
from torch.utils.data import DataLoader

from ..segmentation import TiledH5Dataset
from ..segmentation.utils import collate_no_meta

from alloviewer.image_analysis.segmenter import (
    SegmenterUNet,
    SegmenterConfig,
    InstanceSegmenterConfig,
)

from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,
    center_metrics_hungarian,
    energy_metrics_extended_full,
)
from .config import TrainingValidationConfig


VALIDATION_VERSION = "imagewide_imagej_stored_gt_v2"


def _tile_signature(meta: Dict[str, Any]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    tiles = meta.get("tiles", None)
    if not isinstance(tiles, list) or len(tiles) == 0:
        raise ValueError("meta['tiles'] missing or empty")

    sig: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for tm in tiles:
        if not isinstance(tm, dict):
            raise TypeError(f"Tile metadata must be dict, got {type(tm)}")
        if "tile_xy" not in tm or "tile_hw" not in tm:
            raise KeyError(
                f"Tile metadata missing tile_xy/tile_hw. Keys: {list(tm.keys())}"
            )

        y0, x0 = tm["tile_xy"]
        h, w = tm["tile_hw"]
        sig.append(((int(y0), int(x0)), (int(h), int(w))))

    return sig


def _assert_same_tile_grid(
    meta_gt: Dict[str, Any],
    meta_ij: Dict[str, Any],
    sample_idx: int,
) -> None:
    sig_gt = _tile_signature(meta_gt)
    sig_ij = _tile_signature(meta_ij)

    if sig_gt != sig_ij:
        raise RuntimeError(
            f"Tile grid mismatch at sample_idx={sample_idx}: "
            f"GT has {len(sig_gt)} tiles, ImageJ has {len(sig_ij)} tiles.\n"
            f"GT first/last: {sig_gt[:3]} ... {sig_gt[-3:]}\n"
            f"IJ first/last: {sig_ij[:3]} ... {sig_ij[-3:]}"
        )


def _basename_from_meta(meta: Dict[str, Any]) -> str:
    full = meta.get("full", {})
    src_path = ""

    if isinstance(full, dict):
        src_path = str(full.get("src_path", ""))

    if not src_path:
        tiles = meta.get("tiles", [])
        if isinstance(tiles, list) and len(tiles) > 0 and isinstance(tiles[0], dict):
            full_meta = tiles[0].get("full_meta", {})
            if isinstance(full_meta, dict):
                src_path = str(full_meta.get("src_path", ""))

    return os.path.basename(src_path)


def _assert_same_sample_identity(
    meta_gt: Dict[str, Any],
    meta_ij: Dict[str, Any],
    sample_idx: int,
) -> None:
    """
    Best-effort check.

    If both names are present, compare stems. This catches most order bugs.
    If names are missing, do not fail, because simulated GT metadata may not
    always contain src_path.
    """
    name_gt = _basename_from_meta(meta_gt)
    name_ij = _basename_from_meta(meta_ij)

    if not name_gt or not name_ij:
        return

    stem_gt = os.path.splitext(name_gt)[0]
    stem_ij = os.path.splitext(name_ij)[0]

    if (
        stem_gt != stem_ij
        and not stem_ij.startswith(stem_gt)
        and not stem_gt.startswith(stem_ij)
    ):
        raise RuntimeError(
            f"Sample identity mismatch at sample_idx={sample_idx}: "
            f"GT={name_gt}, ImageJ={name_ij}"
        )


def _extract_full_hw(
    meta: Dict[str, Any],
    tile_metas: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """
    Return full image size as (H, W).

    Priority:
      1. meta['full']['H_in'], meta['full']['W_in']
      2. meta['full']['H'], meta['full']['W']
      3. meta['full']['height'], meta['full']['width']
      4. max tile extent
    """
    full = meta.get("full", {})
    if isinstance(full, dict):
        for hk, wk in (
            ("H_in", "W_in"),
            ("H", "W"),
            ("height", "width"),
            ("image_height", "image_width"),
        ):
            if hk in full and wk in full:
                return int(full[hk]), int(full[wk])

    max_y = 0
    max_x = 0
    for tm in tile_metas:
        y0, x0 = tm["tile_xy"]
        h, w = tm["tile_hw"]
        max_y = max(max_y, int(y0) + int(h))
        max_x = max(max_x, int(x0) + int(w))

    if max_y <= 0 or max_x <= 0:
        raise ValueError("Could not infer full image size.")

    return int(max_y), int(max_x)


def _int_or_nan(x: Any) -> Any:
    try:
        if x is None or pd.isna(x):
            return np.nan
        return int(x)
    except Exception:
        return np.nan


def _float_or_nan(x: Any) -> float:
    try:
        if x is None or pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


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

    return params_out


def stitch_float_tiles_mean(
    tiles: np.ndarray,
    tile_metas: List[Dict[str, Any]],
    full_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Stitch float tiles by averaging overlaps.

    Parameters
    ----------
    tiles:
        [T, C, S, S]
    tile_metas:
        Metadata with tile_xy=(y0,x0), tile_hw=(h,w).
    full_hw:
        (H_full, W_full)

    Returns
    -------
    out:
        [C, H_full, W_full]
    """
    if tiles.ndim != 4:
        raise ValueError(f"Expected tiles [T,C,H,W], got {tiles.shape}")

    T, C, _, _ = tiles.shape

    if len(tile_metas) != T:
        raise ValueError(
            f"Tile count mismatch: tiles={T}, metadata={len(tile_metas)}"
        )

    H_full, W_full = map(int, full_hw)

    acc = np.zeros((C, H_full, W_full), dtype=np.float32)
    wgt = np.zeros((1, H_full, W_full), dtype=np.float32)

    for i, tm in enumerate(tile_metas):
        y0, x0 = tm["tile_xy"]
        h, w = tm["tile_hw"]

        y0 = int(y0)
        x0 = int(x0)
        h = int(h)
        w = int(w)

        y1 = min(H_full, y0 + h)
        x1 = min(W_full, x0 + w)

        hh = y1 - y0
        ww = x1 - x0

        if hh <= 0 or ww <= 0:
            continue

        acc[:, y0:y1, x0:x1] += tiles[i, :, :hh, :ww]
        wgt[:, y0:y1, x0:x1] += 1.0

    wgt[wgt == 0] = 1.0
    return (acc / wgt).astype(np.float32)


def stitch_instance_tiles(
    inst_tiles: np.ndarray,
    tile_metas: List[Dict[str, Any]],
    full_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Stitch global-ID instance tiles.

    This assumes instance labels are global across tiles. The dataset writer
    keeps global IDs in /inst, so this is valid for the simulated GT dataset.
    """
    if inst_tiles.ndim != 3:
        raise ValueError(f"Expected inst_tiles [T,H,W], got {inst_tiles.shape}")

    T, _, _ = inst_tiles.shape

    if len(tile_metas) != T:
        raise ValueError(
            f"Tile count mismatch: inst_tiles={T}, metadata={len(tile_metas)}"
        )

    H_full, W_full = map(int, full_hw)
    out = np.zeros((H_full, W_full), dtype=np.int32)

    for i, tm in enumerate(tile_metas):
        y0, x0 = tm["tile_xy"]
        h, w = tm["tile_hw"]

        y0 = int(y0)
        x0 = int(x0)
        h = int(h)
        w = int(w)

        y1 = min(H_full, y0 + h)
        x1 = min(W_full, x0 + w)

        hh = y1 - y0
        ww = x1 - x0

        if hh <= 0 or ww <= 0:
            continue

        tile = inst_tiles[i, :hh, :ww].astype(np.int32, copy=False)

        region = out[y0:y1, x0:x1]
        mask = tile > 0
        region[mask] = tile[mask]
        out[y0:y1, x0:x1] = region

    return out


def _count_positive_labels(inst: np.ndarray) -> int:
    vals = np.unique(inst)
    vals = vals[vals > 0]
    return int(vals.size)


def _make_seg_out(
    cell: np.ndarray,
    bound: np.ndarray,
    center: np.ndarray,
    energy: np.ndarray,
) -> Dict[str, Any]:
    return {
        "probs": {
            "cell": cell.astype(np.float32),
            "bound": bound.astype(np.float32),
            "center": center.astype(np.float32),
            "energy": energy.astype(np.float32),
        },
        "instance_labels": None,
        "meta": {},
    }


def _segment_full_probs(
    inst_segmenter: Any,
    cell: np.ndarray,
    bound: np.ndarray,
    center: np.ndarray,
    energy: np.ndarray,
    *,
    cell_thr: float,
    update_cell_mask: bool = True,
) -> np.ndarray:
    """
    Convert full-image probability maps into an instance map.

    If inst_segmenter is None, this is the conventional baseline and uses
    cfg.cell_thr via the required cell_thr argument. Do not hard-code 0.5 here.
    """
    if inst_segmenter is None:
        return sklabel(
            (cell >= float(cell_thr)).astype(np.uint8),
            connectivity=1,
        ).astype(np.int32)

    seg_out = _make_seg_out(cell, bound, center, energy)
    seg_out = inst_segmenter(seg_out, update_cell_mask=update_cell_mask)
    return np.asarray(seg_out["instance_labels"], dtype=np.int32)


def _row_metrics_for_prediction(
    *,
    sample_idx: int,
    dataset_mode: str,
    segmentation_method: str,
    cell_pred: np.ndarray,
    bound_pred: np.ndarray,
    center_pred: np.ndarray,
    energy_pred: np.ndarray,
    inst_pred: np.ndarray,
    cell_gt: np.ndarray,
    energy_gt: np.ndarray,
    inst_gt: np.ndarray,
    cfg: TrainingValidationConfig,
    src_path: str,
    n_cells_full_meta: Optional[int],
    frac_positive: Optional[float],
    params_out: Dict[str, Any],
) -> Dict[str, Any]:
    n_gt = _count_positive_labels(inst_gt)
    n_pred_instances = _count_positive_labels(inst_pred)

    cell_pred_bin = (inst_pred > 0).astype(np.uint8)

    cell_pred_thr = (cell_pred >= cfg.cell_thr).astype(np.uint8)
    n_pred_components = int(sklabel(cell_pred_thr, connectivity=1).max())

    peaks = nms_peaks_np(
        center_pred,
        thr=cfg.center_peak_thr,
        min_dist=cfg.center_nms_dist,
    )
    n_centers = int(len(peaks))

    mask_stats = iou_dice_overlap(cell_pred_bin, cell_gt)

    boundary_f1 = boundary_f1_skeletonized(
        bound_pred,
        inst_gt,
        tol=cfg.boundary_tol,
        thr=cfg.boundary_thr,
        sweep=cfg.boundary_sweep,
    )

    center_stats = center_metrics_hungarian(
        center_pred,
        inst_gt,
        peak_thr=cfg.center_peak_thr,
        nms_dist=cfg.center_nms_dist,
        match_radius=cfg.center_match_radius,
        ap_thr_list=cfg.ap_thr_list,
        oks_thresholds=cfg.oks_thresholds,
    )

    energy_stats = energy_metrics_extended_full(
        energy_pred,
        energy_gt,
        cell_gt,
        frac_delta=cfg.energy_frac_delta,
    )

    return {
        "validation_version": VALIDATION_VERSION,
        "sample_idx": int(sample_idx),
        "tile_idx": -1,
        "metric_level": "image",
        "dataset_mode": dataset_mode,
        "segmentation_method": segmentation_method,
        "src_path": src_path if src_path is not None else "",
        "n_cells_full_meta": _int_or_nan(n_cells_full_meta),
        "frac_positive": _float_or_nan(frac_positive),
        "n_cells_gt_instances": int(n_gt),
        "n_cells_pred_components_cell_thr": int(n_pred_components),
        "cell_component_thr": float(cfg.cell_thr),
        "n_cells_pred_centers": int(n_centers),
        "n_cells_pred_instances": int(n_pred_instances),
        "count_error_components_vs_gt_instances": int(n_pred_components - n_gt),
        "count_error_centers_vs_gt_instances": int(n_centers - n_gt),
        "count_error_instances_vs_gt_instances": int(n_pred_instances - n_gt),
        "abs_count_error_instances_vs_gt_instances": int(abs(n_pred_instances - n_gt)),
        **{f"mask_{k}": v for k, v in mask_stats.items()},
        "boundary_f1": float(boundary_f1),
        **{f"center_{k}": v for k, v in center_stats.items()},
        **{f"energy_{k}": v for k, v in energy_stats.items()},
        **params_out,
    }


def _make_segmenter_and_validation_config(
    *,
    gt_h5_path: str,
    out_csv: str,
    out_summary_json: str,
    unet_mode: str,
    model_dir: str,
    model_file: str,
    seg_method: str,
) -> Tuple[SegmenterUNet, SegmenterUNet, TrainingValidationConfig]:
    """
    Build the UNet segmenter, the ImageJ instance segmenter holder, and a
    validation config aligned with the segmentation defaults.
    """
    instance_cfg = InstanceSegmenterConfig()

    seg_params: Dict[str, Any] = dict(
        unet_mode=unet_mode,
        model_dir=model_dir,
        model_file=model_file,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=torch.cuda.is_available(),
        normalize=False,  # TiledH5Dataset/H5CellsDataset already normalizes.
    )

    segmenter_cfg_obj = SegmenterConfig(
        compute_instances=(seg_method == "inst_seg"),
        instance_cfg=instance_cfg.to_dict(),
        **seg_params,
    )

    # This object is used only to access .inst_seg for ImageJ target maps.
    imagej_segmenter_cfg_obj = SegmenterConfig(
        compute_instances=True,
        instance_cfg=instance_cfg.to_dict(),
        **seg_params,
    )

    cfg = TrainingValidationConfig(
        h5_path=gt_h5_path,
        cell_thr=float(segmenter_cfg_obj.thr_cell),
        center_peak_thr=float(instance_cfg.center_thr),
        center_nms_dist=int(instance_cfg.center_min_distance),
        boundary_thr=float(segmenter_cfg_obj.thr_bound),
        boundary_sweep=True,
        batch_size=1,
        out_csv=out_csv,
        out_summary_json=out_summary_json,
    )

    segmenter = SegmenterUNet.from_config(segmenter_cfg_obj.to_dict())
    imagej_segmenter = SegmenterUNet.from_config(imagej_segmenter_cfg_obj.to_dict())

    return segmenter, imagej_segmenter, cfg


def validate_unet_vs_imagej_on_fullres_h5(
    segmenter: SegmenterUNet,
    imagej_segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    imagej_h5_path: str,
    indices: Optional[Sequence[int]] = None,
    segmentation_method: str = "conventional",
    stop: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Image-wide validation on simulated full-resolution images.

    The H5 files are tile-based, but metrics are computed after stitching all
    tiles back to full-image maps.

    Rows:
      - dataset_mode == "UNet"
      - dataset_mode == "imageJ"

    Ground-truth policy:
      - simulated GT uses stitched stored /inst labels;
      - ImageJ instances are reconstructed from ImageJ-generated target maps;
      - UNet conventional mode uses cfg.cell_thr, not a hard-coded threshold.
    """
    if segmentation_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown segmentation_method: {segmentation_method}")

    if imagej_segmenter.inst_seg is None:
        raise ValueError("imagej_segmenter must have compute_instances=True")

    ds_gt = TiledH5Dataset(cfg.h5_path, indices=indices)
    ds_ij = TiledH5Dataset(imagej_h5_path, indices=indices)

    if len(ds_gt) != len(ds_ij):
        raise RuntimeError(
            f"Ground-truth and ImageJ datasets have different lengths: "
            f"{len(ds_gt)} vs {len(ds_ij)}"
        )

    dl_gt = DataLoader(
        ds_gt,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta,
    )
    dl_ij = DataLoader(
        ds_ij,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta,
    )

    rows: List[Dict[str, Any]] = []

    with tqdm(
        total=len(ds_gt),
        desc="validate image-wide GT vs UNet & ImageJ",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for loop_idx, (batch_gt, batch_ij) in enumerate(zip(dl_gt, dl_ij)):
            if stop is not None and loop_idx == stop:
                break

            original_idx = loop_idx if indices is None else int(indices[loop_idx])

            imgs_t, tgts_gt_t, extras_gt = batch_gt
            _, tgts_ij_t, extras_ij = batch_ij

            imgs_t = imgs_t[0]
            tgts_gt_t = tgts_gt_t[0]
            tgts_ij_t = tgts_ij_t[0]

            inst_gt_t = extras_gt["instance_labels"][0]

            meta_gt = extras_gt.get("meta", [{}])[0]
            meta_ij = extras_ij.get("meta", [{}])[0]

            _assert_same_tile_grid(meta_gt, meta_ij, original_idx)
            _assert_same_sample_identity(meta_gt, meta_ij, original_idx)

            tile_metas = meta_gt["tiles"]
            T = len(tile_metas)

            imgs_np = imgs_t[:T].detach().cpu().numpy().astype(np.float32)
            tgts_gt_np = tgts_gt_t[:T].detach().cpu().numpy().astype(np.float32)
            tgts_ij_np = tgts_ij_t[:T].detach().cpu().numpy().astype(np.float32)
            inst_gt_np = inst_gt_t[:T].detach().cpu().numpy().astype(np.int32)

            full_hw = _extract_full_hw(meta_gt, tile_metas)

            tgts_gt_full = stitch_float_tiles_mean(
                tgts_gt_np,
                tile_metas,
                full_hw,
            )

            tgts_ij_full = stitch_float_tiles_mean(
                tgts_ij_np,
                tile_metas,
                full_hw,
            )

            inst_gt_full = stitch_instance_tiles(
                inst_gt_np,
                tile_metas,
                full_hw,
            )

            cell_gt = (tgts_gt_full[0] > 0.5).astype(np.uint8)
            energy_gt = tgts_gt_full[3].astype(np.float32)

            out = segmenter(imgs_np)

            probs_unet_tiles = np.stack(
                [
                    out["probs"]["cell"],
                    out["probs"]["bound"],
                    out["probs"]["center"],
                    out["probs"]["energy"],
                ],
                axis=1,
            ).astype(np.float32)

            probs_unet_full = stitch_float_tiles_mean(
                probs_unet_tiles,
                tile_metas,
                full_hw,
            )

            inst_seg_for_unet = segmenter.inst_seg if segmenter.inst_seg is not None else None

            inst_unet_full = _segment_full_probs(
                inst_segmenter=inst_seg_for_unet,
                cell=probs_unet_full[0],
                bound=probs_unet_full[1],
                center=probs_unet_full[2],
                energy=probs_unet_full[3],
                cell_thr=cfg.cell_thr,
                update_cell_mask=True,
            )

            inst_ij_full = _segment_full_probs(
                inst_segmenter=imagej_segmenter.inst_seg,
                cell=tgts_ij_full[0],
                bound=tgts_ij_full[1],
                center=tgts_ij_full[2],
                energy=tgts_ij_full[3],
                cell_thr=cfg.cell_thr,
                update_cell_mask=True,
            )

            full_meta = meta_gt.get("full", {})
            if not isinstance(full_meta, dict):
                full_meta = {}

            n_cells_from_meta = full_meta.get("n_cells", None)
            frac_positive_from_meta = full_meta.get("frac_positive", None)
            src_path = full_meta.get("src_path", "")

            params = full_meta.get("params", {})
            params_out = _jsonify_params(params if isinstance(params, dict) else {})

            row_unet = _row_metrics_for_prediction(
                sample_idx=original_idx,
                dataset_mode="UNet",
                segmentation_method=segmentation_method,
                cell_pred=probs_unet_full[0],
                bound_pred=probs_unet_full[1],
                center_pred=probs_unet_full[2],
                energy_pred=probs_unet_full[3],
                inst_pred=inst_unet_full,
                cell_gt=cell_gt,
                energy_gt=energy_gt,
                inst_gt=inst_gt_full,
                cfg=cfg,
                src_path=src_path,
                n_cells_full_meta=n_cells_from_meta,
                frac_positive=frac_positive_from_meta,
                params_out=params_out,
            )
            rows.append(row_unet)

            row_ij = _row_metrics_for_prediction(
                sample_idx=original_idx,
                dataset_mode="imageJ",
                segmentation_method="inst_seg",
                cell_pred=tgts_ij_full[0],
                bound_pred=tgts_ij_full[1],
                center_pred=tgts_ij_full[2],
                energy_pred=tgts_ij_full[3],
                inst_pred=inst_ij_full,
                cell_gt=cell_gt,
                energy_gt=energy_gt,
                inst_gt=inst_gt_full,
                cfg=cfg,
                src_path=src_path,
                n_cells_full_meta=n_cells_from_meta,
                frac_positive=frac_positive_from_meta,
                params_out=params_out,
            )
            rows.append(row_ij)

            pbar.update(1)

    df = pd.DataFrame(rows)
    df["unet_mode"] = segmenter.cfg.unet_mode

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
        "validation_version": VALIDATION_VERSION,
        "n_images": int(df["sample_idx"].nunique()) if len(df) else 0,
        "n_rows": int(len(df)),
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


def run_fullres_unet_vs_imagej_validation(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    unet_mode: str,
    seg_method: str,
    gt_h5_name: str = "fullres_ground_truth.h5",
    imagej_h5_name: str = "fullres_imageJ.h5",
    force: bool = False,
) -> pd.DataFrame:
    """
    Run image-wide validation on simulated images.

    Compares:
      - UNet vs simulated ground truth
      - ImageJ/NCISP vs simulated ground truth

    Output:
      testing_val_imageJ_<unet_mode>_<seg_method>.csv
    """
    if seg_method not in ("conventional", "inst_seg"):
        raise ValueError(f"Unknown seg_method: {seg_method}")

    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"testing_val_imageJ_{unet_mode}_{seg_method}.csv",
    )
    out_summary_json = os.path.join(
        out_dir,
        f"testing_val_imageJ_{unet_mode}_{seg_method}_imagewide_summary.json",
    )

    if os.path.isfile(out_csv) and not force:
        return pd.read_csv(out_csv, index_col=None)

    gt_h5_path = os.path.join(h5_dir, gt_h5_name)
    imagej_h5_path = os.path.join(h5_dir, imagej_h5_name)

    model_file = f"best_{unet_mode}_tiles_S512_seed187.pth"

    segmenter, imagej_segmenter, cfg = _make_segmenter_and_validation_config(
        gt_h5_path=gt_h5_path,
        out_csv=out_csv,
        out_summary_json=out_summary_json,
        unet_mode=unet_mode,
        model_dir=model_dir,
        model_file=model_file,
        seg_method=seg_method,
    )

    df, _ = validate_unet_vs_imagej_on_fullres_h5(
        segmenter=segmenter,
        imagej_segmenter=imagej_segmenter,
        cfg=cfg,
        imagej_h5_path=imagej_h5_path,
        segmentation_method=seg_method,
    )

    gc.collect()
    return df

