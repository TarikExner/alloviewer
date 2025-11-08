from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from skimage.measure import label as sklabel

from ..segmentation.utils import collate_no_meta
from ..segmenter import SegmenterUNet, SegmenterConfig

import h5py
import torch

from tqdm import tqdm

from .utils import (
    iou_dice_overlap,
    boundary_f1_skeletonized,
    nms_peaks_np,           
    center_metrics_hungarian,
    energy_metrics_extended_full,
    get_dataset_mode
)
from .config import TrainingValidationConfig
import gc

class TiledH5Dataset(Dataset):
    """
    Read HDF5 made by your create_dataset_h5(...) OR by your
    ExternalCellsTilesDataset-exporter.

    Expect layout:
        /imgs : [N, T_max, 3, S, S]
        /tgts : [N, T_max, 4, S, S]
        /inst : [N, T_max, S, S]
        /meta : length N, JSON

    Real #tiles for sample i = len(meta["tiles"]).
    We return only those real tiles.

    __getitem__ -> (imgs_t, tgts_t, extras):
        imgs_t  : [T, 3, S, S]
        tgts_t  : [T, 4, S, S]
        extras  : {
            "instance_labels": [T, S, S],
            "meta": meta_json_dict
        }
    """

    def __init__(self, h5_path: str, indices: Optional[Sequence[int]] = None):
        super().__init__()
        self.h5_path = str(h5_path)
        self._h5 = None
        self._imgs = self._tgts = self._inst = self._meta = None

        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as f:
            N = int(f["imgs"].shape[0])
            self.N = N
            self.T_max = int(f["imgs"].shape[1])
            self.C_img = int(f["imgs"].shape[2])
            self.S = int(f["imgs"].shape[3])
            self.C_tgt = int(f["tgts"].shape[2])

        if indices is None:
            self.idx = np.arange(self.N, dtype=np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("indices must be 1D")
            if (idx < 0).any() or (idx >= self.N).any():
                raise ValueError("indices out of range")
            self.idx = idx

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._imgs = self._h5["imgs"]
            self._tgts = self._h5["tgts"]
            self._inst = self._h5["inst"]
            self._meta = self._h5["meta"]

    def __getitem__(self, i: int):
        self._ensure_open()
        k = int(self.idx[i])

        imgs = self._imgs[k]   # [T_max, 3, S, S]
        tgts = self._tgts[k]   # [T_max, 4, S, S]
        inst = self._inst[k]   # [T_max, S, S]
        mj = self._meta[k]
        if isinstance(mj, bytes):
            mj = mj.decode("utf-8")
        meta = json.loads(mj)

        tiles_meta = meta.get("tiles", [])
        if tiles_meta:
            T_real = len(tiles_meta)
        else:
            # fallback, but in your case we always store tiles -> this should not happen
            T_real = imgs.shape[0]

        imgs = imgs[:T_real]  # [T,3,S,S]
        tgts = tgts[:T_real]  # [T,4,S,S]
        inst = inst[:T_real]  # [T,S,S]

        imgs_t = torch.from_numpy(np.asarray(imgs, dtype=np.float32))
        tgts_t = torch.from_numpy(np.asarray(tgts, dtype=np.float32))
        inst_t = torch.from_numpy(np.asarray(inst, dtype=np.int32))

        extras = {
            "instance_labels": inst_t,
            "meta": meta,
        }
        return imgs_t, tgts_t, extras

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None


def validate_unet_on_tiled_h5(
    segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,                    # your validation cfg (has cell_thr, boundary_thr, etc.)
    indices: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the UNet on every sample in a tiled H5 (sim or external),
    using meta["tiles"] to know how many tiles to keep.

    For each tile we compute the same metrics you already use.
    """
    ds = TiledH5Dataset(cfg.h5_path, indices=indices)
    dl = DataLoader(
        ds,
        batch_size=1,          # 1 sample -> many tiles
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta
    )

    rows: List[Dict[str, Any]] = []

    with tqdm(total=len(ds), desc="validate tiled h5", unit="img") as pbar:
        for sample_idx, batch in enumerate(dl):
            imgs_t, tgts_t, extras = batch
            # remove batch dim
            imgs_t = imgs_t[0]                    # [T,3,S,S]
            tgts_t = tgts_t[0]                    # [T,4,S,S]
            inst_t = extras["instance_labels"][0] # [T,S,S]
            meta = extras["meta"][0]

            imgs_np = imgs_t.numpy().astype(np.float32)
            tgts_np = tgts_t.numpy().astype(np.float32)
            inst_np = inst_t.numpy().astype(np.int32)
            T, _, H, W = imgs_np.shape

            # run all tiles in one go
            out = segmenter(imgs_np)
            cell_prob = out["probs"]["cell"]    # [T,H,W]
            bound_prob = out["probs"]["bound"]
            center_pred = out["probs"]["center"]
            energy_pred = out["probs"]["energy"]

            # figure out what kind of source this is
            # (sim: meta["full"] has sim fields; external: meta["full"] has src_path)
            full_meta = meta.get("full", {})
            # "n_cells" may or may not be there; we'll fallback to GT when missing
            n_cells_from_meta = full_meta.get("n_cells", None)
            # NEW: pull only the needed bits from meta["full"]
            frac_positive_from_meta = full_meta.get("frac_positive", None)
            src_path = full_meta.get("src_path", None)

            params = full_meta.get("params", {})
            # normalize params into flat columns: floats stay numeric; small lists -> JSON string
            params_out = {}
            for k, v in params.items():
                col = f"param_{k}"
                if isinstance(v, (int, float, np.floating)):
                    params_out[col] = float(v)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    # keep lists compact and consistent in CSV
                    params_out[col] = json.dumps([float(x) for x in list(v)])
                else:
                    # ignore unexpected types
                    continue

            for t in range(T):
                tgt = tgts_np[t]           # [4,H,W]
                cell_gt = (tgt[0] > 0.5).astype(np.uint8)
                energy_gt = tgt[3].astype(np.float32)
                inst_gt = inst_np[t]

                cell_p = cell_prob[t]
                bound_p = bound_prob[t]
                center_p = center_pred[t]
                energy_p = energy_pred[t]

                # components
                cell_pred_bin = (cell_p >= cfg.cell_thr).astype(np.uint8)
                n_cc = int(sklabel(cell_pred_bin, connectivity=1).max())
                n_gt = int(inst_gt.max())

                # centers (count)
                peaks = nms_peaks_np(
                    center_p,
                    thr=cfg.center_peak_thr,
                    min_dist=cfg.center_nms_dist,
                )
                n_centers_pred = int(len(peaks))

                # mask metrics
                mask_stats = iou_dice_overlap(cell_pred_bin, cell_gt)

                # boundary
                boundary_f1 = boundary_f1_skeletonized(
                    bound_p,
                    inst_gt,
                    tol=cfg.boundary_tol,
                    thr=cfg.boundary_thr,
                    sweep=cfg.boundary_sweep,
                )

                # centers (Hungarian + AP)
                center_stats = center_metrics_hungarian(
                    center_p,
                    inst_gt,
                    peak_thr=cfg.center_peak_thr,
                    nms_dist=cfg.center_nms_dist,
                    match_radius=cfg.center_match_radius,
                    ap_thr_list=cfg.ap_thr_list,
                    oks_thresholds=cfg.oks_thresholds,
                )

                # energy
                energy_stats = energy_metrics_extended_full(
                    energy_p,
                    energy_gt,
                    cell_gt,
                    frac_delta=cfg.energy_frac_delta,
                )

                # pick n_cells: meta wins, else GT
                if n_cells_from_meta is not None:
                    n_cells = int(n_cells_from_meta)
                else:
                    n_cells = n_gt

                row: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "tile_idx": int(t),
                    "n_cells": n_cells,
                    "frac_positive": (
                        float(frac_positive_from_meta)
                        if frac_positive_from_meta is not None else np.nan
                    ),
                    "n_cells_gt_instances": n_gt,
                    "n_cells_pred_components_thr0p5": n_cc,
                    "n_cells_pred_centers": n_centers_pred,
                    "count_error_components": int(n_cc - n_gt),
                    "count_error_centers": int(n_centers_pred - n_gt),
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                    "boundary_f1": float(boundary_f1),
                    **{f"center_{k}": v for k, v in center_stats.items()},
                    **{f"energy_{k}": v for k, v in energy_stats.items()},
                    "src_path": src_path if src_path is not None else "",
                    **params_out
                }

                rows.append(row)

            pbar.update(1)

    df = pd.DataFrame(rows)
    df["unet_mode"] = segmenter.cfg.unet_mode
    df["dataset_mode"] = get_dataset_mode(cfg.h5_path)

    metric_cols = [
        c for c in df.columns
        if any(c.startswith(pfx) for pfx in ("mask_", "boundary_", "center_", "energy_", "count_error_"))
    ]
    summary = {
        "n_images": int(len(df)),
        "means": {c: float(np.nanmean(df[c].values.astype(np.float64))) for c in metric_cols},
        "stds":  {c: float(np.nanstd(df[c].values.astype(np.float64)))  for c in metric_cols},
    }

    if cfg.out_csv:
        os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
        df.to_csv(cfg.out_csv, index=False)

    if cfg.out_summary_json:
        os.makedirs(os.path.dirname(cfg.out_summary_json) or ".", exist_ok=True)
        with open(cfg.out_summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    return df, summary


def run_test_ext_validation(out_dir: str,
                            model_dir: str,
                            h5_dir: str) -> None:
    res = []
    for unet_mode in ["large", "medium", "small"]:
        for dataset_mode in ["external_images", "tiles"]:
            print(f"... Starting calculations for UNet {unet_mode} and dataset {dataset_mode}")
            out_csv = os.path.join(out_dir, f"testing_val_{unet_mode}_{dataset_mode}.csv")

            if os.path.isfile(out_csv):
                df = pd.read_csv(out_csv, index_col = None)
                res.append(df)
                continue

            cfg = TrainingValidationConfig(
                h5_path = os.path.join(h5_dir, f"{dataset_mode}_test.h5"),
                cell_thr = 0.1,
                out_csv = out_csv
            )
            segmenter_cfg = SegmenterConfig(
                unet_mode = unet_mode,
                model_dir = model_dir,
                model_file = f"best_{unet_mode}_tiles_S512_seed187.pth",
                device = "cuda" if torch.cuda.is_available() else "cpu",
                use_amp = torch.cuda.is_available()
            ).to_dict()
            segmenter = SegmenterUNet.from_config(segmenter_cfg)
            df, _ = validate_unet_on_tiled_h5(segmenter, cfg)
            res.append(df)
            gc.collect()

    final = pd.concat(res, axis = 0)
    final.to_csv(os.path.join(out_dir, "testing_val_combined.csv"), index = False)

    return
