from __future__ import annotations
import os
import json
import glob
from typing import Optional, Sequence, Tuple, List, Dict, Any

import h5py
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.measure import label as sklabel

from ..segmentation.utils import collate_no_meta

# pull helpers from utils.py (see adjustments below)
from .utils import (
    resize_map,                     # cv2-based (image/binary/label)
    pad_to_square,                  # cv2/np-based
    crop_rect,
    estimate_well_mask,
    square_crop_from_center_radius,
    iou_dice_overlap,
    boundary_f1_skeletonized,
    center_metrics_hungarian,
    nms_peaks_np
)

def _find_pairs_strict(root_dir: str) -> List[Tuple[str, str]]:
    """
    Pair rule: '<name>.tif[f]' with sibling '<name>_mask.tif[f]'.
    Skips files whose stem already ends with '_mask'.
    """
    exts = (".tif", ".tiff")
    files = [p for ext in exts for p in glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True)]
    pairs: List[Tuple[str, str]] = []
    for img_path in files:
        base = os.path.basename(img_path)
        stem, ext = os.path.splitext(base)
        if stem.lower().endswith("_mask"):
            continue
        d = os.path.dirname(img_path)
        m1 = os.path.join(d, f"{stem}_mask.tif")
        m2 = os.path.join(d, f"{stem}_mask.tiff")
        mask_path = m1 if os.path.exists(m1) else (m2 if os.path.exists(m2) else None)
        if mask_path is not None:
            pairs.append((os.path.abspath(img_path), os.path.abspath(mask_path)))
    pairs.sort()
    return pairs


def build_h5_from_folder(
    root_dir: str,
    out_path: str,
    target: int = 512,
    compression: Optional[str] = "lzf",
    chunk_size: int = 128,
    flush_every: int = 16,
    mode: str = "pad_resize",                 # "pad_resize" | "crop_well_resize" | "tiles"
    tiles_per_image: int = 4,                 # only for mode="tiles"
    rng_seed: int = 12345,                    # reproducible tiling
) -> str:
    """
    Create HDF5 with:
      /imgs: float32 [N, 3, S, S]  in [0,1]
      /tgts: float32 [N, 1, S, S]  binary {0,1}
      /inst: int32   [N, S, S]     connected components from mask
      /meta: vlen JSON              per-sample metadata (includes mode & geometry)
    Notes:
      - images are read as 16-bit TIFF (scaled to [0,1]); masks are 8-bit (255 = cell).
      - supports same modes as sim dataset.
      - in tiles mode, we generate 'tiles_per_image' random tiles per image with a fixed RNG seed.
    """
    assert mode in ("pad_resize", "crop_well_resize", "tiles"), f"unknown mode: {mode}"
    pairs = _find_pairs_strict(root_dir)
    if not pairs:
        raise RuntimeError("no (image, image_mask) pairs found")

    # probe to compute N depending on mode
    if mode == "tiles":
        N = len(pairs) * int(tiles_per_image)
    else:
        N = len(pairs)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    vlen_str = h5py.special_dtype(vlen=str)
    with h5py.File(out_path, "w", libver="latest") as f:
        f.attrs.update({
            "version": 1,
            "length": int(N),
            "target": int(target),
            "source": os.path.abspath(root_dir),
            "C_img": 3,
            "C_tgt": 1,
            "mode": mode,
            "tiles_per_image": int(tiles_per_image),
        })

        d_imgs = f.create_dataset(
            "imgs", shape=(N, 3, target, target), dtype=np.float32,
            chunks=(min(chunk_size, N), 3, target, target), compression=compression
        )
        d_tgts = f.create_dataset(
            "tgts", shape=(N, 1, target, target), dtype=np.float32,
            chunks=(min(chunk_size, N), 1, target, target), compression=compression
        )
        d_inst = f.create_dataset(
            "inst", shape=(N, target, target), dtype=np.int32,
            chunks=(min(chunk_size, N), target, target), compression=compression
        )
        d_meta = f.create_dataset(
            "meta", shape=(N,), dtype=vlen_str, chunks=(min(1024, N),)
        )

        pbar = tqdm(total=N, desc="build_h5", dynamic_ncols=True)
        written = 0
        rng = np.random.default_rng(int(rng_seed))

        for img_path, mask_path in pairs:
            # --- read image (normalize to [0,1]) ---
            img = tiff.imread(img_path).astype(np.float32, copy=False)
            if img.ndim == 2:
                pass
            elif img.ndim == 3:
                if img.shape[2] == 1:
                    img = img[..., 0]
                elif img.shape[2] > 3:
                    img = img[..., :3]
            # scale 8/16-bit -> [0,1]
            if img.dtype.kind in ("u", "i"):
                vmax = 65535.0 if img.max() > 255 else 255.0
                img = img / max(vmax, 1.0)
            else:
                img = np.clip(img, 0.0, 1.0)

            # ensure 3 channels
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 2:
                # rare case: pad 2->3
                img = np.concatenate([img, img[..., :1]], axis=-1)

            # --- read mask (255 = cell) ---
            msk = tiff.imread(mask_path)
            if msk.ndim > 2:
                msk = msk[..., 0]
            msk = (msk >= 255).astype(np.uint8)

            H, W = msk.shape

            # --------- mode transforms ----------
            if mode == "pad_resize":
                img_sq, (pt, pl), S = pad_to_square(img, pad_val=0.0)
                msk_sq, _, _ = pad_to_square(msk, pad_val=0)
                img_o = resize_map(img_sq, target, "image")
                mask_o = resize_map(msk_sq, target, "binary")
                inst_o = resize_map(msk_sq, target, "label")
                meta = dict(
                    mode="pad_resize",
                    pad_top=int(pt), pad_left=int(pl),
                    S_in=int(S),
                    scale=float(target / float(S)),
                    src_path=img_path,
                    mask_path=mask_path,
                    H_in=int(H), W_in=int(W),
                )

            elif mode == "crop_well_resize":
                # detect/estimate well on the image
                well_mask, (cy, cx), R = estimate_well_mask(img, blur_sigma=3.0, well_is_brighter="auto")
                y0, y1, x0, x1 = square_crop_from_center_radius(msk.shape, (cy, cx), R, pad=8)

                img_c = crop_rect(img, y0, x0, y1 - y0, x1 - x0)
                msk_c = crop_rect(msk, y0, x0, y1 - y0, x1 - x0)

                img_o = resize_map(img_c, target, "image")
                mask_o = resize_map(msk_c, target, "binary")
                inst_o = resize_map(msk_c, target, "label")

                scale = float(target / max(1.0, float(max(y1 - y0, x1 - x0))))
                meta = dict(
                    mode="crop_well_resize",
                    crop=(int(y0), int(y1), int(x0), int(x1)),
                    scale=scale,
                    well_center=(float(cy), float(cx)),
                    well_radius=float(R),
                    src_path=img_path,
                    mask_path=mask_path,
                    H_in=int(H), W_in=int(W),
                )

            else:  # "tiles"
                # Draw tiles_per_image random tiles of size "target"
                # If image smaller than target, fallback to pad+resize like above for this sample.
                # We repeat per drawn tile.
                for _ in range(int(tiles_per_image)):
                    if H <= target or W <= target:
                        img_sq, (pt, pl), S = pad_to_square(img, pad_val=0.0)
                        msk_sq, _, _ = pad_to_square(msk, pad_val=0)
                        img_o = resize_map(img_sq, target, "image")
                        mask_o = resize_map(msk_sq, target, "binary")
                        inst_o = resize_map(msk_sq, target, "label")
                        meta = dict(
                            mode="tiles",
                            fallback="pad_resize",
                            tile_xy=(0, 0),
                            tile_hw=(target, target),
                            src_path=img_path,
                            mask_path=mask_path,
                            H_in=int(H), W_in=int(W),
                        )
                    else:
                        y0 = int(rng.integers(0, H - target + 1))
                        x0 = int(rng.integers(0, W - target + 1))
                        img_t = crop_rect(img, y0, x0, target, target)
                        msk_t = crop_rect(msk, y0, x0, target, target)

                        # already target size; still run through resize_map for consistency (nearest/bilinear no-op)
                        img_o = resize_map(img_t, target, "image")
                        mask_o = resize_map(msk_t, target, "binary")
                        inst_o = resize_map(msk_t, target, "label")

                        meta = dict(
                            mode="tiles",
                            tile_xy=(int(y0), int(x0)),
                            tile_hw=(int(target), int(target)),
                            src_path=img_path,
                            mask_path=mask_path,
                            H_in=int(H), W_in=int(W),
                        )

                    # write this tile
                    inst_lbl = sklabel((inst_o > 0).astype(np.uint8), connectivity=1).astype(np.int32)
                    d_imgs[written] = np.transpose(img_o.astype(np.float32), (2, 0, 1))
                    d_tgts[written, 0] = (mask_o > 0).astype(np.float32)
                    d_inst[written] = inst_lbl
                    d_meta[written] = json.dumps(meta, separators=(",", ":"))
                    written += 1
                    if written % max(1, flush_every) == 0:
                        f.flush()
                        try:
                            f.id.flush()
                        except Exception:
                            pass
                    pbar.update(1)

                # done with tiles for this image
                continue

            # write single-sample (pad/crop modes)
            inst_lbl = sklabel((inst_o > 0).astype(np.uint8), connectivity=1).astype(np.int32)
            d_imgs[written] = np.transpose(img_o.astype(np.float32), (2, 0, 1))
            d_tgts[written, 0] = (mask_o > 0).astype(np.float32)
            d_inst[written] = inst_lbl
            d_meta[written] = json.dumps(meta, separators=(",", ":"))
            written += 1
            if written % max(1, flush_every) == 0:
                f.flush()
                try:
                    f.id.flush()
                except Exception:
                    pass
            pbar.update(1)

        pbar.close()

    return out_path



# -----------------------------------------------------------------------------
# PART 2 — Reader + validation (skip energy & center heatmap)
# -----------------------------------------------------------------------------

class ExternalMasksH5(Dataset):
    """
    H5 reader for files made by build_h5_from_folder.
    Returns:
      img_t:  torch.float32 [3,S,S]
      tgt_t:  torch.float32 [1,S,S]   (binary cell mask)
      extras: {"instance_labels": torch.int32 [S,S]}
    """
    def __init__(self, h5_path: str, indices: Optional[Sequence[int]] = None):
        super().__init__()
        self.h5_path = str(h5_path)
        self._h5 = None

        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as f:
            self._N = int(f["imgs"].shape[0])
            self._S = int(f["imgs"].shape[2])

        if indices is None:
            self._idx = np.arange(self._N, dtype=np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)
            if (idx < 0).any() or (idx >= self._N).any():
                raise ValueError("indices out of range")
            self._idx = idx

    def __len__(self) -> int:
        return int(self._idx.shape[0])

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._imgs = self._h5["imgs"]
            self._tgts = self._h5["tgts"]
            self._inst = self._h5["inst"]

    def __getitem__(self, i: int):
        self._ensure_open()
        k = int(self._idx[i])
        img = self._imgs[k]      # [3,S,S] float32
        tgt = self._tgts[k]      # [1,S,S] float32
        inst = self._inst[k]     # [S,S]   int32

        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        tgt_t = torch.from_numpy(np.asarray(tgt, dtype=np.float32))
        inst_t = torch.from_numpy(np.asarray(inst, dtype=np.int32))
        return img_t, tgt_t, {"instance_labels": inst_t}

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __del__(self):
        self.close()


def validate_unet_on_external_h5(
    segmenter,                 # your SegmenterUNet
    h5_path: str,
    out_csv: Optional[str] = None,
    out_summary_json: Optional[str] = None,
    batch_size: int = 8,
    workers: int = 4,
    cell_thr: float = 0.5,
    boundary_thr: float = 0.9,
    boundary_tol: int = 2,
    boundary_sweep: bool = False,

    include_center_metrics: bool = False,
    center_peak_thr: float = 0.2,
    center_nms_dist: int = 3,
    center_match_radius: int = 10,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """
    Runs validation on external H5. By default, only mask metrics + boundary F1 are computed.
    If include_center_metrics=True, also computes Hungarian/AP/OKS on the center head.
    """

    ds = ExternalMasksH5(h5_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True, drop_last=False,
                    collate_fn=collate_no_meta)

    rows: List[Dict[str, Any]] = []

    for imgs_t, tgts_t, extras in dl:
        B = imgs_t.shape[0]
        for b in range(B):
            img_chw = imgs_t[b].numpy()
            img_hwc = np.transpose(img_chw, (1, 2, 0))  # [H,W,3]

            gt_mask = (tgts_t[b, 0].numpy() > 0.5).astype(np.uint8)
            inst_gt = extras["instance_labels"][b].numpy().astype(np.int32)

            # run model
            out = segmenter(img_hwc)
            cell_prob = out["probs"]["cell"]
            bound_prob = out["probs"]["bound"]
            center_pred = out["probs"]["center"]

            # binarize predicted cell mask for components
            cell_pred_bin = (cell_prob >= cell_thr).astype(np.uint8)
            n_cc_pred = int(sklabel(cell_pred_bin, connectivity=1).max())
            n_gt = int(inst_gt.max())

            # mask metrics
            mstats = iou_dice_overlap(cell_pred_bin, gt_mask)

            # boundary
            b_f1 = boundary_f1_skeletonized(bound_prob, inst_gt,
                                            tol=boundary_tol, thr=boundary_thr, sweep=boundary_sweep)

            row: Dict[str, Any] = {
                "idx": int(len(rows)),
                "n_cells_gt_instances": n_gt,
                "n_cells_pred_components_thr0p5": n_cc_pred,
                "count_error_components": int(n_cc_pred - n_gt),
                "mask_iou": float(mstats["iou"]),
                "mask_dice": float(mstats["dice"]),
                "mask_overlap": float(mstats["overlap"]),
                "boundary_f1": float(b_f1),
            }

            if include_center_metrics:
                ctr_stats = center_metrics_hungarian(
                    center_pred,
                    inst_gt,
                    peak_thr=center_peak_thr,
                    nms_dist=center_nms_dist,
                    match_radius=center_match_radius,
                )
                # also record predicted center count
                peaks = nms_peaks_np(center_pred, thr=center_peak_thr, min_dist=center_nms_dist)
                row["n_cells_pred_centers"] = int(len(peaks))
                row.update({f"center_{k}": v for k, v in ctr_stats.items()})

            rows.append(row)

    df = pd.DataFrame(rows)

    # summary
    metric_cols = [c for c in df.columns if c not in ("idx",)]
    summary = {
        "n_images": int(len(df)),
        "means": {c: float(np.nanmean(df[c].values.astype(np.float64))) for c in metric_cols},
        "stds":  {c: float(np.nanstd(df[c].values.astype(np.float64))) for c in metric_cols},
    }

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
    if out_summary_json:
        os.makedirs(os.path.dirname(out_summary_json) or ".", exist_ok=True)
        import json
        with open(out_summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    return df, summary

