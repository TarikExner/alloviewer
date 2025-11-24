from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..segmentation.utils import collate_no_meta
from ..segmenter import SegmenterUNet, SegmenterConfig, InstanceSegmenterConfig

import torch

from tqdm import tqdm

from ..segmentation import TiledH5Dataset, UNET_MEAN, UNET_STD
from .config import TrainingValidationConfig

from typing import Union
import tifffile as tiff

ArrayLike = Union[np.ndarray, torch.Tensor]

def denormalize_image(
    img: ArrayLike,
) -> ArrayLike:
    """
    Reverse (img - mean) / std normalization.

    img:  torch.Tensor or np.ndarray, shape [C,H,W] or [T,C,H,W]
    mean: scalar or sequence of length C
    std:  scalar or sequence of length C
    """

    mean = UNET_MEAN
    std = UNET_STD

    is_tensor = isinstance(img, torch.Tensor)

    if is_tensor:
        x = img.clone()
        device = x.device
        dtype = x.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype, device=device)
        std_t = torch.as_tensor(std, dtype=dtype, device=device)
    else:
        x = np.asarray(img)
        mean_t = np.asarray(mean, dtype=x.dtype)
        std_t = np.asarray(std, dtype=x.dtype)

    # Make mean/std broadcast over [C,H,W] or [T,C,H,W]
    # We want shape [C,1,1] or [1,C,1,1] depending on ndim
    if x.ndim == 3:       # [C,H,W]
        shape = (-1, 1, 1)
    elif x.ndim == 4:     # [T,C,H,W]
        shape = (1, -1, 1, 1)
    else:
        raise ValueError(f"Expected img with 3 or 4 dims, got shape {x.shape}")

    if mean_t.ndim == 0:
        # scalar, fine
        pass
    else:
        mean_t = mean_t.reshape(shape)
        std_t = std_t.reshape(shape)

    x = x * std_t + mean_t

    if is_tensor:
        return x
    else:
        return x

def save_tile_triplet_as_tif(
    out_dir: str,
    img_no: int,
    tile_no: int,
    img: ArrayLike,
    imgj_mask: ArrayLike,
    unet_mask: ArrayLike,
) -> None:
    """
    Save original tile + ImageJ GT mask + U-Net mask as .tif files.

    File names (in `out_dir`):
        {tile_number:06d}.tif
        {tile_number:06d}_imageJ.tif
        {tile_number:06d}_unet.tif

    Args
    ----
    out_dir:     Folder where the .tif files will be written.
    tile_number: Integer used for the base filename (ascending index).
    img:         Original image tile.  Can be numpy array or torch tensor.
    imgj_mask:   ImageJ ground truth mask.  Same HxW (or CxHxW) shape.
    unet_mask:   U-Net prediction mask.   Same HxW (or CxHxW) shape.
    """

    os.makedirs(out_dir, exist_ok=True)

    def to_numpy(x: ArrayLike) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    img_np = to_numpy(img)
    imgj_np = to_numpy(imgj_mask)
    unet_np = to_numpy(unet_mask)

    base = f"{img_no}_{tile_no}"

    img_path = os.path.join(out_dir, f"{base}.tif")
    imgj_path = os.path.join(out_dir, f"{base}_imageJ.tif")
    unet_path = os.path.join(out_dir, f"{base}_unet.tif")

    # write as float32 (change dtype here if you prefer uint16, etc.)
    tiff.imwrite(img_path, img_np.astype(np.float32))
    tiff.imwrite(imgj_path, imgj_np.astype(np.float32))
    tiff.imwrite(unet_path, unet_np.astype(np.float32))

def export_images(
    out_dir: str,
    segmenter: SegmenterUNet,
    gt_segmenter: SegmenterUNet,
    cfg: TrainingValidationConfig,
    indices: Optional[Sequence[int]] = None,
    stop: Optional[int] = None,
) -> None:
    """
    Run the UNet on every sample in a tiled H5 (sim or external),
    using meta["tiles"] to know how many tiles to keep.

    For each tile we compute the same metrics you already use.

    segmentation_method:
        - "conventional": use simple thresholding on P(cell)
        - "inst_seg":     use the InstanceSegmenter output (instances > 0)
    """

    ds = TiledH5Dataset(cfg.h5_path, indices=indices)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_no_meta
    )

    with tqdm(total=len(ds), desc="validate tiled h5", unit="img") as pbar:
        for sample_idx, batch in enumerate(dl):
            if sample_idx == stop:
                break
            imgs_t, tgts_t, extras = batch
            # remove batch dim
            imgs_t = imgs_t[0]                    # [T,3,S,S]
            tgts_t = tgts_t[0]                    # [T,4,S,S]

            imgs_np = imgs_t.numpy().astype(np.float32)
            tgts_np = tgts_t.numpy().astype(np.float32)
            T, _, H, W = imgs_np.shape

            # run all tiles in one go
            out = segmenter(imgs_np)

            inst_pred_list = out.get("instance_labels", None)
            if inst_pred_list is None:
                raise RuntimeError(
                    "segmentation_method='inst_seg' but segmenter did not return instance_labels"
                )

            for t in range(T):
                img = imgs_np[t]

                tgt = tgts_np[t]

                gt_inst_seg_dict = {
                    "probs": {
                        "cell":   (tgt[0]>gt_segmenter.cfg.thr_cell).astype(np.uint8),
                        "bound":  (tgt[1]>gt_segmenter.cfg.thr_bound).astype(np.uint8),
                        "center": (tgt[2]),
                        "energy": (tgt[3]),
                    },
                    "cell_mask": (tgt[0]>= segmenter.cfg.thr_cell).astype(np.uint8),
                    "boundary":  (tgt[1]>= segmenter.cfg.thr_bound).astype(np.uint8),
                    "instance_labels": None,
                    "meta": {},

                }
                inst_seg_dict = gt_segmenter.inst_seg(gt_inst_seg_dict, update_cell_mask = True)

                inst_gt = inst_seg_dict["instance_labels"].astype(np.int32)

                inst_pred = np.asarray(inst_pred_list[t], dtype=np.int32)

                save_tile_triplet_as_tif(
                    out_dir=out_dir,
                    img_no=sample_idx,
                    tile_no=t,
                    img=denormalize_image(img),
                    imgj_mask=inst_gt,
                    unet_mask=inst_pred
                )


            pbar.update(1)

    return

def export_segmentation_comparison(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    stop: Optional[int]
) -> None:
    """
    Run validation for a single combination of:
        - unet_mode       in {"large","medium","small", ...}
        - dataset_mode    "external_images" or "tiles"
        - seg_method      in {"conventional", "inst_seg"}

    Writes a CSV for this combo and returns the DataFrame.
    """

    os.makedirs(out_dir, exist_ok=True)

    # cache: if CSV already exists, just read and return
    h5_path = os.path.join(h5_dir, "external_images_test.h5")

    cfg = TrainingValidationConfig(
        h5_path=h5_path,
        cell_thr=0.1,
    )

    seg_params: Dict[str, Any] = dict(
        unet_mode="small",
        model_dir=model_dir,
        model_file="best_small_tiles_S512_seed187.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=torch.cuda.is_available(),
        normalize=False #DiskSimCellsDataset already normalizes
    )

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

    export_images(
        out_dir,
        segmenter,
        gt_segmenter,
        cfg,
        stop = stop
    )

    return



