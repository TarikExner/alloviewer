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
from PIL import Image
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

def instances_to_binary_uint8(mask: ArrayLike) -> np.ndarray:
    """
    Take an instance label map (0 = background, >0 = instance ID)
    and convert it to a binary mask {0,1} with dtype uint8.
    """
    m = to_numpy(mask)

    # squeeze possible extra dims (e.g. [1,H,W] or [H,W,1])
    while m.ndim > 2:
        m = np.squeeze(m, axis=0) if m.shape[0] == 1 else np.squeeze(m, axis=-1)
        if m.ndim <= 2:
            break

    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {m.shape}")

    m_bin = (m != 0).astype(np.uint8)
    return m_bin

def to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def format_img_number(n: int) -> str:
    """Convert image index to zero-padded 5-digit string."""
    return f"{n:05d}"

def format_tile_number(n: int) -> str:
    """Convert tile index to zero-padded 2-digit string."""
    return f"{n:02d}"

def save_tile_triplet_as_jpg(
    out_dir: str,
    img_no: int,
    tile_no: int,
    img: ArrayLike,
    imgj_mask: ArrayLike,
    unet_mask: ArrayLike,
) -> None:
    """
    Save original tile + ImageJ GT mask + U-Net mask as .jpg files.

    Files written:
        {img_no}_{tile_no}.jpg
        {img_no}_{tile_no}_imageJ.jpg
        {img_no}_{tile_no}_unet.jpg
    """

    os.makedirs(out_dir, exist_ok=True)

    # ---- image ----
    img_np = denormalize_image(img)
    img_np = to_numpy(img_np)

    # ensure [H,W,C] or [H,W]
    if img_np.ndim == 3:  # [C,H,W] -> [H,W,C]
        img_np = img_np.transpose(1, 2, 0)
    elif img_np.ndim != 2:
        raise ValueError(f"Unexpected image shape after denorm: {img_np.shape}")

    # scale to uint8 for JPEG
    img_min = img_np.min()
    img_max = img_np.max()
    if img_max > img_min:
        img_np = (img_np - img_min) / (img_max - img_min)
    img_uint8 = (img_np * 255).astype(np.uint8)

    # convert to PIL image
    if img_uint8.ndim == 2:
        img_pil = Image.fromarray(img_uint8, mode="L")
    elif img_uint8.shape[2] == 3:
        img_pil = Image.fromarray(img_uint8, mode="RGB")
    else:
        # fallback: use first channel as grayscale
        img_pil = Image.fromarray(img_uint8[..., 0], mode="L")

    # ---- masks (binary, uint8) ----
    imgj_bin = instances_to_binary_uint8(imgj_mask)
    unet_bin = instances_to_binary_uint8(unet_mask)

    # scale binary masks to 0/255 for viewing
    imgj_pil = Image.fromarray(imgj_bin * 255, mode="L")
    unet_pil = Image.fromarray(unet_bin * 255, mode="L")

    # ---- filenames ----
    img_str = format_img_number(img_no)
    tile_str = format_tile_number(tile_no)
    base = f"{img_str}_{tile_str}"

    img_path = os.path.join(out_dir, f"{base}.jpg")
    imgj_path = os.path.join(out_dir, f"{base}_imageJ.jpg")
    unet_path = os.path.join(out_dir, f"{base}_unet.jpg")

    # ---- write ----
    img_pil.save(img_path, format="JPEG", quality=80)
    imgj_pil.save(imgj_path, format="JPEG", quality=80)
    unet_pil.save(unet_path, format="JPEG", quality=80)

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
                if img.ndim == 4:
                    img = img[0]
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

                save_tile_triplet_as_jpg(
                    out_dir=out_dir,
                    img_no=sample_idx,
                    tile_no=t,
                    img=img,
                    imgj_mask=inst_gt,
                    unet_mask=inst_pred
                )


            pbar.update(1)

    return

def export_segmentation_comparison(
    out_dir: str,
    model_dir: str,
    h5_dir: str,
    stop: Optional[int] = None
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



