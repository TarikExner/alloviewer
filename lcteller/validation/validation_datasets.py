from __future__ import annotations
from typing import List, Optional, Tuple
import os
import json
import glob
import signal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import tifffile as tiff
from skimage import morphology
from skimage.measure import label as sklabel
from skimage.segmentation import relabel_sequential

from ..segmentation.image_dataset import (
    _make_soft_boundary_from_instances,
    _make_center_stem_from_centers,
    _make_center_heatmap,
    _make_energy_from_instances,
)
from .utils import jsonify


def _find_pairs_strict(root_dir: str) -> List[Tuple[str, str]]:
    exts = (".tif", ".tiff")
    files = [
        p
        for ext in exts
        for p in glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True)
    ]
    pairs: List[Tuple[str, str]] = []
    for img_path in files:
        base = os.path.basename(img_path)
        stem, _ = os.path.splitext(base)
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

def _heal_watershed_gaps(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Fix 1px (or very thin) background lines produced by ImageJ watershed.

    Steps:
      1. binarize (in case it's 8-bit 0/255 or 0/1/2/...),
      2. binary closing with a small disk → fills cuts *inside* the mask,
      3. AND with a dilated version of the original to avoid growing too far out.
    """
    mask_bin = (mask > 0)

    if radius <= 0:
        return mask_bin.astype(np.uint8)

    selem = morphology.disk(int(radius))

    # fills the splits
    closed = morphology.binary_closing(mask_bin, selem)

    # limit growth — stay within original mask + radius
    grown = morphology.binary_dilation(mask_bin, selem)

    healed = np.logical_and(closed, grown)
    return healed.astype(np.uint8)

class ExternalCellsTilesDataset(Dataset):
    """
    External images + 8-bit masks → same output shape as SimCellsDataset( mode="tiles", n_tiles=-1 )

    root_dir must contain pairs (img, mask) that _find_pairs_strict(root_dir) can detect.

    Returns per __getitem__:
        imgs_t   : float32 [T, 3, target, target]
        tgts_t   : float32 [T, 4, target, target]
        extras   : {
            "instance_labels": int32 [T, target, target],
            "meta": {
                "full": {
                    "src_path": ...,
                    "mask_path": ...,
                    "H_in": ...,
                    "W_in": ...,
                    "n_cells": ...,
                },
                "tiles": [ tile_meta_0, ..., tile_meta_{T-1} ]
            }
        }
    """
    def __init__(
        self,
        root_dir: str,
        target: int = 512,
        tile_overlap: int = 64,
        heal_radius: int = 1,
        boundary_ring_width: int = 1,
        boundary_soft_band: int = 2,
        boundary_sigma: float = 1.0,
        center_sigma: float = 1.0,
        transforms=None,   # optional Albumentations-style joint transform
    ):
        assert os.path.isdir(root_dir), f"not a dir: {root_dir}"
        self.root_dir = root_dir
        self.target = int(target)
        self.tile_overlap = int(tile_overlap)
        self.heal_radius = int(heal_radius)
        self.boundary_ring_width = int(boundary_ring_width)
        self.boundary_soft_band = int(boundary_soft_band)
        self.boundary_sigma = float(boundary_sigma)
        self.center_sigma = float(center_sigma)
        self.transforms = transforms

        # reuse your finder
        self.pairs = _find_pairs_strict(root_dir)
        if not self.pairs:
            raise RuntimeError("no (img, img_mask) pairs found in external folder")

    def __len__(self):
        return len(self.pairs)

    # --- small helpers -------------------------------------------------

    def _read_image_and_mask(self, img_path: str, mask_path: str):
        # image
        img = tiff.imread(img_path).astype(np.float32, copy=False)
        # squeeze / pick channels
        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            # if single-channel in last axis
            if img.shape[2] == 1:
                img = img[..., 0]
            elif img.shape[2] > 3:
                img = img[..., :3]

        # to [0,1]
        if img.dtype.kind in ("u", "i"):
            vmax = 65535.0 if img.max() > 255 else 255.0
            img = img / max(vmax, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)

        # 3ch
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 2:
            # 2 -> 3
            img = np.concatenate([img, img[..., :1]], axis=-1)

        # mask
        msk = tiff.imread(mask_path)
        if msk.ndim > 2:
            msk = msk[..., 0]
        # your note: "we have the originals and an 8 bit mask"
        # -> usually 255 = cell
        msk = (msk >= 255).astype(np.uint8)

        return img, msk

    def _enumerate_full_tiles(self, H: int, W: int):
        """like SimCellsDataset._enumerate_full_tiles but for fixed H,W"""
        th = self.target
        if H <= th or W <= th:
            # single tile that we will pad below in __getitem__
            return [(0, min(th, H), 0, min(th, W))]
        stride = max(1, th - self.tile_overlap)
        coords = []
        for y0 in range(0, H - th + 1, stride):
            for x0 in range(0, W - th + 1, stride):
                y1 = y0 + th
                x1 = x0 + th
                coords.append((y0, y1, x0, x1))
        # right / bottom edges might still be missing if (H - th) % stride != 0
        # but this is exactly how your earlier external tiler worked, so we keep it
        return coords

    # --- main -----------------------------------------------------------

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img, msk = self._read_image_and_mask(img_path, mask_path)
        H, W = msk.shape

        # 1) heal 1px gaps if wanted
        msk_healed = _heal_watershed_gaps(msk, radius=self.heal_radius)

        # 2) make instances
        inst_full = sklabel(msk_healed, connectivity=1).astype(np.int32)
        inst_full, _, _ = relabel_sequential(inst_full)

        # 3) 4 heads from full image
        cell_full = (inst_full > 0).astype(np.float32)
        bound_full = _make_soft_boundary_from_instances(
            inst_full,
            ring_width=max(1, self.boundary_ring_width),
            soft_band=max(1, self.boundary_soft_band),
            sigma=self.boundary_sigma,
        ).astype(np.float32)

        # centers (from instances)
        centers = []
        max_id = int(inst_full.max())
        for k in range(1, max_id + 1):
            ys, xs = np.where(inst_full == k)
            if ys.size == 0:
                continue
            cy = int(np.mean(ys))
            cx = int(np.mean(xs))
            centers.append((cy, cx))

        center_stem = _make_center_stem_from_centers(centers, (H, W))
        center_heat = _make_center_heatmap(center_stem, sigma=self.center_sigma)
        energy_full = _make_energy_from_instances(inst_full)

        full_tgt = np.stack([cell_full, bound_full, center_heat, energy_full], axis=0).astype(np.float32)
        full_img = np.transpose(img.astype(np.float32), (2, 0, 1))  # [3,H,W]

        full_meta = {
            "src_path": img_path,
            "mask_path": mask_path,
            "H_in": int(H),
            "W_in": int(W),
            "n_cells": int(max_id),
        }

        # 4) tile like SimCellsDataset(n_tiles = -1)
        tiles = self._enumerate_full_tiles(H, W)

        imgs_out = []
        tgts_out = []
        inst_out = []
        tiles_meta = []

        for (y0, y1, x0, x1) in tiles:
            # crop
            img_t = img[y0:y1, x0:x1, :]
            cell_t = cell_full[y0:y1, x0:x1]
            bound_t = bound_full[y0:y1, x0:x1]
            center_t = center_heat[y0:y1, x0:x1]
            energy_t = energy_full[y0:y1, x0:x1]
            inst_t = inst_full[y0:y1, x0:x1]

            th = img_t.shape[0]
            tw = img_t.shape[1]
            pad_y = self.target - th
            pad_x = self.target - tw

            if pad_y > 0 or pad_x > 0:
                # pad to target
                img_t = np.pad(img_t, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant", constant_values=0.0)
                cell_t = np.pad(cell_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                bound_t = np.pad(bound_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                center_t = np.pad(center_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                energy_t = np.pad(energy_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                inst_t = np.pad(inst_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)

            # re-label per tile
            inst_t, _, _ = relabel_sequential(inst_t.astype(np.int32))

            # pack tgt
            tgt_t = np.stack([cell_t, bound_t, center_t, energy_t], axis=0).astype(np.float32)

            # optional transforms (Albumentations style)
            if self.transforms is not None:
                out = self.transforms(
                    image=img_t,
                    masks=[cell_t, bound_t, center_t, energy_t],
                )
                img_t = out["image"]
                cell_t, bound_t, center_t, energy_t = out["masks"]
                tgt_t = np.stack([cell_t, bound_t, center_t, energy_t], axis=0).astype(np.float32)

            # to CHW
            img_chw = np.transpose(img_t.astype(np.float32), (2, 0, 1))

            imgs_out.append(img_chw)
            tgts_out.append(tgt_t)
            inst_out.append(inst_t.astype(np.int32))

            tile_meta = {
                "mode": "tiles",
                "tile_xy": (int(y0), int(x0)),
                "tile_hw": (self.target, self.target),
                "full_H": int(H),
                "full_W": int(W),
            }
            tiles_meta.append(tile_meta)

        # stack
        imgs_t = torch.from_numpy(np.stack(imgs_out, axis=0).astype(np.float32))  # [T,3,S,S]
        tgts_t = torch.from_numpy(np.stack(tgts_out, axis=0).astype(np.float32))  # [T,4,S,S]
        inst_t = torch.from_numpy(np.stack(inst_out, axis=0).astype(np.int32))    # [T,S,S]

        extras = {
            "instance_labels": inst_t,
            "meta": {
                "full": full_meta,
                "tiles": tiles_meta,
            },
        }
        return imgs_t, tgts_t, extras

def create_external_cells_h5_tiles(
    root_dir: str,
    out_path: str,
    target: int = 512,
    tile_overlap: int = 64,
    heal_radius: int = 1,
    num_workers: int = 4,
    compression: Optional[str] = "lzf",
    flush_every: int = 8,
    resume: bool = True,
    transforms=None,
):
    """
    Build an HDF5 with tiled external images (real images + 8-bit masks).

    Layout (flat over tiles, not over images):

        /imgs : float32 [N_tiles, 3, target, target]
        /tgts : float32 [N_tiles, 4, target, target]
        /inst : int32   [N_tiles, target, target]
        /meta : vlen JSON, per tile:
            {
              "tile":  <tile meta>,
              "full":  <full-image meta>,
              "image_idx": <index in dataset>
            }

    Attributes:
        - version
        - source
        - n_images
        - n_images_written
        - n_tiles_written
        - target
        - tile_overlap
        - heal_radius
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # dataset
    ds = ExternalCellsTilesDataset(
        root_dir=root_dir,
        target=target,
        tile_overlap=tile_overlap,
        heal_radius=heal_radius,
        transforms=transforms,
    )

    # loader: batch_size=1 because each image has variable #tiles
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        persistent_workers=(num_workers > 0),
        pin_memory=False,
    )

    # for ctrl+c
    stop = {"flag": False}

    def _handle(sig, frm):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    vlen_str = h5py.special_dtype(vlen=str)
    new_file = (not os.path.exists(out_path))

    with h5py.File(out_path, "a", libver="latest") as f:
        if new_file:
            f.attrs.update({
                "version": 1,
                "source": os.path.abspath(root_dir),
                "n_images": len(ds),
                "n_images_written": 0,
                "n_tiles_written": 0,
                "target": int(target),
                "tile_overlap": int(tile_overlap),
                "heal_radius": int(heal_radius),
            })
            f.create_dataset(
                "imgs",
                shape=(0, 3, target, target),
                maxshape=(None, 3, target, target),
                dtype=np.float32,
                chunks=(1, 3, target, target),
                compression=compression,
            )
            f.create_dataset(
                "tgts",
                shape=(0, 4, target, target),
                maxshape=(None, 4, target, target),
                dtype=np.float32,
                chunks=(1, 4, target, target),
                compression=compression,
            )
            f.create_dataset(
                "inst",
                shape=(0, target, target),
                maxshape=(None, target, target),
                dtype=np.int32,
                chunks=(1, target, target),
                compression=compression,
            )
            f.create_dataset(
                "meta",
                shape=(0,),
                maxshape=(None,),
                dtype=vlen_str,
                chunks=(1024,),
            )
        else:
            # sanity
            assert int(f.attrs["target"]) == int(target), "target mismatch"
            assert int(f.attrs["tile_overlap"]) == int(tile_overlap), "tile_overlap mismatch"
            if "n_images_written" not in f.attrs:
                f.attrs["n_images_written"] = 0
            if "n_tiles_written" not in f.attrs:
                f.attrs["n_tiles_written"] = 0

        d_imgs = f["imgs"]
        d_tgts = f["tgts"]
        d_inst = f["inst"]
        d_meta = f["meta"]

        n_images_written = int(f.attrs["n_images_written"])
        n_tiles_written = int(f.attrs["n_tiles_written"])

        it = iter(dl)

        # resume: skip already written images
        if resume and n_images_written > 0:
            for _ in range(n_images_written):
                try:
                    next(it)
                except StopIteration:
                    break

        pbar = tqdm(
            total=len(ds),
            initial=n_images_written,
            desc="export external cells tiles",
            dynamic_ncols=True,
        )

        def _flush_safe():
            f.flush()
            try:
                f.id.flush()
            except Exception:
                pass
            try:
                fd = f.id.get_vfd_handle()
                if fd is not None:
                    os.fsync(fd)
            except Exception:
                pass

        img_since_flush = 0

        for img_idx in range(n_images_written, len(ds)):
            if stop["flag"]:
                break

            try:
                imgs_tiles, tgts_tiles, extras = next(it)
            except StopIteration:
                break


            # strip the loader batch dim
            imgs_tiles = imgs_tiles[0]   # [T, 3, S, S]
            tgts_tiles = tgts_tiles[0]   # [T, 4, S, S]
            inst_tiles = extras["instance_labels"][0]  # [T, S, S]
            tiles_meta = extras["meta"]["tiles"]
            full_meta = extras["meta"]["full"]

            T = int(imgs_tiles.shape[0])
            cur_n = n_tiles_written
            new_n = cur_n + T

            # grow datasets
            d_imgs.resize((new_n, 3, target, target))
            d_tgts.resize((new_n, 4, target, target))
            d_inst.resize((new_n, target, target))
            d_meta.resize((new_n,))

            # write data
            d_imgs[cur_n:new_n] = imgs_tiles.detach().cpu().numpy().astype(np.float32)
            d_tgts[cur_n:new_n] = tgts_tiles.detach().cpu().numpy().astype(np.float32)
            d_inst[cur_n:new_n] = inst_tiles.detach().cpu().numpy().astype(np.int32)

            # write meta (fixed part)
            meta_jsons = []
            # run both tile-level and full-image meta through jsonify
            full_meta_jsonable = jsonify(full_meta)
            for tm in tiles_meta:
                tm_jsonable = jsonify(tm)
                m = {
                    "tile": tm_jsonable,
                    "full": full_meta_jsonable,
                    "image_idx": int(img_idx),
                }
                meta_jsons.append(json.dumps(m, separators=(",", ":")))
            d_meta[cur_n:new_n] = meta_jsons

            # update counters
            n_tiles_written = new_n
            n_images_written += 1
            f.attrs.modify("n_tiles_written", int(n_tiles_written))
            f.attrs.modify("n_images_written", int(n_images_written))

            img_since_flush += 1
            if img_since_flush % max(1, flush_every) == 0:
                _flush_safe()
                img_since_flush = 0

            pbar.update(1)

        # final flush
        _flush_safe()
        pbar.close()

        print(
            f"[export] done: images={n_images_written}/{len(ds)}, tiles={n_tiles_written} → {out_path}"
        )

    return out_path

