from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Literal
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

from ..image_utils.tiling import iter_sliding_windows
from ..segmentation.config import test_scene, test_camera
from ..segmentation.image_simulation import simulate_image
from ..segmentation.image_dataset import (
    _make_soft_boundary_from_instances,
    _make_center_stem_from_centers,
    _make_center_heatmap,
    _make_energy_from_instances,
)
from .utils import jsonify

def collate_tiles_single(batch):
    """
    For *grouped* tile datasets, but with batch_size=1.
    Your dataset returns:
        imgs:  [T, 3, S, S]
        tgts:  [T, 4, S, S]
        extras: {"inst_tiles": [T,S,S], "tiles_meta": [...], "full": {...}}
    We just unwrap the single element so the caller sees exactly that.
    """
    assert len(batch) == 1, "collate_tiles_single expects batch_size=1"
    return batch[0]

def collate_no_meta_flat(batch):
    """
    Use this for flat/tile datasets:
      (img: [3,H,W], tgt: [C,H,W], extras{"instance_labels": [H,W], "meta": dict})
    → stacks nicely.
    """
    imgs, tgts, exs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)                 # [B,3,H,W]
    tgts = torch.stack(tgts, dim=0)                 # [B,C,H,W]
    inst = torch.stack([e["instance_labels"] for e in exs], dim=0)  # [B,H,W]
    metas = [e["meta"] for e in exs]
    extras_out = {
        "instance_labels": inst,
        "meta": metas,
    }
    return imgs, tgts, extras_out


def collate_groups_keep_lists(batch):
    """
    Use this for group-mode datasets:
      (imgs: [T,3,H,W], tgts: [T,4,H,W], extras{"inst_tiles": [T,H,W], "tiles_meta": [...]})
    T may differ per sample, so we CANNOT stack across batch.
    We return lists, 1 entry per sample in the batch.
    DataLoader(batch_size>1) is still okay, but you must loop over the list.
    """
    imgs, tgts, exs = zip(*batch)   # each imgs[i] is [T_i,3,H,W]
    return list(imgs), list(tgts), list(exs)


def collate_smart(batch):
    """
    Auto-detects if this is flat or grouped.
    - if first img is 3D  -> flat  -> stack
    - if first img is 4D  -> group -> keep lists
    """
    first_img = batch[0][0]
    if isinstance(first_img, torch.Tensor):
        ndim = first_img.dim()
    else:
        ndim = torch.as_tensor(first_img).dim()

    if ndim == 3:
        return collate_no_meta_flat(batch)
    elif ndim == 4:
        return collate_groups_keep_lists(batch)
    else:
        raise ValueError(f"Unknown sample shape for collate: ndim={ndim}")

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
    selem = morphology.disk(int(radius))
    healed = morphology.binary_dilation(mask.astype(bool), selem)
    return healed.astype(np.uint8)

def _crop_full_meta_to_tile(
    full_meta: Dict[str, Any],
    y0: int,
    x0: int,
    h: int,
    w: int,
) -> Dict[str, Any]:
    """
    For now real data has little meta, but we still build a tile-level dict
    that tells exactly which tile this is and where it came from.
    """
    tm = dict(full_meta)
    tm.update({
        "tile_xy": (int(y0), int(x0)),
        "tile_hw": (int(h), int(w)),
    })
    return tm

def _crop_sim_meta_to_tile(meta: Dict[str, Any], y0: int, x0: int, h: int, w: int) -> Dict[str, Any]:
    """
    Same logic we agreed on:
      - keep + shift centers
      - filter labels / final_sigmas
      - recompute n_cells / frac_positive
      - shift well_center
      - update params.{H,W,n_cells,frac_positive,well_center}
      - keep radius_px
    """
    new_meta = dict(meta)

    centers = meta.get("centers", [])
    labels = meta.get("labels", [])
    sigmas = meta.get("final_sigmas", None)

    kept_centers = []
    kept_labels = []
    kept_sigmas = []

    for i, (cy, cx) in enumerate(centers):
        ny = cy - y0
        nx = cx - x0
        if 0 <= ny < h and 0 <= nx < w:
            kept_centers.append((int(ny), int(nx)))
            if i < len(labels):
                kept_labels.append(int(labels[i]))
            if sigmas is not None and i < len(sigmas):
                kept_sigmas.append(float(sigmas[i]))

    new_meta["centers"] = kept_centers

    if isinstance(labels, np.ndarray):
        new_meta["labels"] = np.array(kept_labels, dtype=labels.dtype)
    else:
        new_meta["labels"] = kept_labels

    if sigmas is not None:
        new_meta["final_sigmas"] = np.array(kept_sigmas, dtype=np.float32)

    n_cells_tile = len(kept_centers)
    new_meta["n_cells"] = int(n_cells_tile)

    if n_cells_tile > 0 and len(kept_labels) == n_cells_tile:
        new_meta["frac_positive"] = float(np.mean(kept_labels))
    else:
        new_meta["frac_positive"] = 0.0

    if "well_center" in meta and meta["well_center"] is not None:
        wy, wx = meta["well_center"]
        new_meta["well_center"] = (float(wy - y0), float(wx - x0))

    params = meta.get("params", None)
    if isinstance(params, dict):
        new_params = dict(params)
        new_params["H"] = int(h)
        new_params["W"] = int(w)
        if "n_cells" in new_params:
            new_params["n_cells"] = int(n_cells_tile)
        if "frac_positive" in new_params:
            new_params["frac_positive"] = float(new_meta["frac_positive"])
        if "well_center" in new_params and "well_center" in new_meta:
            new_params["well_center"] = new_meta["well_center"]
        new_meta["params"] = new_params

    return new_meta

class DiskValidationDataset(Dataset):
    """
    HDF5 reader for *tile-based* validation sets we exported.

    Works with both:
      - val_sim_tiles.h5  (meta has "scene_idx")
      - val_external_tiles.h5 (meta has "image_idx")

    Modes:
      - mode="flat":
          __getitem__(i) -> (img: [3,S,S], tgt: [C,S,S], extras)
          like DiskSimCellsDataset
      - mode="group":
          __getitem__(i) -> (imgs: [T,3,S,S], tgts: [T,C,S,S], extras)
          returns *all tiles* for one scene/image

    extras always has:
      - "instance_labels" (flat) or "inst_tiles" (group)
      - "meta" (flat) or "tiles_meta" (group)
    """

    def __init__(
        self,
        h5_path: str,
        mode: Literal["flat", "group"] = "flat",
    ):
        super().__init__()
        self.h5_path = str(h5_path)
        self.mode = mode
        assert self.mode in ("flat", "group")

        # lightweight probe
        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as f:
            self._N = int(f["imgs"].shape[0])
            self._C_img = int(f["imgs"].shape[1])
            self._S = int(f["imgs"].shape[2])
            self._C_tgt = int(f["tgts"].shape[1])
            assert "inst" in f and "meta" in f, "HDF5 must have /inst and /meta"

            # we need to know how to group -> read all metas once
            metas_raw = f["meta"][:]

        # parse metas to build groups
        metas: List[Dict[str, Any]] = []
        for mj in metas_raw:
            if isinstance(mj, bytes):
                m = json.loads(mj.decode("utf-8"))
            else:
                m = json.loads(mj)
            metas.append(m)

        # figure out key name: "scene_idx" (sim) or "image_idx" (external)
        # look at first meta that has one of them
        group_key = None
        for m in metas:
            if "scene_idx" in m:
                group_key = "scene_idx"
                break
            if "image_idx" in m:
                group_key = "image_idx"
                break
        if group_key is None:
            # fallback: treat each row as its own group
            group_key = "_row"

        self._group_key = group_key

        if self.mode == "flat":
            # just store metas; we already know N
            self._metas = metas
            self._groups = None
        else:
            # build groups: group_id -> list of tile indices
            from collections import defaultdict
            g = defaultdict(list)
            for i, m in enumerate(metas):
                gid = m.get(group_key, i)
                g[int(gid)].append(i)
            # keep groups in index order
            group_ids = sorted(g.keys())
            self._groups: List[List[int]] = [g[g_id] for g_id in group_ids]
            self._metas = metas  # keep to pass through

        # lazy open
        self._h5: Optional[h5py.File] = None
        self._imgs = self._tgts = self._inst = self._meta = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._imgs = self._h5["imgs"]
            self._tgts = self._h5["tgts"]
            self._inst = self._h5["inst"]
            self._meta = self._h5["meta"]

    def __len__(self) -> int:
        if self.mode == "flat":
            return self._N
        else:
            return len(self._groups)

    def __getitem__(self, idx: int):
        self._ensure_open()

        if self.mode == "flat":
            k = int(idx)
            img = self._imgs[k]   # [3,S,S]
            tgt = self._tgts[k]   # [C,S,S]
            inst = self._inst[k]  # [S,S]
            meta_json = self._meta[k]
            if isinstance(meta_json, bytes):
                meta = json.loads(meta_json.decode("utf-8"))
            else:
                meta = json.loads(meta_json)

            img_t = torch.from_numpy(np.asarray(img, dtype=np.float32))
            tgt_t = torch.from_numpy(np.asarray(tgt, dtype=np.float32))
            inst_t = torch.from_numpy(np.asarray(inst, dtype=np.int32))

            extras = {
                "instance_labels": inst_t,
                "meta": meta,
            }
            return img_t, tgt_t, extras

        else:  # group mode
            tile_ids = self._groups[idx]  # list of h5 rows
            imgs = []
            tgts = []
            insts = []
            metas = []
            for k in tile_ids:
                img = self._imgs[k]
                tgt = self._tgts[k]
                inst = self._inst[k]
                meta_json = self._meta[k]
                if isinstance(meta_json, bytes):
                    meta = json.loads(meta_json.decode("utf-8"))
                else:
                    meta = json.loads(meta_json)

                imgs.append(np.asarray(img, dtype=np.float32))
                tgts.append(np.asarray(tgt, dtype=np.float32))
                insts.append(np.asarray(inst, dtype=np.int32))
                metas.append(meta)

            imgs_t = torch.from_numpy(np.stack(imgs, axis=0))
            tgts_t = torch.from_numpy(np.stack(tgts, axis=0))
            insts_t = torch.from_numpy(np.stack(insts, axis=0))

            extras = {
                "inst_tiles": insts_t,
                "tiles_meta": metas,
            }
            return imgs_t, tgts_t, extras

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
            self._imgs = self._tgts = self._inst = self._meta = None

    def __del__(self):
        self.close()


class ValidationSimDataset(Dataset):
    """
    Sim-based val dataset that:
      - simulates full scene
      - builds full 4-head GT
      - splits into tiles on a grid (tile, overlap)
      - makes per-tile metadata
      - also returns full scene

    __getitem__ returns:
      imgs_tiles : [T, 3, tile, tile]
      tgts_tiles : [T, 4, tile, tile]
      extras     : {
          "inst_tiles": [T, tile, tile],
          "tiles_meta": list[dict],  # per-tile cropped meta
          "full": {
              "img":  [3, H, W],
              "tgt":  [4, H, W],
              "inst": [H, W],
              "meta": dict,          # full meta (uncropped)
          },
          "sim_kwargs": dict
      }
    """

    def __init__(
        self,
        length: int = 200,
        scene_cfg=None,
        camera_cfg=None,
        rng_seed: int = 187,
        tile: int = 512,
        overlap: int = 64,
        boundary_ring_width: int = 1,
        boundary_soft_band: int = 2,
        boundary_sigma: float = 1.0,
        center_sigma: float = 1.0,
    ):
        assert scene_cfg is not None, "scene_cfg is required"
        assert camera_cfg is not None, "camera_cfg is required"

        self.length = int(length)
        self.scene_cfg = scene_cfg
        self.camera_cfg = camera_cfg

        self.tile = int(tile)
        self.overlap = int(overlap)

        self.boundary_ring_width = int(boundary_ring_width)
        self.boundary_soft_band = int(boundary_soft_band)
        self.boundary_sigma = float(boundary_sigma)
        self.center_sigma = float(center_sigma)

        self.base_rng = np.random.default_rng(int(rng_seed))

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # --- simulate full ---
        rng = np.random.default_rng(
            int(self.base_rng.integers(0, 2**31 - 1)) ^ int(idx)
        )
        sim_kwargs = self.scene_cfg.sample_kwargs(rng, camera=self.camera_cfg)
        sim_kwargs.setdefault("return_targets", True)

        img, meta_full, targets = simulate_image(**sim_kwargs)
        # img: (H, W, 3)
        cell = targets["cell_mask"].astype(np.float32)
        bound = targets["boundary"].astype(np.float32)
        inst = targets["instance_labels"].astype(np.int32)

        H, W = cell.shape

        # --- build full 4-head ---
        bound_soft_full = _make_soft_boundary_from_instances(
            inst,
            ring_width=max(1, self.boundary_ring_width),
            soft_band=max(1, self.boundary_soft_band),
            sigma=self.boundary_sigma,
        ).astype(np.float32)

        if "centers" in meta_full and meta_full["centers"]:
            centers_full = meta_full["centers"]
            center_stem_full = _make_center_stem_from_centers(centers_full, (H, W))
        else:
            centers_tmp = []
            for k in range(1, int(inst.max()) + 1):
                ys, xs = np.where(inst == k)
                if ys.size == 0:
                    continue
                cy = int(np.mean(ys))
                cx = int(np.mean(xs))
                centers_tmp.append((cy, cx))
            center_stem_full = _make_center_stem_from_centers(centers_tmp, (H, W))

        center_heat_full = _make_center_heatmap(center_stem_full, sigma=self.center_sigma)
        energy_full = _make_energy_from_instances(inst)

        full_tgt = np.stack(
            [cell, bound_soft_full, center_heat_full, energy_full],
            axis=0,
        ).astype(np.float32)
        full_img = np.transpose(img.astype(np.float32), (2, 0, 1))  # [3, H, W]

        # relabel full inst cleanly
        inst_rel_full, _, _ = relabel_sequential(inst.astype(np.int32))

        # --- now make tiles on grid ---
        T_imgs: List[np.ndarray] = []
        T_tgts: List[np.ndarray] = []
        T_inst: List[np.ndarray] = []
        T_meta: List[Dict[str, Any]] = []

        for (y0, y1, x0, x1) in iter_sliding_windows(H, W, self.tile, self.overlap):
            # crop
            img_t = img[y0:y1, x0:x1, :]                     # (th, tw, 3)
            cell_t = cell[y0:y1, x0:x1]
            bound_t = bound_soft_full[y0:y1, x0:x1]
            center_t = center_heat_full[y0:y1, x0:x1]
            energy_t = energy_full[y0:y1, x0:x1]
            inst_t = inst_rel_full[y0:y1, x0:x1]

            th = img_t.shape[0]
            tw = img_t.shape[1]

            # pad to full tile if needed
            py = self.tile - th
            px = self.tile - tw
            if py > 0 or px > 0:
                # image
                img_t = np.pad(
                    img_t,
                    pad_width=((0, py), (0, px), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
                # 4-head
                cell_t = np.pad(cell_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                bound_t = np.pad(bound_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                center_t = np.pad(center_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                energy_t = np.pad(energy_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                # inst
                inst_t = np.pad(inst_t, ((0, py), (0, px)), mode="constant", constant_values=0)

            # relabel tile inst compact (after pad)
            inst_t, _, _ = relabel_sequential(inst_t.astype(np.int32))

            tgt_t = np.stack([cell_t, bound_t, center_t, energy_t], axis=0).astype(np.float32)
            img_t = np.transpose(img_t.astype(np.float32), (2, 0, 1))  # [3, tile, tile]

            T_imgs.append(img_t)
            T_tgts.append(tgt_t)
            T_inst.append(inst_t.astype(np.int32))

            # tile-specific meta
            tile_meta = _crop_sim_meta_to_tile(
                meta_full,
                y0=y0,
                x0=x0,
                h=min(self.tile, H - y0),
                w=min(self.tile, W - x0),
            )
            tile_meta.update({
                "mode": "tiles",
                "tile_xy": (int(y0), int(x0)),
                "tile_hw": (int(self.tile), int(self.tile)),
                "H_full": int(H),
                "W_full": int(W),
            })
            T_meta.append(tile_meta)

        imgs_tiles = torch.from_numpy(np.stack(T_imgs, axis=0).copy())   # [T, 3, tile, tile]
        tgts_tiles = torch.from_numpy(np.stack(T_tgts, axis=0).copy())   # [T, 4, tile, tile]
        inst_tiles = torch.from_numpy(np.stack(T_inst, axis=0).copy())   # [T, tile, tile]

        extras = {
            "inst_tiles": inst_tiles,
            "tiles_meta": T_meta,
            "full": {
                "img": torch.from_numpy(full_img.copy()),
                "tgt": torch.from_numpy(full_tgt.copy()),
                "inst": torch.from_numpy(inst_rel_full.copy()),
                "meta": meta_full,  # full image meta
            },
            "sim_kwargs": {k: v for k, v in sim_kwargs.items() if k != "seed"},
        }

        return imgs_tiles, tgts_tiles, extras

class ExternalImageValDataset(Dataset):
    """
    External (real) image + mask dataset that:
      - loads pairs from a folder
      - heals 1px splits from ImageJ
      - makes instances
      - builds 4 heads from THAT
      - tiles in a grid (tile, overlap)
      - returns tiles as a batch, but also the full image

    __getitem__ returns:
      imgs_tiles : [T, 3, tile, tile]
      tgts_tiles : [T, 4, tile, tile]
      extras     : {
          "inst_tiles": [T, tile, tile],
          "tiles_meta": list[dict],
          "full": {
              "img":  [3, H, W],
              "tgt":  [4, H, W],
              "inst": [H, W],
              "meta": dict,
          }
      }
    """
    def __init__(
        self,
        root_dir: str,
        tile: int = 512,
        overlap: int = 64,
        heal_radius: int = 1,
    ):
        assert os.path.isdir(root_dir), f"not a dir: {root_dir}"
        self.root_dir = root_dir
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.heal_radius = int(heal_radius)

        self.pairs = _find_pairs_strict(root_dir)
        if not self.pairs:
            raise RuntimeError("no (img, img_mask) pairs found")

    def __len__(self):
        return len(self.pairs)

    def _read_image_and_mask(self, img_path: str, mask_path: str):
        # image
        img = tiff.imread(img_path).astype(np.float32, copy=False)
        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = img[..., 0]
            elif img.shape[2] > 3:
                img = img[..., :3]

        # scale to [0,1]
        if img.dtype.kind in ("u", "i"):
            vmax = 65535.0 if img.max() > 255 else 255.0
            img = img / max(vmax, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)

        # 3ch
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 2:
            img = np.concatenate([img, img[..., :1]], axis=-1)

        # mask
        msk = tiff.imread(mask_path)
        if msk.ndim > 2:
            msk = msk[..., 0]
        msk = (msk >= 255).astype(np.uint8)

        return img, msk

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img, msk = self._read_image_and_mask(img_path, mask_path)
        H, W = msk.shape

        # heal watershed 1px gaps
        msk_healed = _heal_watershed_gaps(msk, radius=self.heal_radius)

        # instances
        inst_full = sklabel(msk_healed, connectivity=1).astype(np.int32)
        inst_full, _, _ = relabel_sequential(inst_full)

        # 4 heads from instances
        cell = (inst_full > 0).astype(np.float32)
        bound_soft = _make_soft_boundary_from_instances(
            inst_full,
            ring_width=2,
            soft_band=2,
            sigma=1.0,
        ).astype(np.float32)

        # centers from instance labels
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
        center_heat = _make_center_heatmap(center_stem, sigma=1.0)
        energy = _make_energy_from_instances(inst_full)

        full_tgt = np.stack([cell, bound_soft, center_heat, energy], axis=0).astype(np.float32)
        full_img = np.transpose(img.astype(np.float32), (2, 0, 1))

        full_meta = {
            "src_path": img_path,
            "mask_path": mask_path,
            "H_in": int(H),
            "W_in": int(W),
            "n_cells": int(max_id),
            # frac_pos we don’t have yet (will come from CSV later)
        }

        # ---- tiling on grid ----
        T_imgs = []
        T_tgts = []
        T_inst = []
        T_meta = []

        for (y0, y1, x0, x1) in iter_sliding_windows(H, W, self.tile, self.overlap):
            img_t = img[y0:y1, x0:x1, :]
            cell_t = cell[y0:y1, x0:x1]
            bound_t = bound_soft[y0:y1, x0:x1]
            center_t = center_heat[y0:y1, x0:x1]
            energy_t = energy[y0:y1, x0:x1]
            inst_t = inst_full[y0:y1, x0:x1]

            th = img_t.shape[0]
            tw = img_t.shape[1]
            py = self.tile - th
            px = self.tile - tw

            if py > 0 or px > 0:
                img_t = np.pad(img_t, ((0, py), (0, px), (0, 0)), mode="constant", constant_values=0.0)
                cell_t = np.pad(cell_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                bound_t = np.pad(bound_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                center_t = np.pad(center_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                energy_t = np.pad(energy_t, ((0, py), (0, px)), mode="constant", constant_values=0.0)
                inst_t = np.pad(inst_t, ((0, py), (0, px)), mode="constant", constant_values=0)

            inst_t, _, _ = relabel_sequential(inst_t.astype(np.int32))

            tgt_t = np.stack([cell_t, bound_t, center_t, energy_t], axis=0).astype(np.float32)
            img_t = np.transpose(img_t.astype(np.float32), (2, 0, 1))

            T_imgs.append(img_t)
            T_tgts.append(tgt_t)
            T_inst.append(inst_t.astype(np.int32))

            tile_meta = _crop_full_meta_to_tile(
                full_meta,
                y0=y0,
                x0=x0,
                h=min(self.tile, H - y0),
                w=min(self.tile, W - x0),
            )
            T_meta.append(tile_meta)

        imgs_tiles = torch.from_numpy(np.stack(T_imgs, axis=0).copy())   # [T, 3, tile, tile]
        tgts_tiles = torch.from_numpy(np.stack(T_tgts, axis=0).copy())   # [T, 4, tile, tile]
        inst_tiles = torch.from_numpy(np.stack(T_inst, axis=0).copy())   # [T, tile, tile]

        extras = {
            "inst_tiles": inst_tiles,
            "tiles_meta": T_meta,
            "full": {
                "img": torch.from_numpy(full_img.copy()),
                "tgt": torch.from_numpy(full_tgt.copy()),
                "inst": torch.from_numpy(inst_full.copy()),
                "meta": full_meta,
            },
        }

        return imgs_tiles, tgts_tiles, extras

def create_validation_h5_tiles(
    out_path: str,
    length: int,
    tile: int = 512,
    overlap: int = 64,
    rng_seed: int = 187,
    num_workers_gen: int = 4,
    compression: Optional[str] = "lzf",
    flush_every: int = 8,
    resume: bool = True,
    camera_cfg=None,
    scene_cfg=None,
):
    """
    Build an HDF5 with *tiled* simulated validation data.

    Layout (flat over tiles, not over scenes):

      /imgs : float32 [N_tiles, 3, tile, tile]
      /tgts : float32 [N_tiles, 4, tile, tile]
      /inst : int32   [N_tiles, tile, tile]
      /meta : vlen JSON per tile

    Each JSON entry contains **both**:
      {
        "tile": <tile-specific meta from dataset>,
        "full": <full-image meta from dataset>,
        "scene_idx": <which simulated scene this tile came from>
      }

    Attributes:
      - version
      - scene_length     = length           (how many scenes we wanted)
      - n_scenes_written = actual scenes written
      - tile
      - overlap
      - rng_seed

    Why resizable:
      - we don’t know N_tiles in advance (depends on (H, W) per scene)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if camera_cfg is None:
        camera_cfg = test_camera()
    if scene_cfg is None:
        scene_cfg = test_scene()

    # --- graceful stop ---
    stop = {"flag": False}
    def _handle_signal(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- dataset & dataloader ---
    ds = ValidationSimDataset(
        length=length,
        scene_cfg=scene_cfg,
        camera_cfg=camera_cfg,
        rng_seed=rng_seed,
        tile=tile,
        overlap=overlap,
    )
    # we must use batch_size=1 because each item has variable #tiles
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers_gen),
        pin_memory=False,
        persistent_workers=(num_workers_gen > 0),
        collate_fn=collate_tiles_single
    )

    # --- open/create HDF5 ---
    vlen_str = h5py.special_dtype(vlen=str)
    new_file = (not os.path.exists(out_path))
    with h5py.File(out_path, "a", libver="latest") as f:
        if new_file:
            # create empty, resizable datasets
            f.attrs.update({
                "version": 1,
                "scene_length": int(length),
                "n_scenes_written": 0,
                "n_tiles_written": 0,
                "tile": int(tile),
                "overlap": int(overlap),
                "rng_seed": int(rng_seed),
            })
            maxshape_imgs = (None, 3, tile, tile)
            maxshape_tgts = (None, 4, tile, tile)
            maxshape_inst = (None, tile, tile)

            f.create_dataset(
                "imgs",
                shape=(0, 3, tile, tile),
                maxshape=maxshape_imgs,
                dtype=np.float32,
                chunks=(1, 3, tile, tile),
                compression=compression,
            )
            f.create_dataset(
                "tgts",
                shape=(0, 4, tile, tile),
                maxshape=maxshape_tgts,
                dtype=np.float32,
                chunks=(1, 4, tile, tile),
                compression=compression,
            )
            f.create_dataset(
                "inst",
                shape=(0, tile, tile),
                maxshape=maxshape_inst,
                dtype=np.int32,
                chunks=(1, tile, tile),
                compression=compression,
            )
            f.create_dataset(
                "meta",
                shape=(0,),
                maxshape=(None,),
                dtype=vlen_str,
                chunks=(min(1024, length * 4),),
            )
        else:
            # basic checks
            assert int(f.attrs["tile"]) == int(tile), "tile mismatch"
            assert int(f.attrs["overlap"]) == int(overlap), "overlap mismatch"
            if "n_scenes_written" not in f.attrs:
                f.attrs["n_scenes_written"] = 0
            if "n_tiles_written" not in f.attrs:
                f.attrs["n_tiles_written"] = 0

        d_imgs = f["imgs"]
        d_tgts = f["tgts"]
        d_inst = f["inst"]
        d_meta = f["meta"]

        n_scenes_written = int(f.attrs["n_scenes_written"])
        n_tiles_written = int(f.attrs["n_tiles_written"])

        # if resume: just skip first n_scenes_written scenes from dataloader
        it = iter(dl)

        # skip scenes
        if resume and n_scenes_written > 0:
            for _ in range(n_scenes_written):
                try:
                    next(it)
                except StopIteration:
                    break

        pbar = tqdm(
            total=length,
            initial=n_scenes_written,
            desc="export val tiles h5",
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

        scenes_done_since_flush = 0

        # --- main loop over scenes ---
        for scene_idx in range(n_scenes_written, length):
            if stop["flag"]:
                break

            try:
                batch = next(it)
            except StopIteration:
                break

            # batch is: (imgs_tiles, tgts_tiles, extras) but batched with batch_size=1
            imgs_tiles, tgts_tiles, extras = batch
            # remove batch dim
            imgs_tiles = imgs_tiles[0]     # [T, 3, tile, tile]
            tgts_tiles = tgts_tiles[0]     # [T, 4, tile, tile]
            inst_tiles = extras["inst_tiles"][0]  # [T, tile, tile]
            tiles_meta = extras["tiles_meta"]     # list len T
            full_meta = extras["full"]["meta"]

            T = int(imgs_tiles.shape[0])

            # current total tiles
            cur_n = n_tiles_written
            new_n = cur_n + T

            # resize datasets to fit new tiles
            d_imgs.resize((new_n, 3, tile, tile))
            d_tgts.resize((new_n, 4, tile, tile))
            d_inst.resize((new_n, tile, tile))
            d_meta.resize((new_n,))

            # write slice
            d_imgs[cur_n:new_n] = imgs_tiles.detach().cpu().numpy().astype(np.float32)
            d_tgts[cur_n:new_n] = tgts_tiles.detach().cpu().numpy().astype(np.float32)
            d_inst[cur_n:new_n] = inst_tiles.detach().cpu().numpy().astype(np.int32)

            # meta per tile (pack tile + full so we have both)
            meta_jsons = []
            for tm in tiles_meta:
                m = {
                    "tile": jsonify(tm),
                    "full": jsonify(full_meta),
                    "scene_idx": int(scene_idx),
                }
                meta_jsons.append(json.dumps(m, separators=(",", ":")))
            d_meta[cur_n:new_n] = meta_jsons

            # update counters
            n_tiles_written = new_n
            n_scenes_written += 1
            f.attrs.modify("n_tiles_written", int(n_tiles_written))
            f.attrs.modify("n_scenes_written", int(n_scenes_written))

            scenes_done_since_flush += 1
            if scenes_done_since_flush % max(1, flush_every) == 0:
                _flush_safe()
                scenes_done_since_flush = 0

            pbar.update(1)

        # final flush
        _flush_safe()
        pbar.close()

        print(
            f"[export] done: scenes={n_scenes_written}/{length}, tiles={n_tiles_written} → {out_path}"
        )

    return out_path

def create_external_images_h5_tiles(
    root_dir: str,
    out_path: str,
    tile: int = 512,
    overlap: int = 64,
    num_workers: int = 4,
    compression: Optional[str] = "lzf",
    flush_every: int = 8,
    resume: bool = True,
    heal_radius: int = 1,
):
    """
    Export external (real) images → tiled 4-head dataset to a single HDF5.

    One HDF5 row = one TILE.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # graceful stop
    stop = {"flag": False}
    def _handle(sig, frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    ds = ExternalImageValDataset(
        root_dir=root_dir,
        tile=tile,
        overlap=overlap,
        heal_radius=heal_radius,
    )
    dl = DataLoader(
        ds,
        batch_size=1,    # variable #tiles per image
        shuffle=False,
        num_workers=int(num_workers),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_smart
    )

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
                "tile": int(tile),
                "overlap": int(overlap),
            })
            f.create_dataset(
                "imgs",
                shape=(0, 3, tile, tile),
                maxshape=(None, 3, tile, tile),
                dtype=np.float32,
                chunks=(1, 3, tile, tile),
                compression=compression,
            )
            f.create_dataset(
                "tgts",
                shape=(0, 4, tile, tile),
                maxshape=(None, 4, tile, tile),
                dtype=np.float32,
                chunks=(1, 4, tile, tile),
                compression=compression,
            )
            f.create_dataset(
                "inst",
                shape=(0, tile, tile),
                maxshape=(None, tile, tile),
                dtype=np.int32,
                chunks=(1, tile, tile),
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
            assert int(f.attrs["tile"]) == int(tile), "tile mismatch"
            assert int(f.attrs["overlap"]) == int(overlap), "overlap mismatch"
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

        # resume
        if resume and n_images_written > 0:
            for _ in range(n_images_written):
                try:
                    next(it)
                except StopIteration:
                    break

        pbar = tqdm(
            total=len(ds),
            initial=n_images_written,
            desc="export external tiles",
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

            imgs_tiles = imgs_tiles[0]
            tgts_tiles = tgts_tiles[0]
            inst_tiles = extras["inst_tiles"][0]
            tiles_meta = extras["tiles_meta"]
            full_meta = extras["full"]["meta"]

            T = int(imgs_tiles.shape[0])
            cur_n = n_tiles_written
            new_n = cur_n + T

            d_imgs.resize((new_n, 3, tile, tile))
            d_tgts.resize((new_n, 4, tile, tile))
            d_inst.resize((new_n, tile, tile))
            d_meta.resize((new_n,))

            d_imgs[cur_n:new_n] = imgs_tiles.detach().cpu().numpy().astype(np.float32)
            d_tgts[cur_n:new_n] = tgts_tiles.detach().cpu().numpy().astype(np.float32)
            d_inst[cur_n:new_n] = inst_tiles.detach().cpu().numpy().astype(np.int32)

            meta_jsons = []
            for tm in tiles_meta:
                m = {
                    "tile": jsonify(tm),
                    "full": jsonify(full_meta),
                    "image_idx": int(img_idx),
                }
                meta_jsons.append(json.dumps(m, separators=(",", ":")))
            d_meta[cur_n:new_n] = meta_jsons

            n_tiles_written = new_n
            n_images_written += 1
            f.attrs.modify("n_tiles_written", int(n_tiles_written))
            f.attrs.modify("n_images_written", int(n_images_written))

            img_since_flush += 1
            if img_since_flush % max(1, flush_every) == 0:
                _flush_safe()
                img_since_flush = 0

            pbar.update(1)

        _flush_safe()
        pbar.close()

        print(f"[export] done: images={n_images_written}/{len(ds)}, tiles={n_tiles_written} → {out_path}")

    return out_path
