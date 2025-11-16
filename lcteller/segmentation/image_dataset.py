import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
from skimage.segmentation import relabel_sequential
from skimage.measure import label as sklabel

from .image_simulation import simulate_image
from .utils import (
    crop_sim_meta_to_tile,
    make_energy_from_instances,
    make_center_heatmap,
    make_center_stem_from_centers,
    make_soft_boundary_from_instances,
    square_crop_from_center_radius,
    estimate_well_mask,
    crop_rect,
    pad_to_square,
    resize_map,
    find_pairs_strict,
    heal_watershed_gaps,
    guess_data_csv_path,
    load_com_labels_csv,
    crop_external_meta_to_tile,
)


class SimCellsDataset(Dataset):
    """
    Modes:
      - pad_resize: pad to square, then resize to target
      - crop_well_resize: square-crop around well, then resize
      - tiles: return a random target×target tile
               (or many tiles if n_tiles>0, or all tiles if n_tiles==-1)
      - fullres: keep original resolution (no crop, no resize), one tile per scene

    Returns:
      img_t   : float32 [T, 3, S, S] in [0,1]
      tgt_t   : float32 [T, 4, S, S]
      extras  : dict with:
                  - instance_labels: int32 [T, S, S]
                  - meta: {
                        "full": original_sim_meta (possibly cropped if tiles),
                        "tiles": [tile_meta_0, ..., tile_meta_{T-1}]
                    }
    """
    def __init__(
        self,
        length=1000,
        mode="pad_resize",            # "pad_resize" | "crop_well_resize" | "tiles" | "fullres"
        target=512,
        n_tiles: int = 1,             # >0: pick that many random tiles; -1: cover full image; only for mode="tiles"
        tile_overlap=64,              # used when n_tiles == -1
        boundary_ring_width=1,
        boundary_soft_band=2,
        boundary_sigma=1.0,
        center_sigma=1.0,
        rng_seed=123,
        well_is_brighter="auto",
        transforms=None,              # optional Albumentations-style joint transforms

        # required
        scene_cfg=None,               # SimulatorConfig
        camera_cfg=None,              # CameraSetup
    ):
        assert scene_cfg is not None, "scene_cfg (SimulatorConfig) is required"
        assert camera_cfg is not None, "camera_cfg (CameraSetup) is required"

        self.length = int(length)
        self.mode = str(mode)
        assert self.mode in ("pad_resize", "crop_well_resize", "tiles", "fullres")

        self.target = int(target)
        self.n_tiles = int(n_tiles)
        self.tile_overlap = int(tile_overlap)

        self.boundary_ring_width = int(boundary_ring_width)
        self.boundary_soft_band = int(boundary_soft_band)
        self.boundary_sigma = float(boundary_sigma)
        self.center_sigma = float(center_sigma)

        self.well_is_brighter = well_is_brighter
        self.transforms = transforms
        self.base_rng = np.random.default_rng(rng_seed)

        self.scene_cfg = scene_cfg
        self.camera_cfg = camera_cfg

    def __len__(self):
        # length = number of simulated scenes, NOT number of tiles
        return self.length

    # ---- mode mappers (single-tile helpers) ----
    def _mode_pad_resize(self, img, cell, bound, inst):
        sq, (pad_top, pad_left), S = pad_to_square(img, pad_val=0.0)
        cell_sq, _, _ = pad_to_square(cell, pad_val=0.0)
        bound_sq, _, _ = pad_to_square(bound, pad_val=0.0)
        inst_sq,  _, _ = pad_to_square(inst,  pad_val=0)

        img_o   = resize_map(sq,       self.target, "image")
        cell_o  = resize_map(cell_sq,  self.target, "binary")
        bound_o = resize_map(bound_sq, self.target, "binary")
        inst_o  = resize_map(inst_sq,  self.target, "label")

        meta = dict(mode="pad_resize", pad_top=pad_top, pad_left=pad_left, S_in=S, scale=self.target/float(S))
        return img_o, cell_o, bound_o, inst_o, meta

    def _mode_crop_well_resize(self, img, cell, bound, inst, sim_meta):
        cycx = sim_meta.get("well_center", None)
        R = sim_meta.get("radius_px", None)
        if (cycx is None) or (R is None):
            _, center, radius = estimate_well_mask(img, blur_sigma=3.0, well_is_brighter=self.well_is_brighter)
        else:
            center = (float(cycx[0]), float(cycx[1]))
            radius = float(R)

        y0, y1, x0, x1 = square_crop_from_center_radius(cell.shape, center, radius, pad=8)
        img_c   = crop_rect(img,  y0, x0, y1-y0, x1-x0)
        cell_c  = crop_rect(cell, y0, x0, y1-y0, x1-x0)
        bound_c = crop_rect(bound,y0, x0, y1-y0, x1-x0)
        inst_c  = crop_rect(inst, y0, x0, y1-y0, x1-x0)

        img_o   = resize_map(img_c,  self.target, "image")
        cell_o  = resize_map(cell_c, self.target, "binary")
        bound_o = resize_map(bound_c,self.target, "binary")
        inst_o  = resize_map(inst_c, self.target, "label")

        scale = self.target / max(1.0, float(max(y1-y0, x1-x0)))
        meta = dict(mode="crop_well_resize",
                    crop=(int(y0), int(y1), int(x0), int(x1)),
                    scale=scale,
                    well_center=(float(center[0]), float(center[1])),
                    well_radius=float(radius))
        return img_o, cell_o, bound_o, inst_o, meta

    def _mode_fullres(self, img, cell, bound, inst, sim_meta):
        """
        Keep full resolution, do not crop or resize.
        Returns arrays as float32 / int32 with a simple meta dict.
        """
        img_o = img.astype(np.float32)
        cell_o = cell.astype(np.float32)
        bound_o = bound.astype(np.float32)
        inst_o = inst.astype(np.int32)

        H, W = cell_o.shape
        meta = dict(
            mode="fullres",
            height=int(H),
            width=int(W),
            scale=1.0,
        )
        return img_o, cell_o, bound_o, inst_o, meta

    def _mode_tiles_single(self, img, cell, bound, inst, rng):
        """old behavior: pick ONE random tile"""
        H, W = cell.shape
        th = self.target
        if H <= th or W <= th:
            # fallback
            img_o, cell_o, bound_o, inst_o, _ = self._mode_pad_resize(img, cell, bound, inst)
            meta = dict(mode="tiles", tile_xy=(0,0), tile_hw=(th, th))
            return img_o, cell_o, bound_o, inst_o, meta

        y0 = int(rng.integers(0, H - th + 1))
        x0 = int(rng.integers(0, W - th + 1))

        img_t   = crop_rect(img,  y0, x0, th, th)
        cell_t  = crop_rect(cell, y0, x0, th, th)
        bound_t = crop_rect(bound,y0, x0, th, th)
        inst_t  = crop_rect(inst, y0, x0, th, th)

        meta = dict(mode="tiles", tile_xy=(int(y0), int(x0)), tile_hw=(th, th))
        return (
            img_t.astype(np.float32),
            cell_t.astype(np.float32),
            bound_t.astype(np.float32),
            inst_t.astype(np.int32),
            meta
        )

    # ---- new: full sliding tiler for mode="tiles", n_tiles == -1 ----
    def _enumerate_full_tiles(self, img, cell, bound, inst):
        """
        Return list of (img_t, cell_t, bound_t, inst_t, tile_meta)
        covering the whole image with stride = target - tile_overlap.
        """
        H, W = cell.shape
        th = self.target
        if H <= th or W <= th:
            # same fallback as above
            img_o, cell_o, bound_o, inst_o, meta = self._mode_pad_resize(img, cell, bound, inst)
            return [(img_o, cell_o, bound_o, inst_o, meta)]

        stride = max(1, th - self.tile_overlap)
        tiles = []
        for y0 in range(0, H - th + 1, stride):
            for x0 in range(0, W - th + 1, stride):
                img_t   = crop_rect(img,  y0, x0, th, th)
                cell_t  = crop_rect(cell, y0, x0, th, th)
                bound_t = crop_rect(bound,y0, x0, th, th)
                inst_t  = crop_rect(inst, y0, x0, th, th)
                meta_t = dict(mode="tiles", tile_xy=(int(y0), int(x0)), tile_hw=(th, th))
                tiles.append((img_t.astype(np.float32),
                              cell_t.astype(np.float32),
                              bound_t.astype(np.float32),
                              inst_t.astype(np.int32),
                              meta_t))
        return tiles

    def __getitem__(self, idx):
        # per-scene RNG
        rng = np.random.default_rng(int(self.base_rng.integers(0, 2**31 - 1)) ^ int(idx))

        # build sim kwargs strictly from configs
        sim_kwargs = self.scene_cfg.sample_kwargs(rng, camera=self.camera_cfg)
        sim_kwargs.setdefault("return_targets", True)

        # simulate
        img, meta, targets = simulate_image(**sim_kwargs)
        cell  = targets["cell_mask"].astype(np.float32)
        bound = targets["boundary"].astype(np.float32)
        inst  = targets["instance_labels"].astype(np.int32)

        # non-tile modes stay simple
        if self.mode == "pad_resize":
            img_o, cell_o, bound_o, inst_o, mode_meta = self._mode_pad_resize(img, cell, bound, inst)
            tiles_raw = [(img_o, cell_o, bound_o, inst_o, mode_meta)]
            full_meta = meta
        elif self.mode == "crop_well_resize":
            img_o, cell_o, bound_o, inst_o, mode_meta = self._mode_crop_well_resize(img, cell, bound, inst, meta)
            tiles_raw = [(img_o, cell_o, bound_o, inst_o, mode_meta)]
            full_meta = meta
        elif self.mode == "fullres":
            img_o, cell_o, bound_o, inst_o, mode_meta = self._mode_fullres(img, cell, bound, inst, meta)
            tiles_raw = [(img_o, cell_o, bound_o, inst_o, mode_meta)]
            full_meta = meta
        else:
            # mode == "tiles"
            if self.n_tiles == -1:
                # full coverage -> we get a list of (img_t, cell_t, bound_t, inst_t, mode_meta)
                tiles_raw = self._enumerate_full_tiles(img, cell, bound, inst)
                # now attach *tile-specific* sim meta to each tile
                tiles_with_meta = []
                for (img_t, cell_t, bound_t, inst_t, mode_meta) in tiles_raw:
                    y0, x0 = mode_meta["tile_xy"]
                    th, tw = mode_meta["tile_hw"]
                    meta_t = crop_sim_meta_to_tile(meta, y0, x0, th, tw)
                    # store both: per-tile (cropped) and full (original)
                    mode_meta_full = {
                        **mode_meta,
                        "sim_meta": meta_t,
                        "full_meta": meta,
                    }
                    tiles_with_meta.append((img_t, cell_t, bound_t, inst_t, mode_meta_full))
                tiles_raw = tiles_with_meta
                # full_meta stays the original simulator meta (full image)
                full_meta = meta

            elif self.n_tiles > 0:
                # pick N random tiles
                tiles_raw = []
                for _ in range(self.n_tiles):
                    img_o, cell_o, bound_o, inst_o, mode_meta = self._mode_tiles_single(img, cell, bound, inst, rng)
                    y0, x0 = mode_meta["tile_xy"]
                    th, tw = mode_meta["tile_hw"]
                    meta_t = crop_sim_meta_to_tile(meta, y0, x0, th, tw)
                    tiles_raw.append(
                        (
                            img_o,
                            cell_o,
                            bound_o,
                            inst_o,
                            {
                                **mode_meta,
                                "sim_meta": meta_t,
                                "full_meta": meta,
                            },
                        )
                    )
                full_meta = meta

            else:
                raise ValueError("Choose n_tiles to be a pos. integer or -1")

        # ---- now build tensors for ALL tiles in this sample ----
        imgs_out = []
        tgts_out = []
        inst_out = []
        tiles_meta_out = []

        for (img_o, cell_o, bound_o, inst_o, mode_meta) in tiles_raw:
            # targets (4 heads)
            bound_soft = make_soft_boundary_from_instances(
                inst_o,
                ring_width=max(1, self.boundary_ring_width),
                soft_band=max(1, self.boundary_soft_band),
                sigma=self.boundary_sigma,
            ).astype(np.float32)

            # centers: use meta centers if available and map through mode transform; otherwise derive
            # get tile-level sim meta if present
            tile_sim_meta = full_meta
            if isinstance(mode_meta, dict) and "sim_meta" in mode_meta:
                tile_sim_meta = mode_meta["sim_meta"]

            if "centers" in tile_sim_meta and isinstance(tile_sim_meta["centers"], (list, tuple)) and len(tile_sim_meta["centers"]) > 0:
                if mode_meta["mode"] == "pad_resize":
                    sq, (pt, pl), S = pad_to_square(img, pad_val=0.0)
                    scale = self.target / float(S)
                    centers = []
                    for (y, x) in tile_sim_meta["centers"]:
                        yy = int(round((y + pt) * scale))
                        xx = int(round((x + pl) * scale))
                        if 0 <= yy < self.target and 0 <= xx < self.target:
                            centers.append((yy, xx))
                    center_shape = (self.target, self.target)

                elif mode_meta["mode"] == "crop_well_resize":
                    y0, y1, x0, x1 = mode_meta["crop"]
                    scale = mode_meta["scale"]
                    centers = []
                    for (y, x) in tile_sim_meta["centers"]:
                        yy = int(round((y - y0) * scale))
                        xx = int(round((x - x0) * scale))
                        if 0 <= yy < self.target and 0 <= xx < self.target:
                            centers.append((yy, xx))
                    center_shape = (self.target, self.target)

                elif mode_meta["mode"] == "fullres":
                    # centers already in full-res coordinates
                    Hc, Wc = inst_o.shape
                    centers = []
                    for (y, x) in tile_sim_meta["centers"]:
                        yy = int(round(y))
                        xx = int(round(x))
                        if 0 <= yy < Hc and 0 <= xx < Wc:
                            centers.append((yy, xx))
                    center_shape = (Hc, Wc)

                else:  # tiles
                    centers = []
                    for (y, x) in tile_sim_meta["centers"]:
                        yy = int(round(y))
                        xx = int(round(x))
                        if 0 <= yy < self.target and 0 <= xx < self.target:
                            centers.append((yy, xx))
                    center_shape = (self.target, self.target)

                center_stem = make_center_stem_from_centers(centers, center_shape)

            else:
                lbl = inst_o
                centers = []
                for k in range(1, int(lbl.max())+1):
                    ys, xs = np.where(lbl == k)
                    if ys.size == 0:
                        continue
                    cy = int(np.mean(ys))
                    cx = int(np.mean(xs))
                    centers.append((cy, cx))
                center_stem = make_center_stem_from_centers(centers, (lbl.shape[0], lbl.shape[1]))

            center_heat = make_center_heatmap(center_stem, sigma=self.center_sigma)
            energy = make_energy_from_instances(inst_o)

            tgt = np.stack([cell_o, bound_soft, center_heat, energy], axis=0).astype(np.float32)

            # optional transforms (note: per tile)
            if self.transforms is not None:
                out = self.transforms(image=img_o, masks=[cell_o, bound_soft, center_heat, energy])
                img_o = out["image"]
                cell_o, bound_soft, center_heat, energy = out["masks"]
                tgt = np.stack([cell_o, bound_soft, center_heat, energy], axis=0).astype(np.float32)

            # tensors
            img_c = np.transpose(img_o, (2, 0, 1)).astype(np.float32)   # [3,H,W]
            inst_o, _, _ = relabel_sequential(inst_o.astype(np.int32))

            imgs_out.append(img_c)
            tgts_out.append(tgt)
            inst_out.append(inst_o)
            tiles_meta_out.append(mode_meta)

        # stack over tile dim
        imgs_t = torch.from_numpy(np.stack(imgs_out, axis=0).astype(np.float32))      # [T,3,S,S]
        tgts_t = torch.from_numpy(np.stack(tgts_out, axis=0).astype(np.float32))      # [T,4,S,S]
        inst_t = torch.from_numpy(np.stack(inst_out, axis=0).astype(np.int32))        # [T,S,S]

        extras = {
            "instance_labels": inst_t,
            "meta": {
                "full": full_meta,
                "tiles": tiles_meta_out,
                "sim_kwargs": {k: v for k, v in sim_kwargs.items() if k != "seed"},
            },
        }
        return imgs_t, tgts_t, extras

class ExternalCellsTilesDataset(Dataset):
    """
    External images + 8-bit masks → same output shape as SimCellsDataset(mode="tiles", n_tiles=-1)
    Adds support for per-image COM+labels CSV ({image}_data.csv) and passes
    them to per-tile meta (cropped/shifted).
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

        self.pairs = find_pairs_strict(root_dir)
        if not self.pairs:
            raise RuntimeError("no (img, img_mask) pairs found in external folder")

    def __len__(self):
        return len(self.pairs)

    def _read_image_and_mask(self, img_path: str, mask_path: str):
        """
        Returns:
          img_rgb_f32 : float32 [H, W, 3] in [0,1]
          msk_bin     : uint8   [H, W] with {0,1}
          info        : dict with bit-depth details for auditing
        """

        def _detect_bitdepth_u16(a: np.ndarray, p: float = 99.9):
            # a: uint16 HxW or HxWxC
            vmax = int(a.max())
            if vmax == 0:
                return 16, (1 << 16) - 1, False  # b, white, shifted

            # try left-shifted patterns (lower bits all zero), highest first
            for b in (14, 12, 10, 8):
                shift = 16 - b
                low_mask = (1 << shift) - 1
                if (a & low_mask).max() == 0:
                    white_shifted = ((1 << b) - 1) << shift
                    if vmax <= white_shifted:
                        return b, white_shifted, True

            # robust estimate from percentile
            sample = float(np.percentile(a, p))
            sample = max(1.0, sample)
            est_bits = int(math.ceil(math.log2(sample + 1.0)))
            est_bits = min(16, max(2, est_bits))
            allowed = (8, 10, 12, 14, 16)
            b = min(allowed, key=lambda k: abs(k - est_bits))
            white = (1 << b) - 1
            if vmax > white:
                return 16, (1 << 16) - 1, False
            return b, white, False

        # --- read image (keep native depth) ---
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        shape_in = tuple(img.shape)
        dtype_in = str(img.dtype)

        # ensure HxWxC
        if img.ndim == 2:
            img = img[:, :, None]

        # convert to RGB 3-ch
        if img.shape[2] >= 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 2:
            c0 = img[:, :, 0:1]
            c1 = img[:, :, 1:2]
            img = np.concatenate([c0, c1, c0], axis=-1)
        else:
            img = np.repeat(img, 3, axis=-1)

        # scale to float32 [0,1] with bit-depth sanity
        info = {
            "dtype_in": dtype_in,
            "shape_in": shape_in,
            "bit_depth": None,
            "white_level": None,
            "shifted": False,
        }

        if img.dtype == np.uint8:
            info["bit_depth"] = 8
            info["white_level"] = 255
            img_f32 = img.astype(np.float32) / 255.0

        elif img.dtype == np.uint16:
            b, white, shifted = _detect_bitdepth_u16(img)
            info.update({"bit_depth": b, "white_level": int(white), "shifted": bool(shifted)})

            # if left-shifted, undo shift before scaling
            if shifted and b < 16:
                shift = 16 - b
                img = (img >> shift).astype(np.uint16)
                white = (1 << b) - 1

            img_f32 = img.astype(np.float32) / float(white)
            img_f32 = np.clip(img_f32, 0.0, 1.0)

        else:
            # any other numeric type → clip to [0,1]
            img_f32 = np.clip(img.astype(np.float32), 0.0, 1.0)
            info.update({"bit_depth": 32, "white_level": 1, "shifted": False})

        # --- read mask (binary uint8) ---
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # accept both 0/255 and 0/1 sources
        if m.max() <= 1:
            msk_bin = (m >= 1).astype(np.uint8)
        else:
            msk_bin = (m >= 255).astype(np.uint8)

        return img_f32, msk_bin, info

    def _enumerate_full_tiles(self, H: int, W: int):
        th = self.target
        if H <= th or W <= th:
            return [(0, min(th, H), 0, min(th, W))]
        stride = max(1, th - self.tile_overlap)
        coords = []
        for y0 in range(0, H - th + 1, stride):
            for x0 in range(0, W - th + 1, stride):
                y1 = y0 + th
                x1 = x0 + th
                coords.append((y0, y1, x0, x1))
        return coords

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img, msk, read_info = self._read_image_and_mask(img_path, mask_path)
        H, W = msk.shape

        # heal watershed gaps if wanted
        msk_healed = heal_watershed_gaps(msk, radius=self.heal_radius)

        # instances from mask
        inst_full = sklabel(msk_healed, connectivity=1).astype(np.int32)
        inst_full, _, _ = relabel_sequential(inst_full)

        # try to load COM+labels CSV
        csv_path = guess_data_csv_path(img_path, mask_path)
        centers_csv, labels_csv = load_com_labels_csv(csv_path)

        # centers: prefer CSV; fallback to instance centroids
        if centers_csv:
            centers_full = centers_csv
            labels_full = labels_csv
        else:
            centers_full = []
            max_id = int(inst_full.max())
            for k in range(1, max_id + 1):
                ys, xs = np.where(inst_full == k)
                if ys.size == 0:
                    continue
                cy = int(np.mean(ys))
                cx = int(np.mean(xs))
                centers_full.append((cy, cx))
            # If no CSV, labels unknown → map all to 0
            labels_full = [0 for _ in range(len(centers_full))]

        # full meta (include COM+labels so tiles can crop them)
        full_meta = {
            "src_path": img_path,
            "mask_path": mask_path,
            "data_csv": csv_path if os.path.exists(csv_path) else "",
            "H_in": int(H),
            "W_in": int(W),
            "n_cells": int(inst_full.max()),
            "centers": [(int(y), int(x)) for (y, x) in centers_full],
            "labels": [int(v) for v in labels_full],  # 1/0/-1
            "read_info": read_info,
        }

        n_cells_full = len(centers_full)
        if n_cells_full > 0 and len(labels_full) == n_cells_full:
            frac_positive_full = float(np.mean([1 if v == 1 else 0 for v in labels_full]))
        else:
            frac_positive_full = 0.0
        full_meta["frac_positive"] = frac_positive_full

        # enumerate tiles like SimCellsDataset._enumerate_full_tiles
        th = self.target
        if H <= th or W <= th:
            tile_coords = [(0, 0)]
        else:
            stride = max(1, th - self.tile_overlap)
            tile_coords = []
            for y0 in range(0, H - th + 1, stride):
                for x0 in range(0, W - th + 1, stride):
                    tile_coords.append((int(y0), int(x0)))

        imgs_out = []
        tgts_out = []
        inst_out = []
        tiles_meta_out = []

        for (y0, x0) in tile_coords:
            if H <= th or W <= th:
                # fallback: pad+resize whole image to target
                img_sq, _, _ = pad_to_square(img, pad_val=0.0)
                inst_sq, _, _ = pad_to_square(inst_full, pad_val=0)

                img_t = resize_map(img_sq, th, mode="image")   # [th,th,3]
                inst_t = resize_map(inst_sq, th, mode="label") # [th,th]
            else:
                # standard tile crop
                img_t = crop_rect(img, y0, x0, th, th)         # [th,th,3]
                inst_t = crop_rect(inst_full, y0, x0, th, th)  # [th,th]

            # relabel per tile
            inst_t, _, _ = relabel_sequential(inst_t.astype(np.int32))

            # tile-level external meta (centers/labels shifted into tile coords)
            tile_sim_meta = crop_external_meta_to_tile(
                full_meta,
                y0, x0,
                th if H > th and W > th else th,  # h, w (both = target)
                th if H > th and W > th else th,
            )

            # compute cell / boundary / center / energy for this tile
            cell_t = (inst_t > 0).astype(np.float32)
            bound_soft = make_soft_boundary_from_instances(
                inst_t,
                ring_width=max(1, self.boundary_ring_width),
                soft_band=max(1, self.boundary_soft_band),
                sigma=self.boundary_sigma,
            ).astype(np.float32)

            # centers: use meta centers if present, else from instances
            centers = []
            if "centers" in tile_sim_meta and isinstance(tile_sim_meta["centers"], (list, tuple)) and len(tile_sim_meta["centers"]) > 0:
                for (y, x) in tile_sim_meta["centers"]:
                    yy = int(round(y))
                    xx = int(round(x))
                    if 0 <= yy < th and 0 <= xx < th:
                        centers.append((yy, xx))
            else:
                lbl = inst_t
                for k in range(1, int(lbl.max()) + 1):
                    ys, xs = np.where(lbl == k)
                    if ys.size == 0:
                        continue
                    cy = int(np.mean(ys))
                    cx = int(np.mean(xs))
                    centers.append((cy, cx))

            center_stem = make_center_stem_from_centers(centers, (th, th))
            center_heat = make_center_heatmap(center_stem, sigma=self.center_sigma)
            energy = make_energy_from_instances(inst_t)

            tgt_t = np.stack([cell_t, bound_soft, center_heat, energy], axis=0).astype(np.float32)

            # optional transforms (Albumentations-style)
            if self.transforms is not None:
                out = self.transforms(
                    image=img_t,
                    masks=[cell_t, bound_soft, center_heat, energy],
                )
                img_t = out["image"]
                cell_t, bound_soft, center_heat, energy = out["masks"]
                tgt_t = np.stack([cell_t, bound_soft, center_heat, energy], axis=0).astype(np.float32)

            # final tensors
            img_chw = np.transpose(img_t.astype(np.float32), (2, 0, 1))   # [3,th,th]
            imgs_out.append(img_chw)
            tgts_out.append(tgt_t)
            inst_out.append(inst_t.astype(np.int32))

            # tile meta, *same structure* as SimCellsDataset tiles
            mode_meta = {
                "mode": "tiles",
                "tile_xy": (int(y0), int(x0)),
                "tile_hw": (int(th), int(th)),
                "sim_meta": tile_sim_meta,
                "full_meta": full_meta,
            }
            tiles_meta_out.append(mode_meta)

        # stack over tile dim
        imgs_t = torch.from_numpy(np.stack(imgs_out, axis=0).astype(np.float32))  # [T,3,S,S]
        tgts_t = torch.from_numpy(np.stack(tgts_out, axis=0).astype(np.float32))  # [T,4,S,S]
        inst_t = torch.from_numpy(np.stack(inst_out, axis=0).astype(np.int32))    # [T,S,S]

        extras = {
            "instance_labels": inst_t,
            "meta": {
                "full": full_meta,
                "tiles": tiles_meta_out,
                "sim_kwargs": None,  # external data has no sim kwargs
            },
        }
        return imgs_t, tgts_t, extras
