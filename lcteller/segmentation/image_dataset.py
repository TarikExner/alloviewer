import math
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import json
import h5py
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, exposure
from skimage.segmentation import relabel_sequential

from . import simulate_image

from typing import Optional, Sequence, Union


def _resize_map(x, side, mode="image"):
    H, W = x.shape[:2]
    down = (side < H) or (side < W)
    if mode == "image":
        interp = cv2.INTER_AREA if down else cv2.INTER_CUBIC
        y = cv2.resize(np.ascontiguousarray(x.astype(np.float32, copy=False)),
                       (side, side), interpolation=interp)
        return y.astype(np.float32, copy=False)
    elif mode == "binary":
        y = cv2.resize(np.ascontiguousarray(x.astype(np.uint8, copy=False)),
                       (side, side), interpolation=cv2.INTER_NEAREST)
        return y.astype(np.float32, copy=False)
    elif mode == "label":
        xin = np.ascontiguousarray(x.astype(np.float32, copy=False))
        y = cv2.resize(xin, (side, side), interpolation=cv2.INTER_NEAREST)
        return y.astype(np.int32, copy=False)
    else:
        raise ValueError(mode)

def _pad_to_square(arr, pad_val=0):
    H, W = arr.shape[:2]
    S = max(H, W)
    dy = S - H
    dx = S - W
    top = dy // 2
    bottom = dy - top
    left = dx // 2
    right = dx - left
    if arr.ndim == 3:
        out = np.pad(arr, ((top, bottom), (left, right), (0, 0)),
                     mode="constant", constant_values=((pad_val, pad_val), (pad_val, pad_val), (0, 0)))
    else:
        out = np.pad(arr, ((top, bottom), (left, right)),
                     mode="constant", constant_values=pad_val)
    return out, (top, left), S

def _crop_rect(arr, y0, x0, h, w):
    return arr[y0:y0+h, x0:x0+w, ...] if arr.ndim == 3 else arr[y0:y0+h, x0:x0+w]

def _estimate_well_mask(img, blur_sigma=3.0, well_is_brighter="auto"):
    g = img if img.ndim == 2 else (0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2])
    g = ndi.gaussian_filter(g.astype(np.float32), blur_sigma)
    g = exposure.rescale_intensity(g, in_range='image', out_range=(0, 1))
    thr = filters.threshold_otsu(g)
    m1 = g > thr      # brighter region
    m2 = g < thr      # darker region
    if well_is_brighter == "auto":
        m = m1 if m1.sum() >= m2.sum() else m2
    elif well_is_brighter:
        m = m1
    else:
        m = m2
    m = morphology.remove_small_objects(m, 500)
    m = morphology.remove_small_holes(m, 500)
    if m.sum() == 0:
        H, W = g.shape
        return np.zeros_like(m, dtype=bool), (H/2, W/2), min(H, W)/2 * 0.9
    lbl = measure.label(m)
    props = measure.regionprops(lbl)
    props.sort(key=lambda p: p.area, reverse=True)
    p = props[0]
    cy, cx = p.centroid
    r = math.sqrt(p.area/np.pi)
    return (lbl == p.label), (cy, cx), r

def _square_crop_from_center_radius(mask_shape, center, radius, pad=8):
    H, W = mask_shape
    cy, cx = center
    half = int(math.ceil(radius + pad))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    # make square
    h = y1 - y0
    w = x1 - x0
    if h > w:
        d = h - w
        x0 -= d//2
        x1 += d - d//2
    elif w > h:
        d = w - h
        y0 -= d//2
        y1 += d - d//2
    # clip
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)
    return y0, y1, x0, x1

def _compute_inner_boundary(inst_np: np.ndarray) -> np.ndarray:
    a = inst_np
    H, W = a.shape
    up    = (a != np.roll(a, -1, axis=0))
    down  = (a != np.roll(a,  1, axis=0))
    left  = (a != np.roll(a, -1, axis=1))
    right = (a != np.roll(a,  1, axis=1))
    b = (up | down | left | right)
    b &= (a > 0)
    b[H-1,:] &= (a[H-1,:] != 0)
    b[0,:]   &= (a[0,:]   != 0)
    b[:,W-1] &= (a[:,W-1] != 0)
    b[:,0]   &= (a[:,0]   != 0)
    return b.astype(np.uint8)

def _make_soft_boundary_from_instances(inst: np.ndarray,
                                       ring_width: int = 1,
                                       soft_band: int = 2,
                                       sigma: float = 1.0) -> np.ndarray:
    ring = _compute_inner_boundary(inst).astype(bool)
    if ring_width > 1:
        rad = max(1, int(ring_width // 2))
        ring = ndi.binary_dilation(ring, structure=ndi.generate_binary_structure(2,1), iterations=rad)
    cell = (inst > 0)
    if soft_band > 0:
        not_ring = ~ring
        dist = ndi.distance_transform_edt(not_ring)
        dist[~cell] = np.inf
        soft = np.exp(-(dist**2) / (2.0 * (sigma**2)))
        soft[dist > float(soft_band)] = 0.0
        soft[~np.isfinite(soft)] = 0.0
        m = soft.max()
        if m > 0:
            soft = soft / m
        return soft.astype(np.float32)
    else:
        return ring.astype(np.float32)

def _make_center_stem_from_centers(centers, shape):
    H, W = shape
    stem = np.zeros((H, W), dtype=np.float32)
    for (y, x) in centers or []:
        if 0 <= y < H and 0 <= x < W:
            stem[int(y), int(x)] = 1.0
    return stem

def _make_center_heatmap(stem, sigma: Union[int, float] = 1.0):
    heat = stem.astype(np.float32)
    if sigma and sigma > 0:
        heat = ndi.gaussian_filter(heat, float(sigma))
        m = float(heat.max())
        if m > 0:
            heat /= m
    return heat.astype(np.float32)

def _make_energy_from_instances(instances):
    cell = (instances > 0)
    dist = ndi.distance_transform_edt(cell).astype(np.float32)
    if cell.any():
        dmax = float(dist[cell].max())
        if dmax > 0:
            dist /= dmax
    dist[~cell] = 0.0
    return dist.astype(np.float32)

def _crop_sim_meta_to_tile(meta, y0, x0, h, w):
    """
    Take simulator meta (full image) and make it consistent with a tile
    that starts at (y0, x0) and has size (h, w).
    """
    # shallow copy so we don't mutate caller's dict
    new_meta = dict(meta)

    centers = meta.get("centers", [])
    labels  = meta.get("labels", [])
    sigmas  = meta.get("final_sigmas", None)

    kept_centers = []
    kept_labels  = []
    kept_sigmas  = []

    for i, c in enumerate(centers):
        cy, cx = c
        ny = cy - y0
        nx = cx - x0
        if 0 <= ny < h and 0 <= nx < w:
            kept_centers.append((int(ny), int(nx)))
            if isinstance(labels, (list, tuple, np.ndarray)) and i < len(labels):
                kept_labels.append(int(labels[i]))
            # sigmas can be np.ndarray
            if sigmas is not None and i < len(sigmas):
                kept_sigmas.append(float(sigmas[i]))

    # update centers
    new_meta["centers"] = kept_centers

    # update labels (keep type: list of int)
    if isinstance(labels, np.ndarray):
        new_meta["labels"] = np.array(kept_labels, dtype=labels.dtype)
    else:
        new_meta["labels"] = kept_labels

    # update final_sigmas
    if sigmas is not None:
        new_meta["final_sigmas"] = np.array(kept_sigmas, dtype=np.float32)

    # counts
    n_cells_tile = len(kept_centers)
    new_meta["n_cells"] = int(n_cells_tile)

    # frac_positive
    if n_cells_tile > 0 and len(kept_labels) == n_cells_tile:
        new_meta["frac_positive"] = float(np.mean(kept_labels))
    else:
        new_meta["frac_positive"] = 0.0

    # well center shift (can be outside tile, that's fine)
    if "well_center" in meta and meta["well_center"] is not None:
        wy, wx = meta["well_center"]
        new_meta["well_center"] = (float(wy - y0), float(wx - x0))

    # radius stays as is

    # fix params (the captured simulator args)
    params = meta.get("params", None)
    if isinstance(params, dict):
        new_params = dict(params)
        # sizes
        new_params["H"] = int(h)
        new_params["W"] = int(w)
        # counts
        if "n_cells" in new_params:
            new_params["n_cells"] = int(n_cells_tile)
        if "frac_positive" in new_params:
            new_params["frac_positive"] = float(new_meta["frac_positive"])
        if "well_center" in new_params and "well_center" in new_meta:
            new_params["well_center"] = new_meta["well_center"]
        # radius_px stays
        new_meta["params"] = new_params

    return new_meta

# -----------------------------
# main dataset
# -----------------------------

class SimCellsDataset(Dataset):
    """
    Modes:
      - pad_resize: pad to square, then resize to target
      - crop_well_resize: square-crop around well, then resize
      - tiles: return a random target×target tile
               (or many tiles if n_tiles>0, or all tiles if n_tiles==-1)

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
        mode="pad_resize",            # "pad_resize" | "crop_well_resize" | "tiles"
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
        assert self.mode in ("pad_resize", "crop_well_resize", "tiles")

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
        sq, (pad_top, pad_left), S = _pad_to_square(img, pad_val=0.0)
        cell_sq, _, _ = _pad_to_square(cell, pad_val=0.0)
        bound_sq, _, _ = _pad_to_square(bound, pad_val=0.0)
        inst_sq,  _, _ = _pad_to_square(inst,  pad_val=0)

        img_o   = _resize_map(sq,       self.target, "image")
        cell_o  = _resize_map(cell_sq,  self.target, "binary")
        bound_o = _resize_map(bound_sq, self.target, "binary")
        inst_o  = _resize_map(inst_sq,  self.target, "label")

        meta = dict(mode="pad_resize", pad_top=pad_top, pad_left=pad_left, S_in=S, scale=self.target/float(S))
        return img_o, cell_o, bound_o, inst_o, meta

    def _mode_crop_well_resize(self, img, cell, bound, inst, sim_meta):
        cycx = sim_meta.get("well_center", None)
        R = sim_meta.get("radius_px", None)
        if (cycx is None) or (R is None):
            _, center, radius = _estimate_well_mask(img, blur_sigma=3.0, well_is_brighter=self.well_is_brighter)
        else:
            center = (float(cycx[0]), float(cycx[1]))
            radius = float(R)

        y0, y1, x0, x1 = _square_crop_from_center_radius(cell.shape, center, radius, pad=8)
        img_c   = _crop_rect(img,  y0, x0, y1-y0, x1-x0)
        cell_c  = _crop_rect(cell, y0, x0, y1-y0, x1-x0)
        bound_c = _crop_rect(bound,y0, x0, y1-y0, x1-x0)
        inst_c  = _crop_rect(inst, y0, x0, y1-y0, x1-x0)

        img_o   = _resize_map(img_c,  self.target, "image")
        cell_o  = _resize_map(cell_c, self.target, "binary")
        bound_o = _resize_map(bound_c,self.target, "binary")
        inst_o  = _resize_map(inst_c, self.target, "label")

        scale = self.target / max(1.0, float(max(y1-y0, x1-x0)))
        meta = dict(mode="crop_well_resize",
                    crop=(int(y0), int(y1), int(x0), int(x1)),
                    scale=scale,
                    well_center=(float(center[0]), float(center[1])),
                    well_radius=float(radius))
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

        img_t   = _crop_rect(img,  y0, x0, th, th)
        cell_t  = _crop_rect(cell, y0, x0, th, th)
        bound_t = _crop_rect(bound,y0, x0, th, th)
        inst_t  = _crop_rect(inst, y0, x0, th, th)

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
                img_t   = _crop_rect(img,  y0, x0, th, th)
                cell_t  = _crop_rect(cell, y0, x0, th, th)
                bound_t = _crop_rect(bound,y0, x0, th, th)
                inst_t  = _crop_rect(inst, y0, x0, th, th)
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
                    meta_t = _crop_sim_meta_to_tile(meta, y0, x0, th, tw)
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
                    meta_t = _crop_sim_meta_to_tile(meta, y0, x0, th, tw)
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
            bound_soft = _make_soft_boundary_from_instances(
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
                    sq, (pt, pl), S = _pad_to_square(img, pad_val=0.0)
                    scale = self.target / float(S)
                    centers = []
                    for (y, x) in tile_sim_meta["centers"]:
                        yy = int(round((y + pt) * scale))
                        xx = int(round((x + pl) * scale))
                        if 0 <= yy < self.target and 0 <= xx < self.target:
                            centers.append((yy, xx))
                elif mode_meta["mode"] == "crop_well_resize":
                    y0, y1, x0, x1 = mode_meta["crop"]
                    scale = mode_meta["scale"]
                    centers = []
                    for (y, x) in tile_sim_meta["centers"]:
                        yy = int(round((y - y0) * scale))
                        xx = int(round((x - x0) * scale))
                        if 0 <= yy < self.target and 0 <= xx < self.target:
                            centers.append((yy, xx))
                else:  # tiles
                    centers = []
                    for (y, x) in tile_sim_meta["centers"]:
                        yy = int(round(y))
                        xx = int(round(x))
                        if 0 <= yy < self.target and 0 <= xx < self.target:
                            centers.append((yy, xx))
                center_stem = _make_center_stem_from_centers(centers, (self.target, self.target))
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
                center_stem = _make_center_stem_from_centers(centers, (lbl.shape[0], lbl.shape[1]))

            center_heat = _make_center_heatmap(center_stem, sigma=self.center_sigma)
            energy = _make_energy_from_instances(inst_o)

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

class DiskSimCellsDataset(Dataset):
    """
    Read-only dataset for HDF5 files produced by create_dataset_h5/export_to_prealloc_h5_safe.

    Expects datasets:
      /imgs: float32 [N, 1, 3, S, S]
      /tgts: float32 [N, 1, C, S, S]
      /inst: int32   [N, 1, S, S]
      /meta: vlen JSON strings (one per sample)

    Returns (per __getitem__):
      img_t:  torch.float32 [1, 3, S, S]
      tgt_t:  torch.float32 [1, C, S, S]
      extras: {
        "instance_labels": torch.int32 [1, S, S],
        "meta": dict
      }
    """
    def __init__(self, h5_path: str, indices: Optional[Sequence[int]] = None):
        super().__init__()
        self.h5_path = str(h5_path)
        self._h5: Optional[h5py.File] = None
        self._imgs = self._tgts = self._inst = self._meta = None

        # lightweight probe (no persistent handle) to get shapes, attrs
        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as f:
            n = int(f["imgs"].shape[0])
            self._N = n
            self._T = int(f["imgs"].shape[1])  # should be 1
            self._C_img = int(f["imgs"].shape[2])
            self._S = int(f["imgs"].shape[3])
            self._C_tgt = int(f["tgts"].shape[2])
            # optional sanity
            assert "inst" in f and "meta" in f, "Missing datasets 'inst' or 'meta' in HDF5."

        if indices is None:
            self._idx = np.arange(self._N, dtype=np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("indices must be 1D")
            if (idx < 0).any() or (idx >= self._N).any():
                raise ValueError("indices out of range")
            self._idx = idx

    def __len__(self) -> int:
        return int(self._idx.shape[0])

    def _ensure_open(self):
        # open per worker/process lazily
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._imgs = self._h5["imgs"]
            self._tgts = self._h5["tgts"]
            self._inst = self._h5["inst"]
            self._meta = self._h5["meta"]

    def __getitem__(self, i: int):
        self._ensure_open()
        k = int(self._idx[i])

        # read numpy views; h5py returns arrays on slicing
        img = self._imgs[k]   # float32 [1, 3, S, S]
        tgt = self._tgts[k]   # float32 [1, C, S, S]
        inst = self._inst[k]  # int32   [1, S, S]
        meta_json = self._meta[k]

        # h5py vlen str returns bytes or str depending on build
        if isinstance(meta_json, bytes):
            meta = json.loads(meta_json.decode("utf-8"))
        else:
            meta = json.loads(meta_json)

        # convert to torch with exact dtypes
        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32))      # [1, 3, S, S]
        tgt_t = torch.from_numpy(np.asarray(tgt, dtype=np.float32))      # [1, C, S, S]
        inst_t = torch.from_numpy(np.asarray(inst, dtype=np.int32))      # [1, S, S]

        extras = {"instance_labels": inst_t, "meta": meta}
        return img_t, tgt_t, extras

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

