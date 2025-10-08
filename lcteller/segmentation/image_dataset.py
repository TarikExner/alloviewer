import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage as ndi
import cv2

from . import simulate_image


def _sample_camera_params(rng, src_side_range=(640, 1024), aspect_ratio_range=(0.6, 1.6), content_scale_range=(0.6, 0.95)):
    S_src = int(rng.integers(int(src_side_range[0]), int(src_side_range[1]) + 1))
    ar = float(rng.uniform(aspect_ratio_range[0], aspect_ratio_range[1]))
    s  = float(rng.uniform(content_scale_range[0], content_scale_range[1]))
    return S_src, ar, s

def _resize_map(x, size, mode="image"):
    """Resize to (size,size). mode: image|binary|label"""
    if mode == "image":
        return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    elif mode == "binary":
        y = cv2.resize(x.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        return y.astype(np.float32)
    elif mode == "label":
        y = cv2.resize(x.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST)
        return y.astype(np.int32)
    else:
        raise ValueError(mode)

def _crop_rect(arr, y0, x0, h, w):
    return arr[y0:y0+h, x0:x0+w, ...] if arr.ndim == 3 else arr[y0:y0+h, x0:x0+w]

def _pad_to_square(arr, target_side, pad_color=0.0):
    """Pad HxW or HxWxC to target_side with a constant pad_color.
    Returns (padded_array, (pad_top, pad_left)).
    """
    H, W = arr.shape[:2]
    dy = target_side - H
    dx = target_side - W
    if dy < 0 or dx < 0:
        raise ValueError(f"target_side {target_side} is smaller than input {(H, W)}")

    top = dy // 2
    bottom = dy - top
    left = dx // 2
    right = dx - left

    if arr.ndim == 3:
        # Need a pair per axis for constant_values
        const = (
            (float(pad_color), float(pad_color)),  # along H (top, bottom)
            (float(pad_color), float(pad_color)),  # along W (left, right)
            (0, 0),                                # channels: no pad
        )
        out = np.pad(
            arr,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=const,
        )
    else:
        out = np.pad(
            arr,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=float(pad_color),
        )

    return out, (top, left)

def _transform_centers_for_crop_pad_scale(centers, crop_y0, crop_x0, pad_top, pad_left, scale, out_side):
    out = []
    for (y, x) in centers:
        yy = (y - crop_y0) + pad_top
        xx = (x - crop_x0) + pad_left
        yy = int(round(yy * scale))
        xx = int(round(xx * scale))
        if 0 <= yy < out_side and 0 <= xx < out_side:
            out.append((yy, xx))
    return out

def _make_center_stem_from_centers(centers, shape):
    """Binary image with 1.0 at each (y,x) center."""
    H, W = shape
    stem = np.zeros((H, W), dtype=np.float32)
    for (y, x) in centers or []:
        if 0 <= y < H and 0 <= x < W:
            stem[y, x] = 1.0
    return stem

def _camera_rect_transform(
    rng,
    img, cell, bound, inst, centers,
    out_side=512,
    src_side_range=(640, 1024),
    aspect_ratio_range=(0.6, 1.6),
    content_scale_range=(0.6, 0.95),
    dark_margin_bias=0.0,
    center_stem=None,   # NEW: carry center stem along the pipeline
):
    """
    1) up/downscale square sim to S_src
    2) crop random rectangle (H_rect x W_rect)
    3) pad to square with black margin (+bias)
    4) resize to out_side x out_side
    """
    # current sim is square N x N
    N = img.shape[0]
    S_src, ar, s = _sample_camera_params(rng, src_side_range, aspect_ratio_range, content_scale_range)

    # 1) resize everything to S_src (square)
    if S_src != N:
        img  = _resize_map(img,  S_src, mode="image")
        cell = _resize_map(cell, S_src, mode="binary")
        bound= _resize_map(bound,S_src, mode="binary")
        inst = _resize_map(inst, S_src, mode="label")
        if center_stem is not None:
            center_stem = _resize_map(center_stem, S_src, mode="binary")
        scale0 = S_src / float(N)
        centers = [(int(round(y*scale0)), int(round(x*scale0))) for (y, x) in centers]
    else:
        scale0 = 1.0  # not used further; kept for completeness

    # 2) pick rectangle
    H_rect = max(8, int(round(S_src * s)))
    W_rect = max(8, int(round(H_rect * ar)))
    W_rect = min(W_rect, S_src)
    H_rect = min(H_rect, S_src)
    y0 = int(rng.integers(0, S_src - H_rect + 1))
    x0 = int(rng.integers(0, S_src - W_rect + 1))

    img_r   = _crop_rect(img,   y0, x0, H_rect, W_rect)
    cell_r  = _crop_rect(cell,  y0, x0, H_rect, W_rect)
    bound_r = _crop_rect(bound, y0, x0, H_rect, W_rect)
    inst_r  = _crop_rect(inst,  y0, x0, H_rect, W_rect)
    center_r = None if center_stem is None else _crop_rect(center_stem, y0, x0, H_rect, W_rect)

    # 3) pad to square
    S_pad = max(H_rect, W_rect)
    img_sq, (pad_top, pad_left) = _pad_to_square(img_r,   S_pad, pad_color=float(dark_margin_bias))
    cell_sq, _                  = _pad_to_square(cell_r,  S_pad, pad_color=0.0)
    bound_sq, _                 = _pad_to_square(bound_r, S_pad, pad_color=0.0)
    inst_sq, _                  = _pad_to_square(inst_r,  S_pad, pad_color=0)
    center_sq = None if center_r is None else _pad_to_square(center_r, S_pad, pad_color=0.0)[0]

    # 4) resize to out_side
    img_out   = _resize_map(img_sq,   out_side, mode="image").astype(np.float32)
    cell_out  = _resize_map(cell_sq,  out_side, mode="binary")
    bound_out = _resize_map(bound_sq, out_side, mode="binary")
    inst_out  = _resize_map(inst_sq,  out_side, mode="label")
    center_out = None if center_sq is None else _resize_map(center_sq, out_side, mode="binary")

    # centers transformed across crop->pad->resize (kept in meta; not used for heatmap now)
    scale = out_side / float(S_pad)
    centers_out = _transform_centers_for_crop_pad_scale(centers, y0, x0, pad_top, pad_left, scale, out_side)

    return img_out, cell_out, bound_out, inst_out, centers_out, center_out  # NOTE extra return

def rand_choice(rng, val_or_range):
    if isinstance(val_or_range, (list, tuple)) and len(val_or_range) == 2:
        lo, hi = val_or_range
        if isinstance(lo, int) and isinstance(hi, int):
            return int(rng.integers(lo, hi + 1))
        return float(rng.uniform(float(lo), float(hi)))
    return val_or_range

def _warp_centers(centers, H, W, flip_x=False, flip_y=False, rot_k=0):
    # kept for reference; no longer used for the heatmap path
    if not centers:
        return []
    pts = np.asarray(centers, dtype=np.int32)
    if flip_x:
        pts[:, 1] = (W - 1) - pts[:, 1]
    if flip_y:
        pts[:, 0] = (H - 1) - pts[:, 0]
    k = int(rot_k) % 4
    if k == 1:
        y_new = pts[:, 1]
        x_new = (W - 1) - pts[:, 0]
        H, W = W, H
        pts = np.stack([y_new, x_new], axis=1)
    elif k == 2:
        y_new = (H - 1) - pts[:, 0]
        x_new = (W - 1) - pts[:, 1]
        pts = np.stack([y_new, x_new], axis=1)
    elif k == 3:
        y_new = (W - 1) - pts[:, 1]
        x_new = pts[:, 0]
        H, W = W, H
        pts = np.stack([y_new, x_new], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)
    return [(int(y), int(x)) for y, x in pts]

def _make_center_heatmap_from_centers(centers, shape, sigma=2.0):
    """Set 1.0 at each center, then blur to get smooth peaks."""
    H, W = shape
    heat = np.zeros((H, W), dtype=np.float32)
    for (y, x) in centers:
        if 0 <= y < H and 0 <= x < W:
            heat[y, x] = 1.0
    if sigma and sigma > 0:
        heat = ndi.gaussian_filter(heat, float(sigma))
        m = float(heat.max())
        if m > 0:
            heat /= m
    return heat.astype(np.float32)

def _make_energy_from_instances(instances):
    """Energy = normalized distance inside cells, 0 outside."""
    cell = (instances > 0)
    dist = ndi.distance_transform_edt(cell).astype(np.float32)
    if cell.any():
        dmax = float(dist[cell].max())
        if dmax > 0:
            dist /= dmax
    dist[~cell] = 0.0
    return dist.astype(np.float32)

class SimCellsDataset(Dataset):
    """
    Returns:
      img_t:  float32 [3,H,W] in [0,1]
      tgt_t:  float32 [C,H,W]  (C=2..4: 0=cell, 1=boundary, 2=center?, 3=energy?)
      extras: dict with:
              - "instance_labels": int32 [H,W]
              - "meta": meta from sim
    """
    def __init__(
        self,
        length=10000,
        tile_size=512,
        n_cells=(10, 400),
        cell_diameter=(4, 28),
        frac_positive=(0, 1),
        blur_sigma=(0, 2.0),
        background_level=(0.0, 0.04),
        color_jitter=(0.0, 0.2),
        photon_level=(1500, 4000),
        boundary_width=2,
        aug_flip=True,
        aug_rot90=True,
        aug_gamma=(0.90, 1.12),
        rng_seed=123,
        sim_fn=None,
        add_center=True,
        add_energy=True,
        center_sigma=2.0,
        random_camera_rect=True,
        cam_src_side_range=(640, 1024),
        cam_aspect_ratio_range=(0.6, 1.6),
        cam_content_scale_range=(0.6, 0.95),
        cam_out_side=512,
        cam_dark_margin_bias=0.0,
    ):
        if sim_fn is None:
            self.sim_fn = simulate_image
        else:
            self.sim_fn = sim_fn
        self.length = int(length)
        self.tile_size = int(tile_size)
        self.n_cells = n_cells
        self.cell_diameter = cell_diameter
        self.frac_positive = frac_positive
        self.blur_sigma = blur_sigma
        self.background_level = background_level
        self.color_jitter = color_jitter
        self.photon_level = photon_level
        self.boundary_width = int(boundary_width)
        self.aug_flip = aug_flip
        self.aug_rot90 = aug_rot90
        self.aug_gamma = aug_gamma
        self.rng = np.random.default_rng(rng_seed)

        self.add_center = bool(add_center)
        self.add_energy = bool(add_energy)
        self.center_sigma = float(center_sigma)

        self.random_camera_rect = bool(random_camera_rect)
        self.cam_src_side_range = cam_src_side_range
        self.cam_aspect_ratio_range = cam_aspect_ratio_range
        self.cam_content_scale_range = cam_content_scale_range
        self.cam_out_side = int(cam_out_side)
        self.cam_dark_margin_bias = float(cam_dark_margin_bias)

    def __len__(self):
        return self.length

    def _apply_aug(self, img, cell, bound, inst):
        H, W = img.shape[0], img.shape[1]
        flip_x = False
        flip_y = False

        if self.aug_flip and self.rng.random() < 0.5:
            img = np.flip(img, axis=1)
            cell = np.flip(cell, axis=1)
            bound = np.flip(bound, axis=1)
            inst = np.flip(inst, axis=1)
            flip_x = True

        if self.aug_flip and self.rng.random() < 0.5:
            img = np.flip(img, axis=0)
            cell = np.flip(cell, axis=0)
            bound = np.flip(bound, axis=0)
            inst = np.flip(inst, axis=0)
            flip_y = True

        k = 0
        if self.aug_rot90:
            k = int(self.rng.integers(0, 4))
            if k:
                img = np.rot90(img,   k, axes=(0, 1))
                cell = np.rot90(cell,  k, axes=(0, 1))
                bound = np.rot90(bound, k, axes=(0, 1))
                inst = np.rot90(inst,  k, axes=(0, 1))

        if isinstance(self.aug_gamma, (list, tuple)) and len(self.aug_gamma) == 2:
            g = float(self.rng.uniform(self.aug_gamma[0], self.aug_gamma[1]))
            img = np.clip(img, 1e-4, 1.0) ** g
            img = np.clip(img, 0.0, 1.0)

        # return the ops taken so we can warp centers the same way (legacy path)
        return img, cell, bound, inst, (flip_x, flip_y, k)

    def _apply_aug_with_center(self, img, cell, bound, inst, center_stem):
        """Same as _apply_aug but moves the center_stem along; gamma only on image."""
        flip_x = False
        flip_y = False

        if self.aug_flip and self.rng.random() < 0.5:
            img = np.flip(img, axis=1)
            cell = np.flip(cell, axis=1)
            bound = np.flip(bound, axis=1)
            inst = np.flip(inst, axis=1)
            center_stem = np.flip(center_stem, axis=1)
            flip_x = True

        if self.aug_flip and self.rng.random() < 0.5:
            img = np.flip(img, axis=0)
            cell = np.flip(cell, axis=0)
            bound = np.flip(bound, axis=0)
            inst = np.flip(inst, axis=0)
            center_stem = np.flip(center_stem, axis=0)
            flip_y = True

        k = 0
        if self.aug_rot90:
            k = int(self.rng.integers(0, 4))
            if k:
                img = np.rot90(img,   k, axes=(0, 1))
                cell = np.rot90(cell,  k, axes=(0, 1))
                bound = np.rot90(bound, k, axes=(0, 1))
                inst = np.rot90(inst,  k, axes=(0, 1))
                center_stem = np.rot90(center_stem, k, axes=(0, 1))

        if isinstance(self.aug_gamma, (list, tuple)) and len(self.aug_gamma) == 2:
            g = float(self.rng.uniform(self.aug_gamma[0], self.aug_gamma[1]))
            img = np.clip(img, 1e-4, 1.0) ** g
            img = np.clip(img, 0.0, 1.0)

        return img, cell, bound, inst, center_stem, (flip_x, flip_y, k)

    def __getitem__(self, idx):
        N   = self.tile_size
        nC  = rand_choice(self.rng, self.n_cells)
        dia = rand_choice(self.rng, self.cell_diameter)
        fp  = rand_choice(self.rng, self.frac_positive)
        blr = rand_choice(self.rng, self.blur_sigma)
        bg  = rand_choice(self.rng, self.background_level)
        cj  = rand_choice(self.rng, self.color_jitter)
        ph  = rand_choice(self.rng, self.photon_level)
        seed = int(self.rng.integers(0, 2**31 - 1))

        assert self.sim_fn is not None

        img, meta, targets = self.sim_fn(
            N=N, n_cells=nC, cell_diameter=dia, frac_positive=fp,
            background_level=bg, color_jitter=cj, blur_sigma=blr,
            photon_level=ph, seed=seed, return_targets=True,
            boundary_width=self.boundary_width,
        )
        cell  = targets["cell_mask"].astype(np.float32)     # H,W
        bound = targets["boundary"].astype(np.float32)      # H,W
        inst  = targets["instance_labels"].astype(np.int32) # H,W
        centers = list(meta.get("centers", []))             # [(y,x), ...]

        # --- paint a binary center stem BEFORE any warp ---
        center_stem = _make_center_stem_from_centers(centers, cell.shape)

        # --- camera-rect step (crop -> pad -> resize), carry the stem too ---
        if self.random_camera_rect:
            img, cell, bound, inst, centers, center_stem = _camera_rect_transform(
                self.rng, img, cell, bound, inst, centers,
                out_side=self.cam_out_side,
                src_side_range=self.cam_src_side_range,
                aspect_ratio_range=self.cam_aspect_ratio_range,
                content_scale_range=self.cam_content_scale_range,
                dark_margin_bias=self.cam_dark_margin_bias,
                center_stem=center_stem,
            )

        # --- flips/rot/gamma on ALL maps (center stem included) ---
        img, cell, bound, inst, center_stem, aug_ops = self._apply_aug_with_center(
            img, cell, bound, inst, center_stem
        )

        # --- build final targets ---
        tgt_maps = [cell, bound]
        if self.add_center:
            center_map = center_stem
            if self.center_sigma and self.center_sigma > 0:
                center_map = ndi.gaussian_filter(center_map.astype(np.float32), float(self.center_sigma))
                m = float(center_map.max())
                if m > 0:
                    center_map = center_map / m
            tgt_maps.append(center_map.astype(np.float32))

        if self.add_energy:
            energy_map = _make_energy_from_instances(inst)   # EDT after all warps -> aligned
            tgt_maps.append(energy_map.astype(np.float32))

        tgt = np.stack(tgt_maps, axis=0).astype(np.float32)  # [C,H,W]

        # pack tensors
        img_c = np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32)
        tgt_c = np.ascontiguousarray(tgt, dtype=np.float32)
        inst_c = np.ascontiguousarray(inst, dtype=np.int32)

        img_t = torch.from_numpy(img_c)
        tgt_t = torch.from_numpy(tgt_c)
        extras = {"instance_labels": torch.from_numpy(inst_c), "meta": meta}
        return img_t, tgt_t, extras

