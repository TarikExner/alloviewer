import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from typing import Literal, List, Optional, Tuple
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import random
from scipy.spatial import cKDTree

from skimage.morphology import skeletonize

from scipy import ndimage as ndi
import cv2

from . import (
    DiskSimCellsDataset,
    build_unet_cpu_small, build_unet_cpu_medium, build_unet_cpu_large
)
from .utils import collate_no_meta

# --------------------- utils ---------------------
def flatten_tiles(img, tgt, extras):
    # nothing to do if already 4D
    if img.ndim == 4 and tgt.ndim == 4:
        return img, tgt, extras

    assert img.ndim == 5, f"img must be 4D or 5D, got {img.shape}"
    assert tgt.ndim == 5, f"tgt must be 4D or 5D, got {tgt.shape}"

    B, T, C, H, W = img.shape
    _, T2, C2, H2, W2 = tgt.shape
    assert T2 == T and H2 == H and W2 == W

    # img: [B, T, 3, H, W] -> [B*T, 3, H, W]
    img = img.view(B * T, C, H, W)
    # tgt: [B, T, C, H, W] -> [B*T, C, H, W]
    tgt = tgt.view(B * T, C2, H2, W2)

    if "instance_labels" in extras:
        inst = extras["instance_labels"]
        # can be [B, T, H, W]
        if inst.ndim == 4:
            inst = inst.view(B * T, H, W)
        # can be [B, 1, H, W]
        elif inst.ndim == 4 and inst.shape[1] == 1:
            inst = inst.view(B * T, H, W)
        # can be [B, H, W] (rare)
        elif inst.ndim == 3:
            inst = inst.unsqueeze(1).expand(B, T, H, W).contiguous().view(B * T, H, W)
        else:
            raise ValueError(f"unexpected instance_labels shape {inst.shape}")
        extras["instance_labels"] = inst

    return img, tgt, extras

_GK_CACHE = {}
def gaussian_kernel1d(sigma: float, radius: int | None = None, device=None, dtype=None):
    if sigma <= 0:
        # trivial kernel
        return torch.tensor([1.0], device=device, dtype=dtype)
    if radius is None:
        radius = int(max(1, round(3*sigma)))
    key = (float(sigma), int(radius), device, dtype)
    k = _GK_CACHE.get(key)
    if k is None:
        x = torch.arange(-radius, radius+1, device=device, dtype=dtype)
        k = torch.exp(-(x**2)/(2*sigma*sigma))
        k = k / k.sum()
        _GK_CACHE[key] = k
    return k

def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: [B,1,H,W]
    if sigma <= 0:
        return x
    b, c, h, w = x.shape
    k1d = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, kx.shape[-1]//2), groups=1)
    x = F.conv2d(x, ky, padding=(ky.shape[-2]//2, 0), groups=1)
    return x

def worker_init_fn(worker_id):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    # If your Dataset keeps an RNG, re-seed it:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "rng"):
        worker_info.dataset.rng = np.random.default_rng(base_seed + worker_id)

def _tqdm(iterable, desc, position, total=None, disable=None):
    return tqdm(
        iterable, desc=desc, total=total, leave=False,
        dynamic_ncols=True, mininterval=0.2, smoothing=0.1,
        position=position, disable=disable
    )

def ddp_is_active():
    return dist.is_available() and dist.is_initialized()

def _ddp_broadcast_bool(flag: bool, device):
    if not ddp_is_active():
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    if ddp_rank() == 0:
        t[...] = 1 if flag else 0
    torch.distributed.broadcast(t, src=0)
    return bool(int(t.item()))

def ddp_world_size():
    return dist.get_world_size() if ddp_is_active() else 1

def ddp_rank():
    return dist.get_rank() if ddp_is_active() else 0

def reduce_mean_scalar(val: float, device):
    if not ddp_is_active():
        return float(val)
    t = torch.tensor([val], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / ddp_world_size()).item())

def reduce_mean_dict(d: dict, device):
    if not ddp_is_active():
        return d
    return {k: reduce_mean_scalar(float(v), device) for k, v in d.items()}

def _concat_extras(ex_a: dict, ex_b: dict) -> dict:
    out = {}

    if "instance_labels" in ex_a and "instance_labels" in ex_b:
        out["instance_labels"] = torch.cat(
            [ex_a["instance_labels"], ex_b["instance_labels"]],
            dim=0,
        )

    if "meta" in ex_a and "meta" in ex_b:
        # collate_no_meta usually gives a list of dicts
        meta_a = ex_a["meta"]
        meta_b = ex_b["meta"]
        if isinstance(meta_a, list) and isinstance(meta_b, list):
            out["meta"] = meta_a + meta_b
        else:
            out["meta"] = meta_a

    return out

def _next_bg_batch(bg_iter, bg_loader):
    try:
        batch = next(bg_iter)
    except StopIteration:
        bg_iter = iter(bg_loader)
        batch = next(bg_iter)
    return batch, bg_iter

# --------------------- classic dice helpers ---------------------

def dice_loss_from_probs(probs, target, eps=1e-6):
    # probs/target: [N,1,H,W]
    num = 2 * (probs * target).sum(dim=(0,2,3)) + eps
    den = (probs + target).sum(dim=(0,2,3)) + eps
    return (1 - num / den).mean()

def dice_from_probs(probs, target, eps=1e-6):
    # probs/target: [N,1,H,W]
    num = 2 * (probs * target).sum(dim=(0,2,3))
    den = (probs + target).sum(dim=(0,2,3)) + eps
    return (num / den).mean().item()

# --------------------- target shaping (batch) ---------------------

def _compute_inner_boundary(inst_np: np.ndarray) -> np.ndarray:
    """1-px inner boundary from integer labels (H,W)."""
    a = inst_np
    H, W = a.shape
    up    = (a != np.roll(a, -1, axis=0))
    down  = (a != np.roll(a,  1, axis=0))
    left  = (a != np.roll(a, -1, axis=1))
    right = (a != np.roll(a,  1, axis=1))
    b = (up | down | left | right)
    b &= (a > 0)
    # zero wrap-around on edges
    b[0, :]   &= (a[0, :]   != 0)
    b[-1, :]  &= (a[-1, :]  != 0)
    b[:, 0]   &= (a[:, 0]   != 0)
    b[:, -1]  &= (a[:, -1]  != 0)
    return b.astype(np.uint8)

def make_soft_boundary_batch(inst_batch: torch.Tensor,
                             ring_width: int = 1,
                             soft_band: int = 2,
                             sigma: float = 1.0,
                             device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Build soft boundary targets for a batch of instance labels.
    inst_batch: [B,H,W] int
    returns: [B,1,H,W] float32 in [0,1]
    """
    B, H, W = inst_batch.shape
    out = np.zeros((B, H, W), dtype=np.float32)
    for i in range(B):
        inst = inst_batch[i].cpu().numpy()
        ring = _compute_inner_boundary(inst).astype(bool)
        if ring_width > 1:
            rad = max(1, int(ring_width//2))
            ring = ndi.binary_dilation(
                ring, structure=ndi.generate_binary_structure(2,1), iterations=rad
            )
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
            out[i] = soft.astype(np.float32)
        else:
            out[i] = ring.astype(np.float32)
    out_t = torch.from_numpy(out).unsqueeze(1)  # [B,1,H,W]
    if device is not None:
        out_t = out_t.to(device, non_blocking=True)
    return out_t

def make_outside_mask_centers_batch(meta_list: List[dict],
                                    H: int, W: int,
                                    radius: int = 6,
                                    device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Build 1 where peaks are allowed (small disks around GT centers), else 0.
    Returns [B,1,H,W] float32 in {0,1}

    Notes:
      - Maps centers from original sim coords -> current [H,W] coords using meta['mode_meta'].
      - Supports modes: pad_resize, crop_well_resize, tiles.
      - If 'mode_meta' is missing, uses centers as-is.
    """
    yy, xx = np.ogrid[:H, :W]
    B = len(meta_list)
    mask = np.zeros((B, H, W), dtype=np.float32)
    r2 = int(radius) * int(radius)

    def _map_centers(m: dict) -> List[tuple]:
        # original centers from the simulator
        ctrs = m.get("centers", []) or []
        mm = m.get("mode_meta", None)
        if not ctrs:
            return []
        if mm is None or "mode" not in mm:
            # assume already in [H,W]
            return [(int(round(y)), int(round(x))) for (y, x) in ctrs]

        mode = mm["mode"]
        mapped = []
        if mode == "pad_resize":
            # We padded to a square (S_in) with top/left, then scaled by 'scale' to target=[H,W]
            pt = int(mm.get("pad_top", 0))
            pl = int(mm.get("pad_left", 0))
            scale = float(mm.get("scale", 1.0))
            for (y, x) in ctrs:
                yy_m = int(round((float(y) + pt) * scale))
                xx_m = int(round((float(x) + pl) * scale))
                mapped.append((yy_m, xx_m))

        elif mode == "crop_well_resize":
            # We cropped [y0:y1, x0:x1] then scaled by 'scale' to target=[H,W]
            y0, y1, x0, x1 = mm.get("crop", (0, 0, 0, 0))
            scale = float(mm.get("scale", 1.0))
            for (y, x) in ctrs:
                yy_m = int(round((float(y) - float(y0)) * scale))
                xx_m = int(round((float(x) - float(x0)) * scale))
                mapped.append((yy_m, xx_m))

        elif mode == "tiles":
            # We cut a tile with top-left (y0,x0); no scaling
            y0, x0 = mm.get("tile_xy", (0, 0))
            for (y, x) in ctrs:
                yy_m = int(round(float(y) - float(y0)))
                xx_m = int(round(float(x) - float(x0)))
                mapped.append((yy_m, xx_m))
        else:
            # unknown mode -> best effort: assume already in [H,W]
            mapped = [(int(round(y)), int(round(x))) for (y, x) in ctrs]

        return mapped

    for i, m in enumerate(meta_list):
        ctrs_mapped = _map_centers(m)
        if not ctrs_mapped:
            continue

        acc = np.zeros((H, W), dtype=np.uint8)
        for (y, x) in ctrs_mapped:
            if 0 <= y < H and 0 <= x < W:
                rr = (yy - y) * (yy - y) + (xx - x) * (xx - x)
                acc[rr <= r2] = 1
        mask[i] = acc

    mask_t = torch.from_numpy(mask).unsqueeze(1)  # [B,1,H,W]
    if device is not None:
        mask_t = mask_t.to(device, non_blocking=True)

    return mask_t

# --------------------- specialized losses ---------------------

def make_ring_and_far_bg_masks(cell_mask: torch.Tensor,
                               ring_px: int = 3,
                               far_px: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """
    cell_mask: [B,1,H,W] in {0,1} (or [0..1])
    Returns:
      ring_mask: [B,1,H,W] ~1 on a thin outer ring around cell borders
      far_bg_mask: [B,1,H,W] ~1 on background pixels at least `far_px` away from cells
    """
    m = (cell_mask > 0.5).float()
    # outer ring via dilation - mask
    if ring_px > 0:
        ring_dil = F.max_pool2d(m, kernel_size=2*ring_px+1, stride=1, padding=ring_px)
        ring_mask = (ring_dil - m).clamp_(0, 1)
    else:
        ring_mask = torch.zeros_like(m)

    # far background: invert a *larger* dilation
    if far_px > 0:
        far_dil = F.max_pool2d(m, kernel_size=2*far_px+1, stride=1, padding=far_px)
        far_bg_mask = (1.0 - far_dil).clamp_(0, 1)
    else:
        far_bg_mask = (1.0 - m).clamp_(0, 1)

    return ring_mask, far_bg_mask

def bce_dice_loss(logits, target, w_bce=0.3, w_dice=0.7):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    d    = dice_loss_from_probs(probs, target)
    return w_bce * bce + w_dice * d

def boundary_exclusion_loss(cell_prob, boundary_target, weight=0.2):
    """Penalize cell prob where boundary target is high."""
    if weight is None or weight <= 0:
        return torch.tensor(0.0, device=cell_prob.device, dtype=cell_prob.dtype)
    return float(weight) * (cell_prob * boundary_target).mean()

def center_head_loss(center_logits, center_target, cell_mask=None,
                     pos_weight=10.0, w_bce=0.6, w_mse=0.4,
                     sparsity_weight: float = 0.0, outside_mask: Optional[torch.Tensor] = None,
                     count_weight: float = 0.0):
    """
    Composite: pos-weighted BCE + MSE (+ optional sparsity outside disks, + count consistency).
    Shapes: [B,1,H,W]
    """
    p = torch.sigmoid(center_logits)
    t = center_target

    if cell_mask is not None:
        m = (cell_mask > 0.5).float()
        denom = m.sum().clamp_min(1.0)
        pw = torch.tensor([pos_weight], device=center_logits.device, dtype=center_logits.dtype)
        bce_map = F.binary_cross_entropy_with_logits(center_logits, t, pos_weight=pw, reduction="none")
        bce = (m * bce_map).sum() / denom
        mse = (m * (p - t).pow(2)).sum() / denom
    else:
        pw = torch.tensor([pos_weight], device=center_logits.device, dtype=center_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(center_logits, t, pos_weight=pw)
        mse = F.mse_loss(p, t)

    loss = w_bce * bce + w_mse * mse

    if sparsity_weight > 0.0 and outside_mask is not None:
        outside = (outside_mask <= 0.5).float()
        if cell_mask is not None:
            outside = outside * (cell_mask > 0.5).float()
        denom_o = outside.sum().clamp_min(1.0)
        sparsity = (outside * p).sum() / denom_o
        loss = loss + float(sparsity_weight) * sparsity

    if count_weight > 0.0:
        target_mass = t.sum(dim=(2,3))
        pred_mass   = p.sum(dim=(2,3))
        count_l1 = (pred_mass - target_mass).abs().mean()
        loss = loss + float(count_weight) * count_l1

    return loss

def energy_head_loss_masked(energy_logits, energy_target, cell_mask,
                            w_l1=0.5, w_mse=0.5):
    """Regression inside cells: L1 + MSE on sigmoid(logits)."""
    p = torch.sigmoid(energy_logits)
    t = energy_target
    m = (cell_mask > 0.5).float()
    denom = m.sum().clamp_min(1.0)
    l1  = (m * (p - t).abs()).sum() / denom
    mse = (m * (p - t).pow(2)).sum() / denom
    return w_l1 * l1 + w_mse * mse

# --------------------- better metrics ---------------------

def boundary_fscore_np(pred_bound: np.ndarray, gt_bound: np.ndarray, tol: int = 2) -> dict:
    """
    pred_bound, gt_bound: float maps in [0,1] (H,W).
    tol in pixels. Dilation tolerance.
    """
    pred = (pred_bound >= 0.5).astype(np.uint8)
    gt   = (gt_bound   >= 0.5).astype(np.uint8)
    k = 2*tol + 1
    se = np.ones((k, k), np.uint8)
    pred_d = cv2.dilate(pred, se)
    gt_d   = cv2.dilate(gt,   se)
    tp_p = (pred & gt_d).sum()
    tp_g = (gt   & pred_d).sum()
    prec = tp_p / (pred.sum() + 1e-8)
    rec  = tp_g / (gt.sum()   + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

def compute_thin_gt_boundary_from_instances(inst_np: np.ndarray) -> np.ndarray:
    # 1-px inner boundary from integer instances
    a = inst_np
    H, W = a.shape
    up    = (a != np.roll(a, -1, axis=0))
    down  = (a != np.roll(a,  1, axis=0))
    left  = (a != np.roll(a, -1, axis=1))
    right = (a != np.roll(a,  1, axis=1))
    b = (up | down | left | right)
    b &= (a > 0)
    b[H-1,:] = b[H-1,:] & (a[H-1,:] != 0)
    b[0,:]   = b[0,:]   & (a[0,:]   != 0)
    b[:,W-1] = b[:,W-1] & (a[:,W-1] != 0)
    b[:,0]   = b[:,0]   & (a[:,0]   != 0)
    return b.astype(np.uint8)

def boundary_f1_skeletonized(pred_prob: np.ndarray,
                             inst_gt: np.ndarray,
                             tol: int = 2,
                             thr: float = 0.9,
                             sweep: bool = False) -> float:
    """
    F1 between predicted boundary map and a thin GT boundary extracted from instances.
    Optionally sweep thresholds and return best F1.
    """
    gt_thin = compute_thin_gt_boundary_from_instances(inst_gt)

    def f1_at(t):
        pred_bin = (pred_prob >= t).astype(np.uint8)
        pred_skel = skeletonize(pred_bin > 0).astype(np.uint8)
        stats = boundary_fscore_np(pred_skel, gt_thin, tol=tol)
        return float(stats["f1"])

    if sweep:
        ts = np.linspace(0.05, 0.7, 14)  # coarse sweep; cheap
        return max(f1_at(t) for t in ts)
    else:
        return f1_at(thr)

def _nms_peaks_np(heat: np.ndarray, thr: float = 0.2, min_dist: int = 3) -> List[Tuple[int,int,float]]:
    h = (heat >= thr).astype(np.uint8)
    if min_dist > 1:
        k = 2*min_dist + 1
        heat_max = cv2.dilate(heat, np.ones((k, k), np.uint8))
        h = np.logical_and(h, heat == heat_max)
    ys, xs = np.nonzero(h)
    scores = heat[ys, xs]
    idx = np.argsort(-scores)
    return [(int(ys[i]), int(xs[i]), float(scores[i])) for i in idx]


def center_metrics_np(center_pred, gt_centers, peak_thr=0.2, nms_dist=3, match_radius=10):
    preds = _nms_peaks_np(center_pred, thr=peak_thr, min_dist=nms_dist)
    P = np.array([(y,x) for y,x,_ in preds], dtype=float)
    G = np.array(gt_centers, dtype=float)
    if len(P)==0 or len(G)==0:
        TP, FP, FN = 0, len(P), len(G)
    else:
        tree = cKDTree(G)
        taken = np.zeros(len(G), dtype=bool)
        TP = 0
        for i, p in enumerate(P):
            d, j = tree.query(p, distance_upper_bound=match_radius)
            if np.isfinite(d) and j < len(G) and not taken[j]:
                taken[j] = True
                TP += 1
        FP = len(P) - TP
        FN = len(G) - TP
    prec = TP / (TP + FP + 1e-8)
    rec = TP / (TP + FN + 1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    return {"precision":float(prec), "recall":float(rec), "f1":float(f1),
            "n_pred":len(P), "n_gt":len(G), "n_tp":TP}


def energy_errors_np(energy_pred: np.ndarray,
                     energy_gt: np.ndarray,
                     cell_mask_gt: np.ndarray) -> dict:
    """
    Returns RMSE and Pearson r inside cells.
    If variance is ~0, marks Pearson invalid and returns NaN.
    """
    m = (cell_mask_gt > 0.5)
    if not np.any(m):
        return {"rmse": np.nan, "pearson": np.nan, "valid_rmse": False, "valid_r": False}

    p = energy_pred[m].astype(np.float32)
    g = energy_gt[m].astype(np.float32)

    # RMSE is always well-defined as long as we have pixels
    rmse = float(np.sqrt(np.mean((p - g) ** 2))) if p.size else np.nan
    valid_rmse = bool(np.isfinite(rmse))

    # Pearson: guard zero-variance and clamp to [-1, 1]
    p0 = p - p.mean()
    g0 = g - g.mean()
    denom = np.sqrt((p0**2).sum() * (g0**2).sum())
    if denom <= 1e-8:
        r = np.nan
        valid_r = False
    else:
        r = float((p0 * g0).sum() / denom)
        # numerical safety
        r = float(np.clip(r, -1.0, 1.0))
        valid_r = True

    return {"rmse": rmse, "pearson": r, "valid_rmse": valid_rmse, "valid_r": valid_r}


# --------------------- per-epoch weight schedule ---------------------

def make_weight_schedule(
    C: int,
    epoch: int,
    warmup_epochs: int = 10,
    decay_epochs: int = 50,
) -> list[float]:
    """
    Channel order:
      0: cell, 1: boundary, 2: center, 3: energy
    hi until warmup is reached, then linearly decay to lo over decay_epochs.
    """
    hi = [1.0, 1.8, 3.0, 3.0]
    lo = [1.0, 1.3, 0.5, 0.5]

    # still in warmup -> use hi exactly
    if epoch <= warmup_epochs:
        return hi[:C]

    # warmup passed
    if decay_epochs <= 0:
        return lo[:C]  # immediate switch if no decay window

    # blend factor from 0..1 across the decay window
    t = min(1.0, max(0.0, (epoch - warmup_epochs) / float(decay_epochs)))
    w = [hi[i] + t * (lo[i] - hi[i]) for i in range(C)]
    return w[:C]

# --------------------- training loops ---------------------

class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = float(val_loss)
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

def _build_aux_targets(extras: dict,
                       tgt: torch.Tensor,
                       device: torch.device,
                       center_outside_radius: int = 6,
                       bound_sigma: float = 1.0):
    """
    Build:
      - soft boundary target from instances
      - outside mask for center loss (optional)
      - cell mask tensor from targets (for masking energy/center)
    """
    inst_b = extras["instance_labels"]  # [B,H,W] int
    if inst_b.dtype != torch.int32 and inst_b.dtype != torch.int64:
        inst_b = inst_b.to(torch.int32)
    B, H, W = inst_b.shape

    bound_soft = tgt[:, 1:2]
    bound_soft = gaussian_blur_2d(bound_soft, sigma=bound_sigma)
    cell_mask = tgt[:, 0:1]  # [B,1,H,W] float
    metas = extras.get("meta", None)
    center_allow = None
    if metas is not None and isinstance(metas, list):
        center_allow = make_outside_mask_centers_batch(metas, H, W, radius=center_outside_radius, device=device)  # [B,1,H,W]
    return bound_soft, cell_mask, center_allow

def train_epoch(
    model,
    loader,
    opt, scaler, device, use_amp, weights, w_bce, w_dice, show_bar, max_steps=None,
    grad_clip_norm: Optional[float] = None,
    center_pos_weight: float = 10.0,
    center_w_bce: float = 0.6,
    center_w_mse: float = 0.4,
    center_sparsity_weight: float = 0.0,
    energy_w_l1: float = 0.5,
    energy_w_mse: float = 0.5,
    excl_weight: float = 0.2,
    halo_ring_px: int = 3,
    halo_far_px: int = 12,
    halo_w_cell: float = 0.05,
    halo_w_energy_ring: float = 0.05,
    halo_w_energy_far: float = 0.10,
    halo_w_center_ring: float = 0.05,
    halo_w_center_far: float = 0.10,
    bg_loader=None,
    bg_mix_frac: float = 0.0,
):

        
    bg_iter = iter(bg_loader) if (bg_loader is not None and bg_mix_frac > 0.0) else None
    model.train()
    # running sums
    steps = 0
    running_loss_weighted = 0.0
    running_loss_unweighted = 0.0
    running_loss_cell = 0.0
    running_loss_bound = 0.0
    running_loss_center = 0.0
    running_loss_energy = 0.0
    running_loss_excl = 0.0
    running_loss_cell_halo = 0.0
    running_loss_energy_halo = 0.0
    running_loss_center_halo = 0.0

    running_dice_cell  = 0.0
    running_dice_bound = 0.0
    running_dice_center = 0.0
    running_dice_energy = 0.0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    total = (min(max_steps, len(loader)) if (max_steps is not None) else len(loader))
    pbar = _tqdm(loader, desc="train", position=1, disable=not show_bar, total=total)

    for step_idx, (img, tgt, extras) in enumerate(pbar, start=1):
        img, tgt, extras = flatten_tiles(img, tgt, extras)

        # ---- optional background mixing ----
        if bg_iter is not None and bg_mix_frac > 0.0:
            B_main = img.shape[0]
            n_bg = np.random.binomial(B_main, bg_mix_frac)

            if n_bg > 0:
                bg_img, bg_tgt, bg_extras = None, None, None
                collected = 0
                bg_img_parts, bg_tgt_parts, bg_inst_parts, bg_meta_parts = [], [], [], []

                while collected < n_bg:
                    (b_img, b_tgt, b_extras), bg_iter = _next_bg_batch(bg_iter, bg_loader)
                    b_img, b_tgt, b_extras = flatten_tiles(b_img, b_tgt, b_extras)

                    take = min(n_bg - collected, b_img.shape[0])

                    bg_img_parts.append(b_img[:take])
                    bg_tgt_parts.append(b_tgt[:take])

                    if "instance_labels" in b_extras:
                        bg_inst_parts.append(b_extras["instance_labels"][:take])

                    if "meta" in b_extras and isinstance(b_extras["meta"], list):
                        bg_meta_parts.extend(b_extras["meta"][:take])

                    collected += take

                bg_img = torch.cat(bg_img_parts, dim=0)
                bg_tgt = torch.cat(bg_tgt_parts, dim=0)
                bg_extras = {
                    "instance_labels": torch.cat(bg_inst_parts, dim=0) if bg_inst_parts else None,
                    "meta": bg_meta_parts,
                }

                n_keep = B_main - n_bg

                img = torch.cat([img[:n_keep], bg_img], dim=0)
                tgt = torch.cat([tgt[:n_keep], bg_tgt], dim=0)

                main_extras = {
                    "instance_labels": extras["instance_labels"][:n_keep],
                    "meta": extras["meta"][:n_keep] if isinstance(extras.get("meta", None), list) else extras.get("meta", None),
                }
                extras = _concat_extras(main_extras, bg_extras)

        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        B, C, H, W = tgt.shape

        bound_soft, cell_mask, center_allow = _build_aux_targets(
            extras, tgt, device,
            center_outside_radius=3,
            bound_sigma=1.0
        )

        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)  # [B,C,H,W]
            probs  = torch.sigmoid(logits)

            # --- per-head losses ---
            loss_cell  = bce_dice_loss(logits[:,0:1], tgt[:,0:1], w_bce=w_bce, w_dice=w_dice)
            loss_bound = bce_dice_loss(logits[:,1:2], bound_soft,  w_bce=w_bce, w_dice=w_dice)
            loss_center = torch.tensor(0.0, device=device)
            if C >= 3:
                loss_center = center_head_loss(
                    logits[:,2:3], tgt[:,2:3], cell_mask=cell_mask,
                    pos_weight=center_pos_weight, w_bce=center_w_bce, w_mse=center_w_mse,
                    sparsity_weight=center_sparsity_weight, outside_mask=center_allow,
                    count_weight=0.0
                )
            loss_energy = torch.tensor(0.0, device=device)
            if C >= 4:
                loss_energy = energy_head_loss_masked(
                    logits[:,3:4], tgt[:,3:4], cell_mask=cell_mask,
                    w_l1=energy_w_l1, w_mse=energy_w_mse
                )

            # exclusion (regularizer)
            loss_excl = boundary_exclusion_loss(probs[:,0:1], bound_soft, weight=excl_weight)

            # --- anti-halo (regularizers) ---
            m = (cell_mask > 0.5).float()
            if halo_ring_px > 0:
                ring_dil = F.max_pool2d(m, kernel_size=2*halo_ring_px+1, stride=1, padding=halo_ring_px)
                ring_mask = (ring_dil - m).clamp_(0, 1)
            else:
                ring_mask = torch.zeros_like(m)
            if halo_far_px > 0:
                far_dil = F.max_pool2d(m, kernel_size=2*halo_far_px+1, stride=1, padding=halo_far_px)
                far_bg_mask = (1.0 - far_dil).clamp_(0, 1)
            else:
                far_bg_mask = (1.0 - m).clamp_(0, 1)

            cell_prob = probs[:, 0:1]
            loss_cell_halo = halo_w_cell * (ring_mask * cell_prob).mean()

            loss_energy_halo = torch.tensor(0.0, device=device)
            if C >= 4:
                energy_prob = probs[:, 3:4]
                ring_leak_e  = (ring_mask   * energy_prob).mean()
                far_leak_e   = (far_bg_mask * energy_prob).mean()
                loss_energy_halo = halo_w_energy_ring * ring_leak_e + halo_w_energy_far * far_leak_e

            loss_center_halo = torch.tensor(0.0, device=device)
            if C >= 3:
                center_prob = probs[:, 2:3]
                ring_leak_c = (ring_mask   * center_prob).mean()
                far_leak_c  = (far_bg_mask * center_prob).mean()
                loss_center_halo = halo_w_center_ring * ring_leak_c + halo_w_center_far * far_leak_c

            # totals
            heads = [loss_cell, loss_bound]
            if C >= 3:
                heads.append(loss_center)
            if C >= 4:
                heads.append(loss_energy)
            loss_unweighted = torch.stack(heads).mean()

            loss_weighted = (
                weights[0] * loss_cell +
                (weights[1] if len(weights) > 1 else 0.0) * loss_bound +
                (weights[2] if len(weights) > 2 else 0.0) * loss_center +
                (weights[3] if len(weights) > 3 else 0.0) * loss_energy +
                loss_excl + loss_cell_halo + loss_energy_halo + loss_center_halo
            )

        # backprop on weighted loss (as before)
        scaler.scale(loss_weighted).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        scaler.step(opt)
        scaler.update()

        # metrics
        with torch.no_grad():
            running_dice_cell  += dice_from_probs(probs[:,0:1], tgt[:,0:1])
            running_dice_bound += dice_from_probs(probs[:,1:2], bound_soft)
            if C >= 3:
                running_dice_center += dice_from_probs(probs[:,2:3], tgt[:,2:3])
            if C >= 4:
                running_dice_energy += dice_from_probs(probs[:,3:4], tgt[:,3:4])

        # accumulate losses
        running_loss_cell         += float(loss_cell.item())
        running_loss_bound        += float(loss_bound.item())
        running_loss_center       += float(loss_center.item())
        running_loss_energy       += float(loss_energy.item())
        running_loss_excl         += float(loss_excl.item())
        running_loss_cell_halo    += float(loss_cell_halo.item())
        running_loss_energy_halo  += float(loss_energy_halo.item()) if C >= 4 else 0.0
        running_loss_center_halo  += float(loss_center_halo.item()) if C >= 3 else 0.0
        running_loss_unweighted   += float(loss_unweighted.item())
        running_loss_weighted     += float(loss_weighted.item())

        steps += 1

        if show_bar:
            post = {
                "loss_w": f"{running_loss_weighted/steps:.4f}",
                "loss_u": f"{running_loss_unweighted/steps:.4f}",
                "d_cell": f"{running_dice_cell/steps:.3f}",
                "d_bound": f"{running_dice_bound/steps:.3f}",
            }
            if C >= 3:
                post["d_center"] = f"{running_dice_center/steps:.3f}"
            if C >= 4:
                post["d_energy"] = f"{running_dice_energy/steps:.3f}"
            pbar.set_postfix(post)

        if (max_steps is not None) and (step_idx >= int(max_steps)):
            break

    out = {
        "loss_weighted": running_loss_weighted / max(1, steps),
        "loss_unweighted": running_loss_unweighted / max(1, steps),
        "loss_cell":   running_loss_cell   / max(1, steps),
        "loss_bound":  running_loss_bound  / max(1, steps),
        "loss_center": running_loss_center / max(1, steps) if steps else float("nan"),
        "loss_energy": running_loss_energy / max(1, steps) if steps else float("nan"),
        "loss_excl":   running_loss_excl   / max(1, steps),
        "loss_cell_halo":   running_loss_cell_halo   / max(1, steps),
        "loss_energy_halo": running_loss_energy_halo / max(1, steps),
        "loss_center_halo": running_loss_center_halo / max(1, steps),
    }
    # classic dice
    if steps:
        out["dice_cell"]   = running_dice_cell / steps
        out["dice_bound"]  = running_dice_bound / steps
        if C >= 3:
            out["dice_center"] = running_dice_center / steps
        if C >= 4:
            out["dice_energy"] = running_dice_energy / steps
    return out


@torch.no_grad()
def eval_epoch(model, loader, device, use_amp, weights, w_bce, w_dice, show_bar,
               center_match_radius_px: int = 10, center_thr: float = 0.2, center_nms: int = 3,
               bound_tol_px: int = 2, bound_thr: float = 0.7, bound_sweep: bool = False,
               halo_ring_px: int = 3, halo_far_px: int = 12):
    model.eval()
    # loss accumulators
    n = 0
    loss_w_sum = 0.0
    loss_u_sum = 0.0
    loss_cell_sum = 0.0
    loss_bound_sum = 0.0
    loss_center_sum = 0.0
    loss_energy_sum = 0.0
    loss_excl_sum = 0.0

    # dice/metrics accumulators (as before)
    dice_cell_sum   = 0.0
    dice_bound_sum  = 0.0
    dice_center_sum = 0.0
    dice_energy_sum = 0.0
    boundF_sum = 0.0
    n_bound_valid = 0
    centerF_sum = 0.0
    n_center_valid = 0
    energy_rmse_sum = 0.0
    energy_rmse_n = 0
    energy_r_sum = 0.0
    energy_r_n = 0
    # anti-halo diagnostics
    ring_leak_cell_sum = 0.0
    far_leak_cell_sum  = 0.0
    ring_leak_energy_sum = 0.0
    far_leak_energy_sum  = 0.0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    pbar = _tqdm(loader, desc="evaluation", position=2, disable=not show_bar)

    for img, tgt, extras in pbar:
        img, tgt, extras = flatten_tiles(img, tgt, extras)
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        B, C, H, W = tgt.shape

        bound_soft, cell_mask, center_allow = _build_aux_targets(
            extras, tgt, device,
            center_outside_radius=3,
            bound_sigma=1.0
        )

        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)
            probs  = torch.sigmoid(logits)

            # per-head
            loss_cell  = bce_dice_loss(logits[:,0:1], tgt[:,0:1], w_bce=w_bce, w_dice=w_dice)
            loss_bound = bce_dice_loss(logits[:,1:2], bound_soft,  w_bce=w_bce, w_dice=w_dice)
            loss_center = torch.tensor(0.0, device=device)
            if C >= 3:
                loss_center = center_head_loss(
                    logits[:,2:3], tgt[:,2:3], cell_mask=cell_mask,
                    pos_weight=10.0, w_bce=0.6, w_mse=0.4,
                    sparsity_weight=0.0, outside_mask=center_allow, count_weight=0.0
                )
            loss_energy = torch.tensor(0.0, device=device)
            if C >= 4:
                loss_energy = energy_head_loss_masked(
                    logits[:,3:4], tgt[:,3:4], cell_mask=cell_mask,
                    w_l1=0.5, w_mse=0.5
                )
            loss_excl  = boundary_exclusion_loss(probs[:,0:1], bound_soft, weight=0.2)

            # totals
            heads = [loss_cell, loss_bound]
            if C >= 3:
                heads.append(loss_center)
            if C >= 4:
                heads.append(loss_energy)
            loss_unweighted = torch.stack(heads).mean()

            loss_weighted = (
                weights[0] * loss_cell +
                (weights[1] if len(weights) > 1 else 0.0) * loss_bound +
                (weights[2] if len(weights) > 2 else 0.0) * loss_center +
                (weights[3] if len(weights) > 3 else 0.0) * loss_energy +
                loss_excl
            )

        # dice summaries
        dice_cell_sum  += dice_from_probs(probs[:,0:1], tgt[:,0:1])
        dice_bound_sum += dice_from_probs(probs[:,1:2], bound_soft)
        if C >= 3:
            dice_center_sum += dice_from_probs(probs[:,2:3], tgt[:,2:3])
        if C >= 4:
            dice_energy_sum += dice_from_probs(probs[:,3:4], tgt[:,3:4])

        # anti-halo diagnostics (unchanged)
        m = (cell_mask > 0.5).float()
        if halo_ring_px > 0:
            ring_dil = F.max_pool2d(m, kernel_size=2*halo_ring_px+1, stride=1, padding=halo_ring_px)
            ring_mask = (ring_dil - m).clamp_(0, 1)
        else:
            ring_mask = torch.zeros_like(m)
        if halo_far_px > 0:
            far_dil = F.max_pool2d(m, kernel_size=2*halo_far_px+1, stride=1, padding=halo_far_px)
            far_bg_mask = (1.0 - far_dil).clamp_(0, 1)
        else:
            far_bg_mask = (1.0 - m).clamp_(0, 1)

        cell_prob = probs[:, 0:1]
        ring_leak_cell_sum += float((ring_mask * cell_prob).mean().item())
        far_leak_cell_sum  += float((far_bg_mask * cell_prob).mean().item())

        if C >= 4:
            energy_prob = probs[:, 3:4]
            ring_leak_energy_sum += float((ring_mask   * energy_prob).mean().item())
            far_leak_energy_sum  += float((far_bg_mask * energy_prob).mean().item())

        # numpy metrics (unchanged)
        probs_np = probs.detach().float().cpu().numpy()
        tgt_np   = tgt.detach().float().cpu().numpy()
        metas = extras.get("meta", [])
        insts = extras["instance_labels"].detach().cpu().numpy()

        for i in range(B):
            cell_gt   = tgt_np[i, 0]
            center_pr = probs_np[i, 2] if C >= 3 else None
            energy_pr = probs_np[i, 3] if C >= 4 else None

            bf_f1 = boundary_f1_skeletonized(
                pred_prob=probs_np[i, 1],
                inst_gt=insts[i],
                tol=bound_tol_px,
                thr=bound_thr,
                sweep=bound_sweep
            )
            if not np.isnan(bf_f1):
                boundF_sum += bf_f1
                n_bound_valid += 1

            if center_pr is not None and i < len(metas):
                gt_centers = metas[i].get("centers", []) or []
                cm = center_metrics_np(center_pr, gt_centers,
                                       peak_thr=center_thr, nms_dist=center_nms,
                                       match_radius=center_match_radius_px)
                if not np.isnan(cm["f1"]):
                    centerF_sum += cm["f1"]
                    n_center_valid += 1

            if energy_pr is not None:
                energy_gt = tgt_np[i, 3] if C >= 4 else np.zeros_like(cell_gt)
                ee = energy_errors_np(energy_pr, energy_gt, cell_gt)
                if ee.get("valid_rmse", False) and np.isfinite(ee["rmse"]):
                    energy_rmse_sum += ee["rmse"]
                    energy_rmse_n += 1
                if ee.get("valid_r", False) and np.isfinite(ee["pearson"]):
                    energy_r_sum += ee["pearson"]
                    energy_r_n += 1

        # accumulate losses
        loss_w_sum     += float(loss_weighted.item())
        loss_u_sum     += float(loss_unweighted.item())
        loss_cell_sum  += float(loss_cell.item())
        loss_bound_sum += float(loss_bound.item())
        loss_center_sum+= float(loss_center.item())
        loss_energy_sum+= float(loss_energy.item())
        loss_excl_sum  += float(loss_excl.item())
        n += 1

        if show_bar:
            post = {
                "loss_w": f"{loss_w_sum/n:.4f}",
                "loss_u": f"{loss_u_sum/n:.4f}",
                "d_cell": f"{dice_cell_sum/n:.3f}",
                "d_bound": f"{dice_bound_sum/n:.3f}",
            }
            if C >= 3:
                post["d_center"] = f"{dice_center_sum/n:.3f}"
            if C >= 4:
                post["d_energy"] = f"{dice_energy_sum/n:.3f}"
            pbar.set_postfix(post)

    out = {
        # selection loss (unweighted)
        "loss_unweighted": loss_u_sum / max(1, n),
        # also keep weighted and per-head for logging
        "loss_weighted":   loss_w_sum / max(1, n),
        "loss_cell":       loss_cell_sum  / max(1, n),
        "loss_bound":      loss_bound_sum / max(1, n),
        "loss_center":     loss_center_sum/ max(1, n) if n else float("nan"),
        "loss_energy":     loss_energy_sum/ max(1, n) if n else float("nan"),
        "loss_excl":       loss_excl_sum  / max(1, n),
    }

    if n:
        out["dice_cell"]   = dice_cell_sum / n
        out["dice_bound"]  = dice_bound_sum / n
        if C >= 3:
            out["dice_center"] = dice_center_sum / n
        if C >= 4:
            out["dice_energy"] = dice_energy_sum / n

        out["bound_f1_tol2"]  = boundF_sum / n_bound_valid if n_bound_valid > 0 else np.nan
        out["center_f1_r10"]  = centerF_sum / n_center_valid if n_center_valid > 0 else np.nan
        out["energy_rmse"]    = (energy_rmse_sum / energy_rmse_n) if energy_rmse_n > 0 else np.nan
        out["energy_pearson"] = (energy_r_sum   / energy_r_n)    if energy_r_n   > 0 else np.nan

        out["ring_leak_cell"]   = ring_leak_cell_sum / n
        out["far_leak_cell"]    = far_leak_cell_sum / n
        out["ring_leak_energy"] = ring_leak_energy_sum / n if C >= 4 else np.nan
        out["far_leak_energy"]  = far_leak_energy_sum  / n if C >= 4 else np.nan

    return out

def _probe_total_len(h5_paths: list[str]) -> int:
    tot = 0
    for p in h5_paths:
        with h5py.File(p, "r", libver="latest", swmr=True) as f:
            if "imgs" not in f:
                raise KeyError(f"'imgs' dataset missing in {p}")
            tot += int(f["imgs"].shape[0])
    return tot

def build_background_loader(
    h5_path: str,
    batch_size: int,
    workers: int,
    distributed: bool,
    device: torch.device,
):
    bg_ds = DiskSimCellsDataset(h5_path)

    bg_sampler = DistributedSampler(
        bg_ds, shuffle=True, drop_last=True
    ) if distributed else None

    pin_mem = (device.type == "cuda")
    bg_dl = DataLoader(
        bg_ds,
        batch_size=batch_size,
        shuffle=(bg_sampler is None),
        sampler=bg_sampler,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=(workers > 0),
        prefetch_factor=2,
        drop_last=True,
        collate_fn=collate_no_meta,
    )
    return bg_ds, bg_dl, bg_sampler

def build_h5_loaders(
    h5_path: str,
    batch_size: int,
    workers: int,
    seed: int,
    train_pct: float,
    val_pct: Optional[float],
    distributed: bool,
    device: torch.device,
):
    # discover files and build a concatenated base dataset
    base_ds = DiskSimCellsDataset(h5_path)
    N = len(base_ds)
    if N == 0:
        raise RuntimeError("Empty HDF5 dataset (no samples).")

    # split by shuffled indices (deterministic per seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)

    train_len = train_pct*N
    if val_pct is None:
        val_pct = 1-train_pct
    val_len=val_pct*N

    if val_len > 700:
        val_len = 700
        train_len = N - 700

    v_n = min(int(val_len), N)
    t_n = min(int(train_len), max(0, N - v_n))
    if t_n + v_n > N:
        v_n = min(v_n, N - t_n)

    train_idx = perm[:t_n]
    val_idx   = perm[t_n:t_n + v_n]

    train_ds = Subset(base_ds, train_idx)
    val_ds = Subset(base_ds, val_idx)

    # samplers for DDP
    train_sampler = DistributedSampler(train_ds, shuffle=True,  drop_last=True)  if distributed else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if distributed else None

    pin_mem = (device.type == "cuda")
    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,
        sampler=train_sampler,
        num_workers=workers, pin_memory=pin_mem,
        persistent_workers=(workers > 0),
        prefetch_factor=2,
        drop_last=True,
        collate_fn=collate_no_meta,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=workers, pin_memory=pin_mem,
        persistent_workers=(workers > 0),
        prefetch_factor=2,
        drop_last=False,
        collate_fn=collate_no_meta,
    )
    return train_ds, val_ds, train_dl, val_dl

# --------------------- main train() ---------------------

def train(
    out_dir="./models/",
    h5_path="./image_datasets/",
    epochs=500,
    batch_size=64,                 # per-GPU batch
    lr=1e-3,
    weight_decay=1e-4,
    workers=8,
    target: int = 512,
    mode: Literal["pad_resize", "crop_well_resize", "tiles"] = "crop_well_resize",
    train_pct=0.9,
    val_pct=None,
    seed=187,
    amp=True,
    unet_mode: Literal["small", "medium", "large"] = "small",
    bg_mix_frac: float = 0.075,
    w_bce=0.3,
    w_dice=0.7,
    max_steps_per_epoch: int | None = 250,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-5,
    warmup_epochs: int = 10,            # channel-weight warmup epochs
    lr_warmup_epochs: int = 3,          # LR warmup
    lr_warmup_start_factor: float = 1e-2,
    grad_clip_norm: float | None = 1.0,
):
    os.makedirs(out_dir, exist_ok=True)

    background_h5_path = os.path.dirname(h5_path)
    background_h5_path = os.path.join(background_h5_path, "background_dataset.h5")

    # DDP bootstrap
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    is_rank0 = (ddp_rank() == 0)

    # seeds per-rank
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)

    # AMP setup
    use_amp = bool(amp and device.type == "cuda")
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    scaler = GradScaler(enabled=(use_amp and not use_bf16))

    train_ds, val_ds, train_dl, val_dl = build_h5_loaders(
        h5_path=h5_path,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        train_pct=train_pct,
        val_pct=val_pct,
        distributed=distributed,
        device=device,
    )

    bg_dl = None
    bg_sampler = None

    if background_h5_path is not None and bg_mix_frac > 0.0:
        _, bg_dl, bg_sampler = build_background_loader(
            h5_path=background_h5_path,
            batch_size=batch_size,
            workers=workers,
            distributed=distributed,
            device=device,
        )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True,  drop_last=True)  if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False) if distributed else None

    # Model: out_channels matches dataset targets (cell, boundary, center, energy)
    n_out = 4
    if unet_mode == "small":
        model = build_unet_cpu_small(in_channels=3, out_channels=n_out).to(device)
    elif unet_mode == "medium":
        model = build_unet_cpu_medium(in_channels=3, out_channels=n_out).to(device)
    elif unet_mode == "large":
        model = build_unet_cpu_large(in_channels=3, out_channels=n_out).to(device)
    else:
        raise ValueError(f"Unknown unet mode '{unet_mode}'")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = LinearLR(opt, start_factor=lr_warmup_start_factor, total_iters=max(1, lr_warmup_epochs))
    cos  = CosineAnnealingLR(opt, T_max=max(1, epochs - lr_warmup_epochs))
    sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[lr_warmup_epochs])

    stopper = EarlyStopper(patience=early_stop_patience, min_delta=early_stop_min_delta)

    # progress bar
    epoch_bar = _tqdm(range(1, epochs+1), desc="epochs", position=0, disable=not is_rank0)

    best_val = float("inf")
    
    tag = f"{mode}_S{int(target)}_seed{int(seed)}"
    best_path = os.path.join(out_dir, f"best_{unet_mode}_{tag}.pth")
    log_path   = os.path.join(out_dir, f"log_{unet_mode}_{tag}.jsonl")

    if is_rank0:
        run_meta = {
            "unet_mode": unet_mode,
            "mode": mode,
            "target": int(target),
            "seed": int(seed),
            "world_size": ddp_world_size(),
            "epochs": int(epochs),
            "batch_size_per_gpu": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "amp": bool(amp),
            "out_dir": out_dir,
            "best_path": best_path,
            "log_path": log_path,
        }
        if device.type == "cuda":
            run_meta["gpu_name"] = torch.cuda.get_device_name(device)
        with open(os.path.join(out_dir, f"run_meta_{unet_mode}_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

    for ep in epoch_bar:

        cur_lr = opt.param_groups[0]["lr"]
        if is_rank0:
            print(f"epoch {ep} | lr={cur_lr:.6g}")

        # DDP sampler epoch
        sampler = getattr(train_dl, "sampler", None)
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(ep)

        t0 = time.time()

        # per-epoch channel weights
        weights = make_weight_schedule(C=n_out, epoch=ep, warmup_epochs=warmup_epochs)

        if bg_sampler is not None:
            bg_sampler.set_epoch(ep)

        tr_local = train_epoch(
            model, train_dl, opt, scaler, device, use_amp,
            weights=weights, w_bce=w_bce, w_dice=w_dice,
            show_bar=is_rank0, max_steps=max_steps_per_epoch,
            grad_clip_norm=grad_clip_norm,
            bg_loader=bg_dl,
            bg_mix_frac=bg_mix_frac,
        )
        va_local = eval_epoch(
            model, val_dl, device, use_amp,
            weights=weights, w_bce=w_bce, w_dice=w_dice,
            show_bar=is_rank0
        )

        tr = reduce_mean_dict(tr_local, device)
        va = reduce_mean_dict(va_local, device)

        sched.step()

        if is_rank0:
            state = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())
            # epoch_path = os.path.join(out_dir, f"{unet_mode}_{tag}_epoch_{ep}.pth")
            # torch.save(state, epoch_path)

            # use UNWEIGHTED val loss for model selection
            sel = va["loss_unweighted"]
            if sel < best_val:
                best_val = sel
                torch.save(state, best_path)

            post = {
                # train
                "tr_loss_w": f"{tr['loss_weighted']:.4f}",
                "tr_loss_u": f"{tr['loss_unweighted']:.4f}",
                "tr_l_cell": f"{tr['loss_cell']:.4f}",
                "tr_l_bound": f"{tr['loss_bound']:.4f}",
                "tr_l_center": f"{tr.get('loss_center', float('nan')):.4f}",
                "tr_l_energy": f"{tr.get('loss_energy', float('nan')):.4f}",
                # val
                "va_loss_w": f"{va['loss_weighted']:.4f}",
                "va_loss_u": f"{va['loss_unweighted']:.4f}",
                "best_u": f"{best_val:.4f}",
                "sec": f"{time.time()-t0:.1f}",
                "bs/gpu": batch_size,
                "gpus": ddp_world_size(),
                "w": str([round(w,3) for w in weights]),
                "mode": mode,
                "target": int(target),
            }
            # classic dice logs
            for k_alias, k in [("tr_d_cell","dice_cell"), ("tr_d_bound","dice_bound"),
                               ("tr_d_center","dice_center"), ("tr_d_energy","dice_energy")]:
                if k in tr:
                    post[k_alias] = f"{tr[k]:.3f}"
            for k_alias, k in [("va_d_cell","dice_cell"), ("va_d_bound","dice_bound"),
                               ("va_d_center","dice_center"), ("va_d_energy","dice_energy")]:
                if k in va:
                    post[k_alias] = f"{va[k]:.3f}"
            # new metrics (val)
            for k in ["bound_f1_tol2","center_f1_r10","energy_rmse","energy_pearson"]:
                if k in va:
                    post[k] = f"{va[k]:.3f}"

            epoch_bar.set_postfix(post)

            rec = {
                "epoch": ep,
                "train": tr,          # contains per-head + weighted/unweighted
                "val": va,            # contains per-head + weighted/unweighted
                "lr": float(opt.param_groups[0]["lr"]),
                "time_sec": round(time.time()-t0, 2),
                "best_val_unweighted": float(best_val),
                "world_size": ddp_world_size(),
                "batch_size_per_gpu": batch_size,
                "weights": weights,
                "mode": mode,
                "target": int(target),
                "unet_mode": unet_mode,
                # "checkpoint": epoch_path,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        # early stop on UNWEIGHTED val loss
        stop_local = stopper.step(va["loss_unweighted"])
        stop_all = _ddp_broadcast_bool(stop_local if is_rank0 else False, device)
        if stop_all:
            if is_rank0:
                print(f"Early stopping at epoch {ep} (best unweighted val loss: {best_val:.6f})")
            if ddp_is_active():
                dist.barrier()
            break

        if ddp_is_active():
            dist.barrier()

    if is_rank0:
        print(f"\nDone. Best val loss: {best_val:.4f}\nBest: {best_path}")

    if ddp_is_active():
        dist.barrier()
        dist.destroy_process_group()

    return best_path if is_rank0 else None

