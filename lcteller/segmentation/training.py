import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from typing import Literal, List, Optional, Tuple
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from skimage.morphology import skeletonize

from scipy import ndimage as ndi
import cv2
from scipy.optimize import linear_sum_assignment

from . import (
    SimCellsDataset, simulate_image,
    build_unet_cpu_small, build_unet_cpu_medium, build_unet_cpu_large
)

# --------------------- utils ---------------------

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
    """
    yy, xx = np.ogrid[:H, :W]
    B = len(meta_list)
    mask = np.zeros((B, H, W), dtype=np.float32)
    r2 = radius * radius
    for i, m in enumerate(meta_list):
        centers = m.get("centers", []) or []
        acc = np.zeros((H, W), dtype=np.uint8)
        for (y, x) in centers:
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
                             thr: float = 0.2,
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

def center_metrics_np(center_pred: np.ndarray, gt_centers: List[Tuple[int,int]],
                      peak_thr: float = 0.2, nms_dist: int = 3, match_radius: int = 5) -> dict:
    preds = _nms_peaks_np(center_pred, thr=peak_thr, min_dist=nms_dist)
    pred_xy = [(y, x) for (y, x, _) in preds]
    if len(pred_xy) == 0 or len(gt_centers) == 0:
        TP = 0
        FP = len(pred_xy)
        FN = len(gt_centers)
        prec = TP / (TP + FP + 1e-8)
        rec  = TP / (TP + FN + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        return {"precision":prec, "recall":rec, "f1":f1, "median_err_px":np.nan,
                "n_pred":len(pred_xy), "n_gt":len(gt_centers), "n_tp":0}
    P = np.array(pred_xy, dtype=float)
    G = np.array(gt_centers, dtype=float)
    d2 = ((P[:, None, :] - G[None, :, :]) ** 2).sum(axis=2)
    D = np.sqrt(d2)
    row_ind, col_ind = linear_sum_assignment(D)
    matches = []
    unmatched_p = set(range(len(pred_xy)))
    unmatched_g = set(range(len(gt_centers)))
    for r, c in zip(row_ind, col_ind):
        if D[r, c] <= match_radius:
            matches.append((r, c, float(D[r, c])))
            unmatched_p.discard(r)
            unmatched_g.discard(c)
    TP = len(matches)
    FP = len(unmatched_p)
    FN = len(unmatched_g)
    prec = TP / (TP + FP + 1e-8)
    rec  = TP / (TP + FN + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    med_err = float(np.median([m[2] for m in matches])) if matches else np.nan
    return {"precision":float(prec), "recall":float(rec), "f1":float(f1), "median_err_px":med_err,
            "n_pred":len(pred_xy), "n_gt":len(gt_centers), "n_tp":TP}

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

def make_weight_schedule(C: int, epoch: int, warmup_epochs: int = 10) -> List[float]:
    """
    Channel order:
      0: cell, 1: boundary, 2: center, 3: energy
    Warmup -> [1.0, 1.8, 2.0, 2.0]
    After  -> [1.0, 1.3, 0.5, 0.5]
    """
    if epoch <= warmup_epochs:
        post = [1.0, 1.8, 2.0, 2.0]
        return post[:C]
    post = [1.0, 1.3, 0.5, 0.5]
    return post[:C]

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
                       bound_ring_width: int = 1,
                       bound_soft_band: int = 2,
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
    bound_soft = make_soft_boundary_batch(inst_b, ring_width=bound_ring_width,
                                          soft_band=bound_soft_band, sigma=bound_sigma, device=device)  # [B,1,H,W]
    cell_mask = tgt[:, 0:1]  # [B,1,H,W] float
    metas = extras.get("meta", None)
    center_allow = None
    if metas is not None and isinstance(metas, list):
        center_allow = make_outside_mask_centers_batch(metas, H, W, radius=center_outside_radius, device=device)  # [B,1,H,W]
    return bound_soft, cell_mask, center_allow

def train_epoch(model,
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
                ):
    model.train()
    running_loss = 0.0
    running_dice_cell  = 0.0
    running_dice_bound = 0.0
    running_dice_center = 0.0
    running_dice_energy = 0.0
    steps = 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    total = (min(max_steps, len(loader)) if (max_steps is not None) else len(loader))
    pbar = _tqdm(loader, desc="train", position=1, disable=not show_bar, total=total)

    for step_idx, (img, tgt, extras) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)            # [B,3,H,W]
        tgt = tgt.to(device, non_blocking=True)            # [B,C,H,W]
        B, C, H, W = tgt.shape

        bound_soft, cell_mask, center_allow = _build_aux_targets(
            extras, tgt, device,
            center_outside_radius=6,
            bound_ring_width=1, bound_soft_band=2, bound_sigma=1.0
        )

        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)  # [B,C,H,W]
            probs  = torch.sigmoid(logits)

            # --- base losses ---
            loss_cell = bce_dice_loss(logits[:,0:1], tgt[:,0:1], w_bce=w_bce, w_dice=w_dice)
            loss_bound = bce_dice_loss(logits[:,1:2], bound_soft, w_bce=w_bce, w_dice=w_dice)
            loss_excl  = boundary_exclusion_loss(probs[:,0:1], bound_soft, weight=excl_weight)

            if C >= 3:
                loss_center = center_head_loss(
                    logits[:,2:3], tgt[:,2:3], cell_mask=cell_mask,
                    pos_weight=center_pos_weight, w_bce=center_w_bce, w_mse=center_w_mse,
                    sparsity_weight=center_sparsity_weight, outside_mask=center_allow,
                    count_weight=0.0
                )
            else:
                loss_center = torch.tensor(0.0, device=device)

            if C >= 4:
                loss_energy = energy_head_loss_masked(
                    logits[:,3:4], tgt[:,3:4], cell_mask=cell_mask,
                    w_l1=energy_w_l1, w_mse=energy_w_mse
                )
            else:
                loss_energy = torch.tensor(0.0, device=device)

            # --- anti-halo penalties (NEW) ---
            # thin ring just outside cells & far background mask (GPU-friendly via max_pool2d)
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
                ring_leak  = (ring_mask   * energy_prob).mean()
                far_leak   = (far_bg_mask * energy_prob).mean()
                loss_energy_halo = halo_w_energy_ring * ring_leak + halo_w_energy_far * far_leak

            total_loss = (
                weights[0] * loss_cell +
                (weights[1] if len(weights) > 1 else 0.0) * loss_bound +
                (weights[2] if len(weights) > 2 else 0.0) * loss_center +
                (weights[3] if len(weights) > 3 else 0.0) * loss_energy +
                loss_excl +
                loss_cell_halo +         # NEW
                loss_energy_halo         # NEW
            )

        scaler.scale(total_loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            running_dice_cell  += dice_from_probs(probs[:,0:1], tgt[:,0:1])
            running_dice_bound += dice_from_probs(probs[:,1:2], bound_soft)
            if C >= 3:
                running_dice_center += dice_from_probs(probs[:,2:3], tgt[:,2:3])
            if C >= 4:
                running_dice_energy += dice_from_probs(probs[:,3:4], tgt[:,3:4])

        running_loss += float(total_loss.item())
        steps += 1

        if show_bar:
            post = {"loss": f"{running_loss/steps:.4f}"}
            post["dice_cell"] = f"{running_dice_cell/steps:.3f}"
            post["dice_bound"] = f"{running_dice_bound/steps:.3f}"
            if C >= 3:
                post["dice_center"] = f"{running_dice_center/steps:.3f}"
            if C >= 4:
                post["dice_energy"] = f"{running_dice_energy/steps:.3f}"
            pbar.set_postfix(post)

        if (max_steps is not None) and (step_idx >= int(max_steps)):
            break

    out = {"loss": running_loss / max(1, steps)}
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
               center_match_radius_px: int = 5, center_thr: float = 0.2, center_nms: int = 3,
               bound_tol_px: int = 2, bound_thr: float = 0.2, bound_sweep: bool = False,
               halo_ring_px: int = 3, halo_far_px: int = 12):
    model.eval()
    loss_sum = 0.0
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
    n = 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    pbar = _tqdm(loader, desc="evaluation", position=2, disable=not show_bar)

    for img, tgt, extras in pbar:
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        B, C, H, W = tgt.shape

        bound_soft, cell_mask, center_allow = _build_aux_targets(
            extras, tgt, device,
            center_outside_radius=6,
            bound_ring_width=1, bound_soft_band=2, bound_sigma=1.0
        )

        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)
            probs  = torch.sigmoid(logits)

            loss_cell  = bce_dice_loss(logits[:,0:1], tgt[:,0:1], w_bce=w_bce, w_dice=w_dice)
            loss_bound = bce_dice_loss(logits[:,1:2], bound_soft, w_bce=w_bce, w_dice=w_dice)
            loss_excl  = boundary_exclusion_loss(probs[:,0:1], bound_soft, weight=0.2)

            if C >= 3:
                loss_center = center_head_loss(
                    logits[:,2:3], tgt[:,2:3], cell_mask=cell_mask,
                    pos_weight=10.0, w_bce=0.6, w_mse=0.4,
                    sparsity_weight=0.0, outside_mask=center_allow, count_weight=0.0
                )
            else:
                loss_center = torch.tensor(0.0, device=device)

            if C >= 4:
                loss_energy = energy_head_loss_masked(
                    logits[:,3:4], tgt[:,3:4], cell_mask=cell_mask,
                    w_l1=0.5, w_mse=0.5
                )
            else:
                loss_energy = torch.tensor(0.0, device=device)

            total_loss = (
                weights[0] * loss_cell +
                (weights[1] if len(weights) > 1 else 0.0) * loss_bound +
                (weights[2] if len(weights) > 2 else 0.0) * loss_center +
                (weights[3] if len(weights) > 3 else 0.0) * loss_energy +
                loss_excl
            )

        dice_cell_sum  += dice_from_probs(probs[:,0:1], tgt[:,0:1])
        dice_bound_sum += dice_from_probs(probs[:,1:2], bound_soft)
        if C >= 3:
            dice_center_sum += dice_from_probs(probs[:,2:3], tgt[:,2:3])
        if C >= 4:
            dice_energy_sum += dice_from_probs(probs[:,3:4], tgt[:,3:4])

        # --- anti-halo diagnostics (do NOT affect eval loss) ---
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

        # --- classic + new metrics (per-sample numpy) ---
        probs_np = probs.detach().float().cpu().numpy()  # [B,C,H,W]
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
                    energy_rmse_sum += ee["rmse"]; energy_rmse_n += 1
                if ee.get("valid_r", False) and np.isfinite(ee["pearson"]):
                    energy_r_sum += ee["pearson"]; energy_r_n += 1

        loss_sum += float(total_loss.item())
        n += 1

        if show_bar:
            post = {
                "loss": f"{loss_sum/n:.4f}",
                "dice_cell": f"{dice_cell_sum/n:.3f}",
                "dice_bound": f"{dice_bound_sum/n:.3f}",
            }
            if C >= 3:
                post["dice_center"] = f"{dice_center_sum/n:.3f}"
            if C >= 4:
                post["dice_energy"] = f"{dice_energy_sum/n:.3f}"
            pbar.set_postfix(post)

    out = {"loss": loss_sum / max(1, n)}
    if n:
        out["dice_cell"]   = dice_cell_sum / n
        out["dice_bound"]  = dice_bound_sum / n
        if C >= 3:
            out["dice_center"] = dice_center_sum / n
        if C >= 4:
            out["dice_energy"] = dice_energy_sum / n
        out["bound_f1_tol2"]   = boundF_sum / n_bound_valid if n_bound_valid > 0 else np.nan
        out["center_f1_r5"]    = centerF_sum / n_center_valid if n_center_valid > 0 else np.nan
        out["energy_rmse"]     = (energy_rmse_sum / energy_rmse_n) if energy_rmse_n > 0 else np.nan
        out["energy_pearson"]  = (energy_r_sum / energy_r_n) if energy_r_n > 0 else np.nan
        # anti-halo diagnostics
        out["ring_leak_cell"]   = ring_leak_cell_sum / n
        out["far_leak_cell"]    = far_leak_cell_sum / n
        out["ring_leak_energy"] = ring_leak_energy_sum / n if C >= 4 else np.nan
        out["far_leak_energy"]  = far_leak_energy_sum  / n if C >= 4 else np.nan

    return out

# --------------------- collate ---------------------

def collate_no_meta(batch):
    """
    batch: list of (img_t, tgt_t, extras)
    Stacks img and target; stacks extras['instance_labels']; keeps meta as a list.
    Works for any number of target channels.
    """
    imgs, tgts, exs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)                # [B,3,H,W]
    tgts = torch.stack(tgts, dim=0)                # [B,C,H,W]
    inst = torch.stack([e["instance_labels"] for e in exs], dim=0)  # [B,H,W]
    metas = [e["meta"] for e in exs]
    extras_out = {"instance_labels": inst, "meta": metas}
    return imgs, tgts, extras_out

# --------------------- main train() ---------------------

def train(
    out_dir="./models/",
    epochs=500,
    batch_size=64,                 # per-GPU batch
    lr=1e-3,
    weight_decay=1e-4,
    workers=8,
    tile_size=512,
    train_len=100_000,
    val_len=2000,
    seed=187,
    amp=True,
    unet_mode: Literal["small", "medium", "large"] = "small",
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

    # DDP bootstrap
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_rank0 = (ddp_rank() == 0)

    # seeds per-rank
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)

    # AMP setup
    use_amp = bool(amp and device.type == "cuda")
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    scaler = GradScaler(enabled=(use_amp and not use_bf16))

    # Data (keep Dataset simple; no special soft-boundary logic inside it)
    train_ds = SimCellsDataset(length=train_len, tile_size=tile_size, rng_seed=seed,
                               sim_fn=simulate_image, add_center=True, add_energy=True)
    val_ds   = SimCellsDataset(length=val_len,  tile_size=tile_size, rng_seed=seed+1,
                               sim_fn=simulate_image, add_center=True, add_energy=True)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True,  drop_last=True)  if distributed else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False) if distributed else None

    pin_mem = (device.type == "cuda")
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=(train_sampler is None),
                          sampler=train_sampler,
                          num_workers=workers, pin_memory=pin_mem,
                          persistent_workers=(workers > 0),
                          drop_last=True, collate_fn=collate_no_meta)
    val_dl   = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, sampler=val_sampler,
                          num_workers=workers, pin_memory=pin_mem,
                          persistent_workers=(workers > 0),
                          drop_last=False, collate_fn=collate_no_meta)

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
    best_path = os.path.join(out_dir, f"best_{unet_mode}.pth")

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

        tr_local = train_epoch(
            model, train_dl, opt, scaler, device, use_amp,
            weights=weights, w_bce=w_bce, w_dice=w_dice,
            show_bar=is_rank0, max_steps=max_steps_per_epoch,
            grad_clip_norm=grad_clip_norm
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
            torch.save(state, os.path.join(out_dir, f"{unet_mode}_epoch_{ep}.pth"))
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(state, best_path)

            post = {
                "tr_loss": f"{tr['loss']:.4f}",
                "va_loss": f"{va['loss']:.4f}",
                "best": f"{best_val:.4f}",
                "sec": f"{time.time()-t0:.1f}",
                "bs/gpu": batch_size,
                "gpus": ddp_world_size(),
                "w": str([round(w,3) for w in weights]),
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
            for k in ["bound_f1_tol2","center_f1_r5","energy_rmse","energy_pearson"]:
                if k in va:
                    post[k] = f"{va[k]:.3f}"

            epoch_bar.set_postfix(post)

            rec = {
                "epoch": ep, "train": tr, "val": va,
                "lr": float(opt.param_groups[0]["lr"]),
                "time_sec": round(time.time()-t0, 2),
                "best_val": float(best_val),
                "world_size": ddp_world_size(),
                "batch_size_per_gpu": batch_size,
                "weights": weights,
            }
            with open(os.path.join(out_dir, f"log_{unet_mode}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        stop_local = stopper.step(va["loss"])
        stop_all = _ddp_broadcast_bool(stop_local if is_rank0 else False, device)
        if stop_all:
            if is_rank0:
                print(f"Early stopping at epoch {ep} (best val loss: {best_val:.6f})")
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

