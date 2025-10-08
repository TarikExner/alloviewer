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
from typing import Literal, List

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

# --------------------- losses ---------------------

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

def bce_dice_multi(
    logits: torch.Tensor,   # [B,C,H,W]
    target: torch.Tensor,   # [B,C,H,W]
    weights: List[float],
    w_bce: float = 0.3,
    w_dice: float = 0.7,
) -> torch.Tensor:
    """Sum over channels: w_c * (0.3 * BCE + 0.7 * Dice)."""
    assert logits.shape[:2] == target.shape[:2], "logits/target shape mismatch"
    C = logits.shape[1]
    assert len(weights) == C, f"weights length {len(weights)} != channels {C}"
    probs = torch.sigmoid(logits)
    total = 0.0
    for c in range(C):
        bce = F.binary_cross_entropy_with_logits(logits[:, c:c+1], target[:, c:c+1])
        d   = dice_loss_from_probs(probs[:, c:c+1], target[:, c:c+1])
        total = total + weights[c] * (w_bce * bce + w_dice * d)
    return total

# --------------------- schedulers ---------------------

def make_weight_schedule(C: int, epoch: int, warmup_epochs: int = 5) -> List[float]:
    """
    Returns per-channel weights for the current epoch.

    Channel order convention:
      0: cell, 1: boundary, 2: center (optional), 3: energy (optional)
    """
    if epoch <= warmup_epochs:
        post = [1.0, 1.5, 1.5, 1.5]
        return  post[:C]

    # base after warmup for up to 4 channels
    post = [1.0, 1.3, 0.5, 0.5]
    # truncate to C channels
    return post[:C]

# --------------------- training loops ---------------------

class EarlyStopper:
    def __init__(self, patience=2, min_delta=1e-4):
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

def train_epoch(model, loader, opt, scaler, device, use_amp, weights, w_bce, w_dice, show_bar, max_steps=None):
    model.train()
    running_loss = 0.0
    running_dice_cell  = 0.0
    running_dice_bound = 0.0
    running_dice_center = 0.0
    running_dice_energy = 0.0
    steps = 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    pbar = _tqdm(loader, desc="train", position=1, disable=not show_bar)

    for step_idx, (img, tgt, _) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)  # [B,C,H,W]
            loss = bce_dice_multi(logits, tgt, weights, w_bce=w_bce, w_dice=w_dice)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            if tgt.shape[1] >= 1:
                running_dice_cell  += dice_from_probs(probs[:,0:1], tgt[:,0:1])
            if tgt.shape[1] >= 2:
                running_dice_bound += dice_from_probs(probs[:,1:2], tgt[:,1:2])
            if tgt.shape[1] >= 3:
                running_dice_center += dice_from_probs(probs[:,2:3], tgt[:,2:3])
            if tgt.shape[1] >= 4:
                running_dice_energy += dice_from_probs(probs[:,3:4], tgt[:,3:4])

        running_loss += float(loss.item())
        steps += 1

        if show_bar:
            post = {"loss": f"{running_loss/steps:.4f}"}
            if tgt.shape[1] >= 1:
                post["dice_cell"] = f"{running_dice_cell/steps:.3f}"
            if tgt.shape[1] >= 2:
                post["dice_bound"] = f"{running_dice_bound/steps:.3f}"
            if tgt.shape[1] >= 3:
                post["dice_center"] = f"{running_dice_center/steps:.3f}"
            if tgt.shape[1] >= 4:
                post["dice_energy"] = f"{running_dice_energy/steps:.3f}"
            pbar.set_postfix(post)

        if (max_steps is not None) and (step_idx >= int(max_steps)):
            break

    out = {"loss": running_loss / max(1, steps)}
    if steps and tgt.shape[1] >= 1:
        out["dice_cell"]   = running_dice_cell / steps
    if steps and tgt.shape[1] >= 2:
        out["dice_bound"]  = running_dice_bound / steps
    if steps and tgt.shape[1] >= 3:
        out["dice_center"] = running_dice_center / steps
    if steps and tgt.shape[1] >= 4:
        out["dice_energy"] = running_dice_energy / steps
    return out

@torch.no_grad()
def eval_epoch(model, loader, device, use_amp, weights, w_bce, w_dice, show_bar):
    model.eval()
    loss_sum = 0.0
    dice_cell_sum   = 0.0
    dice_bound_sum  = 0.0
    dice_center_sum = 0.0
    dice_energy_sum = 0.0
    n = 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    pbar = _tqdm(loader, desc="evaluation", position=2, disable=not show_bar)

    for img, tgt, _ in pbar:
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)
            loss = bce_dice_multi(logits, tgt, weights, w_bce=w_bce, w_dice=w_dice)
            probs = torch.sigmoid(logits)

        loss_sum += float(loss.item())
        if tgt.shape[1] >= 1:
            dice_cell_sum  += dice_from_probs(probs[:,0:1], tgt[:,0:1])
        if tgt.shape[1] >= 2:
            dice_bound_sum += dice_from_probs(probs[:,1:2], tgt[:,1:2])
        if tgt.shape[1] >= 3:
            dice_center_sum += dice_from_probs(probs[:,2:3], tgt[:,2:3])
        if tgt.shape[1] >= 4:
            dice_energy_sum += dice_from_probs(probs[:,3:4], tgt[:,3:4])
        n += 1

        if show_bar:
            post = {"loss": f"{loss_sum/n:.4f}"}
            if tgt.shape[1] >= 1:
                post["dice_cell"]   = f"{dice_cell_sum/n:.3f}"
            if tgt.shape[1] >= 2:
                post["dice_bound"]  = f"{dice_bound_sum/n:.3f}"
            if tgt.shape[1] >= 3:
                post["dice_center"] = f"{dice_center_sum/n:.3f}"
            if tgt.shape[1] >= 4:
                post["dice_energy"] = f"{dice_energy_sum/n:.3f}"
            pbar.set_postfix(post)

    out = {"loss": loss_sum / max(1, n)}
    if n and tgt.shape[1] >= 1:
        out["dice_cell"]   = dice_cell_sum / n
    if n and tgt.shape[1] >= 2:
        out["dice_bound"]  = dice_bound_sum / n
    if n and tgt.shape[1] >= 3:
        out["dice_center"] = dice_center_sum / n
    if n and tgt.shape[1] >= 4:
        out["dice_energy"] = dice_energy_sum / n
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
    epochs=50,
    batch_size=16,                 # per-GPU batch
    lr=1e-3,
    weight_decay=1e-4,
    workers=4,
    tile_size=512,
    train_len=100_000,
    val_len=2000,
    seed=187,
    amp=True,
    unet_mode: Literal["small", "medium", "large"] = "small",
    w_bce=0.3,
    w_dice=0.7,
    max_steps_per_epoch: int | None = 250,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-5,
    warmup_epochs: int = 5
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

    # Data
    train_ds = SimCellsDataset(length=train_len, tile_size=tile_size, rng_seed=seed,
                               sim_fn=simulate_image, add_center=True, add_energy=True)
    val_ds   = SimCellsDataset(length=val_len, tile_size=tile_size, rng_seed=seed+1,
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

    # Model: out_channels matches dataset targets
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    stopper = EarlyStopper(patience=early_stop_patience, min_delta=early_stop_min_delta)

    # progress bar
    epoch_bar = _tqdm(range(1, epochs+1), desc="epochs", position=0, disable=not is_rank0)

    best_val = float("inf")
    best_path = os.path.join(out_dir, f"best_{unet_mode}.pth")

    for ep in epoch_bar:
        # set epoch on sampler
        sampler = getattr(train_dl, "sampler", None)
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(ep)

        t0 = time.time()

        # weight schedule for this epoch
        weights = make_weight_schedule(C=n_out, epoch=ep, warmup_epochs=warmup_epochs)

        tr_local = train_epoch(
            model, train_dl, opt, scaler, device, use_amp,
            weights=weights, w_bce=w_bce, w_dice=w_dice,
            show_bar=is_rank0, max_steps=max_steps_per_epoch
        )
        va_local = eval_epoch(
            model, val_dl, device, use_amp,
            weights=weights, w_bce=w_bce, w_dice=w_dice,
            show_bar=is_rank0
        )

        # reduce metrics
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
            if "dice_cell" in tr:
                post["tr_d_cell"] = f"{tr['dice_cell']:.3f}"
            if "dice_bound" in tr:
                post["tr_d_bound"] = f"{tr['dice_bound']:.3f}"
            if "dice_cell" in va:
                post["va_d_cell"] = f"{va['dice_cell']:.3f}"
            if "dice_bound" in va:
                post["va_d_bound"] = f"{va['dice_bound']:.3f}"

            epoch_bar.set_postfix(post)

            with open(os.path.join(out_dir, f"log_{unet_mode}.jsonl"), "a", encoding="utf-8") as f:
                rec = {
                    "epoch": ep, "train": tr, "val": va,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "time_sec": round(time.time()-t0, 2),
                    "best_val": float(best_val),
                    "world_size": ddp_world_size(),
                    "batch_size_per_gpu": batch_size,
                    "weights": weights,
                }
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

