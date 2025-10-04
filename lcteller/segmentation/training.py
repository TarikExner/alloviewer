import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from typing import Literal

from . import (SimCellsDataset, build_unet_cpu_small, simulate_image,
               build_unet_cpu_medium, build_unet_cpu_large)

def _tqdm(iterable, desc, position, total=None, disable=None):
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.1,
        position=position,
        disable=disable
    )
def ddp_is_active():
    return dist.is_available() and dist.is_initialized()

def _ddp_broadcast_bool(flag: bool, device):
    """Broadcast a boolean stop flag from rank 0 to all ranks (no-op if not DDP)."""
    if not ddp_is_active():
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    # by convention rank 0 writes its value; others keep whatever, then we broadcast 0->all
    if ddp_rank() == 0:
        t[...] = 1 if flag else 0
    torch.distributed.broadcast(t, src=0)
    return bool(int(t.item()))

def ddp_world_size():
    return dist.get_world_size() if ddp_is_active() else 1

def ddp_rank():
    return dist.get_rank() if ddp_is_active() else 0

def reduce_mean_scalar(val: float, device):
    """All-reduce mean for a single float."""
    if not ddp_is_active():
        return float(val)
    t = torch.tensor([val], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / ddp_world_size()).item())

def reduce_mean_dict(d: dict, device):
    if not ddp_is_active():
        return d
    return {k: reduce_mean_scalar(float(v), device) for k, v in d.items()}

def dice_loss_from_probs(probs, target, eps=1e-6):
    # probs/target: [N,C,H,W]
    num = 2 * (probs * target).sum(dim=(0,2,3)) + eps
    den = (probs + target).sum(dim=(0,2,3)) + eps
    return (1 - num / den).mean()

def dice_from_probs(probs, target, eps=1e-6):
    # per-batch mean dice (for logging), probs/target: [N,1,H,W]
    num = 2 * (probs * target).sum(dim=(0,2,3))
    den = (probs + target).sum(dim=(0,2,3)) + eps
    return (num / den).mean().item()

def bce_dice_loss(logits, target, w_bce=0.3, w_dice=0.7):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    d    = dice_loss_from_probs(probs, target)
    return w_bce * bce + w_dice * d

def bce_dice_loss_weighted_channels(logits, target, w_cell=1.0, w_bound=1.2, w_bce=0.3, w_dice=0.7):
    loss_cell  = bce_dice_loss(logits[:,0:1], target[:,0:1], w_bce, w_dice)
    loss_bound = bce_dice_loss(logits[:,1:2], target[:,1:2], w_bce, w_dice)
    return w_cell * loss_cell + w_bound * loss_bound

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
            return False  # keep training
        self.bad += 1
        return self.bad >= self.patience  # True => stop

def train_epoch(model, loader, opt, scaler, device, use_amp, loss_mode, w_cell, w_bound, w_bce, w_dice, show_bar, max_steps=None):
    model.train()
    running_loss = 0.0
    running_dice_cell = 0.0
    running_dice_bound = 0.0
    steps = 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    pbar = _tqdm(loader, desc="train", position=1, disable=not show_bar)

    for step_idx, (img, tgt, _) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)
            if loss_mode == "joint":
                loss = bce_dice_loss(logits, tgt, w_bce=w_bce, w_dice=w_dice)
            else:
                loss = bce_dice_loss_weighted_channels(
                    logits, tgt, w_cell=w_cell, w_bound=w_bound, w_bce=w_bce, w_dice=w_dice
                )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dc = dice_from_probs(probs[:,0:1], tgt[:,0:1])
            db = dice_from_probs(probs[:,1:2], tgt[:,1:2])

        running_loss += float(loss.item())
        running_dice_cell  += dc
        running_dice_bound += db
        steps += 1

        if show_bar:
            pbar.set_postfix({
                "loss": f"{running_loss/steps:.4f}",
                "dice_cell": f"{running_dice_cell/steps:.3f}",
                "dice_bound": f"{running_dice_bound/steps:.3f}"
            })

        if (max_steps is not None) and (step_idx >= int(max_steps)):
            break

    out = {
        "loss": running_loss / max(1, steps),
        "dice_cell": running_dice_cell / max(1, steps),
        "dice_bound": running_dice_bound / max(1, steps),
    }
    return out


@torch.no_grad()
def eval_epoch(model, loader, device, use_amp, loss_mode, w_cell, w_bound, w_bce, w_dice, show_bar):
    model.eval()
    loss_sum = 0.0
    dice_cell_sum = 0.0
    dice_bound_sum = 0.0
    n = 0

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    pbar = _tqdm(loader, desc="evaluation", position=2, disable=not show_bar)
    for img, tgt, _ in pbar:
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
            logits = model(img)
            if loss_mode == "joint":
                loss = bce_dice_loss(logits, tgt, w_bce=w_bce, w_dice=w_dice)
            else:
                loss = bce_dice_loss_weighted_channels(
                    logits, tgt, w_cell=w_cell, w_bound=w_bound, w_bce=w_bce, w_dice=w_dice
                )
            probs = torch.sigmoid(logits)

        dc = dice_from_probs(probs[:,0:1], tgt[:,0:1])
        db = dice_from_probs(probs[:,1:2], tgt[:,1:2])

        loss_sum += float(loss.item())
        dice_cell_sum  += dc
        dice_bound_sum += db
        n += 1

        if show_bar:
            pbar.set_postfix({
                "loss": f"{loss_sum/n:.4f}",
                "dice_cell": f"{dice_cell_sum/n:.3f}",
                "dice_bound": f"{dice_bound_sum/n:.3f}"
            })

    out = {
        "loss": loss_sum / max(1, n),
        "dice_cell": dice_cell_sum / max(1, n),
        "dice_bound": dice_bound_sum / max(1, n),
    }
    return out

def collate_no_meta(batch):
    """
    batch: list of (img_t, tgt_t, extras)
    Stacks img and target; stacks extras['instance_labels']; keeps meta as a list.
    """
    imgs, tgts, exs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)          # [B,3,H,W]
    tgts = torch.stack(tgts, dim=0)          # [B,2,H,W]

    # instance labels are tensors of HxW; safe to stack
    inst = torch.stack([e["instance_labels"] for e in exs], dim=0)  # [B,H,W]

    metas = [e["meta"] for e in exs]         # keep as list (no stacking)

    extras_out = {"instance_labels": inst, "meta": metas}
    return imgs, tgts, extras_out

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
    loss_mode="weighted",
    unet_mode: Literal["small", "medium", "large"] = "small",
    w_cell=1.0,
    w_bound=1.3,
    w_bce=0.3,
    w_dice=0.7,
    max_steps_per_epoch: int | None = 250,
    early_stop_patience: int = 2,
    early_stop_min_delta: float = 1e-4, 
):
    os.makedirs(out_dir, exist_ok=True)

    # DDP bootstrap
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.cuda.device_count()
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
    train_ds = SimCellsDataset(length=train_len, tile_size=tile_size, rng_seed=seed, sim_fn=simulate_image)
    val_ds = SimCellsDataset(length=val_len, tile_size=tile_size, rng_seed=seed+1, sim_fn=simulate_image)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True,  drop_last=True)  if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False) if distributed else None

    pin_mem = (device.type == "cuda")
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=(train_sampler is None),
                          sampler=train_sampler,
                          num_workers=workers,
                          pin_memory=pin_mem,
                          persistent_workers=(workers > 0),
                          drop_last=True,
                          collate_fn=collate_no_meta)
    val_dl   = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False,
                          sampler=val_sampler,
                          num_workers=workers,
                          pin_memory=pin_mem,
                          persistent_workers=(workers > 0),
                          drop_last=False,
                          collate_fn=collate_no_meta)

    if unet_mode == "small":
        model = build_unet_cpu_small(in_channels=3, out_channels=2).to(device)
    elif unet_mode == "medium":
        model = build_unet_cpu_medium(in_channels=3, out_channels=2).to(device)
    elif unet_mode == "large":
        model = build_unet_cpu_large(in_channels=3, out_channels=2).to(device)
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
        # safer sampler epoch set
        sampler = getattr(train_dl, "sampler", None)
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(ep)

        t0 = time.time()

        tr_local = train_epoch(
            model, train_dl, opt, scaler, device, use_amp, loss_mode,
            w_cell, w_bound, w_bce, w_dice, show_bar=is_rank0,
            max_steps=max_steps_per_epoch
        )
        va_local = eval_epoch(
            model, val_dl, device, use_amp, loss_mode,
            w_cell, w_bound, w_bce, w_dice, show_bar=is_rank0
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

            epoch_bar.set_postfix({
                "tr_loss": f"{tr['loss']:.4f}",
                "tr_d_cell": f"{tr['dice_cell']:.3f}",
                "tr_d_bound": f"{tr['dice_bound']:.3f}",
                "va_loss": f"{va['loss']:.4f}",
                "va_d_cell": f"{va['dice_cell']:.3f}",
                "va_d_bound": f"{va['dice_bound']:.3f}",
                "best": f"{best_val:.4f}",
                "sec": f"{time.time()-t0:.1f}",
                "bs/gpu": batch_size,
                "gpus": world_size
            })
            with open(os.path.join(out_dir, f"log_{unet_mode}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "epoch": ep, "train": tr, "val": va,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "time_sec": round(time.time()-t0, 2),
                    "best_val": float(best_val),
                    "world_size": world_size,
                    "batch_size_per_gpu": batch_size
                }) + "\n")


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
