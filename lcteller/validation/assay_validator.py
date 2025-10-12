# assay_validator.py
from __future__ import annotations

import os
import json
import time
import hashlib
import uuid
import socket
import platform
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, wasserstein_distance
from skimage import measure

import pyarrow as pa
import pyarrow.parquet as pq

import torch

from tqdm import tqdm

from ..segmenter import SegmenterUNet, InstanceSegmenter
from ..config import UNET_CONFIG, INSTANCE_CONFIG
from ..segmentation import SimCellsDataset

DEFAULT_DATASET_CFG = dict(
    length=100000,          # will be capped by unet_validation(num_images=...)
    tile_size=512,
    n_cells=(10, 400),
    cell_diameter=(4, 28),
    frac_positive=(0.0, 1.0),
    blur_sigma=(0.0, 2.0),
    background_level=(0.0, 0.04),
    color_jitter=(0.0, 0.2),
    photon_level=(1500, 4000),
    boundary_width=2,
    aug_flip=True,
    aug_rot90=True,
    aug_gamma=(0.90, 1.12),
    rng_seed=123,
    add_center=True,
    add_energy=True,
    center_sigma=2.0,
    random_camera_rect=False,  # baseline; we also run a second sweep with True
    cam_src_side_range=(640, 1024),
    cam_aspect_ratio_range=(0.6, 1.6),
    cam_content_scale_range=(0.6, 0.95),
    cam_out_side=512,
    cam_dark_margin_bias=0.0,
    bound_ring_width=1,
    bound_soft_band=2,
    bound_sigma=1.0,
)

# -------------------- small utils --------------------
import pandas as pd
import numpy as np
import os

def _append_run_index(base_dir: str, run_dir: str, meta: dict, df_head: pd.DataFrame) -> None:
    """
    Append a row per (unet_mode, random_camera_rect) to a global runs_index.csv.

    Parameters
    ----------
    base_dir : str
        Root folder that contains run_* subfolders (e.g., "./validation_runs").
    run_dir : str
        Full path to the current run folder (e.g., "./validation_runs/run_121025").
    meta : dict
        Parsed JSON from run_meta.json. Must include "run_id" and "created_at".
    df_head : pd.DataFrame
        A small slice of the run's results (we usually load selected columns only)
        used to compute quick summaries, not the full dataset.
    """
    idx_path = os.path.join(base_dir, "runs_index.csv")

    # Try both new and old column names
    f1col = next((c for c in ("f1_0_5", "f1@0.5") if c in df_head.columns), None)
    ce_col = next((c for c in ("count_error_frac", "count_error_pct") if c in df_head.columns), None)

    # Group by mode and camera flag if present; fall back to mode only
    group_cols = [c for c in ["unet_mode", "random_camera_rect"] if c in df_head.columns]
    if not group_cols:
        group_cols = ["unet_mode"] if "unet_mode" in df_head.columns else []

    rows = []
    if group_cols:
        for keys, grp in df_head.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = {
                "run_id": meta.get("run_id"),
                "created_at": meta.get("created_at"),
                "run_dir": run_dir,
                "description": meta.get("description", ""),
                "tags": "|".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else "",
                "num_rows": int(len(grp)),
                "f1_0_5_mean": float(grp[f1col].mean()) if f1col else np.nan,
                "count_err_mean": float(grp[ce_col].mean()) if ce_col else np.nan,
            }
            # attach grouping keys as columns
            for col, val in zip(group_cols, keys):
                rec[col] = bool(val) if col == "random_camera_rect" else val
            rows.append(rec)
    else:
        # No grouping columns found; write a single summary row
        rows.append({
            "run_id": meta.get("run_id"),
            "created_at": meta.get("created_at"),
            "run_dir": run_dir,
            "description": meta.get("description", ""),
            "tags": "|".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else "",
            "num_rows": int(len(df_head)),
            "f1_0_5_mean": float(df_head[f1col].mean()) if f1col else np.nan,
            "count_err_mean": float(df_head[ce_col].mean()) if ce_col else np.nan,
        })

    r = pd.DataFrame(rows)

    if os.path.exists(idx_path):
        old = pd.read_csv(idx_path)
        pd.concat([old, r], ignore_index=True).to_csv(idx_path, index=False)
    else:
        r.to_csv(idx_path, index=False)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _date_id_ddmmyy() -> str:
    return datetime.datetime.now().strftime("%d%m%y")

def _next_run_id(base_dir: str) -> str:
    """
    Returns DDMMYY, or DDMMYY-001, DDMMYY-002, ... if a run with that ID already exists.
    """
    base = _date_id_ddmmyy()
    run_id = base
    k = 1
    while os.path.exists(os.path.join(base_dir, f"run_{run_id}")):
        run_id = f"{base}-{k:03d}"
        k += 1
    return run_id

def _hash_dict(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _count_instances(lab: np.ndarray) -> int:
    u = np.unique(lab)
    return int((u > 0).sum())

def _compact_labels(lab: np.ndarray) -> np.ndarray:
    lab = lab.astype(np.int32, copy=False)
    u = np.unique(lab)
    u = u[u > 0]
    if u.size == 0:
        return np.zeros_like(lab, dtype=np.int32)
    mapping = {int(old): i + 1 for i, old in enumerate(u)}
    out = np.zeros_like(lab, dtype=np.int32)
    inv = np.zeros(lab.max() + 1, dtype=np.int32) if lab.max() > 0 else np.zeros(1, dtype=np.int32)
    for old, new in mapping.items():
        if old < inv.size:
            inv[old] = new
    m = lab > 0
    out[m] = inv[lab[m]]
    return out

def _regionprops_fast(lab: np.ndarray) -> Dict[str, np.ndarray]:
    if lab.max() == 0:
        # Return empty arrays to avoid downstream errors
        return {
            "label": np.array([], dtype=int),
            "area": np.array([], dtype=float),
            "perimeter": np.array([], dtype=float),
            "eccentricity": np.array([], dtype=float),
            "solidity": np.array([], dtype=float),
            "centroid-0": np.array([], dtype=float),
            "centroid-1": np.array([], dtype=float),
            "circularity": np.array([], dtype=float),
        }
    props = measure.regionprops_table(
        lab,
        properties=("label", "area", "perimeter", "eccentricity", "solidity", "centroid")
    )
    perim = np.maximum(props["perimeter"], 1.0)
    circ = 4.0 * np.pi * props["area"] / (perim ** 2)
    props["circularity"] = circ
    return props

def _contingency_iou(gt: np.ndarray, pr: np.ndarray) -> Tuple[np.ndarray, List[int], List[int], Dict[int,int], Dict[int,int]]:
    gt_ids = np.unique(gt)[1:]
    pr_ids = np.unique(pr)[1:]
    gt_idx = {int(k): i for i, k in enumerate(gt_ids)}
    pr_idx = {int(k): i for i, k in enumerate(pr_ids)}
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return np.zeros((0, 0), dtype=np.float32), list(gt_ids), list(pr_ids), gt_idx, pr_idx

    pr_max = int(pr_ids.max()) + 1
    keys = gt.astype(np.int64) * pr_max + pr.astype(np.int64)
    mask = (gt > 0) & (pr > 0)
    if not np.any(mask):
        return np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32), list(gt_ids), list(pr_ids), gt_idx, pr_idx

    keys_fg = keys[mask].ravel()
    counts = np.bincount(keys_fg)
    nz = np.nonzero(counts)[0]
    inter = counts[nz]
    p_ids = (nz % pr_max).astype(np.int64)
    g_ids = (nz // pr_max).astype(np.int64)

    gt_area = np.bincount(gt.ravel(), minlength=int(gt_ids.max()) + 1).astype(np.int64)
    pr_area = np.bincount(pr.ravel(), minlength=int(pr_ids.max()) + 1).astype(np.int64)

    IoU = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    for gi, pi, inter_ in zip(g_ids, p_ids, inter):
        if gi == 0 or pi == 0:
            continue
        i = gt_idx.get(int(gi), None)
        j = pr_idx.get(int(pi), None)
        if i is None or j is None:
            continue
        u = gt_area[int(gi)] + pr_area[int(pi)] - int(inter_)
        if u > 0:
            IoU[i, j] = float(inter_) / float(u)
    return IoU, list(gt_ids), list(pr_ids), gt_idx, pr_idx

def _hungarian(iou: np.ndarray, thr: float) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int,float]]]:
    if iou.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), []
    cost = 1.0 - iou
    ri, cj = linear_sum_assignment(cost)
    pairs = [(int(r), int(c), float(iou[r, c])) for r, c in zip(ri, cj) if iou[r, c] >= thr]
    return ri, cj, pairs

def _sanitize_for_json(o):
    if isinstance(o, dict):
        return {str(k): _sanitize_for_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_for_json(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [_sanitize_for_json(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dict__"):
        try:
            return _sanitize_for_json(vars(o))
        except Exception:
            return str(o)
    return o


# -------------------- sim → "probabilities" (from tgt_t) --------------------
def _probs_from_targets(tgt_t: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Map SimCellsDataset target tensor [C,H,W] → dict of per-map arrays in [0,1] (best-effort).
    Channel order per dataset: 0=cell(0/1), 1=bound_soft, 2=center?, 3=energy?
    Energy is EDT-like; we normalize to [0,1] per image for compatibility.
    """
    t = tgt_t.detach().cpu().numpy().astype(np.float32)
    C, H, W = t.shape
    cell = np.clip(t[0], 0.0, 1.0)
    bound = np.clip(t[1], 0.0, 1.0)
    center = t[2] if C >= 3 else None
    energy = t[3] if C >= 4 else None

    out: Dict[str, np.ndarray] = {
        "cell": cell,
        "bound": bound,
    }
    if center is not None:
        out["center"] = np.clip(center.astype(np.float32), 0.0, 1.0)
    else:
        out["center"] = None

    if energy is not None:
        e = energy.astype(np.float32)
        emax = float(e.max())
        if emax > 1e-6:
            e = e / emax
        e = np.clip(e, 0.0, 1.0)
        out["energy"] = e
    else:
        out["energy"] = None

    return out


def _run_instance_segmenter_from_probs(inst_seg: InstanceSegmenter, probs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Build a seg_out-like dict and run InstanceSegmenter on it."""
    seg_out = {
        "probs": {
            "cell": probs["cell"],
            "bound": probs["bound"],
            "center": probs.get("center", None),
            "energy": probs.get("energy", None),
        },
        "cell_mask": (probs["cell"] >= 0.5).astype(np.uint8),  # convenience
        "instance_labels": None,
        "meta": {},
    }
    seg_out = inst_seg(seg_out, update_cell_mask=True)
    return seg_out


# -------------------- per-image evaluation --------------------
def _shape_distributions(lab_a: np.ndarray, lab_b: np.ndarray) -> Dict[str, float]:
    """KS and Wasserstein between instance shape summaries."""
    props_a = _regionprops_fast(lab_a)
    props_b = _regionprops_fast(lab_b)
    out: Dict[str, float] = {}

    def _stat(x: np.ndarray, y: np.ndarray, name: str):
        if x.size == 0 and y.size == 0:
            out[f"ks_{name}"] = 0.0
            out[f"ks_p_{name}"] = 1.0
            out[f"wd_{name}"] = 0.0
            return
        if x.size == 0 or y.size == 0:
            # maximal difference if one side empty
            out[f"ks_{name}"] = 1.0
            out[f"ks_p_{name}"] = 0.0
            out[f"wd_{name}"] = float(np.inf)
            return
        ks = ks_2samp(x, y, alternative="two-sided", mode="auto")
        wd = wasserstein_distance(x, y)
        out[f"ks_{name}"] = float(ks.statistic)
        out[f"ks_p_{name}"] = float(ks.pvalue)
        out[f"wd_{name}"] = float(wd)

    for key in ("area", "eccentricity", "solidity", "circularity"):
        xa = props_a[key].astype(np.float64)
        xb = props_b[key].astype(np.float64)
        _stat(xa, xb, key)

    return out


def evaluate_one_image(
    img_rgb: np.ndarray,
    tgt_t: torch.Tensor,
    meta: Dict[str, Any],
    unet: SegmenterUNet,
    shared_inst_seg: InstanceSegmenter,
    *,
    iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
    save_artifacts: bool = False,
    artifacts_dir: Optional[str] = None,
    image_id: str = "",
    derived_params: Optional[Dict[str, Any]] = None,
    mode_name: str = "",
) -> Dict[str, Any]:
    """
    GT instances are produced by running the SAME InstanceSegmenter on simulated targets (tgt_t).
    Pred instances are produced by UNet->probs and the SAME InstanceSegmenter.
    """
    # --- 0) build SIM "probs" and instances ---
    sim_probs = _probs_from_targets(tgt_t)
    sim_inst_out = _run_instance_segmenter_from_probs(shared_inst_seg, sim_probs)
    inst_sim = sim_inst_out["instance_labels"].astype(np.int32)

    # --- 1) UNET forward ---
    t0 = time.time()
    seg_out = unet(img_rgb)  # compute_instances must be False in cfg
    t1 = time.time()

    # --- 2) Run SAME instance segmenter on UNET probs ---
    unet_probs = seg_out["probs"]
    pr_inst_out = _run_instance_segmenter_from_probs(shared_inst_seg, unet_probs)
    t2 = time.time()
    inst_pred = pr_inst_out["instance_labels"].astype(np.int32)

    # --- 3) metrics ---
    N_gt = _count_instances(inst_sim)
    N_pr = _count_instances(inst_pred)
    res: Dict[str, Any] = {
        "image_id": image_id,
        "unet_mode": mode_name,
        "N_gt": N_gt,
        "N_pred": N_pr,
        "count_error_frac": (abs(N_pr - N_gt) / max(1, N_gt)),
        "unet_ms": (t1 - t0) * 1000.0,
        "instance_ms": (t2 - t1) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
    }

    gt_bin_frac = float((inst_sim > 0).mean())
    pr_bin_frac = float((inst_pred > 0).mean())
    res["gt_area_frac"] = gt_bin_frac
    res["pr_area_frac"] = pr_bin_frac
    res["coverage_error_frac"] = abs(pr_bin_frac - gt_bin_frac) / max(gt_bin_frac, 1e-8)

    # IoU/matching across thresholds
    gt_c = _compact_labels(inst_sim)
    pr_c = _compact_labels(inst_pred)
    IoU, gt_ids, pr_ids, _, _ = _contingency_iou(gt_c, pr_c)
    for thr in iou_thresholds:
        ri, cj, pairs = _hungarian(IoU, thr)
        matched_rows = set(p[0] for p in pairs)
        matched_cols = set(p[1] for p in pairs)
        TP = len(pairs)
        FP = len(pr_ids) - len(matched_cols)
        FN = len(gt_ids) - len(matched_rows)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        ious = [p[2] for p in pairs]
        thr_tag = str(thr).replace(".", "_")
        res[f"TP_{thr_tag}"] = TP
        res[f"FP_{thr_tag}"] = FP
        res[f"FN_{thr_tag}"] = FN
        res[f"precision_{thr_tag}"] = prec
        res[f"recall_{thr_tag}"] = rec
        res[f"f1_{thr_tag}"] = f1
        res[f"mean_iou_{thr_tag}"] = (float(np.mean(ious)) if ious else 0.0)

    # shape distribution checks
    res.update(_shape_distributions(inst_sim, inst_pred))

    # artifacts
    if save_artifacts and artifacts_dir:
        np.savez_compressed(os.path.join(artifacts_dir, f"{image_id}.gt.npz"), lab=inst_sim)
        np.savez_compressed(os.path.join(artifacts_dir, f"{image_id}.pred.npz"), lab=inst_pred)

    # flatten meta (from simulate_image) + derived params
    # meta has: centers, labels, radius, params{N, n_cells, cell_diameter, frac_positive, seed, ... maybe more}
    meta = dict(meta or {})
    params = dict(meta.get("params", {}))
    res["meta_radius"] = int(meta.get("radius", -1))
    res["meta_centers_count"] = int(len(meta.get("centers", [])))
    res["meta_labels_pos_frac"] = float(np.mean(meta.get("labels", []))) if len(meta.get("labels", [])) else np.nan

    # store params.* individually
    for k, v in params.items():
        res[f"meta_param_{k}"] = v

    # add derived (scale-aware) params if given
    for k, v in (derived_params or {}).items():
        res[f"meta_derived_{k}"] = v

    return res


# -------------------- main runner --------------------

def unet_validation(
    *,
    dataset_cfg: Optional[Dict[str, Any]] = None,
    num_images: int = 100_000,
    batch_size: int = 4,
    n_jobs: int = 32,
    output_dir: str = "./validation_runs",
    save_artifacts: bool = True,
    iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    modes: Tuple[str, ...] = ("small", "medium", "large"),
    description: str = "",
    tags: Optional[list[str]] = None,
) -> str:
    """
    Evaluate UNet modes against GT built by running the SAME InstanceSegmenter on simulated targets.
    Runs each experiment twice: random_camera_rect=False and True.
    """
    # Prepare run folder
    run_id = _next_run_id(output_dir)   # e.g. "121025" or "121025-001"
    run_dir = os.path.join(output_dir, f"unet_validation_{run_id}")
    art_dir = os.path.join(run_dir, "artifacts")
    _ensure_dir(run_dir)
    if save_artifacts:
        _ensure_dir(art_dir)

    # Resolve dataset cfg (optional)
    base_ds_cfg = dict(DEFAULT_DATASET_CFG if dataset_cfg is None else dataset_cfg)
    base_ds_cfg["length"] = num_images
    length = int(base_ds_cfg.get("length", num_images))

    # Snapshot configs
    unet_base_cfg = dict(UNET_CONFIG)
    inst_cfg = dict(INSTANCE_CONFIG)

    run_snap = {
        "run_id": run_id,
        "created_at": _date_id_ddmmyy(),
        "description": description,
        "tags": list(tags or []),
        "unet_base_config": unet_base_cfg,
        "instance_config": inst_cfg,
        "dataset_config": dict(base_ds_cfg),   # what we started from
        "device": device,
        "iou_thresholds": list(iou_thresholds),
        "modes": list(modes),
        "camera_sweep": [False, True],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "code_hash": _hash_dict({"this_file": Path(__file__).read_text(encoding="utf-8")}),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(run_snap), f, indent=2)

    # Write a human summary
    summary = []
    summary.append(f"Run ID: {run_id}")
    summary.append(f"Created: {run_snap['created_at']}")
    summary.append(f"Description: {description}")
    summary.append(f"Tags: {', '.join(tags or [])}")
    summary.append(f"Modes: {', '.join(modes)}")
    summary.append(f"Device: {device}")
    summary.append(f"Camera sweep: random_camera_rect in [False, True]")
    summary.append(f"Dataset cfg keys: {', '.join(sorted(base_ds_cfg.keys()))}")

    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    # Shared InstanceSegmenter (same for GT and prediction)
    shared_inst = InstanceSegmenter.from_config(inst_cfg)

    # Parquet writer
    pq_path = os.path.join(run_dir, "results.parquet")
    pq_writer = None

    # Derived scale-aware params
    def _derived_from_diameter(ds: SimCellsDataset, dia: float) -> Dict[str, Any]:
        c_sigma = float(max(ds.center_sigma_min, 0.25 * float(dia)))
        d = float(dia)
        ring = int(np.clip(round(0.15 * d), ds.bound_ring_width_min, ds.bound_ring_width_max))
        band = int(np.clip(round(0.40 * d), ds.bound_soft_band_min, ds.bound_soft_band_max))
        ring = min(ring, max(1, ds.bound_ring_width))
        band = min(band, max(1, ds.bound_soft_band))
        return {
            "center_sigma_used": c_sigma,
            "ring_width_used": ring,
            "soft_band_used": band,
            "is_tiny": bool(float(dia) < ds.small_diam_thresh),
        }

    per_cam_total = min(num_images, length)
    overall_total = per_cam_total * len(modes) * 2
    overall_bar = tqdm(total=overall_total, unit="img", desc="Total", leave=True)

    try:
        for cam_flag in (False, True):
            cam_bar = tqdm(total=per_cam_total * len(modes), unit="img",
                           desc=f"Camera={'ON' if cam_flag else 'OFF'}", leave=False)

            for mode in modes:
                # UNet config for this mode
                unet_cfg = dict(unet_base_cfg)
                unet_cfg["unet_mode"] = mode
                unet_cfg["device"] = device
                unet_cfg["compute_instances"] = False
                unet = SegmenterUNet.from_config(unet_cfg)

                # Dataset for this mode + camera flag (same rng => matching sequences)
                ds_cfg = dict(base_ds_cfg)
                ds_cfg["random_camera_rect"] = bool(cam_flag)
                ds = SimCellsDataset(**ds_cfg)

                idx = 0
                processed = 0
                mode_bar = tqdm(total=per_cam_total, unit="img",
                                desc=f"Mode={mode} Cam={'ON' if cam_flag else 'OFF'}", leave=False)

                while idx < per_cam_total:
                    current_bs = min(batch_size, per_cam_total - idx)
                    batch_items: List[Tuple[str, np.ndarray, torch.Tensor, Dict[str, Any]]] = []

                    t_load0 = time.time()
                    for _ in range(current_bs):
                        img_t, tgt_t, extras = ds[idx]
                        img = img_t.numpy().transpose(1, 2, 0).astype(np.float32)
                        img = np.clip(img, 0.0, 1.0)
                        meta = dict(extras.get("meta", {}))
                        dia = float(meta.get("cell_diameter", ds.cell_diameter if isinstance(ds.cell_diameter, (int, float)) else -1))
                        derived = _derived_from_diameter(ds, dia)
                        image_id = f"im_{idx:07d}"
                        batch_items.append((image_id, img, tgt_t, meta | {"random_camera_rect": cam_flag}, derived))
                        idx += 1
                    t_load1 = time.time()

                    t_gpu0 = time.time()
                    def _job(item):
                        image_id, img, tgt_t, meta, derived = item
                        # carry camera flag into results
                        row = evaluate_one_image(
                            img_rgb=img,
                            tgt_t=tgt_t,
                            meta=meta,
                            unet=unet,
                            shared_inst_seg=shared_inst,
                            iou_thresholds=iou_thresholds,
                            save_artifacts=save_artifacts,
                            artifacts_dir=art_dir if save_artifacts else None,
                            image_id=image_id,
                            derived_params=derived,
                            mode_name=mode,
                        )
                        # add camera flag column
                        row["random_camera_rect"] = bool(meta.get("random_camera_rect", False))
                        return row

                    results_list = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                        delayed(_job)(itm) for itm in batch_items
                    )
                    t_gpu1 = time.time()

                    df = pd.DataFrame(results_list)
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if pq_writer is None:
                        pq_writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
                    pq_writer.write_table(table)

                    processed += current_bs
                    mode_bar.update(current_bs)
                    cam_bar.update(current_bs)
                    overall_bar.update(current_bs)

                    batch_time = (t_gpu1 - t_gpu0)
                    imgs_per_sec = current_bs / max(1e-6, (t_load1 - t_load0) + batch_time)
                    mode_bar.set_postfix({
                        "batch_gpu_ms": f"{batch_time*1000.0:.0f}",
                        "img/s": f"{imgs_per_sec:.1f}",
                        "done": processed
                    })

                mode_bar.close()
            cam_bar.close()

    finally:
        overall_bar.close()
        if pq_writer is not None:
            pq_writer.close()

    # Update global runs index with quick stats
    with open(os.path.join(run_dir, "run_meta.json"), "r", encoding="utf-8") as f:
        meta_saved = json.load(f)
    head_cols = [c for c in ("unet_mode","random_camera_rect","f1_0_5","f1@0.5","count_error_frac","count_error_pct") if c]
    try:
        head_df = pd.read_parquet(os.path.join(run_dir, "results.parquet"), columns=head_cols)
    except Exception:
        head_df = pd.read_parquet(os.path.join(run_dir, "results.parquet"))
    _append_run_index(output_dir, run_dir, meta_saved, head_df)

    return run_dir

