from __future__ import annotations
import os, json, time, hashlib, socket, platform, uuid
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, wasserstein_distance
from skimage import measure
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from ..segmenter import SegmenterUNet, InstanceSegmenter
from ..config import UNET_CONFIG, INSTANCE_CONFIG
from ..segmentation import SimCellsDataset

try:
    from lcteller.qc import QCMonitorConfig
except Exception:
    QCMonitorConfig = None  # type: ignore


# -------------------- small utils --------------------
def _count_instances(lab: np.ndarray) -> int:
    # number of distinct nonzero IDs
    u = np.unique(lab)
    return int((u > 0).sum())

def _compact_labels(lab: np.ndarray) -> np.ndarray:
    # optional: remap present IDs to 1..K to make max == count for internal use
    lab = lab.astype(np.int32, copy=False)
    u = np.unique(lab)
    u = u[u > 0]
    if u.size == 0:
        return np.zeros_like(lab, dtype=np.int32)
    mapping = {int(old): i+1 for i, old in enumerate(u)}
    out = np.zeros_like(lab, dtype=np.int32)
    # vectorized remap
    # build arrays for fancy indexing
    inv = np.zeros(lab.max() + 1, dtype=np.int32) if lab.max() > 0 else np.zeros(1, dtype=np.int32)
    for old, new in mapping.items():
        if old < inv.size:
            inv[old] = new
    m = lab > 0
    out[m] = inv[lab[m]]
    return out

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


def _hash_dict(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _regionprops_fast(lab: np.ndarray) -> Dict[str, np.ndarray]:
    props = measure.regionprops_table(
        lab, properties=("label", "area", "perimeter", "eccentricity", "solidity", "centroid")
    )
    perim = np.maximum(props["perimeter"], 1.0)
    circ = 4.0 * np.pi * props["area"] / (perim ** 2)
    props["circularity"] = circ
    return props


def _contingency_iou(gt: np.ndarray, pr: np.ndarray) -> Tuple[np.ndarray, List[int], List[int], Dict[int,int], Dict[int,int]]:
    """Build IoU contingency table."""
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

    gt_area = np.bincount(gt.ravel(), minlength=int(gt_ids.max())+1).astype(np.int64)
    pr_area = np.bincount(pr.ravel(), minlength=int(pr_ids.max())+1).astype(np.int64)

    IoU = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    for gi, pi, inter_ in zip(g_ids, p_ids, inter):
        if gi == 0 or pi == 0: continue
        i = gt_idx.get(int(gi), None)
        j = pr_idx.get(int(pi), None)
        if i is None or j is None: continue
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
# -------------------- per-image evaluation --------------------

def evaluate_one_image(
    image_rgb: np.ndarray,
    gt_instances: np.ndarray,
    unet: SegmenterUNet,
    inst_seg: InstanceSegmenter,
    *,
    iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
    save_artifacts: bool = False,
    artifacts_dir: Optional[str] = None,
    image_id: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    seg_out = unet(image_rgb)
    t1 = time.time()
    inst_out = inst_seg(seg_out, update_cell_mask=True)
    t2 = time.time()

    pred_instances = inst_out["instance_labels"].astype(np.int32)

    # robust counts (don’t use .max())
    N_gt = _count_instances(gt_instances)
    N_pr = _count_instances(pred_instances)

    # compact copies for IoU/matching
    gt_for_eval = _compact_labels(gt_instances)
    pr_for_eval = _compact_labels(pred_instances)

    # IoU + matching
    IoU, gt_ids, pr_ids, _, _ = _contingency_iou(gt_for_eval, pr_for_eval)

    results: Dict[str, Any] = {}
    results["N_gt"] = N_gt
    results["N_pred"] = N_pr
    results["count_error_pct"] = abs(N_pr - N_gt) / max(1, N_gt)

    for thr in iou_thresholds:
        ri, cj, pairs = _hungarian(IoU, thr)
        matched_rows = set(p[0] for p in pairs)
        matched_cols = set(p[1] for p in pairs)
        TP = len(pairs)
        FP = len(pr_ids) - len(matched_cols)
        FN = len(gt_ids) - len(matched_rows)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        ious = [p[2] for p in pairs]
        results[f"TP@{thr}"] = TP
        results[f"FP@{thr}"] = FP
        results[f"FN@{thr}"] = FN
        results[f"precision@{thr}"] = prec
        results[f"recall@{thr}"] = rec
        results[f"f1@{thr}"] = f1
        results[f"mean_iou@{thr}"] = float(np.mean(ious)) if ious else 0.0

    # shape summaries (as in your version)
    props_gt = _regionprops_fast(gt_instances)
    props_pr = _regionprops_fast(pred_instances)

    results["gt_area_frac"] = float((gt_instances > 0).mean())
    results["pr_area_frac"] = float((pred_instances > 0).mean())
    results["coverage_error_pct"] = abs(results["pr_area_frac"] - results["gt_area_frac"]) / max(results["gt_area_frac"], 1e-8)

    results["unet_ms"] = (t1 - t0) * 1000.0
    results["instance_ms"] = (t2 - t1) * 1000.0
    results["total_ms"] = (t2 - t0) * 1000.0

    # save artifacts (GT/Pred masks) if requested
    if save_artifacts and artifacts_dir and image_id:
        gt_path = os.path.join(artifacts_dir, f"{image_id}.gt.npz")
        pr_path = os.path.join(artifacts_dir, f"{image_id}.pred.npz")
        np.savez_compressed(gt_path, lab=gt_instances.astype(np.int32))
        np.savez_compressed(pr_path, lab=pred_instances.astype(np.int32))
        results["gt_mask_path"] = gt_path
        results["pred_mask_path"] = pr_path

    if extra_meta:
        results.update(extra_meta)

    return results


# -------------------- runner --------------------

def run_validation(
    *,
    dataset_cfg: Dict[str, Any],
    num_images: int = 100000,
    batch_size: int = 4,
    n_jobs: int = 32,
    output_dir: str = "./validation_runs",
    save_artifacts: bool = True,
    iou_thresholds: Tuple[float,...] = (0.3, 0.5, 0.7),
    qc_config: Optional[Dict[str, Any]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    run_id = _now_id()
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    art_dir = os.path.join(run_dir, "artifacts")
    _ensure_dir(run_dir)
    if save_artifacts:
        _ensure_dir(art_dir)

    unet_cfg = dict(UNET_CONFIG)
    inst_cfg = dict(INSTANCE_CONFIG)
    qc_cfg = dict(qc_config or {})
    snap = {
        "unet_config": unet_cfg,
        "instance_config": inst_cfg,
        "qc_config": qc_cfg,
        "dataset_config": dict(dataset_cfg),
        "device": device,
        "run_id": run_id,
        "iou_thresholds": list(iou_thresholds),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(_sanitize_for_json(snap), f, indent=2)

    unet = SegmenterUNet.from_config(unet_cfg)
    inst_seg = InstanceSegmenter.from_config(inst_cfg)
    ds_cfg = dict(dataset_cfg)
    length = int(ds_cfg.pop("length", num_images))
    ds = SimCellsDataset(length=length, **ds_cfg)

    pq_path = os.path.join(run_dir, "results.parquet")
    pq_writer = None

    total_images = min(num_images, length)
    processed = 0
    idx = 0

    with tqdm(total=total_images, unit="img", desc="Validation") as pbar:
        while idx < total_images:
            current_bs = min(batch_size, total_images - idx)
            batch_imgs, batch_gt, batch_meta = [], [], []
            for _ in range(current_bs):
                img_t, tgt_t, extras = ds[idx]
                img = img_t.numpy().transpose(1, 2, 0).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)
                gt_lab = extras["instance_labels"].numpy().astype(np.int32)
                N_gt = _count_instances(gt_lab)
                meta = dict(extras.get("meta", {}))
                image_id = f"im_{idx:07d}"
                meta.update({
                    "image_id": image_id,
                    "N_gt_post": N_gt,
                })
                batch_imgs.append((image_id, img))
                batch_gt.append(gt_lab)
                batch_meta.append(meta)
                idx += 1

            # GPU inference
            seg_outs = []
            t_gpu0 = time.time()
            for image_id, img in batch_imgs:
                with torch.no_grad():
                    seg_out = unet(img)
                seg_outs.append((image_id, img, seg_out))
            t_gpu1 = time.time()

            def _eval_job(triple, gt_lab, meta):
                image_id, img, seg_out = triple
                res = evaluate_one_image(
                    image_rgb=img,
                    gt_instances=gt_lab,
                    unet=lambda x: seg_out,
                    inst_seg=inst_seg,
                    iou_thresholds=iou_thresholds,
                    save_artifacts=save_artifacts,
                    artifacts_dir=art_dir if save_artifacts else None,
                    image_id=image_id,
                    extra_meta=meta,
                )

                wanted = ["labels", "radius", "image_id", "N", "cell_diameter",
                          "frac_positive", "n_cells", "seed"]
                for k in wanted:
                    res[f"meta_{k}"] = meta.get(k, None)
                return res

            results_list = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_eval_job)(triple, gt_lab, meta)
                for triple, gt_lab, meta in zip(seg_outs, batch_gt, batch_meta)
            )

            df = pd.DataFrame(results_list)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
            pq_writer.write_table(table)

            processed += current_bs
            gpu_ms = (t_gpu1 - t_gpu0) * 1000.0
            pbar.set_postfix({"gpu_ms_batch": f"{gpu_ms:.0f}", "written": processed})
            pbar.update(current_bs)

    if pq_writer is not None:
        pq_writer.close()

    return run_dir

