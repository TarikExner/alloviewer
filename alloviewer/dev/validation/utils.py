from __future__ import annotations
import os
import math
import json
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import cv2
import torch
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure

from alloviewer.image_analysis.segmenter import SegmenterUNet

# ----------------------------
# geometry & resizing
# ----------------------------
def crop_rect(arr: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    return arr[y0:y0+h, x0:x0+w, ...] if arr.ndim == 3 else arr[y0:y0+h, x0:x0+w]


def estimate_well_mask(img: np.ndarray, blur_sigma: float = 3.0, well_is_brighter: str | bool = "auto"):
    g = img if img.ndim == 2 else (0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2])
    g = ndi.gaussian_filter(g.astype(np.float32), blur_sigma)
    g = exposure.rescale_intensity(g, in_range='image', out_range=(0, 1))
    thr = filters.threshold_otsu(g)
    m1 = g > thr
    m2 = g < thr
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

def square_crop_from_center_radius(mask_shape: Tuple[int,int], center: Tuple[float,float], radius: float, pad: int = 8):
    H, W = mask_shape
    cy, cx = center
    half = int(math.ceil(radius + pad))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
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
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)
    return y0, y1, x0, x1


# ----------------------------
# mask metrics
# ----------------------------

def iou_dice_overlap(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    """
    pred_bin, gt_bin: binary arrays {0,1} or {False,True}
    Returns IoU (Jaccard), Dice, and Overlap coefficient.
    """
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    p_sum = float(p.sum())
    g_sum = float(g.sum())

    iou = inter / (union + 1e-8)
    dice = (2.0 * inter) / (p_sum + g_sum + 1e-8)
    overlap = inter / (min(p_sum, g_sum) + 1e-8)
    return {"iou": float(iou), "dice": float(dice), "overlap": float(overlap)}


# ----------------------------
# boundary metrics
# ----------------------------

def compute_thin_gt_boundary_from_instances(inst_gt: np.ndarray) -> np.ndarray:
    """
    Make a 1-pixel thin boundary map from instance labels via morphological gradient + skeletonize.
    """
    # edge from gradient on binary instance map
    bin_gt = (inst_gt > 0).astype(np.uint8)
    edge = cv2.morphologyEx(bin_gt, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = (edge > 0).astype(np.uint8)
    # thin
    thin = skeletonize(edge > 0).astype(np.uint8)
    return thin


def boundary_fscore_np(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> Dict[str, float]:
    """
    Boundary F-score with tolerance (like BSDS).
    pred, gt are binary edges (uint8 or bool).
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # distance transforms for tolerance matching
    from scipy.ndimage import distance_transform_edt as dt
    dt_pred = dt(1 - pred)
    dt_gt = dt(1 - gt)

    # match pred to gt (within tol)
    pred_to_gt = (dt_gt <= tol) & (pred > 0)
    gt_to_pred = (dt_pred <= tol) & (gt > 0)

    tp = float(pred_to_gt.sum())
    fp = float((pred > 0).sum()) - tp
    fn = float((gt > 0).sum()) - float(gt_to_pred.sum())

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def boundary_f1_skeletonized(pred_prob: np.ndarray,
                             inst_gt: np.ndarray,
                             tol: int = 2,
                             thr: float = 0.9,
                             sweep: bool = False) -> float:
    """
    F1 between predicted boundary prob and thin GT boundary from instances.
    If sweep=True, returns best F1 over a small threshold grid.
    """
    gt_thin = compute_thin_gt_boundary_from_instances(inst_gt)

    def f1_at(t):
        pred_bin = (pred_prob >= t).astype(np.uint8)
        pred_skel = skeletonize(pred_bin > 0).astype(np.uint8)
        stats = boundary_fscore_np(pred_skel, gt_thin, tol=tol)
        return float(stats["f1"])

    if sweep:
        ts = np.linspace(0.05, 0.7, 14)
        return max(f1_at(t) for t in ts)
    else:
        return f1_at(thr)


# ----------------------------
# centers (peaks, matching, AP, OKS)
# ----------------------------

def centroids_from_instances(inst: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get simple centroids (mean y/x) for each instance id >=1.
    """
    centers: List[Tuple[int, int]] = []
    max_id = int(inst.max())
    for k in range(1, max_id + 1):
        ys, xs = np.where(inst == k)
        if ys.size == 0:
            continue
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        centers.append((cy, cx))
    return centers


def nms_peaks_np(heat: np.ndarray, thr: float = 0.2, min_dist: int = 3) -> List[Tuple[int, int, float]]:
    """
    Simple NMS on a heatmap using dilation to find local maxima above thr.
    Returns (y, x, score) sorted by score desc.
    """
    h = (heat >= thr).astype(np.uint8)
    if min_dist > 1:
        k = 2 * min_dist + 1
        heat_max = cv2.dilate(heat, np.ones((k, k), np.uint8))
        h = np.logical_and(h, heat == heat_max)
    ys, xs = np.nonzero(h)
    scores = heat[ys, xs]
    idx = np.argsort(-scores)
    return [(int(ys[i]), int(xs[i]), float(scores[i])) for i in idx]


def _hungarian_match(P: np.ndarray, G: np.ndarray, max_dist: float) -> Tuple[int, List[float]]:
    """
    Hungarian 1-1 matching by Euclidean distance with a max distance cap.
    Returns (TP, list_of_distances_for_TP).
    """
    if len(P) == 0 or len(G) == 0:
        return 0, []
    dists = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    cost = dists.copy()
    cost[dists > max_dist] = 1e6
    ri, cj = linear_sum_assignment(cost)
    matched = []
    for r, c in zip(ri, cj):
        if dists[r, c] <= max_dist:
            matched.append(dists[r, c])
    return len(matched), matched


def _pr_ap_from_thresholds(center_pred: np.ndarray,
                           G: np.ndarray,
                           thr_list: Sequence[float],
                           nms_dist: int,
                           match_radius: int) -> Dict[str, Any]:
    prs, rcs = [], []
    for t in thr_list:
        preds = nms_peaks_np(center_pred, thr=t, min_dist=nms_dist)
        P = np.array([(y, x) for y, x, _ in preds], dtype=float)
        TP, _ = _hungarian_match(P, G, match_radius)
        FP = len(P) - TP
        FN = len(G) - TP
        prec = TP / (TP + FP + 1e-8)
        rec = TP / (TP + FN + 1e-8)
        prs.append(prec)
        rcs.append(rec)
    order = np.argsort(rcs)
    rcs = np.array(rcs)[order]
    prs = np.array(prs)[order]
    ap = 0.0
    for i in range(1, len(rcs)):
        ap += (rcs[i] - rcs[i - 1]) * (prs[i] + prs[i - 1]) * 0.5
    return {"ap": float(ap), "precisions": prs.tolist(), "recalls": rcs.tolist()}


def _oks_for_matches(match_dists: List[float],
                     inst_gt: np.ndarray,
                     k: float = 0.5) -> List[float]:
    """
    OKS per matched pair. Scale s from median instance area: s = sqrt(area_median).
    OKS = exp( - d^2 / (2 * (k*s)^2) ).
    """
    areas = []
    max_id = int(inst_gt.max())
    for lab in range(1, max_id + 1):
        areas.append(float((inst_gt == lab).sum()))
    if len(areas) == 0:
        return []
    s = math.sqrt(np.median(areas))
    oks_vals = [math.exp(-(d * d) / (2.0 * (k * s) ** 2 + 1e-8)) for d in match_dists]
    return oks_vals


def center_metrics_hungarian(center_pred: np.ndarray,
                             inst_gt: np.ndarray,
                             peak_thr: float = 0.2,
                             nms_dist: int = 3,
                             match_radius: int = 10,
                             ap_thr_list: Sequence[float] = tuple(np.linspace(0.05, 0.7, 14)),
                             oks_thresholds: Sequence[float] = (0.5, 0.75, 0.9)) -> Dict[str, Any]:
    """
    Main center metrics with Hungarian match, PR/AP sweep, and OKS mAP.
    """
    preds = nms_peaks_np(center_pred, thr=peak_thr, min_dist=nms_dist)
    P = np.array([(y, x) for y, x, _ in preds], dtype=float)
    G = np.array(centroids_from_instances(inst_gt), dtype=float)

    TP, match_dists = _hungarian_match(P, G, match_radius)
    FP = len(P) - TP
    FN = len(G) - TP
    prec = TP / (TP + FP + 1e-8)
    rec = TP / (TP + FN + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    loc_mean = float(np.mean(match_dists)) if match_dists else np.nan
    loc_median = float(np.median(match_dists)) if match_dists else np.nan
    loc_q95 = float(np.quantile(match_dists, 0.95)) if match_dists else np.nan

    ap_pack = _pr_ap_from_thresholds(center_pred, G, ap_thr_list, nms_dist, match_radius)

    oks_vals = _oks_for_matches(match_dists, inst_gt, k=0.5)
    mAP_oks = {}
    if oks_vals:
        for th in oks_thresholds:
            mAP_oks[f"oks@{th:.2f}"] = float(np.mean([1.0 if v >= th else 0.0 for v in oks_vals]))
        mAP_oks["oks_mAP"] = float(np.mean(list(mAP_oks.values())))
    else:
        for th in oks_thresholds:
            mAP_oks[f"oks@{th:.2f}"] = np.nan
        mAP_oks["oks_mAP"] = np.nan

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_pred": int(len(P)),
        "n_gt": int(len(G)),
        "n_tp": int(TP),
        "loc_mean": loc_mean,
        "loc_median": loc_median,
        "loc_q95": loc_q95,
        "ap": ap_pack["ap"],
        **{f"pr_prec_t{i}": v for i, v in enumerate(ap_pack["precisions"])},
        **{f"pr_rec_t{i}": v for i, v in enumerate(ap_pack["recalls"])},
        **mAP_oks,
    }


# (optional) original greedy matcher for centers, kept for parity with past code
def center_metrics_np(center_pred: np.ndarray,
                      gt_centers: Sequence[Tuple[int, int]],
                      peak_thr: float = 0.2,
                      nms_dist: int = 3,
                      match_radius: int = 10) -> Dict[str, Any]:
    preds = nms_peaks_np(center_pred, thr=peak_thr, min_dist=nms_dist)
    P = np.array([(y, x) for y, x, _ in preds], dtype=float)
    G = np.array(gt_centers, dtype=float)
    if len(P) == 0 or len(G) == 0:
        TP, FP, FN = 0, len(P), len(G)
    else:
        tree = cKDTree(G)
        taken = np.zeros(len(G), dtype=bool)
        TP = 0
        for p in P:
            d, j = tree.query(p, distance_upper_bound=match_radius)
            if np.isfinite(d) and j < len(G) and not taken[j]:
                taken[j] = True
                TP += 1
        FP = len(P) - TP
        FN = len(G) - TP
    prec = TP / (TP + FP + 1e-8)
    rec = TP / (TP + FN + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "n_pred": int(len(P)), "n_gt": int(len(G)), "n_tp": int(TP)
    }


# ----------------------------
# energy metrics (kept for reuse)
# ----------------------------

def concordance_ccc(x: np.ndarray, y: np.ndarray) -> float:
    xm = x.mean()
    ym = y.mean()
    vx = x.var()
    vy = y.var()
    cov = ((x - xm) * (y - ym)).mean()
    denom = vx + vy + (xm - ym) ** 2
    if denom <= 1e-12:
        return np.nan
    return float((2 * cov) / (denom + 1e-12))


def energy_errors_np(energy_pred: np.ndarray,
                     energy_gt: np.ndarray,
                     cell_mask_gt: np.ndarray) -> dict:
    """
    RMSE and Pearson r inside cells. NaN-safe, with validity flags.
    """
    m = (cell_mask_gt > 0.5)
    if not np.any(m):
        return {"rmse": np.nan, "pearson": np.nan, "valid_rmse": False, "valid_r": False}

    p = energy_pred[m].astype(np.float32)
    g = energy_gt[m].astype(np.float32)

    rmse = float(np.sqrt(np.mean((p - g) ** 2))) if p.size else np.nan
    valid_rmse = bool(np.isfinite(rmse))

    p0 = p - p.mean()
    g0 = g - g.mean()
    denom = np.sqrt((p0 ** 2).sum() * (g0 ** 2).sum())
    if denom <= 1e-8:
        r = np.nan
        valid_r = False
    else:
        r = float((p0 * g0).sum() / denom)
        r = float(np.clip(r, -1.0, 1.0))
        valid_r = True

    return {"rmse": rmse, "pearson": r, "valid_rmse": valid_rmse, "valid_r": valid_r}


def energy_metrics_extended_full(
    energy_pred: np.ndarray,
    energy_gt: np.ndarray,
    cell_mask: np.ndarray,
    frac_delta: Optional[float] = 0.05,
) -> Dict[str, Any]:
    m = cell_mask > 0.5
    n_pix = int(m.sum())
    if n_pix == 0:
        return {
            "rmse": np.nan, "mae": np.nan, "nrmse_range": np.nan,
            "pearson": np.nan, "spearman": np.nan, "ccc": np.nan,
            "ssim": np.nan, "grad_corr": np.nan, "wasserstein": np.nan,
            "bias": np.nan, "slope": np.nan, "intercept": np.nan,
            "frac_within_delta": np.nan, "valid_pixels": 0
        }

    p = energy_pred[m].astype(np.float32)
    g = energy_gt[m].astype(np.float32)

    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))
    rng = float(g.max() - g.min()) if np.isfinite(g.max() - g.min()) else np.nan
    nrmse_range = float(rmse / (rng + 1e-8)) if np.isfinite(rng) and rng > 1e-12 else np.nan

    pear = float(np.corrcoef(p, g)[0, 1]) if p.size > 1 else np.nan
    sp_r = float(spearmanr(p, g).correlation) if p.size > 1 else np.nan
    ccc = concordance_ccc(p, g)

    # linear fit (g = a*p + b)
    A = np.vstack([p, np.ones_like(p)]).T
    a, b = np.linalg.lstsq(A, g, rcond=None)[0]
    bias = float((p - g).mean())

    # --- SSIM + gradient part ---
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    p_patch = energy_pred[y0:y1, x0:x1].astype(np.float32)
    g_patch = energy_gt[y0:y1, x0:x1].astype(np.float32)
    m_patch = m[y0:y1, x0:x1]

    # default
    ssim_val = np.nan
    grad_corr = np.nan

    # check size to avoid np.gradient crash
    if p_patch.shape[0] >= 2 and p_patch.shape[1] >= 2:
        p_in = p_patch[m_patch]
        g_in = g_patch[m_patch]
        if p_in.size > 0 and g_in.size > 0:
            p_f = p_patch.copy()
            g_f = g_patch.copy()
            p_f[~m_patch] = p_in.mean()
            g_f[~m_patch] = g_in.mean()

            def _mm(a):
                amin, amax = float(a.min()), float(a.max())
                return (a - amin) / (amax - amin + 1e-8)

            p_n = _mm(p_f)
            g_n = _mm(g_f)
            try:
                ssim_val = float(ssim(p_n, g_n, data_range=1.0))
            except Exception:
                ssim_val = np.nan

            # safe now
            gp_y, gp_x = np.gradient(p_n)
            gg_y, gg_x = np.gradient(g_n)
            gp_mag = np.sqrt(gp_y[m_patch] ** 2 + gp_x[m_patch] ** 2)
            gg_mag = np.sqrt(gg_y[m_patch] ** 2 + gg_x[m_patch] ** 2)
            grad_corr = float(np.corrcoef(gp_mag, gg_mag)[0, 1]) if gp_mag.size > 1 else np.nan

    # wasserstein
    try:
        w1 = float(wasserstein_distance(p, g))
    except Exception:
        w1 = np.nan

    # fraction within delta
    if frac_delta is None:
        delta = 0.05 * (float(np.nanmax(g) - np.nanmin(g)) + 1e-8)
    else:
        if 0 < frac_delta <= 1.0:
            delta = frac_delta * (float(np.nanmax(g) - np.nanmin(g)) + 1e-8)
        else:
            delta = float(frac_delta)
    frac_ok = float(np.mean(np.abs(p - g) <= delta)) if p.size else np.nan

    return {
        "rmse": rmse, "mae": mae, "nrmse_range": nrmse_range,
        "pearson": pear, "spearman": sp_r, "ccc": ccc,
        "ssim": ssim_val, "grad_corr": grad_corr, "wasserstein": w1,
        "bias": bias, "slope": float(a), "intercept": float(b),
        "frac_within_delta": frac_ok, "valid_pixels": n_pix
    }

def jsonify(x: Any):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    return x

def flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k == "params" and isinstance(v, dict):
            for pk, pv in v.items():
                out[f"param__{pk}"] = pv
        elif isinstance(v, (dict, list, tuple)):
            out[f"meta__{k}"] = json.dumps(v)
        else:
            out[f"meta__{k}"] = v
    return out

def get_dataset_mode(h5_path: str) -> str:
    hp = h5_path.lower()
    if "tile" in hp:
        return "tiling"
    elif "crop_well" in hp:
        return "crop_well_resize"
    elif "pad_resize" in hp:
        return "pad_resize"
    else:
        return "unknown"


def decode_json_maybe(x: Any) -> Dict[str, Any]:
    """
    Decode one metadata entry from H5.

    Supports:
      - bytes
      - np.bytes_
      - JSON strings
      - dicts
      - zero-dimensional numpy object/string arrays
    """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    if isinstance(x, np.bytes_):
        x = x.tobytes().decode("utf-8")

    if isinstance(x, str):
        return json.loads(x)

    if isinstance(x, dict):
        return x

    raise TypeError(f"Could not decode meta entry of type {type(x)}")


def _first_present(d: Dict[str, Any], keys: List[str], default=None):
    """
    Return the first present non-None value from a dict.
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _as_int_pair(x: Any) -> Optional[Tuple[int, int]]:
    """
    Convert a list/tuple/array-like object with at least two entries to an int pair.
    """
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return int(x[0]), int(x[1])

    return None


def count_positive_labels(lbl: np.ndarray) -> int:
    """
    Count instance labels greater than zero.
    """
    vals = np.unique(lbl)
    vals = vals[vals > 0]
    return int(vals.size)


def load_human_roi_counts(csv_dir: str) -> pd.DataFrame:
    """
    Load human ROI counts from human_annotations.csv.

    Expected columns:
      - Folder
      - image_name
      - roi_id

    Returns one row per image:
      Folder, image_name, human_roi_count
    """
    csv_path = os.path.join(csv_dir, "human_annotations.csv")
    df = pd.read_csv(csv_path)

    needed = {"Folder", "image_name", "roi_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    out = (
        df.loc[:, ["Folder", "image_name", "roi_id"]]
        .dropna(subset=["Folder", "image_name", "roi_id"])
        .copy()
    )

    out["Folder"] = out["Folder"].astype(str)
    out["image_name"] = out["image_name"].astype(str)

    return (
        out.groupby(["Folder", "image_name"], as_index=False)["roi_id"]
        .nunique()
        .rename(columns={"roi_id": "human_roi_count"})
    )


def extract_image_identity(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract (Folder, image_name) from H5 metadata.

    Priority:
      1) direct fields in meta["full"]
      2) src_path / image_path / img_path / path in meta["full"]
      3) same path fields in first tile's full_meta
    """
    full = meta.get("full", {})
    if not isinstance(full, dict):
        full = {}

    folder = _first_present(
        full,
        ["Folder", "folder", "subfolder", "dir_name", "dirname", "source_folder"],
        default=None,
    )
    image_name = _first_present(
        full,
        ["image_name", "filename", "file_name", "img_name", "name"],
        default=None,
    )

    if folder is not None and image_name is not None:
        return str(folder), str(image_name)

    src_path = _first_present(
        full,
        ["src_path", "image_path", "img_path", "path"],
        default=None,
    )

    if src_path is None:
        tiles = meta.get("tiles", [])
        if isinstance(tiles, list) and len(tiles) > 0:
            first_tile = tiles[0]
            if isinstance(first_tile, dict):
                full_meta = first_tile.get("full_meta", {})
                if isinstance(full_meta, dict):
                    src_path = _first_present(
                        full_meta,
                        ["src_path", "image_path", "img_path", "path"],
                        default=None,
                    )

    if src_path is not None:
        src_path = os.path.normpath(str(src_path))
        image_name = os.path.basename(src_path)
        folder = os.path.basename(os.path.dirname(src_path))
        if folder and image_name:
            return folder, image_name

    raise KeyError(
        "Could not extract Folder/image_name from H5 meta. "
        f"Available top-level keys: {list(meta.keys())}, "
        f"full keys: {list(full.keys()) if isinstance(full, dict) else 'n/a'}"
    )


def extract_tile_box(
    tile_meta: Dict[str, Any],
    tile_hw: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Return tile box as (y0, y1, x0, x1).

    For the current tile export:
      - tile_xy = (y0, x0)
      - tile_hw = (h, w)

    Also supports generic y0/y1/x0/x1-style metadata.
    """
    if "tile_xy" in tile_meta:
        xy = tile_meta["tile_xy"]

        if isinstance(xy, np.ndarray):
            xy = xy.tolist()

        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            raise ValueError(f"tile_xy has invalid format: {xy}")

        y0 = int(xy[0])
        x0 = int(xy[1])

        hw = tile_meta.get("tile_hw", tile_hw)

        if isinstance(hw, np.ndarray):
            hw = hw.tolist()

        if not isinstance(hw, (list, tuple)) or len(hw) < 2:
            raise ValueError(f"tile_hw has invalid format: {hw}")

        th = int(hw[0])
        tw = int(hw[1])

        return y0, y0 + th, x0, x0 + tw

    y0 = _first_present(tile_meta, ["y0", "top", "row0", "r0"], default=None)
    y1 = _first_present(tile_meta, ["y1", "bottom", "row1", "r1"], default=None)
    x0 = _first_present(tile_meta, ["x0", "left", "col0", "c0"], default=None)
    x1 = _first_present(tile_meta, ["x1", "right", "col1", "c1"], default=None)

    if None not in (y0, y1, x0, x1):
        return int(y0), int(y1), int(x0), int(x1)

    raise KeyError(
        "Could not extract tile box from tile metadata. "
        f"Available tile keys: {list(tile_meta.keys())}"
    )


def extract_full_hw(
    meta: Dict[str, Any],
    tile_metas: List[Dict[str, Any]],
    tile_hw: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Infer the full image size as (H, W).

    Priority:
      1) meta["full"] height/width fields
      2) first tile's full_meta
      3) shape-like fields
      4) fallback from tile extents
    """
    full = meta.get("full", {})
    if not isinstance(full, dict):
        full = {}

    if not full and len(tile_metas) > 0:
        first_tile = tile_metas[0]
        if isinstance(first_tile, dict):
            full_meta = first_tile.get("full_meta", {})
            if isinstance(full_meta, dict):
                full = full_meta

    H = _first_present(
        full,
        ["height", "H", "img_h", "image_height", "H_in"],
        default=None,
    )
    W = _first_present(
        full,
        ["width", "W", "img_w", "image_width", "W_in"],
        default=None,
    )

    if H is not None and W is not None:
        return int(H), int(W)

    shape = _first_present(
        full,
        ["shape", "image_shape", "full_shape", "hw"],
        default=None,
    )
    hw = _as_int_pair(shape)
    if hw is not None:
        return int(hw[0]), int(hw[1])

    max_y = 0
    max_x = 0

    for tm in tile_metas:
        y0, y1, x0, x1 = extract_tile_box(tm, tile_hw)
        max_y = max(max_y, y1)
        max_x = max(max_x, x1)

    if max_y <= 0 or max_x <= 0:
        raise ValueError("Could not infer full image size from H5 metadata.")

    return int(max_y), int(max_x)


def stitch_prob_tiles(
    prob_tiles: np.ndarray,
    tile_metas: List[Dict[str, Any]],
    full_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Stitch probability tiles into one full-image probability array.

    Parameters
    ----------
    prob_tiles:
        Array with shape [T, C, tile_h, tile_w].
    tile_metas:
        List of tile metadata dicts, length T.
    full_hw:
        Full image size as (H, W).

    Returns
    -------
    np.ndarray
        Array with shape [C, H, W].

    Important correction:
    This version handles clipped tiles correctly. If a tile starts outside the
    full image coordinate range, the matching source offset inside the tile is
    used instead of always reading from [:th, :tw].
    """
    if prob_tiles.ndim != 4:
        raise ValueError(f"Expected prob_tiles [T,C,h,w], got {prob_tiles.shape}")

    T, C, tile_h, tile_w = prob_tiles.shape

    if len(tile_metas) != T:
        raise ValueError(
            f"Number of tile metas ({len(tile_metas)}) does not match "
            f"number of tiles ({T})"
        )

    H, W = int(full_hw[0]), int(full_hw[1])

    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid full_hw={full_hw}")

    acc = np.zeros((C, H, W), dtype=np.float32)
    wgt = np.zeros((1, H, W), dtype=np.float32)

    for i, tm in enumerate(tile_metas):
        y0, y1, x0, x1 = extract_tile_box(tm, (tile_h, tile_w))

        dst_y0 = max(y0, 0)
        dst_x0 = max(x0, 0)
        dst_y1 = min(y1, H)
        dst_x1 = min(x1, W)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            continue

        src_y0 = dst_y0 - y0
        src_x0 = dst_x0 - x0
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        src_y0 = max(src_y0, 0)
        src_x0 = max(src_x0, 0)
        src_y1 = min(src_y1, tile_h)
        src_x1 = min(src_x1, tile_w)

        copy_h = min(dst_y1 - dst_y0, src_y1 - src_y0)
        copy_w = min(dst_x1 - dst_x0, src_x1 - src_x0)

        if copy_h <= 0 or copy_w <= 0:
            continue

        acc[
            :,
            dst_y0:dst_y0 + copy_h,
            dst_x0:dst_x0 + copy_w,
        ] += prob_tiles[
            i,
            :,
            src_y0:src_y0 + copy_h,
            src_x0:src_x0 + copy_w,
        ]

        wgt[
            :,
            dst_y0:dst_y0 + copy_h,
            dst_x0:dst_x0 + copy_w,
        ] += 1.0

    wgt[wgt == 0] = 1.0

    return (acc / wgt).astype(np.float32)


def segment_one_h5_entry(
    segmenter: SegmenterUNet,
    imgs_tiles: np.ndarray,
    meta: Dict[str, Any],
    *,
    update_cell_mask: bool = False,
) -> Dict[str, Any]:
    """
    Segment one tiled H5 entry and return one count row.

    imgs_tiles: [Tmax, 3, S, S]
    meta: decoded JSON dict with a 'tiles' list
    """
    if imgs_tiles.ndim != 4 or imgs_tiles.shape[1] != 3:
        raise ValueError(f"Expected imgs_tiles [T,3,H,W], got {imgs_tiles.shape}")

    tile_metas = meta.get("tiles", None)
    if not isinstance(tile_metas, list) or len(tile_metas) == 0:
        raise ValueError("H5 meta['tiles'] is missing or empty.")

    n_tiles = len(tile_metas)
    imgs_tiles = imgs_tiles[:n_tiles]

    with torch.no_grad():
        tiles_t = segmenter._to_tensor_tiles(imgs_tiles)
        probs_t = segmenter.predict_tiles(tiles_t)

    if torch.is_tensor(probs_t):
        probs_tiles = probs_t.detach().cpu().numpy()
    else:
        probs_tiles = np.asarray(probs_t)

    full_hw = extract_full_hw(meta, tile_metas, imgs_tiles.shape[-2:])
    probs_full = stitch_prob_tiles(probs_tiles, tile_metas, full_hw)

    seg_out = {
        "probs": {
            "cell": probs_full[0],
            "bound": probs_full[1],
            "center": probs_full[2],
            "energy": probs_full[3],
        },
        "instance_labels": None,
        "meta": {},
    }

    if segmenter.inst_seg is None:
        raise RuntimeError(
            "Segmenter has no instance segmenter. Set cfg.compute_instances=True."
        )

    seg_out = segmenter.inst_seg(seg_out, update_cell_mask=update_cell_mask)
    labels = seg_out["instance_labels"]
    folder, image_name = extract_image_identity(meta)

    return {
        "Folder": str(folder),
        "image_name": str(image_name),
        "unet_roi_count": count_positive_labels(labels),
    }
