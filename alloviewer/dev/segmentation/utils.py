import os
from pathlib import Path
import csv
import math
import numpy as np
import torch

from tqdm import tqdm

from scipy import ndimage as ndi
from skimage import filters, measure, morphology, exposure
from skimage.segmentation import watershed
from skimage.segmentation import relabel_sequential

from typing import Union, List, Tuple, Iterable, Sequence, Optional

import cv2

# folders with images for the mean+-STD calculation
EXT_IMAGES_FOLDERS = [
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251106_25722169_iPhone_XR_JPEG",
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251107_25065521_GooglePixel",
    "./ext_images/20251107_25722332_GooglePixel",
    "./ext_images/20251014_25719960",
    "./ext_images/20251014_25720084",
    "./ext_images/20251107_25065521",
    "./ext_images/20251107_25722332"
]

def collate_no_meta(batch):
    imgs, tgts, exs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)   # works for [3,H,W] AND for [1,3,H,W]
    tgts = torch.stack(tgts, dim=0)
    inst = torch.stack([e["instance_labels"] for e in exs], dim=0)
    metas = [e["meta"] for e in exs]
    return imgs, tgts, {"instance_labels": inst, "meta": metas}

def resize_map(x, side, mode="image"):
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

def pad_to_square(arr, pad_val=0.0):
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

def crop_rect(arr, y0, x0, h, w):
    return arr[y0:y0+h, x0:x0+w, ...] if arr.ndim == 3 else arr[y0:y0+h, x0:x0+w]

def estimate_well_mask(img, blur_sigma=3.0, well_is_brighter="auto"):
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

def square_crop_from_center_radius(mask_shape, center, radius, pad=8):
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

def compute_inner_boundary(inst_np: np.ndarray) -> np.ndarray:
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

def make_soft_boundary_from_instances(inst: np.ndarray,
                                      ring_width: int = 1,
                                      soft_band: int = 2,
                                      sigma: float = 1.0) -> np.ndarray:
    ring = compute_inner_boundary(inst).astype(bool)
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

def make_center_stem_from_centers(centers, shape):
    H, W = shape
    stem = np.zeros((H, W), dtype=np.float32)
    for (y, x) in centers or []:
        if 0 <= y < H and 0 <= x < W:
            stem[int(y), int(x)] = 1.0
    return stem

def make_center_heatmap(stem, sigma: Union[int, float] = 1.0):
    heat = stem.astype(np.float32)
    if sigma and sigma > 0:
        heat = ndi.gaussian_filter(heat, float(sigma))
        m = float(heat.max())
        if m > 0:
            heat /= m
    return heat.astype(np.float32)

def make_energy_from_instances(instances):
    cell = (instances > 0)
    dist = ndi.distance_transform_edt(cell).astype(np.float32)
    if cell.any():
        dmax = float(dist[cell].max())
        if dmax > 0:
            dist /= dmax
    dist[~cell] = 0.0
    return dist.astype(np.float32)

def crop_sim_meta_to_tile(meta, y0, x0, h, w):
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

def heal_watershed_gaps(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Return a healed *binary* foreground mask (no instance labels)."""
    mask_bin = (mask > 0)

    if radius <= 0:
        return mask_bin

    selem = morphology.disk(int(radius))

    closed = morphology.binary_closing(mask_bin, selem)
    grown  = morphology.binary_dilation(mask_bin, selem)

    # keep it from growing too far outward
    healed = np.logical_and(closed, grown)
    return healed

def seeded_watershed_from_mask(mask_healed: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Expand seed labels into mask_healed using distance-transform watershed.
    Keeps instances separate even if mask_healed is connected.
    """
    mask_healed = mask_healed.astype(bool)

    # make sure seeds are inside the allowed region
    markers = seeds.copy()
    markers[~mask_healed] = 0

    # distance inside foreground; watershed on -distance expands seeds to fill mask
    dist = ndi.distance_transform_edt(mask_healed)
    inst = watershed(-dist, markers=markers, mask=mask_healed)

    inst = inst.astype(np.int32)
    inst, _, _ = relabel_sequential(inst)
    return inst

# --- helpers for external COM/labels -------------------------------------

def load_com_labels_csv(
    image_name: str,
    folder: str,
    csv_path
):
    """
    Read rows from ONE combined CSV (ROOT/results/*.csv or a merged file)
    with columns:
      Folder, file_name, X, Y, mean_red, mean_green, mean_blue

    Filters by (Folder == folder) AND (file_name == image_name).

    Returns:
      centers: List[(cy, cx)]                 # (row, col)
      means  : List[(mean_r, mean_g, mean_b)] # floats
    """
    centers: List[Tuple[int, int]] = []
    means: List[Tuple[float, float, float]] = []

    if not os.path.exists(csv_path):
        return centers, means

    # normalize for safer matching (optional but helps)
    folder_key = folder.strip()
    name_key = image_name.strip()

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                if row.get("Folder", "").strip() != folder_key:
                    continue
                if row.get("file_name", "").strip() != name_key:
                    continue

                cx = float(row["X"])
                cy = float(row["Y"])

                centers.append((int(round(cy)), int(round(cx))))

            except Exception:
                continue

    return centers

def crop_external_meta_to_tile(meta_full: dict, y0: int, x0: int, h: int, w: int):
    """
    From a full-image external meta (with 'centers' and 'labels'),
    keep only entries inside the tile [y0:y0+h, x0:x0+w] and shift coords.
    Returns a NEW dict with:
      centers: [(ny, nx)] in tile coords
      labels : [int]
      n_cells: int
      frac_positive: float   # mean(label==1), mapping -1→0
    """
    new_meta = dict(meta_full)
    centers = meta_full.get("centers", [])
    labels  = meta_full.get("labels", [])

    kept_c = []
    kept_l = []

    for i, c in enumerate(centers):
        cy, cx = c
        ny = cy - y0
        nx = cx - x0
        if 0 <= ny < h and 0 <= nx < w:
            kept_c.append((int(ny), int(nx)))
            if isinstance(labels, (list, tuple, np.ndarray)) and i < len(labels):
                kept_l.append(int(labels[i]))

    new_meta["centers"] = kept_c
    if isinstance(labels, np.ndarray):
        new_meta["labels"] = np.array(kept_l, dtype=labels.dtype)
    else:
        new_meta["labels"] = kept_l

    n_cells_tile = len(kept_c)
    new_meta["n_cells"] = int(n_cells_tile)

    # map -1 to 0 for frac (same as pos/n in macro)
    if n_cells_tile > 0 and len(kept_l) == n_cells_tile:
        vals = [(1 if v == 1 else 0) for v in kept_l]
        new_meta["frac_positive"] = float(np.mean(vals))
    else:
        new_meta["frac_positive"] = 0.0

    return new_meta


def extract_real_image_feature_table_cv2(
    folders: Optional[Sequence[str | Path]] = None,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    recursive: bool = True,
    ignore_failures: bool = True,
):
    """
    Extract per-image RGB histogram features from real images.

    Purpose
    -------
    This is meant for later clustering of real image appearance into a few
    camera / rendering families.

    What it returns
    ---------------
    rows : list[dict]
        One dict per image with scalar features and histogram-bin features.
    feature_names : list[str]
        Ordered numeric feature names for clustering.
    X : np.ndarray, shape (N_images, N_features), dtype float32
        Numeric feature matrix built from rows and feature_names.

    Notes
    -----
    - Images are read with cv2 in BGR, then converted to RGB.
    - All values are converted to float32 in [0, 1].
    - Histograms are normalized per channel so each channel histogram sums to 1.
    - If sample_pixels is not None and an image is very large, a random subset
      of pixels is used for percentile / histogram / skew calculations.
      Mean and std are still computed on the sampled set only in this function.
      That is usually fine for clustering camera looks.
    """

    if folders is None:
        folders = EXT_IMAGES_FOLDERS
        if folders is None:
            raise ValueError("folders must be provided")

    folders = [Path(f) for f in folders]
    exts = tuple(e.lower() for e in exts)
    percentiles = tuple(float(p) for p in percentiles)

    # -------------------------
    # collect image paths
    # -------------------------
    image_paths = []
    for folder in folders:
        if not folder.exists():
            continue
        walker = folder.rglob("*") if recursive else folder.glob("*")
        for path in walker:
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)

    if not image_paths:
        raise RuntimeError("No image files found in the given folders.")

    # -------------------------
    # feature naming
    # -------------------------
    channel_names = ("r", "g", "b")
    feature_names = []

    # per-channel scalar stats
    for ch in channel_names:
        feature_names.append(f"{ch}_mean")
        feature_names.append(f"{ch}_std")
        feature_names.append(f"{ch}_skew")

        for p in percentiles:
            p_name = str(p).replace(".", "_")
            feature_names.append(f"{ch}_p{p_name}")

        for b in range(hist_bins):
            feature_names.append(f"{ch}_hist_{b:02d}")

    # global scalar stats
    feature_names.extend([
        "gray_mean",
        "gray_std",
        "gray_skew",
        "sat_mean",
        "sat_std",
        "sat_skew",
        "dark_frac",
        "bright_frac",
        "n_pixels_used",
        "aspect_ratio",
        "height",
        "width",
    ])

    rows = []

    # -------------------------
    # helpers
    # -------------------------
    def _safe_skew(x: np.ndarray) -> float:
        # x is 1D float32/float64
        m = float(x.mean())
        s = float(x.std())
        if s < 1e-12:
            return 0.0
        z = (x - m) / s
        return float(np.mean(z ** 3))

    def _norm_hist_01(x: np.ndarray, bins: int) -> np.ndarray:
        h, _ = np.histogram(x, bins=bins, range=(0.0, 1.0))
        h = h.astype(np.float32)
        h /= (h.sum() + 1e-8)
        return h

    # -------------------------
    # main loop
    # -------------------------
    for path in tqdm(image_paths, desc="Extracting real-image histogram features"):
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            if ignore_failures:
                continue
            raise RuntimeError(f"Could not read image: {path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W, _ = img.shape

        pixels = img.reshape(-1, 3)
        n_total = pixels.shape[0]

        if sample_pixels is not None and n_total > sample_pixels:
            idx = np.random.choice(n_total, size=sample_pixels, replace=False)
            pixels_use = pixels[idx]
        else:
            pixels_use = pixels

        # channel views
        r = pixels_use[:, 0]
        g = pixels_use[:, 1]
        b = pixels_use[:, 2]

        # gray and saturation
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        rgb_max = np.max(pixels_use, axis=1)
        rgb_min = np.min(pixels_use, axis=1)
        sat = np.where(rgb_max > 1e-8, (rgb_max - rgb_min) / (rgb_max + 1e-8), 0.0).astype(np.float32)

        row = {
            "path": str(path),
            "filename": path.name,
            "height": int(H),
            "width": int(W),
            "aspect_ratio": float(W / max(H, 1)),
            "n_pixels_used": int(pixels_use.shape[0]),
        }

        for ch_name, x in zip(channel_names, (r, g, b)):
            row[f"{ch_name}_mean"] = float(np.mean(x))
            row[f"{ch_name}_std"] = float(np.std(x))
            row[f"{ch_name}_skew"] = float(_safe_skew(x))

            pvals = np.percentile(x, percentiles)
            for p, val in zip(percentiles, pvals):
                p_name = str(p).replace(".", "_")
                row[f"{ch_name}_p{p_name}"] = float(val)

            hist = _norm_hist_01(x, bins=hist_bins)
            for i, val in enumerate(hist):
                row[f"{ch_name}_hist_{i:02d}"] = float(val)

        row["gray_mean"] = float(np.mean(gray))
        row["gray_std"] = float(np.std(gray))
        row["gray_skew"] = float(_safe_skew(gray))

        row["sat_mean"] = float(np.mean(sat))
        row["sat_std"] = float(np.std(sat))
        row["sat_skew"] = float(_safe_skew(sat))

        # crude clipping / tail mass indicators
        row["dark_frac"] = float(np.mean(gray <= 0.02))
        row["bright_frac"] = float(np.mean(gray >= 0.98))

        rows.append(row)

    if not rows:
        raise RuntimeError("No usable images were processed.")

    X = np.array(
        [[row[name] for name in feature_names] for row in rows],
        dtype=np.float32,
    )

    return rows, feature_names, X
