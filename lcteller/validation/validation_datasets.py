from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import os
import json
import glob
import signal

import h5py
import numpy as np
import torch
import csv
import cv2
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import math

from skimage import morphology
from skimage.measure import label as sklabel
from skimage.segmentation import relabel_sequential

from ..segmentation.config import test_camera, test_scene

from ..segmentation.image_dataset import (
    _make_soft_boundary_from_instances,
    _make_center_stem_from_centers,
    _make_center_heatmap,
    _make_energy_from_instances,
    DiskSimCellsDataset
)
from ..segmentation import simulate_image
from ..segmentation.utils import collate_no_meta
from .utils import jsonify


def _find_pairs_strict(root_dir: str) -> List[Tuple[str, str]]:
    exts = (".tif", ".tiff")
    files = [
        p
        for ext in exts
        for p in glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True)
    ]
    pairs: List[Tuple[str, str]] = []
    for img_path in files:
        base = os.path.basename(img_path)
        stem, _ = os.path.splitext(base)
        if stem.lower().endswith("_mask"):
            continue
        d = os.path.dirname(img_path)
        m1 = os.path.join(d, f"{stem}_mask.tif")
        m2 = os.path.join(d, f"{stem}_mask.tiff")
        mask_path = m1 if os.path.exists(m1) else (m2 if os.path.exists(m2) else None)
        if mask_path is not None:
            pairs.append((os.path.abspath(img_path), os.path.abspath(mask_path)))
    pairs.sort()
    return pairs

def _heal_watershed_gaps(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Fix 1px (or very thin) background lines produced by ImageJ watershed.

    Steps:
      1. binarize (in case it's 8-bit 0/255 or 0/1/2/...),
      2. binary closing with a small disk → fills cuts *inside* the mask,
      3. AND with a dilated version of the original to avoid growing too far out.
    """
    mask_bin = (mask > 0)

    if radius <= 0:
        return mask_bin.astype(np.uint8)

    selem = morphology.disk(int(radius))

    # fills the splits
    closed = morphology.binary_closing(mask_bin, selem)

    # limit growth — stay within original mask + radius
    grown = morphology.binary_dilation(mask_bin, selem)

    healed = np.logical_and(closed, grown)
    return healed.astype(np.uint8)

# --- helpers for external COM/labels -------------------------------------

def _guess_data_csv_path(img_path: str, mask_path: str) -> str:
    """
    Try to locate the per-image CSV saved by ImageJ:
      Preferred: {mask_base_without '_mask'}_data.csv
      Fallback : {image_base}_data.csv
    """
    d_mask, mname = os.path.split(mask_path)
    base_mask, ext = os.path.splitext(mname)
    if base_mask.endswith("_mask"):
        base = base_mask[:-5]  # strip "_mask"
        cand = os.path.join(d_mask, f"{base}_data.csv")
        if os.path.exists(cand):
            return cand

    d_img, iname = os.path.split(img_path)
    base_img, _ = os.path.splitext(iname)
    cand2 = os.path.join(d_img, f"{base_img}_data.csv")
    return cand2  # may or may not exist; caller checks


def _load_com_labels_csv(csv_path: str):
    """
    Read X,Y,label from {image}_data.csv.
    Returns:
      centers: List[(cy, cx)]  # (row, col) in image coords
      labels : List[int]       # 1=pos, 0=neg, -1=ambiguous
    """
    centers = []
    labels = []
    if not os.path.exists(csv_path):
        return centers, labels  # empty → caller can fallback

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Expect headers X,Y,label
        for row in reader:
            try:
                # macro saved X (col), Y (row)
                cx = float(row["X"])
                cy = float(row["Y"])
                labs = int(float(row["label"]))
                centers.append((int(round(cy)), int(round(cx))))
                labels.append(int(labs))
            except Exception:
                # skip malformed rows
                continue
    return centers, labels


def _crop_external_meta_to_tile(meta_full: dict, y0: int, x0: int, h: int, w: int):
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

class ExternalCellsTilesDataset(Dataset):
    """
    External images + 8-bit masks → same output shape as SimCellsDataset(mode="tiles", n_tiles=-1)
    Adds support for per-image COM+labels CSV ({image}_data.csv) and passes
    them to per-tile meta (cropped/shifted).
    """
    def __init__(
        self,
        root_dir: str,
        target: int = 512,
        tile_overlap: int = 64,
        heal_radius: int = 1,
        boundary_ring_width: int = 1,
        boundary_soft_band: int = 2,
        boundary_sigma: float = 1.0,
        center_sigma: float = 1.0,
        transforms=None,   # optional Albumentations-style joint transform
    ):
        assert os.path.isdir(root_dir), f"not a dir: {root_dir}"
        self.root_dir = root_dir
        self.target = int(target)
        self.tile_overlap = int(tile_overlap)
        self.heal_radius = int(heal_radius)
        self.boundary_ring_width = int(boundary_ring_width)
        self.boundary_soft_band = int(boundary_soft_band)
        self.boundary_sigma = float(boundary_sigma)
        self.center_sigma = float(center_sigma)
        self.transforms = transforms

        self.pairs = _find_pairs_strict(root_dir)
        if not self.pairs:
            raise RuntimeError("no (img, img_mask) pairs found in external folder")

    def __len__(self):
        return len(self.pairs)

    def _read_image_and_mask(self, img_path: str, mask_path: str):
        """
        Returns:
          img_rgb_f32 : float32 [H, W, 3] in [0,1]
          msk_bin     : uint8   [H, W] with {0,1}
          info        : dict with bit-depth details for auditing
        """

        def _detect_bitdepth_u16(a: np.ndarray, p: float = 99.9):
            # a: uint16 HxW or HxWxC
            vmax = int(a.max())
            if vmax == 0:
                return 16, (1 << 16) - 1, False  # b, white, shifted

            # try left-shifted patterns (lower bits all zero), highest first
            for b in (14, 12, 10, 8):
                shift = 16 - b
                low_mask = (1 << shift) - 1
                if (a & low_mask).max() == 0:
                    white_shifted = ((1 << b) - 1) << shift
                    if vmax <= white_shifted:
                        return b, white_shifted, True

            # robust estimate from percentile
            sample = float(np.percentile(a, p))
            sample = max(1.0, sample)
            est_bits = int(math.ceil(math.log2(sample + 1.0)))
            est_bits = min(16, max(2, est_bits))
            allowed = (8, 10, 12, 14, 16)
            b = min(allowed, key=lambda k: abs(k - est_bits))
            white = (1 << b) - 1
            if vmax > white:
                return 16, (1 << 16) - 1, False
            return b, white, False

        # --- read image (keep native depth) ---
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        shape_in = tuple(img.shape)
        dtype_in = str(img.dtype)

        # ensure HxWxC
        if img.ndim == 2:
            img = img[:, :, None]

        # convert to RGB 3-ch
        if img.shape[2] >= 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 2:
            c0 = img[:, :, 0:1]
            c1 = img[:, :, 1:2]
            img = np.concatenate([c0, c1, c0], axis=-1)
        else:
            img = np.repeat(img, 3, axis=-1)

        # scale to float32 [0,1] with bit-depth sanity
        info = {
            "dtype_in": dtype_in,
            "shape_in": shape_in,
            "bit_depth": None,
            "white_level": None,
            "shifted": False,
        }

        if img.dtype == np.uint8:
            info["bit_depth"] = 8
            info["white_level"] = 255
            img_f32 = img.astype(np.float32) / 255.0

        elif img.dtype == np.uint16:
            b, white, shifted = _detect_bitdepth_u16(img)
            info.update({"bit_depth": b, "white_level": int(white), "shifted": bool(shifted)})

            # if left-shifted, undo shift before scaling
            if shifted and b < 16:
                shift = 16 - b
                img = (img >> shift).astype(np.uint16)
                white = (1 << b) - 1

            img_f32 = img.astype(np.float32) / float(white)
            img_f32 = np.clip(img_f32, 0.0, 1.0)

        else:
            # any other numeric type → clip to [0,1]
            img_f32 = np.clip(img.astype(np.float32), 0.0, 1.0)
            info.update({"bit_depth": 32, "white_level": 1, "shifted": False})

        # --- read mask (binary uint8) ---
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # accept both 0/255 and 0/1 sources
        if m.max() <= 1:
            msk_bin = (m >= 1).astype(np.uint8)
        else:
            msk_bin = (m >= 255).astype(np.uint8)

        return img_f32, msk_bin, info

    def _enumerate_full_tiles(self, H: int, W: int):
        th = self.target
        if H <= th or W <= th:
            return [(0, min(th, H), 0, min(th, W))]
        stride = max(1, th - self.tile_overlap)
        coords = []
        for y0 in range(0, H - th + 1, stride):
            for x0 in range(0, W - th + 1, stride):
                y1 = y0 + th
                x1 = x0 + th
                coords.append((y0, y1, x0, x1))
        return coords

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img, msk, read_info = self._read_image_and_mask(img_path, mask_path)
        H, W = msk.shape

        # heal watershed gaps if wanted
        msk_healed = _heal_watershed_gaps(msk, radius=self.heal_radius)

        # instances from mask
        inst_full = sklabel(msk_healed, connectivity=1).astype(np.int32)
        inst_full, _, _ = relabel_sequential(inst_full)

        # four target heads from full
        cell_full = (inst_full > 0).astype(np.float32)
        bound_full = _make_soft_boundary_from_instances(
            inst_full,
            ring_width=max(1, self.boundary_ring_width),
            soft_band=max(1, self.boundary_soft_band),
            sigma=self.boundary_sigma,
        ).astype(np.float32)

        # try to load COM+labels CSV
        csv_path = _guess_data_csv_path(img_path, mask_path)
        centers_csv, labels_csv = _load_com_labels_csv(csv_path)

        # centers: prefer CSV; fallback to instance centroids
        if centers_csv:
            centers_full = centers_csv
            labels_full = labels_csv
        else:
            centers_full = []
            max_id = int(inst_full.max())
            for k in range(1, max_id + 1):
                ys, xs = np.where(inst_full == k)
                if ys.size == 0:
                    continue
                cy = int(np.mean(ys))
                cx = int(np.mean(xs))
                centers_full.append((cy, cx))
            # If no CSV, labels unknown → map all to 0
            labels_full = [0 for _ in range(len(centers_full))]

        center_stem = _make_center_stem_from_centers(centers_full, (H, W))
        center_heat = _make_center_heatmap(center_stem, sigma=self.center_sigma)
        energy_full = _make_energy_from_instances(inst_full)

        # full meta (include COM+labels so tiles can crop them)
        full_meta = {
            "src_path": img_path,
            "mask_path": mask_path,
            "data_csv": csv_path if os.path.exists(csv_path) else "",
            "H_in": int(H),
            "W_in": int(W),
            "n_cells": int(inst_full.max()),
            "centers": [(int(y), int(x)) for (y, x) in centers_full],
            "labels": [int(v) for v in labels_full],  # 1/0/-1
            "read_info": read_info,
        }

        n_cells_full = len(centers_full)
        if n_cells_full > 0 and len(labels_full) == n_cells_full:
            frac_positive_full = float(np.mean([1 if v == 1 else 0 for v in labels_full]))
        else:
            frac_positive_full = 0.0

        full_meta["frac_positive"] = frac_positive_full
        # tile enumeration
        tiles = self._enumerate_full_tiles(H, W)

        imgs_out, tgts_out, inst_out, tiles_meta = [], [], [], []

        for (y0, y1, x0, x1) in tiles:
            # crop arrays
            img_t = img[y0:y1, x0:x1, :]
            cell_t = cell_full[y0:y1, x0:x1]
            bound_t = bound_full[y0:y1, x0:x1]
            center_t = center_heat[y0:y1, x0:x1]
            energy_t = energy_full[y0:y1, x0:x1]
            inst_t = inst_full[y0:y1, x0:x1]

            th = img_t.shape[0]
            tw = img_t.shape[1]
            pad_y = self.target - th
            pad_x = self.target - tw

            if pad_y > 0 or pad_x > 0:
                img_t = np.pad(img_t, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant", constant_values=0.0)
                cell_t = np.pad(cell_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                bound_t = np.pad(bound_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                center_t = np.pad(center_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                energy_t = np.pad(energy_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                inst_t = np.pad(inst_t, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)

            # relabel per tile
            inst_t, _, _ = relabel_sequential(inst_t.astype(np.int32))

            # per-tile meta with cropped COM+labels
            tile_meta = {
                "mode": "tiles",
                "tile_xy": (int(y0), int(x0)),
                "tile_hw": (self.target, self.target),
            }
            # crop COM+labels to this tile
            meta_tile = _crop_external_meta_to_tile(
                {"centers": full_meta["centers"], "labels": full_meta["labels"]},
                y0, x0, (y1 - y0), (x1 - x0),
            )
            tile_meta.update({
                "centers": meta_tile["centers"],      # in tile coords
                "labels": meta_tile["labels"],        # 1/0/-1
                "n_cells": meta_tile["n_cells"],
                "frac_positive": meta_tile["frac_positive"],
            })

            # pack 4-head target
            tgt_t = np.stack([cell_t, bound_t, center_t, energy_t], axis=0).astype(np.float32)

            # optional transforms (Albumentations style)
            if self.transforms is not None:
                out = self.transforms(
                    image=img_t,
                    masks=[cell_t, bound_t, center_t, energy_t],
                )
                img_t = out["image"]
                cell_t, bound_t, center_t, energy_t = out["masks"]
                tgt_t = np.stack([cell_t, bound_t, center_t, energy_t], axis=0).astype(np.float32)

            # to CHW
            img_chw = np.transpose(img_t.astype(np.float32), (2, 0, 1))

            imgs_out.append(img_chw)
            tgts_out.append(tgt_t)
            inst_out.append(inst_t.astype(np.int32))
            tiles_meta.append(tile_meta)

        # stack tensors
        imgs_t = torch.from_numpy(np.stack(imgs_out, axis=0).astype(np.float32))  # [T,3,S,S]
        tgts_t = torch.from_numpy(np.stack(tgts_out, axis=0).astype(np.float32))  # [T,4,S,S]
        inst_t = torch.from_numpy(np.stack(inst_out, axis=0).astype(np.int32))    # [T,S,S]

        extras = {
            "instance_labels": inst_t,
            "meta": {
                "full": full_meta,
                "tiles": tiles_meta,
            },
        }
        return imgs_t, tgts_t, extras

def create_external_cells_h5_tiles(
    root_dir: str,
    out_path: str,
    target: int = 512,
    tile_overlap: int = 64,
    heal_radius: int = 1,
    num_workers: int = 4,
    compression: Optional[str] = "lzf",
    flush_every: int = 8,
    resume: bool = True,
    transforms=None,
):
    """
    Export external images (with {_mask.tif} + {_data.csv}) into an HDF5 that
    mirrors create_dataset_h5:
        /imgs: float32 [N, T, 3, S, S]
        /tgts: float32 [N, T, 4, S, S]
        /inst: int32   [N, T, S, S]
        /meta: vlen JSON (one per image), holding {"full": ..., "tiles": [...]}
    The tile dimension T is growable (None) and may increase if later images
    have more tiles than earlier ones.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # dataset + dataloader
    ds = ExternalCellsTilesDataset(
        root_dir=root_dir,
        target=target,
        tile_overlap=tile_overlap,
        heal_radius=heal_radius,
        transforms=transforms,
    )
    N = len(ds)

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        persistent_workers=(num_workers > 0),
        pin_memory=False,
    )

    # graceful stop
    stop = {"flag": False}
    def _handle_signal(signum, frame): stop["flag"] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- peek first sample to get initial T0 ---
    it = iter(dl)
    try:
        first_imgs, first_tgts, first_extras = next(it)
    except StopIteration:
        raise RuntimeError("Empty dataset (no external pairs found).")

    # first batch shapes
    # first_imgs: [1, T0, 3, S, S]
    # first_tgts: [1, T0, 4, S, S]
    _, T0, C_img, S, _ = first_imgs.shape
    _, _, C_tgt, _, _ = first_tgts.shape
    assert int(S) == int(target), "target mismatch between dataset and writer"

    vlen_str = h5py.special_dtype(vlen=str)
    new_file = (not os.path.exists(out_path))

    with h5py.File(out_path, "a", libver="latest") as f:

        # file init / checks, matching create_dataset_h5 style
        if new_file:
            f.attrs.update({
                "version": 2,          # keep 2 for external pipeline
                "length": int(N),      # number of images
                "target": int(target),
                "tile_overlap": int(tile_overlap),
                "heal_radius": int(heal_radius),
                "T": int(T0),          # current max tiles per image
                "C_img": int(C_img),   # 3
                "C_tgt": int(C_tgt),   # 4
                "written": 0,
                "source": os.path.abspath(root_dir),
            })
            f.create_dataset(
                "imgs",
                shape=(N, T0, C_img, S, S),
                maxshape=(N, None, C_img, S, S),  # grow along tile dim
                dtype=np.float32,
                chunks=(max(1, min(16, N)), T0, C_img, S, S),
                compression=compression,
            )
            f.create_dataset(
                "tgts",
                shape=(N, T0, C_tgt, S, S),
                maxshape=(N, None, C_tgt, S, S),
                dtype=np.float32,
                chunks=(max(1, min(16, N)), T0, C_tgt, S, S),
                compression=compression,
            )
            f.create_dataset(
                "inst",
                shape=(N, T0, S, S),
                maxshape=(N, None, S, S),
                dtype=np.int32,
                chunks=(max(1, min(16, N)), T0, S, S),
                compression=compression,
            )
            f.create_dataset(
                "meta",
                shape=(N,),
                dtype=vlen_str,                 # ONE JSON per image (same as create_dataset_h5)
                chunks=(min(1024, N),),
            )
        else:
            # minimal compatibility checks (match target; read T from file)
            assert int(f.attrs["length"]) == int(N), "length mismatch"
            assert int(f.attrs["target"]) == int(target), "target mismatch"
            file_T = int(f.attrs.get("T", T0))
            # grow immediately if first sample needs more tiles
            if T0 > file_T:
                f["imgs"].resize((N, T0, C_img, S, S))
                f["tgts"].resize((N, T0, C_tgt, S, S))
                f["inst"].resize((N, T0, S, S))
                f.attrs.modify("T", int(T0))
            else:
                T0 = file_T
            if "written" not in f.attrs:
                f.attrs["written"] = 0

        # handles + state
        d_imgs, d_tgts, d_inst, d_meta = f["imgs"], f["tgts"], f["inst"], f["meta"]
        written = int(f.attrs["written"])

        # small helpers
        def _flush_safe():
            f.flush()
            try:
                f.id.flush()
            except Exception: 
                pass
            try:
                fd = f.id.get_vfd_handle()
                if fd is not None:
                    os.fsync(fd)
            except Exception:
                pass

        def _ensure_tile_dim(n_tiles_needed: int):
            cur_T = int(f.attrs.get("T", 1))
            if n_tiles_needed <= cur_T:
                return cur_T
            new_T = int(n_tiles_needed)
            d_imgs.resize((N, new_T, C_img, S, S))
            d_tgts.resize((N, new_T, C_tgt, S, S))
            d_inst.resize((N, new_T, S, S))
            f.attrs.modify("T", new_T)
            return new_T

        # write first sample if not resuming
        def _write_one(index: int, imgs_t: torch.Tensor, tgts_t: torch.Tensor, inst_t: torch.Tensor, meta_one: dict):
            # imgs_t: [T, 3, S, S], tgts_t: [T, 4, S, S], inst_t: [T, S, S]
            T = int(imgs_t.shape[0])
            _ensure_tile_dim(T)

            # write arrays (pad is not required; datasets expand to max T in file)
            d_imgs[index, :T, ...] = imgs_t.detach().cpu().numpy().astype(np.float32)
            d_tgts[index, :T, ...] = tgts_t.detach().cpu().numpy().astype(np.float32)
            d_inst[index, :T, ...] = inst_t.detach().cpu().numpy().astype(np.int32)

            # if file has larger T than this sample, zero the tail
            file_T = int(f.attrs["T"])
            if T < file_T:
                d_imgs[index, T:, ...] = 0.0
                d_tgts[index, T:, ...] = 0.0
                d_inst[index, T:, ...] = 0

            # ONE JSON per image (same as create_dataset_h5)
            d_meta[index] = json.dumps(jsonify(meta_one), separators=(",", ":"))

        # progress
        pbar = tqdm(total=N, initial=written, desc="export external h5", dynamic_ncols=True)
        img_since_flush = 0

        # fast-forward iterator if resuming
        if resume and written > 0:
            for _ in range(written):
                try: next(it)
                except StopIteration: break
        else:
            # write the peeked first sample at index 0
            if written == 0:
                _write_one(
                    0,
                    first_imgs[0],                 # [T0,3,S,S]
                    first_tgts[0],                 # [T0,4,S,S]
                    first_extras["instance_labels"][0],  # [T0,S,S]
                    first_extras["meta"],          # dict {"full":..., "tiles":[...]}
                )
                written = 1
                f.attrs.modify("written", int(written))
                pbar.update(1)
                img_since_flush += 1
                if (img_since_flush % max(1, flush_every)) == 0:
                    _flush_safe()
                    img_since_flush = 0

        # main loop
        cur_idx = written
        for imgs_tiles, tgts_tiles, extras in it:
            if stop["flag"]:
                break

            _write_one(
                cur_idx,
                imgs_tiles[0],                        # [T,3,S,S]
                tgts_tiles[0],                        # [T,4,S,S]
                extras["instance_labels"][0],        # [T,S,S]
                extras["meta"],                       # dict per image
            )

            cur_idx += 1
            written = cur_idx
            f.attrs.modify("written", int(written))
            pbar.update(1)

            img_since_flush += 1
            if (img_since_flush % max(1, flush_every)) == 0:
                _flush_safe()
                img_since_flush = 0

            if written >= N:
                break

        _flush_safe()
        pbar.close()
        print(f"[export] done: images={written}/{N}, T={int(f.attrs['T'])} → {out_path}")
        return out_path

class FullResSimValDataset(Dataset):
    """
    Returns (per index):
      img_t : float32 [1, 3, H, W] in [0,1]
      tgt_t : float32 [1, 4, H, W]
      extras: { "instance_labels": int32 [1, H, W], "meta": dict }
    """
    def __init__(
        self,
        length: int,
        scene_cfg,
        camera_cfg,
        *,
        rng_seed: int = 123,
        boundary_ring_width: int = 1,
        boundary_soft_band: int = 2,
        boundary_sigma: float = 1.0,
        center_sigma: float = 1.0,
    ):
        self.length = int(length)
        self.scene_cfg = scene_cfg
        self.camera_cfg = camera_cfg

        self.rng_seed = int(rng_seed)
        self.boundary_ring_width = int(boundary_ring_width)
        self.boundary_soft_band = int(boundary_soft_band)
        self.boundary_sigma = float(boundary_sigma)
        self.center_sigma = float(center_sigma)

        # probe one sample to fix H, W
        rng0 = np.random.default_rng(self.rng_seed)
        skw = self.scene_cfg.sample_kwargs(rng0, camera=self.camera_cfg)
        skw.setdefault("return_targets", True)
        img0, _, _ = simulate_image(**skw)  # (H,W,3)
        H0, W0, C0 = img0.shape
        if C0 != 3:
            raise ValueError(f"Simulator must return 3 channels, got {C0}")
        self.H, self.W = int(H0), int(W0)

    def __len__(self) -> int:
        return self.length

    def _build_targets(self, inst_lbl: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        bound_soft = _make_soft_boundary_from_instances(
            inst_lbl.astype(np.int32),
            ring_width=max(1, self.boundary_ring_width),
            soft_band=max(1, self.boundary_soft_band),
            sigma=self.boundary_sigma,
        ).astype(np.float32)

        # centers from instances
        centers = []
        lbl = inst_lbl.astype(np.int32)
        for k in range(1, int(lbl.max()) + 1):
            ys, xs = np.where(lbl == k)
            if ys.size:
                centers.append((int(np.mean(ys)), int(np.mean(xs))))
        center_stem = _make_center_stem_from_centers(centers, (H, W))
        center_heat = _make_center_heatmap(center_stem, sigma=self.center_sigma).astype(np.float32)
        energy = _make_energy_from_instances(lbl).astype(np.float32)

        return np.stack([cell_mask.astype(np.float32), bound_soft, center_heat, energy], axis=0).astype(np.float32)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.rng_seed ^ int(idx))
        skw = self.scene_cfg.sample_kwargs(rng, camera=self.camera_cfg)
        skw.setdefault("return_targets", True)
        img, meta, targets = simulate_image(**skw)  # (H,W,3)

        if img.shape != (self.H, self.W, 3):
            raise ValueError(f"Sample {idx}: expected {(self.H, self.W, 3)}, got {img.shape}")

        cell = targets["cell_mask"].astype(np.float32)     # (H,W)
        inst = targets["instance_labels"].astype(np.int32) # (H,W)
        inst_rel, _, _ = relabel_sequential(inst)

        tgt = self._build_targets(inst_rel, cell)          # [4,H,W]
        img_cyx = np.transpose(img.astype(np.float32), (2, 0, 1))  # [3,H,W]

        img_t = torch.from_numpy(img_cyx[None, ...])       # [1,3,H,W]
        tgt_t = torch.from_numpy(tgt[None, ...])           # [1,4,H,W]
        inst_t = torch.from_numpy(inst_rel[None, ...])     # [1,H,W]

        extras = {"instance_labels": inst_t, "meta": meta}
        return img_t, tgt_t, extras


def create_validation_h5_fullres(
    out_path: str,
    length: int,
    batch_size: int = 4,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    compression: Optional[str] = "lzf",          # None | "lzf" | "gzip"
    rng_seed: int = 187,
    gen_batch_size: int = 16,
    num_workers_gen: int = 16,
    compression_level: Optional[int] = None,     # for gzip
    flush_every: int = 16,
    resume: bool = True,
    camera_cfg=None,
    scene_cfg=None,
    progress_desc: str = "export fullres h5"
) -> Tuple[int, int]:
    """
    Build a full-res HDF5 using a multi-worker DataLoader and write:
      /imgs: float32 [N, 1, 3, S, S]
      /tgts: float32 [N, 1, 4, S, S]
      /inst: int32   [N, 1, S, S]
      /meta: vlen UTF-8 JSON

    Returns (N, S).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if camera_cfg is None:
        camera_cfg = test_camera()
    if scene_cfg is None:
        scene_cfg = test_scene()


    # signals for clean stop
    stop = {"flag": False}
    def _handle_signal(signum, frame): stop["flag"] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # resume offset read happens after file open
    vlen_str = h5py.string_dtype(encoding="utf-8")
    new_file = (not os.path.exists(out_path))

    gen_dataset = FullResSimValDataset(
        length=length,
        rng_seed=rng_seed,
        camera_cfg=camera_cfg,
        scene_cfg=scene_cfg,
    )
    N, H, W = len(gen_dataset), gen_dataset.H, gen_dataset.W
    T_fixed = 1

    with h5py.File(out_path, "a", libver="latest") as f:
        # create or validate file
        if new_file:
            f.attrs.update({
                "version": 1,
                "length": int(N),
                "H": int(H),
                "W": int(W),
                "T": int(T_fixed),
                "C_img": 3,
                "C_tgt": 4,
                "written": 0,
                "source": "simulate_image(fullres)",
            })

            dargs: Dict[str, Any] = {}
            if compression:
                dargs["compression"] = compression
                if compression == "gzip" and compression_level is not None:
                    dargs["compression_opts"] = int(compression_level)

            chunk_N = max(1, min(16, N))
            f.create_dataset("imgs",
                shape=(N, T_fixed, 3, H, W), dtype="float32",
                chunks=(chunk_N, T_fixed, 3, min(H, 256), min(W, 256)), **dargs)
            f.create_dataset("tgts",
                shape=(N, T_fixed, 4, H, W), dtype="float32",
                chunks=(chunk_N, T_fixed, 4, min(H, 256), min(W, 256)), **dargs)
            f.create_dataset("inst",
                shape=(N, T_fixed, H, W), dtype="int32",
                chunks=(chunk_N, T_fixed, min(H, 256), min(W, 256)), **dargs)
            f.create_dataset("meta", shape=(N,), dtype=vlen_str, chunks=(min(1024, N),))
        else:
            assert int(f.attrs["length"]) == int(N), "length mismatch"
            assert int(f.attrs["H"]) == int(H), "H mismatch"
            assert int(f.attrs["W"]) == int(W), "W mismatch"
            assert int(f.attrs["T"]) == int(T_fixed), "T mismatch"
            if "written" not in f.attrs:
                f.attrs["written"] = 0

        d_imgs, d_tgts, d_inst, d_meta = f["imgs"], f["tgts"], f["inst"], f["meta"]
        written = int(f.attrs["written"])
        if written >= N:
            print(f"[export] already complete: {written}/{N} → {out_path}")
            return N, (H, W)

        # build loader over the unwritten tail
        if resume and written > 0:
            gen_subset = Subset(gen_dataset, list(range(written, N)))
        else:
            gen_subset = gen_dataset

        dl = DataLoader(
            gen_subset,
            batch_size=gen_batch_size,
            shuffle=False,
            num_workers=int(num_workers_gen),
            pin_memory=bool(pin_memory),
            persistent_workers=(num_workers > 0),
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
            drop_last=False,
            collate_fn=collate_no_meta,
        )

        def _flush_safe():
            f.flush()
            try:
                f.id.flush()
            except Exception:
                pass
            try:
                fd = f.id.get_vfd_handle()
                if fd is not None:
                    os.fsync(fd)
            except Exception:
                pass

        def _write_batch(start_abs: int,
                         imgs_b: torch.Tensor,          # [B,1,3,H,W]
                         tgts_b: torch.Tensor,          # [B,1,4,H,W]
                         inst_b: torch.Tensor,          # [B,1,H,W]
                         metas_b: List[dict]) -> int:
            B = imgs_b.shape[0]
            end_abs = start_abs + B
            d_imgs[start_abs:end_abs, :, :, :, :] = imgs_b.cpu().numpy().astype(np.float32)
            d_tgts[start_abs:end_abs, :, :, :, :] = tgts_b.cpu().numpy().astype(np.float32)
            d_inst[start_abs:end_abs, :, :, :]    = inst_b.cpu().numpy().astype(np.int32)
            for j in range(B):
                try:
                    d_meta[start_abs + j] = json.dumps(metas_b[j], separators=(",", ":"))
                except TypeError:
                    d_meta[start_abs + j] = json.dumps(jsonify(metas_b[j]), separators=(",", ":"))
            return B

        # --- progress bar here ---
        pbar = tqdm(total=N, initial=written, desc=progress_desc, dynamic_ncols=True)
        idx_next = written
        since_flush = 0

        for imgs_b, tgts_b, extras_b in dl:
            if stop["flag"]:
                break
            wrote = _write_batch(idx_next, imgs_b, tgts_b, extras_b["instance_labels"], extras_b["meta"])
            idx_next += wrote
            pbar.update(wrote)

            f.attrs.modify("written", int(idx_next))
            since_flush += wrote
            if (since_flush % max(1, flush_every)) == 0:
                _flush_safe(); since_flush = 0

            if idx_next >= N:
                break

        _flush_safe()
        pbar.close()
        print(f"[export] done: images={idx_next}/{N}, T=1, size={H}x{W} → {out_path}")
        return N, (H, W)
