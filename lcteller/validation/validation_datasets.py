from __future__ import annotations
from typing import List, Optional, Tuple
import os
import json
import glob
import signal

import h5py
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import tifffile as tiff
from skimage import morphology
from skimage.measure import label as sklabel
from skimage.segmentation import relabel_sequential

from ..segmentation.image_dataset import (
    _make_soft_boundary_from_instances,
    _make_center_stem_from_centers,
    _make_center_heatmap,
    _make_energy_from_instances,
)
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
        # image
        img = tiff.imread(img_path).astype(np.float32, copy=False)
        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = img[..., 0]
            elif img.shape[2] > 3:
                img = img[..., :3]

        # to [0,1]
        if img.dtype.kind in ("u", "i"):
            vmax = 65535.0 if img.max() > 255 else 255.0
            img = img / max(vmax, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 2:
            img = np.concatenate([img, img[..., :1]], axis=-1)

        # mask (8-bit, 255=cell)
        msk = tiff.imread(mask_path)
        if msk.ndim > 2:
            msk = msk[..., 0]
        msk = (msk >= 255).astype(np.uint8)

        return img, msk

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
        img, msk = self._read_image_and_mask(img_path, mask_path)
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
            try: f.id.flush()
            except Exception: pass
            try:
                fd = f.id.get_vfd_handle()
                if fd is not None: os.fsync(fd)
            except Exception: pass

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

