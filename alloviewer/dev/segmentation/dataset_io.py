import os
import numpy as np
import torch
import signal
from torch.utils.data import Dataset, DataLoader
import glob
import json
import h5py
from tqdm import tqdm

from typing import Optional, Sequence, List, Tuple, Any, Dict

from .image_dataset import SimCellsDataset, ExternalCellsTilesDataset

from .config import default_scene, default_camera, UNET_MEAN, UNET_STD

from .io_utils import (
    imwrite_tiff_tifffile,
    scale_to_10bit,
    init_or_validate_varT_in_file,
    jsonify,
    flush_safe_file,
    setup_stop_flag
)

from .utils import (
    collate_no_meta,
    crop_rect,
    pad_to_square,
    resize_map,
    make_soft_boundary_from_instances,
    make_center_stem_from_centers,
    make_center_heatmap,
    make_energy_from_instances,
    crop_sim_meta_to_tile,
)


class H5CellsDataset(Dataset):
    """
    Generic HDF5-backed dataset for cells/tiles.

    Supports:
      - Tiled layout (created from ExternalCellsTilesDataset or similar):
          /imgs: float32 [N, T_max, 3, S, S]
          /tgts: float32 [N, T_max, C, S, S]
          /inst: int32   [N, T_max, S, S]
          /meta: vlen JSON (one per sample, with meta["tiles"] list)

        Real #tiles for sample i = len(meta["tiles"]).
        We return only those real tiles.

      - Single-tile layout (DiskSim style):
          /imgs: float32 [N, 1, 3, S, S]
          /tgts: float32 [N, 1, C, S, S]
          /inst: int32   [N, 1, S, S]
          /meta: vlen JSON (may or may not have "tiles")

        In this case T_max == 1 and:
          - if meta["tiles"] exists → T_real = len(meta["tiles"])
          - else → T_real = 1

    __getitem__ returns:
      img_t  : torch.float32 [T, 3, S, S]
      tgt_t  : torch.float32 [T, C, S, S]
      extras : {
        "instance_labels": torch.int32 [T, S, S],
        "meta": dict
      }
    """

    def __init__(self, h5_path: str, indices: Optional[Sequence[int]] = None):
        super().__init__()
        self.h5_path = str(h5_path)
        self._h5: Optional[h5py.File] = None
        self._imgs = self._tgts = self._inst = self._meta = None

        # lightweight probe to read shapes; no persistent handle here
        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as f:
            imgs = f["imgs"]
            tgts = f["tgts"]

            self.N = int(imgs.shape[0])
            self.T_max = int(imgs.shape[1])
            self.C_img = int(imgs.shape[2])
            self.S = int(imgs.shape[3])
            self.C_tgt = int(tgts.shape[2])

            if "inst" not in f or "meta" not in f:
                raise RuntimeError("HDF5 file must contain datasets 'inst' and 'meta'.")

        # subset of indices (optional)
        if indices is None:
            self.idx = np.arange(self.N, dtype=np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("indices must be 1D")
            if (idx < 0).any() or (idx >= self.N).any():
                raise ValueError("indices out of range")
            self.idx = idx

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    def _ensure_open(self):
        """Open HDF5 file (per worker) on first access."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._imgs = self._h5["imgs"]
            self._tgts = self._h5["tgts"]
            self._inst = self._h5["inst"]
            self._meta = self._h5["meta"]

    def _normalize_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [T, 3, S, S], float32, values in [0,1]
        returns: normalized imgs with per-channel (x - mean) / std
        """
        mean = torch.as_tensor(UNET_MEAN, dtype=imgs.dtype, device=imgs.device).view(1, 3, 1, 1)
        std  = torch.as_tensor(UNET_STD,  dtype=imgs.dtype, device=imgs.device).view(1, 3, 1, 1)
        return (imgs - mean) / std

    def __getitem__(self, i: int):
        self._ensure_open()
        k = int(self.idx[i])

        imgs = self._imgs[k]   # [T_max, 3, S, S]
        tgts = self._tgts[k]   # [T_max, C, S, S]
        inst = self._inst[k]   # [T_max, S, S]
        meta_raw = self._meta[k]

        # h5py vlen str can be bytes or str depending on build
        if isinstance(meta_raw, bytes):
            meta = json.loads(meta_raw.decode("utf-8"))
        else:
            meta = json.loads(meta_raw)

        tiles_meta = meta.get("tiles", [])
        if tiles_meta:
            T_real = len(tiles_meta)
        else:
            # fallback for single-tile data without a "tiles" list
            T_real = imgs.shape[0]

        imgs = imgs[:T_real]  # [T,3,S,S]
        tgts = tgts[:T_real]  # [T,C,S,S]
        inst = inst[:T_real]  # [T,S,S]

        imgs_t = torch.from_numpy(np.asarray(imgs, dtype=np.float32))
        tgts_t = torch.from_numpy(np.asarray(tgts, dtype=np.float32))
        inst_t = torch.from_numpy(np.asarray(inst, dtype=np.int32))

        imgs_t = self._normalize_imgs(imgs_t)

        extras = {
            "instance_labels": inst_t,
            "meta": meta,
        }
        return imgs_t, tgts_t, extras

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
            self._imgs = self._tgts = self._inst = self._meta = None

    def __del__(self):
        self.close()

class TiledH5Dataset(H5CellsDataset):
    """
    Backwards-compatible alias for tiled H5 datasets.

    Same behaviour as H5CellsDataset, but keeps the old name.
    """
    def __init__(self, h5_path: str, indices: Optional[Sequence[int]] = None):
        super().__init__(h5_path=h5_path, indices=indices)

class DiskSimCellsDataset(H5CellsDataset):
    """
    Backwards-compatible alias for single-tile H5 datasets.

    Optionally checks that T_max == 1.
    """
    def __init__(self, h5_path: str, indices: Optional[Sequence[int]] = None):
        super().__init__(h5_path=h5_path, indices=indices)
        # optional sanity: enforce single tile layout if you want
        if self.T_max != 1:
            raise ValueError(
                f"DiskSimCellsDataset expects T_max == 1, got {self.T_max}. "
                f"Use H5CellsDataset or TiledH5Dataset instead."
            )

def create_sim_cells_dataset_h5(
    out_path: str,
    length: int,
    mode: str = "crop_well_resize",            # "pad_resize" | "crop_well_resize" | "tiles" | "fullres"
    n_tiles: Optional[int] = 1,                # >0 random tiles, -1 = full cover, None/0 -> 1
    tile_overlap: int = 64,
    target: int = 512,
    rng_seed: int = 187,
    gen_batch_size: int = 16,
    num_workers_gen: int = 16,
    compression: Optional[str] = "lzf",
    flush_every: int = 8,
    resume: bool = True,
    camera_cfg=None,
    scene_cfg=None,
    camera_style_cfg=None,
):
    """
    Pre-allocate a single HDF5 and append EXACT tensors from SimCellsDataset.__getitem__:

      /imgs: float32 [N, T, 3, S, S]
      /tgts: float32 [N, T, C, S, S]
      /inst: int32   [N, T, S, S]
      /meta: vlen JSON (one per sample)

    Now supports *variable* T per sample (because in tiles mode with n_tiles=-1
    the number of tiles depends on image size). We start with T0 from the first
    batch and grow the tile dimension if we later see a sample with more tiles.
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if camera_cfg is None:
        camera_cfg = default_camera()
    if scene_cfg is None:
        scene_cfg = default_scene()

    # stop flag with signal handling
    stop = setup_stop_flag()

    # dataset & dataloader (generation on CPU)
    if not n_tiles:
        n_tiles = 1

    ds = SimCellsDataset(
        length=length,
        mode=mode,
        target=target,
        n_tiles=n_tiles,
        tile_overlap=tile_overlap,
        rng_seed=rng_seed,
        camera_cfg=camera_cfg,
        scene_cfg=scene_cfg,
        camera_style_cfg=camera_style_cfg
    )
    dl = DataLoader(
        ds,
        batch_size=gen_batch_size,
        shuffle=False,
        num_workers=int(num_workers_gen),
        pin_memory=False,
        persistent_workers=(num_workers_gen > 0),
        prefetch_factor=4,
        drop_last=False,
        collate_fn=collate_no_meta,
    )

    # peek first batch
    it = iter(dl)
    try:
        first_imgs, first_tgts, first_extras = next(it)
    except StopIteration:
        raise RuntimeError("Empty dataset (length=0).")

    B0, T0, C_img, H, W = first_imgs.shape
    _, _, C_tgt, _, _ = first_tgts.shape

    new_file = (not os.path.exists(out_path))

    with h5py.File(out_path, "a", libver="latest") as f:
        # init or validate HDF5 structure
        d_imgs, d_tgts, d_inst, d_meta, written = init_or_validate_varT_in_file(
            f,
            new_file=new_file,
            length=length,
            T0=T0,
            C_img=C_img,
            C_tgt=C_tgt,
            H=H,
            W=W,
            compression=compression,
            chunk_N=gen_batch_size,
            extra_attrs={
                "mode": mode,
                "target": int(target),
                "rng_seed": int(rng_seed),
            },
            check_attrs={
                "mode": mode,
                "target": int(target),
            },
        )

        if written >= length:
            print(f"[export] already complete: {written}/{length}")
            return out_path

        pbar = tqdm(total=length, initial=written, desc="export h5", dynamic_ncols=True)
        last_flush = 0

        def _ensure_tile_dim(n_tiles_needed: int):
            cur_T = int(f.attrs.get("T", 1))
            if n_tiles_needed <= cur_T:
                return cur_T
            new_T = int(n_tiles_needed)
            d_imgs.resize((length, new_T, C_img, H, W))
            d_tgts.resize((length, new_T, C_tgt, H, W))
            d_inst.resize((length, new_T, H, W))
            f.attrs.modify("T", new_T)
            return new_T

        def _write_slice(imgs, tgts, inst, metas):
            nonlocal written, last_flush
            # imgs: [B, T, 3, S, S]
            B, T, _, _, _ = imgs.shape

            _ensure_tile_dim(T)

            end = min(length, written + B)
            take = end - written
            if take <= 0:
                return 0

            d_imgs[written:end, :T, ...] = imgs[:take].detach().cpu().numpy().astype(np.float32)
            d_tgts[written:end, :T, ...] = tgts[:take].detach().cpu().numpy().astype(np.float32)
            d_inst[written:end, :T, ...] = inst[:take].detach().cpu().numpy().astype(np.int32)
            d_meta[written:end] = [
                json.dumps(jsonify(m), separators=(",", ":"))
                for m in metas[:take]
            ]

            written = end
            f.attrs.modify("written", int(written))
            last_flush += 1
            if (last_flush % int(max(1, flush_every))) == 0:
                flush_safe_file(f)

            pbar.update(take)
            return take

        # fast-forward if resuming
        to_skip = written
        if resume and to_skip > 0:
            skip_batches = to_skip // gen_batch_size
            for _ in range(skip_batches):
                try:
                    next(it)
                except StopIteration:
                    break
            skip_left = to_skip % gen_batch_size
            if skip_left > 0:
                try:
                    b_img, b_tgt, b_ex = next(it)
                    if b_img.shape[0] > skip_left:
                        _write_slice(
                            b_img[skip_left:], b_tgt[skip_left:],
                            b_ex["instance_labels"][skip_left:],
                            b_ex["meta"][skip_left:],
                        )
                except StopIteration:
                    pass
        else:
            # write first batch we already have
            _write_slice(
                first_imgs, first_tgts,
                first_extras["instance_labels"],
                first_extras["meta"],
            )

        # main loop
        for imgs, tgts, extras in it:
            if stop["flag"]:
                break
            _write_slice(imgs, tgts, extras["instance_labels"], extras["meta"])
            if written >= length:
                break

        flush_safe_file(f)
        pbar.close()
        print(f"done: {written}/{length} → {out_path}")
        return out_path


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
    Export external images (with *_mask.tif + *_data.csv) into an HDF5 that
    mirrors create_sim_cells_dataset_h5:

        /imgs: float32 [N, T, 3, S, S]
        /tgts: float32 [N, T, 4, S, S]
        /inst: int32   [N, T, S, S]
        /meta: vlen JSON (one per image), holding:
               {
                 "full":  <external full meta>,
                 "tiles": [tile_meta_0, ..., tile_meta_{T-1}],
                 "sim_kwargs": None
               }

    The tile dimension T is growable and may increase if later images
    have more tiles than earlier ones.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

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
        collate_fn=collate_no_meta,
    )

    stop = setup_stop_flag()

    # peek first sample
    it = iter(dl)
    try:
        first_imgs, first_tgts, first_extras = next(it)
    except StopIteration:
        raise RuntimeError("Empty dataset (no external pairs found).")

    _, T0, C_img, H, W = first_imgs.shape
    _, _, C_tgt, _, _ = first_tgts.shape
    assert int(H) == int(target), "target mismatch between dataset and writer"

    new_file = (not os.path.exists(out_path))

    with h5py.File(out_path, "a", libver="latest") as f:
        # init or validate HDF5 structure
        d_imgs, d_tgts, d_inst, d_meta, written = init_or_validate_varT_in_file(
            f,
            new_file=new_file,
            length=N,
            T0=T0,
            C_img=C_img,
            C_tgt=C_tgt,
            H=H,
            W=W,
            compression=compression,
            chunk_N=max(1, min(16, N)),
            extra_attrs={
                "target": int(target),
                "tile_overlap": int(tile_overlap),
                "heal_radius": int(heal_radius),
                "source": os.path.abspath(root_dir),
            },
            check_attrs={
                "target": int(target),
            },
        )

        def _ensure_tile_dim(n_tiles_needed: int):
            cur_T = int(f.attrs.get("T", 1))
            if n_tiles_needed <= cur_T:
                return cur_T
            new_T = int(n_tiles_needed)
            d_imgs.resize((N, new_T, C_img, H, W))
            d_tgts.resize((N, new_T, C_tgt, H, W))
            d_inst.resize((N, new_T, H, W))
            f.attrs.modify("T", new_T)
            return new_T

        def _write_one(index: int, imgs_t: torch.Tensor, tgts_t: torch.Tensor, inst_t: torch.Tensor, meta_one: dict):
            # imgs_t: [T,3,S,S], tgts_t: [T,4,S,S], inst_t: [T,S,S]
            T = int(imgs_t.shape[0])
            _ensure_tile_dim(T)

            d_imgs[index, :T, ...] = imgs_t.detach().cpu().numpy().astype(np.float32)
            d_tgts[index, :T, ...] = tgts_t.detach().cpu().numpy().astype(np.float32)
            d_inst[index, :T, ...] = inst_t.detach().cpu().numpy().astype(np.int32)

            file_T = int(f.attrs["T"])
            if T < file_T:
                d_imgs[index, T:, ...] = 0.0
                d_tgts[index, T:, ...] = 0.0
                d_inst[index, T:, ...] = 0

            d_meta[index] = json.dumps(jsonify(meta_one), separators=(",", ":"))

        pbar = tqdm(total=N, initial=written, desc="export external h5", dynamic_ncols=True)
        since_flush = 0

        # handle resume / writing first sample
        if resume and written > 0:
            for _ in range(written):
                try:
                    next(it)
                except StopIteration:
                    break
        else:
            if written == 0:
                _write_one(
                    0,
                    first_imgs[0],                        # [T0,3,S,S]
                    first_tgts[0],                        # [T0,4,S,S]
                    first_extras["instance_labels"][0],  # [T0,S,S]
                    first_extras["meta"][0],             # dict {"full":..., "tiles":[...], "sim_kwargs": None}
                )
                written = 1
                f.attrs.modify("written", int(written))
                pbar.update(1)
                since_flush += 1
                if (since_flush % max(1, flush_every)) == 0:
                    flush_safe_file(f)
                    since_flush = 0

        idx = written
        for imgs_b, tgts_b, extras_b in it:
            if stop["flag"]:
                break

            _write_one(
                idx,
                imgs_b[0],                            # [T,3,S,S]
                tgts_b[0],                            # [T,4,S,S]
                extras_b["instance_labels"][0],      # [T,S,S]
                extras_b["meta"][0],                 # dict
            )

            idx += 1
            written = idx
            f.attrs.modify("written", int(written))
            pbar.update(1)

            since_flush += 1
            if (since_flush % max(1, flush_every)) == 0:
                flush_safe_file(f)
                since_flush = 0

            if written >= N:
                break

        flush_safe_file(f)
        pbar.close()
        print(f"[export] done: images={written}/{N}, T={int(f.attrs['T'])} → {out_path}")
        return out_path


def export_h5_to_tiff(
    h5_path: str,
    out_dir: str,
    n_images: Optional[int] = None,           # cap how many to export
    indices: Optional[Sequence[int]] = None,  # explicit sample indices
    scale_mode: str = "clip01",               # 'clip01' | 'minmax' | 'percentile'
    scale_percentiles: Tuple[float, float] = (1.0, 99.0),
    overwrite: bool = False,
) -> None:
    """
    Open the HDF5 once (context manager) and export up to n_images
    from /imgs as 10-bit data (0..1023) stored in uint16 TIFFs via OpenCV.

    Channel handling:
      - C == 3  -> one 3-channel TIFF (H,W,3)
      - C == 1  -> one single-channel TIFF (H,W)
      - C  > 3  -> one TIFF per channel: *_c{idx}.tif
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        # basic checks
        for key in ("imgs", "tgts", "inst", "meta"):
            if key not in f:
                raise KeyError(f"Missing dataset '{key}' in HDF5.")

        imgs = f["imgs"]     # float32 [N, 1, C, S, S]
        meta_ds = f["meta"]  # vlen JSON strings
        N = int(imgs.shape[0])

        # pick samples
        if indices is None:
            chosen = list(range(N))
        else:
            chosen = [int(x) for x in indices]
            if any(x < 0 or x >= N for x in chosen):
                raise ValueError("indices out of range")

        if n_images is not None:
            chosen = chosen[:int(n_images)]

        for k in chosen:
            # read one sample
            arr = imgs[k]                                   # [1, C, S, S]
            img = np.asarray(arr, dtype=np.float32).squeeze(0)  # -> (C, S, S)
            C, H, W = img.shape

            # meta for naming (optional)
            mj = meta_ds[k]
            meta = json.loads(mj.decode("utf-8")) if isinstance(mj, bytes) else json.loads(mj)
            uid = meta.get("uid") or meta.get("id")

            # scale to 10-bit (uint16 0..1023)
            img_u16 = scale_to_10bit(img, mode=scale_mode, percentiles=scale_percentiles)

            stem = f"{k:06d}_{uid}" if uid is not None else f"{k:06d}"

            if C == 3:
                # stack to (H, W, 3). OpenCV expects BGR; we pass as-is.
                im = np.moveaxis(img_u16, 0, -1)  # (H, W, 3)
                out_path = os.path.join(out_dir, f"{stem}.tif")
                if (not overwrite) and os.path.exists(out_path):
                    raise FileExistsError(f"File exists: {out_path}. Set overwrite=True to replace.")
                imwrite_tiff_tifffile(out_path, im)
                written.append(out_path)

            elif C == 1:
                # single-channel
                im = img_u16[0]  # (H, W)
                out_path = os.path.join(out_dir, f"{stem}.tif")
                if (not overwrite) and os.path.exists(out_path):
                    raise FileExistsError(f"File exists: {out_path}. Set overwrite=True to replace.")
                imwrite_tiff_tifffile(out_path, im)
                written.append(out_path)

            else:
                # more than 3 channels -> one file per channel
                for c in range(C):
                    im = img_u16[c]  # (H, W)
                    out_path = os.path.join(out_dir, f"{stem}_c{c}.tif")
                    if (not overwrite) and os.path.exists(out_path):
                        raise FileExistsError(f"File exists: {out_path}. Set overwrite=True to replace.")
                    imwrite_tiff_tifffile(out_path, im)
                    written.append(out_path)

    return

def create_tiled_from_fullres(
    in_path: str,
    out_path: str,
    target: int = 512,
    tile_overlap: int = 64,
    boundary_ring_width: int = 1,
    boundary_soft_band: int = 2,
    boundary_sigma: float = 1.0,
    center_sigma: float = 1.0,
    compression: Optional[str] = "lzf",          # None | "lzf" | "gzip"
    compression_level: Optional[int] = None,     # for gzip
    flush_every: int = 16,
    resume: bool = True,
    progress_desc: str = "export tiled h5",
) -> Tuple[int, int, int]:
    """
    Read a full-res validation HDF5 file created by create_validation_h5_fullres
    and build a tiled HDF5 file where each full image is split into fixed tiles.

    Input file layout (full-res):
      /imgs: float32 [N, 1, 3, H_in, W_in]
      /tgts: float32 [N, 1, 4, H_in, W_in]   (ignored, we recompute per tile)
      /inst: int32   [N, 1, H_in, W_in]
      /meta: vlen UTF-8 JSON (per scene, simulator meta)

    Output file layout (tiled):
      /imgs: float32 [N, T_tiles, 3, target, target]
      /tgts: float32 [N, T_tiles, 4, target, target]
      /inst: int32   [N, T_tiles, target, target]
      /meta: vlen UTF-8 JSON [N]
             each entry is a dict:
                 {
                   "full": <full-image simulator meta>,
                   "tiles": [
                      {
                        "mode": "tiles",
                        "tile_xy": (y0, x0),
                        "tile_hw": (target, target),
                        "sim_meta": <tile-level simulator meta>,
                        "full_meta": <full-image simulator meta>,
                      },
                      ...
                   ],
                   "sim_kwargs": {},   # cannot be recovered, left empty
                 }

    Tiling follows the same logic as SimCellsDataset with:
        mode="tiles", n_tiles == -1,
        stride = target - tile_overlap

    Targets per tile are recomputed from inst using:
        - make_soft_boundary_from_instances
        - make_center_stem_from_centers + make_center_heatmap
        - make_energy_from_instances

    Returns:
        (N, T_tiles, target)
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- signal handling ---
    stop = {"flag": False}

    def _handle_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- open input file, read basic info ---
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input H5 not found: {in_path}")

    with h5py.File(in_path, "r", libver="latest") as f_in:
        d_imgs_in = f_in["imgs"]  # [N, 1, 3, H_in, W_in]
        d_inst_in = f_in["inst"]  # [N, 1, H_in, W_in]
        d_meta_in = f_in["meta"]  # [N]

        N = int(f_in.attrs["length"])
        H_in = int(f_in.attrs["H"])
        W_in = int(f_in.attrs["W"])
        T_in = int(f_in.attrs["T"])
        assert T_in == 1, f"Expected T=1 in input file, got {T_in}"

        # --- determine tiling pattern (same as _enumerate_full_tiles) ---
        th = int(target)
        tile_overlap = int(tile_overlap)

        if H_in <= th or W_in <= th:
            # fallback: single tile; handled in main loop
            stride = None
            tile_coords: List[Tuple[int, int]] = [(0, 0)]
            T_tiles = 1
        else:
            stride = max(1, th - tile_overlap)
            tile_coords = []
            for y0 in range(0, H_in - th + 1, stride):
                for x0 in range(0, W_in - th + 1, stride):
                    tile_coords.append((int(y0), int(x0)))
            T_tiles = len(tile_coords)

    # --- (re-)create output file ---
    vlen_str = h5py.string_dtype(encoding="utf-8")
    new_file = (not os.path.exists(out_path))

    with h5py.File(out_path, "a", libver="latest") as f_out:
        if new_file:
            f_out.attrs.update(
                {
                    "version": 1,
                    "length": int(N),
                    "H": int(target),
                    "W": int(target),
                    "T": int(T_tiles),
                    "C_img": 3,
                    "C_tgt": 4,
                    "written": 0,
                    "source": "simulate_image(tiles_from_fullres)",
                    "tile_target": int(target),
                    "tile_overlap": int(tile_overlap),
                    "H_in": int(H_in),
                    "W_in": int(W_in),
                }
            )

            dargs: Dict[str, Any] = {}
            if compression:
                dargs["compression"] = compression
                if compression == "gzip" and compression_level is not None:
                    dargs["compression_opts"] = int(compression_level)

            chunk_N = max(1, min(16, N))
            chunk_T = max(1, min(T_tiles, 16))

            f_out.create_dataset(
                "imgs",
                shape=(N, T_tiles, 3, target, target),
                dtype="float32",
                chunks=(chunk_N, chunk_T, 3, min(target, 256), min(target, 256)),
                **dargs,
            )
            f_out.create_dataset(
                "tgts",
                shape=(N, T_tiles, 4, target, target),
                dtype="float32",
                chunks=(chunk_N, chunk_T, 4, min(target, 256), min(target, 256)),
                **dargs,
            )
            f_out.create_dataset(
                "inst",
                shape=(N, T_tiles, target, target),
                dtype="int32",
                chunks=(chunk_N, chunk_T, min(target, 256), min(target, 256)),
                **dargs,
            )
            # one meta JSON per scene, same style as SimCellsDataset → create_sim_cells_dataset_h5
            f_out.create_dataset(
                "meta",
                shape=(N,),
                dtype=vlen_str,
                chunks=(min(1024, N),),
            )
        else:
            # basic sanity checks
            assert int(f_out.attrs["length"]) == int(N), "length mismatch (output vs input)"
            assert int(f_out.attrs["H"]) == int(target), "H mismatch (output target)"
            assert int(f_out.attrs["W"]) == int(target), "W mismatch (output target)"
            assert int(f_out.attrs["T"]) == int(T_tiles), "T mismatch (tiles per image)"
            if "written" not in f_out.attrs:
                f_out.attrs["written"] = 0

        d_imgs_out = f_out["imgs"]
        d_tgts_out = f_out["tgts"]
        d_inst_out = f_out["inst"]
        d_meta_out = f_out["meta"]

        written = int(f_out.attrs["written"])
        if written >= N:
            print(f"[export_tiles] already complete: {written}/{N} → {out_path}")
            return N, T_tiles, target

        # --- helpers for flushing ---
        def _flush_safe():
            f_out.flush()
            try:
                f_out.id.flush()
            except Exception:
                pass
            try:
                fd = f_out.id.get_vfd_handle()
                if fd is not None:
                    os.fsync(fd)
            except Exception:
                pass

        # --- main loop over scenes ---
        pbar = tqdm(total=N, initial=written, desc=progress_desc, dynamic_ncols=True)
        since_flush = 0

        with h5py.File(in_path, "r", libver="latest") as f_in_loop:
            d_imgs_in = f_in_loop["imgs"]
            d_inst_in = f_in_loop["inst"]
            d_meta_in = f_in_loop["meta"]

            for i in range(written, N):
                if stop["flag"]:
                    break

                # load one full image, instances and meta
                img_full = d_imgs_in[i, 0]  # [3, H_in, W_in]
                img_full = np.transpose(img_full, (1, 2, 0)).astype(np.float32)  # [H_in, W_in, 3]

                inst_full = d_inst_in[i, 0].astype(np.int32)  # [H_in, W_in]

                meta_full = json.loads(d_meta_in[i])

                # safe check on shapes
                H, W = inst_full.shape
                assert H == H_in and W == W_in, "Instance map size mismatch with attrs"

                # prepare output buffers for this scene
                imgs_scene = np.zeros((T_tiles, 3, target, target), dtype=np.float32)
                tgts_scene = np.zeros((T_tiles, 4, target, target), dtype=np.float32)
                inst_scene = np.zeros((T_tiles, target, target), dtype=np.int32)
                tile_meta_list: List[Dict[str, Any]] = []

                # small-image fallback (pad+resize)
                if H_in <= th or W_in <= th:
                    # pad to square
                    img_sq, _, _ = pad_to_square(img_full, pad_val=0.0)
                    inst_sq, _, _ = pad_to_square(inst_full, pad_val=0)

                    # resize to target
                    img_t = resize_map(img_sq, th, mode="image")   # [th, th, 3]
                    inst_t = resize_map(inst_sq, th, mode="label") # [th, th]

                    tile_idx = 0

                    # recompute targets from inst_t
                    cell_t = (inst_t > 0).astype(np.float32)
                    bound_soft = make_soft_boundary_from_instances(
                        inst_t,
                        ring_width=max(1, boundary_ring_width),
                        soft_band=max(1, boundary_soft_band),
                        sigma=float(boundary_sigma),
                    ).astype(np.float32)

                    # centers from instances
                    centers: List[Tuple[int, int]] = []
                    lbl = inst_t
                    for k in range(1, int(lbl.max()) + 1):
                        ys, xs = np.where(lbl == k)
                        if ys.size == 0:
                            continue
                        cy = int(np.mean(ys))
                        cx = int(np.mean(xs))
                        centers.append((cy, cx))

                    center_stem = make_center_stem_from_centers(centers, (th, th))
                    center_heat = make_center_heatmap(center_stem, sigma=float(center_sigma))
                    energy = make_energy_from_instances(inst_t)

                    tgt_t = np.stack([cell_t, bound_soft, center_heat, energy], axis=0).astype(np.float32)

                    imgs_scene[tile_idx] = np.transpose(img_t, (2, 0, 1))
                    tgts_scene[tile_idx] = tgt_t
                    inst_scene[tile_idx] = inst_t.astype(np.int32)

                    # meta: one tile covering whole (rescaled) image
                    meta_t = crop_sim_meta_to_tile(meta_full, 0, 0, th, th)
                    mode_meta = {
                        "mode": "tiles",
                        "tile_xy": (0, 0),
                        "tile_hw": (int(th), int(th)),
                        "sim_meta": meta_t,
                        "full_meta": meta_full,
                    }
                    tile_meta_list.append(mode_meta)

                else:
                    # standard sliding tiles
                    tile_idx = 0
                    for (y0, x0) in tile_coords:
                        img_t = crop_rect(img_full, y0, x0, th, th)   # [th, th, 3]
                        inst_t = crop_rect(inst_full, y0, x0, th, th) # [th, th]

                        # recompute targets for this tile (same as SimCellsDataset)
                        cell_t = (inst_t > 0).astype(np.float32)
                        bound_soft = make_soft_boundary_from_instances(
                            inst_t,
                            ring_width=max(1, boundary_ring_width),
                            soft_band=max(1, boundary_soft_band),
                            sigma=float(boundary_sigma),
                        ).astype(np.float32)

                        # tile-level sim meta
                        meta_t = crop_sim_meta_to_tile(meta_full, y0, x0, th, th)

                        # centers from meta_t if available; else from instances
                        centers: List[Tuple[int, int]] = []
                        if (
                            "centers" in meta_t
                            and isinstance(meta_t["centers"], (list, tuple))
                            and len(meta_t["centers"]) > 0
                        ):
                            for (y, x) in meta_t["centers"]:
                                yy = int(round(y))
                                xx = int(round(x))
                                if 0 <= yy < th and 0 <= xx < th:
                                    centers.append((yy, xx))
                        else:
                            lbl = inst_t
                            for k in range(1, int(lbl.max()) + 1):
                                ys, xs = np.where(lbl == k)
                                if ys.size == 0:
                                    continue
                                cy = int(np.mean(ys))
                                cx = int(np.mean(xs))
                                centers.append((cy, cx))

                        center_stem = make_center_stem_from_centers(centers, (th, th))
                        center_heat = make_center_heatmap(center_stem, sigma=float(center_sigma))
                        energy = make_energy_from_instances(inst_t)

                        tgt_t = np.stack(
                            [cell_t, bound_soft, center_heat, energy], axis=0
                        ).astype(np.float32)

                        # fill buffers
                        imgs_scene[tile_idx] = np.transpose(img_t, (2, 0, 1)).astype(np.float32)
                        tgts_scene[tile_idx] = tgt_t
                        inst_scene[tile_idx] = inst_t.astype(np.int32)

                        mode_meta = {
                            "mode": "tiles",
                            "tile_xy": (int(y0), int(x0)),
                            "tile_hw": (int(th), int(th)),
                            "sim_meta": meta_t,
                            "full_meta": meta_full,
                        }
                        tile_meta_list.append(mode_meta)

                        tile_idx += 1

                    assert tile_idx == T_tiles, "Tile count mismatch"

                # write this scene to output
                d_imgs_out[i, :, :, :, :] = imgs_scene
                d_tgts_out[i, :, :, :, :] = tgts_scene
                d_inst_out[i, :, :, :] = inst_scene

                # meta: mirror SimCellsDataset → create_sim_cells_dataset_h5
                scene_meta = {
                    "full": meta_full,
                    "tiles": tile_meta_list,
                    "sim_kwargs": {},  # cannot reconstruct; leave empty
                }
                d_meta_out[i] = json.dumps(scene_meta, separators=(",", ":"))

                f_out.attrs.modify("written", int(i + 1))
                since_flush += 1
                pbar.update(1)

                if (since_flush % max(1, flush_every)) == 0:
                    _flush_safe()
                    since_flush = 0

        _flush_safe()
        pbar.close()

        print(
            f"[export_tiles] done: scenes={f_out.attrs['written']}/{N}, "
            f"T_tiles={T_tiles}, size={target}x{target} → {out_path}"
        )
        return N, T_tiles, target


def create_background_dataset_h5(
    root_dir: str,
    out_path: str,
    target: int = 512,
    compression: Optional[str] = "lzf",
    flush_every: int = 64,
    glob_pattern: str = "**/*.npy",
):
    """
    Export background-only tiles from .npy files into an HDF5 file
    with ONE tile per sample, i.e. T_max == 1.

    Input
    -----
    Each .npy file contains either:
      - [T, 3, 512, 512]
      - [3, 512, 512]   -> treated as one tile

    Output
    ------
      /imgs: float32 [N_total_tiles, 1, 3, target, target]
      /tgts: float32 [N_total_tiles, 1, 4, target, target]   all zeros
      /inst: int32   [N_total_tiles, 1, target, target]      all zeros
      /meta: vlen JSON [N_total_tiles]

    This is compatible with the single-tile layout used when n_tiles=1.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    npy_files = sorted(glob.glob(os.path.join(root_dir, glob_pattern), recursive=True))
    if len(npy_files) == 0:
        raise RuntimeError(f"No .npy files found under: {root_dir}")

    stop = setup_stop_flag()

    def _load_tiles(path: str) -> np.ndarray:
        arr = np.load(path, allow_pickle=False)

        if arr.ndim == 3:
            # [3,H,W] -> [1,3,H,W]
            if arr.shape[0] != 3:
                raise ValueError(f"{path}: expected [3,H,W] or [T,3,H,W], got {arr.shape}")
            arr = arr[None, ...]
        elif arr.ndim == 4:
            # [T,3,H,W]
            if arr.shape[1] != 3:
                raise ValueError(f"{path}: expected shape [T,3,H,W], got {arr.shape}")
        else:
            raise ValueError(f"{path}: expected 3D or 4D array, got {arr.shape}")

        T, C, H, W = arr.shape
        if C != 3:
            raise ValueError(f"{path}: expected 3 channels, got {C}")
        if H != target or W != target:
            raise ValueError(
                f"{path}: expected spatial size ({target},{target}), got ({H},{W})"
            )

        arr = np.asarray(arr, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{path}: contains non-finite values")

        arr = np.clip(arr, 0.0, 1.0)
        return arr

    # count total number of tiles
    total_tiles = 0
    file_tile_counts: List[int] = []

    for path in tqdm(npy_files, desc="count tiles", dynamic_ncols=True):
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 3:
            t = 1
        elif arr.ndim == 4:
            t = int(arr.shape[0])
        else:
            raise ValueError(f"{path}: expected 3D or 4D array, got {arr.shape}")
        file_tile_counts.append(t)
        total_tiles += t

    if total_tiles == 0:
        raise RuntimeError("No tiles found in npy files")

    vlen_str = h5py.string_dtype(encoding="utf-8")
    new_file = not os.path.exists(out_path)

    with h5py.File(out_path, "a", libver="latest") as f:
        if new_file:
            dargs = {}
            if compression:
                dargs["compression"] = compression

            chunk_N = max(1, min(64, total_tiles))

            f.attrs.update(
                {
                    "version": 1,
                    "length": int(total_tiles),
                    "H": int(target),
                    "W": int(target),
                    "T": 1,
                    "C_img": 3,
                    "C_tgt": 4,
                    "written": 0,
                    "source": os.path.abspath(root_dir),
                    "dataset_kind": "background_tiles_single",
                    "target": int(target),
                }
            )

            d_imgs = f.create_dataset(
                "imgs",
                shape=(total_tiles, 1, 3, target, target),
                dtype="float32",
                chunks=(chunk_N, 1, 3, min(target, 256), min(target, 256)),
                **dargs,
            )
            d_tgts = f.create_dataset(
                "tgts",
                shape=(total_tiles, 1, 4, target, target),
                dtype="float32",
                chunks=(chunk_N, 1, 4, min(target, 256), min(target, 256)),
                **dargs,
            )
            d_inst = f.create_dataset(
                "inst",
                shape=(total_tiles, 1, target, target),
                dtype="int32",
                chunks=(chunk_N, 1, min(target, 256), min(target, 256)),
                **dargs,
            )
            d_meta = f.create_dataset(
                "meta",
                shape=(total_tiles,),
                dtype=vlen_str,
                chunks=(min(1024, total_tiles),),
            )
            written = 0
        else:
            d_imgs = f["imgs"]
            d_tgts = f["tgts"]
            d_inst = f["inst"]
            d_meta = f["meta"]
            written = int(f.attrs.get("written", 0))

            # sanity
            if int(f.attrs["length"]) != int(total_tiles):
                raise ValueError(
                    f"Existing H5 length={int(f.attrs['length'])}, "
                    f"but current data has total_tiles={total_tiles}"
                )
            if int(f.attrs["T"]) != 1:
                raise ValueError(f"Expected T=1, found T={int(f.attrs['T'])}")
            if int(f.attrs["H"]) != int(target) or int(f.attrs["W"]) != int(target):
                raise ValueError("target mismatch with existing H5")

        def _make_meta(path: str, tile_index_in_file: int) -> Dict[str, Any]:
            full_meta = {
                "src_path": os.path.abspath(path),
                "kind": "background_tile",
                "target": int(target),
            }

            tile_meta = {
                "mode": "background",
                "tile_index_in_file": int(tile_index_in_file),
                "tile_xy": (0, 0),
                "tile_hw": (int(target), int(target)),
                "sim_meta": None,
                "full_meta": full_meta,
            }

            return {
                "full": full_meta,
                "tiles": [tile_meta],   # length 1 on purpose
                "sim_kwargs": None,
            }

        out_idx = written
        pbar = tqdm(total=total_tiles, initial=written, desc="export background h5", dynamic_ncols=True)
        since_flush = 0

        skipped_tiles = 0
        if written > 0:
            skipped_tiles = written

        for path, n_tiles in zip(npy_files, file_tile_counts):
            if stop["flag"]:
                break

            tiles = _load_tiles(path)  # [T,3,512,512]

            for t in range(n_tiles):
                if skipped_tiles > 0:
                    skipped_tiles -= 1
                    continue

                tile = tiles[t]  # [3,512,512]

                d_imgs[out_idx, 0] = tile.astype(np.float32)
                d_tgts[out_idx, 0] = np.zeros((4, target, target), dtype=np.float32)
                d_inst[out_idx, 0] = np.zeros((target, target), dtype=np.int32)
                d_meta[out_idx] = json.dumps(
                    jsonify(_make_meta(path, t)),
                    separators=(",", ":"),
                )

                out_idx += 1
                f.attrs.modify("written", int(out_idx))
                pbar.update(1)

                since_flush += 1
                if (since_flush % max(1, flush_every)) == 0:
                    flush_safe_file(f)
                    since_flush = 0

        flush_safe_file(f)
        pbar.close()

        print(f"[export background] done: tiles={out_idx}/{total_tiles} → {out_path}")
        return out_path
