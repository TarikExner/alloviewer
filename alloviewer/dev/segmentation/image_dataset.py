import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from skimage.segmentation import relabel_sequential
from skimage.measure import label as sklabel

from typing import Optional, List, Tuple

from .image_simulation.image_simulation import simulate_image
from .image_simulation.camera_style_application import apply_camera_style
from .utils import (
    crop_sim_meta_to_tile,
    make_energy_from_instances,
    make_center_heatmap,
    make_center_stem_from_centers,
    make_soft_boundary_from_instances,
    square_crop_from_center_radius,
    estimate_well_mask,
    crop_rect,
    pad_to_square,
    resize_map,
    heal_watershed_gaps,
    seeded_watershed_from_mask,
    load_com_labels_csv,
    crop_external_meta_to_tile,
)
from .image_simulation import (
    CameraDimension,
    SimulatorConfig,
    CameraStyleConfig,
    STYLE_PARAMS_REGISTRY,
    diverse_cameras,
    load_or_build_quantile_band_cache
)

class BaseCellsTilesDataset(Dataset):
    """
    Shared base for datasets that produce tiles and 4 heads:
        - cell       : [H,W]
        - boundary   : [H,W] (soft boundary)
        - center     : [H,W] (center heatmap)
        - energy     : [H,W]

    Subclasses must:
      - implement __len__
      - in __getitem__, build a list of tiles and call _finalize_tiles
        tiles = [
          {
            "img":      float32 [H,W,3] in [0,1],
            "inst":     int32   [H,W],
            "cell":     float32 [H,W] or None (None → derived from inst>0),
            "centers":  list[(y,x)] in tile coords or None,
            "mode_meta": dict (tile-specific meta, kept in extras["meta"]["tiles"]),
            "sim_meta":  optional dict with per-tile meta (not used here, but stored),
          },
          ...
        ]
    """

    def __init__(
        self,
        target: int = 512,
        tile_overlap: int = 64,
        boundary_ring_width: int = 1,
        boundary_soft_band: int = 2,
        boundary_sigma: float = 1.0,
        center_sigma: float = 1.0,
        transforms=None,
    ):
        super().__init__()
        self.target = int(target)
        self.tile_overlap = int(tile_overlap)
        self.boundary_ring_width = int(boundary_ring_width)
        self.boundary_soft_band = int(boundary_soft_band)
        self.boundary_sigma = float(boundary_sigma)
        self.center_sigma = float(center_sigma)
        self.transforms = transforms

    # ---- shared helpers ----

    def _enumerate_full_tiles(self, H: int, W: int):
        """
        Sliding window tiling with overlap; returns [(y0, x0), ...]
        covering the full image with tiles of size target×target.
        If the image is smaller than the target, returns a single (0,0) tile.
        """
        th = self.target
        if H <= th or W <= th:
            return [(0, 0)]

        stride = max(1, th - self.tile_overlap)

        ys = list(range(0, max(1, H - th + 1), stride))
        if ys[-1] + th < H:
            ys.append(H - th)

        xs = list(range(0, max(1, W - th + 1), stride))
        if xs[-1] + th < W:
            xs.append(W - th)

        return [(int(y0), int(x0)) for y0 in ys for x0 in xs]

    @staticmethod
    def _compute_centers_from_instances(inst: np.ndarray):
        centers = []
        max_id = int(inst.max())
        for k in range(1, max_id + 1):
            ys, xs = np.where(inst == k)
            if ys.size == 0:
                continue
            cy = int(np.mean(ys))
            cx = int(np.mean(xs))
            centers.append((cy, cx))
        return centers

    def _make_heads(
        self,
        inst_t: np.ndarray,
        cell_t: np.ndarray | None = None,
        centers: list[tuple[int, int]] | None = None,
    ):
        """
        Build the 4 heads for one tile.
        inst_t : int32 [H,W]
        cell_t : optional float32 [H,W], if None → (inst_t>0)
        centers: optional list of (y,x) in tile coords, if None → from inst_t
        """
        inst_t = inst_t.astype(np.int32)
        H, W = inst_t.shape

        if cell_t is None:
            cell_t = (inst_t > 0).astype(np.float32)
        else:
            cell_t = cell_t.astype(np.float32)

        bound_soft = make_soft_boundary_from_instances(
            inst_t,
            ring_width=max(1, self.boundary_ring_width),
            soft_band=max(1, self.boundary_soft_band),
            sigma=self.boundary_sigma,
        ).astype(np.float32)

        if centers is None or len(centers) == 0:
            centers = self._compute_centers_from_instances(inst_t)

        center_stem = make_center_stem_from_centers(centers, (H, W))
        center_heat = make_center_heatmap(center_stem, sigma=self.center_sigma)
        energy = make_energy_from_instances(inst_t)

        return cell_t, bound_soft, center_heat, energy

    def _finalize_tiles(self, tiles, full_meta, sim_kwargs):
        """
        tiles: list of dicts as described in class docstring.
        Returns:
          imgs_t : float32 [T,3,H,W]
          tgts_t : float32 [T,4,H,W]
          extras : dict with instance labels + meta

        Important:
          - heads are built from LOCAL relabeled instances
          - stored /inst keeps GLOBAL instance ids so tiles remain stitchable
        """
        imgs_out = []
        tgts_out = []
        inst_out = []
        tiles_meta_out = []

        for t in tiles:
            img_t = t["img"]  # [H,W,3] float32

            # keep global ids for output/stitching
            inst_global = t["inst"].astype(np.int32)

            # make a local compact copy only for head generation
            inst_local, _, _ = relabel_sequential(inst_global)

            cell_t_init = t.get("cell", None)
            centers = t.get("centers", None)

            # heads before transforms, based on LOCAL labels
            cell_t, bound_soft, center_heat, energy = self._make_heads(
                inst_local, cell_t_init, centers
            )

            # optional transforms
            if self.transforms is not None:
                out = self.transforms(
                    image=img_t,
                    masks=[cell_t, bound_soft, center_heat, energy],
                )
                img_t = out["image"]
                cell_t, bound_soft, center_heat, energy = out["masks"]

            tgt_t = np.stack(
                [
                    cell_t.astype(np.float32),
                    bound_soft.astype(np.float32),
                    center_heat.astype(np.float32),
                    energy.astype(np.float32),
                ],
                axis=0,
            )

            img_chw = np.transpose(img_t.astype(np.float32), (2, 0, 1))

            imgs_out.append(img_chw)
            tgts_out.append(tgt_t)

            # store GLOBAL ids, not relabeled local ids
            inst_out.append(inst_global.astype(np.int32))

            tiles_meta_out.append(t["mode_meta"])

        imgs_t = torch.from_numpy(np.stack(imgs_out, axis=0).astype(np.float32))
        tgts_t = torch.from_numpy(np.stack(tgts_out, axis=0).astype(np.float32))
        inst_t = torch.from_numpy(np.stack(inst_out, axis=0).astype(np.int32))

        extras = {
            "instance_labels": inst_t,
            "meta": {
                "full": full_meta,
                "tiles": tiles_meta_out,
                "sim_kwargs": sim_kwargs,
            },
        }
        return imgs_t, tgts_t, extras

class SimCellsDataset(BaseCellsTilesDataset):
    """
    Modes:
      - pad_resize: pad to square, then resize to target
      - crop_well_resize: square crop around well, then resize
      - tiles: random target×target tiles
               (n_tiles>0 → that many random tiles;
                n_tiles==-1 → full sliding coverage)
      - fullres: keep original resolution (no crop, no resize), one tile per scene

    Returns:
      img_t   : float32 [T, 3, S, S] in [0,1]  (S = target, except in fullres)
      tgt_t   : float32 [T, 4, S, S]
      extras  : dict with:
                  - instance_labels: int32 [T, S, S]
                  - meta: {
                        "full": original_sim_meta,
                        "tiles": [tile_meta_0, ..., tile_meta_{T-1}],
                        "sim_kwargs": simulator kwargs (without "seed")
                    }
    """

    def __init__(
        self,
        length: int = 1000,
        mode: str = "pad_resize",     # "pad_resize" | "crop_well_resize" | "tiles" | "fullres"
        target: int = 512,
        n_tiles: int = 1,             # >0: pick that many random tiles; -1: cover full image; only for mode="tiles"
        tile_overlap: int = 64,       # used when n_tiles == -1
        boundary_ring_width: int = 1,
        boundary_soft_band: int = 2,
        boundary_sigma: float = 1.0,
        center_sigma: float = 1.0,
        rng_seed: int = 123,
        well_is_brighter: str = "auto",
        transforms=None,              # optional Albumentations-style joint transforms

        # required
        scene_cfg: Optional[SimulatorConfig]=None,
        camera_cfg: Optional[CameraDimension]=None,
        camera_style_cfg: Optional[CameraStyleConfig]=None,
    ):
        assert scene_cfg is not None, "scene_cfg (SimulatorConfig) is required"
        assert camera_cfg is not None, "camera_cfg (CameraSetup) is required"

        if camera_style_cfg is None:
            # self.camera_style_cfg = simulated_raw_style()
            self.camera_style_cfg = diverse_cameras()
        else:
            self.camera_style_cfg = camera_style_cfg


        self.quantile_band_cache = load_or_build_quantile_band_cache(
            folders=None,
            force_recompute=False,
        )

        self.camera_style_registry = STYLE_PARAMS_REGISTRY

        super().__init__(
            target=target,
            tile_overlap=tile_overlap,
            boundary_ring_width=boundary_ring_width,
            boundary_soft_band=boundary_soft_band,
            boundary_sigma=boundary_sigma,
            center_sigma=center_sigma,
            transforms=transforms,
        )

        self.length = int(length)
        self.mode = str(mode)
        assert self.mode in ("pad_resize", "crop_well_resize", "tiles", "fullres")

        self.n_tiles = int(n_tiles)
        self.well_is_brighter = well_is_brighter
        self.base_rng = np.random.default_rng(rng_seed)
        self.scene_cfg = scene_cfg
        self.camera_cfg = camera_cfg

    def __len__(self):
        # number of simulated scenes, not tiles
        return self.length

    # ---- mode helpers that produce single tiles (img, cell, inst, mode_meta, centers) ----

    def _mode_pad_resize(self, img, cell, inst, sim_meta):
        """
        Pad image/masks to square and resize to self.target.
        Centers are mapped from full-res coordinates through pad+resize.
        """
        img_sq, (pad_top, pad_left), S = pad_to_square(img, pad_val=0.0)
        cell_sq, _, _ = pad_to_square(cell, pad_val=0.0)
        inst_sq, _, _ = pad_to_square(inst, pad_val=0)

        img_o = resize_map(img_sq, self.target, "image")
        cell_o = resize_map(cell_sq, self.target, "binary")
        inst_o = resize_map(inst_sq, self.target, "label")

        scale = self.target / float(S)
        centers = []
        if "centers" in sim_meta and isinstance(sim_meta["centers"], (list, tuple)):
            for (y, x) in sim_meta["centers"]:
                yy = int(round((y + pad_top) * scale))
                xx = int(round((x + pad_left) * scale))
                if 0 <= yy < self.target and 0 <= xx < self.target:
                    centers.append((yy, xx))

        mode_meta = dict(
            mode="pad_resize",
            pad_top=int(pad_top),
            pad_left=int(pad_left),
            S_in=int(S),
            scale=float(scale),
        )

        return img_o, cell_o, inst_o, centers, mode_meta

    def _mode_crop_well_resize(self, img, cell, inst, sim_meta):
        """
        Crop a square region around the well (from sim_meta if possible),
        then resize to target. Centers are mapped through crop+resize.
        """
        cycx = sim_meta.get("well_center", None)
        R = sim_meta.get("radius_px", None)
        if (cycx is None) or (R is None):
            _, center, radius = estimate_well_mask(
                img, blur_sigma=3.0, well_is_brighter=self.well_is_brighter
            )
        else:
            center = (float(cycx[0]), float(cycx[1]))
            radius = float(R)

        y0, y1, x0, x1 = square_crop_from_center_radius(cell.shape, center, radius, pad=8)
        h = y1 - y0
        w = x1 - x0

        img_c = crop_rect(img, y0, x0, h, w)
        cell_c = crop_rect(cell, y0, x0, h, w)
        inst_c = crop_rect(inst, y0, x0, h, w)

        img_o = resize_map(img_c, self.target, "image")
        cell_o = resize_map(cell_c, self.target, "binary")
        inst_o = resize_map(inst_c, self.target, "label")

        scale = self.target / max(1.0, float(max(h, w)))
        centers = []
        if "centers" in sim_meta and isinstance(sim_meta["centers"], (list, tuple)):
            for (y, x) in sim_meta["centers"]:
                yy = int(round((y - y0) * scale))
                xx = int(round((x - x0) * scale))
                if 0 <= yy < self.target and 0 <= xx < self.target:
                    centers.append((yy, xx))

        mode_meta = dict(
            mode="crop_well_resize",
            crop=(int(y0), int(y1), int(x0), int(x1)),
            scale=float(scale),
            well_center=(float(center[0]), float(center[1])),
            well_radius=float(radius),
        )

        return img_o, cell_o, inst_o, centers, mode_meta

    def _mode_fullres(self, img, cell, inst, sim_meta):
        """
        Keep full resolution, no crop/resize.
        Centers are taken from sim_meta in full-res coordinates.
        """
        img_o = img.astype(np.float32)
        cell_o = cell.astype(np.float32)
        inst_o = inst.astype(np.int32)

        H, W = cell_o.shape
        centers = []
        if "centers" in sim_meta and isinstance(sim_meta["centers"], (list, tuple)):
            for (y, x) in sim_meta["centers"]:
                yy = int(round(y))
                xx = int(round(x))
                if 0 <= yy < H and 0 <= xx < W:
                    centers.append((yy, xx))

        mode_meta = dict(
            mode="fullres",
            height=int(H),
            width=int(W),
            scale=1.0,
        )
        return img_o, cell_o, inst_o, centers, mode_meta

    def _mode_tiles_full_coverage(self, img, cell, inst, sim_meta):
        """
        mode == "tiles", n_tiles == -1
        Sliding tiles with overlap; centers taken from cropped sim_meta per tile.
        """
        H, W = cell.shape
        th = self.target

        tiles = []
        if H <= th or W <= th:
            # fallback: same as pad_resize
            img_o, cell_o, inst_o, centers, mode_meta = self._mode_pad_resize(
                img, cell, inst, sim_meta
            )
            tiles.append(
                dict(
                    img=img_o,
                    inst=inst_o,
                    cell=cell_o,
                    centers=centers,
                    mode_meta=mode_meta,
                    sim_meta=sim_meta,
                )
            )
            return tiles

        for (y0, x0) in self._enumerate_full_tiles(H, W):
            img_t = crop_rect(img, y0, x0, th, th)
            cell_t = crop_rect(cell, y0, x0, th, th)
            inst_t = crop_rect(inst, y0, x0, th, th)

            tile_sim_meta = crop_sim_meta_to_tile(sim_meta, y0, x0, th, th)
            centers = tile_sim_meta.get("centers", [])

            mode_meta = {
                "mode": "tiles",
                "tile_xy": (int(y0), int(x0)),
                "tile_hw": (int(th), int(th)),
                "sim_meta": tile_sim_meta,
                "full_meta": sim_meta,
            }

            tiles.append(
                dict(
                    img=img_t.astype(np.float32),
                    inst=inst_t.astype(np.int32),
                    cell=cell_t.astype(np.float32),
                    centers=centers,
                    mode_meta=mode_meta,
                    sim_meta=tile_sim_meta,
                )
            )

        return tiles

    def _mode_tiles_random(self, img, cell, inst, sim_meta, rng):
        """
        mode == "tiles", n_tiles > 0
        Pick n_tiles random tiles; fallback to pad_resize if image too small.
        """
        H, W = cell.shape
        th = self.target
        tiles = []

        for _ in range(self.n_tiles):
            if H <= th or W <= th:
                # same fallback as before
                img_o, cell_o, inst_o, centers, mode_meta = self._mode_pad_resize(
                    img, cell, inst, sim_meta
                )
                mode_meta = {
                    **mode_meta,
                    "sim_meta": sim_meta,
                    "full_meta": sim_meta,
                }
                tiles.append(
                    dict(
                        img=img_o,
                        inst=inst_o,
                        cell=cell_o,
                        centers=centers,
                        mode_meta=mode_meta,
                        sim_meta=sim_meta,
                    )
                )
                continue

            y0 = int(rng.integers(0, H - th + 1))
            x0 = int(rng.integers(0, W - th + 1))

            img_t = crop_rect(img, y0, x0, th, th)
            cell_t = crop_rect(cell, y0, x0, th, th)
            inst_t = crop_rect(inst, y0, x0, th, th)

            tile_sim_meta = crop_sim_meta_to_tile(sim_meta, y0, x0, th, th)
            centers = tile_sim_meta.get("centers", [])

            mode_meta = {
                "mode": "tiles",
                "tile_xy": (int(y0), int(x0)),
                "tile_hw": (int(th), int(th)),
                "sim_meta": tile_sim_meta,
                "full_meta": sim_meta,
            }

            tiles.append(
                dict(
                    img=img_t.astype(np.float32),
                    inst=inst_t.astype(np.int32),
                    cell=cell_t.astype(np.float32),
                    centers=centers,
                    mode_meta=mode_meta,
                    sim_meta=tile_sim_meta,
                )
            )

        return tiles

    def __getitem__(self, idx):
        rng = np.random.default_rng(
            int(self.base_rng.integers(0, 2**31 - 1)) ^ int(idx)
        )

        # build sim kwargs from configs
        sim_kwargs = self.scene_cfg.sample_kwargs(rng, camera=self.camera_cfg)
        sim_kwargs.setdefault("return_targets", True)

        # simulate
        img, meta, targets = simulate_image(**sim_kwargs)
        cell = targets["cell_mask"].astype(np.float32)
        inst = targets["instance_labels"].astype(np.int32)
        img = apply_camera_style(
            img,
            rng,
            self.camera_style_cfg,
            self.camera_style_registry,
            quantile_band_cache=self.quantile_band_cache,
            cell_mask=cell
        )

        tiles = []
        full_meta = meta

        if self.mode == "pad_resize":
            img_o, cell_o, inst_o, centers, mode_meta = self._mode_pad_resize(
                img, cell, inst, meta
            )
            tiles.append(
                dict(
                    img=img_o,
                    inst=inst_o,
                    cell=cell_o,
                    centers=centers,
                    mode_meta=mode_meta,
                    sim_meta=meta,
                )
            )

        elif self.mode == "crop_well_resize":
            img_o, cell_o, inst_o, centers, mode_meta = self._mode_crop_well_resize(
                img, cell, inst, meta
            )
            tiles.append(
                dict(
                    img=img_o,
                    inst=inst_o,
                    cell=cell_o,
                    centers=centers,
                    mode_meta=mode_meta,
                    sim_meta=meta,
                )
            )

        elif self.mode == "fullres":
            img_o, cell_o, inst_o, centers, mode_meta = self._mode_fullres(
                img, cell, inst, meta
            )
            tiles.append(
                dict(
                    img=img_o,
                    inst=inst_o,
                    cell=cell_o,
                    centers=centers,
                    mode_meta=mode_meta,
                    sim_meta=meta,
                )
            )

        else:  # mode == "tiles"
            if self.n_tiles == -1:
                tiles = self._mode_tiles_full_coverage(img, cell, inst, meta)
            elif self.n_tiles > 0:
                tiles = self._mode_tiles_random(img, cell, inst, meta, rng)
            else:
                raise ValueError("Choose n_tiles to be a positive integer or -1")

        # finalize: heads, transforms, stack
        return self._finalize_tiles(
            tiles=tiles,
            full_meta=full_meta,
            sim_kwargs={k: v for k, v in sim_kwargs.items() if k != "seed"},
        )


class ExternalCellsTilesDataset(BaseCellsTilesDataset):
    """
    External images + 8-bit masks → same output format as
    SimCellsDataset(mode="tiles", n_tiles=-1).

    Per-image COM+labels CSV ({image}_data.csv) is used if present and
    passed into tile meta (cropped to each tile).
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
        super().__init__(
            target=target,
            tile_overlap=tile_overlap,
            boundary_ring_width=boundary_ring_width,
            boundary_soft_band=boundary_soft_band,
            boundary_sigma=boundary_sigma,
            center_sigma=center_sigma,
            transforms=transforms,
        )

        self.root_dir = root_dir
        self.heal_radius = int(heal_radius)

        self.pairs = self.find_pairs_strict(root_dir)
        if not self.pairs:
            raise RuntimeError("no (img, img_mask) pairs found in external folder")

    def __len__(self):
        return len(self.pairs)

    # ---- IO helpers ----
    def find_pairs_strict(self, root_dir: str) -> List[Tuple[str, str]]:
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

            # skip masks themselves if they got indexed
            if stem.lower().endswith("_mask"):
                continue

            img_dir = os.path.dirname(img_path)                  # .../<folder>
            masks_dir = img_dir + "_masks"                       # .../<folder>_masks

            m1 = os.path.join(masks_dir, f"{stem}_mask.tif")
            m2 = os.path.join(masks_dir, f"{stem}_mask.tiff")
            mask_path = m1 if os.path.exists(m1) else (m2 if os.path.exists(m2) else None)

            if mask_path is not None:
                pairs.append((os.path.abspath(img_path), os.path.abspath(mask_path)))

        pairs.sort()
        return pairs

    def _detect_bitdepth_u16(self, a: np.ndarray, p: float = 99.9):
        # a: uint16 HxW or HxWxC
        vmax = int(a.max())
        if vmax == 0:
            return 16, (1 << 16) - 1, False  # bit_depth, white, shifted

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

    def _read_image_and_mask(self, img_path: str, mask_path: str):
        """
        Returns:
          img_rgb_f32 : float32 [H, W, 3] in [0,1]
          msk_bin     : uint8   [H, W] with {0,1}
          info        : dict with bit-depth details for auditing
        """
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
            b, white, shifted = self._detect_bitdepth_u16(img)
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

        if m.max() <= 1:
            msk_bin = (m >= 1).astype(np.uint8)
        else:
            msk_bin = (m >= 255).astype(np.uint8)

        return img_f32, msk_bin, info

    # ---- main __getitem__ ----
    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img, msk, read_info = self._read_image_and_mask(img_path, mask_path)
        H, W = msk.shape

        # start from the raw mask
        mask_bin = (msk > 0).astype(np.uint8)

        # seeds from the *unhealed* separated mask
        seeds = sklabel(mask_bin, connectivity=1).astype(np.int32)
        seeds, _, _ = relabel_sequential(seeds)

        # heal only the *binary* mask (fills 1px cracks)
        mask_healed = heal_watershed_gaps(mask_bin, radius=self.heal_radius)

        # re-grow labels into healed mask without merging instances
        inst_full = seeded_watershed_from_mask(mask_healed, seeds)

        # try to load COM+labels CSV
        d_img, iname = os.path.split(img_path)
        folder = os.path.basename(d_img)
        input_folder = os.path.dirname(d_img)
        results_csv_path = os.path.join(input_folder, "results.csv")
        centers_csv = load_com_labels_csv(iname, folder, results_csv_path)

        # centers: prefer CSV; fallback to instance centroids
        if centers_csv:
            centers_full = centers_csv
        else:
            centers_full = self._compute_centers_from_instances(inst_full)

        full_meta = {
            "src_path": img_path,
            "mask_path": mask_path,
            "data_csv": results_csv_path if os.path.exists(results_csv_path) else "",
            "H_in": int(H),
            "W_in": int(W),
            "n_cells": int(inst_full.max()),
            "centers": [(int(y), int(x)) for (y, x) in centers_full],
            "read_info": read_info,
        }

        # enumerate tiles
        th = self.target
        tiles = []

        if H <= th or W <= th:
            # fallback: pad+resize whole image to target
            img_sq, _, _ = pad_to_square(img, pad_val=0.0)
            inst_sq, _, _ = pad_to_square(inst_full, pad_val=0)

            img_t = resize_map(img_sq, th, mode="image")   # [th,th,3]
            inst_t = resize_map(inst_sq, th, mode="label") # [th,th]

            tile_sim_meta = crop_external_meta_to_tile(
                full_meta,
                0, 0,
                th, th,
            )

            centers_tile = tile_sim_meta.get("centers", [])

            mode_meta = {
                "mode": "tiles",
                "tile_xy": (0, 0),
                "tile_hw": (int(th), int(th)),
                "sim_meta": tile_sim_meta,
                "full_meta": full_meta,
            }

            tiles.append(
                dict(
                    img=img_t.astype(np.float32),
                    inst=inst_t.astype(np.int32),
                    cell=None,  # derive from inst>0
                    centers=centers_tile,
                    mode_meta=mode_meta,
                    sim_meta=tile_sim_meta,
                )
            )

        else:
            for (y0, x0) in self._enumerate_full_tiles(H, W):
                img_t = crop_rect(img, y0, x0, th, th)         # [th,th,3]
                inst_t = crop_rect(inst_full, y0, x0, th, th)  # [th,th] with GLOBAL ids

                tile_sim_meta = crop_external_meta_to_tile(
                    full_meta,
                    y0, x0,
                    th, th,
                )

                centers_tile = tile_sim_meta.get("centers", [])

                mode_meta = {
                    "mode": "tiles",
                    "tile_xy": (int(y0), int(x0)),
                    "tile_hw": (int(th), int(th)),
                    "sim_meta": tile_sim_meta,
                    "full_meta": full_meta,
                }

                tiles.append(
                    dict(
                        img=img_t.astype(np.float32),
                        inst=inst_t.astype(np.int32),   # keep GLOBAL ids
                        cell=None,
                        centers=centers_tile,
                        mode_meta=mode_meta,
                        sim_meta=tile_sim_meta,
                    )
                )

        # same finalize path as simulated data
        return self._finalize_tiles(
            tiles=tiles,
            full_meta=full_meta,
            sim_kwargs=None,  # external data has no sim kwargs
        )

