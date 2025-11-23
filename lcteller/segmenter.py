import os
import copy
from dataclasses import dataclass, field, asdict
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, List, TypeVar, Type
from scipy import ndimage as ndi
from collections import deque
import torch

from skimage import morphology, segmentation, measure, feature

from .contracts import ISegmenter
from .segmentation import (
    build_unet_cpu_small,
    build_unet_cpu_medium,
    build_unet_cpu_large,
    UNET_MEAN,
    UNET_STD
)

MaskProvider = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
Segmenter = TypeVar("Segmenter", bound="SegmenterUNet")

@dataclass
class SegmenterConfig:
    # model
    unet_mode: str = "small"
    model_dir: str = "./models"
    model_file: Optional[str] = None
    device: str = "cuda"
    use_amp: bool = True
    return_logits: bool = False

    # post
    compute_instances: bool = True

    # convenience thresholds (not used for watershed internals)
    thr_cell: float = 0.5
    thr_bound: float = 0.5

    # tell the segmenter that the input to __call__ will be a batch of tiles
    # instead of a single image
    input_is_tiles: bool = False

    # nested instance config (dict or InstanceSegmenterConfig)
    instance_cfg: Dict[str, Any] = field(default_factory=dict)

    # normalize data
    normalize: bool = True

    # tiling parameters
    tile_size: int = 512
    tile_overlap: int = 64
    tile_pad_value: float = 0.0

    # ---- helpers ----
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config; nests instance_cfg as a dict."""
        d = asdict(self)
        icfg = d.get("instance_cfg", {})
        if isinstance(icfg, InstanceSegmenterConfig):
            d["instance_cfg"] = icfg.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "SegmenterConfig":
        """Construct from dict; accepts nested instance_cfg as dict or InstanceSegmenterConfig."""
        d = dict(d or {})
        raw_icfg = d.pop("instance_cfg", None)

        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in allowed}

        if isinstance(raw_icfg, InstanceSegmenterConfig):
            filtered["instance_cfg"] = raw_icfg.to_dict()
        elif isinstance(raw_icfg, dict):
            filtered["instance_cfg"] = InstanceSegmenterConfig.from_dict(raw_icfg).to_dict()
        elif raw_icfg is None:
            filtered["instance_cfg"] = {}
        else:
            filtered["instance_cfg"] = {}

        return cls(**filtered)


@dataclass
class InstanceSegmenterConfig:
    # -------- Probability → binary mask (hysteresis) --------
    cell_mask_low_thr: float = 0.10     # low threshold for hysteresis
    cell_mask_high_thr: float = 0.60    # high threshold for hysteresis
    mask_close_radius: int = 2          # morphological closing radius (px)

    # Cleanup after hysteresis
    min_hole_area: int = 10              # remove small holes (px^2)
    min_object_area: int = 10            # remove small objects (px^2)

    # -------- Distance transform --------
    distance_smooth_sigma: float = 1.0  # smooth EDT (px)
    distance_weight: float = 1.0        # how much -EDT contributes to lowering elevation

    # -------- Boundary term (adds to elevation) --------
    use_boundary: bool = True
    smooth_boundary_sigma: float = 1.0  # blur P(bound) before using
    gamma_boundary: float = 1.0         # weight for boundary cost

    # -------- Edge (grad of P(cell)) term (adds to elevation) --------
    use_edge_term: bool = True
    edge_sigma: float = 1.0             # gradient smoothing
    edge_weight: float = 1.0            # weight for edge cost

    # -------- Energy term (subtracts from elevation) --------
    use_energy: bool = True
    energy_weight: float = 0.5          # attraction inside cells
    energy_smooth_sigma: float = 1.0    # optional smoothing of P(energy)

    # center-driven seeds
    use_centers: bool = True
    center_seed_method: str = "nms"     # {"nms","thr"}
    center_min_distance: int = 1        # NMS min distance (px)
    center_thr: float = 0.2             # threshold for centers (abs prob)

    # -------- Watershed --------
    compactness: float = 0.0
    watershed_line: bool = False        # draw seams as 0-valued pixels in labels

    # -------- Post-processing --------
    min_instance_area: int = 3          # remove tiny instances (px^2)

    # -------- Helpers --------
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "InstanceSegmenterConfig":
        """Construct from dict, ignoring unknown keys."""
        d = dict(d or {})
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)


class SegmenterUNet:
    """
    Wraps the UNet (4 heads) and returns probabilities plus (optional) instances.

    If cfg.input_is_tiles == False (default):
        __call__(img: np.ndarray (H,W,3)) -> dict for ONE image

    If cfg.input_is_tiles == True OR you pass a 4D array:
        __call__(tiles: np.ndarray (T,3,H,W) or (T,H,W,3)) -> dict of BATched probs/masks

    Output dict for single image:
    {
        "cell_mask": np.uint8 [H, W]
        "boundary":  np.uint8 [H, W]
        "probs": {
            "cell":   float32 [H, W],
            "bound":  float32 [H, W],
            "center": float32 [H, W],
            "energy": float32 [H, W],
        }
        "instance_labels": np.int32 [H, W] | None
        "meta": {...}
        ("logits": float32 [4, H, W]) if return_logits
    }

    Output dict for tiles (T tiles):
    {
        "cell_mask": np.uint8 [T, H, W]
        "boundary":  np.uint8 [T, H, W]
        "probs": {
            "cell":   float32 [T, H, W],
            "bound":  float32 [T, H, W],
            "center": float32 [T, H, W],
            "energy": float32 [T, H, W],
        }
        "instance_labels": List[np.ndarray] | None   # only if compute_instances=True
        "meta": {..., "batched": True, "n_tiles": T}
        ("logits": float32 [T, 4, H, W]) if return_logits
    }
    """
    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg

        # pick builder
        if cfg.unet_mode == "small":
            builder = build_unet_cpu_small
        elif cfg.unet_mode == "medium":
            builder = build_unet_cpu_medium
        elif cfg.unet_mode == "large":
            builder = build_unet_cpu_large
        else:
            raise ValueError(f"Unknown unet_mode: {cfg.unet_mode}")

        # device
        if torch.cuda.is_available() and "cuda" in cfg.device:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cpu")

        # 4 output channels: cell, boundary, center, energy
        self.model = builder(in_channels=3, out_channels=4).to(self.device)
        self.model.eval()


        # resolve checkpoint
        model_file = cfg.model_file or f"best_{cfg.unet_mode}.pth"
        self.ckpt_path = os.path.join(cfg.model_dir, model_file)
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        # load weights (handle plain state_dict or ddp-wrapped dict)
        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            fixed = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(fixed, strict=True)

        # AMP policy
        self.use_amp = bool(cfg.use_amp and (self.device.type == "cuda"))

        # optional instance segmenter
        self.inst_seg = InstanceSegmenter(
            InstanceSegmenterConfig(**(cfg.instance_cfg or {}))
        ) if cfg.compute_instances else None

    def _to_chw_numpy(self, img: np.ndarray) -> np.ndarray:
        """
        Convert various input shapes to [3, H, W] float32 in [0,1].
        Accepts:
          - (H, W, 3)
          - (3, H, W)
          - (1, 3, H, W)
          - (H, W)  -> broadcast to 3ch
        """

        # Fast path: already CHW float32, typically from `load_image(..., as_chw=True, scale=True)`
        if (
            img.ndim == 3
            and img.shape[0] == 3
            and img.dtype == np.float32
        ):
            # assume in [0,1]; just ensure contiguous
            return np.ascontiguousarray(img)

        # (1,3,H,W) -> (3,H,W)
        if img.ndim == 4 and img.shape[0] == 1 and img.shape[1] == 3:
            img = img[0]

        # (3,H,W) -> (H,W,3)
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
            # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        if img.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels last, got shape {img.shape}")

        x = img.astype(np.float32, copy=False)
        if x.max() > 1.0:
            x = x / 255.0

        x = np.transpose(x, (2, 0, 1))  # [3, H, W]
        x = np.ascontiguousarray(x, dtype=np.float32)
        return x

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Accepts:
          - (H, W, 3)
          - (3, H, W)
          - (1, 3, H, W)
          - (H, W)  -> broadcast to 3ch
        Returns: [1, 3, H, W] on device in [0,1] (then normalized if cfg.normalize)
        """
        x = self._to_chw_numpy(img)  # [3, H, W] float32 in [0,1]
        t = torch.from_numpy(x).unsqueeze(0).to(self.device, non_blocking=True)
        if self.cfg.normalize:
            t = self._normalize(t)
        return t

    def _to_tensor_tiles(self, tiles: np.ndarray) -> torch.Tensor:
        """
        tiles: (T,3,H,W) or (T,H,W,3) -> torch [T,3,H,W] on device
        """
        if tiles.ndim != 4:
            raise ValueError(f"Expected 4D tiles array, got shape {tiles.shape}")

        if tiles.shape[1] == 3:  # (T,3,H,W)
            x = tiles.astype(np.float32, copy=False)
        elif tiles.shape[-1] == 3:  # (T,H,W,3)
            x = np.transpose(tiles, (0, 3, 1, 2)).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Expected tiles with 3 channels, got shape {tiles.shape}")

        if x.max() > 1.0:
            x = x / 255.0

        x = np.ascontiguousarray(x, dtype=np.float32)
        t = torch.from_numpy(x).to(self.device, non_blocking=True)  # [T,3,H,W]
        if self.cfg.normalize:
            t = self._normalize(t)
        return t

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] on self.device
        """
        mean = torch.as_tensor(UNET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std  = torch.as_tensor(UNET_STD,  dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.inference_mode()
    def predict_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        if tiles.device != self.device:
            tiles = tiles.to(self.device, non_blocking=True)
        use_bf16 = (self.use_amp and torch.cuda.is_bf16_supported())

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            enabled=self.use_amp
        ):
            logits = self.model(tiles)          # [B, 4, H, W]
            probs = torch.sigmoid(logits)

        return probs.detach().to(torch.float32).cpu()

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        """
        If cfg.input_is_tiles == False (default):
            behave like before → single image
        If cfg.input_is_tiles == True OR img.ndim == 4:
            treat input as batch of tiles
        """
        # --- case 1: tiles ---
        if self.cfg.input_is_tiles or img.ndim == 4:
            tiles_t = self._to_tensor_tiles(img)           # [T,3,H,W] on device
            use_bf16 = (self.use_amp and torch.cuda.is_bf16_supported())
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=(torch.bfloat16 if use_bf16 else torch.float16),
                enabled=self.use_amp
            ):
                logits = self.model(tiles_t)               # [T,4,H,W]
                probs = torch.sigmoid(logits)              # [T,4,H,W]

            probs_np = probs.detach().to(torch.float32).cpu().numpy()  # [T,4,H,W]
            T, C, H, W = probs_np.shape
            cell_p  = probs_np[:, 0, ...]
            bound_p = probs_np[:, 1, ...]
            center_p= probs_np[:, 2, ...]
            energy_p= probs_np[:, 3, ...]

            out: Dict[str, Any] = {
                "probs": {
                    "cell":   cell_p.astype(np.float32),
                    "bound":  bound_p.astype(np.float32),
                    "center": center_p.astype(np.float32),
                    "energy": energy_p.astype(np.float32),
                },
                "cell_mask": (cell_p >= self.cfg.thr_cell).astype(np.uint8),
                "boundary":  (bound_p >= self.cfg.thr_bound).astype(np.uint8),
                "instance_labels": None,
                "meta": {
                    "unet_mode": self.cfg.unet_mode,
                    "device": str(self.device),
                    "checkpoint": os.path.abspath(self.ckpt_path),
                    "batched": True,
                    "n_tiles": int(T),
                },
            }

            if self.cfg.return_logits:
                out["logits"] = logits.detach().cpu().float().numpy()  # [T,4,H,W]

            # optional per-tile instance segmentation
            if self.inst_seg is not None:
                inst_list = []
                # apply instance segmenter tile-by-tile to keep old interface
                for i in range(T):
                    single_out = {
                        "probs": {
                            "cell":   cell_p[i],
                            "bound":  bound_p[i],
                            "center": center_p[i],
                            "energy": energy_p[i],
                        },
                        "cell_mask": (cell_p[i] >= self.cfg.thr_cell).astype(np.uint8),
                        "boundary":  (bound_p[i] >= self.cfg.thr_bound).astype(np.uint8),
                        "instance_labels": None,
                        "meta": {},
                    }
                    single_out = self.inst_seg(single_out, update_cell_mask=True)
                    inst_list.append(single_out["instance_labels"])
                out["instance_labels"] = inst_list

            return out

        x = self._to_tensor(img)
        use_bf16 = (self.use_amp and torch.cuda.is_bf16_supported())

        with torch.amp.autocast(
            device_type=self.cfg.device,
            dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            enabled=self.use_amp
        ):
            logits = self.model(x)           # [1,4,H,W]
            probs  = torch.sigmoid(logits)   # [1,4,H,W]

        probs_np = probs.squeeze(0).detach().to(torch.float32).cpu().numpy()  # [4,H,W]
        cell_p, bound_p, center_p, energy_p = probs_np

        out: Dict[str, Any] = {
            "probs": {
                "cell":   cell_p.astype(np.float32),
                "bound":  bound_p.astype(np.float32),
                "center": center_p.astype(np.float32),
                "energy": energy_p.astype(np.float32),
            },
            "instance_labels": None,
            "meta": {
                "unet_mode": self.cfg.unet_mode,
                "device": str(self.device),
                "checkpoint": os.path.abspath(self.ckpt_path),
            },
        }

        out["cell_mask"] = (cell_p >= self.cfg.thr_cell).astype(np.uint8)
        out["boundary"]  = (bound_p >= self.cfg.thr_bound).astype(np.uint8)

        if self.cfg.return_logits:
            out["logits"] = logits.squeeze(0).detach().cpu().float().numpy()

        if self.inst_seg is not None:
            out = self.inst_seg(out, update_cell_mask=True)

        return out

    @classmethod
    def from_config(cls: Type[Segmenter], cfg_dict: Dict[str, Any]) -> Segmenter:
        known = {f.name for f in SegmenterConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(cfg_dict or {}).items() if k in known}
        cfg = SegmenterConfig(**filtered)
        return cls(cfg)


class SegmenterUNetInference(SegmenterUNet):
    """
    Inference wrapper:

    - Takes full images.
    - Tiles into [T, 3, tile_size, tile_size] (cfg.tile_size / cfg.tile_overlap).
    - Runs UNet on tiles (batch).
    - Stitches probs (and optional logits) back to [4, H, W].
    - Always computes instance segmentation on the stitched result.
    """

    def __init__(self, cfg: SegmenterConfig):
        cfg = copy.deepcopy(cfg)

        cfg.compute_instances = True
        cfg.input_is_tiles = False

        super().__init__(cfg)

        self.tile_size = int(self.cfg.tile_size)
        self.overlap = int(self.cfg.tile_overlap)
        self.pad_value = float(self.cfg.tile_pad_value)

        if self.inst_seg is None:
            self.inst_seg = InstanceSegmenter(
                InstanceSegmenterConfig(**(self.cfg.instance_cfg or {}))
            )

    def _iter_sliding_windows(self, H: int, W: int):
        tile = self.tile_size
        overlap = self.overlap
        stride = tile - overlap
        assert stride > 0, "overlap must be smaller than tile_size"

        ys = list(range(0, max(1, H - tile + 1), stride))
        if ys[-1] + tile < H:
            ys.append(H - tile)

        xs = list(range(0, max(1, W - tile + 1), stride))
        if xs[-1] + tile < W:
            xs.append(W - tile)

        for y0 in ys:
            y1 = y0 + tile
            for x0 in xs:
                x1 = x0 + tile
                yield (y0, y1, x0, x1)

    def _tile_image_numpy(self, image_chw: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        image_chw: [3, H, W] -> tiles [N, 3, tile_size, tile_size], (H, W)
        """
        assert image_chw.ndim == 3 and image_chw.shape[0] == 3, "image must be [3, H, W]"
        _, H, W = image_chw.shape

        tile = self.tile_size
        overlap = self.overlap
        stride = tile - overlap
        assert stride > 0, "overlap must be smaller than tile_size"

        # same logic as _iter_sliding_windows, but we keep windows in a list
        ys = list(range(0, max(1, H - tile + 1), stride))
        if ys[-1] + tile < H:
            ys.append(H - tile)

        xs = list(range(0, max(1, W - tile + 1), stride))
        if xs[-1] + tile < W:
            xs.append(W - tile)

        N = len(ys) * len(xs)
        tiles_arr = np.empty((N, 3, tile, tile), dtype=image_chw.dtype)

        idx = 0
        for y0 in ys:
            y1 = y0 + tile
            for x0 in xs:
                x1 = x0 + tile

                crop = image_chw[:, y0:y1, x0:x1]   # [3, th, tw]
                th, tw = crop.shape[1], crop.shape[2]

                # fill actual content
                tiles_arr[idx, :, :th, :tw] = crop

                # pad bottom/right if needed
                if th < tile:
                    tiles_arr[idx, :, th:tile, :tw] = self.pad_value
                if tw < tile:
                    tiles_arr[idx, :, :th, tw:tile] = self.pad_value
                if th < tile and tw < tile:
                    tiles_arr[idx, :, th:tile, tw:tile] = self.pad_value

                idx += 1

        if N == 0:
            raise RuntimeError(
                "No tiles produced. Check tile_size / overlap / image size."
            )

        return tiles_arr, (H, W)

    def _reconstruct_from_tiles_probability(
        self,
        tiles: np.ndarray,
        orig_hw: Tuple[int, int],
    ) -> np.ndarray:
        """
        tiles: [N, C, tile_size, tile_size] -> [C, H, W] by averaging overlaps
        """
        assert tiles.ndim == 4, "tiles must be [N, C, tile_size, tile_size]"
        N, C, tH, tW = tiles.shape
        assert tH == self.tile_size and tW == self.tile_size, "tile_size mismatch"

        H, W = orig_hw

        def _iter_tiles():
            idx = 0
            for (y0, y1, x0, x1) in self._iter_sliding_windows(H, W):
                if idx >= N:
                    raise RuntimeError(
                        "Not enough tiles for given H/W/tile_size/overlap."
                    )
                th = min(self.tile_size, H - y0)
                tw = min(self.tile_size, W - x0)
                patch = tiles[idx, :, :th, :tw]  # [C, th, tw]
                yield idx, y0, x0, th, tw, patch
                idx += 1
            if idx != N:
                raise RuntimeError(
                    f"Number of tiles ({N}) does not match tiling scheme ({idx})."
                )

        acc = np.zeros((C, H, W), dtype=np.float32)
        acc_w = np.zeros((1, H, W), dtype=np.float32)

        for idx, y0, x0, th, tw, patch in _iter_tiles():
            acc[:, y0:y0 + th, x0:x0 + tw] += patch
            acc_w[:, y0:y0 + th, x0:x0 + tw] += 1.0

        acc_w[acc_w == 0] = 1.0
        out = acc / acc_w
        return out.astype(np.float32)  # [C, H, W]

    # ---- main call ----

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Full-image inference with internal tiling.

        If you pass tiles directly (4D array), we just fall back to the
        base class behavior and do not tile again.
        """
        # If user passes tiles explicitly, keep old interface
        if img.ndim == 4:
            return super().__call__(img)

        # 1) preprocess to CHW numpy using base helper
        img_chw = self._to_chw_numpy(img)          # [3, H, W], float32 in [0,1]
        tiles_np, orig_hw = self._tile_image_numpy(img_chw)  # [T, 3, tile, tile], (H, W)

        # 2) convert tiles to torch with same path as everywhere else
        tiles_t = self._to_tensor_tiles(tiles_np)  # [T, 3, tile, tile] on device

        # 3) forward pass on tiles using predict_tiles (no logits)
        probs_t = self.predict_tiles(tiles_t)      # [T, 4, tile, tile] on CPU (float32)
        probs_np_tiles = probs_t.numpy()           # view, no copy

        # 4) stitch probability maps to full size
        probs_full = self._reconstruct_from_tiles_probability(
            probs_np_tiles, orig_hw
        )  # [4, H, W], float32
        cell_p, bound_p, center_p, energy_p = probs_full

        # Build minimal seg_out for InstanceSegmenter (no copies here)
        seg_out: Dict[str, Any] = {
            "probs": {
                "cell":   cell_p,
                "bound":  bound_p,
                "center": center_p,
                "energy": energy_p,
            },
            "instance_labels": None,
            "meta": {
                "unet_mode": self.cfg.unet_mode,
                "device": str(self.device),
                "checkpoint": os.path.abspath(self.ckpt_path),
                "tile_size": int(self.tile_size),
                "overlap": int(self.overlap),
            },
        }

        # 5) instance segmentation updates seg_out in place
        if self.inst_seg is not None:
            seg_out = self.inst_seg(seg_out, update_cell_mask=False)

        instances = seg_out["instance_labels"]

        # 6) Final result: only keep what is really needed outside
        result: Dict[str, Any] = {
            "instance_labels": instances,   # np.int32 [H, W]
            "probs": {                     # only those needed for QC
                "cell":  cell_p,
                "bound": bound_p,
            },
            "meta": seg_out["meta"],
        }
        return result

class InstanceSegmenter:
    def __init__(self, cfg: InstanceSegmenterConfig):
        self.cfg = cfg
        r = int(self.cfg.mask_close_radius)
        self._mask_close_selem = morphology.disk(r) if r > 0 else None

    def _hysteresis_mask(self, pc: np.ndarray) -> np.ndarray:
        """Two-threshold hysteresis on cell prob to get a robust watershed mask."""
        low = float(self.cfg.cell_mask_low_thr)
        high = float(self.cfg.cell_mask_high_thr)

        strong = (pc >= high)
        weak   = (pc >= low)

        # label weak components and keep those connected to any strong pixel
        lab = measure.label(weak, connectivity=1)

        if strong.any():
            # labels that touch any strong pixel
            strong_ids = np.unique(lab[strong])
            # drop background label
            strong_ids = strong_ids[strong_ids != 0]

            if strong_ids.size > 0:
                # build label -> keep lookup table
                max_lab = lab.max()
                lut = np.zeros(max_lab + 1, dtype=bool)
                lut[strong_ids] = True

                # vectorized "keep |= (lab == sid)" for all sids
                mask = lut[lab]
            else:
                mask = np.zeros_like(weak, dtype=bool)
        else:
            mask = np.zeros_like(weak, dtype=bool)

        # small closing & hole handling
        if self._mask_close_selem is not None:
            mask = morphology.binary_closing(mask, self._mask_close_selem)

        mask = ndi.binary_fill_holes(mask)

        if self.cfg.min_hole_area > 0:
            mask = morphology.remove_small_holes(
                mask,
                area_threshold=int(self.cfg.min_hole_area)
            )

        if self.cfg.min_object_area > 0:
            mask = morphology.remove_small_objects(
                mask,
                min_size=int(self.cfg.min_object_area)
            )

        return mask.astype(bool, copy=False)

    def _smooth01(self, x: np.ndarray, sigma: float) -> np.ndarray:
        if sigma and sigma > 0:
            y = ndi.gaussian_filter(x, float(sigma))
            y = np.clip(y, 0.0, 1.0)
            return y
        return x

    def _make_markers(self, mask: np.ndarray, p_center: np.ndarray | None, dist_s: np.ndarray) -> np.ndarray:
        # ensure boolean working mask
        work_mask = mask if mask.dtype == bool else mask.astype(bool, copy=False)

        seeds_bool = np.zeros_like(work_mask, dtype=bool)

        # ---- center-driven seeds ----
        if self.cfg.use_centers and (p_center is not None):
            if getattr(self.cfg, "center_seed_method", "nms") == "nms":
                coords = feature.peak_local_max(
                    p_center,  # already float32
                    min_distance=int(self.cfg.center_min_distance),
                    threshold_abs=float(self.cfg.center_thr),
                    labels=work_mask,
                    exclude_border=False,
                )
                if coords.size:
                    seeds_bool[tuple(coords.T)] = True
            else:
                # simple threshold
                seeds_bool |= (p_center >= float(self.cfg.center_thr)) & work_mask

        # Label seeds; fallback to CCs of mask if empty
        markers = measure.label(seeds_bool, connectivity=1).astype(np.int32)
        if markers.max() == 0:
            markers = measure.label(work_mask, connectivity=1).astype(np.int32)

        return markers

    def __call__(self, seg_out: Dict[str, Any], update_cell_mask: bool = True) -> Dict[str, Any]:
        probs = seg_out.get("probs", {})
        p_cell   = probs.get("cell",   None)
        p_bound  = probs.get("bound",  None)
        p_center = probs.get("center", None)
        p_energy = probs.get("energy", None)


        def as_f32(x):
            if x is None:
                return None
            if x.dtype == np.float32 and x.flags['C_CONTIGUOUS']:
                return x
            return np.ascontiguousarray(x, dtype=np.float32)

        p_cell = as_f32(p_cell)
        p_bound = as_f32(p_bound)
        p_center = as_f32(p_center)
        p_energy = as_f32(p_energy)

        if p_cell is None:
            raise ValueError("seg_out['probs']['cell'] required")

        H, W = p_cell.shape

        # --- 1) watershed mask from cell probability (hysteresis) ---
        mask = self._hysteresis_mask(p_cell)

        # --- 2) base distance inside the (binary) mask (still keeps prob influence) ---
        dist = ndi.distance_transform_edt(mask).astype(np.float32)

        if self.cfg.distance_smooth_sigma > 0:
            ndi.gaussian_filter(
                dist,
                float(self.cfg.distance_smooth_sigma),
                output=dist,
            )
        dist_s = dist

        # --- 3) elevation (minimize) built from PROBABILITIES ---
        # start with zero elevation, then add/subtract weighted terms
        elevation = np.zeros((H, W), dtype=np.float32)
        tmp = np.empty_like(elevation, dtype=np.float32)  # scratch buffer

        # subtract distance (attract to centers => lower elevation)
        if self.cfg.distance_weight != 0:
            # normalize to [0,1] for stability
            dmax = dist_s.max()
            if dmax > 1e-6:
                # elevation -= self.cfg.distance_weight * (dist_s / dmax)
                np.divide(dist_s, dmax, out=tmp)
                elevation -= self.cfg.distance_weight * tmp

        # add boundary (higher elevation where boundary prob is high)
        if self.cfg.use_boundary and (p_bound is not None):
            b = self._smooth01(p_bound, self.cfg.smooth_boundary_sigma)
            elevation += self.cfg.gamma_boundary * b

        # add edge term from cell prob gradient magnitude
        if self.cfg.use_edge_term and (self.cfg.edge_weight != 0):
            g = ndi.gaussian_gradient_magnitude(p_cell, sigma=float(self.cfg.edge_sigma))
            gmax = g.max()
            if gmax > 1e-6:
                # elevation += self.cfg.edge_weight * (g / gmax)
                np.divide(g, gmax, out=g)
                elevation += self.cfg.edge_weight * g

        # subtract energy (attract to high energy inside cells)
        if self.cfg.use_energy and (p_energy is not None) and (self.cfg.energy_weight != 0):
            e = self._smooth01(p_energy, self.cfg.energy_smooth_sigma)
            elevation -= self.cfg.energy_weight * e
        # --- 4) seeds from centers and/or distance peaks (probability-driven) ---
        # (For centers we use center prob directly; for distance we use dist_s)
        markers = self._make_markers(mask, p_center, dist_s)

        # --- 5) watershed inside mask ---
        instances = segmentation.watershed(
            image=elevation,
            markers=markers,
            mask=mask,
            compactness=float(self.cfg.compactness),
            watershed_line=bool(self.cfg.watershed_line),
        ).astype(np.int32)

        # --- 6) post-process: drop tiny shards and relabel compactly ---
        if self.cfg.min_instance_area > 0:
            # this removes small pixel objects. This is mainly to keep the count consistent
            instances = morphology.remove_small_objects(instances, min_size=int(self.cfg.min_instance_area)).astype(np.int32)

        # push updated fields back
        if update_cell_mask:
            seg_out["cell_mask"] = mask.astype(np.uint8)
        seg_out["instance_labels"] = instances
        # seg_out["elevation"] = elevation
        # seg_out["markers"] = markers
        return seg_out

    @classmethod
    def from_config(cls, cfg_dict: Dict[str, Any]) -> "InstanceSegmenter":
        # Accept a plain dict (ignore unknown keys)
        cfg_dict = dict(cfg_dict or {})
        known = set(InstanceSegmenterConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in cfg_dict.items() if k in known}
        cfg = InstanceSegmenterConfig(**filtered)
        return cls(cfg)


class DummyByOrderSegmenter(ISegmenter):
    """
    Returns the next (cell_mask, bound_mask [, instances]) each time it's called.
    If instances are given, it uses them; else it builds them from cell/boundary.
    """
    def __init__(
        self,
        cell_masks: List[np.ndarray],
        bound_masks: List[np.ndarray],
        instance_labels: Optional[List[np.ndarray]] = None,
        min_size: int = 16,
    ):
        self.cells = deque([m.astype(bool) for m in cell_masks])
        self.bounds = deque([m.astype(bool) for m in bound_masks])
        self.insts = deque([m.astype(np.int32) for m in instance_labels]) if instance_labels is not None else None
        self.min_size = int(min_size)

    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        assert img.ndim == 3 and img.shape[2] == 3, "img must be HxWx3"
        if not self.cells or not self.bounds:
            raise RuntimeError("Ran out of masks in DummyByOrderSegmenter.")

        cell = ndi.binary_fill_holes(self.cells.popleft())
        cell = _remove_small(cell, self.min_size)
        bound = self.bounds.popleft()

        if self.insts is not None and len(self.insts) > 0:
            inst = self.insts.popleft().astype(np.int32, copy=False)
        else:
            # seeds = interior (cell minus boundary)
            seeds = cell & (~bound)
            markers, _ = ndi.label(seeds)
            inst = _grow_labels(markers, mask=cell, boundary=bound)

        return {
            "instances": inst.astype(np.int32),
            "cell_mask": cell.astype(np.uint8),
            "bound_mask": bound.astype(np.uint8),
            "probs": None,
            "qc": {"n_rois": int(inst.max())}
        }


class DummySegmenter(ISegmenter):
    """
    Dummy segmenter that uses masks from the simulator.
    Keeps ISegmenter: __call__(img) -> dict.
    You can provide masks via:
      - set_targets(cell_mask, bound_mask), or
      - set_targets_from_dict(targets), or
      - mask_provider(img) -> (cell_mask, bound_mask) at call time.
    """

    def __init__(self, min_size: int = 16, mask_provider: Optional[MaskProvider] = None):
        self.min_size = int(min_size)
        self._cell: Optional[np.ndarray] = None
        self._bound: Optional[np.ndarray] = None
        self._provider = mask_provider

    def set_targets(self, cell_mask: np.ndarray, bound_mask: np.ndarray) -> None:
        self._cell = cell_mask.astype(bool, copy=False)
        self._bound = bound_mask.astype(bool, copy=False)

    def set_targets_from_dict(self, targets: Dict[str, np.ndarray]) -> None:
        self.set_targets(targets["cell_mask"], targets["boundary"])

    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        assert img.ndim == 3 and img.shape[2] == 3, "img must be HxWx3 float"
        H, W, _ = img.shape

        # fetch masks
        if self._provider is not None:
            cell_mask, bound_mask = self._provider(img)
        else:
            if self._cell is None or self._bound is None:
                raise RuntimeError("No masks set. Call set_targets(...) or provide mask_provider in ctor.")
            cell_mask, bound_mask = self._cell, self._bound

        assert cell_mask.shape == (H, W) and bound_mask.shape == (H, W), "mask shapes must match image"

        # clean + instance grow (inside cell, not across boundary)
        cell = ndi.binary_fill_holes(cell_mask.astype(bool))
        cell = _remove_small(cell, self.min_size)

        seeds = cell & (~bound_mask.astype(bool))
        markers, _ = ndi.label(seeds)
        inst = _grow_labels(markers, mask=cell, boundary=bound_mask.astype(bool))

        return {
            "instances": inst.astype(np.int32),
            "cell_mask": cell.astype(np.uint8),
            "bound_mask": bound_mask.astype(np.uint8),
            "probs": None,  # dummy has no probabilities
            "qc": {"n_rois": int(inst.max())},
        }


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.where(sizes >= min_size)[0]] = True
    keep[0] = False
    return keep[lbl]

def _grow_labels(markers: np.ndarray, mask: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """
    Simple flood fill that grows labels from markers within `mask`
    and does not cross `boundary`. Good enough for the dummy.
    """
    inst = markers.copy()
    H, W = mask.shape

    # pixels we may still assign
    free = mask & (~boundary) & (inst == 0)

    # init queue with current labeled pixels
    from collections import deque
    q = deque(map(tuple, np.argwhere(inst > 0)))

    # 4-neighborhood
    nb = [(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        y, x = q.popleft()
        k = inst[y, x]
        for dy, dx in nb:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx]:
                inst[ny, nx] = k
                free[ny, nx] = False
                q.append((ny, nx))

    return inst

