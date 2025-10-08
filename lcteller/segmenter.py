import os
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, List, Literal
from scipy import ndimage as ndi
from collections import deque
import torch

from skimage import morphology, segmentation, measure, feature
from skimage.measure import label as cc_label

from .contracts import ISegmenter
from .config import DEVICE
from .segmentation import (
    build_unet_cpu_small,
    build_unet_cpu_medium,
    build_unet_cpu_large,
)

MaskProvider = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

@dataclass
class SegmenterConfig:
    """
    Minimal config for the segmenter.

    Attributes
    ----------
    unet_mode: which backbone to build ("small" | "medium" | "large")
    model_dir: folder that holds the checkpoint files
    model_file: filename of the checkpoint; if None -> f"best_{unet_mode}.pth"
    device: e.g. "cuda:0" or "cpu"
    thr_cell: threshold for cell probability to make a binary mask
    thr_bound: threshold for boundary probability to make a binary mask
    use_amp: enable autocast on CUDA
    return_logits: include raw logits in the output dict (False by default)
    compute_instances: run connected components on the final cell mask (if skimage is installed)
    """
    unet_mode: Literal["small", "medium", "large"] = "small"
    model_dir: str = "./models"
    model_file: Optional[str] = None
    device: str = DEVICE
    thr_cell: float = 0.5
    thr_bound: float = 0.5
    use_amp: bool = True
    return_logits: bool = False
    compute_instances: bool = True

@dataclass
class InstanceSegmenterConfig:
    # cleanup
    min_object_area: int = 100
    min_hole_area:   int = 20
    min_instance_area: int = 120

    # elevation terms
    distance_smooth_sigma: float = 0.0   # ↓ from 1.0 (keep peaks!)
    use_boundary: bool = True
    gamma: float = 3.0                   # stronger boundary push
    smooth_boundary_sigma: float = 0.0

    # extra edge term from cell prob gradient (helps at the neck)
    use_edge_term: bool = True
    edge_sigma: float = 1.0
    edge_weight: float = 1.5            # adds to elevation

    # markers
    seed_method: Literal["hmax","spacing"] = "hmax"
    h_maxima: float = 0.6               # lower -> more seeds
    min_peak_distance: int = 6
    marker_erosion_radius: int = 1      # erode ONLY for seed finding

    # watershed options
    compactness: float = 0.0
    watershed_line: bool = False


class SegmenterUNet(ISegmenter):
    """

    Input
    -----
    img: np.ndarray with shape (H, W, 3) or (H, W).
         Values can be in [0, 1] or [0, 255]. dtype will be cast to float32.

    Output dict
    -----------
    {
        "cell_mask": np.uint8 [H, W]               # 1 = cell, 0 = background
        "boundary":  np.uint8 [H, W]               # 1 = boundary, 0 = non-boundary
        "probs": {
            "cell":  np.float32 [H, W],            # sigmoid probs
            "bound": np.float32 [H, W]
        }
        "instance_labels": np.int32 [H, W] | None  # optional, if skimage present
        "meta": {
            "unet_mode": str,
            "device": str,
            "checkpoint": str
        }
        # (optionally) "logits": np.float32 [2, H, W]
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

        self.device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" in cfg.device else "cpu")
        self.model = builder(in_channels=3, out_channels=2).to(self.device)
        self.model.eval()

        # resolve checkpoint path
        model_file = cfg.model_file or f"unet_{cfg.unet_mode}.pth"
        self.ckpt_path = os.path.join(cfg.model_dir, model_file)
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        # load weights (state_dict only)
        state = torch.load(self.ckpt_path, map_location="cpu")
        # some users save with DDP wrapper; handle both
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            # try to strip a possible "module." prefix from DDP
            fixed = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(fixed, strict=True)

        # AMP policy
        self.use_amp = bool(cfg.use_amp and (self.device.type == DEVICE))

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        img: (H,W,3) or (H,W). Returns float tensor [1,3,H,W] on self.device.
        Normalizes to [0,1].
        """
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)  # gray -> RGB
        if img.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {img.shape}")

        x = img.astype(np.float32, copy=False)
        if x.max() > 1.0:
            x = x / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = np.ascontiguousarray(x, dtype=np.float32)
        t = torch.from_numpy(x).unsqueeze(0).to(self.device, non_blocking=True)  # [1,3,H,W]
        return t

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        x = self._to_tensor(img)
        use_bf16 = (self.use_amp and torch.cuda.is_bf16_supported())

        with torch.amp.autocast(
            device_type=DEVICE,
            dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            enabled=self.use_amp
        ):
            logits = self.model(x)           # [1,2,H,W]
            probs = torch.sigmoid(logits)    # [1,2,H,W]

        # to numpy
        probs_np = probs.squeeze(0).detach().to(torch.float32).cpu().numpy()  # [2,H,W]
        cell_p = probs_np[0]
        bound_p = probs_np[1]

        cell_bin = (cell_p >= self.cfg.thr_cell).astype(np.uint8)
        bound_bin = (bound_p >= self.cfg.thr_bound).astype(np.uint8)


        # Optional instance labeling
        instances = None
        if self.cfg.compute_instances:
            if cc_label is not None:
                # avoid marking boundaries as part of cells
                instances = cc_label(cell_bin, connectivity=2).astype(np.int32)
            else:
                # skimage not installed
                instances = None

        out: Dict[str, Any] = {
            "cell_mask": cell_bin,
            "boundary": bound_bin,
            "probs": {"cell": cell_p.astype(np.float32), "bound": bound_p.astype(np.float32)},
            "instance_labels": instances,
            "meta": {
                "unet_mode": self.cfg.unet_mode,
                "device": str(self.device),
                "checkpoint": os.path.abspath(self.ckpt_path),
            },
        }
        if self.cfg.return_logits:
            out["logits"] = logits.squeeze(0).detach().cpu().float().numpy()
        return out

    @classmethod
    def from_config(cls, cfg_dict: Dict[str, Any]) -> "SegmenterUNet":
        """
        Helper to build from a plain dict (e.g., loaded from your YAML/JSON config).
        Unknown keys are ignored.
        """
        known = {f.name for f in SegmenterConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(cfg_dict).items() if k in known}
        cfg = SegmenterConfig(**filtered)
        return cls(cfg)



class InstanceSegmenter:
    def __init__(self, cfg: InstanceSegmenterConfig):
        self.cfg = cfg

    def __call__(self, seg_out: dict, update_cell_mask: bool = True) -> dict:
        cell = (seg_out["cell_mask"] > 0)

        # remove tiny bits first
        if self.cfg.min_object_area > 0:
            cell = morphology.remove_small_objects(cell, min_size=int(self.cfg.min_object_area))

        # fill holes
        cell = ndi.binary_fill_holes(cell)
        if self.cfg.min_hole_area > 0:
            cell = morphology.remove_small_holes(cell, area_threshold=int(self.cfg.min_hole_area))

        # distance (keep sharp)
        dist = ndi.distance_transform_edt(cell).astype(np.float32)
        if self.cfg.distance_smooth_sigma > 0:
            dist_s = ndi.gaussian_filter(dist, float(self.cfg.distance_smooth_sigma))
        else:
            dist_s = dist

        # base elevation: LOW is labeled first
        elevation = -dist_s

        # add boundary prob (higher cost at boundary)
        if self.cfg.use_boundary and "probs" in seg_out and "bound" in seg_out["probs"]:
            b = seg_out["probs"]["bound"].astype(np.float32)
            if self.cfg.smooth_boundary_sigma > 0:
                b = ndi.gaussian_filter(b, float(self.cfg.smooth_boundary_sigma))
            if b.max() > b.min():
                b = (b - b.min()) / (b.max() - b.min())
                elevation = elevation + self.cfg.gamma * b

        # add edge term from cell prob gradient (push cut through necks)
        if self.cfg.use_edge_term and "probs" in seg_out and "cell" in seg_out["probs"]:
            p = seg_out["probs"]["cell"].astype(np.float32)
            g = ndi.gaussian_gradient_magnitude(p, sigma=float(self.cfg.edge_sigma))
            if g.max() > 0:
                g = g / g.max()
                elevation = elevation + self.cfg.edge_weight * g

        # --- robust markers: erode ONLY for seeds to split dumbbells ---
        seed_mask = cell
        if self.cfg.marker_erosion_radius > 0:
            seed_mask = morphology.binary_erosion(
                seed_mask, morphology.disk(int(self.cfg.marker_erosion_radius))
            )

        if self.cfg.seed_method == "hmax" and self.cfg.h_maxima > 0:
            seeds_bool = morphology.h_maxima(dist_s, h=float(self.cfg.h_maxima))
            seeds_bool &= seed_mask
        else:
            coords = feature.peak_local_max(
                dist_s, min_distance=int(self.cfg.min_peak_distance),
                labels=seed_mask, exclude_border=False
            )
            seeds_bool = np.zeros_like(cell, dtype=bool)
            if coords.size:
                seeds_bool[tuple(coords.T)] = True

        markers = measure.label(seeds_bool, connectivity=1).astype(np.int32)
        if markers.max() == 0:
            # fallback: CC of eroded mask
            markers = measure.label(seed_mask, connectivity=1).astype(np.int32)

        # watershed inside cell mask
        instances = segmentation.watershed(
            image=elevation,
            markers=markers,
            mask=cell,
            compactness=float(self.cfg.compactness),
            watershed_line=bool(self.cfg.watershed_line),
        ).astype(np.int32)

        # drop tiny shards and relabel compactly
        if self.cfg.min_instance_area > 0:
            instances = morphology.remove_small_objects(instances, min_size=int(self.cfg.min_instance_area)).astype(np.int32)
            instances = measure.label(instances > 0, connectivity=1).astype(np.int32)

        if update_cell_mask:
            seg_out["cell_mask"] = cell.astype(np.uint8)
        seg_out["instance_labels"] = instances
        return seg_out

    @classmethod
    def from_config(cls, cfg_dict: Dict[str, Any]) -> "InstanceSegmenter":
        known = {f.name for f in InstanceSegmenterConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(cfg_dict or {}).items() if k in known}
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

