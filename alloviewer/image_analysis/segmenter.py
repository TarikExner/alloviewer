import os
import copy
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Any, Optional, Tuple, TypeVar, Type

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import morphology, segmentation

from ..dev.segmentation import (
    build_unet_cpu_small,
    build_unet_cpu_medium,
    build_unet_cpu_large,
)

from .utils import (
    as_contiguous_f32,
    build_unet_by_mode,
    hysteresis_mask,
    iter_sliding_windows,
    make_markers,
    normalize_tensor,
    reconstruct_from_tiles_probability,
    smooth01,
    tile_image_numpy,
    to_chw_numpy,
    to_tensor_tiles,
)

Segmenter = TypeVar("Segmenter", bound="SegmenterUNet")


@dataclass
class InstanceSegmenterConfig:
    """Configuration for watershed-based instance segmentation.

    Attributes
    ----------
    cell_mask_low_thr : float
        Low threshold used for hysteresis on the cell probability map.
    cell_mask_high_thr : float
        High threshold used for hysteresis on the cell probability map.
    mask_close_radius : int
        Radius of the disk used for binary closing of the cell mask.
    min_hole_area : int
        Minimum hole area retained in the binary cell mask.
    min_object_area : int
        Minimum object area retained in the binary cell mask.
    distance_smooth_sigma : float
        Gaussian smoothing sigma applied to the distance transform.
    distance_weight : float
        Weight of the distance-transform term in the watershed elevation.
    use_boundary : bool
        Whether to add the boundary probability term to the elevation.
    smooth_boundary_sigma : float
        Gaussian smoothing sigma applied to the boundary probability map.
    gamma_boundary : float
        Weight of the boundary term in the watershed elevation.
    use_edge_term : bool
        Whether to add a cell-probability gradient term to the elevation.
    edge_sigma : float
        Sigma used for the cell-probability gradient magnitude.
    edge_weight : float
        Weight of the edge term in the watershed elevation.
    use_energy : bool
        Whether to subtract the energy probability term from the elevation.
    energy_weight : float
        Weight of the energy term in the watershed elevation.
    energy_smooth_sigma : float
        Gaussian smoothing sigma applied to the energy probability map.
    use_centers : bool
        Whether to use center probabilities for watershed seeds.
    center_seed_method : str
        Seed creation method. Supported values are ``"nms"`` and ``"thr"``.
    center_min_distance : int
        Minimum distance between NMS center seeds.
    center_thr : float
        Absolute center probability threshold.
    compactness : float
        Compactness parameter passed to watershed.
    watershed_line : bool
        Whether watershed should draw 0-valued seam pixels between regions.
    min_instance_area : int
        Minimum area retained for final instances.
    """

    cell_mask_low_thr: float = 0.10
    cell_mask_high_thr: float = 0.60
    mask_close_radius: int = 2

    min_hole_area: int = 10
    min_object_area: int = 10

    distance_smooth_sigma: float = 1.0
    distance_weight: float = 1.0

    use_boundary: bool = True
    smooth_boundary_sigma: float = 1.0
    gamma_boundary: float = 1.0

    use_edge_term: bool = True
    edge_sigma: float = 1.0
    edge_weight: float = 1.0

    use_energy: bool = True
    energy_weight: float = 0.5
    energy_smooth_sigma: float = 1.0

    use_centers: bool = True
    center_seed_method: str = "nms"
    center_min_distance: int = 1
    center_thr: float = 0.2

    compactness: float = 0.0
    watershed_line: bool = False

    min_instance_area: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration.

        Returns
        -------
        dict
            Configuration values as a plain dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        d: Optional[Dict[str, Any]] = None,
    ) -> "InstanceSegmenterConfig":
        """Create an instance segmentation configuration from a dictionary.

        Parameters
        ----------
        d : dict or None, optional
            Input dictionary. Unknown keys are ignored.

        Returns
        -------
        InstanceSegmenterConfig
            Parsed configuration object.
        """
        d = dict(d or {})
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)


@dataclass
class SegmenterConfig:
    """Configuration for UNet segmentation.

    Attributes
    ----------
    unet_mode : str
        UNet size. Supported values are ``"small"``, ``"medium"``, and
        ``"large"``.
    model_dir : str
        Directory containing model checkpoints.
    model_file : str or None
        Checkpoint filename. If ``None``, ``best_{unet_mode}.pth`` is used.
    device : str
        Requested torch device.
    use_amp : bool
        Whether to use automatic mixed precision on CUDA.
    return_logits : bool
        Whether to include raw logits in the output.
    compute_instances : bool
        Whether to compute instance labels from predicted probability maps.
    thr_cell : float
        Threshold for the binary cell mask.
    thr_bound : float
        Threshold for the binary boundary mask.
    input_is_tiles : bool
        Whether input passed to ``__call__`` is already a tile batch.
    instance_cfg : dict
        Configuration passed to :class:`InstanceSegmenter`.
    normalize : bool
        Whether to normalize input tensors with UNet training statistics.
    tile_size : int
        Tile size used by :class:`SegmenterUNetInference`.
    tile_overlap : int
        Overlap between neighboring tiles.
    tile_pad_value : float
        Value used to pad tiles at image borders.
    """

    unet_mode: str = "small"
    model_dir: str = "./models"
    model_file: Optional[str] = None
    device: str = "cuda"
    use_amp: bool = True
    return_logits: bool = False

    compute_instances: bool = True

    thr_cell: float = 0.1
    thr_bound: float = 0.1

    input_is_tiles: bool = False
    instance_cfg: Dict[str, Any] = field(default_factory=dict)

    normalize: bool = True

    tile_size: int = 512
    tile_overlap: int = 64
    tile_pad_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration.

        Returns
        -------
        dict
            Configuration values as a plain dictionary.
        """
        d = asdict(self)
        icfg = d.get("instance_cfg", {})

        if isinstance(icfg, InstanceSegmenterConfig):
            d["instance_cfg"] = icfg.to_dict()

        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "SegmenterConfig":
        """Create a segmenter configuration from a dictionary.

        Parameters
        ----------
        d : dict or None, optional
            Input dictionary. Unknown top-level keys are ignored. Nested
            ``instance_cfg`` values may be dictionaries or
            :class:`InstanceSegmenterConfig` objects.

        Returns
        -------
        SegmenterConfig
            Parsed configuration object.
        """
        d = dict(d or {})
        raw_icfg = d.pop("instance_cfg", None)

        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in allowed}

        if isinstance(raw_icfg, InstanceSegmenterConfig):
            filtered["instance_cfg"] = raw_icfg.to_dict()
        elif isinstance(raw_icfg, dict):
            filtered["instance_cfg"] = InstanceSegmenterConfig.from_dict(raw_icfg).to_dict()
        else:
            filtered["instance_cfg"] = {}

        return cls(**filtered)


class SegmenterUNet:
    """Run UNet segmentation on a single image or a batch of tiles.

    The model predicts four probability maps: cell, boundary, center, and
    energy. Instance labels are optionally computed from these maps.

    Parameters
    ----------
    cfg : SegmenterConfig
        Segmenter configuration.

    Notes
    -----
    For a single image, accepted inputs include ``(H, W, 3)``, ``(3, H, W)``,
    ``(1, 3, H, W)``, and ``(H, W)``. For tile batches, accepted inputs are
    ``(T, 3, H, W)`` and ``(T, H, W, 3)``.
    """

    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg

        builder = build_unet_by_mode(
            cfg.unet_mode,
            {
                "small": build_unet_cpu_small,
                "medium": build_unet_cpu_medium,
                "large": build_unet_cpu_large,
            },
        )

        if torch.cuda.is_available() and "cuda" in cfg.device:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cpu")

        self.model = builder(in_channels=3, out_channels=4).to(self.device)
        self.model.eval()

        model_file = cfg.model_file or f"best_{cfg.unet_mode}.pth"
        self.ckpt_path = os.path.join(cfg.model_dir, model_file)

        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            fixed = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(fixed, strict=True)

        self.use_amp = bool(cfg.use_amp and self.device.type == "cuda")

        self.inst_seg = (
            InstanceSegmenter(InstanceSegmenterConfig.from_dict(cfg.instance_cfg))
            if cfg.compute_instances
            else None
        )

    def _to_chw_numpy(self, img: np.ndarray) -> np.ndarray:
        """Convert an image to ``(3, H, W)`` float32 in ``[0, 1]``.

        Parameters
        ----------
        img : numpy.ndarray
            Input image.

        Returns
        -------
        numpy.ndarray
            Contiguous ``float32`` image with shape ``(3, H, W)``.
        """
        return to_chw_numpy(img)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert a single image to a normalized torch tensor.

        Parameters
        ----------
        img : numpy.ndarray
            Input image.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(1, 3, H, W)`` on the configured device.
        """
        x = self._to_chw_numpy(img)
        t = torch.from_numpy(x).unsqueeze(0).to(self.device, non_blocking=True)

        if self.cfg.normalize:
            t = normalize_tensor(t)

        return t

    def _to_tensor_tiles(self, tiles: np.ndarray) -> torch.Tensor:
        """Convert image tiles to a torch tensor.

        Parameters
        ----------
        tiles : numpy.ndarray
            Tile batch with shape ``(T, 3, H, W)`` or ``(T, H, W, 3)``.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(T, 3, H, W)`` on the configured device.
        """
        return to_tensor_tiles(
            tiles=tiles,
            device=self.device,
            normalize=self.cfg.normalize,
        )

    @torch.inference_mode()
    def predict_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """Predict probability maps for a tile tensor.

        Parameters
        ----------
        tiles : torch.Tensor
            Tile tensor with shape ``(T, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            CPU tensor with shape ``(T, 4, H, W)`` and dtype ``float32``.
        """
        if tiles.device != self.device:
            tiles = tiles.to(self.device, non_blocking=True)

        use_bf16 = self.use_amp and torch.cuda.is_bf16_supported()

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            enabled=self.use_amp,
        ):
            logits = self.model(tiles)
            probs = torch.sigmoid(logits)

        return probs.detach().to(torch.float32).cpu()

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        """Run segmentation on a single image or a tile batch.

        Parameters
        ----------
        img : numpy.ndarray
            Single image or tile batch.

        Returns
        -------
        dict
            Segmentation output. For single images, probability maps have shape
            ``(H, W)``. For tile batches, probability maps have shape
            ``(T, H, W)``.
        """
        if self.cfg.input_is_tiles or img.ndim == 4:
            return self._predict_tile_batch(img)

        x = self._to_tensor(img)
        use_bf16 = self.use_amp and torch.cuda.is_bf16_supported()

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            enabled=self.use_amp,
        ):
            logits = self.model(x)
            probs = torch.sigmoid(logits)

        probs_np = probs.squeeze(0).detach().to(torch.float32).cpu().numpy()
        cell_p, bound_p, center_p, energy_p = probs_np

        out: Dict[str, Any] = {
            "probs": {
                "cell": cell_p.astype(np.float32),
                "bound": bound_p.astype(np.float32),
                "center": center_p.astype(np.float32),
                "energy": energy_p.astype(np.float32),
            },
            "cell_mask": (cell_p >= self.cfg.thr_cell).astype(np.uint8),
            "boundary": (bound_p >= self.cfg.thr_bound).astype(np.uint8),
            "instance_labels": None,
            "meta": {
                "unet_mode": self.cfg.unet_mode,
                "device": str(self.device),
                "checkpoint": os.path.abspath(self.ckpt_path),
            },
        }

        if self.cfg.return_logits:
            out["logits"] = logits.squeeze(0).detach().cpu().float().numpy()

        if self.inst_seg is not None:
            out = self.inst_seg(out, update_cell_mask=True)

        return out

    def _predict_tile_batch(self, img: np.ndarray) -> Dict[str, Any]:
        """Run segmentation on an input batch of tiles.

        Parameters
        ----------
        img : numpy.ndarray
            Tile batch with shape ``(T, 3, H, W)`` or ``(T, H, W, 3)``.

        Returns
        -------
        dict
            Batched segmentation output.
        """
        tiles_t = self._to_tensor_tiles(img)
        use_bf16 = self.use_amp and torch.cuda.is_bf16_supported()

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=(torch.bfloat16 if use_bf16 else torch.float16),
            enabled=self.use_amp,
        ):
            logits = self.model(tiles_t)
            probs = torch.sigmoid(logits)

        probs_np = probs.detach().to(torch.float32).cpu().numpy()
        t_count, _, _, _ = probs_np.shape

        cell_p = probs_np[:, 0, ...]
        bound_p = probs_np[:, 1, ...]
        center_p = probs_np[:, 2, ...]
        energy_p = probs_np[:, 3, ...]

        out: Dict[str, Any] = {
            "probs": {
                "cell": cell_p.astype(np.float32),
                "bound": bound_p.astype(np.float32),
                "center": center_p.astype(np.float32),
                "energy": energy_p.astype(np.float32),
            },
            "cell_mask": (cell_p >= self.cfg.thr_cell).astype(np.uint8),
            "boundary": (bound_p >= self.cfg.thr_bound).astype(np.uint8),
            "instance_labels": None,
            "meta": {
                "unet_mode": self.cfg.unet_mode,
                "device": str(self.device),
                "checkpoint": os.path.abspath(self.ckpt_path),
                "batched": True,
                "n_tiles": int(t_count),
            },
        }

        if self.cfg.return_logits:
            out["logits"] = logits.detach().cpu().float().numpy()

        if self.inst_seg is not None:
            out["instance_labels"] = [
                self.inst_seg(
                    {
                        "probs": {
                            "cell": cell_p[i],
                            "bound": bound_p[i],
                            "center": center_p[i],
                            "energy": energy_p[i],
                        },
                        "cell_mask": (cell_p[i] >= self.cfg.thr_cell).astype(np.uint8),
                        "boundary": (bound_p[i] >= self.cfg.thr_bound).astype(np.uint8),
                        "instance_labels": None,
                        "meta": {},
                    },
                    update_cell_mask=True,
                )["instance_labels"]
                for i in range(t_count)
            ]

        return out

    @classmethod
    def from_config(cls: Type[Segmenter], cfg_dict: Dict[str, Any]) -> Segmenter:
        """Create a segmenter from a configuration dictionary.

        Parameters
        ----------
        cfg_dict : dict
            Segmenter configuration values. Unknown keys are ignored.

        Returns
        -------
        SegmenterUNet
            Initialized segmenter.
        """
        cfg = SegmenterConfig.from_dict(cfg_dict)
        return cls(cfg)


class SegmenterUNetInference(SegmenterUNet):
    """Run full-image UNet inference with internal tiling.

    This class splits a full image into tiles, predicts probability maps for
    all tiles, averages overlapping regions, and computes instance labels on
    the stitched probability maps.

    Parameters
    ----------
    cfg : SegmenterConfig
        Segmenter configuration. ``compute_instances`` is forced to ``True``.
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
                InstanceSegmenterConfig.from_dict(self.cfg.instance_cfg)
            )

    def _iter_sliding_windows(self, height: int, width: int):
        """Yield sliding-window coordinates.

        Parameters
        ----------
        height : int
            Image height.
        width : int
            Image width.

        Yields
        ------
        tuple of int
            Coordinates ``(y0, y1, x0, x1)``.
        """
        yield from iter_sliding_windows(
            height=height,
            width=width,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

    def _tile_image_numpy(
        self,
        image_chw: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Split a CHW image into fixed-size tiles.

        Parameters
        ----------
        image_chw : numpy.ndarray
            Image with shape ``(3, H, W)``.

        Returns
        -------
        tiles : numpy.ndarray
            Tile batch with shape ``(T, 3, tile_size, tile_size)``.
        orig_hw : tuple of int
            Original image size as ``(H, W)``.
        """
        return tile_image_numpy(
            image_chw=image_chw,
            tile_size=self.tile_size,
            overlap=self.overlap,
            pad_value=self.pad_value,
        )

    def _reconstruct_from_tiles_probability(
        self,
        tiles: np.ndarray,
        orig_hw: Tuple[int, int],
    ) -> np.ndarray:
        """Reconstruct full-size probability maps from tiled predictions.

        Parameters
        ----------
        tiles : numpy.ndarray
            Tile predictions with shape ``(T, C, tile_size, tile_size)``.
        orig_hw : tuple of int
            Original image size as ``(H, W)``.

        Returns
        -------
        numpy.ndarray
            Reconstructed probability maps with shape ``(C, H, W)``.
        """
        return reconstruct_from_tiles_probability(
            tiles=tiles,
            orig_hw=orig_hw,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        """Run full-image inference.

        Parameters
        ----------
        img : numpy.ndarray
            Input image. If a 4D array is passed, the base tile-batch behavior
            is used.

        Returns
        -------
        dict
            Output dictionary containing stitched probability maps, instance
            labels, and metadata.
        """
        if img.ndim == 4:
            return super().__call__(img)

        img_chw = self._to_chw_numpy(img)
        tiles_np, orig_hw = self._tile_image_numpy(img_chw)

        tiles_t = self._to_tensor_tiles(tiles_np)
        probs_t = self.predict_tiles(tiles_t)
        probs_np_tiles = probs_t.numpy()

        probs_full = self._reconstruct_from_tiles_probability(
            probs_np_tiles,
            orig_hw,
        )

        cell_p, bound_p, center_p, energy_p = probs_full

        seg_out: Dict[str, Any] = {
            "probs": {
                "cell": cell_p,
                "bound": bound_p,
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

        if self.inst_seg is not None:
            seg_out = self.inst_seg(seg_out, update_cell_mask=False)

        return {
            "instance_labels": seg_out["instance_labels"],
            "probs": {
                "cell": cell_p,
                "bound": bound_p,
                "center": center_p,
                "energy": energy_p,
            },
            "meta": seg_out["meta"],
        }


class InstanceSegmenter:
    """Convert UNet probability maps into instance labels.

    The instance segmentation uses a hysteresis cell mask, center-based
    watershed seeds, and an elevation image built from distance, boundary,
    edge, and energy terms.

    Parameters
    ----------
    cfg : InstanceSegmenterConfig
        Instance segmentation configuration.
    """

    def __init__(self, cfg: InstanceSegmenterConfig):
        self.cfg = cfg

        radius = int(self.cfg.mask_close_radius)
        self._mask_close_selem = morphology.disk(radius) if radius > 0 else None

    def __call__(
        self,
        seg_out: Dict[str, Any],
        update_cell_mask: bool = True,
    ) -> Dict[str, Any]:
        """Compute instance labels from segmentation probabilities.

        Parameters
        ----------
        seg_out : dict
            Segmentation output dictionary containing ``seg_out["probs"]["cell"]``.
            Optional probability maps are ``"bound"``, ``"center"``, and
            ``"energy"``.
        update_cell_mask : bool, optional
            Whether to write the hysteresis-derived cell mask back into
            ``seg_out["cell_mask"]``. The default is ``True``.

        Returns
        -------
        dict
            The input dictionary with ``"instance_labels"`` added or updated.

        Raises
        ------
        ValueError
            If the cell probability map is missing.
        """
        probs = seg_out.get("probs", {})

        p_cell = as_contiguous_f32(probs.get("cell"))
        p_bound = as_contiguous_f32(probs.get("bound"))
        p_center = as_contiguous_f32(probs.get("center"))
        p_energy = as_contiguous_f32(probs.get("energy"))

        if p_cell is None:
            raise ValueError("seg_out['probs']['cell'] required")

        height, width = p_cell.shape

        mask = hysteresis_mask(
            p_cell=p_cell,
            low_thr=self.cfg.cell_mask_low_thr,
            high_thr=self.cfg.cell_mask_high_thr,
            close_selem=self._mask_close_selem,
            min_hole_area=self.cfg.min_hole_area,
            min_object_area=self.cfg.min_object_area,
        )

        dist = ndi.distance_transform_edt(mask).astype(np.float32)

        if self.cfg.distance_smooth_sigma > 0:
            ndi.gaussian_filter(
                dist,
                float(self.cfg.distance_smooth_sigma),
                output=dist,
            )

        dist_s = dist

        elevation = np.zeros((height, width), dtype=np.float32)
        tmp = np.empty_like(elevation, dtype=np.float32)

        if self.cfg.distance_weight != 0:
            dmax = dist_s.max()
            if dmax > 1e-6:
                np.divide(dist_s, dmax, out=tmp)
                elevation -= self.cfg.distance_weight * tmp

        if self.cfg.use_boundary and p_bound is not None:
            b = smooth01(p_bound, self.cfg.smooth_boundary_sigma)
            elevation += self.cfg.gamma_boundary * b

        if self.cfg.use_edge_term and self.cfg.edge_weight != 0:
            g = ndi.gaussian_gradient_magnitude(
                p_cell,
                sigma=float(self.cfg.edge_sigma),
            )
            gmax = g.max()
            if gmax > 1e-6:
                np.divide(g, gmax, out=g)
                elevation += self.cfg.edge_weight * g

        if self.cfg.use_energy and p_energy is not None and self.cfg.energy_weight != 0:
            e = smooth01(p_energy, self.cfg.energy_smooth_sigma)
            elevation -= self.cfg.energy_weight * e

        markers = make_markers(
            mask=mask,
            p_center=p_center,
            dist_s=dist_s,
            use_centers=self.cfg.use_centers,
            center_seed_method=self.cfg.center_seed_method,
            center_min_distance=self.cfg.center_min_distance,
            center_thr=self.cfg.center_thr,
        )

        instances = segmentation.watershed(
            image=elevation,
            markers=markers,
            mask=mask,
            compactness=float(self.cfg.compactness),
            watershed_line=bool(self.cfg.watershed_line),
        ).astype(np.int32)

        if self.cfg.min_instance_area > 0:
            instances = morphology.remove_small_objects(
                instances,
                min_size=int(self.cfg.min_instance_area),
            ).astype(np.int32)

        if update_cell_mask:
            seg_out["cell_mask"] = mask.astype(np.uint8)

        seg_out["instance_labels"] = instances
        return seg_out

    @classmethod
    def from_config(cls, cfg_dict: Dict[str, Any]) -> "InstanceSegmenter":
        """Create an instance segmenter from a configuration dictionary.

        Parameters
        ----------
        cfg_dict : dict
            Instance segmentation configuration values. Unknown keys are
            ignored.

        Returns
        -------
        InstanceSegmenter
            Initialized instance segmenter.
        """
        cfg = InstanceSegmenterConfig.from_dict(cfg_dict)
        return cls(cfg)
