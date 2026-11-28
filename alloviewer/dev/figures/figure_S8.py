"""Supplementary Figure S8: UNet vs Cellpose on simulated test images.

A: four metrics of final instance masks against stored H5 /inst ground truth.
B: representative RGB image; GT, UNet and Cellpose masks, boundaries, centers,
   and *instance-derived energy* (not a learned probability/energy head).

The energy image in each row is computed from its final instance segmentation
using the same distance-transform definition for all three methods.

PER_IMAGE: original full-resolution CPIJ scene scores.
PER_TILE: crops of reconstructed full-image predictions; overlapping tiles
          are not statistically independent.

Shared CSV loading and model inference live in figure_data_generation.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from alloviewer.dev.figures import figure_config as cfg
from alloviewer.dev.figures import figure_utils as utils
from .figure_data_generation import (
    load_figure_s8_simulated_metrics,
    load_figure_s8_simulated_example,
    plot_segmentation_supplement_metric,
)

MetricLevel = Literal["PER_IMAGE", "PER_TILE"]
S8_METRICS = (
    ("pixel_iou", "Foreground IoU"),
    ("boundary_f1_tol_px", "Boundary F1"),
    ("F1_center_4px", "Center F1 (4 px)"),
    ("instance_f1_iou_0p5", "Instance F1 (IoU ≥ 0.5)"),
)

# Same dark-blue-to-red color scale for the binary and energy maps, matching
# the older supplementary illustration. Values are always displayed in [0, 1].
MAP_CMAP = "jet"


def prepare_figure_S8_data(
    validation_results_dir: str,
    model_output_dir: str,
    h5_path: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
    seg_method: str = "inst_seg",
    img_index: int = 12,
    tile_idx: int = 7,
    model_file: str | None = None,
) -> dict[str, Any]:
    """Load CPIJ metrics and reconstruct one representative simulated field."""
    metrics, source = load_figure_s8_simulated_metrics(
        validation_results_dir,
        metrics=tuple(key for key, _ in S8_METRICS),
        metric_level=metric_level,
        unet_size=unet_size,
        seg_method=seg_method,
    )
    field = load_figure_s8_simulated_example(
        model_output_dir, h5_path, source,
        img_index=img_index, tile_idx=tile_idx,
        unet_size=unet_size, seg_method=seg_method, model_file=model_file,
    )
    return {"metrics": metrics, "field": field, "metric_level": metric_level}


def _instance_centers(labels: np.ndarray) -> np.ndarray:
    """Centroids in (y,x) of nonzero instance IDs, in the displayed crop."""
    yy, xx = np.nonzero(labels)
    if not len(yy):
        return np.empty((0, 2), dtype=float)
    _, compact = np.unique(labels[yy, xx], return_inverse=True)
    count = np.bincount(compact)
    return np.column_stack((
        np.bincount(compact, weights=yy) / count,
        np.bincount(compact, weights=xx) / count,
    ))


def make_energy_from_instances(instances: np.ndarray) -> np.ndarray:
    """Calculate normalized distance-transform energy for each instance."""

    instances = np.asarray(instances)
    energy = np.zeros(instances.shape, dtype=np.float32)

    if not np.any(instances):
        return energy

    # Process each instance separately.
    for instance_id, slices in enumerate(
        ndi.find_objects(instances), start=1
    ):
        if slices is None:
            continue

        instance_mask = instances[slices] == instance_id

        # Pad to ensure proper distance calculation at instance edges.
        padded = np.pad(
            instance_mask,
            1,
            mode="constant",
            constant_values=False,
        )

        dist = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
        dist = dist.astype(np.float32)

        # Normalize each instance individually.
        peak = float(dist[instance_mask].max())

        if peak > 0:
            dist /= peak

        crop = energy[slices]
        crop[instance_mask] = dist[instance_mask]

    return energy

def _enhance_rgb_for_display(
    rgb: np.ndarray,
    black_percentile: float = 50.0,
    white_percentile: float = 99.7,
) -> np.ndarray:
    """Improve fluorescence image contrast while preserving RGB ratios.

    Display only. Does not alter segmentation or model input.
    """

    rgb = np.asarray(rgb, dtype=np.float32)

    # Use one brightness value per pixel.
    brightness = rgb.max(axis=2)

    black, white = np.percentile(
        brightness,
        [black_percentile, white_percentile],
    )

    if white <= black:
        return np.clip(rgb, 0.0, 1.0)

    # Map background toward black and bright cells toward white.
    adjusted = np.clip(
        (brightness - black) / (white - black),
        0.0,
        1.0,
    )

    # Apply the same multiplier to all three channels at each pixel.
    # This preserves the R:G:B ratio of nonzero pixels.
    factor = np.divide(
        adjusted,
        brightness,
        out=np.zeros_like(brightness),
        where=brightness > 0,
    )

    return np.clip(rgb * factor[..., None], 0.0, 1.0)


def _visual_map(labels: np.ndarray, kind: str) -> np.ndarray:
    """Plotting maps derived strictly from the final instance labels."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected a 2D integer instance-label map")
    if kind == "mask":
        return (labels > 0).astype(np.float32)
    if kind == "boundary":
        return find_boundaries(labels, mode="inner").astype(np.float32)
    if kind == "centers":
        out = np.zeros(labels.shape, dtype=np.float32)
        centers = np.rint(_instance_centers(labels)).astype(int)
        if len(centers):
            centers[:, 0] = np.clip(centers[:, 0], 0, labels.shape[0] - 1)
            centers[:, 1] = np.clip(centers[:, 1], 0, labels.shape[1] - 1)
            out[centers[:, 0], centers[:, 1]] = 1.0
            out = ndi.maximum_filter(out, size=3)
        return out
    if kind == "energy":
        return make_energy_from_instances(labels)
    raise ValueError(f"Unknown visual kind: {kind}")


def _display_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image data to genuine RGB for Matplotlib without color distortion.

    The H5 tiles fed to UNet are channel-normalized using UNET_MEAN/STD.
    If a local figure_data_generation.py returns these normalized values
    instead of denormalized RGB, undo that normalization before showing them.
    Already-correct RGB in [0,1] is preserved. NO independent per-channel
    min/max rescaling is performed (which would alter red/green ratios).
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("Image contains non-finite RGB values")

    if np.issubdtype(rgb.dtype, np.integer):
        if rgb.min() < 0:
            raise ValueError("RGB image has negative integer values")
        # Matplotlib accepts uint8 RGB directly, but not uint16 as standard RGB.
        if rgb.dtype == np.uint8:
            return rgb
        maximum = float(np.iinfo(rgb.dtype).max)
        return np.clip(rgb.astype(np.float32) / maximum, 0.0, 1.0)

    rgb = rgb.astype(np.float32, copy=False)
    minimum, maximum = float(rgb.min()), float(rgb.max())
    if minimum >= 0.0 and maximum <= 1.0:
        return rgb

    # Typical normalized H5 tile range includes negative pixels and values >1.
    # This branch also handles a positive-only standardized tile, whose max >1.
    if minimum >= -10.0 and maximum <= 10.0:
        from alloviewer.dev.segmentation import UNET_MEAN, UNET_STD
        mean = np.asarray(UNET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(UNET_STD, dtype=np.float32).reshape(1, 1, 3)
        restored = rgb * std + mean
        if restored.min() < -0.05 or restored.max() > 1.05:
            raise ValueError(
                "H5 RGB values could not be restored using UNET_MEAN/STD. "
                f"Before [{minimum:.3f}, {maximum:.3f}], after "
                f"[{restored.min():.3f}, {restored.max():.3f}]."
            )
        return np.clip(restored, 0.0, 1.0).astype(np.float32)

    if minimum >= 0.0 and maximum <= 255.0:
        # Handle explicit float-valued 8-bit RGB, with a single scale for all channels.
        return np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unexpected RGB range [{minimum:.3f}, {maximum:.3f}]")


def _image_with_inset(
    ax: Axes,
    array: np.ndarray,
    *,
    cmap: str | None,
    inset_center: tuple[int, int],
    inset_box_px: int,
) -> None:
    options: dict[str, Any] = {"interpolation": "nearest"}
    if cmap is not None:
        options.update({"cmap": cmap, "vmin": 0.0, "vmax": 1.0})
    ax.imshow(array, **options)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    h, w = array.shape[:2]
    cy, cx = inset_center
    width = min(inset_box_px, h, w)
    if width <= 0:
        raise ValueError("Cannot draw inset for empty array")
    y0 = max(0, min(h - width, int(cy - width // 2)))
    x0 = max(0, min(w - width, int(cx - width // 2)))
    ax.add_patch(Rectangle(
        (x0, y0), width, width, fill=False, edgecolor="red", linewidth=0.9
    ))
    inset = inset_axes(
        ax, width="48%", height="48%", loc="upper right", borderpad=0.15
    )
    inset.imshow(array[y0:y0 + width, x0:x0 + width], **options)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.6)


def _generate_main_figure(
    figure_data: dict[str, Any],
    figure_output_dir: str,
    figure_name: str,
    *,
    inset_center: tuple[int, int] = (450, 150),
    inset_box_px: int = 96,
    max_strip_points: int | None = 400,
) -> None:
    metrics = figure_data["metrics"]
    field = figure_data["field"]
    rgb_display = _enhance_rgb_for_display(
        _display_rgb(field["image"])
    )

    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, letter: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, letter, x=0)
        cells = gs.subgridspec(1, 4, wspace=0.35)
        for i, (metric, title) in enumerate(S8_METRICS):
            plot_ax = fig.add_subplot(cells[0, i])
            plot_segmentation_supplement_metric(
                plot_ax, metrics, metric, title,
                max_strip_points=max_strip_points,
            )

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, letter: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, letter, x=0)
        cells = gs.subgridspec(3, 5, wspace=0.03, hspace=0.10)
        titles = ("Image", "Cell mask", "Cell boundary", "Cell centers", "Energy")
        row_sources = (("Ground truth", "gt"), ("UNet", "unet"), ("Cellpose", "cellpose"))
        for row, (label, key) in enumerate(row_sources):
            labels = field[key]
            for col, kind in enumerate(("image", "mask", "boundary", "centers", "energy")):
                current = fig.add_subplot(cells[row, col])
                if kind == "image":
                    panel_image, cmap = rgb_display, None
                else:
                    panel_image = _visual_map(labels, kind)
                    cmap = MAP_CMAP
                _image_with_inset(
                    current, panel_image, cmap=cmap,
                    inset_center=inset_center, inset_box_px=inset_box_px,
                )
                if row == 0:
                    current.set_title(titles[col], fontsize=cfg.TITLE_SIZE)
                if col == 0:
                    current.set_ylabel(label, fontsize=cfg.AXIS_LABEL_SIZE)

    fig = plt.figure(figsize=(6.375, 6.7), layout="constrained")
    try:
        grid = GridSpec(2, 1, figure=fig, height_ratios=(1, 2.25), hspace=0.15)
        generate_subfigure_a(fig, fig.add_subplot(grid[0]), grid[0], "A")
        generate_subfigure_b(fig, fig.add_subplot(grid[1]), grid[1], "B")
        Path(figure_output_dir).mkdir(parents=True, exist_ok=True)
        destination = Path(figure_output_dir) / figure_name
        fig.savefig(destination.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def figure_S8_generation(
    figure_output_dir: str,
    model_output_dir: str,
    validation_results_dir: str,
    h5_path: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
    seg_method: str = "inst_seg",
    img_index: int = 12,
    tile_idx: int = 7,
    model_file: str | None = None,
    figure_name: str | None = None,
    inset_center: tuple[int, int] = (450, 150),
    inset_box_px: int = 96,
    max_strip_points: int | None = 400,
    **kwargs,
) -> None:
    """Public S8 entry point: simulated CPIJ metrics and example energy maps."""
    figure_data = prepare_figure_S8_data(
        validation_results_dir, model_output_dir, h5_path,
        metric_level=metric_level, unet_size=unet_size, seg_method=seg_method,
        img_index=img_index, tile_idx=tile_idx, model_file=model_file,
    )
    _generate_main_figure(
        figure_data, figure_output_dir,
        figure_name=figure_name or (
            "Figure_S8" if metric_level == "PER_IMAGE" else "Figure_S8_PER_TILE"
        ),
        inset_center=inset_center, inset_box_px=inset_box_px,
        max_strip_points=max_strip_points,
    )

