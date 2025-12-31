
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, feature

from typing import Any

from .figure_data_generation import get_validation_data, generate_unet_comparison

from . import figure_config as cfg
from . import figure_utils as utils


def _hysteresis_mask(pc, cfg_inst):
    low = float(cfg_inst.cell_mask_low_thr)
    high = float(cfg_inst.cell_mask_high_thr)

    strong = (pc >= high)
    weak   = (pc >= low)

    lab = measure.label(weak, connectivity=1)
    keep = np.zeros_like(weak, dtype=bool)

    if strong.any():
        strong_ids = np.unique(lab[strong])
        strong_ids = strong_ids[strong_ids != 0]
        for sid in strong_ids:
            keep |= (lab == sid)

    mask = keep

    if cfg_inst.mask_close_radius > 0:
        mask = morphology.binary_closing(
            mask,
            morphology.disk(int(cfg_inst.mask_close_radius))
        )

    mask = ndi.binary_fill_holes(mask)

    if cfg_inst.min_hole_area > 0:
        mask = morphology.remove_small_holes(
            mask,
            area_threshold=int(cfg_inst.min_hole_area)
        )

    if cfg_inst.min_object_area > 0:
        mask = morphology.remove_small_objects(
            mask,
            min_size=int(cfg_inst.min_object_area)
        )

    return mask.astype(bool)


def _smooth01(x, sigma):
    if sigma and sigma > 0:
        y = ndi.gaussian_filter(x.astype(np.float32), float(sigma))
        y = np.clip(y, 0.0, 1.0)
        return y
    return x.astype(np.float32)


def _make_markers(mask, p_center, dist_s, cfg_inst):
    work_mask = (mask.astype(bool) if mask.dtype != bool else mask)
    seeds_bool = np.zeros_like(work_mask, dtype=bool)

    if cfg_inst.use_centers and (p_center is not None):
        if getattr(cfg_inst, "center_seed_method", "nms") == "nms":
            coords = feature.peak_local_max(
                p_center.astype(np.float32),
                min_distance=int(cfg_inst.center_min_distance),
                threshold_abs=float(cfg_inst.center_thr),
                labels=work_mask,
                exclude_border=False,
            )
            seeds_center = np.zeros_like(work_mask, dtype=bool)
            if coords.size:
                seeds_center[tuple(coords.T)] = True
        else:
            seeds_center = (p_center >= float(cfg_inst.center_thr)) & work_mask

        seeds_bool |= seeds_center

    markers = measure.label(seeds_bool, connectivity=1).astype(np.int32)
    if markers.max() == 0:
        markers = measure.label(work_mask, connectivity=1).astype(np.int32)

    return markers


def compute_instance_steps(seg_out, cfg_inst, tile_idx=5):
    """
    seg_out: dict from SegmenterUNet (tiles version; probs[T,H,W])
    cfg_inst: InstanceSegmenterConfig (or any object with same attrs)
    tile_idx: which tile to use for visualization

    Returns a dict with intermediate arrays for that tile only.
    """
    p_cell   = seg_out["probs"]["cell"].astype(np.float32)[tile_idx]
    p_bound  = seg_out["probs"]["bound"].astype(np.float32)[tile_idx]
    p_center = seg_out["probs"]["center"].astype(np.float32)[tile_idx]
    p_energy = seg_out["probs"]["energy"].astype(np.float32)[tile_idx]

    h, w = p_cell.shape

    steps = {}
    steps["p_cell"]   = p_cell
    steps["p_bound"]  = p_bound
    steps["p_center"] = p_center
    steps["p_energy"] = p_energy

    # mask via hysteresis
    mask = _hysteresis_mask(p_cell, cfg_inst)
    steps["mask"] = mask

    # distance map and smoothed distance
    dist = ndi.distance_transform_edt(mask).astype(np.float32)
    dist_s = dist.copy()
    if cfg_inst.distance_smooth_sigma > 0:
        dist_s = ndi.gaussian_filter(dist, float(cfg_inst.distance_smooth_sigma))

    steps["dist"] = dist
    steps["dist_s"] = dist_s

    # elevation (cost) map
    elevation = np.zeros((h, w), dtype=np.float32)

    # distance term
    if cfg_inst.distance_weight != 0:
        dmax = dist_s.max()
        if dmax > 1e-6:
            elevation -= cfg_inst.distance_weight * (dist_s / dmax)

    # boundary term
    if cfg_inst.use_boundary and (p_bound is not None):
        b = _smooth01(p_bound, cfg_inst.smooth_boundary_sigma)
        elevation += cfg_inst.gamma_boundary * b

    # edge term
    if cfg_inst.use_edge_term and (cfg_inst.edge_weight != 0):
        g = ndi.gaussian_gradient_magnitude(
            p_cell.astype(np.float32),
            sigma=float(cfg_inst.edge_sigma)
        )
        gmax = g.max()
        if gmax > 1e-6:
            elevation += cfg_inst.edge_weight * (g / gmax)

    # energy term
    if cfg_inst.use_energy and (p_energy is not None) and (cfg_inst.energy_weight != 0):
        e = _smooth01(p_energy, cfg_inst.energy_smooth_sigma)
        elevation -= cfg_inst.energy_weight * e

    steps["elevation"] = elevation

    # markers (seeds)
    markers = _make_markers(mask, p_center, dist_s, cfg_inst)
    steps["markers"] = markers

    # watershed
    instances = segmentation.watershed(
        image=elevation,
        markers=markers,
        mask=mask,
        compactness=float(cfg_inst.compactness),
        watershed_line=bool(cfg_inst.watershed_line),
    ).astype(np.int32)

    if cfg_inst.min_instance_area > 0:
        instances = morphology.remove_small_objects(
            instances,
            min_size=int(cfg_inst.min_instance_area)
        ).astype(np.int32)

    steps["instances"] = instances

    return steps

def random_instance_colors(instances, background_label=0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.unique(instances)
    labels = labels[labels != background_label]

    h, w = instances.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    for lab in labels:
        color = rng.uniform(0.2, 1.0, size=3)
        out[instances == lab] = color

    return out

def show_panel(ax, img, title=None, cmap="gray", vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    return im

def _generate_main_figure(
    img,
    seg_out,
    inst_cfg,
    figure_output_dir: str,
    figure_name: str,
    conv_data: pd.DataFrame,
    inst_seg_data: pd.DataFrame,
    inset_start=None,
    inset_size: int = 50,
) -> None:
    """
    img: np.ndarray [H,W] or [H,W,3]
    seg_out: dict from SegmenterUNet (tiles/batched, with 'probs' and shape [T,H,W])
    inst_cfg: InstanceSegmenterConfig
    inset_start: (row, col) start of inset crop for display.
    inset_size: side length of square inset (default 50).

    All analysis is done on full-size arrays; cropping is only applied for display.
    """

    tile_idx = 5

    img_disp = img.astype(np.float32)
    if img_disp.max() > 0:
        img_disp = img_disp / img_disp.max()

    # baseline instances from UNet cell prob (conventional mask segmentation)
    cell_mask = (seg_out["probs"]["cell"][tile_idx] > 0.5).astype(np.uint8)
    instances_baseline = measure.label(cell_mask, connectivity=1).astype(np.int32)

    steps = compute_instance_steps(seg_out, inst_cfg, tile_idx=tile_idx)
    instances_instseg = steps["instances"]

    inst_baseline_rgb_full = random_instance_colors(
        instances_baseline, background_label=0, seed=1
    )
    inst_instseg_rgb_full = random_instance_colors(
        instances_instseg, background_label=0, seed=2
    )

    def _crop_array(a, sy, sx, size):
        if a is None:
            return None
        if a.ndim == 2:
            return a[sy:sy + size, sx:sx + size]
        elif a.ndim == 3:
            return a[sy:sy + size, sx:sx + size, :]
        return a

    if inset_start is not None:
        y0, x0 = inset_start
        H, W = steps["p_cell"].shape
        y0 = max(0, min(y0, H - 1))
        x0 = max(0, min(x0, W - 1))
        if y0 + inset_size > H:
            y0 = max(0, H - inset_size)
        if x0 + inset_size > W:
            x0 = max(0, W - inset_size)

        img_disp = _crop_array(img_disp, y0, x0, inset_size)
        for key in list(steps.keys()):
            steps[key] = _crop_array(steps[key], y0, x0, inset_size)
        inst_baseline_rgb = _crop_array(inst_baseline_rgb_full, y0, x0, inset_size)
        inst_instseg_rgb = _crop_array(inst_instseg_rgb_full, y0, x0, inset_size)
    else:
        inst_baseline_rgb = inst_baseline_rgb_full
        inst_instseg_rgb = inst_instseg_rgb_full

    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(2, 4)
        axes_A = fig_sgs.subplots()

        # Row 0: original, p_cell, mask, distance
        show_panel(axes_A[0, 0], img_disp, "Original")
        show_panel(axes_A[0, 1], steps["p_cell"], "Cell prob. $p_{cell}$", cmap="viridis")
        show_panel(axes_A[0, 2], steps["mask"], "Hysteresis mask", cmap="gray")
        show_panel(axes_A[0, 3], steps["dist_s"], "Distance (smoothed)", cmap="magma")

        # Row 1: boundary, elevation, markers, instances
        show_panel(axes_A[1, 0], steps["p_bound"], "Boundary prob.", cmap="viridis")
        show_panel(axes_A[1, 1], steps["elevation"], "Elevation map", cmap="magma")
        show_panel(axes_A[1, 2], steps["markers"], "Markers", cmap="jet")

        axes_A[1, 3].imshow(img_disp, cmap="gray", interpolation="nearest")
        axes_A[1, 3].imshow(
            random_instance_colors(steps["instances"]), alpha=0.7, interpolation="nearest"
        )
        axes_A[1, 3].set_title("Instances", fontsize=cfg.TITLE_SIZE)
        axes_A[1, 3].set_xticks([])
        axes_A[1, 3].set_yticks([])

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1,3)
        axes_B = fig_sgs.subplots()

        show_panel(axes_B[0], img_disp, "Original")

        axes_B[1].imshow(inst_baseline_rgb)
        axes_B[1].set_title("Conventional mask segmentation", fontsize=cfg.TITLE_SIZE)
        axes_B[1].set_xticks([])
        axes_B[1].set_yticks([])

        axes_B[2].imshow(inst_instseg_rgb)
        axes_B[2].set_title("InstanceSegmenter mask segmentation", fontsize=cfg.TITLE_SIZE)
        axes_B[2].set_xticks([])
        axes_B[2].set_yticks([])

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1,2)
        left_plot = fig.add_subplot(fig_sgs[0])
        right_plot = fig.add_subplot(fig_sgs[1])
        sns.scatterplot(
            data = conv_data,
            x = "n_cells_gt_instances",
            y = "n_cells_pred_components_thr0p5",
            ax = left_plot,
            **cfg.SCATTER_KWARGS
        )
        sns.scatterplot(
            data = inst_seg_data,
            x = "n_cells_gt_instances",
            y = "n_cells_pred_instances",
            ax = right_plot,
            **cfg.SCATTER_KWARGS
        )

        left_plot.set_title("Conventional mask segmentation", fontsize = cfg.TITLE_SIZE)
        right_plot.set_title("InstanceSegmenter mask segmentation", fontsize = cfg.TITLE_SIZE)

        y_min_l, y_max_l = left_plot.get_ylim()
        y_min_r, y_max_r = right_plot.get_ylim()
        x_min_l, x_max_l = left_plot.get_xlim()
        x_min_r, x_max_r = right_plot.get_xlim()

        y_max = max(y_max_l, y_max_r)
        x_max = max(x_max_l, x_max_r)

        ax_max = max(y_max, x_max)

        left_plot.set_xlim(x_min_l, ax_max)
        left_plot.set_ylim(y_min_l, ax_max)
        right_plot.set_xlim(x_min_l, ax_max)
        right_plot.set_ylim(y_min_l, ax_max)

        left_plot.set_xlabel("n_cells ground truth")
        left_plot.set_ylabel("n_cells predicted")
        right_plot.set_xlabel("n_cells ground truth")
        right_plot.set_ylabel("n_cells predicted")

        utils.adjust_fontsize_ticklabels(left_plot, cfg.AXIS_LABEL_SIZE)
        utils.adjust_fontsize_ticklabels(right_plot, cfg.AXIS_LABEL_SIZE)

        return
        
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )
    gs = GridSpec(
        ncols=1,
        nrows=3,
        figure=fig,
        height_ratios=[1.5,1,1 ],
    )

    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def figure_S7_generation(
    figure_output_dir: str,
    model_output_dir: str,
    figure_data_dir: str,
    validation_results_dir: str,
    h5_path: str,
    unet_base_config: Any,
    instance_seg_config: Any,
    segmenter_class: Any,
    **kwargs
):
    
    seg_out = {}
    seg_out["probs"] = {}
    res = generate_unet_comparison(models_dir = model_output_dir,
                                   h5_path = h5_path,
                                   unet_base_config = unet_base_config,
                                   segmenter_class = segmenter_class,
                                   output_dir = figure_data_dir)

    conv_data = get_validation_data(results_dir = validation_results_dir,
                                    mode = "testing",
                                    unet_size = "small",
                                    comparison_images = "tiles",
                                    seg_method = "conventional")
    inst_seg_data = get_validation_data(results_dir = validation_results_dir,
                                        mode = "testing",
                                        unet_size = "small",
                                        comparison_images = "tiles",
                                        seg_method = "inst_seg")

    tile_idx = 5

    seg_out["probs"] = res["small"]
    img = res["original"][tile_idx].transpose((1,2,0))

    inset_start = (400,200)
    inset_size = 100

    _generate_main_figure(
        img=img,
        seg_out=seg_out,
        inst_cfg=instance_seg_config,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S7",
        inset_start=inset_start,
        inset_size=inset_size,
        conv_data=conv_data,
        inst_seg_data=inst_seg_data
    )
