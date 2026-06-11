import os
import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from alloviewer.image_analysis.config import UNET_CONFIG, INSTANCE_CONFIG
from alloviewer.image_analysis.segmenter import (
    SegmenterUNet,
    InstanceSegmenter,
    InstanceSegmenterConfig,
)
from alloviewer.image_analysis.utils import (
    as_contiguous_f32,
    hysteresis_mask,
    make_markers,
    smooth01,
)

from .figure_data_generation import get_validation_data, generate_unet_comparison
from . import figure_config as cfg
from . import figure_utils as utils


def _as_instance_cfg(cfg_inst):
    if isinstance(cfg_inst, InstanceSegmenterConfig):
        return cfg_inst

    if isinstance(cfg_inst, dict):
        return InstanceSegmenterConfig.from_dict(cfg_inst)

    return cfg_inst


def _as_2d_prob(x: np.ndarray, name: str) -> np.ndarray:
    x = as_contiguous_f32(np.asarray(x))

    if x.ndim == 2:
        return x

    if x.ndim == 3 and x.shape[0] == 1:
        return as_contiguous_f32(x[0])

    raise ValueError(f"Expected {name} as [H,W], got shape {x.shape}.")


def _make_seg_out_from_probs(probs: dict) -> dict:
    return {
        "probs": {
            "cell": _as_2d_prob(probs["cell"], "cell"),
            "bound": _as_2d_prob(probs["bound"], "bound"),
            "center": _as_2d_prob(probs["center"], "center"),
            "energy": _as_2d_prob(probs["energy"], "energy"),
        },
        "cell_mask": None,
        "boundary": None,
        "instance_labels": None,
        "meta": {},
    }


def compute_instance_steps(seg_out: dict, cfg_inst) -> dict:
    """
    Compute display intermediates for one 2D UNet probability output.

    Final instances are generated through InstanceSegmenter. The intermediate
    arrays are reconstructed using the same utility functions used by
    InstanceSegmenter.
    """
    cfg_inst = _as_instance_cfg(cfg_inst)

    p_cell = _as_2d_prob(seg_out["probs"]["cell"], "cell")
    p_bound = _as_2d_prob(seg_out["probs"]["bound"], "bound")
    p_center = _as_2d_prob(seg_out["probs"]["center"], "center")
    p_energy = _as_2d_prob(seg_out["probs"]["energy"], "energy")

    close_selem = (
        morphology.disk(int(cfg_inst.mask_close_radius))
        if int(cfg_inst.mask_close_radius) > 0
        else None
    )

    mask = hysteresis_mask(
        p_cell=p_cell,
        low_thr=cfg_inst.cell_mask_low_thr,
        high_thr=cfg_inst.cell_mask_high_thr,
        close_selem=close_selem,
        min_hole_area=cfg_inst.min_hole_area,
        min_object_area=cfg_inst.min_object_area,
    )

    dist = ndi.distance_transform_edt(mask).astype(np.float32)

    if cfg_inst.distance_smooth_sigma > 0:
        ndi.gaussian_filter(
            dist,
            float(cfg_inst.distance_smooth_sigma),
            output=dist,
        )

    dist_s = dist

    height, width = p_cell.shape
    elevation = np.zeros((height, width), dtype=np.float32)
    tmp = np.empty_like(elevation, dtype=np.float32)

    if cfg_inst.distance_weight != 0:
        dmax = dist_s.max()
        if dmax > 1e-6:
            np.divide(dist_s, dmax, out=tmp)
            elevation -= cfg_inst.distance_weight * tmp

    if cfg_inst.use_boundary and p_bound is not None:
        b = smooth01(p_bound, cfg_inst.smooth_boundary_sigma)
        elevation += cfg_inst.gamma_boundary * b

    if cfg_inst.use_edge_term and cfg_inst.edge_weight != 0:
        g = ndi.gaussian_gradient_magnitude(
            p_cell,
            sigma=float(cfg_inst.edge_sigma),
        )
        gmax = g.max()
        if gmax > 1e-6:
            np.divide(g, gmax, out=g)
            elevation += cfg_inst.edge_weight * g

    if cfg_inst.use_energy and p_energy is not None and cfg_inst.energy_weight != 0:
        e = smooth01(p_energy, cfg_inst.energy_smooth_sigma)
        elevation -= cfg_inst.energy_weight * e

    markers = make_markers(
        mask=mask,
        p_center=p_center,
        dist_s=dist_s,
        use_centers=cfg_inst.use_centers,
        center_seed_method=cfg_inst.center_seed_method,
        center_min_distance=cfg_inst.center_min_distance,
        center_thr=cfg_inst.center_thr,
    )

    instance_segmenter = InstanceSegmenter(cfg_inst)
    inst_out = instance_segmenter(
        _make_seg_out_from_probs(
            {
                "cell": p_cell,
                "bound": p_bound,
                "center": p_center,
                "energy": p_energy,
            }
        ),
        update_cell_mask=True,
    )

    return {
        "p_cell": p_cell,
        "p_bound": p_bound,
        "p_center": p_center,
        "p_energy": p_energy,
        "mask": inst_out.get("cell_mask", mask),
        "dist": dist,
        "dist_s": dist_s,
        "elevation": elevation,
        "markers": markers,
        "instances": inst_out["instance_labels"],
    }


def random_instance_colors(instances, background_label=0, seed=0):
    rng = np.random.default_rng(seed)

    instances = np.asarray(instances)
    labels = np.unique(instances)
    labels = labels[labels != background_label]

    h, w = instances.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    for lab in labels:
        color = rng.uniform(0.2, 1.0, size=3)
        out[instances == lab] = color

    return out


def show_panel(ax, img, title=None, cmap="gray", vmin=None, vmax=None):
    im = ax.imshow(
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=cfg.TITLE_SIZE)

    return im


def _crop_array(a, sy, sx, size):
    if a is None:
        return None

    a = np.asarray(a)

    if a.ndim == 2:
        return a[sy:sy + size, sx:sx + size]

    if a.ndim == 3:
        return a[sy:sy + size, sx:sx + size, :]

    raise ValueError(f"Cannot crop array with shape {a.shape}.")


def _prepare_image_for_display(img):
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"Expected image [H,W] or [H,W,C], got shape {img.shape}.")

    if img.max() > 1.0:
        img = img / img.max()

    return img


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
    img_disp = _prepare_image_for_display(img)

    cell_prob = _as_2d_prob(seg_out["probs"]["cell"], "cell")
    cell_mask = (cell_prob > 0.5).astype(np.uint8)
    instances_baseline = measure.label(cell_mask, connectivity=1).astype(np.int32)

    steps = compute_instance_steps(seg_out, inst_cfg)

    inst_baseline_rgb_full = random_instance_colors(
        instances_baseline,
        background_label=0,
        seed=1,
    )
    inst_instseg_rgb_full = random_instance_colors(
        steps["instances"],
        background_label=0,
        seed=2,
    )

    if inset_start is not None:
        y0, x0 = inset_start
        h, w = steps["p_cell"].shape

        y0 = max(0, min(int(y0), h - 1))
        x0 = max(0, min(int(x0), w - 1))

        if y0 + inset_size > h:
            y0 = max(0, h - inset_size)

        if x0 + inset_size > w:
            x0 = max(0, w - inset_size)

        img_disp = _crop_array(img_disp, y0, x0, inset_size)

        for key in list(steps.keys()):
            steps[key] = _crop_array(steps[key], y0, x0, inset_size)

        inst_baseline_rgb = _crop_array(
            inst_baseline_rgb_full,
            y0,
            x0,
            inset_size,
        )
        inst_instseg_rgb = _crop_array(
            inst_instseg_rgb_full,
            y0,
            x0,
            inset_size,
        )
    else:
        inst_baseline_rgb = inst_baseline_rgb_full
        inst_instseg_rgb = inst_instseg_rgb_full

    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(2, 4)
        axes_A = fig_sgs.subplots()

        show_panel(axes_A[0, 0], img_disp, "Original")
        show_panel(axes_A[0, 1], steps["p_cell"], "Cell prob. $p_{cell}$", cmap="viridis")
        show_panel(axes_A[0, 2], steps["mask"], "Hysteresis mask", cmap="gray")
        show_panel(axes_A[0, 3], steps["dist_s"], "Distance (smoothed)", cmap="magma")

        show_panel(axes_A[1, 0], steps["p_bound"], "Boundary prob.", cmap="viridis")
        show_panel(axes_A[1, 1], steps["elevation"], "Elevation map", cmap="magma")
        show_panel(axes_A[1, 2], steps["markers"], "Markers", cmap="jet")

        axes_A[1, 3].imshow(img_disp, cmap="gray", interpolation="nearest")
        axes_A[1, 3].imshow(
            random_instance_colors(steps["instances"]),
            alpha=0.7,
            interpolation="nearest",
        )
        axes_A[1, 3].set_title("Instances", fontsize=cfg.TITLE_SIZE)
        axes_A[1, 3].set_xticks([])
        axes_A[1, 3].set_yticks([])

    def generate_subfigure_b(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 3)
        axes_B = fig_sgs.subplots()

        show_panel(axes_B[0], img_disp, "Original")

        axes_B[1].imshow(inst_baseline_rgb)
        axes_B[1].set_title(
            "Conventional mask segmentation",
            fontsize=cfg.TITLE_SIZE,
        )
        axes_B[1].set_xticks([])
        axes_B[1].set_yticks([])

        axes_B[2].imshow(inst_instseg_rgb)
        axes_B[2].set_title(
            "InstanceSegmenter mask segmentation",
            fontsize=cfg.TITLE_SIZE,
        )
        axes_B[2].set_xticks([])
        axes_B[2].set_yticks([])

    def generate_subfigure_c(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        conv_df = conv_data.copy().reset_index(drop=True)
        inst_df = inst_seg_data.copy().reset_index(drop=True)

        fig_sgs = gs.subgridspec(1, 2)
        left_plot = fig.add_subplot(fig_sgs[0])
        right_plot = fig.add_subplot(fig_sgs[1])

        sns.scatterplot(
            data=conv_df,
            x="n_cells_gt_instances",
            y="n_cells_pred_components_thr0p5",
            ax=left_plot,
            **cfg.SCATTER_KWARGS,
        )

        sns.scatterplot(
            data=inst_df,
            x="n_cells_gt_instances",
            y="n_cells_pred_instances",
            ax=right_plot,
            **cfg.SCATTER_KWARGS,
        )

        left_plot.set_title(
            "Conventional mask segmentation",
            fontsize=cfg.TITLE_SIZE,
        )
        right_plot.set_title(
            "InstanceSegmenter mask segmentation",
            fontsize=cfg.TITLE_SIZE,
        )

        x_min_l, x_max_l = left_plot.get_xlim()
        x_min_r, x_max_r = right_plot.get_xlim()
        y_min_l, y_max_l = left_plot.get_ylim()
        y_min_r, y_max_r = right_plot.get_ylim()

        lo = min(x_min_l, x_min_r, y_min_l, y_min_r)
        hi = max(x_max_l, x_max_r, y_max_l, y_max_r)

        left_plot.set_xlim(lo, hi)
        left_plot.set_ylim(lo, hi)
        right_plot.set_xlim(lo, hi)
        right_plot.set_ylim(lo, hi)

        left_plot.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1)
        right_plot.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1)

        left_plot.set_xlabel("n_cells ground truth", fontsize=cfg.AXIS_LABEL_SIZE)
        left_plot.set_ylabel("n_cells predicted", fontsize=cfg.AXIS_LABEL_SIZE)
        right_plot.set_xlabel("n_cells ground truth", fontsize=cfg.AXIS_LABEL_SIZE)
        right_plot.set_ylabel("n_cells predicted", fontsize=cfg.AXIS_LABEL_SIZE)

        for plot_ax in [left_plot, right_plot]:
            plot_ax.tick_params(axis="both", which="major", labelsize=cfg.AXIS_LABEL_SIZE)

            for tick_label in plot_ax.get_xticklabels():
                tick_label.set_fontsize(cfg.AXIS_LABEL_SIZE)

            for tick_label in plot_ax.get_yticklabels():
                tick_label.set_fontsize(cfg.AXIS_LABEL_SIZE)

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=1,
        nrows=3,
        figure=fig,
        height_ratios=[1.5, 1, 1],
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
    ext_images_dir: str,
    figure_data_dir: str,
    validation_results_dir: str,
    h5_path: str,
    **kwargs,
):
    unet_base_config = copy.deepcopy(UNET_CONFIG)
    instance_seg_config = _as_instance_cfg(INSTANCE_CONFIG)
    segmenter_class = SegmenterUNet

    res = generate_unet_comparison(
        models_dir=model_output_dir,
        ext_images_dir=ext_images_dir,
        unet_base_config=unet_base_config,
        segmenter_class=segmenter_class,
        output_dir=validation_results_dir,
        output_filename="unet_segmentation_comparison",
        redo_analysis=kwargs.get("redo_analysis", False),
    )

    conv_data = get_validation_data(
        results_dir=validation_results_dir,
        mode="testing",
        unet_size="small",
        comparison_images="tiles",
        seg_method="conventional",
    )

    inst_seg_data = get_validation_data(
        results_dir=validation_results_dir,
        mode="testing",
        unet_size="small",
        comparison_images="tiles",
        seg_method="inst_seg",
    )

    seg_out = {
        "probs": res["small"],
    }

    img = res["original"]

    inset_start = kwargs.get("inset_start", (210, 200))
    inset_size = kwargs.get("inset_size", 100)

    _generate_main_figure(
        img=img,
        seg_out=seg_out,
        inst_cfg=instance_seg_config,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S7",
        inset_start=inset_start,
        inset_size=inset_size,
        conv_data=conv_data,
        inst_seg_data=inst_seg_data,
    )
