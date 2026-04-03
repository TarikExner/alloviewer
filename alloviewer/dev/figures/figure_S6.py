import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import seaborn as sns

import copy

from typing import Any

from .figure_data_generation import get_validation_data, generate_unet_comparison

from alloviewer.image_analysis.config import UNET_CONFIG
from alloviewer.image_analysis.segmenter import SegmenterUNet

from . import figure_config as cfg
from . import figure_utils as utils


def _generate_main_figure(
    data: pd.DataFrame,
    res: dict,
    figure_output_dir: str = "",
    figure_name: str = "",
    tile_idx: int = 5
    
):

    plot_params = {
        "data": data,
        "x": "dataset_mode",
        "order": ["pad_resize", "crop_well_resize", "tiling"],
        "hue": "unet_mode",
        "whis": (0,100)
    }

    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1, 1)
        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.boxplot(**plot_params, y = "mask_iou", ax = accuracy_plot)

        accuracy_plot.legend(bbox_to_anchor = (0.5, -0.25), loc = "upper center",
                             title = "UNET size", fontsize = cfg.AXIS_LABEL_SIZE,
                             title_fontsize = cfg.AXIS_LABEL_SIZE, ncols = 3)
        accuracy_plot.set_xlabel("")
        accuracy_plot.set_title("Mask prediction", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylabel("jaccard score", fontsize = cfg.AXIS_LABEL_SIZE)

        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)


    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1, 1)
        accuracy_plot = fig.add_subplot(fig_sgs[0])

        p_params = plot_params.copy()
        p_params.pop("whis")
        sns.violinplot(**p_params, y = "boundary_f1", ax = accuracy_plot, inner = None)

        accuracy_plot.legend(bbox_to_anchor = (0.5, -0.25), loc = "upper center",
                             title = "UNET size", fontsize = cfg.AXIS_LABEL_SIZE,
                             title_fontsize = cfg.AXIS_LABEL_SIZE, ncols = 3)
        accuracy_plot.set_xlabel("")
        accuracy_plot.set_title("Boundary prediction", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)

        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1, 1)
        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.boxplot(**plot_params, y = "center_f1", ax = accuracy_plot)

        accuracy_plot.legend(bbox_to_anchor = (0.5, -0.25), loc = "upper center",
                             title = "UNET size", fontsize = cfg.AXIS_LABEL_SIZE,
                             title_fontsize = cfg.AXIS_LABEL_SIZE, ncols = 3)
        accuracy_plot.set_xlabel("")
        accuracy_plot.set_title("Center prediction", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)

        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        
    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1, 1, wspace = 0, hspace = 0)
        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.boxplot(**plot_params, y = "energy_ssim", ax = accuracy_plot)

        accuracy_plot.legend(bbox_to_anchor = (0.5, -0.25), loc = "upper center",
                             title = "UNET size", fontsize = cfg.AXIS_LABEL_SIZE,
                             title_fontsize = cfg.AXIS_LABEL_SIZE, ncols = 3)
        accuracy_plot.set_xlabel("")
        accuracy_plot.set_title("Energy prediction", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylabel("SSIM score", fontsize = cfg.AXIS_LABEL_SIZE)

        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

    def generate_subfigure_e(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(3, 4)
        sub_axes = fig_sgs.subplots()  # shape: (3, 4)
        for i, unet_mode in enumerate(["small", "med", "large"]):
            for j, img_type in enumerate(["cell", "bound", "center", "energy"]):
                ax_img = sub_axes[i, j]
                ax_img.imshow(res[unet_mode][img_type][tile_idx])
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_visible(False)
                # column labels (top row)
                if i == 0:
                    ax_img.set_title(img_type, fontsize=cfg.TITLE_SIZE, pad=4)
    
                # row labels (left-most column)
                if j == 0:
                    ax_img.set_ylabel(
                        f"UNET size: {unet_mode}",
                        fontsize=cfg.TITLE_SIZE,
                        rotation=90,
                        color="black",
                    )
    
    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=3, figure=fig, height_ratios=[1,1,2.5])
    a_coords = gs[0, :3]
    b_coords = gs[0, 3:]
    c_coords = gs[1, :3]
    d_coords = gs[1, 3:]
    e_coords = gs[2, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)
    fig_e = fig.add_subplot(e_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")
    generate_subfigure_d(fig, fig_d, d_coords, "D")
    generate_subfigure_e(fig, fig_e, e_coords, "E")

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")


def figure_S6_generation(
    figure_output_dir: str,
    model_output_dir: str,
    validation_results_dir: str,
    figure_data_dir: str,
    h5_path: str,
    **kwargs
):

    unet_base_config = copy.deepcopy(UNET_CONFIG)

    segmenter_class = SegmenterUNet

    data = get_validation_data(validation_results_dir,
                               mode = "training")
    res = generate_unet_comparison(models_dir = model_output_dir,
                                   h5_path = h5_path,
                                   unet_base_config = unet_base_config,
                                   segmenter_class = segmenter_class,
                                   output_dir = figure_data_dir)
    tile_idx = 5
    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S6",
        data = data,
        res = res,
        tile_idx = tile_idx
    )
