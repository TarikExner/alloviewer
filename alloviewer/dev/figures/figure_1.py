
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, feature

from typing import Any

import cv2

from figures.figure_data_generation import get_validation_data, generate_unet_comparison

from figures import figure_config as cfg
from figures import figure_utils as utils

def unify_axis_limits(ax: Axes):
    y_lim_min, y_lim_max = ax.get_ylim()
    x_lim_min, x_lim_max = ax.get_xlim()

    _min = min(y_lim_min, x_lim_min)
    _max = max(y_lim_max, x_lim_max)

    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)

    return
def adjust_fontsize_ticklabels(ax: Axes, fontsize: int):
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)

def _generate_main_figure(
    unet_on_ext: pd.DataFrame,
    unet_on_sim: pd.DataFrame,
    imageJ_on_sim: pd.DataFrame,
    sketch_dir: str = "./sketches/",
    
) -> None:
    SCATTER_KWARGS = {
        "s": 4,
        "edgecolor": "black",
        "linewidth": 0.3,
        "rasterized": True
    }
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1,1)
        sketch = fig.add_subplot(fig_sgs[0])
        utils.prep_image_axis(sketch)
        img = cv2.imread(os.path.join(sketch_dir, "sim_unet_scheme.png"), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sketch.imshow(img)
        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1,3)

        unet_on_sim_plot = fig.add_subplot(fig_sgs[0])
        sns.scatterplot(data = unet_on_sim,
                        x = "n_cells_gt_instances",
                        y = "n_cells_pred_instances",
                        ax = unet_on_sim_plot,
                        **SCATTER_KWARGS)
        unet_on_sim_plot.set_title("UNet performance on\nsimulated images", fontsize = cfg.TITLE_SIZE)
        unet_on_sim_plot.set_xlabel("n_cells ground truth", fontsize = cfg.AXIS_LABEL_SIZE)
        unet_on_sim_plot.set_ylabel("n_cells predicted", fontsize = cfg.AXIS_LABEL_SIZE)
        unify_axis_limits(unet_on_sim_plot)
        adjust_fontsize_ticklabels(unet_on_sim_plot, cfg.AXIS_LABEL_SIZE)
                
        imageJ_on_sim_plot = fig.add_subplot(fig_sgs[1])
        imageJ_on_sim["dataset_mode"] = imageJ_on_sim["dataset_mode"].map({"UNet": "UNet", "imageJ": "ImageJ"})
        sns.scatterplot(data = imageJ_on_sim,
                        x = "n_cells_gt_instances",
                        y = "n_cells_pred_instances",
                        hue = "dataset_mode",
                        ax = imageJ_on_sim_plot,
                        **SCATTER_KWARGS)
        imageJ_on_sim_plot.set_title("UNet comparison to imageJ on\nsimulated images", fontsize = cfg.TITLE_SIZE)
        imageJ_on_sim_plot.set_xlabel("n_cells ground truth", fontsize = cfg.AXIS_LABEL_SIZE)
        imageJ_on_sim_plot.set_ylabel("n_cells predicted", fontsize = cfg.AXIS_LABEL_SIZE)
        handles, labels = imageJ_on_sim_plot.get_legend_handles_labels()
        imageJ_on_sim_plot.legend(handles, labels, markerscale = 4, title = "", fontsize = cfg.TITLE_SIZE)
        unify_axis_limits(imageJ_on_sim_plot)
        adjust_fontsize_ticklabels(imageJ_on_sim_plot, cfg.AXIS_LABEL_SIZE)
        
        unet_on_ext_plot = fig.add_subplot(fig_sgs[2])
        sns.scatterplot(data = unet_on_ext,
                        x = "n_cells_gt_instances",
                        y = "n_cells_pred_instances",
                        ax = unet_on_ext_plot,
                        **SCATTER_KWARGS)
        unet_on_ext_plot.set_title("UNet performance on\nreal images", fontsize = cfg.TITLE_SIZE)
        unet_on_ext_plot.set_xlabel("n_cells ground truth", fontsize = cfg.AXIS_LABEL_SIZE)
        unet_on_ext_plot.set_ylabel("n_cells predicted", fontsize = cfg.AXIS_LABEL_SIZE)
        unify_axis_limits(unet_on_ext_plot)
        adjust_fontsize_ticklabels(unet_on_ext_plot, cfg.AXIS_LABEL_SIZE)
        

    
    

    # --------------------------------------------------------
    # Layout with GridSpec: A on top, B on bottom
    # --------------------------------------------------------
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL *0.65),
    )
    gs = GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        height_ratios=[1.8,1],  # same ratio as your final code
    )

    a_coords = gs[0, :]
    b_coords = gs[1, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")


    plt.show(fig)

def figure_S7_generation(
    figure_output_dir: str,
    model_output_dir: str,
    figure_data_dir: str,
    h5_path: str,
    unet_base_config: Any,
    instance_seg_config: Any,
    segmenter_class: Any,
    **kwargs
):
    unet_on_sim = pd.read_csv("../scripts/results/testing_val_small_tiles_inst_seg.csv")
    
    unet_on_ext = pd.read_csv("../scripts/results/testing_val_small_external_images_inst_seg.csv")
    
    imageJ_on_sim = pd.read_csv("../scripts/results/testing_val_imageJ_small_inst_seg.csv")
    
    _generate_main_figure(
        unet_on_ext = unet_on_ext,
        unet_on_sim = unet_on_sim,
        imageJ_on_sim = imageJ_on_sim
    )
