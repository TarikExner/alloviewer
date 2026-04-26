import os
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from alloviewer.dev.figures.figure_data_generation import get_validation_data
from alloviewer.dev.figures import figure_config as cfg
from alloviewer.dev.figures import figure_utils as utils


from .figure_data_generation import (
    load_or_create_figure_1_image_cache,
    _read_rgb_image,
    crop_square,
    _prepare_segmentation_for_display,
    _prepare_image
)


SCATTER_KWARGS = {
    "s": 4,
    "edgecolor": "black",
    "linewidth": 0.3,
    "rasterized": True,
}


INSET_SIDE_LENGTH = 128
INSET_LINEWIDTH = 2
INSET_RECT_COLOR = "red"

INSET_WIDTH = "50%"
INSET_HEIGHT = "50%"
INSET_LOCATION = "upper right"
INSET_BORDER_COLOR = "white"
INSET_BORDER_LINEWIDTH = 2

INSET_COORDS = {
    "simulated_image": (250, 250),
    "simulated_segmentation": (250, 250),
    "microscopy_image": (150, 150),
    "microscopy_segmentation": (150, 150),
    "googlepixel_image": (70, 280),
    "googlepixel_segmentation": (70, 280),
    "iphone_image": (220, 120),
    "iphone_segmentation": (220, 120),
}

def _plot_identity_scatter(
    ax: Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    hue_col: str | None = None,
    legend: bool = False,
    legend_fontsize: int | None = None,
) -> None:
    sns.scatterplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        ax=ax,
        **SCATTER_KWARGS,
    )

    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=cfg.AXIS_LABEL_SIZE)

    utils.unify_axis_limits(ax)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    x = np.array(lims)
    ax.plot(x, x, linestyle="--", color="red")

    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            markerscale=4,
            title="",
            fontsize=legend_fontsize if legend_fontsize is not None else cfg.TITLE_SIZE,
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()


def _add_inset_overlay(
    ax: Axes,
    image: np.ndarray,
    inset_coords: tuple[int, int],
    inset_side_length: int,
    title: str | None = None,
) -> None:
    ax.imshow(image)
    x, y = inset_coords

    rect = Rectangle(
        (x, y),
        inset_side_length,
        inset_side_length,
        fill=False,
        edgecolor=INSET_RECT_COLOR,
        linewidth=INSET_LINEWIDTH,
    )
    ax.add_patch(rect)

    inset_img = crop_square(image, x=x, y=y, length=inset_side_length)

    axins = inset_axes(
        ax,
        width=INSET_WIDTH,
        height=INSET_HEIGHT,
        loc=INSET_LOCATION,
        borderpad=0.0,
    )
    axins.imshow(inset_img)
    axins.set_xticks([])
    axins.set_yticks([])

    for spine in axins.spines.values():
        spine.set_edgecolor(INSET_BORDER_COLOR)
        spine.set_linewidth(INSET_BORDER_LINEWIDTH)

    if title is not None:
        ax.set_title(title, fontsize=cfg.TITLE_SIZE)

    ax.set_xticks([])
    ax.set_yticks([])


def _generate_subfigure_image_grid(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    simulated_image: np.ndarray,
    simulated_segmentation: np.ndarray,
    microscopy_image: np.ndarray,
    microscopy_segmentation: np.ndarray,
    googlepixel_image: np.ndarray,
    googlepixel_segmentation: np.ndarray,
    iphone_image: np.ndarray,
    iphone_segmentation: np.ndarray,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    fig_sgs = gs.subgridspec(2, 4)

    sim_img = _prepare_image(simulated_image, is_segmentation=False)
    micro_img = _prepare_image(microscopy_image, is_segmentation=False)
    gpixel_img = _prepare_image(googlepixel_image, is_segmentation=False)
    iphone_img = _prepare_image(iphone_image, is_segmentation=False)

    sim_seg = _prepare_segmentation_for_display(simulated_segmentation)
    micro_seg = _prepare_segmentation_for_display(microscopy_segmentation)
    gpixel_seg = _prepare_segmentation_for_display(googlepixel_segmentation)
    iphone_seg = _prepare_segmentation_for_display(iphone_segmentation)

    ax_sim_img = fig.add_subplot(fig_sgs[0, 0])
    _add_inset_overlay(
        ax=ax_sim_img,
        image=sim_img,
        inset_coords=INSET_COORDS["simulated_image"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="Simulated",
    )

    ax_micro_img = fig.add_subplot(fig_sgs[0, 1])
    _add_inset_overlay(
        ax=ax_micro_img,
        image=micro_img,
        inset_coords=INSET_COORDS["microscopy_image"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="Microscopy",
    )

    ax_gpixel_img = fig.add_subplot(fig_sgs[0, 2])
    _add_inset_overlay(
        ax=ax_gpixel_img,
        image=gpixel_img,
        inset_coords=INSET_COORDS["googlepixel_image"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="Google Pixel",
    )

    ax_iphone_img = fig.add_subplot(fig_sgs[0, 3])
    _add_inset_overlay(
        ax=ax_iphone_img,
        image=iphone_img,
        inset_coords=INSET_COORDS["iphone_image"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="iPhone",
    )

    # Row 2: segmentation results
    ax_sim_seg = fig.add_subplot(fig_sgs[1, 0])
    _add_inset_overlay(
        ax=ax_sim_seg,
        image=sim_seg,
        inset_coords=INSET_COORDS["simulated_segmentation"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="Simulated\nsegmentation",
    )

    ax_micro_seg = fig.add_subplot(fig_sgs[1, 1])
    _add_inset_overlay(
        ax=ax_micro_seg,
        image=micro_seg,
        inset_coords=INSET_COORDS["microscopy_segmentation"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="Microscopy\nsegmentation",
    )

    ax_gpixel_seg = fig.add_subplot(fig_sgs[1, 2])
    _add_inset_overlay(
        ax=ax_gpixel_seg,
        image=gpixel_seg,
        inset_coords=INSET_COORDS["googlepixel_segmentation"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="Google Pixel\nsegmentation",
    )

    ax_iphone_seg = fig.add_subplot(fig_sgs[1, 3])
    _add_inset_overlay(
        ax=ax_iphone_seg,
        image=iphone_seg,
        inset_coords=INSET_COORDS["iphone_segmentation"],
        inset_side_length=INSET_SIDE_LENGTH,
        title="iPhone\nsegmentation",
    )


def _generate_main_figure(
    unet_on_human: pd.DataFrame,
    unet_on_sim: pd.DataFrame,
    imageJ_on_sim: pd.DataFrame,
    simulated_image: np.ndarray,
    simulated_segmentation: np.ndarray,
    microscopy_image: np.ndarray,
    microscopy_segmentation: np.ndarray,
    googlepixel_image: np.ndarray,
    googlepixel_segmentation: np.ndarray,
    iphone_image: np.ndarray,
    iphone_segmentation: np.ndarray,
    sketch_dir: str,
    figure_output_dir: str,
    figure_name: str,
) -> None:

    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 1)
        sketch_ax = fig.add_subplot(fig_sgs[0])
        utils.prep_image_axis(sketch_ax)

        sketch_path = os.path.join(sketch_dir, "sim_unet_scheme.png")
        img = _read_rgb_image(sketch_path)
        sketch_ax.imshow(img)

    def generate_subfigure_b(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 1)
        plot_ax = fig.add_subplot(fig_sgs[0])

        _plot_identity_scatter(
            ax=plot_ax,
            data=unet_on_sim,
            x_col="n_cells_gt_instances",
            y_col="n_cells_pred_instances",
            title="UNet performance on\nsimulated images",
            xlabel="n_cells ground truth",
            ylabel="n_cells predicted",
        )


    def generate_subfigure_c(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 1)
        plot_ax = fig.add_subplot(fig_sgs[0])

        plot_df = unet_on_human.copy()

        plot_df = plot_df.melt(
            id_vars=["Folder", "image_name", "human_roi_count"],
            value_vars=["unet_roi_count", "imageJ_roi_count"],
            var_name="method",
            value_name="predicted_roi_count",
        )

        plot_df["method"] = plot_df["method"].map(
            {
                "unet_roi_count": "UNet",
                "imageJ_roi_count": "ImageJ",
            }
        )

        _plot_identity_scatter(
            ax=plot_ax,
            data=plot_df,
            x_col="human_roi_count",
            y_col="predicted_roi_count",
            title="UNet and ImageJ performance on\nhuman annotated images",
            xlabel="Human ROI count",
            ylabel="Predicted ROI count",
            hue_col="method",
            legend=True,
            legend_fontsize=cfg.TITLE_SIZE,
        )

    def generate_subfigure_d(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 1)
        plot_ax = fig.add_subplot(fig_sgs[0])

        plot_df = imageJ_on_sim.copy()
        plot_df["dataset_mode"] = plot_df["dataset_mode"].map(
            {"UNet": "UNet", "imageJ": "ImageJ"}
        )

        _plot_identity_scatter(
            ax=plot_ax,
            data=plot_df,
            x_col="n_cells_gt_instances",
            y_col="n_cells_pred_instances",
            title="UNet comparison to imageJ on\nsimulated images",
            xlabel="n_cells ground truth",
            ylabel="n_cells predicted",
            hue_col="dataset_mode",
            legend=True,
            legend_fontsize=cfg.TITLE_SIZE,
        )

    def generate_subfigure_e(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        _generate_subfigure_image_grid(
            fig=fig,
            ax=ax,
            gs=gs,
            subfigure_label=subfigure_label,
            simulated_image=simulated_image,
            simulated_segmentation=simulated_segmentation,
            microscopy_image=microscopy_image,
            microscopy_segmentation=microscopy_segmentation,
            googlepixel_image=googlepixel_image,
            googlepixel_segmentation=googlepixel_segmentation,
            iphone_image=iphone_image,
            iphone_segmentation=iphone_segmentation,
        )

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=3,
        nrows=3,
        figure=fig,
        height_ratios=[1.0, 0.7, 1],
    )

    a_coords = gs[0, :]
    b_coords = gs[1, 0]
    c_coords = gs[1, 1]
    d_coords = gs[1, 2]
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
        
    return

def figure_1_generation(validation_results_dir: str,
                        sketch_dir: str,
                        figure_output_dir: str,
                        model_output_dir: str,
                        figure_data_dir,

                        **kwargs) -> None:

    unet_size = kwargs.get("unet_size", "small")
    comparison_images = kwargs.get("comparison_images", "tiles")

    image_cache_path = os.path.join(figure_data_dir, "figure_1_image_cache.npz")
    model_file = kwargs.get("model_file", "best_small_tiles_S512_seed187.pth")

    unet_on_sim = get_validation_data(
        results_dir=validation_results_dir,
        mode="testing",
        unet_size=unet_size,
        comparison_images=comparison_images,
        seg_method="inst_seg",
    )

    unet_on_human = get_validation_data(
        results_dir=validation_results_dir,
        mode="human",
    )

    imageJ_on_sim = get_validation_data(
        results_dir=validation_results_dir,
        mode="imageJ",
    )

    image_data = load_or_create_figure_1_image_cache(
        cache_path=image_cache_path,
        model_dir=model_output_dir,
        model_file=model_file,
        force_recompute=False,
    )

    _generate_main_figure(
        unet_on_human=unet_on_human,
        unet_on_sim=unet_on_sim,
        imageJ_on_sim=imageJ_on_sim,

        simulated_image=image_data["simulated_image"],
        simulated_segmentation=image_data["simulated_segmentation"],
        microscopy_image=image_data["microscopy_image"],
        microscopy_segmentation=image_data["microscopy_segmentation"],
        googlepixel_image=image_data["googlepixel_image"],
        googlepixel_segmentation=image_data["googlepixel_segmentation"],
        iphone_image=image_data["iphone_image"],
        iphone_segmentation=image_data["iphone_segmentation"],

        sketch_dir=sketch_dir,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_1",
    )

