import os
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from alloviewer.dev.figures import figure_config as cfg
from alloviewer.dev.figures import figure_utils as utils

from .figure_data_generation import (
    get_validation_data,
    load_or_create_figure_1_image_cache,
    _read_rgb_image,
    crop_square,
    prepare_image,
)


SCATTER_KWARGS = {
    "s": 4,
    "edgecolor": "black",
    "linewidth": 0.3,
}


INSET_SIDE_LENGTH = 128
INSET_LINEWIDTH = 2
INSET_RECT_COLOR = "red"

INSET_WIDTH = "50%"
INSET_HEIGHT = "50%"
INSET_LOCATION = "upper right"
INSET_BORDER_COLOR = "black"
INSET_BORDER_LINEWIDTH = 2


INSET_COORDS = {
    "simulated_image": (250, 250),
    "simulated_segmentation": (250, 250),

    "microscopy_image": (150, 150),
    "microscopy_segmentation": (150, 150),

    "googlepixel_image": (70, 200),
    "googlepixel_segmentation": (70, 200),

    "iphone_image": (60, 650),
    "iphone_segmentation": (60, 650),

    "monochrome_image": (140, 498),
    "monochrome_segmentation": (140, 498),
}

def _segmentation_white_background(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert RGB instance segmentation display image to a white background.

    Assumes the cached segmentation RGB image uses black background.
    Does not add object outlines.
    """
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        return image

    if image.dtype == np.uint8:
        out = image.copy()
        bg = np.all(out == 0, axis=-1)
        out[bg] = 255
        return out

    out = image.astype(np.float32, copy=True)
    bg = np.all(out <= 1e-6, axis=-1)
    out[bg] = 1.0
    return out

def _prepare_for_panel_e(
    image: np.ndarray,
    *,
    is_segmentation: bool,
) -> np.ndarray:
    image = prepare_image(
        image,
        is_segmentation=is_segmentation,
    )

    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)

    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]

    else:
        raise ValueError(f"Unsupported image shape for panel E: {image.shape}")

    if image.dtype == np.uint8:
        if is_segmentation:
            image = _segmentation_white_background(image)
        return image

    image = image.astype(np.float32, copy=False)

    if image.max() > 1.0:
        image = image / image.max()

    image = np.clip(image, 0.0, 1.0)

    if is_segmentation:
        image = _segmentation_white_background(image)

    return image

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
    plot_data = data.copy()

    # Make sure seaborn sees numeric values.
    plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors="coerce")
    plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors="coerce")

    before = len(plot_data)
    plot_data = plot_data.dropna(subset=[x_col, y_col])
    after = len(plot_data)

    print(f"\n{title}")
    print(f"rows before/after dropna: {before} -> {after}")
    print(f"{x_col}: {plot_data[x_col].min()} .. {plot_data[x_col].max()}")
    print(f"{y_col}: {plot_data[y_col].min()} .. {plot_data[y_col].max()}")

    sns.scatterplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        ax=ax,
        **SCATTER_KWARGS,
    )

    print("after seaborn:")
    print("  xlim:", ax.get_xlim())
    print("  ylim:", ax.get_ylim())
    print("  collections:", len(ax.collections))
    for i, coll in enumerate(ax.collections):
        if hasattr(coll, "get_offsets"):
            offsets = coll.get_offsets()
            print(f"  collection {i}: offsets={offsets.shape}")

    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=cfg.AXIS_LABEL_SIZE)

    # Do NOT call utils.unify_axis_limits(ax) for now.
    xmin = float(plot_data[x_col].min())
    xmax = float(plot_data[x_col].max())
    ymin = float(plot_data[y_col].min())
    ymax = float(plot_data[y_col].max())

    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    x = np.array([lo, hi])
    ax.plot(x, x, linestyle="--", color="red", zorder=1)

    # Force scatter collections above identity line.
    for coll in ax.collections:
        coll.set_zorder(3)

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

def _clip_inset_coords(
    image: np.ndarray,
    inset_coords: tuple[int, int],
    inset_side_length: int,
) -> tuple[int, int]:
    h, w = image.shape[:2]
    x, y = inset_coords

    x = int(max(0, min(x, w - inset_side_length)))
    y = int(max(0, min(y, h - inset_side_length)))

    return x, y


def _add_inset_overlay(
    ax: Axes,
    image: np.ndarray,
    inset_coords: tuple[int, int],
    inset_side_length: int,
    title: str | None = None,
) -> None:
    ax.imshow(image)

    x, y = _clip_inset_coords(
        image=image,
        inset_coords=inset_coords,
        inset_side_length=inset_side_length,
    )

    rect = Rectangle(
        (x, y),
        inset_side_length,
        inset_side_length,
        fill=False,
        edgecolor=INSET_RECT_COLOR,
        linewidth=INSET_LINEWIDTH,
    )
    ax.add_patch(rect)

    inset_img = crop_square(
        image,
        x=x,
        y=y,
        length=inset_side_length,
    )

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
    monochrome_image: np.ndarray,
    monochrome_segmentation: np.ndarray,
    inset_coords: Optional[dict[str, tuple[int, int]]] = None,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    if inset_coords is None:
        inset_coords = INSET_COORDS

    fig_sgs = gs.subgridspec(
        2,
        5,
        wspace=0.03,
        hspace=0.12,
    )

    sim_img = _prepare_for_panel_e(simulated_image, is_segmentation=False)
    micro_img = _prepare_for_panel_e(microscopy_image, is_segmentation=False)
    gpixel_img = _prepare_for_panel_e(googlepixel_image, is_segmentation=False)
    iphone_img = _prepare_for_panel_e(iphone_image, is_segmentation=False)
    mono_img = _prepare_for_panel_e(monochrome_image, is_segmentation=False)

    sim_seg = _prepare_for_panel_e(simulated_segmentation, is_segmentation=True)
    micro_seg = _prepare_for_panel_e(microscopy_segmentation, is_segmentation=True)
    gpixel_seg = _prepare_for_panel_e(googlepixel_segmentation, is_segmentation=True)
    iphone_seg = _prepare_for_panel_e(iphone_segmentation, is_segmentation=True)
    mono_seg = _prepare_for_panel_e(monochrome_segmentation, is_segmentation=True)

    image_panels = [
        (sim_img, inset_coords["simulated_image"], "Simulated"),
        (micro_img, inset_coords["microscopy_image"], "Microscopy"),
        (gpixel_img, inset_coords["googlepixel_image"], "Google Pixel"),
        (iphone_img, inset_coords["iphone_image"], "iPhone"),
        (mono_img, inset_coords["monochrome_image"], "Monochrome"),
    ]

    segmentation_panels = [
        (sim_seg, inset_coords["simulated_segmentation"], "Simulated\nsegmentation"),
        (micro_seg, inset_coords["microscopy_segmentation"], "Microscopy\nsegmentation"),
        (gpixel_seg, inset_coords["googlepixel_segmentation"], "Google Pixel\nsegmentation"),
        (iphone_seg, inset_coords["iphone_segmentation"], "iPhone\nsegmentation"),
        (mono_seg, inset_coords["monochrome_segmentation"], "Monochrome\nsegmentation"),
    ]

    for col, (image, coords, title) in enumerate(image_panels):
        panel_ax = fig.add_subplot(fig_sgs[0, col])
        _add_inset_overlay(
            ax=panel_ax,
            image=image,
            inset_coords=coords,
            inset_side_length=INSET_SIDE_LENGTH,
            title=title,
        )

    for col, (image, coords, title) in enumerate(segmentation_panels):
        panel_ax = fig.add_subplot(fig_sgs[1, col])
        _add_inset_overlay(
            ax=panel_ax,
            image=image,
            inset_coords=coords,
            inset_side_length=INSET_SIDE_LENGTH,
            title=title,
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
    monochrome_image: np.ndarray,
    monochrome_segmentation: np.ndarray,
    sketch_dir: str,
    figure_output_dir: str,
    figure_name: str,
    inset_coords: Optional[dict[str, tuple[int, int]]] = None,
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
                "imageJ_roi_count": "NCISP",
            }
        )

        _plot_identity_scatter(
            ax=plot_ax,
            data=plot_df,
            x_col="human_roi_count",
            y_col="predicted_roi_count",
            title="UNet and NCISP performance on\nhuman annotated images",
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
            {"UNet": "UNet", "imageJ": "NCISP"}
        )

        _plot_identity_scatter(
            ax=plot_ax,
            data=plot_df,
            x_col="n_cells_gt_instances",
            y_col="n_cells_pred_instances",
            title="UNet comparison to NCISP on\nsimulated images",
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
            monochrome_image=monochrome_image,
            monochrome_segmentation=monochrome_segmentation,
            inset_coords=inset_coords,
        )

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=3,
        nrows=3,
        figure=fig,
        height_ratios=[1.0, 0.7, 1.05],
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
    plt.close(fig)


def figure_1_generation(
    validation_results_dir: str,
    sketch_dir: str,
    figure_output_dir: str,
    model_output_dir: str,
    **kwargs,
) -> None:
    unet_size = kwargs.get("unet_size", "small")
    comparison_images = kwargs.get("comparison_images", "tiles")

    image_cache_path = os.path.join(
        validation_results_dir,
        "figure_1_image_cache_fullres.npz",
    )

    model_file = kwargs.get(
        "model_file",
        "best_small_tiles_S512_seed187.pth",
    )

    unet_on_sim = get_validation_data(
        results_dir=validation_results_dir,
        mode="testing",
        unet_size=unet_size,
        comparison_images=comparison_images,
        seg_method="inst_seg",
    )
    unet_on_sim = unet_on_sim.sample(n=2000, replace = False)

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
        force_recompute=kwargs.get("redo_analysis", False),
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

        monochrome_image=image_data["monochrome_image"],
        monochrome_segmentation=image_data["monochrome_segmentation"],

        sketch_dir=sketch_dir,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_1",
        inset_coords=kwargs.get("inset_coords", None),
    )
