import os
import copy
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .figure_data_generation import get_validation_data, generate_unet_comparison

from alloviewer.image_analysis.config import UNET_CONFIG
from alloviewer.image_analysis.segmenter import SegmenterUNet

from . import figure_config as cfg
from . import figure_utils as utils


DATASET_ORDER = [
    "pad_resize",
    "crop_well_resize",
    "tiles",
]

DATASET_DISPLAY_MAP = {
    "pad_resize": "Pad + resize",
    "crop_well_resize": "Crop well + resize",
    "tiles": "Tiling",
}

UNET_ORDER = [
    "small",
    "medium",
    "large",
]

UNET_DISPLAY_MAP = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}

PROB_HEADS = [
    "cell",
    "bound",
    "center",
    "energy",
]

PROB_HEAD_DISPLAY_MAP = {
    "cell": "Cell",
    "bound": "Boundary",
    "center": "Center",
    "energy": "Energy",
}


def _prepare_training_plot_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().reset_index(drop=True)

    if "dataset_mode" not in data.columns:
        raise ValueError("Expected column 'dataset_mode' in training validation data.")

    if "unet_mode" not in data.columns:
        raise ValueError("Expected column 'unet_mode' in training validation data.")

    data["dataset_mode"] = data["dataset_mode"].replace(
        {
            "tiling": "tiles",
            "tile": "tiles",
        }
    )

    data["dataset_display"] = (
        data["dataset_mode"]
        .map(DATASET_DISPLAY_MAP)
        .fillna(data["dataset_mode"])
    )

    data["dataset_display"] = pd.Categorical(
        data["dataset_display"],
        categories=[DATASET_DISPLAY_MAP[x] for x in DATASET_ORDER],
        ordered=True,
    )

    data["unet_display"] = (
        data["unet_mode"]
        .map(UNET_DISPLAY_MAP)
        .fillna(data["unet_mode"])
    )

    data["unet_display"] = pd.Categorical(
        data["unet_display"],
        categories=[UNET_DISPLAY_MAP[x] for x in UNET_ORDER],
        ordered=True,
    )

    return data


def _get_prob_for_display(prob: Any) -> np.ndarray:
    prob = np.asarray(prob)

    if prob.ndim == 2:
        return prob

    if prob.ndim == 3:
        return prob[0]

    raise ValueError(f"Expected 2D or 3D probability map, got shape {prob.shape}.")


def _format_unet_key(unet_mode: str) -> str:
    if unet_mode == "medium":
        return "med"
    return unet_mode


def _plot_metric_boxplot(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    use_violin: bool = False,
) -> None:
    plot_params = {
        "data": data,
        "x": "dataset_display",
        "order": [DATASET_DISPLAY_MAP[x] for x in DATASET_ORDER],
        "hue": "unet_display",
        "hue_order": [UNET_DISPLAY_MAP[x] for x in UNET_ORDER],
    }

    if use_violin:
        sns.violinplot(
            **plot_params,
            y=metric,
            ax=ax,
            inner=None,
        )
    else:
        sns.boxplot(
            **plot_params,
            y=metric,
            ax=ax,
            whis=(0, 100),
        )

    ax.legend(
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        title="UNet size",
        fontsize=cfg.AXIS_LABEL_SIZE,
        title_fontsize=cfg.AXIS_LABEL_SIZE,
        ncols=3,
        frameon=False,
    )

    ax.set_xlabel("")
    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=cfg.AXIS_LABEL_SIZE)
    ax.tick_params(**cfg.TICKPARAMS_PARAMS)


def _generate_main_figure(
    data: pd.DataFrame,
    res: dict,
    figure_output_dir: str = "",
    figure_name: str = "",
) -> None:
    data = _prepare_training_plot_data(data)

    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 1)
        plot_ax = fig.add_subplot(fig_sgs[0])

        _plot_metric_boxplot(
            ax=plot_ax,
            data=data,
            metric="mask_iou",
            title="Mask prediction",
            ylabel="Jaccard score",
        )

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

        _plot_metric_boxplot(
            ax=plot_ax,
            data=data,
            metric="boundary_f1",
            title="Boundary prediction",
            ylabel="F1 score",
            use_violin=True,
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

        _plot_metric_boxplot(
            ax=plot_ax,
            data=data,
            metric="center_f1",
            title="Center prediction",
            ylabel="F1 score",
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

        _plot_metric_boxplot(
            ax=plot_ax,
            data=data,
            metric="energy_ssim",
            title="Energy prediction",
            ylabel="SSIM score",
        )

    def generate_subfigure_e(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(
            nrows=3,
            ncols=4,
            wspace=0.02,
            hspace=0.08,
        )

        for i, unet_mode in enumerate(UNET_ORDER):
            res_key = _format_unet_key(unet_mode)

            if res_key not in res:
                raise KeyError(
                    f"Missing key '{res_key}' in UNet comparison result. "
                    f"Available keys: {list(res.keys())}"
                )

            for j, head in enumerate(PROB_HEADS):
                ax_img = fig.add_subplot(fig_sgs[i, j])

                if head not in res[res_key]:
                    raise KeyError(
                        f"Missing probability head '{head}' in res['{res_key}']. "
                        f"Available keys: {list(res[res_key].keys())}"
                    )

                prob = _get_prob_for_display(res[res_key][head])

                ax_img.imshow(
                    prob,
                    vmin=0.0,
                    vmax=1.0,
                )

                ax_img.set_xticks([])
                ax_img.set_yticks([])

                for spine in ax_img.spines.values():
                    spine.set_visible(False)

                if i == 0:
                    ax_img.set_title(
                        PROB_HEAD_DISPLAY_MAP[head],
                        fontsize=cfg.TITLE_SIZE,
                        pad=4,
                    )

                if j == 0:
                    ax_img.set_ylabel(
                        UNET_DISPLAY_MAP[unet_mode],
                        fontsize=cfg.TITLE_SIZE,
                        rotation=90,
                        color="black",
                    )

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=6,
        nrows=3,
        figure=fig,
        height_ratios=[1, 1, 2.5],
    )

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
    plt.close(fig)


def figure_S6_generation(
    figure_output_dir: str,
    model_output_dir: str,
    ext_images_dir: str,
    validation_results_dir: str,
    figure_data_dir: str,
    **kwargs,
) -> None:
    unet_base_config = copy.deepcopy(UNET_CONFIG)
    segmenter_class = SegmenterUNet

    data = get_validation_data(
        results_dir=validation_results_dir,
        mode="training",
    ).reset_index(drop=True)

    res = generate_unet_comparison(
        models_dir=model_output_dir,
        ext_images_dir=ext_images_dir,
        unet_base_config=unet_base_config,
        segmenter_class=segmenter_class,
        output_dir=figure_data_dir,
        output_filename="unet_segmentation_comparison",
        redo_analysis=kwargs.get("redo_analysis", False),
    )

    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S6",
        data=data,
        res=res,
    )
