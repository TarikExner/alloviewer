import os
from typing import Any
import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from alloviewer.image_analysis.config import UNET_CONFIG
from alloviewer.image_analysis.segmenter import SegmenterUNet

from .figure_data_generation import (
    fetch_image_with_targets,
    segment_image_unet,
)

from . import figure_config as cfg
from . import figure_utils as utils


def _generate_main_figure(
    data: pd.DataFrame,
    img: np.ndarray,
    tgts: np.ndarray,
    preds: dict[str, np.ndarray],
    ij: np.ndarray,
    figure_output_dir: str = "",
    figure_name: str = "",
):
    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1, 4)

        plot_df = data.copy()

        # Display name.
        plot_df["dataset_mode"] = plot_df["dataset_mode"].map(
            {
                "UNet": "UNet",
                "imageJ": "NCISP",
            }
        ).fillna(plot_df["dataset_mode"])

        plot_params = {
            "data": plot_df,
            "x": "dataset_mode",
        }

        mask_iou_plot = fig.add_subplot(fig_sgs[0])
        sns.violinplot(**plot_params, y="mask_iou", ax=mask_iou_plot)
        mask_iou_plot.set_xlabel("")
        mask_iou_plot.set_ylabel("")
        mask_iou_plot.set_title("Mask Jaccard Score", fontsize=cfg.TITLE_SIZE)
        mask_iou_plot.set_ylim(-0.1, 1.15)

        boundary_plot = fig.add_subplot(fig_sgs[1])
        sns.violinplot(**plot_params, y="boundary_f1", ax=boundary_plot)
        boundary_plot.set_xlabel("")
        boundary_plot.set_ylabel("")
        boundary_plot.set_title("Boundary F1 Score", fontsize=cfg.TITLE_SIZE)
        boundary_plot.set_ylim(-0.1, 1.15)

        center_plot = fig.add_subplot(fig_sgs[2])
        sns.violinplot(**plot_params, y="center_f1", ax=center_plot)
        center_plot.set_xlabel("")
        center_plot.set_ylabel("")
        center_plot.set_title("Center F1 Score", fontsize=cfg.TITLE_SIZE)
        center_plot.set_ylim(-0.1, 1.15)

        energy_plot = fig.add_subplot(fig_sgs[3])
        sns.violinplot(**plot_params, y="energy_ssim", ax=energy_plot)
        energy_plot.set_xlabel("")
        energy_plot.set_ylabel("")
        energy_plot.set_title("Energy SSIM score", fontsize=cfg.TITLE_SIZE)
        energy_plot.set_ylim(-0.1, 1.15)

    def generate_subfigure_b(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        # Keep tile-level visual display.
        fig_sgs = gs.subgridspec(3, 5, wspace=0.04, hspace=0.10)

        titles = ["Image", "Cell mask", "Cell boundary", "Cell centers", "Energy"]
        pred_keys = ["cell", "bound", "center", "energy"]

        inset_centers = (450, 150)
        inset_box_px = 96
        inset_zoom = 3.0

        H, W = img.shape[:2]
        default_center = (H // 2, W // 2)

        if inset_centers is None:
            centers = [default_center] * 5
        elif isinstance(inset_centers, tuple) and len(inset_centers) == 2:
            centers = [tuple(map(int, inset_centers))] * 5
        else:
            centers = [tuple(map(int, c)) for c in inset_centers]

        def _add_inset_top_right(ax_in, arr, center, is_target: bool):
            cy, cx = center
            half = inset_box_px // 2

            y0 = int(np.clip(cy - half, 0, arr.shape[0] - 1))
            x0 = int(np.clip(cx - half, 0, arr.shape[1] - 1))
            y1 = int(np.clip(y0 + inset_box_px, 1, arr.shape[0]))
            x1 = int(np.clip(x0 + inset_box_px, 1, arr.shape[1]))

            axins = zoomed_inset_axes(
                ax_in,
                zoom=inset_zoom,
                loc="upper right",
                borderpad=0.2,
            )

            if is_target:
                axins.imshow(arr, cmap="jet", interpolation="nearest")
            else:
                axins.imshow(arr, interpolation="nearest")

            axins.set_xlim(x0, x1)
            axins.set_ylim(y1, y0)
            axins.set_xticks([])
            axins.set_yticks([])

            for sp in axins.spines.values():
                sp.set_linewidth(0.8)
                sp.set_edgecolor("white")

        # ---- row 0: image + GT masks ----
        ax_img_gt = fig.add_subplot(fig_sgs[0, 0])
        ax_img_gt.imshow(img)
        ax_img_gt.set_title(titles[0], fontsize=9)
        ax_img_gt.set_xticks([])
        ax_img_gt.set_yticks([])
        for sp in ax_img_gt.spines.values():
            sp.set_visible(False)
        _add_inset_top_right(ax_img_gt, img, centers[0], is_target=False)

        for c in range(4):
            ax_gt = fig.add_subplot(fig_sgs[0, c + 1])
            ax_gt.imshow(tgts[..., c], cmap="jet", interpolation="nearest")
            ax_gt.set_title(titles[c + 1], fontsize=9)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            for sp in ax_gt.spines.values():
                sp.set_visible(False)
            _add_inset_top_right(ax_gt, tgts[..., c], centers[c + 1], is_target=True)

        ax_img_gt.set_ylabel("ground truth", fontsize=8)

        # ---- row 1: image + U-Net predictions ----
        ax_img_pred = fig.add_subplot(fig_sgs[1, 0])
        ax_img_pred.imshow(img)
        ax_img_pred.set_title(titles[0], fontsize=9)
        ax_img_pred.set_xticks([])
        ax_img_pred.set_yticks([])
        for sp in ax_img_pred.spines.values():
            sp.set_visible(False)
        _add_inset_top_right(ax_img_pred, img, centers[0], is_target=False)

        for c, key in enumerate(pred_keys):
            arr_pred = preds[key]
            ax_pred = fig.add_subplot(fig_sgs[1, c + 1])
            ax_pred.imshow(arr_pred, cmap="jet", interpolation="nearest")
            ax_pred.set_title(f"{titles[c + 1]} (pred)", fontsize=9)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            for sp in ax_pred.spines.values():
                sp.set_visible(False)
            _add_inset_top_right(ax_pred, arr_pred, centers[c + 1], is_target=True)

        ax_img_pred.set_ylabel("U-Net", fontsize=8)

        # ---- row 2: image + ImageJ / NCISP maps ----
        ax_img_ij = fig.add_subplot(fig_sgs[2, 0])
        ax_img_ij.imshow(img)
        ax_img_ij.set_title(titles[0], fontsize=9)
        ax_img_ij.set_xticks([])
        ax_img_ij.set_yticks([])
        for sp in ax_img_ij.spines.values():
            sp.set_visible(False)
        _add_inset_top_right(ax_img_ij, img, centers[0], is_target=False)

        for c in range(4):
            arr_ij = ij[..., c]
            ax_ij = fig.add_subplot(fig_sgs[2, c + 1])
            ax_ij.imshow(arr_ij, cmap="jet", interpolation="nearest")
            ax_ij.set_title(f"{titles[c + 1]} (NCISP)", fontsize=9)
            ax_ij.set_xticks([])
            ax_ij.set_yticks([])
            for sp in ax_ij.spines.values():
                sp.set_visible(False)
            _add_inset_top_right(ax_ij, arr_ij, centers[c + 1], is_target=True)

        ax_img_ij.set_ylabel("NCISP", fontsize=8)

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL * 0.7),
    )

    gs = GridSpec(
        ncols=5,
        nrows=2,
        figure=fig,
        height_ratios=[1, 2],
    )

    a_coords = gs[0, :]
    b_coords = gs[1, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_S7_generation(
    figure_output_dir: str,
    model_output_dir: str,
    validation_results_dir: str,
    figure_data_dir: str,
    h5_path: str,
    **kwargs,
):
    unet_base_config = copy.deepcopy(UNET_CONFIG)
    segmenter_class = SegmenterUNet

    unet_size = kwargs.get("unet_size", "small")
    seg_method = kwargs.get("seg_method", "inst_seg")

    imagewide_csv = os.path.join(
        validation_results_dir,
        f"testing_val_imageJ_{unet_size}_{seg_method}.csv",
    )

    if not os.path.isfile(imagewide_csv):
        raise FileNotFoundError(
            f"Image-wide validation CSV not found: {imagewide_csv}\n"
            "Run run_fullres_unet_vs_imagej_validation(..., force=True) first."
        )

    data = pd.read_csv(imagewide_csv)

    # Keep tile display.
    img_index = kwargs.get("img_index", 12)
    tile_idx = kwargs.get("tile_idx", 7)

    gt_path = os.path.join(h5_path, "fullres_ground_truth.h5")
    gt_rgb, gt_tgts = fetch_image_with_targets(
        h5_path=gt_path,
        index=img_index,
        tile_idx=tile_idx,
        image_key="imgs",
        target_key="tgts",
        resize_to=(512, 512),
        target_resize_to=None,
        return_channel_last=True,
    )

    ij_path = os.path.join(h5_path, "fullres_imageJ.h5")
    _, ij_tgts = fetch_image_with_targets(
        h5_path=ij_path,
        index=img_index,
        tile_idx=tile_idx,
        image_key="imgs",
        target_key="tgts",
        resize_to=(512, 512),
        target_resize_to=None,
        return_channel_last=True,
    )

    unet_base_config = unet_base_config.copy()
    unet_base_config["normalize"] = True

    preds = segment_image_unet(
        models_dir=model_output_dir,
        img=gt_rgb,
        unet_base_config=unet_base_config,
        segmenter_class=segmenter_class,
    )

    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S7",
        data=data,
        img=gt_rgb,
        tgts=gt_tgts,
        preds=preds,
        ij=ij_tgts,
    )

