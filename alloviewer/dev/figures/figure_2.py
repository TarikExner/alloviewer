
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, feature

import seaborn as sns
from typing import Any

import cv2

from figures.figure_data_generation import get_validation_data, generate_unet_comparison

from figures import figure_config as cfg
from figures import figure_utils as utils

import pickle


def _generate_main_figure(
    results_df: pd.DataFrame,
    plate: Any,
    scoring: pd.DataFrame,
    figure_output_dir: str = "",
    figure_name: str = "Figure_2",
) -> None:
    """
    Create Figure 2:
      - Subfigure A: 2x2 grid: negative / positive control images + RG scatter.
      - Subfigure B: image for well C4 + RG scatter plots colored by label.
      - Subfigure C: placeholder (second row, full width).
    """

    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        # 2x2 grid:
        # [0,0] negative control image
        # [0,1] negative control RG scatter
        # [1,0] positive control image
        # [1,1] positive control RG scatter
        fig_sgs = gs.subgridspec(2, 2)

        neg_wells = plate.get("negative")
        pos_wells = plate.get("positive")

        if len(neg_wells) == 0:
            raise ValueError("No wells with role 'negative' found in plate.")
        if len(pos_wells) == 0:
            raise ValueError("No wells with role 'positive' found in plate.")

        neg_well = neg_wells[0]
        pos_well = pos_wells[0]

        # You can change which calibrator/classifier to show here if needed
        calib_for_A = "PCNCGaussianRGCalibrator"
        df_neg = results_df[
            (results_df["well_id"] == neg_well.well_id)
            & (results_df["calibrator"] == calib_for_A)
        ]
        df_pos = results_df[
            (results_df["well_id"] == pos_well.well_id)
            & (results_df["calibrator"] == calib_for_A)
        ]

        # --- Negative control image ---
        ax_img_neg = fig.add_subplot(fig_sgs[0, 0])
        ax_img_neg.imshow(np.rot90(neg_well.image.transpose((1,2,0))))
        ax_img_neg.set_title(
            f"Negative control", fontsize=cfg.TITLE_SIZE
        )
        ax_img_neg.set_xticks([])
        ax_img_neg.set_yticks([])

        # --- Negative control RG scatter ---
        ax_scatter_neg = fig.add_subplot(fig_sgs[0, 1])
        sns.scatterplot(
            data=df_neg,
            x="mean_r",
            y="mean_g",
            ax=ax_scatter_neg,
            **cfg.SCATTER_KWARGS
        )
        ax_scatter_neg.set_xlabel("Mean red intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_neg.set_ylabel("Mean green intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_neg.set_title("Negative control RG", fontsize=cfg.TITLE_SIZE)
        utils.adjust_fontsize_ticklabels(ax_scatter_neg, cfg.AXIS_LABEL_SIZE)
        ax_scatter_neg.set_xlim(-0.1, 1.1)
        ax_scatter_neg.set_ylim(-0.1, 1.1)

        # --- Positive control image ---
        ax_img_pos = fig.add_subplot(fig_sgs[1, 0])
        ax_img_pos.imshow(np.rot90(pos_well.image.transpose((1,2,0))))
        ax_img_pos.set_title(
            f"Positive control", fontsize=cfg.TITLE_SIZE
        )
        ax_img_pos.set_xticks([])
        ax_img_pos.set_yticks([])

        # --- Positive control RG scatter ---
        ax_scatter_pos = fig.add_subplot(fig_sgs[1, 1])
        sns.scatterplot(
            data=df_pos,
            x="mean_r",
            y="mean_g",
            ax=ax_scatter_pos,
            **cfg.SCATTER_KWARGS,
        )
        ax_scatter_pos.set_xlabel("Mean red intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_pos.set_ylabel("Mean green intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_pos.set_title("Positive control RG", fontsize=cfg.TITLE_SIZE)
        utils.adjust_fontsize_ticklabels(ax_scatter_pos, cfg.AXIS_LABEL_SIZE)
        ax_scatter_pos.set_xlim(-0.1, 1.1)
        ax_scatter_pos.set_ylim(-0.1, 1.1)

    def generate_subfigure_b(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        """
        Subfigure B:
          - 2x2 grid:
                [0,0] image of well C4
                [0,1] RG scatter for calibrator 1 (colored by label)
                [1,0] RG scatter for calibrator 2 (colored by label)
                [1,1] RG scatter for calibrator 3 (colored by label)
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(2, 2)

        # Get well C4
        well_id_c4 = "C4"
        if well_id_c4 not in plate.wells:
            raise ValueError(f"Well '{well_id_c4}' not found in plate.")
        well_c4 = plate.wells[well_id_c4]

        df_c4 = results_df[results_df["well_id"] == well_id_c4].copy()
        if df_c4.empty:
            raise ValueError(f"No ROI results for well '{well_id_c4}' in results_df.")

        calibrators = [
            "PCNCGaussianRGCalibrator",
            "PCNCMeanCalibrator",
            "PCNCMedianCalibrator",
        ]

        # --- Image axis (top left) ---
        ax_img_c4 = fig.add_subplot(fig_sgs[0, 0])
        ax_img_c4.imshow(np.rot90(well_c4.image.transpose((1,2,0))))
        ax_img_c4.set_title(f"Well {well_id_c4} image", fontsize=cfg.TITLE_SIZE)
        ax_img_c4.set_xticks([])
        ax_img_c4.set_yticks([])

        # Axes for the three scatter plots
        ax_scatter_1 = fig.add_subplot(fig_sgs[0, 1])
        ax_scatter_2 = fig.add_subplot(fig_sgs[1, 0])
        ax_scatter_3 = fig.add_subplot(fig_sgs[1, 1])
        scatter_axes = [ax_scatter_1, ax_scatter_2, ax_scatter_3]
    

        legend_handles = None
        legend_labels = None

        for calib, ax_sc in zip(calibrators, scatter_axes):
            df_calib = df_c4[df_c4["calibrator"] == calib]
            if df_calib.empty:
                ax_sc.axis("off")
                continue

            sns.scatterplot(
                data=df_calib,
                x="mean_r",
                y="mean_g",
                hue="label",
                ax=ax_sc,
                **cfg.SCATTER_KWARGS,
            )
            ax_sc.set_xlabel("Mean red intensity", fontsize=cfg.AXIS_LABEL_SIZE)
            ax_sc.set_ylabel("Mean green intensity", fontsize=cfg.AXIS_LABEL_SIZE)
            ax_sc.set_title(calib, fontsize=cfg.TITLE_SIZE)
            utils.adjust_fontsize_ticklabels(ax_sc, cfg.AXIS_LABEL_SIZE)
            # Save legend from the first non-empty axis
            if legend_handles is None:
                legend_handles, legend_labels = ax_sc.get_legend_handles_labels()
            # Remove individual legends
            ax_sc.legend(markerscale = 4,
                         fontsize = cfg.AXIS_LABEL_SIZE,
                         title = "",
                         loc = "lower left")
            ax_sc.set_xlim(-0.1,1.1)
            ax_sc.set_ylim(-0.1,1.1)



    def generate_subfigure_c(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        """
        Placeholder for subfigure C.
        Uses the full width of the second row.
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1,1)

        scoring_plot = fig.add_subplot(fig_sgs[0])
        sns.barplot(data = scoring,
                    x = "class",
                    y = "accuracy",
                    hue = "method",
                    ax = scoring_plot,
                    edgecolor = "black",
                    linewidth = 1)
        scoring_plot.set_ylabel("accuracy score", fontsize = cfg.AXIS_LABEL_SIZE)
        scoring_plot.set_title("Accuracy per class label\namong annotators", fontsize = cfg.TITLE_SIZE)
        scoring_plot.set_xlabel("")
        handles, labels = scoring_plot.get_legend_handles_labels()
        scoring_plot.legend(handles, labels,
                            bbox_to_anchor = (1.05, 0.5),
                            loc = "center left",
                            title = "annotated by",
                            fontsize = cfg.TITLE_SIZE)
        utils.adjust_fontsize_ticklabels(scoring_plot, cfg.AXIS_LABEL_SIZE)

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL*0.75),
    )
    gs = GridSpec(
        ncols=4,
        nrows=2,
        figure=fig,
        height_ratios=[1, 1.3],
    )

    a_coords = gs[0, 0:2]   # row 0, left half
    b_coords = gs[0, 2:4]   # row 0, right half
    c_coords = gs[1, :]     # row 1, full width

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")

    save = False
    if save:
    
        os.makedirs(figure_output_dir, exist_ok=True)
        pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
        png_path = os.path.join(figure_output_dir, f"{figure_name}.png")
    
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    plt.show()

def figure_2_generation(
    figure_output_dir: str,
    figure_data_dir: str,
    **kwargs,
) -> None:

    data_path = os.path.join(figure_data_dir, "calibration_data.dict")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    plate = data["plate"]
    results_df = data["res_df"]

    scoring_df = pd.read_csv("./figure_data/scoring_comparison_annotators_unet.csv", index_col =False)

    _generate_main_figure(
        results_df=results_df,
        plate=plate,
        scoring=scoring_df,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_2",
    )
