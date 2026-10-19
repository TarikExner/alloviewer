import os
from typing import Any

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import (
    _read_rgb_image,
    make_simulation_parameter_mosaic,
    SIMULATION_LABEL_Y_OFFSET
)


def _generate_main_figure(
    sketch_dir: str,
    figure_output_dir: str,
    figure_name: str,
    simulation_showcase: np.ndarray,
    simulation_tile_info: list[dict[str, Any]],
) -> None:
    """
    Assemble and save the figure showing the simulation workflow and parameter examples.

    Parameters
    ----------
    sketch_dir : str
        Directory containing workflow and schematic illustrations.
    figure_output_dir : str
        Directory in which the figure files are written.
    figure_name : str
        Base name used for the output figure files.
    simulation_showcase : numpy.ndarray
        Prepared simulation examples used in the figure panel.
    simulation_tile_info : list[dict[str, Any]]
        Metadata identifying the simulation image tiles displayed in the figure.

    Notes
    -----
    Creates Matplotlib subplots and writes the figure outputs selected in this module. Figure input data is supplied by the enclosing preparation workflow.
    """
    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        """
        Draw panel A of Figure 1 on the supplied figure and subplot grid.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure containing the subplot grid.
        ax : Axes
            Matplotlib axis receiving the plot or formatting changes.
        gs : SubplotSpec
            Subplot specification defining the area available to this panel.
        subfigure_label : str
            Panel label drawn next to the subplot.

        Notes
        -----
        Uses the panel data and display settings prepared by this module. Adds artists to the supplied Matplotlib figure.
        """
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
        """
        Draw panel B of Figure 1 on the supplied figure and subplot grid.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure containing the subplot grid.
        ax : Axes
            Matplotlib axis receiving the plot or formatting changes.
        gs : SubplotSpec
            Subplot specification defining the area available to this panel.
        subfigure_label : str
            Panel label drawn next to the subplot.

        Notes
        -----
        Uses the panel data and display settings prepared by this module. Adds artists to the supplied Matplotlib figure.
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 1)
        outer_ax = fig.add_subplot(fig_sgs[0])
        outer_ax.axis("off")
        
        panel_ax = outer_ax.inset_axes(
            [0.10, 0.08, 0.80, 0.84]
        )
        
        panel_ax.imshow(simulation_showcase)
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])

        crop_size = simulation_showcase.shape[0] // 4

        for info in simulation_tile_info:
            row = info["row"]
            col = info["col"]
            label = info["label"]

            x = col * crop_size + 5
            y = row * crop_size + SIMULATION_LABEL_Y_OFFSET

            panel_ax.text(
                x,
                y,
                label,
                color="white",
                fontsize=max(cfg.AXIS_LABEL_SIZE - 2, 7),
                fontweight="bold",
                ha="left",
                va="top",
                bbox=dict(
                    facecolor="black",
                    alpha=0.60,
                    edgecolor="none",
                    boxstyle="round,pad=0.20",
                ),
            )

        for r in range(1, 4):
            panel_ax.axhline(
                r * crop_size - 0.5,
                color="white",
                linewidth=0.7,
                alpha=0.45,
            )

        for c in range(1, 4):
            panel_ax.axvline(
                c * crop_size - 0.5,
                color="white",
                linewidth=0.7,
                alpha=0.45,
            )

    fig = plt.figure(
        layout="constrained",
        figsize=(
            cfg.FIGURE_WIDTH_FULL,
            cfg.FIGURE_HEIGHT_FULL,
        ),
    )

    gs = GridSpec(
        ncols=1,
        nrows=2,
        figure=fig,
        height_ratios=[1, 2.2],
    )

    a_coords = gs[0, 0]
    b_coords = gs[1, 0]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.close()

    return


def figure_1_generation(
    sketch_dir: str,
    figure_output_dir: str,
    **kwargs
) -> None:

    """
    Prepare the data and generate Figure 1 showing the simulation workflow and parameter examples.

    Parameters
    ----------
    sketch_dir : str
        Directory containing workflow and schematic illustrations.
    figure_output_dir : str
        Directory in which the figure files are written.
    **kwargs : dict
        Accepted by the interface but not read by this implementation.
    """
    simulation_showcase, simulation_tile_info = make_simulation_parameter_mosaic()

    _generate_main_figure(
        sketch_dir=sketch_dir,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_1",
        simulation_showcase=simulation_showcase,
        simulation_tile_info=simulation_tile_info,
    )
