import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Sequence

from .figure_data_generation import fetch_images

from . import figure_config as cfg
from . import figure_utils as utils

def _grid_of_images(
    fig: Figure,
    parent_gs: SubplotSpec,
    imgs: Sequence[np.ndarray],
    *,
    nrows: int = 2,
    ncols: int = 3,
    wspace: float = 0.02,
    hspace: float = 0.02,
) -> None:
    """
    Place images in a nested subplot grid without axis decorations.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure containing the subplot grid.
    parent_gs : SubplotSpec
        Parent subplot specification for the nested grid.
    imgs : Sequence[numpy.ndarray]
        Images included in the current grid or batch.
    nrows : int
        Number of rows in the image or subplot grid. Default is ``2``.
    ncols : int
        Number of columns in the image or subplot grid. Default is ``3``.
    wspace : float
        Horizontal spacing between subplot columns. Default is ``0.02``.
    hspace : float
        Vertical spacing between subplot rows. Default is ``0.02``.
    """
    sub_gs = GridSpecFromSubplotSpec(
        nrows=nrows,
        ncols=ncols,
        subplot_spec=parent_gs,
        wspace=wspace,
        hspace=hspace,
    )
    count = min(len(imgs), nrows * ncols)
    for i in range(count):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(sub_gs[r, c])
        utils.imshow_no_axes(ax, imgs[i])


def _generate_main_figure(
    figure_output_dir: str = "",
    figure_name: str = "",
    *,
    a_images: Sequence[np.ndarray] = (),
    b_images: Sequence[np.ndarray] = (),
    c_images: Sequence[np.ndarray] = (),
):
    """
    Assemble and save the figure showing examples from the simulated image datasets.

    Parameters
    ----------
    figure_output_dir : str
        Directory in which the figure files are written. Default is ``''``.
    figure_name : str
        Base name used for the output figure files. Default is ``''``.
    a_images : Sequence[numpy.ndarray]
        Images displayed in panel A. Default is ``()``.
    b_images : Sequence[numpy.ndarray]
        Images displayed in panel B. Default is ``()``.
    c_images : Sequence[numpy.ndarray]
        Images displayed in panel C. Default is ``()``.

    Notes
    -----
    Creates Matplotlib subplots and writes the figure outputs selected in this module. Figure input data is supplied by the enclosing preparation workflow.
    """
    a_imgs = list(a_images)
    b_imgs = list(b_images)
    c_imgs = list(c_images)

    ncols = 4
    nrows = 2

    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """
        Draw panel A of Figure S1 on the supplied figure and subplot grid.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure containing the subplot grid.
        ax : Axes
            Matplotlib axis receiving the plot or formatting changes.
        gs : SubplotSpec
            Subplot specification defining the area available to this panel.
        subfigure_label : str or None
            Panel label drawn next to the subplot.

        Notes
        -----
        Uses the panel data and display settings prepared by this module. Adds artists to the supplied Matplotlib figure.
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=-0.4)
        _grid_of_images(fig, gs, a_imgs, nrows=nrows, ncols=ncols)

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """
        Draw panel B of Figure S1 on the supplied figure and subplot grid.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure containing the subplot grid.
        ax : Axes
            Matplotlib axis receiving the plot or formatting changes.
        gs : SubplotSpec
            Subplot specification defining the area available to this panel.
        subfigure_label : str or None
            Panel label drawn next to the subplot.

        Notes
        -----
        Uses the panel data and display settings prepared by this module. Adds artists to the supplied Matplotlib figure.
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=-0.4)
        _grid_of_images(fig, gs, b_imgs, nrows=nrows, ncols=ncols)

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """
        Draw panel C of Figure S1 on the supplied figure and subplot grid.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure containing the subplot grid.
        ax : Axes
            Matplotlib axis receiving the plot or formatting changes.
        gs : SubplotSpec
            Subplot specification defining the area available to this panel.
        subfigure_label : str or None
            Panel label drawn next to the subplot.

        Notes
        -----
        Uses the panel data and display settings prepared by this module. Adds artists to the supplied Matplotlib figure.
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=-0.4)
        _grid_of_images(fig, gs, c_imgs, nrows=nrows, ncols=ncols)

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=3, figure=fig, height_ratios=[1,1,1])
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

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.close()

    return


def figure_S1_generation(
    h5_path: str,
    figure_output_dir: str,
    **kwargs
):
    """
    Prepare the data and generate Figure S1 showing examples from the simulated image datasets.

    Parameters
    ----------
    h5_path : str
        Path to the simulated-image HDF5 dataset.
    figure_output_dir : str
        Directory in which the figure files are written.
    **kwargs : dict
        Accepted by the interface but not read by this implementation.
    """
    n_images = 8
    indices = list(np.arange(1, n_images+1))

    a_images = fetch_images(
        h5_path = os.path.join(h5_path, "pad_resize_val.h5"),
        indices = indices
    )
    b_images = fetch_images(
        h5_path = os.path.join(h5_path, "crop_well_resize_val.h5"),
        indices = indices
    )
    c_images = fetch_images(
        h5_path = os.path.join(h5_path, "tiles_val.h5"),
        indices = indices
    )

    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S1",
        a_images=a_images,
        b_images=b_images,
        c_images=c_images,
    )
