import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Sequence, Tuple, List

from .figure_data_generation import get_loss_data

from . import figure_config as cfg
from . import figure_utils as utils


# ---------- helpers shared by A and B ----------

def _dedup_legend(ax: Axes) -> Tuple[List, List]:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    return h2, l2


def _plot_lines(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    show_legend: bool,
    inset_frac: float = 0.25,   # fraction of last epochs to zoom
    inset_size: str = "35%",    # width/height for inset
) -> Tuple[List, List]:
    """
    One line per (unet_mode, seed, target). Legend grouped by unet_mode.
    Inset is placed in the TOP RIGHT corner.
    """
    if data.empty or metric not in data.columns:
        ax.set_title(f"{title} (no data)")
        ax.set_xticks([])
        ax.set_yticks([])
        return [], []

    data = data.sort_values(["unet_mode", "seed", "target", "epoch"])

    # main lines
    for (unet_mode, seed, target), grp in data.groupby(
        ["unet_mode", "seed", "target"], dropna=False
    ):
        ax.plot(grp["epoch"], grp[metric], label=unet_mode)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric.replace("_", " "))

    # inset (top right)
    try:
        max_ep = int(data["epoch"].max())
        min_ep = int(data["epoch"].min())
        span = max(1, max_ep - min_ep + 1)
        tail = max(min_ep, max_ep - int(span * inset_frac) + 1)

        axins = inset_axes(
            ax, width=inset_size, height=inset_size, loc="upper right", borderpad=0.8
        )
        for (unet_mode, seed, target), grp in data.groupby(
            ["unet_mode", "seed", "target"], dropna=False
        ):
            tail_grp = grp[grp["epoch"] >= tail]
            axins.plot(tail_grp["epoch"], tail_grp[metric], label=unet_mode)
        axins.set_xlim(tail, max_ep)
        axins.tick_params(axis="both", labelsize=8)
        axins.set_title("", pad=0)
    except Exception:
        pass

    if show_legend:
        h, l = _dedup_legend(ax)
        if h:
            leg = ax.legend(
                h, l, title="UNET mode", loc="upper right", frameon=False, ncol=1
            )
            if leg and leg.get_title():
                leg.get_title().set_fontsize(10)
        return h, l
    return [], []


# ---------- main figure ----------

def _generate_main_figure(
    figure_output_dir: str = "",
    figure_name: str = "",
    *,
    data: pd.DataFrame,
    inset_frac: float = 0.5,
):

    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """
        Subfigure A:
        - 1x3 plots with shared y-axis
        - Right-side legend column (ncol=1)
        - Only the left plot shows y-label/ticks
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=-0.4)

        if "split" not in data.columns:
            raise ValueError("Expected column 'split' in the dataframe.")

        val_df = data[data["split"] == "val"].copy()

        modes = ("pad_resize", "crop_well_resize", "tiles")
        metric = "loss_unweighted"

        # subgrid: 1 row, 3 plot columns + 1 legend column
        sub = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=4,
            subplot_spec=gs,
            width_ratios=[1.0, 1.0, 1.0, 0.25],
            wspace=0,
            hspace=0.0,
        )

        # compute common y-lims if possible
        y_min, y_max = None, None
        vals = []
        for m in modes:
            dd = val_df[val_df["mode"] == m]
            if metric in dd.columns:
                vals.append(dd[metric].to_numpy())
        if vals:
            vv = np.concatenate(vals)
            if vv.size > 0:
                y_min = float(np.nanmin(vv))
                y_max = float(np.nanmax(vv))
                pad = 0.05 * (y_max - y_min + 1e-12)
                y_min -= pad
                y_max += pad

        axes: List[Axes] = []

        # first axis
        ax0 = fig.add_subplot(sub[0, 0])
        axes.append(ax0)
        d0 = val_df[val_df["mode"] == modes[0]]
        _plot_lines(ax0, d0, metric, modes[0], show_legend=False, inset_frac=inset_frac)

        # remaining axes, sharey with ax0
        for j, mode in enumerate(modes[1:], start=1):
            axi = fig.add_subplot(sub[0, j], sharey=ax0)
            axes.append(axi)
            dj = val_df[val_df["mode"] == mode]
            _plot_lines(axi, dj, metric, mode, show_legend=False, inset_frac=inset_frac)

        # set shared limits if available
        if y_min is not None and y_max is not None:
            ax0.set_ylim(y_min, y_max)

        # only leftmost shows y-label/ticks
        ax0.set_ylabel(metric.replace("_", " "))
        for axi in axes[1:]:
            axi.set_ylabel("")
            axi.tick_params(labelleft=False)

        # legend (right column)
        handles, labels = _dedup_legend(ax0)
        if not handles and len(axes) > 1:
            handles, labels = _dedup_legend(axes[1])

        if handles:
            ax_leg = fig.add_subplot(sub[0, 3])
            ax_leg.axis("off")
            leg = ax_leg.legend(
                handles,
                labels,
                title="UNET mode",
                loc="center left",
                frameon=False,
                ncol=1,
            )
            if leg and leg.get_title():
                leg.get_title().set_fontsize(10)

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """
        Subfigure B:
        - Grid of selected metrics for mode == "tiles"
        - Only leftmost axes per row show y-label "loss value"; others hide y labels
        - Right-side legend column (ncol=1) spanning rows
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=-0.4)

        if "split" not in data.columns:
            raise ValueError("Expected column 'split' in the dataframe.")

        val_df = data[data["split"] == "val"].copy()
        df = val_df[val_df["mode"] == "tiles"].copy()

        metrics = [
            "loss_weighted",
            "loss_unweighted",
            "loss_cell",
            "loss_bound",
            "loss_center",
            "loss_energy",
        ]
        metrics = [m for m in metrics if m in df.columns]
        if not metrics:
            sub = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs)
            ax_msg = fig.add_subplot(sub[0, 0])
            ax_msg.text(0.5, 0.5, "No selected losses found", ha="center", va="center")
            ax_msg.axis("off")
            return

        ncols = 3
        n = len(metrics)
        grid_rows = int(np.ceil(n / ncols))

        # grid with extra legend column
        sub = GridSpecFromSubplotSpec(
            nrows=grid_rows,
            ncols=ncols + 1,
            subplot_spec=gs,
            width_ratios=[1.0] * ncols + [0.25],
            wspace=0,
            hspace=0,
        )

        legend_handles: List = []
        legend_labels: List = []
        axes_meta: List[Tuple[int, int, Axes]] = []

        for idx, metric in enumerate(metrics):
            r, c = divmod(idx, ncols)
            axi = fig.add_subplot(sub[r, c])
            axes_meta.append((r, c, axi))
            _plot_lines(axi, df, metric, metric, show_legend=False, inset_frac=inset_frac)

            if not legend_handles:
                legend_handles, legend_labels = _dedup_legend(axi)

        # turn off unused cells (plot area only)
        for k in range(n, grid_rows * ncols):
            r, c = divmod(k, ncols)
            ax_blank = fig.add_subplot(sub[r, c])
            ax_blank.axis("off")

        # y-label rules
        seen_rows = set()
        for r, c, axi in axes_meta:
            if c == 0 and r not in seen_rows:
                axi.set_ylabel("loss value")
                seen_rows.add(r)
            else:
                axi.set_ylabel("")
                axi.tick_params(labelleft=False)

        # legend column spanning all rows
        if legend_handles:
            ax_leg = fig.add_subplot(sub[:, ncols])
            ax_leg.axis("off")
            leg = ax_leg.legend(
                legend_handles,
                legend_labels,
                title="UNET mode",
                loc="center left",
                frameon=False,
                ncol=1,
            )
            if leg and leg.get_title():
                leg.get_title().set_fontsize(10)

    # ---- figure scaffold ----
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )
    gs = GridSpec(ncols=6, nrows=2, figure=fig, height_ratios=[1, 1.5])
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


def figure_S4_generation(
    h5_path: str,
    figure_output_dir: str,
    model_output_dir: str,
    figure_data_dir: str,
    **kwargs
):
    loss_data = get_loss_data(
        model_output_dir=model_output_dir, output_dir=figure_data_dir
    )
    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S4",
        data=loss_data,
        inset_frac=0.5,
    )

