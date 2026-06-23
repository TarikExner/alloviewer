import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Tuple, List, Optional

from .figure_data_generation import get_loss_data

from . import figure_config as cfg
from . import figure_utils as utils


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


def get_real_image_performance_data(
    validation_results_dir: str,
    *,
    summary_filename: str = "summary_by_epoch.csv",
    preferred_metric: str = "mae",
) -> pd.DataFrame:
    """
    Load and prepare real-image checkpoint-selection results for Figure S4C.

    Expected input file:
        <validation_results_dir>/summary_by_epoch.csv

    Expected best-model-finder columns include at least:
        epoch, unet_mode, dataset_mode, mae

    Returns a plotting dataframe with standardized columns:
        epoch
        error_value
        error_metric
        error_label
        unet_mode
        dataset_mode
        selected
    """
    summary_path = os.path.join(validation_results_dir, summary_filename)

    empty_cols = [
        "epoch",
        "error_value",
        "error_metric",
        "error_label",
        "unet_mode",
        "dataset_mode",
        "selected",
    ]

    if not os.path.isfile(summary_path):
        out = pd.DataFrame(columns=empty_cols)
        out.attrs["message"] = f"No best-model-finder summary found:\n{summary_path}"
        return out

    summary_df = pd.read_csv(summary_path)

    if summary_df.empty:
        out = pd.DataFrame(columns=empty_cols)
        out.attrs["message"] = "Best-model-finder summary is empty"
        return out

    if "epoch" not in summary_df.columns:
        out = pd.DataFrame(columns=empty_cols)
        out.attrs["message"] = "Best-model-finder summary has no 'epoch' column"
        return out

    metric_candidates = [
        preferred_metric,
        "mae",
        "mean_abs_relative_error",
        "rmse",
        "median_abs_error",
        "bias",
    ]

    # Preserve order while removing duplicates.
    seen = set()
    metric_candidates = [
        m for m in metric_candidates
        if not (m in seen or seen.add(m))
    ]

    metric = next((m for m in metric_candidates if m in summary_df.columns), None)

    if metric is None:
        out = pd.DataFrame(columns=empty_cols)
        out.attrs["message"] = (
            "No suitable real-image error metric found in summary_by_epoch.csv"
        )
        return out

    metric_labels = {
        "mae": "mean absolute count error",
        "mean_abs_relative_error": "mean absolute relative error",
        "rmse": "RMSE",
        "median_abs_error": "median absolute error",
        "bias": "count bias",
    }
    error_label = metric_labels.get(metric, metric.replace("_", " "))

    df = summary_df.copy()

    if "dataset_mode" not in df.columns:
        df["dataset_mode"] = "unknown"
    if "unet_mode" not in df.columns:
        df["unet_mode"] = "unknown"
    if "selected" not in df.columns:
        df["selected"] = False

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["error_value"] = pd.to_numeric(df[metric], errors="coerce")

    if df["selected"].dtype != bool:
        df["selected"] = (
            df["selected"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    df["error_metric"] = metric
    df["error_label"] = error_label

    df = df.dropna(subset=["epoch", "error_value"]).copy()
    df["epoch"] = df["epoch"].astype(int)

    keep_cols = [
        "epoch",
        "error_value",
        "error_metric",
        "error_label",
        "unet_mode",
        "dataset_mode",
        "selected",
    ]

    optional_cols = [
        c for c in ["model_file", "model_path", "n_matched"]
        if c in df.columns
    ]

    df = df[keep_cols + optional_cols].sort_values(
        ["unet_mode", "dataset_mode", "epoch"]
    ).reset_index(drop=True)

    if df.empty:
        df.attrs["message"] = (
            f"No finite values found for metric '{metric}' in summary_by_epoch.csv"
        )

    return df


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


def _generate_main_figure(
    figure_output_dir: str,
    figure_name: str,
    *,
    data: pd.DataFrame,
    real_image_performance_data: pd.DataFrame,
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

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """
        Subfigure C:
        - Real microscopy image performance from best-model-finder output.
        - x-axis: epoch
        - y-axis: count error term
        """
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=-0.4)

        sub = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=1,
            subplot_spec=gs,
            wspace=0.0,
            hspace=0.0,
        )
        axc = fig.add_subplot(sub[0, 0])

        df = real_image_performance_data.copy()

        if df.empty:
            message = df.attrs.get(
                "message",
                "No real-image performance data found",
            )
            axc.text(0.5, 0.5, message, ha="center", va="center")
            axc.set_xticks([])
            axc.set_yticks([])
            return

        df = df.sort_values(["unet_mode", "dataset_mode", "epoch"])

        for (unet_mode, dataset_mode), grp in df.groupby(
            ["unet_mode", "dataset_mode"], dropna=False
        ):
            grp = grp.sort_values("epoch")
            label = f"{unet_mode} ({dataset_mode})"
            axc.plot(
                grp["epoch"],
                grp["error_value"],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=label,
            )

            if "selected" in grp.columns:
                selected = grp[grp["selected"]]
                if not selected.empty:
                    axc.scatter(
                        selected["epoch"],
                        selected["error_value"],
                        s=36,
                        zorder=5,
                        edgecolor="black",
                        linewidth=0.7,
                    )

        error_label = str(df["error_label"].iloc[0])
        axc.set_title("Performance on real microscopy images")
        axc.set_xlabel("epoch")
        axc.set_ylabel(error_label)
        axc.grid(True, axis="y", alpha=0.25)

        handles, labels = _dedup_legend(axc)
        if handles:
            axc.legend(
                handles,
                labels,
                title="model",
                loc="best",
                frameon=False,
                ncol=1,
            )

    # ---- figure scaffold ----
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )
    gs = GridSpec(
        ncols=6,
        nrows=3,
        figure=fig,
        height_ratios=[1, 1.5, 0.44],
    )
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


def figure_S4_generation(
    figure_output_dir: str,
    model_output_dir: str,
    validation_results_dir: str,
    **kwargs
):
    loss_data = get_loss_data(
        model_output_dir=model_output_dir, output_dir=validation_results_dir
    )
    real_image_performance_data = get_real_image_performance_data(
        validation_results_dir=validation_results_dir
    )
    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S4",
        data=loss_data,
        real_image_performance_data=real_image_performance_data,
        inset_frac=0.5,
    )

