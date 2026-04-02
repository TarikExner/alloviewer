import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from .figure_data_generation import get_dataset_statistics

from . import figure_config as cfg

def _generate_main_figure(
    figure_output_dir: str = "",
    figure_name: str = "",
    *,
    data: pd.DataFrame,
    cols_to_plot: list[str]
):
    """
    Build a 6x3 grid (18 tiles). First 17 tiles show histograms of cols_to_plot
    with hue=dataset_col using stacked counts. The last tile shows the legend only.
    Saves PDF/PNG if figure_output_dir is given, and also returns (fig, axes).
    """

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(nrows=6, ncols=3, figure=fig)

    # palette and hue levels
    levels = list(pd.Index(data["crop_method"].astype(str)).unique())
    palette = sns.color_palette(cfg.HIST_CMAP, n_colors=len(levels))

    total_tiles = 5 * 3
    last_idx = total_tiles - 1  # legend tile index = 17
    axes = []

    def _process_title(title: str):
        title = title.replace("_", " ")
        if title == "H":
            title = "image height"
        if title == "W":
            title = "image width"
        return title


    def _make_hue_legend(ax, levels, palette, alpha=0.45, title=None, fontsize=9):
        handles = [
            Patch(facecolor=palette[i], edgecolor="none", alpha=alpha, label=str(lv))
            for i, lv in enumerate(levels)
        ]
        return ax.legend(
            handles=handles,
            title=title,
            frameon=False,
            bbox_to_anchor = (0, 0.5),
            loc="center left",
            fontsize=fontsize
        )


    for i in range(total_tiles):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        if i == last_idx:
            ax.axis("off")
            continue

        # out of columns? hide axis (except last)
        if i >= len(cols_to_plot):
            continue

        col = cols_to_plot[i]
        # common bin range (numeric coercion for safety)
        x = pd.to_numeric(data[col], errors="coerce").dropna().to_numpy()
        if x.size == 0 or not np.isfinite(x).any():
            ax.axis("off")
            continue

        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 0.5, hi + 0.5

        sns.histplot(
            data=data,
            x=col,
            hue="crop_method",
            bins=30,
            binrange=(lo, hi),
            stat="count",
            multiple="stack",
            alpha=0.45,
            element="poly",
            edgecolor=None,
            ax=ax,
            palette=palette,
            kde=False,
        )

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        ax.set_title(_process_title(col), fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("count")

    # draw legend in the last tile
    ax_leg = axes[last_idx]
    _make_hue_legend(
        ax_leg,
        levels=levels,
        palette=[palette[i] for i in range(len(levels))],
        title="crop method",
        fontsize=9,
    )

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    return


def figure_S3_generation(
    h5_path: str,
    figure_output_dir: str,
    figure_data_dir: str,
    **kwargs
):
    crop_well_stats = get_dataset_statistics(
        h5_path = os.path.join(h5_path, "crop_well_resize_train.h5"),
        output_dir = figure_data_dir
    )
    pad_stats = get_dataset_statistics(
        h5_path = os.path.join(h5_path, "pad_resize_train.h5"),
        output_dir = figure_data_dir
    )
    tile_stats = get_dataset_statistics(
        h5_path = os.path.join(h5_path, "tiles_train.h5"),
        output_dir = figure_data_dir
    )

    data = pd.concat([crop_well_stats, pad_stats, tile_stats], axis = 0)

    cols_to_plot = [
        "n_cells",
        "frac_positive",
        "photon_level",
        "read_noise",
        "H",
        "W",
        "background_level",
        "color_jitter",
        "rim_bias",
        "rim_band",
        "edge_clamp",
        "pack_strength",
        "ghost_density",
        "dirt_density",
    ]

    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S3",
        data = data,
        cols_to_plot = cols_to_plot
    )
