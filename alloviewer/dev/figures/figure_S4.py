import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from .figure_data_generation import fetch_image_with_targets

from . import figure_config as cfg

def _generate_main_figure(
    figure_output_dir: str = "",
    figure_name: str = "",
    *,
    img: np.ndarray,
    tgts: np.ndarray,
):
    """
    Build a 6x3 grid (18 tiles). First 17 tiles show histograms of cols_to_plot
    with hue=dataset_col using stacked counts. The last tile shows the legend only.
    Saves PDF/PNG if figure_output_dir is given, and also returns (fig, axes).
    """

    titles = ["image","cell mask","cell boundary","cell centers","energy"]
    inset_centers=(260, 260)
    inset_box_px=96
    inset_zoom=3.0

    fig = plt.figure(layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL/4))
    gs = GridSpec(nrows=1, ncols=5, figure=fig, wspace=0.04, hspace=0.0)

    # titles
    if titles is None:
        titles = ["image", "tgt0", "tgt1", "tgt2", "tgt3"]
    if len(titles) != 5:
        titles = titles[:5] + [""] * max(0, 5 - len(titles))

    # inset centers
    H, W = img.shape[:2]
    default_center = (H // 2, W // 2)
    if inset_centers is None:
        centers = [default_center] * 5
    elif isinstance(inset_centers, tuple) and len(inset_centers) == 2:
        centers = [tuple(map(int, inset_centers))] * 5
    else:
        assert isinstance(inset_centers, list) and len(inset_centers) == 5, \
            "inset_centers must be None, a single (y,x) tuple, or a list of 5 (y,x) tuples."
        centers = [tuple(map(int, c)) for c in inset_centers]

    def _add_inset_top_right(ax, arr, center, is_target: bool):
        cy, cx = center
        half = inset_box_px // 2
        y0 = int(np.clip(cy - half, 0, arr.shape[0] - 1))
        x0 = int(np.clip(cx - half, 0, arr.shape[1] - 1))
        y1 = int(np.clip(y0 + inset_box_px, 1, arr.shape[0]))
        x1 = int(np.clip(x0 + inset_box_px, 1, arr.shape[1]))

        # top-right inset (no rectangle, no connectors)
        axins = zoomed_inset_axes(ax, zoom=inset_zoom, loc="upper right", borderpad=0.2)
        if is_target:
            axins.imshow(arr, cmap="jet", interpolation="nearest")
        else:
            axins.imshow(arr, interpolation="nearest")
        axins.set_xlim(x0, x1)
        axins.set_ylim(y1, y0)  # flip y for image coords
        axins.set_xticks([])
        axins.set_yticks([])
        for sp in axins.spines.values():
            sp.set_linewidth(0.8)
            sp.set_edgecolor("white")

    # panel 0: image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.set_title(titles[0], fontsize=9)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_visible(False)
    _add_inset_top_right(ax0, img, centers[0], is_target=False)

    # panels 1..4: targets (jet)
    for c in range(4):
        ax = fig.add_subplot(gs[0, c + 1])
        ax.imshow(tgts[..., c], cmap="jet", interpolation="nearest")
        ax.set_title(titles[c + 1], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        _add_inset_top_right(ax, tgts[..., c], centers[c + 1], is_target=True)

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    return


def figure_S4_generation(
    h5_path: str,
    figure_output_dir: str,
    **kwargs
):
    h5_path = os.path.join(h5_path, "tiles_val.h5")

    index = 12
    rgb, tgts = fetch_image_with_targets(
        h5_path=h5_path,
        index=index,
        image_key="imgs",
        target_key="tgts",
        resize_to=(512,512),
        target_resize_to=None,
        return_channel_last=True,   # -> (H, W, C)
    )

    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S4",
        img=rgb,
        tgts=tgts
    )
