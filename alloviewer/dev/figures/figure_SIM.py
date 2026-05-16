import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from typing import Callable, Any, Dict

from .figure_data_generation import generate_param_showcase

from . import figure_config as cfg

from alloviewer.dev.segmentation.image_simulation import CameraDimension, SimulatorConfig


def _generate_main_figure(
    bundle: Dict[str, Any],
    *,
    fig_scale: float = 1.0,
    annotate_mode: str = "textbox",  # "textbox" or "title"
    cmap=None,  # only used if images are single-channel
    row_label_at_left: bool = True,
    figure_output_dir: str,
    figure_name: str,
) -> None:
    """
    Build a figure with GridSpec:
      - outer GridSpec has n_rows
      - each row is a GridSpecFromSubplotSpec with n_cols
    """
    images = bundle["images"]
    param_names = bundle["param_names"]
    value_lists = bundle["value_lists"]
    n_rows = bundle["n_rows"]
    n_cols = bundle["n_cols"]

    fig = plt.figure(
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL), layout = "constrained")
    outer: GridSpec = GridSpec(
        nrows=n_rows, ncols=1, figure=fig, height_ratios=[1]*n_rows, hspace=0.05
    )

    for r in range(n_rows):
        inner = GridSpecFromSubplotSpec(
            nrows=1, ncols=n_cols, subplot_spec=outer[r], wspace=0.02
        )
        for c in range(n_cols):
            ax = fig.add_subplot(inner[0, c])  # inner[...] returns a SubplotSpec

            img = images[r][c]
            if img is None:
                ax.axis("off")
                continue

            if img.ndim == 2:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))

            ax.set_xticks([])
            ax.set_yticks([])

            # per-tile annotation
            if c < len(value_lists[r]):
                label_val = value_lists[r][c]
                if isinstance(label_val, tuple):
                    label_val = label_val[0]
                if annotate_mode == "title":
                    ax.set_title(f"{param_names[r]} = {label_val}", fontsize=4)
                else:
                    ax.text(
                        0.02, 0.04, f"{param_names[r]} = {label_val}",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55),
                        ha="left", va="bottom",
                    )

        # row label on the left margin
        if row_label_at_left:
            fig.text(
                -0.015,
                (n_rows - r - 0.5) / n_rows,
                param_names[r],
                ha="left", va="center", fontsize=8, rotation=90, alpha=0.8
            )

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    return


def figure_SIM_generation(
    figure_output_dir: str,
    **kwargs
):

    sim_config = SimulatorConfig()
    camera = CameraDimension(
        name = "Extended_Data_Cam",
        W = 1024,
        H = 1024
    )
    sweep = {
        "background_level": [0.00, 0.10, 0.18, 0.26],
        "color_jitter": [0.00, 0.1, 0.2, 0.3],
        "edge_boost": [0, 0.3, 0.6, 1],
        "n_cells": [100, 400, 800, 1500],
        "frac_positive": [0.00, 0.25, 0.50, 0.75],
        "cell_diameter": [2, 4, 6, 8],
    }
    bundle = generate_param_showcase(
        sim_config=sim_config,
        camera=camera,
        sweep=sweep,
        base_overrides=dict(H=512, W=512, n_cells=400, cell_diameter=4, bg_hue = 0.5),
        seed=42,
    )

    _generate_main_figure(
        bundle=bundle,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_SU1",
    )

    sweep = {
        "sigma_in": [(0,0), (0.5,0.5), (1,1), (1.5,1.5)],
        "sigma_out": [(0,0), (0.5,0.5), (1,1), (1.5,1.5)],
        "rim_bias": [0.2, 0.5, 0.8, 0.95],
        "rim_band": [0.05, 0.2, 0.5, 0.7],
        "edge_clamp": [0.2, 0.4, 0.6, 0.8],
        "rim_min_sep_px": [0, 20, 50, 100],
    }
    bundle = generate_param_showcase(
        sim_config=sim_config,
        camera=camera,
        sweep=sweep,
        base_overrides=dict(H=512, W=512, n_cells=400, cell_diameter=4, bg_hue = 0.5, focus_frac_in = 1),
        seed=42,
    )

    _generate_main_figure(
        bundle=bundle,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_SU2",
    )

    sweep = {
        "side_bias_strength": [0.00, 0.5, 0.8, 1.0],
        "side_bias_theta": [0,1,2,3],
        "side_bias_kappa": [0, 1, 5, 10],
        "side_bias_inner_frac": [0.1, 0.3, 0.6, 0.9],
        "ring_artifacts": [0, 1, 3, 5],
        "ring_alpha_range": [(0,0), (0.03, 0.03), (0.12, 0.12), (0.5, 0.5)],
    }
    bundle = generate_param_showcase(
        sim_config=sim_config,
        camera=camera,
        sweep=sweep,
        base_overrides=dict(H=512, W=512, n_cells=400, cell_diameter=4, bg_hue = 0.5,
                            side_bias_enable = True,
                            side_bias_strength = 1,
                            edge_clamp = 0.2,
                            side_bias_inner_frac = 0.3,
                            rim_min_sep_px = 4,
                            rim_bias = 0.8,
                            rim_band = 0.4,
                            side_bias_kappa = 2),
        seed=42,
    )

    _generate_main_figure(
        bundle=bundle,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_SU3",
    )

    sweep = {
        "ghost_density": [0.1, 0.3, 0.5, 1],
        "ghost_stretch": [1, 2, 3, 4],
        "ghost_offset_jitter": [0, 5, 10, 20],
        "ghost_sigma": [(1,1), (2,2), (4,4), (6,6)],
        "ghost_dilate": [0, 1, 3, 5],
        "ghost_intensity": [(0.01, 0.01), (0.05, 0.05), (0.1, 0.1), (0.3, 0.3)],
    }
    bundle = generate_param_showcase(
        sim_config=sim_config,
        camera=camera,
        sweep=sweep,
        base_overrides=dict(H=512, W=512, n_cells=400, cell_diameter=4, bg_hue = 0.5),
        seed=42,
    )

    _generate_main_figure(
        bundle=bundle,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_SU4",
    )

    sweep = {
        "dirt_density": [0.0001, 0.0004, 0.0008, 0.001],
        "dirt_size": [(4,4), (6,6), (8,8), (12,12)],
        "dirt_sigma": [(0,0), (0.5, 0.5), (1,1), (2,2)],
        "dirt_alpha": [(0.1,0.1), (0.3, 0.3), (0.6, 0.6), (1,1)],
        "blur_sigma_global": [0, 0.5, 1.25, 2],
    }
    bundle = generate_param_showcase(
        sim_config=sim_config,
        camera=camera,
        sweep=sweep,
        base_overrides=dict(H=512, W=512, n_cells=50, cell_diameter=4, bg_hue = 0.5,
                            dirt_density = 0.001),
        seed=42,
    )

    _generate_main_figure(
        bundle=bundle,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_SU5",
    )

    sweep = {
        "reflect_n": [1,3,7,10],
        "reflect_theta_sigma": [0.5,0.1,0.15,0.2],
        "reflect_radial_sigma": [2,6,12,18],
        "reflect_wobble": [0, 0.2, 0.4, 1],
        "reflect_alpha_range": [(0.1,0.1), (0.3,0.3), (0.6, 0.6), (1,1)],
        "reflect_harmonic_decay": [0, 0.2, 0.5, 1],
    }
    bundle = generate_param_showcase(
        sim_config=sim_config,
        camera=camera,
        sweep=sweep,
        base_overrides=dict(H=512, W=512, n_cells=400, cell_diameter=4, bg_hue = 0.5),
        seed=42,
    )

    _generate_main_figure(
        bundle=bundle,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_SU6",
    )

