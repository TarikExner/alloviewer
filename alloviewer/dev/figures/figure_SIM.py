import os
from typing import Any, Dict, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .figure_data_generation import generate_param_showcase
from . import figure_config as cfg

from alloviewer.dev.segmentation.image_simulation import (
    CameraDimension,
    SimulatorConfig,
)


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

_PARAM_DISPLAY_NAMES = {
    # geometry / well
    "well_radius_frac": "well radius",
    "well_center_jitter": "well center jitter",

    # background
    "background_level": "background level",
    "edge_boost": "edge boost",
    "radial_gamma": "radial gamma",
    "vignette_strength": "vignette",
    "background_texture_enable": "texture on",
    "background_texture_strength": "texture strength",
    "background_texture_sigma_fine": "fine texture sigma",
    "background_texture_sigma_coarse": "coarse texture sigma",
    "background_texture_fine_weight": "fine texture weight",
    "background_texture_coarse_weight": "coarse texture weight",
    "background_texture_downsample": "texture downsample",
    "background_texture_fullres_fine_strength": "full-res fine noise",
    "wall_blur_sigma": "wall blur",
    "ring_artifacts": "ring artifacts",
    "ring_sigma_range": "ring width",
    "ring_alpha_range": "ring intensity",

    # cell number / color / size / shape / focus
    "n_cells": "cell count",
    "frac_positive": "positive fraction",
    "cell_diameter": "cell diameter",
    "large_cell_frac": "large-cell fraction",
    "large_cell_diameter_factor": "large-cell factor",
    "cell_axis_jitter": "ellipse jitter",
    "cell_intensity_range": "cell brightness",
    "color_jitter": "color jitter",
    "focus_frac_in": "in-focus fraction",
    "sigma_in": "in-focus blur",
    "sigma_out": "out-of-focus blur",

    # placement
    "rim_bias": "rim bias",
    "rim_band": "rim band",
    "edge_clamp": "edge clamp",
    "rim_min_sep_px": "rim min. spacing",
    "wall_margin_px": "wall margin",
    "min_cell_sep_px": "min. cell spacing",
    "pack_iters": "packing iterations",
    "pack_strength": "packing strength",

    # side bias
    "side_bias_strength": "side-bias strength",
    "side_bias_theta": "side-bias angle",
    "side_bias_kappa": "side-bias concentration",
    "side_bias_inner_frac": "side-bias inner limit",

    # clusters
    "clustered_cell_frac": "clustered fraction",
    "cluster_size_range": "cluster size",
    "cluster_contact_factor_range": "cluster contact",
    "cluster_chain_probability": "chain probability",
    "cluster_angle_jitter": "cluster angle jitter",
    "cluster_packed_probability": "packed probability",
    "cluster_packed_contact_factor_range": "packed contact",
    "cluster_packed_region_join_probability": "packed-region joining",
    "cluster_pack_min_sep_factor": "cluster min. spacing",

    # ghosts
    "ghost_density": "ghost density",
    "ghost_offset_px": "ghost offset",
    "ghost_offset_jitter": "ghost offset jitter",
    "ghost_sigma": "ghost blur",
    "ghost_dilate": "ghost size",
    "ghost_intensity": "ghost intensity",
    "ghost_stretch": "ghost stretch",
    "ghost_trail": "ghost trail count",
    "ghost_trail_decay": "ghost trail decay",

    # debris
    "dirt_density": "debris density",
    "dirt_size": "debris size",
    "dirt_sigma": "debris blur",
    "dirt_alpha": "debris intensity",

    # reflections
    "reflect_n": "reflection count",
    "reflect_theta_sigma": "reflection angular width",
    "reflect_radial_sigma": "reflection radial width",
    "reflect_offset_range": "reflection offset",
    "reflect_alpha_range": "reflection intensity",
    "reflect_wobble": "reflection wobble",
    "reflect_harmonics": "reflection harmonics",
    "reflect_harmonic_decay": "harmonic decay",
}


def _display_name(param_name: str) -> str:
    return _PARAM_DISPLAY_NAMES.get(param_name, param_name)


def _format_scalar(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        if abs(xf) >= 100:
            return f"{xf:.0f}"
        if abs(xf) >= 10:
            return f"{xf:.1f}"
        if abs(xf) >= 1:
            return f"{xf:.2g}"
        return f"{xf:.3g}"
    return str(x)


def _format_value(x: Any) -> str:
    if isinstance(x, tuple):
        if len(x) == 2 and x[0] == x[1]:
            return _format_scalar(x[0])
        if len(x) <= 4:
            return "(" + ", ".join(_format_scalar(v) for v in x) + ")"
        return "tuple"
    if isinstance(x, list):
        if len(x) <= 4:
            return "[" + ", ".join(_format_scalar(v) for v in x) + "]"
        return "list"
    return _format_scalar(x)


def _merge_overrides(*dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        if d:
            out.update(d)
    return out


# ---------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------

def _generate_main_figure(
    bundle: Dict[str, Any],
    *,
    fig_scale: float = 1.0,
    annotate_mode: str = "textbox",  # "textbox" or "title"
    cmap=None,                       # only used if images are single-channel
    row_label_at_left: bool = True,
    figure_output_dir: str,
    figure_name: str,
    dpi: int = 300,
) -> None:
    """
    Build a multi-row parameter figure.

    The expected bundle format is the output of generate_param_showcase():
      images[r][c]      image for row r and column c
      param_names[r]    parameter changed in row r
      value_lists[r][c] parameter value used in column c
      n_rows            number of parameter rows
      n_cols            number of values per row
    """
    images = bundle["images"]
    param_names = bundle["param_names"]
    value_lists = bundle["value_lists"]
    n_rows = int(bundle["n_rows"])
    n_cols = int(bundle["n_cols"])

    fig_w = float(cfg.FIGURE_WIDTH_FULL) * fig_scale
    fig_h = max(float(cfg.FIGURE_HEIGHT_FULL), 0.78 * n_rows) * fig_scale

    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    outer: GridSpec = GridSpec(
        nrows=n_rows,
        ncols=1,
        figure=fig,
        height_ratios=[1] * n_rows,
        hspace=0.04,
    )

    for r in range(n_rows):
        param = param_names[r]
        param_label = _display_name(param)

        inner = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=n_cols,
            subplot_spec=outer[r],
            wspace=0.015,
        )

        for c in range(n_cols):
            ax = fig.add_subplot(inner[0, c])

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
            for spine in ax.spines.values():
                spine.set_visible(False)

            if c < len(value_lists[r]):
                label_val = _format_value(value_lists[r][c])
                if row_label_at_left:
                    tile_label = label_val
                else:
                    tile_label = f"{param_label} = {label_val}"

                if annotate_mode == "title":
                    ax.set_title(tile_label, fontsize=6)
                else:
                    ax.text(
                        0.02,
                        0.04,
                        tile_label,
                        transform=ax.transAxes,
                        fontsize=6,
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.20",
                            fc="black",
                            ec="none",
                            alpha=0.58,
                        ),
                        ha="left",
                        va="bottom",
                    )

        if row_label_at_left:
            fig.text(
                0.006,
                (n_rows - r - 0.5) / n_rows,
                param_label,
                ha="left",
                va="center",
                fontsize=7,
                rotation=90,
                alpha=0.85,
            )

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _run_showcase_figure(
    *,
    sim_config: SimulatorConfig,
    camera: CameraDimension,
    sweep: Dict[str, list],
    base_overrides: Dict[str, Any],
    figure_output_dir: str,
    figure_name: str,
    seed: int,
    fig_scale: float,
    annotate_mode: str,
    dpi: int,
) -> None:
    bundle = generate_param_showcase(
        sim_config=sim_config,
        camera=camera,
        sweep=sweep,
        base_overrides=base_overrides,
        seed=seed,
    )

    _generate_main_figure(
        bundle=bundle,
        figure_output_dir=figure_output_dir,
        figure_name=figure_name,
        fig_scale=fig_scale,
        annotate_mode=annotate_mode,
        dpi=dpi,
    )


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def figure_SIM_generation(
    figure_output_dir: str,
    **kwargs,
):
    """
    Generate supplementary simulator parameter figures.

    Optional kwargs:
      seed: int
      fig_scale: float
      annotate_mode: "textbox" or "title"
      dpi: int
      sim_config: SimulatorConfig
      camera: CameraDimension
      only: optional iterable of figure names, e.g. {"Figure_SU3", "Figure_SU7"}
    """
    seed = int(kwargs.get("seed", 42))
    fig_scale = float(kwargs.get("fig_scale", 1.0))
    annotate_mode = kwargs.get("annotate_mode", "textbox")
    dpi = int(kwargs.get("dpi", 300))
    only = kwargs.get("only", None)
    only = set(only) if only is not None else None

    sim_config = kwargs.get("sim_config", SimulatorConfig())
    camera = kwargs.get(
        "camera",
        CameraDimension(
            name="Extended_Data_Cam",
            W=512,
            H=512,
        ),
    )

    # Clean baseline. Each figure turns on the artifact class it needs.
    base_clean = dict(
        H=512,
        W=512,
        n_cells=400,
        frac_positive=0.5,
        cluster_enable=False,
        ghost_enable=False,
        reflect_enable=False,
        dirt_density=0.0,
        ring_artifacts=0,
        background_texture_enable=True,
        background_texture_strength=0.04,
        focus_frac_in=1.0,
        sigma_in=(0.45, 0.45),
        return_targets=False,
    )

    jobs = []

    jobs.append(
        dict(
            figure_name="Extended_Figure_1",
            sweep={
                "well_radius_frac": [0.30, 0.36, 0.42, 0.48],
                "well_center_jitter": [0.00, 0.01, 0.03, 0.06],
                "background_level": [0.00, 0.05, 0.10, 0.18],
                "edge_boost": [0.00, 0.15, 0.35, 0.70],
                "radial_gamma": [0.50, 1.00, 1.80, 3.00],
                "vignette_strength": [0.00, 0.10, 0.25, 0.45],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=250,
                    background_texture_strength=0.02,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_2",
            sweep={
                "background_texture_strength": [0.00, 0.03, 0.08, 0.16],
                "background_texture_sigma_fine": [0.00, 0.30, 0.80, 1.60],
                "background_texture_sigma_coarse": [1.00, 4.00, 10.00, 20.00],
                "background_texture_fine_weight": [0.00, 0.35, 0.70, 1.00],
                "background_texture_downsample": [1, 2, 4, 8],
                "background_texture_fullres_fine_strength": [0.00, 0.005, 0.02, 0.08],
                "wall_blur_sigma": [0.00, 4.00, 12.00, 30.00],
                "ring_artifacts": [0, 1, 3, 6],
                "ring_sigma_range": [(2.0, 2.0), (6.0, 6.0), (12.0, 12.0), (24.0, 24.0)],
                "ring_alpha_range": [(0.00, 0.00), (0.03, 0.03), (0.12, 0.12), (0.35, 0.35)],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=200,
                    background_texture_enable=True,
                    background_texture_strength=0.08,
                    ring_artifacts=2,
                    ring_alpha_range=(0.08, 0.08),
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_3",
            sweep={
                "n_cells": [50, 200, 600, 1200],
                "frac_positive": [0.00, 0.25, 0.50, 1.00],
                "cell_diameter": [4, 7, 10, 14],
                "large_cell_frac": [0.00, 0.10, 0.30, 0.60],
                "large_cell_diameter_factor": [1.00, 1.30, 1.80, 2.40],
                "cell_axis_jitter": [0.00, 0.15, 0.35, 0.60],
                "cell_intensity_range": [(0.35, 0.35), (0.65, 0.65), (0.95, 0.95), (1.20, 1.20)],
                "color_jitter": [0.00, 0.05, 0.15, 0.35],
                "focus_frac_in": [1.00, 0.70, 0.35, 0.00],
                "sigma_in": [(0.10, 0.10), (0.40, 0.40), (0.80, 0.80), (1.40, 1.40)],
                "sigma_out": [(0.80, 0.80), (1.40, 1.40), (2.40, 2.40), (3.40, 3.40)],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=400,
                    cell_diameter_bounds_by_short_side=None,
                    cell_diameter=8,
                    large_cell_frac=0.0,
                    large_cell_diameter_factor=1.8,
                    focus_frac_in=0.50,
                    sigma_in=(0.45, 0.45),
                    sigma_out=(2.20, 2.20),
                    background_texture_strength=0.03,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_4",
            sweep={
                "rim_bias": [0.00, 0.30, 0.70, 0.95],
                "rim_band": [0.05, 0.15, 0.35, 0.60],
                "edge_clamp": [0.00, 0.25, 0.65, 0.95],
                "rim_min_sep_px": [0, 4, 12, 24],
                "wall_margin_px": [0.00, 2.00, 8.00, 16.00],
                "min_cell_sep_px": [None, 2, 6, 12],
                "pack_iters": [0, 5, 20, 60],
                "pack_strength": [0.00, 0.20, 0.45, 0.90],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=650,
                    cell_diameter_bounds_by_short_side=None,
                    cell_diameter=7,
                    rim_bias=0.70,
                    rim_band=0.20,
                    edge_clamp=0.50,
                    pack_iters=20,
                    pack_strength=0.45,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_5",
            sweep={
                "side_bias_strength": [0.00, 0.35, 0.70, 1.00],
                "side_bias_theta": [0.00, 0.50 * np.pi, np.pi, 1.50 * np.pi],
                "side_bias_kappa": [0.10, 1.00, 5.00, 15.00],
                "side_bias_inner_frac": [0.10, 0.30, 0.60, 0.85],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=700,
                    side_bias_enable=True,
                    side_bias_strength=1.0,
                    side_bias_theta=0.0,
                    side_bias_kappa=6.0,
                    side_bias_inner_frac=0.35,
                    rim_bias=0.85,
                    rim_band=0.35,
                    edge_clamp=0.30,
                    cell_diameter_bounds_by_short_side=None,
                    cell_diameter=7,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_6",
            sweep={
                "clustered_cell_frac": [0.00, 0.25, 0.60, 1.00],
                "cluster_size_range": [(2, 3), (3, 6), (6, 12), (10, 20)],
                "cluster_contact_factor_range": [(1.25, 1.25), (1.05, 1.05), (0.95, 0.95), (0.85, 0.85)],
                "cluster_chain_probability": [0.00, 0.35, 0.70, 1.00],
                "cluster_angle_jitter": [0.00, 0.25, 0.65, 1.20],
                "cluster_packed_probability": [0.00, 0.35, 0.70, 1.00],
                "cluster_packed_contact_factor_range": [(1.15, 1.15), (1.00, 1.00), (0.92, 0.92), (0.84, 0.84)],
                "cluster_packed_region_join_probability": [0.00, 0.25, 0.60, 1.00],
                "cluster_pack_min_sep_factor": [0.40, 0.65, 0.84, 0.90],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=350,
                    cell_diameter_bounds_by_short_side=None,
                    cell_diameter=9,
                    cluster_enable=True,
                    clustered_cell_frac=0.75,
                    cluster_size_range=(4, 10),
                    cluster_chain_probability=0.55,
                    cluster_packed_probability=0.55,
                    cluster_packed_region_join_probability=0.30,
                    rim_bias=0.45,
                    rim_band=0.25,
                    edge_clamp=0.25,
                    pack_iters=30,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_7",
            sweep={
                "ghost_density": [0.00, 0.20, 0.50, 1.00],
                "ghost_offset_px": [0.00, 6.00, 14.00, 28.00],
                "ghost_offset_jitter": [0.00, 4.00, 10.00, 20.00],
                "ghost_sigma": [(0.80, 0.80), (2.00, 2.00), (4.00, 4.00), (7.00, 7.00)],
                "ghost_dilate": [0.25, 0.75, 1.50, 3.00],
                "ghost_intensity": [(0.05, 0.05), (0.20, 0.20), (0.60, 0.60), (1.20, 1.20)],
                "ghost_stretch": [1.00, 2.00, 3.00, 5.00],
                "ghost_trail": [1, 2, 3, 5],
                "ghost_trail_decay": [0.20, 0.45, 0.65, 0.85],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=450,
                    ghost_enable=True,
                    ghost_density=0.50,
                    ghost_offset_px=12.0,
                    ghost_offset_jitter=5.0,
                    ghost_sigma=(3.0, 3.0),
                    ghost_dilate=1.2,
                    ghost_intensity=(0.45, 0.45),
                    ghost_stretch=3.0,
                    ghost_trail=3,
                    rim_bias=0.95,
                    rim_band=0.15,
                    edge_clamp=0.70,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_8",
            sweep={
                "dirt_density": [0.0000, 0.0002, 0.0008, 0.0020],
                "dirt_size": [(2, 2), (4, 4), (8, 8), (14, 14)],
                "dirt_sigma": [(0.20, 0.20), (0.70, 0.70), (1.40, 1.40), (2.50, 2.50)],
                "dirt_alpha": [(0.02, 0.02), (0.08, 0.08), (0.25, 0.25), (0.70, 0.70)],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=80,
                    dirt_density=0.0010,
                    dirt_size=(5, 5),
                    dirt_sigma=(1.0, 1.0),
                    dirt_alpha=(0.18, 0.18),
                    background_texture_strength=0.03,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Figure_9",
            sweep={
                "reflect_n": [0, 1, 4, 9],
                "reflect_theta_sigma": [0.04, 0.08, 0.14, 0.24],
                "reflect_radial_sigma": [2.00, 6.00, 12.00, 22.00],
                "reflect_offset_range": [(2.0, 2.0), (8.0, 8.0), (18.0, 18.0), (32.0, 32.0)],
                "reflect_alpha_range": [(0.00, 0.00), (0.08, 0.08), (0.25, 0.25), (0.70, 0.70)],
                "reflect_wobble": [0.00, 0.15, 0.40, 1.00],
                "reflect_harmonics": [0, 1, 2, 4],
                "reflect_harmonic_decay": [0.00, 0.20, 0.55, 0.90],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=180,
                    reflect_enable=True,
                    reflect_n=5,
                    reflect_theta_sigma=0.10,
                    reflect_radial_sigma=8.0,
                    reflect_offset_range=(8.0, 24.0),
                    reflect_alpha_range=(0.18, 0.18),
                    reflect_wobble=0.25,
                    reflect_harmonics=2,
                    reflect_harmonic_decay=0.55,
                    background_texture_strength=0.03,
                ),
            ),
        )
    )

    generated = []

    for job in jobs:
        figure_name = job["figure_name"]
        if only is not None and figure_name not in only:
            continue

        _run_showcase_figure(
            sim_config=sim_config,
            camera=camera,
            sweep=job["sweep"],
            base_overrides=job["base_overrides"],
            figure_output_dir=figure_output_dir,
            figure_name=figure_name,
            seed=seed,
            fig_scale=fig_scale,
            annotate_mode=annotate_mode,
            dpi=dpi,
        )

        generated.append(
            {
                "figure": figure_name,
                "pdf": os.path.join(figure_output_dir, f"{figure_name}.pdf"),
                "png": os.path.join(figure_output_dir, f"{figure_name}.png"),
            }
        )

    return generated
