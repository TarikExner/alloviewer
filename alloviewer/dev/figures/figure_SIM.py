from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from alloviewer.dev.figures import figure_config as cfg
from alloviewer.dev.figures import figure_data_generation as fdg

from alloviewer.dev.segmentation.image_simulation import (
    CameraDimension,
    SimulatorConfig,
)


_PARAM_DISPLAY_NAMES = {
    # geometry / well
    "well_radius_frac": "well radius",
    "well_center_jitter": "well center jitter",

    # background
    "background_level": "background level",
    "edge_boost": "edge boost",
    "radial_gamma": "radial gamma",
    "vignette_strength": "vignette",
    "background_texture_strength": "texture strength",
    "background_texture_sigma_fine": "fine texture sigma",
    "background_texture_sigma_coarse": "coarse texture sigma",
    "background_texture_fine_weight": "fine texture weight",
    "background_texture_coarse_weight": "coarse texture weight",
    "background_texture_downsample": "texture downsample",
    "background_texture_fullres_fine_strength": "fine full-res noise",
    "wall_blur_sigma": "wall blur",
    "ring_artifacts": "ring count",
    "ring_sigma_range": "ring width",
    "ring_alpha_range": "ring intensity",

    # cells
    "n_cells": "cell count",
    "frac_positive": "positive fraction",
    "cell_diameter": "cell diameter",
    "large_cell_frac": "large-cell fraction",
    "large_cell_diameter_factor": "large-cell factor",
    "cell_axis_jitter": "ellipse jitter",
    "cell_intensity_range": "cell brightness",
    "color_jitter": "color jitter",
    "focus_frac_in": "in-focus fraction",
    "sigma_in": "cell blur",
    "sigma_out": "out-of-focus blur",

    # placement
    "rim_bias": "rim bias",
    "rim_band": "rim band",
    "edge_clamp": "edge clamp",
    "rim_min_sep_px": "rim spacing",
    "wall_margin_px": "wall margin",
    "min_cell_sep_px": "cell spacing",
    "pack_iters": "packing iterations",
    "pack_strength": "packing strength",

    # side bias
    "side_bias_strength": "side-bias strength",
    "side_bias_theta": "side-bias angle",
    "side_bias_kappa": "side-bias concentration",
    "side_bias_inner_frac": "side-bias inner limit",

    # clustering
    "clustered_cell_frac": "clustered fraction",
    "cluster_size_range": "cluster size",
    "cluster_contact_factor_range": "cluster spacing",
    "cluster_chain_probability": "chain probability",
    "cluster_angle_jitter": "chain angle jitter",
    "cluster_packed_probability": "packed probability",
    "cluster_packed_contact_factor_range": "packed spacing",
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
        if np.isclose(xf, np.pi):
            return "π"
        if np.isclose(xf, 0.5 * np.pi):
            return "π/2"
        if np.isclose(xf, 1.5 * np.pi):
            return "3π/2"
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


def _merge_overrides(*dicts: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        if d:
            out.update(dict(d))
    return out


def _generate_param_showcase_row_seeded(
    sim_config: Any,
    camera: Optional[Any],
    sweep: Mapping[str, Sequence[Any]],
    base_overrides: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Build a parameter grid where each row has one fixed random scene.

    This is stricter than the old generate_param_showcase() behavior:
    columns within a row share the same seed, so visual changes are mostly
    caused by the swept parameter rather than random scene changes.
    """
    rng = np.random.default_rng(seed)

    base_kwargs = fdg.config_to_kwargs_image_sim(sim_config, rng, camera)
    if base_overrides:
        base_kwargs = fdg.merge_kwargs_image_sim(base_kwargs, base_overrides)

    base_kwargs.setdefault("H", 1024)
    base_kwargs.setdefault("W", 1024)
    base_kwargs.setdefault("n_cells", 800)
    base_kwargs.setdefault("cell_diameter", 8.0)
    base_kwargs.setdefault("return_targets", False)
    base_kwargs.setdefault("return_aux_targets", False)

    param_names = list(sweep.keys())
    value_lists = [list(sweep[p]) for p in param_names]

    n_rows = len(param_names)
    n_cols = max(len(vs) for vs in value_lists) if value_lists else 0

    if n_rows != 6:
        raise ValueError(
            f"Each figure must contain exactly 6 parameter rows; got {n_rows}."
        )
    if n_cols != 4:
        raise ValueError(
            f"Each parameter row must contain exactly 4 values; got {n_cols}."
        )
    for p, vals in zip(param_names, value_lists):
        if len(vals) != 4:
            raise ValueError(
                f"Parameter '{p}' must contain exactly 4 values; got {len(vals)}."
            )

    images: List[List[np.ndarray]] = [[None] * n_cols for _ in range(n_rows)]
    metas: List[List[Dict[str, Any]]] = [[None] * n_cols for _ in range(n_rows)]
    targets: List[List[Dict[str, Any]]] = [[None] * n_cols for _ in range(n_rows)]

    for r, p in enumerate(param_names):
        row_seed = int(seed + 9973 * r)
        vals = value_lists[r]

        for c, val in enumerate(vals):
            kwargs = dict(base_kwargs)
            kwargs["seed"] = row_seed
            kwargs[p] = val

            img, meta, tgt = fdg.simulate_image(**kwargs)

            images[r][c] = img
            metas[r][c] = meta
            targets[r][c] = tgt

    return {
        "images": images,
        "metas": metas,
        "targets": targets,
        "param_names": param_names,
        "value_lists": value_lists,
        "base_kwargs": base_kwargs,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


def _generate_main_figure(
    bundle: Dict[str, Any],
    *,
    figure_output_dir: str,
    figure_name: str,
    fig_scale: float = 1.0,
    annotate_mode: str = "textbox",
    cmap=None,
    dpi: int = 300,
) -> None:

    images = bundle["images"]
    param_names = bundle["param_names"]
    value_lists = bundle["value_lists"]
    n_rows = int(bundle["n_rows"])
    n_cols = int(bundle["n_cols"])

    if n_rows != 6 or n_cols != 4:
        raise ValueError(
            f"Expected a 6 x 4 grid, got {n_rows} x {n_cols}."
        )

    fig_w = cfg.FIGURE_WIDTH_FULL
    fig_h = cfg.FIGURE_HEIGHT_FULL

    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")

    outer: GridSpec = GridSpec(
        nrows=n_rows,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 0.18],
        height_ratios=[1] * n_rows,
        wspace=0.03,
        hspace=0.025,
    )

    for r in range(n_rows):
        param = param_names[r]
        param_label = _display_name(param)

        inner = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=n_cols,
            subplot_spec=outer[r, 0],
            wspace=0.012,
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

            value_label = _format_value(value_lists[r][c])
            tile_label = f"{param_label}: {value_label}"

            if annotate_mode == "title":
                ax.set_title(tile_label, fontsize=5.5, pad=1.5)
            else:
                ax.text(
                    0.02,
                    0.04,
                    tile_label,
                    transform=ax.transAxes,
                    fontsize=5.2,
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        fc="black",
                        ec="none",
                        alpha=0.60,
                    ),
                    ha="left",
                    va="bottom",
                )

        label_ax = fig.add_subplot(outer[r, 1])
        label_ax.axis("off")
        label_ax.text(
            0.02,
            0.50,
            param_label,
            ha="left",
            va="center",
            fontsize=7.0,
            rotation=0,
            alpha=0.90,
            wrap=True,
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
    sweep: Mapping[str, Sequence[Any]],
    base_overrides: Mapping[str, Any],
    figure_output_dir: str,
    figure_name: str,
    seed: int,
    fig_scale: float,
    annotate_mode: str,
    dpi: int,
) -> None:
    bundle = _generate_param_showcase_row_seeded(
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



def figure_SIM_generation(
    figure_output_dir: str,
    **kwargs,
):
    """
    Generate extended-data simulator parameter figures.

    Optional kwargs:
      seed: int
      fig_scale: float
      annotate_mode: "textbox" or "title"
      dpi: int
      sim_config: SimulatorConfig
      camera: CameraDimension
      only: optional iterable of names, e.g. {"Extended_Data_Figure_1"}
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
            W=1024,
            H=1024,
        ),
    )

    # Calm baseline for parameter figures.
    #
    # The simulator's calibrated diameter mode makes cells too large for these
    # 1024 x 1024 overview figures. Therefore this baseline disables calibrated
    # diameter bounds and uses a direct 4 px legacy diameter with no scaling.
    base_clean = dict(
        H=1024,
        W=1024,

        # geometry
        well_radius_frac=0.42,
        well_center_jitter=0.00,

        # background
        background_level=0.08,
        edge_boost=0.25,
        radial_gamma=1.20,
        vignette_strength=0.12,
        background_texture_enable=True,
        background_texture_sigma_fine=0.50,
        background_texture_sigma_coarse=1.80,
        background_texture_fine_weight=0.95,
        background_texture_coarse_weight=0.05,
        background_texture_strength=0.025,
        background_texture_clip=(0.1, 1.6),
        background_texture_downsample=4,
        background_texture_fullres_fine_strength=0.004,

        # cells
        n_cells=800,
        frac_positive=0.50,
        cell_diameter_bounds_by_short_side=None,
        cell_diameter=8.0,
        cell_diameter_reference_short_side=1024.0,
        cell_diameter_size_exponent=1.0,
        cell_diameter_scale_clip=(1.0, 1.0),
        large_cell_frac=0.04,
        large_cell_diameter_factor=1.5,
        cell_ellipse_enable=True,
        cell_axis_jitter=0.18,
        cell_random_rotation=True,
        cell_intensity_range=(0.70, 1.05),
        color_jitter=0.06,
        sigma_in=(0.06, 0.08),
        sigma_out=(0.10, 0.14),
        focus_frac_in=1.0,

        # placement
        rim_bias=0.75,
        rim_band=0.20,
        edge_clamp=0.28,
        min_cell_sep_px=None,
        rim_min_sep_px=4,
        pack_iters=15,
        pack_strength=0.45,
        wall_margin_px=4.0,

        # off by default unless shown
        cluster_enable=False,
        side_bias_enable=False,
        ghost_enable=True,
        reflect_enable=True,
        ghost_density=0.3,
        ghost_offset_px=20,
        ghost_sigma=(3.5, 3.5),
        ghost_intensity=(0.1, 0.1),
        dirt_density=0.0,
        ring_artifacts=0,

        # output speed
        return_targets=False,
        return_aux_targets=False,
    )

    jobs = []

    jobs.append(
        dict(
            figure_name="Extended_Data_Figure_1",
            sweep={
                "well_radius_frac": [0.32, 0.38, 0.44, 0.50],
                "well_center_jitter": [0.00, 0.05, 0.1, 0.15],
                "background_level": [0.00, 0.12, 0.24, 0.36],
                "edge_boost": [0.00, 0.15, 0.35, 0.60],
                "radial_gamma": [0.30, 1.00, 1.90, 3.00],
                "vignette_strength": [0.00, 0.18, 0.35, 0.7],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=650,
                    background_texture_strength=0.018,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Data_Figure_2",
            sweep={
                "n_cells": [150, 500, 1000, 1800],
                "frac_positive": [0.00, 0.25, 0.50, 1.00],
                "cell_diameter": [4.0, 8.0, 12.0, 15.0],
                "cell_axis_jitter": [0.00, 0.15, 0.35, 0.55],
                "color_jitter": [0.00, 0.05, 0.12, 0.25],
                "sigma_in": [(0.02, 0.02), (0.4, 0.46), (1.0, 1.0), (2.0, 2.0)],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=800,
                    cell_diameter=8.0,
                    large_cell_frac=0.0,
                    focus_frac_in=1.0,
                    background_texture_strength=0.018,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Data_Figure_3",
            sweep={
                "rim_bias": [0.10, 0.45, 0.75, 0.95],
                "rim_band": [0.08, 0.18, 0.35, 0.55],
                "edge_clamp": [0.00, 0.15, 0.35, 0.65],
                "rim_min_sep_px": [0, 4, 10, 20],
                "wall_margin_px": [0.0, 4.0, 10.0, 20.0],
                "pack_strength": [0.00, 0.25, 0.55, 0.90],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=1300,
                    cell_diameter=8.0,
                    rim_bias=0.75,
                    rim_band=0.25,
                    edge_clamp=0.25,
                    pack_iters=20,
                    background_texture_strength=0.018,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Data_Figure_4",
            sweep={
                "side_bias_strength": [0.00, 0.35, 0.70, 1.00],
                "side_bias_theta": [0.00, 0.50 * np.pi, np.pi, 1.50 * np.pi],
                "side_bias_kappa": [0.10, 1.00, 5.00, 12.00],
                "side_bias_inner_frac": [0.05, 0.25, 0.55, 0.85],
                "rim_bias": [0.30, 0.60, 0.85, 0.98],
                "edge_clamp": [0.00, 0.15, 0.35, 0.60],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=1200,
                    side_bias_enable=True,
                    side_bias_strength=0.80,
                    side_bias_theta=0.0,
                    side_bias_kappa=5.0,
                    side_bias_inner_frac=0.35,
                    rim_bias=0.85,
                    rim_band=0.35,
                    edge_clamp=0.20,
                    background_texture_strength=0.018,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Data_Figure_5",
            sweep={
                "clustered_cell_frac": [0.00, 0.30, 0.60, 0.90],
                "cluster_size_range": [(2, 4), (3, 8), (8, 16), (16, 35)],
                "cluster_contact_factor_range": [(1.50, 1.50), (1.20, 1.20), (1.05, 1.05), (0.95, 0.95)],
                "cluster_chain_probability": [0.00, 0.30, 0.65, 1.00],
                "cluster_packed_probability": [0.00, 0.35, 0.70, 1.00],
                "cluster_packed_region_join_probability": [0.00, 0.20, 0.45, 0.80],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=900,
                    cluster_enable=True,
                    clustered_cell_frac=0.60,
                    cluster_size_range=(3, 14),
                    cluster_contact_factor_range=(1.08, 1.25),
                    cluster_chain_probability=0.55,
                    cluster_angle_jitter=0.70,
                    cluster_packed_probability=0.55,
                    cluster_packed_contact_factor_range=(1.05, 1.20),
                    cluster_packed_region_join_probability=0.30,
                    cluster_pack_min_sep_factor=0.92,
                    rim_bias=0.65,
                    rim_band=0.25,
                    edge_clamp=0.20,
                    pack_iters=20,
                    background_texture_strength=0.018,
                ),
            ),
        )
    )

    jobs.append(
        dict(
            figure_name="Extended_Data_Figure_6",
            sweep={
                "ghost_density": [0.00, 0.20, 0.50, 1.00],
                "ghost_intensity": [(0.00, 0.00), (0.025, 0.025), (0.07, 0.07), (0.15, 0.15)],
                "ghost_stretch": [0.50, 1.20, 2.20, 3.50],
                "dirt_density": [0.00000, 0.00002, 0.00005, 0.00012],
                "reflect_n": [0, 2, 5, 10],
                "reflect_alpha_range": [(0.00, 0.00), (0.05, 0.05), (0.12, 0.12), (0.25, 0.25)],
            },
            base_overrides=_merge_overrides(
                base_clean,
                dict(
                    n_cells=650,
                    rim_bias=0.90,
                    rim_band=0.18,
                    edge_clamp=0.45,

                    ghost_enable=True,
                    ghost_density=0.35,
                    ghost_offset_px=25.0,
                    ghost_offset_jitter=5.0,
                    ghost_sigma=(3.0, 3.0),
                    ghost_dilate=1.0,
                    ghost_intensity=(0.06, 0.06),
                    ghost_stretch=2.2,
                    ghost_trail=2,
                    ghost_trail_decay=0.60,

                    dirt_density=0.00004,
                    dirt_size=(2, 6),
                    dirt_sigma=(0.8, 1.2),
                    dirt_alpha=(0.15, 0.40),

                    reflect_enable=True,
                    reflect_n=4,
                    reflect_theta_sigma=0.10,
                    reflect_radial_sigma=10.0,
                    reflect_offset_range=(20.0, 60.0),
                    reflect_alpha_range=(0.08, 0.08),
                    reflect_wobble=0.35,
                    reflect_harmonics=2,
                    reflect_harmonic_decay=0.55,

                    background_texture_strength=0.018,
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
