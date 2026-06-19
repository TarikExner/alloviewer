from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import figure_config as cfg
from . import figure_utils as utils

from alloviewer.image_analysis.io import load_image
from alloviewer.dev.segmentation.image_simulation import simulate_image
from alloviewer.dev.segmentation.image_simulation import (
    apply_camera_style,
    CameraStyleConfig,
    load_or_build_quantile_band_cache,
)
from alloviewer.dev.segmentation.image_simulation.camera_style_config import (
    STYLE_PARAMS_REGISTRY,
    with_histogram_adherence,
)
from .camera_style_utils import (
    get_feature_cache_path,
    collect_synthetic_feature_rows_from_dataset,
)

from ..segmentation.dataset_io import DiskSimCellsDataset


def crop_image(image, x, y, width, height):
    """
    Crop an image using top-left corner (x, y) and crop size.
    """
    h, w = image.shape[:2]

    if x < 0 or y < 0:
        raise ValueError("x and y must be >= 0")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    if x + width > w or y + height > h:
        raise ValueError("Crop region goes outside image bounds")

    return image[y:y + height, x:x + width]


def plot_rgb_histogram(
    ax: Axes,
    image: np.ndarray,
    bins: int = 256,
    density: bool = True,
    value_range: tuple[int, int] | tuple[float, float] | None = None,
    linewidth: float = 0.5,
) -> None:
    """
    Plot RGB histogram curves onto an existing axis.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if value_range is None:
        if np.issubdtype(image.dtype, np.integer):
            value_range = (0, 255)
        else:
            value_range = (0.0, 1.0)

    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["red", "green", "blue"]

    for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
        values = image[..., i].ravel()
        hist, bin_edges = np.histogram(
            values,
            bins=bins,
            range=value_range,
            density=density,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(
            bin_centers,
            hist,
            color=color,
            label=name,
            linewidth=linewidth,
        )

    ax.set_xlabel("Pixel intensity", fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel("Density" if density else "Count", fontsize=cfg.AXIS_LABEL_SIZE)
    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)
    ax.legend(frameon=False, fontsize=cfg.AXIS_LABEL_SIZE)


def _select_pca_features(
    feature_names: Sequence[str],
    normalized: bool,
    feature_subset: Optional[Sequence[str]] = None,
    drop_size_features: bool = True,
) -> list[str]:
    if feature_subset is not None:
        return list(feature_subset)

    if drop_size_features:
        if normalized:
            drop = {
                "height",
                "width",
                "aspect_ratio",
                "n_pixels_used",
                "sat_mean",
                "sat_std",
                "sat_skew",
                "dark_frac",
                "bright_frac",
            }
        else:
            drop = {
                "height",
                "width",
                "aspect_ratio",
                "n_pixels_used",
            }
    else:
        drop = set()
        if normalized:
            drop = {"dark_frac", "bright_frac"}

    return [f for f in feature_names if f not in drop]


def plot_real_and_synthetic_pca(
    ax: Axes,
    dataset,
    cache_path: str | Path = "",
    n_synthetic: int = 100,
    normalized: bool = True,
    feature_subset: Optional[Sequence[str]] = None,
    drop_size_features: bool = True,
    hist_bins: int = 16,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
    sample_pixels: Optional[int] = 300_000,
    rng_seed: int = 0,
    alpha_real: float = 0.45,
    alpha_syn: float = 0.80,
    s_real: float = 16,
    s_syn: float = 36,
    use_first_tile_only: bool = True,
    normalized_hist_range: Tuple[float, float] = (-3.0, 3.0),
    title: Optional[str] = None,
) -> None:
    """
    PCA comparison between real images and dataset images, plotted onto an axis.
    """
    final_cache_path = get_feature_cache_path(
        cache_path=cache_path,
        normalized=normalized,
    )

    if not final_cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {final_cache_path}")

    with open(final_cache_path, "rb") as f:
        payload = pickle.load(f)

    real_rows = payload["rows"]
    feature_names = payload["feature_names"]

    feature_names_used = _select_pca_features(
        feature_names=feature_names,
        normalized=normalized,
        feature_subset=feature_subset,
        drop_size_features=drop_size_features,
    )

    if not feature_names_used:
        raise ValueError("No features selected for PCA")

    synthetic_rows = collect_synthetic_feature_rows_from_dataset(
        dataset=dataset,
        n_synthetic=n_synthetic,
        hist_bins=hist_bins,
        percentiles=percentiles,
        sample_pixels=sample_pixels,
        rng_seed=rng_seed,
        use_first_tile_only=use_first_tile_only,
        normalized_features=normalized,
        normalized_hist_range=normalized_hist_range,
    )

    all_rows = list(real_rows) + list(synthetic_rows)
    X = np.array(
        [[row[f] for f in feature_names_used] for row in all_rows],
        dtype=np.float64,
    )

    finite_cols = np.all(np.isfinite(X), axis=0)
    if not finite_cols.all():
        feature_names_used = [
            f for f, keep in zip(feature_names_used, finite_cols)
            if keep
        ]
        X = X[:, finite_cols]

    if X.shape[0] < 2:
        raise ValueError("Need at least two images for PCA.")
    if X.shape[1] < 2:
        raise ValueError("Need at least two finite features for PCA.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=rng_seed)
    X_pca = pca.fit_transform(X_scaled)

    labels = [str(row["phone"]).lower() for row in all_rows]
    is_real = np.array(
        [not str(row["path"]).startswith("<synthetic_") for row in all_rows],
        dtype=bool,
    )

    device_order = [
        "iphone",
        "googlepixel",
        "microscope",
        "monochrome_real",
        "monochrome_generic",
        "simulated_raw",
        "synthetic",
    ]

    syn_markers = {
        "iphone": "x",
        "googlepixel": "^",
        "microscope": "s",
        "monochrome_real": "v",
        "monochrome_generic": "P",
        "simulated_raw": "D",
        "synthetic": "D",
    }

    for dev in device_order:
        idx = [i for i, lab in enumerate(labels) if lab == dev and is_real[i]]
        if idx:
            pts = X_pca[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                alpha=alpha_real,
                s=s_real,
                label=f"real {dev}",
            )

    for dev in device_order:
        idx = [i for i, lab in enumerate(labels) if lab == dev and not is_real[i]]
        if idx:
            pts = X_pca[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                alpha=alpha_syn,
                s=s_syn,
                marker=syn_markers.get(dev, "x"),
                label=f"synthetic {dev}",
            )

    ax.set_xlabel("PC1", fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel("PC2", fontsize=cfg.AXIS_LABEL_SIZE)

    if title is None:
        title = (
            "PCA of normalized real vs normalized synthetic image features"
            if normalized
            else "PCA of real vs denormalized synthetic image features"
        )

    ax.set_title(title, fontsize=cfg.TITLE_SIZE)

    handles, legend_labels = ax.get_legend_handles_labels()
    label_map = {
        "real iphone": cfg.PHONE_DICT['iPhone'],
        "real googlepixel": cfg.PHONE_DICT['GooglePixel'],
        "real microscope": cfg.PHONE_DICT['Microscope'],
        "real monochrome_real": cfg.PHONE_DICT['Monochrome'],
        "real monochrome_generic": "Monochrome generic",
        "synthetic iphone": f"Simulated {cfg.PHONE_DICT['iPhone']}",
        "synthetic googlepixel": f"Simulated {cfg.PHONE_DICT['GooglePixel']}",
        "synthetic microscope": f"Simulated {cfg.PHONE_DICT['Microscope']}",
        "synthetic monochrome_real": f"Simulated {cfg.PHONE_DICT['Monochrome']}",
        "synthetic monochrome_generic": f"Simulated {cfg.PHONE_DICT['Generic']}",
        "synthetic simulated_raw": "Simulated raw",
        "synthetic synthetic": "Simulated",
    }

    mapped_labels = [label_map.get(label, label) for label in legend_labels]
    ax.legend(
        handles,
        mapped_labels,
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        frameon=False,
        fontsize=cfg.AXIS_LABEL_SIZE,
    )

    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)


def _get_simulated_image():
    sim_img, _, targets = simulate_image(
        H=1024,
        W=1300,

        well_radius_frac=0.42,
        well_center_jitter=0.02,

        background_level=0.08,
        edge_boost=0.22,
        radial_gamma=1.2,
        vignette_strength=0.12,

        # cell count
        n_cells=900,

        # use the calibrated diameter model, not the old fixed diameter
        cell_diameter_bounds_by_short_side=(
            (1024.0, 7.0, 10.0),
            (1620.0, 7.0, 11.0),
            (3024.0, 11.0, 15.0),
        ),
        cell_diameter_center_margin_frac=0.20,
        cell_diameter_sigma_frac=0.18,
        cell_diameter_min_sigma_px=0.25,

        # legacy diameter fields are ignored when bounds are given,
        # but keep them harmless
        cell_diameter=10,

        large_cell_frac=0.0,
        large_cell_diameter_factor=1.5,

        cell_ellipse_enable=True,
        cell_axis_jitter=0.20,
        cell_random_rotation=True,
        cell_intensity_range=(0.70, 1.05),

        frac_positive=0.2,
        color_jitter=0.07,

        # these are fractions of the core diameter
        sigma_in=(0.06, 0.10),
        sigma_out=(0.08, 0.16),
        focus_frac_in=0.90,
        in_focus_sigma_thresh=None,

        boundary_width=1,

        rim_bias=0.60,
        rim_band=0.2,
        edge_clamp=0.35,

        # clustering
        cluster_enable=True,
        clustered_cell_frac=0.55,
        cluster_size_range=(2, 24),

        # slightly spaced, but still packed enough
        cluster_contact_factor_range=(1.00, 1.12),
        cluster_core_min_sep_factor=0.95,
        cluster_chain_probability=0.50,
        cluster_angle_jitter=0.85,

        # mix of packed and lengthy clusters
        cluster_packed_probability=0.55,
        cluster_packed_size_bias_range=(3, 15),
        cluster_packed_contact_factor_range=(0.98, 1.08),
        cluster_packed_candidate_count=8,
        cluster_packed_contact_bonus=1.5,
        cluster_packed_region_join_probability=0.25,
        cluster_packed_region_contact_factor_range=(1.00, 1.10),

        cluster_seed_tries=120,
        cluster_member_tries=32,
        cluster_pack_min_sep_factor=0.95,

        min_cell_sep_px=None,
        rim_min_sep_px=20,
        pack_iters=20,
        pack_strength=0.5,
        wall_margin_px=2.0,

        side_bias_enable=True,
        side_bias_theta=1.0,
        side_bias_strength=0.75,
        side_bias_kappa=5.0,
        side_bias_inner_frac=0.55,

        wall_blur_sigma=10.0,
        ring_artifacts=0,
        ring_sigma_range=(6.0, 14.0),
        ring_alpha_range=(0.02, 0.08),

        # keep ghosts, but make them less aggressive
        ghost_enable=True,
        ghost_density=0.03,
        ghost_offset_px=20.0,
        ghost_offset_jitter=4.0,
        ghost_sigma=(2.5, 5.0),
        ghost_dilate=1.0,
        ghost_intensity=(0.03, 0.08),
        ghost_stretch=2.0,
        ghost_trail=2,
        ghost_trail_decay=0.6,

        # keep dirt subtle
        dirt_density=0.00015,
        dirt_size=(2, 4),
        dirt_sigma=(0.8, 1.4),
        dirt_alpha=(0.01, 0.035),

        # reflections toned down
        reflect_enable=True,
        reflect_n=3,
        reflect_theta_sigma=0.08,
        reflect_radial_sigma=7.0,
        reflect_offset_range=(6.0, 18.0),
        reflect_alpha_range=(0.02, 0.08),
        reflect_wobble=0.25,
        reflect_harmonics=2,
        reflect_harmonic_decay=0.55,

        seed=187,
        return_targets=True,
    )

    return sim_img, targets


def _build_quantile_folders(ext_images_dir: str | Path) -> list[str]:
    ext_images_dir = Path(ext_images_dir)

    return [
        str(ext_images_dir / "20251106_25065441_iPhone_XR_JPEG"),
        str(ext_images_dir / "20251106_25722169_iPhone_XR_JPEG"),
        str(ext_images_dir / "20251106_25722269_iPhone_XR_JPEG"),
        str(ext_images_dir / "20251107_25065521_GooglePixel"),
        str(ext_images_dir / "20251107_25722332_GooglePixel"),
        str(ext_images_dir / "20251014_25719960"),
        str(ext_images_dir / "20251014_25720084"),
        str(ext_images_dir / "20251107_25065521"),
        str(ext_images_dir / "20251107_25722332"),
        str(ext_images_dir / "20260507_XM1_+DTT_mono_rgb"),
    ]


def _generate_main_figure(
    mic_img,
    gp_img,
    iphone_img,
    mono_img,
    sim_img,
    mic_adj,
    gp_adj,
    iphone_adj,
    mono_adj,
    ds,
    style_cache_path,
    figure_output_dir,
    figure_name,
) -> None:
    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(2, 5)

        orig_sim = fig.add_subplot(fig_sgs[0, 0])
        orig_sim.imshow(sim_img)
        orig_sim.set_title(f"{cfg.PHONE_DICT['Simulated']}", fontsize=cfg.TITLE_SIZE)

        mic = fig.add_subplot(fig_sgs[0, 1])
        mic.imshow(mic_img)
        mic.set_title(cfg.PHONE_DICT["Microscope"], fontsize=cfg.TITLE_SIZE)

        iphone = fig.add_subplot(fig_sgs[0, 3])
        iphone.imshow(iphone_img)
        iphone.set_title(cfg.PHONE_DICT["iPhone"], fontsize=cfg.TITLE_SIZE)

        gp = fig.add_subplot(fig_sgs[0, 2])
        gp.imshow(gp_img)
        gp.set_title(cfg.PHONE_DICT["GooglePixel"], fontsize=cfg.TITLE_SIZE)

        mono = fig.add_subplot(fig_sgs[0, 4])
        mono.imshow(mono_img)
        mono.set_title(cfg.PHONE_DICT["Monochrome"], fontsize=cfg.TITLE_SIZE)

        orig_hist = fig.add_subplot(fig_sgs[1, 0])
        plot_rgb_histogram(orig_hist, sim_img)
        orig_hist.set_title(f"{cfg.PHONE_DICT['Simulated']} histogram\n", fontsize=cfg.TITLE_SIZE)

        mic_hist = fig.add_subplot(fig_sgs[1, 1])
        plot_rgb_histogram(mic_hist, mic_img)
        mic_hist.set_title(f"{cfg.PHONE_DICT['Microscope']} histogram", fontsize=cfg.TITLE_SIZE)

        iphone_hist = fig.add_subplot(fig_sgs[1, 3])
        plot_rgb_histogram(iphone_hist, iphone_img)
        iphone_hist.set_title(f"{cfg.PHONE_DICT['iPhone']} histogram", fontsize=cfg.TITLE_SIZE)

        gp_hist = fig.add_subplot(fig_sgs[1, 2])
        plot_rgb_histogram(gp_hist, gp_img)
        gp_hist.set_title(f"{cfg.PHONE_DICT['GooglePixel']} histogram", fontsize=cfg.TITLE_SIZE)

        mono_hist = fig.add_subplot(fig_sgs[1, 4])
        plot_rgb_histogram(mono_hist, mono_img)
        mono_hist.set_title(f"{cfg.PHONE_DICT['Monochrome']} histogram", fontsize=cfg.TITLE_SIZE)

        for im_ax in (orig_sim, mic, gp, iphone, mono):
            utils.prep_image_axis(im_ax)

    def generate_subfigure_b(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1, 1)

        pca_plot = fig.add_subplot(fig_sgs[0])
        plot_real_and_synthetic_pca(
            pca_plot,
            ds,
            style_cache_path,
        )
        utils.remove_axis_labels(pca_plot)

    def generate_subfigure_c(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(2, 5)

        orig_sim = fig.add_subplot(fig_sgs[0, 0])
        orig_sim.imshow(sim_img)
        orig_sim.set_title(f"{cfg.PHONE_DICT['Simulated']}\n", fontsize=cfg.TITLE_SIZE)
        utils.prep_image_axis(orig_sim)

        mic_sim = fig.add_subplot(fig_sgs[0, 1])
        mic_sim.imshow(mic_adj)
        mic_sim.set_title(f"adjusted to\n{cfg.PHONE_DICT['Microscope']}", fontsize=cfg.TITLE_SIZE)
        utils.prep_image_axis(mic_sim)

        gp_sim = fig.add_subplot(fig_sgs[0, 2])
        gp_sim.imshow(gp_adj)
        gp_sim.set_title(f"adjusted to\n{cfg.PHONE_DICT['GooglePixel']}", fontsize=cfg.TITLE_SIZE)
        utils.prep_image_axis(gp_sim)

        iphone_sim = fig.add_subplot(fig_sgs[0, 3])
        iphone_sim.imshow(iphone_adj)
        iphone_sim.set_title(f"adjusted to\n{cfg.PHONE_DICT['iPhone']}", fontsize=cfg.TITLE_SIZE)
        utils.prep_image_axis(iphone_sim)

        mono_sim = fig.add_subplot(fig_sgs[0, 4])
        mono_sim.imshow(mono_adj)
        mono_sim.set_title(f"adjusted to\n{cfg.PHONE_DICT['Monochrome']}", fontsize=cfg.TITLE_SIZE)
        utils.prep_image_axis(mono_sim)

        orig_hist = fig.add_subplot(fig_sgs[1, 0])
        plot_rgb_histogram(orig_hist, sim_img)
        orig_hist.set_title(f"{cfg.PHONE_DICT['Simulated']} histogram\n", fontsize=cfg.TITLE_SIZE)

        mic_hist = fig.add_subplot(fig_sgs[1, 1])
        plot_rgb_histogram(mic_hist, mic_adj)
        mic_hist.set_title(f"{cfg.PHONE_DICT['Microscope']} histogram\n(simulated)", fontsize=cfg.TITLE_SIZE)

        gp_hist = fig.add_subplot(fig_sgs[1, 2])
        plot_rgb_histogram(gp_hist, gp_adj)
        gp_hist.set_title(f"{cfg.PHONE_DICT['GooglePixel']} histogram\n(simulated)", fontsize=cfg.TITLE_SIZE)

        iphone_hist = fig.add_subplot(fig_sgs[1, 3])
        plot_rgb_histogram(iphone_hist, iphone_adj)
        iphone_hist.set_title(f"{cfg.PHONE_DICT['iPhone']} histogram\n(simulated)", fontsize=cfg.TITLE_SIZE)

        mono_hist = fig.add_subplot(fig_sgs[1, 4])
        plot_rgb_histogram(mono_hist, mono_adj)
        mono_hist.set_title(f"{cfg.PHONE_DICT['Monochrome']} histogram\n(simulated)", fontsize=cfg.TITLE_SIZE)

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL * 1.15, cfg.FIGURE_HEIGHT_FULL * 1.08),
    )
    gs = GridSpec(
        ncols=1,
        nrows=3,
        figure=fig,
        height_ratios=[1, 1, 1],
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

    return


def figure_2_generation(
    validation_results_dir: str,
    ext_images_dir,
    figure_output_dir,
    **kwargs,
):
    ds = DiskSimCellsDataset(
        os.path.join(validation_results_dir, "test_ds_diverse_mc.ds")
    )

    style_cache_path = os.path.join(validation_results_dir, "style_cache.cache")

    mic, _ = load_image(
        "Bild_3139.tif",
        base_dir=os.path.join(ext_images_dir, "20251106_25065441/"),
        as_chw=False,
    )
    mic = crop_image(mic, 150, 0, 1750, 1500)

    iphone, _ = load_image(
        "IMG_3857.jpeg",
        base_dir=os.path.join(ext_images_dir, "20251106_25065441_iPhone_XR_JPEG/"),
        as_chw=False,
    )
    iphone = crop_image(iphone, 1000, 100, 2700, 2500)

    gp, _ = load_image(
        "PXL_20251107_130200415.jpg",
        base_dir=os.path.join(ext_images_dir, "20251107_25065521_GooglePixel/"),
        as_chw=False,
    )
    gp = np.transpose(gp, (1, 0, 2))
    gp = crop_image(gp, 1100, 300, 2900, 2700)

    mono, _ = load_image(
        "auto1_4b.tif",
        base_dir=os.path.join(ext_images_dir, "20260504_Auto1_mono_rgb"),
        as_chw=False,
    )

    sim_img, sim_targets = _get_simulated_image()
    cell_mask = None
    if isinstance(sim_targets, dict) and "cell_mask" in sim_targets:
        cell_mask = sim_targets["cell_mask"].astype(np.float32)

    q_band_cache = load_or_build_quantile_band_cache(
        folders=_build_quantile_folders(ext_images_dir),
        annotations_dir="../scripts/region_annotations",
        cache_path=os.path.join(validation_results_dir, "camera_quantile_band_cache.pkl"),
    )

    style_registry_strict = with_histogram_adherence(
        STYLE_PARAMS_REGISTRY,
        mode="figure",
    )

    iphone_adj = apply_camera_style(
        sim_img,
        rng=np.random.default_rng(187),
        style_cfg=CameraStyleConfig(("iphone",)),
        style_registry=style_registry_strict,
        quantile_band_cache=q_band_cache,
        cell_mask=cell_mask,
    )

    gp_adj = apply_camera_style(
        sim_img,
        rng=np.random.default_rng(187),
        style_cfg=CameraStyleConfig(("googlepixel",)),
        style_registry=style_registry_strict,
        quantile_band_cache=q_band_cache,
        cell_mask=cell_mask,
    )

    mic_adj = apply_camera_style(
        sim_img,
        rng=np.random.default_rng(187),
        style_cfg=CameraStyleConfig(("microscope",)),
        style_registry=style_registry_strict,
        quantile_band_cache=q_band_cache,
        cell_mask=cell_mask,
    )

    mono_adj = apply_camera_style(
        sim_img,
        rng=np.random.default_rng(187),
        style_cfg=CameraStyleConfig(("monochrome_real",)),
        style_registry=style_registry_strict,
        quantile_band_cache=q_band_cache,
        cell_mask=cell_mask,
    )

    _generate_main_figure(
        mic_img=mic,
        gp_img=gp,
        iphone_img=iphone,
        mono_img=mono,
        sim_img=sim_img,
        mic_adj=mic_adj,
        gp_adj=gp_adj,
        iphone_adj=iphone_adj,
        mono_adj=mono_adj,
        ds=ds,
        style_cache_path=style_cache_path,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_2",
    )
