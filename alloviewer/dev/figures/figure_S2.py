
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, feature

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import figure_config as cfg
from . import figure_utils as utils

from ..segmentation.dataset_io import DiskSimCellsDataset

from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

from alloviewer.image_analysis.io import load_image
from alloviewer.dev.segmentation.image_simulation import simulate_image
from alloviewer.dev.segmentation.config import SimulatorConfig, CameraSetup
from alloviewer.dev.segmentation.camera_style_utils import get_feature_cache_path, collect_synthetic_feature_rows_from_dataset

from alloviewer.dev.segmentation.image_simulation import apply_camera_style
from alloviewer.dev.segmentation.camera_styles import CameraStyleParams, CameraStyleConfig
from alloviewer.dev.segmentation.camera_styles import load_or_build_quantile_band_cache

from alloviewer.dev.segmentation.dataset_io import DiskSimCellsDataset

def crop_image(image, x, y, width, height):
    """
    Crop an image using top-left corner (x, y) and crop size.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    x : int
        Left coordinate.
    y : int
        Top coordinate.
    width : int
        Crop width.
    height : int
        Crop height.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    h, w = image.shape[:2]

    if x < 0 or y < 0:
        raise ValueError("x and y must be >= 0")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    if x + width > w or y + height > h:
        raise ValueError("Crop region goes outside image bounds")

    return image[y:y+height, x:x+width]

def plot_rgb_histogram(
    ax: Axes,
    image: np.ndarray,
    bins: int = 256,
    density: bool = True,
    value_range: tuple[int, int] | None = None,
    linewidth: float = 0.5,
) -> None:
    """
    Plot RGB histogram curves onto an existing axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis to draw on.
    image : np.ndarray
        RGB image with shape (H, W, 3).
    bins : int
        Number of histogram bins.
    density : bool
        If True, plot normalized histogram densities.
    value_range : tuple[int, int] | None
        Histogram range. If None, inferred from dtype.
    linewidth : float
        Line width for the curves.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if value_range is None:
        if np.issubdtype(image.dtype, np.integer):
            value_range = (0, 255)
        else:
            value_range = (0, 1)

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
        ax.plot(bin_centers, hist, color=color, label=name, linewidth=linewidth)

    ax.set_xlabel("Pixel intensity", fontsize = cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel("Density" if density else "Count", fontsize = cfg.AXIS_LABEL_SIZE)
    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)
    ax.legend(frameon=False, fontsize = cfg.AXIS_LABEL_SIZE)

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
):
    """
    PCA comparison between real images and dataset images, plotted onto an existing axis.

    Parameters
    ----------
    ax
        Matplotlib axis to draw on.

    normalized
        False:
            real cache is image-space cache
            dataset images are denormalized before feature extraction

        True:
            real cache is normalized-image cache
            dataset images are used as stored
    """
    final_cache_path = get_feature_cache_path(cache_path=cache_path, normalized=normalized)

    if not final_cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {final_cache_path}")

    with open(final_cache_path, "rb") as f:
        payload = pickle.load(f)

    real_rows = payload["rows"]
    feature_names = payload["feature_names"]

    if feature_subset is None:
        if drop_size_features:
            if normalized:
                drop = {
                    "height", "width", "aspect_ratio", "n_pixels_used",
                    "sat_mean", "sat_std", "sat_skew"
                }
            else:
                drop = {"height", "width", "aspect_ratio", "n_pixels_used"}

            feature_names_used = [f for f in feature_names if f not in drop]
        else:
            feature_names_used = list(feature_names)

        if normalized:
            drop_norm_only = {"dark_frac", "bright_frac"}
            feature_names_used = [f for f in feature_names_used if f not in drop_norm_only]
    else:
        feature_names_used = list(feature_subset)

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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=rng_seed)
    X_pca = pca.fit_transform(X_scaled)

    labels = [str(row["phone"]).lower() for row in all_rows]
    is_real = np.array([not str(row["path"]).startswith("<synthetic_") for row in all_rows])

    real_devices = ["iphone", "googlepixel", "microscope"]
    syn_devices = ["iphone", "googlepixel", "microscope", "synthetic"]

    for dev in real_devices:
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

    syn_markers = {
        "iphone": "x",
        "googlepixel": "^",
        "microscope": "s",
        "synthetic": "D",
    }

    for dev in syn_devices:
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


    ax.set_xlabel("PC1", fontsize = cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel("PC2", fontsize = cfg.AXIS_LABEL_SIZE)

    if title is None:
        title = (
            "PCA of normalized real vs normalized synthetic image features"
            if normalized
            else "PCA of real vs denormalized synthetic image features"
        )

    ax.set_title(title, fontsize = cfg.TITLE_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "real iphone": "iPhone",
        "real googlepixel": "GooglePixel",
        "real microscope": "Microscope",
        "synthetic synthetic": "Simulated"
    }
    labels = [label_map[label] for label in labels]
    ax.legend(handles, labels, bbox_to_anchor = (1.05, 0.5), loc = "center left")

    return 

def _get_simulated_image():
    sim_img, _, _ = simulate_image(
        H=1024,
        W=1300,
        well_radius_frac=0.42,
        well_center_jitter=0.02,

        # --- radial look of the well ---
        background_level=0.08,
        edge_boost=0.25,
        radial_gamma=1.2,
        vignette_strength=0.20,

        # --- color mix ---
        bg_hue=0.25,             # 0=orange, 1=green

        # --- cells (sharp ones inside the well) ---
        n_cells=2000,
        cell_diameter=10,

        large_cell_frac=0.0,              # fraction of inside-well cells that are "large"
        large_cell_diameter_factor=1.5,   # large size = factor * cell_diameter

        # --- cells: shape + brightness ---
        cell_ellipse_enable=True,
        cell_axis_jitter=0.20,          # ±20% axis ratio
        cell_random_rotation=True,      # random rotation angle
        cell_intensity_range=(0.70, 1.05),  # per-cell brightness multiplier (was ~0.9..1.1)

        frac_positive=0.01,
        color_jitter=0.07,
        sigma_in=(0.5, 1.0),
        sigma_out=(1.6, 3.2),    # used if focus_frac_in<1
        focus_frac_in=1.0,       # default: draw sharp; ghosts carry the blur
        in_focus_sigma_thresh=None,
        boundary_width=1,

        # crowd cells near the *outer* wall, but keep a filled center
        rim_bias=0.85,
        rim_band=0.3,
        edge_clamp=0.5,

        # --- collision / packing control ---
        min_cell_sep_px=None,   # if None -> 0.9 * cell_diameter
        rim_min_sep_px=12,
        pack_iters=20,          # fewer iters thanks to vectorized packing
        pack_strength=0.5,     # 0..1, how far to push per step
        wall_margin_px=2.0,     # keep centers this far from the wall

        # --- sidedness (pile-up on one side near the rim) ---
        side_bias_enable=True,     # set True to activate
        side_bias_theta=1.0,        # radians; 0=right, +pi/2=up, pi=left, -pi/2=down
        side_bias_strength=0.75,    # 0..1 mixture with uniform (higher = stronger bias)
        side_bias_kappa=5.0,        # von Mises concentration at the rim (higher = tighter)
        side_bias_inner_frac=0.55,  # start fading bias below this fraction of R (center stays even)

        # --- visual wall (soft rim) ---
        wall_blur_sigma=12.0,
        ring_artifacts=0,
        ring_sigma_range=(6.0, 18.0),
        ring_alpha_range=(0.03, 0.12),

        # --- “ghost cells” OUTSIDE the well (big, elongated, not in masks) ---
        ghost_enable=True,
        ghost_density=0.10,      # fraction relative to number of rim cells
        ghost_offset_px=25.0,
        ghost_offset_jitter=6.0,
        ghost_sigma=(2.5, 6.0),  # base sigma (minor axis)
        ghost_dilate=1.0,
        ghost_intensity=(0.1, 0.1),

        # NEW: outward elongation and short trail
        ghost_stretch=3.0,       # major/minor axis ratio (>1 stretches outward)
        ghost_trail=3,           # number of faded lobes outward
        ghost_trail_decay=0.6,   # amplitude decay per lobe (0..1)

        # --- debris INSIDE the well (small + dim) ---
        dirt_density=0.0007,
        dirt_size=(2, 4),
        dirt_sigma=(1.2, 2.0),
        dirt_alpha=(0.01, 0.04),

        # --- noise / camera ---
        blur_sigma_global=0.0,
        photon_level=2500,
        read_noise=0.003,

        # --- radial reflections on the wall (outside the well) ---
        reflect_enable=True,
        reflect_n=6,                 # number of streak groups
        reflect_theta_sigma=0.10,    # angular width of a streak (radians)
        reflect_radial_sigma=8.0,    # radial softness (pixels)
        reflect_offset_range=(6.0, 24.0),   # how far outside R the streak sits
        reflect_alpha_range=(0.05, 0.20),   # strength
        reflect_wobble=0.35,         # small angular wiggle per streak (radians)
        reflect_harmonics=2,         # add faint copies to get a comb feel
        reflect_harmonic_decay=0.55, # falloff for those copies

        seed=None,
        return_targets=True,
    )
    return sim_img

IPHONE_STYLE = CameraStyleParams(
    name="iphone",
    exposure_range=(1.2, 1.2),
    c_range=(0.90, 0.99),
    b_range=(0.01, 0.035),
    gamma_range=(1.00, 1.03),

    shadow_lift_range=(0.05, 0.10),
    highlight_rolloff_range=(0.10, 0.18),
    midtone_contrast_range=(-0.03, 0.02),

    mix_range=(0.01, 0.04),
    wb_range=(0.98, 1.03),
    saturation_range=(0.76, 0.96),
    green_magenta_shift_range=(-0.015, 0.015),
    blue_yellow_shift_range=(-0.015, 0.015),

    blur_sigma_range=(0.18, 0.45),
    sharpen_strength_range=(0.02, 0.08),
    noise_std_base_range=(0.002, 0.005),

    vignette_amp_range=(0.00, 0.02),
    illum_amp_range=(0.00, 0.015),

    clip_prob=0.00,
    jpeg_prob=0.3,
    jpeg_quality_range=(80, 100),

    resize_prob=0.00,
    resize_scale_range=(0.95, 1.00),

    histogram_match_strength_range=(0.99, 1.00),
    use_histogram_match=True,

    median_match_strength=(0.7, 0.7)

)

GOOGLEPIXEL_STYLE = CameraStyleParams(
    name="googlepixel",
    exposure_range=(1.2, 1.2),
    c_range=(0.90, 1.10),
    b_range=(-0.01, 0.03),
    gamma_range=(0.90, 1.08),

    shadow_lift_range=(0.02, 0.08),
    highlight_rolloff_range=(0.03, 0.10),
    midtone_contrast_range=(-0.02, 0.10),

    mix_range=(0.02, 0.10),
    wb_range=(0.94, 1.08),
    saturation_range=(0.75, 1.00),
    green_magenta_shift_range=(-0.04, 0.04),
    blue_yellow_shift_range=(-0.05, 0.05),

    blur_sigma_range=(0.35, 1.00),
    sharpen_strength_range=(0.12, 0.40),
    noise_std_base_range=(0.004, 0.012),

    vignette_amp_range=(0.02, 0.08),
    illum_amp_range=(0.02, 0.06),

    clip_prob=0.08,
    jpeg_prob=0.3,
    jpeg_quality_range=(80, 100),

    resize_prob=0.10,
    resize_scale_range=(0.80, 0.96),

    histogram_match_strength_range=(0.99, 1.00),
    use_histogram_match=True,

    median_match_strength=(0.7, 0.7)
)

MICROSCOPE_STYLE = CameraStyleParams(
    name="microscope",
    exposure_range=(0.76, 0.98),
    c_range=(0.90, 1.02),
    b_range=(-0.03, 0.005),
    gamma_range=(0.98, 1.12),

    shadow_lift_range=(0.00, 0.015),
    highlight_rolloff_range=(0.00, 0.025),
    midtone_contrast_range=(-0.04, 0.025),

    mix_range=(0.00, 0.015),
    wb_range=(0.985, 1.015),
    saturation_range=(0.24, 0.52),
    green_magenta_shift_range=(-0.01, 0.01),
    blue_yellow_shift_range=(-0.05, -0.01),

    blur_sigma_range=(0.08, 0.28),
    sharpen_strength_range=(0.00, 0.035),
    noise_std_base_range=(0.0002, 0.0012),

    vignette_amp_range=(0.00, 0.008),
    illum_amp_range=(0.00, 0.01),

    clip_prob=0.00,
    jpeg_prob=0.3,
    jpeg_quality_range=(80, 100),

    resize_prob=0.0,
    resize_scale_range=(1.0, 1.0),

    histogram_match_strength_range=(0.99, 1.00),
    use_histogram_match=True,

    median_match_strength=(0.7, 0.7)
)

SIMULATED_RAW_STYLE = CameraStyleParams(
    name="simulated_raw",
    exposure_range=(1.0, 1.0),
    c_range=(1.0, 1.0),
    b_range=(0.0, 0.0),
    gamma_range=(1.0, 1.0),

    shadow_lift_range=(0.0, 0.0),
    highlight_rolloff_range=(0.0, 0.0),
    midtone_contrast_range=(0.0, 0.0),

    mix_range=(0.0, 0.0),
    wb_range=(1.0, 1.0),
    saturation_range=(1.0, 1.0),
    green_magenta_shift_range=(0.0, 0.0),
    blue_yellow_shift_range=(0.0, 0.0),

    blur_sigma_range=(0.0, 0.0),
    sharpen_strength_range=(0.0, 0.0),
    noise_std_base_range=(0.0, 0.0),

    vignette_amp_range=(0.0, 0.0),
    illum_amp_range=(0.0, 0.0),

    clip_prob=0.0,
    jpeg_prob=0.3,
    jpeg_quality_range=(80, 100),

    resize_prob=0.0,
    resize_scale_range=(1.0, 1.0),

    histogram_match_strength_range=(0.0, 0.0),
    use_histogram_match=False,

    median_match_strength = (0.0, 0.0)
)

STYLE_PARAMS_REGISTRY: Dict[str, CameraStyleParams] = {
    "iphone": IPHONE_STYLE,
    "googlepixel": GOOGLEPIXEL_STYLE,
    "microscope": MICROSCOPE_STYLE,
    "simulated_raw": SIMULATED_RAW_STYLE,
}

def _generate_main_figure(
    mic_img,
    gp_img,
    iphone_img,
    sim_img,
    mic_adj,
    gp_adj,
    iphone_adj,
    ds,
    style_cache_path,
    figure_output_dir,
    figure_name
) -> None:
    """
    img: np.ndarray [H,W] or [H,W,3]
    seg_out: dict from SegmenterUNet (tiles/batched, with 'probs' and shape [T,H,W])
    inst_cfg: InstanceSegmenterConfig
    inset_start: (row, col) start of inset crop for display.
    inset_size: side length of square inset (default 50).

    All analysis is done on full-size arrays; cropping is only applied for display.
    """


    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(2, 4)

        orig_sim = fig.add_subplot(fig_sgs[0,0])
        orig_sim.imshow(sim_img)
        orig_sim.set_title("Simulated", fontsize = cfg.TITLE_SIZE)

        mic = fig.add_subplot(fig_sgs[0,1])
        mic.imshow(mic_img)
        mic.set_title("Microscope", fontsize = cfg.TITLE_SIZE)
        
        gp = fig.add_subplot(fig_sgs[0,2])
        gp.imshow(gp_img)
        gp.set_title("GooglePixel", fontsize = cfg.TITLE_SIZE)

        iphone = fig.add_subplot(fig_sgs[0,3])
        iphone.imshow(iphone_img)
        iphone.set_title("iPhone", fontsize = cfg.TITLE_SIZE)

        orig_hist = fig.add_subplot(fig_sgs[1,0])
        plot_rgb_histogram(orig_hist, sim_img)
        orig_hist.set_title("Simulated histogram\n", fontsize = cfg.TITLE_SIZE)
        
        mic_hist = fig.add_subplot(fig_sgs[1,1])
        plot_rgb_histogram(mic_hist, mic_img)
        mic_hist.set_title("Microscope histogram", fontsize = cfg.TITLE_SIZE)

        gp_hist = fig.add_subplot(fig_sgs[1,2])
        plot_rgb_histogram(gp_hist, gp_img)
        gp_hist.set_title("GooglePixel histogram", fontsize = cfg.TITLE_SIZE)
    
        iphone_hist = fig.add_subplot(fig_sgs[1,3])
        plot_rgb_histogram(iphone_hist, iphone_img)
        iphone_hist.set_title("iPhone histogram", fontsize = cfg.TITLE_SIZE)
        
        utils.prep_image_axis(orig_sim)
        utils.prep_image_axis(mic)
        utils.prep_image_axis(gp)
        utils.prep_image_axis(iphone)

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(1,1)

        pca_plot = fig.add_subplot(fig_sgs[0])
        plot_real_and_synthetic_pca(
            pca_plot,
            ds,
            style_cache_path
        )
        utils.remove_axis_labels(pca_plot)

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label: str
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        fig_sgs = gs.subgridspec(2,4)

        orig_sim = fig.add_subplot(fig_sgs[0,0])
        orig_sim.imshow(sim_img)
        orig_sim.set_title("Simulated", fontsize = cfg.TITLE_SIZE)
        utils.prep_image_axis(orig_sim)

        mic_sim = fig.add_subplot(fig_sgs[0,1])
        mic_sim.imshow(mic_adj)
        mic_sim.set_title("Adjusted to Microscope", fontsize = cfg.TITLE_SIZE)
        utils.prep_image_axis(mic_sim)

        gp_sim = fig.add_subplot(fig_sgs[0,2])
        gp_sim.imshow(gp_adj)
        gp_sim.set_title("Adjusted to GooglePixel", fontsize = cfg.TITLE_SIZE)
        utils.prep_image_axis(gp_sim)

        iphone_sim = fig.add_subplot(fig_sgs[0,3])
        iphone_sim.imshow(iphone_adj)
        iphone_sim.set_title("Adjusted to iPhone", fontsize = cfg.TITLE_SIZE)
        utils.prep_image_axis(iphone_sim)


        orig_hist = fig.add_subplot(fig_sgs[1,0])
        plot_rgb_histogram(orig_hist, sim_img)
        orig_hist.set_title("Simulated histogram\n", fontsize = cfg.TITLE_SIZE)
        
        mic_hist = fig.add_subplot(fig_sgs[1,1])
        plot_rgb_histogram(mic_hist, mic_adj)
        mic_hist.set_title("Microscope histogram\n(simulated)", fontsize = cfg.TITLE_SIZE)

        gp_hist = fig.add_subplot(fig_sgs[1,2])
        plot_rgb_histogram(gp_hist, gp_adj)
        gp_hist.set_title("GooglePixel histogram\n(simulated)", fontsize = cfg.TITLE_SIZE)
    
        iphone_hist = fig.add_subplot(fig_sgs[1,3])
        plot_rgb_histogram(iphone_hist, iphone_adj)
        iphone_hist.set_title("iPhone histogram\n(simulated)", fontsize = cfg.TITLE_SIZE)
        
        return
        
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )
    gs = GridSpec(
        ncols=1,
        nrows=3,
        figure=fig,
        height_ratios=[1,1,1],
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


def figure_S2_generation(validation_results_dir: str,
                         ext_images_dir,
                         figure_output_dir,
                         **kwargs):


    ds = DiskSimCellsDataset(
        os.path.join(validation_results_dir, "test_ds_phonemix.ds")
    )
    style_cache_path = os.path.join(validation_results_dir, "style_cache.cache")

    mic, _ = load_image(
        "Bild_3139.tif",
        base_dir = os.path.join(ext_images_dir, "20251106_25065441/"),
        as_chw=False
    )
    mic = crop_image(mic, 150, 0, 1750, 1500)

    iphone, _ = load_image(
        "IMG_3857.jpeg",
        base_dir = os.path.join(ext_images_dir, "20251106_25065441_iPhone_XR_JPEG/"),
        as_chw=False
    )
    iphone = crop_image(iphone, 1000, 100, 2700, 2500)

    gp, _ = load_image(
        "PXL_20251107_130200415.jpg",
        base_dir = os.path.join(ext_images_dir, "20251107_25065521_GooglePixel/"), 
        as_chw=False
    )
    gp = np.transpose(gp, (1,0,2))
    gp = crop_image(gp, 1100, 300, 2900, 2700)


    sim_img = _get_simulated_image()

    q_band_cache = load_or_build_quantile_band_cache(
        folders = None,
        cache_path = os.path.join(validation_results_dir, "quantile_band_cache.pkl")
    )
        
    cam_adj = apply_camera_style(sim_img, rng = np.random.default_rng(187), style_cfg = CameraStyleConfig(("googlepixel",)), style_registry = STYLE_PARAMS_REGISTRY,
                                 quantile_band_cache = q_band_cache)

    iphone_adj = apply_camera_style(
        sim_img,#
        rng = np.random.default_rng(187),
        style_cfg = CameraStyleConfig(("iphone",)),
        style_registry = STYLE_PARAMS_REGISTRY,
        quantile_band_cache = q_band_cache
    )
    gp_adj = apply_camera_style(
        sim_img,#
        rng = np.random.default_rng(187),
        style_cfg = CameraStyleConfig(("googlepixel",)),
        style_registry = STYLE_PARAMS_REGISTRY,
        quantile_band_cache = q_band_cache
    )
    mic_adj = apply_camera_style(
        sim_img,#
        rng = np.random.default_rng(187),
        style_cfg = CameraStyleConfig(("microscope",)),
        style_registry = STYLE_PARAMS_REGISTRY,
        quantile_band_cache = q_band_cache
    )

    _generate_main_figure(
        mic_img = mic,
        gp_img = gp,
        iphone_img = iphone,
        sim_img = sim_img,
        mic_adj = mic_adj,
        gp_adj = gp_adj,
        iphone_adj = iphone_adj,
        ds = ds,
        style_cache_path = style_cache_path,
        figure_output_dir = figure_output_dir,
        figure_name = "Figure S2"
    )


