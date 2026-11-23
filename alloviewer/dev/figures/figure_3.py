import os
import copy
from typing import Any, Optional, Literal
from pathlib import Path
import csv
from itertools import chain

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import ndimage as ndi
from skimage import measure, morphology

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from alloviewer.image_analysis.config import UNET_CONFIG, INSTANCE_CONFIG
from alloviewer.image_analysis.segmenter import (
    SegmenterUNet,
    InstanceSegmenter,
    InstanceSegmenterConfig,
)
from alloviewer.image_analysis.utils import (
    as_contiguous_f32,
    hysteresis_mask,
    make_markers,
    smooth01,
)

from alloviewer.dev.figures import figure_config as cfg
from alloviewer.dev.figures import figure_utils as utils

from alloviewer.dev.figures.figure_data_generation import (
    get_validation_data,
    generate_unet_comparison,
    load_or_create_figure_1_image_cache as load_or_create_multimodal_image_cache,
    crop_square,
    prepare_image,
)


SCATTER_KWARGS = {
    "s": 4,
    "edgecolor": "black",
    "linewidth": 0.3,
}


INSET_SIDE_LENGTH = 128
INSET_LINEWIDTH = 2
INSET_RECT_COLOR = "red"

INSET_WIDTH = "50%"
INSET_HEIGHT = "50%"
INSET_LOCATION = "upper right"
INSET_BORDER_COLOR = "black"
INSET_BORDER_LINEWIDTH = 2


INSET_COORDS = {
    "simulated_image": (250, 120),
    "simulated_segmentation": (250, 120),

    "microscopy_image": (340, 240),
    "microscopy_segmentation": (340, 240),

    "googlepixel_image": (70, 200),
    "googlepixel_segmentation": (70, 200),

    "iphone_image": (60, 650),
    "iphone_segmentation": (60, 650),

    "monochrome_image": (140, 498),
    "monochrome_segmentation": (140, 498),
}

PANEL_A_MICROSCOPY_PIXEL_SIZE_UM = 1.32
PANEL_A_SCALE_BAR_UM = 50.0
PANEL_A_SCALE_BAR_MARGIN_PX = 6
PANEL_A_SCALE_BAR_THICKNESS_PX = 4


def _add_scale_bar_to_image(
    image: np.ndarray,
    *,
    length_um: float = PANEL_A_SCALE_BAR_UM,
    pixel_size_um: float = PANEL_A_MICROSCOPY_PIXEL_SIZE_UM,
    margin_px: int = PANEL_A_SCALE_BAR_MARGIN_PX,
    thickness_px: int = PANEL_A_SCALE_BAR_THICKNESS_PX,
) -> np.ndarray:
    """Return a copy of `image` with a white bottom-right scale bar."""
    out = np.asarray(image).copy()

    if out.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or HWC image, got shape {out.shape}.")

    h, w = out.shape[:2]
    length_px = max(1, int(round(length_um / pixel_size_um)))

    if length_px + 2 * margin_px > w:
        raise ValueError(
            f"Scale bar is too wide for image width {w}: "
            f"length_px={length_px}, margin_px={margin_px}."
        )

    x1 = w - int(margin_px)
    x0 = x1 - length_px
    y1 = h - int(margin_px)
    y0 = y1 - int(thickness_px)

    white = 255 if np.issubdtype(out.dtype, np.integer) else 1.0

    if out.ndim == 2:
        out[y0:y1, x0:x1] = white
    else:
        out[y0:y1, x0:x1, :3] = white
        if out.shape[-1] == 4:
            out[y0:y1, x0:x1, 3] = white

    return out


def _as_instance_cfg(cfg_inst: Any) -> Any:
    if isinstance(cfg_inst, InstanceSegmenterConfig):
        return cfg_inst

    if isinstance(cfg_inst, dict):
        return InstanceSegmenterConfig.from_dict(cfg_inst)

    return cfg_inst


def _as_2d_prob(x: np.ndarray, name: str) -> np.ndarray:
    x = as_contiguous_f32(np.asarray(x))

    if x.ndim == 2:
        return x

    if x.ndim == 3 and x.shape[0] == 1:
        return as_contiguous_f32(x[0])

    raise ValueError(f"Expected {name} as [H,W], got shape {x.shape}.")


def _make_seg_out_from_probs(probs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "probs": {
            "cell": _as_2d_prob(probs["cell"], "cell"),
            "bound": _as_2d_prob(probs["bound"], "bound"),
            "center": _as_2d_prob(probs["center"], "center"),
            "energy": _as_2d_prob(probs["energy"], "energy"),
        },
        "cell_mask": None,
        "boundary": None,
        "instance_labels": None,
        "meta": {},
    }


def compute_instance_steps(
    seg_out: dict[str, Any],
    cfg_inst: Any,
) -> dict[str, np.ndarray]:
    cfg_inst = _as_instance_cfg(cfg_inst)

    p_cell = _as_2d_prob(seg_out["probs"]["cell"], "cell")
    p_bound = _as_2d_prob(seg_out["probs"]["bound"], "bound")
    p_center = _as_2d_prob(seg_out["probs"]["center"], "center")
    p_energy = _as_2d_prob(seg_out["probs"]["energy"], "energy")

    close_selem = (
        morphology.disk(int(cfg_inst.mask_close_radius))
        if int(cfg_inst.mask_close_radius) > 0
        else None
    )

    mask = hysteresis_mask(
        p_cell=p_cell,
        low_thr=cfg_inst.cell_mask_low_thr,
        high_thr=cfg_inst.cell_mask_high_thr,
        close_selem=close_selem,
        min_hole_area=cfg_inst.min_hole_area,
        min_object_area=cfg_inst.min_object_area,
    )

    dist = ndi.distance_transform_edt(mask).astype(np.float32)

    if cfg_inst.distance_smooth_sigma > 0:
        ndi.gaussian_filter(
            dist,
            float(cfg_inst.distance_smooth_sigma),
            output=dist,
        )

    dist_s = dist

    height, width = p_cell.shape
    elevation = np.zeros((height, width), dtype=np.float32)
    tmp = np.empty_like(elevation, dtype=np.float32)

    if cfg_inst.distance_weight != 0:
        dmax = dist_s.max()
        if dmax > 1e-6:
            np.divide(dist_s, dmax, out=tmp)
            elevation -= cfg_inst.distance_weight * tmp

    if cfg_inst.use_boundary and p_bound is not None:
        b = smooth01(p_bound, cfg_inst.smooth_boundary_sigma)
        elevation += cfg_inst.gamma_boundary * b

    if cfg_inst.use_edge_term and cfg_inst.edge_weight != 0:
        g = ndi.gaussian_gradient_magnitude(
            p_cell,
            sigma=float(cfg_inst.edge_sigma),
        )
        gmax = g.max()
        if gmax > 1e-6:
            np.divide(g, gmax, out=g)
            elevation += cfg_inst.edge_weight * g

    if cfg_inst.use_energy and p_energy is not None and cfg_inst.energy_weight != 0:
        e = smooth01(p_energy, cfg_inst.energy_smooth_sigma)
        elevation -= cfg_inst.energy_weight * e

    markers = make_markers(
        mask=mask,
        p_center=p_center,
        dist_s=dist_s,
        use_centers=cfg_inst.use_centers,
        center_seed_method=cfg_inst.center_seed_method,
        center_min_distance=cfg_inst.center_min_distance,
        center_thr=cfg_inst.center_thr,
    )

    instance_segmenter = InstanceSegmenter(cfg_inst)
    inst_out = instance_segmenter(
        _make_seg_out_from_probs(
            {
                "cell": p_cell,
                "bound": p_bound,
                "center": p_center,
                "energy": p_energy,
            }
        ),
        update_cell_mask=True,
    )

    return {
        "p_cell": p_cell,
        "p_bound": p_bound,
        "p_center": p_center,
        "p_energy": p_energy,
        "mask": inst_out.get("cell_mask", mask),
        "dist": dist,
        "dist_s": dist_s,
        "elevation": elevation,
        "markers": markers,
        "instances": inst_out["instance_labels"],
    }


def random_instance_colors(
    instances: np.ndarray,
    background_label: int = 0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    instances = np.asarray(instances)
    labels = np.unique(instances)
    labels = labels[labels != background_label]

    h, w = instances.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    for lab in labels:
        color = rng.uniform(0.2, 1.0, size=3)
        out[instances == lab] = color

    return out


def show_panel(
    ax: Axes,
    img: np.ndarray,
    title: str | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
):
    im = ax.imshow(
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=cfg.TITLE_SIZE)

    return im


def _crop_array(
    a: np.ndarray | None,
    sy: int,
    sx: int,
    size: int,
) -> np.ndarray | None:
    if a is None:
        return None

    a = np.asarray(a)

    if a.ndim == 2:
        return a[sy:sy + size, sx:sx + size]

    if a.ndim == 3:
        return a[sy:sy + size, sx:sx + size, :]

    raise ValueError(f"Cannot crop array with shape {a.shape}.")


def _prepare_image_for_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"Expected image [H,W] or [H,W,C], got shape {img.shape}.")

    if img.max() > 1.0:
        img = img / img.max()

    return img


def _segmentation_white_background(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        return image

    if image.dtype == np.uint8:
        out = image.copy()
        bg = np.all(out == 0, axis=-1)
        out[bg] = 255
        return out

    out = image.astype(np.float32, copy=True)
    bg = np.all(out <= 1e-6, axis=-1)
    out[bg] = 1.0
    return out


def _prepare_for_image_grid(
    image: np.ndarray,
    *,
    is_segmentation: bool,
) -> np.ndarray:
    image = prepare_image(
        image,
        is_segmentation=is_segmentation,
    )

    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)

    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]

    else:
        raise ValueError(f"Unsupported image shape for image grid: {image.shape}")

    if image.dtype == np.uint8:
        if is_segmentation:
            image = _segmentation_white_background(image)
        return image

    image = image.astype(np.float32, copy=False)

    if image.max() > 1.0:
        image = image / image.max()

    image = np.clip(image, 0.0, 1.0)

    if is_segmentation:
        image = _segmentation_white_background(image)

    return image


def _clip_inset_coords(
    image: np.ndarray,
    inset_coords: tuple[int, int],
    inset_side_length: int,
) -> tuple[int, int]:
    h, w = image.shape[:2]
    x, y = inset_coords

    x = int(max(0, min(x, w - inset_side_length)))
    y = int(max(0, min(y, h - inset_side_length)))

    return x, y


def _add_inset_overlay(
    ax: Axes,
    image: np.ndarray,
    inset_coords: tuple[int, int],
    inset_side_length: int,
    title: str | None = None,
) -> None:
    ax.imshow(image)

    x, y = _clip_inset_coords(
        image=image,
        inset_coords=inset_coords,
        inset_side_length=inset_side_length,
    )

    rect = Rectangle(
        (x, y),
        inset_side_length,
        inset_side_length,
        fill=False,
        edgecolor=INSET_RECT_COLOR,
        linewidth=INSET_LINEWIDTH,
    )
    ax.add_patch(rect)

    inset_img = crop_square(
        image,
        x=x,
        y=y,
        length=inset_side_length,
    )

    axins = inset_axes(
        ax,
        width=INSET_WIDTH,
        height=INSET_HEIGHT,
        loc=INSET_LOCATION,
        bbox_to_anchor=(
            0.0,
            0.0,
            0.95,
            0.95,
        ),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    axins.imshow(inset_img)
    axins.set_xticks([])
    axins.set_yticks([])

    assert title is not None

    if "monochrome" in title.lower() and "segmentation" not in title.lower():
        inset_color = "white"
    else:
        inset_color = INSET_BORDER_COLOR

    for spine in axins.spines.values():
        assert isinstance(title, str)
        spine.set_edgecolor(
            inset_color
        )
        spine.set_linewidth(INSET_BORDER_LINEWIDTH)

    ax.set_title(title, fontsize=cfg.TITLE_SIZE)

    ax.set_xticks([])
    ax.set_yticks([])


def _plot_identity_scatter(
    ax: Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    hue_col: str | None = None,
    legend: bool = False,
    legend_fontsize: int | None = None,
) -> None:
    plot_data = data.copy()

    plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors="coerce")
    plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[x_col, y_col])

    sns.scatterplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        ax=ax,
        **SCATTER_KWARGS,
    )

    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=cfg.AXIS_LABEL_SIZE)

    xmin = float(plot_data[x_col].min())
    xmax = float(plot_data[x_col].max())
    ymin = float(plot_data[y_col].min())
    ymax = float(plot_data[y_col].max())

    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    x = np.array([lo, hi])
    ax.plot(
        x,
        x,
        linestyle="--",
        color="red",
        zorder=1,
    )

    for coll in ax.collections:
        coll.set_zorder(3)

    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            markerscale=4,
            title="",
            fontsize=legend_fontsize if legend_fontsize is not None else cfg.TITLE_SIZE,
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()


def prepare_instance_segmentation_panel_data(
    model_output_dir: str,
    ext_images_dir: str,
    validation_results_dir: str,
    *,
    redo_analysis: bool = False,
    inset_start: tuple[int, int] = (210, 200),
    inset_size: int = 100,
) -> dict[str, Any]:
    unet_base_config = copy.deepcopy(UNET_CONFIG)
    instance_seg_config = _as_instance_cfg(INSTANCE_CONFIG)

    res = generate_unet_comparison(
        models_dir=model_output_dir,
        ext_images_dir=ext_images_dir,
        unet_base_config=unet_base_config,
        segmenter_class=SegmenterUNet,
        output_dir=validation_results_dir,
        output_filename="figure_3_unet_segmentation_comparison",
        redo_analysis=redo_analysis,
    )

    seg_out = {"probs": res["small"]}
    img_disp = _prepare_image_for_display(res["original"])

    cell_prob = _as_2d_prob(seg_out["probs"]["cell"], "cell")
    cell_mask = (cell_prob > 0.5).astype(np.uint8)
    instances_baseline = measure.label(cell_mask, connectivity=1).astype(np.int32)

    steps = compute_instance_steps(seg_out, instance_seg_config)

    inst_baseline_rgb_full = random_instance_colors(
        instances_baseline,
        background_label=0,
        seed=1,
    )
    inst_instance_segmenter_rgb_full = random_instance_colors(
        steps["instances"],
        background_label=0,
        seed=2,
    )

    if inset_start is not None:
        y0, x0 = inset_start
        h, w = steps["p_cell"].shape

        y0 = max(0, min(int(y0), h - 1))
        x0 = max(0, min(int(x0), w - 1))

        if y0 + inset_size > h:
            y0 = max(0, h - inset_size)

        if x0 + inset_size > w:
            x0 = max(0, w - inset_size)

        img_disp = _crop_array(img_disp, y0, x0, inset_size)

        for key in list(steps.keys()):
            steps[key] = _crop_array(steps[key], y0, x0, inset_size)

        inst_baseline_rgb = _crop_array(
            inst_baseline_rgb_full,
            y0,
            x0,
            inset_size,
        )
        inst_instance_segmenter_rgb = _crop_array(
            inst_instance_segmenter_rgb_full,
            y0,
            x0,
            inset_size,
        )
    else:
        inst_baseline_rgb = inst_baseline_rgb_full
        inst_instance_segmenter_rgb = inst_instance_segmenter_rgb_full

    return {
        "img_disp": img_disp,
        "steps": steps,
        "inst_baseline_rgb": inst_baseline_rgb,
        "inst_instance_segmenter_rgb": inst_instance_segmenter_rgb,
    }


# ---- Figure 3 C–E: consistent, instance-derived evaluation metrics ------------
# The two supported evaluation scopes are controlled by ONE parameter. The
# validation scenes have incomplete tiling, so "PER_IMAGE" in panel C means the
# average of its recorded tile scores, NOT a reconstructed full-image score.
METRIC_LEVELS = ("PER_IMAGE", "PER_TILE")
PRIMARY_METRIC = "F1_center_4px"
REAL_METRICS = (
    PRIMARY_METRIC,
    "center_precision_r4",
    "center_recall_r4",
    "instance_f1_iou_0p5",
    "mean_instance_dice_all_gt_iou_0p5",
    "pixel_dice",
)


def _validate_metric_level(metric_level: str) -> str:
    if metric_level not in METRIC_LEVELS:
        raise ValueError(f"metric_level must be one of {METRIC_LEVELS}, got {metric_level!r}")
    return metric_level


def _read_metric_csv(path: str | Path) -> pd.DataFrame:
    """Load a result CSV. Also handle fully quoted rows in exported CSV copies."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required Figure 3 results file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "sample_idx" in df.columns and len(df) and df["sample_idx"].isna().all():
        # Some sample exports quote the entire data row as a single CSV field.
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            first = next(reader, None)
            if first is not None and len(first) == 1 and len(columns) > 1:
                rows = []
                for row in chain((first,), reader):
                    parsed = next(csv.reader([row[0]])) if len(row) == 1 else row
                    if len(parsed) != len(columns):
                        raise ValueError(f"{path.name}: expected {len(columns)} fields, got {len(parsed)}")
                    rows.append(parsed)
                df = pd.DataFrame(rows, columns=columns)
    return df


def _require_metric_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...], source: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise KeyError(f"{source}: missing required columns {missing}")


def _make_numeric(df: pd.DataFrame, metrics: tuple[str, ...], source: str) -> pd.DataFrame:
    _require_metric_columns(df, metrics, source)
    df = df.copy()
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        if metric != "count_error_relative":
            invalid = df[metric].notna() & ~df[metric].between(0, 1)
            if invalid.any():
                raise ValueError(f"{source}: {metric} has values outside [0,1]")
    return df


def _paired_metric_frame(
    df: pd.DataFrame, metric: str, methods: tuple[str, ...]
) -> pd.DataFrame:
    """Retain only observations scored by every listed algorithm."""
    _require_metric_columns(df, ("image_id", "algorithm", metric), "comparison")
    subset = df.loc[df["algorithm"].isin(methods), ["image_id", "algorithm", metric]].copy()
    if subset.duplicated(["image_id", "algorithm"]).any():
        raise ValueError("Repeated image/tile and method combinations in comparison data")
    wide = subset.pivot(index="image_id", columns="algorithm", values=metric)
    absent = set(methods).difference(wide.columns)
    if absent:
        raise ValueError(f"Missing comparison algorithms: {sorted(absent)}")
    wide = wide.dropna(subset=list(methods))
    if wide.empty:
        raise ValueError(f"No matched observations with finite {metric} for {methods}")
    return wide[list(methods)].stack().rename(metric).reset_index()


def _load_simulated_validation(
    validation_results_dir: str, metric_level: str,
) -> pd.DataFrame:
    """Panel C: only the small, tiles-trained UNet and saved per-tile F1."""
    path = Path(validation_results_dir) / "training_val_small_tiles_inst_seg.csv"
    df = _read_metric_csv(path)
    _require_metric_columns(
        df, ("sample_idx", "tile_idx", "metric_level", "dataset_mode", PRIMARY_METRIC),
        path.name,
    )
    df = df.loc[df["metric_level"].eq("tile") &
                df["dataset_mode"].isin(("tiles", "tiling"))].copy()
    if "segmentation_method" in df.columns:
        df = df.loc[df["segmentation_method"].eq("inst_seg")]
    df = _make_numeric(df, (PRIMARY_METRIC,), path.name)
    df = df.dropna(subset=[PRIMARY_METRIC, "sample_idx", "tile_idx"])
    if df.empty:
        raise ValueError(f"No usable tile F1 values in {path}")
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="raise").astype(int)
    df["tile_idx"] = pd.to_numeric(df["tile_idx"], errors="raise").astype(int)
    if df.duplicated(["sample_idx", "tile_idx"]).any():
        raise ValueError(f"Duplicate simulated validation tile identity in {path.name}")
    if metric_level == "PER_IMAGE":
        # NOT a true whole-image F1: incomplete validation tile coverage.
        df = df.groupby("sample_idx", as_index=False)[PRIMARY_METRIC].mean()
        df["image_id"] = df["sample_idx"].astype(str)
    else:
        df["image_id"] = (df["sample_idx"].astype(str) + ":" +
                          df["tile_idx"].astype(str))
    df["algorithm"] = "UNet"
    print(f"Figure 3C ({metric_level}): {len(df):,} small-UNet observations")
    return df[["image_id", "algorithm", PRIMARY_METRIC]]


def _load_real_annotations(
    validation_results_dir: str, metric_level: str,
    metrics: tuple[str, ...] = REAL_METRICS,
) -> pd.DataFrame:
    """Human annotated microscopy: matched UNet and Cellpose tile/image rows."""
    path = Path(validation_results_dir) / "manual_segmentation_comparison_v2.csv"
    df = _read_metric_csv(path)
    _require_metric_columns(
        df, ("level", "status", "source_image_key", "image_name", "human_annotator", "method"),
        path.name,
    )
    scope = "image" if metric_level == "PER_IMAGE" else "tile"
    df = df.loc[df["level"].eq(scope) & df["status"].eq("ok")].copy()
    if "unet_mode" in df.columns:
        # The manuscript main figure uses only the small UNet.
        df = df.loc[df["unet_mode"].eq("small")]
    df["algorithm"] = df["method"].replace({"AlloViewer_UNet": "UNet"})
    df = df.loc[df["algorithm"].isin(("UNet", "Cellpose"))].copy()
    df = _make_numeric(df, metrics, path.name)
    if df.empty:
        raise ValueError(f"No usable {scope} rows in {path.name}")
    if metric_level == "PER_IMAGE":
        if df.duplicated(["source_image_key", "human_annotator", "algorithm"]).any():
            raise ValueError("Duplicate image/annotator/method combinations in manual data")
        # Each image gets one measurement per method; multiple independent
        # human annotations are averaged after their tile-confusion pooling.
        df = df.groupby(["source_image_key", "algorithm"], as_index=False)[list(metrics)].mean()
        df["image_id"] = df["source_image_key"].astype(str)
    else:
        if df.duplicated(["source_image_key", "image_name", "human_annotator", "algorithm"]).any():
            raise ValueError("Duplicate tile/annotator/method combinations in manual data")
        # Keep both human annotations if present: the reference is annotator-
        # specific; multiple tiles from a source image may still overlap.
        df["image_id"] = (df["source_image_key"].astype(str) + "|" +
                          df["image_name"].astype(str) + "|" +
                          df["human_annotator"].astype(str))
    print(f"Real-image data ({metric_level}): {df['image_id'].nunique():,} scored units")
    return df[["image_id", "algorithm", *metrics]]


def _load_simulated_testing(
    validation_results_dir: str, metric_level: str,
) -> pd.DataFrame:
    """Panel E: small-UNet / Cellpose / ImageJ from common CPIJ CSV."""
    path = Path(validation_results_dir) / "testing_val_CPIJ_small_inst_seg.csv"
    df = _read_metric_csv(path)
    _require_metric_columns(
        df, ("metric_level", "sample_idx", "tile_idx", "dataset_mode", PRIMARY_METRIC),
        path.name,
    )
    scope = "image" if metric_level == "PER_IMAGE" else "tile"
    df = df.loc[df["metric_level"].eq(scope)].copy()
    # CPIJ identifies the tested algorithm in dataset_mode (also has 'method').
    df["algorithm"] = df["dataset_mode"].replace({
        "AlloViewer_UNet": "UNet", "imageJ": "ImageJ", "NCISP": "ImageJ",
    })
    df = df.loc[df["algorithm"].isin(("UNet", "Cellpose", "ImageJ"))].copy()
    df = _make_numeric(df, (PRIMARY_METRIC,), path.name)
    df = df.dropna(subset=[PRIMARY_METRIC, "sample_idx"])
    if df.empty:
        raise ValueError(f"No usable CPIJ {scope} results in {path.name}")
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="raise").astype(int)
    if metric_level == "PER_TILE":
        df["tile_idx"] = pd.to_numeric(df["tile_idx"], errors="raise").astype(int)
        df["image_id"] = (df["sample_idx"].astype(str) + ":" +
                          df["tile_idx"].astype(str))
    else:
        df["image_id"] = df["sample_idx"].astype(str)
    print(f"Figure 3E ({metric_level}): {df['image_id'].nunique():,} scored units")
    return df[["image_id", "algorithm", PRIMARY_METRIC]]


def _plot_metric_boxstrip(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    methods: tuple[str, ...],
    title: str,
    ylabel: str,
    *,
    max_strip_points: int | None = 400,
) -> None:
    """White boxes, full-range whiskers, and black-edged gray strip points."""
    df = data.loc[data["algorithm"].isin(methods), ["algorithm", metric]].dropna()
    if df.empty:
        raise ValueError(f"No data for {title}: {metric}")
    absent = set(methods).difference(df["algorithm"].unique())
    if absent:
        raise ValueError(f"{title}: no observations for {sorted(absent)}")

    sns.boxplot(
        data=df, x="algorithm", y=metric, order=list(methods), ax=ax,
        width=0.56, whis=(5, 95), showfliers=False,
        color="white", saturation=1,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.75},
        whiskerprops={"color": "black", "linewidth": 0.65},
        capprops={"color": "black", "linewidth": 0.65},
        medianprops={"color": "black", "linewidth": 1.0},
        zorder=2,
    )
    # Strip points are sampled only to prevent dense plots from becoming solid
    # gray blocks; boxes use every observation. Set None to show all points.
    dots = df
    if max_strip_points is not None:
        dots = pd.concat([
            part.sample(n=min(len(part), max_strip_points), random_state=187)
            for _, part in df.groupby("algorithm", sort=False)
        ], ignore_index=True)
    sns.stripplot(
        data=dots, x="algorithm", y=metric, order=list(methods), ax=ax,
        color="0.55", size=1.9, jitter=0.22, alpha=0.60,
        linewidth=0.3, edgecolor="black", zorder=3,
    )
    counts = df.groupby("algorithm").size()
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([f"{m}\n(n={counts[m]:,})" for m in methods])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linewidth=0.35, color="0.85", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", length=2, pad=2)
    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)


def prepare_model_evaluation_panel_data(
    validation_results_dir: str,
    model_output_dir: str,
    *,
    model_file: str = "best_small_tiles_S512_seed187.pth",
    unet_size: Literal["small", "medium", "large"] = "small",
    comparison_images: Literal["external_images", "tiles"] = "tiles",
    redo_analysis: bool = False,
    metric_level: Literal["PER_IMAGE", "PER_TILE"] = "PER_IMAGE",
) -> dict[str, Any]:
    """Load matched comparisons for panels C–E and unchanged image grid F.

    Only the small UNet and the tiles configuration are used in the main figure;
    the old arguments are retained for call-site compatibility.
    """
    _validate_metric_level(metric_level)
    if unet_size != "small" or comparison_images != "tiles":
        raise ValueError("Main Figure 3 C–E is restricted to small UNet and tiles")

    # Fail early if a required result is absent; avoid generating image cache first.
    panel_c = _load_simulated_validation(validation_results_dir, metric_level)
    panel_d = _load_real_annotations(validation_results_dir, metric_level)
    panel_e = _load_simulated_testing(validation_results_dir, metric_level)

    image_cache_path = os.path.join(
        validation_results_dir, "figure_3_image_cache_fullres.npz",
    )
    image_data = load_or_create_multimodal_image_cache(
        cache_path=image_cache_path,
        model_dir=model_output_dir,
        model_file=model_file,
        force_recompute=redo_analysis,
    )

    return {
        "unet_on_sim": panel_c,
        "unet_on_human": panel_d,
        "imagej_on_sim": panel_e,
        "image_data": image_data,
        "metric_level": metric_level,
    }


def prepare_figure_3_data(
    validation_results_dir: str,
    model_output_dir: str,
    ext_images_dir: str,
    metric_level: Literal["PER_IMAGE", "PER_TILE"] = "PER_IMAGE",
) -> dict[str, Any]:
    return {
        "instance_segmentation": prepare_instance_segmentation_panel_data(
            model_output_dir=model_output_dir,
            ext_images_dir=ext_images_dir,
            validation_results_dir=validation_results_dir,
            redo_analysis=False,
            inset_start=(210, 200),
            inset_size=100,
        ),
        "model_evaluation": prepare_model_evaluation_panel_data(
            validation_results_dir=validation_results_dir,
            model_output_dir=model_output_dir,
            redo_analysis=False,
            metric_level=metric_level,
        ),
    }


def _generate_main_figure(
    figure_data: dict[str, Any],
    figure_output_dir: str,
    figure_name: str,
    inset_coords: Optional[dict[str, tuple[int, int]]] = None,
) -> None:
    if inset_coords is None:
        inset_coords = INSET_COORDS

    instance_data = figure_data["instance_segmentation"]
    evaluation_data = figure_data["model_evaluation"]

    img_disp = instance_data["img_disp"]
    steps = instance_data["steps"]
    inst_baseline_rgb = instance_data["inst_baseline_rgb"]
    inst_instance_segmenter_rgb = instance_data["inst_instance_segmenter_rgb"]

    unet_on_sim = evaluation_data["unet_on_sim"]
    unet_on_human = evaluation_data["unet_on_human"]
    imagej_on_sim = evaluation_data["imagej_on_sim"]
    image_data = evaluation_data["image_data"]

    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(2, 4)
        axes_a = fig_sgs.subplots()

        img_disp_with_scale_bar = _add_scale_bar_to_image(img_disp)

        show_panel(axes_a[0, 0], img_disp_with_scale_bar, "original")
        show_panel(axes_a[0, 1], steps["p_cell"], "cell prob. $p_{cell}$", cmap="viridis")
        show_panel(axes_a[0, 2], steps["mask"], "hysteresis mask", cmap="gray")
        show_panel(axes_a[0, 3], steps["dist_s"], "distance", cmap="magma")

        show_panel(axes_a[1, 0], steps["p_bound"], "boundary prob.", cmap="viridis")
        show_panel(axes_a[1, 1], steps["elevation"], "elevation map", cmap="magma")
        show_panel(axes_a[1, 2], steps["markers"], "markers", cmap="jet")

        axes_a[1, 3].imshow(img_disp, cmap="gray", interpolation="nearest")
        axes_a[1, 3].imshow(
            random_instance_colors(steps["instances"]),
            alpha=0.7,
            interpolation="nearest",
        )
        axes_a[1, 3].set_title("instances", fontsize=cfg.TITLE_SIZE)
        axes_a[1, 3].set_xticks([])
        axes_a[1, 3].set_yticks([])

    def generate_subfigure_b(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(1, 3)
        axes_b = fig_sgs.subplots()

        img_disp_with_scale_bar = _add_scale_bar_to_image(img_disp)

        show_panel(axes_b[0], img_disp_with_scale_bar, "original")

        axes_b[1].imshow(inst_baseline_rgb)
        axes_b[1].set_title(
            "conventional mask segmentation",
            fontsize=cfg.TITLE_SIZE,
        )
        axes_b[1].set_xticks([])
        axes_b[1].set_yticks([])

        axes_b[2].imshow(inst_instance_segmenter_rgb)
        axes_b[2].set_title(
            "instanceSegmenter mask segmentation",
            fontsize=cfg.TITLE_SIZE,
        )
        axes_b[2].set_xticks([])
        axes_b[2].set_yticks([])

    def generate_subfigure_c(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        plot_ax = fig.add_subplot(gs.subgridspec(1, 1)[0])
        # Small UNet only; other checkpoints belong in the supplement.
        _plot_metric_boxstrip(
            plot_ax, unet_on_sim, PRIMARY_METRIC, ("UNet",),
            "UNet performance on\nsimulated images", "Center F1 (4 px)",
        )

    def generate_subfigure_d(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        plot_ax = fig.add_subplot(gs.subgridspec(1, 1)[0])
        paired = _paired_metric_frame(unet_on_human, PRIMARY_METRIC, ("UNet", "Cellpose"))
        _plot_metric_boxstrip(
            plot_ax, paired, PRIMARY_METRIC, ("UNet", "Cellpose"),
            "UNet and CellPoseSAM performance\non human annotated real images", "Center F1 (4 px)",
        )

    def generate_subfigure_e(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)
        plot_ax = fig.add_subplot(gs.subgridspec(1, 1)[0])
        methods = ("UNet", "Cellpose")
        paired = _paired_metric_frame(imagej_on_sim, PRIMARY_METRIC, methods)
        _plot_metric_boxstrip(
            plot_ax, paired, PRIMARY_METRIC, methods,
            "UNet comparison to CellPoseSAM on\nvariable simulated images", "Center F1 (4 px)",
        )

    def generate_subfigure_f(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(2, 5)

        sim_img = _prepare_for_image_grid(
            image_data["simulated_image"],
            is_segmentation=False,
        )
        micro_img = _prepare_for_image_grid(
            image_data["microscopy_image"],
            is_segmentation=False,
        )
        gpixel_img = _prepare_for_image_grid(
            image_data["googlepixel_image"],
            is_segmentation=False,
        )
        iphone_img = _prepare_for_image_grid(
            image_data["iphone_image"],
            is_segmentation=False,
        )
        mono_img = _prepare_for_image_grid(
            image_data["monochrome_image"],
            is_segmentation=False,
        )

        sim_seg = _prepare_for_image_grid(
            image_data["simulated_segmentation"],
            is_segmentation=True,
        )
        micro_seg = _prepare_for_image_grid(
            image_data["microscopy_segmentation"],
            is_segmentation=True,
        )
        gpixel_seg = _prepare_for_image_grid(
            image_data["googlepixel_segmentation"],
            is_segmentation=True,
        )
        iphone_seg = _prepare_for_image_grid(
            image_data["iphone_segmentation"],
            is_segmentation=True,
        )
        mono_seg = _prepare_for_image_grid(
            image_data["monochrome_segmentation"],
            is_segmentation=True,
        )

        image_panels = [
            (sim_img, inset_coords["simulated_image"], cfg.PHONE_DICT["Simulated"]),
            (micro_img, inset_coords["microscopy_image"], cfg.PHONE_DICT["Microscope"]),
            (gpixel_img, inset_coords["googlepixel_image"], cfg.PHONE_DICT["GooglePixel"]),
            (iphone_img, inset_coords["iphone_image"], cfg.PHONE_DICT["iPhone"]),
            (mono_img, inset_coords["monochrome_image"], cfg.PHONE_DICT["Monochrome"]),
        ]

        segmentation_panels = [
            (
                sim_seg,
                inset_coords["simulated_segmentation"],
                f"{cfg.PHONE_DICT['Simulated']}\nSegmentation",
            ),
            (
                micro_seg,
                inset_coords["microscopy_segmentation"],
                f"{cfg.PHONE_DICT['Microscope']}\nSegmentation",
            ),
            (
                gpixel_seg,
                inset_coords["googlepixel_segmentation"],
                f"{cfg.PHONE_DICT['GooglePixel']}\nSegmentation",
            ),
            (
                iphone_seg,
                inset_coords["iphone_segmentation"],
                f"{cfg.PHONE_DICT['iPhone']}\nSegmentation",
            ),
            (
                mono_seg,
                inset_coords["monochrome_segmentation"],
                f"{cfg.PHONE_DICT['Monochrome']}\nSegmentation",
            ),
        ]

        for col, (image, coords, title) in enumerate(image_panels):
            panel_ax = fig.add_subplot(fig_sgs[0, col])
            _add_inset_overlay(
                ax=panel_ax,
                image=image,
                inset_coords=coords,
                inset_side_length=INSET_SIDE_LENGTH,
                title=title,
            )

        for col, (image, coords, title) in enumerate(segmentation_panels):
            panel_ax = fig.add_subplot(fig_sgs[1, col])
            _add_inset_overlay(
                ax=panel_ax,
                image=image,
                inset_coords=coords,
                inset_side_length=INSET_SIDE_LENGTH,
                title=title,
            )

    fig = plt.figure(
        layout="constrained",
        figsize=(
            cfg.FIGURE_WIDTH_FULL,
            cfg.FIGURE_HEIGHT_FULL,
        ),
    )

    gs = GridSpec(
        ncols=3,
        nrows=4,
        figure=fig,
        height_ratios=[1.2, 0.85, 0.7, 1.25],
    )

    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, 0]
    d_coords = gs[2, 1]
    e_coords = gs[2, 2]
    f_coords = gs[3, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)
    fig_e = fig.add_subplot(e_coords)
    fig_f = fig.add_subplot(f_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")
    generate_subfigure_d(fig, fig_d, d_coords, "D")
    generate_subfigure_e(fig, fig_e, e_coords, "E")
    generate_subfigure_f(fig, fig_f, f_coords, "F")

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return


def figure_3_generation(
    validation_results_dir: str,
    figure_output_dir: str,
    model_output_dir: str,
    ext_images_dir: str,
    metric_level: Literal["PER_IMAGE", "PER_TILE"] = "PER_IMAGE",
    **kwargs
) -> None:
    _generate_main_figure(
        figure_data=prepare_figure_3_data(
            validation_results_dir=validation_results_dir,
            model_output_dir=model_output_dir,
            ext_images_dir=ext_images_dir,
            metric_level=metric_level,
        ),
        figure_output_dir=figure_output_dir,
        figure_name=("Figure_3" if metric_level == "PER_IMAGE" else "Figure_3_PER_TILE"),
    )
