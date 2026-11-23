"""Supplementary Figure S7: UNet versus Cellpose on simulated test images.

A: four shared, final-instance metrics from testing_val_CPIJ_small_inst_seg.csv.
B: one cropped simulated field, showing actual simulator instances and the
   reconstructed full-image UNet/Cellpose predictions used by the CPIJ pipeline.

The old Energy SSIM comparison is intentionally absent: Cellpose does not
produce UNet-like energy heads. Cell boundaries and centroid markers in B are
DERIVED FROM INSTANCE LABELS (not the UNet boundary/center probability heads).

PER_IMAGE: one image-wide measurement per simulated scene.
PER_TILE:  one metric per crop of a *full-image* prediction. Overlapping tile
           measurements are not statistically independent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from alloviewer.dev.figures import figure_config as cfg
from alloviewer.dev.figures import figure_utils as utils

MetricLevel = Literal["PER_IMAGE", "PER_TILE"]
METHODS = ("UNet", "Cellpose")
FIGURE_WIDTH = 6.375

# Shared, directly comparable final-instance metrics. Do not use UNet-only
# legacy boundary_f1, center_f1 or energy_ssim target-head metrics.
S7_METRICS = (
    ("pixel_iou", "Foreground IoU"),
    ("boundary_f1_tol_px", "Boundary F1"),
    ("F1_center_4px", "Center F1 (4 px)"),
    ("instance_f1_iou_0p5", "Instance F1 (IoU ≥ 0.5)"),
)


def validate_metric_level(metric_level: str) -> str:
    if metric_level not in {"PER_IMAGE", "PER_TILE"}:
        raise ValueError("metric_level must be 'PER_IMAGE' or 'PER_TILE'")
    return metric_level


def read_metrics_csv(path: str | Path) -> pd.DataFrame:
    """Read normal CPIJ/manual CSVs; reject missing files/empty datasets."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Metrics CSV contains no rows: {path}")
    return df


def require_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str], source: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{source}: missing required columns {missing}")


def make_numeric_metrics(df: pd.DataFrame, metrics: tuple[str, ...], source: str) -> pd.DataFrame:
    require_columns(df, list(metrics), source)
    df = df.copy()
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        ok = df[metric].dropna()
        if not np.isfinite(ok).all() or not ok.between(0, 1).all():
            raise ValueError(f"{source}: invalid metric values in {metric} (expected [0,1])")
    return df


def paired_metric(df: pd.DataFrame, metric: str, methods: tuple[str, ...] = METHODS) -> pd.DataFrame:
    """Use the identical source image/tile for every compared method."""
    require_columns(df, ["image_id", "algorithm", metric], "pairing input")
    sub = df.loc[df["algorithm"].isin(methods), ["image_id", "algorithm", metric]].copy()
    if sub.duplicated(["image_id", "algorithm"]).any():
        raise ValueError(f"Duplicate image/tile and algorithm entries for {metric}")
    wide = sub.pivot(index="image_id", columns="algorithm", values=metric)
    missing = set(methods) - set(wide.columns)
    if missing:
        raise ValueError(f"{metric}: missing algorithms {sorted(missing)}")
    wide = wide.dropna(subset=list(methods))
    if wide.empty:
        raise ValueError(f"{metric}: no matched observations with finite values")
    return wide[list(methods)].stack().rename(metric).reset_index()


def plot_boxstrip(
    ax: Axes, data: pd.DataFrame, metric: str, title: str,
    *, methods: tuple[str, ...] = METHODS, max_strip_points: int | None = 400,
) -> None:
    """White boxplots, full-range whiskers and black-edged gray strip points."""
    plot = paired_metric(data, metric, methods)
    sns.boxplot(
        data=plot, x="algorithm", y=metric, order=list(methods), ax=ax,
        width=0.55, whis=(0, 100), showfliers=False,
        color="white", saturation=1,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.75},
        whiskerprops={"color": "black", "linewidth": 0.65},
        capprops={"color": "black", "linewidth": 0.65},
        medianprops={"color": "black", "linewidth": 1.0},
        zorder=2,
    )
    if max_strip_points is None:
        dots = plot
    else:
        dots = pd.concat([
            group.sample(n=min(len(group), max_strip_points), random_state=187)
            for _, group in plot.groupby("algorithm", sort=False)
        ], ignore_index=True)
    sns.stripplot(
        data=dots, x="algorithm", y=metric, order=list(methods), ax=ax,
        color="0.55", size=2.0, jitter=0.22, alpha=0.6,
        linewidth=0.3, edgecolor="black", zorder=3,
    )
    counts = plot.groupby("algorithm").size()
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([f"{m}\n(n={counts[m]:,})" for m in methods])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", linewidth=0.3, color="0.86", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)


def load_S8_metric_data(
    validation_results_dir: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
    seg_method: str = "inst_seg",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read CPIJ result once, return paired metrics and the unfiltered source."""
    validate_metric_level(metric_level)
    csv_path = Path(validation_results_dir) / f"testing_val_CPIJ_{unet_size}_{seg_method}.csv"
    full = read_metrics_csv(csv_path)
    metrics = tuple(key for key, _ in S7_METRICS)
    require_columns(full, ["sample_idx", "tile_idx", "metric_level", "method", *metrics], str(csv_path))
    scope = "image" if metric_level == "PER_IMAGE" else "tile"
    df = full.loc[full["metric_level"].astype(str).str.lower().eq(scope)].copy()
    df["algorithm"] = df["method"].replace({"AlloViewer_UNet": "UNet"})
    df = df.loc[df["algorithm"].isin(METHODS)].copy()
    df = make_numeric_metrics(df, metrics, csv_path.name)
    if df.empty:
        raise ValueError(f"No UNet/Cellpose {scope} rows in {csv_path.name}")
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="raise").astype(int)
    if metric_level == "PER_TILE":
        df["tile_idx"] = pd.to_numeric(df["tile_idx"], errors="raise").astype(int)
        df["image_id"] = df["sample_idx"].astype(str) + ":" + df["tile_idx"].astype(str)
    else:
        df["image_id"] = df["sample_idx"].astype(str)
    if df.duplicated(["image_id", "algorithm"]).any():
        raise ValueError(f"Duplicate CPIJ image/tile and algorithm rows: {csv_path.name}")
    print(f"S7 {metric_level}: {df['image_id'].nunique():,} source images/tiles")
    return df[["image_id", "algorithm", *metrics]], full


def _source_setting(df: pd.DataFrame, key: str, default: Any) -> Any:
    """Recover fixed Cellpose settings actually recorded by CPIJ."""
    if key not in df.columns:
        raise KeyError(f"CPIJ CSV must record {key!r} to reproduce the visual panel")
    candidates = df[key].dropna().unique().tolist()
    if len(candidates) != 1:
        raise ValueError(f"Expected one recorded {key} for the CPIJ run, got {candidates}")
    return candidates[0]


def segment_image_cellpose(
    img: np.ndarray,
    *,
    use_gpu: bool,
    cellpose_model: str | None,
    diameter: float,
    min_size: int,
    flow_threshold: float,
    cellprob_threshold: float,
    expected_version: str | None = None,
) -> np.ndarray:
    """Run the EXACT Cellpose model/eval adapter used by the comparison script.

    Accepts RGB HWC in [0,1]. Does not invent boundary, center or energy heads.
    """
    from alloviewer.dev.validation.manual_segmentation_comparison import (
        _make_cellpose, _predict_cellpose,
    )
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3 or not np.isfinite(arr).all():
        raise ValueError("Cellpose input must be finite RGB HWC")
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError("Cellpose input must be normalized RGB in [0,1]")
    model, major, version, _label = _make_cellpose(cellpose_model, use_gpu)
    if expected_version is not None and str(version) != str(expected_version):
        raise RuntimeError(
            f"Saved CPIJ results used Cellpose {expected_version}, installed version is {version}. "
            "Use the same environment/model to regenerate the representative image."
        )
    masks = _predict_cellpose(
        model, arr, major=major, diameter=float(diameter), min_size=int(min_size),
        flow_threshold=float(flow_threshold), cellprob_threshold=float(cellprob_threshold),
    )
    masks = np.asarray(masks, dtype=np.int32)
    if masks.shape != arr.shape[:2] or (masks < 0).any():
        raise ValueError("Cellpose result is not a valid image-sized instance label array")
    return masks


def _load_representative_field(
    model_output_dir: str,
    h5_dir: str,
    source: pd.DataFrame,
    *,
    img_index: int,
    tile_idx: int,
    unet_size: str,
    seg_method: str,
    model_file: str | None,
) -> dict[str, np.ndarray]:
    """Recreate one CPIJ full-image inference and then crop one saved tile.

    The full image is the inference unit for Cellpose. UNet tiles are stitched
    by averaging the heads before instance segmentation, exactly as in CPIJ.
    """
    import torch
    from torch.utils.data import DataLoader
    from alloviewer.dev.segmentation import TiledH5Dataset
    from alloviewer.dev.segmentation.utils import collate_no_meta
    from alloviewer.dev.validation.imageJ_comparison import (
        _extract_full_hw, _make_segmenter_and_validation_config,
        _segment_full_probs, stitch_float_tiles_mean, stitch_instance_tiles,
    )
    from alloviewer.dev.validation.simulated_segmentation_comparison import _full_rgb_from_tiles

    gt_path = Path(h5_dir) / "fullres_ground_truth.h5"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing stored simulator GT: {gt_path}")
    if seg_method != "inst_seg":
        raise ValueError("S7 representative inference currently requires seg_method='inst_seg'")
    if img_index not in set(pd.to_numeric(source["sample_idx"], errors="raise").astype(int)):
        raise ValueError(f"Requested img_index={img_index} was not scored in CPIJ CSV")
    chosen = model_file or f"best_{unet_size}_tiles_S512_seed187.pth"
    segmenter, imagej_segmenter, validation_cfg = _make_segmenter_and_validation_config(
        gt_h5_path=str(gt_path),
        out_csv=str(Path(h5_dir) / "_unused_S7.csv"),
        out_summary_json=str(Path(h5_dir) / "_unused_S7.json"),
        unet_mode=unet_size,
        model_dir=str(model_output_dir),
        model_file=chosen,
        seg_method=seg_method,
    )
    del imagej_segmenter  # Not required: S7 compares only UNet and Cellpose.

    ds = TiledH5Dataset(str(gt_path), indices=[int(img_index)])
    try:
        batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False,
                                     num_workers=0, collate_fn=collate_no_meta)))
        inputs, _targets, extras = batch
        meta = extras["meta"][0]
        tile_metas = meta["tiles"]
        n_tiles = len(tile_metas)
        if n_tiles == 0 or not (0 <= tile_idx < n_tiles):
            raise ValueError(f"tile_idx={tile_idx} invalid; image has {n_tiles} tiles")
        tiles = inputs[0, :n_tiles].detach().cpu().numpy().astype(np.float32)
        inst_tiles = extras["instance_labels"][0, :n_tiles].detach().cpu().numpy().astype(np.int32)
        full_hw = _extract_full_hw(meta, tile_metas)
        gt_full = stitch_instance_tiles(inst_tiles, tile_metas, full_hw)
        rgb_full = _full_rgb_from_tiles(tiles, tile_metas, full_hw)

        with torch.inference_mode():
            unet_out = segmenter(tiles)
        probs_tiles = np.stack(
            [unet_out["probs"][key] for key in ("cell", "bound", "center", "energy")],
            axis=1,
        ).astype(np.float32)
        probs_full = stitch_float_tiles_mean(probs_tiles, tile_metas, full_hw)
        unet_full = _segment_full_probs(
            segmenter.inst_seg, *probs_full, cell_thr=validation_cfg.cell_thr,
            update_cell_mask=True,
        )

        cp_rows = source.loc[source["method"].eq("Cellpose")]
        if cp_rows.empty:
            raise ValueError("CPIJ results have no Cellpose settings")
        model_name = str(_source_setting(cp_rows, "cellpose_model", None))
        if model_name == "installed_cellpose4_default":
            model_name = None
        cellpose_full = segment_image_cellpose(
            rgb_full, use_gpu=str(segmenter.device).startswith("cuda"),
            cellpose_model=model_name,
            diameter=float(_source_setting(cp_rows, "cellpose_diameter", 10)),
            min_size=int(_source_setting(cp_rows, "cellpose_min_size", 3)),
            flow_threshold=float(_source_setting(cp_rows, "cellpose_flow_threshold", 0.4)),
            cellprob_threshold=float(_source_setting(cp_rows, "cellpose_cellprob_threshold", 0)),
            expected_version=str(_source_setting(cp_rows, "cellpose_version", "")),
        )
        if np.asarray(unet_full).shape != full_hw or cellpose_full.shape != full_hw:
            raise ValueError("Reconstructed predictions and GT do not have identical dimensions")

        tm = tile_metas[tile_idx]
        y0, x0 = [int(x) for x in tm["tile_xy"]]
        h, w = [int(x) for x in tm["tile_hw"]]
        if h <= 0 or w <= 0 or y0 < 0 or x0 < 0 or y0+h > full_hw[0] or x0+w > full_hw[1]:
            raise ValueError("Invalid saved tile coordinates")
        area = np.s_[y0:y0+h, x0:x0+w]
        return {
            "image": rgb_full[area],
            "gt": np.asarray(gt_full[area], dtype=np.int32),
            "unet": np.asarray(unet_full[area], dtype=np.int32),
            "cellpose": np.asarray(cellpose_full[area], dtype=np.int32),
        }
    finally:
        ds.close()
        del segmenter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prepare_figure_S8_data(
    validation_results_dir: str,
    model_output_dir: str,
    h5_path: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
    seg_method: str = "inst_seg",
    img_index: int = 12,
    tile_idx: int = 7,
    model_file: str | None = None,
) -> dict[str, Any]:
    """Read CPIJ metrics; infer one representative example with the saved settings."""
        unet_size=unet_size, seg_method=seg_method,
    )
    field = _load_representative_field(
        model_output_dir, h5_path, source, img_index=img_index, tile_idx=tile_idx,
        unet_size=unet_size, seg_method=seg_method, model_file=model_file,
    )
    return {"metrics": metrics, "field": field, "metric_level": metric_level}


def _instance_centers(labels: np.ndarray) -> np.ndarray:
    """Geometric centroids of cropped instance labels (y,x)."""
    y, x = np.nonzero(labels)
    if not len(y):
        return np.empty((0, 2), dtype=float)
    ids, inv = np.unique(labels[y, x], return_inverse=True)
    counts = np.bincount(inv)
    return np.column_stack([
        np.bincount(inv, weights=y) / counts,
        np.bincount(inv, weights=x) / counts,
    ])


def _colored_instances(labels: np.ndarray) -> np.ndarray:
    """Fixed random palette; image background remains white."""
    unique = np.unique(labels)
    unique = unique[unique > 0]
    out = np.ones((*labels.shape, 3), dtype=np.float32)
    if not len(unique):
        return out
    rng = np.random.default_rng(187)
    colors = rng.uniform(0.2, 0.88, size=(len(unique), 3)).astype(np.float32)
    valid = labels > 0
    out[valid] = colors[np.searchsorted(unique, labels[valid])]
    return out


def _visual_map(labels: np.ndarray, kind: str) -> np.ndarray:
    if kind == "mask":
        return (labels > 0).astype(float)
    if kind == "boundary":
        return find_boundaries(labels, mode="inner").astype(float)
    if kind == "centers":
        dots = np.zeros_like(labels, dtype=float)
        centers = np.rint(_instance_centers(labels)).astype(int)
        if len(centers):
            centers[:, 0] = np.clip(centers[:, 0], 0, labels.shape[0] - 1)
            centers[:, 1] = np.clip(centers[:, 1], 0, labels.shape[1] - 1)
            dots[centers[:, 0], centers[:, 1]] = 1
            dots = ndi.maximum_filter(dots, size=3)
        return dots
    if kind == "instances":
        return _colored_instances(labels)
    raise ValueError(kind)


def _render_with_inset(
    ax: Axes, array: np.ndarray, *, cmap: str | None,
    inset_center: tuple[int, int], inset_box_px: int,
) -> None:
    kw = {"interpolation": "nearest"}
    if cmap is not None:
        kw.update({"cmap": cmap, "vmin": 0, "vmax": 1})
    ax.imshow(array, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    h, w = array.shape[:2]
    cy, cx = inset_center
    size = min(inset_box_px, h, w)
    y0 = max(0, min(h - size, int(cy - size//2)))
    x0 = max(0, min(w - size, int(cx - size//2)))
    ax.add_patch(Rectangle((x0, y0), size, size, fill=False,
                           edgecolor="red", linewidth=0.9))
    axins = inset_axes(ax, width="48%", height="48%", loc="upper right", borderpad=0.15)
    axins.imshow(array[y0:y0+size, x0:x0+size], **kw)
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("black"); s.set_linewidth(0.6)


def _generate_main_figure(
    figure_data: dict[str, Any],
    figure_output_dir: str,
    figure_name: str,
    *,
    inset_center: tuple[int, int] = (450, 150),
    inset_box_px: int = 96,
    max_strip_points: int | None = 400,
) -> None:
    metrics = figure_data["metrics"]
    field = figure_data["field"]

    def generate_subfigure_a(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        ax.axis("off")
        utils.figure_label(ax, letter, x=0)
        subspec = gs.subgridspec(1, 4, wspace=0.35)
        for i, (key, label) in enumerate(S7_METRICS):
            panel = fig.add_subplot(subspec[0, i])
            plot_boxstrip(panel, metrics, key, label, max_strip_points=max_strip_points)

    def generate_subfigure_b(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        ax.axis("off")
        utils.figure_label(ax, letter, x=0)
        subspec = gs.subgridspec(3, 5, wspace=0.03, hspace=0.10)
        col_titles = ("Image", "Cell mask", "Boundaries", "Cell centers", "Instances")
        row_sources = (("Ground truth", "gt"), ("UNet", "unet"), ("Cellpose", "cellpose"))
        for r, (row_name, method) in enumerate(row_sources):
            labels = field[method]
            for c, kind in enumerate(("image", "mask", "boundary", "centers", "instances")):
                axis = fig.add_subplot(subspec[r, c])
                if kind == "image":
                    arr, cmap = field["image"], None
                else:
                    arr = _visual_map(labels, kind)
                    cmap = "gray" if kind != "instances" else None
                _render_with_inset(axis, arr, cmap=cmap,
                                   inset_center=inset_center, inset_box_px=inset_box_px)
                if r == 0:
                    axis.set_title(col_titles[c], fontsize=cfg.TITLE_SIZE)
                if c == 0:
                    axis.set_ylabel(row_name, fontsize=cfg.AXIS_LABEL_SIZE)

    fig = plt.figure(figsize=(FIGURE_WIDTH, 6.7), layout="constrained")
    gs = GridSpec(2, 1, figure=fig, height_ratios=(1, 2.25), hspace=0.15)
    a_ax, b_ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    generate_subfigure_a(fig, a_ax, gs[0], "A")
    generate_subfigure_b(fig, b_ax, gs[1], "B")
    Path(figure_output_dir).mkdir(parents=True, exist_ok=True)
    destination = Path(figure_output_dir) / figure_name
    fig.savefig(destination.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_S8_generation(
    figure_output_dir: str,
    model_output_dir: str,
    validation_results_dir: str,
    h5_path: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
    seg_method: str = "inst_seg",
    img_index: int = 12,
    tile_idx: int = 7,
    model_file: str | None = None,
    figure_name: str | None = None,
    inset_center: tuple[int, int] = (450, 150),
    inset_box_px: int = 96,
    max_strip_points: int | None = 400,
    **kwargs,
) -> None:
    """Public S8 entry point; runs inference for visual B, reads saved F1 data."""
    data = prepare_figure_S8_data(
        validation_results_dir, model_output_dir, h5_path,
        metric_level=metric_level, unet_size=unet_size, seg_method=seg_method,
        img_index=img_index, tile_idx=tile_idx, model_file=model_file,
    )
    _generate_main_figure(
        data, figure_output_dir, figure_name or f"Figure_S8_{metric_level}",
        inset_center=inset_center, inset_box_px=inset_box_px,
        max_strip_points=max_strip_points,
    )

