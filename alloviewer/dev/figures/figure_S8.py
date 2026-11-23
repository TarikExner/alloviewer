"""Supplementary Figure S8: UNet versus Cellpose on human-annotated real images.

Plots only (no images). Maintains the Figure S7 pattern: one preparation
function, named per-panel plotting functions, a figure creator and a public
entry point. Real mask metrics are comparisons to approximate manual circles,
not to pixel-perfect hand-traced cell outlines.

PER_IMAGE averages the available human-annotator image-summary rows for each
source image, and PER_TILE keeps separate tile/annotator measurements. The
human-annotator summaries pool overlapping tiles and are not whole-image masks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec

from alloviewer.dev.figures import figure_utils as utils
from .figure_S7 import (
    MetricLevel, METHODS, FIGURE_WIDTH, validate_metric_level,
    read_metrics_csv, require_columns, make_numeric_metrics, plot_boxstrip,
)

S8_METRICS = (
    ("center_precision_r4", "Center precision (4 px)"),
    ("center_recall_r4", "Center recall (4 px)"),
    ("instance_f1_iou_0p5", "Instance F1 (IoU ≥ 0.5)"),
    ("mean_instance_dice_all_gt_iou_0p5", "Mean GT-cell Dice"),
    ("pixel_dice", "Foreground pixel Dice"),
)


def prepare_figure_S8_data(
    validation_results_dir: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
) -> dict[str, Any]:
    """Read only manual_segmentation_comparison_v2.csv; pair human references."""
    validate_metric_level(metric_level)
    csv_path = Path(validation_results_dir) / "manual_segmentation_comparison_v2.csv"
    df = read_metrics_csv(csv_path)
    metrics = tuple(key for key, _ in S8_METRICS)
    require_columns(
        df, ["level", "status", "source_image_key", "image_name", "human_annotator", "method", *metrics],
        str(csv_path),
    )
    scope = "image" if metric_level == "PER_IMAGE" else "tile"
    df = df.loc[df["level"].eq(scope) & df["status"].eq("ok")].copy()
    if "unet_mode" in df.columns:
        df = df.loc[df["unet_mode"].eq(unet_size)]
    df["algorithm"] = df["method"].replace({"AlloViewer_UNet": "UNet"})
    df = df.loc[df["algorithm"].isin(METHODS)].copy()
    df = make_numeric_metrics(df, metrics, csv_path.name)
    if df.empty:
        raise ValueError(f"No real-image {scope} rows in {csv_path.name}")

    if metric_level == "PER_IMAGE":
        keys = ["source_image_key", "human_annotator"]
    else:
        keys = ["source_image_key", "image_name", "human_annotator"]
    if df.duplicated([*keys, "algorithm"]).any():
        raise ValueError(f"Duplicate real-image reference and algorithm keys: {keys}")
    matched_refs = df.groupby(keys, dropna=False)["algorithm"].nunique().eq(len(METHODS))
    matched_refs = matched_refs[matched_refs].index.to_frame(index=False)
    df = df.merge(matched_refs, on=keys, how="inner", validate="many_to_one")
    if df.empty:
        raise ValueError("No matching UNet/Cellpose human annotation references")

    if metric_level == "PER_IMAGE":
        # Average the same matched annotators for both algorithms.
        df = df.groupby(["source_image_key", "algorithm"], as_index=False)[list(metrics)].mean()
        df["image_id"] = df["source_image_key"].astype(str)
    else:
        df["image_id"] = (
            df["source_image_key"].astype(str) + "|" +
            df["image_name"].astype(str) + "|" +
            df["human_annotator"].astype(str)
        )
    print(f"S8 {metric_level}: {df['image_id'].nunique():,} scored source images/tiles")
    return {"metrics": df[["image_id", "algorithm", *metrics]], "metric_level": metric_level}


def _generate_main_figure(
    figure_data: dict[str, Any],
    figure_output_dir: str,
    figure_name: str,
    *,
    max_strip_points: int | None = 400,
) -> None:
    data = figure_data["metrics"]

    def _subfigure(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str,
                   metric: str, title: str) -> None:
        ax.axis("off")
        utils.figure_label(ax, letter, x=0)
        panel = fig.add_subplot(gs.subgridspec(1, 1)[0])
        plot_boxstrip(panel, data, metric, title, max_strip_points=max_strip_points)

    def generate_subfigure_a(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _subfigure(fig, ax, gs, letter, *S8_METRICS[0])

    def generate_subfigure_b(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _subfigure(fig, ax, gs, letter, *S8_METRICS[1])

    def generate_subfigure_c(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _subfigure(fig, ax, gs, letter, *S8_METRICS[2])

    def generate_subfigure_d(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _subfigure(fig, ax, gs, letter, *S8_METRICS[3])

    def generate_subfigure_e(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _subfigure(fig, ax, gs, letter, *S8_METRICS[4])

    fig = plt.figure(figsize=(FIGURE_WIDTH, 4.25), layout="constrained")
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.42)
    slots = (gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1])
    functions = (
        generate_subfigure_a, generate_subfigure_b, generate_subfigure_c,
        generate_subfigure_d, generate_subfigure_e,
    )
    for i, (slot, plot_function) in enumerate(zip(slots, functions)):
        plot_function(fig, fig.add_subplot(slot), slot, "abcde"[i])
    Path(figure_output_dir).mkdir(parents=True, exist_ok=True)
    destination = Path(figure_output_dir) / figure_name
    fig.savefig(destination.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_S8_generation(
    validation_results_dir: str,
    figure_output_dir: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
    figure_name: str | None = None,
    max_strip_points: int | None = 400,
    **kwargs,
) -> None:
    """Public S8 entry point; no model inference required."""
    data = prepare_figure_S8_data(
        validation_results_dir, metric_level=metric_level, unet_size=unet_size,
    )
    _generate_main_figure(
        data, figure_output_dir, figure_name or f"Figure_S8_{metric_level}",
        max_strip_points=max_strip_points,
    )

