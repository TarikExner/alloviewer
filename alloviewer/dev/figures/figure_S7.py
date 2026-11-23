"""Supplementary Figure S7: UNet vs Cellpose on human-annotated real images.

Five quantitative panels, no example images.
PER_IMAGE: per-source-image mean across matched human annotators. These source
  rows pool annotated tiles and must NOT be described as full-image GT scores.
PER_TILE: individual tile/annotator measurements (overlapping tiles possible).

Shared CSV parsing and boxplot style are in figure_data_generation.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec

from alloviewer.dev.figures import figure_utils as utils
from .figure_data_generation import (
    load_figure_s7_real_metrics,
    plot_segmentation_supplement_metric,
)

MetricLevel = Literal["PER_IMAGE", "PER_TILE"]
S7_METRICS = (
    ("center_precision_r4", "Center precision (4 px)"),
    ("center_recall_r4", "Center recall (4 px)"),
    ("instance_f1_iou_0p5", "Instance F1 (IoU ≥ 0.5)"),
    ("mean_instance_dice_all_gt_iou_0p5", "Mean GT-cell Dice"),
    ("pixel_dice", "Foreground pixel Dice"),
)


def prepare_figure_S7_data(
    validation_results_dir: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
) -> dict[str, Any]:
    """Load the *real* manually annotated comparison results once."""
    metrics = load_figure_s7_real_metrics(
        validation_results_dir,
        metrics=tuple(key for key, _ in S7_METRICS),
        metric_level=metric_level,
        unet_size=unet_size,
    )
    return {"metrics": metrics, "metric_level": metric_level}


def _generate_main_figure(
    figure_data: dict[str, Any],
    figure_output_dir: str,
    figure_name: str,
    *,
    max_strip_points: int | None = 400,
) -> None:
    """Create the figure using separate functions for its five panels."""
    data = figure_data["metrics"]

    def _metric_panel(fig: Figure, ax: Axes, gs: SubplotSpec,
                      letter: str, metric: str, title: str) -> None:
        ax.axis("off")
        utils.figure_label(ax, letter, x=-0.3)
        inner = fig.add_subplot(gs.subgridspec(1, 1)[0])
        plot_segmentation_supplement_metric(
            inner, data, metric, title, max_strip_points=max_strip_points,
        )

    def generate_subfigure_a(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _metric_panel(fig, ax, gs, letter, *S7_METRICS[0])

    def generate_subfigure_b(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _metric_panel(fig, ax, gs, letter, *S7_METRICS[1])

    def generate_subfigure_c(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _metric_panel(fig, ax, gs, letter, *S7_METRICS[2])

    def generate_subfigure_d(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _metric_panel(fig, ax, gs, letter, *S7_METRICS[3])

    def generate_subfigure_e(fig: Figure, ax: Axes, gs: SubplotSpec, letter: str) -> None:
        _metric_panel(fig, ax, gs, letter, *S7_METRICS[4])

    fig = plt.figure(figsize=(6.375, 4.25), layout="constrained")
    grid = GridSpec(2, 3, figure=fig, hspace=0, wspace=0)
    slots = [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1]]
    panels = (
        generate_subfigure_a, generate_subfigure_b, generate_subfigure_c,
        generate_subfigure_d, generate_subfigure_e,
    )
    for letter, slot, panel_fn in zip("ABCDE", slots, panels):
        panel_fn(fig, fig.add_subplot(slot), slot, letter)

    Path(figure_output_dir).mkdir(parents=True, exist_ok=True)
    destination = Path(figure_output_dir) / figure_name
    fig.savefig(destination.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_S7_generation(
    validation_results_dir: str,
    figure_output_dir: str,
    *,
    metric_level: MetricLevel = "PER_IMAGE",
    unet_size: str = "small",
    figure_name: str | None = None,
    max_strip_points: int | None = 400,
    **kwargs,
) -> None:
    """Public S7 entry point: real images, five quantitative panels."""
    data = prepare_figure_S7_data(
        validation_results_dir,
        metric_level=metric_level,
        unet_size=unet_size,
    )
    _generate_main_figure(
        data, figure_output_dir,
        figure_name=("Figure_S7" if metric_level == "PER_IMAGE" else "Figure_S7_PER_TILE"),
        max_strip_points=max_strip_points,
    )

