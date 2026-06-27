import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator

from alloviewer.flow_cytometry.fcs_file import FCSFile
from alloviewer.flow_cytometry.gating.clusterers import default_clusterer_factory
from alloviewer.flow_cytometry.gating.clustering import fit_clustering, predict_file_in_mask
from alloviewer.flow_cytometry.gating.config import GatingConfig, PARCConfig
from alloviewer.flow_cytometry.gating.labeling import label_clusters
from alloviewer.flow_cytometry.panel import Panel

from alloviewer.dev.validation.flow_validation import (
    _read_csv,
    _compute_qc_and_lymphocyte_metrics,
)

from . import figure_config as cfg
from . import figure_utils as utils


EDGE_GATE_CSV = "root/edge_exclusion"
SINGLET_GATE_CSV = "root/edge_exclusion/singlets"
LYMPHOCYTES_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes"
CD3_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes/CD3+"
CD19_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes/CD19+"

PANEL = Panel(
    fsc_a="FSC-A",
    fsc_h="FSC-H",
    ssc_a="SSC-A",
    igg="FITC-A",
    markers={
        "CD3 PerCP-A": "PerCP-A",
        "CD19 APC-Cy7-A": "APC-Cy7-A",
    },
)


def _events_to_dataframe(events: Any, fcs: FCSFile) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events.copy()

    if not isinstance(events, np.ndarray):
        raise TypeError(f"Unsupported events type: {type(events)}")

    if events.ndim != 2:
        raise ValueError(f"Expected 2D events array, got shape {events.shape}")

    channel_names = list(fcs.channels.index)

    if events.shape[1] != len(channel_names):
        raise ValueError(
            f"Mismatch between event columns and FCS channels: "
            f"{events.shape[1]} vs {len(channel_names)}"
        )

    return pd.DataFrame(events, columns=channel_names)


def _k_formatter(x, pos):
    if abs(x) < 1e-12:
        return "0"
    if abs(x) >= 1000:
        if float(x) % 1000 == 0:
            return f"{int(x / 1000)}k"
        return f"{x / 1000:.1f}k"
    return f"{int(x)}"


def _subsample_xy(
    x: np.ndarray,
    y: np.ndarray,
    n_max: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= n_max:
        return x, y

    idx = rng.choice(x.size, size=n_max, replace=False)
    return x[idx], y[idx]


def _prepare_panel_a_image(panel_a_image_path: str) -> np.ndarray:
    return np.asarray(Image.open(panel_a_image_path))


def _select_best_setting_overall_from_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "algorithm",
        "algorithm_params_json",
        "gate",
        "mean_f1",
    }
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {missing}")

    focus_gates = ["lymphocytes", "t_cells", "b_cells"]

    ranking_df = summary_df.loc[summary_df["gate"].isin(focus_gates)].copy()
    if ranking_df.empty:
        raise ValueError(
            "No rows found for focus gates lymphocytes/t_cells/b_cells in summary_df."
        )

    ranking_df = (
        ranking_df
        .groupby(["algorithm", "algorithm_params_json"], as_index=False)
        .agg(score=("mean_f1", "mean"))
        .sort_values("score", ascending=False)
    )

    return ranking_df.iloc[[0]].copy()


def _prepare_panel_c_data(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    required_metric_cols = {
        "experiment",
        "algorithm",
        "algorithm_params_json",
        "file_name",
        "gate",
        "f1",
    }
    missing_metrics = required_metric_cols - set(metrics_df.columns)
    if missing_metrics:
        raise ValueError(f"metrics_df is missing required columns: {missing_metrics}")

    best_df = _select_best_setting_overall_from_summary(summary_df)

    gate_display_map = {
        "edge_exclusion": "Edge exclusion",
        "singlets": "Singlets",
        "lymphocytes": "Lymphocytes",
        "t_cells": "T cells",
        "b_cells": "B cells",
    }

    gate_order = [
        "Edge exclusion",
        "Singlets",
        "Lymphocytes",
        "T cells",
        "B cells",
    ]

    plot_df = metrics_df.merge(
        best_df[["algorithm", "algorithm_params_json"]],
        on=["algorithm", "algorithm_params_json"],
        how="inner",
    ).copy()

    if plot_df.empty:
        raise ValueError(
            "Panel C data is empty after selecting the best algorithm setting."
        )

    plot_df["gate_display"] = (
        plot_df["gate"]
        .map(gate_display_map)
        .fillna(plot_df["gate"])
    )
    plot_df["gate_display"] = pd.Categorical(
        plot_df["gate_display"],
        categories=gate_order,
        ordered=True,
    )

    return plot_df


def _prepare_panel_b_data(
    *,
    data_dir: str,
    experiment: str,
    sample_name: str,
    algorithm: str = "parc",
    resolution_parameter: float = 4.0,
) -> dict[str, Any]:
    config = GatingConfig(
        clusterer=PARCConfig(resolution_parameter=resolution_parameter)
    )

    ground_truth_df = _read_csv(data_dir, experiment)

    (
        file_records,
        _metric_rows,
        _stage_times,
        marker_cofactors,
        marker_thresholds,
    ) = _compute_qc_and_lymphocyte_metrics(
        data_dir=data_dir,
        experiment=experiment,
        config=config,
        algorithm=algorithm,
        algorithm_params={"resolution_parameter": resolution_parameter},
    )

    marker_names = list(PANEL.markers.keys())
    rng = np.random.default_rng(int(config.random_state))
    clusterer = default_clusterer_factory(config.clusterer)

    (
        feature_scaler,
        clusterer,
        outlier_thr,
        X_train_scatter,
        T_train,
        y_train,
    ) = fit_clustering(
        transform_cfg=config.transform,
        feature_scaling_cfg=config.feature_scaling,
        cluster_sampling_cfg=config.cluster_sampling,
        prediction_cfg=config.prediction,
        panel=PANEL,
        file_records=file_records,
        marker_cofactors=marker_cofactors,
        clusterer=clusterer,
        rng=rng,
    )

    cluster_to_type = label_clusters(
        debris_cfg=config.debris_cluster_label,
        cluster_label_cfg=config.cluster_label,
        X_scatter=X_train_scatter,
        T_markers=T_train,
        y_train=y_train,
        marker_names=marker_names,
        marker_thresholds=marker_thresholds,
    )

    rec = None
    for r in file_records:
        if r["sample"].name == sample_name:
            rec = r
            break

    if rec is None:
        available = [r["sample"].name for r in file_records[:10]]
        raise ValueError(
            f"Sample '{sample_name}' not found in {experiment}. "
            f"First available samples: {available}"
        )

    events = rec["events"]
    fcs = rec["fcs"]

    gt_df = ground_truth_df.loc[
        ground_truth_df["file_name"] == sample_name,
        :,
    ].copy()

    if gt_df.shape[0] == 0:
        raise ValueError(f"No ground-truth rows found for sample '{sample_name}'")

    if gt_df.shape[0] != events.shape[0]:
        raise ValueError(
            f"Event count mismatch for '{sample_name}': "
            f"{gt_df.shape[0]} rows in CSV vs {events.shape[0]} events"
        )

    events_df = _events_to_dataframe(events, fcs)

    mask_edge = np.asarray(rec["mask_edge"], dtype=bool)
    mask_sing = np.asarray(rec["mask_sing"], dtype=bool)
    mask_lymph = np.asarray(rec["mask_lymph"], dtype=bool)

    pred = predict_file_in_mask(
        transform_cfg=config.transform,
        prediction_cfg=config.prediction,
        panel=PANEL,
        fcs=fcs,
        events=events,
        mask_in=mask_lymph,
        marker_cofactors=marker_cofactors,
        clusterer=clusterer,
        feature_scaler=feature_scaler,
        outlier_thr=float(outlier_thr),
        cluster_to_type=cluster_to_type,
        marker_names=marker_names,
    )

    mask_t = np.asarray(pred.mask_by_marker["CD3 PerCP-A"], dtype=bool)
    mask_b = np.asarray(pred.mask_by_marker["CD19 APC-Cy7-A"], dtype=bool)

    n_events = events.shape[0]
    mask_all = np.ones(n_events, dtype=bool)

    panels = [
        {
            "name": "edge_exclusion",
            "title": "Edge exclusion",
            "x": events_df[PANEL.fsc_a].to_numpy(),
            "y": events_df[PANEL.ssc_a].to_numpy(),
            "x_label": PANEL.fsc_a,
            "y_label": PANEL.ssc_a,
            "parent_mask": mask_all,
            "gt_mask": gt_df[EDGE_GATE_CSV].astype(bool).to_numpy(),
            "pred_mask": mask_edge,
        },
        {
            "name": "singlets",
            "title": "Singlets",
            "x": events_df[PANEL.fsc_a].to_numpy(),
            "y": events_df[PANEL.fsc_h].to_numpy(),
            "x_label": PANEL.fsc_a,
            "y_label": PANEL.fsc_h,
            "parent_mask": mask_edge,
            "gt_mask": gt_df[SINGLET_GATE_CSV].astype(bool).to_numpy(),
            "pred_mask": mask_sing,
        },
        {
            "name": "lymphocytes",
            "title": "Lymphocytes",
            "x": events_df[PANEL.fsc_a].to_numpy(),
            "y": events_df[PANEL.ssc_a].to_numpy(),
            "x_label": PANEL.fsc_a,
            "y_label": PANEL.ssc_a,
            "parent_mask": mask_sing,
            "gt_mask": gt_df[LYMPHOCYTES_GATE_CSV].astype(bool).to_numpy(),
            "pred_mask": mask_lymph,
        },
        {
            "name": "t_cells",
            "title": "T cells",
            "x": events_df["PerCP-A"].to_numpy(),
            "y": events_df[PANEL.ssc_a].to_numpy(),
            "x_label": "CD3 PerCP-A",
            "y_label": PANEL.ssc_a,
            "parent_mask": mask_lymph,
            "gt_mask": gt_df[CD3_GATE_CSV].astype(bool).to_numpy(),
            "pred_mask": mask_t,
        },
        {
            "name": "b_cells",
            "title": "B cells",
            "x": events_df["APC-Cy7-A"].to_numpy(),
            "y": events_df[PANEL.ssc_a].to_numpy(),
            "x_label": "CD19 APC-Cy7-A",
            "y_label": PANEL.ssc_a,
            "parent_mask": mask_lymph,
            "gt_mask": gt_df[CD19_GATE_CSV].astype(bool).to_numpy(),
            "pred_mask": mask_b,
        },
    ]

    for panel in panels:
        parent_mask = np.asarray(panel["parent_mask"], dtype=bool)
        gt_mask = np.asarray(panel["gt_mask"], dtype=bool) & parent_mask
        pred_mask = np.asarray(panel["pred_mask"], dtype=bool) & parent_mask

        panel["gt_mask"] = gt_mask
        panel["pred_mask"] = pred_mask
        panel["nongated_mask"] = parent_mask & (~pred_mask)

    return {
        "experiment": experiment,
        "sample_name": sample_name,
        "config": config,
        "panels": panels,
        "igg_all": events_df[PANEL.igg].to_numpy(),
    }


def _generate_subfigure_a(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    panel_a_image: np.ndarray,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    fig_sgs = gs.subgridspec(1, 1)
    sub_ax = fig.add_subplot(fig_sgs[0, 0])

    utils.prep_image_axis(sub_ax)
    sub_ax.imshow(panel_a_image)


def _generate_subfigure_b(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    plot_data: dict[str, Any],
    *,
    black_dot_size: float = 3.5,
    black_dot_alpha: float = 0.35,
    gt_point_size: float = 4.0,
    pred_point_size: float = 4.0,
    gt_point_alpha: float = 0.35,
    pred_point_alpha: float = 0.35,
    kde_levels: int = 6,
    kde_thresh: float = 0.1,
    kde_bw_adjust: float = 1.0,
    kde_linewidth: float = 1.0,
    biex_linthresh: float = 100.0,
    max_points_per_layer: int = 500,
    random_state: int = 42,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    rng = np.random.default_rng(random_state)
    panels = plot_data["panels"]
    igg_all = np.asarray(plot_data["igg_all"])

    fig_sgs = gs.subgridspec(2, 4, wspace=0.08, hspace=0.10)
    axes = np.asarray(
        [[fig.add_subplot(fig_sgs[r, c]) for c in range(4)] for r in range(2)]
    )

    scatter_axes = [
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
    ]
    ax_legend = axes[0, 3]
    ax_igg = axes[1, 2]
    ax_medians = axes[1, 3]

    for sub_ax, panel in zip(scatter_axes, panels):
        x = np.asarray(panel["x"])
        y = np.asarray(panel["y"])
        mask_gt = np.asarray(panel["gt_mask"], dtype=bool)
        mask_pred = np.asarray(panel["pred_mask"], dtype=bool)
        mask_nongated = np.asarray(panel["nongated_mask"], dtype=bool)

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        mask_gt = mask_gt[valid]
        mask_pred = mask_pred[valid]
        mask_nongated = mask_nongated[valid]

        is_marker_panel = panel["name"] in {"t_cells", "b_cells"}

        if x.size > 0:
            xmin, xmax = np.nanpercentile(x, [0.5, 99.5])
            ymin, ymax = np.nanpercentile(y, [0.5, 99.5])

            if xmin == xmax:
                xmin, xmax = x.min(), x.max()
            if ymin == ymax:
                ymin, ymax = y.min(), y.max()

            ymin = 0.0

            if not is_marker_panel:
                xmin = 0.0
                if panel["x_label"] == PANEL.fsc_a:
                    xmax = min(xmax, 150000.0)

            if panel["y_label"] == PANEL.ssc_a:
                ymax = min(ymax, 75000.0)

            sub_ax.set_xlim(xmin, xmax)
            sub_ax.set_ylim(ymin, ymax)

        if np.any(mask_nongated):
            x_ng, y_ng = _subsample_xy(
                x[mask_nongated],
                y[mask_nongated],
                max_points_per_layer,
                rng,
            )
            sub_ax.scatter(
                x_ng,
                y_ng,
                s=black_dot_size,
                c="black",
                alpha=black_dot_alpha,
                linewidths=0,
                rasterized=False,
                zorder=1,
            )

        if np.any(mask_gt):
            x_gt, y_gt = _subsample_xy(
                x[mask_gt],
                y[mask_gt],
                max_points_per_layer,
                rng,
            )
            sub_ax.scatter(
                x_gt,
                y_gt,
                s=gt_point_size,
                c="red",
                alpha=gt_point_alpha,
                linewidths=0,
                rasterized=False,
                zorder=2,
            )

        if np.any(mask_pred):
            x_pred, y_pred = _subsample_xy(
                x[mask_pred],
                y[mask_pred],
                max_points_per_layer,
                rng,
            )
            sub_ax.scatter(
                x_pred,
                y_pred,
                s=pred_point_size,
                c="blue",
                alpha=pred_point_alpha,
                linewidths=0,
                rasterized=False,
                zorder=3,
            )

        if np.sum(mask_gt) > 10:
            x_gt, y_gt = _subsample_xy(
                x[mask_gt],
                y[mask_gt],
                max_points_per_layer,
                rng,
            )
            if x_gt.size > 10:
                sns.kdeplot(
                    x=x_gt,
                    y=y_gt,
                    ax=sub_ax,
                    fill=False,
                    color="red",
                    levels=kde_levels,
                    thresh=kde_thresh,
                    bw_adjust=kde_bw_adjust,
                    linewidths=kde_linewidth,
                    warn_singular=False,
                    zorder=4,
                )

        if np.sum(mask_pred) > 10:
            x_pred, y_pred = _subsample_xy(
                x[mask_pred],
                y[mask_pred],
                max_points_per_layer,
                rng,
            )
            if x_pred.size > 10:
                sns.kdeplot(
                    x=x_pred,
                    y=y_pred,
                    ax=sub_ax,
                    fill=False,
                    color="blue",
                    levels=kde_levels,
                    thresh=kde_thresh,
                    bw_adjust=kde_bw_adjust,
                    linewidths=kde_linewidth,
                    warn_singular=False,
                    zorder=5,
                )

        if is_marker_panel:
            sub_ax.set_xscale("symlog", linthresh=biex_linthresh)
            sub_ax.set_yscale("linear")
        else:
            sub_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            sub_ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))

        sub_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        sub_ax.yaxis.set_major_formatter(FuncFormatter(_k_formatter))

        sub_ax.set_title(panel["title"], fontsize=cfg.TITLE_SIZE, pad=2)
        sub_ax.set_xlabel(panel["x_label"], fontsize=cfg.AXIS_LABEL_SIZE, labelpad=1)
        sub_ax.set_ylabel(panel["y_label"], fontsize=cfg.AXIS_LABEL_SIZE, labelpad=1)
        sub_ax.tick_params(
            axis="both",
            which="major",
            labelsize=cfg.AXIS_LABEL_SIZE,
            pad=1,
            length=2,
        )

    ax_legend.axis("off")
    ax_legend.plot([], [], color="red", linewidth=1.5, label="Manual gate")
    ax_legend.plot([], [], color="blue", linewidth=1.5, label="Automated gate")
    ax_legend.scatter([], [], c="black", s=20, alpha=0.35, label="Parent population")
    ax_legend.legend(
        loc="center",
        frameon=False,
        fontsize=max(cfg.AXIS_LABEL_SIZE, 7),
    )
    ax_legend.set_title("", fontsize=cfg.TITLE_SIZE, pad=2)

    t_panel = next(p for p in panels if p["name"] == "t_cells")
    t_gt_mask = np.asarray(t_panel["gt_mask"], dtype=bool)
    t_pred_mask = np.asarray(t_panel["pred_mask"], dtype=bool)

    igg_valid = np.isfinite(igg_all)
    t_gt_igg = igg_all[igg_valid & t_gt_mask]
    t_pred_igg = igg_all[igg_valid & t_pred_mask]

    if t_gt_igg.size > 5:
        sns.kdeplot(
            x=t_gt_igg,
            ax=ax_igg,
            color="red",
            fill=False,
            linewidth=1.5,
            bw_adjust=1.0,
            warn_singular=False,
        )

    if t_pred_igg.size > 5:
        sns.kdeplot(
            x=t_pred_igg,
            ax=ax_igg,
            color="blue",
            fill=False,
            linewidth=1.5,
            bw_adjust=1.0,
            warn_singular=False,
        )

    med_gt = float(np.median(t_gt_igg)) if t_gt_igg.size > 0 else np.nan
    med_pred = float(np.median(t_pred_igg)) if t_pred_igg.size > 0 else np.nan

    if np.isfinite(med_gt):
        ax_igg.axvline(med_gt, color="red", linestyle="--", linewidth=1.0)
    if np.isfinite(med_pred):
        ax_igg.axvline(med_pred, color="blue", linestyle="--", linewidth=1.0)

    ax_igg.set_title("T cell IgG", fontsize=cfg.TITLE_SIZE, pad=2)
    ax_igg.set_xlabel("FITC-A", fontsize=cfg.AXIS_LABEL_SIZE, labelpad=1)
    ax_igg.set_ylabel("Density", fontsize=cfg.AXIS_LABEL_SIZE, labelpad=1)
    ax_igg.set_xscale("symlog", linthresh=biex_linthresh)
    ax_igg.tick_params(
        axis="both",
        which="major",
        labelsize=cfg.AXIS_LABEL_SIZE,
        pad=1,
        length=2,
    )
    ax_igg.tick_params(axis="y", labelleft=False)
    ax_igg.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax_medians.axis("off")
    median_text = (
        "T cell IgG medians\n\n"
        f"Manual: {med_gt:.0f}\n"
        f"Automated: {med_pred:.0f}"
    )
    ax_medians.text(
        0.5,
        0.5,
        median_text,
        ha="center",
        va="center",
        fontsize=max(cfg.AXIS_LABEL_SIZE + 1, 8),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.9,
            edgecolor="black",
        ),
        transform=ax_medians.transAxes,
    )


def _generate_subfigure_c(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    plot_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    gate_order = [
        "Edge exclusion",
        "Singlets",
        "Lymphocytes",
        "T cells",
        "B cells",
    ]

    fig_sgs = gs.subgridspec(1, 1)
    sub_ax = fig.add_subplot(fig_sgs[0, 0])

    sns.boxplot(
        data=plot_df,
        x="gate_display",
        y="f1",
        order=gate_order,
        ax=sub_ax,
        showcaps=True,
        fliersize=0,
        width=0.65,
        boxprops=dict(facecolor="white"),
    )

    sns.stripplot(
        data=plot_df,
        x="gate_display",
        y="f1",
        order=gate_order,
        ax=sub_ax,
        size=3,
        linewidth=0.5,
        edgecolor="black",
    )

    sub_ax.set_xlabel("")
    sub_ax.set_ylabel("F1 score across files", fontsize=cfg.AXIS_LABEL_SIZE)
    sub_ax.set_ylim(0.2, 1.02)
    sub_ax.set_title(
        "Overall performance of the selected final pipeline",
        fontsize=cfg.TITLE_SIZE,
    )
    sub_ax.tick_params(
        axis="both",
        which="major",
        labelsize=cfg.AXIS_LABEL_SIZE,
    )


def _generate_main_figure(
    figure_output_dir: str,
    figure_name: str,
    *,
    panel_a_image: np.ndarray,
    panel_b_plot_data: dict[str, Any],
    panel_c_plot_df: pd.DataFrame,
) -> None:
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=1,
        nrows=3,
        figure=fig,
        height_ratios=[1.1, 1.6, 0.9],
    )

    a_coords = gs[0, 0]
    b_coords = gs[1, 0]
    c_coords = gs[2, 0]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    _generate_subfigure_a(
        fig=fig,
        ax=fig_a,
        gs=a_coords,
        subfigure_label="A",
        panel_a_image=panel_a_image,
    )

    _generate_subfigure_b(
        fig=fig,
        ax=fig_b,
        gs=b_coords,
        subfigure_label="B",
        plot_data=panel_b_plot_data,
    )

    _generate_subfigure_c(
        fig=fig,
        ax=fig_c,
        gs=c_coords,
        subfigure_label="C",
        plot_df=panel_c_plot_df,
    )

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return


def figure_5_generation(
    figure_output_dir: str,
    validation_results_dir: str,
    sketch_dir: str,
    flow_data_dir: str,
    **kwargs,
) -> None:

    panel_a_image_path = kwargs.get(
        "panel_a_image_path",
        os.path.join(sketch_dir, "sketch_3A.jpg"),
    )

    panel_b_experiment = "val_exp_15"
    panel_b_sample_name = "Worklist_001_FCXM_Routine_V2 UD_26204459_004_20260310_125733.fcs"

    panel_b_algorithm = "parc"
    panel_b_resolution_parameter = 4.0

    summary_path = os.path.join(validation_results_dir, "flow_validation_summary.csv")
    metrics_path = os.path.join(validation_results_dir, "flow_validation_metrics_long.csv")

    panel_a_image = _prepare_panel_a_image(panel_a_image_path)

    summary_df = pd.read_csv(summary_path)
    metrics_df = pd.read_csv(metrics_path)

    panel_b_plot_data = _prepare_panel_b_data(
        data_dir=flow_data_dir,
        experiment=panel_b_experiment,
        sample_name=panel_b_sample_name,
        algorithm=panel_b_algorithm,
        resolution_parameter=panel_b_resolution_parameter,
    )

    panel_c_plot_df = _prepare_panel_c_data(
        metrics_df=metrics_df,
        summary_df=summary_df,
    )

    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_5",
        panel_a_image=panel_a_image,
        panel_b_plot_data=panel_b_plot_data,
        panel_c_plot_df=panel_c_plot_df,
    )
