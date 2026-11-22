import os
import warnings
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
from sklearn.metrics import confusion_matrix

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


# Figure 5C: IgG median / NC median positivity thresholds.
IGG_RATIO_CUTOFFS: dict[str, dict[str, float]] = {
    "Ulm": {"T_cells": 3.5, "B_cells": 3.8},
    "Frankfurt": {"T_cells": 3.5, "B_cells": 3.5},
}

# Optional single-sample overrides: (center, experiment, FCS filename, cell_type).
# Example: ("Ulm", "val_exp_1", "FCS_000001.fcs", "T_cells"): 4.0
IGG_RATIO_SAMPLE_CUTOFFS: dict[tuple[str, str, str, str], float] = {}


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
        "dataset",
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

    # Frankfurt has no FSC-H and therefore no singlet gate.
    # Exclude any stale singlet rows from earlier result files.
    plot_df = plot_df.loc[
        ~((plot_df["dataset"] == "Frankfurt") & (plot_df["gate"] == "singlets"))
    ].copy()

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


def _prepare_panel_c_igg_decisions(
    igg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply editable IgG median-ratio cutoffs to the selected model only.

    The reference is the manual-gate median ratio, not a separate clinical
    diagnosis. Each (center, experiment, sample, cell type) contributes at most
    one observation. Controls are excluded from the decision matrix.
    """
    required = {
        "dataset", "experiment", "sample", "role", "cell_type",
        "algorithm", "algorithm_params_json",
        "igg_median_ratio_ground_truth", "igg_median_ratio_predicted",
    }
    missing = required - set(igg_df.columns)
    if missing:
        raise ValueError(f"IgG ratio CSV is missing columns: {sorted(missing)}")

    best = _select_best_setting_overall_from_summary(summary_df)
    selected = igg_df.merge(
        best[["algorithm", "algorithm_params_json"]],
        on=["algorithm", "algorithm_params_json"],
        how="inner", validate="many_to_one",
    ).copy()
    if selected.empty:
        raise ValueError("No IgG ratios for the clustering setting selected in panel C")

    # Remove controls, but preserve all patient/sample rows, including missing
    # ratios, for a transparent record of exclusions.
    role = selected["role"].fillna("").astype(str).str.strip().str.upper()
    role = role.str.replace(r"[\s_-]+", "", regex=True)
    control_roles = {"NC", "PC", "NEGATIVE", "POSITIVE", "NEGATIVECONTROL", "POSITIVECONTROL"}
    selected = selected.loc[~role.isin(control_roles)].copy()
    if selected.empty:
        raise ValueError("No patient/sample rows remain after removing NC/PC controls")
    if selected["role"].isna().any():
        raise ValueError("IgG ratios contain rows with an unspecified sample role")

    key_cols = ["dataset", "experiment", "sample", "cell_type"]
    duplicated = selected.duplicated(key_cols, keep=False)
    if duplicated.any():
        examples = selected.loc[duplicated, key_cols].head(6).to_dict("records")
        raise ValueError(f"Repeated sample/cell-type rows in the selected setting: {examples}")

    def cutoff_for(row: pd.Series) -> float:
        center, experiment, sample, cell_type = (
            str(row["dataset"]), str(row["experiment"]),
            str(row["sample"]), str(row["cell_type"]),
        )
        override_key = (center, experiment, sample, cell_type)
        if override_key in IGG_RATIO_SAMPLE_CUTOFFS:
            cutoff = IGG_RATIO_SAMPLE_CUTOFFS[override_key]
        else:
            try:
                cutoff = IGG_RATIO_CUTOFFS[center][cell_type]
            except KeyError as exc:
                raise ValueError(f"No IgG cutoff configured for {center}/{cell_type}") from exc
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError(f"Invalid IgG ratio cutoff {cutoff} for {override_key}")
        return cutoff

    selected["decision_cutoff"] = selected.apply(cutoff_for, axis=1)
    gt = pd.to_numeric(selected["igg_median_ratio_ground_truth"], errors="coerce")
    predicted = pd.to_numeric(selected["igg_median_ratio_predicted"], errors="coerce")
    selected["valid_ratio_pair"] = np.isfinite(gt) & np.isfinite(predicted)
    selected["reference_positive"] = pd.Series(pd.NA, index=selected.index, dtype="boolean")
    selected["automated_positive"] = pd.Series(pd.NA, index=selected.index, dtype="boolean")
    valid = selected["valid_ratio_pair"]
    selected.loc[valid, "reference_positive"] = (gt.loc[valid] >= selected.loc[valid, "decision_cutoff"]).to_numpy()
    selected.loc[valid, "automated_positive"] = (
        predicted.loc[valid] >= selected.loc[valid, "decision_cutoff"]
    ).to_numpy()
    selected["decision_status"] = np.where(valid, "ok", "missing_or_nonfinite_ratio")

    records = []
    for center in ("Ulm", "Frankfurt"):
        for cell_type in ("T_cells", "B_cells"):
            group = selected.loc[
                (selected["dataset"] == center) & (selected["cell_type"] == cell_type)
            ]
            usable = group.loc[group["valid_ratio_pair"]]
            cm = confusion_matrix(
                usable["reference_positive"].astype(bool).to_numpy(),
                usable["automated_positive"].astype(bool).to_numpy(),
                labels=[False, True],
            )
            tn, fp, fn, tp = map(int, cm.ravel())
            cutoffs = group["decision_cutoff"].dropna().unique()
            cutoff_text = (f"{cutoffs[0]:g}" if len(cutoffs) == 1 else
                           "variable" if len(cutoffs) > 1 else "n/a")
            records.append({
                "dataset": center, "cell_type": cell_type,
                "algorithm": str(best.iloc[0]["algorithm"]),
                "algorithm_params_json": str(best.iloc[0]["algorithm_params_json"]),
                "cutoff": cutoff_text,
                "n_samples_total": len(group),
                "n_samples_evaluable": len(usable),
                "n_samples_missing_ratio": len(group) - len(usable),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
            })
            if group.empty:
                warnings.warn(f"No IgG ratio rows for {center}/{cell_type}")
    return selected, pd.DataFrame.from_records(records)


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
        dataset="Ulm",
        panel=PANEL,
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
    confusion_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    # Keep the original center-wise gating F1 plots in columns 1 and 2.
    # The right-hand column contains four 2x2 decision matrices, as in sketch.
    gate_orders = {
        "Ulm": ["Edge exclusion", "Singlets", "Lymphocytes", "T cells", "B cells"],
        "Frankfurt": ["Edge exclusion", "Lymphocytes", "T cells", "B cells"],
    }
    fig_sgs = gs.subgridspec(
        1, 3, width_ratios=[1.05, 1.05, 1.8], wspace=0.24,
    )
    centers = [("Ulm", "Center 1"), ("Frankfurt", "Center 2")]
    left_ax = None

    for i, (dataset, title) in enumerate(centers):
        gate_order = gate_orders[dataset]
        center_df = plot_df.loc[
            (plot_df["dataset"] == dataset)
            & plot_df["gate_display"].isin(gate_order)
        ].copy()
        if center_df.empty:
            raise ValueError(
                f"Panel C cannot show {title}: no validation data for dataset '{dataset}'."
            )

        if i == 0:
            sub_ax = fig.add_subplot(fig_sgs[0, i])
            left_ax = sub_ax
        else:
            sub_ax = fig.add_subplot(fig_sgs[0, i], sharey=left_ax)

        sns.boxplot(
            data=center_df, x="gate_display", y="f1", order=gate_order,
            ax=sub_ax, showcaps=True, fliersize=0, width=0.65,
            boxprops=dict(facecolor="white"),
        )
        sns.stripplot(
            data=center_df, x="gate_display", y="f1", order=gate_order,
            ax=sub_ax, size=3, linewidth=0.5, edgecolor="black",
        )
        sub_ax.set_xlabel("")
        sub_ax.set_ylabel(
            "F1 score across files" if i == 0 else "",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        sub_ax.set_ylim(0.2, 1.02)
        sub_ax.set_title(title, fontsize=cfg.TITLE_SIZE)
        if dataset == "Frankfurt":
            sub_ax.text(
                0.99, 0.03, "Singlet gating not performed",
                transform=sub_ax.transAxes, ha="right", va="bottom",
                fontsize=max(cfg.AXIS_LABEL_SIZE - 1, 6), color="0.4",
            )
        sub_ax.tick_params(axis="both", which="major", labelsize=cfg.AXIS_LABEL_SIZE)
        sub_ax.tick_params(axis="x", labelrotation=35)
        for tick in sub_ax.get_xticklabels():
            tick.set_ha("right")

    # Third column: rows are centers, columns are T/B cells. Each inset is
    # a 2x2 confusion matrix. Insets are excluded from constrained layout,
    # preventing Matplotlib from collapsing them at journal column widths.
    matrix_parent = fig.add_subplot(fig_sgs[0, 2])
    matrix_parent.axis("off")
    matrix_parent.set_xlim(0, 1)
    matrix_parent.set_ylim(0, 1)
    for col, name in enumerate(("T cells", "B cells")):
        matrix_parent.text(
            (0.16, 0.59)[col] + 0.18, 0.99, name,
            ha="center", va="top", fontsize=cfg.TITLE_SIZE,
            transform=matrix_parent.transAxes,
        )
    for row, (dataset, title) in enumerate(centers):
        y = (0.58, 0.10)[row]
        matrix_parent.text(
            0.025, y + 0.17, title, rotation=90, ha="center", va="center",
            fontsize=cfg.TITLE_SIZE, transform=matrix_parent.transAxes,
        )
        for col, cell_type in enumerate(("T_cells", "B_cells")):
            values = confusion_df.loc[
                (confusion_df["dataset"] == dataset)
                & (confusion_df["cell_type"] == cell_type)
            ]
            if len(values) != 1:
                raise ValueError(f"Expected one matrix for {dataset}/{cell_type}; got {len(values)}")
            summary = values.iloc[0]
            matrix = np.array([
                [int(summary["tn"]), int(summary["fp"])],
                [int(summary["fn"]), int(summary["tp"])],
            ])
            heat_ax = matrix_parent.inset_axes([(0.16, 0.59)[col], y, 0.36, 0.34])
            heat_ax.set_in_layout(False)
            sns.heatmap(
                matrix, ax=heat_ax, cmap="Blues", cbar=False,
                annot=np.array([
                    [f"TN\n{matrix[0, 0]}", f"FP\n{matrix[0, 1]}"],
                    [f"FN\n{matrix[1, 0]}", f"TP\n{matrix[1, 1]}"],
                ]),
                fmt="", linewidths=0.7, linecolor="black", square=False,
                annot_kws={"fontsize": max(cfg.AXIS_LABEL_SIZE - 1, 6)},
            )
            heat_ax.set_xticklabels(["−", "+"], rotation=0)
            heat_ax.set_yticklabels(["−", "+"] if col == 0 else ["", ""], rotation=0)
            heat_ax.tick_params(axis="both", labelsize=max(cfg.AXIS_LABEL_SIZE - 1, 6), length=0)
            heat_ax.set_xlabel("Automated" if row == 1 else "", fontsize=max(cfg.AXIS_LABEL_SIZE - 1, 6), labelpad=5)
            heat_ax.set_ylabel("Manual" if col == 0 else "", fontsize=max(cfg.AXIS_LABEL_SIZE - 1, 6), labelpad=0)
            cutoff = summary["cutoff"]
            n = int(summary["n_samples_evaluable"])
            n_missing = int(summary["n_samples_missing_ratio"])
            heat_ax.set_title(
                f"≥{cutoff}, n={n}" + (f" ({n_missing} missing)" if n_missing else ""),
                fontsize=max(cfg.AXIS_LABEL_SIZE - 1, 6), pad=2,
            )


def _generate_main_figure(
    figure_output_dir: str,
    figure_name: str,
    *,
    panel_a_image: np.ndarray,
    panel_b_plot_data: dict[str, Any],
    panel_c_plot_df: pd.DataFrame,
    panel_c_confusion_df: pd.DataFrame,
) -> None:
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL * 1.20),
    )

    gs = GridSpec(
        ncols=1,
        nrows=3,
        figure=fig,
        height_ratios=[1.1, 1.6, 1.45],
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
        confusion_df=panel_c_confusion_df,
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
    panel_b_sample_name = "FCS_000151.fcs"

    panel_b_algorithm = "parc"
    panel_b_resolution_parameter = 4.0

    summary_path = os.path.join(validation_results_dir, "flow_validation_summary.csv")
    metrics_path = os.path.join(validation_results_dir, "flow_validation_metrics_long.csv")
    igg_path = os.path.join(validation_results_dir, "flow_validation_igg_median_ratios.csv")

    panel_a_image = _prepare_panel_a_image(panel_a_image_path)

    summary_df = pd.read_csv(summary_path)
    metrics_df = pd.read_csv(metrics_path)
    igg_df = pd.read_csv(igg_path)

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

    decision_df, confusion_df = _prepare_panel_c_igg_decisions(
        igg_df=igg_df,
        summary_df=summary_df,
    )
    os.makedirs(figure_output_dir, exist_ok=True)
    decision_df.to_csv(os.path.join(figure_output_dir, "Figure_5_ratio_decisions.csv"), index=False)
    confusion_df.to_csv(os.path.join(figure_output_dir, "Figure_5_confusion_counts.csv"), index=False)

    _generate_main_figure(
        figure_output_dir=figure_output_dir,
        figure_name="Figure_5",
        panel_a_image=panel_a_image,
        panel_b_plot_data=panel_b_plot_data,
        panel_c_plot_df=panel_c_plot_df,
        panel_c_confusion_df=confusion_df,
    )

