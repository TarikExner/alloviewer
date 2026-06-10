import os
import json
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import figure_utils as utils
from . import figure_config as cfg


GATE_DISPLAY_MAP = {
    "edge_exclusion": "Edge exclusion",
    "singlets": "Singlets",
    "lymphocytes": "Lymphocytes",
    "t_cells": "T cells",
    "b_cells": "B cells",
}

GATE_DISPLAY_ORDER = [
    "Edge exclusion",
    "Singlets",
    "Lymphocytes",
    "T cells",
    "B cells",
]

GATE_KEY_ORDER = [
    "edge_exclusion",
    "singlets",
    "lymphocytes",
    "t_cells",
    "b_cells",
]

FOCUS_GATES = [
    "lymphocytes",
    "t_cells",
    "b_cells",
]

ALGO_DISPLAY_MAP = {
    "parc": "PARC",
    "flowsom": "FlowSOM",
    "hdbscan": "HDBSCAN",
}

ALGO_KEY_ORDER = [
    "parc",
    "flowsom",
    "hdbscan",
]

ALGO_DISPLAY_ORDER = [
    "PARC",
    "FlowSOM",
    "HDBSCAN",
]


def _select_best_settings_per_algorithm(summary_df: pd.DataFrame) -> pd.DataFrame:
    focus_df = summary_df.loc[summary_df["gate"].isin(FOCUS_GATES)].copy()

    ranking_df = (
        focus_df.groupby(["algorithm", "algorithm_params_json"], as_index=False)
        .agg(score=("mean_f1", "mean"))
    )

    best_settings_df = (
        ranking_df.sort_values(["algorithm", "score"], ascending=[True, False])
        .groupby("algorithm", as_index=False)
        .head(1)
        .copy()
    )

    return best_settings_df


def _prepare_runtime_data(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {
        "experiment",
        "algorithm",
        "algorithm_params_json",
        "gate",
        "mean_f1",
        "total_runtime_s",
        "qc_plus_lymph_runtime_s",
        "fit_clustering_runtime_s",
        "label_clusters_runtime_s",
        "predict_runtime_s",
    }

    optional_memory_cols = [
        "memory_current_mb",
        "memory_delta_mb",
        "memory_peak_mb",
    ]

    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {missing}")

    focus_df = summary_df.loc[summary_df["gate"].isin(FOCUS_GATES)].copy()

    focus_df["gate_display"] = (
        focus_df["gate"]
        .map(GATE_DISPLAY_MAP)
        .fillna(focus_df["gate"])
    )

    focus_df["algorithm_display"] = (
        focus_df["algorithm"]
        .map(ALGO_DISPLAY_MAP)
        .fillna(focus_df["algorithm"])
    )

    agg_kwargs = {
        "mean_focus_f1": ("mean_f1", "mean"),
        "total_runtime_s": ("total_runtime_s", "mean"),
    }

    for col in optional_memory_cols:
        if col in focus_df.columns:
            agg_kwargs[col] = (col, "mean")

    scatter_df = (
        focus_df.groupby(
            ["experiment", "algorithm", "algorithm_display", "algorithm_params_json"],
            as_index=False,
        )
        .agg(**agg_kwargs)
    )

    best_settings_df = _select_best_settings_per_algorithm(summary_df)

    best_df = summary_df.merge(
        best_settings_df[["algorithm", "algorithm_params_json"]],
        on=["algorithm", "algorithm_params_json"],
        how="inner",
    ).copy()

    best_df["algorithm_display"] = (
        best_df["algorithm"]
        .map(ALGO_DISPLAY_MAP)
        .fillna(best_df["algorithm"])
    )

    runtime_breakdown_df = (
        best_df.groupby(
            ["algorithm", "algorithm_display", "algorithm_params_json"],
            as_index=False,
        )
        .agg(
            qc_plus_lymph_runtime_s=("qc_plus_lymph_runtime_s", "mean"),
            fit_clustering_runtime_s=("fit_clustering_runtime_s", "mean"),
            label_clusters_runtime_s=("label_clusters_runtime_s", "mean"),
            predict_runtime_s=("predict_runtime_s", "mean"),
            total_runtime_s=("total_runtime_s", "mean"),
        )
    )

    runtime_breakdown_df["algorithm_display"] = pd.Categorical(
        runtime_breakdown_df["algorithm_display"],
        categories=ALGO_DISPLAY_ORDER,
        ordered=True,
    )

    runtime_breakdown_df = runtime_breakdown_df.sort_values("algorithm_display")

    return scatter_df, runtime_breakdown_df


def _param_label(algo: str, param_json: str) -> str:
    params = json.loads(param_json)

    if algo == "parc":
        return f"res={params['resolution_parameter']}"

    if algo == "flowsom":
        return f"k={params['n_clusters']}"

    if algo == "hdbscan":
        return f"{params['cluster_selection_method']}, ms={params['min_samples']}"

    return str(params)


def _sort_key(algo: str, param_json: str):
    params = json.loads(param_json)

    if algo == "parc":
        return params["resolution_parameter"]

    if algo == "flowsom":
        return params["n_clusters"]

    if algo == "hdbscan":
        method_order = {"leaf": 0, "eom": 1}
        return (
            method_order.get(params["cluster_selection_method"], 999),
            params["min_samples"],
        )

    return str(params)


def _palette_for_algo(algo: str, n: int):
    if algo == "parc":
        return sns.color_palette("tab20", n_colors=n)

    if algo == "flowsom":
        return sns.color_palette("PuRd", n_colors=max(n + 2, 3))[2:2 + n]

    if algo == "hdbscan":
        return sns.color_palette("tab20c", n_colors=n)

    return sns.color_palette("tab10", n_colors=n)


def _generate_subfigure_a(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    summary_df: pd.DataFrame,
) -> None:
    import matplotlib.patches as mpatches

    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    panel_a_df = summary_df.copy()

    panel_a_df["gate_display"] = (
        panel_a_df["gate"]
        .map(GATE_DISPLAY_MAP)
        .fillna(panel_a_df["gate"])
    )

    panel_a_df["param_label"] = panel_a_df.apply(
        lambda x: _param_label(x["algorithm"], x["algorithm_params_json"]),
        axis=1,
    )

    panel_a_df = (
        panel_a_df.groupby(
            ["algorithm", "algorithm_params_json", "param_label", "gate", "gate_display"],
            as_index=False,
        )
        .agg(mean_f1=("mean_f1", "mean"))
    )

    fig_sgs = gs.subgridspec(
        4,
        4,
        height_ratios=[0.2, 0.9, 12.0, 4.0],
        width_ratios=[1.0, 1.0, 1.0, 0.08],
        hspace=0,
        wspace=0,
    )

    cbar_ax = fig.add_subplot(fig_sgs[2, 3])
    first_hm = None

    for i, algo in enumerate(ALGO_KEY_ORDER):
        algo_df = panel_a_df.loc[panel_a_df["algorithm"] == algo].copy()
        if algo_df.empty:
            continue

        setting_order_df = (
            algo_df[["algorithm_params_json", "param_label"]]
            .drop_duplicates()
            .assign(
                sort_key=lambda d: d["algorithm_params_json"].map(
                    lambda s: _sort_key(algo, s)
                )
            )
            .sort_values("sort_key")
        )

        param_order = setting_order_df["param_label"].tolist()

        pivot_df = (
            algo_df.pivot_table(
                index="gate_display",
                columns="param_label",
                values="mean_f1",
                aggfunc="mean",
            )
            .reindex(index=GATE_DISPLAY_ORDER, columns=param_order)
        )

        param_colors = _palette_for_algo(algo, len(param_order))
        param_to_color = {p: c for p, c in zip(param_order, param_colors)}

        title_ax = fig.add_subplot(fig_sgs[0, i])
        annot_ax = fig.add_subplot(fig_sgs[1, i])
        heat_ax = fig.add_subplot(fig_sgs[2, i])
        legend_ax = fig.add_subplot(fig_sgs[3, i])

        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.5,
            ALGO_DISPLAY_MAP[algo],
            ha="center",
            va="center",
            fontsize=cfg.TITLE_SIZE,
        )

        annot_ax.set_xlim(0, len(param_order))
        annot_ax.set_ylim(0, 1)
        annot_ax.set_xticks([])
        annot_ax.set_yticks([])

        for spine in annot_ax.spines.values():
            spine.set_visible(False)

        for j, param_label in enumerate(param_order):
            annot_ax.add_patch(
                plt.Rectangle(
                    (j, 0),
                    1,
                    1,
                    facecolor=param_to_color[param_label],
                    edgecolor="white",
                    linewidth=0.5,
                )
            )

        hm = sns.heatmap(
            pivot_df,
            ax=heat_ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar=(first_hm is None),
            cbar_ax=cbar_ax if first_hm is None else None,
            linewidths=0.4,
            linecolor="white",
            xticklabels=False,
            yticklabels=True,
        )

        if first_hm is None:
            first_hm = hm
            cbar = hm.collections[0].colorbar
            cbar.set_label("F1 score", rotation=90)
            cbar.set_ticks([0.0, 1.0])
            cbar.set_ticklabels(["0", "1"])

        heat_ax.set_title("")
        heat_ax.set_xlabel("")
        heat_ax.set_ylabel("")

        if i == 0:
            heat_ax.set_yticklabels(heat_ax.get_yticklabels(), rotation=0)
            heat_ax.tick_params(axis="y", labelsize=8)
        else:
            heat_ax.set_yticklabels([])
            heat_ax.tick_params(axis="y", length=0)

        heat_ax.tick_params(axis="x", length=0)

        legend_ax.axis("off")

        handles = [
            mpatches.Patch(
                facecolor=param_to_color[p],
                edgecolor="none",
                label=p,
            )
            for p in param_order
        ]

        ncol = 2 if len(handles) >= 6 else 1

        legend_ax.legend(
            handles=handles,
            loc="center",
            frameon=False,
            fontsize=7,
            title="Parameter settings",
            title_fontsize=8,
            ncol=ncol,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.8,
        )


def _generate_subfigure_b(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    scatter_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    fig_sgs = gs.subgridspec(1, 1)
    sub_ax = fig.add_subplot(fig_sgs[0, 0])

    plot_df = scatter_df.sample(frac=1.0, random_state=42).copy()

    sns.scatterplot(
        data=plot_df,
        x="total_runtime_s",
        y="mean_focus_f1",
        hue="algorithm_display",
        hue_order=ALGO_DISPLAY_ORDER,
        s=32,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
        ax=sub_ax,
    )

    sub_ax.set_xlabel("Total runtime (s)", fontsize=cfg.AXIS_LABEL_SIZE)
    sub_ax.set_ylabel(
        "Mean F1 score\n(Lymphocytes, T cells, B cells)",
        fontsize=cfg.AXIS_LABEL_SIZE,
    )
    sub_ax.set_xscale("log")
    sub_ax.set_title(
        "Accuracy versus runtime across all settings",
        fontsize=cfg.TITLE_SIZE,
    )
    sub_ax.set_ylim(0.0, 1.02)

    sub_ax.legend(
        frameon=False,
        title="Algorithm family",
        fontsize=cfg.TITLE_SIZE,
        title_fontsize=cfg.TITLE_SIZE,
    )

    utils.adjust_fontsize_ticklabels(sub_ax, cfg.AXIS_LABEL_SIZE)


def _generate_subfigure_c(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    runtime_breakdown_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    stage_cols = [
        "qc_plus_lymph_runtime_s",
        "fit_clustering_runtime_s",
        "label_clusters_runtime_s",
        "predict_runtime_s",
    ]

    stage_labels = [
        "QC + lymphocytes",
        "Clustering fit",
        "Cluster labeling",
        "Prediction",
    ]

    plot_df = runtime_breakdown_df.copy()

    plot_df["algorithm_display"] = pd.Categorical(
        plot_df["algorithm_display"],
        categories=ALGO_DISPLAY_ORDER,
        ordered=True,
    )

    plot_df = plot_df.sort_values("algorithm_display")

    fig_sgs = gs.subgridspec(1, 1)
    sub_ax = fig.add_subplot(fig_sgs[0, 0])

    bottoms = [0.0] * len(plot_df)

    for stage_col, stage_label in zip(stage_cols, stage_labels):
        values = plot_df[stage_col].to_numpy(dtype=float)

        sub_ax.bar(
            plot_df["algorithm_display"],
            values,
            bottom=bottoms,
            label=stage_label,
        )

        bottoms = [b + v for b, v in zip(bottoms, values)]

    sub_ax.set_xlabel("Clustering algorithm", fontsize=cfg.AXIS_LABEL_SIZE)
    sub_ax.set_ylabel("Mean runtime (s)", fontsize=cfg.AXIS_LABEL_SIZE)

    sub_ax.set_title(
        "Runtime breakdown for each algorithm family",
        fontsize=cfg.TITLE_SIZE,
    )

    sub_ax.legend(
        frameon=False,
        title="Stage",
        loc="upper right",
        fontsize=cfg.TITLE_SIZE,
        title_fontsize=cfg.TITLE_SIZE,
    )

    y_min, y_max = sub_ax.get_ylim()
    sub_ax.set_ylim(y_min, y_max * 1.5)

    utils.adjust_fontsize_ticklabels(sub_ax, cfg.AXIS_LABEL_SIZE)


def _generate_main_figure(
    summary_df: pd.DataFrame,
    scatter_df: pd.DataFrame,
    runtime_breakdown_df: pd.DataFrame,
    figure_output_dir: str = "",
    figure_name: str = "",
):
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        width_ratios=[1.25, 1.0],
        height_ratios=[1.25, 1.0],
    )

    a_coords = gs[0, :]
    b_coords = gs[1, 0]
    c_coords = gs[1, 1]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    _generate_subfigure_a(
        fig=fig,
        ax=fig_a,
        gs=a_coords,
        subfigure_label="A",
        summary_df=summary_df,
    )

    _generate_subfigure_b(
        fig=fig,
        ax=fig_b,
        gs=b_coords,
        subfigure_label="B",
        scatter_df=scatter_df,
    )

    _generate_subfigure_c(
        fig=fig,
        ax=fig_c,
        gs=c_coords,
        subfigure_label="C",
        runtime_breakdown_df=runtime_breakdown_df,
    )

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_S10_generation(
    figure_output_dir: str,
    validation_results_dir: str,
    **kwargs,
):
    summary_path = os.path.join(validation_results_dir, "flow_validation_summary.csv")

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing flow validation summary: {summary_path}")

    summary_df = pd.read_csv(summary_path)

    scatter_df, runtime_breakdown_df = _prepare_runtime_data(summary_df)

    _generate_main_figure(
        summary_df=summary_df,
        scatter_df=scatter_df,
        runtime_breakdown_df=runtime_breakdown_df,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S10",
    )
