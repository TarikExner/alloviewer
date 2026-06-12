import os
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

ALGO_ORDER = [
    "PARC",
    "FlowSOM",
    "HDBSCAN",
]

ALGO_KEY_ORDER = [
    "parc",
    "flowsom",
    "hdbscan",
]


def _select_best_settings_per_algorithm(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = summary_df.loc[summary_df["gate"].isin(FOCUS_GATES)].copy()

    ranking_df = (
        ranking_df.groupby(["algorithm", "algorithm_params_json"], as_index=False)
        .agg(score=("mean_f1", "mean"))
    )

    best_settings_df = (
        ranking_df.sort_values(["algorithm", "score"], ascending=[True, False])
        .groupby("algorithm", as_index=False)
        .head(1)
        .copy()
    )

    return best_settings_df


def _prepare_figure_S11_data(summary_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "experiment",
        "algorithm",
        "algorithm_params_json",
        "gate",
        "mean_f1",
    }

    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {missing}")

    best_settings_df = _select_best_settings_per_algorithm(summary_df)

    plot_df = summary_df.merge(
        best_settings_df[["algorithm", "algorithm_params_json"]],
        on=["algorithm", "algorithm_params_json"],
        how="inner",
    ).copy()

    plot_df["gate_display"] = (
        plot_df["gate"]
        .map(GATE_DISPLAY_MAP)
        .fillna(plot_df["gate"])
    )

    plot_df["gate_display"] = pd.Categorical(
        plot_df["gate_display"],
        categories=GATE_DISPLAY_ORDER,
        ordered=True,
    )

    plot_df["algorithm_display"] = (
        plot_df["algorithm"]
        .map(ALGO_DISPLAY_MAP)
        .fillna(plot_df["algorithm"])
    )

    plot_df["algorithm_display"] = pd.Categorical(
        plot_df["algorithm_display"],
        categories=ALGO_ORDER,
        ordered=True,
    )

    return plot_df


def _generate_subfigure_a(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    summary_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    ranking_df = summary_df.loc[summary_df["gate"].isin(FOCUS_GATES)].copy()

    ranking_df = (
        ranking_df.groupby(["algorithm", "algorithm_params_json"], as_index=False)
        .agg(score=("mean_f1", "mean"))
    )

    best_settings_df = (
        ranking_df.sort_values(["algorithm", "score"], ascending=[True, False])
        .groupby("algorithm", as_index=False)
        .head(1)
        .copy()
    )

    best_settings_df["row_label"] = best_settings_df["algorithm"].map(ALGO_DISPLAY_MAP)

    best_settings_df["algo_order"] = best_settings_df["algorithm"].map(
        {algo: i for i, algo in enumerate(ALGO_KEY_ORDER)}
    )

    best_settings_df = best_settings_df.sort_values("algo_order")
    row_order = best_settings_df["row_label"].tolist()

    best_summary_df = summary_df.merge(
        best_settings_df[["algorithm", "algorithm_params_json", "row_label"]],
        on=["algorithm", "algorithm_params_json"],
        how="inner",
    ).copy()

    best_summary_df["gate_display"] = (
        best_summary_df["gate"]
        .map(GATE_DISPLAY_MAP)
        .fillna(best_summary_df["gate"])
    )

    experiment_order = sorted(
        best_summary_df["experiment"].dropna().unique().tolist()
    )

    fig_sgs = gs.subgridspec(
        5,
        2,
        width_ratios=[1.0, 0.035],
        height_ratios=[1, 1, 1, 1, 1],
        hspace=0,
        wspace=0,
    )

    cbar_ax = fig.add_subplot(fig_sgs[:, 1])
    first_hm = None

    for i, gate in enumerate(GATE_KEY_ORDER):
        gate_label = GATE_DISPLAY_MAP[gate]

        gate_df = best_summary_df.loc[best_summary_df["gate"] == gate].copy()

        pivot_df = (
            gate_df.pivot_table(
                index="row_label",
                columns="experiment",
                values="mean_f1",
                aggfunc="mean",
            )
            .reindex(index=row_order, columns=experiment_order)
        )

        heat_ax = fig.add_subplot(fig_sgs[i, 0])

        hm = sns.heatmap(
            pivot_df,
            ax=heat_ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar=(first_hm is None),
            cbar_ax=cbar_ax if first_hm is None else None,
            linewidths=0.35,
            linecolor="white",
            xticklabels=True,
            yticklabels=True,
        )

        if first_hm is None:
            first_hm = hm
            cbar = hm.collections[0].colorbar
            cbar.set_label("F1 score", rotation=90)
            cbar.set_ticks([0.0, 1.0])
            cbar.set_ticklabels(["0", "1"])

        heat_ax.set_title(gate_label, fontsize=cfg.TITLE_SIZE, pad=4)
        heat_ax.set_xlabel("")
        heat_ax.set_ylabel("")

        if i < len(GATE_KEY_ORDER) - 1:
            heat_ax.set_xticklabels([])
            heat_ax.tick_params(axis="x", length=0)
        else:
            heat_ax.tick_params(axis="x", rotation=90, labelsize=7)

        heat_ax.tick_params(axis="y", rotation=0, labelsize=8)


def _generate_subfigure_b(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    plot_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    fig_sgs = gs.subgridspec(1, 5, wspace=0)

    for i, gate in enumerate(GATE_DISPLAY_ORDER):
        gate_df = plot_df.loc[plot_df["gate_display"] == gate].copy()

        gate_df["algorithm_display"] = pd.Categorical(
            gate_df["algorithm_display"],
            categories=ALGO_ORDER,
            ordered=True,
        )

        sub_ax = fig.add_subplot(fig_sgs[0, i])

        sns.boxplot(
            data=gate_df,
            x="algorithm_display",
            y="mean_f1",
            order=ALGO_ORDER,
            ax=sub_ax,
            showcaps=True,
            fliersize=0,
            width=0.65,
        )

        sns.stripplot(
            data=gate_df,
            x="algorithm_display",
            y="mean_f1",
            order=ALGO_ORDER,
            ax=sub_ax,
            dodge=False,
            size=3,
            alpha=0.7,
            color="black",
        )

        sub_ax.set_title(gate, fontsize=cfg.TITLE_SIZE)
        sub_ax.set_xlabel("")
        sub_ax.set_ylabel(
            "Mean F1 score" if i == 0 else "",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        sub_ax.set_ylim(0.80, 1.02)
        sub_ax.tick_params(axis="x", rotation=45)
        utils.adjust_fontsize_ticklabels(sub_ax, cfg.AXIS_LABEL_SIZE)

        if i > 0:
            sub_ax.set_yticklabels([])


def _generate_subfigure_c(
    fig: Figure,
    ax: Axes,
    gs: SubplotSpec,
    subfigure_label: str,
    plot_df: pd.DataFrame,
) -> None:
    ax.axis("off")
    utils.figure_label(ax, subfigure_label, x=0)

    fig_sgs = gs.subgridspec(1, 5, wspace=0)

    for i, gate in enumerate(GATE_DISPLAY_ORDER):
        gate_df = plot_df.loc[plot_df["gate_display"] == gate].copy()

        pivot_df = (
            gate_df.pivot_table(
                index="experiment",
                columns="algorithm_display",
                values="mean_f1",
                aggfunc="mean",
            )
            .reindex(columns=ALGO_ORDER)
            .sort_index()
        )

        sub_ax = fig.add_subplot(fig_sgs[0, i])

        for _, row in pivot_df.iterrows():
            y = row.to_numpy(dtype=float)
            x = range(len(ALGO_ORDER))

            sub_ax.plot(
                x,
                y,
                marker="o",
                linewidth=0.7,
                markersize=2.5,
                alpha=0.35,
                color="0.6",
                zorder=1,
            )

        mean_vals = pivot_df.mean(axis=0).reindex(ALGO_ORDER)

        sub_ax.plot(
            range(len(ALGO_ORDER)),
            mean_vals.to_numpy(dtype=float),
            marker="o",
            linewidth=2.5,
            markersize=5,
            alpha=1.0,
            color="black",
            zorder=3,
        )

        sub_ax.set_title(gate, fontsize=cfg.TITLE_SIZE)
        sub_ax.set_xticks(range(len(ALGO_ORDER)))
        sub_ax.set_xticklabels(ALGO_ORDER, rotation=45)
        sub_ax.set_xlabel("")
        sub_ax.set_ylabel(
            "Mean F1 score" if i == 0 else "",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        sub_ax.set_ylim(0.8, 1.02)

        utils.adjust_fontsize_ticklabels(sub_ax, cfg.AXIS_LABEL_SIZE)

        if i > 0:
            sub_ax.set_yticklabels([])


def _generate_main_figure(
    summary_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    figure_output_dir: str = "",
    figure_name: str = "",
):
    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=1,
        nrows=3,
        figure=fig,
        height_ratios=[1.25, 1.0, 1.0],
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
        summary_df=summary_df,
    )

    _generate_subfigure_b(
        fig=fig,
        ax=fig_b,
        gs=b_coords,
        subfigure_label="B",
        plot_df=plot_df,
    )

    _generate_subfigure_c(
        fig=fig,
        ax=fig_c,
        gs=c_coords,
        subfigure_label="C",
        plot_df=plot_df,
    )

    os.makedirs(figure_output_dir, exist_ok=True)

    pdf_path = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(figure_output_dir, f"{figure_name}.png")

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_S9_generation(
    figure_output_dir: str,
    validation_results_dir: str,
    **kwargs,
):
    summary_path = os.path.join(validation_results_dir, "flow_validation_summary.csv")

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing flow validation summary: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    plot_df = _prepare_figure_S11_data(summary_df)

    _generate_main_figure(
        summary_df=summary_df,
        plot_df=plot_df,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_S9",
    )
