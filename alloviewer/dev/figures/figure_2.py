import os
import pickle
from typing import Any, Sequence, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec

import seaborn as sns
from scipy.stats import wilcoxon

from matplotlib.ticker import FuncFormatter

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import get_score_frame


def _get_eval_score(
    df: pd.DataFrame,
    *,
    human_annotators=("1", "2"),
    method_annotators=("unet", "imageJ"),
    annotator_col="Annotator",
    human_score_col="score",
    method_score_col="adjusted_score",
    fallback_method_score_col="score",
    output_col="_eval_score",
) -> pd.DataFrame:
    """
    Create one score column used consistently for all comparisons.

    Humans use `human_score_col`.
    Methods use `method_score_col`, with fallback to `fallback_method_score_col`.
    """

    out = df.copy()

    human_annotators = [str(a) for a in human_annotators]
    method_annotators = [str(a) for a in method_annotators]

    out["_annotator"] = out[annotator_col].astype(str)
    out[output_col] = np.nan

    human_mask = out["_annotator"].isin(human_annotators)
    method_mask = out["_annotator"].isin(method_annotators)

    out.loc[human_mask, output_col] = out.loc[human_mask, human_score_col]

    if method_score_col in out.columns:
        out.loc[method_mask, output_col] = out.loc[method_mask, method_score_col]

    if fallback_method_score_col in out.columns:
        missing_method = method_mask & out[output_col].isna()
        out.loc[missing_method, output_col] = out.loc[
            missing_method, fallback_method_score_col
        ]

    return out


def compute_confusion_matrix_between_annotators(
    df: pd.DataFrame,
    *,
    x_annotator,
    y_annotator,
    id_cols: Sequence[str] = ("Folder", "well", "image_name", "role"),
    annotator_col: str = "Annotator",
    human_score_col: str = "score",
    method_score_col: str = "adjusted_score",
    fallback_method_score_col: str = "score",
    human_annotators=("1", "2"),
    method_annotators=("unet", "imageJ"),
    allowed_scores: Sequence[int] = (1, 2, 4, 6, 8),
) -> pd.DataFrame:
    """
    Compute confusion matrix between two annotators/methods.

    Rows = y_annotator
    Columns = x_annotator
    """

    x_annotator = str(x_annotator)
    y_annotator = str(y_annotator)

    data = _get_eval_score(
        df,
        human_annotators=human_annotators,
        method_annotators=method_annotators,
        annotator_col=annotator_col,
        human_score_col=human_score_col,
        method_score_col=method_score_col,
        fallback_method_score_col=fallback_method_score_col,
        output_col="_eval_score",
    )

    data = data[data["_annotator"].isin([x_annotator, y_annotator])].copy()

    wide = (
        data.pivot_table(
            index=list(id_cols),
            columns="_annotator",
            values="_eval_score",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None

    missing = [a for a in [x_annotator, y_annotator] if a not in wide.columns]
    if missing:
        raise ValueError(f"Missing annotator(s) after pivot: {missing}")

    wide = wide.dropna(subset=[x_annotator, y_annotator]).copy()

    wide[x_annotator] = wide[x_annotator].astype(int)
    wide[y_annotator] = wide[y_annotator].astype(int)

    cm = pd.crosstab(
        wide[y_annotator],
        wide[x_annotator],
        rownames=[str(y_annotator)],
        colnames=[str(x_annotator)],
        dropna=False,
    )

    cm = cm.reindex(
        index=list(allowed_scores),
        columns=list(allowed_scores),
        fill_value=0,
    )

    return cm


def plot_confusion_matrix_on_ax(
    cm: pd.DataFrame,
    ax: Axes,
    *,
    title: str = "",
    normalize: bool = True,
    cmap: str = "Reds",
    xlabel: str = "",
    ylabel: str = "",
    show_colorbar: bool = True,
    fig: Optional[Figure] = None,
) -> None:
    """
    Plot one confusion matrix into an existing axis.

    Rows = y-axis rater
    Columns = x-axis rater
    """

    cm_plot = cm.astype(float).copy()

    if normalize:
        row_sums = cm_plot.sum(axis=1).replace(0, np.nan)
        cm_plot = cm_plot.div(row_sums, axis=0)

    im = ax.imshow(
        cm_plot.values,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1 if normalize else None,
    )

    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, fontsize=cfg.AXIS_LABEL_SIZE)

    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index, fontsize=cfg.AXIS_LABEL_SIZE)

    ax.set_xlabel(xlabel, fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=cfg.AXIS_LABEL_SIZE)

    ax.set_title(title, fontsize=cfg.TITLE_SIZE)

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot.iloc[i, j]
            raw = cm.iloc[i, j]

            if pd.isna(val):
                text = f"nan\n(n={raw})"
            elif normalize:
                text = f"{val:.2f}\n(n={raw})"
            else:
                text = f"{int(raw)}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=max(cfg.AXIS_LABEL_SIZE - 2, 6),
            )

    if show_colorbar and fig is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if normalize:
            ticks = np.linspace(0, 1, 6)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        else:
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

        cbar.ax.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)


def compare_reference_and_human_consensus_estimate(
    df: pd.DataFrame,
    *,
    reference_human=1,
    comparison_human=2,
    methods=("unet",),
    id_cols=("Folder", "well", "image_name", "role"),
    annotator_col="Annotator",
    human_score_col="score",
    method_score_col="adjusted_score",
    fallback_method_score_col="score",
    experiment_col="Folder",
    allowed_scores=(1, 2, 4, 6, 8),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paper-style comparison.

    Per experiment:
        human_human_error_rate =
            fraction where comparison_human differs from reference_human

        estimated_human_consensus_error_rate =
            human_human_error_rate / 2

        method_error_rate_to_reference =
            fraction where method differs from reference_human
    """

    reference_human = str(reference_human)
    comparison_human = str(comparison_human)
    methods = [str(m) for m in methods]

    data = _get_eval_score(
        df,
        human_annotators=(reference_human, comparison_human),
        method_annotators=methods,
        annotator_col=annotator_col,
        human_score_col=human_score_col,
        method_score_col=method_score_col,
        fallback_method_score_col=fallback_method_score_col,
        output_col="_eval_score",
    )

    keep = [reference_human, comparison_human] + methods
    data = data[data["_annotator"].isin(keep)].copy()

    observed_scores = data["_eval_score"].dropna()
    invalid_scores = sorted(set(observed_scores.unique()) - set(allowed_scores))
    if invalid_scores:
        raise ValueError(
            f"Found scores outside allowed set {allowed_scores}: {invalid_scores}"
        )

    per_item = (
        data.pivot_table(
            index=list(id_cols),
            columns="_annotator",
            values="_eval_score",
            aggfunc="first",
        )
        .reset_index()
    )

    per_item.columns.name = None

    missing_humans = [
        h for h in [reference_human, comparison_human]
        if h not in per_item.columns
    ]
    if missing_humans:
        raise ValueError(
            f"Missing human annotator columns after pivoting: {missing_humans}"
        )

    per_item = per_item.dropna(subset=[reference_human]).copy()

    per_item["reference_human"] = reference_human
    per_item["comparison_human"] = comparison_human
    per_item["reference_score"] = per_item[reference_human]

    valid_human_pair = (
        per_item[reference_human].notna()
        & per_item[comparison_human].notna()
    )

    per_item["human_human_error"] = np.where(
        valid_human_pair,
        (per_item[comparison_human] != per_item[reference_human]).astype(int),
        np.nan,
    )

    per_item["estimated_human_consensus_error"] = (
        per_item["human_human_error"] / 2
    )

    for method in methods:
        if method not in per_item.columns:
            per_item[method] = np.nan

        valid_method = per_item[method].notna() & per_item["reference_score"].notna()

        per_item[f"{method}_error_to_reference"] = np.where(
            valid_method,
            (per_item[method] != per_item["reference_score"]).astype(int),
            np.nan,
        )

    def summarize(group: pd.DataFrame) -> pd.Series:
        out = {}

        out["reference_human"] = reference_human
        out["comparison_human"] = comparison_human

        out["n_human_pair_evaluable"] = int(group["human_human_error"].notna().sum())
        out["human_human_error_rate"] = group["human_human_error"].mean()
        out["estimated_human_consensus_error_rate"] = (
            out["human_human_error_rate"] / 2
        )

        for method in methods:
            col = f"{method}_error_to_reference"
            out[f"{method}_n_reference_evaluable"] = int(group[col].notna().sum())
            out[f"{method}_error_rate_to_reference"] = group[col].mean()

            human_bound = out["estimated_human_consensus_error_rate"]
            method_error = out[f"{method}_error_rate_to_reference"]

            if pd.notna(human_bound) and human_bound > 0:
                out[f"{method}_error_vs_estimated_human_consensus_error"] = (
                    method_error / human_bound
                )
            else:
                out[f"{method}_error_vs_estimated_human_consensus_error"] = np.nan

        return pd.Series(out)

    summary_by_experiment = (
        per_item
        .groupby(experiment_col, dropna=False)
        .apply(summarize)
        .reset_index()
    )

    overall = summarize(per_item).to_frame().T
    overall.insert(0, experiment_col, "OVERALL")

    summary = pd.concat([summary_by_experiment, overall], ignore_index=True)

    return per_item, summary

def _paired_wilcoxon_pvalue(
    data: pd.DataFrame,
    left_col: str,
    right_col: str,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> tuple[float, int, int]:
    """
    Run a paired Wilcoxon test safely.

    Returns:
        p_value
        n_paired
        n_nonzero_diff

    Important:
        This function forces both columns to real numeric arrays.
        It does not silently label failed tests as non-significant.
    """

    paired = data[[left_col, right_col]].copy()

    paired[left_col] = pd.to_numeric(paired[left_col], errors="coerce")
    paired[right_col] = pd.to_numeric(paired[right_col], errors="coerce")

    paired = paired.dropna()

    n_paired = len(paired)

    if n_paired < 3:
        return np.nan, n_paired, 0

    left = paired[left_col].to_numpy(dtype=float)
    right = paired[right_col].to_numpy(dtype=float)

    diff = right - left
    n_nonzero = int(np.sum(~np.isclose(diff, 0)))

    if n_nonzero == 0:
        return np.nan, n_paired, n_nonzero

    _, p = wilcoxon(
        right,
        left,
        zero_method=zero_method,
        alternative=alternative,
    )

    return float(p), n_paired, n_nonzero

def plot_unet_reference_boxplot_on_ax(
    summary: pd.DataFrame,
    ax: Axes,
    *,
    method="unet",
    experiment_col="Folder",
    include_overall=False,
    ylim=(0, 1),
    show_significance=True,
    comparisons=("annot2_vs_unet", "consensus_vs_unet", "annot2_vs_consensus"),
) -> pd.DataFrame:
    """
    Plot:
        Annotator 2 vs Annotator 1
        estimated human-consensus error
        UNET vs Annotator 1

    Optional statistical brackets:
        annot2_vs_unet:
            human_human_error_rate vs UNET error

        consensus_vs_unet:
            estimated human-consensus error vs UNET error

        annot2_vs_consensus:
            human_human_error_rate vs estimated human-consensus error

    Note:
        The human-consensus estimate is derived as half the human-human error.
        Therefore, annot2_vs_consensus is not an independent comparison.
    """

    method = str(method)

    data = summary.copy()
    if not include_overall:
        data = data[data[experiment_col] != "OVERALL"].copy()

    plot_cols = {
        "Annotator 2 vs\nAnnotator 1": "human_human_error_rate",
        "Estimated human\nconsensus error": "estimated_human_consensus_error_rate",
        f"{method} vs\nAnnotator 1": f"{method}_error_rate_to_reference",
    }

    missing = [col for col in plot_cols.values() if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for boxplot: {missing}")

    # Force plotted/tested values to real numeric dtype.
    # This prevents SciPy's wilcoxon from failing on object dtype columns.
    for col in plot_cols.values():
        data[col] = pd.to_numeric(data[col], errors="coerce")

    plot_df = data[[experiment_col] + list(plot_cols.values())].melt(
        id_vars=experiment_col,
        var_name="metric",
        value_name="error_rate",
    )

    label_map = {v: k for k, v in plot_cols.items()}
    plot_df["group"] = plot_df["metric"].map(label_map)

    order = list(plot_cols.keys())

    sns.boxplot(
        data=plot_df,
        x="group",
        y="error_rate",
        order=order,
        showfliers=False,
        ax=ax,
        color="white",
        boxprops={"edgecolor": "black"},
        medianprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )

    sns.stripplot(
        data=plot_df,
        x="group",
        y="error_rate",
        order=order,
        jitter=True,
        size=4,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )

    if show_significance:
        comparison_map = {
            "annot2_vs_unet": (
                "human_human_error_rate",
                f"{method}_error_rate_to_reference",
                "Annotator 2 vs\nAnnotator 1",
                f"{method} vs\nAnnotator 1",
            ),
            "consensus_vs_unet": (
                "estimated_human_consensus_error_rate",
                f"{method}_error_rate_to_reference",
                "Estimated human\nconsensus error",
                f"{method} vs\nAnnotator 1",
            ),
            "annot2_vs_consensus": (
                "human_human_error_rate",
                "estimated_human_consensus_error_rate",
                "Annotator 2 vs\nAnnotator 1",
                "Estimated human\nconsensus error",
            ),
        }

        y_range = ylim[1] - ylim[0]
        y = ylim[1] - 0.16 * y_range
        y_step = 0.07 * y_range
        h = 0.02 * y_range

        for comp in comparisons:
            if comp not in comparison_map:
                raise ValueError(
                    f"Unknown comparison '{comp}'. Valid options are: "
                    f"{list(comparison_map.keys())}"
                )

            left_col, right_col, left_label, right_label = comparison_map[comp]

            paired = data[[left_col, right_col]].dropna()

            if len(paired) < 3:
                continue

            p, n_paired, n_nonzero = _paired_wilcoxon_pvalue(
                data,
                left_col,
                right_col,
                alternative="two-sided",
                zero_method="wilcox",
            )

            if n_paired < 3:
                continue

            if n_nonzero == 0:
                label = "all zero"
            elif pd.isna(p):
                label = "test error"
            elif p < 0.001:
                label = "***"
            elif p < 0.01:
                label = "**"
            elif p < 0.05:
                label = "*"
            else:
                label = "n.s."

            print(
                f"{comp}: {left_col} vs {right_col}; "
                f"n={n_paired}, nonzero={n_nonzero}, p={p}"
            )

            x1 = order.index(left_label)
            x2 = order.index(right_label)

            ax.plot(
                [x1, x1, x2, x2],
                [y, y + h, y + h, y],
                color="black",
                linewidth=1.2,
            )

            ax.text(
                (x1 + x2) / 2,
                y + h,
                label,
                ha="center",
                va="bottom",
                fontsize=cfg.AXIS_LABEL_SIZE,
            )

            y += y_step

    ax.set_xlabel("")
    ax.set_ylabel("Error rate", fontsize=cfg.AXIS_LABEL_SIZE)
    ax.set_ylim(*ylim)

    ax.set_title(
        f"UNET scoring fidelity\nn = {data[experiment_col].nunique()} experiments",
        fontsize=cfg.TITLE_SIZE,
    )

    utils.adjust_fontsize_ticklabels(ax, cfg.AXIS_LABEL_SIZE)
    ax.tick_params(axis="x", rotation=30)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return plot_df


def _generate_main_figure(
    results_df: pd.DataFrame,
    plate: Any,
    scoring: pd.DataFrame,
    figure_output_dir: str = "",
    figure_name: str = "Figure_2",
    save: bool = False,
) -> None:
    def generate_subfigure_a(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(2, 2)

        neg_wells = plate.get("negative")
        pos_wells = plate.get("positive")

        if len(neg_wells) == 0:
            raise ValueError("No wells with role 'negative' found in plate.")
        if len(pos_wells) == 0:
            raise ValueError("No wells with role 'positive' found in plate.")

        neg_well = neg_wells[0]
        pos_well = pos_wells[0]

        calib_for_a = "PCNCGaussianRGCalibrator"

        df_neg = results_df[
            (results_df["well_id"] == neg_well.well_id)
            & (results_df["calibrator"] == calib_for_a)
        ]

        df_pos = results_df[
            (results_df["well_id"] == pos_well.well_id)
            & (results_df["calibrator"] == calib_for_a)
        ]

        ax_img_neg = fig.add_subplot(fig_sgs[0, 0])
        ax_img_neg.imshow(np.rot90(neg_well.image.transpose((1, 2, 0))))
        ax_img_neg.set_title("Negative control", fontsize=cfg.TITLE_SIZE)
        ax_img_neg.set_xticks([])
        ax_img_neg.set_yticks([])

        ax_scatter_neg = fig.add_subplot(fig_sgs[0, 1])
        sns.scatterplot(
            data=df_neg,
            x="mean_r",
            y="mean_g",
            ax=ax_scatter_neg,
            **cfg.SCATTER_KWARGS,
        )
        ax_scatter_neg.set_xlabel("Mean red intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_neg.set_ylabel("Mean green intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_neg.set_title("Negative control RG", fontsize=cfg.TITLE_SIZE)
        utils.adjust_fontsize_ticklabels(ax_scatter_neg, cfg.AXIS_LABEL_SIZE)
        ax_scatter_neg.set_xlim(-0.1, 1.1)
        ax_scatter_neg.set_ylim(-0.1, 1.1)

        ax_img_pos = fig.add_subplot(fig_sgs[1, 0])
        ax_img_pos.imshow(np.rot90(pos_well.image.transpose((1, 2, 0))))
        ax_img_pos.set_title("Positive control", fontsize=cfg.TITLE_SIZE)
        ax_img_pos.set_xticks([])
        ax_img_pos.set_yticks([])

        ax_scatter_pos = fig.add_subplot(fig_sgs[1, 1])
        sns.scatterplot(
            data=df_pos,
            x="mean_r",
            y="mean_g",
            ax=ax_scatter_pos,
            **cfg.SCATTER_KWARGS,
        )
        ax_scatter_pos.set_xlabel("Mean red intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_pos.set_ylabel("Mean green intensity", fontsize=cfg.AXIS_LABEL_SIZE)
        ax_scatter_pos.set_title("Positive control RG", fontsize=cfg.TITLE_SIZE)
        utils.adjust_fontsize_ticklabels(ax_scatter_pos, cfg.AXIS_LABEL_SIZE)
        ax_scatter_pos.set_xlim(-0.1, 1.1)
        ax_scatter_pos.set_ylim(-0.1, 1.1)

    def generate_subfigure_b(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        fig_sgs = gs.subgridspec(2, 2)

        well_id_c4 = "C4"
        if well_id_c4 not in plate.wells:
            raise ValueError(f"Well '{well_id_c4}' not found in plate.")

        well_c4 = plate.wells[well_id_c4]
        df_c4 = results_df[results_df["well_id"] == well_id_c4].copy()

        if df_c4.empty:
            raise ValueError(f"No ROI results for well '{well_id_c4}' in results_df.")

        calibrators = [
            "PCNCGaussian2DCalibrator",
            "PCNCGaussianRGCalibrator",
            "PCNCMedianCalibrator",
        ]

        ax_img_c4 = fig.add_subplot(fig_sgs[0, 0])
        ax_img_c4.imshow(np.rot90(well_c4.image.transpose((1, 2, 0))))
        ax_img_c4.set_title(f"Patient Sample Well", fontsize=cfg.TITLE_SIZE)
        ax_img_c4.set_xticks([])
        ax_img_c4.set_yticks([])

        scatter_axes = [
            fig.add_subplot(fig_sgs[0, 1]),
            fig.add_subplot(fig_sgs[1, 0]),
            fig.add_subplot(fig_sgs[1, 1]),
        ]

        for calib, ax_sc in zip(calibrators, scatter_axes):
            df_calib = df_c4[df_c4["calibrator"] == calib]

            if df_calib.empty:
                ax_sc.axis("off")
                continue

            label_palette = {
                "neg": "green",
                "pos": "orange",
                "uncertain": "blue",
            }

            sns.scatterplot(
                data=df_calib,
                x="mean_r",
                y="mean_g",
                hue="label",
                hue_order=["neg", "pos", "uncertain"],
                palette=label_palette,
                ax=ax_sc,
                **cfg.SCATTER_KWARGS,
            )

            ax_sc.set_xlabel("Mean red intensity", fontsize=cfg.AXIS_LABEL_SIZE)
            ax_sc.set_ylabel("Mean green intensity", fontsize=cfg.AXIS_LABEL_SIZE)
            ax_sc.set_title(calib, fontsize=cfg.TITLE_SIZE)
            utils.adjust_fontsize_ticklabels(ax_sc, cfg.AXIS_LABEL_SIZE)

            ax_sc.legend(
                markerscale=4,
                fontsize=cfg.AXIS_LABEL_SIZE,
                title="",
                loc="lower left",
            )

            ax_sc.set_xlim(-0.1, 1.1)
            ax_sc.set_ylim(-0.1, 1.1)

    def generate_subfigure_c(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        inner_gs = gs.subgridspec(1, 1)
        cm_ax = fig.add_subplot(inner_gs[0])

        cm = compute_confusion_matrix_between_annotators(
            scoring,
            x_annotator="1",
            y_annotator="2",
            human_score_col="score",
            method_score_col="adjusted_score",
            fallback_method_score_col="score",
        )

        plot_confusion_matrix_on_ax(
            cm,
            cm_ax,
            fig=fig,
            title="Annotator 2 vs Annotator 1",
            xlabel="Annotator 1",
            ylabel="Annotator 2",
            normalize=True,
            cmap="Reds",
            show_colorbar=True,
        )

    def generate_subfigure_d(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        inner_gs = gs.subgridspec(1, 1)
        cm_ax = fig.add_subplot(inner_gs[0])

        cm = compute_confusion_matrix_between_annotators(
            scoring,
            x_annotator="imageJ",
            y_annotator="unet",
            human_score_col="score",
            method_score_col="adjusted_score",
            fallback_method_score_col="score",
        )

        plot_confusion_matrix_on_ax(
            cm,
            cm_ax,
            fig=fig,
            title="UNET vs NCISP",
            xlabel="NCISP",
            ylabel="UNET",
            normalize=True,
            cmap="Reds",
            show_colorbar=True,
        )


    def generate_subfigure_e(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        inner_gs = gs.subgridspec(1, 1)
        cm_ax = fig.add_subplot(inner_gs[0])

        cm = compute_confusion_matrix_between_annotators(
            scoring,
            x_annotator="2",
            y_annotator="unet",
            human_score_col="score",
            method_score_col="adjusted_score",
            fallback_method_score_col="score",
        )

        plot_confusion_matrix_on_ax(
            cm,
            cm_ax,
            fig=fig,
            title="UNET vs Annotator 2",
            xlabel="Annotator 2",
            ylabel="UNET",
            normalize=True,
            cmap="Reds",
            show_colorbar=True,
        )

    def generate_subfigure_f(
        fig: Figure,
        ax: Axes,
        gs: SubplotSpec,
        subfigure_label: str,
    ) -> None:
        ax.axis("off")
        utils.figure_label(ax, subfigure_label, x=0)

        inner_gs = gs.subgridspec(1, 1)
        plot_ax = fig.add_subplot(inner_gs[0])

        _, summary = compare_reference_and_human_consensus_estimate(
            scoring,
            reference_human=1,
            comparison_human=2,
            methods=("unet",),
            human_score_col="score",
            method_score_col="adjusted_score",
            fallback_method_score_col="score",
            experiment_col="Folder",
        )

        plot_unet_reference_boxplot_on_ax(
            summary,
            plot_ax,
            method="unet",
            experiment_col="Folder",
            ylim=(0, 0.87),
            show_significance=True,
        )

    fig = plt.figure(
        layout="constrained",
        figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL),
    )

    gs = GridSpec(
        ncols=6,
        nrows=3,
        figure=fig,
        height_ratios=[1, 0.8, 0.8],
    )

    a_coords = gs[0, 0:3]
    b_coords = gs[0, 3:6]

    c_coords = gs[1, 0:3]
    d_coords = gs[1, 3:6]

    e_coords = gs[2, 0:3]
    f_coords = gs[2, 3:6]

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

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    return


def figure_2_generation(
    figure_output_dir: str,
    figure_data_dir: str,
    **kwargs,
) -> None:
    """
    Load figure data and generate Figure 2.

    Required:
        calibration_data.dict must contain:
            data["plate"]
            data["res_df"]

    You must pass:
        scoring_df=scoring_df

    Example:
        figure_2_generation(
            figure_output_dir=figure_output_dir,
            figure_data_dir=figure_data_dir,
            scoring_df=scoring_df,
            save=True,
        )
    """

    data_path = os.path.join(figure_data_dir, "calibration_data.dict")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    plate = data["plate"]
    results_df = data["res_df"]

    scoring_df = kwargs.get("scoring_df", get_score_frame())

    _generate_main_figure(
        results_df=results_df,
        plate=plate,
        scoring=scoring_df,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_2",
    )
