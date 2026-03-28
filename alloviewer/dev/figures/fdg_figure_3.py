from __future__ import annotations

import gc
import json
import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Generator, Literal, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, jaccard_score

from ...flow_cytometry.fcs_file import FCSFile
from ...flow_cytometry.gating.clusterers import default_clusterer_factory
from ...flow_cytometry.gating.clustering import fit_clustering, predict_file_in_mask
from ...flow_cytometry.gating.config import (
    FlowSOMConfig,
    GatingConfig,
    HDBSCANConfig,
    PARCConfig,
)
from ...flow_cytometry.gating.labeling import label_clusters
from ...flow_cytometry.gating.lymphocytes import gate_lymphocytes
from ...flow_cytometry.gating.marker_calibration import calibrate_markers
from ...flow_cytometry.gating.qc import QCGater
from ...flow_cytometry.panel import Panel
from ...flow_cytometry.sample import Dataset, Sample

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


EDGE_GATE_CSV = "root/edge_exclusion"
SINGLET_GATE_CSV = "root/edge_exclusion/singlets"
LYMPHOCYTES_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes"
CD3_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes/CD3+"
CD19_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes/CD19+"
CD3_IGG_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes/CD3+/IgG+"
CD19_IGG_GATE_CSV = "root/edge_exclusion/singlets/lymphocytes/CD19+/IgG+"

EXPERIMENTS = Literal["exp1", "exp2", "exp3", "exp4"]

PANEL = Panel(
    fsc_a="FSC-A",
    fsc_h="FSC-H",
    ssc_a="SSC-A",
    igg="FITC-A",
    markers={"CD3 PerCP-A": "PerCP-A", "CD19 APC-Cy7-A": "APC-Cy7-A"},
)


@contextmanager
def _stage_timer() -> Generator[dict[str, float], None, None]:
    payload: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield payload
    finally:
        payload["runtime_s"] = time.perf_counter() - t0


class MemoryMonitor:
    """
    Windows-safe memory monitor.

    Preferred source:
    - psutil RSS for current process, if available

    Fallback:
    - tracemalloc peak Python allocation, which is not total process memory
    """

    def __init__(self) -> None:
        self._use_psutil = psutil is not None
        self._process = psutil.Process(os.getpid()) if self._use_psutil else None
        self._rss_start_mb = self._get_rss_mb() if self._use_psutil else np.nan
        self._peak_python_mb = np.nan
        self._is_tracing_here = False

        if not self._use_psutil:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._is_tracing_here = True

    def _get_rss_mb(self) -> float:
        assert self._process is not None
        return float(self._process.memory_info().rss) / (1024.0 * 1024.0)

    def stop(self) -> dict[str, Any]:
        out: dict[str, Any] = {}

        if self._use_psutil:
            rss_end_mb = self._get_rss_mb()
            out["memory_metric"] = "rss_mb"
            out["memory_current_mb"] = rss_end_mb
            out["memory_delta_mb"] = rss_end_mb - float(self._rss_start_mb)
            out["memory_peak_mb"] = np.nan
        else:
            current, peak = tracemalloc.get_traced_memory()
            self._peak_python_mb = float(peak) / (1024.0 * 1024.0)
            out["memory_metric"] = "tracemalloc_python_mb"
            out["memory_current_mb"] = float(current) / (1024.0 * 1024.0)
            out["memory_delta_mb"] = np.nan
            out["memory_peak_mb"] = self._peak_python_mb
            if self._is_tracing_here:
                tracemalloc.stop()

        return out


def _read_csv(data_dir: str, experiment: EXPERIMENTS) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, f"{experiment}_expr_data.csv"))


def _read_fcs(data_dir: str, experiment: EXPERIMENTS, fcs_file: str) -> FCSFile:
    return FCSFile(os.path.join(data_dir, experiment, fcs_file))


def _create_fcs_file_list(data_dir: str, experiment: EXPERIMENTS) -> list[str]:
    file_list = os.listdir(os.path.join(data_dir, experiment))
    return sorted([file for file in file_list if file.endswith(".fcs")])


def _create_fcs_file_list_full_path(data_dir: str, experiment: EXPERIMENTS) -> list[str]:
    fcs_files = _create_fcs_file_list(data_dir, experiment)
    return [os.path.join(data_dir, experiment, file) for file in fcs_files]


def _create_dataset(data_dir: str, experiment: EXPERIMENTS) -> Dataset:
    role_map = pd.read_csv(os.path.join(data_dir, "role_map.csv"))

    fcs_files = _create_fcs_file_list(data_dir, experiment)
    raw_paths = _create_fcs_file_list_full_path(data_dir, experiment)

    samples: list[Sample] = []
    for file in fcs_files:
        role = role_map.loc[role_map["file_name"] == file, "role"].iloc[0]
        path = [path for path in raw_paths if path.endswith(file)]
        assert len(path) == 1
        samples.append(Sample(name=file, role=role, file_paths=path))

    return Dataset(samples=samples)


def _calculate_overlap(
    ground_truth_df: pd.DataFrame,
    gate: str,
    comp_vector: np.ndarray,
    overlap_fn: Literal["f1", "jaccard"],
) -> float:
    gt_vector = ground_truth_df[gate].astype(bool).to_numpy()
    comp_vector = np.asarray(comp_vector, dtype=bool)

    if overlap_fn == "f1":
        return float(f1_score(gt_vector, comp_vector, average="weighted"))
    if overlap_fn == "jaccard":
        return float(jaccard_score(gt_vector, comp_vector, average="weighted"))

    raise ValueError(f"Unknown overlap_fn: {overlap_fn}")


def _confusion_counts(
    ground_truth_df: pd.DataFrame,
    gate: str,
    comp_vector: np.ndarray,
) -> dict[str, int]:
    gt_vector = ground_truth_df[gate].astype(bool).to_numpy()
    comp_vector = np.asarray(comp_vector, dtype=bool)

    tn, fp, fn, tp = confusion_matrix(gt_vector, comp_vector, labels=[False, True]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def _build_metric_row(
    *,
    experiment: str,
    algorithm: str,
    algorithm_params: dict[str, Any],
    file_name: str,
    gate_name: str,
    gate_column: str,
    gt_df: pd.DataFrame,
    comp_vector: np.ndarray,
) -> dict[str, Any]:
    comp_vector = np.asarray(comp_vector, dtype=bool)
    counts = _confusion_counts(gt_df, gate_column, comp_vector)
    return {
        "experiment": experiment,
        "algorithm": algorithm,
        "algorithm_params_json": json.dumps(algorithm_params, sort_keys=True),
        "file_name": file_name,
        "gate": gate_name,
        "f1": _calculate_overlap(gt_df, gate_column, comp_vector, "f1"),
        "jaccard": _calculate_overlap(gt_df, gate_column, comp_vector, "jaccard"),
        **counts,
        "n_events": int(gt_df.shape[0]),
        "n_pred_positive": int(comp_vector.sum()),
        "n_gt_positive": int(gt_df[gate_column].astype(bool).sum()),
    }


def _iter_sweep_configs(base_config: Optional[GatingConfig] = None):
    base = base_config or GatingConfig()

    flowsom_clusters = [1, 2, 4, 8, 15, 30]
    parc_resolutions = [0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 8.0]
    hdbscan_methods = ["leaf", "eom"]
    hdbscan_min_samples = [10, 50, 100, 200]

    for n_clusters in flowsom_clusters:
        cfg = replace(base, clusterer=FlowSOMConfig(n_clusters=n_clusters, seed=base.random_state))
        yield {
            "algorithm": "flowsom",
            "algorithm_params": {
                "n_clusters": n_clusters,
                "label_kind": cfg.clusterer.label_kind,
            },
            "config": cfg,
        }

    for resolution in parc_resolutions:
        cfg = replace(base, clusterer=PARCConfig(resolution_parameter=resolution))
        yield {
            "algorithm": "parc",
            "algorithm_params": {"resolution_parameter": resolution},
            "config": cfg,
        }

    for cluster_selection_method in hdbscan_methods:
        for min_samples in hdbscan_min_samples:
            cfg = replace(
                base,
                clusterer=HDBSCANConfig(
                    cluster_selection_method=cluster_selection_method,
                    min_samples=min_samples,
                ),
            )
            yield {
                "algorithm": "hdbscan",
                "algorithm_params": {
                    "cluster_selection_method": cluster_selection_method,
                    "min_samples": min_samples,
                },
                "config": cfg,
            }


def _compute_qc_and_lymphocyte_metrics(
    *,
    data_dir: str,
    experiment: str,
    config: GatingConfig,
    algorithm: str,
    algorithm_params: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, float],
    dict[str, Any],
    dict[str, Any],
]:
    ground_truth_df = _read_csv(data_dir, experiment)
    ds = _create_dataset(data_dir, experiment)
    qc_gater = QCGater(panel=PANEL, config=config.qc)

    file_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    stage_times: dict[str, float] = {}

    with _stage_timer() as payload:
        for sample in ds.samples:
            for fcs in sample.files:
                qc = qc_gater.compute_qc(fcs)
                file_records.append(
                    {
                        "sample": sample,
                        "fcs": fcs,
                        "events": qc.events,
                        "mask_qc": qc.mask_qc,
                        "mask_edge": np.asarray(qc.mask_edge, dtype=bool),
                        "mask_sing": np.asarray(qc.mask_sing, dtype=bool),
                        "notes": qc.notes,
                    }
                )
    stage_times["qc_runtime_s"] = payload["runtime_s"]

    marker_names = list(PANEL.markers.keys())
    with _stage_timer() as payload:
        cal = calibrate_markers(
            transform_cfg=config.transform,
            cofactor_cfg=config.marker_cofactor,
            threshold_cfg=config.marker_threshold,
            random_state=config.random_state,
            panel=PANEL,
            file_records=file_records,
            marker_names=marker_names,
        )
    stage_times["marker_calibration_runtime_s"] = payload["runtime_s"]

    marker_cofactors = cal.marker_cofactors
    marker_thresholds = cal.marker_thresholds

    with _stage_timer() as payload:
        for rec in file_records:
            file_name = rec["sample"].name
            gt_df = ground_truth_df.loc[ground_truth_df["file_name"] == file_name, :].copy()
            assert gt_df.shape[0] != 0, f"No ground truth rows found for {file_name}"

            n_fcs = rec["events"].shape[0]
            n_csv = gt_df.shape[0]
            assert n_fcs == n_csv, f"Event count mismatch for {file_name}: {n_fcs} != {n_csv}"

            metric_rows.append(
                _build_metric_row(
                    experiment=experiment,
                    algorithm=algorithm,
                    algorithm_params=algorithm_params,
                    file_name=file_name,
                    gate_name="edge_exclusion",
                    gate_column=EDGE_GATE_CSV,
                    gt_df=gt_df,
                    comp_vector=rec["mask_edge"],
                )
            )
            metric_rows.append(
                _build_metric_row(
                    experiment=experiment,
                    algorithm=algorithm,
                    algorithm_params=algorithm_params,
                    file_name=file_name,
                    gate_name="singlets",
                    gate_column=SINGLET_GATE_CSV,
                    gt_df=gt_df,
                    comp_vector=rec["mask_sing"],
                )
            )

            lymph_res = gate_lymphocytes(
                lymph_cfg=config.lymphocyte,
                transform_cfg=config.transform,
                random_state=config.random_state,
                panel=PANEL,
                fcs=rec["fcs"],
                events=rec["events"],
                mask_qc=rec["mask_qc"],
                marker_thresholds=marker_thresholds,
                marker_cofactors=marker_cofactors,
            )
            rec["mask_lymph"] = np.asarray(lymph_res.mask_lymph, dtype=bool)

            metric_rows.append(
                _build_metric_row(
                    experiment=experiment,
                    algorithm=algorithm,
                    algorithm_params=algorithm_params,
                    file_name=file_name,
                    gate_name="lymphocytes",
                    gate_column=LYMPHOCYTES_GATE_CSV,
                    gt_df=gt_df,
                    comp_vector=rec["mask_lymph"],
                )
            )
    stage_times["qc_plus_lymph_runtime_s"] = payload["runtime_s"]

    return file_records, metric_rows, stage_times, marker_cofactors, marker_thresholds


def _compute_cluster_metrics(
    *,
    file_records: list[dict[str, Any]],
    ground_truth_df: pd.DataFrame,
    experiment: str,
    config: GatingConfig,
    algorithm: str,
    algorithm_params: dict[str, Any],
    marker_cofactors: dict[str, Any],
    marker_thresholds: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    stage_times: dict[str, float] = {}
    stage_info: dict[str, Any] = {}

    marker_names = list(PANEL.markers.keys())
    rng = np.random.default_rng(int(config.random_state))
    clusterer = default_clusterer_factory(config.clusterer)

    with _stage_timer() as payload:
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
    stage_times["fit_clustering_runtime_s"] = payload["runtime_s"]

    n_train = int(y_train.size)
    n_noise = int(np.sum(y_train == -1))
    frac_noise = float(n_noise / n_train) if n_train > 0 else np.nan
    stage_info.update({"n_train": n_train, "n_noise": n_noise, "frac_noise": frac_noise})

    with _stage_timer() as payload:
        cluster_to_type = label_clusters(
            debris_cfg=config.debris_cluster_label,
            cluster_label_cfg=config.cluster_label,
            X_scatter=X_train_scatter,
            T_markers=T_train,
            y_train=y_train,
            marker_names=marker_names,
            marker_thresholds=marker_thresholds,
        )
    stage_times["label_clusters_runtime_s"] = payload["runtime_s"]
    stage_info["cluster_to_type_json"] = json.dumps(cluster_to_type, sort_keys=True)

    with _stage_timer() as payload:
        for rec in file_records:
            file_name = rec["sample"].name
            mask_lymph = np.asarray(rec["mask_lymph"], dtype=bool)
            if not np.any(mask_lymph):
                continue

            gt_df = ground_truth_df.loc[ground_truth_df["file_name"] == file_name, :].copy()
            assert gt_df.shape[0] != 0, f"No ground truth rows found for {file_name}"

            pred = predict_file_in_mask(
                transform_cfg=config.transform,
                prediction_cfg=config.prediction,
                panel=PANEL,
                fcs=rec["fcs"],
                events=rec["events"],
                mask_in=mask_lymph,
                marker_cofactors=marker_cofactors,
                clusterer=clusterer,
                feature_scaler=feature_scaler,
                outlier_thr=float(outlier_thr),
                cluster_to_type=cluster_to_type,
                marker_names=marker_names,
            )

            metric_rows.append(
                _build_metric_row(
                    experiment=experiment,
                    algorithm=algorithm,
                    algorithm_params=algorithm_params,
                    file_name=file_name,
                    gate_name="t_cells",
                    gate_column=CD3_GATE_CSV,
                    gt_df=gt_df,
                    comp_vector=np.asarray(pred.mask_by_marker["CD3 PerCP-A"], dtype=bool),
                )
            )
            metric_rows.append(
                _build_metric_row(
                    experiment=experiment,
                    algorithm=algorithm,
                    algorithm_params=algorithm_params,
                    file_name=file_name,
                    gate_name="b_cells",
                    gate_column=CD19_GATE_CSV,
                    gt_df=gt_df,
                    comp_vector=np.asarray(pred.mask_by_marker["CD19 APC-Cy7-A"], dtype=bool),
                )
            )
    stage_times["predict_runtime_s"] = payload["runtime_s"]

    return metric_rows, stage_times, stage_info


def run_single_flow_validation(
    data_dir: str,
    experiment: EXPERIMENTS,
    config: GatingConfig,
    algorithm: str,
    algorithm_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    monitor = MemoryMonitor()
    overall_t0 = time.perf_counter()

    ground_truth_df = _read_csv(data_dir, experiment)

    (
        file_records,
        qc_metric_rows,
        qc_stage_times,
        marker_cofactors,
        marker_thresholds,
    ) = _compute_qc_and_lymphocyte_metrics(
        data_dir=data_dir,
        experiment=experiment,
        config=config,
        algorithm=algorithm,
        algorithm_params=algorithm_params,
    )

    cluster_metric_rows, cluster_stage_times, cluster_stage_info = _compute_cluster_metrics(
        file_records=file_records,
        ground_truth_df=ground_truth_df,
        experiment=experiment,
        config=config,
        algorithm=algorithm,
        algorithm_params=algorithm_params,
        marker_cofactors=marker_cofactors,
        marker_thresholds=marker_thresholds,
    )

    metrics_df = pd.DataFrame(qc_metric_rows + cluster_metric_rows)

    run_info: dict[str, Any] = {
        "experiment": experiment,
        "algorithm": algorithm,
        "algorithm_params_json": json.dumps(algorithm_params, sort_keys=True),
        "n_files": int(metrics_df["file_name"].nunique()) if not metrics_df.empty else 0,
        "total_runtime_s": time.perf_counter() - overall_t0,
        **monitor.stop(),
        **qc_stage_times,
        **cluster_stage_times,
        **cluster_stage_info,
    }

    del file_records
    gc.collect()

    return metrics_df, run_info


def run_flow_validation(
    data_dir: str,
    output_dir: Optional[str] = None,
    base_config: Optional[GatingConfig] = None,
    experiments_to_run: Optional[list[EXPERIMENTS]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiments = experiments_to_run or ["exp1", "exp2", "exp3", "exp4"]

    all_metric_dfs: list[pd.DataFrame] = []
    run_infos: list[dict[str, Any]] = []

    for sweep in _iter_sweep_configs(base_config):
        algorithm = sweep["algorithm"]
        algorithm_params = sweep["algorithm_params"]
        config = sweep["config"]

        for experiment in experiments:
            print(
                f"[flow-validation] experiment={experiment} algorithm={algorithm} params={algorithm_params}"
            )
            metrics_df, run_info = run_single_flow_validation(
                data_dir=data_dir,
                experiment=experiment,
                config=config,
                algorithm=algorithm,
                algorithm_params=algorithm_params,
            )
            all_metric_dfs.append(metrics_df)
            run_infos.append(run_info)

    if all_metric_dfs:
        metrics_df = pd.concat(all_metric_dfs, ignore_index=True)
    else:
        metrics_df = pd.DataFrame(
            columns=[
                "experiment",
                "algorithm",
                "algorithm_params_json",
                "file_name",
                "gate",
                "f1",
                "jaccard",
                "tp",
                "tn",
                "fp",
                "fn",
                "n_events",
                "n_pred_positive",
                "n_gt_positive",
            ]
        )

    runs_df = pd.DataFrame(run_infos)

    if metrics_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            metrics_df.groupby(
                ["experiment", "algorithm", "algorithm_params_json", "gate"],
                as_index=False,
            )
            .agg(
                mean_f1=("f1", "mean"),
                std_f1=("f1", "std"),
                mean_jaccard=("jaccard", "mean"),
                std_jaccard=("jaccard", "std"),
                sum_tp=("tp", "sum"),
                sum_tn=("tn", "sum"),
                sum_fp=("fp", "sum"),
                sum_fn=("fn", "sum"),
                n_files=("file_name", "nunique"),
            )
            .merge(
                runs_df,
                on=["experiment", "algorithm", "algorithm_params_json"],
                how="left",
            )
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(output_dir, "flow_validation_metrics_long.csv"), index=False)
        summary_df.to_csv(os.path.join(output_dir, "flow_validation_summary.csv"), index=False)
        runs_df.to_csv(os.path.join(output_dir, "flow_validation_runs.csv"), index=False)

    return metrics_df, summary_df, runs_df

