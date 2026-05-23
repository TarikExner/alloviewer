from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._utils import fcs_display_name, freeze_mapping
from .clusterers import default_clusterer_factory
from .clustering import BaseClusterer, fit_clustering, predict_file_in_mask
from .config import ClustererConfig, GatingConfig
from .igg import (
    IgGControlStats,
    build_igg_control_stats,
    compute_igg_readouts,
    get_igg_values_from_mask,
)
from .labeling import label_clusters
from .lymphocytes import gate_lymphocytes
from .marker_calibration import calibrate_markers
from .qc import QCGater
from .types import (
    FileAnalysis,
    FileResult,
    FittedGater,
    PopulationResult,
    SampleResult,
)
from ..fcs_file import FCSFile
from ..panel import Panel
from ..sample import Dataset


ProgressEvent = Dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]


class Gater:
    """Fit and apply a flow-cytometry gating workflow.

    The gater performs quality-control gating, lymphocyte gating, marker
    calibration, global clustering, cluster labeling, and IgG readout
    calculation.

    Parameters
    ----------
    panel : Panel
        Panel definition with scatter, IgG, and marker channels.
    config : GatingConfig or None, optional
        Gating configuration. If ``None``, a default ``GatingConfig`` is used.
    clusterer_factory : callable or None, optional
        Factory that receives a ``ClustererConfig`` and returns a
        ``BaseClusterer``. If ``None``, ``default_clusterer_factory`` is used.

    Attributes
    ----------
    panel : Panel
        Panel used by the workflow.
    config : GatingConfig
        Gating configuration.
    qc : QCGater
        Quality-control gater.
    clusterer_factory : callable
        Factory used to create the clustering backend.

    Raises
    ------
    ValueError
        If ``panel.markers`` is empty or ``panel.igg`` is not set.
    """

    def __init__(
        self,
        panel: Panel,
        config: Optional[GatingConfig] = None,
        clusterer_factory: Optional[
            Callable[[ClustererConfig], BaseClusterer]
        ] = None,
    ) -> None:
        self.panel = panel
        self.config = config or GatingConfig()
        self.qc = QCGater(panel=self.panel, config=self.config.qc)

        if not self.panel.markers:
            raise ValueError("Panel.markers must not be empty.")
        if not self.panel.igg:
            raise ValueError("Panel.igg must be set.")

        self.clusterer_factory = clusterer_factory or default_clusterer_factory

        self._cache_file_analysis: Dict[Tuple[int, Any, int], FileAnalysis] = {}
        self._cfg_token = freeze_mapping(asdict(self.config))

    def transform_channel(
        self,
        values: np.ndarray,
        cofactor: Optional[float] = None,
    ) -> np.ndarray:
        """Apply arcsinh transformation to channel values.

        Parameters
        ----------
        values : numpy.ndarray
            Raw channel values.
        cofactor : float or None, optional
            Transformation cofactor. If ``None``, the default cofactor from the
            configuration is used.

        Returns
        -------
        numpy.ndarray
            Arcsinh-transformed channel values.
        """
        c = (
            float(self.config.transform.default_cofactor)
            if cofactor is None
            else float(cofactor)
        )
        return np.arcsinh(values / max(c, 1e-12))

    def analyze_file_cached(
        self,
        fcs: FCSFile,
        fitted: FittedGater,
    ) -> FileAnalysis:
        """Analyze a file with caching.

        Parameters
        ----------
        fcs : FCSFile
            FCS file to analyze.
        fitted : FittedGater
            Fitted gater state.

        Returns
        -------
        FileAnalysis
            File-level masks, events, marker assignments, and notes.
        """
        key = (id(fcs), self._cfg_token, id(fitted))
        hit = self._cache_file_analysis.get(key)

        if hit is not None:
            return hit

        fa = self.analyze_file(fcs=fcs, fitted=fitted)
        self._cache_file_analysis[key] = fa

        return fa

    def analyze_file(
        self,
        fcs: FCSFile,
        fitted: FittedGater,
    ) -> FileAnalysis:
        """Analyze one FCS file with a fitted gater.

        Parameters
        ----------
        fcs : FCSFile
            FCS file to analyze.
        fitted : FittedGater
            Fitted gater state.

        Returns
        -------
        FileAnalysis
            Event matrix, QC masks, lymphocyte masks, marker masks, and notes.
        """
        qc = self.qc.compute_qc(fcs)
        events = qc.events
        n = int(events.shape[0])

        notes: List[str] = list(qc.notes or [])

        mask_edge = np.asarray(qc.mask_edge, dtype=bool).copy()
        mask_sing = (
            np.asarray(qc.mask_sing, dtype=bool).copy()
            if qc.mask_sing is not None
            else None
        )
        mask_qc = np.asarray(qc.mask_qc, dtype=bool).copy()

        lr = gate_lymphocytes(
            lymph_cfg=fitted.config.lymphocyte,
            transform_cfg=fitted.config.transform,
            random_state=fitted.config.random_state,
            panel=fitted.panel,
            fcs=fcs,
            events=events,
            mask_qc=mask_qc,
            marker_thresholds=fitted.marker_thresholds,
            marker_cofactors=fitted.marker_cofactors,
        )

        mask_lymph_raw = np.asarray(lr.mask_lymph, dtype=bool).copy()

        if isinstance(lr.info, dict) and lr.info.get("fallback"):
            notes.append(f"Lymph fallback: {lr.info.get('fallback')}")

        marker_names = list((fitted.panel.markers or {}).keys())
        m_by_marker: Dict[str, np.ndarray] = {
            mn: np.zeros(n, dtype=bool) for mn in marker_names
        }
        mask_lymph_clean = mask_lymph_raw.copy()

        if np.any(mask_lymph_raw):
            pred = predict_file_in_mask(
                transform_cfg=fitted.config.transform,
                prediction_cfg=fitted.config.prediction,
                panel=fitted.panel,
                fcs=fcs,
                events=events,
                mask_in=mask_lymph_raw,
                marker_cofactors=fitted.marker_cofactors,
                clusterer=fitted.clusterer,
                feature_scaler=fitted.feature_scaler,
                outlier_thr=float(fitted.outlier_score_threshold),
                cluster_to_type=fitted.cluster_to_type,
                marker_names=marker_names,
            )

            mask_lymph_clean = pred.mask_all_in_lymph
            m_by_marker = pred.mask_by_marker

        return FileAnalysis(
            events=events,
            mask_edge=mask_edge,
            mask_sing=mask_sing,
            mask_qc=mask_qc,
            mask_lymph_raw=mask_lymph_raw,
            mask_lymph=mask_lymph_clean,
            m_by_marker=m_by_marker,
            notes=notes,
        )

    def _get_gate_options(self) -> List[str]:
        """Return gate labels available for the current panel.

        Returns
        -------
        list of str
            Gate labels, including core gates and marker gates.
        """
        gate_options: List[str] = ["All Cells"]

        if self.panel.fsc_a and self.panel.fsc_h:
            gate_options.append("Singlets")

        gate_options.append("Lymphocytes")
        gate_options.extend(list(self.panel.markers.keys()))

        return gate_options

    def _get_gate_masks(
        self,
        *,
        fa: FileAnalysis,
        fitted: FittedGater,
    ) -> Dict[str, np.ndarray]:
        """Build gate masks for one analyzed file.

        Parameters
        ----------
        fa : FileAnalysis
            File analysis object.
        fitted : FittedGater
            Fitted gater state.

        Returns
        -------
        dict
            Mapping from gate label to boolean mask.
        """
        n_events = int(fa.events.shape[0])
        marker_names = list(fitted.panel.markers.keys())

        mask_all = np.asarray(fa.mask_edge, dtype=bool)
        mask_sing = (
            np.asarray(fa.mask_sing, dtype=bool)
            if fa.mask_sing is not None
            else mask_all
        )
        mask_lymph = np.asarray(fa.mask_lymph, dtype=bool)

        m_by_marker = fa.m_by_marker or {
            mn: np.zeros(n_events, dtype=bool) for mn in marker_names
        }

        out: Dict[str, np.ndarray] = {
            "All Cells": mask_all,
            "Singlets": mask_sing,
            "Lymphocytes": mask_lymph,
        }

        for mn in marker_names:
            out[mn] = np.asarray(
                m_by_marker.get(mn, np.zeros(n_events, dtype=bool)),
                dtype=bool,
            )

        return out

    def _emit_progress(
        self,
        progress_cb: Optional[ProgressCallback],
        *,
        stage: str,
        sample: Any = None,
        fcs: Optional[FCSFile] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a progress event.

        Parameters
        ----------
        progress_cb : callable or None
            Callback receiving a progress-event dictionary. If ``None``, no
            event is emitted.
        stage : str
            Processing stage name.
        sample : Any, optional
            Sample object used to populate sample metadata.
        fcs : FCSFile or None, optional
            FCS file used to populate file metadata.
        extra : dict or None, optional
            Extra event fields.
        """
        if progress_cb is None:
            return

        event: Dict[str, Any] = {"stage": stage}

        if sample is not None:
            event["sample_name"] = getattr(sample, "name", None)
            event["role"] = getattr(sample, "role", None)

        if fcs is not None:
            event["file_name"] = fcs_display_name(fcs)

        if extra:
            event.update(extra)

        progress_cb(event)

    def _make_temporary_fitted(
        self,
        *,
        cfg: GatingConfig,
        marker_thresholds: Dict[str, float],
        marker_cofactors: Dict[str, float],
        feature_scaler: Any,
        clusterer: BaseClusterer,
        outlier_thr: float,
        cluster_to_type: Dict[Any, Any],
        gate_options: List[str],
        marker_info: Dict[str, Any],
    ) -> FittedGater:
        """Build an intermediate fitted state during fitting.

        Parameters
        ----------
        cfg : GatingConfig
            Gating configuration.
        marker_thresholds : dict
            Marker thresholds by marker name.
        marker_cofactors : dict
            Marker transformation cofactors by marker name.
        feature_scaler : Any
            Fitted feature scaler.
        clusterer : BaseClusterer
            Fitted clusterer.
        outlier_thr : float
            Outlier score threshold.
        cluster_to_type : dict
            Cluster-to-population mapping.
        gate_options : list of str
            Available gate labels.
        marker_info : dict
            Marker calibration metadata.

        Returns
        -------
        FittedGater
            Temporary fitted state with empty IgG control statistics.
        """
        return FittedGater(
            panel=self.panel,
            config=cfg,
            marker_thresholds=marker_thresholds,
            marker_cofactors=marker_cofactors,
            feature_scaler=feature_scaler,
            clusterer=clusterer,
            outlier_score_threshold=float(outlier_thr),
            cluster_to_type=cluster_to_type,
            gate_options=gate_options,
            igg_cutoff_by_gate={},
            igg_control_stats_by_gate={},
            marker_calibration_info=marker_info,
        )

    def fit(
        self,
        dataset: Dataset,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> FittedGater:
        """Fit the gating workflow on a dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing NC, optional PC, and sample files.
        progress_cb : callable or None, optional
            Callback receiving progress-event dictionaries.

        Returns
        -------
        FittedGater
            Fitted gater state with marker thresholds, marker cofactors,
            clustering model, gate labels, and IgG control statistics.
        """
        cfg = self.config
        rng = np.random.default_rng(int(cfg.random_state))

        file_records: List[Dict[str, Any]] = []

        for sample in dataset.samples:
            for fcs in sample.files:
                qc = self.qc.compute_qc(fcs)

                file_records.append(
                    {
                        "sample": sample,
                        "fcs": fcs,
                        "events": qc.events,
                        "mask_qc": qc.mask_qc,
                        "notes": qc.notes,
                    }
                )

                self._emit_progress(
                    progress_cb,
                    stage="fit_qc",
                    sample=sample,
                    fcs=fcs,
                )

        marker_names = list(self.panel.markers.keys())

        self._emit_progress(progress_cb, stage="fit_marker_calibration")

        cal = calibrate_markers(
            transform_cfg=cfg.transform,
            cofactor_cfg=cfg.marker_cofactor,
            threshold_cfg=cfg.marker_threshold,
            random_state=cfg.random_state,
            panel=self.panel,
            file_records=file_records,
            marker_names=marker_names,
        )

        marker_cofactors = cal.marker_cofactors
        marker_thresholds = cal.marker_thresholds
        marker_info = cal.marker_info

        for rec in file_records:
            sample = rec["sample"]
            fcs: FCSFile = rec["fcs"]

            lr = gate_lymphocytes(
                lymph_cfg=cfg.lymphocyte,
                transform_cfg=cfg.transform,
                random_state=cfg.random_state,
                panel=self.panel,
                fcs=fcs,
                events=rec["events"],
                mask_qc=rec["mask_qc"],
                marker_thresholds=marker_thresholds,
                marker_cofactors=marker_cofactors,
            )

            rec["mask_lymph"] = np.asarray(lr.mask_lymph, dtype=bool)
            rec["lymph_info"] = lr.info

            self._emit_progress(
                progress_cb,
                stage="fit_lymphocytes",
                sample=sample,
                fcs=fcs,
            )

        self._emit_progress(progress_cb, stage="fit_clustering")

        clusterer = self.clusterer_factory(cfg.clusterer)

        (
            feature_scaler,
            clusterer,
            outlier_thr,
            X_train_scatter,
            T_train,
            y_train,
        ) = fit_clustering(
            transform_cfg=cfg.transform,
            feature_scaling_cfg=cfg.feature_scaling,
            cluster_sampling_cfg=cfg.cluster_sampling,
            prediction_cfg=cfg.prediction,
            panel=self.panel,
            file_records=file_records,
            marker_cofactors=marker_cofactors,
            clusterer=clusterer,
            rng=rng,
        )

        self._emit_progress(progress_cb, stage="fit_cluster_labels")

        cluster_to_type = label_clusters(
            debris_cfg=cfg.debris_cluster_label,
            cluster_label_cfg=cfg.cluster_label,
            X_scatter=X_train_scatter,
            T_markers=T_train,
            y_train=y_train,
            marker_names=marker_names,
            marker_thresholds=marker_thresholds,
        )

        gate_options = self._get_gate_options()
        has_pc = bool(dataset.get("PC"))

        nc_raw_parts: Dict[str, List[np.ndarray]] = {g: [] for g in gate_options}
        nc_t_parts: Dict[str, List[np.ndarray]] = {g: [] for g in gate_options}
        pc_raw_parts: Dict[str, List[np.ndarray]] = {g: [] for g in gate_options}
        pc_t_parts: Dict[str, List[np.ndarray]] = {g: [] for g in gate_options}

        temp_fitted = self._make_temporary_fitted(
            cfg=cfg,
            marker_thresholds=marker_thresholds,
            marker_cofactors=marker_cofactors,
            feature_scaler=feature_scaler,
            clusterer=clusterer,
            outlier_thr=float(outlier_thr),
            cluster_to_type=cluster_to_type,
            gate_options=gate_options,
            marker_info=marker_info,
        )

        for rec in file_records:
            sample = rec["sample"]
            fcs: FCSFile = rec["fcs"]

            fa = self.analyze_file(fcs=fcs, fitted=temp_fitted)
            gate_masks = self._get_gate_masks(fa=fa, fitted=temp_fitted)

            for gate in gate_options:
                mask = gate_masks[gate]

                raw_vals, t_vals = get_igg_values_from_mask(
                    transform_cfg=cfg.transform,
                    fcs=fcs,
                    events=fa.events,
                    mask=mask,
                    igg_channel=self.panel.igg,
                )

                if raw_vals.size == 0:
                    continue

                if sample.role == "NC":
                    nc_raw_parts[gate].append(raw_vals)
                    nc_t_parts[gate].append(t_vals)
                elif sample.role == "PC":
                    pc_raw_parts[gate].append(raw_vals)
                    pc_t_parts[gate].append(t_vals)

            self._emit_progress(
                progress_cb,
                stage="fit_control_stats",
                sample=sample,
                fcs=fcs,
            )

        igg_control_stats_by_gate: Dict[str, IgGControlStats] = {}
        igg_cutoff_by_gate: Dict[str, float] = {}

        for gate in gate_options:
            nc_raw = (
                np.concatenate(nc_raw_parts[gate]).astype(np.float32, copy=False)
                if nc_raw_parts[gate]
                else np.array([], dtype=np.float32)
            )
            nc_t = (
                np.concatenate(nc_t_parts[gate]).astype(np.float32, copy=False)
                if nc_t_parts[gate]
                else np.array([], dtype=np.float32)
            )
            pc_raw = (
                np.concatenate(pc_raw_parts[gate]).astype(np.float32, copy=False)
                if has_pc and pc_raw_parts[gate]
                else None
            )
            pc_t = (
                np.concatenate(pc_t_parts[gate]).astype(np.float32, copy=False)
                if has_pc and pc_t_parts[gate]
                else None
            )

            stats = build_igg_control_stats(
                gate=gate,
                igg_cfg=cfg.igg_cutoff,
                transform_cfg=cfg.transform,
                nc_raw=nc_raw,
                nc_t=nc_t,
                pc_raw=pc_raw,
                pc_t=pc_t,
            )

            igg_control_stats_by_gate[gate] = stats
            igg_cutoff_by_gate[gate] = float(stats.cutoff_t)

        fitted = FittedGater(
            panel=self.panel,
            config=cfg,
            marker_thresholds=marker_thresholds,
            marker_cofactors=marker_cofactors,
            feature_scaler=feature_scaler,
            clusterer=clusterer,
            outlier_score_threshold=float(outlier_thr),
            cluster_to_type=cluster_to_type,
            gate_options=gate_options,
            igg_cutoff_by_gate=igg_cutoff_by_gate,
            igg_control_stats_by_gate=igg_control_stats_by_gate,
            marker_calibration_info=marker_info,
        )

        return fitted

    def apply_file(
        self,
        *,
        sample: Any,
        fcs: FCSFile,
        fitted: FittedGater,
    ) -> Tuple[
        FileResult,
        Dict[str, List[np.ndarray]],
        Dict[str, List[np.ndarray]],
        List[str],
    ]:
        """Apply a fitted gater to one FCS file.

        Parameters
        ----------
        sample : Any
            Sample object containing at least ``name`` and ``role`` attributes.
        fcs : FCSFile
            FCS file to analyze.
        fitted : FittedGater
            Fitted gater state.

        Returns
        -------
        file_result : FileResult
            File-level population metrics.
        raw_parts_by_gate : dict
            Raw IgG values by gate, used for sample-level combination.
        t_parts_by_gate : dict
            Transformed IgG values by gate, used for sample-level combination.
        notes : list of str
            File-level notes.
        """
        fa = self.analyze_file_cached(fcs, fitted)
        notes: List[str] = list(fa.notes or [])
        gate_masks = self._get_gate_masks(fa=fa, fitted=fitted)

        gate_options = list(fitted.gate_options)

        raw_parts_by_gate: Dict[str, List[np.ndarray]] = {
            g: [] for g in gate_options
        }
        t_parts_by_gate: Dict[str, List[np.ndarray]] = {
            g: [] for g in gate_options
        }

        pops: List[PopulationResult] = []

        for gate in gate_options:
            mask = gate_masks.get(gate, np.zeros(fa.events.shape[0], dtype=bool))

            raw_vals, t_vals = get_igg_values_from_mask(
                transform_cfg=fitted.config.transform,
                fcs=fcs,
                events=fa.events,
                mask=mask,
                igg_channel=fitted.panel.igg,
            )

            control = fitted.igg_control_stats_by_gate[gate]

            metrics = compute_igg_readouts(
                raw_vals=raw_vals,
                t_vals=t_vals,
                control=control,
            )

            pops.append(
                PopulationResult(
                    label=gate,
                    n_events=int(metrics["n_events"]),
                    igg_pos_fraction=float(metrics["igg_pos_fraction"]),
                    igg_median_raw=float(metrics["igg_median_raw"]),
                    igg_median_t=float(metrics["igg_median_t"]),
                    igg_median_shift=float(metrics["igg_median_shift"]),
                    igg_median_ratio=float(metrics["igg_median_ratio"]),
                    igg_fluorescence_index=float(
                        metrics["igg_fluorescence_index"]
                    ),
                    igg_cutoff_t=float(control.cutoff_t),
                    igg_nc_median_raw=float(control.nc_median_raw),
                    igg_pc_median_raw=(
                        None
                        if control.pc_median_raw is None
                        else float(control.pc_median_raw)
                    ),
                )
            )

            if raw_vals.size:
                raw_parts_by_gate[gate].append(raw_vals)
                t_parts_by_gate[gate].append(t_vals)

        file_name = fcs_display_name(fcs)

        file_result = FileResult(
            file_name=file_name,
            populations=pops,
            notes=notes,
        )

        return file_result, raw_parts_by_gate, t_parts_by_gate, notes

    def combine_sample_files(
        self,
        *,
        sample: Any,
        fitted: FittedGater,
        per_file_results: List[FileResult],
        raw_parts_by_gate: Dict[str, List[np.ndarray]],
        t_parts_by_gate: Dict[str, List[np.ndarray]],
        sample_notes: List[str],
    ) -> SampleResult:
        """Combine per-file results into one sample-level result.

        Parameters
        ----------
        sample : Any
            Sample object containing ``name``, ``role``, and ``files``.
        fitted : FittedGater
            Fitted gater state.
        per_file_results : list of FileResult
            File-level results for the sample.
        raw_parts_by_gate : dict
            Raw IgG values grouped by gate.
        t_parts_by_gate : dict
            Transformed IgG values grouped by gate.
        sample_notes : list of str
            Notes collected across the sample files.

        Returns
        -------
        SampleResult
            Combined sample-level result.
        """
        gate_options = list(fitted.gate_options)

        combined: List[PopulationResult] = []

        for gate in gate_options:
            raw_vals = (
                np.concatenate(raw_parts_by_gate[gate]).astype(
                    np.float32,
                    copy=False,
                )
                if raw_parts_by_gate[gate]
                else np.array([], dtype=np.float32)
            )

            t_vals = (
                np.concatenate(t_parts_by_gate[gate]).astype(
                    np.float32,
                    copy=False,
                )
                if t_parts_by_gate[gate]
                else np.array([], dtype=np.float32)
            )

            control = fitted.igg_control_stats_by_gate[gate]

            metrics = compute_igg_readouts(
                raw_vals=raw_vals,
                t_vals=t_vals,
                control=control,
            )

            combined.append(
                PopulationResult(
                    label=gate,
                    n_events=int(metrics["n_events"]),
                    igg_pos_fraction=float(metrics["igg_pos_fraction"]),
                    igg_median_raw=float(metrics["igg_median_raw"]),
                    igg_median_t=float(metrics["igg_median_t"]),
                    igg_median_shift=float(metrics["igg_median_shift"]),
                    igg_median_ratio=float(metrics["igg_median_ratio"]),
                    igg_fluorescence_index=float(
                        metrics["igg_fluorescence_index"]
                    ),
                    igg_cutoff_t=float(control.cutoff_t),
                    igg_nc_median_raw=float(control.nc_median_raw),
                    igg_pc_median_raw=(
                        None
                        if control.pc_median_raw is None
                        else float(control.pc_median_raw)
                    ),
                )
            )

        return SampleResult(
            sample_name=sample.name,
            role=sample.role,
            n_files=len(sample.files),
            per_file=per_file_results,
            combined=combined,
            notes=sample_notes,
        )

    def apply_sample(
        self,
        *,
        sample: Any,
        fitted: FittedGater,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> SampleResult:
        """Apply a fitted gater to all files of one sample.

        Parameters
        ----------
        sample : Any
            Sample object containing FCS files.
        fitted : FittedGater
            Fitted gater state.
        progress_cb : callable or None, optional
            Callback receiving progress-event dictionaries.

        Returns
        -------
        SampleResult
            Sample-level result with per-file and combined metrics.
        """
        gate_options = list(fitted.gate_options)

        sample_notes: List[str] = []
        per_file_results: List[FileResult] = []

        raw_parts_by_gate: Dict[str, List[np.ndarray]] = {
            g: [] for g in gate_options
        }
        t_parts_by_gate: Dict[str, List[np.ndarray]] = {
            g: [] for g in gate_options
        }

        for fcs in sample.files:
            (
                file_result,
                file_raw_parts,
                file_t_parts,
                notes,
            ) = self.apply_file(
                sample=sample,
                fcs=fcs,
                fitted=fitted,
            )

            per_file_results.append(file_result)
            sample_notes += notes

            for gate in gate_options:
                raw_parts_by_gate[gate].extend(file_raw_parts[gate])
                t_parts_by_gate[gate].extend(file_t_parts[gate])

            self._emit_progress(
                progress_cb,
                stage="apply_file",
                sample=sample,
                fcs=fcs,
                extra={"file_name": file_result.file_name},
            )

        return self.combine_sample_files(
            sample=sample,
            fitted=fitted,
            per_file_results=per_file_results,
            raw_parts_by_gate=raw_parts_by_gate,
            t_parts_by_gate=t_parts_by_gate,
            sample_notes=sample_notes,
        )

    def apply(
        self,
        dataset: Dataset,
        fitted: FittedGater,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[SampleResult]:
        """Apply a fitted gater to a dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to analyze.
        fitted : FittedGater
            Fitted gater state returned by :meth:`fit`.
        progress_cb : callable or None, optional
            Callback receiving progress-event dictionaries.

        Returns
        -------
        list of SampleResult
            One result per sample, preserving dataset order.
        """
        out: List[SampleResult] = []

        for sample in dataset.samples:
            result = self.apply_sample(
                sample=sample,
                fitted=fitted,
                progress_cb=progress_cb,
            )
            out.append(result)

        return out
