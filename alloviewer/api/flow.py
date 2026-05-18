# alloviewer/api/flow.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from alloviewer.flow_cytometry.panel import Panel
from alloviewer.flow_cytometry.metadata import Metadata
from alloviewer.flow_cytometry.sample import Dataset, Sample
from alloviewer.flow_cytometry.gating import Gater, GatingConfig
from alloviewer.flow_cytometry.gating.types import FittedGater, SampleResult
from alloviewer.flow_cytometry.plots import (
    make_results_payload,
    build_results_response_from_cache,
)
from alloviewer.flow_cytometry.pdf_report import (
    build_fcxm_summary_pdf,
    ReportMeta,
    ScoreRule,
    RatioRule,
    PAGE_SIZE_MODE_DEFAULT,
)

from .utils import (
    full_path,
    assert_dfs_equal,
    is_scatter_channel,
    normalize_scatter_channel,
)
"""
Short tutorial: public flow cytometry API

This API is meant for users who want to run the FCXM flow cytometry pipeline
from Python without using the web app.

Basic workflow
--------------

1. Read metadata

The metadata file should contain at least:

    file_name, role

Optional but recommended:

    sample_name

Allowed roles are:

    NC      negative control
    PC      positive control
    SAMPLE  patient/sample file

Example metadata.csv:

    file_name,role,sample_name
    NC_001.fcs,NC,Negative Control
    PC_001.fcs,PC,Positive Control
    Patient_001.fcs,SAMPLE,Patient 001
    Patient_002.fcs,SAMPLE,Patient 001


2. Create a Dataset

    import alloviewer.api.flow as avf

    metadata = avf.read_metadata("./metadata.csv")
    dataset = avf.create_dataset("./fcs_files", metadata)


3. Create or infer a Panel

Option A: create the panel manually.

    panel = avf.create_panel({
        "fsc_a": "FSC-A",
        "fsc_h": "FSC-H",
        "ssc_a": "SSC-A",
        "igg": "IgG-A",
        "markers": {
            "CD3": "CD3-FITC-A",
            "CD19": "CD19-PE-A",
        },
    })

The keys inside "markers" are the population marker names used internally.
The values are the matching FCS channel names.

Option B: infer the panel from the dataset.

    panel = avf.infer_panel_from_dataset(dataset)

Panel inference is useful as a starting point, but it may not correctly detect
the IgG channel or population markers in every experiment. For production use,
check the inferred panel before running the analysis.


4. Run the analysis

    run = avf.process_dataset(dataset, panel)

The returned object contains all important results:

    run.results       sample-level and file-level metrics
    run.payload       JSON-friendly frontend/report payload
    run.plot_cache    cached plot data
    run.fitted        fitted gating model
    run.gater         Gater instance
    run.panel         panel used for the run
    run.dataset       dataset used for the run


5. Save a PDF report

    avf.save_output_pdf(run, "./fcxm_summary.pdf")

Or get the PDF as bytes:

    pdf_bytes = avf.create_output_pdf(run)


6. Access result data for one file

    view = avf.get_file_result_view(
        run,
        file_name="Patient_001.fcs",
        gate="Lymphocytes",
    )

The returned view contains:

    view["gate_options"]
    view["selected_gate"]
    view["gating_plots"]
    view["final_scatter_series"]
    view["line_series"]
    view["cutoff"]
    view["selected_file_metrics"]
    view["selected_sample_metrics"]


7. Get gating strategy data

    gating = avf.gating_strategy(
        run,
        file_name="Patient_001.fcs",
    )

This returns plot-ready data for the gating steps. It does not create a
matplotlib figure yet.


8. Get IgG histogram / line plot data

    hist = avf.histogram(
        run,
        file_name="Patient_001.fcs",
        gate="T cells",
    )

This returns line-series data, cutoff, and selected metrics.


9. Get IgG scatter data

    scatter = avf.igg_scatter(
        run,
        file_name="Patient_001.fcs",
        gate="T cells",
    )

This returns scatter-series data and the cutoff for the selected gate.


Complete minimal example
------------------------

    import alloviewer.api.flow as avf

    metadata = avf.read_metadata("./metadata.csv")
    dataset = avf.create_dataset("./fcs_files", metadata)

    panel = avf.create_panel({
        "fsc_a": "FSC-A",
        "fsc_h": "FSC-H",
        "ssc_a": "SSC-A",
        "igg": "IgG-A",
        "markers": {
            "CD3": "CD3-FITC-A",
            "CD19": "CD19-PE-A",
        },
    })

    run = avf.process_dataset(dataset, panel)

    avf.save_output_pdf(run, "./fcxm_summary.pdf")

    lymph_view = avf.get_file_result_view(
        run,
        file_name="Patient_001.fcs",
        gate="Lymphocytes",
    )

    print(lymph_view["selected_file_metrics"])


Notes
-----

- The public entry point should be:

      import alloviewer.api.flow as avf

- Prefer returning the full FCXMRun object from process_dataset().
  Do not return only raw SampleResult objects. The full run object keeps
  the fitted model, payload, and plot cache available for reports and plots.

- If file names are ambiguous, use the full file path when calling
  get_file_result_view(), gating_strategy(), histogram(), or igg_scatter().

- The plotting helper functions currently return plot-ready data.
  Actual matplotlib or interactive plotting can be added later without
  changing the main analysis API.
"""

PathLike = Union[str, Path]


@dataclass
class FCXMRun:
    dataset: Dataset
    panel: Panel
    gater: Gater
    fitted: FittedGater
    results: List[SampleResult]
    payload: Dict[str, Any]
    plot_cache: Dict[str, Any]
    marker_to_population: Dict[str, str]


def read_metadata(input_file: PathLike) -> Metadata:
    return Metadata(input_file)


def create_metadata(df: pd.DataFrame) -> Metadata:
    return Metadata.from_df(df)


def create_dataset(
    file_input_dir: PathLike,
    metadata: Metadata,
) -> Dataset:
    df = metadata.get_df()

    required = {"file_name", "role"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    samples: List[Sample] = []

    for _, row in df.iterrows():
        file_name = str(row["file_name"])
        role = str(row["role"]).strip().upper()

        if role not in {"NC", "PC", "SAMPLE"}:
            raise ValueError(
                f"Invalid role for {file_name}: {role}. "
                "Allowed roles are NC, PC, SAMPLE."
            )

        sample_name = (
            str(row["sample_name"]).strip()
            if "sample_name" in df.columns and pd.notna(row["sample_name"])
            else Path(file_name).stem
        )

        samples.append(
            Sample(
                name=sample_name,
                role=role,
                file_paths=[full_path(file_input_dir, file_name)],
            )
        )

    return Dataset(samples=samples)


def create_panel(markers: Dict[str, Any]) -> Panel:
    return Panel.from_dict(markers)


def infer_panel_from_dataset(ds: Dataset) -> Panel:
    channel_frames = []

    for sample in ds.samples:
        if not sample.files:
            continue

        fcs = sample.files[0]

        channels = getattr(fcs, "channels", None)
        if channels is None:
            channels = getattr(fcs, "channel_table", None)

        if channels is None:
            raise ValueError(
                "Could not infer panel. FCSFile does not expose a channel table."
            )

        channel_frames.append(channels)

    if not channel_frames:
        raise ValueError("Dataset contains no FCS files.")

    try:
        assert_dfs_equal(channel_frames)
    except AssertionError:
        raise ValueError("Panels in the dataset are not the same.")

    channels = channel_frames[0]

    marker_dict: Dict[str, Any] = {"markers": {}}

    internal_scatter = {
        "FSC-A": "fsc_a",
        "FSC-H": "fsc_h",
        "SSC-A": "ssc_a",
    }

    for pnn, pns in zip(channels.index, channels["pns"]):
        pnn = str(pnn).strip()
        pns = "" if pd.isna(pns) else str(pns).strip()

        if is_scatter_channel(pnn):
            normalized = normalize_scatter_channel(pnn)
            key = internal_scatter.get(normalized)
            if key:
                marker_dict[key] = pnn
            continue

        if pns:
            if pns.lower() in {"igg", "anti-human igg", "human igg"}:
                marker_dict["igg"] = pnn
            else:
                marker_dict["markers"][pns] = pnn

    return Panel.from_dict(marker_dict)


def process_dataset(
    ds: Dataset,
    panel: Panel,
    *,
    config: Optional[GatingConfig] = None,
    marker_to_population: Optional[Dict[str, str]] = None,
    max_points: int = 5000,
    seed: int = 0,
) -> FCXMRun:
    gater = Gater(panel, config or GatingConfig())

    fitted = gater.fit(ds)
    results = gater.apply(ds, fitted)

    marker_to_population = marker_to_population or {
        marker_name: marker_name
        for marker_name in (panel.markers or {}).keys()
    }

    payload, plot_cache = make_results_payload(
        ds=ds,
        gater=gater,
        fitted=fitted,
        results=results,
        marker_to_population=marker_to_population,
        max_points=max_points,
        seed=seed,
    )

    return FCXMRun(
        dataset=ds,
        panel=panel,
        gater=gater,
        fitted=fitted,
        results=results,
        payload=payload,
        plot_cache=plot_cache,
        marker_to_population=marker_to_population,
    )


def create_output_pdf(
    run: FCXMRun,
    *,
    meta: Optional[ReportMeta] = None,
    score_rules: Optional[Iterable[ScoreRule]] = None,
    ratio_score_rules: Optional[Iterable[RatioRule]] = None,
    page_size_mode: str = PAGE_SIZE_MODE_DEFAULT,
) -> bytes:
    return build_fcxm_summary_pdf(
        payload=run.payload,
        plot_cache=run.plot_cache,
        meta=meta,
        score_rules=score_rules,
        ratio_score_rules=ratio_score_rules,
        page_size_mode=page_size_mode,
    )


def save_output_pdf(
    run: FCXMRun,
    output_file: PathLike,
    *,
    meta: Optional[ReportMeta] = None,
    score_rules: Optional[Iterable[ScoreRule]] = None,
    ratio_score_rules: Optional[Iterable[RatioRule]] = None,
    page_size_mode: str = PAGE_SIZE_MODE_DEFAULT,
) -> Path:
    output_path = Path(output_file)

    pdf_bytes = create_output_pdf(
        run,
        meta=meta,
        score_rules=score_rules,
        ratio_score_rules=ratio_score_rules,
        page_size_mode=page_size_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_bytes)

    return output_path


def get_payload(run: FCXMRun) -> Dict[str, Any]:
    return run.payload


def get_plot_cache(run: FCXMRun) -> Dict[str, Any]:
    return run.plot_cache


def get_sample_results(run: FCXMRun) -> List[SampleResult]:
    return run.results


def _resolve_plot_cache_key(run: FCXMRun, file_name: PathLike) -> str:
    text = str(file_name)

    if text in run.plot_cache:
        return text

    for key, entry in run.plot_cache.items():
        if str(entry.get("file_key_raw")) == text:
            return key

    target_name = Path(text).name

    matches = [
        key
        for key, entry in run.plot_cache.items()
        if Path(str(entry.get("file_key_raw", ""))).name == target_name
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"File name is ambiguous: {file_name}. Use the full path instead."
        )

    raise KeyError(f"File not found in plot cache: {file_name}")


def get_file_result_view(
    run: FCXMRun,
    file_name: PathLike,
    gate: str = "",
) -> Dict[str, Any]:
    key = _resolve_plot_cache_key(run, file_name)

    return build_results_response_from_cache(
        plot_cache=run.plot_cache,
        selected_key=key,
        selected_gate=gate,
    )


def gating_strategy(
    run: FCXMRun,
    file_name: PathLike,
    gate: str = "",
) -> Dict[str, Any]:
    view = get_file_result_view(run, file_name=file_name, gate=gate)

    return {
        "gate_options": view["gate_options"],
        "selected_gate": view["selected_gate"],
        "gating_plots": view["gating_plots"],
    }


def histogram(
    run: FCXMRun,
    file_name: PathLike,
    gate: str = "",
) -> Dict[str, Any]:
    view = get_file_result_view(run, file_name=file_name, gate=gate)

    return {
        "selected_gate": view["selected_gate"],
        "line_series": view["line_series"],
        "cutoff": view["cutoff"],
        "selected_file_metrics": view["selected_file_metrics"],
        "selected_sample_metrics": view["selected_sample_metrics"],
    }


def igg_scatter(
    run: FCXMRun,
    file_name: PathLike,
    gate: str = "",
) -> Dict[str, Any]:
    view = get_file_result_view(run, file_name=file_name, gate=gate)

    return {
        "selected_gate": view["selected_gate"],
        "final_scatter_series": view["final_scatter_series"],
        "cutoff": view["cutoff"],
    }
