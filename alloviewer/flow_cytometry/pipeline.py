from __future__ import annotations

from typing import Any, Dict

from alloviewer.flow_cytometry.sample import Dataset, Sample
from alloviewer.flow_cytometry.panel_utils import build_panel_from_rows
from alloviewer.flow_cytometry.gating import Gater, GatingConfig

from .plots import make_results_payload

def run_fcxm_pipeline(req_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    req_dict is the dict created from FCXMRunRequest.model_dump().
    It contains:
      - "panel_rows": list of dicts
      - "samples": list of dicts with file_paths relative to DATA_DIR

    Returns a JSON-friendly dict that will be stored in FCXM_JOB_RESULTS[job_id].
    """
    # 1) build Dataset (resolves file paths under DATA_DIR)
    samples = []
    for s in req_dict["samples"]:
        samples.append(
            Sample(
                name=s["name"],
                role=s["role"],
                file_paths=s["file_paths"],
            )
        )
    ds = Dataset(samples=samples)

    # 2) build Panel from user-edited panel rows

    panel, marker_to_population = build_panel_from_rows(req_dict["panel_rows"])

    # 3) run analysis
    gater = Gater(panel, GatingConfig())
    fitted = gater.fit(ds)
    results = gater.apply(ds, fitted)

    print("gating finished")

    # 4) Build payload + plot cache
    payload, plot_cache = make_results_payload(
        ds=ds,
        gater=gater,
        fitted=fitted,
        results=results,
        marker_to_population=marker_to_population,
        max_points=5000,
        seed=0,
    )
    print("payload calculated")
    return {
        "payload": payload,
        "plot_cache": plot_cache,
    }
