# api/routes/fcs_panel.py
from typing import Any, Dict, List, Optional, Literal, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from alloviewer.flow_cytometry.panel_utils import (
    guess_role,
    get_panel_rows_cached,
    is_time_channel,
    guess_population_name,
)

from ...core.settings import settings
from ...core.paths import resolve_under_base_dir

router = APIRouter(tags=["flow-cytometry"])

ChannelRole = Literal["Scatter", "Population Marker", "IgG Marker"]

class FcsPanelRequest(BaseModel):
    fcs_filenames: List[str]

class PanelRowModel(BaseModel):
    channel: str
    role: ChannelRole
    antibody: str
    population: str

class FcsPanelResponse(BaseModel):
    panel_name: Optional[str] = None
    rows: List[PanelRowModel]
    files_seen: int
    example_file: Optional[str] = None
    warning: Optional[Dict[str, Any]] = None

@router.post("/api/fcs/panel", response_model=FcsPanelResponse)
async def extract_fcs_panel(req: FcsPanelRequest) -> Dict[str, Any]:
    if not req.fcs_filenames:
        return {
            "panel_name": None,
            "rows": [],
            "files_seen": 0,
            "example_file": None,
            "warning": None,
        }

    per_file: List[Dict[str, Any]] = []
    for rel in req.fcs_filenames:
        abs_path = resolve_under_base_dir(settings.data_dir, rel)

        if abs_path.suffix.lower() != ".fcs":
            raise HTTPException(status_code=415, detail=f"Not an .fcs file: {rel}")
        if not abs_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"File not found under DATA_DIR: '{rel}' (resolved: '{abs_path}')",
            )

        try:
            sig, rows = get_panel_rows_cached(abs_path)  # sig: ((PnN,PnS),...) in order
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read panel from '{rel}': {e}")

        pnn_order = [pnn for (pnn, _pns) in sig]
        pnn_set = set(pnn_order)

        per_file.append({
            "rel": rel,
            "sig": sig,
            "rows": rows,
            "pnn_order": pnn_order,
            "pnn_set": pnn_set,
        })

    example_file = per_file[0]["rel"]

    union_set = set().union(*[x["pnn_set"] for x in per_file])
    inter_set = set(per_file[0]["pnn_set"])
    for x in per_file[1:]:
        inter_set &= x["pnn_set"]

    union_set = {ch for ch in union_set if not is_time_channel(ch)}
    inter_set = {ch for ch in inter_set if not is_time_channel(ch)}

    ref_sig: Tuple[Tuple[str, str], ...] = per_file[0]["sig"]
    inter_in_ref_order = [
        (pnn, pns)
        for (pnn, pns) in ref_sig
        if (pnn in inter_set) and (not is_time_channel(pnn))
    ]

    rows_out: List[PanelRowModel] = []
    for (pnn, pns) in inter_in_ref_order:
        role = guess_role(pnn, pns)
        antibody = pnn if role == "Scatter" else str(pns or "").strip()
        population = guess_population_name(pnn, pns, role)

        rows_out.append(
            PanelRowModel(
                channel=pnn,
                role=role,
                antibody=antibody,
                population=population,
            )
        )

    warning: Optional[Dict[str, Any]] = None
    if len(inter_set) != len(union_set):
        dropped = sorted(list(union_set - inter_set))

        file_details = []
        for x in per_file:
            file_set = {ch for ch in x["pnn_set"] if not is_time_channel(ch)}
            missing = sorted(list(union_set - file_set))
            extras = sorted(list(file_set - inter_set))
            file_details.append({
                "file": x["rel"],
                "missing_channels": missing,
                "extra_channels": extras,
                "n_channels": len(file_set),
            })

        warning = {
            "type": "PANEL_MISMATCH",
            "message": (
                "Not all files share the exact same panel. "
                "We will show only channels that are present in every file."
            ),
            "files_seen": len(per_file),
            "common_channels_count": len(inter_set),
            "dropped_channels": dropped,
            "files": file_details,
            "example_file": example_file,
        }

    return {
        "panel_name": None,
        "rows": [r.model_dump() for r in rows_out],
        "files_seen": len(req.fcs_filenames),
        "example_file": example_file,
        "warning": warning,
    }

