from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from alloviewer.flow_cytometry.panel_utils import (
    get_panel_rows_cached,
    guess_population_name,
    guess_role,
    is_time_channel,
)

from app.services.job_paths import (
    JobPathError,
    get_job_paths,
    resolve_job_path,
)
from app.services.job_registry import require_job_type


router = APIRouter(
    prefix="/api/jobs/{job_id}/fcxm",
    tags=["flow-cytometry"],
)


ChannelRole = Literal[
    "Scatter",
    "Population Marker",
    "IgG Marker",
]


class FcsPanelRequest(BaseModel):
    fcs_filenames: list[str] = Field(
        default_factory=list
    )


class PanelRowModel(BaseModel):
    channel: str
    role: ChannelRole
    antibody: str
    population: str


class FcsPanelResponse(BaseModel):
    panel_name: str | None = None
    rows: list[PanelRowModel] = Field(
        default_factory=list
    )
    files_seen: int
    example_file: str | None = None
    warning: dict[str, Any] | None = None


def _require_fcxm_job(
    job_id: str,
) -> dict[str, Any]:
    try:
        return require_job_type(
            job_id,
            {"fcxm"},
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )


def _resolve_fcs_file(
    *,
    job_id: str,
    filename: str,
):
    paths = get_job_paths(job_id)

    try:
        path = resolve_job_path(
            job_id,
            filename,
            required_root=paths.fcs_uploads,
            must_exist=True,
        )
    except JobPathError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid FCS path '{filename}': "
                f"{exc}"
            ),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=(
                "FCS file was not found in this "
                f"job: {filename}"
            ),
        )

    if not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                "FCS path is not a file: "
                f"{filename}"
            ),
        )

    if path.suffix.lower() != ".fcs":
        raise HTTPException(
            status_code=415,
            detail=(
                "File is not an .fcs file: "
                f"{filename}"
            ),
        )

    return path


@router.post(
    "/panel",
    response_model=FcsPanelResponse,
)
async def extract_fcs_panel(
    job_id: str,
    req: FcsPanelRequest,
) -> FcsPanelResponse:
    _require_fcxm_job(job_id)

    if not req.fcs_filenames:
        return FcsPanelResponse(
            panel_name=None,
            rows=[],
            files_seen=0,
            example_file=None,
            warning=None,
        )

    per_file: list[dict[str, Any]] = []

    for filename in req.fcs_filenames:
        path = _resolve_fcs_file(
            job_id=job_id,
            filename=filename,
        )

        try:
            signature, rows = get_panel_rows_cached(
                path
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read panel from "
                    f"'{filename}': {exc}"
                ),
            )

        pnn_order = [
            pnn
            for pnn, _pns in signature
        ]

        per_file.append(
            {
                "filename": filename,
                "signature": signature,
                "rows": rows,
                "pnn_order": pnn_order,
                "pnn_set": set(pnn_order),
            }
        )

    example_file = str(
        per_file[0]["filename"]
    )

    union_set: set[str] = set()

    for file_data in per_file:
        union_set.update(
            file_data["pnn_set"]
        )

    intersection_set = set(
        per_file[0]["pnn_set"]
    )

    for file_data in per_file[1:]:
        intersection_set.intersection_update(
            file_data["pnn_set"]
        )

    union_set = {
        channel
        for channel in union_set
        if not is_time_channel(channel)
    }

    intersection_set = {
        channel
        for channel in intersection_set
        if not is_time_channel(channel)
    }

    reference_signature: tuple[
        tuple[str, str],
        ...,
    ] = per_file[0]["signature"]

    common_channels_in_reference_order = [
        (pnn, pns)
        for pnn, pns in reference_signature
        if (
            pnn in intersection_set
            and not is_time_channel(pnn)
        )
    ]

    rows_out: list[PanelRowModel] = []

    for pnn, pns in (
        common_channels_in_reference_order
    ):
        role = guess_role(
            pnn,
            pns,
        )

        antibody = (
            pnn
            if role == "Scatter"
            else str(pns or "").strip()
        )

        population = guess_population_name(
            pnn,
            pns,
            role,
        )

        rows_out.append(
            PanelRowModel(
                channel=pnn,
                role=role,
                antibody=antibody,
                population=population,
            )
        )

    warning: dict[str, Any] | None = None

    if (
        len(intersection_set)
        != len(union_set)
    ):
        dropped_channels = sorted(
            union_set
            - intersection_set
        )

        file_details: list[
            dict[str, Any]
        ] = []

        for file_data in per_file:
            file_channels = {
                channel
                for channel
                in file_data["pnn_set"]
                if not is_time_channel(
                    channel
                )
            }

            file_details.append(
                {
                    "file": (
                        file_data[
                            "filename"
                        ]
                    ),
                    "missing_channels": sorted(
                        union_set
                        - file_channels
                    ),
                    "extra_channels": sorted(
                        file_channels
                        - intersection_set
                    ),
                    "n_channels": len(
                        file_channels
                    ),
                }
            )

        warning = {
            "type": "PANEL_MISMATCH",
            "message": (
                "Not all files share the exact "
                "same panel. Only channels "
                "present in every file are shown."
            ),
            "files_seen": len(
                per_file
            ),
            "common_channels_count": len(
                intersection_set
            ),
            "dropped_channels": (
                dropped_channels
            ),
            "files": file_details,
            "example_file": example_file,
        }

    return FcsPanelResponse(
        panel_name=None,
        rows=rows_out,
        files_seen=len(
            req.fcs_filenames
        ),
        example_file=example_file,
        warning=warning,
    )
