from pathlib import Path
import io

from fastapi import APIRouter, File, HTTPException, UploadFile

from alloviewer.image_analysis.services.layout_parser import (
    parse_excel_layout,
)
from alloviewer.image_analysis.structs import (
    dc_to_dict,
    parsed_plate_from_dict,
)

from app.config import LAYOUT_EXTENSIONS
from app.services.job_paths import get_job_paths
from app.services.job_registry import (
    read_json,
    require_job_type,
    update_job,
    write_json_atomic,
)


router = APIRouter(
    prefix="/api/jobs/{job_id}/plate-layout",
    tags=["plate-layouts"],
)


@router.post("")
async def parse_layout(job_id: str, xlsx: UploadFile = File(...)):
    try:
        require_job_type(job_id, {"pra"})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    filename = xlsx.filename or "layout.xlsx"
    extension = Path(filename).suffix.lower()

    if extension not in LAYOUT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Please upload an .xlsx or .xlsm file.",
        )

    raw = await xlsx.read()
    paths = get_job_paths(job_id, create=True)

    original_path = paths.layout_uploads / f"original{extension}"
    original_path.write_bytes(raw)

    try:
        sha256, layout = parse_excel_layout(io.BytesIO(raw))
    except Exception as exc:
        original_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Parsing failed: {exc}",
        )

    layout.upload_id = job_id
    layout.sha256 = sha256

    parsed = dc_to_dict(layout)
    write_json_atomic(paths.plate_layout, parsed)

    update_job(
        job_id,
        status="draft",
        layout_filename=original_path.name,
    )

    return parsed


@router.get("")
async def get_layout(job_id: str):
    paths = get_job_paths(job_id)

    if not paths.plate_layout.exists():
        raise HTTPException(
            status_code=404,
            detail="Plate layout not found",
        )

    return read_json(paths.plate_layout)
