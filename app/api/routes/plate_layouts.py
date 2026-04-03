# api/routes/plate_layouts.py
import io
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette import status

from alloviewer.image_analysis.services.layout_parser import parse_excel_layout, _sha256_bytes
from alloviewer.image_analysis.structs import dc_to_dict
from alloviewer.image_analysis.storage.repo import LayoutRepo

from ...core.deps import get_repo

router = APIRouter(prefix="/api/plate-layouts", tags=["plate-layouts"])

@router.post("/parse", status_code=status.HTTP_200_OK)
async def parse_layout(
    xlsx: UploadFile = File(...),
    repo: LayoutRepo = Depends(get_repo),
) -> Dict[str, Any]:
    if not (xlsx.filename or "").lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="Please upload an .xlsx or .xlsm file.")

    raw = await xlsx.read()
    sha256 = _sha256_bytes(raw)

    cached = repo.find_by_sha(sha256)
    if cached:
        return dc_to_dict(cached)

    try:
        sha, layout = parse_excel_layout(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parsing failed: {e}")

    layout.sha256 = sha
    repo.save_layout(layout)
    saved = repo.get_by_id(layout.upload_id)
    return dc_to_dict(saved)

@router.get("/{upload_id}")
async def get_layout(
    upload_id: str,
    repo: LayoutRepo = Depends(get_repo),
) -> Dict[str, Any]:
    layout = repo.get_by_id(upload_id)
    if not layout:
        raise HTTPException(status_code=404, detail="upload_id not found")
    return dc_to_dict(layout)

