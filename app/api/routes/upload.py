# api/routes/upload.py
from pathlib import Path
import posixpath
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from ...config import ALLOWED_EXT, ALLOWED_MIME, MAX_FILE_SIZE_MB
from ...core.settings import settings

router = APIRouter(tags=["upload"])

@router.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Accepts single files or folder uploads (webkitdirectory).
    Creates subfolders under DATA_DIR to match relative paths.
    """
    saved = []
    base_dir = settings.data_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        raw_name = f.filename or ""
        rel_path = raw_name.replace("\\", "/")
        rel_path = posixpath.normpath(rel_path)
        if rel_path.startswith("../") or rel_path.startswith("/"):
            rel_path = posixpath.basename(rel_path)

        ctype = f.content_type or "application/octet-stream"
        ext = Path(rel_path).suffix.lower()

        if ctype not in ALLOWED_MIME and ext not in ALLOWED_EXT:
            raise HTTPException(status_code=415, detail=f"Type not allowed: {ctype} ({ext})")

        dest = (base_dir / rel_path).resolve()
        if not str(dest).startswith(str(base_dir)):
            raise HTTPException(status_code=400, detail="Bad path")

        dest.parent.mkdir(parents=True, exist_ok=True)

        size_mb = 0.0
        with dest.open("wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                out.write(chunk)
                size_mb += len(chunk) / (1024 * 1024)

                # stop early if already too large
                if size_mb > MAX_FILE_SIZE_MB:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File too large: {raw_name}")

        saved.append({
            "filename": str(dest.relative_to(base_dir)).replace("\\", "/"),
            "size_mb": round(size_mb, 2),
        })

    return {"saved": saved}

