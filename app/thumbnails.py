from pathlib import Path
import posixpath
from typing import List

import numpy as np
import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from alloviewer.image_analysis.io import load_image

from .config import MAX_FILE_SIZE_MB, MAX_THUMB_SIZE
from .core.settings import settings

router = APIRouter(tags=["upload"])

DATA_DIR = settings.data_dir.resolve()
THUMB_ROOT = DATA_DIR / "_thumbs"
THUMB_ROOT.mkdir(parents=True, exist_ok=True)


def _secure_join(base: Path, rel: str) -> Path:
    rel = rel.replace("\\", "/")
    full = (base / rel).resolve()
    if not str(full).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Bad path")
    return full


def _thumb_path_for(rel: str) -> Path:
    rel_path = Path(rel.replace("\\", "/"))
    return (THUMB_ROOT / rel_path).with_suffix(".png").resolve()


@router.get("/api/thumbnails/{rel_path:path}")
async def get_thumbnail(rel_path: str):
    src = _secure_join(DATA_DIR, rel_path)

    print("THUMB DATA_DIR =", DATA_DIR)
    print("THUMB rel_path =", rel_path)
    print("THUMB src =", src)
    print("THUMB exists =", src.exists())

    if not src.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    thumb_path = _thumb_path_for(rel_path)
    if thumb_path.exists():
        return FileResponse(thumb_path, media_type="image/png")

    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    img, _report = load_image(
        rel_path,
        base_dir=DATA_DIR,
        page=0,
        max_mp=200.0,
        as_chw=False,
        scale=True,
        fast_scale=True,
    )

    img8 = np.clip(img * 255.0, 0, 255).astype("uint8")

    h, w, _ = img8.shape
    scale = min(MAX_THUMB_SIZE / h, MAX_THUMB_SIZE / w, 1.0)
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img8 = cv2.resize(img8, new_size, interpolation=cv2.INTER_AREA)

    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Thumbnail encode failed")

    data = buf.tobytes()
    thumb_path.write_bytes(data)

    return Response(content=data, media_type="image/png")
