import os
from pathlib import Path

import numpy as np
import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from alloviewer.image_analysis.io import load_image

from .config import MAX_THUMB_SIZE

router = APIRouter()

# same base dir logic as your FastAPI DATA_DIR

DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")).resolve()
THUMB_ROOT = DATA_DIR / "_thumbs"
os.makedirs(THUMB_ROOT, exist_ok=True)

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
    """
    rel_path is exactly the 'filename' returned by /api/upload,
    e.g. 'plate1/A1.tif'.
    """
    src = _secure_join(DATA_DIR, rel_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    thumb_path = _thumb_path_for(rel_path)
    if thumb_path.exists():
        # already cached
        return FileResponse(thumb_path, media_type="image/png")

    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    # use your loader: HxWx3, float32 in [0,1]
    img, _report = load_image(
        rel_path,
        base_dir=DATA_DIR,
        page=0,
        max_mp=200.0,       # your default
        as_chw=False,       # we want HxWx3
        scale=True,         # use your bit-depth logic, gives [0,1]
        fast_scale=True,
    )

    # convert to uint8
    img8 = np.clip(img * 255.0, 0, 255).astype("uint8")

    # downscale so the largest side is MAX_THUMB_SIZE
    h, w, _ = img8.shape
    scale = min(MAX_THUMB_SIZE / h, MAX_THUMB_SIZE / w, 1.0)
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        # cv2 expects width, height
        img8 = cv2.resize(img8, new_size, interpolation=cv2.INTER_AREA)

    # encode PNG (cv2 expects BGR)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Thumbnail encode failed")

    data = buf.tobytes()
    thumb_path.write_bytes(data)  # cache for next time

    return Response(content=data, media_type="image/png")

