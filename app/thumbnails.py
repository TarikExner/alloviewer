from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import (
    IMAGE_EXTENSIONS,
    MAX_THUMB_SIZE,
)
from app.services.job_paths import (
    JobPathError,
    get_job_paths,
    resolve_job_path,
)
from app.services.job_registry import require_job_type


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/jobs/{job_id}/thumbnails",
    tags=["thumbnails"],
)

ImageSource = Literal["opencv", "tifffile"]


def _require_image_job(
    job_id: str,
) -> dict[str, Any]:
    try:
        return require_job_type(
            job_id,
            {"pra", "crossmatch"},
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


def _resolve_source_image(
    *,
    job_id: str,
    image_path: str,
) -> Path:
    paths = get_job_paths(job_id)

    try:
        source = resolve_job_path(
            job_id,
            image_path,
            required_root=paths.image_uploads,
            must_exist=True,
        )
    except JobPathError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image path '{image_path}': {exc}"
            ),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded image was not found in this job: "
                f"{image_path}"
            ),
        )

    if not source.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image path is not a file: {image_path}",
        )

    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {image_path}",
        )

    return source


def _read_image(
    path: Path,
) -> tuple[np.ndarray, ImageSource]:
    """
    Read an image using OpenCV first and tifffile as a TIFF fallback.
    """
    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is not None:
        return image, "opencv"

    try:
        import tifffile

        image = np.asarray(
            tifffile.imread(path)
        )
    except Exception as exc:
        raise ValueError(
            f"Could not decode image: {path.name}"
        ) from exc

    if image.size == 0:
        raise ValueError(
            f"Decoded image is empty: {path.name}"
        )

    return image, "tifffile"


def _prepare_dimensions(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert common TIFF layouts to H×W or H×W×C.
    """
    array = np.asarray(image)
    array = np.squeeze(array)

    while array.ndim > 3:
        array = array[0]

    if array.ndim == 3:
        # Convert C×H×W to H×W×C when the channel axis is first.
        if (
            array.shape[0] in {1, 3, 4}
            and array.shape[-1] not in {1, 3, 4}
        ):
            array = np.moveaxis(
                array,
                0,
                -1,
            )

        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[-1] > 4:
            array = array[..., :3]

    if array.ndim not in {2, 3}:
        raise ValueError(
            f"Unsupported image shape: {array.shape}"
        )

    return array


def _to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert integer or floating-point microscopy data to displayable uint8.
    """
    array = np.asarray(image)

    if array.dtype == np.uint8:
        return array

    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255

    array = array.astype(
        np.float32,
        copy=False,
    )

    finite = np.isfinite(array)

    if not finite.any():
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    finite_values = array[finite]

    lower = float(
        np.percentile(
            finite_values,
            0.5,
        )
    )
    upper = float(
        np.percentile(
            finite_values,
            99.5,
        )
    )

    if not np.isfinite(lower):
        lower = float(
            np.nanmin(finite_values)
        )

    if not np.isfinite(upper):
        upper = float(
            np.nanmax(finite_values)
        )

    if upper <= lower:
        lower = float(
            np.nanmin(finite_values)
        )
        upper = float(
            np.nanmax(finite_values)
        )

    if upper <= lower:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    array = (
        (array - lower)
        / (upper - lower)
        * 255.0
    )

    array[~finite] = 0

    return np.clip(
        array,
        0,
        255,
    ).astype(np.uint8)


def _to_opencv_color(
    image: np.ndarray,
    source: ImageSource,
) -> np.ndarray:
    """
    Convert color images to the BGR representation expected by OpenCV.
    """
    if image.ndim == 2:
        return image

    channels = image.shape[-1]

    if channels == 1:
        return image[..., 0]

    if source == "tifffile":
        if channels == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        if channels == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )

    if channels == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR,
        )

    if channels == 3:
        return image

    return image[..., :3]


def _resize_thumbnail(
    image: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image has invalid dimensions."
        )

    longest_side = max(
        height,
        width,
    )

    if longest_side <= MAX_THUMB_SIZE:
        return image

    scale = (
        float(MAX_THUMB_SIZE)
        / float(longest_side)
    )

    target_width = max(
        1,
        int(round(width * scale)),
    )
    target_height = max(
        1,
        int(round(height * scale)),
    )

    return cv2.resize(
        image,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def _thumbnail_path(
    *,
    job_id: str,
    source: Path,
) -> Path:
    paths = get_job_paths(
        job_id,
        create=True,
    )

    try:
        relative_source = source.relative_to(
            paths.image_uploads.resolve()
        )
    except ValueError as exc:
        raise RuntimeError(
            "Resolved source image is outside the job image directory."
        ) from exc

    destination = (
        paths.thumbnails
        / relative_source
    ).with_suffix(".jpg")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return destination


def _thumbnail_is_current(
    *,
    source: Path,
    thumbnail: Path,
) -> bool:
    if not thumbnail.exists():
        return False

    if not thumbnail.is_file():
        return False

    try:
        return (
            thumbnail.stat().st_mtime
            >= source.stat().st_mtime
        )
    except FileNotFoundError:
        return False


def _write_thumbnail(
    *,
    source: Path,
    destination: Path,
) -> None:
    image, image_source = _read_image(
        source
    )

    image = _prepare_dimensions(
        image
    )
    image = _to_uint8(
        image
    )
    image = _to_opencv_color(
        image,
        image_source,
    )
    image = _resize_thumbnail(
        image
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            88,
        ],
    )

    if not success:
        raise ValueError(
            f"Could not encode thumbnail for {source.name}."
        )

    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.jpg"
    )

    try:
        temporary.write_bytes(
            encoded.tobytes()
        )

        os.replace(
            temporary,
            destination,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )


@router.get("/{image_path:path}")
async def get_thumbnail(
    job_id: str,
    image_path: str,
):
    _require_image_job(job_id)

    source = _resolve_source_image(
        job_id=job_id,
        image_path=image_path,
    )

    destination = _thumbnail_path(
        job_id=job_id,
        source=source,
    )

    if not _thumbnail_is_current(
        source=source,
        thumbnail=destination,
    ):
        try:
            _write_thumbnail(
                source=source,
                destination=destination,
            )

            logger.info(
                "Created thumbnail: job_id=%s source=%s destination=%s",
                job_id,
                source,
                destination,
            )

        except Exception as exc:
            logger.exception(
                "Thumbnail generation failed: job_id=%s source=%s",
                job_id,
                source,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Thumbnail generation failed for "
                    f"'{image_path}': {exc}"
                ),
            )

    return FileResponse(
        destination,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=3600",
        },
    )
