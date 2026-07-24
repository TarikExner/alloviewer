from pathlib import Path
from typing import Literal
import unicodedata
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.config import (
    FCS_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MAX_FILE_SIZE_MB,
)
from app.services.job_paths import (
    JobPathError,
    get_job_paths,
    job_relative_path,
    normalize_relative_path,
)
from app.services.job_registry import (
    require_job_type,
    update_job,
)


router = APIRouter(
    prefix="/api/jobs/{job_id}/uploads",
    tags=["uploads"],
)


def _duplicate_key(value: str) -> str:
    return unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()


@router.post("/{upload_kind}")
async def upload_job_files(
    job_id: str,
    upload_kind: Literal[
        "images",
        "fcs",
    ],
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(...),
):
    if len(files) != len(relative_paths):
        raise HTTPException(
            status_code=400,
            detail=(
                "Each uploaded file requires "
                "one relative path."
            ),
        )

    try:
        if upload_kind == "images":
            require_job_type(
                job_id,
                {"pra", "crossmatch"},
            )
        else:
            require_job_type(
                job_id,
                {"fcxm"},
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )

    paths = get_job_paths(
        job_id,
        create=True,
    )

    target_root = (
        paths.image_uploads
        if upload_kind == "images"
        else paths.fcs_uploads
    )

    allowed_extensions = (
        IMAGE_EXTENSIONS
        if upload_kind == "images"
        else FCS_EXTENSIONS
    )

    prepared: list[
        tuple[
            int,
            UploadFile,
            Path,
            Path,
            str,
        ]
    ] = []

    seen_paths: set[str] = set()
    seen_filenames: set[str] = set()

    for index, (
        upload,
        raw_relative_path,
    ) in enumerate(
        zip(files, relative_paths)
    ):
        try:
            relative = (
                normalize_relative_path(
                    raw_relative_path
                )
            )
        except JobPathError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        relative_key = _duplicate_key(
            relative.as_posix()
        )

        if relative_key in seen_paths:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Duplicate upload path: "
                    f"{relative.as_posix()}"
                ),
            )

        seen_paths.add(relative_key)

        if upload_kind == "images":
            filename_key = _duplicate_key(
                relative.name
            )

            if filename_key in seen_filenames:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Duplicate filename in one "
                        "image upload is not allowed: "
                        f"{relative.name}"
                    ),
                )

            seen_filenames.add(filename_key)

        extension = relative.suffix.lower()

        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=(
                    "File type not allowed: "
                    f"{relative.name}"
                ),
            )

        destination = (
            target_root / relative
        ).resolve()

        try:
            destination.relative_to(
                target_root.resolve()
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid upload path.",
            )

        original_filename = Path(
            upload.filename or relative.name
        ).name

        prepared.append(
            (
                index,
                upload,
                relative,
                destination,
                original_filename,
            )
        )

    saved: list[dict] = []
    temporary_files: list[
        tuple[Path, Path]
    ] = []

    try:
        for (
            index,
            upload,
            relative,
            destination,
            original_filename,
        ) in prepared:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = (
                destination.parent
                / (
                    f".{destination.name}."
                    f"{uuid4().hex}.uploading"
                )
            )

            size_bytes = 0
            temporary_files.append(
                (temporary, destination)
            )

            with temporary.open("xb") as output:
                while chunk := await upload.read(
                    1024 * 1024
                ):
                    size_bytes += len(chunk)

                    if (
                        size_bytes
                        > MAX_FILE_SIZE_MB
                        * 1024
                        * 1024
                    ):
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "File too large: "
                                f"{relative.name}"
                            ),
                        )

                    output.write(chunk)

            saved.append(
                {
                    "index": index,
                    "filename": job_relative_path(
                        job_id,
                        destination,
                    ),
                    "original_filename": (
                        original_filename
                    ),
                    "relative_path": (
                        relative.as_posix()
                    ),
                    "size_mb": round(
                        size_bytes
                        / (1024 * 1024),
                        2,
                    ),
                }
            )

        for temporary, destination in (
            temporary_files
        ):
            temporary.replace(destination)

    except Exception:
        for temporary, _ in (
            temporary_files
        ):
            temporary.unlink(
                missing_ok=True
            )
        raise

    update_job(
        job_id,
        status="draft",
        upload_count=len(saved),
    )

    return {"saved": saved}
