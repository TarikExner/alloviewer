from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from app.services.job_paths import get_job_paths, normalize_job_id


JobType = Literal["pra", "crossmatch", "fcxm"]
JobStatus = Literal[
    "draft",
    "queued",
    "running",
    "done",
    "error",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def create_job(job_type: JobType) -> dict[str, Any]:
    if job_type not in {"pra", "crossmatch", "fcxm"}:
        raise ValueError(f"Unsupported job type: {job_type}")

    job_id = str(uuid.uuid4())
    now = time.time()

    paths = get_job_paths(job_id, create=True)

    metadata = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": job_type,
        "status": "draft",
        "stage": None,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "progress": {},
    }

    write_json_atomic(paths.metadata, metadata)
    return metadata


def get_job(job_id: str) -> dict[str, Any] | None:
    paths = get_job_paths(job_id)

    if not paths.metadata.exists():
        return None

    return read_json(paths.metadata)


def require_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)

    if job is None:
        raise FileNotFoundError(f"Job not found: {normalize_job_id(job_id)}")

    return job


def require_job_type(
    job_id: str,
    allowed: set[JobType],
) -> dict[str, Any]:
    job = require_job(job_id)
    job_type = job.get("job_type")

    if job_type not in allowed:
        raise ValueError(
            f"Job type '{job_type}' is not valid for this operation."
        )

    return job


def update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    paths = get_job_paths(job_id)
    current = require_job(job_id)

    current.update(updates)
    current["updated_at"] = time.time()

    write_json_atomic(paths.metadata, current)
    return current


def delete_job_directory(job_id: str) -> bool:
    paths = get_job_paths(job_id)

    if not paths.root.exists():
        return False

    shutil.rmtree(paths.root)
    return True
