from __future__ import annotations

import json
import time
from typing import Any

from redis.exceptions import WatchError

from app.core.redis_settings import redis_settings
from app.services.job_paths import get_job_paths
from app.services.job_registry import (
    get_job,
    read_json,
    update_job,
    write_json_atomic,
)
from app.services.redis_client import redis_client


TTL = redis_settings.job_state_ttl_seconds


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "__dict__"):
        return value.__dict__

    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
    )


def _loads(raw: str | bytes | None) -> Any:
    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    return json.loads(raw)


def _set_json(
    key: str,
    value: Any,
) -> None:
    redis_client.set(
        key,
        _dumps(value),
        ex=TTL,
    )


def _get_json(key: str) -> Any:
    return _loads(redis_client.get(key))


def _merge_json(
    key: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    pipe = redis_client.pipeline()

    while True:
        try:
            pipe.watch(key)

            current_raw = pipe.get(key)
            current = _loads(current_raw) or {}

            current.update(updates)

            pipe.multi()
            pipe.set(
                key,
                _dumps(current),
                ex=TTL,
            )
            pipe.execute()

            return current

        except WatchError:
            continue

        finally:
            pipe.reset()


def _touch(keys: list[str]) -> None:
    for key in keys:
        redis_client.expire(key, TTL)


def _persist_progress(
    job_id: str,
    progress: dict[str, Any],
) -> None:
    """
    Mirror the latest Redis progress state into job.json.
    """
    error = None

    if progress.get("error"):
        error = {
            "message": progress.get("error"),
            "type": progress.get("error_type"),
        }

    update_job(
        job_id,
        status=str(progress.get("status") or "draft"),
        stage=progress.get("stage"),
        progress=progress,
        error=error,
    )


def _progress_from_job_file(
    job_id: str,
) -> dict[str, Any] | None:
    """
    Recover progress from job.json when the Redis entry has expired.
    """
    job = get_job(job_id)

    if job is None:
        return None

    progress = job.get("progress")

    if not isinstance(progress, dict):
        return None

    if not progress:
        return None

    return progress


def _result_from_job_file(job_id: str) -> Any:
    """
    Recover the persisted result when the Redis entry has expired.
    """
    path = get_job_paths(job_id).result

    if not path.exists() or not path.is_file():
        return None

    return read_json(path)


# ---------------------------------------------------------------------------
# Image-analysis state
# ---------------------------------------------------------------------------


def image_progress_key(job_id: str) -> str:
    return f"image:{job_id}:progress"


def image_result_key(job_id: str) -> str:
    return f"image:{job_id}:result"


def set_image_progress(
    job_id: str,
    progress: dict[str, Any],
) -> None:
    complete_progress = {
        **progress,
        "last_access": time.time(),
    }

    _set_json(
        image_progress_key(job_id),
        complete_progress,
    )

    _persist_progress(
        job_id,
        complete_progress,
    )


def update_image_progress(
    job_id: str,
    **updates: Any,
) -> dict[str, Any]:
    complete_updates = {
        **updates,
        "last_access": time.time(),
    }

    progress = _merge_json(
        image_progress_key(job_id),
        complete_updates,
    )

    _persist_progress(
        job_id,
        progress,
    )

    return progress


def get_image_progress(
    job_id: str,
) -> dict[str, Any] | None:
    key = image_progress_key(job_id)
    progress = _get_json(key)

    if progress is not None:
        _touch(
            [
                key,
                image_result_key(job_id),
            ]
        )
        return progress

    return _progress_from_job_file(job_id)


def set_image_result(
    job_id: str,
    result: Any,
) -> None:
    _set_json(
        image_result_key(job_id),
        result,
    )

    write_json_atomic(
        get_job_paths(job_id).result,
        result,
    )


def get_image_result(job_id: str) -> Any:
    key = image_result_key(job_id)
    result = _get_json(key)

    if result is not None:
        _touch(
            [
                key,
                image_progress_key(job_id),
            ]
        )
        return result

    return _result_from_job_file(job_id)


def append_image_done_well(
    job_id: str,
    well_id: str,
    done: int | None = None,
) -> None:
    key = image_progress_key(job_id)
    pipe = redis_client.pipeline()

    while True:
        try:
            pipe.watch(key)

            current = _loads(pipe.get(key)) or {}

            done_wells = list(
                current.get("done_wells", [])
            )

            if well_id not in done_wells:
                done_wells.append(well_id)

            current["done_wells"] = done_wells

            if done is not None:
                current["done"] = done

            current["last_access"] = time.time()

            pipe.multi()
            pipe.set(
                key,
                _dumps(current),
                ex=TTL,
            )
            pipe.execute()

            _persist_progress(
                job_id,
                current,
            )

            return

        except WatchError:
            continue

        finally:
            pipe.reset()


def delete_image_job(job_id: str) -> None:
    redis_client.delete(
        image_progress_key(job_id),
        image_result_key(job_id),
    )


# ---------------------------------------------------------------------------
# FCXM state
# ---------------------------------------------------------------------------


def fcxm_progress_key(job_id: str) -> str:
    return f"fcxm:{job_id}:progress"


def fcxm_result_key(job_id: str) -> str:
    return f"fcxm:{job_id}:result"


def set_fcxm_progress(
    job_id: str,
    progress: dict[str, Any],
) -> None:
    complete_progress = {
        **progress,
        "last_access": time.time(),
    }

    _set_json(
        fcxm_progress_key(job_id),
        complete_progress,
    )

    _persist_progress(
        job_id,
        complete_progress,
    )


def update_fcxm_progress(
    job_id: str,
    **updates: Any,
) -> dict[str, Any]:
    complete_updates = {
        **updates,
        "last_access": time.time(),
    }

    progress = _merge_json(
        fcxm_progress_key(job_id),
        complete_updates,
    )

    _persist_progress(
        job_id,
        progress,
    )

    return progress


def get_fcxm_progress(
    job_id: str,
) -> dict[str, Any] | None:
    key = fcxm_progress_key(job_id)
    progress = _get_json(key)

    if progress is not None:
        _touch(
            [
                key,
                fcxm_result_key(job_id),
            ]
        )
        return progress

    return _progress_from_job_file(job_id)


def set_fcxm_result(
    job_id: str,
    result: Any,
) -> None:
    _set_json(
        fcxm_result_key(job_id),
        result,
    )

    write_json_atomic(
        get_job_paths(job_id).result,
        result,
    )


def get_fcxm_result(job_id: str) -> Any:
    key = fcxm_result_key(job_id)
    result = _get_json(key)

    if result is not None:
        _touch(
            [
                key,
                fcxm_progress_key(job_id),
            ]
        )
        return result

    return _result_from_job_file(job_id)


def append_fcxm_done_filename(
    job_id: str,
    filename: str,
    done_files: int | None = None,
) -> None:
    key = fcxm_progress_key(job_id)
    pipe = redis_client.pipeline()

    while True:
        try:
            pipe.watch(key)

            current = _loads(pipe.get(key)) or {}

            done_filenames = list(
                current.get("done_filenames", [])
            )

            if filename not in done_filenames:
                done_filenames.append(filename)

            current["done_filenames"] = done_filenames

            if done_files is not None:
                current["done_files"] = done_files

            current["last_access"] = time.time()

            pipe.multi()
            pipe.set(
                key,
                _dumps(current),
                ex=TTL,
            )
            pipe.execute()

            _persist_progress(
                job_id,
                current,
            )

            return

        except WatchError:
            continue

        finally:
            pipe.reset()


def delete_fcxm_job(job_id: str) -> None:
    redis_client.delete(
        fcxm_progress_key(job_id),
        fcxm_result_key(job_id),
    )


def delete_all_job_state(job_id: str) -> None:
    """
    Delete all Redis state associated with one job.

    Filesystem cleanup is handled separately by the job registry or cleanup
    service.
    """
    redis_client.delete(
        image_progress_key(job_id),
        image_result_key(job_id),
        fcxm_progress_key(job_id),
        fcxm_result_key(job_id),
    )
