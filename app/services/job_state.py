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
    """
    Convert model-like objects, objects with attributes, or other values for JSON.

    Parameters
    ----------
    value : Any
        Input value to validate, convert, or format as specified by this helper.

    Returns
    -------
    Any
        Depending on the input and branch, returns ``value.model_dump()``,
        ``value.__dict__``, ``str(value)``.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "__dict__"):
        return value.__dict__

    return str(value)


def _dumps(value: Any) -> str:
    """
    Encode job state as JSON using the custom value converter.

    Parameters
    ----------
    value : Any
        Input value to validate, convert, or format as specified by this helper.

    Returns
    -------
    str
        Computed value of ``json.dumps(value, default=_json_default)``.
    """
    return json.dumps(
        value,
        default=_json_default,
    )


def _loads(raw: str | bytes | None) -> Any:
    """
    Decode Redis JSON bytes or text, preserving None for a missing key.

    Parameters
    ----------
    raw : str | bytes | None
        Serialized JSON bytes or text read from Redis, or None for an absent key.

    Returns
    -------
    Any
        Depending on the input and branch, returns ``None``, ``json.loads(raw)``.
    """
    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    return json.loads(raw)


def _set_json(
    key: str,
    value: Any,
) -> None:
    """
    Store JSON in Redis with the configured job-state expiry.

    Parameters
    ----------
    key : str
        Mapping key or attribute name selected by the helper.
    value : Any
        Input value to validate, convert, or format as specified by this helper.
    """
    redis_client.set(
        key,
        _dumps(value),
        ex=TTL,
    )


def _get_json(key: str) -> Any:
    """
    Read and decode a Redis JSON key.

    Parameters
    ----------
    key : str
        Mapping key or attribute name selected by the helper.

    Returns
    -------
    Any
        And decode a Redis JSON key.
    """
    return _loads(redis_client.get(key))


def _merge_json(
    key: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge fields into a Redis JSON object using an optimistic transaction.

    Parameters
    ----------
    key : str
        Mapping key or attribute name selected by the helper.
    updates : dict[str, Any]
        Fields to merge into the existing state.

    Returns
    -------
    dict[str, Any]
        Computed value of ``current``.
    """
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
    """
    Refresh the expiry of each supplied Redis key.

    Parameters
    ----------
    keys : list[str]
        Ordered candidate keys checked when reading the mapping.
    """
    for key in keys:
        redis_client.expire(key, TTL)


def _persist_progress(
    job_id: str,
    progress: dict[str, Any],
) -> None:
    """
    Mirror the latest Redis progress state into job.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    progress : dict[str, Any]
        Job progress fields to persist.
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

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    dict[str, Any] | None
        Progress from job.json when the Redis entry has expired.
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

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    Any
        The persisted result when the Redis entry has expired.
    """
    path = get_job_paths(job_id).result

    if not path.exists() or not path.is_file():
        return None

    return read_json(path)


# ---------------------------------------------------------------------------
# Image-analysis state
# ---------------------------------------------------------------------------


def image_progress_key(job_id: str) -> str:
    """
    Build the Redis key for image job progress.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    str
        The Redis key for image job progress.
    """
    return f"image:{job_id}:progress"


def image_result_key(job_id: str) -> str:
    """
    Build the Redis key for image job results.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    str
        The Redis key for image job results.
    """
    return f"image:{job_id}:result"


def set_image_progress(
    job_id: str,
    progress: dict[str, Any],
) -> None:
    """
    Store image progress in Redis and mirror it into job.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    progress : dict[str, Any]
        Job progress fields to persist.
    """
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
    """
    Merge image progress fields in Redis and mirror them into job.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    **updates : Any
        Fields to merge into the existing state.

    Returns
    -------
    dict[str, Any]
        Computed value of ``progress``.
    """
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
    """
    Read image progress from Redis, falling back to job.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    dict[str, Any] | None
        Image progress from Redis, falling back to job.json.
    """
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
    """
    Store the image result in Redis and result.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    result : Any
        Analysis result containing per-well values and assay summaries.
    """
    _set_json(
        image_result_key(job_id),
        result,
    )

    write_json_atomic(
        get_job_paths(job_id).result,
        result,
    )


def get_image_result(job_id: str) -> Any:
    """
    Read the image result from Redis, falling back to result.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    Any
        The image result from Redis, falling back to result.json.
    """
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
    """
    Add a completed well once and optionally update the count in job progress.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    well_id : str
        Plate well identifier, such as ``A1``.
    done : int | None
        Updated completed-well count; None leaves the count unchanged. Default is ``None``.
    """
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
    """
    Delete the image progress and result keys from Redis.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    """
    redis_client.delete(
        image_progress_key(job_id),
        image_result_key(job_id),
    )


# ---------------------------------------------------------------------------
# FCXM state
# ---------------------------------------------------------------------------


def fcxm_progress_key(job_id: str) -> str:
    """
    Build the Redis key for fcxm job progress.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    str
        The Redis key for fcxm job progress.
    """
    return f"fcxm:{job_id}:progress"


def fcxm_result_key(job_id: str) -> str:
    """
    Build the Redis key for fcxm job results.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    str
        The Redis key for fcxm job results.
    """
    return f"fcxm:{job_id}:result"


def set_fcxm_progress(
    job_id: str,
    progress: dict[str, Any],
) -> None:
    """
    Store fcxm progress in Redis and mirror it into job.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    progress : dict[str, Any]
        Job progress fields to persist.
    """
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
    """
    Merge fcxm progress fields in Redis and mirror them into job.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    **updates : Any
        Fields to merge into the existing state.

    Returns
    -------
    dict[str, Any]
        Computed value of ``progress``.
    """
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
    """
    Read fcxm progress from Redis, falling back to job.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    dict[str, Any] | None
        Fcxm progress from Redis, falling back to job.json.
    """
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
    """
    Store the fcxm result in Redis and result.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    result : Any
        Analysis result containing per-well values and assay summaries.
    """
    _set_json(
        fcxm_result_key(job_id),
        result,
    )

    write_json_atomic(
        get_job_paths(job_id).result,
        result,
    )


def get_fcxm_result(job_id: str) -> Any:
    """
    Read the fcxm result from Redis, falling back to result.json.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    Any
        The fcxm result from Redis, falling back to result.json.
    """
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
    """
    Add a completed filename once and optionally update the count in job progress.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    filename : str
        Name or path of the file selected by this operation.
    done_files : int | None
        Updated completed-file count; None leaves the count unchanged. Default is ``None``.
    """
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
    """
    Delete the fcxm progress and result keys from Redis.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    """
    redis_client.delete(
        fcxm_progress_key(job_id),
        fcxm_result_key(job_id),
    )


def delete_all_job_state(job_id: str) -> None:
    """
    Delete all Redis state associated with one job.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Notes
    -----
    Filesystem cleanup is handled separately by the job registry or cleanup
    service.
    """
    redis_client.delete(
        image_progress_key(job_id),
        image_result_key(job_id),
        fcxm_progress_key(job_id),
        fcxm_result_key(job_id),
    )
