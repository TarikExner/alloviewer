import json
import time
from typing import Any

from app.core.redis_settings import redis_settings
from app.services.redis_client import redis_client
from redis.exceptions import WatchError

TTL = redis_settings.job_state_ttl_seconds


def _json_default(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "__dict__"):
        return value.__dict__

    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _loads(raw: str | None) -> Any:
    if raw is None:
        return None
    return json.loads(raw)


def _set_json(key: str, value: Any) -> None:
    redis_client.set(key, _dumps(value), ex=TTL)


def _get_json(key: str) -> Any:
    return _loads(redis_client.get(key))


def _merge_json(key: str, updates: dict[str, Any]) -> dict[str, Any]:
    pipe = redis_client.pipeline()
    while True:
        try:
            pipe.watch(key)
            current_raw = pipe.get(key)
            current = _loads(current_raw) or {}
            current.update(updates)
            pipe.multi()
            pipe.set(key, _dumps(current), ex=TTL)
            pipe.execute()
            return current
        except WatchError:
            continue
        finally:
            pipe.reset()


def _touch(keys: list[str]) -> None:
    for key in keys:
        redis_client.expire(key, TTL)


def image_progress_key(job_id: str) -> str:
    return f"image:{job_id}:progress"


def image_result_key(job_id: str) -> str:
    return f"image:{job_id}:result"


def set_image_progress(job_id: str, progress: dict[str, Any]) -> None:
    progress = {
        **progress,
        "last_access": time.time(),
    }
    _set_json(image_progress_key(job_id), progress)


def update_image_progress(job_id: str, **updates: Any) -> dict[str, Any]:
    updates = {
        **updates,
        "last_access": time.time(),
    }
    return _merge_json(image_progress_key(job_id), updates)


def get_image_progress(job_id: str) -> dict[str, Any] | None:
    key = image_progress_key(job_id)
    progress = _get_json(key)
    if progress is not None:
        _touch([key, image_result_key(job_id)])
    return progress


def set_image_result(job_id: str, result: Any) -> None:
    _set_json(image_result_key(job_id), result)


def get_image_result(job_id: str) -> Any:
    key = image_result_key(job_id)
    result = _get_json(key)
    if result is not None:
        _touch([key, image_progress_key(job_id)])
    return result


def append_image_done_well(job_id: str, well_id: str, done: int | None = None) -> None:
    key = image_progress_key(job_id)

    pipe = redis_client.pipeline()
    while True:
        try:
            pipe.watch(key)
            current = _loads(pipe.get(key)) or {}

            done_wells = list(current.get("done_wells", []))
            if well_id not in done_wells:
                done_wells.append(well_id)

            current["done_wells"] = done_wells

            if done is not None:
                current["done"] = done

            current["last_access"] = time.time()

            pipe.multi()
            pipe.set(key, _dumps(current), ex=TTL)
            pipe.execute()
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

def fcxm_progress_key(job_id: str) -> str:
    return f"fcxm:{job_id}:progress"


def fcxm_result_key(job_id: str) -> str:
    return f"fcxm:{job_id}:result"


def fcxm_request_key(job_id: str) -> str:
    return f"fcxm:{job_id}:request"


def fcxm_plot_cache_path_key(job_id: str) -> str:
    return f"fcxm:{job_id}:plot_cache_path"


def set_fcxm_progress(job_id: str, progress: dict[str, Any]) -> None:
    progress = {
        **progress,
        "last_access": time.time(),
    }
    _set_json(fcxm_progress_key(job_id), progress)


def update_fcxm_progress(job_id: str, **updates: Any) -> dict[str, Any]:
    updates = {
        **updates,
        "last_access": time.time(),
    }
    return _merge_json(fcxm_progress_key(job_id), updates)


def get_fcxm_progress(job_id: str) -> dict[str, Any] | None:
    key = fcxm_progress_key(job_id)
    progress = _get_json(key)
    if progress is not None:
        _touch(
            [
                key,
                fcxm_result_key(job_id),
                fcxm_request_key(job_id),
                fcxm_plot_cache_path_key(job_id),
            ]
        )
    return progress


def set_fcxm_result(job_id: str, result: Any) -> None:
    _set_json(fcxm_result_key(job_id), result)


def get_fcxm_result(job_id: str) -> Any:
    key = fcxm_result_key(job_id)
    result = _get_json(key)
    if result is not None:
        _touch([key, fcxm_progress_key(job_id)])
    return result


def set_fcxm_request(job_id: str, request: dict[str, Any]) -> None:
    _set_json(fcxm_request_key(job_id), request)


def get_fcxm_request(job_id: str) -> dict[str, Any] | None:
    return _get_json(fcxm_request_key(job_id))


def set_fcxm_plot_cache_path(job_id: str, path: str) -> None:
    _set_json(fcxm_plot_cache_path_key(job_id), path)


def get_fcxm_plot_cache_path(job_id: str) -> str | None:
    return _get_json(fcxm_plot_cache_path_key(job_id))


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

            done_filenames = list(current.get("done_filenames", []))
            if filename not in done_filenames:
                done_filenames.append(filename)

            current["done_filenames"] = done_filenames

            if done_files is not None:
                current["done_files"] = done_files

            current["last_access"] = time.time()

            pipe.multi()
            pipe.set(key, _dumps(current), ex=TTL)
            pipe.execute()
            return
        except WatchError:
            continue
        finally:
            pipe.reset()


def delete_fcxm_job(job_id: str) -> None:
    redis_client.delete(
        fcxm_progress_key(job_id),
        fcxm_result_key(job_id),
        fcxm_request_key(job_id),
        fcxm_plot_cache_path_key(job_id),
    )

def delete_all_job_state(job_id: str) -> None:
    redis_client.delete(
        image_progress_key(job_id),
        image_result_key(job_id),
        fcxm_progress_key(job_id),
        fcxm_result_key(job_id),
        fcxm_request_key(job_id),
        fcxm_plot_cache_path_key(job_id),
    )
