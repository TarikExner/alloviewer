import pickle
from typing import Any

from app.services.job_paths import get_job_paths


def save_fcxm_plot_cache(job_id: str, plot_cache: Any) -> None:
    path = get_job_paths(job_id, create=True).plot_cache

    with path.open("wb") as handle:
        pickle.dump(plot_cache, handle)


def load_fcxm_plot_cache(job_id: str) -> Any:
    path = get_job_paths(job_id).plot_cache

    with path.open("rb") as handle:
        return pickle.load(handle)
