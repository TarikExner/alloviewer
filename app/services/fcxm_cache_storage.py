import pickle
from pathlib import Path
from typing import Any

from app.core.settings import settings


def get_fcxm_cache_dir(job_id: str) -> Path:
    path = Path(settings.data_dir) / "jobs" / job_id / "fcxm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fcxm_plot_cache(job_id: str, plot_cache: Any) -> str:
    path = get_fcxm_cache_dir(job_id) / "plot_cache.pkl"

    with path.open("wb") as f:
        pickle.dump(plot_cache, f)

    return str(path)


def load_fcxm_plot_cache(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
