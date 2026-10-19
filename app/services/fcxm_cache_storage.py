import pickle
from typing import Any

from app.services.job_paths import get_job_paths


def save_fcxm_plot_cache(job_id: str, plot_cache: Any) -> None:
    """
    Serialize the FCXM plot cache to the job's pickle file.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    plot_cache : Any
        Cached per-file FCXM plotting data.
    """
    path = get_job_paths(job_id, create=True).plot_cache

    with path.open("wb") as handle:
        pickle.dump(plot_cache, handle)


def load_fcxm_plot_cache(job_id: str) -> Any:
    """
    Load the FCXM plot cache from the job's pickle file.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    Any
        The FCXM plot cache from the job's pickle file.
    """
    path = get_job_paths(job_id).plot_cache

    with path.open("rb") as handle:
        return pickle.load(handle)
