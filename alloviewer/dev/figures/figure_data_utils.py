import os
import pandas as pd
from typing import Optional, Any, Dict, Mapping, Union
import pickle
import numpy as np

from dataclasses import is_dataclass

def check_for_file(output_file: str) -> Optional[Union[pd.DataFrame, Any]]:
    """
    Read an existing CSV or pickled .dict cache, returning None otherwise.

    Parameters
    ----------
    output_file : str
        Destination file path.

    Returns
    -------
    Optional[Union[pandas.DataFrame, Any]]
        An existing CSV or pickled .dict cache, returning None otherwise.
    """
    if os.path.isfile(output_file):
        if output_file.endswith(".csv"):
            return pd.read_csv(output_file, index_col=False)
        elif output_file.endswith(".dict"):
            with open(output_file, "rb") as file:
                res = pickle.load(file)
            return res
    return

def config_to_kwargs_image_sim(
    sim_config: Any,
    rng: np.random.Generator,
    camera: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Turn either:
      - your SimulatorConfig (with .sample_kwargs(rng, camera)),
      - or a plain dict of defaults
    into kwargs for simulate_image.

    Parameters
    ----------
    sim_config : Any
        Scene-simulation configuration used to sample simulator keyword arguments.
    rng : np.random.Generator
        NumPy random-number generator used for sampling.
    camera : Optional[Any]
        Camera-dimension configuration used to sample image height and width. Default is
        ``None``.

    Returns
    -------
    Dict[str, Any]
        Depending on the input and branch, returns ``dict(sim_config.sample_kwargs(rng,
        camera=camera))``, ``dict(sim_config)``, ``{k: getattr(sim_config, k) for k in
        sim_config.__dataclass_fields__.keys()}``.

    Raises
    ------
    TypeError
        sim_config must be a SimulatorConfig-like object or a dict
    """
    if hasattr(sim_config, "sample_kwargs") and callable(sim_config.sample_kwargs):
        return dict(sim_config.sample_kwargs(rng, camera=camera))
    elif isinstance(sim_config, Mapping):
        return dict(sim_config)
    elif is_dataclass(sim_config):
        return {k: getattr(sim_config, k) for k in sim_config.__dataclass_fields__.keys()}
    else:
        raise TypeError("sim_config must be a SimulatorConfig-like object or a dict")

def merge_kwargs_image_sim(base: Dict[str, Any], overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Copy base simulation arguments and replace keys supplied in overrides.

    Parameters
    ----------
    base : Dict[str, Any]
        Base keyword-argument mapping copied before applying additional simulation settings.
    overrides : Optional[Mapping[str, Any]]
        Manual-call records to display.

    Returns
    -------
    Dict[str, Any]
        Computed value of ``out``.
    """
    out = dict(base)
    if overrides:
        for k, v in overrides.items():
            out[k] = v
    return out
