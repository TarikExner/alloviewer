import pandas as pd
from typing import Union
from pathlib import Path
import os

SCATTER_CHANNELS = {
    "FSC-A",
    "FSC-H",
    "FSC-W",
    "SSC-A",
    "SSC-H",
    "SSC-W",
}

def normalize_scatter_channel(channel: str) -> str:
    """
    Strip a channel name, uppercase it, and replace underscores with hyphens.

    Parameters
    ----------
    channel : str
        FCS detector or channel name.

    Returns
    -------
    str
        Computed value of ``channel.strip().upper().replace('_', '-')``.
    """
    return channel.strip().upper().replace("_", "-")

def is_scatter_channel(channel: str) -> bool:
    """
    Test whether a normalized channel name belongs to the known scatter set.

    Parameters
    ----------
    channel : str
        FCS detector or channel name.

    Returns
    -------
    bool
        True when a normalized channel name belongs to the known scatter set; False
        otherwise.
    """
    return normalize_scatter_channel(channel) in SCATTER_CHANNELS

def full_path(dir: Union[Path, str], file_name: str):
    """
    Join a directory and filename using the platform's path rules.

    Parameters
    ----------
    dir : Union[Path, str]
        Directory containing the file.
    file_name : str
        File path or unique basename identifying the selected FCS file.

    Returns
    -------
    str
        Joined file path.
    """
    return os.path.join(dir, file_name)

def assert_dfs_equal(dfs: list[pd.DataFrame]) -> None:
    """
    Require a nonempty list of pairwise equal DataFrames.

    Parameters
    ----------
    dfs : list[pandas.DataFrame]
        DataFrames to compare against the first entry.

    Raises
    ------
    ValueError
        No dataframes provided.
    AssertionError
        DataFrame at index {i} is not equal to DataFrame at index 0:
        {e}
    """
    if not dfs:
        raise ValueError("No dataframes provided.")

    reference = dfs[0]

    for i, df in enumerate(dfs[1:], start=1):
        try:
            pd.testing.assert_frame_equal(reference, df)
        except AssertionError as e:
            raise AssertionError(
                f"DataFrame at index {i} is not equal to DataFrame at index 0:\n{e}"
            ) from e
