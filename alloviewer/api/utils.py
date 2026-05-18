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
    return channel.strip().upper().replace("_", "-")

def is_scatter_channel(channel: str) -> bool:
    return normalize_scatter_channel(channel) in SCATTER_CHANNELS

def full_path(dir: Union[Path, str], file_name: str):
    return os.path.join(dir, file_name)

def assert_dfs_equal(dfs: list[pd.DataFrame]) -> None:
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
