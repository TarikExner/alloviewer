from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


def natural_sort_key(value: str | Path) -> tuple[tuple[int, int | str], ...]:
    """
    Return a deterministic key that sorts embedded numbers numerically.

    Parameters
    ----------
    value : str | Path
        Input value to validate, convert, or format as specified by this helper.

    Returns
    -------
    tuple[tuple[int, int | str], ...]
        A deterministic key that sorts embedded numbers numerically.
    """
    parts = re.split(r"(\d+)", str(value))
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part != ""
    )


def ordered_anonymous_filename(
    position: int,
    suffix: str,
    *,
    prefix: str = "IMG",
    minimum_width: int = 4,
) -> str:
    """
    Create a sortable anonymous filename such as ``IMG_0001.tif``.

    Parameters
    ----------
    position : int
        Terminal row assigned to the progress bar.
    suffix : str
        File suffix appended to an image stem or used to select an image format.
    prefix : str
        Prefix prepended to a generated identifier. Default is ``'IMG'``.
    minimum_width : int
        Minimum number of digits used when formatting a numeric identifier. Default is
        ``4``.

    Returns
    -------
    str
        A sortable anonymous filename such as ``IMG_0001.tif``.

    Raises
    ------
    ValueError
        position must be at least 1
    """
    if position < 1:
        raise ValueError("position must be at least 1")
    clean_suffix = suffix.casefold()
    if clean_suffix and not clean_suffix.startswith("."):
        clean_suffix = f".{clean_suffix}"
    width = max(minimum_width, len(str(position)))
    return f"{prefix}_{position:0{width}d}{clean_suffix}"


def sort_paths_naturally(paths: Iterable[Path]) -> list[Path]:
    """
    Sort paths by the natural ordering of their filenames.

    Parameters
    ----------
    paths : Iterable[Path]
        File or directory paths to include in the operation.

    Returns
    -------
    list[Path]
        Computed value of ``sorted(paths, key=lambda path: natural_sort_key(path.name))``.
    """
    return sorted(paths, key=lambda path: natural_sort_key(path.name))

