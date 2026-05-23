from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .fcs_file import FCSFile


def resolve_under_data_dir(rel_or_abs: str, data_dir: str) -> str:
    """Resolve an uploaded FCS path under a data directory.

    Parameters
    ----------
    rel_or_abs : str
        Relative or absolute path received from the frontend.
    data_dir : str
        Base directory under which uploaded files are stored.

    Returns
    -------
    str
        Absolute resolved path to an existing ``.fcs`` file.

    Raises
    ------
    ValueError
        If ``data_dir`` is empty, the resolved path escapes ``data_dir``, the
        file does not exist, or the file is not an ``.fcs`` file.

    Notes
    -----
    Unsafe absolute paths and parent-directory references are reduced to their
    basename before resolution.
    """
    if not data_dir:
        raise ValueError("data_dir is required to resolve uploaded files.")

    base = Path(data_dir).resolve()
    raw = (rel_or_abs or "").replace("\\", "/").strip()

    if raw.startswith("/") or raw.startswith("../") or "/.." in raw:
        raw = Path(raw).name

    dest = (base / raw).resolve()

    if not str(dest).startswith(str(base)):
        raise ValueError("Bad path (outside DATA_DIR).")

    if not dest.exists():
        raise ValueError(f"File not found: {raw} (resolved: {dest})")

    if dest.suffix.lower() != ".fcs":
        raise ValueError(f"Not an .fcs file: {raw}")

    return str(dest)


class Sample:
    """Sample containing one or more FCS files.

    Parameters
    ----------
    name : str
        Sample name.
    role : str
        Sample role, such as ``"NC"``, ``"PC"``, or ``"SAMPLE"``.
    file_paths : list of str
        Paths to FCS files belonging to the sample.
    fcs_kwargs : dict or None, optional
        Keyword arguments passed to :class:`FCSFile`.

    Attributes
    ----------
    name : str
        Sample name.
    role : str
        Sample role.
    file_paths : list of str
        Stored FCS file paths.
    files : list of FCSFile
        Loaded FCS file objects.

    Raises
    ------
    ValueError
        If ``file_paths`` is empty.
    """

    def __init__(
        self,
        name: str,
        role: str,
        file_paths: List[str],
        fcs_kwargs: Optional[Dict] = None,
    ) -> None:
        if fcs_kwargs is None:
            fcs_kwargs = {}

        if not file_paths:
            raise ValueError("file_paths must have at least one entry.")

        self.name = name
        self.role = role
        self.file_paths = list(file_paths)

        self.files = [FCSFile(p, **fcs_kwargs) for p in self.file_paths]

    def __repr__(self) -> str:
        """Return a compact sample summary.

        Returns
        -------
        str
            Summary containing sample name, role, and file count.
        """
        return (
            f"Sample(name='{self.name}', "
            f"role='{self.role}', "
            f"n_files={len(self.files)})"
        )


class Dataset:
    """Collection of samples.

    Parameters
    ----------
    samples : list of Sample
        Samples included in the dataset.

    Attributes
    ----------
    samples : list of Sample
        Stored samples.
    """

    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def get(self, role: str) -> List[Sample]:
        """Return samples with a given role.

        Parameters
        ----------
        role : str
            Sample role to filter by.

        Returns
        -------
        list of Sample
            Samples whose ``role`` equals the requested role.
        """
        return [s for s in self.samples if s.role == role]

    def get_one(self, role: str) -> Sample:
        """Return exactly one sample with a given role.

        Parameters
        ----------
        role : str
            Sample role to filter by.

        Returns
        -------
        Sample
            The single matching sample.

        Raises
        ------
        ValueError
            If zero or more than one matching sample exists.
        """
        xs = self.get(role)

        if len(xs) != 1:
            raise ValueError(
                f"Expected exactly one Sample with role='{role}', got {len(xs)}."
            )

        return xs[0]

    def __repr__(self) -> str:
        """Return a compact dataset summary.

        Returns
        -------
        str
            Summary containing sample count and role counts.
        """
        roles = {}

        for s in self.samples:
            roles[s.role] = roles.get(s.role, 0) + 1

        return f"Dataset(n_samples={len(self.samples)}, roles={roles})"
