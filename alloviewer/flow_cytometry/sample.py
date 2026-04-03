
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .fcs_file import FCSFile


def resolve_under_data_dir(rel_or_abs: str, data_dir: str) -> str:
    """
    Frontend sends file paths returned by /api/upload, which are relative to DATA_DIR.
    This resolves them safely under DATA_DIR and returns an absolute string path.
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
        return f"Sample(name='{self.name}', role='{self.role}', n_files={len(self.files)})"

class Dataset:
    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def get(self, role: str) -> List[Sample]:
        return [s for s in self.samples if s.role == role]

    def get_one(self, role: str) -> Sample:
        xs = self.get(role)
        if len(xs) != 1:
            raise ValueError(f"Expected exactly one Sample with role='{role}', got {len(xs)}.")
        return xs[0]

    def __repr__(self) -> str:
        roles = {}
        for s in self.samples:
            roles[s.role] = roles.get(s.role, 0) + 1
        return f"Dataset(n_samples={len(self.samples)}, roles={roles})"
