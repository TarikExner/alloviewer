from pathlib import Path
import posixpath
from fastapi import HTTPException

def resolve_under_base_dir(base_dir: Path, rel_or_abs: str) -> Path:
    """
    Resolve a user-provided path safely under base_dir.

    Parameters
    ----------
    base_dir : Path
        Directory under which the result is resolved.
    rel_or_abs : str
        Submitted path; absolute or parent-prefixed paths are reduced to their basename.

    Returns
    -------
    Path
        Computed value of ``dest``.

    Raises
    ------
    HTTPException
        HTTP 400 for the corresponding request validation, job-state, or processing error.
    """
    base_dir = base_dir.resolve()
    raw = (rel_or_abs or "").replace("\\", "/").strip()

    # normalize and block traversal
    rel_path = posixpath.normpath(raw)
    if rel_path.startswith("../") or rel_path.startswith("/"):
        rel_path = posixpath.basename(rel_path)

    dest = (base_dir / rel_path).resolve()
    if not str(dest).startswith(str(base_dir)):
        raise HTTPException(status_code=400, detail="Bad path (outside data dir)")
    return dest

