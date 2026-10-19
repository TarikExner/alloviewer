from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from app.core.settings import settings


class JobPathError(ValueError):
    """
    Signal an invalid job identifier or a path outside the allowed job folder.
    """
    pass


def normalize_job_id(job_id: str) -> str:
    """
    Parse a UUID and return its standard string form.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.

    Returns
    -------
    str
        UUID in its standard lowercase, hyphen-separated string form.

    Raises
    ------
    JobPathError
        Invalid job ID: {job_id}
    """
    try:
        return str(UUID(str(job_id)))
    except (TypeError, ValueError) as exc:
        raise JobPathError(f"Invalid job ID: {job_id!r}") from exc


def normalize_relative_path(value: str) -> Path:
    """
    Normalize separators and reject absolute, parent, and colon-containing paths.

    Parameters
    ----------
    value : str
        Input value to validate, convert, or format as specified by this helper.

    Returns
    -------
    Path
        Validated relative path with normalized separators.

    Raises
    ------
    JobPathError
        A relative file path is required. Invalid relative file path.
    """
    raw = str(value or "").replace("\\", "/").strip()
    pure = PurePosixPath(raw)

    if not raw or pure.is_absolute():
        raise JobPathError("A relative file path is required.")

    parts = pure.parts

    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise JobPathError("Invalid relative file path.")

    if any(":" in part for part in parts):
        raise JobPathError("Invalid relative file path.")

    return Path(*parts)


@dataclass(frozen=True)
class JobPaths:
    """
    Store the paths used for one job's inputs, outputs, logs, and metadata.

    Attributes
    ----------
    root : Path
        Root directory of this job.
    metadata : Path
        Path to job.json containing job metadata.
    request : Path
        Path to request.json containing the submitted analysis request.
    result : Path
        Path to result.json containing the saved analysis result.
    log : Path
        Path to the job's log file.
    uploads : Path
        Root directory for files uploaded to the job.
    image_uploads : Path
        Directory containing the job's uploaded images.
    fcs_uploads : Path
        Directory containing the job's uploaded FCS files.
    layout_uploads : Path
        Directory containing uploaded plate-layout files.
    inputs : Path
        Directory containing normalized job input files.
    plate_layout : Path
        Path to the parsed plate-layout JSON file.
    outputs : Path
        Directory containing the job's generated analysis files.
    segmented : Path
        Directory containing generated segmentation images.
    thumbnails : Path
        Directory containing generated image thumbnails.
    plot_cache : Path
        Path to the serialized FCXM plot_cache.pkl file.
    reports : Path
        Directory containing generated job reports.
    summary_pdf : Path
        Path to the job's generated summary PDF.
    """
    root: Path

    metadata: Path
    request: Path
    result: Path
    log: Path

    uploads: Path
    image_uploads: Path
    fcs_uploads: Path
    layout_uploads: Path

    inputs: Path
    plate_layout: Path

    outputs: Path
    segmented: Path
    thumbnails: Path
    plot_cache: Path

    reports: Path
    summary_pdf: Path

    def create(self) -> "JobPaths":
        """
        Create the job directory tree and return this path collection.

        Returns
        -------
        JobPaths
            This instance, after the updates described above.
        """
        for directory in (
            self.root,
            self.uploads,
            self.image_uploads,
            self.fcs_uploads,
            self.layout_uploads,
            self.inputs,
            self.outputs,
            self.segmented,
            self.thumbnails,
            self.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        return self


def get_job_paths(job_id: str, *, create: bool = False) -> JobPaths:
    """
    Build the standard paths for a validated job UUID.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    create : bool
        If True, create the job directory tree before returning its paths. Default is
        ``False``.

    Returns
    -------
    JobPaths
        The standard paths for a validated job UUID.
    """
    normalized = normalize_job_id(job_id)
    root = Path(settings.data_dir).resolve() / "jobs" / normalized

    paths = JobPaths(
        root=root,
        metadata=root / "job.json",
        request=root / "request.json",
        result=root / "result.json",
        log=root / "job.log",
        uploads=root / "uploads",
        image_uploads=root / "uploads" / "images",
        fcs_uploads=root / "uploads" / "fcs",
        layout_uploads=root / "uploads" / "layout",
        inputs=root / "inputs",
        plate_layout=root / "inputs" / "plate_layout.json",
        outputs=root / "outputs",
        segmented=root / "outputs" / "segmented",
        thumbnails=root / "outputs" / "thumbnails",
        plot_cache=root / "outputs" / "plot_cache.pkl",
        reports=root / "reports",
        summary_pdf=root / "reports" / "summary.pdf",
    )

    return paths.create() if create else paths


def resolve_job_path(
    job_id: str,
    relative_path: str,
    *,
    required_root: Path | None = None,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a job-relative path and check the permitted directory boundaries.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    relative_path : str
        Path relative to the job root.
    required_root : Path | None
        Additional directory that must contain the resolved path; None adds no extra
        boundary. Default is ``None``.
    must_exist : bool
        If True, reject a resolved path that does not exist. Default is ``False``.

    Returns
    -------
    Path
        Computed value of ``candidate``.

    Raises
    ------
    JobPathError
        Path points outside the job directory. Path does not belong to the expected job
        folder.
    """
    paths = get_job_paths(job_id)
    rel = normalize_relative_path(relative_path)

    candidate = (paths.root / rel).resolve()
    root = paths.root.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise JobPathError("Path points outside the job directory.") from exc

    if required_root is not None:
        required = required_root.resolve()

        try:
            candidate.relative_to(required)
        except ValueError as exc:
            raise JobPathError(
                "Path does not belong to the expected job folder."
            ) from exc

    if must_exist and not candidate.exists():
        raise FileNotFoundError(str(candidate))

    return candidate


def job_relative_path(job_id: str, path: Path) -> str:
    """
    Express a resolved path relative to the job root using forward slashes.

    Parameters
    ----------
    job_id : str
        UUID string identifying the job within the configured data directory.
    path : Path
        Filesystem path of the file or directory processed by this helper.

    Returns
    -------
    str
        Computed value of ``path.resolve().relative_to(root).as_posix()``.
    """
    root = get_job_paths(job_id).root.resolve()
    return path.resolve().relative_to(root).as_posix()
