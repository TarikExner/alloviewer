from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from app.core.settings import settings


class JobPathError(ValueError):
    pass


def normalize_job_id(job_id: str) -> str:
    try:
        return str(UUID(str(job_id)))
    except (TypeError, ValueError) as exc:
        raise JobPathError(f"Invalid job ID: {job_id!r}") from exc


def normalize_relative_path(value: str) -> Path:
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
    root = get_job_paths(job_id).root.resolve()
    return path.resolve().relative_to(root).as_posix()
