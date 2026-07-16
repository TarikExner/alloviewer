"""
Module for anonymization of human annotated data.

Converts the folders with run numbers according to the translation key file
obtained from .image_anonymizer and copies the images accordingly.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping


HUMAN_ANNOTATIONS_CSV = "human_annotations.csv"
SCORING_SHEET_CSV = "Expert_Scoring_Sheet_CDC-PRA.csv"
EXPERIMENTAL_READOUT_CSV = "expert_scoring_long.csv"

FOLDER_COLUMN_ALIASES = {
    "folder",
    "folder_name",
    "directory",
    "directory_name",
    "run",
    "run_name",
}

IMAGE_COLUMN_ALIASES = {
    "image",
    "image_name",
    "filename",
    "file_name",
}

FILE_MAPPING_REQUIRED_COLUMNS = {
    "original_relative_path",
    "anonymous_relative_path",
}


class HumanAnnotationsAnonymizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranslatedCSVResult:
    source: Path
    output: Path
    delimiter: str
    row_count: int
    translated_folder_cells: int
    translated_image_cells: int


@dataclass(frozen=True)
class CopiedImageResult:
    source: Path
    output: Path
    sha256: str


@dataclass
class HumanAnnotationDatasetsReport:
    input_root: Path
    human_output_root: Path
    experimental_output_root: Path
    translation_file: Path
    anonymized_images_root: Path
    human_csv_files: list[TranslatedCSVResult] = field(default_factory=list)
    experimental_csv_files: list[TranslatedCSVResult] = field(default_factory=list)
    human_images: list[CopiedImageResult] = field(default_factory=list)
    experimental_images: list[CopiedImageResult] = field(default_factory=list)
    excluded_source_files: list[Path] = field(default_factory=list)

    @property
    def csv_count(self) -> int:
        return len(self.human_csv_files) + len(self.experimental_csv_files)

    @property
    def image_count(self) -> int:
        return len(self.human_images) + len(self.experimental_images)

    @property
    def human_image_count(self) -> int:
        return len(self.human_images)

    @property
    def experimental_image_count(self) -> int:
        return len(self.experimental_images)


@dataclass(frozen=True)
class _FileMapping:
    original_relative_path: str
    anonymous_relative_path: str
    original_folder: str
    anonymous_folder: str
    original_filename: str
    anonymous_filename: str
    output_sha256: str | None


@dataclass(frozen=True)
class _CSVFormat:
    write_bom: bool
    delimiter: str
    quotechar: str


class _ProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        callback: Callable[[str], None] | None,
    ) -> None:
        self.enabled = enabled
        self.callback = callback or (lambda message: print(message, flush=True))
        self.started = time.perf_counter()

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        self.callback(f"[annotation-export +{elapsed:7.1f}s] {message}")


def _normalise_header(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _normalise_relative_path(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text.strip("/")
    if not text or text == ".":
        return "."

    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise HumanAnnotationsAnonymizationError(
            f"Unsafe relative path encountered: {value!r}."
        )
    return path.as_posix()


def _path_key(value: str) -> str:
    return _normalise_relative_path(value).casefold()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    candidates = (
        ("utf-8-sig", raw.startswith(b"\xef\xbb\xbf")),
        ("utf-8", False),
        ("cp1252", False),
        ("latin-1", False),
    )
    for encoding, write_bom in candidates:
        try:
            return raw.decode(encoding), write_bom
        except UnicodeDecodeError:
            continue
    raise HumanAnnotationsAnonymizationError(
        f"Could not decode CSV file '{path}'."
    )


def _detect_csv_format(path: Path) -> tuple[str, _CSVFormat]:
    text, write_bom = _read_text(path)
    sample = text[:65536]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar or '"'
    except csv.Error:
        counts = {
            delimiter: sample.count(delimiter)
            for delimiter in (",", ";", "\t", "|")
        }
        delimiter = max(counts, key=counts.get)
        quotechar = '"'

    return text, _CSVFormat(
        write_bom=write_bom,
        delimiter=delimiter,
        quotechar=quotechar,
    )


def _merge_sibling_folder_mapping(
    file_mapping_path: Path,
    anonymous_folder_by_original_folder: dict[str, str],
) -> None:
    folder_mapping_path = file_mapping_path.with_name(
        "ext_images_folder_mapping.csv"
    )
    if not folder_mapping_path.is_file():
        return

    text, csv_format = _detect_csv_format(folder_mapping_path)
    reader = csv.DictReader(
        io.StringIO(text, newline=""),
        delimiter=csv_format.delimiter,
        quotechar=csv_format.quotechar,
    )
    if reader.fieldnames is None:
        raise HumanAnnotationsAnonymizationError(
            f"Folder translation file has no header: {folder_mapping_path}"
        )

    normalised_fields = {
        _normalise_header(field): field for field in reader.fieldnames
    }
    original_field = normalised_fields.get("original_relative_path")
    anonymous_field = normalised_fields.get("anonymous_relative_path")
    if original_field is None or anonymous_field is None:
        raise HumanAnnotationsAnonymizationError(
            f"Folder translation file has an unexpected schema: "
            f"{folder_mapping_path}"
        )

    for line_number, row in enumerate(reader, start=2):
        original_folder = _normalise_relative_path(
            row.get(original_field, "")
        )
        anonymous_folder = _normalise_relative_path(
            row.get(anonymous_field, "")
        )
        if original_folder == "." or anonymous_folder == ".":
            continue

        folder_key = _path_key(original_folder)
        existing = anonymous_folder_by_original_folder.get(folder_key)
        if existing is not None and _path_key(existing) != _path_key(
            anonymous_folder
        ):
            raise HumanAnnotationsAnonymizationError(
                f"Conflicting folder translation for '{original_folder}' "
                f"at line {line_number} of '{folder_mapping_path}'."
            )
        anonymous_folder_by_original_folder[folder_key] = anonymous_folder


def _load_translation_file(
    path: Path,
) -> tuple[
    dict[str, _FileMapping],
    dict[str, str],
    dict[str, list[_FileMapping]],
]:
    if not path.is_file():
        raise FileNotFoundError(f"Translation file does not exist: {path}")

    text, csv_format = _detect_csv_format(path)
    reader = csv.DictReader(
        io.StringIO(text, newline=""),
        delimiter=csv_format.delimiter,
        quotechar=csv_format.quotechar,
    )
    if reader.fieldnames is None:
        raise HumanAnnotationsAnonymizationError(
            f"Translation file has no header: {path}"
        )

    normalised_fields = {
        _normalise_header(field): field for field in reader.fieldnames
    }
    missing = FILE_MAPPING_REQUIRED_COLUMNS - set(normalised_fields)
    if missing:
        raise HumanAnnotationsAnonymizationError(
            "The translation file must be ext_images_file_mapping.csv and "
            f"contain {sorted(FILE_MAPPING_REQUIRED_COLUMNS)}. "
            f"Missing: {sorted(missing)}."
        )

    original_relative_field = normalised_fields["original_relative_path"]
    anonymous_relative_field = normalised_fields["anonymous_relative_path"]
    output_hash_field = normalised_fields.get("output_sha256")

    file_by_original_path: dict[str, _FileMapping] = {}
    anonymous_folder_by_original_folder: dict[str, str] = {}
    files_by_original_filename: dict[str, list[_FileMapping]] = {}

    for line_number, row in enumerate(reader, start=2):
        original_relative = _normalise_relative_path(
            row.get(original_relative_field, "")
        )
        anonymous_relative = _normalise_relative_path(
            row.get(anonymous_relative_field, "")
        )
        if original_relative == "." or anonymous_relative == ".":
            raise HumanAnnotationsAnonymizationError(
                f"Invalid empty mapping path in '{path}' at line "
                f"{line_number}."
            )

        original_path = PurePosixPath(original_relative)
        anonymous_path = PurePosixPath(anonymous_relative)
        original_folder = original_path.parent.as_posix()
        anonymous_folder = anonymous_path.parent.as_posix()
        output_hash = (
            row.get(output_hash_field, "").strip()
            if output_hash_field
            else ""
        ) or None

        mapping = _FileMapping(
            original_relative_path=original_relative,
            anonymous_relative_path=anonymous_relative,
            original_folder=original_folder,
            anonymous_folder=anonymous_folder,
            original_filename=original_path.name,
            anonymous_filename=anonymous_path.name,
            output_sha256=output_hash,
        )

        file_key = _path_key(original_relative)
        existing = file_by_original_path.get(file_key)
        if existing is not None and existing != mapping:
            raise HumanAnnotationsAnonymizationError(
                f"Conflicting translation entries for "
                f"'{original_relative}'."
            )
        file_by_original_path[file_key] = mapping

        folder_key = _path_key(original_folder)
        existing_folder = anonymous_folder_by_original_folder.get(folder_key)
        if existing_folder is not None and _path_key(
            existing_folder
        ) != _path_key(anonymous_folder):
            raise HumanAnnotationsAnonymizationError(
                f"Original folder '{original_folder}' maps to multiple "
                "anonymous folders."
            )
        anonymous_folder_by_original_folder[folder_key] = anonymous_folder

        files_by_original_filename.setdefault(
            original_path.name.casefold(), []
        ).append(mapping)

    if not file_by_original_path:
        raise HumanAnnotationsAnonymizationError(
            f"Translation file contains no image mappings: {path}"
        )

    _merge_sibling_folder_mapping(
        path,
        anonymous_folder_by_original_folder,
    )
    return (
        file_by_original_path,
        anonymous_folder_by_original_folder,
        files_by_original_filename,
    )


def _find_matching_column(
    fieldnames: Iterable[str],
    aliases: set[str],
) -> str | None:
    matches = [
        field
        for field in fieldnames
        if _normalise_header(field) in aliases
    ]
    if len(matches) > 1:
        raise HumanAnnotationsAnonymizationError(
            f"CSV contains multiple matching columns for "
            f"{sorted(aliases)}: {matches}."
        )
    return matches[0] if matches else None


def _resolve_image_mapping(
    *,
    folder_value: str | None,
    image_value: str,
    file_by_original_path: Mapping[str, _FileMapping],
    files_by_original_filename: Mapping[str, list[_FileMapping]],
    csv_path: Path,
    row_number: int,
) -> _FileMapping:
    image_text = _normalise_relative_path(image_value)

    candidate_paths: list[str] = []
    if folder_value and str(folder_value).strip():
        folder_text = _normalise_relative_path(folder_value)
        if image_text.casefold().startswith(
            folder_text.casefold() + "/"
        ):
            candidate_paths.append(image_text)
        else:
            candidate_paths.append(
                (PurePosixPath(folder_text) / image_text).as_posix()
            )
    candidate_paths.append(image_text)

    for candidate in candidate_paths:
        mapping = file_by_original_path.get(_path_key(candidate))
        if mapping is not None:
            return mapping

    filename_matches = files_by_original_filename.get(
        PurePosixPath(image_text).name.casefold(), []
    )
    if len(filename_matches) == 1:
        return filename_matches[0]
    if len(filename_matches) > 1:
        raise HumanAnnotationsAnonymizationError(
            f"Ambiguous image '{image_value}' in '{csv_path}' row "
            f"{row_number}; the filename occurs in multiple folders."
        )

    attempted = ", ".join(repr(candidate) for candidate in candidate_paths)
    raise HumanAnnotationsAnonymizationError(
        f"No image translation found for '{image_value}' in "
        f"'{csv_path}' row {row_number}. Attempted paths: {attempted}."
    )


def _translate_csv(
    *,
    source: Path,
    output: Path,
    file_by_original_path: Mapping[str, _FileMapping],
    anonymous_folder_by_original_folder: Mapping[str, str],
    files_by_original_filename: Mapping[str, list[_FileMapping]],
    referenced_images: dict[str, _FileMapping] | None,
    require_image_column: bool,
) -> TranslatedCSVResult:
    text, csv_format = _detect_csv_format(source)
    reader = csv.DictReader(
        io.StringIO(text, newline=""),
        delimiter=csv_format.delimiter,
        quotechar=csv_format.quotechar,
    )
    if reader.fieldnames is None:
        raise HumanAnnotationsAnonymizationError(f"CSV has no header: {source}")

    folder_column = _find_matching_column(
        reader.fieldnames,
        FOLDER_COLUMN_ALIASES,
    )
    image_column = _find_matching_column(
        reader.fieldnames,
        IMAGE_COLUMN_ALIASES,
    )
    if folder_column is None:
        raise HumanAnnotationsAnonymizationError(
            f"Required folder column was not found in '{source}'."
        )
    if require_image_column and image_column is None:
        raise HumanAnnotationsAnonymizationError(
            f"Required image-name column was not found in '{source}'."
        )

    rows: list[dict[str, str]] = []
    translated_folder_cells = 0
    translated_image_cells = 0

    for row_number, row in enumerate(reader, start=2):
        translated = dict(row)
        folder_value = row.get(folder_column, "")
        image_value = row.get(image_column, "") if image_column else ""

        if image_column and image_value and image_value.strip():
            mapping = _resolve_image_mapping(
                folder_value=folder_value,
                image_value=image_value,
                file_by_original_path=file_by_original_path,
                files_by_original_filename=files_by_original_filename,
                csv_path=source,
                row_number=row_number,
            )
            translated[folder_column] = mapping.anonymous_folder
            translated[image_column] = mapping.anonymous_filename
            translated_folder_cells += 1
            translated_image_cells += 1
            if referenced_images is not None:
                referenced_images[
                    _path_key(mapping.anonymous_relative_path)
                ] = mapping
        elif folder_value and folder_value.strip():
            folder_key = _path_key(folder_value)
            anonymous_folder = anonymous_folder_by_original_folder.get(
                folder_key
            )
            if anonymous_folder is None:
                raise HumanAnnotationsAnonymizationError(
                    f"No folder translation found for '{folder_value}' in "
                    f"'{source}' row {row_number}."
                )
            translated[folder_column] = anonymous_folder
            translated_folder_cells += 1
        else:
            raise HumanAnnotationsAnonymizationError(
                f"Empty folder value in '{source}' row {row_number}."
            )

        rows.append(translated)

    output.parent.mkdir(parents=True, exist_ok=True)
    write_encoding = "utf-8-sig" if csv_format.write_bom else "utf-8"
    with output.open("w", encoding=write_encoding, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=reader.fieldnames,
            delimiter=csv_format.delimiter,
            quotechar=csv_format.quotechar,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return TranslatedCSVResult(
        source=source,
        output=output,
        delimiter=csv_format.delimiter,
        row_count=len(rows),
        translated_folder_cells=translated_folder_cells,
        translated_image_cells=translated_image_cells,
    )


def _find_unique_required_csv(input_root: Path, filename: str) -> Path:
    matches = sorted(
        (
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.name.casefold() == filename.casefold()
        ),
        key=lambda path: path.relative_to(input_root).as_posix().casefold(),
    )
    if not matches:
        raise FileNotFoundError(
            f"Required CSV '{filename}' was not found below '{input_root}'."
        )
    if len(matches) > 1:
        locations = ", ".join(
            path.relative_to(input_root).as_posix() for path in matches
        )
        raise HumanAnnotationsAnonymizationError(
            f"Required CSV '{filename}' exists more than once: {locations}."
        )
    return matches[0]


def _assert_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HumanAnnotationsAnonymizationError(
                f"Symbolic links are not accepted: {path}"
            )


def _validate_roots(
    *,
    input_root: Path,
    human_output_root: Path,
    experimental_output_root: Path,
    translation_file: Path,
    anonymized_images_root: Path,
) -> None:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_root}")
    if not translation_file.is_file():
        raise FileNotFoundError(
            f"Translation file does not exist: {translation_file}"
        )
    if not anonymized_images_root.is_dir():
        raise NotADirectoryError(
            f"Anonymized image folder does not exist: "
            f"{anonymized_images_root}"
        )
    if human_output_root == experimental_output_root:
        raise ValueError("The two output folders must be different.")

    for output_root in (human_output_root, experimental_output_root):
        if output_root == input_root:
            raise ValueError("Input and output folders must be different.")
        try:
            output_root.relative_to(input_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Output folder must not be inside the input folder: "
                f"{output_root}"
            )


def _copy_referenced_images(
    *,
    mappings: Mapping[str, _FileMapping],
    anonymized_images_root: Path,
    staging_root: Path,
    final_output_root: Path,
    progress: _ProgressReporter,
    progress_every: int,
) -> list[CopiedImageResult]:
    ordered = sorted(
        mappings.values(),
        key=lambda item: item.anonymous_relative_path.casefold(),
    )
    copied: list[CopiedImageResult] = []

    for index, mapping in enumerate(ordered, start=1):
        relative_path = Path(mapping.anonymous_relative_path)
        source = anonymized_images_root / relative_path
        destination = staging_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(
                f"Referenced anonymized image does not exist: {source}"
            )

        source_hash = _sha256(source)
        if mapping.output_sha256 and source_hash.casefold() != (
            mapping.output_sha256.casefold()
        ):
            raise HumanAnnotationsAnonymizationError(
                f"SHA-256 mismatch for anonymized image '{source}'."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination_hash = _sha256(destination)
        if destination_hash != source_hash:
            raise HumanAnnotationsAnonymizationError(
                f"Copied image differs from source: {source}"
            )

        copied.append(
            CopiedImageResult(
                source=source,
                output=final_output_root / relative_path,
                sha256=source_hash,
            )
        )
        if (
            index == 1
            or index == len(ordered)
            or index % progress_every == 0
        ):
            progress.log(
                f"[{index}/{len(ordered)}] Copied "
                f"{mapping.anonymous_relative_path}."
            )

    return copied


def _publish_two_outputs(
    *,
    human_staging_root: Path,
    human_output_root: Path,
    experimental_staging_root: Path,
    experimental_output_root: Path,
    overwrite: bool,
) -> None:
    output_pairs = (
        (human_staging_root, human_output_root),
        (experimental_staging_root, experimental_output_root),
    )
    backups: dict[Path, Path] = {}
    published: list[Path] = []

    try:
        for _, output_root in output_pairs:
            if output_root.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Output folder already exists: {output_root}. "
                        "Pass overwrite=True to replace it."
                    )
                backup = output_root.with_name(
                    f".{output_root.name}.backup-{uuid.uuid4().hex}"
                )
                os.replace(output_root, backup)
                backups[output_root] = backup

        for staging_root, output_root in output_pairs:
            os.replace(staging_root, output_root)
            published.append(output_root)

        for backup in backups.values():
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        for output_root in reversed(published):
            shutil.rmtree(output_root, ignore_errors=True)
        for output_root, backup in backups.items():
            if backup.exists():
                os.replace(backup, output_root)
        raise


def prepare_human_annotations(
    input_folder: str | os.PathLike[str] = "./human_annotations",
    human_output_folder: str | os.PathLike[str] = "../final/human_annotations",
    translation_file: str | os.PathLike[str] = "../final/private_mappings/ext_images_file_mapping.csv",
    *,
    experimental_output_folder: str | os.PathLike[str] = "../final/experimental_readout_images",
    anonymized_images_folder: str | os.PathLike[str] = "../final/ext_images",
    overwrite: bool = False,
    reject_symlinks: bool = True,
    verbose: bool = True,
    progress_every: int = 100,
    progress_callback: Callable[[str], None] | None = None,
) -> HumanAnnotationDatasetsReport:
    """Create the two final anonymized annotation datasets.

    Only these source CSV files are included:

    - ``human_annotations.csv`` and
      ``Expert_Scoring_Sheet_CDC-PRA.csv`` are written to
      ``human_output_folder``.
    - ``expert_scoring_long.csv`` is written to
      ``experimental_output_folder``.

    Images referenced by ``human_annotations.csv`` are copied from the already
    anonymized image tree into ``human_output_folder``. Images referenced by
    ``expert_scoring_long.csv`` are copied into
    ``experimental_output_folder``. No source folders, intermediate CSV files,
    or other source files are copied.
    """

    if progress_every < 1:
        raise ValueError("progress_every must be at least 1.")

    progress = _ProgressReporter(
        enabled=verbose,
        callback=progress_callback,
    )

    input_root = Path(input_folder).expanduser().resolve()
    human_output_root = Path(human_output_folder).expanduser().resolve()
    experimental_output_root = Path(
        experimental_output_folder
    ).expanduser().resolve()
    mapping_path = Path(translation_file).expanduser().resolve()
    anonymized_images_root = Path(
        anonymized_images_folder
    ).expanduser().resolve()

    progress.log(f"Input annotations: {input_root}")
    progress.log(f"Human output: {human_output_root}")
    progress.log(f"Experimental-readout output: {experimental_output_root}")
    progress.log(f"Translation file: {mapping_path}")
    progress.log(f"Anonymized images: {anonymized_images_root}")

    _validate_roots(
        input_root=input_root,
        human_output_root=human_output_root,
        experimental_output_root=experimental_output_root,
        translation_file=mapping_path,
        anonymized_images_root=anonymized_images_root,
    )
    if reject_symlinks:
        progress.log("Checking annotation input for symbolic links.")
        _assert_no_symlinks(input_root)

    if not overwrite:
        existing_outputs = [
            path
            for path in (human_output_root, experimental_output_root)
            if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "Output folder already exists: "
                + ", ".join(str(path) for path in existing_outputs)
                + ". Pass overwrite=True to replace it."
            )

    progress.log("Locating the three required CSV files.")
    human_annotations_csv = _find_unique_required_csv(
        input_root,
        HUMAN_ANNOTATIONS_CSV,
    )
    scoring_sheet_csv = _find_unique_required_csv(
        input_root,
        SCORING_SHEET_CSV,
    )
    experimental_readout_csv = _find_unique_required_csv(
        input_root,
        EXPERIMENTAL_READOUT_CSV,
    )
    selected_csvs = {
        human_annotations_csv.resolve(),
        scoring_sheet_csv.resolve(),
        experimental_readout_csv.resolve(),
    }
    all_source_files = {
        path.resolve()
        for path in input_root.rglob("*")
        if path.is_file()
    }
    excluded_source_files = sorted(
        all_source_files - selected_csvs,
        key=lambda path: path.relative_to(input_root).as_posix().casefold(),
    )
    progress.log(
        f"Selected exactly 3 CSV files; excluding "
        f"{len(excluded_source_files)} other source files."
    )

    progress.log("Loading image translation mappings.")
    (
        file_by_original_path,
        anonymous_folder_by_original_folder,
        files_by_original_filename,
    ) = _load_translation_file(mapping_path)
    progress.log(f"Loaded {len(file_by_original_path)} image translations.")

    human_output_root.parent.mkdir(parents=True, exist_ok=True)
    experimental_output_root.parent.mkdir(parents=True, exist_ok=True)
    human_staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{human_output_root.name}.preparing-",
            dir=human_output_root.parent,
        )
    )
    experimental_staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{experimental_output_root.name}.preparing-",
            dir=experimental_output_root.parent,
        )
    )

    report = HumanAnnotationDatasetsReport(
        input_root=input_root,
        human_output_root=human_output_root,
        experimental_output_root=experimental_output_root,
        translation_file=mapping_path,
        anonymized_images_root=anonymized_images_root,
        excluded_source_files=excluded_source_files,
    )
    human_referenced_images: dict[str, _FileMapping] = {}
    experimental_referenced_images: dict[str, _FileMapping] = {}

    try:
        progress.log(f"Translating {HUMAN_ANNOTATIONS_CSV}.")
        human_result = _translate_csv(
            source=human_annotations_csv,
            output=human_staging_root / HUMAN_ANNOTATIONS_CSV,
            file_by_original_path=file_by_original_path,
            anonymous_folder_by_original_folder=(
                anonymous_folder_by_original_folder
            ),
            files_by_original_filename=files_by_original_filename,
            referenced_images=human_referenced_images,
            require_image_column=True,
        )
        report.human_csv_files.append(
            TranslatedCSVResult(
                source=human_result.source,
                output=human_output_root / HUMAN_ANNOTATIONS_CSV,
                delimiter=human_result.delimiter,
                row_count=human_result.row_count,
                translated_folder_cells=(
                    human_result.translated_folder_cells
                ),
                translated_image_cells=(
                    human_result.translated_image_cells
                ),
            )
        )

        progress.log(f"Translating {SCORING_SHEET_CSV}.")
        scoring_result = _translate_csv(
            source=scoring_sheet_csv,
            output=human_staging_root / SCORING_SHEET_CSV,
            file_by_original_path=file_by_original_path,
            anonymous_folder_by_original_folder=(
                anonymous_folder_by_original_folder
            ),
            files_by_original_filename=files_by_original_filename,
            referenced_images=None,
            require_image_column=False,
        )
        report.human_csv_files.append(
            TranslatedCSVResult(
                source=scoring_result.source,
                output=human_output_root / SCORING_SHEET_CSV,
                delimiter=scoring_result.delimiter,
                row_count=scoring_result.row_count,
                translated_folder_cells=(
                    scoring_result.translated_folder_cells
                ),
                translated_image_cells=(
                    scoring_result.translated_image_cells
                ),
            )
        )

        progress.log(f"Translating {EXPERIMENTAL_READOUT_CSV}.")
        experimental_result = _translate_csv(
            source=experimental_readout_csv,
            output=(
                experimental_staging_root / EXPERIMENTAL_READOUT_CSV
            ),
            file_by_original_path=file_by_original_path,
            anonymous_folder_by_original_folder=(
                anonymous_folder_by_original_folder
            ),
            files_by_original_filename=files_by_original_filename,
            referenced_images=experimental_referenced_images,
            require_image_column=True,
        )
        report.experimental_csv_files.append(
            TranslatedCSVResult(
                source=experimental_result.source,
                output=(
                    experimental_output_root / EXPERIMENTAL_READOUT_CSV
                ),
                delimiter=experimental_result.delimiter,
                row_count=experimental_result.row_count,
                translated_folder_cells=(
                    experimental_result.translated_folder_cells
                ),
                translated_image_cells=(
                    experimental_result.translated_image_cells
                ),
            )
        )

        progress.log(
            f"Copying {len(human_referenced_images)} images referenced by "
            f"{HUMAN_ANNOTATIONS_CSV}."
        )
        report.human_images = _copy_referenced_images(
            mappings=human_referenced_images,
            anonymized_images_root=anonymized_images_root,
            staging_root=human_staging_root,
            final_output_root=human_output_root,
            progress=progress,
            progress_every=progress_every,
        )

        progress.log(
            f"Copying {len(experimental_referenced_images)} images "
            f"referenced by {EXPERIMENTAL_READOUT_CSV}."
        )
        report.experimental_images = _copy_referenced_images(
            mappings=experimental_referenced_images,
            anonymized_images_root=anonymized_images_root,
            staging_root=experimental_staging_root,
            final_output_root=experimental_output_root,
            progress=progress,
            progress_every=progress_every,
        )

        progress.log("Publishing both completed output trees.")
        _publish_two_outputs(
            human_staging_root=human_staging_root,
            human_output_root=human_output_root,
            experimental_staging_root=experimental_staging_root,
            experimental_output_root=experimental_output_root,
            overwrite=overwrite,
        )
        progress.log(
            f"FINISHED: {len(report.human_csv_files)} CSV files and "
            f"{report.human_image_count} images in human_annotations; "
            f"{len(report.experimental_csv_files)} CSV file and "
            f"{report.experimental_image_count} images in "
            "experimental_readout_images."
        )
        return report
    except Exception as exc:
        progress.log(
            f"FAILED: {type(exc).__name__}: {exc}. Removing staging output."
        )
        shutil.rmtree(human_staging_root, ignore_errors=True)
        shutil.rmtree(experimental_staging_root, ignore_errors=True)
        raise


def prepare_annotation_datasets(*args, **kwargs) -> HumanAnnotationDatasetsReport:
    return prepare_human_annotations(*args, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate the three approved annotation CSV files and split "
            "their referenced anonymized images into human_annotations and "
            "experimental_readout_images."
        )
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default="./human_annotations",
    )
    parser.add_argument(
        "human_output_folder",
        nargs="?",
        default="../final/human_annotations",
    )
    parser.add_argument(
        "translation_file",
        nargs="?",
        default=(
            "../final/private_mappings/ext_images_file_mapping.csv"
        ),
    )
    parser.add_argument(
        "--experimental-output-folder",
        default="../final/experimental_readout_images",
    )
    parser.add_argument(
        "--anonymized-images-folder",
        default="../final/ext_images",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-symlinks", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = prepare_human_annotations(
        input_folder=args.input_folder,
        human_output_folder=args.human_output_folder,
        translation_file=args.translation_file,
        experimental_output_folder=args.experimental_output_folder,
        anonymized_images_folder=args.anonymized_images_folder,
        overwrite=args.overwrite,
        reject_symlinks=not args.allow_symlinks,
        verbose=not args.quiet,
        progress_every=args.progress_every,
    )
    print(f"Human CSV files: {len(report.human_csv_files)}")
    print(f"Human images: {report.human_image_count}")
    print(
        f"Experimental CSV files: {len(report.experimental_csv_files)}"
    )
    print(f"Experimental images: {report.experimental_image_count}")
    print(f"Human output: {report.human_output_root}")
    print(f"Experimental output: {report.experimental_output_root}")


if __name__ == "__main__":
    main()

