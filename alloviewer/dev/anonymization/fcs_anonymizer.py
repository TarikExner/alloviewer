"""Anonymize FCS directory trees and keep linked files consistent.

Requires FlowIO 1.3.0.

For each FCS file, the module:

* assigns a stable anonymous filename such as ``FCS_000001.fcs``;
* rewrites identifying FCS TEXT metadata;
* sets ``$FIL`` to the new anonymous filename;
* verifies that event values and technical channel metadata are unchanged;
* updates ``file_name`` columns in CSV files;
* updates FlowJo ``.wsp`` and ``.wspt`` file references and copied keywords;
* writes a private filename mapping CSV in the output root.

Directory names and non-FCS filenames are retained. CSV files without a
``file_name`` column and unrelated non-FCS files are copied unchanged.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote, unquote

import flowio
import numpy as np


REDACTED = "REDACTED"
REQUIRED_FLOWIO_VERSION = "1.3.0"
DEFAULT_IDENTIFIER_PREFIX = "FCS"
DEFAULT_IDENTIFIER_WIDTH = 6
DEFAULT_MAPPING_FILENAME = "_PRIVATE_fcs_filename_mapping.csv"
WORKSPACE_SUFFIXES = {".wsp", ".wspt"}


class FCSAnonymizationError(RuntimeError):
    """Raised when a file cannot be anonymized or checked safely."""


@dataclass(frozen=True)
class FCSFilenameMapping:
    """Filename mapping for one FCS file."""

    source: Path
    original_relative_path: Path
    original_filename: str
    anonymous_id: str
    anonymous_filename: str
    anonymous_relative_path: Path
    output: Path


@dataclass(frozen=True)
class FCSFileResult:
    """Result for one rewritten FCS file."""

    source: Path
    output: Path
    original_filename: str
    anonymous_filename: str
    event_count: int
    channel_count: int
    redacted_metadata_keys: tuple[str, ...]
    maximum_absolute_event_difference: float


@dataclass(frozen=True)
class CSVRewriteResult:
    """Result for one CSV file containing a file_name column."""

    source: Path
    output: Path
    replaced_rows: int


@dataclass(frozen=True)
class WorkspaceRewriteResult:
    """Result for one FlowJo workspace."""

    source: Path
    output: Path
    replaced_dataset_uris: int
    replaced_filename_references: int
    redacted_keyword_values: int


@dataclass
class FCSAnonymizationReport:
    """Summary returned by :func:`anonymize_fcs_tree`."""

    input_root: Path
    output_root: Path
    mapping_file: Path
    filename_mappings: list[FCSFilenameMapping] = field(default_factory=list)
    fcs_files: list[FCSFileResult] = field(default_factory=list)
    csv_files: list[CSVRewriteResult] = field(default_factory=list)
    workspace_files: list[WorkspaceRewriteResult] = field(default_factory=list)
    copied_files: list[Path] = field(default_factory=list)
    created_directories: int = 0

    @property
    def fcs_file_count(self) -> int:
        return len(self.fcs_files)

    @property
    def rewritten_csv_count(self) -> int:
        return len(self.csv_files)

    @property
    def rewritten_workspace_count(self) -> int:
        return len(self.workspace_files)

    @property
    def copied_file_count(self) -> int:
        return len(self.copied_files)


# FlowIO regenerates these required structural keywords.
_GENERATED_TEXT_KEYS = {
    "beginanalysis",
    "begindata",
    "beginstext",
    "byteord",
    "datatype",
    "endanalysis",
    "enddata",
    "endstext",
    "mode",
    "nextdata",
    "par",
    "tot",
}

# FlowIO regenerates these parameter keywords from channel definitions.
_GENERATED_PARAMETER_KEY = re.compile(r"^p\d+[begnrs]$", re.IGNORECASE)

# Keys that can identify a person, sample, run, project, site, or instrument.
_EXACT_SENSITIVE_KEYS = {
    "assay id",
    "btim",
    "cells",
    "com",
    "comment",
    "comments",
    "cst bead lot expiration date",
    "cst bead lot id",
    "cytometer configuration date created",
    "cytometer configuration date modified",
    "cytometer configuration name",
    "cytsn",
    "date",
    "department",
    "description",
    "etim",
    "exp",
    "export time",
    "export user name",
    "fil",
    "guid",
    "inst",
    "last modified",
    "last modifier",
    "last_modified",
    "last_modifier",
    "note",
    "notes",
    "op",
    "operator",
    "patient",
    "patient id",
    "patient name",
    "plateid",
    "platename",
    "proj",
    "project",
    "sample",
    "sample id",
    "sample name",
    "smno",
    "source",
    "specimen",
    "specimen id",
    "specimen name",
    "src",
    "subject",
    "subject id",
    "subject name",
    "tube settings id",
    "tube settings name",
    "wellid",
}

# Broad matches such as every key containing "name" or "id" are avoided
# because that would alter technical fields such as laser1name.
_SENSITIVE_KEY_PATTERNS = (
    re.compile(r"(?:^|[\s_\-])(patient|subject|donor|recipient)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])(sample|specimen)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])(project|study)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])(operator|user|username|owner|author)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])(institution|institute|laboratory|department|site)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])(guid|uuid|serial)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])(date|time|timestamp)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])(comment|message|note|description)(?:$|[\s_\-])"),
    re.compile(r"(?:^|[\s_\-])qcmessage\d*(?:$|[\s_\-])"),
)

_XML_TAG_RE = re.compile(r"<(?P<name>DataSet|Keyword)\b[^>]*>", re.IGNORECASE)
_XML_ATTRIBUTE_TEMPLATE = r"(?P<prefix>\b{name}\s*=\s*)(?P<quote>['\"])(?P<value>.*?)(?P=quote)"
_NON_AUTOSAVE_RE = re.compile(
    _XML_ATTRIBUTE_TEMPLATE.format(name="nonAutoSaveFileName"),
    re.IGNORECASE,
)


def _require_flowio_130() -> None:
    version = str(getattr(flowio, "__version__", "unknown"))
    if version != REQUIRED_FLOWIO_VERSION:
        raise RuntimeError(
            f"fcs_anonymizer.py requires FlowIO {REQUIRED_FLOWIO_VERSION}; "
            f"installed version is {version}. Install it with "
            f"'pip install flowio=={REQUIRED_FLOWIO_VERSION}'."
        )


def _normalise_metadata_key(key: object) -> str:
    return " ".join(
        str(key).strip().lstrip("$").casefold().replace("_", " ").split()
    )


def _normalised_text(flow_data: flowio.FlowData) -> dict[str, str]:
    return {
        _normalise_metadata_key(key): str(value)
        for key, value in flow_data.text.items()
    }


def _normalised_additional_keys(
    additional_sensitive_keys: Iterable[str] | None,
) -> frozenset[str]:
    if additional_sensitive_keys is None:
        return frozenset()
    return frozenset(
        _normalise_metadata_key(item) for item in additional_sensitive_keys
    )


def _is_sensitive_key(
    key: object,
    additional_sensitive_keys: Iterable[str] | None,
) -> bool:
    normalised = _normalise_metadata_key(key)

    if normalised in _EXACT_SENSITIVE_KEYS:
        return True

    if normalised in _normalised_additional_keys(additional_sensitive_keys):
        return True

    return any(pattern.search(normalised) for pattern in _SENSITIVE_KEY_PATTERNS)


def _is_generated_key(key: object) -> bool:
    compact = _normalise_metadata_key(key).replace(" ", "")
    return compact in _GENERATED_TEXT_KEYS or bool(
        _GENERATED_PARAMETER_KEY.fullmatch(compact)
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_roots(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_root}")
    if input_root == output_root:
        raise ValueError("Input and output folders must be different.")
    if _is_relative_to(output_root, input_root):
        raise ValueError("The output folder must not be inside the input folder.")
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(
            f"The output path exists but is not a folder: {output_root}"
        )
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output folder already exists: {output_root}. "
            "Pass overwrite=True to replace it."
        )


def _validate_mapping_filename(mapping_filename: str | Path) -> Path:
    mapping_path = Path(mapping_filename)
    if mapping_path.is_absolute() or ".." in mapping_path.parts:
        raise ValueError(
            "mapping_filename must be a relative path inside the output folder."
        )
    if mapping_path.suffix.casefold() != ".csv":
        raise ValueError("mapping_filename must end in .csv.")
    return mapping_path


def _load_flow_data(
    path: Path,
    *,
    ignore_offset_error: bool,
    ignore_offset_discrepancy: bool,
    use_header_offsets: bool,
) -> flowio.FlowData:
    try:
        return flowio.FlowData(
            os.fspath(path),
            ignore_offset_error=ignore_offset_error,
            ignore_offset_discrepancy=ignore_offset_discrepancy,
            use_header_offsets=use_header_offsets,
        )
    except Exception as exc:
        raise FCSAnonymizationError(
            f"Could not read FCS file '{path}': {exc}"
        ) from exc


def _channel_labels(flow_data: flowio.FlowData) -> tuple[list[str], list[str]]:
    """Extract PnN and PnS labels using the FlowIO 1.3.0 channel API."""

    pnn_labels = [""] * int(flow_data.channel_count)
    pns_labels = [""] * int(flow_data.channel_count)

    for channel_number, channel in flow_data.channels.items():
        index = int(channel_number) - 1
        pnn_labels[index] = str(channel["PnN"])
        if "PnS" in channel:
            pns_labels[index] = str(channel["PnS"])

    if any(not label for label in pnn_labels):
        raise FCSAnonymizationError(
            "At least one required PnN channel label is missing."
        )

    return pnn_labels, pns_labels


def _validate_source_format(flow_data: flowio.FlowData, source: Path) -> None:
    data_type = str(flow_data.text.get("datatype", "")).upper()
    if data_type != "F":
        raise FCSAnonymizationError(
            f"'{source}' uses FCS datatype {data_type!r}. FlowIO 1.3.0 only "
            "writes floating-point FCS files, so this module accepts only "
            "floating-point source files to avoid changing data semantics."
        )

    for number in range(1, int(flow_data.channel_count) + 1):
        pne = str(flow_data.text.get(f"p{number}e", "0,0"))
        try:
            decades, log_zero = (float(value) for value in pne.split(",", 1))
        except (TypeError, ValueError) as exc:
            raise FCSAnonymizationError(
                f"Invalid P{number}E value {pne!r} in '{source}'."
            ) from exc

        if decades != 0.0 or log_zero != 0.0:
            raise FCSAnonymizationError(
                f"'{source}' has non-linear P{number}E={pne!r}. FlowIO 1.3.0 "
                "normalizes PnE to '0,0' when writing floating-point files, "
                "so the file is rejected instead of silently changing scaling."
            )


def _build_output_metadata(
    source_data: flowio.FlowData,
    *,
    anonymous_filename: str,
    additional_sensitive_keys: Iterable[str] | None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    metadata: dict[str, str] = {}
    redacted_keys: list[str] = []

    for original_key, original_value in source_data.text.items():
        normalised = _normalise_metadata_key(original_key)

        if _is_generated_key(original_key):
            continue

        if normalised == "fil":
            metadata[str(original_key).lstrip("$")] = anonymous_filename
            continue

        if _is_sensitive_key(original_key, additional_sensitive_keys):
            metadata[str(original_key).lstrip("$")] = REDACTED
            redacted_keys.append(normalised)
            continue

        value = str(original_value)
        if value != "":
            metadata[str(original_key).lstrip("$")] = value

    # Always write FIL even if the source omitted it.
    metadata["FIL"] = anonymous_filename

    # PnG and PnR must be supplied to retain source scaling and ranges.
    for number in range(1, int(source_data.channel_count) + 1):
        metadata[f"P{number}G"] = str(
            source_data.text.get(f"p{number}g", "1.0")
        )
        metadata[f"P{number}R"] = str(
            source_data.text.get(f"p{number}r", "262144")
        )
        metadata[f"P{number}E"] = "0,0"

    return metadata, tuple(sorted(set(redacted_keys)))


def _write_fcs(
    source_data: flowio.FlowData,
    output_path: Path,
    metadata: dict[str, str],
) -> None:
    try:
        source_data.write_fcs(os.fspath(output_path), metadata=metadata)
    except Exception as exc:
        raise FCSAnonymizationError(
            f"Could not write anonymized FCS file '{output_path}': {exc}"
        ) from exc


def _verify_redaction(
    source_data: flowio.FlowData,
    output_data: flowio.FlowData,
    output_path: Path,
    *,
    anonymous_filename: str,
    additional_sensitive_keys: Iterable[str] | None,
) -> tuple[str, ...]:
    source_text = _normalised_text(source_data)
    output_text = _normalised_text(output_data)

    sensitive_keys = sorted(
        key
        for key in source_text
        if key != "fil"
        and _is_sensitive_key(key, additional_sensitive_keys)
        and not _is_generated_key(key)
    )

    failures = [
        f"{key!r} -> {output_text.get(key)!r}"
        for key in sensitive_keys
        if output_text.get(key) != REDACTED
    ]

    if output_text.get("fil") != anonymous_filename:
        failures.append(
            f"'fil' -> {output_text.get('fil')!r}; "
            f"expected {anonymous_filename!r}"
        )

    if failures:
        raise FCSAnonymizationError(
            f"'{output_path}' did not replace all expected metadata fields: "
            + ", ".join(failures)
        )

    if output_data.analysis:
        raise FCSAnonymizationError(
            f"'{output_path}' still contains an ANALYSIS segment."
        )

    return tuple(sensitive_keys)


def _verify_technical_metadata(
    source_data: flowio.FlowData,
    output_data: flowio.FlowData,
    *,
    additional_sensitive_keys: Iterable[str] | None,
) -> None:
    source_text = _normalised_text(source_data)
    output_text = _normalised_text(output_data)

    for key, source_value in source_text.items():
        if _is_generated_key(key):
            continue
        if _is_sensitive_key(key, additional_sensitive_keys):
            continue
        if source_value == "":
            continue

        output_value = output_text.get(key)
        if output_value != source_value:
            raise FCSAnonymizationError(
                f"Technical metadata field {key!r} changed: "
                f"{source_value!r} -> {output_value!r}."
            )


def _verify_event_data(
    source_data: flowio.FlowData,
    output_data: flowio.FlowData,
    source_path: Path,
) -> float:
    if int(output_data.event_count) != int(source_data.event_count):
        raise FCSAnonymizationError(
            f"Event count changed for '{source_path}': "
            f"{source_data.event_count} -> {output_data.event_count}."
        )

    if int(output_data.channel_count) != int(source_data.channel_count):
        raise FCSAnonymizationError(
            f"Channel count changed for '{source_path}': "
            f"{source_data.channel_count} -> {output_data.channel_count}."
        )

    source_pnn, source_pns = _channel_labels(source_data)
    output_pnn, output_pns = _channel_labels(output_data)

    if output_pnn != source_pnn:
        raise FCSAnonymizationError(
            f"PnN channel labels changed while rewriting '{source_path}'."
        )
    if output_pns != source_pns:
        raise FCSAnonymizationError(
            f"PnS channel labels changed while rewriting '{source_path}'."
        )

    source_events = np.asarray(source_data.events, dtype=np.float32)
    output_events = np.asarray(output_data.events, dtype=np.float32)

    if source_events.shape != output_events.shape:
        raise FCSAnonymizationError(
            f"Event data shape changed for '{source_path}': "
            f"{source_events.shape} -> {output_events.shape}."
        )

    if source_events.size == 0:
        return 0.0

    difference = np.abs(
        source_events.astype(np.float64) - output_events.astype(np.float64)
    )
    maximum_difference = float(np.nanmax(difference))

    if not np.allclose(
        source_events,
        output_events,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise FCSAnonymizationError(
            f"Event values changed while rewriting '{source_path}'. "
            f"Maximum absolute difference: {maximum_difference:g}."
        )

    return maximum_difference


def anonymize_fcs_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    anonymous_filename: str | None = None,
    verify: bool = True,
    additional_sensitive_keys: Iterable[str] | None = None,
    ignore_offset_error: bool = False,
    ignore_offset_discrepancy: bool = False,
    use_header_offsets: bool = False,
) -> FCSFileResult:
    """Rewrite one FCS file with identifying metadata values replaced."""

    _require_flowio_130()

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    expected_filename = anonymous_filename or output.name

    if source.suffix.casefold() != ".fcs":
        raise ValueError(f"Source file is not an .fcs file: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"FCS file does not exist: {source}")
    if source == output:
        raise ValueError("Source and output FCS paths must be different.")
    if Path(expected_filename).name != expected_filename:
        raise ValueError("anonymous_filename must be a filename, not a path.")
    if not expected_filename.casefold().endswith(".fcs"):
        raise ValueError("anonymous_filename must end in .fcs.")

    output.parent.mkdir(parents=True, exist_ok=True)

    source_data = _load_flow_data(
        source,
        ignore_offset_error=ignore_offset_error,
        ignore_offset_discrepancy=ignore_offset_discrepancy,
        use_header_offsets=use_header_offsets,
    )
    _validate_source_format(source_data, source)

    metadata, redacted_keys = _build_output_metadata(
        source_data,
        anonymous_filename=expected_filename,
        additional_sensitive_keys=additional_sensitive_keys,
    )

    temporary_output = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}.fcs"
    )

    output_data: flowio.FlowData | None = None
    maximum_difference = 0.0

    try:
        _write_fcs(source_data, temporary_output, metadata)
        output_data = _load_flow_data(
            temporary_output,
            ignore_offset_error=False,
            ignore_offset_discrepancy=False,
            use_header_offsets=False,
        )

        verified_redacted_keys = _verify_redaction(
            source_data,
            output_data,
            temporary_output,
            anonymous_filename=expected_filename,
            additional_sensitive_keys=additional_sensitive_keys,
        )

        if verify:
            _verify_technical_metadata(
                source_data,
                output_data,
                additional_sensitive_keys=additional_sensitive_keys,
            )
            maximum_difference = _verify_event_data(
                source_data,
                output_data,
                source,
            )

        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    assert output_data is not None
    return FCSFileResult(
        source=source,
        output=output,
        original_filename=source.name,
        anonymous_filename=expected_filename,
        event_count=int(source_data.event_count),
        channel_count=int(source_data.channel_count),
        redacted_metadata_keys=(
            verified_redacted_keys if verify else redacted_keys
        ),
        maximum_absolute_event_difference=maximum_difference,
    )


def _build_filename_mappings(
    input_root: Path,
    output_root: Path,
    *,
    identifier_prefix: str,
    identifier_width: int,
) -> list[FCSFilenameMapping]:
    if not identifier_prefix or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier_prefix):
        raise ValueError(
            "identifier_prefix must contain only letters, digits, '_' or '-'."
        )
    if identifier_width < 1:
        raise ValueError("identifier_width must be at least 1.")

    fcs_paths = sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".fcs"
    )

    seen_basenames: dict[str, Path] = {}
    mappings: list[FCSFilenameMapping] = []

    for index, source in enumerate(fcs_paths, start=1):
        basename_key = source.name.casefold()
        previous = seen_basenames.get(basename_key)
        if previous is not None:
            raise FCSAnonymizationError(
                "The same FCS filename occurs in more than one folder, so a "
                "root-level expression table containing only file_name would "
                "be ambiguous: "
                f"'{previous}' and '{source}'. Rename one source file first."
            )
        seen_basenames[basename_key] = source

        original_relative = source.relative_to(input_root)
        anonymous_id = f"{identifier_prefix}_{index:0{identifier_width}d}"
        anonymous_filename = f"{anonymous_id}.fcs"
        anonymous_relative = original_relative.with_name(anonymous_filename)

        mappings.append(
            FCSFilenameMapping(
                source=source,
                original_relative_path=original_relative,
                original_filename=source.name,
                anonymous_id=anonymous_id,
                anonymous_filename=anonymous_filename,
                anonymous_relative_path=anonymous_relative,
                output=output_root / anonymous_relative,
            )
        )

    return mappings


def _mapping_by_basename(
    mappings: Sequence[FCSFilenameMapping],
) -> dict[str, FCSFilenameMapping]:
    return {mapping.original_filename.casefold(): mapping for mapping in mappings}


def _basename_from_reference(value: str) -> str:
    decoded = unquote(html.unescape(value.strip()))
    return re.split(r"[/\\]", decoded)[-1]


def _detect_text_encoding(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(1024 * 1024)
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        try:
            raw.decode("cp1252")
            return "cp1252"
        except UnicodeDecodeError as exc:
            raise FCSAnonymizationError(
                f"Could not decode text file '{path}' as UTF-8 or cp1252."
            ) from exc


def _detect_csv_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        delimiter = ";" if sample.partition("\n")[0].count(";") > sample.partition("\n")[0].count(",") else ","

        class FallbackDialect(csv.Dialect):
            quoting = csv.QUOTE_MINIMAL
            quotechar = '"'
            doublequote = True
            escapechar = None
            lineterminator = "\n"
            skipinitialspace = False
            strict = True

        FallbackDialect.delimiter = delimiter
        return FallbackDialect()


def _line_terminator(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _rewrite_csv_file(
    source: Path,
    output: Path,
    *,
    mapping_by_name: Mapping[str, FCSFilenameMapping],
    strict_references: bool,
) -> CSVRewriteResult | None:
    encoding = _detect_text_encoding(source)
    with source.open("r", encoding=encoding, newline="") as handle:
        sample = handle.read(65536)
    dialect = _detect_csv_dialect(sample)

    with source.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, dialect=dialect)
        try:
            header = next(reader)
        except StopIteration:
            shutil.copyfile(source, output)
            return None

    matching_columns = [
        index
        for index, value in enumerate(header)
        if value.strip().casefold() == "file_name"
    ]
    if not matching_columns:
        shutil.copyfile(source, output)
        return None
    if len(matching_columns) != 1:
        raise FCSAnonymizationError(
            f"CSV file '{source}' contains more than one file_name column."
        )

    file_name_index = matching_columns[0]
    missing: set[str] = set()
    replaced_rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with (
            source.open("r", encoding=encoding, newline="") as input_handle,
            temporary_output.open("w", encoding=encoding, newline="") as output_handle,
        ):
            reader = csv.reader(input_handle, dialect=dialect)
            writer = csv.writer(
                output_handle,
                delimiter=dialect.delimiter,
                quotechar=dialect.quotechar,
                doublequote=dialect.doublequote,
                escapechar=dialect.escapechar,
                quoting=dialect.quoting,
                lineterminator=_line_terminator(sample),
            )

            writer.writerow(next(reader))
            for row_number, row in enumerate(reader, start=2):
                if file_name_index >= len(row):
                    raise FCSAnonymizationError(
                        f"CSV file '{source}' has too few columns on row "
                        f"{row_number}."
                    )

                value = row[file_name_index].strip()
                if value:
                    basename = _basename_from_reference(value)
                    mapping = mapping_by_name.get(basename.casefold())
                    if mapping is None:
                        if strict_references:
                            missing.add(value)
                    else:
                        row[file_name_index] = mapping.anonymous_filename
                        replaced_rows += 1

                writer.writerow(row)

        if missing:
            examples = ", ".join(
                repr(value) for value in sorted(missing)[:10]
            )
            raise FCSAnonymizationError(
                f"CSV file '{source}' contains file_name values with no "
                f"matching FCS file in the input tree: {examples}"
            )

        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    return CSVRewriteResult(
        source=source,
        output=output,
        replaced_rows=replaced_rows,
    )


def _get_xml_attribute(tag: str, name: str) -> str | None:
    pattern = re.compile(
        _XML_ATTRIBUTE_TEMPLATE.format(name=re.escape(name)),
        re.IGNORECASE,
    )
    match = pattern.search(tag)
    if match is None:
        return None
    return html.unescape(match.group("value"))


def _replace_xml_attribute(tag: str, name: str, value: str) -> str:
    pattern = re.compile(
        _XML_ATTRIBUTE_TEMPLATE.format(name=re.escape(name)),
        re.IGNORECASE,
    )
    escaped = html.escape(value, quote=True)

    def replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{match.group('quote')}{escaped}{match.group('quote')}"

    updated, count = pattern.subn(replace, tag, count=1)
    if count != 1:
        raise FCSAnonymizationError(
            f"Expected XML attribute {name!r} was not found in tag: {tag[:200]}"
        )
    return updated


def _relative_file_uri(workspace_relative_path: Path, target_relative_path: Path) -> str:
    workspace_parent = workspace_relative_path.parent
    relative = os.path.relpath(target_relative_path, start=workspace_parent)
    relative_posix = relative.replace(os.sep, "/")
    if not relative_posix.startswith("."):
        relative_posix = f"./{relative_posix}"
    return f"file:{quote(relative_posix, safe='/._-')}"


def _replace_known_filename_forms(
    text: str,
    mappings: Sequence[FCSFilenameMapping],
) -> tuple[str, int]:
    replacements = 0

    for mapping in sorted(
        mappings,
        key=lambda item: len(item.original_filename),
        reverse=True,
    ):
        original = mapping.original_filename
        anonymous = mapping.anonymous_filename
        forms = {
            original,
            html.escape(original, quote=True),
            quote(original, safe=""),
            quote(original, safe="-_."),
        }
        for old in sorted(forms, key=len, reverse=True):
            count = text.count(old)
            if count:
                text = text.replace(old, anonymous)
                replacements += count

    return text, replacements


def _rewrite_flowjo_workspace(
    source: Path,
    output: Path,
    *,
    workspace_relative_path: Path,
    mappings: Sequence[FCSFilenameMapping],
    mapping_by_name: Mapping[str, FCSFilenameMapping],
    additional_sensitive_keys: Iterable[str] | None,
    strict_references: bool,
) -> WorkspaceRewriteResult:
    encoding = _detect_text_encoding(source)
    text = source.read_text(encoding=encoding)

    missing: set[str] = set()
    dataset_uri_count = 0
    redacted_keyword_count = 0

    def rewrite_tag(match: re.Match[str]) -> str:
        nonlocal dataset_uri_count, redacted_keyword_count
        tag = match.group(0)
        tag_name = match.group("name").casefold()

        if tag_name == "dataset":
            uri = _get_xml_attribute(tag, "uri")
            if uri is None:
                return tag
            basename = _basename_from_reference(uri)
            mapping = mapping_by_name.get(basename.casefold())
            if mapping is None:
                if strict_references and basename.casefold().endswith(".fcs"):
                    missing.add(uri)
                return tag
            new_uri = _relative_file_uri(
                workspace_relative_path,
                mapping.anonymous_relative_path,
            )
            dataset_uri_count += 1
            return _replace_xml_attribute(tag, "uri", new_uri)

        keyword_name = _get_xml_attribute(tag, "name")
        keyword_value = _get_xml_attribute(tag, "value")
        if keyword_name is None or keyword_value is None:
            return tag

        normalised_key = _normalise_metadata_key(keyword_name)
        if normalised_key == "fil":
            basename = _basename_from_reference(keyword_value)
            mapping = mapping_by_name.get(basename.casefold())
            if mapping is None:
                if strict_references and basename.casefold().endswith(".fcs"):
                    missing.add(keyword_value)
                return tag
            return _replace_xml_attribute(
                tag,
                "value",
                mapping.anonymous_filename,
            )

        if _is_sensitive_key(keyword_name, additional_sensitive_keys):
            if keyword_value != REDACTED:
                redacted_keyword_count += 1
            return _replace_xml_attribute(tag, "value", REDACTED)

        return tag

    text = _XML_TAG_RE.sub(rewrite_tag, text)

    # This changes SampleNode names and any other plain or encoded filename
    # copies not covered by the DataSet and Keyword handlers above.
    text, filename_reference_count = _replace_known_filename_forms(text, mappings)

    # Avoid retaining the source workstation path in the workspace header.
    relative_workspace_uri = f"file:./{quote(output.name, safe='._-')}"
    text = _NON_AUTOSAVE_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"{relative_workspace_uri}{match.group('quote')}"
        ),
        text,
        count=1,
    )

    if missing:
        examples = ", ".join(repr(value) for value in sorted(missing)[:10])
        raise FCSAnonymizationError(
            f"FlowJo workspace '{source}' contains FCS references with no "
            f"matching file in the input tree: {examples}"
        )

    remaining: list[str] = []
    for mapping in mappings:
        if mapping.original_filename in text:
            remaining.append(mapping.original_filename)
        encoded = quote(mapping.original_filename, safe="")
        if encoded in text:
            remaining.append(encoded)
    if remaining:
        raise FCSAnonymizationError(
            f"FlowJo workspace '{source}' still contains original FCS "
            f"filename text after rewriting: {remaining[:10]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding=encoding, newline="")

    return WorkspaceRewriteResult(
        source=source,
        output=output,
        replaced_dataset_uris=dataset_uri_count,
        replaced_filename_references=filename_reference_count,
        redacted_keyword_values=redacted_keyword_count,
    )


def _write_mapping_csv(
    path: Path,
    mappings: Sequence[FCSFilenameMapping],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "anonymous_id",
                "anonymous_filename",
                "original_filename",
                "original_relative_path",
                "anonymous_relative_path",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for mapping in mappings:
            writer.writerow(
                {
                    "anonymous_id": mapping.anonymous_id,
                    "anonymous_filename": mapping.anonymous_filename,
                    "original_filename": mapping.original_filename,
                    "original_relative_path": mapping.original_relative_path.as_posix(),
                    "anonymous_relative_path": mapping.anonymous_relative_path.as_posix(),
                }
            )


def anonymize_fcs_tree(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    overwrite: bool = False,
    verify: bool = True,
    additional_sensitive_keys: Iterable[str] | None = None,
    ignore_offset_error: bool = False,
    ignore_offset_discrepancy: bool = False,
    use_header_offsets: bool = False,
    reject_symlinks: bool = True,
    strict_references: bool = True,
    identifier_prefix: str = DEFAULT_IDENTIFIER_PREFIX,
    identifier_width: int = DEFAULT_IDENTIFIER_WIDTH,
    mapping_filename: str | Path = DEFAULT_MAPPING_FILENAME,
) -> FCSAnonymizationReport:
    """Copy a tree while anonymizing FCS files and linked metadata files.

    FCS files receive unique names across the full input tree. Directory names
    are retained. CSV files with a ``file_name`` column and FlowJo workspaces
    are rewritten against the same filename map.
    """

    _require_flowio_130()

    input_root = Path(input_folder).expanduser().resolve()
    output_root = Path(output_folder).expanduser().resolve()
    mapping_relative_path = _validate_mapping_filename(mapping_filename)
    _validate_roots(input_root, output_root, overwrite)

    mappings = _build_filename_mappings(
        input_root,
        output_root,
        identifier_prefix=identifier_prefix,
        identifier_width=identifier_width,
    )
    mapping_by_source = {mapping.source: mapping for mapping in mappings}
    mapping_by_name = _mapping_by_basename(mappings)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / (
        f".{output_root.name}.staging.{uuid.uuid4().hex}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)

    report = FCSAnonymizationReport(
        input_root=input_root,
        output_root=output_root,
        mapping_file=output_root / mapping_relative_path,
        filename_mappings=list(mappings),
    )

    try:
        for source_path in sorted(input_root.rglob("*")):
            relative_path = source_path.relative_to(input_root)

            if source_path.is_symlink():
                if reject_symlinks:
                    raise FCSAnonymizationError(
                        f"Symbolic links are not permitted: {source_path}"
                    )
                continue

            if source_path.is_dir():
                (staging_root / relative_path).mkdir(parents=True, exist_ok=True)
                report.created_directories += 1
                continue

            if not source_path.is_file():
                continue

            if source_path.suffix.casefold() == ".fcs":
                mapping = mapping_by_source[source_path]
                staging_output = staging_root / mapping.anonymous_relative_path
                staging_output.parent.mkdir(parents=True, exist_ok=True)

                result = anonymize_fcs_file(
                    source_path,
                    staging_output,
                    anonymous_filename=mapping.anonymous_filename,
                    verify=verify,
                    additional_sensitive_keys=additional_sensitive_keys,
                    ignore_offset_error=ignore_offset_error,
                    ignore_offset_discrepancy=ignore_offset_discrepancy,
                    use_header_offsets=use_header_offsets,
                )
                report.fcs_files.append(
                    FCSFileResult(
                        source=result.source,
                        output=mapping.output,
                        original_filename=result.original_filename,
                        anonymous_filename=result.anonymous_filename,
                        event_count=result.event_count,
                        channel_count=result.channel_count,
                        redacted_metadata_keys=result.redacted_metadata_keys,
                        maximum_absolute_event_difference=(
                            result.maximum_absolute_event_difference
                        ),
                    )
                )
                continue

            staging_output = staging_root / relative_path
            staging_output.parent.mkdir(parents=True, exist_ok=True)
            suffix = source_path.suffix.casefold()

            if suffix == ".csv":
                csv_result = _rewrite_csv_file(
                    source_path,
                    staging_output,
                    mapping_by_name=mapping_by_name,
                    strict_references=strict_references,
                )
                if csv_result is None:
                    report.copied_files.append(output_root / relative_path)
                else:
                    report.csv_files.append(
                        CSVRewriteResult(
                            source=csv_result.source,
                            output=output_root / relative_path,
                            replaced_rows=csv_result.replaced_rows,
                        )
                    )
                continue

            if suffix in WORKSPACE_SUFFIXES:
                workspace_result = _rewrite_flowjo_workspace(
                    source_path,
                    staging_output,
                    workspace_relative_path=relative_path,
                    mappings=mappings,
                    mapping_by_name=mapping_by_name,
                    additional_sensitive_keys=additional_sensitive_keys,
                    strict_references=strict_references,
                )
                report.workspace_files.append(
                    WorkspaceRewriteResult(
                        source=workspace_result.source,
                        output=output_root / relative_path,
                        replaced_dataset_uris=(
                            workspace_result.replaced_dataset_uris
                        ),
                        replaced_filename_references=(
                            workspace_result.replaced_filename_references
                        ),
                        redacted_keyword_values=(
                            workspace_result.redacted_keyword_values
                        ),
                    )
                )
                continue

            shutil.copyfile(source_path, staging_output)
            report.copied_files.append(output_root / relative_path)

        _write_mapping_csv(staging_root / mapping_relative_path, mappings)

        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.replace(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a directory tree while anonymizing FCS metadata, assigning "
            "new FCS filenames, and updating CSV and FlowJo references. "
            "Requires FlowIO 1.3.0."
        )
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output folder if it already exists.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip technical metadata and event-value checks.",
    )
    parser.add_argument(
        "--allow-symlinks",
        action="store_true",
        help="Ignore symbolic links instead of rejecting the input tree.",
    )
    parser.add_argument(
        "--allow-missing-references",
        action="store_true",
        help=(
            "Leave unknown file_name or workspace references unchanged instead "
            "of stopping with an error."
        ),
    )
    parser.add_argument(
        "--identifier-prefix",
        default=DEFAULT_IDENTIFIER_PREFIX,
        help=f"Anonymous filename prefix. Default: {DEFAULT_IDENTIFIER_PREFIX}",
    )
    parser.add_argument(
        "--identifier-width",
        type=int,
        default=DEFAULT_IDENTIFIER_WIDTH,
        help=f"Number width used in anonymous filenames. Default: {DEFAULT_IDENTIFIER_WIDTH}",
    )
    parser.add_argument(
        "--mapping-filename",
        default=DEFAULT_MAPPING_FILENAME,
        help=(
            "Relative CSV path written inside the output folder. Default: "
            f"{DEFAULT_MAPPING_FILENAME}"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = anonymize_fcs_tree(
        args.input_folder,
        args.output_folder,
        overwrite=args.overwrite,
        verify=not args.no_verify,
        reject_symlinks=not args.allow_symlinks,
        strict_references=not args.allow_missing_references,
        identifier_prefix=args.identifier_prefix,
        identifier_width=args.identifier_width,
        mapping_filename=args.mapping_filename,
    )
    print(
        f"Created '{report.output_root}' with "
        f"{report.fcs_file_count} anonymized FCS file(s), "
        f"{report.rewritten_csv_count} rewritten CSV file(s), "
        f"{report.rewritten_workspace_count} rewritten FlowJo workspace(s), "
        f"and {report.copied_file_count} copied unrelated file(s)."
    )
    print(f"Private filename mapping: {report.mapping_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

