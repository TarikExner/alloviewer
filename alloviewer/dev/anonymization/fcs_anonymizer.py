"""Anonymize directory trees containing FCS files using FlowIO 1.3.0.

The directory structure and filenames are preserved. Every ``.fcs`` file is
read with FlowIO 1.3.0 and rewritten as a new FCS file. Potentially identifying
TEXT metadata keys remain present, but their values are replaced with
``REDACTED``. Technical metadata, channel labels, spillover information, and
event values are retained.

Files that are not FCS files are copied byte-for-byte. Their contents and names
are not inspected or anonymized.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import flowio
import numpy as np


REDACTED = "REDACTED"
REQUIRED_FLOWIO_VERSION = "1.3.0"


class FCSAnonymizationError(RuntimeError):
    """Raised when an FCS file cannot be anonymized or verified safely."""


@dataclass(frozen=True)
class FCSFileResult:
    """Result for one rewritten FCS file."""

    source: Path
    output: Path
    event_count: int
    channel_count: int
    redacted_metadata_keys: tuple[str, ...]
    maximum_absolute_event_difference: float


@dataclass
class FCSAnonymizationReport:
    """Summary returned by :func:`anonymize_fcs_tree`."""

    input_root: Path
    output_root: Path
    fcs_files: list[FCSFileResult] = field(default_factory=list)
    copied_files: list[Path] = field(default_factory=list)
    created_directories: int = 0

    @property
    def fcs_file_count(self) -> int:
        return len(self.fcs_files)

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

# FlowIO regenerates these parameter keywords from the channel definitions.
_GENERATED_PARAMETER_KEY = re.compile(r"^p\d+[begnrs]$", re.IGNORECASE)

# Exact fields that can identify a person, sample, run, project, site, or
# individual instrument. Their keys are preserved and values are redacted.
_EXACT_SENSITIVE_KEYS = {
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

# Conservative patterns for vendor-specific metadata names. Broad matches such
# as any key containing "name" or "id" are intentionally avoided because they
# would redact technical fields such as laser1name.
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


def _is_sensitive_key(
    key: object,
    additional_sensitive_keys: Iterable[str] | None,
) -> bool:
    normalised = _normalise_metadata_key(key)

    if normalised in _EXACT_SENSITIVE_KEYS:
        return True

    if additional_sensitive_keys is not None:
        additional = {
            _normalise_metadata_key(item) for item in additional_sensitive_keys
        }
        if normalised in additional:
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


def _load_flow_data(
    path: Path,
    *,
    ignore_offset_error: bool,
    ignore_offset_discrepancy: bool,
    use_header_offsets: bool,
) -> flowio.FlowData:
    try:
        # FlowIO 1.3.0 expects a string path, not pathlib.Path.
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
                "normalizes PnE to '0,0' when writing floating-point files, so "
                "the file is rejected instead of silently changing scaling."
            )


def _build_output_metadata(
    source_data: flowio.FlowData,
    *,
    additional_sensitive_keys: Iterable[str] | None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    metadata: dict[str, str] = {}
    redacted_keys: list[str] = []

    for original_key, original_value in source_data.text.items():
        normalised = _normalise_metadata_key(original_key)

        if _is_generated_key(original_key):
            continue

        if _is_sensitive_key(original_key, additional_sensitive_keys):
            metadata[str(original_key).lstrip("$")] = REDACTED
            redacted_keys.append(normalised)
            continue

        value = str(original_value)
        if value != "":
            metadata[str(original_key).lstrip("$")] = value

    # PnG and PnR are generated by FlowIO but must be supplied to preserve the
    # source channel scaling and display ranges. PnE is known to be 0,0 from
    # _validate_source_format().
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
        # FlowIO 1.3.0 write_fcs expects a string filename.
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
    additional_sensitive_keys: Iterable[str] | None,
) -> tuple[str, ...]:
    source_text = _normalised_text(source_data)
    output_text = _normalised_text(output_data)

    sensitive_keys = sorted(
        key
        for key in source_text
        if _is_sensitive_key(key, additional_sensitive_keys)
        and not _is_generated_key(key)
    )

    failures = [
        f"{key!r} -> {output_text.get(key)!r}"
        for key in sensitive_keys
        if output_text.get(key) != REDACTED
    ]

    if "proj" in source_text and output_text.get("proj") != REDACTED:
        failures.append(f"'proj' -> {output_text.get('proj')!r}")

    if failures:
        raise FCSAnonymizationError(
            f"'{output_path}' did not redact all expected metadata fields: "
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
    verify: bool = True,
    additional_sensitive_keys: Iterable[str] | None = None,
    ignore_offset_error: bool = False,
    ignore_offset_discrepancy: bool = False,
    use_header_offsets: bool = False,
) -> FCSFileResult:
    """Rewrite one FCS file with identifying metadata values redacted."""

    _require_flowio_130()

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()

    if source.suffix.casefold() != ".fcs":
        raise ValueError(f"Source file is not an .fcs file: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"FCS file does not exist: {source}")
    if source == output:
        raise ValueError("Source and output FCS paths must be different.")

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
        event_count=int(source_data.event_count),
        channel_count=int(source_data.channel_count),
        redacted_metadata_keys=(
            verified_redacted_keys if verify else redacted_keys
        ),
        maximum_absolute_event_difference=maximum_difference,
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
) -> FCSAnonymizationReport:
    """Copy a directory tree and anonymize every FCS file in it.

    Directory names and filenames are preserved. Non-FCS files are copied
    byte-for-byte without preserving filesystem metadata.
    """

    _require_flowio_130()

    input_root = Path(input_folder).expanduser().resolve()
    output_root = Path(output_folder).expanduser().resolve()
    _validate_roots(input_root, output_root, overwrite)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / (
        f".{output_root.name}.staging.{uuid.uuid4().hex}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)

    report = FCSAnonymizationReport(
        input_root=input_root,
        output_root=output_root,
    )

    try:
        for source_path in sorted(input_root.rglob("*")):
            relative_path = source_path.relative_to(input_root)
            staging_output_path = staging_root / relative_path

            if source_path.is_symlink():
                if reject_symlinks:
                    raise FCSAnonymizationError(
                        f"Symbolic links are not permitted: {source_path}"
                    )
                continue

            if source_path.is_dir():
                staging_output_path.mkdir(parents=True, exist_ok=True)
                report.created_directories += 1
                continue

            if not source_path.is_file():
                continue

            staging_output_path.parent.mkdir(parents=True, exist_ok=True)

            if source_path.suffix.casefold() == ".fcs":
                result = anonymize_fcs_file(
                    source_path,
                    staging_output_path,
                    verify=verify,
                    additional_sensitive_keys=additional_sensitive_keys,
                    ignore_offset_error=ignore_offset_error,
                    ignore_offset_discrepancy=ignore_offset_discrepancy,
                    use_header_offsets=use_header_offsets,
                )
                report.fcs_files.append(
                    FCSFileResult(
                        source=result.source,
                        output=output_root / relative_path,
                        event_count=result.event_count,
                        channel_count=result.channel_count,
                        redacted_metadata_keys=result.redacted_metadata_keys,
                        maximum_absolute_event_difference=(
                            result.maximum_absolute_event_difference
                        ),
                    )
                )
            else:
                shutil.copyfile(source_path, staging_output_path)
                report.copied_files.append(output_root / relative_path)

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
            "Recursively copy a directory tree while redacting identifying "
            "metadata values in FCS files. Requires FlowIO 1.3.0."
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
        help="Skip technical metadata and event-value verification.",
    )
    parser.add_argument(
        "--allow-symlinks",
        action="store_true",
        help="Ignore symbolic links instead of rejecting the input tree.",
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
    )
    print(
        f"Created '{report.output_root}' with "
        f"{report.fcs_file_count} anonymized FCS file(s) and "
        f"{report.copied_file_count} copied non-FCS file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

