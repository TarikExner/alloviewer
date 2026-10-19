from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


ANNOTATION_SUFFIXES = (
    "_regions.npy",
    "_well_mask.npy",
    "_regions.json",
    "_preview.png",
)
REQUIRED_ANNOTATION_SUFFIXES = ANNOTATION_SUFFIXES[:3]


@dataclass(frozen=True)
class _ImageMapping:
    """
    Pair an original image path with its anonymous replacement.

    Attributes
    ----------
    original : PurePosixPath
        Original name or relative image path before anonymization.
    anonymous : PurePosixPath
        Replacement anonymous name or relative image path.
    """
    original: PurePosixPath
    anonymous: PurePosixPath

    @property
    def original_folder(self) -> PurePosixPath:
        """
        Return the original image's parent folder.

        Returns
        -------
        PurePosixPath
            The original image's parent folder.
        """
        return self.original.parent

    @property
    def anonymous_folder(self) -> PurePosixPath:
        """
        Return the anonymous image's parent folder.

        Returns
        -------
        PurePosixPath
            The anonymous image's parent folder.
        """
        return self.anonymous.parent

    @property
    def original_stem(self) -> str:
        """
        Return the original filename without its suffix.

        Returns
        -------
        str
            The original filename without its suffix.
        """
        return self.original.stem

    @property
    def anonymous_stem(self) -> str:
        """
        Return the anonymous filename without its suffix.

        Returns
        -------
        str
            The anonymous filename without its suffix.
        """
        return self.anonymous.stem


def _read_mapping_csv(path: Path) -> list[_ImageMapping]:
    """
    Read and validate unique relative image-path translations from CSV.

    Parameters
    ----------
    path : Path
        Filesystem path of the file or directory processed by this helper.

    Returns
    -------
    list[_ImageMapping]
        And validate unique relative image-path translations from CSV.

    Raises
    ------
    FileNotFoundError
        Mapping CSV does not exist: {path}
    ValueError
        {path} is missing required columns: {sorted(missing)} Unsafe mapping at line
        {line_number} of {path} Conflicting mapping for {original.as_posix()} at line
        {line_number} Multiple source images map to {anonymous.as_posix()} No image mappings
        found in {path}
    """
    if not path.is_file():
        raise FileNotFoundError(f"Mapping CSV does not exist: {path}")

    raw = path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(raw[:65536], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(raw.splitlines(), dialect=dialect)
    required = {"original_relative_path", "anonymous_relative_path"}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    mappings: list[_ImageMapping] = []
    seen_original: dict[str, str] = {}
    seen_anonymous: dict[str, str] = {}

    for line_number, row in enumerate(reader, start=2):
        original_text = str(row["original_relative_path"]).strip().replace("\\", "/")
        anonymous_text = str(row["anonymous_relative_path"]).strip().replace("\\", "/")

        while original_text.startswith("./"):
            original_text = original_text[2:]
        while anonymous_text.startswith("./"):
            anonymous_text = anonymous_text[2:]

        original = PurePosixPath(original_text)
        anonymous = PurePosixPath(anonymous_text)

        if (
            not original_text
            or not anonymous_text
            or original.is_absolute()
            or anonymous.is_absolute()
            or ".." in original.parts
            or ".." in anonymous.parts
        ):
            raise ValueError(f"Unsafe mapping at line {line_number} of {path}")

        original_key = original.as_posix().casefold()
        anonymous_key = anonymous.as_posix().casefold()

        previous = seen_original.get(original_key)
        if previous is not None and previous != anonymous.as_posix():
            raise ValueError(
                f"Conflicting mapping for {original.as_posix()} at line {line_number}"
            )

        previous = seen_anonymous.get(anonymous_key)
        if previous is not None and previous != original.as_posix():
            raise ValueError(
                f"Multiple source images map to {anonymous.as_posix()}"
            )

        seen_original[original_key] = anonymous.as_posix()
        seen_anonymous[anonymous_key] = original.as_posix()
        mappings.append(_ImageMapping(original=original, anonymous=anonymous))

    if not mappings:
        raise ValueError(f"No image mappings found in {path}")

    return mappings


def _infer_device(path: PurePosixPath) -> str:
    """
    Infer the acquisition-device group from known path fragments.

    Parameters
    ----------
    path : PurePosixPath
        Filesystem path of the file or directory processed by this helper.

    Returns
    -------
    str
        The acquisition-device group from known path fragments.
    """
    text = path.as_posix().casefold()
    if "mono_rgb" in text or "mono_real" in text:
        return "monochrome_real"
    if "iphone" in text:
        return "iphone"
    if "googlepixel" in text or "pixel" in text:
        return "googlepixel"
    return "microscope"


def _annotation_relative_paths(
    mapping: _ImageMapping,
    suffix: str,
) -> tuple[list[PurePosixPath], PurePosixPath]:
    """
    Build candidate original annotation paths and the anonymous target path.

    Parameters
    ----------
    mapping : _ImageMapping
        Translation record for one image.
    suffix : str
        File suffix appended to an image stem or used to select an image format.

    Returns
    -------
    tuple[list[PurePosixPath], PurePosixPath]
        Tuple containing ``sources``, ``target``, in that order.
    """
    old_folder_name = mapping.original_folder.name
    new_folder_name = mapping.anonymous_folder.name
    device = _infer_device(mapping.original)

    sources = [
        PurePosixPath(old_folder_name) / f"{mapping.original_stem}{suffix}",
        PurePosixPath(device)
        / f"{old_folder_name}_{mapping.original_stem}{suffix}",
    ]
    target = (
        PurePosixPath(new_folder_name)
        / f"{mapping.anonymous_stem}{suffix}"
    )
    return sources, target


def _build_rewrite_tables(
    mappings: Sequence[_ImageMapping],
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """
    Build exact and longest-first substring rules for cached image references.

    Parameters
    ----------
    mappings : Sequence[_ImageMapping]
        Original-to-anonymous file translations used by this operation.

    Returns
    -------
    tuple[list[tuple[str, str]], dict[str, str]]
        Tuple containing ``ordered_substring_rules``, ``exact_rules``, in that order.
    """
    substring_rules: dict[str, str] = {}
    exact_rules: dict[str, str] = {}

    original_name_counts = Counter(item.original.name for item in mappings)
    annotation_name_counts: Counter[str] = Counter()

    for item in mappings:
        old_rel = item.original.as_posix()
        new_rel = item.anonymous.as_posix()
        old_folder = item.original_folder.as_posix()
        new_folder = item.anonymous_folder.as_posix()

        substring_rules[old_rel] = new_rel
        substring_rules[f"ext_images/{old_rel}"] = f"ext_images/{new_rel}"
        substring_rules[f"ext_images/{old_folder}"] = f"ext_images/{new_folder}"

        exact_rules[old_rel] = new_rel
        exact_rules[old_folder] = new_folder
        exact_rules[item.original_folder.name] = item.anonymous_folder.name

        if original_name_counts[item.original.name] == 1:
            exact_rules[item.original.name] = item.anonymous.name

        for suffix in ANNOTATION_SUFFIXES:
            sources, target = _annotation_relative_paths(item, suffix)
            target_text = f"region_annotations/{target.as_posix()}"

            for source in sources:
                source_text = f"region_annotations/{source.as_posix()}"
                substring_rules[source_text] = target_text
                exact_rules[source.as_posix()] = target.as_posix()
                annotation_name_counts[source.name] += 1

    for item in mappings:
        for suffix in ANNOTATION_SUFFIXES:
            sources, target = _annotation_relative_paths(item, suffix)
            for source in sources:
                if annotation_name_counts[source.name] == 1:
                    exact_rules[source.name] = target.name

    ordered_substring_rules = sorted(
        substring_rules.items(),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    return ordered_substring_rules, exact_rules


def _rewrite_string(
    value: str,
    substring_rules: Sequence[tuple[str, str]],
    exact_rules: dict[str, str],
) -> tuple[str, int]:
    """
    Rewrite one string using the image translation rules and count replacements.

    Parameters
    ----------
    value : str
        Input value to validate, convert, or format as specified by this helper.
    substring_rules : Sequence[tuple[str, str]]
        Ordered old/new text pairs for substring replacement.
    exact_rules : dict[str, str]
        Mapping of complete original strings to replacement strings.

    Returns
    -------
    tuple[str, int]
        Tuple containing ``rewritten``, ``replacements``, in that order.
    """
    used_backslashes = "\\" in value and "/" not in value
    normalized = value.replace("\\", "/")
    rewritten = exact_rules.get(normalized, normalized)
    replacements = int(rewritten != normalized)

    for old, new in substring_rules:
        count = rewritten.count(old)
        if count:
            rewritten = rewritten.replace(old, new)
            replacements += count

    if used_backslashes:
        rewritten = rewritten.replace("/", "\\")

    return rewritten, replacements


def _rewrite_object(
    value: Any,
    substring_rules: Sequence[tuple[str, str]],
    exact_rules: dict[str, str],
) -> tuple[Any, int]:
    """
    Recursively rewrite references in containers while rejecting mapping-key collisions.

    Parameters
    ----------
    value : Any
        Input value to validate, convert, or format as specified by this helper.
    substring_rules : Sequence[tuple[str, str]]
        Ordered old/new text pairs for substring replacement.
    exact_rules : dict[str, str]
        Mapping of complete original strings to replacement strings.

    Returns
    -------
    tuple
        Rewritten value, preserving supported container types, and the number of
        replacements made.

    Raises
    ------
    ValueError
        Cache-key collision after renaming: {key} -> {new_key}
    """
    if isinstance(value, str):
        return _rewrite_string(value, substring_rules, exact_rules)

    if isinstance(value, Path):
        rewritten, count = _rewrite_string(
            str(value), substring_rules, exact_rules
        )
        return type(value)(rewritten), count

    if isinstance(value, dict):
        result: dict[Any, Any]
        result = OrderedDict() if isinstance(value, OrderedDict) else {}
        total = 0

        for key, item in value.items():
            new_key, key_count = _rewrite_object(
                key, substring_rules, exact_rules
            )
            new_item, item_count = _rewrite_object(
                item, substring_rules, exact_rules
            )

            if new_key in result:
                raise ValueError(
                    f"Cache-key collision after renaming: {key!r} -> {new_key!r}"
                )

            result[new_key] = new_item
            total += key_count + item_count

        return result, total

    if isinstance(value, list):
        result = []
        total = 0
        for item in value:
            new_item, count = _rewrite_object(
                item, substring_rules, exact_rules
            )
            result.append(new_item)
            total += count
        return result, total

    if isinstance(value, tuple):
        rewritten_items = []
        total = 0
        for item in value:
            new_item, count = _rewrite_object(
                item, substring_rules, exact_rules
            )
            rewritten_items.append(new_item)
            total += count

        if hasattr(value, "_fields"):
            return type(value)(*rewritten_items), total
        return tuple(rewritten_items), total

    if isinstance(value, set):
        rewritten_items = []
        total = 0
        for item in value:
            new_item, count = _rewrite_object(
                item, substring_rules, exact_rules
            )
            rewritten_items.append(new_item)
            total += count
        return set(rewritten_items), total

    if isinstance(value, frozenset):
        rewritten_items = []
        total = 0
        for item in value:
            new_item, count = _rewrite_object(
                item, substring_rules, exact_rules
            )
            rewritten_items.append(new_item)
            total += count
        return frozenset(rewritten_items), total

    return value, 0


def _atomic_write_json(path: Path, payload: Any) -> None:
    """
    Replace a JSON file through a temporary UTF-8 file.

    Parameters
    ----------
    path : Path
        Filesystem path of the file or directory processed by this helper.
    payload : Any
        JSON-serializable value to write atomically.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    """
    Replace a pickle through a temporary file while preserving its permission mode.

    Parameters
    ----------
    path : Path
        Filesystem path of the file or directory processed by this helper.
    payload : Any
        Python object to serialize with pickle.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copymode(path, temporary)
    os.replace(temporary, path)


def _backup_inputs(
    *,
    annotation_dir: Path,
    cache_paths: Sequence[Path],
    backup_dir: Path,
) -> None:
    """
    Copy annotations and existing cache files into a new backup directory.

    Parameters
    ----------
    annotation_dir : Path
        Directory containing image or cell annotation files.
    cache_paths : Sequence[Path]
        Paths of cached analysis files to back up or rewrite.
    backup_dir : Path
        New directory used to keep copies of the original inputs.

    Raises
    ------
    FileExistsError
        Backup directory already exists: {backup_dir}
    """
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup directory already exists: {backup_dir}"
        )

    backup_dir.mkdir(parents=True)
    if annotation_dir.exists():
        shutil.copytree(annotation_dir, backup_dir / "region_annotations")

    cache_backup_dir = backup_dir / "results"
    cache_backup_dir.mkdir()
    for path in cache_paths:
        if path.is_file():
            shutil.copy2(path, cache_backup_dir / path.name)


def rename_region_annotations_and_caches(
    *,
    mapping_csv: str | Path = "./private_mappings/ext_images_file_mapping.csv",
    annotation_dir: str | Path = "./region_annotations",
    results_dir: str | Path = "./results",
    cache_paths: Sequence[str | Path] | None = None,
    backup: bool = True,
    backup_dir: str | Path = "./private_mappings/region_cache_rename_backup",
) -> dict[str, Any]:
    """
    Rename reviewed region annotations and rewrite cached path references.

    Parameters
    ----------
    mapping_csv : str | Path
        Path to the CSV containing original-to-anonymous name mappings. Default is
        ``'./private_mappings/ext_images_file_mapping.csv'``.
    annotation_dir : str | Path
        Directory containing image or cell annotation files. Default is
        ``'./region_annotations'``.
    results_dir : str | Path
        Directory containing or receiving analysis and validation results. Default is
        ``'./results'``.
    cache_paths : Sequence[str | Path] | None
        Paths of cached analysis files to back up or rewrite. Default is ``None``.
    backup : bool
        Whether to preserve a backup before replacing a rewritten file. Default is ``True``.
    backup_dir : str | Path
        New directory used to keep copies of the original inputs. Default is
        ``'./private_mappings/region_cache_rename_backup'``.

    Returns
    -------
    dict[str, Any]
        Result mapping with keys ``mapping_rows``, ``annotated_images``,
        ``annotation_files_moved``, ``annotation_files_already_renamed``,
        ``annotation_json_replacements``, ``cache_replacements``, ``backup_dir``.

    Raises
    ------
    FileNotFoundError
        Annotation directory does not exist: {annotation_dir} Expected exactly one source
        annotation for {mapping.original} and suffix {suffix}; found {existing_sources}
        Missing required annotation for {mapping.original}: {suffix}
    FileExistsError
        Both old and new annotation files exist for {mapping.original}:
        {existing_sources[0]} and {target}

    Notes
    -----
    Run this from the final dataset directory. Image data and numerical cache
    arrays are not modified; only annotation filenames/directories and nested
    string or pathlib.Path references are changed.
    """
    mapping_csv = Path(mapping_csv)
    annotation_dir = Path(annotation_dir)
    results_dir = Path(results_dir)

    if cache_paths is None:
        required_cache_paths = [
            results_dir / "style_cache.cache",
            results_dir / "style_cache_normalized.cache",
        ]
        camera_candidates = [
            results_dir / "camera_quantile_band_cache.pkl",
            results_dir / "camera_quantile_band_cache.cache",
            results_dir / "camera_quantile_band_cache",
        ]

        missing_required = [
            path for path in required_cache_paths if not path.is_file()
        ]
        if missing_required:
            raise FileNotFoundError(
                "Missing required cache file(s): "
                + ", ".join(str(path) for path in missing_required)
            )

        existing_camera = [path for path in camera_candidates if path.is_file()]
        if len(existing_camera) != 1:
            raise FileNotFoundError(
                "Expected exactly one camera quantile cache among: "
                + ", ".join(str(path) for path in camera_candidates)
            )

        resolved_cache_paths = required_cache_paths + existing_camera
    else:
        resolved_cache_paths = [Path(path) for path in cache_paths]
        missing = [path for path in resolved_cache_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing cache file(s): "
                + ", ".join(str(path) for path in missing)
            )

    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {annotation_dir}"
        )

    mappings = _read_mapping_csv(mapping_csv)
    substring_rules, exact_rules = _build_rewrite_tables(mappings)

    if backup:
        _backup_inputs(
            annotation_dir=annotation_dir,
            cache_paths=resolved_cache_paths,
            backup_dir=Path(backup_dir),
        )

    moved_files = 0
    already_renamed = 0
    annotated_images = 0
    rewritten_json_values = 0

    for mapping in mappings:
        paths_by_suffix: dict[str, tuple[list[Path], Path]] = {}
        mapping_has_annotations = False

        for suffix in ANNOTATION_SUFFIXES:
            source_relatives, target_relative = _annotation_relative_paths(
                mapping, suffix
            )
            sources = [annotation_dir / relative for relative in source_relatives]
            target = annotation_dir / target_relative
            paths_by_suffix[suffix] = (sources, target)

            if target.exists() or any(source.exists() for source in sources):
                mapping_has_annotations = True

        if not mapping_has_annotations:
            continue

        annotated_images += 1

        for suffix in REQUIRED_ANNOTATION_SUFFIXES:
            sources, target = paths_by_suffix[suffix]
            existing_sources = [source for source in sources if source.is_file()]

            if target.is_file():
                if existing_sources:
                    raise FileExistsError(
                        f"Both old and new annotation files exist for {mapping.original}: "
                        f"{existing_sources[0]} and {target}"
                    )
                continue

            if len(existing_sources) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one source annotation for {mapping.original} "
                    f"and suffix {suffix}; found {existing_sources}"
                )

        for suffix in ANNOTATION_SUFFIXES:
            sources, target = paths_by_suffix[suffix]
            existing_sources = [source for source in sources if source.is_file()]

            if target.is_file():
                already_renamed += 1
                continue

            if not existing_sources:
                if suffix == "_preview.png":
                    continue
                raise FileNotFoundError(
                    f"Missing required annotation for {mapping.original}: {suffix}"
                )

            if len(existing_sources) > 1:
                raise RuntimeError(
                    f"Multiple source annotations found for {mapping.original}: "
                    + ", ".join(str(path) for path in existing_sources)
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            existing_sources[0].replace(target)
            moved_files += 1

        metadata_path = paths_by_suffix["_regions.json"][1]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rewritten_metadata, count = _rewrite_object(
            metadata, substring_rules, exact_rules
        )
        rewritten_json_values += count
        _atomic_write_json(metadata_path, rewritten_metadata)

    for directory in sorted(
        (path for path in annotation_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    cache_replacements: dict[str, int] = {}
    for cache_path in resolved_cache_paths:
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)

        rewritten_cache, count = _rewrite_object(
            cache, substring_rules, exact_rules
        )
        _atomic_write_pickle(cache_path, rewritten_cache)
        cache_replacements[str(cache_path)] = count

    return {
        "mapping_rows": len(mappings),
        "annotated_images": annotated_images,
        "annotation_files_moved": moved_files,
        "annotation_files_already_renamed": already_renamed,
        "annotation_json_replacements": rewritten_json_values,
        "cache_replacements": cache_replacements,
        "backup_dir": str(backup_dir) if backup else None,
    }

