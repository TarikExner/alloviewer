from __future__ import annotations

import argparse
import csv
import hashlib
import os
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tifffile
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".heic",
    ".heif",
}

TIFF_SENSITIVE_TAGS = {
    "artist",
    "copyright",
    "datetime",
    "documentname",
    "hostcomputer",
    "imagedescription",
    "make",
    "model",
    "pagename",
    "software",
    "uniquecameramodel",
    "xpauthor",
    "xpcomment",
    "xpkeywords",
    "xpsubject",
    "xptitle",
}

JPEG_ALLOWED_APP_MARKERS = {0xE0, 0xEE}  # JFIF and Adobe color-transform markers.
JPEG_STRIPPED_MARKERS = set(range(0xE1, 0xEE)) | {0xEF, 0xFE}
JPEG_STANDALONE_MARKERS = {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}

FOLDER_MAPPING_FIELDS = (
    "original_relative_path",
    "anonymous_relative_path",
    "original_name",
    "anonymous_name",
    "folder_kind",
    "created_utc",
)

FILE_MAPPING_FIELDS = (
    "original_relative_path",
    "anonymous_relative_path",
    "original_filename",
    "anonymous_filename",
    "source_format",
    "output_format",
    "conversion",
    "source_sha256",
    "output_sha256",
    "created_utc",
)


class ImageAnonymizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageResult:
    source: Path
    output: Path
    source_format: str
    output_format: str
    conversion: str
    source_sha256: str
    output_sha256: str


@dataclass
class AnonymizationReport:
    input_root: Path
    output_root: Path
    mapping_root: Path
    folder_mapping_csv: Path
    file_mapping_csv: Path
    images: list[ImageResult] = field(default_factory=list)
    skipped_non_images: list[Path] = field(default_factory=list)
    copied_non_images: list[Path] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def folder_count(self) -> int:
        with self.folder_mapping_csv.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))


@dataclass(frozen=True)
class _TiffPageSpec:
    data: np.ndarray
    photometric: object | None
    planarconfig: object | None
    extrasamples: tuple[object, ...] | None
    colormap: np.ndarray | None
    bits_per_sample: tuple[int, ...]


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
        self.callback(f"[image-anonymizer +{elapsed:7.1f}s] {message}")


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_roots(input_root: Path, output_root: Path, mapping_root: Path) -> None:
    if not input_root.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_root}")

    if input_root == output_root:
        raise ValueError("Input and output folders must be different.")

    if _is_relative_to(output_root, input_root):
        raise ValueError("The output folder must not be inside the input folder.")

    if _is_relative_to(mapping_root, input_root):
        raise ValueError("The private mapping folder must not be inside the input folder.")

    if _is_relative_to(mapping_root, output_root):
        raise ValueError(
            "The private mapping folder must not be inside the shareable output folder."
        )


def _load_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = (row.get(key) or "").strip()
        if not value:
            raise ImageAnonymizationError(
                f"Mapping file '{path}' contains a row without '{key}'."
            )
        if value in result:
            raise ImageAnonymizationError(
                f"Mapping file '{path}' contains duplicate key '{value}'."
            )
        result[value] = row
    return result


def _write_csv_atomic(path: Path, fields: Iterable[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_identifier(prefix: str, used: set[str]) -> str:
    while True:
        candidate = f"{prefix}_{secrets.token_hex(6).upper()}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _assert_no_symlinks(input_root: Path) -> None:
    for path in input_root.rglob("*"):
        if path.is_symlink():
            raise ImageAnonymizationError(
                f"Symbolic links are not accepted in anonymization input: {path}"
            )


def _normalise_bits_per_sample(value: object, samples: int) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    if value is None:
        return tuple()
    return tuple([int(value)] * max(1, samples))


def _read_tiff_pages(path: Path) -> tuple[list[_TiffPageSpec], bool]:
    specs: list[_TiffPageSpec] = []
    with tifffile.TiffFile(path) as tif:
        is_bigtiff = bool(tif.is_bigtiff)
        for page_index, page in enumerate(tif.pages):
            orientation_tag = page.tags.get("Orientation")
            orientation = int(orientation_tag.value) if orientation_tag is not None else 1
            if orientation != 1:
                raise ImageAnonymizationError(
                    f"TIFF '{path}' page {page_index} uses Orientation={orientation}. "
                    "Refusing to remove the orientation tag without an explicit pixel transform."
                )

            try:
                data = page.asarray()
            except ValueError as exc:
                if "imagecodecs" in str(exc).casefold():
                    raise ImageAnonymizationError(
                        f"TIFF '{path}' page {page_index} uses compression that requires "
                        "the optional 'imagecodecs' package. Install it with: "
                        "pip install imagecodecs"
                    ) from exc
                raise
            samples = int(getattr(page, "samplesperpixel", 1) or 1)
            bits = _normalise_bits_per_sample(page.bitspersample, samples)
            expected_bits = int(data.dtype.itemsize * 8)
            if not bits or any(bit != expected_bits for bit in bits):
                raise ImageAnonymizationError(
                    f"TIFF '{path}' page {page_index} stores {bits or 'unknown'} bits per "
                    f"sample but decodes to {data.dtype}. Packed or unusual TIFF bit depths "
                    "are intentionally rejected to prevent silent conversion."
                )

            extrasamples = None
            if getattr(page, "extrasamples", None):
                extrasamples = tuple(page.extrasamples)

            colormap = None
            if getattr(page, "colormap", None) is not None:
                colormap = np.array(page.colormap, copy=True)

            specs.append(
                _TiffPageSpec(
                    data=np.array(data, copy=True),
                    photometric=getattr(page, "photometric", None),
                    planarconfig=getattr(page, "planarconfig", None),
                    extrasamples=extrasamples,
                    colormap=colormap,
                    bits_per_sample=bits,
                )
            )

    if not specs:
        raise ImageAnonymizationError(f"TIFF contains no image pages: {path}")
    return specs, is_bigtiff


def _write_clean_tiff(source: Path, output: Path) -> None:
    source_pages, is_bigtiff = _read_tiff_pages(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffWriter(output, bigtiff=is_bigtiff) as writer:
        for page in source_pages:
            writer.write(
                page.data,
                photometric=page.photometric,
                planarconfig=page.planarconfig,
                extrasamples=page.extrasamples,
                colormap=page.colormap,
                bitspersample=page.bits_per_sample[0],
                compression=None,
                predictor=False,
                metadata=None,
                description=None,
                datetime=False,
                software=False,
            )

    output_pages, output_is_bigtiff = _read_tiff_pages(output)
    if output_is_bigtiff != is_bigtiff:
        raise ImageAnonymizationError(f"BigTIFF status changed while rewriting '{source}'.")
    if len(output_pages) != len(source_pages):
        raise ImageAnonymizationError(f"TIFF page count changed while rewriting '{source}'.")

    for page_index, (before, after) in enumerate(zip(source_pages, output_pages, strict=True)):
        if before.data.shape != after.data.shape:
            raise ImageAnonymizationError(
                f"TIFF page shape changed for '{source}', page {page_index}."
            )
        if before.data.dtype != after.data.dtype:
            raise ImageAnonymizationError(
                f"TIFF dtype changed for '{source}', page {page_index}: "
                f"{before.data.dtype} -> {after.data.dtype}."
            )
        if before.bits_per_sample != after.bits_per_sample:
            raise ImageAnonymizationError(
                f"TIFF bit depth changed for '{source}', page {page_index}: "
                f"{before.bits_per_sample} -> {after.bits_per_sample}."
            )
        if not np.array_equal(before.data, after.data):
            raise ImageAnonymizationError(
                f"TIFF pixel values changed for '{source}', page {page_index}."
            )

    with tifffile.TiffFile(output) as tif:
        for page_index, page in enumerate(tif.pages):
            present = {str(tag.name).casefold() for tag in page.tags.values()}
            retained = sorted(TIFF_SENSITIVE_TAGS & present)
            if retained:
                raise ImageAnonymizationError(
                    f"Sensitive TIFF tags remain in '{output}', page {page_index}: {retained}"
                )


def _jpeg_exif_orientation(path: Path) -> int:
    with Image.open(path) as image:
        try:
            return int(image.getexif().get(274, 1) or 1)
        except Exception:
            return 1


def _strip_jpeg_segments(data: bytes) -> tuple[bytes, tuple[int, ...]]:
    if len(data) < 4 or data[:2] != b"\xFF\xD8":
        raise ImageAnonymizationError("Input is not a valid JPEG stream.")

    output = bytearray(data[:2])
    stripped: list[int] = []
    index = 2
    saw_eoi = False

    while index < len(data):
        if data[index] != 0xFF:
            raise ImageAnonymizationError(
                f"Malformed JPEG marker stream at byte offset {index}."
            )

        marker_start = index
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            raise ImageAnonymizationError("JPEG ends inside a marker prefix.")

        marker = data[index]
        index += 1
        marker_prefix = data[marker_start:index]

        if marker == 0xD9:
            output.extend(marker_prefix)
            saw_eoi = True
            break

        if marker in JPEG_STANDALONE_MARKERS:
            output.extend(marker_prefix)
            continue

        if index + 2 > len(data):
            raise ImageAnonymizationError("JPEG ends before a segment length field.")
        segment_length = int.from_bytes(data[index:index + 2], "big")
        if segment_length < 2:
            raise ImageAnonymizationError("JPEG contains an invalid segment length.")
        segment_end = index + segment_length
        if segment_end > len(data):
            raise ImageAnonymizationError("JPEG segment extends beyond the end of the file.")

        segment = data[marker_start:segment_end]
        index = segment_end

        if marker == 0xDA:  # Start of Scan
            output.extend(segment)
            scan_start = index
            while index < len(data):
                if data[index] != 0xFF:
                    index += 1
                    continue

                next_index = index + 1
                while next_index < len(data) and data[next_index] == 0xFF:
                    next_index += 1
                if next_index >= len(data):
                    raise ImageAnonymizationError("JPEG ends inside entropy-coded data.")

                next_marker = data[next_index]
                if next_marker == 0x00 or 0xD0 <= next_marker <= 0xD7:
                    index = next_index + 1
                    continue

                output.extend(data[scan_start:index])
                break
            continue

        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            if marker in JPEG_ALLOWED_APP_MARKERS:
                output.extend(segment)
            else:
                stripped.append(marker)
            continue

        output.extend(segment)

    if not saw_eoi:
        raise ImageAnonymizationError("JPEG does not contain an End-of-Image marker.")

    return bytes(output), tuple(stripped)


def _jpeg_metadata_markers(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    if data[:2] != b"\xFF\xD8":
        raise ImageAnonymizationError(f"Not a JPEG file: {path}")

    markers: list[int] = []
    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            raise ImageAnonymizationError(f"Malformed JPEG marker stream in '{path}'.")
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker == 0xD9:
            break
        if marker in JPEG_STANDALONE_MARKERS:
            continue
        if index + 2 > len(data):
            break
        length = int.from_bytes(data[index:index + 2], "big")
        end = index + length
        if marker == 0xDA:
            index = end
            while index < len(data):
                if data[index] != 0xFF:
                    index += 1
                    continue
                probe = index + 1
                while probe < len(data) and data[probe] == 0xFF:
                    probe += 1
                if probe >= len(data):
                    return tuple(markers)
                code = data[probe]
                if code == 0x00 or 0xD0 <= code <= 0xD7:
                    index = probe + 1
                    continue
                break
            continue
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            markers.append(marker)
        index = end
    return tuple(markers)


def _decoded_array(path: Path, *, apply_orientation: bool) -> tuple[np.ndarray, str]:
    with Image.open(path) as image:
        image.load()
        converted = ImageOps.exif_transpose(image) if apply_orientation else image
        converted.load()
        return np.array(converted), converted.mode


def _write_lossless_clean_jpeg(source: Path, output: Path) -> None:
    before_array, before_mode = _decoded_array(source, apply_orientation=False)
    cleaned, _ = _strip_jpeg_segments(source.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cleaned)

    remaining_markers = set(_jpeg_metadata_markers(output))
    forbidden = remaining_markers - JPEG_ALLOWED_APP_MARKERS
    if forbidden:
        marker_names = [f"0x{marker:02X}" for marker in sorted(forbidden)]
        raise ImageAnonymizationError(
            f"JPEG metadata markers remain in '{output}': {marker_names}"
        )

    after_array, after_mode = _decoded_array(output, apply_orientation=False)
    if before_mode != after_mode:
        raise ImageAnonymizationError(
            f"JPEG image mode changed for '{source}': {before_mode} -> {after_mode}."
        )
    if before_array.shape != after_array.shape or before_array.dtype != after_array.dtype:
        raise ImageAnonymizationError(f"JPEG decoded structure changed for '{source}'.")
    if not np.array_equal(before_array, after_array):
        raise ImageAnonymizationError(
            f"JPEG decoded pixels changed while stripping metadata from '{source}'."
        )


def _register_heif_support() -> None:
    try:
        import pillow_heif
    except ImportError as exc:
        raise ImageAnonymizationError(
            "HEIC/HEIF input requires 'pillow-heif'. Install it with: pip install pillow-heif"
        ) from exc
    pillow_heif.register_heif_opener()


def _load_oriented_pillow_image(source: Path) -> tuple[Image.Image, np.ndarray, str]:
    if source.suffix.casefold() in {".heic", ".heif"}:
        _register_heif_support()

    with Image.open(source) as image:
        image.load()
        oriented = ImageOps.exif_transpose(image)
        oriented.load()
        clean = oriented.copy()
        return clean, np.array(clean), clean.mode


def _write_clean_png(source: Path, output: Path) -> None:
    image, expected_array, expected_mode = _load_oriented_pillow_image(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Palette transparency affects the decoded appearance and therefore has to
    # be preserved. All other source metadata, including ICC profiles, EXIF,
    # timestamps, comments, and device information, must be discarded.
    transparency = (
        image.info.get("transparency")
        if image.mode == "P"
        else None
    )

    clean_image = image.copy()
    clean_image.info.clear()

    save_kwargs: dict[str, object] = {
        "format": "PNG",
        "compress_level": 9,
        "optimize": False,
    }

    if transparency is not None:
        save_kwargs["transparency"] = transparency

    clean_image.save(output, **save_kwargs)

    with Image.open(output) as rewritten:
        rewritten.load()

        actual_array = np.array(rewritten)
        actual_mode = rewritten.mode

        # Some Pillow versions may expose empty metadata fields such as
        # icc_profile=None. Only non-empty metadata counts as retained.
        metadata_keys = {
            key
            for key, value in rewritten.info.items()
            if key != "transparency"
            and value not in (None, b"", "")
        }

    if metadata_keys:
        raise ImageAnonymizationError(
            f"PNG metadata remains in '{output}': {sorted(metadata_keys)}"
        )

    if actual_mode != expected_mode:
        raise ImageAnonymizationError(
            f"PNG image mode changed for '{source}': "
            f"{expected_mode} -> {actual_mode}."
        )

    if (
        expected_array.shape != actual_array.shape
        or expected_array.dtype != actual_array.dtype
    ):
        raise ImageAnonymizationError(
            f"PNG decoded structure changed for '{source}'."
        )

    if not np.array_equal(expected_array, actual_array):
        raise ImageAnonymizationError(
            f"PNG pixel values changed for '{source}'."
        )


def _source_format(path: Path) -> str:
    suffix = path.suffix.casefold()
    return {
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".webp": "WEBP",
        ".heic": "HEIC",
        ".heif": "HEIF",
    }.get(suffix, suffix.lstrip(".").upper())


def _output_policy(source: Path) -> tuple[str, str, str]:
    suffix = source.suffix.casefold()
    if suffix in {".tif", ".tiff"}:
        return ".tif", "TIFF", "metadata-free TIFF rewrite"
    if suffix in {".jpg", ".jpeg"}:
        orientation = _jpeg_exif_orientation(source)
        if orientation == 1:
            return ".jpg", "JPEG", "lossless JPEG metadata stripping"
        return ".png", "PNG", f"JPEG orientation {orientation} applied; converted to PNG"
    if suffix == ".png":
        return ".png", "PNG", "metadata-free PNG rewrite"
    if suffix == ".bmp":
        return ".png", "PNG", "BMP decoded and stored losslessly as PNG"
    if suffix == ".webp":
        return ".png", "PNG", "WebP decoded and stored losslessly as PNG"
    if suffix in {".heic", ".heif"}:
        return ".png", "PNG", "HEIC/HEIF decoded, oriented, and stored as PNG"
    raise ImageAnonymizationError(f"Unsupported image format: {source}")


def _process_image(source: Path, output: Path) -> tuple[str, str, str]:
    source_format = _source_format(source)
    _, output_format, conversion = _output_policy(source)

    if source_format == "TIFF":
        _write_clean_tiff(source, output)
    elif source_format == "JPEG" and output_format == "JPEG":
        _write_lossless_clean_jpeg(source, output)
    else:
        _write_clean_png(source, output)

    return source_format, output_format, conversion


def anonymize_image_tree(
    input_folder: str | os.PathLike[str] = "./ext_images",
    output_folder: str | os.PathLike[str] = "../final/ext_images",
    *,
    mapping_folder: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    copy_non_images: bool = False,
    reuse_existing_mappings: bool = True,
    reject_symlinks: bool = True,
    verbose: bool = True,
    progress_every: int = 1,
    progress_callback: Callable[[str], None] | None = None,
) -> AnonymizationReport:
    """Anonymize an image directory tree while preserving image measurements.

    Every directory and image filename is replaced with a random identifier. The
    private CSV mappings are stored outside the shareable output directory.

    TIFF files are decoded and rewritten as metadata-free TIFF while requiring
    exact page, shape, dtype, bit-depth, and pixel equality. JPEG files without a
    nontrivial EXIF orientation are stripped losslessly by removing identifying
    APP/COM marker segments without recompressing the image. Oriented JPEG, HEIC,
    HEIF, PNG, BMP, and WebP inputs are written as metadata-free PNG and verified
    against the decoded, correctly oriented source pixels.
    """

    if progress_every < 1:
        raise ValueError("progress_every must be at least 1.")

    progress = _ProgressReporter(enabled=verbose, callback=progress_callback)

    input_root = Path(input_folder).expanduser().resolve()
    output_root = Path(output_folder).expanduser().resolve()
    mapping_root = (
        Path(mapping_folder).expanduser().resolve()
        if mapping_folder is not None
        else (output_root.parent / "private_mappings").resolve()
    )
    folder_mapping_csv = mapping_root / "ext_images_folder_mapping.csv"
    file_mapping_csv = mapping_root / "ext_images_file_mapping.csv"

    progress.log(f"Input root: {input_root}")
    progress.log(f"Output root: {output_root}")
    progress.log(f"Private mapping root: {mapping_root}")
    progress.log("Validating input/output locations.")
    _validate_roots(input_root, output_root, mapping_root)
    if reject_symlinks:
        progress.log("Scanning the input tree for symbolic links.")
        _assert_no_symlinks(input_root)
        progress.log("Symbolic-link scan completed.")

    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output folder already exists: {output_root}. Pass overwrite=True to replace it."
        )

    progress.log(
        "Loading existing private mappings."
        if reuse_existing_mappings
        else "Existing private mappings will not be reused."
    )
    existing_folder_rows = (
        _load_csv_by_key(folder_mapping_csv, "original_relative_path")
        if reuse_existing_mappings
        else {}
    )
    existing_file_rows = (
        _load_csv_by_key(file_mapping_csv, "original_relative_path")
        if reuse_existing_mappings
        else {}
    )

    used_folder_names = {
        Path(row["anonymous_relative_path"]).name for row in existing_folder_rows.values()
    }
    used_file_names = {
        Path(row["anonymous_relative_path"]).name for row in existing_file_rows.values()
    }

    folder_map: dict[str, Path] = {".": Path(".")}
    folder_rows: dict[str, dict[str, str]] = {}

    progress.log("Enumerating directories.")
    directories = sorted(
        (path for path in input_root.rglob("*") if path.is_dir()),
        key=lambda path: (len(path.relative_to(input_root).parts), path.as_posix().casefold()),
    )
    progress.log(f"Found {len(directories)} directories below the input root.")

    for directory in directories:
        relative = directory.relative_to(input_root)
        relative_key = relative.as_posix()
        parent_key = relative.parent.as_posix() if relative.parent != Path(".") else "."
        anonymous_parent = folder_map[parent_key]

        existing = existing_folder_rows.get(relative_key)
        if existing:
            anonymous_relative = Path(existing["anonymous_relative_path"])
            if anonymous_relative.parent != anonymous_parent:
                raise ImageAnonymizationError(
                    f"Existing folder mapping has an inconsistent parent for '{relative_key}'."
                )
            anonymous_name = anonymous_relative.name
            created_utc = existing.get("created_utc") or _utc_now()
        else:
            prefix = "RUN" if len(relative.parts) == 1 else "DIR"
            anonymous_name = _new_identifier(prefix, used_folder_names)
            anonymous_relative = anonymous_parent / anonymous_name
            created_utc = _utc_now()

        folder_map[relative_key] = anonymous_relative
        folder_rows[relative_key] = {
            "original_relative_path": relative_key,
            "anonymous_relative_path": anonymous_relative.as_posix(),
            "original_name": directory.name,
            "anonymous_name": anonymous_name,
            "folder_kind": "run" if len(relative.parts) == 1 else "nested",
            "created_utc": created_utc,
        }

    staging_parent = output_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.anonymizing-", dir=staging_parent)
    )
    progress.log(f"Created temporary staging directory: {staging_root}")

    report = AnonymizationReport(
        input_root=input_root,
        output_root=output_root,
        mapping_root=mapping_root,
        folder_mapping_csv=folder_mapping_csv,
        file_mapping_csv=file_mapping_csv,
    )
    file_rows: dict[str, dict[str, str]] = {}

    try:
        for anonymous_directory in folder_map.values():
            (staging_root / anonymous_directory).mkdir(parents=True, exist_ok=True)

        progress.log("Enumerating files.")
        files = sorted(
            (path for path in input_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(input_root).as_posix().casefold(),
        )
        image_total = sum(
            1 for path in files if path.suffix.casefold() in SUPPORTED_EXTENSIONS
        )
        non_image_total = len(files) - image_total
        progress.log(
            f"Found {len(files)} files: {image_total} supported images and "
            f"{non_image_total} other files."
        )

        processed_images = 0
        for source in files:
            relative = source.relative_to(input_root)
            relative_key = relative.as_posix()
            parent_key = relative.parent.as_posix() if relative.parent != Path(".") else "."
            anonymous_parent = folder_map[parent_key]

            if source.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                if copy_non_images:
                    anonymous_name = _new_identifier("FILE", used_file_names) + source.suffix.casefold()
                    destination = staging_root / anonymous_parent / anonymous_name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    report.copied_non_images.append(source)
                    progress.log(f"COPIED non-image file: {relative_key}")
                else:
                    report.skipped_non_images.append(source)
                    progress.log(f"SKIPPED non-image file: {relative_key}")
                continue

            processed_images += 1
            file_started = time.perf_counter()
            should_report_file = (
                processed_images == 1
                or processed_images == image_total
                or processed_images % progress_every == 0
            )
            if should_report_file:
                progress.log(
                    f"[{processed_images}/{image_total}] START {relative_key} "
                    f"({_format_bytes(source.stat().st_size)})."
                )
                progress.log(
                    f"[{processed_images}/{image_total}] Computing source SHA-256 and selecting output policy."
                )

            source_hash = _sha256(source)
            output_suffix, expected_output_format, expected_conversion = _output_policy(source)
            existing = existing_file_rows.get(relative_key)

            if existing:
                prior_hash = (existing.get("source_sha256") or "").strip()
                if prior_hash and prior_hash != source_hash:
                    raise ImageAnonymizationError(
                        f"Source file changed since its mapping was created: {source}"
                    )
                anonymous_relative = Path(existing["anonymous_relative_path"])
                if anonymous_relative.parent != anonymous_parent:
                    raise ImageAnonymizationError(
                        f"Existing file mapping has an inconsistent parent for '{relative_key}'."
                    )
                if anonymous_relative.suffix.casefold() != output_suffix:
                    raise ImageAnonymizationError(
                        f"Existing file mapping output extension no longer matches policy for '{source}'."
                    )
                anonymous_name = anonymous_relative.name
                created_utc = existing.get("created_utc") or _utc_now()
            else:
                anonymous_name = _new_identifier("IMG", used_file_names) + output_suffix
                anonymous_relative = anonymous_parent / anonymous_name
                created_utc = _utc_now()

            destination = staging_root / anonymous_relative
            if should_report_file:
                progress.log(
                    f"[{processed_images}/{image_total}] Writing {anonymous_relative.as_posix()} "
                    f"using: {expected_conversion}."
                )
            source_format, output_format, conversion = _process_image(source, destination)
            if output_format != expected_output_format or conversion != expected_conversion:
                raise ImageAnonymizationError(
                    f"Internal output-policy mismatch while processing '{source}'."
                )
            output_hash = _sha256(destination)

            file_rows[relative_key] = {
                "original_relative_path": relative_key,
                "anonymous_relative_path": anonymous_relative.as_posix(),
                "original_filename": source.name,
                "anonymous_filename": anonymous_name,
                "source_format": source_format,
                "output_format": output_format,
                "conversion": conversion,
                "source_sha256": source_hash,
                "output_sha256": output_hash,
                "created_utc": created_utc,
            }
            report.images.append(
                ImageResult(
                    source=source,
                    output=output_root / anonymous_relative,
                    source_format=source_format,
                    output_format=output_format,
                    conversion=conversion,
                    source_sha256=source_hash,
                    output_sha256=output_hash,
                )
            )
            if should_report_file:
                file_elapsed = time.perf_counter() - file_started
                progress.log(
                    f"[{processed_images}/{image_total}] DONE in {file_elapsed:.1f}s: "
                    f"{source_format} -> {output_format}, "
                    f"output {_format_bytes(destination.stat().st_size)}."
                )

        progress.log("All files passed conversion and verification.")
        if output_root.exists():
            progress.log(f"Removing existing output tree because overwrite=True: {output_root}")
            shutil.rmtree(output_root)
        progress.log("Publishing the completed staging tree to the final output location.")
        os.replace(staging_root, output_root)
        progress.log("Output tree published successfully.")

        sorted_folder_rows = [folder_rows[key] for key in sorted(folder_rows, key=str.casefold)]
        sorted_file_rows = [file_rows[key] for key in sorted(file_rows, key=str.casefold)]
        progress.log("Writing private folder and file mapping CSVs.")
        _write_csv_atomic(folder_mapping_csv, FOLDER_MAPPING_FIELDS, sorted_folder_rows)
        _write_csv_atomic(file_mapping_csv, FILE_MAPPING_FIELDS, sorted_file_rows)
        progress.log(
            f"FINISHED: {report.image_count} images anonymized, "
            f"{len(report.copied_non_images)} non-image files copied, "
            f"{len(report.skipped_non_images)} non-image files skipped."
        )
        progress.log(f"Folder mapping CSV: {folder_mapping_csv}")
        progress.log(f"File mapping CSV: {file_mapping_csv}")

        return report
    except Exception as exc:
        progress.log(
            f"FAILED: {type(exc).__name__}: {exc}. Removing temporary staging output."
        )
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anonymize an external image dataset and create private CSV mappings."
    )
    parser.add_argument("input_folder", nargs="?", default="./ext_images")
    parser.add_argument("output_folder", nargs="?", default="../final/ext_images")
    parser.add_argument("--mapping-folder", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-non-images", action="store_true")
    parser.add_argument("--no-reuse-mappings", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser

def anonymize_ordered_image_folders(
    folders: Iterable[str | os.PathLike[str]],
    input_root: str | os.PathLike[str] = "./ext_images",
    output_folder: str | os.PathLike[str] = "../final/show_images",
    *,
    mapping_folder: str | os.PathLike[str] | None = None,
    overwrite: bool = True,
    reuse_existing_mappings: bool = True,
    reject_symlinks: bool = True,
    verbose: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    code_path_prefix: str = "./ext_images",
    preserve_metadata_folder_markers: tuple[str, ...] = (
        "iphone",
        "googlepixel",
    ),
    ignored_directory_names: tuple[str, ...] = (
        ".ipynb_checkpoints",
    ),
) -> list[str]:
    """
    Anonymize an ordered list of image folders.

    Folder names are replaced with RUN_* identifiers while image filenames
    remain unchanged.

    Processing policy
    -----------------
    TIFF images:
        Rewritten using the existing TIFF anonymization implementation.
        Metadata is removed and decoded pixels are verified.

    Non-TIFF images in iPhone or GooglePixel folders:
        Copied byte-for-byte. Embedded metadata, including EXIF orientation,
        is retained by design.

    Non-TIFF images elsewhere:
        Rejected to avoid unintentionally retaining metadata.

    Returns replacement folder paths in exactly the same order as ``folders``.
    """

    progress = _ProgressReporter(
        enabled=verbose,
        callback=progress_callback,
    )

    source_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_folder).expanduser().resolve()

    mapping_root = (
        Path(mapping_folder).expanduser().resolve()
        if mapping_folder is not None
        else (output_root.parent / "private_mappings").resolve()
    )

    _validate_roots(
        source_root,
        output_root,
        mapping_root,
    )

    folder_mapping_csv = mapping_root / "show_images_folder_mapping.csv"
    file_mapping_csv = mapping_root / "show_images_file_mapping.csv"
    ordered_mapping_csv = mapping_root / "show_images_ordered_folder_mapping.csv"

    supplied_folders = list(folders)

    if not supplied_folders:
        raise ValueError(
            "At least one source folder must be supplied."
        )

    markers = tuple(
        marker.casefold()
        for marker in preserve_metadata_folder_markers
    )
    ignored_names = {
        name.casefold()
        for name in ignored_directory_names
    }

    progress.log(
        f"Preparing {len(supplied_folders)} ordered image folders."
    )
    progress.log(f"Input root: {source_root}")
    progress.log(f"Output root: {output_root}")
    progress.log(f"Mapping root: {mapping_root}")

    selected_folders: list[
        tuple[str, Path, Path, bool]
    ] = []

    seen_relative_paths: set[str] = set()

    for position, folder_value in enumerate(
        supplied_folders,
        start=1,
    ):
        source_folder = Path(
            folder_value
        ).expanduser().resolve()

        if not source_folder.is_dir():
            raise FileNotFoundError(
                f"Source folder does not exist: "
                f"{source_folder}"
            )

        if not _is_relative_to(
            source_folder,
            source_root,
        ):
            raise ValueError(
                f"Source folder is outside input_root: "
                f"{source_folder}"
            )

        relative_folder = source_folder.relative_to(
            source_root
        )

        if len(relative_folder.parts) != 1:
            raise ValueError(
                "Expected top-level folders directly below "
                f"'{source_root}', but received "
                f"'{relative_folder.as_posix()}'."
            )

        relative_key = relative_folder.as_posix()

        if relative_key in seen_relative_paths:
            raise ValueError(
                f"Source folder supplied more than once: "
                f"{relative_key}"
            )

        seen_relative_paths.add(relative_key)

        if reject_symlinks:
            if source_folder.is_symlink():
                raise ImageAnonymizationError(
                    f"Source folder is a symbolic link: "
                    f"{source_folder}"
                )

            _assert_no_symlinks(source_folder)

        nested_directories = sorted(
            path
            for path in source_folder.iterdir()
            if (
                path.is_dir()
                and path.name.casefold()
                not in ignored_names
            )
        )

        if nested_directories:
            raise ImageAnonymizationError(
                "Unexpected nested directories found in "
                f"'{source_folder}': "
                + ", ".join(
                    path.name
                    for path in nested_directories[:10]
                )
            )

        folder_name_folded = source_folder.name.casefold()

        preserve_non_tiff_metadata = any(
            marker in folder_name_folded
            for marker in markers
        )

        selected_folders.append(
            (
                str(folder_value),
                source_folder,
                relative_folder,
                preserve_non_tiff_metadata,
            )
        )

        policy = (
            "TIFF sanitization; non-TIFF byte-for-byte copy"
            if preserve_non_tiff_metadata
            else "TIFF sanitization only"
        )

        progress.log(
            f"[{position}/{len(supplied_folders)}] "
            f"Selected {relative_key}: {policy}."
        )

    existing_folder_rows = (
        _load_csv_by_key(
            folder_mapping_csv,
            "original_relative_path",
        )
        if reuse_existing_mappings
        else {}
    )

    existing_file_rows = (
        _load_csv_by_key(
            file_mapping_csv,
            "original_relative_path",
        )
        if reuse_existing_mappings
        else {}
    )

    used_folder_names = {
        Path(
            row["anonymous_relative_path"]
        ).parts[0]
        for row in existing_folder_rows.values()
        if (
            row.get("anonymous_relative_path")
            or ""
        ).strip()
    }

    if output_root.exists():
        used_folder_names.update(
            path.name
            for path in output_root.iterdir()
            if path.is_dir()
        )

    folder_assignments: list[
        tuple[
            str,
            Path,
            Path,
            str,
            str,
            bool,
        ]
    ] = []

    selected_folder_rows: dict[
        str,
        dict[str, str],
    ] = {}

    replacement_paths: list[str] = []
    ordered_rows: list[dict[str, str]] = []

    normalized_code_prefix = (
        code_path_prefix.rstrip("/\\")
    )

    for position, (
        original_argument,
        source_folder,
        relative_folder,
        preserve_non_tiff_metadata,
    ) in enumerate(
        selected_folders,
        start=1,
    ):
        relative_key = relative_folder.as_posix()
        existing = existing_folder_rows.get(
            relative_key
        )

        if existing is not None:
            anonymous_relative = Path(
                existing["anonymous_relative_path"]
            )

            if len(anonymous_relative.parts) != 1:
                raise ImageAnonymizationError(
                    "Existing mapping is not a top-level "
                    f"folder mapping for '{relative_key}': "
                    f"{anonymous_relative.as_posix()}"
                )

            anonymous_folder_name = (
                anonymous_relative.name
            )
            created_utc = (
                existing.get("created_utc")
                or _utc_now()
            )
        else:
            anonymous_folder_name = _new_identifier(
                "RUN",
                used_folder_names,
            )
            anonymous_relative = Path(
                anonymous_folder_name
            )
            created_utc = _utc_now()

        replacement_code_path = (
            f"{normalized_code_prefix}/"
            f"{anonymous_folder_name}"
        )

        folder_assignments.append(
            (
                relative_key,
                source_folder,
                anonymous_relative,
                anonymous_folder_name,
                created_utc,
                preserve_non_tiff_metadata,
            )
        )

        replacement_paths.append(
            replacement_code_path
        )

        selected_folder_rows[relative_key] = {
            "original_relative_path": relative_key,
            "anonymous_relative_path": (
                anonymous_relative.as_posix()
            ),
            "original_name": source_folder.name,
            "anonymous_name": anonymous_folder_name,
            "folder_kind": "run",
            "created_utc": created_utc,
        }

        ordered_rows.append(
            {
                "position": str(position),
                "original_argument": original_argument,
                "original_relative_path": relative_key,
                "original_folder_name": (
                    source_folder.name
                ),
                "anonymous_folder_name": (
                    anonymous_folder_name
                ),
                "anonymous_relative_path": (
                    anonymous_relative.as_posix()
                ),
                "replacement_code_path": (
                    replacement_code_path
                ),
                "created_utc": created_utc,
            }
        )

    staging_parent = output_root.parent
    staging_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{output_root.name}"
                ".ordered-anonymizing-"
            ),
            dir=staging_parent,
        )
    )

    backup_root = (
        staging_root / "__backups__"
    )
    backup_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress.log(
        f"Created staging directory: {staging_root}"
    )

    selected_file_rows: dict[
        str,
        dict[str, str],
    ] = {}

    published_targets: list[Path] = []
    backed_up_targets: dict[Path, Path] = {}

    mapping_files = (
        folder_mapping_csv,
        file_mapping_csv,
        ordered_mapping_csv,
    )

    original_mapping_contents: dict[
        Path,
        bytes | None,
    ] = {
        path: (
            path.read_bytes()
            if path.exists()
            else None
        )
        for path in mapping_files
    }

    def restore_mapping_files() -> None:
        for (
            path,
            original_content,
        ) in original_mapping_contents.items():
            if original_content is None:
                path.unlink(missing_ok=True)
                continue

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = path.with_name(
                f".{path.name}."
                f"{secrets.token_hex(8)}.restore"
            )
            temporary.write_bytes(
                original_content
            )
            os.replace(
                temporary,
                path,
            )

    def rollback_published_folders() -> None:
        for target in reversed(
            published_targets
        ):
            if target.exists():
                shutil.rmtree(target)

        for (
            target,
            backup,
        ) in backed_up_targets.items():
            if backup.exists():
                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                os.replace(
                    backup,
                    target,
                )

    def source_format_from_suffix(
        suffix: str,
    ) -> str:
        return {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".bmp": "BMP",
            ".webp": "WEBP",
            ".heic": "HEIC",
            ".heif": "HEIF",
            ".tif": "TIFF",
            ".tiff": "TIFF",
        }.get(
            suffix.casefold(),
            suffix.lstrip(".").upper(),
        )

    try:
        total_images = 0

        for (
            _,
            source_folder,
            _,
            _,
            _,
            _,
        ) in folder_assignments:
            total_images += sum(
                1
                for path in source_folder.iterdir()
                if (
                    path.is_file()
                    and path.suffix.casefold()
                    in SUPPORTED_EXTENSIONS
                )
            )

        progress.log(
            f"Found {total_images} supported images."
        )

        processed_images = 0

        for folder_position, (
            relative_folder_key,
            source_folder,
            anonymous_relative,
            anonymous_folder_name,
            folder_created_utc,
            preserve_non_tiff_metadata,
        ) in enumerate(
            folder_assignments,
            start=1,
        ):
            staging_folder = (
                staging_root
                / anonymous_relative
            )
            staging_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            source_files = sorted(
                (
                    path
                    for path in source_folder.iterdir()
                    if path.is_file()
                ),
                key=lambda path: (
                    path.name.casefold()
                ),
            )

            image_files = [
                path
                for path in source_files
                if path.suffix.casefold()
                in SUPPORTED_EXTENSIONS
            ]

            skipped_files = [
                path
                for path in source_files
                if path.suffix.casefold()
                not in SUPPORTED_EXTENSIONS
            ]

            progress.log(
                f"[folder {folder_position}/"
                f"{len(folder_assignments)}] "
                f"{source_folder.name} -> "
                f"{anonymous_folder_name}: "
                f"{len(image_files)} images."
            )

            for skipped in skipped_files:
                progress.log(
                    "SKIPPED non-image file: "
                    f"{relative_folder_key}/"
                    f"{skipped.name}"
                )

            if not image_files:
                raise ImageAnonymizationError(
                    "No supported images found in "
                    f"'{source_folder}'."
                )

            for source_image in image_files:
                processed_images += 1
                started = time.perf_counter()

                suffix = (
                    source_image.suffix.casefold()
                )
                is_tiff = suffix in {
                    ".tif",
                    ".tiff",
                }

                original_relative_file = (
                    Path(relative_folder_key)
                    / source_image.name
                )
                anonymous_relative_file = (
                    anonymous_relative
                    / source_image.name
                )
                destination = (
                    staging_root
                    / anonymous_relative_file
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                progress.log(
                    f"[{processed_images}/"
                    f"{total_images}] "
                    f"{original_relative_file.as_posix()} "
                    f"-> "
                    f"{anonymous_relative_file.as_posix()}"
                )

                source_hash = _sha256(
                    source_image
                )

                if is_tiff:
                    progress.log(
                        f"[{processed_images}/"
                        f"{total_images}] "
                        "Rewriting TIFF without metadata."
                    )

                    (
                        source_format,
                        output_format,
                        conversion,
                    ) = _process_image(
                        source_image,
                        destination,
                    )

                    if output_format != "TIFF":
                        raise ImageAnonymizationError(
                            "TIFF anonymization produced "
                            "an unexpected output format for "
                            f"'{source_image}': "
                            f"{output_format}"
                        )

                elif preserve_non_tiff_metadata:
                    progress.log(
                        f"[{processed_images}/"
                        f"{total_images}] "
                        "Copying byte-for-byte; embedded "
                        "metadata retained by request."
                    )

                    shutil.copyfile(
                        source_image,
                        destination,
                    )

                    source_format = (
                        source_format_from_suffix(
                            suffix
                        )
                    )
                    output_format = source_format
                    conversion = (
                        "byte-for-byte copy; "
                        "embedded metadata retained"
                    )

                else:
                    raise ImageAnonymizationError(
                        "A non-TIFF image was found outside "
                        "an iPhone or GooglePixel folder: "
                        f"'{source_image}'."
                    )

                output_hash = _sha256(
                    destination
                )

                if (
                    not is_tiff
                    and output_hash != source_hash
                ):
                    raise ImageAnonymizationError(
                        "Byte-for-byte copy verification "
                        f"failed for '{source_image}'."
                    )

                existing_file = (
                    existing_file_rows.get(
                        original_relative_file.as_posix()
                    )
                )

                file_created_utc = (
                    existing_file.get("created_utc")
                    if existing_file
                    else None
                ) or folder_created_utc

                selected_file_rows[
                    original_relative_file.as_posix()
                ] = {
                    "original_relative_path": (
                        original_relative_file.as_posix()
                    ),
                    "anonymous_relative_path": (
                        anonymous_relative_file.as_posix()
                    ),
                    "original_filename": (
                        source_image.name
                    ),
                    "anonymous_filename": (
                        source_image.name
                    ),
                    "source_format": source_format,
                    "output_format": output_format,
                    "conversion": conversion,
                    "source_sha256": source_hash,
                    "output_sha256": output_hash,
                    "created_utc": file_created_utc,
                }

                elapsed = (
                    time.perf_counter()
                    - started
                )

                progress.log(
                    f"[{processed_images}/"
                    f"{total_images}] "
                    f"DONE in {elapsed:.1f}s; "
                    f"{_format_bytes(destination.stat().st_size)}."
                )

        progress.log(
            "All selected images passed processing "
            "and verification."
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for (
            _,
            _,
            anonymous_relative,
            anonymous_folder_name,
            _,
            _,
        ) in folder_assignments:
            staged_folder = (
                staging_root
                / anonymous_relative
            )
            final_folder = (
                output_root
                / anonymous_relative
            )

            if final_folder.exists():
                if not overwrite:
                    raise FileExistsError(
                        "Output folder already exists: "
                        f"{final_folder}"
                    )

                backup_folder = (
                    backup_root
                    / anonymous_folder_name
                )

                os.replace(
                    final_folder,
                    backup_folder,
                )

                backed_up_targets[
                    final_folder
                ] = backup_folder

            os.replace(
                staged_folder,
                final_folder,
            )
            published_targets.append(
                final_folder
            )

        merged_folder_rows = dict(
            existing_folder_rows
        )
        merged_folder_rows.update(
            selected_folder_rows
        )

        merged_file_rows = dict(
            existing_file_rows
        )
        merged_file_rows.update(
            selected_file_rows
        )

        _write_csv_atomic(
            folder_mapping_csv,
            FOLDER_MAPPING_FIELDS,
            (
                merged_folder_rows[key]
                for key in sorted(
                    merged_folder_rows,
                    key=str.casefold,
                )
            ),
        )

        _write_csv_atomic(
            file_mapping_csv,
            FILE_MAPPING_FIELDS,
            (
                merged_file_rows[key]
                for key in sorted(
                    merged_file_rows,
                    key=str.casefold,
                )
            ),
        )

        _write_csv_atomic(
            ordered_mapping_csv,
            (
                "position",
                "original_argument",
                "original_relative_path",
                "original_folder_name",
                "anonymous_folder_name",
                "anonymous_relative_path",
                "replacement_code_path",
                "created_utc",
            ),
            ordered_rows,
        )

        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )

        progress.log(
            f"Finished: {len(folder_assignments)} "
            f"folders and {processed_images} images."
        )
        progress.log(
            f"Ordered mapping CSV: "
            f"{ordered_mapping_csv}"
        )

        if verbose:
            print("", flush=True)
            print(
                "EXT_IMAGES_FOLDERS = [",
                flush=True,
            )

            for replacement_path in replacement_paths:
                print(
                    f'    "{replacement_path}",',
                    flush=True,
                )

            print("]", flush=True)

        return replacement_paths

    except Exception:
        rollback_published_folders()
        restore_mapping_files()
        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )
        raise

def main() -> None:
    args = _build_parser().parse_args()
    report = anonymize_image_tree(
        args.input_folder,
        args.output_folder,
        mapping_folder=args.mapping_folder,
        overwrite=args.overwrite,
        copy_non_images=args.copy_non_images,
        reuse_existing_mappings=not args.no_reuse_mappings,
        verbose=not args.quiet,
        progress_every=args.progress_every,
    )
    print(f"Anonymized images: {report.image_count}")
    print(f"Output: {report.output_root}")
    print(f"Folder mapping: {report.folder_mapping_csv}")
    print(f"File mapping: {report.file_mapping_csv}")
    if report.skipped_non_images:
        print(f"Skipped non-image files: {len(report.skipped_non_images)}")


if __name__ == "__main__":
    main()

