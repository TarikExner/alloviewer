from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


SCORES = frozenset({1, 2, 4, 6, 8})
IDENTITY = ("Folder", "well", "image_name", "role")
CASE_COLUMNS = (
    "center", *IDENTITY, "annotator_scores", "n_annotators",
    "n_distinct_scores", "majority_score", "disagreement_type",
)
ISSUE_COLUMNS = ("center", *IDENTITY, "annotator_scores", "reason")
CELL_COLUMNS = (
    "center", *IDENTITY, "roi_id", "predicted_label", "classifier_score",
    "area", "mean_r", "mean_g", "mean_b", "reviewed_label", "review_notes",
)
IMAGE_COLUMNS = (
    *CASE_COLUMNS, "source_path", "annotated_path", "n_positive",
    "n_negative", "n_uncertain", "n_cells", "height", "width", "dtype",
    "export_dtype", "display_white_level",
)
DEFAULT_COLORS = {
    "pos": (255, 165, 0),       # orange, RGB
    "neg": (0, 220, 0),         # green, RGB
    "uncertain": (0, 130, 255), # blue, RGB
}


@dataclass
class AnnotationConfig:
    """Settings for rendering and production-cell classification."""

    outline_width: int = 2
    colors_rgb: Mapping[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_COLORS)
    )
    classifier_margin: float = 1.0  # production classifier default
    qc: bool = False              # production run_image_analysis default
    unet_config: dict[str, Any] | None = None
    verify_tiff: bool = True
    overwrite: bool = False
    # review_8bit makes the microscopy signal visible in standard TIFF viewers.
    # native preserves the original uint8/uint16 data outside the outlines.
    render_mode: str = "review_8bit"
    display_percentile: float = 99.7
    display_gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.outline_width < 1:
            raise ValueError("outline_width must be >= 1")
        if self.render_mode not in {"review_8bit", "native"}:
            raise ValueError("render_mode must be 'review_8bit' or 'native'")
        if not np.isfinite(self.display_percentile) or not 0 < self.display_percentile <= 100:
            raise ValueError("display_percentile must be within (0, 100]")
        if not np.isfinite(self.display_gamma) or self.display_gamma <= 0:
            raise ValueError("display_gamma must be a positive finite number")
        if not np.isfinite(self.classifier_margin) or self.classifier_margin < 0:
            raise ValueError("classifier_margin must be a finite nonnegative number")
        if set(self.colors_rgb) != {"pos", "neg", "uncertain"}:
            raise ValueError("colors_rgb must define exactly pos, neg, uncertain")
        for label, rgb in self.colors_rgb.items():
            if len(rgb) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in rgb):
                raise ValueError(f"{label}: expected RGB integers in [0,255]")


def _component(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or any(x in text for x in ("/", "\\", ":", "\x00")):
        raise ValueError(f"Unsafe {name}: {text!r}")
    return text


def _valid_score(raw: str) -> int:
    value = float(raw)
    if not np.isfinite(value) or int(value) != value or int(value) not in SCORES:
        raise ValueError(f"Invalid CDC score: {raw!r}")
    return int(value)


def _read_center(csv_path: str | Path, center: str, expected_n: int):
    """Return disagreement cases, rejected records, and run control paths."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    raw: dict[tuple[str, ...], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    path_roles: dict[tuple[str, str], tuple[str, str]] = {}
    run_controls: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"positive": set(), "negative": set()}
    )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = set(IDENTITY) | {"Annotator", "score"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        for lineno, row in enumerate(reader, 2):
            folder = _component(row["Folder"], "Folder")
            well = _component(row["well"], "well")
            name = _component(row["image_name"], "image_name")
            role = _component(row["role"], "role").lower()
            if Path(name).suffix.lower() not in {".tif", ".tiff"}:
                raise ValueError(f"{csv_path}:{lineno}: expected .tif/.tiff: {name!r}")
            key = (folder, well, name, role)
            ident = (folder, name)
            if ident in path_roles and path_roles[ident] != (well, role):
                raise ValueError(f"{center}: contradictory well/role for {ident}")
            path_roles[ident] = (well, role)
            annotator = str(row["Annotator"]).strip()
            raw[key][annotator].add(str(row["score"]).strip())
            if role in {"positive", "negative"}:
                run_controls[folder][role].add(name)

    cases: list[dict] = []
    issues: list[dict] = []
    expected = {str(i) for i in range(1, expected_n + 1)}
    for key, by_annotator in sorted(raw.items()):
        folder, well, name, role = key
        recorded = ";".join(
            f"{annot}:{'|'.join(sorted(values))}"
            for annot, values in sorted(by_annotator.items())
        )
        base = {"center": center, **dict(zip(IDENTITY, key)), "annotator_scores": recorded}
        if set(by_annotator) != expected or any(len(v) != 1 for v in by_annotator.values()):
            issues.append({**base, "reason": "missing_or_conflicting_annotations"})
            continue
        try:
            votes = [_valid_score(next(iter(by_annotator[a]))) for a in sorted(expected)]
        except (ValueError, OverflowError):
            issues.append({**base, "reason": "invalid_score"})
            continue
        if len(set(votes)) == 1:
            continue
        counts = Counter(votes)
        winners = [score for score, n in counts.items() if n == max(counts.values())]
        majority = winners[0] if len(winners) == 1 else None
        cases.append({
            **base,
            "n_annotators": expected_n,
            "n_distinct_scores": len(counts),
            "majority_score": majority,
            "disagreement_type": (
                "three_way" if len(counts) == 3 else "two_vs_one"
                if expected_n == 3 else "two_annotator"
            ),
        })

    # The classifier calibration uses the full set of controls, not just
    # controls with discordant expert scores.
    return cases, issues, dict(run_controls)


def find_disagreements(
    berlin_csv: str | Path,
    frankfurt_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, dict[str, set[str]]]]]:
    """Find all valid discordant images; return (cases, issues, controls).

    The returned ``controls`` mapping has the form
    ``controls[center][Folder]['positive'|'negative'] -> set[image_name]``.
    A three-annotator 1:1:1 split is retained without a majority label.
    """
    all_cases, all_issues, controls = [], [], {}
    for center, csv_file, n in (
        ("Berlin", berlin_csv, 3), ("Frankfurt", frankfurt_csv, 2)
    ):
        cases, issues, run_controls = _read_center(csv_file, center, n)
        all_cases.extend(cases)
        all_issues.extend(issues)
        controls[center] = run_controls
    case_df = pd.DataFrame(all_cases, columns=CASE_COLUMNS)
    issue_df = pd.DataFrame(all_issues, columns=ISSUE_COLUMNS)
    return case_df, issue_df, controls


def prepare_review_image(
    image_rgb: np.ndarray,
    *,
    percentile: float = 99.7,
    gamma: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Make an RGB uint8 view of a dark uint8/uint16 microscopy image.

    Uses a SINGLE scale factor for all RGB channels, rather than separate
    channel normalization, to avoid changing the relative red/green signals.
    A global brightness level is estimated only from nonblack pixels so that
    large dark fields do not force the upper percentile to zero.

    This is for human display only. It does not alter raw source data,
    segmentation, extracted ROI values, or classifier predictions.
    ``gamma < 1`` makes dim cells easier to see.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image_rgb.shape}")
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise TypeError("Expected uint8/uint16 image")
    if not np.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must lie in (0, 100]")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be positive and finite")

    brightest = image_rgb.max(axis=2)
    nonzero = brightest[brightest > 0]
    white = float(np.percentile(nonzero, percentile)) if nonzero.size else 1.0
    white = max(white, 1.0)
    normalized = np.clip(image_rgb.astype(np.float32) / white, 0.0, 1.0)
    if gamma != 1.0:
        np.power(normalized, gamma, out=normalized)
    # np.rint before cast avoids integer conversion bias.
    display = np.rint(normalized * 255.0).astype(np.uint8)
    return display, white


def draw_cell_outlines(
    image_rgb: np.ndarray,
    instance_labels: np.ndarray,
    rois: Sequence[Any],
    *,
    outline_width: int = 2,
    colors_rgb: Mapping[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw per-instance, class-colored borders over the *original* RGB pixels.

    Returns an RGB array with the same shape and dtype (uint8/uint16) as input.
    Non-border pixels are unchanged; the array passed in is not mutated.
    ``roi_id`` must match the actual instance label, not a list position.
    """
    if outline_width < 1:
        raise ValueError("outline_width must be >= 1")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {image_rgb.shape}")
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise TypeError("Input image must be uint8 or uint16 for lossless TIFF output")
    labels = np.asarray(instance_labels)
    if labels.ndim != 2 or labels.shape != image_rgb.shape[:2]:
        raise ValueError("Instance map must be HxW and have the same size as the image")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise TypeError("Instance labels must be nonnegative integers")
    palette = colors_rgb if colors_rgb is not None else DEFAULT_COLORS
    max_val = int(np.iinfo(image_rgb.dtype).max)
    output = image_rgb.copy()
    objects = ndi.find_objects(labels.astype(np.int32, copy=False))
    height, width = labels.shape
    pad = max(2, outline_width + 1)

    for roi in rois:
        rid = int(roi.get("roi_id") if isinstance(roi, dict) else roi.roi_id)
        cls = roi.get("label") if isinstance(roi, dict) else roi.label
        if cls not in palette:
            raise ValueError(f"Unknown ROI label {cls!r} for ROI {rid}")
        if not 1 <= rid <= len(objects) or objects[rid - 1] is None:
            raise ValueError(f"ROI ID {rid} has no pixels in instance label map")
        ys, xs = objects[rid - 1]
        y0, y1 = max(0, ys.start - pad), min(height, ys.stop + pad)
        x0, x1 = max(0, xs.start - pad), min(width, xs.stop + pad)
        roi_mask = (labels[y0:y1, x0:x1] == rid).astype(np.uint8)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(round(channel * max_val / 255.0)) for channel in palette[cls])
        # cv2 draws literal RGB component values into our RGB array. Conversion
        # to OpenCV BGR happens only at the TIFF-writing boundary.
        cv2.drawContours(output[y0:y1, x0:x1], contours, -1, color,
                         thickness=outline_width, lineType=cv2.LINE_8)
    return output


def save_annotated_tiff(path: str | Path, image_rgb: np.ndarray, *, verify: bool = True):
    """Write uint8/uint16 RGB TIFF without rescaling or lossy compression."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".partial.tif")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    params = ([cv2.IMWRITE_TIFF_COMPRESSION, 1]
              if hasattr(cv2, "IMWRITE_TIFF_COMPRESSION") else [])
    try:
        if not cv2.imwrite(str(tmp), bgr, params):
            raise IOError(f"Cannot write TIFF: {tmp}")
        if verify:
            read_back = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED)
            if read_back is None or read_back.dtype != bgr.dtype or not np.array_equal(read_back, bgr):
                raise IOError(f"TIFF pixel/depth verification failed: {tmp}")
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return destination


def _scaled_and_raw(path: Path):
    """Apply the exact scale=True image preprocessing used by the app pipeline."""
    from ...image_analysis.io.io import load_image, scale_image

    raw, report = load_image(path, scale=False, as_chw=False)
    if raw.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"Expected uint8/uint16 RGB TIFF: {path}: {raw.dtype}")
    scaled, _ = scale_image(raw, report, fast=True)
    return raw, scaled


def _segment_extract(path: Path, segmenter: Any, qc: bool):
    """Reuse pipeline segmentation + optional QC + RGB ROI extraction."""
    from ...image_analysis.pipeline import _extract_roi_from_image
    from ...image_analysis.extractor import RGBExtractor
    from ...image_analysis.qc import QCMonitor

    raw, scaled = _scaled_and_raw(path)
    result, seg = _extract_roi_from_image(
        image=scaled,
        segmenter=segmenter,
        qc_monitor=QCMonitor() if qc else None,
        extractor=RGBExtractor(),
        well_id=path.stem,
        qc=qc,
    )
    labels = seg["instance_labels_for_rois"]
    if labels.shape != raw.shape[:2]:
        raise ValueError(f"Segmentation shape mismatch for {path}")
    return raw, labels, [roi.__dict__ for roi in result.rois]


def _path(roots: Mapping[str, str | Path], center: str, folder: str, name: str) -> Path:
    root = Path(roots[center]).expanduser().resolve()
    candidate = (root / _component(folder, "Folder") / _component(name, "image_name")).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes source root: {candidate}")
    return candidate


def _calibrate_run(
    center: str, folder: str, control_map: Mapping[str, Any],
    roots: Mapping[str, str | Path], segmenter: Any, qc: bool,
) -> dict[str, Any]:
    """Reproduce the per-run PC/NC Gaussian 2D calibration in the main pipeline."""
    from ...image_analysis.calibrators import PCNCGaussian2DCalibrator

    pc_wells, nc_wells = [], []
    for role, destination in (("positive", pc_wells), ("negative", nc_wells)):
        names = sorted(control_map.get(role, []))
        if not names:
            raise ValueError(f"No {role} controls for {center}/{folder}")
        for name in names:
            path = _path(roots, center, folder, name)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {role} control: {path}")
            _, _, rois = _segment_extract(path, segmenter, qc)
            if not rois:
                raise RuntimeError(f"No extracted cells in {role} control: {path}")
            destination.append(rois)
    return PCNCGaussian2DCalibrator().fit(pc_wells=pc_wells, nc_wells=nc_wells)


def _make_segmenter(unet_config: dict[str, Any] | None):
    """Load the same checkpoint/config as the production image-analysis pipeline."""
    import copy
    from ...image_analysis.config import INSTANCE_CONFIG, UNET_CONFIG
    from ...image_analysis.segmenter import SegmenterUNetInference

    config = copy.deepcopy(unet_config if unet_config is not None else UNET_CONFIG)
    config.setdefault("instance_cfg", INSTANCE_CONFIG.to_dict())
    return SegmenterUNetInference.from_config(config)


def _write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def annotate_disagreements(
    cases: pd.DataFrame,
    controls: Mapping[str, Mapping[str, Mapping[str, set[str]]]],
    roots: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    config: AnnotationConfig | None = None,
    limit_per_center: int | None = None,
    segmenter: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify disagreeing images, save outlined TIFFs and review CSVs.

    Parameters
    ----------
    cases
        Output of ``find_disagreements``. Includes all three-way splits.
    controls
        Original complete PC and NC well list from ``find_disagreements``.
    roots
        ``{'Berlin': Path(...), 'Frankfurt': Path(...)}``.
    output_dir
        Export root. Creates ``Berlin/`` and ``Frankfurt/`` folders.
    config
        Outline color/width, classifier margin, QC and model settings.
    limit_per_center
        Use 1-3 for first notebook test; None processes every disagreement.
    segmenter
        Optional preloaded segmenter (also useful for unit tests).

    Returns
    -------
    images, cells : pandas.DataFrame
        ``images`` describes every exported TIFF; ``cells`` is a row per
        classified ROI with blank ``reviewed_label``/``review_notes`` columns.
    """
    from ...image_analysis.classifiers import ROIClassifierGaussian2D3Way

    cfg = config or AnnotationConfig()
    if limit_per_center is not None and limit_per_center < 1:
        raise ValueError("limit_per_center must be >= 1 or None")
    out = Path(output_dir).expanduser().resolve()
    missing_cols = set(CASE_COLUMNS) - set(cases.columns)
    if missing_cols:
        raise ValueError(f"cases is missing columns: {sorted(missing_cols)}")
    selection = cases.sort_values(["center", "Folder", "well", "image_name"], kind="stable")
    if limit_per_center is not None:
        selection = selection.groupby("center", sort=False).head(limit_per_center)
    if selection.empty:
        raise ValueError("No discordant images selected")

    # Check all referenced inputs and required calibration controls BEFORE
    # initializing the model or creating partial outputs.
    destinations = []
    for row in selection.itertuples(index=False):
        path = _path(roots, row.center, row.Folder, row.image_name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected TIFF: {path}")
        ctrl = controls[row.center][row.Folder]
        for role in ("positive", "negative"):
            names = ctrl.get(role, set())
            if not names:
                raise ValueError(f"Missing {role} control metadata for {row.center}/{row.Folder}")
            for name in names:
                control_path = _path(roots, row.center, row.Folder, name)
                if not control_path.is_file():
                    raise FileNotFoundError(f"Missing calibration TIFF: {control_path}")
        name = f"{row.Folder}__{row.well}__{Path(row.image_name).stem}__annotated.tif"
        dst = out / row.center / name
        if dst.exists() and not cfg.overwrite:
            raise FileExistsError(f"Output exists (set overwrite=True or use a fresh folder): {dst}")
        destinations.append((row, path, dst))

    engine = segmenter if segmenter is not None else _make_segmenter(cfg.unet_config)
    calibrations: dict[tuple[str, str], dict[str, Any]] = {}
    image_rows, cell_rows = [], []
    for count, (row, source, destination) in enumerate(destinations, 1):
        run_key = (row.center, row.Folder)
        if run_key not in calibrations:
            print(f"Calibrating {row.center}/{row.Folder} with all PC/NC wells...", flush=True)
            calibrations[run_key] = _calibrate_run(
                row.center, row.Folder, controls[row.center][row.Folder],
                roots, engine, cfg.qc,
            )
        classifier = ROIClassifierGaussian2D3Way(
            calibrations[run_key], margin=cfg.classifier_margin
        )
        rgb_raw, labels, rois = _segment_extract(source, engine, cfg.qc)
        if not rois:
            raise RuntimeError(f"No cells extracted from selected image: {source}")
        classified = classifier(rois)
        if cfg.render_mode == "review_8bit":
            image_base, display_white = prepare_review_image(
                rgb_raw,
                percentile=cfg.display_percentile,
                gamma=cfg.display_gamma,
            )
        else:
            image_base, display_white = rgb_raw, None
        rendered = draw_cell_outlines(
            image_base, labels, classified,
            outline_width=cfg.outline_width,
            colors_rgb=cfg.colors_rgb,
        )
        save_annotated_tiff(destination, rendered, verify=cfg.verify_tiff)
        label_counts = Counter(r["label"] for r in classified)
        base = {k: getattr(row, k) for k in CASE_COLUMNS}
        image_rows.append({
            **base, "source_path": str(source), "annotated_path": str(destination),
            "n_positive": label_counts["pos"], "n_negative": label_counts["neg"],
            "n_uncertain": label_counts["uncertain"], "n_cells": len(classified),
            "height": rgb_raw.shape[0], "width": rgb_raw.shape[1],
            "dtype": str(rgb_raw.dtype), "export_dtype": str(rendered.dtype),
            "display_white_level": display_white,
        })
        for roi in classified:
            cell_rows.append({
                **{k: base[k] for k in ("center", *IDENTITY)},
                "roi_id": roi["roi_id"], "predicted_label": roi["label"],
                "classifier_score": roi.get("score"), "area": roi["area"],
                "mean_r": roi["mean_r"], "mean_g": roi["mean_g"],
                "mean_b": roi["mean_b"],
                "reviewed_label": "", "review_notes": "",
            })
        print(f"[{count}/{len(destinations)}] {row.center}/{row.Folder}/{row.image_name}: "
              f"{len(classified)} cells -> {destination.name}", flush=True)

    images = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    cells = pd.DataFrame(cell_rows, columns=CELL_COLUMNS)
    _write_table(out / "disagreement_image_manifest.csv", images)
    _write_table(out / "cell_review_template.csv", cells)
    return images, cells


def export_review_overlays_from_existing(
    image_manifest: str | Path,
    output_dir: str | Path,
    *,
    display_percentile: float = 99.7,
    display_gamma: float = 1.0,
    outline_colors_rgb: Mapping[str, tuple[int, int, int]] | None = None,
    verify_tiff: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Turn existing native outlined 16-bit TIFFs into viewable RGB TIFFs.

    This avoids re-running segmentation, control calibration, or classification.
    Reconstructs the outlines by finding the exact RGB drawing colors in the
    previous native TIFF. The source TIFF is used for the display base; it is
    never overwritten. Both image paths are obtained from the original manifest.

    Works with outputs made by ``draw_cell_outlines`` using LINE_8.
    If colors were customized in the original run, pass the same original
    palette through ``outline_colors_rgb``.
    """
    manifest = Path(image_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = pd.read_csv(manifest, dtype={"Folder": str, "well": str})
    required = {"center", "source_path", "annotated_path"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(rows.columns))}")
    palette = outline_colors_rgb if outline_colors_rgb is not None else DEFAULT_COLORS
    if set(palette) != {"pos", "neg", "uncertain"}:
        raise ValueError("outline_colors_rgb must contain pos, neg, uncertain")

    out = Path(output_dir).expanduser().resolve()
    targets = []
    for row in rows.itertuples(index=False):
        center = _component(row.center, "center")
        src, old = Path(row.source_path), Path(row.annotated_path)
        if not src.is_file() or not old.is_file():
            raise FileNotFoundError(f"Missing raw/annotated file: {src} / {old}")
        dst = out / center / old.name
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Already exists: {dst}")
        targets.append((src, old, dst, center))

    results = []
    from ...image_analysis.io.io import load_image
    for i, (src, old, dst, center) in enumerate(targets, 1):
        raw, _ = load_image(src, scale=False, as_chw=False)
        old_bgr = cv2.imread(str(old), cv2.IMREAD_UNCHANGED)
        if old_bgr is None or old_bgr.ndim != 3 or old_bgr.shape[2] != 3:
            raise ValueError(f"Unable to decode native outlined RGB TIFF: {old}")
        annotated_rgb = cv2.cvtColor(old_bgr, cv2.COLOR_BGR2RGB)
        if raw.shape != annotated_rgb.shape or raw.dtype != annotated_rgb.dtype:
            raise ValueError(f"Old overlay does not match source TIFF: {old}")
        if raw.dtype not in (np.uint8, np.uint16):
            raise TypeError(f"Unsupported source type {raw.dtype} for {src}")

        display, white = prepare_review_image(
            raw, percentile=display_percentile, gamma=display_gamma
        )
        max_val = int(np.iinfo(raw.dtype).max)
        changed = np.any(annotated_rgb != raw, axis=2)
        counts = {}
        for label, color in palette.items():
            old_color = tuple(int(round(channel * max_val / 255.0)) for channel in color)
            mask = np.all(annotated_rgb == old_color, axis=2) & changed
            display[mask] = color
            counts[label] = int(mask.sum())
        if not any(counts.values()):
            raise RuntimeError(
                f"No colored outline pixels detected in {old}. "
                "Is it an original native TIFF with the specified color palette?"
            )
        save_annotated_tiff(dst, display, verify=verify_tiff)
        results.append({
            "center": center, "source_path": str(src),
            "old_annotated_path": str(old), "review_path": str(dst),
            "white_level": white, "positive_outline_pixels": counts["pos"],
            "negative_outline_pixels": counts["neg"],
            "uncertain_outline_pixels": counts["uncertain"],
        })
        print(f"[{i}/{len(targets)}] {center}: {dst.name}", flush=True)
    table = pd.DataFrame(results)
    _write_table(out / "review_overlay_manifest.csv", table)
    return table

